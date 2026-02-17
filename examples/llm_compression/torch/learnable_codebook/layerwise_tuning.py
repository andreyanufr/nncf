import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.progress import track
from typing import Optional

from codebook_wrapper import CodebookWrapperLinear, wrap_model, unwrap_model, wrap_model_block




class BlockInputCacher(nn.Module):
    def __init__(self, block: nn.Module, name: str):
        super().__init__()
        self.block = block
        self.name = name
        self.cached_inputs = {}
        
        self.cached_inputs["hidden_states"] = []

    @property
    def attention_type(self):
        if hasattr(self.block, "attention_type"):
            return self.block.attention_type
        return None

    def forward(self, hidden_states, **kwargs):
        self.cached_inputs["hidden_states"].append(hidden_states.detach())
        for key in kwargs:
            if key not in self.cached_inputs:
                self.cached_inputs[key] = kwargs[key]
        return self.block(hidden_states, **kwargs)
    
    def dump_cached_inputs(self, dir: str):
        names = []
        for key, tensors in self.cached_inputs.items():
            stacked = torch.cat(tensors, dim=0)
            names.append(f"{dir}/{self.name}_{key}.pt")
            torch.save(stacked, f"{dir}/{self.name}_{key}.pt")
        return names


@torch.no_grad()
def dump_block_inputs(model: nn.Module, dir: str, dataset: list[Tensor]):
    model.eval()
    
    if not os.path.exists(dir):
        os.makedirs(dir)

    names = {}
    
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = BlockInputCacher(layer, name=f"layer_{i}")

    with torch.no_grad():
        for batch in track(dataset, description="Caching block inputs..."):
            model(batch)
    
    # After running through the dataset, dump the cached inputs for each block
    for name, module in model.named_modules():
        if isinstance(module, BlockInputCacher):
            names[name] = module.dump_cached_inputs(dir)

    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = layer.block  # Unwrap the original block to restore model functionality

    return names


@torch.no_grad()
def get_first_block_inputs(model: nn.Module, dataset: list[Tensor]):
    model.eval()
    
    model.model.layers[0] = BlockInputCacher(model.model.layers[0], name="layer_0")

    with torch.no_grad():
        for batch in track(dataset, description="Caching block inputs..."):
            model(batch)
    
    # After running through the dataset, dump the cached inputs for each block
    res = model.model.layers[0].cached_inputs

    model.model.layers[0] = model.model.layers[0].block  # Unwrap the original block to restore model functionality

    return res


def load_cached_inputs(names: list[str]) -> dict[str, Tensor]:
    inputs = {}
    for name in names:
        key = name.split("/")[-1].rsplit(".", 1)[0]  # Extract "layer_{i}_{key}"
        inputs[key] = torch.load(name)
    return inputs


def finetune_layerwise(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    train_loader: list[Tensor],
    lr: float = 1e-4,
    epochs_per_layer: int = 10,
    batch_size: int = 64,
    microbatch_size: int = 8,
    device: torch.device = "cuda",
    tb: Optional[SummaryWriter] = None,
) -> nn.Module:
    # dump block inputs for each layer
    inputs = get_first_block_inputs(model, dataset=train_loader)
    
    model.to('cpu')
    
    # For each layer, load the cached inputs and fine-tune that layer
    for layer_idx in range(len(model.model.layers)):
        print(f"\n{'='*80}")
        print(f"Fine-tuning Layer {layer_idx}")
        print(f"{'='*80}\n")
        
        fp_inputs = inputs
        fp_outputs = []
        
        model.model.layers[layer_idx].to(device)

        with torch.no_grad():
            for i in range(len(fp_inputs["hidden_states"])):
                batch_input = {}
                batch_input["hidden_states"] = fp_inputs["hidden_states"][i]
                for key in fp_inputs:
                    if key != "hidden_states":
                        batch_input[key] = fp_inputs[key]

                output = model.model.layers[layer_idx](**batch_input)[0]
                fp_outputs.append(output)

        print(f"  Starting fine-tuning for layer {layer_idx}...")
        
        layer = wrap_model_block(model.model.layers[layer_idx].to(device), n_bits=2)  # Wrap the layer with codebook wrapper for fine-tuning
        model.model.layers[layer_idx] = finetune_layer_l2(
            layer=layer,
            fp_inputs=fp_inputs,
            fp_outputs=fp_outputs,
            layer_idx=layer_idx,
            lr=lr,
            epochs_per_layer=epochs_per_layer,
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            device=device,
            tb=tb
        ).to('cpu')  # Move back to CPU after fine-tuning this layer


def collate_fn(batch):
    hidden_states = torch.cat([item["hidden_states"] for item in batch], dim=0)

    kwargs = {}
    for key in batch[0]:
        if key != "hidden_states":
            kwargs[key] = torch.cat([item[key] for item in batch], dim=0)
    return hidden_states, kwargs


def finetune_layer_l2(
    layer: nn.Module,
    fp_inputs: list[Tensor],
    fp_outputs: list[Tensor],
    layer_idx: int = -1,
    lr: float = 1e-4,
    epochs_per_layer: int = 10,
    batch_size: int = 64,
    microbatch_size: int = 8,
    device: torch.device = "cuda",
    tb: Optional[SummaryWriter] = None,
    return_next_layer_inputs: bool = False
) -> nn.Module:
    """
    Fine-tunes a compressed model layer-wise using L2 loss between outputs of original and compressed models.
    
    :param layer: The layer to be fine-tuned
    :param fp_inputs: List of input tensors for the layer
    :param fp_outputs: List of output tensors for the layer
    :param lr: Learning rate for optimization
    :param epochs_per_layer: Number of epochs to train each layer
    :param batch_size: Total batch size for training
    :param microbatch_size: Size of each microbatch for gradient accumulation
    :param device: Device to run training on
    :param tb: Optional TensorBoard SummaryWriter for logging
    :param return_next_layer_inputs: Whether to return the inputs for the next layer
    :return: The fine-tuned compressed model
    """
        
    # Set up parameters to train for this layer only
    param_to_train = []
    for name, param in layer.named_parameters():
        if "codebook" in name or "scale" in name:
            param.requires_grad = True
            param_to_train.append(param)
        else:
            param.requires_grad = False
    
    if not param_to_train:
        print(f"WARNING: No trainable parameters found in layer {layer_idx}, skipping fine-tuning for this layer.")
        return layer  # No parameters to train, return original layer

    # Create optimizer for this layer
    opt = torch.optim.AdamW(param_to_train, lr=lr)
    lambda_lr = lambda epoch: 0.99 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)
    
    # Training loop for this layer
    grad_accumulation_steps = batch_size // microbatch_size
    num_samples = len(fp_inputs)
    epoch_samples = num_samples - num_samples % microbatch_size
    microbatches_per_epoch = epoch_samples // microbatch_size
    
    global_step = epochs_per_layer * microbatches_per_epoch
    
    for epoch in range(epochs_per_layer):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        epoch_loss = 0.0
        num_batches = 0
        loss_numerator = grad_steps = 0
        
        for indices in track(
            batch_indices_epoch, 
            description=f"  Layer {layer_idx}, Epoch {epoch}/{epochs_per_layer}"
        ):
            indices = indices.tolist()
            
            # Form batch
            batch_inputs = collate_fn([{"hidden_states": fp_inputs[i], "attention_mask": None, "position_ids": None} for i in indices])
            layer_outputs = layer(**batch_inputs)[0]
            orig_output = torch.cat([fp_outputs[i] for i in indices], dim=0).to(device)
            
        
            
            # Compute L2 loss between outputs
            loss = F.mse_loss(layer_outputs, orig_output.to(dtype=layer_outputs.dtype))
            
            # Gradient accumulation
            loss_numerator += loss.item()
            grad_steps += 1
            
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss} at layer {layer_idx}"
                raise ValueError(err)
            
            (loss / grad_accumulation_steps).backward()
            
            if grad_steps == grad_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(param_to_train, 1.0)
                opt.step()
                opt.zero_grad()
                
                aggregated_loss = loss_numerator / grad_steps
                epoch_loss += aggregated_loss
                num_batches += 1
                loss_numerator = grad_steps = 0
                
                if tb is not None:
                    tb.add_scalar(f"layerwise_loss/layer_{layer_idx}", aggregated_loss, global_step)
                    tb.add_scalar(f"layerwise_lr/layer_{layer_idx}", opt.param_groups[0]["lr"], global_step)
                
                global_step += 1

        scheduler.step()
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"    Epoch {epoch}: avg loss = {avg_epoch_loss:.6f}")
    
    # Clear cache after each layer
    torch.cuda.empty_cache()
    print(f"\n{'='*80}")
    print("Layer-wise fine-tuning complete!")
    print(f"{'='*80}\n")

    if return_next_layer_inputs:
        next_inputs = []
        with torch.no_grad():
            for i in range(num_samples):
                batch_input = collate_fn(fp_inputs[i:i+1])
                next_input = layer(**batch_input)[0].cpu()
                next_inputs.append(next_input)
        return layer, next_inputs

    return layer



def example_layerwise_training():
    """
    Example demonstrating how to use the layer-wise fine-tuning function independently.
    
    This example shows:
    1. Loading original and compressed models
    2. Preparing training data
    3. Running layer-wise fine-tuning with L2 loss
    4. Saving the fine-tuned model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16
    
    # Load models
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    orig_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
    compressed_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Wrap compressed model with codebook layers
    compressed_model = wrap_model(compressed_model, n_bits=2)
    
    # Prepare training data
    train_loader = get_compression_calibration(
        num_samples=128, 
        seqlen=512, 
        tokenizer=tokenizer, 
        device=device
    )
    
    # Run layer-wise fine-tuning
    compressed_model = finetune_layerwise_l2(
        orig_model=orig_model,
        compressed_model=compressed_model,
        train_loader=train_loader,
        lr=1e-3,
        epochs_per_layer=5,
        batch_size=32,
        microbatch_size=4,
        device=device,
        tb=None,  # Can pass SummaryWriter for logging
        layer_type="both"  # Train both MLP and attention layers
    )
    
    # Unwrap and save the model
    compressed_model = unwrap_model(compressed_model)
    compressed_model.save_pretrained("./layerwise_finetuned_model")
    tokenizer.save_pretrained("./layerwise_finetuned_model")
    
    print("Layer-wise fine-tuned model saved successfully!")
