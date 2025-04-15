# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import pprint
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from optimum.modeling_base import OptimizedModel
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
import nncf.torch
from nncf.common.graph.layer_attributes import ConstantLayerAttributes
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.experimental.torch2.function_hook.hook_storage import HookStorage
from nncf.experimental.torch2.function_hook.hook_storage import decode_hook_name
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.model_creation import load_from_config
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import AsymmetricLoraScaleQuantizer
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraScaleQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer

warnings.filterwarnings("ignore", category=TracerWarning)


def get_wikitext2(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> List[Tensor]:
    """
    Loads and processes the Wikitext-2 dataset for training.

    :param num_samples: Number of samples to generate.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer to encode the text.
    :param device: Device to move the tensors to (e.g., 'cpu' or 'cuda').
    :return: A list of tensors containing the tokenized text samples.
    """
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    limit = num_samples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    trainloader = []
    for _ in range(num_samples):
        # Crop a sequence of tokens of length seqlen starting at a random position
        i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
    return trainloader


def measure_perplexity(
    optimum_model: OptimizedModel, max_length: Optional[int] = None, limit: Optional[Union[int, float]] = None
) -> float:
    """
    Measure perplexity on the Wikitext dataset, via rolling loglikelihoods for a given model.

    :param optimum_model: A model to be evaluated.
    :param max_length: The maximum sequence length for evaluation.
    :param limit: Limit the number of examples per task (only use this for testing).
        If <1, limit is a percentage of the total number of examples.
    :return: The similarity score as a float.
    """
    print("#" * 50 + " Evaluate via lm-eval-harness " + "#" * 50)
    lm_obj = OptimumLM(pretrained=optimum_model, max_length=max_length)
    results = simple_evaluate(lm_obj, tasks=["wikitext"], limit=limit)
    return results["results"]["wikitext"]["word_perplexity,none"]


@torch.no_grad()
def calc_hiddens(model: nn.Module, dataloader: List[Tensor]) -> List[Tensor]:
    """
    Calculate the hidden states for each input in the dataloader using the given model.

    :param model: The model used to calculate the hidden states.
    :param dataloader: The dataloader providing the inputs to the model.
    :return: A list of hidden states for each input in the dataloader.
    """
    orig_hiddens = []
    for data in track(dataloader, description="Calculating original hiddens"):
        model_input = get_model_input(data)
        orig_hiddens.append(model.model(**model_input).last_hidden_state)
    torch.cuda.empty_cache()
    return orig_hiddens


def get_model_input(input_ids: Tensor) -> Dict[str, Tensor]:
    """
    Prepares the model input dictionary with input IDs, attention mask, and position IDs.

    :param input_ids: Tensor containing the input IDs.
    :return: A dictionary with keys "input_ids", "attention_mask", and "position_ids",
        each mapping to their respective tensors.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def kl_div(student_hiddens: torch.Tensor, teacher_hiddens: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence loss between the student and teacher hidden states.
    The input tensors are expected to have the same shape, and the last dimension represents the number of classes.

    :param student_hiddens: The hidden states from the student model.
    :param teacher_hiddens: The hidden states from the teacher model.
    :returns: The computed KL divergence loss.
    """
    num_classes = student_hiddens.shape[-1]
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, num_classes), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def set_trainable(model: nn.Module, lora_lr: float, fq_lr: float) -> List[Dict[str, Any]]:
    """
    Sets the trainable parameters of the model for quantization-aware training with LoRA (Low-Rank Adaptation).

    This function disables gradients for all parameters in the model, then selectively enables gradients for
    specific quantizers (AsymmetricLoraQuantizer, SymmetricLoraQuantizer) that have 4-bit quantization.
    It collects the trainable parameters and adapters from these quantizers and returns them in a format
    suitable for an optimizer.

    :param model: The model to be trained.
    :param lora_lr: Learning rate for the LoRA adapters.
    :param fq_lr: Learning rate for the quantizer scales.
    :return: A list of dictionaries containing the parameters to be optimized and their corresponding learning rates.
    """
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    hook_storage = get_hook_storage(model)

    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer, AsymmetricLoraScaleQuantizer, SymmetricLoraScaleQuantizer)) and (module.num_bits == 4):
            module.enable_gradients()
            params = module.get_trainable_params()
            adapters = module.get_adapters()
            adapters_to_train.extend(adapters.values())
            scales_to_train.extend(param for name, param in params.items() if name not in adapters)

    params = list(model.parameters())
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    all_param = sum(p.numel() for p in params)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    model.train()
    return [{"params": adapters_to_train, "lr": lora_lr}, {"params": scales_to_train, "lr": fq_lr}]


def save_checkpoint(model: nn.Module, ckpt_file: Path) -> None:
    """
    Saves the state of a tuned model from a checkpoint.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    """
    hook_storage = get_hook_storage(model)
    ckpt = {"nncf_state_dict": hook_storage.state_dict(), "nncf_config": nncf.torch.get_config(model)}
    torch.save(ckpt, ckpt_file)


def load_checkpoint(model: nn.Module, example_input: Any, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param example_input: An example input that will be used for model tracing. It's required to insert and run FQs.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False)
    model = load_from_config(model, ckpt["nncf_config"], example_input=example_input)
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


@torch.no_grad()
def export_to_openvino(
    pretrained: str, example_input: torch.Tensor, ckpt_file: Path, ir_dir: Path
) -> OVModelForCausalLM:
    """
    Create a wrapper of OpenVINO model from the checkpoint for evaluation on CPU via WWB.

    :param pretrained: The name or path of the pretrained model.
    :param example_input: A tensor representing an example input for the model.
    :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
    :param last_dir: The directory where the OpenVINO model will be saved.
    :return: A wrapper of OpenVINO model ready for evaluation.
    """
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float32, device_map="cpu")
    model_input = get_model_input(example_input.to("cpu"))
    model_to_eval = load_checkpoint(model_to_eval, model_input, ckpt_file)
    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.DQ, example_input=model_input)
    export_from_model(model_to_eval, ir_dir, device="cpu")
    return OVModelForCausalLM.from_pretrained(
        model_id=ir_dir,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=True,
    )


def limit_type(astr: str):
    value = int(astr)
    if value < 0 or value > 1:
        msg = "value not in range [0,1]"
        raise argparse.ArgumentTypeError(msg)
    return value


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained",
        type=str,
        default="HuggingFaceTB/SmolLM-1.7B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output",
        help="Path to the directory for storing logs, tuning checkpoint, compressed model, validation references.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to start from previously saved checkpoint. If not specified or checkpoint does not exist, "
        "start from scratch by post-training weight compression initialization.",
    )
    parser.add_argument("--lora_rank", type=int, default=256, help="Rank of lora adapters")

    # Data params
    parser.add_argument("--num_train_samples", type=int, default=1024, help="Number of training samples")
    parser.add_argument("--num_val_samples", type=int, default=None, help="Number of validation samples for WWB.")
    parser.add_argument("--calib_seqlen", type=int, default=1024, help="Calibration data context length.")
    parser.add_argument("--eval_seqlen", type=int, default=2048, help="Evaluation data context length.")
    parser.add_argument(
        "--limit",
        type=limit_type,
        default=None,
        help="A percentage of the total number of examples for evaluation. "
        "Should be on the range [0,1]. If None, all samples will be used.",
    )

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning. "
        "For larger models (over 3 billion parameters), a learning rate of 5e-5 is recommended.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=8,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    return parser


@torch.no_grad()
def remove_fq_in_torch_model(
    orig_model: AutoModelForCausalLM, model_input: torch.Tensor, ckpt_file: Path
) -> AutoModelForCausalLM:
    model = load_checkpoint(orig_model, model_input, ckpt_file)
    # model_to_eval = nncf.strip(model_to_eval, strip_format=StripFormat.FLOAT)
    transformations = model.nncf.transformation_layout().transformations
    graph = model.nncf.get_graph()
    model = model.nncf.get_clean_shallow_copy()
    for command in transformations:
        quantizer = command.fn
        tp = command.target_points[0]
        weight_node = graph.get_node_by_name(tp.target_node_name)
        # node_with_weight = graph.get_node_by_name(tp.target_node_name)
        # weight_node = get_const_node(node_with_weight, tp.input_port_id, graph)
        weight = get_const_data(weight_node, model)
        fq_weight = quantizer.quantize(weight)

        weight_name = weight_node.layer_attributes.name
        module_name, const_attr_name = split_const_name(weight_name)
        module = get_module_by_name(module_name, model)
        weight = getattr(module, const_attr_name)
        weight.requires_grad = False
        weight.data = fq_weight
    return model


def remove_fq_in_torch2_model(orig_model: AutoModelForCausalLM, model_input: torch.Tensor, ckpt_file: Path):
    model = load_checkpoint(orig_model, model_input, ckpt_file)
    graph = build_nncf_graph(model, model_input)
    hook_storage = get_hook_storage(orig_model)

    for name, module in hook_storage.named_hooks():
        if not isinstance(module, (SymmetricQuantizer, AsymmetricQuantizer)):
            continue
        msg = ""
        if module._qspec.half_range or module._qspec.narrow_range:
            msg += "Unexpected parameters of quantizers on strip: half_range and narrow_range should be False.\n"
        if module.num_bits not in [4, 8]:
            msg += f"Unsupported number of bits {module.num_bits} for the quantizer {module}.\n"
        if msg:
            raise nncf.ValidationError(msg)

        _, op_name, _ = decode_hook_name(name)
        weight_node = graph.get_node_by_name(op_name)

        if weight_node is None:
            msg = "FQ is not assigned to weight. Strip to DQ format is not supported for FQ on activation."
            raise nncf.UnsupportedModelError(msg)

        if not isinstance(weight_node.layer_attributes, ConstantLayerAttributes):
            msg = f"Unexpected layer attributes type {type(weight_node.layer_attributes)}"
            raise nncf.InternalError(msg)

        weight = get_const_data(weight_node, model)
        fq_weight = module.quantize(weight)

        module_name, weight_attr_name = split_const_name(weight_node.layer_attributes.name)
        module = get_module_by_name(module_name, model)
        weight_param = getattr(module, weight_attr_name)

        weight_param.requires_grad = False
        weight_param.data = fq_weight

        # ATR_HOOK_STORAGE
    model.__nncf_hooks = HookStorage()
    # hook_storage.set_submodule(name, decompressor)

    # hook_key = HookStorage._generate_key(op_name, port_id)
    # hook_storage.set_submodule(name, decompressor)

    # if hook_key not in storage_dict:
    #     return value
    # for hook in storage_dict[hook_key].values():
    #     value = hook(value)

    # storage = getattr(hook_storage, hook_type).storage_ref()

    # if storage is not None and self.key in storage and self.id in storage[self.key]:
    #     del storage[self.key][self.id]
    #     if not storage[self.key]:
    #         del storage[self.key]
    return model


def main(argv) -> float:
    """
    Fine-tunes the specified model and returns the difference between initial and best validation similarity scores.
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    assert torch.cuda.is_available()
    transformers.set_seed(42)
    device = "cuda"
    torch_dtype = torch.bfloat16
    compression_config = dict(
        mode=CompressWeightsMode.INT4_SYM,
        group_size=128,
        compression_format=CompressionFormat.FQ_LORA_SCALE,
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=args.lora_rank),
    )

    # Configure output and log files.
    output_dir = Path(args.output_dir)
    tensorboard_dir = output_dir / "tb" / datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    last_dir = output_dir / "last"
    best_dir = output_dir / "best"
    if not args.resume:
        shutil.rmtree(output_dir, ignore_errors=True)
    for path in [output_dir, tensorboard_dir, last_dir, best_dir]:
        path.mkdir(exist_ok=True, parents=True)
    ckpt_file = last_dir / "nncf_checkpoint.pth"
    print(f"To visualize the loss and validation metrics, open Tensorboard using the logs from: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with absorbable LoRA")
    hparams = vars(args)
    hparams["output_dir"] = str(hparams["output_dir"])
    pprint.pprint(hparams)
    tb.add_hparams(hparams, {})

    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Prepare training data and pre-compute hiddens of teacher model for distillation loss.
    train_loader = get_wikitext2(
        num_samples=args.num_train_samples, seqlen=args.calib_seqlen, tokenizer=tokenizer, device=device
    )
    orig_hiddens = calc_hiddens(model, train_loader)

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    example_input = get_model_input(train_loader[0])
    if args.resume and ckpt_file.exists():
        model = load_checkpoint(model, example_input, ckpt_file)
    else:
        model = compress_weights(model, dataset=Dataset([example_input]), **compression_config)
        save_checkpoint(model, ckpt_file)
    fq_lr = args.lr / 10
    weight_decay = args.lr
    param_to_train = set_trainable(model, lora_lr=args.lr, fq_lr=fq_lr)
    opt = torch.optim.AdamW(param_to_train, weight_decay=weight_decay)

    # Convert torch checkpoint to an OpenVINO model and evaluate it via LM-Evaluation-Harness.
    # model_to_eval = export_to_openvino(args.pretrained, train_loader[0], ckpt_file, last_dir)
    model_to_eval = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch.bfloat16, device_map="cuda")
    model_to_eval = remove_fq_in_torch2_model(model_to_eval, example_input, ckpt_file)
    best_perplexity = measure_perplexity(model_to_eval, args.eval_seqlen, args.limit)
    tb.add_scalar("perplexity", best_perplexity, 0)
    print(f"Initial perplexity on wikitext = {best_perplexity:.4f}")
    del model_to_eval

    # Run tuning with distillation loss and validation after each epoch.
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_microbatches = 0
    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for indices in track(batch_indices_epoch, description=f"Train epoch {epoch}"):
            indices = indices.tolist()
            total_microbatches += 1

            def form_batch(inputs: List[Tensor], model_input: bool):
                batch = torch.cat([inputs[i] for i in indices], dim=0)
                return get_model_input(batch) if model_input else batch.to(device=device, dtype=torch_dtype)

            # Compute distillation loss between logits of the original model and the model with FQ + LoRA.
            inputs = form_batch(train_loader, model_input=True)
            with torch.no_grad():
                targets = model.lm_head(form_batch(orig_hiddens, model_input=False))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma has post-processing after lm_head
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls
            outputs = model(**inputs).logits
            loss = kl_div(outputs, targets.to(dtype=torch_dtype, device=device))

            # Perform an optimization step after accumulating gradients over multiple minibatches.
            loss_numerator += loss.item()
            grad_steps += 1
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)
            (loss / grad_accumulation_steps).backward()
            if grad_steps == grad_accumulation_steps:
                opt.step()
                opt.zero_grad()
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
            tb.add_scalar("loss", aggregated_loss, total_microbatches)

        # Export tuned model to OpenVINO and evaluate it using LM-Evaluation-Harness.
        # Save the best checkpoint and OpenVINO IR for the lowest perplexity.
        save_checkpoint(model, ckpt_file)
        model_to_eval = AutoModelForCausalLM.from_pretrained(
            args.pretrained, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model_to_eval = remove_fq_in_torch2_model(model_to_eval, example_input, ckpt_file)
        model_for_eval = export_to_openvino(args.pretrained, train_loader[0], ckpt_file, last_dir)
        perplexity = measure_perplexity(model_to_eval, args.eval_seqlen, args.limit)
        tb.add_scalar("perplexity", perplexity, total_microbatches)
        print(f"[Epoch {epoch}], perplexity on wikitext = {perplexity:.4f}")
        del model_to_eval

        if perplexity < best_perplexity:
            print(f"New best perplexity = {perplexity:.4f}")
            best_perplexity = perplexity
            shutil.copytree(last_dir, best_dir, dirs_exist_ok=True)

    print(f"The finetuned OV model with the best perplexity={best_perplexity} saved to: {best_dir}")
    max_cuda_mem_eval = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    tb.add_hparams({}, metric_dict={"max_cuda_mem_eval_Gb": max_cuda_mem_eval})
    print(f"Max CUDA memory allocated in Gb: {max_cuda_mem_eval}")
    return best_perplexity


if __name__ == "__main__":
    main(sys.argv[1:])
