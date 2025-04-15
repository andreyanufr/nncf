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
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from optimum.modeling_base import OptimizedModel
from scipy.signal import savgol_filter
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import TextEvaluator

import nncf
import nncf.torch
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.parameters import BackupMode
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import AsymmetricLoraScaleQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraScaleQuantizer

from tqdm import tqdm

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


def measure_perplexity_pt(
    optimum_model, max_length: Optional[int] = None, limit: Optional[Union[int, float]] = None
) -> float:
    """
    Measure perplexity on the Wikitext dataset, via rolling loglikelihoods for a given model.

    :param optimum_model: A model to be evaluated.
    :param max_length: The maximum sequence length for evaluation.
    :param limit: Limit the number of examples per task (only use this for testing).
        If <1, limit is a percentage of the total number of examples.
    :return: The similarity score as a float.
    """
    optimum_model.eval()
    print("#" * 50 + " Evaluate via lm-eval-harness " + "#" * 50)
    with torch.inference_mode():
        lm_obj = OptimumLM(pretrained=optimum_model, max_length=max_length)
        results = simple_evaluate(lm_obj, tasks=["wikitext"], limit=limit)
    optimum_model.train()

    return results["results"]["wikitext"]["word_perplexity,none"]


@torch.no_grad()
def save_wwb_ref(model: str, tokenizer: Any, wwb_ref_file: Path, num_samples: Optional[int] = None) -> None:
    """
    Save the reference answers for the WWB (WhoWhatBenchmark) evaluation.

    :param model: The model to be evaluated.
    :param tokenizer: The tokenizer used for processing text inputs.
    :param wwb_ref_file: The file path where the reference answers will be saved.
    """
    if not wwb_ref_file.exists():
        print("#" * 50 + " Collect reference answers for WWB " + "#" * 50)
        wwb_eval = TextEvaluator(base_model=model, tokenizer=tokenizer, use_chat_template=True, num_samples=num_samples)
        wwb_eval.dump_gt(str(wwb_ref_file))
        torch.cuda.empty_cache()


def measure_similarity(
    model_for_eval: OVModelForCausalLM, tokenizer: Any, wwb_ref_file: Path, num_samples: Optional[int] = None
) -> float:
    """
    Measures the similarity of a model's output to a reference outputs from a given file using WWB evaluation.

    :param model_for_eval: An OpenVINO model to be evaluated.
    :param tokenizer: The tokenizer used for processing text data.
    :param wwb_ref_file: The file path to the reference data for WWB evaluation.
    :return: The similarity score as a float.
    """
    print("#" * 50 + " Evaluate via WWB " + "#" * 50)
    wwb_eval = TextEvaluator(
        tokenizer=tokenizer,
        gt_data=wwb_ref_file,
        test_data=str(wwb_ref_file),
        use_chat_template=True,
        num_samples=num_samples,
    )
    _, all_metrics = wwb_eval.score(model_for_eval)
    return float(all_metrics["similarity"].iloc[0])


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
        if isinstance(module, (AsymmetricLoraQuantizer,
                                AsymmetricLoraScaleQuantizer,
                                SymmetricLoraQuantizer,
                                SymmetricLoraScaleQuantizer)) and (module.num_bits == 4):
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
    if value <= 0:
        msg = "value less than 1"
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
    parser.add_argument("--seqlen", type=int, default=1024, help="Calibration data context length.")
    parser.add_argument("--num_val_samples", type=int, default=None, help="Number of validation samples for WWB.")

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning. "
        "For larger models (over 2 billion parameters), a learning rate of 5e-4 is recommended.",
    )
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=8,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    return parser


# @torch.inference_mode()
def smooth_down_proj(model):
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Smooth down_proj")):
        A = layer.mlp.down_proj.weight.data  # .double().cpu().numpy()
        s = torch.mean(torch.abs(A), dim=0)
        s = torch.sqrt(s)
        sd = 1.0 / s.unsqueeze(0)

        if hasattr(layer.mlp, "up_proj"):  # llama
            sug = s.unsqueeze(1).to(layer.mlp.up_proj.weight.data.device)
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data * sug
        elif hasattr(layer.mlp, "gate_up_proj"):  # phi
            sug = s.unsqueeze(1).to(layer.mlp.gate_up_proj.weight.data.device)
            sz = layer.mlp.gate_up_proj.weight.data.shape
            layer.mlp.gate_up_proj.weight.data[sz[0] // 2 :, :] = (
                layer.mlp.gate_up_proj.weight.data[sz[0] // 2 :, :] * sug
            )
        else:
            continue
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data * sd


def ov_correction(module, head_dim, R2, output=True):
    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()
    R2 = R2.to(W_.device)
    if output:
        W_ = W_.t()
        transposed_shape = W_.shape
        temp = W_.reshape(-1, transposed_shape[-1] // head_dim, head_dim)
        temp = temp.to(torch.float64) @ R2
        W_ = temp.reshape(transposed_shape).t()
    else:
        init_shape = W_.shape
        temp = W_.reshape(-1, init_shape[-1] // head_dim, head_dim)
        temp = temp.to(torch.float64) @ R2
        W_ = temp.reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def rotate_model_R2(model):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating v_proj-o_proj")):
        A = layer.self_attn.v_proj.weight.data.T
        B = 1.0 * A
        B = B.double().cpu().numpy()
        B = savgol_filter(B, 7, 3, axis=0)  # , mode='nearest')
        B = torch.Tensor(B)

        shape = A.shape
        A = A.reshape(-1, shape[-1] // head_dim, head_dim)
        A = A.reshape(-1, head_dim)
        A = A.double().cpu().numpy()

        B = B.reshape(-1, shape[-1] // head_dim, head_dim)
        B = B.reshape(-1, head_dim)
        B = B.double().cpu().numpy()

        # B = savgol_filter(A, 7, 3, axis=0)
        R2 = np.linalg.pinv(A) @ B
        R2_inv = np.linalg.inv(R2)
        # R2 = R1.cpu().numpy()
        # R2_inv = R1.T.cpu().numpy()

        print("A - B ", np.mean(np.abs(A - B)))
        print("A @ R2 - B ", np.mean(np.abs(A @ R2 - B)))

        R2 = torch.Tensor(R2).to(layer.self_attn.v_proj.weight.device).double()
        R2_inv = torch.Tensor(R2_inv).to(layer.self_attn.o_proj.weight.device).double()

        ov_correction(layer.self_attn.v_proj, head_dim, R2, True)
        ov_correction(layer.self_attn.o_proj, head_dim, R2_inv, False)

        A_ = layer.self_attn.v_proj.weight.data.T
        shape = A_.shape
        A_ = A_.reshape(-1, shape[-1] // head_dim, head_dim)
        A_ = A_.reshape(-1, head_dim)
        A_ = A_.double().cpu().numpy()
        print("A @ R2 - A_ ", np.mean(np.abs(A @ R2.cpu().numpy() - A_)))


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
        backup_mode=BackupMode.NONE,
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=args.lora_rank),
    )

    # Configure output and log files.
    output_dir = Path(args.output_dir)
    tensorboard_dir = output_dir / "tb" / datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    last_dir = output_dir / "last"
    best_dir = output_dir / "best"
    # if not args.resume:
    #     shutil.rmtree(output_dir, ignore_errors=True)
    for path in [output_dir, tensorboard_dir, last_dir, best_dir]:
        path.mkdir(exist_ok=True, parents=True)
    wwb_ref_file = output_dir / "wwb_ref.csv"
    ckpt_file = last_dir / "nncf_checkpoint.pth"
    print(f"To visualize the loss and validation metrics, open Tensorboard using the logs from: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with absorbable LoRA")

    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    if model.config.tie_word_embeddings:
        model.config.tie_word_embeddings = False
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    smooth_down_proj(model)

    # Use WhoWhatBench tool (WWB) is for validation during tuning. It estimates the similarity score between embedding
    # computed by for data generated by two models, original floating-point one and optimized.
    # TODO: (nlyalyus) Use original model for collecting reference, once the bug in WWB resolved.
    # wwb_ref_model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="cpu")
    # save_wwb_ref(wwb_ref_model, tokenizer, wwb_ref_file)
    # del wwb_ref_model

    # Prepare training data and pre-compute hiddens of teacher model for distillation loss.
    train_loader = get_wikitext2(
        num_samples=args.num_train_samples, seqlen=args.seqlen, tokenizer=tokenizer, device=device
    )
    orig_hiddens = calc_hiddens(model, train_loader)

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    example_input = get_model_input(train_loader[0])
    if args.resume and ckpt_file.exists():
        model = load_checkpoint(model, example_input, ckpt_file)
    else:
        model = compress_weights(model, dataset=Dataset([example_input]), **compression_config)
        save_checkpoint(model, ckpt_file)
    fq_lr = args.lr
    weight_decay = args.lr / 10
    param_to_train = set_trainable(model, lora_lr=args.lr, fq_lr=fq_lr)
    opt = torch.optim.AdamW(param_to_train, weight_decay=weight_decay)

    # Convert torch checkpoint to an OpenVINO model and evaluate it via WWB.
    # model_for_eval = export_to_openvino(args.pretrained, train_loader[0], ckpt_file, last_dir)
    best_perplexity = measure_perplexity_pt(model, model.config.max_, args.limit)
    tb.add_scalar("perplexity", best_perplexity, 0)
    print(f"Initial perplexity on wikitext = {best_perplexity:.4f}")
    # del model_for_eval

    # Run tuning with distillation loss and validation on WWB after each epoch.
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * num_samples // grad_accumulation_steps, eta_min=0.)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
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
            loss = kl_div(outputs, targets.to(device=device, dtype=torch_dtype))
            tb.add_scalar("iter loss", loss.item(), total_microbatches)


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
                # scheduler.step()
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
            tb.add_scalar("loss", aggregated_loss, total_microbatches)
        scheduler.step()
        # Export tuned model to OpenVINO and evaluate it using WWB.
        # Save the best checkpoint and OpenVINO IR for the highest similarity score obtained from WWB.
        save_checkpoint(model, ckpt_file)
        # model_for_eval = export_to_openvino(args.pretrained, train_loader[0], ckpt_file, last_dir)
        perplexity = measure_perplexity_pt(model, args.eval_seqlen, args.limit)
        tb.add_scalar("perplexity", perplexity, total_microbatches)
        print(f"[Epoch {epoch}], perplexity on wikitext = {perplexity:.4f}")
        # del model_for_eval
        if perplexity < best_perplexity:
            print(f"New best perplexity = {perplexity:.4f}")
            best_perplexity = perplexity
            shutil.copytree(last_dir, best_dir, dirs_exist_ok=True)

    perplexity = measure_perplexity_pt(model, args.eval_seqlen, None)
    print(f"Final PT perplexity on wikitext = {perplexity:.4f}")

    model_for_eval = export_to_openvino(args.pretrained, train_loader[0], ckpt_file, best_dir)
    perplexity = measure_perplexity(model_for_eval, args.eval_seqlen, None)
    print(f"Final OV perplexity on wikitext = {perplexity:.4f}")

    print(f"The finetuned OV model with the best perplexity={best_perplexity} saved to: {best_dir}")
    return best_perplexity


if __name__ == "__main__":
    main(sys.argv[1:])
