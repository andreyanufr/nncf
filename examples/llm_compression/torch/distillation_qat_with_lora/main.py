# Copyright (c) 2026 Intel Corporation
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
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Optional, Union

import mlflow
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset

# from lm_eval import simple_evaluate
# from lm_eval.models.optimum_lm import OptimumLM
# from optimum.exporters.openvino.convert import export_from_model
# from optimum.intel.openvino import OVModelForCausalLM
# from optimum.modeling_base import OptimizedModel
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedAWQParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import StretchedSymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.quantize_functions import set_use_autograd_quantize

warnings.filterwarnings("ignore", category=TracerWarning)


def get_wikitext2(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    """
    Loads and processes the Wikitext-2 dataset for training.

    :param num_samples: Number of samples to generate.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer to encode the text.
    :param device: Device to move the tensors to (e.g., 'cpu' or 'cuda').
    :return: A list of tensors containing the tokenized text samples.
    """
    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
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


def get_pile(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device):
    # def preprocess_fn(example):
    #     return {"text": tokenizer.apply_chat_template(example["text"], add_generation_prompt=False, tokenize=False)}
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    # ds = ds.shuffle(seed=42).select(range(10 * num_samples))
    # ds = ds.map(preprocess_fn)

    trainloader = []
    for example in ds:
        trainenc = tokenizer(example["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break

    return trainloader


DATASET_LOADERS = {
    "pile": get_pile,
    "wikitext": get_wikitext2,
}


def evaluate_with_vllm(
    checkpoint_dir: Union[str, Path],
    tasks: list[str],
    tensor_parallel_size: int = 2,
    dtype: str = "auto",
    fewshot_as_multiturn: bool = True,
    apply_chat_template: bool = True,
    batch_size: str = "auto",
    limit: Optional[Union[int, float]] = None,
    cuda_devices: str = "1,2",
) -> dict:
    """
    Evaluate a model using lm_eval with vLLM backend in a subprocess.

    Runs evaluation in a subprocess to ensure CUDA_VISIBLE_DEVICES is set before
    CUDA initialization, which is required for proper GPU memory management.

    :param checkpoint_dir: Path to the model checkpoint directory.
    :param tasks: List of evaluation tasks (e.g., ["gsm8k"]).
    :param tensor_parallel_size: Number of GPUs for tensor parallelism.
    :param dtype: Data type for the model (e.g., "auto", "float16", "bfloat16").
    :param fewshot_as_multiturn: Whether to use fewshot examples as multi-turn conversation.
    :param apply_chat_template: Whether to apply the chat template.
    :param batch_size: Batch size for evaluation ("auto" for automatic).
    :param limit: Limit the number of examples per task (only use this for testing).
    :param cuda_devices: Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES.
    :return: Dictionary containing evaluation results.
    """
    print("#" * 50 + " Evaluate via lm-eval-harness (vLLM) " + "#" * 50)

    checkpoint_path = str(checkpoint_dir)
    model_args = f"pretrained={checkpoint_path},dtype={dtype},tensor_parallel_size={tensor_parallel_size}"

    cmd = [
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(checkpoint_dir / "lm_eval_results"),
    ]
    if fewshot_as_multiturn:
        cmd.append("--fewshot_as_multiturn")
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    # Set CUDA_VISIBLE_DEVICES in subprocess environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    print(f"Running: CUDA_VISIBLE_DEVICES={cuda_devices} {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        msg = f"lm_eval failed with return code {result.returncode}"
        raise RuntimeError(msg)

    # Parse results from the output JSON file
    results_dir = checkpoint_dir / "lm_eval_results"
    # Find the most recent results file (lm_eval names them results_<timestamp>.json)
    result_files = list(results_dir.glob("**/results_*.json"))
    if not result_files:
        msg = f"No results_*.json found in {results_dir}"
        raise FileNotFoundError(msg)
    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
    with open(latest_result) as f:
        results = json.load(f)

    # Print results summary
    print("\nEvaluation Results:")
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if not metric.endswith("_stderr"):
                print(f"  {metric}: {value}")

    return results


@torch.no_grad()
def calc_hiddens(model: nn.Module, dataloader: list[Tensor]) -> list[Tensor]:
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


def get_model_input(input_ids: Tensor) -> dict[str, Tensor]:
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
        log_target=True,  # TODO: try without log for teacher with log_target=False
        reduction="batchmean",
    )


def log_quantizer_stats(model: nn.Module, step: int, optimizer: torch.optim.Optimizer) -> None:
    """
    Logs per-quantizer statistics to MLflow: norms of lora_A, lora_B, input_low,
    input_range (asymmetric) or scale (symmetric), gradient norms, and learning rates.

    :param model: The model containing quantizers in its hook storage.
    :param step: Current global training step.
    :param optimizer: The optimizer, used to retrieve per-parameter learning rates.
    """
    # Build a mapping from parameter id to its current learning rate
    param_id_to_lr: dict[int, float] = {}
    for group in optimizer.param_groups:
        lr = group["lr"]
        for p in group["params"]:
            param_id_to_lr[id(p)] = lr

    metrics: dict[str, float] = {}

    # Only log metrics for the first layer encountered at each bit-width to reduce MLflow overhead.
    logged_bits: set[int] = set()
    hook_storage = get_hook_storage(model)
    for name, module in hook_storage.named_hooks():
        if not isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer, StretchedSymmetricLoraQuantizer)):
            continue
        if module.num_bits not in (2, 3, 4):
            continue
        if module.num_bits in logged_bits:
            continue
        logged_bits.add(module.num_bits)

        # Shorten the name for cleaner metric keys (MLflow uses '.' as separator)
        short = name.replace("post_hooks.", "").replace("pre_hooks.", "").replace(".weight__0", "")
        prefix = f"quantizers/{short}"

        # LoRA adapter norms
        metrics[f"{prefix}/lora_A_norm"] = module.lora_A.data.float().norm().item()
        metrics[f"{prefix}/lora_B_norm"] = module.lora_B.data.float().norm().item()

        # Scale / input_low / input_range / alpha norms depending on quantizer type
        if isinstance(module, AsymmetricLoraQuantizer):
            metrics[f"{prefix}/input_low_norm"] = module.input_low.data.float().norm().item()
            metrics[f"{prefix}/input_range_norm"] = module.input_range.data.float().norm().item()
        elif isinstance(module, StretchedSymmetricLoraQuantizer):
            metrics[f"{prefix}/alpha_norm"] = module.alpha.data.float().norm().item()
        elif isinstance(module, SymmetricLoraQuantizer):
            metrics[f"{prefix}/scale_norm"] = module.scale.data.float().norm().item()

        # Gradient norms
        for param_name, param in [("lora_A", module.lora_A), ("lora_B", module.lora_B)]:
            if param.grad is not None:
                metrics[f"{prefix}/{param_name}_grad_norm"] = param.grad.data.float().norm().item()

        if isinstance(module, AsymmetricLoraQuantizer):
            for param_name, param in [("input_low", module.input_low), ("input_range", module.input_range)]:
                if param.grad is not None:
                    metrics[f"{prefix}/{param_name}_grad_norm"] = param.grad.data.float().norm().item()
        elif isinstance(module, StretchedSymmetricLoraQuantizer):
            s = module._alpha_param_storage
            if s.grad is not None:
                metrics[f"{prefix}/alpha_grad_norm"] = s.grad.data.float().norm().item()
        elif isinstance(module, SymmetricLoraQuantizer):
            s = module._scale_param_storage
            if s.grad is not None:
                metrics[f"{prefix}/scale_grad_norm"] = s.grad.data.float().norm().item()

        # Learning rates
        for param_name, param in [("lora_A", module.lora_A), ("lora_B", module.lora_B)]:
            lr_val = param_id_to_lr.get(id(param))
            if lr_val is not None:
                metrics[f"{prefix}/{param_name}_lr"] = lr_val

        if isinstance(module, AsymmetricLoraQuantizer):
            for param_name, param in [("input_low", module.input_low), ("input_range", module.input_range)]:
                lr_val = param_id_to_lr.get(id(param))
                if lr_val is not None:
                    metrics[f"{prefix}/{param_name}_lr"] = lr_val
        elif isinstance(module, StretchedSymmetricLoraQuantizer):
            lr_val = param_id_to_lr.get(id(module._alpha_param_storage))
            if lr_val is not None:
                metrics[f"{prefix}/alpha_lr"] = lr_val
        elif isinstance(module, SymmetricLoraQuantizer):
            lr_val = param_id_to_lr.get(id(module._scale_param_storage))
            if lr_val is not None:
                metrics[f"{prefix}/scale_lr"] = lr_val

    if metrics:
        mlflow.log_metrics(metrics, step=step)


def set_trainable(
    model: nn.Module,
    lora_lr: float,
    fq_lr: float,
    lora_weight_decay: float = 1e-4,
    fq_weight_decay: float = 0.0,
    tune_bits: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """
    Sets the trainable parameters of the model for quantization-aware training with LoRA (Low-Rank Adaptation).

    This function disables gradients for all parameters in the model, then selectively enables gradients for
    specific quantizers (AsymmetricLoraQuantizer, SymmetricLoraQuantizer) that have 4-bit quantization.
    It collects the trainable parameters and adapters from these quantizers and returns them in a format
    suitable for an optimizer.

    :param model: The model to be trained.
    :param lora_lr: Learning rate for the LoRA adapters.
    :param fq_lr: Learning rate for the quantizer scales.
    :param lora_weight_decay: Weight decay for LoRA adapter parameters.
    :param fq_weight_decay: Weight decay for quantizer scale parameters.
    :param tune_bits: List of bit-widths to tune (e.g. [2], [3], [4], or [2, 3, 4]).
        If None, tunes 2, 3, and 4 bit layers.
    :return: A list of dictionaries containing the parameters to be optimized and their corresponding learning rates.
    """
    if tune_bits is None:
        tune_bits = [2, 3, 4]
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    train_lora = lora_lr > 0
    train_scales = fq_lr > 0
    hook_storage = get_hook_storage(model)
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer, StretchedSymmetricLoraQuantizer)) and (
            module.num_bits in tune_bits
        ):
            # Only enable gradients for param groups that will actually be trained.
            if train_lora:
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
                adapters_to_train.extend(module.get_adapters().values())
            if train_scales:
                if isinstance(module, AsymmetricLoraQuantizer):
                    module.input_low.requires_grad = True
                    module._input_range_param_storage.requires_grad = True
                elif isinstance(module, StretchedSymmetricLoraQuantizer):
                    module._alpha_param_storage.requires_grad = True
                elif isinstance(module, SymmetricLoraQuantizer):
                    module._scale_param_storage.requires_grad = True
                params = module.get_trainable_params()
                adapters = module.get_adapters()
                scales_to_train.extend(param for name, param in params.items() if name not in adapters)

    param_groups = []
    if train_lora:
        param_groups.append({"params": adapters_to_train, "lr": lora_lr, "weight_decay": lora_weight_decay})
    if train_scales:
        param_groups.append({"params": scales_to_train, "lr": fq_lr, "weight_decay": fq_weight_decay})

    params = list(model.parameters())
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    all_param = sum(p.numel() for p in params)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
        f" (lora={'ON' if train_lora else 'OFF'}, scales={'ON' if train_scales else 'OFF'}, "
        f"tune_bits={tune_bits})"
    )
    model.train()
    return param_groups


def save_checkpoint(model: nn.Module, ckpt_file: Path, model_state: bool = True) -> None:
    """
    Stores the current state of a quantized model to a checkpoint file.

    :param model: The model whose state will be saved to checkpoint.
    :param ckpt_file: Path to store the checkpoint file.
    :param model_state: Whether to save the complete model weights in addition to NNCF state. Required when using
        AWQ method which fuses scaling factors into weights. When False, only NNCF configuration and state are saved,
        as they're maintained separately from the model's weights.
    """
    hook_storage = get_hook_storage(model)
    ckpt = {"nncf_state_dict": hook_storage.state_dict(), "nncf_config": nncf.torch.get_config(model)}
    if model_state:
        ckpt["model_state"] = model.state_dict()
    torch.save(ckpt, ckpt_file)


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    model = load_from_config(model, ckpt["nncf_config"])
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


def limit_type(astr: str):
    value = float(astr)
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
        # default="Qwen/Qwen3-8B",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output",
        help="Path to the directory for storing logs, tuning checkpoint, compressed model, validation references.",
    )
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of lora adapters")
    parser.add_argument(
        "--basic_init",
        action="store_true",
        help="Whether to initialize quantization with basic min-max round-to-nearest schema. By default, advanced "
        "data-aware post-training methods are used: AWQ + Scale Estimation. These methods typically provide better "
        "accuracy, but require a calibration dataset and additional initialization time "
        "(~20 sec for 1B and ~80 sec for 8B models).",
    )

    # Data params
    parser.add_argument(
        "--dataset",
        type=str,
        default="pile",
        choices=list(DATASET_LOADERS.keys()),
        help="Training dataset to use. 'pile' = NeelNanda/pile-10k, 'wikitext' = Salesforce/wikitext-2-raw-v1. "
        "Default: pile.",
    )
    parser.add_argument("--num_train_samples", type=int, default=256, help="Number of training samples")
    parser.add_argument("--train_seqlen", type=int, default=256, help="Train data context length.")
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
        "--fq_lr",
        type=float,
        default=1e-3,
        help="Learning rate for quantizer scales (input_low, input_range, scale). Set to 0 to freeze scales entirely.",
    )
    parser.add_argument(
        "--lora_lr",
        type=float,
        default=0.0,
        help="Learning rate for LoRA adapters (lora_A, lora_B). "
        "Set to 0 to freeze LoRA (default). Typical values: 1e-5 to 1e-4.",
    )
    parser.add_argument(
        "--fq_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization) for quantizer scale parameters. 0 disables.",
    )
    parser.add_argument(
        "--lora_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) for LoRA adapter parameters. 0 disables.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=4,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    parser.add_argument(
        "--constant_epochs",
        type=int,
        default=0,
        help="Number of epochs with constant LR (always the first phase). "
        "If >0 and cosine_epochs==0, training uses only constant LR.",
    )
    parser.add_argument(
        "--cosine_epochs",
        type=int,
        default=1,
        help="Number of epochs with cosine annealing LR (always the second phase). "
        "If >0 and constant_epochs==0, training uses only cosine annealing.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.0,
        help="Minimum LR at end of cosine annealing, expressed as a fraction of the initial LR. "
        "0.0 means LR decays to zero; 0.1 means it decays to 10%% of peak. "
        "Ignored when cosine_epochs==0.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=0.0,
        help="Number of epochs for linear warmup (LR ramps from 0 to peak). "
        "Supports fractional values, e.g. 0.5 for half an epoch. 0 disables warmup.",
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 15],
        help="Epoch numbers (0-indexed) at which to save a checkpoint. "
        "Epoch 0 means after initialization (before any training). "
        "The last epoch always saves regardless of this list.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name. If not specified, uses a timestamp.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: uses a separate MLflow database (mlflow_debug.db) and experiment prefix "
        "so that experimental runs do not pollute production results.",
    )
    parser.add_argument(
        "--mlflow_db",
        type=str,
        default=None,
        help="Override the MLflow SQLite database path. "
        "By default: <output_dir>/mlflow.db (or mlflow_debug.db in --debug mode).",
    )
    parser.add_argument(
        "--compression_format",
        type=str,
        default="FQ_LORA",
        choices=[f.name for f in CompressionFormat],
        help="Compression format to use. Key options: "
        "FQ_LORA (standard fake-quantize + LoRA), "
        "FQ_STRETCHED_LORA (ParetoQ-style stretched quantization + LoRA). "
        "Default: FQ_LORA.",
    )
    parser.add_argument(
        "--init_ckpt",
        type=Path,
        default=None,
        help="Path to the initial compression checkpoint (.pth) to resume from. "
        "If not specified, auto-derived from --compression_format: "
        "<output_dir>/nncf_init_<format>.pth (e.g. nncf_init_fq_stretched_lora.pth). "
        "This allows different formats to maintain separate init checkpoints.",
    )
    parser.add_argument(
        "--use_autograd_quantize",
        action="store_true",
        help="Use STE-based autograd for gradient computation instead of hand-written backward.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to trade compute for memory. "
        "Recomputes activations during backward instead of storing them, "
        "reducing peak GPU memory at the cost of ~30%% slower training.",
    )
    parser.add_argument(
        "--tune_bits",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Which bit-width layers to tune. Use for staged tuning: "
        "first run with --tune_bits 2 to tune only 2-bit layers, "
        "then run with --tune_bits 4 --init_ckpt <2bit_ckpt> to tune only 4-bit layers "
        "while keeping 2-bit layers frozen. Default: [2, 3, 4] (tune all).",
    )
    return parser


def main(argv) -> int:
    """
    Fine-tunes the specified model and returns 0 on success or 1 on failure.
    Errors are caught and printed so that multiple sequential runs (e.g. from shell scripts)
    are not interrupted by a single failure.
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    try:
        return _main_impl(args)
    except Exception:
        traceback.print_exc()
        print(f"\n{'!' * 60}")
        print(f"RUN FAILED: {args.run_name or 'unnamed'}")
        print(f"{'!' * 60}\n")
        # Clean up GPU memory so the next run can start fresh.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 1


def _main_impl(args) -> int:
    """Core implementation of main(). Raises on error."""
    if args.fq_lr == 0 and args.lora_lr == 0:
        print("Both --fq_lr and --lora_lr are 0 — nothing to train. Skipping.")
        return 0
    assert torch.cuda.is_available()
    transformers.set_seed(42)
    set_use_autograd_quantize(args.use_autograd_quantize)
    device = "cuda"
    torch_dtype = torch.bfloat16
    compression_config = dict(
        mode=CompressWeightsMode.INT3_SYM,
        group_size=64,
        awq=not args.basic_init,
        backup_mode=nncf.BackupMode.INT8_SYM,
        scale_estimation=not args.basic_init,
        compression_format=CompressionFormat[args.compression_format],
        # ignored_scope=nncf.IgnoredScope(patterns=[r"(?!.*5.mlp.gate_proj.*).*"]),
        # ignored_scope=nncf.IgnoredScope(
        #     patterns=[r"(?!model/mlp/(gate_proj|up_proj)/linear/5$|model/self_attn/(k_proj|q_proj|o_proj)/linear/5$).*"]
        # ),
    )
    pprint({"CLI arguments": vars(args), "Major compression parameters": compression_config})
    compression_config["advanced_parameters"] = AdvancedCompressionParameters(
        awq_params=AdvancedAWQParameters(prefer_data_aware_scaling=not args.basic_init),
        # scale_estimation_params=AdvancedScaleEstimationParameters(subset_size=-1, initial_steps=10, scale_steps=10),
        lora_adapter_rank=args.lora_rank,
    )
    # Configure output and log files.
    output_dir = Path(args.output_dir)
    last_dir = output_dir / "last"
    shutil.rmtree(last_dir, ignore_errors=True)
    for path in [output_dir, last_dir]:
        path.mkdir(exist_ok=True, parents=True)
    # Derive initial compression checkpoint path from format (or use explicit override).
    if args.init_ckpt is not None:
        ckpt_file = args.init_ckpt
    else:
        fmt_tag = args.compression_format.lower()
        ckpt_file = output_dir / f"nncf_init_{fmt_tag}.pth"
    hidden_file = output_dir / "hiddens.pth"

    # Configure MLflow tracking (SQLite backend for indexing & future-proofing; file store is deprecated).
    if args.mlflow_db:
        db_path = Path(args.mlflow_db).resolve()
    else:
        db_name = "mlflow_debug.db" if args.debug else "mlflow.db"
        db_path = output_dir.resolve() / db_name
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_prefix = "debug_" if args.debug else ""
    mlflow.set_experiment(f"{experiment_prefix}fqlora_{Path(args.pretrained).name}")
    print(f"MLflow tracking URI: {tracking_uri}  (run `mlflow ui --backend-store-uri {tracking_uri}`)")
    if args.debug:
        print("  ** DEBUG MODE — results go to a separate database and experiment **")
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    with mlflow.start_run(run_name=run_name):
        _train(args, compression_config, device, torch_dtype, last_dir, output_dir, ckpt_file, hidden_file)
    return 0


def _train(args, compression_config, device, torch_dtype, last_dir, output_dir, ckpt_file, hidden_file):
    """Load model, prepare data, run distillation QAT, strip, and save. All heavy objects are local."""
    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Prepare training and calibration data
    load_fn = DATASET_LOADERS[args.dataset]
    train_loader = load_fn(
        num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
    )
    if args.basic_init:
        example_input = {k: v.to(device) for k, v in model.dummy_inputs.items()}
        dataset = Dataset([example_input])
    else:
        calib_loader = load_fn(num_samples=128, seqlen=128, tokenizer=tokenizer, device=device)
        dataset = Dataset(map(get_model_input, calib_loader))

    # Pre-compute hiddens of teacher model for distillation loss.
    if hidden_file.exists():
        orig_hiddens = torch.load(hidden_file, weights_only=False, map_location="cpu")
    else:
        orig_hiddens = calc_hiddens(model, train_loader)
        torch.save(orig_hiddens, hidden_file)

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    if ckpt_file.exists():
        print(f"Loading existing init checkpoint: {ckpt_file}")
        model = load_checkpoint(model, ckpt_file)
    else:
        model = compress_weights(model, dataset=dataset, **compression_config)
        save_checkpoint(model, ckpt_file, model_state=False)
        print(f"Saved init checkpoint: {ckpt_file}")

    # Enable gradient checkpointing to reduce activation memory.
    # ForwardWithHooks keeps FunctionHookMode alive across forward+backward so that
    # checkpoint recomputation sees the same FQ hooks as the original forward.
    # use_reentrant=False is preferred: it counts saved tensors and the persistent
    # mode ensures the counts match between forward and recomputation.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled (use_reentrant=False)")

    param_to_train = set_trainable(
        model,
        lora_lr=args.lora_lr,
        fq_lr=args.fq_lr,
        lora_weight_decay=args.lora_weight_decay,
        fq_weight_decay=args.fq_weight_decay,
        tune_bits=args.tune_bits,
    )
    opt = torch.optim.AdamW(param_to_train)

    # Run tuning with distillation loss and validation after each epoch.
    args.epochs = args.constant_epochs + args.cosine_epochs
    assert args.epochs > 0, "At least one of --constant_epochs or --cosine_epochs must be > 0."
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    # Compute step counts from total microbatches (not per-epoch * epochs) because
    # grad_steps carries over across epoch boundaries when microbatches don't divide evenly.
    total_training_steps = (microbatches_per_epoch * args.epochs) // grad_accumulation_steps
    constant_steps = (microbatches_per_epoch * args.constant_epochs) // grad_accumulation_steps
    cosine_steps = total_training_steps - constant_steps

    # Build LR scheduler: [linear warmup] -> [constant phase] -> [cosine annealing phase].
    # Uses LambdaLR so that cosine smoothly decays from 1.0 to min_lr_ratio per param group
    # (no floor-clipping artifacts).
    warmup_steps = int(microbatches_per_epoch * args.warmup_epochs) // grad_accumulation_steps

    def _make_lr_lambda(warmup_steps: int, constant_steps: int, cosine_steps: int, min_lr_ratio: float):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            adjusted_step = step - warmup_steps
            if adjusted_step < constant_steps:
                return 1.0
            if cosine_steps <= 0:
                return 1.0
            progress = (adjusted_step - constant_steps) / cosine_steps
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

        return lr_lambda

    scheduler = LambdaLR(opt, lr_lambda=_make_lr_lambda(warmup_steps, constant_steps, cosine_steps, args.min_lr_ratio))

    print(
        f"LR schedule: {warmup_steps} warmup steps, "
        f"{args.constant_epochs} constant epoch(s) ({constant_steps} steps), "
        f"then {args.cosine_epochs} cosine epoch(s) ({cosine_steps} steps, min_lr_ratio={args.min_lr_ratio}). "
        f"Total: {args.epochs} epoch(s), {total_training_steps} steps."
    )

    # Log all meaningful parameters to MLflow.
    mlflow.log_params(
        {
            "pretrained": args.pretrained,
            "lora_rank": args.lora_rank,
            "basic_init": args.basic_init,
            "num_train_samples": args.num_train_samples,
            "train_seqlen": args.train_seqlen,
            "fq_lr": args.fq_lr,
            "lora_lr": args.lora_lr,
            "weight_decay_lora": args.lora_weight_decay,
            "weight_decay_fq": args.fq_weight_decay,
            "batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "grad_accumulation_steps": grad_accumulation_steps,
            "constant_epochs": args.constant_epochs,
            "cosine_epochs": args.cosine_epochs,
            "total_epochs": args.epochs,
            "min_lr_ratio": args.min_lr_ratio,
            "warmup_epochs": args.warmup_epochs,
            "warmup_steps": warmup_steps,
            "constant_steps": constant_steps,
            "cosine_steps": cosine_steps,
            "total_training_steps": total_training_steps,
            "compression_mode": str(compression_config["mode"]),
            "group_size": compression_config["group_size"],
            "compression_format": str(compression_config["compression_format"]),
            "awq": compression_config["awq"],
            "scale_estimation": compression_config["scale_estimation"],
            "use_autograd_quantize": args.use_autograd_quantize,
            "gradient_checkpointing": args.gradient_checkpointing,
            "tune_bits": args.tune_bits,
            "dataset": args.dataset,
        }
    )

    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_steps = 0
    save_epochs = set(args.save_epochs)

    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for indices in track(batch_indices_epoch, description=f"Train epoch {epoch}"):
            indices = indices.tolist()

            def form_batch(inputs: list[Tensor], model_input: bool):
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
                scheduler.step()
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
                total_steps += 1
                mlflow.log_metric("loss", aggregated_loss, step=total_steps)
                log_quantizer_stats(model, total_steps, opt)
                opt.zero_grad()
        # Save checkpoint at scheduled epochs (epoch+1 because we just finished this epoch).
        finished_epoch = epoch + 1
        if finished_epoch in save_epochs or epoch == args.epochs - 1:
            ckpt_name = f"nncf_checkpoint_epoch{finished_epoch}.pth"
            save_checkpoint(model, last_dir / ckpt_name, model_state=False)


if __name__ == "__main__":
    main(sys.argv[1:])
