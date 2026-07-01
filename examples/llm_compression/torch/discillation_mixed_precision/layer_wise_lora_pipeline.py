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
"""
Layer-wise LoRA + grouped quantization compression pipeline.

This script implements the requested constraints:

1) Layer-wise L2 loss with 5% activation outlier removal.
2) The input to layer ``i + 1`` in the student comes from the quantized
   output of layer ``i`` (full student forward pass with quantized wrappers).
3) The optimization target is always the original FP teacher layer output.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from quant_lora_linear import QuantizedLoraLinear
from quant_lora_linear import unwrap_linear_layers
from torch import Tensor
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def get_pile(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    """
    Sample random fixed-length windows from Pile-10k.

    :param num_samples: Number of training samples.
    :param seqlen: Sequence length of each sample.
    :param tokenizer: HF tokenizer.
    :param device: Device where token tensors are stored.
    :return: List of token-id tensors of shape ``[1, seqlen]``.
    """
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    trainloader: list[Tensor] = []
    for example in ds:
        trainenc = tokenizer(example["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        inp = trainenc.input_ids[:, i : i + seqlen].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break
    return trainloader


def get_model_input(input_ids: Tensor) -> dict[str, Tensor]:
    """
    Build model input dict from token ids.

    :param input_ids: Token IDs tensor of shape ``[batch, seq_len]``.
    :return: Standard HF causal-LM input dictionary.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def masked_l2_loss(student_h: Tensor, teacher_h: Tensor, outlier_ratio: float = 0.05) -> Tensor:
    """
    Compute L2 loss while dropping top activation outliers from the target.

    The mask is built from ``|teacher_h|``. Elements above the
    ``1 - outlier_ratio`` quantile are excluded.

    :param student_h: Student hidden states.
    :param teacher_h: Teacher hidden states.
    :param outlier_ratio: Ratio of largest activations to remove.
    :return: Scalar masked MSE loss.
    """
    if outlier_ratio <= 0.0:
        return F.mse_loss(student_h, teacher_h)

    target_abs = teacher_h.detach().abs().float().reshape(-1)
    threshold = torch.quantile(target_abs, q=1.0 - outlier_ratio)
    mask = (teacher_h.detach().abs() <= threshold).to(student_h.dtype)

    sq = (student_h - teacher_h).pow(2)
    denom = mask.sum().clamp_min(1.0)
    return (sq * mask).sum() / denom


def get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """
    Resolve the list of decoder layers from a HuggingFace CausalLM.

    :param model: CausalLM model.
    :return: Ordered list of transformer layers.
    :raises ValueError: If layers cannot be resolved.
    """
    base = getattr(model, "model", None)
    if base is None:
        raise ValueError("Expected model.model to exist for decoder layer access.")

    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("Expected model.model.layers to exist.")
    return list(layers)


def wrap_layer_linears_with_quant_lora(
    layer: nn.Module,
    num_bits: int,
    group_size: int,
    symmetric: bool,
    lora_rank: int,
    log_scale: bool,
) -> int:
    """
    Replace every eligible ``nn.Linear`` in a layer with QuantizedLoraLinear.

    :param layer: Transformer block.
    :param num_bits: Quantization bit-width.
    :param group_size: Quantization group size.
    :param symmetric: Use symmetric quantization when ``True``.
    :param lora_rank: LoRA rank.
    :param log_scale: Use log-scale parameterization when ``True``.
    :return: Number of wrapped linear modules.
    """
    replaced = 0
    for parent in layer.modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear) or isinstance(child, QuantizedLoraLinear):
                continue
            effective_group = child.in_features if group_size == -1 else group_size
            if effective_group <= 0 or (child.in_features % effective_group) != 0:
                continue
            wrapped = QuantizedLoraLinear.from_linear(
                child,
                num_bits=num_bits,
                group_size=group_size,
                symmetric=symmetric,
                lora_rank=lora_rank,
                log_scale=log_scale,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(parent, child_name, wrapped)
            replaced += 1
    return replaced


def make_layer_param_groups(layer: nn.Module, lora_lr: float, scale_lr: float) -> list[dict[str, Any]]:
    """
    Collect trainable params for a single layer's QuantizedLoraLinear modules.

    :param layer: Transformer block whose wrappers should be trained.
    :param lora_lr: Learning rate for LoRA parameters.
    :param scale_lr: Learning rate for quantization scales.
    :return: Optimizer parameter groups.
    """
    adapters: list[nn.Parameter] = []
    scales: list[nn.Parameter] = []

    for module in layer.modules():
        if not isinstance(module, QuantizedLoraLinear):
            continue
        module._scale_param.requires_grad_(True)
        scales.append(module._scale_param)
        if module.lora_rank > 0:
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
            adapters.extend([module.lora_a, module.lora_b])

    groups: list[dict[str, Any]] = []
    if adapters:
        groups.append({"params": adapters, "lr": lora_lr})
    if scales:
        groups.append({"params": scales, "lr": scale_lr})
    return groups


def build_batches(train_loader: list[Tensor], batch_size: int) -> list[Tensor]:
    """
    Build shuffled token batches from a sample list.

    :param train_loader: List of ``[1, seqlen]`` token tensors.
    :param batch_size: Batch size.
    :return: List of ``[batch, seqlen]`` batches.
    """
    indices = list(range(len(train_loader)))
    random.shuffle(indices)
    batches: list[Tensor] = []
    for i in range(0, len(indices), batch_size):
        chunk = indices[i : i + batch_size]
        if len(chunk) < batch_size:
            continue
        batches.append(torch.cat([train_loader[j] for j in chunk], dim=0))
    return batches


def get_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=Path, default=Path("output_layer_wise_lora"))

    parser.add_argument("--num_bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--log_scale", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)

    parser.add_argument("--num_train_samples", type=int, default=256)
    parser.add_argument("--train_seqlen", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--epochs_per_layer", type=int, default=1)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--scale_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--outlier_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str]) -> None:
    """
    Execute layer-wise LoRA + quantization compression training.

    :param argv: Command line arguments.
    """
    args = get_argument_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    if args.outlier_ratio < 0.0 or args.outlier_ratio >= 1.0:
        raise ValueError("outlier_ratio must be in [0, 1).")

    transformers.set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        args.pretrained,
        torch_dtype=dtype,
        device_map="auto",
        use_cache=False,
    ).eval()
    teacher.requires_grad_(False)

    student = copy.deepcopy(teacher)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    train_loader = get_pile(args.num_train_samples, args.train_seqlen, tokenizer, device)
    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty after tokenization and slicing.")

    teacher_layers = get_transformer_layers(teacher)
    student_layers = get_transformer_layers(student)
    if len(teacher_layers) != len(student_layers):
        raise ValueError("Teacher and student layer counts do not match.")

    wrapped_total = 0
    for layer in student_layers:
        wrapped_total += wrap_layer_linears_with_quant_lora(
            layer,
            num_bits=args.num_bits,
            group_size=args.group_size,
            symmetric=args.symmetric,
            lora_rank=args.lora_rank,
            log_scale=args.log_scale,
        )
    print(f"Wrapped {wrapped_total} linear submodules with QuantizedLoraLinear.")

    student.requires_grad_(False)

    num_layers = len(student_layers)
    for layer_idx in range(num_layers):
        current_layer = student_layers[layer_idx]
        param_groups = make_layer_param_groups(current_layer, lora_lr=args.lora_lr, scale_lr=args.scale_lr)
        if not param_groups:
            print(f"Layer {layer_idx}: no trainable quant/LoRA params, skipping.")
            continue

        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"Layer {layer_idx}: training started.")

        for epoch in range(args.epochs_per_layer):
            epoch_loss = 0.0
            n_steps = 0
            batches = build_batches(train_loader, args.batch_size)
            progress = tqdm(batches, desc=f"Layer {layer_idx} epoch {epoch}", dynamic_ncols=True, leave=True)
            for token_batch in progress:
                model_input = get_model_input(token_batch.to(device))

                with torch.no_grad():
                    teacher_out = teacher(**model_input, output_hidden_states=True)
                    teacher_h = teacher_out.hidden_states[layer_idx + 1]

                student_out = student(**model_input, output_hidden_states=True)
                student_h = student_out.hidden_states[layer_idx + 1]

                loss = masked_l2_loss(student_h, teacher_h.to(dtype=student_h.dtype), outlier_ratio=args.outlier_ratio)
                if not torch.isfinite(loss).item():
                    raise ValueError(f"Non-finite loss at layer {layer_idx}, epoch {epoch}: {loss.item()}")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_steps += 1
                progress.set_postfix({"loss": f"{loss.item():.6f}", "avg": f"{(epoch_loss / max(1, n_steps)):.6f}"})

            print(f"Layer {layer_idx} epoch {epoch}: mean loss = {epoch_loss / max(1, n_steps):.6f}")

        # Keep this layer frozen once tuned and move to the next one.
        current_layer.requires_grad_(False)

    student.eval()
    torch.save(student.state_dict(), output_dir / "layer_wise_q_lora_state_dict.pth")

    unwrap_linear_layers(student)
    final_dir = output_dir / "layer_wise_q_lora_unwrapped"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved wrapped checkpoint: {output_dir / 'layer_wise_q_lora_state_dict.pth'}")
    print(f"Saved unwrapped model: {final_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
