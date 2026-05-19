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
QAT training pipeline that uses :mod:`quant_lora_linear` instead of NNCF.

The pipeline mirrors ``main.py`` (distillation against the original FP model
via teacher hidden states + ``lm_head``), but the student model is built by
wrapping every ``nn.Linear`` with :class:`QuantizedLoraLinear` from
``quant_lora_linear.py``.

After training the script:

* dequantizes the wrappers into plain ``nn.Linear`` layers (LoRA folded in)
  and saves the resulting model with ``save_pretrained`` together with the
  tokenizer;
* dumps every quantizer's ``scale`` / ``zero_point`` / ``lora_a`` /
  ``lora_b`` to a single ``quant_params.pt`` file for later inspection.

This file does not depend on NNCF.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from quant_lora_linear import QuantizedLoraLinear
from quant_lora_linear import unwrap_linear_layers
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from utils import replace_linear_with_mixer
from utils import replace_mixer_with_linear

warnings.filterwarnings("ignore", category=TracerWarning)


# ---------------------------------------------------------------------- #
# Data utilities
# ---------------------------------------------------------------------- #
def generate_answer(
    model: nn.Module, tokenizer: AutoTokenizer, question: str = "What is AI? ", max_new_tokens: int = 32
) -> str:
    """Generate an answer with greedy decoding using the chat template."""
    messages = [{"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device=model.device)
    input_len = len(input_ids[0])
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)[0]
    return tokenizer.decode(output[input_len:], skip_special_tokens=True)


def get_pile(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    """Sample ``num_samples`` random ``seqlen``-long token windows from Pile-10k."""
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


def get_distill_dataset(
    num_samples: int,
    seqlen: int,
    tokenizer: Any,
    device: torch.device,
    name: str = "mlfoundations-dev/DeepSeek-R1-Distill-Qwen-7B_eval_03-07-25_17-46_2870",
) -> list[Tensor]:
    """Distillation corpus shaped like :func:`get_pile`."""
    ds = load_dataset(name, split="train")
    trainloader: list[Tensor] = []
    for example in ds:
        text = example["context"][0]["content"] + " " + example["model_outputs"]
        trainenc = tokenizer(text, return_tensors="pt")
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
    """Build the standard ``input_ids`` / ``attention_mask`` / ``position_ids`` dict."""
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


@torch.no_grad()
def calc_hiddens(model: nn.Module, dataloader: list[Tensor]) -> list[Tensor]:
    """Cache the teacher's last hidden state for every training sample."""
    orig_hiddens: list[Tensor] = []
    for data in dataloader:
        orig_hiddens.append(model.model(**get_model_input(data)).last_hidden_state.to("cpu"))
    torch.cuda.empty_cache()
    return orig_hiddens


def kl_div(student_hiddens: Tensor, teacher_hiddens: Tensor) -> Tensor:
    """Token-wise KL divergence between student and teacher logits."""
    num_classes = student_hiddens.shape[-1]
    return F.kl_div(
        input=F.log_softmax(student_hiddens.reshape(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_hiddens.reshape(-1, num_classes), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


# ---------------------------------------------------------------------- #
# MLP equalization (average up_proj/gate_proj input scale absorbed into layer norm weights)
# ---------------------------------------------------------------------- #
def _find_up_gate_groups(model: nn.Module) -> list[tuple[nn.Module, nn.Linear, list[nn.Linear]]]:
    """
    Collect ``(parent, down_proj, [producers])`` triples where ``producers``
    are the sibling linears whose outputs are consumed by ``down_proj`` along
    its input-channel dimension.

    Recognized layouts:
      * Llama-style: ``down_proj`` consumes ``up_proj`` * SiLU(``gate_proj``);
        producers = ``[up_proj, gate_proj]``.
      * Generic: only ``up_proj`` present -> producers = ``[up_proj]``.
    """
    groups: list[tuple[nn.Module, nn.Linear, list[nn.Linear]]] = []
    for parent in model.modules():
        if not hasattr(parent, "mlp"):
            continue
        mlp = getattr(parent, "mlp")
        gate = getattr(mlp, "gate_proj", None)
        if not isinstance(gate, nn.Linear):
            continue

        up = getattr(mlp, "up_proj", None)
        if not isinstance(up, nn.Linear):
            continue

        producer = None
        for attr in ("post_attention_layernorm",):  # , "gate_proj"):
            sib = getattr(parent, attr, None)
            producer = sib

        if producer:
            groups.append((up, gate, producer))
    return groups


@torch.no_grad()
def equalize_up_gate_with_layernorm(model: nn.Module, eps: float = 1e-5) -> int:
    groups = _find_up_gate_groups(model)
    if not groups:
        return 0

    n_done = 0
    for up, gate, producer in groups:
        s_gate = gate.weight.abs().mean(dim=0).clamp_min(eps).to(device=gate.weight.device, dtype=gate.weight.dtype)
        s_up = up.weight.abs().mean(dim=0).clamp_min(eps).to(device=up.weight.device, dtype=up.weight.dtype)

        s_gate = s_gate / s_gate.norm(p=2, dim=0, keepdim=True)
        s_up = s_up / s_up.norm(p=2, dim=0, keepdim=True)

        # up_proj theoretically more sensitive to quantization
        s = 0.1 * s_gate + 0.9 * s_up
        # Divide down_proj input columns by s.
        gate.weight.mul_(1.0 / s.unsqueeze(0))
        up.weight.mul_(1.0 / s.unsqueeze(0))

        # Scale producer output rows by s.
        s_dev = s.to(device=producer.weight.device, dtype=producer.weight.dtype)
        producer.weight.mul_(s_dev)
        if hasattr(producer, "bias") and producer.bias is not None:
            producer.bias.mul_(s_dev)
        n_done += 1
    return n_done


# ---------------------------------------------------------------------- #
# MLP equalization (down_proj input scale absorbed into up_proj/gate_proj)
# ---------------------------------------------------------------------- #
def _find_mlp_groups(model: nn.Module) -> list[tuple[nn.Module, nn.Linear, list[nn.Linear]]]:
    """
    Collect ``(parent, down_proj, [producers])`` triples where ``producers``
    are the sibling linears whose outputs are consumed by ``down_proj`` along
    its input-channel dimension.

    Recognized layouts:
      * Llama-style: ``down_proj`` consumes ``up_proj`` * SiLU(``gate_proj``);
        producers = ``[up_proj, gate_proj]``.
      * Generic: only ``up_proj`` present -> producers = ``[up_proj]``.
    """
    groups: list[tuple[nn.Module, nn.Linear, list[nn.Linear]]] = []
    for parent in model.modules():
        down = getattr(parent, "down_proj", None)
        if not isinstance(down, nn.Linear):
            continue
        producers: list[nn.Linear] = []
        for attr in ("up_proj",):  # , "gate_proj"):
            sib = getattr(parent, attr, None)
            if isinstance(sib, nn.Linear) and sib.out_features == down.in_features:
                producers.append(sib)
        if producers:
            groups.append((parent, down, producers))
    return groups


@torch.no_grad()
def equalize_down_proj(
    model: nn.Module,
    calib_inputs: list[Tensor],
    eps: float = 1e-5,
) -> int:
    """
    Equalize each ``down_proj`` layer by absorbing the per-input-channel
    activation magnitude into its producers (``up_proj`` and, when present,
    ``gate_proj``).

    For every MLP block let ``s = mean(|x|, dim=batch_seq)`` measured at the
    input of ``down_proj`` over the calibration set. Then:

    * ``down_proj.weight  /= s[None, :]`` (divide along input channels)
    * For each producer ``L`` (e.g. ``up_proj``, ``gate_proj``):
      ``L.weight *= s[:, None]``  (scale output channels)
      ``L.bias   *= s``           (if a bias exists)

    Mathematically, ``down(up(x) * silu(gate(x))) = down((up(x)*s) * (silu(gate(x)*s)/s))``
    is *not* exact for the SiLU branch in general, but in practice this
    pre-quantization equalization (cf. SmoothQuant / AWQ) significantly
    flattens the weight magnitudes seen by the per-group quantizer. The
    transformation is exact when no SiLU is present (``producers == [up_proj]``).

    :param model: Model whose MLP blocks expose ``down_proj`` (and optional
        ``up_proj``/``gate_proj`` siblings) as direct attributes. Must be
        called on plain ``nn.Linear`` layers (i.e. **before** wrapping them
        with :class:`QuantizedLoraLinear`).
    :param calib_inputs: Token-id tensors used for activation statistics.
    :param eps: Lower bound for ``s`` to avoid division by zero.
    :return: Number of equalized MLP groups.
    """
    groups = _find_mlp_groups(model)
    if not groups:
        return 0

    abs_sum: dict[int, Tensor] = {}
    counts: dict[int, int] = {}
    handles = []
    for _, down, _ in groups:

        def make_hook(key: int):
            def hook(_mod, args):
                x = args[0].detach()
                x_flat = x.reshape(-1, x.shape[-1]).float()
                a = x_flat.abs().sum(dim=0)
                if key in abs_sum:
                    abs_sum[key] += a
                    counts[key] += x_flat.shape[0]
                else:
                    abs_sum[key] = a
                    counts[key] = x_flat.shape[0]

            return hook

        handles.append(down.register_forward_pre_hook(make_hook(id(down))))

    was_training = model.training
    model.eval()
    try:
        for ids in tqdm(calib_inputs, desc="MLP equalization: collecting activations"):
            model(**get_model_input(ids))
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    n_done = 0
    for _, down, producers in groups:
        key = id(down)
        if key not in abs_sum:
            continue
        # s = (abs_sum[key] / max(counts[key], 1)).clamp_min(eps).to(
        #     device=down.weight.device, dtype=down.weight.dtype
        # )
        s = down.weight.abs().mean(dim=0).clamp_min(eps).to(device=down.weight.device, dtype=down.weight.dtype)
        # Divide down_proj input columns by s.
        down.weight.mul_(1.0 / s.unsqueeze(0))
        # Scale producer output rows by s.
        for prod in producers:
            s_dev = s.to(device=prod.weight.device, dtype=prod.weight.dtype)
            prod.weight.mul_(s_dev.unsqueeze(1))
            if prod.bias is not None:
                prod.bias.mul_(s_dev)
        n_done += 1
    return n_done


# ---------------------------------------------------------------------- #
# Per-head Hadamard rotation between v_proj and o_proj
# ---------------------------------------------------------------------- #
def _build_hadamard(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Build a normalized (orthogonal) Hadamard matrix of size ``n x n``.

    Requires ``n`` to be a power of two. The returned matrix ``H`` satisfies
    ``H @ H.T == I`` (so ``H^{-1} == H.T``).

    :param n: Matrix size; must be a power of two.
    :param device: Target device.
    :param dtype: Target dtype.
    :return: ``(n, n)`` orthogonal Hadamard matrix.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be a positive power of two, got {n}")
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    h = h / (n**0.5)
    return h.to(device=device, dtype=dtype)


def _build_random_orthogonal(n: int, device: torch.device, dtype: torch.dtype, seed: int) -> Tensor:
    """Random orthogonal ``n x n`` matrix via QR of a Gaussian matrix."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn((n, n), generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # Make the decomposition unique (sign of diag(r)) so q is uniform on O(n).
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def _find_attention_groups(
    model: nn.Module,
) -> list[tuple[nn.Module, nn.Linear, nn.Linear, int, int, int]]:
    """
    Collect ``(parent, v_proj, o_proj, head_dim, num_kv_heads, num_q_heads)``
    triples for each transformer attention block.

    Recognized layout (Llama / Qwen / Mistral): the parent module exposes
    ``v_proj`` and ``o_proj`` as direct ``nn.Linear`` attributes, and the
    head dimension can be inferred from ``v_proj.out_features`` and the
    model config (``num_attention_heads``, ``num_key_value_heads``).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return []
    num_q_heads = getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = getattr(cfg, "head_dim", None)
    if num_q_heads is None or num_kv_heads is None:
        return []

    groups: list[tuple[nn.Module, nn.Linear, nn.Linear, int, int, int]] = []
    for parent in model.modules():
        v = getattr(parent, "v_proj", None)
        o = getattr(parent, "o_proj", None)
        if not (isinstance(v, nn.Linear) and isinstance(o, nn.Linear)):
            continue
        # Resolve head_dim from shapes if config does not provide it.
        hd = head_dim
        if hd is None:
            if v.out_features % num_kv_heads != 0:
                continue
            hd = v.out_features // num_kv_heads
        if v.out_features != num_kv_heads * hd or o.in_features != num_q_heads * hd:
            # Layout does not match the GQA convention we expect.
            continue
        groups.append((parent, v, o, hd, num_kv_heads, num_q_heads))
    return groups


@torch.no_grad()
def equalize_v_o_with_hadamard(model: nn.Module, seed: int = 0) -> int:
    """
    Apply a per-head invertible rotation between ``v_proj`` and ``o_proj`` so
    that quantization of either weight sees a more uniform per-channel
    magnitude distribution. The transformation preserves the attention output
    exactly (no calibration data is required).

    For each attention head ``h`` with head dimension ``d`` we choose an
    orthogonal matrix ``H`` of shape ``(d, d)``. With the standard
    ``nn.Linear`` convention ``y = x @ W.T``, the per-head V/O contribution is

    .. code::

        attn_h @ (X @ W_v[h_slice].T) @ W_o[:, h_slice].T

    Inserting ``H @ H.T = I`` between the two factors keeps the output
    invariant under

    .. code::

        W_v[h_slice]  <- H.T @ W_v[h_slice]      # rows of v_proj for head h
        W_o[:, h_slice] <- W_o[:, h_slice] @ H   # cols of o_proj for head h

    For Grouped-Query Attention the same KV head feeds ``num_q_heads /
    num_kv_heads`` Q heads; the corresponding ``num_q_heads / num_kv_heads``
    column slices of ``o_proj`` all receive the matching ``H``.

    A Hadamard matrix is used when ``head_dim`` is a power of two (it
    maximally spreads each weight row across the head dimension); otherwise
    the function falls back to a random orthogonal matrix.

    :param model: Model with attention modules exposing ``v_proj`` / ``o_proj``.
        Must be called on plain ``nn.Linear`` layers (i.e. **before**
        wrapping them with :class:`QuantizedLoraLinear`).
    :param seed: Seed for the random orthogonal fallback.
    :return: Number of attention layers transformed.
    """
    groups = _find_attention_groups(model)
    if not groups:
        return 0

    n_done = 0
    for layer_idx, (_, v, o, head_dim, num_kv, num_q) in enumerate(groups):
        if num_q % num_kv != 0:
            continue
        group_size = num_q // num_kv
        device = v.weight.device
        dtype = v.weight.dtype
        try:
            h_mat = _build_hadamard(head_dim, device=device, dtype=dtype)
        except ValueError:
            h_mat = _build_random_orthogonal(head_dim, device=device, dtype=dtype, seed=seed + layer_idx)
        h_t = h_mat.t().contiguous()

        # v_proj: rows for KV head h_kv occupy [h_kv*head_dim : (h_kv+1)*head_dim].
        v_w = v.weight
        for h_kv in range(num_kv):
            r0 = h_kv * head_dim
            r1 = r0 + head_dim
            v_w[r0:r1, :] = h_t @ v_w[r0:r1, :]
            if v.bias is not None:
                # Bias of v_proj is part of V; rotate the per-head slice the same way.
                v.bias[r0:r1] = h_t @ v.bias[r0:r1]

        # o_proj: each KV head feeds `group_size` consecutive Q heads; rotate
        # each Q head's column slice by the same H.
        o_w = o.weight
        for h_kv in range(num_kv):
            for q_in_group in range(group_size):
                q = h_kv * group_size + q_in_group
                c0 = q * head_dim
                c1 = c0 + head_dim
                o_w[:, c0:c1] = o_w[:, c0:c1] @ h_mat
        n_done += 1
    return n_done


# ---------------------------------------------------------------------- #
# Scale estimation calibration
# ---------------------------------------------------------------------- #
@torch.no_grad()
def run_scale_estimation(
    model: nn.Module,
    calib_inputs: list[Tensor],
    subset_size: int = 32,
    initial_steps: int = 5,
    scale_steps: int = 10,
    weight_penalty: float = -1.0,
) -> None:
    """
    Refine ``_scale_param`` of every :class:`QuantizedLoraLinear` using NNCF's
    scale-estimation procedure, ported in
    :meth:`QuantizedLoraLinear.apply_scale_estimation`.

    Activations are captured via forward pre-hooks during a single forward
    pass over ``calib_inputs``. For each layer we keep a running
    ``mean(|x|, dim=batch_seq)`` (per-channel importance) and a small subset
    of input rows used as the calibration matrix ``X``.

    :param model: Model containing :class:`QuantizedLoraLinear` modules.
    :param calib_inputs: List of input-id tensors (same shape conventions as
        the training loader).
    :param subset_size: Maximum number of input rows kept per layer for the
        per-group MSE objective.
    :param initial_steps: See :meth:`QuantizedLoraLinear.apply_scale_estimation`.
    :param scale_steps: See :meth:`QuantizedLoraLinear.apply_scale_estimation`.
    :param weight_penalty: See :meth:`QuantizedLoraLinear.apply_scale_estimation`.
    """
    layers: list[tuple[str, QuantizedLoraLinear]] = [
        (n, m) for n, m in model.named_modules() if isinstance(m, QuantizedLoraLinear)
    ]
    if not layers:
        return

    abs_sum: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    samples: dict[str, list[Tensor]] = {n: [] for n, _ in layers}
    samples_count: dict[str, int] = {n: 0 for n, _ in layers}

    handles = []
    for name, module in layers:

        def make_hook(n: str):
            def hook(_mod, args):
                x = args[0].detach()
                x_flat = x.reshape(-1, x.shape[-1]).float()
                a = x_flat.abs().sum(dim=0)
                if n in abs_sum:
                    abs_sum[n] += a
                    counts[n] += x_flat.shape[0]
                else:
                    abs_sum[n] = a
                    counts[n] = x_flat.shape[0]
                if samples_count[n] < subset_size:
                    take = min(subset_size - samples_count[n], x_flat.shape[0])
                    samples[n].append(x_flat[:take].cpu())
                    samples_count[n] += take

            return hook

        handles.append(module.register_forward_pre_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        for ids in tqdm(calib_inputs, desc="Scale estimation: collecting activations"):
            model(**get_model_input(ids))
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    for name, module in tqdm(layers, desc="Scale estimation: refining scales"):
        if name not in abs_sum or samples_count[name] == 0:
            continue
        s = abs_sum[name] / max(counts[name], 1)
        x = torch.cat(samples[name], dim=0).to(module.module.weight.device)
        module.apply_scale_estimation(
            s_per_channel=s.to(module.module.weight.device),
            x_calib=x,
            initial_steps=initial_steps,
            scale_steps=scale_steps,
            weight_penalty=weight_penalty,
        )


# ---------------------------------------------------------------------- #
# GPTQ initialization
# ---------------------------------------------------------------------- #
@torch.no_grad()
def _collect_layer_calibration(
    model: nn.Module,
    calib_inputs: list[Tensor],
    subset_size: int,
    only_num_bits: set[int] | None = None,
) -> tuple[
    list[tuple[str, QuantizedLoraLinear]],
    dict[str, Tensor],
    dict[str, list[Tensor]],
    dict[str, int],
    dict[str, int],
]:
    """
    Run a single forward pass over ``calib_inputs`` and capture, per
    :class:`QuantizedLoraLinear` layer:

    * ``abs_sum`` of the per-channel absolute activations (used as importance
      weights);
    * up to ``subset_size`` rows of the layer's input matrix (used as the
      calibration ``X`` for GPTQ / clip search).

    :param only_num_bits: If given, restrict capture to layers whose
        ``num_bits`` is in this set (saves memory).
    :return: ``(layers, abs_sum, samples, samples_count, counts)``.
    """
    layers: list[tuple[str, QuantizedLoraLinear]] = []
    for n, m in model.named_modules():
        if not isinstance(m, QuantizedLoraLinear):
            continue
        if only_num_bits is not None and m.num_bits not in only_num_bits:
            continue
        layers.append((n, m))
    if not layers:
        return [], {}, {}, {}, {}

    abs_sum: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    samples: dict[str, list[Tensor]] = {n: [] for n, _ in layers}
    samples_count: dict[str, int] = {n: 0 for n, _ in layers}

    handles = []
    for name, module in layers:

        def make_hook(n: str):
            def hook(_mod, args):
                x = args[0].detach()
                x_flat = x.reshape(-1, x.shape[-1]).float()
                a = x_flat.abs().sum(dim=0)
                if n in abs_sum:
                    abs_sum[n] += a
                    counts[n] += x_flat.shape[0]
                else:
                    abs_sum[n] = a
                    counts[n] = x_flat.shape[0]
                if samples_count[n] < subset_size:
                    take = min(subset_size - samples_count[n], x_flat.shape[0])
                    samples[n].append(x_flat[:take].cpu())
                    samples_count[n] += take

            return hook

        handles.append(module.register_forward_pre_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        for ids in tqdm(calib_inputs, desc="Calibration: collecting activations"):
            model(**get_model_input(ids))
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return layers, abs_sum, samples, samples_count, counts


# ---------------------------------------------------------------------- #
# Trainable parameter selection / persistence
# ---------------------------------------------------------------------- #
def _wrap_mixer_linears(
    model: nn.Module,
    num_bits_int4: int,
    num_bits_int2: int,
    group_size: int,
    symmetric: bool,
    lora_rank: int,
    log_scale: bool,
) -> nn.Module:
    """
    Replace ``nn.Linear`` submodules produced by ``replace_linear_with_mixer``
    with :class:`QuantizedLoraLinear`. The bit-width is selected by the
    attribute name held by the parent module:

    * ``model_int4`` -> ``num_bits_int4`` (e.g. 4)
    * ``model_int2`` -> ``num_bits_int2`` (e.g. 2)

    Linears with any other parent attribute name (notably ``lm_head``) are
    left unchanged.
    """
    bits_for_attr = {"model_int4": num_bits_int4, "model_int2": num_bits_int2}
    for parent in model.modules():
        for attr_name, child in list(parent.named_children()):
            if attr_name not in bits_for_attr:
                continue
            if not isinstance(child, nn.Linear) or isinstance(child, QuantizedLoraLinear):
                continue
            if child.in_features % group_size != 0:
                continue
            wrapped = QuantizedLoraLinear.from_linear(
                child,
                num_bits=bits_for_attr[attr_name],
                group_size=group_size,
                symmetric=symmetric,
                lora_rank=lora_rank,
                log_scale=log_scale,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(parent, attr_name, wrapped)
            print(attr_name, wrapped.module.weight.min().item(), wrapped.module.weight.max().item())
    return model


def set_trainable(
    model: nn.Module,
    lora_lr: float,
    fq_lr_int4: float,
    fq_lr_int2: float,
) -> list[dict[str, Any]]:
    """
    Freeze everything except :class:`QuantizedLoraLinear` parameters.

    Returns up to three optimizer parameter groups:

    * LoRA adapters (``lora_a``, ``lora_b``) at ``lora_lr``;
    * INT4 (and any non-INT2) per-group scales at ``fq_lr_int4``;
    * INT2 per-group scales at ``fq_lr_int2``.

    INT2 scales typically need a larger learning rate than INT4 because the
    quant grid has only 4 levels and the per-group scale is the dominant
    knob for output reconstruction.
    """
    model.requires_grad_(False)
    adapters_to_train: list[nn.Parameter] = []
    scales_int2: list[nn.Parameter] = []
    scales_int4: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, QuantizedLoraLinear):
            module._scale_param.requires_grad_(True)
            if module.num_bits <= 2:
                scales_int2.append(module._scale_param)
            else:
                scales_int4.append(module._scale_param)
            # zero_point is a non-trainable integer buffer in the asymmetric case.
            if module.lora_rank > 0:
                module.lora_a.requires_grad_(True)
                module.lora_b.requires_grad_(True)
                adapters_to_train.append(module.lora_a)
                adapters_to_train.append(module.lora_b)

    params = list(model.parameters())
    trainable = sum(p.numel() for p in params if p.requires_grad)
    total = sum(p.numel() for p in params)
    print(f"trainable params: {trainable:,d} || all params: {total:,d} || trainable%: {100 * trainable / total:.4f}")
    print(
        f"  LoRA params:    {sum(p.numel() for p in adapters_to_train):,d} @ lr={lora_lr}\n"
        f"  INT4 scales:    {sum(p.numel() for p in scales_int4):,d} @ lr={fq_lr_int4}\n"
        f"  INT2 scales:    {sum(p.numel() for p in scales_int2):,d} @ lr={fq_lr_int2}"
    )
    model.train()
    groups: list[dict[str, Any]] = []
    if adapters_to_train:
        groups.append({"params": adapters_to_train, "lr": lora_lr})
    if scales_int4:
        groups.append({"params": scales_int4, "lr": fq_lr_int4})
    if scales_int2:
        groups.append({"params": scales_int2, "lr": fq_lr_int2})
    return groups


def collect_quant_params(model: nn.Module) -> dict[str, dict[str, Tensor]]:
    """
    Snapshot the quantization parameters of every :class:`QuantizedLoraLinear`
    in ``model`` into a name-keyed dict suitable for ``torch.save``.
    """
    out: dict[str, dict[str, Tensor]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLoraLinear):
            continue
        entry: dict[str, Tensor] = {
            "scale": module.scale.detach().cpu(),
            "_scale_param": module._scale_param.detach().cpu(),
            "num_bits": torch.tensor(module.num_bits),
            "group_size": torch.tensor(module.group_size),
            "symmetric": torch.tensor(module.symmetric),
            "log_scale": torch.tensor(module.log_scale),
        }
        if module.zero_point is not None:
            entry["zero_point"] = module.zero_point.detach().cpu()
        if module.lora_rank > 0:
            entry["lora_a"] = module.lora_a.detach().cpu()
            entry["lora_b"] = module.lora_b.detach().cpu()
        out[name] = entry
    return out


def save_checkpoint(model: nn.Module, ckpt_file: Path) -> None:
    """Save the full state dict and quantization parameters to ``ckpt_file``."""
    ckpt = {
        "model_state": model.state_dict(),
        "quant_params": collect_quant_params(model),
    }
    torch.save(ckpt, ckpt_file)


# ---------------------------------------------------------------------- #
# Size estimation
# ---------------------------------------------------------------------- #
_DEFAULT_FP_BITS = 16
_DEFAULT_SCALE_BITS = 16
_EMBEDDING_BITS = 8


def estimate_quantized_size(model: nn.Module) -> dict[str, float]:
    """
    Estimate the on-disk weight size of ``model`` after quantization.

    Accounting:

    * :class:`QuantizedLoraLinear` weight: ``out * in * num_bits`` bits, plus
      per-group scale (``_DEFAULT_SCALE_BITS``) and, for the asymmetric case,
      per-group integer zero-point (``num_bits``). LoRA adapters and bias are
      counted at ``_DEFAULT_FP_BITS``.
    * ``nn.Embedding`` and the ``lm_head`` ``nn.Linear``: 8-bit per channel
      along the shortest dimension (one fp16 scale per channel of that
      dimension).
    * Tied ``lm_head`` / embedding weights are counted once.
    * All other parameters and buffers are kept at ``_DEFAULT_FP_BITS``.

    :param model: Wrapped model containing :class:`QuantizedLoraLinear` modules.
    :return: Dict with ``total_bytes``, ``total_mib`` and a per-bucket breakdown
        in bits.
    """
    seen_ids: set[int] = set()
    bits_quant_weights = 0
    bits_quant_meta = 0
    bits_lora = 0
    bits_embed = 0
    bits_other = 0
    # Element counts per effective precision (LoRA excluded: it is merged
    # into the dequantized weights at export time).
    elements_per_bits: dict[int, int] = {}

    embed_modules: list[tuple[nn.Module, str]] = []  # (module, weight_name)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            embed_modules.append((module, "weight"))

    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        embed_modules.append((lm_head, "weight"))

    # 8-bit channel-wise quantization of embeddings / lm_head.
    for emb_module, w_name in embed_modules:
        w: Tensor = getattr(emb_module, w_name)
        if id(w) in seen_ids:
            continue
        seen_ids.add(id(w))
        numel = w.numel()
        # Per-channel scale along the shortest dimension.
        n_channels = max(w.shape) if w.ndim > 0 else 1
        bits_embed += numel * _EMBEDDING_BITS + n_channels * _DEFAULT_SCALE_BITS
        elements_per_bits[_EMBEDDING_BITS] = elements_per_bits.get(_EMBEDDING_BITS, 0) + numel
        if isinstance(emb_module, nn.Linear) and emb_module.bias is not None:
            bits_embed += emb_module.bias.numel() * _DEFAULT_FP_BITS
            elements_per_bits[_DEFAULT_FP_BITS] = elements_per_bits.get(_DEFAULT_FP_BITS, 0) + emb_module.bias.numel()

    # Quantized linears.
    for module in model.modules():
        if not isinstance(module, QuantizedLoraLinear):
            continue

        w_id = id(module.module.weight)
        if w_id in seen_ids:
            continue
        seen_ids.add(w_id)
        numel = module.module.weight.numel()
        bits_quant_weights += numel * module.num_bits
        elements_per_bits[module.num_bits] = elements_per_bits.get(module.num_bits, 0) + numel
        n_groups = module.out_features * module.num_groups
        bits_quant_meta += n_groups * _DEFAULT_SCALE_BITS
        if module.zero_point is not None:
            bits_quant_meta += n_groups * module.num_bits
        if module.lora_rank > 0:
            bits_lora += (module.lora_a.numel() + module.lora_b.numel()) * _DEFAULT_FP_BITS
            # LoRA adapters intentionally not added to elements_per_bits:
            # they are folded into the weights when the model is unwrapped.
        if module.module.bias is not None:
            bits_other += module.module.bias.numel() * _DEFAULT_FP_BITS
            elements_per_bits[_DEFAULT_FP_BITS] = (
                elements_per_bits.get(_DEFAULT_FP_BITS, 0) + module.module.bias.numel()
            )

    # Everything else.
    for _, p in model.named_parameters():
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        bits_other += p.numel() * _DEFAULT_FP_BITS
        elements_per_bits[_DEFAULT_FP_BITS] = elements_per_bits.get(_DEFAULT_FP_BITS, 0) + p.numel()

    total_bits = bits_quant_weights + bits_quant_meta + bits_lora + bits_embed + bits_other
    total_bytes = total_bits / 8
    return {
        "total_bytes": total_bytes,
        "total_mib": total_bytes / (1024 * 1024),
        "quant_weight_bits": bits_quant_weights,
        "quant_meta_bits": bits_quant_meta,
        "lora_bits": bits_lora,
        "embed_bits": bits_embed,
        "other_bits": bits_other,
        "elements_per_bits": elements_per_bits,
    }


def report_quantized_size(model: nn.Module) -> None:
    """Pretty-print the output of :func:`estimate_quantized_size`."""
    info = estimate_quantized_size(model)
    print("Estimated quantized model size:")
    print(f"  total: {info['total_mib']:.2f} MiB ({info['total_bytes']:.0f} bytes)")
    print(f"  - quantized weights : {info['quant_weight_bits'] / 8 / (1024 * 1024):.2f} MiB")
    print(f"  - scales/zero-points: {info['quant_meta_bits'] / 8 / (1024 * 1024):.2f} MiB")
    print(f"  - LoRA adapters     : {info['lora_bits'] / 8 / (1024 * 1024):.2f} MiB")
    print(f"  - embed/lm_head (8b): {info['embed_bits'] / 8 / (1024 * 1024):.2f} MiB")
    print(f"  - other (fp16)      : {info['other_bits'] / 8 / (1024 * 1024):.2f} MiB")

    elements_per_bits: dict[int, int] = info["elements_per_bits"]
    total_elements = sum(elements_per_bits.values())
    print("Weights by precision (LoRA adapters excluded — they are merged into weights):")
    print(f"  {'bits':>6} | {'elements':>15} | {'share':>7} | {'size (MiB)':>12}")
    print(f"  {'-' * 6}-+-{'-' * 15}-+-{'-' * 7}-+-{'-' * 12}")
    for bits in sorted(elements_per_bits):
        n = elements_per_bits[bits]
        share = (100.0 * n / total_elements) if total_elements else 0.0
        size_mib = n * bits / 8 / (1024 * 1024)
        print(f"  {bits:>6} | {n:>15,d} | {share:>6.2f}% | {size_mib:>12.2f}")
    print(f"  {'total':>6} | {total_elements:>15,d} |  100.00% |")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model
    parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=Path, default=Path("output_quant"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--description", type=str, default=None)

    # Quantization
    parser.add_argument("--num_bits_int4", type=int, default=4, help="Bit-width for `model_int4` submodules.")
    parser.add_argument("--num_bits_int2", type=int, default=2, help="Bit-width for `model_int2` submodules.")
    parser.add_argument("--mixer_ratio", type=float, default=0.5, help="Ratio passed to replace_linear_with_mixer.")
    parser.add_argument(
        "--equalize_down_proj",
        action="store_true",
        help="Absorb per-input-channel activation magnitude of down_proj into up_proj/gate_proj before quantization.",
    )
    parser.add_argument(
        "--eq_num_calib_samples",
        type=int,
        default=128,
        help="Number of calibration samples for down_proj equalization.",
    )
    parser.add_argument(
        "--eq_calib_seqlen",
        type=int,
        default=128,
        help="Sequence length of calibration samples for down_proj equalization.",
    )
    parser.add_argument(
        "--hadamard_vo",
        action="store_true",
        help="Apply a per-head orthogonal (Hadamard when head_dim is a power of two) "
        "rotation between v_proj and o_proj. Preserves attention output exactly and "
        "spreads outliers across each head's channels prior to quantization.",
    )
    parser.add_argument(
        "--hadamard_vo_seed",
        type=int,
        default=0,
        help="Seed for the random orthogonal fallback when head_dim is not a power of two.",
    )
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--symmetric", action="store_true", help="Use symmetric quantization.")
    parser.add_argument("--log_scale", action="store_true", help="Parameterize scale as exp(log_scale).")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument(
        "--svd_lora_init",
        action="store_true",
        help="Initialize LoRA via SVD of the quant residual. "
        "Disabled by default: A=0 is already optimal for Q(W + A@B) at start.",
    )
    parser.add_argument(
        "--scale_estimation",
        action="store_true",
        help="Refine per-group scales with NNCF-style scale estimation before training.",
    )
    parser.add_argument(
        "--se_num_calib_samples", type=int, default=128, help="Number of calibration samples for scale estimation."
    )
    parser.add_argument(
        "--se_calib_seqlen", type=int, default=128, help="Sequence length of calibration samples for scale estimation."
    )
    parser.add_argument(
        "--se_subset_size", type=int, default=32, help="Per-layer rows kept for the per-group MSE objective."
    )
    parser.add_argument("--se_initial_steps", type=int, default=5)
    parser.add_argument("--se_scale_steps", type=int, default=10)
    parser.add_argument("--se_weight_penalty", type=float, default=-1.0)

    # GPTQ initialization
    parser.add_argument(
        "--gptq_init",
        action="store_true",
        help="Initialize quantizers via GPTQ/OBQ on a calibration subset before training.",
    )
    parser.add_argument(
        "--gptq_int2_only",
        action="store_true",
        help="Restrict GPTQ initialization to INT2 layers (faster, focuses budget where it helps most).",
    )
    parser.add_argument("--gptq_num_calib_samples", type=int, default=128)
    parser.add_argument("--gptq_calib_seqlen", type=int, default=512)
    parser.add_argument(
        "--gptq_subset_size", type=int, default=512, help="Per-layer rows kept for the activation Hessian."
    )
    parser.add_argument("--gptq_percdamp", type=float, default=0.01)

    # Per-group MSE clip search
    parser.add_argument(
        "--clip_search",
        action="store_true",
        help="Refine per-group scales via multiplicative clip search (AWQ-style).",
    )
    parser.add_argument("--clip_search_num_steps", type=int, default=21)
    parser.add_argument("--clip_search_min_factor", type=float, default=0.5)
    parser.add_argument("--clip_search_max_factor", type=float, default=1.0)
    parser.add_argument(
        "--clip_search_int2_only",
        action="store_true",
        help="Restrict clip search to INT2 layers.",
    )
    parser.add_argument(
        "--clip_search_use_activations",
        action="store_true",
        help="Weight the per-group MSE by per-channel activation magnitude.",
    )
    parser.add_argument("--clip_num_calib_samples", type=int, default=64)
    parser.add_argument("--clip_calib_seqlen", type=int, default=512)

    # Data
    parser.add_argument("--num_train_samples", type=int, default=512)
    parser.add_argument("--train_seqlen", type=int, default=1024)
    parser.add_argument(
        "--distill_dataset_name",
        type=str,
        default="mlfoundations-dev/DeepSeek-R1-Distill-Qwen-7B_eval_03-07-25_17-46_2870",
    )

    # Training
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Base learning rate for LoRA adapters (and default for fq scales)."
    )
    parser.add_argument(
        "--lora_lr", type=float, default=None, help="Learning rate for LoRA adapters. Defaults to --lr."
    )
    parser.add_argument(
        "--fq_lr_int4",
        type=float,
        default=None,
        help="Learning rate for INT4 (and other non-INT2) per-group scales. Defaults to --lr / 10.",
    )
    parser.add_argument(
        "--fq_lr_int2", type=float, default=None, help="Learning rate for INT2 per-group scales. Defaults to --lr / 2."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    return parser


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main(argv: list[str]) -> None:
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    transformers.set_seed(42)

    device = "cuda"
    torch_dtype = torch.bfloat16

    pprint({"CLI arguments": vars(args)})

    output_dir = Path(args.output_dir)
    suffix = args.description or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    tensorboard_dir = output_dir / "tb" / suffix
    last_dir = output_dir / (f"last_{args.description}" if args.description else "last")
    if not args.resume:
        shutil.rmtree(last_dir, ignore_errors=True)
    for path in (output_dir, tensorboard_dir, last_dir):
        path.mkdir(exist_ok=True, parents=True)
    ckpt_file = last_dir / "qat_checkpoint.pth"
    print(f"Tensorboard logs: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with QuantizedLoraLinear")

    # Load original (teacher) model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch_dtype, device_map="auto", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Build training data.
    train_loader = get_pile(args.num_train_samples, args.train_seqlen, tokenizer, device)
    if args.distill_dataset_name:
        train_loader.extend(
            get_distill_dataset(
                args.num_train_samples,
                args.train_seqlen,
                tokenizer,
                device,
                name=args.distill_dataset_name,
            )
        )

    # Cache teacher hiddens for distillation.
    model_id = args.pretrained.split("/")[-1]
    hiddens_pth = last_dir / f"orig_hiddens_{model_id}_{args.num_train_samples}_{args.train_seqlen}.pt"
    if hiddens_pth.exists():
        orig_hiddens = torch.load(hiddens_pth)
    else:
        orig_hiddens = calc_hiddens(model, train_loader)
        torch.save(orig_hiddens, hiddens_pth)

    # Keep a frozen copy of the teacher's lm_head for logits distillation.
    gpu_id, best_memory = None, float("inf")
    for i in range(torch.cuda.device_count()):
        if torch.cuda.memory_allocated(i) < best_memory:
            best_memory = torch.cuda.memory_allocated(i)
            gpu_id = i
    teacher_lm_head = copy.deepcopy(model.lm_head).eval().requires_grad_(False)
    if gpu_id is not None:
        teacher_lm_head = teacher_lm_head.to(f"cuda:{gpu_id}")
    if hasattr(teacher_lm_head, "_old_forward"):
        teacher_lm_head.forward = teacher_lm_head._old_forward

    print(f"Answer (FP teacher): {generate_answer(model, tokenizer)}\n")

    # Optional pre-quantization equalization of down_proj input scale into
    # the producing up_proj / gate_proj output channels.
    if args.equalize_down_proj:
        answer_before_equalization = generate_answer(model, tokenizer)
        eq_loader = get_pile(
            num_samples=args.eq_num_calib_samples,
            seqlen=args.eq_calib_seqlen,
            tokenizer=tokenizer,
            device=device,
        )
        n_eq = equalize_down_proj(model, eq_loader)
        print(f"Equalized {n_eq} down_proj layers.")
        print(f"Answer before equalization: {answer_before_equalization}")
        print(f"Answer (post equalization):  {generate_answer(model, tokenizer)}\n")

        n_eq = equalize_up_gate_with_layernorm(model)
        print(f"Equalized {n_eq} up_proj/gate_proj layers with layernorm.")
        print(f"Answer before equalization: {answer_before_equalization}")
        print(f"Answer (post equalization):  {generate_answer(model, tokenizer)}\n")

    # Optional per-head Hadamard / orthogonal rotation between v_proj and o_proj.
    # Output-invariant; intended to flatten per-channel weight magnitudes seen by
    # the per-group quantizer of v_proj and o_proj.
    if args.hadamard_vo:
        answer_before_hadamard = generate_answer(model, tokenizer)
        n_rot = equalize_v_o_with_hadamard(model, seed=args.hadamard_vo_seed)
        print(f"Applied Hadamard rotation to {n_rot} attention layers (v_proj/o_proj).")
        print(f"Answer before Hadamard:    {answer_before_hadamard}")
        print(f"Answer (post Hadamard):    {generate_answer(model, tokenizer)}\n")

    # Split each Linear into LinearMIXER (and LinearINT4 for the few protected
    # layers); the resulting submodules are named ``model_int4`` / ``model_int2``
    # and drive the per-layer bit-width assignment below.
    answer_before_mixer = generate_answer(model, tokenizer)
    model = replace_linear_with_mixer(model, ratio=args.mixer_ratio)
    answer_after_mixer = generate_answer(model, tokenizer)
    print(f"Answer before mixer: {answer_before_mixer}")
    print(f"Answer after mixer:  {answer_after_mixer}\n")

    # Wrap linear layers with QuantizedLoraLinear: 4-bit for ``model_int4``
    # children, 2-bit for ``model_int2`` children, others left untouched.
    _wrap_mixer_linears(
        model,
        num_bits_int4=args.num_bits_int4,
        num_bits_int2=args.num_bits_int2,
        group_size=args.group_size,
        symmetric=args.symmetric,
        lora_rank=args.lora_rank,
        log_scale=args.log_scale,
    )

    # Optional NNCF-style scale estimation on a small calibration subset.
    if args.scale_estimation and False:
        calib_loader = get_pile(
            num_samples=args.se_num_calib_samples,
            seqlen=args.se_calib_seqlen,
            tokenizer=tokenizer,
            device=device,
        )
        run_scale_estimation(
            model,
            calib_inputs=calib_loader,
            subset_size=args.se_subset_size,
            initial_steps=args.se_initial_steps,
            scale_steps=args.se_scale_steps,
            weight_penalty=args.se_weight_penalty,
        )
        print(f"Answer (post scale-estimation): {generate_answer(model, tokenizer)}\n")

    report_quantized_size(model)

    # Optionally resume from checkpoint.
    if args.resume and ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    print(f"Answer (post-init quant): {generate_answer(model, tokenizer)}\n")

    lora_lr = args.lora_lr if args.lora_lr is not None else args.lr
    fq_lr_int4 = args.fq_lr_int4 if args.fq_lr_int4 is not None else args.lr / 10
    fq_lr_int2 = args.fq_lr_int2 if args.fq_lr_int2 is not None else args.lr / 2
    weight_decay = args.lr
    param_groups = set_trainable(
        model,
        lora_lr=lora_lr,
        fq_lr_int4=fq_lr_int4,
        fq_lr_int2=fq_lr_int2,
    )
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    optimizer_steps_per_epoch = max(1, microbatches_per_epoch // grad_accumulation_steps)
    total_optimizer_steps = max(1, args.epochs * optimizer_steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_ratio * total_optimizer_steps)) if args.warmup_ratio > 0 else 0
    scheduler = transformers.get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps
    )

    loss_numerator = grad_steps = total_steps = 0
    aggregated_kl_loss = 0.0
    aggregated_l1_loss = 0.0

    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        progress = tqdm(
            batch_indices_epoch,
            total=len(batch_indices_epoch),
            desc=f"Train epoch {epoch}",
            dynamic_ncols=True,
            leave=True,
        )
        for indices in progress:
            indices_list = indices.tolist()

            def form_batch(inputs: list[Tensor], model_input: bool):
                batch = torch.cat([inputs[i] for i in indices_list], dim=0)
                return get_model_input(batch) if model_input else batch.to(device=device, dtype=torch_dtype)

            inputs = form_batch(train_loader, model_input=True)
            with torch.no_grad():
                cur_teacher_hiddens = form_batch(orig_hiddens, model_input=False)
                cur_teacher_hiddens = cur_teacher_hiddens.to(device=teacher_lm_head.weight.device)
                targets = teacher_lm_head(cur_teacher_hiddens)
                fls = getattr(model.config, "final_logit_softcapping", None)
                if fls is not None:
                    targets = torch.tanh(targets / fls) * fls

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                cur_student_hiddens = outputs.hidden_states[-1]

            seq_start = cur_student_hiddens.shape[1] // 3
            kl_loss = kl_div(
                logits[:, seq_start:],
                targets[:, seq_start:].to(dtype=torch_dtype, device=device),
            )
            l1_loss = F.l1_loss(
                cur_student_hiddens[:, seq_start:],
                cur_teacher_hiddens[:, seq_start:].to(dtype=torch_dtype, device=device),
            )
            loss = kl_loss

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")

            loss_numerator += loss.item()
            grad_steps += 1
            (loss / grad_accumulation_steps).backward()

            aggregated_kl_loss += kl_loss.item()
            aggregated_l1_loss += l1_loss.item()

            epoch_loss_sum += loss.item()
            epoch_loss_count += 1
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "mean_loss": f"{epoch_loss_sum / epoch_loss_count:.4f}",
                }
            )

            if grad_steps == grad_accumulation_steps:
                opt.step()
                scheduler.step()
                opt.zero_grad()
                total_steps += 1
                tb.add_scalar("loss", loss_numerator / grad_steps, total_steps)
                tb.add_scalar("kl_loss", aggregated_kl_loss / grad_steps, total_steps)
                tb.add_scalar("l1_loss", aggregated_l1_loss / grad_steps, total_steps)
                for i, pg in enumerate(opt.param_groups):
                    tb.add_scalar(f"lr/group_{i}", pg["lr"], total_steps)
                loss_numerator = grad_steps = 0
                aggregated_kl_loss = aggregated_l1_loss = 0.0

        save_checkpoint(model, ckpt_file)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                print(f"Answer after epoch {epoch}: {generate_answer(model, tokenizer)}\n")

    # ------------------------------------------------------------------ #
    # Save dequantized model + quantization parameters.
    # ------------------------------------------------------------------ #
    quant_params = collect_quant_params(model)
    torch.save(quant_params, last_dir / "quant_params.pt")
    print(f"Quantization parameters saved to: {last_dir / 'quant_params.pt'}")

    model.eval()
    unwrap_linear_layers(model)
    replace_mixer_with_linear(model)

    dequant_dir = last_dir / "dequantized_model"
    dequant_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dequant_dir)
    tokenizer.save_pretrained(dequant_dir)
    print(f"Dequantized model saved to: {dequant_dir}")

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            print(f"Answer after epoch {epoch}: {generate_answer(model, tokenizer)}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
