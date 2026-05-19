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
Trainable per-input-channel grouped quantization wrapper for ``torch.nn.Linear``
with LoRA adapters applied *before* quantization.

Quantization formulas
---------------------
Let ``W`` be the original weight of shape ``[out_features, in_features]``,
``A`` (``lora_a``) of shape ``[out_features, r]`` and
``B`` (``lora_b``) of shape ``[r, in_features]`` so that ``A @ B`` has the
same shape as ``W``. Grouping is performed along the input-channel dimension
with group size ``g``.

Asymmetric (``n`` bits, levels ``[0, 2**n - 1]``)::

    W' = W + A @ B
    Q  = clamp(round(W' / scale + zero_point), 0, 2**n - 1)
    W_dq = (Q - zero_point) * scale

Symmetric (``n`` bits, levels ``[-2**(n-1), 2**(n-1) - 1]``)::

    W' = W + A @ B
    Q  = clamp(round(W' / scale), -2**(n-1), 2**(n-1) - 1)
    W_dq = Q * scale

Straight-through estimator is used for ``round`` and ``clamp`` so that
gradients flow into ``scale``, ``zero_point``, ``A`` and ``B``.

If ``log_scale=True`` the underlying trainable parameter stores
``log(scale)`` and the actual scale is recovered with ``exp``. This keeps
``scale`` strictly positive during optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _RoundSTE(torch.autograd.Function):
    """``round`` with a straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:  # type: ignore[override]
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        return grad_output


class ClippedSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save the input for the backward pass mask
        ctx.save_for_backward(x)
        # Hard clip between -1 and 1
        return torch.clamp(x, min=-1.0, max=1.0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Gradient is passed through only where -1 <= x <= 1
        mask = (x >= -1.0) & (x <= 1.0)
        return grad_output * mask.float()


def _round_ste(x: Tensor) -> Tensor:
    return _RoundSTE.apply(x)


def _clamp_ste(x: Tensor, lo: float, hi: float) -> Tensor:
    """``clamp`` with straight-through gradients on the saturated region."""
    return x + (x.clamp(lo, hi) - x).detach()


class QuantizedLoraLinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` that performs trainable
    fake-quantization of the weight with optional LoRA correction applied
    before quantization.

    :param in_features: Number of input features.
    :param out_features: Number of output features.
    :param num_bits: Quantization bit-width.
    :param group_size: Group size along the input-channel dimension.
        Use ``-1`` for per-output-channel quantization (i.e. a single group
        spanning all input channels).
    :param symmetric: If ``True``, use symmetric quantization, otherwise
        asymmetric.
    :param lora_rank: Rank of the LoRA adapters. ``0`` disables LoRA.
    :param log_scale: If ``True``, parameterize ``scale`` as ``exp(log_scale)``.
    :param bias: Whether the underlying linear layer has a bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bits: int = 4,
        group_size: int = 128,
        symmetric: bool = False,
        lora_rank: int = 0,
        log_scale: bool = False,
        module: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        if group_size == -1:
            group_size = in_features
        if in_features % group_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size}).")
        if num_bits < 2:
            raise ValueError(f"num_bits must be >= 2, got {num_bits}.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.symmetric = symmetric
        self.lora_rank = lora_rank
        self.log_scale = log_scale

        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**num_bits - 1

        # Frozen base weight; trainable correction comes from LoRA.
        self.module = module
        dtype = module.weight.dtype if module is not None else torch.bfloat16

        scale_shape = (out_features, self.num_groups)
        scale_init = torch.ones(scale_shape, dtype=dtype)
        if log_scale:
            # store log(scale); init scale = 1 -> log_scale = 0
            self._scale_param = nn.Parameter(torch.zeros(scale_shape, dtype=dtype))
        else:
            self._scale_param = nn.Parameter(scale_init)

        if not symmetric:
            # Non-trainable integer zero point in [0, 2**num_bits - 1].
            zp_init = torch.full(scale_shape, 2 ** (num_bits - 1), dtype=torch.int32)
            self.register_buffer("zero_point", zp_init)
        else:
            self.zero_point: Tensor | None = None

        if lora_rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(out_features, lora_rank, dtype=dtype))
            self.lora_b = nn.Parameter(torch.empty(lora_rank, in_features, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_b, a=5**0.5)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_bits: int = 4,
        group_size: int = 128,
        symmetric: bool = False,
        lora_rank: int = 0,
        log_scale: bool = False,
    ) -> QuantizedLoraLinear:
        """
        Build a ``QuantizedLoraLinear`` from an existing ``nn.Linear``.
        ``scale`` (and ``zero_point`` for asymmetric mode) are initialized
        from the weight statistics of ``linear``.

        :param linear: Source linear layer. Its weight is copied into a
            non-trainable buffer.
        :return: New ``QuantizedLoraLinear`` instance.
        """
        wrapper = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            num_bits=num_bits,
            group_size=group_size,
            symmetric=symmetric,
            lora_rank=lora_rank,
            log_scale=log_scale,
            module=linear,
        )
        with torch.no_grad():
            wrapper._init_qparams_from_weight()
        return wrapper

    @torch.no_grad()
    def _init_qparams_from_weight(self) -> None:
        """Initialize scale (and zero point) from per-group weight statistics."""
        w = self.module.weight.clone()
        w_grouped = w.reshape(self.out_features, self.num_groups, self.group_size)
        if self.symmetric:
            absmax = w_grouped.abs().amax(dim=-1).clamp_min(1e-8)
            scale = absmax / max(abs(self.qmin), abs(self.qmax))
        else:
            wmin = w_grouped.amin(dim=-1)
            wmax = w_grouped.amax(dim=-1)
            scale = ((wmax - wmin) / (self.qmax - self.qmin)).clamp_min(1e-8)
            zp = (self.qmin - wmin / scale).round().clamp(self.qmin, self.qmax).to(torch.int32)
            self.zero_point.copy_(zp)
        if self.log_scale:
            self._scale_param.copy_(torch.log(scale))
        else:
            self._scale_param.copy_(scale)

        # apply scale for weight to avoid this in forward pass and apply regularization on LoRA to range (-1, 1)
        w_grouped.div_(scale.unsqueeze(-1))
        self.module.weight.data = w_grouped.reshape(self.out_features, self.in_features).data

    @torch.no_grad()
    def rescale_weight(self) -> Tensor:
        """Rescale the weight by the current scale. Useful before clip search."""
        w = self.module.weight.clone()
        w_grouped = w.reshape(self.out_features, self.num_groups, self.group_size)
        w_grouped.mul_(self.scale.unsqueeze(-1))
        return w_grouped.reshape(self.out_features, self.in_features)

    # ------------------------------------------------------------------ #
    # Quantization
    # ------------------------------------------------------------------ #
    @property
    def scale(self) -> Tensor:
        """Effective positive scale tensor."""
        if self.log_scale:
            return torch.exp(self._scale_param)
        return self._scale_param

    def _effective_weight(self) -> Tensor:
        """Apply LoRA correction before quantization."""
        w = self.module.weight
        if self.lora_rank > 0:
            lora = self.lora_a @ self.lora_b
            # restrict LoRA values to range (-1, 1) to avoid instability during quantization and large updates
            # lora = torch.tanh(lora)
            lora = 0.5 * ClippedSTE.apply(lora)
            w = w + lora
        return w

    def quantize_dequantize(self) -> Tensor:
        """Return the fake-quantized weight (with LoRA applied)."""
        w = self._effective_weight()
        w_g = w.reshape(self.out_features, self.num_groups, self.group_size)
        scale = self.scale.unsqueeze(-1)

        if self.symmetric:
            q = _round_ste(w_g)
            q = _clamp_ste(q, self.qmin, self.qmax)
            w_dq = q * scale
        else:
            zp = self.zero_point.to(scale.dtype).unsqueeze(-1)
            q = _round_ste(w_g + zp)
            q = _clamp_ste(q, self.qmin, self.qmax)
            w_dq = (q - zp) * scale

        return w_dq.reshape(self.out_features, self.in_features).to(w.dtype)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.quantize_dequantize(), self.module.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_bits={self.num_bits}, group_size={self.group_size}, "
            f"symmetric={self.symmetric}, lora_rank={self.lora_rank}, "
            f"log_scale={self.log_scale}, bias={self.bias is not None}"
        )

    # ------------------------------------------------------------------ #
    # Scale estimation (port of NNCF ScaleEstimation.calculate_quantization_params)
    # ------------------------------------------------------------------ #
    def _quantize_unscaled(self, scale: Tensor) -> tuple[Tensor, Tensor]:
        """
        Quantize the *base* weight (no LoRA) with a given scale tensor.

        :param scale: ``[out_features, num_groups]`` positive scale tensor.
        :return: Tuple ``(w_dq, target)`` of grouped tensors of shape
            ``[out_features, num_groups, group_size]``. ``target`` is the
            "centered" integer code (``q`` for symmetric, ``q - zp`` for
            asymmetric), matching NNCF's ``get_target_zero_mask`` semantics.
        """
        w_g = self.module.weight.float().reshape(self.out_features, self.num_groups, self.group_size)
        s = scale.unsqueeze(-1)
        if self.symmetric:
            q = torch.round(w_g).clamp(self.qmin, self.qmax)
            target = q
            w_dq = q * s
        else:
            zp = self.zero_point.to(s.dtype).unsqueeze(-1)
            q = torch.round(w_g + zp).clamp(self.qmin, self.qmax)
            target = q - zp
            w_dq = target * s
        return w_dq, target

    @torch.no_grad()
    def apply_scale_estimation(
        self,
        s_per_channel: Tensor,
        x_calib: Tensor,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
        zero_scale: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        """
        Refine the per-group scale by minimizing the per-group MSE between the
        FP and quantized linear outputs on a small calibration subset, using
        the same iterative procedure as NNCF's ``ScaleEstimation``.

        :param s_per_channel: ``[in_features]`` tensor of per-input-channel
            activation importance (e.g. ``mean(|X|, dim=batch_seq)``).
        :param x_calib: ``[N, in_features]`` calibration input matrix.
        :param initial_steps: Number of iterative ideal-scale rectification
            steps.
        :param scale_steps: Number of grid-search refinement steps over
            ``factor = 1 - 0.05 * step``.
        :param weight_penalty: Weight-MSE penalty added to the output-MSE
            objective. ``< 0`` disables it.
        :param zero_scale: Small positive value substituted for the
            zero-target mask to avoid division-by-zero in the ideal-scale
            estimator.
        :param eps: Numerical stabilizer.
        """
        device = self.module.weight.device
        out_f, in_f = self.out_features, self.in_features
        n_g, g = self.num_groups, self.group_size

        s = s_per_channel.detach().to(device=device, dtype=torch.float32).reshape(1, n_g, g)
        x = x_calib.detach().to(device=device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != in_f:
            raise ValueError(f"x_calib must have shape [N, {in_f}], got {tuple(x.shape)}.")
        x_g = x.reshape(x.shape[0], n_g, g)

        w_full = self.rescale_weight().float()  # self.module.weight.float()

        w_g = w_full.reshape(out_f, n_g, g)
        # FP per-group output contributions: [N, out_f, n_g]
        fp_out_g = torch.einsum("ngk,ogk->nog", x_g, w_g)

        def diffs_for(scale: Tensor) -> tuple[Tensor, Tensor]:
            """Return (per_group_mse[out_f, n_g], target[out_f, n_g, g])."""
            w_dq, target = self._quantize_unscaled(scale)
            q_out_g = torch.einsum("ngk,ogk->nog", x_g, w_dq)
            mse = ((fp_out_g - q_out_g) ** 2).mean(dim=0)  # [out_f, n_g]
            if weight_penalty > 0.0:
                mse = mse + weight_penalty * ((w_dq - w_g) ** 2).mean(dim=-1)
            return mse, target

        # Baseline: current scale.
        scale = self.scale.detach().float().clone()  # [out_f, n_g]
        scale_sign = torch.sign(scale).clamp_min(1e-12)  # keep zero-scales positive
        result_scale = scale.clone()
        best_diffs, target = diffs_for(scale)

        # Per-element importance broadcast from activation stats.
        importance_template = s.expand(out_f, n_g, g).clone()  # [out_f, n_g, g]

        def estimate_ideal(target_t: Tensor) -> Tensor:
            zero_mask = target_t.abs() < eps
            zero_mask_f = zero_mask.to(w_g.dtype) * zero_scale
            importance = torch.where(zero_mask, torch.zeros_like(importance_template), importance_template)
            denom = importance.sum(dim=-1, keepdim=True)
            importance = importance / (denom + eps)
            ideal = w_g.abs() / (target_t.abs() + zero_mask_f)
            return (ideal * importance).sum(dim=-1)  # [out_f, n_g]

        # 1) Iterative ideal-scale rectification.
        for step in range(initial_steps):
            ideal = estimate_ideal(target) * scale_sign
            cand_diffs, cand_target = diffs_for(ideal)
            improved = cand_diffs < best_diffs
            best_diffs = torch.where(improved, cand_diffs, best_diffs)
            result_scale = torch.where(improved, ideal, result_scale)
            if step < initial_steps - 1:
                _, target = self._quantize_unscaled(result_scale)

        # 2) Grid search around the original scale.
        for step in range(scale_steps):
            factor = 1.0 - 0.05 * step
            scaled = factor * scale
            _, target_scaled = self._quantize_unscaled(scaled)
            ideal = estimate_ideal(target_scaled) * scale_sign
            cand_diffs, _ = diffs_for(ideal)
            improved = cand_diffs < best_diffs
            best_diffs = torch.where(improved, cand_diffs, best_diffs)
            result_scale = torch.where(improved, ideal, result_scale)

        result_scale = result_scale.clamp_min(eps).to(self._scale_param.dtype)
        if self.log_scale:
            self._scale_param.copy_(torch.log(result_scale))
        else:
            self._scale_param.copy_(result_scale)


# ---------------------------------------------------------------------- #
# Wrap / unwrap utilities
# ---------------------------------------------------------------------- #
def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parent = root
    parts = qualified_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def wrap_linear_layers(
    model: nn.Module,
    num_bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    lora_rank: int = 0,
    log_scale: bool = False,
    skip_name_substrings: list[str] | None = None,
) -> nn.Module:
    """
    Replace every ``nn.Linear`` inside ``model`` with a
    :class:`QuantizedLoraLinear` of matching shape.

    :param model: The model to modify in place (typically an
        ``AutoModelForCausalLM``).
    :param num_bits: Bit-width passed to each wrapper.
    :param group_size: Group size along the input-channel dimension.
    :param symmetric: Use symmetric quantization if ``True``.
    :param lora_rank: LoRA rank (``0`` disables LoRA).
    :param log_scale: Parameterize the scale in log space if ``True``.
    :param skip_name_substrings: List of substrings; any module whose
        qualified name contains one of them is left untouched. Defaults to
        ``["lm_head"]``.
    :return: The same ``model`` instance, mutated in place.
    """
    if skip_name_substrings is None:
        skip_name_substrings = ["lm_head"]

    target_names: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, QuantizedLoraLinear):
            if any(s in name for s in skip_name_substrings):
                continue
            if module.in_features % (group_size if group_size != -1 else module.in_features) != 0:
                continue
            target_names.append((name, module))

    for name, linear in target_names:
        wrapped = QuantizedLoraLinear.from_linear(
            linear,
            num_bits=num_bits,
            group_size=group_size,
            symmetric=symmetric,
            lora_rank=lora_rank,
            log_scale=log_scale,
        )
        wrapped = wrapped.to(device=linear.weight.device, dtype=linear.weight.dtype)
        _set_submodule(model, name, wrapped)
    return model


def unwrap_linear_layers(model: nn.Module) -> nn.Module:
    """
    Replace every :class:`QuantizedLoraLinear` in ``model`` with a plain
    ``nn.Linear`` whose weight is the dequantized weight (LoRA folded in).

    :param model: Model produced by :func:`wrap_linear_layers`. Modified
        in place.
    :return: The same ``model`` instance with wrappers removed.
    """
    targets: list[tuple[str, QuantizedLoraLinear]] = [
        (name, m) for name, m in model.named_modules() if isinstance(m, QuantizedLoraLinear)
    ]
    for name, qmod in targets:
        with torch.no_grad():
            w_dq = qmod.quantize_dequantize().detach()
        linear = nn.Linear(
            qmod.in_features,
            qmod.out_features,
            bias=qmod.module.bias is not None,
        ).to(device=w_dq.device, dtype=w_dq.dtype)
        with torch.no_grad():
            linear.weight.copy_(w_dq)
            if qmod.module.bias is not None:
                linear.bias.copy_(qmod.module.bias.detach())
        _set_submodule(model, name, linear)
    return model
