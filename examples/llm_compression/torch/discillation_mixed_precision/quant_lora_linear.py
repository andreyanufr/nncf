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



import torch
import torch.nn as nn
import math

class LSQQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scale, q_min, q_max, grad_scale):
        # Save variables for the backward pass
        ctx.save_for_backward(tensor, scale)
        ctx.q_min = q_min
        ctx.q_max = q_max
        ctx.grad_scale = grad_scale

        # 1. Scale the input tensor
        scaled_tensor = tensor / scale
        
        # 2. Quantize (Round & Clip)
        quantized = torch.clamp(torch.round(scaled_tensor), q_min, q_max)
        
        # 3. Dequantize
        dequantized = quantized * scale
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        tensor, scale = ctx.saved_tensors
        q_min = ctx.q_min
        q_max = ctx.q_max
        grad_scale = ctx.grad_scale

        scaled_tensor = tensor / scale
        
        # Determine clipping regions
        lower_mask = (scaled_tensor < q_min).float()
        upper_mask = (scaled_tensor > q_max).float()
        middle_mask = 1.0 - lower_mask - upper_mask

        # --- Gradient with respect to the Input Tensor (Standard STE) ---
        grad_tensor = grad_output * middle_mask

        # --- Gradient with respect to the Scale Factor (s) ---
        # Look at how the masks alter the derivative based on clipping:
        # Inside bounds: it evaluates to (round(x/s) - x/s) which is the quantization error.
        # Outside bounds: it evaluates to either q_min or q_max.
        quantized_or_clipped = torch.where(scaled_tensor < q_min, float(q_min), scaled_tensor)
        quantized_or_clipped = torch.where(scaled_tensor > q_max, float(q_max), quantized_or_clipped)
        
        # LSQ formulation accounts for the exact rounding error inside the grid
        round_error = torch.round(scaled_tensor) - scaled_tensor
        derivative_s = torch.where(middle_mask.bool(), round_error, quantized_or_clipped)
        
        # Sum across the tensor elements and apply the LSQ gradient scaling factor (grad_scale)
        grad_scale_param = torch.sum(grad_output * derivative_s) * grad_scale

        # Return gradients matching the order of forward() arguments
        # (tensor, scale, q_min, q_max, grad_scale)
        return grad_tensor, grad_scale_param, None, None, None



class LSQQuantizerFunctionWithLoRA(torch.autograd.Function):
    """
    LSQ fake-quantization of a 2D weight ``[out_features, in_features]`` with a
    LoRA correction (``A @ B``, clipped to ``[-1, 1]``) added in *scaled* space,
    using per-input-channel grouped scales.

    Shapes:
        tensor : [O, I]
        lora_a : [O, r]
        lora_b : [r, I]
        scale  : [O, G]  where G = I // group_size
    """

    @staticmethod
    def forward(ctx, tensor, lora_a, lora_b, scale, q_min, q_max, grad_scale, group_size):
        out_f, in_f = tensor.shape
        if group_size <= 0 or in_f % group_size != 0:
            raise ValueError(
                f"in_features ({in_f}) must be a positive multiple of group_size ({group_size})."
            )
        num_groups = in_f // group_size
        if scale.shape != (out_f, num_groups):
            raise ValueError(
                f"scale shape {tuple(scale.shape)} does not match expected ({out_f}, {num_groups})."
            )

        # LoRA correction, clipped to [-1, 1].
        lora = lora_a @ lora_b  # [O, I]
        clip_mask = (lora >= -1.0) & (lora <= 1.0)
        lora_c = torch.clamp(lora, -1.0, 1.0)

        # Group along the input-channel dimension.
        w_g = tensor.reshape(out_f, num_groups, group_size)
        l_g = lora_c.reshape(out_f, num_groups, group_size)
        s_g = scale.unsqueeze(-1)  # [O, G, 1]

        v = w_g / s_g
        y = v + l_g
        q = torch.clamp(torch.round(y), q_min, q_max)
        dequantized = (q * s_g).reshape(out_f, in_f)

        ctx.save_for_backward(tensor, scale, lora_a, lora_b, clip_mask)
        ctx.q_min = q_min
        ctx.q_max = q_max
        ctx.grad_scale = grad_scale
        ctx.group_size = group_size
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        tensor, scale, lora_a, lora_b, clip_mask = ctx.saved_tensors
        q_min = ctx.q_min
        q_max = ctx.q_max
        grad_scale = ctx.grad_scale
        group_size = ctx.group_size

        out_f, in_f = tensor.shape
        num_groups = in_f // group_size

        # Recompute forward intermediates (cheap; avoids saving large activations).
        lora = lora_a @ lora_b
        lora_c = torch.clamp(lora, -1.0, 1.0)
        w_g = tensor.reshape(out_f, num_groups, group_size)
        l_g = lora_c.reshape(out_f, num_groups, group_size)
        s_g = scale.unsqueeze(-1)
        v = w_g / s_g
        y = v + l_g

        lower = y < q_min
        upper = y > q_max
        middle = ~(lower | upper)
        middle_f = middle.to(grad_output.dtype)

        grad_out_g = grad_output.reshape(out_f, num_groups, group_size)

        # --- Gradient w.r.t. the input tensor (STE through round+clamp) ---
        # d(out)/d(tensor) inside = 1, outside = 0.
        #grad_tensor = (grad_out_g * middle_f).reshape(out_f, in_f)

        # --- Gradient w.r.t. LoRA factors ---
        # d(out)/d(L) inside = scale, outside = 0; then masked by the LoRA clip region.
        grad_lora_eff_g = grad_out_g * middle_f * s_g
        grad_lora_eff = grad_lora_eff_g.reshape(out_f, in_f)
        grad_lora_eff = grad_lora_eff * clip_mask.to(grad_lora_eff.dtype)
        # L = A @ B  =>  dA = dL @ B^T, dB = A^T @ dL
        grad_lora_a = grad_lora_eff @ lora_b.transpose(-1, -2)
        grad_lora_b = lora_a.transpose(-1, -2) @ grad_lora_eff

        # --- Gradient w.r.t. scale (LSQ) ---
        # Inside bounds: derivative = round(y) - v  (quantization residual w.r.t. unscaled weight).
        # Outside bounds: derivative = q_min or q_max.
        q_clipped = torch.where(lower, torch.full_like(y, float(q_min)), y)
        q_clipped = torch.where(upper, torch.full_like(y, float(q_max)), q_clipped)
        round_err = torch.round(y) - v
        derivative_s = torch.where(middle, round_err, q_clipped)
        # Sum across group elements -> per-group scale gradient [O, G].
        grad_scale_param = (grad_out_g * derivative_s).sum(dim=-1) * grad_scale

        # Order matches forward(): (tensor, lora_a, lora_b, scale, q_min, q_max, grad_scale, group_size)
        return None, grad_lora_a, grad_lora_b, grad_scale_param, None, None, None, None


class LSQQuantizer(nn.Module):
    def __init__(self, bits=4, signed=True, per_channel=False, channels=1):
        super().__init__()
        self.bits = bits
        self.signed = signed
        self.per_channel = per_channel
        
        if signed:
            self.q_min = -(2 ** (bits - 1))
            self.q_max = (2 ** (bits - 1)) - 1
        else:
            self.q_min = 0
            self.q_max = (2 ** bits) - 1

        # Scale parameter initialization
        # We initialize it as a regular Parameter so PyTorch tracks it
        if per_channel:
            self.scale = nn.Parameter(torch.ones(channels, 1))
        else:
            self.scale = nn.Parameter(torch.tensor(1.0))
            
        self.initialized = False

    def init_scale(self, tensor):
        """ LSQ initializes the scale as: 2 * E[|v|] / sqrt(Q_max) """
        with torch.no_grad():
            if self.per_channel:
                # Expecting a weight tensor of shape (out_features, in_features)
                mean_abs = tensor.abs().mean(dim=1, keepdim=True)
            else:
                mean_abs = tensor.abs().mean()
            
            init_val = 2.0 * mean_abs / math.sqrt(self.q_max)
            self.scale.copy_(init_val)
            self.initialized = True

    def forward(self, tensor):
        if not self.initialized and self.training:
            self.init_scale(tensor)

        # Calculate the LSQ gradient scaling factor (gamma)
        num_elements = tensor.numel() if not self.per_channel else tensor.shape[1]
        grad_scale = 1.0 / math.sqrt(num_elements * self.q_max)

        return LSQQuantizerFunction.apply(
            tensor, self.scale, self.q_min, self.q_max, grad_scale
        )


class LSQLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_bits=4, act_bits=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Weights are naturally signed, per-channel quantization is industry standard for LLMs
        self.weight_quantizer = LSQQuantizer(bits=weight_bits, signed=True, per_channel=True, channels=out_features)
        
        # Activations can be unsigned if following a ReLU, signed for GELU/SiLU/linear outputs
        self.act_quantizer = LSQQuantizer(bits=act_bits, signed=True, per_channel=False)

    def forward(self, x):
        # 1. Quantize inputs/activations arriving at the layer
        quant_x = self.act_quantizer(x)
        
        # 2. Quantize weights per-channel
        quant_w = self.weight_quantizer(self.linear.weight)
        
        # 3. Perform linear projection with quantized elements
        return nn.functional.linear(quant_x, quant_w, self.linear.bias)


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
        self.forward_orig = False

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
            lora = 2 * ClippedSTE.apply(lora)
            return w, lora
        return w, 0

    def quantize_dequantize(self) -> Tensor:
        """Return the fake-quantized weight (with LoRA applied)."""
        w, lora = self._effective_weight()
        w_g = w.reshape(self.out_features, self.num_groups, self.group_size)
        scale = self.scale.unsqueeze(-1)
        w_g = w_g / scale

        if self.lora_rank > 0:
            lora_g = lora.reshape(self.out_features, self.num_groups, self.group_size)
            w_g = w_g + lora_g

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
        if self.forward_orig:
            return F.linear(x, self.module.weight, self.module.bias)
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
        w_g = w_g / s

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

        w_full = self.module.weight.float()

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
