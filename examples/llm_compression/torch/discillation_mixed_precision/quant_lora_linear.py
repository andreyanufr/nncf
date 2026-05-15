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

from typing import Optional

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
        bias: bool = True,
    ) -> None:
        super().__init__()
        if group_size == -1:
            group_size = in_features
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})."
            )
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
        self.register_buffer("weight", torch.empty(out_features, in_features))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        scale_shape = (out_features, self.num_groups)
        scale_init = torch.ones(scale_shape)
        if log_scale:
            # store log(scale); init scale = 1 -> log_scale = 0
            self._scale_param = nn.Parameter(torch.zeros(scale_shape))
        else:
            self._scale_param = nn.Parameter(scale_init)

        if not symmetric:
            # Non-trainable integer zero point in [0, 2**num_bits - 1].
            zp_init = torch.full(scale_shape, 2 ** (num_bits - 1), dtype=torch.int32)
            self.register_buffer("zero_point", zp_init)
        else:
            self.zero_point: Optional[Tensor] = None

        if lora_rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(out_features, lora_rank))
            self.lora_b = nn.Parameter(torch.empty(lora_rank, in_features))
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
    ) -> "QuantizedLoraLinear":
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
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            w = linear.weight.detach().to(wrapper.weight.dtype)
            wrapper.weight.copy_(w)
            if linear.bias is not None:
                wrapper.bias.copy_(linear.bias.detach())
            wrapper._init_qparams_from_weight(w)
        return wrapper

    @torch.no_grad()
    def _init_qparams_from_weight(self, w: Tensor) -> None:
        """Initialize scale (and zero point) from per-group weight statistics."""
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
        w = self.weight
        if self.lora_rank > 0:
            w = w + self.lora_a @ self.lora_b
        return w

    def quantize_dequantize(self) -> Tensor:
        """Return the fake-quantized weight (with LoRA applied)."""
        w = self._effective_weight()
        w_g = w.reshape(self.out_features, self.num_groups, self.group_size)
        scale = self.scale.unsqueeze(-1)

        if self.symmetric:
            q = _round_ste(w_g / scale)
            q = _clamp_ste(q, self.qmin, self.qmax)
            w_dq = q * scale
        else:
            zp = self.zero_point.to(scale.dtype).unsqueeze(-1)
            q = _round_ste(w_g / scale + zp)
            q = _clamp_ste(q, self.qmin, self.qmax)
            w_dq = (q - zp) * scale

        return w_dq.reshape(self.out_features, self.in_features).to(w.dtype)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.quantize_dequantize(), self.bias)

    @torch.no_grad()
    def init_lora_from_svd(self) -> None:
        """
        Initialize LoRA adapters with a truncated SVD of the quantization
        residual ``Q(W) - W``.

        With ``A @ B`` set to that residual, the corrected pre-quantization
        weight ``W + A @ B`` equals ``Q(W)`` and therefore quantizes back to
        itself, yielding zero residual error at initialization. Equivalent to
        the SVD-based init used by NNCF for FQ_LORA.
        """
        if self.lora_rank <= 0:
            return
        # Quantization residual without LoRA contribution.
        w = self.weight
        w_g = w.reshape(self.out_features, self.num_groups, self.group_size)
        scale = self.scale.unsqueeze(-1)
        if self.symmetric:
            q = torch.round(w_g / scale).clamp(self.qmin, self.qmax)
            w_q = q * scale
        else:
            zp = self.zero_point.to(scale.dtype).unsqueeze(-1)
            q = torch.round(w_g / scale + zp).clamp(self.qmin, self.qmax)
            w_q = (q - zp) * scale
        w_q = w_q.reshape(self.out_features, self.in_features).to(w.dtype)
        residual = (w_q - w).float() / 100.0  # [out, in]

        # Truncated SVD: residual = U S V^T  ->  A = U sqrt(S), B = sqrt(S) V^T
        u_full, s_full, v_full = torch.linalg.svd(residual, full_matrices=False)
        rank = self.lora_rank
        s_sqrt = torch.sqrt(s_full[:rank])
        a = u_full[:, :rank] * s_sqrt.unsqueeze(0)        # [out, r]
        b = v_full[:rank, :] * s_sqrt.unsqueeze(1)        # [r, in]
        self.lora_a.copy_(a.to(self.lora_a.dtype))
        self.lora_b.copy_(b.to(self.lora_b.dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_bits={self.num_bits}, group_size={self.group_size}, "
            f"symmetric={self.symmetric}, lora_rank={self.lora_rank}, "
            f"log_scale={self.log_scale}, bias={self.bias is not None}"
        )

    # ------------------------------------------------------------------ #
    # Per-group multiplicative clip search on the scale (AWQ-style).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def apply_clip_search(
        self,
        num_steps: int = 21,
        min_factor: float = 0.5,
        max_factor: float = 1.0,
        importance: Optional[Tensor] = None,
        update_zero_point: bool = True,
    ) -> None:
        """
        Search a multiplicative factor ``c in [min_factor, max_factor]`` per
        ``(out_channel, group)`` that minimizes the per-group MSE between the
        FP weight and its quantize-dequantize image when the scale is
        ``c * scale``. For asymmetric quantization, the integer ``zero_point``
        of each (out, group) is also recomputed for the candidate scale when
        ``update_zero_point`` is ``True``.

        Especially impactful for INT2, where a small range shrink trades a
        couple of saturated tails for finer in-range resolution and reduces
        per-group MSE.

        :param num_steps: Number of grid points in ``[min_factor, max_factor]``.
        :param min_factor: Smallest multiplicative clip factor.
        :param max_factor: Largest multiplicative clip factor (``1.0`` keeps
            the current scale).
        :param importance: Optional ``[in_features]`` activation importance
            used to weight the per-group MSE. ``None`` -> uniform.
        :param update_zero_point: Recompute the per-group integer zero point
            from the clipped per-group range (asymmetric only).
        """
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}.")

        out_f = self.out_features
        n_g, g = self.num_groups, self.group_size
        device = self.weight.device

        w_g = self.weight.float().reshape(out_f, n_g, g)
        s_init = self.scale.detach().float()  # [out_f, n_g]

        if importance is None:
            imp = None
        else:
            if importance.numel() != self.in_features:
                raise ValueError(
                    f"importance must have {self.in_features} elements, got {importance.numel()}."
                )
            imp = importance.detach().to(device=device, dtype=torch.float32).reshape(1, n_g, g)

        # For asymmetric mode, derive (wmin, wmax) per group once.
        if not self.symmetric:
            wmin = w_g.amin(dim=-1)  # [out_f, n_g]
            wmax = w_g.amax(dim=-1)
            absmax_per_group = w_g.abs().amax(dim=-1).clamp_min(1e-8)
        else:
            absmax_per_group = w_g.abs().amax(dim=-1).clamp_min(1e-8)

        factors = torch.linspace(
            min_factor, max_factor, num_steps, device=device, dtype=torch.float32
        )
        best_err = torch.full((out_f, n_g), float("inf"), device=device)
        best_scale = s_init.clone()
        if not self.symmetric:
            best_zp = self.zero_point.detach().clone()

        for c in factors:
            if self.symmetric:
                # Symmetric: the natural clip is on absmax -> scale.
                s_try = (absmax_per_group * c / max(abs(self.qmin), abs(self.qmax))).clamp_min(1e-8)
                s_try_b = s_try.unsqueeze(-1)
                q = torch.round(w_g / s_try_b).clamp(self.qmin, self.qmax)
                w_dq = q * s_try_b
                zp_try = None
            else:
                # Asymmetric: clip both ends symmetrically around the midpoint.
                mid = 0.5 * (wmax + wmin)
                half = 0.5 * (wmax - wmin) * c
                wmin_c = mid - half
                wmax_c = mid + half
                s_try = ((wmax_c - wmin_c) / (self.qmax - self.qmin)).clamp_min(1e-8)
                if update_zero_point:
                    zp_try = (
                        (self.qmin - wmin_c / s_try).round().clamp(self.qmin, self.qmax).to(torch.int32)
                    )
                else:
                    zp_try = self.zero_point
                s_try_b = s_try.unsqueeze(-1)
                zp_b = zp_try.float().unsqueeze(-1)
                q = torch.round(w_g / s_try_b + zp_b).clamp(self.qmin, self.qmax)
                w_dq = (q - zp_b) * s_try_b

            sq = (w_dq - w_g) ** 2
            if imp is None:
                err = sq.mean(dim=-1)
            else:
                err = (sq * imp).sum(dim=-1) / imp.sum(dim=-1).clamp_min(1e-12)

            improved = err < best_err
            best_err = torch.where(improved, err, best_err)
            best_scale = torch.where(improved, s_try, best_scale)
            if not self.symmetric:
                best_zp = torch.where(improved, zp_try, best_zp)

        if self.log_scale:
            self._scale_param.copy_(torch.log(best_scale.clamp_min(1e-8)).to(self._scale_param.dtype))
        else:
            self._scale_param.copy_(best_scale.to(self._scale_param.dtype))
        if not self.symmetric:
            self.zero_point.copy_(best_zp)

    # ------------------------------------------------------------------ #
    # GPTQ-style initialization from calibration activations.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def init_from_gptq(
        self,
        x_calib: Tensor,
        percdamp: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize the quantizer (scale, zero point) and overwrite the base
        ``weight`` buffer with the GPTQ/OBQ-compensated dequantized weight.

        Implements the per-group OBQ update of `Frantar et al. 2022/23
        <https://arxiv.org/abs/2210.17323>`_:

        1. Compute the activation Hessian ``H = (1/N) X^T X`` and add
           proportional damping.
        2. Cholesky-invert and Cholesky-decompose ``H^{-1}`` to obtain an
           upper-triangular ``Hinv`` whose diagonal is used as a per-column
           normalizer for the OBQ update.
        3. Walk input columns in order; at the start of each input-channel
           group, recompute the per-group ``scale`` (and ``zero_point`` for
           asymmetric quantization) from the *currently compensated* weights
           in that group; then quantize each column and propagate the
           quantization error to the remaining columns via ``Hinv``.
        4. Store the per-group scales/zero-points and replace ``weight`` by
           the quantized-dequantized values, so a subsequent
           :meth:`quantize_dequantize` call returns the same tensor (LoRA at
           ``A=0`` keeps this fixed point).

        :param x_calib: ``[N, in_features]`` calibration input matrix.
        :param percdamp: Diagonal damping as a fraction of ``mean(diag(H))``.
            Stabilizes the inverse for ill-conditioned ``H``.
        :param eps: Numerical stabilizer.
        """
        device = self.weight.device
        out_f, in_f = self.out_features, self.in_features
        g, n_g = self.group_size, self.num_groups

        x = x_calib.detach().to(device=device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != in_f:
            raise ValueError(f"x_calib must have shape [N, {in_f}], got {tuple(x.shape)}.")

        w_full = self.weight.detach().float().clone()  # [out_f, in_f]; mutated in place below
        h_mat = (x.T @ x) / max(x.shape[0], 1)  # [in_f, in_f]

        # Drop dead input channels (zero column / zero diagonal).
        diag = torch.arange(in_f, device=device)
        dead = h_mat[diag, diag] == 0
        if dead.any():
            h_mat[dead, dead] = 1.0
            w_full[:, dead] = 0.0

        damp = percdamp * h_mat.diagonal().mean().clamp_min(eps)
        h_mat[diag, diag] += damp

        # Hinv (upper triangular Cholesky factor of the inverse Hessian).
        l_chol = torch.linalg.cholesky(h_mat)
        h_inv = torch.cholesky_inverse(l_chol)
        h_inv_chol = torch.linalg.cholesky(h_inv, upper=True)  # upper triangular, [in_f, in_f]

        new_scale = torch.empty((out_f, n_g), device=device, dtype=torch.float32)
        if not self.symmetric:
            new_zp = torch.empty((out_f, n_g), device=device, dtype=torch.int32)
        q_full = torch.empty_like(w_full)

        for grp_idx in range(n_g):
            i0 = grp_idx * g
            i1 = i0 + g
            w_grp = w_full[:, i0:i1].clone()  # [out_f, g] (will be modified locally)
            hinv_grp = h_inv_chol[i0:i1, i0:i1]  # [g, g] upper triangular

            # Recompute scale (and zero point) from the currently compensated
            # group weights so later groups benefit from the propagated error.
            if self.symmetric:
                absmax = w_grp.abs().amax(dim=-1).clamp_min(eps)
                s = absmax / max(abs(self.qmin), abs(self.qmax))  # [out_f]
                zp_row = None
            else:
                wmin = w_grp.amin(dim=-1)
                wmax = w_grp.amax(dim=-1)
                s = ((wmax - wmin) / (self.qmax - self.qmin)).clamp_min(eps)
                zp_row = (self.qmin - wmin / s).round().clamp(self.qmin, self.qmax)
            new_scale[:, grp_idx] = s
            if not self.symmetric:
                new_zp[:, grp_idx] = zp_row.to(torch.int32)

            err_grp = torch.zeros_like(w_grp)
            q_grp = torch.zeros_like(w_grp)
            for j in range(g):
                w_col = w_grp[:, j]
                d = hinv_grp[j, j]
                if self.symmetric:
                    q = torch.round(w_col / s).clamp(self.qmin, self.qmax)
                    w_dq = q * s
                else:
                    q = torch.round(w_col / s + zp_row).clamp(self.qmin, self.qmax)
                    w_dq = (q - zp_row) * s
                q_grp[:, j] = w_dq
                err = (w_col - w_dq) / d.clamp_min(eps)
                err_grp[:, j] = err
                if j + 1 < g:
                    w_grp[:, j + 1 :] -= err.unsqueeze(1) * hinv_grp[j, j + 1 :].unsqueeze(0)

            q_full[:, i0:i1] = q_grp
            # Propagate per-group error to all subsequent groups.
            if i1 < in_f:
                w_full[:, i1:] -= err_grp @ h_inv_chol[i0:i1, i1:]

        # Persist quantizer parameters and the compensated dequantized weight.
        if self.log_scale:
            self._scale_param.copy_(torch.log(new_scale.clamp_min(eps)).to(self._scale_param.dtype))
        else:
            self._scale_param.copy_(new_scale.to(self._scale_param.dtype))
        if not self.symmetric:
            self.zero_point.copy_(new_zp)

        self.weight.copy_(q_full.to(self.weight.dtype))
        # Reset LoRA so that effective_weight == quantized weight at start.
        if self.lora_rank > 0:
            self.lora_a.zero_()
            nn.init.kaiming_uniform_(self.lora_b, a=5**0.5)

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
        w_g = self.weight.float().reshape(self.out_features, self.num_groups, self.group_size)
        s = scale.unsqueeze(-1)
        if self.symmetric:
            q = torch.round(w_g / s).clamp(self.qmin, self.qmax)
            target = q
            w_dq = q * s
        else:
            zp = self.zero_point.to(s.dtype).unsqueeze(-1)
            q = torch.round(w_g / s + zp).clamp(self.qmin, self.qmax)
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
        device = self.weight.device
        out_f, in_f = self.out_features, self.in_features
        n_g, g = self.num_groups, self.group_size

        s = s_per_channel.detach().to(device=device, dtype=torch.float32).reshape(1, n_g, g)
        x = x_calib.detach().to(device=device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != in_f:
            raise ValueError(f"x_calib must have shape [N, {in_f}], got {tuple(x.shape)}.")
        x_g = x.reshape(x.shape[0], n_g, g)

        w_full = self.weight.float()
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
            zero_mask = (target_t.abs() < eps)
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
    skip_name_substrings: Optional[list[str]] = None,
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
            bias=qmod.bias is not None,
        ).to(device=w_dq.device, dtype=w_dq.dtype)
        with torch.no_grad():
            linear.weight.copy_(w_dq)
            if qmod.bias is not None:
                linear.bias.copy_(qmod.bias.detach())
        _set_submodule(model, name, linear)
    return model
