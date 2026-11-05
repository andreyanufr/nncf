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

import csv
import os
from copy import deepcopy
from pathlib import Path
from typing import TypeVar

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dc_int4_asymmetric_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_float_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorBackend

TModel = TypeVar("TModel")

SCALE_ESTIMATION_DUMP_ENV = "NNCF_SCALE_ESTIMATION_DUMP"


class ScaleEstimation:
    """
    Scale estimation algorithm implementation.
    """

    def __init__(
        self,
        subset_size: int = 32,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
    ):
        """
        :param subset_size: The number of samples for scale estimation.
        :param initial_steps: The number of the steps for absmax scale rectification.
        :param scale_steps: The number of the steps for grid search scale rectification
                            from 1.0 to 1.0 - 0.05 * scale_step.
        :param weight_penalty: coefficient for penalty between fp and compressed weights. If -1 then doesn't apply.
        """
        super().__init__()
        self._subset_size = subset_size
        self._initial_steps = initial_steps
        self._scale_steps = scale_steps
        self._weight_penalty = weight_penalty

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.ONNX]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend

            self._backend_entity = PTWeightCompressionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXWeightCompressionAlgoBackend

            self._backend_entity = FXWeightCompressionAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXWeightCompressionAlgoBackend

            self._backend_entity = ONNXWeightCompressionAlgoBackend()
        else:
            msg = (
                "Cannot return backend-specific Scale Estimation entity because"
                f" {model_backend.value} is not supported!"
            )
            raise nncf.UnsupportedBackendError(msg)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: list[WeightCompressionParameters],
        statistics: dict[str, WCTensorStatistic],
        backend_entity: WeightCompressionAlgoBackend | None = None,
    ) -> dict[str, CompressedWeight]:
        """
        Estimates better scale for the int4 nodes in the model.
        Minimizes per-group difference between floating point MatMul and
        MatMul with compressed weights.
        The algorithm computes weighted scale for the group of weights in MatMul, which
        shared the same scale.
        For DC_INT4_SYM, a final subscale search keeps the estimated primary scale fixed
        and refines the integer subscales and corresponding INT4 weights.
        For DC_INT4_ASYM, both primary scales are estimated with their UINT6 subgroup
        parameters requantized for every candidate.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param backend_entity: Weight compression algorithm backend.
        :return: Estimated compression parameters keyed by weight name. DC_INT4_SYM entries
            also contain the refined compressed tensor and subscales.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)
        res = dict()

        for wp in track(all_weight_params, description="Applying Scale Estimation"):
            weight_name = wp.weight_name
            node_name = wp.node_with_weight.node_name
            config = wp.compression_config

            if config.num_bits > 4 or node_name not in statistics:
                res[weight_name] = CompressedWeight()
                continue

            stats = statistics[node_name]

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)
            act_ch_axis, _ = self._backend_entity.get_activation_channel_axis_and_shape(graph, wp.node_with_weight)
            if config.mode == nncf.CompressWeightsMode.DC_INT4_ASYM:
                res[weight_name] = self._estimate_dc_asymmetric(
                    stats, weight, wp.reduction_axes, config, act_ch_axis, layer_name=node_name
                )
                continue

            scale, zero_point = self.calculate_quantization_params(
                stats,
                weight,
                wp.reduction_axes,
                config,
                act_ch_axis,
                self._subset_size,
                self._initial_steps,
                self._scale_steps,
                self._weight_penalty,
                layer_name=node_name,
            )
            if config.mode == nncf.CompressWeightsMode.DC_INT4_SYM:
                res[weight_name] = self._refine_dc_subscales(
                    stats, weight, wp.reduction_axes, config, scale, act_ch_axis, layer_name=node_name
                )
            else:
                res[weight_name] = CompressedWeight(None, scale, zero_point, None)

        return res

    def _estimate_dc_asymmetric(
        self,
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        act_ch_axis: int = -1,
        layer_name: str | None = None,
    ) -> CompressedWeight:
        """
        Estimates both FP16 primary scales for DC_INT4_ASYM, requantizing UINT6 parameters.

        Accepts only strict per-block improvements of the activation reconstruction
        error, including the optional weight penalty. The complete compressed weight
        is returned so export preserves the selected scales and subgroup parameters.

        :param statistics: Collected activation statistics.
        :param weight: Original weight tensor.
        :param reduction_axes: Weight reduction axis.
        :param config: DC_INT4_ASYM compression configuration.
        :param act_ch_axis: Activation channel axis.
        :param layer_name: Name of the layer used to dump the initial and final objective values.
            Nothing is dumped if None.
        :return: Compressed weights with estimated hierarchical parameters.
        """
        if weight.backend == TensorBackend.ov:
            weight = weight.as_numpy_tensor()
        weight = weight.astype(TensorDataType.float32)
        magnitudes, activations = process_stats(statistics, self._subset_size, act_ch_axis)
        if activations.backend == TensorBackend.ov:
            activations = activations.as_numpy_tensor()
            magnitudes = magnitudes.as_numpy_tensor()
        activations = activations.astype(TensorDataType.float32)
        axis = reduction_axes[0] % weight.ndim
        weight = fns.moveaxis(weight, axis, -1)
        grouped, _ = reshape_weight_for_grouped_quantization(weight, -1, 256)
        activations, _ = reshape_weight_for_grouped_quantization(activations, -2, 256)
        magnitudes, _ = reshape_weight_for_grouped_quantization(magnitudes, -1, 256)
        importance = fns.unsqueeze(magnitudes**2, -3)
        fp_out = fns.matmul(fns.moveaxis(grouped, -2, -3), activations)
        best = do_dc_int4_asymmetric_quantization(grouped, config, None)

        def objective(compressed: CompressedWeight) -> Tensor:
            dequantized = do_integer_dequantization(compressed).reshape(grouped.shape)
            output = fns.matmul(fns.moveaxis(dequantized, -2, -3), activations)
            error = fns.moveaxis(fns.sum((fp_out - output) ** 2, axis=-1), -1, -2)
            if self._weight_penalty > 0:
                error = error + self._weight_penalty * fns.mean((grouped - dequantized) ** 2, axis=-1)
            return error

        def weight_error(compressed: CompressedWeight) -> Tensor:
            return fns.mean((grouped - do_integer_dequantization(compressed).reshape(grouped.shape)) ** 2)

        best_error = initial_error = objective(best)
        initial_weight_error = weight_error(best)

        def consider(scale: Tensor, min_scale: Tensor) -> None:
            nonlocal best_error, best
            candidate = do_dc_int4_asymmetric_quantization(grouped, config, None, scale, min_scale)
            error = objective(candidate)
            improved = error < best_error
            best_error = fns.where(improved, error, best_error)
            mask = fns.unsqueeze(fns.unsqueeze(improved, -1), -1)
            best = CompressedWeight(
                tensor=fns.where(mask, candidate.tensor, best.tensor),
                scale=fns.where(mask, candidate.scale, best.scale),
                subscale=fns.where(mask, candidate.subscale, best.subscale),
                min_scale=fns.where(mask, candidate.min_scale, best.min_scale),
                submin=fns.where(mask, candidate.submin, best.submin),
            )

        initial_scale = fns.squeeze(best.scale, -2)
        initial_min_scale = fns.squeeze(best.min_scale, -2)
        for _ in range(self._initial_steps):
            target = (best.tensor.astype(TensorDataType.float32) * best.subscale).reshape(grouped.shape)
            minimum = (best.submin * fns.ones_like(best.tensor)).reshape(grouped.shape)
            scale = fns.squeeze(best.scale, -2)
            min_scale = fns.squeeze(best.min_scale, -2)
            denominator = fns.sum(importance * target**2, axis=-1, keepdims=True)
            numerator = fns.sum(importance * target * (grouped + min_scale * minimum), axis=-1, keepdims=True)
            candidate_scale = fns.where(denominator > 0, numerator / fns.maximum(denominator, 1e-20), scale)
            consider(fns.clip(candidate_scale, 2.0**-24, 65504.0), min_scale)

            target = (best.tensor.astype(TensorDataType.float32) * best.subscale).reshape(grouped.shape)
            minimum = (best.submin * fns.ones_like(best.tensor)).reshape(grouped.shape)
            scale = fns.squeeze(best.scale, -2)
            denominator = fns.sum(importance * minimum**2, axis=-1, keepdims=True)
            numerator = fns.sum(importance * minimum * (scale * target - grouped), axis=-1, keepdims=True)
            candidate_min_scale = fns.where(
                denominator > 0, numerator / fns.maximum(denominator, 1e-20), fns.squeeze(best.min_scale, -2)
            )
            consider(scale, fns.clip(candidate_min_scale, 2.0**-24, 65504.0))

        for step in range(self._scale_steps):
            factor = 1.0 - 0.05 * step
            if factor <= 0:
                break
            consider(initial_scale * factor, fns.squeeze(best.min_scale, -2))
            consider(fns.squeeze(best.scale, -2), initial_min_scale * factor)

        _dump_objective(
            layer_name, config, "scale", initial_error, best_error, initial_weight_error, weight_error(best)
        )

        if axis != weight.ndim - 1:
            best = CompressedWeight(
                tensor=fns.moveaxis(best.tensor, -4, -1),
                scale=fns.moveaxis(best.scale, -4, -1),
                subscale=fns.moveaxis(best.subscale, -4, -1),
                min_scale=fns.moveaxis(best.min_scale, -4, -1),
                submin=fns.moveaxis(best.submin, -4, -1),
            )
        return best

    def _refine_dc_subscales(
        self,
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        scale: Tensor,
        act_ch_axis: int = -1,
        layer_name: str | None = None,
    ) -> CompressedWeight:
        """
        Refines DC_INT4_SYM subscales without changing the estimated primary scale.

        Searches all legal subscales in [1, 16], one subgroup at a time. Candidates
        minimize the full 128-weight group's matmul error, including the configured
        weight penalty. Strict improvements only are accepted.

        :param statistics: Collected activation statistics.
        :param weight: Original weight tensor.
        :param reduction_axes: Weight reduction axis.
        :param config: DC_INT4_SYM compression configuration.
        :param scale: Primary scale selected by the two scale-estimation stages.
        :param act_ch_axis: Activation channel axis.
        :param layer_name: Name of the layer used to dump the initial and final objective values.
            Nothing is dumped if None.
        :return: Compressed weights with refined subscales and unchanged primary scale.
        """
        compressed = do_integer_quantization(weight, config, reduction_axes, precomputed_scale=scale)
        weight = weight.astype(TensorDataType.float32)
        _, activations = process_stats(statistics, self._subset_size, act_ch_axis)
        activations = activations.astype(TensorDataType.float32)

        # Work with [..., output channels, groups, subgroups, subgroup size].
        transposed = reduction_axes[0] % weight.ndim == weight.ndim - 2
        tensor, subscale, primary_scale = compressed.tensor, compressed.subscale, compressed.scale
        if tensor is None or subscale is None or primary_scale is None:
            msg = "DC_INT4_SYM quantization must provide a compressed tensor, primary scale, and subscales."
            raise nncf.InternalError(msg)
        if transposed:
            weight = fns.moveaxis(weight, -1, -2)
            tensor = fns.moveaxis(tensor, -1, -4)
            subscale = fns.moveaxis(subscale, -1, -4)
            primary_scale = fns.moveaxis(primary_scale, -1, -4)

        grouped_weight, _ = reshape_weight_for_grouped_quantization(weight, weight.ndim - 1, 128)
        weight = grouped_weight.reshape(tensor.shape)
        activations, _ = reshape_weight_for_grouped_quantization(activations, activations.ndim - 2, 128)
        subgroup_activations = activations.reshape((*activations.shape[:-2], 4, 32, activations.shape[-1]))
        int8_weight = fns.clip(fns.round(weight / primary_scale), -128, 127)
        dequantized = tensor.astype(TensorDataType.float32) * subscale * primary_scale
        # The residual and the weight penalty are updated incrementally, one subgroup at a time.
        residual = fns.matmul(
            fns.moveaxis(grouped_weight - dequantized.reshape(grouped_weight.shape), -2, -3), activations
        )
        penalty = fns.mean((dequantized.reshape(grouped_weight.shape) - grouped_weight) ** 2, axis=-1)

        def objective(residual: Tensor, penalty: Tensor) -> Tensor:
            error = fns.moveaxis(fns.sum(residual**2, axis=-1), -1, -2)
            if self._weight_penalty > 0:
                error = error + self._weight_penalty * penalty
            return error

        best_error = initial_error = objective(residual, penalty)
        initial_weight_error = fns.mean((dequantized - weight) ** 2)

        def consider(subgroup: int, candidate: int, base_residual: Tensor, base_penalty: Tensor) -> None:
            nonlocal best_error, residual, penalty
            candidate_tensor = fns.clip(fns.round(int8_weight[..., subgroup, :] / candidate), -8, 7)
            candidate_weight = candidate_tensor * candidate * primary_scale[..., 0, :]
            candidate_out = fns.matmul(
                fns.moveaxis(candidate_weight, -2, -3), subgroup_activations[..., subgroup, :, :]
            )
            candidate_residual = base_residual - candidate_out
            candidate_penalty = base_penalty + fns.mean((candidate_weight - weight[..., subgroup, :]) ** 2, axis=-1) / 4
            error = objective(candidate_residual, candidate_penalty)
            improved = error < best_error
            best_error = fns.where(improved, error, best_error)
            penalty = fns.where(improved, candidate_penalty, penalty)
            residual = fns.where(fns.unsqueeze(fns.moveaxis(improved, -1, -2), -1), candidate_residual, residual)
            mask = fns.unsqueeze(improved, -1)
            tensor[..., subgroup, :] = fns.where(mask, candidate_tensor, tensor[..., subgroup, :]).astype(tensor.dtype)
            subscale[..., subgroup, :] = fns.where(mask, candidate, subscale[..., subgroup, :]).astype(subscale.dtype)

        for subgroup in range(4):
            # Exclude the initial contribution of the subgroup, so candidates only add their own one.
            initial_weight = dequantized[..., subgroup, :]
            initial_out = fns.matmul(fns.moveaxis(initial_weight, -2, -3), subgroup_activations[..., subgroup, :, :])
            base_residual = residual + initial_out
            base_penalty = penalty - fns.mean((initial_weight - weight[..., subgroup, :]) ** 2, axis=-1) / 4
            for candidate in range(1, 17):
                consider(subgroup, candidate, base_residual, base_penalty)

        final_weight_error = fns.mean((tensor.astype(TensorDataType.float32) * subscale * primary_scale - weight) ** 2)
        _dump_objective(
            layer_name, config, "subscale", initial_error, best_error, initial_weight_error, final_weight_error
        )

        if transposed:
            tensor = fns.moveaxis(tensor, -4, -1)
            subscale = fns.moveaxis(subscale, -4, -1)
        return CompressedWeight(tensor=tensor, scale=compressed.scale, subscale=subscale)

    @staticmethod
    def calculate_quantization_params(
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        act_ch_axis: int = -1,
        subset_size: int = 32,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
        layer_name: str | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Calculates the quantization parameters for a given set of weights and activations.
        This function estimates the optimal quantization scale for weight compression by
        minimizing the difference between floating-point operations and operations with
        quantized weights.

        The function uses an iterative process:
        1. Initial scale rectification based on activation statistics.
        2. A grid search to further refine the scale parameters.

        :param statistics: The input activations of the layer reduced over batch and sequence length dimensions,
            together with original activation tensor shapes.
        :param weight: The weight tensor that is being quantized.
        :param reduction_axes: Tuple specifying the axes along which the reduction is performed for quantization.
        :param config: Configuration parameters for the weight compression, including quantization settings.
        :param act_ch_axis: The activation channel axis. Defaults to -1.
        :param subset_size: The number of samples to use for scale estimation. Defaults to 32.
        :param initial_steps: The number of steps for initial scale rectification using activation statistics.
            Defaults to 5.
        :param scale_steps: The number of steps for refining the scale using a grid search. Defaults to 10.
        :param weight_penalty: Penalty coefficient applied to the difference between floating-point
            and quantized weights. A value of -1 disables the penalty. Defaults to -1.0.
        :param layer_name: Name of the layer used to dump the initial and final objective values.
            Nothing is dumped if None.
        :return: A tensor containing the calculated quantization scales and zero points if applicable.
        """
        reduction_axis = reduction_axes[0]

        s, X = process_stats(statistics, subset_size, act_ch_axis=act_ch_axis)

        X = X.astype(TensorDataType.float32)
        weight = weight.astype(TensorDataType.float32)
        eps = fns.finfo(weight).eps
        is_3d_weight = len(weight.shape) == 3

        if is_3d_weight and len(X.shape) == 2:
            # For cases such as grouped matmul, where the activation can be 2D or 3D but weights are 3D
            s = fns.unsqueeze(s, 0)
            X = fns.unsqueeze(X, 0)

        was_transposed = False
        if reduction_axis == 0 or (reduction_axis == 1 and is_3d_weight):
            # Weights
            # 3D: [num_experts, hidden_dimension, out_features] -> [num_experts, out_features, hidden_dimension]
            # 2D: [hidden_dimension, out_features] -> [out_features, hidden_dimension]
            weight = fns.moveaxis(weight, -1, -2)
            reduction_axis = weight.ndim - 1
            was_transposed = True

        group_size = config.group_size if config.group_size != -1 else weight.shape[reduction_axis]
        cur_config = deepcopy(config)
        cur_config.group_size = group_size
        is_dc_sym = config.mode == nncf.CompressWeightsMode.DC_INT4_SYM

        if not config.is_integer:
            q_weights, compressed = float_quantize_dequantize_weight(
                weight, cur_config, reduction_axis, return_compressed_weight=True
            )
            zp = None
            scale = compressed.scale
        else:
            q_weights, compressed = integer_quantize_dequantize_weight(
                weight, cur_config, reduction_axis, return_compressed_weight=True
            )
            zp = compressed.zero_point
            scale = fns.squeeze(compressed.scale, axis=-2) if is_dc_sym else compressed.scale
            if zp is not None:
                zp = zp.astype(compressed.scale.dtype)

        def unscaled_tensor(compressed: CompressedWeight) -> Tensor:
            tensor = compressed.get_unscaled_tensor()
            if is_dc_sym:
                # Merge [..., N_GROUPS, N_SUBGROUPS, SUBGROUP_SIZE] into [..., N_GROUPS, GROUP_SIZE].
                shape = list(tensor.shape)
                subgroup_size = shape.pop()
                shape[-1] *= subgroup_size
                tensor = tensor.reshape(shape)
            return tensor

        s = fns.unsqueeze(s, -2)
        s, _ = reshape_weight_for_grouped_quantization(s, reduction_axis, group_size)
        weight, _ = reshape_weight_for_grouped_quantization(weight, reduction_axis, group_size)
        X, _ = reshape_weight_for_grouped_quantization(X, -2, group_size)

        # All weights in a group have importance based on the corresponding input activations,
        # normalized for every group of weights to make their sum equal to 1.0.
        target, zero_mask = get_target_zero_mask(unscaled_tensor(compressed), zp)
        importance = fns.where(zero_mask, 0.0, fns.ones_like(weight) * s)
        importance = importance / (fns.sum(importance, axis=-1, keepdims=True) + eps)

        zero_scale = 0.001
        zero_mask = zero_scale * zero_mask.astype(weight.dtype)
        scale_sign = scale / fns.abs(scale)
        fp_outs = fns.matmul(fns.moveaxis(weight, -2, -3), X)

        def quantize_dequantize(candidate_scale: Tensor) -> Tensor:
            if not config.is_integer:
                return float_quantize_dequantize_weight(weight, config, precomputed_scale=candidate_scale)
            return integer_quantize_dequantize_weight(
                weight, config, precomputed_scale=candidate_scale, precomputed_zero_point=zp
            )

        def get_target(candidate_scale: Tensor) -> tuple[Tensor, Tensor]:
            if not config.is_integer:
                compressed = do_float_quantization(weight, config, precomputed_scale=candidate_scale)
            else:
                compressed = do_integer_quantization(
                    weight, config, precomputed_scale=candidate_scale, precomputed_zero_point=zp
                )
            target, zero_mask = get_target_zero_mask(unscaled_tensor(compressed), zp)
            return target, zero_scale * zero_mask.astype(weight.dtype)

        def objective(dequantized: Tensor) -> Tensor:
            # Shape [C_OUT, N_GROUPS], N_GROUPS = C_IN / GROUP_SIZE; [Batch Size, C_OUT, N_GROUPS] for 3D weights.
            output = fns.matmul(fns.moveaxis(dequantized, -2, -3), X)
            error = fns.moveaxis(fns.sum((fp_outs - output) ** 2, axis=-1), -1, -2)
            if weight_penalty > 0.0:
                error = error + weight_penalty * fns.sum((dequantized - weight) ** 2, axis=-1)
            return error

        # Baseline values ensure correct behavior when initial_steps=0 and/or scale_steps=0.
        best_error = initial_error = objective(q_weights)
        result_scale = scale

        def consider(target: Tensor, zero_mask: Tensor) -> None:
            nonlocal best_error, result_scale
            candidate = estimate_scales(weight, target, zero_mask, importance) * scale_sign
            candidate = fns.where(fns.abs(candidate) < eps, result_scale, candidate)
            error = objective(quantize_dequantize(candidate))
            improved = error <= best_error
            best_error = fns.where(improved, error, best_error)
            result_scale = fns.where(fns.unsqueeze(improved, axis=-1), candidate, result_scale)

        # iterative rectification of initial scale
        for i in range(initial_steps):
            consider(target, zero_mask)
            if i < initial_steps - 1:
                target, zero_mask = get_target(result_scale)

        # iterative rectification of scale based on grid search
        for scale_step in range(scale_steps):
            consider(*get_target((1.0 - 0.05 * scale_step) * scale))

        if layer_name is not None:
            _dump_objective(
                layer_name,
                config,
                "scale",
                initial_error,
                best_error,
                fns.mean((q_weights - weight) ** 2),
                fns.sum((quantize_dequantize(result_scale) - weight) ** 2),
            )

        if config.group_size == -1:
            result_scale = fns.squeeze(result_scale, axis=-2)
        if zp is not None and config.group_size == -1:
            zp = fns.squeeze(zp, axis=-2)
        if was_transposed:
            if config.group_size == -1:
                result_scale = fns.moveaxis(result_scale, -1, -2)
                if zp is not None:
                    zp = fns.moveaxis(zp, -1, -2)
            else:
                result_scale = fns.moveaxis(result_scale, (-1, -2, -3), (-2, -3, -1))
                if zp is not None:
                    zp = fns.moveaxis(zp, (-1, -2, -3), (-2, -3, -1))

        return result_scale, zp


def _dump_objective(
    layer_name: str | None,
    config: WeightCompressionConfig,
    stage: str,
    initial_error: Tensor,
    final_error: Tensor,
    initial_weight_error: Tensor,
    final_weight_error: Tensor,
) -> None:
    """
    Logs the average scale estimation objective and the mean squared error between the original
    and the quantized-dequantized weights before and after minimization. If the
    NNCF_SCALE_ESTIMATION_DUMP environment variable is set, the values are also appended
    to the CSV file it points to.

    :param layer_name: Name of the layer. Nothing is dumped if None.
    :param config: Compression configuration of the layer.
    :param stage: Name of the minimization stage, e.g. "scale" or "subscale".
    :param initial_error: Per-group objective values before minimization.
    :param final_error: Per-group objective values after minimization.
    :param initial_weight_error: Mean squared weight quantization error before minimization.
    :param final_weight_error: Mean squared weight quantization error after minimization.
    """
    if layer_name is None:
        return
    initial = float(fns.sum(initial_error).item())
    final = float(fns.sum(final_error).item())
    initial_weight = float(initial_weight_error.item())
    final_weight = float(final_weight_error.item())
    nncf_logger.info(
        f"Scale estimation [{config.mode.value}, gs={config.group_size}, {stage}] {layer_name}: "
        f"objective {initial:.6e} -> {final:.6e}, weight MSE {initial_weight:.6e} -> {final_weight:.6e}"
    )
    dump_path = os.environ.get(SCALE_ESTIMATION_DUMP_ENV)
    if not dump_path:
        return
    path = Path(dump_path)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "layer",
                    "mode",
                    "group_size",
                    "stage",
                    "initial_objective",
                    "final_objective",
                    "initial_weight_mse",
                    "final_weight_mse",
                ]
            )
        writer.writerow(
            [layer_name, config.mode.value, config.group_size, stage, initial, final, initial_weight, final_weight]
        )


def get_target_zero_mask(compressed_weights: Tensor, zp: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    Computes the target values and a mask indicating zero values in the target.

    :param compressed_weights: The compressed weights tensor.
    :param zp: The zero point tensor.
    :return: The compressed weights optionally adjusted by the zero point and
        a boolean mask indicating positions in the target that are close to zero.
    """
    target = compressed_weights
    if zp is not None:
        target = target.astype(dtype=zp.dtype) - zp
    zero_mask = fns.isclose(target, 0)
    return target, zero_mask


def estimate_scales(weight: Tensor, target: Tensor, zero_mask: Tensor, importance: Tensor) -> Tensor:
    """
    Estimates scales for the given weight, target, zero mask, and importance.

    :param weight: The weights tensor.
    :param target: The target values tensor.
    :param zero_mask: A boolean mask indicating positions in the target that are close to zero.
    :param importance: The importance values tensor.
    :return: The estimated scales
    """
    ideal_scale = fns.abs(weight) / (fns.abs(target) + zero_mask)
    weighted_scale = ideal_scale * importance
    near_to_ideal_scale = fns.sum(weighted_scale, axis=-1, keepdims=True)
    return near_to_ideal_scale
