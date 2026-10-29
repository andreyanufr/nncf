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

from copy import deepcopy
from typing import TypeVar

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_float_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")


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
    
            print("Layer: ", wp.weight_name)
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
            )
            if config.mode == nncf.CompressWeightsMode.DC_INT4_SYM:
                res[weight_name] = self._refine_dc_subscales(
                    stats, weight, wp.reduction_axes, config, scale, act_ch_axis
                )
            else:
                res[weight_name] = CompressedWeight(None, scale, zero_point, None)

        return res

    def _refine_dc_subscales(
        self,
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        scale: Tensor,
        act_ch_axis: int = -1,
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
        residual = fns.matmul(
            fns.moveaxis(grouped_weight - dequantized.reshape(grouped_weight.shape), -2, -3), activations
        )
        diff_before = fns.mean(residual**2)
        penalty = fns.mean((dequantized.reshape(grouped_weight.shape) - grouped_weight) ** 2, axis=-1)
        best_diffs = fns.moveaxis(fns.mean(residual**2, axis=-1), -1, -2)
        if self._weight_penalty > 0:
            best_diffs += self._weight_penalty * penalty

        for subgroup in range(4):
            subgroup_weight = weight[..., subgroup, :]
            subgroup_input = subgroup_activations[..., subgroup, :, :]
            original_weight = dequantized[..., subgroup, :].clone()
            original_out = fns.matmul(fns.moveaxis(original_weight, -2, -3), subgroup_input)
            original_penalty = fns.mean((original_weight - subgroup_weight) ** 2, axis=-1) / 4
            base_residual = residual + original_out
            base_penalty = penalty - original_penalty
            best_residual, best_penalty = residual, penalty

            for candidate in range(1, 17):
                candidate_tensor = fns.clip(fns.round(int8_weight[..., subgroup, :] / candidate), -8, 7)
                candidate_weight = candidate_tensor * candidate * primary_scale[..., 0, :]
                candidate_out = fns.matmul(fns.moveaxis(candidate_weight, -2, -3), subgroup_input)
                candidate_residual = base_residual - candidate_out
                candidate_penalty = base_penalty + fns.mean((candidate_weight - subgroup_weight) ** 2, axis=-1) / 4
                diffs = fns.moveaxis(fns.mean(candidate_residual**2, axis=-1), -1, -2)
                if self._weight_penalty > 0:
                    diffs += self._weight_penalty * candidate_penalty
                improved = diffs < best_diffs
                best_diffs = fns.where(improved, diffs, best_diffs)
                best_penalty = fns.where(improved, candidate_penalty, best_penalty)
                best_residual = fns.where(
                    fns.unsqueeze(fns.moveaxis(improved, -1, -2), -1), candidate_residual, best_residual
                )
                mask = fns.unsqueeze(improved, -1)
                tensor[..., subgroup, :] = fns.where(mask, candidate_tensor, tensor[..., subgroup, :]).astype(
                    tensor.dtype
                )
                subscale[..., subgroup, :] = fns.where(mask, candidate, subscale[..., subgroup, :]).astype(
                    subscale.dtype
                )
            residual, penalty = best_residual, best_penalty

        dequantized = tensor.astype(TensorDataType.float32) * subscale * primary_scale
        fp_outs = fns.matmul(fns.moveaxis(grouped_weight, -2, -3), activations)
        q_outs = fns.matmul(fns.moveaxis(dequantized.reshape(grouped_weight.shape), -2, -3), activations)
        print("Diff before :", diff_before)
        print("Diff after :", fns.mean((fp_outs - q_outs) ** 2))

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

        if not config.is_integer:
            q_weights, compressed_weight = float_quantize_dequantize_weight(
                weight, cur_config, reduction_axis, return_compressed_weight=True
            )
            zp = None
        else:
            q_weights, compressed_weight = integer_quantize_dequantize_weight(
                weight, cur_config, reduction_axis, return_compressed_weight=True
            )
            zp = compressed_weight.zero_point
            if config.mode == nncf.CompressWeightsMode.DC_INT4_SYM:
                scale = fns.squeeze(compressed_weight.scale, axis=-2)
            else:
                scale = compressed_weight.scale
            if zp is not None:
                zp = zp.astype(compressed_weight.scale.dtype)
        if not config.is_integer:
            scale = compressed_weight.scale
        compressed_weight = compressed_weight.get_unscaled_tensor()
        if config.mode == nncf.CompressWeightsMode.DC_INT4_SYM:
            compressed_shape = list(compressed_weight.shape)
            compressed_shape[-2] *= compressed_shape[-1]
            del compressed_shape[-1]
            compressed_weight = compressed_weight.reshape(compressed_shape)

        s = fns.unsqueeze(s, -2)
        s, _ = reshape_weight_for_grouped_quantization(s, reduction_axis, group_size)

        weight, _ = reshape_weight_for_grouped_quantization(weight, reduction_axis, group_size)

        # all weight in group has importance based on corresponding input activations
        importance = fns.ones_like(weight)
        importance = importance * s

        target, zero_mask = get_target_zero_mask(compressed_weight, zp)
        importance = fns.where(zero_mask, 0.0, importance)

        # normalize importances for every group of weights to make sum of them equal to 1.0
        denum = fns.sum(importance, axis=-1, keepdims=True)
        importance = importance / (denum + eps)

        X, _ = reshape_weight_for_grouped_quantization(X, -2, group_size)
        fp_outs = fns.matmul(fns.moveaxis(weight, -2, -3), X)
        q_outs = fns.matmul(fns.moveaxis(q_weights, -2, -3), X)
        print("Diff before :", fns.mean((fp_outs - q_outs) ** 2))

        # metric for minimization with shape [C_OUT, N_GROUPS], N_GROUPS = C_IN / GROUP_SIZE
        # For 3D weights, it is [Batch Size, C_OUT, N_GROUPS]
        min_max_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
        min_max_scale_diffs = fns.moveaxis(min_max_scale_diffs, -1, -2)

        if weight_penalty > 0.0:
            min_max_scale_diffs += weight_penalty * fns.mean((q_weights - weight) ** 2, axis=-1)

        # Baseline values ensure correct behavior when initial_steps=0 and/or scale_steps=0.
        best_diffs = min_max_scale_diffs
        result_scale = scale

        scale_sign = scale / fns.abs(scale)
        zero_scale = 0.001
        zero_mask = zero_scale * zero_mask.astype(weight.dtype)

        # iterative rectification of initial scale
        for i in range(initial_steps):
            near_to_ideal_scale = estimate_scales(weight, target, zero_mask, importance)
            near_to_ideal_scale = near_to_ideal_scale * scale_sign
            near_to_ideal_scale = fns.where(fns.abs(near_to_ideal_scale) < eps, result_scale, near_to_ideal_scale)

            if not config.is_integer:
                q_weights_ = float_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                )
            else:
                q_weights_ = integer_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                    precomputed_zero_point=zp,
                )

            q_outs = fns.matmul(fns.moveaxis(q_weights_, -2, -3), X)

            ideal_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
            ideal_scale_diffs = fns.moveaxis(ideal_scale_diffs, -1, -2)
            if weight_penalty > 0.0:
                ideal_scale_diffs += weight_penalty * fns.mean((q_weights_ - weight) ** 2, axis=-1)

            mask = (ideal_scale_diffs > best_diffs).astype(best_diffs.dtype)

            best_diffs = mask * best_diffs + (1.0 - mask) * ideal_scale_diffs

            mask = fns.unsqueeze(mask, axis=-1)

            result_scale = mask * result_scale + (1.0 - mask) * near_to_ideal_scale

            if i < initial_steps - 1:
                if not config.is_integer:
                    compressed_weight = do_float_quantization(weight, config, precomputed_scale=result_scale)
                else:
                    compressed_weight = do_integer_quantization(
                        weight,
                        config,
                        precomputed_scale=result_scale,
                        precomputed_zero_point=zp,
                    )
                compressed_weight = compressed_weight.get_unscaled_tensor()
                if config.mode == nncf.CompressWeightsMode.DC_INT4_SYM:
                    compressed_shape = list(compressed_weight.shape)
                    compressed_shape[-2] *= compressed_shape[-1]
                    del compressed_shape[-1]
                    compressed_weight = compressed_weight.reshape(compressed_shape)
                target, zero_mask = get_target_zero_mask(compressed_weight, zp)
                zero_mask = zero_scale * zero_mask.astype(weight.dtype)

        # iterative rectification of scale based on grid search
        for scale_step in range(scale_steps):
            factor = 1.0 - 0.05 * scale_step
            scaled_scale = factor * scale

            if not config.is_integer:
                compressed_weight = do_float_quantization(weight, config, precomputed_scale=scaled_scale)
            else:
                compressed_weight = do_integer_quantization(
                    weight,
                    config,
                    precomputed_scale=scaled_scale,
                    precomputed_zero_point=zp,
                )
            compressed_weight = compressed_weight.get_unscaled_tensor()
            if config.mode == nncf.CompressWeightsMode.DC_INT4_SYM:
                compressed_shape = list(compressed_weight.shape)
                compressed_shape[-2] *= compressed_shape[-1]
                del compressed_shape[-1]
                compressed_weight = compressed_weight.reshape(compressed_shape)

            target, zero_mask = get_target_zero_mask(compressed_weight, zp)
            zero_mask = zero_scale * zero_mask.astype(weight.dtype)
            near_to_ideal_scale = estimate_scales(weight, target, zero_mask, importance)
            near_to_ideal_scale = near_to_ideal_scale * scale_sign
            near_to_ideal_scale = fns.where(fns.abs(near_to_ideal_scale) < eps, result_scale, near_to_ideal_scale)

            if not config.is_integer:
                q_weights_ = float_quantize_dequantize_weight(weight, config, precomputed_scale=near_to_ideal_scale)
            else:
                q_weights_ = integer_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                    precomputed_zero_point=zp,
                )

            q_outs = fns.matmul(fns.moveaxis(q_weights_, -2, -3), X)
            ideal_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
            ideal_scale_diffs = fns.moveaxis(ideal_scale_diffs, -1, -2)
            if weight_penalty > 0.0:
                ideal_scale_diffs += weight_penalty * fns.mean((q_weights_ - weight) ** 2, axis=-1)

            mask = (ideal_scale_diffs > best_diffs).astype(best_diffs.dtype)

            best_diffs = mask * best_diffs + (1.0 - mask) * ideal_scale_diffs

            mask = fns.unsqueeze(mask, axis=-1)

            result_scale = mask * result_scale + (1.0 - mask) * near_to_ideal_scale

        if not config.is_integer:
            q_weights_ = float_quantize_dequantize_weight(weight, config, precomputed_scale=result_scale)
        else:
            q_weights_ = integer_quantize_dequantize_weight(
                weight,
                config,
                precomputed_scale=result_scale,
                precomputed_zero_point=zp,
            )
        q_outs = fns.matmul(fns.moveaxis(q_weights_, -2, -3), X)
        print("Diff after :", fns.mean((fp_outs - q_outs) ** 2))

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
