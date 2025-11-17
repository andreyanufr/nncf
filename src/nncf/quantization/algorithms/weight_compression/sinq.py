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

from copy import deepcopy
from typing import Optional, TypeVar

import nncf
from nncf import nncf_logger
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.quantization.passes import transform_to_inference_graph
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
TWeightType = TypeVar("TWeightType")


FP16_MAX_VALUE = 65504.0
FP16_OVERFLOW_MARGIN = 0.25


class SINQ(Algorithm):
    """
    Modified SINQ algorithm implementation.
    """

    def __init__(
        self,
        steps: int = 16,
    ):
        """
        :param subset_size: The number of samples for AWQ.
        :param percent_to_apply: The percent of outliers for correction.
        :param alpha_min: Minimum value of smoothness parameter for grid search.
        :param alpha_max: Maximal value of smoothness parameter for grid search.
        :param steps: The number of the steps in grid search.
        :param prefer_data_aware_scaling: Determines whether to use activations to calculate scales.
        """
        super().__init__()
        self._steps = steps
        self._scale_per_target_node = {}

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.ONNX]

    def _set_backend_entity(
        self, model: TModel, wc_backend_entity: Optional[WeightCompressionAlgoBackend] = None
    ) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        :param wc_backend_entity: Weight compression algorithm backend.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVAWQAlgoAlgoBackend

            self._backend_entity = OVAWQAlgoAlgoBackend(model, wc_backend_entity.name_to_node_mapping)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTAWQAlgoAlgoBackend

            self._backend_entity = PTAWQAlgoAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXAWQAlgoAlgoBackend

            self._backend_entity = FXAWQAlgoAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXAWQAlgoAlgoBackend

            self._backend_entity = ONNXAWQAlgoAlgoBackend(model)
        else:
            msg = f"Cannot return backend-specific AWQ entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)
        self._patterns = self._backend_entity.get_sinq_patterns()

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: list[WeightCompressionParameters],
        statistics: Optional[dict[str, WCTensorStatistic]] = None,
        wc_backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> TModel:
        """
        Applies the algorithm to the model.
        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param wc_backend_entity: Weight compression algorithm backend.
        :return: A resulting model.
        """
        self._set_backend_entity(model, wc_backend_entity)

        transformation_layout = TransformationLayout()
        model_transformer = ModelTransformerFactory.create(model, inplace=True)

        description = "Applying SINQ"

        sinq_data = self._get_sinq_data(graph, all_weight_params)

        if not len(sinq_data) == 0:
            for ln_node, mm_nodes in track(sinq_data.items(), description=description):
                if len(mm_nodes) == 1:
                    continue
                weight_datas = [
                    self._backend_entity.get_weight_names_and_port_ids(mm_node, graph) for mm_node in mm_nodes
                ]

                nncf_logger.debug(f"{description} for: {ln_node.node_name}")

                weight_port_id = [weight_data[0][1] for weight_data in weight_datas]
                weights = [
                    self._backend_entity.get_weight(mm_node, wp_id, model, graph)
                    for mm_node, wp_id in zip(mm_nodes, weight_port_id)
                ]
                weight_dtypes = [weight.dtype for weight in weights]
                weights = [weight.astype(TensorDataType.float32) for weight in weights]

                wps = [wp for mm in mm_nodes for wp in all_weight_params if wp.node_with_weight == mm]

                split_sizes = [w.shape[0] for w in weights]
                for i in range(len(split_sizes) - 1):
                    split_sizes[i + 1] += split_sizes[i]

                combined_weights = fns.concatenate(weights, axis=0)
                scaled_weight, scale1, scale2 = self._step(
                    combined_weights, wps[0].reduction_axes, wps[0].compression_config
                )
                split_scaled_weights = fns.split(scaled_weight, split_sizes, axis=0)
                split_scale2 = fns.split(scale2, split_sizes, axis=0)

                self._backend_entity.scale_constant(ln_node, model, graph, scale1)

                for i, wp in enumerate(wps):
                    scaled_weight = split_scaled_weights[i]
                    sinq_scale2 = split_scale2[i]
                    scaled_weight = scaled_weight.astype(weight_dtypes[i])
                    self._backend_entity.set_weight(wp.node_with_weight, weight_port_id[i], model, graph, scaled_weight)
                    a_scale = scale1.astype(weight_dtypes[i])

                    if sinq_scale2.shape[1] == 1 and sinq_scale2.shape[2] == 1:
                        sinq_scale2 = sinq_scale2[:, 0, :]
                    self._scale_per_target_node[wp.node_with_weight.node_name] = a_scale

                    wp.sinq_scales = sinq_scale2.astype(weight_dtypes[i])
        else:
            for wp in track(all_weight_params, description=description):
                weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
                if len(weight_data) != 1:  # not supported by the algorithm
                    continue

                nncf_logger.debug(f"{description} for: {wp.node_with_weight.node_name}")

                _, weight_port_id = weight_data[0]
                weight = self._backend_entity.get_weight(
                    wp.node_with_weight, weight_port_id, model, graph
                )  # get_const_value(wp.weight_node)
                weight_dtype = weight.dtype
                weight = weight.astype(TensorDataType.float32)

                weight, scale1, scale2 = self._step(weight, wp.reduction_axes, wp.compression_config)

                scaled_weight = weight.astype(weight_dtype)
                self._backend_entity.set_weight(wp.node_with_weight, weight_port_id, model, graph, scaled_weight)

                a_scale = scale1.astype(weight_dtype)
                prev_nodes = [pn for pn in graph.get_previous_nodes(wp.node_with_weight) if pn.node_type != "Convert"][
                    0
                ]
                edge = graph._get_edges(prev_nodes, wp.node_with_weight)

                source_node_output_port = edge[0].output_port_id
                scale_insertion_command = self._backend_entity.scale_insertion_command(
                    prev_nodes, [wp.node_with_weight], source_node_output_port, a_scale.data
                )
                transformation_layout.register(scale_insertion_command)

                self._scale_per_target_node[wp.node_with_weight.node_name] = a_scale

                wp.sinq_scales = scale2.astype(weight_dtype)

        transformed_model = model_transformer.transform(transformation_layout)

        return transformed_model

    def _step(self, weight: TTensor, reduction_axes: list[int], config: WeightCompressionConfig):
        reduction_axis = reduction_axes[0]
        weight = weight.astype(TensorDataType.float32)
        eps = fns.finfo(weight).eps

        was_transposed = False
        if reduction_axis == 0:
            weight = fns.transpose(weight)
            reduction_axis = 1
            was_transposed = True

        group_size = config.group_size if config.group_size != -1 else weight.shape[reduction_axis]
        original_weight, _ = reshape_weight_for_grouped_quantization(weight, reduction_axis, group_size)
        original_weight = fns.transpose(original_weight, (1, 0, 2))

        s1 = []
        s2 = []
        scales_weight = []
        for i in range(original_weight.shape[0]):
            w = original_weight[i, :, :]
            scaled_w, scale1, scale2 = self._sinkhorn(w, order=self._steps)
            s1.append(scale1)
            s2.append(scale2)
            scales_weight.append(scaled_w)

        res_s1 = fns.concatenate(s1, axis=1)
        res_s2 = fns.expand_dims(fns.concatenate(s2, axis=1), -1)
        scales_weight = fns.concatenate(scales_weight, axis=1)

        return scales_weight, res_s1, res_s2

    def _sinkhorn(
        self, matrix: TTensor, order=8, clip_min=1e-3, clip_max=1e3, eps=1e-6, stop_on_increasing_imbalance=True
    ):
        """
        vmap-friendly Sinkhorn that returns *the* mu1 / mu2 corresponding
        to the matrix with the minimal imbalance encountered during the
        iteration.

        The return value is a tuple
            (scaled_matrix, mu1_at_minimum, mu2_at_minimum)
        """
        measure = fns.std
        dtype = TensorDataType.float32
        backend = matrix.backend
        m = matrix.astype(dtype)
        
        mu1_star = fns.mean(fns.abs(m), axis=0, keepdims=True)
        mu2_star = fns.zeros((m.shape[0], 1), dtype=TensorDataType.float32, backend=backend) + 1.0
        
        # scaled = m / mu1_star / mu2_star
        # return scaled, mu1_star, mu2_star

        def imbalance(mat):
            s1, s2 = measure(mat, 1), measure(mat, 0)
            s_min = fns.clip(fns.minimum(s1.min(), s2.min()), a_min=1e-12, a_max=None)
            s_max = fns.maximum(s1.max(), s2.max())
            return s_max / s_min  # scalar

        imb_min = fns.tensor(float("inf"), dtype=TensorDataType.float32, backend=backend)
        gate = fns.tensor(0.0, dtype=TensorDataType.float32, backend=backend)

        tgt_small = (
            fns.minimum(
                fns.clip(fns.std(m, 1), clip_min, clip_max).min(), fns.clip(fns.std(m, 0), clip_min, clip_max).min()
            )
            + eps
        )

        log_mu1 = fns.zeros((1, m.shape[1]), dtype=TensorDataType.float32, backend=backend)
        log_mu2 = fns.zeros((m.shape[0], 1), dtype=TensorDataType.float32, backend=backend)

        # Known-good candidates for the step k=0
        cur0 = m
        ib0 = imbalance(cur0)
        imb_min = fns.minimum(imb_min, ib0)

        mu1_star = fns.exp(log_mu1)  # .clone()
        mu2_star = fns.exp(log_mu2)  # .clone()

        for _ in range(order):
            cur = (m / fns.exp(log_mu1)) / fns.exp(log_mu2)
            ib = imbalance(cur)

            # update the best-so-far candidates
            better = (ib <= imb_min).item()  # astype(dtype)   # 1 if new best
            imb_min = min(imb_min, ib)
            mu1_star = fns.exp(log_mu1) if better else mu1_star
            mu2_star = fns.exp(log_mu2) if better else mu2_star

            # early-exit condition
            if stop_on_increasing_imbalance:
                rising = (ib > imb_min).astype(dtype)
                gate = fns.clip(gate + rising, a_min=None, a_max=1.0)  # once 1 → always 1

            # still-running samples update the dual variables
            g = 1.0 - gate

            std_r = fns.clip(measure(cur, 1), clip_min, clip_max)
            std_c = fns.clip(measure(cur, 0), clip_min, clip_max)

            sal_col = fns.clip((std_c / tgt_small), 0.7, 2.0)
            sal_col = fns.log(sal_col)

            sal_row = fns.log(fns.clip((std_r[:, None] / tgt_small), 0.7, 2.0))

            log_mu1 = fns.clip((log_mu1 + (sal_col * g)), -0.3, 10.0)
            log_mu2 = fns.clip((log_mu2 + (sal_row * g)), -0.3, 10.0)

        # final scaled matrix and the recorded best scaling vectors
        scaled = m / mu1_star / mu2_star
        return scaled, mu1_star, mu2_star

    @staticmethod
    def _clamp_scale(magnitudes, threshold, scale, clamped_scale):
        return fns.where(magnitudes < threshold, scale, clamped_scale)

    def _data_free_step(self, weight, axis):
        eps = fns.finfo(weight).eps
        scale = fns.maximum(fns.mean(fns.abs(weight), axis=axis), eps)
        return 1 / scale

    def update_statistics(self, statistics):
        if not statistics:
            return statistics

        # Multiply activations by the computed scales
        for node_name, scale in self._scale_per_target_node.items():
            for mean_stat in statistics[node_name].mean_values:
                mean_stat *= fns.squeeze(scale)
        return statistics

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        return StatisticPointsContainer()

    def _get_sinq_data(
        self, graph: NNCFGraph, all_weight_params: list[WeightCompressionParameters]
    ) -> dict[NNCFNode, list[NNCFNode]]:
        """
        Finds sinq patterns in graph and returns it.
        :param graph: Model graph.
        :param all_weight_params: list of all weight parameters.
        :return: A dict with node names and matched sinq patterns.
        """
        matches = []
        inference_nncf_graph = transform_to_inference_graph(deepcopy(graph), [], [], [], [])
        nx_graph = inference_nncf_graph.get_nx_graph_copy()
        for pattern_graph in self._patterns.values():
            matches.extend(find_subgraphs_matching_pattern(nx_graph, pattern_graph(), strict=False))

        if len(matches) == 0:
            nncf_logger.info("No matching patterns were found for applying AWQ algorithm, it will be skipped.")
            return {}

        sinq_data = {}
        processed_names = {wp.node_with_weight.node_name for wp in all_weight_params}

        for match in matches:
            mm_nodes = [graph.get_node_by_key(m) for m in match[2:]]
            if any(not self._backend_entity.is_node_with_weights(node, graph) for node in mm_nodes):
                continue
            ln_node = graph.get_node_by_key(match[0])

            # if ln_node in sinq_data:
            #     sinq_data[ln_node].append(mm_nodes[0])
            # else:
            #     sinq_data[ln_node] = [mm_nodes[0]]

            if ln_node in sinq_data:
                if len(mm_nodes) > len(sinq_data[ln_node]):
                    sinq_data[ln_node] = mm_nodes
            else:
                sinq_data[ln_node] = mm_nodes

        res = {}
        for k, v in sinq_data.items():
            if any(n.node_name not in processed_names for n in v):
                continue
            res[k] = v

        return res
