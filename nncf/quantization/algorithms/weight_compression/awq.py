# Copyright (c) 2024 Intel Corporation
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar

from nncf import Dataset
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
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_scale_and_zp
from nncf.quantization.passes import transform_to_inference_graph
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
TWeightType = TypeVar("TWeightType")


@dataclass
class AWQCompressionInfo:
    """
    Information on AWQ nodes.
    """

    weight_params: WeightCompressionParameters = None
    target_node: NNCFNode = None
    merge_node: NNCFNode = None


class AWQ(Algorithm):
    """
    Modified AWQ algorithm implementation.
    """

    def __init__(
        self,
        model: TModel,
        name_to_node_mapping: Dict[str, Any],
        all_weight_params: List[WeightCompressionParameters],
        nodes_to_compress: List[NNCFNode],
        activations: Optional[Dict[str, TTensor]] = None,
        subset_size: int = 32,
        percent_to_apply=0.002,
        alpha_min=0.0,
        alpha_max=1.0,
        steps=100,
    ):
        """
        :param model: Model for applying algorithm.
        :param name_to_node_mapping: Name to node mapping for updating node weights.
        :param all_weight_params: List of all weight parameters.
        :param nodes_to_compress: List of nodes for processing.
        :param activations: The input activations of the layers considered for compression.
        :param subset_size: The number of samples for AWQ.
        :param percent_to_apply: The percent of outliers for correction.
        :param alpha_min: Minimum value of smoothness parameter for grid search.
        :param alpha_max: Maximal value of smoothness parameter for grid search.
        :param steps: The number of the steps in grid search.
        """
        super().__init__()
        self.name_to_node_mapping = name_to_node_mapping
        self._all_weight_params = all_weight_params
        self._nodes_to_compress = nodes_to_compress
        self._activations = activations
        self._subset_size = subset_size
        self._percent_to_apply = percent_to_apply
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._steps = steps
        self._backend_entity = None
        self._patterns = None

        self._set_backend_entity(model)

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        :param all_weight_params: List of all weight parameters.
        :param nodes_to_compress: List of nodes for processing.
        :param activations: The input activations of the layers considered for compression.
        """

        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVAWQAlgoAlgoBackend

            self._backend_entity = OVAWQAlgoAlgoBackend(model, self.name_to_node_mapping)
            self._patterns = self._backend_entity.get_awq_patterns()
        else:
            raise RuntimeError(
                "Cannot return backend-specific AWQ entity because {} is not supported!".format(model_backend.value)
            )

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        """
        Applies the algorithm to the model.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param statistic_points: Statistic points with collected statistics values.
        :param dataset: A representative dataset for the calibration process.
        :return: A resulting model.
        """
        matches = []

        inference_nncf_graph = transform_to_inference_graph(deepcopy(graph), [], [], [])
        nx_graph = inference_nncf_graph.get_nx_graph_copy()
        for _, pattern_graph in self._patterns.items():
            matches.extend(find_subgraphs_matching_pattern(nx_graph, pattern_graph(), strict=False))

        if len(matches) == 0:
            nncf_logger.info("No matching patterns were found for applying AWQ algorithm, it will be skipped.")
            return model

        transformation_layout = TransformationLayout()
        model_transformer = ModelTransformerFactory.create(model, inplace=True)

        awq_data = {}
        name_mapping = {wp.weight_name: idx for idx, wp in enumerate(self._all_weight_params)}

        for match in matches:
            nncf_node = graph.get_node_by_key(match[-1])
            if not self._backend_entity.is_node_with_weights(nncf_node, graph):
                continue

            target_node_names = []
            for weight_op_friendly_name, _ in self._backend_entity.get_weight_names_and_port_ids(nncf_node, graph):
                target_node_names.append(weight_op_friendly_name)

            weight_params = self._all_weight_params[name_mapping[target_node_names[-1]]]

            if weight_params.compression_config.num_bits != 4:
                continue
            target_node = self._nodes_to_compress[name_mapping[target_node_names[-1]]]

            # avoid matching different patterns for the same node
            if target_node.node_name in awq_data:
                continue

            nncf_node = graph.get_node_by_key(match[0])

            if self._backend_entity.is_node_with_weights(nncf_node, graph):  # pattern MatMul->Multiply->MatMul
                merge_node_names = []
                for weight_op_friendly_name, _ in self._backend_entity.get_weight_names_and_port_ids(nncf_node, graph):
                    merge_node_names.append(weight_op_friendly_name)
                merge_node = self._nodes_to_compress[name_mapping[merge_node_names[-1]]]
            else:  # pattern Act->MatMul or Act->Multiply->MatMul
                merge_node = nncf_node

            awq_data[target_node.node_name] = AWQCompressionInfo(weight_params, target_node, merge_node)

        alpha_step = (self._alpha_max - self._alpha_min) / self._steps
        models_cache = dict()

        for k, awq_data_item in track(awq_data.items(), description="Applying AWQ"):
            wp = awq_data_item.weight_params
            target_node = awq_data_item.target_node
            merge_node = awq_data_item.merge_node
            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue

            nncf_logger.debug(f"Apply AWQ for: {wp.node_with_weight.node_name}")

            _, weight_port_id = weight_data[0]

            config = wp.compression_config

            stats = self._activations[k]
            X = fns.stack([fns.mean(stat, axis=0) for stat in stats])
            X = fns.transpose(X)

            s = fns.max(fns.abs(X), axis=1)

            if X.shape[1] > self._subset_size:
                lens = [stat.shape[0] for stat in stats]
                step = X.shape[1] // self._subset_size
                idxs = [i[0] for i in sorted(enumerate(lens), key=lambda x: -x[1])][::step]
                X = X[:, idxs]

            top_k = max(int(s.shape[0] * self._percent_to_apply), 1)
            topk_idxs = fns.argsort(-s)[:top_k]

            group_size = config.group_size
            if group_size == -1:
                group_size = s.shape[0]

            groups_to_correct = set()
            for idx in topk_idxs:
                groups_to_correct.add(idx.data // group_size)

            groups_to_correct = list(groups_to_correct)

            weight = self._backend_entity.get_weight(
                wp.node_with_weight, weight_port_id, model, graph
            )  # get_const_value(wp.weight_node)
            assert isinstance(wp.reduction_axes, tuple) and len(wp.reduction_axes) == 1
            reduction_axis = wp.reduction_axes[0]

            if reduction_axis == 0:
                weight = fns.transpose(weight)
                reduction_axis = 1

            shape_vector = fns.mean(X, axis=1)
            scale = fns.ones_like(shape_vector)

            awq_config = deepcopy(config)
            awq_config.group_size = -1

            for gi in groups_to_correct:
                offset = gi * group_size
                gscale = s[offset : offset + group_size]

                a_min = fns.quantile(gscale, 0.1)
                a_max = 1e2
                gscale = fns.clip(gscale, a_min=a_min, a_max=a_max)

                gweight = weight[:, offset : offset + group_size]
                gacts = X[offset : offset + group_size, :]

                fp32_out = fns.matmul(gweight, gacts)
                min_diff = fns.max(fns.abs(fp32_out))
                best_scale = None

                alpha = self._alpha_min
                awq_pipeline = None
                for _ in range(self._steps):
                    cur_scale = gscale**alpha
                    cur_w = gweight * cur_scale

                    g_c_scale, g_c_zp = get_scale_and_zp(cur_w, reduction_axis, awq_config)
                    zp_shape = None if g_c_zp is None else g_c_zp.shape
                    if awq_pipeline is None:
                        model_k = (awq_config.mode, awq_config.num_bits, gweight.shape, g_c_scale.shape, zp_shape)
                        if model_k in models_cache:
                            awq_pipeline = models_cache[model_k]
                        else:
                            awq_pipeline = self._backend_entity.get_awq_pipeline(awq_config, gweight.shape, g_c_scale.shape, zp_shape,\
                                                                             cur_scale.shape, gacts.shape, fp32_out.shape)
                            models_cache[model_k] = awq_pipeline

                    if g_c_zp is None:
                        cur_diff = awq_pipeline([cur_w.data, g_c_scale.data, cur_scale.data, gacts.data, fp32_out.data])
                    else:
                        cur_diff = awq_pipeline([cur_w.data, g_c_scale.data, g_c_zp.data,
                                                 cur_scale.data, gacts.data, fp32_out.data])
                    if cur_diff < min_diff:
                        min_diff = cur_diff
                        best_scale = cur_scale
                    alpha += alpha_step

                if best_scale is not None:
                    scale.data[offset : offset + group_size] = best_scale.data

            a_scale = scale
            w_scale = scale
            if wp.reduction_axes[0] == 0:
                w_scale = fns.unsqueeze(w_scale, 1)
                a_scale = fns.unsqueeze(1.0 / a_scale, 0)
            else:
                w_scale = fns.unsqueeze(w_scale, 0)
                a_scale = fns.unsqueeze(1.0 / a_scale, 1)

            scaled_weight = weight * w_scale
            self._backend_entity.set_weight(wp.node_with_weight, weight_port_id, model, graph, scaled_weight)

            if self._backend_entity.is_node_with_weights(
                merge_node, graph
            ):  # for MatMul->Multiply->MatMul pattern scale merged to first MatMul
                for _, port_id in self._backend_entity.get_weight_names_and_port_ids(merge_node, graph):
                    merge_weight = self._backend_entity.get_weight(merge_node, port_id, model, graph)
                    merge_weight = merge_weight * a_scale
                    self._backend_entity.set_weight(merge_node, port_id, model, graph, merge_weight)
                a_scale = fns.transpose(a_scale)
            else:  # for Act->Multiply->MatMul and Act->MatMul patterns scale inserted after Act as extra node
                a_scale = fns.transpose(a_scale)
                next_nodes = graph.get_next_nodes(merge_node)
                source_node_output_port = graph.get_output_edges(merge_node)[0].output_port_id
                scale_insertion_command = self._backend_entity.scale_insertion_command(
                    merge_node, next_nodes, source_node_output_port, a_scale.data
                )
                transformation_layout.register(scale_insertion_command)

            # update activations for next usage
            for i, stat in enumerate(self._activations[k]):
                stat = stat * a_scale
                self._activations[k][i] = stat

        transformed_model = model_transformer.transform(transformation_layout)

        return transformed_model

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        return StatisticPointsContainer()
