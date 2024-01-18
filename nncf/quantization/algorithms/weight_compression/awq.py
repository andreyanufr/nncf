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
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.tensor import functions as fns
from nncf.quantization.algorithms.weight_compression.backend import AWQAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.passes import transform_to_inference_graph

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
TWeightType = TypeVar("TWeightType")


class AWQ(AWQAlgoBackend):
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
        alpha_min=0.01,
        alpha_max=1.0,
        steps=100,
    ):
        """ """
        super().__init__(model)
        self.name_to_node_mapping = name_to_node_mapping
        self._all_weight_params = all_weight_params
        self._nodes_to_compress = nodes_to_compress
        self._activations = activations
        self._subset_size = subset_size
        self._percent_to_apply = percent_to_apply
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._steps = steps
        self._patterns = self.get_awq_patterns()

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
            return model

        @dataclass
        class AWQCompressionInfo:
            """
            Information on how to compress (quantize) a specific weight.

            :param mode: Defines a mode for weight compression. Defaults to INT8_ASYM mode.
            :param group_size: Number of weights (e.g. 128) in the channel dimension,
                            that share quantization parameters (scale).
                The value -1 means no grouping. Defaults to -1.
            """

            weight_params: WeightCompressionParameters = None
            target_node: NNCFNode = None
            merge_node: NNCFNode = None

        target_node_names = []
        merge_node_names = []
        awq_data = {}
        name_mapping = {wp.weight_name: idx for idx, wp in enumerate(self._all_weight_params)}

        for match in matches:
            skip = False
            for m in match[:-1]:
                node = graph.get_node_by_key(m)
                n_outupts = len(graph.get_output_edges(node))
                if n_outupts > 1:
                    skip = True
            if skip:
                continue

            nncf_node = graph.get_node_by_key(match[-1])
            weight_port_ids = nncf_node.layer_attributes.get_const_port_ids()
            for weight_port_id in weight_port_ids:
                weight_op_friendly_name = nncf_node.layer_attributes.constant_attributes[weight_port_id]["name"]
                target_node_names.append(weight_op_friendly_name)

            nncf_node = graph.get_node_by_key(match[0])
            weight_port_ids = nncf_node.layer_attributes.get_const_port_ids()
            for weight_port_id in weight_port_ids:
                weight_op_friendly_name = nncf_node.layer_attributes.constant_attributes[weight_port_id]["name"]
                merge_node_names.append(weight_op_friendly_name)

            assert len(target_node_names) == len(merge_node_names)
            weight_params = self._all_weight_params[name_mapping[target_node_names[-1]]]
            if weight_params.compression_config.num_bits != 4:
                continue
            target_node = self._nodes_to_compress[name_mapping[target_node_names[-1]]]
            merge_node = self._nodes_to_compress[name_mapping[merge_node_names[-1]]]

            awq_data[target_node.node_name] = AWQCompressionInfo(weight_params, target_node, merge_node)

        alpha_step = (self._alpha_max - self._alpha_min) / self._steps

        for k, awq_data_item in track(awq_data.items(), description="Applying AWQ"):
            wp = awq_data_item.weight_params
            target_node = awq_data_item.target_node
            merge_node = awq_data_item.merge_node

            weight_data = self.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            config = wp.compression_config

            stats = self._activations[k]
            X = fns.stack([fns.mean(stat, axis=0) for stat in stats])
            X = fns.transpose(X)
            if X.shape[1] > self._subset_size:
                X = X[:, : self._subset_size]

            s = fns.max(fns.abs(X), axis=1)

            top_k = max(int(s.shape[0] * self._percent_to_apply), 1)
            topk_idxs = fns.argsort(-s)[:top_k]

            groups_to_correct = set()
            for idx in topk_idxs:
                groups_to_correct.add(idx.data // config.group_size)

            groups_to_correct = list(groups_to_correct)

            weight = self.get_weight(
                wp.node_with_weight, weight_port_id, model, graph
            )  # get_const_value(wp.weight_node)
            reduction_axis = wp.reduction_axis

            if reduction_axis == 0:
                weight = fns.transpose(weight)
                reduction_axis = 1

            shape_vector = fns.mean(X, axis=1)
            scale = fns.ones_like(shape_vector)

            awq_config = deepcopy(config)
            awq_config.group_size = -1

            for gi in groups_to_correct:
                offset = gi * config.group_size
                gscale = s[offset : offset + config.group_size]

                a_min = fns.quantile(gscale, 0.1)
                a_max = 1e2
                gscale = fns.clip(gscale, a_min=a_min, a_max=a_max)

                gweight = weight[:, offset : offset + config.group_size]
                gacts = X[offset : offset + config.group_size, :]

                fp32_out = fns.matmul(gweight, gacts)
                min_diff = fns.max(fns.abs(fp32_out))
                best_scale = None

                alpha = self._alpha_min
                for _ in range(self._steps):
                    cur_scale = gscale**alpha

                    g_compressed_weighs, g_c_scale, g_c_zp = do_integer_quantization(
                        gweight * cur_scale, reduction_axis, awq_config
                    )
                    g_decompressed_weighs = do_dequantization(g_compressed_weighs, g_c_scale, g_c_zp)
                    sacts = gacts / fns.unsqueeze(cur_scale, 1)

                    cur_out = fns.matmul(g_decompressed_weighs, sacts)
                    cur_diff = fns.mean((cur_out - fp32_out) ** 2)
                    if cur_diff < min_diff:
                        min_diff = cur_diff
                        best_scale = cur_scale
                    alpha += alpha_step

                if best_scale is not None:
                    scale.data[offset : offset + config.group_size] = best_scale.data

            a_scale = scale
            w_scale = scale
            if wp.reduction_axis == 0:
                w_scale = fns.unsqueeze(w_scale, 1)
                a_scale = fns.unsqueeze(1.0 / a_scale, 0)
            else:
                w_scale = fns.unsqueeze(w_scale, 0)
                a_scale = fns.unsqueeze(1.0 / a_scale, 1)

            scaled_weight = weight * w_scale
            self.set_weight(wp.node_with_weight, weight_port_id, model, graph, scaled_weight)

            for _, port_id in self.get_weight_names_and_port_ids(merge_node, graph):
                merge_weight = self.get_weight(merge_node, port_id, model, graph)
                merge_weight = merge_weight * a_scale
                self.set_weight(merge_node, port_id, model, graph, merge_weight)

        return model

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        return StatisticPointsContainer()
