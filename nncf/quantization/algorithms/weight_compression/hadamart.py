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
from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar

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
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_nf4_scale
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import quantize_dequantize_weight
import nncf.quantization.algorithms.weight_compression.hadamard_utils as hadamard_utils
from nncf.quantization.passes import transform_to_inference_graph
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

from nncf.openvino.graph.node_utils import convert_op
from nncf.openvino.graph.node_utils import create_ov_const_from_tensor
from nncf.openvino.graph.node_utils import get_const_value_as_numpy_tensor
from nncf.openvino.graph.node_utils import get_const_value_as_ov_tensor
from nncf.openvino.graph.node_utils import get_weight_channel_axes
import openvino as ov
import torch
from openvino.runtime import opset13 as opset


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


class Hadamart(Algorithm):
    def __init__(
        self,
        subset_size: int = 32,
        percent_to_apply: float = 0.002,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        steps: int = 100,
    ):
        """
        :param subset_size: The number of samples for AWQ.
        :param percent_to_apply: The percent of outliers for correction.
        :param alpha_min: Minimum value of smoothness parameter for grid search.
        :param alpha_max: Maximal value of smoothness parameter for grid search.
        :param steps: The number of the steps in grid search.
        """
        super().__init__()
        self._subset_size = subset_size
        self._percent_to_apply = percent_to_apply
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._steps = steps
        self._backend_entity = None
        self._patterns = None
        self._scale_per_target_node = {}

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH]

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

        else:
            msg = f"Cannot return backend-specific AWQ entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)
        self._patterns = self._backend_entity.get_awq_patterns()

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: List[WeightCompressionParameters],
        nodes_to_compress: List[NNCFNode],
        statistics: Dict[str, WCTensorStatistic] = None,
        wc_backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> TModel:
        """
        Applies the algorithm to the model.
        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param nodes_to_compress: List of nodes for processing.
        :param statistics: Input activation statistics for each node.
        :param wc_backend_entity: Weight compression algorithm backend.
        :return: A resulting model.
        """
        self._set_backend_entity(model, wc_backend_entity)
        matches = []

        inference_nncf_graph = transform_to_inference_graph(deepcopy(graph), [], [], [], [])
        nx_graph = inference_nncf_graph.get_nx_graph_copy()
        for _, pattern_graph in self._patterns.items():
            matches.extend(find_subgraphs_matching_pattern(nx_graph, pattern_graph(), strict=False))

        if len(matches) == 0:
            nncf_logger.info("No matching patterns were found for applying AWQ algorithm, it will be skipped.")
            return model

        awq_data = {}
        name_mapping = {wp.weight_name: idx for idx, wp in enumerate(all_weight_params)}

        for match in matches:
            nncf_node = graph.get_node_by_key(match[-1])
            if not self._backend_entity.is_node_with_weights(nncf_node, graph):
                continue

            target_node_names = []
            for weight_op_friendly_name, _ in self._backend_entity.get_weight_names_and_port_ids(nncf_node, graph):
                target_node_names.append(weight_op_friendly_name)

            # skip node if it is in IgnoredScope or should not be compressed
            if target_node_names[-1] not in name_mapping:
                continue

            weight_params = all_weight_params[name_mapping[target_node_names[-1]]]

            if weight_params.compression_config.num_bits != 4:
                continue
            target_node = nodes_to_compress[name_mapping[target_node_names[-1]]]

            # avoid matching different patterns for the same node
            if target_node.node_name in awq_data:
                continue

            nncf_node = graph.get_node_by_key(match[0])

            if self._backend_entity.is_node_with_weights(nncf_node, graph):  # pattern MatMul->Multiply->MatMul
                merge_node_names = []
                for weight_op_friendly_name, _ in self._backend_entity.get_weight_names_and_port_ids(nncf_node, graph):
                    merge_node_names.append(weight_op_friendly_name)
                merge_node = nodes_to_compress[name_mapping[merge_node_names[-1]]]
            else:  # pattern Act->MatMul or Act->Multiply->MatMul
                merge_node = nncf_node

            awq_data[target_node.node_name] = AWQCompressionInfo(weight_params, target_node, merge_node)

        for k, awq_data_item in track(awq_data.items(), description="Applying Hadamart transform"):
            wp = awq_data_item.weight_params
            target_node = awq_data_item.target_node
            merge_node = awq_data_item.merge_node
            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue

            nncf_logger.debug(f"Hadamart transform for: {wp.node_with_weight.node_name}")

            _, weight_port_id = weight_data[0]

            config = wp.compression_config

            weight = self._backend_entity.get_weight(
                wp.node_with_weight, weight_port_id, model, graph
            )  # get_const_value(wp.weight_node)
            weight_dtype = weight.dtype
            weight = weight.astype(TensorDataType.float32)
            assert isinstance(wp.reduction_axes, tuple) and len(wp.reduction_axes) == 1
            reduction_axis = wp.reduction_axes[0]

            if reduction_axis == 0:
                weight = fns.transpose(weight)
                reduction_axis = 1
            
            in_features = weight.shape[-1]
            hadK, K, H = hadamard_utils.get_hadK(in_features)
            
            H = H.to(torch.float32)

            h_weight = hadamard_utils.matmul_had_cuda_H(torch.tensor(weight.data), hadK, H, K)

            h_weight = h_weight.cpu().numpy()
            weight = fns.zeros_like(weight) + h_weight
            self._backend_entity.set_weight(wp.node_with_weight, weight_port_id, model, graph, weight)
            
            # def matmul_had_cuda_H(X, hadK, H, K):
            #     n = X.shape[-1]
            #     input = X.view(-1, K, n // K)
            #     input = input @ H
            #     input = hadK.to(input.device).to(input.dtype) @ input
            #     return input.reshape(X.shape)

            next_node = graph.get_next_nodes(merge_node)[0]
            next_node = self._backend_entity.name_to_node_mapping[next_node.node_name]

            if K == 1:
                H_pow2 = opset.constant(H.cpu().numpy())
                out = opset.matmul(next_node, H_pow2, transpose_a=False, transpose_b=False, name=wp.node_with_weight.node_name+"_H_pow2")
                input = out
            else:
                input = opset.reshape(next_node, (-1, K, in_features // k), False)
                H_pow2 = opset.constant(H.cpu().numpy())
                mm_h_pow2 = opset.matmul(input, H_pow2, transpose_a=False, transpose_b=False, name=wp.node_with_weight.node_name+"_H_pow2")
                H_K = opset.constant(hadK.cpu().numpy())
                mm_k = opset.matmul(mm_h_pow2, H_K, transpose_a=True, transpose_b=False, name=wp.node_with_weight.node_name+"_H_K")
                out = opset.reshape(mm_k, (-1, in_features), False)


            # node_input = node.input_value(0)

            # for node_output in node.outputs():
            #     for target_input in node_output.get_target_inputs():
            #         target_input.replace_source_output(node_input)
        
            node_output_port = next_node.output(0)
            node_output_source_ports = node_output_port.get_target_inputs()

            for node_output_source_port in node_output_source_ports:
                if node_output_source_port.get_node().friendly_name == input.friendly_name:
                    continue
                node_output_source_port.replace_source_output(out.output(0))

        return model

    def update_statistics(self, statistics):
        return statistics

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        return StatisticPointsContainer()
