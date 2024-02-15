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
from math import ceil
from typing import Any, Dict, List, Optional, TypeVar

from nncf import Dataset
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.tensor import functions as fns
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.quantization.passes import transform_to_inference_graph

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
        alpha_min=0.01,
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

            self._backend_entity = OVAWQAlgoAlgoBackend(model)
            self._backend_entity.name_to_node_mapping = self.name_to_node_mapping
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
            return model

        target_node_names = []
        merge_node_names = []
        awq_data = {}
        name_mapping = {wp.weight_name: idx for idx, wp in enumerate(self._all_weight_params)}

        for match in matches:
            nncf_node = graph.get_node_by_key(match[-1])
            for weight_op_friendly_name, _ in self._backend_entity.get_weight_names_and_port_ids(nncf_node, graph):
                target_node_names.append(weight_op_friendly_name)

            nncf_node = graph.get_node_by_key(match[0])
            for weight_op_friendly_name, _ in self._backend_entity.get_weight_names_and_port_ids(nncf_node, graph):
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

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            config = wp.compression_config

            stats = self._activations[k]
            X = fns.stack([fns.mean(stat, axis=0) for stat in stats])
            X = fns.transpose(X)

            s = fns.max(fns.abs(X), axis=1)

            if X.shape[1] > self._subset_size:
                lens = [stat.shape[0] for stat in stats]
                idxs = [i[0] for i in sorted(enumerate(lens), key=lambda x:-x[1])][:self._subset_size]
                X = X[:, idxs]

            top_k = max(int(s.shape[0] * self._percent_to_apply), 1)
            topk_idxs = fns.argsort(-s)[:top_k]

            groups_to_correct = set()
            for idx in topk_idxs:
                groups_to_correct.add(idx.data // config.group_size)

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
                    cur_diff = fns.mean(fns.abs(cur_out - fp32_out))
                    if cur_diff < min_diff:
                        min_diff = cur_diff
                        best_scale = cur_scale
                    alpha += alpha_step

                if best_scale is not None:
                    scale.data[offset : offset + config.group_size] = best_scale.data

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

            # update activations for next usage
            a_scale_t = fns.transpose(a_scale)
            for stat in self._activations[k]:
                stat = stat * a_scale_t

            for _, port_id in self._backend_entity.get_weight_names_and_port_ids(merge_node, graph):
                merge_weight = self._backend_entity.get_weight(merge_node, port_id, model, graph)
                merge_weight = merge_weight * a_scale
                self._backend_entity.set_weight(merge_node, port_id, model, graph, merge_weight)

        return model

    def apply_scale_correction(
        self,
        model: TModel,
        graph: NNCFGraph
    ) -> TModel:
        """
        Applies the algorithm to the model.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :return: A resulting model.
        """
        name_mapping = {wp.node_with_weight.node_name: idx for idx, wp in enumerate(self._all_weight_params)}

        for k, activations in track(self._activations.items(), description="Applying Scale Selection"):
            wp = self._all_weight_params[name_mapping[k]]
            reduction_axis = wp.reduction_axes[0]
            config = wp.compression_config
            if config.num_bits != 4:
                continue
            
            cur_config = deepcopy(config)
            cur_config.group_size = -1
            
            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            split = int(0.7 * len(activations))
            X = fns.stack([fns.mean(stat, axis=0) for stat in activations[:split]])
            X = fns.transpose(X) #[d_in, seq_len]

            if X.shape[1] > self._subset_size:
                lens = [stat.shape[0] for stat in activations[:split]]
                idxs = [i[0] for i in sorted(enumerate(lens), key=lambda x:-x[1])][:self._subset_size]
                X = X[:, idxs]

            X_cnt = fns.stack([fns.mean(stat, axis=0) for stat in activations[split:]])
            X_cnt = fns.transpose(X_cnt)#[d_in, seq_len]

            if X_cnt.shape[1] > self._subset_size:
                lens = [stat.shape[0] for stat in activations[split:]]
                idxs = [i[0] for i in sorted(enumerate(lens), key=lambda x:-x[1])][:self._subset_size]
                X_cnt = X_cnt[:, idxs]

            weight = self._backend_entity.get_weight(
                wp.node_with_weight, weight_port_id, model, graph
            )

            if reduction_axis == 0:
                weight = fns.transpose(weight)
                reduction_axis = 1

            original_weight = fns.zeros_like(weight) + weight

            g_compressed_weighs, g_c_scale, g_c_zp = do_integer_quantization(
                original_weight, reduction_axis, config
            )

            q_weights = do_dequantization(g_compressed_weighs, g_c_scale, g_c_zp, reduction_axis)

            fp_out = fns.matmul(original_weight, X) # [d_out, seq_len]
            fp_out_cnt = fns.matmul(original_weight, X_cnt) # [d_out, seq_len]

            q_out = fns.matmul(q_weights, X_cnt)
            diff_before = fns.mean(fns.abs(fp_out_cnt - q_out))

            s = fns.max(fns.abs(X), axis=1)
            s = fns.unsqueeze(s, 0)
            s, _ = reshape_weight_for_grouped_quantization(s, reduction_axis, config.group_size)

            original_weight, _ = reshape_weight_for_grouped_quantization(original_weight, reduction_axis, config.group_size)
            www = fns.abs(original_weight)
            www = 0.0 * www + 1.0
            ww = www * s

            target = g_compressed_weighs.astype(dtype=g_c_scale.dtype) - g_c_zp
            zero_mask = g_compressed_weighs == g_c_zp

            ww = fns.where(zero_mask, 0.0, ww)

            denum = fns.sum(ww, axis=2, keepdims=True)
            ww = ww / denum

            scaled_weight = original_weight #/ g_c_scale


            X, _ = reshape_weight_for_grouped_quantization(X, 0, config.group_size)
            q_weights, _ = reshape_weight_for_grouped_quantization(q_weights, reduction_axis, config.group_size)
            best_diffs = None
            result_scale = None
            eps = fns.finfo(weight).eps
            # fp_outs = []
            # for si in range(X.shape[0]):
            #     fp_out = fns.matmul(original_weight[:, si, :], X[si, :, :])
            #     fp_outs.append(fp_out)

            fp_outs = fns.matmul(fns.transpose(original_weight, (1, 0, 2)), X)
            q_outs = fns.matmul(fns.transpose(q_weights, (1, 0, 2)), X)
            min_max_scale_diffs = fns.mean((fp_outs - q_outs)**2, axis=-1)
            min_max_scale_diffs = fns.transpose(min_max_scale_diffs, (1, 0))
            ideal_scale_diffs = fns.zeros_like(min_max_scale_diffs)

            for algo_iters in range(5):
                ideal_scale = fns.abs(scaled_weight) / (fns.abs(target) + 0.0000000000000001)
                ideal_scale = fns.where(zero_mask, eps, ideal_scale)
                
                weighted_scale = ideal_scale * ww

                near_to_ideal_scale = fns.sum(weighted_scale, axis=2, keepdims=True)

                # scaled_weights = original_weight / near_to_ideal_scale + g_c_zp
                # compressed_weights = fns.round(scaled_weights)
                # compressed_weights = fns.clip(compressed_weights, level_low, level_high).astype(TensorDataType.uint8)
                
                # RETURN BACK
                compressed_weights, _, _ = do_integer_quantization(original_weight, -1, cur_config, near_to_ideal_scale)
                q_weights_ = do_dequantization(compressed_weights, near_to_ideal_scale, g_c_zp)

                
                # min_max_scale_diffs = []
                # ideal_scale_diffs = []

                q_outs = fns.matmul(fns.transpose(q_weights_, (1, 0, 2)), X)
                ideal_scale_diffs = fns.mean((fp_outs - q_outs)**2, axis=-1)
                ideal_scale_diffs = fns.transpose(ideal_scale_diffs, (1, 0))
                 
                # for si in range(X.shape[0]):
                #     # [N/g, g, n_samples]
                #     for o_dim in q_weights_.shape[0]:
                #         tmp = np.dot(q_weights_.data[o_dim, si, :], X.data[si])
                #         ideal_scale_diffs[o_dim, si] = 
                #     # q_out_ = fns.matmul(q_weights_[:, si, :], X[si, :, :])
                #     # tmp = q_outs[si, :, :]
                #     # q_diff = fns.mean(fns.abs(tmp - q_out_))
                #     # ideal_scale_diffs.append(fns.mean((fp_outs[si] - q_out_)**2, axis=1))

                #     # if best_diffs is None:
                #     #     q_out = fns.matmul(q_weights[:, si, :], X[si, :, :])
                #     #     min_max_scale_diffs.append(fns.mean((fp_outs[si] - q_out)**2, axis=1))
                    
                
                #ideal_scale_diffs = fns.stack(ideal_scale_diffs, axis=1)
                if best_diffs is None:
                    #min_max_scale_diffs = fns.stack(min_max_scale_diffs, axis=1)
                    best_diffs = min_max_scale_diffs
                
                mask = ideal_scale_diffs>best_diffs
                
                best_diffs = fns.where(mask , best_diffs, ideal_scale_diffs)

                mask = fns.unsqueeze(mask, axis=2)

                if result_scale is None:
                    near_to_ideal_scale = fns.where(mask , g_c_scale, near_to_ideal_scale)
                else:
                    near_to_ideal_scale = fns.where(mask , result_scale, near_to_ideal_scale)
                result_scale = near_to_ideal_scale

                scaled_weights = original_weight / near_to_ideal_scale + g_c_zp
                # compressed_weights = fns.round(scaled_weights)
                # compressed_weights = fns.clip(compressed_weights, level_low, level_high).astype(TensorDataType.uint8)
                compressed_weights, _, _ = do_integer_quantization(original_weight, -1, cur_config, near_to_ideal_scale)
                target = compressed_weights.astype(dtype=g_c_scale.dtype) - g_c_zp


            for scale_steps in range(10):
                scale = 1.5 - 0.1 * scale_steps
                scaled_scale = scale * g_c_scale

                # scaled_weights = original_weight / scaled_scale + g_c_zp
                # compressed_weights = fns.round(scaled_weights)
                # compressed_weights = fns.clip(compressed_weights, level_low, level_high).astype(TensorDataType.uint8)
                
                # RETURN BACK                
                compressed_weights, _, _ = do_integer_quantization(original_weight, -1, cur_config, scaled_scale)
                q_weights_ = do_dequantization(compressed_weights, near_to_ideal_scale, g_c_zp)
                #q_weights_ = q_weights

                target = compressed_weights.astype(dtype=g_c_scale.dtype) - g_c_zp
                
                ideal_scale = fns.abs(scaled_weight) / (fns.abs(target) + eps)
                ideal_scale = fns.where(zero_mask, eps, ideal_scale)
                weighted_scale = ideal_scale * ww
                near_to_ideal_scale = fns.sum(weighted_scale, axis=2, keepdims=True)

                compressed_weights, _, _ = do_integer_quantization(original_weight, -1, cur_config, near_to_ideal_scale)
                q_weights_ = do_dequantization(compressed_weights, near_to_ideal_scale, g_c_zp)

                q_outs = fns.matmul(fns.transpose(q_weights_, (1, 0, 2)), X)
                ideal_scale_diffs = fns.mean((fp_outs - q_outs)**2, axis=-1)
                ideal_scale_diffs = fns.transpose(ideal_scale_diffs, (1, 0))
                
                mask = ideal_scale_diffs>best_diffs
                
                best_diffs = fns.where(mask , best_diffs, ideal_scale_diffs)

                mask = fns.unsqueeze(mask, axis=2)

                if result_scale is None:
                    near_to_ideal_scale = fns.where(mask , g_c_scale, near_to_ideal_scale)
                else:
                    near_to_ideal_scale = fns.where(mask , result_scale, near_to_ideal_scale)
                result_scale = near_to_ideal_scale

            wp.precomputed_scale = result_scale
        return model

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        return StatisticPointsContainer()
