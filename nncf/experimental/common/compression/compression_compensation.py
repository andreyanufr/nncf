from typing import Dict, List, Optional

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.openvino.graph.transformations.commands import OVTargetPoint, TargetType
from nncf.openvino.statistics.collectors import OVMeanPerChanelReducer, TensorCollector, OVMeanTensorStatistic, OVRawTensorStatistic, OVNoopReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from scipy.optimize import least_squares
from functools import partial

sz = 1000

def func_sym(params, x, y):
    return np.mean((x * params[0] + params[1] - y)**2)



def get_noop_statistic_collector(
    num_samples: int, channel_axis: int, window_size: Optional[int] = None, inplace: bool = True
):
    """
    Mean statistic collector builder.

    :param num_samples: Maximum number of samples to collect.
    :param channel_axis: Channel axis to use during reduction phase.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :param inplace: Whether the mean reducer should be calculated inplace or out of place.
    :return: Mean statistic collector.
    """
    # TODO(dlyakhov): use inplace OVBatchMeanReducer and OVMeanPerChanelReducer
    # after migration on openvino-dev=2023.0
    inplace = False
    reducer = OVMeanPerChanelReducer(channel_axis=channel_axis, inplace=inplace)#OVNoopReducer() #

    aggregate_mean = NoopAggregator(num_samples)

    collector = TensorCollector(OVMeanTensorStatistic)
    collector.register_statistic_branch(OVMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    
    # collector = TensorCollector(OVRawTensorStatistic)
    # collector.register_statistic_branch(OVMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    return collector

def mean_statistic_collector(
    channel_axis: int,
    inplace: bool,
    num_samples: Optional[int] = None,
    window_size: Optional[int] = None,
):
    return get_noop_statistic_collector(num_samples, channel_axis, window_size, inplace)


def get_statistic_points(model, graph, nodes, subset_size) -> StatisticPointsContainer:
    statistic_container = StatisticPointsContainer()
    OUTPUT_PORT_OF_NODE = 0

    # Collection of statistics after layers where biases will be corrected.
    for node in nodes:
        node_name = node.node_name
        channel_axis = node.metatype.output_channel_axis
        if channel_axis is None:
            channel_axis = -1

        # For layers with weights, there is only one output port - 0.
        statistic_point = OVTargetPoint(
            TargetType.POST_LAYER_OPERATION, node_name, port_id=OUTPUT_PORT_OF_NODE
        )
        stat_collector = mean_statistic_collector(
            channel_axis=channel_axis, num_samples=subset_size, inplace=False
        )
        statistic_container.add_statistic_point(
            StatisticPoint(
                target_point=statistic_point, tensor_collector=stat_collector, algorithm="compensation"
            )
        )

    return statistic_container


def compression_compensation(model, dataset, compression_algorithm, subset_size=128):
    # find last MatMul
    for op in model.get_ordered_ops():
        print(op.get_type_name(), op.get_friendly_name())
    
    matmuls = [op for op in model.get_ordered_ops() if "aten::rsqrt" in op.get_friendly_name()]
    #matmuls = [op for op in model.get_ordered_ops() if op.get_type_name() == "MatMul"]
    assert len(matmuls) != 0
    last_matmul = matmuls[-8]

    # insert identity Multiply and Add
    port_id = 0 # ????
    target_node_output = last_matmul.input_value(port_id)

    sz = target_node_output.partial_shape.get_dimension(2).max_length
    
    mul = opset.multiply(target_node_output, np.ones(sz, dtype=np.float32), name="NormalizerMul")
    add = opset.add(mul, np.zeros(sz, dtype=np.float32), name="NormalizerAdd")
    last_matmul.input(port_id).replace_source_output(add.output(0))

    graph = GraphConverter.create_nncf_graph(model)
    statistics_aggregator = OVStatisticsAggregator(dataset)
    
    node_keys = graph.get_all_node_keys()
    target_name = [val for val in node_keys if 'NormalizerMul' in val][0]
    target_mul = graph.get_node_by_key(target_name)

    statistic_points = get_statistic_points(model, graph, [target_mul], subset_size)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(model, graph)
    
    x_fp = list(statistics_aggregator.statistic_points['NormalizerMul'][0].algorithm_to_tensor_collectors["compensation"][0].aggregators.values())[0]._container
    
    # statistics_aggregator.statistic_points ???

    compressed_model = compression_algorithm(model)
    
    graph = GraphConverter.create_nncf_graph(compressed_model)
    statistics_aggregator = OVStatisticsAggregator(dataset)
    statistic_points = get_statistic_points(model, graph, [target_mul], subset_size)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(compressed_model, graph)
    
    x_q = list(statistics_aggregator.statistic_points['NormalizerMul'][0].algorithm_to_tensor_collectors["compensation"][0].aggregators.values())[0]._container
    
    x_fp = np.vstack(x_fp)
    x_q = np.vstack(x_q)
    
    # x_fp = np.concatenate(x_fp, axis=1).squeeze()
    # x_q = np.concatenate(x_q, axis=1).squeeze()
    
    mul_data = np.ones(sz, dtype=np.float32)
    add_data = np.zeros(sz, dtype=np.float32)
    y_fp = x_fp
    mean_residual = 0.0
    for i in range(x_q.shape[1]): # shape is [seq_len, data_dim]
        A = x_q[:, i]
        
        # coeff = np.array([1.0, 0.0])#np.random.randn(2)
        # res = least_squares(partial(func_sym, x=A, y=y_fp[:, i]), coeff)

        ones = np.ones_like(A)
        A = np.stack((A, ones), axis=1)
        B = np.expand_dims(y_fp[:, i], 1)
        solution = np.linalg.lstsq(A, B)
        gamma_beta = solution[0]
        residual = solution[1]
        mean_residual += np.mean(residual)
        mul_data[i] = gamma_beta[0]
        add_data[i] = gamma_beta[1]

    print("Mean residual:", mean_residual / x_q.shape[1])
    print("Mean residual/mean target:", (mean_residual / x_q.shape[1]) / np.mean(np.abs(y_fp)))
    # th = np.quantile(np.abs(add_data), 0.9)
    # modifications = 0

    # for i in range(sz):
    #     if abs(add_data[i]) > th:
    #         modifications += 1
    #         add_data[i] = np.sign(add_data[i]) * th
    print('Mul: ', np.min(mul_data), np.max(mul_data), np.mean(mul_data))
    print('Add: ', np.min(add_data), np.max(add_data), np.mean(np.abs(add_data)))
    
    OVModelTransformer._set_const_value(mul, 1, mul_data)
    OVModelTransformer._set_const_value(add, 1, add_data)
    
    return compressed_model
