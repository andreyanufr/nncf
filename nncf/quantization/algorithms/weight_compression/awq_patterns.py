# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.common.graph.patterns import GraphPattern
from nncf.common.utils.registry import Registry
from nncf.openvino.graph.metatypes import openvino_metatypes as om

AWQ_PATTERNS = Registry("awq")

# BLOCK PATTERNS


@AWQ_PATTERNS.register("MatMul_Mul_MatMul")
def create_matmul_mul_matmul() -> GraphPattern:
    pattern = GraphPattern()
    linear_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "LINEAR", GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})
    mul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    linear_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "LINEAR", GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})

    pattern.add_edge(linear_node_1, mul_node)
    pattern.add_edge(mul_node, linear_node_2)
    return pattern


def get_awq_patterns():
    return AWQ_PATTERNS.registry_dict
