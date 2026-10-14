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

import pytest
import torch

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import PTLoraNLSSpec
from nncf.torch.quantization.layers import PTLoraSpec
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricLoraSigmoidQuantizer


@pytest.mark.parametrize("registred", list(QUANTIZATION_MODULES.registry_dict.items()))
def test_quantizer_layers_accepts_return_type(registred):
    mode, quantizer_cls = registred

    actual_input = torch.range(0, 10)
    input_ = torch.return_types.max((actual_input, actual_input))

    num_bits = 2 if mode == QuantizationMode.SYMMETRIC_LORA_SIGMOID else 8
    quantizer_spec = PTQuantizerSpec(
        num_bits=num_bits,
        mode=mode,
        signedness_to_force=True,
        narrow_range=True,
        half_range=True,
        scale_shape=(1,),
        logarithm_scale=False,
    )
    if mode not in [QuantizationMode.ASYMMETRIC, QuantizationMode.SYMMETRIC]:
        shape = actual_input.unsqueeze(dim=0).shape
        lora_spec = PTLoraSpec(0, shape, shape)
        if mode in [QuantizationMode.ASYMMETRIC_LORA_NLS, QuantizationMode.SYMMETRIC_LORA_NLS]:
            lora_spec = PTLoraNLSSpec(0, 0, shape, shape)
        quantizer = quantizer_cls(quantizer_spec, lora_spec)
    else:
        quantizer = quantizer_cls(quantizer_spec)

    visited = False

    def check_types(fn):
        def wrapped(x: torch.Tensor):
            assert isinstance(x, torch.Tensor)
            nonlocal visited
            visited = True
            return fn(x)

        return wrapped

    quantizer._forward_impl = check_types(quantizer._forward_impl)
    quantizer(input_)
    assert visited


def test_symmetric_lora_sigmoid_quantizer():
    weight_shape = [2, 1, 4]
    quantizer = SymmetricLoraSigmoidQuantizer(
        PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationMode.SYMMETRIC_LORA_SIGMOID,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(2, 1, 1),
            logarithm_scale=False,
        ),
        PTLoraSpec(lora_rank=1, orig_weight_shape=[2, 4], weight_shape=weight_shape),
        temperature=0.5,
    )
    with torch.no_grad():
        quantizer.scale.copy_(torch.tensor([[[2.0]], [[-2.0]]]))
        quantizer.threshold_1.copy_(torch.tensor([[[0.1]], [[0.2]]]))
        quantizer.threshold_2.copy_(torch.tensor([[[0.3]], [[0.4]]]))
        quantizer.threshold_3.copy_(torch.tensor([[[0.5]], [[0.6]]]))
        quantizer.lora_A.zero_()
        quantizer.lora_B.zero_()

    weights = torch.arange(-4, 4, dtype=torch.float32).reshape(2, 4).requires_grad_()
    actual = quantizer.quantize(weights)
    grouped_weights = weights.reshape(weight_shape)
    effective_weights = torch.where(quantizer.scale < 0, -grouped_weights, grouped_weights)
    expected = quantizer.scale * (
        -torch.sigmoid((effective_weights + quantizer.threshold_1.square() + quantizer.threshold_2.square()) / 0.5)
        - torch.sigmoid((effective_weights + quantizer.threshold_2.square()) / 0.5)
        + torch.sigmoid((effective_weights - quantizer.threshold_3.square()) / 0.5)
    )
    torch.testing.assert_close(actual, expected.reshape_as(weights))

    actual.sum().backward()
    assert quantizer.threshold_1.grad is not None
    assert quantizer.threshold_2.grad is not None
    assert quantizer.threshold_3.grad is not None
    assert quantizer.lora_A.grad is not None
    assert quantizer.lora_B.grad is not None


def test_symmetric_lora_sigmoid_quantizer_initializes_thresholds_from_scale():
    quantizer = SymmetricLoraSigmoidQuantizer(
        PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationMode.SYMMETRIC_LORA_SIGMOID,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(2, 1, 1),
            logarithm_scale=False,
        ),
        PTLoraSpec(lora_rank=1, orig_weight_shape=[2, 4], weight_shape=[2, 1, 4]),
    )
    with torch.no_grad():
        quantizer.scale.copy_(torch.tensor([[[4.0]], [[-16.0]]]))
    quantizer.initialize_thresholds()

    torch.testing.assert_close(quantizer.threshold_1, torch.tensor([[[2**0.5]], [[8**0.5]]]))
    torch.testing.assert_close(quantizer.threshold_2, torch.tensor([[[1.0]], [[2.0]]]))
    torch.testing.assert_close(quantizer.threshold_3, torch.tensor([[[1.0]], [[2.0]]]))
