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
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.reference import ReferenceQuantizedFunctions as RQ


@pytest.mark.parametrize("registred", list(QUANTIZATION_MODULES.registry_dict.items()))
def test_quantizer_layers_accepts_return_type(registred):
    mode, quantizer_cls = registred

    actual_input = torch.range(0, 10)
    input_ = torch.return_types.max((actual_input, actual_input))

    quantizer_spec = PTQuantizerSpec(
        num_bits=8,
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


def test_symmetric_lora_quantizer_negative_scale_gradient():
    """
    Verifies that a symmetric LoRA quantizer differentiates its negative scale through both quantization bounds.
    """
    input_ = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
    qspec = PTQuantizerSpec(
        num_bits=3,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        narrow_range=False,
        half_range=False,
        scale_shape=(1, 1),
        logarithm_scale=False,
    )
    lspec = PTLoraSpec(lora_rank=1, orig_weight_shape=list(input_.shape), weight_shape=list(input_.shape))
    lora_quantizer = SymmetricLoraQuantizer(qspec, lspec)

    with torch.no_grad():
        lora_quantizer.scale.fill_(-1.0)
        lora_quantizer.lora_A.zero_()
        lora_quantizer.lora_B.zero_()

    lora_quantizer(input_).sum().backward()
    scale = lora_quantizer.scale.detach()
    input_low = -scale / lora_quantizer.level_low * lora_quantizer.level_high
    input_range = torch.abs((2 + 1 / lora_quantizer.level_low) * scale)
    _, grad_input_low, grad_input_range = RQ.Quantize_backward(
        torch.ones_like(input_),
        input_,
        input_low,
        input_range,
        lora_quantizer.levels,
        lora_quantizer.level_low,
        lora_quantizer.level_high,
    )
    expected_gradient = grad_input_low * (-lora_quantizer.level_high / lora_quantizer.level_low)
    expected_gradient -= grad_input_range * (2 + 1 / lora_quantizer.level_low)

    torch.testing.assert_close(lora_quantizer.scale.grad, expected_gradient.float())
