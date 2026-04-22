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


import torch


def pack_uint4(tensor):
    *others, last_dim = tensor.shape
    packed_tensor = tensor.reshape(*others, -1, 2)
    packed_tensor = torch.bitwise_and(packed_tensor[..., ::2], 15) | packed_tensor[..., 1::2] << 4
    packed_tensor = packed_tensor.squeeze()
    return packed_tensor


def unpack_uint4(packed_tensor):
    *shape, last_dim = packed_tensor.shape
    res = torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1)
    return res.reshape(*shape, -1)


def get_compressed_weight(shape, bits, group_size=-1, symmetric=True):
    if bits <= 8 and bits > 1:
        qweight = torch.randint(0, 2**bits, shape, dtype=torch.uint8)
        for i in range(qweight.shape[0]):
            qweight[i, :] = i % (2**bits)  # to have different values in each row for testing
        if group_size > 0:
            n_groups = shape[1] // group_size
            qweight = qweight.view(shape[0], n_groups, group_size)
            qzeros = (
                torch.tensor(2 ** (bits - 1), dtype=torch.uint8) if symmetric else qweight.min(dim=-1, keepdim=True)[0]
            )
            scale = torch.ones((shape[0], n_groups, 1), dtype=torch.float16)
        else:
            qzeros = (
                torch.tensor(2 ** (bits - 1), dtype=torch.uint8) if symmetric else qweight.min(dim=-1, keepdim=True)[0]
            )
            scale = torch.ones((shape[0], 1), dtype=torch.float16)
        return qweight, qzeros, scale
    err = f"Unsupported bit width: {bits}"
    raise NotImplementedError(err)


class NNCFQLinear(torch.nn.Module):
    """
    A wrapper for `torch.nn.Linear` that allows replacing it with a single
    operation during model tracing.

    This class is used in NNCF quantization to replace `torch.nn.Linear` with
    a single operation that represents the quantized linear layer. It is not
    intended for direct use by users.
    """

    def __init__(
        self, qweight=None, qzeros=None, scales=None, bits=None, group_size=None, sym=False, bias=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bits = bits
        self.group_size = group_size if group_size > 0 else 0
        self.sym = sym
        self.bias = bias

        self.out_features = qweight.shape[0]
        if bits in (2, 3, 4):
            self.in_features = qweight.shape[1] * 2  # packed: two values per byte
        else:
            self.in_features = qweight.shape[1]

    def unpack_weights(self):
        if self.bits in (6, 8):
            return self.qweight
        if self.bits in (2, 3, 4):
            return unpack_uint4(self.qweight)
        err = f"Unsupported bit width: {self.bits}"
        raise NotImplementedError(err)

    def forward(self, x):
        weight = self.unpack_weights().to(x.dtype).to(x.device)
        if self.group_size > 0:
            if self.qzeros is not None:
                zeros = self.qzeros.float().to(x.dtype).to(x.device)
                weight = weight - zeros
            weight = weight * self.scales.to(x.dtype).to(x.device)
            weight = weight.view(self.out_features, -1)
        else:
            if self.qzeros is not None:
                zeros = self.qzeros.float().to(x.dtype).to(x.device)
                weight = weight - zeros
            weight = weight * self.scales.to(x.dtype).to(x.device)
        return torch.matmul(x, weight.T)


def create_nncf_qlinear(shape, bits, group_size=-1, symmetric=True):
    qweight, qzeros, scales = get_compressed_weight(shape, bits, group_size, symmetric)
    if bits in (2, 3, 4):
        qweight = pack_uint4(qweight)

    qlinear = NNCFQLinear(
        qweight=qweight, qzeros=qzeros, scales=scales, bits=bits, group_size=group_size, sym=symmetric
    )
    return qlinear


if __name__ == "__main__":
    print("=== Testing NNCFQLinear ===")
    test_configs = [
        # (shape, bits, group_size, symmetric)
        ((8, 64), 4, 32, True),
        ((8, 64), 4, 32, False),
        ((8, 64), 4, -1, True),
        ((8, 64), 4, -1, False),
        ((8, 64), 8, -1, True),
        ((8, 64), 8, -1, False),
        ((8, 64), 8, 32, True),
        ((8, 64), 8, 32, False),
        ((8, 64), 3, -1, True),
        ((8, 64), 3, 32, True),
        ((8, 64), 2, -1, True),
        ((8, 64), 2, 32, True),
    ]
    for shape, bits, group_size, symmetric in test_configs:
        tag = f"bits={bits}, group_size={group_size}, sym={symmetric}"
        try:
            qlinear = create_nncf_qlinear(shape, bits, group_size, symmetric)
            x = torch.randn(1, shape[1], dtype=torch.float16)
            y = qlinear(x)
            assert y.shape == (1, shape[0]), f"Expected (1, {shape[0]}), got {y.shape}"
            print(f"  PASS: {tag} -> out={y.shape}")
        except Exception as e:
            print(f"  FAIL: {tag} -> {e}")
    print("=== Done ===")
