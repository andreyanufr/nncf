# simple wraper for quantized linear layer
# packs integer values with (6, 8) bits to 8 bit values by one per byte and (2, 3, 4) bits to 4 bit values by two per byte
# the wrapper is used to support quantized linear layers with 6, 8 bits for converting PT models OV
# symmetric and assymetric weight saved in uint format, but for symmtric quantization the zero point is fixed and equal to2**(bit_width - 1)


import torch


def pack_uint4(tensor):
    #packed_tensor = tensor.contiguous()
    packed_tensor = tensor.reshape(tensor.shape[0], -1, 2)
    packed_tensor = torch.bitwise_and(packed_tensor[..., ::2], 15) | packed_tensor[..., 1::2] << 4
    packed_tensor = packed_tensor.squeeze()
    return packed_tensor


def unpack_uint4(packed_tensor):
    res = torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1)
    return res.reshape(res.shape[0], -1)
    #return torch.stack((torch.bitwise_and(packed_tensor, 15), packed_tensor >> 4), dim=-1)


def get_compressed_weight(shape, bits, group_size=-1, symmetric=True):
    if bits <= 8 and bits > 1:
        qweight = torch.randint(0, 2**bits, shape, dtype=torch.uint8)
        for i in range(qweight.shape[0]):
            qweight[i, :] = i % (2**bits)  # to have different values in each row for testing
        if group_size > 0:
            qweight = qweight.view(shape[0], -1, group_size)
        qzeros = torch.tensor(2**(bits - 1), dtype=torch.uint8) if symmetric else qweight.min(dim=-1, keepdim=True)[0]
        scale = torch.ones((shape[0], shape[1] // group_size if group_size > 0 else 1), dtype=torch.float16)
        return qweight, qzeros, scale
    else:
        raise NotImplementedError(f"Unsupported bit width: {bits}")



class NNCFQLinear(torch.nn.Module):
    """A wrapper for `torch.nn.Linear` that allows replacing it with a single
    operation during model tracing.

    This class is used in NNCF quantization to replace `torch.nn.Linear` with
    a single operation that represents the quantized linear layer. It is not
    intended for direct use by users.
    """

    def __init__(self, qweight=None, qzeros=None, scales=None, bits=None, group_size=None, sym=False, bias=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bits = bits
        self.group_size = group_size if group_size > 0 else 0
        self.sym = sym
        self.bias = bias
        
        self.out_features = qweight.shape[0]
        self.in_features = qweight.shape[1] * self.group_size if self.group_size > 0 else qweight.shape[1]
        #self.register_buffer('bits', torch.tensor(bits))
        #self.register_buffer('group_size', torch.tensor(group_size))
        #self.register_buffer('sym', torch.tensor(sym))

    def unpack_weights(self):
        if self.bits in (6, 8):
            return self.qweight
        elif self.bits in (2, 3, 4):
            return unpack_uint4(self.qweight)
        else:
            raise NotImplementedError(f"Unsupported bit width: {self.bits}")

    def forward(self, x):
        weight = self.unpack_weights().to(x.dtype).to(x.device)
        if self.qzeros is not None:
            zeros = self.qzeros.float().to(x.dtype).to(x.device)
            weight = weight - zeros
        weight = weight * self.scales
        return torch.matmul(x, weight)


def create_nncf_qlinear(shape, bits, group_size=-1, symmetric=True):
    qweight, qzeros, scales = get_compressed_weight(shape, bits, group_size, symmetric)
    if bits in (2, 3, 4):
        qweight = pack_uint4(qweight)

    qlinear = NNCFQLinear(qweight=qweight, qzeros=qzeros, scales=scales, bits=bits, group_size=group_size, sym=symmetric)
    return qlinear
