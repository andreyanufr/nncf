# modified by SINQ authors 2025

import copy
from typing import Union, Optional

import torch
from torch import nn, Tensor
import gemlite

try:
    import gemlite
    has_gemlite = True
    gemlite.set_autotune("fast")
    gemlite.set_kernel_caching(True)
    print('found gemlite installation')

except:
    has_gemlite = False

class QLinear(nn.Module):
    def __init__(
        self,
        linear_layer: Union[nn.Module, None],
        quant_config: Optional[dict] = None,
        compute_dtype: torch.dtype = torch.float16,
        w_quantized: Optional[Tensor] = None,
        wscale: Optional[Tensor] = None,
        zero: Optional[Tensor] = None,
        ascale: Optional[Tensor] = torch.tensor(1.0),
    ):
        super().__init__()

        self.bias = None
        self.axis = None
        self.channel_wise = None
        self.device = linear_layer.weight.device if linear_layer is not None else torch.device('cpu')
        self.compute_dtype = compute_dtype
        self.quant_config = copy.deepcopy(quant_config) if quant_config is not None else None

        self.linear_layer = linear_layer
        # self.w_quantized = w_quantized
        # self.wscale = wscale
        # self.zero = zero
        self.ascale = ascale
        
        if self.quant_config["group_size"] == None:
            self.quant_config["group_size"] = (
                self.linear_layer.in_features
            )
   
        self.gemlite_linear = gemlite.GemLiteLinear(self.quant_config['nbits'], 
                                                    self.quant_config['group_size'], 
                                                    self.linear_layer.in_features,
                                                    self.linear_layer.out_features, 
                                                    input_dtype=gemlite.DType.FP16, 
                                                    output_dtype=gemlite.DType.FP16)


        bias = None if self.linear_layer.bias is None else self.linear_layer.bias.clone().to(device=self.device, dtype=self.compute_dtype)  
        # print(W_q.shape, self.linear_layer.weight.data.shape)
        self.gemlite_linear.pack(w_quantized.to(torch.uint8), wscale, zero, bias)
        
        
        # for name, param in self.linear_layer.named_parameters():
        #     setattr(self.linear_layer, name, None)
        del self.linear_layer.weight
        del self.linear_layer
        # torch.cuda.empty_cache()
            
        #del self.linear_layer
        torch.cuda.empty_cache()


    def forward(self, x:Tensor) -> Tensor:
        out = self.gemlite_linear(self.ascale * x)
        return out

    
    def _meta_to_cpu(self, meta: dict) -> dict:
        if meta is None:
            return None

        def to_cpu(v):
            import torch
            if isinstance(v, torch.Tensor):
                return v.detach().cpu()
            if isinstance(v, dict):
                return {k: to_cpu(vi) for k, vi in v.items()}
            if isinstance(v, (list, tuple)):
                # Detect quantAux 4-tuple: (x, s, m, shape)
                if (
                    len(v) == 4
                    and isinstance(v[0], torch.Tensor)
                    and isinstance(v[1], torch.Tensor)
                    and isinstance(v[2], torch.Tensor)
                ):
                    x, s, m, shape = v
                    return {
                        "x": to_cpu(x),
                        "s": to_cpu(s),
                        "m": to_cpu(m),
                        "shape": list(shape),  # JSON-friendly
                    }
                return [to_cpu(e) for e in v]
            return v

        return {k: to_cpu(v) for k, v in meta.items()}

    def state_dict(self, destination=None, prefix: str = '', keep_vars: bool = False):
        """
        Export quantized tensors for saving:
          - W_q (Tensor)
          - bias (Tensor or omitted if None)
          - meta (dict; tensors moved to CPU)
        """
        sd = {}
        if self.w_quantized is not None:
            sd["w_quantized"] = self.w_quantized.detach().cpu()
        if self.bias is not None:
            sd["bias"] = self.bias.detach().cpu()
        if self.wscale is not None:
            sd["wscale"] = self.wscale.detach().cpu()
        if self.zero is not None:
            sd["zero"] = self.zero.detach().cpu()
        if self.ascale is not None:
            sd["ascale"] = self.ascale.detach().cpu()
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Restore pre-quantized tensors without re-quantizing.
        Assumes self.device / self.compute_dtype are set by the caller.
        """
        # Required
        self.w_quantized = state_dict["w_quantized"].to(device=self.device)
        self.wscale = state_dict["wscale"].to(device=self.device)
        self.zero = state_dict["zero"].to(device=self.device)
        self.ascale = state_dict.get("ascale", torch.tensor(1.0)).to(device=self.device)

        # Optional bias
        b = state_dict.get("bias", None)
        self.bias = b.to(device=self.device, dtype=self.compute_dtype) if b is not None else None

        # Infer features for nicer repr and possible use elsewhere
        if isinstance(self.meta, dict) and "shape" in self.meta:
            out_f, in_f = self.meta["shape"]  # meta stores (out_features, in_features)
            self.in_features, self.out_features = in_f, out_f

        self.ready = True

        # Match nn.Module API return
        from torch.nn.modules.module import _IncompatibleKeys
        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])


def sinq_base_quant_config(
    nbits: int = 4,
    group_size: int = 64,
    quant_zero: bool = False,
    quant_scale: bool = False,
    view_as_float: bool = False,
    axis: int = 1,
    tiling_mode: str = '1D',
    method: str = 'dual',
):
    assert (
        nbits in Quantizer.SUPPORTED_BITS
    ), "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
    if method == "asinq":
        # Remap sinq_awq_l1_quantAux to behave like asinq (A-SINQ in the paper)
        method = "sinq_awq_l1_quantAux"
    elif method == "sinq":
        # Remap so that users can use sinq_quantAux as sinq (scales and zeros are quantized to 8-bit)
        method = "sinq_quantAux"
    if group_size is not None:
        assert is_divisible(
            group_size, 8
        ), "Invalid group_size param: the value should be a multiple of 8."

    weight_quant_params = {
        "nbits": nbits,
        "group_size": group_size,
        "round_zero": True if nbits == 4 else False,
        "axis": axis,
        "view_as_float": view_as_float,
        "tiling_mode": tiling_mode,
        "method": method,
    }

    if quant_zero or quant_scale:
        print(
            colored(
                "Warning: Quantized meta-data is deprecated and will be removed. It is not supported for quantized model serialization.",
                "yellow",
            )
        )

    scale_quant_params = (
        {"nbits": 8, "group_size": 128}
        if (quant_scale)
        else None
    )
    zero_quant_params = (
        {"nbits": 8, "group_size": None}
        if (quant_zero)
        else None
    )

    return {
        "weight_quant_params": weight_quant_params,
        "scale_quant_params": scale_quant_params,
        "zero_quant_params": zero_quant_params
    }


# Alias: follow similar Auto-GPTQ naming
BaseQuantizeConfig = sinq_base_quant_config
