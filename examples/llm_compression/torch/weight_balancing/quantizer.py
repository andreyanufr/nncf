import torch
from dataclasses import dataclass
from typing import Literal, Optional
from qlinear import QLinear
from tqdm import tqdm
from torch import vmap

@dataclass
class QuantizationConfig:
    """Configuration for weight quantization."""
    nbits: int = 8
    sym: bool = True
    balancing_method: Literal['sinq', 'absmean'] = 'sinq'
    group_size: Optional[int] = 64

    def __post_init__(self):
        if self.nbits not in Quantizer.SUPPORTED_BITS:
            raise ValueError(f"nbits must be one of {Quantizer.SUPPORTED_BITS}, got {self.nbits}")
        if self.balancing_method not in ['sinq', 'absmean']:
            raise ValueError(f"balancing_method must be 'sinq' or 'absmean', got {self.balancing_method}")
    
    def to_dict(self):
        """Convert config to dictionary format for backward compatibility."""
        return {
            'nbits': self.nbits,
            'sym': self.sym,
            'balancing_method': self.balancing_method,
            'group_size': self.group_size
        }


class WeightsBalancer():
    def __init__(self, quant_config):
        if isinstance(quant_config, QuantizationConfig):
            self.quant_config = quant_config.to_dict()
        else:
            self.quant_config = quant_config
    
    @staticmethod
    def sinkhorn_log(matrix,
                 order=8,
                 clip_min=1e-3,
                 clip_max=1e3,
                 eps=1e-6,
                 stop_on_increasing_imbalance=True):
        """
        vmap-friendly Sinkhorn that returns *the* mu1 / mu2 corresponding
        to the matrix with the minimal imbalance encountered during the
        iteration.

        The return value is a tuple
            (scaled_matrix, mu1_at_minimum, mu2_at_minimum)
        """
        dtype = torch.float32
        m = matrix.to(dtype)
        dev = m.device
        measure = torch.std

        def imbalance(mat):
            s1, s2 = measure(mat, 1), measure(mat, 0)
            s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
            s_max = torch.maximum(s1.max(), s2.max())
            return s_max / s_min          # scalar

        imb_min = torch.tensor(float('inf'), dtype=dtype, device=dev)
        gate    = torch.tensor(0.0, dtype=dtype, device=dev)

        tgt_small = torch.minimum(
            m.std(1).clamp(clip_min, clip_max).min(),
            m.std(0).clamp(clip_min, clip_max).min()
        ) + eps

        log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=dev)
        log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=dev)

        # Known-good candidates for the step k=0
        cur0          = m
        ib0           = imbalance(cur0)
        imb_min       = torch.minimum(imb_min, ib0)
        mu1_star      = log_mu1.exp().clone()
        mu2_star      = log_mu2.exp().clone()

        for _ in range(order):
            cur       = (m / log_mu1.exp()) / log_mu2.exp()
            ib        = imbalance(cur)

            # update the best-so-far candidates
            better    = (ib <= imb_min).to(dtype)   # 1 if new best
            imb_min   = torch.min(imb_min, ib)
            mu1_star  = torch.where(better.bool(), log_mu1.exp(), mu1_star)
            mu2_star  = torch.where(better.bool(), log_mu2.exp(), mu2_star)

            # early-exit condition
            if stop_on_increasing_imbalance:
                rising = (ib > imb_min).to(dtype)
                gate   = torch.clip(gate + rising, max=1.0)   # once 1 → always 1

            # still-running samples update the dual variables
            g  = 1.0 - gate

            std_r  = measure(cur, 1).clamp(clip_min, clip_max)
            std_c  = measure(cur,0).clamp(clip_min, clip_max)

            sal_col = (std_c / tgt_small).clamp(0.7, 2.0).log()
            sal_row = (std_r[:, None] / tgt_small).clamp(0.7, 2.0).log()

            log_mu1 = (log_mu1 + (sal_col * g)).clip(-.3, 10.)
            log_mu2 = (log_mu2 + (sal_row * g)).clip(-.3, 10.)

        # final scaled matrix and the recorded best scaling vectors
        scaled = m / mu1_star / mu2_star
        return scaled, mu1_star.unsqueeze(0), mu2_star
    
    
    @staticmethod
    def abs_mean(matrix,
                 eps=1e-6):
        dtype = torch.float32
        m = matrix.to(dtype)
        dev = m.device
        
        mu1_star = m.abs().mean(0) + eps
        mu2_star = torch.ones(m.shape[0], 1, dtype=torch.dtype, device=dev)

        # final scaled matrix and the recorded best scaling vectors
        scaled = m / mu1_star / mu2_star
        return scaled, mu1_star.unsqueeze(0), mu2_star

   
    def balance(self, matrix, method='sinq'):
        dev = matrix.device
        matrix = matrix.float()
        
        if method == 'sinq':
            matrix, mu1, mu2 = self.sinkhorn_log(matrix, 16)
        elif method == 'absmean':
            matrix, mu1, mu2 = self.abs_mean(matrix)
        else:
            raise NotImplementedError(f'Unknown balancing method: {method}')

        return matrix, mu1.to(dev), mu2.to(dev)
    
    
    def balance_groupped(self, matrix, method='sinq', block=64):
        q = self.balance
        mshape = matrix.shape
        H,W = matrix.shape
        assert W%block==0, 'block must divide W'
        n_w = W//block

        matrix = matrix.view(H, W//block, block)
        M_batched = matrix.permute(1,0,2).contiguous().view(n_w, H, block)


        def process_block(mat):
            return q(mat, method) 
        Q, s1, s2 = vmap(process_block, randomness='different')(M_batched)
        
        del M_batched
        torch.cuda.empty_cache()

        Q = Q.permute(1,0,2).reshape(-1, block)
        s2 = s2.permute(1,0,2).reshape(-1,1)
        s1 = s1.permute(1,0,2).view(1, -1)
        return Q, s1, s2


class Quantizer:
    SUPPORTED_BITS = [2, 4, 8]
    
    def __init__(self, quant_config):
        if isinstance(quant_config, QuantizationConfig):
            self.quant_config = quant_config.to_dict()
            self._config_obj = quant_config
        elif isinstance(quant_config, dict):
            self.quant_config = quant_config
            self._config_obj = QuantizationConfig(**quant_config)
        else:
            raise TypeError(f"quant_config must be QuantizationConfig or dict, got {type(quant_config)}")
        self.balancer = WeightsBalancer(self.quant_config)

    @staticmethod
    def quantize(weight: torch.Tensor, quant_config: dict):
        # return quantized_weight, scale, zero
        if quant_config['group_size'] is not None:
            weight = weight.view(-1, quant_config['group_size'])

        if quant_config['sym']:
            scale = torch.max(torch.abs(weight)) / (2 ** (quant_config['nbits'] - 1) - 1)
            zero = torch.zeros(1, device=weight.device)
            q_weight = torch.clamp(torch.round(weight / scale), -(2 ** (quant_config['nbits'] - 1)), 2 ** (quant_config['nbits'] - 1) - 1).to(torch.int8)
        else:
            _min = weight.min(axis=1, keepdim=True)[0]
            _max = weight.max(axis=1, keepdim=True)[0]

            max_v = round(2**quant_config['nbits'] - 1)
            min_v = 0

            # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
            denom = (_max - _min)
            scale = (max_v / denom)
            scale = torch.where(denom.abs() <= 1e-4, torch.full_like(scale, 1.0), scale) #Avoid small denom values
            scale = 1.0 / scale.clamp(max=2e4) # clamp to avoid half-precision problems
            zero = -_min / scale
            zero = torch.round(zero)
            
            q_weight = torch.clamp(torch.round(weight / scale) + zero, min_v, max_v).to(torch.uint8)
            
            del _min, _max, denom
            torch.cuda.empty_cache()

        return q_weight, scale, zero
    
    def quantize_linear_layers(self, linear_layers: list[torch.nn.Linear], quant_config: dict):
        super_weighst = [layer.weight.data for layer in linear_layers]
        split_dims = [layer.out_features for layer in linear_layers]
        mul = super_weighst[0].shape[1] // quant_config['group_size']
        split_dims = [s * mul for s in split_dims]
        # for i in range(1, len(split_dims)):
        #     split_dims[i] += split_dims[i - 1]
        # split_dims = split_dims[:-1]

        super_weights = torch.cat(super_weighst, dim=0).to('cuda')
        super_weights, mu1, mu2 = self.balancer.balance_groupped(super_weights, method=self.quant_config.get('balancing_method', 'sinq'), block=self.quant_config.get('group_size', 64))
        
        w_quantized, wscale, zero = Quantizer.quantize(super_weights, quant_config)
        
        del super_weighst
        torch.cuda.empty_cache()

        wscale = wscale * mu2
        
        w_quantized = torch.split(w_quantized, split_dims, dim=0)
        wscale = torch.split(wscale, split_dims, dim=0)
        zero = torch.split(zero, split_dims, dim=0)
        
        return w_quantized, wscale, zero, mu1, mu2


    def quantize_llama(self, model):
        for module in tqdm(model.model.layers, desc="Quantizing LLaMA layers"):
            device = next(module.parameters()).device
            if not hasattr(module, 'self_attn') or not hasattr(module, 'mlp'):
                raise NotImplementedError("Only LLaMA model structure is supported in quantize_llama.")

            # self attention
            self_attn = module.self_attn
            dtype = module.input_layernorm.weight.data.dtype
            
            
            # w_quantized, wscale, zero, mu1, mu2 = self.quantize_linear_layers(
            #     [self_attn.o_proj],
            #     self.quant_config
            # )
            # self_attn.o_proj = QLinear(
            #     self_attn.o_proj,
            #     quant_config=self.quant_config,
            #     w_quantized=w_quantized[0],
            #     wscale=wscale[0],
            #     zero=zero[0],
            #     ascale=mu1.unsqueeze(0).to(dtype)
            # ).to(device)
            #self_attn.v_proj.weight.data = self_attn.v_proj.weight.data * mu1.view(-1, 1).to(device).to(dtype)
            
            w_quantized, wscale, zero, mu1, mu2 = self.quantize_linear_layers(
                [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj],
                self.quant_config
            )
            
            # linear_layer: Union[nn.Module, None],
            # quant_config: Optional[dict] = None,
            # compute_dtype: torch.dtype = torch.float16,
            # w_quantized: Optional[Tensor] = None,
            # wscale: Optional[Tensor] = None,
            # zero: Optional[Tensor] = None,
            # ascale: Optional[Tensor] = torch.tensor(1.0),
        
            self_attn.q_proj = QLinear(
                self_attn.q_proj,
                quant_config=self.quant_config,
                w_quantized=w_quantized[0],
                wscale=wscale[0],
                zero=zero[0],
            ).to(device)
            
            self_attn.k_proj = QLinear(
                self_attn.k_proj,
                quant_config=self.quant_config,
                w_quantized=w_quantized[1],
                wscale=wscale[1],
                zero=zero[1],
            ).to(device)
            
            self_attn.v_proj = QLinear(
                self_attn.v_proj,
                quant_config=self.quant_config,
                w_quantized=w_quantized[2],
                wscale=wscale[2],
                zero=zero[2],
            ).to(device)
            
            module.input_layernorm.weight.data = module.input_layernorm.weight.data * mu1.view(-1).to(device).to(dtype)
            
            
            mlp = module.mlp
            
            w_quantized, wscale, zero, mu1, mu2 = self.quantize_linear_layers(
                [mlp.down_proj],
                self.quant_config
            )
            
            mlp.down_proj = QLinear(
                mlp.down_proj,
                quant_config=self.quant_config,
                w_quantized=w_quantized[0],
                wscale=wscale[0],
                zero=zero[0],
            ).to(device)
            
            # merge scale to previsous matmul
            mlp.up_proj.weight.data = mlp.up_proj.weight.data * mu1.view(-1, 1).to(device).to(dtype)
            
            
            w_quantized, wscale, zero, mu1, mu2 = self.quantize_linear_layers(
                [mlp.up_proj, mlp.gate_proj],
                self.quant_config
            )
            
            mlp.up_proj = QLinear(
                mlp.up_proj,
                quant_config=self.quant_config,
                w_quantized=w_quantized[0],
                wscale=wscale[0],
                zero=zero[0],
            ).to(device)
            
            mlp.gate_proj = QLinear(
                mlp.gate_proj,
                quant_config=self.quant_config,
                w_quantized=w_quantized[1],
                wscale=wscale[1],
                zero=zero[1],
            ).to(device)
            
            module.post_attention_layernorm.weight.data = module.post_attention_layernorm.weight.data * mu1.view(-1).to(device).to(dtype)

            torch.cuda.empty_cache()
            
            
            
