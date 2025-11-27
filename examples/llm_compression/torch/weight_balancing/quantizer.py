import torch
from dataclasses import dataclass
from typing import Literal, Optional
from tqdm import tqdm
from torch import vmap
import torch.nn as nn
import gc
from compressed_tensors.utils import match_named_modules



from qlinear import QLinear
from fusing_patterns import FUSING_PATTERN_REGISTRY


def remove_non_model_tensors(model):
    keys = set()
    for name, param in model.named_parameters():
        keys.add(param.data_ptr())

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.data_ptr() not in keys:
                    del obj
        except:
            pass
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class QuantizationConfig:
    """Configuration for weight quantization."""
    nbits: int = 8
    sym: bool = True
    balancing_method: Literal['sinq', 'absmean', 'none'] = 'sinq'
    group_size: Optional[int] = 64

    def __post_init__(self):
        if self.nbits not in Quantizer.SUPPORTED_BITS:
            raise ValueError(f"nbits must be one of {Quantizer.SUPPORTED_BITS}, got {self.nbits}")
        if self.balancing_method not in ['sinq', 'absmean', 'none']:
            raise ValueError(f"balancing_method must be 'sinq', 'absmean', or 'none', got {self.balancing_method}")
    
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
        mu2_star = torch.ones(m.shape[0], 1, dtype=dtype, device=dev)

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
        elif method == 'none':
            mu1 = torch.ones(1, matrix.shape[1], device=dev, dtype = torch.float32)
            mu2 = torch.ones(matrix.shape[0], 1, device=dev, dtype = torch.float32)
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
        s1 = s1.permute(1,0,2).contiguous().view(1, -1)
        return Q, s1, s2

# Finds the parent of a node module named "name"
def find_parent(model, name: str) -> nn.Module:
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent

class Quantizer:
    SUPPORTED_BITS = [2, 4, 8, 16]
    
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
            min_v = -(2 ** (quant_config['nbits'] - 1))
            max_v = 2 ** (quant_config['nbits'] - 1) - 1
            scale = torch.abs(weight).max(dim=1, keepdim=True)[0] / (-min_v)
            zero = None #torch.zeros(1, device=weight.device)
            zero = torch.zeros_like(scale) - min_v
            q_weight = (torch.clamp(torch.round(weight / scale), min_v, max_v) + zero).to(torch.uint8)
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
    
    def quantize_linear_layers(self, linear_layers: list[torch.nn.Linear], quant_config: dict, activations: torch.Tensor = None):
        super_weighst = [layer.weight.data for layer in linear_layers]
        split_dims = [layer.out_features for layer in linear_layers]
        mul = super_weighst[0].shape[1] // quant_config['group_size']
        split_dims = [s * mul for s in split_dims]
        
        dtype = super_weighst[0].dtype
        # for i in range(1, len(split_dims)):
        #     split_dims[i] += split_dims[i - 1]
        # split_dims = split_dims[:-1]

        super_weights = torch.cat(super_weighst, dim=0).to('cuda')
        super_weights, mu1, mu2 = self.balancer.balance_groupped(super_weights, method=self.quant_config.get('balancing_method', 'sinq'), block=self.quant_config.get('group_size', 64))
        
        w_quantized, wscale, zero = Quantizer.quantize(super_weights, quant_config)
        w_quantized = w_quantized

        del super_weighst
        torch.cuda.empty_cache()

        wscale = wscale * mu2
        
        wscale = wscale.to(dtype)
        
        w_quantized = torch.split(w_quantized, split_dims, dim=0)
        wscale = torch.split(wscale, split_dims, dim=0)
        if zero is not None:
            zero = zero.to(dtype)
            zero = torch.split(zero, split_dims, dim=0)
        else:
            zero = [None] * len(w_quantized)
        
        del mu2
        torch.cuda.empty_cache()
        
        return w_quantized, wscale, zero, mu1, None


    def quantize_llama(self, model):
        for module in tqdm(model.model.layers, desc="Quantizing LLaMA layers"):
            device = next(module.parameters()).device
            if not hasattr(module, 'self_attn') or not hasattr(module, 'mlp'):
                raise NotImplementedError("Only LLaMA model structure is supported in quantize_llama.")

            # self attention
            self_attn = module.self_attn
            dtype = module.input_layernorm.weight.data.dtype
            
            
            w_quantized, wscale, zero, mu1, _ = self.quantize_linear_layers(
                [self_attn.o_proj],
                self.quant_config
            )
            # no place to merge scale, so we add it as ascale
            self_attn.o_proj = QLinear(
                self_attn.o_proj,
                quant_config=self.quant_config,
                w_quantized=w_quantized[0],
                wscale=wscale[0],
                zero=zero[0],
                ascale=mu1.unsqueeze(0).to(dtype)
            ).to(device)

            
            w_quantized, wscale, zero, mu1, _ = self.quantize_linear_layers(
                [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj],
                self.quant_config
            )

        
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
            
            w_quantized, wscale, zero, mu1, _ = self.quantize_linear_layers(
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
            
            
            w_quantized, wscale, zero, mu1, _ = self.quantize_linear_layers(
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
        self.cleanup(model)


    def quantize_per_layer(self, model):
        tmp_mapping = {}
        for name, module in model.model.named_modules():
            if (type(module) is torch.nn.Linear) and ('lm_head' not in name):
                tmp_mapping[name] = module
        
        device = next(model.model.parameters()).device

        for name in tqdm(tmp_mapping.keys(), desc="Quantizing per-layer"):
            dtype = tmp_mapping[name].weight.data.dtype
            w_quantized, wscale, zero, mu1, _ = self.quantize_linear_layers(
                [tmp_mapping[name]],
                self.quant_config
            )
            
            qlinear = QLinear(
                tmp_mapping[name],
                quant_config=self.quant_config,
                w_quantized=w_quantized[0],
                wscale=wscale[0],
                zero=zero[0],
                ascale=mu1.unsqueeze(0).to(dtype)
            ).to(device)
            
            setattr(
                find_parent(model.model, name),
                name.split(".")[-1],
                qlinear,
            )
            
            torch.cuda.empty_cache()
        self.cleanup(model)
    
    
    
    def quantize_by_patterns(self, model):
        class_name = model.__class__.__name__
        if class_name not in FUSING_PATTERN_REGISTRY:
            raise ValueError(f"No fusing patterns registered for model class {class_name}")
        
        patterns = FUSING_PATTERN_REGISTRY[class_name]  
        
        for pattern in tqdm(patterns, desc="Quantizing by patterns"):
            layers_for_fusing = [[layer_for_fusing_name, layer_for_fusing] for layer_for_fusing_name, layer_for_fusing in match_named_modules(model, [pattern.layer_for_fusing])]
            scaled_layers = [[scaled_layer_names, layers] for scaled_layer_names, layers in match_named_modules(model, pattern.scaled_layers)]
            
            step = len(pattern.scaled_layers)
            scaled_layers = [scaled_layers[x:x+step] for x,_ in list(enumerate(scaled_layers))[::step]]
            
            for layer_for_fusing, scaled_layer_group in zip(layers_for_fusing, scaled_layers):
                print(f"Fusing {layer_for_fusing[0]} with {[name for name, _ in scaled_layer_group]}")
                
                
                w_quantized, wscale, zero, mu1, _ = self.quantize_linear_layers(
                    [layer for _, layer in scaled_layer_group],
                    self.quant_config
                )
                dtype = layer_for_fusing[1].weight.data.dtype
                device = next(layer_for_fusing[1].parameters()).device
                
                if type(layer_for_fusing[1]) is not torch.nn.Linear: # probably layer norm
                    layer_for_fusing[1].weight.data = layer_for_fusing[1].weight.data * mu1.view(-1).to(device).to(dtype)
                else:
                    mu1 = mu1.view(-1, 1)
                    if 'Phi' in class_name:
                        # merge scale to half of previsous matmul (up_proj in MLP)
                        sz = mu1.shape[0]
                        layer_for_fusing[1].weight.data[sz:, :] = layer_for_fusing[1].weight.data[sz:, :] * mu1.to(device).to(dtype)
                    else:
                        sz = mu1.shape[0]
                        if sz != layer_for_fusing[1].weight.data.shape[0]:
                            print(f"Warning: size mismatch in fusing pattern for {layer_for_fusing[0]}, skipping scale merge. Shapes: mu1 {mu1.shape}, weight {layer_for_fusing[1].weight.data.shape}")
                            continue
                        layer_for_fusing[1].weight.data = layer_for_fusing[1].weight.data * mu1.to(device).to(dtype)
                
                for i, (scaled_layer_name, scaled_layer) in enumerate(scaled_layer_group):
                    qlinear = QLinear(
                        scaled_layer,
                        quant_config=self.quant_config,
                        w_quantized=w_quantized[i],
                        wscale=wscale[i],
                        zero=zero[i],
                    ).to(device)
                    
                    setattr(
                        find_parent(model, scaled_layer_name),
                        scaled_layer_name.split(".")[-1],
                        qlinear,
                    )
        for name, layer in model.model.named_modules():
            if isinstance(layer, nn.Linear):
                print(f"Quantized layer with extra scale: {name}")
                w_quantized, wscale, zero, mu1, _ = self.quantize_linear_layers(
                    [layer],
                    self.quant_config
                )
                # no place to merge scale, so we add it as ascale
                qlinear = QLinear(
                    layer,
                    quant_config=self.quant_config,
                    w_quantized=w_quantized[0],
                    wscale=wscale[0],
                    zero=zero[0],
                    ascale=mu1.unsqueeze(0).to(dtype)
                ).to(device)
                setattr(
                    find_parent(model.model, name),
                    name.split(".")[-1],
                    qlinear,
                )

        torch.cuda.empty_cache()
        self.cleanup(model)
            
        
    def cleanup(self, model):
        remove_non_model_tensors(model)
            
            
            
