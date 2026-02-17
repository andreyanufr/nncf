import torch
import torch.nn as nn
import torch.nn.functional as F


def get_reciprocal(tensor):
    """
    Memory-frugal reciprocal:
    - Inplace operations on original tensor
    - Only allocates small boolean mask
    """
    eps = 1e-5 if tensor.dtype == torch.float16 else 1e-30

    # Create mask for very small elements (small overhead)
    mask = tensor.abs() < eps

    # Prepare output in place: reuse tensor if allowed, otherwise create once
    recip = torch.empty_like(tensor)

    # Safe reciprocal: for nonzero elements
    nonzero_mask = ~mask
    recip[nonzero_mask] = 1.0 / tensor[nonzero_mask]

    # Zero out elements below threshold
    recip[mask] = 0.0

    return recip


class CodebookWrapperLinear(torch.nn.Module):
    def __init__(
        self,
        orig_layer,
        group_size: int = 32,
        signed_scale: bool = False,
        n_bits: int = 2,
        use_exp_for_scale: bool = True
    ):
        super().__init__()
        
        assert isinstance(orig_layer, torch.nn.Linear), "Only linear layers are supported"
        assert orig_layer.bias is None, "Bias is not supported in this example"

        self.orig_layer = orig_layer
        self.group_size = group_size
        self.signed_scale = signed_scale
        self.n_bits = n_bits
        self.use_exp_for_scale = use_exp_for_scale
        
        if n_bits == 2:
            initial_codebook = torch.tensor([-1.0, -0.25,  0.25, 1.0], dtype=orig_layer.weight.dtype).to(orig_layer.weight.device)
        else:
            initial_codebook = torch.tensor([i for i in range(-2 ** (n_bits - 1) + 1, 2 ** (n_bits - 1) + 1)], dtype=orig_layer.weight.dtype).to(orig_layer.weight.device) / (2 ** (n_bits - 1))
        
        
        self.codebook = torch.nn.Parameter(initial_codebook, requires_grad=True)

        self.init_indexes_and_scale()
        
        self.orig_layer.weight.requires_grad = False  # Freeze original weight, only update codebook and scale
        self.orig_layer.to('cpu')
        self.orig_layer.weight.to('cpu')
        
        print("")

    
    @torch.no_grad()
    def init_indexes_and_scale(self):
        # reshape weight to (out_features, in_features // group_size, group_size)
        weight = self.orig_layer.weight.data
        out_features, in_features = weight.shape
        weight = weight.view(out_features, in_features // self.group_size, self.group_size)
        
        # calculate scale and indexes
        scale = weight.abs().max(dim=2, keepdim=True)[0]
        scale[scale < 1e-10] = 1e-5
        if self.use_exp_for_scale:
            scale = torch.log(scale)

        self.scale = torch.nn.Parameter(scale, requires_grad=True)
        
        self.update_indexes()
    
    
    @torch.no_grad()
    def update_indexes(self):
        weight = self.orig_layer.weight.data
        out_features, in_features = weight.shape
        weight = weight.view(out_features, in_features // self.group_size, self.group_size)

        if self.use_exp_for_scale:
            iscale = get_reciprocal(self.scale.exp())
        else:
            iscale = get_reciprocal(self.scale)
        
        iscale = iscale * self.codebook.abs().max()  # Scale is relative to codebook range, so we can multiply by max codebook value to get better numerical stability
        self.indexes = torch.argmin(torch.abs(weight.unsqueeze(3) * iscale.unsqueeze(3) - self.codebook), dim=3).to(torch.uint8)

    #     self.update_one_hot()
    
    @torch.no_grad()
    def update_one_hot(self):
        pass
        #self.one_hot = torch.nn.functional.one_hot(self.indexes, num_classes=2 ** self.n_bits).to(self.codebook.device).to(self.codebook.dtype)

        # if True:
        #     tmp = self.codebook * self.one_hot
        #     tmp = tmp.mean(dim=3)
        #     tmp = tmp * self.scale.exp()
        #     out_features, in_features = self.orig_layer.weight.shape
        #     weight = tmp.view(out_features, in_features)
            
        #     diff = (weight - self.orig_layer.weight).abs().mean()
        #     print(f"Mean absolute difference between original and reconstructed weight: {diff.item():.6f}")

    def dequantize_weight(self, memory_saving: bool = True):
        if memory_saving:
            flat_indexes = self.indexes.reshape(-1).long()
            weight = self.codebook.index_select(0, flat_indexes).view_as(self.indexes)
        else:
            one_hot = F.one_hot(self.indexes.long(), num_classes=2 ** self.n_bits).to(self.codebook.device).to(self.codebook.dtype)
            weight = (self.codebook * one_hot).sum(dim=3)

        if self.use_exp_for_scale:
            weight = weight * self.scale.exp()
        else:
            weight = weight * self.scale

        out_features, in_features = self.orig_layer.weight.shape
        weight = weight.view(out_features, in_features)
        return weight
        
            
    def forward(self, x):
        weight = self.dequantize_weight()
        
        return F.linear(x, weight)


    @torch.no_grad()
    def dequantize(self):
        return self.dequantize_weight()


def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
    setattr(module, name_list[-1], new_module)


@torch.no_grad()
def update_indexes(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, CodebookWrapperLinear):
            module.update_indexes()


def wrap_model(model: nn.Module, n_bits: int = 2) -> nn.Module:
    skip = 5
    for i, layer in enumerate(model.model.layers):
        if i < skip or i > len(model.model.layers) - skip - 1:  # Skip first 2 layers and layers after 5 for faster example, can be removed for full training
            continue
        print(f"Wrapping layer {i} with CodebookWrapperLinear")
        model.model.layers[i] = wrap_model_block(layer, n_bits=n_bits)
    return model

def unwrap_model(model: nn.Module) -> nn.Module:
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = unwrap_model_block(layer)
    return model

def wrap_model_block(model: nn.Module, n_bits: int = 2) -> nn.Module:
    """
    Wraps the linear layers of the model with CodebookWrapperLinear for adaptive codebook compression.

    :param model: The original model to be wrapped.
    :param n_bits: The number of bits for quantization.

    :param adaptive_codebook: A boolean flag indicating whether to use adaptive codebook compression.
    :return: The wrapped model with CodebookWrapperLinear layers.
    """
    
    changed_modules = {}

    for name, module in model.named_modules():
        if 'lm_head' in name or 'v_proj' in name or 'down_proj' in name:
            continue
        # if not 'k_proj' in name:
        #     continue
        if isinstance(module, nn.Linear):
            changed_modules[name] = module
    
    for name, module in changed_modules.items():
        print(f"Wrapping layer {name} with CodebookWrapperLinear")
        set_module(model, name, CodebookWrapperLinear(module, n_bits=n_bits))
    return model


def unwrap_model_block(model: nn.Module) -> nn.Module:
    """
    Unwraps the CodebookWrapperLinear layers in the model back to their original linear layers.

    :param model: The model with CodebookWrapperLinear layers to be unwrapped.
    :return: The unwrapped model with original linear layers.
    """
    changed_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, CodebookWrapperLinear):
            module.orig_layer.weight.data.copy_(module.dequantize())
            changed_modules[name] = module.orig_layer
    for name, orig_layer in changed_modules.items():
        set_module(model, name, orig_layer)
    return model
