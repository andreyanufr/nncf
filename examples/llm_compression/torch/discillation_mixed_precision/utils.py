from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.parameters import StripFormat
from nncf.torch import load_from_config
from nncf.torch.function_hook.wrapper import get_hook_storage


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    model = load_from_config(model, ckpt["nncf_config"])
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


class LinearINT4(nn.Module):
    """
    A linear layer that simulates 4-bit quantization by scaling the weights and activations to the range of int4 values.
    This is a simplified version for demonstration purposes and does not perform actual quantization. The weights are scaled to the range of [-8, 7] and the activations are scaled to the range of [0, 15].
    """

    def __init__(self, model: nn.Linear):
        super().__init__()
        if not isinstance(model, nn.Linear):
            raise ValueError("LinearINT4 can only be applied to nn.Linear modules.")
        self.model_int4 = model

    def forward(self, x):
        return self.model_int4(x)


class LinearMIXER(nn.Module):
    """
    class for mixing 4-bit and 2-bit quantization in the same linear layer. The output channels are split into two parts, one part is quantized to 4 bits and the other part is quantized to 2 bits.
    The ratio of the split is determined by the `ratio` parameter. The weights of the original linear layer are copied to the two new linear layers according to the split.
    The forward pass concatenates the outputs of the two linear layers. The decision on which part of the output channels to quantize to 4 bits is based on a heuristic that checks if the first layers are more sensitive to quantization.
    Args:
        nn (_type_): _description_
    """

    def __init__(self, model: nn.Module, ratio=0.5):
        super().__init__()
        if not isinstance(model, nn.Linear):
            raise ValueError("LinearMIXER can only be applied to nn.Linear modules.")
        self.ratio = ratio
        self.int4_first = self.if_first_layers_more_sensitive(model.weight.data)

        dim_div = int(model.out_features * self.ratio)
        out_channel_1 = dim_div
        out_channel_2 = model.out_features - dim_div

        device = model.weight.data.device
        dtype = model.weight.data.dtype
        bias = model.bias is not None

        if self.int4_first:
            self.model_int4 = nn.Linear(model.in_features, out_channel_1, bias=bias, device=device, dtype=dtype)
            self.model_int2 = nn.Linear(model.in_features, out_channel_2, bias=bias, device=device, dtype=dtype)

            self.model_int4.weight.data.copy_(model.weight.data[:out_channel_1])
            self.model_int2.weight.data.copy_(model.weight.data[out_channel_1:])
            if model.bias is not None:
                self.model_int4.bias.data.copy_(model.bias.data[:out_channel_1])
                self.model_int2.bias.data.copy_(model.bias.data[out_channel_1:])
        else:
            self.model_int4 = nn.Linear(model.in_features, out_channel_2, bias=bias, device=device, dtype=dtype)
            self.model_int2 = nn.Linear(model.in_features, out_channel_1, bias=bias, device=device, dtype=dtype)

            self.model_int4.weight.data.copy_(model.weight.data[out_channel_1:])
            self.model_int2.weight.data.copy_(model.weight.data[:out_channel_1])
            if model.bias is not None:
                self.model_int4.bias.data.copy_(model.bias.data[out_channel_1:])
                self.model_int2.bias.data.copy_(model.bias.data[:out_channel_1])

    def forward(self, x):
        if self.int4_first:
            return torch.cat([self.model_int4(x), self.model_int2(x)], dim=-1)
        return torch.cat([self.model_int2(x), self.model_int4(x)], dim=-1)

    # def if_first_layers_more_sensitive(self, weight: Tensor) -> bool:
    #     # This is a heuristic to determine if the first layers are more sensitive to quantization.
    #     # It checks if the standard deviation of the weights in the first half of the output dimension is greater than the second half.
    #     out_dim = int(weight.shape[0] * self.ratio)

    #     first_half_mean = weight[:out_dim].std().item()
    #     second_half_mean = weight[-out_dim:].std().item()
    #     print(f"First half mean: {first_half_mean}, Second half mean: {second_half_mean}")
    #     return first_half_mean > second_half_mean

    def if_first_layers_more_sensitive(
        self,
        weight: Tensor,
        *,
        eps: float = 1e-6,
        rel_margin: float = 0.05,
    ) -> bool:
        """
        Decide whether the first output-channel block is more quantization-sensitive
        than the last block of the same size.

        Sensitivity per output channel is approximated by the ratio
            max(|w|) / (mean(|w|) + eps)
        which captures outlier-driven quantization error far better than a single std.
        The two blocks are compared via the mean of the top-k channel scores
        (robust to a handful of extreme channels). A relative margin avoids
        flip-flopping on near-ties.

        :param weight: 2D weight tensor of shape ``[out_features, in_features]``.
        :param eps: Numerical stabilizer for the per-channel ratio.
        :param rel_margin: Minimum relative gap required to declare the first block
            more sensitive; otherwise returns ``False`` (deterministic tie-break).
        :return: ``True`` if the first block is deemed more sensitive.
        """
        out_features = weight.shape[0]
        block = int(out_features * self.ratio)
        if block <= 0 or block >= out_features:
            raise ValueError(f"Invalid block size {block} for out_features {out_features} and ratio {self.ratio}")

        w = weight.detach().float().abs()
        # Per-channel outlier score: peak-to-average ratio.
        per_channel = w.amax(dim=1) / (w.mean(dim=1) + eps)

        first = per_channel[:block]
        last = per_channel[-block:]

        # Robust aggregation: average of top-k (k = 10% of the block, at least 1).
        k = max(1, block // 10)
        first_score = torch.topk(first, k).values.mean()
        last_score = torch.topk(last, k).values.mean()

        # Dead-zone to suppress noise-level flips.
        return (first_score - last_score).item() > rel_margin * last_score.item()

    def to_linear(self) -> nn.Linear:
        """
        Converts the LinearMIXER back to a standard nn.Linear layer by concatenating the weights and biases of the two linear layers.
        The output channels are ordered according to the original order in the input linear layer.
        Returns:
            nn.Linear: The converted nn.Linear layer with the same output features as the original input linear layer.
        """
        out_features = self.model_int4.out_features + self.model_int2.out_features
        device = self.model_int4.weight.data.device
        dtype = self.model_int4.weight.data.dtype
        bias = self.model_int4.bias is not None

        linear_layer = nn.Linear(self.model_int4.in_features, out_features, bias=bias, device=device, dtype=dtype)

        if self.int4_first:
            linear_layer.weight.data.copy_(torch.cat([self.model_int4.weight.data, self.model_int2.weight.data], dim=0))
            if bias:
                linear_layer.bias.data.copy_(torch.cat([self.model_int4.bias.data, self.model_int2.bias.data], dim=0))
        else:
            linear_layer.weight.data.copy_(torch.cat([self.model_int2.weight.data, self.model_int4.weight.data], dim=0))
            if bias:
                linear_layer.bias.data.copy_(torch.cat([self.model_int2.bias.data, self.model_int4.bias.data], dim=0))
        return linear_layer


def replace_linear_with_mixer(model: nn.Module, parent_name="", ratio=0.5, n_layers=-1) -> nn.Module:
    """
    Recursively replaces all nn.Linear modules in the given model with LinearMIXER modules.
    First v_proj, mlp and last mlp always in 4 bit.

    Args:
        model (nn.Module): The input model to be modified.
        ratio (float): The ratio of output channels to be quantized to 4 bits in the LinearMIXER. Default is 0.5.

    Returns:
        nn.Module: The modified model with nn.Linear modules replaced by LinearMIXER modules.
    """
    if n_layers == -1 and hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers - 1
    for name, module in model.named_children():
        if "lm_head" in name:
            continue
        full_name = parent_name + "." + name if parent_name else name
        if isinstance(module, nn.Linear):
            if ("v_proj" in name or "mlp" in parent_name) and ('.0.' in parent_name or f'.{n_layers}.' in parent_name):
                print(f"Replacing {full_name} with LinearINT4")
                setattr(model, name, LinearINT4(module))
            else:
                print(f"Replacing {full_name} with LinearMIXER")
                setattr(model, name, LinearMIXER(module, ratio))
        else:
            replace_linear_with_mixer(module, full_name, ratio=ratio, n_layers=n_layers)
    return model


def replace_mixer_with_linear(model: nn.Module) -> nn.Module:
    """
    Recursively replaces all LinearMIXER modules in the given model with nn.Linear modules.

    Args:
        model (nn.Module): The input model to be modified.
    """
    for name, module in model.named_children():
        if isinstance(module, LinearMIXER):
            setattr(model, name, module.to_linear())
        elif isinstance(module, LinearINT4):
            setattr(model, name, module.model_int4)
        else:
            replace_mixer_with_linear(module)
    return model


@torch.no_grad()
def export_to_pytorch(pretrained: str, ckpt_file: Path, model_dir: Path) -> None:
    """
    Create a wrapper of OpenVINO model from the checkpoint for evaluation on CPU via WWB.

    :param pretrained: The name or path of the pretrained model.
    :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
    :param last_dir: The directory where the OpenVINO model will be saved.
    :return: A wrapper of OpenVINO model ready for evaluation.
    """
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16, device_map="cpu")
    model_to_eval = replace_linear_with_mixer(model_to_eval)
    model_to_eval = load_checkpoint(model_to_eval, ckpt_file)

    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.IN_PLACE)
    model_to_eval = replace_mixer_with_linear(model_to_eval)

    model_to_eval.save_pretrained(model_dir)


@torch.no_grad()
def load_to_pytorch(pretrained: str, ckpt_file: Path) -> None:
    """
    Create a wrapper of OpenVINO model from the checkpoint for evaluation on CPU via WWB.

    :param pretrained: The name or path of the pretrained model.
    :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
    :param last_dir: The directory where the OpenVINO model will be saved.
    :return: A wrapper of OpenVINO model ready for evaluation.
    """
    torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch_dtype, device_map="cuda")
    model_to_eval = replace_linear_with_mixer(model_to_eval)
    model_to_eval = load_checkpoint(model_to_eval, ckpt_file)
    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.IN_PLACE)

    return model_to_eval, tokenizer
