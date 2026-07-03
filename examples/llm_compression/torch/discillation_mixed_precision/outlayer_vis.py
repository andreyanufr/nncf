import argparse
import base64
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize outlier activations in LLM linear layers.")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID or local path.")
    parser.add_argument("--output", type=str, default="outlayer_activations.html", help="Output HTML file path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--permute", action="store_true",
        help="Compute and apply outlier permutation to down_proj/up_proj/gate_proj, then compare generation.",
    )
    parser.add_argument("--threshold", type=float, default=6.0, help="Threshold factor for outlier detection.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation check.")
    parser.add_argument("--calibration_samples", type=int, default=32, help="Number of calibration samples from OpenThoughts.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length for calibration chunks.")
    return parser.parse_args()


def create_chain_of_thought_input(tokenizer) -> Dict[str, torch.Tensor]:
    """
    Creates a long chain-of-thought prompt, applies chat template, and tokenizes.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "Solve step by step: A farmer has 3 fields. The first field produces 120 kg of wheat per hectare "
                "and is 5 hectares. The second field produces 95 kg per hectare and is 8 hectares. The third field "
                "produces 110 kg per hectare and is 6 hectares. He sells wheat at $0.30 per kg but has to pay $50 "
                "per hectare in maintenance costs. What is his total profit? Think through each step carefully, "
                "showing all intermediate calculations. Then verify your answer by computing it a different way."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Let me solve this step by step.\n\n"
                "Step 1: Calculate production for each field.\n"
                "- Field 1: 120 kg/ha × 5 ha = 600 kg\n"
                "- Field 2: 95 kg/ha × 8 ha = 760 kg\n"
                "- Field 3: 110 kg/ha × 6 ha = 660 kg\n\n"
                "Step 2: Calculate total production.\n"
                "- Total = 600 + 760 + 660 = 2020 kg\n\n"
                "Step 3: Calculate revenue.\n"
                "- Revenue = 2020 kg × $0.30/kg = $606.00\n\n"
                "Step 4: Calculate total maintenance costs.\n"
                "- Total hectares = 5 + 8 + 6 = 19 ha\n"
                "- Maintenance = 19 ha × $50/ha = $950.00\n\n"
                "Step 5: Calculate profit.\n"
                "- Profit = Revenue - Costs = $606.00 - $950.00 = -$344.00\n\n"
                "The farmer actually has a loss of $344.00.\n\n"
                "Verification using per-field profit:\n"
                "- Field 1: (600 × 0.30) - (5 × 50) = 180 - 250 = -70\n"
                "- Field 2: (760 × 0.30) - (8 × 50) = 228 - 400 = -172\n"
                "- Field 3: (660 × 0.30) - (6 × 50) = 198 - 300 = -102\n"
                "- Total: -70 + (-172) + (-102) = -344 ✓"
            ),
        },
        {
            "role": "user",
            "content": (
                "Now consider: if the farmer could increase yield by 20% on all fields by investing an additional "
                "$30 per hectare, would it be worth it? Also calculate the break-even price per kg of wheat he "
                "would need to charge to make zero profit with the original yields. Show all work step by step."
            ),
        },
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        text = messages[0]["content"]
    inputs = tokenizer(text, return_tensors="pt")
    return inputs


def split_thought_solution(text: str):
    thought_re = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
    solution_re = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)

    thought = thought_re.search(text).group(1).strip()
    solution = solution_re.search(text).group(1).strip()

    return thought, solution

def make_concat_chunks(data, tokenizer, max_length, num_samples, seed=0, add_eos=False, device: torch.device = torch.device("cpu")):
    eos_id = tokenizer.eos_token_id
    token_buffer = []
    chunks = []
    for ex in data:
        ids = tokenizer(ex["text"], return_tensors=None)["input_ids"]
        if add_eos and eos_id is not None:
            ids = ids + [eos_id]
        token_buffer.extend(ids)

        while len(token_buffer) >= max_length:
            chunk = token_buffer[:max_length]
            del token_buffer[:max_length]
            chunks.append(torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0))
            if len(chunks) >= num_samples:
                return chunks

    return chunks

def open_thoughts(tokenizer, train_samples, max_length,
                  shuffle_seed=1234, seed=42, open_thoughts_max_samples=10_000, device: torch.device = torch.device("cpu")):

    tmpl = tokenizer.chat_template
    if tmpl is not None:
        tmpl = tmpl.replace(
            "<think></think>{{render_content(message)}}",
            "{%- set rc = message.get('reasoning_content', '') -%}"
            "<think>{{rc}}</think>{{render_content(message)}}"
        )
        tokenizer.chat_template = tmpl

    total_needed = train_samples


    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    ds = ds.shuffle(seed=seed).select(range(open_thoughts_max_samples))

    def preprocess(example):
        messages = []
        messages.append({
            "role": "system",
            "content": (
                "You are Kimi, an AI assistant created by Moonshot AI."
            ),
        })
        for msg in example["conversations"]:
            role = msg["from"]
            if role == "user":
                messages.append({"role": "user", "content": msg["value"]})
            else:
                thought, solution = split_thought_solution(msg["value"])
                messages.append({"role": "assistant", "content": solution, "reasoning_content": thought})

        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    print(f"Preprocessing {len(ds)} OpenThoughts samples (chat template)...", flush=True)
    ds = ds.map(preprocess, num_proc=min(8, os.cpu_count() or 1))
    print(f"Tokenizing into {total_needed} chunks of length {max_length}...", flush=True)
    all_chunks = make_concat_chunks(ds, tokenizer, max_length, total_needed, seed=shuffle_seed, device=device)


    return all_chunks


def register_hooks(model) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    Registers forward hooks on all Linear layers except lm_head to capture input activations.

    :return: Tuple of (activations dict, list of hook handles).
    """
    activations: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:

            def hook_fn(mod, inp, name=name):
                x = inp[0].detach().cpu()
                activations[name] = x.abs()

            handle = module.register_forward_pre_hook(hook_fn)
            hooks.append(handle)

    return activations, hooks


def register_accumulating_hooks(
    model,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], List[torch.utils.hooks.RemovableHandle]]:
    """
    Registers hooks that accumulate (sum) absolute activations across multiple forward passes.
    This gives aggregated statistics rather than a single-sample snapshot.

    :return: Tuple of (sum_activations dict, sample_count dict, list of hook handles).
    """
    sum_activations: Dict[str, torch.Tensor] = {}
    sample_counts: Dict[str, int] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:

            def hook_fn(mod, inp, name=name):
                x = inp[0].detach().cpu().abs()
                # Sum over token dimension to get per-channel statistics: shape (channels,)
                channel_sum = x.squeeze(0) #.sum(dim=0)
                if name in sum_activations:
                    sum_activations[name] += channel_sum
                    sample_counts[name] += 1 #x.shape[-2]  # number of tokens
                else:
                    sum_activations[name] = channel_sum
                    sample_counts[name] = 1 #x.shape[-2]

            handle = module.register_forward_pre_hook(hook_fn)
            hooks.append(handle)

    return sum_activations, sample_counts, hooks


def collect_calibration_activations(
    model, tokenizer, device: str, calibration_samples: int = 32, max_length: int = 2048
) -> Dict[str, torch.Tensor]:
    """
    Collects aggregated activation statistics using OpenThoughts calibration data.
    Returns per-channel mean absolute activation for each linear layer (excluding lm_head).

    :param model: The model to collect activations from.
    :param tokenizer: Tokenizer for the model.
    :param device: Device to run on.
    :param calibration_samples: Number of calibration chunks.
    :param max_length: Max sequence length per chunk.
    :return: Dict mapping layer name to mean abs activation tensor of shape (channels,).
    """
    print(f"Loading OpenThoughts calibration data ({calibration_samples} samples, max_length={max_length})...")
    chunks = open_thoughts(
        tokenizer, train_samples=calibration_samples, max_length=max_length, device=torch.device(device)
    )
    print(f"Collected {len(chunks)} calibration chunks.")

    sum_activations, sample_counts, hooks = register_accumulating_hooks(model)

    print(f"Running {len(chunks)} forward passes for calibration...")
    for i, chunk in enumerate(chunks):
        with torch.no_grad():
            model(input_ids=chunk.to(device))
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks")

    for h in hooks:
        h.remove()

    # Compute mean: sum / num_tokens -> shape (channels,)
    mean_activations: Dict[str, torch.Tensor] = {}
    for name in sum_activations:
        mean_activations[name] = sum_activations[name] / sample_counts[name]

    print(f"Collected aggregated stats for {len(mean_activations)} layers.")
    return mean_activations


def classify_outliers(activation: torch.Tensor, threshold_factor: float = 6.0) -> str:
    """
    Classifies whether activation outliers are channel-wise, token-wise, or absent.

    Strategy:
    - Compute per-channel (dim=-1 aggregation) and per-token (dim=-2 aggregation) max values.
    - If a few channels have max values >> median, it's channel-wise.
    - If a few tokens have max values >> median, it's token-wise.
    - If both or neither, classify accordingly.

    :param activation: Tensor of shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim).
    :param threshold_factor: Multiplier over median to consider as outlier.
    :return: Classification string.
    """
    if activation.dim() == 3:
        activation = activation.squeeze(0)

    # Per-channel: max over token dimension -> shape (hidden_dim,)
    channel_max = activation.max(dim=0).values
    channel_median = channel_max.median()
    channel_outlier_ratio = (channel_max > threshold_factor * channel_median).float().mean().item()

    # Per-token: max over channel dimension -> shape (seq_len,)
    token_max = activation.max(dim=1).values
    token_median = token_max.median()
    token_outlier_ratio = (token_max > threshold_factor * token_median).float().mean().item()

    channel_is_outlier = channel_outlier_ratio > 0.0 and channel_outlier_ratio < 0.15
    token_is_outlier = token_outlier_ratio > 0.0 and token_outlier_ratio < 0.15

    if channel_is_outlier and token_is_outlier:
        return f"Mixed outliers (channel: {channel_outlier_ratio:.2%}, token: {token_outlier_ratio:.2%})"
    elif channel_is_outlier:
        return f"Channel-wise outliers ({channel_outlier_ratio:.2%} channels are outliers)"
    elif token_is_outlier:
        return f"Token-wise outliers ({token_outlier_ratio:.2%} tokens are outliers)"
    else:
        return "No significant outliers detected"


def render_activation_image(activation: torch.Tensor, layer_name: str) -> str:
    """
    Renders activation heatmap with outlier highlighting and returns base64-encoded PNG.

    Uses log-scale normalization and marks outlier channels/tokens with red lines.
    """
    if activation.dim() == 3:
        activation = activation.squeeze(0)

    data = activation.numpy()
    # Use log1p for better dynamic range visualization
    data_log = np.log1p(data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    im = ax.imshow(data_log, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="log1p(|activation|)")

    # Highlight outlier channels (columns with high max values)
    # channel_max = data.max(axis=0)
    # channel_median = np.median(channel_max)
    # outlier_channels = np.where(channel_max > 6.0 * channel_median)[0]
    # for ch in outlier_channels:
    #     ax.axvline(x=ch, color="cyan", alpha=0.5, linewidth=0.8)

    # Highlight outlier tokens (rows with high max values)
    token_max = data.max(axis=1)
    token_median = np.median(token_max)
    outlier_tokens = np.where(token_max > 6.0 * token_median)[0]
    for tk in outlier_tokens:
        ax.axhline(y=tk, xmin=0.0, xmax=0.1, color="lime", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Token")
    ax.set_title(f"{layer_name} | shape: {list(activation.shape)}")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_aggregated_activation_image(
    activation: torch.Tensor, layer_name: str, threshold_factor: float = 6.0
) -> str:
    """
    Renders a bar chart of per-channel aggregated (mean) activation with outlier highlighting.

    :param activation: 1D tensor of shape (channels,) — per-channel mean abs activation.
    :param layer_name: Name of the layer for the title.
    :param threshold_factor: Threshold for marking outlier channels.
    :return: Base64-encoded PNG.
    """
    data = activation.numpy()
    n_channels = len(data)
    median_val = np.median(data)
    threshold = threshold_factor * median_val
    is_outlier = data > threshold

    fig, ax = plt.subplots(1, 1, figsize=(14, 3))

    colors = np.where(is_outlier, "red", "steelblue")
    ax.bar(range(n_channels), data, color=colors, width=1.0, edgecolor="none")
    ax.axhline(y=threshold, color="lime", linestyle="--", linewidth=1.0, label=f"threshold ({threshold_factor}×median)")
    ax.axhline(y=median_val, color="yellow", linestyle=":", linewidth=0.8, label="median")

    n_outliers = int(is_outlier.sum())
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Mean |activation|")
    ax.set_title(f"{layer_name} | {n_channels} channels, {n_outliers} outliers (red)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, n_channels)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_aggregated_html(
    calibration_stats: Dict[str, torch.Tensor], output_path: str, threshold_factor: float = 6.0
) -> None:
    """
    Generates an HTML file visualizing aggregated per-channel activations used for permutation.
    Only includes down_proj layers.
    """
    down_proj_stats = {name: act for name, act in calibration_stats.items() if "down_proj" in name}

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Aggregated Activation Stats (down_proj) for Permutation</title>",
        "<style>",
        "body { font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; }",
        ".layer { margin-bottom: 30px; border: 1px solid #444; padding: 15px; border-radius: 8px; }",
        ".layer img { max-width: 100%; }",
        ".info { font-size: 13px; margin-top: 8px; padding: 5px 10px; background: #222; border-radius: 4px; }",
        "h2 { color: #aaa; font-size: 13px; margin: 0 0 10px 0; word-break: break-all; }",
        "h1 { color: #fff; }",
        ".summary { margin-bottom: 20px; padding: 10px; background: #222; border-radius: 4px; }",
        "</style>",
        "</head><body>",
        "<h1>Aggregated Per-Channel Mean |Activation| for down_proj (used for permutation)</h1>",
        f"<div class='summary'>Layers: {len(down_proj_stats)} | "
        f"Threshold: {threshold_factor}× median | "
        f"Red bars = outlier channels moved to the left by permutation</div>",
    ]

    for layer_name, act in down_proj_stats.items():
        # Original activation visualization
        img_b64 = render_activation_image(act, layer_name)
        median_val = act.median().item()
        max_val = act.max().item()
        n_outliers = (act > threshold_factor * median_val).sum().item()
        ratio = n_outliers / act.shape[0] * 100

        # Permuted activation visualization (outliers moved to the left)
        perm = compute_outlier_permutation(act, threshold_factor)
        act_permuted = act[:, perm]
        img_permuted_b64 = render_activation_image(act_permuted, f"{layer_name} [PERMUTED]")

        html_parts.append(f"<div class='layer'>")
        html_parts.append(f"<h2>{layer_name}</h2>")
        html_parts.append("<h3 style='color:#888; font-size:12px; margin:5px 0;'>Original channel order:</h3>")
        html_parts.append(f"<img src='data:image/png;base64,{img_b64}' />")
        html_parts.append(
            f"<div class='info'>outliers: {n_outliers}/{act.shape[0]} ({ratio:.1f}%) | "
            f"median: {median_val:.4f} | max: {max_val:.4f} | max/median: {max_val/median_val:.1f}x</div>"
        )
        html_parts.append("<h3 style='color:#888; font-size:12px; margin:10px 0 5px 0;'>After permutation (outliers → left):</h3>")
        html_parts.append(f"<img src='data:image/png;base64,{img_permuted_b64}' />")
        html_parts.append(
            f"<div class='info'>First {n_outliers} channels are outliers | "
            f"remaining {act.shape[0] - n_outliers} channels are normal</div>"
        )
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    output = Path(output_path)
    output.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Aggregated stats HTML saved to: {output.resolve()}")


def generate_html(activations: Dict[str, torch.Tensor], output_path: str) -> None:
    """
    Generates an HTML file with activation visualizations and outlier classifications.
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Outlier Activation Visualization</title>",
        "<style>",
        "body { font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; }",
        ".layer { margin-bottom: 30px; border: 1px solid #444; padding: 15px; border-radius: 8px; }",
        ".layer img { max-width: 100%; }",
        ".classification { font-size: 14px; margin-top: 8px; padding: 5px 10px; border-radius: 4px; }",
        ".channel-wise { background: #4a1a1a; color: #ff6b6b; }",
        ".token-wise { background: #1a4a1a; color: #6bff6b; }",
        ".mixed { background: #4a4a1a; color: #ffff6b; }",
        ".none { background: #1a1a4a; color: #6b6bff; }",
        "h2 { color: #aaa; font-size: 13px; margin: 0 0 10px 0; word-break: break-all; }",
        "h1 { color: #fff; }",
        ".legend { margin-bottom: 20px; padding: 10px; background: #222; border-radius: 4px; }",
        ".legend span { margin-right: 20px; }",
        "</style>",
        "</head><body>",
        "<h1>LLM Linear Layer Input Activation Outlier Visualization</h1>",
        "<div class='legend'>",
        "<span style='color:cyan'>&#9474; Cyan lines = outlier channels</span>",
        "<span style='color:lime'>&#9472; Green lines = outlier tokens</span>",
        "</div>",
    ]

    for layer_name, act in activations.items():
        img_b64 = render_activation_image(act, layer_name)
        classification = classify_outliers(act)

        if "Channel-wise" in classification:
            css_class = "channel-wise"
        elif "Token-wise" in classification:
            css_class = "token-wise"
        elif "Mixed" in classification:
            css_class = "mixed"
        else:
            css_class = "none"

        html_parts.append(f"<div class='layer'>")
        html_parts.append(f"<h2>{layer_name}</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{img_b64}' />")
        html_parts.append(f"<div class='classification {css_class}'>{classification}</div>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    output = Path(output_path)
    output.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"HTML visualization saved to: {output.resolve()}")


def compute_outlier_permutation(
    activation: torch.Tensor, threshold_factor: float = 6.0
) -> torch.Tensor:
    """
    Computes a permutation that moves outlier channels to the left (beginning) of the channel dimension.

    :param activation: Tensor of shape (batch, seq_len, channels), (seq_len, channels), or (channels,).
    :param threshold_factor: Multiplier over median to identify outlier channels.
    :return: Permutation indices tensor of shape (channels,).
    """
    if activation.dim() == 3:
        activation = activation.squeeze(0)

    if activation.dim() == 2:
        # Per-channel max across token dimension
        # channel_stat = activation.max(dim=0).values
        # channel_median = activation.median(dim=0).values
        # order = torch.argsort(channel_stat**2 / (channel_median + 1e-6), descending=True)
        
        channel_stat = activation.mean(dim=0)
        order = torch.argsort(channel_stat, descending=True)
        
        perm = order
    else:
        # Already per-channel (1D from aggregated stats)
        channel_stat = activation
        channel_median = channel_stat.median()

        is_outlier = channel_stat > threshold_factor * channel_median

        outlier_indices = torch.where(is_outlier)[0]
        normal_indices = torch.where(~is_outlier)[0]

        # Outliers first, then normal channels
        perm = torch.cat([outlier_indices, normal_indices], dim=0)
    return perm


def apply_permutation_to_mlp_layers(
    model, activations: Dict[str, torch.Tensor], threshold_factor: float = 6.0
) -> Dict[str, int]:
    """
    For each MLP block, computes the outlier permutation from down_proj input activations
    and applies it:
    - Permute output channels (dim=0) of up_proj.weight and gate_proj.weight
    - Permute input channels (dim=1) of down_proj.weight

    This is a mathematically equivalent transformation that moves outlier channels
    to the left part of the intermediate representation.

    Accepts activations as either:
    - 2D/3D tensors (from single-sample hooks)
    - 1D tensors (aggregated per-channel stats from calibration)

    :param model: The transformer model.
    :param activations: Dict of layer_name -> activation stats (1D, 2D, or 3D tensor).
    :param threshold_factor: Threshold for outlier detection.
    :return: Dict mapping layer block index to number of outlier channels.
    """
    permutation_info: Dict[str, int] = {}

    # Find all down_proj layers and their corresponding up_proj/gate_proj
    down_proj_layers = {name: act for name, act in activations.items() if "down_proj" in name}

    for down_proj_name, down_proj_act in down_proj_layers.items():
        # Extract the block prefix (e.g., "model.layers.0.mlp")
        mlp_prefix = down_proj_name.rsplit(".down_proj", 1)[0]
        up_proj_name = f"{mlp_prefix}.up_proj"
        gate_proj_name = f"{mlp_prefix}.gate_proj"

        # Get the modules
        down_proj_module = dict(model.named_modules()).get(down_proj_name)
        up_proj_module = dict(model.named_modules()).get(up_proj_name)
        gate_proj_module = dict(model.named_modules()).get(gate_proj_name)

        if down_proj_module is None or up_proj_module is None:
            print(f"  Skipping {mlp_prefix}: could not find all modules")
            continue

        # Compute permutation from down_proj input activation stats
        perm = compute_outlier_permutation(down_proj_act, threshold_factor)

        # Count outliers based on dimensionality
        if down_proj_act.dim() == 1:
            channel_stat = down_proj_act
        elif down_proj_act.dim() == 2:
            channel_stat = down_proj_act.max(dim=0).values
        else:
            channel_stat = down_proj_act.squeeze(0).max(dim=0).values

        n_outliers = (channel_stat > threshold_factor * channel_stat.median()).sum().item()

        if n_outliers == 0:
            continue

        perm_device = down_proj_module.weight.device
        perm = perm.to(perm_device)

        # Apply permutation to down_proj input channels (dim=1)
        with torch.no_grad():
            down_proj_module.weight.copy_(down_proj_module.weight[:, perm])

            # Apply permutation to up_proj output channels (dim=0)
            up_proj_module.weight.copy_(up_proj_module.weight[perm, :])
            if up_proj_module.bias is not None:
                up_proj_module.bias.copy_(up_proj_module.bias[perm])

            # Apply permutation to gate_proj output channels (dim=0)
            if gate_proj_module is not None:
                gate_proj_module.weight.copy_(gate_proj_module.weight[perm, :])
                if gate_proj_module.bias is not None:
                    gate_proj_module.bias.copy_(gate_proj_module.bias[perm])

        permutation_info[mlp_prefix] = n_outliers
        print(f"  {mlp_prefix}: permuted {n_outliers} outlier channels to the left")

    return permutation_info


def generate_text(model, tokenizer, device: str, max_new_tokens: int = 128) -> str:
    """
    Generates text using a simple prompt to verify model correctness.
    """
    prompt = "Explain in 3 sentences what is machine learning:"
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_permutation_experiment(
    model, tokenizer, activations: Dict[str, torch.Tensor], args
) -> Dict[str, torch.Tensor]:
    """
    Runs the full permutation experiment using aggregated OpenThoughts calibration stats:
    1. Generate text before permutation
    2. Collect aggregated activation stats from OpenThoughts
    3. Apply permutation based on aggregated stats
    4. Generate text after permutation
    5. Compare outputs

    :return: The calibration_stats dict (per-channel mean activations) for visualization.
    """
    print("\n" + "=" * 80)
    print("PERMUTATION EXPERIMENT (using OpenThoughts calibration)")
    print("=" * 80)

    # Generation BEFORE permutation
    print("\n[Before permutation] Generating text...")
    text_before = generate_text(model, tokenizer, args.device, args.max_new_tokens)
    print(f"Output: {text_before[:500]}")

    # Collect aggregated stats from OpenThoughts calibration data
    print("\nCollecting aggregated activation statistics from OpenThoughts...")
    calibration_stats = collect_calibration_activations(
        model, tokenizer, args.device,
        calibration_samples=args.calibration_samples,
        max_length=args.max_length,
    )

    # Apply permutation using aggregated stats
    print("\nApplying outlier channel permutation to MLP layers (based on aggregated stats)...")
    perm_info = apply_permutation_to_mlp_layers(model, calibration_stats, args.threshold)
    print(f"\nPermuted {len(perm_info)} MLP blocks total.")

    # Generation AFTER permutation
    print("\n[After permutation] Generating text...")
    text_after = generate_text(model, tokenizer, args.device, args.max_new_tokens)
    print(f"Output: {text_after[:500]}")

    # Compare
    print("\n" + "-" * 80)
    if text_before == text_after:
        print("RESULT: Outputs are IDENTICAL. Permutation is mathematically correct.")
    else:
        print("RESULT: Outputs DIFFER. Something went wrong with the permutation.")
        print(f"\n  Before ({len(text_before)} chars): {text_before[:200]}...")
        print(f"  After  ({len(text_after)} chars): {text_after[:200]}...")
    print("-" * 80)

    return calibration_stats


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    print("Preparing input...")
    inputs = create_chain_of_thought_input(tokenizer)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    print(f"Input sequence length: {inputs['input_ids'].shape[1]} tokens")

    print("Registering hooks...")
    activations, hooks = register_hooks(model)

    print("Running forward pass...")
    with torch.no_grad():
        model(**inputs)

    print(f"Captured activations from {len(activations)} layers")

    # Remove hooks
    for h in hooks:
        h.remove()

    print("Generating visualization...")
    #generate_html(activations, args.output)

    if args.permute:
        calibration_stats = run_permutation_experiment(model, tokenizer, activations, args)

        # Visualize aggregated down_proj stats used for permutation
        aggregated_output = args.output.replace(".html", "_aggregated_stats.html")
        generate_aggregated_html(calibration_stats, aggregated_output, args.threshold)

        model.save_pretrained(args.model_id + "_permuted_mean")
        tokenizer.save_pretrained(args.model_id + "_permuted_mean")
        
        print("Registering hooks after permutation...")
        activations, hooks = register_hooks(model)

        print("Running forward pass after permutation...")
        with torch.no_grad():
            model(**inputs)

        print(f"Captured activations from {len(activations)} layers")

        # Remove hooks
        for h in hooks:
            h.remove()

        print("Generating visualization...")
        generate_html(activations, args.output.replace(".html", "_after_permutation.html"))


if __name__ == "__main__":
    main()
