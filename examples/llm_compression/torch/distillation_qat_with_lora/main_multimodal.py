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
"""
Distillation QAT with absorbable LoRA for the language-model part of
Qwen/Qwen3.5-35B-A3B (Qwen3-Omni MoE). The LM submodule is `model.thinker`;
audio and vision encoders are kept frozen in original precision.

Note: this model is NOT compatible with `AutoModelForCausalLM`; we load it
with `AutoModel`.

Pipeline:
1. Build an image-text calibration/training dataset.
2. Run the multimodal model and dump LM inputs (`inputs_embeds`,
   `attention_mask`, `position_ids`) plus teacher logits to disk.
3. If the cache directory already exists, reload tensors from disk.
4. Compress the LM weights with NNCF (scale_estimation, FQ + LoRA) using the
   dumped LM inputs.
5. Distillation QAT trains FQ scales + LoRA adapters of the LM only.
"""
import argparse
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoConfig
from transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoProcessor

import nncf
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch import load_from_config
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer

from PIL import Image
import requests
from io import BytesIO

warnings.filterwarnings("ignore", category=TracerWarning)


DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEST_MARKER = "test"

# Attribute -> value overrides applied recursively to every (sub)config when
# building the tiny test variant. Keeps the model shape compatible with the
# original architecture while pushing total parameter count well under 1B.
_TEST_OVERRIDES: dict[str, int] = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 256,
    "shared_expert_intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "head_dim": 32,
    "max_position_embeddings": 1024,
}


def _shrink_config(config: Any) -> None:
    """
    Recursively apply ``_TEST_OVERRIDES`` to ``config`` and any nested
    PretrainedConfig sub-configs (text/vision/audio/thinker/...).
    """
    from transformers import PretrainedConfig

    for key, value in _TEST_OVERRIDES.items():
        if hasattr(config, key):
            setattr(config, key, value)
    for value in vars(config).values():
        if isinstance(value, PretrainedConfig):
            _shrink_config(value)

def get_test_model() -> nn.Module:
    """
    Build a tiny randomly-initialized variant of ``DEFAULT_MODEL`` (<1B parameters) for smoke-testing the pipeline without downloading the full checkpoint.
    """
    config = AutoConfig.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    _shrink_config(config)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model.to("cuda")


def build_model(pretrained: str, torch_dtype: torch.dtype) -> nn.Module:
    """
    Load the multimodal model. When ``pretrained == "test"``, build a tiny
    randomly-initialized variant of ``DEFAULT_MODEL`` (<1B parameters) for
    smoke-testing the pipeline without downloading the full checkpoint.
    """
    if pretrained == TEST_MARKER:
        config = AutoConfig.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
        _shrink_config(config)
        config.torch_dtype = torch_dtype
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch_dtype)
        return model.to("cuda")
    return AutoModelForCausalLM.from_pretrained(
        pretrained,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )


def build_processor(pretrained: str) -> Any:
    """
    Load the processor. The tiny test variant reuses the real processor from
    ``DEFAULT_MODEL`` since processors are not affected by config shrinking.
    """
    src = DEFAULT_MODEL if pretrained == TEST_MARKER else pretrained
    return AutoProcessor.from_pretrained(src, trust_remote_code=True)


def get_language_model(model: nn.Module) -> nn.Module:
    """
    Return the LM submodule of Qwen3-Omni (the `thinker`).
    """
    return model #.thinker


def freeze_non_lm(model: nn.Module) -> None:
    """
    Freeze every parameter that does not belong to the language-model submodule.
    """
    lm_params = {id(p) for p in get_language_model(model).parameters()}
    for p in model.parameters():
        if id(p) not in lm_params:
            p.requires_grad_(False)


def load_image(url: str) -> Image.Image:
    """
    Load an image from a URL and return it as a PIL Image. Dumb image to local directory to avoid repeatedly downloading the same image when re-running the script multiple times during development.
    """
    local_name = url.split("/")[-1]
    local_path = Path("images") / local_name
    if local_path.is_file():
        return Image.open(local_path).convert("RGB")
    response = requests.get(url)
    response.raise_for_status()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(response.content)
    return Image.open(local_path).convert("RGB")

def load_image_text_dataset(num_samples: int) -> list[dict[str, Any]]:
    """
    Load image-text samples from ``CaptionEmporium/conceptual-captions-cc12m-llavanext`` (image + first caption).
    """
    ds = load_dataset("CaptionEmporium/conceptual-captions-cc12m-llavanext", split="train", streaming=True)
    out: list[dict[str, Any]] = []
    for ex in ds:
        try:
            image = load_image(ex["url"])
        except Exception as e:
            print(f"Warning: failed to load image from {ex['url']}: {e}")
            continue
        out.append({"image": image, "text": ex["caption_llava"]})
        if len(out) >= num_samples:
            break
    return out


def load_text_dataset(num_samples: int) -> list[dict[str, Any]]:
    """
    Load text-only samples from Open-Orca (first 2K samples from the training split).
    """
    ds = load_dataset("jtatman/python-code-dataset-500k", split="train", streaming=True)
    out: list[dict[str, Any]] = []
    for ex in ds:
        out.append({"instruction": ex["instruction"], "response": ex["output"], "system": ex["system"]})
        if len(out) >= num_samples:
            break
    return out


def build_processor_inputs(
    processor: Any, samples: list[dict[str, Any]]
) -> list[dict[str, Tensor]]:
    """
    Convert raw ``{image, text}`` samples to processor outputs (CPU tensors).
    """
    batches: list[dict[str, Tensor]] = []
    for s in samples:
        is_text_only = "instruction" in s and "response" in s and "image" not in s

        if is_text_only:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": s["system"] + s["instruction"]}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": s["response"]}],
                },
            ]
            enc = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True, return_tensors="pt", return_dict=True)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": s["image"]},
                        {"type": "text", "text": s["text"]},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            enc = processor(text=[text], images=[s["image"]], return_tensors="pt")
            
        batches.append({k: v.cpu() if isinstance(v, Tensor) else v for k, v in enc.items()})
    return batches


LM_INPUTS_FILE = "lm_inputs.pt"
LM_LOGITS_FILE = "lm_logits.pt"


@torch.no_grad()
def compute_lm_inputs_and_logits(
    model: nn.Module,
    processor_batches: list[dict[str, Tensor]],
    device: torch.device,
) -> tuple[list[dict[str, Tensor]], list[Tensor]]:
    """
    Run the multimodal model to obtain LM inputs (``inputs_embeds``,
    ``attention_mask``, ``position_ids``) and teacher logits. Tensors are
    returned on CPU.
    """
    lm_inputs: list[dict[str, Tensor]] = []
    logits_list: list[Tensor] = []

    for batch in tqdm(processor_batches, desc="Dump teacher LM inputs/logits"):
        gpu_batch = {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
        out = model(**gpu_batch, output_hidden_states=True, use_cache=False, return_dict=True)

        # hidden_states[0] is the LM input after multimodal feature merging.
        inputs_embeds = out.hidden_states[0].detach().to("cpu")
        logits = out.logits.detach().to("cpu")

        attention_mask = gpu_batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)
        else:
            attention_mask = attention_mask.detach().to("cpu")
        position_ids = torch.cumsum(attention_mask, dim=1) - 1

        lm_inputs.append(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        logits_list.append(logits)

        del gpu_batch, out
        torch.cuda.empty_cache()

    return lm_inputs, logits_list


def save_lm_cache(cache_dir: Path, lm_inputs: list[dict[str, Tensor]], logits: list[Tensor]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lm_inputs, cache_dir / LM_INPUTS_FILE)
    torch.save(logits, cache_dir / LM_LOGITS_FILE)


def load_lm_cache(cache_dir: Path) -> tuple[list[dict[str, Tensor]], list[Tensor]]:
    lm_inputs = torch.load(cache_dir / LM_INPUTS_FILE, map_location="cpu", weights_only=False)
    logits = torch.load(cache_dir / LM_LOGITS_FILE, map_location="cpu", weights_only=False)
    return lm_inputs, logits


def has_lm_cache(cache_dir: Path) -> bool:
    return (cache_dir / LM_INPUTS_FILE).is_file() and (cache_dir / LM_LOGITS_FILE).is_file()


def kl_div(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """
    KL divergence between student and teacher logits (averaged per token).
    """
    num_classes = student_logits.shape[-1]
    return F.kl_div(
        input=F.log_softmax(student_logits.view(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_logits.view(-1, num_classes), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def set_trainable(model: nn.Module, lora_lr: float, fq_lr: float) -> list[dict[str, Any]]:
    """
    Enable gradients only on FQ scales and LoRA adapters of 4-bit quantizers
    inside the language model. Returns parameter groups for the optimizer.
    """
    model.requires_grad_(False)
    scales_to_train: list[Tensor] = []
    adapters_to_train: list[Tensor] = []
    hook_storage = get_hook_storage(model)
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)) and module.num_bits == 4:
            module.enable_gradients()
            params = module.get_trainable_params()
            adapters = module.get_adapters()
            adapters_to_train.extend(adapters.values())
            scales_to_train.extend(p for name, p in params.items() if name not in adapters)

    all_params = list(model.parameters())
    trainable = sum(p.numel() for p in all_params if p.requires_grad)
    total = sum(p.numel() for p in all_params)
    print(
        f"trainable params: {trainable:,d} || all params: {total:,d} || "
        f"trainable%: {100 * trainable / total:.4f}"
    )
    model.train()
    return [{"params": adapters_to_train, "lr": lora_lr}, {"params": scales_to_train, "lr": fq_lr}]


def save_checkpoint(model: nn.Module, ckpt_file: Path) -> None:
    lm = get_language_model(model)
    ckpt = {
        "nncf_state_dict": get_hook_storage(lm).state_dict(),
        "nncf_config": nncf.torch.get_config(lm),
    }
    torch.save(ckpt, ckpt_file)


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    lm = get_language_model(model)
    lm = load_from_config(lm, ckpt["nncf_config"])
    get_hook_storage(lm).load_state_dict(ckpt["nncf_state_dict"])
    return model


def lm_forward_logits(model: nn.Module, lm_input: dict[str, Tensor]) -> Tensor:
    out = get_language_model(model)(
        inputs_embeds=lm_input["inputs_embeds"],
        attention_mask=lm_input["attention_mask"],
        position_ids=lm_input["position_ids"],
        use_cache=False,
        return_dict=True,
    )
    return out.logits


def iterate_lm_inputs_for_calibration(
    lm_inputs: list[dict[str, Tensor]], device: torch.device
) -> Iterator[dict[str, Tensor]]:
    for item in lm_inputs:
        yield {k: v.to(device) for k, v in item.items()}


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--pretrained", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=Path("output_mm"))
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)

    parser.add_argument("--num_train_samples", type=int, default=256)
    parser.add_argument("--num_calib_samples", type=int, default=64)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--microbatch_size", type=int, default=1)
    parser.add_argument("--use_image", action="store_true", help="Whether to include images in the training data (default: False, text-only).")
    return parser


def main(argv) -> None:
    args = get_argument_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    transformers.set_seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda")
    torch_dtype = torch.bfloat16

    compression_config = dict(
        mode=CompressWeightsMode.INT4_SYM,
        group_size=128,
        awq=False,
        scale_estimation=True,
        compression_format=CompressionFormat.FQ_LORA,
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=args.lora_rank if args.pretrained != TEST_MARKER else 1),
    )
    pprint({"CLI arguments": vars(args), "Major compression parameters": compression_config})

    description = args.description or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    run_dir = args.output_dir / description
    cache_dir = run_dir / "lm_cache"
    last_dir = run_dir / "last"
    tb_dir = args.output_dir / "tb" / description

    cache_existed = run_dir.exists() and has_lm_cache(cache_dir)
    if not args.resume and not cache_existed:
        shutil.rmtree(run_dir, ignore_errors=True)
    for path in (run_dir, cache_dir, last_dir, tb_dir):
        path.mkdir(parents=True, exist_ok=True)

    ckpt_file = last_dir / "nncf_checkpoint.pth"
    tb = SummaryWriter(tb_dir, "MM QAT with absorbable LoRA")
    print(f"Run directory: {run_dir}")

    # -------------------------------------------------------------------
    # Step 1: obtain LM inputs + teacher logits (from cache or fresh).
    # -------------------------------------------------------------------
    if cache_existed:
        print(f"Found cached LM inputs/logits at {cache_dir}; loading from disk.")
        lm_inputs, teacher_logits = load_lm_cache(cache_dir)
    else:
        print("Building dataset and dumping LM inputs/logits...")
        processor = build_processor(args.pretrained)
        # AutoModelForCausalLM is not compatible with Qwen/Qwen3.5-35B-A3B; use AutoModel.
        teacher = build_model(args.pretrained, torch_dtype)
        teacher.eval()
        if args.use_image:
            samples = load_image_text_dataset(max(args.num_train_samples, args.num_calib_samples))
        else:
            samples = load_text_dataset(max(args.num_train_samples, args.num_calib_samples))

        proc_batches = build_processor_inputs(processor, samples)
        lm_inputs, teacher_logits = compute_lm_inputs_and_logits(teacher, proc_batches, device)
        save_lm_cache(cache_dir, lm_inputs, teacher_logits)
        del teacher, processor, proc_batches
        torch.cuda.empty_cache()

    # Calibration uses a subset of dumped LM inputs (kept on CPU; moved to GPU
    # only inside NNCF's forward).
    calib_inputs = lm_inputs[: args.num_calib_samples]
    train_inputs = lm_inputs[: args.num_train_samples]
    train_targets = teacher_logits[: args.num_train_samples]

    # -------------------------------------------------------------------
    # Step 2: load student model and compress only the language model.
    # -------------------------------------------------------------------
    student = build_model(args.pretrained, torch_dtype)
    freeze_non_lm(student)
    lm = get_language_model(student)

    if args.resume and ckpt_file.exists():
        print(f"Resuming from checkpoint {ckpt_file}")
        student = load_checkpoint(student, ckpt_file)
    else:
        nncf_dataset = Dataset(iterate_lm_inputs_for_calibration(calib_inputs, device))
        compress_weights(lm, dataset=nncf_dataset, **compression_config)
        save_checkpoint(student, ckpt_file)

    # -------------------------------------------------------------------
    # Step 3: distillation QAT loop on LM inputs.
    # -------------------------------------------------------------------
    fq_lr = args.lr / 10
    weight_decay = args.lr
    param_groups = set_trainable(student, lora_lr=args.lr, fq_lr=fq_lr)
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    grad_accum = max(1, args.batch_size // args.microbatch_size)
    n = len(train_inputs)
    epoch_n = n - n % args.microbatch_size
    steps_per_epoch = epoch_n // args.microbatch_size

    total_step = 0
    for epoch in range(args.epochs):
        order = torch.randperm(n)[:epoch_n].chunk(steps_per_epoch)
        loss_num = 0.0
        grad_steps = 0
        pbar = tqdm(order, desc=f"Train epoch {epoch}", total=len(order))
        for chunk in pbar:
            i = chunk.tolist()[0]
            inputs_embeds = train_inputs[i]["inputs_embeds"].to(device=device, dtype=torch_dtype)
            attention_mask = train_inputs[i]["attention_mask"].to(device)
            position_ids = train_inputs[i]["position_ids"].to(device)
            targets = train_targets[i].to(device=device, dtype=torch_dtype)

            student_logits = lm_forward_logits(
                student,
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
            )
            loss = kl_div(student_logits, targets)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")

            (loss / grad_accum).backward()
            loss_num += loss.item()
            grad_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if grad_steps == grad_accum:
                opt.step()
                opt.zero_grad()
                tb.add_scalar("loss", loss_num / grad_steps, total_step)
                total_step += 1
                loss_num = 0.0
                grad_steps = 0

        save_checkpoint(student, ckpt_file)

    print(f"Training finished. Checkpoint saved to: {ckpt_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
