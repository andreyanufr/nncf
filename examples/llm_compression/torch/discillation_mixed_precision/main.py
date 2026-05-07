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
import argparse
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from optimum.modeling_base import OptimizedModel
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from utils import replace_linear_with_mixer

import nncf
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedAWQParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch import load_from_config
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer

warnings.filterwarnings("ignore", category=TracerWarning)


def generate_answer(
    model: OptimizedModel, tokenizer: AutoTokenizer, question: str = "What is AI? ", max_new_tokens=32
) -> str:
    """
    Generate an answer to a given question using the provided model and tokenizer.

    :param question: The input question to generate an answer for.
    :param model: The optimized model used for generating the answer.
    :param tokenizer: The tokenizer used to process the input question and decode the output.
    :param max_new_tokens: The maximum number of new tokens to generate in the answer. Default is 32.
    :return: The generated answer as a string.
    """
    messages = [{"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device=model.device)
    input_len = len(input_ids[0])

    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)[0]
    answer = tokenizer.decode(output[input_len:], skip_special_tokens=True)
    return answer


def get_pile(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    ds = load_dataset("NeelNanda/pile-10k", split="train")

    trainloader = []
    for example in ds:
        trainenc = tokenizer(example["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break

    return trainloader


def get_distill_dataset(
    num_samples: int,
    seqlen: int,
    tokenizer: Any,
    device: torch.device,
    name="mlfoundations-dev/DeepSeek-R1-Distill-Qwen-7B_eval_03-07-25_17-46_2870",
) -> Dataset:
    """
    Prepares a dataset for distillation by tokenizing and processing the input data.

    :param num_samples: The number of samples to include in the dataset.
    :param seqlen: The sequence length to which the input data should be truncated or padded.
    :param tokenizer: The tokenizer used to process the input data.
    :param device: The device on which the tensors will be stored (e.g., CPU or GPU).
    :return: A Dataset object containing the processed input data for distillation.
    """
    ds = load_dataset(name, split="train")
    trainloader = []
    for example in ds:
        trainenc = tokenizer(example["context"][0]["content"] + " " + example["model_outputs"], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break
    return trainloader


@torch.no_grad()
def calc_hiddens(model: nn.Module, dataloader: list[Tensor]) -> list[Tensor]:
    """
    Calculate the hidden states for each input in the dataloader using the given model.

    :param model: The model used to calculate the hidden states.
    :param dataloader: The dataloader providing the inputs to the model.
    :return: A list of hidden states for each input in the dataloader.
    """
    orig_hiddens = []
    for data in track(dataloader, description="Calculating original hiddens"):
        model_input = get_model_input(data)
        orig_hiddens.append(model.model(**model_input).last_hidden_state.to("cpu"))
    torch.cuda.empty_cache()
    return orig_hiddens


def get_model_input(input_ids: Tensor) -> dict[str, Tensor]:
    """
    Prepares the model input dictionary with input IDs, attention mask, and position IDs.

    :param input_ids: Tensor containing the input IDs.
    :return: A dictionary with keys "input_ids", "attention_mask", and "position_ids",
        each mapping to their respective tensors.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def kl_div(student_hiddens: torch.Tensor, teacher_hiddens: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence loss between the student and teacher hidden states.
    The input tensors are expected to have the same shape, and the last dimension represents the number of classes.

    :param student_hiddens: The hidden states from the student model.
    :param teacher_hiddens: The hidden states from the teacher model.
    :returns: The computed KL divergence loss.
    """
    num_classes = student_hiddens.shape[-1]
    return F.kl_div(
        # input=F.log_softmax(student_hiddens.view(-1, num_classes), dim=-1),
        # target=F.log_softmax(teacher_hiddens.view(-1, num_classes), dim=-1),
        input=F.log_softmax(student_hiddens.reshape(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_hiddens.reshape(-1, num_classes), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def set_trainable(model: nn.Module, lora_lr: float, fq_lr: float) -> list[dict[str, Any]]:
    """
    Sets the trainable parameters of the model for quantization-aware training with LoRA (Low-Rank Adaptation).

    This function disables gradients for all parameters in the model, then selectively enables gradients for
    specific quantizers (AsymmetricLoraQuantizer, SymmetricLoraQuantizer) that have 4-bit quantization.
    It collects the trainable parameters and adapters from these quantizers and returns them in a format
    suitable for an optimizer.

    :param model: The model to be trained.
    :param lora_lr: Learning rate for the LoRA adapters.
    :param fq_lr: Learning rate for the quantizer scales.
    :return: A list of dictionaries containing the parameters to be optimized and their corresponding learning rates.
    """
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    hook_storage = get_hook_storage(model)
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)) and (module.num_bits == 4):
            module.enable_gradients()
            params = module.get_trainable_params()
            adapters = module.get_adapters()
            adapters_to_train.extend(adapters.values())
            scales_to_train.extend(param for name, param in params.items() if name not in adapters)

    params = list(model.parameters())
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    all_param = sum(p.numel() for p in params)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    model.train()
    return [{"params": adapters_to_train, "lr": lora_lr}, {"params": scales_to_train, "lr": fq_lr}]


def save_checkpoint(model: nn.Module, ckpt_file: Path, model_state: bool = True) -> None:
    """
    Stores the current state of a quantized model to a checkpoint file.

    :param model: The model whose state will be saved to checkpoint.
    :param ckpt_file: Path to store the checkpoint file.
    :param model_state: Whether to save the complete model weights in addition to NNCF state. Required when using
        AWQ method which fuses scaling factors into weights. When False, only NNCF configuration and state are saved,
        as they're maintained separately from the model's weights.
    """
    hook_storage = get_hook_storage(model)
    ckpt = {"nncf_state_dict": hook_storage.state_dict(), "nncf_config": nncf.torch.get_config(model)}
    if model_state:
        ckpt["model_state"] = model.state_dict()
    torch.save(ckpt, ckpt_file)


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


def limit_type(astr: str):
    value = float(astr)
    if value < 0 or value > 1:
        msg = "value not in range [0,1]"
        raise argparse.ArgumentTypeError(msg)
    return value


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output",
        help="Path to the directory for storing logs, tuning checkpoint, compressed model, validation references.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to start from previously saved checkpoint. If not specified or checkpoint does not exist, "
        "start from scratch by post-training weight compression initialization.",
    )
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of lora adapters")
    parser.add_argument(
        "--basic_init",
        action="store_true",
        help="Whether to initialize quantization with basic min-max round-to-nearest schema. By default, advanced "
        "data-aware post-training methods are used: AWQ + Scale Estimation. These methods typically provide better "
        "accuracy, but require a calibration dataset and additional initialization time "
        "(~20 sec for 1B and ~80 sec for 8B models).",
    )

    # Data params
    parser.add_argument("--num_train_samples", type=int, default=512, help="Number of training samples")
    parser.add_argument("--train_seqlen", type=int, default=1024, help="Train data context length.")
    parser.add_argument("--eval_seqlen", type=int, default=2048, help="Evaluation data context length.")
    parser.add_argument(
        "--limit",
        type=limit_type,
        default=None,
        help="A percentage of the total number of examples for evaluation. "
        "Should be on the range [0,1]. If None, all samples will be used.",
    )

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning. "
        "For larger models (over 3 billion parameters), a learning rate of 5e-5 is recommended.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=2,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    parser.add_argument(
        "--distill_dataset_name",
        type=str,
        default="mlfoundations-dev/DeepSeek-R1-Distill-Qwen-7B_eval_03-07-25_17-46_2870",
        help="Name of the dataset to use for distillation.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Fraction of total optimizer steps used for linear warmup before cosine decay.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="Final LR as a fraction of the peak LR at the end of the cosine schedule.",
    )
    return parser


def main(argv) -> float:
    """
    Fine-tunes the specified model and returns the difference between initial and best validation perplexity in Torch,
    and the test perplexity for best model exported to OpenVINO.
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    assert torch.cuda.is_available()
    transformers.set_seed(42)

    device = "cuda"
    torch_dtype = torch.bfloat16
    compression_config = dict(
        mode=CompressWeightsMode.INT2_SYM,
        group_size=32,
        awq=False,  # avoid awq for splitted linear layers
        scale_estimation=not args.basic_init,
        compression_format=CompressionFormat.FQ_LORA,
    )
    pprint({"CLI arguments": vars(args), "Major compression parameters": compression_config})
    compression_config["advanced_parameters"] = AdvancedCompressionParameters(
        awq_params=AdvancedAWQParameters(prefer_data_aware_scaling=False),  # not args.basic_init),
        lora_adapter_rank=args.lora_rank,
    )
    # Configure output and log files.
    output_dir = Path(args.output_dir)
    tensorboard_dir = output_dir / "tb" / datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    last_dir = output_dir / "last"
    if not args.resume:
        shutil.rmtree(last_dir, ignore_errors=True)
    for path in [output_dir, tensorboard_dir, last_dir]:
        path.mkdir(exist_ok=True, parents=True)
    ckpt_file = last_dir / "nncf_checkpoint.pth"
    print(f"To visualize the loss and validation metrics, open Tensorboard using the logs from: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with absorbable LoRA")

    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch_dtype, device_map="auto", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Prepare training and calibration data
    train_loader = get_pile(
        num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
    )
    if args.distill_dataset_name:
        dataset = get_distill_dataset(
            num_samples=args.num_train_samples,
            seqlen=args.train_seqlen,
            tokenizer=tokenizer,
            device=device,
            name=args.distill_dataset_name,
        )
        train_loader.extend(dataset)

    answer1 = generate_answer(model, tokenizer)
    print(f"Answer before mixed: {answer1}\n")
    model = replace_linear_with_mixer(model)
    answer2 = generate_answer(model, tokenizer)
    print(f"Answer after mixed: {answer2}\n")
    if answer1 != answer2:
        print(
            "The answers are different after replacing linear layers with LinearMIXER. This may be due to the fact that the model has not been fine-tuned yet, and the weights of the new LinearMIXER layers have been initialized based on the original linear layers. Fine-tuning the model with the new LinearMIXER layers should help to recover the original performance."
        )
        exit(1)

    if args.basic_init:
        example_input = {k: v.to(device) for k, v in model.dummy_inputs.items()}
        dataset = Dataset([example_input])
    else:
        calib_loader = get_pile(num_samples=128, seqlen=128, tokenizer=tokenizer, device=device)
        dataset = Dataset(map(get_model_input, calib_loader))

    # Pre-compute hiddens of teacher model for distillation loss.
    model_id = args.pretrained.split("/")[-1]
    hiddens_pth = last_dir / f"orig_hiddens_{model_id}_{args.num_train_samples}_{args.train_seqlen}.pt"
    if hiddens_pth.exists():
        orig_hiddens = torch.load(hiddens_pth)
    else:
        orig_hiddens = calc_hiddens(model, train_loader)
        torch.save(orig_hiddens, hiddens_pth)

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    if args.resume and ckpt_file.exists():
        model = load_checkpoint(model, ckpt_file)
    else:
        model = compress_weights(model, dataset=dataset, **compression_config)
        save_checkpoint(model, ckpt_file, model_state=not args.basic_init)
    fq_lr = args.lr / 10
    weight_decay = args.lr
    param_to_train = set_trainable(model, lora_lr=args.lr, fq_lr=fq_lr)
    opt = torch.optim.AdamW(param_to_train, weight_decay=weight_decay)

    # Run tuning with distillation loss and validation after each epoch.
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    optimizer_steps_per_epoch = max(1, microbatches_per_epoch // grad_accumulation_steps)
    total_optimizer_steps = max(1, args.epochs * optimizer_steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_ratio * total_optimizer_steps)) if args.warmup_ratio > 0 else 0
    scheduler = transformers.get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_steps = 0
    aggregated_kl_loss = 0.0
    aggregated_l1_loss = 0.0

    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for indices in track(batch_indices_epoch, description=f"Train epoch {epoch}"):
            indices = indices.tolist()

            def form_batch(inputs: list[Tensor], model_input: bool):
                batch = torch.cat([inputs[i] for i in indices], dim=0)
                return get_model_input(batch) if model_input else batch.to(device=device, dtype=torch_dtype)

            # Compute distillation loss between logits of the original model and the model with FQ + LoRA.
            inputs = form_batch(train_loader, model_input=True)
            with torch.no_grad():
                cur_teacher_hiddens = form_batch(orig_hiddens, model_input=False)
                targets = model.lm_head(cur_teacher_hiddens)
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma has post-processing after lm_head
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                cur_student_hiddens = outputs.hidden_states[-1]

            # compute loss only for second half of the sequence, to let the model see enough context before computing loss and getting meaningful gradients for distillation
            kl_loss = kl_div(
                logits[:, logits.shape[1] // 2 :],
                targets[:, targets.shape[1] // 2 :].to(dtype=torch_dtype, device=device),
            )
            l1_loss = F.l1_loss(
                cur_student_hiddens[:, cur_student_hiddens.shape[1] // 2 :],
                cur_teacher_hiddens[:, cur_teacher_hiddens.shape[1] // 2 :].to(dtype=torch_dtype, device=device),
            )
            loss = kl_loss + 0.1 * l1_loss

            # Perform an optimization step after accumulating gradients over multiple minibatches.
            loss_numerator += loss.item()
            grad_steps += 1
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)
            (loss / grad_accumulation_steps).backward()

            aggregated_kl_loss += kl_loss.item()
            aggregated_l1_loss += l1_loss.item()

            if grad_steps == grad_accumulation_steps:
                opt.step()
                scheduler.step()
                opt.zero_grad()
                aggregated_loss = loss_numerator / grad_steps
                total_steps += 1
                tb.add_scalar("loss", aggregated_loss, total_steps)
                tb.add_scalar("kl_loss", aggregated_kl_loss / grad_steps, total_steps)
                tb.add_scalar("l1_loss", aggregated_l1_loss / grad_steps, total_steps)
                for i, pg in enumerate(opt.param_groups):
                    tb.add_scalar(f"lr/group_{i}", pg["lr"], total_steps)
                loss_numerator = grad_steps = aggregated_kl_loss = aggregated_l1_loss = 0

        save_checkpoint(model, ckpt_file, model_state=not args.basic_init)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                answer2 = generate_answer(model, tokenizer)
                print(f"Answer after epoch {epoch}: {answer2}\n")

    del model
    # Export the best tuned model to OpenVINO and evaluate it using LM-Evaluation-Harness.
    # export_to_pytorch(args.pretrained, ckpt_file, ckpt_file.parent / "pt_model_for_eval")
    # tokenizer.save_pretrained(ckpt_file.parent / "pt_model_for_eval")
    # model_for_eval = export_to_openvino(args.pretrained, ckpt_file, ckpt_file.parent)
    # ov_perplexity = measure_perplexity(model_for_eval, args.eval_seqlen, args.limit)
    # tb.add_scalar("ov_perplexity", ov_perplexity, 0)
    # print(
    #     f"The finetuned model has been exported to OpenVINO and saved to: {last_dir}\n"
    #     f"The word perplexity on wikitext (test) = {ov_perplexity:.4f}"
    # )
    # return ov_perplexity


if __name__ == "__main__":
    main(sys.argv[1:])
