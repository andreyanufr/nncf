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
from pathlib import Path

import torch
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel import OVConfig
from optimum.intel.openvino import OVModelForCausalLM
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from utils import replace_linear_with_mixer

import nncf
from nncf.parameters import StripFormat
from nncf.torch import load_from_config
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.quantization.layers import SymmetricLoraQuantizer  # noqa: F401


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


# PRETRAINED = "Qwen/Qwen3-8B"
# CHECKPOINT_FILE = "output_qwen_3_8B/last_sym_nncf_equalizing_gs_64_64_fq_lr01_ep5_safe_scale/nncf_checkpoint.pth"
# IR_DIR = "output_qwen_3_8B/last_sym_nncf_equalizing_gs_64_64_fq_lr01_ep5_safe_scale/ov_model_fp16/"


PRETRAINED = "Qwen/Qwen3-4B"
CHECKPOINT_FILE = "output_qwen_3_4B/last_sym4_sym2_nncf_equalizing_open_thought_data_aware_mix_permuted_wider_kaiming/nncf_checkpoint.pth"
IR_DIR = "output_qwen_3_4B/last_sym4_sym2_nncf_equalizing_open_thought_data_aware_mix_permuted_wider_kaiming/ov_model_fp16_one____/"

def get_input_data(hf_tokenizer: AutoTokenizer, text: str = "Hello world!") -> dict[str, torch.Tensor]:
    """
    Tokenizes text into a trace-ready PyTorch input dictionary.

    :param tokenizer: Tokenizer used for text preprocessing.
    :param text: Input prompt text.
    :return: Dictionary with tensors required by the model forward.
    """
    tokenized_text = hf_tokenizer(text, return_tensors="pt")
    return {
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
    }


with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    if True:
        model_to_eval = AutoModelForCausalLM.from_pretrained(PRETRAINED, torch_dtype=torch.float32, device_map="cpu")
        model_to_eval, _ = replace_linear_with_mixer(model_to_eval, ratio=0.5)
        model_to_eval = load_checkpoint(model_to_eval, CHECKPOINT_FILE)

        model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.OV)
        model_to_eval.eval()
        
        example_input = get_input_data(tokenizer, "Hello world!")
        # output = model_to_eval.generate(**example_input, max_new_tokens=128, do_sample=False)[0]
        # answer = tokenizer.decode(output, skip_special_tokens=True)
        # print(f"Answer from the model: {answer}")
        
        

        # with torch.inference_mode():
        #     with torch.autocast(device_type="cpu", dtype=torch.float32):
        export_from_model(
            model_to_eval,
            IR_DIR,
            device="cpu",
            # compression_option="fp16",
            # ov_config=OVConfig(dtype="fp16"),
            # model_kwargs={"torch_dtype": "float16"},
            # input_data=example_input,
        )

        # ov_model = ov.convert_model(tracing_model, example_input=input_data)
        output_path = Path(IR_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        # ov.save_model(ov_model, output_path / "openvino_model.xml")
        tokenizer.save_pretrained(IR_DIR)

    # ov_model = OVModelForCausalLM.from_pretrained(IR_DIR)
    # ov_model.config.quantization_config = None
    # ov_model.model = nncf.compress_weights(ov_model.model)

    # IR_DIR = IR_DIR + "_full_compression"
    # ov_model.save_pretrained(IR_DIR)
    # tokenizer.save_pretrained(IR_DIR)
