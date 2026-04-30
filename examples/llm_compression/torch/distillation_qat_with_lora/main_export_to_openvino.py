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
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import openvino as ov

import nncf
from nncf.parameters import StripFormat
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import SymmetricLoraQuantizer  # noqa: F401
from optimum.intel.openvino import OVConfig

from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM


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


PRETRAINED = "meta-llama/Llama-3.2-1B-Instruct"
CHECKPOINT_FILE = "output/last/nncf_checkpoint_epoch1.pth"
IR_DIR = "output/ov_model/"


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


class CausalLMForTracing(nn.Module):
    """
    Wraps a causal LM and returns only logits as a tensor for TorchScript tracing.

    :param model: Source causal language model.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Runs model inference in tensor-only mode.

        :param input_ids: Token ids tensor.
        :param attention_mask: Attention mask tensor.
        :return: Logits tensor.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]

with torch.no_grad():
    model_to_eval = AutoModelForCausalLM.from_pretrained(PRETRAINED, torch_dtype=torch.float32, device_map="cpu")
    model_to_eval = load_checkpoint(model_to_eval, CHECKPOINT_FILE)
    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.OV)
    model_to_eval.eval()
    # if hasattr(model_to_eval, "config"):
    #     model_to_eval.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    tracing_model = model_to_eval
    #tracing_model = CausalLMForTracing(model_to_eval)

    input_data = get_input_data(tokenizer)
    
    model_to_eval(**input_data)

    if False:
        export_from_model(model_to_eval, IR_DIR, device="cpu", compression_option="fp16", ov_config=OVConfig(dtype="fp16"))
        
        #ov_model = ov.convert_model(tracing_model, example_input=input_data)

        output_path = Path(IR_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        #ov.save_model(ov_model, output_path / "openvino_model.xml")
        tokenizer.save_pretrained(IR_DIR)
    
    ov_model = OVModelForCausalLM.from_pretrained(IR_DIR)
    ov_model.model = nncf.compress_weights(ov_model.model)
    
    IR_DIR = IR_DIR + "_full_compression"
    ov_model.save_pretrained(IR_DIR)
    tokenizer.save_pretrained(IR_DIR)
    
    
