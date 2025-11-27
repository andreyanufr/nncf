import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig
import numpy as np
from huggingface_hub import snapshot_download
from typing import List, Tuple, Optional, Dict
import time

from eval import evaluate_model

from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import BaseQuantizeConfig
from hqq.utils.patching import prepare_for_inference
from hqq.utils.generation_hf import patch_model_for_compiled_runtime

model_name = "meta-llama/Llama-3.2-3B-Instruct"
device = "cuda:0"
compute_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=compute_dtype, 
    #attn_implementation="kernels-community/flash-attn3:flash_attention", 
    device_map=device,
    #quantization_config=HqqConfig(nbits=4, group_size=64),
)

model.config.use_cache = True
model.generation_config.cache_implementation = "static"
AutoHQQHFModel.quantize_model(model, quant_config=BaseQuantizeConfig(nbits=4, group_size=64, axis=1), compute_dtype=compute_dtype, device=device)
prepare_for_inference(model, backend="gemlite") 
patch_model_for_compiled_runtime(model, tokenizer, warmup=False, max_new_tokens=1000, patch_accelerate=True, pre_compile=None)

results = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    tasks="",
    eval_ppl="gsm8k",
    batch_size=8
)
task_results = results["gsm8k"] #perplexity / ppl

print(model_name, task_results)
