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
import time
from functools import partial

import numpy as np
from datasets import load_dataset
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

import nncf


def main():
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    OUTPUT_DIR = "llama_3_2_1b_precision_scope_compressed"

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False, compile=False)

    def transform_fn(data, tokenizer):
        tokenized_text = tokenizer(data["text"], return_tensors="np")
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]

        inputs = {}
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1
        inputs["position_ids"] = position_ids

        batch_size = input_ids.shape[0]
        inputs["beam_idx"] = np.arange(batch_size, dtype=int)

        return inputs

    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

    # Llama-3.2-1B has 16 layers (0..15), first=0, last=15
    precision_scope = {
        # First and last layer down_proj to INT8_SYM for accuracy
        "*layers.0*down_proj*": nncf.CompressWeightsMode.INT8_SYM,
        "*layers.15*down_proj*": nncf.CompressWeightsMode.INT8_SYM,
        # All o_proj layers to INT2_SYM for aggressive compression
        "*o_proj*": nncf.CompressWeightsMode.INT2_SYM,
    }

    model.model = nncf.compress_weights(
        model.model,
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        precision_scope=precision_scope,
        scale_estimation=True,
        awq=True,
    )
    model.save_pretrained(OUTPUT_DIR)

    model = OVModelForCausalLM.from_pretrained(OUTPUT_DIR)

    messages = [{"role": "user", "content": "What is PyTorch?"}]
    batch_feature = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = batch_feature["input_ids"]

    start_t = time.time()
    output = model.generate(input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
