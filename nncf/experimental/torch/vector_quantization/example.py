import torch
from scheme import compress_llama
from scheme import compress_phi
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def compress_phi_example():
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    compress_phi(model.model)

    dst_path = "/home/aanuf/proj/int4_with_data/ov_microsoft/pt"

    model.save_pretrained(dst_path)
    tokenizer.save_pretrained(dst_path)


def compress_llama_example():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    compress_llama(model.model)

    dst_path = "/home/aanuf/proj/int4_with_data/ov_meta-llama/Meta-Llama-3-8B-Instruct/pt_kp_op_rect_group_"

    model.save_pretrained(dst_path)
    tokenizer.save_pretrained(dst_path)


compress_llama_example()

# compress_phi_example()
