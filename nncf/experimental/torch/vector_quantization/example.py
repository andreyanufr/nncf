import torch
from scheme import compress_llama, compute_stats
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

    model = AutoModelForCausalLM.from_pretrained(model_id)#, device_map="cuda:1", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_tokens=128)

    vq_names = ["o_proj", "k_proj"]#, "up_proj"]
    exclude = []#".0.", ".1.", ".2.", ".30.", ".31."]
    stats = None #compute_stats(model, tokenizer, vq_names, 128)#128)
    model = model.cpu().to(dtype=torch.float32)
    compress_llama(model.model, vq_names, exclude, stats=stats)

    dst_path = "/home/aanuf/proj/int4_with_data/ov_meta-llama/Meta-Llama-3-8B-Instruct/pt_mixed_2_8_4_group_wise_per_row_k_means_all"#_stats_2d_se"

    model.save_pretrained(dst_path)
    tokenizer.save_pretrained(dst_path)


compress_llama_example()

# compress_phi_example()
