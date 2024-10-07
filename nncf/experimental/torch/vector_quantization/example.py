from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from scheme import compress_phi

model_id = "microsoft/Phi-3-mini-4k-instruct"


model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


compress_phi(model.model)



