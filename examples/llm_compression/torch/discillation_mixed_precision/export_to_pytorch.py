from utils import export_to_pytorch
from utils import load_to_pytorch
from main import generate_answer
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    # pretrained = "meta-llama/Llama-3.2-1B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(pretrained)
    # ckpt_file = Path("output/last/nncf_checkpoint.pth")
    # model_dir = Path("output/last/pt_model_for_eval")
    # tokenizer.save_pretrained(model_dir)
    
    
    pretrained = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    ckpt_file = Path("output_qwen_3_4B/last/nncf_checkpoint.pth")
    model_dir = Path("output_qwen_3_4B/last/pt_model_for_eval")
    export_to_pytorch(pretrained, ckpt_file, model_dir)
    tokenizer.save_pretrained(model_dir)

