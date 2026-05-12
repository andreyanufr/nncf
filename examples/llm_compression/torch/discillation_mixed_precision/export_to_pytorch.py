from pathlib import Path

from transformers import AutoTokenizer
from utils import export_to_pytorch

if __name__ == "__main__":
    # pretrained = "meta-llama/Llama-3.2-1B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(pretrained)
    # ckpt_file = Path("output/last/nncf_checkpoint.pth")
    # model_dir = Path("output/last/pt_model_for_eval")
    # tokenizer.save_pretrained(model_dir)

    pretrained = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    ckpt_file = Path("output_qwen_3_4B_l1/last_sym_no_nncf_equalizing_gs_32_64_distill_vdiv_no_hvo/nncf_checkpoint.pth")
    #ckpt_file = Path("output_qwen_3_4B_l1/nncf_checkpoint_ep_5.pth")
    model_dir = Path("output_qwen_3_4B_l1/last_sym_no_nncf_equalizing_gs_32_64_distill_vdiv_no_hvo/pt_model_for_eval")
    export_to_pytorch(pretrained, ckpt_file, model_dir)
    tokenizer.save_pretrained(model_dir)
