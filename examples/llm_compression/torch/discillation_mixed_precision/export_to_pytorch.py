from pathlib import Path

from transformers import AutoTokenizer
from utils import export_to_pytorch

if __name__ == "__main__":
    # pretrained = "meta-llama/Llama-3.2-1B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(pretrained)
    # ckpt_file = Path("output/last/nncf_checkpoint.pth")
    # model_dir = Path("output/last/pt_model_for_eval")
    # tokenizer.save_pretrained(model_dir)

    pretrained = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    # ckpt_file = Path("output_qwen_3_4B_l1/last_sym_no_nncf_equalizing_gs_32_64_distill_vdiv_another_sens_up_09/nncf_checkpoint.pth")
    # model_dir = Path("output_qwen_3_4B_l1/last_sym_no_nncf_equalizing_gs_32_64_distill_vdiv_another_sens_up_09/pt_model_for_eval")

    dst_dir = "output_qwen_3_8B/last_sym_nncf_equalizing_gs_64_64_fq_lr01_ep5_safe_scale"
    ckpt_file = Path(f"{dst_dir}/nncf_checkpoint.pth")
    model_dir = Path(f"{dst_dir}/pt_model_for_eval")

    export_to_pytorch(pretrained, ckpt_file, model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Model exported to {model_dir}")
