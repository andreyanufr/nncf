from pathlib import Path

from transformers import AutoTokenizer
from utils import export_to_pytorch


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Export NNCF compressed model to PyTorch")
    parser.add_argument("--pretrained", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--ckpt_file", type=str, required=True, help="Path to NNCF checkpoint file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pretrained = args.pretrained
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    dst_dir = args.ckpt_file
    ckpt_file = Path(f"{dst_dir}/nncf_checkpoint.pth")
    model_dir = Path(f"{dst_dir}/pt_model_for_eval")
    mixture_file = Path(f"{dst_dir}/mixer_config.json")

    export_to_pytorch(pretrained, ckpt_file, model_dir, mixture_file=mixture_file)
    tokenizer.save_pretrained(model_dir)

    print(f"Model exported to {model_dir}")
