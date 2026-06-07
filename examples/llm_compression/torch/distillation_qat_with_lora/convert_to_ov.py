from argparse import ArgumentParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#from examples.llm_compression.torch.distillation_qat_with_lora.main import export_to_openvino
from main import export_to_openvino





def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--nncf_checkpoint", type=str, help="Path to the NNCF checkpoint")
    parser.add_argument("--output_dir", type=str, default="./ov_model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pretrained == "test":
        from main_multimodal import get_test_model
        args.pretrained = get_test_model()
    export_to_openvino(args.pretrained, args.nncf_checkpoint, args.output_dir)
