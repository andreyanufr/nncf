#!/usr/bin/env python3
"""
Example script for quantizing Llama-3.1-8B-Instruct model with weight balancing.

This script demonstrates how to:
- Load a Llama model from HuggingFace
- Apply weight balancing with configurable parameters
- Quantize the model to lower precision (2, 4, or 8 bits)
- Save the quantized model
- Optionally evaluate the model on sample prompts
"""

import argparse
import logging
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantizer import Quantizer, QuantizationConfig
from eval import evaluate_model

import gc


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize Llama-3.1-8B-Instruct with weight balancing"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID or local path to model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_llama_3.1_8b",
        help="Directory to save the quantized model"
    )
    
    # Quantization arguments
    parser.add_argument(
        "--nbits",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Number of bits for quantization (2, 4, or 8)"
    )
    parser.add_argument(
        "--sym",
        action="store_true",
        help="Use symmetric quantization (default: asymmetric)"
    )
    parser.add_argument(
        "--balancing_method",
        type=str,
        default="absmean",
        choices=["sinq", "absmean"],
        help="Weight balancing method: 'sinq' (Sinkhorn) or 'absmean' (absolute mean)"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=64,
        help="Group size for quantization (default: 64)"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for model loading and quantization"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model loading"
    )
    
    parser.add_argument(
        "--per_layer",
        action="store_true",
        #default=True,
        help="Per layer quantization with not mergable scales."
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation on sample prompts after quantization"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate during evaluation"
    )
    
    # Other arguments
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for accessing gated models"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext2",
        help="Name of the dataset for evaluation"
    )
    
    return parser.parse_args()


def get_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def load_model(args):
    """Load the model and tokenizer from HuggingFace."""
    logger.info(f"Loading model: {args.model_id}")
    logger.info(f"Device: {args.device}, Dtype: {args.dtype}")
    
    dtype = get_dtype(args.dtype)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        token=args.token
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        #device_map=args.device if args.device == "cuda" else None,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
        use_cache=True,
    )#.to(args.device)
    
    print(
        "CUDA memory allocated and reserved (GB):",
        torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9  # in GB
    )
    memory_alloc = torch.cuda.memory_allocated() / 1e9
    print("Memory allocated after model loading (GB):", memory_alloc)
    
    # if args.device == "cpu":
    #     model = model.to(args.device)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def print_tensor_memory_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.dtype, obj.device, obj.shape)
        except:
            pass


def quantize_model(model, args):
    """Quantize the model using the specified configuration."""
    logger.info("Starting model quantization...")
    logger.info(f"Configuration: nbits={args.nbits}, sym={args.sym}, "
                f"balancing_method={args.balancing_method}, group_size={args.group_size}")
    
    # Create quantization config
    quant_config = QuantizationConfig(
        nbits=args.nbits,
        sym=args.sym,
        balancing_method=args.balancing_method,
        group_size=args.group_size
    )
    
    # Create quantizer
    quantizer = Quantizer(quant_config)
    
    # Quantize the model
    start_time = time.time()
    if args.per_layer:
        quantizer.quantize_per_layer(model)
    else:
        quantizer.quantize_llama(model)
    elapsed_time = time.time() - start_time
    
    model = model.to(args.device)
    
    model = torch.compile(model)

    cleanup()
    #print_tensor_memory_usage()
    cleanup()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    logger.info(f"Quantization completed in {elapsed_time:.2f} seconds")
    
    print(
        "CUDA memory allocated and reserved (GB):",
        torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9  # in GB
    )
    memory_alloc = torch.cuda.memory_allocated() / 1e9
    print("Memory allocated after quantization (GB):", memory_alloc)
    
    return model


def save_model(model, tokenizer, output_dir):
    """Save the quantized model and tokenizer."""
    logger.info(f"Saving quantized model to: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    logger.info("Model and tokenizer saved successfully")


def generate_model(model, tokenizer, args):
    """Evaluate the quantized model on sample prompts."""
    logger.info("Evaluating quantized model...")
    
    sample_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
    ]
    
    model.eval()
    
    for i, prompt in enumerate(sample_prompts, 1):
        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"Prompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        # Generate
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            generation_time = time.time() - start_time
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated: {generated_text}")
        logger.info(f"Generation time: {generation_time:.2f}s")


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Llama-3.1-8B-Instruct Quantization with Weight Balancing")
    logger.info("="*80)
    
    # Load model and tokenizer
    model, tokenizer = load_model(args)
    
    #Quantize model
    model = quantize_model(model, args)
    
    # Save quantized model
    # save_model(model, tokenizer, args.output_dir)
    
    # Optional evaluation
    if args.eval:
        generate_model(model, tokenizer, args)
    
    logger.info("="*80)
    logger.info("Quantization process completed successfully!")
    logger.info("="*80)
    
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks="",
        eval_ppl=args.dataset_name,
        batch_size=8
    )
    task_results = results[args.dataset_name] #perplexity / ppl

    print(args.model_id, task_results)


if __name__ == "__main__":
    main()
