#!/usr/bin/env python3
"""
Simple example of programmatic usage for Llama quantization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantizer import Quantizer, QuantizationConfig


def main():
    # Configuration
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "./quantized_llama_programmatic"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda"  # or "cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Creating quantization config...")
    # Create quantization configuration
    config = QuantizationConfig(
        nbits=4,              # 4-bit quantization
        sym=True,             # Symmetric quantization
        balancing_method='sinq',  # Sinkhorn balancing
        group_size=128        # Group size 128
    )
    
    print("Quantizing model...")
    # Create quantizer and quantize
    quantizer = Quantizer(config)
    quantizer.quantize_llama(model)
    
    print("Saving model...")
    # Save quantized model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Done! Model saved to {output_dir}")
    
    # Optional: Test the quantized model
    print("\nTesting quantized model...")
    prompt = "What is the meaning of life?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
