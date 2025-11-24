
from datasets import load_dataset
from collections import defaultdict
import torch
import torch.nn as nn


def collect_activations(model, tokenizer, device='cuda', apply_chat_template=True, subset_size=128):
    """
    Load AutoModelForCausalLM, collect activations from linear layers on GSM8K data,
    and visualize them with matplotlib.
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to run on ('cuda' or 'cpu')
    """

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prev_device = model.device
    model.to(device)
    model.eval()
    
    # Dictionary to store activations
    activations = defaultdict(list)
    
    def get_activation_hook(name):
        """Create a hook function that captures activations."""
        def hook(module, input, output):
            # Get input tensor
            if isinstance(input, tuple):
                tensor = input[0]
            else:
                tensor = input
            
            # Store statistics: max absolute value per input channel
            if len(tensor.shape) >= 2:
                # tensor shape: [batch, seq_len, features] or [batch, features]
                # We want max over batch and sequence dimensions
                if len(tensor.shape) == 3:
                    max_vals = tensor.abs().max(dim=0)[0].max(dim=0)[0]  # max over batch and seq
                else:
                    max_vals = tensor.abs().max(dim=0)[0]  # max over batch
                
                activations[name].append(max_vals)#tensor[0, :, :].detach().cpu())

        return hook
    
    # Register hooks for all linear layers except lm_head
    hooks = []
    linear_layer_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
            linear_layer_names.append(name)
            print(f"Registered hook for: {name}")
    
    print(f"\nTotal linear layers hooked: {len(hooks)}")
    
    # Load GSM8K dataset
    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Get one sample from GSM8K
    
    
    for i, sample in enumerate(dataset):
        if i >= subset_size:
            break
        question = sample['question']
        answer = sample['answer']
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer[:100]}...")  # Print first 100 chars
        
        if apply_chat_template:
            # Prepare input
            messages = [
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"Question: {question}\nAnswer: {answer}"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Forward pass to collect activations
        print("\nRunning forward pass to collect activations...")
        with torch.no_grad():
            model(**inputs)

    print(f"Activations collected for {len(activations)} layers")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    
    model.to(prev_device)
    return activations
