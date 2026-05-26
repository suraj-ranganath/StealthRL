"""
Model loading utilities for StealthRL.
"""

from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def load_base_model(
    model_name: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base language model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        torch_dtype: PyTorch dtype for model weights
        **kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        **kwargs
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def load_stealthrl_model(
    base_model: str,
    lora_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    **kwargs
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load a StealthRL model with LoRA adapters.
    
    Args:
        base_model: HuggingFace base model identifier
        lora_path: Path to LoRA adapter weights
        device: Device to load model on
        torch_dtype: PyTorch dtype for model weights
        **kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (peft_model, tokenizer)
    """
    # Load base model
    model, tokenizer = load_base_model(
        base_model,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer
