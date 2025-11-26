#!/usr/bin/env python3
"""
Example: Paraphrase text using a trained StealthRL model.
"""

import argparse
import json
from pathlib import Path

from stealthrl.models import load_stealthrl_model


def paraphrase_text(text: str, model, tokenizer) -> str:
    """
    Paraphrase input text using StealthRL model.
    
    Args:
        text: Input text to paraphrase
        model: Trained StealthRL model
        tokenizer: Model tokenizer
        
    Returns:
        Paraphrased text
    """
    # Format as instruction-following prompt
    prompt = f"Paraphrase the following text while preserving its meaning:\n\n{text}\n\nParaphrased version:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the paraphrased portion
    paraphrased = paraphrased.split("Paraphrased version:")[-1].strip()
    
    return paraphrased


def main():
    parser = argparse.ArgumentParser(description="Paraphrase text with StealthRL")
    parser.add_argument("--input", type=str, required=True, help="Input text or file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model name")
    parser.add_argument("--output_path", type=str, default="outputs/stealthrl_samples.jsonl",
                        help="Output file path")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading StealthRL model from {args.model_path}")
    model, tokenizer = load_stealthrl_model(
        base_model=args.base_model,
        lora_path=args.model_path
    )
    
    # Determine if input is text or file
    input_path = Path(args.input)
    if input_path.exists():
        with open(input_path, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [args.input]
    
    # Paraphrase texts
    results = []
    for text in texts:
        print(f"\nOriginal: {text[:100]}...")
        paraphrased = paraphrase_text(text, model, tokenizer)
        print(f"Paraphrased: {paraphrased[:100]}...")
        
        results.append({
            "original": text,
            "paraphrased": paraphrased
        })
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
