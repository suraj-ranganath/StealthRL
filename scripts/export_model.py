#!/usr/bin/env python3
"""
Export and access trained models from StealthRL training runs.

This script helps you access trained model checkpoints stored on Tinker's servers.
Tinker stores checkpoints remotely and provides them as tinker:// URIs.

Usage:
    python scripts/export_model.py <run_directory>
    
Example:
    python scripts/export_model.py outputs/runs/my_training_run
"""

import argparse
import json
import sys
from pathlib import Path


def read_checkpoint_info(run_dir: Path) -> dict:
    """Read checkpoint info from the run directory."""
    checkpoint_info_path = run_dir / "checkpoints" / "final_checkpoint_info.json"
    
    if not checkpoint_info_path.exists():
        raise FileNotFoundError(
            f"Checkpoint info not found at: {checkpoint_info_path}\n"
            f"Make sure the training run completed successfully."
        )
    
    with open(checkpoint_info_path) as f:
        return json.load(f)


def print_checkpoint_info(info: dict):
    """Pretty print checkpoint information."""
    print("\n" + "="*70)
    print("📦 Trained Model Checkpoints")
    print("="*70)
    
    print(f"\n🎯 Model Details:")
    print(f"  Model ID:    {info['model_id']}")
    print(f"  Base Model:  {info['base_model']}")
    print(f"  LoRA Rank:   {info['lora_rank']}")
    
    print(f"\n💾 Checkpoint Paths:")
    for name, path in info['checkpoints'].items():
        print(f"  {name}: {path}")
    
    print(f"\n📝 Usage Examples:")
    print(f"\n  1. For Inference (sampling from trained model):")
    print(f"     ```python")
    print(f"     import tinker")
    print(f"     service_client = tinker.ServiceClient()")
    print(f"     sampling_client = service_client.create_sampling_client(")
    print(f"         model_path='{info['checkpoints']['sampler_weights']}'")
    print(f"     )")
    print(f"     ```")
    
    print(f"\n  2. To Resume Training:")
    print(f"     ```python")
    print(f"     import tinker")
    print(f"     service_client = tinker.ServiceClient()")
    print(f"     training_client = service_client.create_lora_training_client(")
    print(f"         base_model='{info['base_model']}',")
    print(f"         rank={info['lora_rank']}")
    print(f"     )")
    print(f"     training_client.load_state('{info['checkpoints']['final_state']}')") 
    print(f"     ```")
    
    print("\n" + "="*70)


def generate_inference_script(info: dict, output_path: Path):
    """Generate a ready-to-use inference script."""
    script = f'''#!/usr/bin/env python3
"""
Inference script for trained model: {info['model_id']}
Generated from: {output_path.parent}
"""

import asyncio
import tinker
from tinker.types import ModelInput, SamplingParams


async def main():
    # Initialize Tinker client
    service_client = tinker.ServiceClient()
    
    # Load the trained model
    print("Loading trained model...")
    sampling_client = service_client.create_sampling_client(
        model_path="{info['checkpoints']['sampler_weights']}"
    )
    
    # Get tokenizer
    tokenizer = sampling_client.get_tokenizer()
    
    # Example prompt (modify as needed)
    prompt_text = "Your prompt here"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt = ModelInput.from_ints(tokens=prompt_tokens)
    
    # Sampling parameters (adjust as needed)
    params = SamplingParams(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9
    )
    
    # Generate response
    print(f"Prompt: {{prompt_text}}")
    print("Generating response...")
    result = await sampling_client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
    response_tokens = (await result.result_async()).sequences[0].tokens
    response_text = tokenizer.decode(response_tokens)
    
    print(f"Response: {{response_text}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open(output_path, 'w') as f:
        f.write(script)
    
    # Make executable
    output_path.chmod(0o755)
    print(f"\n✓ Inference script saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Access trained model checkpoints from StealthRL runs"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the training run directory (e.g., outputs/runs/my_run)"
    )
    parser.add_argument(
        "--generate-script",
        action="store_true",
        help="Generate a ready-to-use inference script"
    )
    
    args = parser.parse_args()
    
    # Validate run directory
    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Read checkpoint info
        info = read_checkpoint_info(args.run_dir)
        
        # Print checkpoint information
        print_checkpoint_info(info)
        
        # Generate inference script if requested
        if args.generate_script:
            script_path = args.run_dir / "inference.py"
            generate_inference_script(info, script_path)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nMake sure your training run completed successfully.", file=sys.stderr)
        print("The checkpoint info file is created at the end of training.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
