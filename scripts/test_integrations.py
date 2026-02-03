#!/usr/bin/env python
"""Test all integrations: Tinker, SilverSpeak, PadBen."""

import os
import sys


def test_tinker_api_key():
    """Check Tinker API key."""
    print("=== Checking Tinker API Key ===")
    api_key = os.environ.get("TINKER_API_KEY", "")
    if api_key:
        print(f"✓ TINKER_API_KEY set ({len(api_key)} chars)")
        return True
    else:
        print("✗ TINKER_API_KEY not set")
        print("  Get your key from: https://tinker-console.thinkingmachines.ai/")
        print('  Set it: export TINKER_API_KEY="your-key-here"')
        return False


def test_tinker_import():
    """Test Tinker imports."""
    print("\n=== Testing Tinker Import ===")
    try:
        import tinker
        print("✓ tinker module imported")
        from tinker import ServiceClient
        print("✓ ServiceClient imported")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_tinker_connection():
    """Test actual Tinker connection."""
    print("\n=== Testing Tinker Connection ===")
    try:
        from tinker import ServiceClient
        
        client = ServiceClient()
        caps = client.get_server_capabilities()
        print(f"✓ Connected to Tinker server")
        print(f"  Supported models: {caps.supported_models[:3]}...")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def test_silverspeak():
    """Test SilverSpeak import and functionality."""
    print("\n=== Testing SilverSpeak ===")
    try:
        # New API (v0.2+)
        from silverspeak import random_attack, greedy_attack
        print("✓ SilverSpeak imported successfully (v0.2+ API)")
        
        # Quick test
        test_text = "Hello World"
        result = random_attack(test_text, 0.5)
        print(f'  Test: "{test_text}" -> "{result}"')
        
        # Check if actually changed
        if test_text != result:
            print("  ✓ Text was transformed")
        else:
            print("  (No change at 50% rate - try higher rate)")
        return True
    except ImportError as e:
        print(f"✗ SilverSpeak import failed: {e}")
        print("  Install with: pip install silverspeak")
        return False


def test_padben():
    """Test PadBen dataset loading."""
    print("\n=== Testing PadBen Dataset ===")
    try:
        from datasets import load_dataset
        
        # PadBen uses configs, not splits directly
        ds = load_dataset("JonathanZha/PADBen", "exhaustive-task2", split="train[:10]")
        print(f"✓ PadBen loaded: {len(ds)} samples")
        print(f"  Columns: {ds.column_names}")
        
        if len(ds) > 0:
            sample = ds[0]
            text = sample.get('sentence', sample.get('text', ''))[:80]
            print(f"  Sample: text='{text}...', label={sample.get('label')}")
        return True
    except Exception as e:
        print(f"✗ PadBen loading failed: {e}")
        return False


def test_eval_methods():
    """Test eval methods can be loaded."""
    print("\n=== Testing Eval Methods ===")
    try:
        # Add parent to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from eval.methods import METHOD_REGISTRY, get_method
        
        print(f"✓ Method registry loaded ({len(METHOD_REGISTRY)} methods)")
        print(f"  Available: {list(METHOD_REGISTRY.keys())}")
        
        # Test M0 (no deps)
        m0 = get_method("m0")
        print(f"  ✓ M0 (NoAttack) instantiated")
        
        # Test M5 (SilverSpeak)
        try:
            m5 = get_method("m5")
            print(f"  ✓ M5 (Homoglyph/SilverSpeak) instantiated")
        except Exception as e:
            print(f"  ✗ M5 failed: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Eval methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stealthrl_checkpoint():
    """Test StealthRL checkpoint loading."""
    print("\n=== Testing StealthRL Checkpoint ===")
    try:
        import json
        checkpoint_path = "checkpoints/ckpt_example.json"
        
        if not os.path.exists(checkpoint_path):
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            return False
        
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        
        print(f"✓ Checkpoint loaded")
        print(f"  Base model: {ckpt.get('base_model')}")
        print(f"  Model ID: {ckpt.get('model_id')}")
        print(f"  Sampler path: {ckpt['checkpoints'].get('sampler_weights')}")
        
        return True
    except Exception as e:
        print(f"✗ Checkpoint loading failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("StealthRL Integration Tests")
    print("=" * 60)
    
    results = {}
    
    results["tinker_api_key"] = test_tinker_api_key()
    results["tinker_import"] = test_tinker_import()
    
    if results["tinker_api_key"] and results["tinker_import"]:
        results["tinker_connection"] = test_tinker_connection()
    else:
        results["tinker_connection"] = False
        print("\n=== Skipping Tinker Connection (API key or import missing) ===")
    
    results["silverspeak"] = test_silverspeak()
    results["padben"] = test_padben()
    results["eval_methods"] = test_eval_methods()
    results["stealthrl_checkpoint"] = test_stealthrl_checkpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("All integrations ready! ✓")
        return 0
    else:
        print("Some integrations need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
