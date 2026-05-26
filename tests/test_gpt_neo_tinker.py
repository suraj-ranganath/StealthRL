#!/usr/bin/env python3
"""Check if GPT-Neo can be used with Tinker API."""

from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers

# Test 1: Can we load GPT-Neo tokenizer?
print("=" * 80)
print("TEST 1: Loading GPT-Neo tokenizer")
print("=" * 80)
try:
    tokenizer = get_tokenizer("EleutherAI/gpt-neo-2.7B")
    print(f"✅ Successfully loaded gpt-neo-2.7B tokenizer")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Tokenizer type: {type(tokenizer).__name__}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: What renderers are available?
print("\n" + "=" * 80)
print("TEST 2: Available Tinker Renderers")
print("=" * 80)
available_renderers = {
    "qwen3": "Qwen3Renderer",
    "qwen3_instruct": "Qwen3InstructRenderer", 
    "qwen3_no_thinking": "Qwen3DisableThinkingRenderer",
    "deepseek": "DeepSeekV3Renderer",
    "deepseek_no_thinking": "DeepSeekV3DisableThinkingRenderer",
    "llama3": "Llama3Renderer",
    "gpt_oss": "GptOssRenderer",
    "role_colon": "RoleColonRenderer",
}

for name, cls in available_renderers.items():
    try:
        tokenizer = get_tokenizer("gpt2")  # Use simple tokenizer for test
        renderer = renderers.get_renderer(name, tokenizer=tokenizer)
        print(f"✅ {name:25} -> {cls}")
    except Exception as e:
        print(f"❌ {name:25} -> Error: {str(e)[:50]}")

# Test 3: GPT-Neo inference capability
print("\n" + "=" * 80)
print("TEST 3: GPT-Neo Compatibility Analysis")
print("=" * 80)
print("""
GPT-Neo in StealthRL:
├── USE CASE 1: Detector (WORKING ✅)
│   └── Fast-DetectGPT uses gpt-neo-2.7B for detection
│       - Loads model with transformers
│       - Computes logits/probabilities
│       - No generation needed
│
├── USE CASE 2: RL Training Base Model (LIMITED ❌)
│   └── Tinker API requires specific renderers
│       - Qwen3, DeepSeek, Llama3, etc.
│       - GPT-Neo NOT in supported renderers
│       - Would need custom GptOssRenderer or RoleColonRenderer
│
└── USE CASE 3: Alternative Detector LM (POSSIBLE ⚠️)
    └── Could replace gpt2 perplexity model with gpt-neo
        - gpt-neo-2.7B > gpt2 for quality
        - Load via transformers.pipeline()
        - Compute PPL via logprobs

RECOMMENDATION:
  For RL training: Use Qwen3, DeepSeek, or Llama3 (native Tinker support)
  For detection: Already using gpt-neo-2.7B ✅
  For perplexity: Could upgrade gpt2 → gpt-neo-1.3B (faster)
""")

# Test 4: Can we modify config to use different base model?
print("\n" + "=" * 80)
print("TEST 4: Supported RL Training Models")
print("=" * 80)
print("""
Current config uses: Qwen/Qwen3-4B-Instruct-2507

Options that work with Tinker:
1. Qwen models: Qwen3, Qwen2.5 (via qwen3 renderer)
2. DeepSeek: DeepSeek-V3 (via deepseek renderer) 
3. Llama: Llama3 (via llama3 renderer)
4. Other: Anything that follows GPT convention (via gpt_oss or role_colon)

GPT-Neo status: 
  - Not directly supported by Tinker renderers
  - But CAN use via GptOssRenderer if format compatible
  - Recommend: Stick with Qwen3, or try "gpt_oss" renderer
""")
