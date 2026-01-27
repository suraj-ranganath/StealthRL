"""
Optimized sampling for Tinker GRPO using sample_async with num_samples.

This module provides a faster sampling implementation that generates all
group_size rollouts in a single API call instead of making separate calls.

Expected speedup: 2-4x faster sampling (60s → 15-30s for 2048 texts)
"""

import logging
from typing import List
import tinker
from tinker import types

logger = logging.getLogger(__name__)


async def sample_group_optimized(
    sampling_client: tinker.SamplingClient,
    prompt: types.ModelInput,
    group_size: int,
    sampling_params: types.SamplingParams,
) -> List[types.ModelInput]:
    """
    Generate multiple rollouts from a single prompt using optimized batching.
    
    Uses sample_async with num_samples parameter to generate all rollouts
    in a single API call, significantly reducing overhead.
    
    Args:
        sampling_client: Tinker sampling client
        prompt: Tokenized prompt (ModelInput)
        group_size: Number of rollouts to generate (GRPO group_size)
        sampling_params: Sampling parameters (temp, top_p, max_tokens)
    
    Returns:
        List of ModelInput samples (length = group_size)
    """
    # Single API call generates all group_size samples together!
    response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=group_size,  # All rollouts in one call
        sampling_params=sampling_params,
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )
    
    # Extract samples from response
    samples = []
    for sample in response.samples:
        # Convert Sample to ModelInput
        # Sample has: tokens, logprobs, etc.
        # We need to combine prompt + generated tokens
        full_tokens = prompt.tokens + sample.tokens
        samples.append(types.ModelInput.from_ints(tokens=full_tokens))
    
    return samples


async def sample_batch_optimized(
    sampling_client: tinker.SamplingClient,
    prompts: List[types.ModelInput],
    group_size: int,
    sampling_params: types.SamplingParams,
) -> List[List[types.ModelInput]]:
    """
    Sample multiple prompts, each with multiple rollouts.
    
    For batch_size=256 prompts × group_size=8 rollouts:
    - Old: 2048 sequential API calls (slow!)
    - New: 256 API calls with num_samples=8 (much faster!)
    
    Args:
        sampling_client: Tinker sampling client
        prompts: List of tokenized prompts
        group_size: Number of rollouts per prompt
        sampling_params: Sampling parameters
    
    Returns:
        List of lists: [batch_size][group_size] samples
    """
    import asyncio
    
    # Create sampling tasks for all prompts
    tasks = [
        sample_group_optimized(
            sampling_client=sampling_client,
            prompt=prompt,
            group_size=group_size,
            sampling_params=sampling_params,
        )
        for prompt in prompts
    ]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    return results


def monkey_patch_tinker_sampling():
    """
    Monkey-patch Tinker's sampling to use optimized sample_async method.
    
    Call this before training to enable faster sampling.
    
    WARNING: This modifies Tinker's internal sampling behavior.
    Only use if you understand the implications.
    """
    try:
        from tinker_cookbook.rl import train as rl_train
        
        # Save original sampling function
        original_sample_group = getattr(rl_train, 'sample_group', None)
        
        if original_sample_group is None:
            logger.warning("Could not find rl_train.sample_group to patch")
            return False
        
        # Replace with optimized version
        rl_train.sample_group = sample_group_optimized
        
        logger.info("✓ Patched Tinker sampling to use optimized sample_async")
        logger.info("  Expected speedup: 2-4x faster sampling")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch Tinker sampling: {e}")
        return False


# Usage example
"""
from stealthrl.tinker.fast_sampling import monkey_patch_tinker_sampling

# Enable before training
if monkey_patch_tinker_sampling():
    print("Optimized sampling enabled!")
else:
    print("Using default Tinker sampling")

# Then run training as normal
trainer = StealthRLTrainer(config)
await trainer.train()
"""
