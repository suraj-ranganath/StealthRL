"""
Optimized sampling for StealthRL using Tinker's sample_async with num_samples parameter.

This module patches Tinker's do_group_rollout_and_filter_constant_reward to use
OptimizedTinkerTokenCompleter, which generates all group_size rollouts in ONE API call.

Expected speedup: 2-4x for sampling (60s → 15-30s for 2048 texts).
"""

import asyncio
import logging
from typing import Optional, List, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import Tinker modules (may not be available in all environments)
try:
    import tinker
    from tinker_cookbook.rl import train as rl_train
    from tinker_cookbook.rl.types import (
        EnvGroupBuilder, TokenCompleter, StopCondition,
        TokensWithLogprobs, TrajectoryGroup, Env
    )
    from tinker_cookbook.logtree import logtree
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False


# Only define these classes if Tinker is available
if TINKER_AVAILABLE:
    
    @dataclass
    class OptimizedTinkerTokenCompleter(TokenCompleter):
        """
        Optimized TokenCompleter using sample_async with num_samples.
        
        Default TinkerTokenCompleter: Calls sample_async(num_samples=1) 8 times
        OptimizedTinkerTokenCompleter: Calls sample_async(num_samples=8) ONCE
        
        First __call__: Generates all group_size samples together, caches them
        Subsequent calls: Returns next cached sample
        
        Expected: 2-4x speedup from shared KV cache + parallel decoding.
        """
        
        sampling_client: tinker.SamplingClient
        max_tokens: int
        group_size: int
        temperature: float = 1.0
        _samples_cache: Optional[List[TokensWithLogprobs]] = field(default=None, init=False, repr=False)
        _cache_index: int = field(default=0, init=False, repr=False)
        
        async def __call__(
            self, model_input: tinker.ModelInput, stop: StopCondition
        ) -> TokensWithLogprobs:
            """Sample action - batches all group samples on first call."""
            
            # First call: generate all samples together
            if self._samples_cache is None:
                sample_result = await self.sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=self.group_size,  # Generate all rollouts together!
                    sampling_params=tinker.SamplingParams(
                        stop=stop,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    ),
                )
                
                # Cache all samples
                self._samples_cache = []
                for sequence in sample_result.sequences:
                    tokens = sequence.tokens
                    logprobs = sequence.logprobs
                    assert logprobs is not None, "Logprobs required for GRPO"
                    self._samples_cache.append(
                        TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs)
                    )
                
                self._cache_index = 0
            
            # Return next cached sample
            sample = self._samples_cache[self._cache_index]
            self._cache_index += 1
            return sample
    
    
    @logtree.scope_header_decorator
    async def do_group_rollout_optimized(
        env_group_builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> TrajectoryGroup:
        """Modified do_group_rollout with cache reset for OptimizedTinkerTokenCompleter."""
        
        envs_G: Sequence[Env] = await env_group_builder.make_envs()
        
        # Reset cache for new group
        if isinstance(policy, OptimizedTinkerTokenCompleter):
            policy._samples_cache = None
            policy._cache_index = 0
        
        # Do rollouts (policy batches sampling on first call)
        trajectories_G = await asyncio.gather(*[
            rl_train.do_single_rollout(policy, env) for env in envs_G
        ])
        
        # Compute rewards
        rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G, envs_G)
        rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
        
        # Log trajectories
        with logtree.scope_header("Trajectory Summary"):
            for i, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
                rows = []
                step_reward_sum = 0.0
                for t_idx, t in enumerate(traj.transitions):
                    step_reward_sum += t.reward
                    rows.append({
                        "step": t_idx,
                        "ob_len": t.ob.length,
                        "ac_len": len(t.ac.tokens),
                        "reward": f"{t.reward:.3f}",
                    })
                rows.append({
                    "step": "final",
                    "ob_len": traj.final_ob.length,
                    "ac_len": "-",
                    "reward": f"{final_reward:.3f}",
                })
                rows.append({
                    "step": "total",
                    "ob_len": "-",
                    "ac_len": "-",
                    "reward": f"{step_reward_sum + final_reward:.3f}",
                })
                logtree.table(rows, caption=f"Trajectory {i}")
        
        return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
    
    
    @logtree.scope_header_decorator
    async def do_group_rollout_and_filter_constant_reward_optimized(
        sampling_client: tinker.SamplingClient,
        env_group_builder: EnvGroupBuilder,
        max_tokens: int,
        temperature: float,
        do_remove_constant_reward_groups: bool,
        enable_logging: bool = True,
    ) -> Optional[TrajectoryGroup]:
        """
        Optimized replacement for rl_train.do_group_rollout_and_filter_constant_reward.
        
        Uses OptimizedTinkerTokenCompleter: sample_async(num_samples=group_size)
        instead of TinkerTokenCompleter: sample_async(num_samples=1) × group_size.
        """
        # Get group size
        envs_G = await env_group_builder.make_envs()
        group_size = len(envs_G)
        
        # Use optimized policy
        policy = OptimizedTinkerTokenCompleter(
            sampling_client=sampling_client,
            max_tokens=max_tokens,
            temperature=temperature,
            group_size=group_size,
        )
        
        with logtree.optional_enable_logging(enable_logging):
            trajectory_group = await do_group_rollout_optimized(env_group_builder, policy)
        
        # Filter constant rewards
        trajectory_groups = [trajectory_group]
        if do_remove_constant_reward_groups:
            trajectory_groups = rl_train.remove_constant_reward_groups(trajectory_groups)
        if len(trajectory_groups) == 0:
            return None
        return trajectory_groups[0]


def monkey_patch_tinker_sampling() -> bool:
    """
    Patch Tinker's do_group_rollout_and_filter_constant_reward to use optimized sampling.
    
    Returns:
        True if patch successful, False otherwise
    """
    if not TINKER_AVAILABLE:
        logger.warning("Tinker not available, cannot apply sampling optimization")
        return False
    
    if not hasattr(rl_train, 'do_group_rollout_and_filter_constant_reward'):
        logger.warning("Could not find rl_train.do_group_rollout_and_filter_constant_reward")
        return False
    
    try:
        # Store original
        if not hasattr(rl_train, '_original_do_group_rollout_and_filter_constant_reward'):
            rl_train._original_do_group_rollout_and_filter_constant_reward = (
                rl_train.do_group_rollout_and_filter_constant_reward
            )
        
        # Replace with optimized version
        rl_train.do_group_rollout_and_filter_constant_reward = (
            do_group_rollout_and_filter_constant_reward_optimized
        )
        
        logger.info("✓ Patched rl_train.do_group_rollout_and_filter_constant_reward")
        logger.info("  Using sample_async(num_samples=group_size) for 2-4x speedup")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch: {e}")
        return False
