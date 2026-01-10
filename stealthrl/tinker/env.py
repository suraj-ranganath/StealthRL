"""
StealthRL RL Environment for Tinker.

This module defines the RL environment for training StealthRL policies using Tinker.
The environment takes AI-generated text and rewards the policy for generating
semantically-equivalent, high-quality, detector-evasive, and fair paraphrases.
"""

import logging
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition, TokensWithLogprobs
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)

logger = logging.getLogger(__name__)


class StealthEnv(Env):
    """
    RL environment for StealthRL text paraphrasing.
    
    The agent receives an AI-generated text and metadata (domain, ESL flag),
    then generates a paraphrase. The reward is computed using:
    - R_det: Detector evasion (weighted ensemble of detectors)
    - R_sem: Semantic similarity to original (E5 encoder)
    - R_ppl: Fluency/perplexity (frozen LM, penalize extremes)
    - R_fair: ESL fairness penalty
    
    Total reward: R = α*R_det + β*R_sem + γ*R_ppl + δ*R_fair
    """
    
    def __init__(
        self,
        ai_text: str,
        human_reference: str,
        domain: str,
        is_esl: bool,
        renderer: renderers.Renderer,
        reward_fn,  # Will be CompositeReward instance
        convo_prefix: List[renderers.Message] | None = None,
    ):
        """
        Initialize StealthRL environment.
        
        Args:
            ai_text: The AI-generated text to paraphrase
            human_reference: Human-written reference text (for similarity baseline)
            domain: Domain of text (academic, informal, etc.)
            is_esl: Whether text is ESL-style (for fairness evaluation)
            renderer: Tinker renderer for formatting prompts
            reward_fn: Composite reward function instance
            convo_prefix: Optional conversation prefix (few-shot examples)
        """
        self.ai_text = ai_text
        self.human_reference = human_reference
        self.domain = domain
        self.is_esl = is_esl
        self.renderer = renderer
        self.reward_fn = reward_fn
        self.convo_prefix = convo_prefix or []
        self.last_paraphrase_text = ""
        self.last_parse_success = False
        
    @property
    def stop_condition(self) -> StopCondition:
        """Return stop sequences for generation."""
        return self.renderer.get_stop_sequences()
    
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """
        Return initial observation with prompt.
        
        The prompt instructs the model to paraphrase the AI text while
        maintaining semantic meaning and quality.
        """
        # Build prompt with instruction and AI text
        prompt_messages = self.convo_prefix + [
            {
                "role": "user",
                "content": self._build_paraphrase_prompt()
            }
        ]
        
        observation = self.renderer.build_generation_prompt(prompt_messages)
        return observation, self.stop_condition
    
    def _build_paraphrase_prompt(self) -> str:
        """Build the paraphrase instruction prompt."""
        return (
            f"Please paraphrase the following text while maintaining its meaning "
            f"and ensuring it reads naturally:\n\n"
            f"{self.ai_text}\n\n"
            f"Paraphrased text:"
        )
    
    async def step(self, action: Action) -> StepResult:
        """
        Compute reward for the generated paraphrase.
        
        Args:
            action: Generated token sequence (paraphrase)
        
        Returns:
            StepResult with reward and metrics
        """
        # Parse the generated paraphrase
        message, parse_success = self.renderer.parse_response(action)
        paraphrase_text = message["content"]
        
        self.last_paraphrase_text = paraphrase_text
        self.last_parse_success = bool(parse_success and paraphrase_text.strip())

        metrics = {
            "parse_success": 1.0 if self.last_parse_success else 0.0,
            "text_length": len(paraphrase_text) if self.last_parse_success else 0,
        }

        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


@dataclass(frozen=True)
class StealthEnvGroupBuilder(EnvGroupBuilder):
    """
    Builds a group of StealthRL environments for GRPO-style training.
    
    All environments in the group share the same AI text (prompt), so
    we can center rewards across multiple paraphrase attempts (GRPO).
    """
    
    ai_text: str
    human_reference: str
    domain: str
    is_esl: bool
    renderer: renderers.Renderer
    reward_fn: Any  # CompositeReward instance
    num_envs: int
    convo_prefix: List[renderers.Message] | None = None
    
    async def make_envs(self) -> Sequence[Env]:
        """Create group of identical environments (same prompt, different rollouts)."""
        return [
            StealthEnv(
                ai_text=self.ai_text,
                human_reference=self.human_reference,
                domain=self.domain,
                is_esl=self.is_esl,
                renderer=self.renderer,
                reward_fn=self.reward_fn,
                convo_prefix=self.convo_prefix,
            )
            for _ in range(self.num_envs)
        ]
    
    async def compute_group_rewards(
        self, trajectory_group: List[Trajectory], env_group: Sequence[Env]
    ) -> List[tuple[float, Metrics]]:
        """
        Compute group-level rewards (optional).
        
        For StealthRL, we use per-step rewards only. Group rewards
        could be used for comparing multiple paraphrases, but we
        leave this at 0 for now.
        """
        results: List[tuple[float, Metrics]] = []
        valid_indices: List[int] = []
        originals: List[str] = []
        paraphrases: List[str] = []
        references: List[str] = []
        domains: List[str] = []
        esl_flags: List[bool] = []

        for idx, env in enumerate(env_group):
            env_obj = env
            paraphrase_text = getattr(env_obj, "last_paraphrase_text", "")
            parse_success = getattr(env_obj, "last_parse_success", False)
            if not parse_success or not paraphrase_text.strip():
                results.append(
                    (
                        -1.0,
                        {
                            "reward/total": -1.0,
                            "reward/detector": 0.0,
                            "reward/semantic": 0.0,
                            "reward/perplexity": 0.0,
                            "reward/fairness": 0.0,
                            "detector_prob": 0.5,
                            "semantic_sim": 0.0,
                            "perplexity": 0.0,
                            "text_length": 0,
                        },
                    )
                )
                continue

            valid_indices.append(idx)
            originals.append(env_obj.ai_text)
            paraphrases.append(paraphrase_text)
            references.append(env_obj.human_reference)
            domains.append(env_obj.domain)
            esl_flags.append(env_obj.is_esl)
            results.append((0.0, {}))

        if valid_indices:
            if hasattr(self.reward_fn, "compute_batch"):
                reward_results = await self.reward_fn.compute_batch(
                    original_texts=originals,
                    paraphrase_texts=paraphrases,
                    human_references=references,
                    domains=domains,
                    is_esl_flags=esl_flags,
                )
            else:
                reward_results = []
                for original, paraphrase, reference, domain, is_esl in zip(
                    originals, paraphrases, references, domains, esl_flags, strict=True
                ):
                    reward_results.append(
                        await self.reward_fn.compute(
                            original_text=original,
                            paraphrase_text=paraphrase,
                            human_reference=reference,
                            domain=domain,
                            is_esl=is_esl,
                        )
                    )

            for idx, reward_result in zip(valid_indices, reward_results, strict=True):
                total_reward = reward_result["total_reward"]
                metrics = {
                    "reward/total": total_reward,
                    "reward/detector": reward_result.get("detector_reward", 0.0),
                    "reward/semantic": reward_result.get("semantic_reward", 0.0),
                    "reward/perplexity": reward_result.get("perplexity_reward", 0.0),
                    "reward/fairness": reward_result.get("fairness_reward", 0.0),
                    "detector_prob": reward_result.get("detector_prob", 0.5),
                    "semantic_sim": reward_result.get("semantic_sim", 0.0),
                    "perplexity": reward_result.get("perplexity", 0.0),
                    "text_length": reward_result.get("text_length", 0),
                }
                for key in (
                    "time/reward/total",
                    "time/reward/detector",
                    "time/reward/semantic",
                    "time/reward/perplexity",
                ):
                    if key in reward_result:
                        metrics[key] = reward_result[key]
                results[idx] = (total_reward, metrics)

        return results
