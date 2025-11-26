"""
SICO (Stylistic Imitation for Content Obfuscation) Baseline.

Prompt-based paraphrasing technique that uses carefully crafted prompts
to elicit natural-sounding paraphrases from LLMs without training.

Reference: Used as baseline in various AI detection evasion studies.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SICOConfig:
    """Configuration for SICO paraphrasing."""
    
    model_name: str = "gpt-3.5-turbo"  # or any LLM
    temperature: float = 0.7
    max_tokens: int = 512
    prompt_template: str = "paraphrase_preserve_meaning"
    num_candidates: int = 1  # Generate N candidates and pick best


class SICOParaphraser:
    """
    SICO prompt-based paraphrasing baseline.
    
    Uses zero-shot or few-shot prompting to generate paraphrases
    without any training. Serves as a strong, low-cost baseline.
    """
    
    PROMPT_TEMPLATES = {
        "paraphrase_preserve_meaning": (
            "Rewrite the following text in a natural, human-like way "
            "while preserving its exact meaning. Vary sentence structure "
            "and word choice, but maintain clarity and readability:\n\n{text}\n\n"
            "Paraphrased version:"
        ),
        
        "academic_style": (
            "Rewrite this academic text with natural, human-like phrasing. "
            "Maintain the scholarly tone but use varied sentence structures "
            "and natural transitions:\n\n{text}\n\n"
            "Rewritten version:"
        ),
        
        "conversational": (
            "Rewrite this text in a more conversational, natural style "
            "while keeping all the key information. Make it sound like "
            "something a knowledgeable person would say:\n\n{text}\n\n"
            "Conversational version:"
        ),
        
        "formal_informal": (
            "Convert this text to a slightly less formal style while "
            "maintaining professionalism. Use natural phrasing that a "
            "human writer would use:\n\n{text}\n\n"
            "Less formal version:"
        ),
        
        "dipper_style": (
            "Paraphrase the following text. Use different words and "
            "sentence structures, but keep the same meaning and level "
            "of detail:\n\n{text}\n\n"
            "Paraphrase:"
        ),
        
        "few_shot": (
            "I will show you examples of paraphrasing, then ask you to paraphrase a new text.\n\n"
            "Original: The implementation of neural networks requires careful consideration of architecture design.\n"
            "Paraphrase: Designing neural network architectures demands thoughtful planning and attention to structural details.\n\n"
            "Original: Machine learning algorithms can process large datasets efficiently when properly optimized.\n"
            "Paraphrase: With appropriate optimization, ML algorithms handle big data processing tasks effectively.\n\n"
            "Now paraphrase this text:\n\n{text}\n\n"
            "Paraphrase:"
        ),
    }
    
    def __init__(
        self,
        model,  # Tinker model or any LLM API
        config: Optional[SICOConfig] = None,
    ):
        """
        Initialize SICO paraphraser.
        
        Args:
            model: Model instance with generate() method
            config: SICO configuration
        """
        self.model = model
        self.config = config or SICOConfig()
        
        if self.config.prompt_template not in self.PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown prompt template: {self.config.prompt_template}. "
                f"Available: {list(self.PROMPT_TEMPLATES.keys())}"
            )
    
    def _build_prompt(self, text: str) -> str:
        """Build prompt from template."""
        template = self.PROMPT_TEMPLATES[self.config.prompt_template]
        return template.format(text=text)
    
    async def paraphrase(
        self,
        text: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate SICO paraphrase.
        
        Args:
            text: Input text to paraphrase
            temperature: Sampling temperature (override config)
            max_tokens: Max tokens (override config)
            
        Returns:
            Paraphrased text
        """
        prompt = self._build_prompt(text)
        
        paraphrase = await self.model.generate(
            prompt,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        
        # Post-process: remove common artifacts
        paraphrase = self._postprocess(paraphrase)
        
        return paraphrase
    
    def _postprocess(self, text: str) -> str:
        """Post-process generated paraphrase."""
        # Remove common prompt artifacts
        text = text.strip()
        
        # Remove "Paraphrased version:", "Rewritten version:", etc.
        prefixes = [
            "Paraphrased version:",
            "Rewritten version:",
            "Conversational version:",
            "Less formal version:",
            "Paraphrase:",
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text
    
    async def paraphrase_with_selection(
        self,
        text: str,
        reward_fn,
        human_reference: Optional[str] = None,
        domain: str = "general",
        is_esl: bool = False,
    ) -> Dict:
        """
        Generate multiple SICO candidates and select best based on reward.
        
        Args:
            text: Input text
            reward_fn: Reward function for scoring candidates
            human_reference: Optional human reference for semantic similarity
            domain: Text domain
            is_esl: Whether author is ESL
            
        Returns:
            Dictionary with best paraphrase and scores
        """
        candidates = []
        
        for i in range(self.config.num_candidates):
            # Generate candidate with slight temperature variation
            temp = self.config.temperature + (i * 0.1)
            paraphrase = await self.paraphrase(text, temperature=temp)
            
            # Score candidate
            reward_result = await reward_fn.compute(
                original_text=text,
                paraphrase_text=paraphrase,
                human_reference=human_reference,
                domain=domain,
                is_esl=is_esl,
            )
            
            candidates.append({
                "paraphrase": paraphrase,
                "reward": reward_result["total_reward"],
                "detector_prob": reward_result["detector_prob"],
                "semantic_sim": reward_result["semantic_sim"],
            })
        
        # Select best by total reward
        best = max(candidates, key=lambda c: c["reward"])
        
        return {
            "paraphrase": best["paraphrase"],
            "reward": best["reward"],
            "detector_prob": best["detector_prob"],
            "semantic_sim": best["semantic_sim"],
            "all_candidates": candidates,
        }


class SICOEnsemble:
    """
    Ensemble of SICO paraphrasers with different prompt templates.
    
    Tests multiple prompt strategies and selects best result.
    """
    
    def __init__(
        self,
        model,
        prompt_templates: Optional[List[str]] = None,
        config: Optional[SICOConfig] = None,
    ):
        """
        Initialize SICO ensemble.
        
        Args:
            model: Model instance
            prompt_templates: List of template names to use
            config: Base configuration
        """
        self.model = model
        self.config = config or SICOConfig()
        
        if prompt_templates is None:
            prompt_templates = [
                "paraphrase_preserve_meaning",
                "academic_style",
                "dipper_style",
            ]
        
        self.paraphrasers = []
        for template in prompt_templates:
            template_config = SICOConfig(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                prompt_template=template,
                num_candidates=1,
            )
            self.paraphrasers.append(SICOParaphraser(model, template_config))
    
    async def paraphrase(
        self,
        text: str,
        reward_fn,
        human_reference: Optional[str] = None,
        domain: str = "general",
        is_esl: bool = False,
    ) -> Dict:
        """
        Generate paraphrases with all templates and select best.
        
        Args:
            text: Input text
            reward_fn: Reward function for scoring
            human_reference: Optional human reference
            domain: Text domain
            is_esl: Whether author is ESL
            
        Returns:
            Dictionary with best paraphrase across all templates
        """
        results = []
        
        for paraphraser in self.paraphrasers:
            result = await paraphraser.paraphrase_with_selection(
                text=text,
                reward_fn=reward_fn,
                human_reference=human_reference,
                domain=domain,
                is_esl=is_esl,
            )
            result["template"] = paraphraser.config.prompt_template
            results.append(result)
        
        # Select best across all templates
        best = max(results, key=lambda r: r["reward"])
        
        return {
            "paraphrase": best["paraphrase"],
            "template": best["template"],
            "reward": best["reward"],
            "detector_prob": best["detector_prob"],
            "semantic_sim": best["semantic_sim"],
            "all_results": results,
        }


# Example usage
async def main():
    """Example: SICO paraphrasing."""
    from stealthrl.tinker import TinkerCompositeReward
    
    # Mock model for testing
    class MockModel:
        async def generate(self, prompt, **kwargs):
            # In practice: call actual LLM
            text = prompt.split(":")[-1].strip()
            return f"Paraphrased: {text}"
    
    model = MockModel()
    
    # Initialize SICO
    sico = SICOParaphraser(
        model=model,
        config=SICOConfig(
            prompt_template="paraphrase_preserve_meaning",
            temperature=0.7,
            num_candidates=3,
        ),
    )
    
    # Initialize reward function
    reward_fn = TinkerCompositeReward(
        detector_weight=1.0,
        semantic_weight=1.0,
        perplexity_weight=0.5,
        fairness_weight=0.2,
        use_mock=True,
    )
    
    # Paraphrase with candidate selection
    text = "The implementation of neural networks requires careful consideration."
    result = await sico.paraphrase_with_selection(
        text=text,
        reward_fn=reward_fn,
        domain="academic",
        is_esl=False,
    )
    
    print(f"Original: {text}")
    print(f"Best paraphrase: {result['paraphrase']}")
    print(f"Reward: {result['reward']:.3f}")
    print(f"Detector prob: {result['detector_prob']:.3f}")
    print(f"Semantic sim: {result['semantic_sim']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
