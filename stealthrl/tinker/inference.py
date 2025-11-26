"""
Chunking inference pipeline for long text generation.

Handles texts > 512 tokens by:
1. Splitting into overlapping chunks
2. Generating N candidates per chunk
3. Selecting best candidate based on reward
4. Merging chunks with overlap handling
"""

import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tiktoken


@dataclass
class ChunkCandidate:
    """Single candidate paraphrase for a chunk."""
    
    text: str
    reward: float
    detector_prob: float
    semantic_sim: float
    perplexity_score: float
    
    
@dataclass
class ChunkResult:
    """Result for a single chunk with best candidate selected."""
    
    chunk_id: int
    original_text: str
    best_candidate: ChunkCandidate
    all_candidates: List[ChunkCandidate]
    start_token: int
    end_token: int


class ChunkingInference:
    """
    Chunking inference for long text paraphrasing.
    
    Splits long texts into overlapping chunks, generates multiple candidates
    per chunk, selects best based on reward, and merges with overlap handling.
    """
    
    def __init__(
        self,
        model,
        reward_fn,
        chunk_size: int = 512,
        overlap: int = 50,
        num_candidates: int = 4,
        temperature: float = 0.9,
        top_p: float = 0.95,
        encoding: str = "cl100k_base",
    ):
        """
        Initialize chunking inference.
        
        Args:
            model: Tinker model or sampling client
            reward_fn: TinkerCompositeReward for scoring candidates
            chunk_size: Maximum tokens per chunk (default: 512)
            overlap: Overlapping tokens between chunks (default: 50)
            num_candidates: Number of candidates to generate per chunk (default: 4)
            temperature: Sampling temperature (default: 0.9)
            top_p: Nucleus sampling threshold (default: 0.95)
            encoding: Tiktoken encoding name (default: "cl100k_base")
        """
        self.model = model
        self.reward_fn = reward_fn
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.top_p = top_p
        
        try:
            self.tokenizer = tiktoken.get_encoding(encoding)
        except Exception:
            # Fallback to simple word-based chunking
            self.tokenizer = None
    
    def _split_into_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of (chunk_text, start_token, end_token) tuples
        """
        if self.tokenizer is not None:
            # Token-based chunking
            tokens = self.tokenizer.encode(text)
            chunks = []
            
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append((chunk_text, start, end))
                
                if end >= len(tokens):
                    break
                    
                # Move forward by (chunk_size - overlap)
                start += self.chunk_size - self.overlap
            
            return chunks
        else:
            # Fallback: sentence-based chunking
            sentences = re.split(r'([.!?]+\s+)', text)
            chunks = []
            current_chunk = ""
            current_start = 0
            current_tokens = 0
            
            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]  # Include punctuation
                
                sentence_tokens = len(sentence.split())
                
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append((current_chunk.strip(), current_start, current_start + current_tokens))
                    
                    # Start new chunk with overlap
                    overlap_text = " ".join(current_chunk.split()[-self.overlap:])
                    current_chunk = overlap_text + " " + sentence
                    current_start = current_start + current_tokens - self.overlap
                    current_tokens = self.overlap + sentence_tokens
                else:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append((current_chunk.strip(), current_start, current_start + current_tokens))
            
            return chunks
    
    async def _generate_candidates(
        self,
        chunk_text: str,
        original_full_text: str,
        human_reference: Optional[str],
        domain: str,
        is_esl: bool,
    ) -> List[ChunkCandidate]:
        """
        Generate multiple candidate paraphrases for a chunk.
        
        Args:
            chunk_text: Text chunk to paraphrase
            original_full_text: Full original text (for context)
            human_reference: Optional human reference for full text
            domain: Text domain
            is_esl: Whether author is ESL
            
        Returns:
            List of ChunkCandidate objects
        """
        prompt = f"Please paraphrase the following text while maintaining its meaning and style:\n\n{chunk_text}"
        
        # Generate N candidates
        candidates = []
        for _ in range(self.num_candidates):
            try:
                # Generate candidate
                paraphrase = await self.model.generate(
                    prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.chunk_size + 100,  # Allow some extra room
                )
                
                # Score candidate using reward function
                reward_result = await self.reward_fn.compute(
                    original_text=chunk_text,
                    paraphrase_text=paraphrase,
                    human_reference=human_reference,  # Use full reference if available
                    domain=domain,
                    is_esl=is_esl,
                )
                
                candidates.append(ChunkCandidate(
                    text=paraphrase,
                    reward=reward_result["total_reward"],
                    detector_prob=reward_result["detector_prob"],
                    semantic_sim=reward_result["semantic_sim"],
                    perplexity_score=reward_result.get("perplexity_reward", 0.0),
                ))
            except Exception as e:
                print(f"Warning: Failed to generate/score candidate: {e}")
                continue
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[ChunkCandidate]) -> ChunkCandidate:
        """
        Select best candidate based on reward.
        
        Args:
            candidates: List of ChunkCandidate objects
            
        Returns:
            Best ChunkCandidate
        """
        if not candidates:
            raise ValueError("No candidates available for selection")
        
        # Sort by total reward (descending)
        candidates_sorted = sorted(candidates, key=lambda c: c.reward, reverse=True)
        return candidates_sorted[0]
    
    def _merge_chunks(self, chunk_results: List[ChunkResult]) -> str:
        """
        Merge chunk results handling overlaps.
        
        Strategy:
        - For overlapping regions, use text from chunk with higher reward
        - Smooth transitions at sentence boundaries when possible
        
        Args:
            chunk_results: List of ChunkResult objects
            
        Returns:
            Merged paraphrase text
        """
        if not chunk_results:
            return ""
        
        if len(chunk_results) == 1:
            return chunk_results[0].best_candidate.text
        
        # Sort by position
        chunk_results = sorted(chunk_results, key=lambda cr: cr.start_token)
        
        merged_text = chunk_results[0].best_candidate.text
        
        for i in range(1, len(chunk_results)):
            prev_result = chunk_results[i - 1]
            curr_result = chunk_results[i]
            
            # Check if chunks overlap
            if curr_result.start_token < prev_result.end_token:
                # Overlapping: try to find good merge point
                curr_text = curr_result.best_candidate.text
                
                # Simple strategy: find last sentence boundary in overlap region
                # and use that as merge point
                overlap_size = prev_result.end_token - curr_result.start_token
                
                # Estimate character position of overlap
                words_in_prev = len(merged_text.split())
                chars_per_word = len(merged_text) / max(words_in_prev, 1)
                overlap_chars = int(overlap_size * chars_per_word)
                
                # Find merge point (prefer sentence boundary)
                merge_point = len(merged_text) - overlap_chars
                sentence_breaks = [m.end() for m in re.finditer(r'[.!?]+\s+', merged_text[merge_point:])]
                
                if sentence_breaks:
                    # Use first sentence boundary after overlap start
                    merge_point = merge_point + sentence_breaks[0]
                    merged_text = merged_text[:merge_point] + curr_text
                else:
                    # No sentence boundary: use middle of overlap
                    merge_point = len(merged_text) - overlap_chars // 2
                    merged_text = merged_text[:merge_point] + curr_text
            else:
                # Non-overlapping: just concatenate
                merged_text += " " + curr_result.best_candidate.text
        
        return merged_text.strip()
    
    async def paraphrase(
        self,
        text: str,
        human_reference: Optional[str] = None,
        domain: str = "general",
        is_esl: bool = False,
    ) -> dict:
        """
        Paraphrase long text using chunking strategy.
        
        Args:
            text: Input text (can be > 512 tokens)
            human_reference: Optional human-written reference
            domain: Text domain
            is_esl: Whether author is ESL
            
        Returns:
            Dictionary with:
                - paraphrase: Merged paraphrase text
                - chunk_results: List of ChunkResult objects
                - num_chunks: Number of chunks processed
                - avg_reward: Average reward across chunks
                - avg_detector_prob: Average detector probability
        """
        # Split into chunks
        chunks = self._split_into_chunks(text)
        
        if len(chunks) == 0:
            return {
                "paraphrase": "",
                "chunk_results": [],
                "num_chunks": 0,
                "avg_reward": 0.0,
                "avg_detector_prob": 1.0,
            }
        
        # Process each chunk
        chunk_results = []
        for i, (chunk_text, start_token, end_token) in enumerate(chunks):
            # Generate candidates
            candidates = await self._generate_candidates(
                chunk_text=chunk_text,
                original_full_text=text,
                human_reference=human_reference,
                domain=domain,
                is_esl=is_esl,
            )
            
            if not candidates:
                # Fallback: use original chunk
                candidates = [ChunkCandidate(
                    text=chunk_text,
                    reward=-1.0,
                    detector_prob=1.0,
                    semantic_sim=1.0,
                    perplexity_score=0.0,
                )]
            
            # Select best
            best = self._select_best_candidate(candidates)
            
            chunk_results.append(ChunkResult(
                chunk_id=i,
                original_text=chunk_text,
                best_candidate=best,
                all_candidates=candidates,
                start_token=start_token,
                end_token=end_token,
            ))
        
        # Merge chunks
        merged_paraphrase = self._merge_chunks(chunk_results)
        
        # Compute aggregate statistics
        avg_reward = sum(cr.best_candidate.reward for cr in chunk_results) / len(chunk_results)
        avg_detector_prob = sum(cr.best_candidate.detector_prob for cr in chunk_results) / len(chunk_results)
        
        return {
            "paraphrase": merged_paraphrase,
            "chunk_results": chunk_results,
            "num_chunks": len(chunks),
            "avg_reward": avg_reward,
            "avg_detector_prob": avg_detector_prob,
        }


# Example usage
async def main():
    """Example: Paraphrase long text using chunking."""
    from stealthrl.tinker import TinkerCompositeReward
    import tinker
    
    # Initialize model (mock for example)
    class MockModel:
        async def generate(self, prompt, **kwargs):
            # In practice: call actual Tinker model
            return prompt.replace("Please paraphrase the following text while maintaining its meaning and style:\n\n", "")
    
    model = MockModel()
    
    # Initialize reward function
    reward_fn = TinkerCompositeReward(
        detector_weight=1.0,
        semantic_weight=1.0,
        perplexity_weight=0.5,
        fairness_weight=0.2,
        use_mock=True,  # Use mock for testing
    )
    
    # Initialize chunking inference
    chunker = ChunkingInference(
        model=model,
        reward_fn=reward_fn,
        chunk_size=512,
        overlap=50,
        num_candidates=4,
    )
    
    # Long text example
    long_text = """
    The implementation of neural networks requires careful consideration of multiple factors.
    First, the architecture must be designed to match the complexity of the task at hand.
    Second, hyperparameter tuning is essential for achieving optimal performance.
    Third, regularization techniques help prevent overfitting on training data.
    Finally, proper evaluation metrics must be selected to measure model quality.
    """ * 20  # Repeat to make it long
    
    # Paraphrase
    result = await chunker.paraphrase(
        text=long_text,
        domain="academic",
        is_esl=False,
    )
    
    print(f"Original length: {len(long_text)} chars")
    print(f"Paraphrase length: {len(result['paraphrase'])} chars")
    print(f"Number of chunks: {result['num_chunks']}")
    print(f"Average reward: {result['avg_reward']:.3f}")
    print(f"Average detector prob: {result['avg_detector_prob']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
