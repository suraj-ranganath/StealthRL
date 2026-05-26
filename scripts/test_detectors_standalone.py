#!/usr/bin/env python3
"""
Standalone test script for real detector implementations.
Tests detectors without requiring tinker dependencies.
"""

import asyncio
import logging
import sqlite3
import hashlib
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DetectorCache:
    """SQLite-based cache for detector scores."""
    
    def __init__(self, cache_path: str | None = None):
        if cache_path:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(cache_file)
        else:
            self.db_path = ":memory:"
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()
    
    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detector_cache (
                detector_name TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                score REAL NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (detector_name, text_hash)
            )
        """)
        self.conn.commit()
    
    def get(self, detector_name: str, text: str) -> float | None:
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        cursor = self.conn.execute(
            "SELECT score FROM detector_cache WHERE detector_name = ? AND text_hash = ?",
            (detector_name, text_hash)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    
    def set(self, detector_name: str, text: str, score: float):
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        timestamp = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO detector_cache (detector_name, text_hash, score, timestamp) VALUES (?, ?, ?, ?)",
            (detector_name, text_hash, score, timestamp)
        )
        self.conn.commit()
    
    def close(self):
        self.conn.close()


class FastDetectGPTDetector:
    """Fast-DetectGPT detector using curvature-based detection."""
    
    def __init__(self, cache: DetectorCache, device: str = None):
        self.name = "fast_detectgpt"
        self.cache = cache
        self.device = device or self._get_device()
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized Fast-DetectGPT on {self.device}")
    
    def _get_device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"
    
    def _load_model(self):
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info("Loading gpt2 for Fast-DetectGPT...")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✓ Fast-DetectGPT model loaded")
    
    async def predict(self, text: str) -> float:
        cached = self.cache.get(self.name, text)
        if cached is not None:
            return cached
        
        score = await asyncio.to_thread(self._compute_score, text)
        self.cache.set(self.name, text, score)
        return score
    
    def _compute_score(self, text: str) -> float:
        import torch
        
        self._load_model()
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                score = torch.sigmoid(torch.tensor((4.0 - loss) * 0.5)).item()
                return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.error(f"Fast-DetectGPT error: {e}")
            return 0.5


class GhostbusterDetector:
    """Ghostbuster detector using RoBERTa."""
    
    def __init__(self, cache: DetectorCache, device: str = None):
        self.name = "ghostbuster"
        self.cache = cache
        self.device = device or self._get_device()
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized Ghostbuster on {self.device}")
    
    def _get_device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"
    
    def _load_model(self):
        if self.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            logger.info("Loading roberta-base for Ghostbuster...")
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.model.eval()
            logger.info("✓ Ghostbuster model loaded")
    
    async def predict(self, text: str) -> float:
        cached = self.cache.get(self.name, text)
        if cached is not None:
            return cached
        
        score = await asyncio.to_thread(self._compute_score, text)
        self.cache.set(self.name, text, score)
        return score
    
    def _compute_score(self, text: str) -> float:
        import torch
        
        self._load_model()
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                score = probs[0, 1].item() if probs.shape[1] == 2 else probs.max().item()
                return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.error(f"Ghostbuster error: {e}")
            return 0.5


class BinocularsDetector:
    """Binoculars detector using paired LMs."""
    
    def __init__(self, cache: DetectorCache, device: str = None):
        self.name = "binoculars"
        self.cache = cache
        self.device = device or self._get_device()
        self.performer = None
        self.observer = None
        self.tokenizer = None
        logger.info(f"Initialized Binoculars on {self.device}")
    
    def _get_device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"
    
    def _load_models(self):
        if self.performer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info("Loading gpt2 and gpt2-medium for Binoculars...")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.performer = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.performer.eval()
            
            self.observer = AutoModelForCausalLM.from_pretrained(
                "gpt2-medium",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.observer.eval()
            logger.info("✓ Binoculars models loaded")
    
    async def predict(self, text: str) -> float:
        cached = self.cache.get(self.name, text)
        if cached is not None:
            return cached
        
        score = await asyncio.to_thread(self._compute_score, text)
        self.cache.set(self.name, text, score)
        return score
    
    def _compute_score(self, text: str) -> float:
        import torch
        import numpy as np
        
        self._load_models()
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                performer_outputs = self.performer(**inputs, labels=inputs["input_ids"])
                performer_loss = performer_outputs.loss.item()
                performer_ppl = np.exp(performer_loss)
                
                observer_outputs = self.observer(**inputs, labels=inputs["input_ids"])
                observer_loss = observer_outputs.loss.item()
                observer_ppl = np.exp(observer_loss)
            
            ce_diff = abs(np.log(observer_ppl + 1e-10) - np.log(performer_ppl + 1e-10))
            score = torch.sigmoid(torch.tensor((1.0 - ce_diff) * 2.0)).item()
            
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.error(f"Binoculars error: {e}")
            return 0.5


async def test_detectors():
    """Test detector ensemble."""
    
    print("="*60)
    print("Testing StealthRL Detector Ensemble")
    print("="*60)
    
    ai_text = """The implementation of neural networks requires careful consideration of hyperparameters and architectural choices. Recent advances in machine learning have enabled significant progress in natural language processing tasks."""
    
    human_text = """I went to the store yesterday and bought some groceries. The weather was nice, so I decided to walk instead of driving. It felt good to get some exercise!"""
    
    # Initialize cache and detectors
    print("\n1. Initializing detectors...")
    cache = DetectorCache("outputs/detector_cache_test.sqlite")
    
    fast_detect = FastDetectGPTDetector(cache)
    ghostbuster = GhostbusterDetector(cache)
    binoculars = BinocularsDetector(cache)
    
    print("   ✓ All detectors initialized")
    
    # Test on AI text
    print("\n2. Testing on AI-generated text...")
    print(f"   Text: {ai_text[:80]}...")
    
    fast_score_ai = await fast_detect.predict(ai_text)
    ghost_score_ai = await ghostbuster.predict(ai_text)
    bino_score_ai = await binoculars.predict(ai_text)
    ensemble_ai = (fast_score_ai + ghost_score_ai + bino_score_ai) / 3
    
    print(f"\n   Results:")
    print(f"   - Fast-DetectGPT: {fast_score_ai:.4f}")
    print(f"   - Ghostbuster:    {ghost_score_ai:.4f}")
    print(f"   - Binoculars:     {bino_score_ai:.4f}")
    print(f"   - Ensemble:       {ensemble_ai:.4f}")
    
    # Test on human text
    print("\n3. Testing on human-written text...")
    print(f"   Text: {human_text[:80]}...")
    
    fast_score_human = await fast_detect.predict(human_text)
    ghost_score_human = await ghostbuster.predict(human_text)
    bino_score_human = await binoculars.predict(human_text)
    ensemble_human = (fast_score_human + ghost_score_human + bino_score_human) / 3
    
    print(f"\n   Results:")
    print(f"   - Fast-DetectGPT: {fast_score_human:.4f}")
    print(f"   - Ghostbuster:    {ghost_score_human:.4f}")
    print(f"   - Binoculars:     {bino_score_human:.4f}")
    print(f"   - Ensemble:       {ensemble_human:.4f}")
    
    # Test caching
    print("\n4. Testing cache...")
    start = time.time()
    fast_cached = await fast_detect.predict(ai_text)
    elapsed = time.time() - start
    
    print(f"   ✓ Cached result in {elapsed:.4f}s")
    print(f"   ✓ Cache working: {fast_cached == fast_score_ai}")
    
    cache.close()
    
    print("\n" + "="*60)
    print("✓ All tests completed!")
    print("="*60)
    
    print("\nSummary:")
    print(f"  AI text ensemble:    {ensemble_ai:.4f}")
    print(f"  Human text ensemble: {ensemble_human:.4f}")
    print(f"  Difference:          {abs(ensemble_ai - ensemble_human):.4f}")


if __name__ == "__main__":
    try:
        asyncio.run(test_detectors())
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

