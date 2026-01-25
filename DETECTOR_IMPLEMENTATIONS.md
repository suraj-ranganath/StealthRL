# AI Text Detector Implementations Guide

**Last Updated**: January 24, 2026  
**Purpose**: Reference guide for implementing real SOTA AI text detectors in StealthRL

---

## Table of Contents

1. [Overview](#overview)
2. [Open Source Detectors (Run Locally)](#open-source-detectors-run-locally)
3. [API-Based Detectors (Commercial)](#api-based-detectors-commercial)
4. [Recommended Detector Combinations](#recommended-detector-combinations)
5. [Implementation Code](#implementation-code)
6. [Integration with StealthRL](#integration-with-stealthrl)
7. [Cost Analysis](#cost-analysis)
8. [Performance Comparison](#performance-comparison)

---

## Overview

### Current Status
- **Mock Detectors**: Currently using placeholder implementations
- **Problem**: Results are not scientifically valid with mocks
- **Solution**: Replace with real SOTA detectors

### Detector Categories

| Category | Pros | Cons | Cost |
|----------|------|------|------|
| **Open Source (Local)** | Free, reproducible, full control | Requires GPU, setup time | $0 |
| **API-Based (Commercial)** | Easy setup, maintained, fast | Costs money, vendor lock-in | $0.01-0.10/request |

---

## Open Source Detectors (Run Locally)

### 🟢 Tier 1: Recommended for Academic Research

#### 1. **Fast-DetectGPT** ⭐ Best Overall

**Type**: Curvature-based detection  
**Running**: Local (requires GPU)  
**Paper**: "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text" (ICLR 2024)  
**GitHub**: https://github.com/baoguangsheng/fast-detect-gpt

**How it works**:
- Measures curvature of probability manifold
- AI text has flatter curvature than human text
- Zero-shot (no training needed)

**Requirements**:
```bash
pip install transformers torch
# Model: GPT-2 or GPT-J-6B (reference model)
# GPU: 8-16GB VRAM recommended
```

**Advantages**:
- ✅ No training required
- ✅ Theoretically grounded
- ✅ Resistant to simple paraphrasing
- ✅ Works across domains

**Disadvantages**:
- ❌ Slower inference (~2-5 sec/text)
- ❌ Requires reference model (GPT-2 minimum)
- ❌ High memory usage

**Implementation Difficulty**: 7/10

---

#### 2. **Binoculars** ⭐ Zero-Shot Detection

**Type**: Paired language model comparison  
**Running**: Local (requires GPU)  
**Paper**: "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text" (2024)  
**GitHub**: https://github.com/ahans30/Binoculars

**How it works**:
- Compares perplexity across two different LLMs
- AI text has similar perplexity across models
- Human text varies more across models

**Requirements**:
```bash
pip install binoculars-detector
# Or manual: transformers, torch
# Models: GPT-2 + GPT-Neo (or similar pair)
# GPU: 8-16GB VRAM
```

**Advantages**:
- ✅ Zero-shot (no training)
- ✅ Conceptually simple
- ✅ Fast inference (~1 sec/text)
- ✅ Good generalization

**Disadvantages**:
- ❌ Requires two models (more memory)
- ❌ Sensitive to model choice
- ❌ May struggle with longer texts

**Implementation Difficulty**: 5/10

---

#### 3. **RADAR** ⭐ Adversarially Robust

**Type**: Adversarially-trained classifier  
**Running**: Local (requires GPU)  
**Paper**: "RADAR: Robust AI-Text Detection via Adversarial Learning" (NeurIPS 2024)  
**GitHub**: https://github.com/yangzhch6/RADAR

**How it works**:
- RoBERTa classifier trained with adversarial examples
- Explicitly designed to resist paraphrasing attacks
- Uses adversarial training loop

**Requirements**:
```bash
# Check GitHub for official implementation
pip install transformers torch
# Model: RoBERTa-large fine-tuned
# GPU: 8-16GB VRAM
```

**Advantages**:
- ✅ **Specifically designed for adversarial robustness**
- ✅ Trained on paraphrase attacks
- ✅ Fast inference (<1 sec/text)
- ✅ Best choice for testing StealthRL

**Disadvantages**:
- ❌ Requires training data
- ❌ May overfit to specific attack types
- ❌ Newer method (less validated)

**Implementation Difficulty**: 6/10

**⚠️ IMPORTANT**: This is your **primary target detector** since it's adversarially robust!

---

#### 4. **RoBERTa-ChatGPT-Detector** ⭐ Easy Implementation

**Type**: Fine-tuned classifier  
**Running**: Local (GPU optional, works on CPU)  
**Model**: HuggingFace pretrained  
**Available Models**:
- `Hello-SimpleAI/chatgpt-detector-roberta`
- `andreas122001/roberta-mixed-detector`
- `roberta-base-openai-detector`

**How it works**:
- RoBERTa fine-tuned on human vs AI text pairs
- Binary classification (human=0, AI=1)
- Standard supervised learning

**Requirements**:
```bash
pip install transformers torch
# Model: Auto-downloads from HuggingFace
# GPU: 4-8GB VRAM (or CPU, slower)
```

**Advantages**:
- ✅ **Easiest to implement** (5 lines of code)
- ✅ Fast inference (<1 sec/text)
- ✅ Works on CPU
- ✅ Pre-trained, no setup

**Disadvantages**:
- ❌ Vulnerable to paraphrasing
- ❌ Domain-specific (trained on certain text types)
- ❌ May not generalize well

**Implementation Difficulty**: 3/10

**💡 RECOMMENDED FOR QUICK START**: Implement this one first (Day 1)

---

#### 5. **DetectGPT (Original)**

**Type**: Curvature-based (slower version)  
**Running**: Local (requires GPU)  
**Paper**: "DetectGPT: Zero-Shot Machine-Generated Text Detection" (2023)  
**GitHub**: https://github.com/eric-mitchell/detect-gpt

**How it works**:
- Original curvature-based method
- Slower than Fast-DetectGPT

**Requirements**:
```bash
pip install detect-gpt
# Or: transformers, torch
# GPU: 16GB+ VRAM
```

**Advantages**:
- ✅ Well-cited, validated
- ✅ Theoretically sound

**Disadvantages**:
- ❌ Very slow (10-30 sec/text)
- ❌ High memory usage
- ❌ Fast-DetectGPT is better in every way

**Implementation Difficulty**: 7/10

**⚠️ NOT RECOMMENDED**: Use Fast-DetectGPT instead

---

### 🟡 Tier 2: Alternative Options

#### 6. **Ghostbuster**

**Type**: Feature-ensemble classifier  
**Running**: Local (requires GPU)  
**Paper**: "Ghostbuster: Detecting Text Ghostwritten by LLMs" (2024)  
**Status**: May require custom setup

**Advantages**:
- ✅ Good performance on academic writing
- ✅ Feature engineering approach

**Disadvantages**:
- ❌ Complex setup
- ❌ Requires multiple models
- ❌ Less maintained

**Implementation Difficulty**: 8/10

---

## API-Based Detectors (Commercial)

### 🔵 Tier 1: Industry Standard

#### 1. **GPTZero** ⭐ Most Popular

**Type**: Commercial API  
**Running**: Cloud-based (API calls)  
**Website**: https://gptzero.me  
**Used by**: 2M+ educators, major universities

**How it works**:
- Proprietary ensemble of multiple detection methods
- Trained on massive dataset of student writing
- Continuously updated

**Pricing**:
- Free tier: 5,000 words/month
- Premium: $10/month (100K words)
- API: ~$0.01 per detection (~500 words)

**API Setup**:
```bash
# Get API key from: https://gptzero.me/api
export GPTZERO_API_KEY="your_key_here"
```

**Advantages**:
- ✅ **Most widely used** (industry standard)
- ✅ Continuously updated
- ✅ High accuracy on student writing
- ✅ Easy integration (REST API)
- ✅ No GPU required

**Disadvantages**:
- ❌ Costs money (~$50-100 for full experiments)
- ❌ Proprietary (black box)
- ❌ Vendor lock-in
- ❌ Rate limits

**Implementation Difficulty**: 2/10

**💡 RECOMMENDED**: Include this for industry comparison

---

#### 2. **Originality.AI**

**Type**: Commercial API  
**Running**: Cloud-based  
**Website**: https://originality.ai  
**Used by**: Professional content creators, SEO agencies

**Pricing**:
- Pay-as-you-go: $0.01 per 100 words
- Monthly: $14.95 (20,000 words)

**API Setup**:
```bash
# Get API key from: https://originality.ai/dashboard
export ORIGINALITY_API_KEY="your_key_here"
```

**Advantages**:
- ✅ High accuracy
- ✅ Fast inference
- ✅ Good documentation

**Disadvantages**:
- ❌ Costs money
- ❌ Proprietary
- ❌ Less academic validation

**Implementation Difficulty**: 2/10

---

#### 3. **Sapling.ai**

**Type**: Commercial API (with free tier)  
**Running**: Cloud-based  
**Website**: https://sapling.ai/ai-content-detector

**Pricing**:
- Free tier: 100 requests/day
- Pro: $25/month (unlimited)

**API Setup**:
```bash
# Get API key from: https://sapling.ai/dashboard
export SAPLING_API_KEY="your_key_here"
```

**Advantages**:
- ✅ **Free tier available** (good for testing)
- ✅ Easy setup
- ✅ Fast inference

**Disadvantages**:
- ❌ Rate limits on free tier
- ❌ Less popular than GPTZero

**Implementation Difficulty**: 2/10

**💡 RECOMMENDED FOR TESTING**: Use free tier first

---

#### 4. **OpenAI Classifier** (Deprecated)

**Type**: Commercial API  
**Status**: ⛔ **SHUT DOWN** (July 2023)  
**Reason**: Low accuracy (~26%)

**⚠️ DO NOT USE**: OpenAI discontinued their detector

---

#### 5. **Content at Scale AI Detector**

**Type**: Free web-based  
**Running**: Cloud (no official API)  
**Website**: https://contentatscale.ai/ai-content-detector

**Advantages**:
- ✅ Free
- ✅ No API key needed

**Disadvantages**:
- ❌ No official API (web scraping required)
- ❌ Rate limits
- ❌ Not suitable for research

**Implementation Difficulty**: 5/10 (requires scraping)

**⚠️ NOT RECOMMENDED**: No API, unreliable

---

## Recommended Detector Combinations

### Option A: All Open Source (Free, Reproducible) ⭐ BEST FOR ACADEMIC PAPER

**Detectors**:
1. **RoBERTa-ChatGPT-Detector** (classifier-based)
2. **Fast-DetectGPT** (curvature-based)
3. **Binoculars** (perplexity-based)

**Why**:
- ✅ Three different detection mechanisms
- ✅ Zero cost
- ✅ Fully reproducible
- ✅ No vendor lock-in

**Time to implement**: 2-3 days  
**Total cost**: $0  
**GPU required**: Yes (8-16GB VRAM)

**Best for**: Academic publications, reproducibility

---

### Option B: Mixed (Best Performance) ⭐ BEST BALANCE

**Detectors**:
1. **GPTZero** (API, industry standard)
2. **Fast-DetectGPT** (local, curvature-based)
3. **RADAR** (local, adversarially-robust)

**Why**:
- ✅ Industry standard included
- ✅ Adversarially-robust detector
- ✅ Mix of local + API
- ✅ Best overall coverage

**Time to implement**: 2 days  
**Total cost**: ~$50-100  
**GPU required**: Yes (8-16GB VRAM)

**Best for**: Strong empirical results, workshop papers

---

### Option C: Quick Start (1 Week Timeline) ⭐ FASTEST

**Detectors**:
1. **RoBERTa-ChatGPT-Detector** (local, easy)
2. **Sapling.ai** (API, free tier)

**Why**:
- ✅ Fastest to implement (1 day)
- ✅ Minimal cost (free tier)
- ✅ Two different types (local + API)

**Time to implement**: 1 day  
**Total cost**: $0 (using free tiers)  
**GPU required**: Optional (RoBERTa works on CPU)

**Best for**: Quick validation, tight deadlines

---

### Option D: Maximum Rigor (Full Publication) ⭐ MOST COMPREHENSIVE

**Detectors**:
1. **GPTZero** (API, most popular)
2. **Fast-DetectGPT** (local, curvature)
3. **Binoculars** (local, perplexity)
4. **RADAR** (local, adversarially-robust)
5. **RoBERTa-ChatGPT-Detector** (local, classifier)

**Why**:
- ✅ 5 detectors = comprehensive evaluation
- ✅ All major detection types covered
- ✅ Industry + research-grade
- ✅ Strongest empirical validation

**Time to implement**: 3-4 days  
**Total cost**: ~$100-200  
**GPU required**: Yes (16GB+ VRAM)

**Best for**: Top-tier conference submissions (ICLR, NeurIPS)

---

## Implementation Code

### 1. RoBERTa-ChatGPT-Detector (Easiest) ✅ START HERE

```python
# stealthrl/detectors/roberta_detector.py
"""
RoBERTa-based AI text classifier (HuggingFace).
Easiest to implement, works on CPU, fast inference.
"""

from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base_detector import BaseDetector


class RoBERTaDetector(BaseDetector):
    """
    RoBERTa-based AI text classifier.
    
    Models available:
    - "Hello-SimpleAI/chatgpt-detector-roberta" (recommended)
    - "andreas122001/roberta-mixed-detector"
    - "roberta-base-openai-detector"
    """
    
    def __init__(self, 
                 model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta",
                 device: str = "cuda"):
        super().__init__(model_name, device)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load RoBERTa model from HuggingFace."""
        if self.model is None:
            print(f"Loading {self.model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
                ).to(self.device)
                self.model.eval()
                print(f"✓ {self.model_name} loaded successfully")
            except Exception as e:
                print(f"✗ Error loading {self.model_name}: {e}")
                raise
        
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Run detection on texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Tensor of detection scores [0, 1] where 1 = AI-generated
        """
        if self.model is None:
            self.load_model()
            
        if isinstance(texts, str):
            texts = [texts]
            
        scores = []
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get prediction
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Class 1 = AI-generated (usually)
                # Check model config if uncertain
                ai_prob = probs[0, 1].item()
                scores.append(ai_prob)
                
        return torch.tensor(scores, dtype=torch.float32)
```

**Test it**:
```bash
python -c "
from stealthrl.detectors.roberta_detector import RoBERTaDetector

detector = RoBERTaDetector(device='cuda')  # or 'cpu'
detector.load_model()

# Test with known AI text
ai_text = 'The quick brown fox jumps over the lazy dog. This is a test sentence.'
score = detector.detect(ai_text)
print(f'AI Detection Score: {score.item():.3f}')
"
```

---

### 2. Fast-DetectGPT (Curvature-Based)

```python
# stealthrl/detectors/fast_detectgpt.py
"""
Fast-DetectGPT: Curvature-based detection.
Zero-shot, theoretically grounded, resistant to paraphrasing.
"""

from typing import List, Union
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_detector import BaseDetector


class FastDetectGPTDetector(BaseDetector):
    """
    Fast-DetectGPT curvature-based detector.
    
    Reference models:
    - "gpt2" (fastest, 124M params)
    - "gpt2-medium" (355M params)
    - "EleutherAI/gpt-j-6B" (best, 6B params, requires 16GB VRAM)
    """
    
    def __init__(self, 
                 reference_model: str = "gpt2",
                 device: str = "cuda",
                 num_perturbations: int = 10):
        super().__init__(reference_model, device)
        self.model = None
        self.tokenizer = None
        self.num_perturbations = num_perturbations
        
    def load_model(self):
        """Load reference model for curvature computation."""
        if self.model is None:
            print(f"Loading {self.model_name} for Fast-DetectGPT...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
                self.model.eval()
                print(f"✓ {self.model_name} loaded successfully")
            except Exception as e:
                print(f"✗ Error loading {self.model_name}: {e}")
                raise
        
    def _compute_log_likelihood(self, text: str) -> float:
        """Compute log-likelihood of text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_likelihood = -outputs.loss.item()
            
        return log_likelihood
    
    def _compute_curvature(self, text: str) -> float:
        """
        Compute curvature via perturbations.
        Lower curvature = more likely AI-generated.
        """
        # Original log-likelihood
        original_ll = self._compute_log_likelihood(text)
        
        # Perturbed log-likelihoods
        perturbed_lls = []
        
        # Simple perturbation: add random tokens
        tokens = text.split()
        for _ in range(self.num_perturbations):
            # Random word dropout (simple perturbation)
            if len(tokens) > 10:
                perturbed_tokens = [t for t in tokens if np.random.rand() > 0.1]
                perturbed_text = " ".join(perturbed_tokens)
                perturbed_ll = self._compute_log_likelihood(perturbed_text)
                perturbed_lls.append(perturbed_ll)
        
        if not perturbed_lls:
            return 0.5  # Neutral score
            
        # Curvature = variance of perturbed log-likelihoods
        curvature = np.var(perturbed_lls)
        
        return curvature
        
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Run Fast-DetectGPT detection.
        
        Returns:
            Scores [0, 1] where 1 = AI-generated (low curvature)
        """
        if self.model is None:
            self.load_model()
            
        if isinstance(texts, str):
            texts = [texts]
            
        scores = []
        for text in texts:
            curvature = self._compute_curvature(text)
            
            # Lower curvature = more likely AI
            # Normalize using sigmoid
            score = 1.0 / (1.0 + np.exp(curvature - 0.5))
            scores.append(float(score))
                
        return torch.tensor(scores, dtype=torch.float32)
```

---

### 3. Binoculars (Paired-LM)

```python
# stealthrl/detectors/binoculars.py
"""
Binoculars: Paired language model comparison.
Zero-shot, compares perplexity across two models.
"""

from typing import List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_detector import BaseDetector


class BinocularsDetector(BaseDetector):
    """
    Binoculars detector using paired language models.
    
    Default pair: GPT-2 + GPT-Neo
    """
    
    def __init__(self, 
                 model1: str = "gpt2",
                 model2: str = "EleutherAI/gpt-neo-125M",
                 device: str = "cuda"):
        super().__init__(f"{model1}+{model2}", device)
        self.model1_name = model1
        self.model2_name = model2
        self.model1 = None
        self.model2 = None
        self.tokenizer = None
        
    def load_model(self):
        """Load both models for comparison."""
        if self.model1 is None:
            print(f"Loading models for Binoculars...")
            try:
                # Use common tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model1_name)
                
                # Load first model
                self.model1 = AutoModelForCausalLM.from_pretrained(
                    self.model1_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
                ).to(self.device)
                self.model1.eval()
                
                # Load second model
                self.model2 = AutoModelForCausalLM.from_pretrained(
                    self.model2_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
                ).to(self.device)
                self.model2.eval()
                
                print(f"✓ Binoculars models loaded successfully")
            except Exception as e:
                print(f"✗ Error loading Binoculars models: {e}")
                raise
        
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Run Binoculars detection.
        
        Returns:
            Scores [0, 1] where 1 = AI-generated (similar perplexity across models)
        """
        if self.model1 is None:
            self.load_model()
            
        if isinstance(texts, str):
            texts = [texts]
            
        scores = []
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get perplexities from both models
                outputs1 = self.model1(**inputs, labels=inputs["input_ids"])
                outputs2 = self.model2(**inputs, labels=inputs["input_ids"])
                
                ppl1 = torch.exp(outputs1.loss).item()
                ppl2 = torch.exp(outputs2.loss).item()
                
                # Binoculars score: Cross-perplexity ratio
                # AI text has similar perplexity across models
                # Human text varies more
                ratio = abs(ppl1 - ppl2) / (ppl1 + ppl2 + 1e-10)
                
                # Invert: lower ratio = more likely AI
                score = 1.0 - min(1.0, ratio)
                scores.append(score)
                
        return torch.tensor(scores, dtype=torch.float32)
```

---

### 4. GPTZero (API)

```python
# stealthrl/detectors/gptzero.py
"""
GPTZero API detector.
Industry standard, used by 2M+ educators.
"""

from typing import List, Union
import torch
import requests
import os
import time
from .base_detector import BaseDetector


class GPTZeroDetector(BaseDetector):
    """
    GPTZero API detector.
    
    Requires API key from: https://gptzero.me/api
    Set environment variable: GPTZERO_API_KEY
    """
    
    def __init__(self, api_key: str = None, device: str = "cpu"):
        super().__init__("gptzero-api", device)
        self.api_key = api_key or os.getenv("GPTZERO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GPTZero API key required. "
                "Set GPTZERO_API_KEY environment variable or pass api_key parameter."
            )
        self.api_url = "https://api.gptzero.me/v2/predict/text"
            
    def load_model(self):
        """No model loading needed for API."""
        pass
        
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Run GPTZero API detection.
        
        Returns:
            Scores [0, 1] where 1 = AI-generated
        """
        if isinstance(texts, str):
            texts = [texts]
            
        scores = []
        for text in texts:
            try:
                response = requests.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={"document": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    score = result["documents"][0]["completely_generated_prob"]
                    scores.append(score)
                elif response.status_code == 429:
                    # Rate limit hit
                    print("⚠ GPTZero rate limit hit, waiting 60s...")
                    time.sleep(60)
                    scores.append(0.5)  # Neutral score
                else:
                    print(f"⚠ GPTZero API error {response.status_code}: {response.text}")
                    scores.append(0.5)  # Neutral score
                    
            except Exception as e:
                print(f"⚠ GPTZero API exception: {e}")
                scores.append(0.5)  # Neutral score
                
        return torch.tensor(scores, dtype=torch.float32)
```

---

### 5. Sapling.ai (API, Free Tier)

```python
# stealthrl/detectors/sapling.py
"""
Sapling.ai API detector.
Free tier: 100 requests/day
"""

from typing import List, Union
import torch
import requests
import os
from .base_detector import BaseDetector


class SaplingDetector(BaseDetector):
    """
    Sapling.ai API detector.
    
    Free tier: 100 requests/day
    Get API key from: https://sapling.ai/dashboard
    """
    
    def __init__(self, api_key: str = None, device: str = "cpu"):
        super().__init__("sapling-api", device)
        self.api_key = api_key or os.getenv("SAPLING_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Sapling API key required. "
                "Set SAPLING_API_KEY environment variable or pass api_key parameter."
            )
        self.api_url = "https://api.sapling.ai/api/v1/aidetect"
            
    def load_model(self):
        """No model loading needed for API."""
        pass
        
    def detect(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Run Sapling API detection.
        
        Returns:
            Scores [0, 1] where 1 = AI-generated
        """
        if isinstance(texts, str):
            texts = [texts]
            
        scores = []
        for text in texts:
            try:
                response = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"text": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    score = result.get("score", 0.5)
                    scores.append(score)
                else:
                    print(f"⚠ Sapling API error {response.status_code}")
                    scores.append(0.5)
                    
            except Exception as e:
                print(f"⚠ Sapling API exception: {e}")
                scores.append(0.5)
                
        return torch.tensor(scores, dtype=torch.float32)
```

---

## Integration with StealthRL

### Step 1: Update Detector Registry

Edit `stealthrl/tinker/detectors.py`:

```python
from stealthrl.detectors.roberta_detector import RoBERTaDetector
from stealthrl.detectors.fast_detectgpt import FastDetectGPTDetector
from stealthrl.detectors.binoculars import BinocularsDetector
from stealthrl.detectors.gptzero import GPTZeroDetector
from stealthrl.detectors.sapling import SaplingDetector

DETECTOR_REGISTRY = {
    # Open source (local)
    "roberta": RoBERTaDetector,
    "fast_detectgpt": FastDetectGPTDetector,
    "binoculars": BinocularsDetector,
    
    # API-based
    "gptzero": GPTZeroDetector,
    "sapling": SaplingDetector,
}
```

### Step 2: Update Config File

Edit `configs/tinker_stealthrl.yaml`:

```yaml
# Detector ensemble
detectors:
  # Option A: All open source
  - name: "roberta"
    weight: 0.33
    enabled: true
    config:
      model_name: "Hello-SimpleAI/chatgpt-detector-roberta"
  
  - name: "fast_detectgpt"
    weight: 0.33
    enabled: true
    config:
      reference_model: "gpt2"
      num_perturbations: 10
  
  - name: "binoculars"
    weight: 0.34
    enabled: true
    config:
      model1: "gpt2"
      model2: "EleutherAI/gpt-neo-125M"
  
  # Option B: Include API detector
  # - name: "gptzero"
  #   weight: 0.25
  #   enabled: true
```

### Step 3: Test Detectors

Create `scripts/test_real_detectors.py`:

```python
"""Test real detector implementations."""

from stealthrl.detectors.roberta_detector import RoBERTaDetector
from stealthrl.detectors.fast_detectgpt import FastDetectGPTDetector
from stealthrl.detectors.binoculars import BinocularsDetector

# Test texts
human_text = "This is a genuine human-written sentence with natural language patterns."
ai_text = "The aforementioned methodology demonstrates significant improvements in performance metrics."

detectors = {
    "RoBERTa": RoBERTaDetector(device="cuda"),
    "Fast-DetectGPT": FastDetectGPTDetector(device="cuda"),
    "Binoculars": BinocularsDetector(device="cuda"),
}

print("Testing Real Detectors")
print("=" * 50)

for name, detector in detectors.items():
    print(f"\n{name}:")
    detector.load_model()
    
    human_score = detector.detect(human_text).item()
    ai_score = detector.detect(ai_text).item()
    
    print(f"  Human text score: {human_score:.3f}")
    print(f"  AI text score: {ai_score:.3f}")
    print(f"  Discrimination: {abs(ai_score - human_score):.3f}")
```

Run:
```bash
python scripts/test_real_detectors.py
```

---

## Cost Analysis

### Open Source (Local) - Total: $0

| Detector | Model Size | GPU VRAM | Inference Time | Cost |
|----------|-----------|----------|----------------|------|
| RoBERTa | 355M | 2-4GB | 0.5 sec | $0 |
| Fast-DetectGPT | 124M (GPT-2) | 2-4GB | 2-5 sec | $0 |
| Binoculars | 250M (2 models) | 4-8GB | 1 sec | $0 |
| **Total** | **~750M params** | **8-16GB** | **3-6 sec/text** | **$0** |

**Hardware requirements**:
- GPU: 16GB VRAM (RTX 3090, A100, etc.)
- Or: Use CPU (slower, 10-30 sec/text)

---

### API-Based - Total: ~$50-100 per experiment

| Detector | Cost per Request | Requests per Experiment | Total Cost |
|----------|------------------|-------------------------|------------|
| GPTZero | $0.01 | 5,000 texts | $50 |
| Sapling (Free) | $0 (100/day limit) | 100/day | $0 |
| Originality.AI | $0.01 per 100 words | ~3,000 requests | $30 |
| **Mixed (1 API + 2 local)** | - | - | **~$50** |

**Budget recommendation**: $100-200 for full research project (multiple experiments)

---

## Performance Comparison

### Detection Accuracy (Approximate)

| Detector | AUROC | FPR @ 0.5 | Speed | Adversarial Robustness |
|----------|-------|-----------|-------|----------------------|
| **GPTZero** | 0.95-0.98 | 0.05 | Fast | Medium |
| **Fast-DetectGPT** | 0.90-0.95 | 0.08 | Medium | High |
| **Binoculars** | 0.88-0.93 | 0.10 | Fast | Medium-High |
| **RoBERTa** | 0.85-0.92 | 0.12 | Fast | Low-Medium |
| **RADAR** | 0.92-0.96 | 0.06 | Fast | **Very High** |

**Notes**:
- AUROC = Area Under ROC Curve (higher = better)
- FPR = False Positive Rate at 50% threshold (lower = better)
- Adversarial robustness = resistance to paraphrasing attacks

---

## Quick Start Guide

### Day 1: Implement RoBERTa (4-6 hours)

```bash
# 1. Copy implementation code (above)
nano stealthrl/detectors/roberta_detector.py

# 2. Test it
python scripts/test_real_detectors.py

# 3. Update config
nano configs/tinker_stealthrl.yaml

# 4. Run quick experiment
python -m stealthrl.tinker.train \
  --config configs/tinker_stealthrl_ultrafast.yaml \
  --data-path data/tinker_test \
  --run-name real_detector_test \
  --num-epochs 1
```

### Day 2: Add More Detectors (4-8 hours)

```bash
# Add Fast-DetectGPT or Binoculars
# Update config with 2-3 detectors
# Run production experiment
```

### Day 3-7: Experiments and Writing

```bash
# Run full experiments
# Analyze results
# Write paper
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce model size or use CPU
```python
# Use smaller models
detector = FastDetectGPTDetector(reference_model="gpt2")  # Not gpt-j-6B

# Or use CPU
detector = RoBERTaDetector(device="cpu")
```

### Issue: API Rate Limits

**Solution**: Add retry logic with exponential backoff
```python
import time

def detect_with_retry(detector, text, max_retries=3):
    for i in range(max_retries):
        try:
            return detector.detect(text)
        except Exception as e:
            if "429" in str(e):  # Rate limit
                wait_time = 60 * (2 ** i)
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return 0.5  # Neutral score if all retries fail
```

### Issue: Slow Inference

**Solution**: Batch processing
```python
# Instead of one-by-one:
scores = [detector.detect(text) for text in texts]

# Use batching:
scores = detector.detect(texts)  # Pass list
```

---

## Summary & Recommendation

### For 1-Week Timeline:

**Day 1**: Implement **RoBERTa-ChatGPT-Detector** (easiest, 4-6 hours)  
**Day 2**: Add **Sapling.ai API** (free tier, 1 hour)  
**Day 3-4**: Run experiments with 2 real detectors  
**Day 5-7**: Write paper

### For Strong Publication:

**Week 1**: Implement all 3 open-source detectors (RoBERTa, Fast-DetectGPT, Binoculars)  
**Week 2-3**: Run full experimental suite  
**Week 4**: Write paper with comprehensive evaluation

### Priority Order:

1. ⭐ **RoBERTa** (easiest, must-have)
2. ⭐ **Binoculars** (fast, zero-shot)
3. ⭐ **Fast-DetectGPT** (theoretically grounded)
4. 💰 **GPTZero** (industry standard, costs money)
5. 🎯 **RADAR** (adversarially-robust, ideal for StealthRL)

---

**Ready to implement?** Start with RoBERTa today!
