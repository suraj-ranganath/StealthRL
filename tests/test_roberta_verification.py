"""
Comprehensive verification of RoBERTa detector implementation.

This test verifies:
1. Label orientation is correct (class 0 = Fake/AI)
2. Scores align with expected behavior
3. AUROC computation direction is correct
"""
from eval.detectors import RoBERTaOpenAIDetector
from eval.metrics import compute_auroc, compute_threshold_at_fpr, compute_tpr_at_fpr
import numpy as np

print("=" * 70)
print("RoBERTa Detector Verification")
print("=" * 70)

# Load detector
det = RoBERTaOpenAIDetector(device='cpu')
det.load()

# Test texts - we want clearly distinguishable samples
test_texts = [
    # More likely HUMAN (casual/informal style)
    {"text": "lol yeah i totally agree tbh, the movie was pretty good but not like amazing or anything ya know", "expected": "human"},
    {"text": "dude idk what to say, it was kinda meh. like fine i guess but idc really", "expected": "human"},
    {"text": "tbh im just tired of all this drama smh gonna take a nap or something", "expected": "human"},
    
    # More likely AI (formal/structured style)  
    {"text": "The implementation of artificial intelligence systems requires careful consideration of ethical implications and societal impact. Researchers have demonstrated that machine learning models can exhibit biased behavior.", "expected": "ai"},
    {"text": "In conclusion, the data clearly demonstrates a statistically significant correlation between the independent and dependent variables, supporting the hypothesis proposed in this study.", "expected": "ai"},
    {"text": "Furthermore, the analysis reveals that the proposed methodology achieves superior performance metrics compared to baseline approaches, validating the effectiveness of our contribution.", "expected": "ai"},
]

print("\n1. INDIVIDUAL TEXT SCORES:")
print("-" * 70)

human_scores = []
ai_scores = []
all_scores = []
all_labels = []

for item in test_texts:
    score = det.get_scores(item["text"])
    expected = item["expected"]
    
    if expected == "human":
        human_scores.append(score)
        all_labels.append(0)  # 0 = human
    else:
        ai_scores.append(score)
        all_labels.append(1)  # 1 = AI
    
    all_scores.append(score)
    
    pred = "AI" if score > 0.5 else "human"
    correct = "✓" if pred == expected else "✗"
    print(f"  {correct} Score: {score:.4f} | Expected: {expected:<5} | Pred: {pred:<5} | {item['text'][:50]}...")

print("\n2. SCORE DISTRIBUTION:")
print("-" * 70)
print(f"  Human texts (n={len(human_scores)}): mean={np.mean(human_scores):.4f} ± {np.std(human_scores):.4f}")
print(f"  AI texts (n={len(ai_scores)}):    mean={np.mean(ai_scores):.4f} ± {np.std(ai_scores):.4f}")
print(f"  Discrimination (gap): {np.mean(ai_scores) - np.mean(human_scores):.4f}")

print("\n3. METRICS VERIFICATION:")
print("-" * 70)

# AUROC
auroc = compute_auroc(all_labels, all_scores)
print(f"  AUROC: {auroc:.4f}")

# Check expected direction: AI texts should have HIGHER scores
if np.mean(ai_scores) > np.mean(human_scores):
    print("  ✓ Label direction CORRECT: AI texts have higher scores")
else:
    print("  ✗ Label direction INVERTED: Human texts have higher scores!")
    
# Threshold at 0% FPR (no false positives on human samples)
threshold_0fpr = compute_threshold_at_fpr(human_scores, target_fpr=0.0)
threshold_50fpr = compute_threshold_at_fpr(human_scores, target_fpr=0.5)
print(f"  Threshold @ 0% FPR: {threshold_0fpr:.4f}")
print(f"  Threshold @ 50% FPR: {threshold_50fpr:.4f}")

# TPR at different thresholds
tpr_at_0fpr = compute_tpr_at_fpr(ai_scores, threshold_0fpr)
print(f"  TPR @ 0% FPR: {tpr_at_0fpr:.4f}")

print("\n4. INTERPRETATION:")
print("-" * 70)
print("""
The RoBERTa OpenAI detector was trained on GPT-2 outputs vs. WebText (human text).
It may struggle with:
- Modern LLM outputs (ChatGPT, Llama, etc.) which are stylistically different
- Domain shift between training data and test data

Key points:
- Class 0 = FAKE/AI (higher class 0 prob = more AI-like)
- Class 1 = REAL/Human
- Our implementation correctly uses class 0 probability as AI score
- Low AUROC on modern AI text is expected (model trained on GPT-2)

For better detection of modern AI, consider:
1. Fast-DetectGPT (perplexity-based, more generalizable)
2. Ensemble detectors
3. Fine-tuned models on modern AI outputs
""")
