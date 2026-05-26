"""Quick test of RoBERTa with MAGE after label fix."""
from eval.detectors import RoBERTaOpenAIDetector
import numpy as np
from datasets import load_dataset

print('Loading RoBERTa detector...')
det = RoBERTaOpenAIDetector(device='cpu')
det.load()

print('Loading MAGE test (will use cache)...')
ds = load_dataset('yaful/MAGE', split='test', trust_remote_code=True)
print(f'  Total samples: {len(ds)}')

# Get scores for a few samples
print()
print('Scoring samples...')
human_scores = []
ai_scores = []

# MAGE labels: 1=human, 0=machine/AI
import random
random.seed(42)
indices = random.sample(range(len(ds)), min(100, len(ds)))

for i, idx in enumerate(indices):
    item = ds[idx]
    score = det.get_scores(item['text'])
    if item['label'] == 1:  # Human
        human_scores.append(score)
    else:  # AI (label=0)
        ai_scores.append(score)
    if i % 20 == 0:
        print(f'  Processed {i} samples (human={len(human_scores)}, ai={len(ai_scores)})...')

print()
print('Results:')
print(f'  Human scores: mean={np.mean(human_scores):.4f} ± {np.std(human_scores):.4f}')
print(f'  AI scores:    mean={np.mean(ai_scores):.4f} ± {np.std(ai_scores):.4f}')
print(f'  Gap:          {np.mean(ai_scores) - np.mean(human_scores):.4f}')
print()

if np.mean(ai_scores) > np.mean(human_scores):
    print('✓ CORRECT: AI texts have HIGHER scores than human texts')
else:
    print('✗ WRONG: Human texts have higher scores - labels may be inverted!')
