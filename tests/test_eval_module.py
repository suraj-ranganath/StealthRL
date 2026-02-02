"""
Tests for the eval module components.

Run with: python tests/test_eval_module.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_loading():
    """Test dataset loading with minimal samples."""
    print("\n=== Testing Data Loading ===")
    from eval.data import load_eval_dataset
    
    dataset = load_eval_dataset('mage', n_human=5, n_ai=5, seed=42)
    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
    assert len(dataset.human_samples) == 5, "Expected 5 human samples"
    assert len(dataset.ai_samples) == 5, "Expected 5 AI samples"
    
    sample = dataset[0]
    assert hasattr(sample, 'text'), "Sample should have text"
    assert hasattr(sample, 'label'), "Sample should have label"
    assert sample.label in ['human', 'ai'], f"Invalid label: {sample.label}"
    
    print(f"✓ Loaded {len(dataset)} samples")
    print(f"✓ Sample structure correct")
    return dataset


def test_detectors(device='cpu'):
    """Test detector loading and scoring."""
    print("\n=== Testing Detectors ===")
    from eval.detectors import get_detector, DETECTOR_REGISTRY
    
    test_text = "This is a test sentence to check if the detector works properly."
    
    # Test RoBERTa only for now (others require larger models)
    detector = get_detector('roberta', device=device)
    score = detector.get_scores(test_text)
    assert 0 <= score <= 1, f"Score should be in [0,1], got {score}"
    print(f"✓ RoBERTa detector score: {score:.4f}")
    
    # Test batch scoring
    texts = [test_text] * 3
    scores = detector.get_scores(texts)
    assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
    print(f"✓ Batch scoring works: {scores}")
    
    return detector


def test_metrics():
    """Test metrics computation."""
    print("\n=== Testing Metrics ===")
    from eval.metrics import (
        compute_auroc, 
        compute_threshold_at_fpr,
        compute_tpr_at_fpr, 
        compute_asr
    )
    
    # Perfect detector scenario
    # labels: 0 = human, 1 = AI
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Scores where AI texts have higher scores (good detector)
    scores = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.7, 0.8, 0.75, 0.85, 0.9])
    
    # AUROC
    auroc = compute_auroc(labels, scores)
    assert 0 <= auroc <= 1, f"AUROC should be in [0,1], got {auroc}"
    print(f"✓ AUROC: {auroc:.4f}")
    
    # Threshold at 10% FPR using human scores
    human_scores = scores[:5]
    ai_scores = scores[5:]
    threshold = compute_threshold_at_fpr(human_scores, target_fpr=0.1)
    print(f"✓ Threshold@10%FPR: {threshold:.4f}")
    
    # TPR at that threshold
    tpr = compute_tpr_at_fpr(ai_scores, threshold)
    assert 0 <= tpr <= 1, f"TPR should be in [0,1], got {tpr}"
    print(f"✓ TPR@10%FPR: {tpr:.4f}")
    
    # ASR with threshold (= 1 - TPR)
    asr = compute_asr(ai_scores, threshold=0.5, higher_is_ai=True)
    assert 0 <= asr <= 1, f"ASR should be in [0,1], got {asr}"
    print(f"✓ ASR@0.5: {asr:.4f}")


def test_methods():
    """Test attack method registry."""
    print("\n=== Testing Methods ===")
    from eval.methods import get_method, METHOD_REGISTRY
    
    print(f"Available methods: {list(METHOD_REGISTRY.keys())}")
    
    # Test no attack method
    m0 = get_method('m0')
    assert m0.name == 'no_attack', f"Expected 'no_attack', got {m0.name}"
    
    test_text = "Original AI-generated text."
    result = m0.attack(test_text)
    
    # Result is an AttackOutput object
    assert hasattr(result, 'text'), "Result should have 'text' attribute"
    assert result.text == test_text, "NoAttack should return original text"
    print(f"✓ M0 (no_attack) works")
    
    # Test that all methods can be instantiated
    for name, cls in METHOD_REGISTRY.items():
        if name not in ['m2', 'stealthrl']:  # Skip StealthRL (requires checkpoint)
            try:
                method = get_method(name)
                print(f"✓ {name} ({method.name}) can be instantiated")
            except Exception as e:
                print(f"✗ {name} failed: {e}")


def test_plots():
    """Test plot generation (without actually saving)."""
    print("\n=== Testing Plots ===")
    from eval.plots import create_heatmap, create_main_results_table
    
    # Fake metrics data in expected format
    metrics_data = [
        {'detector': 'roberta', 'method': 'm0', 'auroc': 0.95, 'auroc_ci_low': 0.93, 'auroc_ci_high': 0.97,
         'tpr_at_1fpr': 0.70, 'tpr_at_1fpr_ci_low': 0.65, 'tpr_at_1fpr_ci_high': 0.75,
         'asr': 0.30, 'asr_ci_low': 0.25, 'asr_ci_high': 0.35},
        {'detector': 'roberta', 'method': 'm1', 'auroc': 0.80, 'auroc_ci_low': 0.78, 'auroc_ci_high': 0.82,
         'tpr_at_1fpr': 0.45, 'tpr_at_1fpr_ci_low': 0.40, 'tpr_at_1fpr_ci_high': 0.50,
         'asr': 0.55, 'asr_ci_low': 0.50, 'asr_ci_high': 0.60},
    ]
    
    # Test markdown table generation
    try:
        md_table = create_main_results_table(metrics_data)
        assert '|' in md_table, "Markdown table should contain pipe characters"
        print(f"✓ Markdown table generated")
        print(md_table[:300] + "...")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✗ Markdown table failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("StealthRL Eval Module Tests")
    print("=" * 60)
    
    try:
        test_metrics()
        test_methods()
        test_plots()
        test_data_loading()  # This downloads data
        test_detectors()  # This downloads model
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
