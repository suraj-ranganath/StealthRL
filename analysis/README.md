# Analysis Directory

This directory contains data analysis and exploration scripts for understanding datasets and detector behavior.

## Analysis Scripts

### Dataset Analysis
- **analyze_dataset_size.py** - Analyze dataset statistics (line counts, sample distributions)
- **analyze_mage_domains.py** - Analyze MAGE dataset domains (14 human + 200+ AI sources)
- **check_mage_labels.py** - Validate MAGE label distribution (human vs AI)
- **inspect_mage.py** - Detailed inspection of MAGE dataset structure

### Detector Analysis
- **analyze_detector_fairness.py** - Analyze detector fairness metrics (ESL/non-ESL performance)
- **eval_mage_detector.py** - Evaluate detectors on MAGE dataset
- **eval_tinker_detector.py** - Evaluate detectors on Tinker dataset
- **quick_eval_detector.py** - Quick evaluation script for baseline checks

### Utilities
- **load_ds.py** - Dataset loading utilities

## Running Analysis Scripts

### Quick dataset overview:
```bash
cd /Users/atharvramesh/Projects/StealthRL
source stealthrl/bin/activate
python analysis/analyze_mage_domains.py
```

### Detector fairness analysis:
```bash
python analysis/analyze_detector_fairness.py
```

### Evaluate detectors:
```bash
python analysis/eval_mage_detector.py
```

## Key Findings

### MAGE Dataset
- **Total samples**: 60,743 (30,265 human + 30,478 AI)
- **Domains**: 14 (academic, news, creative, reviews, reasoning, etc.)
- **AI variants**: 200+ (different LLMs: GPT-3.5, GPT-4, Flan-T5, OPT, GLM130B, BLOOM, etc.)
- **ESL metadata**: ❌ NOT available

### Tinker Dataset
- **Total samples**: 20,397 (16,317 train + 4,080 test)
- **ESL metadata**: ✅ Available (`is_esl` field)
- **Domain metadata**: ✅ Available
- **Format**: Paired (human_reference + ai_text)

### Detector Performance
- **RoBERTa-large-openai**: AUROC 0.891 (very high, preferred)
- **Fast-DetectGPT (gpt-neo-2.7B)**: AUROC 0.691 (complementary detection method)
- **Ensemble (60/40)**: Balanced detection across domains

## Creating Analysis Scripts

When adding analysis scripts:
1. Place in this directory
2. Name as `analyze_<topic>.py` or `eval_<target>.py`
3. Add documentation at top of file
4. Include results/findings in comments
5. Update this README with findings

## Data Sources

| Dataset | Location | Format | Size |
|---------|----------|--------|------|
| MAGE | `data/mage/test` | HuggingFace dataset | 60K+ |
| Tinker | `data/tinker/` | JSONL files | 20K |
| Cache | `cache/` | SQLite | Variable |

## Notes

- Analysis scripts are exploratory and may be slow (loading large datasets)
- Use `max_examples` parameter to limit for quick exploration
- Results are typically printed to stdout or saved to files
- See individual script headers for specific usage details
