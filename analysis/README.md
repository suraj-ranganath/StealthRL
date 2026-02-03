# Analysis

This directory contains exploratory scripts for understanding datasets and detector behavior. These scripts are optional and are not required for core training or evaluation.

**Scripts**
- Dataset analysis: `analyze_dataset_size.py`, `analyze_mage_domains.py`, `check_mage_labels.py`, `inspect_mage.py`
- Detector analysis: `analyze_detector_fairness.py`, `eval_mage_detector.py`, `eval_tinker_detector.py`, `quick_eval_detector.py`
- Utilities: `load_ds.py`

**Run an Analysis Script**
```bash
python analysis/analyze_mage_domains.py
```

**Notes**
- Many scripts are exploratory and may be slow on full datasets.
- Use optional parameters inside each script to limit the dataset size for quick checks.
