# Tests

This directory contains test scripts for validating core components and integrations.

**Test Scripts**
- `test_detector_integration.py` detector ensemble integration
- `test_detectors_local.py` individual detector checks
- `test_fast_detectgpt_models.py` Fast-DetectGPT variants
- `test_mage_loading.py` dataset loading
- `test_gpt_neo_tinker.py` remote backend compatibility
- `test.py` legacy test file

**Run a Single Test**
```bash
python tests/test_mage_loading.py
```

**Run a Small Set of Tests**
```bash
python tests/test_detector_integration.py
python tests/test_gpt_neo_tinker.py
```

**Adding Tests**
- Name files as `test_<component>.py`
- Keep tests self-contained and document any required assets
