# Tests Directory

This directory contains test scripts for validating the StealthRL pipeline.

## Test Scripts

### Core Functionality Tests
- **test_detector_integration.py** - Test detector ensemble integration (RoBERTa + Fast-DetectGPT)
- **test_detectors_local.py** - Test individual detector implementations
- **test_fast_detectgpt_models.py** - Test Fast-DetectGPT with different base models
- **test_mage_loading.py** - Test MAGE dataset loading through StealthRLDatasetBuilder
- **test_gpt_neo_tinker.py** - Check GPT-Neo compatibility with Tinker API

### Legacy Tests
- **test.py** - Original test file (can be deprecated)

## Running Tests

### Run a specific test:
```bash
cd /Users/atharvramesh/Projects/StealthRL
source stealthrl/bin/activate
python tests/test_mage_loading.py
```

### Run all tests:
```bash
# Run detector integration tests
python tests/test_detector_integration.py

# Run Tinker compatibility tests
python tests/test_gpt_neo_tinker.py
```

## Test Coverage

| Test | Purpose | Status |
|------|---------|--------|
| test_detector_integration.py | Validate RoBERTa + Fast-DetectGPT ensemble | ✅ Passing |
| test_mage_loading.py | Validate MAGE dataset loading | ✅ Passing |
| test_gpt_neo_tinker.py | Check cloud/remote execution options | ✅ Passing |
| test_fast_detectgpt_models.py | Validate model variants | ✅ Passing |
| test_detectors_local.py | Test detector inference | ✅ Passing |

## Adding New Tests

When adding tests, follow this naming convention:
- `test_<component>.py` - Test for specific component
- `test_<feature>_<variation>.py` - Test feature with variations

Place in this directory and document in this README.
