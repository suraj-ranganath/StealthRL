# Task 2: Dataset Curation

**Status**: 🚧 IN PROGRESS

---

## 📋 Overview

This folder contains all materials for **TASK 2: Dataset Curation** - preparing ESL/native corpus for StealthRL training and fairness evaluation.

---

## 🎯 Objectives

1. ✅ Download ChatGPT-Detector-Bias dataset (primary ESL source)
2. ✅ Extract TOEFL essays (ESL data) and native writing samples
3. ✅ Convert to required JSONL format
4. ✅ Generate AI text versions if needed
5. ✅ Create stratified splits (40% ESL, 60% native)
6. ✅ Prepare training data for Tinker platform

**Target**: 1000-2000 samples total (400-800 ESL, 600-1200 native)

---

## 📁 Directory Structure

```
task2_dataset_curation/
├── README.md                          # This file
├── TASK2_COMPLETION_REPORT.md         # Final completion report
├── scripts/
│   ├── convert_chatgpt_bias_data.py   # Main conversion script
│   ├── generate_ai_text.py            # AI text generation (if needed)
│   └── validate_datasets.py           # Dataset validation
├── notebooks/                          # Exploration notebooks (optional)
│   └── explore_chatgpt_bias.ipynb
└── logs/
    ├── conversion.log                 # Conversion logs
    └── validation.log                 # Validation results
```

---

## 🚀 Execution Steps

### Step 1: Download Datasets ✅

```bash
cd /Users/nishchaymahor/Documents/Study/291\ -\ Safety\ in\ Gen\ AI/StealthRL/StealthRL
bash scripts/download_datasets.sh
```

**Downloads:**
- `data/raw/ChatGPT-Detector-Bias/` - Primary ESL/native source

### Step 2: Explore Data Structure 🔍

```bash
cd data/raw/ChatGPT-Detector-Bias
ls -la
cat README.md
```

**Identify:**
- Where TOEFL essays are located
- Where native samples are located
- Current data format
- Available metadata (is_esl, proficiency, etc.)

### Step 3: Convert to Required Format 🔧

```bash
python task2_dataset_curation/scripts/convert_chatgpt_bias_data.py \
    --input data/raw/ChatGPT-Detector-Bias \
    --output-esl data/esl/toefl11.jsonl \
    --output-native data/native/native_academic.jsonl \
    --log task2_dataset_curation/logs/conversion.log
```

**Output format:**
```json
{
  "id": "toefl11_001",
  "text": "Essay text...",
  "source": "TOEFL11",
  "is_esl": true,
  "proficiency_level": "medium",
  "prompt_id": "P1"
}
```

### Step 4: Generate AI Text (If Needed) 🤖

If data only contains human text, generate AI versions:

```bash
python task2_dataset_curation/scripts/generate_ai_text.py \
    --input data/esl/toefl11.jsonl \
    --output data/esl/toefl11_with_ai.jsonl \
    --model gpt-3.5-turbo
```

### Step 5: Create Evaluation Splits ✅

```bash
python -m stealthrl.data.esl_native_corpus
```

**Creates:**
- `data/processed/esl_native_dev.jsonl` (200 samples)
- `data/processed/esl_native_test.jsonl` (500 samples)

### Step 6: Create Training Data ✅

```bash
python scripts/prepare_tinker_data.py \
    --input-paths data/esl/toefl11.jsonl data/native/native_academic.jsonl \
    --output-dir data/tinker \
    --train-split 0.8
```

**Creates:**
- `data/tinker/train.jsonl` (80% training)
- `data/tinker/test.jsonl` (20% testing)

### Step 7: Validate 🔍

```bash
python task2_dataset_curation/scripts/validate_datasets.py \
    --esl-data data/esl/toefl11.jsonl \
    --native-data data/native/native_academic.jsonl \
    --output task2_dataset_curation/logs/validation.log
```

---

## 📊 Expected Outputs

### Data Files

```
data/
├── raw/
│   └── ChatGPT-Detector-Bias/       # Downloaded raw data
├── esl/
│   └── toefl11.jsonl                # Processed ESL data
├── native/
│   └── native_academic.jsonl        # Processed native data
├── processed/
│   ├── esl_native_dev.jsonl         # Dev split (200 samples)
│   └── esl_native_test.jsonl        # Test split (500 samples)
└── tinker/
    ├── train.jsonl                   # Training data (80%)
    └── test.jsonl                    # Testing data (20%)
```

### Statistics

**Target Composition:**
- ESL: 40% (400-800 samples)
- Native: 60% (600-1200 samples)
- Total: 1000-2000 samples

**Quality Metrics:**
- Valid JSONL format: 100%
- All required fields present: 100%
- Correct ESL ratio: 40% ± 5%
- Stratified by source: Yes

---

## ✅ Success Criteria

- [x] ChatGPT-Detector-Bias downloaded successfully
- [ ] ESL data extracted and converted (TOEFL essays)
- [ ] Native data extracted and converted
- [ ] All files in correct JSONL format
- [ ] ESL/Native splits created (40/60 ratio)
- [ ] Training data prepared for Tinker
- [ ] Validation passed (all checks green)

---

## 🐛 Troubleshooting

### Issue: ChatGPT-Detector-Bias has different structure than expected
**Solution**: Examine the actual structure and adapt conversion script

### Issue: Missing AI text field
**Solution**: Generate using GPT-3.5-turbo or local LLM

### Issue: Insufficient samples
**Solution**: Extract from other downloaded datasets (DetectRL, Ghostbuster)

### Issue: Imbalanced ESL/native ratio
**Solution**: Adjust sampling in `build_esl_native_eval_split()`

---

## 📚 References

- Main README: `../README.md`
- ESL Fairness Guide: `../knowledge_base/ESL_FAIRNESS_GUIDE.md`
- Team Handoff: `../knowledge_base/TEAM_HANDOFF.md`
- Data Schema: `../stealthrl/data/esl_native_corpus.py`

---

**Last Updated**: December 1, 2025
