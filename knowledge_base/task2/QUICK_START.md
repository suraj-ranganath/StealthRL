# TASK 2: Dataset Curation - Quick Start Guide

**Complete execution guide for TASK 2 - organized in task2_dataset_curation/ folder**

---

## ✅ Progress Tracker

- [ ] Step 1: Download datasets
- [ ] Step 2: Explore ChatGPT-Detector-Bias
- [ ] Step 3: Run conversion script
- [ ] Step 4: Validate converted data
- [ ] Step 5: Generate training splits
- [ ] Step 6: Final validation

---

## 🚀 Step-by-Step Execution

### Step 1: Download ChatGPT-Detector-Bias Dataset

```bash
cd /Users/nishchaymahor/Documents/Study/291\ -\ Safety\ in\ Gen\ AI/StealthRL/StealthRL

# Option A: Use the full download script (downloads all datasets)
bash scripts/download_datasets.sh

# Option B: Download just ChatGPT-Detector-Bias (faster)
cd data/raw
git clone https://github.com/Weixin-Liang/ChatGPT-Detector-Bias.git
cd ../..
```

**Expected outcome:**
- `data/raw/ChatGPT-Detector-Bias/` directory created
- Contains TOEFL and native writing samples

---

### Step 2: Explore the Downloaded Data

```bash
# Navigate to the dataset
cd data/raw/ChatGPT-Detector-Bias

# List contents
ls -la

# Check README for structure
cat README.md

# Find data files
find . -name "*.json" -o -name "*.jsonl" -o -name "*.csv"

# Look for TOEFL/ESL data
find . -iname "*toefl*" -o -iname "*esl*"

# Look for native data
find . -iname "*native*"
```

**What to look for:**
- Files containing "toefl", "esl", or "non-native"
- Files containing "native" or "human"
- Data format (JSON, JSONL, CSV)
- Field names (text, essay, is_esl, etc.)

---

### Step 3: Run the Conversion Script

```bash
cd /Users/nishchaymahor/Documents/Study/291\ -\ Safety\ in\ Gen\ AI/StealthRL/StealthRL

# Run the conversion script
python scripts/convert_chatgpt_bias_data.py \
    --input data/raw/ChatGPT-Detector-Bias \
    --output-esl data/esl/toefl11.jsonl \
    --output-native data/native/native_academic.jsonl \
    --min-samples 100

# Check the output
ls -lh data/esl/
ls -lh data/native/
```

**Expected output:**
```
CONVERSION SUMMARY
==============================================================
ESL records:    400-800
Native records: 600-1200
Total records:  1000-2000

ESL ratio:      40.0%
Native ratio:   60.0%
==============================================================
```

**If conversion fails:**
- The script will tell you what files it found
- You may need to manually inspect the data structure
- Check `logs/conversion.log` for details

---

### Step 4: Validate the Converted Data

```bash
# Run validation script
python scripts/validate_datasets.py \
    --esl-data data/esl/toefl11.jsonl \
    --native-data data/native/native_academic.jsonl \
    --output logs/validation.log

# Check validation results
cat logs/validation.log
```

**Expected output:**
```
VALIDATION SUMMARY
==============================================================
✅ All validation checks passed!

STATISTICS:
  total_records: 1500
  esl_records: 600
  native_records: 900
  esl_ratio: 0.40
  ESL_min_length: 25
  ESL_max_length: 450
  ESL_mean_length: 180.5
==============================================================
```

**If validation fails:**
- Fix the errors reported
- Re-run conversion if needed
- Check JSONL format with: `head -1 data/esl/toefl11.jsonl | jq`

---

### Step 5: Generate Training and Evaluation Splits

#### 5a. Create ESL/Native Evaluation Splits

```bash
# Run the built-in corpus builder
python -m stealthrl.data.esl_native_corpus
```

**Expected output:**
```
Loaded 600 ESL records from toefl11.jsonl
Loaded 900 native records from native_academic.jsonl
Total: 600 ESL, 900 native records

Dev split: 200 total
  ESL: 80 (40.0%)
  Native: 120 (60.0%)

Test split: 500 total
  ESL: 200 (40.0%)
  Native: 300 (60.0%)

Saved splits:
  Dev: data/processed/esl_native_dev.jsonl
  Test: data/processed/esl_native_test.jsonl
```

**Files created:**
- `data/processed/esl_native_dev.jsonl` (200 samples for development)
- `data/processed/esl_native_test.jsonl` (500 samples for testing)

#### 5b. Create Tinker Training Data

```bash
# Prepare Tinker-format training data
python scripts/prepare_tinker_data.py \
    --input-paths data/esl/toefl11.jsonl data/native/native_academic.jsonl \
    --output-dir data/tinker \
    --train-split 0.8
```

**Expected output:**
```
Loaded 600 examples from data/esl/toefl11.jsonl
Loaded 900 examples from data/native/native_academic.jsonl
Saved 1200 train examples
Saved 300 test examples

Train Statistics:
  Total examples: 1200
  Domain distribution:
    academic: 1200 (100.0%)
  ESL examples: 480 (40.0%)
  Non-ESL examples: 720 (60.0%)
```

**Files created:**
- `data/tinker/train.jsonl` (1200 samples for RL training)
- `data/tinker/test.jsonl` (300 samples for testing)

---

### Step 6: Final Validation

```bash
# Verify all splits exist
ls -lh data/processed/
ls -lh data/tinker/

# Check ESL/Native splits
python -c "
from pathlib import Path
from stealthrl.data.esl_native_corpus import load_esl_native_jsonl

# Check dev split
dev_records = load_esl_native_jsonl(Path('data/processed/esl_native_dev.jsonl'))
esl_dev = sum(1 for r in dev_records if r.is_esl)
print(f'Dev: {esl_dev}/{len(dev_records)} ESL = {esl_dev/len(dev_records):.1%}')

# Check test split
test_records = load_esl_native_jsonl(Path('data/processed/esl_native_test.jsonl'))
esl_test = sum(1 for r in test_records if r.is_esl)
print(f'Test: {esl_test}/{len(test_records)} ESL = {esl_test/len(test_records):.1%}')
"

# Inspect sample records
head -1 data/esl/toefl11.jsonl | jq
head -1 data/native/native_academic.jsonl | jq
head -1 data/tinker/train.jsonl | jq
```

---

## 📊 Expected Final Structure

```
data/
├── raw/
│   └── ChatGPT-Detector-Bias/       # Downloaded (Step 1)
├── esl/
│   └── toefl11.jsonl                # Created (Step 3) - ~600 records
├── native/
│   └── native_academic.jsonl        # Created (Step 3) - ~900 records
├── processed/
│   ├── esl_native_dev.jsonl         # Created (Step 5a) - 200 records
│   └── esl_native_test.jsonl        # Created (Step 5a) - 500 records
└── tinker/
    ├── train.jsonl                   # Created (Step 5b) - 1200 records
    └── test.jsonl                    # Created (Step 5b) - 300 records
```

---

## ✅ Success Criteria

Your TASK 2 is complete when:

- [x] `data/esl/toefl11.jsonl` exists with ~600 ESL records
- [x] `data/native/native_academic.jsonl` exists with ~900 native records
- [x] All JSONL files have valid format (checked by validation script)
- [x] ESL ratio is 40% ± 10%
- [x] `data/processed/esl_native_dev.jsonl` exists (200 samples)
- [x] `data/processed/esl_native_test.jsonl` exists (500 samples)
- [x] `data/tinker/train.jsonl` exists (1200+ samples)
- [x] `data/tinker/test.jsonl` exists (300+ samples)
- [x] All validation checks pass

---

## 🐛 Common Issues & Solutions

### Issue 1: ChatGPT-Detector-Bias has unexpected structure

**Solution:**
```bash
# Manually explore and find the actual data files
cd data/raw/ChatGPT-Detector-Bias
find . -type f | grep -E '\.(json|jsonl|csv)$'

# Check a sample file
head -1 path/to/data/file.json | jq

# Update conversion script if needed or manually extract
```

### Issue 2: Not enough samples extracted

**Possible causes:**
- Data files in different location than expected
- Different field names in JSON
- Text filtering too strict (min length)

**Solution:**
```bash
# Lower minimum sample threshold
python task2_dataset_curation/scripts/convert_chatgpt_bias_data.py \
    --input data/raw/ChatGPT-Detector-Bias \
    --output-esl data/esl/toefl11.jsonl \
    --output-native data/native/native_academic.jsonl \
    --min-samples 50  # Lower threshold
```

### Issue 3: Imbalanced ESL/native ratio

**Solution:**
The conversion script tries to maintain 40/60 ratio. If imbalanced:
- Extract more from the underrepresented category
- Use other downloaded datasets (DetectRL, Ghostbuster) as supplements
- Adjust `esl_ratio` in `build_esl_native_eval_split()`

### Issue 4: Missing AI text field

If data only has human text, you need to generate AI paraphrases:
- Use GPT-3.5-turbo API (requires OpenAI key)
- Use local LLM (Llama, Mistral)
- Use existing AI-generated samples from the dataset

---

## 📞 Next Steps After TASK 2

Once TASK 2 is complete, you can proceed to:

**TASK 3: Main RL Training**
```bash
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --run-name stealthrl_full_ensemble
```

---

## 📚 Additional Resources

- **Main README**: `../README.md`
- **ESL Guide**: `../knowledge_base/ESL_FAIRNESS_GUIDE.md`
- **Team Handoff**: `../knowledge_base/TEAM_HANDOFF.md`
- **Data Schema**: `../stealthrl/data/esl_native_corpus.py`

---

**Good luck! 🚀**
