# TASK 2: Dataset Curation

**Status**: ✅ **SETUP COMPLETE** - Ready for execution

---

## 📁 Task 2 Organization

All TASK 2 materials are organized in the `task2_dataset_curation/` folder, following the same structure as TASK 1.

```
task2_dataset_curation/
├── README.md                          # Full task overview
├── QUICK_START.md                     # ⭐ START HERE - Step-by-step guide
├── scripts/
│   ├── convert_chatgpt_bias_data.py   # Conversion script (ready)
│   └── validate_datasets.py           # Validation script (ready)
└── logs/
    ├── conversion.log                 # Will be created
    └── validation.log                 # Will be created
```

---

## 🎯 What is TASK 2?

**Objective**: Curate ESL (English as Second Language) and native English writing samples for fair AI text detection training.

**Target**:
- 1000-2000 total samples
- 40% ESL (400-800 samples from TOEFL essays)
- 60% Native (600-1200 samples from academic writing)

---

## 🚀 How to Execute TASK 2

### Quick Start (Recommended)

```bash
cd /Users/nishchaymahor/Documents/Study/291\ -\ Safety\ in\ Gen\ AI/StealthRL/StealthRL

# Read the quick start guide
cat task2_dataset_curation/QUICK_START.md

# Or open it in your editor
code task2_dataset_curation/QUICK_START.md
```

### Summary of Steps

1. **Download** ChatGPT-Detector-Bias dataset
2. **Explore** the data structure
3. **Convert** to required JSONL format using our script
4. **Validate** the converted data
5. **Generate** training and evaluation splits
6. **Verify** all outputs

**Estimated time**: 1-2 hours (mostly waiting for downloads)

---

## 📊 Expected Outputs

After completing TASK 2, you will have:

```
data/
├── esl/
│   └── toefl11.jsonl                # ESL data (~600 records)
├── native/
│   └── native_academic.jsonl        # Native data (~900 records)
├── processed/
│   ├── esl_native_dev.jsonl         # Dev split (200 samples)
│   └── esl_native_test.jsonl        # Test split (500 samples)
└── tinker/
    ├── train.jsonl                   # Training data (1200+ samples)
    └── test.jsonl                    # Testing data (300+ samples)
```

---

## ✅ Tools Provided

### 1. Conversion Script ✅

**Purpose**: Extract and convert ChatGPT-Detector-Bias data to required format

**Usage**:
```bash
python task2_dataset_curation/scripts/convert_chatgpt_bias_data.py \
    --input data/raw/ChatGPT-Detector-Bias \
    --output-esl data/esl/toefl11.jsonl \
    --output-native data/native/native_academic.jsonl
```

**Features**:
- Automatic file discovery (finds TOEFL/ESL/native files)
- Handles multiple JSON/JSONL formats
- Validates minimum sample requirements
- Reports ESL/native ratio

### 2. Validation Script ✅

**Purpose**: Verify data quality and format correctness

**Usage**:
```bash
python task2_dataset_curation/scripts/validate_datasets.py \
    --esl-data data/esl/toefl11.jsonl \
    --native-data data/native/native_academic.jsonl \
    --output task2_dataset_curation/logs/validation.log
```

**Checks**:
- ✓ Valid JSONL format
- ✓ Required fields present (id, text, source, is_esl)
- ✓ Field type validation
- ✓ ESL/native ratio (40/60 target)
- ✓ Text length statistics
- ✓ Duplicate detection

---

## 🔗 Integration with StealthRL

The prepared datasets integrate seamlessly with the existing StealthRL pipeline:

### For ESL Fairness Evaluation

```bash
# Uses: data/processed/esl_native_test.jsonl
python scripts/run_esl_eval.py \
    --eval_data data/processed/esl_native_test.jsonl \
    --stealthrl_model outputs/stealthrl_policy \
    --output_dir results/esl_eval
```

### For RL Training

```bash
# Uses: data/tinker/train.jsonl
python -m stealthrl.tinker.train \
    --config configs/tinker_stealthrl.yaml \
    --data-path data/tinker \
    --run-name stealthrl_v1
```

---

## 📚 Documentation

- **Quick Start**: `task2_dataset_curation/QUICK_START.md` ⭐
- **Full README**: `task2_dataset_curation/README.md`
- **ESL Guide**: `knowledge_base/ESL_FAIRNESS_GUIDE.md`
- **Team Handoff**: `knowledge_base/TEAM_HANDOFF.md`

---

## 🎓 Next Steps After TASK 2

Once TASK 2 is complete:

1. ✅ **Verify data**: All validation checks pass
2. ➡️ **TASK 3**: Main RL Training with real data
3. ➡️ **TASK 4**: ESL Fairness Evaluation
4. ➡️ **TASK 5**: Results Analysis & Paper Writing

---

## 💡 Key Differences from TASK 1

**TASK 1** (Detector Implementation):
- Focused on implementing AI detectors
- All code provided, just needed testing
- Output: Working detector implementations

**TASK 2** (Dataset Curation):
- Focused on data preparation
- Scripts provided, but data discovery is manual
- Output: Training-ready datasets

---

## 🚦 Current Status

| Component | Status | Location |
|-----------|--------|----------|
| Task folder | ✅ Created | `task2_dataset_curation/` |
| Conversion script | ✅ Ready | `task2_dataset_curation/scripts/convert_chatgpt_bias_data.py` |
| Validation script | ✅ Ready | `task2_dataset_curation/scripts/validate_datasets.py` |
| Quick start guide | ✅ Ready | `task2_dataset_curation/QUICK_START.md` |
| Data download | ⏸️ Pending | Run `scripts/download_datasets.sh` |
| Data conversion | ⏸️ Pending | Follow QUICK_START.md |
| Data validation | ⏸️ Pending | Follow QUICK_START.md |

---

## 📞 Questions?

- Check `task2_dataset_curation/QUICK_START.md` for step-by-step instructions
- Refer to troubleshooting section for common issues
- Review logs in `task2_dataset_curation/logs/` for detailed output

---

**Ready to begin? Start with: `task2_dataset_curation/QUICK_START.md`**
