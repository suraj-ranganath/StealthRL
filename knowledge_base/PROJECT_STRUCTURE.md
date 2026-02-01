# Project Structure

StealthRL organized with clear separation of concerns.

```
StealthRL/
├── 📚 Core Code
│   ├── stealthrl/               # Main package
│   │   ├── tinker/              # Tinker RL training
│   │   │   ├── env.py           # RL environment (DEFENSIVE mode)
│   │   │   ├── dataset.py       # Dataset loader (MAGE + Tinker)
│   │   │   ├── reward.py        # Reward computation
│   │   │   ├── detectors.py     # Detector ensemble
│   │   │   ├── train.py         # Training loop
│   │   │   └── ...
│   │   └── ...
│   └── __init__.py
│
├── 🧪 Testing & Validation
│   ├── tests/                   # Test scripts
│   │   ├── test_detector_integration.py
│   │   ├── test_mage_loading.py
│   │   ├── test_gpt_neo_tinker.py
│   │   ├── test_fast_detectgpt_models.py
│   │   ├── test_detectors_local.py
│   │   ├── README.md
│   │   └── ...
│   └── analysis/                # Data analysis & exploration
│       ├── analyze_mage_domains.py
│       ├── analyze_detector_fairness.py
│       ├── eval_mage_detector.py
│       ├── eval_tinker_detector.py
│       ├── inspect_mage.py
│       ├── check_mage_labels.py
│       ├── analyze_dataset_size.py
│       ├── README.md
│       └── ...
│
├── 🚀 Training & Deployment
│   ├── scripts/                 # Training/deployment scripts
│   │   ├── train_stealthrl.py
│   │   ├── run_ultrafast_training.py
│   │   ├── run_research_pipeline.py
│   │   ├── evaluate_detectors.py
│   │   ├── evaluate_transfer.py
│   │   ├── monitor_training.py
│   │   ├── cancel_tinker_runs.py
│   │   └── ...
│   └── configs/                 # Configuration files
│       ├── tinker_stealthrl.yaml
│       ├── tinker_stealthrl_ultrafast.yaml
│       ├── stealthrl_small.yaml
│       └── ...
│
├── 📊 Data
│   └── data/                    # Datasets
│       ├── mage/                # MAGE dataset (60K+ samples)
│       ├── tinker/              # Tinker dataset (20K samples)
│       └── ...
│
├── 📚 Documentation
│   ├── knowledge_base/          # Comprehensive guides
│   │   ├── MAGE_DOMAINS_REFERENCE.md
│   │   ├── ESL_REMOVAL_MAGE_INTEGRATION.md
│   │   ├── DETECTOR_CLOUD_OFFLOADING.md
│   │   ├── DETECTOR_SETUP.md
│   │   ├── CHECKPOINT_GUIDE.md
│   │   ├── ESL_FAIRNESS_GUIDE.md
│   │   └── ...
│   ├── README.md               # Main readme
│   ├── atharv_readme.md        # Session notes
│   └── ...
│
├── 📝 Results
│   ├── outputs/                # Training outputs
│   │   ├── runs/               # Training run results
│   │   └── fairness/           # Fairness evaluation results
│   └── report/                 # Reports & analysis
│       ├── REPORT.md
│       └── report.tex
│
├── ⚙️ Configuration
│   ├── environment.yml         # Conda environment
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example           # Environment template
│   └── .gitignore
│
└── 🔧 Build & Metadata
    ├── cache/                  # Model/detection cache
    └── interactions_records.md # Session history
```

## Directory Purposes

### stealthrl/
- **Main implementation code**
- Organized by component (tinker, detectors, etc.)
- All production code goes here

### tests/
- **Test and validation scripts**
- One-off verification tests
- Integration tests
- Kept separate so they don't clutter main code

### analysis/
- **Data exploration and analysis**
- Dataset inspection scripts
- Detector evaluation scripts
- Fairness analysis
- Results and findings documented

### scripts/
- **End-to-end pipeline scripts**
- Training launchers
- Deployment utilities
- Monitoring scripts
- Entry points for production workflows

### configs/
- **Configuration files**
- YAML training configs
- Model hyperparameters
- Dataset paths

### knowledge_base/
- **Documentation and guides**
- How-to guides
- Research findings
- Architecture explanations
- Development notes

## Key Changes

### Before
```
├── test_*.py (mixed in root)
├── analyze_*.py (mixed in root)
├── eval_*.py (mixed in root)
├── quick_*.py (mixed in root)
├── scripts/ (training + tests)
└── ...
```

### After
```
├── tests/        (all testing)
├── analysis/     (all data analysis)
├── scripts/      (production training only)
├── stealthrl/    (core code)
└── ...
```

## Benefits

✅ **Clear separation of concerns** - easy to find code  
✅ **Cleaner imports** - `from tests.test_detector_integration import ...`  
✅ **Better organization** - no root clutter  
✅ **Easier CI/CD** - can ignore tests/ for deployment  
✅ **Professional structure** - matches industry standards  

## Navigation Guide

| I want to... | Go to... |
|-------------|----------|
| Run training | `scripts/train_stealthrl.py` |
| Analyze MAGE | `analysis/analyze_mage_domains.py` |
| Test detectors | `tests/test_detector_integration.py` |
| Understand architecture | `knowledge_base/` |
| Configure training | `configs/tinker_stealthrl.yaml` |
| Check implementation | `stealthrl/tinker/` |
| Review findings | `analysis/` or `outputs/` |

## Running Commands

```bash
# Training
python scripts/train_stealthrl.py

# Testing
python tests/test_mage_loading.py

# Analysis
python analysis/analyze_mage_domains.py

# Check results
ls outputs/runs/
```
