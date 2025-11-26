# Data Directory

This directory contains datasets for StealthRL training and evaluation.

## Directory Structure

```
data/
├── raw/                    # Raw datasets downloaded from sources
│   ├── detectrl/
│   ├── ai-detection-paraphrases/
│   ├── chatgpt-detector-bias/
│   └── ghostbuster/
├── processed/              # Processed datasets ready for training
│   ├── train.jsonl
│   ├── eval.jsonl
│   ├── test.jsonl
│   ├── esl_validation.jsonl
│   └── native_validation.jsonl
└── README.md
```

## Downloading Datasets

Run the download script from the root directory:

```bash
bash scripts/download_datasets.sh
```

## Dataset Sources

All datasets are obtained from their original sources with proper attribution:

- **DetectRL**: Real-world detection benchmark
- **ai-detection-paraphrases**: Paraphrase evasion benchmark (DIPPER)
- **ChatGPT-Detector-Bias**: ESL vs native bias analysis
- **Ghostbuster**: Human vs AI text pairs
- **Human Detectors**: Human judgment alignment

## Data Format

Processed datasets are in JSONL format with the following structure:

```json
{
  "text": "Example text content...",
  "label": 0,  // 0 = human, 1 = AI-generated
  "metadata": {
    "source": "dataset_name",
    "language_background": "native" // or "esl"
  }
}
```

## License and Attribution

Please respect the original licenses of each dataset. See individual dataset
directories for specific license information and citation requirements.
