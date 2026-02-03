# StealthRL: Reinforcement Learning Paraphrase Attacks for\\
Multi-Detector Evasion of AI-Text Detectors

\begin{abstract}
We introduce \emph{StealthRL}, a reinforcement-learning framework that paraphrases AI-generated text to evade detection while preserving meaning and fluency, enabling systematic red-teaming of AI-text detectors. The challenge is to reduce detector confidence at strict low-false-positive operating points without collapsing semantic fidelity or overfitting to a single detector family. StealthRL fine-tunes a Qwen-family paraphraser with LoRA and a multi-objective reward combining detector evasion, semantic similarity, and fluency. On the MAGE benchmark with multiple detectors, StealthRL reduces detection at stringent operating points while maintaining high semantic similarity, outperforming simple paraphrasing and a detector-guided baseline. We release an anonymized code package in the supplementary material with a placeholder anonymous link for reproducibility.
\end{abstract}

**Implementation Overview**
The rest of the implementation is organized as a modular, configuration-driven research codebase:
- `stealthrl/`: Core library (training, rewards, detectors, evaluation, data utilities)
- `eval/`: Standalone evaluation harness and reporting utilities
- `configs/`: YAML configurations for training, evaluation, and ablations
- `scripts/`: Entry points for training, evaluation, and visualization
- `analysis/`: Exploratory analysis helpers
- `tests/`: Sanity checks and integration tests
- `models/`: Optional baseline model adapters
- `checkpoints/`: Pointers to trained checkpoints (not included)

**Setup**
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- Optional environment variables:
  - `HF_HOME` / `TRANSFORMERS_CACHE` for model cache location
  - `OPENAI_API_KEY` only if enabling GPT-based quality evaluation

**Quick Start**
- Train a small StealthRL run:
  ```bash
  python scripts/train_stealthrl.py --config configs/stealthrl_small.yaml
  ```
- Run evaluation:
  ```bash
  python scripts/run_stealthbench.py --config configs/stealthbench.yaml
  ```

**Reproducibility**
- The anonymized release package uses a placeholder link included in the supplementary material.
- All experiments are driven by YAML configs under `configs/` and logged to `outputs/`.

**Responsible Use**
This repository is intended for research and evaluation of AI-text detectors. It is not intended to facilitate academic dishonesty or evasion of legitimate safeguards.

**License**
MIT License. See `LICENSE`.
