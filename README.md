# StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors

[Paper (arXiv)](https://arxiv.org/abs/2602.08934)  
[Model (Hugging Face)](https://huggingface.co/suraj-ranganath/StealthRL)

![StealthRL Pipeline Overview](figures/StealthRL_Pipeline_Final_Final.png)

## Abstract

AI-text detectors are increasingly used in high-stakes settings, yet their robustness to meaning-preserving adversarial rewriting remains uncertain. We introduce StealthRL, a reinforcement learning framework for stress-testing AI-text detectors with adaptive paraphrase attacks. StealthRL trains a paraphrase policy against a detector ensemble while preserving semantic content, then evaluates transfer to held-out detector families. On the full filtered MAGE test pool (15,310 human / 14,656 AI), StealthRL reduces mean AUROC from 0.79 to 0.43 and achieves a 0.024 mean TPR@1%FPR across RoBERTa, Fast-DetectGPT, Binoculars, and MAGE. The attack transfers to two detectors not used during training, exposing shared vulnerabilities rather than a single-detector failure. We further analyze detector score distributions and evaluate quality with E5, BERTScore, and LLM-based Likert ratings. Our results show that current AI-text detectors remain brittle under realistic paraphrasing pressure and provide a reproducible protocol for adversarial robustness evaluation.

## What This Repository Contains

This repository is the research and engineering codebase behind StealthRL. It contains:

- GRPO-based training code for the StealthRL paraphrase policy
- attack-method implementations for the paper methods M0-M5
- detector wrappers and evaluation utilities
- the staged full-MAGE evaluation pipeline used to produce the reported results
- plotting, metrics, and quality-judging code for paper-ready figures and tables

The repo is intended to be a useful starting point for researchers who want to:

- reproduce our detector-robustness evaluation
- swap in new detectors or attack methods
- study transfer to held-out detector families
- benchmark their own paraphrasing defenses or red-team methods

## Repository Map

- `stealthrl/`: training code, detector wrappers, rewards, data utilities, and the original StealthBench package
- `eval/`: research-grade evaluation harness used for the paper results
- `scripts/`: runnable entry points for training, evaluation, orchestration, plotting, and utilities
- `configs/`: YAML configs for training, evaluation, and ablations
- `figures/`: pipeline diagrams and static assets
- `tests/`: integration and sanity checks
- `analysis/`: ad hoc analysis helpers and one-off utilities

## Environment Setup

### Base environment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional dependencies

Depending on which parts of the project you want to run, you may also need:

```bash
pip install tinker
pip install openai
pip install vllm
```

Notes:

- `tinker` is required for the cloud-backed StealthRL checkpoint inference path used by `M2`.
- `openai` is only required for the GPT/Likert quality evaluation step.
- `vllm` is recommended for fast local generation in the paper baselines.

### Environment variables

Typical environment variables used by the repo:

```bash
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export OPENAI_API_KEY=...
export TINKER_API_KEY=...
```

For the staged paper evaluation pipeline, we used an env file plus a checkpoint descriptor JSON. The public scripts let you override both paths explicitly:

```bash
python scripts/run_full_mage_research_eval.py \
  --env-file ~/.config/stealthrl/eval.env \
  --checkpoint-json ~/.config/stealthrl/m2_checkpoint.json
```

## Quick Start

### Quick evaluation smoke test

```bash
python scripts/run_eval.py --quick
```

### Run the legacy StealthBench harness

```bash
python scripts/run_stealthbench.py --config configs/stealthbench.yaml
```

This now loads the configured text files, runs the configured detectors, saves CSV outputs, and generates comparison plots. The old TODO-only stub has been removed.

### Run a staged full MAGE evaluation

```bash
python scripts/run_full_mage_research_eval.py \
  --env-file ~/.config/stealthrl/eval.env \
  --checkpoint-json ~/.config/stealthrl/m2_checkpoint.json \
  --run-root outputs/eval_runs/full_mage_public \
  --gpus 0 1 2 3
```

### Run the demo website

The repository includes a FastAPI-backed demo website under `demo/`. It serves a polished static UI and exposes `POST /api/paraphrase` with API-key support plus a 20/day public quota for unauthenticated users.

```bash
pip install -r demo/requirements.txt
uvicorn demo.stealthrl_demo.app:app --reload --port 8080
```

By default the demo runs in zero-cost `mock` mode for UI testing. To use the real StealthRL sampler, set `STEALTHRL_DEMO_INFERENCE_BACKEND=tinker`, `STEALTHRL_DEMO_CHECKPOINT_JSON`, and `TINKER_API_KEY`. See `demo/README.md` for API-key, quota, Docker, and AWS deployment notes.

## Reproducing the Paper Results

The paper reports results on the full filtered MAGE evaluation pool:

- 15,310 human samples
- 14,656 AI samples
- 29,966 total

The research-grade evaluation pipeline is implemented in the `eval/` module plus the staged scripts under `scripts/`.

### Recommended reproduction flow

1. Prepare credentials and checkpoint metadata.

   Create an env file containing `OPENAI_API_KEY` and `TINKER_API_KEY`, and a checkpoint JSON describing the StealthRL Tinker sampler path.

2. Run preflight.

```bash
python scripts/preflight_research_eval.py \
  --env-file ~/.config/stealthrl/eval.env \
  --checkpoint-json ~/.config/stealthrl/m2_checkpoint.json
```

3. Launch the full evaluation.

```bash
python scripts/run_full_mage_research_eval.py \
  --env-file ~/.config/stealthrl/eval.env \
  --checkpoint-json ~/.config/stealthrl/m2_checkpoint.json \
  --run-root outputs/eval_runs/full_mage_repro \
  --gpus 0 1 2 3
```

4. Inspect generated method outputs, detector scores, metrics, and plots under the chosen run directory.

5. If you only need to rerun GPT-based quality judging on cached outputs:

```bash
python scripts/run_gpt_quality_only.py \
  --run-root outputs/eval_runs/full_mage_repro \
  --env-file ~/.config/stealthrl/eval.env
```

6. If you only need to add BERTScore to an assembled run with cached outputs:

```bash
python scripts/compute_bertscore_for_run.py \
  --run-dir outputs/eval_runs/full_mage_repro/assembled \
  --device cuda:0 \
  --batch-size 16 \
  --chunk-size 512
```

The BERTScore script updates `quality.parquet` and `quality.csv` in place after every chunk, so interrupted runs can be resumed without recomputing completed rows. Use `--limit-per-method` for a small smoke test and `--force` only when intentionally recomputing existing BERTScore columns.

### Main output artifacts

The staged run produces:

- `method_runs/`: per-method generated outputs
- `detector_scores/`: per-detector parquet score files
- `assembled/metrics.json`: aggregate detector metrics
- `assembled/thresholds.json`: calibrated detector thresholds
- `assembled/quality.parquet`: automatic quality metrics
- `assembled/quality_gpt.parquet`: GPT/Likert quality ratings
- `assembled/figures/`: paper-ready plots

### Canonical paper settings

The primary paper checkpoint uses `configs/tinker_mage_10k.yaml` as the canonical training configuration: Qwen3-4B-Instruct with LoRA rank 32, GRPO group size 8, 10,000 MAGE training samples, three epochs, learning rate `2.8e-4`, KL coefficient `0.05`, temperature `1.0`, top-p `0.9`, and a two-detector reward ensemble weighted `0.6` RoBERTa / `0.4` Fast-DetectGPT. Other configs in `configs/` are retained as legacy examples, smoke-test settings, or ablation templates and should not be treated as paper-authoritative unless explicitly documented.

## Training Implementation

StealthRL trains a paraphrase policy rather than a detector. The core idea is to optimize a model that rewrites AI-generated text so that it remains semantically faithful while reducing detector confidence.

### Model and optimization

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Adaptation: LoRA
- RL algorithm: GRPO
- Training style: detector-guided paraphrase policy optimization

### Reward design

The reward is multi-objective and balances:

- detector evasion against the in-training detector ensemble
- semantic preservation relative to the source text
- generation validity and stability constraints

The implementation lives primarily in:

- `stealthrl/tinker/train.py`
- `stealthrl/rewards/`
- `configs/`

### Inference-time behavior

The paper’s StealthRL attack is single-shot at test time:

- one policy generation call per sample
- no iterative refinement loop
- no target-detector queries during evaluation

Detector access is used during offline RL training and for external evaluation, not for adaptive query-time search in `M2`.

## Evaluation Implementation

The paper evaluation is implemented in the newer `eval/` stack rather than the older `stealthrl/evaluation/` harness.

### Methods

- `M0`: no attack
- `M1`: simple paraphrase baseline
- `M2`: StealthRL
- `M3`: detector-guided adversarial paraphrasing baseline
- `M4`: AuthorMist baseline
- `M5`: character-level obfuscation baseline

Relevant code:

- `eval/methods/`
- `scripts/generate_method_outputs.py`

### Detector panel

The main paper detector panel uses:

- `roberta`: `openai-community/roberta-large-openai-detector`
- `fast_detectgpt`
- `binoculars`
- `mage`: `yaful/MAGE`

The repo also retains `ghostbuster` support for legacy/compatibility experiments, but Ghostbuster is not part of the final four-detector paper panel.

Relevant code:

- `eval/detectors.py`
- `scripts/score_detector_outputs.py`
- `eval/runner.py`

### Metrics and quality analysis

The public evaluation code computes:

- AUROC
- TPR@1%FPR
- TPR@5%FPR
- ASR
- E5 similarity
- BERTScore precision/recall/F1
- perplexity
- edit rate
- GPT/Likert quality and similarity scores
- bootstrap confidence intervals

Relevant code:

- `eval/metrics.py`
- `eval/plots.py`
- `eval/quality_judge.py`
- `scripts/finalize_eval_run.py`

## Engineering Notes

### vLLM integration

The current evaluation code supports vLLM-backed local generation for high-throughput baseline evaluation. This is implemented in:

- `eval/methods/vllm_backend.py`

### Tinker integration

The StealthRL `M2` method supports Tinker-backed sampling via a checkpoint descriptor JSON. This path is implemented in:

- `eval/methods/stealthrl.py`

### Resume and staged orchestration

The full-MAGE pipeline is intentionally staged:

- preflight checks fail fast on detector/model setup problems
- per-method generation is isolated and resumable
- detector scoring is separated from generation
- final assembly and GPT judging are resumable

This makes long multi-GPU runs more robust and easier to debug.

## Extending the Repository

### Add a new attack method

1. Add a class under `eval/methods/`
2. Implement the `BaseAttackMethod` interface
3. Register the method in `eval/methods/__init__.py`
4. Add it to the staged runner if you want it included in orchestrated runs

### Add a new detector

1. Add a detector wrapper to the evaluation stack
2. Implement loading and batch scoring
3. Register it in the detector registry used by `eval/runner.py`
4. Add thresholds, plots, and table hooks as needed

### Add a new benchmark dataset

1. Add a dataset loader or adapter
2. Normalize it into the evaluation sample schema
3. Update the run scripts with the dataset name and sampling logic

## Responsible Use

This repository is released for research on AI-text detector robustness, adversarial evaluation, and defensive benchmarking. It is not intended to support cheating, plagiarism, or evasion of legitimate safety and integrity systems.

If you build on this work, please use it to improve detector robustness, calibration, transfer evaluation, and transparency around deployment limitations.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{ranganath2026stealthrl,
  title={StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors},
  author={Ranganath, Suraj and others},
  journal={arXiv preprint arXiv:2602.08934},
  year={2026}
}
```
