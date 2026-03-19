# StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors

[Paper (arXiv)](https://arxiv.org/abs/2602.08934)  
[Model (Hugging Face)](https://huggingface.co/suraj-ranganath/StealthRL)

![StealthRL Pipeline Overview](figures/StealthRL_Pipeline_Final_Final.png)

## Abstract

AI-text detectors face a critical robustness challenge: adversarial paraphrasing attacks that preserve semantics while evading detection. We introduce StealthRL, a reinforcement learning framework that stress-tests detector robustness under realistic adversarial conditions. StealthRL trains a paraphrase policy against a multi-detector ensemble using Group Relative Policy Optimization (GRPO) with LoRA adapters on Qwen3-4B, optimizing a composite reward that balances detector evasion with semantic preservation. We evaluate six attack settings (M0-M5) on the full filtered MAGE test pool (15,310 human / 14,656 AI) against four detectors: RoBERTa, Fast-DetectGPT, Binoculars, and MAGE. StealthRL achieves near-zero detection on three of the four detectors and a 0.024 mean TPR@1%FPR, reducing mean AUROC from 0.79 to 0.43 and attaining a 97.6% attack success rate. Critically, attacks transfer to two held-out detectors not seen during training, revealing shared architectural vulnerabilities rather than detector-specific brittleness. We additionally conduct LLM-based quality evaluation via Likert scoring on 500 matched samples per method, analyze detector score distributions to explain why evasion succeeds, and provide per-detector AUROC with bootstrap confidence intervals. Our results expose significant robustness gaps in current AI-text detection and establish StealthRL as a principled adversarial evaluation protocol.

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

### Main output artifacts

The staged run produces:

- `method_runs/`: per-method generated outputs
- `detector_scores/`: per-detector parquet score files
- `assembled/metrics.json`: aggregate detector metrics
- `assembled/thresholds.json`: calibrated detector thresholds
- `assembled/quality.parquet`: automatic quality metrics
- `assembled/quality_gpt.parquet`: GPT/Likert quality ratings
- `assembled/figures/`: paper-ready plots

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
