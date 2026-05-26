# StealthRL Anonymous Review Code

This branch contains anonymized software for the ARR submission
`StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors`.

It is curated for double-blind review. It intentionally excludes author-identifying documentation,
historical submission folders, local caches, raw environment files, model/API secrets, paper source,
and generated evaluation artifacts. The compact result/data artifacts are provided separately through
the ARR data upload field.

## Contents

```text
stealthrl/                         Core training, reward, detector, and model modules
eval/                              Evaluation pipeline and attack/detector wrappers
scripts/                           Reproduction and analysis entry points
configs/                           Training and evaluation YAML configs
analysis/                          Dataset and detector-analysis helper scripts
tests/                             Lightweight smoke and integration tests
requirements.txt                   Python dependency list
environment.yml                    Conda environment specification
```

## Setup

```bash
conda env create -f environment.yml
conda activate stealthrl
pip install -r requirements.txt
```

For API-backed components, create a local secret file that is not committed or uploaded:

```bash
mkdir -p .secrets
cp .env.example .secrets/eval.env
```

Then fill only the keys required for the command being run. The archived package does not contain
any private keys.

## Reproducing Tables And Plots From A Run Directory

```bash
python -m eval --help
python scripts/finalize_eval_run.py --help
python eval/plots.py --help
```

## Running A Fresh Evaluation

The staged full-MAGE runner is:

```bash
python scripts/run_full_mage_research_eval.py \
  --run-root outputs/eval_runs/full_mage_research \
  --env-file .secrets/eval.env \
  --checkpoint-json .secrets/m2_checkpoint.json \
  --methods m0 m1 m2 m3 m4 m5 \
  --detectors roberta fast_detectgpt binoculars mage
```

The M2 checkpoint/adapter path must be supplied by the user in `.secrets/m2_checkpoint.json` or via
the relevant script arguments. Large model weights and generated evaluation artifacts are not stored
in this branch.

## Responsible Use

This package is intended for confidential peer review and reproducibility checks of robustness
experiments. The code evaluates detector robustness under adversarial paraphrasing; it should not be
used to facilitate academic misconduct, impersonation, or operational detector evasion.
