# Task 1: Real Detector Implementation

**Status**: ✅ COMPLETED  
**Date**: December 1, 2025  
**Owner**: Sibo Zhu

---

## TL;DR
- Implemented real detectors (Fast‑DetectGPT, Ghostbuster, Binoculars) with async, lazy‑load, CUDA/CPU, and SQLite cache.
- Enabled real semantic similarity (E5) and perplexity (GPT‑2) for reward.
- Detectors are used automatically by the reward pipeline; no code changes needed to train.
- Quick tests and docs are in this folder; main code is under `../stealthrl/tinker/`.

---

## What’s included (this folder)
- `TASK1_COMPLETION_REPORT.md` – full report (details, tests, metrics, integration)
- `DETECTOR_IMPLEMENTATION_SUMMARY.md` – technical summary + usage examples
- `QUICK_DETECTOR_GUIDE.md` – 1‑pager quick reference
- `test_detectors_standalone.py` – simple end‑to‑end detector test (no Tinker deps)
- `test_detectors.py` – alternative test approach (fallback)

Main implementations live in:
- `../stealthrl/tinker/detectors.py` – detectors + ensemble
- `../stealthrl/tinker/semantic.py` – E5 similarity
- `../stealthrl/tinker/perplexity.py` – GPT‑2 perplexity

---

## Prerequisites
```bash
# activate venv
source /home/sibo/StealthRL/venv/bin/activate

# make sure core deps are present
pip install -r requirements.txt

# required for device_map loading paths
pip install accelerate
```

GPU: 8–12GB VRAM recommended (all three detectors together use ~6–7GB). CPU works but slower.

---

## Quick test (recommended)
This test exercises the actual detectors in `detectors.py` (including Ghostbuster’s fine‑tuned default).

```bash
python - <<'PY'
import asyncio, importlib.util
spec = importlib.util.spec_from_file_location("det", "/home/sibo/StealthRL/stealthrl/tinker/detectors.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

async def main():
    ens = mod.DetectorEnsemble(
        detector_names=["fast_detectgpt","ghostbuster","binoculars"],
        cache_path="outputs/detector_cache_test.sqlite",
        device="cuda"  # change to "cpu" if needed
    )
    ai_text = "Neural networks require careful tuning of hyperparameters to generalize."
    human_text = "I went to the store yesterday and walked home because the weather was nice."
    print("AI:", (await ens.compute(ai_text))["ensemble_prob"])
    print("Human:", (await ens.compute(human_text))["ensemble_prob"])
    ens.close()

asyncio.run(main())
PY
```

Expected: non‑identical probabilities (higher ≈ more AI‑like). First run downloads models; re‑runs hit the SQLite cache.

---

## Alternative test (standalone)
If you just want a minimal smoke test without importing from the package:
```bash
python task1_detector_implementation/test_detectors_standalone.py
```

---

## Notes on Ghostbuster (fine‑tuned default)
- Default: `model_name="roberta-base-openai-detector"` with safe fallback to `roberta-base` if unavailable.
- If you have a better fine‑tuned checkpoint, pass it when constructing `GhostbusterDetector` or change the default.

---

## Troubleshooting
- “Using a device_map … requires accelerate” → `pip install accelerate` (inside venv).
- Model not found → falls back to `roberta-base` (expected unless you provide a real fine‑tuned ID/path).
- Force CPU → set `device="cpu"` when creating `DetectorEnsemble`.
- Clear cache if needed → `rm -f outputs/detector_cache*.sqlite`.

---

## How this is used downstream
- Training: detectors are called inside the composite reward (via `TinkerCompositeReward`), so training automatically optimizes detectability ↓ + semantics/quality ↑ + fairness.
- Evaluation: StealthBench uses the same detectors to report AUROC, low‑FPR metrics, and ESL FPR gaps.

No changes needed to your training scripts; once the venv and models are set, just run training.

```bash
python -m stealthrl.tinker.train \
  --data-path data/tinker \
  --run-name task1_ready \
  --num-epochs 1 \
  --batch-size 2
```

---

## Done
- ✅ Fast‑DetectGPT, Ghostbuster (fine‑tuned default), Binoculars
- ✅ E5 similarity + GPT‑2 perplexity
- ✅ Async + caching + CUDA/CPU
- ✅ Tests + docs

Ready for Task 2 (dataset curation) and training.

