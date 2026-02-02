# StealthRL Evaluation SPEC (ICLR Workshop Submission Ready)

This document is **builder-agent instructions** for implementing the full evaluation suite for **StealthRL** (assume the RL-trained paraphrase policy checkpoint already exists).  
Goal: produce **paper-ready tables/figures**, reproducible metrics, and standardized comparisons vs strong prior work.

---

## 0) High-level goal and philosophy

StealthRL is a **paraphrase attack**: it rewrites **AI-generated text** to reduce detection by AI text detectors while preserving meaning & fluency.

We want evaluations that match what **strong detector-evasion papers** report:

- **Multi-detector panel, multiple detector families** (classifier, curvature/likelihood, zero-shot, and optionally watermark).  
  Inspired by: *Evading AI-Generated Content Detectors using Homoglyphs* (A. Creo et al., arXiv:2406.11239).  
  - Paper: https://arxiv.org/abs/2406.11239  
  - HTML: https://arxiv.org/html/2406.11239v1  
- **Low-FPR operating point** metrics (e.g., **TPR@1%FPR**, sometimes called **T@1%F**) and robustness/transfer across detectors.  
  Inspired by: *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text* (Cheng et al., arXiv:2506.07001).  
  - Paper: https://arxiv.org/abs/2506.07001  
  - HTML: https://arxiv.org/html/2506.07001v1  
  - Repo: https://github.com/chengez/Adversarial-Paraphrasing  
- **Transfer matrices/heatmaps across detectors** and **ASR** framing.  
  Inspired by AuthorMist (arXiv:2503.08716).  
  - Paper: https://arxiv.org/abs/2503.08716  
  - HF “paper” page: https://huggingface.co/papers/2503.08716  
  - Model (Originality): https://huggingface.co/authormist/authormist-originality  

---

## 1) Assumptions about current StealthRL repo state

Builder should assume:

- StealthRL generates paraphrases using a policy LM (Qwen/Qwen3-4B-Instruct-2507 + LoRA) with:
  - **one-shot per chunk** (no multi-round loop)
  - can sample **N candidates** (default 4) and **select best** by a score
- Training reward used:
  - Detector ensemble: `openai-community/roberta-large-openai-detector` + Fast-DetectGPT
  - Semantic similarity (E5)
  - Perplexity reward  
- Current local detectors implemented: RoBERTa OpenAI detector + Fast-DetectGPT, both output continuous scores.

---

## 2) Deliverables (must ship)

### 2.1 Files + outputs
Implement a single evaluation runner that produces:

- `artifacts/scores.parquet` (or `.csv` if needed): **per-sample** detector scores for every (dataset × method × detector)
- `artifacts/quality.parquet`: similarity, perplexity, edit metrics, validity checks
- `artifacts/metrics.json`: aggregated metrics + bootstrap CIs
- `artifacts/figures/`:
  - `fig_heatmap_t1f.png` (detector × method heatmap)
  - `fig_tradeoff.png` (evasion–quality tradeoff)
  - `fig_budget_sweep.png` (candidate budget sweep)
  - `fig_sanitize.png` (homoglyph “defense-aware” sanitize effect; optional but recommended)
- `artifacts/tables/`:
  - `table_main_mage.md` (or `.tex`)
  - `table_transfer.md`
  - `table_padben.md` (if PadBen included)
  - `table_sanitize.md` (if homoglyph defense included)
- `artifacts/qual_examples.md`: 6–10 curated examples

### 2.2 Paper-ready “required” figures/tables
Minimum paper-ready items:

1) **Table A (Main Results, MAGE)**  
   - Columns per detector: `AUC`, `TPR@1%FPR`, `ASR@1%FPR-threshold`, plus **mean across detectors**
2) **Figure 1 (Transfer Heatmap)**  
   - Detector × Method heatmap for `TPR@1%FPR` (preferred)
3) **Figure 2 (Tradeoff curve)**  
   - x: similarity, y: mean TPR@1%FPR across detectors (lower is “better” for attacker)
4) **Figure 3 (Budget sweep)**  
   - N candidates vs mean TPR@1%FPR + similarity trend

---

## 3) Standard datasets / benchmarks to use

We want at least **one recognizable benchmark** + one **held-out / stress test**.

### 3.1 Primary benchmark: MAGE (recommended)
- Paper: https://arxiv.org/html/2305.13242v3  
- GitHub: https://github.com/yafuly/MAGE  
- Hugging Face dataset card: https://huggingface.co/datasets/yaful/MAGE  

**Builder tasks (MAGE):**
1) Download MAGE.
2) Construct a balanced evaluation set:
   - `N_human = 1000`, `N_ai = 1000` (if tight compute: 500/500)
3) Enforce length window: e.g. 100–500 tokens (configurable).
4) Save to `data/mage_eval.jsonl` with schema:
   - `id, label (human|ai), domain, generator (if available), text`

### 3.2 Secondary benchmark: RAID (OOD/adversarial stress test)
RAID is a large benchmark for robust evaluation.  
- ACL Anthology: https://aclanthology.org/2024.acl-long.674/  
- Paper PDF: https://aclanthology.org/2024.acl-long.674.pdf  
- arXiv: https://arxiv.org/abs/2405.07940  
- GitHub: https://github.com/liamdugan/raid  

**Builder tasks (RAID, lightweight slice):**
1) Sample a small subset, e.g.:
   - 200 human + 200 AI from 2 domains and 2 generators (choose easiest fields available)
2) Save to `data/raid_slice.jsonl` with same schema as above.

### 3.3 Optional: PadBen (paraphrase robustness benchmark)
- GitHub: https://github.com/JonathanZha47/PadBen-Paraphrase-Attack-Benchmark  
- HF dataset: https://huggingface.co/datasets/JonathanZha/PADBen  
- Paper (arXiv): https://arxiv.org/pdf/2511.00416  

**Builder tasks (PadBen):**
- Add an evaluation section that runs StealthRL outputs through PadBen’s relevant tasks.
- Start with PadBen dataset + its evaluation scripts; keep results separate from MAGE/RAID.

---

## 4) Detector panel (must cover families)

You already have two detectors. Add at least **2 held-out detectors** so “transfer” is credible.

### 4.1 Required detectors (minimum set = 4)
1) **RoBERTa OpenAI Detector (classifier family)**  
   - HF model: https://huggingface.co/openai-community/roberta-large-openai-detector
2) **Fast-DetectGPT (curvature family)**  
   - Repo: https://github.com/baoguangsheng/fast-detect-gpt  
   - OpenReview: https://openreview.net/forum?id=Bpcgcr8E8Z  
   - arXiv: https://arxiv.org/abs/2310.05130
3) **DetectGPT (curvature family; classic)**  
   - Paper: https://arxiv.org/abs/2301.11305  
   - Repo: https://github.com/eric-mitchell/detect-gpt
4) **Binoculars (zero-shot family)**  
   - Paper: https://arxiv.org/abs/2401.12070  
   - Repo: https://github.com/ahans30/Binoculars

### 4.2 Optional (strong): Ghostbuster
- Paper: https://arxiv.org/abs/2305.15047  
- Code: https://github.com/vivek3141/ghostbuster  
- NAACL paper page: https://aclanthology.org/2024.naacl-long.95/

### 4.3 Optional (very strong): watermark-based detector(s)
If feasible, add at least one watermark detector to cover the watermark family (as in Adversarial Paraphrasing).  
- Repo (Adversarial Paraphrasing): https://github.com/chengez/Adversarial-Paraphrasing

> If watermarking is too expensive, skip; do not block core eval.

---

## 5) Methods / baselines to compare

Implement each method as a **generator** that takes `{id, text}` and outputs one or more paraphrases.

### 5.1 Required methods

**M0: No attack**
- Use original AI text.

**M1: Simple paraphrase**
- Base LM: Qwen/Qwen3-4B-Instruct-2507 **without** RL/LoRA (or with a “vanilla” paraphraser model).
- Prompt template: same “Please paraphrase…” template as StealthRL.
- Generate:
  - single sample (N=1)
  - best-of-4 (N=4) via reranking

**M2: StealthRL**
- Use RL policy checkpoint.
- Generate:
  - single sample (N=1)
  - best-of-4 (N=4) via reranking

**M3: Adversarial Paraphrasing baseline (NeurIPS 2025 paper)**
Preferred: run their repo and export outputs.  
- Repo: https://github.com/chengez/Adversarial-Paraphrasing  
- Paper: https://arxiv.org/abs/2506.07001  

Fallback (fast integrated approximation): “detector-guided selection”
- Sample K paraphrases from the base LM (K in {4,8}).
- Select candidate minimizing “AI score” under a **guidance detector** (e.g., RoBERTa detector).
- Enforce similarity threshold to avoid semantic drift, e.g. E5 cosine >= 0.90.

**M4: AuthorMist**
Run the published model:
- HF model: https://huggingface.co/authormist/authormist-originality  
- Paper: https://arxiv.org/abs/2503.08716

### 5.2 Optional (but recommended): Homoglyph baseline
Inspired by SilverSpeak/homoglyph attack.
- Paper: https://arxiv.org/abs/2406.11239  
Implement a simple homoglyph substitution transform with a **budget sweep**: p ∈ {0.1%, 0.5%, 1%, 2% chars}.

---

## 6) Core metrics (match strong papers)

We compute metrics per **detector**, per **method**, per **dataset**.

### 6.1 Calibration: define 1% FPR threshold on human text
For each detector `d`:
1) Run `d` on **human-original** set → scores `s_h`.
2) Define threshold `t_d` such that **FPR = 1%**:
   - If higher score = “more AI”, `t_d = quantile(s_h, 0.99)`
   - If lower score = “more AI”, use `t_d = quantile(s_h, 0.01)` and invert direction.
3) Persist thresholds to `artifacts/thresholds.json`.

### 6.2 Detector-side metrics
Per detector `d` and method `m`:

- **AUROC**
  - binary labels: human vs (AI attacked by method m)
  - score: detector raw score (ensure unified “higher=more AI” convention)

- **TPR@1%FPR** (a.k.a. T@1%F)
  - `TPR_d,m = mean(score_ai_attacked >= t_d)`
  - Report both:
    - `TPR_d,origAI` (baseline)
    - `TPR_d,m` (after attack)
  - This metric is central in Adversarial Paraphrasing.  
    Paper: https://arxiv.org/abs/2506.07001

- **ASR (Attack Success Rate)**
  - `ASR_d,m = mean(score_ai_attacked < t_d)`
  - This aligns with “bypass rate” framing and is used widely in detector evasion / AuthorMist style reporting.

### 6.3 Text quality metrics
Compute per sample; aggregate mean/median and report distributions:

- **Semantic similarity**: E5 cosine (already used in StealthRL)
- **Perplexity**: use a fixed LM scorer for all methods
- **Edit metrics**:
  - token change rate (edit distance / % changed tokens)
  - chrF (optional)
- **Validity metrics**:
  - parse success / non-empty / length ratio constraints (already in generation)

### 6.4 Statistical reliability
- Bootstrap over examples (e.g., 1000 resamples):
  - CI for AUROC and TPR@1%FPR
- Save in `metrics.json` and include in table footnotes.

---

## 7) Transfer evaluation (headline figure)

### 7.1 Transfer heatmap
Create a detector × method heatmap:
- y-axis: detectors
- x-axis: methods
- value: **TPR@1%FPR** (preferred) or AUROC

This mimics the “cross-detector” story used in strong works (AuthorMist matrices / transfer tests).

### 7.2 “Guidance vs deploy” transfer (recommended)
To emulate Adversarial Paraphrasing’s “guidance_classifier vs deploy_classifier” idea:
- During selection/reranking, try 3 scoring rules:
  1) choose candidate minimizing RoBERTa AI score
  2) choose candidate minimizing Fast-DetectGPT AI score
  3) choose candidate minimizing ensemble mean score  
Evaluate all three across **all** detectors. This becomes a “selection-guidance transfer” ablation.

---

## 8) Budget sweeps and tradeoff curves (fast + valuable)

### 8.1 Candidate budget sweep (must do)
For M1, M2, and M3(fallback guided selection):
- N ∈ {1, 2, 4, 8}
- Generate N candidates per sample; select best by method scoring rule.
- Plot:
  - **mean TPR@1%FPR** across detectors vs N
  - also plot median similarity vs N (on secondary axis or separate plot)

### 8.2 Tradeoff / Pareto curve (must do)
Plot each method (and/or each N setting) as:
- x = median similarity
- y = mean TPR@1%FPR across detectors  
Lower y = more successful attack; higher x = better fidelity.

### 8.3 Homoglyph sweep (optional but recommended)
For homoglyph baseline:
- p ∈ {0.1%, 0.5%, 1%, 2%} chars substituted
- Plot p vs TPR@1%FPR and p vs readability proxy

---

## 9) Defense-aware “sanitize” check (quick win)

Inspired by homoglyph paper’s discussion of tokenization changes + defenses:
- Paper: https://arxiv.org/abs/2406.11239

Implement:
- `sanitize(text)`:
  - Unicode normalize: NFKC
  - remove zero-width characters
  - optionally map known homoglyphs back to ASCII
Re-score detectors on:
- homoglyph outputs, before and after sanitize
- (optionally) StealthRL outputs too (should be stable)

Produce `table_sanitize.md` + `fig_sanitize.png`.

---

## 10) Output schema and logging (make this robust)

### 10.1 scores table schema (`scores.parquet`)
One row per `(sample_id, dataset, method, detector, setting, candidate_rank)`:

- `sample_id`
- `dataset` (mage / raid_slice / padben)
- `label` (human/ai)
- `method` (M0..M5)
- `setting` (e.g., `N=4_rerank=roberta`, `homoglyph_p=0.005`)
- `candidate_rank` (0 = selected output)
- `text_out`
- `len_tokens_out`
- `len_ratio`
- `detector_name`
- `detector_score_raw`
- `detector_score_ai` (unified direction: higher = more AI)

### 10.2 quality table schema (`quality.parquet`)
- `sample_id, dataset, method, setting`
- `sim_e5`
- `ppl_score`
- `edit_rate`
- `chrf` (optional)
- `valid` (bool)
- `fail_reason` (string)

### 10.3 metrics json schema (`metrics.json`)
- nested dict by `dataset -> detector -> method -> metrics`
- include CIs:
  - `auc_mean, auc_ci_low, auc_ci_high`
  - `tpr1_mean, tpr1_ci_low, tpr1_ci_high`
  - `asr_mean, asr_ci_low, asr_ci_high`
- include thresholds `t_d` per detector

---

## 11) Implementation plan (suggested structure)

### 11.1 Modules
- `eval/data.py`
  - download/load MAGE/RAID/PadBen
  - sampling, token length filters
- `eval/methods/`
  - `stealthrl.py` (RL policy + sampling + reranking)
  - `simple_paraphrase.py` (base LM)
  - `adversarial_paraphrasing.py` (wrapper to repo OR fallback guided selection)
  - `authormist.py` (HF inference)
  - `homoglyph.py` (text transform + budget sweep)
- `eval/detectors/`
  - `roberta_openai.py`
  - `fast_detectgpt.py`
  - `detectgpt.py`
  - `binoculars.py`
  - (optional) `ghostbuster.py`
- `eval/metrics.py`
  - calibration thresholds
  - AUC, TPR@1%FPR, ASR
  - bootstrap CIs
- `eval/plots.py`
  - heatmap, tradeoff, sweeps

### 11.2 CLI
Provide one command that runs everything:
```bash
python -m eval.run \
  --datasets mage raid_slice \
  --methods m0 m1 m2 m3 m4 m5 \
  --detectors roberta fast_detectgpt detectgpt binoculars \
  --n_candidates 1 2 4 8 \
  --out_dir artifacts/
```

---

## 12) Acceptance criteria (what “done” means)

Evaluation work is complete when:

1) Running the CLI produces **all artifacts** in §2 with no manual steps.
2) Table A + Figure 1 + Figure 2 exist and look sane:
   - original AI has high TPR@1%FPR
   - attacks reduce TPR@1%FPR (more is better for attacker)
   - similarity stays above a target threshold (e.g., median >= 0.90 for StealthRL at N=4)
3) Transfer story is supported:
   - StealthRL reduces detection on at least 2 held-out detectors vs trained-on ones
4) Comparisons included:
   - AuthorMist results present
   - Adversarial Paraphrasing baseline present (repo-run or fallback guided selection)
   - Homoglyph baseline present (at least p=1% and a sweep)

---

## 13) Notes / pitfalls

- **Score direction**: detectors differ (some higher=AI, some lower=AI). Normalize to a single convention.
- **Calibration** must be done on **human text only** for each dataset slice; do not leak AI samples into thresholding.
- Ensure paraphrase outputs pass basic validity checks:
  - non-empty
  - >= 10 words
  - <= 3× original length
- Keep all randomness reproducible with `seed`.
- Log **raw outputs** for auditing (JSONL is fine).

---

## 14) Handy references (quick link list)

### Benchmarks
- MAGE paper: https://arxiv.org/html/2305.13242v3  
- MAGE GitHub: https://github.com/yafuly/MAGE  
- MAGE HF dataset: https://huggingface.co/datasets/yaful/MAGE  
- RAID ACL page: https://aclanthology.org/2024.acl-long.674/  
- RAID PDF: https://aclanthology.org/2024.acl-long.674.pdf  
- RAID GitHub: https://github.com/liamdugan/raid  
- PadBen GitHub: https://github.com/JonathanZha47/PadBen-Paraphrase-Attack-Benchmark  
- PadBen HF dataset: https://huggingface.co/datasets/JonathanZha/PADBen  

### Detector & detector-robustness papers
- Homoglyph attack paper: https://arxiv.org/abs/2406.11239  
- Adversarial Paraphrasing paper: https://arxiv.org/abs/2506.07001  
- Adversarial Paraphrasing repo: https://github.com/chengez/Adversarial-Paraphrasing  
- AuthorMist paper: https://arxiv.org/abs/2503.08716  
- AuthorMist model: https://huggingface.co/authormist/authormist-originality  
- DetectGPT paper: https://arxiv.org/abs/2301.11305  
- DetectGPT repo: https://github.com/eric-mitchell/detect-gpt  
- Fast-DetectGPT repo: https://github.com/baoguangsheng/fast-detect-gpt  
- Fast-DetectGPT OpenReview: https://openreview.net/forum?id=Bpcgcr8E8Z  
- Binoculars paper: https://arxiv.org/abs/2401.12070  
- Binoculars repo: https://github.com/ahans30/Binoculars  
- Ghostbuster paper: https://arxiv.org/abs/2305.15047  
- Ghostbuster repo: https://github.com/vivek3141/ghostbuster  

### Models used in StealthRL reward/training
- RoBERTa OpenAI detector model: https://huggingface.co/openai-community/roberta-large-openai-detector

---

*End of SPEC.*
