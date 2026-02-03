#!/usr/bin/env python3
"""
Generate multi-FPR TPR tables from a completed eval run.

Outputs:
- tpr_by_fpr.csv (long format, per detector)
- table_fpr_mean.tex (LaTeX table; mean across detectors)
- table_fpr_by_detector.tex (optional; per detector)

Example:
  python scripts/make_fpr_table.py --run-dir outputs/eval_runs/mage_qvse \
    --fprs 0.001 0.01 0.05 --n-bootstrap 500
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics import compute_threshold_at_fpr, compute_tpr_at_fpr
from eval.plots import METHOD_NAMES

logger = logging.getLogger(__name__)


def _bootstrap_ci_tpr(
    ai_scores: np.ndarray,
    threshold: float,
    n_bootstrap: int = 0,
    confidence: float = 0.95,
    seed: int = 42,
) -> Optional[Tuple[float, float]]:
    if n_bootstrap <= 0:
        return None
    if len(ai_scores) == 0:
        return None

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        sample = rng.choice(ai_scores, size=len(ai_scores), replace=True)
        boot.append(np.mean(sample >= threshold))
    alpha = 1.0 - confidence
    lo = np.percentile(boot, (alpha / 2.0) * 100)
    hi = np.percentile(boot, (1.0 - alpha / 2.0) * 100)
    return float(lo), float(hi)


def _format_cell(val: float, ci: Optional[Tuple[float, float]], precision: int) -> str:
    if ci is None:
        return f"{val:.{precision}f}"
    lo, hi = ci
    return f"{val:.{precision}f} [{lo:.{precision}f},{hi:.{precision}f}]"


def _latex_table(
    rows: List[List[str]],
    header: List[str],
    caption: str,
    label: str,
    align: str,
) -> str:
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\begin{{tabular}}{{{align}}}")
    lines.append("  \\toprule")
    lines.append("  " + " & ".join(header) + " \\")
    lines.append("  \\midrule")
    for row in rows:
        lines.append("  " + " & ".join(row) + " \\")
    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate multi-FPR TPR tables from eval run")
    parser.add_argument("--run-dir", type=str, required=True, help="Eval run directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: run-dir/tables)")
    parser.add_argument("--fprs", nargs="+", type=float, default=[0.001, 0.01, 0.05], help="FPRs to report")
    parser.add_argument("--n-bootstrap", type=int, default=0, help="Bootstrap samples for CI (0=off)")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI")
    parser.add_argument("--precision", type=int, default=3, help="Decimal precision")
    parser.add_argument("--by-detector", action="store_true", help="Also emit per-detector LaTeX table")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = run_dir / "scores.parquet"
    if not scores_path.exists():
        scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.parquet or scores.csv not found in {run_dir}")

    scores = pd.read_parquet(scores_path) if scores_path.suffix == ".parquet" else pd.read_csv(scores_path)
    score_col = "detector_score_ai" if "detector_score_ai" in scores.columns else "detector_score"

    required_cols = {"sample_id", "dataset", "label", "method", "detector_name", score_col}
    missing = required_cols - set(scores.columns)
    if missing:
        raise ValueError(f"Scores file missing columns: {missing}")

    datasets = sorted(scores["dataset"].dropna().unique().tolist())
    detectors = sorted(scores["detector_name"].dropna().unique().tolist())
    methods = [m for m in scores["method"].unique().tolist() if m != "human"]

    # Preferred method order
    preferred = ["m0", "m1", "m2", "m3", "m4", "m5"]
    methods_sorted = [m for m in preferred if m in methods] + [m for m in methods if m not in preferred]

    rows = []
    for dataset in datasets:
        ds_scores = scores[scores["dataset"] == dataset]

        for fpr in args.fprs:
            for det in detectors:
                human_scores = ds_scores[
                    (ds_scores["label"] == "human") & (ds_scores["detector_name"] == det)
                ][score_col].dropna().to_numpy()
                if len(human_scores) == 0:
                    continue
                thr = compute_threshold_at_fpr(human_scores, target_fpr=fpr)

                for method in methods_sorted:
                    ai_scores = ds_scores[
                        (ds_scores["label"] == "ai") &
                        (ds_scores["detector_name"] == det) &
                        (ds_scores["method"] == method)
                    ][score_col].dropna().to_numpy()
                    if len(ai_scores) == 0:
                        continue
                    tpr = compute_tpr_at_fpr(ai_scores, thr)
                    ci = _bootstrap_ci_tpr(
                        ai_scores,
                        thr,
                        n_bootstrap=args.n_bootstrap,
                        confidence=args.confidence,
                        seed=args.seed,
                    )
                    rows.append({
                        "dataset": dataset,
                        "detector": det,
                        "method": method,
                        "fpr": fpr,
                        "tpr": tpr,
                        "ci_low": None if ci is None else ci[0],
                        "ci_high": None if ci is None else ci[1],
                    })

    if not rows:
        raise RuntimeError("No rows generated. Check inputs and run directory.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "tpr_by_fpr.csv", index=False)

    # Create mean table per dataset (mean across detectors)
    for dataset in datasets:
        ds = out_df[out_df["dataset"] == dataset].copy()
        mean_df = ds.groupby(["method", "fpr"]).agg({"tpr": "mean"}).reset_index()

        # Optional CI for means (use mean of bounds if provided)
        if args.n_bootstrap > 0 and "ci_low" in ds.columns:
            ci_mean = ds.groupby(["method", "fpr"]).agg({
                "ci_low": "mean",
                "ci_high": "mean",
            }).reset_index()
            mean_df = mean_df.merge(ci_mean, on=["method", "fpr"], how="left")
        else:
            mean_df["ci_low"] = None
            mean_df["ci_high"] = None

        # Build LaTeX table
        header = ["Method"] + [f"TPR@{int(f*1000)/10:g}\\%FPR" for f in args.fprs]
        align = "l" + "c" * len(args.fprs)
        table_rows = []
        for method in methods_sorted:
            row = [METHOD_NAMES.get(method, method)]
            for fpr in args.fprs:
                cell = mean_df[(mean_df["method"] == method) & (mean_df["fpr"] == fpr)]
                if cell.empty:
                    row.append("-")
                else:
                    tpr = float(cell["tpr"].values[0])
                    ci = None
                    if args.n_bootstrap > 0 and not pd.isna(cell["ci_low"].values[0]):
                        ci = (float(cell["ci_low"].values[0]), float(cell["ci_high"].values[0]))
                    row.append(_format_cell(tpr, ci, args.precision))
            table_rows.append(row)

        caption = f"Mean TPR at multiple FPR thresholds on {dataset.upper()} (lower is better for the attacker)."
        label = f"tab:fpr_mean_{dataset}"
        tex = _latex_table(table_rows, header, caption, label, align)

        out_path = out_dir / f"table_fpr_mean_{dataset}.tex"
        out_path.write_text(tex)
        logger.info(f"Saved {out_path}")

    # Optional per-detector table (single dataset only to avoid huge tables)
    if args.by_detector and len(datasets) == 1:
        dataset = datasets[0]
        ds = out_df[out_df["dataset"] == dataset].copy()
        header = ["Method"]
        for det in detectors:
            for fpr in args.fprs:
                header.append(f"{det}\\, {int(fpr*1000)/10:g}\\%")
        align = "l" + "c" * (len(detectors) * len(args.fprs))
        table_rows = []
        for method in methods_sorted:
            row = [METHOD_NAMES.get(method, method)]
            for det in detectors:
                for fpr in args.fprs:
                    cell = ds[(ds["method"] == method) & (ds["detector"] == det) & (ds["fpr"] == fpr)]
                    if cell.empty:
                        row.append("-")
                    else:
                        tpr = float(cell["tpr"].values[0])
                        ci = None
                        if args.n_bootstrap > 0 and not pd.isna(cell["ci_low"].values[0]):
                            ci = (float(cell["ci_low"].values[0]), float(cell["ci_high"].values[0]))
                        row.append(_format_cell(tpr, ci, args.precision))
            table_rows.append(row)

        caption = f"TPR at multiple FPR thresholds per detector on {dataset.upper()}."
        label = f"tab:fpr_by_detector_{dataset}"
        tex = _latex_table(table_rows, header, caption, label, align)

        out_path = out_dir / f"table_fpr_by_detector_{dataset}.tex"
        out_path.write_text(tex)
        logger.info(f"Saved {out_path}")

    # Save metadata
    meta = {
        "run_dir": str(run_dir),
        "datasets": datasets,
        "detectors": detectors,
        "methods": methods_sorted,
        "fprs": args.fprs,
        "n_bootstrap": args.n_bootstrap,
        "confidence": args.confidence,
    }
    with open(out_dir / "tpr_by_fpr_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()

