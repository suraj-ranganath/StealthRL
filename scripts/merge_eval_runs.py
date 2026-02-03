#!/usr/bin/env python3
"""
Merge eval runs to combine methods (e.g., base run without M2 + later M2-only run).

This produces a merged run directory with:
- scores.parquet/csv (combined)
- quality.parquet/csv (combined)
- raw_outputs.json (combined, if present)
- metrics.json + thresholds.json (recomputed)
- figures/ + tables/ (regenerated)

Example:
  python scripts/merge_eval_runs.py \
    --base-dir outputs/eval_runs/mage_no_m2 \
    --add-dir outputs/eval_runs/mage_only_m2 \
    --add-methods m2 \
    --out-dir outputs/eval_runs/mage_merged
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics import calibrate_thresholds, compute_detector_metrics, save_thresholds
from eval.plots import generate_all_plots, generate_all_tables

logger = logging.getLogger(__name__)


def _read_df(run_dir: Path, stem: str) -> pd.DataFrame:
    parquet = run_dir / f"{stem}.parquet"
    csv = run_dir / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv in {run_dir}")


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _validate_sample_ids(
    base_scores: pd.DataFrame,
    add_scores: pd.DataFrame,
    datasets: List[str],
) -> None:
    for dataset in datasets:
        base_ds = base_scores[base_scores["dataset"] == dataset]
        add_ds = add_scores[add_scores["dataset"] == dataset]

        # Human IDs
        base_h = set(base_ds[base_ds["label"] == "human"]["sample_id"].unique())
        add_h = set(add_ds[add_ds["label"] == "human"]["sample_id"].unique())
        if add_h and base_h != add_h:
            raise ValueError(
                f"Human sample_id mismatch for dataset '{dataset}'. "
                f"Base={len(base_h)} Add={len(add_h)}"
            )

        # AI IDs
        base_a = set(base_ds[base_ds["label"] == "ai"]["sample_id"].unique())
        add_a = set(add_ds[add_ds["label"] == "ai"]["sample_id"].unique())
        if add_a and base_a != add_a:
            raise ValueError(
                f"AI sample_id mismatch for dataset '{dataset}'. "
                f"Base={len(base_a)} Add={len(add_a)}"
            )


def _merge_raw_outputs(base_dir: Path, add_dir: Path, out_dir: Path) -> None:
    base_raw = _read_json(base_dir / "raw_outputs.json") or {}
    add_raw = _read_json(add_dir / "raw_outputs.json") or {}
    if not base_raw and not add_raw:
        return

    merged = {}
    for dataset, methods in base_raw.items():
        merged[dataset] = dict(methods)
    for dataset, methods in add_raw.items():
        merged.setdefault(dataset, {})
        for method, outputs in methods.items():
            if method not in merged[dataset]:
                merged[dataset][method] = outputs

    with open(out_dir / "raw_outputs.json", "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Merged raw_outputs.json")


def _recompute_metrics(
    scores_df: pd.DataFrame,
    detectors: List[str],
    methods: List[str],
    datasets: List[str],
    n_bootstrap: int,
    seed: int,
    out_dir: Path,
) -> List[Dict]:
    all_metrics: List[Dict] = []

    # Compute thresholds per detector (human only, per dataset pooled)
    thresholds: Dict[str, float] = {}
    for det in detectors:
        human_scores = scores_df[
            (scores_df["label"] == "human") & (scores_df["detector_name"] == det)
        ]["detector_score"].dropna().tolist()
        if not human_scores:
            continue
        thresholds[det] = calibrate_thresholds({det: human_scores})[det]
    save_thresholds(thresholds, str(out_dir / "thresholds.json"))

    for dataset in datasets:
        ds = scores_df[scores_df["dataset"] == dataset]
        human_all = ds[ds["label"] == "human"]

        for det in detectors:
            human_scores = human_all[
                human_all["detector_name"] == det
            ]["detector_score"].dropna().tolist()
            if not human_scores:
                continue

            for method in methods:
                ai_scores = ds[
                    (ds["label"] == "ai") &
                    (ds["method"] == method) &
                    (ds["detector_name"] == det)
                ]["detector_score"].dropna().tolist()
                if not ai_scores:
                    continue

                metrics = compute_detector_metrics(
                    human_scores=human_scores,
                    ai_scores=ai_scores,
                    detector=det,
                    method=method,
                    dataset=dataset,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                all_metrics.append(metrics.to_dict())

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Merge StealthRL eval runs")
    parser.add_argument("--base-dir", type=str, required=True, help="Base run dir (no M2)")
    parser.add_argument("--add-dir", type=str, required=True, help="Add run dir (M2-only)")
    parser.add_argument("--out-dir", type=str, required=True, help="Merged output dir")
    parser.add_argument("--add-methods", nargs="+", default=["m2"], help="Methods to import from add run")
    parser.add_argument("--n-bootstrap", type=int, default=500, help="Bootstrap samples for metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables too")
    parser.add_argument("--no-fpr-table", action="store_true", help="Skip multi-FPR table generation")
    parser.add_argument("--fprs", nargs="+", type=float, default=[0.001, 0.01, 0.05], help="FPRs for table")
    parser.add_argument("--fpr-bootstrap", type=int, default=500, help="Bootstrap samples for FPR table")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    add_dir = Path(args.add_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_scores = _read_df(base_dir, "scores")
    add_scores = _read_df(add_dir, "scores")
    # Normalize score column name if needed
    for df in (base_scores, add_scores):
        if "detector_score" not in df.columns and "detector_score_ai" in df.columns:
            df["detector_score"] = df["detector_score_ai"]
    base_quality = _read_df(base_dir, "quality") if (base_dir / "quality.parquet").exists() or (base_dir / "quality.csv").exists() else pd.DataFrame()
    add_quality = _read_df(add_dir, "quality") if (add_dir / "quality.parquet").exists() or (add_dir / "quality.csv").exists() else pd.DataFrame()

    detectors_base = sorted(base_scores["detector_name"].dropna().unique().tolist())
    detectors_add = sorted(add_scores["detector_name"].dropna().unique().tolist())
    if detectors_base != detectors_add:
        raise ValueError(f"Detector mismatch: base={detectors_base}, add={detectors_add}")

    datasets = sorted(base_scores["dataset"].dropna().unique().tolist())
    datasets_add = sorted(add_scores["dataset"].dropna().unique().tolist())
    if datasets != datasets_add:
        raise ValueError(f"Dataset mismatch: base={datasets}, add={datasets_add}")

    _validate_sample_ids(base_scores, add_scores, datasets)

    # Filter add to methods we want
    add_scores = add_scores[add_scores["method"].isin(args.add_methods)]
    if not add_scores.empty:
        logger.info(f"Adding methods from add run: {sorted(add_scores['method'].unique().tolist())}")
    else:
        raise ValueError("No rows found for requested add methods in add run.")

    # Merge scores (dedupe on (sample_id, dataset, label, method, detector))
    merged_scores = pd.concat([base_scores, add_scores], ignore_index=True)
    merged_scores = merged_scores.drop_duplicates(
        subset=["sample_id", "dataset", "label", "method", "detector_name"],
        keep="first",
    )

    # Merge quality metrics (dedupe on (sample_id, dataset, method, setting))
    if not add_quality.empty:
        add_quality = add_quality[add_quality["method"].isin(args.add_methods)]
    merged_quality = pd.concat([base_quality, add_quality], ignore_index=True) if not base_quality.empty or not add_quality.empty else pd.DataFrame()
    if not merged_quality.empty:
        dedup_cols = ["sample_id", "method", "setting"]
        if "dataset" in merged_quality.columns:
            dedup_cols.insert(1, "dataset")
        merged_quality = merged_quality.drop_duplicates(
            subset=dedup_cols,
            keep="first",
        )

    # Write merged scores/quality
    merged_scores.to_parquet(out_dir / "scores.parquet")
    merged_scores.to_csv(out_dir / "scores.csv", index=False)
    if not merged_quality.empty:
        merged_quality.to_parquet(out_dir / "quality.parquet")
        merged_quality.to_csv(out_dir / "quality.csv", index=False)

    # Merge raw outputs if present
    _merge_raw_outputs(base_dir, add_dir, out_dir)

    # Save merged sample ids for reproducibility
    samples_path = base_dir / "dataset_samples.json"
    if samples_path.exists():
        (out_dir / "dataset_samples.json").write_text(samples_path.read_text())
    else:
        # Derive from merged scores
        sample_ids = {}
        for ds in merged_scores["dataset"].unique():
            ds_df = merged_scores[merged_scores["dataset"] == ds]
            human_ids = ds_df[ds_df["label"] == "human"]["sample_id"].drop_duplicates().tolist()
            ai_ids = ds_df[ds_df["label"] == "ai"]["sample_id"].drop_duplicates().tolist()
            sample_ids[ds] = {"human_ids": human_ids, "ai_ids": ai_ids}
        with open(out_dir / "dataset_samples.json", "w") as f:
            json.dump(sample_ids, f, indent=2)

    # Recompute metrics + thresholds
    methods = [m for m in merged_scores["method"].unique().tolist() if m != "human"]
    metrics = _recompute_metrics(
        scores_df=merged_scores,
        detectors=detectors_base,
        methods=methods,
        datasets=datasets,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        out_dir=out_dir,
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "thresholds": _read_json(out_dir / "thresholds.json") or {},
                "config": {
                    "seed": args.seed,
                    "n_bootstrap": args.n_bootstrap,
                    "datasets": datasets,
                    "methods": methods,
                    "detectors": detectors_base,
                },
            },
            f,
            indent=2,
        )

    # Generate figures + tables
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        generate_all_plots(
            detector_metrics=metrics_df,
            quality_metrics=merged_quality if not merged_quality.empty else pd.DataFrame(),
            scores_data=merged_scores,
            output_dir=str(out_dir / "figures"),
        )

        generate_all_tables(
            detector_metrics=metrics,
            quality_metrics=merged_quality.to_dict("records") if not merged_quality.empty else [],
            output_dir=str(out_dir / "tables"),
            format="markdown",
        )
        if args.latex:
            generate_all_tables(
                detector_metrics=metrics,
                quality_metrics=merged_quality.to_dict("records") if not merged_quality.empty else [],
                output_dir=str(out_dir / "tables"),
                format="latex",
            )

    # Multi-FPR table (pipeline output)
    if not args.no_fpr_table:
        import subprocess
        import sys as _sys
        cmd = [
            _sys.executable,
            str(project_root / "scripts" / "make_fpr_table.py"),
            "--run-dir",
            str(out_dir),
            "--n-bootstrap",
            str(args.fpr_bootstrap),
        ]
        if args.fprs:
            cmd += ["--fprs"] + [str(f) for f in args.fprs]
        subprocess.run(cmd, check=True)

    logger.info(f"Merged run saved to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
