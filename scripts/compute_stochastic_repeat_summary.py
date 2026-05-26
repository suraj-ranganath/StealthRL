#!/usr/bin/env python3
"""Summarize stochastic repeat detection metrics for StealthRL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DETECTOR_ORDER = ["binoculars", "fast_detectgpt", "mage", "roberta"]
METHOD_ORDER = ["m2", "m2_s1", "m2_s2"]
RUN_LABELS = {
    "m2": "Original M2",
    "m2_s1": "Seed 4201",
    "m2_s2": "Seed 4202",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-scores", required=True, help="Full-run assembled scores.parquet")
    parser.add_argument("--thresholds", required=True, help="Full-run thresholds.json calibrated on human samples")
    parser.add_argument("--repeat-score-dir", required=True, help="Directory with repeat detector parquet files")
    parser.add_argument("--out-dir", required=True, help="Output directory for summary files")
    parser.add_argument("--detectors", nargs="+", default=DETECTOR_ORDER)
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    return parser.parse_args()


def load_repeat_scores(score_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(score_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No parquet score files found under {score_dir}")
    return pd.concat(frames, ignore_index=True)


def compute_metrics(
    original_scores: pd.DataFrame,
    repeat_scores: pd.DataFrame,
    thresholds: dict[str, float],
    detectors: list[str],
    methods: list[str],
) -> pd.DataFrame:
    records = []
    original_scores = original_scores.rename(columns={"detector_name": "detector"})
    repeat_scores = repeat_scores.rename(columns={"detector_name": "detector"})

    for detector in detectors:
        human = original_scores[
            (original_scores["detector"] == detector) & (original_scores["method"] == "human")
        ]["detector_score"].to_numpy()
        if len(human) == 0:
            raise RuntimeError(f"Missing original human scores for {detector}")
        threshold = float(thresholds[detector])

        for method in methods:
            source = original_scores if method == "m2" else repeat_scores
            ai = source[
                (source["detector"] == detector) & (source["method"] == method)
            ]["detector_score"].to_numpy()
            if len(ai) == 0:
                raise RuntimeError(f"Missing AI scores for {detector}/{method}")
            y_true = np.concatenate([np.zeros(len(human)), np.ones(len(ai))])
            y_score = np.concatenate([human, ai])
            tpr = float(np.mean(ai >= threshold))
            records.append(
                {
                    "run": method,
                    "run_label": RUN_LABELS.get(method, method),
                    "detector": detector,
                    "auroc": float(roc_auc_score(y_true, y_score)),
                    "tpr_at_1fpr": tpr,
                    "asr": 1.0 - tpr,
                    "threshold_1fpr": threshold,
                    "n_human": int(len(human)),
                    "n_ai": int(len(ai)),
                }
            )
    return pd.DataFrame(records)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def make_latex_table(metrics: pd.DataFrame) -> str:
    rows = []
    agg = metrics.groupby(["run", "run_label"], sort=False).agg(
        mean_auroc=("auroc", "mean"),
        mean_tpr=("tpr_at_1fpr", "mean"),
        mean_asr=("asr", "mean"),
    )
    tpr_pivot = metrics.pivot(index="run", columns="detector", values="tpr_at_1fpr")
    for run in METHOD_ORDER:
        if run not in tpr_pivot.index:
            continue
        label = RUN_LABELS.get(run, run)
        a = agg.loc[(run, label)]
        rows.append(
            [
                label,
                fmt(a["mean_auroc"]),
                fmt(a["mean_tpr"]),
                fmt(a["mean_asr"]),
                fmt(tpr_pivot.loc[run, "binoculars"]),
                fmt(tpr_pivot.loc[run, "fast_detectgpt"]),
                fmt(tpr_pivot.loc[run, "mage"]),
                fmt(tpr_pivot.loc[run, "roberta"]),
            ]
        )

    draw_agg = metrics.groupby("run").agg(
        mean_auroc=("auroc", "mean"),
        mean_tpr=("tpr_at_1fpr", "mean"),
        mean_asr=("asr", "mean"),
    ).reindex(METHOD_ORDER)
    tpr_draws = tpr_pivot.reindex(METHOD_ORDER)
    mean_values = [
        f"{draw_agg['mean_auroc'].mean():.3f} $\\pm$ {draw_agg['mean_auroc'].std(ddof=1):.3f}",
        f"{draw_agg['mean_tpr'].mean():.3f} $\\pm$ {draw_agg['mean_tpr'].std(ddof=1):.3f}",
        f"{draw_agg['mean_asr'].mean():.3f} $\\pm$ {draw_agg['mean_asr'].std(ddof=1):.3f}",
        f"{tpr_draws['binoculars'].mean():.3f} $\\pm$ {tpr_draws['binoculars'].std(ddof=1):.3f}",
        f"{tpr_draws['fast_detectgpt'].mean():.3f} $\\pm$ {tpr_draws['fast_detectgpt'].std(ddof=1):.3f}",
        f"{tpr_draws['mage'].mean():.3f} $\\pm$ {tpr_draws['mage'].std(ddof=1):.3f}",
        f"{tpr_draws['roberta'].mean():.3f} $\\pm$ {tpr_draws['roberta'].std(ddof=1):.3f}",
    ]

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Run & Mean AUROC & Mean TPR & Mean ASR & Binoc. & Fast-DGPT & MAGE & RoBERTa \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\midrule",
            "Mean $\\pm$ SD & " + " & ".join(mean_values) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_scores = pd.read_parquet(args.original_scores)
    repeat_scores = load_repeat_scores(Path(args.repeat_score_dir))
    thresholds = json.loads(Path(args.thresholds).read_text())

    metrics = compute_metrics(
        original_scores=original_scores,
        repeat_scores=repeat_scores,
        thresholds=thresholds,
        detectors=args.detectors,
        methods=args.methods,
    )
    metrics.to_csv(out_dir / "stochastic_repeat_metrics.csv", index=False)
    metrics.to_parquet(out_dir / "stochastic_repeat_metrics.parquet")

    summary = metrics.groupby("run").agg(
        mean_auroc=("auroc", "mean"),
        mean_tpr_at_1fpr=("tpr_at_1fpr", "mean"),
        mean_asr=("asr", "mean"),
    ).reindex(args.methods)
    summary.to_csv(out_dir / "stochastic_repeat_summary.csv")
    (out_dir / "table_stochastic_repeats.tex").write_text(make_latex_table(metrics))

    payload = {
        "per_detector": metrics.to_dict(orient="records"),
        "summary": summary.reset_index().to_dict(orient="records"),
        "aggregate_across_runs": {
            "mean_auroc_mean": float(summary["mean_auroc"].mean()),
            "mean_auroc_sd": float(summary["mean_auroc"].std(ddof=1)),
            "mean_tpr_at_1fpr_mean": float(summary["mean_tpr_at_1fpr"].mean()),
            "mean_tpr_at_1fpr_sd": float(summary["mean_tpr_at_1fpr"].std(ddof=1)),
            "mean_asr_mean": float(summary["mean_asr"].mean()),
            "mean_asr_sd": float(summary["mean_asr"].std(ddof=1)),
        },
    }
    (out_dir / "stochastic_repeat_summary.json").write_text(json.dumps(payload, indent=2))
    print(summary)
    print(payload["aggregate_across_runs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
