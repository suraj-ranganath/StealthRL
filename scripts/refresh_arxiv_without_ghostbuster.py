#!/usr/bin/env python3
"""
Refresh arXiv paper artifacts after removing Ghostbuster from the reported panel.

This is intentionally paper-facing only: it does not mutate the assembled run.
It filters the completed run artifacts down to the four-detector panel used in
the manuscript and rewrites the paper tables/figures accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics import compute_threshold_at_fpr, compute_tpr_at_fpr
from eval.plots import create_quality_table, create_tradeoff_plot, generate_all_plots


KEEP_DETECTORS = ["binoculars", "fast_detectgpt", "mage", "roberta"]
METHOD_ORDER = ["m0", "m1", "m2", "m3", "m4", "m5"]

METHOD_LABELS = {
    "m0": "M0",
    "m1": "M1",
    "m2": "M2 (Ours)",
    "m3": "M3",
    "m4": "M4",
    "m5": "M5",
}

DETECTOR_LABELS = {
    "binoculars": "Binoc.",
    "fast_detectgpt": "FastDGPT",
    "mage": "MAGE",
    "roberta": "RoBERTa",
}

APPENDIX_DETECTOR_LABELS = {
    "binoculars": "Binoculars",
    "fast_detectgpt": r"\shortstack[l]{Fast-\\DetectGPT}",
    "mage": "MAGE",
    "roberta": "RoBERTa",
}

METHOD_LONG_LABELS = {
    "m0": "M0 No Attack",
    "m1": "M1 Simple Para.",
    "m2": "M2 StealthRL (Ours)",
    "m3": r"M3 Adv.\ Para.",
    "m4": "M4 AuthorMist",
    "m5": "M5 Homoglyph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh arXiv artifacts without Ghostbuster")
    parser.add_argument("--run-dir", required=True, help="Assembled eval run directory")
    parser.add_argument("--arxiv-dir", default="arxiv/submission")
    parser.add_argument("--work-dir", default="arxiv/_tmp_no_ghostbuster")
    return parser.parse_args()


def _load_metrics(run_dir: Path) -> pd.DataFrame:
    payload = json.loads((run_dir / "metrics.json").read_text())
    df = pd.DataFrame(payload["metrics"])
    df = df[(df["dataset"] == "mage") & (df["detector"].isin(KEEP_DETECTORS))].copy()
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df["detector"] = pd.Categorical(df["detector"], KEEP_DETECTORS, ordered=True)
    return df.sort_values(["method", "detector"]).reset_index(drop=True)


def _load_quality(run_dir: Path) -> pd.DataFrame:
    quality_path = run_dir / "quality.parquet"
    df = pd.read_parquet(quality_path)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    return df.sort_values(["method", "sample_id"]).reset_index(drop=True)


def _load_scores(run_dir: Path) -> pd.DataFrame:
    scores_path = run_dir / "scores.parquet"
    df = pd.read_parquet(scores_path)
    df = df[df["detector_name"].isin(KEEP_DETECTORS)].copy()
    return df.reset_index(drop=True)


def _png_to_pdf(src_png: Path, dst_pdf: Path) -> None:
    image = Image.open(src_png).convert("RGB")
    image.save(dst_pdf, "PDF", resolution=300.0)


def _fmt(value: float, bold: bool = False) -> str:
    body = f"{value:.3f}"
    return rf"\textbf{{{body}}}" if bold else body


def _best_masks(metrics_df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    masks: dict[tuple[str, str], set[str]] = {}
    for detector in KEEP_DETECTORS:
        det = metrics_df[metrics_df["detector"] == detector]
        if det.empty:
            continue
        masks[(detector, "auroc")] = set(det.loc[det["auroc"] == det["auroc"].min(), "method"])
        masks[(detector, "tpr_at_1fpr")] = set(det.loc[det["tpr_at_1fpr"] == det["tpr_at_1fpr"].min(), "method"])
        masks[(detector, "asr")] = set(det.loc[det["asr"] == det["asr"].max(), "method"])
    by_method = metrics_df.groupby("method").agg(mean_tpr=("tpr_at_1fpr", "mean"), mean_asr=("asr", "mean"))
    masks[("mean", "tpr_at_1fpr")] = set(by_method[by_method["mean_tpr"] == by_method["mean_tpr"].min()].index.tolist())
    masks[("mean", "asr")] = set(by_method[by_method["mean_asr"] == by_method["mean_asr"].max()].index.tolist())
    return masks


def _write_table_main(metrics_df: pd.DataFrame, output_path: Path) -> None:
    best = _best_masks(metrics_df)
    lines = [
        r"\begin{tabular}{lrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Method & Binoc.\ AUC & Binoc.\ TPR & Binoc.\ ASR & FastDGPT AUC & FastDGPT TPR & FastDGPT ASR & MAGE AUC & MAGE TPR & MAGE ASR & RoBERTa AUC & RoBERTa TPR & RoBERTa ASR & Mean TPR & Mean ASR \\",
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        method_df = metrics_df[metrics_df["method"] == method]
        cells = [METHOD_LABELS[method]]
        for detector in KEEP_DETECTORS:
            row = method_df[method_df["detector"] == detector].iloc[0]
            cells.append(_fmt(row["auroc"], method in best[(detector, "auroc")]))
            cells.append(_fmt(row["tpr_at_1fpr"], method in best[(detector, "tpr_at_1fpr")]))
            cells.append(_fmt(row["asr"], method in best[(detector, "asr")]))
        mean_tpr = method_df["tpr_at_1fpr"].mean()
        mean_asr = method_df["asr"].mean()
        cells.append(_fmt(mean_tpr, method in best[("mean", "tpr_at_1fpr")]))
        cells.append(_fmt(mean_asr, method in best[("mean", "asr")]))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n")


def _write_table_main_split(metrics_df: pd.DataFrame, output_path_left: Path, output_path_right: Path) -> None:
    best = _best_masks(metrics_df)

    left_lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Method & Binoc.\ AUC & Binoc.\ TPR & Binoc.\ ASR & FastDGPT AUC & FastDGPT TPR & FastDGPT ASR & MAGE AUC & MAGE TPR & MAGE ASR \\",
        r"\midrule",
    ]
    right_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & RoBERTa AUC & RoBERTa TPR & RoBERTa ASR & Mean TPR & Mean ASR \\",
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        method_df = metrics_df[metrics_df["method"] == method]
        left = [METHOD_LABELS[method]]
        right = [METHOD_LABELS[method]]

        for detector in ["binoculars", "fast_detectgpt", "mage"]:
            row = method_df[method_df["detector"] == detector].iloc[0]
            left.append(_fmt(row["auroc"], method in best[(detector, "auroc")]))
            left.append(_fmt(row["tpr_at_1fpr"], method in best[(detector, "tpr_at_1fpr")]))
            left.append(_fmt(row["asr"], method in best[(detector, "asr")]))

        for detector in ["roberta"]:
            row = method_df[method_df["detector"] == detector].iloc[0]
            right.append(_fmt(row["auroc"], method in best[(detector, "auroc")]))
            right.append(_fmt(row["tpr_at_1fpr"], method in best[(detector, "tpr_at_1fpr")]))
            right.append(_fmt(row["asr"], method in best[(detector, "asr")]))

        mean_tpr = method_df["tpr_at_1fpr"].mean()
        mean_asr = method_df["asr"].mean()
        right.append(_fmt(mean_tpr, method in best[("mean", "tpr_at_1fpr")]))
        right.append(_fmt(mean_asr, method in best[("mean", "asr")]))

        left_lines.append(" & ".join(left) + r" \\")
        right_lines.append(" & ".join(right) + r" \\")

    left_lines.extend([r"\bottomrule", r"\end{tabular}"])
    right_lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path_left.write_text("\n".join(left_lines) + "\n")
    output_path_right.write_text("\n".join(right_lines) + "\n")


def _write_table_per_detector(metrics_df: pd.DataFrame, output_path: Path) -> None:
    best = _best_masks(metrics_df)
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Detector & Method & AUROC [95\% CI] & TPR@1\%FPR [95\% CI] & ASR [95\% CI] \\",
        r"\midrule",
    ]

    for det_index, detector in enumerate(["roberta", "fast_detectgpt", "binoculars", "mage"]):
        det_df = metrics_df[metrics_df["detector"] == detector]
        label = APPENDIX_DETECTOR_LABELS[detector]
        lines.append(rf"\multirow{{6}}{{*}}{{{label}}}")
        for row_index, method in enumerate(METHOD_ORDER):
            row = det_df[det_df["method"] == method].iloc[0]
            prefix = " & " if row_index > 0 else " & "
            auroc = _fmt(row["auroc"], method in best[(detector, "auroc")])
            tpr = _fmt(row["tpr_at_1fpr"], method in best[(detector, "tpr_at_1fpr")])
            asr = _fmt(row["asr"], method in best[(detector, "asr")])
            lines.append(
                prefix
                + METHOD_LONG_LABELS[method]
                + " & "
                + rf"{auroc} [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}]"
                + " & "
                + rf"{tpr} [{row['tpr_at_1fpr_ci_low']:.3f}, {row['tpr_at_1fpr_ci_high']:.3f}]"
                + " & "
                + rf"{asr} [{row['asr_ci_low']:.3f}, {row['asr_ci_high']:.3f}]"
                + r" \\"
            )
        if det_index < 3:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n")


def _write_table_transfer(metrics_df: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        metrics_df.pivot(index="detector", columns="method", values="tpr_at_1fpr")
        .reindex(index=KEEP_DETECTORS, columns=METHOD_ORDER)
        .round(3)
    )
    output_path.write_text(pivot.to_latex())


def _compute_tpr5(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for detector in KEEP_DETECTORS:
        human_scores = scores_df[
            (scores_df["detector_name"] == detector) & (scores_df["label"] == "human")
        ]["detector_score"].to_numpy()
        threshold = compute_threshold_at_fpr(human_scores, target_fpr=0.05, higher_is_ai=True)
        for method in METHOD_ORDER:
            ai_scores = scores_df[
                (scores_df["detector_name"] == detector)
                & (scores_df["label"] == "ai")
                & (scores_df["method"] == method)
            ]["detector_score"].to_numpy()
            tpr = compute_tpr_at_fpr(ai_scores, threshold, higher_is_ai=True)
            rows.append(
                {
                    "detector": detector,
                    "method": method,
                    "threshold_5fpr": threshold,
                    "tpr_at_5fpr": tpr,
                }
            )
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df["detector"] = pd.Categorical(df["detector"], KEEP_DETECTORS, ordered=True)
    return df.sort_values(["method", "detector"]).reset_index(drop=True)


def _write_table_tpr5(tpr5_df: pd.DataFrame, output_path: Path) -> None:
    best = {}
    for detector in KEEP_DETECTORS:
        det = tpr5_df[tpr5_df["detector"] == detector]
        best[detector] = set(det.loc[det["tpr_at_5fpr"] == det["tpr_at_5fpr"].min(), "method"])
    mean_tpr = tpr5_df.groupby("method").agg(mean_tpr_5=("tpr_at_5fpr", "mean"))
    best_mean = set(mean_tpr[mean_tpr["mean_tpr_5"] == mean_tpr["mean_tpr_5"].min()].index.tolist())

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Binoc.\ TPR@5\% & FastDGPT TPR@5\% & MAGE TPR@5\% & RoBERTa TPR@5\% & Mean TPR@5\% \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        method_df = tpr5_df[tpr5_df["method"] == method]
        cells = [METHOD_LABELS[method]]
        for detector in KEEP_DETECTORS:
            row = method_df[method_df["detector"] == detector].iloc[0]
            cells.append(_fmt(row["tpr_at_5fpr"], method in best[detector]))
        cells.append(_fmt(method_df["tpr_at_5fpr"].mean(), method in best_mean))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n")


def _create_auroc_ci_figure(metrics_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font="serif")

    methods = METHOD_ORDER
    detectors = KEEP_DETECTORS
    colors = {
        "m0": "#0072B2",
        "m1": "#E69F00",
        "m2": "#009E73",
        "m3": "#D55E00",
        "m4": "#CC79A7",
        "m5": "#56B4E9",
    }

    x = np.arange(len(detectors))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    for i, method in enumerate(methods):
        method_df = metrics_df[metrics_df["method"] == method]
        vals = []
        errs_low = []
        errs_high = []
        for detector in detectors:
            row = method_df[method_df["detector"] == detector].iloc[0]
            vals.append(row["auroc"])
            errs_low.append(row["auroc"] - row["auroc_ci_low"])
            errs_high.append(row["auroc_ci_high"] - row["auroc"])
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=[errs_low, errs_high],
            capsize=2,
            label=METHOD_LABELS[method],
            color=colors[method],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([DETECTOR_LABELS[d] for d in detectors])
    ax.set_ylabel("AUROC")
    ax.set_title("Per-detector AUROC with 95% bootstrap CIs")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _create_tradeoff_5fpr_figure(
    quality_df: pd.DataFrame,
    tpr5_df: pd.DataFrame,
    output_path: Path,
) -> None:
    tradeoff_5 = tpr5_df.groupby("method").agg(mean_tpr_5=("tpr_at_5fpr", "mean")).reset_index()
    tradeoff_5["mean_asr_5"] = 1.0 - tradeoff_5["mean_tpr_5"]
    quality_agg = quality_df.groupby("method")["sim_e5"].median().reset_index()
    tradeoff_5 = tradeoff_5.merge(quality_agg, on="method", how="left")

    create_tradeoff_plot(
        tradeoff_5,
        y_col="mean_asr_5",
        title="Evasion-Quality Tradeoff (5% FPR)",
        output_path=str(output_path),
        y_label="Mean ASR@5%FPR (higher is better)",
        y_higher_is_better=True,
        reference_y=0.95,
        reference_label="Random detector @5% FPR",
    )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    arxiv_dir = Path(args.arxiv_dir)
    work_dir = Path(args.work_dir)
    figures_dir = arxiv_dir / "figures"
    work_fig_dir = work_dir / "figures"
    work_table_dir = work_dir / "tables"

    work_fig_dir.mkdir(parents=True, exist_ok=True)
    work_table_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _load_metrics(run_dir)
    quality_df = _load_quality(run_dir)
    scores_df = _load_scores(run_dir)
    tpr5_df = _compute_tpr5(scores_df)

    generate_all_plots(
        detector_metrics=metrics_df,
        quality_metrics=quality_df,
        scores_data=scores_df,
        output_dir=str(work_fig_dir),
    )
    _create_auroc_ci_figure(metrics_df, work_fig_dir / "fig_auroc_ci.png")
    _create_tradeoff_5fpr_figure(quality_df, tpr5_df, work_fig_dir / "fig_tradeoff_5fpr.png")

    _write_table_main(metrics_df, work_table_dir / "table_main.tex")
    _write_table_main_split(metrics_df, work_table_dir / "table_main_part1.tex", work_table_dir / "table_main_part2.tex")
    _write_table_per_detector(metrics_df, work_table_dir / "table_per_detector.tex")
    _write_table_transfer(metrics_df, work_table_dir / "table_transfer.tex")
    _write_table_tpr5(tpr5_df, work_table_dir / "table_tpr5.tex")
    create_quality_table(
        quality_df.to_dict(orient="records"),
        output_path=str(work_table_dir / "table_quality.tex"),
        format="latex",
    )

    for name in [
        "fig_method_comparison",
        "fig_heatmap_tpr",
        "fig_score_distributions",
        "fig_tradeoff",
        "fig_quality_likert",
        "fig_auroc_ci",
        "fig_tradeoff_5fpr",
    ]:
        src_png = work_fig_dir / f"{name}.png"
        if not src_png.exists():
            continue
        dst_png = figures_dir / f"{name}.png"
        dst_pdf = figures_dir / f"{name}.pdf"
        dst_png.write_bytes(src_png.read_bytes())
        _png_to_pdf(src_png, dst_pdf)

    for table_name in [
        "table_main.tex",
        "table_main_part1.tex",
        "table_main_part2.tex",
        "table_per_detector.tex",
        "table_quality.tex",
        "table_transfer.tex",
        "table_tpr5.tex",
    ]:
        src = work_table_dir / table_name
        dst = arxiv_dir / table_name
        dst.write_text(src.read_text())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
