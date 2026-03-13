#!/usr/bin/env python3
"""
Copy finalized evaluation artifacts into the paper directories.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync eval tables/figures into paper directories")
    parser.add_argument("--run-dir", required=True, help="Finalized eval run directory")
    parser.add_argument("--arxiv-dir", default="arxiv/submission")
    parser.add_argument("--iclr-dir", default="iclr2026")
    return parser.parse_args()


def _copy_table(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_png_and_pdf(src_png: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_png = dst_dir / src_png.name
    shutil.copy2(src_png, dst_png)
    dst_pdf = dst_dir / f"{src_png.stem}.pdf"
    image = Image.open(src_png).convert("RGB")
    image.save(dst_pdf, "PDF", resolution=300.0)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"

    arxiv_dir = Path(args.arxiv_dir)
    iclr_dir = Path(args.iclr_dir)

    _copy_table(tables_dir / "table_main_mage.tex", arxiv_dir / "table_main.tex")
    _copy_table(tables_dir / "table_quality.tex", arxiv_dir / "table_quality.tex")
    _copy_table(tables_dir / "table_main_mage.tex", iclr_dir / "table_main.tex")

    figure_names = [
        "fig_method_comparison.png",
        "fig_tradeoff.png",
        "fig_quality_likert.png",
        "fig_score_distributions.png",
        "fig_heatmap_tpr.png",
    ]
    for figure_name in figure_names:
        src = figures_dir / figure_name
        if not src.exists():
            continue
        _copy_png_and_pdf(src, arxiv_dir / "figures")
        _copy_png_and_pdf(src, iclr_dir / "figures")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
