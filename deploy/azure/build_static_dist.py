#!/usr/bin/env python3
"""Build a static artifact for Azure Static Web Apps or Cloudflare Pages."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


def _replace_required(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise SystemExit(f"Could not find expected text in {path}: {old[:80]!r}")
    return text.replace(old, new)


def _apply_anonymous_variant(out: Path) -> None:
    """Strip public project/author identifiers from the static review demo."""

    index_path = out / "index.html"
    index = index_path.read_text(encoding="utf-8")
    index = _replace_required(
        index,
        "<title>StealthRL Demo</title>",
        "<title>StealthRL Anonymous Demo</title>",
        index_path,
    )
    index = _replace_required(
        index,
        '<p class="kicker">RESEARCH DEMO</p>',
        '<p class="kicker">ANONYMOUS RESEARCH DEMO</p>',
        index_path,
    )
    index = _replace_required(
        index,
        """        <nav class="link-row" aria-label="Project links">
          <a href="https://arxiv.org/abs/2602.08934" target="_blank" rel="noreferrer">Paper</a>
          <a href="https://huggingface.co/suraj-ranganath/StealthRL" target="_blank" rel="noreferrer">Model</a>
          <a href="https://github.com/suraj-ranganath/StealthRL" target="_blank" rel="noreferrer">GitHub</a>
        </nav>""",
        """        <nav class="link-row" aria-label="Submission demo note">
          <span>Anonymous review demo</span>
        </nav>""",
        index_path,
    )
    index = _replace_required(
        index,
        """        <span>
          <a href="https://www.linkedin.com/in/suraj-ranganath/" target="_blank" rel="noreferrer">Suraj Ranganath</a>
          &amp;
          <a href="https://www.linkedin.com/in/atharv-ramesh/" target="_blank" rel="noreferrer">Atharv Ramesh</a>
        </span>""",
        "        <span>Anonymous submission demo</span>",
        index_path,
    )
    index_path.write_text(index, encoding="utf-8")

    privacy_path = out / "privacy.html"
    privacy = privacy_path.read_text(encoding="utf-8")
    privacy = _replace_required(
        privacy,
        "<title>Privacy information · StealthRL Demo</title>",
        "<title>Privacy information · StealthRL Anonymous Demo</title>",
        privacy_path,
    )
    privacy = _replace_required(
        privacy,
        """          This page explains what the StealthRL Demo processes when you submit text to the
          research preview.""",
        """          This page explains what the anonymous StealthRL review demo processes when you
          submit text to the research preview.""",
        privacy_path,
    )
    privacy = _replace_required(
        privacy,
        """        <h2>Who runs this demo</h2>
        <p>
          The StealthRL Demo is operated by Suraj Ranganath and Atharv Ramesh for research
          demonstration and evaluation purposes.
        </p>""",
        """        <h2>Who runs this demo</h2>
        <p>
          The StealthRL anonymous review demo is operated by the authors of an anonymous
          manuscript submission for research demonstration and evaluation purposes.
        </p>""",
        privacy_path,
    )
    privacy = _replace_required(
        privacy,
        """        <p>
          For privacy or demo-operation questions, contact the project maintainers through the
          public project links on the demo page.
        </p>""",
        """        <p>
          For privacy or demo-operation questions during review, use the submission discussion
          channel or the anonymous contact mechanism provided by the venue.
        </p>""",
        privacy_path,
    )
    privacy = _replace_required(
        privacy,
        """        <span>
          <a href="https://www.linkedin.com/in/suraj-ranganath/" target="_blank" rel="noreferrer">Suraj Ranganath</a>
          &amp;
          <a href="https://www.linkedin.com/in/atharv-ramesh/" target="_blank" rel="noreferrer">Atharv Ramesh</a>
        </span>""",
        "        <span>Anonymous submission demo</span>",
        privacy_path,
    )
    privacy_path.write_text(privacy, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", required=True, help="External Container Apps API origin")
    parser.add_argument("--source", default="demo/static", help="Static source directory")
    parser.add_argument("--out", default="deploy/azure/static_dist", help="Output directory")
    parser.add_argument(
        "--variant",
        choices=("public", "anonymous"),
        default="public",
        help="Static-site variant to build",
    )
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, out)
    static_mount = out / "static"
    static_mount.mkdir(exist_ok=True)
    for asset_name in ("app.js", "styles.css"):
        shutil.copy2(source / asset_name, static_mount / asset_name)

    index_path = out / "index.html"
    index = index_path.read_text(encoding="utf-8")
    api_base_url = html.escape(args.api_base_url.rstrip("/"), quote=True)
    marker = '<meta name="stealthrl-api-base-url" content="" />'
    replacement = f'<meta name="stealthrl-api-base-url" content="{api_base_url}" />'
    if marker not in index:
        raise SystemExit(f"Could not find API base URL meta tag in {index_path}")
    index_path.write_text(index.replace(marker, replacement), encoding="utf-8")

    if args.variant == "anonymous":
        _apply_anonymous_variant(out)

    (out / "staticwebapp.config.json").write_text(
        json.dumps(
            {
                "routes": [
                    {"route": "/privacy", "rewrite": "/privacy.html"},
                ],
                "navigationFallback": {"rewrite": "/index.html"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
