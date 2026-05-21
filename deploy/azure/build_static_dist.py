#!/usr/bin/env python3
"""Build a static artifact for Azure Static Web Apps."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", required=True, help="External Container Apps API origin")
    parser.add_argument("--source", default="demo/static", help="Static source directory")
    parser.add_argument("--out", default="deploy/azure/static_dist", help="Output directory")
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
