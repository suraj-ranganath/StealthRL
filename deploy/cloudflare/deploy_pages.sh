#!/usr/bin/env bash
set -euo pipefail

export PATH="$PWD/.cloudflare-tools/node_modules/.bin:$PATH"

if ! command -v wrangler >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Wrangler is required but was not found.
Install it locally from the repo root with:
  npm install --prefix .cloudflare-tools wrangler
EOF
  exit 1
fi

CLOUDFLARE_PAGES_PROJECT="${CLOUDFLARE_PAGES_PROJECT:-stealthrl}"
CLOUDFLARE_PAGES_BRANCH="${CLOUDFLARE_PAGES_BRANCH:-$(git branch --show-current 2>/dev/null || echo demo-website)}"
STEALTHRL_API_BASE_URL="${STEALTHRL_API_BASE_URL:?Set STEALTHRL_API_BASE_URL, e.g. https://<container-app-fqdn>}"
STATIC_DIST="${STATIC_DIST:-deploy/cloudflare/static_dist}"

python deploy/azure/build_static_dist.py \
  --api-base-url "$STEALTHRL_API_BASE_URL" \
  --out "$STATIC_DIST"

mkdir -p "$STATIC_DIST/privacy"
cp "$STATIC_DIST/privacy.html" "$STATIC_DIST/privacy/index.html"

cat > "$STATIC_DIST/_redirects" <<'EOF'
/privacy  /privacy/  308
EOF

if ! create_output="$(wrangler pages project create "$CLOUDFLARE_PAGES_PROJECT" \
  --production-branch "$CLOUDFLARE_PAGES_BRANCH" 2>&1)"; then
  if ! grep -Eiq 'already exists|project with this name already exists|project.*exists' <<<"$create_output"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
fi

wrangler pages deploy "$STATIC_DIST" \
  --project-name "$CLOUDFLARE_PAGES_PROJECT" \
  --branch "$CLOUDFLARE_PAGES_BRANCH" \
  --commit-dirty=true

echo "Cloudflare Pages URL: https://$CLOUDFLARE_PAGES_PROJECT.pages.dev"
