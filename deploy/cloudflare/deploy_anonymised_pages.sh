#!/usr/bin/env bash
set -euo pipefail

# Deploy the anonymous review-safe frontend to:
#   https://stealthrl-anonymised.pages.dev
#
# Required:
#   CLOUDFLARE_API_TOKEN
#   STEALTHRL_API_BASE_URL

export CLOUDFLARE_PAGES_PROJECT="${CLOUDFLARE_PAGES_PROJECT:-stealthrl-anonymised}"
export STATIC_VARIANT="${STATIC_VARIANT:-anonymous}"
export STATIC_DIST="${STATIC_DIST:-deploy/cloudflare/static_dist_anonymised}"

exec "$(dirname "$0")/deploy_pages.sh"
