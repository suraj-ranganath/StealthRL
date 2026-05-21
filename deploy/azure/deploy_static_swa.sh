#!/usr/bin/env bash
set -euo pipefail

export PATH="$PWD/.venv-azure/bin:$PWD/.azure-tools/node_modules/.bin:$PATH"

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI is required. Install it and run az login first." >&2
  exit 1
fi

if ! command -v swa >/dev/null 2>&1; then
  echo "Azure Static Web Apps CLI is required. Install with: npm install -g @azure/static-web-apps-cli" >&2
  exit 1
fi

az account show >/dev/null

ensure_provider() {
  local namespace="$1"
  local state
  state="$(az provider show --namespace "$namespace" --query registrationState -o tsv 2>/dev/null || true)"
  if [[ "$state" != "Registered" ]]; then
    az provider register --namespace "$namespace" --output none
  fi

  for _ in {1..60}; do
    state="$(az provider show --namespace "$namespace" --query registrationState -o tsv 2>/dev/null || true)"
    if [[ "$state" == "Registered" ]]; then
      return 0
    fi
    sleep 10
  done

  echo "Timed out waiting for provider $namespace to register; last state: ${state:-unknown}" >&2
  return 1
}

AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-stealthrl-demo-rg}"
AZURE_STATIC_LOCATION="${AZURE_STATIC_LOCATION:-eastus2}"
AZURE_STATIC_WEBAPP_NAME="${AZURE_STATIC_WEBAPP_NAME:-stealthrl-demo-web}"
STEALTHRL_API_BASE_URL="${STEALTHRL_API_BASE_URL:?Set STEALTHRL_API_BASE_URL, e.g. https://<container-app-fqdn>}"
STATIC_DIST="${STATIC_DIST:-deploy/azure/static_dist}"

ensure_provider Microsoft.Web

az staticwebapp create \
  --name "$AZURE_STATIC_WEBAPP_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_STATIC_LOCATION" \
  --sku Free \
  --output table

python deploy/azure/build_static_dist.py \
  --api-base-url "$STEALTHRL_API_BASE_URL" \
  --out "$STATIC_DIST"

DEPLOYMENT_TOKEN="$(az staticwebapp secrets list \
  --name "$AZURE_STATIC_WEBAPP_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query properties.apiKey \
  -o tsv)"

swa deploy "$STATIC_DIST" \
  --deployment-token "$DEPLOYMENT_TOKEN" \
  --env production

DEFAULT_HOSTNAME="$(az staticwebapp show \
  --name "$AZURE_STATIC_WEBAPP_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query defaultHostname \
  -o tsv)"

echo "Static site URL: https://$DEFAULT_HOSTNAME"
echo "For stricter CORS, rerun deploy_aca_gpu.sh with:"
echo "  export STEALTHRL_DEMO_CORS_ORIGINS=https://$DEFAULT_HOSTNAME"
