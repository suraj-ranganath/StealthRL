#!/usr/bin/env bash
set -euo pipefail

export PATH="$PWD/.venv-azure/bin:$PWD/.azure-tools/node_modules/.bin:$PATH"

if ! command -v az >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Azure CLI is required but was not found.
Install it, or run from the repo root after installing local tooling, then run:
  az login --use-device-code
  az account set --subscription <SUBSCRIPTION_ID_OR_NAME>
EOF
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
AZURE_LOCATION="${AZURE_LOCATION:-eastus}"
AZURE_ACR_NAME="${AZURE_ACR_NAME:?Set AZURE_ACR_NAME to a globally unique lowercase registry name, e.g. stealthrldemo123}"
AZURE_ACR_SKU="${AZURE_ACR_SKU:-Basic}"
AZURE_CONTAINERAPP_ENV="${AZURE_CONTAINERAPP_ENV:-stealthrl-demo-env}"
AZURE_CONTAINERAPP_NAME="${AZURE_CONTAINERAPP_NAME:-stealthrl-demo-api}"
AZURE_WORKLOAD_PROFILE_NAME="${AZURE_WORKLOAD_PROFILE_NAME:-gpu-t4}"
AZURE_IMAGE_NAME="${AZURE_IMAGE_NAME:-stealthrl-demo-gpu}"
AZURE_IMAGE_TAG="${AZURE_IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
STEALTHRL_DEMO_CORS_ORIGINS="${STEALTHRL_DEMO_CORS_ORIGINS:-*}"
STEALTHRL_DEMO_PUBLIC_DAILY_LIMIT="${STEALTHRL_DEMO_PUBLIC_DAILY_LIMIT:-20}"
STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE="${STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE:-ip}"
STEALTHRL_DEMO_MAX_CHARS="${STEALTHRL_DEMO_MAX_CHARS:-5000}"
STEALTHRL_DEMO_API_KEYS="${STEALTHRL_DEMO_API_KEYS:-}"

ensure_provider Microsoft.App
ensure_provider Microsoft.OperationalInsights
ensure_provider Microsoft.ContainerRegistry

az group create \
  --name "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION" \
  --output table

az acr create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$AZURE_ACR_NAME" \
  --sku "$AZURE_ACR_SKU" \
  --output table

az acr update --name "$AZURE_ACR_NAME" --admin-enabled true --output none

IMAGE="$AZURE_ACR_NAME.azurecr.io/$AZURE_IMAGE_NAME:$AZURE_IMAGE_TAG"
az acr build \
  --registry "$AZURE_ACR_NAME" \
  --image "$AZURE_IMAGE_NAME:$AZURE_IMAGE_TAG" \
  --file demo/Dockerfile.azure-gpu \
  .

if ! az containerapp env show \
  --name "$AZURE_CONTAINERAPP_ENV" \
  --resource-group "$AZURE_RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp env create \
    --name "$AZURE_CONTAINERAPP_ENV" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --location "$AZURE_LOCATION" \
    --output table
fi

if ! az containerapp env workload-profile show \
  --name "$AZURE_CONTAINERAPP_ENV" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --workload-profile-name "$AZURE_WORKLOAD_PROFILE_NAME" >/dev/null 2>&1; then
  az containerapp env workload-profile add \
    --name "$AZURE_CONTAINERAPP_ENV" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --workload-profile-name "$AZURE_WORKLOAD_PROFILE_NAME" \
    --workload-profile-type Consumption-GPU-NC8as-T4
fi

ACR_USERNAME="$(az acr credential show --name "$AZURE_ACR_NAME" --query username -o tsv)"
ACR_PASSWORD="$(az acr credential show --name "$AZURE_ACR_NAME" --query 'passwords[0].value' -o tsv)"

ENV_VARS=(
  "STEALTHRL_DEMO_INFERENCE_BACKEND=hf"
  "STEALTHRL_DEMO_HF_DTYPE=float16"
  "STEALTHRL_DEMO_HF_DEVICE_MAP=auto"
  "STEALTHRL_DEMO_REQUEST_TIMEOUT_S=240"
  "STEALTHRL_DEMO_PUBLIC_DAILY_LIMIT=$STEALTHRL_DEMO_PUBLIC_DAILY_LIMIT"
  "STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE=$STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE"
  "STEALTHRL_DEMO_MAX_CHARS=$STEALTHRL_DEMO_MAX_CHARS"
  "STEALTHRL_DEMO_CORS_ORIGINS=$STEALTHRL_DEMO_CORS_ORIGINS"
)

if [[ -n "$STEALTHRL_DEMO_API_KEYS" ]]; then
  ENV_VARS+=("STEALTHRL_DEMO_API_KEYS=$STEALTHRL_DEMO_API_KEYS")
fi

if az containerapp show \
  --name "$AZURE_CONTAINERAPP_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp update \
    --name "$AZURE_CONTAINERAPP_NAME" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --image "$IMAGE" \
    --set-env-vars "${ENV_VARS[@]}" \
    --min-replicas 0 \
    --max-replicas 1 \
    --output table
else
  az containerapp create \
    --name "$AZURE_CONTAINERAPP_NAME" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --environment "$AZURE_CONTAINERAPP_ENV" \
    --image "$IMAGE" \
    --registry-server "$AZURE_ACR_NAME.azurecr.io" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --target-port 8080 \
    --ingress external \
    --cpu 8.0 \
    --memory 56.0Gi \
    --workload-profile-name "$AZURE_WORKLOAD_PROFILE_NAME" \
    --min-replicas 0 \
    --max-replicas 1 \
    --env-vars "${ENV_VARS[@]}" \
    --output table
fi

FQDN="$(az containerapp show \
  --name "$AZURE_CONTAINERAPP_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  -o tsv)"

echo "API URL: https://$FQDN"
echo "Use this as --api-base-url for deploy_static_swa.sh."
