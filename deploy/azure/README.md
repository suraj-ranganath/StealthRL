# Azure Deployment

This deploys the StealthRL demo as:

- Azure Static Web Apps Free for the static UI.
- Azure Container Apps serverless GPU for the FastAPI inference API.
- Azure Container Registry for the GPU image.

The GPU app uses `Consumption-GPU-NC8as-T4`, `minReplicas=0`, and `maxReplicas=1`.

## Prerequisites

Install and authenticate:

```bash
export PATH="$PWD/.venv-azure/bin:$PWD/.azure-tools/node_modules/.bin:$PATH"
az login --use-device-code
az account set --subscription <SUBSCRIPTION_ID_OR_NAME>
```

On this server, Azure CLI is installed locally in `.venv-azure/` and the Static Web Apps CLI is installed locally in `.azure-tools/`. If you are setting up a new machine, install equivalent tools with Azure's official installer and `npm install -g @azure/static-web-apps-cli`, or recreate the local installs.

You also need serverless T4 GPU quota in the target region. Check support with:

```bash
az containerapp env workload-profile list-supported --location eastus --output table
```

Look for `Consumption-GPU-NC8as-T4`.

## 1. Deploy the GPU API

Choose a globally unique ACR name. It must be lowercase alphanumeric.

```bash
export AZURE_RESOURCE_GROUP=stealthrl-demo-rg
export AZURE_LOCATION=eastus
export AZURE_ACR_NAME=<uniqueacrname>
export AZURE_CONTAINERAPP_ENV=stealthrl-demo-env
export AZURE_CONTAINERAPP_NAME=stealthrl-demo-api
export STEALTHRL_DEMO_PUBLIC_DAILY_LIMIT=20
export STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE=ip
export STEALTHRL_DEMO_CORS_ORIGINS='*'

# Optional API keys. JSON object gives per-key labels/limits.
export STEALTHRL_DEMO_API_KEYS='{"stealth-demo-lab":{"label":"lab","daily_limit":500}}'

bash deploy/azure/deploy_aca_gpu.sh
```

The script prints:

```text
API URL: https://<container-app-fqdn>
```

The first build is slow because the Docker image downloads and bakes in:

- `Qwen/Qwen3-4B-Instruct-2507`
- `suraj-ranganath/StealthRL`

This intentionally makes the image large, but avoids redownloading weights from Hugging Face on every scale-to-zero cold start.

## 2. Deploy the Static Web App

```bash
export STEALTHRL_API_BASE_URL=https://<container-app-fqdn>
export AZURE_STATIC_WEBAPP_NAME=stealthrl-demo-web
export AZURE_STATIC_LOCATION=eastus2

bash deploy/azure/deploy_static_swa.sh
```

The script prints:

```text
Static site URL: https://<static-web-app-hostname>
```

After the Static Web App exists, tighten CORS on the API:

```bash
export STEALTHRL_DEMO_CORS_ORIGINS=https://<static-web-app-hostname>
bash deploy/azure/deploy_aca_gpu.sh
```

If you add a custom domain later, include both origins:

```bash
export STEALTHRL_DEMO_CORS_ORIGINS=https://<static-web-app-hostname>,https://demo.yourdomain.com
```

## Cost Controls

- Keep `minReplicas=0`.
- Keep `maxReplicas=1` until traffic/cost is understood.
- Use `AZURE_ACR_SKU=Basic` for the cheapest registry.
- Consider `AZURE_ACR_SKU=Premium` only if cold starts are too slow and you want ACR artifact streaming.

## Current MVP Limitation

The first deployment is synchronous: the browser calls `/api/paraphrase` directly. Azure Container Apps has a 240-second ingress timeout, so if cold starts plus model loading exceed that, the next step is to add a queue-backed async API.

The quota SQLite file is stored inside the container filesystem by default. For strict public quota enforcement across scale-to-zero restarts, mount Azure Files and set:

```bash
export STEALTHRL_DEMO_DB_PATH=/mnt/stealthrl-data/demo_usage.sqlite3
```

This is not enabled by the MVP script yet.
