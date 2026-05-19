# StealthRL Demo Website

This is a lightweight FastAPI demo for StealthRL. It serves a static single-page UI and exposes a small JSON API for paraphrasing text.

The demo is designed to be cheap by default:

- `mock` backend is zero-cost and deterministic for UI/local testing.
- `hf` backend lazily loads the released Hugging Face PEFT adapter for real local inference.
- `tinker` backend lazily calls a StealthRL sampler through Tinker when configured.
- unauthenticated users are limited to 20 public responses per day by default.
- API-key users can bypass the public quota or receive their own daily quota.

## Run Locally

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
pip install -r demo/requirements.txt
uvicorn demo.stealthrl_demo.app:app --reload --port 8080
```

Then open `http://localhost:8080`.

## Configure Access

Copy `demo/.env.example` to `demo/.env` or export environment variables.

```bash
export STEALTHRL_DEMO_PUBLIC_DAILY_LIMIT=20
export STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE=ip
export STEALTHRL_DEMO_API_KEYS='{"stealth-demo-lab":{"label":"lab","daily_limit":500}}'
```

Clients can pass API keys as either:

```http
Authorization: Bearer stealth-demo-lab
```

or:

```http
X-StealthRL-API-Key: stealth-demo-lab
```

The quota database stores only daily counters keyed by hashed identifiers; it does not store submitted text.

## Use Real StealthRL Inference

The default `mock` backend is intentionally cost-free. To run the released Hugging Face adapter locally:

```bash
export CUDA_VISIBLE_DEVICES=4
export STEALTHRL_DEMO_INFERENCE_BACKEND=hf
export STEALTHRL_DEMO_HF_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
export STEALTHRL_DEMO_HF_ADAPTER_MODEL=suraj-ranganath/StealthRL
export STEALTHRL_DEMO_REQUEST_TIMEOUT_S=240
uvicorn demo.stealthrl_demo.app:app --host 0.0.0.0 --port 8080
```

The Hugging Face backend uses the released PEFT adapter and chunks multi-sentence inputs before generation so the demo preserves paragraph coverage instead of summarizing to a single sentence.

To call a Tinker-hosted sampler instead:

```bash
export STEALTHRL_DEMO_INFERENCE_BACKEND=tinker
export STEALTHRL_DEMO_CHECKPOINT_JSON=/path/to/m2_checkpoint.json
export TINKER_API_KEY=...
uvicorn demo.stealthrl_demo.app:app --host 0.0.0.0 --port 8080
```

The checkpoint JSON must contain the `checkpoints.sampler_weights` field used by `eval.methods.stealthrl.StealthRLTinker`.

## Docker

```bash
docker build -f demo/Dockerfile -t stealthrl-demo .
docker run --rm -p 8080:8080 --env-file demo/.env stealthrl-demo
```

## AWS Deployment Notes

For a judicious first deployment, keep the FastAPI service small and choose the inference path explicitly:

- AWS App Runner or ECS Fargate with `0.5-1 vCPU` and `1-2 GB` RAM is enough for the FastAPI frontend/backend when inference is remote or mocked.
- For fully local real inference, use a GPU host with enough memory for `Qwen/Qwen3-4B-Instruct-2507` plus the PEFT adapter, then set `STEALTHRL_DEMO_INFERENCE_BACKEND=hf`.
- Mount persistent storage only for `STEALTHRL_DEMO_DB_PATH` if you need quota counters to survive container replacement; otherwise use a small managed database later.
- Store `STEALTHRL_DEMO_API_KEYS` and any provider credentials in AWS Secrets Manager or the service secret-env mechanism.
- Set `STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE=global` if you want a hard public cost cap across all unauthenticated users instead of a per-IP quota.

If you later add a hosted remote sampler, keep the web tier CPU-only and call that sampler from the backend. The current real local path is the Hugging Face PEFT backend.

## API

`POST /api/paraphrase`

```json
{
  "text": "Paste AI-generated text here...",
  "temperature": 1.0,
  "top_p": 0.9
}
```

Response:

```json
{
  "request_id": "...",
  "input_text": "...",
  "output_text": "...",
  "backend": "mock",
  "metrics": {
    "input_words": 50,
    "output_words": 49,
    "word_delta_pct": -2.0,
    "char_edit_rate": 0.41
  },
  "quota": {
    "authenticated": false,
    "limit": 20,
    "remaining": 19
  }
}
```
