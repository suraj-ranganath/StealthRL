# StealthRL Demo Website

This is a lightweight FastAPI demo for StealthRL. It serves a static single-page UI and exposes a small JSON API for paraphrasing text.

The demo is designed to be cheap by default:

- `mock` backend is zero-cost and deterministic for UI/local testing.
- `tinker` backend lazily calls the released StealthRL sampler through Tinker when configured.
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

The default `mock` backend is intentionally cost-free. To call the actual StealthRL sampler:

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

For a judicious first deployment, use a small CPU container service and Tinker-backed inference rather than renting a GPU instance:

- AWS App Runner or ECS Fargate with `0.5-1 vCPU` and `1-2 GB` RAM is enough for the FastAPI frontend/backend.
- Mount persistent storage only for `STEALTHRL_DEMO_DB_PATH` if you need quota counters to survive container replacement; otherwise use a small managed database later.
- Store `TINKER_API_KEY`, `STEALTHRL_DEMO_API_KEYS`, and checkpoint path/JSON in AWS Secrets Manager or the service secret-env mechanism.
- Set `STEALTHRL_DEMO_PUBLIC_QUOTA_SCOPE=global` if you want a hard public cost cap across all unauthenticated users instead of a per-IP quota.

If you later want fully local inference on AWS, use a GPU instance only after measuring Tinker latency/cost. A CPU web service plus remote sampler is the cheapest credible demo architecture.

## API

`POST /api/paraphrase`

```json
{
  "text": "Paste AI-generated text here...",
  "temperature": 0.9,
  "top_p": 0.95
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
