# Operations

Running, configuring, deploying and monitoring TheraGraph.

## Local development

### Without Docker

```bash
make setup            # uv venv + backend (editable) + npm install
make migrate          # create the schema
make seed-smoke       # build the corpus from committed fixtures (no network)
make train            # train and register the models
make api              # API on :8000 with auto-reload
make frontend-dev     # UI on :5173, proxying /api to :8000
```

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), Node 20+, and PostgreSQL if you
want the production datastore. Without PostgreSQL, point `DATABASE_URL` at SQLite —
everything works, minus trigram search:

```bash
DATABASE_URL=sqlite+aiosqlite:///./theragraph.sqlite3 make migrate
```

### With Docker

```bash
cp .env.example .env
make docker-up        # postgres, redis, minio, migrate, api, 2 workers, frontend
make docker-logs      # tail api + worker
make docker-down      # stop, keep volumes
```

| Service | Host port | Notes |
|---|---|---|
| frontend | `5173` | nginx serving the built SPA, proxying `/api` |
| api | `8000` | FastAPI; `/docs` for Swagger, `/healthz`, `/readyz` |
| postgres | `55432` | non-default, so it cannot collide with a local PostgreSQL |
| redis | `63790` | non-default for the same reason |
| minio | `9000` / `9001` | S3 API and console |

`migrate` is a one-shot service: it runs `alembic upgrade head` and exits, and `api` and
`worker` wait for it to complete successfully. Schema migration is never a race between
two starting containers.

## Configuration

Every setting is a typed field on `theragraph.config.Settings` (pydantic-settings), read
from the environment or `.env`. `.env.example` documents all of them. The ones that
change behaviour most:

| Variable | Default | Effect |
|---|---|---|
| `ENVIRONMENT` | `local` | `production` enables fail-closed checks (see below) |
| `DATABASE_URL` | local PostgreSQL | `postgresql+asyncpg://…` or `sqlite+aiosqlite://…` |
| `REDIS_URL` | `redis://localhost:63790/0` | `memory://` selects the in-process cache |
| `CACHE_ENABLED` | `true` | Redis outages degrade to pass-through, not errors |
| `JOB_QUEUE_BACKEND` | `redis` | `inline` runs jobs in the request process (tests, single-container demos) |
| `STORAGE_BACKEND` | `filesystem` | `s3` for MinIO or real S3 |
| `ANTHROPIC_API_KEY` | empty | Empty ⇒ the agent uses its deterministic planner |
| `AGENT_LLM_BACKEND` | `anthropic` | `deterministic` forces the no-LLM path |
| `AGENT_MODEL` | `claude-opus-5` | |
| `ALLOW_LIVE_INGESTION` | `true` | `false` makes ingestion endpoints accept replay only |
| `AUTH_ENABLED` | `false` | `true` requires a bearer token or API key |
| `RATE_LIMIT_REQUESTS` | 120/min | Per caller |
| `RATE_LIMIT_AGENT_REQUESTS` | 20/min | Tighter: these requests spend model tokens |
| `OTEL_ENABLED` | `false` | Spans are emitted only with an OTLP endpoint configured |

### Production fails closed

`docker-compose.prod.yml` uses `${VAR:?}` for every secret, so the stack refuses to start
rather than falling back to a development default:

```
$ docker compose -f docker-compose.yml -f docker-compose.prod.yml config
required variable DATABASE_URL is missing a value: DATABASE_URL is required in production
```

The overlay also sets `ALLOW_LIVE_INGESTION=false` (a public demo should not generate
traffic against EBI), `LOG_FORMAT=json`, and marks postgres/redis/minio with
`profiles: ["never"]` — in production those are managed services, and leaving the local
containers reachable would be a way to accidentally deploy a database with no backups.

The application itself refuses to boot in `ENVIRONMENT=production` with the development
JWT secret.

## Security

* **Input validation** — every request body and query parameter is a pydantic model with
  bounds. SMILES and SMARTS strings are parsed by RDKit before use, and a parse failure
  is a 422 with the reason.
* **Authentication** — off by default for local development; `AUTH_ENABLED=true` requires
  a bearer JWT or an API key from `API_KEYS`. Any public deployment must set it. There is
  no partial-credit path: a request either authenticates or is refused with a message
  naming the two accepted headers. `/healthz` stays open, because a liveness probe that
  needs a credential reports the wrong thing. The enforcement itself is tested — an
  unauthenticated request, a wrong key, and a token signed with another secret each
  produce 401.
* **Rate limiting** — a global per-caller bucket in middleware (so it also covers
  unmatched routes and runs before any database work), plus a tighter bucket on the agent
  endpoints. Responses carry `X-RateLimit-*` headers and a throttled one carries
  `Retry-After`; `remaining` is clamped at zero. Also tested against a real limit rather
  than assumed.
* **Secrets** — only ever read from the environment. `theragraph info` prints the
  effective configuration with secrets redacted, and logs never contain a key.
* **API keys for upstream services** — none are required. The public APIs TheraGraph uses
  are open; the only optional key is `ANTHROPIC_API_KEY`.
* **Request size** — `MAX_REQUEST_BYTES` (2 MB) is enforced before the body is buffered.
* **Egress** — `ALLOW_LIVE_INGESTION=false` blocks outbound calls to the data sources.
* **Containers** — non-root (UID 10001), no build toolchain in the runtime stage, and a
  `HEALTHCHECK` on `/healthz`.
* **`/metrics`** — served on the API port for the scrape target, and returned as 404 by
  nginx at the edge. Queue depths and error counts are operational data, not public.
* **Traversal** — object-store keys are validated; `../` cannot escape the storage root.

## Reliability

| Concern | Mechanism |
|---|---|
| Upstream API down | Circuit breaker per source; retry with jittered exponential backoff on 429/5xx/network errors, no retry on other 4xx |
| Redis down | Cache degrades to pass-through with one warning log, not an error per request; the job queue falls back to sweeping the PostgreSQL table |
| Duplicate submission | Idempotency keys on jobs and research runs — the second request returns the first result, flagged `idempotent_hit` |
| Worker crash mid-job | Lease with a visibility timeout; a reaper requeues expired leases |
| Repeatedly failing job | `attempts` counted, backoff applied, `dead` after `JOB_MAX_ATTEMPTS`, visible via the jobs API |
| Half-written run | One transaction per run stage; a failed run is recorded as failed with its error, not left dangling |
| Schema drift | `alembic check` in CI fails when the models and migrations disagree |

## Observability

**Logs** — structlog, JSON in deployments. Every line carries `service`, `logger`,
`request_id`, and `run_id` where applicable, so one research run is one `grep`.

**Metrics** — Prometheus on `/metrics`:

| Metric | Labels |
|---|---|
| `http_requests_total`, `http_request_duration_seconds` | method, route, status |
| `db_query_duration_seconds`, slow-query counter | operation |
| `cache_events_total` | result (hit/miss) |
| `job_queue_depth`, `jobs_total` | queue, task, outcome |
| `ml_inference_duration_seconds` | endpoint, model family |
| `agent_tool_duration_seconds`, `agent_tool_failures_total` | tool |
| `agent_tokens_total`, `agent_cost_usd_total` | model |

**Traces** — OpenTelemetry spans for HTTP handlers and SQLAlchemy, exported when
`OTEL_EXPORTER_ENDPOINT` is set. Trace ids are bound into the log context, so a slow
request links its logs and its span.

**Health** — `/healthz` is liveness (the process is up). `/readyz` is readiness: it
checks the database and the cache, and returns 503 when a dependency is unavailable so a
load balancer stops sending traffic.

## Runbook

**API returns 503 on `/readyz`** — check `docker compose logs postgres`. The readiness
body names the failing dependency.

**Predictions return 503 `model_not_available`** — no model is registered for that
endpoint. Run `theragraph ml models` to see what exists, then `theragraph ml train`. The
error body says exactly this rather than returning an empty prediction.

**Jobs stuck in `pending`** — check that a worker is running (`make worker` or the
compose `worker` service) and that Redis is reachable. `GET /api/v1/jobs` shows depth and
status; a worker sweeps the table even without Redis, so this is usually no worker at all.

**Jobs going to `dead`** — `GET /api/v1/jobs/{id}` includes `error_history` with one
entry per attempt.

**Ingestion fails with `LiveIngestionDisabledError`** — either `ALLOW_LIVE_INGESTION` is
false (expected in production) or replay mode wanted a fixture that is not committed. The
error names the file path it looked for.

**Search finds nothing for a real disease** — it is probably not ingested.
`GET /api/v1/search/corpus` reports what is there; `theragraph ingest disease <EFO id>`
adds it.

**An agent run failed** — `GET /api/v1/runs/{id}` returns the full trace including which
tool failed, with what input and what error. Runs are never deleted on failure.

**A replayed run gave a different answer** — that is what `GET /api/v1/runs/{id}/replay-plan`
is for: it reports corpus growth, model retraining and prompt changes since the original.

## Deploying

The images are plain containers; anything that runs a container will do.

```bash
docker build -f backend/Dockerfile -t theragraph-api:$(git rev-parse --short HEAD) .
docker build -f frontend/Dockerfile -t theragraph-web:$(git rev-parse --short HEAD) .
```

The backend image runs the API, the worker, or the MCP server depending on its command,
so one image covers all three processes. Deployment order:

1. `alembic upgrade head` as a one-shot task (the compose `migrate` service does this).
2. Roll the API. Migrations are additive, so an old and a new API can briefly coexist.
3. Roll the workers.

Minimum production configuration:

```bash
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://…      # managed PostgreSQL 16
REDIS_URL=redis://…                      # managed Redis
JWT_SECRET=…                             # not the development default
AUTH_ENABLED=true
ALLOW_LIVE_INGESTION=false
LOG_FORMAT=json
STORAGE_BACKEND=s3
S3_BUCKET=…  S3_ACCESS_KEY_ID=…  S3_SECRET_ACCESS_KEY=…
```

A public demo should additionally ship a pre-ingested corpus and pre-trained models
(train once, upload the artifacts to S3, point `MODEL_REGISTRY_DIR` at the mount), so a
first visitor is not waiting on ingestion.

## CI

`.github/workflows/ci.yml` — six jobs, all offline:

| Job | Gate |
|---|---|
| `backend` | ruff lint, ruff format check, mypy, pytest with coverage, and `docs/openapi.json` is current |
| `postgres` | the full suite against real PostgreSQL 16 + Redis |
| `migrations` | upgrade → `alembic check` → downgrade → upgrade → `alembic check`, on PostgreSQL and on SQLite |
| `frontend` | eslint (zero warnings), prettier check, `tsc --noEmit`, vitest, production build |
| `agent-evals` | rebuild the corpus from fixtures, train a model, run the eval suite against its thresholds |
| `images` | build both images, validate compose, smoke-test the backend image's CLI |

The `postgres` job exists because the dialect is not an implementation detail: bulk
upserts compile different SQL per backend, JSON becomes JSONB, and timestamps come back
aware rather than naive. It has already caught a schema bug that SQLite accepted — three
tables had declared a unique constraint with the same name, which PostgreSQL rejects
because the index backing a constraint shares the relation namespace.
