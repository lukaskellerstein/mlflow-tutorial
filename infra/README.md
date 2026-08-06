# Infrastructure

All services needed for the MLflow tutorial, managed via a single Podman Compose file.

## Services

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| MLflow | 5555 | <http://localhost:5555> | Tracking server + UI |
| Temporal UI | 8080 | <http://localhost:8080> | Workflow dashboard |
| Temporal gRPC | 7233 | localhost:7233 | Workflow engine |
| Qdrant | 6333 | <http://localhost:6333/dashboard> | Vector DB |
| Grafana | 3000 | <http://localhost:3000> | Monitoring dashboards |
| Prometheus | 9090 | <http://localhost:9090> | Metrics collection |
| PostgreSQL | 5432 | — | Shared database (MLflow + Temporal) |
| Elasticsearch | — | — | Temporal search/visibility (internal) |

**LMStudio** runs natively on macOS (not containerized) for Apple Silicon GPU access,
serving an OpenAI-compatible API at <http://localhost:1234/v1/>. No lesson calls
it directly — every lesson goes through the **LiteLLM gateway** on
<http://localhost:4000/v1/>, which owns the alias-to-model mapping, the fallback
order and each model's declared context window (`litellm/config.yaml`).

## Prerequisites

- [Podman](https://podman.io/) installed (`brew install podman`)
- [Podman Compose](https://github.com/containers/podman-compose) installed (`brew install podman-compose`)
- Podman machine initialized and running:

  ```bash
  podman machine init
  podman machine start
  ```

- [LMStudio](https://lmstudio.ai/) installed natively

## Quick Start

### 1. Start LMStudio and load models (native, not in compose)

```bash
lms server start
lms ls                                       # what is downloaded
lms unload --all                                                             # one model resident only
lms load google/gemma-4-26b-a4b --context-length 262144 --parallel 1 --gpu max  # serves every gemma-* alias
lms load text-embedding-nomic-embed-text-v1.5                                # -> nomic-embed
lms ps --json                                # confirm what is ACTUALLY loaded
```

Load only what the lesson needs — the models are large and share GPU memory.

Two flags that are not optional if you care about the numbers in
`litellm/config.yaml`:

- **`--context-length`** must match the `max_input_tokens` declared there. LMStudio's
  own default is much smaller, and a model loaded smaller than declared still
  receives oversized prompts — the gateway's pre-call check trusts the declaration,
  not the model.
- **`--parallel 1`** because the lessons are sequential loops. Four slots do make
  four *concurrent* requests 2.68x faster, but nothing here issues them: the same
  lesson took 199s at `--parallel 1` and 275s at `--parallel 4`, and the
  evaluation lesson was unchanged (121s vs 118s).
- **`lms unload --all` first.** A second resident model measurably slows the one
  you are using — the cheapest speedup available.

And a trap worth knowing: if a model is *not* resident when a request arrives,
LMStudio just-in-time loads it — **ignoring both flags** and attaching a 1h TTL.
A model hand-loaded at 262144 that idles out can come back far smaller. `lms ps
--json` reports the live `contextLength`; the UI does not always agree.

### 2. Start all services

```bash
cd infra
cp .env.example .env   # first time only — .env is local-only, never committed
podman compose up -d
```

### 3. Verify

```bash
# Check all services are running
podman compose ps

# MLflow UI
open http://localhost:5555

# Temporal UI
open http://localhost:8080

# Qdrant dashboard
open http://localhost:6333/dashboard

# Grafana (admin/admin)
open http://localhost:3000
```

### 4. Run a lesson

```bash
cd ../tutorial/level_1_models/M1_tracking/1_tracking_fundamentals
uv sync
uv run python main.py
```

## Managing Services

```bash
# Start all services
podman compose up -d

# Stop all services (preserves data)
podman compose down

# Stop and remove all data (fresh start)
podman compose down -v

# View logs
podman compose logs -f mlflow
podman compose logs -f temporal

# Restart a single service
podman compose restart mlflow

# Rebuild MLflow after Dockerfile changes
podman compose build mlflow
podman compose up -d mlflow
```

## Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| Grafana | admin | admin |
| PostgreSQL (admin) | admin | admin |
| PostgreSQL (MLflow) | mlflow | mlflow |
| PostgreSQL (Temporal) | temporal | temporal |

Change these in `.env` before deploying outside of local development.

## Data Persistence

All data is stored in named Podman volumes:

| Volume | Service | Content |
|--------|---------|---------|
| `postgres_data` | PostgreSQL | MLflow + Temporal databases |
| `mlflow_artifacts` | MLflow | Model artifacts, logged files |
| `elasticsearch_data` | Elasticsearch | Temporal search index |
| `qdrant_data` | Qdrant | Vector collections |
| `grafana_data` | Grafana | Dashboards, settings |

Data survives `podman compose down`. To reset everything: `podman compose down -v`.

## Architecture

```text
┌──────────────────────────────────────────────────────┐
│                    Host (macOS)                       │
│                                                      │
│  ┌───────────┐                                       │
│  │ LMStudio  │  (native, GPU access)                 │
│  │ :1234     │                                       │
│  └───────────┘                                       │
│                                                      │
│  ┌────────────── Podman Compose ──────────────────┐  │
│  │                                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │  │ MLflow   │  │ Temporal │  │ Temporal UI │  │  │
│  │  │ :5555    │  │ :7233    │  │ :8080       │  │  │
│  │  └────┬─────┘  └────┬─────┘  └─────────────┘  │  │
│  │       │              │                         │  │
│  │       └──────┬───────┘                         │  │
│  │              │                                 │  │
│  │  ┌───────────▼──────────┐  ┌───────────────┐  │  │
│  │  │ PostgreSQL :5432     │  │ Elasticsearch │  │  │
│  │  │ ├─ mlflow_db         │  │ (Temporal     │  │  │
│  │  │ ├─ temporal_db       │  │  visibility)  │  │  │
│  │  │ └─ temporal_visibility│  └───────────────┘  │  │
│  │  └──────────────────────┘                      │  │
│  │                                                │  │
│  │  ┌──────────┐  ┌───────────┐  ┌────────────┐  │  │
│  │  │ Qdrant   │  │Prometheus │  │ Grafana    │  │  │
│  │  │ :6333    │  │ :9090     │  │ :3000      │  │  │
│  │  └──────────┘  └───────────┘  └────────────┘  │  │
│  │                                                │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Troubleshooting

**Podman machine not running:**

```bash
podman machine start
```

**Port already in use:**

```bash
# Find what's using the port (e.g., 5555)
lsof -i :5555
# Kill it or change the port in compose.yml
```

**MLflow can't connect to PostgreSQL:**
Wait for PostgreSQL to be healthy. Check logs:

```bash
podman compose logs postgres
podman compose logs mlflow
```

**Temporal fails to start:**
Elasticsearch and PostgreSQL must be healthy first. Temporal's auto-setup creates the schema on first run — this can take 30-60 seconds.

```bash
podman compose logs temporal
```

**Reset everything:**

```bash
podman compose down -v
podman compose up -d
```
