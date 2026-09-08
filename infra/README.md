# Infrastructure

All services needed for the MLflow tutorial, managed via a single Podman Compose
file in **two tiers**, selected by a compose profile:

| Tier | Command | Starts |
|------|---------|--------|
| **Level 1 + Level 2** (default) | `podman compose up -d` | PostgreSQL, MLflow, mlflow-seed, Qdrant |
| **Level 3** | `podman compose --profile level3 up -d` | the above **plus** Temporal (+ Elasticsearch, UI, admin tools), Prometheus, Grafana |

Level 1 and 2 lessons only ever talk to MLflow, which is also the gateway, so the
default tier is deliberately small — Temporal alone brings four containers and
an Elasticsearch JVM that nothing outside Level 3 uses. Volumes and the network
are shared between tiers, so switching never loses data.

## Services

| Service | Tier | Port | URL | Purpose |
|---------|------|------|-----|---------|
| MLflow | L1+ | 5555 | <http://localhost:5555> | Tracking server + UI |
| MLflow AI Gateway | L1+ | 5555 | <http://localhost:5555/gateway/mlflow/v1> | Every lesson's LLM entry point — the same server |
| mlflow-seed | L1+ | — | — | One-shot: `mlflow/gateway/seed_gateway.py` writes its alias list into the gateway, then exits |
| Qdrant | L1+ | 6333 | <http://localhost:6333/dashboard> | Vector DB |
| PostgreSQL | L1+ | 5432 | — | Shared database (MLflow + Temporal) |
| Temporal UI | L3 | 8080 | <http://localhost:8080> | Workflow dashboard |
| Temporal gRPC | L3 | 7233 | localhost:7233 | Workflow engine |
| Grafana | L3 | 3000 | <http://localhost:3000> | Monitoring dashboards |
| Prometheus | L3 | 9090 | <http://localhost:9090> | Metrics collection |
| Elasticsearch | L3 | — | — | Temporal search/visibility (internal) |

**Unsloth Studio** runs natively on macOS (not containerized) for Apple Silicon
GPU access, serving an OpenAI-compatible API at <http://127.0.0.1:8888/v1/>. No
lesson calls it directly — every lesson goes through the **MLflow AI Gateway** on
<http://localhost:5555/gateway/mlflow/v1>, which owns the alias-to-model mapping
and the fallback order (`mlflow/gateway/seed_gateway.py`).

There is no separate proxy container. The MLflow server IS the gateway, which is
why a server-side judge needs no wiring at all: it names `gateway:/gemma-judge`
and the server already holds that endpoint.

## Prerequisites

- [Podman](https://podman.io/) installed (`brew install podman`)
- [Podman Compose](https://github.com/containers/podman-compose) installed (`brew install podman-compose`)
- Podman machine initialized and running:

  ```bash
  podman machine init
  podman machine start
  ```

- [Unsloth Studio](https://unsloth.ai/) installed natively, with an API key exported
  as `UNSLOTH_API_KEY` and **Settings → API → Model auto-switch ON**

## Quick Start

### 1. Start Unsloth Studio (native, not in compose)

Three models cover every alias. Download them in Unsloth, do not load them by
hand — auto-switch does that on demand:

```text
unsloth/gemma-4-26B-A4B-it-qat-GGUF                gemma-chat, gemma-judge, gemma-agent, gemma-tight, gpt-4.1-mini
unsloth/gemma-4-31B-it-qat-GGUF                    gemma-31b-local
second-state/Nomic-embed-text-v1.5-Embedding-GGUF  nomic-embed, text-embedding-3-small
```

Two settings in **Settings → API**, both load-bearing:

- **Model auto-switch: ON.** Unsloth holds ONE model at a time. With this off,
  every alias except the currently loaded model fails with
  `400 ... 'Switch model by request' is off`. With it on, a call to another alias
  unloads the current model and loads the new one — measured at 14 s cold and
  4–10 s once the file is in the page cache, so alternating between aliases is a
  few seconds, not a coffee break.
- **auto_download_model: OFF.** On, an unknown model id becomes a multi-gigabyte
  download rather than an error.

The API key is required on **every** route, `/v1/models` included. Export it:

```bash
export UNSLOTH_API_KEY=...          # from Settings -> API
curl -s http://127.0.0.1:8888/v1/status -H "Authorization: Bearer $UNSLOTH_API_KEY"
```

Compose passes it to `mlflow-seed`, which needs it at seed time to build the
local aliases' secret. With it blank the seeder SKIPS every local alias and says
so — better than building endpoints that 401 hours later.

### 2. Start the stack

```bash
cd infra
cp .env.example .env   # first time only — .env is local-only, never committed

podman compose up -d                     # Level 1 + Level 2: mlflow, mlflow-seed, qdrant, postgres
podman compose --profile level3 up -d    # Level 3: adds temporal, prometheus, grafana
```

The second form is additive — run it on top of a running default tier and only
the six Level 3 containers are created. Working through Level 3 for a while?
Set `COMPOSE_PROFILES=level3` in `.env` and every compose command includes the
Level 3 services without the flag.

### 3. Verify

```bash
# What is running (lists both tiers, whichever are up)
podman compose ps

# MLflow UI
open http://localhost:5555

# Qdrant dashboard
open http://localhost:6333/dashboard

# Level 3 only:
open http://localhost:8080          # Temporal UI
open http://localhost:3000          # Grafana (admin/admin)
```

### 4. Run a lesson

```bash
cd ../tutorial/level_1_models/M1_tracking/1_tracking_fundamentals
uv sync
uv run python main.py
```

## Managing Services

```bash
# Start the Level 1 + Level 2 tier
podman compose up -d

# Start everything (Level 3)
podman compose --profile level3 up -d

# Stop the tier you can see (preserves data)
podman compose down                    # stops the four default-tier services ONLY
podman compose --profile level3 down   # stops all ten

# Stop and remove all data (fresh start) — same rule: the profile decides scope
podman compose --profile level3 down -v

# View logs — naming a service enables its profile, no flag needed
podman compose logs -f mlflow
podman compose logs -f temporal

# Restart a single service
podman compose restart mlflow

# Rebuild MLflow after Dockerfile changes
podman compose build mlflow
podman compose up -d mlflow
```

> [!warning]
> **`podman compose down` without `--profile level3` leaves running Level 3
> containers untouched** — compose only acts on services it can see, and it
> does not treat the invisible ones as orphans either. If you started the full
> stack, stop it with `--profile level3 down` (or set `COMPOSE_PROFILES=level3`
> in `.env` so the flag is implied). `podman compose ps` always shows both tiers,
> so it will tell you what is still up.

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
┌──────────────────────────────────────────────────────────┐
│                      Host (macOS)                        │
│                                                          │
│  ┌────────────────┐                                      │
│  │ Unsloth Studio │  native, GPU access — MLflow reaches  │
│  │ :8888          │  it as host.containers.internal:8888  │
│  └────────▲───────┘                                      │
│           │                                              │
│  ┌────────┼─────── Podman Compose ────────────────────┐  │
│  │        │                                           │  │
│  │  Level 1 + Level 2 tier — podman compose up -d     │  │
│  │        │                                           │  │
│  │  ┌─────┴──────────────┐  ┌───────────┐  ┌────────┐ │  │
│  │  │ MLflow :5555       │◄─┤mlflow-seed│  │ Qdrant │ │  │
│  │  │ tracking + GATEWAY │  │ exits (0) │  │ :6333  │ │  │
│  │  └────┬───────────────┘  └───────────┘  └────────┘ │  │
│  │       │                                            │  │
│  │  ┌────▼────────────────────────┐                   │  │
│  │  │ PostgreSQL :5432            │                   │  │
│  │  │ ├─ mlflow_db                │                   │  │
│  │  │ ├─ temporal_db         (L3) │                   │  │
│  │  │ └─ temporal_visibility (L3) │                   │  │
│  │  └────────────▲────────────────┘                   │  │
│  │               │                                    │  │
│  │  ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │  │
│  │               │                                    │  │
│  │  Level 3 tier │— podman compose --profile level3   │  │
│  │               │                                    │  │
│  │  ┌────────────┴─┐  ┌─────────────┐  ┌───────────┐  │  │
│  │  │ Temporal     │◄─┤ Temporal UI │  │ Elastic-  │  │  │
│  │  │ :7233        │◄─┤ admin-tools │  │ search    │  │  │
│  │  └──────┬───────┘  └─────────────┘  └─────▲─────┘  │  │
│  │         └──────────── visibility ─────────┘        │  │
│  │  ┌──────────────┐  ┌─────────────┐                 │  │
│  │  │ Prometheus   │◄─┤ Grafana     │                 │  │
│  │  │ :9090        │  │ :3000       │                 │  │
│  │  └──────────────┘  └─────────────┘                 │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

The Temporal databases are created by `postgres/init-databases.sh` on the
first start regardless of tier — that runs once, when the `postgres_data`
volume is empty, and creating three unused databases is cheaper than a
first-time Level 3 start that has to re-initialise Postgres.

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

**Temporal / Grafana / Prometheus not reachable, but MLflow is:**
You are on the default tier. Those services only exist under the `level3`
profile — `podman compose ps` will show them missing.

```bash
podman compose --profile level3 up -d
```

**Temporal fails to start:**
Elasticsearch and PostgreSQL must be healthy first. Temporal's auto-setup creates the schema on first run — this can take 30-60 seconds.

```bash
podman compose logs temporal
```

**Reset everything:**

```bash
podman compose --profile level3 down -v   # without the profile the L3 containers keep running
podman compose up -d                      # or --profile level3 up -d
```
