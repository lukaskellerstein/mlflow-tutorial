# Infrastructure

All services needed for the MLflow tutorial, managed via a single Podman Compose file.

## Services

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| MLflow | 5000 | http://localhost:5000 | Tracking server + UI |
| Temporal UI | 8080 | http://localhost:8080 | Workflow dashboard |
| Temporal gRPC | 7233 | localhost:7233 | Workflow engine |
| Qdrant | 6333 | http://localhost:6333/dashboard | Vector DB |
| Grafana | 3000 | http://localhost:3000 | Monitoring dashboards |
| Prometheus | 9090 | http://localhost:9090 | Metrics collection |
| PostgreSQL | 5432 | — | Shared database (MLflow + Temporal) |
| Elasticsearch | — | — | Temporal search/visibility (internal) |

**Ollama** runs natively on macOS (not containerized) for Apple Silicon GPU access.

## Prerequisites

- [Podman](https://podman.io/) installed (`brew install podman`)
- [Podman Compose](https://github.com/containers/podman-compose) installed (`brew install podman-compose`)
- Podman machine initialized and running:
  ```bash
  podman machine init
  podman machine start
  ```
- [Ollama](https://ollama.ai/) installed natively

## Quick Start

### 1. Pull Ollama models (native, not in compose)

```bash
ollama pull gemma4:e2b
ollama pull gemma4:26b
ollama pull nomic-embed-text
```

### 2. Start all services

```bash
cd infra
podman compose up -d
```

### 3. Verify

```bash
# Check all services are running
podman compose ps

# MLflow UI
open http://localhost:5000

# Temporal UI
open http://localhost:8080

# Qdrant dashboard
open http://localhost:6333/dashboard

# Grafana (admin/admin)
open http://localhost:3000
```

### 4. Run a lesson

```bash
cd ../tutorial/level_1/M1_core_platform/1_architecture_overview
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

```
┌──────────────────────────────────────────────────────┐
│                    Host (macOS)                       │
│                                                      │
│  ┌─────────┐                                         │
│  │ Ollama  │  (native, GPU access)                   │
│  │ :11434  │                                         │
│  └─────────┘                                         │
│                                                      │
│  ┌────────────── Podman Compose ──────────────────┐  │
│  │                                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │  │ MLflow   │  │ Temporal │  │ Temporal UI │  │  │
│  │  │ :5000    │  │ :7233    │  │ :8080       │  │  │
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
# Find what's using the port (e.g., 5000)
lsof -i :5000
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
