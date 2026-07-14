# L1-M8.1 — Authentication and Permissions

**Level:** Essentials
**Duration:** ~20 minutes

## Overview

MLflow supports built-in authentication and permission management for multi-user deployments. This lesson explains when and how to enable authentication, the permission model, and how credentials are configured. Our local dev server runs without auth, so this lesson is primarily educational with a short demo.

## Prerequisites

- Completed: L1-M1 through L1-M9 (familiarity with MLflow basics)
- MLflow server running at http://127.0.0.1:5000
- Infrastructure started with `podman compose up -d` from `infra/`

## Concepts

### When Do You Need Authentication?

For local, single-user development (like this tutorial), authentication is unnecessary. But in shared or production environments with multiple data scientists, you need auth to:

- **Prevent accidental overwrites** — stop one user from modifying another's experiments
- **Enforce access control** — restrict who can delete runs or register models
- **Audit activity** — track which user performed which actions

### Enabling Authentication

MLflow ships with a built-in `basic-auth` plugin. Start the server with:

```bash
mlflow server --app-name basic-auth
```

This creates a default admin user (`admin` / `password`). Change the password immediately.

### Setting Credentials

Clients authenticate via environment variables:

```bash
export MLFLOW_TRACKING_USERNAME="your_username"
export MLFLOW_TRACKING_PASSWORD="your_password"
```

Or set them in Python before any MLflow calls:

```python
import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_password"
```

### Permission Levels

MLflow defines four permission levels, from least to most access:

| Permission | Capabilities |
|---|---|
| `NO_PERMISSIONS` | No access |
| `READ` | View experiments, runs, and models |
| `EDIT` | READ + create/update runs and models |
| `MANAGE` | EDIT + delete resources, grant permissions to others |

Permissions are scoped to individual **experiments** and **registered models**. An admin can grant different users different access levels on each experiment.

### Managing Users and Permissions

When auth is enabled, admins use the `AuthServiceClient`:

```python
from mlflow.server import get_app_client

auth_client = get_app_client("basic-auth", "http://127.0.0.1:5000")

# Create a user
auth_client.create_user("alice", "secret123")

# Grant experiment access
auth_client.update_experiment_permission(
    experiment_id="1",
    username="alice",
    permission="EDIT"
)

# Grant model access
auth_client.update_registered_model_permission(
    name="my-model",
    username="alice",
    permission="READ"
)
```

## Step-by-Step

### Step 1: Understand the Scenarios

The lesson prints a comparison of local dev (no auth) vs. production (auth required). Read through the output to understand when authentication matters.

### Step 2: Review Auth Configuration

The lesson shows how to start MLflow with auth enabled and how to set credentials. Note the two approaches: environment variables and programmatic configuration.

### Step 3: Learn the Permission Model

Four levels (NO_PERMISSIONS, READ, EDIT, MANAGE) applied at experiment and model scope. Admins use the `AuthServiceClient` to manage users and permissions.

### Step 4: Demo Without Auth

A simple run is logged to our local server to demonstrate that it works without credentials. In production, you would enable auth and require credentials.

## Running the Lesson

```bash
cd tutorial/level_1/M8_auth/1_auth_permissions
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Part 1: When You Need Authentication
============================================================

Local development (our setup):
  - Single user, no auth needed
  - MLflow server started without --app-name basic-auth

Multi-user / production:
  - Multiple data scientists sharing one server
  - Need to control who can modify experiments/models
  - Enable auth to protect data and enforce access control

============================================================
Part 2: How to Enable Authentication
============================================================

Start the server with auth enabled:
  mlflow server --app-name basic-auth

Default admin credentials (change immediately!):
  Username: admin
  Password: password

...

============================================================
Part 4: Demo — Our Server Works Without Auth
============================================================

Run ID:     <run_id>
Experiment: <experiment_id>
Logged param:  auth_enabled=False
Logged param:  environment=local_development
Logged metric: demo_score=1.0

Our local server accepted the run without credentials.
In production, enable auth to protect your MLflow data.

============================================================
Lesson complete! View results at http://127.0.0.1:5000
============================================================
```

## Key Takeaways

- MLflow authentication is optional for local dev, essential for production
- Enable with `mlflow server --app-name basic-auth`
- Four permission levels: NO_PERMISSIONS, READ, EDIT, MANAGE
- Permissions are scoped per experiment and per registered model
- Credentials are set via `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`

## Next Steps

This completes Level 1 of the MLflow tutorial. You now have a broad understanding of every major MLflow feature. In Level 2, you will go deeper into each area with real-world projects and production patterns.
