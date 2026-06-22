# Installing MLflow Locally on macOS

This guide covers the installation and setup of MLflow on macOS.

## Prerequisites

- macOS (10.14 or later)
- Python 3.8 or later

## Installation Methods

### Method 1: Using uv (Recommended)

`uv` is a fast Python package and project manager that handles both virtual environment creation and dependency management in one tool.

1. **Install uv**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Initialize a new project** (creates `pyproject.toml` and `.venv`)
   ```bash
   uv init mlflow-tutorial
   cd mlflow-tutorial
   ```
   
   Or if you're in an existing directory:
   ```bash
   uv init
   ```

3. **Add MLflow as a dependency** (automatically creates/activates venv)
   ```bash
   uv add mlflow
   ```

4. **Run MLflow commands with uv**
   ```bash
   uv run mlflow --version
   ```

5. **Activate the virtual environment** (optional, for direct access)
   ```bash
   source .venv/bin/activate
   mlflow --version
   ```

### Method 2: Using pip

1. **Verify Python installation**
   ```bash
   python3 --version
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install MLflow**
   ```bash
   pip install mlflow
   ```

4. **Verify installation**
   ```bash
   mlflow --version
   ```

### Method 3: Using Homebrew

1. **Install Homebrew** (if not already installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python via Homebrew**
   ```bash
   brew install python
   ```

3. **Install MLflow**
   ```bash
   pip3 install mlflow
   ```

## Quick Start

### Start MLflow Tracking Server

Run the MLflow UI locally:

**With uv:**
```bash
uv run mlflow ui
```

**With activated venv:**
```bash
mlflow ui
```

By default, the UI will be available at `http://localhost:5000`

### Start with Custom Configuration

**With uv:**
```bash
# Specify a different port
uv run mlflow ui --port 8080

# Use a specific backend store
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# Use a specific artifact root
uv run mlflow ui --default-artifact-root ./mlruns
```

**With activated venv:**
```bash
# Specify a different port
mlflow ui --port 8080

# Use a specific backend store
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Use a specific artifact root
mlflow ui --default-artifact-root ./mlruns
```

## Directory Structure

MLflow stores runs in a local directory by default:

```
./mlruns/
├── 0/                    # Default experiment
│   ├── meta.yaml
│   └── <run_id>/
│       ├── artifacts/
│       ├── metrics/
│       ├── params/
│       └── tags/
```

## Common Installation Issues

### Issue: Python version incompatibility (ImportError with Python 3.14+)

**Problem:** MLflow doesn't support Python 3.14+ yet. You may see errors like:
```
ImportError: cannot import name 'Traversable' from 'importlib.abc'
```

**Solution with uv:** Pin to a compatible Python version (3.10-3.13):
```bash
# Install Python 3.12
uv python install 3.12

# Pin the project to use Python 3.12
uv python pin 3.12

# Update pyproject.toml requires-python
# Change: requires-python = ">=3.14"
# To: requires-python = ">=3.10,<3.14"

# Sync dependencies
uv sync
```

**Recommended Python versions for MLflow:**
- Python 3.10, 3.11, 3.12, or 3.13
- Avoid Python 3.14+ until MLflow adds support

### Issue: Command not found after installation

**Solution:** Ensure Python's bin directory is in your PATH:
```bash
export PATH="$HOME/Library/Python/3.x/bin:$PATH"
```

Add this line to your `~/.zshrc` or `~/.bash_profile` to make it permanent.

### Issue: Permission denied during installation

**Solution with pip:** Use the `--user` flag:
```bash
pip install --user mlflow
```

**Solution with uv:** This shouldn't happen as uv uses virtual environments by default.

### Issue: SSL certificate errors

**Solution:** Update certificates:
```bash
pip install --upgrade certifi
```

## Additional Dependencies

### For Database Backend (PostgreSQL)

**With uv:**
```bash
brew install postgresql
uv add psycopg2-binary
```

**With pip:**
```bash
brew install postgresql
pip install psycopg2-binary
```

### For Cloud Storage (AWS S3)

**With uv:**
```bash
uv add boto3
```

**With pip:**
```bash
pip install boto3
```

### For Azure Storage

**With uv:**
```bash
uv add azure-storage-blob
```

**With pip:**
```bash
pip install azure-storage-blob
```

## Managing Dependencies with uv

### Add a dependency
```bash
uv add <package-name>
```

### Add a dev dependency
```bash
uv add --dev <package-name>
```

### Remove a dependency
```bash
uv remove <package-name>
```

### Sync dependencies from pyproject.toml
```bash
uv sync
```

### Upgrade all dependencies
```bash
uv lock --upgrade
```

## Upgrading MLflow

**With uv:**
```bash
uv add mlflow --upgrade
```

**With pip:**
```bash
pip install --upgrade mlflow
```

## Uninstalling MLflow

**With uv:**
```bash
uv remove mlflow
```

**With pip:**
```bash
pip uninstall mlflow
```

## Next Steps

- Check out the [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)
- Explore [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- Learn about [MLflow Projects](https://mlflow.org/docs/latest/projects.html)
- Read about [MLflow Models](https://mlflow.org/docs/latest/models.html)

## Resources

- [Official MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)
- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
