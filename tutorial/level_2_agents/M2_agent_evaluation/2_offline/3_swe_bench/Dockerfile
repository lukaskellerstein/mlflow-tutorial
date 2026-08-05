FROM python:3.11-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The scoring step runs each repo's tests with pytest; no repo declares it,
# so it must be baked into the image or every run_tests call dies with
# "No module named pytest" and scores 0/0.
RUN pip install --no-cache-dir pytest

# The agent runs INSIDE this container: the SDK on the host spawns
# `podman exec ... claude` (see container_claude.sh), so the Claude Code CLI
# must exist in the image.
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && ln -s /root/.local/bin/claude /usr/local/bin/claude

WORKDIR /workspace
