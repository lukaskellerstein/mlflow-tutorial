# LiteLLM proxy plus the MLflow client its success callback needs in-process.
#
# The base image ships a venv at /app/.venv WITHOUT pip, so bootstrap pip via
# ensurepip first, then install mlflow-skinny (client + tracing only, none of the
# heavy server dependencies — the tracking server is its own container).
FROM ghcr.io/berriai/litellm:main-stable

RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir "mlflow-skinny>=3.0"

# The base image's ENTRYPOINT is `litellm`; compose passes --config and --port.
