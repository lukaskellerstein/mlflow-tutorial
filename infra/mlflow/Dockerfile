FROM ghcr.io/mlflow/mlflow:latest

RUN pip install --no-cache-dir psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow_db", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--artifacts-destination", "/mlartifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
