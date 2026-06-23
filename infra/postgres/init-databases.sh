#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    CREATE DATABASE mlflow_db OWNER mlflow;

    CREATE USER temporal WITH PASSWORD 'temporal' CREATEDB;
    CREATE DATABASE temporal_db OWNER temporal;
    CREATE DATABASE temporal_visibility OWNER temporal;

    GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;
    GRANT ALL PRIVILEGES ON DATABASE temporal_db TO temporal;
    GRANT ALL PRIVILEGES ON DATABASE temporal_visibility TO temporal;
EOSQL
