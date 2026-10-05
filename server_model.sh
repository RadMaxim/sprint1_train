#!/bin/sh

set -e

set -a
. ./.env
set +a


# ============================================================
# S3
# ============================================================

export MLFLOW_S3_ENDPOINT_URL="https://storage.yandexcloud.net"


# ============================================================
# PostgreSQL
# ============================================================

DB_DESTINATION_HOST="rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net"
DB_DESTINATION_PORT="6432"
DB_DESTINATION_NAME="playground_mle_20260525_e31877c13e"
DB_DESTINATION_USER="mle_20260525_e31877c13e_freetrack"
DB_DESTINATION_PASSWORD="ТВОЙ_ПАРОЛЬ"


# ============================================================
# Используем отдельную PostgreSQL schema для MLflow
# ============================================================

export PGOPTIONS="-c search_path=mlflow_clean"


# ============================================================
# Database URI
# ============================================================

DB_URI="postgresql://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"


# ============================================================
# MLflow
# ============================================================

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*"