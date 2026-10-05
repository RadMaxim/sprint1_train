from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import yaml

from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

from utils.utils import (
    remove_duplicates,
    cast_types,
    fill_missing_values,
    validate_missing_values,
    remove_outliers_iqr,
)


# ---------------------------------------------------------
# Загрузка параметров
# ---------------------------------------------------------

PARAMS_PATH = Path("/opt/airflow/params.yml")

with open(
    PARAMS_PATH,
    "r",
    encoding="utf-8",
) as file:
    params = yaml.safe_load(file)


DATA_DIR = params["data"]["dir"]
RAW_PATH = params["data"]["raw_path"]
TRANSFORMED_PATH = params["data"]["transformed_path"]
FINAL_PATH = params["data"]["final_path"]

SOURCE_CONN_ID = params["connections"]["source_db"]

CATEGORICAL_COLUMNS = params["features"]["categorical"]
NUMERIC_COLUMNS = params["features"]["numeric"]

IQR_THRESHOLD = params["outliers"]["threshold"]


@dag(
    dag_id="sprint1_r_etl",
    schedule="@once",
    start_date=datetime(
        2023,
        1,
        1,
    ),
    catchup=False,
    tags=["ETL"],
)
def users_churn_etl():

    # =====================================================
    # EXTRACT
    # =====================================================

    @task
    def extract() -> str:

        os.makedirs(
            os.path.dirname(RAW_PATH),
            exist_ok=True,
        )

        hook = PostgresHook(
            postgres_conn_id=SOURCE_CONN_ID
        )

        query = """
        SELECT
            c.*,
            i.internet_service,
            i.online_security,
            i.online_backup,
            i.device_protection,
            i.tech_support,
            i.streaming_tv,
            i.streaming_movies,
            p.gender,
            p.senior_citizen,
            p.partner,
            p.dependents,
            ph.multiple_lines
        FROM contracts c
        LEFT JOIN internet i
            ON i.customer_id = c.customer_id
        LEFT JOIN personal p
            ON p.customer_id = c.customer_id
        LEFT JOIN phone ph
            ON ph.customer_id = c.customer_id
        """

        engine = (
            hook
            .get_sqlalchemy_engine()
        )

        data = pd.read_sql(
            query,
            engine,
        )

        print(
            f"Extracted shape: "
            f"{data.shape}"
        )

        data.to_parquet(
            RAW_PATH,
            index=False,
        )

        print(
            f"Raw dataset saved to: "
            f"{RAW_PATH}"
        )

        return RAW_PATH

    # =====================================================
    # TRANSFORM
    # =====================================================

    @task
    def transform(
        input_path: str
    ) -> str:

        os.makedirs(
            os.path.dirname(
                TRANSFORMED_PATH
            ),
            exist_ok=True,
        )

        data = pd.read_parquet(
            input_path
        )

        print(
            f"Initial shape: "
            f"{data.shape}"
        )

        # -------------------------------------------------
        # Удаление технических колонок
        # -------------------------------------------------
        data.drop(
            columns=["begin_date"],
            inplace=True
        )
        data = data.drop(
            columns=[
                "index",
                "customer_id",
            ],
            errors="ignore",
        )

        # -------------------------------------------------
        # Создание target
        # -------------------------------------------------

        data["target"] = (
            data["end_date"] != "No"
        ).astype("int8")

        data = data.drop(
            columns=[
                "end_date",
            ]
        )

        # -------------------------------------------------
        # Удаление дубликатов
        # -------------------------------------------------

        data = remove_duplicates(
            data
        )

        # -------------------------------------------------
        # Приведение типов
        # -------------------------------------------------

        data = cast_types(
            data=data,
            categorical_columns=CATEGORICAL_COLUMNS,
            numeric_columns=NUMERIC_COLUMNS,
        )

        # -------------------------------------------------
        # Заполнение пропусков
        # -------------------------------------------------

        data = fill_missing_values(
            data=data,
            categorical_columns=CATEGORICAL_COLUMNS,
            numeric_columns=NUMERIC_COLUMNS,
        )

        # -------------------------------------------------
        # Проверка пропусков
        # -------------------------------------------------

        validate_missing_values(
            data
        )

        # -------------------------------------------------
        # Поиск и удаление выбросов
        # -------------------------------------------------

        data = remove_outliers_iqr(
            data=data,
            numeric_columns=NUMERIC_COLUMNS,
            threshold=IQR_THRESHOLD,
        )

        print(
            f"Final shape after transform: "
            f"{data.shape}"
        )

        print(
            "Target distribution:"
        )

        print(
            data["target"]
            .value_counts()
        )

        # -------------------------------------------------
        # Сохранение transformed
        # -------------------------------------------------

        data.to_parquet(
            TRANSFORMED_PATH,
            index=False,
        )

        print(
            f"Transformed dataset saved to: "
            f"{TRANSFORMED_PATH}"
        )

        return TRANSFORMED_PATH

    # =====================================================
    # LOAD
    # =====================================================

    @task
    def load(
        input_path: str
    ) -> str:

        os.makedirs(
            os.path.dirname(
                FINAL_PATH
            ),
            exist_ok=True,
        )

        data = pd.read_parquet(
            input_path
        )

        data.to_parquet(
            FINAL_PATH,
            index=False,
        )

        print(
            f"Final dataset saved to: "
            f"{FINAL_PATH}"
        )

        print(
            f"Final shape: "
            f"{data.shape}"
        )

        return FINAL_PATH

    # =====================================================
    # PIPELINE
    # =====================================================

    raw_path = extract()

    transformed_path = transform(
        raw_path
    )

    load(
        transformed_path
    )


users_churn_etl()