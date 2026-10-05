import os

import pandas as pd
import pendulum
# dags/churn.py
from steps.messages import send_telegram_success_message
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook


DATA_DIR = "/opt/airflow/data"

RAW_PATH = f"{DATA_DIR}/raw/users_churn_raw.parquet"
TRANSFORMED_PATH = f"{DATA_DIR}/processed/users_churn_transformed.parquet"
FINAL_PATH = f"{DATA_DIR}/users_churn_sprint1.parquet"


@dag(
    schedule="@once",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ETL"],
    dag_id="sprint1_r",
    on_success_callback=send_telegram_success_message
)
def prepare_churn_dataset():

    @task()
    def extract() -> str:

        hook = PostgresHook(
            postgres_conn_id="source_db"
        )

        engine = hook.get_sqlalchemy_engine()

        sql = """
        SELECT
            c.customer_id,
            c.begin_date,
            c.end_date,
            c.type,
            c.paperless_billing,
            c.payment_method,
            c.monthly_charges,
            c.total_charges,

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

        FROM contracts AS c

        LEFT JOIN internet AS i
            ON i.customer_id = c.customer_id

        LEFT JOIN personal AS p
            ON p.customer_id = c.customer_id

        LEFT JOIN phone AS ph
            ON ph.customer_id = c.customer_id
        """

        data = pd.read_sql(
            sql,
            engine
        )

        os.makedirs(
            os.path.dirname(RAW_PATH),
            exist_ok=True
        )

        data.to_parquet(
            RAW_PATH,
            index=False
        )

        print(f"Raw dataset saved: {RAW_PATH}")
        print(f"Rows: {len(data)}")

        return RAW_PATH

    # ============================================================
    # 2. TRANSFORM
    # ============================================================

    @task()
    def transform(input_path: str) -> str:

        data = pd.read_parquet(
            input_path
        )

        data["target"] = (
            data["end_date"] != "No"
        ).astype(int)

        data["end_date"] = data["end_date"].replace(
            {"No": None}
        )

        os.makedirs(
            os.path.dirname(TRANSFORMED_PATH),
            exist_ok=True
        )

        data.to_parquet(
            TRANSFORMED_PATH,
            index=False
        )

        print(
            f"Transformed dataset saved: {TRANSFORMED_PATH}"
        )

        print(
            f"Rows: {len(data)}"
        )

        return TRANSFORMED_PATH

    # ============================================================
    # 3. LOAD / SAVE FINAL DATASET
    # ============================================================

    @task()
    def load(input_path: str) -> str:

        data = pd.read_parquet(
            input_path
        )

        os.makedirs(
            os.path.dirname(FINAL_PATH),
            exist_ok=True
        )
        data.to_parquet(
                    FINAL_PATH,
                    index=False
                )
        

        print(
            f"Final dataset saved: {FINAL_PATH}"
        )

        print(
            f"Rows: {len(data)}"
        )

        return FINAL_PATH

    # ============================================================
    # PIPELINE
    # ============================================================

    raw_path = extract()

    transformed_path = transform(
        raw_path
    )

    load(
        transformed_path
    )


prepare_churn_dataset()