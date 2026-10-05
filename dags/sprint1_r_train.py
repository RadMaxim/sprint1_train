# dags/sprint1_r_train.py

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pendulum
import yaml

from airflow.sdk import dag, task

from catboost import CatBoostClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)


# ============================================================
# CONFIG
# ============================================================

PARAMS_PATH = Path(
    "/opt/airflow/params.yml"
)

with open(
    PARAMS_PATH,
    "r",
    encoding="utf-8",
) as file:
    params = yaml.safe_load(file)


# ============================================================
# DATA
# ============================================================

DATA_DIR = Path(
    params["data"]["dir"]
)

FINAL_PATH = Path(
    params["data"]["final_path"]
)


# ============================================================
# FEATURES
# ============================================================

CATEGORICAL_COLUMNS = (
    params["features"]["categorical"]
)


# ============================================================
# SPLIT
# ============================================================

SPLIT_PARAMS = params["split"]


# ============================================================
# CROSS VALIDATION
# ============================================================

CV_PARAMS = params["cross_validation"]


# ============================================================
# MODEL
# ============================================================

MODEL_PARAMS = params["model"]


# ============================================================
# ARTIFACTS
# ============================================================

ARTIFACTS = params["artifacts"]

MODEL_DIR = Path(
    ARTIFACTS["model_dir"]
)

MODEL_PATH = Path(
    ARTIFACTS["model_path"]
)

TRAIN_DIR = Path(
    ARTIFACTS["train_dir"]
)

X_TRAIN_PATH = Path(
    ARTIFACTS["x_train_path"]
)

X_VAL_PATH = Path(
    ARTIFACTS["x_val_path"]
)

Y_TRAIN_PATH = Path(
    ARTIFACTS["y_train_path"]
)

Y_VAL_PATH = Path(
    ARTIFACTS["y_val_path"]
)

METRICS_DIR = Path(
    ARTIFACTS["metrics_dir"]
)

METRICS_PATH = Path(
    ARTIFACTS["metrics_path"]
)

CV_METRICS_PATH = Path(
    ARTIFACTS["cv_metrics_path"]
)


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="sprint1_r_train",
    schedule="@once",
    start_date=pendulum.datetime(
        2023,
        1,
        1,
        tz="UTC",
    ),
    catchup=False,
    tags=[
        "ML",
        "CatBoost",
        "Churn",
    ],
)
def churn_training_pipeline():

    # ========================================================
    # 1. SPLIT DATA
    # ========================================================

    @task
    def split_data() -> dict:

        TRAIN_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        data = pd.read_parquet(
            FINAL_PATH
        )

        print(
            f"Dataset shape: {data.shape}"
        )

        # ----------------------------------------------------
        # Restore categorical dtype
        # ----------------------------------------------------

        for column in CATEGORICAL_COLUMNS:
            data[column] = (
                data[column]
                .astype("category")
            )

        # ----------------------------------------------------
        # Features / target
        # ----------------------------------------------------

        X = data.drop(
            columns=["target"]
        )

        y = data["target"]

        # ----------------------------------------------------
        # Train / validation split
        # ----------------------------------------------------

        (
            X_train,
            X_val,
            y_train,
            y_val,
        ) = train_test_split(
            X,
            y,
            test_size=SPLIT_PARAMS[
                "test_size"
            ],
            random_state=SPLIT_PARAMS[
                "random_state"
            ],
            stratify=y,
        )

        print(
            f"Train shape: "
            f"{X_train.shape}"
        )

        print(
            f"Validation shape: "
            f"{X_val.shape}"
        )

        print(
            "Train target distribution:"
        )

        print(
            y_train.value_counts(
                normalize=True
            )
        )

        print(
            "Validation target distribution:"
        )

        print(
            y_val.value_counts(
                normalize=True
            )
        )

        # ----------------------------------------------------
        # Save datasets
        # ----------------------------------------------------

        X_train.to_parquet(
            X_TRAIN_PATH,
            index=False,
        )

        X_val.to_parquet(
            X_VAL_PATH,
            index=False,
        )

        (
            y_train
            .to_frame(
                name="target"
            )
            .to_parquet(
                Y_TRAIN_PATH,
                index=False,
            )
        )

        (
            y_val
            .to_frame(
                name="target"
            )
            .to_parquet(
                Y_VAL_PATH,
                index=False,
            )
        )

        print(
            f"X_train saved: "
            f"{X_TRAIN_PATH}"
        )

        print(
            f"X_val saved: "
            f"{X_VAL_PATH}"
        )

        return {
            "x_train": str(
                X_TRAIN_PATH
            ),
            "x_val": str(
                X_VAL_PATH
            ),
            "y_train": str(
                Y_TRAIN_PATH
            ),
            "y_val": str(
                Y_VAL_PATH
            ),
        }

    # ========================================================
    # 2. TRAIN MODEL
    # ========================================================

    @task
    def train_model(
        dataset_paths: dict,
    ) -> str:

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load train
        # ----------------------------------------------------

        X_train = pd.read_parquet(
            dataset_paths["x_train"]
        )

        y_train = pd.read_parquet(
            dataset_paths["y_train"]
        )["target"]

        # ----------------------------------------------------
        # Restore categorical dtype
        # ----------------------------------------------------

        for column in CATEGORICAL_COLUMNS:
            X_train[column] = (
                X_train[column]
                .astype("category")
            )

        cat_features = (
            X_train
            .select_dtypes(
                include=["category"]
            )
            .columns
            .tolist()
        )

        print(
            "Categorical features:"
        )

        print(
            cat_features
        )

        print(
            "Model parameters:"
        )

        print(
            MODEL_PARAMS
        )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = CatBoostClassifier(
            **MODEL_PARAMS
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model.save_model(
            str(MODEL_PATH)
        )

        print(
            f"Model saved to: "
            f"{MODEL_PATH}"
        )

        return str(
            MODEL_PATH
        )

    # ========================================================
    # 3. EVALUATE MODEL
    # ========================================================

    @task
    def evaluate_model(
        model_path: str,
        dataset_paths: dict,
    ) -> dict:

        METRICS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load validation
        # ----------------------------------------------------

        X_val = pd.read_parquet(
            dataset_paths["x_val"]
        )

        y_val = pd.read_parquet(
            dataset_paths["y_val"]
        )["target"]

        # ----------------------------------------------------
        # Restore categorical dtype
        # ----------------------------------------------------

        for column in CATEGORICAL_COLUMNS:
            X_val[column] = (
                X_val[column]
                .astype("category")
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = CatBoostClassifier()

        model.load_model(
            model_path
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = model.predict(
            X_val
        )

        y_proba = (
            model
            .predict_proba(
                X_val
            )[:, 1]
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        f1 = f1_score(
            y_val,
            y_pred,
        )

        roc_auc = roc_auc_score(
            y_val,
            y_proba,
        )

        metrics = {
            "f1_score": float(
                f1
            ),
            "roc_auc": float(
                roc_auc
            ),
        }

        print(
            f"F1-score: "
            f"{f1:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{roc_auc:.4f}"
        )

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        with open(
            METRICS_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=4,
            )

        print(
            f"Metrics saved to: "
            f"{METRICS_PATH}"
        )

        return metrics

    # ========================================================
    # 4. CROSS VALIDATION
    # ========================================================

    @task
    def cross_validate_model(
        dataset_paths: dict,
    ) -> dict:

        METRICS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load train
        # ----------------------------------------------------

        X_train = pd.read_parquet(
            dataset_paths["x_train"]
        )

        y_train = pd.read_parquet(
            dataset_paths["y_train"]
        )["target"]

        # ----------------------------------------------------
        # Restore categorical dtype
        # ----------------------------------------------------

        for column in CATEGORICAL_COLUMNS:
            X_train[column] = (
                X_train[column]
                .astype("category")
            )

        cat_features = (
            X_train
            .select_dtypes(
                include=["category"]
            )
            .columns
            .tolist()
        )

        # ----------------------------------------------------
        # Stratified K-Fold
        # ----------------------------------------------------

        cv_strategy = StratifiedKFold(
            n_splits=CV_PARAMS[
                "n_splits"
            ],
            shuffle=CV_PARAMS[
                "shuffle"
            ],
            random_state=CV_PARAMS[
                "random_state"
            ],
        )

        f1_scores = []
        roc_auc_scores = []

        # ----------------------------------------------------
        # Manual cross-validation
        # ----------------------------------------------------

        for (
            fold,
            (
                train_idx,
                val_idx,
            ),
        ) in enumerate(
            cv_strategy.split(
                X_train,
                y_train,
            ),
            start=1,
        ):

            print(
                f"Fold "
                f"{fold}/"
                f"{CV_PARAMS['n_splits']}"
            )

            # ------------------------------------------------
            # Fold train / validation
            # ------------------------------------------------

            X_fold_train = (
                X_train.iloc[
                    train_idx
                ]
            )

            X_fold_val = (
                X_train.iloc[
                    val_idx
                ]
            )

            y_fold_train = (
                y_train.iloc[
                    train_idx
                ]
            )

            y_fold_val = (
                y_train.iloc[
                    val_idx
                ]
            )

            # ------------------------------------------------
            # New model for each fold
            # ------------------------------------------------

            model = CatBoostClassifier(
                **MODEL_PARAMS
            )

            # ------------------------------------------------
            # Train fold
            # ------------------------------------------------

            model.fit(
                X_fold_train,
                y_fold_train,
                cat_features=cat_features,
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            y_pred = model.predict(
                X_fold_val
            )

            y_proba = (
                model
                .predict_proba(
                    X_fold_val
                )[:, 1]
            )

            # ------------------------------------------------
            # Fold metrics
            # ------------------------------------------------

            fold_f1 = f1_score(
                y_fold_val,
                y_pred,
            )

            fold_roc_auc = (
                roc_auc_score(
                    y_fold_val,
                    y_proba,
                )
            )

            f1_scores.append(
                fold_f1
            )

            roc_auc_scores.append(
                fold_roc_auc
            )

            print(
                f"Fold {fold}: "
                f"F1="
                f"{fold_f1:.4f}, "
                f"ROC-AUC="
                f"{fold_roc_auc:.4f}"
            )

        # ----------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------

        f1_scores = np.array(
            f1_scores
        )

        roc_auc_scores = np.array(
            roc_auc_scores
        )

        metrics = {
            "f1_mean": float(
                f1_scores.mean()
            ),
            "f1_std": float(
                f1_scores.std()
            ),
            "roc_auc_mean": float(
                roc_auc_scores.mean()
            ),
            "roc_auc_std": float(
                roc_auc_scores.std()
            ),
        }

        print(
            "Cross-validation results:"
        )

        print(
            f"F1: "
            f"{metrics['f1_mean']:.4f} "
            f"± "
            f"{metrics['f1_std']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc_mean']:.4f} "
            f"± "
            f"{metrics['roc_auc_std']:.4f}"
        )

        # ----------------------------------------------------
        # Save CV metrics
        # ----------------------------------------------------

        with open(
            CV_METRICS_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=4,
            )

        print(
            f"CV metrics saved to: "
            f"{CV_METRICS_PATH}"
        )

        return metrics
    # ========================================================
# 5. LOG TO MLFLOW
# ========================================================

    @task
    def log_mlflow(
            model_path: str,
            evaluation_metrics: dict,
            cv_metrics: dict,
            ) -> None:
            import os
            import mlflow
            mlflow.set_tracking_uri(
                "http://host.docker.internal:5000"
            )
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
            mlflow.set_experiment("BASE MODEL PROXY")
            

            

            with mlflow.start_run():

                # ------------------------------------------------
                # Model parameters
                # ------------------------------------------------

                mlflow.log_params(
                    MODEL_PARAMS
                )

                # ------------------------------------------------
                # Split parameters
                # ------------------------------------------------

                mlflow.log_params({
                    "split_test_size":
                        SPLIT_PARAMS["test_size"],

                    "split_random_state":
                        SPLIT_PARAMS["random_state"],
                })

                # ------------------------------------------------
                # Cross-validation parameters
                # ------------------------------------------------

                mlflow.log_params({
                    "cv_n_splits":
                        CV_PARAMS["n_splits"],

                    "cv_shuffle":
                        CV_PARAMS["shuffle"],

                    "cv_random_state":
                        CV_PARAMS["random_state"],
                })

                # ------------------------------------------------
                # Validation metrics
                # ------------------------------------------------

                mlflow.log_metrics({
                    "f1_score":
                        evaluation_metrics[
                            "f1_score"
                        ],

                    "roc_auc":
                        evaluation_metrics[
                            "roc_auc"
                        ],
                })

                # ------------------------------------------------
                # CV metrics
                # ------------------------------------------------

                mlflow.log_metrics({
                    "cv_f1_mean":
                        cv_metrics[
                            "f1_mean"
                        ],

                    "cv_f1_std":
                        cv_metrics[
                            "f1_std"
                        ],

                    "cv_roc_auc_mean":
                        cv_metrics[
                            "roc_auc_mean"
                        ],

                    "cv_roc_auc_std":
                        cv_metrics[
                            "roc_auc_std"
                        ],
                })

                # ------------------------------------------------
                # params.yml
                # ------------------------------------------------

                mlflow.log_artifact(
                    str(PARAMS_PATH),
                    artifact_path="config",
                )

                # ------------------------------------------------
                # CatBoost model file
                # ------------------------------------------------

                mlflow.log_artifact(
                    model_path,
                    artifact_path="model",
                )

                print(
                    "MLflow logging completed"
                )

        # ========================================================
        # PIPELINE
        # ========================================================

    # ========================================================
# PIPELINE
# ========================================================

    dataset_paths = split_data()

    model_path = train_model(
        dataset_paths
    )

    evaluation_metrics = evaluate_model(
        model_path,
        dataset_paths,
    )

    cv_metrics = cross_validate_model(
        dataset_paths
    )

    log_mlflow(
        model_path,
        evaluation_metrics,
        cv_metrics,
    )


churn_training_pipeline()