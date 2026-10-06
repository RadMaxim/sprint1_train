# dags/sprint1_r_train.py

from pathlib import Path

import pandas as pd
import pendulum
import yaml
from airflow.sdk import dag, task
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from xgboost import XGBClassifier

def aggregate_cv_metrics(model_name: str, fold_metrics: list[dict]) -> dict:
    metrics_df = pd.DataFrame(fold_metrics)

    return {
        "model": model_name,
        "f1_mean": float(metrics_df["f1"].mean()),
        "f1_std": float(metrics_df["f1"].std()),
        "roc_auc_mean": float(metrics_df["roc_auc"].mean()),
        "roc_auc_std": float(metrics_df["roc_auc"].std()),
        "precision_mean": float(metrics_df["precision"].mean()),
        "precision_std": float(metrics_df["precision"].std()),
        "recall_mean": float(metrics_df["recall"].mean()),
        "recall_std": float(metrics_df["recall"].std()),
        "pr_auc_mean": float(metrics_df["pr_auc"].mean()),
        "pr_auc_std": float(metrics_df["pr_auc"].std()),
    }
def calculate_metrics(
    model_name,
    y_true,
    y_pred,
    y_proba,
):

    return {
        "model": model_name,
        "f1": float(
            f1_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_proba,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_proba,
            )
        ),
    }


# ============================================================
# CONFIG
# ============================================================

PARAMS_PATH = Path("/opt/airflow/params.yml")

with open(
    PARAMS_PATH,
    "r",
    encoding="utf-8",
) as file:
    params = yaml.safe_load(file)


# ============================================================
# DATA
# ============================================================

DATA_DIR = Path(params["data"]["dir"])

FINAL_PATH = Path(params["data"]["final_path"])


# ============================================================
# FEATURES
# ============================================================

CATEGORICAL_COLUMNS = params["features"]["categorical"]

MODELS_PARAMS = params["models"]

CATBOOST_PARAMS = MODELS_PARAMS["catboost"]
LIGHTGBM_PARAMS = MODELS_PARAMS["lightgbm"]
XGBOOST_PARAMS = MODELS_PARAMS["xgboost"]
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


# ============================================================
# ARTIFACTS
# ============================================================

ARTIFACTS = params["artifacts"]

MODEL_DIR = Path(ARTIFACTS["model_dir"])

MODEL_PATH = Path(ARTIFACTS["model_path"])

TRAIN_DIR = Path(ARTIFACTS["train_dir"])

X_TRAIN_PATH = Path(ARTIFACTS["x_train_path"])

X_VAL_PATH = Path(ARTIFACTS["x_val_path"])

Y_TRAIN_PATH = Path(ARTIFACTS["y_train_path"])

Y_VAL_PATH = Path(ARTIFACTS["y_val_path"])

METRICS_DIR = Path(ARTIFACTS["metrics_dir"])

METRICS_PATH = Path(ARTIFACTS["metrics_path"])

CV_METRICS_PATH = Path(ARTIFACTS["cv_metrics_path"])


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
    def evaluate_xgboost(
        model_path: str,
        dataset_paths: dict,
    ) -> dict:

        model = XGBClassifier()

        model.load_model(model_path)

        X_val = pd.read_parquet(dataset_paths["x_val"])

        y_val = pd.read_parquet(dataset_paths["y_val"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X_val[column] = X_val[column].astype("category")

        y_proba = model.predict_proba(X_val)[:, 1]

        y_pred = (y_proba >= 0.5).astype(int)

        return calculate_metrics(
            model_name="xgboost",
            y_true=y_val,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    @task
    def evaluate_lightgbm(
        model_path: str,
        dataset_paths: dict,
    ) -> dict:

        import lightgbm as lgb

        X_val = pd.read_parquet(dataset_paths["x_val"])

        y_val = pd.read_parquet(dataset_paths["y_val"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X_val[column] = X_val[column].astype("category")

        model = lgb.Booster(model_file=model_path)

        y_proba = model.predict(X_val)

        y_pred = (y_proba >= 0.5).astype(int)

        return calculate_metrics(
            model_name="lightgbm",
            y_true=y_val,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    @task
    def split_data() -> dict:

        TRAIN_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        data = pd.read_parquet(FINAL_PATH)

        print(f"Dataset shape: {data.shape}")

        # ----------------------------------------------------
        # Restore categorical dtype
        # ----------------------------------------------------

        for column in CATEGORICAL_COLUMNS:
            data[column] = data[column].astype("category")

        # ----------------------------------------------------
        # Features / target
        # ----------------------------------------------------

        X = data.drop(columns=["target"])

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
            test_size=SPLIT_PARAMS["test_size"],
            random_state=SPLIT_PARAMS["random_state"],
            stratify=y,
        )

        print(f"Train shape: {X_train.shape}")

        print(f"Validation shape: {X_val.shape}")

        print("Train target distribution:")

        print(y_train.value_counts(normalize=True))

        print("Validation target distribution:")

        print(y_val.value_counts(normalize=True))

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
            y_train.to_frame(name="target").to_parquet(
                Y_TRAIN_PATH,
                index=False,
            )
        )

        (
            y_val.to_frame(name="target").to_parquet(
                Y_VAL_PATH,
                index=False,
            )
        )

        print(f"X_train saved: {X_TRAIN_PATH}")

        print(f"X_val saved: {X_VAL_PATH}")

        return {
            "x_train": str(X_TRAIN_PATH),
            "x_val": str(X_VAL_PATH),
            "y_train": str(Y_TRAIN_PATH),
            "y_val": str(Y_VAL_PATH),
        }

 
    @task
    def train_xgboost(
        dataset_paths: dict,
    ) -> str:

        X_train = pd.read_parquet(dataset_paths["x_train"])

        y_train = pd.read_parquet(dataset_paths["y_train"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X_train[column] = X_train[column].astype("category")

        # ---------------------------------------------
        # Вес положительного класса
        # ---------------------------------------------

        class_counts = y_train.value_counts()

        scale_pos_weight = class_counts[0] / class_counts[1]

        model_params = {
            **XGBOOST_PARAMS,
            "scale_pos_weight": scale_pos_weight,
        }

        model = XGBClassifier(**model_params)

        model.fit(
            X_train,
            y_train,
        )

        model_path = MODEL_DIR / "xgboost.json"

        model.save_model(str(model_path))

        return str(model_path)

    @task
    def train_lightgbm(
        dataset_paths: dict,
    ) -> str:

        X_train = pd.read_parquet(dataset_paths["x_train"])

        y_train = pd.read_parquet(dataset_paths["y_train"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X_train[column] = X_train[column].astype("category")

        model = LGBMClassifier(**LIGHTGBM_PARAMS)

        model.fit(
            X_train,
            y_train,
        )

        model_path = MODEL_DIR / "lightgbm.txt"

        model.booster_.save_model(str(model_path))

        return str(model_path)

    @task
    def train_catboost(
        dataset_paths: dict,
    ) -> str:

        X_train = pd.read_parquet(dataset_paths["x_train"])

        y_train = pd.read_parquet(dataset_paths["y_train"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X_train[column] = X_train[column].astype("category")

        cat_features = X_train.select_dtypes(include=["category"]).columns.tolist()

        model = CatBoostClassifier(**CATBOOST_PARAMS)

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
        )

        model_path = MODEL_DIR / "catboost.cbm"

        model.save_model(str(model_path))

        return str(model_path)

    @task
    def evaluate_catboost(
        model_path: str,
        dataset_paths: dict,
    ) -> dict:

        X_val = pd.read_parquet(dataset_paths["x_val"])

        y_val = pd.read_parquet(dataset_paths["y_val"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X_val[column] = X_val[column].astype("category")

        model = CatBoostClassifier()

        model.load_model(model_path)

        y_pred = model.predict(X_val)

        y_proba = model.predict_proba(X_val)[:, 1]

        return calculate_metrics(
            model_name="catboost",
            y_true=y_val,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    @task
    def log_mlflow(
        model_name: str,
        model_path: str,
        model_params: dict,
        evaluation_metrics: dict,
    ) -> None:

        import mlflow

        mlflow.set_tracking_uri("http://host.docker.internal:5000")

        mlflow.set_experiment("Model Compare")

        with mlflow.start_run(run_name=model_name):
            # --------------------------------------------
            # Model name
            # --------------------------------------------

            mlflow.set_tag(
                "model",
                model_name,
            )

            # --------------------------------------------
            # Model parameters
            # --------------------------------------------

            mlflow.log_params(model_params)

            # --------------------------------------------
            # Split parameters
            # --------------------------------------------

            mlflow.log_params(
                {
                    "split_test_size": SPLIT_PARAMS["test_size"],
                    "split_random_state": SPLIT_PARAMS["random_state"],
                }
            )

            # --------------------------------------------
            # Validation metrics
            # --------------------------------------------

            mlflow.log_metrics(
                {
                    "f1": evaluation_metrics["f1"],
                    "roc_auc": evaluation_metrics["roc_auc"],
                    "precision": evaluation_metrics["precision"],
                    "recall": evaluation_metrics["recall"],
                    "pr_auc": evaluation_metrics["pr_auc"],
                }
            )

            # --------------------------------------------
            # params.yml
            # --------------------------------------------

            mlflow.log_artifact(
                str(PARAMS_PATH),
                artifact_path="config",
            )

            # --------------------------------------------
            # Model file
            # --------------------------------------------

            mlflow.log_artifact(
                model_path,
                artifact_path="model",
            )

            print(f"MLflow logging completed: {model_name}")

        # ========================================================
        # PIPELINE
        # ========================================================


    @task
    def cross_validate_catboost(
        dataset_paths: dict,
    ) -> dict:

        X = pd.read_parquet(dataset_paths["x_train"])
        y = pd.read_parquet(dataset_paths["y_train"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X[column] = X[column].astype("category")

        cat_features = (
            X.select_dtypes(include=["category"])
            .columns
            .tolist()
        )

        cv = StratifiedKFold(
            n_splits=CV_PARAMS["n_splits"],
            shuffle=CV_PARAMS["shuffle"],
            random_state=CV_PARAMS["random_state"],
        )

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            print(f"CatBoost fold {fold}")

            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]

            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            model = CatBoostClassifier(
                **CATBOOST_PARAMS
            )

            model.fit(
                X_fold_train,
                y_fold_train,
                cat_features=cat_features,
                verbose=False,
            )

            y_proba = model.predict_proba(
                X_fold_val
            )[:, 1]

            y_pred = (
                y_proba >= 0.5
            ).astype(int)

            metrics = calculate_metrics(
                model_name="catboost",
                y_true=y_fold_val,
                y_pred=y_pred,
                y_proba=y_proba,
            )

            metrics["fold"] = fold

            fold_metrics.append(metrics)

            print(metrics)

        return aggregate_cv_metrics(
            "catboost",
            fold_metrics,
        )
    
    @task
    def cross_validate_lightgbm(
        dataset_paths: dict,
    ) -> dict:

        X = pd.read_parquet(dataset_paths["x_train"])
        y = pd.read_parquet(dataset_paths["y_train"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X[column] = X[column].astype("category")

        cv = StratifiedKFold(
            n_splits=CV_PARAMS["n_splits"],
            shuffle=CV_PARAMS["shuffle"],
            random_state=CV_PARAMS["random_state"],
        )

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            print(f"LightGBM fold {fold}")

            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]

            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            model = LGBMClassifier(
                **LIGHTGBM_PARAMS
            )

            model.fit(
                X_fold_train,
                y_fold_train,
            )

            y_proba = model.predict_proba(
                X_fold_val
            )[:, 1]

            y_pred = (
                y_proba >= 0.5
            ).astype(int)

            metrics = calculate_metrics(
                model_name="lightgbm",
                y_true=y_fold_val,
                y_pred=y_pred,
                y_proba=y_proba,
            )

            metrics["fold"] = fold
            fold_metrics.append(metrics)

        return aggregate_cv_metrics(
            "lightgbm",
            fold_metrics,
        )
    @task
    def cross_validate_xgboost(
        dataset_paths: dict,
    ) -> dict:

        X = pd.read_parquet(dataset_paths["x_train"])
        y = pd.read_parquet(dataset_paths["y_train"])["target"]

        for column in CATEGORICAL_COLUMNS:
            X[column] = X[column].astype("category")

        cv = StratifiedKFold(
            n_splits=CV_PARAMS["n_splits"],
            shuffle=CV_PARAMS["shuffle"],
            random_state=CV_PARAMS["random_state"],
        )

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            print(f"XGBoost fold {fold}")

            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]

            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            class_counts = y_fold_train.value_counts()

            scale_pos_weight = (
                class_counts[0]
                / class_counts[1]
            )

            model = XGBClassifier(
                **XGBOOST_PARAMS,
                scale_pos_weight=scale_pos_weight,
            )

            model.fit(
                X_fold_train,
                y_fold_train,
            )

            y_proba = model.predict_proba(
                X_fold_val
            )[:, 1]

            y_pred = (
                y_proba >= 0.5
            ).astype(int)

            metrics = calculate_metrics(
                model_name="xgboost",
                y_true=y_fold_val,
                y_pred=y_pred,
                y_proba=y_proba,
            )

            metrics["fold"] = fold
            fold_metrics.append(metrics)

        return aggregate_cv_metrics(
            "xgboost",
            fold_metrics,
        )
    # ========================================================
    # TRAIN
    # ========================================================
    @task
    def compare_cv_models(
        catboost_cv: dict,
        lightgbm_cv: dict,
        xgboost_cv: dict,
    ) -> dict:

        results = pd.DataFrame(
            [
                catboost_cv,
                lightgbm_cv,
                xgboost_cv,
            ]
        )

        print("\nCross-validation comparison:")
        print(
            results.to_string(index=False)
        )

        best_index = (
            results["pr_auc_mean"]
            .idxmax()
        )

        best_model = results.loc[
            best_index
        ]

        results.to_csv(
            CV_METRICS_PATH,
            index=False,
        )

        print(
            "\nBest CV model:",
            best_model["model"],
        )

        return {
            "best_model": best_model["model"],
            "best_pr_auc": float(
                best_model["pr_auc_mean"]
            ),
            "best_roc_auc": float(
                best_model["roc_auc_mean"]
            ),
            "best_f1": float(
                best_model["f1_mean"]
            ),
        }
    dataset_paths = split_data()
    catboost_cv = cross_validate_catboost(
    dataset_paths
)

    lightgbm_cv = cross_validate_lightgbm(
        dataset_paths
    )

    xgboost_cv = cross_validate_xgboost(
        dataset_paths
    )

    cv_comparison = compare_cv_models(
        catboost_cv,
        lightgbm_cv,
        xgboost_cv,
    )
    catboost_model = train_catboost(dataset_paths)

    lightgbm_model = train_lightgbm(dataset_paths)

    xgboost_model = train_xgboost(dataset_paths)

    # ========================================================
    # EVALUATE
    # ========================================================

    catboost_metrics = evaluate_catboost(
        catboost_model,
        dataset_paths,
    )

    lightgbm_metrics = evaluate_lightgbm(
        lightgbm_model,
        dataset_paths,
    )

    xgboost_metrics = evaluate_xgboost(
        xgboost_model,
        dataset_paths,
    )


    # ========================================================
    # MLFLOW
    # ========================================================

    log_mlflow(
        "catboost",
        catboost_model,
        CATBOOST_PARAMS,
        catboost_metrics,
    )

    log_mlflow(
        "lightgbm",
        lightgbm_model,
        LIGHTGBM_PARAMS,
        lightgbm_metrics,
    )

    log_mlflow(
        "xgboost",
        xgboost_model,
        XGBOOST_PARAMS,
        xgboost_metrics,
    )


churn_training_pipeline()
