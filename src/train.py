"""Train and compare baseline fraud classifiers; save best pipeline and artifacts."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "onlinefraud.csv"
TARGET_COLUMN = "isFraud"  # CHANGE THIS to the exact target column in your CSV
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = ROOT / "models" / "fraud_pipeline.joblib"
ARTIFACT_DIR = ROOT / "artifacts"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put your CSV there and update DATA_PATH/TARGET_COLUMN."
        )
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values (top):\n", df.isna().sum().sort_values(ascending=False).head(20))
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' not found. Update TARGET_COLUMN in src/train.py. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    # Normalize common binary encodings. If this fails, explicitly map your dataset's labels.
    if set(pd.Series(y).dropna().unique()).issubset({0, 1, 0.0, 1.0, False, True}):
        y = y.astype(int)
    else:
        raise ValueError(
            "Target must be encoded as 0=normal and 1=fraud. Map labels before training."
        )
    if y.nunique() != 2:
        raise ValueError("Target must contain both classes 0 and 1.")

    X = df.drop(columns=[TARGET_COLUMN,"isFlaggedFraud"],errors="ignore").copy()
    # Drop likely unique identifiers; validate this decision for your dataset.
    id_like = [
        c for c in X.columns
        if c.lower() in {"id", "transactionid", "transaction_id", "nameorig", "namedest", "unnamed: 0"}
    ]
    if id_like:
        print("Dropping likely identifier columns:", id_like)
        X = X.drop(columns=id_like)

    # Remove columns that are entirely empty; imputation cannot learn from them.
    empty_cols = X.columns[X.isna().all()].tolist()
    if empty_cols:
        print("Dropping all-missing columns:", empty_cols)
        X = X.drop(columns=empty_cols)

    print("Target distribution:\n", y.value_counts())
    print("Positive rate:", round(float(y.mean()), 6))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120, class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    best_name, best_pipeline, best_ap = None, None, -1.0
    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
        )
    )
    mlflow.set_experiment("online-fraud-detection")

    for name, estimator in models.items():
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            scores = positive_scores(pipeline, X_test)
            ap = float(average_precision_score(y_test, scores))
            try:
                auc = float(roc_auc_score(y_test, scores))
            except ValueError:
                auc = float("nan")
            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
                "roc_auc": auc,
                "pr_auc_average_precision": ap,
            }
            mlflow.log_params({"model_name": name, "test_size": TEST_SIZE, "random_state": RANDOM_STATE})
            mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
            )
            print(f"\n{name}\n", json.dumps(metrics, indent=2))
            print(classification_report(y_test, preds, zero_division=0))
            results.append({"model": name, **metrics})

            if ap > best_ap:
                best_ap = ap
                best_name = name
                best_pipeline = pipeline
                best_preds = preds
                best_scores = scores
                best_metrics = metrics

    results_df = pd.DataFrame(results).sort_values("pr_auc_average_precision", ascending=False)
    results_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "metrics": best_metrics}, f, indent=2)

    joblib.dump(best_pipeline, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    check_n = min(5, len(X_test))
    if not np.allclose(reloaded.predict(X_test.iloc[:check_n]), best_pipeline.predict(X_test.iloc[:check_n])):
        raise RuntimeError("Saved/reloaded pipeline predictions do not match.")
    print(f"\nSelected by PR-AUC: {best_name}; saved to {MODEL_PATH}")

    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_test, best_scores)
    plt.title(f"Precision-Recall Curve — {best_name}")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()

    print("\nModel comparison:\n", results_df.to_string(index=False))
    print("\nArtifacts saved under:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
