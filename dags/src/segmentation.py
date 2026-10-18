"""Clustering steps for the customer segmentation pipeline.

Each function is one pipeline step. They pass file paths to each other rather
than the data itself, so what travels through Airflow XCom stays small.
"""

import json
import os

import pandas as pd
from joblib import dump, load
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
RANDOM_STATE = 42

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
DATA_DIR = os.path.join(AIRFLOW_HOME, "dags", "data")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join(AIRFLOW_HOME, "artifacts"))

KMEANS_KWARGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": RANDOM_STATE}


def _artifact(name):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    return os.path.join(ARTIFACT_DIR, name)


def validate_data(csv_path):
    """Check the raw file before anything downstream touches it.

    Returns a report dict. 'passed' is what the branch task reads to decide
    whether training should run at all.
    """
    report = {"path": csv_path, "checks": {}, "passed": True}

    if not os.path.exists(csv_path):
        report["checks"]["file_exists"] = False
        report["passed"] = False
        return report
    report["checks"]["file_exists"] = True

    df = pd.read_csv(csv_path)
    report["rows"] = int(len(df))
    report["columns"] = int(df.shape[1])

    missing = [c for c in FEATURES if c not in df.columns]
    report["checks"]["required_columns_present"] = not missing
    report["missing_columns"] = missing

    report["checks"]["has_minimum_rows"] = len(df) >= 100

    if not missing:
        usable = df[FEATURES].dropna()
        report["usable_rows"] = int(len(usable))
        # More than 20% of rows unusable means something upstream changed.
        report["checks"]["null_rate_acceptable"] = len(usable) >= 0.8 * len(df)
        report["checks"]["no_negative_values"] = bool((usable >= 0).all().all())
    else:
        report["usable_rows"] = 0
        report["checks"]["null_rate_acceptable"] = False
        report["checks"]["no_negative_values"] = False

    report["passed"] = all(report["checks"].values())
    return report


def preprocess(csv_path):
    """Scale the clustering features and save the matrix and the scaler.

    The scaler is saved next to the data because anything scoring new rows
    later has to apply the exact same scaling.
    """
    df = pd.read_csv(csv_path)[FEATURES].dropna()

    scaler = MinMaxScaler()
    matrix = scaler.fit_transform(df)

    matrix_path = _artifact("scaled_matrix.joblib")
    scaler_path = _artifact("scaler.joblib")
    dump(matrix, matrix_path)
    dump(scaler, scaler_path)

    return {"matrix_path": matrix_path, "scaler_path": scaler_path, "rows": int(len(df))}


def fit_one_k(k, matrix_path):
    """Fit KMeans for a single k.

    One k per call, so Airflow can map this across many values and run them in
    parallel instead of looping inside one task.
    """
    matrix = load(matrix_path)
    model = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(matrix)
    return {"k": int(k), "sse": float(model.inertia_)}


def select_best_k(results):
    """Pick k at the elbow of the SSE curve."""
    ordered = sorted(results, key=lambda r: r["k"])
    ks = [r["k"] for r in ordered]
    sse = [r["sse"] for r in ordered]

    knee = KneeLocator(ks, sse, curve="convex", direction="decreasing").elbow
    # KneeLocator returns None when the curve has no clear bend.
    best_k = int(knee) if knee is not None else ks[len(ks) // 2]

    summary = {"best_k": best_k, "elbow_found": knee is not None, "curve": ordered}
    with open(_artifact("k_selection.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def train_final_model(best_k, matrix_path):
    """Refit at the chosen k and save that model."""
    matrix = load(matrix_path)
    model = KMeans(n_clusters=best_k, **KMEANS_KWARGS).fit(matrix)

    model_path = _artifact(f"kmeans_k{best_k}.joblib")
    dump(model, model_path)
    return {"model_path": model_path, "k": int(best_k), "sse": float(model.inertia_)}


def score_sample(model_path, scaler_path, sample_csv):
    """Assign a cluster to new rows, applying the saved scaler first."""
    model = load(model_path)
    scaler = load(scaler_path)

    df = pd.read_csv(sample_csv)[FEATURES]
    predictions = model.predict(scaler.transform(df))

    result = {
        "sample_rows": int(len(df)),
        "clusters": [int(p) for p in predictions],
    }
    with open(_artifact("predictions.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result
