"""Customer segmentation DAG.

Clusters credit card customers with KMeans. The k search runs as parallel
mapped tasks, and a validation branch stops the run before training if the
input data is bad. See the README for the full flow.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.sdk import TaskGroup, dag, task
from airflow.task.trigger_rule import TriggerRule

from src.segmentation import (
    DATA_DIR,
    fit_one_k,
    preprocess,
    score_sample,
    select_best_k,
    train_final_model,
    validate_data,
)

SAMPLE_CSV = os.path.join(DATA_DIR, "test.csv")

# Which input file to run on. Overridable per run, so the quality gate can be
# exercised by triggering with {"data_file": "bad_sample.csv"}.
DEFAULT_DATA_FILE = "file.csv"

# Each value becomes its own task at runtime.
K_VALUES = list(range(2, 16))


@dag(
    dag_id="customer_segmentation",
    description="Parallel k-search customer segmentation with a data quality gate",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "segmentation"],
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    params={"data_file": DEFAULT_DATA_FILE},
)
def customer_segmentation():

    # reschedule mode frees the worker slot between checks. filepath is
    # templated so the param resolves at run time.
    wait_for_data = FileSensor(
        task_id="wait_for_data",
        filepath=os.path.join(DATA_DIR, "{{ params.data_file }}"),
        poke_interval=10,
        timeout=120,
        mode="reschedule",
    )

    @task
    def validate(**context) -> dict:
        path = os.path.join(DATA_DIR, context["params"]["data_file"])
        report = validate_data(path)
        print(f"[validate] file={path} rows={report.get('rows')} checks={report['checks']}")
        return report

    @task.branch
    def route_on_quality(report: dict) -> str:
        """Decide whether training runs, by returning the id of the next task."""
        if report["passed"]:
            print("[route] data passed validation, training")
            return "preprocess_data"
        print(f"[route] data failed validation: {report['checks']}, stopping")
        return "data_rejected"

    data_rejected = EmptyOperator(task_id="data_rejected")

    @task(task_id="preprocess_data")
    def preprocess_data(**context) -> dict:
        path = os.path.join(DATA_DIR, context["params"]["data_file"])
        paths = preprocess(path)
        print(f"[preprocess] scaled {paths['rows']} rows -> {paths['matrix_path']}")
        return paths

    @task
    def fit_k(k: int, matrix_path: str) -> dict:
        """One KMeans fit. Airflow maps this over K_VALUES."""
        result = fit_one_k(k, matrix_path)
        print(f"[fit_k] k={result['k']} sse={result['sse']:.2f}")
        return result

    @task
    def pick_k(results: list[dict]) -> dict:
        summary = select_best_k(results)
        print(f"[pick_k] best_k={summary['best_k']} elbow_found={summary['elbow_found']}")
        return summary

    @task
    def train_final(summary: dict, paths: dict) -> dict:
        info = train_final_model(summary["best_k"], paths["matrix_path"])
        print(f"[train_final] saved k={info['k']} -> {info['model_path']}")
        return info

    @task
    def score(info: dict, paths: dict) -> dict:
        result = score_sample(info["model_path"], paths["scaler_path"], SAMPLE_CSV)
        print(f"[score] clusters={result['clusters']}")
        return result

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def pipeline_summary(**context) -> str:
        """Report how the run ended, whichever branch it took.

        ALL_DONE means this also runs after a failure, and a failed task
        leaves no XCom, so the report is pulled rather than passed in.
        """
        report = context["ti"].xcom_pull(task_ids="validate")
        if report is None:
            print("[summary] validation did not complete, nothing was trained")
            return "incomplete"

        outcome = "trained" if report["passed"] else "rejected"
        print(f"[summary] input rows={report.get('rows')} outcome={outcome}")
        return outcome

    # ---- wiring ----
    report = validate()
    wait_for_data >> report

    branch = route_on_quality(report)

    paths = preprocess_data()
    branch >> [paths, data_rejected]

    # The group is opened here, not where the tasks are defined: with the
    # TaskFlow API a task joins whichever group is active when it is called.
    with TaskGroup("k_search"):
        fits = fit_k.partial(matrix_path=paths["matrix_path"]).expand(k=K_VALUES)
        best = pick_k(fits)

    info = train_final(best, paths)
    scored = score(info, paths)

    summary = pipeline_summary()
    [report, scored, data_rejected] >> summary


customer_segmentation()
