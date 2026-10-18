# Customer Segmentation Pipeline with Airflow

IE 7374 MLOps, Lab Assignment 2

Based on [Airflow Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_1)
from the course repo, with the changes listed below.

An Airflow DAG that segments credit card customers with KMeans. The search for
the best number of clusters runs as parallel mapped tasks, and a validation
gate stops the pipeline before training if the input data is bad.

## The DAG

```
wait_for_data            FileSensor, waits for the CSV to appear
      |
   validate              quality report on the raw file
      |
 route_on_quality        branch: train, or stop here
      |-------------------------> data_rejected
      |
 preprocess_data         scales features, saves the scaler
      |
 [k_search]              fit_k mapped over k=2..15, running in parallel
      |                  pick_k reduces the SSE curve to one k
 train_final             refits at the chosen k
      |
    score                assigns a cluster to a new row
      |
 pipeline_summary        runs either way, reports which path was taken
```

14 of these tasks are created at run time, one per value of k. They run in
parallel across the Airflow workers.

## Prerequisites

You only need Docker. Everything else runs inside the containers.

| Requirement | Version used | Notes |
|---|---|---|
| Docker Desktop | 29.5.2 | Must be running before you start |
| Docker Compose | v5.1.4 | Ships with Docker Desktop |
| Memory for Docker | 4 GB minimum, 8 GB better | Airflow will not start reliably below 4 GB |
| Disk space | about 5 GB | For the Airflow image and Postgres volume |

To raise the memory: Docker Desktop, Settings, Resources, Memory.

Python, Airflow, scikit-learn and the rest are all inside the image. Nothing is
installed on your machine.

Versions pinned in this project:

- Apache Airflow 3.3.2
- scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, joblib 1.6.0 (from the Airflow image)
- kneed 0.8.5, pytest 8.4.2 (added in `requirements.txt`)

## Setup

```bash
git clone <this-repo-url>
cd mlops-airflow-customer-segmentation

# Tell the containers which user id to write files as
echo "AIRFLOW_UID=$(id -u)" > .env

mkdir -p logs plugins config artifacts

# Builds the image and starts Postgres, Redis and the Airflow services.
# First run takes a few minutes while the image downloads.
docker compose up -d
```

Wait until every service reports healthy:

```bash
docker compose ps
```

```
SERVICE                 STATUS
airflow-apiserver       Up (healthy)
airflow-dag-processor   Up (healthy)
airflow-scheduler       Up (healthy)
airflow-triggerer       Up (healthy)
airflow-worker          Up (healthy)
postgres                Up (healthy)
redis                   Up (healthy)
```

The web UI is at **http://localhost:8080**, username `airflow`, password `airflow`.

## Running the pipeline

From the UI: open the `customer_segmentation` DAG, unpause it with the toggle,
then press the play button.

From the command line:

```bash
docker compose exec airflow-scheduler airflow dags unpause customer_segmentation
docker compose exec airflow-scheduler airflow dags trigger customer_segmentation
```

## What a successful run looks like

The run takes about 20 seconds once the containers are up. Every task turns
dark green in the UI except `data_rejected`, which is pink because the branch
skipped it.

Checking from the command line:

```bash
docker compose exec airflow-scheduler airflow tasks states-for-dag-run \
  customer_segmentation <run_id> -o plain
```

```
wait_for_data          success
validate               success
route_on_quality       success
data_rejected          skipped      <- branch chose to train
preprocess_data        success
k_search.fit_k         success      map=0 .. map=13   <- 14 parallel tasks
k_search.pick_k        success
train_final            success
score                  success
pipeline_summary       success
```

Four files appear in `artifacts/` on your machine:

```
artifacts/
├── k_selection.json      the SSE curve and the k that was chosen
├── kmeans_k5.joblib      the trained model, named after the chosen k
├── scaler.joblib         the fitted scaler
└── predictions.json      cluster assigned to the row in test.csv
```

`k_selection.json` should show:

```json
{ "best_k": 5, "elbow_found": true, ... }
```

and the log line from the last task:

```
[summary] input rows=8950 outcome=trained
```

## Seeing the quality gate reject bad data

`dags/data/bad_sample.csv` is deliberately broken. It is missing the
`CREDIT_LIMIT` column, has a null, has a negative balance and only three rows.

```bash
docker compose exec airflow-scheduler airflow dags trigger customer_segmentation \
  --conf '{"data_file": "bad_sample.csv"}'
```

The run still succeeds, but nothing trains:

```
wait_for_data          success
validate               success
route_on_quality       success
data_rejected          success      <- rejection path taken
preprocess_data        skipped
k_search.fit_k         skipped
k_search.pick_k        skipped
train_final            skipped
score                  skipped
pipeline_summary       success
```

```
[summary] input rows=3 outcome=rejected
```

No new files are written to `artifacts/`.

## Running the tests

```bash
docker compose run --rm airflow-cli bash -c "cd /opt/airflow && pytest tests -v"
```

16 tests. They cover the validation rules, the scaling, the elbow selection,
and the DAG structure itself, including a check that `fit_k` really is a mapped
task and that nothing can train without passing through the branch.

## Shutting down

```bash
docker compose down        # stop
docker compose down -v     # stop and delete the database volume
```

## What I changed from the original lab

### Orchestration

| Change | What it does |
|---|---|
| **Parallel k search** | The original fit 49 KMeans models in a `for` loop inside one task. Each k is now its own task created with `.expand()`, so they run in parallel and each one shows separately in the UI with its own log and retry |
| **Data quality gate** | New `validate` task plus a `@task.branch` that routes to `data_rejected` and skips training when the input fails its checks |
| **FileSensor** | The DAG waits for the input file instead of assuming it is there. Uses `reschedule` mode so it frees the worker slot between checks |
| **TaskGroup** | The mapped fits and the reducer are grouped under `k_search` so the graph view stays readable with 14 parallel tasks |
| **Trigger rule** | `pipeline_summary` uses `ALL_DONE` so a run reports its outcome even when something upstream failed |
| **DAG params** | The input filename is a parameter, so the gate can be exercised without editing code |
| **Retries** | Tasks retry once after 30 seconds |
| **TaskFlow API** | Rewritten with `@task` decorators instead of `PythonOperator` boilerplate |

### Things I fixed while getting it working

| What I ran into | What I did |
|---|---|
| The loop kept the model from its last iteration, k=49, while the elbow method reported a different k | `train_final` refits at the chosen k and saves that one |
| `test.csv` was scored on raw values while the model had trained on scaled data, and the scaler was not saved anywhere | The scaler is saved in `preprocess` and applied in `score`, with a test covering it |
| The full 8,950 row dataframe was base64 pickled through XCom | Tasks write to `artifacts/` and pass file paths |
| Nothing checked the input before training | The validate and branch tasks above |

### Other

- Added `tests/test_pipeline.py` with 16 tests
- Added a `Dockerfile` and `docker-compose.override.yaml` so Airflow's official
  compose file stays unmodified

## Layout

```
dags/customer_segmentation_dag.py   the DAG
dags/src/segmentation.py            the clustering steps
dags/data/file.csv                  8,950 credit card customers
dags/data/test.csv                  one row to score
dags/data/bad_sample.csv            broken input for testing the gate
tests/test_pipeline.py              16 tests
Dockerfile                          Airflow image plus kneed and pytest
docker-compose.yaml                 Airflow's official compose file, unmodified
docker-compose.override.yaml        build, artifacts mount, fs_default connection
requirements.txt                    the two packages the image is missing
artifacts/                          written by the pipeline
```

## Things that caught me out

**`The conn_id 'fs_default' isn't defined`.** Airflow 3 no longer creates
default connections, and `FileSensor` needs one. It is set as an environment
variable in `docker-compose.override.yaml`, so you should not hit this, but
that is what the error means if you do.

**The compose file is committed here** rather than fetched during setup. The
`stable` URL serves whatever the current Airflow release is, so a future
version would stop matching the override.

**TaskGroups and the TaskFlow API.** A task joins the group that is active when
it is *called*, not where it is defined. Defining `@task` functions inside the
`with TaskGroup(...)` block but calling them outside leaves them ungrouped.

## Limitations

- Artifacts are overwritten each run rather than versioned by timestamp
- The SSE curve is not plotted anywhere, only written to JSON
- `K_VALUES` is hardcoded to 2 through 15 rather than being a parameter
- The pipeline is triggered manually and has no schedule
