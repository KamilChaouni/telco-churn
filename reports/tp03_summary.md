# TP03 — Experiment tracking summary (MLflow 3.18.0)

Local tracking server: `sqlite:///mlflow.db` (metadata) + `./mlartifacts` (files), project root,
`http://127.0.0.1:5000`. Experiment **`telco-churn-tp03`** (id 2). Machine-readable record:
[reports/tp03/tracking_record.json](tp03/tracking_record.json).

## 1. Candidate comparison (validation, threshold 0.5)

Screenshot — runs compared in the UI (select the 4 runs → *Compare*):

![comparison](tp03/screenshots/01_compare_runs.png)

Table exported with `mlflow.search_runs` ([tp03/comparison_runs.csv](tp03/comparison_runs.csv)), not transcribed:

| Run ID | Candidate / changed setting | Val AP | Val precision | Val recall | Val F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `c4d5dc46fa734fce9b2c2680f8e8637c` | `baseline-most-frequent` — DummyClassifier | 0.265 | 0.000 | 0.000 | 0.000 | reference only |
| `4ba04e3c3fa649729d647dc978fbc529` | `logistic-default-C1` — LogisticRegression, max_iter 1000 | 0.647 | 0.664 | 0.529 | 0.589 | runner-up (−0.014) |
| `6adb45e74eb24813852a4b3e8954aa40` | `tree-depth5-leaf20` — DecisionTree, max_depth 5, min_samples_leaf 20 | 0.617 | 0.636 | 0.602 | 0.618 | rejected (best recall/F1, lowest AP) |
| `2f5e21cba9054c5394efb91b170f2123` | `forest-300trees-leaf10` — RandomForest, 300 trees, min_samples_leaf 10 | **0.662** | 0.646 | 0.479 | 0.550 | **selected** |

All runs: same six features, same customer allocation (split assignment SHA-256 `bf235075…fafeb6`,
identical to the TP02 file), seed 42, threshold 0.5, same evaluation function. Test split not evaluated.

**Selection.** The primary metric, validation average precision, and the tie rule (within 0.01
keep the simpler model) were fixed in `configs/train.json` in TP02, before these runs. The random
forest has the highest AP (0.662); the logistic regression is 0.014 lower, outside the tolerance,
so the forest is selected. The margin is small and comes from one split of 1 409 customers, so it
is not evidence of a significant improvement; the forest is also ~18× slower to fit (3.1 s vs 0.17 s)
and has lower recall at 0.5 than the tree. If interpretability or recall at a fixed threshold
mattered more than ranking quality, logistic regression would be a defensible alternative.

## 2. Selected run

Screenshots — run `forest-300trees-leaf10`: *Overview* (parameters, metrics, tags) and the
*Datasets used* panel (`telco_train` / context `training`, `telco_valid` / context `validation`):

![selected run](tp03/screenshots/02_selected_run.png)
![datasets](tp03/screenshots/03_selected_run_datasets.png)

Logged per run: parameters (`model_type`, `clf_*` hyperparameters, `split_seed`, `split_ratios`,
`threshold`, preprocessing choices), metrics (`val_average_precision`, `val_precision`,
`val_recall`, `val_f1`, `val_accuracy`, `fit_seconds`), tags (`git_commit`, `git_dirty`,
`data_sha256`, `split_assignment_sha256`), artifacts (`evaluation/confusion_matrix_valid.csv|json`
— rows true [0=No, 1=Yes], columns predicted [0, 1]; `provenance.json`; `split/split_assignment.csv`),
dataset inputs, and the fitted pipeline `churn_pipeline` (`mlflow.sklearn.log_model`, cloudpickle,
input example of 3 parsed rows; signature: 3 `double` + 3 `string` columns, output `long`).

## 3. Registry

Screenshot — *Models → telco-churn*: version 1 with alias `@candidate`:

![registry](tp03/screenshots/04_registry_alias.png)

## 4. Identifiers

| Identifier | Value |
|---|---|
| Source run ID | `2f5e21cba9054c5394efb91b170f2123` |
| Logged model URI | `models:/m-22d2201ec2d54f20aa80b867ce43df6b` |
| Registered model / version | `telco-churn` / **1** (`models:/telco-churn/1`) |
| Alias | `candidate` → version 1 (`models:/telco-churn@candidate`) |
| Git commit | `f0a23dacf9d4a4858d643b69c2609ca59acb5130` |
| Uncommitted changes at training | `evidence/tp01/handover.md`, `instructions.txt` (docs only, no code; recorded in `provenance.json`, tag `git_dirty=True`) |
| Original data SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Environment | Python 3.12.15, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, mlflow 3.18.0 (`uv.lock`) |

## 5. Fresh-process inference

```sh
uv run python -m telco_churn.registry_inference --label before-restart
```

A new process loads `models:/telco-churn@candidate`, resolves it to version 1 / run
`2f5e21cb…`, parses the 20 reference customers (no `Churn`) with the TP02 parsing rules and
compares with the scores computed in memory at training time:
max |score difference| = **0.0** (tolerance 1e-9), class predictions identical, no `fit`, no
training module imported ([tp03/inference_before-restart.json](tp03/inference_before-restart.json)).

## 6. Restart and storage

Server stopped, port 5000 confirmed free, restarted from the project root with the same command.
After restart ([tp03/restart_check.txt](tp03/restart_check.txt)): experiment 2 still has the 4
FINISHED runs, `telco-churn` still has version 1, `@candidate` still resolves to version 1 / run
`2f5e21cb…`, and the inference command again gives max difference 0.0
([tp03/inference_after-restart.json](tp03/inference_after-restart.json)).

**What does each store preserve?** `mlflow.db` keeps the *records*: experiments, runs, parameters,
metrics, tags, dataset input records (name, digest, schema, source), logged-model entries and the
registry (names, versions, aliases), plus the *locations* of artifacts. `mlartifacts/` keeps the
*files*: the pickled pipelines with `MLmodel`, `requirements.txt`/`conda.yaml`, input examples,
confusion matrices, provenance JSON, split assignment. If only the database survived, every run
and the alias would still be listed, but loading `@candidate` would fail: the model bytes are
gone. If only the artifact files survived, the pickles could be loaded by path, but nothing would
say which run, parameters, metrics, data hash or version/alias each one belongs to.

**What must a colleague on another machine obtain separately?** Both stores (or access to a
shared server), the original CSV (only its hash is recorded — a hash cannot recover data), the
code at commit `f0a23da` with the same `uv.lock` environment so the pickle can be unpickled, and
the same server URI. A run ID or `models:/…` URI alone points into *my* local server and is
useless elsewhere.
