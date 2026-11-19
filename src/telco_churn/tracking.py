"""TP03 - tracked training with MLflow: one run per candidate, then register the selection.

Start the tracking server first (separate terminal, project root), see README:

    uv run python -m mlflow server --host 127.0.0.1 --port 5000 \
        --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

Then:

    uv run python -m telco_churn.tracking            # or: uv run telco-track

Reuses the TP02 code unchanged (same config, split, features, pipeline builder, evaluation).
Each candidate run records parameters, validation metrics, Git/data tags, training and
validation dataset inputs, a confusion matrix, a provenance JSON and the complete fitted
pipeline (mlflow.sklearn.log_model). The selected run's model is registered as
`telco-churn` and the alias `candidate` is pointed at that version.
The test split is NOT evaluated here: it stays reserved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
if hasattr(sys.stdout, "reconfigure"):  # MLflow prints emoji; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow  # noqa: E402
import mlflow.data  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from telco_churn.dataset import PROJECT_ROOT  # noqa: E402
from telco_churn.evaluation import evaluate, positive_scores  # noqa: E402
from telco_churn.pipeline import build_classifier, build_pipeline  # noqa: E402
from telco_churn.training import (  # noqa: E402
    DEFAULT_CONFIG,
    candidate_specs,
    file_sha256,
    git_identity,
    load_config,
    load_data,
    preprocessing_kwargs,
    select_model,
    split_assignment,
    split_data,
    versions,
    xy,
)

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "telco-churn-tp03"
REGISTERED_MODEL = "telco-churn"
ALIAS = "candidate"
REPORTS = PROJECT_ROOT / "reports" / "tp03"

RUN_NAMES = {  # descriptive names: what each candidate actually is
    "majority_baseline": "baseline-most-frequent",
    "logistic_regression": "logistic-default-C1",
    "decision_tree": "tree-depth5-leaf20",
    "random_forest": "forest-300trees-leaf10",
}


def dirty_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ["<git unavailable>"]
    return [line[3:] for line in out.splitlines() if line.strip()]


def dataset_frame(part: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Columns described by the dataset record: id + six raw features + target.

    customerID is included so the digest changes when the customer allocation changes.
    It is NOT a model input (the pipeline only receives prepare_features(...) output).
    """
    f = cfg["features"]
    cols = [cfg["id_column"], *f["numeric"], *f["categorical"], cfg["target"]["column"]]
    return part[cols].reset_index(drop=True)


def confusion_csv(metrics: dict, path: Path) -> Path:
    cm = metrics["confusion_matrix"]
    table = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=pd.Index(["true_0_No", "true_1_Yes"], name="rows=true / cols=predicted"),
        columns=["pred_0_No", "pred_1_Yes"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return path


def provenance(cfg, spec, source_hash, assignment_sha, sizes, code, dirty, pipe) -> dict:
    t = cfg["target"]
    return {
        "candidate": spec["name"],
        "estimator": spec["estimator"],
        "config_params": spec["params"],
        "classifier_get_params": {
            k: repr(v) for k, v in pipe.named_steps["classifier"].get_params().items()
        },
        "features": cfg["features"],
        "never_inputs": [cfg["id_column"], t["column"]],
        "class_mapping": {t["negative_label"]: 0, t["positive_label"]: 1},
        "positive_class": f"{t['column']} = {t['positive_label']} -> 1",
        "preprocessing": cfg["preprocessing"],
        "input_parsing": "telco_churn.features.prepare_features: blank numeric text -> NaN, "
        "malformed numeric text -> error, missing column -> error",
        "source": {"path": cfg["source"], "sha256": source_hash},
        "split": {
            **cfg["split"],
            "stratified_on": t["column"],
            "sizes": sizes,
            "assignment_artifact": "split/split_assignment.csv",
            "assignment_sha256": assignment_sha,
        },
        "evaluation": {
            "split": "valid (test reserved, not evaluated in TP03)",
            "primary_metric": "average_precision from predict_proba column of class 1 "
            "(located via classes_)",
            "threshold": cfg["threshold"],
            "thresholded_metrics": ["precision", "recall", "f1", "accuracy"],
            "confusion_matrix_order": "rows true [0, 1], cols predicted [0, 1]",
        },
        "selection_rule": cfg["selection"]["rule"],
        "code": {**code, "dirty_files": dirty},
        "environment": versions(),
    }


def run_candidate(spec, cfg, data, ctx) -> dict:
    X_train, y_train, X_valid, y_valid = data
    name = spec["name"]
    run_name = RUN_NAMES.get(name, name)
    with mlflow.start_run(run_name=run_name) as run:
        pipe = build_pipeline(
            build_classifier(spec["estimator"], spec["params"]), **preprocessing_kwargs(cfg)
        )  # fresh, unfitted
        mlflow.log_params(
            {
                "candidate": name,
                "model_type": spec["estimator"],
                **{f"clf_{k}": v for k, v in spec["params"].items()},
                "split_seed": cfg["split"]["seed"],
                "split_ratios": f"{cfg['split']['train']}/{cfg['split']['valid']}/{cfg['split']['test']}",
                "threshold": cfg["threshold"],
                "numeric_imputer": cfg["preprocessing"]["numeric_imputer"],
                "numeric_scaler": cfg["preprocessing"]["numeric_scaler"],
                "categorical_unknown": cfg["preprocessing"]["unknown_category"],
                "n_features_raw": len(X_train.columns),
            }
        )
        mlflow.set_tags(
            {
                "git_commit": ctx["code"]["commit"],
                "git_dirty": str(bool(ctx["dirty"])),
                "data_sha256": ctx["source_hash"],
                "split_assignment_sha256": ctx["assignment_sha"],
                "baseline": str(spec.get("baseline", False)),
                "complexity_rank": str(spec["complexity_rank"]),
                "selection_metric": cfg["selection"]["metric"],
            }
        )
        mlflow.log_input(ctx["train_ds"], context="training")
        mlflow.log_input(ctx["valid_ds"], context="validation")

        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        metrics = evaluate(pipe, X_valid, y_valid, cfg["threshold"])

        mlflow.log_metrics(
            {
                "val_average_precision": metrics["average_precision"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
                "val_accuracy": metrics["accuracy"],
                "fit_seconds": fit_seconds,
            }
        )
        cm_path = confusion_csv(metrics, ctx["tmp"] / name / "confusion_matrix_valid.csv")
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
        mlflow.log_dict(metrics["confusion_matrix"], "evaluation/confusion_matrix_valid.json")
        mlflow.log_dict(
            provenance(
                cfg,
                spec,
                ctx["source_hash"],
                ctx["assignment_sha"],
                ctx["sizes"],
                ctx["code"],
                ctx["dirty"],
                pipe,
            ),
            "provenance.json",
        )
        mlflow.log_artifact(str(ctx["assignment_path"]), artifact_path="split")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            name="churn_pipeline",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        print(f"  {run_name:26} run {run.info.run_id}  AP={metrics['average_precision']:.4f}")
        return {
            "model": name,
            "run_name": run_name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model_id": model_info.model_id,
            "estimator": spec["estimator"],
            "baseline": spec.get("baseline", False),
            "complexity_rank": spec["complexity_rank"],
            "params": spec["params"],
            "fit_seconds": fit_seconds,
            "pipeline": pipe,
            **metrics,
        }


def register(selected: dict) -> dict:
    version = mlflow.register_model(selected["model_uri"], REGISTERED_MODEL)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, ALIAS, version.version)
    client.set_model_version_tag(
        REGISTERED_MODEL, version.version, "source_run_id", selected["run_id"]
    )
    resolved = client.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    return {
        "name": REGISTERED_MODEL,
        "version": str(version.version),
        "alias": ALIAS,
        "alias_resolves_to_version": str(resolved.version),
        "alias_uri": f"models:/{REGISTERED_MODEL}@{ALIAS}",
        "version_uri": f"models:/{REGISTERED_MODEL}/{version.version}",
    }


def export_comparison(experiment_id: str, run_ids: list[str]) -> pd.DataFrame:
    """Optional extension: comparison table from mlflow.search_runs, not transcribed."""
    runs = mlflow.search_runs([experiment_id], order_by=["metrics.val_average_precision DESC"])
    runs = runs[runs["run_id"].isin(run_ids)]
    cols = [
        "run_id",
        "tags.mlflow.runName",
        "params.model_type",
        "metrics.val_average_precision",
        "metrics.val_precision",
        "metrics.val_recall",
        "metrics.val_f1",
        "metrics.val_accuracy",
        "metrics.fit_seconds",
        "tags.git_commit",
        "tags.data_sha256",
    ]
    return runs[cols]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Tracked candidate training with MLflow.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="log runs only; register the model by hand in the UI",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config.resolve())
    code, dirty = git_identity(), dirty_files()
    df, source_hash = load_data(cfg)
    splits = split_data(df, cfg)  # same seed + same file => same customers as TP02

    REPORTS.mkdir(parents=True, exist_ok=True)
    assignment_path = REPORTS / "split_assignment.csv"
    split_assignment(splits, cfg["id_column"]).to_csv(assignment_path, index=False)
    assignment_sha = file_sha256(assignment_path)
    tp02_assignment = PROJECT_ROOT / "models" / "split_assignment.csv"
    same_as_tp02 = tp02_assignment.exists() and file_sha256(tp02_assignment) == assignment_sha

    X_train, y_train = xy(splits["train"], cfg)
    X_valid, y_valid = xy(splits["valid"], cfg)

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.set_experiment(args.experiment)
    ctx = {
        "code": code,
        "dirty": dirty,
        "source_hash": source_hash,
        "assignment_sha": assignment_sha,
        "assignment_path": assignment_path,
        "sizes": {k: len(v) for k, v in splits.items()},
        "tmp": REPORTS / "_tmp",
        "train_ds": mlflow.data.from_pandas(
            dataset_frame(splits["train"], cfg),
            name="telco_train",
            targets=cfg["target"]["column"],
            source=cfg["source"],
        ),
        "valid_ds": mlflow.data.from_pandas(
            dataset_frame(splits["valid"], cfg),
            name="telco_valid",
            targets=cfg["target"]["column"],
            source=cfg["source"],
        ),
    }
    print(
        f"Tracking {args.tracking_uri}  experiment {args.experiment} (id {experiment.experiment_id})"
    )
    print(f"Split assignment identical to TP02: {same_as_tp02}")

    rows = [
        run_candidate(spec, cfg, (X_train, y_train, X_valid, y_valid), ctx)
        for spec in candidate_specs(cfg)
    ]
    chosen, reason = select_model(rows, cfg)
    selected = next(r for r in rows if r["model"] == chosen)

    # Reference batch for the fresh-process agreement check: raw-format validation
    # customers (no Churn) + scores of the in-memory fitted pipeline.
    ref = splits["valid"].head(20)[[cfg["id_column"]] + list(X_valid.columns)].copy()
    ref["score_original"] = positive_scores(selected["pipeline"], X_valid.head(20))
    ref["pred_original"] = (ref["score_original"] >= cfg["threshold"]).astype(int)
    ref.to_csv(REPORTS / "reference_batch.csv", index=False)

    registry = None
    if not args.no_register:
        registry = register(selected)

    comparison = export_comparison(experiment.experiment_id, [r["run_id"] for r in rows])
    comparison.to_csv(REPORTS / "comparison_runs.csv", index=False)
    record = {
        "tracking_uri": args.tracking_uri,
        "experiment": {"name": args.experiment, "id": experiment.experiment_id},
        "selected": {
            "candidate": chosen,
            "run_name": selected["run_name"],
            "run_id": selected["run_id"],
            "logged_model_uri": selected["model_uri"],
            "logged_model_id": selected["model_id"],
        },
        "registry": registry,
        "selection": {
            "metric": cfg["selection"]["metric"],
            "split": "valid",
            "tie_tolerance": cfg["selection"]["tie_tolerance"],
            "decision": reason,
        },
        "candidates": [
            {
                k: r[k]
                for k in (
                    "model",
                    "run_name",
                    "run_id",
                    "model_uri",
                    "average_precision",
                    "precision",
                    "recall",
                    "f1",
                    "accuracy",
                    "fit_seconds",
                )
            }
            for r in rows
        ],
        "git_commit": code["commit"],
        "git_dirty_files": dirty,
        "data_sha256": source_hash,
        "split_assignment_sha256": assignment_sha,
        "split_assignment_same_as_tp02": same_as_tp02,
        "threshold": cfg["threshold"],
        "environment": {**versions(), "mlflow": mlflow.__version__},
    }
    (REPORTS / "tracking_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Selected: {selected['run_name']} - {reason}")
    print(f"  run id    : {selected['run_id']}")
    print(f"  model uri : {selected['model_uri']}")
    if registry:
        print(f"  registered: {registry['name']} v{registry['version']} @{registry['alias']}")
    print("Record: reports/tp03/tracking_record.json, comparison_runs.csv, reference_batch.csv")


if __name__ == "__main__":
    main()
