"""Training workflow: load -> split -> first model -> compare candidates -> select -> save.

Command line (from the project root):

    uv run telco-train                       # uses configs/train.json
    uv run telco-train --config configs/train.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from telco_churn.dataset import PROJECT_ROOT, load_raw, sha256_of
from telco_churn.evaluation import evaluate, positive_scores
from telco_churn.features import encode_target, prepare_features
from telco_churn.pipeline import build_classifier, build_pipeline

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.json"


# ---------- loading ----------

def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_data(cfg: dict) -> tuple[pd.DataFrame, str]:
    source = PROJECT_ROOT / cfg["source"]
    if not source.exists():
        sys.exit(f"Source not found: {cfg['source']}. See references/data_source.md.")
    file_hash = sha256_of(source)
    if file_hash != cfg["expected_sha256"]:
        sys.exit(f"Source hash {file_hash} differs from expected {cfg['expected_sha256']}.")
    return load_raw(source), file_hash


# ---------- splitting ----------

def split_data(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Stratified 60/20/20 with one seed. Same seed + same file => same customers."""
    s, target = cfg["split"], cfg["target"]["column"]
    train, rest = train_test_split(
        df, train_size=s["train"], stratify=df[target], random_state=s["seed"]
    )
    valid_share = s["valid"] / (s["valid"] + s["test"])
    valid, test = train_test_split(
        rest, train_size=valid_share, stratify=rest[target], random_state=s["seed"]
    )
    return {"train": train, "valid": valid, "test": test}


def split_assignment(splits: dict[str, pd.DataFrame], id_column: str) -> pd.DataFrame:
    rows = [
        pd.DataFrame({id_column: part[id_column].to_numpy(), "split": name})
        for name, part in splits.items()
    ]
    return pd.concat(rows, ignore_index=True).sort_values(id_column, ignore_index=True)


def xy(part: pd.DataFrame, cfg: dict, numeric=None, categorical=None):
    f = cfg["features"]
    X = prepare_features(
        part,
        numeric if numeric is not None else f["numeric"],
        categorical if categorical is not None else f["categorical"],
    )
    t = cfg["target"]
    y = encode_target(part[t["column"]], t["positive_label"], t["negative_label"])
    return X, y


# ---------- milestone 1: first model ----------

def first_model_report(splits, cfg) -> dict:
    cols = cfg["first_model_features"]
    X_train, y_train = xy(splits["train"], cfg, numeric=cols, categorical=[])
    X_valid, y_valid = xy(splits["valid"], cfg, numeric=cols, categorical=[])
    report = {"features": cols, "fitted_on": "train", "evaluated_on": "valid"}
    for name, model in {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression_2_features": LogisticRegression(max_iter=1000),
    }.items():
        model.fit(X_train, y_train)  # no missing values in these two columns
        report[name] = evaluate(model, X_valid, y_valid, cfg["threshold"])
    return report


# ---------- milestone 3: fair comparison ----------

def preprocessing_kwargs(cfg) -> dict:
    p, f = cfg["preprocessing"], cfg["features"]
    return {
        "numeric": f["numeric"], "categorical": f["categorical"],
        "numeric_imputer": p["numeric_imputer"], "numeric_scaler": p["numeric_scaler"],
        "categorical_fill_value": p["categorical_imputer_fill_value"],
        "unknown_category": p["unknown_category"],
    }


def candidate_specs(cfg) -> list[dict]:
    baseline = {"name": "majority_baseline", "estimator": "DummyClassifier",
                "complexity_rank": 0, "params": {"strategy": "most_frequent"}, "baseline": True}
    return [baseline] + cfg["candidates"]


def compare_candidates(X_train, y_train, X_valid, y_valid, cfg):
    fitted, rows = {}, []
    for spec in candidate_specs(cfg):
        pipe = build_pipeline(build_classifier(spec["estimator"], spec["params"]),
                              **preprocessing_kwargs(cfg))  # fresh state every time
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        metrics = evaluate(pipe, X_valid, y_valid, cfg["threshold"])
        fitted[spec["name"]] = pipe
        rows.append({
            "model": spec["name"], "estimator": spec["estimator"],
            "baseline": spec.get("baseline", False),
            "complexity_rank": spec["complexity_rank"], "params": spec["params"],
            "fit_seconds": round(fit_seconds, 3), **metrics,
        })
    return fitted, rows


def select_model(rows, cfg) -> tuple[str, str]:
    sel = cfg["selection"]
    learned = [r for r in rows if not r["baseline"]]
    best = max(r[sel["metric"]] for r in learned)
    close = [r for r in learned if best - r[sel["metric"]] <= sel["tie_tolerance"]]
    chosen = min(close, key=lambda r: r["complexity_rank"])
    reason = (
        f"Best validation {sel['metric']} = {best:.4f}. Candidates within "
        f"{sel['tie_tolerance']}: {[r['model'] for r in close]}. "
        f"Chosen: {chosen['model']} ({sel['metric']} = {chosen[sel['metric']]:.4f})"
        + (", the simplest of them." if len(close) > 1 else ", no other candidate is that close.")
    )
    return chosen["model"], reason


# ---------- milestone 4: saving ----------

def git_identity() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = run("rev-parse", "HEAD")
    return {"commit": commit or "unknown",
            "dirty_working_tree": bool(run("status", "--porcelain")) if commit else None}


def versions() -> dict:
    names = ["scikit-learn", "pandas", "numpy", "scipy", "joblib", "telco-churn"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{n: metadata.version(n) for n in names}}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def learned_statistics(pipe) -> dict:
    pre = pipe.named_steps["preprocessor"]
    num = pre.named_transformers_["num"]
    cat = pre.named_transformers_["cat"]
    return {
        "numeric_imputer_medians": dict(zip(pre.transformers_[0][2],
                                            map(float, num.named_steps["imputer"].statistics_))),
        "scaler_means": dict(zip(pre.transformers_[0][2],
                                 map(float, num.named_steps["scaler"].mean_))),
        "categories": {c: [str(v) for v in cats] for c, cats in
                       zip(pre.transformers_[1][2], cat.named_steps["encoder"].categories_)},
    }


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def train(config_path: Path = DEFAULT_CONFIG) -> dict:
    cfg = load_config(config_path)
    code = git_identity()  # before any output is written
    out = {k: PROJECT_ROOT / v for k, v in cfg["outputs"].items()}
    reports = out["reports_dir"]

    df, source_hash = load_data(cfg)
    splits = split_data(df, cfg)
    id_col = cfg["id_column"]
    assignment = split_assignment(splits, id_col)
    out["split_assignment"].parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(out["split_assignment"], index=False)

    first = first_model_report(splits, cfg)
    write_json(reports / "first_model.json", first)

    X_train, y_train = xy(splits["train"], cfg)
    X_valid, y_valid = xy(splits["valid"], cfg)
    fitted, rows = compare_candidates(X_train, y_train, X_valid, y_valid, cfg)
    write_json(reports / "comparison.json", rows)
    flat = pd.json_normalize(rows).drop(columns=["confusion_matrix.labels"], errors="ignore")
    flat.to_csv(reports / "comparison.csv", index=False)

    chosen, reason = select_model(rows, cfg)
    selected = fitted[chosen]

    # Test set: used ONCE, for the already-selected model only, after selection.
    X_test, y_test = xy(splits["test"], cfg)
    test_metrics = evaluate(selected, X_test, y_test, cfg["threshold"])

    joblib.dump(selected, out["model"], compress=3)

    # Reference batch: raw-format validation customers (no Churn) + original scores.
    ref = splits["valid"].head(20)[[id_col] + list(X_valid.columns)].copy()
    ref["score_original"] = positive_scores(selected, X_valid.head(20))
    ref.to_csv(out["reference_batch"], index=False)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": {"path": rel(out["model"]), "sha256": file_sha256(out["model"]),
                     "format": "joblib (pickle) - load trusted files only"},
        "task": {"target": cfg["target"]["column"], "positive_class": "Yes -> 1",
                 "negative_class": "No -> 0", "threshold": cfg["threshold"]},
        "source": {"path": cfg["source"], "sha256": source_hash},
        "split": {**cfg["split"], "stratified_on": cfg["target"]["column"],
                  "sizes": {k: len(v) for k, v in splits.items()},
                  "assignment_file": rel(out["split_assignment"]),
                  "assignment_sha256": file_sha256(out["split_assignment"])},
        "features": {**cfg["features"], "never_inputs": [id_col, cfg["target"]["column"]],
                     "input_types": "numeric: float (NaN allowed, blank text -> NaN, "
                                    "malformed text rejected); categorical: str",
                     "parsing": "telco_churn.features.prepare_features (before the pipeline)"},
        "preprocessing": cfg["preprocessing"],
        "learned_statistics": learned_statistics(selected),
        "selected_model": {"name": chosen,
                           "params": selected.named_steps["classifier"].get_params()},
        "selection": {**cfg["selection"], "decision": reason},
        "validation_metrics": next(r for r in rows if r["model"] == chosen),
        "test_metrics_after_selection": test_metrics,
        "reference_batch": rel(out["reference_batch"]),
        "config": {"path": rel(config_path), "sha256": file_sha256(config_path)},
        "code": code,
        "environment": versions(),
    }
    write_json(out["metadata"], meta)
    write_selection_note(reports / "selection.md", rows, chosen, reason, test_metrics, cfg)
    return meta


def write_selection_note(path, rows, chosen, reason, test_metrics, cfg):
    lines = [
        "# Model selection (generated by `uv run telco-train`)", "",
        f"Primary metric chosen **before** comparison: `{cfg['selection']['metric']}` "
        f"on the validation split (positive class Churn = Yes).",
        f"Rule: {cfg['selection']['rule']}", "",
        "| model | avg precision | precision | recall | F1 | accuracy | fit s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['average_precision']:.4f} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} | {r['accuracy']:.3f} | "
                     f"{r['fit_seconds']:.2f} |")
    lines += ["", f"Threshold for precision/recall/F1/accuracy: {cfg['threshold']}.", "",
              f"**Decision:** {reason}", "",
              "Test split, evaluated once after the decision (not used to choose): "
              f"average precision {test_metrics['average_precision']:.4f}, "
              f"recall {test_metrics['recall']:.3f}, precision {test_metrics['precision']:.3f}.",
              ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare churn pipelines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    meta = train(args.config.resolve())
    v = meta["validation_metrics"]
    print(f"Selected: {meta['selected_model']['name']}")
    print(f"  {meta['selection']['decision']}")
    print(f"  valid: AP={v['average_precision']:.4f} recall={v['recall']:.3f} "
          f"precision={v['precision']:.3f} accuracy={v['accuracy']:.3f}")
    print(f"Artifact : {meta['artifact']['path']} (sha256 {meta['artifact']['sha256'][:12]}...)")
    print(f"Metadata : models/churn_pipeline.metadata.json")
    print(f"Reports  : reports/tp02/ (first_model.json, comparison.csv/json, selection.md)")


if __name__ == "__main__":
    main()
