"""Inference only: load a trusted fitted pipeline and score new customers. Never fits.

Command line:
    uv run telco-predict --input new_customers.csv --output predictions.csv
The input CSV needs the six feature columns (customerID optional, Churn not needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from telco_churn.dataset import PROJECT_ROOT
from telco_churn.evaluation import positive_scores
from telco_churn.features import prepare_features

DEFAULT_MODEL = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
DEFAULT_METADATA = PROJECT_ROOT / "models" / "churn_pipeline.metadata.json"


def load_pipeline(path: Path = DEFAULT_MODEL):
    """joblib uses pickle: only load artifacts you produced or trust."""
    return joblib.load(path)


def load_metadata(path: Path = DEFAULT_METADATA) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict(pipeline, customers: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Parse the documented feature schema, then score. Raises on missing columns."""
    X = prepare_features(customers)
    scores = positive_scores(pipeline, X)
    result = pd.DataFrame(index=customers.index)
    if "customerID" in customers.columns:
        result["customerID"] = customers["customerID"]
    result["churn_score"] = scores
    result["churn_predicted"] = (scores >= threshold).astype(int)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict churn with the saved pipeline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args(argv)
    threshold = load_metadata(args.metadata)["task"]["threshold"]
    customers = pd.read_csv(args.input)
    out = predict(load_pipeline(args.model), customers, threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"{len(out)} customers scored -> {args.output} (threshold {threshold})")


if __name__ == "__main__":
    main()
