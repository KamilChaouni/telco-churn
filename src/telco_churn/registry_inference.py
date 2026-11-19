"""TP03 - inference only: load the pipeline through the registry alias and score customers.

    uv run python -m telco_churn.registry_inference --label before-restart

Never fits and never imports the training modules. Needs the tracking server running.
Applies the same stateless input parsing as TP02 (telco_churn.features.prepare_features),
then compares the loaded pipeline's scores with the scores recorded at training time
(reports/tp03/reference_batch.csv, written by telco_churn.tracking).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
if hasattr(sys.stdout, "reconfigure"):  # MLflow prints emoji; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from telco_churn.dataset import PROJECT_ROOT  # noqa: E402
from telco_churn.evaluation import positive_scores  # noqa: E402
from telco_churn.features import prepare_features  # noqa: E402

TOLERANCE = 1e-9
REPORTS = PROJECT_ROOT / "reports" / "tp03"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Score customers with models:/telco-churn@candidate"
    )
    parser.add_argument("--model-uri", default="models:/telco-churn@candidate")
    parser.add_argument("--input", type=Path, default=REPORTS / "reference_batch.csv")
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label", default="check")
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    name, _, alias = args.model_uri.removeprefix("models:/").partition("@")
    version = MlflowClient().get_model_version_by_alias(name, alias) if alias else None

    pipeline = mlflow.sklearn.load_model(args.model_uri)  # trusted, own artifact only

    batch = pd.read_csv(args.input)
    assert "Churn" not in batch.columns, "inference input must not contain the label"
    original = batch.pop("score_original").to_numpy() if "score_original" in batch else None
    original_pred = batch.pop("pred_original").to_numpy() if "pred_original" in batch else None

    X = prepare_features(batch)  # documented schema + type rules, no learning
    scores = positive_scores(pipeline, X)
    preds = (scores >= args.threshold).astype(int)

    result = {
        "label": args.label,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_uri": args.model_uri,
        "registered_version": version.version if version else None,
        "source_run_id": version.run_id if version else None,
        "logged_model_source": version.source if version else None,
        "rows": int(len(batch)),
        "predicted_churners": int(preds.sum()),
        "fit_called": False,
        "labels_used": False,
    }
    if original is not None:
        diff = float(np.max(np.abs(scores - original)))
        result.update(
            {
                "max_abs_score_difference": diff,
                "tolerance": TOLERANCE,
                "scores_agree": diff <= TOLERANCE,
                "class_predictions_agree": bool(np.array_equal(preds, original_pred)),
            }
        )
    out = REPORTS / f"inference_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    ok = result.get("scores_agree", True) and result.get("class_predictions_agree", True)
    print("AGREEMENT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
