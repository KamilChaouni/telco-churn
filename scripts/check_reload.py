"""Fresh-process reload check. Run AFTER `uv run telco-train`, as a separate command:

    uv run python scripts/check_reload.py

It only loads the artifact and predicts: no training data, no labels, no fit call.
Writes reports/tp02/reload_check.json.
"""

import json
import sys

import numpy as np
import pandas as pd

from telco_churn.dataset import PROJECT_ROOT
from telco_churn.prediction import load_metadata, load_pipeline, predict
from telco_churn.training import file_sha256, learned_statistics

TOLERANCE = 1e-9


def main() -> int:
    meta = load_metadata()
    model_path = PROJECT_ROOT / meta["artifact"]["path"]
    sha_ok = file_sha256(model_path) == meta["artifact"]["sha256"]
    pipeline = load_pipeline(model_path)

    batch = pd.read_csv(PROJECT_ROOT / meta["reference_batch"])
    original = batch.pop("score_original").to_numpy()
    assert "Churn" not in batch.columns

    stats_before = learned_statistics(pipeline)
    reloaded = predict(pipeline, batch, meta["task"]["threshold"])["churn_score"].to_numpy()
    stats_after = learned_statistics(pipeline)

    max_diff = float(np.max(np.abs(reloaded - original)))
    result = {
        "artifact_sha256_matches_metadata": sha_ok,
        "rows_compared": int(len(batch)),
        "max_abs_score_difference": max_diff,
        "tolerance": TOLERANCE,
        "scores_agree": max_diff <= TOLERANCE,
        "learned_statistics_match_metadata": stats_before == meta["learned_statistics"],
        "predict_left_statistics_unchanged": stats_before == stats_after,
        "labels_used": False,
        "fit_called": False,
    }
    out = PROJECT_ROOT / "reports" / "tp02" / "reload_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    ok = all(result[k] for k in ("artifact_sha256_matches_metadata", "scores_agree",
                                 "learned_statistics_match_metadata",
                                 "predict_left_statistics_unchanged"))
    print("RELOAD CHECK:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
