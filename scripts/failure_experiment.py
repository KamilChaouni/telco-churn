"""TP02 optional extension 3 - failure experiment. Run AFTER `uv run telco-train`:

    uv run python scripts/failure_experiment.py

Starts from a COPY of the reference batch (raw CSV and artifact untouched) and applies
three kinds of change, to show they are different situations:

1. schema failure      - a required column is dropped        -> must raise, no prediction
2. supported missingness - TotalCharges blank / NaN          -> imputed (documented policy)
3. distribution shift  - categories altered / values extreme -> runs silently, scores move

Writes reports/tp02/failure_experiment.json. Inference only: no fit, no labels.
"""

import json
import sys

import numpy as np
import pandas as pd

from telco_churn.dataset import PROJECT_ROOT
from telco_churn.prediction import load_metadata, load_pipeline, predict


def scores(pipeline, batch, threshold):
    return predict(pipeline, batch, threshold)["churn_score"].to_numpy()


def summary(base, new):
    diff = new - base
    return {"mean_score_before": round(float(base.mean()), 4),
            "mean_score_after": round(float(new.mean()), 4),
            "max_abs_change": round(float(np.max(np.abs(diff))), 4)}


def main() -> int:
    meta = load_metadata()
    pipeline = load_pipeline(PROJECT_ROOT / meta["artifact"]["path"])
    threshold = meta["task"]["threshold"]
    batch = pd.read_csv(PROJECT_ROOT / meta["reference_batch"]).drop(columns=["score_original"])
    base = scores(pipeline, batch, threshold)
    seen = meta["learned_statistics"]["categories"]
    results = {}

    # 1. Schema failure
    broken = batch.copy().drop(columns=["Contract"])
    try:
        scores(pipeline, broken, threshold)
        results["schema_failure"] = {"raised": False, "verdict": "PROBLEM: silent prediction"}
    except (ValueError, KeyError) as error:
        results["schema_failure"] = {"raised": True, "error": str(error),
                                     "verdict": "visible failure, no prediction produced"}

    # 2. Supported missingness
    missing = batch.copy()
    missing["TotalCharges"] = missing["TotalCharges"].astype("object")
    missing.loc[missing.index[:5], "TotalCharges"] = " "
    missing.loc[missing.index[5:10], "TotalCharges"] = np.nan
    results["supported_missingness"] = {
        "rows_changed": 10,
        "imputed_with_train_median": meta["learned_statistics"]["numeric_imputer_medians"]["TotalCharges"],
        **summary(base, scores(pipeline, missing, threshold)),
        "verdict": "accepted by policy (blank text and NaN -> train median); scores change for those rows",
    }

    # 3a. Distribution shift - category renamed (e.g. upstream system changes a label)
    renamed = batch.copy()
    renamed["Contract"] = renamed["Contract"].replace({"Month-to-month": "Monthly"})
    n_renamed = int((batch["Contract"] == "Month-to-month").sum())
    unseen = sorted(set(renamed["Contract"]) - set(seen["Contract"]))
    results["shift_renamed_category"] = {
        "rows_changed": n_renamed, "unseen_values": unseen,
        **summary(base, scores(pipeline, renamed, threshold)),
        "raised": False,
        "verdict": "no error (handle_unknown='ignore' -> all-zero one-hot), but predictions "
                   "for these customers are no longer backed by training data",
    }

    # 3b. Distribution shift - values far outside the training range
    extreme = batch.copy()
    extreme["MonthlyCharges"] = extreme["MonthlyCharges"] * 10
    results["shift_out_of_range_numeric"] = {
        "train_max_MonthlyCharges_hint": "about 119 in the source data",
        "batch_max_after": float(extreme["MonthlyCharges"].max()),
        **summary(base, scores(pipeline, extreme, threshold)),
        "raised": False,
        "verdict": "no error: the schema is valid, only monitoring of input distributions would catch it",
    }

    out = PROJECT_ROOT / "reports" / "tp02" / "failure_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if results["schema_failure"]["raised"] else 1


if __name__ == "__main__":
    sys.exit(main())
