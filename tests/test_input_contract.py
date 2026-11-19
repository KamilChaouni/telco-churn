"""Tests A and B - the inference input contract (telco_churn.prediction.predict boundary)."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.features import FEATURES, prepare_features
from telco_churn.prediction import predict

# ---------- Test A: a required feature is missing ----------


@pytest.mark.parametrize("missing", FEATURES)
def test_missing_required_feature_is_rejected(fitted_pipeline, sample_customers, missing):
    """Defect caught: a dropped/renamed column silently scored (e.g. column check removed)."""
    broken = sample_customers.drop(columns=[missing])
    with pytest.raises(ValueError, match=rf"Missing required feature column\(s\): \['{missing}'\]"):
        predict(fitted_pipeline, broken)


def test_label_and_id_are_not_inputs(fitted_pipeline, sample_customers):
    """Defect caught: Churn or customerID leaking into the features."""
    with_label = predict(fitted_pipeline, sample_customers)
    flipped = sample_customers.assign(
        Churn=sample_customers["Churn"].map({"Yes": "No", "No": "Yes"}),
        customerID=[f"OTHER-{i}" for i in range(len(sample_customers))],
    )
    without = sample_customers.drop(columns=["Churn", "customerID"])
    np.testing.assert_array_equal(
        predict(fitted_pipeline, flipped)["churn_score"], with_label["churn_score"]
    )
    np.testing.assert_array_equal(
        predict(fitted_pipeline, without)["churn_score"], with_label["churn_score"]
    )


# ---------- Test B: supported input variation ----------


def test_missing_numeric_and_unknown_category_follow_policy(
    fitted_pipeline, sample_customers, sample_features, unseen_payment
):
    """Policy: blank/NaN numeric -> train median; unseen category -> all-zero one-hot.

    Defects caught: imputer removed or fitted elsewhere, handle_unknown switched to
    'error', blank text no longer treated as missing.
    """
    pre = fitted_pipeline.named_steps["preprocessor"]
    encoder = pre.named_transformers_["cat"].named_steps["encoder"]
    cat_columns = next(cols for name, _, cols in pre.transformers_ if name == "cat")
    payment_idx = list(cat_columns).index("PaymentMethod")
    assert unseen_payment not in encoder.categories_[payment_idx]  # really unseen

    batch = sample_customers.copy()  # never mutate the fixture
    batch["TotalCharges"] = batch["TotalCharges"].astype("object")
    batch.loc[0, "TotalCharges"] = np.nan
    batch.loc[1, "TotalCharges"] = "   "  # whitespace-only = permitted missing value
    batch.loc[2, "PaymentMethod"] = unseen_payment

    out = predict(fitted_pipeline, batch)

    assert len(out) == len(batch)
    assert out["churn_score"].between(0, 1).all() and np.isfinite(out["churn_score"]).all()
    assert set(out["churn_predicted"]) <= {0, 1}

    X = prepare_features(batch)
    encoded = pre.transform(X)
    names = list(pre.get_feature_names_out())
    # unknown category -> no PaymentMethod indicator set
    pay_cols = [i for i, n in enumerate(names) if n.startswith("cat__PaymentMethod_")]
    assert encoded[2, pay_cols].sum() == 0
    # missing TotalCharges encoded exactly like the training median
    median = sample_features["TotalCharges"].median()
    as_median = pre.transform(X.iloc[[0]].assign(TotalCharges=median))
    np.testing.assert_allclose(encoded[[0]], as_median)


def test_blank_versus_malformed_numeric_text():
    """Defect caught: errors='coerce' turning '12,50' into a silent NaN."""
    ok = pd.DataFrame({c: ["x"] for c in FEATURES}).assign(
        tenure=[1], MonthlyCharges=[10.0], TotalCharges=[" "]
    )
    assert prepare_features(ok)["TotalCharges"].isna().all()
    with pytest.raises(ValueError, match="non-numeric values"):
        prepare_features(ok.assign(TotalCharges=["12,50"]))
