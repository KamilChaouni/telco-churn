"""Test C - probability and prediction output contract. Class order: classes_ == [0, 1],
0 = Churn No, 1 = Churn Yes (positive class)."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.evaluation import positive_scores
from telco_churn.prediction import predict


def test_probability_contract(fitted_pipeline, sample_features):
    """Defects caught: wrong shape, non-probabilities, columns not summing to 1."""
    proba = fitted_pipeline.predict_proba(sample_features)
    assert proba.shape == (len(sample_features), 2)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert list(fitted_pipeline.classes_) == [0, 1]
    np.testing.assert_array_equal(
        positive_scores(fitted_pipeline, sample_features),
        proba[:, list(fitted_pipeline.classes_).index(1)],
    )


class ReversedClasses:
    """Minimal model whose classes_ are NOT in [0, 1] order."""

    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.9), np.full(len(X), 0.1)])


def test_positive_score_uses_classes_not_column_position():
    """Defect caught: proba[:, 1] hard-coded -> scores of the NEGATIVE class when order differs."""
    scores = positive_scores(ReversedClasses(), pd.DataFrame({"a": [1, 2, 3]}))
    np.testing.assert_allclose(scores, 0.9)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_one_row_per_customer_and_threshold(fitted_pipeline, sample_customers, threshold):
    """Defects caught: rows dropped/reordered, ID misaligned, threshold ignored or inverted."""
    out = predict(fitted_pipeline, sample_customers, threshold=threshold)
    assert len(out) == len(sample_customers)
    assert list(out.index) == list(sample_customers.index)
    assert list(out["customerID"]) == list(sample_customers["customerID"])
    expected = (out["churn_score"] >= threshold).astype(int)
    pd.testing.assert_series_equal(out["churn_predicted"], expected, check_names=False)


def test_prediction_does_not_refit(fitted_pipeline, sample_customers, numeric_imputer_statistics):
    """Defect caught: inference calling fit/fit_transform and relearning statistics."""
    before = numeric_imputer_statistics(fitted_pipeline)
    shifted = sample_customers.assign(tenure=sample_customers["tenure"] * 100)
    predict(fitted_pipeline, shifted)
    np.testing.assert_array_equal(before, numeric_imputer_statistics(fitted_pipeline))
