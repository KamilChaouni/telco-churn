"""Test D - serialization preserves behaviour (joblib, temporary file, no refit)."""

import joblib
import numpy as np

from telco_churn.prediction import load_pipeline, predict


def test_reload_preserves_predictions(tmp_path, fitted_pipeline, sample_customers):
    """Defects caught: only the classifier saved (preprocessing lost), fitted state lost,
    loader returning another object. Uses the project's own load_pipeline."""
    artifact = tmp_path / "pipeline.joblib"
    before = predict(fitted_pipeline, sample_customers)

    joblib.dump(fitted_pipeline, artifact)
    reloaded = load_pipeline(artifact)  # trusted file we just wrote
    after = predict(reloaded, sample_customers)  # raw, unencoded columns: needs preprocessing

    assert "preprocessor" in reloaded.named_steps
    np.testing.assert_array_equal(after["churn_predicted"], before["churn_predicted"])
    np.testing.assert_allclose(after["churn_score"], before["churn_score"], rtol=0, atol=1e-12)
