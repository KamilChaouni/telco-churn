"""TP02 section 5 - focused reliability checks. Run: uv run pytest -q

Tests marked with the raw data are skipped if data/raw/telco.csv is absent.
"""

import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telco_churn.dataset import PROJECT_ROOT
from telco_churn.evaluation import positive_scores
from telco_churn.features import FEATURES, prepare_features
from telco_churn.pipeline import build_pipeline
from telco_churn.training import DEFAULT_CONFIG, load_config, load_data, split_data, xy

RAW = PROJECT_ROOT / "data" / "raw" / "telco.csv"
needs_data = pytest.mark.skipif(not RAW.exists(), reason="raw CSV not downloaded")


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(DEFAULT_CONFIG)
    df, _ = load_data(cfg)
    splits = split_data(df, cfg)
    X_train, y_train = xy(splits["train"], cfg)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    return cfg, df, splits, X_train, pipe


def numeric_imputer(pipe):
    return pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]


def new_customers():
    return pd.DataFrame({
        "customerID": ["NEW-1", "NEW-2", "NEW-3"],
        "tenure": [0, 12, 40],
        "MonthlyCharges": [20.0, 70.5, 99.9],
        "TotalCharges": [" ", np.nan, "3996.0"],          # blank text, NaN, numeric text
        "Contract": ["Month-to-month", "One year", "Weekly plan"],  # last one unseen
        "InternetService": ["DSL", "Fiber optic", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Crypto wallet"],  # unseen
    })


@needs_data
def test_splits_disjoint_complete_and_reproducible(setup):
    cfg, df, splits, *_ = setup
    ids = {k: set(v["customerID"]) for k, v in splits.items()}
    assert not ids["train"] & ids["valid"]
    assert not ids["train"] & ids["test"]
    assert not ids["valid"] & ids["test"]
    assert sum(map(len, ids.values())) == len(df)
    again = split_data(df, cfg)
    for name in splits:
        assert list(again[name]["customerID"]) == list(splits[name]["customerID"])


@needs_data
def test_imputer_median_learned_from_train_only(setup):
    cfg, df, splits, X_train, pipe = setup
    learned = numeric_imputer(pipe).statistics_
    train_medians = X_train[cfg["features"]["numeric"]].median().to_numpy()
    np.testing.assert_allclose(learned, train_medians)
    full = prepare_features(df)[cfg["features"]["numeric"]].median().to_numpy()
    print("train medians", train_medians, "| full-data medians", full)


@needs_data
def test_predict_does_not_refit(setup):
    _, _, splits, _, pipe = setup
    before = numeric_imputer(pipe).statistics_.copy()
    pipe.predict_proba(prepare_features(splits["valid"]))
    np.testing.assert_array_equal(before, numeric_imputer(pipe).statistics_)


@needs_data
def test_missing_numeric_and_unknown_category_follow_policy(setup):
    *_, pipe = setup
    X = prepare_features(new_customers())
    assert X["TotalCharges"].isna().tolist() == [True, True, False]
    scores = positive_scores(pipe, X)  # no error: median imputed, unknown -> all zeros
    assert np.all((scores >= 0) & (scores <= 1))
    encoded = pipe.named_steps["preprocessor"].transform(X)
    cat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    contract_cols = [i for i, n in enumerate(cat_names) if n.startswith("cat__Contract_")]
    assert encoded[2, contract_cols].sum() == 0  # unseen 'Weekly plan' -> no category


def test_malformed_numeric_text_is_rejected():
    bad = new_customers()
    bad.loc[0, "TotalCharges"] = "12,50"
    with pytest.raises(ValueError, match="non-numeric"):
        prepare_features(bad)


@needs_data
def test_missing_required_feature_fails_visibly(setup):
    *_, pipe = setup
    incomplete = new_customers().drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Missing required"):
        prepare_features(incomplete)
    # Even bypassing our parser, the fitted pipeline refuses the missing column.
    X = prepare_features(new_customers()).drop(columns=["Contract"])
    with pytest.raises((ValueError, KeyError)):
        pipe.predict(X)


@needs_data
def test_reload_in_fresh_process_gives_same_scores(setup, tmp_path):
    *_, pipe = setup
    model_path, batch_path = tmp_path / "pipe.joblib", tmp_path / "batch.csv"
    joblib.dump(pipe, model_path)
    new_customers().to_csv(batch_path, index=False)  # no Churn column
    original = positive_scores(pipe, prepare_features(new_customers()))
    code = (
        "import joblib, pandas as pd;"
        "from telco_churn.features import prepare_features;"
        "from telco_churn.evaluation import positive_scores;"
        f"p = joblib.load(r'{model_path}');"
        f"X = prepare_features(pd.read_csv(r'{batch_path}'));"
        "print(','.join(repr(float(s)) for s in positive_scores(p, X)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    reloaded = np.array([float(v) for v in out.stdout.strip().split(",")])
    np.testing.assert_allclose(reloaded, original, rtol=0, atol=1e-12)


def test_feature_list_excludes_id_and_target():
    assert "customerID" not in FEATURES and "Churn" not in FEATURES
