"""Builders that return *unfitted* sklearn objects. They never read data or train."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from telco_churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Explicit allow-list: the config can only name these estimators.
ESTIMATORS = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "DummyClassifier": DummyClassifier,
}
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}


def build_classifier(estimator: str, params: dict | None = None):
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator!r}; allowed: {sorted(ESTIMATORS)}")
    return ESTIMATORS[estimator](**(params or {}))


def build_preprocessor(
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
    numeric_imputer: str = "median",
    numeric_scaler: str = "standard",
    categorical_fill_value: str = "missing",
    unknown_category: str = "ignore",
) -> ColumnTransformer:
    """Numeric: impute (median) -> scale. Categorical: impute constant -> one-hot.

    remainder="drop": any extra column (customerID, Churn...) is never used as input.
    Unknown categories at inference are encoded as all-zeros (handle_unknown="ignore").
    """
    numeric_branch = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_imputer)),
        ("scaler", SCALERS[numeric_scaler]()),
    ])
    categorical_branch = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=categorical_fill_value)),
        ("encoder", OneHotEncoder(handle_unknown=unknown_category, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_branch, list(numeric)), ("cat", categorical_branch, list(categorical))],
        remainder="drop",
    )


def build_pipeline(classifier, **preprocessing) -> Pipeline:
    """A fresh preprocessor + the given (unfitted) classifier, as one estimator."""
    return Pipeline([
        ("preprocessor", build_preprocessor(**preprocessing)),
        ("classifier", classifier),
    ])
