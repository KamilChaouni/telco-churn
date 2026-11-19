"""Shared, small synthetic fixtures for the TP04 behaviour tests.

No raw CSV, no network, no MLflow: 16 invented customers, both classes, the six TP02
features in their documented raw input types, fitted with the project's own builder.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telco_churn.features import encode_target, prepare_features
from telco_churn.pipeline import build_pipeline

# Categories seen during fixture fitting. "Credit card (automatic)" (a real Telco value)
# is deliberately left out so tests can use it as an unseen category.
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)"]
UNSEEN_PAYMENT = "Credit card (automatic)"


@pytest.fixture
def sample_customers() -> pd.DataFrame:
    """Raw-format inference batch: customerID + six features + Churn (as in the CSV)."""
    n = 16
    tenure = [1, 2, 5, 8, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 70, 72]
    monthly = [
        95.0,
        89.5,
        80.2,
        75.0,
        70.1,
        99.9,
        20.0,
        25.5,
        55.0,
        60.3,
        19.9,
        45.0,
        105.0,
        30.0,
        65.0,
        110.0,
    ]
    return pd.DataFrame(
        {
            "customerID": [f"TEST-{i:02d}" for i in range(n)],
            "tenure": tenure,
            "MonthlyCharges": monthly,
            # raw CSV stores TotalCharges as text
            "TotalCharges": [f"{t * m:.2f}" for t, m in zip(tenure, monthly)],
            "Contract": [CONTRACTS[i % 3] for i in range(n)],
            "InternetService": [INTERNET[(i // 2) % 3] for i in range(n)],
            "PaymentMethod": [PAYMENTS[(i // 3) % 3] for i in range(n)],
            # short tenure + month-to-month tends to churn; both classes present (7 Yes / 9 No)
            "Churn": [
                "Yes",
                "Yes",
                "Yes",
                "No",
                "Yes",
                "Yes",
                "No",
                "No",
                "Yes",
                "No",
                "No",
                "No",
                "Yes",
                "No",
                "No",
                "No",
            ],
        }
    )


@pytest.fixture
def sample_features(sample_customers) -> pd.DataFrame:
    """Only the six model inputs, parsed by the documented boundary."""
    return prepare_features(sample_customers)


@pytest.fixture
def sample_target(sample_customers) -> pd.Series:
    return encode_target(sample_customers["Churn"])  # Yes -> 1, No -> 0


@pytest.fixture
def fitted_pipeline(sample_features, sample_target):
    """The project's real preprocessing + classifier, fitted on the fixture only."""
    assert set(sample_target) == {0, 1}
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    return pipe.fit(sample_features, sample_target)


@pytest.fixture
def numeric_imputer_statistics():
    def get(pipe) -> np.ndarray:
        num = pipe.named_steps["preprocessor"].named_transformers_["num"]
        return num.named_steps["imputer"].statistics_.copy()

    return get


@pytest.fixture
def unseen_payment() -> str:
    return UNSEEN_PAYMENT
