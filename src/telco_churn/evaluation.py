"""Metrics for the positive class `Churn = Yes` (encoded 1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

POSITIVE = 1


def positive_scores(model, X) -> np.ndarray:
    """Probability of class 1, taking the column that matches classes_ (not assumed)."""
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(POSITIVE)]


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    scores = positive_scores(model, X)
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "threshold": threshold,
        "confusion_matrix": {
            "labels": "rows = true [0=No, 1=Yes], columns = predicted [0=No, 1=Yes]",
            "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
        },
        "n": int(len(y)),
        "positives": int(np.sum(y)),
    }
