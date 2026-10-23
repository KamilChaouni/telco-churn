"""Feature schema and stateless input parsing (the boundary before the Pipeline).

Rules, applied identically at training and at inference:
- numeric columns: whitespace-only / empty text becomes NaN (allowed missing value);
  any other non-numeric text raises a ValueError (never silently turned into NaN).
- categorical columns: kept as text (object dtype); missing stays missing.
- a missing required column raises a ValueError.
Nothing here is learned from data, so it can safely run before the split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {missing}")


def parse_numeric(series: pd.Series) -> pd.Series:
    """Blank text -> NaN; malformed nonblank text -> ValueError; numbers unchanged."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    text = series.astype("string").str.strip()
    blank = text.eq("").fillna(False)
    text = text.mask(blank)  # blank -> <NA>
    converted = pd.to_numeric(text, errors="coerce")
    malformed = converted.isna() & text.notna()
    if malformed.any():
        examples = text[malformed].unique()[:5].tolist()
        raise ValueError(f"Column {series.name!r}: non-numeric values {examples}")
    return converted.astype("float64")


def prepare_features(
    df: pd.DataFrame,
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """Return only the model input columns, in documented types. Does not modify df."""
    check_columns(df, numeric + categorical)
    out = pd.DataFrame(index=df.index)
    for column in numeric:
        out[column] = parse_numeric(df[column])
    for column in categorical:
        values = df[column].astype("object")
        out[column] = values.where(values.notna(), np.nan)
    return out


def encode_target(series: pd.Series, positive: str = "Yes", negative: str = "No") -> pd.Series:
    unexpected = set(series.dropna().unique()) - {positive, negative}
    if unexpected or series.isna().any():
        raise ValueError(f"Unexpected target values: {sorted(map(str, unexpected))}")
    return series.map({negative: 0, positive: 1}).astype("int64")
