"""Focused checks for the inspection code (no dependency on the raw CSV)."""

from pathlib import Path

import pandas as pd
import pytest

from telco_churn.dataset import blank_counts, inspect, sha256_of


def test_blank_counts_differs_from_isna():
    df = pd.DataFrame({"a": ["x", " ", "", None]})
    assert blank_counts(df)["a"] == 2  # " " and "" are blank
    assert int(df["a"].isna().sum()) == 1  # only None is NaN


def test_hash_changes_with_bytes(tmp_path: Path):
    f1, f2 = tmp_path / "a.csv", tmp_path / "b.csv"
    f1.write_bytes(b"x,y\n1,2\n")
    f2.write_bytes(b"x,y\r\n1,2\r\n")  # same content for a human, other bytes
    assert sha256_of(f1) != sha256_of(f2)


def test_wrongly_saved_csv_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text('"customerID,Churn"\n"0001-A,No"\n', encoding="utf-8-sig")
    with pytest.raises(ValueError, match="probably not the original"):
        inspect(bad)
