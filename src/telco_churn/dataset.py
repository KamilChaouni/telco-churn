"""Read the raw Telco snapshot and produce a machine-readable inspection report.

This module only *inspects* the data: it never drops rows, imputes values,
splits the data or trains anything. The original CSV is opened read-only.

Command line (from the project root):

    uv run telco-inspect
    uv run telco-inspect --source data/raw/telco.csv --output reports/data_overview.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "telco.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "data_overview.json"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def sha256_of(path: Path) -> str:
    """SHA-256 of the file bytes, read in chunks so large files are fine."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV exactly as pandas parses it, without any cleaning."""
    return pd.read_csv(path)


def blank_counts(df: pd.DataFrame) -> dict[str, int]:
    """Per column: values that are present (not NaN) but empty or whitespace-only."""
    counts = {}
    for column in df.columns:
        as_text = df[column].astype("string")
        counts[column] = int(as_text.str.strip().eq("").fillna(False).sum())
    return counts


def environment_info() -> dict:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: metadata.version(name)
            for name in ("pandas", "numpy", "telco-churn")
        },
    }


def inspect(source: Path) -> dict:
    """Build the inspection report as a JSON-serializable dictionary."""
    source = source.resolve()
    file_hash = sha256_of(source)
    df = load_raw(source)

    missing_cols = [c for c in (ID_COLUMN, TARGET_COLUMN) if c not in df.columns]
    if missing_cols:
        first = str(df.columns[0])
        raise ValueError(
            f"columns {missing_cols} not found (pandas read {df.shape[1]} column(s), "
            f"first: {first[:40]!r}{'...' if len(first) > 40 else ''}).\n"
            f"  sha256 {file_hash}\n"
            f"  expected {EXPECTED_SHA256}\n"
            "  This is probably not the original CSV (re-saved by a spreadsheet?). "
            "Download the pinned original: see references/data_source.md."
        )

    missing_na = {c: int(v) for c, v in df.isna().sum().items()}
    missing_blank = blank_counts(df)

    target = df[TARGET_COLUMN]
    target_counts = {str(k): int(v) for k, v in target.value_counts(dropna=False).items()}
    target_props = {
        str(k): round(float(v), 6)
        for k, v in target.value_counts(normalize=True, dropna=False).items()
    }

    ids = df[ID_COLUMN]
    numeric_coercion = {}
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        coerced = pd.to_numeric(df[column], errors="coerce")
        numeric_coercion[column] = {
            "dtype_as_read": str(df[column].dtype),
            "non_numeric_values": int((coerced.isna() & df[column].notna()).sum()),
        }
    blank_total = df["TotalCharges"].astype("string").str.strip().eq("").fillna(False)
    numeric_coercion["TotalCharges"]["tenure_of_blank_rows"] = {
        str(k): int(v) for k, v in df.loc[blank_total, "tenure"].value_counts().items()
    }

    return {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(source.relative_to(PROJECT_ROOT)) if source.is_relative_to(PROJECT_ROOT) else str(source),
            "size_bytes": source.stat().st_size,
            "sha256": file_hash,
            "matches_expected_sha256": file_hash == EXPECTED_SHA256,
            "expected_sha256": EXPECTED_SHA256,
        },
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "target": {
            "column": TARGET_COLUMN,
            "counts": target_counts,
            "proportions": target_props,
        },
        "customer_id": {
            "column": ID_COLUMN,
            "unique": int(ids.nunique(dropna=False)),
            "duplicated_rows": int(ids.duplicated().sum()),
            "missing": int(ids.isna().sum()),
        },
        "missing_values": {
            "isna_per_column": missing_na,
            "blank_or_whitespace_per_column": missing_blank,
            "isna_total": int(sum(missing_na.values())),
            "blank_or_whitespace_total": int(sum(missing_blank.values())),
        },
        "numeric_coercion_check": numeric_coercion,
        "environment": environment_info(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect the raw Telco churn CSV.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.source.exists():
        sys.exit(f"Source file not found: {args.source}. See references/data_source.md.")

    try:
        report = inspect(args.source)
    except ValueError as error:
        sys.exit(f"ERROR: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Report written to {args.output}")
    print(f"Rows x columns : {report['shape']['rows']} x {report['shape']['columns']}")
    print(f"SHA-256        : {report['source']['sha256']} "
          f"(expected: {report['source']['matches_expected_sha256']})")
    print(f"Target counts  : {report['target']['counts']}")
    print(f"Duplicate IDs  : {report['customer_id']['duplicated_rows']}")
    print(f"NaN / blank    : {report['missing_values']['isna_total']} / "
          f"{report['missing_values']['blank_or_whitespace_total']}")
    if not report["source"]["matches_expected_sha256"]:
        print("WARNING: source hash differs from the documented original snapshot.")


if __name__ == "__main__":
    main()
