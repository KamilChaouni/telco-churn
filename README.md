# Telco Churn

**Business question:** which customers of a telecom operator are likely to leave
(`Churn = Yes`)? The company supplied a one-off customer snapshot (IBM Telco Customer
Churn, 7 043 customers, 21 fields). TP01 makes the project installable, identifies the
input data and inspects it. TP02 will add preprocessing and training Pipelines here.

## Quick start (new teammate)

Prerequisites: Git and [uv](https://docs.astral.sh/uv/) (uv installs Python 3.12 itself if needed).

```sh
git clone <repo-url> telco-churn
cd telco-churn

# 1. Environment: exact versions from uv.lock, fails instead of re-resolving
uv sync --locked

# 2. Raw data (not in Git) — download the pinned original, then verify the hash
mkdir -p data/raw
curl -L -o data/raw/telco.csv "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
# expected SHA-256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91

# 3. Inspection report
uv run telco-inspect
```

Windows: `curl.exe` works the same in PowerShell; hash with `Get-FileHash data/raw/telco.csv`.
If you received the CSV from the instructor instead, check its hash first — see
[references/data_source.md](references/data_source.md): one class copy was re-saved
by a spreadsheet and is refused by the inspection.

Expected terminal output (the report is written to `reports/data_overview.json`):

```
Rows x columns : 7043 x 21
SHA-256        : 16320c9c...055e91 (expected: True)
Target counts  : {'No': 5174, 'Yes': 1869}
Duplicate IDs  : 0
NaN / blank    : 0 / 11
```

Other commands:

| Command | Purpose |
|---|---|
| `uv run telco-inspect --source <csv> --output <json>` | Inspect another file / write elsewhere |
| `uv run pytest -q` | Focused tests of the inspection logic (no raw data needed) |
| `uv run python scripts/compare_reports.py A.json B.json` | Compare two reports, ignoring timestamps/paths |
| `uv run python -c "import telco_churn; print(telco_churn.__file__)"` | Check the package imports from the project env |

## What the inspection report contains

`reports/data_overview.json` (machine-readable, regenerated on each run, not committed):

- `source`: relative path, size, SHA-256, and whether it matches the documented hash
- `shape`, `dtypes`: dimensions and the types pandas inferred
- `target`: `Churn` counts and proportions (≈ 73.5 % No / 26.5 % Yes)
- `customer_id`: unique count, duplicated rows, missing
- `missing_values`: `isna()` per column **and** empty/whitespace-only strings per column —
  they differ: `TotalCharges` has 0 NaN but 11 values equal to `" "` (all with `tenure = 0`)
- `numeric_coercion_check`: why `TotalCharges` is read as text
- `environment`: Python version, platform, pandas/numpy/package versions

The program only inspects: no row is dropped, imputed or split, and the raw file is read-only.

## Environment

| | |
|---|---|
| Python | 3.12.13 (pinned by `.python-version`; `requires-python = ">=3.11,<3.13"`) |
| Manager | uv 0.11.32, project created with `uv init --package` (src layout, Option B — no CCDS) |
| Runtime deps | `pandas` (3.1.0 locked → numpy 2.5.4) |
| Dev deps | `pytest` (dev group) |

- `pyproject.toml` — what we *ask for* (ranges, metadata, entry point `telco-inspect`).
- `uv.lock` — the exact *resolved* versions + hashes for every platform; committed.
- `.venv/` — the local installed result; disposable, never committed or copied.

Recreation was tested: the `.venv` was set aside, rebuilt with `uv sync --locked`, and the
report regenerated — identical package list and 0 differences on 97 report fields
(see `evidence/tp01/`).

## Project layout

```
telco-churn/
├── README.md                 setup, data, commands, handover
├── pyproject.toml, uv.lock   dependency declaration / resolved lock
├── .python-version           interpreter for uv
├── .gitignore                env, credentials, data and generated outputs policy
├── src/telco_churn/          importable package (reusable code)
│   └── dataset.py            raw reading + inspection, CLI `telco-inspect`
├── scripts/compare_reports.py thin helper for recreation evidence
├── tests/                    focused behaviour checks
├── data/raw/telco.csv        original snapshot, immutable, NOT in Git
├── references/data_source.md provenance, hash, field roles and notes
├── reports/                  generated reports (ignored by Git)
└── evidence/tp01/            tracked, small submission evidence
```

Five location choices:
1. The CSV goes in `data/raw/` and is never edited: everything downstream is recomputable from it.
2. Inspection code lives in the package (`src/telco_churn/dataset.py`), not in a notebook, so it is importable, testable and runnable by one command.
3. Generated reports go to `reports/` and are ignored — they are reproducible outputs; the copies needed for assessment are deliberately kept in `evidence/tp01/`.
4. Provenance (URL, commit, hash, incident) is documentation → `references/`.
5. Tests sit in `tests/` and use tiny synthetic data, so they run without the raw CSV.

Added in TP02 (not created yet, to avoid empty placeholders):
`data/interim/`, `data/processed/` (persisted transformed data, only if useful),
`configs/train.json` (hyperparameters), `src/telco_churn/pipeline.py` / `training.py` /
`prediction.py` (model-building **code**), `models/` (fitted pipelines saved with joblib —
**outputs**, ignored by Git), `notebooks/` (exploration only), `reports/figures/`.

## Git policy

- Not versioned: `.venv*`, credentials, everything under `data/`, `reports/`, `models/`.
- Versioned: code, lock file, docs, and `evidence/tp01/` (small, non-sensitive).
- Changes go through a branch + pull request (e.g. `docs/project-onboarding`).

## Known limits

What another engineer can reproduce now: the same environment (package versions), the
same input bytes (by hash) and the same inspection statistics.
Not yet covered: OS/hardware-level differences, a trained model. Reproducing a model will
additionally require the training config and seed, the split definition, the code commit,
library versions at training time, the fitted artifact with its hash and its metrics.

See [evidence/tp01/](evidence/tp01/) for the recreation comparison, PR review and handover notes.
