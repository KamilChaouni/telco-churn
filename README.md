# Telco Churn

**Business question:** which customers of a telecom operator are likely to leave
(`Churn = Yes`)? The company supplied a one-off customer snapshot (IBM Telco Customer
Churn, 7 043 customers, 21 fields). TP01 makes the project installable, identifies the
input data and inspects it. TP02 adds a reusable training Pipeline, a fair model comparison and a saved artifact (see "TP02" below).

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

## TP02 — Training pipeline, comparison and reusable artifact

### Commands

```sh
uv sync --locked                          # adds scikit-learn from uv.lock
uv run telco-train                        # split, first model, comparison, selection, save
uv run python scripts/check_reload.py     # fresh process: reload artifact, predict, compare
uv run pytest -q                          # focused reliability checks (section 5)
uv run telco-predict --input models/reference_batch.csv --output reports/tp02/example_predictions.csv
```

Everything is driven by [configs/train.json](configs/train.json): source + expected hash,
seed 42, split 60/20/20 stratified on `Churn`, features, preprocessing, threshold 0.5,
selection rule and the three candidates.

### Task and data rules

- Positive class: `Churn = Yes` → 1. `customerID` and `Churn` are never model inputs
  (`ColumnTransformer(remainder="drop")` only keeps the six features).
- Features: numeric `tenure`, `MonthlyCharges`, `TotalCharges`; categorical `Contract`,
  `InternetService`, `PaymentMethod`.
- Stateless parsing before the pipeline (`telco_churn.features.prepare_features`), same at
  training and inference: blank/whitespace numeric text → NaN (the 11 `TotalCharges = " "`);
  any other non-numeric text → **error** (never silently NaN); missing column → **error**.
- The raw CSV is never modified.

### Pipeline (`src/telco_churn/pipeline.py`)

```
Pipeline
├── preprocessor: ColumnTransformer
│   ├── num: SimpleImputer(median) → StandardScaler        [tenure, MonthlyCharges, TotalCharges]
│   └── cat: SimpleImputer(constant "missing") → OneHotEncoder(handle_unknown="ignore")
└── classifier
```

One `fit` on the training rows learns medians, means/stds, categories and the model.
Validation, test and new customers only go through `transform`/`predict`.
An unseen category is encoded as all zeros: the call works, but such predictions are not
validated by any data.

### Results (validation, threshold 0.5) — `reports/tp02/`

| model | avg precision | precision | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|
| majority baseline | 0.265 | 0.000 | 0.000 | 0.000 | 0.735 |
| logistic regression | 0.648 | 0.664 | 0.529 | 0.589 | 0.804 |
| decision tree (depth 5) | 0.617 | 0.636 | 0.602 | 0.618 | 0.803 |
| **random forest (300 trees)** | **0.662** | 0.646 | 0.479 | 0.550 | 0.792 |

- First model (milestone 1): logistic regression on `tenure` + `MonthlyCharges` only →
  accuracy 0.783, recall 0.409 (`reports/tp02/first_model.json`). Not comparable as an
  "algorithm improvement": the feature set changed too.
- Selection rule fixed before comparing: highest validation **average precision**
  (ranking quality for the minority churn class, threshold-free); within 0.01 keep the simpler
  model. Random forest wins by 0.014 → selected.
- The test split is used once, after selection, for the chosen model only: AP 0.621.

### Artifact

| File | Content |
|---|---|
| `models/churn_pipeline.joblib` | fitted preprocessing **+** classifier (joblib/pickle: load trusted files only) |
| `models/churn_pipeline.metadata.json` | source hash, seed, split sizes, features, preprocessing, learned statistics, params, metrics, threshold, Git commit, library versions, artifact hash |
| `models/split_assignment.csv` | which customer is in train / valid / test |
| `models/reference_batch.csv` | 20 validation customers (no label) + original scores, for the reload check |

Load it with the same library versions (`uv sync --locked`). Predicting new customers needs
only the six feature columns, no `Churn`.

### Reliability evidence

`uv run pytest -q` checks: disjoint and reproducible splits; imputer medians equal the
training medians (not the full-data ones); predicting does not change them; blank/NaN
`TotalCharges` and unseen categories follow the policy; malformed numbers and missing columns
fail visibly; a fresh Python process reloads the artifact and gets identical scores.
`scripts/check_reload.py` writes `reports/tp02/reload_check.json`.

### Limits

Threshold 0.5 is not tuned to a business cost; probabilities are not checked for calibration;
median imputation of `TotalCharges` for `tenure = 0` customers is a simplification (0 would be
more literal); timings are descriptive only; a pickle artifact depends on compatible library
versions. Experiment tracking and model versioning come in TP03.
