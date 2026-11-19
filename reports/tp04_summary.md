# TP04 — Testing and code quality summary

Tools (from `uv.lock`): pytest 9.1.1, pytest-cov 7.1.0, Ruff 0.17.5, pre-commit 4.7.0.
Raw logs: [evidence/tp04/](../evidence/tp04/).

## 1. Input/output contract

1. The six TP02 features (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`,
   `InternetService`, `PaymentMethod`) are required; a missing one raises
   `ValueError("Missing required feature column(s): [...]")`. `Churn` and `customerID` are never inputs.
2. Numeric parsing (`prepare_features`): blank/whitespace text → NaN (permitted); other non-numeric
   text (`"12,50"`) → `ValueError`.
3. Permitted missing numeric → training median; unseen category → all-zero one-hot
   (`handle_unknown="ignore"`), so the call succeeds.
4. A nonempty batch gets exactly one score and one 0/1 prediction per customer, same order and IDs;
   `predict_proba` has 2 columns, class order `classes_ == [0, 1]` (0 = No, 1 = Churn Yes).
5. A saved and reloaded fitted pipeline gives the same predictions.

**Four realistic defects the checks must catch:** (a) a colleague removes the column check and a
renamed column is silently scored; (b) the unknown-category policy is switched to `"error"`, so a
new payment method crashes production scoring; (c) the positive score is taken as `proba[:, 1]`
without checking `classes_`, or the threshold is applied the wrong way; (d) only the classifier
is saved, so the reloaded artifact cannot read raw columns.

## 2. Tests and the risk each one covers

Fixture: `tests/conftest.py`, 16 invented customers (7 Yes / 9 No), raw input types (`TotalCharges`
as text), fitted with the project's own `build_pipeline` + `LogisticRegression(random_state=0)`.
`"Credit card (automatic)"` is deliberately absent from the fixture so it is truly unseen.

| Test | File | Risk / defect detected |
|---|---|---|
| **A** `test_missing_required_feature_is_rejected` ×6 (parametrized) | `test_input_contract.py` | column check removed → silent scoring of incomplete data; asserts `ValueError` + message naming the column |
| `test_label_and_id_are_not_inputs` | `test_input_contract.py` | target/ID leakage: flipping `Churn` or changing IDs must not change scores |
| **B** `test_missing_numeric_and_unknown_category_follow_policy` | `test_input_contract.py` | imputer removed, blank text not treated as missing, `handle_unknown` changed; checks category was unseen, NaN row encoded exactly as the train median, unseen category → zero indicators |
| `test_blank_versus_malformed_numeric_text` | `test_input_contract.py` | `errors="coerce"` hiding malformed values as NaN |
| **C** `test_probability_contract` | `test_prediction.py` | shape `(n, 2)`, finite, in [0, 1], rows sum to 1 (atol 1e-9), positive score = column of class 1 in `classes_` |
| `test_positive_score_uses_classes_not_column_position` | `test_prediction.py` | hard-coded `[:, 1]`: a model with `classes_ = [1, 0]` must still return the class-1 column |
| `test_one_row_per_customer_and_threshold` ×3 | `test_prediction.py` | rows dropped/reordered, IDs misaligned, threshold ignored or inverted |
| `test_prediction_does_not_refit` | `test_prediction.py` | inference calling `fit` and relearning the imputer medians |
| **D** `test_reload_preserves_predictions` | `test_persistence.py` | partial artifact / lost fitted state; joblib to `tmp_path`, reload with `load_pipeline`, labels equal and scores within 1e-12 on raw columns |

Result: `uv run pytest -q` → **27 passed in ~2 s** (16 new synthetic tests + 11 earlier ones).
No exact accuracy/AUC is asserted on the tiny fixture: valid probabilities say nothing about
calibration or quality.

**Does a successful prediction on an unknown category prove it is accurate?** No. It only proves
the policy is applied: the customer is scored as if they had *none* of the known payment methods,
a combination never seen in training. The score is plausible-looking but not supported by data.

## 3. Regression demonstration ([evidence/tp04/regression_demo.txt](../evidence/tp04/regression_demo.txt))

Change (a realistic "make it stricter" edit) in `src/telco_churn/pipeline.py`:
`unknown_category: str = "ignore"` → `"error"`.

`uv run pytest -q` → **3 failed, 24 passed**:
`test_missing_numeric_and_unknown_category_follow_policy` (new, Test B) failed with
`ValueError: Found unknown categories ['Credit card (automatic)'] in column 2 during transform`,
plus the two TP02 tests that also send an unseen `Contract`. The failure is in `predict()`,
i.e. exactly where a production batch with a new payment method would crash.

Repair: restored `"ignore"` (`git checkout -- src/telco_churn/pipeline.py`, empty diff) →
**27 passed**.

## 4. Ruff and coverage

Configured in `pyproject.toml` (`[tool.ruff]` line length 100, py311; `[tool.ruff.lint]`
`select = ["E4", "E7", "E9", "F", "I"]`: import-level pycodestyle errors, statement errors,
syntax/IO errors, pyflakes, isort). Scope: `src/telco_churn tests scripts`.

- Before ([ruff_before.txt](../evidence/tp04/ruff_before.txt)): **6 findings** — 3× I001 unsorted
  imports (incl. an `import sys` placed after the stdlib block in `tracking.py` and
  `registry_inference.py`), 3× F541 f-strings without placeholders (`training.py`, `tracking.py`);
  formatter: 12 files would be reformatted.
- Fixes reviewed in `git diff` (import order, `f` prefix removal, line wrapping only), then the now
  useless `# noqa: E402` removed by hand. Nothing ignored or excluded.
- After ([ruff_after.txt](../evidence/tp04/ruff_after.txt)): `All checks passed!`,
  `18 files already formatted`, tests still 27 passed.

Coverage ([coverage.txt](../evidence/tp04/coverage.txt)), with branch coverage:
**36 % total** (517 statements). `features.py` 96 %, `pipeline.py` 79 %, `prediction.py` 65 %,
`evaluation.py` 67 %, `training.py` 36 %, `dataset.py` 52 %, `tracking.py` and
`registry_inference.py` 0 % (they need a running MLflow server; the tests deliberately do not
start one). Coverage's C tracer is blocked by Smart App Control here, so the slower Python tracer
was used — same results.

**Uncovered relevant behaviour:** `features.py:61`, the guard in `encode_target` that rejects an
unexpected target label. If someone removed it, a CSV with `"yes"`/`"Churned"`/blank labels would
be mapped to NaN or silently miscounted during training. **Proposed test:**
`encode_target(pd.Series(["Yes", "No", "yes"]))` and `pd.Series(["Yes", None])` must both raise
`ValueError(match="Unexpected target values")`, and `["Yes", "No"]` must give `[1, 0]`.
(Similarly `pipeline.py:27-29`: an estimator name outside the allow-list must be refused.)

**What does an executed line tell you?** Only that some test ran it. A test can execute a line
without asserting anything about its result (e.g. calling `predict` and only checking it returns),
so 100 % coverage is not correctness. Third-party code (scikit-learn, pandas, MLflow) is outside
the scope (`[tool.coverage.run] source = ["telco_churn"]`); we test how we *use* it, not the library.

## 5. pre-commit

[.pre-commit-config.yaml](../.pre-commit-config.yaml): `astral-sh/ruff-pre-commit` **v0.17.5**
(same as the CLI) with `ruff-check --fix` then `ruff-format`, scoped to `src|tests|scripts`;
`pre-commit/pre-commit-hooks` **v6.0.0**: trailing whitespace, end-of-file, check-yaml/toml,
large files (> 3 MB). Data, models, reports, evidence, `.venv*`, `mlartifacts/`, `mlruns/` excluded.
On this machine the hook console launchers (`*.exe`) are blocked by Smart App Control, so those
hooks' `entry` is overridden to `python -m pre_commit_hooks.<module>` (same pinned code, portable).

`validate-config` OK → `install` → `run --all-files`: all hooks Passed.

Blocked commit ([precommit_demo.txt](../evidence/tp04/precommit_demo.txt)): a real docstring
added to `evaluate()` with trailing spaces and `positive_scores( model,X )`. Attempt 1:
`ruff format ... Failed — files were modified by this hook`; HEAD unchanged. `git diff` reviewed
(formatting only), file re-staged, attempt 2: all hooks Passed → commit `320dfd5`. `--no-verify`
was never used.

## 6. Quality Gate reflection (SonarQube not run — optional)

- **Tests** check behaviour we thought of. **Static analysis** (Ruff, SonarQube) finds suspicious
  code patterns without running it. **Coverage** measures what tests executed, not what they
  verified. A **Quality Gate** is a *policy* over those results (e.g. "no new bugs, ≥ 80 % coverage
  on new code"); "analysis completed" and "gate passed" are separate outcomes.
- **If tests pass, what can still be wrong?** Everything no test asks about: wrong business
  threshold, data drift, a bad label in a new snapshot, unrealistic fixture, poor calibration,
  behaviour on the real distribution.
- **Does a passing gate prove the model is useful, fair or secure?** No. It says the code meets
  the configured rules. Model usefulness needs validation metrics against a business objective;
  fairness needs group-level analysis; security needs dependency/secret scanning and review
  (TP05). Local hooks can also be bypassed, which is why TP05 repeats the checks in CI.
