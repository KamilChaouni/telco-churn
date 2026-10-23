# PR: docs/project-onboarding → main

## Title
docs: onboarding README so a new teammate can install, get the data and run the inspection

## What changes
- README rewritten: business question, quick start (`uv sync --locked` → download + hash → `uv run telco-inspect`),
  expected output, report fields, environment table, layout and five location choices,
  TP02 locations, Git policy, known limits.
- Inspection error for a wrong CSV is now actionable: shows received vs expected SHA-256
  and points to `references/data_source.md` (found during handover dry run).

## Verification
Done from a fresh `git clone` of this branch in an empty directory, following only the README:
- `uv sync --locked` → OK (Python 3.12.13, pandas 3.1.0)
- downloaded CSV hash = `16320c9c…055e91` (matches)
- `uv run telco-inspect` → 7043 × 21, No 5174 / Yes 1869, 0 duplicate IDs, 0 NaN / 11 blank
- `uv run pytest -q` → 3 passed
- with the class copy `datasets.csv` → clear error instead of a stack trace

## Question for the reviewer
The raw CSV is kept out of Git and fetched with `curl` from a commit-pinned URL.
Is that enough for you on Windows / without network, or should we add a
`telco-download` command that downloads **and** verifies the hash in one step?

## Review discussion (to be filled on GitHub)
- Reviewer comment: _…_
- Our response: _…_
