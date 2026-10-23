# Handover record — TP01

## Environment recreation (section 4)

1. First environment recorded: `env_before.txt`, `report_before.json`.
2. `.venv` moved aside (`mv .venv .venv.bak-tp01`, ignored by Git), then `uv sync --locked`
   (`recreate_log.txt`). Lock was consistent, nothing re-resolved.
3. Report regenerated: `env_after.txt`, `report_after.json`.
4. `diff env_before.txt env_after.txt` → identical (Python 3.12.13, pandas 3.1.0, numpy 2.5.4, …).
5. `scripts/compare_reports.py` → 97 fields, 0 unexpected differences (`comparison.txt`).
   Only `generated_at_utc` and the interpreter path string differ — expected and ignored.
   Same SHA-256, same shape, dtypes, target counts, missing values.

## Handover dry run (section 6)

Run by: _(name of the other pair)_ — date: _…_
Setup: fresh clone in an empty directory, README only, no oral help.

| Step | Result |
|---|---|
| `uv sync --locked` | OK |
| Download + hash check | OK, hash matches |
| `uv run telco-inspect` | OK, report written |
| `uv run pytest -q` | 3 passed |

### Issue found
Using the CSV distributed in class (`datasets.csv`, re-saved by a spreadsheet: BOM, CRLF,
each line quoted) made pandas read a single column. The first version stopped with an
error message that dumped the whole header and did not say what to do.

### Fix
Commit `Make the wrong-file error actionable`: the error now shows the received and expected
SHA-256 and points to `references/data_source.md`; the README tells users to check the hash
of any instructor-supplied copy first.

### Did the revised instructions work?
Yes: re-run from a clean clone, the pinned download passes and the class copy is refused
with an explicit message. _(Confirm with the other pair: …)_
