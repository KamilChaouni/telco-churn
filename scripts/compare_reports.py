"""Compare two inspection reports, ignoring fields expected to differ between runs.

Usage: uv run python scripts/compare_reports.py evidence/tp01/report_before.json evidence/tp01/report_after.json
"""

import json
import sys

VOLATILE = {"generated_at_utc", "python_executable"}


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from flatten(value, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, obj


def main(before_path: str, after_path: str) -> int:
    before = dict(flatten(json.load(open(before_path, encoding="utf-8"))))
    after = dict(flatten(json.load(open(after_path, encoding="utf-8"))))
    differences = 0
    for key in sorted(before.keys() | after.keys()):
        if key.split(".")[-1] in VOLATILE:
            status = "ignored (volatile)"
        elif before.get(key) == after.get(key):
            continue
        else:
            status = "DIFFERENT"
            differences += 1
        print(f"{status:20} {key}: {before.get(key)!r} -> {after.get(key)!r}")
    print(f"\n{len(before)} fields compared, {differences} unexpected difference(s).")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
