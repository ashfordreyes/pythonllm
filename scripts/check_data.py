#!/usr/bin/env python3
"""Validate the stage1/stage2 JSONL datasets.

Run: python scripts/check_data.py

Checks:
  - each line in both files is valid JSON with the expected keys
  - stage1 and stage2 have the same number of lines
  - the pseudocode at line N in stage1 matches line N in stage2 (1:1 pairing)
  - every pseudocode plan passes src.dsl.validator.validate_plan
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsl.validator import validate_plan  # noqa: E402

STAGE1_PATH = ROOT / "data" / "stage1_planner" / "englishtopseudo.jsonl"
STAGE2_PATH = ROOT / "data" / "stage2_coder" / "pseudotopython.jsonl"


def load_jsonl(path: Path, required_keys: set[str]) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i}: invalid JSON: {e}")
        missing = required_keys - row.keys()
        if missing:
            raise SystemExit(f"{path}:{i}: missing keys {missing}")
        rows.append(row)
    return rows


def main() -> int:
    stage1 = load_jsonl(STAGE1_PATH, {"english", "pseudocode"})
    stage2 = load_jsonl(STAGE2_PATH, {"pseudocode", "python_code"})

    errors = []

    if len(stage1) != len(stage2):
        errors.append(
            f"line count mismatch: {STAGE1_PATH.name} has {len(stage1)}, "
            f"{STAGE2_PATH.name} has {len(stage2)}"
        )

    for i, (s1, s2) in enumerate(zip(stage1, stage2)):
        if s1["pseudocode"] != s2["pseudocode"]:
            errors.append(f"line {i + 1}: pseudocode differs between the two files")

    for i, row in enumerate(stage1):
        plan_errors = validate_plan(row["pseudocode"])
        for err in plan_errors:
            errors.append(f"{STAGE1_PATH.name}:{i + 1}: {err}")

    if errors:
        print(f"FAILED: {len(errors)} issue(s)\n")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK: {len(stage1)} pairs in {STAGE1_PATH.name}, all validated and aligned with {STAGE2_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
