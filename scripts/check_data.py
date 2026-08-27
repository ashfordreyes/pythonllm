#!/usr/bin/env python3
"""Validate the stage1/stage2 JSONL datasets.

Run: python scripts/check_data.py

Checks:
  - each line in both files is valid JSON with the expected keys
  - stage1 and stage2 have the same number of lines
  - the pseudocode at line N in stage1 matches line N in stage2 (1:1 pairing)
  - every pseudocode plan passes src.dsl.validator.validate_plan

Also reports the train/eval split the notebooks will resolve, and how many
held-out examples the execution tier can actually score. That number is
usually small (most reference snippets need a display, the network, or data
files that aren't in the repo), so it is worth seeing next to the data check
rather than discovering it mid-eval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsl.validator import validate_plan  # noqa: E402
from splits import EVAL_FRACTION, SPLIT_SEED, read_jsonl, split_indices  # noqa: E402

STAGE1_PATH = ROOT / "data" / "stage1_planner" / "englishtopseudo.jsonl"
STAGE2_PATH = ROOT / "data" / "stage2_coder" / "pseudotopython.jsonl"


def load_jsonl(path: Path, required_keys: set[str]) -> list[dict]:
    """Read a JSONL file and check every row carries the keys this stage needs.

    Parsing is `splits.read_jsonl` so this script and the notebooks agree on
    what a row is; the key check and the SystemExit reporting stay here.
    """
    try:
        rows = read_jsonl(path)
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid JSON: {e}")
    for i, row in enumerate(rows, start=1):
        missing = required_keys - row.keys()
        if missing:
            raise SystemExit(f"{path}:{i}: missing keys {missing}")
    return rows


def report_split(stage2: list[dict]) -> None:
    """Print the resolved split plus the execution tier's real coverage."""
    train_idx, eval_idx = split_indices(len(stage2))
    print(
        f"split (seed={SPLIT_SEED}, eval_fraction={EVAL_FRACTION}): "
        f"{len(train_idx)} train / {len(eval_idx)} eval"
    )
    print("  held out (1-based lines): " + ", ".join(str(i + 1) for i in eval_idx))

    try:
        from eval.scoring import is_executable_example
    except ImportError as e:  # scoring has no third-party deps, but be explicit
        print(f"  (skipped execution-coverage check: {e})")
        return

    runnable = {i for i, row in enumerate(stage2) if is_executable_example(row["python_code"])}
    in_eval = sorted(runnable & set(eval_idx))
    print(
        f"  execution tier: {len(runnable)}/{len(stage2)} references run cleanly here, "
        f"{len(in_eval)} of them held out"
    )
    if not in_eval:
        print("  note: no held-out example is executable, so that tier reports "
              "'not attempted' rather than a rate")


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
    report_split(stage2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
