"""Deterministic train/eval split shared by both stages.

The Stage 1 and Stage 2 JSONL files are line-aligned 1:1 -- line N's
`pseudocode` is byte-identical in both (an invariant `scripts/check_data.py`
enforces). Holding out examples therefore has to be done by *row index*, using
the same indices for both files, or end-to-end eval would score Stage 1 and
Stage 2 on different tasks.

Splitting by index from a fixed seed keeps that property by construction: no
extra files to keep in sync, and no split metadata mixed into the training
payload. `SPLIT_SEED` is part of the contract -- changing it silently changes
which examples the model has seen.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SPLIT_SEED = 0
EVAL_FRACTION = 0.2

TRAIN = "train"
EVAL = "eval"


def read_jsonl(path: str | Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts, skipping blank lines.

    Raises json.JSONDecodeError with the offending line number in the message.
    """
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"{path}:{i}: {e.msg}", e.doc, e.pos) from None
    return rows


def split_indices(
    n: int,
    seed: int = SPLIT_SEED,
    eval_fraction: float = EVAL_FRACTION,
) -> tuple[list[int], list[int]]:
    """Return (train_indices, eval_indices), both sorted ascending.

    The two lists are disjoint and their union is exactly range(n). At least
    one example is always held out for any n >= 2.
    """
    if n < 2:
        raise ValueError(f"need at least 2 examples to split, got {n}")

    n_eval = max(1, round(n * eval_fraction))
    if n_eval >= n:
        raise ValueError(
            f"eval_fraction={eval_fraction} would hold out {n_eval} of {n} "
            "examples, leaving nothing to train on"
        )

    order = list(range(n))
    random.Random(seed).shuffle(order)
    return sorted(order[n_eval:]), sorted(order[:n_eval])


def load_split(
    path: str | Path,
    which: str,
    seed: int = SPLIT_SEED,
    eval_fraction: float = EVAL_FRACTION,
) -> list[dict]:
    """Read a JSONL file and return only its train or eval rows."""
    rows = read_jsonl(path)
    train_idx, eval_idx = split_indices(len(rows), seed, eval_fraction)
    if which == TRAIN:
        wanted = train_idx
    elif which == EVAL:
        wanted = eval_idx
    else:
        raise ValueError(f"which must be {TRAIN!r} or {EVAL!r}, got {which!r}")
    return [rows[i] for i in wanted]


def split_dataset(dataset, seed: int = SPLIT_SEED, eval_fraction: float = EVAL_FRACTION):
    """Split a HuggingFace Dataset into (train, eval) using the same indices.

    Kept separate from `load_split` so the notebooks can go on using
    `datasets.load_dataset(...)` rather than hand-rolling a Dataset.
    """
    train_idx, eval_idx = split_indices(len(dataset), seed, eval_fraction)
    return dataset.select(train_idx), dataset.select(eval_idx)
