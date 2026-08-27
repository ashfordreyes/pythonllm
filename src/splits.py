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

Callers may additionally pass `priority`: a set of row indices to stratify
into their own train/eval split, merged with the rest afterwards. This exists
so the execution-scoring tier isn't left permanently dormant -- of the 50
reference snippets only 4 are safe to `exec()` (see
`src/eval/scoring.is_executable_example`), and an unstratified shuffle has no
way to know that, so under `SPLIT_SEED` none of them happened to land in eval.
Stratifying by that fact is not the same as trying seeds until the draw looks
good: `is_executable_example` is a static property of the *reference* code,
checked once, before any split is drawn and long before any model generates
anything -- it never depends on a generation or a score.
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


def _split_group(indices: list[int], seed: int, eval_fraction: float) -> tuple[list[int], list[int]]:
    """Shuffle-then-slice one pool of indices under `seed`.

    Mirrors `split_indices`'s single-pool logic exactly, factored out so
    stratification can apply it once per stratum. Groups smaller than 2 stay
    entirely in train -- there's nothing sensible to hold out of a singleton.
    """
    if len(indices) < 2:
        return list(indices), []

    n_eval = max(1, round(len(indices) * eval_fraction))
    n_eval = min(n_eval, len(indices) - 1)

    order = list(indices)
    random.Random(seed).shuffle(order)
    return order[n_eval:], order[:n_eval]


def split_indices(
    n: int,
    seed: int = SPLIT_SEED,
    eval_fraction: float = EVAL_FRACTION,
    priority: set[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Return (train_indices, eval_indices), both sorted ascending.

    The two lists are disjoint and their union is exactly range(n). At least
    one example is always held out for any n >= 2.

    If `priority` is given, `range(n)` is partitioned into `priority` and
    everything else, each partition is split independently (same `seed`,
    same `eval_fraction`), and the results are merged. This guarantees a
    proportional share of `priority` rows lands in eval instead of leaving
    it to chance -- see the module docstring for why that's not "peeking."
    """
    if n < 2:
        raise ValueError(f"need at least 2 examples to split, got {n}")

    n_eval = max(1, round(n * eval_fraction))
    if n_eval >= n:
        raise ValueError(
            f"eval_fraction={eval_fraction} would hold out {n_eval} of {n} "
            "examples, leaving nothing to train on"
        )

    if not priority:
        order = list(range(n))
        random.Random(seed).shuffle(order)
        return sorted(order[n_eval:]), sorted(order[:n_eval])

    priority_group = sorted(i for i in priority if 0 <= i < n)
    rest_group = [i for i in range(n) if i not in set(priority_group)]

    train, ev = [], []
    for group in (priority_group, rest_group):
        t, e = _split_group(group, seed, eval_fraction)
        train += t
        ev += e
    return sorted(train), sorted(ev)


def load_split(
    path: str | Path,
    which: str,
    seed: int = SPLIT_SEED,
    eval_fraction: float = EVAL_FRACTION,
    priority: set[int] | None = None,
) -> list[dict]:
    """Read a JSONL file and return only its train or eval rows."""
    rows = read_jsonl(path)
    train_idx, eval_idx = split_indices(len(rows), seed, eval_fraction, priority)
    if which == TRAIN:
        wanted = train_idx
    elif which == EVAL:
        wanted = eval_idx
    else:
        raise ValueError(f"which must be {TRAIN!r} or {EVAL!r}, got {which!r}")
    return [rows[i] for i in wanted]


def split_dataset(
    dataset,
    seed: int = SPLIT_SEED,
    eval_fraction: float = EVAL_FRACTION,
    priority: set[int] | None = None,
):
    """Split a HuggingFace Dataset into (train, eval) using the same indices.

    Kept separate from `load_split` so the notebooks can go on using
    `datasets.load_dataset(...)` rather than hand-rolling a Dataset.
    """
    train_idx, eval_idx = split_indices(len(dataset), seed, eval_fraction, priority)
    return dataset.select(train_idx), dataset.select(eval_idx)
