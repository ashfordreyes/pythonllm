"""Evaluation for the two-stage pipeline: scoring, execution, and charts.

Three layers, deliberately separate:

- `harness` — runs a generated snippet in a fresh namespace and reports
  pass/fail against a caller-supplied check. Execution-based, DS-1000 style.
- `scoring` — layered metrics (plan validity, verb sequence, code syntax,
  execution) so every held-out example produces signal. Most of this dataset's
  snippets cannot be executed here at all, so execution is one tier among
  several rather than the only score.
- `plots` — PNG charts written to disk for the notebooks and eval runs.

`plots` imports matplotlib at module load; the other two do not, so a caller
that only needs scoring never pays for it.
"""

from .harness import EvalResult, run_and_check
from .scoring import (
    CodeScore,
    PlanScore,
    aggregate,
    classify_plan_error,
    describe_blockers,
    is_executable_example,
    score_code,
    score_plan,
)

__all__ = [
    "EvalResult",
    "run_and_check",
    "PlanScore",
    "CodeScore",
    "score_plan",
    "score_code",
    "is_executable_example",
    "describe_blockers",
    "classify_plan_error",
    "aggregate",
]
