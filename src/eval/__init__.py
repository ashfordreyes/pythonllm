"""Execution-based evaluation harness for the Stage 2 (pseudocode -> Python) coder.

Given a generated Python snippet and a reference check (a callable that inspects
the resulting namespace, e.g. asserting a variable's value or shape), this runs
the snippet in a restricted namespace and reports pass/fail. This is the same
style of check used by DS-1000: "does the generated code actually produce the
expected result," not just "does it look right."
"""

from .harness import EvalResult, run_and_check

__all__ = ["EvalResult", "run_and_check"]
