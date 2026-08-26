"""Sandboxed execution + check harness.

NOTE: this uses plain exec() in-process, which is fine for trusted, locally
authored eval cases but is NOT a security sandbox. Do not run untrusted or
model-generated-and-unreviewed code with this against a machine that has
access to anything sensitive -- run it in an isolated container/VM when
evaluating at scale (e.g. from a Colab or CI runner with no credentials).
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable


@dataclass
class EvalResult:
    passed: bool
    error: str | None = None


def run_and_check(code: str, check: Callable[[dict], bool], timeout_note: str = "") -> EvalResult:
    """Execute `code` in a fresh namespace, then call `check(namespace) -> bool`.

    `check` should inspect the namespace produced by `code` (e.g. a DataFrame's
    shape, a model's accuracy) and return True/False.
    """
    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102 -- intentional; see module docstring
    except Exception:
        return EvalResult(passed=False, error=traceback.format_exc())

    try:
        ok = check(namespace)
    except Exception:
        return EvalResult(passed=False, error=traceback.format_exc())

    return EvalResult(passed=bool(ok))
