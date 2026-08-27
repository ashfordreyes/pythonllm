"""Layered scoring for Stage 1 / Stage 2 generations.

Execution alone is not a usable metric for this dataset. Of the 50 reference
snippets, most need the network, a display (tkinter), filesystem fixtures, or
heavy ML dependencies, and `harness.run_and_check` needs a per-example
`check(namespace)` callable that no example has. Scoring one tier would leave
a 10-example held-out set with almost nothing to report.

So each generation is scored on several tiers, cheapest first, and every tier
records its own denominator -- how many examples it was *attempted* on. A
snippet that was never safe to run is "not attempted", not "failed".
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field

from dsl.validator import ValidationError, parse_plan, validate_plan

from .harness import run_and_check

# Top-level module names that make a snippet unsafe or impossible to exec()
# in-process: no network, no window that blocks on a mainloop, no dependency
# we cannot assume is installed. Filesystem modules are excluded because the
# dataset's file paths are illustrative and do not exist.
NETWORK_MODULES = frozenset(
    {"requests", "urllib", "urllib2", "http", "socket", "flask", "aiohttp",
     "httpx", "ftplib", "smtplib", "telnetlib", "xmlrpc"}
)
GUI_MODULES = frozenset({"tkinter", "Tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "kivy", "pygame", "wx"})
FILESYSTEM_MODULES = frozenset({"os", "pathlib", "shutil", "glob", "tempfile", "watchdog", "zipfile", "tarfile"})
HEAVY_MODULES = frozenset(
    {"torch", "torchvision", "tensorflow", "keras", "sklearn", "transformers",
     "datasets", "PIL", "joblib", "xgboost", "lightgbm", "matplotlib", "cv2", "tqdm"}
)

BLOCKING_CALLS = frozenset({"open", "input", "exec", "eval", "compile", "__import__"})


def _normalize(text: str) -> str:
    return " ".join(text.split())


# --- Stage 1: plans -------------------------------------------------------


@dataclass
class PlanScore:
    well_formed: bool
    verb_sequence_match: bool
    exact_match: bool
    errors: list[str] = field(default_factory=list)
    generated_verbs: list[str] = field(default_factory=list)
    reference_verbs: list[str] = field(default_factory=list)


def _verbs(text: str) -> list[str]:
    try:
        return [step.verb for step in parse_plan(text)]
    except ValidationError:
        return []


def score_plan(generated: str, reference: str) -> PlanScore:
    """Score one generated plan against its reference plan."""
    errors = validate_plan(generated)
    gen_verbs = _verbs(generated)
    ref_verbs = _verbs(reference)
    return PlanScore(
        well_formed=not errors,
        # An unparseable plan yields [] verbs; guard so it cannot tie with an
        # equally unparseable reference and be scored as a match.
        verb_sequence_match=bool(gen_verbs) and gen_verbs == ref_verbs,
        exact_match=_normalize(generated) == _normalize(reference),
        errors=errors,
        generated_verbs=gen_verbs,
        reference_verbs=ref_verbs,
    )


# --- Stage 2: code --------------------------------------------------------


@dataclass
class CodeScore:
    parses: bool
    self_contained: bool
    executes: bool | None  # None == not attempted
    syntax_error: str | None = None
    blockers: list[str] = field(default_factory=list)
    execution_error: str | None = None


def _imported_roots(tree: ast.AST) -> set[str]:
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def describe_blockers(tree: ast.AST) -> list[str]:
    """Reasons this snippet cannot safely be exec()'d, empty if it can."""
    roots = _imported_roots(tree)
    blockers = []
    for label, modules in (
        ("network", NETWORK_MODULES),
        ("gui", GUI_MODULES),
        ("filesystem", FILESYSTEM_MODULES),
        ("heavy-dependency", HEAVY_MODULES),
    ):
        hit = sorted(roots & modules)
        if hit:
            blockers.append(f"{label}: {', '.join(hit)}")

    calls = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BLOCKING_CALLS
        }
    )
    if calls:
        blockers.append(f"builtin: {', '.join(calls)}")
    return blockers


def score_code(generated: str, attempt_execution: bool = True) -> CodeScore:
    """Score one generated Python snippet.

    Execution is attempted only when the snippet is self-contained. The check
    passed to `run_and_check` is trivially True, so the metric is "ran to
    completion without raising" -- the strongest signal available without
    per-example reference checkers.
    """
    try:
        tree = ast.parse(generated)
    except SyntaxError as e:
        return CodeScore(parses=False, self_contained=False, executes=None, syntax_error=str(e))

    blockers = describe_blockers(tree)
    self_contained = not blockers
    if not (self_contained and attempt_execution):
        return CodeScore(parses=True, self_contained=self_contained, executes=None, blockers=blockers)

    result = run_and_check(generated, check=lambda namespace: True)
    return CodeScore(
        parses=True,
        self_contained=True,
        executes=result.passed,
        execution_error=result.error,
    )


def is_executable_example(reference_code: str) -> bool:
    """True if this example's own reference code runs cleanly.

    `self_contained` is a *safety* gate -- it stops us exec'ing something that
    would hit the network or block on a GUI mainloop. It cannot tell whether
    the fixtures a snippet needs exist, because file access hides behind calls
    like `pd.read_csv("sales.csv")` that no import-level check will catch.

    Rather than chase that statically, gate on the reference: if the reference
    itself cannot run here, a correct generation could not either, and scoring
    the example on execution would punish the model for a missing CSV. Callers
    pass the result as `attempt_execution` so the denominator only ever counts
    examples where execution is a fair test.
    """
    return score_code(reference_code).executes is True


# --- Aggregation ----------------------------------------------------------


def _rate(passed: int, attempted: int) -> float | None:
    return passed / attempted if attempted else None


def aggregate(plan_scores: list[PlanScore], code_scores: list[CodeScore] | None = None) -> dict:
    """Roll per-example scores into tier counts, rates, and denominators.

    Each tier carries `attempted` alongside `passed` so a chart can show
    "1 of 1 attempted, 9 not attempted" instead of implying 1 of 10.
    """
    tiers: dict[str, dict] = {}

    def add(name, passed, attempted):
        tiers[name] = {"passed": passed, "attempted": attempted, "rate": _rate(passed, attempted)}

    n_plans = len(plan_scores)
    add("plan_well_formed", sum(s.well_formed for s in plan_scores), n_plans)
    add("plan_verb_sequence_match", sum(s.verb_sequence_match for s in plan_scores), n_plans)
    add("plan_exact_match", sum(s.exact_match for s in plan_scores), n_plans)

    error_types: Counter = Counter()
    for score in plan_scores:
        for message in score.errors:
            error_types[classify_plan_error(message)] += 1

    if code_scores:
        n_code = len(code_scores)
        add("code_parses", sum(s.parses for s in code_scores), n_code)
        add("code_self_contained", sum(s.self_contained for s in code_scores), n_code)
        attempted = [s for s in code_scores if s.executes is not None]
        add("code_executes", sum(s.executes for s in attempted), len(attempted))

    return {"n_examples": n_plans, "tiers": tiers, "plan_error_types": dict(error_types)}


def classify_plan_error(message: str) -> str:
    """Bucket a validate_plan message into a category for the breakdown chart."""
    if "must be wrapped in" in message:
        return "bad delimiters"
    if "no steps" in message:
        return "no steps"
    if "unknown verb" in message:
        return "unknown verb"
    if "has no arguments" in message:
        return "missing arguments"
    if "empty step" in message:
        return "empty step"
    return "other"
