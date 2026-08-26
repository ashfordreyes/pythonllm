"""Minimal structural validator for the pseudocode DSL.

This does not attempt to fully parse each verb's arguments (that grammar is
still being finalized during the literature review / dataset design phase).
It checks the things that matter for training data quality: well-formed
plan delimiters and known verbs on every step.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import PLAN_END, PLAN_START, STEP_TOKEN, VERBS


class ValidationError(ValueError):
    pass


@dataclass
class Step:
    verb: str
    rest: str


def parse_plan(text: str) -> list[Step]:
    text = text.strip()
    if not text.startswith(PLAN_START) or not text.endswith(PLAN_END):
        raise ValidationError(f"plan must be wrapped in {PLAN_START} ... {PLAN_END}")

    body = text[len(PLAN_START):-len(PLAN_END)].strip()
    raw_steps = [s.strip() for s in body.split(STEP_TOKEN) if s.strip()]
    if not raw_steps:
        raise ValidationError("plan has no steps")

    steps = []
    for raw in raw_steps:
        parts = raw.split(None, 1)
        if not parts:
            raise ValidationError(f"empty step in plan: {raw!r}")
        verb, rest = parts[0], parts[1] if len(parts) > 1 else ""
        steps.append(Step(verb=verb, rest=rest))
    return steps


def validate_plan(text: str) -> list[str]:
    """Return a list of validation error messages; empty list means valid."""
    errors: list[str] = []
    try:
        steps = parse_plan(text)
    except ValidationError as e:
        return [str(e)]

    for i, step in enumerate(steps):
        if step.verb not in VERBS:
            errors.append(f"step {i}: unknown verb {step.verb!r} (expected one of {sorted(VERBS)})")
        if not step.rest:
            errors.append(f"step {i}: verb {step.verb!r} has no arguments")
    return errors
