"""Pseudocode DSL: schema and validator for the Stage 1 -> Stage 2 intermediate representation.

A plan is a sequence of steps. Each step is one DSL statement, e.g.:

    LOAD df FROM "train.csv"
    SPLIT df INTO train, test RATIO 0.8
    TRAIN model TYPE RandomForestClassifier ON train
    EVALUATE model ON test METRIC accuracy

This module defines the grammar and a validator so generated plans can be checked
programmatically before being handed to the Stage 2 coder model.
"""

from .schema import PLAN_START, PLAN_END, STEP_TOKEN, VERBS
from .validator import ValidationError, parse_plan, validate_plan

__all__ = [
    "PLAN_START",
    "PLAN_END",
    "STEP_TOKEN",
    "VERBS",
    "ValidationError",
    "parse_plan",
    "validate_plan",
]
