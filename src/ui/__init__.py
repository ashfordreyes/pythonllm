"""Presentation layer: an ASCII/ipywidgets console for the two-stage pipeline.

Kept separate from `src/eval` on purpose -- scoring a model and talking to one
are different jobs, and `console.py` must stay importable without torch so its
layout can be exercised off a GPU runtime.
"""

from .console import Console, Theme, banner, highlight_code, highlight_plan, launch, panel

__all__ = [
    "Console",
    "Theme",
    "banner",
    "highlight_code",
    "highlight_plan",
    "launch",
    "panel",
]
