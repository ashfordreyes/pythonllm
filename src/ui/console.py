"""ASCII console for talking to the Planner and Coder from a Colab cell.

Colab renders a cell's stdout as a monospace block and can render ipywidgets
underneath it, which is enough to build the same kind of boxed, colored
"almost-GUI" a CLI tool draws in a terminal. This module is that surface: box
panels, DSL/Python syntax highlighting, and a small input widget, so a session
with the two-stage pipeline reads as a conversation instead of a wall of
`print()` output.

It is deliberately model-agnostic. `Console` takes two callables --
`plan_fn(english) -> pseudocode` and `code_fn(pseudocode) -> python` -- so it
works with the Stage 1 adapter alone, with both adapters, or with stubs for
testing the layout without a GPU. `src/ui/generate.py` builds those callables
from a loaded transformers model.

Either callable may return a plain string or an iterator of *cumulative*
snapshots (what `TextIteratorStreamer` gives you when accumulated); the
console re-renders on each snapshot, so generation appears live rather than
arriving in one block after a minute of silence.

Nothing here imports torch, transformers or ipywidgets at module level -- the
rendering half has to stay importable (and testable) off a runtime.
"""

from __future__ import annotations

import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

try:
    from dsl.schema import PLAN_END, PLAN_START, STEP_TOKEN, VERBS
except ImportError:  # pasted into a notebook without the repo's src/ on sys.path
    PLAN_START, PLAN_END, STEP_TOKEN = "<PLAN>", "</PLAN>", "<STEP>"
    VERBS = frozenset()  # verb highlighting degrades to off, nothing else changes

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

# 256-color mid-tones, chosen to stay legible against *both* Colab themes.
# Anything near 0 or 15 (near-black / near-white) reads on one and vanishes on
# the other, so the palette avoids the ends of the ramp entirely.
_BORDER = 244  # gray
_ACCENT = 39   # blue
_PLAN = 170    # magenta -- DSL special tokens
_VERB = 36     # teal -- DSL verbs
_STRING = 172  # orange -- quoted literals
_MUTED = 244
_ERROR = 167   # red

RESET = "\x1b[0m"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class Theme:
    """Presentation knobs. `Console` holds one; module helpers default to `THEME`."""

    color: bool = True
    unicode: bool = True
    width: int = 78
    # Any pygments style name. "friendly" is tuned for a light background
    # (Colab's default); "monokai" or "native" suit the dark theme better.
    code_style: str = "friendly"


THEME = Theme()

_BOX_UNICODE = {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
                "h": "─", "v": "│"}
_BOX_ASCII = {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"}


def _paint(text: str, color: int, *, bold: bool = False, theme: Theme | None = None) -> str:
    th = theme or THEME
    if not th.color:
        return text
    codes = "1;" if bold else ""
    return f"\x1b[{codes}38;5;{color}m{text}{RESET}"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    """Printed width of a string, ignoring SGR escapes.

    Panel padding is computed from this rather than `len()`; otherwise every
    colored line pads short by the length of its escape sequences and the right
    border comes out ragged.
    """
    return len(strip_ansi(text))


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def wrap_block(text: str, width: int, *, indent: str = "  ") -> list[str]:
    """Hard-wrap to `width`, preserving blank lines and leading indentation.

    Wrapping happens on the *raw* text, before any highlighting, so no escape
    sequence is ever split across two output lines.
    """
    lines: list[str] = []
    for raw in text.expandtabs(4).splitlines() or [""]:
        if not raw.strip():
            lines.append("")
            continue
        lead = raw[: len(raw) - len(raw.lstrip())]
        wrapped = textwrap.wrap(
            raw.strip(),
            width=max(width - len(lead), 8),
            initial_indent=lead,
            subsequent_indent=lead + indent,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [lead])
    return lines


def panel(body: str, title: str = "", *, accent: int = _ACCENT,
          highlight: Callable[[str], str] | None = None,
          theme: Theme | None = None) -> str:
    """Render `body` inside a titled box.

    `highlight` is applied to the already-wrapped block, so it sees the exact
    lines that will be printed.
    """
    th = theme or THEME
    box = _BOX_UNICODE if th.unicode else _BOX_ASCII
    inner = th.width - 4

    lines = wrap_block(body, inner)
    if highlight is not None:
        lines = highlight("\n".join(lines)).splitlines() or [""]

    if title:
        cap = _paint(title.upper(), accent, bold=True, theme=th)
        left = _paint(box["tl"] + box["h"] + " ", _BORDER, theme=th) + cap + " "
        used = 3 + len(title) + 1
    else:
        left = _paint(box["tl"] + box["h"], _BORDER, theme=th)
        used = 2
    top = left + _paint(box["h"] * max(th.width - used - 1, 0) + box["tr"], _BORDER, theme=th)

    v = _paint(box["v"], _BORDER, theme=th)
    out = [top]
    for line in lines:
        pad = " " * max(inner - visible_len(line), 0)
        out.append(f"{v} {line}{pad} {v}")
    out.append(_paint(box["bl"] + box["h"] * (th.width - 2) + box["br"], _BORDER, theme=th))
    return "\n".join(out)


def banner(theme: Theme | None = None) -> str:
    """The title card printed once when a console starts."""
    th = theme or THEME
    # figlet "small" -- 5 lines, 39 columns, fits inside the default 78-wide panel.
    art = [
        '           _   _             _ _       ',
        ' _ __ _  _| |_| |_  ___ _ _ | | |_ __  ',
        "| '_ \\ || |  _| ' \\/ _ \\ ' \\| | | '  \\ ",
        '| .__/\\_, |\\__|_||_\\___/_||_|_|_|_|_|_|',
        '|_|   |__/                             ',
    ]
    if not th.unicode:
        art = ["pythonllm"]
    tagline = "planner \u2192 coder" if th.unicode else "planner -> coder"
    return panel("\n".join(art) + f"\n\n{tagline}  |  two-stage pipeline", "", theme=th)


# ---------------------------------------------------------------------------
# Highlighting
# ---------------------------------------------------------------------------

# One pass, alternation-ordered: special tokens, then quoted literals, then
# bare uppercase words. Matching in a single pass matters -- a second regex run
# over already-colored text would match inside the escape sequences the first
# one inserted.
_PLAN_RE = re.compile(
    r"(?P<tok>" + "|".join(re.escape(t) for t in (PLAN_START, PLAN_END, STEP_TOKEN)) + r")"
    r'|(?P<str>"[^"\n]*")'
    r"|(?P<verb>\b[A-Z][A-Z_]{1,}\b)"
)


def highlight_plan(text: str, theme: Theme | None = None) -> str:
    """Color DSL special tokens, verbs from `schema.VERBS`, and quoted literals."""
    th = theme or THEME
    if not th.color:
        return text

    def repl(m: re.Match) -> str:
        if m.lastgroup == "tok":
            return _paint(m.group(0), _PLAN, bold=True, theme=th)
        if m.lastgroup == "str":
            return _paint(m.group(0), _STRING, theme=th)
        if m.group(0) in VERBS:
            return _paint(m.group(0), _VERB, bold=True, theme=th)
        return m.group(0)

    return _PLAN_RE.sub(repl, text)


def highlight_code(code: str, theme: Theme | None = None) -> str:
    """Pygments-highlight Python. Returns the input unchanged if pygments is absent."""
    th = theme or THEME
    if not th.color:
        return code
    try:
        from pygments import highlight
        from pygments.formatters import Terminal256Formatter
        from pygments.lexers import PythonLexer
    except ImportError:  # pygments ships with IPython, but don't hard-depend on it
        return code
    try:
        formatter = Terminal256Formatter(style=th.code_style)
    except Exception:
        formatter = Terminal256Formatter()
    return highlight(code, PythonLexer(), formatter).rstrip("\n")


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

Generator = Callable[[str], "str | Iterator[str]"]


def _consume(result, on_update: Callable[[str], None]) -> str:
    """Normalize a generator's return value to a final string.

    A plain string is used as-is; an iterator is treated as a stream of
    cumulative snapshots and each one is pushed to `on_update`, which is what
    makes generation visible while it happens.
    """
    if isinstance(result, str):
        return result
    latest = ""
    for snapshot in result:
        latest = snapshot
        on_update(latest)
    return latest


@dataclass
class Console:
    """A rendering surface bound to a planner and/or coder callable."""

    plan_fn: Generator | None = None
    code_fn: Generator | None = None
    theme: Theme = field(default_factory=Theme)
    #: Every completed exchange, so a session can be inspected or saved after.
    history: list[dict] = field(default_factory=list)
    last_plan: str = ""
    last_code: str = ""

    # -- frame construction ------------------------------------------------

    def frame(self, task: str, plan: str = "", code: str = "", status: str = "") -> str:
        """Build the whole output block for one exchange, from current state.

        The console re-renders the *entire* frame on every streamed update
        rather than appending, because Colab's cell output can only be cleared
        and rewritten wholesale, not edited in place.
        """
        parts = []
        # An empty task means the caller fed pseudocode straight to the coder,
        # so the input is already shown by the plan panel below.
        if task:
            parts.append(panel(task, "task", accent=_ACCENT, theme=self.theme))
        if plan:
            parts.append(panel(plan, "plan", accent=_PLAN,
                               highlight=lambda t: highlight_plan(t, self.theme),
                               theme=self.theme))
        if code:
            parts.append(panel(code, "python", accent=_VERB,
                               highlight=lambda t: highlight_code(t, self.theme),
                               theme=self.theme))
        if status:
            parts.append(_paint(f"  {status}", _MUTED, theme=self.theme))
        return "\n".join(parts)

    def error_frame(self, task: str, exc: BaseException) -> str:
        return "\n".join([
            panel(task, "task", accent=_ACCENT, theme=self.theme),
            panel(f"{type(exc).__name__}: {exc}", "error", accent=_ERROR, theme=self.theme),
        ])

    # -- driving the models ------------------------------------------------

    def ask(self, task: str, *, plan: bool = True, code: bool = True,
            sink: Callable[[str], None] | None = None,
            redraw: bool = True, min_redraw: float = 1 / 12) -> dict:
        """Run one exchange and render it.

        `plan=False` treats `task` as pseudocode and goes straight to the coder;
        `code=False` stops after the plan. `sink` receives each rendered frame
        (the widget UI passes one that writes into its Output area); the default
        clears and reprints the cell's own output.

        `redraw=False` is for sinks that *append* instead of replacing -- a bare
        `print`, or a saved transcript. Each frame contains the whole exchange,
        so redrawing into an appending sink stacks half-finished copies of it;
        with `redraw=False` only the finished frame is emitted, at the cost of
        the live streaming effect.
        """
        sink = sink or self._default_sink
        state = {"task": task, "plan": "", "code": ""}
        # With no planner in the loop the input *is* the plan; don't render it
        # twice under two different headings.
        render_task = task if plan else ""
        last = [0.0]

        def draw(status: str = "", final: bool = False) -> None:
            if not final:
                now = time.monotonic()
                if not redraw or now - last[0] < min_redraw:
                    return
                last[0] = now
            sink(self.frame(render_task, state["plan"], state["code"], status))

        try:
            if plan and self.plan_fn is not None:
                draw("planning…")

                def on_plan(snapshot: str) -> None:
                    state["plan"] = snapshot
                    draw("planning…")

                state["plan"] = _consume(self.plan_fn(task), on_plan)
                draw()
            else:
                # No planner: the task text *is* the plan handed to the coder.
                state["plan"] = "" if plan else task

            if code and self.code_fn is not None:
                source = state["plan"] or task
                draw("writing code…")

                def on_code(snapshot: str) -> None:
                    state["code"] = snapshot
                    draw("writing code…")

                state["code"] = _consume(self.code_fn(source), on_code)

            draw(final=True)
        except Exception as exc:  # a bad generation shouldn't kill the session
            sink(self.error_frame(task, exc))
            state["error"] = f"{type(exc).__name__}: {exc}"

        self.last_plan = state["plan"] or self.last_plan
        self.last_code = state["code"] or self.last_code
        self.history.append(state)
        return state

    def _default_sink(self, text: str) -> None:
        try:
            from IPython.display import clear_output

            clear_output(wait=True)
        except ImportError:
            pass
        print(text)

    # -- entry points ------------------------------------------------------

    def repl(self, *, plan: bool = True, code: bool = True) -> None:
        """Prompt loop built on `input()`.

        Colab renders `input()` as a text box under the cell, so this is a real
        interactive session with no widget dependency -- the fallback when
        ipywidgets is unavailable, and the mode to use over SSH or in a plain
        terminal.
        """
        print(banner(self.theme))
        print(_paint("  type a task, or /quit to exit, /code <plan> for the coder alone",
                     _MUTED, theme=self.theme))
        while True:
            try:
                task = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return
            if not task:
                continue
            if task in ("/quit", "/exit", "/q"):
                print("bye")
                return
            want_plan, want_code = plan, code
            if task.startswith("/code "):
                task, want_plan, want_code = task[6:].strip(), False, True
            elif task.startswith("/plan "):
                task, want_plan, want_code = task[6:].strip(), True, False
            # The prompt loop appends rather than redraws: clearing output would
            # wipe the transcript the user is scrolling back through. That rules
            # out live streaming here, so print the status line by hand.
            print(_paint("  working…", _MUTED, theme=self.theme))
            self.ask(task, plan=want_plan, code=want_code, sink=print, redraw=False)

    def launch(self):
        """Render the ipywidgets UI beneath the current cell.

        Returns the widget container (already displayed) so a caller can tweak
        it. Falls back to `repl()` if ipywidgets isn't importable.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("ipywidgets not available; falling back to the prompt loop")
            return self.repl()

        box_style = {"description_width": "initial"}
        task_box = widgets.Textarea(
            placeholder="Describe the task in English…",
            layout=widgets.Layout(width="100%", height="80px"),
            style=box_style,
        )
        buttons = {
            "plan": widgets.Button(description="Plan", button_style="info", icon="list-ol"),
            "both": widgets.Button(description="Plan → Code", button_style="success",
                                   icon="play"),
            "code": widgets.Button(description="Code from plan", icon="code"),
            "clear": widgets.Button(description="Clear", icon="eraser"),
        }
        status = widgets.HTML("")
        out = widgets.Output()
        ui = widgets.VBox([
            task_box,
            widgets.HBox(list(buttons.values())),
            status,
            out,
        ])

        def sink(text: str) -> None:
            # wait=True double-buffers, so a streaming redraw doesn't flicker.
            out.clear_output(wait=True)
            with out:
                print(text)

        def run(plan: bool, code: bool):
            def handler(_btn):
                task = task_box.value.strip()
                if not task:
                    return
                for b in buttons.values():
                    b.disabled = True
                status.value = "<i>working…</i>"
                try:
                    self.ask(task, plan=plan, code=code, sink=sink)
                finally:
                    status.value = ""
                    for b in buttons.values():
                        b.disabled = False
            return handler

        def clear(_btn):
            out.clear_output()
            task_box.value = ""

        buttons["plan"].on_click(run(True, False))
        buttons["both"].on_click(run(True, True))
        buttons["code"].on_click(run(False, True))
        buttons["clear"].on_click(clear)

        display(ui)
        with out:
            print(banner(self.theme))
        return ui


def launch(plan_fn: Generator | None = None, code_fn: Generator | None = None,
           **theme_kwargs) -> Console:
    """Build a `Console` and show its UI. Returns the console (for `.history`)."""
    console = Console(plan_fn=plan_fn, code_fn=code_fn, theme=Theme(**theme_kwargs))
    console.launch()
    return console
