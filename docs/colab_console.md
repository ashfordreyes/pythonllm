# Chatting with the Planner and Coder in Colab

A Colab cell renders stdout as a monospace block and can render widgets
underneath it, so the same boxed, colored "almost-GUI" a CLI tool draws in a
terminal works here too. `src/ui/console.py` is that surface; running one cell
puts a text box, a row of buttons and a live-updating output panel directly
beneath the cell.

```
╭─ TASK ─────────────────────────────────────────────────────────────────────╮
│ Load sales.csv, drop rows with missing revenue, fit a linear model         │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ PLAN ─────────────────────────────────────────────────────────────────────╮
│ <PLAN>                                                                     │
│ <STEP> LOAD df FROM "sales.csv"                                            │
│ <STEP> TRANSFORM df USING dropna ON revenue                                │
│ <STEP> TRAIN model TYPE LinearRegression ON df                             │
│ </PLAN>                                                                    │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ PYTHON ───────────────────────────────────────────────────────────────────╮
│ import pandas as pd                                                        │
│ from sklearn.linear_model import LinearRegression                          │
│                                                                            │
│ df = pd.read_csv("sales.csv").dropna(subset=["revenue"])                   │
╰────────────────────────────────────────────────────────────────────────────╯
```

DSL special tokens, DSL verbs and quoted literals are colored in the plan
panel; the Python panel is syntax-highlighted with pygments. Both models
stream, so the panels fill in as tokens arrive instead of appearing all at
once after a minute of nothing.

## What it needs

Two callables: `plan_fn(english) -> pseudocode` and
`code_fn(pseudocode) -> python`. Either can be `None`. **You do not need the
Coder to use this** — with only the Stage 1 adapter, pass `plan_fn` alone and
the console is a planner UI (see [`colab_setup.md`](colab_setup.md) §9 for
where Stage 2 stands).

The console never imports torch or transformers, so you can also hand it
stubs and check the layout on a CPU runtime before spending compute units.

## The cell

Paste this into a fresh Colab notebook (GPU runtime). It clones the repo,
mounts Drive, loads whichever adapters exist, and launches the UI.

```python
# --- pythonllm console -------------------------------------------------------
!pip install -q -U transformers accelerate peft bitsandbytes

import os, sys, torch

REPO_DIR = "/content/pythonllm"
if not os.path.isdir(REPO_DIR):
    !git clone -b main https://github.com/ashfordreyes/pythonllm.git {REPO_DIR}
sys.path.insert(0, f"{REPO_DIR}/src")

from google.colab import drive
drive.mount("/content/drive")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ui.console import launch
from ui.generate import make_coder, make_planner

CKPT = "/content/drive/MyDrive/pythonllm_checkpoints"
PLANNER_ADAPTER = f"{CKPT}/stage1_planner"
CODER_ADAPTER   = f"{CKPT}/stage2_coder"      # may not exist yet

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def load_adapter(adapter_dir, base_model):
    tok = AutoTokenizer.from_pretrained(adapter_dir)   # carries the DSL tokens
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
    )
    return PeftModel.from_pretrained(base, adapter_dir).eval(), tok

plan_fn = code_fn = None
if os.path.isdir(PLANNER_ADAPTER):
    m, t = load_adapter(PLANNER_ADAPTER, "Qwen/Qwen2.5-7B-Instruct")
    plan_fn = make_planner(m, t)
if os.path.isdir(CODER_ADAPTER):
    m, t = load_adapter(CODER_ADAPTER, "Qwen/Qwen2.5-Coder-7B")
    code_fn = make_coder(m, t)

console = launch(plan_fn=plan_fn, code_fn=code_fn)
```

Then type a task and press **Plan → Code**. `console.history` holds every
exchange as `{"task", "plan", "code"}` dicts, and `console.last_code` is the
most recent generation — `exec(console.last_code)` or paste it into a new cell
to run it.

### Memory

Both models loaded at once is ~10-11GB of 4-bit weights plus activations —
comfortable on an A100 (40GB) or L4 (24GB), the same GPUs
[`colab_setup.md`](colab_setup.md) §3 recommends for training. If you hit OOM,
load one at a time and drive the stages in separate cells.

## Buttons

| Button | What it does |
| --- | --- |
| **Plan** | Stage 1 only: English → pseudocode |
| **Plan → Code** | The full pipeline, code generated from the *generated* plan |
| **Code from plan** | Stage 2 only: treats the text box as pseudocode |
| **Clear** | Empties the output and the text box |

"Code from plan" is the one to use when you want to hand-write a plan and see
what the Coder does with it — useful for debugging which stage a bad answer
came from.

## If widgets don't render

Colab ships ipywidgets, but a stale runtime or a third-party widget upgrade can
leave the container blank. Don't `pip install -U ipywidgets` to fix it — Colab
pins its own widget manager and upgrading is as likely to break rendering as to
repair it. Restart the runtime first (Runtime → Restart session), and if that
doesn't help, use the prompt loop instead, which needs no widgets at all:

```python
from ui.console import Console
from ui.generate import make_coder, make_planner

Console(plan_fn=plan_fn, code_fn=code_fn).repl()
```

`input()` renders as a text box under the cell, so this is still interactive.
It accepts `/plan <task>`, `/code <pseudocode>` and `/quit`, and appends each
exchange rather than redrawing, so the transcript scrolls back.

## Theming

`launch()` forwards keyword arguments to `Theme`:

```python
console = launch(plan_fn, code_fn, width=100, code_style="monokai")
```

- `width` (default 78) — panel width in characters.
- `code_style` (default `"friendly"`) — any pygments style name. `"friendly"`
  is tuned for Colab's light theme; use `"monokai"` or `"native"` on the dark
  one.
- `color=False` — plain text, for copying output somewhere that eats escapes.
- `unicode=False` — `+--+` box drawing instead of `╭──╮`.

The default palette is 256-color mid-tones deliberately: values near the ends
of the ramp read on one Colab theme and vanish on the other.

## Trying the layout without a model

The console only needs two callables, so stubs exercise it on any runtime —
no GPU, no download:

```python
import sys; sys.path.insert(0, "src")
from ui.console import Console

PLAN = '<PLAN>\n<STEP> LOAD df FROM "sales.csv"\n<STEP> PRINT df\n</PLAN>'
Console(plan_fn=lambda task: PLAN,
        code_fn=lambda plan: 'import pandas as pd\ndf = pd.read_csv("sales.csv")\nprint(df)').launch()
```

This also works locally (`python -m IPython`), which is how the panel padding
and wrapping are checked without burning compute units.
