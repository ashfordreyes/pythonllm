# pythonllm

An LLM for everyday Python coding: DS/DL work plus general scripting, GUI
apps, API clients, and debugging existing code.

**Before starting work, read `context.md`** for the current state, open
questions, and recent decisions that aren't reflected here yet.

## Design

Two-stage pipeline:

1. **Planner (Stage 1):** English task description -> pseudocode DSL (see
   `src/dsl/schema.py` for verbs and special tokens). Fine-tuned from a
   general instruction model (target: Qwen2.5-7B-Instruct).
2. **Coder (Stage 2):** pseudocode -> Python code (pandas/numpy/sklearn/
   PyTorch/TensorFlow for DS/DL; tkinter/requests/stdlib for general
   scripting/GUI). Fine-tuned from a code-pretrained model (target:
   Qwen2.5-Coder-7B).

Both stages: QLoRA/PEFT fine-tuning on Colab (A100/L4, 40GB), reusing each
base model's tokenizer plus the DSL special tokens (`<PLAN>`, `</PLAN>`,
`<STEP>`) for the Planner.

## Layout

```
data/
  raw/                # scraped/sourced code + notebooks
  stage1_planner/     # (english, pseudocode) jsonl
  stage2_coder/       # (pseudocode, python_code) jsonl
notebooks/            # Colab notebooks: data collection, fine-tuning, eval
src/
  dsl/                # pseudocode DSL schema (schema.py) + validator
  eval/               # execution-based eval harness
```

- `src/dsl/schema.py` — DSL verb vocabulary + special tokens. Extend `VERBS`
  here when the dataset design needs new operation types.
- `src/dsl/validator.py` — checks plan well-formedness.
- `src/eval/harness.py` — execution-based, DS-1000-style scoring for
  Stage 1/Stage 2/end-to-end outputs.
- `notebooks/01-04` — data collection, Stage 1 finetune, Stage 2 finetune,
  eval pipeline (in that order; see `notebooks/README.md`).

## Conventions

- DSL verbs are UPPERCASE keywords in a line-oriented grammar (e.g. `LOAD
  <name> FROM "<path>"`). See existing entries in `schema.py` for the pattern
  before adding new verbs.
- Data files are JSONL, one `(input, output)` pair per line, split by stage
  into `data/stage1_planner/` and `data/stage2_coder/`.

## Prompt logging

Every prompt I (the user) send you — in Claude on the web, Claude in VS Code,
Claude Code in the terminal, or any other coding agent working in this
repo — must be appended verbatim to `docs/prompts.txt`, one prompt per
entry, before or as you begin working on it. This applies regardless of
which device or client the prompt came from: the goal is a single running
record of every prompt across all of them.

- Append, never overwrite or edit past entries.
- Prefix each entry with an ISO 8601 UTC timestamp and the client/agent
  name, e.g. `2026-08-26T21:48:59Z [claude-code-terminal]`. Get the
  timestamp by checking the computer's actual clock (e.g. running
  `date -u +"%Y-%m-%dT%H:%M:%SZ"`) rather than guessing — you have
  permission to run this to log prompts accurately.
- `docs/prompts.txt` must never be gitignored — it is tracked in version
  control so the prompt history travels with the repo across devices.

## Status

Early scaffolding — dataset design and base model selection are being
finalized during an ongoing literature review (SPoC, PAL, DS-1000,
CodeAlpaca/Evol-Instruct, StarCoder2/DeepSeek-Coder/Qwen2.5-Coder). Notebooks
are stubs pending that work.
