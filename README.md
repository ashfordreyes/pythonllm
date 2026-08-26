# pythonllm

An LLM for everyday Python coding: data-science/deep-learning work as well as
general scripting, GUI apps, API clients, and debugging/fixing existing
Python code.

## Design

A two-stage pipeline:

1. **Planner (Stage 1):** English task description -> pseudocode, using a
   pseudocode DSL (see `src/dsl/`). Covers both the DS/DL pipeline verbs
   (LOAD/TRAIN/EVALUATE/...) and general-purpose verbs (BUILD/FIX/DEBUG/...)
   for the kind of everyday Python tasks that come up outside a model
   pipeline. Fine-tuned from a general instruction model (e.g.
   Qwen2.5-7B-Instruct).
2. **Coder (Stage 2):** pseudocode -> Python code (pandas/numpy/sklearn/
   PyTorch/TensorFlow for DS/DL steps; tkinter/requests/stdlib etc. for
   general scripting and GUI steps). Fine-tuned from a code-pretrained model
   (e.g. Qwen2.5-Coder-7B).

Both stages are fine-tuned with QLoRA/PEFT on Colab (target: A100/L4, 40GB),
reusing each base model's existing tokenizer plus a small set of DSL special
tokens for the Planner (`<PLAN>`, `</PLAN>`, `<STEP>`).

## Layout

```
data/
  raw/                # scraped/sourced code + notebooks
  stage1_planner/     # (english, pseudocode) jsonl
  stage2_coder/       # (pseudocode, python_code) jsonl
notebooks/            # Colab notebooks: data collection, fine-tuning, eval
src/
  dsl/                # pseudocode DSL schema + validator
  eval/               # execution-based eval harness
```

## Status

Early scaffolding stage — dataset design and base model selection are being
finalized during an ongoing literature review (relevant prior work: SPoC,
PAL, DS-1000, CodeAlpaca/Evol-Instruct, StarCoder2/DeepSeek-Coder/Qwen2.5-Coder).
Notebooks are stubs to be filled in as that lands (see `notebooks/README.md`).

## Prompt log

`docs/prompts.txt` is a running, timestamped log of every prompt used to
build this project across coding agents/clients. Kept for transparency,
since this is a class project on prompt engineering.

## License

[MIT](LICENSE)
