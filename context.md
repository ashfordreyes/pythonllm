# Context log

Running log of decisions, open questions, and in-progress threads that
aren't settled enough to belong in `CLAUDE.md` yet. Newest entries at the
top. Periodically fold anything durable into `CLAUDE.md` and prune this file.

## How to use this file

- Append a dated entry whenever a decision is made, a question is opened, or
  a direction changes.
- When something here becomes stable/settled, move the summary into
  `CLAUDE.md` and delete the entry here.

---

## 2026-08-26 — Fixed section 10 sanity-check crash (`apply_chat_template` return type)

Training completed and the adapter was already saved to Drive (sections
7-9) when section 10's quick sanity check crashed with `KeyError: 'shape'`
surfacing as an `AttributeError` deep in
`transformers/tokenization_utils_base.py`. Root cause: `code-14` called
`tokenizer.apply_chat_template(..., tokenize=True, return_tensors="pt")`
without an explicit `return_dict`, and on the `transformers` version pulled
by section 2b's `pip install -U`, that returns a `BatchEncoding` rather
than a raw tensor. `.to(model.device)` works on either, so it got that far
before `inputs.shape[1]` failed — `BatchEncoding` has no `.shape`. Fixed by
pinning `return_dict=True` and indexing `encoded["input_ids"]` /
`encoded["attention_mask"]` explicitly, so behavior no longer depends on
library version. Documented as a troubleshooting entry in
`docs/colab_setup.md` §7 too.

## 2026-08-26 — First real training run in progress; eval + weight-hosting gaps identified

User ran `02_stage1_finetune.ipynb` section 7 (`trainer.train()`) for the
first time on Colab. Discussion surfaced several gaps to close before the
"generate `.png` eval graphs" work the user wants can be built:

- **No held-out split exists yet.** Section 6 (`code-09`) loads all of
  `data/stage1_planner/englishtopseudo.jsonl` (50 examples) as
  `split="train"` with nothing carved out for eval. `04_eval_pipeline.ipynb`
  (described in `notebooks/README.md`: run held-out examples through Stage
  1 alone, Stage 2 alone, and end-to-end; score with `src/eval/harness.py`
  execution checks plus `src/dsl/validator.py` for plan well-formedness)
  does not exist yet — only the README description does. Before that
  notebook can produce anything, either the jsonl needs a train/eval split
  or a separate held-out file needs to be written.
- **`src/eval/harness.py` has no plotting.** `run_and_check()` is just an
  `exec()`-based pass/fail + traceback runner — no chart/PNG output. The
  user wants `.png` graphs (loss curves, pass-rate breakdowns, etc.) out of
  eval runs, coming from vision-model eval experience where that's normal;
  that plotting layer doesn't exist anywhere in the repo and needs to be
  built new (likely matplotlib, added to whatever `04_eval_pipeline.ipynb`
  becomes) rather than reused from somewhere.
- **`src/dsl/validator.py`'s `validate_plan()` isn't wired to model
  output anywhere.** It only validates hand-written/dataset plans today.
  An eval pipeline would want to run it against Stage 1 generations too.
- **Section 10 (`code-14`) is a smoke test, not an eval** — it decodes one
  *training* example back through the model to confirm tokens/format look
  right, not a quality signal (the model has seen that example).
- **Local inference path confirmed:** the Drive-saved adapter dir
  (`stage1_planner_qlora/final`, copied to
  `/content/drive/MyDrive/pythonllm_checkpoints/stage1_planner` by section
  9) contains the adapter *and* the tokenizer with DSL special tokens
  together (deliberate — an adapter loaded against a stock-vocab tokenizer
  won't work). Local use: load base `Qwen/Qwen2.5-7B-Instruct`, apply with
  `peft.PeftModel.from_pretrained(base_model, adapter_dir)`. 4-bit
  quantization was only needed for Colab GPU memory during training, not a
  requirement for using the artifact afterward.
- **Weight hosting/publishing:** nothing in the repo pushes weights
  anywhere beyond the Drive checkpoint copy in section 9. Since these are
  adapters on top of an open base model, Hugging Face Hub
  (`push_to_hub()` on the adapter + tokenizer) is the natural place to
  publish, not Weights & Biases — W&B is for run/experiment tracking
  (loss curves, comparing runs), which the notebook also doesn't do yet
  (`TrainingArguments(report_to="none")` in section 7). If experiment
  tracking is wanted later, that's a `report_to="wandb"` + `wandb.init()`
  addition to section 7, separate from weight publishing.

## 2026-08-26 — Scaffolding, set up CLAUDE.md/context.md

- Repo is at the initial scaffold stage: `src/dsl/` (schema + validator),
  `src/eval/harness.py`, notebook stubs, and one seed JSONL file in
  `data/stage1_planner/`.
- Open question (per README "Status"): base model selection and dataset
  design are still being finalized via literature review (SPoC, PAL,
  DS-1000, CodeAlpaca/Evol-Instruct, StarCoder2/DeepSeek-Coder/Qwen2.5-Coder).
  Nothing to reuse-check against yet on model choice.
- Added this file plus `CLAUDE.md` so future sessions pick up current state
  automatically instead of re-deriving it from scratch.
