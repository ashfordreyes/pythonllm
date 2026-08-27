# Weight saving & hosting

Guidance on where trained weights (Stage 1 planner adapter, Stage 2 coder
adapter) should live, and why Google Drive alone isn't enough.

## Why not Drive-only

Colab training saves the adapter to Google Drive mid-run (see
`notebooks/02_stage1_finetune.ipynb` section 9,
`/content/drive/MyDrive/pythonllm_checkpoints/stage1_planner`). That's a
fine working cache during a session, but it's a single point of failure —
account issues, an accidental overwrite by a later run, or a Colab/Drive
sync hiccup can lose it. It's also not a real release location: hard to
version, hard to share, hard to pull into a fresh environment cleanly.

## What each hosting option is for

Both stages are QLoRA/PEFT adapters on top of open base models
(Qwen2.5-7B-Instruct for Stage 1, Qwen2.5-Coder-7B for Stage 2), not full
model weights — this shapes what's appropriate:

- **Hugging Face Hub** — canonical, versioned home for the adapter weights
  themselves. Push with `push_to_hub()` on the adapter *and* the tokenizer
  (private repo if not ready to be public). Reload from anywhere with
  `peft.PeftModel.from_pretrained(base_model, "your-username/adapter-name")`,
  no Drive mount required. This is the source of truth for "what are the
  current best weights."
- **Weights & Biases** — experiment tracking, not artifact hosting: loss
  curves, comparing hyperparameter choices across runs. Enable via
  `TrainingArguments(report_to="wandb")` + `wandb.init()`. (Currently
  `report_to="none"` in the notebooks — turning this on is still open.)
  W&B *can* store model artifacts too, but that's not its strength for
  adapters meant to be reloaded for inference — use HF Hub for that.
- **Local copy** — pull a copy down after each successful run. The adapter
  dir is small (LoRA weights, not the full base model), so this is cheap
  insurance and lets you experiment offline without re-downloading from HF.
- **Google Drive** — keep as the Colab-session working copy (training
  needs it mid-run anyway), not the permanent archive.

## What to save from Drive

Pull the full `stage1_planner_qlora/final` directory (Stage 1) at
`/content/drive/MyDrive/pythonllm_checkpoints/stage1_planner` — not just
the adapter weights. It contains **both** the adapter *and* the tokenizer
with the DSL special tokens (`<PLAN>`, `</PLAN>`, `<STEP>`) added,
deliberately saved together. An adapter loaded against a stock-vocab
tokenizer that's missing those special tokens will not work correctly.
(Stage 2's coder adapter, once that notebook exists, will follow the same
pattern under its own checkpoint path.)

## Status

- HF Hub publishing and W&B run tracking are not wired into the notebooks
  yet — this doc records the intended setup, not a completed one.
- See `context.md`'s 2026-08-26 entry ("First real training run in
  progress; eval + weight-hosting gaps identified") for the discussion
  this was captured from.
