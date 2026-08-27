# Next fixes

Open items to address before the next training run, found while reviewing
the first Stage 1 checkpoint upload (2026-08-26).

## `notebooks/02_stage1_finetune.ipynb` — `LoraConfig` missing `modules_to_save`

`model.resize_token_embeddings(len(tokenizer))` (needed for the `<PLAN>`,
`</PLAN>`, `<STEP>` DSL special tokens) makes `embed_tokens` and `lm_head`
trainable, and the saved adapter (`adapter_model-001.safetensors`) does
contain full fp32 copies of both — but `LoraConfig` (line 339) never sets
`modules_to_save=["embed_tokens", "lm_head"]`, so `adapter_config.json`
has `"modules_to_save": null`. PEFT isn't formally told to track/restore
these two layers, so a plain `PeftModel.from_pretrained(base_model,
adapter_dir)` may not reload them correctly or may warn/error on
unexpected keys.

**Fix:** add `modules_to_save=["embed_tokens", "lm_head"]` to the
`LoraConfig` call.

## Same notebook — adapter artifact is far larger than necessary

The two full fp32 matrices (`embed_tokens.weight`, `lm_head.weight`, each
`[151668, 3584]`) account for the entire ~4.3GB file size, vs. the
~100-200MB a bare LoRA adapter (r=16, 7 target modules) would otherwise
be.

**Fix:** cast `embed_tokens`/`lm_head` to bf16 before `trainer.save_model()`
— halves the artifact to ~2.2GB with no meaningful precision loss, since
training itself already ran in low precision.

## Held-out eval split still doesn't exist

Carried over from `context.md` (2026-08-26 entry): `02_stage1_finetune.ipynb`
loads all of `data/stage1_planner/englishtopseudo.jsonl` as
`split="train"` with nothing held out, and `04_eval_pipeline.ipynb`
doesn't exist yet. Needed before any real eval/plotting work can start.

## `src/eval/harness.py` has no plotting

Also carried over from `context.md`: the user wants `.png` graphs (loss
curves, pass-rate breakdowns) out of eval runs; nothing in the repo
produces those yet.
