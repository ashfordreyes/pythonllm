# Next fixes

Open items to address before the next training run, found while reviewing
the first Stage 1 checkpoint upload (2026-08-26).

## ~~`02_stage1_finetune.ipynb` — DSL token embeddings never trained~~ (fixed 2026-08-27)

Originally filed as "`LoraConfig` missing `modules_to_save`", on the
premise that `model.resize_token_embeddings(len(tokenizer))` made
`embed_tokens`/`lm_head` trainable and PEFT merely wasn't told to track
them. That premise was backwards, and the real defect was worse.

- `Qwen2.5-7B-Instruct` has `vocab_size: 152064` while its stock tokenizer
  reaches only id 151664. The three DSL tokens land at 151665-151667, so
  `len(tokenizer)` is 151668 — already inside the matrix. The resize call
  was *shrinking* it 152064 -> 151668, not growing it. It was never needed.
- `prepare_model_for_kbit_training` sets `requires_grad = False` on every
  base parameter, and `get_peft_model` only unfreezes LoRA modules plus
  `modules_to_save`. So the embedding and `lm_head` rows for `<PLAN>`,
  `</PLAN>` and `<STEP>` were frozen throughout, and a shrinking resize
  truncates rows without re-initializing them — the tokens the notebook
  exists to teach were left at their untrained base-model values.
- The multi-GB artifact (next item) was a symptom of the same resize:
  PEFT's `save_embedding_layers="auto"` fires purely because
  `config.vocab_size` no longer matches the base config.

**Fixed by** dropping the resize (replaced with an assert that the ids fit)
and adding `trainable_token_indices={"embed_tokens": ids, "lm_head": ids}`
to `LoraConfig`, which trains exactly those three rows on both sides
(~21K parameters). `modules_to_save=["embed_tokens", "lm_head"]` was
considered and rejected: it would have added ~1.09B trainable fp32
parameters (~13-17GB of extra gradient and optimizer state, likely OOM on
an L4) and still required a manual resize before
`PeftModel.from_pretrained`.

**The existing Drive adapter should be retrained** — its DSL token
embeddings are untrained, and it needs
`base.resize_token_embeddings(len(tokenizer))` to load at all.

## ~~Same notebook — adapter artifact is far larger than necessary~~ (fixed 2026-08-27)

The two full fp32 matrices (`embed_tokens.weight`, `lm_head.weight`, each
`[151668, 3584]`) accounted for the entire ~4.3GB file size.

Resolved by the same change: with the vocab size left alone,
`save_embedding_layers="auto"` no longer triggers, so PEFT saves a bare
LoRA adapter plus the three trained token rows — roughly 160-200MB. The
previously proposed bf16 cast before `trainer.save_model()` is moot.

## ~~Held-out eval split still doesn't exist~~ (fixed 2026-08-27)

`02_stage1_finetune.ipynb` loaded all 50 examples with
`load_dataset(..., split="train")`, `TrainingArguments` had no
`eval_dataset`, and the section 10 "sanity check" generated for
`raw_dataset[0]` — an example the model had trained on.

Diagnosing turned up a constraint the original note missed:
`scripts/check_data.py` enforces that line N's `pseudocode` is byte-identical
in `data/stage1_planner/englishtopseudo.jsonl` and
`data/stage2_coder/pseudotopython.jsonl`. Any split therefore has to be
**index-based and identical across both files**, or end-to-end eval would
score Stage 1 and Stage 2 on different tasks.

**Fixed by** `src/splits.py`: `split_indices` shuffles `range(n)` under
`random.Random(SPLIT_SEED)` and returns 40/10. Both stages derive the same
held-out rows from the seed alone — the invariant holds by construction, with
no split file to keep in sync and no split metadata in the training payload.
Notebook 02 now trains on the train split, evaluates on the held-out split
each epoch, and section 10 generates for a held-out task.
`scripts/check_data.py` prints the resolved split.

`load_best_model_at_end` was considered and rejected: selecting a checkpoint
on the loss of 10 examples selects on noise. The loss curve is logged instead.

## ~~`src/eval/harness.py` has no plotting~~ (fixed 2026-08-27)

Plotting now lives in `src/eval/plots.py` (matplotlib, `Agg` backend), not in
`harness.py` — the harness executes code, which is a different job.
`plot_loss_curve`, `plot_score_breakdown` and `plot_plan_error_types` each
write a `.png` and return its path. Notebook 02 dumps
`trainer.state.log_history` and charts it; `notebooks/04_eval_pipeline.ipynb`
charts the held-out scores.

Scoring had to be built first, because execution alone measures almost
nothing on this data: only 8 of 50 reference snippets are safe to `exec`, and
only 4 actually run (the rest want a display, the network, heavy deps, or CSV
files that aren't in the repo). `src/eval/scoring.py` is therefore layered —
plan validity, verb sequence, exact match, code parse, self-containment,
execution — each tier with its own denominator, so a chart says "not
attempted" instead of implying 0%.

## Known limitation: the execution tier is dormant

With `SPLIT_SEED = 0`, none of the 4 executable examples land in the held-out
set, so `code_executes` reports *not attempted*. Re-seeding until the draw
looks better would be choosing the split after seeing the answers, so the
seed stays put. `python scripts/check_data.py` prints this coverage on every
run. The real fix is more executable examples in the dataset, not a different
seed.
