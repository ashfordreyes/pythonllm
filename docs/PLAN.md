# Plan — integrating the `h0ney-badger` 1.5B distill

Written 2026-08-27 in response to a direct request for a concrete,
step-by-step checklist. This is **not** a replacement for
`docs/mega_plan.md` — it's the immediately-actionable slice of it, expanded
with the detail needed to actually execute, now that a real Python-only 1.5B
distill exists to test against the plan's own D2/D-5 questions.

Read `context.md`'s 2026-08-27 "`h0ney-badger`" entry first for the verified
facts this plan is built on. Everything below respects the phase boundary
`mega_plan.md` already draws: **Phase A/B are CPU-only and agent-doable;
everything from Gate 1 onward needs a GPU and is explicitly gated on the
user.** This plan does not skip that boundary, and neither should any
session that picks it up.

---

## 0. What I changed already (done, no action needed)

- `context.md` — dated entry with the verified model facts (base, teacher,
  license, training recipe, what's actually published, the unverified
  81.5% pass@1 claim flagged as such).
- `docs/mega_plan.md` — added the model as a D2 bake-off arm, noted it as a
  candidate G2 single-stage baseline, and added `Qwen2.5-Coder-14B-Instruct`
  as a second, verified D-5 teacher candidate alongside the unresolved Gemma
  one.
- `docs/prompts.txt` — this session's prompts logged, per `CLAUDE.md`.
- `CHANGELOG.md` — entry for this change, with reasoning.

I did **not** touch `notebooks/03_stage2_finetune.ipynb`, any data file, or
write new inference code. Reasons below (§3).

---

## 1. What you do first — costs nothing, no Colab needed

You already have Ollama running this model (Q4_K_M, from the earlier
question about it in this session). Before spending any Colab time:

1. **Sanity-check it against a handful of tasks from this project's own
   distribution**, not generic prompts. Pull a few `english` values straight
   out of `data/stage1_planner/englishtopseudo.jsonl` and feed them to the
   model directly (plain instruction, no DSL/pseudocode — this model was
   never trained on this project's plan format):
   ```bash
   ollama run hf.co/h0ney-badger/qwen2.5-coder-1.5b-python-distill:Q4_K_M
   ```
   Compare its output by eye against the matching line in
   `data/stage2_coder/pseudotopython.jsonl`. This is not a scored eval — it's
   a 10-minute gut check for whether this is worth spending a Colab session
   on at all.
2. **Do not trust the card's 81.5% pass@1 figure for any decision.** It's
   unverified here, on an unstated benchmark/split, from the model's own
   author. Treat it the same way `mega_plan.md` treats every other
   number in Phase D: something to measure yourself.

If step 1 looks obviously bad (nonsense output, ignores instructions,
constant repetition), stop here and say so in `context.md` — that's a valid,
useful outcome and it means the rest of this plan can wait.

If it looks reasonable, continue.

---

## 2. Colab — add it to the Gate 1 / Phase D bake-off

This slots into `mega_plan.md` Gate 1 (G1/G2) and Phase D, which the plan
already says only you can run. Concretely, for *this* model:

### 2a. Run it as a G2-style single-stage baseline
`mega_plan.md` G2 asks: does the two-stage Planner->Coder pipeline beat a
single strong model doing English->Python directly, in the same VRAM
budget? Right now G2's own arm is `Qwen2.5-Coder-7B-Instruct` @ Q4_K_M
(~4.7GB). Add this 1.5B distill (~0.94GB) as a second single-stage arm —
it's free to run once C1's held-out set exists (or informally, on the
current 10-example split, understanding the noise `mega_plan.md` Phase C
already flags). If the 1.5B distill gets close to the 7B baseline on
Python-specific tasks, that's a strong signal for D3 (small model,
domain-specialized, beats a larger generalist) independent of anything
this project trains itself.

### 2b. Run it as a D2 arm
Same held-out set, scored the same way as the other D2 arms
(`src/eval/scoring.py`, execution tier via `src/eval/harness.py`). This
tells you where it sits relative to `Qwen2.5-Coder-1.5B-Instruct @ fp16`
(the stock version of its own base model) — i.e., whether the Python-only
distillation recipe earned its keep over just using the stock instruct
model at higher precision.

### 2c. What you need for 2a/2b
Nothing new to build — `04_eval_pipeline.ipynb` already has the scoring
machinery; you're pointing it at one more model. The one wrinkle: this
model expects plain instructions, not this project's plan format, so when
scoring it under G2 use the `english` column directly; when comparing it as
a D2 Coder-stage arm, use the `pseudocode` column as input instead (it may
handle plan-shaped input reasonably even though it wasn't trained on it —
that's part of what you're testing).

---

## 3. If you want to fine-tune it further — read this before touching notebook 03

This is the part that needs the most care, because of what's actually
published:

**`h0ney-badger`'s repo has only one file: `qwen-coder-1.5b-py-Q4_K_M.gguf`
(986MB, already quantized). No fp16/safetensors checkpoint, no standalone
LoRA adapter.** `transformers`/PEFT (what `notebooks/03_stage2_finetune.ipynb`
is built on) cannot load a GGUF file to continue fine-tuning it. There is no
"continue training this exact checkpoint" path available.

Your two real options:

- **Option 1 — use it as-is.** Treat the GGUF as a finished artifact you
  evaluate (§2) and, if it does well, deploy directly. No fine-tuning
  needed. This is the cheapest path and the one to default to unless §2's
  numbers say otherwise.
- **Option 2 — start from its *base*, not from it.** Fine-tune
  `Qwen/Qwen2.5-Coder-1.5B-Instruct` (the stock, fp16, HF-hosted model this
  distill was built from) on this project's own `(pseudocode, python_code)`
  pairs. This is exactly `mega_plan.md` D2's existing
  "`Qwen2.5-Coder-1.5B-Instruct` @ fp16 — the true high-bits arm," so if you
  do this, you are not adding new scope — you're executing a step the plan
  already called for, using the same base model this distill happens to
  confirm is a good choice.

**Do not attempt to dequantize the GGUF back to fp16 and fine-tune that** —
you'd be fine-tuning a lossy re-expansion of a 4-bit model, which is strictly
worse than starting from the real fp16 weights Option 2 already gives you
for free.

### 3a. Adapting notebook 03 for Option 2
`03_stage2_finetune.ipynb` already targets `Qwen2.5-Coder-7B-Instruct`. The
change to point it at `Qwen2.5-Coder-1.5B-Instruct` instead is small
(swap the base model id, re-check `MAX_SEQ_LEN` against the new tokenizer,
confirm the LoRA target modules still match Qwen2.5's architecture — they
should, it's the same model family), but **hold off on making it** until
Phase A (format infrastructure) lands, per `mega_plan.md`'s own sequencing:
notebook 03's Coder input format is still being decided (D-1, DSL vs
comment), and editing it now means editing it twice. If you want this done
in parallel with Phase A rather than after it, say so explicitly — it's a
reasonable call, just a different one than what `mega_plan.md` currently
says, and it should be recorded as a sequencing decision in `context.md`
when made, not silently done.

### 3b. If you want your *own* distillation, not just a fine-tune
`h0ney-badger`'s recipe (14B teacher generates Python tasks + solutions +
tests, execution-filters, ~566 verified samples, QLoRA r=16 3 epochs,
merge, quantize) is close to `mega_plan.md` Phase F1's plan already, just
run once, at small scale, outside this repo. If Phase D's numbers make you
want a bigger/better version of the same idea:
- Reuse `Qwen2.5-Coder-14B-Instruct` as the teacher (resolves D-5 for this
  path — see `mega_plan.md`'s updated D-5 note) or go straight to a 32B
  variant if Colab A100 time allows.
- Reuse this project's own execution-filtering: `src/eval/harness.py`'s
  `run_and_check` already does "does this run to completion," which is the
  same filter the card describes informally.
- This is Phase F work, which `mega_plan.md` gates behind Phase D's
  results. Don't start it before Phase D has an answer — that's the whole
  point of the bake-off.

---

## 4. What's explicitly out of scope for this plan

- **The original "context resets on navigation" question from earlier in
  this session** was about a different repo (`test-test`, with
  `main.py`/`ollama-chat.html`) — a standalone local chat server, not
  anything in `pythonllm`. Nothing here builds that. If you want a
  persistent local chat/serving layer for `pythonllm` specifically, that's
  closest to `mega_plan.md` Phase F4 (measuring the finished pipeline on
  the actual 2060) — worth its own explicit ask once there's a model worth
  serving, rather than building it speculatively now.
- **No code in `src/` or `notebooks/` was changed.** Per §3a, the one
  concrete code change this plan implies (notebook 03's base model) is
  intentionally left to you to sequence relative to Phase A, not done here.

---

## 5. Checklist

- [ ] §1: run the model against a few of this project's own `english`
      prompts locally, decide whether it's worth a Colab session
- [ ] §2a: score it as a G2-style single-stage baseline
- [ ] §2b: score it as a D2 arm against stock `Qwen2.5-Coder-1.5B-Instruct` @ fp16
- [ ] Record the numbers from 2a/2b in `context.md`, same as every other
      Phase D result
- [ ] Decide Option 1 vs Option 2 (§3) based on those numbers
- [ ] If Option 2: decide whether to edit notebook 03 now (parallel with
      Phase A) or wait for Phase A to land, and record that sequencing call
- [ ] If considering Phase F (§3b): revisit only after Phase D has an answer
