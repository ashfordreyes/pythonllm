# Mega plan — comment-format plans, a reasoning Planner, and a 6GB deployment

Working plan for the current research phase. Written 2026-08-27 out of a
research/critique session; supersedes nothing, but it is the first document
that states the **end-to-end deployment target** rather than the next fix.

Read alongside `context.md` (current state) and `docs/next_fixes.md` (open
defects). When a step here is done, record *why* it landed the way it did in
`CHANGELOG.md`, per the convention in `CLAUDE.md`.

---

## 0. The goal, stated as a constraint

Two-stage Planner -> Coder pipeline that runs **entirely on one RTX 2060,
6GB VRAM**, doing Python-only coding tasks. Training happens on Colab
(A100/L4); only inference has to fit the 2060.

### The VRAM budget is the whole design

6144 MiB total. A Windows desktop consumes 0.6-1.0 GB, so the real budget is
**~4.8-5.4 GB** for weights + KV cache + activations, for *both* stages
together.

| Params | fp16 | Q8_0 | Q5_K_M | Q4_K_M |
|---|---|---|---|---|
| 12B | 24 GB | ~12.8 GB | ~8.5 GB | ~7.3 GB |
| 7B | 15 GB | ~8.1 GB | ~5.4 GB | ~4.7 GB |
| 4B | 8.6 GB | ~4.3 GB | ~3.1 GB | ~2.5 GB |
| 3B | 6.6 GB | ~3.3 GB | ~2.3 GB | - |
| 1.5B | ~3.1 GB | ~1.7 GB | - | - |

Consequences that constrain every step below:

- **No 12B model fits at any usable quantization.** Distillation to a smaller
  student is the only path, not one option among several.
- **"High bit-width" caps model size.** True fp16 caps you near 2B params; Q8
  caps you near 4B. "High bits AND large model AND 6GB" is arithmetically
  impossible — the real question is where to sit on the params x bit-width
  curve, and that is an empirical question (step D3).
- **Two separate 7B models do not fit** (~9.4 GB at Q4). Either the stages
  share one base with hot-swapped LoRA adapters, or the Planner runs on CPU,
  or the pipeline does not fit at all. See decision D-2.

### Hardware facts about the target card

The RTX 2060 is Turing, compute capability sm_75:

- **No bf16.** Train in bf16 on the A100; that is fine, it only affects the
  training host.
- **FlashAttention-2 requires Ampere (sm_80+) and will not run.**
- bitsandbytes 4-bit works but is slow on Turing. The deployment path should
  be **llama.cpp / GGUF** or ExLlamaV2, both of which are good on sm_75.
- Memory bandwidth ~336 GB/s, so decode speed is roughly bandwidth over model
  size: a 3.3 GB Q8 model lands around 60-80 tok/s, a 4.7 GB Q4 7B around
  45-60 tok/s. Two-stage inference pays that cost twice per request, which is
  an accepted trade.

---

## 1. Open decisions this plan deliberately does not prejudge

These are gated on measurements, not on argument. Do not let an
implementation step quietly settle one.

- **D-1 — Plan format.** Current `<PLAN>/<STEP>` DSL, or Python-comment
  rendering. Phase A makes this a flag so it can be measured rather than
  guessed. Decided by Gate 1 / step D3.
- **D-2 — Shared base model.** If the Planner and Coder share one base, one
  weight set plus two ~100 MB LoRA adapters fits in 6GB and adapters swap in
  milliseconds. If they do not, the pipeline needs sequential load/unload
  (5-20 s stall per turn) or a CPU-resident Planner. Decided by step D3.
- **D-3 — Params vs bit-width.** 7B @ Q4_K_M vs 3B @ Q8_0 vs 1.5B @ fp16.
  The literature genuinely disagrees in this regime: Dettmers & Zettlemoyer
  (arXiv 2212.09720) put 4-bit on the Pareto frontier for a fixed memory
  budget, while "Not All Bits Are Equal" (arXiv 2510.10964) finds that below
  ~8B, higher-precision weights win on reasoning-heavy tasks. Measure on our
  own task. Decided by step D3.
- **D-4 — Where reasoning lives.** Planner, Coder, both, or neither.
  Argument for the Coder: English -> a handful of known verbs is the easy
  half; pseudocode -> correct pandas/tkinter/requests code is the hard half.
  Decided by step E4.
- **D-5 — Teacher model for distillation.** The
  `projectj/Instinct-Python-Coder-Gemma4-12B-KimiK3` repo named in the
  2026-08-27 session **could not be verified to exist** (no search hit; HF was
  blocked by the session's egress proxy). The nearest verified thing is the
  `gemma-4-12B-coder-fable5-composer2.5-v1` family — Gemma 4 12B-it fine-tuned
  on execution-verified Python, 256K context. Resolve in step P2 before
  building on it.

---

## 2. Phases at a glance

| Phase | What | Who | Blocked by |
|---|---|---|---|
| P | Prerequisites / facts | user + agent | - |
| A | Format infrastructure | agent (CPU) | - |
| B | Notebook updates | agent (CPU) | A |
| C | Eval capacity | user + agent | A |
| Gate 1 | First Colab baseline | user | A, B |
| D | Base model bake-off | user | C, Gate 1 |
| E | Reasoning traces | user + agent | D |
| F | Distill + deploy to 6GB | user | D, E |

**A hard boundary runs between A/B and everything downstream.** Phases A and
B are verifiable on CPU in this repo. Everything from Gate 1 on needs a GPU
and model weights, and it produces the measurements that decide the
architecture. Do not bundle across that boundary — an agent that keeps going
past it is committing to unmeasured assumptions.

---

## Phase P — Prerequisites

### P1. Measure the real free VRAM on the 2060
Run `nvidia-smi` on the target machine with the desktop up, as it will be in
normal use. Record the actual free figure in `context.md`. Every budget in
this document is an estimate until this number exists.

**Done when:** a measured MiB figure is in `context.md`.

### P2. Resolve the teacher model (D-5)
Confirm whether `projectj/Instinct-Python-Coder-Gemma4-12B-KimiK3` exists and
paste its model card. If it does not, pick a verified teacher — the
`gemma-4-12B-coder-fable5-composer2.5-v1` family, or a large Qwen2.5-Coder.

Two license consequences to check while there:
- **Qwen2.5-Coder-3B is under the Qwen Research License (non-commercial)**,
  unlike 0.5B/1.5B/7B/14B which are Apache-2.0. This matters if the 3B @ Q8
  arm wins in D3.
- **Anything Gemma-derived carries the Gemma terms**, and those propagate to
  a student trained on its outputs.

**Done when:** a named teacher with a verified URL and a license note is in
`context.md`.

### P3. Add CI for the data validator
The repo has no `.github/workflows` at all, so nothing runs
`scripts/check_data.py` automatically. It enforces the invariant that line N's
`pseudocode` is byte-identical across `data/stage1_planner/` and
`data/stage2_coder/` — the invariant `src/splits.py` depends on for
index-based splitting. Phase A rewrites both files, which is exactly when that
check earns its keep.

**Done when:** a push runs `python scripts/check_data.py` and fails the build
on a non-zero exit.

---

## Phase A — Format infrastructure (agent, CPU-verifiable)

**Principle: the plan format is a *rendering*, not the schema.** The valuable
part of the DSL is the verb vocabulary and the ordered step list, not the
`<PLAN>/<STEP>` serialization. Keep the schema; make the wire format a
parameter. This turns D-1 into a one-line flag instead of a repo fork, and it
preserves `VERBS`, `validate_plan`, `score_plan`'s `verb_sequence_match`,
`classify_plan_error`, and `check_data.py`.

Target rendering, with a controlled first word per step so the verb stays
recoverable:

```
DSL (today):  <PLAN><STEP> FETCH response FROM "https://..." <STEP> PARSE response USING json </PLAN>
Comments:     # 1. fetch response from "https://..."
              # 2. parse response as json
```

Free-form prose comments were rejected: marginally more natural, but they
destroy every structural metric in `src/eval/scoring.py`.

### A1. Add the renderer/parser pair
In `src/dsl/`, add `render_plan(steps, format=...) -> str` and make
`parse_plan(text)` dispatch on the format it sees. `src/dsl/validator.py`
lines 30-32 currently hard-require the `<PLAN>`/`</PLAN>` delimiters; that
becomes one branch of two. `validate_plan`'s per-step logic is unchanged.

**Done when:** `parse_plan(render_plan(steps, fmt)) == steps` round-trips for
both formats, over all 50 rows in the dataset.

### A2. Add the format flag
One module-level constant (alongside `SPLIT_SEED` in spirit — part of the
contract, changing it changes what the model sees). Both formats must stay
working simultaneously so D-1 can be measured.

**Done when:** flipping the flag changes rendered output and nothing raises.

### A3. Extend `classify_plan_error`
`src/eval/scoring.py`'s `"bad delimiters"` bucket is DSL-specific and needs a
comment-format sibling (e.g. "no numbered steps found") so the error
breakdown chart still classifies rather than dumping everything into
`"other"`.

**Done when:** a deliberately malformed comment plan lands in a named bucket.

### A4. Regenerate both JSONL files
Derive the comment form by parsing the existing DSL and re-rendering — do not
hand-edit. **Both files must be rewritten together**: `check_data.py` enforces
byte-identical `pseudocode` across them and will fail otherwise.

**Done when:** `python scripts/check_data.py` prints `OK: 50 pairs` and the
split report is unchanged.

### A5. Update `src/ui/generate.py`
- Rewrite `PLANNER_SYSTEM_PROMPT` for the chosen format.
- **Delete `_clean()` and the `skip_special_tokens=False` argument.** They
  exist only to keep `<PLAN>`/`<STEP>` visible through decoding. With comment
  format there are no special tokens to protect.
- The Planner likely now needs `strip_fences=True` as well — a model emitting
  `#` comments will often wrap them in a ```python fence, and that would land
  as literal backticks in the console panel.

**Done when:** `ui.console` still imports without torch/transformers, and the
stub-driven console test still renders.

### A6. Verify nothing regressed
Re-run the checks `context.md` records from the last verification pass:
split determinism/disjointness/40-10, both stages' eval splits byte-identical
on `pseudocode`, reference-vs-reference scoring 50/50 on all three plan tiers
and on `code_parses`, 8/50 `self_contained`, 4/4 `executes`, all three plot
functions producing non-empty PNGs.

**Done when:** every figure above matches, or a deviation is explained.

---

## Phase B — Notebook updates (agent, separate session)

Kept separate from Phase A on purpose: notebooks are JSON blobs with noisy
diffs, nothing here can execute them, so verification is inspection-only. A
broken cell hides easily in a large mixed diff and only surfaces later while
burning Colab compute.

### B1. Strip the special-token machinery from `02_stage1_finetune.ipynb`
Delete §3's `add_special_tokens` call, §4's vocab-size assert, and §5's
`trainable_token_indices`. Per `context.md`, this machinery caused three
separate defects already: the shrinking `resize_token_embeddings` bug, the
untrained DSL embedding rows, and the 4.3 GB artifact. Comment format needs
zero tokenizer surgery, so the entire failure class disappears.

**Note:** this is also what unblocks D-2 — without registered special tokens
the Planner uses a stock tokenizer, which is a precondition for sharing one
base with the Coder and hot-swapping adapters.

### B2. Raise `MAX_SEQ_LEN` in notebook 02, on evidence
Currently 512. Comment plans are more verbose than the DSL, and reasoning
traces (Phase E) will add several hundred tokens. **Do not guess** — tokenize
the dataset and take the p95 length, then pick. Probably 1024.

### B3. Update `03_stage2_finetune.ipynb` §5
The Coder's user-message content changes format. `MAX_SEQ_LEN=1024` was sized
on the `python_code` side (~2900 chars) so it is probably still fine, but
re-check against the new input length.

### B4. Update `04_eval_pipeline.ipynb`
Remove §3's DSL-token assert; update §4's prompt. §7 already generates from
both reference and generated plans — keep that, it is what measures the
error-compounding cost of the two-stage design.

### B5. Update the docs that describe the removed machinery
`docs/colab_setup.md` §8/§9 references the special-token flow. `CLAUDE.md`'s
Design section describes the DSL tokens as part of the Planner. Both need to
match reality after B1.

---

## Phase C — Eval capacity

**This phase exists because the current eval cannot settle any decision.**
`scripts/check_data.py` on 2026-08-27 reports **3 of 50** reference snippets
running cleanly, 1 of them held out, against a 10-example eval set. (The
count is environment-dependent — `context.md`'s earlier entry recorded 4/50
on a differently-provisioned machine, since `is_executable_example` gates on
what actually imports and runs there. Treat it as "3-4, and which ones varies
by host.")

So the execution tier is no longer fully dormant — the stratification in
`src/splits.py` did its job — but it now reports a rate over **a single
example**. Comparing two plan formats on n=10, with the one tier that
measures correctness resting on n=1, produces noise that reads like signal.

### C1. Expand the held-out set toward ~100 verified pairs
Hand-checked, execution-verified, quarantined from training. This is the
single highest-value unglamorous task in the plan.

### C2. Raise the executable fraction
`src/splits.py` already stratifies by `is_executable_example` so runnable
references land proportionally in eval. That machinery is fine; it has almost
nothing to work with. New examples should be written to be runnable — no
network, no GUI mainloop, no missing CSV fixtures — wherever the task allows.

### C3. Add public benchmarks alongside the in-repo eval
EvalPlus (HumanEval+/MBPP+), DS-1000 for the pandas/numpy side, BigCodeBench
for library use. These give an absolute reading that 100 in-repo examples
cannot.

**Done when:** the execution tier reports a real rate instead of "not
attempted", on a held-out set large enough that a 5-point difference means
something.

---

## Gate 1 — First Colab baseline (user)

Blocking. Nobody but you can run this.

### G1. Few-shot both formats, un-tuned
Prompt an un-tuned instruct model with both the DSL and the comment format,
few-shot, and score on the held-out set. **Run this before any fine-tune.**
If prompting already gets most of the way, the fine-tune's real job is
distilling that behavior into a smaller model — a much better-posed project
than teaching it from 40 examples.

### G2. Run the one-stage baseline
One `Qwen2.5-Coder-7B-Instruct` @ Q4_K_M doing English -> Python directly, in
the same 5 GB. **If the two-stage pipeline does not beat this, the
architecture is not earning its complexity.** This is the most important
single experiment in the document and it is cheap.

---

## Phase D — Base model bake-off (user, Colab)

### D1. Evaluate quantized, not fp16
Score what ships. An fp16 A100 number does not predict Q4 behavior on a 2060.

### D2. Arms
- `Qwen2.5-Coder-7B-Instruct` @ Q4_K_M (~4.7 GB) — current baseline
- `Qwen2.5-Coder-3B-Instruct` @ Q8_0 (~3.3 GB) — the "high bits" arm (license, P2)
- `Qwen2.5-Coder-1.5B-Instruct` @ fp16 (~3.1 GB) — the true high-bits arm
- a 4B (Gemma 4 4B or similar) @ Q6_K
- the 12B teacher, unquantized — as a **ceiling**, not a candidate

### D3. Settle D-1, D-2, D-3 from the results
Format, shared base, and the params/bit-width point all fall out of this
table. Record the decision and the numbers behind it in `context.md`.

---

## Phase E — Reasoning traces

### E1. Generate traces by rationalization
You have `(english, pseudocode)` and need `(english, reasoning + plan)`.
Nobody hand-writes those. Give a teacher the English **and the reference
plan**, and ask for the reasoning that would lead there.

### E2. Verify traces by re-extraction
Discard any trace whose extracted plan does not match the reference. The
filter already exists: `score_plan(generated, reference).verb_sequence_match`
in `src/eval/scoring.py`. This is rejection sampling built almost entirely
from code already in the repo.

### E3. Use plain-text delimiters
`<think>...</think>` as **ordinary text**, tokenizing into normal tokens. Do
not register them as special tokens — that is the exact trap of B1, and
re-entering it would undo the simplification.

### E4. Strip reasoning before the Coder sees it
The Coder is trained on plan -> code; reasoning in its input is
out-of-distribution, and those tokens burn KV cache the 2060 does not have.
Add an `extract_plan()` helper between `plan_fn` and `code_fn` in the console
wiring. Then measure Planner-reasoning vs Coder-reasoning to settle D-4.

### E5. Planner fine-tune config
- QLoRA NF4 4-bit, `r=32`, `alpha=64`, `dropout=0.05`, all linear projections
  (`q,k,v,o,gate,up,down`), no `trainable_token_indices`.
- Keep the existing prompt/completion loss masking; the completion is now
  reasoning + plan and both should be trained.
- LR 1e-4 cosine, 3-5% warmup, 2-4 epochs, batch 1 x grad-accum 16, bf16 +
  gradient checkpointing on the A100.
- Keep appending `tokenizer.eos_token` — with longer, freer completions,
  teaching the model to stop matters more than it did.

**Caveat that outranks all of E: 40 training rows will not teach reasoning.**
Reasoning SFT wants thousands. Phase C is a prerequisite in practice.

---

## Phase F — Distill and deploy

### F1. Sequence-level distillation, not logit distillation
A Gemma teacher (262K vocab) and a Qwen student (152K vocab) would need
cross-tokenizer machinery — ULD (arXiv 2402.12030), MultiLevelOT (arXiv
2412.14528), byte-level interfaces (arXiv 2604.07466). Interesting, wrong
priority. Instead: run the teacher on the A100 to generate
`(plan, python)` pairs, **filter by execution**, and SFT the student on what
survives. No tokenizer alignment needed. It is also how the Gemma coder model
was built in the first place.

A same-family teacher (a large Qwen2.5-Coder) would additionally leave true
logit KD available later.

### F2. Merge, then quantize
Merge the LoRA into fp16 first, then quantize to GGUF. Do not try to serve a
LoRA on top of already-quantized weights.

### F3. Build the imatrix from in-domain data
Calibrate on your own plan -> Python pairs, not a generic corpus. This is a
free quality win at Q4 and it partially closes the gap the "high bits"
instinct is worried about. Consider QAT if more is needed.

### F4. Measure on the actual 2060
Not on Colab. Weights + KV + activations against the P1 number, plus real
tok/s for both stages and the adapter-swap latency if D-2 went that way.

**Done when:** the full pipeline answers a held-out task on the 2060, within
the measured VRAM budget, at a latency you will actually tolerate.

---

## 3. The thing this plan cannot fix

`data/stage1_planner/englishtopseudo.jsonl` and
`data/stage2_coder/pseudotopython.jsonl` hold **50 rows each** — 40 train, 10
eval. At that size, base model choice is not the limiting factor and no amount
of model shopping will change it. Phases A, B and D are all cheap; Phase C is
the one that actually moves the ceiling, and it is the least fun. Do it
anyway.
