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

## 2026-08-27 — mega_plan review: sequential stages, Fedora host

First review pass over `docs/mega_plan.md`. Two premises in it were wrong and
both change what the later phases measure.

- **The stages run sequentially. They never co-reside.** Planner emits a plan,
  the plan goes to the Coder as a prompt, the Coder emits the code. The plan
  had costed both models as simultaneously VRAM-resident, which made "two 7B
  models are ~9.4GB at Q4 and do not fit" a wall. It is not one — the pipeline
  needs one ~5GB slot reused twice per request.
- **Consequence 1 — the budget is per stage.** D-3 (params vs bit-width) is
  now two decisions and they may differ. A 1.5B Planner with a full-budget
  Coder is legal; so is 7B @ Q5_K_M (~5.4GB) for the Coder alone.
- **Consequence 2 — D-2 is reframed.** Not "does it fit" but "what does the
  handoff cost". Shared base + hot-swapped LoRA adapters is milliseconds and
  preferred; a `llama-swap`-style process swap is the fallback, and it is
  cheaper than the plan claimed — ~1-3s page-cached, not 5-20s, because the
  copy is PCIe-bound rather than a disk read. That the fallback works is what
  keeps model choice free: do not pick a shared base only to avoid the swap.
- **Consequence 3 — system RAM becomes the binding resource** if the swap
  route wins, since it is only cheap while both GGUFs stay in the page cache.
  New step P4 records RAM/disk/disk-type. 16GB floor, 32GB comfortable.
- **Unchanged:** no 12B fits (7.3GB at Q4 vs ~5.9GB even headless), so
  distillation remains the only path.
- **The host is Fedora 44, not Windows.** Budget figures revised to
  ~5.3-5.6GB with a GNOME/Wayland desktop on the card and ~5.9GB headless or
  with the display on an iGPU. P1 now takes three readings instead of one and
  new P1b picks the display configuration, because every later measurement has
  to be taken under whichever one is chosen. Secondary Linux wins recorded: no
  WDDM VRAM oversubscription (over-budget configs OOM loudly instead of
  silently spilling at 5-10x latency), and an easier llama.cpp CUDA build.
  Setup cost: RPM Fusion `akmod-nvidia`, plus MOK enrollment if Secure Boot is
  on.
- **Corrected fact:** the sm_80 FlashAttention requirement is about the
  `flash-attn` PyTorch package. llama.cpp's `--flash-attn` is a separate
  implementation expected to work on Turing — worth confirming on the card,
  since it affects KV-cache size.

**Open — for P1/P1b/P4, nobody but the user can answer:** measured free VRAM
in all three display configurations; whether the CPU has an iGPU; system RAM,
free disk and disk type.

**Open — the review is not finished.** This pass covered §0, D-2, D-3, P1, P4,
G2, D2/D3 and F4. Phases A-C, E and F have not been reviewed.

## 2026-08-27 — 6GB deployment target written down (`docs/mega_plan.md`)

A research session on base-model selection surfaced that the project's actual
constraint — the whole pipeline running on an RTX 2060, 6GB VRAM — was
nowhere in the repo. `docs/mega_plan.md` now holds it, phased P/A/B/C/D/E/F.
The parts that change how current work should be sequenced:

- **Phase A reframes the format question.** The `<PLAN>/<STEP>` DSL is a
  *rendering* of the schema, not the schema. Keeping `VERBS` + the ordered
  step list canonical and making the wire format a flag lets DSL vs
  Python-comment output be measured rather than argued, and leaves
  `validate_plan`, `score_plan`, `classify_plan_error` and
  `scripts/check_data.py` working across both.
- **Dropping the DSL special tokens is load-bearing, not cosmetic.** Comment
  format needs no tokenizer surgery, which retires the machinery behind three
  recorded defects (the shrinking `resize_token_embeddings`, the untrained
  embedding rows, the 4.3GB artifact) *and* is the precondition for the two
  stages sharing one base with hot-swapped LoRA adapters. **Superseded in part
  by the review entry above:** the stages run sequentially and never
  co-reside, so the shared base is a latency optimization, not a fit
  requirement. Dropping the special tokens is still load-bearing for the three
  defects, and still what unblocks the adapter route.
- **The eval cannot currently settle anything.** `scripts/check_data.py` now
  reports 3/50 references running cleanly with 1 held out (not the 4/50-and-
  none-held-out in the entry below: stratification fixed the held-out half,
  and the count itself is host-dependent). So the execution tier reports a
  rate over a single example, and plan formats would be compared on n=10.
  Phase C
  (expand toward ~100 verified pairs, raise the executable fraction) is a
  prerequisite for the Colab baseline, not a follow-up to it.
- **Cheapest decisive experiment, not yet run:** one
  `Qwen2.5-Coder-7B-Instruct` @ Q4_K_M doing English -> Python directly in the
  same 5GB. If two stages don't beat it, the architecture isn't earning its
  complexity.

**Open — blocks Phase F:** the teacher model the session was premised on,
`projectj/Instinct-Python-Coder-Gemma4-12B-KimiK3`, could not be verified to
exist (no search hit; huggingface.co blocked by the session's egress proxy).
Nearest verified equivalent is the `gemma-4-12B-coder-fable5-composer2.5-v1`
family. Resolve before building on it. Licenses to check at the same time:
Qwen2.5-Coder-3B is Qwen Research (non-commercial) unlike its siblings, and
Gemma terms propagate to a student trained on Gemma outputs.

**Open:** nothing in `docs/mega_plan.md` is implemented. Phase A has not
started; no code, notebook or data file has changed for it.

## 2026-08-27 — Stage 2 notebook written; `04` section 7 wired up

- **`notebooks/03_stage2_finetune.ipynb` now exists.** Base model is
  `Qwen2.5-Coder-7B-Instruct` — the Instruct variant was picked specifically
  so this notebook and `src/ui/generate.py`'s existing chat-template-based
  `make_coder` machinery stay compatible without changes. It mirrors notebook
  02's shape closely; see the CHANGELOG entry for the concrete differences
  (no special-token section, `MAX_SEQ_LEN=1024` not 512, system prompt
  imported from `src/ui/generate.py` instead of duplicated).
- **The `CODER_SYSTEM_PROMPT` guess is resolved.** The previous entry below
  flagged it as untrusted until a real Stage 2 notebook trained against it.
  Notebook 03 now imports it directly rather than writing a second copy, so
  the two can't drift — if the prompt wording ever needs to change, it
  changes in `src/ui/generate.py` and both the notebook and the console pick
  it up.
- **`04_eval_pipeline.ipynb` section 7 is runnable now**, not just
  documented. It loads the Stage 2 adapter, generates code two ways
  (reference pseudocode, and the plan section 4 already generated), and
  scores both with `score_code`/`aggregate`. Not yet run end-to-end on a real
  GPU in this session — that still has to happen interactively on Colab, same
  as every other notebook here so far.
- **Open follow-up:** nobody has actually trained a Stage 2 adapter yet, so
  none of this — notebook 03's own sanity check, or `04`'s section 7 — has
  been exercised against real weights. Next Colab session should run
  `03_stage2_finetune.ipynb` end to end and confirm the adapter lands at
  `MyDrive/pythonllm_checkpoints/stage2_coder/`, matching what `04` now
  expects.

## 2026-08-27 — Colab chat console (`src/ui/`)

- **The Stage 2 notebook still doesn't exist**, and `docs/colab_setup.md`
  referred to `03_stage2_finetune.ipynb` in §8/§9 as if it did. Fixed the doc
  (scope note + what writing 03 would involve). The one non-obvious part
  recorded there: the Coder gets **no DSL special tokens** — it reads the plan
  as ordinary text, and only the Planner has to *emit* `<PLAN>`/`<STEP>` as
  single tokens, so notebook 03 skips notebook 02's section 3 entirely.
- **`src/ui/console.py` renders, it does not own a model.** It takes
  `plan_fn`/`code_fn` callables, which is what lets it ship before the Coder
  exists and be exercised with stubs on a CPU runtime. Keeping torch,
  transformers and ipywidgets out of module scope is part of the same
  decision — `ui.console` has to import off a GPU runtime or the layout can't
  be tested without burning compute units.
- **Streaming is the reason `src/ui/generate.py` exists** rather than reusing
  notebook 04's `generate_plan` verbatim: eval wants one final string, a chat
  UI wants tokens as they arrive. Same chat template and
  `skip_special_tokens=False` otherwise. If notebook 04 ever grows a shared
  helper, that's the function to converge on.
- **Open question:** the coder system prompt in `generate.py`
  (`CODER_SYSTEM_PROMPT`) is a guess — whatever notebook 03 actually trains
  with must replace it, or the console will prompt the Coder differently than
  it was tuned. Worth pinning both stages' prompts in one module once 03 is
  written.

## 2026-08-27 — Held-out split + layered eval scoring + PNG charts

Closed the last two `docs/next_fixes.md` items together, since neither is
usable alone: a split with nothing to score, or charts with no data source.

- **The split had to be index-based.** `scripts/check_data.py` enforces that
  line N's `pseudocode` is byte-identical in the Stage 1 and Stage 2 JSONLs.
  `src/splits.py` shuffles `range(n)` under `random.Random(SPLIT_SEED)`, so
  both stages hold out the same rows from the seed alone — no split file, no
  split metadata in the training payload, and end-to-end eval necessarily
  scores both stages on the same tasks. 50 -> 40 train / 10 eval.
- **Execution-based scoring measures almost nothing on this data.** Only 8 of
  50 reference snippets are statically safe to `exec`, and only 4 actually
  run — the other 4 die on `FileNotFoundError` from calls like
  `pd.read_csv("sales.csv")`, which no import-level check can catch. Rather
  than chase that with more static analysis, `is_executable_example()` gates
  empirically on the *reference*: if the reference can't run here, a correct
  generation couldn't either, so the example is excluded from that tier's
  denominator instead of counted as a failure.
- **So scoring is layered** (`src/eval/scoring.py`): plan well-formedness,
  verb sequence, exact match, code parse, self-containment, execution — each
  with its own `attempted` count, so a dormant tier reports "not attempted"
  rather than 0%. `harness.run_and_check(code, check=lambda ns: True)` is
  reused for the execution tier; passing a trivial check makes the metric
  "ran to completion without raising", which is all this data supports (no
  example carries a per-example checker).
- **Charts** are `src/eval/plots.py`, not `harness.py` — executing code and
  drawing charts are different jobs. matplotlib/`Agg`, light mode, palette
  `#2a78d6`/`#eb6834` validated for CVD separation.
- Notebook 02 now trains on the train split, evaluates each epoch, dumps
  `trainer.state.log_history` + a loss PNG (new section 7b) to Drive
  alongside the adapter, and section 10 generates for a **held-out** task —
  closing the "code-14 is a smoke test, not an eval" note below.
  `load_best_model_at_end` was rejected: choosing a checkpoint by the loss of
  10 examples selects on noise.
- `notebooks/04_eval_pipeline.ipynb` now exists (Stage 1 end to end; Stage 2
  and end-to-end sections documented but gated on notebook 03).

**Open limitation, deliberately not papered over:** with `SPLIT_SEED = 0`
none of the 4 executable examples land in the held-out set, so `code_executes`
reports "not attempted". Picking a seed that gives better coverage would be
choosing the split after seeing the answers. `scripts/check_data.py` prints
this on every run; the real fix is more executable examples in the dataset.

Verified locally (no GPU): split determinism/disjointness/40-10, Stage 1 and
Stage 2 eval splits byte-identical on `pseudocode`, reference-vs-reference
scoring 50/50 on all three plan tiers and on `code_parses`, 8/50
`self_contained`, 4/4 `executes`, all three plot functions producing non-empty
PNGs (including on empty input), and `scripts/check_data.py` still printing
`OK: 50 pairs`. Not yet run on Colab.

## 2026-08-27 — Stage 1 DSL special tokens were never actually trained

Investigating `docs/next_fixes.md` item 1 ("`LoraConfig` missing
`modules_to_save`") showed the filed diagnosis was backwards, and the
underlying defect invalidates the first training run's handling of the DSL
tokens.

- `Qwen2.5-7B-Instruct` declares `vocab_size: 152064` but its stock
  tokenizer only reaches id 151664, so `<PLAN>`/`</PLAN>`/`<STEP>` land at
  151665-151667 and `len(tokenizer)` is 151668 — inside the existing
  matrix. `model.resize_token_embeddings(len(tokenizer))` in section 4 was
  therefore *shrinking* the embedding 152064 -> 151668, never growing it.
  The notebook's claim that skipping it would break the first `<PLAN>` id
  lookup was wrong for this base model.
- `prepare_model_for_kbit_training` freezes every base parameter and
  `get_peft_model` only unfreezes LoRA plus `modules_to_save`, so
  `embed_tokens` and `lm_head` were frozen for the whole run. A shrinking
  resize truncates rows without re-initializing them. **The three DSL token
  rows in the existing Drive adapter are untrained base-model values on
  both the input and output side** — the adapter should be retrained rather
  than evaluated.
- The ~4.3GB artifact was a side effect: PEFT's
  `save_embedding_layers="auto"` triggers on `config.vocab_size` differing
  from the base config, dumping fp32 copies of two frozen matrices.

Fixed in `02_stage1_finetune.ipynb` by removing the resize (replaced with
an assert that the ids fit the base vocab) and adding
`trainable_token_indices={"embed_tokens": ids, "lm_head": ids}` to
`LoraConfig` — PEFT's `TrainableTokens`, which trains exactly those three
rows (~21K parameters) and supports both `nn.Embedding` and the untied
`nn.Linear` `lm_head`. `modules_to_save=["embed_tokens", "lm_head"]` was
rejected: ~1.09B trainable fp32 parameters, roughly 13-17GB of extra
gradient and optimizer state (likely OOM on the L4 the notebook supports),
a multi-GB artifact, and it would still not have fixed reloading.

Consequences elsewhere: `docs/colab_setup.md` §8 no longer needs
`base.resize_token_embeddings(...)` before `PeftModel.from_pretrained` (a
note covers pre-2026-08-27 adapters, which do), and `docs/next_fixes.md`
items 1 and 2 are both closed by this one change. Not yet verified on
hardware — the checks to run during the next Colab run are listed in the
plan and in the notebook's section 4/5/8 markdown.

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
