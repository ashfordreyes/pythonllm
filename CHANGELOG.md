# Changelog

Every entry says *why* the change was made — the problem, error, or request
that prompted it — not just what changed, per the `## Changelog` convention
in `CLAUDE.md`. Newest first, grouped by date. Reconstructed on 2026-08-27
from `git log`, `docs/prompts.txt`, `context.md`, and local session
transcripts; see those for full detail behind any entry here.

## 2026-08-27

### Added

- **`notebooks/03_stage2_finetune.ipynb` — the Coder QLoRA fine-tune.**
  Planned with a subagent-assisted session that read `context.md`,
  `docs/colab_setup.md` §9, `src/dsl/schema.py`, `src/eval/scoring.py`,
  `src/splits.py` and `src/ui/generate.py` first, since the notebook had to
  match decisions already made elsewhere in the repo rather than invent new
  ones. Base model is `Qwen2.5-Coder-7B-Instruct` (the Instruct variant, so
  `apply_chat_template` and `src/ui/generate.py`'s existing chat-based
  `make_coder` work against it unmodified) on
  `data/stage2_coder/pseudotopython.jsonl`. Structurally mirrors notebook 02,
  with three deliberate differences: no DSL special-token registration (only
  the Planner has to emit `<PLAN>`/`<STEP>` as single tokens, so this
  notebook has one fewer numbered section and no `trainable_token_indices`);
  `MAX_SEQ_LEN` raised from 512 to 1024 because reference `python_code`
  completions run up to ~2900 characters, longer than any Stage 1 pseudocode
  plan, and 512 would have silently truncated the longest training examples;
  and the system prompt is *imported* from `src/ui/generate.py`'s
  `CODER_SYSTEM_PROMPT` rather than duplicated inline, closing the exact gap
  `context.md`'s 2026-08-27 entry flagged (that prompt was "a guess" that had
  to be replaced by whatever this notebook trained with, or the console would
  prompt the Coder differently than it was tuned). The notebook's own sanity
  check calls `ui.generate.make_coder` directly so it exercises the same code
  path the console uses.
- **Wired up `04_eval_pipeline.ipynb` section 7 (Stage 2 + end-to-end
  scoring), previously documented but not runnable** because it needed a
  coder adapter that didn't exist until the entry above. Added
  `CODER_ADAPTER_DIR`/`CODER_BASE_MODEL`/`CODER_RESULTS_DIR`/
  `CODER_MAX_NEW_TOKENS` to the config cell, a cell that loads the Stage 2
  adapter the same way section 3 loads Stage 1's, and cells that score both
  "Stage 2 alone" (reference pseudocode -> generated code) and "end to end"
  (english -> generated plan -> generated code) with `score_code`, writing
  JSON results and score-breakdown PNGs to
  `MyDrive/pythonllm_checkpoints/eval_stage2/`. Reuses `ui.generate.make_coder`
  for generation rather than a third hand-rolled `generate()` call.
- Added an optional `title` parameter to `src/eval/plots.py`'s
  `plot_loss_curve` (default unchanged, `"Stage 1 training loss"`) because
  reusing it as-is in notebook 03 would have mislabeled the Stage 2 loss
  chart. Notebook 03 passes `title="Stage 2 training loss"`.
- Updated `notebooks/README.md` and `docs/colab_setup.md` to stop describing
  `03_stage2_finetune.ipynb` as unwritten, and to call out where it differs
  from notebook 02 for anyone comparing the two side by side.
- **An ASCII chat console for the two stages (`src/ui/`), plus
  `docs/colab_console.md`.** Asked whether the Colab experience could look
  like a CLI tool's box-drawn "almost-GUI" instead of raw `print()` output,
  and whether that could appear beneath a single cell. It can: Colab renders
  stdout as monospace and ipywidgets underneath it, so `src/ui/console.py`
  draws titled panels for the task, the plan and the code, colors the DSL
  special tokens/verbs/literals, syntax-highlights the Python with pygments,
  and `Console.launch()` puts a text box and buttons under the cell.
  - The console takes two *callables* (`plan_fn`, `code_fn`) rather than a
    model, so it works with the Stage 1 adapter alone — the Coder does not
    exist yet — and with stubs on a CPU runtime, which is how the padding and
    wrapping were checked without spending compute units. It imports neither
    torch nor ipywidgets at module level for the same reason.
  - Generations stream. A 7B model on an L4 takes tens of seconds per answer
    and a silent cell is indistinguishable from a hung one, so
    `src/ui/generate.py` wraps `TextIteratorStreamer` and yields cumulative
    snapshots that the console re-renders on. It keeps notebook 04's chat
    template and `skip_special_tokens=False` so `<PLAN>`/`<STEP>` stay visible,
    and strips the ``` fences a code-pretrained base tends to emit.
  - Panel padding is computed from a visible length that ignores SGR escapes,
    and text is wrapped *before* highlighting, because otherwise every colored
    line pads short by the length of its escape codes and long lines split an
    escape sequence across two rows. The palette is 256-color mid-tones: the
    ends of the ramp read on one Colab theme and vanish on the other.
  - A widget-free `Console.repl()` is the fallback, since Colab pins its own
    widget manager and `pip install -U ipywidgets` is as likely to break
    rendering as fix it.

### Changed

- **`docs/colab_setup.md` now says up front that it is Stage 1 only.** Asked
  whether it explains how to get the notebook that trains the Coder — it does
  not, and worse, §8/§9 referred to `03_stage2_finetune.ipynb` as though it
  existed. It has never been written. Added a scope note in the intro, dropped
  the stale "point `03_stage2_finetune.ipynb` at the Drive path" instruction,
  and added a §9 subsection describing what writing it would involve (notebook
  02's shape, `Qwen2.5-Coder-7B`, the stage 2 JSONL, and *no* DSL
  special-token registration — only the Planner has to emit them as single
  tokens).

### Changed (uncommitted at time of writing)

- **Stratified the train/eval split so the execution-scoring tier stops
  reporting "not attempted."** With `SPLIT_SEED = 0`, none of the dataset's
  4 executable reference snippets landed in the held-out set (~35% of seeds
  miss all 4, checked across 200), so `code_executes` was permanently dark
  regardless of model quality. Re-seeding until the draw looked better was
  rejected as choosing the split after seeing the outcome; adding fixture
  files (`sales.csv` etc.) was rejected because the dataset's file paths are
  deliberately illustrative and the executable rows don't happen to be the
  ones missing files anyway. Fixed by having `src/splits.py` accept a
  `priority` set of indices and split that pool separately from the rest
  under the same seed, then merge — a static, pre-computed fact about the
  *reference* code, not a search over generations or scores. `docs/next_fixes.md`
  and `context.md` record the before/after (`4/50` references executable,
  `1/10` now deterministically held out).

- **Bug fixes & prompt log (`85546db`).** Closed both remaining
  `docs/next_fixes.md` items in one pass because neither was independently
  usable — a split with no scoring is inert, and scoring with nothing to
  chart is too:
  - Added `src/splits.py` (index-based 40/10 split, shared seed) because
    `scripts/check_data.py` requires line N's `pseudocode` to be
    byte-identical between the Stage 1 and Stage 2 JSONL files, so any
    train/eval split has to be derived the same way for both rather than
    stored per-file.
  - Added `src/eval/scoring.py` because pure execution-based scoring
    measures almost nothing here: only 8 of 50 references are statically
    safe to `exec`, and only 4 actually run. Layered scoring (plan
    well-formedness, verb sequence, exact match, code parse,
    self-containment, execution) with a per-tier `attempted` count lets a
    dormant tier report "not attempted" instead of a misleading 0%.
  - Added `src/eval/plots.py` (matplotlib/`Agg`, CVD-checked palette) kept
    separate from `harness.py` since executing code and drawing charts are
    different responsibilities.
  - `notebooks/02_stage1_finetune.ipynb` now trains on the train split,
    evaluates each epoch, and section 10's sanity check generates for a
    **held-out** task instead of a training example. `load_best_model_at_end`
    was considered and rejected — choosing a checkpoint by 10 examples'
    loss selects on noise.
  - Added `notebooks/04_eval_pipeline.ipynb` (Stage 1 end-to-end scoring).
  - Updated `docs/colab_setup.md` §8/§9 so the resume-from-Drive snippet no
    longer calls `resize_token_embeddings` (see below) and points at the
    new eval notebook, with a note that pre-2026-08-27 adapters still need
    the old resize since they were trained with it.

### Discussed, not (yet) separately committed

- Diagnosing `docs/next_fixes.md`'s first item ("`LoraConfig` missing
  `modules_to_save`") turned up that the filed premise was backwards — see
  the `context.md` entry "Stage 1 DSL special tokens were never actually
  trained," folded into the `85546db` fix above rather than logged as its
  own commit.

## 2026-08-26

### Changed

- **Add files via upload (`bedffb4`).** Follow-up to the first real Colab
  training run (`trainer.train()` completed, adapter saved to Drive):
  - Added `context.md` to start tracking decisions/open questions
    separately from `CLAUDE.md`, since the session surfaced several
    (no held-out split, no plotting, validator not wired to model output)
    that weren't settled enough to be conventions yet.
  - Added `docs/weight_saving.md`, requested directly ("Create a file to
    /docs/ called 'weight_saving.md' with these details for my use and
    future contributors/users use") after a discussion about whether
    Google Drive alone was a safe place to keep trained weights.
  - Fixed `notebooks/02_stage1_finetune.ipynb` section 10 crashing with
    `KeyError: 'shape'` (surfacing as `AttributeError` deep in
    `tokenization_utils_base.py`) — `apply_chat_template(...,
    return_tensors="pt")` returned a `BatchEncoding` rather than a raw
    tensor on the `transformers` version section 2b installs, and
    `BatchEncoding` has no `.shape`. Fixed by pinning `return_dict=True`
    and indexing `input_ids`/`attention_mask` explicitly so behavior
    doesn't depend on the installed library version; documented as a
    troubleshooting entry in `docs/colab_setup.md` §7.
  - Extended `docs/colab_setup.md` with the local-inference path (load the
    stock base model, apply the Drive adapter with
    `PeftModel.from_pretrained` — 4-bit quantization was only needed for
    Colab GPU memory during training, not for later use) and weight-hosting
    guidance (Hugging Face Hub for the adapter itself; Weights & Biases is
    for run tracking, which the notebook doesn't do yet, not weight
    hosting).

- **Updates (`de6d711`).** Filed `docs/next_fixes.md` while reviewing the
  first Stage 1 checkpoint: the adapter's `LoraConfig` never declared
  `modules_to_save=["embed_tokens", "lm_head"]` even though the resize call
  needed for the DSL special tokens made those layers trainable, which
  risked the two layers not reloading correctly; the artifact was also
  ~4.3GB because PEFT was dumping full fp32 copies of both. (Both notes
  turned out to have a different, more serious root cause — see the
  2026-08-27 entry above, where the actual fix landed.)

- **Merge origin/main (`09ca6a9`).** Colab's own "Created using Colab" push
  (`58d6601`) had only reformatted JSON indentation and copied cell ids
  into metadata with identical cell sources; the local branch had
  meanwhile rewritten the notebook (23 → 30 cells) and `colab_setup.md`.
  Took the local tree wholesale rather than merge cosmetic noise into real
  changes, per the user's ask to "reconcile the divergent branches."

- **Colab fixes (`b235928`).** Fixed the chain of errors reported after
  running the notebook cell-by-cell on Colab for the first time: section 3
  `ModuleNotFoundError: No module named 'dsl'` (the repo's `src/` wasn't on
  `sys.path` the way the notebook assumed), section 4's
  `NameError: name 'tokenizer' is not defined` and section 5's matching
  `NameError: name 'model' is not defined` (cells run out of the order the
  notebook implied), plus a rewrite of `docs/colab_setup.md` and the
  notebook itself so each fix (and the reasoning for it) lives inline in
  the notebook cells as the user asked, instead of only in a separate doc
  that "doesn't make sense" against the actual run order.

- **Merge PR #2 / Fix notebook cell ids, add Colab link helper, rework
  colab_setup.md (`cdd6817`, `92ba9b0`).** The Stage 1 notebook declared
  `nbformat_minor` 5 but no cell carried the `id` the format requires, so
  every cell violated the schema — fixed by assigning stable ids to all 23
  cells. Added `scripts/colab_link.py` to pre-flight a notebook (valid
  JSON, non-empty sources, ids present, pushed to `origin/<branch>`) and
  print a direct `colab.research.google.com/github` link, because the
  previous "just visit this link" instructions silently produced an empty
  notebook when a drag-and-drop upload was used instead. Reworked
  `docs/colab_setup.md` around *getting the notebook open* and *what
  persists between runtimes*, which it had previously assumed rather than
  explained.

- **Created using Colab (`58d6601`).** Colab's own auto-save of the
  notebook — cosmetic only (JSON re-indentation, cell ids copied into
  metadata); no source changes. Superseded by `b235928`/`09ca6a9` above.

- **Merge PR #1 / Add colab_setup.md guidance on minimizing connected
  runtime time (`1649582`, `d73815f`).** Added §8 on avoiding wasted Colab
  compute units — front-load GPU-free setup, cache the model to Drive,
  checkpoint to Drive, disconnect promptly via `runtime.unassign()` —
  directly in response to "is there anything I can do to keep the amount
  of time I am connected to a runtime limited so I can maximise my compute
  units?"

- **Improve colab setup (`d25ece3`) / Improve setup instructions
  (`94f539b`).** Rewrote and then tightened `docs/colab_setup.md` per two
  explicit requests: first to specify *which* code goes in *which* cell
  (rather than leaving "add this" ambiguous), and to cover Google Drive
  persistence across runtime restarts; then to cut wording while keeping
  the explanations easy to follow, since the doc had grown long.

- **Add caveman, rtk ai, and codegraph (`8d6cf55`).** Brought in local
  agent-skill tooling (`.agents/skills/...`) rather than anything specific
  to the pythonllm pipeline. No corresponding `docs/prompts.txt` entry
  describes the reasoning — noted here for completeness rather than
  invented.

- **`works` (`2e2fb8d`).** Prompt-log-only commit (3 lines to
  `docs/prompts.txt`), following the "test prompt to see if prompts.txt
  works" verification prompt.

- **Initial commit (`6edbb71`).** Scaffolds the two-stage pipeline design:
  `src/dsl/` (schema + validator for the pseudocode DSL), `src/eval/harness.py`
  (execution-based scoring), notebook stubs, and 50 seed
  `(english, pseudocode)` / `(pseudocode, python)` pairs. `CLAUDE.md` already
  includes the prompt-logging convention and `docs/prompts.txt` already has
  29 lines at this commit — both were set up as the very first ask of the
  project ("write down every single prompt I make... to a folder called
  docs... it should never be gitignored").

### Discussed, not committed

- Dataset sizing for the Stage 1 planner set: whether 50 examples is
  enough, whether hand-written prompts can be "made up," and whether the
  target 7B models can be made agentic the way Sonnet/Opus are. Settled on
  keeping the existing 50 `(english, pseudocode)` pairs as a working v1
  rather than growing the set further at this stage.
- Training-time and weight-location questions while `trainer.train()` was
  running ("is your time off," how long training takes, whether weights
  land in the mounted Drive folder) — answered without code changes, per
  explicit "do not make any edits" instructions; the follow-up ("should I
  save weights somewhere other than only Google Drive") led directly to
  `docs/weight_saving.md` above.
