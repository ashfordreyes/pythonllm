# Colab setup — Stage 1 Planner fine-tune

How to run `notebooks/02_stage1_finetune.ipynb` on Google Colab. The
notebook QLoRA fine-tunes `Qwen2.5-7B-Instruct` on the 50
`(english -> pseudocode)` pairs in
`data/stage1_planner/englishtopseudo.jsonl`, after registering the DSL
special tokens (`<PLAN>`, `</PLAN>`, `<STEP>`) from `src/dsl/schema.py`.

**The notebook is self-contained.** Its section 2 checks the GPU, installs
the libraries, mounts Drive and clones the repo, so opening the link and
running top to bottom is the whole procedure — there are no cells to add by
hand, and each section's markdown says what its output should look like.

This file is the part that lives *outside* the notebook: what to do before
you connect to a runtime (§1-2, free), which GPU to pick (§3), what survives
a session ending (§4), how not to waste compute units (§6), and what to do
when something breaks (§7).

**Scope: Stage 1 only.** This file covers training the *Planner*. There is no
Stage 2 notebook yet — `notebooks/03_stage2_finetune.ipynb` appears in
`notebooks/README.md` and in §8/§9 below as the thing that will consume the
Stage 1 adapter, but it has not been written, so nothing here tells you how to
train the Coder. §9 says what does exist for Stage 2 today.

## Do this before you connect to a runtime

Compute units are billed on GPU-connected wall time, so everything that can
happen off a runtime should. None of these cost anything:

- [ ] Run `python scripts/check_data.py` locally (§1).
- [ ] Run `python scripts/colab_link.py` and open the link it prints (§2).
- [ ] `File -> Save a copy in Drive` so your copy is durable (§2).
- [ ] Edit the notebook's section 1 config cell — branch, `OUTPUT_DIR`,
      epochs, batch size. Colab lets you edit cell source with no runtime
      attached; only *running* needs one.
- [ ] Decide A100 vs L4 (§3) — but don't change the runtime type yet.
      Choosing the type is what connects you.

Only then: Runtime -> Change runtime type -> GPU, then Run all.

## 1. Before you start

- A Google account with Colab Pro or Pay As You Go credits (see §3).
- ~20GB free runtime disk for the base model (the 7B safetensors are ~15GB;
  the *download* is fp16/bf16, only the in-memory weights are 4-bit).
- Run `python scripts/check_data.py` locally first. It checks both JSONL
  files parse, that stage1/stage2 line up 1:1, and that every plan passes
  `src/dsl/validator.py`. Cheaper to fix here than after a model download.
  Expect `OK: 50 pairs in englishtopseudo.jsonl, ...`.

## 2. Open the notebook in Colab

No runtime needed for any of this. `ashfordreyes/pythonllm` is a **public**
repo, so Colab can fetch the notebook itself — nothing to upload, no token.

**Option A — open from GitHub (recommended).** Run this locally:

```
python scripts/colab_link.py
```

It prints a `colab.research.google.com/github/...` link for the notebook on
your current branch, after checking the notebook is well-formed and that the
commit Colab will read is actually pushed. Pass a path to link a different
notebook (`python scripts/colab_link.py notebooks/04_eval_pipeline.ipynb`).

For `main` the link is always:

```
https://colab.research.google.com/github/ashfordreyes/pythonllm/blob/main/notebooks/02_stage1_finetune.ipynb
```

The equivalent UI route is File -> Open notebook -> **GitHub** tab -> paste
`ashfordreyes/pythonllm`. Use that route for a branch whose name contains a
slash (like `claude/...`): a `/blob/` URL can't tell where such a branch name
ends and the path begins, but the dropdown can. `colab_link.py` warns you
when that applies.

**This view is a render of the pushed file, and your edits are not saved
anywhere.** Do `File -> Save a copy in Drive` immediately. That copy lands in
`MyDrive/Colab Notebooks/` and autosaves from then on — see §4.

**Option B — File -> Upload notebook.** Use this when you have local notebook
edits you haven't pushed. Colab stores the uploaded copy in
`MyDrive/Colab Notebooks/` straight away, so it's durable without the extra
save step. The tradeoff is that it's now a fork: changes you make there don't
come back to the repo unless you download it and commit it.

**Option C — dragging the `.ipynb` into the left sidebar. Don't.** See below.

### If the cells come up empty

The left sidebar (folder icon) is a **file manager for the runtime's disk**,
not a way to open a notebook. Dropping a `.ipynb` there copies bytes into
`/content` on a connected machine; it does not load that file into the
editor you're looking at. So you get exactly the symptom of an upload that
"worked" — the file is visibly there in the sidebar — while the notebook on
screen is a different, empty one. It also requires a runtime to accept the
drop, and the file is deleted when that runtime goes away.

Use Option A or B instead. Both open the notebook *as* a notebook.

## 3. Choosing a GPU

Runtime -> Change runtime type -> Hardware accelerator: **GPU**. This is the
step that starts billing, so make sure the checklist above is done first.

- **A100** (40GB, Pro+/PAYG) — fastest. The 4-bit model needs ~6-8GB for
  weights plus activations, gradients and optimizer state, so this has
  plenty of headroom.
- **L4** (24GB, Pro) — also fine for 7B in 4-bit QLoRA, and cheaper in
  compute units.
- Avoid the free **T4** (16GB): no bf16 support, so you'd have to change
  `bf16=True` and `bnb_4bit_compute_dtype` to fp16, and it's slow enough
  that download plus training stops being practical.

Colab silently falls back to whatever's free, so the notebook's section 2a
prints the GPU name and `torch.cuda.is_bf16_supported()`. Check it — that
cell is also the fastest way to confirm you didn't get a different GPU after
a reconnect.

## 4. What persists between sessions

Colab stores things in three places, and only two of them outlive the
session:

| What | Lives on | Survives disconnect |
| --- | --- | --- |
| Notebook opened via the GitHub link (§2A) | GitHub, rendered read-only | Yes — but **your edits are not saved** until you Save a copy in Drive |
| Notebook uploaded or saved to Drive (§2B) | `MyDrive/Colab Notebooks/` | Yes, autosaved as you type |
| Anything under `/content` — the clone, sidebar uploads, the HF cache, `OUTPUT_DIR` checkpoints | runtime disk | **No.** All of it is destroyed |
| Anything under `/content/drive/MyDrive` | Google Drive | Yes |

So: **nothing in the runtime persists.** When the session ends, disconnects,
or is reclaimed for idling, the whole disk goes. Google Drive is the only
durable storage Colab gives you. The notebook mounts it in section 2c, which
is early enough to point things at Drive *before* they write anything:

- **Trained adapter.** Section 9 copies it to
  `MyDrive/pythonllm_checkpoints/stage1_planner/` (`DRIVE_CHECKPOINT_DIR` in
  the config cell, created for you at mount time). This is the one thing you
  must not lose.
- **Mid-training checkpoints.** `save_strategy="epoch"` writes under
  `OUTPUT_DIR`, which is runtime-local by default. If you expect
  interruptions, set it in the config cell to a path inside
  `DRIVE_CHECKPOINT_DIR` so epoch checkpoints survive a disconnect and you
  can resume with `trainer.train(resume_from_checkpoint=True)`.
- **Model download cache.** The ~15GB base model re-downloads every session.
  Set `CACHE_MODEL_ON_DRIVE = True` in the config cell to keep it on Drive
  instead (the notebook sets `HF_HOME` before importing transformers, which
  is the only point at which that takes effect). Worth it only if your Drive
  has the space and your Drive reads beat your Colab downloads — often they
  don't, and 15GB is a big bite out of a 15GB free tier.

Code is the exception: it lives in git, so re-running the clone cell is the
simplest way to restore it. Don't work out of a Drive copy of the repo.

## 5. Run it

Runtime -> Run all, or Shift+Enter down. No manual edits needed beyond the
config cell. Checkpoints to eyeball, by notebook section:

- **2a** prints the GPU name and `bf16 supported: True`.
- **2d** prints `OK: 50 examples`. If it raises instead, the clone failed or
  `REPO_BRANCH` points somewhere without the data — fix it here.
- **3** prints `Added 3 new special tokens`. `0` means they were already in
  the vocab or `SPECIAL_TOKENS` imported empty.
- **5** prints ~40M trainable of ~7.6B (well under 1%).
- **6** prints the dataset, the first row, and how many tokens are
  supervised after loss masking. The decoded text should be the plan and
  nothing else.
- **7** logs loss every step. It should fall; see §7 if not.
- **9** copies the adapter to Drive. **Do not skip it** — see §4.
- **10** generates a plan for one held-in example, decoded with
  `skip_special_tokens=False`, so you should literally see `<PLAN>`,
  `<STEP>`, `</PLAN>`.
- **11** calls `runtime.unassign()` and ends the session. Comment it out if
  you want to keep poking at the model.

Keep the tab open and interact occasionally during the run; idle sessions
get reclaimed.

Rough timings for the 50-example dataset: install 1-2 min, model download
3-10 min, training (15 epochs, ~45 steps) a few minutes on A100/L4, sanity
check seconds. Well under an hour, mostly download.

## 6. Minimizing connected time to save compute units

Compute units are spent on GPU-connected wall time, not just active
training — an idle-but-connected runtime still burns them, and closing the
browser tab doesn't disconnect it (Colab keeps the runtime alive in the
background for a while, still billing). Given the timings above, the
highest-leverage moves are:

- **Do everything that doesn't need a GPU before connecting** — the
  checklist at the top of this file. Fixing a bad JSONL line, or discovering
  the notebook never opened properly, after you've already spun up an A100
  wastes the whole download+install time on a run that was going to fail
  anyway.
- **Set `CACHE_MODEL_ON_DRIVE = True`** if you'll re-run this notebook
  across multiple sessions — the 3-10 min download is the biggest single
  chunk of connected time, and skipping it on every session after the first
  adds up fast.
- **Point `OUTPUT_DIR` at Drive** (§4) so you don't have to keep the runtime
  connected "just in case" — you can disconnect as soon as a checkpoint
  lands and resume later with `trainer.train(resume_from_checkpoint=True)`
  instead of babysitting a connected session.
- **Let section 11 run.** `runtime.unassign()` actually frees the GPU;
  closing the tab doesn't. If you skip it, use Runtime -> Disconnect and
  delete runtime by hand.
- **Avoid avoidable restarts.** Each `bitsandbytes`/CUDA error in §7 costs a
  full runtime restart and (without the Drive cache) a re-download. Section
  2a's GPU check and noticing whether pip upgraded an already-imported
  package catch most of these before they cost you a restart mid-run.

## 7. Troubleshooting

**`ModuleNotFoundError: No module named 'dsl'` in section 3.** The repo
isn't on the runtime's disk. Re-run section 2d and confirm it prints
`OK: 50 examples`; everything downstream depends on it, and skipping past it
produces a cascade of `NameError`s in sections 4 and 5 as each cell misses
the variable the failed one should have defined.

**`NameError: name 'tokenizer'` / `name 'model'` is not defined.** An
earlier cell errored and you kept going. Fix the *first* failure and re-run
from there rather than continuing down the notebook.

**Cells look empty after uploading the notebook.** See §2 — you almost
certainly dropped the file into the sidebar file browser instead of opening
it. Open it via `python scripts/colab_link.py` or File -> Upload notebook.

**Edits to the notebook vanished between sessions.** You were working in the
read-only GitHub view (§2A) and never did File -> Save a copy in Drive. §4
has the full persistence table.

**CUDA out of memory.** Lower `PER_DEVICE_BATCH_SIZE` to 2 or 1 and raise
`GRAD_ACCUM_STEPS` to keep the effective batch at 16; lower `MAX_SEQ_LEN`;
add `gradient_checkpointing=True` to `TrainingArguments`. Restart the
runtime to actually free VRAM — re-running after an OOM often leaves memory
pinned.

**Loss flat or `nan`.** Section 6 prints how many tokens survive loss
masking; `0 of N supervised` means the example got fully masked, usually a
chat template change pushing the prompt past `MAX_SEQ_LEN`. `nan` usually
means bf16 isn't really supported — check section 2a's output against §3.

**`bitsandbytes` import or CUDA errors.** Almost always a stale runtime
after the pip upgrade in section 2b; restart and re-run from section 1.

**Model emits `<`, `PLAN`, `>` separately.** The generation-time tokenizer
lacks the special tokens. Load the one saved beside the adapter (section 8
saves both to `FINAL_DIR`), not a fresh one from the hub.

**Disconnected mid-training.** Just re-run — it's short. Checkpoints under a
runtime-local `OUTPUT_DIR` are gone with the runtime; point it at Drive (§4)
if you expect interruptions.

**`AttributeError` from deep inside `transformers/tokenization_utils_base.py`
in section 10, tracing back to a `KeyError: 'shape'`.** Version-dependent
`apply_chat_template` behavior: without `return_dict` set explicitly, some
`transformers` versions return a `BatchEncoding` instead of a raw tensor,
and a `BatchEncoding` has no `.shape`. The notebook pins `return_dict=True`
for this reason — if you hit this, you likely edited section 10 and dropped
that argument.

## 8. Reusing the adapter later

The saved directory is a LoRA adapter plus tokenizer, not a full model, so
reload the base and attach it.

This is **not** a cell for `02_stage1_finetune.ipynb` — it goes at the top
of whatever consumes the adapter (`03_stage2_finetune`, `04_eval_pipeline`,
or a fresh session of this one). It assumes `BASE_MODEL` and `bnb_config`
exist there; if not, copy those definitions from the notebook's sections 1
and 4 into the same cell first. Mount Drive before running it.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER = "/content/drive/MyDrive/pythonllm_checkpoints/stage1_planner"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)          # has the DSL tokens
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
```

No `resize_token_embeddings` call is needed: section 4 of the notebook
leaves the base vocab size alone (the DSL token ids already fit
Qwen2.5-7B's 152064-row matrix), so the adapter loads straight onto a stock
base model. Loading the tokenizer from `ADAPTER` rather than from
`BASE_MODEL` still matters, though — that is where the DSL special tokens
live.

Adapters produced *before* 2026-08-27 were trained with the old shrinking
resize and do need `base.resize_token_embeddings(len(tokenizer))` before
this line. They also have untrained DSL token embeddings (see `context.md`),
so retraining is the better option.

## 9. Next steps

With an adapter in Drive, download it or point `04_eval_pipeline.ipynb` at the
same Drive path to keep working across sessions.

`04_eval_pipeline.ipynb` is the natural next run. Open it with
`python scripts/colab_link.py notebooks/04_eval_pipeline.ipynb`; its
`ADAPTER_DIR` already defaults to the path section 9 of notebook 02 writes,
so on a fresh runtime it is Run all. It generates plans for the 10 held-out
tasks, scores them, and writes `stage1_results.json`, `score_breakdown.png`
and `plan_error_types.png` into
`/content/drive/MyDrive/pythonllm_checkpoints/eval_stage1`.

Its `SPLIT_SEED` and `EVAL_FRACTION` must match the values notebook 02
trained with — the held-out rows are derived from the seed, not stored in a
file, so a mismatch silently evaluates on training data. Both notebooks
default to `SPLIT_SEED = 0` / `EVAL_FRACTION = 0.2`.

Scoring is layered rather than execution-only: plan well-formedness via
`src/dsl/validator.py`, verb-sequence and exact match, then code
parse/self-containment/execution via `src/eval/harness.py`. Only 4 of the 50
reference snippets run in a bare runtime, so an execution-only score would
rest on almost nothing; each tier reports its own denominator instead.

### Stage 2 (the Coder)

`03_stage2_finetune.ipynb` does not exist. Training the Coder means writing
it first — the shape is notebook 02 with `Qwen2.5-Coder-7B` as the base,
`data/stage2_coder/pseudotopython.jsonl` as the data, `pseudocode` as the
prompt field and `python_code` as the completion, and **no DSL special-token
registration** (the Coder reads the plan as ordinary text; only the Planner
has to emit `<PLAN>`/`<STEP>` as single tokens). The split helper already
hands back the same held-out rows for both files, so
`load_split(STAGE2_DATA, ...)` under the same `SPLIT_SEED` needs no extra
work. Section 7 of `04_eval_pipeline.ipynb` documents the eval side that
turns on once a coder adapter exists.

### A nicer way to use the models

`docs/colab_console.md` covers `src/ui/console.py`: an ASCII-panel chat UI
that runs in a Colab cell, with the plan and the code syntax-highlighted and
streamed as they generate. It takes a planner callable and a coder callable,
either of which may be absent — so it is usable with the Stage 1 adapter
alone, today, and picks up the Coder when there is one.
