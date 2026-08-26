# Colab setup — Stage 1 Planner fine-tune

Instructions for running `notebooks/02_stage1_finetune.ipynb` on Google
Colab. The notebook QLoRA fine-tunes `Qwen2.5-7B-Instruct` on the 50
`(english -> pseudocode)` pairs in
`data/stage1_planner/englishtopseudo.jsonl`, after registering the DSL
special tokens (`<PLAN>`, `</PLAN>`, `<STEP>`) from `src/dsl/schema.py` into
the tokenizer.

Read this top to bottom the first time; sections 1-4 are setup, 5-6 are the
run itself, and 7-9 are troubleshooting and what to do afterwards. Start
with the cell conventions immediately below — they tell you which code
blocks you paste into new cells and which are already in the notebook.

## How to use the code blocks in this document

**Every fenced code block below is one Colab cell — one block, one cell.**
Do not paste several blocks into a single big cell. Colab reports errors and
timings per cell, and most of the recovery advice in §8 is "re-run that one
cell", which only works if each step stands alone.

Each block is labelled with what to do with it:

- **New cell** — you create this cell yourself. The section says exactly
  where it goes in the notebook (all of them go *above* the notebook's
  existing `## 1. Install dependencies` cell, in the order given below).
- **Already in the notebook** — shown here only so you can read what it
  does. Do **not** paste it again; you would run the step twice.
- **Scratch cell** — a throwaway diagnostic. The section says which cell it
  must sit below (it needs variables that cell defined). Run it, then delete
  it; it is not part of the pipeline.

The notebook ships with 12 code cells (its sections 1-10). By the time you
are ready to run, the top of your notebook should look like this:

| Order | Cell | Where it comes from |
| --- | --- | --- |
| 1 | `!nvidia-smi` | New cell — §1 |
| 2 | `torch.cuda` check | New cell — §1 |
| 3 | `!git clone ...` | New cell — §2 (skip if you used the upload option) |
| 4 | `!ls` / `!wc -l` verification | New cell — §2 |
| 5 | `login()` | New cell — §3, **optional**, only if you hit rate limits |
| 6 | `!pip install ...` | Already in the notebook — its section 1 |
| 7+ | config, tokenizer, model, LoRA, data, train, save | Already in the notebook — its sections 2-10 |

To add these: hover over the gap just above the notebook's `!pip install`
cell and click **+ Code**, which inserts a cell there. Selecting a cell and
using the toolbar's **+ Code** inserts *below* it instead — so the simplest
order is to create cell 1 in that gap, then keep selecting the cell you just
made and adding below it, working down the list. Drag cells with the handle
in their toolbar if you get the order wrong.

After the setup cells run clean once, you normally never touch them again in
that session — the rest of the run is Runtime -> Run all, or top-to-bottom
Shift+Enter from the install cell.

## 0. Before you start

- A Google account with Colab Pro or Pay As You Go credits (see §1 for why
  the free tier isn't practical here).
- The repo pushed to GitHub, or the two files listed in §2 handy on your
  local machine.
- Roughly 20GB of free runtime disk for the base model download (the 7B
  safetensors are ~15GB, downloaded in fp16/bf16 and quantized to 4-bit at
  load time — the *download* is not 4-bit, only the in-memory weights are).
- Run `python scripts/check_data.py` locally first. It validates that both
  JSONL files parse, that stage1/stage2 line up 1:1, and that every plan
  passes `src/dsl/validator.py`. Fixing data problems locally is much
  cheaper than discovering them after a model download in Colab.

## 1. GPU

Runtime -> Change runtime type -> Hardware accelerator: **GPU**.

- Pick **A100** if you have Colab Pro+ (or Pay As You Go credits) — the
  notebook loads Qwen2.5-7B-Instruct in 4-bit, which needs roughly 6-8GB of
  VRAM for weights plus overhead for activations, gradients and optimizer
  state, so an A100 (40GB) has plenty of headroom and trains fastest.
- **L4** (available on Colab Pro, 24GB) also works fine for a 7B model in
  4-bit QLoRA and is cheaper in compute units — use this if A100 isn't
  available or you want to save credits.
- Avoid the free-tier **T4** (16GB) for a 7B model. It has no bf16 support
  (the notebook sets `bf16=True` and `bnb_4bit_compute_dtype=torch.bfloat16`),
  so you would have to switch those to fp16, and it is slow enough that the
  download plus training stops being practical.

Confirm what you actually got before installing anything — Colab silently
falls back to whatever is free.

**New cell — the first cell of the notebook**, above `## 1. Install
dependencies`:

```python
!nvidia-smi
```

You want to see the expected GPU name and ~40GB (A100) or ~23GB (L4) of
total memory.

**New cell — directly below the one above.** Keep it separate: if the
runtime has no GPU at all, `nvidia-smi` fails loudly on its own and you can
fix the runtime type before the import cell muddies the output.

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())  # must be True for bf16=True
```

Both of these are cheap and safe to re-run at any point in the session —
they are the fastest way to check you did not silently get a different GPU
after a reconnect.

## 2. Get the repo files into the runtime

The notebook expects the repo checked out at `pythonllm/` in the Colab
working directory (`/content/pythonllm/`), because the config cell in
section 2 uses these relative paths.

**Already in the notebook** (its section 2 config cell) — do not paste this
anywhere; it is quoted so you can see what the paths must match:

```python
DATA_PATH   = "pythonllm/data/stage1_planner/englishtopseudo.jsonl"
SCHEMA_PATH = "pythonllm/src/dsl/schema.py"
```

and section 3 does `sys.path.insert(0, "pythonllm/src")` before
`from dsl.schema import SPECIAL_TOKENS`. If you put the files somewhere
else, edit those three lines to match rather than fighting the layout.

**Option A — clone (recommended).** **New cell**, below the two GPU-check
cells and still above `## 1. Install dependencies`. Use *one* of the two
variants below, not both:

```python
!git clone https://github.com/<your-username>/pythonllm.git
```

If the repo is private, use this variant instead — same position, same
single cell. A fine-grained GitHub personal access token with read-only
Contents access is enough:

```python
import getpass
token = getpass.getpass("GitHub token: ")   # avoids pasting it into a saved cell
!git clone https://{token}@github.com/<your-username>/pythonllm.git
```

Keep the `getpass` prompt and the `git clone` in the same cell so `token`
is still in scope when the clone runs, and so a saved copy of the notebook
never contains the token.

Do not hardcode the token in a cell you will save or share — the notebook
output and source are stored in Drive.

**Option B — upload just the needed files.** No cell to add — this is done
in the Colab UI. Only two files are actually read by this notebook:

1. In the Colab file browser (left sidebar, folder icon), create
   `pythonllm/data/stage1_planner/` and `pythonllm/src/dsl/`.
2. Drag `data/stage1_planner/englishtopseudo.jsonl` into the first folder.
3. Drag `src/dsl/schema.py` and `src/dsl/__init__.py` into the second
   (`__init__.py` is what makes `from dsl.schema import ...` work).

Note that uploads are lost on disconnect just like clones, so Option A is
less painful to redo.

Either way, verify before training. **New cell**, immediately below the
clone cell (or, with Option B, directly below the GPU-check cells):

```python
!ls pythonllm/data/stage1_planner/ pythonllm/src/dsl/
!wc -l pythonllm/data/stage1_planner/englishtopseudo.jsonl   # expect 50
```

If this cell errors or prints a count other than 50, stop here and fix the
checkout — every later cell depends on these two paths resolving.

## 3. Hugging Face access

Qwen2.5-7B-Instruct is a public, ungated model, so no token is required to
download it, and in the normal case you add no cell here at all.

**New cell — optional**, only if you hit a rate limit or want the download
attributed to your account. Put it last among the setup cells, directly
above `## 1. Install dependencies`, and run it before the model-loading
cell in the notebook's section 4:

```python
from huggingface_hub import login
login()  # paste a token from https://huggingface.co/settings/tokens
```

`login()` renders an input widget in the cell output — paste the token there
rather than into the cell source, so it is not saved with the notebook.

The download lands in `~/.cache/huggingface` inside the runtime and is
re-fetched from scratch on every new session — budget a few minutes for it
each run.

## 4. Install dependencies

**Already in the notebook** — this is its section 1 cell, the first cell
that ships with the notebook. Run it, don't re-add it:

```python
!pip install -q -U transformers accelerate peft bitsandbytes datasets trl
```

`bitsandbytes` provides the 4-bit (NF4) quantization used by
`BitsAndBytesConfig`; `peft` provides LoRA. If pip reports that it upgraded
a package that was already imported (commonly `transformers` or `torch`),
restart the runtime (Runtime -> Restart session) and re-run from section 1 —
otherwise you get confusing import or version-mismatch errors later.

## 5. What each config knob does

These live in the notebook's own section 2 config cell — a single cell you
edit in place rather than adding new cells for. Change values there and
re-run that one cell, then re-run the cells downstream of it.

| Setting | Value | Notes |
| --- | --- | --- |
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Instruction-tuned base, so the chat template in section 6 is meaningful. |
| `OUTPUT_DIR` | `stage1_planner_qlora` | Runtime-local; the final adapter is copied to Drive in section 9. |
| `MAX_SEQ_LEN` | 512 | Prompt + plan, truncated. Plans in the current dataset are far shorter; raise it if you add long ones. |
| `NUM_EPOCHS` | 15 | High on purpose — 50 examples is a proof-of-concept, not a real run. |
| `LEARNING_RATE` | 2e-4 | Standard LoRA range (1e-4 to 3e-4). |
| `PER_DEVICE_BATCH_SIZE` × `GRAD_ACCUM_STEPS` | 4 × 4 | Effective batch 16, i.e. ~3 optimizer steps per epoch over 50 examples. |

LoRA config (section 5): rank 16, alpha 32, dropout 0.05, applied to all
attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`).

Two details worth knowing, since they are the parts most likely to break if
you edit the notebook:

- **Embedding resize.** Adding 3 special tokens grows the tokenizer, so
  section 4 calls `model.resize_token_embeddings(len(tokenizer))` *before*
  LoRA wrapping. Skipping it produces an index error the first time a
  `<PLAN>` token id is looked up.
- **Loss masking.** `format_example` in section 6 builds the prompt with
  `tokenizer.apply_chat_template(..., add_generation_prompt=True)`, appends
  the pseudocode and EOS, then sets the prompt's label positions to `-100`
  so only the plan contributes to the loss.

## 6. Run the notebook

With the setup cells in place, run everything top to bottom — Runtime -> Run
all, or Shift+Enter down the notebook. Section numbers below refer to the
notebook's own headings, not to this document. Checkpoints to eyeball as you
go:

- Section 3 prints `Added 3 new special tokens` — if it says 0, the tokens
  were already in the vocab or `SPECIAL_TOKENS` imported empty.
- Section 5 prints trainable parameters — expect roughly 40M trainable out
  of ~7.6B total (well under 1%).
- Section 6 prints the raw dataset and its first row; confirm the `english`
  and `pseudocode` keys are there.
- Section 7 logs loss every step (`logging_steps=1`). It should fall over
  the run; see §8 if it doesn't move.
- Section 9 mounts Google Drive and copies the adapter to
  `My Drive/pythonllm_checkpoints/stage1_planner/` — approve the Drive
  mount prompt when it appears.
- Section 10 generates a plan for one held-in example. Because the sanity
  check decodes with `skip_special_tokens=False`, you should literally see
  `<PLAN>`, `<STEP>` and `</PLAN>` in the output.

**Do not skip section 9.** Colab runtimes, including their local disk, are
wiped when the session ends or disconnects, so anything under `OUTPUT_DIR`
is gone with it. Keep the browser tab open and interact with it
occasionally — idle sessions get reclaimed.

## 7. Expected runtime

Rough figures for the current 50-example dataset:

- Dependency install: 1-2 minutes.
- Base model download: 3-10 minutes depending on network and cache.
- Training (15 epochs, ~45 optimizer steps): a few minutes on A100/L4.
- Sanity-check generation: seconds.

Total is well under an hour, most of it download. This is a
proof-of-concept fine-tune, not a full training run.

## 8. Troubleshooting

**CUDA out of memory.** Lower `PER_DEVICE_BATCH_SIZE` to 2 or 1 and raise
`GRAD_ACCUM_STEPS` to keep the effective batch at 16; lower `MAX_SEQ_LEN`
if your examples are short; add `gradient_checkpointing=True` to
`TrainingArguments`. Restart the runtime to actually free VRAM before
retrying — re-running a cell after an OOM often leaves memory pinned.

**Loss stays flat / prints `nan`.** Check that `labels` aren't entirely
`-100`, which means the whole example got masked (usually caused by a chat
template change making the prompt longer than `MAX_SEQ_LEN`).

**Scratch cell** — add it just below the notebook's section 6 formatting
cell (it needs `tokenized_dataset` and `tokenizer` to already exist), run
it, then delete it:

```python
row = tokenized_dataset[0]
print(sum(1 for x in row["labels"] if x != -100), "of", len(row["labels"]), "supervised")
print(tokenizer.decode([t for t, l in zip(row["input_ids"], row["labels"]) if l != -100]))
```

The decoded text should be the pseudocode plan and nothing else. `nan` loss
usually means bf16 wasn't actually supported — see §1.

**`bitsandbytes` import or CUDA errors.** Almost always a stale runtime
after the pip upgrade; restart the session and re-run from section 1.

**Model emits `<`, `PLAN`, `>` as separate pieces.** The tokenizer used at
generation time doesn't have the special tokens. Make sure you load the
tokenizer saved alongside the adapter (section 8 saves both to
`FINAL_DIR`), not a fresh one from the hub.

**Disconnected mid-training.** `save_strategy="epoch"` writes checkpoints
under `OUTPUT_DIR`, but those live on the ephemeral runtime disk, so a
disconnect loses them too. Just re-run — the run is short. If you expect to
be interrupted, point `OUTPUT_DIR` at a mounted Drive path instead.

## 9. Using the adapter later

The saved directory is a LoRA adapter plus tokenizer, not a full model, so
reload the base model the same way and attach it.

This is **not** a cell for `02_stage1_finetune.ipynb` — it belongs at the
top of whatever notebook consumes the adapter later (`03_stage2_finetune`,
`04_eval_pipeline`, or a fresh session of this one after a disconnect). One
cell, and it assumes `BASE_MODEL` and `bnb_config` are already defined in
that notebook; if they aren't, copy those two definitions from the notebook's
sections 2 and 4 into the same cell first.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER = "/content/drive/MyDrive/pythonllm_checkpoints/stage1_planner"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)          # has the DSL tokens
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
base.resize_token_embeddings(len(tokenizer))                # must match the saved adapter
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
```

The resize call is required — the adapter was trained against the enlarged
embedding matrix and will not load onto the stock vocab size.

## 10. Next steps

Once you have a saved adapter in Drive, download it or point
`03_stage2_finetune.ipynb` / `04_eval_pipeline.ipynb` at the same Drive path
to keep working across sessions. `04_eval_pipeline.ipynb` scores plans with
`src/dsl/validator.py` (well-formedness) and `src/eval/harness.py`
(execution-based, DS-1000-style).
