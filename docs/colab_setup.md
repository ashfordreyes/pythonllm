# Colab setup — Stage 1 Planner fine-tune

How to run `notebooks/02_stage1_finetune.ipynb` on Google Colab. The
notebook QLoRA fine-tunes `Qwen2.5-7B-Instruct` on the 50
`(english -> pseudocode)` pairs in
`data/stage1_planner/englishtopseudo.jsonl`, after registering the DSL
special tokens (`<PLAN>`, `</PLAN>`, `<STEP>`) from `src/dsl/schema.py`.

Read top to bottom the first time: §1-6 are setup, §7 the run itself,
§8-10 troubleshooting and what comes after.

## How to use the code blocks here

**Every fenced block is one Colab cell — one block, one cell.** Don't merge
them: Colab reports errors per cell, and most recovery advice in §8 is
"re-run that one cell".

Each block is labelled:

- **New cell** — you add it, above the notebook's existing
  `## 1. Install dependencies` cell, in the order given here.
- **Already in the notebook** — quoted for reference only. Don't paste it
  again.
- **Scratch cell** — a throwaway diagnostic. Run it, then delete it.

The notebook ships with 12 code cells (its sections 1-10). Once your setup
cells are in, the top should read:

| Order | Cell | From |
| --- | --- | --- |
| 1 | `!nvidia-smi` | New — §2 |
| 2 | `torch.cuda` check | New — §2 |
| 3 | `drive.mount(...)` | New — §3 |
| 4 | `!git clone ...` | New — §4 (skip if you upload instead) |
| 5 | `!ls` / `!wc -l` check | New — §4 |
| 6 | `login()` | New — §5, **optional** |
| 7 | `!pip install ...` | Notebook, section 1 |
| 8+ | config, tokenizer, model, LoRA, data, train, save | Notebook, sections 2-10 |

To add them: hover over the gap just above the `!pip install` cell and click
**+ Code**. That inserts *there*; selecting a cell and using the toolbar's
**+ Code** inserts *below* it — so make cell 1 in the gap, then keep adding
below the cell you just made. Drag by the cell toolbar handle to reorder.

Once these run clean, you don't touch them again that session — the rest is
Runtime -> Run all.

## 1. Before you start

- A Google account with Colab Pro or Pay As You Go credits (see §2).
- The repo on GitHub, or the two files from §4 on your local machine.
- ~20GB free runtime disk for the base model (the 7B safetensors are ~15GB;
  the *download* is fp16/bf16, only the in-memory weights are 4-bit).
- Run `python scripts/check_data.py` locally first. It checks both JSONL
  files parse, that stage1/stage2 line up 1:1, and that every plan passes
  `src/dsl/validator.py`. Cheaper to fix here than after a model download.

## 2. GPU

Runtime -> Change runtime type -> Hardware accelerator: **GPU**.

- **A100** (40GB, Pro+/PAYG) — fastest. The 4-bit model needs ~6-8GB for
  weights plus activations, gradients and optimizer state, so this has
  plenty of headroom.
- **L4** (24GB, Pro) — also fine for 7B in 4-bit QLoRA, and cheaper in
  compute units.
- Avoid the free **T4** (16GB): no bf16 support, so you'd have to change
  `bf16=True` and `bnb_4bit_compute_dtype` to fp16, and it's slow enough
  that download plus training stops being practical.

Confirm what you got — Colab silently falls back to whatever's free.

**New cell — first cell of the notebook:**

```python
!nvidia-smi
```

Expect the GPU name and ~40GB (A100) or ~23GB (L4) total memory.

**New cell — directly below.** Keep it separate: with no GPU at all,
`nvidia-smi` fails loudly on its own, before the import muddies the output.

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())  # must be True for bf16=True
```

Both are cheap to re-run any time — the fastest check that you didn't
silently get a different GPU after a reconnect.

## 3. Google Drive — what survives a runtime shutdown

**Nothing in the runtime persists.** When the session ends, disconnects, or
is reclaimed for idling, the whole disk goes: your clone, uploaded files,
the model cache, and every checkpoint under `OUTPUT_DIR`. Google Drive is
the only durable storage Colab gives you, and mounting it is one cell.

**New cell — below the two GPU checks:**

```python
from google.colab import drive
drive.mount("/content/drive")
```

The first run opens a Google auth popup; approve it. Drive then appears at
`/content/drive/MyDrive/` and behaves like a normal folder — `!ls`, `!cp`,
and Python file I/O all work. Re-running the cell when already mounted is
harmless. You reauthorize once per session, not once per notebook.

Mounting early (rather than at the notebook's section 9) lets you point
things at Drive *before* they write anything:

- **Trained adapter.** The notebook's section 9 copies it to
  `MyDrive/pythonllm_checkpoints/stage1_planner/`. This is the one thing
  you must not lose.
- **Mid-training checkpoints.** `save_strategy="epoch"` writes under
  `OUTPUT_DIR`, which is runtime-local by default. If you expect
  interruptions, set it in the notebook's section 2 config cell to
  `"/content/drive/MyDrive/pythonllm_checkpoints/stage1_planner_qlora"` so
  epoch checkpoints survive a disconnect and you can resume with
  `trainer.train(resume_from_checkpoint=True)`.
- **Model download cache** (optional). The ~15GB base model re-downloads
  every session. To cache it on Drive, add
  `import os; os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"` to
  the *same* cell as the mount, before any transformers import. Worth it
  only if your Drive has the space and your Drive reads are faster than
  your Colab downloads — often they aren't, and 15GB is a big bite out of a
  15GB free tier. Skip this unless downloads are your bottleneck.

Code is the exception: it lives in git, so re-cloning (§4) is the simplest
way to restore it. Don't work out of a Drive copy of the repo.

## 4. Get the repo files into the runtime

The notebook expects the repo at `/content/pythonllm/`, because its section
2 config uses these relative paths.

**Already in the notebook** (section 2), quoted so you can match them:

```python
DATA_PATH   = "pythonllm/data/stage1_planner/englishtopseudo.jsonl"
SCHEMA_PATH = "pythonllm/src/dsl/schema.py"
```

Section 3 also does `sys.path.insert(0, "pythonllm/src")` before
`from dsl.schema import SPECIAL_TOKENS`. Put files elsewhere and you must
edit all three lines.

**Option A — clone (recommended). New cell**, below the Drive mount. Use
*one* of these two variants:

```python
!git clone https://github.com/<your-username>/pythonllm.git
```

For a private repo, use this instead — same position, same single cell. A
fine-grained token with read-only Contents access is enough:

```python
import getpass
token = getpass.getpass("GitHub token: ")   # keeps it out of saved cell source
!git clone https://{token}@github.com/<your-username>/pythonllm.git
```

Keep `getpass` and the clone in one cell so `token` is still in scope. Never
hardcode a token in a cell — notebook source and output are saved to Drive.

**Option B — upload the two files.** No cell; done in the Colab UI. In the
file browser (left sidebar, folder icon):

1. Create `pythonllm/data/stage1_planner/` and `pythonllm/src/dsl/`.
2. Drag `englishtopseudo.jsonl` into the first.
3. Drag `schema.py` and `__init__.py` into the second (`__init__.py` is what
   makes the import work).

Uploads are lost on disconnect just like clones, so Option A is less
painful to redo.

Either way, verify. **New cell**, immediately below:

```python
!ls pythonllm/data/stage1_planner/ pythonllm/src/dsl/
!wc -l pythonllm/data/stage1_planner/englishtopseudo.jsonl   # expect 50
```

If this errors or prints anything but 50, stop and fix the checkout.

## 5. Hugging Face access

Qwen2.5-7B-Instruct is public and ungated, so normally you add no cell here.

**New cell — optional**, only for rate limits or download attribution. Put
it last among the setup cells:

```python
from huggingface_hub import login
login()  # paste a token from https://huggingface.co/settings/tokens
```

Paste into the widget in the cell *output*, not the source. The download
lands in `~/.cache/huggingface` and is re-fetched each session unless you
set `HF_HOME` (§3) — budget a few minutes.

## 6. Config and dependencies

**Already in the notebook** — its section 1 cell. Run it, don't re-add it:

```python
!pip install -q -U transformers accelerate peft bitsandbytes datasets trl
```

`bitsandbytes` provides the 4-bit (NF4) quantization; `peft` provides LoRA.
If pip reports it upgraded an already-imported package (usually
`transformers` or `torch`), restart the runtime and re-run from section 1 —
otherwise you get version-mismatch errors later.

The knobs live in the notebook's section 2 config cell. Edit in place and
re-run that cell plus everything downstream.

| Setting | Value | Notes |
| --- | --- | --- |
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Instruction-tuned, so the chat template in section 6 is meaningful. |
| `OUTPUT_DIR` | `stage1_planner_qlora` | Runtime-local; see §3 to put it on Drive. |
| `MAX_SEQ_LEN` | 512 | Prompt + plan, truncated. Raise if you add long plans. |
| `NUM_EPOCHS` | 15 | High on purpose — 50 examples is a proof of concept. |
| `LEARNING_RATE` | 2e-4 | Standard LoRA range (1e-4 to 3e-4). |
| `PER_DEVICE_BATCH_SIZE` × `GRAD_ACCUM_STEPS` | 4 × 4 | Effective batch 16, ~3 steps per epoch. |

LoRA (section 5): rank 16, alpha 32, dropout 0.05, on all attention and MLP
projections (`q/k/v/o_proj`, `gate/up/down_proj`).

Two details most likely to break if you edit the notebook:

- **Embedding resize.** The 3 new tokens grow the tokenizer, so section 4
  calls `model.resize_token_embeddings(len(tokenizer))` *before* LoRA
  wrapping. Skip it and the first `<PLAN>` token id lookup errors.
- **Loss masking.** `format_example` (section 6) builds the prompt with
  `apply_chat_template(..., add_generation_prompt=True)`, appends the
  pseudocode and EOS, then sets prompt labels to `-100` so only the plan
  contributes to the loss.

## 7. Run it

Runtime -> Run all, or Shift+Enter down. Section numbers below are the
notebook's. Checkpoints to eyeball:

- Section 3 prints `Added 3 new special tokens` — 0 means they were already
  in the vocab or `SPECIAL_TOKENS` imported empty.
- Section 5 prints ~40M trainable of ~7.6B (well under 1%).
- Section 6 prints the dataset and first row; confirm `english` and
  `pseudocode` keys.
- Section 7 logs loss every step. It should fall; see §8 if not.
- Section 9 mounts Drive (already done if you did §3) and copies the
  adapter to `MyDrive/pythonllm_checkpoints/stage1_planner/`.
- Section 10 generates a plan for one held-in example. It decodes with
  `skip_special_tokens=False`, so you should literally see `<PLAN>`,
  `<STEP>`, `</PLAN>`.

**Do not skip section 9** — see §3. Keep the tab open and interact
occasionally; idle sessions get reclaimed.

Rough timings for the 50-example dataset: install 1-2 min, model download
3-10 min, training (15 epochs, ~45 steps) a few minutes on A100/L4, sanity
check seconds. Well under an hour, mostly download.

## 8. Troubleshooting

**CUDA out of memory.** Lower `PER_DEVICE_BATCH_SIZE` to 2 or 1 and raise
`GRAD_ACCUM_STEPS` to keep the effective batch at 16; lower `MAX_SEQ_LEN`;
add `gradient_checkpointing=True` to `TrainingArguments`. Restart the
runtime to actually free VRAM — re-running after an OOM often leaves memory
pinned.

**Loss flat or `nan`.** Check `labels` aren't entirely `-100`, meaning the
example got fully masked (usually a chat template change pushing the prompt
past `MAX_SEQ_LEN`). `nan` usually means bf16 isn't really supported — §2.

**Scratch cell** — below the section 6 formatting cell (it needs
`tokenized_dataset` and `tokenizer`). Run, then delete:

```python
row = tokenized_dataset[0]
print(sum(1 for x in row["labels"] if x != -100), "of", len(row["labels"]), "supervised")
print(tokenizer.decode([t for t, l in zip(row["input_ids"], row["labels"]) if l != -100]))
```

The decoded text should be the plan and nothing else.

**`bitsandbytes` import or CUDA errors.** Almost always a stale runtime
after the pip upgrade; restart and re-run from section 1.

**Model emits `<`, `PLAN`, `>` separately.** The generation-time tokenizer
lacks the special tokens. Load the one saved beside the adapter (section 8
saves both to `FINAL_DIR`), not a fresh one from the hub.

**Disconnected mid-training.** Just re-run — it's short. Checkpoints under a
runtime-local `OUTPUT_DIR` are gone with the runtime; point it at Drive
(§3) if you expect interruptions.

## 9. Reusing the adapter later

The saved directory is a LoRA adapter plus tokenizer, not a full model, so
reload the base and attach it.

This is **not** a cell for `02_stage1_finetune.ipynb` — it goes at the top
of whatever consumes the adapter (`03_stage2_finetune`, `04_eval_pipeline`,
or a fresh session of this one). It assumes `BASE_MODEL` and `bnb_config`
exist there; if not, copy those definitions from the notebook's sections 2
and 4 into the same cell first. Mount Drive (§3) before running it.

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

The resize is required — the adapter was trained against the enlarged
embedding matrix and won't load onto the stock vocab size.

## 10. Next steps

With an adapter in Drive, download it or point `03_stage2_finetune.ipynb` /
`04_eval_pipeline.ipynb` at the same Drive path to keep working across
sessions. `04_eval_pipeline.ipynb` scores plans with `src/dsl/validator.py`
(well-formedness) and `src/eval/harness.py` (execution-based, DS-1000-style).
