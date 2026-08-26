# Colab setup — Stage 1 Planner fine-tune

Instructions for running `notebooks/02_stage1_finetune.ipynb` on Google
Colab.

## 1. GPU

Runtime -> Change runtime type -> Hardware accelerator: **GPU**.

- Pick **A100** if you have Colab Pro+ (or Pay As You Go credits) — the
  notebook loads Qwen2.5-7B-Instruct in 4-bit, which needs roughly 6-8GB of
  VRAM plus overhead for training, so an A100 (40GB) has plenty of headroom
  and trains fastest.
- **L4** (available on Colab Pro) also works fine for a 7B model in 4-bit
  QLoRA and is cheaper in compute units — use this if A100 isn't available
  or you want to save credits. Avoid the free-tier T4 for a 7B model; it's
  usually too little VRAM/too slow for the base model download and forward
  pass to be practical.

## 2. Get the repo files into the runtime

The notebook expects the repo checked out at `pythonllm/` in the Colab
working directory (i.e. `pythonllm/data/...`, `pythonllm/src/...`). Easiest
way — clone it directly in a notebook cell before running the rest:

```python
!git clone https://github.com/<your-username>/pythonllm.git
```

If the repo is private, use a fine-grained GitHub personal access token:

```python
!git clone https://<TOKEN>@github.com/<your-username>/pythonllm.git
```

Alternatively, upload just what's needed without cloning:

1. In the Colab file browser (left sidebar, folder icon), create a
   `pythonllm/data/stage1_planner/` folder and a `pythonllm/src/dsl/`
   folder.
2. Drag `data/stage1_planner/englishtopseudo.jsonl` from your machine into
   the first folder.
3. Drag `src/dsl/schema.py` and `src/dsl/__init__.py` into the second.

Either way, run a quick check before training:

```python
!ls pythonllm/data/stage1_planner/
!ls pythonllm/src/dsl/
```

## 3. Hugging Face access

Qwen2.5-7B-Instruct is a public, ungated model on Hugging Face, so no token
is required to download it. If you hit a rate limit or want faster/cached
downloads, log in first:

```python
from huggingface_hub import login
login()  # paste a token from https://huggingface.co/settings/tokens
```

## 4. Run the notebook

Run cells top to bottom. Section 9 mounts Google Drive and copies the
trained LoRA adapter to
`My Drive/pythonllm_checkpoints/stage1_planner/` — approve the Drive
mount prompt when it appears. Do this before the runtime disconnects;
Colab runtimes (including their local disk) are wiped when the session
ends.

## 5. Expected runtime

With only 50 training examples and QLoRA, 15 epochs should take a few
minutes on an A100/L4 — this is a small proof-of-concept fine-tune, not a
full training run. Watch the logged loss decrease each step; if it doesn't
move, check that `labels` in the tokenized dataset aren't all `-100`
(that would mean the whole example is masked).

## 6. Next steps

Once you have a saved adapter in Drive, download it or point
`03_stage2_finetune.ipynb` / `04_eval_pipeline.ipynb` at the same Drive path
to keep working across sessions.
