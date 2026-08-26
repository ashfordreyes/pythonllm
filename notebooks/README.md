# Notebooks

These are meant to run on Google Colab (Pro, A100/L4) and are added as stubs for
now — fill each in as the literature review and dataset design settle:

1. `01_data_collection.ipynb` — source real Python code (DS/DL notebooks
   from Kaggle/DS-1000/GitHub, plus everyday scripts, GUI apps, and
   bugfix/debugging examples), reverse-engineer `(english, pseudocode,
   python_code)` triples with a strong LLM, execution-filter for validity,
   write to `data/stage1_planner/*.jsonl` and `data/stage2_coder/*.jsonl`.
2. `02_stage1_finetune.ipynb` — QLoRA fine-tune the Planner base model
   (e.g. Qwen2.5-7B-Instruct) on `(english -> pseudocode)` pairs. Register the
   DSL special tokens from `src/dsl/schema.py` before training.
3. `03_stage2_finetune.ipynb` — QLoRA fine-tune the Coder base model
   (e.g. Qwen2.5-Coder-7B) on `(pseudocode -> python_code)` pairs.
4. `04_eval_pipeline.ipynb` — run held-out examples through Stage 1 alone,
   Stage 2 alone, and end-to-end; score with `src/eval/harness.py`
   (execution-based, DS-1000-style checks) plus `src/dsl/validator.py` for
   plan well-formedness.

To open one in Colab, run `python scripts/colab_link.py [notebook]` — it
pre-flights the notebook and prints a link that opens it straight from
GitHub. `02_stage1_finetune.ipynb` then sets itself up (GPU check, installs,
Drive mount, repo clone) in its own section 2, so Run all is the whole
procedure; `docs/colab_setup.md` covers what to do before connecting to a
runtime, GPU choice, and troubleshooting.

See the project plan for the full rationale and model/dataset choices.
