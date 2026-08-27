# Notebooks

These are meant to run on Google Colab (Pro, A100/L4). `02` and `04` are
written; `01` and `03` are still stubs, to fill in as the literature review
and dataset design settle:

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
4. `04_eval_pipeline.ipynb` — load the Stage 1 adapter from Drive, generate
   plans for the 10 held-out tasks, score them with `src/eval/scoring.py`,
   and write a results JSON plus `.png` charts back to Drive. Scoring is
   layered (plan validity via `src/dsl/validator.py`, verb-sequence and exact
   match, then code parse/self-containment/execution via
   `src/eval/harness.py`), because only 4 of the 50 reference snippets can
   actually be executed in a bare runtime. Its Stage 2 and end-to-end
   sections are documented but not runnable until `03` exists.

Both stages hold out the *same* rows: `src/splits.py` derives them by index
from `SPLIT_SEED`, so end-to-end eval scores Stage 1 and Stage 2 on the same
tasks without a split file. `python scripts/check_data.py` prints the
resolved split.

To open one in Colab, run `python scripts/colab_link.py [notebook]` — it
pre-flights the notebook and prints a link that opens it straight from
GitHub. `02_stage1_finetune.ipynb` then sets itself up (GPU check, installs,
Drive mount, repo clone) in its own section 2, so Run all is the whole
procedure; `docs/colab_setup.md` covers what to do before connecting to a
runtime, GPU choice, and troubleshooting.

Once an adapter exists, `docs/colab_console.md` covers `src/ui/console.py` —
an ASCII-panel chat UI that runs in a Colab cell rather than a notebook of its
own, so you can talk to the Planner (and the Coder, when `03` exists) without
re-running an eval notebook. It takes a planner callable and a coder callable
and works with either one missing.

See the project plan for the full rationale and model/dataset choices.
