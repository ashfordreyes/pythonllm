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
