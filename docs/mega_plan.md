# Mega plan — comment-format plans, a reasoning Planner, and a 6GB deployment

Working plan for the current research phase. Written 2026-08-27 out of a
research/critique session; supersedes nothing, but it is the first document
that states the **end-to-end deployment target** rather than the next fix.

Read alongside `context.md` (current state) and `docs/next_fixes.md`
(historical defect log — every item there was closed as of 2026-08-27; kept
for the diagnosis detail, not as an open list). When a step here is done,
record *why* it landed the way it did in `CHANGELOG.md`, per the convention
in `CLAUDE.md`.

---

## 0. The goal, stated as a constraint

Two-stage Planner -> Coder pipeline that runs **entirely on one RTX 2060,
6GB VRAM**, doing Python-only coding tasks. Training happens on Colab
(A100/L4); only inference has to fit the 2060.

### Why local, and what that changes

The point is not that a 2060 is what's available. It is **offline
resilience**: maximum useful work per GB of a small GPU, so that losing
internet does not stop the work. That rationale is not decoration — it
settles arguments the rest of this document would otherwise leave open.

- **There is no fallback when it matters.** A hosted model is not a backstop
  for a local one that fails during an outage; the outage is exactly when the
  backstop is gone. So a configuration that usually fits is not "mostly
  fine", it is a system that breaks precisely when it is needed.
- **Therefore reliability outranks peak quality.** Where the two conflict —
  and the headroom arithmetic below shows they do — **take the headroom.** A
  3B that always runs beats a 7B that OOMs when a video is playing. This is
  the tie-breaker for D-3 and it should not have to be re-argued per arm.
- **Efficiency puts the two-stage architecture on trial.** Two stages means
  two forward passes and a stage transition to answer one question, in a
  budget that would hold one model doing it directly. That is a real
  efficiency cost, and it is the case Gate 1's **G2** exists to test. The
  hypothesis that justifies paying it: a small model may do English -> plan
  and plan -> code as two easy problems better than one hard problem. If G2
  says otherwise, the honest move for this goal is one model, not two.
- **The deliverable is code you can actually run offline.** Generated code
  that imports something not installed, or that fetches a URL, is worthless
  during an outage. This turns C2's "write runnable examples — no network, no
  missing fixtures" from eval hygiene into a **product requirement**: the
  library surface the model is trained to emit has to match what is installed
  on the machine. See F5.

### The VRAM budget is the whole design

6144 MiB total. What is left after the desktop depends on the host OS and on
how the machine is used — see "Host OS" below.

**The operating assumption is normal desktop use.** The model runs while the
machine is being used as a machine: browser open, editor open, whatever else.
Nothing in this plan may assume a cleared-off GPU before a run. Two
consequences, and the second is the one that bites:

- **The realistic budget is ~4.8-5.2 GB**, not the ~5.9 GB a headless host
  would give. A Fedora/Wayland session is ~0.3-0.6 GB, and a
  hardware-accelerated browser with real tabs open adds ~0.2-0.5 GB on top —
  more with video playing.
- **It is not a fixed budget, it varies during the session.** Opening tabs,
  playing video, plugging in a second monitor, screen sharing: all of it
  moves the floor under the model *while the model is loaded*. A
  configuration measured to fit at 5.2 GB will OOM later when the browser
  grows. So the design target is not "fits against the measured maximum", it
  is **"fits with enough headroom to absorb the desktop's own variation"**.
  See the headroom rule below.

**The stages run sequentially, so that budget is per stage, not shared.**
The Planner produces a plan, the process (or adapter) hands it to the Coder,
the Coder produces the code. At no point do both sets of weights need to be
resident. This is the intended design and the rest of the document is written
around it; see D-2 for what it costs. Under a varying budget this matters
more, not less — one model resident at a time is a lower peak and more slack
for the desktop to fluctuate into.

**The stages run sequentially, so that budget is per stage, not shared.**
The Planner produces a plan, the process (or adapter) hands it to the Coder,
the Coder produces the code. At no point do both sets of weights need to be
resident. This is the intended design and the rest of the document is written
around it; see D-2 for what it costs.

| Params | fp16 | Q8_0 | Q5_K_M | Q4_K_M |
|---|---|---|---|---|
| 12B | 24 GB | ~12.8 GB | ~8.5 GB | ~7.3 GB |
| 7B | 15 GB | ~8.1 GB | ~5.4 GB | ~4.7 GB |
| 4B | 8.6 GB | ~4.3 GB | ~3.1 GB | ~2.5 GB |
| 3B | 6.6 GB | ~3.3 GB | ~2.3 GB | - |
| 1.5B | ~3.1 GB | ~1.7 GB | - | - |

Consequences that constrain every step below:

- **No 12B model fits at any usable quantization.** 12B @ Q4_K_M is ~7.3 GB
  against ~5.9 GB headless, and that is before KV cache. Sequential execution
  does not rescue this — the ceiling is one model at a time, not one model
  per half-turn. Distillation to a smaller student is still the only path,
  not one option among several.
- **"High bit-width" caps model size.** True fp16 caps you near 2-3B params;
  Q8 caps you near 4-5B. "High bits AND large model AND 6GB" is arithmetically
  impossible — the real question is where to sit on the params x bit-width
  curve, and that is an empirical question (step D3).
- **Two separate 7B models do not need to co-reside** (~9.4 GB at Q4, which
  would not fit). Because the stages are sequential they never have to. What
  the pipeline actually needs is one ~5 GB slot reused twice per request. The
  open question is therefore *how* the swap happens, not *whether* the
  pipeline fits. See decision D-2.
- **Each stage gets the full budget.** This is the main consequence of the
  sequential design and it changes D-3: the params x bit-width point is
  chosen per stage, not once for a shared 5 GB. A 7B @ Q4_K_M Coder and a
  1.5B @ fp16 Planner is a legal configuration. 7B @ Q5_K_M (~5.4 GB) is
  **not** legal under normal desktop use — it needs the iGPU path (P1b).
- **The headroom rule.** Weights are not the whole footprint. Budget
  `weights + KV cache + ~0.4 GB` of llama.cpp CUDA context and compute
  buffers, then leave **at least 0.5 GB unspent** for the desktop to grow
  into. KV cache is small on Qwen2.5-7B (GQA: 28 layers x 4 KV heads x 128
  dim, ~56 KB/token at fp16, so ~115 MB at 2048 ctx) — the 0.4 GB of buffers
  and the 0.5 GB of slack are the terms people forget.
- **The current baseline arm is marginal, and on this machine that is now
  decisive.** 7B @ Q4_K_M under normal desktop use: 4.7 weights + ~0.12 KV +
  ~0.4 buffers = **~5.2 GB against a ~4.8-5.2 GB varying budget, with zero
  headroom.** It may load and run fine, then die when a video starts. The
  escape hatch was the iGPU path — **and P1b has resolved: this machine has no
  integrated graphics.** So on a 6 GB card D-3 has to come down a size. Do not
  treat 7B @ Q4 as the safe default it looks like in the table. (P1c is
  checking whether the card is in fact 6 GB.)
- **The binding resource moves from VRAM to system RAM and disk.** Two GGUFs
  are ~9.4 GB on disk, and the swap is only cheap if both stay in the page
  cache. Budget 16 GB of system RAM as a floor, 32 GB to be comfortable. See
  P4.

### Hardware facts about the target card

The RTX 2060 is Turing, compute capability sm_75:

- **No bf16.** Train in bf16 on the A100; that is fine, it only affects the
  training host.
- **The `flash-attn` package (FlashAttention-2) requires Ampere (sm_80+) and
  will not run.** This is a statement about that PyTorch package only.
  llama.cpp's `--flash-attn` is a separate implementation with kernels for
  older architectures and is expected to work on Turing — confirm on the card
  rather than inheriting the sm_80 claim, because it affects KV-cache size.
- bitsandbytes 4-bit works but is slow on Turing. The deployment path should
  be **llama.cpp / GGUF** or ExLlamaV2, both of which are good on sm_75.
- Memory bandwidth ~336 GB/s, so decode speed is roughly bandwidth over model
  size: a 3.3 GB Q8 model lands around 60-80 tok/s, a 4.7 GB Q4 7B around
  45-60 tok/s. Two-stage inference pays that cost twice per request, plus
  whatever the stage transition costs (D-2), which is an accepted trade.

### Host OS: Fedora 44, not Windows

The target host is Fedora 44. This is a real advantage over the Windows
baseline the first draft assumed, but the size of it depends on how the
desktop is configured, and one of the differences matters more for
*measurement* than for capacity.

Ordered by what they cost *you*, not by how much VRAM they recover. The
machine is in normal use while the model runs, so an option that requires
clearing the desktop first is not really available.

- **An iGPU is the only option that changes nothing about how you work, and
  it is therefore the recommended one.** If the CPU has integrated graphics,
  set it primary in the BIOS and plug the monitor into the motherboard. The
  desktop, the browser and the compositor all move off the 2060 permanently,
  which restores the full ~5.9 GB *and* removes the variability — the model's
  budget stops depending on how many tabs are open. One-time BIOS change, no
  ongoing discipline. This is what makes 7B @ Q4_K_M comfortable instead of
  marginal. Check with `lspci | grep -iE 'vga|3d|display'`.
- **A GNOME/Wayland session on the 2060 costs roughly 0.3-0.6 GB**, against
  0.6-1.0 GB for a Windows desktop, and a hardware-accelerated browser adds
  ~0.2-0.5 GB on top. This is the fallback if there is no iGPU, and it is the
  case the ~4.8-5.2 GB working budget describes. Disabling browser hardware
  acceleration recovers a few hundred MB and is the one cheap mitigation that
  does not change how the machine is used — worth doing, not worth counting
  on.
- **Headless recovers the most VRAM and is listed last on purpose.** Dropping
  to `multi-user.target` leaves the card at tens of MiB (~5.9 GB usable), but
  it means not using the desktop while the model runs, which contradicts the
  operating assumption. Useful for *measurement* — reading 3 in P1 is what
  isolates the desktop's cost — and for a batch job left running. Do not size
  the deployed configuration against it.
- **No WDDM VRAM oversubscription — which cuts both ways.** Windows silently
  spills VRAM to system RAM under pressure, turning an over-budget
  configuration into a mysterious 5-10x slowdown instead of an error. The
  CUDA path on Linux fails loudly with an OOM. For measurement that is worth
  more than the 0.3 GB. But under normal desktop use it is also a risk the
  Windows setup did not have: if the model takes the card to the edge and the
  compositor then needs memory it cannot get, you can stutter or lose the
  session, not just the model. That is the concrete reason the headroom rule
  above reserves 0.5 GB rather than fitting to the measured maximum.
- **Setup cost, to expect once:** the NVIDIA driver comes from RPM Fusion
  (`akmod-nvidia` + `xorg-x11-drv-nvidia-cuda`), and with Secure Boot on you
  must sign the kernel module and enroll the key (MOK) or the driver will not
  load after a kernel update. llama.cpp's CUDA build is also markedly less
  troublesome on Linux than on Windows, which is a second, smaller reason
  this move helps.

---

## 1. Open decisions this plan deliberately does not prejudge

These are gated on measurements, not on argument. Do not let an
implementation step quietly settle one.

- **D-1 — Plan format.** Current `<PLAN>/<STEP>` DSL, or Python-comment
  rendering. Phase A makes this a flag so it can be measured rather than
  guessed. Decided by Gate 1 / step D3.
- **D-2 — How the stage transition happens.** Reframed: the stages run
  sequentially, so this is no longer "does the pipeline fit" but "what does
  the handoff cost". Three mechanisms, cheapest first:
  1. **One shared base + two LoRA adapters.** ~4.7 GB of weights resident
     plus two ~100 MB adapters; llama.cpp can hold both adapters loaded and
     switch which one is active per request, so the transition is
     milliseconds and nothing reloads. Requires that a single base serves
     both stages well — which is what B1 (dropping the DSL special tokens,
     so the Planner uses a stock tokenizer) unblocks.
  2. **Process swap.** Kill the Planner server, start the Coder server;
     `llama-swap` in front of them does this behind one OpenAI-compatible
     endpoint. The old "5-20 s stall" figure was pessimistic for GGUF: with
     the weights hot in the page cache the copy to VRAM is a
     PCIe-bandwidth-bound transfer, so expect **~1-3 s** per swap on a
     PCIe 3.0 x16 slot and worse only on a cold cache. Two swaps per request
     if you swap back.
  3. **CPU-resident Planner.** Only if a big Planner turns out to be
     necessary; the sequential design mostly removes the reason to consider
     it.
  Preference is 1, with 2 as the fallback that makes the design safe — the
  pipeline works either way, which is the point of writing the sequential
  assumption down. What is *not* free about 2: it costs seconds per turn, and
  it costs page cache (P4). Decided by step D3, measured by F4.
- **D-3 — Params vs bit-width, chosen per stage.** 7B @ Q4_K_M vs 3B @ Q8_0
  vs 1.5B @ fp16. The literature genuinely disagrees in this regime:
  Dettmers & Zettlemoyer (arXiv 2212.09720) put 4-bit on the Pareto frontier
  for a fixed memory budget, while "Not All Bits Are Equal"
  (arXiv 2510.10964) finds that below ~8B, higher-precision weights win on
  reasoning-heavy tasks. Measure on our own task.
  Because the stages are sequential, this is **two decisions, not one**, and
  they can differ. The asymmetry worth testing explicitly: English -> a short
  list of known verbs is the easy half and may be served by a 1.5B, while
  pseudocode -> correct pandas/tkinter/requests code is the hard half and can
  take the whole ~5 GB. Note the tension with D-2: picking different families
  per stage forecloses the shared-base option and forces the process swap.
  **Tie-breaker, per §0: when quality and headroom conflict, take the
  headroom.** An arm that needs an idle desktop has not passed. Decided by
  step D3.
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
  building on it. A second, verified candidate surfaced 2026-08-27:
  `Qwen2.5-Coder-14B-Instruct`, the teacher behind
  `h0ney-badger/qwen2.5-coder-1.5b-python-distill` (D2) — same family as the
  student, Apache-2.0, sidesteps the Gemma-terms and unverified-repo problems
  entirely. Worth weighing against the Gemma option in P2 rather than
  defaulting to it, since a same-family teacher may cap achievable quality
  lower than a larger, differently-trained one.
- **D-6 — Does the pipeline need execution feedback (an interpreter tool)
  to be trustworthy?** Raised in conversation 2026-08-27, then again
  2026-08-28. A small model can't reliably judge its own code's correctness
  from static reasoning alone the way a much larger model sometimes can, so
  the working assumption is yes — but whether the *specific* untuned model
  already in hand (§D2) can actually follow a tool-call convention and use
  the feedback it's given is untested. Decided by step G4, which needs no
  GPU and can run any time.

---

## 2. Phases at a glance

| Phase | What | Who | Blocked by |
|---|---|---|---|
| P | Prerequisites / facts | user + agent | - |
| A | Format infrastructure | agent (CPU) | - |
| B | Notebook updates | agent (CPU) | A |
| C | Eval capacity | user + agent | A |
| G | Agentic execution loop (interpreter) | agent + user (Ollama) | - |
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

### P1. Measure the desktop's VRAM **high-water mark**, not its minimum
Every budget in this document is an estimate until these numbers exist. The
number that matters is the worst realistic case, because the model has to
survive it while already loaded — a one-off idle reading will size the
deployment too big and it will OOM later.

Sample over a normal working session rather than taking a single reading:

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 | tee ~/vram.log
```

Leave that running for an hour of ordinary use — browser with your usual
tabs, editor, whatever else, and at some point a video playing full-screen,
since that is the realistic worst case. Take the **maximum**, not the mean.

Then two reference readings to isolate where it goes:

1. Desktop with the browser closed.
2. Headless: `sudo systemctl isolate multi-user.target`, read over SSH or on
   the TTY. This is not a deployment configuration (see "Host OS"); it is how
   you find out what the desktop is costing you.

**Done when:** the high-water figure, the two reference figures, and the
resulting budget after the headroom rule are in `context.md`.

### P1b. Decide how the display is driven — **RESOLVED 2026-08-27: no iGPU**
`lspci` on the target reports exactly one display device:

```
VGA compatible controller: NVIDIA Corporation TU104 [GeForce RTX 2060] (rev a1)
```

No integrated graphics. **The iGPU path does not exist on this machine**, so
the recommended configuration is off the table and the fallback is the
deployment case:

- The desktop stays on the 2060, permanently and during every run.
- The deployment budget is P1's high-water figure minus the headroom rule,
  and D-3 must be sized against that — not against 6144 MiB, and not against
  a headless reading.
- Headless is not an option here. The machine is in normal use while the
  model runs (§0), so `multi-user.target` stays a measurement tool only.
- **The 0.5 GB reserve is now mandatory rather than prudent**, because
  nothing else absorbs the desktop's variation.

The one cheap mitigation that survives: disabling browser hardware
acceleration recovers a few hundred MB without changing how the machine is
used. Worth doing. Not worth sizing against.

**Done when:** done — recorded here and in `context.md`. What is still open is
P1's high-water figure and the card's true VRAM (see P1c).

### P1c. Confirm the card's actual VRAM — **blocks every budget in this doc**
`lspci` names the die **TU104**, which is not the usual RTX 2060 silicon.
TU106 is the normal RTX 2060 die; TU104 is what RTX 2070 SUPER / 2080 /
2080 SUPER are built on, and it appears on RTX 2060-class boards only as
late-run die salvage. So the label is either a genuine TU104-salvage RTX 2060
(6 GB, everything here stands) or a `pci.ids` mislabel of a different,
possibly **8 GB**, card.

This is not a curiosity. Every number in this document is derived from
6144 MiB. An 8 GB card would move the working budget from ~4.8-5.2 GB to
~6.8-7.2 GB, which puts 7B @ Q5_K_M comfortably in range and changes D-3's
answer. Do not take `lspci`'s marketing string as authoritative — ask the
driver:

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

Note that either way the Turing facts above are unaffected: TU104 and TU106
are both sm_75, so no bf16, the same llama.cpp/GGUF deployment path, and
similar memory bandwidth.

**Done when:** the driver-reported name, total VRAM and compute capability are
in `context.md`, and if it is not 6144 MiB every budget figure in this
document has been recomputed.

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

### P4. Record system RAM and free disk
**Partially answered 2026-08-27:** the boot device is a SanDisk/WD SN530-class
**DRAM-less** NVMe (PCIe 3.0 x4). Two consequences for D-2's process-swap
fallback. Good: model loading is a large *sequential* read, which is where
these drives are least penalized — expect roughly 2-2.5 GB/s, so a cold
~4.7 GB GGUF load lands around **2-3 s**, not the disaster "DRAM-less" might
suggest. Bad: DRAM-less means the drive leans on Host Memory Buffer and
degrades sharply under random I/O and sustained writes, so a swap forced to
hit disk on a busy system is less predictable than the sequential number
implies. Net: **the page cache matters more here, not less** — RAM is what
keeps the swap off this drive entirely. Still needed: the RAM figure.
Only relevant because the design is sequential. If D-2 lands on the process
swap, the swap is fast only while both GGUFs sit in the page cache; if they
do not, every stage transition re-reads gigabytes from disk. `free -h` and
`df -h` on the target, into `context.md`. 16 GB of RAM is the floor for two
~4.7 GB models plus the OS, 32 GB is comfortable. Note the disk type too —
NVMe vs SATA SSD changes the cold-cache number by several seconds.

**Done when:** RAM, free disk and disk type are in `context.md`, with a note
on whether the process-swap fallback is viable on this machine.

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
the same budget. **If the two-stage pipeline does not beat this, the
architecture is not earning its complexity.** This is the most important
single experiment in the document and it is cheap.

The sequential design raises this bar rather than lowering it. The one-stage
baseline gets the full ~5 GB *and* pays no stage-transition cost, so the
two-stage pipeline now has to beat it on quality by enough to justify two
forward passes plus a swap. Score both, and record the wall clock alongside
the quality number — a two-stage win that costs 4x the latency is a different
result from one that costs 2x.

**Under the efficiency goal (§0) this is the experiment that decides whether
the project's architecture survives**, not merely a sanity check. Two stages
is a real cost paid per request, in a budget that would hold one model doing
the job directly. Run G2 before spending Colab hours on either fine-tune. If
one stage wins, the plan is one model, and most of Phases B, E and the D-2
machinery stop being necessary — a much cheaper outcome to discover now than
after Phase E.

---

## Phase D — Base model bake-off (user, Colab)

### D1. Evaluate quantized, not fp16
Score what ships. An fp16 A100 number does not predict Q4 behavior on a 2060.

Score against the **P1 high-water budget minus the headroom rule**, not
against 6144 MiB and not against the headless figure. An arm that only fits
on an idle desktop has not passed.

### D2. Arms
Score every arm **on both stages separately**, since the sequential design
lets the two stages pick different points (D-3).

- `Qwen2.5-Coder-7B-Instruct` @ Q4_K_M (~4.7 GB) — the original baseline.
  **P1b resolved against it on a 6 GB card:** there is no iGPU, so ~5.2 GB
  all-in against a ~4.8-5.2 GB varying budget leaves no headroom, and §0's
  tie-breaker says take the headroom. Keep it in the bake-off as the quality
  reference — it is what the smaller arms are trying to match — but treat it
  as **not deployable** unless P1c reports more than 6144 MiB.
- `Qwen2.5-Coder-7B-Instruct` @ Q5_K_M (~5.4 GB) — **drop unless P1c reports
  an 8 GB card.** No iGPU path exists to make it fit at 6 GB.
- `Qwen2.5-Coder-3B-Instruct` @ Q4_K_M or Q5_K_M (~2.0-2.3 GB) — the arm that
  exists specifically because the budget varies. Comfortable headroom under
  normal desktop use with no BIOS change and no habits to keep. If the
  quality gap to 7B is small on our task, this is the honest answer for a
  machine in daily use. (License: Qwen Research, non-commercial — P2.)
- `Qwen2.5-Coder-3B-Instruct` @ Q8_0 (~3.3 GB) — the "high bits" arm (license, P2)
- `Qwen2.5-Coder-1.5B-Instruct` @ fp16 (~3.1 GB) — the true high-bits arm
- `h0ney-badger/qwen2.5-coder-1.5b-python-distill` @ Q4_K_M (~0.94 GB) —
  already-published Python-only distill of the same 1.5B base (teacher:
  `Qwen2.5-Coder-14B-Instruct`, Apache-2.0 throughout); free to add, no
  training required. See `context.md` 2026-08-27 for verified details and
  `docs/PLAN.md` for how to run it. Doubles as a candidate for **G2**
  (single-stage, plain-instruction baseline), since it wasn't trained on this
  project's DSL/pseudocode format.
- a 4B (Gemma 4 4B or similar) @ Q6_K
- the 12B teacher, unquantized — as a **ceiling**, not a candidate

### D3. Settle D-1, D-2, D-3 from the results
Format, the stage-transition mechanism, and the params/bit-width point for
each stage all fall out of this table. Record the decision and the numbers
behind it in `context.md`.

The question D-2 actually turns on: **does the best Planner arm and the best
Coder arm share a base?** If the same family and size wins both, take the
shared-base + adapters route and the transition is free. If a small Planner
and a large Coder win, price the process swap (P4, F4) and decide whether the
quality delta is worth seconds per turn. Do not let the elegance of shared
adapters pick the models — the swap is a working fallback, which is exactly
why it was worth writing the sequential assumption into the plan.

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

### F4. Measure on the actual 2060, under Fedora
Not on Colab, and under the display configuration P1b settled. Record:

- Peak VRAM (weights + KV + activations) for each stage separately, against
  the P1 figure for that configuration.
- Real tok/s for each stage, prompt processing and decode separately.
- **The stage-transition cost**, whichever mechanism D-2 chose: adapter
  switch latency, or process-swap latency measured both warm (page cache
  primed) and cold (`echo 3 | sudo tee /proc/sys/vm/drop_caches` first). The
  cold number is what a first run after boot actually feels like.
- End-to-end wall clock for one English task -> final Python, which is the
  only number that answers "will I tolerate this".
- **A soak test under real use**, not a clean-room run: load the pipeline,
  then use the machine normally for an hour — tabs, video, whatever you
  actually do — and confirm it neither OOMs nor drags the desktop down. This
  is the step that catches a configuration sized against an idle reading, and
  it is the one that decides whether the chosen arm actually ships.

### F5. Verify it actually works offline
The goal is offline resilience, so the acceptance test is an outage, not a
benchmark. Disable networking (`nmcli networking off`) and confirm the whole
loop still closes:

- **No runtime downloads.** GGUFs on local disk; `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` set, so nothing silently reaches for the hub. A
  `from_pretrained` that works only because the hub is up is a latent failure.
- **llama.cpp already built and installed**, not compiled on demand.
- **The target libraries already installed** in the environment the generated
  code runs in — pandas, numpy, sklearn, torch, requests, tkinter, whatever
  the DSL verbs imply. Keep a wheel cache so a fresh venv is possible offline.
  This is the half of "it works offline" that is about the *output* rather
  than the model, and it is the half that is easy to forget.
- **The eval harness runs offline** (it already does — `src/eval/` is local),
  so you can still score changes during an outage.

**Done when:** with networking disabled, an English task goes to final Python
and that Python executes.

**Done when (phase):** the full pipeline answers a held-out task on the 2060,
within the measured VRAM budget, at a latency you will actually tolerate,
with the desktop in normal use and the network down.

---

## Phase G — Agentic execution loop (interpreter-in-the-loop)

Raised in conversation 2026-08-27 and explicitly deferred — see
`CHANGELOG.md`'s "Discussed, not committed" entry ("whether the target 7B
models can be made agentic the way Sonnet/Opus are... settled on keeping the
existing 50 pairs as a working v1 rather than growing the set further"). No
harness or loop exists anywhere in the repo. Picked back up 2026-08-28.

**What "agentic" means here, specifically:** an execution-feedback loop, not
general tool use. The interpreter *is* the agentic capability for a coding
model — a loop with no execution tool is just a longer chat, and it doesn't
address the actual problem (the model can't tell if its own code is good).
Grounding the model in real `stdout`/`stderr`/tracebacks compensates for
exactly the self-assessment weakness a small model has, which is also why
this phase is not deferred behind Phase D/E/F: a model too small to reason
its way to correct code is the *reason* to build the feedback loop, not a
reason to wait for a bigger one.

**Blocked by:** nothing. G1-G4 need no GPU and don't depend on which base
model Phase D eventually picks, or on the D-1 format decision — they test
the loop mechanism itself against whatever instruct model is already running
locally (see `docs/PLAN.md` for the `h0ney-badger` Ollama setup). Can run in
parallel with Phase A/B/C.

### G1. Extend the interpreter tool to capture output, not just pass/fail
`src/eval/harness.py`'s `run_and_check()` returns `EvalResult(passed, error)`
— `error` is only a traceback on failure, and nothing is captured on
success. An agent loop needs the model to see what its code actually
printed, not only whether it raised. Add `stdout`/`stderr` capture (e.g.
`contextlib.redirect_stdout`/`redirect_stderr`) to `EvalResult`. Extend the
existing function rather than writing a second exec runner.

**Also close the sandbox gap before this executes model-generated code.**
The module's own docstring already flags plain `exec()` as "NOT a security
sandbox" — an acceptable trade while the only input was hand-authored
reference/eval code. Once a model is generating what runs inside the loop,
unreviewed output needs a real boundary: a subprocess with a timeout and
resource limits at minimum, a container if this ever runs anywhere with
credentials nearby.

**Done when:** the extended runner returns captured stdout/stderr for both a
passing and a failing example, and executes model-supplied code through
something stronger than in-process `exec()`.

### G2. Define a minimal tool-call convention
A plain text marker the model emits to request execution (e.g. a fenced
` ```run ` block) — not a JSON function-calling schema. Small, non-tool-
tuned instruct models follow plain textual conventions far more reliably
than a structured schema they were never trained to emit.

**Done when:** the convention is written down as a system-prompt fragment,
and a handful of hand-written example completions parse correctly against it.

### G3. Build the orchestration loop
New code — nothing in the repo does this today. Call model -> parse for a
tool call -> execute via G1's interpreter -> append the captured output as
the next turn's context -> call model again -> repeat until the model emits
a final-answer marker or an iteration cap is hit.

**Done when:** the loop runs end-to-end against a stub model function (the
same pattern `src/ui/console.py` uses to stay testable without a real
model), and correctly terminates on both the final-answer marker and the
iteration cap.

### G4. Test against the current *untuned* model before training anything
Same principle Gate 1 already applies to the Planner/Coder split — prompt an
un-tuned model before spending any fine-tuning effort. Point G3's loop at
the `h0ney-badger` distill already running locally on Ollama and observe
whether it follows G2's convention and actually uses the execution feedback
to fix wrong code, or ignores it, or gets confused by the convention itself.
This needs no training investment to answer, and it settles D-6.

**Done when:** a small hand-picked set of tasks (reuse a few `english` rows
from `data/stage1_planner/englishtopseudo.jsonl`) has been run through the
loop, and the outcome is recorded in `context.md`.

### G5. Decide whether to fine-tune for tool use, based on G4
Only pursue this if G4 shows the model can't follow the convention or
ignores the feedback it's given. That would mean building a new kind of
training data — execution-feedback traces — a separate track from the
existing Planner/Coder SFT data and from Phase E's reasoning traces, not a
default next step taken regardless of what G4 shows.

---

## 3. The thing this plan cannot fix

`data/stage1_planner/englishtopseudo.jsonl` and
`data/stage2_coder/pseudotopython.jsonl` hold **50 rows each** — 40 train, 10
eval. At that size, base model choice is not the limiting factor and no amount
of model shopping will change it. Phases A, B and D are all cheap; Phase C is
the one that actually moves the ceiling, and it is the least fun. Do it
anyway.
