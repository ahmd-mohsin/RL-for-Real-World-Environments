# Long-horizon RLM harness, study 05: the baseline the new harness must beat

- **What.** The M16 validation ladders (design §4) run against the **current** harness (`RLMPolicy` / `RLMHarness` at
  HEAD 775f30f), so the new harness has numbers to beat.
- **Run:** 2026-10-13, CPU-only Mac, local Docker. Every RLM run used the **Docker kernel** (`python:3.12-slim`).
  No paid API, no GPU.
- **Clean tree.** Phase-1 agents were editing `kernel.py`, `repl.py` and `record.py` in the shared worktree during the
  runs. The baselines therefore ran on a `git archive HEAD` export with only agent E's new files laid over it
  (`scripted_rlm.py`, `bench/longctx/`, `longhorizon/ladder.py`, `scripts/lh_ladder.py`). An early run on the live tree
  reported invalid protocol frames on every Docker cell (217 of 217) at the 5,000 rung. The clean tree reported none.
  That is probably a mid-edit state of the kernel protocol. Phase 2 should still check it.

## Method

- **Model.** `crucible.runtime.scripted_rlm.BacklogRLM`, a scripted model that **writes Python cells** and keeps **no
  hidden memory**. Each reply is a function of the messages it receives.
  - Its beliefs come from evidence it can see: the policy document it read, the ops announcement, a summary's `RULE` line,
    the pending listing it printed, and the customer replies in INBOX announcements. Its own earlier code never counts as
    evidence.
  - It defines `work(threshold)`, `handle` and `handle_reply` in the kernel once. It then handles one open ticket per
    cell, and it handles customer replies once the open queue is empty (`replies="batch"`), so a reply has to survive a
    compaction.
- **Exact TITO.** Calls go through `TitoLLM`: `EngineLLM` over `HFChatCodec` with the real Qwen2.5 tokenizer, as in
  study 03. Every token number below is a real Qwen token unless it is marked otherwise.
- **Runner.** The M10 `Runner` with `rollout_mode="train"`, budget 32k, trigger 0.8, 3 kept turns, a checkpoint every
  500 steps. Backlog seed 0.
- **Variants:**

  | Variant | Summary | Re-reads history after a compaction |
  |---|---|---|
  | faithful | every schema section | yes |
  | forgetful | drops the 50 → 25 amendment, the awaiting ids and the unhandled replies | no |
  | lazy | faithful | no |
  | forgetful_reread | forgetful | yes (INBOX / OBS) |

- **Lost-constraint rate.** The share of rule-bearing actions after the first compaction (refunds and status changes,
  read from the World trace) that break the rule in force when they were made:
  - an approval was needed above the threshold of the time and there was none;
  - a second refund on the same order;
  - an awaiting ticket was changed before its reply arrived.

  "Missed" counts awaiting tickets that were never resolved.

## 1. What the model loses (reward; lost-constraint rate after the first compaction; missed awaiting tickets)

| Rung (tickets) | Calls / compactions (faithful) | faithful | forgetful | lazy | forgetful_reread |
|---|---|---|---|---|---|
| 100 (15) | 94 / 0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 500 (80) | 496 / 1 | 1.0; 0; 0/5 | **0.913; 0.083 (2 approvals); 5/5** | 1.0 | 1.0 |
| 1,000 (160) | 978 / 2 | 1.0; 0; 0/21 | **0.788; 0.100 (13); 21/21** | 1.0 | 1.0 |
| 2,500 (410) | 2,457 / 6 | 1.0; 0; 0/64 | **0.754; 0.084 (37); 64/64** | 1.0 | 1.0 |
| 5,000 (820), default config | 1,150 / 3 | **0.227, stopped `too_many_errors`** | 0.227 | 0.227 | 0.227 |
| 5,000 (820), `max_stalls=10⁶` | 4,976 / 15 | 1.0; 0; 0/125 | **0.741; 0.089 (87); 125/125** | 1.0 | **0.935; 0; 53/125** |

- **The ladder discriminates.** The forgetful summary costs 9–25% of the reward from the 500 rung on.
  - Its rule breaks start only after the first compaction (0 before it in every run).
  - Its ΔΦ per segment drops after the summary, for example 0.325, 0.325, 0.138 at the 1,000 rung against 0.325, 0.356,
    0.319 for the faithful model.
  - The faithful and lazy models stay at 1.0, so a faithful summary alone is enough in this harness.
- **Episode-wide stall counter (B6).** At 5,000 calls, every variant was **ended by the stall counter** at 1,150 steps.
  - Two message deferrals in a row repeat the same `work(...)` cell. Three such repeats over 1,000 steps add up to
    `max_stalls=3` for the whole episode.
  - It recurs: 1 stall at the 1,000 rung and 2 at the 2,500 rung.
- **The 8,192-character output cap breaks recovery from history.** `forgetful_reread` repairs everything up to the 2,500
  rung. At 5,000 its re-read cell prints 72 pending replies, and the output is cut to 8,192 characters from the 5th
  compaction on (verified). The 53 replies in the cut middle are lost.
  - The new harness's `OUT(i)` (full output in the ledger) and paged history reads must turn this into 1.0.

## 2. What the harness costs (faithful model)

| Rung | Prompt peak, root / compaction (real tokens) | ÷ 32k window | Calls over the window | Entered root tokens | Wall s (ms per call) | Kernel restore (journal replay) | Checkpoint: last / all saves |
|---|---|---|---|---|---|---|---|
| 100 | 11.6k / – | 0.36 | 0 | 11.6k | 1.4 (14.6) | 0.46 s, 20 cells | – |
| 500 | 38.7k / 39.4k | **1.23** | 13 | 53.8k | 2.9 (5.9) | 0.78 s, 90 cells | – |
| 1,000 | 39.6k / 40.2k | **1.26** | 28 | 106k | 5.9 (6.0) | 1.2 s, 181 cells | 1.0 MB / 1.0 MB |
| 2,500 | 39.8k / 40.6k | **1.27** | 135 | 289k | 17.1 (7.0) | 3.9 s, 453 cells | 3.7 MB / 9.7 MB |
| 5,000 (`max_stalls=10⁶`) | 40.0k / 40.9k | **1.28** | 237 | 650k | 42.9 (8.6) | 9.8 s, 918 cells | 8.7 MB / 44.8 MB |

- **B1 confirmed with the real tokenizer.** The chars/4 trigger lets every segment run to 1.23–1.28× the window, and the
  compaction call is always the largest prompt.
- **B2.** With `enforce_window` (the engine refuses a prompt longer than 32k, as a served model would), the 1,000 rung
  overflows at step 274. All 3 attempts are masked `agent_run_error`, and **the episode yields no reward**.
- **B4 and B5.** Restore and checkpoint costs grow with the episode: restore time is linear in cells, and checkpoint bytes
  written are superlinear.
- **Records.** Every RLM run was `valid_for_training`.
- **Native reference** (`LLMToolPolicy` + `ContextConfig`, `ReplayLLM` over the diligent plan, 1,000 rung). This is the
  plumbing reference, not a no-memory model.
  - Reward 1.0, 918 steps, 5 compactions, wall 3.9 s.
  - The prompt peak is 25.6k, an estimate in chars/4 units (ReplayLLM has no tokens).

## 3. Faults (1,000 rung, faithful, attempt 1)

| Fault | Current harness |
|---|---|
| Lost worker at model call 100 (checkpoint every 500) | a1 masked `agent_run_error`; a2 resumed from the checkpoint, finished, and was masked `capture_incomplete` (**B3**); a3 restarted from step 0 and scored 1.0. Wall time 16.4 s against 5.9 s |
| Kernel killed mid-cell (at World call 300) | a1 masked `agent_run_error` ("ConnectionResetError: Connection lost"). **No in-episode restore.** a2 restarted from step 0 and scored 1.0 |
| First compaction call overflows | a1 masked `agent_run_error` (the overflow escapes `_maybe_compact`, **B2**); a2 restarted from 0 |

- **Kernel killed between cells.** Restored by replay, as study 03 found.
- **Pathological regex in a cell.** Stopped by the cell deadline.
- **50 MB print.** Capped by the kernel at 1 MiB per stream.
- **`history.search` with the regex** (phase-1 ledger). Bounded.

The last three are covered by tests in `tests/unit/test_m16_e_ladder.py`.

## 4. Long-context ladder (controls)

- **Families:** oolong, MRCR and docqa, each generated from seed 0. The document is served by the World `docs` app, and
  the answer is graded from World state.
- **Controls at every rung from 8k to 4M tokens:**
  - the scripted RLM solver scores 1.0;
  - the gold answer scores 1.0;
  - the no-op scores 0.
- **Reading cost.** The solver reads the whole document in 65,536-character `docs.read` calls:

  | Size | World calls | Wall time |
  |---|---|---|
  | 8k | 3 | 0.7 s |
  | 128k | 10 | 1.2 s |
  | 512k | 34 | 2.5 s |
  | 1M | 64 | 4.5 s |
  | 2M | 125 | 8.5 s |
  | 4M | 247 | 16 s |

- **World output cap.** The World's default `max_tool_output_chars` (30,000) cut the middle out of a full read, so the
  docs World sets 140,000.

## Targets for the new harness

1. **Faithful:** 1.0 at every rung, 5,000 included, with the **default** config (no stall stop on honest repeats).
2. **forgetful_reread:** 1.0 at 5,000, because history is paged rather than cut at 8,192.
3. **forgetful:** still loses. The ladder must keep telling a bad summary from a good one.
4. **Prompt peak** ≤ the threshold with real tokens: 0 calls over the window, the compaction call included.
5. **`enforce_window`:** scored episodes, never masked.
6. **Crash and resume:** a1 resumes, and the resumed attempt is train-valid.
7. **Kernel killed mid-cell:** restored in the episode.
8. **Restore time:** sublinear, from snapshots.
9. **Checkpoint bytes:** O(segment).

## Reproduce

```
git -C crucible archive HEAD src packs/_toy scripts | tar -x -C head   # or run in the tree once phase 2 lands
# lay agent E's files over it (see scratchpad e/sync.sh), then from head/ with PYTHONPATH=src:packs/_toy/src:
python scripts/lh_ladder.py --rungs 100,500,1000,2500,5000 --variants faithful,forgetful,lazy,forgetful_reread --out b_main.json
python scripts/lh_ladder.py --rungs 5000 --variants faithful,forgetful,lazy,forgetful_reread --rlm max_stalls=1000000 --out b_5000_nostall.json
python scripts/lh_ladder.py --rungs 1000 --variants faithful,forgetful --enforce-window --out b_window.json
python scripts/lh_ladder.py --rungs 1000 --variants faithful --harness native --out b_native.json
python scripts/lh_ladder.py --rungs 1000 --variants faithful --faults crash_at_call=100 --out b_fault_crash.json   # also kill_at_world_call=300, overflow_summaries=1
python scripts/lh_ladder.py --longctx --families oolong,mrcr,docqa --sizes 8000,32000,128000,512000,1000000 --out b_longctx.json
```

- **Point at the new harness.** Pass its config with `--rlm window_tokens=...,tail_tokens=...` (each report lists
  `applied` and `ignored` fields).
- **Raw reports:** `docs/research/long-horizon-rlm/baseline/b_*.json`. Each run lists its Φ per segment, its constraint examples, its segment
  token counts (`source: tito`) and its counters.
