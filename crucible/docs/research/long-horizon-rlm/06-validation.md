# Long-horizon RLM harness, study 06: validation of the M16 harness against the baseline

- **What.** E's ladders (`05-baseline.md`) run against the integrated M16 harness (`actors/rlm/agent.py`, `policy.py`,
  `runtime/harness/rlm.py`) with its **default config** at a 32k window: no stall or turn overrides.
- **Run:** 2026-10-13, CPU-only Mac, local Docker. Every RLM run used the **Docker kernel** (`python:3.12-slim`, dill pushed
  in for snapshots). Calls went through `TitoLLM` with the real Qwen2.5 tokenizer, so every token number is a real Qwen token.
  No paid API, no GPU.
- **Same models, same seeds.** `BacklogRLM` (no hidden memory), backlog seed 0, the faithful / forgetful / lazy /
  forgetful_reread variants of study 05. Two adaptations of the scripted model, both read from its visible messages only:
  - in a summary request in update mode, the `<previous-summary>` block is read as the oldest evidence (the M16 summary
    request puts the previous checkpoint in its last message; read last, it overrode the amendment that arrived since);
  - a model whose checkpoint kept no pending items (forgetful) pages a cut re-read from the ledger copy the cut names
    (`OUT(i, cursor=...)`), with a 300-character overlap.
- **Raw reports:** `validation/v_main.json`, `v_window.json`, `v_faults.json`, `v_longctx.json`, `campaign.log`.

## 1. What the model loses (reward; lost-constraint rate after the first compaction; missed awaiting tickets)

| Rung | Variant | Baseline (m14 harness) | M16 harness |
|---|---|---|---|
| 500 | faithful | 1.0 | 1.0; 0; 0/5 |
| 500 | forgetful | 0.913; 0.083; 5/5 | 0.85; 0.099; 5/5 |
| 500 | lazy / forgetful_reread | 1.0 / 1.0 | 1.0 / 1.0 |
| 1,000 | faithful | 1.0 | 1.0; 0; 0/21 |
| 1,000 | forgetful | 0.788; 0.100; 21/21 | 0.80; 0.065; 21/21 |
| 1,000 | lazy / forgetful_reread | 1.0 / 1.0 | 1.0 / 1.0 |
| 2,500 | faithful | 1.0 | 1.0; 0; 0/64 |
| 2,500 | forgetful | 0.754; 0.084; 64/64 | 0.737; 0.092; 64/64 |
| 2,500 | lazy / forgetful_reread | 1.0 / 1.0 | 1.0 / 1.0 |
| 5,000 | faithful, **default config** | **0.227, stopped `too_many_errors`** | **1.0; 0; 0/125** |
| 5,000 | forgetful, default config | 0.227 | 0.732; 0.093; 125/125 |
| 5,000 | lazy, default config | 0.227 | **1.0** |
| 5,000 | forgetful_reread, default config | 0.227 (0.935 with `max_stalls=10⁶`, 53/125 missed) | **1.0; 0; 0/125** |
| 100 | all | 1.0 | 1.0 |

- **Target 1 met.** Faithful scores 1.0 at every rung, 5,000 included, with the default config. The windowed stall detector
  no longer ends honest episodes (B6): 1 stall verdict in the whole 5,000-step run, a notice.
- **Target 2 met.** forgetful_reread repairs the forgetful summary at 5,000: 339 history reads after compaction, 256 of them
  pages of cut re-reads (`OUT(i, cursor)`), 0 lost constraints, 0 missed. The m14 harness cut the same output at 8,192
  characters and lost 53 replies.
- **Target 3 met.** The ladder still tells a bad summary from a good one: forgetful loses 15–27% of the reward from the 500
  rung on, its rule breaks start only after the first compaction, and every awaiting ticket is missed.

## 2. What the harness costs (faithful; real Qwen tokens; window 32,000, threshold 24,000)

| Rung | Prompt peak root / compaction, baseline | M16 | Calls over the window, baseline → M16 | Wall s, baseline → M16 | Kernel restore, baseline → M16 | Checkpoint last / all saves, baseline → M16 |
|---|---|---|---|---|---|---|
| 100 | 11.6k / – | 12.0k / – | 0 → 0 | 1.4 → 1.5 | 0.46 s replay of 20 cells → 0.54 s (no snapshot yet) | – |
| 500 | 38.7k / 39.4k | 23.2k / 23.9k | 13 → **0** | 2.9 → 3.1 | 0.78 s, 90 cells → **0.60 s**, snapshot + 10 cells | – |
| 1,000 | 39.6k / 40.2k | 22.8k / 23.6k | 28 → **0** | 5.9 → 5.7 | 1.2 s, 181 cells → **0.65 s**, snapshot + 34 | 1.0 MB / 1.0 MB → **0.14 MB / 0.14 MB** |
| 2,500 | 39.8k / 40.6k | 22.9k / 23.6k | 135 → **0** | 17.1 → 15.9 | 3.9 s, 453 cells → **0.68 s**, snapshot + 19 | 3.7 / 9.7 MB → **0.19 / 0.70 MB** |
| 5,000 | 40.0k / 40.9k (`max_stalls=10⁶`) | 22.8k / 23.5k | 237 → **0** | 42.9 → 42.4 | 9.8 s, 918 cells → **0.69 s**, snapshot + 9 | 8.7 / 44.8 MB → **0.30 / 1.93 MB** |

- **Target 4 met (B1).** No call of any kind is above the 24,000-token threshold, the summary call included: the compaction
  trigger is the usage-anchored meter plus the summary request's own prompt, and the summary request is bounded at 95% of
  the threshold. The baseline ran every segment to 1.23–1.28× the 32k window.
- **Target 8 met (B4).** Restore time is flat (0.5–0.7 s at every size): the last snapshot plus the cells since it. At 2,000
  cells in a dedicated run (`scratchpad/f/ckpt2000.py`, Docker) every probe restored from a snapshot in 0.48–0.57 s.
- **Target 9 met (B5).** A checkpoint holds one context window, the journal since the last snapshot, and an offset into the
  streamed `calls.<actor>.jsonl`. The 2,000-cell run's checkpoints were 71–145 kB at cells 500, 1,000, 1,500 and 2,000
  (study 03 measured 16.9 MB per save at 1,914 steps).
- **Wall time** is unchanged within noise, ledger and snapshots included.

## 3. Faults (1,000 rung, faithful, attempt 1)

| Fault | Baseline | M16 harness |
|---|---|---|
| Root calls 40 and 41 raise `ContextOverflow` (41 is the summary call it triggers) | – | compacted, the summary hollowed and retried; 1 overflow retry, no turn charged; reward 1.0, train-valid |
| Silent overflow at root call 40 (empty reply, `finish=length`, window-sized prompt) | – | discarded (not a trajectory call), compacted, retried; 1.0, train-valid |
| First compaction call overflows | a1 masked `agent_run_error` (B2) | the attempt is recorded (not trained), the view hollowed, the retry accepted; 1.0 in a1, train-valid |
| Kernel killed mid-cell at World call 300 | a1 masked `agent_run_error` ("Connection lost"), no restore | **restored in the episode** (snapshot + suffix), the model told the World actions still happened; 1.0 in a1, train-valid |
| Worker lost at model call 100, checkpoint every 500 steps | a2 resumed and was masked `capture_incomplete` (B3) | a2 resumed (a `resume` segment, whole first prompt) and is **train-valid**; 1.0 |
| `--enforce-window` (the engine refuses a prompt over 32k) | step 274 overflowed, all 3 attempts masked | faithful 1.0 and forgetful 0.80, scored, 0 refusals: no prompt reached the window |
| Pathological regex in a cell; the same regex through `history.search`; a 50 MB print | – (unit tests only) | the cell stopped at its deadline and the kernel restored; the search bounded by the ledger; the print capped by the kernel, the full copy in the ledger; 1.0 |

- **Targets 5, 6 and 7 met.** Every fault episode is scored and train-valid; the kill is handled inside the episode.

## 4. Long-context ladder through `CONTEXT` (M16 RLM harness, Docker)

| Size | oolong | MRCR | docqa | `CONTEXT.read` calls | Largest root prompt (Qwen tokens) |
|---|---|---|---|---|---|
| 8k | 1.0 | 1.0 | 1.0 | 2 | 1.5k–1.7k |
| 32k | 1.0 | 1.0 | 1.0 | 3 | 1.4k–2.1k |
| 128k | 1.0 | 1.0 | 1.0 | 9 | 1.4k–2.3k |
| 512k | 1.0 | 1.0 | 1.0 | 33 | 1.4k–2.3k |

- The solver reads the document through the kernel's `CONTEXT` (host requests served from the task's host file, journaled
  as small records), computes in the kernel and submits through the World: 2 World steps at every size, against 3–34
  `docs.read` World calls in study 05. The gold answer scores 1.0 and the no-op 0 at every rung (controls OK).

## Reproduce

```
# from crucible/ with PYTHONPATH=src:packs/_toy/src (Docker running; the Qwen2.5 tokenizer in the local HF cache)
python scripts/lh_validate.py ladder  --out docs/research/long-horizon-rlm/validation/v_main.json     # 100..5,000 x 4 variants
python scripts/lh_validate.py window  --rungs 1000 --variants faithful,forgetful --out .../v_window.json
python scripts/lh_validate.py faults  --out .../v_faults.json      # --cases overflow_root,silent_root,overflow_summary,kill_mid_cell,crash_resume,cells
python scripts/lh_validate.py longctx --out .../v_longctx.json     # --sizes 8000,32000,128000,512000
```

Each step-ladder row in `v_main.json` holds its Φ per segment, constraint examples, segment token counts (`source: tito`),
counters, the B1 peaks and the harness threshold, checkpoint sizes, and the restore probe (`policy.restore_probe`).

## Open

- The scripted models are stand-ins; a real model (Qwen2.5-0.5B in the Docker kernel) was not rerun here.
- Child agents and the tree budget are covered by unit tests (`test_m16_f_agent.py`), not by a ladder rung.
- The ledger is append-only: after a resume, events the crashed attempt wrote past the checkpoint stay in an earlier window
  (the `resume` window's meta counts them as `abandoned_events`).
