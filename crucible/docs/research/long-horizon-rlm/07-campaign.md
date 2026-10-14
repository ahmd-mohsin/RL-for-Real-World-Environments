# Long-horizon RLM harness, study 07: child agents, a real model, a 10,000-call probe, and the hunt

- **Run:** 2026-10-13, CPU-only Mac (Qwen on Apple MPS), local Docker kernel, no paid API, no GPU cluster.
- **Raw reports:** `campaign/c1_children.json`, `c1_children_w12k.json`, `c1_adversarial.json`, `c2_qwen05.json`,
  `c2_qwen05_compact.json`, `c3_scale.json`. Script: `scripts/lh_campaign.py`. Scripted tree models:
  `src/crucible/runtime/scripted_rlm_tree.py` (no hidden memory, like study 05's).
- **Invariants checked on all 36 runs of campaigns 1 and 2: 0 violations.**
  - Train-valid through `runtime.record.build`, and never masked.
  - Every World step is attributed to exactly one model call.
  - Φ telescopes on the root chain.
  - Forwarded tokens = entered tokens + node prompts (ratio 1.0).
  - 0 containers left.

## 1. Child agents on the backlog ladder (32k window; and 12k so children compact)

| Shape | Rungs (steps) | Runs | Reward | Compactions root / children (12k) | Tree use |
|---|---|---|---|---|---|
| depth 1, 4 workers, `world="shared"` (children step the World) | 91 / 468 / 892 | 6 + 6 | 1.0 all | up to 15 / 23 | 4 spawns, peak 4 running |
| depth 1, 4 analysts, `world="none"` (root executes) | 91 / 467 / 893 | 6 + 6 | 1.0 all | up to 21 / 4 | 4 spawns |
| depth 2, 2 leads × 2 workers | 91 / 468 / 892 | 6 + 6 | 1.0 all | up to 14 / 22 | 6 spawns, peak 6 |
| adversarial: over-spawn, exceed depth, finish with children running | 1 | 2 | 0 (did no work) | 0 | spawns capped at 64, 17–18 refused, peak ≤ 8 concurrent, every child cleaned up |

- Shared-World children's steps are credited to the child calls that made them. In a 1,000-step run, 217 steps came
  from the root and the rest from 4 workers.
- Lost constraints after compaction: 0.

## 2. A real model: Qwen2.5-0.5B-Instruct in the Docker kernel (13 + 1 runs)

| Task | Runs | Reward | Cells run | By-name corrections | Notes |
|---|---|---|---|---|---|
| backlog 100 | 5 | 0 | 0 | 0–29 per run | emits `<tool_call>` naming World tools (14 in one reply), then echoes the task |
| backlog 500, window 8k | 2 + 1 | 0 | 0 | 4–13 | the 8k run compacted twice; **both summaries accepted**; largest prompt 5,666 vs a 6,488 threshold; ended `budget:turns` (truncation) |
| long-context docqa 8k / 32k via `CONTEXT` | 6 | 0 | 0–1 | 0–2 | never reads `CONTEXT` |

- **The model's failure, not the harness's.** The harness gave a correction for every by-name call. It never masked an
  episode. Every record was train-valid. With a real model, compaction ran and the summaries were accepted.
- **Follow-up.** A larger model is needed to measure task skill; that needs approval for GPU or paid runs. One harness
  choice is worth an ablation: the one-line tool index after the task (`tool_index=True`) may invite by-name calls from
  small models.

## 3. Scale probe: 10,000 World calls (1,640 tickets, faithful scripted model, default config, 32k window)

| World calls | Reward | Compactions | Largest prompt (turn / summary) vs threshold 24,000 | Calls above the window | Wall |
|---|---|---|---|---|---|
| 9,906 | 1.0 | 180 | 22,604 / 23,435 | 0 | 190 s (19 ms per World call) |

This run took no checkpoints (none configured). Checkpoint size at 10k is not measured here; study 06 measured 0.30 MB
per save at 5,000.

## 4. The hunt (red team on the integrated harness): 12 defects, all fixed

Each fix has a test in `tests/unit/test_m16_hunt.py` that fails without it. H11 and H12 were re-checked by disabling
the fix.

| Id | Defect | Fix |
|---|---|---|
| H1 | A `world.call` with 60k CJK characters in its arguments, then `history.get` of it, masked the episode (`LedgerError` → infra) | the ledger cuts structured fields as capped JSON; request-shaped faults are `ValueError` |
| H2 | A `world="none"` child could spawn a `world="shared"` grandchild that stepped the World | a child can only grant what it holds |
| H3 | A planted object whose `__reduce__` fails on load made every restore fail, which masked the episode | restart from the bootstrap and tell the model that every variable was lost; the episode stays scored |
| H4 | A `CONTEXT.search` stalled the host's event loop for 2 s; replay re-ran it with a different result | a search runs in a thread; the journal records where the live search stopped |
| H6 | The host path appeared in restore notices readable through `history.get` | replaced with `<ledger>` |
| H7 | A checkpoint taken mid-compaction trained a summary that the episode never continued from | on resume, that summary is untrained and marked `abandoned` |
| H8 | A resumed in-flight cell that took another path raised `KernelDivergence` and masked the episode | restore to before the cell and notify the model |
| H9 | The ledger had no disk cap | `RLMConfig.ledger_max_bytes` (2 GiB, tree-wide); World results are never cut |
| H10 | After a resume, `history.search` showed events from the abandoned future | the checkpoint keeps `ledger.lengths()`; resume runs `rewind_to` before replay |
| H11 | A task that outlived its cell had its World call served during the next cell and credited to that cell's call | requests from an earlier cell are refused (`stray_requests`) |
| H12 | `rlm.compact()` every turn bought one summary call and one snapshot of up to 256 MiB per turn | refused under 2 turns; a snapshot only when the compaction freed context |
| H13 | A test asserted that no RLM container existed on the machine, so other processes made it fail | filter on this process's owner label |

**No defect found in:**
- about 3,000 fuzzed `history.*`, `agent.*` and `context.*` requests;
- calls made outside a cell;
- spawn races;
- child clean-up;
- `give_up` scoring;
- truncation flags;
- real tree rollouts through `record.build` (H5).
