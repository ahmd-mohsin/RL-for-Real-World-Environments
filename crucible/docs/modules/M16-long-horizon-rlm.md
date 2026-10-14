# M16: the very-long-horizon RLM harness

**Status: built 2026-10-13.** DESIGN §10.5. It replaces the m14 RLM loop (`docs/modules/M14-rlm.md` keeps the kernel
security model, which is unchanged).

The harness runs World episodes of thousands of steps and millions of tokens. The model works in a persistent Python
kernel, loses nothing at a compaction, bounds every cost, and produces training records that stay correct across
compactions, sub-calls, child agents and resumes.

**Research:**
- `docs/research/long-horizon-rlm/01-frameworks.md`: 17 frameworks and how they manage agent context on long tasks.
- `02-papers-articles.md`: papers, articles, the reward rules for long tasks.
- `03-crucible-stress.md`: the m14 harness stress-tested to 7,161 steps (break points B1–B10).
- `04-design.md`: the design, interfaces and owners.
- `05-baseline.md`: the ladders and the m14 baselines.
- `06-validation.md`: the M16 harness on the same ladders.
- `07-campaign.md`: child agents, a real model, a 10,000-call probe, and the red-team hunt.

## Three tiers of state

| Tier | Where | What |
|---|---|---|
| L1 prompt | the model's context | system, the task verbatim, the summary, a python-state row, a boundary marker, the recent tail (~20k tokens) |
| L2 kernel | the Docker kernel's variables | `TASK`, `TOOLS`, `OBS`, `INBOX`, `CONTEXT`, the model's own variables; they survive compaction because the kernel keeps running |
| L3 ledger | an append-only log on the host, outside the kernel | every message, the **full** output of every cell, every World result (stored once), sub-calls, summaries, child events; read from the kernel with `history` and `OUT(i, cursor)` |

## Code: `src/crucible/actors/rlm/`

| File | What |
|---|---|
| `agent.py` | `RLMConfig` (the harness id), `AgentLoop` (one model loop over one kernel, ledger, meter and stall detector; the root and every child), `Tree` (state shared by an episode's agents), `CallLog` (model calls streamed to `calls.<actor>.jsonl`), `ContextFile` (the `CONTEXT` handle). |
| `policy.py` | `RLMPolicy`: the root `AgentLoop` plus the World `act()` bridge; checkpoint and resume. |
| `ledger.py` | `Ledger`: JSONL events, an O(1) index sidecar, content-addressed blobs, bounded search (2 s, 64 MiB; regex in a killable worker), windows, child ledgers, crash recovery, `HISTORY_API` (the kernel half), journal markers that replay a history reply exactly. |
| `context.py` | `ContextSpec` (threshold `max(W/2, min(0.95·W, W − max_out − reserve))`, ≤ `seq_len − max_out`), `ContextMeter` (usage-anchored), overflow detection (9 markers, silent overflow), `summary_view`, `hollow_middle`, `split_tail`, the summary schema and update mode, `rebuild`, `Compactor` (≤ 5 attempts), the footer. |
| `stall.py` | `StallDetector`: windowed, decaying; tells waiting from looping; notice → reflection turn → scored stop `stalled`. Uses only what the policy observes. |
| `tree.py` | `TreeLimits`, `TreeSupervisor` (tree-wide budgets, per-depth reserved concurrency, no nesting deadlock), `ChildRegistry` (persistent children: spawn, result, send, steer, cancel cascading, watch, bounded event inbox), `AGENT_API`. |
| `kernel.py`, `repl.py` | The kernel: dill snapshots streamed in signed chunks (the host never unpickles), restore from snapshot + journal suffix with full and lenient fallbacks, a bounded per-cell fingerprint, clock and `os.urandom` replay, `describe()`, `JournalReply` / `resolve` for journal markers, `lost_between_cells`, a renewable container lifetime. |

Elsewhere: `runtime/harness/rlm.py` (`RLMHarness`, per-attempt ledger and call stream, `CONTEXT` mount, evidence file),
`runtime/record.py` and `core/schema/rollout.py` (per-agent chains, typed edges), `train/trainer.py` (`max_seq_len`,
per-chain critic), `train/bridges/common.py`, `train/credit.py`, `studio/datasets.py` (chain keys).

## The loop (per turn)

1. Drain child events, parent messages and INBOX.
2. Pre-send check: usage-anchored tokens + the summary request's own size ≥ threshold, or `await rlm.compact()` asked →
   compact.
3. Call the model. A raised or silent overflow discards the reply, compacts and retries without charging a turn. With
   compaction off it is the scored stop `context_overflow`.
4. Anchor the meter to the call's usage; charge the tree.
5. Run the cell. The full output goes to the ledger; the context gets a head + tail view (8,192 chars) that names
   `OUT(i, cursor)`.
6. Footer from 80% of the threshold; a wrap-up notice at 0.9 of each budget; stall check.

**Compaction:** the `Compactor` writes the summary from a bounded view (tool outputs head + tail, old images and reasoning
pruned, hollowed to fit). It retries up to 5 times. All failing is the clean stop `compaction_failed` (truncation). The
new context is system + task verbatim + summary (update mode over the previous one) + python-state row (names, types,
sizes, lost names) + a boundary marker naming the history API + the recent tail, cut only before an assistant row. Each
compaction opens a ledger window, takes a kernel snapshot and opens a token segment.

**Kernel names:** `world.call` / `world.send`, `rlm.query` / `rlm.batch`, `rlm.finish(answer)`, `rlm.give_up(reason,
evidence)` (stop `agent_gave_up`; scoring is the task's), `await rlm.compact(instructions)`, `await rlm.budget()`,
`history.*`, `OUT(i, cursor)`, `CONTEXT.info()` / `await CONTEXT.read(offset, n)` / `await CONTEXT.search(pattern)`,
`rlm.agent.*`.

**Stops (typed):**

| Stop | Treated as |
|---|---|
| `agent_done`, `agent_gave_up` | the outcome, scored |
| `stalled` | scored |
| `context_overflow` | scored (only with compaction off) |
| `compaction_failed`, `too_many_compactions`, `budget:<kind>` | truncation |
| `policy_error` | masked with the fault's infra kind |

## Child agents

- `await rlm.agent.spawn(brief, name=None, persistent=True, world="none")`. Each child has its own kernel (same isolation
  rule), its own ledger directory and a fresh context from the brief.
- `world="none"`: `world.call` is refused (least privilege). `world="shared"`: World calls go through the root's action
  queue under one lock; each is a World step attributed to the child's model call.
- Budgets are tree-wide (`tree_*` fields): new tokens, turns, spawns, depth ≤ 3 (default 1), concurrency with slots
  reserved for deeper levels, a hard wall clock and turn cap per child, sub-calls. Child usage never counts against the
  parent's context.
- Running children are not resumed after a checkpoint; they are reported `failed: lost at resume`.

## Training records

- `model_calls` carry `agent` (`""` root, a child path), `depth`, `edge` (`continuation`, `compaction_attempt`,
  `compaction`, `subagent_call`, `subagent_return`, `agent_message`, `subagent_cancel`, `sub`), `accepted`, `trainable`,
  `window`, `ledger_index`.
- A sequence chain is `(actor, agent, segment)`. Sub-calls and summary attempts are nodes: their own sequences that never
  break or advance a chain. Children are their own chains and share the advantage of the root turn that spawned them.
- A summary is trained only when accepted. Calls answered by another model are recorded, never trained.
- A resume opens a new segment on every agent; its first call stores the whole prompt, so a resumed rollout is
  train-valid.
- Weighting is per logical rollout: 1, 2 or 8 segments give every trainable token the same weight (tested under every
  `loss_agg` mode). An over-length row (`TrainerConfig.max_seq_len`) is dropped whole and counted, never tail-cut.
- **B7:** the m14 records forwarded 6.98× the entered tokens (19.73× with 256 sub-calls); now 1.000×.

## Checkpoint

`state_dict` holds one context window, counters, the meter / stall / supervisor / registry state, the ledger path and
length, the last snapshot, the journal since it, and the offset into `calls.<actor>.jsonl`. History, `CONTEXT` and World
replies in the journal are small markers rebuilt from the ledger on replay.

## Validation (32k window, Docker kernel, real Qwen2.5 tokens, default config)

| Measure | m14 harness | M16 |
|---|---|---|
| faithful scripted model, 5,000 World calls | 0.227 (stall counter) | 1.0 |
| forgetful summary + history re-reads, 5,000 | 0.935 (with stalls lifted) | 1.0 |
| forgetful summary (must keep losing) | 0.75–0.91 | 0.73–0.85 |
| calls above the window, 5,000 | 237 | 0 |
| kernel restore, 5,000 | 9.8 s (918 cells replayed) | 0.69 s (snapshot + 9 cells) |
| checkpoint, last / all saves, 5,000 | 8.7 / 44.8 MB | 0.30 / 1.93 MB |
| faults: overflow (root, silent, summary), kernel kill mid-cell, crash + resume, enforced window | masked | scored and train-valid |
| long-context controls through `CONTEXT`, 8k–512k | – | solver 1.0, gold 1.0, no-op 0 |
| 10,000 World calls (180 compactions) | – | 1.0; 0 calls above the window |
| child agents, depth 1–2, shared and no-World, up to 893 steps (36 runs incl. the real model) | – | 1.0 for every scripted tree; train-valid, every World step attributed once, token ratio 1.0, 0 containers left |
| Qwen2.5-0.5B (real model, Docker kernel) | ran 0 cells | ran 0 cells (the model calls tools by name); corrected, scored, train-valid; its summaries accepted |

**Hunt** (`07-campaign.md` §4): 12 defects found in the integrated harness and fixed. Among them, a policy could mask its own episode through a history read or an unloadable snapshot, a no-World child could grant the World to a grandchild, a stray task's World call was credited to the next cell, the ledger had no disk cap, and a resume showed abandoned events.

**Tests:** `tests/unit/test_m16_*.py`.

## Remaining

- A real model larger than 0.5B on long rungs (needs approval for GPU or paid runs).
- Children resumed mid-flight; steer that interrupts a running cell.
- An ablation of the tool index after the task (`tool_index`), which may invite by-name calls from small models.
- Compaction mode rotated per GRPO group (the mode is in the harness id; rotation is the trainer's).
- The task-creation method for long tasks (next, from the user).
