# Long horizons — million-token episodes with dense, honest reward

**Status: core built (2026-10-09). The xl gate passes.** DESIGN §10.4.

- **Research:** `docs/research/long-horizon/01-frameworks.md` (NeMo-Gym, verifiers, MiMo/verl, Harbor, agentenv, OpenEnv, Long-Horizon-Terminal-Bench).
- **Code:**
  - `src/crucible/verify/progress.py`;
  - `src/crucible/runtime/checkpoint.py`;
  - `src/crucible/actors/policy.py` (`ContextConfig`, `state_dict`, `observers`/`after_step` in `run_episode`);
  - `src/crucible/world/world.py` (`stream_trace`, `read_trace`);
  - `src/crucible/core/schema/rollout.py` (delta tokens, segments);
  - `src/crucible/runtime/record.py` (`credit_from_progress`);
  - `src/crucible/runtime/inject.py` (`ReplayLLM`);
  - toy pack `longhorizon/` (`world.py`, `backlog_tools.py`, `task.py`, `gate.py`).
- **Tests:** `tests/unit/test_longhorizon.py` (7 tests). **Report:** `reports/longhorizon_xl.json`.

## Gate result (xl)

| Measure | Value |
|---|---|
| Tickets / steps / model calls | 1,200 / 7,623 / 7,627 |
| Trajectory tokens (every token that entered the context, replies by completion tokens) | **1,114,725** |
| Context budget / compactions / largest context before a compaction | 128,000 / 4 / 102,853 |
| Crash on attempt 1 → resume | from the step-3000 checkpoint |
| Reward | 1.0 |
| Dense progress | Φ₀ 0, Φ_T 1.0, Σ ΔΦ = 1.0000, drift none; per-call credit sums to 1.0 |
| Wall time | 66 s (API-free) |

## What the backlog world tests

| Long-horizon property | How |
|---|---|
| Revisit after hundreds of steps | `awaiting` tickets: act only after the customer's reply, which fires hours of simulated time later |
| A rule that changes mid-episode | the approval threshold drops 50 → 25 USD at `amend_at`; refunds are judged by the rule in force when issued |
| Long-range consistency | `repeat` tickets about an order an earlier ticket refunds (or one refunded before the shift) |
| Bulk context | per-ticket customer threads; 14 filler knowledge-base documents |
| Honest approvals | a recorded approver needs an in-episode approval of that amount (B.15) |

## Remaining
- Observation offloading and a memory app (never credited by Φ).
- Compaction mode varied per GRPO group.
- `calls.jsonl` streaming; per-segment ATIF export.
- Staged checkpoint milestones and a fast-tier feedback loop.
- A real-model run (budgeted).
- Cross-segment GAE (M12).
