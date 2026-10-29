# M4 `signals` — expert signals

**Status: built; audited 2026-10-12 (`docs/audit/2026-10-12/M04-signals.md`).**

- **Code:**
  - `src/crucible/signals/` (`taxonomy`, `extract`, `session`, `attribute`, `reliability`, `pivots`, `gate`)
  - `packs/coding/src/crucible_pack_coding/signals.py` (`CodingSignalMapper`, `AgentRunSignalMapper`, `CIRunSignalMapper`), `outcomes.py`, `m4/`
  - `packs/_toy/src/crucible_pack_toy/signals.py`, `outcomes.py`, `m4/`
- **Tests:** `tests/unit/test_m4_audit.py`, `tests/unit/test_m4_audit_packs.py`, `tests/unit/test_probes.py`.
- **Gate reports:** `reports/m4_gate_coding_dev.json`, `reports/m4_gate_toy_test.json`.

## What it is
A trace's raw events become typed expert signals (`taxonomy.py`: outcome, correction, rejection, approval, rework,
revert, escalation, violation, sla_breach, interrupt, suggestion_accepted, check_result, target). Each signal points at
the step it grades. Packs never add types: they map their own events onto these.

1. **Extraction** (`extract`): the pack's `signal_mapper` for the trace's family (longest `/`-bounded prefix), plus the
   generic session rules (`session.py`) on every trace, then the pack's `outcome_labeler`, then decision points.
   Output is validated (duplicate ids, unknown grades and polarity against the taxonomy raise and name the mapper;
   grades after the signal are dropped and flagged). Mapper output replaces mapper output by id; signals another
   producer added (`meta.by`) stay. In `crucible label` a trace whose extraction raised is reported
   (`extract_failed`) and kept in `traces.jsonl`, but yields no SFT, KTO or DPO row: with no signals nothing in it was
   graded wrong, so a corrected step became a positive target (M4 audit 2026-10-27, R-2).
2. **Ids** are `(trace, type, the source event's stable key, extra)`. The key (`core.schema.trace.event_key`) is the
   source's row id (a GitHub comment, review, check-run id or commit sha; a ticket-log row), else the event's time,
   type, actor and text. A re-ingest that adds an earlier row changes no id.
3. **Attribution** (`attribute`): packs grade what they know (blame inside a PR); the generic rule fills the rest with
   the last event before the signal that wrote its object. `file:` objects are paths (`/` hierarchy, normalised,
   absolute vs relative by suffix); record objects use `.` fields (`ticket:T7.status`); the App must match. A
   rejection, violation or approval never grades its own author's write. An ungraded step signal says why.
4. **Session signals** (`session.py`): an interrupt grades the agent's last action; a rejected tool use grades the call
   it answers (it never ran, so it wrote nothing); mini-swe-agent's rejection message is a user turn of type
   `rejection`.
5. **Reliability** (`reliability.py`): a verdict per signal from later evidence its mapping rule did not use (a revert
   or hotfix of the graded step, the fix itself being reverted, the same check later, the trace ending well without
   the request). Posterior mean of Beta(prior + confirmed, prior + contradicted) per (family, type, human or automated
   source), with a 90% interval. Only judgeable signals get a value.
6. **Decision points** (`pivots.py`): steps graded wrong and then corrected; `studio.datasets` pairs only corrections
   and applied suggestions (a revert says what was wrong, not what was right), once per turn, with the signal's
   reliability and a `state_shift` flag. `graded_wrong` is wider: every step a negative step signal grades (a rejection,
   escalation, violation, rework, interrupt or failing check too) that the trace does not contradict. `crucible label`
   never makes such a policy turn a positive target: no SFT row, KTO False, never DPO chosen. Only corrective signals
   counted, so the toy pack trained a reply the customer rejected and one rejected then escalated (M4 audit 2026-10-27,
   R-1).
7. **The gate** (`gate.py`): labels name events by stable key; scoring fails closed; at least 50 events and 10
   outcomes; Wilson intervals, per-type recall, outcome coverage and per-actor-class accuracy are reported.

### Pack mappers

| Family | Mapper | Signals |
|---|---|---|
| `coding/pull_request` | `CodingSignalMapper` | root review comments by intent (request / clarify / praise) and what happened to the commented lines; reviews; CI check runs (flaky: both outcomes for one job on one commit; a run that timed out or never started is polarity 0 with `meta.inconclusive`, as the CI-log path masks the runner's time limit: M4 audit 2026-10-27, R-3); reverts and hotfixes after the merge |
| `coding/agent_run` | `AgentRunSignalMapper` | one `check_result` per failing hidden gate, +1 when all pass; nothing for a contaminated green run or an ungraded one |
| `coding/ci_run` | `CIRunSignalMapper` | one `check_result` per test; infra-failed or cut-off runs not scored; flaky tests polarity 0 |
| `toy/support_ticket` | `ToyTicketSignalMapper` | policy flags, reversals, supervisor edits, reopens, escalations, SLA breaches, approvals, customer replies, CSAT as the outcome |

## Exit gate

| Pack | Command | Result (2026-10-12) |
|---|---|---|
| coding | `python -m crucible_pack_coding.m4.gate dev --report` (rebuilds 28 PCL PR traces from the API cache, offline) | type 1.0, attribution precision 1.0 (32 events), outcome 1.0 (14), passed; not `confident` |
| toy | `python -m crucible_pack_toy.m4.gate test --report` | type 1.0, attribution precision 1.0 (17 events), outcome 1.0 (12), passed; not `confident` |

Both label sets were written by the builder. The toy labels were written before the mapper, but this audit's toy
fixes were made with the label errors visible; the coding dev labels are a tuning split. Held-out coding labels and a
second labeller are the open part of the gate.

## Not done
- CI reruns and per-commit CI from the API (`filter=all` over every PR commit needs a new fetch).
- A review verdict grades the reviewed head commit, not the commits it is about.
- A judge behind the comment-intent rules (non-English comments are abstentions today).
- No M6 operator reads `grades` or `reliability` yet (the M6 audit).
- `finance_ap` has no mapper or outcome labeler (no recorded AP workflow data yet; its World episodes get session signals and record their writes).
