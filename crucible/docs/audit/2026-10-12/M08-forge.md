# M8 forge: audit (2026-10-12)

Scope: `src/crucible/forge/` (orchestrator, budget, repo, swarm, economy, curator, dedup, ladder, learnability, compose,
stitch, tighten, hacker, specialists), the M9 subset the forge uses (`accept/contract.py`, A16 in
`accept/checks/redteam.py`), the sandboxed roles (`interop/harbor/role_runner.py`), the LLM client's spend booking, and the
pack generators (`packs/_toy/.../forge/`, `packs/coding/.../forge/derive.py`).

Checked against ARCHITECTURE §M8, DESIGN §7 and `docs/modules/M08-forge.md`, with the M0 protocol. The framework
comparison is the shared M8-M10 sweep (`M08-M10-study-sweep.md` and its five group documents); the defect hunt is
`M08-study-defects.md` (31 ledger rows, V1-V23). 22 regression tests (`tests/unit/test_m8_audit.py`) all fail on the M7
audit commit (`526fdb5`) and pass after the fixes.

## 1. Requirement ledger (the rows that were wrong or partial)

| # | Requirement | Before | After |
|---|---|---|---|
| L2 | Journal reuse by input hash | a stricter validator or a new canary reused an unchecked output; a no-op re-run reset `accepted` | validators (qualified name + version) and the canary flag are in the key; an unchanged re-run keeps `audited`/`accepted`; a torn journal tail no longer makes the journal unreadable |
| L3 | Budgets never half-run, spend accounted exactly | **wrong**: concurrent nodes double-counted ($0.04 booked for $0.02), an over-cap settle aborted every plan, a role-side cap leaked its reservation | spend is booked per asyncio task (`actors/llm.SpendScope`, `book_scoped`; sandboxed trials book theirs), so each node is charged its own calls; a role with no scope falls back to the delta, clamped to the observed total; settle never raises, and an exhausted ledger refuses new work |
| L4 | Retries with feedback, infra never a content attempt | any unexpected role exception aborted all plans | a role error abandons its node only; `run_plans` never lets one plan end another; per-node timeout (default 1800 s) and exponential infra backoff |
| L5, L13 | Asymmetric information; A5 sees hidden material | **wrong**: the full oracle pasted into the instruction passed A5; the ticket data the agent reads was never scanned | toy A5 scans the ticket rows and tool code; hidden strings are the oracle-only values and each serialized oracle action; A5 matches whitespace-normalized text |
| L6 | Zoned repo | **wrong**: an adversary could copy oracle lines and the canary into `public/` | information-flow rule: what is written into zone Z may not come from any zone some reader of Z cannot read; 6-token shingles over the added text (prefixes, splits, rewraps) and short numeric lines; the "already readable" exemption is taken from the parent tree |
| L7 | Sandboxed roles | a forbidden line with quotes passed the JSON-escaped scan; a failed trial cost $0 | raw, JSON-escaped and whitespace-collapsed forms are scanned; a failed trial carries its cost (`RoleTrialFailed`) |
| L9 | A8 grounding | any non-empty string counted | closed grammar: `sig_*` (resolved), `kb:`, `run:`, `trace:`, `reference_differential:`, `defect:`, registry attack/audit ids; `instruction:` checks reported apart (A7 covers them); unknown provenance fails |
| L10, L11 | Spans and ≥ 1 trap per seed pass M9 | **wrong**: trap tasks had empty instructions and gates that required docstring words to be absent; spans carried 8 C2 gates no grader ran | C2 checks are Φ milestones, never gates, kept only if they separate oracle and frontier with comments and docstrings stripped (the docstring words are now dropped); a trap task states the seed's requirements and its gate is the seed grader's behavioural gate; span gates are the suite gates only |
| L14 | A7 two-way coverage | the first 40 characters of a property counted; a check on oracle wording passed | the whole sentence; A3 builds a valid alternative of its own (the oracle with every oracle-only prose argument cut to the words the instruction states), so a check on unstated wording fails A3 |
| L15 | Composition | the fused task's cross check graded no stated property | `fuse`/`nest` take cross properties; their sentences join the instruction; a cross check without one is refused |
| L16 | Stitch | a span that pre-applied a file it owned was accepted | refused |
| L17 | Tighten | one flaky frontier failure made a gate "separating" | the frontier's failure rate must have a Wilson lower bound ≥ 0.2 |
| L18, L19 | Learnability and ladder | a masked rollout counted as a failure | outcomes are `bool | None`: masked draws are excluded from n and re-drawn (up to 2× the last step); an all-masked rung is pending, never "easier" |
| L20 | A16 red team | passed when every attempt was masked; memory replay suppressed the search | error when every attempt (or more than 25%) is masked or an audit is undecided (new label `undecided`); A16 runs the specialist's fallback search when the replay confirms no hack (`propose_more`), so known hacks still cost no episode |
| L21 | Swarm | a worker exception aborted the stage; announcing was cooperative | a failed worker is a charged failed episode; a contribution needs an uncollided announcement; every announcement is cleared at episode end; self-nomination is refused |
| L22 | Economy | `resume` was lost on reload; a re-run wiped M9 verdicts | `resume` is replayed; verdicts, hacks and quota decisions survive an unchanged re-run |
| L23, L24 | Curator, dedup, decontamination | sealing was in memory; L1 was an 80-character prefix; no decontamination | decisions, seals and kept candidates are durable; lineage is transitive and taken from a composite's components; L1 is a word-shingle Jaccard; a decontamination hook holds contaminated bench tasks and notes unchecked ones |

Still **missing** (unchanged): L28 G-transplant; L29 the SWE-smith patch-mutation port; L30 Harbor `compile`/`analyze` and
the 10% human review; L31 frontier calibration (A10) in the subset.

## 2. Defects (reproduced) and fixes

| Id | Sev | Fix (files) |
|---|---|---|
| V1, V2 | P0, P1 | `forge/repo.py`: `protected`, `_marks`, parent-tree exemption, atomic index writes |
| V3 | P1 | `actors/llm.py` (`SpendScope`, `book_scoped`), `forge/budget.py`, `forge/orchestrator.py`, `interop/harbor/role_runner.py` |
| V4 | P1 | `forge/budget.py` (non-raising settle, `exhausted`), `forge/orchestrator.py` (per-node containment, `run_plans`) |
| V5, V6, V18 | P1, P2 | `forge/orchestrator.py` (key, `_fingerprint`, kept judged state) |
| V7 | P1 | `accept/checks/redteam.py`, `forge/hacker.py` (`undecided`), `forge/specialists.py` (`propose_more`), `accept/policy.py` (`a16_max_infra_share`) |
| V8 | P1 | `accept/contract.py` (whole sentence; generated alternatives), toy `task.py` (`generated_alternatives`) |
| V9 | P0 | toy `task.py` (`agent_visible`, `hidden_strings`), `accept/contract.py` (normalized match) |
| V10, V19 | P1, P2 | `interop/harbor/role_runner.py` (`_leaks`, `RoleTrialFailed`) |
| V11 | P1 | `accept/contract.py` (`GROUNDED`, `STATED`); toy `fused.py` cites a document |
| V12, V13 | P1 | coding `derive.py` (`_CodeOnly`, `_drop_docstrings`, `trap_property`, milestones instead of gates, no milestone exemption in the CPU A7) |
| V14 | P1 | `forge/compose.py` (`cross_props`), toy `fused.py` |
| V15 | P1 | `forge/economy.py` |
| V16 | P2 | `forge/swarm.py` (plus sweep G2 D8: worker failures, self-nomination) |
| V17, V21 | P1, P2 | `forge/curator.py`, `forge/dedup.py` |
| V20, V22 | P2 | `forge/stitch.py`, `forge/tighten.py` |
| sweeps | — | masked outcomes (`learnability.py`, `ladder.py`, toy `calibrate.py`); torn journal tail; node timeout and backoff; decontamination hook; contract A1/A2 refuse masked grades (M9 V2, shared code) |

Not fixed: **V23** (the toy builder writes `forge/scenario.json` into the pack root; nothing mounts or exports it today) and
the behavioural part of **V13** (two more oracle grades and a naive floor per span need the lab GPU).

## 3. Coverage, contracts, gates

- `crucible.forge` over the full suite: **96%** (the hunt measured 66% on its targeted set; every forge module is now at
  83% or more: `fuse.py` 83%, the rest 92-100%).
- Suite (`tests/unit`, `tests/toy_e2e`): **1015 passed, 3 skipped**, plus one timing test
  (`test_workers.py::test_throughput_scales_with_worker_processes`) that failed under the suite's load and passes alone
  (5/5). 994 before, plus the 22 new tests.
- Toy world synthesis, fresh build: 5/5 tasks accepted, identical per-check verdicts and task hashes to the recorded
  report; the journal replay reuses every node. The report now shows the generated alternative (passing on all 5), the
  ticket data in A5's scan, and A8 over hidden checks (6/6 grounded, 6 stated checks reported apart).
- Toy hindsight gate: report unchanged.
- Coding derivation (CPU): spans pass; the people span has 10 gate checks (18 before: the 8 C2 gates are now Φ
  milestones); every seed trap states its requirement; C2 candidates that only separated through comments or docstrings
  are dropped (`defines`, `schemas`, `Pure`, `packaging`, `unspecified`, ...). **The behavioural M9 run of the reworked
  trap tasks (seed graders, GPU) has not been run**; until it is, "≥ 1 trap per seed passes M9" is not claimed.
- `lint-imports`: 4 kept; `ruff` clean.

## 4. Comparison with the local frameworks

The picks table is in `M08-M10-study-sweep.md` (M8 rows 1-11). Built here: rows 1-6. Queued: durable spend receipts
(Repo2RLEnv), the SWE-smith bug generator with an execution check, per-prompt pass rates fed back from training,
red-team-until-clean, imported tasks as seeds.

## 5. Open

- The behavioural M9 run of the seed trap tasks and the extra span grades (GPU).
- G-transplant, the SWE-smith port, Harbor `compile`/`analyze`, the human review sample, A10 in the subset.
- V23; wiring the decontamination hook to `bench.contamination.scan` in the bench layer.
