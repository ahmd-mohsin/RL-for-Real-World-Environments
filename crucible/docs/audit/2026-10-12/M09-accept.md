# M9 accept: audit (2026-10-12)

Scope: `src/crucible/accept/` (model, policy, key, cache, plan, run, contract, repair, checks, mutation, registry), the
user paths that report acceptance (`cli/studio.py` `accept`, `studio/build.py`), flaky-task admission
(`train/admission.py`), and the toy acceptance adapter (`packs/_toy/.../forge/task.py`).

Checked against ARCHITECTURE §M9, DESIGN §9 and `docs/modules/M09-accept.md`, with the M0 protocol. The framework
comparison is `M08-M10-study-sweep.md` (M9 rows); the defect hunt is `M09-study-defects.md` (27 ledger rows, V1-V19).
16 regression tests (`tests/unit/test_m9_audit.py`) all fail on the M8 audit commit (`2bf33e9`) and pass after the fixes.

## 1. Requirement ledger (rows that were wrong, missing or partial)

| # | Requirement | Before | After |
|---|---|---|---|
| L1 | A task enters the pool only through the full contract | **wrong**: `crucible accept` and `build` ran the M8 subset (A4, A6, A9-A13 skipped) and called it accepted; `build` shipped 9 toy tasks the full contract rejects | both run `accept_task`; `accept.run.verdict` gives accepted / provisional / rejected; `--docker`/`--paid` declare the needs; provisional tasks are exported labelled, never as accepted |
| L7 | A5 agent-image leak audit | **wrong**: read `rep.hits`, which does not exist, so it never failed | reads `findings`; whitespace-normalized text matches |
| L3 | A3 mutation score | passed with zero mutants; a dropped approval message (B.15) was "equivalent" | zero scored mutants fails; only `drop_read` may be equivalent; equivalent share reported; truncation stratified by family; generated wording-only alternatives (M8) run in A3 too |
| L4-L6 | A4/A9 detectors on real adapters | failed open: the Haar and stale-image detectors needed capabilities no real adapter had | a task fact (`reference_build`, `perf_bars`, `fresh_gates`) requires its capability; an image with no pinned digest fails A9 |
| L8 | Acceptance key | hashed `verify/runtime` only; A10 changed twice and stayed `A10@1` | the kit hash covers `accept`, `verify`, the forge's learnability/ladder/hacker/specialists and `world/leaks`; changed checks also bumped |
| L10 | Red team (A6, A13) | an attack or tamper without a control passed; an all-skip audit passed | no control is a failure; the audit must pass at least one item |
| L11 | N/A only from task facts | `attack_surface="none"` and a missing property audit gave N/A | a World/container task cannot claim no surface; a missing property audit fails unless the fact `no_free_form_submission` holds |
| L12 | A10 calibration | stopped sampling when the estimate entered the band (biased to accept); 1.3 counted as a failure; bench band on the point estimate; counted runs of older task versions | fixed n; success is `reward >= 1`; the bench band is the ladder's (upper bound ≤ 0.2); runs count only if bound to the task's current grading environment |
| L13 | A12 tiers | errored when A1/A3 came from the cache | grades through the store itself |
| L14 | Repair cannot weaken a task | verified A1, A7, A16 only; a dropped bound check was committed | verifies A1, A2, A3, A7, A13, A16 |
| L16 | Backfill quarantine | kept a task whose detector errored; nothing read the quarantine | anything but pass/N/A is quarantined and recorded (`quarantine.jsonl`, `quarantined()`) |
| L17 | Cache and grade store | non-atomic writes; a torn file crashed every later run; `Path` solutions keyed by repr | atomic writes; a torn entry is a miss; solutions keyed by content |
| L19 | Admission (A4 rule) | a 0.5 reference was "stable"; a dict verdict aborted the screen; the flake bound assumed k verdicts | full reward to pass; dict verdicts frozen; the bound uses the fewest verdicts a kept task had |
| L20 | Overrides | S1 and A14-A18 overridable; nothing bound to the key | not overridable; an override must name its author and date and is bound to the key |
| L24 | Imports I0-I10 | **missing** (ARCHITECTURE said they existed) | still missing; ARCHITECTURE corrected (they belong in `interop`, above `accept`) |
| L22 | Seed adapters for tasks 01-04, patch mutation operators | missing | still missing |

## 2. Defects and fixes

| Id | Sev | Fix (files) |
|---|---|---|
| V1 | P0 | `accept/checks/core.py` (A5 `findings`) |
| V2 | P0 | `accept/run.py` (`verdict`), `cli/studio.py` (`accept`, `build --docker/--paid`), `studio/build.py`; legacy A1/A2 refuse masked grades (landed with M8) |
| V3, V18 | P1, P2 | `accept/checks/core.py` (A3), `accept/mutation/actions.py` (`drop_read`, stratified), registry fixtures carry a real mutant |
| V4 | P1 | `accept/repair.py` |
| V5 | P1 | `accept/key.py` (`DECIDING`), versions bumped (A1-A10, A12, A13, A16) |
| V6 | P1 | `accept/checks/core.py` (facts → capabilities), `accept/checks/extended.py` (A9 pinned digests) |
| V7 | P1 | `accept/registry/run.py` |
| V9, V10 | P1 | `accept/checks/extended.py` (A6, A13), `accept/checks/redteam.py` (A16) |
| V11 | P2 | `accept/checks/extended.py` (A12) |
| V8, V12 | P2 | `accept/checks/base.py` (`_content`, `_atomic_write`), `accept/cache.py` |
| V13 | P2 | `train/admission.py` |
| V14 | P2 | `accept/checks/base.py` (`InfraUnavailable.infra`), A1 |
| V15 | P2 | A10; toy `task.py` (`calibration_runs` bound by grading environment, `calibrate` writes the binding) |
| V16 | P2 | `accept/model.py` |
| V17 | P2 | not fixed (no `original()` floor; other reward channels unchecked) |
| V19 | P2 | not fixed (A11 does not inject the dirty-workspace and ignored-build cases) |

## 3. Gates

- Toy full contract (`python -m crucible_pack_toy.accept_gate --no-paid`): every check but A10 passes on 11/11 tasks,
  including the stricter A3 (mutation score 1.0, no equivalents), A6, A13 (with controls) and A16. A10 is **pending**
  on all 11: the recorded calibration runs are not bound to the tasks' current grading environment (this audit changed
  the toy adapter), and new runs are paid. Before the audit the gate reported 2/11 accepted, 9 rejected by A10 on runs
  of earlier task versions. Power tests: 13 proven, 10 per kit.
- `crucible build toy` (no Docker, no paid calls): 5 tasks, 0 rejected; each is accepted or provisional and exported
  labelled.
- `crucible.accept` over the full suite: **91%** (85% in the hunt; `contract.py` 34% -> 90%, `cache.py` 100%,
  `checks/core.py` 92%, `repair.py` 81%).
- Suite (`tests/unit`, `tests/toy_e2e`): **1032 passed, 3 skipped** (1016 before, plus the 16 new tests).
  `lint-imports`: 4 kept; `ruff` clean.

## 4. Picks taken and queued

Built: the full contract on every user path; zero-mutant and equivalence rules (SWE-smith/Repo2RLEnv); finite floor
rewards (NeMo-Gym fixtures); one bench band and success by the task's predicate (SkyRL/prime-rl); atomic,
content-keyed stores (Repo2RLEnv); quarantine on error (MiMo/NeMo-Gym); repair commits after the full set (Prime Agent).
Queued: imports over the interop task adapters (M), Karotte's credential scan in A5, A19 snapshot round trip, Harbor's
quality checker as advisory A7b, A11's missing injections, seed adapters for tasks 01-04 with patch mutation operators.

## 5. Open (needs a decision or resources)

- A10 re-calibration of the 11 toy tasks: paid frontier runs (the earlier run cost $0.0385 for k = 2; fixed n = 8 costs
  about 4x that). Not run without approval.
- Imports I0-I10 and the seed (GPU) gates.
