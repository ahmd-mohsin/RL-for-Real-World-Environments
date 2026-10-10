# M9 `accept` — the Acceptance Contract and the defect registry

**Status: core built (2026-10-09); seed (GPU) gates pending.**
- **G1 (toy, full contract):** every check runs on all 11 toy tasks; A1–A9, A11–A13 pass on all 11. **2/11 accepted.** A10 (frontier calibration, `gpt-6-luna`, k = 2, $0.0385 total) puts 7 tasks at p = 1.0 and 2 at p = 0.0, outside the train band [0.1, 0.7]. That is a correct rejection: those tasks do not train that policy.
- **G4c/G4d/G5/G6 (CPU) ✅:** all 13 per-task detectors are proven by re-injection (clean fixture passes, defective fixture fails); 10 per-kit detectors point at their red-team/unit tests; backfill quarantines only failing tasks; editing one negative changes the key and re-grades only that negative.
- **G2/G3/G4a/G4b ⏳:** need the seed adapters on the lab GPU.

- **Code:** `src/crucible/accept/`:
  - `model.py`, `policy.py`, `key.py`, `cache.py`, `plan.py`, `run.py`;
  - `checks/` (`base.py`, `core.py` A1–A5/A7/A8, `extended.py` A6/A9–A13/S1);
  - `mutation/` (action-sequence operators);
  - `registry/` (`model.py`, `defects/*.yaml`, `fixtures.py`, `run.py`);
  - `contract.py` (the M8 subset, kept for the forge).
- **Toy:** `crucible_pack_toy/forge/task.py` capabilities, `crucible_pack_toy/accept_gate.py`. **Coding:** `crucible_pack_coding/accept_fixtures.py`.
- **Tests:** `tests/unit/test_accept.py` (9 tests).
- **Report:** `reports/m9_toy_g1.json`.
- **Research:** `docs/research/m9-m10/01-acceptance.md`, `03-hf-rl-env-datasets.md`.

## What it is

| Piece | What | Built on |
|---|---|---|
| `model.py` | `Status` pass/fail/not_applicable/pending/error; N/A only from a closed vocabulary of task facts; pass/fail need evidence; `AcceptanceReport.accepted` derived on every read (no stored boolean); overrides never on A1–A6/A13 | OpenEnv grader protocol; Repo2RLEnv `report.py`; agentenv skip policy |
| `key.py` | Acceptance key = Harbor hash ⊕ overlay hash ⊕ image digests ⊕ kit hash (`verify/runtime` + the plan module) ⊕ contract, policy and registry versions | research §15.7 (Harbor's hash excludes `crucible/`) |
| `cache.py` + grade store | Check results cached per (key, check, version), reused only if their evidence still hashes; grades stored per (grading environment, solution content), so a key change re-grades only what changed | `vf-validate` resume-by-hash |
| `plan.py`, `run.py` | Dependencies, then cost class (static → cpu → docker → gpu → paid → human); missing need → pending; failed dependency → pending; missing adapter capability → fail (absence never implies N/A); crash/infra → error | NeMo-Gym TE-6 |
| A1 | ≥ 3 oracle grades = 1, replay identity, gold check (state changes; an outcome gate false at start), perf headroom ≥ 15%, ≥ 10 fresh-seed grades without a grader crash | R2E controls; H OracleAgent |
| A2 | No-op and floors score 0 on every channel, with a clean grade; a naive floor exists | `vf-validate` noop; FineEnvs |
| A3 | Authored negatives, valid alternatives, gate-family coverage, **generated mutants** (drop step, numeric shift, enum swap, drop message, reorder); a mutant with the oracle's end state is equivalent and excluded; score over valid, non-equivalent mutants ≥ 0.9; survivors reported as weak checks | R2E execution contrast; SWE-smith procedural |
| A4 | ≥ 8 identical full verdicts; fresh-gate draws (+ induced) with a Clopper-Pearson bound; ≥ 3 identical reference runs | `verify/calibrate.py` |
| A5 | Hidden strings and canary vs everything agent-visible, instruction leak tokens, sibling overlap, agent image audit | `verify/leak_audit.py` |
| A6 | Containment audit + tamper attempts that must be blocked (reward 0 **and** no state change) | `world/audit.py` |
| A9 | Images resolve to digests equal to the pins, recorded vs fresh reference fingerprints, static task lint; World tasks: two fresh starts agree | task04 `fingerprints.json` |
| A10 | ≥ 2 scored frontier runs (masked excluded) with failure taxonomy; pass rate in the split's band; paid runs through the M10 runner | `forge/learnability.py` |
| A11 | The B.5 injections on the task's own runtime path through the M10 runner (blank usage, credits, contention, grader crash) + oracle/no-op controls | M10 `inject.py` |
| A12 | Slowest full grade within 80% of the verifier timeout; fast vs full Spearman ≥ 0.8 and no negative that only the full tier kills | `verify/tiers.py` |
| A13 | Attacks that must take effect (control) and still score 0. Toy: claim-without-doing, forged approval, shotgun all tickets | M7 finding 1 |
| S1 | `signoff.yaml` bound to the key, for H4+ | agentenv review |
| Registry | One YAML per defect (incident, detector, fix, re-injection, status); a detector is admitted only with a power test or a named per-kit test; the registry version is part of every key; backfill quarantines | Karotte replay; OpenEnv `cases.json` |

## Found by M9 (DESIGN B.15)
- **Forged approval passes on state-only checks.** On forged toy tasks, refunds recorded with the supervisor's name and never asked for scored 1.0. The name is hidden, but the world is constant across episodes, so a policy learns it once and forges it afterwards. Fixed with the `event` check kind `approval_grounded`: the approval must appear in the episode.
- **The hindsight supervisor rejected every request.** An empty in-transit list became the regex `(?:)`. The oracle refunded anyway, and state-only checks never saw it. Fixed.
