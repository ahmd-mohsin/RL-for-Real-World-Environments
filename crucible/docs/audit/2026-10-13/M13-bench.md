# M13 `bench` audit (2026-10-13)

Scope: `src/crucible/bench/` (metrics, compare, report, run, splits, IRT, contamination, dynamic), its CLI and studio call
sites, and what the benchmark campaign showed it depends on: the Harbor in-place grader (`interop/harbor/`), the M6
progress tracker, the M10 runner and the OpenEnv runner path. Base `b6d89fc` (M12). Study: `M13-study-sweep.md` (two
sweeps, two hunts, a benchmark campaign over 8 local sources, ≈1,750 episodes). Regression tests:
`tests/unit/test_m13_audit_{metrics,splits,harbor}.py` (50 + 33 + 13 functions); each was run against a pristine
checkout of `b6d89fc` and fails there except two mutant-killers in the metrics file that pin behaviour the old code
already had right.

## 1. Requirement ledger

| # | Requirement (source) | Before | After |
|---|---|---|---|
| L1 | A hack never outranks an honest model (G2-1, campaign B1) | exposed-answer episodes were masked, so a hacker's failures vanished (pass@1 1.0 vs 0.5); in Harbor in place, stubbing the grader's own tools (pytest, curl, uvx, a verifier venv) scored 1.0 on every Harbor set with no flag | bench scores exposure as 0 with a flag and reports a `hacks` block; a grader-toolchain manifest is taken before the policy acts and checked host-side at grade time: tampering is the policy's scored 0 (`crucible:grader_tampered`), flagged; models compared on common support |
| L2 | Infra faults never score (M10 rule; campaign B2) | Harbor in place: a lost container or a grade-time network drop scored 0 (9 of 9) | host-side liveness (`docker inspect` / events) and grade-time network evidence mask (`env_container_lost`, `grader_network_error`); only a container the policy demonstrably broke is scored |
| L3 | Only valid tasks are benchmarked (campaign B4/B5; M9 A1/A2) | a no-op passed 13.4% of workplace rows; a broken-oracle task counted as a hard item; no oracle gate | a validity manifest (oracle + no-op controls by default, or M9 acceptance, or passed in) removes broken, flaky and no-op-passing tasks from every metric, IRT and compare (`invalid_tasks` with reasons); no manifest ⇒ `validity: unknown` |
| L4 | Every number states its denominator (G1 D2/D3, H1 V5/V8, G2-6/7) | each k had its own task set (pass@2 > pass@1); the headline was an episode mean; a stopped run read as the whole benchmark | headline = mean of per-task means; one task set for all k (`pass@k_tasks`, `pass@k_excluded`); planned vs run vs owed with the stop reason |
| L5 | Uncertainty that covers (G1 D5, H1 V9) | percentile task bootstrap covered 0.76–0.90 at 3–10 tasks, 12.6% in a masked-heavy case | t interval on task means (scipy-free quantile), `n_tasks` shown, marked approximate below 10 tasks; coverage ≥ 0.92 at 5 tasks in a seeded test |
| L6 | Comparisons are paired at the right unit (G1 D1/D7, H1 V7, G2-8, campaign B3/B11) | pairs keyed on (task, sample) only: policies mixed silently, order-dependent results; McNemar treated a task's k samples as independent (p = 1.2e-7 vs 0.0078) | one (policy, harness) per side or refusal; flips per task; exact McNemar on task flips; t interval and exact sign-flip test on per-task deltas |
| L7 | Spend is all spend (G1 D4, H1 V4) | masked retries and uncertain USD never reached cost ($0.10 vs $2.10 per success) | every attempt's spend, uncertain spend included, retries counted |
| L8 | Progress metrics read the episode as it happened (H1 V1–V3) | the end-of-episode correction row was applied first; the start state was ignored; the last step had zero AUC weight | rows in episode order with the correction at the end; a `step=0` row from the M6 tracker seeds Φ0 and initial milestones; AUC = mean of Φ over steps 1..T |
| L9 | Held-out data cannot leak (H2 V1/V2/V7–V10, G2-9/10/11) | a held-out task's lineage siblings stayed in train (499/500 fuzz cases violated `check`); the seal covered ids only; eligibility failed open; real timestamps crashed | union-find lineage components decide splits; full-label HMAC seal; fail-closed eligibility; naive / aware / epoch timestamp parsing |
| L10 | Dynamic items cannot be predicted (H2 V3/V4/V11/V25) | published ids carried the seeds; default keys were public strings; unsalted commitment; key on argv | ids = HMAC(K, bench, version, family, index); a ≥ 32-byte secret key from env or file (0600 file created when none is given); salted commitment; short benches and starved families reported |
| L11 | IRT estimates are right and honest (H2 V5/V6/V14/V24, G2-2/3) | the 2PL fit 2-cycled (28 nats off the MAP); anchored fits shrank new models toward 0 with 26–79% CI coverage; drift quarantined clean anchors 17–89% of the time | Levenberg–Marquardt joint MAP with a convergence flag (matches scipy to ≈1e-5); anchor-centred priors and a delta-method ability SE (coverage 0.94–0.95, bias ≈ −0.1); robust drift with two-sided tests and Holm (1/40 false quarantines; real drift caught 38–40/40) |
| L12 | Contamination tools work on the real data (G2 d, H2 V13/V15/V16/V20–V23) | the Terminal-Bench canaries in the local data were missed; 12–18% precision (template boilerplate); near-dups missed once a solution was appended; the leak regex fired on hex colours; nothing was wired | canary registry (foreign GUIDs, case / spacing / bare GUIDs), template subtraction, containment score, a fix-reference leak check; `curator_hook` for the forge curator; splits and contamination run in `studio/build` before any run |
| L13 | Every environment can be benchmarked from the CLI (campaign B8/B9/B17, G1 D6) | OpenEnv tasks could not run through the runner; `--k 3` reported pass@1 only; no path for non-pack tasks | OpenEnv runs (string instruction at spec time, Harbor optional); default ks from k; `bench run --harbor <dir>` and `--tasks module:attr` |
| L14 | Other ecosystems can read the results (G1 §4) | none | `harbor_result.json` (Harbor JobStats shape), `crucible_aggregate_metrics.json` (NeMo-Gym shape with `coverage/*`), the verifiers-style dual headline |

## 2. Corrections to earlier claims

- `docs/modules/M13-bench.md`: the toy gate's "12 regressions, p < 0.001" rested on per-sample pairs; the task-level
  result is 6 task regressions, p = 2/64.
- `docs/research/bench-studio/01-framework-study.md`: NeMo-Gym is not the only framework that excludes errors
  (verifiers, prime-rl and OpenEnv do too), and NeMo-Gym's per-k task drop is the defect Crucible had copied; the
  promised t-intervals for AUC and cost were never built.
- The committed `reports/m13_coding_recorded` digest was stale; re-derived (`4436b086e23c2bf5`, task-mean headline,
  distinct sample indices).

## 3. Gates and numbers

- Regression tests: metrics 50, splits 33 (36 cases), harbor 13; every one fails on `b6d89fc` except 2 metrics
  mutant-killers. All 17 mutant classes of the metrics module are killed.
- Campaign re-checked at the fix: real TB2.1 `adaptive-rejection-sampler` oracle 1.0, no-op 0, an agent that installs
  curl itself and solves 1.0 unflagged, the campaign's shadow adversary 0 as `grader_tampered`; the 9 infra cases mask.
- Full suite: in the commit message. lint-imports 4 kept; ruff clean.

## 4. Open

- Row → runnable-task adapters for the six SWE row sets and Reverse-Text (M11 I0–I10); NeMo ReasoningGym,
  instruction_following and structured_outputs need server deps; CFBench / SysBench need a paid judge.
- Grader-integrity residual risks (documented in `interop/harbor/grader_guard.py`): shared libraries, a forged package
  database, project-local uv / pip config, `.pth` in a system Python the grader shares by design, a tool swapped inside
  the grade window.
- A deterministic harness exception (same class and message twice) is still retried; `RunRecord` does not copy rollout
  flags (bench reads them from `rollout.json`).
- `item_qa` does nothing below 15 respondents; forged toy tasks expose no card, so build uses synthetic lineage.
