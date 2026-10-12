# M7 verify: audit (2026-10-12)

Scope: `src/crucible/verify/` (the in-container runtime `runtime/{run,plan,trusted}.py`, the patterns
`hidden_tests`, `differential`, `perf`, `static`, `compare`, `record`, `invariants`; `answers`, `calibrate`, `tiers`,
`grader`, `leak_audit`, `collect`, `stage`, `state`), the coding pack graders (`task01`-`task04`, the goldens), the toy
plan, the workspace harness's collect step (`runtime/collect.py`, `runtime/harness/workspace.py`) and the Harbor
test.sh Crucible writes (`interop/harbor/role_runner.py`).

Checked against ARCHITECTURE §M7, DESIGN §8 (P16, P21, P26, P27, P29) and `docs/modules/M07-verify.md`, with the M0
protocol. The framework comparison is the shared M5-M7 sweep (`M05-M07-study-sweep.md`, M7 picks); the defect hunt is
`M07-study-defects.md` (V1-V23, probes `m7hunt/p1`-`p11`, one run end to end as root in the A13 verifier image).
29 regression tests (`tests/unit/test_m7_audit.py`) all fail on the M6 commit (`76ec547`) and pass after the fixes.

## 1. Requirement ledger

| # | Requirement (source) | Before | After |
|---|---|---|---|
| L1 | `hidden_tests`: verdict from a results channel, exact expected tests | **wrong**: a candidate `pytest11` entry point rewrote every report; a wrong library scored `tests` = 1 in the root container (V1); count only | plugin autoload off; outcomes from `call.excinfo` in an outermost hookwrapper; `unittest` assertions snapshotted; `expected_ids` + `collect_pytest_ids` (repo2rlenv) |
| L2 | `differential` + fresh draws | holes: NaN/±inf reference accepted anything (V12); empty reference passed empty candidate (V13) | same-class rule for non-finite references, bools by type; an empty or failing reference is infra (V13, V14); outputs read through `safe_read` (no links, FIFOs or oversize files) |
| L3 | `perf`: best-of-k, spread, load guard | partial: candidate-side noise masked (V2); ≤ 0 timings passed (V5); task03 differencing forgeable (V4) | mask only on original-side noise; timings positive and finite; record mode keeps `wall_ratio_max`; task03 rejects a non-positive or implausibly fast per-step time |
| L4 | `static` | partial (V6-V8) | `ast` imports, builtin `exec`/`eval`/`compile` and `__builtins__` lookups, `.pyc`/zip/`ar`/wheel files, `pytest11` declarations, binaries in skipped directories, every file in frozen directories, spliced C continuations, `.inc`/`.tpp`/`.cc`/`.cxx` |
| L5 | `state_predicate`, `invariants` | partial: a zero-row scalar or a dropped table masked (V10) | no row is a negative answer; a schema error on mutable state is a fail; transport errors stay infra |
| L8 | Separate verifier environment | host hole: a container that never ran reported the previous grade (V9) | the host removes every reward file first; rc 125-127 with no file → `verifier_not_started`; unreadable JSON → masked; the container is removed in `finally`; params always JSON (V21) |
| L9 | Trusted runtime (P26) | infra errors scored against the candidate (V14) | a failed privilege drop or a missing grader `cwd` is `InfraError`; `seal` removes links, FIFOs, sockets and devices (submission custody, repo2rlenv/Karotte) |
| L10 | Collect hooks (B.5) | committed work lost; nested `build/` sources dropped (V20) | the harness records the base sha before the agent runs; `build`/`dist`/`.venv` anchored at the root; `tree_changed` ignores derived files (a do-nothing agent that ran Python was masked) |
| L11 | Leak audit (A5, P21, P27) | `/tests`, `/solution`, `/usr` unscanned; refs to blobs/trees passed; > 4 MiB skipped (V11) | every top-level directory but `proc`/`sys`/`dev`; non-commit refs flagged; chunked scan of any size |
| L12 | Masks never scored, never forced (P16) | a certain hard-gate fail was masked by any infra check (V3); `abstained` or `setup` errors on candidate content masked (V16); NaN in `reward.json` (V18) | a definite hard-gate fail scores 0 with the unmeasured checks listed; candidate-caused setup errors fail every hard gate; non-finite values mask; host-side masked documents carry `reward: null` |
| L13 | `calibrate`: tolerance tiers kill negatives, ≥ 40 natural draws | passed with no negatives; induced draws counted (V22) | at least one negative must be killed; only natural draws count; `flake_rate` counts masked draws apart |
| L14 | `tiers`: fast/full agreement (A12) | `ok` ignored `fast_pass_full_fail` (V17) | `ok` requires zero; `Plan.fast_unmeasured_hard` and the fast report name hard gates the fast tier never measures |
| L15 | `grader` reward files | a score gate with two checks paid 2.0 (V15) | averaged per gate, then across gates |
| — | `answers` (V19) | nested JSON gave the innermost value; deep nesting crashed; a NaN judge scored 1.0; "A good choice is C" gave A | outermost value ending last; `RecursionError` → no answer; non-finite or non-numeric judge output → `CheckerUnavailable`; a leading letter needs a delimiter |
| — | Masked grades under other runtimes (sweep D4, G5) | rLLM read an agent-writable `/tmp/rllm/reward.json` first and treats no file as 0; stock Harbor `Mean` counts a mask as 0; `plan.py` claimed every consumer masks | `prewrite_zero` and the role-runner test.sh remove rLLM's file; `interop/harbor/metrics.py` (masked mean + mask rate, a Harbor `uv-script` metric); the claim corrected |
| L6 | `adversarial_inputs` pattern | missing | still missing (fresh draws are the nearest equivalent) |
| L7 | `judge_item` | unwired | still no production caller (M6 hardened its parsing and votes) |
| L18 | Exit gate: tasks 01-04 re-expressed | partial | unchanged: tasks 01/01v2/02 static only; task03 GPU golden open |

## 2. Defects (reproduced) and fixes

Ids are from `M07-study-defects.md`. Every row has a regression test in `test_m7_audit.py`.

| Id | Sev | Defect | Fix |
|---|---|---|---|
| V1 | P0 | `pytest11` plugin forgery of hidden tests | `PYTEST_DISABLE_PLUGIN_AUTOLOAD`; outcomes from `call.excinfo` in a `tryfirst` hookwrapper; static flags `pytest11` and pytest imports parsed with `ast` |
| V2 | P1 | candidate-side jitter masked | original-side contamination only |
| V3 | P1 | one infra check masked a certain 0 | `_gate_status`: fail before infra; definite hard-gate fail → scored, `infra.unmeasured_checks` |
| V4 | P1 | task03 start-up forging | per-step time must be positive and ≥ 5% of the original's |
| V5 | P1 | ≤ 0 timings; unbounded record reward | candidate → `CandidateError`, original → `InfraError`; a "min" record score must be positive; record mode keeps the wall bar |
| V6 | P1 | Python static bypasses | `py_imports` (every alias), `DYNAMIC_EXEC` (builtins only), compiled/archived suffixes and magics |
| V7 | P1 | binaries in skipped directories | forbidden/native checks before `skip`; frozen digest over every non-prose file |
| V8 | P2 | C scanner gaps | continuations spliced; `.inc`/`.tpp` in the kit and task04 `CODE` |
| V9 | P1 | stale grade when the container never ran | host unlinks `REWARD_FILES`; `verifier_not_started`, `unreadable_reward_json` |
| V10 | P1 | missing row / dropped table masked | see L5 |
| V11 | P1 | leak audit holes | see L11 |
| V12 | P1 | non-finite reference accepted anything | see L2 |
| V13 | P2 | empty reference | `References.get` raises `InfraError` |
| V14 | P2 | infra counted against the candidate | `SubprocessError` and missing `cwd` → `InfraError`; reference exceptions wrapped |
| V15 | P2 | score gate summed | per-gate average |
| V16 | P2 | candidate content outside checks masked | `_gave_up` swallows errors; `CandidateError` or a copy error inside the candidate's tree during `setup` fails every hard gate (`CandidateError` moved to `trusted.py`, re-exported by `perf`) |
| V17 | P2 | fast tier paid what full rejected | see L14 |
| V18 | P2 | NaN in `reward.json` | masked `verifier_error` |
| V19 | P2 | answer edge cases | see the `answers` row |
| V20 | P2 | collect lost work | see L10. **Not fixed:** the patch is still decoded with `errors="replace"`, so a non-UTF-8 context line can break `git apply` |
| V21 | P2 | `--param` changed string types | the host always JSON-encodes |
| V22 | P2 | tolerance tiers vacuous | see L13 |
| V23 | — | open research items | `kill_cohort` survivors still raise `InfraError` (a mask the candidate may trigger, research V2); `judge_item` unwired; xfail in hidden suites still counts as a skip; the results fd stays open in the candidate (documented P26 limit) |

Found while fixing: the first `DYNAMIC_EXEC` pattern matched `re.compile(` and flagged 28 lines of honest task03 code;
the CPU golden caught it (task03 static fell to 2/10) and the rule now matches only the builtins, with a control test.

## 3. Coverage and contracts

- `crucible.verify` over the full suite: **79%** (the study measured 71% on a 97-test subset). Per module (before = the study's subset): `collect.py` 100%,
  `leak_audit.py` 96%, `static.py` 95%, `plan.py` 94%, `hidden_tests.py` 63% (36% before; the real pytest runner now
  runs offline), `stage.py` 60% (0%). Still thin: `runtime/run.py` 0% (in-container entry, docker tests only),
  `differential.py` 32%, `trusted.py` 53% (root-only paths need Linux root).
- Suite (`tests/unit`, `tests/toy_e2e`): **994 passed, 3 skipped** (965 before, plus the 29 new tests).
- `lint-imports`: 4 kept. `ruff`: clean (two lint issues left in `test_m6_audit.py` by the M6 commit are fixed here).
- Goldens re-run: `golden_cpu` passes (task04 36/36 recorded verdicts, task03 static 10/10); `m7_golden_cpu.json`
  changes in two places only: `astra_v3`'s expected mask now names the original's contaminated rounds, and one
  truncated violation list shows the new `banned_import` row. The toy golden passes and `m7_toy.json` is unchanged.
  **The GPU goldens were not re-run** (task04 8/8 and the open task03 run need the lab GPU); the perf changes touch
  both paths, so the task04 GPU golden should be re-run before the next GPU campaign.

## 4. Comparison with the local frameworks

The picks are in `M05-M07-study-sweep.md` (M7 table). Built here: repo2rlenv's exact test ids and assertion-tamper
detection; submission custody (repo2rlenv `swe_smith/grade.py`, Karotte `save_submission.py`): links and special
files removed at seal, outputs read without following links; a masked-mean Harbor metric (Harbor `uv_script`
interface) and the rLLM reward-file guard (sweep G5); NeMo-Gym's fail-closed reading of reward documents on the host.
Queued: verifier versioning with automatic regrade (Harbor `job_diff`/`regrade`), determinism calibration of the
verifier itself, an always-fail audit, record-mode repeat measurements, agentic confirmation of hack flags, and the
Harbor job config that retries `RewardFileNotFoundError` when `mask.json` exists (M11).

Where Crucible is stronger: no surveyed framework separates "the candidate failed" from "the grader could not
measure" at check level, or prevents the candidate from choosing between them; Harbor, rLLM and agentenv all read a
missing or broken reward as 0 (agentenv reads NaN as 1.0).

## 5. Open

- V20 patch bytes (decode with `errors="replace"`), V8 two-level `##` macro pastes, and research V2 (`kill_cohort`
  survivors masking) are not fixed.
- `adversarial_inputs` (L6) and a production caller for `judge_item` (L7).
- Kit plans for tasks 01/01v2/02; the task03 GPU golden; a task04 GPU golden re-run after these perf changes.
- Wiring `MASKED_MEAN_PY` and the retry rule into the M11 Harbor export.
