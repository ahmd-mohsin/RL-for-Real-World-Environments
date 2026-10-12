# M7 study: defects in `verify` (verifier kit, isolation, calibration, seed graders)

Scope: `C/verify/` (`runtime/{run,plan,trusted}.py`, `runtime/patterns/{hidden_tests,differential,invariants,perf,static,compare,record}.py`, `answers.py`, `calibrate.py`, `tiers.py`, `grader.py`, `leak_audit.py`, `collect.py`, `stage.py`, `progress.py`, `sql_checks.py`, `state.py`, `judge_item.py`), the coding pack graders `CP/verify/{task01,task02,task03,task04,golden_cpu,golden_gpu}.py`, the toy plan `TP/verify.py`, the red-team task `tests/redteam/grader_task/plan_minitask.py` (read only) and the reports `docs/modules/reports/m7_*.json`.

Spec: ARCHITECTURE "### M7 `verify`". Its exit gate (golden) says that the graders of tasks 01–04, re-expressed with the kit, reproduce every historical result: every oracle scores 1.0, every negative fails at its intended gate, the historical GPT 6 Astra and GPT 5.6 sol patches reproduce their gate profiles, and the toy checks pass and fail as designed. Also read: `docs/modules/M07-verify.md`, DESIGN P16, P21, P26, P27 and P29, and `docs/research/improvements/M06-M08.md` (V1, V2 and V4 in its M7 section).

Paths: `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`. `TP` = `crucible/packs/_toy/src/crucible_pack_toy/`.

Method: a read-only study on HEAD `b65754b`, using `crucible/.venv` (Python 3.12.9, pytest 9.1.1) on macOS. **[ran]** means reproduced with a probe; **[read]** means established from the code alone. The probes are in the session scratchpad at `m7hunt/p1_perf_plan.py … p11_state.py`. One probe ran end to end in Docker as root (`p3_docker_entrypoint.py`). It used the A13 red-team image `crucible-verify-test:py312`, which was already present locally, and the unchanged `plan_minitask`. No GPU, paid or networked run was made.

Offline gates re-run:
- `python -m crucible_pack_coding.verify.golden_cpu` passes in 20 s and rewrites `m7_golden_cpu.json` byte for byte (no git diff).
- `python -m crucible_pack_toy.verify` passes (oracle 1, no-op 0, 7/7 negatives as designed). `m7_toy.json` is unchanged.

## 1. Requirement ledger

Status: **live** = built and behaves as specified; **partial** = built, but missing a specified part, or with a reproduced hole; **wrong** = contradicts the spec; **missing** = not built.

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| L1 | Pattern `hidden_tests`: verdict from a results channel, exact expected tests (ARCH Takes: Repo2RLEnv) | **wrong** | A candidate-shipped `pytest11` entry point rewrites outcomes before the kit's plugin records them. A wrong library gets `tests` = 1 in the root verifier container (V1). The check compares the **count** only, never the expected node-id list (`hidden_tests.py:84-85`) |
| L2 | Pattern `differential` + `fresh_draws`, references root-only and cached, self-consistency | live, with holes | The A13 suite covers this. A reference NaN or ±inf accepts anything (V12). An empty reference passes an empty candidate (V13) |
| L3 | Pattern `perf`: best-of-k, spread, load guard | partial | Live in shape. A candidate can still mask (V2; research V1 is still open). Non-positive candidate timings pass (V5). task03's outside-clock differencing can be forged (V4) |
| L4 | Pattern `static` (comments, macros, dynamic imports, native files) | partial | Python import lists, sourceless `.pyc`, `ar` archives, skipped directories, multi-line `##` and the C suffix lists all bypass the rules (V6, V7, V8) |
| L5 | Patterns `state_predicate` and `invariants` | partial | Live, and the toy golden passes. A zero-row scalar or a dropped table masks the grade instead of failing it (V10) |
| L6 | Pattern `adversarial_inputs` (ARCH contents) | **missing** | Nothing named or shaped like it exists in `C/verify` (grep). Fresh draws are the closest equivalent |
| L7 | `judge_item` (validated items only, never a gate) | partial | `admit` is correct, but nothing calls `judge_items` (0% coverage, no caller in `src/` or `packs/`). Evidence goes into the prompt unfenced (research V4) |
| L8 | Separate verifier environment (Harbor separate mode; `--network none`; hidden mounts) | live, with a host-side hole | `stage.docker_grade` runs the image as root with `--network none` and the bundle mounted read-only. It reads a stale reward when the container never ran (V9) |
| L9 | Sealing / trusted runtime (P26) | live | Cohort kill, then root-owned read-only copies. `python -I` ordering, results fds. The A13 attacks were re-read. Infra errors inside `Trusted.run` are scored as candidate failures (V14) |
| L10 | Collect hooks (Appendix B.5) | partial | Committed work, `build/`/`dist/` source directories and non-UTF-8 files are lost or corrupted (V20) |
| L11 | Leak audit (A5, P21, P27) | partial | Image roots skip `/tests`, `/solution` and `/usr`. Refs to blobs or trees pass. Files over 4 MiB are silently skipped (V11) |
| L12 | "No parseable reward → masked" (MiMo); masks never scored (P16) | partial | In-container: live. Host: a truncated `crucible_reward.json` raises (V9). `reward.json` can carry `NaN` (V18). A definite gate fail is masked when another check is infra (V3) |
| L13 | `calibrate.py`: Clopper–Pearson, tolerance tiers that still kill negatives, ≥ 40 natural draws | partial | The CP upper bound is correct (`n=40, 0 fails → 0.0722`). With no negatives, `tolerance_tiers` reports `kills_negatives=True, ok=True`. Induced draws count toward the 40 (V22) |
| L14 | `tiers.py`: fast/full agreement (A12) | partial | Spearman is tie-aware and correct. `ok` ignores `fast_pass_full_fail` [read], and a gate with no fast check is unmeasured in the fast tier (p1: fast 1.0, full 0.0) |
| L15 | `grader.py` writes `reward.json` and `steps.jsonl` | live | Single producer `write_reward_files`. Masked means no `reward.json`. Non-finite values are not validated (V18) |
| L16 | Coding: sanitizers, concurrency vs solo, bit-exact, perf rounds | live | `CP/verify/task04.py` 138-148, 348-379. The GPU golden for task04 is 8/8 |
| L17 | Toy: SQL invariants and state predicates pass and fail as designed | live | The re-run passes. Masks are reachable through V10, but not on the toy's own predicates (they use `COUNT`/`COALESCE`) |
| L18 | Exit gate: every oracle 1.0, every negative fails at its intended gate, Astra/sol profiles, tasks 01–04 | **partial** | task04: GPU 8/8 (`oracle3` 1.0, `astra_v3b` fails perf, `sol_v3` 0). task03: `m7_golden_gpu_t3.json` has `pass: false` (`neg_floatgrid` and `cheat_tamper` have `match: false`). tasks 01/01v2/02: only the static gate is re-expressed (`task01.py:1-15`); golden_cpu re-scores the static, perf and sanitizer verdicts, not the full profiles |

## 2. Defect candidates

Severity: **P0** = a policy can obtain reward it did not earn, in the kit's main path; **P1** = a wrong verdict, a mask the policy can trigger, or an audit hole in normal use; **P2** = an edge case, wrong attribution or a robustness gap.

### Verdict forgery

**V1 [P0][ran] A `pytest11` entry point in the candidate's path forges every hidden-test outcome.** `pytest_suite` runs `python -I` with the candidate's site directory appended to `sys.path` (`hidden_tests.py:104-106`, `trusted.py` `python()`). pytest auto-loads every `pytest11` entry point that `importlib.metadata` finds on `sys.path`, and entry-point plugins register *after* `-p crucible_verify_pytest`, so their hooks run first. A plugin with `def pytest_runtest_logreport(report): report.outcome = "passed"` needs no `import pytest`, so `runner_tamper` does not see it.
- Kit, local (p2): hidden suite of 2 tests, library returns a wrong value. Honest-wrong: `passed: false, reasons ["2 failed", ...]`. With `mdlite-0.1.dist-info/entry_points.txt` = `[pytest11] mdlite_helpers = mdlite_helpers`: `passed: true, collected 2, passed_n 2, failed []`.
- End to end, as root in `crucible-verify-test:py312` with the unchanged `plan_minitask` (p3): `wrong_plain` gives gates `tests 0`; `wrong_entrypoint` gives gates `tests 1`, `collected 4, passed 4, reasons []`, `trusted: true`.
- task03's static gate passes the source tree that produces this. The tree contains `pyproject.toml` with `[project.entry-points.pytest11]` and the hook module, and gives `pass: True, violations []`. `pip install --target` writes the `entry_points.txt`.
- Expected: tests gate 0. Impact: on task03, `api/forces/dynamics/physics/app` become free. `fresh` and `perf` still bind correctness there, but any plan whose correctness evidence is `pytest_suite` alone is fully forgeable.
- Fix:
  - set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` in `pytest_suite`'s env (children inherit it);
  - have the kit plugin record outcomes from `pytest_runtest_makereport` as a `tryfirst` hookwrapper over `call.excinfo`, not from the mutable report;
  - static: flag `pytest11` / `entry-points` and `*.dist-info` in candidate trees;
  - add an A13 case `entrypoint_plugin`, with its weak-grader control.

**V4 [P1][ran] task03 perf: the outside-clock difference can be forged.** `res[side][case] = max(1e-6, (w2 - w1) / (4n) * 1e3)` (`CP/verify/task03.py:221`) assumes start-up cost does not depend on `k`, but the candidate controls start-up.
- Probe (p8, real `Task03.perf_round`/`perf` with `_timed` stubbed). The original costs 2 s + 10 ms/step. The candidate costs 2 s + **30 ms/step** (3× slower) plus a start-up sleep of 6000/k s. Got `passed: True`, `candidate 0.0 ms/step, ratio 0.0 ≤ 0.5`. The end-state sanity check passes because the physics is right.
- Fix:
  - treat `w2 ≤ w1`, or a per-step time below a floor (for example 5% of the original's), as `CandidateError`;
  - also bound the long run's whole-process wall ratio (`wall_ratio_max` on `w2`), as task04 does.

**V5 [P1][ran] Zero or negative candidate timings pass; record mode turns them into unbounded reward.**
- `PerfBars` accepts `c ≤ 0`, because `ratio = c / o` (`perf.py:111`):
  - candidate `-5.0` gives `passed: true`;
  - `0.0` gives `passed: true`;
  - an original of `0` fails the candidate instead of masking.
- task04 record mode measures with `PerfBars({m: 10.0}, …, remeasure=0)` and no `wall_ratio_max` (`task04.py:495-497`). The per-metric numbers are read from JSON the candidate's process wrote:
  - `record_score({"kinfu_frame_ms": {"ratio": -1000}}, …)` gives `score 9.3826, new_records ["kinfu_frame_ms"]`;
  - `RecordSpec(0.8, 0.571, "min").reward(-1e6)` gives `16.29`.
- Fix:
  - a non-finite or non-positive candidate metric raises `CandidateError`, and a non-positive original raises `InfraError`;
  - record mode must use the whole-process wall check;
  - `record()` should reject scores outside the metric's physical domain.

**V12 [P1][ran] `json_close` accepts any candidate when the reference is NaN or ±inf** (`compare.py:71-77`):
- `json_close(123.0, nan)` gives `[]`;
- `json_close(0.0, inf, atol=1e-6)` gives `[]`;
- `json_close(0.0, inf, rtol=1e-3)` gives `[]`;
- `json_close(nan, nan)` fails;
- `json_close(1, True)` passes.

Fix: if `ref` is not finite, require the same class (both NaN, or the same signed inf); compare bools by type.

**V13 [P2][ran] An empty reference passes an empty candidate.** `References.get` never checks that the reference produced anything, and `bitwise_tree(empty, empty)` returns `[]`. p5: a reference runner that writes nothing and does not raise, plus a candidate that writes nothing, gives `differential … passed True`. Fix: `References.get` raises `InfraError` when the reference output is empty, or when it lacks the files the plan declares.

**V17 [P2][ran] The fast tier can pay what the full tier rejects, and A12 does not catch it.** A hard gate with no fast check is dropped from the fast verdict, so `all_hard` is vacuously true. p1: `build` (full only) fails and `tests` (fast) passes; the fast reward is 1.0 and the full reward is 0.0. `tiers.agreement` counts these cases as `fast_pass_full_fail`, but its `ok` depends only on Spearman [read]. Fix: `ok` requires `fast_pass_full_fail == 0` (or a bounded rate). `Plan.__post_init__` warns when a hard gate has no fast check.

### Masks a policy can trigger (P16 says masks are for infrastructure only)

**V2 [P1][ran] A slow, jittery candidate is masked instead of failed** (research V1, still open). `perf.py:141-142` masks whenever the failing attempt is contaminated, including on the candidate's side alone.
- p1: original a stable 100 ms, candidate alternating 300/600 ms, bar 1.0. Got `infra: "perf rounds contaminated: ['candidate/m']"`.
- Fix (as the research doc says): mask only for original-side contamination or a busy guard sample. A failing best-of-k on the candidate side is a fail, because noise only slows a run down.

**V3 [P1][ran] One infra check masks a grade whose hard gate already failed for certain.** `_gate_status` ranks infra above fail (`plan.py:233`), and any infra check masks the whole grade (`plan.py:192-194`).
- p1: `static` = fail, `perf` = infra gives `masked` (reward never trained as 0).
- p1: one gate with a fail and an infra check gives a gate status of `infra`.
- Gates are a conjunction, so a definite hard-gate fail decides the reward whatever is unmeasured. Combined with masks the candidate can trigger (V2 above, research V2 `kill_cohort` survivors, V10, V16), every negative can be turned into a mask.
- Fix: if any hard gate is `fail`, score 0 with `infra.status = ok` and list the infra checks in the detail. Mask only when no hard gate failed.

**V10 [P1][ran] State predicates mask on a missing row or a dropped table.**
- The sql adapter returns `ok=True, value=None` for a scalar query over zero rows (`C/world/state/sql.py:114-118`). `phi` gives `None` for `between`, `score` and counts (`workflow/milestones/evaluate.py`). `state_predicates` turns any `None` into infra (`state.py:51-56`).
- `view_query` raises `InfraError` on any query error (`state.py:63-65`), even though `sql_checks.py` says that "a query that fails is a failed check (the policy may have broken the schema)".
- p11, a refund predicate (`between 129.98–130`) when the policy never refunded: `t1: infra`, `masked`. With the policy-dropped table: `pol: infra, t1: infra`, `masked`.
- Fix: `ok=True, value=None` is a negative answer (φ = 0) for `between`/`score`. A query error on state the policy can mutate is a fail. Keep infra for adapter or transport errors only.

**V16 [P2][ran] Candidate content touched outside checks masks the grade.**
- `abstained(ctx)` raising (for example, `ABSTAIN` is a directory) gives `grader_exception`, masked (p1).
- A FIFO in the candidate tree, copied in `Plan.setup` (the `plan_minitask` pattern, `plan_minitask.py:87`), gives `masked, grader_exception: … is a named pipe` (p8).
- task03 and task04 copy inside `static()`, so there it is a fail.
- Fix: wrap `abstained` so that an exception means "did not abstain". Document, and lint, that `setup` never reads candidate trees, or map `shutil.Error`/`SpecialFileError` there to a candidate fail.

**V14 [P2][ran] Infra failures inside `Trusted.run(candidate=True)` count against the candidate.**
- A failing privilege drop (preexec raising) gives `SubprocessError`, and the check reads `fail` (p10).
- A grader-chosen `cwd` that does not exist gives `rc 127 "cannot execute"`, a candidate failure (`trusted.py:331-334`).
- A reference runner that raises a plain exception (not `InfraError`) also gives `fail` (p5).
- Fix: catch `SubprocessError` and turn it into `InfraError`. Check that `cwd` exists before `Popen`. In `References.get`, wrap any exception from `run` as `InfraError`.

### Isolation, audit, collect

**V9 [P1][ran] `docker_grade` reports a previous grade when the container never ran.** The host never clears `out_dir`; only `run.py` (inside the container) does.
- p6, fake `docker` exiting 125 ("invalid mount config") with an old `crucible_reward.json` in `out_dir`: `rc 125 → reward 1.0, infra ok`.
- A truncated `crucible_reward.json` raises `JSONDecodeError` from `docker_grade` (`stage.py:136`) instead of masking.
- Fix: unlink `REWARD_FILES` on the host before `docker run`. When the return code is non-zero and no file was written after the start time, mask `verifier_not_started`. Wrap the parse to mask `unreadable_reward_json`.

**V11 [P1][ran] The leak audit misses the usual answer locations.**
- `audit_image` scans only `workspace, app, root, home, opt, tmp, srv` (`leak_audit.py:90`). p6, a fake export holding `/tests/test_hidden.py`, `/solution/solution.patch` and a canary under `/usr/local/lib/…/site-packages`: `ok: True, scanned_files 1, findings []`.
- `audit_git` uses `rev-list --all` (commits only, `:123`). A tag pointing at a blob that holds the oracle gives `ok: True` with `refs/tags/leak` listed. A ref to a tree also gives `ok: True`.
- `audit_tree`: a hidden string inside a 5 MiB file gives `ok: True`. Oversized files are silently skipped (`:85`).
- Fix:
  - scan every top-level directory except `proc`, `sys` and `dev`;
  - in `audit_git`, flag any ref whose object type is not `commit`, and diff `rev-list --all --objects` against the base;
  - report skipped files as findings, or scan them in chunks.

**V20 [P2][ran] The collect hook loses or corrupts work.**
- `base="HEAD"` by default (`collect.py:28`, and `${2:-HEAD}` in `COLLECT_SH`): an agent that committed its fix yields `files 0, patch bytes 0`. With the recorded base sha it yields `files 1`. `runtime/collect.classify` then sees a clean tree and scores it "nothing changed" (0). The workspace harness passes `base="base"`; the default and `COLLECT_SH` do not.
- `DERIVED` names such as `build` and `dist` match at any depth: the new sources `tools/build/gen.py` and `dist/packaging.cmake` are excluded, giving `files 0`.
- The patch is decoded with `errors="replace"` (`:48`). A file with a Latin-1 byte in a context line produces a patch that no longer applies (`error: patch failed: a.c:1`).
- Fix:
  - require an explicit base sha;
  - anchor the patterns as `/build/`, or exclude only gitignored build trees;
  - keep the patch as bytes.

**V21 [P2][ran] `run.py --param` changes the types of string parameters.** The host passes strings raw (`stage.py:127`). `run._param` calls `json.loads` on them: `name=true` becomes `True`, `tag=[1]` becomes `[1]`, and `n=1e3` becomes `1000.0`. Fix: on the host, always `json.dumps` the value.

### Static checks

**V6 [P1][ran] Python bypasses of the task03/task02 static gates.**
- `foreign_kernels` matches only `import X` at the start of a line or as the first name:
  - task03: `import os, ctypes` passes;
  - task02: `import os, cupy` passes.
- The runner-tamper patterns behave the same way: `import os, pytest` and `import \⏎ pytest` pass.
- Dynamic imports pass: `__builtins__.__dict__['__imp'+'ort__']('cty'+'pes')` with `getattr(m, 'CD' 'LL')`, and `exec('import cty'+'pes')`.
- A sourceless `mdlite/_k.pyc` (compiled from `import ctypes; ctypes.CDLL(...)`) passes the task03 rules. It imports under `python -I` (p4, `pyc_demo`). `!<arch>` static archives are not in `NATIVE_MAGIC`.
- Fix:
  - parse imports with `ast` (every alias, every `Import`/`ImportFrom`), and flag `exec`/`eval`/`compile`/`__builtins__` subscripts;
  - forbid `.pyc`/`.pyo`/`.zip`/`.egg` in candidate trees;
  - add `!<arch>`, `PK\x03\x04` and the CPython pyc magic to the native check.

**V7 [P1][ran] task01: native files in skipped directories are never reported.** `scan` tests `skip` before `forbidden_files` and `forbid_native` (`static.py:150`), and the frozen-directory digest covers only "buildable" suffixes. With task01 v2 rules and `frozen_base`:
- a new `third_party/libk.a` gives `pass=True`;
- a new `third_party/k.so` (ELF) gives `pass=True`;
- control: `src/libk.a` gives `forbidden_file`;
- control: `third_party/kern.cu` gives `frozen_dir_changed`.

Fix: apply the forbidden-file and native checks before `skip`, and digest every file in the frozen directories.

**V8 [P2][ran] C-family scanner gaps.**
- task04 `CODE` lacks `.inc .tpp .cxx .cc`: `texture<float,2>` in `gpu/a.inc`, included from `a.cu`, passes (control in `a.cu`: `texref0`).
- A backslash-newline inside an identifier (`cudaBindTex\⏎ture2D`) passes.
- Token pasting with `##` on a continuation line passes, two-level `CAT→CAT_` passes, and a continued object-like alias passes (control: single-line `CAT` gives `bind:token_paste`).
- The module docstring says shebang scripts are reported, but `NATIVE_MAGIC` has no `#!` (p4).
- task04 impact is limited: CUDA 12 removed texture references, so `build` fails them anyway.
- Fix: splice `\⏎` before scanning; scan every file in the rule's roots, not only the suffix list; run `cpp -E` (or record macro bodies across continuations) before matching.

### Plan, reward files, answers, calibration

**V15 [P2][ran] A score gate with several checks sums them.** `term = sum(vals) / len(scores)` (`plan.py`, score branch): two checks with `normalized_reward` 1.0 each, in one score gate, give `reward 2.0` (p1). Fix: average per gate first.

**V18 [P2][ran] `reward.json` can contain `NaN`.** A score check whose `normalized_reward` is NaN writes `{"check/s": 1.0, "outcome": 1.0, "reward": NaN}`, which strict JSON rejects. Harbor raises on it, and `flat_reward` promises finite numbers. Fix: validate finiteness in `write_reward_files`, and mask (`non_finite_reward`) when a value is not finite.

**V19 [P2][ran] Answer parsing and checking edge cases** (`answers.py`):
- `json_value('{"a": {"b": 1}}')` gives `{'b': 1}`, and `'[1, [2, 3]]'` gives `[2, 3]`: the innermost last value wins, so `json_subset` fails correct answers.
- `ParserSpec("json").parse("[" * 20000)` raises `RecursionError`. Only `ValueError` is caught, so the single-turn grade crashes instead of scoring 0.
- A judge returning NaN scores **1.0**, because `max(0, min(1, nan))` gives 1.0 (`:470`). This is the agentenv defect the module doc warns about.
- A judge returning `"yes"` raises `ValueError`, not `CheckerUnavailable`.
- `choice_letter("I think the answer is B")` gives `None`, and `"A good choice is C"` gives `A`.
- Fix:
  - take the outermost value that ends last;
  - catch `RecursionError`;
  - mask non-finite or non-numeric judge output;
  - try the leading letter only when it is followed by a delimiter (`B)`, `B.`, `B:`).

**V22 [P2][ran] `tolerance_tiers` passes without negatives and counts induced draws as natural.**
- 40 natural draws and no negatives give `kills_negatives: True, ok: True` (`calibrate.py:99`).
- 1 natural + 39 induced draws give `ok: True`, but DESIGN §8.2 asks for ≥ 40 natural draws.
- Fix: require at least one negative, and count natural draws alone toward `min_draws`.

**V23 [P2][read] Open research items, confirmed in the code.**
- `kill_cohort` raises `InfraError` on survivors (`trusted.py:157-160`), so a mask the candidate can trigger remains (research V2).
- `judge_item` puts the evidence into `ITEM_PROMPT` unfenced, and `parse_verdict` takes the last `VERDICT:` (research V4). It is also unwired (L7).
- `tiers.agreement.ok` ignores `fast_pass_full_fail`.
- `hidden_tests` treats an xfail in a hidden suite as a skip, so with `allowed_skips=0` it can never pass. Nothing flags this at authoring time.
- The results fd stays open in the candidate process, which is the documented P26 limitation.

## 3. Coverage

Run from `crucible/` with `COVERAGE_FILE` pointed at the scratchpad (the repo's `.coverage` is unchanged):

```
pytest -q --cov=crucible.verify tests/unit/{test_verify,test_verifier_separate,test_single_turn,test_enterprise_primitives,
  test_longhorizon,test_judge_specialists,test_multistep,test_turns,test_interop_gate,test_agentenv_plugin_fixes,
  test_train_audit,test_train_seats}.py
97 passed in 44.5 s; TOTAL 2395 stmts, 71%
```

| Module | Cover | Notes |
|---|---|---|
| `runtime/run.py` | 0% | the in-container entry point; only docker red-team tests reach it |
| `stage.py` | 0% | `docker_grade`, `bundle`; V9 and V21 are untested |
| `judge_item.py` | 0% | no caller anywhere |
| `runtime/patterns/differential.py` | 23% | `References`, `differential`, `concurrency_equivalence` and `fresh` only run in docker tests |
| `runtime/patterns/hidden_tests.py` | 36% | `pytest_suite`/`doctest_suite`/`count_pytest` are unexercised offline. V1 lives here |
| `runtime/trusted.py` | 48% | root paths (`kill_cohort`, `_demote`, `seal` as root) need Linux root |
| `runtime/patterns/perf.py` | 73% | `LoadGuard.sample`/`wait` uncovered |
| `leak_audit.py` | 76% | `audit_image` uncovered (V11) |
| `state.py` | 74% | `cross_check` uncovered |
| `runtime/patterns/invariants.py` | 77% | `sqlite_query` uncovered |
| `plan.py` 95%, `static.py` 90%, `compare.py` 90%, `answers.py` 88%, `calibrate.py` 92%, `sql_checks.py` 92%, `grader.py` 93%, `progress.py` 96%, `tiers.py` 97%, `record.py` 97%, `collect.py` 89% | | |

The hidden-test path (V1), the differential path (V12, V13) and staging (V9) have no offline test, so the probes `p2`, `p5` and `p6` are the cheapest regression seeds. Each one runs in under 2 s without docker.
