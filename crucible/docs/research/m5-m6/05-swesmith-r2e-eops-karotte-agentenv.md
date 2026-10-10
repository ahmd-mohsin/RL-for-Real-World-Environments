# 05 — SWE-smith, Repo2RLEnv, EnterpriseOps-Gym, Karotte, agentenv: grading, milestones, negatives, snapshots

Scope: M5 (`workflow`: replay, CI-derived milestones, tracker) and M6 (`credit`: C1 Φ, C2 defect→check + negative mutant, C3 branch value, C4 rubric items, R1–R6) plus the harness layer.
Read-only survey. Every mechanism is cited `path:line`.

Path prefixes:
- `SS/` = `RL-Environments/06-frameworks-and-env-code/swe-bench-swe-smith/`
- `R2E/` = `RL-Environments/06-frameworks-and-env-code/huggingface-repo2rlenv/src/repo2rlenv/`
- `EOPS/` = `RL-Environments/06-frameworks-and-env-code/servicenow-enterpriseops-gym/`
- `KAR/` = `RL-Environments/06-frameworks-and-env-code/preferencemodel-karotte/`
- `AE/` = `agentenv-framework/src/agent_env/`

---

## 1. TL;DR: 18 takeaways

1. **SWE-smith's F2P/P2P algorithm is a 4-way contingency over two status maps. It is easy to port and easy to get backwards.** `get_valid_report` iterates over the second map's tests, skips any test missing from the first map, and buckets exact `PASSED`/`FAILED` pairs into F2P/P2P/F2F/P2F (`SS/swesmith/harness/grading.py:40-92`). The direction is inconsistent inside the repo. `valid.py` passes `(buggy, clean)` (`SS/swesmith/harness/valid.py:117,131-135`). `scripts/bug_gen_modal.py:222-226` passes `(clean, buggy)`. `gather.py` reads `PASS_TO_FAIL` as F2P (`SS/swesmith/harness/gather.py:333-348`). **Crucible should** port the contingency as `transition_report(before: StatusMap, after: StatusMap)` with explicit named states (`s_t`, `s_T`), and pin the direction with a fixture test.

2. **Use Repo2RLEnv's two-stage validation as the per-step CI-milestone primitive.** Stage A runs the *hidden final test patch only* at base. Stage B runs base + gold + test patch (`R2E/pipelines/pr_runtime_validate.py:240-299`). F2P counts `FAILED|ERROR → PASSED` (`:288-296`); `ERROR` matters because new tests error at import. **Crucible should** overlay the *final* trace's test files onto every replayed step state `s_t` (the "test-patch overlay") and run the suite. Tests that do not exist at `s_t` otherwise silently drop out of the map. Without the overlay, per-step F2P is empty for any test written late in the trace.

3. **Dense reward from tests: `reward = f2p_rate · p2p_rate`.** The flags `resolved`, `command_resolved` and `untracked_failed` are kept separate (`R2E/pipelines/_pr_runtime_verifier.py:350-427`). A missing test counts as failure (`:402-403`; `SS/.../grading.py:102-106`). **Crucible should** use exactly this as the CI-derived component of Φ(s_t): weighted F2P fraction × P2P survival. Log `regressions` per step as the deviation signal for the Workflow State Tracker.

4. **Strict comparability gate for replay.** Repo2RLEnv's `execution_contrast` raises when (a) the healthy run is not green, (b) the mutation changed the *test collection* (`healthy.statuses.keys() != defective.statuses.keys()`), or (c) the defective exit code ≠ 1 (`R2E/quality/test_results.py:55-65`). JUnit parsing rejects DTDs, duplicate ids, empty results, and exit-0-with-failures (`:20-52`). **Crucible should** adopt these exact quarantine conditions for M5 replay: a step whose collection differs from the final collection is "collection drift", not a milestone failure.

5. **Parsers that synthesize test names poison per-step F2P.** The ctest and gtest parsers fall back to `synthetic_pass_{i}` / `test_passed_{i}` when only summary lines exist (`SS/swesmith/profiles/cpp.py:476-487, 541-562`). Those names are not stable identities across states. **Crucible should** reject (quarantine) any status map that contains synthetic names. Our GPU-migration family (CMake/ctest/gtest) is exactly where this would bite.

6. **Repair flaky oracles by re-running gold.** SWE-smith evaluates the gold patch. It drops F2P/P2P tests that fail under gold. It deletes the instance if it timed out or no F2P passed (`SS/swesmith/harness/repair.py:122-215`). **Crucible should** run the final recorded state K≥3 times and the base state K times. Tests whose status is not constant across runs are excluded from milestone sets before mining. This feeds the "monotone on good traces" filter.

7. **Anti-tamper evaluation is three mechanisms.** SWE-smith and Repo2RLEnv both reset test files to base and re-apply hidden tests after the agent's edit (`SS/swesmith/harness/utils.py:224-237`; `R2E/pipelines/pr_runtime.py:466-502`). Repo2RLEnv fails closed if the hidden test patch does not apply (`:477-494`). Karotte kills every student process, then grades a root-only copy (`KAR/docs/why-karotte.md:166-171`). **Crucible should** make all three mandatory in the World grading path; they back R5 ("stubbing not rewarded") and R6.

8. **Negative-mutant sources (C2): SWE-smith gives four complementary generators.**
   - **Procedural:** AST operator flips, constant changes, removed loops/conditionals/assignments, if/else inversion. C++ uses likelihood 0.5 (`SS/swesmith/bug_gen/procedural/base.py:58-145`, `SS/swesmith/bug_gen/procedural/cpp/__init__.py:32-49`).
   - **LM-modify:** subtle bugs at T=1 (`SS/configs/bug_gen/lm_modify.yml`).
   - **LM-rewrite:** blank the body, then re-implement at T=0, producing "plausible but wrong" code (`SS/swesmith/bug_gen/llm/rewrite.py:95-155`).
   - **PR-mirror:** an LM reverse-applies a historical diff onto the current code (`SS/swesmith/bug_gen/mirror/prompts.py:1-20`).

   **Crucible should** use PR-mirror semantics as "undo step k's diff, keep later steps". That is a natural trace-derived negative for every mined milestone. Use procedural mutants to compute a check-kill matrix.

9. **A mutant only counts if it kills ≥1 test and keeps ≥1 P2P.** `gather.py` skips instances with timeout, `n_f2p == 0` or `n_p2p == 0` (`SS/swesmith/harness/gather.py:337-341`). **Crucible should** apply the same validity rule before a mutant enters the R3 ("negatives score 0") property suite.

10. **Control evidence is the R1/R2 release gate.** Each Repo2RLEnv bundle must show `nop` ×2 scoring exactly 0 and `oracle` ×4 scoring exactly 1. Every trial is bound to the bundle hash and the result's sha256 (`R2E/pipelines/recipes/codemidas/audit.py:196-219`). **Crucible should** require a stored control receipt (no-op→0, oracle→max, mutants→0) per task version, hash-bound, before the task is admitted to training.

11. **Diff-similarity is a weak, calibratable shaping signal, not an outcome.** pr_diff combines six components with weights 0/.08/.12/.20/.10/.50. The LLM judge carries 0.5. The final score is capped at 0.40 when size_sanity < 0.10 (`R2E/pipelines/_pr_diff_verifier.py:405-437, 525-526`). It is calibrated per task as `(raw−b)/(1−b)`, with b = the empty-diff score (`R2E/pipelines/pr_diff.py:397-434`). **Crucible should** borrow only `file_targeting` (F1) and `region_overlap` as cheap step-level localization features for the tracker. Never use them as Φ, since they reward text, not state (R6).

12. **SWE-Flow's `development_schedule` is a ready-made milestone ladder.** It groups tests by their traced set of core functions, sorts groups by size, and emits steps `{nodes_to_develop, test_ids, previous_tests}` (`R2E/pipelines/recipes/swe_flow/schedule.py:34-61`). **Crucible should** use it as a *prior* for stage induction: milestone k = "test group k passes while all previous_tests still pass". It also gives a precedence graph for free.

13. **EnterpriseOps-Gym is milestone-as-SQL-assertion.** Each task carries 2–19 named `database_state` checks (mean 7.8 across the 13 shipped tasks). Each is `SELECT COUNT(*) … WHERE <full row predicate>` with `expected_value = 1, equals` (`EOPS/data/revised/*`; schema `EOPS/benchmark/verifier.py:81-127`). The score is all-pass plus `verifier_level_pass_rate` partial credit (`EOPS/benchmark/executor.py:371-395, 565-612`). **Crucible should** express enterprise milestones the same way (one named state predicate per milestone) and must add collateral-damage checks: only 1 of 102 shipped checks asserts absence.

14. **Action-based checks violate R6.** EOPS `tool_execution` passes if listed tools were called (`EOPS/benchmark/verifier.py:166-193`). **Crucible should** keep such checks as tracker diagnostics only, with weight 0 in R.

15. **Karotte step semantics: per-step judge, `continue_task` gating, run score = the score of the last scored step** (`KAR/src/karotte/evaluation_runner.py:207-219`; `KAR/docs/tasks/scoring.md:17`). Misbehavior scores 0 and halts; infra errors fail the run (`KAR/src/karotte/step.py:82-92`, `KAR/src/karotte/student_misbehavior.py:11-18`). **Crucible should** mirror this as the distinction between C6 (infra mask) and agent fault (score 0). Do not copy the "last step only" aggregation: it throws away the dense channel.

16. **agentenv's weighted aggregator with negative weights is the right algebra for C4 and the mistake penalties.**
    `score = clip( (Σ_{w>0} s·w − Σ_{w<0} (1−s)·|w|) / Σ_{w>0} w , 0, 1)` (`AE/task_step/task_steps/verifiers/scoring.py:25-39`).
    Mistake findings carry a severity penalty: low .02, medium .08, high .25, critical 1.0. The penalty is scaled by `pos_den`, so it lands as an absolute hit (`AE/.../judge_utils/judge_output_format.py:1042-1135`). Each finding also carries a 0-based `step` index (`:192-194`). **Crucible should** DEPEND on `aggregate_score` for C4 and map judge `findings[].step` onto `steps.jsonl` as candidate defect locations for C2.

17. **Exact snapshot APIs:**
    - `snapshot_env` exports every MCP service via `get_data`/`/export-state` into a new `EnvironmentUniverseArtifact`. It is all-or-nothing. Its id is stable per `instance_id` (`derive_id(instance_id, "snapshot-"+step.id)`), so retries bump versions. It writes `context.metadata["env_snapshotted_universes"][step.id] = {id, version}` (`AE/task_step/task_steps/snapshot_env.py:141-273, 366-378, 501-560`).
    - `snapshot_agent_state` captures the agent workspace (+ optional per-service JSON) to `metadata["agent_snapshots"]` / `metadata["snapshot_json_url"]` (`AE/task_step/task_steps/snapshot_agent_state.py:95-201`).
    - `SnapshotConfig(per_turn=True, max_snapshots=24, timeout_seconds=300)` on `prompt_agent` yields "a curve of gradable points" (`AE/task_step/snapshot_utils/snapshot_series.py:1-66`).
    - Restore is `load_artifact(artifact_from_step_id=<snapshot step>)` (`AE/task_step/task_steps/load_artifact.py:28-31, 336-347`).

    **Crucible should** build C3 branch rollouts on these: per-turn snapshot, then K × (deploy, load_artifact, `Task.run(start_step=…)`) (`AE/task/task.py:390-457`).

18. **Graders get hacked in the frameworks we build on.** Karotte's comparison table (`KAR/docs/why-karotte.md:346-365`) reports two hacks:
    - In agentenv, the agent replaced `bash`, and a `bash_cmd_succeeds` grader then passed a wrong answer.
    - In verifiers, the agent overwrote the grading script staged in `/tmp/vf-scripts/` and a wrong answer scored 1.0.

    **Crucible should** never grade via binaries or paths the agent could write. Run checks from a root-owned interpreter (`sys.executable`) on a custody copy, after killing agent processes.

---

## 2. Per-source mechanisms

### 2.1 SWE-smith (`SS/`)

**Log capture contract.**
- `run_patch_in_container` writes `eval.sh` (`SS/swesmith/harness/utils.py:241-262`): `set -uxo pipefail`, then `: '>>>>> Start Test Output'`, the test command, and `: '>>>>> End Test Output'`.
- Because of xtrace, the markers appear in the log as `+ : '<marker>'`. `read_test_output` slices between them (`SS/swesmith/harness/grading.py:21-37`).
- `APPLY_PATCH_FAIL` or `TESTS_TIMEOUT` in the content returns `(None, False)`: not gradable, which is not the same as failed (`:23-26`).
- **Port:** in Crucible replay, distinguish `ungradable` (apply failure / timeout / markers missing) from `failed`. Only the latter is a milestone outcome; `ungradable` goes to quarantine or the C6 infra mask.

**Validation (bug → F2P/P2P), `SS/swesmith/harness/valid.py:56-147`.**
1. The reference run of the clean repo happens once per repo with `timeout_ref = 900 s` (`:198-219`; defaults at `SS/swesmith/profiles/base.py:108-109`). With `min_pregold`, the reference runs per instance using the targeted test command (`valid.py:74-100`).
2. The bug patch is applied with `timeout = 90 s` (`valid.py:102-108`).
3. `get_valid_report` produces the buckets (`grading.py:61-92`). Exact algorithm:

```python
for t in postgold_sm:
    if t not in pregold_sm: continue          # asymmetric tests silently dropped
    a, b = pregold_sm[t], postgold_sm[t]
    PASSED,PASSED -> P2P ; FAILED,PASSED -> F2P ; FAILED,FAILED -> F2F ; PASSED,FAILED -> P2F
    # ERROR / SKIPPED / XFAIL in either map: dropped from every bucket
```

Status 0_f2p vs 1+_f2p is decided from `FAIL_TO_PASS` (`valid.py:144-147`). The direction caveat from TL;DR 1 applies. The fixture `SS/tests/test_logs/run_validation/report.json` has F2P=2, P2F=0 under valid.py's argument order (`SS/tests/harness/test_grading.py:72-80`).

**Eval-time grading (`grading.py:95-195`, `:198-238`).**
- `test_passed` = status ∈ {PASSED, XFAIL}. `test_failed` = test missing OR status ∈ {FAILED, ERROR}. A missing test is a failure, which closes the "delete the test" hack.
- The report has success/failure lists for F2P and P2P. F2F gives "extra credit" and P2F is "not considered"; both are optional (`calculate_to_fail`).
- Resolution comes from `swebench.harness.grading.get_resolution_status`. That package is **not vendored** here, so its exact source was not found.
- Repo2RLEnv's re-implementation documents the rule as: FULL iff `f2p_rate == 1 and p2p_rate == 1`; PARTIAL iff `0 < f2p_rate < 1 and p2p_rate == 1`; else NO (`R2E/reward.py:161-169`).
- `f2p_only` filters F2P/P2P to the F2P test files (`grading.py:221-230`).

**Eval-time anti-tamper (`utils.py:179-237`).**
- Each task branch has two commits, "Bug Patch" and then "Remove F2P Tests" (`gather.py:530-556`). The agent works on the version with the F2P test files *deleted*.
- At eval: `git checkout HEAD~1` restores the tests (`utils.py:191-205`). The patch's touched files are reset, the prediction is applied with fallbacks (`GIT_APPLY_CMDS`: `git apply`, `--reject`, `patch --fuzz=5`; `SS/swesmith/constants.py:40-44`), and then `git checkout -- <f2p+p2p test files>` reverts any agent edits to tests (`utils.py:224-237`).

**Profiles (`SS/swesmith/profiles/base.py`).**
- `timeout=90`, `timeout_ref=900`, `min_testing`, `min_pregold` (`:108-118`).
- Test-path heuristic: basename starts with `test`, stem ends with `test`, or a directory named `tests|test|specs` (`:549-558`).
- `get_test_cmd` (`:564-656`) has three modes:
  - (a) F2P files only.
  - (b) F2P + P2P files when `min_testing`.
  - (c) Map each patched file to a test file by naming convention (`test_{stem}`, `{stem}_test`, …), else fall back to a sibling test directory.
- Log parsers return `{test_name: status}`. Python uses the regex `^(\S+)(\s+){STATUS}` over pytest `-v` output (`SS/swesmith/profiles/python.py:93-102`). C/jq, Valkey and others are in `SS/swesmith/profiles/c.py:55-66, 96-111`. C++ has ctest/gtest/catch2/boost/qtest parsers (`SS/swesmith/profiles/cpp.py:449-760`), each with the synthetic-name fallbacks noted in TL;DR 5.
- **Port:** the gtest and ctest parsers are directly relevant to the GPU-migration family. PORT them, but delete the synthetic fallback, or mark its output `ungradable`.

**Repair (`repair.py:110-215`).** Input: a run of the gold patch through eval.
- Timeout → delete the instance.
- `len(f2p_passes)==0` → delete.
- Otherwise remove `F2P.failure` and `P2P.failure` from the instance's lists (`:186-214`).

This is a one-shot flake filter. Crucible should generalize it to K-run stability.

**Bug generation as negative-mutant sources.**
- **Procedural:** `ProceduralModifier` has `max_attempts=5` and `min_complexity=3`, and fires on a coin flip at `likelihood` (default `DEFAULT_PM_LIKELIHOOD=0.2`, `SS/swesmith/constants.py:15`; C++ uses 0.5). Applicability is gated by entity tags such as HAS_IF_ELSE or HAS_LOOP (`SS/swesmith/bug_gen/procedural/base.py:14-52`). It has 17 operator families (`:58-145`). C++ operator tables, e.g. flips `== ↔ !=`, `< ↔ >=`, `&& ↔ ||`, `<< ↔ >>`, are in `SS/swesmith/bug_gen/procedural/cpp/operations.py:14-62`. Generation enumerates (entity, modifier) pairs, optionally interleaved, and caps them with `max_bugs` / `max_candidates` (`SS/swesmith/bug_gen/procedural/generate.py:110-180`). Note that `.cu`/`.cuh` are **not** in `MAP_EXT_TO_MODIFIERS` (`SS/swesmith/bug_gen/procedural/__init__.py:20-35`); mapping them to the C++ modifiers is a one-line extension.
- **LM-modify:** a chaos-monkey prompt with 3 shuffled bug archetypes, including off-by-one, swallowed exceptions, wrong defaults and edge-case mishandling. Rules: no compile errors, no signature change, no comments revealing the bug. It samples n bugs at T=1 (`SS/configs/bug_gen/lm_modify.yml:1-52`; `SS/swesmith/bug_gen/llm/modify.py:57-112`).
- **LM-rewrite:** blank the body to a stub, ask the LM to implement it given the whole file at T=0, then diff (`SS/swesmith/bug_gen/llm/rewrite.py:95-155`). This produces the most realistic "agent-like" wrong solutions, and is ideal for R5 (stub/plausible-but-wrong must score low).
- **PR-mirror:** an LM rewrites the file "reversing the changes in the diff, nothing else" (`SS/swesmith/bug_gen/mirror/prompts.py:1-76`), with recovery bookkeeping in `SS/swesmith/bug_gen/mirror/generate.py:104-330`.
- **Combine:** `same_file` / `same_module` merge validated single bugs into multi-bug patches, taking `num_patches`, `limit_per_file` and `max_combos`. Excluded types are `func_basic`, the combine types and `pr_mirror` (`SS/swesmith/bug_gen/combine/same_file.py:1-60`).

**Leak-safe issue generation (`SS/configs/issue_gen/ig_v2.yaml`).**
- Inputs: 2 demonstration issues, the bug patch, the test output, and up to 5 F2P test sources (sampled in shuffled order; `SS/swesmith/issue_gen/generate.py:278-294`).
- Hard rules: never give away the fix, never say tests failed, never suggest running pytest. Describe reproduction, not cause.
- The dataset ships the F2P tests *removed* from the agent's branch (`gather.py:539-556`).

### 2.2 Repo2RLEnv (`R2E/`)

**Candidate mining filters.**
- *pr_runtime pre-filter* (`R2E/pipelines/pr_runtime.py:964-982`) rejects drafts, unmerged PRs, PRs with no files, `_is_non_bug_pr` titles (backport / cherry-pick / release / revert / bump, `:333-335`), and bodies shorter than `min_problem_statement_words`.
- *Structural filters* (`:984-1010`): reject CI-only patches (all under `.github/`) and test patches with no new test functions (`require_new_test_funcs`).
- *Lite filters* (`:1012-1032`): exactly 1 source file, ≥40 words, no images, no external links, no commit SHAs.
- *commit_runtime metadata filter* (`R2E/pipelines/commit_runtime.py:467-499`):
  - Rejects merges, excluded authors, messages under `min_message_words=5`, and conventional-commit types chore/docs/feat/refactor/style/test/ci/build/perf/revert (`:91-94`).
  - Requires a bugfix signal: a `fix:` prefix, a linked issue, or a keyword regex (`:100-104`).
- Defaults (`R2E/spec/options.py:49-140`):
  - pr_runtime: `min_fail_to_pass=1`, `validation_timeout_sec=600`, `max_source_files_per_pr=50`.
  - pr_diff: `min_loc_changed=3`, `max_files_per_pr=5`.
  - commit_runtime: `max_pass_to_pass=50`, `max_source_files_per_commit=10`.

**Patch/test split.** A file block goes to `test_patch` iff either path is a test file. Docs roots never count as tests. A test-dir component or test basename does (`R2E/pipelines/pr_runtime.py:191-262`).

**Targeted test selection.**
- `targeted_test_cmds_for_pr` appends the changed test files for pytest and jest/mocha/vitest. For `go test` it replaces `./...` with the changed packages. Cargo is not targeted (`R2E/pipelines/pr_runtime.py:720-781`).
- The P2P cap keeps "near" tests (those in patched files) first, then fills to 50 (`R2E/pipelines/commit_runtime.py:396-405`). This bounds both flake risk and runtime.

**Two-stage validation** (`R2E/pipelines/pr_runtime_validate.py:206-327`).
- Fetch the base commit.
- Stage A: reset to base + apply the test patch, run.
- Stage B: reset to base + patch + test patch, run.
- Parse up to 5 M characters. Truncating to 20 k had elided the test section (`:247-249`).
- Patch-apply failure → `failed` with a reason.
- F2P = `pre ∈ {FAILED,ERROR} ∧ post == PASSED`; P2P = `PASSED → PASSED`. The loop runs over pre's test names (`:288-296`).
- `verified` iff F2P is non-empty.

**Eval script** (`R2E/pipelines/pr_runtime.py:435-546`).
- Steps: `git checkout base -- <test files>`, apply the hidden test patch, and **fail closed** (reward 0, `parse_status=test_patch_apply_failed`) if it does not apply. Then run the commands into a log and call `verifier.py` with `f2p.json`/`p2p.json`. Test files are reset again on exit. The script always exits 0; `reward.txt` is the verdict.
- Fallback when no F2P oracle exists: binary exit-code reward (`:518-525`).
- Verifier fallback when the log is unparseable: binary exit code, but `resolved=False` whenever an oracle exists, and `eval_trustworthy=False` (`R2E/pipelines/_pr_runtime_verifier.py:460-484`).

**Graded reward** (`R2E/pipelines/_pr_runtime_verifier.py:350-427`):

```python
f2p_rate = f2p_passed/f2p_total if f2p_total else 0.0
p2p_rate = p2p_passed/p2p_total if p2p_total else 1.0
reward   = clip(f2p_rate * p2p_rate, 0, 1)
resolved = f2p_total>0 and all F2P pass and all P2P pass
command_resolved = resolved and untracked_failed==0 and exit_code==0
regressions = [t for t in P2P if status[t] != PASSED]
untracked_failed = FAILED tests outside F2P∪P2P   # go: ignore subtests under a failed parent (:338-347)
```

**pr_diff (diff-similarity) reward** (`R2E/pipelines/_pr_diff_verifier.py`):

| Component | Definition | Lines |
| --- | --- | --- |
| `format_valid` | 0/1 | `:142-161` |
| `file_targeting` | F1 = 2TP/(2TP+FN+FP) over changed-file sets | `:164-187` |
| `size_sanity` | min(loc)/max(loc) | `:199-215` |
| `region_overlap` | fraction of oracle hunks with a predicted hunk within ±5 lines in the same file | `:218-240` |
| `similarity` | SequenceMatcher over +/- lines only, no context lines | `:243-256` |
| `llm_judge` | 0..1 or null | `:287-375` |

- Combination: `combine` renormalizes over the non-null components (`:383-397`). Default weights are 0/.08/.12/.20/.10/.50, chosen from a 23-task pilot (`:405-430`). Hard cap: `final ≤ 0.40` if `size_sanity < 0.10` (`:436-437, 525-526`).
- The eval shim deletes any pre-written reward file and runs `git add -A` so new files appear in the diff (`R2E/pipelines/pr_diff.py:261-309`). The oracle ships only in `tests/`, which is mounted at verify time (`:312-328`).
- Per-task no-op baseline: `calibrated = (raw−b)/(1−b)` (`:397-434`). Difficulty buckets by LOC are ≤5, 6–20, 21–80 and 81+ (`:437-453`).
- Leak stripping: remove `Closes #N`, markdown/GitHub URLs, `Co-authored-by` style trailers, and the squash suffix `(#1234)` (`:100-141`). commit_runtime can LLM-synthesize a leak-free statement, preferring the linked issue body (`R2E/pipelines/commit_runtime.py:206-232, 413-420`).

**code_instruct / equivalence_tests: stub-fail / oracle-pass contrast.**
- equivalence: Stage A stubs `name` (raises NotImplementedError) with `reference_name` intact, and the test must FAIL. Stage B puts the oracle in place, and the test must PASS. Pre-flight checks that the stub module is importable (`R2E/pipelines/equivalence_tests.py:528-610`).
- The instruction shows the signature and docstring only. It previously leaked the full body, a bug that was found by the baseline audit (`:712-748`). Residual hack: `reference_<name>` is present in the agent image (`:195-216`), so `name = reference_name` solves the task trivially.
- code_instruct uses the same two-stage contrast for LLM-written test + solution (`R2E/pipelines/code_instruct.py:1-20`; options `require_test_fails_without_oracle`, `require_test_passes_with_oracle` at `R2E/spec/options.py:180-181`).

**cve_patches.** OSV → fix-commit URL → split patch/test → same two-stage harness. When a CVE fix has no test patch, the task is emitted with `validation_status="no_test_patch"` and a weaker "tests still pass" signal (`R2E/pipelines/cve_patches.py:1-17`). The P2P cap is 50 (`R2E/spec/options.py:217-220`).

**Control evidence** (`R2E/pipelines/recipes/codemidas/audit.py:196-219`).

```python
for agent, count, expected in (("nop",2,0), ("oracle",4,1)):
    for i in range(count):
        receipt = read(f"{agent}-{i}/trial.json")
        assert receipt.state=="completed" and receipt.bundle_hash==identity
        assert sha256(trial.result)==receipt.result_sha256 and trial.reward==expected
```

nop/oracle trials are wired into nearly every recipe (`R2E/pipelines/pr_to_env.py:213`, `R2E/pipelines/recipes/r2e/pipeline.py:186`, terminal runner `:92,:359`). The terminal runner sets `expected = 0 if agent=="nop" else 1` (`R2E/pipelines/recipes/terminal/runner.py:92`).

**SWE-Flow schedule** (`R2E/pipelines/recipes/swe_flow/`).
- *Trace:* tests are run under `sys.setprofile` with a 5 s per-test window. Truncated traces are excluded (`provenance.md`).
- *Group and order:* `development_schedule` groups traces by `frozenset(core_nodes ∩ catalog)` and sorts by `(len, names)`. Each step introduces `new = core − developed` and records `test_ids` plus the cumulative `previous_tests` (`schedule.py:34-61`).
- *Skeletonize:* replace target function bodies with docstring + `...`, and delete newly introduced dependent functions (`:64-84`).
- *Validate:* contrast healthy vs skeleton with `execution_contrast` (`R2E/pipelines/recipes/swe_flow/worker.py:106-114`).

**swe_smith recipe grader** (`R2E/pipelines/recipes/swe_smith/grade.py`).
- Trust model: a trusted parent runs pytest as uid 1001 in a new session, with a scrubbed environment (`child_environment` whitelists only GPU/HF variables; `:20-43`).
- Submission checks: `validate_submission` refuses symlinks and non-.py additions (`:46-80`).
- Timeout and cleanup: after the timeout, `killpg` the whole process group. The JUnit report must not be a symlink and must be ≤16 MB (`:146-178`).
- Verdict: `passed = (code==0) ∧ expected ≠ ∅ ∧ parsed.passed == expected`, i.e. exact set equality (`:186-190`).

Its mutations module is a LibCST single-site enumeration port of SWE-smith (`R2E/pipelines/recipes/swe_smith/mutations.py:1-50`).

### 2.3 EnterpriseOps-Gym (`EOPS/`)

**Verifier types** (`EOPS/benchmark/models.py:6-37`; engine `EOPS/benchmark/verifier.py:48-79`).
- `database_state` (`:81-127`): run a SQL query via `POST {gym}/api/sql-runner` with header `x-database-id` (`:195-255`, 30 s timeout). Extract a value, then compare with `equals | greater_than | less_than | contains` (`:374-399`).
  - Value extraction collapses results: 1 row × 1 col → scalar; 1 row → dict; n rows → list (`:291-372`). This is fragile: a query that returns 0 rows yields `[]`, not 0.
- `response_check` (`:129-164, 412-483`): an LLM judge compares the SQL result with the agent's final text on a 1–10 scale. Pass iff `score ≥ minimum_comparison_value` (default 7). Judge failure → `passed: False`.
- `tool_execution` (`:166-193`): every `selected_tools` entry was called and `len(calls) ≥ minimum_tool_calls`. This is action-based (see TL;DR 14).

**Task structure** (`EOPS/data/revised/{csm,itsm}/task_*.json`; keys `system_prompt, user_prompt, selected_tools, restricted_tools, gym_servers_config, number_of_runs, reset_database_between_runs, verifiers`).
- Each verifier is `{verifier_type, name, description, gym_name, validation_config}`.
- In the 13 shipped tasks, all 102 verifiers are `database_state`/`equals`. 83 expect `1`, i.e. a COUNT of an exact-row predicate. Only one expects `0`.
- Names read like workflow milestones: "Verify case creation", "Verify group membership", "Verify case sla 1", …
- The README reports 9.15 avg steps per task (max 34), 5.3 avg verification conditions, and 34.1 % best success (`EOPS/README.md:43-51`). The final-state philosophy is explicit: "evaluated by SQL verifiers that check final environment state, not action sequences" (`:43`).

**Episode lifecycle and scoring** (`EOPS/benchmark/executor.py`).
- Per run: a fresh DB is created from `seed_database_file` per gym (`:320-349`). The orchestrator executes, then verifiers run against that DB (`:412-480`).
- Run outcomes: `overall_success = all(passed)`; `verification_summary = passed/total` (`:371-395`). Databases are deleted afterwards (`:555-563`).
- Aggregate: `overall_success_rate` (all verifiers pass), `pass_at_1`, and `verifier_level_pass_rate = Σpassed/Σtotal` (`:565-640`). `compute_score.py:12-50` averages the two rates across task files.

**Orchestrators.** `decomposing_planner` splits the task into 2–5 sequential subtasks with shared WorkingMemory, then runs a ReAct loop per subtask with `max_iterations=25` (`EOPS/orchestrators/decomposing_planner.py:1-10, 537-563, 785-800`). This is a useful harness baseline: an LLM-proposed stage list we can compare against induced stages.

### 2.4 Karotte (`KAR/`)

**Task/Step model.**
- `Task.steps` is an ordered iterable of `Step` (`KAR/src/karotte/task.py:94-97`).
- A `Step` has `instructions`, a `judge`, optional `submission_paths`, a `pre_scoring_hook` (collect submission) and a `post_hook` (`KAR/src/karotte/step.py:59-80`).
- `Step.score` runs the hook and then the judge. `StudentMisbehaviorError` → `Scoring(0, {"misbehavior":…}, continue_task=False)` (`:82-92`; `KAR/src/karotte/student_misbehavior.py:11-18`).
- The runner loop executes each step, scores it, and breaks with status `failed` when `continue_task` is False (`KAR/src/karotte/evaluation_runner.py:207-219`).
- Extra artifacts are saved *before* `pre_scoring_hook`, because hooks and judges may delete student files (`:242-247`).
- The run's score is the last scored step's score (`KAR/docs/tasks/scoring.md:17`).

**Scoring type.** `Scoring(score: float, metadata: dict, continue_task: bool)` (`KAR/src/karotte/schemas/scoring.py:12-16`).

**Judges.**
- `AndJudge` / `OrJudge` short-circuit on the `continue_task` truth value. They return the deciding judge's Scoring and render a tree of evaluated and unevaluated children (`KAR/src/karotte/judges/_composite_judge.py:104-194`; operator sugar `&`/`|` at `KAR/src/karotte/judges/judge.py:15-27`).
- `RubricJudge`: one LLM call per criterion, answering YES/NO on the first line. Score = Σ weights of YES criteria; `continue_task = score > continue_threshold` (default 0.0); temperature 0.3; `max_tokens` 32k; no context → score 0 (`KAR/src/karotte/judges/rubric_judge.py:30-126, 180-210`). It has no negative-weight semantics and no normalization: the score is unbounded if weights sum above 1.
- `ExecutableJudge`: a subprocess with heartbeat logging. It reads `{"score","metadata"}` from a temp file. Nonzero exit → score 0 and halt; `continue_task = score ≥ continue_threshold` (default −1) (`KAR/src/karotte/judges/executable_judge.py:17-185`; `KAR/docs/tasks/scoring.md:47-56`).

**Grading custody (harness-relevant)** (`KAR/docs/why-karotte.md:147-176`).
- `collect_submission` kills every student-owned process and confirms they are gone.
- It deletes other student files and copies the submission into a root-only directory, refusing symlinks, FIFOs, special files, and trees that are too deep or too big (`KAR/src/karotte/save_submission.py:46-243`).
- It strips student-writable dirs from `PATH`, `LD_LIBRARY_PATH`, `LD_PRELOAD` and `LD_AUDIT`. Scorers run under `sys.executable`.

The cross-framework attack table (`:346-365`) is the closest thing to a red-team checklist for the Crucible harness layer.

### 2.5 agentenv (`AE/`)

**Score algebra** (`AE/task_step/task_steps/verifiers/scoring.py:12-40`).
- `ScoreAggregator` ∈ {ALL_PASS, ANY_PASS, WEIGHTED_AVERAGE}.
- A row is `{result: bool, score?: float, weight?: float}`. Its score defaults to `1.0 if result else 0.0`.
- WEIGHTED_AVERAGE as in TL;DR 16. A negative-weight row penalizes `(1−score)·|w|`, so negative criteria must be phrased so that `score=1` means "anti-pattern absent". The judge prompt enforces this inversion explicitly, with a two-step ABSENT/PRESENT verdict (`AE/task_step/task_steps/verifiers/judge_utils/judge_output_format.py:76-100`). The rubric verifier flags `weight<0` criteria as `"negative": true` (`AE/task_step/task_steps/verifiers/rubrics_verifier.py:596-599`).
- Empty results → 0.0. `pos_den == 0` → 0.0.

**Mistake penalties** (`judge_output_format.py:1042-1135`).
- Rows: `summary` (weight 0), `botched` (positive, weight 1), one `coverage_*` row per criterion (positive), and one `mistake_i` per finding with `weight = −penalty·pos_den`, where `pos_den = 1 + len(coverage)`.
- The final score is therefore `coverage_pass_rate − Σ penalties`, clipped.
- The penalty table is `{low .02, medium .08, high .25, critical 1.0}`, with unknown severities falling back to medium.
- Findings schema: `{severity, step: int|null (0-based action index), should, description}` (`:192-194, 340-352`).
- Rules: "what you cannot see is not a mistake"; "a harness fault is not the agent's mistake", though blind repetition after a fault is (`:146-185`). These are exactly the C6-vs-agent-fault rules Crucible needs.

**Verifier steps.**
- `env_outcome_verifier` loads a Python file artifact, calls `await module.verify(mcp_url) -> list[row]`, aggregates (default ALL_PASS), and writes `metadata["verifications"][verifier_id] = {results, score}` (`AE/task_step/task_steps/verifiers/env_outcome_verifier.py:92-127`).
- `aggregate_verifiers` unions rows from several verifier ids, drops `skipped` rows, and re-aggregates (default WEIGHTED_AVERAGE) (`AE/task_step/task_steps/verifiers/aggregate_verifiers.py:63-90`).
- `verify_sandbox` supports probe types `probe_file_exists`, `probe_dir_exists`, `probe_file_contains` (regex) and `bash_cmd_succeeds` (`AE/task_step/task_steps/verifiers/verify_sandbox.py:1-20`). The last one is the type Karotte broke.
- `run_container_unit_tests_verifier`, in Harbor reward-file mode: `score = reward`, `result = reward ≥ 1.0` (`AE/task_step/task_steps/verifiers/run_container_unit_tests_verifier.py:331-358`).
- Judge output formats are RUBRIC_BINARY, RUBRIC_PARTIAL, TRAJECTORY_MISTAKES, RUBRIC_EVIDENCE and RUBRIC_EVIDENCE_WITH_MISTAKES (`judge_output_format.py:1198-1207`).
- No judge ensembling or majority vote was found in the verifiers.

**Snapshot API (exact).**

*`SnapshotEnvTaskStep`, type `"snapshot_env"`* (`AE/task_step/task_steps/snapshot_env.py:141-273`).
- Constructor fields: `id`, `version`, `env_id?`, `env_instance_id?`, `env_step_id?`, `gateway_url?`, `snapshot_id?`, `original_universe_artifact_id?`, `export_timeout_seconds=600`, `include_env_trajectory=False`, `depends_on?`, `fail_task_on_error=True`.
- Export:
  - Per service, prefer v1 `get_data` (DataPart → `.json`; FilePart → bytes/uri/object url), else legacy `GET /export-state` with a JSON first-byte guard (`:408-499`).
  - Concurrency 3. All-or-nothing: it fails the step if any service fails (`:501-560`).
  - Universe metadata is carried from the source.
- Output: `metadata["env_snapshotted_universes"][step.id] = {"id","version"}` (`:264-271`).
- Idempotence: `snapshot_id = derive_id(instance_id, f"snapshot-{step.id}")` (`:366-378`).

*`SnapshotAgentStateTaskStep`, type `"snapshot_agent_state"`* (`AE/task_step/task_steps/snapshot_agent_state.py:17-201`).
- Fields: `artifact_id`, `prompt_id`, `agent_name?`, `env_id?`, `universe_artifact_id?`, `timeout_seconds=120`.
- It calls the agent's `/ext/snapshot` via `capture_workspace` and appends `{id, version, bundle_object_url, source_agent_name, source_context_id}` to `metadata["agent_snapshots"]`.
- With both `env_id` and `universe_artifact_id` set, it also writes per-service state JSON under `services/<name>.json` and stores `metadata["snapshot_json_url"] = json.dumps({service: url})`.

*`SnapshotConfig` on `prompt_agent`* (`AE/task_step/snapshot_utils/snapshot_series.py:37-86`).
- Fields: `interval_seconds?`, `at_end=True`, `per_turn=False`, `env_id?`, `timeout_seconds=300`, `max_snapshots=24` (interior captures only; the final capture is extra).
- Rows: `{turn (1-based), id, version, bundle_object_url, source_agent_name, source_context_id, elapsed_s, captured_at_utc, is_final, capture_status, capture_reason, trajectory_object_url, env_universe_id, env_universe_version}` (`:545-571`).
- The final row's universe is also published into `env_snapshotted_universes[step_id]` so that `load_artifact` can consume it (`:593-610`).

*Restore, branch and resume.*
- `load_artifact(env_id, artifact_from_step_id=<snapshot step id>)` resolves via `_STEP_ARTIFACT_SOURCES = (collected_artifacts, env_snapshotted_universes)` (`AE/task_step/task_steps/load_artifact.py:28-31, 336-347`), with an optional `snapshot_after_load`.
- `reset_env` is best-effort `Env.reset()` (`AE/task_step/task_steps/reset_env.py:1-60`).
- `Task.run(start_step=…, end_step=…, context=…, instance_id=…)` runs a slice of the step DAG (`AE/task/task.py:390-457`). `retry_from_step_id` plus journal rollback `undo_steps` gives resumable retries (`:166-250, 327-336`).

---

## 3. Mechanism table

| Mechanism | Source path:line | Use in Crucible (M5/M6/harness) | DEPEND/PORT/PATTERN |
| --- | --- | --- | --- |
| 4-bucket F2P/P2P/F2F/P2F contingency | `SS/swesmith/harness/grading.py:40-92` | M5 CI-derived per-step milestones `transition_report(s_0, s_t)` | PORT (fix direction) |
| Eval pass/fail (missing = fail, XFAIL = pass) | `SS/swesmith/harness/grading.py:95-106` | M6 Φ test component; R5 anti-delete | PORT |
| Marker-sliced logs + ungradable states | `SS/swesmith/harness/grading.py:21-37`; `SS/swesmith/harness/utils.py:241-262` | M5 replay: ungradable ≠ failed → quarantine/C6 | PORT |
| Test-file revert after agent patch | `SS/swesmith/harness/utils.py:224-237`; `R2E/pipelines/pr_runtime.py:466-502` | Harness grading path; R5 | PORT |
| Flaky repair via gold re-eval | `SS/swesmith/harness/repair.py:122-215` | M5 milestone filter (K-run stability) | PATTERN |
| Profiles: test_cmd, timeouts, test-file heuristic | `SS/swesmith/profiles/base.py:104-118, 549-656` | M5 replay runner config per repo | PATTERN |
| gtest/ctest/catch2 parsers | `SS/swesmith/profiles/cpp.py:449-760` | GPU-migration CI parsing | PORT (drop synthetic names) |
| Procedural mutants (17 ops, C++ tables) | `SS/swesmith/bug_gen/procedural/base.py:58-145`; `SS/swesmith/bug_gen/procedural/cpp/operations.py:14-62` | C2 negatives, kill matrix, R3 | PORT (+ .cu ext) |
| LM-modify / LM-rewrite | `SS/configs/bug_gen/lm_modify.yml`; `SS/swesmith/bug_gen/llm/rewrite.py:95-155` | C2 semantic negatives; R5 plausible-wrong | PATTERN |
| PR-mirror (reverse-apply diff) | `SS/swesmith/bug_gen/mirror/prompts.py:1-76` | "Undo step k" trace-derived negatives | PATTERN |
| Combine multi-bug | `SS/swesmith/bug_gen/combine/same_file.py:1-60` | Multi-defect R3 negatives | PATTERN |
| Mutant validity (F2P>0 ∧ P2P>0) | `SS/swesmith/harness/gather.py:337-341` | Admit negatives to R3 suite | PORT |
| Leak-safe issue rules; F2P tests removed from branch | `SS/configs/issue_gen/ig_v2.yaml`; `SS/swesmith/harness/gather.py:539-556` | Task instructions from traces | PATTERN |
| Two-stage validation (test patch at base, ERROR→PASS) | `R2E/pipelines/pr_runtime_validate.py:206-327` | M5 per-step F2P via final-test overlay | PORT |
| `reward=f2p_rate·p2p_rate`, `command_resolved`, untracked | `R2E/pipelines/_pr_runtime_verifier.py:350-427` | M6 Φ_CI(s_t); tracker regressions | PORT |
| Fail-closed on hidden-test apply; unparseable → untrusted | `R2E/pipelines/pr_runtime.py:477-494`; `R2E/pipelines/_pr_runtime_verifier.py:460-484` | Harness + C6 | PORT |
| `execution_contrast` + strict JUnit | `R2E/quality/test_results.py:20-65` | M5 replay quarantine rules | PORT |
| Targeted tests; P2P cap 50 near-first | `R2E/pipelines/pr_runtime.py:720-781`; `R2E/pipelines/commit_runtime.py:396-405` | M5 cheap/medium check tiers | PORT |
| Commit mining filters / bugfix signal | `R2E/pipelines/commit_runtime.py:83-104, 467-499` | M3/M5 trace selection | PATTERN |
| Diff components (F1 files, region overlap ±5) | `R2E/pipelines/_pr_diff_verifier.py:164-256` | Tracker localization features (weight 0 in R) | PORT (subset) |
| No-op baseline calibration `(raw−b)/(1−b)` | `R2E/pipelines/pr_diff.py:397-434` | R2 normalization of any shaped score | PATTERN |
| Catastrophic size cap 0.40 | `R2E/pipelines/_pr_diff_verifier.py:436-437, 525-526` | Gate G for judge-heavy rubric scores | PATTERN |
| Control evidence nop×2=0 / oracle×4=1, hash-bound | `R2E/pipelines/recipes/codemidas/audit.py:196-219` | R1/R2 release gate per task version | PORT |
| SWE-Flow development_schedule | `R2E/pipelines/recipes/swe_flow/schedule.py:34-84` | M5 stage-induction prior; milestone ladder | PORT |
| Trusted-parent grader, uid 1001, killpg, exact set | `R2E/pipelines/recipes/swe_smith/grade.py:20-190` | Harness grading sandbox | PATTERN |
| Stub-fail / oracle-pass contrast | `R2E/pipelines/equivalence_tests.py:528-610` | Milestone non-triviality check at base | PATTERN |
| SQL state verifier + comparison ops | `EOPS/benchmark/verifier.py:81-127, 374-399` | Enterprise milestones (state predicates) | PATTERN |
| SQL-grounded LLM response judge (≥7/10) | `EOPS/benchmark/verifier.py:129-164, 412-483` | C4 candidate with factual grounding | PATTERN |
| tool_execution verifier | `EOPS/benchmark/verifier.py:166-193` | Diagnostic only (violates R6) | AVOID in R |
| All-pass + verifier-level pass rate; fresh DB/run | `EOPS/benchmark/executor.py:320-395, 565-640` | Outcome O vs dense milestone fraction | PATTERN |
| Decomposing planner (2–5 subtasks) | `EOPS/orchestrators/decomposing_planner.py:1-10` | Harness baseline; stage-label comparison | PATTERN |
| Step + judge + `continue_task` gating | `KAR/src/karotte/step.py:59-92`; `KAR/src/karotte/evaluation_runner.py:207-219` | Staged tasks; gate G per stage | PATTERN |
| Misbehavior → 0 vs infra → error | `KAR/src/karotte/student_misbehavior.py:11-18` | C6 mask vs agent-fault semantics | PATTERN |
| And/Or composite judges | `KAR/src/karotte/judges/_composite_judge.py:104-194` | Check composition in VerifierView | PATTERN |
| RubricJudge per-criterion YES/NO | `KAR/src/karotte/judges/rubric_judge.py:87-210` | C4 candidate item evaluation | PATTERN |
| Submission custody (kill, root-only copy, no links) | `KAR/docs/why-karotte.md:147-176`; `KAR/src/karotte/save_submission.py:46-243` | Harness grading hardening | PATTERN |
| `aggregate_score` with negative weights | `AE/task_step/task_steps/verifiers/scoring.py:18-40` | C4 composition; mistake penalties | DEPEND |
| Severity penalties scaled by pos_den; findings.step | `AE/task_step/task_steps/verifiers/judge_utils/judge_output_format.py:1042-1135, 192-194` | C2 localization; penalty channel | DEPEND |
| Negative-criterion inversion prompt | `AE/task_step/task_steps/verifiers/judge_utils/judge_output_format.py:76-100` | C4 negative rubric items | DEPEND |
| env_outcome_verifier `verify(mcp_url)->rows` | `AE/task_step/task_steps/verifiers/env_outcome_verifier.py:92-127` | Enterprise milestone checks on Worlds | DEPEND |
| aggregate_verifiers | `AE/task_step/task_steps/verifiers/aggregate_verifiers.py:63-90` | Union of check families → one score | DEPEND |
| snapshot_env (universe artifact, idempotent id) | `AE/task_step/task_steps/snapshot_env.py:141-273, 366-378, 501-560` | M1/M5 state at step boundary; C3 | DEPEND |
| snapshot_agent_state (+ per-service JSON) | `AE/task_step/task_steps/snapshot_agent_state.py:95-201` | C3 branch start state incl. agent context | DEPEND |
| SnapshotConfig per_turn / max 24 | `AE/task_step/snapshot_utils/snapshot_series.py:37-86, 545-610` | steps.jsonl dense channel capture | DEPEND |
| load_artifact(artifact_from_step_id) + Task.run(start_step) | `AE/task_step/task_steps/load_artifact.py:28-31, 336-347`; `AE/task/task.py:390-457` | C3 K-continuation rollouts | DEPEND |

---

## 4. Concrete port sketches for M5/M6

**4.1 CI-derived per-step milestone mining (M5),** composed from `SS` grading, `R2E` two-stage validation and execution_contrast.

```python
def mine_ci_milestones(states: list[WorldRef], final_tests: TestOverlay, runner, K=3):
    # overlay = hidden final test files applied onto each s_t  (R2E stage-A pattern)
    maps = [stable_status_map(runner, s.with_overlay(final_tests), K) for s in states]  # K-run, drop unstable (SS repair)
    if any(m.ungradable or m.has_synthetic_names for m in maps[::checkpoint]): quarantine()
    base, final = maps[0], maps[-1]
    F2P = {t for t in final if final[t]=="PASSED" and base.get(t) in ("FAILED","ERROR")}  # R2E :288-296
    P2P = {t for t in final if final[t]=="PASSED" and base.get(t)=="PASSED"}
    milestones = {}
    for t in F2P:
        k = first index with maps[k][t]=="PASSED" and all(maps[j][t]=="PASSED" for j>=k)  # monotone on good traces
        if k is None: reject(t, "non_monotone")          # oscillating test, not a milestone
        milestones.setdefault(k, set()).add(t)
    return milestones, P2P   # Φ_CI(s) = (Σ_{t∈F2P} w_t·pass_t / Σ w_t) · p2p_rate  (R2E reward)
```

Admission rules:
- *Non-trivial at base:* holds by construction, since F2P fails at s_0. Add a stub-fail/oracle-pass check for synthesized tests (`R2E/pipelines/equivalence_tests.py:564-608`).
- *Executable:* the test runs on a state with the same collection (`R2E/quality/test_results.py:58-59`).
- *Predictive:* the M6 §6.5 time-split AUC gate.

**4.2 C2 defect→check kill matrix.**
1. For each validated mutant m: apply it to the oracle final state s_T, and run the checks.
2. `kill[m][c] = c passes on s_T ∧ c fails on m(s_T)`.
3. Keep m iff some F2P/milestone check kills it and ≥1 P2P survives (`SS/swesmith/harness/gather.py:337-341`).
4. Mutant sources:
   - SWE-smith procedural, PORT with `.cu` added.
   - LM-rewrite of the functions touched in the trace, for plausible-wrong mutants.
   - PR-mirror "undo step k", which is checked against the mined milestone k (that milestone should flip).
5. R3: every kept mutant must have `R(m) ≈ 0` under the composed reward.

**4.3 Composed C4/penalty channel.** DEPEND on `aggregate_score(rows, WEIGHTED_AVERAGE)`:
- Positive rows: validated rubric items, with weights from AUC.
- Negative rows: anti-pattern items phrased so that score=1 means absent, plus judge `findings` with `weight = −penalty·pos_den`.
- Map each finding's `step` to `steps.jsonl` rows. Severity tables should be calibrated by Crucible (agentenv says to calibrate against human review: `AE/task_step/task_steps/verifiers/judge_utils/judge_output_format.py:1043-1047`).

**4.4 R1/R2 control receipt.** Per task version, store `{bundle_hash, nop:[r,r]=0, oracle:[r×4]=max, mutants:[…]≈0, result_sha256[]}` and refuse admission otherwise (`R2E/pipelines/recipes/codemidas/audit.py:196-219`).

---

## 5. Open questions and risks

1. **swebench `get_resolution_status` is not vendored.** Its exact source was not found locally. We rely on R2E's re-statement (`R2E/reward.py:161-169`). If exact SWE-bench parity matters, vendor it.
2. **SWE-smith pregold/postgold direction is inconsistent** across `valid.py`, `gather.py` and `scripts/bug_gen_modal.py`. Our port must define direction by state names and test it. Do not copy the variable names.
3. **The final-test overlay can be infeasible at early steps.** At s_t the overlay may not even compile (C++/CUDA, renamed APIs). R2E only handles the single base→gold case and fails closed on patch-apply errors. For multi-step traces we need per-step "compiles?" as its own cheap milestone, separate from test milestones (a build-family check).
4. **No snapshot of GPU or process state.** agentenv snapshots export *service* state (`get_data` / `export-state`) and agent workspace tarballs, not GPU or process state. For task04-style GPU worlds, C3 branching must re-materialize from git state plus a build cache. Branch cost is dominated by rebuilds.
5. **`Task.run(start_step)` replays a static step DAG.** It is not a mid-conversation branch. C3 continuations from turn t need the agent's conversation context restored (`snapshot_agent_state` captures the transcript, but resuming an external harness from it is harness-specific). Open: which harnesses (mini-swe-agent, Claude Code, Codex) accept an injected history.
6. **EOPS exact-row COUNT predicates are brittle to equivalent states** (e.g. an extra benign column update). Its value extraction collapses shapes (`EOPS/benchmark/verifier.py:291-372`). Crucible enterprise milestones should prefer typed field assertions and add collateral-damage checks; EOPS has essentially none (1/102).
7. **LLM-judge-weighted rewards dominate R2E pr_diff (0.5).** They also back EOPS `response_check` and agentenv rubrics. None ensembles judges. C4 items must pass the time-split AUC gate before getting weight, and the C5 PRM stays baseline-only.
8. **Karotte "run score = last step's score" and EOPS "all-pass" both discard dense information.** Crucible's gated R keeps both O and Φ. Make sure the C6 infra mask does not reuse Karotte's semantics, where a judge failure scores 0 and halts.
9. **Known residual hacks to red-team in Crucible Worlds:**
   - equivalence-task `reference_<name>` in the image (`R2E/pipelines/equivalence_tests.py:195-216`);
   - `bash_cmd_succeeds` with a replaced `bash`;
   - overwritable staged grader scripts;
   - FIFO/symlink answer files;
   - pre-written reward files (R2E deletes them: `R2E/pipelines/pr_diff.py:283-287`).
10. **The mutant likelihood/seed defaults** (0.2 or 0.5, seed 24) are sampling knobs, not validated; mutant yield per repo is unknown for CUDA code. Budget a pilot.
