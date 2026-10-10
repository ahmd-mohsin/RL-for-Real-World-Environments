# M7 research 01 — hidden verifiers and isolation: what to take from each framework

Date: 2026-10-09. Scope: ARCHITECTURE §4 M7 `verify`, DESIGN §8 (Layer 4), §9 (A1–A13), Appendix B.10.
Method: a read-only survey of 12 local codebases plus our four seed graders. I re-read every load-bearing claim marked ✓
myself; the other line references come from the same survey (file:line at the pinned local checkout).
Earlier notes that this report builds on are `docs/research/m5-m6/02b-harbor-fineenvs.md` (§2, §5),
`04-nemogym-mimo-verl.md` (§2.5–2.6, §4.6) and `05-swesmith-r2e-eops-karotte-agentenv.md` (§2.3–2.5).

Path prefixes:

| Prefix | Path | License |
|---|---|---|
| `H/` | `RL-Environments/06-frameworks-and-env-code/harbor-framework-harbor` (0.23.0, a2fafbc) | Apache-2.0 |
| `V/` | `…/primeintellect-ai-verifiers/verifiers/v1` (ae819de; v1 stack only, no legacy `Rubric`) | MIT |
| `PE/` | `…/primeintellect-ai-prime-environments` (legacy verifiers API) | Apache-2.0 |
| `RE/` | `…/primeintellect-ai-research-environments` (verifiers v1) | Apache-2.0 |
| `NG/` | `…/nvidia-nemo-gym` | Apache-2.0 |
| `EOG/` | `…/servicenow-enterpriseops-gym` | Apache-2.0 |
| `MM/` | `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss`; `ENV/` = `MM/general/envs/s3k_0004_…_rl_001` | apache-2.0 (README front matter; no LICENSE file) |
| `MV/` | `…/xiaomimimo-verl`; `MA/` = `MV/third_party/mimoagent-osr/src/mimoagent` | Apache-2.0; mimoagent-osr MIT |
| `OE/` | `…/meta-pytorch-openenv` | BSD-3-Clause |
| `SS/` | `…/swe-bench-swe-smith/swesmith` | MIT |
| `R2E/` | `…/huggingface-repo2rlenv/src/repo2rlenv` | Apache-2.0 |
| `K/` | `…/preferencemodel-karotte` | MIT |
| `AE/` | `agentenv-framework/src/agent_env` | Apache-2.0 |
| `MT/` | `migration_tasks/` (our seeds) | ours |

---

## 0. TL;DR

1. **No framework enforces P26 out of the box.** The best trusted-channel grader in the survey is Repo2RLEnv's
   per-task `recipes/swe_smith/grade.py` + `test_driver.py`, at about 240 lines (Apache-2.0). It pre-writes reward 0,
   validates the submission, runs the tests as uid 1001 with `-I` and a scrubbed env, `killpg`s descendants, reads JUnit
   XML from a grader-owned temp dir, and passes only on `rc == 0 ∧ expected-set == passed-set`. **PORT it** as the
   skeleton of `verify/grader.py` + `patterns/hidden_tests.py`. Our own task04 grader supplies the rest:
   reference/candidate split, sealing, fresh secret seed, perf rounds. The existing M1 guard (Karotte-derived)
   provides privilege drop and quiesce.
2. **Harbor: DEPEND only on separate mode, collect hooks and regrade; never on shared mode.**
   - In single-step shared mode, `_run_shared_verifier` goes straight to `verify()` (`H/src/harbor/trial/trial.py:852-883` ✓).
     `/logs/verifier` is a world-writable bind mount during the agent phase, and `reward.json` wins over `reward.txt`
     (`H/src/harbor/verifier/verifier.py:257-266` ✓). An agent-planted `reward.json` therefore beats `test.sh`.
   - Harbor also ignores test.sh's exit code, runs the verifier as the image's default user (usually root), kills no
     agent processes, and silently drops unknown TOML keys.
3. **DESIGN §4.3.1's nested `reward.json` is not readable by Harbor or verifiers.** Harbor raises
   `VerifierOutputParseError` on any non-numeric value (`H/…/verifier.py:104-121` ✓). verifiers' adapter is
   `float | dict[str, float]`, strict, with no inf/NaN (`V/tasksets/harbor/taskset.py:52-55` ✓). The `RewardJson`
   docstring ("stays readable by Harbor", `src/crucible/core/schema/reward.py:3`) is wrong for its nested
   `gates`/`rubric`/`infra` fields. **Fix:** write a flat numeric `reward.json` (`reward`, `outcome`, `gate/<g>`,
   `progress`, `infra/masked`, …) and put the rich document in `crucible_reward.json`.
4. **Infra vs candidate attribution is the core design problem, and every framework gets it wrong in one direction.**
   - *Errors on candidate-caused failures, which trainers drop:* verifiers' `IsolatedVerifierEnv` re-raises after
     retries (`V/envs/isolated_verifier/env.py:179-182`), and verifiers' Harbor taskset raises `TaskError` when no
     reward file exists (`V/tasksets/harbor/taskset.py:399-403` ✓). A candidate that crashes the grader escapes the 0.
   - *Infra errors scored as 0:* NeMo `swemer_oml` returns `reward 0.0` with `evaluation_completed=False` but never
     sets `mask_sample` (`NG/resources_servers/swemer_oml/app.py:383-402` ✓). Our seeds lump infra and malformed
     output into one `error` gate.
   - *Crucible rule:* candidate-attributable failures score 0. Grader-precondition failures are masked. Any
     unattributed exception writes 0 plus `infra/masked=1` plus an alarm. An A13 "crash the grader" mutant must score
     0, not masked.
5. **Timing from inside the candidate's process cannot be trusted. No surveyed perf grader does better.**
   - Every perf grader found times inside or next to candidate code: kernelbench CUDA events in the exec'd process
     (`PE/environments/kernelbench/utils/eval.py:467-522`), and our task03/task04 timers read from candidate-produced
     rows or `result.json`.
   - **`patterns/perf.py` must:**
     - gate on parent-observed wall time (or an external profiler that writes to a grader dir);
     - give every rep fresh inputs whose outputs are verified;
     - interleave reference and candidate on the same node;
     - apply a load guard and a spread check.
6. **For isolation, PATTERN NeMo `swemer_oml` and the MiMo sidecar.** Only the patch/artifacts cross; the agent box
   is stopped; hidden material exists only in the fresh grading box; edits to test-surface files are stripped; the
   verifier is forbidden in `main`; the verifier gets the workspace read-only.
   **PORT verifiers' `artifacts._validate_restore`** (MIT) for untrusted artifact archives, and **pi_deepswe's
   `PROTECTED` regex** (Apache-2.0) for resetting the test surface (conftest, sitecustomize, runner configs).
7. **Validation:**
   - **PATTERN** OpenEnv RFC 008's gap contract (`reward(oracle) ≥ max−tol ∧ reward(reset) = floor ∧ max−floor ≥ margin`,
     "no oracle → FAIL", "floor is measured, not declared"). Only its static level is implemented
     (`OE/src/openenv/validation/runner.py:154`).
   - **PATTERN** NeMo `verifier_fixture` (full/zero/malformed/determinism, a fresh server per case, finite-and-in-range).
   - **PATTERN** `vf-validate` gold/no-op (`V/cli/validate.py:242-260`).
   - All three feed `calibrate.py` and M9 A1/A2/A4.
8. **State predicates:**
   - **PATTERN** EOG `database_state` (`query`, `expected_value`, `comparison_type`) and MiMo rule items
     (`check_code` + `fn` → `{score, detail}`). Fix their holes: duplicate names overwrite results; `all([])` passes;
     an unknown `gym_name` is skipped; MCP error text becomes the "value"; a substring `random` ban; the last
     `___R___` stdout line wins.
   - **SKIP** EOG `response_check` as a gate: the policy's own LLM judges itself, with the raw text inlined
     (`EOG/benchmark/executor.py:126`).
9. **Corrections to ARCHITECTURE M7 "Takes":**
   - MiMo has **no `/verify` endpoint**. Grading is a `run_verify.py` command with a "no parseable reward ⇒ masked"
     contract (`ENV/run_verify.py:102-107`).
   - verifiers' `IsolatedVerifierEnv` is an export target (interop/verifiers), not a runtime dependency of M7.
   - mimoagent `antihack.py` is off by default, off for the code-RL configs, absent for blackbox harnesses, and
     regex-bypassable. Use it as a P27 recall list only.
   - Add PORTs:
     - Repo2RLEnv `grade.py`/`test_driver.py`;
     - verifiers `_validate_restore`;
     - pi_deepswe `PROTECTED`;
     - NeMo `verifier_fixture`, into M9's harness.

---

## 1. Our seed graders: what M7 generalizes, and their holes

M7's exit gate re-expresses these graders exactly, so the patterns must reproduce their *results*. They must not
reproduce their *holes*: those are Appendix B.10 defects, and M9 A13 must catch them.

### 1.1 task04 (`MT/task04_pcl_gpu_cuda12/tests/grade.py`, `cases.py`): the template

| Mechanism | Where | Generalize as |
|---|---|---|
| Privilege drop: `setpriv --reuid=65534 --regid=65534 --clear-groups` | `grade.py:75-76` | `TrustedRunner` (on the M1 guard's `demote`: setsid, `PR_SET_NO_NEW_PRIVS`, `RLIMIT_NPROC`, `oom_score_adj`; `src/crucible/world/guard/guard.py:195-226`) |
| Kill candidate cohort before reference measurements: `pkill -9 -u` ×3 + `pgrep` confirm | `grade.py:79-85, 105-106` | guard `quiesce` (`kill(-1)` from a forked child at the uid, `/proc` sweep; `guard.py:280-347`). No `pkill` binary dependency. |
| Root-only references computed at grading time by the original build | `ref()` `grade.py:198-206`; `R.mkdir(mode=0o700)` `:263-264` | `patterns/differential.Reference` |
| Candidate run unprivileged into a uid-owned dir, then **sealed** (chown root, strip write bits) before any read | `cand()` `:208-215`; `seal()` `:189-195`; `udir()` `:180-183` | `isolation/seal.py` + `TrustedRunner.run(out_dir=…)` |
| Pristine consumers built from root-owned originals against the candidate's install | `build()` `:241-259` | `patterns/differential.Consumer` (P26 "grader-owned harness") |
| Reference fingerprints (`--record-fingerprints`) checked every grade (stale image / self-consistency) | `references()` `:262-295` | `isolation/leak_audit.image_fingerprint` + `calibrate.reference_self_consistency` |
| Secret seed drawn at grade time, revealed only after the candidate ran | `gate_fresh()` `:389-410`; `cases.fresh_workloads` `cases.py:76-90` | `patterns/fresh.py` |
| Concurrency equivalence: concurrent run vs the **candidate's own solo** runs, bitwise minus timing keys | `gate_concurrency()` `:339-387`; `compare_bitwise_run` `cases.py:277-284` | `patterns/invariants.concurrency_equivalence` |
| Two-tier tolerance (STRICT / BIFURCATED), calibrated on 40 natural + 40 induced draws | `cases.py:135-149, 260-274` | `calibrate.py` → tolerances in `calibration.json`, not code |
| Sanitizers (racecheck/synccheck/memcheck) with `--error-exitcode 97` | `gate_sanitizer()` `:412-433` | `patterns/sanitizer.py` (under `invariants`) |
| Binary checks (`cuobjdump --list-elf/--list-ptx`, `readelf -d` NEEDED/RUNPATH) | `binary_checks()` `:133-164` | `patterns/static.binary` |
| Static regex over comment-stripped sources | `static_checks()` `:118-130`; `TEXREF` `:63-72` | `patterns/static.source` |
| Perf: interleaved ref/cand rounds, best-of-k (min), one re-measure on failure | `perf_round()` `:447-491`; `gate_perf()` `:493-515` | `patterns/perf.py` |
| Gates all-or-nothing; any exception → `error` gate, reward 0 | `record()` `:174-178`; `main()` `:573-588` | `grader.py` |

**Holes (B.10 plus new findings):**
- **Timing is in-process.**
  - `frame_ms_median`, `track_wall_ms`, `mc_ms` and `ms_median` are read from `result.json`, which the pristine
    consumer wrote *in a process that links the candidate library* (`:486-491, 470`).
  - Candidate code can alter the consumer's timers or outputs.
- **Identical inputs every round** (`perf_round` reuses `cases.KINFU[...]`), so memoization works.
- **One `UNPRIV_HOME` (= TMPDIR) for the whole grade** (`:89-92, 520`), so state can be cached and replayed across
  invocations.
- **No load guard and no spread gate.** Spread is recorded (`last`) but not enforced.
- **Sanitizer verdict parsed from combined candidate + sanitizer output** (`:424-427`). A candidate can print a fake
  `RACECHECK SUMMARY: 0 hazards…` before the real one: `re.search` takes the first match. The `rc == 0` with
  `--error-exitcode` mitigates this. Fix: give the sanitizer its own log file (`--log-file` in a grader dir) and
  parse only that.
- **New: consumers are configured through the candidate's exported CMake package** (`-DCMAKE_PREFIX_PATH={install}`,
  `:249`). The candidate's `PCLConfig.cmake` is candidate code that runs in the consumer's configure step. It can add
  `INTERFACE_COMPILE_DEFINITIONS`/`-include` flags that rewrite the consumer: the same class as
  `DOCTEST_CONFIG_DISABLE`.
  - Fix: export `compile_commands.json` for the consumer build and allowlist its flags, or build the consumers with
    explicit grader-owned include/link flags.
- **Docstring drift:** "perf <= 1.10x the original" (`grade.py:24`) vs `perf_thresholds.json` ratios 0.42–0.75. Only
  the docstring is stale, but it is a P20 smell, so A7 should diff it.
- **KinFu-LS fallback to BIFURCATED whenever the strict bound fails** (`cases.py:143-146`). This is B.10's open P20
  defect: the relaxed tier is reachable by *any* failing run, not only by runs that bifurcated.
- **Infra and malformed candidate output collapse into one `error` gate** (`:574-577`).
  - Some compare calls are unguarded, e.g. `compare_kinfu` in `gate_fresh` `:420`; a `KeyError` on a malformed
    `result.json` becomes "error".
  - `test.sh` creates `reward.txt` via a shell redirect, so it is left **empty** if `report.json` is missing
    (`test.sh:16`).

### 1.2 task03 (`MT/task03_mdlite_pallas/tests/grade.py`, `fresh_cases.py`)

**Good:**
- The CUDA original and the pristine consumers are installed by root **before** any candidate code runs (`:210-220`).
- The candidate is installed unprivileged with no CUDA compiler reachable (`:233-241`).
- The fresh-seed reference goes into a 0700 dir (`:251-264`).
- The perf benchmark must reproduce the original's end state (pot/temperature/steps) before its timing counts
  (`:162-168`). **Port this "perf with end-state check" idea.**

**Holes:**
- **`pytest_gate` passes on `rc == 0`** (`:121-129`), so `os._exit(0)` passes.
- **Candidate site first:** `pp = f"{site}:{cons}:{hid}"` (`:242`). The candidate can shadow `mdbuild`/`mdkit` and
  the hidden test helpers.
- **The original engine is readable:** `ref_site` sits under `work/` (0755) and is root-built with default perms
  (`:194-195, 211`). The candidate can import it and delegate.
- **Perf rows come from candidate-process stdout.** `parse_json_lines`, last row wins per case (`:132-141, 154-156`);
  timing is in-process.
- **`kill_unpriv` sends one `pkill` with no confirmation** (`:65-67`).
- **An exception before `report["reward"]` crashes the final `print`** (`:289, 297`), leaving `reward.txt` empty.
- **`fresh_cases.build`** draws boxes smaller than 2× cutoff on ~18% of seeds (B.10).
- **The collect hook has no `|| true` on `git add`** (`task.toml` `[[verifier.collect]]`). This is B.5's failure mode.

### 1.3 task02 (`MT/task02_pcops_triton/tests/grade.py`)

Everything runs as root:
- `pip install` of the candidate (`:156-161`);
- pytest (`:87-92`);
- the benchmark, straight from `/tests/hidden/bench_pcops.py` (`:111`).

The CUDA original is built *after* candidate code has run as root (`:188-192`). Perf rows come from stdout (`:95-104`).
This is the B.10 "task02" defect in full. **Nothing here is worth porting** beyond the static rules and the
install-without-compiler idea.

### 1.4 task01v2 (`MT/task01v2_tessera_omp/tests/grade.py`)

**Holes:**
- **The hidden suite is copied into the candidate tree and built by the candidate's CMake** (`:269-272, 134-141`).
  So `DOCTEST_CONFIG_DISABLE` passes.
- **doctest is judged by exit code** (`run_doctest` `:149-152`).
- **Offload-probe rows and perf rows are parsed from stdout of candidate-built binaries** (`:194-208, 220-244`).
- **Golden frames are unpacked into the candidate tree** (`:274`).
- **All of it runs as root.**

**Port:** the string-literal-aware C comment stripper (`strip_c_comments` `:63-84`). It is better than task04's
regex version.

---

## 2. Harbor (`H/`, Apache-2.0, 0.23.0)

**Files and interfaces**
- **Verifier** `src/harbor/verifier/verifier.py` (268 lines); `BaseVerifier` `verifier/base.py:14-43`; custom
  verifier through `import_path` (`verifier/factory.py:45-99`).
- **Exceptions** (`verifier.py:20-37`): `AddTestsDirError`, `VerifierOutputParseError`, `DownloadVerifierDirError`,
  `RewardFileNotFoundError`, `RewardFileEmptyError`; `VerifierTimeoutError` (`trial/errors.py:12-13`).
- **Paths** (`models/trial/paths.py:39-48`): `/logs/verifier/{reward.txt,reward.json}`, `/tests`, `/solution`,
  `/logs/artifacts`; stdout goes to `/logs/verifier/test-stdout.txt`.
- **`verify()`** (`:165-268`):
  - uploads `tests/` unless `skip_tests_upload` is set;
  - merges env task < step < job, with `${VAR:-default}` resolution (`utils/env.py:94-124`);
  - runs `(/tests/test.sh) > test-stdout.txt 2>&1` as `environment.default_user`. **The exit code is never
    inspected** (`:221-232`).
  - The timeout is an outer `asyncio.wait_for` (`trial.py:881-883`).
- **Reward files:**
  - `reward.json` takes precedence over `reward.txt` (`:257-266` ✓).
  - `reward.txt`: 0 bytes → Empty; non-float, NaN or inf → ParseError (`:67-87`).
  - `reward.json`: every value must be int/float and finite (`:104-121` ✓). **Bools pass**, `{}` passes, there is no
    range check, and a bare number fails later in pydantic.
  - `VerifierResult{rewards: dict[str, float|int] | None}` (`models/verifier/result.py:4-5`).
  - A verifier exception → `verifier_result=None` → metrics count it as 0 (`metrics/base.py:14-35`).
- **Separate mode** (`models/task/config.py:555-611`; resolution `models/task/verifier_mode.py:10-64`):
  - a fresh env per verify pass (`trial.py:885-996`), session `"{trial}__verifier__{key}"`;
  - `empty_dirs([/logs/verifier], chmod=True)` runs before grading (`trial.py:915` ✓);
  - only declared artifacts + `/logs/artifacts` cross, restored to their original source paths
    (`trial/artifact_handler.py:169-213`);
  - `skip_tests_upload=True`, with tests baked into the image built from `tests/` (`trial.py:934, 1011-1021`);
  - the agent env is stopped first for single-step tasks (`trial/single_step.py:37-55`), but only on the last step
    for multi-step tasks (`trial/multi_step.py:124-142`).
- **`[[verifier.collect]]`:**
  - schema `{command, service="main", timeout_sec=60, user}` (`config.py:726-757`);
  - runs in the **agent** env's services (`trial.py:1417-1525`);
  - best-effort: a non-zero exit only logs a warning (`:1432-1462`);
  - main-service hooks run before main stops, sidecar hooks after, as "a channel the agent cannot write to"
    (`:1474-1478`).
- **`[[steps]]`:**
  - `min_reward: float | dict[str, float]` (`config.py:760-791`);
  - gating treats a missing result or key as `-inf` (`multi_step.py:195-242`);
  - aggregation is `mean` or `final` (`config.py:794-824`, `trial.py:790-824`);
  - shared-mode `/tests` and `/logs/verifier` are wiped between steps (`trial.py:771-776, 826-833`).
  ```python
  # H/src/harbor/trial/multi_step.py (Apache-2.0)
  thresholds = ({"reward": min_reward} if isinstance(min_reward, (int, float)) else min_reward)
  for key, threshold in thresholds.items():
      actual = rewards.get(key, float("-inf")) if rewards else float("-inf")
      if actual < threshold:
          return f"{key}={actual} below min_reward {threshold}"
  return None
  ```
- **Regrade** (`trial/regrade.py:1-13, 432-560`): replays recorded artifacts into separate verifiers, and refuses to
  grade if the declared inputs were not collected (`collected/empty/failed/skipped` manifest).
- **`harbor check`** (`analyze/checker.py`; rubric `cli/quality_checker/default-rubric.toml`): 11 criteria —
  `behavior_in_task_description`, `behavior_in_tests`, `informative_test_structure`, `anti_cheating_measures`,
  `structured_data_schema`, `pinned_dependencies`, `typos`, `tests_or_solution_in_image`, `test_deps_in_image`,
  `hardcoded_solution`, `file_reference_mentioned`.
  - It runs as an LLM-agent Harbor job, and its output is `{criterion: {outcome: pass|fail|not_applicable, explanation}}`.
- **`harbor analyze`** (`analyze/prompts/analyze-rubric.toml`): criteria `reward_hacking` ("modifications to test
  files…, writes to /logs/verifier/reward.txt or reward.json, accessing… solution/") and `task_specification`.
  It is advisory only.
- **Oracle/Nop agents:**
  - `agents/oracle.py:19-156` uploads `/solution` and runs `solve.sh` as root;
  - `agents/nop.py:9-32` does nothing.
  - There is **no built-in "nop scores 0" check**.
- **Rewardkit** (`packages/rewardkit`, Apache-2.0): overlayfs workspace isolation (`isolation.py:111-126`) and a
  judge "guard" criterion ("text addressed to the evaluator", `judges.py:382-400`).

**Good:** separate mode with artifacts-only hand-off; regrade with a coverage manifest; typed reward errors;
finite-number check; `min_reward` per key; the `check`/`analyze` rubrics.

**Holes (trust in verifier output):**
1. **Single-step shared mode never clears `/logs/verifier` or `/tests`.**
   - Agent-written `reward.json` beats `test.sh`'s `reward.txt`.
   - A planted `/tests/conftest.py` survives the upload merge (`trial.py:852-883` ✓, `verifier.py:179-183`).
   - This is P23's finding, confirmed in code.
2. **No process kill** between agent and verifier in shared mode. A daemon can rewrite `reward.*` after `test.sh`.
3. **Multi-step separate mode keeps the agent env up for non-final steps.** It bind-mounts the *same host*
   `verifier/` dir as the verifier env (`trial.py:1003-1009` vs `1886-1890`), so a leftover agent process can write
   `reward.json` after `empty_dirs`.
4. **`test.sh` exit code ignored**; the verifier user defaults to the image's default (root, `config.py:565-568`).
5. **Bools and `{}` are accepted, and nested values are rejected.** The latter is why DESIGN §4.3.1 must flatten.
6. **Collect failures are only warnings**, and main-service hooks run on agent-controlled binaries.
7. **Unknown TOML keys are silently ignored** (no `extra="forbid"` in `models/task/config.py`). Already B.7.
8. **Step verifier timeout bug:** `StepConfig.verifier.timeout_sec` defaults to 600, never `None`, so the task-level
   timeout is ignored for steps (`trial.py:835-845`).
9. **check/analyze run in shared-mode wrapper tasks**, so the reviewer agent could write `reward.txt` itself. Treat
   them as advisory.

**Recommendation**

| M7 piece | Take |
|---|---|
| isolation/separate verifier env | **DEPEND** (separate mode only). `interop/harbor` must refuse `environment_mode != "separate"` for Crucible tasks, set `[verifier].user` to root explicitly, and give the verifier a prebuilt `docker_image` (B.7). |
| isolation/collect hooks | **DEPEND** on `[[verifier.collect]]` as transport. Generate the hook from `isolation/collect.py`: tolerant `git add`, `--binary`, and a marker/manifest so an empty patch is classifiable (B.5, A11). |
| tiers (5% full-tier audit) | **DEPEND** on `regrade` to re-grade stored artifacts with the full tier. |
| grader.py reward I/O | **PATTERN**: typed errors and the finite check. Fix: reject bools and empty dicts, flat numeric only, pre-write and atomically replace. |
| multi-step gating | **DEPEND** on `[[steps]] min_reward` for staged tasks; per-key dict thresholds map onto `gate/<g>` keys. |
| calibrate / A7 | **DEPEND** on the `harbor check` rubric (M9 A7) and `analyze` `reward_hacking` (M9 A3 cheat review), advisory only (ARCH rule 3). |
| Shared mode, Oracle-only validation | **SKIP**. |

---

## 3. verifiers (`V/`, MIT)

**Files and interfaces**
- **`IsolatedVerifierEnv`** (`envs/isolated_verifier/env.py`, 182 lines):
  - `VerifierConfig{runtime, env, retries=2}` (`:29-50`).
  - `run()` refuses model-backed judges and validates the verifier config *before* solving (`:63-70`).
  - The solver runs with `task.defer_scoring()` and `collect_artifacts=True`.
  - `stage_verifier()` = `task.setup` → `vf.restore(artifacts)` → `task.stage_verifier` (trusted, verifier-only
    inputs) (`:111-119`).
  - `grade()` retries over fresh runtimes; it deep-copies the task and trace per attempt, uses separate setup and
    scoring deadlines with teardown outside them, and does `raise last` when attempts run out (`:124-182`).
  - It refuses subprocess runtimes (`:82-87`).
- **Artifacts** (`utils/artifacts.py`):
  - `MAX_ARTIFACT_BYTES=32 MiB` (`:23`);
  - a missing declared path **fails** the rollout instead of grading partial state (`:124-128`);
  - `restore()` treats archives as untrusted, validates them on the host, `rm -rf`s the roots, then extracts
    (`:139-177` ✓);
  - `_validate_restore` rejects members outside the root, duplicates, unsafe hardlinks and symlink traversal
    (`:240-333` ✓);
  - special files are dropped (`:212-237`).
- **Harbor taskset** (`tasksets/harbor/taskset.py`):
  - `REWARD_JSON_ADAPTER` is strict `float | dict[str,float]` with min_length 1, no inf/NaN, 1 MiB cap (`:50-55` ✓).
  - `stage_verifier` refuses artifacts that restore into `/tests`, then wipes and re-stages tests, deleting
    `reward.json`/`reward.txt` first, and raises if that fails (`:318-355` ✓).
  - `run_verifier` treats the reward file as authoritative even on a nonzero exit; it raises `TaskError` if there
    is no valid file (`:370-403` ✓).
- **Decorators** (`utils/decorators.py`):
  - `@vf.metric(priority)`: recorded, not summed, no weight (`:105-115`);
  - `@vf.reward(weight=1.0, priority)` (`:118-131`);
  - `seed()` pre-fills expected keys with `None`, so "didn't run" ≠ 0 (`:48-54`).
  - `Trace.reward = Σ score·weight`, unnormalized (`trace.py:134-141, 472-474`).
- **Score exceptions:** re-raised as `TaskError` (`errors.py:80-98`), giving `trace.ok=False`. The errored trace is
  **excluded** from the error-corrected mean (`utils/format.py:55-70`). A scoring timeout is "a stop", so unscored
  rewards stay None and count as 0 (`rollout.py:211-215`).
- **Judges:**
  - `judge_verdict` raises "so judge failures are not scored against the model" (`judge.py:102-107`);
  - `RubricJudge` uses weighted criteria with worst→best choices, fences the reference with a longer backtick run,
    and treats a malformed verdict as an error (`judges/rubric.py:186-271`).
- **`vf-validate`** (`cli/validate.py`): modes `--only-setup|--only-gold|--only-noop`, with a fresh runtime per check.
  - **Gold** = `task.validate(runtime)`, which defaults to None, i.e. "unchecked" (`task.py:195-197`).
  - **No-op** = score the untouched box; valid iff `reward < 1.0` (`:51-53, 247-260`).
- **TaskData** (`task.py:81-135`):
  - `agent_user`: the harness runs as this user; setup, collection and grading keep the default user (`:102-105`);
  - `resources.gpu: "A100[:n]"` (`:51-63`); `timeout{setup,agent,finalize,scoring}`;
  - `runtime_env()` holds untraced secrets.

**Good:** defer-score → collect → destroy → fresh box → validated restore → trusted stage → score; a hard fail on a
missing artifact; wiping the reward file before `test.sh`; None-seeded signals; metrics vs weighted rewards.

**Holes:**
- **Candidate-caused grader crashes become errors, not 0s**, and errors are dropped from means. This is a hack route
  if a trainer masks errors.
- **Restored artifacts are validated for paths, not semantics.** A restored `conftest.py`/`sitecustomize.py`/`.pth`
  executes in the verifier box. Its `test.sh` runs as the box's default user (root) unless the script drops privileges.
- **`Runtime.run` has no per-call timeout** (`runtimes/base.py:207-209`).
- **The gold check is opt-in**, and the no-op check tolerates partial credit.

**Recommendation**

| M7 piece | Take |
|---|---|
| isolation/separate env | **PATTERN** for the lifecycle (defer, collect, destroy, fresh, restore, stage, score, per-attempt deep copy, setup/scoring deadlines). **DEPEND** only in `interop/verifiers` export (`HarborEnv` already lowers `environment_mode="separate"` onto it). |
| isolation/collect (artifact validation) | **PORT** `restore` + `_validate_restore` + `_drop_special_files` (MIT; ~150 lines into `_vendor/verifiers_artifacts/` with LICENSE + ORIGIN.md). |
| isolation/staging | **PATTERN**: `stage_tests` wipe of reward files and `/tests`; refuse artifacts under hidden paths. |
| grader.py reward parsing | **PORT** the strict adapter idea (strict, no inf/NaN, size cap, min_length 1) for the flat `reward.json` reader in `interop` and `runtime`. |
| judge_item | **PATTERN**: `RubricJudge` parse discipline (malformed → error → *masked*, never 0), reference fencing. |
| calibrate (gold/no-op) | **PATTERN** `vf-validate`; we make gold mandatory (RFC 008 "no oracle → FAIL") and require no-op = 0 exactly (A2), not just < 1. |
| `@vf.reward`/`@vf.metric` | **PATTERN** in M11's export only: `reward` → `@vf.reward(weight=1)`, `gate/*` and `progress` → `@vf.metric`. |

---

## 4. prime-environments / research-environments (`PE/`, `RE/`, Apache-2.0)

| Example | Where | What it does | Holes |
|---|---|---|---|
| **pi_deepswe**, the strongest anti-tamper grader | `RE/environments/swe/pi_deepswe/pi_deepswe/verifier.py` | **`PROTECTED` regex** (`:20-26`): `tests?/`, `__tests__`, fixtures, `test_*.py`, `conftest.py`, **`sitecustomize.py`, `usercustomize.py`**, `pytest.ini`, `tox.ini`, `test.sh`, `package.json`, `*_test.go`, `*.test/spec.[jt]s`, jest/vitest/mocha configs.<br>`prepare()` resets protected paths to HEAD and removes model-added test files (`:137-159`).<br>Rust: restores `#[cfg(test)]` items in place and fails if an edit crosses an item boundary (`:105-134`).<br>`assess()`: missing tests = failure; separate `evaluation_valid` flag; `reward = int(f2p_total>0 ∧ ¬build_failure ∧ ¬failed)` (`:377-450`).<br>Wipes `/logs/verifier` first ("image leftovers are not evidence", `:594-599`).<br>Retries only all-timeout attempts, keeping every attempt.<br>Invalid evidence → evaluation error (`taskset.py:147-165`). | Still parses test output text. |
| **r2e_gym** | `RE/environments/swe/r2e_gym/r2e_gym/taskset.py` | Hidden tests tarred to the host and `rm -rf` before the agent runs (`:256-278`), restored at scoring (`:280-299`).<br>Exact status-map equality to `expected_output_json` (`:140-151`).<br>`validate()` applies the gold patch and runs the *same* reward (`:239-244`). | Grades in the agent's box; `run_tests.sh` lives in the agent-writable repo (`:113, 236`); pytest summary parsed from stdout. |
| **swerebench_v2** | `RE/environments/swe/swerebench_v2/swerebench_v2/taskset.py` | Reverts every test_patch path to base, or `rm`s it (`:132-142`).<br>Fail-closed parsing: empty, unknown parser, exception or empty map → 0.0 (`:229-253`).<br>Exit code > 1 raises. | Markers can be printed by candidate code. |
| **kernelbench**, the only speedup grader | `PE/environments/kernelbench/utils/eval.py:467-522, 544-592`; `kernelbench.py:198-307` | `torch.cuda.Event` timing, 3 warmups + 10 trials, mean.<br>5-seed correctness at `allclose(1e-2)`.<br>Fail-closed zeros on parse, Modal or baseline errors. | Timing is **in the exec'd candidate process** (`:109-110`), so Events can be monkeypatched or side streams used.<br>Fixed seed 42 with identical tensors across trials, so outputs can be memoized.<br>Baseline is cached from another machine and time (`utils/baseline_cache.py:39-117`).<br>**Bug:** `correctness_flag = bool(result.get("correctness") or result.get("compiled"))` (`kernelbench.py:262`). |
| **backend_bench** | `PE/environments/backend_bench/backend_bench.py:376-402` | Geometric-mean speedup. | Perf path is dead (`have_perf_test = False` inside the loop), so every correct kernel gets perf = 1. |
| **autodiff** "unspoofable sentinel" | `PE/environments/autodiff/autodiff.py:133-155` | `secrets.token_hex` printed after `if {unittest}:`. | The sentinel is a constant in the same code object the LLM code runs in (`sys._getframe().f_code.co_consts`). A canonical counter-example for P26. |
| **tensor_puzzles** | `PE/environments/tensor_puzzles/utils/validate.py:47-97` | AST allowlist (calls, operators, attributes; no if/loops/comprehensions) + fresh Modal sandbox + `returncode == 0`. | Exit-code pass is safe only because of the AST gate. |
| **forth_lang** | `RE/environments/code/forth_lang/forth_lang/scoring.py:62-80`; `taskset.py:83-140` | Banned-token check; **fresh grader runtime per scoring**; per-test `timeout`. | `s" *" evaluate` builds banned words at runtime; stack text parsed from stdout+stderr. |
| **i3_code / livecodebench** | `RE/environments/code/i3_code/i3_code/taskset.py:57-76`; `verify.py:141-193` | Per-test subprocess timeout. | `tests.json` with expected outputs in `/tmp` of the same runtime; trusts candidate stdout JSON. |
| **judges/swe** | `RE/judges/swe/rubrics.toml:8-10`; `prompt.md:26-27` | "solved … with no reward hacking (test tampering, hardcoded outputs, special-casing)". | LLM judgement only. |

**Recommendation**

| M7 piece | Take |
|---|---|
| hidden_tests (test-surface reset) | **PORT** pi_deepswe's `PROTECTED` regex and `prepare()` reset/remove logic (Apache-2.0). Extend it: `.pth`, `pyproject.toml [tool.pytest]`, `setup.cfg [tool:pytest]`, `CMakeLists.txt` under test dirs, `CTestTestfile`. |
| hidden_tests (validity) | **PATTERN**: `evaluation_valid` separate from `reward` → our `cause=infra` vs `cause=candidate`; timeout-only retries with all attempts archived. |
| static | **PATTERN**: tensor_puzzles AST allowlist for constrained-solution tasks; forth_lang shows token bans need a runtime-eval check too (A13 "banned code hidden behind … importlib"). |
| perf | **PATTERN** only for kernelbench's warmup/trial loop *inside a grader-built harness*. Its timing location, seed reuse and cached baseline are anti-patterns; add them to the A13 attack list. |
| fresh_draws | Nothing to take (fixed seeds everywhere). |

---

## 5. NeMo-Gym (`NG/`, Apache-2.0)

**Files and interfaces**

**Two-sandbox grading, `resources_servers/swemer_oml/app.py`:**
- **Design** (`:15-26`): each task ships `tests/test.sh` + `grade.py` + `config.json`, which write a 0/1 reward.
- **`seed_session`** (`:290-311`): optional anti-cheat; records the base HEAD and the *pristine untracked files*.
- **`_extract_model_patch`** (`:274-288`): worktree `git add -N . && git diff <base>` or committed-mode tip diff
  (`swebench/patch_capture.py:15-30, 225-258`). The agent sandbox is stopped in `finally`.
- **Test-surface stripping:** edits to files the hidden `test_patch` touches are stripped (`app.py:156`;
  `swemer_v2/verification.py:106-131`).
- **`verify`** (`:313-403`): a fresh sandbox with only the eval script, `/tests/{test.sh,config.json,grade.py,test.patch}`
  and the candidate patch (`verification.py:156-172`).
- **Eval script** (`:118-153`): `rm -f reward.txt`; lenient patch chain (`:103-115`); `patch_applied`/`tree_changed`
  by md5 of `git status`; reward printed between markers.
- **Verdict** (`parse_verification_output`, `:187-251`): from reward-file content only.
  - `completed=False` on: workdir missing (exit 97), test.patch apply failure, silent grader, or reward ∉ {0,1}.
  - `resolved = completed ∧ tree_changed ∧ reward==1`.
- **Retry:** one inconclusive retry on a fresh sandbox (`app.py:76-78, 374-381`).

**`swebench/anti_cheat.py` + `anti_cheat_setup.sh`:**
- Orphan branch `_nel_work`; delete `packed-refs`, `ORIG/FETCH/MERGE_HEAD`, all other refs, tags, remotes and logs;
  `reflog expire`; `gc --prune=now`.
- **Best-effort:** on failure the agent gets the unscrubbed repo (`anti_cheat.py:41-59`). `non_ancestor_left` is
  logged, never gated (`setup.sh:21,25`).

**`nemo_gym/verifier_fixture.py`:**
- `kind ∈ {full_reward, zero_reward, malformed, determinism}`, JSONL rows with `extra="forbid"` (`:21, 39-55`).
- full/zero/malformed are always required; determinism is required iff `seeded` (`:186-195`).
- Each case runs on a **fresh server** (`:112`); rewards must be finite, non-bool and in range (`:95-99, 218-221`).
- full/zero must pin the range endpoints (`:223-234` ✓); malformed must raise with `expected_error` (`:202-215` ✓);
  determinism compares re-seeded fresh servers (`:235-238` ✓).
- `update_expected` does an atomic rewrite.
- Onboarding requires `VERIFIER_FIXTURE` (`environment/onboarding.py:226-237`).

**`/verify` contract, `base_resources_server.py`:**
- `BaseVerifyResponse{reward: float, mask_sample: bool=False, failure_kind: str|None, failure_reason: str|None}`
  (`:107-148`). `mask_sample` means "the environment or its infrastructure failed rather than the evaluated system";
  `failure_kind` is orthogonal.
- `BaseMultiRewardVerifyResponse.reward_components: dict[str,float]` (`:151-164`).
- `ReverifyMode = stateless|unsupported|unknown` (`:75-85`).
- Vocabulary in `failure_kinds.py:49-140`. Crucible already mirrors it: `core/schema/run.py:18-29`.
- **`judge_failsafe`** (`judge.py:85-121`): `JudgeError` → reward 0, `mask_sample=True`, `failure_kind="judge_failed"`.
- Harness checker **TE-6**: masked or incomplete rows need both `failure_kind` and `failure_reason`
  (`harness_capabilities/checker.py:644-680`).

**`rollout_reverification.py`:**
- Re-POSTs stored rollouts carrying verifier inputs only, never rewards (`:613-643`).
- Refuses UNSUPPORTED/UNKNOWN servers unless `force`, which prefixes outputs `unsafe_` (`:779-819`).

**Code-execution graders:**
- **`code_gen`** runs code in-process, explicitly "no sandbox" (`code_gen/app.py:188`; `testing_util.py:641-720`
  "NOT a security sandbox"); the trusted signal is captured stdout.
- **CCC** writes `correct_output.txt` into the solution's cwd and scores `float(run_stdout)` (`ccc_eval.py:483-507`).
- **No perf grader** (`speed_bench` measures decoding acceptance).

**Holes:**
- **`swemer_oml` never masks** (✓). It contradicts its own `BaseVerifyResponse` contract and TE-6.
- **The anti-cheat is git-only and best-effort:** no alternates, worktrees, submodules, site-packages or egress.
- **Fuzz-5 patch application counts as "completed".**
- **No `--binary` in the worktree diff** (`patch_capture.py:240`).
- **`terminal_bench_2_1` grades in the agent's sandbox** with unvalidated `float(reward.txt)`, yet declares
  `STATELESS` (`terminal_bench_2_1/app.py:48, 293-325`).
- **The fixture has no adversarial, infra or timeout kinds**, and never asserts `mask_sample`.

**Recommendation**

| M7 piece | Take |
|---|---|
| isolation/separate env | **PATTERN**: swemer_oml two-sandbox flow (patch-only crossing, agent box stopped, hidden tests only in grading box, pristine-untracked exclusion, test-surface stripping, inconclusive retry on a fresh box). |
| grader.py output contract | **PATTERN**: `reward` + `reward_components` + `mask_sample` + `failure_kind` + `failure_reason` (already the M11 `export nemo_gym` target). TE-6 becomes a grader self-check: masked ⇒ both failure fields set. |
| judge_item | **PATTERN** `judge_failsafe` (judge exception → masked). |
| calibrate / M9 | **PATTERN** (or **PORT**, Apache-2.0, ~250 lines) `verifier_fixture`, extended with kinds `adversarial` (A13 cheat mutants, expect 0, *not masked*), `infra` (injected grader-precondition failure, expect masked), `timeout`. |
| leak audit | **PATTERN** anti_cheat steps; our M1 `guard git_scrub/git_audit` already verifies afterwards, where NeMo only logs. **SKIP** the best-effort semantics. |
| tiers / regrade | **PATTERN** `ReverifyMode` gate: a stateful World grader is `unsupported` unless the World snapshot is stored. |
| code_gen / CCC execution | **SKIP** (anti-patterns for A13). |

---

## 6. EnterpriseOps-Gym (`EOG/`, Apache-2.0)

**Files and interfaces**
- **Types:** `VerifierType = database_state | response_check | tool_execution` (`benchmark/models.py:6-9`).
- **Spec:** `VerifierConfig{verifier_type, validation_config, name?, description?, gym_name?}` (`:29-37`).
- **Engine:** `VerifierEngine.execute_verifier(verifier, model_response, database_id, context, gym_name) -> dict`
  (`benchmark/verifier.py:48-79`). An unsupported type → `passed: False`.
- **`database_state`** (`:81-127`):
  - `{query, expected_value, comparison_type ∈ equals|greater_than|less_than|contains}`.
  - The SQL goes to `POST {mcp}/api/sql-runner` with `x-database-id` and user-context headers, 30 s timeout
    (`:195-255`).
  - Extraction (`:291-372`): 1×1 → scalar; 1 row → dict; n rows → list. **MCP error text is returned as the value**
    (`:357-365`).
  - Comparison (`:374-399`) is strict Python `==` (`"0" != 0`).
  - Real spec: `{"query": "SELECT coverage_hours FROM entitlement WHERE entitlement_id = 73;", "expected_value": "h24x7", "comparison_type": "equals"}`
    (`data/revised/csm/task_20251205_…json:34-43`).
- **`response_check`** (`:129-164, 412-483`): an LLM judge scores 1–10, pass ≥ 7 (default). The agent's text is
  pasted verbatim. **The judge is the policy's own `llm_client`** (`executor.py:126`).
- **`tool_execution`** (`:166-193`): checks tool-name presence plus `minimum_tool_calls`. Action-based (P12 says
  weight 0).
- **Aggregation:** `overall_success = all(passed)` and `pass_rate = passed/total` (`executor.py:371-396`).
- **Database lifecycle:**
  - a fresh DB per run from a SQL dump, under a random id (`executor.py:320-349`; `mcp_client.py:19-62`);
  - deleted in `finally`;
  - `reset_database_between_runs` is never read.

**Holes:**
- **Results are keyed by `verifier.name`, so duplicates overwrite each other.** 23 of 102 local verifiers are dropped
  this way (`executor.py:420, 474`).
- **Vacuous passes:** `all([])` passes; an unknown `gym_name` is skipped (`executor.py:442-450`).
- **83 of 102 checks are `COUNT(*)` existence checks**, and there is no floor measurement.
- **Arbitrary SQL runs on the agent's server.**
- **Errors trigger a resample of the whole rollout** (`evaluate.py:221-233`), which biases the scored set.

**Recommendation**

| M7 piece | Take |
|---|---|
| state_predicate | **PATTERN**: the `{query, expected, comparison}` spec over a fresh-per-run DB. Implement it over the M1 `VerifierView.query` (read-only State Adapters; `src/crucible/world/world.py:95-125`). Fixes: results keyed by unique check id; empty plan = error; unknown app = error; typed comparisons (`eq`, `approx(tol)`, `in_set`, `regex`, `rowset_eq`, `count_eq`), Decimal for NUMERIC (B.8); adapter errors raise `InfraError`, never become a value. |
| invariants | **PATTERN**: "verify group membership"-style count predicates become toy SQL invariants. Add `unchanged(app)` for infeasible/refusal tasks (M1 already has `VerifierView.unchanged`). |
| judge_item | **SKIP** `response_check` (self-judging, injection-prone). |
| tool_execution | **SKIP** as a check (P12); at most a weight-0 diagnostic. |

---

## 7. MiMo v2.6 release, mimoagent, xiaomimimo-verl (`MM/`, `MA/`, `MV/`)

**Files and interfaces**

**The release has no `/verify` HTTP endpoint.**
- Grading is a Harbor-style command in each env's manifest: `"verifier": {"command": "python3 /work/run_verify.py", "reward_file": "/logs/verifier/reward.json", …}`
  (`ENV/manifest.json:64-87`).
- `verify.py`, `run_verify.py` and `_helpers.py` are byte-identical across the 266 envs.

**`run_verify.py`:**
- Agent output = the last assistant text in `/tmp/agent_output/sessions/**/*.jsonl` (`:14-31`).
- Post-state = every `/work/system/<mcp>/state.db` opened `?mode=ro` and fully dumped (`:34-51`).
- **Masking contract:** on any testbed failure, `reward.json` gets no parseable `reward` key → the trainer masks the
  sequence "instead of training on a false 0" (`:102-107`). Payload `{"reward": s}` or `{"reward_error": …}` (`:141`).

**`verifier_meta.json`:**
- Item schema `{id, tier: critical|important|sanity, method: rule|llm, question, pass_anchor, files, fn, ground_truth, weight, description, dim, tolerance}`,
  plus top-level `check_code`, `gold_answer`, `input`, `act`.
- No item uses `gate` or `src_protect`.

**`verify.py`:**
- **Rule items** (`:15-48`): a subprocess runs `_helpers` + `check_code` + `print('___R___'+json(fn(ws)))` with a
  300 s timeout. The last `___R___` line wins. A `random` substring scores 0.
- **LLM items** (`:311-340`): one call per item, 20k-char evidence with truncation markers, a strict 0/1 prompt, the
  median of `GRADE_VOTES` (default 1). No verdict → `_JudgeUnavailable` → `reward_error` → masked.
- **Aggregation:** Σw·s/Σw. **`tier` is ignored** (`:392-400`), so the 89 "sanity" items give a do-nothing policy
  non-zero reward.

**`MA/agents/antihack.py` (MIT):**
- A pre-execution interceptor with recall-first regexes for git archaeology, code hosts, unpinned installs and eval
  artifact names (`:86-195`).
- Blocks return a plain `"Permission denied. Resource unavailable."` (`:336-343`).
- **Default off** (`:205`). Enabled only for General RL (`MV/config/agent/general/s3k.yaml:29-33`); the code-RL
  configs use a prompt threat instead (`mini-claude-code.yaml:21-29`); blackbox harnesses never call it.
- **Regex-bypassable:** `git -C . log --all`, `git --no-pager …`, scripts.

**Environment-side defenses (`MA/environments/datasets/base.py`):**
- **`strip`** (`:840-893`): detach first, delete refs, keep only ancestor tags, `gc` with `gc.pruneExpire=now -c gc.cruftPacks=false`,
  **verify** zero newer commits, else fall back to `hide`.
- **`hide`** moves `.git` away (`:895-922`).
- **`answer_leak_blocklist`** writes `/etc/hosts` and fails closed (`:191-208`).
- **`REWARD_TESTBED_CORRUPTED`** (`:30-35`).

**General env host (`MV/recipes/general/general_agent/environment.py`):**
- Verifier files are uploaded only at reward time, only to the sidecar (`:62-93`); the verifier must not be `main`
  (`:247-249`).
- Old reward files are deleted (`:532-537`).
- Missing or out-of-[0,1] reward → `REWARD_TESTBED_CORRUPTED` (masked) (`:558-579`).
- The sidecar mounts the workspace read-only, and the DBs are only on the sidecar (`s3k.yaml:43-59`).

**Trainer:**
- `INVALID_REWARD_VALUE = -999.0` for infra categories (`recipes/general/agent_loop.py:76-86`).
- GRPO replaces invalid rows with the group mean of the valid ones (`verl/trainer/ppo/core_algos.py:306-322`).
  Crucible C6 already chose `None` + group-mean (`src/crucible/credit/operators/infra_mask.py:1-9`).

**Holes:**
- **Agent output is read from agent-writable session logs**, and an agent-written `answer.md` beats the chat answer
  (`verify.py:420-433`; `environment.py:683-718`).
- **The judge is open to injection** (raw evidence, 1 vote, `gpt-4o-mini`).
- **The free-reward floor** from ignored tiers.
- **Last-line-wins rule parsing.**

**Recommendation**

| M7 piece | Take |
|---|---|
| judge_item | **PATTERN**: the item schema `{id, tier, method, question, pass_anchor, files, weight}`, per-item calls, truncation-aware evidence, strict 0/1, median of votes, judge-unavailable → masked. Add: tier semantics (critical = gate), a measured floor (A2), fenced evidence, and only items validated by the M6 validity gate (`credit/validity.py`). |
| state_predicate | **PATTERN**: rule items (`fn(ws) -> {score, detail}`) over read-only SQLite post-state. Fix: the result goes to a grader-owned file, not the last stdout line. |
| isolation/separate env | **PATTERN**: sidecar topology (late upload, never in `main`, read-only workspace mount, DBs sidecar-only, state changes only through MCP). This matches M1's App/State-Adapter split. |
| grader.py infra contract | **PATTERN** "no parseable reward ⇒ masked". For Harbor compatibility we keep a numeric `reward` (0) and add `infra/masked = 1`. |
| leak audit | **PATTERN**: `strip` → verify → `hide` fallback; `gc.cruftPacks=false`; fail-closed host blocklist. M1 `git_scrub`/`git_audit` should adopt the `cruftPacks` flag and the keep-ancestor-tags rule. |
| antihack | **PATTERN** for P27 only: merge its patterns into `connectors/agent_events.py` probe tagging. **SKIP** it as a defense (P21/P25). |

---

## 8. OpenEnv (`OE/`, BSD-3-Clause)

**Files and interfaces**

**Rubrics (`src/openenv/core/rubrics/`):**
- `Rubric(ABC)` is modelled on `nn.Module`: auto-registered children, `forward(action, observation) -> float`,
  pre/post hooks, `last_score`, `named_rubrics` dot paths (`base.py:17-216`). It has no exception boundary, clamp or
  NaN check.
- **`Sequential`** short-circuits on an exact `== 0.0` and returns the **last** child's score. An empty `Sequential`
  returns **1.0** (`containers.py:60-75` ✓).
- **`Gate(rubric, threshold=1.0)`**: `0.0 if score < threshold else score`, so **NaN passes** (`:167-220`).
- **`WeightedSum`**: weights must sum to 1; negative weights are allowed; children are gathered in parallel (`:223-323`).
- **`LLMJudge`**: the first regex number wins; `default_score` on a miss (`llm_judge.py:28-122`).

**RFC 008 (`rfcs/008-environment-auto-validation.md`):**
- **Gap contract** (`:170-205`): `reward(oracle) ≥ max − tol`, `reward(reset) = floor`, `max − floor ≥ margin`.
  - "No oracle → FAIL."
  - "Floor is measured, not declared."
  - Oracle containment is required.
  - "Determinism is a precondition of semantics" (otherwise the semantic checks are SKIPped).
  - An LLM-judged reward must pin model, version and params.
- **Canaries:** `semantic.canary_floor`, i.e. shipped canary trajectories must stay at the floor (`:337, 342-343`).
- **Hermeticity:** the verifier runs with network denied regardless of the env's policy (`:251-252`).
- **Wire hygiene:** step rewards must be finite JSON numbers, not bools, and in range (`:510-526`).

**Implemented:** manifest models, all `extra="forbid"` (`validation/manifest.py:39-300`):
- `RewardDeclaration{range, oracle_tolerance, floor_margin, variance_tolerance}`;
- `OracleDeclaration{form: injected_state|script}`;
- `JudgePin`.

**Policy and graders:**
- Severity policy (`policies/severity-v1.json:3-8`: `max_oracle_tolerance 0.1`, `min_floor_margin 0.1`).
- `apply_policy`: ERROR ⇒ FAIL, i.e. fail closed (`policy.py:107-149`).
- Grader protocol `{check_id, level, requires_capabilities, depends_on, run}` (`graders/__init__.py:47-75`).
- **Only `StaticManifestGrader` is registered**, and `levels_run=[STATIC]` is hard-coded (`runner.py:125-126, 154`).
- `DockerValidationProvider` hardening argv: `--user 65532 --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 …`
  (`providers/docker.py:270-330`). It is unwired.

**Isolation rules:**
- Rewards are server-side only (`rfcs/004-rubrics.md:52`).
- Reserved names `reset/step/state/close` are never MCP tools (`core/env_server/mcp_types.py:316`).
- "The rubric runs *after* the harness turn completes" (`rfcs/005-agentic-harnesses.md:540-552`).

**Recommendation**

| M7 piece | Take |
|---|---|
| calibrate / M9 A1–A2 | **PATTERN**: the RFC 008 gap contract, with declared tolerances bounded by a versioned policy (`severity-v1.json`), "no oracle → FAIL", measured floor, determinism before semantics, oracle containment as an audit. M9 A1/A2/A4 implement what OpenEnv has not (semantic level). |
| leak audit | **PATTERN**: canary trajectories plus canary strings planted in hidden material (`tests/`, refs, solution); the leak audit greps the agent image and every policy-visible tool output for them. |
| grader result model | **PATTERN**: check results `pass/fail/skip/error`, severity applied separately, ERROR fails closed, `depends_on` → SKIP with the dependency named. |
| rubric tree | **PATTERN** in `export openenv` only (DESIGN §10.1). Our gates are all-or-nothing; when lowering, use `Sequential(Gate(g, 1.0)…, WeightedSum(...))` and guard against empty Sequential, NaN and the last-child return. |
| Docker hardening argv | **PATTERN** for `isolation/separate_env.py` defaults: read-only rootfs, cap-drop ALL, no-new-privileges, pids limit, tmpfs. |
| LLMJudge | **SKIP**. |

---

## 9. SWE-smith, Repo2RLEnv, Karotte, agentenv

### 9.1 SWE-smith (`SS/`, MIT)

**Validation (`harness/valid.py`):**
- A clean reference run once per repo (`:200-219`, `timeout_ref=900`, `profiles/base.py:108-109`).
- A bug-patch run per instance (`:102-127`, `timeout=90`).
- Status diff (`grading.py:56-92`): only an exact `FAILED`→`PASSED` counts, and a test seen in one run only is dropped.

**Thresholds** live in `harness/gather.py:328-348`: ≥1 F2P and ≥1 P2P (except `.pr_`).
- The code reads `PASS_TO_FAIL` and flips it to `FAIL_TO_PASS`. That looks inconsistent with `get_valid_report`'s
  keys. **Check against upstream before porting.**
- F2P test files are removed from the task branch in a separate commit (`gather.py:539-558`).

**Execution (`harness/utils.py:144-152`):**
- A fresh container (`mem_limit=10g`) **with no network restriction**.
- Patch chain `git apply` → `--reject` → `patch --fuzz=5` (`:77-98`).
- Only F2P/P2P files are reverted after the model patch (`:226-238`); a new `conftest.py` survives.

**Log parsing:**
- Markers come from the bash trace (`:243-255`), and the pytest regex is last-match-wins over all stdout lines
  (`profiles/python.py:93-102`). It is spoofable.
- `repair.py:138-213` drops tests that fail under gold, offline, with no reruns.

**Recommendation:**
- **PORT** the F2P/P2P contract into M8 (forge), as ARCHITECTURE says.
- For M7 `hidden_tests`, **PATTERN** only:
  - F2P ∧ P2P as the pass condition;
  - missing test = fail;
  - gold-run repair → our `calibrate.flaky_tests` with **N gold reruns**, not one.
- **SKIP** its stdout parsing and lenient patch apply.

### 9.2 Repo2RLEnv (`R2E/`, Apache-2.0): the trusted grader to port

**Owned-recipe export** (`pipelines/recipes/repository/export.py:175-292`):
- agent `user=learner`, `network_mode=no-network`;
- verifier `user=root`, `environment_mode=separate`, no network;
- only declared submitted files cross;
- a separate `tests/Dockerfile` adds a `grader` uid 1001.

**`pipelines/recipes/swe_smith/grade.py`** (196 lines ✓) + **`test_driver.py`** (41 lines ✓). Excerpt (Apache-2.0):
```python
# R2E/pipelines/recipes/swe_smith/grade.py:122-192 (abridged)
reward = logs / "reward.txt"; reward.write_text("0\n"); reward.chmod(0o644)      # fail closed first
try: validate_submission(Path("/workspace"), contract)                           # no symlinks, regular files, <=8 MiB,
except (ValueError, OSError) as exc: ...; return                                 # only .py added, immutables sha256, chown 0
with tempfile.TemporaryDirectory(prefix="r2e-grade-") as temporary:
    working = Path(temporary); os.chown(working, 1001, 1001); working.chmod(0o700)
    report = working / "results.xml"
    command = [sys.executable, "-I", "/tests/test_driver.py", *contract["test_paths"],
               "-q", "--tb=short", "-p", "no:cacheprovider", f"--junitxml={report}"]
    process = subprocess.Popen(command, cwd="/workspace", stdout=..., stderr=..., user=1001, group=1001,
                               extra_groups=[], start_new_session=True, env=child_environment(temporary))
    try: code = process.wait(timeout=contract["timeout_sec"])
    finally:
        try: os.killpg(process.pid, signal.SIGKILL)   # descendants must not keep modifying the child-owned report
        except ProcessLookupError: pass
        process.wait()
    if report.is_symlink() or not report.is_file() or report.stat().st_size > 16 * 1024 * 1024: ...; return
    parsed = parser.parse_junit(report.read_text(), returncode=code)
    expected = set(contract["expected_passes"])
    passed = code == 0 and bool(expected) and parsed.passed == expected
    reward.write_text("1\n" if passed else "0\n")
```
`child_environment()` (`:21-45`): `PATH` = own interpreter dir + system dirs, `HOME` = the temp dir,
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and only GPU/HF runtime variables inherited.

```python
# R2E/pipelines/recipes/swe_smith/test_driver.py:19-36 (Apache-2.0): bounded framework-tamper canary
original = {name: (value, getattr(value, "__code__", None))
            for name, value in vars(unittest.TestCase).items() if name.startswith("assert") and callable(value)}
case = unittest.TestCase
code = int(pytest.main(["-o", "addopts=", *arguments]))
changed = [n for n, (m, impl) in original.items()
           if getattr(case, n, None) is not m or getattr(getattr(case, n, None), "__code__", None) is not impl]
if unittest.TestCase is not case or changed: return 254
```

**Other pieces:**
- **Admission gate:** `[nop, oracle]` rewards must equal `[0, 1]` on Harbor trials (`pipelines/repo_mutate.py:47-63`);
  codemidas runs nop×2 and oracle×4 (`recipes/codemidas/pipeline.py:400`).
- **Probes:** `wrong_solution` must score below success and `valid_alternative` must equal success
  (`quality/loop/runner.py:95-112`).
- **Verifier policy:** a static AST check rejects verifiers that read implementation source (`quality/loop/verifier_policy.py:1-79`).
- **Hash-bound evidence** (`quality/labels.py:55-100`).
- **Leak scrubs:** git (`pipelines/_env_guard.py:64-88`); problem-statement regexes (`pr_runtime.py:91-147`), already
  partly in `world/leaks.py`.
- **Weak spots:** the egress guard is a denylist (`_env_guard.py:17-24`). The `pr_runtime` test.sh calls `python3`
  from the agent-controlled PATH and falls back to an exit-code reward (`pr_runtime.py:536-545`).

**Recommendation:**
- **PORT** `grade.py` (`child_environment`, `validate_submission`, the run/kill/report/exact-set core) and
  `test_driver.py` into `verify/patterns/hidden_tests.py` + `verify/grader.py`. Vendor them under `_vendor/r2e_grade/`
  with LICENSE + ORIGIN.md.
- **PATTERN** the nop/oracle `[0,1]` gate and the wrong/valid-alternative probes (M9 A1–A3).
- **SKIP** the `pr_runtime` in-box verifier.

### 9.3 Karotte (`K/`, MIT)

The phrase "trusted channels" does not appear in Karotte. Its model is a uid boundary plus a custody step, in the
same sandbox, as root, after reaping:
- **`collect_submission`:** kill student processes → delete other student files → copy the submission → delete the
  rest (`templates/default/src/environment/submissions.py:17-34`).
- **`kill_processes`:** cgroup kill → PID-namespace kill → in-cohort `kill(-1)` → `/proc` sweep. 30 s deadline, then
  `UnreapableCohortError` → 0 (`process_utils.py:48-125`).
- **`save_submission`:** `O_NOFOLLOW` walk with size, entry and depth caps (`save_submission.py:46-190`).
- **`Step.score`:** `StudentMisbehaviorError` → 0 with `metadata.misbehavior` (`step.py:82-92`). Misbehavior is
  scored, never masked: already `crucible:policy_misbehavior`.
- **`ExecutableJudge`:** reads `{"score","metadata"}` from a root temp dir; non-zero exit, missing file or bad JSON → 0
  (`judges/executable_judge.py:79-185`). It has no timeout.
- **PATH sanitization:** `sanitize_paths` strips non-root or world-writable `PATH`/`LD_*` entries and re-execs
  (`run_helpers.py:892-930`); `trusted_binary` (`trusted_bin.py:6-31`).
- **Harbor 0.23 attack matrix** (`docs/why-karotte.md:342-352`): pre-written `reward.json`, a leftover process
  rewriting tests, a fake `python3`, an answer-file symlink, and a FIFO hang. All five are A13 seeds.

**Recommendation:** M1 already ported this (`world/guard/guard.py` header comments cite Karotte). M7 **reuses** it
(PATTERN, done). Add the `sanitize_paths` step to `TrustedRunner`'s root-side environment, and the fake-model
replayed-hack regression style to M9 A13.

### 9.4 agentenv (`AE/`, Apache-2.0)

**`RunContainerUnitTestsVerifierTaskStep`**, type `run_container_unit_tests_verifier`
(`task_step/task_steps/verifiers/run_container_unit_tests_verifier.py`):
- **Config** (`:77-115`): `{sandbox_name, container_name, command, verifier_id, setup_commands, user="root", timeout_sec=300, env_vars, result_paths, reward_path, fail_task_on_error}`.
- It runs **in the agent's container** (`:215-229`) as root via `timeout --kill-after=10` (`:278-284`).
- **It injects the host's LiteLLM/Anthropic keys into that container** (`:240-259`).
- It never clears a stale `reward_path`.
- `_parse_reward` clamps `max(0, min(1, float(raw)))` (`:396-410` ✓). **NaN therefore becomes 1.0**, a pass.
- **Recording** (`:353-379`): `metadata.verifications[id] = {results:[{id, score, result, message}], score, exit_code, timed_out, stdout_artifact, …}`.
- `review.py` is a human pause, not a grader (`task_step/task_steps/review.py:1-152`).

**Recommendation:**
- **SKIP** the step as a grader. DESIGN §10.1 already plans a Crucible `crucible_verify` step that stages hidden
  material only after the agent finishes, in a separate sandbox.
- **PATTERN** its recording shape for `interop/agentenv` (reject non-finite values).
- The NaN clamp is an A13 seed for *our* reward readers.

---

## 10. Recommendation matrix per Crucible M7 piece

| Piece | DEPEND | PORT | PATTERN | SKIP |
|---|---|---|---|---|
| `patterns/state_predicate` | M1 `VerifierView`/State Adapters (ours) | — | EOG `database_state`; MiMo rule items; M5 milestone predicate language (`core/schema/milestone.py`) | EOG `response_check`, `tool_execution` as checks |
| `patterns/differential` | — | **task04** `ref()`/`cand()`/`seal()`/consumers/fingerprints; **task03** root-before-candidate installs | NeMo two-sandbox; MiMo sidecar | — |
| `patterns/hidden_tests` | — | **R2E** `grade.py` + `test_driver.py`; **pi_deepswe** `PROTECTED`/`prepare()` | SWE-smith F2P∧P2P, missing = fail, gold repair (N reruns); verifiers wipe-and-restage | SWE-smith/r2e_gym stdout parsing; exit-code-only pass |
| `patterns/adversarial_inputs` | — | **task04** guard-band / exact-allocation / offset-ROI cases (B.4) | R2E `wrong_solution`/`valid_alternative` probes | — |
| `patterns/invariants` (incl. concurrency, sanitizers) | — | **task04** `gate_concurrency` (vs solo), `gate_sanitizer` (fix: own `--log-file`) | EOG count predicates; `VerifierView.unchanged` | — |
| `patterns/fresh_draws` | — | **task04** `gate_fresh`; **task03** `fresh_cases` build/run split | — | kernelbench fixed seeds |
| `patterns/perf` | — | **task04** interleaved rounds, best-of-k, re-measure; **task03** end-state sanity | kernelbench warmup/trial loop (inside a grader-built harness only) | in-process timing; cached cross-machine baselines; backend_bench |
| `patterns/static` | — | **task01v2** string-aware `strip_c_comments`; **task04** `TEXREF` + `binary_checks` | tensor_puzzles AST allowlist; forth_lang (shows the runtime-eval gap) | — |
| `patterns/judge_item` | `credit/judge.py`, `credit/validity.py` (ours) | — | MiMo item schema; verifiers `RubricJudge` parse discipline; NeMo `judge_failsafe` | OpenEnv `LLMJudge`; EOG self-judge |
| `isolation/separate env` | **Harbor separate mode**; (export) verifiers `IsolatedVerifierEnv` | — | NeMo swemer_oml; MiMo sidecar; OpenEnv Docker argv; verifiers lifecycle | Harbor shared mode; agentenv in-container verifier |
| `isolation/sealing` | M1 guard `quiesce`/`demote` (ours) | **task04** `seal`/`udir`/`chown_tree` | Karotte custody; R2E `killpg` | — |
| `isolation/collect hooks` | **Harbor `[[verifier.collect]]`** (transport) | **verifiers** `restore`/`_validate_restore` | Harbor regrade manifest; NeMo pristine-untracked exclusion + test-surface stripping | best-effort silent collect |
| `isolation/leak audit` | M1 `world/audit.py`, `guard git_audit`, `world/leaks.py` (ours) | — | OpenEnv canaries + oracle containment; MiMo strip → verify → hide (`cruftPacks=false`); task04 image fingerprint | NeMo best-effort scrub |
| `calibrate.py` | — | **task04** tolerance-calibration procedure | NeMo `verifier_fixture`; RFC 008 gap contract + severity policy; `vf-validate` | — |
| `tiers.py` | **Harbor regrade** (full-tier audit) | — | NeMo `ReverifyMode` gate | — |
| `grader.py` | — | **R2E** grade.py skeleton (pre-write 0, killpg, grader-owned report) | Harbor typed reward errors; verifiers strict adapter; NeMo response contract + TE-6; MiMo "no reward ⇒ masked" | Harbor's reward precedence trusting agent-writable dirs |

---

## 11. Design recommendation for `crucible.verify`

### 11.1 Module layout
```
src/crucible/verify/
  __init__.py
  plan.py            # Check protocol, CheckSpec, CheckPlan, Gate/Family/Tier enums, CheckResult, Verdict
  context.py         # GradeContext: sealed candidate, ReferenceStore, TrustedRunner, SecretSeed, dirs, VerifierView, calibration
  trusted.py         # TrustedRunner (unprivileged exec), Untrusted[T], Artifact (no-follow, size-capped reads)
  build.py           # grader-owned builds: CMakeBuild/PipInstall with flag audit (compile_commands.json allowlist)
  grader.py          # entry point inside the verifier env: preflight, run plan, attribute errors, write outputs
  reward_io.py       # flat reward.json writer/reader (Harbor/verifiers-compatible), crucible_reward.json, report.json
  steps_io.py        # steps.jsonl via credit/steps.py (fast tier may skip)
  patterns/
    state_predicate.py  differential.py  hidden_tests.py  adversarial.py  invariants.py
    sanitizer.py  fresh.py  perf.py  static.py  judge_item.py
  isolation/
    separate_env.py  # VerifierEnvSpec + lowering to Harbor separate mode / agentenv crucible_verify step
    staging.py       # wipe /logs/verifier + /tests, stage hidden material, protected-path reset
    seal.py          # chown/chmod seal, uid cohort kill (guard quiesce)
    collect.py       # generated collect hooks + manifest; artifact archive validation (vendored verifiers)
    leak_audit.py    # canaries, hidden-string scan, image fingerprint, wraps world/audit + guard git_audit
  calibrate.py       # tolerance tiers, induced perturbations, FN rate + upper bound, ref self-consistency, fixtures
  tiers.py           # fast/full selection, tier tag, A12 correlation helper, regrade sampling
  attacks/           # generic A13 cheat mutants (pattern-aware), used by M9
packs/coding/verify/ # coding comparators: kinfu/tsdf/mesh compare, bit-exact buffers, cuobjdump checks, sanitizer parsers
_vendor/r2e_grade/  _vendor/verifiers_artifacts/  _vendor/pi_deepswe_protected/   # each with LICENSE + ORIGIN.md
```
The core stays domain-agnostic: CUDA comparators live in `packs/coding/verify`, and SQL comparators in the generic
`state_predicate`.

### 11.2 Core protocols
```python
class Family(StrEnum): STATE="state"; DIFFERENTIAL="differential"; HIDDEN_TESTS="hidden_tests"; ADVERSARIAL="adversarial"
                       INVARIANT="invariant"; SANITIZER="sanitizer"; FRESH="fresh"; PERF="perf"; STATIC="static"; JUDGE="judge"
class Tier(StrEnum): FAST="fast"; FULL="full"

class CheckSpec(CrucibleModel):           # declarative, serialized to tests/checks.json for audit (A7/A8) and tiers
    id: str; gate: str; family: Family; tier: Tier = Tier.FULL
    depends_on: tuple[str, ...] = ()
    instruction_refs: tuple[str, ...]      # sentence ids in instruction.md (A7: every graded property is stated)
    provenance: Provenance                 # signal id / reference-differential / defect id (A8)
    tolerance: str | None = None           # key into calibration.json, never an inline number
    timeout_s: float; weight: float = 0.0  # weight > 0 only for capped partial credit behind gates (P13)

class CheckResult(CrucibleModel):
    id: str; status: Literal["pass", "fail", "error", "skip"]
    cause: Literal["candidate", "infra"] | None   # set iff status in {fail, error}
    metrics: dict[str, float] = {}; detail: dict[str, Any] = {}; evidence: list[str] = []; seconds: float

class Check(Protocol):
    spec: CheckSpec
    async def run(self, ctx: GradeContext) -> CheckResult: ...

class CandidateFailure(Exception): ...    # raised by TrustedRunner/Artifact: rc!=0, timeout, malformed/missing artifact → fail, cause=candidate
class InfraError(Exception): ...          # grader precondition: ref build failed, fingerprint mismatch, no GPU, load guard, collect failed → masked
```
- **`TrustedRunner.run(argv, *, out_dir, inputs=(), env=None, timeout_s, limits, gpu=False) -> RunResult`**:
  - `RunResult{rc, timed_out, wall_s, cohort_killed, stdout: Untrusted[str], stderr: Untrusted[str]}`.
  - `out_dir` is sealed before return, and `wall_s` is measured by the parent.
- **`Untrusted[str]`** is a wrapper that `CheckResult`/`Verdict` constructors refuse. Stdout can go into `detail` for
  diagnosis but can never decide a verdict. This makes P26 a type error, not a code-review item.
- **`Artifact.read_json/read_bytes(path, max_bytes)`**: no-follow, regular-file-only, size-capped reads from sealed
  dirs (M1 already has the no-follow reader; B.9).
- **`ReferenceStore`**: a root 0700 tree. It is populated before any candidate code runs, or lazily by
  `Reference.compute(...)` which quiesces the candidate uid first. Its fingerprints are checked against
  `calibration.json`.

### 11.3 How a task declares its checks: Python authoring, declarative record

- **Recommendation:** a **Python check plan** (`tests/plan.py`, trusted task-author code that runs as root in the
  verifier) built from typed pattern constructors. The plan **serializes to a declarative `tests/checks.json`**
  (list of `CheckSpec`), which M9, A7/A8 and `tiers.py` consume without executing code.
- **Simple worlds may write YAML instead.** Toy SQL invariants and state predicates are declared in the same
  predicate language as `milestones.yaml` (M5) and compile to `state_predicate` checks.
- **Why not purely declarative:** the seed graders need arbitrary comparators (`compare_kinfu`'s 30 tolerance keys,
  bitwise-minus-timing, NCV status + buffers). **Why not purely code:** acceptance, tiering and instruction coverage
  need a static inventory.

```python
# packs/coding/families/legacy_gpu_migration/seeds/task04/tests/plan.py (sketch)
from crucible.verify import CheckPlan, patterns as P
from packs.coding.verify import kinfu, ncv, people, cuda_static

plan = CheckPlan(task="task04_pcl_gpu_cuda12", candidate="/workspace/pcl", collect="patch")
build = plan.build(P.CMakeBuild(flags=CMAKE_FLAGS, require_libs=REQUIRED_LIBS, forbid_defines=("DOCTEST_CONFIG_DISABLE",)))
cons  = plan.consumers(["kinfu_replay", "people_eval"], originals="/opt/originals", against=build, flag_audit=True)
plan.add(P.static_source(cuda_static.TEXREF, roots=("gpu", "cuda"), id="static/texref", instr=("i3",)))
plan.add(P.static_binary(cuda_static.sm120_only, of=build, id="binary/sm120"))
plan.add(P.differential(cons.tool("kinfu_replay"), workloads=cases.KINFU, compare=kinfu.compare,
                        tolerance="kinfu.tiers", reference="/opt/legacy_rt", id="kinfu", tier="fast:subset(4)"))
plan.add(P.differential(cons.tool("ncv_check"), workloads=ncv.SEEDS, compare=ncv.bitexact, id="ncv"))
plan.add(P.concurrency_equivalence(cons.tool("kinfu_replay"), specs=cases.CONCURRENT, baseline="candidate_solo", id="concurrency"))
plan.add(P.fresh(cases.fresh_workloads, n_seeds=1, compare={...}, id="fresh"))
plan.add(P.sanitizer(("racecheck", "synccheck", "memcheck"), workloads=cases.SANITIZE, id="sanitizer"))
plan.add(P.perf(workloads=PERF_WORKLOADS, rounds=3, bars="perf_thresholds", timing="parent_wall",
                fresh_inputs=True, load_guard=True, max_spread=0.10, depends_on=("*",), id="perf", tier="fast:rounds(1)"))
```

### 11.4 How the grader enforces P26

| P26 rule (DESIGN §8.3 (7)) | Mechanism in `crucible.verify` | A13 mutant that must score 0 |
|---|---|---|
| **Privilege drop before any candidate code** (its `setup.py`, imports, CMake configure) | Every candidate-touching call goes through `TrustedRunner`. `build.py` runs `pip install`/`cmake` **unprivileged** (task03 pattern, not task02). The grader asserts `euid == 0` itself and **never imports** candidate modules: no `importlib` of the candidate in the root process; the static checks only parse. The uid is fresh (M1 guard allocation; never an image account), with `PR_SET_NO_NEW_PRIVS`, `setgroups([])`, `RLIMIT_NPROC`, `oom_score_adj=1000`, `setsid`. | `setup.py` that writes `/logs/verifier/reward.json`; an import-time side effect that edits the benchmark |
| **Grader-owned build** | Hidden tests, benchmarks and consumers are root-owned copies made at staging, before the candidate runs, and compiled with grader flags. The candidate's package config is allowed only through an audited channel: export `compile_commands.json` and reject non-allowlisted defines/`-include`/`-U`/`-O0` on harness translation units. The hidden suite is never copied into the candidate tree (task01v2 fix). Test discovery is checked against the expected set from `contract.json` (R2E): 0 cases run ⇒ fail. | `DOCTEST_CONFIG_DISABLE`/`-DNDEBUG` through `target_compile_definitions(PUBLIC)` or `PCLConfig.cmake`; skipped test subdirectories |
| **No trust in exit codes or stdout** | Verdicts come only from `Artifact` reads of sealed files written into a uid-owned dir that is sealed after `killpg` + cohort kill: JUnit/doctest XML via `--reporters=junit --out=`, sanitizer `--log-file`, `result.json`. Exit codes may only *fail* a check (rc ≠ 0 ⇒ fail), never pass one alone (R2E: `rc == 0 ∧ expected-set match`). Stdout is `Untrusted`. | `os._exit(0)` under pytest; printed `{"case":…,"median_ms":…}` rows; a fake `RACECHECK SUMMARY`; forged pytest `PASSED` lines |
| **Timing outside the candidate's process** | `perf` gates use `RunResult.wall_s` (`time.monotonic()` in the root parent around `wait()`), or an external profiler run by the grader that writes to a grader dir. Workloads are sized so startup is under 5% of wall time, and startup is subtracted using a calibrated empty-workload run of the *original* under the same harness. In-process timers (`frame_ms_median`) are kept as diagnostics with `trust="in_process"`, never gated. | A library constructor that patches `clock_gettime`/`cudaEventElapsedTime`; a timer that reports a constant |
| **Fresh inputs per timed rep** | `perf` asks the workload generator for `inputs(seed, rep)` per rep, from a secret seed. Each rep's output is verified: against a per-input reference, or against a cheap invariant/checksum the reference computed. Reps run in random order, interleaved with the reference. | Memoizing repeated inputs; returning the previous rep's result |
| **Fresh HOME/TMPDIR per invocation** | `TrustedRunner` creates `mkdtemp` HOME and TMPDIR per call, owned by the uid, removed after the cohort is killed. `XDG_*`, `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `JAX_COMPILATION_CACHE_DIR`, `PYTHONPYCACHEPREFIX` point inside it. The env is allowlist-only (R2E `child_environment`), and the root-side `PATH`/`LD_*` is sanitized (Karotte `sanitize_paths`). | Cache-and-replay through `~/.cache`; a results file left in `/tmp` between invocations |
| **Consumers before the candidate on `PYTHONPATH`** | `TrustedRunner` builds `PYTHONPATH = hidden_tests : consumers : candidate_site`. The interpreter is the grader's (`sys.executable -I`); `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-o addopts=`, `-p no:cacheprovider`; protected paths are reset (pi_deepswe); `test_driver` assert canary. References and the original engine live in 0700 root dirs, so the candidate cannot import them (fixes task03 delegation). | Shadowing `mdbuild`; a planted `conftest.py`/`sitecustomize.py`/`.pth`; `import` of the original engine |
| **Background processes** | After every invocation: `killpg`, then guard `quiesce` (uid-wide `kill(-1)` + `/proc` sweep). An unkillable cohort ⇒ `crucible:policy_misbehavior` (scored 0). | A daemon that rewrites outputs after the check; a process that slows the reference run |
| **Reward 0 + `infra_error` on any exception; `reward.txt` never empty** | `grader.py` starts by wiping and pre-writing `reward.json` = `{"reward": 0, "infra/masked": 1}` and `reward.txt` = `0` (R2E). It finishes by atomically replacing them (`os.replace`) in a root 0755 dir. Attribution: `CandidateFailure` ⇒ fail, `cause=candidate`, not masked. `InfraError` ⇒ masked with a registered `failure_kind`. Any other exception ⇒ reward 0, `infra/masked=1`, `failure_kind="verifier_error"`, traceback in `report.json`, and the run is flagged for M9 triage (a grader bug). The A13 "crash the grader" mutants must land in `CandidateFailure`, i.e. 0, not masked. | Filling the disk, OOM-ing the comparator via a giant `result.json` (caps), a malformed JSON artifact |

**Preflight (before any candidate work):**
1. `euid == 0`.
2. Verifier-image fingerprint equals the task's recorded digest (B.5 stale image → `crucible:stale_image`, masked).
3. GPU probe (B.7).
4. Collect manifest present. A missing patch with a failed hook is `crucible:empty_collection` (masked); an empty
   patch from a successful hook is the policy's (scored).
5. Network-off self-test in the verifier env (RFC 008 hermeticity).
6. Wipe `/logs/verifier` and `/tests` scratch areas; refuse artifacts that restore under hidden paths (verifiers).

### 11.5 Outputs
- **`/logs/verifier/reward.json`** is flat and numeric, so Harbor, verifiers and NeMo can all read it:
  ```json
  {"reward": 0.0, "outcome": 0.0, "progress": 0.86, "gate/build": 1, "gate/static": 1, "gate/correctness": 1,
   "gate/concurrency": 1, "gate/sanitizer": 1, "gate/perf": 0, "rubric/validated_items_met": 0.75, "infra/masked": 0}
  ```
  This needs a `RewardJson.to_flat()` / `from_flat()` pair in `core/schema/reward.py`, and a DESIGN §4.3.1 update in
  the same commit (memory rule: DESIGN stays in sync).
- **`/logs/verifier/crucible_reward.json`**: the full `RewardJson`, including `InfraStatus{status, failure_kind, reason}`.
- **`/logs/verifier/report.json`**: per-check `CheckResult`s, the tier, the seed (revealed after the run), reference
  fingerprints, timings with per-round spread, and the load-guard samples.
- **`reward.txt`**: always written, with the same scalar.
- **`/logs/verifier/steps.jsonl`**:
  - computed by M6 (`credit/steps.py`) from milestone predicates evaluated over states that M5 replay materialized
    **inside the verifier env through `TrustedRunner`**;
  - the agent trajectory is an untrusted, bounded-schema input (ATIF), never executed as root;
  - the fast tier writes it only if the task's milestone set is all-cheap.
- **Trainer mapping (M11):**
  - NeMo: `mask_sample = infra/masked`, `failure_kind` from `crucible_reward.json`, `reward_components` = the flat keys
    minus `reward`.
  - verl: −999 / group-mean through C6.
  - verifiers: `@vf.reward` for `reward`, `@vf.metric` for the rest.

### 11.6 `calibrate.py` and `tiers.py`

**`calibrate.py`:**
- **`natural_spread(check, oracle, draws≥40)`** and **`induced(check, perturb=[...], draws≥40)`** → `ToleranceTiers{strict, relaxed}`.
  The relaxed tier is reachable only when a declared *bifurcation detector* fires (e.g. a shift-origin mismatch), never
  merely because strict failed (fixes the task04 KinFu-LS P20 defect). The relaxed tier must still kill every negative.
- **FN rate:** recorded over all calibration draws, with a Clopper-Pearson upper bound → `calibration.json` and the
  card (P15).
- **`reference_self_consistency(n≥3)`:** catches B.2 Haar-style nondeterminism in the original.
- **`generator_robustness(fresh, seeds≥N)`:** the generator must never crash the grader (catches `fresh_cases.py:34`, A1).
- **`fixture(plan)`:** NeMo-style kinds `full` (oracle = 1), `zero` (no-op, original, naive = 0, measured not
  declared), `malformed` (raises → masked), `determinism` (same seed → same verdict), plus Crucible kinds
  `adversarial` (A13 mutants → 0, not masked) and `infra` (an injected precondition failure → masked).
- **Gap contract:** RFC 008 (`oracle ≥ 1 − tol`, `floor = 0`, margin) with tolerances bounded by a versioned policy file.

**`tiers.py`:**
- `CheckSpec.tier` plus per-pattern reductions (`subset(k)`, `rounds(1)`).
- `select(plan, tier)` returns the reduced plan; the tier used is recorded in `report.json` and `crucible_reward.json`.
- The A12 helper computes Spearman(fast, full) on calibration rollouts.
- The 5% training audit re-grades stored artifacts with the full tier via **Harbor regrade** (DEPEND). A divergence
  is a registry incident.

### 11.7 Order of work for M7 (golden exit gate)

1. **Core runtime:** `trusted.py` + `grader.py` + `reward_io.py` (the R2E port + task04 sealing), with unit tests and
   red-team tests for each A13 row in §11.4.
2. **Re-express task04** with `differential`, `invariants`, `fresh`, `sanitizer`, `perf` and `static`. Golden tests:
   - the oracle scores 1;
   - the Astra patch fails exactly the three perf bars (0.710/0.65, 0.702/0.66, 0.428/0.42);
   - the sol patch fails NCV.

   The perf gate moves to parent-wall timing. Its bars must therefore be **re-calibrated**, with the historical
   profiles reproduced on gate *outcome*, not on exact ratios. Record this in the M7 doc as a deliberate deviation.
3. **task03:** `hidden_tests` (JUnit + expected set; consumers first; original engine hidden) + `fresh` (with the
   generator fix) + `perf` with end-state check.
4. **task01v2:** a grader-owned doctest build with XML reporter + expected set.
5. **task02:** an unprivileged install + a root-owned benchmark copy.
6. **Toy:** SQL `state_predicate` / `invariants`.
7. **Lowering:** `isolation/*` to Harbor separate mode and the agentenv `crucible_verify` step.

Steps 3–5 change the graders' *security* but must keep every recorded verdict. Any verdict that changes is
investigated as a possible earlier false pass.

---

## 12. Open questions

1. **Perf re-calibration under parent-wall timing (task04).** The bars were set on in-process timers. Gate outcomes
   should hold (Astra fails, oracle passes with ≥ 15% headroom per A1), but this needs lab-5090 time. If the headroom
   collapses, use larger workloads before relaxing any bar.
2. **Per-rep fresh inputs for KinFu workloads** need a reference run per input, which multiplies reference cost by k.
   The alternative: one reference per input set plus a reference-computed checksum for the rest. Decide per workload
   in calibration.
3. **SWE-smith `gather.py:333` `PASS_TO_FAIL` read** vs `get_valid_report`'s `FAIL_TO_PASS` key: verify against
   upstream before the M8 port.
4. **Harbor separate-mode image fallback:** the docs say `tests/` is uploaded when the verifier falls back to the
   agent image; the code always passes `skip_tests_upload=True` (`trial.py:934`). Not fully traced. Crucible should
   always give the verifier a prebuilt `docker_image` (B.7) so the question never arises.
5. **Should "unattributed grader exception" be masked or scored 0?** §11.4 masks it and raises an alarm. The
   alternative, scoring 0, removes the hack incentive but trains on grader bugs. With A13 forcing candidate-induced
   crashes into `CandidateFailure`, masking is the safer default, and M9 should track the masked-unattributed rate
   per task (a ceiling, e.g. ≤ 1%, as an acceptance check).
