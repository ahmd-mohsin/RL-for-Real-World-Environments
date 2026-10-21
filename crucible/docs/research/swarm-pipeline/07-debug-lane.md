# 07 The multi-language debugging lane

Spec: `00` (LongTask §1, generators §2, machine S0-S17 §4, verifier rigor §5, benchmark sealing §10), `04` (rebreak as
built, the World leak lessons). Target model (the only student): **Qwen/Qwen3.6-35B-A3B** (released 2026-04-15). Code: `factory/langs/` (the language layer), `factory/generators/debug.py` (the lane), `scripts/lh_debug.py`
(the funnel). Builder of every prompt: `claude-code:opus-5.5`.

## 1. What a task is

N real fix commits of one repository, all dated after the cutoff, are reverted together at the default branch's head B.
The agent gets a user's bug report (symptoms, how to reproduce, never the fix, the faulty function names or the hidden
test names) and the start tree A'. Each fix's own fail-to-pass tests are its hidden milestone; at least one milestone
needs another fault fixed first (an executed `requires` edge, S6). The oracle is B. It is `rebreak` (`00` §2) on every
language.

**Difficulty ladder.** 2-fault groups are the easy end; 3-4 interacting faults across several files and modules the
hard end. Difficulty is measured later, not here: the 35B in the 10-60% band over 8 attempts (training) or below 30%
(benchmark), failing on the intended skill, and a strong solver (Qwen3.5-397B / DeepSeek-V4.1-Flash) solving it (fairness).

**Contamination.** Fixes after `SINCE` = 2026-05-01 (after the target's release). `FALLBACK_SINCE` = 2025-09-01 is used
only where a repository's supply needs it; every candidate records `cutoff` and `contamination` (`post-release` |
`fallback`) and every fix date.

## 2. The language layer (`factory/langs/`)

`LangProfile` (`base.py`): source / test file / test path / protected / plumbing predicates, test definitions
(`test_defs`, `strip_tests`, `changed_defs`, `key_of_test`), suite targets, the World recipe, the run script and result
parser, the leak hooks. **The active profile** is a `contextvars` scope (`use(profile)`): the window machinery's
predicates (`executable.is_test_path` / `is_protected`, `windows_git.is_source` / `is_test_file` / `_defs` /
`key_of_test`, `windows.suite_targets`, `new_test_names`, `window_files`, `mined_of`, the leak probe call,
`windows_s3.world_s3`, `rebreak.allowed_file`, `pathcheck`'s plumbing) ask `active()` first. No profile or the Python
profile keeps the original code path byte for byte (`python.py` is an adapter: no regression).

| profile | tests and ids | runner, machine output | World (base pinned by digest) | leak probes beyond history and test names |
|---|---|---|---|---|
| python | pytest ids | pytest JUnit through the guard (unchanged) | `windows_world.ImageBook` (unchanged), tagged `crucible-lh-d:python-*` | the lane's own (pip download, installed copy) |
| go | `dir/x_test.go::TestX/sub` | `go test -json` per package; a package that does not build reports its tests `error` | `golang:1` (go1.27.3); module cache and the dependencies' build cache warm, the main module never compiled into the image; `GOPROXY=off` | `go mod download <module>@latest` fails; no copy of the module in the module cache |
| rust | `tests/x.rs::mod::name` (`src/a/tests.rs::a::tests::name`) | `cargo test -p <pkg> --test <bin>` / `--lib` per target, libtest lines, delimited per target | `rust:1` (cargo 1.99.1); deps compiled into `/opt/target`, every workspace-crate artifact deleted; `CARGO_NET_OFFLINE` | `cargo search` fails; no workspace crate in the registry |
| node | `file::describe::title` | vitest `--reporter=json` / jest `--json`, the repo's own runner config; a file that does not load reports its tests `error` | `node:22`; lockfile install (npm / pnpm / yarn with the node-modules linker) into `/opt/deps/node_modules`, linked into each run and the agent's workspace (git-excluded) | `npm view` fails; no package in the tree is the project itself |
| java | `src/test/java/a/FooTest.java::method` | `mvn -o test -Dtest=...`, surefire XML | `maven:3-eclipse-temurin-21`; `~/.m2` filled by `go-offline` and one full test run; single-module Maven only | the central repository is unreachable; no artifact of the project in `~/.m2` |

**Grading sandbox** (`runtime.LangSandbox`): `--network none`, the tree mounted read-only at `/src` and copied to `/w`
(compilers write next to the code; `/w` is the image's build path), tmpfs `/tmp`, CPU / memory / pid caps; the report goes
to a random path under `/tmp`, every other process is killed, then the report is printed between markers (the last block
counts). The root filesystem is the container's copy-on-write layer (the warm caches live there). Containers
`crucible-lh-d-*`, images `crucible-lh-d:<lang>-<repo>-<hash>`.

**Test identity.** The executed fail-to-pass set is the truth. The definition parsers (regex plus a string-, comment-,
template- and regex-literal-aware bracket matcher; no parser dependency) only *propose* which tests a commit added or
changed. `owned_tests`: a caught test is the fix's when the diff proposes it, when the diff proposes nothing for its file
(titles built at run time), or when its id names a case / golden file the fix added or changed (data-driven suites).

**Path rules** per language (protected: lockfiles, manifests, build and CI config, runner config; plumbing, hack
severity: `go.work` / `vendor/`, `build.rs` / `.cargo/`, `.npmrc` / `.pnpmfile.cjs` / `node_modules/`, `.mvn/`); the
path checks learn `go get`, `cargo add|fetch|search`, `npm install|view`, `yarn|pnpm add`, `mvn dependency:get`. The
epilogue names the language's plumbing (`epilogue_for`).

## 3. The lane (`generators/debug.py`)

1. **S1 source:** fix units since the cutoff (rebreak's message and file rules under the profile).
2. **S1 executed:** `fix_check` at B: B with only the fix's source reverted runs the pool's test files (or the whole
   suite); the fix is real when a test it owns passes at B and fails reverted (compiled languages: `failed`, never a
   build error). A suite that runs nothing at B is a World mismatch, recorded as such.
3. **Groups:** rebreak's executed interaction screen (one test case caught by two single reverts) and grouping (2..4,
   shared functions from the profile's definition diff, then shared files).
4. **A'..B'** (`synthesize`): source and test data reverted with the fixes; **revert mode** also reverts the fixes' test
   hunks (a new table row, a changed expectation, a new test: gone from A'); when that does not apply or leaves a tree
   that does not build, **strip mode** removes only the changed test definitions. The fixes' changelog lines are removed.
   B' == B, checked.
5. **Pre-checks:** the agent's own view of A' builds; B's test files build against A' (a fix that adds API is refused).
6. **`windows.validate_window`:** World, leak probes, fail-to-pass with reruns, prefix owners, the executed prober,
   revert mutants.
7. **Visible checks:** no milestone test may fail in the agent's own tree; the ones that do are stripped from A' and the
   window is validated again (one round), else refused.
8. **Gates** (`Debug.gates` = `base.deterministic_gates` in the profile's scope): S6, S5, S8, S3 (60 reruns), S10 (30 + 30
   oracle grades).
9. **Prompts** (`author_debug`): the builder's bug report `prompts/<id>.json` through the window lane's lint, `lint_debug`
   (faulty function names, hidden test names, versions) and the two-way ledger; provenance recorded.

## 4. Sources and sealing

Permissive licences only (MIT, Apache-2.0, BSD, ISC, Unlicense), read from the clone. Registered in
`windows.DEBUG_REPO_URLS`; repositories sharing an org with a held-out source are left out (alecthomas/kong: alecthomas
owns the held-out voluptuous). **Held out for the debug benchmark** (chosen after the S1 count, before executed mining;
every repository of an org together): `urfave-cli` (go), `h3`, `unimport`, `ufo` (node, unjs), `minijinja` (rust),
`beartype` (python). They go to `gen/store/heldout`; the lease refuses them to training roles.

## 5. Funnel and results (run of 2026-10-20, CPU only, $0)

Sources: 98 clones scouted (licence from the clone, language detected, fix units counted), 56 mined at least once.
Yield ledger per language (the last row of each candidate and stage; `data/debug/funnel.jsonl`, table in
`07-debug-lane/funnel_table.md`):

| language | fix units | real fixes | interacting groups | validated windows | dependent + valid mutants | all CPU gates | prompt (S7) | top drop reasons |
|---|---|---|---|---|---|---|---|---|
| go | 503 | 237 | 23 | 17 | 10 | 9 | 7 | its source revert does not apply at B (175); its revert only stops tests from building (51); no own test fails after the revert (25); the suite does not run at B (13) |
| java | 93 | 38 | 0 | 0 | 0 | 0 | 0 | its source revert does not apply at B (31); no own test fails after the revert (13); its revert only stops tests from building (10); masked: sandbox did not start (1) |
| node | 497 | 194 | 9 | 9 | 5 | 2 | 4 | its source revert does not apply at B (96); no own test fails after the revert (79); the suite does not run at B (46); the World image of B does not build (40) |
| python | 132 | 39 | 3 | 3 | 3 | 3 | 3 | its source revert does not apply at B (52); the suite does not run at B (16); no own test fails after the revert (12); the World image of B does not build (8) |
| rust | 154 | 63 | 9 | 8 | 1 | 1 | 1 | no own test fails after the revert (41); its source revert does not apply at B (35); the suite does not run at B (14); independent (6) |

**Candidates that pass every CPU gate** (S6 executed edge, S5 clean-World solvability and null agent, S8 floors 0 /
oracle and its unit steps 1 / every mutant fails its milestone, S3 build + existing suite + leak probes + 60 clean reruns,
S10 60 oracle grades) **and the prompt checks (S7)**: 13 across 4 languages, 3 of them held out.

| id | lang | split | faults | executed edges | oracle lines | milestone tests | level | cutoff |
|---|---|---|---|---|---|---|---|---|
| db-go-go-toml-a02114e3 | go | train | 4 | 1 | 157 | 45 | hard | post-release |
| db-go-mvdan-sh-04056a74 | go | train | 4 | 2 | 42 | 50 | hard | post-release |
| db-go-mvdan-sh-dc951aeb | go | train | 4 | 2 | 68 | 41 | hard | post-release |
| db-go-mvdan-sh-02a801cd | go | train | 4 | 2 | 75 | 49 | hard | post-release |
| db-go-mvdan-sh-5e983709 | go | train | 4 | 1 | 32 | 63 | hard | post-release (shares 3 fixes with 04056a74) |
| db-go-echo-61eca05b | go | train | 2 | 1 | 45 | 14 | easy | post-release |
| db-python-black-ec71b918 | python | train | 3 | 1 | 22 | 4 | hard | post-release |
| db-python-black-ae5830b7 | python | train | 4 | 1 | 33 | 5 | hard | fallback (shares 3 fixes with ec71b918) |
| db-python-black-90117fa0 | python | train | 3 | 1 | 69 | 3 | medium | fallback |
| db-rust-pulldown-cmark-e072a230 | rust | train | 4 | 1 | 140 | 16 | hard | post-release |
| db-go-urfave-cli-2b0adb14 | go | heldout | 3 | 1 | 52 | 46 | hard | post-release |
| db-node-h3-a6cd4654 | node | heldout | 2 | 1 | 34 | 4 | easy | post-release |
| db-node-unstorage-d7aee0b7 | node | heldout | 2 | 1 | 50 | 3 | easy | post-release |

Also through every CPU gate but refused by the builder at S7 (`07-debug-lane/prompt_refusals.json`): minify and esbuild.
Valid but failing a gate: axios-9ff8c996 and lo-1e8f3c39 (S3: load-dependent existing tests flipped again after the
one exclusion the rule allows). Not gated when the run stopped: axios-61cd331d (prompt ready) and axios-56014db8 (refused).

**Intent.** Every candidate carries its intent card (`window.json` -> `debug.intent`, list in `07-debug-lane/candidates.json`):
all 13 test `multi_fault_localization`, `interacting_faults`, `reproduce_from_report` and `regression_safety`; level
hard (3-4 faults over 2+ source files) for 10, easy (2 faults) for 3. The family-level skill cards are
`07-debug-lane/intent_cards.json` (`candidate`: no model failure observed yet). The held-out three test the same four
skills on disjoint repositories and organisations (Go and TypeScript); the training set has no held-out Rust or Python
instance yet (minijinja and beartype produced no valid window).

**What limits yield** (in order):
1. *Reverts that no longer apply at B* (about 40% of real-fix candidates): code moved on after the fix. Earlier B
   points recover some (echo's only group came from B = head - 45 days).
2. *Interacting fixes are rare*, as P5a found for Python: 23 interacting groups from 237 real Go fixes, 9 from 194
   Node fixes, none in Java (38 real fixes; commons-lang's one candidate group held no pair caught by two reverts).
3. *Coarse tests collapse milestones*: a Go table test without subtests (yq's `TestJSONScenarios`, validator's
   `TestTranslations`) holds both fixes' cases in one test id, so the window has one milestone.
4. *Fixes that add API* (51 in Go): their revert stops tests from building; never a behavioural fault.
5. *Timing*: the interacting tests of three ron/bat groups were timing tests (excluded by rule), leaving no edge.
6. *World mismatches* we fixed while running (section in `04`): packaging keys, runner configs, package managers,
   Rust workspaces, Java providers. Remaining: pydantic's suite does not finish in the sandbox, pygments' World does not
   build, Go multi-module repositories (koanf) are out of scope.

**Gap to the targets.** The lane, its gates and 13 accepted candidates are in place; the >= 30 target was not reached
in this run (mining and gating ran about 11 hours at a load of 15-20 on 14 CPUs; S3 and S10 alone take 20-60 minutes per
candidate). The levers, in order of yield per hour: more Go and TypeScript repositories of the mvdan/sh, go-toml,
axios, black kind (formatters, parsers, protocol libraries with table-driven or snapshot tests); the fallback cutoff
on the dense repositories; and two more gate workers on a quieter host.

## 6. Targeting (S15) on Tinker (run of 2026-10-20, $201 of the $250 cap)

Code: `scripts/lh_debug_target.py` (stages `pilot`, `student`, `frontier`, `regrade`, `worldfix`, `diag`, `audit`, `mine`,
`report`, `summary`); framework pieces in `factory/tinker.py` (`benchmark(meter=)`, `regrade`, `policy_id`),
`factory/targeting.py` (`KeepRule`, `keep_verdict`), `factory/traceaudit.py` (`honest_trace_facts`, `budget_binding`,
`regrade_source`), `factory/generators/debug.py` (`failure_skill`), `factory/audit.py` (`quote_retries`),
`factory/langs/{node,runtime}.py` (the policy World fix). Data: `data/targeting/debug/` (durable); summary committed as
`runs/s15_debug_targeting.json`. W&B (`cruciblee`): project `crucible-factory`, run `factory-debug-s15-f62a27bf` (with
the traces) and `factory-debug-s15-63051dee` (the tables after the held-out exclusion below); 156 episodes as Weave
traces, 94 of them with their trace audit attached (https://wandb.ai/cruciblee/crucible-factory/weave).

**Setup: the RL environment exactly.** `factory.tinker.benchmark`: `make_env` (`WorkspaceHarness` + `LLMToolPolicy`),
the task's own Docker World (`--network none`, 2 CPU / 2 GB), the hidden verifier in a fresh grading container, the
checkpoint-Φ sidecar (every 10 tool calls, cap 12). Student `Qwen/Qwen3.6-35B-A3B:peft:262144` (the only student; the
256K variant because a 2-fault episode already reached 74K tokens of context), k = 8, temperature 1, `max_tokens` 4096
(the factory's student default; 2048 truncated 3 of 169 replies in the pilot), **thinking off** (recorded), tool limit
150, agent timeout 2700 s, rollout timeout 4500 s. Frontier: Qwen3.5-397B-A17B and DeepSeek-V4.1-Flash (both
`:peft:262144`), 2 attempts each, same settings. One Tinker meter for the whole phase (`spend.jsonl`, cap 250). Judges:
GLM-5.3 (`auditor`) and GLM-5.3-Flash (`judge`) from `models.DEFAULT_ROLES` (`ModelBook` refuses a same-family judge;
the benchmark policy id now names the model, so `audit.audit` sees the student's family), the seeded control bank in
every batch (every batch trusted: every control labelled right). Final cause: a deterministic fact first (a limit stop
is `budget`), else the judge's trusted label cross-checked by the auditor's; a dispute, or a World or harness fact
beside a skill label, never counts as an on-intent skill gap.

**Pilot** (echo, 2 student + 1 frontier attempts, tool limit 100): student $0.42 per episode, 80-90 tool calls, 6-10
min wall, of which about 330 s is model time (Tinker latency, not Docker, dominates); 397B $1.61 per episode; Docker load
small (3 containers, 0.3 GiB; host load 15-17 from other work). It found the first verifier bug (below). Projected $101
plus the audit.

**Run.** 156 episodes (104 student, 52 frontier), plus 16 student episodes in the broken node World (kept as evidence,
`arms/student-world0`) and 4 diagnostic episodes; up to 12 Worlds at once (mean 8, host load mean 16, peak 48 while the
other agent's gate jobs ran). No episode timed out from load; 2 student episodes were masked (the runtime's
answer-exposure guard on an absolute `/tests/...` path). From the meter's per-episode rows: student $0.79 per episode,
397B $2.43, DeepSeek $1.07; wall per episode: student 20 min, 397B 11 min, DeepSeek 17 min.

| task | lang | split | faults | student pass@1 (8) [95% CI] | mean Φ | frontier (397B; DS) | on-intent skill gap / labelled | dominant failure modes | verdict |
|---|---|---|---|---|---|---|---|---|---|
| go-echo-61eca05b | go | train | 2 | 7/8 = 0.88 [0.53, 0.98] | 0.94 | 2/2; 1/2 | 0/1 | - | too easy |
| go-go-toml-a02114e3 | go | train | 4 | 0/8 [0, 0.32] | 0.25 | 0/2; 1/2 | 3/8 | budget 5, stalled 3 | too hard (simplify) |
| go-mvdan-sh-02a801cd | go | train | 4 | 0/8 | 0.22 | 0/2; 0/2 | 0/8 | budget 7 | unfair at 150 calls |
| go-mvdan-sh-04056a74 | go | train | 4 | 0/8 | 0.19 | 0/2; 0/2 | 0/8 | budget 8 | unfair at 150 calls |
| go-mvdan-sh-5e983709 | go | train | 4 | 0/8 | 0.41 | 0/2; 0/2 | 0/8 | budget 6 | unfair at 150 calls |
| go-mvdan-sh-dc951aeb | go | train | 4 | 0/8 | 0.34 | 0/2; 1/2 | 1/8 | budget 6 | too hard (simplify) |
| go-urfave-cli-2b0adb14 | go | heldout | 3 | 0/8 | 0.63 | 0/2; 0/2 | 8/8 | stalled at the interacting fault 8 | unfair |
| node-h3-a6cd4654 | node | heldout | 2 | 0/8 (fixed World) | 0.00 | 0/2; 1/2 | 6/8 | claimed done 3, never progressed 2 | World bug (open) |
| node-unstorage-d7aee0b7 | node | heldout | 2 | 3/8 = 0.38 [0.14, 0.69] (fixed World; 0/8 before) | 0.75 | 2/2; 2/2 | 5/5 | stalled (regression) 3, claimed done 2 | out of the benchmark band |
| python-black-90117fa0 | python | train | 3 | 0/8 | 0.21 | 0/2; 0/2 | 0/8 | budget 7 | unfair at 150 (DS 2/2 at 300) |
| python-black-ae5830b7 | python | train | 4 | 0/7 | 0.29 | 0/2; 0/2 | 0/7 | budget 5 | unfair at 150 calls |
| python-black-ec71b918 | python | train | 3 | 0/7 | 0.33 | 0/2; 0/2 | 1/7 | budget 5 | unfair at 150 calls |
| rust-pulldown-cmark-e072a230 | rust | train | 4 | 0/8 | 0.16 | 0/2; 0/2 | 0/8 | budget 6 | unfair at 150 calls |

**No task passes the keep rule** (0 training, 0 benchmark). The binding constraint is the environment's budget: 67 of
104 student episodes and 21 of 52 frontier episodes stop at the 150-call limit. A limit stop is the `budget` cause by
the factory's rule: routed to the limits owner, never counted as a skill gap; the keep rule used the env's budget as is.
`traceaudit.budget_binding` reads the Φ series at the cap: in 40 of the 71 budget-stopped student failures Φ was still
rising in the last 30% of the calls (the cap bound them), in 31 it was flat (stuck). **Diagnostic** (never a keep-rule
input): DeepSeek-V4.1-Flash at a 300-call limit solved python-black-90117fa0 2/2 and reached Φ = 1 on mvdan-sh-02a801cd
2/2 (both lost the full-completion credit for an edited test file), 187-202 calls per episode, $2.05 each. The hard Go,
Python and Rust tasks need about twice the budget; at 150 calls they measure the budget, not the intended skills.

**What the run found and fixed** (each with a regression test):
1. *Path check false positive, heredoc bodies* (verifier; severity hack, so a score of 0): a Go reproduction written
   through `cat > x.go << EOF` with `import "net/http"` was read as a run of the `http` client. Here-document bodies are
   no longer commands unless a shell reads them (`pathcheck.strip_heredocs`). Both judges independently called the
   pilot's zeroed correct episode a verifier defect.
2. *Path check false positive, `go mod init`* (install gate): a scratch module for a reproduction installs nothing; only
   `go mod download | tidy | vendor` count. The affected episodes were re-graded from their stored patch and trail
   (`factory.tinker.regrade` over `Runner.regrade`; the trace readers take the trail from the source attempt).
3. *Node policy World* (World bug): node:22 has no `python3`, so `repo.read_file` / `list_dir` failed in every episode,
   and corepack's pnpm cache sat under the build's `HOME`, so `pnpm` tried to download itself offline. Fixed as a derived
   policy World (`runtime.policy_world`, accepted on executed evidence: the reproduction fails in the pinned World and
   passes in the fixed one, the file tools and a test file run, the leak probes pass; the grader keeps the pinned image;
   `world_versions.jsonl`) and in the recipe (`node-4`). unstorage went from 0/8 to 3/8. Still open for h3: its own
   `pnpm test` (pnpm 12 supply-chain checks) verifies the lockfile against the registry and hangs until the 600 s tool
   timeout (8 of 8 episodes). The judges labelled those failures skill gaps, so it is recorded as an open World issue
   (`world_issues.json`) and the task goes back to the World owner.
4. *Accounting*: concurrent arms on one meter each reported the meter's delta (about 2.7x overcount); an arm now reports
   its own sampler's spend, and per-episode dollars come from the meter rows inside each run's journal window.
5. *Family guard*: the benchmark policy id `bench-<model>` read as family `bench`, so a Qwen judge on Qwen traces would
   not have been refused; ids are now `solver:tinker:<model>@tinker`. A `:peft:262144` variant of the target is the
   target (final numerics).
6. *Judges' quotes*: GLM-5.3 and -Flash joined quote spans with `...`, so every label was rejected; a non-verbatim quote
   is now asked again once with the rejection fed back (`AuditConfig.quote_retries`; the verbatim check is unchanged).
7. *A World tool that lacks a command* is a `world_fault` fact (the judges missed the node one).

**Honest-trace checks** (`traceaudit.honest_trace_facts`, student episodes): a STATUS.json over-claim (p >= 0.5 for a
requirement whose hidden tests fail) in 21 episodes (8 of 8 on urfave-cli); a "tests pass / verified" final message with
no test run after the last code edit in 3; a protected or hidden path in the final diff in 2. Also seen: `git log --all`
by DeepSeek (path:history, scored 0), `go get` of the project itself by the student (path:network, scored 0), `pnpm
install` attempts on node (install gate).

**Failure mining.** 94 findings (`findings.jsonl`), 70 routed (57 budget, 2 infrastructure, the rest budget-capped
variants), so few skill findings remain. Builder-intent requests come from **training tasks only** (a held-out task
never informs what is built for training, V9): one request, `stalled` ("Carries a multi-step change through to its
last dependent step"), 5 episodes over go-toml, mvdan-sh-dc951aeb and black-ec71b918 (skills: multi-fault localization
3, interacting faults 2), status `awaiting_human`. The held-out traces show the same pattern (urfave-cli: m01 and m02
fixed, the dependent m04 never) and claimed-done reports on node; they stay sealed.

**Next, by owner.** Limits: re-measure the hard tasks at a 300-call limit (or with compaction) before any keep decision.
World: h3's offline `pnpm`. S2: echo is too easy; go-toml and mvdan-sh-dc951aeb are frontier-solvable at 0 for the
student (simplify(locate|prerequisites) candidates). unstorage would be a training-band task (0.38, 5/5 on-intent, fair)
but is sealed held-out.
