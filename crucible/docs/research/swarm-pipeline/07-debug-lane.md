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
| node | `file::describe::title` | vitest `--reporter=json` / jest `--json`, the repo's own runner config and the node flags its runner script passes (`runner_node_flags`, allow-list: `--experimental-vm-modules` for jest's ESM support) in `NODE_OPTIONS`; a file that does not load reports its tests `error` | `node:22`; lockfile install (npm / pnpm / yarn with the node-modules linker) into `/opt/deps/node_modules`, linked into each run and the agent's workspace (git-excluded) | `npm view` fails; no package in the tree is the project itself |
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
   (faulty function names, hidden test names, versions) and the two-way ledger; provenance recorded. An optional
   `repro` per requirement is linted (`debug_repro.lint_repro`) and shown in the prompt (§6.2).
10. **S7r** (`debug_repro.s7r`, `lh_debug.py repro`): each requirement's reproduction runs in the task World at A' (it
    must show the bug) and at the oracle B (it must pass); a missing repro is `unverified`, never a failure (§6.2).

## 4. Sources and sealing

Permissive licences only (MIT, Apache-2.0, BSD, ISC, Unlicense), read from the clone. Registered in
`windows.DEBUG_REPO_URLS`; repositories sharing an org with a held-out source are left out (alecthomas/kong: alecthomas
owns the held-out voluptuous). **Held out for the debug benchmark** (chosen after the S1 count, before executed mining;
every repository of an org together): `urfave-cli` (go), `h3`, `unimport`, `ufo` (node, unjs), `minijinja` (rust),
`beartype` (python); a second draw `go-zero` and more of unjs (`unstorage`, `ofetch`, `crossws`, `c12`); a third draw
(after the S1 count of 80 new clones, before their executed mining) of orgs with nothing in any training source:
`kong`, `chroma`, `participle` (alecthomas, already sealed by the Python lane's voluptuous), `excelize`, `carbon`,
`templ`, `caddy`, `go-git`, `resty` (go), `gleam`, `rhai`, `comrak` (rust), `liquidjs`, `stylelint`, `i18next`,
`cron-parser` (node), `aiohttp`, `yarl`, `urllib3` (python). They go to `gen/store/heldout`; the lease refuses them to
training roles. The same draw registered 40 training sources (dense parser, formatter and protocol repositories:
`golang-net`, `golang-crypto`, `fasthttp`, `cel-go`, `sqlparser-rs`, `calamine`, `interceptors`, `sqlglot`, `websockets`
and others in `windows.DEBUG_REPO_URLS`); they are not mined yet.

The store lives at `CRUCIBLE_DEBUG_ROOT` (the main checkout's `crucible/data/debug` when the lane runs from a
worktree), else the tree's own `data/debug`; `lh_debug.py scout --repos @file` scouts a list and merges into
`scout.json`.

## 5. Funnel and results (runs of 2026-10-20, CPU only, $0)

Sources: 178 clones scouted (licence from the clone, language detected, fix units counted), 72 mined at least once
(first run 56; second run the 16 held-out repositories of the third draw). Yield ledger per language (the last row of
each candidate and stage; `data/debug/funnel.jsonl`, table in `07-debug-lane/funnel_table.md`; the prompt column
counts written prompts, gated or not):

| language | fix units | real fixes | interacting groups | validated windows | dependent + valid mutants | all CPU gates | prompt (S7) | top drop reasons |
|---|---|---|---|---|---|---|---|---|
| go | 594 | 288 | 28 | 22 | 14 | 13 | 11 | its source revert does not apply at B (187); its revert only stops tests from building (61); no own test fails after the revert (43); the suite does not run at B (13) |
| java | 93 | 38 | 0 | 0 | 0 | 0 | 0 | its source revert does not apply at B (31); no own test fails after the revert (13); its revert only stops tests from building (10); masked: sandbox did not start (1) |
| node | 575 | 242 | 9 | 9 | 5 | 2 | 4 | its source revert does not apply at B (102); no own test fails after the revert (83); the suite does not run at B (66); the World image of B does not build (40) |
| python | 188 | 39 | 3 | 3 | 3 | 3 | 3 | the World image of B does not build (56); its source revert does not apply at B (52); the suite does not run at B (24); no own test fails after the revert (12) |
| rust | 226 | 99 | 11 | 9 | 1 | 1 | 1 | its source revert does not apply at B (59); no own test fails after the revert (51); the suite does not run at B (14); independent (7) |

**Candidates that pass every CPU gate** (S6 executed edge, S5 clean-World solvability and null agent, S8 floors 0 /
oracle and its unit steps 1 / every mutant fails its milestone, S3 build + existing suite + leak probes + 60 clean reruns,
S10 60 oracle grades) **and the prompt checks (S7)**: 17 across 4 languages, 7 of them held out. `faults` counts
milestones (a group's fix whose checks were all stripped as visible stays reverted in A' without a milestone:
excelize-824ac2ce reverts 4 fixes, excelize-65be17e3 4). `contamination`: every fix after the cutoff it was mined under,
or the marked fallback.

| id | lang | split | faults | executed edges | oracle lines | milestone tests | level | contamination |
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
| db-go-excelize-824ac2ce | go | heldout | 2 (4 fixes) | 1 | 59 | 286 | easy | post-release |
| db-go-excelize-65be17e3 | go | heldout | 3 (4 fixes) | 2 | 77 | 3 | medium | post-release |
| db-go-carbon-93371b6f | go | heldout | 3 | 2 | 39 | 11 | medium | post-release |
| db-go-carbon-a783fb3a | go | heldout | 2 | 1 | 11 | 7 | easy | post-release |

Also through every CPU gate but refused by the builder at S7 (`07-debug-lane/prompt_refusals.json`): minify and esbuild.
Valid but failing a gate: axios-9ff8c996 and lo-1e8f3c39 (S3: load-dependent existing tests flipped again after the one
exclusion the rule allows); axios-61cd331d (S6, S5, S8, S3 pass; S10: 1 oracle false negative in 60 grades at load
level 4, flake upper bound 0.077 > 0.05, host load 16-21 on 14 CPUs; not re-run). Not gated: axios-56014db8 (refused).

**Second run (held-out draw 3).** 16 repositories mined (excelize, kong, chroma, carbon, templ, participle, liquidjs,
stylelint, i18next, cron-parser, rhai, comrak, gleam, aiohttp, yarl, urllib3): 4 windows, all through every CPU gate and
S7 (excelize: 37 fix units, 15 real, 11 interacting, 3 groups, 2 windows; carbon: 17, 12, 5, 2 groups, 2 windows).
No interacting pair in kong (13 real fixes), liquidjs (20), cron-parser (21), comrak (8), i18next (7), templ, chroma,
rhai, participle; gleam's 2 groups were refused (hidden checks visible in the agent's tree; independent milestones).
No World: aiohttp and yarl (the image does not build: compiled extensions), urllib3 (no JUnit report), stylelint (every
test file failed to load: its jest needs `--experimental-vm-modules`, now carried by `runner_node_flags`; the repository
is not re-mined). Registered and not mined when the run stopped (scope): caddy, go-git, resty (held out) and the 40 new
training sources; the fallback cutoff on the dense held-out repositories is the next lever.

**Intent.** Every candidate carries its intent card (`window.json` -> `debug.intent`, list in `07-debug-lane/candidates.json`):
all 17 test `multi_fault_localization`, `interacting_faults`, `reproduce_from_report` and `regression_safety`; level
hard (3-4 faults over 2+ source files) for 10, medium for 3, easy (2 faults) for 4. The family-level skill cards are
`07-debug-lane/intent_cards.json` (`candidate`: no model failure observed yet). The held-out seven test the same four
skills on disjoint repositories and organisations (Go and TypeScript); no held-out Rust or Python instance yet.

**Fairness flags** (`debug.fairness_flags`, on each intent card and in `candidates.json`; for review, never a gate):
`internal_function_check` when a hidden milestone test is named after a changed function that is not public API (an
exported Go name is), so a correct fix made elsewhere could fail it; `visible_test_data` when strip mode leaves the
fixes' changes to test files in the agent's tree (re-derived for windows built before the field existed: mvdan-sh
04056a74 and 5e983709 keep `syntax/filetests_test.go`, dc951aeb nothing). Flagged among the 17: black ec71b918 and
ae5830b7 (`test_lines_with_leading_tabs_expanded` calls the internal helper; each milestone also holds the public
`test_simple_format[docstring_tabs]` check, so the milestone is satisfiable through the public API); excelize-824ac2ce
(its m03 holds the whole package's tests, two of them named after internal parser functions, because the reverted panic
stops the test binary); excelize-65be17e3, echo-61eca05b and two mvdan-sh windows (visible test data); urfave-cli, h3
and unstorage (tests named after changed functions; h3's and unstorage's are public API, a false positive of the name
rule).

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

**Gap to the targets.** 17 accepted candidates (10 training, 7 held out) against >= 30 overall, >= 10 held out with the
kernel lane's 5 pending GPU, >= 8 new held-out debug ones and >= 12 more training ones: the held-out total reaches 12 with
the kernel lane (7 + 5) but the second run added 4 held-out debug candidates, not 8, and no training candidate (only
held-out repositories were mined before the scope stop). Mining and gating
ran about 11 + 6 hours at a load of 15-21 on 14 CPUs shared with the kernel lane and S15 targeting; S3 and S10 take
20-60 minutes per candidate. The levers, in order of yield per hour: the registered, unmined Go and Python training
sources (golang/net and crypto, fasthttp, sqlglot, websockets: dense parsers and protocol code with many fixes after the
cutoff), caddy / go-git / resty for the held-out side, the fallback cutoff on the dense repositories (kong, liquidjs,
cron-parser have 13-21 real fixes and no interacting pair since 2026-05-01), stylelint re-mined with the node flag, and
more gate workers on a quieter host.

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

### 6.1 Targeting at 300 tool calls (run of 2026-10-20, $326 of the $450 cap; stopped early by the user)

**The env budget is now 300 tool calls** (user decision after §6): `generators.debug.ENV_BUDGET` (tool limit 300, agent
timeout 5400 s, rollout timeout 9000 s, Φ every 20 calls with cap 15, so the periodic points reach call 280; every
value scales the 150-call run's numbers). The keep rule, the benchmark and RL use the same budget.
`lh_debug_target.py` defaults to it and records it in every report and W&B run (`settings.env_budget`,
`is_lane_budget`, tag `tool_limit:300`, summary `s15/tool_limit`). Everything else is unchanged from §6: the 256K
variants, thinking off, k = 8 / 2 + 2, the same judges, controls, family rules and keep rule. A re-measurement draws
fresh seeds (`--seed-offset 300`), so its run ids (the Weave trace keys) never collide with §6's. Data:
`data/targeting/debug-300/` (sealed). Summary: `runs/s15_debug_targeting_t300.json`, with held-out details sealed
(`sealed`). W&B: `crucible-factory` run `factory-debug-s15-t300-cac4469d`, 132 episode traces with their audits.

**Scope.** 11 of the 15 planned tasks completed every arm. Order: the 4 new held-out tasks, then the training tasks. The
two-task projection was $409, under the cap. The user stopped the phase before black-ec71b918's and pulldown-cmark's
student arms finished and before urfave-cli and h3 ran. Those 4 tasks are excluded (their partial arms are kept but not
reported). Carried over from §6, not re-run: echo (too easy at 150) and unstorage (3/8 at 150, above the benchmark
band; more budget only raises it).

| task | lang | split | faults | student pass@1 (8) [95% CI] | mean Φ | frontier (397B; DS) | budget stops (student; frontier) | on-intent / labelled | verdict | at 150 calls (student; frontier; verdict) |
|---|---|---|---|---|---|---|---|---|---|---|
| go-carbon-93371b6f | go | heldout | 3 | 3/7 = 0.43 [0.16, 0.75] | 0.43 | 0/2; 1/2 | 0; 0 | 4/4 | out of band (not below 0.3) | new |
| go-carbon-a783fb3a | go | heldout | 2 | 7/8 = 0.88 [0.53, 0.98] | 1.00 | 2/2; 1/2 | 0; 0 | 1/1 | too easy | new |
| go-excelize-65be17e3 | go | heldout | 3 | 0/8 [0, 0.32] | 0.67 | 0/2; 0/2 | 1; 0 | 7/8 (judges) | **spec defect** | new |
| go-excelize-824ac2ce | go | heldout | 2 | 0/8 [0, 0.32] | 0.50 | 0/2; 0/2 | 0; 0 | 8/8 (judges) | **spec defect** | new |
| go-go-toml-a02114e3 | go | train | 4 | 0/8 [0, 0.32] | 0.47 | 0/2; 2/2 | 0; 0 | 8/8 | too hard (simplify) | 0/8; 0/2, 1/2; too hard |
| go-mvdan-sh-02a801cd | go | train | 4 | 0/8 | 0.38 | 0/2; 0/2 | 5; 0 | 3/8 | unfair | 0/8; 0/4; unfair |
| go-mvdan-sh-04056a74 | go | train | 4 | 0/8 | 0.34 | 0/2; 0/2 | 4; 0 | 4/8 | unfair | 0/8; 0/4; unfair |
| go-mvdan-sh-5e983709 | go | train | 4 | 0/8 | 0.41 | 0/2; 0/2 | 2; 0 | 5/8 | unfair | 0/8; 0/4; unfair |
| go-mvdan-sh-dc951aeb | go | train | 4 | 0/8 | 0.44 | 0/2; 0/2 | 0; 0 | 8/8 | unfair | 0/8; 0/2, 1/2; too hard |
| python-black-90117fa0 | python | train | 3 | 1/8 = 0.13 [0.02, 0.47] | 0.50 | 1/2; 0/2 | 6; 0 | 2/7 | off intent (budget) | 0/8; 0/4; unfair |
| python-black-ae5830b7 | python | train | 4 | 0/8 | 0.47 | 0/2; 0/2 | 6; 0 | 2/8 | unfair | 0/7; 0/4; unfair |

**No task passes the keep rule** (0 training, 0 benchmark). The budget no longer binds the frontier solvers: 0 of 54
frontier episodes hit the limit, against 21 of 52 at 150. For the student, 24 of 88 hit it, against 67 of 104; 27 of
the 36 student failures at or near the limit were still gaining Φ in the last 30% of their calls (`budget_binding`),
mostly on black and mvdan-sh. black-90117fa0 is the only training task in the band (1/8, and the 397B passes it). 6 of
its 8 student episodes ran to 300 calls, the pass among them, and budget stops stay `budget`, so the on-intent share is
2/7. The keep rule is applied as is.

**What the run found.**
1. *Spec defect in both excelize windows (held out).* All 28 episodes across the three models pass every milestone but
   one, mostly ending on their own well under the limit. The cause: strip mode leaves reverted fixes without a
   milestone (their checks were visible), and one milestone's hidden table test also checks them, while the bug report
   never describes them. The executed check proves it: B with only those fixes' source reverted fails that milestone and
   passes the others, in the pinned World. A frontier patch fails exactly the undescribed cases. Both GLM judges
   labelled every failure `skill_gap` (they never see the hidden tests). New: `generators.debug.disclosure_check`
   (language-general, through the task's own grader) and the stage `disclose`. Of the 15 tasks it flags excelize ×2 and
   urfave-cli. On review urfave's report does describe its silent fix (R3), so only the excelize tasks are defects,
   recorded `kind: spec_defect` in `world_issues.json`. The keep rule now has a `spec_issue` input (verdict
   `spec_defect`, owner the builder). Open lane item: feed the builder the silent units a milestone needs, or keep such
   units out of A'.
2. *h3 World fixed (pnpm 12 offline).* `pnpm <script>` first checks the installed tree against the lockfile
   (`verify-deps-before-run`) and installs when they differ. They always differ, because the recipe deletes the
   project's own installed copies (the leak rule). So every `pnpm test` fetched h3's release tarballs with retries until
   the 600 s timeout. With a network, it would have re-installed the very copy the deletion closes. Fix:
   `ENV pnpm_config_verify_deps_before_run=false pnpm_config_offline=true` (`node.PNPM_OFFLINE`, recipe `node-5`,
   policy World `-pw645aaf43`). Accepted on evidence: the repo's own `pnpm test` reaches the registry in the pinned
   image and hangs in §6's World; in the fixed World it finishes in 5 s. Both hidden milestones fail at A' and pass at B
   there, the leak probes pass, and the grader keeps the pinned image. Its 2 failing tests fail at B too (`post_fail`,
   node 22). h3 was not re-measured (stopped).
3. *mvdan-sh is functionally frontier-solved.* DeepSeek reaches Φ = 1 in 6 of 8 episodes. It loses the full pass to
   `path:install` (5: `go mod tidy` inside a scratch reproduction module under /tmp with a `replace` to the workspace,
   offline, `GOPROXY=off`) or to `path:protected` (2: edited test files). The rule counts `tidy` as an install (§6, item
   2), so the tasks read `unfair`. Whether `tidy` in a scratch module is an install is open for the verifier owner. It
   changes no keep decision: the student is 0/8 on all four.
4. *Fact false positive.* `world_fault` "command 'foo' not found in the World" fires on mvdan-sh, a shell
   interpreter whose test scripts print exactly that. It marked 1 label disputed on 5e983709 (no verdict changes). Open
   for `traceaudit` (owner of the World-tool fact).
5. *Clones.* The 56 generator clones of the first mining run borrowed objects (`alternates`) from a deleted worktree;
   repointed to the same clones in `data/debug/src` (every task's oracle steps resolve again).

**Honest-trace checks** (student): a STATUS over-claim in 38 episodes (35 training), claimed-but-unrun tests in 4.
DeepSeek ran `pip download black` on black-90117fa0 at Φ = 1 (`path:network`, scored 0). One student episode was
masked (the answer-exposure guard).

**Spend and cost per episode at 300 calls.** Total $326.11: student $165.94, 397B $84.31, DeepSeek $54.14, GLM-5.3
$17.83, GLM-5.3-Flash $3.88. About $15 of the student's total went to the cancelled arms. Per episode: student $1.72,
27.6 min wall, 208 calls (§6: $0.79, 20 min); 397B $3.01, 13.6 min, 132 calls ($2.43, 11 min); DeepSeek $2.05, 26.8
min, 183 calls ($1.07, 17 min). Host: 10.6 Worlds at once on average (max 18), load mean 9.7.

**Failure mining** (training tasks only). 77 findings, 25 routed (20 budget). 3 builder-intent requests, all
`awaiting_human`:
- `stalled`, "Carries a multi-step change through to its last dependent step": 28 episodes over go-toml, mvdan-sh ×4 and
  black-ae5830b7 (interacting faults 18, multi-fault localization 9).
- `claimed_done_without_progress`, "Checks its own work against the request before it reports the work done": 2.
- `never_progressed`, "Finds where a requested change belongs ... and makes its first step pass": 2, black-90117fa0.

Held-out findings stay sealed.

**Next, by owner.** S2 / lane: the excelize reports (describe the silent fixes, or rebuild the windows without them),
then re-measure; carbon-a783fb3a is too easy. Verifier: `go mod tidy` in a scratch module. traceaudit: the shell
`not found` fact. Limits: black and mvdan-sh still bind at 300 for the student (27 of 36 still gaining). Remaining at
300: black-ec71b918, pulldown-cmark, urfave-cli, h3 (in its fixed World).

### 6.2 Easing support: calibration in the brief, reproductions per requirement (S7r) (2026-10-21, $0)

Two builder aids for the easing round (handoff §5), patterns from rl-env-use (MIT; `tasks.py` 185-208, `scout.py`
102-193). No model call; S7r needs Docker.

**Calibration in the brief** (`factory/calibration.py`, `generators.debug.builder_calibration`, printed first by
`lh_debug.py brief <id>`; `--targeting` names the S15 reports, default `data/targeting/debug` then `debug-300`, the later
run overriding). For the target student: up to 3 measured training tasks per level (too easy, in band, too hard,
unfair, by the keep rule; World and spec defects are not difficulty and are left out), each with pass k/n, mean Φ,
per-milestone pass rates, budget stops, frontier passes, the verdict and the main failure modes, at the tool limit it
was measured under; this window's own numbers when it is a measured training task; the knobs K1-K6 and the "never"
list. **Held-out tasks never enter a brief**: a record is used only when its `split`, `lanes.windows.split_of` of its
repository and the eval lease all say training; the rendered text is checked against every held-out id
(`HeldOutLeak`); a held-out window's brief gets no calibration block (it is eased only for defects).

**Reproductions, gate S7r** (`generators/debug_repro.py`; easing knob K4). A prompt may add
`"repro": {"m01": {"command": "...", "expect": {"exit": 0, "contains": ["..."], "equals": "..."}, "timeout_s": 120}}`.
The command runs through the World's shell in the repository root and may write scratch files under `/tmp/` only (a
heredoc); `expect` is what a correct build shows. `author` lints each repro with the prompt's leak rules (no changed
function, hidden test name, test name of any language, version, commit or link; paths only under `/tmp/`), plus no
hidden test file's path and no line of 32+ characters copied from the lines the fixes added to the hidden tests, and
renders a "How to reproduce" block after the requirements (the agent sees it; a prompt without `repro` is
byte-identical to before). `lh_debug.py repro [--ids]` runs each repro in a World at A' (it must not pass: it shows the
bug) and in a World at B (A' with the oracle's source files written; it must pass). Per requirement: `verified`,
`no_bug_at_start`, `fails_at_oracle`, `unmeasured` (the World or its shell failed), `unverified` (no repro). The window
is `failed` if a present repro failed, else `unmeasured`, `verified`, `partial` or `unverified`; only `failed` is a
failure. Results: `data/debug/repro/<id>.json` (with the prompt's sha256), the funnel row `S7r`, and `repro_status` in
`authored/<id>.json` (kept by a later `author` while the prompt is unchanged).

### 6.3 Fairness, live: S9b, the second verifier in every language, the install rule (2026-10-23, $0.135)

**The install false fail.** `path:install` read `cd /tmp/r1 && go mod tidy` (a reproduction's own module, `replace` =>
the workspace) as an install, so DeepSeek lost the full pass on mvdan-sh at Φ = 1. The rule now excuses an install in a
scratch directory outside the repo (a literal `/tmp`, `/var/tmp`, `/root`, `/home/<u>` path: a variable, `..` or a
subshell's `cd` never counts) for package-manager subcommands whose effect stays in that directory, when the diff
leaves every manifest and lockfile unchanged and the World was offline. `go install`, `cargo install`, `npm -g`, `pip`
anywhere, and any install in the repo still fail; a bare reinstall of declared dependencies is its own class, off by
default. `go -C dir mod tidy` was not even parsed as `go mod` before (the flag's value was read as the subcommand).

**Re-path of every stored episode** (`scripts/lh_repath.py`, no model call, no test run: the stored trail and diff
under the window's own rules, `runs/repath_install.json`). 321 attempts of the 150- and 300-call targeting runs: 17
flips, 12 rewards up, none down, 11 full passes gained. 8 of the full passes come from the scratch rule (7 DeepSeek
fairness episodes on mvdan-sh ×4 windows and carbon, 1 student episode); the other flips are verdicts stored under
older rules (all 321 lacked a rules version: every path report now carries `RULES_VERSION`). With bare reinstalls also
excused (`--allow-declared-reinstall`, `runs/repath_install_reinstall.json`): 36 flips and 20 full passes, the extra 9
on h3 and unstorage (`pnpm install` in the repo, the unjs owner): a policy decision, left off.

**S9b smoke** (`lh_debug_target.py fairness`, training task `db-go-echo-61eca05b`, cap $10, spent $0.135).
- The second verifier (DeepSeek-V4.1-Flash, `second_verifier`) wrote a Go suite blind (payload proven by `leaks`):
  2 files, 9 tests, all sound on the oracle, both requirements gating (fail at the start and on their mutants); $0.025.
- One alternative session from the bug report alone (GLM-5.3-Flash, 150 calls, $0.110): end-state full pass, Φ = 1.
- 13 candidates after dedup (the stored Qwen3.5-397B, DeepSeek and student episodes plus the GLM one): 11 accepted
  (10 confirmed by the independent suite; DeepSeek's own states are judged by verifier 1 alone), 1 wrong (a student
  state verifier 2 also fails on R2), 1 unknown (a DeepSeek state verifier 1 fails and nothing independent measured),
  0 false fails, 0 lucky passes. κ = 0.889 over 30 (state, requirement) pairs; the one disagreement is the `revert_m01`
  mutant on R2 (verifier 1 fails the dependent milestone, verifier 2 passes it: truth unknown, for the tie-break).
  The gate passes: 10 independently confirmed alternatives from glm and qwen.
- **A remaining strictness, found by the smoke.** The GLM session still lost its full pass on `path:install`: it built
  its reproduction module inside the repo (`cd _chk && go mod tidy`). The rule reads a relative `cd` as staying in the
  repo, so this stays a failure until a nested module's manifests can be told from the project's on the diff.
