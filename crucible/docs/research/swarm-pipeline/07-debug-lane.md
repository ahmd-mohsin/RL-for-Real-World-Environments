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
