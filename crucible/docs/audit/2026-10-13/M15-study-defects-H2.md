# M15 defect hunt H2: build pipeline, exports, explorer, CLI and the public site

Scope: `src/crucible/studio/{build,export,explorer}.py`, `src/crucible/cli/{main,studio}.py`, `site/` (read only), the M15 tests.
HEAD 076ab65. No GPU, no paid calls, `publish.sh` not run. Reproducers are in
`$S = /private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m15/h2/`.
Run them from `crucible/` with `PYTHONPATH=src:packs/_toy/src` (plus the other packs where noted) and `.venv/bin/python`.

Baseline: `crucible build toy` runs cleanly (35 s): 24 traces, 5 forged, 0 accepted / 5 provisional (A10 pending), 5
canonical exports at 1.0, a 5 + 2 bench (oracle 1.0, no-op 0). A second run into the same `--out` is idempotent (same
digest and seal, 21 results, resumed). The bench key is checked (at least 32 bytes) and created 0600 with O_EXCL. The
explorer does not follow symlinks out of its root. The bench report escapes `<img onerror>` in policy ids. `--dry-run`
writes nothing (both `label` and `build`).

## Findings

| id | sev | where | defect |
|---|---|---|---|
| V1 | P0 | studio/build.py:226 | When the oracle replay crashes, the anchor is still marked valid and benched. Build exits 0 |
| V2 | P0 | studio/export.py:357-360 | `_bundle` runs rmtree on any existing `out/pack`, including the source pack itself. It then exits 0 with an empty bundle |
| V3 | P0 | studio/export.py:361-366 | The export bundle copies `src/` and `data/` as they are: `.env` files and raw unredacted ingest logs ship |
| V4 | P1 | studio/build.py:173-261 | Build exits 0 when M9 rejects every task, when forge makes nothing, or when ingest finds nothing. A fresh-only bench is reported as the build |
| V5 | P1 | studio/build.py:155-165 | Build skips the `requires` and `higher_is_better` refusals that `bench run` enforces (M14 B6 and V12) |
| V6 | P1 | studio/build.py:226 | Anchor oracle validity is hard-coded to `>= 1.0`, not the pack's `reward.range` top (M14 V12 again) |
| V7 | P1 | studio/build.py:179, 235 | A failed rebuild leaves a contradictory out dir: stale `summary.json`, a new `accept.json`, an overwritten sealed `splits.json`, and exports of now-rejected tasks |
| V8 | P1 | studio/build.py:230-240 | Splits are sealed and then ignored: the bench scores train-split items, and the seal is not bound into the run. The contamination step that M13 L12 says runs in build is absent |
| V9 | P1 | studio/export.py:381, build.py:190 | A task id becomes a path with only `/` and `@` replaced: `..` escapes the export dir, and `a/b` / `a_b` collide silently |
| V10 | P1 | studio/export.py:304, 320, 331 + launcher | Export never awaits an async `hooks.tasks`, so the coding pack cannot be exported. A partial bundle is left behind |
| V11 | P1 | studio/export.py:337 | `openenv.yaml` sets `app: openenv_env:Environment` (a class). OpenEnv runs `uvicorn <app>`, so the export is not servable. The test only checks that the file exists |
| V12 | P1 | studio/export.py:390-410 | `export --to atif` on the wrong dir writes 0 trajectories and exits 0. With no validator, it copies unvalidated files and still exits 0 |
| V13 | P1 | studio/explorer.py:439, 502-511 | Pack names `..` and `.` pass `SAFE`. `find_pack` treats them as paths from the server CWD, so a GET imports and runs a pack outside the index |
| V14 | P2 | studio/explorer.py | Duplicate run ids shadow each other. A 404 leaks the absolute data root. A GET writes `report.html`. Every request runs rglob over the whole root |
| V15 | P2 | bench/run.py:161-174 | Key-file permissions are never checked: an existing 0644 `bench.key` or a 0666 `--key-file` is accepted silently |
| V16 | P2 | cli/main.py:37, cli/studio.py:14 | `--json` prints nothing to stdout on error. A user error exits 1, the same code as "a check failed". An internal `KeyError` or `ValueError` becomes a one-line user error with no traceback |
| V17 | P2 | cli/studio.py | `configs/resolved/<cmd>.json` is written only by `label` and `build`. The doc says every command writes one |
| V18 | P2 | cli/main.py:159, 189 | `world validate --forbidden` takes hidden answer strings on argv and echoes them in check details |
| V19 | P2 | cli/studio.py:238-248 | `crucible train` always trains `TinyPolicy(32, 48)`, whatever the spec says. Its help says "run a training job spec" |
| V20 | P2 | studio/export.py:375-387 | `_canonical` aborts the whole export when one task's replay fails, and skips cleanup. Build records per-task errors instead |
| V21 | P2 | docs/modules/M15-studio.md:86, README.md:52 | Doc drift. The gate says "5 accepted", but the code gives 5 provisional. The README says build runs "M9 accept (model-free checks)", but it runs the full contract |
| V22 | P0 | site/page.html | Public-material rule: indirect references to other frameworks, methods and papers |
| V23 | P0 | site/page.html, assets/scenes3.js | STE100 copy rule: colons, passive voice, sentences over 25 words, academic register and headings, first person |
| V24 | P1 | site/page.html:399 and others, assets/scenes*.js | M-codes in public material: 15 `id="mN"` anchors that show in the address bar, and M-code comments in the shipped JS |
| V25 | P2 | site | The site gives three different attack counts. `<title>` and `<meta description>` sit in `<body>`. `publish.sh` has no rules gate before it pushes |

---

### V1 (P0): when the oracle replay crashes, the anchor is still valid and benched (silent success)
`build.py:225-227`
```python
orc = exported.get(i.id, {}).get("reward")
bad = [] if orc is None or orc >= 1.0 else ["broken_oracle"]
```
A task whose M11 export raised has no `exported` entry, so `orc is None` and the anchor is `valid: True`. Build still
exits 0. The human output prints only "exported 0" and never prints `export.errors`.

**Reproducer:** `$S/r_build_partial.py export_fail` (patches `crucible.interop.replay.replay` to raise).
```
exit 0
forge 5 accept {'accepted': 0, 'provisional': 5, 'rejected': 0}
export 0 errors ['RuntimeError: replay crashed']
bench items {'anchor': 5, 'fresh': 2} valid_tasks 7 invalid []
validity: 'toy/helpdesk_v2/t1_duplicate': {'oracle': {'replayed_reward': None}, 'reasons': [], 'source': 'M9 accept (provisional)', 'valid': True}
```
**Fix:** when `orc is None`, mark `valid: False` with the reason `oracle_not_replayed`, or keep a task with an export
error out of `anchors`. Make build exit non-zero when `export_errors` is non-empty, and print the errors in the human
output.

**Test:** monkeypatch `replay` to raise, then run `build toy --json`. Assert a non-zero exit, or every anchor in
`validity.json` having `valid is False`.

### V2 (P0): the export deletes whatever sits at `out/pack`, including the source pack
`export.py:357-360`: `dest = out / "pack"; if dest.exists(): shutil.rmtree(dest)`. Export never checks that `out/pack`
is its own earlier output, or that it is not the source pack. When the pack directory is named `pack` and `--out` is
its parent, the source is deleted. The copy then finds nothing, and the command exits 0 with an empty bundle.

**Reproducer:**
```
cp -R packs/_toy $S/x/pack                                 # 100 files
crucible export --to nemo-gym --pack $S/x/pack --out $S/x  # rc=0, "bundle": "$S/x/pack"
find $S/x/pack -type f | wc -l                             # 0: the source pack is gone
```
Any user directory called `pack` under the chosen `--out` is lost in the same way.

**Fix:**
- Refuse when `dest` resolves to `pack_dir`, or to a directory inside it.
- Only replace `out/pack` when it holds a marker that export wrote, such as `crucible_export.json`. Otherwise, refuse.
- Build into a temporary directory and swap it in with `os.replace`.
- Fail if the bundle has no `pack.yaml` after the copy.

**Test:** copy toy to `tmp/x/pack`, export with `--out tmp/x`. Assert a non-zero exit and that the source `pack.yaml`
still exists.

### V3 (P0): the export bundle ships `.env` files and raw unredacted logs
`_bundle` copies `src`, `data`, `worlds` and `families` with `copytree`, ignoring only `__pycache__`. M3 redacts what
it ingests, but the bundle copies the raw files. Dotfiles and any `redaction_env` file kept inside the pack are copied
too. The docstring says the bundle is meant to "move to another machine".

**Reproducer:** `$S/r_bundle_secrets.sh`. It adds a fake `src/.env` (`OPENAI_API_KEY=sk-FAKE-CANARY-0000`) and
`data/ticket_logs/raw_dump.jsonl` (a fake email and card number) to a toy copy.
```
rc=0
$S/sec/out/pack/data/ticket_logs/raw_dump.jsonl
$S/sec/out/pack/src/.env
```
**Fix:**
- Copy an allowlist: the pack's package, `worlds/`, and the data the tasks read. Exclude the ingest globs (raw logs),
  dotfiles, `*.env*`, `*.key` and every `redaction_env` path.
- Scan the bundle with M3's `RedactionPolicy` and refuse when it hits.
- Better still, build the wheel and ship that (`pack check --publish` already checks wheels).

**Test:** a pack with `src/.env` and a raw log under an ingest glob. Assert neither file is in the bundle, or that the
export refuses.

### V4 (P1): build exits 0 when M9 rejects every task, forge makes nothing, or ingest finds nothing
`$S/r_build_partial.py all_rejected | zero_forge`:
```
== all_rejected   exit 0  accept {'accepted': 0, 'provisional': 0, 'rejected': 5}  bench items {'anchor': 0, 'fresh': 2}
== zero_forge     exit 0  forge 0                                                   bench items {'anchor': 0, 'fresh': 2}
```
In both cases the build's benchmark is made only of generator items. `crucible accept` exits 1 on any rejection, but
build has no exit policy at all.

**Fix:** exit 1 (or refuse to bench) when `forge.tasks == 0`, when `accepted + provisional == 0`, or when `traces == 0`.
Exit with a distinct code when any task is rejected, matching `crucible accept`. Add `--allow-rejected` to opt out.

**Test:** patch `verdict` to reject. Assert `exit_code != 0`.

### V5 (P1): build skips the `requires` and `higher_is_better` gates
`bench run` refuses a pack whose `requires` this machine lacks (M14 B6). It also refuses `higher_is_better: false`
(M14 V12). `build` does neither, and it has no `--allow` option.

**Reproducer:** copy toy with `requires: [paid, network]` and `higher_is_better: false`.
```
crucible bench run $S/toyreq  -> Error: ValueError: pack toy requires paid, network, which this machine does not provide ...
crucible build $S/toyreq      -> rc 0, bench {'anchor': 5, 'fresh': 2}: scripted:oracle 1.0 ...  (lower-is-better reported upside down)
```
**Fix:** move both refusals into a shared `bench_preflight(m, allow)` and call it from `build`. Add `--allow` to `build`.
Also pass `extras={"reward_range": ...}` as `bench run` does.

**Test:** the toy copy above. `build` must exit non-zero with the same message that `bench run` gives.

### V6 (P1): anchor oracle validity is hard-coded to 1.0
`build.py:226 orc >= 1.0`. On a pack with `reward.range: [0, 10]`, an oracle that replays to 1.0 counts as valid.

**Reproducer:** copy toy with range `[0.0, 10.0]` and run `build --fresh 0`.
```
oracle_rewards all 1.0; validity {'valid_tasks': 5}; invalid_tasks []
```
**Fix:** use `m.reward.range[1]` and orientation (the same rule as `gold_and_noop_check(best=...)`) and reject non-finite
rewards.

**Test:** the `[0, 10]` copy. Assert the anchors are invalid with the reason `broken_oracle`.

### V7 (P1): a failed rebuild leaves a contradictory out dir; stale exports of rejected tasks remain
Build writes `accept.json` and `splits.json` before `run_bench`. `run_bench` then correctly refuses to resume a run
with a different identity. By that point the sealed `splits.json` of the existing run is overwritten. `summary.json`
still describes the old build, and `export/` still holds the dirs of tasks M9 now rejects.

**Reproducer:** `$S/r_build_stale.py $S/b1 $S/b1stale` (rebuild a copy of a good build with every task rejected).
```
exit 1 Error: ValueError: run bench-toy-build exists with a different identity (...): refusing to resume a different run
b1stale summary.accept {'accepted': 0, 'provisional': 5, 'rejected': 0}   accept.json states ['rejected']
        splits seal acf50b1dadf1   summary seal 0a25a4df1d28
        export/: toy_helpdesk_v2_t1_duplicate ... t5_compose   (all 5 rejected tasks still exported)
```
Anything that reads `summary.json`, such as the explorer or CI, still sees a successful build.

**Fix:**
- Check the run identity first, before writing anything.
- Write `accept.json`, `splits.json`, `export/` and `summary.json` to a staging dir, then rename it into place.
- Before exporting, remove export dirs that are not in this build's accepted set.
- Never overwrite an existing `splits.json` whose seal differs. Refuse instead.

**Test:** run a good build, then patch the rebuild to reject. Assert that `splits.json` is unchanged, `export/` is
empty of the rejected tasks, and `summary.json` is either absent or agrees with `accept.json`.

### V8 (P1): splits are sealed but ignored; contamination is missing from build
- `splits.assign` runs (6 train, 1 bench in toy). Then `run_bench(tasks=[i.task for i in bm.items])` scores all 7, and
  the report and headline include the train-split items.
- The seal is not passed to `run_bench` (`extras` holds only the dynamic manifest), so the run cannot show which split
  it was sealed under.
- `M13-bench.md` L12 says "splits and contamination run in `studio/build` before any run". `grep contamin build.py`
  finds nothing.

**Fix:**
- Pass `extras={"splits": {"seal":..., "counts":...}}`.
- Either bench only the `bench` split or report per split, with the headline taken from the bench split.
- Call the contamination check (the canary registry plus `curator_hook`) on the forged tasks before `run_bench`, and
  record the result in `summary.json`.

**Test:** assert `summary["bench"]["items"]` equals `splits.counts["bench"]`, or that the report carries a per-split
headline. Assert that `run.json` / the report extras hold the seal. Assert that `summary.json` has a `contamination` key.

### V9 (P1): task ids become paths with escapes and collisions
`export.py:381` and `build.py:190`: `tid.replace("/", "_").replace("@", "_")`. A task id of `..` maps to
`out/export/..`, and `replay` writes `reward.json`, `steps.jsonl`, `report.json` and more into the parent of the
export dir. `acme/t1` and `acme_t1` map to the same dir: the second overwrites the first, and the summary lists both.
Task ids come from pack code and from forged workflow data.

**Reproducer:** `$S/r_export_traversal.py`
```
rows {'..': '.../h2-exp-*/export/..'}
written outside out: ['crucible_reward.json', 'export', 'report.json', 'reward.json', 'reward.txt', 'steps.jsonl']
collision {'acme/t1': '.../export2/acme_t1', 'acme_t1': '.../export2/acme_t1'} dirs on disk ['acme_t1']
```
**Fix:** one helper, `safe_dir_name(tid)`, used in both places: a slug restricted to `[A-Za-z0-9._-]`, never `.` or
`..`, plus a short hash of the full id. Assert that `d.resolve()` is under `out.resolve()`, and refuse duplicates.

**Test:** both cases above. Assert no file outside `out` and two distinct dirs.

### V10 (P1): export does not await async `hooks.tasks`
The M14 campaign (B6) fixed `bench run` (`_awaited`), but export calls `resolve(d, ref)()` and iterates the result. The
generated launchers do the same (`resolve(PACK, ref)()[int(task_index)]`).

**Reproducer:** `$S/r_export_async.sh`
```
canonical rc=1 sys:1: RuntimeWarning: coroutine 'tasks' was never awaited   (TypeError: 'coroutine' object is not iterable)
nemo-gym rc=1 TypeError: 'coroutine' object is not iterable
openenv rc=1 TypeError: 'coroutine' object is not iterable   ($S/coding-openenv/pack left behind)
```
**Fix:** await the hook in `export()`, and have the generated factory run `asyncio.run` on an awaitable (or materialise
the task list at import). Build the bundle only after the tasks have loaded.

**Test:** `export --to canonical|nemo-gym|openenv --pack coding` (CPU stand-ins) exits 0, and the factory's `tasks()`
returns two ids.

### V11 (P1): the OpenEnv manifest's `app` is not an ASGI app
Export writes `app: openenv_env:Environment`, which is the class returned by `make_environment`. Upstream manifests use
`app: <pkg>.server.app:app` (see `meta-pytorch-openenv/envs/chess_env/openenv.yaml`). The OpenEnv providers launch
`python -m uvicorn {app}` (`core/containers/runtime/modal_provider.py:286` and `novita_provider.py:838`), and uvicorn
cannot serve an `Environment` class. The manifest also leaves out `port`. `test_g2_d11_...` asserts only that
`openenv.yaml` exists, so it is vacuous for this contract.

**Fix:** the generated module should end with `app = create_app(Environment, CallToolAction, CallToolObservation, ...)`
(OpenEnv's server factory), and the manifest should say `app: openenv_env:app` and `port: 8000`.

**Test:** inside `.envs/openenv`, load `openenv.yaml`, import the `app` target, assert it is an ASGI callable, and run
`TestClient(app).post("/reset")`.

### V12 (P1): `export --to atif` reports success on nothing
```
export --to atif --run $S/b1 (the build root, not the run)   -> rc=0 {"trajectories": 0, "invalid": {}}
```
When `harbor` is missing, every file is copied with `"validated": false` and the exit is 0. Invalid trajectories are
listed but also exit 0.

**Fix:** exit 1 when no trajectory files are found ("no samples/*/a*/trajectory.*.json under DIR"), when any is
invalid, or when there is no validator (unless `--no-validate` is passed).

**Test:** atif export on an empty dir must exit non-zero.

### V13 (P1): the explorer resolves pack names `..` and `.` outside its scope and runs their code
`SAFE = ^[A-Za-z0-9._-]+$` accepts `..` and `.`. `find_pack` treats both as paths (`s in (".", "..")`) relative to the
server's CWD. The static `check` imports the entry point and hooks, so an unauthenticated GET imports and runs the code
of whatever pack sits at `CWD/..`.

**Reproducer:** `$S/r_explorer.py` (CWD = `<toy copy>/src`):
```
pack '..' -> 200 ('toy', True)
```
**Fix:** in the route, require `name in pack_index()` (or `^[A-Za-z0-9_][A-Za-z0-9._-]*$` without a leading dot), and
call `find_pack` only with a name. Apply the same rule to `run_id` (`..` passes the regex today; it only fails to match
by luck).

**Test:** `GET /packs/%2E%2E/check` and `/packs/./check` return 400.

### V14 (P2): explorer smaller defects
From `$S/r_explorer.py`:
- `runs listed: [('dup', 'a/dup'), ('dup', 'b/dup'), ...]`. `/runs/dup/summary` always resolves to `a/dup`, so the
  other run cannot be reached. Address runs by their relative path, or answer 409 when an id is ambiguous.
- The 404 body is `{"detail":"no run 'nope' under /…/root"}`, which leaks the absolute data root to the client. Drop
  the root from the message.
- `GET /runs/{id}/report` writes `bench/report.html` into the data dir. A GET should not have side effects: render in
  memory instead (`report written on GET: True`).
- `/runs` and `/datasets` run `rglob` over the whole root and read every `results.jsonl` in full on each request. A
  large root makes every request slow. Cache, or cap the depth.

### V15 (P2): key-file permissions are not enforced
`$S/r_key_perms.py`
```
fresh key mode 0o600
0644 existing key accepted: True
0666 --key-file accepted: True
```
**Fix:** in `load_key(file=...)`, refuse (or loudly warn about) a group- or world-readable file, and refuse a symlink
(`os.lstat`), like ssh does. The key also sits in `out/bench.key` beside `export/`, `runs/` and `summary.json`, which
are the dirs a user shares. Keep it outside `out`, for example under `~/.config/crucible/keys/<bench>`.

**Test:** chmod 0644 on an existing `bench.key`, then `bench_key(out)` raises.

### V16 (P2): CLI error contract
- With `--json`, every user error prints `Error: ...` to stderr and nothing to stdout, so a JSON consumer cannot
  parse the failure. Tested on `build`, `export`, `label`, `bench run`, `accept` and `pack check` with an unknown pack:
  all `rc=1 stdout=0B`. Fix: with `--json`, print `{"error": {"type":..., "message":...}}` to stdout.
- `pack check` documents exit 1 as "a check failed", but an unknown pack also exits 1. Use a distinct usage code
  (click's 2 is already taken by "skipped"; use 4).
- `USER_ERRORS` includes `ValueError` and `LookupError`, so an internal `KeyError` or `JSONDecodeError` from crucible's
  own code is shown as a one-line "user error" with no traceback. Raise a `UserError` type at the input checks, and let
  everything else show a traceback.

### V17 (P2): `configs/resolved/<cmd>.json` exists only for `label` and `build`
The M15 doc says "every command ... writes a run directory (`configs/resolved/<cmd>.json`, outputs, `summary.json`)".
`forge`, `accept`, `export`, `model`, `bench run` and `ingest` write none. Either write it (with secrets redacted, as
build does with `key_commitment`) or change the doc.

### V18 (P2): `world validate --forbidden` puts hidden answers on argv
The answer strings that "must never reach the policy" are passed on the command line, so they show up in `ps` and in
shell history. They are then echoed in the check detail (`json.dumps(c.detail)[:160]` holds `forbidden_seen`).
Fix: add `--forbidden-file`, and print counts rather than the strings.

### V19 (P2): `crucible train` always trains a toy policy
`run_job` hard-codes `TorchTrainer(TinyPolicy(32, 48, ...))`. The spec's model fields are ignored. Fix: say "toy
launcher" in the help and README, or wire the spec's trainer and model.

### V20 (P2): canonical export is all-or-nothing and leaks Worlds on failure
In `_canonical`, the first replay exception propagates and the cleanup loop never runs. Fix: catch errors per task
(as build does), run cleanup in a `finally`, and exit non-zero when there are errors.

### V21 (P2): doc drift
- `M15-studio.md:86` says "5 tasks forged, 5 accepted". The run gives 0 accepted and 5 provisional (A10 pending), and
  the test asserts `accepted + provisional == 5`.
- `README.md:52` says build runs "M9 accept (model-free checks)". It runs the full contract, with provisional and
  rejected states.

---

## Public site (crucible/site)

The site audit was read only. It extracted 1,474 visible strings: page text, aria-labels, the meta description, and
the JS strings and labels drawn in the 3D figures. All of them, with file:line, are in `$S/strings.tsv`, and every
regex hit is in `$S/checks.txt`. The full list is `$S/site-findings.md`, made by `$S/site_extract.py`.

- **index.html is not stale.** It equals build.py's output from page.html byte for byte.
- **Explicit names:** none. No third-party framework, benchmark or model is named.
- **Banned punctuation and codes:** no dashes, no semicolons, no M-codes in the visible copy.

### V22 (P0): public-material rule, indirect references
Each item below points to another framework, method or paper:
- page.html:416 "Any tool server that uses the standard tool protocol". This is MCP, described without its name.
- page.html:851 "The recursive agent also runs a loop that repairs and improves tasks." With page.html:828 "The model
  accesses them by writing code and delegates questions to sub-models.", this is the Recursive Language Model design in
  that paper's own terms. The anchor is `href="#rlm"` / `id="rlm"` (page.html:495, 818), so "rlm" shows in the address
  bar.
- page.html:444 "It supports group-relative advantages, a critic over segments and on-policy distillation". This spells
  out GRPO and named techniques. It is borderline: algorithm names.
- page.html:350 "...evaluate it on a benchmark and sell the environment or its data to a laboratory." This alludes to
  AI labs and data vendors.
- Borderline algorithm acronyms: figure labels `'SFT', 'DPO', 'KTO'` (assets/scenes2.js:556), and the terminal line
  `sft 3  dpo 0 (best_vs_worst 0, correction_pivot 0)  kto 62` (assets/main.js:55).

**Fix:** rewrite each one in plain enterprise terms that do not point to the source, for example:
- "Any tool server the agent can call";
- "The memory agent also repairs and improves tasks";
- "It supports several ways to score a group of attempts".

Rename the `#rlm` anchor.

### V23 (P0): STE100 copy rule
- **Colons:**
  - page.html:339 aria-label "A crucible: work traces flow in, environments, tasks and benchmarks flow out"
  - page.html:412 "A Crucible world: apps, people and a clock inside a sandbox, with one gate for the agent"
  - page.html:435 "The Crucible RL infrastructure as four layers: worlds, models, data and training"
  - page.html:460 "Building a long task: one real case grows into a shift with waits, rule changes, links and milestones"
  - assets/scenes3.js:143 figure label `records: exact tokens, versions, step scores`
- **Sentences over 25 words:**
  - page.html:732 (28 words) "The contract establishes that the reference solution passes, that trivial attempts fail, that
    the checks detect injected defects, that no answer leaks and that the verifier resists attacks."
  - page.html:362 (26 words) "It reads agent traces and expert feedback from existing workflows and constructs RL
    environments from them, with rubrics that provide a strong signal at each step."
- **Passive voice:**
  - page.html:973 "No model, GPU or API key is required."
  - page.html:378 "An action that is undone and then repeated yields zero net reward."
  - page.html:419 "The network is closed, and a self-test verifies this before the first step"
  - page.html:1039 "Builds are pinned and reproducible"
- **Academic register, not enterprise framing:**
  - Headings: page.html:334 and 348 "Motivation", page.html:997 "Empirical results".
  - page.html:363 "The system thus constitutes a form of recursive self-improvement, in which one model improves
    another. We will use Crucible to construct further enterprise environments." This also uses the first person.
  - Non-plain words, about 55 distinct, for example: "instantiates" (536), "Heterogeneous sources map to a single trace
    schema" (579), "Shaping terms telescope." (378), "Idempotence." (592), "predictive validity for subsequent
    outcomes" (609), "They disregard exit codes and self-reported claims." (683).
  - The register came in with commit b9d358c ("site copy in a formal research register"), which runs against the rule.
- **Fragments:** page.html:444 and 787 "A shared GPU, partially shared GPUs, or disaggregated machines."
- **Mixed UK/US spelling:** "coloured" (main.js:27), "Centre" (main.js:25), "labelled" (main.js:53), "judgement"
  (page.html:602).

### V24 (P1): M-codes in public material
- page.html carries 15 `id="mN"` anchors. Links such as `href="#m3"` … `#m7` (page.html:399) put `#m9` in the reader's
  address bar.
- The shipped JS has M-code comments: `// Scenes, part 1: hero, pipeline, M0–M7.` (scenes1.js:1), `// Scenes, part 2:
  M8, M9, M10, M12, ... RLM ...` (scenes2.js:1), and the `// ===... M0` section banners (scenes1.js:269 and others).
  The second comment also names RLM.

**Fix:** use descriptive anchors (`#ingest`, `#acceptance`), and strip comments from the published JS (minify in
`publish.sh`).

### V25 (P2): other site points
- **Three attack counts:** "more than 20 attacks" (page.html:419), "All 11 known attacks score zero." (scenes1.js:741)
  and `'32 attack attempts'` (scenes2.js:93).
- **Head elements in the body:** build.py's `HEAD` has no `<title>`, so page.html's `<title>`, `<meta
  name=description>` and `<link>` land inside `<body>`. Crawlers and link previews may ignore the description there.
  Fix: build.py should lift those lines into `<head>`.
- **publish.sh:** it handles no secrets itself (it pushes with the user's git credentials and holds no tokens). But it
  pushes with no rules gate: no check for third-party names, M-codes or punctuation before `git push`. Its
  `git commit` also commits anything already staged in the site repo, not only `public/crucible`. Fix: run
  `site_extract.py`-style checks and fail on any hit, and use `git commit -- public/crucible`.
- **build.py:** no injection risk. It joins static strings only, with no external input.

## Reproducers (all in `$S`)
- `r_build_partial.py {export_fail|all_rejected|zero_forge}`: V1, V4
- `toyreq/` and `toy10/` (pack copies): V5, V6
- `r_build_stale.py`: V7
- `r_export_traversal.py`: V9
- `r_export_async.sh`: V10
- `r_cli.sh`: V12, V16
- `r_explorer.py`: V13, V14
- `r_key_perms.py`: V15
- `r_bundle_secrets.sh`: V3
- `x/` (shows the deleted pack): V2
- `site_extract.py`, `site-findings.md`, `strings.tsv`, `checks.txt`: V22 to V25
