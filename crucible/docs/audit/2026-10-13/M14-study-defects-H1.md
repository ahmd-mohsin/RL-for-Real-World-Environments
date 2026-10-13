# M14 packs: defect hunt H1 (manifest, pack check, loading, hooks, ingest, template)

Scope: `src/crucible/core/schema/pack.py` and `src/crucible/studio/pack.py` at HEAD f2d9e18, plus everything that reads a
pack: `cli/studio.py` (`pack new|check`, `label`, `build`, `bench run`, `model`, forge), `studio/{label,build,export,explorer}.py`,
`core/registry.py` (entry points), `connectors/redact.py` (`redaction_env`), `bench/run.py` (fresh-item validation), and
`packs/_template`. Tests: `tests/unit/test_packs.py`, `tests/unit/test_studio.py`, `tests/unit/test_m4_audit_packs.py` (the
last one covers M4 signals of the packs, not `pack check`).

Reproducers live in the session scratchpad, `scratchpad/m14/h1/`. Each runs with `crucible/.venv/bin/python` (or the
`crucible` console script of that venv); nothing needs a GPU, Docker or a paid call.
- `mutants.py`: scaffolds one pack per mutant with `crucible pack new mx_<id>` (a distinct package each, so no module
  cache is shared), mutates `studio.py` / `pack.yaml` / `pyproject.toml` / `__init__.py`, runs
  `crucible pack check <dir> --level semantic --json` in a fresh process, and writes `mutants.json` / `mutants2.json`
  (`python mutants.py` and `python mutants.py v2`). Mutant graders wrap the template grader and return
  `dataclasses.replace(report, ...)`, i.e. a real `GradeReport`.
- `stale.py`: same-process module cache (V7). `nullcheck_plugin.py`: the vacuous-test probe (V16).
- `gates_audit.py`: grades every shipped fixture and checks that each `fails_gate` names a gate the grader emits.
- Shell reproducers for ingest escape, redaction and name collisions are quoted inline below (dirs `ing/`, `shadow/`,
  `ten/`, `cwdtest/`, `rawtpl/`).

## Mutant matrix (`pack check --level semantic`, fresh process each)

| mutant | what is broken | caught by | exit |
|---|---|---|---|
| always1 | grader returns 1.0 for everything, template fixtures | fixture.cases + semantic.noop | 1 |
| always1_gold_only | same, fixtures reduced to the gold case | semantic.noop | 1 |
| always0 | grader returns 0.0 | fixture.cases + semantic.gold | 1 |
| lower_is_better | honest cost-style pack, `higher_is_better: false` | semantic (wrongly, V12) | 1 |
| nondet_grader | grader flips its verdict 15% of the time | fixture.cases in 8/11 runs; **passed 3/11** | 0 or 1 |
| raising | grader raises | nothing: traceback, no report (V14) | 1 |
| **always1_nohooks** | always 1.0, no `fixtures`/`tasks` hooks | survives (two warns) | **0** |
| **always1_builtins_list** | always 1.0, `tasks: builtins:list`, `fixtures: builtins:list` | survives (all pass) | **0** |
| **always1_fixtures_empty** | always 1.0, hooks return `[]` | survives (all pass) | **0** |
| **always1_gold_only_req_net** | always 1.0 + `requires: [network]` | survives (semantic skipped) | **0** |
| **always1_req_gpu_no_fixtures** | always 1.0, no fixtures, `requires: [gpu]` | survives (warn + skip) | **0** |
| **nan_reward** | grader returns NaN for everything, template fixtures (1.0 / 0.0) | survives (all pass) | **0** |
| **out_of_range** | gold 1.5, everything else -3.0 on a declared [0, 1] | survives | **0** |
| **noop_near_top** | untouched task scores 0.999 on [0, 1] | survives | **0** |
| **gate_typo_always_pass_gate** | grader's gate is `result`; fixtures assert `fails_gate: outcome` | survives | **0** |
| **nondet_tasks** | `determinism: seeded`, task word drawn from `SystemRandom` | survives | **0** |
| **foreign_hooks** | own grader always 1.0; `tasks`/`fixtures` point at `crucible_pack_toy` | survives (checks the toy pack) | **0** |
| **entry_point_broken** | pyproject entry point names a missing module; `register()` raises | survives | **0** |
| **todo_in_src** | `TODO` in `studio.py`, "(edit this description)" in README | survives | **0** |
| **license_tbd / license_none** | `license: TBD` / `none` | survives | **0** |
| **threshold_out_of_range** | `label.threshold: 7.0` on range [0, 1] | survives | **0** |
| **rawtpl** | unsubstituted `_template` copy (`name: __PACK_NAME__`), `template_version` line dropped | survives ("placeholders: none") | **0** |
| **base_mut in-process** | always-1.0 copy of `base` checked after `base` in one process | survives (V7) | n/a |

Every bold row is a broken pack that `crucible pack check --level semantic` passes with exit 0.

## Summary

| id | sev | file:line | defect |
|---|---|---|---|
| V1 | P0 | studio/pack.py:45, 249, 273 | Missing `hooks.fixtures` / `hooks.tasks` are `warn`. A pack whose grader always returns 1.0 passes every level with exit 0. |
| V2 | P0 | studio/pack.py:252-266, 275-297 | Empty fixture and task lists pass vacuously ("0 case(s) graded exactly", "0 task(s): gold reaches 1.0"); `tasks: builtins:list` is enough. |
| V3 | P0 | studio/pack.py:261, 288, 290 | A NaN reward passes the fixture and semantic levels: every comparison with NaN is false. |
| V4 | P0 | studio/pack.py:261, 288 | Rewards outside the declared `reward.range` are never rejected (gold 1.5, negatives -3.0 on [0, 1] pass). |
| V5 | P0 | connectors/redact.py:125, studio/label.py:72, studio/pack.py:202-214 | A missing or misspelled `redaction_env` is a silent no-op: static check passes and `crucible label` stores the secret in the trace store and `traces.jsonl`. |
| V6 | P1 | studio/pack.py:229-243, 308-315 | `requires` is self-declared and unverified: declaring any unmet need turns semantic into a `skip` with ok=true and exit 0. The skip is filed under `semantic.gold` only. The coding pack has no `tasks` hook, so its semantic level can never run, even with `--allow gpu,docker`. |
| V7 | P1 | studio/pack.py:98-106 | `resolve` reuses `sys.modules`: in one process, a second pack with the same package name (a fork, a mutated copy, a re-scaffold) is checked with the first pack's code. |
| V8 | P1 | studio/pack.py:263 | `fails_gate` is vacuous when the grader has no such gate (`gates.get(...)` is None, not "pass"). |
| V9 | P1 | core/schema/pack.py:86; studio/pack.py:254-297 | `determinism: seeded` is never checked (no consumer reads the field). Each case is graded once; a flaky grader passed 3 of 11 runs; unseeded tasks always pass. |
| V10 | P1 | studio/pack.py:179-181; core/schema/pack.py:76 | The placeholder check only greps `TODO` in three top-level files. The template has no `TODO`, the raw template (`__PACK_NAME__`) passes every level, `src/` is never scanned, and `name` is not validated. |
| V11 | P1 | studio/pack.py:184-200 | Hooks and the entry point are only imported and tested for `callable`. Hooks can point at another pack or at `builtins`; the entry point is never compared with `pyproject.toml` and `register()` is never called. |
| V12 | P1 | core/schema/pack.py:55, 60, 86; bench/run.py:171; cli/studio.py:280-330 | Manifest fields that nothing honours: `higher_is_better` (semantic assumes higher), `determinism`, and `reward.range` / `label.threshold` outside `pack check` (bench uses threshold 1.0 and `gold == 1.0`). |
| V13 | P1 | packs/_template/src/__PACK_PKG__/studio.py:20-66 | `pack new demo` passes check at every level, but its printed next step `crucible bench run` fails: the template's generated items have no oracle. The check never exercises `generator` / `policies` / `stages`. |
| V14 | P1 | studio/pack.py:25, 302-315; cli/studio.py:55 | A crash in a hook (raising grader, raising fixtures hook, absolute ingest glob) is a traceback with exit 1, not a check result; the documented exit 3 does not exist, and `--all` drops every other pack's report. |
| V15 | P1 | studio/pack.py:100-102 | Each checked pack's `src/` is prepended to `sys.path` for the rest of the process; a pack's stray top-level modules shadow third-party modules (a pack's `numpy.py` replaced numpy). |
| V16 | P1 | tests/unit/test_packs.py, tests/unit/test_studio.py | Vacuous: with the fixture and semantic levels stubbed to always pass, all 7 pack/studio tests still pass. No test asserts that `pack check` fails a broken grader. |
| V17 | P1 | core/schema/pack.py:29-36; studio/pack.py:202-214; studio/label.py:29-36, 66-71 | Ingest paths are not confined: `../`, absolute `root: /etc` and symlinked directories pass `static.ingest` and are ingested; an absolute glob crashes the check but works in `label`. |
| V18 | P2 | studio/pack.py:182 | The licence check accepts `TBD`, `none`, `NOASSERTION`, `proprietary`; no SPDX check, no LICENSE file. |
| V19 | P2 | studio/pack.py:78-95, 117; core/registry.py:107-118 | Pack names are not unique: `pack new toy` succeeds; `find_pack("toy")` prefers `./toy` in the CWD, then the first sorted directory; the registry's `pack` kind is registered by every pack and read by nobody (installed packs cannot be found by name); duplicate entry-point names fail into `registry.failures` with a log line only. |
| V20 | P2 | studio/pack.py:167-168, 213; studio/label.py:29-36, 71 | Ingest contract gaps: plain relative `init`/`scope` paths resolve against the CWD; the static existence check is flat (nested `scope.paths` unchecked); `discover` is not checked to exist; fields the connector ignores are accepted. |
| V21 | P2 | studio/pack.py:290 | "No-op stays below the top" is a strict `< hi`: an untouched task scoring 0.999 of [0, 1] passes. |
| V22 | P2 | studio/pack.py:237, 292-294 | `docker info` runs with no timeout; `task.cleanup()` is skipped when an episode raises. |

---

## V1 (P0): missing fixtures / tasks hooks are warnings, so a trivial grader passes

`SEVERITY_V1` (studio/pack.py:45) maps `fixture.hook` and `semantic.tasks` to `warn`, and `PackReport.ok` only looks at
`fail`. A pack with no `fixtures` and no `tasks` hook therefore passes `--level semantic` without a single graded episode.

```
$ python mutants.py always1_nohooks
always1_nohooks {"exit": 0, "ok": true, "flagged": ["fixture.hook", "semantic.tasks"], ...}
$ python mutants.py v2   # always 1.0, no fixtures hook, requires: [gpu]
always1_req_gpu_no_fixtures {"exit": 0, "ok": true, "flagged": ["fixture.hook", "semantic.gold"(skip)], ...}
```

**Fix.** When the caller asks for `--level fixture`, a missing `hooks.fixtures` is `fail`; for `--level semantic`, a
missing `hooks.tasks` is `fail` (or the level reports `skip` and `ok` is false for that level, see V6). A pack that
cannot have live tasks declares that explicitly (`semantic: none` with a reason) instead of omitting the hook.

**Regression test (fails now).** Scaffold `pack new t`, delete `fixtures:` and `tasks:` from `pack.yaml`, monkeypatch the
grader to return 1.0; assert `check(..., level="semantic").ok is False`.

## V2 (P0): empty case lists pass

`check_fixture` reports `pass` with "0 case(s) graded exactly as expected" and `check_semantic` reports "0 task(s): gold
reaches 1.0" (studio/pack.py:266, 296-297). Any callable that returns an empty iterable satisfies both, including
`builtins:list`.

```
always1_builtins_list  {"exit": 0, "ok": true, "flagged": [], ...}   # tasks: builtins:list, fixtures: builtins:list
always1_fixtures_empty {"exit": 0, "ok": true, "flagged": [], ...}   # def fixtures(): return []; def tasks(): return []
```

**Fix.** `fixture.cases` needs at least one case at the top of the range and at least one designed negative strictly below
it (the doc's "designed negatives prove the grader is not lenient"); `semantic` needs at least one task.

**Regression test.** Pack with `fixtures` / `tasks` returning `[]`: assert `fixture.cases` and `semantic.gold` are `fail`.

## V3 (P0): a NaN reward passes every level

studio/pack.py:261 tests `abs(reward - expected) > 1e-6`, :288 tests `rewards[0] < hi - 1e-9`, :290 tests
`rewards[1] >= hi - 1e-9`. All three are false for NaN, so a grader that returns NaN for every episode passes the
template's own fixtures (which expect 1.0 and 0.0) and both semantic checks. `GradeReport` is a plain dataclass, so the
core does not stop it either (`CrucibleModel`'s `allow_inf_nan=False` does not apply).

```
nan_reward {"exit": 0, "ok": true, "flagged": [], "details": {}}
```

**Fix.** A non-finite reward (or one that is not a real number) is `fail` with the case name, in both levels:
`if not isinstance(reward, (int, float)) or not math.isfinite(reward): bad.append(...)`; write the comparisons in the
"good" direction (`not (reward >= hi - eps)`) so NaN fails by construction.

**Regression test.** Grader returns `float("nan")`; assert `fixture.cases`, `semantic.gold` and `semantic.noop` all fail.

## V4 (P0): rewards outside `reward.range` pass

Neither level compares graded rewards, or the fixtures' expected rewards, with `reward.range`. `semantic.gold` accepts
anything `>= hi`.

```
out_of_range {"exit": 0, "ok": true, ...}   # gold 1.5, every negative -3.0, range [0, 1], fixtures agree with the grader
```

**Fix.** Every graded reward and every fixture `reward` must lie in `[lo, hi]` (fail otherwise); gold must equal `hi`
within tolerance, not exceed it.

**Regression test.** The `out_of_range` mutant: assert `fixture.cases` fails with "outside reward.range".

## V5 (P0): a misspelled `redaction_env` leaks the secret it was meant to remove

`RedactionPolicy.with_env_file` (connectors/redact.py:125) reads the file only `if p.exists()`, otherwise it returns a
policy with no secret values, silently. `label.ingest` (studio/label.py:72) passes `pack_dir / spec.redaction_env`
without checking it, and `static.ingest` (studio/pack.py:202-214) never looks at `redaction_env`. A value that matches no
built-in secret pattern is then stored verbatim.

```
# ing/.env.local: DB_PASSWORD=zq9Xk2pp81QQ7mVt ; ing/data/logs/a.jsonl text: "connect with zq9Xk2pp81QQ7mVt"
== redaction_env ./.env.local
  pass  static.ingest          1 source(s)
label exit 0
"text":"connect with <secret>"
== redaction_env ./.env.lcoal
  pass  static.ingest          1 source(s)
label exit 0
lab/traces.jsonl
lab/store/blobs/ff/ffb95fad...
"text":"connect with zq9Xk2pp81QQ7mVt"
```

The coding pack's `redaction_env: ../../../.env` resolves on this checkout; on any checkout without that file (CI, a
colleague's clone) the same silent path runs.

**Fix.** `static.ingest` fails when a declared `redaction_env` does not exist; `with_env_file` raises `FileNotFoundError`
for an explicit path that does not exist (a caller wanting "optional" passes `missing_ok=True`); `label` records
`redaction_env: loaded N values from <path>` in `summary.json` (count only, never values).

**Regression test.** Pack with a file_drop log containing a value from a dotenv, `redaction_env` misspelled: assert
`static.ingest` fails, and `label.ingest` raises instead of storing the value.

## V6 (P1): `requires` opts a pack out of the semantic level, and the skip is under-recorded

`missing_requirements` treats `gpu`, `paid` and `network` as always missing unless `--allow` names them. The pack itself
declares `requires`, and nothing checks the claim, so any pack can turn semantic into `skip`; `skip` does not affect `ok`
or the exit code.

```
always1_gold_only_req_net {"exit": 0, "ok": true, "flagged": ["semantic.gold"],
  "details": {"semantic.gold": "skipped: live tasks need network (pass --allow network where they are)"}}
```

Also:
- Only `semantic.gold` carries the skip; there is no `semantic.noop` row, and `PackReport.to_json()` has no `skipped`
  list, so a consumer that reads `ok` or `problems` cannot tell a skipped semantic level from a passed one.
- When static has a `fail`, the fixture and semantic levels vanish from the report with no `skip` row (studio/pack.py:308).
- The coding pack declares no `tasks` hook, so `--allow gpu,docker` (the M14 doc: "semantic needs --allow gpu on the lab")
  only produces `warn semantic.tasks` and passes:

```
$ crucible pack check coding --level semantic --allow gpu,docker
  pass  fixture.cases          8 case(s) graded exactly as expected
  warn  semantic.tasks         no hooks.tasks: the gold/no-op check cannot run
```

**Fix.** `PackReport` gets `skipped: [level, reason]` and `complete: bool`; the CLI prints "semantic: SKIPPED (needs gpu)"
in the headline and `--level semantic` exits non-zero (e.g. 2) when it did not run unless `--allow-skip` is given. Emit
skip rows for both semantic ids and for levels not reached. Give the coding pack a `tasks` hook (or correct the doc).

**Regression test.** Template pack with `requires: [network]`: assert `report.to_json()["skipped"]` names semantic and the
CLI exit code is non-zero without `--allow-skip`.

## V7 (P1): a second pack with the same package name is checked with the first one's code

`resolve` (studio/pack.py:98-106) inserts `src/` into `sys.path` and calls `importlib.import_module`, which returns the
cached module when the name is already in `sys.modules`. Package names collide whenever a pack is forked, copied to be
edited, or re-scaffolded with the same name. Long-lived processes hit it: the explorer server, pytest, notebooks, `--all`.

```
$ python stale.py      # base = template pack; base_mut = a copy whose grader always returns 1.0
base      ok: True
base_mut  ok (same process): True ['5 case(s) graded exactly as expected', '2 task(s): gold reaches 1.0', ...]
--- fresh process:
base (semantic): FAILED
  fail  fixture.cases          noop: graded 1.0, expected 0.0; wrong_word: graded 1.0, expected 0.0; ...
```

**Fix.** Before resolving, check `sys.modules[package].__file__` is under `pack_dir/src`; if it is another pack's module,
refuse ("package X is already loaded from Y; check this pack in a fresh process") or run each check in a subprocess
(the CLI already pays one process per invocation; the explorer and `--all` should too).

**Regression test.** In one process: `check(a)` then `check(copy_of_a_with_always_one_grader)`; assert the second is not ok
(or raises the explicit "already loaded from" error).

## V8 (P1): `fails_gate` passes when the gate does not exist

studio/pack.py:263: `elif fails and gates.get(fails) == "pass"`. A gate name the grader never emits (a typo, a renamed
gate) is `None`, so the designed-negative assertion silently holds.

```
gate_typo_always_pass_gate {"exit": 0, "ok": true, ...}   # grader gates {"result": "pass"}; fixtures say fails_gate: outcome
```

The shipped packs are consistent today (`gates_audit.py`: every toy and finance_ap `fails_gate` names an emitted gate and
it is `fail`), so this is a guard, not a live false pass.

**Fix.** `if fails and gates.get(fails) != "fail": bad.append(f"{name}: gate {fails} is {gates.get(fails, 'absent')}, expected fail")`.

**Regression test.** The `gate_typo` mutant: assert `fixture.cases` fails with "absent".

## V9 (P1): `determinism` is declared and never verified

No code reads `PackManifest.determinism` (grep: only its definition). `pack check` grades each fixture once and each task
once, and calls `tasks()` once; it never calls `generator`.

```
nondet_grader: 11 runs -> 8 False, 3 True       # 15% verdict flips, 9 grades per check
nondet_tasks   {"exit": 0, "ok": true, ...}      # word drawn from SystemRandom, determinism: seeded
```

**Fix.** For `seeded` packs: call `tasks()` twice and compare task ids and instruction hashes; call `generator(s)` twice
for two seeds and compare; grade each fixture case twice (fresh workdirs) and require identical reward and gates. Report as
`fixture.determinism` / `semantic.determinism` (fail).

**Regression test.** `nondet_tasks` mutant: assert `semantic.determinism` fails; a grader keyed on a call counter
(1.0 on the first call, 0.0 on the second): assert the fixture determinism check fails.

## V10 (P1): the placeholder check is vacuous for the template it exists for

`static.placeholders` greps `TODO` in `pack.yaml`, `README.md` and `pyproject.toml` only (studio/pack.py:179-181). The
template marks what to edit with "(edit this description)" and "Replace the task in ...", never `TODO`; `src/` is never
scanned; scaffold tokens (`__PACK_NAME__`, `__PACK_PKG__`) are not looked for; `PackManifest.name` is any string (the
`NAME` regex is only applied by `pack new`).

```
$ cp -r packs/_template rawtpl; sed -i '' '/template_version/d' rawtpl/pack.yaml
$ crucible pack check rawtpl --level semantic
__PACK_NAME__ (semantic): ok
  pass  static.placeholders    none
todo_in_src {"exit": 0, "ok": true, ...}   # "# TODO: write a real grader" in studio.py, "(edit this description)" in README
```

An untouched `pack new demo` (the greeting task, "edit this description") also passes every level: the gate "pack new
passes check at every level" is met by a pack that has no domain content.

**Fix.** Scan every text file under the pack for `__PACK_[A-Z_]+__`, `TODO`, `FIXME` and the template's edit markers;
validate `name` with `NAME` in `PackManifest`; warn when `studio.py` still equals the scaffolded template (kind `domain`).

**Regression test.** The raw-template copy without `template_version`: assert `static.placeholders` and `static.manifest`
fail.

## V11 (P1): hooks and the entry point are checked for importability only

`static.entry_point` and `static.hooks` (studio/pack.py:184-200) pass any importable callable:
- a hook may live in another pack (`foreign_hooks`: our grader always returns 1.0, `tasks` / `fixtures` point at
  `crucible_pack_toy.studio`, check runs the toy pack and passes) or in the standard library (`builtins:list`, V2);
- `entry_point` is never compared with `[project.entry-points."crucible.plugins"]` in `pyproject.toml`, and `register()` is
  never called (`entry_point_broken`: pyproject names `no_such_module_...:register`, `register` raises, exit 0). The pack
  then fails at plugin load into `registry.failures` with only a log line (core/registry.py:107-118).

**Fix.** Require every hook's and the entry point's module to be `package` or under it; parse `pyproject.toml` and require
the plugin entry point to equal `entry_point`; call `register(Registry())` on a scratch registry and require it to register
`pack <name>`; optionally check hook arity with `inspect.signature`.

**Regression test.** `foreign_hooks` and `entry_point_broken` mutants: assert `static.hooks` / `static.entry_point` fail.

## V12 (P1): manifest fields that no consumer honours

- `reward.higher_is_better` is read by nothing; the semantic check assumes higher is better, so an honest cost-style pack
  (`higher_is_better: false`, gold 0, untouched 1) fails `semantic.gold` and `semantic.noop` (`lower_is_better` row).
- `determinism` is read by nothing (V9).
- `reward.range` and `label.threshold` are read only by `pack check` and `label`. `crucible bench run` (cli/studio.py
  `_bench_pack`) passes no threshold, so `run_bench` uses 1.0, and `gold_and_noop_check` requires `gold == 1.0`
  (bench/run.py:171). A pack on [0, 10] with threshold 10 passes `pack check --level semantic`, then:

```
$ crucible bench run ten --fresh 0 --k 1 --json
threshold 1.0
scripted:gold {'mean_reward': 10.0, 'pass@1': 1.0}     # a partial-credit 2.0 / 10 would also count as solved
```

- `label.threshold` is not validated against `reward.range` (`threshold_out_of_range`: 7.0 on [0, 1] passes; `label`
  would then mark nothing solved).
- `requires` is ignored by `build` / `bench run` (they take their own `--docker` / `--paid` flags).

**Fix.** Validate in `PackManifest._contract`: `lo <= label.threshold <= hi`; reject `higher_is_better: false` until the
semantic check and bench support it (or implement the mirrored checks); pass `threshold=m.label.threshold` (and the
range) from `_bench_pack` / `build` into `run_bench` and `gold_and_noop_check`.

**Regression test.** `PackManifest(..., label={"threshold": 7}, reward={"range": [0, 1]})` raises; bench run of the `ten`
pack records `config.threshold == 10.0`.

## V13 (P1): a freshly scaffolded pack passes check but its next step fails

`pack new` prints `crucible bench run <dest>` as the third next step (studio/pack.py:157). The template's `GreetingTask`
has `gold()` but no `oracle_actions` / `oracle` / `bench_gold`, so every generated item fails `gold_and_noop_check`
("no oracle to validate against", bench/run.py:139-161). `bench_gold` is attached only to `hooks.tasks` anchors.

```
$ crucible pack new base --dir base && crucible pack check base --level semantic   # ok
$ crucible bench run base --out baseruns --json
Error: ValueError: only 0 of 2 fresh items passed validation; yields {'generated': {'tried': 5, 'passed': 0, 'exhausted': True}};
rejected [{... 'checks': {'gold_and_noop': 'no oracle to validate against'}}, ...]
```

`pack check` never calls `generator`, `policies`, `stages`, `credit`, `model` or `forge`; their contracts are covered
only by "importable and callable".

**Fix.** Give `GreetingTask` an `oracle_actions()` (return `self.gold()`); add a semantic sub-check that calls
`generator(seed)` for two seeds and runs `gold_and_noop_check` on each, and one that `policies()` returns `PolicyRef`s.

**Regression test.** `pack new demo` then `crucible bench run demo --fresh 1 --k 1` exits 0 (extend
`test_readme_path_through_the_cli`).

## V14 (P1): crashes are tracebacks, not check results; the documented exit 3 does not exist

The module docstring (studio/pack.py:25) promises "3 for an internal error". `check()` has no exception handling around
the fixture and semantic levels, and the static ingest glob is not guarded. Click turns the exception into exit 1 with a
traceback; in `--all` mode (cli/studio.py:55) the list comprehension stops at the first crash and prints no report.

```
raising {"exit": 1, "ok": null, "crash": "RuntimeError: grader crashed"}
ingest glob "/etc/hosts":  NotImplementedError: Non-relative patterns are unsupported   (traceback from pack check)
```

**Fix.** Wrap each level: an exception becomes `CheckResult("<level>.error", "fail", "<type>: <msg>")` and the report
continues; reserve exit 3 for errors outside a pack's code; `--all` collects every report.

**Regression test.** `raising` mutant: `check()` returns a report with `fixture.error` failed (no exception), and
`pack check --all` with one raising pack still reports the others.

## V15 (P1): pack `src/` stays first on `sys.path` and shadows other modules

`resolve` prepends `pack_dir/src` and never removes it. `static.layout` checks that `src/<package>` exists but not that
`src/` holds nothing else, so a pack's stray top-level module (a vendored helper, a test stub named after a library) wins
every later import of that name in the process.

```
$ printf 'SHADOW = "pack-supplied numpy"\n' > shadow/src/numpy.py
$ crucible pack check shadow --level semantic        # ok
after check(): numpy.__file__ -> <h1>/shadow/src/numpy.py  pack-supplied numpy
```

**Fix.** `static.layout` fails when `src/` contains top-level modules or packages other than `package`; restore
`sys.path` after the check (or import via a path-scoped finder); run checks of untrusted packs in a subprocess.

**Regression test.** Pack with `src/numpy.py`: assert `static.layout` fails; after `check()`, `sys.path` equals its value
before.

## V16 (P1): the pack-check tests are vacuous for the fixture and semantic levels

`nullcheck_plugin.py` replaces `check_fixture` and `check_semantic` with functions that always return `pass` (with the
same check ids). The whole pack and studio suite stays green:

```
$ PYTHONPATH=<h1> .venv/bin/python -m pytest -p nullcheck_plugin -q tests/unit/test_packs.py tests/unit/test_studio.py --no-cov
7 passed in 38.28s
```

The only negative assertion on `pack check` is `license: unknown` (static). `test_all_packs_fixture_level` also silently
`continue`s past the coding pack when `../migration_tasks` is absent (a pass, not a skip).

**Fix.** Add `tests/unit/test_m14_audit_check.py` with the mutants above as parametrized cases (each must be not ok), and
make the coding exclusion a `pytest.skip` with a reason.

## V17 (P1): ingest paths are not confined to anything

`IngestSpec` documents `./` and `../` as pack-relative "so a pack can point at data outside itself without absolute
paths", but no root bounds them, absolute `root` is accepted, and symlinks are followed. `static.ingest` only checks that
something matches.

```
== glob: "../outside/*.jsonl"        static.ingest pass; would_ingest .../h1/ing/../outside/x.jsonl
== root: "/etc", glob: "hosts"       static.ingest pass; would_ingest /etc/hosts
== glob: "data/link/*.jsonl" (data/link -> ../../outside)   static.ingest pass; would_ingest .../data/link/x.jsonl
== glob: "/etc/hosts"                pack check crashes (NotImplementedError), label accepts it
```

A pack is code, so this is not privilege escalation on its own; the risk is a declarative `pack.yaml` that a reviewer reads
as data pulling arbitrary host files into a trace store (redacted only by pattern, V5). The coding pack relies on `../`
reaching the repository root.

**Fix.** Resolve every ingest path (`Path.resolve()`, following symlinks) and require it under an allowed root: the pack
directory, plus roots named in configuration (e.g. `CRUCIBLE_DATA_ROOTS`, default the repository root); reject absolute
`root` / `glob`; report a glob error as a `static.ingest` failure.

**Regression test.** `root: /etc` and a symlink to outside the allowed roots: assert `static.ingest` fails and
`label.ingest` refuses.

## V18 (P2): licence check

`m.license.strip().lower() not in ("", "unknown", "todo")` accepts `TBD`, `none`, `NOASSERTION`, `proprietary`
(`license_tbd`, `license_none` pass). **Fix:** validate against the SPDX list (or a short allowlist plus
`LicenseRef-*` with a LICENSE file present). **Test:** `license: TBD` fails.

## V19 (P2): pack names are not unique and installed packs cannot be found by name

- `pack new toy --dir X` succeeds although `toy` exists; nothing checks names against `all_packs()` or the registry.
- `find_pack(ref)` first tries `Path(ref)/pack.yaml` relative to the CWD, so a directory called `toy` in the working
  directory replaces the toy pack for `pack check`, `label --pack`, `bench run`, `build`, `model`:

```
$ cd cwdtest && crucible pack new toy --dir toy && crucible pack check toy --json | grep path
 "path": "<h1>/cwdtest/toy",
$ cd .. && crucible pack check toy --json | grep path
 "path": ".../crucible/packs/_toy",
```

- Under `packs/`, the first sorted directory with a matching `name` wins (a `0toy/` or `Toy/` sorts before `_toy/`), with
  no duplicate detection.
- Every pack's `register()` registers kind `pack`, but nothing reads that kind; `find_pack` cannot find a pip-installed
  pack by name, contrary to `resolve`'s "installed packs import as they are". Duplicate entry-point names fail into
  `registry.failures` with a warning only.

**Fix.** Names resolve through one index (packs/ plus the registry's `pack` kind), duplicates are an error naming both
paths, a path is only taken as a path when it contains a separator or `--path` is given, and `pack new` refuses a name
already in the index. **Test:** two packs named `toy` → `find_pack("toy")` raises "ambiguous".

## V20 (P2): ingest contract gaps

- `init` / `scope` strings are pack-relative only with a `./` or `../` prefix (studio/label.py:29-36); `root: data/runs`
  resolves against the CWD while file_drop `glob` / `root` are always pack-relative.
- `_local_paths` (studio/pack.py:167-168) is flat: `scope: {paths: ["./x"]}` or nested dicts are not checked; `root` of a
  non-file_drop source is never checked or used.
- `discover` is not checked to be a method of the connector (a typo passes static and fails in `label`).
- Fields a connector ignores (`init` on file_drop; `glob`, `root`, `extra` on others) are accepted silently.

**Fix.** One resolver shared by `check_static` and `label` (recursive, prefix-free pack-relative for path-typed fields);
`static.ingest` instantiates the connector with resolved `init` and checks `hasattr(conn, discover)`; the IngestSpec
validator rejects fields the connector kind does not use.

## V21 (P2): the no-op threshold is strict

`semantic.noop` fails only at `>= hi` (studio/pack.py:290): an untouched task scoring 0.999 of [0, 1] passes
(`noop_near_top`). **Fix:** require `noop < label.threshold` and a minimum gap (e.g. `gold - noop >= 0.5 * (hi - lo)`,
or a declared `semantic.min_gap`). **Test:** `noop_near_top` fails.

## V22 (P2): robustness

- `missing_requirements` runs `docker info` with no timeout (studio/pack.py:237); a hung daemon hangs `pack check`.
  **Fix:** `timeout=10`, treat a timeout as missing.
- `check_semantic` calls `task.cleanup()` after the grade, outside `finally` (studio/pack.py:292-294); a raising episode
  leaks the task's resources. **Fix:** move it into a `finally`.

## What held

- The conflict-safe scaffold: identical files are kept, any conflict aborts before a write, symlinks inside the
  destination are refused, new files open with `"x"`; `pack new` rejects bad names.
- `PackManifest` rejects unknown keys (`extra="forbid"`), a non-increasing range, `credit` without ingest, and hook strings
  without `module:attr`.
- Secret values from a found `redaction_env` never serialize (`secret_values` is `exclude=True`) and do not appear in
  `configs/resolved/label.json` or `summary.json`.
- The shipped toy and finance_ap fixtures name gates the graders emit, and each designed negative fails its gate.
- Plain always-1.0 and always-0.0 graders with the template's designed fixtures are caught by both deeper levels.
