# M9 study: defects in `accept` (Acceptance Contract, defect registry, repair loop, flaky-task admission)

Scope: `C/accept/` (`model.py`, `policy.py`, `key.py`, `cache.py`, `plan.py`, `run.py`, `contract.py`, `repair.py`, `checks/{base,core,extended,composition,redteam}.py`, `mutation/`, `registry/{model,run,fixtures}.py` and `registry/defects/*.yaml`), `CP/accept_fixtures.py`, `TP/accept_gate.py`, `TP/forge/task.py` (the toy adapter's M9 capabilities), `C/train/admission.py`, and the entry points that run acceptance (`C/cli/studio.py` `accept`, `C/studio/build.py`).

Spec: ARCHITECTURE "### M9 `accept`" and its gate row (`ARCHITECTURE.md:808`). DESIGN §9 (A1–A13, defect registry, imports I0–I10), the task-manager entry (`DESIGN.md:1299-1326`) and the admission entry (`DESIGN.md:2571`). Also read: `docs/modules/M09-accept.md`, `docs/research/m9-m10/01-acceptance.md`, `03-hf-rl-env-datasets.md` and `docs/research/improvements/M09-M11.md`.

Paths: `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`. `TP` = `crucible/packs/_toy/src/crucible_pack_toy/`.

Method: a read-only study on HEAD `526fdb5`, using `crucible/.venv` (Python 3.12.9) on macOS. **[ran]** means reproduced with a probe; **[read]** means established from the code alone. The probes are in the session scratchpad at `m9hunt/`:
- `p0_toy_gate.py`: the toy gate, redirected;
- `p1_checks.py`: the contract runner on `FakeTask`;
- `p2_repair_admission.py`: the repair loop and admission;
- `p3_store.py`: the grade store and cache.

Probes use stubs only. No Docker, GPU, paid, networked or lab run was made.

Offline gate re-run:
- `TP/accept_gate.py` was run with `--no-paid`, the report path and cache in the scratchpad, and the forge outputs redirected. Both forges are needed because `forge_hindsight` writes `docs/modules/reports/m8_toy_hindsight.json` unconditionally (`TP/forge/hindsight.py:202`).
- It reproduces `m9_toy_g1.json`:
  - 2/11 accepted; 9 tasks blocked only by A10 (decided from the recorded `data/runs` calibration runs);
  - the same mutation tables;
  - power tests: 13 proven, 10 per-kit, 0 not proven;
  - registry `reg:23:dc0385c64099`.
- The only difference is that A14–A18 now appear. They were added after the report: A16 passes, and the others are N/A.
- The gate prints `"pass": false` by design (`accept_gate.py:59` requires 11/11), and `spend_usd` is hard-coded to 0.0 (`:39`, `:58`).

## 1. Requirement ledger

Status: **live** = built and behaves as specified; **partial** = built, but missing a specified part, or with a reproduced hole; **wrong** = contradicts the spec; **missing** = not built.

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| L1 | "A task enters the pool only if A1–A12 all pass"; `crucible accept <task>` automates them and writes `calibration.json` (ARCH M9 Purpose; DESIGN §9 l.717) | **wrong** | Neither user path uses `accept_task`. `crucible accept` (`C/cli/studio.py:171-191`) and `crucible build` (`C/studio/build.py:51-54`) both run `contract.accept`, the M8 subset. That subset marks A4, A6 and A9–A13 as `skip`, and `AcceptReport.ok` ignores skips (`contract.py:22-24,72-73`). `build` then exports and benchmarks those tasks. The command takes a pack, not a task, and writes no `calibration.json` (V2) |
| L2 | Status model: N/A only from task facts, absence never implies N/A, pass/fail need evidence, `accepted` derived (DESIGN §9 as-built; M09 doc) | partial | Derivation and evidence are live (`model.py:128-145`). Author-asserted attributes produce N/A, and so does an absent capability: `attack_surface="none"` gives A13 N/A (`extended.py:254`), `horizon` gives S1 N/A (`:271-272`), and no `property_audit` gives A16 N/A (`redteam.py:27-28`) (V10) |
| L3 | A1: ≥ 3 oracle grades = 1 in the declared environment, gold check, ≥ 15% perf headroom, ≥ 10 fresh seeds without a crash (DESIGN §9) | partial | Logic is live (`core.py:21-70`). The headroom and fresh-seed parts run only if the adapter offers `perf_bars`/`fresh_gates`. No real adapter does (grep: none in `packs/`). `declared_env` is recorded but never compared. A grade that stays masked on a fresh seed counts as a task failure (V14) |
| L4 | A2: no-op, unmodified original and naive solution all score 0 (DESIGN §9) | partial | No-op and `floor` negatives only. Nothing requires an "unmodified original". `outcome=None` is accepted (`core.py:81`). In the legacy path a masked no-op grade counts as a floor pass (V2) |
| L5 | A3: ≥ 1 negative per family (≥ 4 for H3+), each failing its gate; ≥ 90% of generic mutants killed (DESIGN §9) | partial | Negatives and families are live. The mutation score is **vacuous** with 0 mutants, and a surviving reward-1 mutant whose end state equals the oracle's is labelled "equivalent" and excluded. Both pass (V3). There are no patch operators, and `swap_target` is documented but not implemented (`mutation/actions.py:9` vs `:38-61`) |
| L6 | A4: zero false negatives over ≥ 40 fresh draws + induced; ≥ 3 identical reference runs (DESIGN §9) | partial | Draws and the CP bound are correct (`calibrate.py:25-65`; 40 draws, 0 fails gives 0.072). Draws are skipped silently without `fresh_gates`. Self-consistency is skipped silently without `reference_fingerprints` (`core.py:168,186`), so A4 passes on stability alone (V6) |
| L7 | A5: hidden strings vs everything agent-visible; nothing hidden in the agent image (DESIGN §9) | **wrong** | The image audit reads `rep.hits`, but `LeakReport` has `.findings` (`core.py:219-220`; `verify/leak_audit.py:47-56`). An agent image holding the answer passes (V1). It also inherits M7 V11 (roots skipped) |
| L8 | A6: containment audit passes and a tamper attempt is blocked (DESIGN §9) | partial | An audit whose items are all `skip` passes (`extended.py:37`). A tamper with no effect probe counts as blocked (`:43`) (V9) |
| L9 | A7: instruction ↔ check coverage, Harbor `harbor check` rubric, manual sign-off for H4+ (DESIGN §9; ARCH Takes) | partial / rubric **missing** | Only a static check: the first 40 characters of each property must appear in the instruction, and each check's author-declared `states` must map to properties (`core.py:225-239`). There is no rubric and no judge, so there is no "judge unavailable" fallback to audit. Because `states` is trusted, a check can claim a property it does not grade (V4 relies on this) |
| L10 | A8: ≥ 80% provenance; a trap cites its signal (DESIGN §9) | live | `contract.py:188-209` via `core.py:242-248`. A dangling provenance is checked only for `sig_` ids |
| L11 | A9: digests pinned; two rebuilds give identical reference fingerprints (DESIGN §9) | partial | The drift check runs only when the adapter offers `pinned_images`. Without it, any digest passes. **[ran]**: verifier digests `b…` and `c…` both pass. The fingerprint check is optional (`extended.py:61-71`). No real adapter offers either (V6) |
| L12 | A10: ≥ 2 frontier runs with failure taxonomy; pass rate in the split's band (DESIGN §9) | live, with caveats | Masked runs are excluded and the Wilson interval is used. Top-up sampling happens only when the estimate is outside the band (an optional-stopping bias). Calibration runs are not tied to the task version (V15) |
| L13 | A11: dirty workspace, ignored build dirs, blank usage classified correctly (DESIGN §9) | partial | It injects blank usage, credits, contention and grader crash (`extended.py:176-187`). Dirty workspace and ignored build directories are not injected. `B.5-empty-patch` names detector `A11.empty_collection`, but A11 has no such case (V19) |
| L14 | A12: full grade fits the timeout; fast/full Spearman ≥ 0.8 (DESIGN §9) | partial | It reads only the in-memory `ctx.grades`. When A1/A3 come from the cache, A12 errors (V11). Mutants and attacks are excluded from the fast-pass/full-fail set (`extended.py:228-233`) |
| L15 | A13: every cheat mutant scores 0, with a control showing the attack took effect (DESIGN §9; M09 doc) | partial | An attack whose `effect` is `None` has no control, yet passes (`extended.py:258-266`) (V9). The attacks themselves are adapter-supplied. Toy: 3 attacks with controls (`TP/forge/task.py:255-288`) |
| L16 | S1: sign-off bound to the key for H4+ | partial | Bound to `key_digest` (`extended.py:280-287`). It can be overridden (`NO_OVERRIDE` omits S1, `model.py:115`), and it can be skipped by declaring `horizon` < H4 (V16, V10) |
| L17 | Acceptance key binds everything that decides a verdict (ARCH; `key.py` docstring) | partial | `kit_hash` covers only `verify/runtime/*.py` (`key.py:23,34-36`). It omits `accept/` itself, `verify/{calibrate,leak_audit,tiers}.py`, `world/leaks.py` and `forge/{learnability,hacker}.py`. Check versions are bumped by hand, and A10 changed twice at version 1 (V5) |
| L18 | Cache and grade store: reuse only if evidence still hashes; grades keyed by (grading env, solution **content**) (M09 doc) | partial | Evidence re-hashing is live (`cache.py:28-37`). Solutions are hashed with `json.dumps(default=repr)`, so a `Path` solution is keyed by name, not content (V8). Writes are not atomic: a torn grade is an error forever, and a torn cache file crashes the whole run (V12) |
| L19 | Masked/infra outcomes never counted as pass or fail | partial | `ctx.grade` retries, then raises `InfraUnavailable`, and the check reports `error` (`base.py:97-112`). Two exceptions: A1 fresh counts a persistent mask as a fail (V14), and legacy A1/A2 count a masked no-op as a floor pass (V2) |
| L20 | Registry: one YAML per defect; admitted only with a power test or a named per-kit test; version in every key; backfill quarantines (DESIGN §9; ARCH) | partial | 23 entries. 13 fixtures are proven and 10 per-kit tests exist (all node ids resolve). Backfill keeps a task whose detector errors or is pending (V7). Nothing in `src/` or `packs/` reads a quarantine list |
| L21 | Exit gate: re-injecting Haar nondeterminism, the weak NCV negative and a stale verifier image is caught (ARCH M9) | partial | All 3 are proven only on `FakeTask`, through capabilities (`reference_fingerprints`, `pinned_images`) that no real adapter implements (V6). On task04 the Haar and stale-image detectors cannot fire |
| L22 | Exit gate: tasks 01–04 pass automatically (ARCH M9) | **missing** | There is no coding-pack acceptance adapter (`grep key_inputs/isolation_audit/attacks packs/coding` finds nothing). There are no patch mutation operators. `CP/accept_fixtures.py` holds only the racecheck power test. M09 doc: "G2/G3/G4a/G4b ⏳" |
| L23 | Exit gate: the toy tasks pass (ARCH M9) | partial (as recorded) | Reproduced 2/11 accepted. The 9 rejections are A10 band verdicts, which the M09 doc calls correct. Every other check passes on all 11 |
| L24 | Imports: `imports/` runs I0–I10 and writes `import_report.json` (ARCH M9; DESIGN §9) | **missing** | There is no `C/imports/` and no `import_report`. Three HF defects (D1, D8, D10) exist only as `FakeTask` registry fixtures. `improvements/M09-M11.md` #3 and #8 already flag this |
| L25 | Repair loop: commit needs a passing objective verification; a candidate can only strengthen its audit (DESIGN l.1300-1312) | partial | Commit is gated (`repair.py:325-330`). The verification set is A1, A7 and A16 only (`:201`). A candidate that drops a check guarding a stated property passes and is committed, and A3 on the committed version then fails (V4) |
| L26 | Flaky-task admission (DESIGN l.2571, "same rule as A4") | partial | 8 reruns, early stop and disagreement handling are live. A partial reward is "stable". Kept tasks with an infra run were judged on 7 runs, but the bound assumes 8. An unhashable verdict crashes the screen (V13) |
| L27 | Research picks (improvements M09-M11 #2–#9) | missing (open) | None is built: short-circuit, A19 malformed submissions, severity policy, verifier lint, concurrency, atomic writes, stratified mutants |

## 2. Defect candidates

Severity: **P0** = a broken, leaking or hackable task is accepted, or a good task's verdict is forged; **P1** = a wrong verdict, a mask the task author can trigger, or an audit hole in normal use; **P2** = an edge case.

### Broken or leaking tasks accepted

**V1 [P0][ran] The A5 agent-image audit can never fail.** `core.py:218-221` calls `audit_image` and then reads `rep.hits`. `LeakReport` has `findings` and no `hits` (`verify/leak_audit.py:47-56`). So `getattr(rep, "hits", None)` is always `None`, and `m["image"]` falls back to `str(rep)`.
- Probe (p1 `a5_image`): `audit_image` stubbed to return one `hidden_string` finding in `solution/solve.sh`. A5 gives `pass`, with the finding visible only inside `measured.image` as a string.
- Expected: `fail`. The unit test never exercises this branch (`core.py` 214-221 are uncovered).
- Fix: use `rep.findings` / `rep.ok`. Add a power test (registry entry "answer in agent image").

**V2 [P0][ran] Both user paths accept tasks with only the M8 subset; the full contract is not wired to any CLI.**
- What the paths do:
  - `crucible accept` (`C/cli/studio.py:171-191`) and `crucible build` (`C/studio/build.py:51-54`) call `contract.accept`.
  - That subset records A4, A6 and A9–A13 as `skip`, and `ok` ignores skipped checks (`contract.py:22-24,72-73`).
  - `build` then exports the tasks to canonical rewards and uses them as benchmark anchors (`build.py:59-75`).
- Probe (p1 `legacy_contract`): a `FakeTask` whose attack `cheat` scores 1.0, with a confirmed effect. Legacy `ok: True` with every implemented check `pass`. The full runner gives `A13 fail: attack cheat (forged) not caught at g: reward 1.0`.
- Probe, same task with the no-op grade **masked** (`infra.status = masked`, reward 0.0): legacy `A1 pass` (an empty gate dict reads as "false at start") and `A2 pass`. Legacy `a1_oracle`/`a2_floors` never check `infra` (`contract.py:86-110`).
- On the toy pack, the M8 subset passes all 11 tasks while the full contract accepts 2/11 (`m9_toy_g1.json`, reproduced). So `crucible build` ships 9 tasks M9 rejects.
- Fix:
  - `crucible accept` and `build` call `accept_task` with the pack's needs.
  - Tasks whose checks are `pending` are listed as not accepted, never exported.
  - The legacy checks require `infra.status == "ok"` on every grade.
  - Write the `calibration.json` that DESIGN promises, or change the doc.

**V3 [P1][ran] A3's mutation score passes vacuously, and "equivalent" hides surviving mutants.**
- `score = killed / scored if scored else None`, and the threshold applies only `if ... scored` (`core.py:142-145`):
  - p1: `mutants() -> []` gives `A3 pass, score None`;
  - `is_actions([])` is `True`, so an empty oracle also yields 0 mutants (`mutation/actions.py:71-72`).
- Equivalence is judged by end state alone (`core.py:135-137`).
  - p1: a `drop_message` mutant (skip the approval) scores **1.0**, and its state fingerprint equals the oracle's. It is labelled `equivalent` and excluded: `A3 pass, score None, survivors []`.
  - This is exactly the B.15 forged-approval class (a missing event check). A message drop never changes state, so `drop_message` survivors are excluded by construction.
- The weak-negative power test itself runs with `mutants: list` (`registry/fixtures.py:175`), so its clean case passes A3 with no mutation evidence.
- Fix:
  - zero valid mutants is a `fail`, or N/A with a closed reason such as `no_mutant_operator`;
  - "equivalent" requires an equal fingerprint **and** an equal verdict vector to the oracle's. A surviving mutant with reward > 0 is never equivalent;
  - report the equivalent share.

**V4 [P1][ran] The repair loop can weaken a task and commit it.**
- `ForgeWorkspace.verify` runs only `("A1", "A7", "A16")` (`repair.py:201,308-314`), and `commit` requires only that verification (`:325-330`).
- A2, A3 and A13 are not run. A16's `MutationHacker` only drops steps or interactions (`forge/hacker.py:73-92`). It never shifts a number, so a dropped bounds check is invisible to it, even with the `audit_floor`.
- Probe (p2):
  - The task has properties p1 ("Refund and note") and p2 ("Refunds never exceed 50 USD"), checks `refund_done` (claims p1, p2) and `amount_cap` (gate `cap`), and a negative `over_cap` (refund 400).
  - The implementer stub calls `drop_check("amount_cap")`. Verify gives `passed: True {A7 pass, A1 pass, A16 pass}`, and commit gives `committed: 1`.
  - On the committed version, A3 gives `fail: negative over_cap not killed at cap: reward 1.0`.
- A7 cannot see the loss because `states` is author-declared (L9). The only remaining guard is the LLM reviewer.
- Fix:
  - verification always includes A2, A3 and A13 (cheap on World tasks);
  - `drop_check` is refused when a committed negative or attack depends on the dropped gate, or every authored negative must still fail at its gate;
  - the A16 hacker also applies the A3 operators (numeric shift, enum swap).

**V5 [P1][ran] The acceptance key omits code that decides verdicts, and check versions were not bumped.**
- `kit_hash` hashes only `verify/runtime/*.py` plus `plan_files` (`key.py:23,34-36`). p1 shows the hashed tree is `['runtime']`.
- `contract_version` is a constant (`model.py:23`). Changes to the checks, to `verify/calibrate.py`, `verify/leak_audit.py`, `verify/tiers.py`, `world/leaks.py` or `forge/learnability.py` reuse cached verdicts unless the author bumps the `@register` version.
- Git shows it was not bumped:
  - A10 gained sequential top-up (`c3070de`) and ladder-inversion blocking (`9a2b401`) and is still `A10@1`;
  - `calibrate.flake_rate` and `leak_audit` changed in `e428bbd`, and A4@2 and A5@2 did not change.
- The cache in `data/accept/` holds `A10@1.json` results from 2026-10-09, before both A10 changes. They would be reused for an unchanged key.
- Fix: add to the key a hash of `accept/checks/*.py` plus every module a check imports (or of each check's function source and its imports). Replace the manual `version` with that hash.

**V6 [P1][ran] The registry's exit-gate re-injections are proven only on `FakeTask`, and their detectors fail open on real adapters.**
- A9's stale-image check exists only if the adapter offers `pinned_images` (`extended.py:61-64`). p1 shows two different verifier digests, both `pass`.
- A4's reference self-consistency runs only if `reference_fingerprints` exists (`core.py:186`). p1: `A4 pass, measured keys ['stability']`.
- The same applies to A1 `perf_bars`/`fresh_gates` and A9 `recorded_fingerprints`.
- No pack adapter implements any of these (grep over `packs/` and `src/` outside `accept/`). So on task04, the Haar (B.2) and stale-image (B.5) detectors can never fire. The registry's `applies_to: {has: [reference_build]}` is consulted only by `backfill`, never by `accept_task`.
- This contradicts "absence never implies N/A".
- Fix: derive task facts (`facts()`). When a fact is true (images, reference build, perf bars, fresh gates), the matching capability is required, and its absence is a `fail`. When no `pinned_images` exist for an image role, A9 fails.

**V7 [P1][ran] Backfill keeps tasks whose new detector errors or is pending.** `registry/run.py:48-49` quarantines only when `st == "fail"`. p1: `HF.D1-oracle-noop` backfilled on a task whose grader raises gives `{'t/err': 'kept'}`. Docs say backfill evaluates only accepted tasks (`run.py:5`), but it runs on every task passed in, and nothing consumes a quarantine. Fix: anything other than `pass`/`not_applicable` is quarantined (or `held`). Persist the list and make pool loaders refuse it.

### Wrong verdicts and audit holes

**V9 [P1][ran] Red-team checks that cannot fail their control.**
- A13: an attack with `effect=None` has no control, and `took is False` is the only "proved nothing" path (`extended.py:263`). p1: an "attack" that is just a wrong solution gives `A13 pass`.
- A6: audit items `skip` count as pass (`:37`), and a tamper with no effect probe is treated as blocked (`:43-46`). p1: an audit of three `skip` items plus one effect-less tamper gives `A6 pass`.
- Fix: `effect is None` is a fail ("attack has no control"). A6 requires at least one `pass` per audit layer and treats `skip` as a fail unless a closed N/A reason applies.

**V10 [P1][ran] N/A from author-asserted attributes or from absent capabilities.**
- p1: with `attack_surface="none"` and `horizon="H3"`:
  - `A13 not_applicable` (`extended.py:254`);
  - `S1 not_applicable` (`:271-272`);
  - `A16 not_applicable` because `property_audit` is absent (`redteam.py:27-28`).
- `model.py:4-6` says N/A is "derived from task facts, never asserted by an author", and `base.py:6-7` says a missing capability fails.
- Fix:
  - derive the attack surface from the task kind (World vs container);
  - derive the horizon from measured steps or the M8 card, not a free attribute;
  - a missing `property_audit` fails A16 unless the task fact `no_free_form_submission` holds.

**V11 [P2][ran] A12 errors when A1/A3 are reused from the cache.** A12 reads only `ctx.grades` (`extended.py:214,228`), which cached checks never fill. p1: A1/A3 pass, then `only=["A12"]` gives `error: ValueError: need two equal-length samples of size >= 2`. In normal use this hits any A12 re-run (version bump, earlier error) with an unchanged key. Fix: A12 grades the oracle and negatives through `ctx.grade` itself (the grade store makes this free).

**V8 [P2][ran] The grade store keys non-JSON solutions by `repr`, not content.** `base.py:81` uses `json.dumps([solution, run, variant], default=repr)`. A `Path` oracle whose file changes keeps its hash. p1: after the oracle file is broken, and the key changes, A1 gives `pass, rewards [1.0, 1.0, 1.0], new_grades 0`. Toy solutions are JSON action lists, so the toy is not affected. A coding adapter that passes patch paths would be. Fix: hash file and directory content for `Path` solutions, and reject other non-JSON solutions.

**V12 [P2][ran] Non-atomic writes in the grade store and cache.**
- `base.py:107-110` and `cache.py:43,47` write in place.
- p3: a truncated grade file gives `A1 error: JSONDecodeError` on every later run (it never heals, because the file is read before grading).
- A truncated `A1@2.json` makes `accept_task` raise `ValidationError` outside the per-check `try` (`run.py:48`), which aborts every check of the task.
- Fix: write to a temp file and `os.replace`. A corrupt store or cache entry is treated as a miss and deleted.

**V13 [P2][ran] Admission edge cases** (`train/admission.py`). p2:
- A reference with reward 0.5 and gate `fail` is `stable` and kept: `_passed` takes `v[0] > 0` (`:37-38`). A4 requires reward 1.
- A prompt with one infra run is `stable` after 7 verdicts. `flake_upper_95` still reports `CP(0, 8) = 0.312` (`:67-69`).
- A dict verdict raises `TypeError: unhashable type: 'dict'` at `set(seen)` (`:112`), outside the `try`, which aborts the whole screen.

Fix: the default `passed` requires reward == 1. Report the bound on the minimum verdict count. Freeze dict verdicts with `json.dumps(sort_keys=True)`.

**V14 [P2][read] A1 counts a fresh-seed grade that stays masked as a task failure** (`core.py:61-63`). A node problem across three retries (contention, OOM) rejects the task. Fix: split `grader_exception` (task) from the infra kinds (node) using `failure_kind`, and report the latter as `error`.

**V15 [P2][read] A10 statistics.**
- Top-up happens only while the estimate is outside the band (`extended.py:114`). Sampling stops as soon as it enters the band, which biases toward acceptance.
- `calibration_runs` reads every `data/runs/*/results.jsonl` row for the task id (`TP/forge/task.py:324-335`), including runs of earlier versions of the task.
- Fix: fix n in advance (or use a sequential test with a stopping rule), and filter runs by key or `grade_env`.

**V16 [P2][read] Overrides.** `NO_OVERRIDE` (`model.py:115`) omits S1 and A14–A18. An `Override` entry turns a human sign-off or a composite rejection into acceptance. Nothing validates `by` or `date`, or binds them to the key. Fix: add S1 and A14–A18 to `NO_OVERRIDE`, and bind overrides to `key_digest`.

**V17 [P2][read] A2 gaps.** There is no "unmodified original" floor, and `outcome=None` counts as 0 (`core.py:77-81`). Other reward channels (dense credit, rubric) are not checked. Fix: an adapter capability `original()`, required when the task is a port or migration. A2 inspects every channel in `GradeReport.meta`.

**V18 [P2][read] Mutation generator gaps.** `swap_target` is documented but absent (`mutation/actions.py:9`). Output is truncated to the first 40 in a fixed order (`:68`), so `drop_step` crowds out the later families. Fix: implement it, and stratify by family (research pick #9).

**V19 [P2][read] A11 does not inject the DESIGN cases.** There is no dirty workspace and no ignored build directory (`extended.py:176-187`). `B.5-empty-patch` names detector `A11.empty_collection`, which A11 does not have; its per-kit test covers the runtime only. Fix: add both injections, or move the detector id to the runtime kit.

## 3. Coverage

Command (scratch coverage file): `pytest -q -p no:cacheprovider --cov=crucible.accept --cov=crucible.train.admission tests/unit/test_{accept,rlm_manager,compose,hacker,fuse,perturb,ladder,env_noise,role_runner,judge_specialists}.py`. Result: 48 passed in 80 s. The `test_role_runner` Harbor tests ran locally.

| File | Cover | Untested risky lines |
|---|---|---|
| `accept/cache.py` | 100% | Corrupt JSON (V12) is not tested |
| `checks/base.py` | 100% | `default=repr` hashing (V8) and torn writes are not tested |
| `checks/core.py` | 87% | **214-221: the A5 image audit (V1)**; 126-127 no mutant source; 135-140 equivalent/survivor branches; 179-181 induced perturbations; 103 valid-alternative failure |
| `checks/extended.py` | 90% | 39/46/48 A6 failure paths; 67-71 A9 fingerprints; 218-244 A12 tiers; 254-256, 264 A13 N/A and no-control paths |
| `checks/composition.py` | 90% | 78-82 A15 bench-frontier taxonomy |
| `checks/redteam.py` | 97% | 49 no attempts |
| `contract.py` | **34%** | The legacy path that `crucible accept`/`build` use is barely tested here (V2) |
| `key.py` | 85% | 40-44 `docker_digest` |
| `model.py` | 97% | 138/140 `no_evidence`/`stale_result` blocking |
| `mutation/actions.py` | 90% | 45, 51-54 bool skip and enum swap |
| `registry/*` | 96–100% | |
| `repair.py` | 81% | FSM branches 417-461 (journal resume, reviewer errors, loop-back) |
| `run.py` | 91% | 55 pending needs; 63-66 `MissingCapability`/`InfraUnavailable`/crash paths |
| `train/admission.py` | 94% | 36/39 non-tuple verdicts; 88 missing rerun |
| **Total** | **85%** | |

`docs/modules/M09-accept.md` says "tests/unit/test_accept.py (9 tests)"; the file has 10.

## 4. Summary of fixes by priority

1. V1: read `findings` in A5.
2. V2: wire `accept_task` into `crucible accept`/`build`, and have the legacy checks reject masked grades.
3. V3: no vacuous mutation pass; equivalence needs an equal verdict.
4. V4: the repair verification set includes A2, A3 and A13.
5. V5: key on check code and its imports.
6. V6: facts make capabilities mandatory, and A9 needs pins.
7. V7: backfill quarantines anything that is not pass.
8. V9, V10: controls required; no author-asserted N/A.
9. The P2 items (V8, V11–V19).

Not built: tasks 01–04 adapters (L22) and `imports/` I0–I10 (L24). Both are exit-gate or ARCH claims that should be marked "to build" in ARCHITECTURE until they exist.
