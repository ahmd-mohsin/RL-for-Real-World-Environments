# M14 sweep G2: how ecosystems package, validate and distribute environments → requirements on Crucible packs

**Scope.** This sweep covers how the local ecosystems declare, validate, version and ship an environment package, and checks
`pack.yaml` / `crucible pack check` and the four packs against them at HEAD f2d9e18. The sources are:
- Harbor;
- verifiers, prime-environments and research-environments;
- NeMo-Gym;
- OpenEnv;
- agentenv-framework;
- FineEnvs and Repo2RLEnv;
- SWE-smith;
- EnterpriseOps-Gym;
- the MiMo verl fork and MiMo v2.6 RL release;
- the articles (HF blog, HF guide, Repo2RLEnv, the three Karotte posts).

**Method.**
- Every Crucible claim was probed with the real CLI (`.venv/bin/crucible pack check ... --level static|fixture|semantic`).
- I made 26 deliberately broken copies of `finance_ap` in the scratchpad.
- I wrote two in-process probes and one simulation over 2,000 `finance_ap` seeds.
- Probes live in `/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m14/g2/` (index in §6).
- `[ran]` means a probe in this sweep produced the evidence. `[read]` means it comes from reading code or the source.
- No GPU and no paid API were used. Nothing under `crucible/src`, `crucible/packs` or `crucible/tests` was edited.

**Keys.**
- `C/` = `crucible/`.
- `SP` = `C/src/crucible/studio/pack.py`.
- `PM` = `C/src/crucible/core/schema/pack.py`.
- `FA` = `C/packs/finance_ap/src/crucible_pack_finance_ap/`.
- Ecosystem paths are under `RL-Environments/06-frameworks-and-env-code/`:
  - `H` = harbor-framework-harbor;
  - `NG` = nvidia-nemo-gym;
  - `OE` = meta-pytorch-openenv;
  - `V` = primeintellect-ai-verifiers;
  - `PE` = primeintellect-ai-prime-environments;
  - `RE` = primeintellect-ai-research-environments;
  - `EOG` = servicenow-enterpriseops-gym;
  - `FE` = adithya-s-k-fineenvs;
  - `R2E` = huggingface-repo2rlenv;
  - `SS` = swe-bench-swe-smith;
  - `XV` = xiaomimimo-verl;
  - `K` = preferencemodel-karotte.
- `AE` = `agentenv-framework/`.
- `ART` = `RL-Environments/07-articles/`.
- `MO` = `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss`.

---

## Summary

### Headline defects

| # | Sev | Defect | Evidence |
|---|---|---|---|
| D1 | **P0** | **A grader that returns NaN passes every level.** The fixture check is `abs(reward - expected) > 1e-6` (`SP:261`). The gold check is `reward < hi - 1e-9` (`SP:288`). The no-op check is `reward >= hi - 1e-9` (`SP:290`). Each is False for NaN, so NaN is "exactly as expected", "reaches the top" and "stays below the top" at once. NeMo-Gym rejects this with `not lower <= observed <= upper` (`NG/nemo_gym/verifier_fixture.py:218-221`). MiMo masks non-finite or out-of-range rewards (`XV/recipes/general/general_agent/environment.py:577-578`). Harbor rejects non-finite reward files (`H/src/harbor/verifier/verifier.py:67-121`) | [ran] `always_nan`: `--level semantic` ok, exit 0 |
| D2 | **P0** | **A hackable grader passes `pack check` with exit 0.** (a) A missing `hooks.fixtures` is only a warn (`SP:45` `"fixture.hook": "warn"`, `SP:248-249`). The semantic level runs only gold and no-op (`SP:279`). So a grader that scores approve-everything 1.0 passes, because gold scores 1 and no-op scores 0. (b) A fixture case may carry its own `grade` callable, which the check trusts without ever calling the pack's grader (`SP:255-256`). Constant "fixtures" then pass a lenient grader cleanly. (c) A grader that ignores the reason code on rejects passes all 9 `finance_ap` fixtures. Only one wrong-reason negative exists, and it is a hold (`FA/studio.py:150-151`). No level runs the pack's own adversarial policies, even though `finance_ap` ships `scripted:approve_all` (`FA/studio.py:117-122`) | [ran] `lenient_grader_no_fixture_hook`: exit 0, `fixture.hook` warn. `lenient_grader_lying_fixtures`: exit 0, all pass. In-process, the lenient variant grades approve-everything at **1.0**, `controls: pass`. `grader_ignores_reject_reason`: exit 0 |
| D3 | **P0** | **No level checks for leaks, and the MVP pack's tasks never reach M9's leak audit.** I put the answer key in the KB (`data/kb/answers.md`). `pack check --level semantic` passes, and the agent reads the key with `kb.read_doc` on step 2. `crucible accept finance_ap` crashes with `AttributeError: 'NoneType' object has no attribute 'split'`: accept resolves `hooks.forge` unconditionally (`C/src/crucible/cli/studio.py:195`), and `finance_ap` has no forge hook. So A5 leak, A3 mutation and the rest of the M9 contract never ran on `finance_ap` tasks. The shipped `finance_ap` KB holds only `ap_policy.md` (`FA/world.py:126-131`) [read] | [ran] `answer_key_leaked_to_kb`: ok, exit 0. `inproc.py leak`: `read_doc` returns `[[100,"rejected","vendor_blocked"],…]`. [ran] accept crash |
| D4 | P1 | **The coding pack's checks are vacuous.** Its fixtures call `compose(gates={}, outcome=True, phi=phi)` for the oracle (`C/packs/coding/src/crucible_pack_coding/studio.py:89-91`). `compose` returns `gate·1 = 1.0` whenever `outcome` is True, whatever Φ is (`C/src/crucible/credit/compose.py:69-71`). With Φ forced to 0 for every patch, all 8 cases still pass. The pack's real grader (the GPU kit verifiers) is never invoked. Semantic cannot run at all: there is no `hooks.tasks` (`C/packs/coding/pack.yaml:30-32`), so `--allow gpu` gives `warn semantic.tasks`. The module doc says "semantic needs `--allow gpu` on the lab" (`C/docs/modules/M14-packs.md` table) | [ran] Φ≡0 gives `fixture.cases pass 8 case(s)`. [ran] `--allow gpu` gives `warn semantic.tasks` |
| D5 | P1 | **Checking a pack can validate another directory's code.** `resolve` imports by module name (`SP:98-107`) and never checks that the module file lies under `pack_dir`. In one process, after the real `finance_ap` was checked, a broken copy whose grader always returns 1.0 (which fails when checked alone) **passes**: its `studio.__file__` is the original's. This hits `pack check --all` and `tests/unit/test_packs.py::test_all_packs_fixture_level`. It is the P26 "package it can shadow" pattern (`C/DESIGN.md:109`) | [ran] `inproc.py shadow`: `real: True / broken copy: True`, module file in `packs/finance_ap` |
| D6 | P1 | **The declared reward range and determinism are never enforced.** Rewards outside `reward.range` pass: I declared `[0, 0.5]` and gold scored 1.0 (no range check anywhere in `SP`). `determinism: seeded` is never re-run. A grader that skips the controls gate on 30% of calls passed 4 of 12 runs. NeMo-Gym requires a `determinism` fixture case on a fresh server when the pack declares `seeded`, and checks the range on every case (`NG/nemo_gym/verifier_fixture.py:190-195, 218-238`) | [ran] `reward_above_declared_range` ok. `flaky.txt`: `0 1 1 0 0 1 1 1 1 1 0 1` |
| D7 | P1 | **Several fixture and semantic assertions are vacuous.** An empty fixture list passes ("0 case(s) graded"). An empty task list passes ("0 task(s): gold reaches 1.0"). A `fails_gate` that names a gate the grader never reports passes, because `gates.get(fails) == "pass"` is False for a missing gate (`SP:263`). Combined with a gate that never fails, it still passes. The `generator` hook is only checked for being callable (`SP:192-200`), so no generated item is ever graded | [ran] `fixtures_empty`, `tasks_empty`, `fails_gate_typo_and_gate_never_fails`: all ok |
| D8 | P1 | **finance_ap's task distribution is shortcut-learnable.** Every seed has exactly one invoice of each of the 7 kinds (2000/2000 seeds; `FA/world.py:63`, n=7). No invoice ever triggers two rules (0 of 14,000), so "apply the rules in this order" (`FA/world.py:111`) is never exercised, and applying them in reverse scores 1.0. A policy that never reads the policy document (handle the 5 flag and receipt traps; of the two over-PO invoices, approve the smaller overage and hold the larger) **solves 100%**. A policy that memorized 2% solves **73.2%**, against the docstring's claim that it "fails" (`FA/world.py:14`). Holding everything above the PO scores 0.857 with no gate cost | [ran] `fap_heuristics.txt` |
| D9 | P1 | **Packs are not distributable outside the checkout.** The hatch wheel ships only `src/<pkg>` (`C/packs/*/pyproject.toml` `[tool.hatch.build.targets.wheel]`). The built toy wheel has 0 `pack.yaml`, `worlds/` or `data/` files, and `PACK_ROOT = Path(__file__).parents[2]` (`C/packs/_toy/src/crucible_pack_toy/__init__.py:9`) points into site-packages. `find_pack` scans only the checkout's `packs/` (`SP:78-91`), and the registry's `pack` kind is registered but never read (grep). `finance_ap` is not installed (`crucible registry [pack]` lists only coding and toy). Every pyproject has `dependencies = []`, although packs import `crucible` and `yaml`. No core-version compatibility is declared; compare agentenv's `Requires-Dist` check on core, `AE/src/agent_env/plugins/_requirements.py:16-55` | [ran] `uv build --wheel --offline` of `_toy`; [ran] `crucible registry` |
| D10 | P1 | **The placeholder and licence checks are vacuous.** Only the literal `TODO` is searched, and only in 3 files (`SP:179-181`). The licence passes unless it is `""`, `unknown` or `todo` (`SP:182`), so `TBD`, `none`, `FIXME`, `XXX`, `<describe your pack here>` and the template's own "(edit this description)" all pass. The licence is not SPDX-checked and is not compared with pyproject (a GPL-3.0 pyproject with an Apache-2.0 pack.yaml passes). The scaffold presets `license: Apache-2.0` and `authors: []` (`C/packs/_template/pack.yaml:6-7`), so an unedited `pack new demo` passes every level. `finance_ap`'s pyproject still carries the template description "A Crucible domain pack" (`C/packs/finance_ap/pyproject.toml:4`). NeMo-Gym scaffolds `licensing: unknown` and `authors: [TODO]` and blocks publication on both (`NG/nemo_gym/environment/scaffold.py:255-276`; `publication.py:38-58`) | [ran] `license_TBD`, `license_none`, `license_mismatch_pyproject_GPL`, `placeholder_FIXME_and_template_text`, `placeholder_TODO_in_src_only`: all ok. [ran] `demo` passes semantic |
| D11 | P1 | **No pack can be exported to another ecosystem as a package.** `crucible export` has targets `canonical`, `atif`, `nemo-gym` and `openenv` (`C/src/crucible/studio/export.py:18`). DESIGN promises `export verifiers` and `export harbor` (`C/DESIGN.md:793-796`). The nemo-gym target writes a launcher with absolute checkout paths, with no `manifest.yaml`, config or JSONL rows (M11 D16 still open). The openenv target exports **only the first task** (`Environment = make_environment(lambda: _TASKS()[0][0])`, `export.py:58-60`), with absolute paths and no `openenv.yaml`, pyproject or Dockerfile. Both export only `hooks.tasks` anchors, which for `finance_ap` are 2 fixed worlds (`FA/studio.py:88-89`). The generator is never exported | [ran] `exp_openenv/openenv_env.py`, `exp_nemo/` |
| D12 | P2 | **A hook that raises crashes `pack check`.** It prints a traceback, exits 1 (the module docstring promises 3, `SP:21-22`) and writes no JSON report, so `--json` consumers get nothing. Manifest identity is not validated: `name: 'Finance AP!! v2'` and `version: banana` pass (the `NAME` regex applies only to `pack new`, `SP:47,112`). A hook may point outside the pack (`stages: os:getcwd` passes). A wrong pyproject entry point passes (`static.entry_point` resolves pack.yaml's copy only) | [ran] `fixtures_hook_raises`, `tasks_hook_raises`, `grade_raises`, `bad_name_version`, `hook_points_outside_pack`, `pyproject_entry_point_wrong` |
| D13 | P2 | **Approve-everything earns progress credit on the bench.** It scores 0 but has progress AUC **0.154**, about 34% of the oracle's 0.455, and `phi_final` 0.286. Φ (`FA/studio.py:38-40`) is gate-blind. M6 fixed the trainer path (`C/docs/audit/2026-10-12/M06-credit.md:35`), but the bench progress view still reports it. Cross-ref M13 | [ran] `bench.json` (3 fresh items + 2 anchors) |
| D14 | P2 | **Several ARCHITECTURE M14 promises are unbuilt.** `pack.yaml` should declare "Apps, adapters, connectors, signal mapper, outcome labeler, milestone library, verifier patterns, generators, horizon ladder, rubric seeds" (`C/ARCHITECTURE.md:751`); the schema has hooks and ingest only (`PM:63-90`). `template_version` is recorded (`PM:84`), but no `pack upgrade` exists; compare Karotte's 3-way `git merge-file` update (`K/docs/environments/updating.md:20-35`). `PM:3` claims NeMo-Gym's shape "with an exported JSON Schema", but no CLI emits it; compare `harbor task schema` (`H/src/harbor/cli/tasks.py:61`) and `NG manifest_json_schema()` (`NG/nemo_gym/environment/manifest.py:386`). The toy pack has no README (warn only) | [read]; [ran] toy static warn |

**Net.** The four packs pass their own checks [ran]: toy, coding and finance_ap pass at every level, and `_template` fails static by design. But `pack check` cannot fail several graders that are known to be bad: NaN, lenient, flaky, out-of-range, reason-blind and shadowed. It also has no leak level, and the coding pack's checks prove nothing about its grader. In the ecosystem, the strongest working fixture contract is NeMo-Gym's: required kinds, range check, and a determinism re-run. The strongest quality ladder is Repo2RLEnv's: controls, then probes, then blind rollouts. The strongest adversarial testing is Karotte's fake-model hack replays. Crucible borrowed the shape of all three and dropped the parts that do the failing.

### Picks (detail in §5)

| # | Pick | Kind | Licence | Effort | Fixes |
|---|---|---|---|---|---|
| 1 | NeMo-Gym verifier-fixture contract in `check_fixture`: kinds `full`/`zero`/`malformed`/`determinism`; `lower ≤ r ≤ upper` on every case (fails NaN); a determinism case re-graded on a fresh World when `seeded`; ≥1 case of each required kind | PORT | Apache-2.0 | S | D1 D6 D7 |
| 2 | Adversary fixtures: pack `policies` tagged `expect_max` (approve-all, hold-all, no-read) run at semantic level, and must score ≤ the bound on anchors **and** on N generator seeds. Karotte fake-model hack replay; Repo2RLEnv "plausibly wrong solution" probe; PortSim "lazy policies stay low" | PATTERN | MIT / Apache-2.0 | S | D2 D7 D8 |
| 3 | `pack check --level accept`: run the M9 contract (A5 leak, A3 mutation) on `hooks.tasks` plus sampled `generator` items, and make `crucible accept` work without `hooks.forge` | DEPEND (own M9) | — | M | D3 D4 |
| 4 | Publication gate (`pack check --publish`): NeMo-Gym `publication.py` placeholders, `licensing: unknown` scaffold, SPDX via `canonicalize_license_expression`; pack.yaml ↔ pyproject agreement (name, version, licence, entry point); `NAME` regex on the manifest; hooks must resolve inside `src/` | PORT | Apache-2.0 | S | D10 D12 |
| 5 | Check isolation: each pack checked in its own subprocess (or assert `module.__file__` under `pack_dir/src`); hook exceptions become `fail` rows with exit 3 | PATTERN (P26) | — | S | D5 D12 |
| 6 | Distributable packs: hatch `force-include` of `pack.yaml`, `worlds/`, `data/`; `PACK_ROOT` via `importlib.resources`; `find_pack` falls back to the registry `pack` kind; `dependencies = ["crucible>=0.1,<0.2", ...]` plus agentenv's Requires-Dist core check | PATTERN | Apache-2.0 | S | D9 |
| 7 | Content-addressed pack versions: Harbor's packager hash (`H/src/harbor/publisher/packager.py:23-95`) over pack.yaml, src and data, recorded in bench and accept outputs; Karotte 3-way template update for `template_version` | PATTERN | Apache-2.0 / MIT | M | D14, reproducibility |
| 8 | Packaging exports: NeMo-Gym `manifest.yaml` (EnvironmentManifest) plus config plus rows; `openenv.yaml` (OE `NormalizedManifest`) plus every task plus the generator; Harbor `dataset.toml` with digests for container packs (coding). Not one task, not absolute paths | DEPEND | Apache-2.0 / BSD-3 | M–L | D11 |
| 9 | `finance_ap` v2 generator, PortSim-style: draw `n` and kinds with replacement, compound cases (rule order), duplicate near-misses (same number from another vendor; earlier one rejected), tolerance-boundary cases, multiple partial receipts, an infeasible or refusal case, over-hold cost; a difficulty filter so naive or lazy policies stay below a floor (`FE/07-simulation-environments/portsim-v1/DESIGN.md:44-49, 89-96`) | PATTERN | Apache-2.0 | M | D8, §3 |

---

## 1. Requirement ledger

Status is about Crucible. "Ecosystem" lists where the requirement comes from.

| Requirement | Ecosystem (file:line) | Crucible (file:line) | Status |
|---|---|---|---|
| Identity: name pattern, semver-ish version, authors | NG `manifest.py:226-243,268-273` (authors ≥1, unique); H `config.py:286-333` (`org/name` regex); PE `tests/test_envs.py:33-46` (name, version, description, tags, placeholders rejected) | `PM:76-80` free strings; `NAME` regex only in `pack new` (`SP:47,112`) | **partial** (D12) |
| Licence, SPDX-valid, not a placeholder | NG `manifest.py:239-242,275-285` (SPDX canonicalize), `publication.py:42-58` (blocks `unknown`); R2E `docs/guides/publishing.mdx` (LICENSES.md); H and OE have none | `PM:79`; `SP:182` (rejects `""`/`unknown`/`todo` only) | **partial / vacuous** (D10) |
| Dataset provenance and licence per data source | NG `config_types.py:456-486` (`license` Literal, `source` gitlab/hf); `manifest.py:143-146,265` (`adopted_from`); R2E `tasks.mdx:67-104` (`[metadata.repo2env]` lineage, bundle hash); FE `data.mdx:39-46` | `IngestSpec` (`PM:28-50`) has no licence, provenance or revision field | **missing** |
| Version pins of upstream data and images | RE `AGENTS.md:35` (7-char rev), `enterprise_ops_gym/taskset.py:28,42-72` (dataset SHA, image digests); FE `05-multi-harness-rl/VALIDATION.md:11,19` | coding ingest points at `../../../migration_tasks` with no rev (`C/packs/coding/pack.yaml:17-27`); no pin field | **missing** |
| Reward range declared | NG `manifest.py:100-108`; OE `validation/manifest.py:39-72` | `PM:53-55,94-96` | done |
| Reward range **enforced** (and finite) | NG `verifier_fixture.py:218-221`; XV `environment.py:577-578`; H `verifier.py:67-121` (finite) | none in `SP` | **WRONG** (D1, D6) |
| Determinism declared and tested | NG `verifier_fixture.py:190-195,235-238` (fresh-server re-run); OE `rfcs/008…:190` | `PM:86` declared; never tested | **partial** (D6) |
| Resource and requirement declaration | H `config.py:422-552` (cpus, memory, gpus, network mode); OE `manifest.py:139-189` | `PM:87` `requires` (docker, gpu, network, paid); skip semantic when unmet (`SP:229-241,311-314`) | partial: no amounts, no network policy |
| Oracle reaches the top | R2E `README.md:141`, `quality.mdx:43-47`; ART `hf-repo2rlenv…md:58`; V `cli/validate.py:242-244` | `SP:288` gold ≥ hi | done, but NaN passes (D1); coding never runs (D4) |
| No-op does not | same; V `validate.py:51-53,247-260` | `SP:290` | done, but NaN or None passes |
| Designed negatives graded exactly | NG `verifier_fixture.py` (zero_reward + malformed required) | `SP:244-265` | **partial**: hook optional (warn), empty list passes, self-graded `grade` callables trusted (D2, D7) |
| Adversarial and red-team policies score 0 | K `pm-karotte-technical-deep-dive.md:214-235` (fake-model replays); R2E `quality.mdx:81` (probes); FE `portsim-v1/DESIGN.md:89-96` (hostile plans, lazy policies) | `policies` hook exists (`FA/studio.py:113-123`), never used by check | **missing** (D2) |
| Flakiness reruns | NG determinism case; SS `harness/repair.py:138-201` (post-hoc gold re-run) | none | **missing** (D6) |
| Leak checks (answers, hidden files, canaries) | H `default-rubric.toml` `tests_or_solution_in_image`, canaries `tasks.py:81-86`; ART `hf-repo2rlenv…md:22-44`; K `deep-dive.md:52-60` | none in pack check; M9 A5 exists (`C/src/crucible/accept/contract.py:176-188`) but is unreachable for `finance_ap` | **missing** (D3) |
| Grader isolation | MO / XV `environment.py:62-64,85-94,233-249` (verifier only in a sidecar, at reward time); K `deep-dive.md:101-111` | check runs pack code in-process; module-name shadowing (D5) | **WRONG** (D5) |
| Schema validation and an exported schema | H `harbor task schema`; NG `manifest_json_schema()` :386; OE `manifest.py:271-386` | pydantic `PackManifest`; no exported schema; `extra` policy inherits `CrucibleModel` | partial (D14) |
| Task cards and README fields | PE `AGENTS.md:66-70` (setup, deps, rubric, knobs, usage); ART `hf-blog…md:137-148,191` (card tags, run command, reward rule); NG `scaffold.py:567-582` (data links, licensing) | README existence only, and a warn (`SP:185`); toy has none | **partial** |
| Eval outputs shipped with the env | PE `.github/pull_request_template.md:12` (≥5×3 rollouts, some reward >0); NG `cli/env.py:972-1037` (5 example rows, metrics, rollouts) | none | missing (P2) |
| CI per environment | PE `tests.yml:32-62` (changed envs: fresh venv, import); RE `tests.yaml:44-45` (one-rollout eval); NG `unit-tests.yml:119`; AE `clean-install.yml:32-33` (wheel clean install) | `tests/unit/test_packs.py:41-51` (fixture level, in one process: D5) | partial |
| Content-hash versioning | H `packager.py:23-95`, `manifest.py:245-262`; PE `env-sync.yml:98-101` | none | **missing** (pick 7) |
| Template upgrade | K `docs/environments/updating.md:20-35` | `template_version` recorded only (`PM:84`) | missing (D14) |
| Installable and distributable | V `loaders.py:71-126` (import by id); OE `cli/_cli_utils.py:30-75` (push needs openenv.yaml, README, Dockerfile); AE `bundle/installed.py:31`; H `publisher.py:173-350` | `pyproject` entry point; wheel drops pack.yaml and data; `find_pack` checkout-only | **WRONG** (D9) |
| Instructions match what the grader checks | ART `pm-building…md:15-21`; RE `AGENTS.md:113-115` | M9 A7 exists; not run for packs | missing (via D3) |

---

## 2. `pack check` probes [ran]

**The four shipped packs** (`cli_all.txt`). Results:
- `_template`: static fails, because `template_version: '__TEMPLATE_VERSION__'` is not an int. This is expected, and `--all` excludes it.
- `toy`: ok at all levels. `static.readme` warns because there is no README. 8 fixtures; semantic over 1 task.
- `coding`: ok. Semantic is skipped for gpu; with `--allow gpu` it gives `warn semantic.tasks` (D4).
- `finance_ap`: ok at all levels, with 9 fixtures and 2 semantic tasks.
- `pack new demo`: ok at every level, unedited (D10).

**Broken copies of finance_ap** (`broken_packs.py`). Each was checked with the real CLI at `--level semantic --json`, with only `crucible/src` on PYTHONPATH.

| Variant | Should | Got | Caught? |
|---|---|---|---|
| control (unchanged) | pass | ok | — |
| grader always 1.0 | fail | `fixture.cases` + `semantic.noop` fail | **yes** |
| grader always NaN | fail | ok, exit 0 | **no** (D1) |
| grader lenient (anything decided = 1), no fixtures hook | fail | ok, `fixture.hook` warn, exit 0 | **no** (D2) |
| grader lenient + constant `grade` fixtures | fail | ok | **no** (D2) |
| grader ignores the reject reason code | fail | ok | **no** (D2c) |
| gold above declared range ([0, 0.5]) | fail | ok | **no** (D6) |
| gate skipped 30% of calls (flaky) | fail | passes 4/12 | **no** (D6) |
| fixtures `[]` | fail | ok | **no** (D7) |
| tasks `[]` | fail | ok | **no** (D7) |
| `fails_gate` typo + gate never fails | fail | ok | **no** (D7) |
| answer key in the KB | fail | ok | **no** (D3) |
| licence `TBD` / `none` | fail | ok | **no** (D10) |
| licence key removed | fail | `static.manifest` fail | yes |
| pyproject GPL-3.0 vs pack Apache-2.0 | fail | ok | **no** (D10) |
| `FIXME` + template description + `XXX` README | fail | ok | **no** (D10) |
| `TODO` only in `src/` | fail | ok | **no** (D10) |
| name `Finance AP!! v2`, version `banana` | fail | ok | **no** (D12) |
| pyproject entry point → missing module | fail | ok | **no** (D12) |
| hook `stages: os:getcwd` | fail | ok | **no** (D12) |
| fixtures, tasks or grade raise | fail with report | traceback, exit 1, no JSON | partial (D12) |
| README removed | warn | warn | yes (warn by policy) |

Of the 21 variants that should fail outright, **4 are caught** (always-1, missing licence key, and the three raising hooks, which crash rather than report). The README variant warns as its policy says.

---

## 3. finance_ap against EnterpriseOps-Gym and Karotte's criteria

| Criterion | EOG / articles | finance_ap | Verdict |
|---|---|---|---|
| Scale and realism | 1,150 expert-curated tasks, 8 domains, 512 tools, 164 tables, 9.15 steps on average, up to 34 (`EOG/README.md:41-50`); seed DBs per domain (`gym_dbs.zip`, 76 snapshots); EOG has no AP or procurement domain (grep) | 1 schema with 4 tables, 9 tools, 7 invoices, about 36 oracle calls (`FA/world.py:118-121`, `ap_tools.py`) | thin, but a real control set; it is the only AP env locally |
| Policy coverage | ~11k-char policy per domain with roles, state transitions, enum rejection and ownership (`EOG data/revised/csm/task_…json:20`); infeasible subset where the best model refuses correctly 53.9% (`EOG/docs/index.html:176,222-225`) | 6 rules (`FA/world.py:95-113`), no roles, no permissions, no refusal case. Rule order is stated but never exercised (D8) | **gap** |
| Grader strictness | hidden SQL `database_state` verifiers, all must pass (`EOG/benchmark/executor.py:371`); partial pass rate reported (:608) | controls gate plus exact (status, reason) share (`FA/studio.py:63-72`). Strict on approvals. Only one wrong-reason negative, and a reason-blind grader passes check (D2c) | strict grader, thin fixtures |
| Hackability | verifiers hidden and out-of-band (`EOG/benchmark/verifier.py:210-212`); DB reset per run; 40% held out (`README.md:302`); no adversarial testing. Karotte: "the agent… will eventually try every possible approach" (`ART/pm-karotte-technical-deep-dive.md:19`); hack replays must score 0 (:214-235) | approve-everything scores 0 [ran: bench `mean_reward 0.0`, controls pass rate 0] ✅, but earns progress AUC 0.154 and Φ 0.286 [ran] (D13). A structural exploit solves 100% without reading the policy (D8). The policy tools cannot write outside pending invoices (`ap_tools.py:70-78`) ✅. No raw SQL tool ✅ | gate holds; distribution is exploitable |
| Distribution shift between seeds | distractor-tool modes `plus_5/10/15_tools` (`EOG/README.md:185`); hybrid cross-domain tasks | only amounts, permutation, names and tolerance ∈ {1,2,3,5}% change (`FA/world.py:62-64`). The kind multiset is constant across 2000/2000 seeds | **gap** (D8) |
| Over-caution cost | Karotte: "give the AI an honest way to give up" (`ART/pm-building…md:39`); grader must check everything the prompt asks (:15-21) | holding a clean invoice costs only 1/7. Hold-all scores 0.143, more than approve-all's 0 [ran] | acceptable, but no SLA or discount penalty |
| Determinism | EOG seeded DB, fresh per run (`benchmark/mcp_client.py:38-39`) | `random.Random(f"finance_ap/{seed}")`. Identical across `PYTHONHASHSEED` 1 and 2 [ran: `99b28b8f83815c21` twice] ✅ | done (but never checked by pack check, D6) |
| Packaging | HF dataset + Docker images on a personal account, mutable `:latest` (`EOG/README.md:124`); Apache-2.0; RE's port pins the dataset SHA and image digests (`RE/environments/tool_use/enterprise_ops_gym/taskset.py:28,42-72`) | not installable (D9), not exportable except as 2-task launchers (D11), no M9 acceptance (D3) | **gap** |

**Already in Crucible:** `C/src/crucible/interop/eog/importer.py` imports EOG tasks as World tasks (M13 campaign). An EOG-derived AP or procurement extension is a cheap way to widen finance_ap's policy coverage (pick 9).

---

## 4. Interop: packaging and distribution (runtime paths were audited in M11)

| Direction | Harbor | verifiers | NeMo-Gym | OpenEnv |
|---|---|---|---|---|
| **Pack → foreign package** | none. World tasks are not Harbor dirs (`C/DESIGN.md:1419` finding). DESIGN still lists `export harbor` (:796). A container pack (coding) could emit `task.toml` + `dataset.toml` with `sha256` digests (`H/src/harbor/models/dataset/manifest.py:24-53`) | none. `vf_taskset.py:68` reads only Harbor dirs. v1 needs a package with `__all__ = [Taskset]` (`V/verifiers/v1/utils/loaders.py:99-126`; `V/skills/create-environments/SKILL.md:52-56`) | launcher only, with absolute paths and 2 anchor tasks; no `manifest.yaml` / `config.yaml` / `data/example.jsonl` (5 rows required, `NG/nemo_gym/cli/env.py:972-978`) [ran] | launcher exposing **task 0 only**, absolute paths; no `openenv.yaml`, `pyproject` (`server` script), `server/Dockerfile` or README, all of which `openenv push` requires (`OE/src/openenv/cli/_cli_utils.py:30-75`) [ran] |
| **Foreign package → pack** | `HarborInPlaceTask` etc. run Harbor dirs, but nothing writes a `pack.yaml`; `registry.json` / `dataset.toml` are never read (M11 G5 R2) | `v1_import` reads `vf-eval` traces only | `import_server` wraps a running server as an App; no manifest read | `import_env` and AWM catalog; no `openenv.yaml` read |
| **Versioned distribution** | Harbor Hub content hash plus `org/name@ref` (`H/src/harbor/models/package/reference.py:54-72`) | `prime env push`, hub `owner/name@version` (`V/skills/create-environments/SKILL.md:244-252`) | `ng_upload_dataset_to_hf`, private repo + card tags (`NG/nemo_gym/hf_utils.py:130-181`) | HF Space, `-<openenv-version>` suffix (`OE/scripts/prepare_hf_deployment.sh`) | 

Crucible has: no `pack publish`, no pack digest, no registry lookup by name (D9), and no dataset card. The packaging contract a foreign runtime could check is therefore missing in both directions. The ecosystem's own validators are also weaker than they look, so they are not a reason to copy them wholesale:
- OpenEnv's `run_validation` runs only the static grader (`OE/src/openenv/validation/runner.py:120-122,152`), even though severity-v2 lists 46 checks.
- `harbor check` is an LLM rubric that never runs the oracle (`H/src/harbor/analyze/checker.py:1-8,144-207`).
- NeMo-Gym CI never runs `gym env validate` or `publish` over its catalog, and 84 of its 174 manifests say `licensing: unknown`.

---

## 5. Picks: detail

1. **PORT NeMo-Gym verifier fixtures** (`NG/nemo_gym/verifier_fixture.py:186-240`, Apache-2.0, S).
   - In `check_fixture`, every case must satisfy `math.isfinite(r) and lo <= r <= hi`.
   - Require at least one gold or full case at `hi` and one zero case at `lo`.
   - If `determinism == "seeded"`, re-grade one case on a freshly opened World and require identical reward and gates.
   - Make a missing `hooks.fixtures` a **fail** for `kind: domain`.
   - A `fails_gate` must name a gate the report actually contains.
   - Refuse an empty case list.
   - Gate: the `always_nan`, `reward_above_declared_range`, `fixtures_empty`, `fails_gate_typo…` and `flaky_grader` variants fail (the last one in at least 9/10 runs, via 3 re-grades).
2. **PATTERN adversary fixtures** (Karotte fake model `ART/pm-karotte-technical-deep-dive.md:214-235`, MIT; R2E `docs/concepts/quality.mdx:81`; FE PortSim `DESIGN.md:89-96`; S).
   - `PolicyRef` gains `expect_max`.
   - The semantic level runs every bounded policy on the anchors and on k generator seeds.
   - finance_ap declares `approve_all ≤ 0`, `hold_all ≤ 0.2` and a `no_policy_read ≤ 0.9` heuristic.
   - Gate: the `lenient_*` and `grader_ignores_reject_reason` variants fail. The last one needs a `reject_with_wrong_reason` negative, which the pack should add.
3. **DEPEND on M9** (M).
   - `pack check --level accept` builds `AcceptTask`s from `hooks.tasks` and `generator` samples and runs the model-free contract (A5 leak and A3 mutation included).
   - `crucible accept` stops assuming `hooks.forge` (`C/src/crucible/cli/studio.py:195`).
   - Gate: `answer_key_leaked_to_kb` is rejected, and `crucible accept finance_ap` returns a verdict.
4. **PORT the NeMo-Gym publication gate** (`NG/nemo_gym/environment/publication.py:38-99`, `manifest.py:275-285`, Apache-2.0, S).
   - The scaffold writes `license: unknown` and `authors: [TODO]`.
   - `pack check --publish` (or `--level publish`) blocks placeholders: case-insensitive `todo|tbd|fixme|xxx|<…>` plus the template's own sentences, in pack.yaml, README, pyproject and `src/`.
   - The licence must be SPDX, internal or proprietary, and must equal pyproject's.
   - The name must match `NAME`.
   - The pyproject `crucible.plugins` entry must equal `entry_point`.
   - Hooks must resolve under `src/`.
   - Dev `pack check` stays lenient, so the `pack new demo` gate still holds.
5. **PATTERN check isolation** (P26, S).
   - Run each pack's check in a subprocess, which is what `--all` and the test should do.
   - Assert `inspect.getfile(module)` is under `pack_dir/src`.
   - Catch hook exceptions into `fail` rows and exit 3.
6. **PATTERN distributable packs** (S).
   - Use hatch `[tool.hatch.build.targets.wheel.force-include]` for `pack.yaml`, `worlds/`, `data/` and `README.md` under the package.
   - Compute `PACK_ROOT` with `importlib.resources.files(pkg)`.
   - `find_pack` should consult `get_registry().get("pack", name)`.
   - Declare `crucible` and `pyyaml` with bounds.
   - Port agentenv's `Requires-Dist` core-compat check (`AE/src/agent_env/plugins/_requirements.py:16-55`, Apache-2.0).
   - CI: a clean-install wheel smoke, like `AE/.github/workflows/clean-install.yml:32-33`.
7. **PATTERN content-hash versions** (`H/src/harbor/publisher/packager.py:23-95`, Apache-2.0, M).
   - `pack digest` = sha256 over sorted `relpath\0filehash` of the pack.
   - Bench, accept and train manifests record it.
   - `pack upgrade` uses Karotte's baseline/target 3-way merge (`K/docs/environments/updating.md:20-35`, MIT).
8. **DEPEND exports** (M–L).
   - NeMo-Gym: `manifest.yaml` validated by `EnvironmentManifest`, plus `config.yaml` and 5 example rows (M11 pick 4).
   - OpenEnv: `openenv.yaml` (`validation:` block with reward range, oracle, resources and network), an env over **all** tasks plus a generator-backed reset, pyproject and Dockerfile.
   - Harbor `dataset.toml` for container packs. A verifiers v1 package exporting `__all__ = ["CrucibleTaskset"]`.
   - All of them self-contained (copied src, no absolute paths).
9. **PATTERN finance_ap v2 generator** (FE PortSim `DESIGN.md:38-49,89-96`, Apache-2.0, M).
   - Kinds drawn with replacement, `n ∈ [5, 25]`.
   - Compound invoices that fire two or more rules.
   - Duplicate near-misses: the same number from another vendor, or a prior invoice that was rejected rather than approved.
   - Overages at tol ± 0.1%, multiple partial receipts, and one infeasible or refusal case.
   - A difficulty filter that keeps a world only if the heuristics in `fap_heuristics.py` stay below a floor.
   - Fixtures for each new trap.

---

## 6. Probe index (scratchpad `…/scratchpad/m14/g2/`)

| File | What |
|---|---|
| `cli_all.txt` | real CLI, 4 packs × 3 levels |
| `broken_packs.py`, `broken_packs.json`, `variants/` | 26 broken finance_ap copies × real CLI `--level semantic --json` |
| `flaky.txt`, `variants/flaky_grader` | 12 runs of a 30%-gate-skip grader: 4 pass |
| `inproc.py`, `inproc.txt` | `leak` (agent reads the answer key), `lenient` (approve-all = 1.0), `shadow` (broken copy passes after the real pack) |
| `fap_heuristics.py`, `fap_heuristics.txt` | 2,000 seeds: kind multiset, rule overlap, 8 heuristic policies |
| `bench.json`, `runs/` | `crucible bench run finance_ap --fresh 3 --k 1`: approve-all AUC 0.154, Φ 0.286, reward 0 |
| `exp_openenv/`, `exp_nemo/` | `crucible export --to openenv|nemo-gym --pack finance_ap` |
| `wheel/` | `uv build --wheel --offline` of `_toy`: no pack.yaml, worlds or data |
| `demo/` | `pack new demo`, unedited, passes semantic |
| `accept_fap/` | `crucible accept finance_ap`: AttributeError |
