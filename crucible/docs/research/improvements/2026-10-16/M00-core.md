# M00 core: audit against the lab RL repos (2026-10-16)

Crucible HEAD `da8146b` (branch `uzair`, after R-1..R-31 of 2026-10-14). Citations: Crucible source relative to
`crucible/src/crucible/`, other Crucible files to `crucible/`, reference code as `<folder>/<path>:line`; a repo's own §5
entry drops the folder, and after its first path the package prefix. Commits in §4. **[ran: x]** = reproduced this run
by `probes/x`. Notes, probes, full fields (`verify-N.md`):
`/Users/uzair/Desktop/RL env repos/.module-audit/2026-10-16/M00-core/`.

## 0. Summary

- **Where it matters.** Every module and pack shares M0's shapes and identities: run records, locks and
  launches (`runtime/lock.py:29-170`), the store and its tags, World snapshots, the registry, and `crucible.toml` with
  its secrets.
- **Ledger.** 145 items: 48 done, 42 partly done, 52 not done, 3 superseded. Of the 31 A14 recommendations, 18 hold
  and 13 are partly done. Four "done" claims are incomplete: AUD-D4, AUD-D11, A14-R22 and A14-R24. run/v2 regressed
  AUD-D8 and now refuses pre-versioning records (R-7).
- **Gaps.**
  - Credentials: a cloned repo's `crucible.toml` sends the AWS secret or the operator's key to its own host, or lifts
    the spend cap (R-1). `request_defaults` re-bills prompts and writes literal keys into `lock.json` (R-2). Another
    user's `/tmp/.env` supplies keys (R-5). A refused key is printed in full (R-6). The default store is mode `0755` (R-4).
  - Integrity: a short write indexes 1,048,576 of 2,097,152 bytes under the full key (R-3). NaN, sets and colliding
    keys in 50 `Any` fields are lost or get per-process ids (R-15).
  - Identity: the fingerprint changes after `register_secret` and collides on dotted keys (R-14). Launches do not record
    plugins (R-12).
  - Process: four kinds of change that move bytes or ids pass CI (R-9, R-10, R-13). Two distributions sharing one
    entry-point name silently drop a pack (R-8). Two MIT ports ship without notices (R-11).
- **Top recommendations.** R-1 (impact H): send credentials only to endpoints an operator paired them with. R-2: keep
  credentials out of provider settings. R-3: make blob writes complete or fail. R-4..R-11 have impact M and low effort.
- **Coverage.** 33 reference folders: 9 primary, 15 secondary, 9 not relevant. No cluster failed; every candidate was
  verified. About 80 candidates became 34 verified items, merged here into 31; 9 were rejected outright and 5 in part.

## 1. What the module does

M0 holds the shapes and plumbing every module relies on, with no domain logic and no framework imports: typed documents
with strict parsing and versioned reads (World, task, run record, lock and launch, rollout, WorldRef, usage); canonical
JSON with scheme-tagged content ids, tree digests and settings fingerprints; build identity; a lazy registry fed by
`crucible.plugins` entry points; layered `crucible.toml` with `env:NAME` secrets and a scrubber; a content-addressed
store with tags; the failure vocabulary, with blame and masking. Size: 5,746 lines in 33 files, plus the vendored `.env`
parser; 97 models.

| | |
|---|---|
| Inputs | dicts, JSON, YAML, TOML layers, `--set`, `.env`, `CRUCIBLE_*`, bytes and files, entry points, task dirs, git metadata |
| Outputs | validated or upcast models; canonical JSON, content ids, `sha256:` digests, fingerprints; store keys, tag revisions; `check()` reports; redacted `Config` |
| Called by | world 40 files, interop 26, train 19, runtime 18, workflow 11, actors 11, nine other modules 48, packs 43 |
| Layer | bottom of the 14-layer import-linter contract (`pyproject.toml:165-183`) |
| Principles | P12, P13, P16 (infra masked), P17, P18, P19, P21; ARCHITECTURE §7 rules 2, 4, 5 |

Gate: `ARCHITECTURE.md:37` says the gate (`:438-442`) passed. 22 M0 suites: 233 passed, 4 skipped, 1 failed
(`tests/unit/test_capture.py:73` lacks a torch skip). 18 formats round-trip and 97 models export a schema with warnings
as errors **[ran: p02_gate]**; `lint-imports` 4 kept, 0 broken; seeds not run (Harbor absent).

| Consumer | Uses it? | Evidence |
|---|---|---|
| Run lock, launch, journal (M10) | fingerprint, `BuildInfo`, redacted config, `child_path` | `runtime/lock.py:99-170`; `runtime/journal.py:74` |
| TraceStore (M3) | tag CAS; SQL on `store._db` | `connectors/trace_store.py:73-124` |
| Training identity (M12) | the same `fingerprint.project` | `train/identity.py:23,83,102` |
| World (M1) | `content_hash` (only production consumer), WorldRef | `world/world.py:196,968-989` |
| Bench (M13) | reads RunRecords, drops unreadable ones | `bench/metrics.py:217-228` |
| Store backends, `schemas/`, `task_digest`, `started_at`, reasoning totals | no consumer yet (grep), which lowers R-13, R-21, R-25, R-26, R-28 | `core/store/__init__.py:562-568` |

## 2. Status ledger

| Items | State | Evidence |
|---|---|---|
| A14-R4, A14-R5, A14-R7, A14-R12, A14-R15, A14-R20, A14-R25, A14-R26, A14-R27, A14-R28, A14-R29, A14-R30, A14-R31 | done | Built and tested: `child_path` (`core/ids.py:74-87`), value scrub, strict loader, index migrations, knobs, lossless docs, dir fsync, MIME, kinds, dotenv; dead code removed |
| A14-R16, A14-R17, A14-R21 | done, residual | Tag CAS (`core/store/__init__.py:476-524`; R-19); `capture` gate, one test without a skip (LED16-6); scheme 1 unpinned (R-10) |
| A14-R22, A14-R24, AUD-D4, AUD-D11 | done, claim incomplete | Kit passes broken backends (R-28); forget unsynced (R-17); `put_file` keys source bytes (R-3); `_consistent` partial (R-27) |
| IMP-S1, IMP-S2, IDX-L1, AUD-D3 | done | 14-layer contract, 0 broken; CI runs it (`../.github/workflows/crucible.yml:42-52`) |
| IMP-M0-4, IMP-M0-6, AUD-D1, AUD-D2, AUD-D5, AUD-D6, AUD-D7, AUD-D9, AUD-D10, AUD-D12, AUD-D14, AUD-D15, AUD-D16, AUD-D19, AUD-D20, AUD-F4 | done | WAL (`core/store/__init__.py:99-109`), upcasters (`core/schema/base.py:108-201`), 2026-10-12 fixes tested (`tests/unit/test_core_audit.py`) |
| SCOUT-1, SCOUT-2, SCOUT-4, NEW-5, NEW-6, CHG-10, CHG-17, CHG-18 | done | Half-failed plugins withheld (`core/registry.py:172-179`); `redacted()` used (`runtime/lock.py:150`); decisions hold |
| A14-R1, IMP-M0-3 | partly | Lock and launch built (`runtime/lock.py:41-169`); undeclared policy settings share a fingerprint **[ran: p04_lock_opaque]**; no image digest |
| A14-R2 | partly | Task digest checked thrice (`runtime/run.py:200-225`); no `world_digest`; regrade unchecked (`:296`); R-26 |
| A14-R3, AUD-F2 | partly | Sent sampling typed (`core/schema/rollout.py:32-73`); verl bridge records none (`train/bridges/verl_agent_loop.py:148-151`) |
| A14-R6, A14-R8, A14-R10 | partly | Old roots open (R-4); `request_defaults` literals in the lock (R-2); `meta` takes `'\udcff'`, then dump fails (R-15) |
| A14-R9 | partly | Two MIT notices missing (R-11); wheel ships no `LICENSES/` (`pyproject.toml:78-79`) |
| A14-R11, AUD-F3 | partly | `BuildInfo` without plugins (`core/build.py:30-35`, R-12); `Provenance` params scalar (`core/prov.py:19-25`) |
| A14-R13, IMP-M0-5 | partly | Staging keyed by `ep.name` (`core/registry.py:165-179`, R-8); no strict mode or API version |
| A14-R14, AUD-F1 | partly | Gate skipped without the undeclared `packaging` (`core/registry.py:221-226`) |
| A14-R18, AUD-D8, AUD-D13, SCOUT-3, CHG-9, DOC-gate-1, DES-M0-1 | partly | Shape-only freeze (R-9, R-13); missing version read as current (R-7); 4 unversioned formats (`tests/unit/_formats.py:96-102`); defaults hashed |
| A14-R19, IMP-M1-8 | partly | Typed usage (`core/schema/usage.py:31-216`); start = end (R-21); no per-model split (R-20) |
| A14-R23, IMP-M0-7 | partly | Layers and `--set` (`core/config.py:198-303`); project bypasses (R-1); no `CRUCIBLE__*` layer |
| IMP-S3, IMP-M2-4, AUD-D17, AUD-D18, AUD-F5, AUD-F6, AUD-F7, RES-CAP-1, RES-FORGE-1 | partly | `prov` optional (`core/prov.py:16`); only `Turn.trainable`; free `attempt`; `Provenance` alias; local `get_store`; no content parts, judge pin, `unchanged_except` |
| DOC-open-1, DOC-open-3, ARCH-M0-3, CHG-14, OQ-2, OQ-3, OQ-5 | partly | AUD-F1..F7 1 done; task04 lacks `network_mode`; card contradicts `ARCHITECTURE.md:29`; eval rollouts lack `capture`; no outside `put_doc`; `exclude_if` exists |
| SCOUT16-5, CHG-11, LED16-1, LED16-2, CHG-2, LED16-3 | not done (R-1, R-2) | `models` not operator-only (`core/config.py:200,249`); a string budget passes (`:246`); `request_defaults` in the lock (`actors/llm.py:250-255`) **[ran: p03_ledger G, H]** |
| SCOUT16-1, SCOUT16-2, NEW-1, SCOUT16-3, NEW-3, SCOUT16-4, SCOUT16-6, SCOUT16-7 | not done (R-7, R-3, R-15, R-8, R-14, R-10) | p03_ledger A-C, F, I, J at `core/schema/base.py:183`, `core/store/__init__.py:324,335`, `core/fingerprint.py:48-55`; D4 test vacuous |
| SCOUT16-8, SCOUT16-9, SCOUT16-10, NEW-4, NEW-9, CHG-4 | not done (R-9, R-28, R-23, R-21, R-24, R-12) | `core/store/conformance.py:28-105`; `core/schema/rollout.py:279-282`; `runtime/run.py:329-331`; `core/schema/actions.py:46` |
| CHG-12, CHG-19 | not done (no recommendation) | No pack policy declares settings; `Provenance` lacks build info |
| CHG-1, CHG-3, CHG-5, CHG-6, CHG-7, CHG-8, CHG-13, CHG-15, CHG-16 | not done (deferrals hold) | Run dirs plain `mkdir`; wheel lacks notices; regrade unchecked; `RunRecord.world` an id (`runtime/run.py:330`); verl bridge; `packaging`; outside records lack usage; local store; checkpoint unfrozen |
| IMP-M0-2, ARCH-M0-2, IMP-M0-8, IMP-M1-9, ARCH-M0-1, RES-ACC-2, DOC-open-2 | not done | No `derived_from` (`connectors/trace_store.py:79`); 15 hard-coded pointers (`core/registry.py:281-339`); no `output_schema`; card stale; no accept kinds; task04 |
| SCOUT-5, HO-1, SCOUT-6, HO-2, HO-3, HO-4 | not done | No GC; 7 global tables without reset; `event_key` without NFC (`core/schema/trace.py:52-62`); `json.dumps(default=str)` (`runtime/keys.py:21`); no AdoptedFrom |
| LED16-4, LED16-5, LED16-6, LED16-7, NEW-7, NEW-8 | not done | `Store.kinds()` holds its lock across yields (`core/store/__init__.py:440-442`); failed-plugin side effects stay; torch skip; stale docs (`runtime/run.py:52`; `docs/modules/M00-core.md:3`) |
| OQ-1, OQ-4 | not done | Harbor tag RPC server-side (`harbor-framework-harbor/src/harbor/db/client.py:717-737`); provider audio formats unknown |
| IMP-M0-1, RES-ACC-1, NEW-2 | superseded | Layering met another way; `crucible.acceptance/v1`; attempts hashed (`runtime/keys.py:37-43`) |

## 3. Comparison questions

| # | Question | Axis |
|---|---|---|
| Q1 | How is a version carried, older or unmarked documents migrated, newer ones refused; one convention? | evolution and versioning |
| Q2 | How is a content id computed, does it name its scheme, and does it survive added fields and canonicalizer edits? | identity and reproducibility |
| Q3 | Are blob writes atomic and durable (short writes, fsync), reads verified, crash leftovers recovered? | correctness (storage) |
| Q4 | How are mutable names layered on content: CAS, revisions, history, GC that knows every holder? | evolution and versioning (refs) |
| Q5 | How are plugins merged: identity, collisions, partial failure, core gate, kinds, reporting? | extensibility |
| Q6 | How is config layered, which keys may a less-trusted layer not set, how are secret refs resolved? | operability (config) |
| Q7 | What does a run record say about its code, deps, plugins, config, digests, times; how is equivalence checked? | identity and reproducibility (runs) |
| Q8 | How are failures classified (blame, retry, masking), unknown kinds read, reward and outcome kept consistent? | correctness (failures) |
| Q9 | How are tools, media, usage, finish and sampling typed and mapped to providers; are names valid? | interop |
| Q10 | How are ids checked before paths, secrets kept out of dumps and logs, stores and run dirs protected? | security |
| Q11 | Is the store an interface with backends and a full conformance suite; are retention and listing bounded? | performance and scale |
| Q12 | What stops an unversioned format change (snapshots, fixtures, validators), and what escapes it? | evolution and versioning (format freeze) |

## 4. Repo triage

33 folders (9 primary, 15 secondary, 9 not relevant). Commit: the HEAD read in this run, [the earlier sweep's pin]
where it differs.

| Repo | Class | Reason | Commit | Licence |
|---|---|---|---|---|
| harbor-framework-harbor | primary | Typed task/trial/ATIF/lock contracts, digests, secret templating | dc933be7 [a2fafbc] | Apache-2.0 |
| agentenv-framework | primary | Store ABCs and kit, layered config, staged plugins; ported by Crucible | 015c89b [041afd7] | Apache-2.0 + NOTICE |
| primeintellect-ai-verifiers | primary | Versioned traces, task hashes, resume checks, client credentials | e22533430 [ae819de] | MIT |
| nvidia-nemo-gym | primary | Failure kinds, versioned records, store Protocols and kit, plugins | 7ece89985 [3ef478d] | Apache-2.0 |
| meta-pytorch-openenv | primary | Versioned manifests, strict JSON, discovery, redaction | 8d3455ab [49aa302] | BSD-3-Clause |
| huggingface-repo2rlenv | primary | Protocol-tagged artifacts, budget CAS, mode-aware bundle identity | 27d04ee [5555443] | Apache-2.0 |
| preferencemodel-karotte | primary | Distribution-identified plugins, protected store, typed usage | 31c693e | MIT |
| infra-nvidia-nemo-rl | primary | Fail-closed run identity (A14-R1's source), failure classifier | 925f70d55 [c9193bc] | Apache-2.0 |
| infra-rllm-org-rllm | primary | Migrated registries, store Protocol with one suite, entry points | 4e828c81 | Apache-2.0 |
| primeintellect-ai-prime-agent | secondary | Versioned sessions, global-only providers, durability classes | 3e0893260 [2ee7e623a] | MIT |
| primeintellect-ai-prime-environments | secondary | 109 environments; pins and an endpoints table only | cf220ee | Apache-2.0 |
| primeintellect-ai-research-environments | secondary | Commit-pinned registry, digest-pinned manifests | f63f07c8a [d4fb80a] | Apache-2.0 |
| adithya-s-k-fineenvs | secondary | Explorer app: store, pointers, retention, env filtering | 5e0b46e | Apache-2.0 |
| 08-hf-rl-env-datasets | secondary | 41 clones, 33,687 `task.toml`: version evidence | per clone | per dataset |
| swe-agent-mini-swe-agent | secondary | Ordered config layers, user-dir `.env` | 04d809ce | MIT |
| xiaomimimo-verl | secondary | Per-component run manifest | a2ad9f61 | Apache-2.0 |
| infra-volcengine-verl | secondary | Legacy layout recognised by shape | 2035f58d [5eb4826] | Apache-2.0 |
| infra-primeintellect-ai-prime-rl | secondary | Config package; removed keys rejected by name | 1e5318108 [f7e8ab6] | Apache-2.0 |
| infra-openrlhf-openrlhf | secondary | Retention by count and bytes; counter-examples | dc2a7ad | Apache-2.0 |
| infra-huggingface-trl | secondary | Installed-version capture, deprecation windows | d9b9f39b [72cf37d] | Apache-2.0 |
| infra-inclusionai-areal | secondary | Legacy pointer by marker, buffered fsynced writes | aedc21ec [298412a] | Apache-2.0 |
| infra-thudm-slime | secondary | Exact codec, pure resume identity, contract tests | 05e3b017 [c933787] | Apache-2.0 |
| infra-novasky-ai-skyrl | secondary | One hashing module, predicated-UPDATE CAS | a1e34dbf [3d0ae22] | Apache-2.0 |
| infra-alibaba-roll | secondary | Unframed hash (counter-example), closed backend list | 0f37fe13 [581046a] | Apache-2.0 |
| servicenow-enterpriseops-gym | not relevant | Plain dataclasses; no versions, hashes or registries | 271f2c3 | Apache-2.0 |
| hf-space-fineenvs-multi-harness-rl | not relevant | Article Space, no M0 code | not git | CC-BY-4.0 |
| hf-space-fineenvs-simulation-rl-environments | not relevant | Article Space plus a chart export script | not git [46bb2be] | CC-BY-4.0 |
| hf-space-rl-environments-guide | not relevant | Article Space, prose only | not git | CC-BY-4.0 |
| swe-bench-swe-smith | not relevant | Counter-examples only (registry overwrite, 8-char ids) | 9b74ac0 | MIT |
| veds12-git-swarm | not relevant | Static project page | 9f66af0 | none |
| karpathy-autoresearch | not relevant | Prompt and training script; crash scored 0.000000 | 228791f | none (README: MIT) |
| xiaomi-mimo-v2.6-rl-oss | not relevant | Data only (925 envs), unversioned; used as evidence | HF 639865fd | none (README: Apache-2.0) |
| infra-moonshotai-checkpoint-engine | not relevant | Weight transfer; persists nothing with a schema | 84631a2 | MIT |

## 5. Findings per repo

### harbor-framework-harbor
- Mechanisms: inputs locked by digest, compared in serialized form (`src/harbor/models/job/lock.py:124-323`); consent for
  host variables (`cli/jobs.py:130-262`); `hide_input_in_errors` on credential models (`models/asp.py:12-13`); a golden
  trajectory corpus; usage recorded per model.
- Weaker: see §8 (versions, cache, plugins, masks).
- Since a2fafbc: 6 commits touch M0 paths, none in the lock or plugin code. Corrections: no migrations and no hashed
  step names, so `core/schema/base.py:12-13` and `core/canonical.py:5-6` overstate parity.
- Verdicts: Harbor better on R-1, R-6, R-13, R-20 and R-26; Crucible better on Q1, Q3, Q5 and Q8.

### agentenv-framework
- Mechanisms: plugin claims grouped by name and by distribution (`src/agent_env/plugins/_discovery.py:32-42`); golden wire
  bytes (`tst/unit/serialization/test_golden_wire_format.py:184-231`); a tree digest that carries its scheme and file
  modes (`utils/build_context.py:21,190`); masking of any value type (`config/describe.py:199-238`); keys scoped to their
  endpoint (`config/model.py:149-153`).
- Weaker: no fsync, unverified reads, and an unbounded config walk.
- Since 041afd7: 62 commits; the ported logic is unchanged. Corrections: A14-R13's port dropped grouping by
  distribution; the mirror described at `core/store/__init__.py:18-19` does not exist; the race tests use threads only.
- Verdicts: agentenv better on R-1, R-2, R-3, R-8, R-9 and R-31; Crucible better on durability, verified reads, tags.

### primeintellect-ai-verifiers
- Mechanisms: every call records model, sampling, usage and a real time span (`verifiers/v1/trace.py:53-82`); tools are
  named by a (namespace, name) pair (`types.py:72-80`); the Prime key is sent only to its own host
  (`configs/client.py:108-144`).
- Weaker: `Trace.version` is never read; `task_key` hashes defaults and NaN, and 1e63d3404 moved every hash
  **[ran: prime/p_pi01]**; there is no fsync.
- Since ae819de: 25 commits. Correction: resume reports top-level keys only (`cli/eval/main.py:103-107`).
- Verdicts: verifiers better on R-1, R-16, R-21 and R-24; Crucible better on Q1-Q5 and Q8.

### nvidia-nemo-gym
- Mechanisms: duplicate entry-point names raise an error (`nemo_gym/sandbox/providers/registry.py:58-84`); a missing
  version is read as 1 (`rollout_outcomes.py:29-30`); golden digest vectors; a warn-once failure-kind check
  (`failure_kinds.py:143-165`); a conformance kit tested against broken backends
  (`token_id_capture/conformance.py:91-322`).
- Weaker: nine or more version conventions; `json.dumps` lets NaN through; reads are not verified.
- Since 3ef478d: delete fences. Corrections: 2026-10-14's claim that both read a missing version as v1 is wrong, and so is
  the pin at `ARCHITECTURE.md:190`.
- Verdicts: NeMo-Gym better on R-3, R-7..R-10, R-17, R-23, R-25, R-27 and R-28; Crucible better on migrations, canonical
  form and tags.

### meta-pytorch-openenv
- Mechanisms: bounded, finite `Any` values and input-free errors
  (`src/openenv/validation/runtime/contracts.py:21-55,157-174`); the digest is taken before redaction
  (`validation/runtime/artifacts.py:63-69`); files are owner-checked (`auto/_discovery.py:340-449`); writers stamp the
  lowest version that fits; accept/refuse tables are tested (`tests/discovery/test_catalog_contract.py:88-115`).
- Weaker: `source_digest` is unframed and collides **[ran: p_openenv_hf_08]**.
- Since 49aa302: the v2 schemas were edited in place. Corrections: `THIRD_PARTY.md:102` should cite 8d3455ab;
  `core/serialize.py:1-2` names the wrong holder; "lowest version: equal" was wrong.
- Verdicts: OpenEnv better on R-5, R-6, R-13, R-14, R-15 and R-30; Crucible better on Q1-Q5, Q7 and Q8.

### huggingface-repo2rlenv
- Mechanisms: unmarked artifacts are read as protocol 1 (`src/repo2rlenv/tasksmith/author/artifact.py:163-195`); file
  modes are hashed (`emitter/bundle.py:111-133`); the default key is never sent to a custom endpoint (`llm.py:79-111`).
- Weaker: digests use `default=str`; a fixed `.tmp` (`tasksmith/author/artifact.py:61-76`; correction: not in
  `emitter/bundle.py`, which uses `mkdtemp`). Since 5555443: 2aebfbb.
- Verdicts: repo2rlenv better on R-1, R-7, R-22 and R-31; Crucible better on Q3, Q5, Q8 and Q11.

### preferencemodel-karotte
- Mechanisms: plugins are identified by distribution, and a collision raises (`src/karotte/templates.py:45-89`); store
  mode 0700 is re-applied and checked with a canary (`protected_store.py:89-120,156-196`).
- Weaker: literal keys; `error` keeps a stale score (`schemas/run_state.py:51-60`). Correction: a crash is not scored 0
  (`evaluation_runner.py:88-104`).
- Verdicts: Karotte better on R-4, R-8 and R-12; Crucible better on Q1, Q6 and Q8.

### infra-nvidia-nemo-rl
- Mechanisms: credentials are dropped by path glob, and the mismatch text carries no secrets
  (`nemo_rl/algorithms/single_controller_utils/rollout_checkpoint.py:64-104,513-550`); retirement checks
  (`utils/outdated_config_checks.py:50-136`); a failure classifier that walks the cause chain
  (`experience/failures.py:141-264`).
- Weaker: only manifest v3 is read; checkpoints are not synced. Since c9193bc: 14 commits, one in `data_plane/`.
- Correction: A14-R1's port dropped `**.token` and added a process-dependent scrub.
- Verdicts: NeMo-RL better on R-14 (and failure classification, M10); Crucible better on Q1, Q3-Q5,
  Q8.

### infra-rllm-org-rllm
- Mechanisms: an unversioned registry is upcast as v1 (`rllm/data/dataset.py:233-237,296-302`); one store suite runs over
  two backends, with paging (`rllm-model-gateway/src/rllm_model_gateway/store/base.py:6-49`).
- Weaker: newer versions are read as v1; plugins resolve first-match (`rllm/eval/agent_loader.py:118-168`). Corrections: GC
  locks per thread only; `env_str` accepts anything; keys are readable before `chmod`.
- Verdicts: rLLM better on R-7 and R-28; Crucible better elsewhere.

### Secondary patterns
- primeintellect-ai-prime-agent: a missing version reads as 1 (`crates/pa-core/src/session/mod.rs:256-274`); providers are
  global only (`settings/manager.rs:587-600`), the rule that A14-R23 implemented weakly (R-1).
- primeintellect-ai-prime-environments: 109 envs pin verifiers by lower bound; 106 use its removed v0 API.
- primeintellect-ai-research-environments: remote manifests are pinned by digest
  (`environments/code/ptx_remote/ptx_remote/taskset.py:62-86`).
- adithya-s-k-fineenvs: a size and sha256 read-back precedes the pointer write
  (`tooling/hf-rl-explorer/app/indexer.py:275-297`).
- 08-hf-rl-env-datasets: 41 git clones (correction); 33,687 `task.toml`: 15,781 legacy, 14,155 versioned, 3,751 bare.
- swe-agent-mini-swe-agent: `.env` comes only from the user dir (`src/minisweagent/__init__.py:26-36`); our port lacks its
  notice (R-11).
- xiaomimimo-verl: records per-component commit, untracked files and diff (`recipes/write_run_manifest.py:76-139`).
- infra-volcengine-verl: detects legacy layouts by shape (`verl/utils/checkpoint/megatron_checkpoint_manager.py:643-678`).
- infra-primeintellect-ai-prime-rl: removed keys are rejected by name (`tests/unit/test_configs.py:188-191`).
- infra-openrlhf-openrlhf: evicts checkpoints before saving (`openrlhf/utils/deepspeed/deepspeed.py:685-758`).
- infra-huggingface-trl: records library versions (`trl/scripts/env.py:56-74`).
- infra-inclusionai-areal: a broken versioned pointer never falls back (`areal/utils/checkpoint_pointer.py:128-138`);
  correction: A14-R17's tokenizer split is Crucible's own.
- infra-thudm-slime: the codec refuses unsupported values (`slime/data/codec.py:230-315`); hook contract tests.
- infra-novasky-ai-skyrl: one hashing module (`skycap/src/skycap/hashing.py:1-48`); correction: its CAS predicate is a
  state check.
- infra-alibaba-roll: unframed `compute_object_hash` collides (`roll/utils/hash_utils.py:6-88`)
  **[ran: trainers-c/p_tc13]**.

## 6. Gap matrix

● built and tested, ◐ built with a stated weakness, ○ absent.

| Q | Crucible | Harbor | agentenv | verifiers | NeMo-Gym | OpenEnv | repo2rlenv | Karotte | NeMo-RL | rLLM |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | ◐ chain; missing = current | ◐ never compared | ○ absent = oldest | ○ never read | ◐ missing = v1 | ◐ lowest-version writer | ◐ markerless = v1 | ○ unversioned | ◐ exact v3 | ◐ newer read as v1 |
| Q2 | ◐ `c<n>:` ids; scheme unpinned | ◐ no NFC, scheme | ◐ schemed tree digest | ○ NaN, defaults hashed | ◐ golden vectors | ◐ unframed digest | ◐ modes hashed | ○ none | ◐ plain `json.dumps` | ○ unframed 12-hex |
| Q3 | ◐ verified; short writes | ○ dir = hit | ◐ complete; no fsync | ○ no fsync | ◐ unverified reads | ○ partial fsync | ○ fixed `.tmp` | ◐ fixed `.tmp` | ◐ snapshots only | ○ no fsync |
| Q4 | ● tag CAS, history | ◐ CAS unknown | ◐ counter CAS | ○ none | ◐ per-rollout CAS | ○ resolved once | ● budget CAS | ○ n/a | ○ highest step | ◐ last writer wins |
| Q5 | ◐ `ep.name` keyed | ○ last wins | ● per-dist claims | ○ import by id | ◐ duplicates raise | ○ last wins | ○ static dict | ◐ dist identity | ○ hard-coded | ○ first match |
| Q6 | ◐ project bypass | ◐ consent; 4+3 mask | ◐ endpoint-scoped key | ◐ host-scoped key | ○ any layer | ◐ owner-checked | ◐ key not forwarded | ○ literal keys | ○ token dumped | ○ literal keys |
| Q7 | ◐ scrub-dependent lock | ◐ digest-locked inputs | ○ times only | ◐ per-call spans | ◐ install identity | ◐ digest pre-redaction | ◐ label bound to hash | ◐ provider versions | ◐ path-glob secrets | ○ six fields |
| Q8 | ◐ partial invariants | ○ class names | ○ class, message | ○ error = 0.0 | ◐ warn-once kinds | ○ free strings | ○ no blame | ○ stale score | ◐ cause chain | ○ masking off |
| Q9 | ◐ tool-name bug | ◐ no tool spec | ◐ no provider map | ◐ (ns, name) tools | ◐ unknown usage kept | ◐ sent vs requested | ○ untyped usage | ◐ service tier | ○ dict messages | ○ dict calls |
| Q10 | ◐ lock, `.env` leaks | ◐ trial-dir scrub | ◐ any-type masking | ◐ MCP auth stripped | ◐ refs in dumps | ● bounded, quiet errors | ◐ name redaction | ● 0700 plus canary | ○ full environ | ○ raw keys |
| Q11 | ◐ kit skips forget | ○ upload only | ● config backends | ○ jsonl only | ● self-tested kit | ○ none | ○ none | ○ n/a | ◐ contract test | ● one suite |
| Q12 | ◐ shape only | ◐ golden outputs | ● golden bytes | ○ none | ◐ field sets | ◐ edited in place | ○ identity tests | ○ n/a | ◐ configs validated | ○ none |

## 7. Recommendations, ranked

Four verifiers confirmed 34 items (full fields in `verify-N.md`, V1-R1..V4-9). Merged: V3-2 and V1-R5 into R-2, V3-5
and V1-R6 into R-6, V2-4 with V1-R3 steps 1-2 into R-9; V1-R3 steps 3-4 joined V2-5 as R-13. Code settled the
conflicts: `looks_like_secret_key` matches `max_tokens` (`core/config.py:142-150`) [trainers-a/p_ta03], so R-2 and R-14
use segment globs; `World.restore` validates inbox media, so R-13 keeps the A14-R26 case.
I/E/R = impact/effort/risk. Prefixes: AE agentenv, H harbor, PI prime-envs, NG nemo-gym, OH openenv-hf, C swe-safety,
MV mimo-verl, TA/TB/TC trainers. Order: R-10 before R-30 and R-31; R-9 before each format bump; R-14 with R-22, R-20
with R-26; R-2 before R-14; R-28 before a second store backend.

| R | Title | Kind | Impact | Effort | Risk |
|---|---|---|---|---|---|
| R-1 | Credentials only to operator-paired endpoints | defect | H | L | M |
| R-2 | Provider settings carry no credentials | defect | M | L | L |
| R-3 | Blob writes complete or fail | defect | M | L | L |
| R-4 | Tighten existing store roots | production gap | M | L | L |
| R-5 | Bounded, owner-checked `.env` search | defect | M | L | L |
| R-6 | Refused config never echoes its value | defect | M | L | L |
| R-7 | Unmarked documents read at written version | defect | M | L | L |
| R-8 | Plugins identified by distribution | defect | M | L | L |
| R-9 | Freeze pins bytes, ids, upcast outputs | production gap | M | L | L |
| R-10 | Canonical-form vectors per scheme | production gap | M | L | L |
| R-11 | MIT notices; licence test per ported row | defect | M | L | L |
| R-12 | Launch records every component | production gap | M | M | L |
| R-13 | Accept/refuse corpus, validator tripwire | production gap | M | M | L |
| R-14 | Run identity a pure function of settings | defect | M | M | M |
| R-15 | `Any` fields hold JSON values only | defect | M | M | M |
| R-16 | World App credentials as references | production gap | M | M | M |
| R-17 | `forget_doc` durable and lock-settled | defect | L | L | L |
| R-18 | One framed file-set digest in core | better design | L | L | L |
| R-19 | Drop `tag(expect=None)` | better design | L | L | L |
| R-20 | Usage split by model | production gap | L | L | L |
| R-21 | Real `started_at` and `ended_at` | defect | L | L | L |
| R-22 | New RunSpec field at default is no change | better design | L | L | L |
| R-23 | Rollout failure kinds in core vocabulary | defect | L | L | L |
| R-24 | One tool wire-name codec | defect | L | L | L |
| R-25 | Reasoning total None unless all reported | defect | L | L | L |
| R-26 | Digest fields hold digests | defect | L | L | M |
| R-27 | RunRecord outcome invariants | defect | L | L | M |
| R-28 | Store Protocol and kit cover consumers | production gap | L | M | L |
| R-29 | Registry kinds declare type contracts | better design | L | M | L |
| R-30 | Minor bumps keep content ids | better design | L | M | M |
| R-31 | Tree digest names its scheme, hashes +x | defect | L | M | M |

### R-1. A credential goes only to an endpoint an operator layer paired it with
- **Kind** defect · **Status** carry-over SCOUT16-5, CHG-11, LED16-1, LED16-2; correction of A14-R23; V3-1 (H16-3, AE-4,
  PI-C2, C3)
- **Problem.** `load_config` checks a project file only against the user file (`core/config.py:287-296`), per dotpath
  (`:249`), and `models` is not operator-only (`:200`). Both load and send the secret to the project's host
  (`actors/llm.py:285-290`):
  - (A) with no user file, a cloned `crucible.toml` points openrouter at `https://evil.example/v1` with
    `env:AWS_SECRET_ACCESS_KEY`;
  - (B) a project adds `[llm_providers.mirror]` holding `env:OPENROUTER_API_KEY` and re-points `models.policy-default`.

  A TOML string passes the budget check (`core/config.py:246`): `"500"` raises the cap to 500.0 **[ran: p03_ledger G]**.
- **Reference.** agentenv scopes a key to its endpoint (`agentenv-framework/src/agent_env/config/model.py:149-153`).
  prime-agent reads providers only globally (`primeintellect-ai-prime-agent/crates/pa-core/src/settings/manager.rs:587-600`).
- **Recommendation.** Invariant: a credential (an `api_key` outside `KEY_PLACEHOLDERS`, or an `env:` header) goes only
  to a `base_url` that an operator layer (defaults, user file, `$CRUCIBLE_CONFIG`, `--set`) paired with that reference.
  - `load_config` builds the operator view O without the project layer and collects (norm(base_url), ref) pairs.
  - A provider whose pair is unbound raises `ConfigError`, naming file, provider, host and variable, never the value.
  - Placeholders are exempt. `_check_project` compares validated values against O, defaults included.
- **Reasoning.** Harbor-style consent (H16-3) adds stale state and a prompt CI cannot answer. Global-only providers
  would refuse local engines that need no credential (`tests/unit/test_config_layers.py:76-79`). ARCHITECTURE §7 rule 5.
- **Established by.** Code read on both sides. **[ran: verify-3/p_v3_project_credential]**: A and B deliver the AWS
  secret or the operator's key. An override of the user's own `base_url` is refused, and the repo's `crucible.toml`
  loads.
- **Impact / Effort / Risk.** H: a cloned repo picks where secrets go. L: one check. M: credentialed project
  providers must move.
- **Mode.** PATTERN (agentenv Apache-2.0, prime-agent MIT).
- **Acceptance.** A, B, a string budget raise and a project-level `headers.Authorization = "env:X"` each raise, naming
  the variable and the host. `test_what_a_project_file_may_still_do` passes.

### R-2. Provider settings carry no credentials
- **Kind** defect · **Status** carry-over CHG-2, LED16-3 (A14-R8 gap); V3-2 + V1-R5 (H16-1 corrected, AE-4, AE-8, NG16-11)
- **Problem.** `request_defaults` is a free dict and is not operator-only (`core/config.py:51`). It is merged into every
  request (`actors/llm.py:267-272`) and copied into the lock, where `SECRET_GLOBS` miss `extra_headers`
  (`core/fingerprint.py:23`). Three leaks follow:
  - a project's `extra_headers.Authorization` replaces the operator's key **[ran: verify-3/p_v3_request_defaults A]**;
  - operator literals reach `lock.json`, and `extra_body` keys reach every `Turn.sampling.extra`
    (`core/schema/rollout.py:63-69`);
  - token-only userinfo and numeric secrets pass `redacted()` (`core/config.py:153-160`) **[ran: verify-1/v1_secrets]**.
- **Reference.** agentenv reserves the call keywords a params table may not set, and masks any-type secrets and userinfo
  (`agentenv-framework/src/agent_env/config/model.py:19-32`; `agentenv-framework/src/agent_env/config/describe.py:199-238`).
- **Recommendation.** Invariant: every header Crucible sends comes from `LLMProvider.headers` (A14-R8's `env:` rule) or
  `api_key`. In every layer, `LLMProvider` refuses the following, naming only the dotpath:
  - `extra_headers` and `extra_query`, pointing to `headers`;
  - client-owned keys (`model`, `messages`, `tools`, `stream`, `timeout`, ...);
  - leaves matching R-14's credential globs;
  - userinfo in `base_url`.

  `_redact` masks every non-null scalar under a secret key.
- **Reasoning.** One validator beats masking at three sinks. `env:` refs inside `request_defaults` would make a second,
  unchecked header path; a substring rule would drop `max_tokens` (preamble).
- **Established by.** Code read on both sides; the probes above, plus verify-1/v1_ng11_sampling.
- **Impact / Effort / Risk.** M: prompts re-billed and literals persisted. L: two validators. L: no repo config affected.
- **Mode.** PATTERN (agentenv, NeMo-Gym, Harbor Apache-2.0).
- **Acceptance.** Cases A and B and `extra_body.api_key` are refused. `redacted()` masks a numeric `auth_token`.
  `max_tokens` stays in the lock.

### R-3. Blob writes complete or fail, and the key names the bytes on disk
- **Kind** defect · **Status** carry-over SCOUT16-2, NEW-1; correction of AUD-D4 and of "hashed from the staged bytes"
  (`core/store/__init__.py:13-14`); V2-1 (AE-5, NG16-5, OH-8, C10, MV16-3, TB16-1, PI-C6 part)
- **Problem.** `put_bytes` and `put_file` drop the count `os.write` returns and key by the source bytes
  (`core/store/__init__.py:322-337`). `_commit` re-hashes only existing keys (`:302-320`). Under `RLIMIT_FSIZE` 1 MiB:
  - a 2 MiB `put_bytes` keeps 1,048,576 bytes;
  - `put_doc` indexes 2,097,163 bytes over a 1 MiB blob;
  - every read then raises "corrupt" **[ran: verify2/v2_short_write]**.

  Snapshots delete their source after `put_file` (`world/state/base.py:75-87`), so a disk that fills mid-snapshot
  leaves a WorldRef that cannot be restored.
- **Reference.** agentenv stages with `write_bytes` and `copyfile`, which write all bytes or raise
  (`agentenv-framework/src/agent_env/store/object_store/local/store.py:116-127`). slime refuses an `fstat` size mismatch
  (`infra-thudm-slime/slime/data/checkpoint.py:104-112,163-166`).
- **Recommendation.** Invariant: `_commit` runs only when the staged file holds exactly the hashed bytes.
  - `_staged` hands out a writer that completes or raises: a write-all loop that raises `StoreError` on a 0 return, or
    `os.fdopen` with flush.
  - After fsync, `os.fstat(fd).st_size` must equal the hashed count. Otherwise the put raises `StoreError` and removes
    the staged file, so neither a blob nor a row is written.
  - The `BlobStore` protocol states this.
- **Reasoning.** A verified read catches the damage only after the store has reported success. Re-hashing would double
  read I/O on multi-GB snapshots. ARCHITECTURE §7 rule 4; P16.
- **Established by.** Code read on both sides; the probe above. A buffered writer under the same limit raised `EFBIG`.
- **Impact / Effort / Risk.** M: a full disk leaves an unreadable blob whose source may be gone. L: ~15 lines. L: only
  incomplete writes change.
- **Mode.** PATTERN (agentenv, slime Apache-2.0).
- **Acceptance.** Under `RLIMIT_FSIZE`, all three puts raise and leave no file and no row. A half-writing `os.write`
  stores every byte. The vacuous D4 test (`tests/unit/test_core_audit.py:92-105`) is replaced.

### R-4. Tighten existing store roots readable by others
**Kind** production gap · **Status** carry-over A14-R6 (new roots only); V1-R8 (C11) · **I/E/R** M/L/L · **Mode** PATTERN (Karotte MIT)\
**Problem.** `_make_private_root` skips existing roots (`core/store/__init__.py:219-232`); this workspace's `data/store` is 0755, with a 0644 index **[ran: swe-safety/p_ss04]**. **Reference.** Karotte re-applies 0700 on every write (`preferencemodel-karotte/src/karotte/protected_store.py:89-120`). **Recommendation.** Each open sets an owned root to 0700 and its index files to 0600, logging the old modes once. An operator-only `store_shared` opts out. `TraceStore` drops its own chmod. **Reasoning.** A warning alone leaves old stores open (P22). **Established by.** `stat`; a08b2b9. **Acceptance.** The p_ss04 root is 0700/0600 after open.

### R-5. Bounded, owner-checked `.env` search
**Kind** defect · **Status** new (A14-R29 changed the parser only); V3-4 (OH-3, C12) · **I/E/R** M/L/L · **Mode** PATTERN (OpenEnv BSD-3-Clause, mini-swe-agent MIT)\
**Problem.** `_dotenv` walks up to `/` with no owner or mode check (`core/config.py:176-190`): a job in `/tmp/job` reads another user's 0666 `/tmp/.env` **[ran: verify-3/p_v3_dotenv_bound]**. **Reference.** OpenEnv accepts only owned files, opened `O_NOFOLLOW` (`meta-pytorch-openenv/src/openenv/auto/_discovery.py:340-449`). **Recommendation.** Search the `crucible.toml` chain (`core/config.py:208-225`), then `~/.config/crucible/.env`. Accept only regular, owned files nobody else can write; raise otherwise. **Reasoning.** Keeps the per-repo `.env` (ARCHITECTURE §7 rule 5). **Established by.** Probe. **Acceptance.** A planted parent `.env` is skipped and named.

### R-6. Refused config never echoes its value
**Kind** defect · **Status** new, continues A14-R8; V3-5 + V1-R6 (H16-2, OH-4) · **I/E/R** M/L/L · **Mode** DEPEND (pydantic `hide_input_in_errors`), PATTERN (Harbor Apache-2.0)\
**Problem.** A14-R8's errors carry `input_value=` unwrapped (`core/config.py:61-76,303`): `crucible config` prints a prefixless key whole, unscrubbed, naming no file **[ran: verify-3/p_v3_secret_echo]**. **Reference.** Harbor hides inputs on its credential models (`harbor-framework-harbor/src/harbor/models/asp.py:12-13`). **Recommendation.** Raise `ConfigError(...) from None`, built from `errors(include_input=False)` plus the files read; the config models set `hide_input_in_errors`. **Reasoning.** No scrubber knows a prefixless key. **Established by.** Probe. **Acceptance.** The probe reports `key in str(e): False`.

### R-7. Unmarked documents read at their written version
**Kind** defect · **Status** correction (A14-R1 step 3 regressed AUD-D8); SCOUT16-1; V1-R1 (AE-7, PI-C1, NG16-2, C1, MV16-2, TA16-2, TB16-3, TC16-3) · **I/E/R** M/L/L · **Mode** PATTERN (verl Apache-2.0)\
**Problem.** A missing version reads as `crucible.run/v2` (`core/schema/base.py:183`); pre-d2ebf6c rows hold `tokens: {}`, so resume raises and bench reports 0.50 USD for 0.75 (`bench/metrics.py:217-228`) **[ran: verify-1/v1_q1_unversioned]**. **Reference.** verl detects legacy data by shape (`infra-volcengine-verl/verl/utils/checkpoint/megatron_checkpoint_manager.py:643-678`). **Recommendation.** Add `register_legacy(schema, version, predicate)`: `'tokens' in d` → run/v1, then upcast; no predicate may match a current example. **Reasoning.** Blanket "missing = v1" breaks constructions. **Established by.** Probe. **Acceptance.** An old-shape run resumes and sums 0.75.

### R-8. Plugins identified by distribution
**Kind** defect · **Status** carry-over SCOUT16-4; correction of A14-R13; V3-3 (AE-1, PI-C9, NG16-1, C8) · **I/E/R** M/L/L · **Mode** PATTERN (agentenv, NeMo-Gym Apache-2.0)\
**Problem.** Staging is keyed by `ep.name` (`core/registry.py:169-196`). If two distributions expose `pack`, one vanishes while `failures` stays `{}`, and `--check` exits 0 **[ran: verify-3/p_v3_plugin_identity]**. **Reference.** agentenv keeps distributions apart (`agentenv-framework/src/agent_env/plugins/_discovery.py:32-42`). **Recommendation.** Key by `<dist>==<version>:<ep.name>`; shared hook labels go through the contested rule; failure lists are never truncated. **Reasoning.** An entry-point name is only a hook label (`packs/*/pyproject.toml:8-9`). **Established by.** Probe. **Acceptance.** Both connectors register.

### R-9. Freeze pins bytes, ids, upcast outputs
**Kind** production gap · **Status** carry-over SCOUT16-8 (A14-R18); corrects `core/schema/base.py:16-19`; V2-4 + V1-R3 (NG16-4, PI-C7, AE-2, H16-7) · **I/E/R** M/L/L · **Mode** PATTERN (agentenv, NeMo-Gym Apache-2.0)\
**Problem.** `test_formats` checks shape only (`tests/unit/test_formats.py:16-41`; 1 of 18 ids pinned): dropping `exclude_if`, an alias, an id-moving field and an edited upcaster all pass **[ran: verify-1/v1_freeze; verify2/v2_format_freeze]**. **Reference.** agentenv compares writer bytes with goldens (`agentenv-framework/tst/unit/serialization/test_golden_wire_format.py:184-231`). **Recommendation.** Freeze `<v>.example.id` and `<old>.upcast.json`; test byte-exact examples, `matches_id` and upcasts. **Reasoning.** Shape cannot see serialization changes. **Established by.** Probes. **Acceptance.** Each of the four changes fails.

### R-10. Canonical-form vectors per scheme
**Kind** production gap · **Status** carry-over SCOUT16-7; V2-3 (NG16-3 part) · **I/E/R** M/L/L · **Mode** PATTERN (NeMo-Gym Apache-2.0)\
**Problem.** `SCHEMES[1] = dumps` binds the live function (`core/canonical.py:64-72`), and only 2 ids are pinned. `ensure_ascii` or a different float spelling passes every test while the ids move **[ran: verify2/v2_canonical_mutations]**. **Reference.** NeMo-Gym ships golden vectors (`nvidia-nemo-gym/nemo_gym/token_id_capture/staging/conformance/__init__.py:16-41`). **Recommendation.** Keep one vector per rule and scheme (input, expected text, id or error); a form change keeps the old function, self-contained, under its number. **Reasoning.** A copy that shares `_form` protects nothing. **Established by.** Probe. **Acceptance.** All 5 mutations fail.

### R-11. MIT notices; licence test per ported row
**Kind** defect · **Status** correction of A14-R9; V4-3 (C13) · **I/E/R** M/L/L · **Mode** PORT obligation (MIT notice)\
**Problem.** The MIT ports from verifiers and mini-swe-agent (`THIRD_PARTY.md:35,43,47`) have no `LICENSES/` file. `actors/harness.py` matches `swe-agent-mini-swe-agent/src/minisweagent/config/mini.yaml:18,41`, yet `tests/unit/test_third_party.py:12-16` checks only the files that are named. **Reference.** `LICENSES/MIT-swe-smith.txt` already covers row 48. **Recommendation.** Add the two notices. The test requires every MIT, BSD or ISC row to name a file holding the upstream copyright line. **Reasoning.** Cheaper than rewriting the code. **Established by.** `ls LICENSES`. **Acceptance.** A ported row without a notice fails.

### R-12. Launch records every component
**Kind** production gap · **Status** carry-over CHG-4 (AUD-F3); V4-2 (NG16-12, C9, MV16-1, TA16-3, OH-10) · **I/E/R** M/M/L · **Mode** PATTERN (NeMo-Gym, MiMo verl, TRL Apache-2.0)\
**Problem.** `BuildInfo` covers only `src/crucible` and ignores untracked files (`core/build.py:26,30-35,73`), so an editable Harbor pull or untracked `grader.py` goes unrecorded **[ran: verify-4/p_v4_verify E]**. **Reference.** MiMo verl records per-component commit, untracked files and diff (`xiaomimimo-verl/recipes/write_run_manifest.py:76-139`). **Recommendation.** `BuildInfo.components` (each plugin, editable or VCS distribution: version, commit, dirty, digest), `deps`, `platform` (run_launch/v1.1), a `changed` list. **Reasoning.** In the lock they would refuse resumes after every fix. **Established by.** Probe. **Acceptance.** A version bump shows up in `changed`.

### R-13. Accept/refuse corpus, validator tripwire
**Kind** production gap · **Status** carry-over SCOUT16-8 (validators), new (version); V2-5 + V1-R3 steps 3-4 (OH-6, OH-7) · **I/E/R** M/M/L · **Mode** PATTERN (Harbor Apache-2.0, OpenEnv BSD-3-Clause); DEPEND `jsonschema` (MIT, dev)\
**Problem.** A14-R26 tightened `MediaRef` (`core/schema/actions.py:71-100`), which `World.restore` applies to inboxes (`world/world.py:989`): old WorldRefs stop restoring under "Formats: none" (`docs/changes/uzair.md:726`); no schema constrains its version **[ran: verify-1/v1_freeze C; verify2/v2_format_freeze D]**. **Reference.** Harbor validates golden real outputs (`harbor-framework-harbor/tests/integration/test_trajectory_validation.py:15-48`). **Recommendation.** An append-only real-producer corpus and accept/refuse cases through readers and schemas; frozen validator digests; a version enum. **Reasoning.** Shape misses tightenings. **Established by.** Probes. **Acceptance.** Re-applying A14-R26 fails.

### R-14. Run identity a pure function of settings
**Kind** defect · **Status** correction of A14-R1; carry-over SCOUT16-6, CHG-2 part; V4-1 (PI-C4, OH-1, OH-2, TA16-1, TB16-4) · **I/E/R** M/M/M · **Mode** PATTERN (NeMo-RL Apache-2.0, OpenEnv BSD-3-Clause)\
**Problem.** `project` scrubs process-registered values and joins paths with `.` (`core/fingerprint.py:23-55`): the fingerprint changes after `register_secret`, `X-Api-Key` lands in `lock.json`, `{'a.b':1}` equals `{'a':{'b':1}}` **[ran: verify-4/p_v4_verify C]**. **Reference.** NeMo-RL drops credentials by segment glob and never scrubs values (`infra-nvidia-nemo-rl/nemo_rl/algorithms/single_controller_utils/rollout_checkpoint.py:68-104`). **Recommendation.** `compat_identity/v2`: segment paths, `CREDENTIAL_GLOBS` (tokenizer tokens exempt), no pre-hash scrub; the lock stores scrubbed values, `withheld`, `redacted`. **Reasoning.** A substring rule fails open on `max_tokens`. **Established by.** Probe. **Acceptance.** Fingerprints are equal across `register_secret`.

### R-15. `Any` fields hold JSON values only
**Kind** defect · **Status** carry-over NEW-3, SCOUT16-3; correction of A14-R10; V2-2 (NG16-10, OH-5, C2, TB16-2) · **I/E/R** M/M/M · **Mode** PATTERN (OpenEnv BSD-3-Clause; slime Apache-2.0)\
**Problem.** `allow_inf_nan=False` covers floats only (`core/schema/base.py:37`); in 50 `Any` fields NaN dumps as null, `{1:'a','1':'b'}` loses a value, a set gets 3 ids in 3 processes, 256-deep nesting fails at write (`runtime/journal.py:94-98`) **[ran: verify2/v2_any_fields]**. **Reference.** OpenEnv bounds `Any` values to finite JSON (`meta-pytorch-openenv/src/openenv/validation/runtime/contracts.py:21-55`). **Recommendation.** `Json = Annotated[Any, AfterValidator(json_value)]`: coerce tuples, int keys; refuse NaN, sets, bytes, objects, collisions, depth > 64. **Reasoning.** `JsonValue` re-renders schemas. **Established by.** Probe; 0 of 182 repo files refused. **Acceptance.** Ids agree across processes.

### R-16. World App credentials as references
**Kind** production gap · **Status** new; V1-R7 (PI-C8) · **I/E/R** M/M/M · **Mode** PATTERN (verifiers MIT, NeMo-Gym Apache-2.0)\
**Problem.** The MCP-server App passes `params` headers and env as literals (`world/apps/mcp_server.py:61-65,86-91`), the only route past `safe_environ`: tokens sit in `stored_json` and move the World id **[ran: verify-1/v1_secrets PI-C8]**. **Reference.** verifiers resolves `headers_from_env` when it builds the client (`primeintellect-ai-verifiers/verifiers/v1/configs/client.py:48-55`). **Recommendation.** Secret-named headers and env must be `env:NAME` or `literal:<fixture>`, resolved at App `start()` (M1). **Reasoning.** Host pass-through defeats `safe_environ`. **Established by.** Code read. **Acceptance.** The World id is the same across values.

### R-17. `forget_doc` durable and lock-settled
**Kind** defect · **Status** correction of A14-R24; V2-6 (AE-6, NG16-6) · **I/E/R** L/L/L · **Mode** PATTERN (agentenv, NeMo-Gym Apache-2.0)\
**Problem.** `forget_doc` syncs no directory (`core/store/__init__.py:445-474`), and settling skips live pids, its own and pid 1 included (`:258-275`); `get_doc` then raises "missing blob" **[ran: verify2/v2_forget]**. **Reference.** agentenv sweeps only files it can lock (`agentenv-framework/src/agent_env/store/object_store/local/store.py:419-431`). **Recommendation.** Hold `LOCK_EX` from rename to unlink, fsync before COMMIT, and settle only on a successful `LOCK_NB`. **Reasoning.** flock is already the store's liveness test. **Established by.** Probe. **Acceptance.** Files named with the store's own pid are settled.

### R-18. One framed file-set digest in core
**Kind** better design · **Status** new; V2-8 (TC16-1; M9, M15 handoffs) · **I/E/R** L/L/L · **Mode** PATTERN (SkyRL Apache-2.0)\
**Problem.** `sha256_files` and `_content` hash path then bytes with no framing (`accept/key.py:31-36`; `accept/checks/base.py:130-139`), so two different trees can share a grade key **[ran: verify2/v2_file_digests A]**. **Reference.** SkyRL hashes in one module (`infra-novasky-ai-skyrl/skycap/src/skycap/hashing.py:1-48`). **Recommendation.** Add `core.ids.files_digest(paths, root)` with `tree_digest`'s framing, and build `tree_digest` on it. **Reasoning.** A local fix would make a third scheme (P19). **Established by.** Probe. **Acceptance.** The two colliding pairs differ.

### R-19. Drop `tag(expect=None)`
**Kind** better design · **Status** carry-over CHG-18; V3-6 (TC16-4) · **I/E/R** L/L/L · **Mode** PATTERN (SkyRL Apache-2.0)\
**Problem.** `expect=None` skips the compare-and-set check (`core/store/__init__.py:476-491`; `core/store/protocols.py:43`). Only `tests/unit/test_tags.py:29` uses it. **Reference.** SkyRL moves state only by predicated UPDATE (`infra-novasky-ai-skyrl/skyrl/tinker/api.py:1145-1214`). **Recommendation.** Make `expect: int` required, with 0 meaning create; the conformance kit asserts the refusal. **Reasoning.** A `force` flag would only document the hazard. **Established by.** Grep of callers. **Acceptance.** `expect=None` raises.

### R-20. Usage split by model
**Kind** production gap · **Status** new, continues A14-R19, IMP-M1-8; V3-8 (H16-5) · **I/E/R** L/L/L · **Mode** PATTERN (Harbor Apache-2.0)\
**Problem.** `UsageTotals.of` merges usage across models (`core/schema/usage.py:138-163`), so bench's `tokens_in` reads 10,000 for 1,000 root and 9,000 sub-model tokens (`bench/metrics.py:145-166`) **[ran: verify-3/p_v3_usage_digest]**. **Reference.** Harbor keeps `model_usage` (`harbor-framework-harbor/src/harbor/models/agent/context.py:8-38`). **Recommendation.** Add `by_model`, omitted for a single model and summing to the totals; ship it in run/v2.1 with R-26. **Reasoning.** Otherwise each consumer re-sums `calls.json`. **Established by.** Probe. **Acceptance.** Per-model rows sum to the totals.

### R-21. Real `started_at` and `ended_at`
**Kind** defect · **Status** carry-over NEW-4; V4-5 (PI-C3) · **I/E/R** L/L/L · **Mode** PATTERN (verifiers MIT)\
**Problem.** One `now`, at one-second resolution, fills both fields (`runtime/run.py:327-331`) **[ran: verify-4/p_v4_verify B]**. **Reference.** verifiers records real span instants (`primeintellect-ai-verifiers/verifiers/v1/trace.py:53-82`). **Recommendation.** Take `t_dispatch` before `harness.run`/`regrade` (`runtime/run.py:205,295`) and `ended_at` at record time; R-27 checks the order. **Reasoning.** No reader needs per-stage spans yet. **Established by.** Probe. **Acceptance.** A harness that sleeps 50 ms gives `started_at < ended_at`.

### R-22. New RunSpec field at default is no change
**Kind** better design · **Status** new; V4-6 (C7) · **I/E/R** L/L/L · **Mode** PATTERN (repo2rlenv Apache-2.0, Karotte MIT)\
**Problem.** A path missing from the old lock counts as a change (`runtime/lock.py:118-125`), so adding a `bool = False` field refuses every in-flight resume **[ran: verify-4/p_v4_verify D]**. **Reference.** repo2rlenv keeps the old identity contract (`huggingface-repo2rlenv/src/repo2rlenv/tasksmith/author/artifact.py:163-195`). **Recommendation.** Record `RunLock.fields` (v1.1, with R-14). A new field at its default is merged as `added_at_default`. **Reasoning.** Ignoring missing paths outright would fail open. **Established by.** Probe. **Acceptance.** The default resumes, and `True` is refused.

### R-23. Rollout failure kinds in core vocabulary
**Kind** defect · **Status** carry-over SCOUT16-10; V4-7 (NG16-9) · **I/E/R** L/L/L · **Mode** PATTERN (NeMo-Gym Apache-2.0)\
**Problem.** `failure_kind` and `blame` are free strings (`core/schema/rollout.py:279-280`). M12 emits five undeclared `crucible:*` kinds no pack may register (`train/runtime.py:402,406,444,534,573`) **[ran: critic/p_crit_kinds]**, plus six bare labels; all read `known=False` **[ran: verify-4/p_v4_verify F]**. **Reference.** NeMo-Gym tests that every label it emits is registered (`nvidia-nemo-gym/tests/unit_tests/test_failure_kinds.py:134-147`). **Recommendation.** Declare those kinds, warn once on unknown ones, type `blame` as `Blame | None`, and scan for literals. **Reasoning.** Refusing unknown kinds would lose stored rollouts. **Established by.** Probe. **Acceptance.** Every `masked/<kind>` is known.

### R-24. One tool wire-name codec
**Kind** defect · **Status** carry-over NEW-9; V4-8 (PI-C5, NG16-13, MV16-4) · **I/E/R** L/L/L · **Mode** PATTERN (verifiers MIT, NeMo-Gym Apache-2.0)\
**Problem.** `to_openai` emits uncapped `app__name`, `_FN_OK` uses `.match` (`core/schema/actions.py:18-50`): 66 chars give 67, a newline passes; split-string decoders misread `my__app` (`actors/simulated.py:172-174`) **[ran: verify-4/p_v4_verify G]**. **Reference.** verifiers keeps (namespace, name) pairs (`primeintellect-ai-verifiers/verifiers/v1/types.py:70-80`). **Recommendation.** Core owns `wire_name` (fullmatch, ≤64) and a `ToolNames` table that refuses collisions (M2, M11). **Reasoning.** OpenAI chat has no namespace field. **Established by.** Probes. **Acceptance.** A property test round-trips every name.

### R-25. Reasoning total None unless all reported
**Kind** defect · **Status** new; V4-9 (NG16-15) · **I/E/R** L/L/L · **Mode** PATTERN (NeMo-Gym Apache-2.0)\
**Problem.** Calls reporting 100, 100 and nothing sum to 200 at coverage 1.0 (`core/schema/usage.py:143-162`) **[ran: verify-4/p_v4_verify H]**. No code reads the total yet. **Reference.** NeMo-Gym leaves unknown counts unknown (`nvidia-nemo-gym/nemo_gym/openai_utils.py:1102-1130`). **Recommendation.** Sum only when every live call reported; otherwise return None. **Reasoning.** The class's own rule: unknown is not 0. **Established by.** Probe. **Acceptance.** Two of three reports give None.

### R-26. Digest fields hold digests
**Kind** defect · **Status** new, continues A14-R2; V3-7 (H16-6) · **I/E/R** L/L/M · **Mode** PATTERN (Harbor Apache-2.0)\
**Problem.** `_digest_now` stores `unreadable: <error>`, including a host path, as `task_digest` (`runtime/run.py:77-84,330`). The lock fields are free strings (`core/schema/run.py:201,256-286`) **[ran: verify-3/p_v3_usage_digest]**. **Reference.** Harbor validates `sha256:` plus 64 hex characters (`harbor-framework-harbor/src/harbor/models/job/lock.py:45-79`). **Recommendation.** Store None and put the text in the `task_changed` reason. A `Digest` type applies from run/v2.1 and lock/launch v1.1. **Reasoning.** Tightening v2 would refuse its own example. **Established by.** Probe. **Acceptance.** v2.1 refuses `'sha256:00'`.

### R-27. RunRecord outcome invariants
**Kind** defect · **Status** correction of AUD-D11; V4-4 (NG16-8) · **I/E/R** L/L/M · **Mode** PATTERN (NeMo-Gym Apache-2.0)\
**Problem.** `_consistent` checks outcome against mask only (`core/schema/run.py:210-216`): `failed` with `transport_timeout` and reward 0.0 is scored (P16), and `ended_at < started_at` passes **[ran: verify-4/p_v4_verify A]**. **Reference.** NeMo-Gym requires exactly one of result and failure (`nvidia-nemo-gym/nemo_gym/episode_types.py:129-136`). **Recommendation.** `completed` implies no failure; `failed` or `masked` implies one; a known kind sets the outcome; times must be ordered. **Reasoning.** A defensive reader hides it. **Established by.** Probe. **Acceptance.** The four probe cases are refused.

### R-28. Store Protocol and kit cover consumers
**Kind** production gap · **Status** carry-over SCOUT16-9, CHG-15; correction of A14-R22; V1-R4 (NG16-7, TC16-2) · **I/E/R** L/M/L · **Mode** PATTERN (NeMo-Gym, rLLM Apache-2.0)\
**Problem.** The kit skips `forget_doc` and cross-name listing (`core/store/conformance.py:28-105`), so 3 of 4 mutants pass **[ran: verify-1/v1_store_kit]**; `TraceStore` queries `store._db` (`connectors/trace_store.py:112-116`). **Reference.** NeMo-Gym tests its kit against broken backends (`nvidia-nemo-gym/tests/unit_tests/test_token_capture_conformance.py:200-280`). **Recommendation.** `DocIndex` gains `names`, paged listing and `provenance`; ban `._db` outside the store; add named checks and mutant self-tests. **Reasoning.** A kit-passing backend must not break consumers. **Established by.** Probes. **Acceptance.** Each mutant fails.

### R-29. Registry kinds declare type contracts
**Kind** better design · **Status** new; V3-9 (TB16-5) · **I/E/R** L/M/L · **Mode** PATTERN (slime Apache-2.0)\
**Problem.** `check()` only resolves pointers (`core/registry.py:135-154`). A function registered as a `state_adapter`, or `json:loads` as a `segmenter`, passes and fails at first use (`workflow/steps.py:180-185`) **[ran: verify-3/p_v3_tb13_rerun]**. **Reference.** slime tests hook signatures in CI (`infra-thudm-slime/tests/plugin_contracts/test_plugin_path_loading_contracts.py:174-230`). **Recommendation.** Each kind declares a lazy base class, Protocol or `callable`; `check()` reports violations and exits 1. **Reasoning.** Smoke calls would need fixtures in core. **Established by.** Probe. **Acceptance.** Both probe entries are reported.

### R-30. Minor bumps keep content ids
**Kind** better design · **Status** correction of `core/schema/base.py:16-19`; V1-R2 (OH-9) · **I/E/R** L/M/M · **Mode** PATTERN (OpenEnv BSD-3-Clause)\
**Problem.** Code-built documents hash the current label (`core/schema/base.py:53-59,152-154`), so a world minor bump changes the ids of WorldSpecs built at 17 code sites and breaks `World.restore` (`world/world.py:968-975`) **[ran: verify-1/v1_oh9_fold]**. **Reference.** OpenEnv stamps the lowest sufficient version (`meta-pytorch-openenv/src/openenv/validation/parsers/openenv_yaml.py:72-96`). **Recommendation.** Scheme `c2:` hashes the label folded to `<name>/v<major>`, with vectors (R-10). **Reasoning.** Lowest-minor stamping cannot express rollout v1.1. **Established by.** Probe. **Acceptance.** A simulated world v1.1 keeps its ids.

### R-31. Tree digest names its scheme, hashes +x
**Kind** defect · **Status** new; V2-7 (AE-3 narrowed, C5) · **I/E/R** L/M/M · **Mode** PATTERN (agentenv, repo2rlenv Apache-2.0)\
**Problem.** `tree_digest` hashes no scheme and no file modes (`core/ids.py:114-147`), although `:5-6` treats the executable bit as state. A `chmod +x` between lock and resume therefore goes unseen **[ran: verify2/v2_file_digests B]**. **Reference.** agentenv versions its scheme and hashes 0755/0644 (`agentenv-framework/src/agent_env/utils/build_context.py:20-21,190`). **Recommendation.** Scheme 2 (`t2:sha256:`) adds the mode and gets a golden tree; locks compare under the scheme they recorded (M10). **Reasoning.** NFC-folding was rejected because Harbor does not fold. **Established by.** Probe. **Acceptance.** `+x` changes only the scheme-2 digest.

### Rejected in verification

| Candidates | Reason |
|---|---|
| C4 | No path writes an unresolved env value into text: `safe_environ` strips credentials (`world/exec.py:112-117`) |
| AE-10, NG16-14 | Premature: no second store backend exists, and `get_store` has 3 callers |
| TA16-2 (config half) | `[backends]` is already refused, and `crucible.toml` is not a persisted format |
| PI-C6 (F_FULLFSYNC half) | Only the developer host runs darwin; CI and training run on Linux |
| NG16-3 (part) | A frozen copy that shares `_form` protects nothing, and tests already catch 2 of its 3 mutations |
| AE-3 (part) | NFC folding matters only for cross-normalization copies, and Harbor does not fold |
| NG16-10 (part) | No realistic writer emits underflowing literals |
| AE-11 | Packs live in this repo, and CI runs their hooks and tests (M14 handoff) |
| MV16-5 | Collisions are already withheld; a selection knob adds a second source of truth |
| AE-9 | R-1 refuses the case at load; `RunLaunch.config` already records the layers |
| TC16-5 | One key was ever retired, and its error names it; the file half is R-6 |
| H16-4 | No caller reorders tasks, the check fails closed, and a fix would re-key locks |
| C6 | Nothing gates on value proofs, and the verdict is bound to an AcceptanceKey (`accept/model.py:113-135`) |

## 8. What we already do better (keep)

- Versions: one upcaster chain refusing newer and unknown keys (`core/schema/base.py:131-201`), where Harbor never compares
  versions (`harbor-framework-harbor/src/harbor/models/task/config.py:801-849`) and rLLM reads newer as v1.
- Ids: canonical JSON with `c<n>:` ids (`core/canonical.py:26-72`), where verifiers hashes `json.dumps` with NaN
  (`primeintellect-ai-verifiers/verifiers/v1/task.py:46-48`), as do NeMo-Gym and NeMo-RL.
- Store: staged, fsynced, verified blobs and tag CAS (`core/store/__init__.py:286-386,476-524`), where Harbor's cache trusts
  any existing dir (`harbor-framework-harbor/src/harbor/tasks/client.py:392-444`) and agentenv never fsyncs.
- Registry: staged merge, withheld collisions, core gate (`core/registry.py:156-242`), where in Harbor the last entry point
  wins (`harbor-framework-harbor/src/harbor/cli/plugin_registry.py:6-10`), and in rLLM the first match does.
- Secrets: operator-only keys, `env:` refs, full masks (`core/config.py:61-80,143-160,198-251`), where Harbor's mask keeps
  4+3 characters (`harbor-framework-harbor/src/harbor/utils/env.py:52-55`) and ROLL and slime send keys to W&B.
- Failures: blame, retry, masking, no reward when masked (`core/schema/run.py:23-229`), where Harbor records class names
  (`harbor-framework-harbor/src/harbor/models/trial/result.py:21-36`) and verifiers scores errors as 0.0.
- Runs: fail-closed lock, guarded `BuildInfo`, typed usage (`runtime/lock.py:29-139`; `core/schema/usage.py:28-176`), where
  Harbor raises a bare `FileExistsError` (`harbor-framework-harbor/src/harbor/job.py:925-949`).
- Freeze: per-version frozen schemas (`tests/unit/_formats.py:55-124`), where OpenEnv's 80309e07 edited v2 in place.

## 9. Open questions

- Data at rest: pre-versioning RunRecords (R-7), records R-27 would refuse, caches R-18 re-keys. None found in code, git
  history or this workspace; needs the operator's inventory. Whether GC is due (SCOUT-5) needs a measured run.
- Owner: plugin identity by distribution or unique names (R-8 fits both); a tree-digest ignore file; resume across upgrades
  (R-22).
- Harbor: is the server-side tag move CAS (OQ-1), do typed agent errors reach `boundary()`, does its hash cover modes?
  Needs Harbor's server, an install, or its maintainers.
- Deployment: macOS stores (PI-C6), where config errors land (R-6), second `Runner.run` per process (R-14), outside readers
  of `schemas/` (R-13); needs the deployment matrix.
- Providers: `input_audio` formats, the 64-character name rule, cross-provider `tool_call_id`; needs provider contracts.
- Not run: verifiers resume before 1e63d3404, OmegaConf unknown keys, prime-rl `pydantic_config`, OpenRLHF missing rewards,
  straw-queue, slime's lock, M10's unserializable records (p_ss03), OQ-3's rest.

## 10. Handoff: findings outside this module

| To | Finding |
|---|---|
| M10 (lock) | Path settings lock as strings, so a patch rewritten in place goes unseen (`runtime/lock.py:84-85`); objects lock as `<TypeName>` (`:76-96`); `lock.json` and launches are 0644 (`:116,161`); run dirs follow the umask (CHG-1); add a preflight mode (MiMo). |
| M10 (journal) | No single-owner lock: a second resume moves live attempts aside (`runtime/journal.py:113-123`) [trainers-b/p_tb12]. A fixed `.tmp` with no dir fsync (`:62-69`). `_read_jsonl` skips bad lines; `sample_id` bypasses `child_path`. |
| M10 (failures) | `boundary()` matches class names, ignores `__cause__`, treats any OSError as infra (`runtime/infra.py:89-109`; cf. `infra-nvidia-nemo-rl/nemo_rl/experience/failures.py:141-264`); NORMALIZE maps 2 Harbor errors; an unused second `default_blame` ignores pack kinds (`runtime/infra.py:215-216`); `agent_request_failed` is a 4xx here, "never ran" in NeMo-Gym (`nvidia-nemo-gym/nemo_gym/failure_kinds.py:69-72`). |
| M10 (secrets) | Port Harbor's by-value trial-dir scrub (`harbor-framework-harbor/src/harbor/trial/trial.py:1118-1170`), plus verifiers' per-rollout bearer tokens (`primeintellect-ai-verifiers/verifiers/v1/interception/server.py:369-391`) and retry budgets. |
| M11 interop | Harbor `${VAR}`: missing → `''`, copied without consent (`interop/harbor/verify_dir.py:63-72`); `schema_version` unchecked, 5 TB-3.0 tasks at 2.0 (`interop/harbor/task_io.py:91`); NeMo-Gym `partition('__')` (`interop/nemo_gym/client.py:70-76`); MiMo ignores `rubric_weight`, and its `src_protect` gate passes when it raises (`xiaomi-mimo-v2.6-rl-oss/general/envs/*/verify.py:360-366`); EOG ids become file names, else `task_{id(row)}` (`servicenow-enterpriseops-gym/evaluate.py:297-298`); pin served tool lists by digest. |
| M12 train | `train/state.py:92-127`: no fsync, `rmtree` before replace, highest step wins, unpickles before the schema check. verl placeholders at 0.0 enter GRPO groups (`train/bridges/verl_agent_loop.py:165-176`). A masked `RolloutRecord` keeps its reward (`train/runtime.py:566`). R-14, R-23 apply. |
| M1 world, M7 verify | `EventWhen` lacks `when.roles` (`core/schema/world.py:217`); `world_digest` deferred (CHG-6); no image credential scan (`preferencemodel-karotte/src/karotte/check_credentials.py:1-80`); R-16's App side; withhold sandbox output (`meta-pytorch-openenv/src/openenv/core/containers/runtime/novita_provider.py:1112-1124`); stale MiMo counts (`ARCHITECTURE.md:38,195`). |
| M2 actors | `request_defaults` merge (R-2, `actors/llm.py:267-272`); simulated tool names (R-24); bounds in `_parse_arguments` (R-15); record requested vs sent sampling; price by service tier (Karotte); correct the skycap cite (`infra-novasky-ai-skyrl/skycap/src/skycap/retry.py:1-21`) and the AReaL one (`docs/research/improvements/M00-M02.md:55`). |
| M3 connectors, M16 | `TraceStore` uses `store._db` and reads every row (`connectors/trace_store.py:112-142`, R-28). OpenEnv's ATIF reconcile fails fatally on token mismatch (`meta-pytorch-openenv/src/openenv/harbor/atif.py:232-300`). |
| M4 signals | The coding gate rebuilds `set(ts.ids())` per lookup, taking 3.8 s for 50 (`packs/coding/src/crucible_pack_coding/m4/gate.py:28-31`). |
| M9 accept, M15 studio | Unframed `sha256_files`/`_content` (`accept/key.py:31-44`; `accept/checks/base.py:130-139`) and the plan hash (`studio/accept_adapter.py:55-60`) should use R-18's framing. |
| M13 bench | `load_run` drops unreadable attempts uncounted (`bench/metrics.py:224-228`); record each dataset's revision. |
| M14 packs | A plugin-surface check (AE-11) once packs ship as wheels; SPDX only in `pack check` (`studio/pack.py:568-624`); `finance_ap` not in CI. |
| M0 core (incidental) | Importing `core.build` first is circular (`core/build.py:24`) [p_mv14]; float knobs accept nan (`core/env.py:123-124`); GC must persist boundaries first (HO-1). |

## 11. Sources

- **References.** The 33 folders at the commits in §4; earlier pins re-read: the bracketed ones there, agentenv ac413aa.
- **Crucible documents.** `ARCHITECTURE.md`, `DESIGN.md` (in part), `THIRD_PARTY.md`, `docs/modules/M00-core.md`,
  `docs/changes/uzair.md`, `docs/audit/2026-10-12/M00-core.md`,
  `docs/research/improvements/{2026-10-14/M00-core.md, M00-M02.md, 00-index.md}`, `pyproject.toml`, `uv.lock`,
  `../.github/workflows/crucible.yml`.
- **Crucible code.** All of `core/` and `_vendor/`; `runtime/{run,lock,journal,keys,infra}.py`;
  `connectors/trace_store.py`; `world/{world,exec,audit}.py` and `world/apps/mcp_server.py`;
  `actors/{llm,policy,simulated,harness}.py`; `train/{trainer,runtime,identity,state,world_env}.py`; `bench/metrics.py`;
  `accept/`; `studio/`; `cli/main.py`; `interop/`; the pack hooks; 22 M0 test files.
- **Probes** (`probes/`). `p01_scout`, `p02_gate`, `p03_ledger`, `p04_lock_opaque`; the sweep probes (`p_harbor_01*`,
  `p_ae01_agentenv`, `prime/`, `p_openenv_hf_*`, `swe-safety/`, `p_mv1*`, `trainers-{a,b,c}/`); the verifier
  probes (`verify-1/`, `verify2/`, `verify-3/`, `verify-4/`); the critic's (`critic/`).
- **Work folder:** `01-brief.md`, `02-ledger.md`, ten `sweep-*.md`, four `verify-*.md`, `notes/`, `probes/`.
