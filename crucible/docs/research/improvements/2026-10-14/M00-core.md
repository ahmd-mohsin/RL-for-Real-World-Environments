# M00 core: audit against the lab RL repos (2026-10-14)

Crucible HEAD `c27f5ca`. Citations: Crucible source under `crucible/src/crucible/` (core/ids.py:35); other Crucible files
relative to `crucible/` (tests/..., DESIGN.md), `../` for the monorepo root; reference code as `<folder>/<path>:line`, the folder
dropped in the repo's own §5 subsection or bullet, which may also drop the package prefix (src/harbor/, src/agent_env/,
nemo_gym/) and cite a file named earlier by its tail; commits are in §4. **[ran: x]** marks a claim reproduced by probe x.
Notes, probes and raw outputs: `/Users/uzair/Desktop/RL env repos/.module-audit/2026-10-14/M00-core/` (§11).

## 0. Summary

- **State.** M0 is built and its gate holds in behaviour: 67 core unit tests pass, `lint-imports` keeps 4 contracts with 0
  broken, all 11 documents round-trip with equal hashes and 82 models export a JSON Schema under warnings-as-errors
  [ran: p02_ledger]. No reference folder matches its versioning, canonical hashing, verified store or failure taxonomy (§8).
- **Ledger.** 69 items: 20 done, 19 partly, 28 not done, 2 superseded, 0 wrong. The 2026-10-12 audit's defects are mostly
  done; the 2026-10-11 study's M0 features (RunLock, layered config, tag CAS, store protocols, typed usage) are open.
- **Where the output matters.** Run and rollout records drive resume, bench and the trainer; their identity is hollow:
  - the resume identity covers 8 of 18 RunSpec fields and display ids (runtime/run.py:55-58); endpoint, sampling sent,
    harness limits, task bytes and code state are outside, so a resumed run mixes regimes silently [ran: p_v3_run_identity];
  - RunRecord.task/world hold names, Turn.sampling the declaration (default `{}`), RunRecord.tokens nothing.
- **Gaps.**
  - Formats: crucible.rollout/v1 changed in place twice, so a c051249 reader rejects every HEAD record; 2 golden hashes,
    4 versioning conventions, 0 production upcasters, HASH_SCHEME in no id.
  - Security: validate_id has no caller (`../escaped` leaves the runs root); non-vendor secrets survive scrubbing; one
    duplicated YAML key makes a no-network World public; stores are 0755; a parent crucible.toml redirects the API key and
    raises the budget 500x; agentenv's NOTICE is missing.
  - Registry and store: half-failed plugins keep their names; contested names follow install order; a tag lost update lets
    purge_superseded delete the newer trace; a forget/put race orphans rows; NFD text is folded on storage.
- **Top recommendations** (31 in §7): **R-1** RunLock (fail-closed identity, resolved config, launch record); **R-2** task
  and World digests re-checked before grading; **R-3** the sampling actually sent, typed per call. **R-4 to R-15** are
  low-effort fixes of medium impact. **R-18** (frozen schemas per version) should land first: R-1, R-2, R-3, R-17 and R-19
  change persisted formats.
- **Coverage.** No sweep cluster or verify batch failed; 34 verified items merge into 31; 5 candidate groups rejected (§7).

## 1. What the module does

M0 holds the shapes every Crucible module agrees on: World, task, trace, reward, run and rollout documents; one canonical JSON
form and the content ids over it; provenance; a lazy `module:attr` registry fed by `crucible.plugins` entry points;
crucible.toml with `env:NAME` secrets and a scrubber; a content-addressed store (blobs, a SQLite document index, tags). It has
no domain logic and imports no framework: 3,932 lines in 25 files under core/.

| Contract | |
|---|---|
| Inputs | dicts, JSON, YAML, TOML (tasks, packs, crucible.toml, connectors, trainers); bytes and files; entry points; env vars and .env |
| Outputs | strict pydantic models, refused or upcast (core/schema/base.py:27-28,94-143); canonical JSON and sha256 ids (core/canonical.py:20-39); store keys, rows, tags; registry objects; a redacted Config |
| Called by | world 39 files, interop 25, packs/_toy 20, train 17, runtime 17, packs/coding 17, workflow 11, actors 11, 11 more; schema.actions has 82 imports |
| Layer | bottom of the 14-layer contract (pyproject.toml:153-171); its builtin pointers name 10 interop and 5 connector modules (core/registry.py:131-189) |
| Principles | in core/schema/: P12 milestone.py:26-46, P13 reward.py:35-45, P16 run.py:76-88, P18 task.py:145-164, P19 world.py:425-522, P28 state.py:91-107; P27 core/agent_events.py:244-343; §7 rule 1 held, rule 5 partly (R-5, R-8), rule 4 not |

**Gate** (ARCHITECTURE.md:438-442): schemas round-trip and export, 5 seeds and the toy World validate, contracts run in CI,
THIRD_PARTY.md exists; status "built | passed" (:37). Measured: 67 passed (72 and 2 skipped with test_triggers); lint-imports
4 kept. Not automated: tests round-trip only WorldSpec and RewardJson, with no warnings filter (DOC-gate-1); a layer break
still landed (24f3b46, NEW-7); seeds through Harbor not re-run (Harbor absent).

| Consumer of M0 output | Uses it? | Evidence (recommendation) |
|---|---|---|
| RunJournal resume (identity_doc) | 8 of 18 RunSpec fields, display ids only | runtime/run.py:55-58; runtime/journal.py:72-79 (R-1) |
| Bench and resume (RunRecord.task/world) | names, not the documented hashes | runtime/run.py:282; core/schema/run.py:158-159 (R-2) |
| Trainer guards (Turn.sampling) | a copy of RunSpec.sampling, default `{}` | runtime/run.py:268; train/trainer.py:745-748 (R-3) |
| Bench token columns (RunRecord.tokens) | no producer fills it | bench/metrics.py:140-160 (R-19) |
| Provenance via put_doc | one producer, empty derived_from | connectors/trace_store.py:71-72 (ledger IMP-S3) |
| validate_id, Config.redacted() | no production caller | core/ids.py:35; core/config.py:84 (R-4, R-1) |
| Store | per-attempt `Store(workdir / "s")` at 21 sites; traces in the global store | runtime/single_turn.py:77 (R-6) |
| Registry kinds segmenter, replayer | read, never registered | workflow/steps.py:180-185; workflow/replay.py:235-240 (R-28) |

## 2. Status ledger

69 items: 20 done, 19 partly, 28 not done, 2 superseded, 0 wrong. Full evidence and probe output: 02-ledger.md.

| Item | State | Evidence |
|---|---|---|
| IMP-S1, IMP-S2, IDX-L1, AUD-D3 | done | 14-layer contract pyproject.toml:153-171 and CI ../.github/workflows/crucible.yml:42-49 (c051249); 4 kept this run |
| IMP-M0-4, IMP-M0-6 (2026-10-11 study) | done | WAL, 5 s busy timeout (core/store/__init__.py:68-78) [ran: p03_store_pragmas]; upcaster chain (core/schema/base.py:105-128) |
| AUD-D1, D2, D4, D5, D6, D7, D9, D10, D11, D12, D14, D15, D19, D20 (2026-10-12 audit) | done | d2ebf6c: core/schema/base.py:28,130-139, core/canonical.py:20-39, core/store/__init__.py:142-193, core/registry.py:93-112; tests c051249 |
| IMP-M0-1, RES-ACC-1 | superseded | met another way: helpers in core/agent_events.py:120,311 and a CI job (c051249); acceptance/v1 (accept/model.py:94-127) |
| IMP-S3 | partly | the sole put_doc caller passes prov with empty derived_from (connectors/trace_store.py:71-72); prov, SourceRef.hash optional |
| IMP-M0-5, AUD-F1 | partly | failures described (core/registry.py:93-112); no strict mode, API version or load-time pin check (studio/pack.py:567-586 only) |
| IMP-M2-4 | partly | Turn.trainable exists (core/schema/rollout.py:76); ActorSpec and trace events carry none |
| AUD-D8, AUD-D13 | partly | reward and run versioned; StepRow, Signal, WorldRef, Record not; defaults hashed, scheme in no id [ran: p02_ledger G] |
| AUD-D16 | partly | validate_id tested but uncalled; `../escaped_run` lands outside the root [ran: p02_ledger J] |
| AUD-D17 | partly | pack kinds and bounded reason built (core/schema/run.py:91-139); attempt is still a free str (:175) |
| AUD-D18 | partly | alias Provenance still imported by 3 callers (core/schema/milestone.py:49); manifest_hash test-only; second SourceRef connectors/base.py:64 |
| AUD-F2 | partly | Turn.sampling dict, Turn.finish str (core/schema/rollout.py:54-55); RunRecord.tokens and RolloutRecord.timings are dicts |
| AUD-F3 | partly | core Provenance unchanged (core/prov.py:19-25); studio stamps sha and dirty flag (studio/runs.py:54-111) |
| AUD-F6, RES-CAP-1 | partly | MediaRef/ImageRef (core/schema/actions.py:60-121, e063fc3); no ordered content parts, no raw arrays |
| AUD-F7, DOC-open-1 | partly | requires + SPDX check (studio/pack.py:504,617-621); no judge pin, adopted_from or schema files; F1-F7: 5 partly, 2 not done |
| DOC-open-3 | partly | importer reports baseline_network_mode (interop/harbor/task_io.py:99-100); untested; task04 baseline still public |
| DOC-gate-1 | partly | holds in behaviour [ran: p02_ledger H, I]; suite round-trips 2 types, no warnings filter (tests/unit/test_core.py:69-72) |
| ARCH-M0-3 | partly | as-built note ARCHITECTURE.md:29; the M0 card (:432-433) still says DEPEND on Harbor |
| RES-FORGE-1 | partly | items 1, 2, 4 built (core/schema/task.py:35-54,109-141; core/schema/milestone.py:46); item 3 verify-side only; item 5 not M0 |
| IMP-M0-2, ARCH-M0-2 | not done | prov optional (core/store/__init__.py:211); one producer, empty derived_from: §7 rule 4 unmet |
| IMP-M0-3, IMP-M0-7, NEW-5 | not done | no lock type; one config file, CRUCIBLE__* ignored (core/config.py:140-157); redacted() has no caller |
| IMP-M0-8 | not done | 10 interop and 5 connector pointers hard-coded in core/registry.py:140-189 |
| IMP-M1-8, NEW-4 | not done | no resource fields (core/schema/run.py:157-176); the runner sets started_at = ended_at (runtime/run.py:281-283) |
| IMP-M1-9, NEW-9 | not done | no output_schema or annotations (core/schema/actions.py:30-50); read_only unverified; to_openai skips openai_name [ran: p02_ledger L] |
| AUD-F4, AUD-F5 | not done | tags are an upsert, 1 row after 2 writes (core/store/__init__.py:274-282); one concrete Store, TraceStore reads `_db` |
| DOC-open-2 | not done | task04's verifier sets only a dockerfile (../migration_tasks/task04_pcl_gpu_cuda12/task.toml:25-27); Harbor has no such field |
| ARCH-M0-1, DES-M0-1, SCOUT-3, SCOUT-4 | not done | no bench schema; 4 versioning conventions; scheme in no id: DESIGN.md:2769-2774 overclaims |
| RES-ACC-2 | not done | no accept_check, mutation_operator or defect_detector kind; accept keeps its own REGISTRY (accept/checks/base.py:161) |
| SCOUT-1, NEW-6 | not done | a half-failed plugin keeps its names; `replace=True` silently overrides a builtin [ran: p02_ledger A] |
| SCOUT-2, NEW-2 | not done | validate_id uncalled; attempt split into path segments (runtime/journal.py:90-96) [ran: p02_ledger J] |
| SCOUT-5 | not done | no GC; narrowed in verification: runtime stores are per attempt (§7 rejected), the forget race remains (R-24) |
| SCOUT-6 | not done | reset_registry, reset_config, set_store uncalled; no reset for _UPCASTERS or _REGISTERED [ran: p02_ledger N] |
| NEW-1, NEW-7, NEW-8 | not done | D4 race test patches an uncalled copyfile [ran: p02_ledger C]; no test runs lint-imports; docs/modules/M00-core.md:3,60 stale |
| NEW-3 | not done | NaN inside Any-typed fields is accepted and written as null [ran: p02_ledger K] |

## 3. Comparison questions

| # | Question | Axis |
|---|---|---|
| Q1 | How does a persisted record carry its version, migrate older and refuse newer, and is there one convention? | evolution and versioning |
| Q2 | How is a content id computed, is the scheme inside it, and do ids survive an added field? | identity and reproducibility |
| Q3 | How are writes atomic under threads and processes, reads verified, crashes recovered? | correctness (storage) |
| Q4 | How are mutable names layered over content: history, compare-and-swap, revisions? | evolution and versioning (refs) |
| Q5 | Is the store an interface with pluggable backends, and how is content collected or retained? | performance and scale |
| Q6 | How are plugins discovered, and how are compatibility, collisions, partial failure and reporting handled? | extensibility |
| Q7 | How is config layered, validated and resolved for secrets, and is it recorded per run? | operability |
| Q8 | What does a record say about what produced it, and can two runs be checked for equivalence? | identity and reproducibility |
| Q9 | How are failures classified (vocabulary, blame, retry, masking), kept consistent and extended? | correctness (failures) |
| Q10 | How are tools, messages and media typed and mapped to MCP, OpenAI, ATIF, with usage and timing? | interop |
| Q11 | How are external ids validated before they become paths, and secrets kept out of dumps and records? | security |

## 4. Repo triage

33 reference folders (9 primary, 14 secondary, 10 not relevant) and 1 cited but absent; commit = HEAD read; licence from LICENSE.

| Repo | Class | Reason | Commit | Licence |
|---|---|---|---|---|
| harbor-framework-harbor | primary | typed contracts, digest locks, package refs, plugins, layered config, secret templating | dc933be7 | Apache-2.0 |
| agentenv-framework | primary | store ABCs with CAS, config refs, plugins; source of Crucible's store port | 015c89b | Apache-2.0 + NOTICE |
| primeintellect-ai-verifiers | primary | typed Trace and Usage, version+commit stamp, saved resolved config | e22533430 | MIT |
| nvidia-nemo-gym | primary | record contracts, failure kinds, layered config, versioned digests, provenance | 7ece89985 | Apache-2.0 |
| meta-pytorch-openenv | primary | versioned documents, committed schemas, digests, per-call capture, redaction | 8d3455ab | BSD-3-Clause |
| huggingface-repo2rlenv | primary | versioned records, protocol-tagged digests, budget CAS, id checks | 27d04ee | Apache-2.0 |
| preferencemodel-karotte | primary | typed events and usage, sorted plugins with providers, durable writes | 31c693e | MIT |
| infra-nvidia-nemo-rl | primary | config inheritance, fail-closed fingerprint, retention, failure classifier | 925f70d55 | Apache-2.0 |
| infra-rllm-org-rllm | primary | core data shapes, plugin loaders, migrating registries, TraceStore protocol | 4e828c81 | Apache-2.0 |
| primeintellect-ai-prime-agent | secondary | Rust agent: session migrations, scoped settings, typed usage | 3e0893260 | MIT |
| primeintellect-ai-prime-environments | secondary | 109 environments; lower-bound pins as a counter-example | cf220ee | Apache-2.0 |
| primeintellect-ai-research-environments | secondary | commit-pinned dataset registry; CI smoke installs | f63f07c8a | Apache-2.0 |
| adithya-s-k-fineenvs | secondary | store backends, pointer-last publish, retention, source hash | 5e0b46e | Apache-2.0 |
| swe-agent-mini-swe-agent | secondary | multi-spec config layering; resolved configs in trajectories | 04d809ce | MIT |
| xiaomimimo-verl | secondary | verl fork: run manifest with code state, sampling fix | a2ad9f61 | Apache-2.0 |
| infra-volcengine-verl | secondary | config drift gate, typed timing, plugin policy, retention | 2035f58d | Apache-2.0 |
| infra-primeintellect-ai-prime-rl | secondary | layered TOML and CLI, per-attempt resolved configs | 1e5318108 | Apache-2.0 |
| infra-openrlhf-openrlhf | secondary | byte-budget retention; a secret-handling counter-example | dc2a7ad | Apache-2.0 |
| infra-huggingface-trl | secondary | deprecation windows, versions in the model card, None unscorable | d9b9f39b | Apache-2.0 |
| infra-inclusionai-areal | secondary | monotonic LATEST pointer, redacted config, tokenizer fingerprints | aedc21ec | Apache-2.0 |
| infra-thudm-slime | secondary | frozen resume identity, immutable verified checkpoints | 05e3b017 | Apache-2.0 |
| infra-novasky-ai-skyrl | secondary | SQL CAS, capture block, one hashing module, boundary ids | a1e34dbf | Apache-2.0 |
| infra-alibaba-roll | secondary | completion manifest, offload backends; a hash framing defect | 0f37fe13 | Apache-2.0 |
| servicenow-enterpriseops-gym | not relevant | plain dataclasses and JSON configs; no versions or hashes | 271f2c3 | Apache-2.0 |
| swe-bench-swe-smith | not relevant | overwriting profile registry, 8-char ids; keeps only | 9b74ac0 | MIT |
| veds12-git-swarm | not relevant | static project page | 9f66af0 | none |
| karpathy-autoresearch | not relevant | a prompt and a training script; crash scored 0 | 228791f | none (README: MIT) |
| xiaomi-mimo-v2.6-rl-oss | not relevant | dataset dump, unversioned formats; still downloading (619 to 676) | not git | none (README: apache-2.0) |
| infra-moonshotai-checkpoint-engine | not relevant | weight transfer; rollback by name is a counter-example | 84631a2 | MIT |
| hf-space-fineenvs-multi-harness-rl | not relevant | article Space; no M0 code | not git | CC-BY-4.0 |
| hf-space-fineenvs-simulation-rl-environments | not relevant | article Space; pin 46bb2be uncheckable | not git | CC-BY-4.0 |
| hf-space-rl-environments-guide | not relevant | article Space; no M0 code | not git | CC-BY-4.0 |
| 08-hf-rl-env-datasets | not relevant | Harbor task data; task.toml version in 13/13 vs 0/37 | not git | none |
| nano-rlm | absent | cited at c38e20b in THIRD_PARTY.md; no folder | none | n/a |

## 5. Findings per repo

### harbor-framework-harbor (pin a2fafbc, 45 commits; M0: resume equality, src/harbor/job.py:273-276; dataset-ref pinning)
- Locks: equality keys over sha256 digests and resolved configs, refused on mismatch (src/harbor/models/job/lock.py:60-288;
  src/harbor/job.py:925-949); flaw: schema_version and concurrency are inside the keys (lock.py:282-288).
- Config files deep-merged, then CLI (src/harbor/cli/config_sources.py:19-51); secrets templated by a serializer
  (src/harbor/models/trial/config.py:157-164); trial dirs scrubbed by value (src/harbor/trial/trial.py:1118-1170);
  exclude_if for additive fields (models/trial/config.py:134-149); tag/revision/digest grammar (models/package/version_ref.py:20-119).
- Weaker: schema_version never read (models/task/config.py:802); unverified cache. Corrections: the M0 audit's "migrations"
  are field shims (task/config.py:503-552); no hash-scheme tag; no image digest or dirty flag in the lock; tag CAS not
  established; manifest_hash lacks the packager's file selection (publisher/packager.py:13-71).
- Verdicts: better on locks, layering, secrets (R-1, R-23, R-8, R-5); of interest R-16, R-18, R-26; Crucible better on Q1-Q3, Q9.

### agentenv-framework (pin 041afd7, 62 commits: nothing Crucible ported changed)
- Four store ABCs built from `module:Class` (src/agent_env/store/object_store/object_store.py:57-213; config/loader.py:212-249);
  compare_and_swap on a counter (store/document_store/document_store.py:534-559); check_id before writes (:374-458).
- Plugins: contested names withheld from every claimant, core requirement checked before import, `plugin check` exits 1
  (plugins/_registration.py:95-187; plugins/_requirements.py:29-55); config layers traced (config/runtime.py:110-165).
- Weaker: no fsync, unverified reads (store/object_store/local/store.py:106-135). Corrections: D12's NFC cite is character
  refusal (NFC: utils/build_context.py:135); D14's migrations are type aliases (artifact/registry.py:105-161); D15's fsync is
  Crucible's own; F3's provenance.py traces config layers; the mirror in core/store/__init__.py:15-16 does not exist (R-22).
- Verdicts: better on backends, CAS, plugins, config tracing (R-22, R-16, R-13, R-14, R-23); Crucible better on Q3, Q9.

### primeintellect-ai-verifiers (pin ae819de, 25 commits: trace.py, errors.py, utils/version.py unchanged)
- Each Trace carries version and commit (verifiers/v1/trace.py:91-101; verifiers/v1/utils/version.py:10-44); ModelCall types
  usage, finish_reason, sampling, timing (trace.py:153-176; verifiers/v1/types.py:125-188).
- The resolved config is saved and a changed resume refused, naming keys (verifiers/v1/cli/eval/main.py:87-113).
- Weaker: version unchecked, extra=ignore (trace.py:38-39); task_key admits NaN and hashes idx (verifiers/v1/task.py:46-48);
  errored traces read 0.0 (trace.py:472-474). Corrections: F3's reference has no dirty flag; F6's has no audio; nothing moved.
- Verdicts: better on per-call accounting and recorded config (R-19, R-1); its git guard fails Crucible (R-11).

### nvidia-nemo-gym (pin 3ef478d, 139 commits: owner retire, eval plan, fingerprint v3; failure_kinds.py unchanged)
- Digest version inside the hashed bytes (nemo_gym/token_id_capture/records.py:58-62); cross-version matches refused
  (token_id_capture/fingerprint.py:46-48); every path builder validates ids (token_id_capture/store.py:55-99).
- Store Protocols with `module:Class` backends (token_id_capture/protocols.py:136-341); layered config recorded per run
  (global_config.py:496-582; orchestration/executors/base.py:71-77); commit with -dirty (orchestration/jobs.py:78-127).
- Weaker: 6 version conventions; NaN allowed (eval_plan.py:66-67); silent shadowing (discovery.py:41-49); leaf-only redaction
  [ran: p_ng01]. Corrections: Crucible's kind port dropped fullmatch (failure_kinds.py:54-60); discovery.py:41 has no strict mode.
- Verdicts: better on scheme-tagged digests, store protocols, id checks (R-21, R-22, R-4); port its grammar (R-27).

### meta-pytorch-openenv (pin 49aa302, 179 commits: runtime validation, post-run digest recheck, redacted bundles)
- Committed schema files with a stale test (tests/test_validation/test_schema_sync.py:13-30), though `--fix` forces no bump.
- source_digest re-checked after a run (src/openenv/validation/runner.py:36-77,398-432); sent vs requested sampling per call
  (src/openenv/core/harness/capture/graph.py:183-188); duplicate-key JSON refused (src/openenv/discovery/serialization.py:19-40).
- Weaker: no migration; last-wins discovery (src/openenv/auto/_discovery.py:464-515). Correction: the M0 audit's manifest cite
  lacks SPDX (src/openenv/discovery/models.py:257-262) and adopted_from (absent).
- Verdicts: better on schema snapshots, digests, sent sampling, strict parsing, kinds (R-18, R-2, R-3, R-7, R-28).

### huggingface-repo2rlenv (pin 5555443, 6 commits: .env via python-dotenv, src/repo2rlenv/cli.py:27-47)
- Budget CAS in BEGIN IMMEDIATE with append-only history and idempotent evidence (src/repo2rlenv/campaigns/budget.py:109-212);
  digests stored with their protocol, old records verified under it (src/repo2rlenv/tasksmith/author/artifact.py:163-195).
- Weaker: Literal versions, fixed .tmp, a lock never recovered (src/repo2rlenv/emitter/bundle.py:183-210).
- Verdicts: better on CAS (R-16), protocol-tagged digests (R-21), .env parsing (R-29); Crucible better on Q1, Q3, Q9.

### preferencemodel-karotte (HEAD equals the pin)
- Sorted entry points (src/karotte/hardware.py:50-55); duplicate ids an error naming both, provider stamped dist==version
  (src/karotte/templates.py:65-86); typed usage with an estimated flag (src/karotte/schemas/transcript.py:140-151).
- Parent fsync of created dirs (src/karotte/durable_write.py:10-35); surrogates escaped at boundaries (src/karotte/text.py:4-24).
- Verdicts: better on providers, durable dirs, surrogates (R-13, R-25, R-10, R-19); weaker on Q1, Q9 (no versions; crash = 0).

### infra-nvidia-nemo-rl (pin c9193bc, 14 commits: only algorithms/single_controller.py in M0 paths)
- Fail-closed snapshot identity: full config minus dotpath globs, refusal listing changed paths
  (nemo_rl/algorithms/single_controller_utils/rollout_checkpoint.py:62-204,490-550); retirement checks
  (nemo_rl/utils/outdated_config_checks.py:14-136); a cause-walking classifier (nemo_rl/experience/failures.py:141-264).
- Weaker: only manifest v3 is accepted; actors get the full environment (nemo_rl/environments/utils.py:131-136).
- Verdicts: better on resume identity (R-1); retention and classifier to M12 and M10; Crucible better on Q1-Q3, Q11.

### infra-rllm-org-rllm (HEAD equals the pin)
- TTL snapshot index with derived refcounts (rllm/sandbox/snapshot.py:156-359); TraceStore Protocol, two backends
  (rllm-model-gateway/src/rllm_model_gateway/store/base.py:6-49); fail-loud env readers (rllm/env.py:1-46).
- Weaker: other versions overwritten as v1 (rllm/data/dataset.py:297-302); raw keys at rest (rllm/eval/config.py:570-590).
- Verdicts: env knobs (R-15) and the migration shape (R-12); GC not needed (§7); the 2026-10-11 study's cites hold.

### Secondary patterns
- primeintellect-ai-prime-agent (pin 2ee7e623a): step session migrations (crates/pa-core/src/session/mod.rs:177-274); protocol
  range check (crates/pa-daemon/src/protocol.rs:226-242).
- primeintellect-ai-prime-environments: lower-bound verifiers pins on 105 v0-API environments broke with v0's removal (R-14).
- primeintellect-ai-research-environments (pin d4fb80a): registry.json pins dataset@version to git commits.
- adithya-s-k-fineenvs: pointer written last, then retention (tooling/hf-rl-explorer/app/indexer.py:275-319); source hash (R-11).
- swe-agent-mini-swe-agent: `-c` specs merged in order (src/minisweagent/config/__init__.py:12-61); configs in trajectories.
- xiaomimimo-verl: run manifest with commit, untracked files, diff (recipes/write_run_manifest.py:69-201), leaking judge keys.
- infra-volcengine-verl (pin 5eb4826): config drift gate (scripts/generate_trainer_config.sh:44-51; R-18); 9 tool parsers, not 4.
- infra-primeintellect-ai-prime-rl (pin f7e8ab6): configs/attempt_N, atomic latest (src/prime_rl/utils/pathing.py:29-57);
  correction: utils/config.py:15-22 only dumps config, with no sha, and never moved.
- infra-openrlhf-openrlhf: byte-budget retention, but eviction precedes the save (openrlhf/utils/deepspeed/deepspeed.py:685-758).
- infra-huggingface-trl (pin 72cf37d): deprecation windows (trl/trainer/dpo_config.py:345-370); None rewards unscorable.
- infra-inclusionai-areal (pin 298412a): monotonic LATEST (areal/utils/checkpoint_pointer.py:246-296) [ran: p_tb03].
- infra-thudm-slime (pin c933787): resume identity frozen unless mutable (R-1).
- infra-novasky-ai-skyrl (pin 3d0ae22): conditional-UPDATE CAS (R-16); capture block (R-17).
- infra-alibaba-roll (pin 581046a): unframed object hash collides (roll/utils/hash_utils.py:21-88) [ran: p_tc01].

## 6. Gap matrix

● built and tested · ◐ built with a stated weakness · ○ absent.

| Q | Crucible | Harbor | agentenv | verifiers | NeMo-Gym | OpenEnv | repo2rlenv | karotte | NeMo-RL | rLLM |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | ◐ 4 conventions, 0 upcasters | ○ never read | ○ aliases only | ○ unchecked | ◐ 6 conventions | ◐ schema files, no migration | ○ literals | ◐ template migrations | ◐ exact v3 | ◐ overwrites newer |
| Q2 | ◐ scheme not in id | ○ no NFC, no scheme | ◐ scheme in digest | ○ NaN, index hashed | ◐ version in bytes; NaN | ◐ no NFC | ◐ protocol stored | ○ none | ◐ scheme in payload | ○ 12-hex |
| Q3 | ◐ verified; forget race, 0755 | ○ unverified cache | ◐ no fsync | ○ no fsync | ◐ unverified reads | ○ in place | ○ fixed tmp | ◐ fixed tmp | ◐ no payload hash | ○ fixed tmp |
| Q4 | ○ upsert, no history | ◐ revisions; CAS unknown | ● versions, CAS | ○ version stripped | ◐ conditional drop | ○ resolved once | ● CAS, history | ○ none | ○ highest step | ○ last writer wins |
| Q5 | ○ one class, no GC | ◐ ABC; cache wipe | ◐ 4 ABCs, no GC | ○ not established | ● Protocols, retire | ○ none | ○ none | ○ none | ● union retention | ● TTL, refcounts |
| Q6 | ◐ non-atomic, order-dependent | ◐ last wins | ● staged, withheld | ○ builtin shadows | ○ shadowing | ◐ strict graders only | ○ fixed map | ◐ sorted, providers | ○ fixed registry | ○ shadowing |
| Q7 | ◐ one file | ● merged, templated | ◐ traced, not recorded | ◐ compared, unredacted | ◐ recorded; leaf-only mask | ○ ad hoc | ◐ file + CLI | ◐ no layers | ◐ secrets dumped | ◐ raw keys |
| Q8 | ○ version only | ◐ digest locks, no dirty flag | ◐ git per image | ◐ commit per trace | ◐ commit + dirty | ◐ digest, sent sampling | ◐ wheel digest | ○ model only | ● fail-closed fingerprint | ○ meta only |
| Q9 | ● blame, masking | ○ class names | ○ 2 error types | ○ error reads 0.0 | ◐ fullmatch names | ○ free strings | ○ no masking | ○ crash scores 0 | ◐ 2 buckets | ○ masking off |
| Q10 | ◐ usage dicts | ◐ typed usage, MIME | ◐ typed Usage | ◐ no audio | ● unknown stays unknown | ◐ sent vs requested | ○ none | ● typed usage | ○ dicts | ○ dicts |
| Q11 | ◐ validate_id uncalled | ◐ value scrub | ● check_id first | ○ no scrubber | ● ids validated | ● strict parse | ◐ boundary checks | ◐ off cmdline | ○ full env leaked | ○ raw keys |

## 7. Recommendations, ranked

Four verify batches confirmed 34 items from 10 clusters; three cross-batch duplicates are merged: the run-config document
and RunLock (R-1), two typed-Usage items (R-19; its sampling part joins R-3), two forget/put transactions (R-24). Ranking:
impact, then risk-adjusted effort; within a tie, security and data integrity first. Order of work: R-18 (frozen schemas)
before the format changes R-1, R-2, R-3, R-17 and R-19; R-12 (index version) before R-16 (its rev column is migration 1 to
2) and R-20; R-11, R-13 and R-15 feed R-1; R-1, R-2, R-11 and R-19 ship as one crucible.run/v2 with one upcaster, the first
in production (IMP-M0-6).

| R | Title (short) | Kind | Impact | Effort | Risk |
|---|---|---|---|---|---|
| R-1 | RunLock: fail-closed run identity, resolved config, launch record | defect | H | M | M |
| R-2 | Task and World digests, re-checked before grading | defect | H | M | M |
| R-3 | Typed per-call SamplingParams: what was sent | defect | H | M | M |
| R-4 | Validate ids where they become paths | defect | M | L | L |
| R-5 | Scrub resolved secret values by value | defect | M | L | L |
| R-6 | Every store private by default | defect | M | L | L |
| R-7 | Strict document loader: duplicate keys refused | defect | M | L | L |
| R-8 | Provider secrets: env: references only, never dumped | defect | M | L | L |
| R-9 | Reproduce agentenv's NOTICE; check NOTICE coverage | production gap | M | L | L |
| R-10 | Validated implies serializable (lone surrogates) | defect | M | L | L |
| R-11 | BuildInfo: guarded code identity | defect | M | L | L |
| R-12 | Version the store index (PRAGMA user_version) | production gap | M | L | L |
| R-13 | Staged, order-independent plugin registration | defect | M | L | L |
| R-14 | Core-compatibility gate at load; registry --check | production gap | M | L | L |
| R-15 | One typed table of CRUCIBLE_* knobs | better design | M | L | L |
| R-16 | Tags: CAS, revisions, append-only history | defect | M | M | L |
| R-17 | RolloutRecord.capture: engine, logprobs mode, tokenizer identity | production gap | M | M | L |
| R-18 | Freeze every persisted format per version | production gap | M | M | M |
| R-19 | Typed Usage, Timing, finish reason; drop RunRecord.tokens | defect | M | M | M |
| R-20 | Lossless document storage; NFC for identity only | defect | M | M | M |
| R-21 | Identity contract: self-describing content ids | better design | M | M | M |
| R-22 | Store protocols, verified reads, conformance kit | production gap | M | M | M |
| R-23 | Layered crucible.toml with a per-key scope policy | defect | M | M | M |
| R-24 | Blob deletion and index writes in one transaction | defect | L | L | L |
| R-25 | fsync the parent of every created store directory | defect | L | L | L |
| R-26 | Canonical media MIME and kind/MIME agreement | defect | L | L | L |
| R-27 | Failure-kind grammar under fullmatch; warn once on read | defect | L | L | L |
| R-28 | Declared registry kinds | defect | L | L | L |
| R-29 | Parse .env with python-dotenv's grammar | defect | L | L | L |
| R-30 | Drop core.ids.manifest_hash | better design | L | L | L |
| R-31 | Drop the unread Config.backends table | defect | L | L | L |

### R-1. RunLock: a fail-closed run identity, the resolved run configuration, and a launch record per invocation
- **Kind** defect · **Status** carry-over IMP-M0-3, IMP-M0-7 (record half), AUD-F3, NEW-5; corrects runtime/journal.py:5-6
  (no "code/framework pins" exist). Merges PRIME-C3, SW-6, TC-3, H1, TA-1, TB-1, TA-3, MV-1, NG-8, AE-4.
- **Problem.** identity_doc() is an allowlist (runtime/run.py:55-58): 10 of 18 RunSpec fields are outside it, and tasks,
  policies and harnesses enter as display ids (`single_turn:<model>` at any temperature, runtime/single_turn.py:192-205;
  `native:llm-tool@1` at any limits); base_url and request_defaults (actors/llm.py:311) are outside too. RunJournal compares
  only that hash (runtime/journal.py:72-79). Scenario: a run resumes after its endpoint, thinking flag, temperature (0.0 to
  1.0), max_steps (30 to 300) and grader changed; pass@k averages both regimes and nothing records the change.
- **Reference.** NeMo-RL hashes the full config minus a dotpath denylist and refuses with up to 20 changed paths
  (infra-nvidia-nemo-rl/nemo_rl/algorithms/single_controller_utils/rollout_checkpoint.py:62-204,490-550); slime freezes every
  field not declared mutable (infra-thudm-slime/slime/ray/training_recovery.py:60-151); Harbor's digest locks put
  schema_version and concurrency in the key (harbor-framework-harbor/src/harbor/models/job/lock.py:60-288), which R-1 avoids.
- **Recommendation.**
  - CompatIdentity (crucible.compat_identity/v1): scheme, excluded dotpath globs, projected doc; fingerprint() over canonical
    JSON; diff() masks secrets. Invariant: every input counts unless the scheme's denylist excludes it.
  - RunLock (crucible.run_lock/v1) projects the whole RunSpec: task digests (R-2), policy config() (base_url, resolved model,
    request_defaults, sampling sent), harness config hash, packs, BuildInfo (R-11), providers (R-13), result knobs (R-15).
    Denylist: concurrency, budget, wall clock, backoff, health thresholds, paths, created_at.
  - RunLaunch (crucible.run_launch/v1) per invocation: scrubbed argv, denylisted values, Config.redacted(), BuildInfo.
  - Protocol: lock.json is written before the first dispatch; a changed fingerprint raises RunLockMismatch naming up to 20
    dotpaths and the remedy (new run_id or a recorded --allow-change); legacy dirs accepted once; RunRecord gains lock and
    launch ids; M12 checkpoints store the train-config identity (train/state.py:125-127 checks none today).
- **Reasoning.** Exclusion fails closed where the allowlist failed open. Hashing everything would refuse the resume after
  increase_limit that runtime/run.py:12 promises; SkyRL saves configs without comparing. §7 rule 4; P16.
- **Established by.** Code read on both sides; [ran: p_v3_run_identity; v4/p_v4_verify §1; p_tb01; p_tr01; p_prime_02].
- **Impact / Effort / Risk.** H: resumes mix regimes silently · M: config() on adapters, an M12 hook · M: some resumes that
  pass today are refused until an override is named.
- **Mode.** PATTERN (NeMo-RL, slime, Harbor: Apache-2.0; verifiers: MIT).
- **Acceptance.** A coverage test over dataclasses.fields(RunSpec); the probe scenarios are refused naming dotpaths; a budget
  change resumes as launch 2; an alias retarget changes the fingerprint, created_at does not; no env: secret in run files.

### R-2. Task and World content digests on RunRecord, taken at dispatch and re-checked before grading
- **Kind** defect · **Status** correction of core/schema/run.py:158-159 (documented as hashes; producers store names);
  merges openenv-hf:C2.
- **Problem.** runtime/run.py:282, train/world_env.py:309 and interop/nemo_gym/client.py:152 store display names
  (`harbor/<name>`, interop/harbor/inplace.py:130). Harbor's hash is computed on import (interop/harbor/task_io.py:85) and used
  nowhere else. Scenario: an author edits tests/test.sh during a multi-hour run; samples before and after share one task id,
  pass@k pools two tasks, and a resume accepts the changed task.
- **Reference.** OpenEnv's source_digest hashes sorted paths and bytes, refuses symlinks and races, and is re-taken after the
  run; a change skips dependent checks (meta-pytorch-openenv/src/openenv/validation/runner.py:36-77,398-432).
- **Recommendation.**
  - Core ids.tree_digest(path) returns `sha256:<hex>` with OpenEnv's refusals; a Harbor task uses Harbor's hash (P18).
  - RunRecord gains task_digest and world_digest (crucible.run/v2; None only for legacy records).
  - M10: RunTask.digest() is taken at plan time (into R-1), recorded at dispatch and recomputed before grading; a mismatch
    masks the attempt as non-retryable crucible:task_changed, never graded. M13: bench refuses two digests under one id.
- **Reasoning.** A plan-time digest misses edits during a run, and a copy per task costs more; the recheck takes 0.6-18.2 ms
  per task directory up to 36 MB. §7 rule 4; P16 (a task changed under the policy is not the policy's failure).
- **Established by.** Code read on the cited lines; grep finds the hash only at task_io.py:85,92; [ran: v4/p_v4_verify §1;
  v4/p_v4_digest_cost].
- **Impact / Effort / Risk.** H: pass@k and resumes pool task versions · M: digest() across adapters · M: a harness that
  writes into its task directory would trip the check (not established).
- **Mode.** PATTERN (OpenEnv: BSD-3-Clause, PORT with notice allowed; FineEnvs carry-over: Apache-2.0).
- **Acceptance.** An edit between dispatch and grade yields an ungraded task_changed; bench refuses mixed digests; a seed's
  digest equals Harbor's task_hash; the toy and coding suites show zero task_changed.

### R-3. Typed per-call SamplingParams on Turn: the parameters sent, apart from those requested
- **Kind** defect · **Status** carry-over AUD-F2 (sampling); merges openenv-hf:C3, mimo-verl:MV-3, PRIME-C1's sampling part.
- **Problem.** Turn.sampling (core/schema/rollout.py:54) is a dict copied from RunSpec.sampling, default `{}`
  (runtime/run.py:268; runtime/record.py:133-158); the policy's temperature (actors/policy.py:375) and request_defaults
  (actors/llm.py:311-313) never reach it. The trainer's temperature guard skips an absent value (train/trainer.py:745-748) and
  top_p defaults to 1.0 (:356). Scenario: policy at 0.7, provider top_p 0.9: the record holds `{}`, is train-valid, needs no
  candidate sets, and a 1.0 trainer stays silent. MV-3's verl top_p case is refuted (train/bridges/verl_agent_loop.py:124-129),
  but the bridge's other parameters go unrecorded (:150).
- **Reference.** OpenEnv records the submitted params over a closed key set beside the requested ones and param_fixes
  (meta-pytorch-openenv/src/openenv/core/harness/capture/graph.py:183-188); verl hard-codes repetition_penalty=1.0
  (infra-volcengine-verl/verl/experimental/agent_loop/agent_loop.py:606-612), fixed only in the MiMo fork.
- **Recommendation.**
  - Core SamplingParams (temperature, top_p, top_k, min_p, penalties, max_tokens, seed, stop, extra; None = not sent; legacy
    dicts mapped); Turn.sampling = sent, requested_sampling only when changed, param_fixes.
  - LLMClient (after the request_defaults merge), the TITO engine and the verl bridge return the applied params; policies keep
    them per call; runtime/record.py reads them; RunSpec.sampling goes only to R-1's lock.
  - A trainable turn without temperature is not train-valid; any truncation needs candidate sets; rollout v1.1 (R-18).
- **Reasoning.** A mirrored declaration is blind to request_defaults, and request-only recording diverged in verl. The
  rollout is the trainer's only input (DESIGN §10.3). Trade-off: records without temperature stop being train-valid.
- **Established by.** Code read on the cited lines; [ran: v4/p_v4_verify §2], the guard run as its verbatim expression
  (torch absent).
- **Impact / Effort / Risk.** H: the off-policy guard reads a declaration · M: one model, three builders, a version bump ·
  M: stored rollouts without temperature stop being train-valid.
- **Mode.** PATTERN (OpenEnv: BSD-3-Clause; verl: Apache-2.0).
- **Acceptance.** A policy at 0.7 under `{}` records 0.7 and a 1.0 trainer refuses it; request_defaults top_p 0.9 is recorded
  and needs candidates; a fake verl server receiving repetition_penalty 1.1 records it; legacy dicts still read.

### R-4. Validate ids where they become paths
**Kind** defect · **Status** carry-over AUD-D16, SCOUT-2, NEW-2 (AE-5, NG-3, TC-5, H11) · **Impact/Effort/Risk** M/L/L · **Mode** PATTERN (NeMo-Gym: Apache-2.0)
**Problem.** No code calls validate_id (core/ids.py:35); RunJournal turns run_id and attempt into paths (runtime/journal.py:72-97), so `../escaped` leaves runs/. **Reference.** Every NeMo-Gym path builder validates (nvidia-nemo-gym/nemo_gym/token_id_capture/store.py:55-99). **Recommendation.** A one-segment id type and a contained child_path(). **Reasoning.** P17. **Established by.** [ran: verify1/v1_ids]. **Acceptance.** `../x`, `a/b`, NUL, symlinks refused.
### R-5. Scrub resolved secret values by value
**Kind** defect · **Status** new (H2, C6) · **Impact/Effort/Risk** M/L/L · **Mode** PATTERN (Harbor: Apache-2.0)
**Problem.** The scrubber knows vendor shapes only (core/secrets.py:11-67): a 16-char gateway key resolved via env: survives into Failure.reason. **Reference.** Harbor replaces resolved values (harbor-framework-harbor/src/harbor/trial/trial.py:1118-1170). **Recommendation.** resolve_secret registers values of 8+ chars, replaced before the patterns. **Reasoning.** §7 rule 5. **Established by.** [ran: verify1/v1_secrets]. **Acceptance.** No resolved value in any record.
### R-6. Make every store private by default
**Kind** defect · **Status** new (AE-6) · **Impact/Effort/Risk** M/L/L · **Mode** PATTERN (agentenv: Apache-2.0)
**Problem.** Stores follow the umask (core/store/__init__.py:121-131): attempt stores are 0755, the index 0644. **Reference.** agentenv-framework/src/agent_env/store/local_state.py:20-37 makes 0700 dirs. **Recommendation.** Root 0700, index 0600, a .gitignore, an opt-out; journals too (M10). **Reasoning.** M3 chose owner-only. **Established by.** [ran: verify2/v2_ae6_sw7_store_fs]. **Acceptance.** Under umask 022 others cannot read a blob.
### R-7. Strict document loader: duplicate keys refused
**Kind** defect · **Status** new (C5); corrects core/schema/base.py:4 · **Impact/Effort/Risk** M/L/L · **Mode** PORT with notice (OpenEnv: BSD-3-Clause)
**Problem.** Loaders keep the last duplicate key (core/schema/base.py:49-59): a seed world.yaml plus `network: public` loads public; 13 sites call yaml.safe_load. **Reference.** meta-pytorch-openenv/src/openenv/discovery/serialization.py:19-40 refuses duplicates. **Recommendation.** core.serialize loaders refusing duplicates; ruff TID251 bans others. **Reasoning.** P21. **Established by.** [ran: verify1/v1_dupkeys]. **Acceptance.** The probe case raises.
### R-8. Provider secrets: env: references only, never dumped
**Kind** defect · **Status** new (H3, PRIME-C5; NEW-5) · **Impact/Effort/Risk** M/L/L · **Mode** PATTERN (Harbor: Apache-2.0)
**Problem.** Keys and headers accept literals that model_dump_json emits (core/config.py:24-27); headers are never resolved (actors/llm.py:222-224). **Reference.** Harbor's serializer templates secrets (harbor-framework-harbor/src/harbor/models/trial/config.py:157-164). **Recommendation.** env:NAME only; one resolved_headers(); dumps keep references. **Reasoning.** Opt-in redaction is forgotten. **Established by.** [ran: p_v3_config a, b]. **Acceptance.** A literal key fails to load.
### R-9. Reproduce agentenv's NOTICE; check NOTICE coverage
**Kind** production gap · **Status** new (AE-11) · **Impact/Effort/Risk** M/L/L · **Mode** PORT obligation (Apache-2.0 §4(d))
**Problem.** Core ports agentenv code (THIRD_PARTY.md:46-49), but its NOTICE appears nowhere in THIRD_PARTY.md or LICENSES/. **Reference.** agentenv-framework/NOTICE, in license-files (agentenv-framework/pyproject.toml:10-11). **Recommendation.** Ship it verbatim, plus a notices manifest with digests and a test. **Reasoning.** Stops the next miss. **Established by.** `git show 041afd7:NOTICE`; grep. **Acceptance.** The file equals upstream.
### R-10. Validated implies serializable (lone surrogates)
**Kind** defect · **Status** new (SW-4, OSError case corrected) · **Impact/Effort/Risk** M/L/L · **Mode** PATTERN (karotte: MIT)
**Problem.** A Failure naming a non-UTF-8 file validates but cannot hash or dump, and RunJournal.finish then raises unmasked (runtime/run.py:185-199). **Reference.** preferencemodel-karotte/src/karotte/text.py:4-24 escapes at boundaries. **Recommendation.** Core safe_text (from world/world.py:70-110) on free text; identities refuse surrogates. **Reasoning.** P16. **Established by.** [ran: verify2/v2_sw4_surrogates]. **Acceptance.** A property test: refuse or serialize.
### R-11. BuildInfo: a guarded code identity
**Kind** defect · **Status** carry-over AUD-F3 (PRIME-C2 corrected; AE-4, C8, NG-8, MV-1) · **Impact/Effort/Risk** M/L/L · **Mode** PORT with notice (NeMo-Gym: Apache-2.0)
**Problem.** Provenance has crucible_version only (core/prov.py:19-25), 0.1.0 after 159 commits; studio reports a foreign enclosing repo (studio/runs.py:46-73). **Reference.** NeMo-Gym trusts git only for a tracked package (nvidia-nemo-gym/nemo_gym/orchestration/jobs.py:78-127). **Recommendation.** build_info(): source digest, guarded commit, scoped dirty flag, uv.lock hash, plugins. **Reasoning.** §7 rule 4. **Established by.** [ran: prime/p_prime_01_build_sha; v4/p_v4_verify §8]. **Acceptance.** A foreign repo gives None.
### R-12. Version the store index (PRAGMA user_version)
**Kind** production gap · **Status** new (TC-2) · **Impact/Effort/Risk** M/L/L · **Mode** PATTERN (rLLM: Apache-2.0)
**Problem.** The index DDL is unversioned CREATE IF NOT EXISTS (core/store/__init__.py:40-52): a new column never arrives, a newer index opens silently. **Reference.** rLLM migrates on read but overwrites newer files (infra-rllm-org-rllm/rllm/data/dataset.py:283-318). **Recommendation.** Append-only migrations under BEGIN IMMEDIATE; newer raises. **Reasoning.** The index is not rebuildable. **Established by.** [ran: verify1/v1_store_index]. **Acceptance.** Fixtures migrate.
### R-13. Staged, order-independent plugin registration
**Kind** defect · **Status** carry-over IMP-M0-5, SCOUT-1, NEW-6 (H8, AE-2, SW-3) · **Impact/Effort/Risk** M/L/L · **Mode** PATTERN (agentenv: Apache-2.0)
**Problem.** Plugins register live in install order (core/registry.py:100-112): a half-failed one keeps its names, contested names follow sys.path. **Reference.** agentenv-framework/src/agent_env/plugins/_registration.py:95-187 withholds contested names. **Recommendation.** Sorted, staged, all-or-nothing merge; source = dist==version. **Reasoning.** P19. **Established by.** [ran: p_v3_registry a-c]. **Acceptance.** Order-free describe().
### R-14. Core-compatibility gate at load; registry --check
**Kind** production gap · **Status** carry-over AUD-F1, IMP-M0-5 (AE-3) · **Impact/Effort/Risk** M/L/L · **Mode** PORT with notice (agentenv: Apache-2.0)
**Problem.** No plugin's crucible requirement is checked (core/registry.py:100-112): a `<0.2` pack loads on 0.2.0, and broken pointers fail at first get. **Reference.** agentenv-framework/src/agent_env/plugins/_requirements.py:29-55 checks first. **Recommendation.** Skip as incompatible-core; stable codes; --check in CI. **Reasoning.** pip warns only at install. **Established by.** [ran: p_v3_registry e, f]. **Acceptance.** --check exits 1.
### R-15. One typed table of CRUCIBLE_* knobs
**Kind** better design · **Status** new (TC-4, narrowed) · **Impact/Effort/Risk** M/L/L · **Mode** PATTERN (rLLM: Apache-2.0)
**Problem.** 17 names are read unchecked at 22 sites; CRUCIBLE_LLM_CACHE=reply disables the cache and pays twice (actors/llm.py:203,264-270). **Reference.** infra-rllm-org-rllm/rllm/env.py:1-46 reads fail-loud. **Recommendation.** core/env.py with types and choices; result knobs enter R-1. **Reasoning.** Workers inherit env. **Established by.** [ran: v4/p_v4_verify §6]. **Acceptance.** `reply` raises.
### R-16. Tags: CAS, revisions, append-only history
**Kind** defect · **Status** carry-over AUD-F4 (H5, AE-1, NG-10, SW-1, TC-1) · **Impact/Effort/Risk** M/M/L · **Mode** PATTERN (SkyRL, repo2rlenv: Apache-2.0)
**Problem.** Tags are an unconditional upsert (core/store/__init__.py:274-282): racing TraceStore.put calls regress the head, and purge_superseded deletes the newer blob. **Reference.** infra-novasky-ai-skyrl/skyrl/tinker/api.py:1145-1176 updates on a version. **Recommendation.** tag(expect=...) with rev and tag_history, TagConflict, a tag grammar. **Reasoning.** §7 rule 4. **Established by.** [ran: verify2/v2_tags_cas]. **Acceptance.** N ingests keep N events.
### R-17. RolloutRecord.capture: engine, logprobs mode, tokenizer identity
**Kind** production gap · **Status** new (TB-2, TC-6) · **Impact/Effort/Risk** M/M/L · **Mode** PORT with notice (AReaL: Apache-2.0)
**Problem.** Rollouts keep ids and logprobs but no tokenizer, template, engine or logprobs mode (core/schema/rollout.py:30-188). **Reference.** infra-inclusionai-areal/areal/dataset/swe_sft/pipeline.py:88-136 splits token map and render. **Recommendation.** Optional capture with a two-part TokenizerId; gates in M12. **Reasoning.** Restarts are invisible. **Established by.** [ran: v4/p_v4_verify §7; p_tb02]. **Acceptance.** p_tb02 differs in render only.
### R-18. Freeze every persisted format per version
**Kind** production gap · **Status** carry-over AUD-D8, D13, D14, SCOUT-3, DOC-gate-1, DES-M0-1 (C1, MV-2, AE-9, PRIME-C6, H7) · **Impact/Effort/Risk** M/M/M · **Mode** PATTERN (OpenEnv: BSD-3-Clause)
**Problem.** Two golden hashes guard drift; Turn.agent and trainable joined literal rollout/v1 (core/schema/rollout.py:39,76), so a c051249 reader rejects HEAD records. **Reference.** meta-pytorch-openenv/tests/test_validation/test_schema_sync.py:13-30 fails on stale schemas. **Recommendation.** One VersionedDocument convention; exclude_if additions; frozen schemas and fixtures per version. **Reasoning.** Tolerant readers fail open. **Established by.** [ran: prime/p_prime_04_rollout_skew; verify1/v1_schema_gate]. **Acceptance.** Unbumped edits fail.
### R-19. Typed Usage, Timing, finish reason; drop RunRecord.tokens
**Kind** defect · **Status** carry-over AUD-F2, IMP-M1-8, NEW-4 (H6, PRIME-C1, NG-7, MV-4, SW-8) · **Impact/Effort/Risk** M/M/M · **Mode** PORT with notice (verifiers: MIT)
**Problem.** No producer fills RunRecord.tokens (core/schema/run.py:173); bench reads 0 and counts replays as live (bench/metrics.py:140-160); cached prompts cost 2.29x (actors/llm.py:246-255). **Reference.** primeintellect-ai-verifiers/verifiers/v1/types.py:125-188 splits cached input. **Recommendation.** Disjoint Usage with a source and coverage; Timing; FinishReason; run/v2. **Reasoning.** Budget stops need true numbers. **Established by.** [ran: verify1/v1_usage; v4/p_v4_verify §5]. **Acceptance.** Provider shapes agree.
### R-20. Lossless document storage; NFC for identity only
**Kind** defect · **Status** new (NG-4) · **Impact/Effort/Risk** M/M/M · **Mode** PATTERN (NeMo-Gym: Apache-2.0)
**Problem.** Documents are stored in the NFC-folded hashing form (core/store/__init__.py:211-213): NFD text returns as NFC, colliding keys merge, NFD events lose signals. **Reference.** nvidia-nemo-gym/nemo_gym/token_id_capture/fingerprint.py:118-123 folds only to compare. **Recommendation.** Store sorted lossless JSON; refuse NFC key collisions. **Reasoning.** event_key stays stable. **Established by.** [ran: verify2/v2_ng4_lossless_store]. **Acceptance.** NFD round-trips.
### R-21. Identity contract: self-describing content ids
**Kind** better design · **Status** carry-over AUD-D13, SCOUT-4 (AE-7, NG-5, SW-2, TC-8) · **Impact/Effort/Risk** M/M/M · **Mode** PATTERN (agentenv: Apache-2.0)
**Problem.** HASH_SCHEME (core/ids.py:22) enters no hash, so the golden test's remedy (tests/unit/test_core_audit.py:261) is a no-op and a canonical change breaks World.restore. **Reference.** agentenv-framework/src/agent_env/bundle/ledger.py:144-157 hashes its scheme. **Recommendation.** Bare hex = scheme 1, later `c<n>:<hex>`; verify_id. **Reasoning.** Ids are embedded. **Established by.** [ran: verify2/v2_scheme_ids]. **Acceptance.** Old WorldRefs restore.
### R-22. Store protocols, verified reads, conformance kit
**Kind** production gap · **Status** carry-over AUD-F5 (AE-8, NG-11); corrects core/store/__init__.py:15-16 · **Impact/Effort/Risk** M/M/M · **Mode** PATTERN (agentenv: Apache-2.0)
**Problem.** One concrete Store (core/store/__init__.py:120); four readers use blob_path and skip the hash check (world/state/base.py:92-100). **Reference.** agentenv-framework/tst/store/conformance.py:390-450 races every backend. **Recommendation.** BlobStore and DocIndex Protocols, a backend kind, a kit. **Reasoning.** No framework in core. **Established by.** [ran: verify2/v2_ae8_blob_path]. **Acceptance.** No blob_path outside core.
### R-23. Layered crucible.toml with a per-key scope policy
**Kind** defect · **Status** carry-over IMP-M0-7 (PRIME-C4, H4, NG-9, AE-12) · **Impact/Effort/Risk** M/M/M · **Mode** PATTERN (Prime Agent: MIT)
**Problem.** The first file found wins (core/config.py:140-157): a crucible.toml above cwd sends the user's API key to its base_url and lifts the budget from 1.0 to 500.0. **Reference.** primeintellect-ai-prime-agent/crates/pa-core/src/settings/manager.rs:587-600 keeps security keys global. **Recommendation.** user < project < env < --set; operator-only credentials. **Reasoning.** Never redirect keys. **Established by.** [ran: p_v3_config d]. **Acceptance.** The parent file is refused.
### R-24. Blob deletion and index writes in one transaction
**Kind** defect · **Status** new (NG-6; TA-2, TC-7 narrowed); corrects core/store/__init__.py:9 · **Impact/Effort/Risk** L/L/L · **Mode** PATTERN (agentenv: Apache-2.0)
**Problem.** forget_doc checks, commits, then unlinks (core/store/__init__.py:258-272); a put or tag in between leaves rows without a blob. **Reference.** agentenv-framework/src/agent_env/store/document_store/sqlite_document_store.py:354-362 writes under BEGIN IMMEDIATE. **Recommendation.** Decide and rename inside one transaction, unlink after COMMIT; put and tag re-check theirs. **Reasoning.** Precedes exposing purge. **Established by.** [ran: verify2/v2_ng6_forget_race; p_v3_store]. **Acceptance.** Stress keeps rows resolvable.
### R-25. fsync the parent of every created store directory
**Kind** defect · **Status** new (SW-7) · **Impact/Effort/Risk** L/L/L · **Mode** PORT with notice (karotte: MIT)
**Problem.** A new blobs/<ab>/ is never synced in its parent (core/store/__init__.py:158-176), so a power loss can drop an indexed blob. **Reference.** preferencemodel-karotte/src/karotte/durable_write.py:10-35 syncs parents. **Recommendation.** fsync each created directory's parent. **Reasoning.** POSIX gives no ordering. **Established by.** [ran: verify2/v2_ae6_sw7_store_fs], code path only. **Acceptance.** A recording test sees blobs/ synced.
### R-26. Canonical media MIME, kind/MIME agreement
**Kind** defect · **Status** new (H9) · **Impact/Effort/Risk** L/L/L · **Mode** PORT with notice (Harbor: Apache-2.0)
**Problem.** MIME is unchecked against kind (core/schema/actions.py:60-101): an image tagged audio/wav goes out as image_url, audio/wave as `wave` (actors/policy.py:118-120). **Reference.** harbor-framework-harbor/src/harbor/models/trajectories/content.py:10-44 aliases MIME. **Recommendation.** Normalize and require major type = kind. **Reasoning.** Adapters need not repeat it. **Established by.** [ran: verify1/v1_media]. **Acceptance.** Inputs normalised or refused.
### R-27. Failure-kind grammar under fullmatch; warn once on read
**Kind** defect · **Status** carry-over AUD-D17 (NG-1) · **Impact/Effort/Risk** L/L/L · **Mode** PORT with notice (NeMo-Gym: Apache-2.0)
**Problem.** Kind names match with `$` (core/schema/run.py:93,100): `pack:kind\n` and `9probe:_x` register; a misspelled read kind gets blame unknown. **Reference.** nvidia-nemo-gym/nemo_gym/failure_kinds.py:52-60 uses fullmatch. **Recommendation.** fullmatch on register; warn once on unknown reads. **Reasoning.** Refusal strands records. **Established by.** [ran: v4/p_v4_verify §3]. **Acceptance.** Bad names refused.
### R-28. Declared registry kinds
**Kind** defect · **Status** new (C4); corrects core/registry.py:8-9 · **Impact/Effort/Risk** L/L/L · **Mode** PATTERN (OpenEnv: BSD-3-Clause)
**Problem.** Any kind registers (core/registry.py:69-74); segmenter and replayer are read, never registered (workflow/steps.py:180-185), so `segmentor` registers unused. **Reference.** meta-pytorch-openenv/src/openenv/validation/graders/__init__.py:91-111 refuses unknown names. **Recommendation.** A KINDS table; undeclared kinds raise. **Reasoning.** P19. **Established by.** [ran: p_v3_registry d]. **Acceptance.** A scan finds all kinds declared.
### R-29. Parse .env with python-dotenv's grammar
**Kind** defect · **Status** new (SW-5; DEPEND changed to PORT) · **Impact/Effort/Risk** L/L/L · **Mode** PORT (python-dotenv: BSD-3-Clause)
**Problem.** The parser splits at the first `=` (core/config.py:123-137): `KEY=sk-...  # note` keeps the comment and calls fail 401. **Reference.** huggingface-repo2rlenv/src/repo2rlenv/cli.py:25-47 uses python-dotenv. **Recommendation.** Vendor its 191-line parser, interpolation off. **Reasoning.** ARCHITECTURE.md:173 limits DEPEND. **Established by.** [ran: p_v3_config e]. **Acceptance.** Fixtures parse alike.
### R-30. Drop core.ids.manifest_hash
**Kind** better design · **Status** carry-over AUD-D18 (H10); corrects core/ids.py:1-8, THIRD_PARTY.md:85 · **Impact/Effort/Risk** L/L/L · **Mode** DEPEND (Harbor: Apache-2.0)
**Problem.** manifest_hash (core/ids.py:77-82) is test-only; production takes Harbor's id via interop/harbor/task_io.py:85. **Reference.** harbor-framework-harbor/src/harbor/publisher/packager.py:13-71 selection keeps changing. **Recommendation.** Delete it; fix the docs. **Reasoning.** P18. **Established by.** grep. **Acceptance.** No manifest_hash left.
### R-31. Drop the unread Config.backends table
**Kind** defect · **Status** new (AE-10) · **Impact/Effort/Risk** L/L/L · **Mode** PATTERN (agentenv: Apache-2.0)
**Problem.** Config.backends (core/config.py:66) has no reader and accepts any table, so `[backends.harbor] docker_host` is ignored. **Reference.** agentenv-framework/src/agent_env/config/describe.py:261-290 warns on unread keys. **Recommendation.** DROP the field. **Reasoning.** Restores extra=forbid. **Established by.** [ran: p_v3_config c]. **Acceptance.** A backends table fails.

### Rejected in verification

| Candidate ids | Reason |
|---|---|
| trainers-a:TA-4 | No config key was ever removed or renamed (git log -p), so retirement checks wait; parts also rejected: PRIME-C6's tolerant reader, H11's casefold key, AE-5's rewriting, NG-7's None-absorbs, C5's size cap, MV-4's with_speed case. |
| nemo-gym:NG-12, openenv-hf:C7 | Worlds use per-attempt stores (runtime/single_turn.py:77), train/chat.py:319 keeps media in memory; growth is in run dirs (M10). |
| mimo-verl:MV-5 | R-13 fixes collisions whatever is selected; registration only binds pointers (core/registry.py:49-59). |
| trainers-a:TA-2, trainers-c:TC-7 (GC part) | Episodes use per-attempt or temporary stores (train/world_env.py:279-280); growth unmeasured; the race survives as R-24. |
| nemo-gym:NG-2 | Journal attempts come only from attempt_id() over 16-hex ids (runtime/keys.py:41-43); no code passes a foreign record. |

## 8. What we already do better (keep)

- Versioned documents refuse newer versions, upcast older majors and reject unknown keys (core/schema/base.py:28,94-143);
  elsewhere versions go unread or match exactly (harbor-framework-harbor/src/harbor/models/task/config.py:802; verifiers, NeMo-Gym,
  OpenEnv, repo2rlenv, NeMo-RL, AReaL, slime, skycap), or are overwritten (infra-rllm-org-rllm/rllm/data/dataset.py:297-302).
- Canonical JSON with NFC, finite floats and -0.0 folded (core/canonical.py:20-39), against plain json.dumps in seven repos;
  ROLL's unframed hash collides (infra-alibaba-roll/roll/utils/hash_utils.py:21-88).
- Staged, fsynced, verified blobs and atomic journals (core/store/__init__.py:81-208; runtime/journal.py:61-69), against
  harbor-framework-harbor/src/harbor/tasks/client.py:396-444 and the fixed-tmp or in-place writes in §6 Q3.
- Blame, retry and masking with scored_reward() (core/schema/run.py:20-202), against class names, 0.0 on error
  (primeintellect-ai-verifiers/verifiers/v1/trace.py:472-474), crashes scored 0 (karotte, autoresearch, EnterpriseOps), MiMo's -999.
- A registry that refuses collisions and cites failures (core/registry.py:66-112), against last-wins or shadowing
  (harbor-framework-harbor/src/harbor/cli/plugin_registry.py:6-10; verifiers, NeMo-Gym, OpenEnv, rLLM, swe-smith, verl, ROLL).
- env:NAME secrets with inherited redaction (core/config.py:84-137), against nvidia-nemo-gym/nemo_gym/secret_utils.py:59-81
  and unredacted dumps in eight repos; id rules beyond agentenv-framework/src/agent_env/store/ids.py:32-86; capped pack pins.
- Typed ToolSpec, ToolCall, MediaRef and fault (core/schema/actions.py:30-145), against name-and-schema tools (agentenv,
  OpenEnv), text+image parts (verifiers), infra-volcengine-verl/verl/tools/schemas.py:21-38, dict messages (NeMo-RL, TRL).

## 9. Open questions

| Question | Tried | Missing; who can supply it |
|---|---|---|
| Is Harbor's server-side tag move atomic or CAS? | harbor-framework-harbor/src/harbor/db/client.py:720-740 | the hosted RPC body; Harbor maintainers |
| Which input_audio formats do target providers accept, and does any echo a full key in errors (R-26, R-5)? | actors/policy.py:108-124; Harbor MIME tables | provider API contracts |
| Can a timed-out attempt finish after its retry; what follows when finish raises, and can a non-UTF-8 name reach CustodyViolation (R-10); are secrets on argv; are HTTP errors pre-classified; which PolicyRef ids use aliases; are images digest-pinned; is GPU-host code checked; does a monitor flag a logprob-mode change (R-17)? | runtime/journal.py:93-110; runtime/run.py:139-142,185-199; world/custody.py:47-52; runtime/infra.py:20-112; runtime/gpu.py:63-71; train/mismatch.py:138-263 | M10, M1, M2, M7, M12 code; those audits |
| How do prime-pydantic-config (verifiers, prime-rl) and OmegaConf struct mode (verl, AReaL) merge and reject keys? | call sites; the prime-rl submodule is empty | package sources or an install |
| Do verifiers/prime-rl mask errored traces; do NeMo-Gym agent_request_failed rows reach RetryRule; does wandb keep OpenRLHF's literal key? | primeintellect-ai-verifiers/verifiers/v1/trace.py:472-474; interop/nemo_gym/client.py:95-158 | the trainer side, the caller trace, wandb |
| straw-queue atomicity and GC; slime's flock scope; mimoagent's judge mapping; a test of Prime Agent's v1 to v2 migration | slime and fork usage read; grep | sources absent or submodules uninitialised |
| Are there put_doc callers or blob keys outside src/ and packs/; how fast does a training store grow; does bench use digests? | grep | scripts, notebooks, a populated data dir |
| Does pydantic 2.12 have exclude_if; may core depend on python-dotenv; did CI flag 24f3b46; should blame=None pack kinds be fixed; what is the Space at 46bb2be? | 2.14.1 only; ARCHITECTURE.md:173; p_agentenv_01 | a 2.12 install, owner decisions, CI logs, the Space's origin |

## 10. Handoff: findings outside this module

| To | Finding |
|---|---|
| M1 world | The gVisor tag is reused per version, so an edited Dockerfile never rebuilds (world/sandbox.py:138-139,230-236; content-addressed tags at harbor-framework-harbor/src/harbor/utils/container_cache.py:10-59); fs fingerprints skip NFC (world/state/snippets/fs.py:11-14); ToolResult.error codes missing from INFRA_ERRORS are scored as policy (world/world.py:49,543-546); env isolation as in meta-pytorch-openenv/src/openenv/validation/runtime/contracts.py:189-225; re-count MiMo envs (ARCHITECTURE.md:38). |
| M1, M7 | Images are not pinned by digest (agentenv pins at agentenv-framework/src/agent_env/store/image_store/registry_api.py:35-50); task04's verifier needs a prebuilt docker_image (../migration_tasks/task04_pcl_gpu_cuda12/task.toml:25-27). |
| M2 actors | Cached tokens at the input rate, two cache conventions (actors/llm.py:246-255; actors/rlm/context.py:119-126); 78-char names (actors/simulated.py:173-174); never retry an accepted paid call (infra-novasky-ai-skyrl/skycap/src/skycap/mirror.py:24-28); docs/research/improvements/M00-M02.md:55 miscredits AReaL's call ids. |
| M3, M4 | TraceStore's head read-merge-tag via _db loses updates (connectors/trace_store.py:43-110); event_key skips NFC (core/schema/trace.py:52-62) [ran: p_mv03]; cargo and unittest log parsers (huggingface-repo2rlenv/src/repo2rlenv/log_parsers/cargo_parser.py:71); M3/M16: reconcile token counts with ATIF (meta-pytorch-openenv/src/openenv/harbor/atif.py:232,498-554). |
| M8, M10 | forge/orchestrator.py:180-193 and runtime/keys.py:20-21 hash json.dumps(default=str) outside core canonical JSON (R-21). |
| M10 runtime | Wire R-1 to R-3 (runtime/run.py:55-58,268,282) and a real start time (NEW-4, :281-283); regrade from a lock diff (harbor-framework-harbor/src/harbor/job_diff.py:183-375); boundary() matches exact class names, skips __cause__ and calls any non-agent OSError transport_unreachable (runtime/infra.py:93-112; infra-nvidia-nemo-rl/nemo_rl/experience/failures.py:141-264); classify harness events only (harbor-framework-harbor/src/harbor/agents/installed/base.py:231-262); journal rows skip scrub_obj (runtime/journal.py:99-101); terminal() is unfenced (agentenv-framework/src/agent_env/task/store.py:420-474); attempt dirs need retention (runtime/run.py:159); pre-check compatibility (nvidia-nemo-gym/nemo_gym/global_config.py:760-819). |
| M10, M11 | agent_request_failed is never retried here but retried per occurrence in NeMo-Gym (core/schema/run.py:66; nvidia-nemo-gym/nemo_gym/failure_kinds.py:21-26); GPU-host code identity is unchecked (runtime/gpu.py:63-71). |
| M11 interop | No ids from id(row) (servicenow-enterpriseops-gym/evaluate.py:297-298); task.toml versions vary (13/13, 0/37) and HF cache metadata holds commit and etag; MiMo's rubric_weight and points are ignored (interop/mimo/check.py:67); TRL's None rewards map to masking but its HarborEnv scores 0.0 (infra-huggingface-trl/trl/experimental/harbor/_env.py:93-95). |
| M11 engines | probe_logprobs_mode has no caller; raw logprobs are caught only for top_p < 1 (train/hf_engine.py:162; train/vllm_http.py:69-71); size binary sidecars (infra-novasky-ai-skyrl/skycap/src/skycap/record.py:1-190). |
| M12 train | Workers have no version handshake and swallow a refused /result (train/workers.py:151-160,325-334; primeintellect-ai-prime-agent/crates/pa-daemon/src/protocol.rs:226-242); the temperature guard skips empty sampling (R-3); restore checks no config (train/state.py:125-127); no fsync, and latest() picks an abandoned future [ran: p_tb03] (infra-thudm-slime/slime/data/checkpoint.py:340-393); step and rollout temp dirs are never pruned (train/runtime.py:1013-1016; train/world_env.py:279); a completion manifest as at infra-alibaba-roll/roll/pipeline/diffusion/dmd/dmd_pipeline.py:548-575; verl's fixed repetition_penalty passes through train/bridges/verl_agent_loop.py:148-155; cross-check masked rows (xiaomimimo-verl/recipes/general/infra_metrics.py:14-144); gate teachers on token_map (train/teachers.py:56-60). |
| M13 bench | Read typed Usage only, drop the key-guessing sum (bench/metrics.py:138-163); refuse two digests under one id (bench/run.py:137-144); report coverage (nvidia-nemo-gym/nemo_gym/rollout_collection.py:865-896). |
| M14 packs | PackManifest lacks AdoptedFrom, SPDX canonicalization, a deprecated state (core/schema/pack.py:111-127; nvidia-nemo-gym/nemo_gym/environment/manifest.py:143-392); template updates (preferencemodel-karotte/src/karotte/update_env.py:122-308); SPDX and framework_requirement checks (meta-pytorch-openenv/src/openenv/discovery/models.py:257-273); plugin API check (agentenv-framework/.github/scripts/check_plugin_api.py:1-50). |
| M15 studio | Provenance in studio/runs.py:45-111 is studio-only and trusts an enclosing foreign repo; replace it with R-11 and R-1. |
| M0, later GC | Persist the holder's boundary before releasing roots; retire a predecessor only once the successor is durable (infra-thudm-slime/slime/ray/training_recovery.py:389-454; infra-inclusionai-areal/areal/utils/checkpoint_pointer.py:282-296). |

## 11. Sources

- **Repos.** The 33 folders of §4 at the commits listed there; earlier sweep pins are named in §5 where they differ.
  python-dotenv 1.2.4 (uv.lock:2729-2736).
- **Crucible documents.** ARCHITECTURE.md (1-57, 133-331, 334-444, 772-841); DESIGN.md (79-112, 116-354, 2745-2775,
  3085-3140); THIRD_PARTY.md; docs/modules/M00-core.md; docs/audit/2026-10-12/M00-core.md; docs/research/improvements/
  (M00-M02.md, 00-index.md, 2026-10-14/00-plan.md); docs/research/m11-m12/00-source-pins.txt; m7-m8/02-forge-synthesis.md:784-791;
  m9-m10/01-acceptance.md:680-695,845-880; m1-taxonomy/07-capability-matrix.md.
- **Crucible code** (at c27f5ca). All 25 files of core/; the cited lines of runtime/, train/, actors/, connectors/, world/,
  interop/, bench/, studio/, signals/, workflow/; tests/unit/ (test_core, test_core_audit, golden_hashes.json); packs/.
- **Probes** (probes/). Scout, ledger: p01_core_contracts, p02_ledger, p03_store_pragmas. Sweeps: p_harbor_01-02,
  p_agentenv_01, prime/p_prime_01-04, p_ng01_nemo_gym_compare, p_openenv_hf, p_openenv_hf_dupkeys, sweep-swe-safety/p_swe_safety,
  p_mv01-04, p_tr01, p_tb01-04, p_tc01-02. Verify: verify1/v1_*, verify2/v2_*, p_v3_*, v4/p_v4_verify, v4/p_v4_digest_cost.
- **Work folder** (path in the header). 01-brief.md, 02-ledger.md, sweep-*.md, verify-1.md to verify-4.md, notes/, probes/,
  and the earlier long draft M00-core.long-v1.md.
