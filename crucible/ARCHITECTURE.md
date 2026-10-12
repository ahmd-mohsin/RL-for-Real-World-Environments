# Crucible — architecture, project structure and module plan (v0.1, 2026-10-09)

`DESIGN.md` says **what** Crucible is and **why**. This document says **how** we build it:
- what open-ended means in code;
- what we take from each existing framework, and how;
- the folder structure;
- the sixteen modules, M0–M15, and the order we build them in.

We build one module at a time. Every module has an **exit gate**, a measurement that must pass before the module counts as done. Every gate is checked on two domain packs: `coding` (the MVP) and `_toy`, a small non-coding pack that exists only to keep the core domain-agnostic.

---

## 0. Decisions made here

1. **The core is domain-agnostic. Coding is the first domain pack.** A toy helpdesk pack (two SQLite apps and a simulated customer) runs through every module in CI. If a coding assumption leaks into the core, the toy pack breaks.
2. **Every environment is a World**: Apps (state plus tools), Actors, an optional Clock, and Events. Verifiers read state only through State Adapters. One abstraction covers a single repo with a shell, and a finance organization with eight systems and four simulated departments.
3. **Each framework has one job:**

   | Framework | Job in Crucible |
   |---|---|
   | **Harbor** | Task and trajectory format, sandboxes, CLI agent harnesses, job runner, task-quality tooling |
   | **agentenv-framework** | Runtime for stateful worlds with many apps and actors (MCP, databases, websites, gateway clock and triggers, peers and humans); evaluation of enterprise agents over A2A |
   | **MiMo sidecar format** | Packaging standard for Apps |
   | **verifiers v1 → prime-rl** and the **XiaomiMiMo verl fork** | The two training paths |
   | **NeMo-Gym** and **OpenEnv** | Export targets, plus design patterns |
   | **SWE-smith** | Synthesis pipeline we port, with its history leak fixed |
   | **EnterpriseOps-Gym** | Patterns for horizontal worlds, and an external OOD benchmark |

4. **Only `src/crucible/interop/<framework>/` may import a framework.** Everything else talks to Crucible interfaces, so upgrading a framework touches one folder. An import-linter contract enforces this, with no exceptions. *(As built: the core does not import Harbor at all. Harbor's `TaskConfig` / `Task` / `Packager` validate task directories at the interop boundary, which keeps the core installable without Harbor's dependency tree.)*
5. **Tasks 01–04 are golden fixtures and the MVP showcase.** Their historical results (oracle 1.0, negatives 0, the Astra and sol gate profiles) must be reproduced exactly by the new code.
6. **The 16 modules are built in five stages, A–E**, which match DESIGN phases P0–P4 (§5).

**Build status (2026-10-09).**

| Module | Status | Gate | Spec + gate report |
|---|---|---|---|
| M0 core | built | passed | `docs/modules/M00-core.md` |
| M1 world | built | passed (5/5 seed worlds 3/3 on the RTX 5090; 266/266 MiMo envs; 100/100 AWM envs; agentenv clock and triggers). Added after the gate: Postgres, any-MCP-server, website, browser and hidden-check CI Apps; leak scrubber; isolation guard on every container backend (Harbor, agentenv sandboxes) with a red-team suite of 25+ attacks; GPU sandboxes everywhere (lab RTX 5090 via Harbor and agentenv, Modal/Daytona/GKE via Harbor, Modal via agentenv); one tool-call choke point; virtual time for tool modules; World audit (`crucible world validate`) | `docs/modules/M01-world.md` |
| M2 actors | built | passed (reviewer location precision 0.96 / 0.98 on real PCL / CCCL reviews; fabrication 0/225 sentences (quote mode) and 0% on toy; LLM agent + simulated customer episode reaches the expert end state). Hardened after the gate: episode limits and typed stop conditions, valid histories under parallel calls, status-code error classes, per-sample cache keys, simulated users that end conversations | `docs/modules/M02-actors.md` |
| M3 connectors | built | passed, offline with 0 API calls: 55 GPU-migration PR traces from git; the task 01–04 construction transcript (9,684 events); 11/11 frontier runs; CI reports, policy docs, toy ticket logs; 106/106 schema-valid; 20 planted canary formats and the workspace's 4 real credentials removed; idempotent re-ingest. Generic: file_drop, sql_audit_log, otel_import, capture_proxy (OpenAI + Anthropic, streaming), atif. Coding: github_pr (+ offline git discovery), claude_code_sessions, agent_runs, ci_logs, policy_docs. Not built: issues, gitlab (need APIs). Article cross-check (2026-10-09): answer-key probes in every agent trace (none exposed in 11 runs), egress self-test on 17 answer-source hosts with TLS/SNI, audit `network.answer_sources`, World traces record writes; seed-grader holes logged in DESIGN Appendix B.10 | `docs/modules/M03-connectors.md`, `docs/reviews/2026-10-09-articles-crosscheck.md` |
| M4 signals | in progress | audited 2026-10-12. Coding dev gate 1.0 / 1.0 / 1.0 on 50 events and 14 outcomes (stricter scoring, labels keyed by row ids); toy gate 1.0 / 1.0 / 1.0 on 84 events and 12 outcomes (scored for the first time). Neither is `confident` (95% lower bounds below the bars at this label count); held-out coding labels and a second labeller pending | `docs/modules/reports/m4_gate_coding_dev.json`, `docs/modules/reports/m4_gate_toy_test.json`, `docs/audit/2026-10-12/M04-signals.md` |
| M5 workflow | built, audited 2026-10-12 | 2 of 3: replay 91.9% (strict 91.0%) on 913 public OpenHands traces with the audited replayer, and 24/24 toy logs; tracker κ 0.87 vs manual stage labels (toy; 0.28 at the module's default filter threshold); rediscovery 4/6 families on 55 PCL GPU PRs, hand-audited (edge semantics: 1 genuine PR; bit-exactness not in upstream history) | `docs/modules/M05-workflow.md`, `docs/audit/2026-10-12/M05-workflow.md` |
| M6 credit | built, audited 2026-10-12 | R1–R7 (stricter: untested properties reported, crafted negatives must hit a gate, shallow progress bounded) pass on the toy and on 4 of 5 coding tasks; task01 cannot test R3 (no failing run). Dense credit is gate-aware, bounded and centred per episode on the trainer path too; `adv_signed` ported. S2 fails as specified: Φ at the midpoint 0.50 vs LLM judge 0.75 (state unchanged by then; with §6.5's keep rule no milestone type is kept at the midpoint), at the end Φ is better within a task (0.72 vs 0.67); C3 vs ΔΦ on task04 pending (GPU + ~$130). Native harness `BashLoopPolicy` (mini-swe-agent v2 loop) | `docs/modules/M06-credit.md`, `docs/audit/2026-10-12/M06-credit.md` |
| M7 verify | built, audited 2026-10-12 | audit: ~22 reproduced defects fixed (a candidate `pytest11` plugin forged hidden tests as root; candidate-triggered masks; perf, static, leak-audit and collect holes), 29 regression tests failing on the M6 commit; CPU and toy goldens re-run and passing (GPU goldens not re-run). toy golden ✅ (oracle 1, no-op 0, 7/7 negatives at their gate); A13 attack suite ✅ (honest 1, 11/11 cheats 0, 6 controls prove the cheats beat a seed-style grader); CPU golden ✅ (task04 36/36 recorded verdicts, task03 static 10/10; one contaminated historical grade masked as designed); GPU golden ✅ task04 8/8 (every recorded profile reproduced in the real verifier image; astra_v3b re-graded on an idle GPU 2026-10-11), task03 GPU golden open; tasks 01/01v2/02 plans not written. Record-relative score gates and honest abstention (P29) | `docs/modules/M07-verify.md`, `docs/audit/2026-10-12/M07-verify.md` |
| M8 forge | built, audited 2026-10-12 | audit: 22 reproduced defects fixed (oracle material could be written down into `public/`; A5 never saw the oracle or the ticket data; concurrent spend double-counted and an over-cap settle aborted every plan; trap tasks had empty instructions and docstring-word gates), 22 regression tests failing on the M7 commit; toy gates re-run (5/5); coding traps reworked, their behavioural M9 run (GPU) pending. Before the audit: toy ✅: the forged variant world `toy/helpdesk_v2` has 5/5 tasks passing the M9 subset (A1 A2 A3 A4-lite A5 A7 A8; trap tasks cite a resolvable M4 signal; 0 paid calls). Coding, CPU ✅: 4 span tasks from task04 v3 (people, NCV, KinFu, KinFu-LS; sibling forks excluded from starts) and traps for 4/5 seeds from M4 frontier-failure signals via C2. Behavioural acceptance ✅: 17/17 grades on the lab RTX 5090 (solutions 1.0; starts, unported mutants and traps 0.0 at their expected gates). task01 has no trap (its only frontier run passed). The cheapest LLM as Test writer failed closed | `docs/modules/M08-forge.md`, `docs/audit/2026-10-12/M08-forge.md` |
| M9 accept | built, audited 2026-10-12 | audit: 18 reproduced defects fixed (the agent-image leak check could never fail; `crucible accept`/`build` ran the model-free subset and called it accepted; A3 passed with zero mutants and hid process mutants as equivalent; the key ignored check code; repair could drop a guarding check), 16 regression tests failing on the M8 commit. Toy full contract: every check but A10 passes on 11/11; A10 pending (its recorded calibration runs are bound to an older grading environment; re-calibration needs paid runs). Power tests 13 proven, 10 per kit. Imports (I0–I10) and the seed adapters not built | `docs/modules/M09-accept.md`, `docs/audit/2026-10-12/M09-accept.md` |
| M10 runtime | built, audited 2026-10-12 | audit: 17 reproduced defects fixed (policy-triggered masks: a broken `.git`, an unreadable file, background load, a raised exception; a provider-echoed credential in 14 record files; one harness exception aborted a run; retries re-ran the agent; resume worked once and double-charged), 18 regression tests failing on the M9 commit. B.5 CPU injections and the long-horizon xl gate pass. R1 is a re-grade; R2/R3 and `harness/harbor.py` not built | `docs/modules/M10-runtime.md`, `docs/audit/2026-10-12/M10-runtime.md` |
| M13 bench | built (2026-10-11) | toy ✅: a generated benchmark reported with every §11.1 metric, reproducible byte for byte. Real models ✅ on toy: Claude Sonnet 5.5 + GPT-6.1 Sol + Qwen3-235B (open), k=2, $2.15 (`reports/m13_toy_live/`). Coding ✅ from recorded frontier runs on tasks 01–04 + spans (3 frontier models, 13 episodes, $0; `reports/m13_coding_recorded/`). Open model on the coding tasks: needs a live GPU-World run | `docs/modules/M13-bench.md` |
| M14 packs | built (2026-10-11) | `pack new demo` passes every check level ✅; `pack check --all --level fixture` passes for toy, coding, finance_ap ✅; `finance_ap` is a real AP-invoice World (3-way match, fraud controls) ✅. "Coding and toy pass every module gate" stays open while module gates are partial | `docs/modules/M14-packs.md` |
| M15 studio | built (2026-10-11) | toy ✅ through the CLI as in the README (`label`, `build`, `bench`, `model`, `export`); coding label ✅ (66 traces, 11 credited); explorer UI plugin ✅. The coding `build` (task02) needs the lab GPU | `docs/modules/M15-studio.md` |

Environments: verifiers v1 pins `mcp==2.0.0` and agentenv needs `mcp<2`, so the `verifiers` extra is declared as conflicting and runs in its own environment (`THIRD_PARTY.md`).

---

## 1. Open-endedness

### 1.1 What "any workflow" requires, and where the core handles it

| Axis of variation | Range | Handled by |
|---|---|---|
| Domain | code, finance ops, support, IT ops, legal, science, … | **Domain packs** (M14); the core imports none |
| State | files + git, SQL, documents, web UI, external APIs, processes/GPUs | **State Adapters** (M1): snapshot, restore, fingerprint, diff, query |
| Tool surface | shell, file edit, MCP, HTTP, browser, GUI | **App** tool surfaces. MCP is the default protocol, as in MiMo, EnterpriseOps-Gym, agentenv `MCPServerEnv`, and OpenEnv's "agent sees only MCP tools" rule |
| Actors | 1 to many: the policy, simulated people, peer agents, humans, systems | **Actor runtime** (M2) |
| Time | instant to days; replies that arrive later | **Clock and Events** (M1): agentenv's gateway clock (`urn:agentenv:clock/v1`) and triggers (`urn:agentenv:triggers/v1`), or an in-process equivalent |
| Coupling | vertical (one department) or horizontal (several organizations on one outcome) | several policy actors scored on a shared outcome; per-App verifiers routed by App id (EnterpriseOps-Gym routes by `gym_name`) |
| Permissions | who may call which tool | role-based tool access per Actor (agentenv `modify_env_tool_access`) |
| Horizon | H0–H5 | horizon ladder defined by the pack; sub-span tasks start from world snapshots |
| Expert signals | CI, reviews, overrides, approvals, recalls, escalations, reopen, SLA | generic signal taxonomy (M4), plus a mapper per pack |
| Outcome | "merged and not reverted in 30 days"; "invoice paid correctly and not recalled"; "ticket resolved and not reopened" | outcome labeler per pack; the validity gate (M6) uses it |

### 1.2 Core abstractions

```python
class StateAdapter(Protocol):                      # verifier-side only; never handed to the policy
    kind: ClassVar[str]                            # fs_git | sql | doc | http_mock | browser | process
    async def snapshot(self) -> StateRef: ...
    async def restore(self, ref: StateRef) -> None: ...
    async def fingerprint(self) -> str: ...        # canonical hash; equal hash = equal state
    async def diff(self, a: StateRef, b: StateRef) -> StateDiff: ...
    async def query(self, q: Query) -> Any: ...    # e.g. SQL, a path read, a build/test/perf run

class App(Protocol):
    id: str
    template: str                                  # e.g. "coding/git_repo", "common/mailbox"
    state: StateAdapter
    def tools(self) -> list[ToolSpec]: ...         # MCP tool schemas the policy may call

class Actor(Protocol):
    id: str
    kind: Literal["policy", "simulated", "peer", "human", "system"]
    async def act(self, obs: Observation) -> Action | None: ...

class World(Protocol):
    spec: WorldSpec                                # crucible.world/v1 (DESIGN §4.4)
    async def reset(self, seed: int, start: WorldRef | None = None) -> Observation: ...
    async def step(self, actor: str, action: Action) -> StepResult: ...   # a tool call or a message
    async def snapshot(self) -> WorldRef: ...      # every App + clock + pending events
    async def restore(self, ref: WorldRef) -> None: ...
    def verifier_view(self) -> VerifierView: ...   # State Adapters; separate container where required
```

Milestones, checks, rubric items and signals refer to state only through `VerifierView`, so credit assignment, verification and acceptance never need to know the domain.

### 1.3 One code path, three sizes of world

| | **World A** — task 04 | **World B** — coding, cross-team (H5) | **World C** — AP invoice exception (horizontal) |
|---|---|---|---|
| Apps | `repo` (fs_git), `gpu` (process) | `lib_repo`, `consumer_repo` (owned by another team), `ci` (CI as a tool, hidden checks), `review`, `issues` | `erp` (Postgres), `mail` (SQLite), `vendor_portal` (browser), `policy` (docs) |
| Actors | agent | agent, reviewer (simulated, grounded on review history), consumer team (simulated) | clerk (policy), procurement, vendor, approver (simulated, grounded) |
| Clock / events | none | CI latency; review rounds | 1 s = 1 virtual hour; vendor replies after 2 virtual days |
| Backend | Harbor + GPU Docker | agentenv (MultiEnv + gateway) or Harbor compose | agentenv (ServiceDB + MCP + website + gateway) |
| Example milestone | `{adapter: process, app: gpu, check: build}` | `{adapter: fs_git, app: consumer_repo, check: tests_pass}` | `{adapter: sql, app: erp, query: "select status from invoices where id=:id", expect: matched}` |
| Expert signal | review defect, revert | review request-changes, consumer breakage | clerk override, payment recall, escalation |
| Outcome label | merged, no revert in 30 days | both repos green, no hotfix in 30 days | paid correctly, no recall in 60 days |

### 1.4 The world model

The thesis asks for a world model that first imitates workflows in detail and then creates tasks. In Crucible it is three parts working together. Each is measured against recorded history:

| Part | Module | Fidelity measure |
|---|---|---|
| **Workflow model**: stages, milestones, transitions | M5 | replay fidelity (S1); accuracy predicting the next milestone on held-out traces |
| **Actor models**: how reviewers, users and other departments respond | M2 | agreement with the real response on held-out rounds; fabrication rate |
| **App models**: how systems respond to actions | M1 | state fingerprints after replaying recorded actions equal the recorded states |

The forge (M8) samples tasks only from parts that pass their fidelity gates.

### 1.5 How the core stays domain-agnostic
- **Import contract:** `crucible.{core, world, actors, connectors, signals, workflow, credit, verify, forge, accept, runtime, train, bench}` may not import `packs.*` or name any domain concept. This is checked in CI.
- **Layer order (enforced, M0 audit 2026-10-12):** a package imports only packages below it; packages on one line are
  independent. `lint-imports` checks it, and `.github/workflows/crucible.yml` runs it with ruff and the CPU suite on every
  push.

  ```
  cli
  studio                         (may import frameworks: hosts the agentenv explorer plugin and exports)
  interop                        (the only lower-level home of framework imports)
  train | bench
  accept                         (includes the RLM task-repair loop, accept/repair.py)
  forge
  runtime
  verify
  credit
  workflow                       (includes the potential Φ, workflow/potential.py)
  connectors | signals
  actors
  world
  core                           (includes the rollout record, agent-event helpers, canonical JSON)
  ```

  A lower layer that needs an adapter (ATIF export, the exact-token engine, a sandboxed role) resolves it through a
  `module:attr` pointer (`core.registry.load_impl`), never an import.
- **`packs/_toy` exercises every module, CPU-only, in under 5 minutes.**
  - World: a helpdesk with a `tickets` SQLite app and a `kb` doc app; actors are an agent and a grounded simulated customer.
  - Signals: reopen, escalation, CSAT.
  - Milestones: SQL predicates.
  - Pipeline: ingest → signals → workflow → credit → forge (span tasks) → accept → export → bench.
- **Every module gate has two halves:** one for `coding` and one for `_toy`.

---

## 2. How we use each framework

### 2.1 Integration modes

| Mode | Meaning | Rules |
|---|---|---|
| **DEPEND** | pip dependency, used through its public API | version pinned in `pyproject.toml`; only imported under `interop/<fw>/` |
| **PORT** | code copied into `src/crucible/_vendor/<name>/` and modified | upstream LICENSE copied; `ORIGIN.md` records repo, commit, paths and every change; Apache/MIT/BSD only |
| **PATTERN** | the design is re-implemented; no code copied | cite the source file in a docstring |
| **TARGET** | a format or API we export to | covered by a round-trip test in M11 |
| **DATA** | their tasks or environments used as test corpora | never in our training or benchmark splits without a contamination check |

### 2.2 Pinned sources

Paths are relative to `crucible/`. Abbreviations: `FW = ../RL-Environments/06-frameworks-and-env-code`, `MM = ../RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss`, `AE = ../agentenv-framework`.

| Source | Local path | Commit (date) | License | Modes |
|---|---|---|---|---|
| Harbor | `FW/harbor-framework-harbor` | a2fafbc (2026-10-03) | Apache-2.0 | DEPEND, TARGET |
| agentenv-framework | `AE` | 041afd7 (2026-10-08) | Apache-2.0 | DEPEND (+ our plugin), TARGET |
| verifiers (v1) | `FW/primeintellect-ai-verifiers` | ae819de (2026-10-04) | MIT | DEPEND, TARGET |
| prime-environments | `FW/primeintellect-ai-prime-environments` | cf220ee (2026-05-07) | Apache-2.0 | PATTERN |
| research-environments | `FW/primeintellect-ai-research-environments` | d4fb80a (2026-10-03) | Apache-2.0 | PATTERN, DATA |
| NeMo-Gym | `FW/nvidia-nemo-gym` | 3ef478d (2026-10-03) | Apache-2.0 | PATTERN, TARGET |
| OpenEnv | `FW/meta-pytorch-openenv` | 49aa302 (2026-10-02) | BSD-3-Clause | PATTERN, TARGET, DATA |
| EnterpriseOps-Gym | `FW/servicenow-enterpriseops-gym` | 271f2c3 (2026-08-19) | Apache-2.0 | PATTERN, DATA |
| SWE-smith | `FW/swe-bench-swe-smith` | 9b74ac0 (2026-03-22) | MIT | PORT |
| XiaomiMiMo verl fork (+ `third_party/mimoagent-osr`, MIT) | `FW/xiaomimimo-verl` | a2ad9f6 (2026-09-26) | Apache-2.0 | DEPEND, PORT, TARGET |
| MiMo v2.6 RL release | `MM` | — | Apache-2.0 (per README) | PORT (sidecar), TARGET, DATA |
| Karotte (Preference Model) | `FW/preferencemodel-karotte` | 31c693e (2026-10-08) | MIT | PORT (guard, no-follow collection), PATTERN (attack tests) |
| Repo2RLEnv (Hugging Face) | `FW/huggingface-repo2rlenv` | 5555443 (2026-09-30) | Apache-2.0 | PORT (leak patterns, git scrub) |

These pins go into `THIRD_PARTY.md` when M0 is built. Each version bump is a separate commit that re-runs every gate.

### 2.3 What we take from each framework

**Harbor** (backbone for format and execution)

| Take | Path (`FW/harbor-framework-harbor/`) | Mode | Used by |
|---|---|---|---|
| Task model, schema 1.4: `[environment]` gpus, network; `[verifier]` separate env, `[[verifier.collect]]`; `[[steps]]` with `min_reward`; multi-key `reward.json` | `src/harbor/models/task/config.py` | DEPEND | M0, M7, M11 |
| ATIF v1.8 trajectories (token ids, logprobs, subagents, `is_copied_context`) and prefix seeding via `trajectory.json` | `src/harbor/models/trajectories/` | DEPEND | M0, M3, M5, M6 (C3) |
| About 35 environment providers, and `import_path` for custom environment classes | `src/harbor/environments/{factory,capabilities}.py`, `docker/` | DEPEND | M1, M10 |
| — Local Docker has **no GPU path**; we add `HarborGpuDocker` (`--runtime=nvidia`, as the lab 5090 needs) and load it via `import_path` | `environments/docker/docker.py` (gap) | we build | M1 |
| 40+ installed agent harnesses with ATIF converters (claude_code, codex, mini_swe_agent, openhands, gemini_cli, …) | `src/harbor/agents/installed/` | DEPEND | M3 (session import), M10 (calibration across harnesses), M8 (builder roles) |
| Job / Trial runner, regrade, metrics, viewer | `src/harbor/{job.py,trial/,metrics/,viewer/}` | DEPEND | M10, M13 |
| rewardkit: criteria, LLM/agent judges with a guard prompt, aggregations, overlayfs isolation | `packages/rewardkit/` | DEPEND | M6 (judge items) |
| Task compiler: compiled-task template, auto-verifier from required artifacts and schemas | `src/harbor/compile/compiler.py` | DEPEND | M8 (scaffolding) |
| Quality checker and analyzer (`harbor check` rubric, `harbor analyze` reward-hacking analysis) | `src/harbor/cli/quality_checker/default-rubric.toml`, `src/harbor/analyze/` | DEPEND | M9 (A3, A7), M8 (Auditor) |
| Content-hash packaging and publishing | `src/harbor/publisher/packager.py` | PATTERN (ids) | M0 |
| ATIF → OpenTelemetry; LangSmith bridge | `packages/harbor-atif2otel`, `packages/harbor-langsmith` | DEPEND (optional) | M3 (enterprise agent telemetry) |
| **Skip:** pass@k for binary rewards only (ours handles graded and gated rewards) | | | |

**agentenv-framework** (backbone for open-ended worlds)

| Take | Path (`AE/src/agent_env/`) | Mode | Used by |
|---|---|---|---|
| `Env` types: `MCPServerEnv`, `MultiEnv` (MCP servers + websites), `ServiceDBEnv` (Postgres + pgweb + db-mcp), `WebsiteEnv`, `GatewayEnv` | `env/envs/` | DEPEND | M1 (agentenv backend) |
| Gateway virtual clock (`t0 + elapsed·rate`, up to 1 s = 1 day) and typed triggers (Action/State predicates; nl / tool / permission actions) | `env/gateway/{clock,triggers}.py` | DEPEND | M1 clock/events, M2 |
| Env-state providers (`local_postgres`); env snapshots into an `EnvironmentUniverseArtifact`; reset; clock sync; role-based tool access | `providers/env_state/`, `task_step/task_steps/{snapshot_env,reset_env,sync_env_clock,modify_env_tool_access}.py` | DEPEND | M1 snapshot/restore, permissions |
| A2A agents, peer routing, human A2A peer (`deploy_human_agent`), agent-state snapshot | `a2a_agent/`, `task_steps/{peer_agents,deploy_human_agent,snapshot_agent_state}.py` | DEPEND | M2 (peers, humans), M6 (C3 branches for A2A agents), M10 (evaluating a customer's deployed agent) |
| Human review gate (pause until an operator continues or aborts) | `task_steps/review.py` | DEPEND | M9 (manual sign-off for H4+), M8 (10% human audit) |
| Task DAG of steps with `depends_on`, step journal, retry/rollback; `eval run --k` | `task/`, `task_step/`, `eval/` | DEPEND | M10 |
| Sandbox providers (local, modal, e2b, sail_vm) with fallback chains | `providers/sandbox_providers/` | DEPEND + **our `gpu_docker` provider plugin** (the local provider passes no GPU flags) | M1, M10 |
| Plugin system (entry points, `module:Class` registries, `plugin check`), versioned artifacts and stores | `plugins/`, `artifact/`, `store/` | DEPEND (plugin) + PATTERN (registry) | M0, M11, M15 |
| Explorer web UI with plugins | `explorer/` | DEPEND | M15 |
| Harbor-style unit-test verifier with `reward_path` | `task_steps/verifiers/run_container_unit_tests_verifier.py` | PATTERN | M11 |
| — Gap: that verifier runs **in the agent's container**. We add a separate-verifier step | | we build | M11 |

**verifiers v1** (training path 1, to prime-rl)

| Take | Path (`FW/primeintellect-ai-verifiers/verifiers/v1/`) | Mode | Used by |
|---|---|---|---|
| `Task` / `Taskset` / `TaskData` (image, network, artifacts, timeouts, GPU resources); `HarborTaskset` already reads Harbor directories | `task.py`, `taskset.py`, `tasksets/harbor/` | DEPEND, TARGET | M11, M12 |
| Envs: `IsolatedVerifierEnv`, `UserSimEnv` ("the user is just another agent"), `AgenticJudgeEnv`, `BestOfNEnv`, `SingleAgentEnv` | `envs/` | DEPEND | M7, M2, M12 |
| `@vf.reward(weight)` / `@vf.metric`, i.e. reward channels; trace message graph with per-token `advantages` / `loss_weights` | `trace.py`, `graph.py` | DEPEND | M12 (dense-credit injection) |
| Harnesses (claude_code, codex, mini_swe_agent, terminus_2, …), runtimes (docker, modal, e2b, prime, apptainer, subprocess), interception proxy for tokens and logprobs, MCP toolsets | `harnesses/`, `runtimes/`, `interception/`, `mcp/` | DEPEND | M10, M12, M3 (capture proxy) |
| `vf-validate` (gold / no-op), `vf-replay`, `vf-eval` | `cli/` | PATTERN / DEPEND | M9, M13 |
| Rubric and reference judges | `judges/` | DEPEND | M6 |
| Proposer/solver learnability reward 4p(1−p) | `../environments/proposer_solver/` | PATTERN | M8 |
| Taskset adapters for NeMo-Gym, OpenEnv, TextArena | `tasksets/{nemo_gym,openenv,textarena}` | DEPEND | M11 (cross-checks) |

**prime-environments / research-environments**
- **PATTERN:**
  - Shaped coding rewards: `kernelbench` speedup tiers, `backend_bench` correctness × perf, `rust_cargo` with 7 shaped rewards.
  - Per-stage sub-scores in `senior_swe_bench`.
  - A code-review "taste" rubric at weight 0 (`judges/swe/rubrics.toml`), as rubric candidates for M6.
  - `restore_test_files` and default-deny egress, for M7.
  - Benchmark porting guides (`BENCHMARK_PORTING.md`, `HARBOR.md`) and the `skills/{evaluation,prime-tasks}` folder, for M13.
- **DATA:** SWE tasksets, and recursive task synthesis (36k Harbor tasks) as a reference corpus for M8.

**NeMo-Gym** (patterns and an export target)

| Take | Path (`FW/nvidia-nemo-gym/`) | Mode | Used by |
|---|---|---|---|
| Resources-server contract `/seed_session`, `/verify`, `/close_session`; `reward_components`, `mask_sample`, `failure_kind` | `nemo_gym/base_resources_server.py` | TARGET | M11 |
| Failure-kind taxonomy; rollout health and outcomes | `nemo_gym/{failure_kinds,rollout_health,rollout_outcomes}.py` | PATTERN | M0 (RunRecord), M10 |
| Verifier fixtures (full / zero / malformed / determinism); rollout and ATIF re-verification | `nemo_gym/{verifier_fixture,rollout_reverification,atif_reverification}.py` | PATTERN | M9, M5 (replay) |
| Harness-capability contracts and checker | `nemo_gym/harness_capabilities/` | PATTERN | M10 |
| Environment manifest, scaffold, onboarding, validation, publication | `nemo_gym/environment/` | PATTERN | M14 (pack scaffold), M9 |
| Token-ID capture for opaque harnesses | `nemo_gym/token_id_capture/` | PATTERN | M3, M12 |
| `swe_pivot` (pivot turns in expert traces), `swebench/anti_cheat.py` (history scrub), `swemer_oml` (second clean grading sandbox) | `resources_servers/` | PATTERN | M4, M7 |
| ATIF v1.7 export | `nemo_gym/atif_v1_7.py`, `atif_export.py` | TARGET | M11 |
| **Skip:** action matching as the main reward (it breaks DESIGN P12); best-effort scrub (ours must be audited) | | | |

**OpenEnv** (patterns, an export target, and corpora)

| Take | Path (`FW/meta-pytorch-openenv/`) | Mode | Used by |
|---|---|---|---|
| `Environment` reset/step/state contract; the agent sees only MCP tools | `src/openenv/core/env_server/`, `rfcs/001,002,003` | PATTERN, TARGET | M1, M11 |
| Rubric tree (`Sequential`, `Gate`, `WeightedSum`, `LLMJudge`), `TrajectoryRubric.compute_step_rewards()` | `src/openenv/core/rubrics/` | PATTERN, TARGET | M6, M11 |
| Auto-validation (oracle reaches max, floor gap, canaries) | `src/openenv/validation/`, `rfcs/008-environment-auto-validation.md` | PATTERN | M9 |
| Capture proxy for agentic-harness interception; Harbor capture providers | `rfcs/006-*.md`, `rfcs/012-*.md` | PATTERN | M3 |
| Serving Harbor tasks as environments | `src/openenv/harbor/` | DEPEND | M11 |
| `agent_world_model_env` (1,000 MCP + database environments, 10,000 tasks), `openapp_env`, `calendar_env` | `envs/` | DATA, PATTERN | M1 generality test, M8 world synthesis |
| **Skip:** `kernrl`'s execution inside the container with no timing hygiene | | | |

**EnterpriseOps-Gym**
- **PATTERN (horizontal worlds):**
  - multi-app MCP;
  - per-app verifiers of three kinds, `database_state` (SQL), `response_check` (judge) and `tool_execution` (`benchmark/verifier.py`), routed by `gym_name`;
  - policy system prompts; distractor-tool modes; infeasible tasks that test refusal;
  - `verifier_level_pass_rate`, which becomes our gate profile in M13.
- **DATA:** an external OOD benchmark for horizontal packs. Its MCP servers ship only as Docker images, so we run them; we can't modify them.

**SWE-smith** (PORT into `_vendor/swesmith`)
- **Ported:**
  - `bug_gen/` (LM modify, LM rewrite, procedural AST, PR mirror, combine);
  - `harness/{valid,grading,repair}.py` (F2P ≥ 1 and P2P ≥ 1; flaky-test repair);
  - `issue_gen/` (problem statements that don't leak the answer);
  - `profiles/` (repo environment build);
  - `train/` (trajectory management for SFT).
- **Fix on port:** instance branches keep the "bug patch" and "remove F2P tests" commits, so `git show HEAD~1` reveals the tests. Our port squashes history to one commit and runs the M9 leak audit.

**XiaomiMiMo verl fork** (training path 2)

| Take | Path (`FW/xiaomimimo-verl/`) | Mode | Used by |
|---|---|---|---|
| Sign-aware, mass-conserving advantage rebalance over marked tokens: `signed_rebalance`, `rebalance_dense`. Positive rows lose a marked span's advantage; negative rows multiply it by κ; each sign conserves its total mass, avoiding the collapse from injecting net-negative mass | `verl/trainer/ppo/signed_rebalance.py` | PORT (`_vendor/verl_signed_rebalance`) + DEPEND when training on verl | M12 (signed-span credit mode) |
| Turn-level penalty operator after GRPO (tool errors, length), with a frozen reference profile | `verl/trainer/ppo/arvo_penalties.py`, `recipes/arvo/REFERENCE_PENALTIES.json` | PATTERN (we write our own operator; theirs is frozen to two rules) | M12 |
| `GeneralAgentLoop`; `DatasetEnvActor`, which runs MiMo sidecar environments; MCP proxy; TokenTrace (generated = 1, tool observations = 0); `chat_delta`; infra metrics; −999 / judge-failure masking; `filter_groups` | `recipes/general/` | DEPEND, TARGET | M12, M11 |
| Code recipe (mimoagent runner, dataset), design recipe (grader service for webdev and music) | `recipes/{code,design}/` | DEPEND / PATTERN | M12 |
| mimoagent: `agents/antihack.py`, `agents/user_agent.py`, `environments/rubric_judge.py` (final_state / process / user_communication), `compaction/` (context surgery for long horizons) | `third_party/mimoagent-osr/src/mimoagent/` | PATTERN | M7 (anti-hack), M2, M6, M10 (long-horizon harness) |

**MiMo v2.6 RL release**

| Take | Path (`MM/`) | Mode | Used by |
|---|---|---|---|
| **App packaging standard:** `manifest.json` (uploads to main or sidecar containers, setup, `wait_ports`, `mcp_servers`); `sidecar_entrypoint.py` (one MCP HTTP server per tools file, plus `/health`, `/dump`, `/verify`); `mcp_http.py`; `mcp_bridge.py` (stdio bridge and one-shot calls) | `general/envs/*/` | PORT (`_vendor/mimo_sidecar`) | M1 packaging |
| App style: `tools/<app>.py` over `system/<app>/{schema.sql,state.db}`; `workspace/` with superseded distractor documents | `general/envs/*/` | PATTERN | M1 app templates, M8 world synthesis |
| Verifier items `{tier: critical/important/sanity, method: rule/llm, question, pass_anchor, weight}` | `general/envs/*/verifier_meta.json` | PATTERN | M6 rubric items, M7 |
| 267 general environments; `code.parquet` (2,698 SWE instances); `cyber.parquet` (1,000 ARVO tasks) | | DATA | M1 generality test, M11 export test |

### 2.4 What no framework provides

These are what we build, and why Crucible exists:
1. **Traces → typed expert signals → step attribution** (M3, M4).
2. **Workflow model:** milestone mining, stage induction, online state tracker, replay fidelity (M5).
3. **Dense credit** that is outcome-anchored, passes the validity gate and can't be farmed, injected into *any* trainer (M6, M12).
4. **Solution-first forging with asymmetric information:** a swarm of roles plus generators grounded in signals (M8).
5. **An acceptance contract that tests the verifier itself:** mutation score, measured flake rate, leak and isolation audits, frontier calibration (M9).
6. **Step-level and OOD benchmarks**, regenerated dynamically (M13).
7. **GPU sandboxes with grading hygiene:** separate verifier, sealing, load guard (M1, M7, M10).
8. **Grounded simulated actors with measured fidelity** (M2).

---

## 3. Project structure

```
crucible/
├── DESIGN.md                    # what and why
├── ARCHITECTURE.md              # this file
├── README.md                    # quickstart (written with M15)
├── THIRD_PARTY.md               # pins, licenses, ported code (§2.2)
├── LICENSES/                    # upstream licenses for ported code
├── pyproject.toml               # one package; extras: harbor, agentenv, verifiers, verl, nemo-gym, openenv, gpu
├── src/crucible/
│   ├── core/                    # M0  schemas, ids, provenance, plugin registry, config, store
│   │   ├── schema/              #     trace, signal, world, state, milestone, task, reward, card, bench, run
│   │   └── store/               #     content-addressed local store for traces, worlds, tasks, runs
│   ├── world/                   # M1  World runtime
│   │   ├── state/               #     State Adapters: fs_git, sql, doc, http_mock, browser, process(+gpu)
│   │   ├── apps/                #     generic App templates: workspace, git_repo, sqlite_app, postgres_app, mailbox, doc_library, website, http_mock
│   │   ├── packaging/           #     App package format (MiMo sidecar standard)
│   │   └── backends/            #     local (CI), harbor, agentenv: the last two call interop/
│   ├── actors/                  # M2  policy, simulated, grounded, scripted (CI, cron, mail), human, peer, fidelity
│   ├── connectors/              # M3  Connector protocol, redaction, trace store, generic connectors
│   ├── signals/                 # M4  taxonomy, extraction, step attribution, reliability, outcome labels
│   ├── workflow/                # M5  segmentation, replay, milestone mining, stages, graph, tracker, fidelity
│   ├── credit/                  # M6  operators C1–C6, reward composition, validity gate, steps.jsonl, property tests
│   ├── verify/                  # M7  check patterns, isolation, calibration, fast/full tiers, grader runtime
│   ├── forge/                   # M8  orchestrator, swarm roles, generators, learnability, dedup
│   ├── accept/                  # M9  contract A1–A12, defect registry, reports
│   ├── runtime/                 # M10 episode runner, harness adapters, collection, infra classification, budgets, perf guard
│   ├── interop/                 # M11 the ONLY code that imports external frameworks
│   │   ├── harbor/              #     task dir I/O, HarborGpuDocker, ATIF, job bridge, compile/analyze bridges
│   │   ├── agentenv/            #     plugin: gpu_docker sandbox provider, crucible_world + crucible_verify steps, artifact mirror
│   │   ├── verifiers/           #     Taskset export (extends HarborTaskset), reward channels, advantage hook
│   │   ├── verl/                #     MiMo-format world export, recipes/general + code datasets, advantage operator
│   │   ├── nemo_gym/            #     resources server + manifest
│   │   ├── openenv/             #     Environment + rubric export + openenv.yaml
│   │   ├── mimo/                #     import MiMo envs as Worlds; export Worlds to MiMo format
│   │   └── eog/                 #     import EnterpriseOps-Gym tasks as an external benchmark
│   ├── train/                   # M12 advantage operators, trainer bridges, SFT export, curriculum, monitors
│   ├── bench/                   # M13 splits, step metrics, dynamic regeneration, IRT, contamination, reports
│   ├── cli/                     # M15 `crucible ...`
│   └── _vendor/                 # ported code, each folder with LICENSE + ORIGIN.md
│       ├── swesmith/
│       ├── mimo_sidecar/
│       └── verl_signed_rebalance/
├── packs/                       # M14 domain packs, each installable; registered via entry points
│   ├── _template/               #     scaffold for `crucible pack new`
│   ├── _toy/                    #     helpdesk world, CPU-only, runs in CI; keeps the core domain-agnostic
│   ├── coding/                  #     MVP pack
│   │   ├── pack.yaml
│   │   ├── apps/                #     git_repo + toolchains (cpu, gpu), ci_service (CI as a tool, hidden checks), review_service
│   │   ├── connectors/          #     github, gitlab, ci_logs, vcs_history, claude_code_sessions, mini_swe_agent_runs, policy_docs, issues
│   │   ├── signals.py           #     coding events → generic signal types
│   │   ├── outcomes.py          #     merged ∧ CI green ∧ no revert/hotfix in 30 days
│   │   ├── milestones/          #     build, static_scan, tests_f2p, differential, sanitizer, concurrency, perf
│   │   ├── verify/              #     coding check patterns (bit-exact compare, perf rounds, sanitizers)
│   │   ├── generators/          #     coding hindsight / span / trap / transplant / compose / tighten
│   │   ├── horizons.yaml        #     H0–H5
│   │   └── families/
│   │       └── legacy_gpu_migration/
│   │           ├── family.yaml
│   │           ├── workflow_graph.yaml
│   │           ├── milestones.yaml
│   │           └── seeds/       #     tasks 01 (v1, v2), 02, 03, 04 imported from ../../migration_tasks
│   └── finance_ap/              #     post-MVP stub: pack.yaml + one World spec (horizontal)
├── tests/
│   ├── unit/                    # per module
│   ├── contract/                # interfaces between modules; import-linter rules
│   ├── property/                # reward properties R1–R6 (hypothesis)
│   ├── golden/                  # historical results of tasks 01–04 reproduced exactly
│   ├── toy_e2e/                 # whole pipeline on _toy, CPU, < 5 min
│   └── gpu/                     # GPU suites, run on the lab 5090 (scripts/lab_gpu.sh)
├── scripts/
├── docs/modules/                # one spec + gate report per module as it is built
└── data/                        # gitignored: trace store, worlds, runs, benchmarks
```

---

## 4. Modules

Every module card has the same fields: **Purpose · Contents · Takes · We build · Coding (MVP) · Toy · Exit gate · Depends on · Size** (S < 1 week, M ≈ 1–2, L ≈ 2–3, XL > 3, for one engineer working with agents).

### M0 `core` — contracts
- **Purpose:** typed schemas and the plumbing every other module uses. No logic about any domain.
- **Contents:**
  - **`schema/`**, versioned and exported as JSON Schema:
    - `trace` (`crucible.trace/v1`, ATIF embedded)
    - `signal`
    - `world` (`crucible.world/v1`)
    - `state` (StateRef, StateDiff, Query)
    - `milestone`
    - `task` (Harbor 1.4 superset plus the `crucible/` extension)
    - `reward` (`reward.json`, `steps.jsonl` rows)
    - `card`
    - `bench`
    - `run` (episode record with `failure_kind` and mask flag)
  - **Utilities:** `ids.py` (content hashes), `prov.py` (every artifact records `derived_from`), `registry.py` (entry-point plugin registry), `config.py` (`crucible.toml`; secrets only as `env:NAME` references).
  - **`store/`:** content-addressed local store under `data/`, with a SQLite index.
- **Takes:**
  - DEPEND: Harbor `TaskConfig` and ATIF models.
  - PATTERN: Harbor packager hashing; agentenv `module:Class` registries and `env:` secret references; NeMo-Gym `failure_kinds.py`; NeMo-Gym `manifest.py` and OpenEnv `openenv.yaml` (for `pack.yaml`).
- **We build:** all of it. The schemas are the contract every later module is tested against.
- **Coding:** the `task.toml` of tasks 01–04 parses as Harbor 1.4 plus the extension.
- **Toy:** the helpdesk World spec validates.
- **Exit gate:**
  - every schema round-trips (model → JSON → model) with JSON Schema exported;
  - the 5 seed task directories and the toy World validate;
  - the import-linter contracts (§1.5, decision 4) are active in CI;
  - `THIRD_PARTY.md` is written.
- **Depends on:** nothing. **Size:** M.

### M1 `world` — runtime for any environment
- **Purpose:** turn a `WorldSpec` into a running world. The policy gets tools; the verifier gets state views; the whole world snapshots and restores.
- **Contents:**
  - `world.py`, `spec.py` (template resolution);
  - `state/` adapters: fs_git, sql (sqlite, postgres), doc, http_mock, browser, process with GPU and perf counters;
  - `apps/` generic templates;
  - `packaging/` (App package = MiMo sidecar format);
  - `clock.py`, `events.py`;
  - `guard/` (built: demoted policy user, quiesce, verified git scrub, egress self-test) and `leaks.py` (scrubber); agent view vs verifier view is the World's `VerifierView`;
  - `backends/{local,harbor,agentenv}`.
- **Takes:**
  - DEPEND: Harbor environments and providers; agentenv `MultiEnv` / `MCPServerEnv` / `ServiceDBEnv` / `WebsiteEnv` / `GatewayEnv`, plus `snapshot_env`, `reset_env`, `sync_env_clock`, `modify_env_tool_access`.
  - PORT: MiMo `sidecar_entrypoint.py`, `mcp_http.py`, `mcp_bridge.py`, `manifest.json`.
  - PATTERN: OpenEnv's reset/step/state and MCP-only rule; EnterpriseOps-Gym routing by app.
- **We build:**
  - the World abstraction;
  - canonical fingerprints per adapter (sorted SQL dumps, normalized git trees, deterministic process records);
  - whole-world snapshot and restore, across Apps, clock and event queue;
  - `HarborGpuDocker` and agentenv's `gpu_docker` provider;
  - a minimal in-process clock and events for the local and Harbor backends.
- **Coding:** the task 01–04 worlds on Harbor with GPU on the lab 5090.
- **Toy:** the helpdesk world on the local backend.
- **Exit gate:**
  - **Snapshot round-trip.** For all 5 seed worlds and the toy world: reset → apply the oracle through the tools → snapshot → restore into a fresh world → identical fingerprints, 3 out of 3 times.
  - **Generality.** ≥ 20 MiMo envs and ≥ 20 OpenEnv AWM envs load through `packaging/` or `interop/`. Tool lists match their manifests, and state fingerprints are stable across `reset`.
  - **Clock.** On the agentenv backend, a scheduled event fires at the correct virtual time, ±1 virtual minute at rate 3600.
- **Depends on:** M0. **Size:** L.

### M2 `actors` — everyone who isn't the policy
- **Purpose:** simulated users, reviewers, approvers and departments; peer agents; humans; system actors.
- **Contents:**
  - `policy.py` (wraps the harness or endpoint under test);
  - `simulated.py` (persona LLM actors);
  - `grounded.py` (responders that answer only from record spans, else "no record");
  - `scripted.py` (rule-based relay bots and the CI bot; timed events live in the World's scheduler);
  - `human.py` (humans and peer agents over transports: callable, file queue, A2A);
  - `fidelity.py`.
- **Takes:**
  - DEPEND: verifiers `UserSimEnv`; agentenv A2A agents, `peer_agents`, `deploy_human_agent`, `register_env_triggers`.
  - PATTERN: mimoagent `user_agent.py`; EnterpriseOps-Gym policy prompts.
- **We build:**
  - **Grounding enforcement:** every factual claim must cite a record span, and a checker rejects uncited claims.
  - **Fidelity measurement** against held-out real responses.
- **Coding:** CI as a tool, which runs the hidden fast-tier checks and returns only pass/fail per public test name. A reviewer simulator grounded on historical review threads, used by trap and H5 tasks.
- **Toy:** a customer simulator grounded on ticket history.
- **Exit gate:**
  - reviewer simulator on held-out review rounds: defect-location precision ≥ 0.8;
  - fabrication rate ≤ 1% on 200 audited replies, for both coding and toy;
  - every simulated reply carries citations.
- **Depends on:** M0, M1, M3 (records). **Size:** M. In the MVP only CI-as-a-tool and a minimal reviewer are built; the rest is needed for H5 and horizontal packs.

### M3 `connectors` — capture and ingest
- **Purpose:** raw records → `crucible.trace/v1`, redacted inside the tenant.
- **Contents:**
  - `base.py` (`discover / fetch / normalize / redact`), `redact.py`, `trace_store.py`;
  - generic connectors:
    - `atif_import`;
    - `file_drop` (JSONL / CSV event logs);
    - `sql_audit_log` (database change logs → state events);
    - `otel_import` (enterprise agent telemetry);
    - `capture_proxy` (an OpenAI- or Anthropic-compatible proxy in front of live agents).
- **Takes:**
  - DEPEND: Harbor installed-agent → ATIF converters; `harbor-atif2otel`.
  - PATTERN: verifiers `interception/`; NeMo-Gym `token_id_capture/` and its Anthropic / Responses converters; OpenEnv RFC 006 and RFC 012; SWE-smith PR mirror and `profiles/`.
- **We build:** the trace envelope (non-agent events, multiple actors, signals), redaction with canary tests, and idempotent ingest.
- **Coding pack connectors:**
  - `github`, `gitlab` (PRs, commit series, review threads, suggestions, check runs);
  - `ci_logs`;
  - `vcs_history` (reverts and hotfixes via blame; today inside `github_pr`);
  - `claude_code_sessions` (our construction transcripts);
  - `mini_swe_agent_runs` (the GPT 6 Astra and GPT 5.6 sol trajectories; built as `agent_runs`);
  - `policy_docs`, `issues`.
- **Toy:** `file_drop` of ticket event logs.
- **Exit gate:**
  - ≥ 30 public GPU-migration PR traces, our task 01–04 construction transcripts and every recorded frontier run are ingested, 100% schema-valid;
  - redaction removes 100% of planted canaries (API-key formats, emails, hostnames);
  - re-ingest is idempotent (same content hash).
- **Depends on:** M0. **Size:** M.

### M4 `signals` — expert signals
- **Purpose:** find, type, attribute and weigh expert signals.
- **Contents:**
  - `taxonomy.py`: outcome, correction, rejection, approval, rework, revert, escalation, violation, sla_breach, interrupt, suggestion_accepted, target;
  - `extract.py` (runs the pack's mapper and outcome labeler, plus the generic session rules; validates the output; ids keyed on stable event keys);
  - `session.py` (interrupts and rejected tool use in any agent session, graded on the proposing call);
  - `attribute.py` (which step each signal grades — the generic rule: the event that last wrote the object the signal is about; the coding rule: blame within the PR);
  - `reliability.py` (agreement with later evidence the mapping rule did not use, per family, type and human or automated source);
  - `pivots.py` (graded-wrong-then-corrected steps), `gate.py` (the exit gate against hand labels);
  - outcome labelers are pack plugins (`outcome_labeler`), like the mappers.
- **Takes:** PATTERN only: SWE-smith `harness/repair.py` (cleaning flaky CI), NeMo-Gym `swe_pivot` (finding decision points), research-environments review rubric (candidates).
- **We build:** all of it. No framework has this layer.
- **Coding:** review defects at lines, accepted suggestions, reverts and hotfixes, CI regressions (check runs and CI logs), frontier-run gate failures, perf targets.
- **Toy:** reopen, escalation, customer replies, CSAT (as the trace's outcome).
- **Exit gate:** on 50 hand-labeled events per pack:
  - type accuracy ≥ 0.9;
  - step-attribution precision ≥ 0.9;
  - outcome labels agree with manual labels ≥ 0.95.
- **Depends on:** M0, M3. **Size:** M.

### M5 `workflow` — workflow model
- **Purpose:** segment, replay, mine milestones, induce stages, track state online (DESIGN §5.3).
- **Contents:** `segment.py`, `replay.py`, `milestones/mine.py` (filters: executable, monotone, non-trivial, predictive), `stages.py`, `graph.py`, `tracker.py`, `fidelity.py`.
- **Takes:**
  - PORT: SWE-smith `harness/grading.py`, applied per step.
  - PATTERN: NeMo-Gym rollout and ATIF re-verification.
  - DEPEND: Harbor `trajectory.json` prefix seeding (starting from mid-trace states); agentenv `snapshot_env` and `snapshot_agent_state`.
- **We build:** all of the mining, induction and tracking.
- **Coding:** the F1 graph mined from upstream PRs and our transcripts.
- **Toy:** the helpdesk graph (triage → diagnose → resolve → confirm).
- **Exit gate:**
  - S1: replay ≥ 90% on ≥ 30 traces;
  - tracker vs manual stage labels: Cohen's κ ≥ 0.7;
  - **rediscovery:** the mined F1 milestones cover every check family that tasks 01–04 graders were hand-built with (build, API removal, bit-exactness, edge semantics, concurrency, perf).
- **Depends on:** M1, M3, M4. **Size:** L.

### M6 `credit` — dense credit and reward composition
- **Purpose:** DESIGN §6, as code.
- **Contents:**
  - `operators/{potential, defect_to_check, branch_value, rubric_item, prm, infra_mask}.py`;
  - `compose.py` (the gated R formula);
  - `validity.py` (time-split AUC gate);
  - `steps.py` (writes `steps.jsonl`);
  - `rubric/` (item lifecycle: candidate → validated → weighted);
  - `properties/` (R1–R6).
- **Takes:**
  - PATTERN: OpenEnv rubric containers and `TrajectoryRubric` (we mirror their tree, so export is one-to-one); MiMo `verifier_meta` item schema; mimoagent `rubric_judge.py`; agentenv `scoring.py` (weighted with negative weights); prime-environments shaped rewards.
  - DEPEND: Harbor rewardkit and verifiers judges for judge items.
- **We build:** C1–C6, the composition, the validity gate, and branch rollouts from world snapshots.
- **Coding:** Φ over F1 milestones; defect → check on review corrections.
- **Toy:** Φ over SQL milestones.
- **Exit gate:**
  - S2: Φ at the midpoint predicts the outcome, AUC ≥ 0.75 and above an LLM-judge baseline, on held-out traces;
  - R1–R6 pass on every coding and toy task;
  - correlation between C3 branch values and ΔΦ is *reported* for 3 task04 trajectories.
- **Depends on:** M1, M4, M5, M7. **Size:** L.

### M7 `verify` — verifier kit
- **Purpose:** build hidden checks that are correct, isolated, calibrated and fast enough to train with.
- **Contents:**
  - **Generic `patterns/`:** state_predicate, differential (against a reference world or build), hidden_tests, adversarial_inputs, invariants, fresh_draws, perf (best-of-k, spread, load guard), static, judge_item (validated items only).
  - **`isolation/`:** separate verifier environment, sealing, collect hooks, leak audit. Hidden-check CI as a policy tool (`world/apps/ci.py`) and the in-sandbox guard were pulled forward into M1.
  - **Support:** `calibrate.py` (tolerance tiers, induced perturbations, false-negative rate), `tiers.py` (fast/full), `grader.py` (runs inside the verifier environment and writes `reward.json` and `steps.jsonl`).
- **Takes:**
  - DEPEND: Harbor verifier (separate mode, collect); verifiers `IsolatedVerifierEnv`.
  - PATTERN: NeMo-Gym two-sandbox grading and `anti_cheat`; EnterpriseOps-Gym `database_state` (turned into violating-row invariants; `response_check` skipped: the policy's own model judges itself); MiMo `run_verify.py` ("no parseable reward → masked"; there is no `/verify` endpoint); Repo2RLEnv `recipes/swe_smith/grade.py` (pre-written 0, exact expected tests); mimoagent `antihack.py` only as a probe list (off by default, easy to bypass). verifiers `IsolatedVerifierEnv` moves to M11 export (as-built 2026-10-09).
  - PORT: **our own task 03/04 graders**, generalized into patterns.
- **Coding:** sanitizers, concurrency equivalence against solo references, bit-exact compares, perf rounds.
- **Toy:** SQL invariants and state predicates.
- **Exit gate (golden):** the graders of tasks 01–04, re-expressed with the kit, reproduce every historical result exactly:
  - every oracle scores 1.0;
  - every negative fails at its intended gate;
  - the historical GPT 6 Astra and GPT 5.6 sol patches reproduce their recorded gate profiles (task04 v3: Astra fails the perf bars; sol fails NCV);
  - the toy checks pass and fail as designed.
- **Depends on:** M0, M1. **Size:** L.

### M8 `forge` — agent-swarm synthesis
- **Purpose:** build worlds and tasks solution-first, with asymmetric information (DESIGN §7).
- **Contents:**
  - `orchestrator.py` (a deterministic DAG with budgets, retries and artifact handoff);
  - `roles/` (Miner, Architect, Environment builder, Oracle builder, Test writer, Adversary, Auditor, Calibrator, Curator; each is a prompt plus a toolset plus an output schema);
  - `generators/` (hindsight, span, trap, transplant, compose, tighten, world_synth);
  - `learnability.py`, `dedup.py`.
- **Takes:**
  - PORT: SWE-smith `bug_gen/`, `harness/valid.py`, `repair`, `issue_gen/`.
  - DEPEND: Harbor `compile` (scaffolding), `analyze` (Auditor), installed agents and environments (to run builder roles in sandboxes); agentenv `review` step (human audit of 10% of tasks).
  - PATTERN: verifiers `proposer_solver` (learnability); OpenEnv AWM (world synthesis); MiMo tiers (difficulty = number of Apps); EnterpriseOps-Gym distractors and infeasible tasks; research-environments recursive synthesis.
- **We build:** the orchestration, the asymmetric-information discipline (roles cannot see what they must not), the signal-cited traps, and world synthesis.
- **Coding:** H1–H3 span tasks from each H4 seed; trap tasks from recorded corrections; tighten.
- **Toy:** `world_synth` builds a variant helpdesk world with 5 tasks.
- **Exit gate (MVP):**
  - every derived span task and ≥ 1 trap task per seed passes M9;
  - each trap cites its signal;
  - the toy variant world and its tasks pass M9.
- **Depends on:** M1, M4–M7, M9, M10. **Size:** XL.

### M9 `accept` — Acceptance Contract and defect registry
- **Purpose:** a task enters the pool only if A1–A12 (DESIGN §9) all pass.
- **Contents (as built):** `model.py`, `policy.py`, `key.py`, `cache.py`, `plan.py`, `run.py` (the runner), `checks/{base,core,extended}.py` (A1–A13, S1), `mutation/` (operators per solution kind), `registry/` (defect YAMLs, power-test fixtures, backfill), `contract.py` (the M8 subset). Still to build: `report.py` card updates, `cli.py`, patch mutation operators, seed adapters.
- **Takes:**
  - DEPEND: Harbor quality checker rubric and analyzer (A7, A3 cheat attempts); OpenEnv validation (RFC 008); agentenv `review` (manual sign-off for H4+).
  - PATTERN: `vf-validate`; NeMo-Gym `verifier_fixture`, `environment/validation.py`, `harness_capabilities`.
- **We build:** the contract runner, the mutation-score harness, flake measurement, and a registry seeded from DESIGN Appendix B.
- **Imports (not built; M9 audit 2026-10-12):** I0–I10 (research m9-m10/03 §5) on Harbor directories, verifiers row tasksets and NeMo JSONL before A1–A13, writing `import_report.json`. The layer rules put the adapters in `interop` (above `accept`): acceptance capabilities over `interop/harbor/inplace.HarborInPlaceTask` and the NeMo-Gym task, honouring separate verifiers and refusing unknown grading keys (sweep `M08-M10-sweep-G5.md`). Test bed: Terminal-Lego (paired, labelled), R2E-Gym-Subset-Verified, HF_ML_Tasksmith (+ equivalence-tests as negatives), terminal-bench-2.1, NeMo workplace_assistant; negative fixtures from BeyondSWE, RetroEnv and SmolDataEnvs.
- **Exit gate (DESIGN P0 gate):**
  - tasks 01–04 pass automatically;
  - re-injecting ≥ 3 historical defects (Haar nondeterminism, the weak NCV negative, a stale verifier image) is caught;
  - the toy tasks pass.
- **Depends on:** M0, M1, M7, M10. **Size:** M.

### M10 `runtime` — running policies reliably
- **Purpose:** run agents or policies on tasks at scale and classify every failure.
- **Contents:**
  - `run.py` (k samples, seeds);
  - `harness/` (Harbor agents, verifiers harnesses, A2A agents through agentenv, an OpenAI-compatible endpoint for policies in training);
  - `collect.py` (tolerant collect hooks);
  - `infra.py` (`failure_kind`, masks);
  - `budget.py` (API cost, credit exhaustion, stop rules);
  - `perf_guard.py`;
  - `gpu.py` (lab 5090, Runpod, Modal);
  - `trajectories.py` (ATIF with step ids that line up with `steps.jsonl`);
  - `checkpoint.py` (mid-episode checkpoints; a masked attempt resumes from the last one, DESIGN §10.4), `perf_guard.py`, `record.py`, `harness/imported.py`.
- **Long horizons (DESIGN §10.4):** `verify/progress.py` (online dense progress over the grader's subgoals, telescoping), `actors/policy.ContextConfig` (policy-written compaction, segments), `World.stream_trace`, linear token storage in `core/schema/rollout.py`; toy `longhorizon/` (backlog world, `xl` > 1M tokens, gate).
- **Takes** (revised after research m9-m10/02):
  - DEPEND: Harbor Job / Trial / regrade and installed agents (through `harness/harbor.py`); verifiers interception proxy for tokens.
  - PORT: Repo2RLEnv `campaigns/budget.py` (reserve/settle ledger) and `EventJournal`; verifiers `RetryRule` and error classes; FineEnvs `evaluate` identity; Karotte limit policy (limits stop, never score); our `run_agent.sh` fixes (collect hook, cost-tracking tolerance, `--init`).
  - PATTERN: NeMo-Gym `failure_kinds`, `rollout_health`, failure sidecars; verl −999 masking; agentenv `eval run --k` and step journal (the shape of resume; agentenv's runner is not used, because it has no masking or blame); mimoagent `compaction/` for long horizons; OpenEnv capture for the endpoint harness.
- **Coding:** frontier calibration runs across several Harbor harnesses.
- **Toy:** a scripted policy plus one LLM policy.
- **Exit gate:**
  - one historical frontier run (mini-swe-agent on task02) is reproduced end to end;
  - all 8 harness failure classes in DESIGN Appendix B.5 are injected, and each is classified as infra and masked, never scored 0.
- **Depends on:** M0, M1. **Size:** M.

### M11 `interop` — the boundary with every framework
- **Purpose:** import from and export to each framework. This is the only place their code is imported.
- **Contents:** one folder per framework, as in §3.
  - **harbor:** task directory I/O, `HarborGpuDocker`, ATIF, job bridge.
  - **agentenv:** a plugin registering the `gpu_docker` sandbox provider, a `crucible_world` step (a World as MultiEnv plus gateway), a `crucible_verify` step (separate verifier container; writes `reward.json` and `steps.jsonl`), artifact mirroring.
  - **verifiers:** Taskset export that extends `HarborTaskset`; reward channels as `@vf.reward` / `@vf.metric`; dense credit written into `MessageNode.advantages`.
  - **verl:** Worlds exported in MiMo format for `recipes/general` (`DatasetEnvActor`); coding tasks as `recipes/code` datasets (the `instance_json` schema in `code.parquet`).
  - **nemo_gym:** a resources server plus `manifest.yaml`.
  - **openenv:** an `Environment` plus rubric tree plus `openenv.yaml`, with per-step `Observation.reward` taken from `steps.jsonl`.
    - **Import (as built 2026-10-10):** any OpenEnv `Environment` as a World App (`openenv/env`): MCP tools or the Action schema as a `step` tool, replay-journal state with digest-verified restore, environment-terminated episodes, the environment's own reward as a score gate; catalogue starts with FineEnvs PortSimEnv (DESIGN M11 part 4).
  - **mimo:** import and export of the MiMo format.
  - **eog:** an external-benchmark importer.
- **Exit gate (cross-runtime consistency):** the same task and the same patch or trajectory give **identical** `reward.json` and `steps.jsonl` through Harbor, agentenv, verifiers, NeMo-Gym and OpenEnv, on task02 and one toy task. One MiMo env is imported, run and re-exported byte-identical.
- **Depends on:** M1, M7. **Size:** L, spread across stages: harbor and agentenv in A; the others in C.

### M12 `train` — trainer bridges and dense advantages
- **Purpose:** connect to trainers, inject dense credit, produce SFT data, steer the curriculum.
- **Contents:**
  - **`advantage/`, three modes:**
    1. **Shaping return-to-go** (DESIGN §6.4): zero mean within a group, so it adds no bias.
    2. **Signed-span**: mark turns with ΔΦ < 0, regressions, policy violations or defect hits, then apply verl's mass-conserving `signed_rebalance`.
    3. **Branch value** (C3).
  - **`bridges/`:** verifiers → prime-rl, verl, OpenEnv → TRL.
  - **`sft/`:** expert traces lifted into tool-call trajectories and checked by replay; rejection-sampled rollouts; ShareGPT / HF export.
  - **`curriculum.py`:** learnability-steered refresh of the task pool.
  - **`monitors.py`:** entropy, share of zero-variance groups, sampled hack audits, divergence between the fast and full verifier tiers.
  - **RL infrastructure (trainer side; designed 2026-10-09 from the 11-lab survey, `docs/research/2026-10-09-agentic-rl-infra-survey.md`, DESIGN §10.3):**
    - **As built (2026-10-10):** `mismatch.py`, `objectives.py`, `batching.py`, `credit.py`, `pool.py`, `scheduler.py`, `sim.py`, `monitors.py`, `engine.py`, `envs.py`, `trainer.py`, `weights.py`, `weightsync.py` (bucketed distributed weight sync, checkpoint-engine pattern), `runtime.py` (DESIGN §10.3 as-built parts 1–4; research `docs/research/m11-m12/`).
    - **`rollout.py`, the rollout record contract `crucible.rollout/v1`:**
      - per turn: prompt and response token ids, **behaviour logprobs as sampled by the inference engine** (token-in-token-out, never detokenize→retokenize), the **policy version of every generated span** (so the oldest version w₀ is known), sampling parameters, and optional MoE routing ids and top-p candidate sets;
      - per rollout: harness@version, truncation flag, context-compaction segments, environment timings, infra status, World trace id.
      - A rollout missing any token record is masked, never guessed.
    - **`topology.py`:** colocated + async partial rollout first (one shared lab GPU), with environments always off the GPU pool (Worlds in CPU sandboxes). Disaggregated later, through a data pool between generation and training.
    - **`staleness.py`:**
      - partial rollout with a version cap S (default 4); drop when w′ − w₀ > τ;
      - per-sample scheduling (a prompt's group starts when enough slots free up);
      - optional windowed FIFO (W = 0.3·N);
      - length-bias control: per-dataset concurrency caps; early short returns held until the length mix is steady.
    - **`mismatch.py`:**
      - token-level corrections, selectable: TIS (truncate π_train/π_rollout), IcePop-style two-sided masking, KPop binary-KL masking, DIS (π_rollout as the denominator when π_old is unavailable), MIS-PO (token + geometric-mean trajectory filter);
      - an FP32 LM-head switch;
      - `rollout_probs_diff` logged;
      - routing replay only when routing confidence is low (MoE policies).
    - **`objectives.py`:**
      - GRPO family with DAPO defaults: clip-higher 0.2/0.28, token-level loss, overlong filtering, dynamic sampling, which keeps all-fail groups that carry κ(1−O)Φ;
      - Dr.GRPO (no std); GSPO (sequence ratio); CISPO (clipped IS weight, every token keeps its gradient);
      - a **critic mode for single-feedback workflows** (group size 1, the normal case for one real enterprise case per prompt): value baseline, skip-observation GAE, critic updated about 2× per policy step, value pretraining; Φ is offered as an extra baseline.
    - **`credit.py` (M6 → tokens):**
      - turn advantages attach to their `call_id` spans; observation and tool-output tokens are masked from the loss;
      - truncated rollouts are masked or Φ-bootstrapped, never scored 0;
      - environment failures the agent caused are scored 0, never masked;
      - optional GAR redistribution among passing solutions (mass-conserving; confirmed hacks zeroed and group statistics recomputed).
    - **`compaction.py`:** context compaction recorded as segments; credit crosses segments (token-level normalization, cross-segment discounting); the harness and context mechanism vary across groups so the policy does not overfit one.
    - **`monitors.py` additions:** staleness histogram, `rollout_probs_diff`, share of clipped/masked tokens, truncation rate, length drift, confirmed-hack rate, and a **pass-rate-jump alarm** that triggers a full-tier re-audit.
    - **Hacks during training:** a blocked shortcut gets a policy-facing refusal and the rollout continues; an exposure masks (P27); a hack confirmed by the full tier or the group grader zeroes the reward and the group is re-normalized.
- **Takes:**
  - PORT/DEPEND: verl `signed_rebalance.py`.
  - DEPEND: verl `recipes/general` (agent loop, TokenTrace, `chat_delta`, MCP proxy, env actor, `filter_groups`); verifiers `graph.py` and interception; prime-rl; Harbor trace export, `RolloutDetail`, Tinker backend.
  - PATTERN: verl `arvo_penalties.py`; SWE-smith `train/`; NeMo-Gym `train_data_utils.py`.
- **Exit gate:**
  - **MVP:**
    - unit tests: signed-span preserves the advantage mass of each sign; shaping advantages have zero group mean;
    - one GRPO step per advantage mode on the toy pack and on one coding task, on both prime-rl and verl;
    - a small sparse-vs-dense run on tasks 01–04 plus their span tasks;
    - RL-infrastructure contract tests:
      - a token-in-token-out round trip (rollout token ids and logprobs reproduce the trainer's input exactly);
      - masking when a token record is missing;
      - the staleness drop rule and windowed FIFO bounds;
      - each mismatch correction on synthetic ratio distributions;
      - truncated and agent-broke-env cases mapped as specified;
    - one colocated async GRPO run on the toy pack on the lab RTX 5090, with staleness and `rollout_probs_diff` monitors recorded.
  - **P4:** S5.
- **Depends on:** M6, M10, M11. **Size:** L.

### M13 `bench` — benchmarks
- **Purpose:** build, run and report step-level and OOD benchmarks, including dynamically regenerated ones.
- **Contents:**
  - `splits.py`;
  - `metrics.py`: gate profile, progress AUC, stage reach, first irrecoverable step, regressions, recovery, cost;
  - `dynamic.py`: anchors plus fresh items via M8;
  - `irt.py` (2PL), `contamination.py`;
  - `report/`: HTML report with per-step heatmaps.
- **Takes:**
  - DEPEND: Harbor jobs, metrics and viewer; agentenv eval and explorer; `vf-eval`.
  - PATTERN: EnterpriseOps-Gym `verifier_level_pass_rate`; NeMo-Gym `benchmarks.py` and `comparison/`; research-environments evaluation skills.
- **Exit gate (MVP):**
  - a report on tasks 01–04 and their span tasks, for 2 frontier models and 1 open model, with every step metric;
  - reproducible: a re-run gives identical deterministic components;
  - a toy benchmark is generated and reported.
- **Depends on:** M5, M6, M8, M10. **Size:** M.

### M14 `packs` — domain knowledge as plugins
- **Purpose:** hold everything that is domain-specific.
- **Contents:**
  - `pack.yaml`: id, version, Apps, adapters, connectors, signal mapper, outcome labeler, milestone library, verifier patterns, generators, horizon ladder, rubric seeds;
  - `_template/` plus `crucible pack new` / `crucible pack check`;
  - `_toy/`, `coding/`, `finance_ap/` (stub).
- **Takes:** PATTERN: NeMo-Gym `environment/{scaffold, onboarding, manifest, publication}.py`; OpenEnv `openenv.yaml` and discovery; agentenv plugin entry points and `plugin check`; verifiers `vf-init` and `prime env push`.
- **Exit gate:**
  - `crucible pack new demo` creates a pack that passes `crucible pack check`;
  - `coding` and `_toy` pass every module gate built so far.
- **Depends on:** M0. Grows with every stage. **Size:** M, plus the content of each pack.

### M15 `studio` — tools for enterprises
- **Purpose:** the two products the thesis promises, behind one CLI, with a UI later.
  1. **`crucible label`**: workflows → traces with dense rewards, plus SFT and DPO data (M3 → M4 → M5 → M6).
  2. **`crucible build`**: workflows → RL environments, tasks and benchmarks (… → M8 → M9 → M11 → M13).
- **Contents:** `cli/` with `ingest | signals | model | label | credit | forge | accept | export | run | bench | train | pack`. The UI is an agentenv explorer plugin showing traces with per-step credit, milestones and task cards; Harbor viewer handles trajectories.
- **Takes:** DEPEND: agentenv `explorer` and `agent_env.cli_plugins`; Harbor `viewer`.
- **Exit gate:** a new user runs `crucible label` and `crucible build` on the toy pack and on task02, using only the README.
- **Depends on:** all. **Size:** M.

---

## 5. Build order

```
Stage A (P0, wk 1–2)   M0 core ─► M1 world ─► M7 verify ─► M10 runtime ─► M9 accept      (+ M14 _toy, coding seeds; M11 harbor/agentenv)
Stage B (P1, wk 2–4)   M3 connectors ─► M4 signals ─► M5 workflow ─► M2 actors (CI-as-tool, reviewer)
Stage C (P2, wk 4–7)   M6 credit ─► M11 verifiers/verl/nemo_gym/openenv ─► M12 train (dry runs)
Stage D (P3, wk 5–9)   M8 forge (span, trap, tighten, world_synth) ─► coding generators
Stage E (P4, wk 8–12)  M12 train (runs) ─► M13 bench ─► M15 studio
```

Dependencies:
- M0 → everything.
- M1 → M2, M5, M7, M10, M11.
- M3 → M4 → M5 → M6.
- M7 → M6, M9.
- M10 → M8, M9, M12, M13.
- M6 + M11 → M12.
- M5 + M6 + M8 → M13.

Parallel work:
- M3 and M4 can start during Stage A, because they depend only on M0.
- M14 `_toy` grows alongside every module.

| Stage | Exit = DESIGN gate |
|---|---|
| A | P0: tasks 01–04 pass `crucible accept`; ≥ 3 historical defects caught on re-injection; toy passes |
| B | P1: S1 ≥ 90%; tracker κ ≥ 0.7; milestones rediscovered |
| C | P2: S2; R1–R6; cross-runtime consistency |
| D | P3 (MVP form): seeds + span + trap tasks pass A1–A12 |
| E | P4: sparse vs dense run; step-level report; `label` / `build` usable from the README |

---

## 6. Tasks 01–04 as the MVP showcase

| Module | What happens to the seeds (task04 as the example) |
|---|---|
| M0 | `task.toml` + `crucible/` extension; card with provenance (Astra/sol results, oracle headroom) |
| M1 | World A: `repo` (fs_git) + `gpu` (process, sm_120); snapshot/restore round-trip on the 5090 |
| M7 | v3 grader re-expressed in patterns: differential (KinFu/KinFuLS/people/NCV), concurrency vs solo refs, fresh draws with two-tier tolerance, perf + throughput bars, sealing |
| M10 | historical runs re-executed; harness failures classified |
| M9 | A1–A12 pass; registry seeded with the 14 incidents |
| M3 | construction transcripts (Claude Code sessions) and frontier trajectories ingested as traces |
| M4 | signals: grader bugs found → corrections; negatives that passed → weak-verifier defects; frontier failures → gate-level outcomes |
| M5 | milestones mined (build unblocked → texrefs gone → semantics bit-exact → concurrency-safe → perf restored) and compared to the hand-built grader |
| M6 | Φ per step for each recorded trajectory; validity on held-out PR traces; `steps.jsonl` emitted |
| M8 | span tasks (e.g. "people module only", "NCV only", "concurrency only") and trap tasks from recorded corrections |
| M11 | exported to Harbor, verifiers, verl, NeMo-Gym, OpenEnv; rewards identical |
| M12 | small sparse vs dense GRPO run on an open coder model |
| M13 | step-level report: gate profile, progress AUC, stage reach per model |

Tasks 01–04 are **never** placed in a benchmark split we publish as OOD. They show the pipeline. The OOD benchmark comes from forged tasks on unseen repos and API families (DESIGN §11.2).

---

## 7. Engineering rules

1. **Python 3.12, `uv`, pydantic v2.** One package with extras per interop target. Packs are separate installable packages.
2. **Tests per module:** unit, contract and property tests, plus `golden/` where historical data exists. Every gate in §4 is an automated test, not a document. GPU suites run on the lab 5090 through `scripts/lab_gpu.sh`. The local machine runs the CPU and toy suites.
3. **No gate is decided by an LLM.** LLMs may propose (milestones, rubric items, traps, simulated replies). Acceptance is executable or statistical.
4. **Provenance is always present.** Every artifact records what it was derived from and the content hashes of its inputs.
5. **Security:** secrets only via `env:NAME` from the gitignored `.env`; redaction before storage; leak audit before any model sees a task; nothing from `data/` is committed.
6. **Docs as we go:** each module starts with `docs/modules/Mxx-<name>.md` (spec, interfaces, gate) and ends with a gate report (numbers, failures, fixes) in the same file.
7. **One module at a time.** A module is done when its gate passes on `coding` and `_toy`, and its doc and tests are committed.

---

## 8. Next step

**M0 `core`** is first: the schemas, ids, provenance, registry, config, store and import-linter contracts, plus the `_toy` World spec and the import of the five seed task directories as schema-valid Crucible tasks. Every later module is written against these contracts, and nothing in M0 needs a GPU, a network or an API key.
