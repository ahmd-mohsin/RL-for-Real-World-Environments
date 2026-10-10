# Crucible — design document (v0.1, 2026-10-09)

Crucible turns **workflow traces that carry expert signal** into three products:

1. **RL environments** with dense, step-level rewards, through credit assignment.
2. **Tasks** at several horizons, calibrated so that frontier models fail.
3. **Benchmarks**, including out-of-distribution (OOD) ones, that score each step of a workflow.

The first vertical is **coding**, and the first workflow family is **legacy code modernization and migration of GPU kernels**: the family behind tasks 01–04 in `../migration_tasks/`.

This document is the contract for what we build. Where it borrows from existing frameworks, it names the file it borrows from (Appendix A). Where it encodes a lesson we learned the hard way, it names the incident (Appendix B).

Status: design only. No code exists yet. Section 13 lists the open decisions. `ARCHITECTURE.md` turns this design into a project structure and a module-by-module build plan, including what we take from each existing framework.

**Scope rule.** The framework is open-ended: any workflow, in any domain, at any size, from one tool and one actor up to a multi-app, multi-actor world with its own clock (§4.4, P19). Coding is the first *domain pack*, not an assumption of the core. Tasks 01–04 are the MVP showcase.

---

## 0. Summary

| | |
|---|---|
| **Input** | Traces of real work that already contain expert judgement. For coding: commits and PR series, code-review threads, CI results, reverts and hotfixes, agent session logs with human interrupts and edits, and written policy (migration guides, style rules). |
| **Core** | (1) Normalize traces into a **Workflow Graph** of stages and milestones. (2) **Assign credit** to each step from the expert signals. (3) Build tasks **solution-first** from the expert's own end state and corrections, then harden them with an agent swarm. (4) Accept a task only if it passes a strict **Acceptance Contract** (oracle passes, every negative fails, no flakes, no leaks, frontier-calibrated, grounded). |
| **Output** | Harbor-format task directories, extended with a dense-reward channel. Adapters run them on agentenv-framework, verifiers v1 (prime-rl), NeMo-Gym (NeMo-RL or verl) and OpenEnv (TRL), plus SFT data and versioned benchmarks with step-level metrics. |
| **Why frontier models still fail on tasks frontier models helped build** | **Asymmetric information.** The builder holds things the solver never sees: the expert's end state, the corrections reviewers made, the original program as a reference, and the hidden checks. Verifying with these is cheap; solving without them is hard. Task 04 was built with help from a frontier model, and GPT 6 Astra and GPT 5.6 sol both score 0 on it (Appendix B.1). |
| **Rule** | The LLM proposes; recorded reality decides. No reward comes only from a model's opinion. Every reward term is an executable check, a recorded outcome, or a rubric item that has passed a predictive-validity test against recorded outcomes. |

---

## 1. Problem statement

This section restates `../Documentation/Enterprise-Native RL Environments Thesis, Landscape, and MVP Plan.docx` (the thesis) and adds two changes:

- The MVP vertical moves from AP invoices to coding.
- Credit assignment is the central technical product, rather than one module among several.

1. **Expert-made environments do not scale.** There are too few experts and the work is slow. Enterprises already produce expert signal every day as a by-product of work: reviews, corrections, CI, outcomes. Nobody has wired that signal to a reward function.
2. **Long-horizon workflows give sparse rewards.** A repo-level migration runs 100–300 agent steps and gets one pass/fail bit at the end. GRPO learns only from groups that contain both successes and failures. On hard tasks almost every group fails, so the gradient is about zero. Dense, *correct* step credit is what makes long-horizon RL work, and no framework we surveyed provides it (Appendix A, row "Dense step rewards").
3. **Synthetic tasks become AI slop unless they are grounded.** A task generator graded by its own opinion trains the policy to imitate the generator. Every task property must trace back to a recorded expert signal, or to execution against a reference.
4. **Benchmarks score outcomes, not workflows.** A pass/fail bit cannot tell whether an agent failed at stage 2 or stage 9, or whether it recovered from a mistake. Enterprises need to know at which step the agent breaks.

**Vertical vs horizontal.** A vertical workflow involves one department (engineering does a migration). A horizontal workflow involves several departments working on one output (finance, procurement and AP resolve an invoice). The MVP is vertical (coding). The data model (§4) and the credit operators (§6) are domain-agnostic, so horizontal workflows reuse them with different connectors and environment simulators (§12.3).

---

## 2. Goals, non-goals, success criteria

### 2.1 Goals (MVP)
- **G0** A **domain-agnostic core**. Worlds, state, actors, signals, milestones, credit and acceptance are defined without reference to coding. Domain knowledge lives only in domain packs (§4.4). A non-coding toy pack has to pass through every module in CI.
- **G1** Ingest coding workflow traces with expert signals into a typed trace store.
- **G2** Build a Workflow Graph for one task family (legacy GPU migration): stages, milestones, and the dependencies between them.
- **G3** Turn expert signals into dense rewards that are **outcome-anchored** and **cannot be farmed**.
- **G4** Push tasks 01–04 through the whole pipeline as the MVP showcase, together with the sub-span tasks (H1–H3) the forge derives from them. Every task must pass the Acceptance Contract (§9), and frontier models must fail at a rate inside a target band. Scaling to 20–50 forged tasks is the first milestone after the MVP.
- **G5** Run an RL experiment on an open model: sparse reward vs dense reward, with equal compute.
- **G6** Ship an OOD benchmark with step-level metrics, plus a version that is regenerated dynamically.

### 2.2 Non-goals (MVP)
- A new RL algorithm. We use vanilla GRPO (thesis principle "boring optimizer"); all novelty goes into data, environments and credit.
- Production connectors for horizontal or non-coding systems (ERP, mail, ticketing). They are designed for (§12.3), and the core is held to them by the toy pack (G0), but none is built for production.
- A hosted SaaS. Everything runs locally or in the customer's cloud account; nothing leaves the building.

### 2.3 Success criteria

These are the MVP gates. Each one is a measurement, not a judgement.

| ID | Criterion | Threshold |
|---|---|---|
| S1 | Replay fidelity: replaying the expert's actions in the environment reproduces the expert's end state and CI outcomes | ≥ 90% of sampled traces |
| S2 | Validity of the dense reward: the progress potential at step t predicts the episode's outcome (AUC on held-out traces), and beats an LLM-judge baseline | AUC ≥ 0.75 and > the judge's AUC. **Measured 2026-10-09 (M6): fails at the midpoint** (Φ 0.50 vs judge 0.75 on 200 held-out SWE-rebench attempts: most attempts have not changed state by then); at the end Φ is better *within* a task (0.72 vs 0.67) and the two combine to 0.77. **Proposed revision (pending a decision):** measure Φ where state has started changing (75% or the midpoint of the state-changing phase) and report within-task AUC, the quantity group-relative advantages use |
| S3 | Every shipped task passes the Acceptance Contract | 100% |
| S4 | Frontier calibration: best frontier pass@1 on the benchmark split | ≤ 20% |
| S5 | Dense reward beats sparse on the RL run: success after N steps, at equal compute | statistically significant (bootstrap 95% CI) |
| S6 | OOD transfer: gain on the held-out repos / migration types / horizon split | > 0 with 95% CI, and no rise in the hack rate |
| S7 | Reward-hack audit: share of top-decile reward rollouts that fail a manual audit | ≤ 2% |

---

## 3. Design principles

The thesis's ten principles hold: the LLM proposes and records decide; rubric items must predict outcomes; experts audit rather than author; hindsight tasks first; generate at the frontier; grade with an agent; every defect becomes a check; train where the agent will live; boring optimizer; nothing leaves the building.

Nine more come from building tasks 01–04 and surveying nine frameworks:

- **P11 — Solution-first construction.** A task is built backwards from a known-good end state (the expert's) and known failure modes (the expert's corrections). The builder never needs to solve the task. It needs to *verify* the task, and it holds the evidence to do so.
- **P12 — Credit goes to states, not to actions.** A milestone is a predicate over the environment's state (the repo, the build, test results), not over the agent's action sequence. Agents take different paths; matching actions against the expert's (as NeMo-Gym's `swe_pivot` does) punishes valid alternatives. Action matching is allowed only as a weak, weight-0 diagnostic.
- **P13 — Two kinds of dense reward that are never confused.**
  1. **Temporal redistribution**, which preserves the optimal policy: potential-based shaping with Φ(terminal)=0. It moves credit to earlier steps without changing which policy is best.
  2. **Terminal partial credit**, which changes the objective. It is allowed only behind hard gates, capped, and dominated by the outcome.
  Section 6.4 gives the math.
- **P14 — The verifier is itself a tested artifact.** Every verifier must kill a set of **negative mutants**: oracle variants with each trap reverted, plus generic defect classes. Its **mutation score** is recorded. Only an oracle that passes and a verifier that kills every negative make a valid task (Appendix B.4: the NCV out-of-bounds negative passed until we added guard bands).
- **P15 — Flakiness is a reward bug.** Reward is all-or-nothing per gate, so any check that fails a correct solution with probability ε lowers the expected reward of the *correct* policy by ε. Every check gets a measured false-negative rate before it ships (Appendix B.2: the KinFuLS bifurcation failed correct solutions about 5% of the time).
- **P16 — Classify infrastructure failures separately from policy failures.** Harness crashes, empty patch collection, blank usage responses, GPU contention and stale images are *infra* failures. They are masked from training (verl's −999, NeMo-Gym's `mask_sample`/`failure_kind`), never scored 0 (Appendix B.5).
- **P17 — Isolation is layered and tested.** The agent image contains no tests, solution, references, or git history leading to them. Candidate code runs unprivileged. References are root-only. Grading happens in a separate verifier image. Outputs are sealed after each step. Every layer has an automated audit (§8.3).
- **P18 — Interoperate, don't fork.** The canonical task format is a Harbor task directory with a superset extension. Thin adapters export it to every major runtime and trainer, so a customer can train in their own harness (thesis principle "train where the agent will live").
- **P19 — One core, many worlds.** Every environment is a *World* built from Apps (state plus a tool surface), Actors (the policy, simulated people, peer agents, systems), an optional virtual Clock, and Events. Every check is a predicate over state read through a *State Adapter*.
  - **Scaling down:** a single repo with one shell tool and one actor is the smallest World.
  - **Scaling up:** a finance org with eight apps, four simulated departments and a clock is the same abstraction.
  - **Domain packs:** everything domain-specific (connectors, signal mappings, app templates, milestone and verifier libraries, generators, horizon ladders) lives in a pack. The core never imports a pack.

Eight more come from public post-mortems of RL environments and our own audits of them: Preference Model's "Building RL environments for superintelligence" series and its MIT framework Karotte, and Hugging Face's "How not to build coding environments" (Repo2RLEnv). Both found frontier models taking every shortcut the environment left open.

- **P20 — Grade outcomes, and the prompt asks only what the grader checks.** "If the grader can't check it, it cannot go into the prompt." An instruction the grader does not enforce is a lie the policy learns to ignore, and a grader that inspects the process (which commands were run) punishes valid alternatives (P12). Every sentence of a task's instructions maps to a check or is removed (Acceptance Contract, §9).
- **P21 — The environment enforces; the prompt never asks.** "Don't use the internet" and "don't read the git history" are not constraints; `network_mode: none` and a verified history scrub are. Every prohibition is an enforced property with an automated audit, and is tested by an attacker that tries it (`tests/redteam/`).
- **P22 — Accept environment awareness: security over realism.** A policy may notice it is in an environment (a user called `agent`, no network, a root-only directory). That costs little; an exploitable environment costs the run. When realism and security conflict, security wins.
- **P23 — Rewards never come from inside the sandbox.** Nothing the policy can write is read as a reward or a verdict: rewards are computed from snapshots taken after the policy's process cohort is killed (quiesce), by code and an interpreter the policy cannot modify. Harbor 0.23 trusts a `reward.json` the agent can pre-write (Karotte's comparison); Crucible never reads one from the sandbox.
- **P25 — One choke point.** Every tool call (the policy's, simulated people's, events') passes through `World._call_tool`: validation, ACL, deadline, crash containment, fault labels, output bounds. MiMo's anti-hack guard never ran on its own RL path because the RL agent called tools around it; a check that is not at the single choke point is a check that can be bypassed.
- **P24 — Deep, non-saturating tasks.** A task should reward doing *better*, not only passing: perf bars, quality margins and multi-property strictness keep the reward informative after correctness saturates (Appendix B.1: correctness-only migrations were within frontier reach; perf and concurrency bars separated the models).
- **P26 — Trust only channels the candidate cannot write.** In the verifier, candidate code runs unprivileged in its own processes and produces *artifacts*; a separate trusted process compares them with references the candidate could neither read nor write. Never evidence: an exit code, a summary line or JSON row on the candidate's stdout, a test binary its own build system produced, timing measured inside its process, a file under a path it can write, a consumer package it can shadow on `PYTHONPATH`. Found in our own seeds (2026-10-09 audit, Appendix B.10): task01's hidden suite was built by the candidate's CMake and judged by doctest's exit code (`DOCTEST_CONFIG_DISABLE` passes it), task03 trusted pytest's exit code (`os._exit(0)` under pytest passes it), task02 ran candidate code as root next to the benchmark it then re-ran. CI-as-a-tool results (exit code + `expect` regex) are feedback to the policy, never reward.
- **P27 — Answer-key probes are recorded, never scored; an exposure masks.** Every tool call that probes git history, a package registry, a code host, grader files, root or other processes is tagged in the trace (`data.probes`, M3 `agent_events.py`), with whether it *ran* and whether its output actually *exposed* an answer source (a `git log --all` listing only `base` ran and exposed nothing). Probing is not punished: exploring history is ordinary work, and grading the process punishes valid alternatives (P12, P20). But an outcome reached after an exposure is not evidence: the World masks the episode (`crucible:answer_exposed`, an environment failure under P21) and imported runs lose their label (Repo2RLEnv: "every green score was contamination").
- **P28 — Many agents, one World.** Swarms scale by peers sharing a workspace, not by a root that holds everyone's context (Prime Intellect, "On the nature of the swarm"). A World therefore drives any number of policy agents as peers: they coordinate through shared Apps (a git repo, an issue tracker, a mailbox) and direct messages under ACLs, with no orchestrator in the loop; persistent peers keep their memory in the `WorldRef`; and credit follows writes (every edit records what it wrote, so M4 grades a later signal to the agent, or sub-agent, that made the change). Communication is a cost the benchmark reports (§11.1).
- **P29 — Depth over bars; an honest way out.** Tasks that a fixed bar grades saturate: once a policy clears "implement this feature", there is no more signal. Deep tasks are ones an expert could pour unbounded time into (the highest-rank elliptic curve: 28 in 2006, 29 in 2024, 30 and 31 in August 2026), and their graders reward doing *better*, measured against the best known result, with no ceiling (`verify/runtime/patterns/record.py`: normalized score, 1 = the best known, above 1 = a new record with a log tail). We often will not know how to solve them; that is fine as long as the outcome is checkable (P20). Some may not be solvable at all, so a task may allow an honest **abstention**: full credit on a task constructed infeasible, nothing on a feasible one, a small constant (below the training band's floor) on an open problem; on an infeasible task any claimed solution earns 0 (§7.5).

---

## 4. Glossary and data model

### 4.1 Terms

| Term | Definition |
|---|---|
| **Workflow family** | A class of recurring work with a shared goal structure, e.g. "migrate a GPU module off removed CUDA APIs". |
| **Trace** | One recorded execution of a workflow: an ordered set of **events** (actions, observations, state snapshots) plus the **signals** attached to them. |
| **Expert signal** | Evidence about the quality of a step or outcome produced by someone or something *other than* the actor: reviewer, CI, later revert, user correction. Typed in §5.2. |
| **State** | The environment's observable world at a point: for coding, repo tree + build artifacts + test/CI results (+ perf numbers). |
| **Milestone** | A named predicate `φ: State → [0,1]` (usually {0,1}) that marks progress, with prerequisites, a weight and provenance. |
| **Stage** | A contiguous phase of a workflow defined by entry/exit milestones (e.g. *build unblocked*, *API replaced*, *semantics preserved*, *concurrency-safe*, *perf restored*). |
| **Workflow Graph** | Family-level DAG of stages and milestones learned from traces (§5.3). |
| **Horizon level H0–H5** | Task length class (§7.3). |
| **Gate** | A hard predicate whose failure zeroes reward (e.g. builds, no forbidden constructs, correctness). |
| **Reward channel** | One named component of reward (outcome, gate, milestone/potential, rubric item, efficiency), reported separately (`reward.json` multi-key, Harbor; `reward_components`, NeMo-Gym). |
| **Oracle** | Known-good solution; for grounded tasks usually derived from the expert's end state. |
| **Negative** | A known-bad solution (mutant) that must score 0 on a specified gate. |
| **Task card** | Provenance + value proof attached to every task (§4.3.4). |
| **World** | The environment a task runs in: a set of Apps, Actors, an optional Clock and Event schedule, and the written policies in force (§4.4). |
| **App** | One system in a World: a state store, the tool surface the policy can call (MCP, shell, HTTP, browser), and optionally a UI. Examples: a git repo with a GPU toolchain; an ERP database. |
| **State Adapter** | The typed way the *verifier* reads, snapshots, restores and diffs one App's state (filesystem+git, SQL, document store, HTTP mock, browser DOM, process/GPU). The policy never uses it. |
| **Actor** | Anyone or anything that acts in a World: the policy under training, simulated humans (users, reviewers, approvers, other departments), peer agents, and system actors (CI, cron, mail delivery). |
| **Domain pack** | A plugin that holds everything domain-specific: connectors, signal mapping, App templates, milestone and verifier libraries, generators, horizon ladder. `coding` is the first pack. |

### 4.2 Trace schema (`crucible.trace/v1`)
Agent trajectories reuse **ATIF** (Harbor `src/harbor/models/trajectories/`, v1.8; NeMo-Gym exports v1.7) so existing harness converters work. Crucible wraps ATIF in a workflow-level envelope that also holds non-agent events (commits, reviews, CI) and signals.

```yaml
trace_id: pcl-gpu-cuda12/upstream-pr-6012          # stable id
family: legacy_gpu_migration
source: {connector: github, repo: PointCloudLibrary/pcl, ref: "pull/6012", privacy: public}
actors: [{id: a1, role: author, kind: human}, {id: r1, role: reviewer, kind: human}, {id: ci, kind: system}]
events:                                            # totally ordered by (time, seq)
  - {seq: 0, kind: state, state_ref: "git:c4719882", time: ...}
  - {seq: 1, kind: action, actor: a1, type: commit, diff_ref: "git:ab12..", message: "replace texture refs in multi_tree.cu"}
  - {seq: 2, kind: observation, actor: ci, type: ci_run, ref: "ci:...", results: {build: pass, tests: {passed: 412, failed: 3}}}
  - {seq: 3, kind: action, actor: r1, type: review_comment, path: gpu/people/src/cuda/multi_tree.cu, line: 88,
     body: "tex1Dfetch returned 0 out of range; __ldg reads garbage here"}
  - {seq: 4, kind: action, actor: a1, type: commit, diff_ref: "git:cd34..", message: "bounds-check fetch"}
  - {seq: 5, kind: agent_session, atif_ref: "atif://sessions/9f2e.json"}   # embedded agent trajectory (optional)
outcome: {merged: true, reverted_within_days: null, hotfixes: [], label: good, label_source: "merged+30d no revert"}
signals:   # derived, see §5.2; each points at event seqs it grades
  - {id: s1, type: review_defect, grades: [1], target: {path: ..., lines: [80,95]}, polarity: -1, source_event: 3}
  - {id: s2, type: correction, grades: [1], corrected_by: 4, polarity: -1}
  - {id: s3, type: ci_regression, grades: [1], tests: ["test_people_labels"], polarity: -1, source_event: 2}
```

### 4.3 Task schema: a Harbor superset

A Crucible task is a valid **Harbor task directory**: schema 1.4 `task.toml`, `instruction.md`, `environment/`, `tests/`, `solution/`. Any Harbor-compatible runner can execute it, and so can NeMo-Gym's `harbor_agent`, OpenEnv's `openenv harbor`, and verifiers' Harbor taskset. Crucible adds the following:

```
task/
  task.toml                 # Harbor schema 1.4: [environment] gpus, network_mode=no-network; [verifier] environment_mode="separate"; [[verifier.collect]]
  instruction.md
  environment/Dockerfile    # agent image (no tests/solution/refs/history)
  environment/Dockerfile.verifier   # verifier image (refs, hidden runtime), root-only paths
  tests/test.sh             # writes /logs/verifier/reward.json (multi-key) + steps.jsonl
  tests/...                 # hidden checks
  solution/solution.patch   # oracle
  crucible/
    card.yaml               # §4.3.4 provenance + value proof
    milestones.yaml         # §6.2 state predicates, weights, prerequisites, provenance
    negatives/*.patch       # §9 mutants, each with expected failing gate
    calibration.json        # §9 frontier runs, flake/bifurcation stats, perf headroom
```

#### 4.3.1 `reward.json` (terminal, multi-key)
The full document is `crucible_reward.json` (`core/schema/reward.py` `RewardJson`):
```json
{"reward": 0.0,
 "gates": {"build": 1, "static": 1, "binary": 1, "correctness": 1, "concurrency": 1, "sanitizer": 1, "perf": 0},
 "outcome": 0.0,
 "progress": 0.86,
 "rubric": {"validated_items_met": 0.75},
 "efficiency": 0.4,
 "infra": {"status": "ok"},
 "components": {"check/t1": 1.0}}
```
`reward.json` is its **flat numeric projection** (`RewardJson.flat()`): `{"reward": 0.0, "outcome": 0.0, "gate/build": 1.0, …, "infra/masked": 0.0}`. *Corrected in M7 (2026-10-09):* Harbor rejects any non-numeric value in `reward.json` (`harbor/verifier/verifier.py:104-121`), so the nested document cannot live under that name. `reward.txt` is always written too, and a grader pre-writes reward 0 before it starts (an interrupted grade never leaves a stale or agent-written reward). `gates` holds hard constraints only; outcome and score checks are `components["check/<gate>"]`.

`reward` is the **training scalar** defined by the task's reward formula (§6.4). The other keys are **reported channels**, readable by trainers that accept multi-component rewards (NeMo-Gym `reward_components`, verifiers `@vf.reward` / `@vf.metric`).

#### 4.3.2 `steps.jsonl` (dense channel)
One line per scored step boundary. A step boundary is one agent turn or tool call, identified by ATIF step id or turn index.
```json
{"step": 37, "atif_step_id": "s37", "phi": 0.42, "delta_phi": 0.08, "milestones_new": ["api_replaced/texrefs_gone"], "milestones_lost": [], "violations": [], "cost_s": 3.1}
```
The trainer adapters (§10.3) convert these lines into per-turn advantages.

#### 4.3.3 `milestones.yaml`
```yaml
family: legacy_gpu_migration
milestones:
  - id: build/unblocked
    check: {kind: build, target: [gpu_kinfu, gpu_kinfu_large_scale, gpu_people], cuda: "12.8", arch: sm_120}
    weight: 1.0
    cost: medium          # how often it can be evaluated during an episode
    provenance: [{signal: "ci:build_fail@base", note: "modules disabled on CUDA>=12"}]
  - id: api/texrefs_gone
    check: {kind: static_scan, rule: "no texture/surface references in gpu/**"}
    requires: []
    weight: 1.0
    cost: cheap
  - id: semantics/ncv_bitexact
    check: {kind: diff_vs_reference, suite: ncv_check, seeds: [0,5,31337]}
    requires: [build/unblocked]
    weight: 2.0
    cost: expensive       # terminal only, or sampled
    provenance: [{signal: "review:multi_tree.cu#L88"}, {signal: "expert_end_state"}]
```

#### 4.3.4 Task card (`card.yaml`)

Every task carries its provenance and a **value proof**. A task without a value proof is not shipped (anti-slop).

```yaml
id: crucible/legacy_gpu_migration/pcl-gpu-cuda12@v3
family: legacy_gpu_migration
horizon: H4
grounding:
  traces: [pcl-gpu-cuda12/upstream-pr-6012, ...]
  oracle_source: expert_end_state | builder_oracle | both
  checks_with_provenance: 41/44            # share of hidden checks tied to an expert signal or the reference
value_proof:
  frontier: [{model: gpt-6-astra, pass: 0, failing_gates: [perf]}, {model: gpt-5.6-sol, pass: 0, failing_gates: [ncv, fresh]}]
  oracle: {passes: 3/3, min_headroom: 0.11}
  mutation_score: 1.0 (6/6 negatives killed)
  flake: {oracle_false_negative_rate: 0/120 induced+natural draws}
  leak_audit: pass
splits: {train: false, bench: true, ood_axes: [repo, api_family]}
```

### 4.4 World spec (`crucible.world/v1`): open-ended environments

A task names a World. The same schema describes the smallest coding task and a multi-department workflow; only the parts listed change.

```yaml
# smallest: task 04 (one repo, one actor, no clock)
world: coding/pcl-gpu-cuda12
apps:
  - {id: repo, template: coding/git_repo, state: {adapter: fs_git, root: /workspace/pcl}}
  - {id: gpu, template: coding/gpu_toolchain, state: {adapter: process}, resources: {gpus: 1, arch: sm_120}}
actors:
  - {id: agent, kind: policy, tools: [repo.shell, repo.edit]}
```

```yaml
# larger: an accounts-payable exception (four apps, four actors, a clock and events)
world: finance_ap/invoice-exception
apps:
  - {id: erp,    template: finance_ap/erp,          state: {adapter: sql, engine: postgres}}
  - {id: mail,   template: common/mailbox,          state: {adapter: sql, engine: sqlite}}
  - {id: portal, template: finance_ap/vendor_portal, state: {adapter: browser}}
  - {id: policy, template: common/doc_library,      state: {adapter: fs}}
actors:
  - {id: clerk,       kind: policy, tools: [erp.*, mail.*, portal.*, policy.read]}
  - {id: procurement, kind: simulated, grounded_on: traces/procurement_replies, tools: [erp.po.*]}
  - {id: vendor,      kind: simulated, grounded_on: traces/vendor_threads}
  - {id: approver,    kind: simulated, policy_doc: policy/approval_matrix.md}
clock: {t0: "2026-03-02T09:00:00Z", rate: 3600}       # 1 real second = 1 virtual hour
events:
  - {when: {after: "2d"}, actor: vendor, action: {type: nl, text: "credit note issued"}}
```

**Rules.**
- Milestones and checks reference state only through State Adapters: `{adapter: sql, app: erp, query: ..., expect: ...}` or `{adapter: fs_git, app: repo, check: build}`. This is what makes credit assignment (§6) domain-agnostic.
- Simulated Actors must be **grounded**: they answer only from recorded history or written policy, and say "no record" otherwise (§5.1, grounded responders).
- Worlds snapshot and restore as a unit. That is what makes branch-rollout values (C3), replay (§5.3) and sub-span tasks (§7.3) possible in any domain.

`ARCHITECTURE.md` (M1, M2) shows how a World runs on Harbor (single- or multi-container sandboxes) or on agentenv-framework (MCP/DB/website environments with a gateway clock, triggers, peer and human agents), and how Apps are packaged in the MiMo sidecar format.

---

## 5. Layer 1: capture, normalize, model the workflow

### 5.1 Connectors (plugin interface)

```python
class Connector(Protocol):
    name: str
    def discover(self, scope: Scope) -> Iterator[SourceRef]: ...          # e.g. PRs matching a query, sessions in a dir
    def fetch(self, ref: SourceRef) -> RawRecord: ...                     # raw, privacy-tagged
    def normalize(self, raw: RawRecord) -> Trace: ...                     # → crucible.trace/v1 (+ATIF)
    def redact(self, trace: Trace, policy: RedactionPolicy) -> Trace: ... # PII / secrets, in-tenant
```

Connectors for the coding MVP:

| Connector | Source | Signals produced |
|---|---|---|
| `github` / `gitlab` | PRs, commit series, review threads, suggested changes, check runs | review_defect, correction, suggestion_accepted, approval, ci_* |
| `ci` | GitHub Actions / Buildkite / Jenkins logs; per-commit test and benchmark results | ci_pass / ci_regression / perf_delta per step |
| `vcs_history` | follow-ups after merge: reverts, hotfixes to the same lines within N days (via blame) | revert, hotfix → negative outcome on the step that wrote those lines |
| `agent_sessions` | Claude Code / Codex / Cursor / mini-swe-agent / OpenHands transcripts → ATIF (reuse Harbor's 40+ converters, `src/harbor/agents/installed/`) | user_interrupt, edit_rejected, manual_fix_after_agent |
| `capture_proxy` | OpenAI- or Anthropic-compatible proxy in front of the enterprise's live agents: tokens, logprobs, tool calls (OpenEnv RFC 006, NeMo-Gym `token_id_capture`, verifiers `interception/server.py`) | passive trace collection, with no change to the agent |
| `policy_docs` | CONTRIBUTING, style guides, migration guides, ADRs, CODEOWNERS | candidate rubric items (must pass §6.5) |
| `issues` | Jira / Linear / GitHub issues: goal statements, acceptance criteria | task instructions, acceptance checks |

Privacy is enforced by the connector contract: `redact()` runs inside the tenant, and the trace store is tenant-local. Public traces (open-source upstream) are marked `privacy: public` and are what the MVP uses (§11.2).

**As built (M3, 2026-10-09; `docs/modules/M03-connectors.md`).**
- **Generic connectors (core):** `file_drop`, `sql_audit_log`, `otel_import` (OTel GenAI conventions), `capture_proxy` (OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, streaming, per-session capability keys, retries collapsed to the attempt the agent saw), and `atif` (Harbor converters, DEPEND).
- **Coding pack:** `github_pr` (plus git-only discovery with no API), `claude_code_sessions`, `agent_runs`, `ci_logs` (JUnit / CTest / GoogleTest / pytest / build logs; infra failures masked per P16; flaky tests listed), `policy_docs` (normative rules with modality, config rules, revision history).
- **Redaction:** provider key formats, bearer tokens, the tenant's known secret values, and human actor ids as well as handles. The gate found a real Docker Hub token and a RunPod key in our transcript that the first patterns missed.
- **Every agent-session connector, and the World itself, tags answer-key probes (P27).**
- **Not built:** `issues` and `gitlab` (they need API calls).

### 5.2 Taxonomy of expert signals in coding

| Signal | Where it is recorded | Credit it implies | Reliability | Use |
|---|---|---|---|---|
| CI result per commit | check runs | outcome of the state after that step | high (deterministic tests; flaky tests are filtered as in SWE-smith `harness/repair.py`) | milestone evidence; step outcome |
| Review comment requesting a change, at a line | review thread | **negative** credit on the step that introduced those lines (blame inside the PR) | medium–high | defect → check (§6.3); rubric candidate |
| Accepted suggestion / the next commit fixing a commented line | review + diff | (wrong, corrected) action pair | high | negative mutant; DPO pair; trap synthesis |
| Revert or hotfix within N days | history after merge | strong **negative** for the originating step | high (survives review) | regression check; outcome label |
| Approval + merge, then 30 days without revert | PR state + history | positive terminal outcome | medium (reviews miss bugs) | outcome label |
| Benchmark numbers in the PR / CI | PR text, perf CI | target perf bar | medium (noisy; recalibrate) | perf gate threshold |
| User interrupt or rejected edit in an agent session | session log | **negative** on that tool call | medium | step label; PRM training |
| Manual edit after the agent finished | session + diff | (agent output, expert fix) pair | high | correction mining |
| Policy text (migration guide) | docs | candidate rubric item | low until validated | rubric candidate (§6.5) |

### 5.3 Workflow modeling

Segment traces into steps and stages, then mine the family-level Workflow Graph.

1. **Step segmentation.** Coding traces already arrive segmented by commits, review rounds and CI runs. Agent sessions are segmented by tool call; consecutive read-only calls are grouped into one "investigate" step.
2. **State reconstruction.** For every step boundary, materialize the state (`git checkout <sha>`, then build, then tests) inside the family's environment image. This is the **replay test**: replaying a trace must reproduce the recorded CI outcomes (S1). Traces that fail replay are quarantined. The environment may be missing a dependency (Patronus's "dropped dependency" risk).
3. **Milestone mining.** Propose milestone predicates from three sources, then filter them:
   - *Diff-derived:* removing a construct ("no `texture<` left"), adding a capability ("module X builds").
   - *CI-derived:* tests that go fail→pass across the trace (SWE-smith's F2P logic in `harness/grading.py`, applied per step instead of per instance).
   - *Signal-derived:* the check a review comment demanded.

   Each candidate milestone must satisfy all of the following:
   - (a) it is **executable** on any state;
   - (b) it is **monotone on successful expert traces**: once satisfied, it stays satisfied;
   - (c) it is **non-trivial**: false at the base state;
   - (d) it **predicts the outcome**, by the validity test in §6.5.
4. **Stage induction.** Cluster milestones by when they are first satisfied, across traces of the family. Use a precedence graph whose edges are "A precedes B in at least 90% of traces", plus LLM labeling of the clusters. Stage *names* are proposed by the LLM; stage *membership* is decided by the precedence data.
5. **Workflow State Tracker** ("where am I?"). At runtime, given the agent's current state and trajectory prefix, it returns:
   - the current stage;
   - Φ, the progress potential (§6.2);
   - the satisfied, regressed and next-expected milestones;
   - any deviations: forbidden actions, policy violations.

   It uses cheap milestones every step, medium milestones at checkpoints (on build/test tool calls, or every k steps), and expensive ones only at the end. The tracker is what lets a long-horizon agent's output be *located* in the workflow, and it is also the labeling engine that turns existing enterprise agent logs into step-credited data.

**As built (M5, 2026-10-09; `docs/modules/M05-workflow.md`).**
- **Segmentation** classifies each call (investigate, edit, build, test, run, communicate, act, submit) with quote-aware shell parsing.
- **State reconstruction** for agent sessions replays editor calls (OpenHands, Claude Code, World tools) and the file effects of shell commands from their text, with a strict diff applier.
- **Replay outcomes:** what cannot be replayed is quarantined. Files no action names are execution byproducts, reported separately. On 913 public OpenHands runs, 94.9% reproduce their recorded final state (91.1% strictly).
- **Event-log state** (tickets, ERP rows) is rebuilt from `writes` and replays every recorded check.
- **Milestones** are mined from diffs (constructs removed or added, the expert's block changed, files created or deleted), from CI (fail-to-pass per step) and from signals, then filtered and matched to steps along prerequisites (ToolSandbox's DAG-matcher pattern).
- **Stages** are layers of a ≥ 90% precedence graph.
- **The tracker** reports stage, Φ, milestones gained and lost, the next expected milestones and deviations; κ = 0.87 against manual stage labels (toy).
- **Rediscovery**, on 55 PCL GPU PRs, audited: build, API removal, edge semantics and concurrency come from code; perf only from review comments; **bit-exactness is not rediscovered**. It comes from reference-differential checks (§7.1), a source upstream history does not contain, so that source is required for any family whose tasks grade exactness.

---

## 6. Layer 2: credit assignment (the core)

### 6.1 Credit operators

Each operator is a pure function from (trace or rollout, family graph, signals) to step credits. They compose; all of them are versioned and unit-tested.

| Operator | Input | Output | Policy-invariant? |
|---|---|---|---|
| **C1 Milestone potential** | state at each step | Φ(s_t), ΔΦ_t | yes, as shaping (§6.4) |
| **C2 Defect → check** | review defect, correction, revert, hotfix | new hidden check; negative mutant | n/a (verifier hardening) |
| **C3 Branch-rollout value** | snapshot at step t; K continuations | V̂(s_t) = success rate from s_t | yes (advantage estimate) |
| **C4 Validated rubric item** | policy text, or a review pattern that recurs | binary item; weight from predictive validity | no (terminal partial credit; capped) |
| **C5 Process reward model (PRM)** | (state, step) labels from C1–C4 | learned step score | used only as a shaping baseline, never as the main reward |
| **C6 Infra mask** | harness and runtime errors | mask flag | n/a |

### 6.2 C1: the milestone potential

Milestones m_i have predicates φ_i ∈ [0,1], weights w_i > 0, and prerequisite sets pre(i). Define:

  Φ(s) = ( Σ_i w_i · φ_i(s) · Π_{j∈pre(i)} φ_j(s) ) / Σ_i w_i ∈ [0,1]

The prerequisite product stops credit for out-of-order "progress" that is not real, for example passing a test before the module builds, which only happens by stubbing. φ is evaluated at the cost level allowed at that step; an unevaluated expensive milestone keeps its last value. Regressions (φ falling back to 0) **lower** Φ automatically, so breaking something that used to work is penalized without extra rules.

### 6.3 C2: defect → check

This is the flywheel from the thesis and from Patronus. Every signal with negative polarity becomes a candidate hidden check:

1. Locate the defect: path, line range, and the construct involved (e.g. an unbounded `__ldg` replacing a `tex1Dfetch`).
2. Synthesize a check that fails on the pre-correction state and passes on the post-correction state. Prefer behavioral checks: differential tests against the reference, sanitizer runs, edge-case inputs. Fall back to static rules only when no behavioral check is possible.
3. Validate the check:
   - it kills the "wrong" version;
   - it passes the expert end state and the oracle;
   - it is not flaky (§9 A4).
4. Register both the check and the mutant (the "wrong" version becomes a negative, §9 A3).

Example from task 04: "out-of-range texture fetches return 0" became the guard-band and exact-allocation tiny-ROI filter checks. Those checks are what caught GPT 5.6 sol's regression (Appendix B.4).

### 6.4 Reward composition

Episode states s_0..s_T, terminal state s_T. Definitions:

- **Gates:** G(s_T) = Π_g 1[g(s_T)].
- **Outcome:** O(s_T) ∈ {0,1}, the family's main success criterion. For migrations, that's all correctness checks plus all perf bars.
- **Shaping:** with Φ(terminal) := 0 and γ = 1, the per-step shaping term is r_t^sh = Φ(s_t) − Φ(s_{t−1}). The terminal step adds −Φ(s_{T−1}). The shaping terms telescope to −Φ(s_0), a constant. Shaping therefore **cannot change which policy is optimal** (Ng, Harada & Russell 1999). It only moves credit to earlier steps, which is exactly what fixes the long-horizon gradient problem.
- **Terminal partial credit** (optional, changes the objective): P(s_T) = Φ(s_T) and the validated rubric share p.

Training scalar, extending the thesis formula:

  **R = G(s_T) · [ O(s_T) · (1 + α·p + β·e) + κ · (1 − O(s_T)) · Φ(s_T) ]**

with κ ≤ 0.3, α ≤ 0.1 and β ≤ 0.05:

- Any gate failure gives 0, so a gate can never be traded for partial credit.
- A full success is worth at least 1, and a failure at most κ < 1. So partial credit can never beat finishing.
- Partial credit exists only *inside* the gates. It gives GRPO variance on hard tasks where every rollout fails (all-fail groups get 0 advantage under sparse reward).

**Open-ended objectives (P24; Preference Model: "reward doing better rather than clearing a fixed bar").** For families with a measurable objective (speedup, memory, accuracy, cost), the efficiency channel e can be *record-relative* instead of a bar:
- e = clip(log(m / bar) / log(record / bar), 0, e_max), where m is the policy's measured value, bar is the pass bar and record is the best value seen so far for the task.
- A run that beats the record gets e > 1, up to e_max, and the record moves, so the task does not saturate.
- m is measured outside the candidate's process on fresh inputs (P26), with a noise band from repeated oracle measurements (P15).
- e only ever adds to a full success (it sits inside O · (…)), so it cannot trade against correctness.
- Designed, not built (M6/M9).

Per-turn advantage for multi-turn GRPO. Let A^GRPO be the usual group-normalized episode advantage. The turn-level advantage is:

  A_t = A^GRPO + η · ( Σ_{u≥t} r_u^sh − mean_group ) / std_group

**Correction (M6, 2026-10-09).** With Φ(terminal) := 0 and γ = 1 the shaping return-to-go telescopes to −Φ(s_{t−1}). It is a state *baseline*: it lowers variance but carries no information about the action at t, and even lowers the advantage of turns taken from good states. Local credit uses the one-step term instead:

  A_t = A^GRPO + η · (ΔΦ_t − c) · σ_R / σ_ΔΦ

- c centres ΔΦ to zero net mass across the group, weighted by the turns' tokens. A net negative shift flattens the policy and lengthens outputs (verl fork).
- σ_R / σ_ΔΦ scales the term in outcome units (Dr.GRPO: no std normalization).
- Masked episodes are excluded from every group statistic.
- This term is a process reward (biased, P13 type 2), so η stays small. The gated scalar R must still carry κ(1−O)Φ, or dynamic-sampling filters evict all-fail groups.
- `credit/compose.py` implements both the baseline and the delta mode.

Injection points:
- verl: after `compute_advantage_for_multi_trajectories` (`trainer_base.py`), fed by `llm_turn_spans`. Contrary to an earlier version of this document, the fork's `arvo_penalties.py` does *not* rewrite turn-level advantages (it marks spans and has a call bug); Crucible ships the operator itself;
- verifiers v1 `MessageNode.advantages` / `loss_weights` (`verifiers/v1/graph.py`);
- OpenEnv `TrajectoryRubric.compute_step_rewards()`.

η is tuned once per family. In expectation the term is zero-mean within a group, so it lowers variance without adding bias to the outcome objective.

C3 (branch-rollout values) gives an alternative advantage, in the style of VinePPO: A_t = V̂(s_t) − V̂(s_{t−1}), where V̂ is estimated by K continuations from a snapshot taken at step t. It costs K× rollouts per branched step, so it is used offline:
- to label PRM data;
- to validate C1 (does ΔΦ correlate with ΔV̂?);
- on a sampled subset of steps during training.

Snapshot support comes from agentenv's snapshot extension and Harbor's `trajectory.json` prefix seeding with `is_copied_context`.

**Properties every reward function must pass** (unit tests in `crucible/credit/tests`):
- **R1** An oracle rollout gets the maximum reward.
- **R2** A no-op gets 0.
- **R3** Every registered negative gets 0.
- **R4** Oscillating rollouts that break and fix the same thing do not earn more than monotone ones (shaping telescopes).
- **R5** A rollout that satisfies milestones by stubbing (with prerequisites unmet) earns ≤ the same rollout with the stubs removed.
- **R6** Reward is a function of state only, not of the agent's text claims (anti-sycophancy; for terminal reward it is computed in the separate verifier).

### 6.5 Predictive-validity gate (anti-slop for rubric items and milestones)

A rubric item or milestone becomes reward-bearing only if it predicts recorded outcomes on **held-out historical traces**. The procedure (thesis §2.4, made concrete):

1. **Labels.** For coding, a trace is *good* if it was merged, had no revert or hotfix to its lines within 30 days, and passed CI. It is *bad* if it was reverted, hotfixed, or abandoned after review.
2. **Score.** Score each item on each trace's states. Measure (a) the AUC of the item's value at the final state for the trace label, and (b) for milestones, the AUC of Φ at the *midpoint* state, which tests whether the item is early-warning.
3. **Keep or drop.** Keep an item if AUC ≥ 0.6 on a *time-split* holdout, with the same sign on each repo subset. Item weight = max(0, 2·(AUC − 0.5)).
4. **Record.** The validity report is part of the task card. A competitor without the customer's outcome history cannot run this test, which is the moat the thesis names.

**Measured (M6, 2026-10-09; `docs/modules/M06-credit.md`).**
- **Coding** (SWE-rebench, time-split by issue date, 200 held-out attempts):
  - gold-derived milestone *types* reach a final-state AUC of only 0.43–0.72 per type;
  - "located" (the expert's block changed) is the most predictive type;
  - token milestones are brittle across implementations: correct alternative solutions score low Φ.
- **Fix:** behavioural (fail-to-pass) milestones, and validation within tasks.
- **The LLM judge used as the S2 baseline is a strong process-level predictor early in an attempt.** Credit for exploration therefore comes from C3 values or a judge-trained C5 PRM, never from Φ (P12).

---

## 7. Layer 3: Task Forge (agent-swarm synthesis)

### 7.1 Why frontier-assisted construction can still produce frontier-hard tasks

There are three sources of asymmetry; a task must use at least two of them:

1. **Privileged end state.** The builder has the expert's final commit or the original reference build. The solver has only the starting state and the instruction.
2. **Privileged failure knowledge.** The builder knows which traps experts fell into, from review comments, corrections and reverts, and turns them into hidden checks. The solver has to discover them.
3. **Strict, multi-property verification.** Hidden checks cover properties solvers tend to skip under budget: bit-exactness, concurrency safety, sanitizer cleanliness, determinism, perf bars against the original. The builder verifies these by differential execution, not by solving.

Empirical evidence: tasks 03 and 04 v3 were built this way and defeat GPT 5.6 sol and GPT 6 Astra on a single attempt (Appendix B.1).

### 7.2 Swarm roles

Each role is an LLM agent with a narrow toolset and a typed output. The orchestrator is deterministic; no role grades its own output.

| Role | Input | Output | Cannot see |
|---|---|---|---|
| **Miner** | trace store, family graph | task seeds: (trace, start step, end step, traps) | — |
| **Architect** | seed, horizon target | task spec: instruction, requirements, workspace layout, consumers/tools | hidden checks of other tasks |
| **Environment builder** | spec | agent Dockerfile, verifier Dockerfile, pinned sources, consumer tools | — |
| **Oracle builder** | spec, expert end state | `solution.patch` (expert-derived where possible; otherwise built and verified) | — |
| **Test writer** | spec, signals, reference | hidden checks with provenance; `milestones.yaml` | the oracle diff (avoids tests overfitting the oracle's internals) |
| **Adversary** | spec, oracle, checks | negatives: trap reversions, defect-class mutants, cheats (hard-coding, test tampering, history spelunking) | — |
| **Auditor** | task dir | leak audit, isolation audit, flake measurement, instruction↔test coverage (Harbor `harbor check` rubric), reference self-consistency | — |
| **Calibrator** | accepted task | frontier and target-policy pass rates; failure taxonomy; difficulty knob adjustment | — |
| **Curator** | task pool | dedup (semantic + content hash), difficulty ladder, split assignment, contamination check | — |

**Coordination (P28).** The orchestrator is deterministic but holds no content: roles coordinate through a shared task repository (drafts, checks, negatives as commits) and an issue tracker (audit findings, calibration results), so no single context has to hold every task. Roles are persistent specialists per family: the auditor that found a family's defects keeps them, instead of a fresh agent rediscovering them. Tasks may target problems nobody can solve yet (Preference Model's elliptic-curve rank) as long as the grader checks the outcome (P20).

**Built in M8 (2026-10-09).** The orchestrator is a deterministic DAG with a journal (resume by input hash), a reserve/settle budget, retries that carry validator findings, and task states bound to the content hash (`forge/orchestrator.py`). "Cannot see" is enforced by staging, not prompts: a role receives only its visible inputs. The oracle's output carries a canary; a node with no visible path to the oracle that emits it fails as a breach. No role grades its own output: acceptance is the deterministic M9 subset (`accept/contract.py`: A1 A2 A3 A4-lite A5 A7 A8). Valid alternatives (different correct solutions that must pass) are part of A3, which catches checks that grade the oracle's internals instead of the property.

### 7.3 Horizon levels for coding

| Level | Typical scope | Agent steps | Example in this family |
|---|---|---|---|
| H0 | single edit / function | 1–10 | replace one texture fetch, preserving semantics at the boundary |
| H1 | one file, one bug or change | 10–40 | port one `.cu` file; kill the out-of-bounds read; keep bit-exactness |
| H2 | multi-file feature or fix | 40–100 | port a module's marching cubes with per-instance state |
| H3 | module-level migration | 100–200 | gpu_people + NCV, all exactness checks |
| H4 | repo-level migration + perf + concurrency | 150–400 | task 04 v3 |
| H5 | cross-repo / cross-team | 300+ | migrate a library and its downstream consumers owned by "other teams" (the first step toward horizontal) |

**Decomposition generator.** One expert trace yields tasks at several horizons. Use start state = the expert's state at step i, goal = the milestones satisfied by step j. The H4 task's trace then provides H1–H3 subtasks, which are graded with the same milestone checks. This gives a curriculum, and it is the basis for step-level benchmarks (§11).

*As built (M8):* task04 v3 spans are module-scoped (people, NCV, KinFu, KinFu-LS). The start state is the base plus the oracle's other modules, with **sibling forks excluded**: KinFu-LS is a fork of KinFu, and with the sibling's port in the start, 67–70% of a span's new lines were already visible. A5 measures this overlap (≤ 20%). A span's hidden gates are the seed's behavioural suites restricted to its scope; mined milestones feed Φ only, never gates, because gold-token milestones fail correct alternative implementations (M6 finding 4).

### 7.4 Generators (all parameterized)

These follow Frontier Learning's procedural-generator idea: each generator's parameters form a search space.

- **G-hindsight:** replay a finished trace. Inputs as at the start, the answer hidden, graded against what really happened and held up (thesis principle 4). *Built for the toy ticket logs (M8):* the net outcome after corrections is the answer; the expert's corrected first response is a trap negative citing its signal; rules a correction revealed become invariants.
- **G-span:** take a sub-span (i, j) of a trace, as in §7.3. Difficulty grows with the span length and the number of traps inside it.
- **G-trap:** inject a review-derived trap into the task's surface. Example: the task's consumer exercises the boundary condition a reviewer caught.
- **G-transplant:** apply the family's migration pattern to a *different* repo that has the same legacy construct, found by static search. The oracle is built and verified by the oracle builder against the original. This is the main source of **OOD** tasks.
- **G-compose:** union of compatible subtasks; harder through interactions (SWE-smith `bug_gen/combine/`).
- **G-tighten:** raise the verification strictness of a solved task. Examples: add perf bars, concurrency bars or sanitizer tools, which is what took task 04 from v1 to v3. Raising the bar is only allowed if the oracle still has ≥10% headroom.

**Learnability steering.** When generating RL training tasks, target a pass rate p ∈ [0.1, 0.7] for the *policy being trained*, using a proposer reward of 4p(1−p) as in verifiers `environments/proposer_solver`. When generating benchmark tasks, target frontier pass@1 ≤ 20%. Difficulty is adjusted with the generator knobs, never by making the instruction more obscure.

### 7.5 Depth: record-relative graders and honest abstention (P29)

The forge prefers depth to breadth. A task is deep when more effort keeps paying: a faster kernel than the best known, a larger rank, a tighter bound, a smaller counterexample, a migration that is also faster than the original. Such tasks are graded with **score gates** (`Gate(kind="score")`): validity checks are hard gates; the score check maps the result onto the known scale, `(score − baseline) / (best_known − baseline)`, so the reward grows past 1 when the policy sets a record (log tail, so records pay without dominating a batch). The task card records `best_known` with its provenance and is updated when a run beats it (verified by the full tier, never self-reported).

**Abstention.** Tasks the forge cannot prove feasible, and tasks constructed infeasible (EnterpriseOps-Gym's impossible requests; a refund on an in-transit order), declare an `AbstainPolicy`. An abstention is judged only on the hard constraints (did the agent do damage on the way out?) and earns the policy's credit: 1 on an infeasible task, 0 on a feasible one, a small constant on an open problem, chosen below the learnability band's floor (p ≥ 0.1, §7.4) so giving up never beats a genuine attempt in expectation. A claimed solution to an infeasible task earns 0.

*Example (M7, task04 record mode):* re-scored from recorded measurements, GPT 6 Astra's v3 port beats our oracle on marching cubes and the people detector, while failing three KinFu bars; oracle v2 beats oracle v3 on three single-tracker metrics, because v3 bought concurrency with them. A bar reports all of this as 0 or 1; the score gate keeps the signal.

**Learnability for deep tasks.** 4p(1−p) is defined on a binary outcome; for score gates, M8 steers on the share of rollouts that beat the baseline (p) and separately tracks the record frontier (the best normalized score per round), which is the benchmark's non-saturating axis (§11.3).

---

## 8. Layer 4: verification and isolation

### 8.1 Patterns of hidden checks (coding)

| Pattern | What it guarantees | Lessons that shaped it (Appendix B) |
|---|---|---|
| **Differential execution against the original** | the new code behaves exactly like the old: bit-exact or tolerance-calibrated | references are computed at grading time in a root-only verifier image, never shipped to the agent |
| **Hidden tests at the expert end state** (F2P/P2P) | functional correctness | SWE-smith's validation contract, plus flaky-test repair |
| **Edge-case adversarial inputs** derived from defects | boundary semantics | guard bands, exact allocations, and offset ROIs exposed out-of-bounds and addressing bugs |
| **Sanitizers** (racecheck, synccheck, memcheck, ASan/TSan) | memory and concurrency safety | short workloads, because racecheck costs about 25 s per frame |
| **Concurrency equivalence** (concurrent = solo, bitwise) and stress | thread safety | compare against the *solo* reference; the original itself may not be concurrency-safe |
| **Fresh secret-seed draws** | no memorization of fixed cases | tolerance tiers calibrated on natural plus induced bifurcations |
| **Perf bars vs the original on the same machine** | real optimization, not just porting | best-of-k rounds, round-spread check, node-load guard, separate concurrency throughput bar |
| **Static and binary checks** | forbidden constructs absent; correct target architecture | fingerprint the verifier image before grading |

### 8.2 Calibrating tolerances

Any check with a tolerance must have the tolerance derived from data:

- **Natural spread:** reference vs oracle over ≥ 40 random draws.
- **Induced perturbations:** perturb parameters that leave semantics unchanged (e.g. a shift threshold ±1%) to expose bifurcating workloads.
- **Two tiers:** strict when trajectories agree, relaxed-but-bounded when they diverge. The relaxed tier must still kill every negative.

We record the oracle false-negative rate. It must be 0 over the calibration set, and its upper confidence bound goes on the task card (P15).

### 8.3 Isolation stack, with an automated audit per layer

1. The agent image holds no `tests/`, `solution/`, references, verifier code or fingerprints. The audit walks the filesystem for known hidden artifacts and canary strings (OpenEnv RFC 008 canaries).
2. Git history is a single commit per repo, with no reflog or packs containing future states. The audit checks `git log --all` and searches the object store for hidden blobs. SWE-smith's instance branches leak the bug and test commits; NeMo-Gym's `swebench/anti_cheat.py` scrubs only on a best-effort basis.
3. Network: `no-network` for agent and verifier. Fetches happen only at image build time.
4. The verifier runs in a separate image (Harbor `environment_mode="separate"`, verifiers `IsolatedVerifierEnv`). Candidate code runs as an unprivileged UID. Reference directories are 0700 root. Outputs are chown-sealed after every candidate step (task-03/04 grader).
5. The collection hook handles a dirty workspace and gitignored build directories, and errors from `git add` cannot abort the diff (Appendix B.5).
6. Perf grading runs only when the node is quiet (load guard) and records the spread across rounds; contaminated rounds are re-graded, not scored (Appendix B.5).
7. **Trusted channels only (P26).** Candidate code runs unprivileged in the verifier too, with privileges dropped *before* any candidate code (its `setup.py`, its import side effects) runs. The test harness, the benchmark and the reference build are root-owned copies made before the candidate runs. Verdicts come from artifacts compared by a trusted process, never from exit codes or the candidate's stdout. Timing is taken outside the candidate's process. Every timed repetition gets fresh inputs, and every candidate invocation a fresh HOME and TMPDIR (no memoization, no replay of cached results). Consumers come before the candidate on `PYTHONPATH`. Any grader exception writes reward 0 plus an `infra_error` record; `reward.txt` is never left empty.

**Built in M1 (2026-10-09), tested by `tests/redteam/test_attacks.py` on a Harbor-backed World with `isolation.policy_user` set:**

| Layer | Mechanism | Attack it defeats |
|---|---|---|
| Unprivileged policy | every policy tool runs through the in-sandbox guard as a demoted uid (setsid, `oom_score_adj=1000`, no groups, clean env); answer and harness paths are root 0700 | reading root-only answers, directly or through a symlink; writing grader paths; killing PID 1 |
| Verifier integrity | verifier snippets run as root with the interpreter resolved at setup and `python -I`; snippets and snapshot payloads are staged in a root-only directory, never `/tmp` | a fake `python3` on the policy's `PATH` or in its rc files; reading a gold snapshot while it is restored |
| Quiesce | before every verification snapshot, the policy uid's whole process cohort is killed (`kill(-1)` from a forked child at that uid, then a `/proc` sweep, repeated until clean) | a background process that keeps editing the code after submission |
| Collection | the file walker never follows symlinks (O_NOFOLLOW at every component), never opens FIFOs or devices, caps file size (sparse files), and re-checks with `fstat`; specials are fingerprinted and reported as anomalies even where git cannot see them | a FIFO that hangs the grader, a 20 GB sparse file, a symlink swapped in after the check |
| Resource caps | `pids_limit`, `mem_limit`; the policy runs at `oom_score_adj=1000`, written by root with `CAP_SYS_RESOURCE` so it is also the policy's floor | fork bombs, memory exhaustion of the grader, the policy opting out of being the OOM victim |
| Git history | refs, remotes, reflogs and unreachable objects are removed, then **verified**: `rev-list --all` ⊆ ancestry of the base, `fsck --unreachable` empty, forbidden SHAs absent | `git log --all`, `git show <future-fix>` |
| Network | `network_mode: none`, plus an egress self-test run as the policy uid that refuses to start the world if anything is reachable | `pip download` of the fixed version, following a PR link |
| Text | instructions, simulated colleagues' replies and CI output are leak-scrubbed (Repo2RLEnv patterns + ours); actors' citations are kept in the trace but hidden from the policy | PR links, fix commits, "fixed in 1.2.3" in a reply |
| Hidden CI | checks live in a separate runner sandbox, run unprivileged by default, re-staged every run, leftovers killed; the policy gets `name: PASS/FAIL` only, under a run budget that is world state | reading, overfitting or rewriting the hidden tests |
| Trusted git | verifier-side git reads only a root-owned copy of `.git` taken before the policy ran; refs are validated and quoted | git-ref shell injection; hooks, filters, fsmonitor, external diff or `info/exclude` planted to run code as root or hide files |
| Fresh account, no escalation | a new uid (never an image account), `PR_SET_NO_NEW_PRIVS` on the policy tree, root's `PATH` and interpreter audited for policy-writable components | the image's uid-1000 user with sudo, setuid-root binaries, a planted `git` on root's `PATH` |
| Episode hygiene | quiesce before restore; restore through directory fds, owned by the policy; `.git` reset; `/tmp`, `/dev/shm`, home, SysV IPC emptied; App-owned state root `0600` | a symlinked parent redirecting a root write; stashing state for the next episode; reading other people's mail from the database file |
| Egress self-test, answer sources | 17 code hosts, registries and mirrors (GitHub incl. `codeload`/`raw`, GitLab, PyPI incl. `files.pythonhosted.org`, npm, crates, Go proxy, conda, Hugging Face, `web.archive.org`, …) probed as the policy uid with a TLS/SNI handshake (a transparent egress proxy accepts every TCP connect); allowlisted hosts are excluded and checked as working lanes; audit `network.answer_sources` fails any allowlist that opens one of them (added 2026-10-09) | `pip download` of the fixed release through an allowlisted index; a mirror or archive after GitHub is blocked |
| Answer-key probes (P27) | every policy tool call is scanned for history, registry, code-host, grader-file, root and `/proc` probes; an exposure masks the episode (`crucible:answer_exposed`) | a pass reached by reading the published fix |

**Built in M7 (2026-10-09): the grader-side trusted channels (layer 7), `crucible.verify.runtime`, standard library only, staged read-only into any verifier image and run as root with `--network none`. Tested by `tests/redteam/test_grader_attacks.py`: the honest solution scores 1, 11 cheat mutants score 0 at their intended gate, and 6 of them are shown to beat a seed-style grader (exit-code verdicts, candidate-first `PYTHONPATH`, one shared HOME, no sealing):**

| Mechanism | Attack it defeats |
|---|---|
| Patch applied and every candidate command run as a fresh uid (new session, `PR_SET_NO_NEW_PRIVS`, no groups) before its first instruction | `setup.py`/CMake/import side effects as root; writing the benchmark, references or reward files |
| Fresh HOME/TMPDIR per invocation, removed afterwards; `/dev/shm` and SysV IPC of the uid cleared | memoizing or replaying results across timed repetitions (`memoize_in_home`) |
| Cohort killed **the moment the main process exits** (not after the pipe-drain grace), outputs sealed root-owned read-only before comparison | a detached child rewriting outputs after the run (`background_fixup`; found by the control: a 1 s grace was enough) |
| Python runs `-I` with grader paths first and the candidate **after the standard library** | shadowing the stdlib (`json.py`), the consumer or the test runner (`pytest/`) |
| Test verdicts from a grader-opened results fd: a finish record, the expected count (collected against the original), every test passed, no skips; exit code ignored | `os._exit(0)` under pytest, `pytest.skip` from library code, fake "4 passed" on stdout |
| Static scan of the pruned tree that gets built: macro expansion, token pasting, dynamic imports, native binaries under any suffix, runner references | banned constructs behind `#define`, `##`, `importlib.import_module("cty" + "pes")`, skipped directories |
| Hidden material under `/root` (0700) or root-only copies; references computed at grade time into 0700 dirs | delegating to the original, reading or rewriting hidden tests |
| Perf: load guard, round-spread check (contaminated → infra, re-graded), wall time from the grader's clock (`wall_ratio_max`, or K/5K differential runs), end-state sanity | timers forged in-process; scoring another user's GPU job as a perf failure |
| `reward.txt` pre-written 0; any grader exception → reward 0, masked `grader_exception`; candidate crashes while judged → fail, never masked | turning a 0 into a masked grade by crashing the grader |

---

## 9. The Acceptance Contract (quality gate for every task)

A task enters the pool only if every check passes. The `crucible accept <task>` command automates all of them and writes `calibration.json`.

| # | Check | Requirement |
|---|---|---|
| A1 | Oracle | reward = 1 on ≥ 3 independent full grades **in the declared environment** (the task's own images, network and privilege settings; never a native run or the agent's own pod); every perf bar has ≥ 15% headroom; ≥ 10 grades on random fresh seeds never crash the grader |
| A2 | Floors | the no-op/empty patch, the unmodified original, and the "obvious naive" solution all score 0 |
| A3 | Mutation score | ≥ 1 negative per hidden-check family (≥ 4 in total for H3+); each fails the intended gate; ≥ 90% of generic defect-class mutants are killed |
| A4 | Flakiness | oracle false-negative rate = 0 over ≥ 40 fresh-seed draws plus induced-perturbation draws; reference self-consistency (≥ 3 identical reference runs) |
| A5 | Leak audit | §8.3 layers 1–2 pass; no hidden string appears in the agent image |
| A6 | Isolation audit | §8.3 layers 3–6 pass; a tamper attempt (writing to grader paths) is blocked |
| A7 | Instruction ↔ test coverage | every graded property is stated in the instruction; nothing graded is unstated (Harbor `harbor check` rubric, plus manual sign-off for H4+) |
| A8 | Grounding | ≥ 80% of hidden checks have provenance (an expert signal or a reference-differential check); every trap cites the signal it came from |
| A9 | Reproducibility | pinned sources and image digests; two rebuilds give identical reference fingerprints |
| A10 | Calibration | ≥ 2 frontier runs recorded with their failure taxonomy; the pass rate is inside the target band for the task's split |
| A11 | Harness hygiene | a dry run with a dirty workspace, ignored build directories and a blank-usage response is classified correctly (infra vs policy) |
| A12 | Cost budget | the full grade fits the verifier timeout; the fast-grade tier (§10.2) correlates with the full grade (Spearman ≥ 0.8 on calibration rollouts) |
| A13 | Grader attack suite | every cheat mutant scores 0: disabling the tests through the candidate's build (`DOCTEST_CONFIG_DISABLE`, `-DNDEBUG`-style defines, skipped test subdirectories); `os._exit(0)` when a test runner is loaded; injected result or timing rows on stdout; rewriting the benchmark, references or reward files; background processes; memoizing repeated inputs; replaying results cached in HOME/TMPDIR; shadowing consumers on `PYTHONPATH`; delegating to the original binary; banned code hidden behind macros, aliases, `importlib`, renamed binaries or skipped directories. Seeded from the 2026-10-09 seed-task audit (Appendix B.10) |

**Defect registry.** Every defect found in any task, by any audit, model run or reviewer, becomes a new automated check that A1–A12 run on *all* tasks from then on. The registry starts with the 14 defects in Appendix B.

**As built (M9, 2026-10-09; `docs/modules/M09-accept.md`).** `crucible.accept.run.accept_task` plans the required checks by dependency and cost, reuses evidence-bound cached results for the same acceptance key, and derives `accepted` from the results on every read. N/A comes only from a closed vocabulary of task facts; a missing adapter capability fails; an unavailable need is pending; a crash is an error. Grades are stored by grading environment and solution content, so changing one negative re-grades only that negative. The defect registry (`accept/registry/defects/*.yaml`, 23 entries) admits a detector only with a power test (13 re-injection fixtures, all proven) or a named per-kit test (10). Its version is part of every key, and backfill quarantines tasks a new detector fails. On the toy tasks A13 found two real defects (B.15).

**Imported tasks are not trusted on the publisher's word (research m9-m10/03, 2026-10-09).** A survey of 32 Hugging Face `rl-environment` datasets found that "verified" means anything from "a model once solved it" to two gold passes with per-row logs, and that about half the Harbor task volume ships no oracle. Defects found by reading and partly reproducing them include:
- oracles that apply nothing (heredoc indentation; 450 of 500 tasks in one set);
- base images on an unresolvable registry;
- a reference solution, with its decryption password, reachable from the agent;
- "private" answer keys downloadable without auth;
- a visible reference implementation that aliasing passes;
- host credentials forwarded into an internet-enabled agent sandbox;
- empty ground truth that gives a do-nothing policy 13%;
- flaky rows kept on purpose;
- failures that appear only at production concurrency.

M9 therefore runs on imports exactly as on forged tasks, plus the import checks I0–I10, which map onto A1–A13:
- an oracle-is-a-no-op check;
- constant-answer and empty-action floors, reported as a dataset floor rate;
- a reference-alias probe;
- gold reruns at two concurrency levels;
- grading with the verifier network off;
- credentials in the agent environment;
- public answer keys (training only, never benchmark);
- stdout-marker spoofing;
- an `import_report.json` diffed against the publisher's own exclusion list where one exists.

The external test bed is the paired Terminal-Lego sets, whose re-validation labels serve as ground truth for M9 recall and precision; R2E-Gym-Subset-Verified; HF_ML_Tasksmith with its degenerate equivalence-tests sibling as a negative control; terminal-bench-2.1; and NeMo workplace_assistant.

---

## 10. Layer 5: runtime, adapters, training

### 10.1 Runtime and adapters
- **Canonical format:** a Harbor task directory plus `crucible/` (§4.3).
- **Execution:**
  - **agentenv-framework** gives sandboxes, A2A agents, `eval run --k`, artifact versioning, trajectories, and the Harbor-style verifier step `run_container_unit_tests_verifier` with `reward_path`.
  - We add a **GPU sandbox provider** plugin (`[sandbox.providers.gpu_docker]`) that adds `--runtime=nvidia`, because the local agentenv provider passes no GPU flags.
  - We add a **separate-verifier step** that stages hidden material only after the agent finishes, because agentenv's verifier runs in the agent's container.
- **Export adapters:**

  | Adapter | Emits | Trainer |
  |---|---|---|
  | `export verifiers` | v1 Taskset package with `TaskData.resources.gpu`, `IsolatedVerifierEnv`, `@vf.reward`/`@vf.metric` per reward channel | prime-rl |
  | `export nemo_gym` | resources server with `/verify` returning `reward`, `reward_components`, `mask_sample`, `failure_kind` | NeMo-RL, verl |
  | `export openenv` | Environment with per-step `Observation.reward` from `steps.jsonl` and a rubric tree (Sequential/Gate/WeightedSum) | TRL, SkyRL, torchforge |
  | `export harbor` | identity | any Harbor runner, with `[[steps]]` for staged tasks |

**As built (M10, 2026-10-09; `docs/modules/M10-runtime.md`).** `crucible.runtime` runs k samples per (task, policy, harness) with derived seeds (`SampleKey`), through a phase machine (seed → agent → collect → verification → cleanup), and records every attempt in a journal that a re-run resumes. Rules:
- **Every failure has a blame** (`infra`, `harness`, `task`, `policy`, `unknown`). Only `policy` failures are scored (reward 0). All others are masked: reward `None`, excluded from mean, pass@k and coverage, never read as 0. A masked sample is retried (never a policy failure) up to `RetryRule`'s limit.
- **Stop rules:** budget (SQLite reserve/settle ledger; a crashed reservation becomes *uncertain* spend), wall clock, masked share > 20%, unknown-blame share > 5%. Running out of credits stops the run as resumable, not as failed samples.
- **Boundaries normalize foreign errors** (Harbor exception names, MiMo categories, provider failures) into the taxonomy. A verifier `infra` dict is classified into `node_contention`, `stale_image`, `stale_build` or `verifier_error`.
- **Workspace harness:** the patch is collected with the tolerant hook. An empty patch from a changed tree is `crucible:empty_collection`; the workspace is preserved and the retry *re-collects only*, without re-running the agent.
- **Docker grading** (`verify/stage.py`) pins the verifier image digest (mismatch → masked `stale image`, before anything runs), names and labels the container, and turns a timeout into `docker rm -f` plus masked `verifier_timeout`. `runtime/gpu.py` preflights PID 1 (no init → `crucible:zombie_wait`) and the GPU arch.
- **Control channels** run with a sentinel; a signal death without it is `crucible:harness_self_kill`, and a lint rejects self-matching full-command-line `pkill`/`pgrep` patterns.
- **Every B.5 class has an injector** and a test showing it is masked, plus policy controls (agent broke the environment, step limit, context overflow, no change) showing they are scored.
- **Retries reuse what is still valid.** A grade masked for infrastructure is retried as a re-grade of the stored patch, and an empty collection as a re-collect of the preserved tree; neither re-runs the agent, which would replace the measured solution with a new sample. `Runner.regrade` re-grades exhausted samples later from their artifacts. `PerfGuard` runs one perf grade per host and only on a quiet host (no foreign GPU memory, low utilization and load); a host that never quiets yields `node_contention` before anything runs.
- **Trajectories join.** Every model call has a `call_id` and the ordered list of actions it produced. The ATIF trajectory (Harbor's models) carries `call_id`, `crucible_seq` (the trace event) and per-call `Metrics` with token ids and logprobs; a call that produced no action (a blank reply the policy nudged past) gets its own step. `steps.jsonl` rows get `atif_step_id` and `call_id` from the M5 steps' seqs. A proxy capture log can be cross-checked against the trajectory (`crucible:capture_incomplete` on mismatch).
- **Harbor task directories run as they are.** `interop/harbor/verify_dir.HarborDirTask` grades a patch with the task's own `[verifier].command` in a fresh container of the pinned task image, with Harbor's layout (`/tests`, `/logs/artifacts`, `/logs/verifier`), declared network, limits and GPU, on any Docker host (the lab over `DOCKER_HOST=ssh://`). It can pin the task to a git ref, so a historical run is graded by the grader it was graded with. `ImportedHarness` turns a recorded run's patch into an attempt, so historical runs and dataset submissions pass through the same journal, classification and re-grading as live ones.
- **Every attempt writes `rollout.json`** (`crucible.rollout/v1`, schema in `train/rollout.py`): per call, token ids, behaviour logprobs, policy version (stamped by `LLMClient.policy_version` at capture time), sampling, and the call's credit; per rollout, reward (None when masked), gates, truncation, failure and blame. In `rollout_mode="train"` a rollout without complete token records is masked, never guessed. Blank usage in tolerant mode books a worst-case estimate as uncertain spend instead of $0.

**As built, M11 part 1 (2026-10-10; research `docs/research/m11-m12/05`).**
- `interop/canonical.py`: canonical JSON, derived `call_id`s, one tool naming across runtimes.
- `interop/replay.py`: runtime-independent grades and dense channels.
  - Every runtime's transcript (World trace, Harbor ATIF, NeMo-Gym Responses items, OpenEnv `CallToolAction`s, agentenv gateway events) is normalized to canonical actions.
  - The actions are replayed in a fresh World and graded through the single reward producer.
  - Progress is measured online over the task's subgoals or its plan's checks.
- Exports:
  - **NeMo-Gym resources server** (`interop/nemo_gym/server.py`): grades the live World session from its state. Masked grades go out as `mask_sample` in both NeMo-RL locations. Responses validate against NeMo-Gym's own pydantic models in its Python 3.13 env.
  - **OpenEnv `Environment`** (`interop/openenv/export.py`): on OpenEnv's own base and MCP types. Per-step rewards Ψ-telescope to the final reward, with Ψ₋₁ = 0 because a plan's invariants can hold at the start. A masked grade yields reward None.
  - **verifiers taskset** (`interop/verifiers/taskset.py`): subclasses verifiers' `HarborTask`/`HarborTaskset`. A masked grade raises a typed `TaskError`, which prime-rl drops; components are recorded as `crucible/` metrics.
- **Framework environments:** `crucible/.envs/{openenv,verifiers,nemo}`. verifiers pins mcp 2, agentenv and OpenEnv mcp < 2, and NeMo-Gym Python ≥ 3.13.14. Each path runs in its own environment, and the gate compares the outputs.
- **agentenv plugin** (`interop/agentenv/plugin/`, registered by entry points):
  - a `gpu_docker` sandbox provider (`--runtime=nvidia --init`);
  - a `crucible_world` step: the World behind a real agentenv gateway, published as a `DeployedGatewayEnv`, so stock agentenv steps work unchanged;
  - a `crucible_verify` step: the single reward producer, unclamped; steps from replaying the gateway's own trajectory; a masked grade records skipped rows and raises;
  - a `crucible_ref` artifact type.
- **MiMo round trip** (`interop/mimo/{verify,export}.py`):
  - The import records a sha256 manifest of the source env.
  - Each env's own `verify()` grades a World run in a fresh grading root: the source's verifier files plus the run's `system/*.db` and `workspace/`.
  - A crash or an unavailable judge is masked (MiMo's own contract), never 0. Items map to `check/<id>` channels.
  - MiMo's verifier output is captured and scrubbed: it prints a judge-key prefix.
  - The export emits the original bytes and checks them against the manifest.
  - **Gate ✅:** one env imported, run, graded (masked without a judge; 0.5 with a cheap judge for a placeholder answer) and re-exported byte-identical.
- **Canonical episode key** `"<task id>|<episode>"`: derived `call_id`s agree across runtimes.
- **Gate (toy) ✅** (`tests/unit/test_interop_gate.py`): the oracle trajectory gives byte-identical `reward.json` and `steps.jsonl` through native, Harbor ATIF, agentenv (real gateway REST transport), NeMo-Gym and OpenEnv, with the same final state fingerprint. verifiers keeps the reward and raises a typed error on a mask.

**As built, M11 part 4: OpenEnv import (2026-10-10; research `docs/research/m11-m12/05` §3.5, §4).** The reverse of the
export, so any environment in the OpenEnv / FineEnvs catalogue (simulations, games, planners, MCP services) runs as a
Crucible World with snapshot/restore, the single reward producer, the dense channel, the M10 runtime and the M12 trainer.
- **App `openenv/env`** (`interop/openenv/import_env.py`, DEPENDs on OpenEnv's `Environment` and MCP types):
  - the environment comes from `factory` (`module:attr`) + `reset` kwargs; `episode_id` is pinned so observations stay
    deterministic;
  - MCP environments expose their tools (`ListToolsAction`, narrowed by `allow_tools`); other environments get one
    `step` tool whose schema is the Action class's JSON schema;
  - `CallToolObservation.error` maps to blame: invalid args / unknown tool → policy, execution error → tool,
    transport / timeout → infra. An exception inside `step` reaches the World's choke point and masks the episode.
- **State = a replay journal** (`json_doc`): reset kwargs, each call's arguments, result digest (+ capped text),
  reward, `done`. Restore = rebuild + reset + re-issue the journal with **every result digest verified**; a
  non-deterministic environment raises `SnapshotMismatch` instead of silently restoring a different state.
- **Environment-terminated episodes** (`ToolResult.ends_episode`, new): an observation with `done` marks the actor done
  in the World (logged `environment_terminal`); the OpenEnv export grades on that same step.
- **Reward** (`OpenEnvTask`): one score gate = the environment's own reward (last step or sum) normalized by
  `reward_range`. No terminal step = 0, scored (N4: "no submission" is a policy outcome, never a mask). Dense
  subgoals over the journal: `terminated`, `positive_reward`, plus environment-specific ones.
- **Catalogue** (`interop/openenv/catalog.py`): `portsim(split, index)` = PortSimEnv v1 (FineEnvs; data CC BY-SA 4.0),
  1,050 train / 50 eval berth-planning tasks graded against a CP-SAT optimum. Progress subgoals
  `feasible_plan_checked`, `submitted_feasible`.
- **Gate ✅** (`tests/unit/test_openenv_import.py`, in `.envs/openenv`):
  - rewards equal the native environment's (oracle 1.0; serial baseline ≤ 0.2; no submission 0, not masked);
  - mid-episode restore reproduces the environment's own counters (`checks_used`);
  - a generic non-MCP environment works in last and sum modes;
  - a noisy environment refuses to restore;
  - native = import = canonical replay = OpenEnv re-export, with step rewards telescoping to 1.0;
  - PortSim trains through the M12 runtime (scripted engine → exact-TITO tokens → rubric → `rollout/v1`).
- `runtime/record.py` imports Harbor only for type checking, so in-process rollouts run without Harbor installed.
- **World capabilities the catalogue needs** (M1; `docs/modules/M01-world.md` "Added for environment diversity"):
  - per-tool call budgets (`LimitsSpec.tool_quota`; counts in `WorldRef`, spec hash unchanged without quotas);
  - a `sim` clock driven by Apps (an imported environment's sim time via `clock_from`), so World events fire on
    simulation time;
  - per-App derived seeds (`reset.seed = "world"` for imports);
  - image tool outputs (`ToolResult.images` as store refs; MCP `ImageContent` mapped and covered by the replay
    digest; vision policies get them as a user turn);
  - World events over an imported environment (`OpenEnvTask(events=...)`);
  - the < 5 ms in-process step budget.

### 10.2 Two verifier tiers

| Tier | Contents | When it runs |
|---|---|---|
| **Fast** (training) | gates, cheap and medium milestones, a fixed subset of correctness checks, one perf round | every rollout |
| **Full** (evaluation and acceptance) | everything | acceptance, benchmarking, and a sampled audit of 5% of training rollouts to catch divergence between tiers |

Task 04's full grade takes 30–60 minutes of GPU time, so RL needs the fast tier. A12 requires the two tiers to agree.

### 10.3 Training integration
- **Policy serving:** an OpenAI-compatible endpoint (vLLM or SGLang) serving the current checkpoint. Tokens and logprobs are captured with the interception or capture proxy (verifiers `interception/`, OpenEnv RFC 006, NeMo-Gym `token_id_capture`). Rollouts missing token records are masked, never guessed.
- **Algorithm:** GRPO, n = 8. Dense credit enters through the turn-level advantage of §6.4. Dynamic sampling drops zero-variance groups. Infra-failed rollouts are masked. Tool outputs are masked from the loss (verl-fork `token_trace.py`).
- **SFT exports:**
  - **Lifted expert traces:** the expert's commits are turned into agent-format tool-call sequences (read, edit, build, test) that replay to the same states, which the replay test checks. They are marked synthetic, and observations are re-executed so they are real.
  - **Successful policy rollouts**, rejection-sampled by the full verifier (SWE-smith `train/traj_mgr`).
- **Curriculum:** the learnability-steered generator (§7.4) refreshes the training pool each round. Policy entropy and the share of zero-variance groups are tracked as collapse warnings (thesis, Stratego lessons).

**RL infrastructure, trainer side (designed 2026-10-09; evidence: 11 labs' 2025–2026 reports, `docs/research/2026-10-09-agentic-rl-infra-survey.md`).** The labs agree that the hard part is keeping trainer GPUs busy while trajectories are long and long-tailed, without letting off-policy drift, engine mismatch or reward hacking corrupt the gradient. Crucible's trainer bridge (M12) therefore fixes these contracts before any training run:

1. **The rollout record is the interface** (`crucible.rollout/v1`). It holds token ids and behaviour logprobs exactly as sampled (token-in-token-out) and the policy version of every generated span. Every correction below needs both; a rollout missing them is masked.
2. **Topology:** colocated and asynchronous with partial rollout to start (one shared GPU), environments always off the GPU pool. Disaggregated later.
3. **Staleness is bounded explicitly:** a version cap (drop when the oldest span is more than τ versions old), per-sample scheduling, optional windowed FIFO, and length-bias control (short rollouts finish first and would otherwise dominate early batches).
4. **Train–inference mismatch is measured and corrected:** `rollout_probs_diff` is always logged; the correction is selectable (TIS, two-sided masking, binary-KL masking, DIS, MIS-PO); an FP32 LM head is a switch; routing replay applies only to MoE policies with low routing confidence.
5. **The objective is a menu, not a bet:** GRPO with DAPO defaults, Dr.GRPO, GSPO, CISPO, and a critic mode. **The critic mode matters for enterprise workflows:** one real case gives one trajectory and one outcome, so group-relative baselines do not exist. That is the single-rollout setting where a value baseline is required; Φ and C3 branch values are extra baselines Crucible can supply.
6. **Credit enters at the right granularity:** M6 turn advantages attach to their model call's tokens, and tool outputs are masked from the loss. Truncation is not failure (mask or Φ-bootstrap). Environments the agent broke are failures. Optional group redistribution among passing solutions (MiMo GAR, mass-conserving).
7. **Long horizons:** context compaction is a recorded segment structure, and credit crosses segments. Harnesses vary across groups.
8. **Monitors that catch silent failure:** staleness, mismatch, clipped-token share, truncation and length drift, confirmed-hack rate, and a pass-rate-jump alarm (a sudden jump in training pass rate is the classic signature of a hack) that triggers a full-tier re-audit.

---

**As built, part 1 (M12, 2026-10-10; research `docs/research/m11-m12/`).** The trainer math and the control plane are
engine-independent and tested on CPU:

- `train/mismatch.py`: every correction as a pure torch function returning per-token objectives: ppo (clip-higher,
  dual clip), tis, icepop, kpop, dis, mimo4, mispo, hispo, cispo, gspo. Golden tests reproduce the article's numbers
  (KPop binary KL 4.05e−5 / 0.4534, IcePop gate, MIS-PO trajectory filter, HisPO dual clip, CISPO stop-gradient).
  Corrections that need π_θold refuse to run without it unless told where it comes from. The default is
  `dis(0.8, 3.0)`, valid under multi-version async with call-level versions.
- `train/objectives.py`: loss aggregation (token-mean global by default; seq-mean, prompt-mean, HisPO constant),
  group advantages (mean, RLOO, optional std; masked rollouts excluded), token-level GAE with skip-observation,
  cross-segment discounting (γλ)^{N_{>s}} with a shared terminal, truncation bootstrap, length-adaptive λ, DAPO
  overlong shaping.
- `train/batching.py`: global, mini and micro batch definitions; largest-first DP balancing (imbalance ≤ 1.1);
  first-fit-decreasing token-budget micro batches; exact global token-mean loss scaling.
- `train/pool.py` + `train/scheduler.py`: a data pool with tags kept separate from values, and a deterministic
  scheduler. Dispatch is per rollout. Groups are admitted whole, gated per source and globally, with deficit-based
  source mix. Staleness is handled by `wait` (default) or `drop`. Group backfill is pad, refill_member or drop_group.
  Dynamic sampling, buffer policies (oldest-first, greedy, windowed FIFO), off-policy share cap, early-short drop.
- `train/sim.py`: a discrete-event simulator of the three topologies (colocated sync, colocated partial rollout with
  abort and re-prefill, disaggregated streaming with call-boundary publishes). It drives the real scheduler; the
  checklist items are tests.
- `train/monitors.py`: staleness, spans, length-bias KS, pass rate with a jump alarm, zero-variance groups, mismatch
  and timing metrics.

**As built, part 2 (M12 runtime, 2026-10-10).** The native runtime executes the scheduler's decisions with real concurrency:

- `train/engine.py`: the engine contract.
  - Token ids in; token ids, behaviour logprobs and the serving version out.
  - `abort_all`, pause/resume, `load_weights`, capacity.
  - Sticky crc32 routing in `EnginePool`.
  - Weights are pinned at call start, so a publish never changes weights under a running call.
  - Version switching is per call (an aborted call is re-issued whole) or per token (AReaL-style resume, recorded as `Turn.version_spans`, validated by `spans_valid`).
  - `TinyPolicy` + `TorchEngine` are a real tiny torch LM; the engine serves bf16-rounded weights, so the train–inference gap is real.
- `train/envs.py`: the token-level env contract. An infra error masks the rollout. `CopyTask` is a multi-turn toy with a latency long tail and failure injection.
- `train/trainer.py`: the `TrainerBackend` contract and `TorchTrainer`.
  - Group advantages exclude masked rollouts; turn credit is optional.
  - Prompts are reconstructed from delta storage.
  - Micro batches are DP-balanced and token-budgeted, with exact global token-mean scaling.
  - π_θold is computed only when the correction needs it.
- `train/weights.py`: in-process and checkpoint publishers. The version is +1 per publish. Hooks for a post-update transform and an optimizer reset.
- `train/runtime.py`: one asyncio task per rollout. Environment steps never wait for weights. Topologies:
  - colocated sync;
  - colocated partial (abort + pause at S1, train, publish, resume);
  - disaggregated stream (trainer in a thread, engines keep generating).

**End-to-end result (CPU, `tests/unit/test_train_runtime.py`).** A tiny real policy learns the copy task under every topology:

| Topology | Reward (first 5 → last 5 steps) | Max staleness | Calls aborted |
|---|---|---|---|
| sync | 0.08 → 0.37 | 0 | 0 |
| partial, call-level | 0.09 → 0.61 | 2 | 906, re-issued |
| partial, token-level | 0.09 → 0.57 | 2 | 925, resumed with valid spans |
| disaggregated | 0.07 → 0.51 | 1 | 0 |

Infra failures are masked and never trained.

**Single GPU.** No surveyed framework overlaps generation and training on one GPU with sleep/wake colocation (research m11-m12/02–03). Colocated async only buys carry-over of unfinished rollouts across steps. Real overlap on one GPU means a memory split: the engine at a fixed memory fraction with the trainer resident beside it. That is `disaggregated_stream` semantics on one device. Placement is therefore a separate knob from topology.

**Two concurrency hazards found and fixed.**
- A `torch.no_grad()` block spanning an `await` leaked grad-off mode into the trainer's coroutine on the same thread. Grad mode is thread-local, so scopes must never cross awaits.
- `rollout_probs_diff` without π_θold also contains the policy drift since sampling. This is documented; the pure engine gap needs `old_logp`.

**As built, part 3 (M12 bridges, 2026-10-10; module doc `docs/modules/M12-train.md`).**
- `bridges/common.flatten` turns a rollout record into one causal training sequence per (actor, segment): response = completions interleaved with observation tokens; the mask is 1 only on sampled tokens; logprobs, per-token advantages and per-token versions are aligned.
- Adapters: verl `AgentLoopOutput`, SkyRL `GeneratorOutput` (outcome on the last sampled token, since GRPO sums token rewards), TRL `rollout_func` fields, rLLM `Step`s with precomputed per-token advantages.
- verl: the `crucible_delta` estimator, registered through verl's own `register_adv_est`, computes A_t = (R − group mean) + c_t, with turn credit encoded in the token rewards; masked rollouts are excluded.
- TRL: `make_rollout_func` runs World episodes concurrently over TRL's colocated vLLM.

**As built, part 4 (M12 distributed weight plane, 2026-10-10; `train/weightsync.py`; research `m11-m12/04` §A).**
The checkpoint-engine protocol as a PATTERN over any `torch.distributed` group (NCCL on GPUs, gloo on CPU):
- **Bucket plan:** gathered once from every owner's shapes. 256-byte-aligned offsets; one owner per bucket (the
  broadcast source); a tensor larger than the bucket is refused. The plan is portable (`to_json`/`from_json`, crc) for
  elastic joins.
- **Bucket size:** `all_reduce(MIN)` of free memory × 0.9; a third when the largest tensor fits (room for staging),
  else half; capped by `max_bucket`, never below the largest tensor.
- **Double-buffered pipeline:** bucket g is staged into half g mod 2 and broadcast while receivers reload bucket g−1 on
  a worker thread. After every bucket the previous reload's status is `all_reduce(MAX)`-ed, carrying the failing
  rank, so any failure aborts every rank with the same culprit named.
- **What Crucible adds to checkpoint-engine:**
  - a version + plan-crc header on every update (plan drift refused everywhere);
  - an **atomic commit**: receivers stage, then `commit(version, state)` only after all buckets succeeded everywhere,
    so a failed update leaves the old weights and version serving;
  - streaming while serving (only the commit swaps, at a call boundary);
  - subgroup updates (`ranks=[…]`) for a restarted engine catching up.
- **Runtime contract:** `DistributedPublisher` (trainer rank: version + 1; local engines load directly; the collective
  runs in a thread so rollouts keep going) and `EngineReceiver` (engine rank: receive, then `load_weights(state, version)`).
- **Tests** (`tests/unit/test_weightsync.py`, real processes over gloo):
  - plan layout, determinism and portability; the bucket-size rule;
  - two versions bitwise across fp32/bf16/fp16/int64/bool/0-d through ≥ 3 buckets;
  - a slow reload never corrupts the double buffer;
  - sharded owners with colocated receivers;
  - an injected reload failure aborts all ranks and nobody commits;
  - a subgroup update plus an elastic plan;
  - publisher → receiver serves exactly the trainer's v2 weights.
- **Open:** zero-copy hand-off of the received bytes to a separate vLLM process (CUDA IPC, checkpoint-engine's
  `VllmColocateWorkerExtension`) and RDMA P2P reads; receivers commit through their engine's `load_weights` today.

**As built, part 5 (M12 real data parallelism, 2026-10-10; `train/trainer.py`).** `DistributedTorchTrainer` turns the
DP-rank unit from a simulation into processes:
- **Setup:** replicas are broadcast identical at start. Every rank holds the same global batch (`broadcast_batch`:
  rank 0 sends it) and computes the same group advantages.
- **Per rank:** it trains only its token-balanced partition in token-budget micro batches, with π_θold computed for its
  own items only.
- **Exactness:** each micro batch's loss is its share of the **global** trained-token mean, so one flattened
  `all_reduce(SUM)` of gradients gives every rank the full-batch gradient and the identical optimizer step.
- **Refused aggregations:** seq-mean and prompt-mean do not split into summable shares, so they are refused.
- **Test** (2 gloo processes, `dis` and `ppo`, 10 micro batches): replicas bitwise identical; equal to the single-process
  full-batch update within 6e-7.
- **Placement fix (B.19):** micro batches and DP partitions count **forward** tokens (prompt + completion), while
  the loss normalizer stays trained tokens.

**As built, part 6 (M12 env workers and bubbles, 2026-10-10; checklist #8, #9).**
- **Env workers:** `RuntimeConfig.env_workers = N` runs environments on an `EnvWorkerPool` (N threads, each with its
  own event loop, least-loaded placement). Token-env `reset`/`step` go to a worker; a whole World episode runs on a
  worker, with its model calls bridged back to the engine pool on the main loop. A blocking environment therefore
  stalls only its worker.
- **Tests:** inline, a blocked environment froze every model call (0 completions during the block); on workers, other
  rollouts' calls keep completing; blocking environments run in parallel (< ½ the inline wall time); an issued env
  step finishes while the engine is paused.
- **Per-step bubbles:** `idle/rollout_s` (no engine call in flight), `idle/train_s`, `idle/env_wait_s` (waiting for a
  worker), `env/busy_s`, `engine/utilization`. Colocated engine idle ≥ train time; disaggregated utilization > colocated.
- **Checklist audit (2026-10-10):** the earlier claim that the 52-item checklist was covered by tests was overstated.
  The honest status table is in `docs/modules/M12-train.md` §9: 24 done, 23 partial, 5 missing (updated as items close). #34 then closed: a π_θold correction with no source refuses rollouts that span several versions. #40 closed: asynchronous reward. A `grading` pool status (untrained, not selectable). A late outcome is counted but stays pending; a hard timeout or a crashing grader masks the rollout; a late auxiliary credit channel falls back to outcome-only advantage with `credit_fallback`. #37 closed: top-p sampling records each token's nucleus (`Turn.candidates`), and the trainer renormalizes π_θ over it; a record sampled with top_p < 1 but without candidates is invalid for training; the vLLM HTTP engine refuses top_p < 1 until it can report candidates.

**Design finding (B.16).** A cap on rollouts *in flight* (Ling's S × B) does not bound staleness when finished rollouts
can wait in the buffer: they age too. In simulation with τ = 2 and `wait`, rollouts were trained at staleness 12. The
admission cap therefore counts every **untrained** rollout, in flight or buffered. With oldest-first selection a
rollout is trained within about S steps, and S = 1 is exactly synchronous.

### 10.4 Long horizons: million-token trajectories with dense, honest reward (built 2026-10-09)

The survey's first two findings are that trajectories got long (Kimi K3 RL at 1M context, SWE RL at 130 turns) and that rewards got sparse (one binary outcome after hundreds of steps). Crucible's environments must therefore produce, survive and credit such episodes natively, not only in the trainer. Evidence and per-framework mechanisms: `docs/research/long-horizon/01-frameworks.md`. Module doc: `docs/modules/long-horizon.md`.

1. **Long-horizon worlds.** `toy/helpdesk_backlog` is a support shift of N tickets.
   - **Long-range dependencies:** tickets that must wait hours for a customer reply; a policy amendment mid-shift; repeat claims on orders an earlier ticket refunded. Plus bulk context (customer threads, filler documents).
   - **Scale:** the `xl` preset (1,200 tickets) is a 7,600-step episode with a 1.1M-token trajectory.
   - Forging long-horizon variants is a forge knob (horizon), not a separate code path.
2. **Dense reward that cannot be farmed** (`verify/progress.py`). A task declares its grade as subgoals, i.e. the per-entity predicates the final grade is made of.
   - Φ(s) is the weighted share of subgoals that hold. The shaping rewards Φ(sₜ₊₁) − Φ(sₜ) telescope exactly to the final reward: Φ₀ = 0, γ = 1, potential-based (Ng et al. 1999). Undo-and-redo nets zero, and no proxy is rewarded.
   - Evaluated online and incrementally: after each step only the subgoals whose entity keys the step touched (arguments, fired events). A full re-check every K steps and at the end reports any **drift**, a missing key dependency, instead of hiding it.
   - Streamed to `progress.jsonl`. Per-call credit is the ΔΦ of the actions each call produced (`record.credit_from_progress`).
   - No framework surveyed implements telescoping shaping; this is Crucible's own.
3. **Context compaction inside the episode** (`actors/policy.ContextConfig`).
   - At 80% of the context budget **the policy writes the summary itself**, as a recorded model call (CompactionRL, verifiers, MiMo). The new context is system + the task verbatim + the summary + the last k complete turns; no fabricated turns.
   - Each compaction opens a segment, recorded in the rollout record. The summary call is trained with the episode's advantage. Group statistics stay on whole rollouts.
   - Cross-segment GAE for the critic mode is M12 work (no framework has it).
4. **Linear token storage** (`train/rollout.Turn`). Within a segment a call stores only the prompt tokens it adds. The first call of a segment stores its compacted prompt. A prompt that does not extend the previous one is a `prefix_break`, which masks the rollout for training. Full prompt ids per call would be quadratic (2×10⁸ ints for 1,000 calls at 200k context).
5. **Checkpoint and resume** (`runtime/checkpoint.py`).
   - Every K steps: the WorldRef, the trace length, each policy's `state_dict()` (its current segment's context, queued actions, calls, segments, token counter) and the progress state, written atomically.
   - A masked attempt's retry restores the World in the previous attempt's work dir, carries the trace and progress streams over to the checkpoint, and continues. It re-prefills one segment, never the trajectory.
6. **Streaming storage.** The World streams every trace event to `trace.jsonl` as it happens (`World.stream_trace`). The progress stream is append-only. The checkpoint is the only rewritten file.

**Gate (`crucible_pack_toy.longhorizon.gate`, `reports/longhorizon_xl.json`):**
- The real `LLMToolPolicy` (driven API-free by `ReplayLLM`) works the `xl` shift with a 128k-token context: 7,623 steps, 7,627 calls, 1.11M trajectory tokens, 4 policy-written compactions (largest context 103k).
- Crashed on attempt 1, it resumed from the step-3000 checkpoint.
- Reward 1.0; dense progress sums to 1.0 with no drift; per-call credit sums to 1.0. 66 s wall.

**Remaining:**
- observation offloading (large outputs to a file plus a pointer, full blob in the trace) and a memory app that Φ never credits;
- compaction mode varied per GRPO group;
- `calls.jsonl` streaming for model calls;
- per-segment ATIF export;
- staged checkpoint milestones and a fast-tier feedback loop (`continue_until_timeout`, declared to A13);
- a real-model run;
- cross-segment GAE (M12).

## 11. Layer 6: benchmarks

### 11.1 Step-level metrics

Each one is reported per task and aggregated per horizon level.

| Metric | Definition |
|---|---|
| **Outcome pass@k** | standard, on the full verifier |
| **Gate profile** | pass rate of each gate, i.e. *where* agents fail (e.g. Astra: correctness ✓, perf ✗; sol: NCV ✗) |
| **Progress AUC** | area under Φ(s_t) vs normalized step (or tokens, or wall time); rewards getting further *sooner* |
| **Stage reach** | the furthest stage whose exit milestones are all satisfied |
| **First irrecoverable step** | first t after which no continuation reaches success; estimated by branch rollouts (C3) on a sampled subset |
| **Regression count** | milestones lost after being gained |
| **Recovery rate** | on fault-injected starts (the H-span task starts at an expert state with a known injected defect), the share of episodes that repair it |
| **Cost** | tokens, tool calls, wall time, GPU-seconds per success |
| **Communication** | multi-agent worlds: messages and tokens exchanged between agents per success (P28: communication is the swarm's cost) |
| **Probe profile** | answer-key probes attempted / ran / exposed per run (P27); an exposed pass is never counted |

### 11.2 OOD design

| Axis | Train | OOD test |
|---|---|---|
| repo | repos A | unseen repos B |
| API / migration type | CUDA texture-reference removal + warp-sync | unseen families (e.g. cuSPARSE legacy API, thrust→CCCL, HIP port) |
| time | traces before T | traces after T, a sealed time-split as in the thesis |
| horizon | H1–H3 | H4–H5 |
| harness | one agent harness | other harnesses (thesis: train across harnesses) |

Contamination control: the benchmark split is sealed before any builder sees the test traces. Benchmark tasks are fingerprinted, and training data is scanned for them.

### 11.3 Dynamic benchmark
- Each benchmark version has a **fixed anchor set** (about 30%, never rotated, for comparability) and a **fresh set** (about 70%), regenerated by the swarm from held-out seed traces with new secret parameters.
- The difficulty of fresh items is placed on the anchor scale with an item-response-theory (IRT) model (2PL) fitted on frontier and baseline runs, so scores across versions stay comparable.
- **Temporal decontamination** (SWE-rebench): a benchmark item derived from a public trace (a PR, an issue, a CVE) is scored for a model only if the trace post-dates that model's training cutoff. M3 keeps every trace's merge and commit times for this; the filter itself belongs to M10/M14.
- A "best-operator series" (thesis): a human expert tries to break the trained agent on a fixed set of cases.

---

## 12. MVP plan: coding vertical, legacy GPU migration family

### 12.1 Scope

**Family F1:** legacy GPU code modernization and migration. Every task in it requires all of the following:
- semantic equivalence to the original;
- safety (sanitizers, concurrency);
- perf bars.

Seed assets:
- Tasks 01–04 (four calibrated tasks, two of which defeat GPT 6 Astra and GPT 5.6 sol).
- Their graders, oracles and negatives.
- Their MIGRATION_LOGs, which record expert-style corrections.

**MVP showcase.** Tasks 01–04 are pushed through every module, end to end:
1. Their construction transcripts are ingested as traces.
2. Signals are extracted and milestones mined.
3. Graders are re-expressed in the verifier kit and emit `steps.jsonl`.
4. The forge derives H1–H3 sub-span tasks from each H4 task.
5. Every task passes `crucible accept`.
6. Tasks are exported to Harbor, verifiers and verl.
7. A small sparse-vs-dense GRPO run and a step-level benchmark report close the loop.

Scaling to 20–50 forged tasks, and to new repos, follows the MVP.

### 12.2 Data, before we have a design partner

| Source | Role |
|---|---|
| **Public upstream histories** | "enterprise traces" with real expert signal. PR series with reviews and CI that migrated deprecated GPU APIs: CUDA 12 removals (texture references, legacy cuSPARSE, …), warp-sync fixes, thrust/CUB/CCCL upgrades, HIP ports, in GPU-heavy OSS repos. Each candidate repo is screened for a buildable environment and a replay test before use. |
| **Our own construction transcripts** | real expert-plus-agent traces with corrections (e.g. bifurcation calibration, NCV guard bands, ABI rebuild, collect-hook fix). They contain the step-level signals §5.2 describes. Used as traces for the credit-assignment pipeline, never as benchmark material. |
| **Frontier and policy rollouts** | Tasks 03/04 runs of GPT 5.6 sol and GPT 6 Astra, plus future runs: negative and partial trajectories for C3/C5 and for calibration. |

### 12.3 Horizontal extension (designed, not built in the MVP)

The same layers apply to horizontal workflows:
- **Connectors:** ERP, mail, ticketing.
- **Environment simulators:** MiMo-style tools over SQLite snapshots (`general/envs/*/manifest.json`), and EnterpriseOps-Gym-style multi-app MCP with SQL state verifiers routed per app (`benchmark/verifier.py`).
- **Milestones:** predicates over database state.
- **Role policies:** coupled by one shared outcome (thesis, Stratego lesson).
- **Horizon and difficulty:** the number of apps and departments, as in MiMo's tiers t1–t5. Infeasible tasks test refusal (EnterpriseOps-Gym).

### 12.4 Phases and gates

| Phase | Weeks | Deliverables | Gate |
|---|---|---|---|
| P0 Foundations | 1–2 | schemas (`trace`, `task` superset, `milestones`, `card`); `crucible accept` automating A1–A12; defect registry seeded with Appendix B; import tasks 01–04 | the 4 tasks pass `crucible accept` automatically, and the registry reproduces at least 3 historical incidents as caught failures |
| P1 Capture & model | 2–4 | `github`, `ci`, `vcs_history` and `agent_sessions` connectors; trace store; replay test; milestone mining + stage induction for F1; state tracker | S1 ≥ 90% replay on ≥ 30 traces; tracker's stage agrees with a manual stage label at Cohen's κ ≥ 0.7 |
| P2 Credit | 4–7 | C1, C2, C4 + validity gate; C3 offline; reward properties R1–R6 as tests; `steps.jsonl` emitted by graders; verl and verifiers advantage hooks | S2 (AUC ≥ 0.75 and > judge); R1–R6 pass on all tasks |
| P3 Forge | 5–9 | swarm roles; generators (hindsight, span, trap, transplant, compose, tighten); calibrator; curator | MVP: tasks 01–04 plus their derived H1–H3 span tasks, all passing A1–A12. After MVP: 20–50 F1 tasks over H1–H4 with 100% A1–A12, frontier pass@1 ≤ 20% on the bench split |
| P4 Train & bench | 8–12 | GRPO runs on an open coder model: sparse vs dense, equal compute; OOD benchmark v1 (anchors + fresh); report | S5, S6 and S7 |

Compute estimate: the fast-tier grade is GPU-minutes per rollout, so GRPO with n = 8 on about 50 tasks per step is the dominant cost. We size this after P2 measures the fast-tier grade time.

---

## 13. Risks and open decisions

| Risk | Mitigation |
|---|---|
| Dense reward is hacked (milestones met by stubs, test tampering) | prerequisite products; gates dominate; R5 tests; verifier tests restored from a pristine copy; top-reward audit (S7) |
| Shaping biases the objective | only the invariant form (Φ(terminal)=0) is used for advantages; partial credit is capped by κ and sits behind the gates |
| Replay fails for old commits: toolchains rot | pinned base images per era; quarantine traces that fail replay; prefer recent traces |
| Too few traces per family | the span generator multiplies tasks per trace; transplant adds repos; the decomposition curriculum |
| GPU grading is too slow for RL | fast tier; sampled C3; shorter workloads under sanitizers |
| Grader flakiness fails correct policies | P15; induced-perturbation calibration; A4 |
| A frontier builder leaks its priors into tasks (homogeneous traps) | traps must cite expert signals (A8); dedup by semantic trap signature; human audit of a random 10% |
| Benchmark contamination | sealed time-split; fingerprints; fresh-set rotation |
| A grader trusts a channel the candidate controls (exit codes, stdout rows, its own build, timing in its process) | P26; A13 grader attack suite run on every task; found in 4 of our 5 seed graders (Appendix B.10) |
| An answer source is reachable (a mirror, an allowlisted index, an archive) | allowlists only; audit `network.answer_sources`; egress self-test on 17 answer-source hosts; answer-key probes mask exposed episodes (P27) |

**Open decisions for the user:**
1. The name ("Crucible").
2. The first open model and trainer for P4. prime-rl and verifiers v1 have the cleanest advantage hook; verl matches the MiMo recipes.
3. Whether to target the agentenv-framework or Harbor as the primary runner in P0 (both are supported via the canonical format).
4. Which public upstream repos form the F1 trace corpus.
5. The policy for using our own construction transcripts.

---

## 14. Repository layout

Superseded by `ARCHITECTURE.md` §3 (project structure) and §4 (modules M0–M15), which also define the domain-pack layout that makes the core open-ended.

---

## Appendix A — What we take from each surveyed framework

*Industry evidence (2026-10-09):* a survey of 11 labs' 2025–2026 agentic-RL infrastructure reports (`docs/research/2026-10-09-agentic-rl-infra-survey.md`) finds environment and grader engineering to be the binding constraint (DeepSeek-V4.1: higher marginal return than algorithmic innovation; MiMo: grader compute is its own scaling axis). It adds eight requirements to the roadmap (survey notes §6), of which the most immediate are:
- truncated episodes are masked or Φ-bootstrapped, never scored as failures;
- environment crashes the agent caused are failures, never masked;
- group advantage redistribution among passing solutions (MiMo GAR, mass-conserving);
- soft-hack negatives;
- rollout records with token ids, behaviour logprobs and policy version per span.

All paths are under `../RL-Environments/06-frameworks-and-env-code/` unless they say otherwise.

| Need | Take | From |
|---|---|---|
| Task format | Harbor task dir, schema 1.4, `[[steps]]` + `min_reward`, multi-key `reward.json`, separate verifier, `[[verifier.collect]]`, content-hash versioning | `harbor-framework-harbor/src/harbor/models/task/config.py`, `src/harbor/publisher/packager.py` |
| Trajectory format | ATIF v1.8 (token ids, logprobs, subagents, `is_copied_context`), 40+ harness converters, `trajectory.json` prefix seeding | `harbor-framework-harbor/src/harbor/models/trajectories/`, `src/harbor/agents/installed/` |
| Composite rewards and judges | rewardkit (criteria, judges with a guard, aggregations); verifiers v1 `@vf.reward`/`@vf.metric`, rubric judge, agentic judge; MiMo item schema (critical/important/sanity, rule/llm, `pass_anchor`) | `harbor-framework-harbor/packages/rewardkit/`, `primeintellect-ai-verifiers/verifiers/v1/judges/`, `../01-lab-releases/xiaomi-mimo-v2.6-rl-oss/general/envs/*/verifier_meta.json` |
| Dense step rewards | **None exist.** Hooks: OpenEnv `TrajectoryRubric.compute_step_rewards()`, verifiers `MessageNode.advantages`/`loss_weights`, verl-fork `arvo_penalties.py` (turn-level advantage operator) | `meta-pytorch-openenv/src/openenv/core/rubrics/trajectory.py`, `primeintellect-ai-verifiers/verifiers/v1/graph.py`, `xiaomimimo-verl/verl/trainer/ppo/arvo_penalties.py` |
| Infra vs policy failure | `mask_sample`, `failure_kind`; −999 masking; judge-failure masking | `nvidia-nemo-gym/nemo_gym/base_resources_server.py`, `xiaomimimo-verl/recipes/general/agent_loop.py` |
| Token capture | interception server; capture proxy (RFC 006); `token_id_capture` with masking on gaps | `primeintellect-ai-verifiers/verifiers/v1/interception/`, `meta-pytorch-openenv/rfcs/006-*.md`, `nvidia-nemo-gym/nemo_gym/token_id_capture/` |
| Multi-turn masking | TokenTrace (response_mask), chat_delta | `xiaomimimo-verl/recipes/general/token_trace.py`, `chat_delta.py` |
| Task synthesis | SWE-smith pipeline (LM modify / rewrite / procedural / PR mirror / combine → F2P/P2P validation → flaky repair → leak-safe issue text); proposer/solver learnability 4p(1−p) | `swe-bench-swe-smith/swesmith/bug_gen/`, `swesmith/harness/{valid,grading,repair}.py`, `primeintellect-ai-verifiers/environments/proposer_solver/` |
| Trace → step reward template | PivotRL (pivot turns in expert traces, graded match). We use the *state*-based form (P12) | `nvidia-nemo-gym/resources_servers/swe_pivot/` |
| Task validation | `vf-validate` gold/no-op; `openenv validate` (oracle, floor gap, canary); `harbor check` and `harbor analyze` rubrics; NeMo-Gym `verifier_fixture`, reward-hack matrix | `primeintellect-ai-verifiers/verifiers/v1/cli/validate.py`, `meta-pytorch-openenv/rfcs/008-*.md`, `harbor-framework-harbor/src/harbor/cli/quality_checker/default-rubric.toml`, `nvidia-nemo-gym/nemo_gym/verifier_fixture.py` |
| Isolation | separate verifier env; default-deny egress; git-history scrub; second clean grading sandbox | Harbor `verifier_mode.py`, verifiers `envs/isolated_verifier/`, NeMo-Gym `resources_servers/swebench/anti_cheat.py`, `swemer_oml` |
| GPU sandboxes | Harbor capabilities (gpus, gpu_types); NeMo-Gym `SandboxResources.gpu` → `--gpus`; verifiers `TaskResources.gpu`; Modal | `harbor-framework-harbor/src/harbor/environments/capabilities.py`, `nvidia-nemo-gym/nemo_gym/sandbox/providers/docker/provider.py` |
| Horizontal workflows | multi-app MCP + per-app SQL state verifiers, distractor tools, infeasible tasks; tools over SQLite + hidden verifier uploaded after the rollout; tier = number of apps | `servicenow-enterpriseops-gym/benchmark/{verifier,executor}.py`; MiMo `general/envs/*/manifest.json` |
| Rollouts / eval / hosting | agentenv `eval run --k`, A2A agents, artifact versioning, snapshots, explorer | `../agentenv-framework/src/agent_env/` |
| Sandbox hardening | Karotte: demoted student user, process-cohort kill before collection, no-follow collection, egress self-test, fake-model attack tests | `preferencemodel-karotte/src/karotte/{subprocess,process_utils,untrusted_paths,confinement,demoted}.py` |
| Leak scrubbing | Repo2RLEnv: fix-pointer regexes for task text, git-history scrub, gold = 1 / no-op = 0 validation | `huggingface-repo2rlenv/src/repo2rlenv/pipelines/{cve_patches,pr_runtime,_env_guard}.py` |

**Known defects in surveyed frameworks that we must not inherit:**
- SWE-smith instance branches keep the bug and test commits in history (`swesmith/profiles/base.py` `get_container` → `git checkout`).
- NeMo-Gym's history scrub is best-effort (logs on failure).
- OpenEnv's kernrl has no sandboxing inside the container and does nothing about timing noise.
- Harbor's pass@k assumes binary rewards.
- Harbor 0.23 (per Karotte's comparison): reads a `reward.json` the agent can pre-write; leftover agent processes can tamper after submission; a fake `python3` or a symlinked answer subverts grading; a FIFO hangs it; no resource limits; the internet is reachable.
- agentenv (same comparison): a replaced `bash` subverts it; no resource limits; the internet is reachable.
- Repo2RLEnv: the git-history scrub is not verified (unreachable objects can survive), and egress uses a denylist.

---

## Appendix B — Lessons from tasks 01–04 (the seed of the defect registry)

**B.1 Calibration record.** Task 04:
- **v1** (correctness only): GPT 6 Astra and GPT 5.6 sol both scored 1.0.
- **v2/v3** (speedup and concurrency bars):
  - **Astra scored 0.** It was correct everywhere but failed three perf bars (0.710/0.65, 0.702/0.66, 0.428/0.42).
  - **Sol scored 0.** Its NCV mirror-border regression was introduced while optimizing.
- **The oracle scored 1.0.**

Lesson: correctness-only migrations are within frontier reach. Strictness on properties experts care about (perf, concurrency, edge semantics) is what separates them.

**B.2 Flaky checks.**
- A KinFuLS fresh draw bifurcated about 5% of the time and failed correct solutions. The fix was a two-tier compare calibrated on 120 runs: 80 natural plus 40 induced.
- The atomic-compaction Haar outputs were nondeterministic *in the original*, because the output buffer truncated. The fix was a capacity bound plus reference self-consistency.

Registry checks: A4, plus "reference self-consistency".

**B.3 Wrong reference.**
- Concurrent end-state was compared against the original's *concurrent* run, but the original isn't concurrency-safe. Fix: compare to solo runs.
- The guard-band offset was 256 B, which isn't texture-aligned, so the original's status differed. Fix: 1 KiB alignment plus a reference status check.

**B.4 Weak verifier.**
- The NCV out-of-bounds negative passed while padding was zero inside the allocation. The fix added guard bands with garbage values and an exact-allocation variant under memcheck.
- The bicubic clamp negative passed until offset ROIs were added.
- The shared-counter negative passed until lockstep stress tests were added.

Registry check: A3, mutation score.

**B.5 Harness and infra failures.**

| Failure | Fix |
|---|---|
| Empty patch: a gitignored `build/` made `git add` fail inside an `&&` chain | tolerant collect hook |
| Blank-usage API response aborted the run | ignore cost-tracking errors; classify as infra |
| Out of API credits mid-run | budget check; classify as infra |
| Zombie wait loops: `sleep infinity` as PID 1 | `--init` |
| Stale verifier image | fingerprint the image before grading |
| ABI mismatch after a header change | always rebuild consumers against the candidate |
| GPU and CPU contention from other users | load guard, round spread, re-grade |
| Self-matching `pkill -f` over ssh | `[x]yz` pattern |

**B.6 Leak.** A history leak is a known class (SWE-smith). Our images use a single-commit history, and the agent images were audited before every model run.

**B.7 Found while building M0–M2 (2026-10-09).** Each becomes a registry check in M9.

| Defect | Where it bites | Check |
|---|---|---|
| Seed `task.toml` used `authors = ["…"]`; Harbor requires objects | the task is not a valid Harbor task at all | validate every task with Harbor's own `TaskConfig` (done at import) |
| Harbor silently drops unknown keys; `verifier.environment.dockerfile` is one | task04's separate verifier would be built from the **agent** image, without the hidden reference | fail import on ignored keys that change semantics; give the verifier env a prebuilt `docker_image` |
| `no-network` set only per phase; `[environment]` baseline public | setup and build phases have network | report the effective network per phase |
| Local Docker has no GPU path in Harbor | GPU tasks rejected or silently CPU-only | probe the GPU inside the sandbox before every episode (`process` adapter `gpu` op) |
| agentenv time triggers need an open MCP session | REST-only drivers never fire time events | keepalive session + a test that a time trigger fires |
| Event actions under a role without tools are refused by the gateway | environment dynamics silently missing on one backend | backend-independence test: the same episode yields identical final fingerprints on every backend |
| Untracked, non-ignored files (`__pycache__`) count as state | spurious diffs / leaked artifacts in collected patches | per-pack derived-artifact excludes, shared by the collect hook and the state adapter |
| AWM sample data uses `datetime('now')` | initial state depends on wall-clock time | freeze "now" when forging tasks from generated environments |
| LLM paraphrase of grounded records drifts (question becomes claim, hedge dropped, location added) | simulated actors fabricate subtly while passing citation checks | quote mode by default; audit paraphrase with a different-family entailment judge |
| Reasoning models return empty completions under small token caps | judges and selectors silently "fail" (and episodes get masked) | caps ≥ 1k tokens for reasoning models; salvage verdicts from truncated JSON |

**B.8 Found while adding the database, website, browser and CI apps and the isolation guard (2026-10-09).**

| Defect | Where it bites | Check |
|---|---|---|
| `json_agg(... ORDER BY 1, 2)` orders by constants, not columns | Postgres fingerprints differ after an exact restore | restore → fingerprint round trip for every adapter |
| `NUMERIC 1200.50` parsed as float `1200.5` | money compares wrong; fingerprints drift | parse decimals as `Decimal` |
| Playwright MCP `--allowed-origins` is documented as "not a security boundary" | a browser App could reach the internet | network isolation of the world, not the browser flag |
| The guard resolved the interpreter's real path on the host, not in the sandbox | the guard fails to start (or pins the wrong binary) | resolve every sandbox path inside the sandbox |
| `git ls-files` never lists FIFOs, sockets or devices | a FIFO is invisible to the state adapter and can hang a grader later | lstat sweep for special files in git mode |
| Tool errors carried the verifier snippet's traceback (paths, internals) | infrastructure details leak to the policy | policy-facing errors are the last exception line |
| Verifier snippets and snapshot tarballs were staged in world-writable `/tmp` | the policy can read a gold snapshot while it is restored | root-only staging directory once a guard is installed |
| A backgrounded process survives its shell call only if the shell lingers briefly | a red-team test passed without the attack ever running | every attack test first asserts that the attack took effect |

**B.9 Found by surveying nine frameworks against M0–M2 (2026-10-09).** Each was reproduced by a test before it was fixed.

| Defect | Where it bites | Check |
|---|---|---|
| `git_diff(ref=...)` was pasted into a root shell | the policy runs commands as root | refs validated and quoted; red-team injection test |
| Root ran git in the policy-owned repo with `safe.directory=*` | a planted hook, filter or fsmonitor runs as root at the next snapshot | trusted `.git` copy; red-team git-trap test |
| The toolchain shell and the doc library ran as root | the policy gets a root shell or reads through symlinks | every policy tool through `policy_exec`; red-team checks the uid |
| Verifier reads followed symlinks and opened FIFOs | a verifier reads the gold answer through the policy's link, or hangs | no-follow, regular-file-only reads |
| Restores wrote through symlinked parents as root; restored files were root-owned | writes into `/etc`; the policy can no longer edit its own files after a reset | dirfd restore with `O_EXCL`, owner uid |
| The guard "created" uid 1000, which the image already used | the policy runs as the image's `ubuntu` account | uid allocation, image accounts never reused |
| A plain `cmd &` kept the call open until its timeout; a fork bomb outlived its call and starved `docker exec` | timeouts in the trace; the harness cannot quiesce | process-group kill at call end; `RLIMIT_NPROC` headroom |
| Tool modules read the wall clock | replay from a WorldRef is not deterministic; tool timestamps contradict the World's clock | virtual time and seed in tool modules; audit `episode_determinism` |
| A malformed argument crashed `World.step`; `to="bob@x"` became five recipients | episodes die or state is corrupted | schema validation at the choke point |
| Batched tool calls interleaved with messages; no-usage responses priced at $0; replay cache shared across samples; context overflow masked | invalid requests masked as outages; the budget never trips; zero-variance groups; long-horizon failures disappear | policy loop and LLM client fixes (M02) |
| agentenv role header and `__actor` were self-asserted; restored worlds re-fired triggers | any MCP client gets every tool or acts as anyone; duplicate events after restore | direct system calls, HMAC actor, absolute trigger instants, settled snapshots |

**B.10 Found by auditing our own seed graders and the M3 gate against the Preference Model, Repo2RLEnv and Prime Intellect articles (2026-10-09).** Full evidence and file:line references in `docs/reviews/2026-10-09-articles-crosscheck.md`. Crucible-side defects are fixed; seed-grader defects are open until the tasks are re-versioned and re-validated on the lab GPU.

| Defect | Where it bites | Check / status |
|---|---|---|
| task01/01v2 build the hidden suite with the candidate's CMake and judge doctest by exit code | a PUBLIC `DOCTEST_CONFIG_DISABLE` passes every test gate with 0 cases run | P26, A13; open |
| task01/01v2 parse probe and perf rows from candidate stdout (last row / minimum wins) | library code injects passing rows | P26, A13; open |
| task02 runs candidate code (`setup.py`, imports) as root in the verifier, then re-runs `/tests/hidden/bench_pcops.py` | the candidate rewrites the benchmark, slows the reference, writes the reward file, leaves processes | §8.3 (7), A13; open |
| task03 pytest gates pass on `rc == 0`; timing measured inside the candidate's process; original engine reachable at grade time | `os._exit(0)` under pytest passes; delegate to the original | P26, A13; open |
| task03 `fresh_cases.py:34` draws boxes smaller than 2 × cutoff on 5/28 ≈ 18% of seeds | the grader crashes whatever the submission (P15/P16) | A1 (≥ 10 fresh grades); open |
| tasks 01–03 leave `reward.txt` empty on any grader exception | an infra error reads as a missing result | §8.3 (7); open |
| task04 perf: repeated identical inputs, a HOME/TMPDIR that persists across the grading run, timers in a process that links candidate code | memoization or cache-and-replay fakes perf and the concurrency check | §8.3 (7), A13; open |
| task04 KinFu-LS falls back to BIFURCATED tolerances (3 cm, 6–14% TSDF) whenever the strict bound fails | "within floating-point noise" in the prompt is not what is graded (P20) | open |
| Unenforced or looser-than-stated instruction sentences in tasks 01–03 | the prompt lies (P20) | A7; open |
| task01/01v2 recorded runs were graded inside the agent's pod, which had internet | the calibration was not measured in the declared environment | A1 (declared environment); open |
| Seed-world gate (M1) checked snapshot/restore only, never gold = 1 / no-op = 0 through the World | an untrustworthy oracle goes unnoticed (Repo2RLEnv's first rule) | A1; open (needs lab GPU time) |
| The egress self-test made plain TCP connects to two hosts | a transparent egress proxy accepts every connect (false alarm); mirrors and wheel hosts were never probed | 17 answer-source hosts, TLS/SNI, allowlist lanes; fixed |
| An allowlist could open a public index or code host | `pip download` of the fixed release through the "narrow lane" | audit `network.answer_sources`; fixed |
| No record of answer-key probes in traces | a contaminated pass counts as a solve | P27, `agent_events.py`, `crucible:answer_exposed`; fixed (no exposure in any of the 11 frontier runs) |
| World traces recorded no `writes` | M4 cannot credit the agent (or peer) that made an edit | `writes` per edit call; fixed |
| Docker Hub and RunPod key formats were not redacted; human actor ids were not pseudonymized | real credentials and identities in stored traces | M3 redaction; fixed |

**B.11 Found while building M5/M6 (2026-10-09).** Each is fixed, or recorded as a design limit, and covered by a test or a gate report.

| Defect / finding | Where it bites | Check |
|---|---|---|
| The editor classifier read JSON-string arguments as empty, so a `view` counted as an edit | segmentation over-counts edits | `test_workflow.py` |
| A keyword family table matched substrings ("round" in "stereo_ground", "stream" in "PointStream") | rediscovery reported 6/6 families that were not there | construct-only matching plus a hand audit (`m5_rediscovery.json`) |
| Blocks in brace languages were cut at the anchor line; comment lines anchored blocks; token novelty ignored comments that grep counts | "located" milestones never fired on C/C++; the oracle's own Φ was 0.69 | R1–R6 on the real tasks (`m6_properties_coding.json`) |
| Agents edit through `rm`/`mv`/here-docs/`git checkout <file>`; harnesses write files themselves | replay reproduced 21/100 traces | shell-effect replay plus byproduct accounting (99/100 agent state) |
| `LLMToolPolicy` treated a cut-off reply as done and pinned seed 0 when sampling | long-horizon episodes end early; C3 branches near-identical | cut-off nudge, `seed=None` when temperature > 0 |
| Reasoning models return empty text under small token caps | the judge baseline parsed nothing | judge cap 2,500 tokens |
| DESIGN's shaping return-to-go is a state baseline | no per-turn credit for the action | delta mode in `credit/compose.py` (§6.4) |
| Φ is blind before the first state change | S2 at the midpoint fails on exploration-heavy harnesses | S2 revision proposed (§2.3) |

**B.12 Found while building M7 (2026-10-09).** Each is fixed (kit or registry check) or recorded for the seed re-versioning.

| Defect / finding | Where it bites | Check / status |
|---|---|---|
| `reward.json` with nested keys | Harbor raises on any non-numeric value (`verifier.py:104-121`): every Crucible task would fail to grade under Harbor | flat `reward.json` + `crucible_reward.json`; fixed |
| task04 sanitizer gate read the *first* `RACECHECK SUMMARY` line of a log the candidate also writes to | warnings-only hazards pass behind a forged first summary | every summary line must report 0, exit code 0; fixed in the kit plan |
| task04 consumers build through the candidate's own `PCLConfig.cmake` | the candidate can inject compile flags into the pristine consumers | open (grader-owned config or flag allowlist at re-versioning) |
| task03 installed the CUDA original where the candidate could read it, and put the candidate first on `PYTHONPATH` | delegation to the original; shadowing consumers | root-only original, isolated interpreter; fixed in the kit plan |
| task03 benchmark rows printed inside the candidate's process | forged ms/step rows | K/5K differential wall timing from the grader; fixed in the kit plan (needs GPU golden) |
| task03 `fresh_cases.py` draws unrunnable systems on ~18% of seeds | grader crash, reward lost | reference runs first, redraw before any candidate run; fixed in the kit plan |
| task04 `astra_v3` was graded while another user's job ran (KinFu 1.6 → 60 ms/frame inside one grade) and scored as a perf failure | a contaminated measurement became a verdict | spread check → infra; the CPU golden re-scores it as masked and reproduces its re-grade `astra_v3b` |
| task04 v3 oracle KinFu headroom 12% (0.571 vs bar 0.65) | A1 requires ≥ 15% headroom on every perf bar | `calibrate.perf_headroom`; open (bar or oracle at re-versioning) |
| `PYTHONPATH` entries precede the standard library | "consumers first" is not enough: a candidate `json.py` shadows the grader's own imports | `Trusted.python` bootstrap; red-team `shadow_stdlib_json` |
| The trusted runner reaped its child in `poll()` and then crashed in `wait4`; and it killed the cohort only after a 1 s drain grace | an attack "failed" because the grader crashed into a fail; a detached child rewrote outputs in the grace | `wait4(WNOHANG)` exit observation, cohort kill at exit; found by the red-team control |
| M5 `phi` compared the sql adapter's exact float encoding (`{"f": hex}`), not the value | every float-valued milestone or predicate was unmeasurable | `evaluate.decode`; toy golden |
| A runner liveness check used `pgrep -f` matching its own ssh command line | the GPU golden runner never started (B.5 repeated) | pidfile |

**B.13 Found while building M8 (2026-10-09).**

| Defect / finding | Where it bites | Check / status |
|---|---|---|
| A span's start held the expert's port of a sibling fork (KinFu ↔ KinFu-LS) | 67–70% of the span's solution visible in the start: asymmetry 1 gone | sibling forks excluded; A5 sibling-overlap ≤ 20% (`forge/derive.py`) |
| The Adversary had no negative for a gate family (T1's policy gate) | a gate nobody ever proved can fail | A3 per-gate-family coverage; generic double-refund mutant |
| A journaled role output was reused after the role's logic changed | stale negatives accepted | role `version` is part of the input hash; bump it with the logic |
| Frontier run traces held gate events no mapper read | the seeds' best trap sources were unusable | `coding/agent_run` signal mapper |
| The seed instruction's base-commit SHA matches the leak scrubber's pattern | every derived instruction failed A5 | derived instructions drop the base commit |
| A cheap LLM Test writer wrote parameterized SQL despite feedback | weak or broken checks | deterministic validators; the node fails closed and the plan is rejected |
| A rule revealed by a recorded correction (a reversed refund on a used add-on) was not enforced by any check | the recorded wrong answer failed only the outcome, not the policy | G-hindsight compiles revealed rules into invariants, with the correction's signal as provenance |
| A node skipped because its upstream failed was labelled "abandoned" | content failures reported as budget/infra exhaustion | `skipped` status; a plan with any failed or breached node is `rejected` |

**B.14 Found while building M10 (2026-10-09).** Each was found by an injection test or a golden run, and fixed.

| Defect / finding | Where it bites | Check / status |
|---|---|---|
| The in-process tool loader reported a `ZeroDivisionError` inside a tool as a domain `ToolError` (fault `tool`) | an environment bug became an observation the policy was "scored" on | Python bug exceptions (`ArithmeticError`, `TypeError`, `sqlite3.OperationalError`, …) → fault `infra`, masked as `tool_crashed`; injection test; fixed |
| task04 golden `astra_v3b` hit GPU out-of-memory while another user's job held the GPU, and scored it as a candidate failure | a contaminated run became a verdict (B.12 repeated, through memory instead of time) | `oom_under_contention`: OOM with > 2 GiB foreign GPU memory → `node_contention`, masked and re-graded; fixed |
| An ungraded result with no failure kind was recorded as a policy 0 | harness bugs that "forget" to classify are scored | unclassified → `unknown` blame, masked, and counted against the 5% stop rule; fixed |
| BSD `pkill` excludes the caller's ancestors; Linux `pkill -f` does not | a self-kill test passed vacuously on macOS | the self-kill test also runs over a Linux container channel (`tests/redteam/test_runtime_docker.py`); fixed |
| A crash injector with a `(*a, **k)` signature changed the tool's schema | the injection produced "invalid arguments", not a crash: the test measured the wrong class | injectors wrap with `functools.wraps`; every injection test asserts the intended failure kind; fixed |
| Harbor's verifier writes no reward file on timeout or grader crash | an infra failure reads as a missing result (B.10 row "empty reward.txt") | no reward file → masked `no_reward_json`; timeout → masked `verifier_timeout`; fixed in `docker_grade` |
| A retry of a masked grade re-ran the agent | the retry measured a different sample than the one the contaminated grade was measuring; the original solution's score was never known | grade-only retry from the stored patch; `Runner.regrade`; fixed |
| Model calls that produced no action (blank or cut-off replies) had no ATIF step; `StepRow.atif_step_id` was never set | per-turn credit could not be placed on the tokens that earned it; ATIF, World steps and calls could not be joined (research H8) | `call_id` + ordered actions per call, `no_action` steps, `align`; fixed |
| Blank-usage responses in tolerant mode were priced at $0 | the budget never sees those calls | worst-case estimate booked as uncertain spend; fixed |

**B.15 Found while building M9 (2026-10-09).** Both found by the A13 attack suite on the toy tasks; both fixed.

| Defect / finding | Where it bites | Check / status |
|---|---|---|
| A refund recorded with the supervisor's name but never requested scored 1.0 (state-only approval check) | the approver's name is hidden, but the world is constant across episodes: a policy learns it once and forges approvals from then on | `event` check `approval_grounded` (the approval must appear in the episode); A13 `forged_approval`; fixed |
| With no in-transit orders the supervisor's rejection rule became `(?:)` and rejected every request; the hindsight oracle refunded anyway | the oracle violated the policy it was meant to model, and state-only checks passed it | rule added only when the list is non-empty; caught by `approval_grounded`; fixed |

**B.16 Found while building M12 (2026-10-10).**

| Defect / finding | Where it bites | Check / status |
|---|---|---|
| An admission cap on in-flight rollouts only (S × B) lets finished rollouts accumulate in the buffer | with τ = 2 and `wait`, simulated rollouts trained at staleness 12, and synchronous mode leaked staleness 7 | the cap counts untrained rollouts (in flight + buffered); simulator tests for every topology; fixed |

**B.17 Found by the M11 interop survey (2026-10-10; `docs/research/m11-m12/05-interop-diversity.md`).** All fixed in M7's
reward writer.

| Defect / finding | Where it bites | Check / status |
|---|---|---|
| Two `reward.json` writers disagreed: the in-container grader wrote the plan reward, the host grader the composed scalar with κ(1−O)Φ | a failed task with Φ = 0.5 scored 0 in the container and 0.1 on the host; foreign runtimes read the container's file, so rewards differed by runtime | one producer `write_reward_files`; `reward.json["reward"]` = the plan's gated reward everywhere; the composed training scalar lives in `crucible_reward.json["training_reward"]`; fixed |
| A masked grade (and the pre-grading placeholder) wrote `"reward": 0.0` | Harbor, verifiers, NeMo-Gym, OpenEnv and TRL read only `reward`: every infrastructure failure would be trained as a real 0 | a masked grade writes **no** `reward.json` / `reward.txt` (a missing reward is an error in every runtime), plus `mask.json`; nothing is pre-written before grading; fixed |
| Dropping only the `reward` key would be worse | verifiers sums every remaining key: `{"infra/masked": 1}` would score 1.0 | no reward file at all when masked; test asserts it; fixed |
| `reward.txt` written with `:g` (6 significant digits); `time/*` seconds inside `reward.json` | files differ across runtimes; agentenv reads the text value | `repr()` (exact round trip); timing only in `report.json`; keys sorted; fixed |

**B.18 Found while importing OpenEnv environments (2026-10-10).**

| Defect / finding | Where it bites | Check / status |
|---|---|---|
| A World ended only on the policy's `Done`; Gym/OpenEnv environments end themselves (a final answer, a step budget) | after PortSim's `submit_plan` the policy could keep calling tools; the exported environment then raised "already declared done" on `crucible__finish` | `ToolResult.ends_episode` marks the actor done; the export grades on the terminating step; tests for both; fixed |
| An imported environment has no snapshot of its own | restore would silently produce a fresh environment, not the episode's state | replay-journal state with per-step digest verification; a non-deterministic environment raises `SnapshotMismatch`; fixed |

**B.19 Found while building real data parallelism (2026-10-10).**

| Defect / finding | Where it bites | Check / status |
|---|---|---|
| Micro batches and DP partitions were packed by **completion** length; the forward pass runs over prompt + completion | multi-turn agent calls (30k-token prompt, 300-token completion): the token budget undercounted memory ~100× (OOM on GPU), and DP balance ignored most of the compute | placement over forward tokens, loss normalizer over trained tokens; `test_micro_batch_budget_counts_forward_tokens_not_just_completions`; fixed |
| A non-global loss aggregation under DP would sum full per-rank losses | seq-mean / prompt-mean gradients `world`× too large | `DistributedTorchTrainer` refuses them; tested; fixed |

