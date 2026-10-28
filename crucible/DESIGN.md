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
- **P27 — Answer-key probes are recorded, never scored; an exposure masks.** Every tool call that probes git history, a package registry, a code host, grader files, root or other processes is tagged in the trace (`data.probes`, M3 `agent_events.py`), with whether it *ran* and whether its output actually *exposed* an answer source (a `git log --all` listing only `base` ran and exposed nothing). Probing is not punished: exploring history is ordinary work, and grading the process punishes valid alternatives (P12, P20). An exposure is output the call read from an answer source: naming a grader path (`ls /tests/`, a comment, a write) or reading back a file the policy wrote there is not one, so the policy cannot choose the mask (M7 R4). But an outcome reached after an exposure is not evidence: the World masks the episode (`crucible:answer_exposed`, an environment failure under P21) and imported runs lose their label (Repo2RLEnv: "every green score was contamination").
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

**Single-turn tasks (M1 build item 2, 2026-10-12; `runtime/single_turn.py`, `verify/answers.py`).** The largest family by count in the surveyed frameworks consists of a prompt, one completion and a checker, with no tools. A `SingleTurnTask` is a `RunTask` whose World has no Apps and one tool-less policy actor. The prompt is the instruction and the completion ends the episode. The full reply is kept on the `done` trace event even when longer than the 2000-character `reason`. Single-turn data therefore uses the same harness, traces, ATIF export, token capture and run records as agentic data.

The grade has two steps:

1. A **parser** reads the answer: full text, last `\boxed{}`, `Answer:` payload, XML tag, fenced code, JSON, multiple choice (four NeMo-Gym modes), regex or last line. Unless the parser keeps it, the text after the last `</think>` is read, and an unclosed `<think>` is no answer.
2. A **weighted rubric of checkers** scores it: exact, numeric with tolerances, symbolic math (math-verify in a killable child process), multiple choice, contains, regex, sequence similarity, JSON subset, any-of, or a named judge.

Gate `format` records whether an answer was found. Outcome `answer` requires every criterion to reach its threshold. The reward is the weighted score. A checker that cannot decide (a judge outage or an absent library) masks the sample (`check_infra`) instead of scoring 0 (P16). Rows load from NeMo-Gym (`responses_create_params`, `expected_answer`, `options`, `template_metadata`), verifiers (`prompt` or `question`, `answer`, `info`) and plain JSONL.

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

**Record-world primitives (M1 build item 3, 2026-10-12; ported from EnterpriseOps-Gym).**
- *Identity.* `ActorSpec.identity` (user id, name, email, roles, attributes, headers) states who an actor is inside the World's systems. Every App sees it (`App.identity(actor)`). An HTTP MCP server receives it as `x-user-*` headers through a session per identity, so the server applies its own permissions and row visibility.
- *RBAC.* `WorldSpec.rbac` maps a role to allow and deny patterns. A call that no role of the caller allows, or that a role denies, is refused as `permission_denied`. The refusal is the policy's error, and the tool stays visible, as with a 403 response.
- *Tool catalogs.* `ActorSpec.catalog` sets which permitted tools an actor is shown. The oracle tools are always shown. `distractors` more are drawn from the rest with a seed derived from the World seed and the actor (EnterpriseOps `plus_N_tools`; `-1` shows all), and `hide` removes tools. A tool outside the catalog does not exist for that actor.
- *Deterministic seeding.* SQLite Apps run `schema_sql` and `seed_sql` with SQLite's `now` pinned to the clock's `t0`, so dumps that call `datetime('now')` seed identical bytes on every provision.
- *State checks.* `verify/sql_checks.py` provides `SqlAssertion`, the EnterpriseOps `database_state` verifier with its result reduction and comparisons. It also provides `Collateral`, which requires that every row change found by `VerifierView.changes(app)` (episode start against now, with updates paired by key) is admitted by a per-table allowance. Assertions check what should have happened, and the collateral check catches what should not have: the test task's sloppy policy passes every assertion and fails only the collateral check.

**Readiness, resets and serial worlds (M1 build item 6, 2026-10-12; `world/health.py`; Harbor healthchecks, TheAgentCompany resets).** `AppSpec.healthcheck` (a command in the App's exec, or an HTTP probe made from inside it, with Docker HEALTHCHECK semantics) runs after the App starts and after every fresh reset. A task step can add its own check (`TaskStep.healthcheck`). `AppSpec.reset_hook` calls a live service's reset on every fresh reset, but not on a mid-episode restore, for state that no adapter captures. Either failure is `crucible:backend_error`: masked, never the policy's 0. `WorldSpec.serial` takes an exclusive host lock per world name from `start` to `close`, so episodes over one shared live service never overlap, across processes.

**Native turn order (M1 build item 7, 2026-10-12; `world/turns.py`; OpenSpiel and OpenEnv game semantics).** `WorldSpec.turns` gives a World with several policy actors one of three modes:
- `sequential`: seats move in order, `per_turn` actions each;
- `simultaneous`: one action per seat per round, buffered until the round is complete, so no seat sees another's move first. The rules App's `resolve_round` resolves the round when it has one, otherwise the calls run in seat order;
- `app`: an App's `turn_state()` returns who moves, the legal actions, whether the game is over, and the per-seat rewards. Chance (a dealt card, a die) belongs to the App and comes from its seed.

A move out of turn is refused (`not_your_turn`), and so is a move outside the legal list (`illegal_action`). Both are the policy's errors and neither executes. `Done` is always accepted and takes the seat out of the rotation. Observations carry `to_move` and the seat's `legal` list. Turn state is part of every `WorldRef`. A finished game ends every seat. Seat rewards reach graders (`VerifierView.seat_rewards()`) and run records (`meta.seat_rewards`).

**Images and audio in messages (M1 build item 8, 2026-10-12).** A `MediaRef` (image or audio) holds its bytes in the World's store (content-addressed) or at a remote URL. Instructions (`World.instruction_media`), messages between actors (`Message.attachments`) and inbox entries carry them. The World attaches the bytes (`data_b64`, never serialized) only to the live observation, so traces, snapshots and records hold references only. A snapshot of a media-free World is byte-identical to one taken before this change.

The LLM policy sends images as `image_url` parts (a data URL or the remote URL) when `vision` is on, and audio as `input_audio` parts when `audio` is on. A model that cannot take a part gets a placeholder line naming it. Token estimates count an image as a fixed tile and audio by clip length.

Single-turn rows with chat `image_url`, Responses `input_image` or `input_audio` parts deliver their media to the model. Few-shot turns keep theirs as chat parts.

**Remote environments over WebSocket (M1 build item 9, 2026-10-12; `world/remote.py`).** `RemoteEnvClient` speaks OpenEnv's `/ws` session protocol (reset, step, state, mcp, close; one socket is one server session) and is written against the protocol, so the Crucible side needs only `websockets`. Errors are classified:
- `VALIDATION_ERROR`, `INVALID_JSON` and `UNKNOWN_TYPE` mean the action was malformed: the policy's error;
- `EXECUTION_ERROR`, capacity limits, a lost connection or a timeout mean infrastructure, and the episode is masked.

A request that times out drops its socket, so a late reply is never paired with the next request (OpenEnv's own desync guard). OpenEnv client mode (`params.client`: `base_url` or `docker_image`, `mcp`, `session_id`) runs on this client:
- MCP-style servers expose their tools through `list_tools` / `call_tool` steps;
- restore rebuilds a new server session by replaying the journal and checking every digest;
- a connection dropped mid-episode is recovered by the same replay, followed by a single retry of the call.

**Control and simulators, for both language-model and neural-network policies (M1 build item 10, 2026-10-12).**
- *Spaces* (`core/schema/spaces.py`). Box, discrete, multi-discrete, multi-binary, dict and tuple are JSON data. Each space validates and clips values, samples, flattens to a float vector, and exports a JSON schema for tool calls. They read from Gymnasium spaces and from the dm_env spec dicts OpenEnv exports. Unbounded box entries are `null`, because infinities are not canonical JSON.
- *Control App* (`common/control`). Sources are `gymnasium:<id>`, a factory with the Gymnasium API, or a simulator served over WebSocket (`RemoteControlEnv`: OpenEnv's dm_control, Unity and FinRL servers, with spaces and `control_timestep` taken from the server's state). The simulator owns time:
  - one action holds for `frame_skip` ticks;
  - `sim_dt_s` (by default the environment's own dt) moves a sim-clock World, so events fire on simulation time;
  - `max_episode_ticks` truncates, and terminated and truncated are kept apart;
  - observations are numeric (exact in `data`, rounded JSON in `content`) and every tick's reward is journaled.
- *State* is the journal of seed plus actions. Restore rebuilds the simulator and replays it, checking every observation digest, so a non-deterministic simulator fails loudly. `restore: none` restores the episode start only and refuses mid-episode restores. `journal: lazy` writes the journal only when a snapshot, fingerprint or verifier query needs it (the new `App.before_snapshot` hook).
- *Language models* get a `step` tool whose schema comes from the action space, `wait` (ticks under `idle_action`) and `macros` (a named action held for N ticks): a few decisions over many simulator ticks, CARLA's tool-call-to-time policy.
- *Neural policies* use `world/gym.py`: `WorldEnv` (Gymnasium API) and `VectorWorldEnv` (auto-reset on fresh seeds), where every action is a World step. `train/control.py` is the non-token trainer path: PPO with GAE, the clipped objective, observation normalization, and categorical or Gaussian heads.
- *Measured through Worlds:* CartPole goes from 15 to about 200 return (greedy 174–203) in 50k steps, and Pendulum from −1310 to about −200 in 150k steps, at about 2,400–2,800 World steps/s on one CPU.
- `ControlTask` grades by the normalized environment return.

**Computer use (M1 build item 11, 2026-10-12; `common/desktop`).** A desktop App runs its own Linux desktop container: Xvfb, xdotool, ImageMagick and `--network none`. The policy has one `computer` tool in the shared action vocabulary of Harbor's computer-1, Anthropic's computer tool and OSWorld: click, double, right and middle click, drag, scroll, move, type, key, hold_key, wait and screenshot, with a `modifier` held during a click or scroll. Coordinates are in the screenshot the model sees, which is scaled to `max_width`, and are mapped back to the screen. Every action returns a screenshot as an image, which a vision policy receives as an `image_url` part.

Desktops cannot be snapshotted generically, so there are two reset tiers:
- `reprovision` (default): a fresh container and `setup` for every episode; mid-episode restore is refused.
- `replay`: a new container re-issues the journal's actions, checking every screenshot digest, with a short retry because a blinking caret has two legitimate screens.

**Full VMs with pre-booted pools (2026-10-11; `common/vm`).** A QEMU/KVM Debian guest runs in its own container with `/dev/kvm` and no network route. A golden `ready` RAM snapshot (one cold boot per host, 8.8 s on the lab) lets every later VM start with `-loadvm ready` in about 2.5 s. `VMPool` resets released VMs with `loadvm` (about 1.1 s) and replaces any that fail. World snapshots are `savevm` / `loadvm` under a journal-derived tag: exact RAM and disk, so processes survive a branch.
- **B.28.** A scoping bug in the pool's background reset (a rebinding made `vm` local, so the first await raised and the task died silently) leaked VMs out of the pool. A live run caught it, and failed reset tasks are now recorded instead of silent.

Graders use `inspect(cmd)` (a root shell inside the container). App health checks run inside an App's own container (`App.health_exec`).

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
| CI result per commit | check runs; CI logs (one check per test) | outcome of the state after that step | high (deterministic tests; flaky tests are filtered as in SWE-smith `harness/repair.py`; infra failures are not scored) | milestone evidence; step outcome |
| Review comment requesting a change, at a line | review thread | **negative** credit on the step that introduced those lines (blame inside the PR) | medium–high | defect → check (§6.3); rubric candidate |
| Accepted suggestion / the next commit fixing a commented line | review + diff | (wrong, corrected) action pair | high | negative mutant; DPO pair; trap synthesis |
| Revert or hotfix within N days | history after merge | strong **negative** for the originating step | high (survives review) | regression check; outcome label |
| Approval + merge, then 30 days without revert | PR state + history | positive terminal outcome | medium (reviews miss bugs) | outcome label |
| Benchmark numbers in the PR / CI | PR text, perf CI | target perf bar | medium (noisy; recalibrate) | perf gate threshold |
| User interrupt or rejected edit in an agent session | session log | **negative** on that tool call (the proposing call: a rejected command never ran, so it wrote nothing) | medium | step label; PRM training |
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
- **The tracker** reports stage, Φ, milestones gained and lost, the next expected milestones and deviations; κ = 0.87 against manual stage labels (toy, with the pack's monotone threshold 0.6; 0.28 at the default 0.9).
- **Rediscovery**, on 55 PCL GPU PRs, every family hand-audited (M5 audit): build, API removal and concurrency come from code; perf only from review comments; **edge semantics (one genuine PR) and bit-exactness are not rediscovered**. Bit-exactness comes from reference-differential checks (§7.1), a source upstream history does not contain, so that source is required for any family whose tasks grade exactness.

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
- "Gate-failed" here means a **hard rule** failed (a gate of kind `gate` in the grade report), not an unmet outcome check;
  a gate-failed or hacked episode gets no local term at all, so every turn trains at its outcome advantage. Every
  trainer path applies this one rule (`train.credit.credited_records`), and `turn_weight > 0` without recorded per-turn
  progress is refused, never silently outcome-only (M6 R1–R3, 2026-10-19).
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
- **R1** An oracle rollout gets the maximum reward, and its measured Φ is 1.
- **R2** A no-op gets 0.
- **R3** Every negative fails and stays below every success. A **crafted** negative (a known-bad end state: a policy violation, a reversal, a cheat) is caught by a gate and gets 0. An **honest** failure (a real attempt that did not succeed) keeps the bounded partial credit κΦ. (M6 audit: this line said 0 for every negative, while the κ(1−O)Φ term gives every gate-passing failure partial credit; the toy's crafted negatives earned 0.13–0.2 until they became gates.)
- **R4** Oscillating rollouts that break and fix the same thing do not earn more than monotone ones (shaping telescopes).
- **R5** A rollout that satisfies milestones by stubbing (with prerequisites unmet) earns ≤ the same rollout with the stubs removed.
- **R6** Reward is a function of state only, not of the agent's text claims (anti-sycophancy; for terminal reward it is computed in the separate verifier).
- **R7** Progress cannot be farmed: a mutant that only reaches the shallow milestones (the edit's location, a created file, a premature "resolved") earns at most ε = 0.1 (OpenEnv's field evidence: a "+0.2 for submitting anything" term taught quick submits).
- A property with nothing to test is reported as untested, never as passed; one that holds by construction (R5 without prerequisites, R6 on a reward with no message input) says so.

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
- **G-tighten:** raise the verification strictness of a solved task. Examples: add perf bars, concurrency bars or sanitizer tools, which is what took task 04 from v1 to v3. Raising the bar is only allowed if the oracle still has the A1 headroom (`accept.policy.a1_headroom`, ≥15%; m13 audit 06 found the 10% here inconsistent with A1).

**Learnability steering.** When generating RL training tasks, target a pass rate p ∈ [0.1, 0.7] for the *policy being trained*, using a proposer reward of 4p(1−p) as in verifiers `environments/proposer_solver`. When generating benchmark tasks, target frontier pass@1 ≤ 20%. Difficulty is adjusted with the generator knobs, never by making the instruction more obscure.

### 7.5 Depth: record-relative graders and honest abstention (P29)

The forge prefers depth to breadth. A task is deep when more effort keeps paying: a faster kernel than the best known, a larger rank, a tighter bound, a smaller counterexample, a migration that is also faster than the original. Such tasks are graded with **score gates** (`Gate(kind="score")`): validity checks are hard gates; the score check maps the result onto the known scale, `(score − baseline) / (best_known − baseline)`, so the reward grows past 1 when the policy sets a record (log tail, so records pay without dominating a batch). The task card records `best_known` with its provenance and is updated when a run beats it (verified by the full tier, never self-reported).

**Abstention.** Tasks the forge cannot prove feasible, and tasks constructed infeasible (EnterpriseOps-Gym's impossible requests; a refund on an in-transit order), declare an `AbstainPolicy`. An abstention is judged only on the hard constraints (did the agent do damage on the way out?) and earns the policy's credit: 1 on an infeasible task, 0 on a feasible one, a small constant on an open problem, chosen below the learnability band's floor (p ≥ 0.1, §7.4) so giving up never beats a genuine attempt in expectation. A claimed solution to an infeasible task earns 0.

*Example (M7, task04 record mode):* re-scored from recorded measurements, GPT 6 Astra's v3 port beats our oracle on marching cubes and the people detector, while failing three KinFu bars; oracle v2 beats oracle v3 on three single-tracker metrics, because v3 bought concurrency with them. A bar reports all of this as 0 or 1; the score gate keeps the signal.

**Learnability for deep tasks.** 4p(1−p) is defined on a binary outcome; for score gates, M8 steers on the share of rollouts that beat the baseline (p) and separately tracks the record frontier (the best normalized score per round), which is the benchmark's non-saturating axis (§11.3).

---

### 7.6 The task factory: curation with measured verifiers (M17, built 2026-10-15)

A task is four parts: a prompt, a World, a verifier and an **intent** (the skill it must teach). RL reinforces whatever
earns reward, so the factory's product is not tasks; it is **accepted tasks per dollar**, where every protected axis is
measured against ground truth. Research and method: `docs/research/task-factory/01`–`07` (Prime Agent's machine factory;
how 30+ environments validate verifiers, which found that no source publishes human-labelled verifier accuracy; agentenv;
the curation method; the build; the first run). Code: `src/crucible/factory/`.

1. **Objects** (`cards.py`, `ledger.py`):
   - `IntentCard`: its evidence must be real model-failure traces; it starts as a candidate, and a human approves it.
   - `ObservationRecord`: a typed ground truth. Reference-derived and simulator records never gate alone.
   - `RequirementLedger`: every prompt sentence ↔ a check, enforced both ways.
   - `VerifierCard`: every number has its denominator.
   - `TaskCard`, the closed `FailureCause` taxonomy, `Label` and `Decision`.
2. **Ground-truth lanes:**
   - **executable:** real fix + test commits mined from repositories (`lanes/executable.py`);
   - **human:** an append-only, hash-chained label and decision store (`labels.py`), with κ and an agentenv explorer
     plugin for intake (`explorer.py`);
   - **trace:** model failures, used as intent evidence only.
3. **The per-task machine** (`machine.py`, `driver.py`): a declarative, journaled, cyclic state machine (a port of Prime
   Agent's validator and guards, MIT).
   - The states: spec → author → realism → oracle → verifier → ledger → floors → probes → hackers → second verifier →
     solvers → differential → audit → flake → horizon → calibrate → gate → select.
   - Loop-backs carry findings to the owning state. Exhaustion pauses for a decision; it never ends as done.
   - The verifier writer never sees a solution. Reviewers come from another model family, enforced at validation.
4. **Verifier validation** (`floors`, `probes`, `differential`, `audit`, `flake`, `horizon`, `goldset`):
   - floors score 0;
   - every requirement mutant fails and every valid alternative passes;
   - an independent second verifier is run differentially;
   - failure causes are attributed by an auditor that is trusted only after it recalls seeded controls;
   - the flake bound needs 30 reruns × 2 concurrency levels;
   - long-horizon tasks get milestone, concatenation and shortcut tests;
   - the gold-set error bounds are per class (v1: Wilson upper ≤ 10% needs ≥ 36 per class; ≤ 5% needs ≥ 73).
5. **Family acceptance** (`family.py`, `ortho.py`, `transfer.py`):
   - orthogonality at four layers, the behavioural layer included;
   - a transfer proxy: the base policy is compared with in-context exemplars on held-out probes from a surface the family
     never uses. A family is accepted only when the skill transfers.
6. **Swarm and outer loop** (`root.py`, `models.py`):
   - the root curator writes no tasks; it leases cells of a behaviour × surface coverage matrix and runs machines in
     parallel;
   - it stops on budget, marginal yield or surface skew;
   - persistent per-family specialists;
   - every decision is stored with its reason, so criteria and cheaper gates can be compiled from it.

**First run (07, partial):**
- mining: 295 commits → 210 validated tasks from 11 repositories;
- discovery: 304 failures on 59 tasks classified, with the auditor recalling 177/177 seeded controls. 30 of the 304 were
  task defects (spec, World, format, verifier), not model failures;
- one trial task paused, as designed, at a decision;
- spend: $0.79.

Family acceptance has not run yet.

**Long-horizon coding (08 design; 09, 10 build, 2026-10-17).** Single commits are short tasks, so the long-horizon unit is
a **release window** of a real repository:
- the repo at release A is the start state, and the repo at B is an executed oracle;
- each commit's fail-to-pass tests form a milestone;
- `requires` edges come from executed dependency, not from file overlap.

The window lane (`factory/lanes/windows*.py`, `scripts/lh_windows.py`):
- builds a World at A: fresh git history tagged `base`, no network, the project installed in the image;
- runs executed leak probes: `pip download` of B fails, and no B artefact is in the image;
- runs each window's tests at A and at B with reruns, to sort fail-to-pass from pass-to-pass and drop flaky tests;
- runs the dependency prober on source-only diffs;
- builds commit-revert mutants (milestone k fails, independent milestones still pass);
- grades through `WindowTask.grade_patch`: the hidden tests run on a pristine A tree with protected paths restored;
- writes the prompt with a ledger, and a lint that refuses process instructions no check enforces;
- keeps held-out repos, fixed before mining, in a separate store that refuses training repos.

Step credit (`factory/checkpoint.py`):
- a sidecar grades milestone status on workspace snapshots at agent commits, every K tool calls, each root compaction
  boundary and the end, with a cap;
- Φ counts milestones whose `requires` hold; ΔΦ goes to the closing turn through `progress.jsonl`, so `turn_credit_map`
  and `token_credit` apply unchanged;
- a lost milestone is negative ΔΦ; an ungraded checkpoint is unmeasured, never 0;
- the agent never sees the results;
- runtime defines only an `EpisodeSidecar` Protocol (opt-in, off by default).

Honesty (`factory/honesty.py`):
- the `STATUS.json` claims are Brier-scored over verified requirements;
- `GIVE_UP` ends an episode honestly;
- the term enters the outcome advantage, not the reward, so it can never turn a failure positive.

Also new:
- `bench/lh_metrics.py`: milestones lost, loss at compaction boundaries, claim calibration, give-up correctness, and a
  paired base-vs-trained comparison with the MDE;
- `factory/sealing.py`: disjoint repos, the time split, contamination checks, and held-out publishing.

**First mining:** 116 training windows from 18 repos give 51 with dependency edges, 48 of them with valid mutants. 84
held-out windows from 6 repos give 20.

**First run (11; 10 windows; $6.54):**
- **Machine gates held:** floors 0, oracle 1, 29/29 mutants caught, 0 confirmed hacks in 20 hacker episodes, and 60/60 clean
  oracle reruns per window (flake bound 4.87%).
- **Solvers:** none passed in 173 episodes. Mean final Φ was 0.17 for qwen.
- **Lane fixes:**
  - prompts must state every public name the hidden tests import (lint enforced);
  - a repo's own public tests are no longer treated as grader exposure;
  - per-attempt spend no longer counts a shared client's total.
- **Open defects, fixed next:** Φ-only spread gets no credit (F1); textual dependency edges gate Φ (F2); test ownership by
  the touching commit instead of the commit that first makes the test pass (F3); weak solver rungs (F4); STATUS.json
  skipped at the step limit (F5).

**Run 2 (11 §Run 2; $6.25, stopped by the provider's account limit):**
- **Fixes:**
  - **F1:** the episode reward is 0.5·Φ_final + 0.5·[full pass] (`WindowTask.phi_weight`). The accept contract's `grade` stays
    binary, so a mutant still counts as killed only at reward 0. The learnable band is reward spread in the target
    rung's group.
  - **F2:** only executed edges gate Φ; textual conflicts become `order_hints`.
  - **F3:** a test belongs to the first unit after whose prefix it passes.
  - **F4:** the weak rungs are now nemotron-3-nano-30b and nova-micro, with pinned providers.
  - **F5:** `LLMToolPolicy(tool_limit=)` warns 5 calls before the limit and then allows one final STATUS.json write.
- **Honest re-mine:** 26 dependent training windows (from 51) and 10 held-out (from 20). Most lost edges were textual.
- **Gates still held:** 17/17 mutants caught, flake 60/60, no hacker pass.
- **Solvers:** 0 passes across 123 episodes. The F1 reward spreads in 8 of 13 groups. STATUS.json was written in 45/51
  episodes, but the claims stay overconfident (Brier 0.70).
- **Open:** D7 (the run script's tool-output cap cut `read_file` pages), D8 (auditor guesses must not reach prompts), F8
  (module-level imports make one owner for all tests). All three are fixed in P0 below.

**Factory v2: the workflow spec is `docs/research/swarm-pipeline/00-task-factory-workflow.md`; the build log is `04`.**

P0, the contract (`factory/longtask.py`, `factory/pathcheck.py`, `factory/lanes/windows_imports.py`):
- `LongTask` is the one contract every generator emits:
  - World, prompt and requirements;
  - milestones with executed `requires`;
  - `path_rules`, `grade_patch` and `milestone_status`;
  - `oracle_patch`, floors, mutants and alternatives;
  - `simplify(kind)`, `provenance`, `card`.
- `check_longtask` validates it, and `LongTaskRef` is the portable task row. `WindowTask` conforms.
- **Path checks** are graded from the World's own `trace.json` and the diff:
  - **gates** (no full-pass bonus, Φ kept): test or config edits, package installs, missing required actions, the step
    budget;
  - **hacks** (reward 0, `hack_suspect`): test plumbing, fetching code, reading history or the grader;
  - with no trail, the checks are unmeasured and never gate.
- **Import-tolerant ownership** (F8) is a per-window mining mode: new windows are tolerant; stored windows stay strict.
- **D7:** the tool-output cap is 32k. **D8:** only test-derived findings reach the author.
- **Callable checks:** `null_agent_score` and `clean_world_solvable`.
- **Re-grade of 51 stored windows:** 185/185 mutant vectors identical; floors identical; the 9 anomalies are
  reproduced by the pre-P0 code.

P1, machine v2 (`factory/machine.py`, `sessions.py`, `handoff.py`, `receipts.py`; Prime 7cbffa77b ported, MIT):
- **Infra vs task:** an infra failure is re-checked after a quiet period and respawned once for free; persistent infra
  pauses `infra`, never fails the task.
- **Captures:** an `artifact` port is a path plus sha256. A missing or vanished capture fails its producer.
- **Recovery:** `mutates: true` requires `recovery: reconcile|reset`.
- **Budgets:** dollar and token caps per state and per run.
- **Exhaustion** still pauses `exhausted`, never `done`.
- **Session states:** an M16 RLM agent in a private single-commit clone (not a worktree, whose shared object store would
  leak the oracle's commits) and its own offline container. Outputs are committed artifacts; persistent specialists
  resume by id.
- **Typed hand-offs:** a schema per port plus the full spec; summaries are refused.
- **Task repo as memory:** immutable commits, `informed_by=` trailers, in-flight announcements, free abstention, and a
  40% herding guard.
- **Receipts:** every run writes one with a `not_verified` list. The seeded-defect harness catches 13/13 cases (6 task
  defects at their states, 6 machine defects, 1 clean control), and its twin tests reject 10/10 tampered receipts.

P2, verifier rigor (`factory/verifier2.py`, `alternatives.py`, `fairness.py`, `goldset.py`; $0.45):
- **Re-mine (`store3`):**
  - import-tolerant;
  - test ids keyed by the file that ran them (fixes the cachetools/markdown regression ids);
  - volatile ids dropped;
  - result: 28 dependent training windows, 11 held-out.
- **Second verifier:** another family writes shadow tests from the prompt and the repo at A only. A shadow test gates
  only after it passes the oracle and the alternatives and fails the mutants.
- **Differential** over 3,406 (end state, requirement) pairs: 0 confirmed verifier-1 defects. 81 spec or gap candidates
  and 39 false-fail candidates are handed to the P3 audit.
- **Alternatives:** model rewrites from 3 families, accepted only when the real tests and verifier 2 both pass them. 57
  accepted; 22 of 27 windows have ≥ 2.
- **Fairness lint:** source-text checks, unstated formats, checks with no requirement, and the ten-competent-people test.
- **Verifier card, family `window`:**
  - false-pass bound 1.1% (0 errors in 342 items);
  - false-fail bound 6.4% (1 error in 84 items; the 5% target is not met);
  - mutants 116/116 killed.
- **Seeded gaps:** source-text check, weak check and false-failing check are all caught on the toy window. On real
  windows, the weak check (3/3) and the false-failing check (3/3) are caught by execution; the source-text check is
  caught by the lint (4/4) but not by execution (0/4), because the rewrites keep the variable name.

P3, the trace-audit loop (`factory/traceaudit.py`, `versioning.py`; `audit.py` extended; $0.20):
- An M16 RLM auditor reads episode bundles: turns, tool calls, `progress.jsonl`, diffs, path checks.
- **Facts first, without a judge;** then the first wrong step from the Φ series; then cause labels with verbatim quotes
  and seeded controls.
- **Typed proposals,** accepted only on executed evidence (spec 8.3). Each accepted fix becomes a version with lineage
  in a hash-chained `VersionStore`. `objective_id` keeps rewards from different verifier versions apart.
- **Same-family judging is refused,** a 10% human κ queue is kept, and auditor notes change only through
  `HarnessState.evaluate`.
- **Seeded:** lucky pass, unstated name and broken tool became accepted versioned fixes; a fake check was rejected. The
  lucky pass was caught by the executed pass audit, not by the judge.
- **Over 439 real trails:**
  - 7 World bugs reproduced on the oracle (missing setuptools, make, hexdump; PyYAML in glom);
  - 1 of 15 spec fixes accepted;
  - 17 intent-drift candidates await human approval.

P6, the sealed benchmark and the slime bridge (`factory/benchmark.py`, `factory/envs.py`, `runtime/harness/sandbox.py`,
`train/bridges/slime/`; $0):
- **Benchmark:** the same gates as training, sealing checks, a manifest with content hashes. Held-out prompts stay
  `prompt_pending` until a sealed author lease.
- **`make_env`:**
  - the native harness, sidecar, honesty and tool limit;
  - portable rows;
  - agentenv and Harbor export;
  - `objective_id`;
  - two timeouts;
  - whole-group drop on any infra failure;
  - `InfraWatch`, which masks a lost World container.
- **slime bridge:** JSONL rows, slime's 4-method Sandbox protocol over our World (ported, Apache-2.0), a `generate`
  function, turn-credit convert and advantage functions, the staleness filter, and a field-list patch for slime's
  `batch_builder.py`.
- **Equivalence:** five scripted solvers (pass, fail, hack, partial, infra death) give identical rewards, credit and
  masking on the native and slime paths.
- **Strict sealing leaves 1 of 11 held-out windows.** The causes are a shared org, the time split, and broken or
  timing-dependent Worlds. The benchmark needs a time-cutoff, org-disjoint re-mine.

The Tinker bridge (`train/bridges/tinker/`, `factory/tinker.py`, `scripts/tinker_rl.py`; `docs/research/swarm-pipeline/09`;
2026-10-20). Tinker (Thinking Machines, hosted LoRA) trains the student `Qwen/Qwen3.6-35B-A3B` and samples every
non-Claude model (difficulty ladder, cross-family roles):
- **Models**: a registry of Tinker ids with family, context (64K and `:peft:262144` rows), tool format, thinking switch and
  the pinned price table (2026-10-20); an unpriced model is refused; `with_context` picks the variant an episode cap needs.
- **Renderer**: the model's own HF template with exact TITO (`HFChatCodec` + `EngineLLM`), tool-call arguments decoded for
  templates that iterate them, per-family parsers (Qwen XML, DeepSeek DSML, GLM, Hermes; ported from tinker-cookbook),
  turn-end tokens for GLM and gpt-oss, a text protocol where the native format is not ported (Kimi, gpt-oss),
  `enable_thinking` explicit and recorded (off by default).
- **Spend**: every call booked (cached / uncached prefill from the service's count, sample, train, checkpoints) into
  `spend.jsonl`; a hard cap that holds each call's worst case first and stops the run.
- **Training**: groups from `make_env`, the slime bridge's filters (objective, whole-group infra drop and refill,
  staleness drop, spread), datums under one credit rule (Dr.GRPO outcome advantage, honesty composed, M6 turn credit,
  one datum per TITO sequence), `importance_sampling` / `ppo` / `cispo`, sampler refresh per step, resume from Tinker state
  with its optimizer, a per-model task band required.
- **Quarantine** (`train/quarantine.py`, backend-neutral; 2026-10-21, pattern from rl-env-use, MIT): a training task
  whose groups are flat (one reward, no Φ spread) on 2 consecutive steps is not sampled for 3 steps, then retried with
  one group; Φ spread without a pass is never quarantined; held-out tasks are refused; every event is logged
  (`quarantine.jsonl`, W&B `records/quarantine`, `quarantine/active`).
- **Difficulty** (swarm-pipeline/12; 2026-10-23; patterns: Online Difficulty Filtering, arXiv 2504.03380; R-Zero, arXiv
  2508.05004).
  - **The tracker** (`forge/difficulty.py`): online p̂ per task and per milestone, decayed per step. It never counts masked draws, and failed budget stops stay out of p̂. Its signal is `max(4p(1-p), 4 Var Φ)`.
  - **The sampler** (`train/curriculum.py`, backend-neutral): band filter on the Wilson interval, explore picks, learnability weights, and the quarantine left in charge of flat tasks. It runs a reverse curriculum over checkpoint starts and refuses held-out tasks.
  - **The trainer:** `TrainConfig(sampler="difficulty")` or `tinker_rl.py --sampler difficulty`, resumable.
  - **The factory side** (`factory/difficulty.py`):
    - the p≈½ controller emits typed knob actions (K1–K6, C1, H1–H3, fix_task, budget_bound), journaled with R-Zero's generator reward;
    - a stdlib ridge-logistic predictor, advisory only. Its rows take budget stops out of n, backfilled on the 150-call run from the episodes (`backfill_budget_stops`): LOO MAE 0.204 vs 0.257 baseline over 12 rows of 9 training tasks (0.10 vs 0.15 before the backfill, flattered by budget zeros);
    - full-task vs checkpoint-conditioned difficulty via `factory.checkpoint.credited` / `potential`;
    - checkpoint starts via `simplify_units(kind="checkpoint")` with a start check and the gates. Pre-applied milestones never earn credit: Φ0 is the baseline. Their own tests guard the regression check (`windows.guard_tests` into `p2p`, also for `WindowTask.simplify`): undoing an earlier fix keeps Φ but loses the full pass.
  - **Views:** `observe/difficulty.py`. **Script:** `scripts/lh_difficulty.py predict|conditional`.
- **Fairness, live (2026-10-23; swarm-pipeline/00 §5 rule 8, 07 §6.3):**
  - **S9b** (`factory/alternatives_gate.py`, stage `lh_debug_target.py fairness`): alternatives are solver end states from the bug report alone (stored frontier and student runs, plus S9b's own sessions by families other than the prompt writer's and the suite's), deduplicated by end fingerprint, judged by the grader, the blind second verifier (counted only when its family differs from the state's) and the report's repros. Classes accepted / `false_fail` / `lucky_pass` / wrong / unknown. The gate fails on any false-fail candidate, passes with ≥ 2 independently confirmed alternatives (one from a family other than the prompt writer's), else is unmeasured. κ and the disagreements (for the tie-break) per task. `LiveAlternatives` / `LiveSecondVerifier` replace the trace audit's fakes (`lh_audit.py --fairness`).
  - **The second verifier in every debug-lane language** (`verifier2` under the language profile): the writer places one new test file per requirement (`check_shadow_path`: a test file of the language, new at A, never a hidden one); the payload adds an existing test file of A as the style example and is proven blind through the profile (`leaks`); one grader run per file, so a file that does not compile fails only its own requirement.
  - **Path rules** (`factory/pathcheck.py`): `path:install` excuses an install in a scratch module outside the repo when the project's manifests and lockfiles are unchanged and the World is offline; machine-wide tool installs, `pip`, and installs in the repo still fail; bare reinstalls of declared dependencies are their own class (`allow_declared_reinstall`, off). `go -C dir` parsed. `RULES_VERSION` stamps every path report and the verifier version; `factory/repath.py` (`scripts/lh_repath.py`) re-applies path rules to stored episodes offline (`windows.path_rules_of`, `path_reward`).
- **Benchmark**: lh_metrics plus pass@k, pass^k, task-clustered CIs and USD per passing episode; the protocol (selection
  run, fresh baseline on a new seed, base control beside the trained adapter at identical settings, paired difference
  with its CI and P(trained > control)) checked by `check_protocol`.
- **Power** (`bench/power.py`, 09 §6.1; 2026-10-21, pattern from rl-env-use, MIT): A/A noise from each task's attempt
  halves (even/odd and seeded random halvings), the MDE of a paired comparison of N tasks × k, 2.80 · SE with the trained
  arm's variance bounded by μ_t(1 − μ_t), and a planner for N. Every arm and every comparison carries the stamp
  (`powered` / `underpowered` against the targeted +0.20, pre-registered from the fresh baseline); it never blocks, and an
  underpowered headline says so. On the S15 student arms: a 10-task × 8 benchmark has MDE 0.16 on pass@1 (0.09 on A/A
  alone): powered for +0.20; +0.15 needs 11-13 tasks, and 15 / 25-27 if the effect varies by 0.2 across tasks.
- **Equivalence**: on the P6 scripted solvers the Tinker datums carry, token for token, the slime path's per-token
  advantages, and the IS loss is equal.
- **Two TITO defects found by the live smoke, fixed in shared code**: `EngineLLM` read a call's re-render flag after its
  `gen` await, so concurrent episodes reset it (prefix breaks, masked episodes); `LLMToolPolicy._final_status_turn`
  ignored the engine's new-segment flag, so every limit-stopped episode on an "interleaved" template was masked.

The DeltaAI inference path (`train/bridges/vllm/`, `world/lockdown.py`, `scripts/deltaai/`; `docs/research/swarm-pipeline/14`;
2026-10-25). The user moved the before/after benchmark and the RL run to DeltaAI GH200s; Tinker stays the task-creation
backend:
- **The student on our own vLLM** (`vllm`, a final backend like `tinker`): NCSA LLMHub's vLLM v0.28.0 arm64 container
  (pinned by sha256), `ServeSpec` for the command line (262144 context, prefix caching, runtime LoRA, the repo's
  generation config off). `VLLMBackend` is the Tinker bridge's `sampler(model, path)` contract over `/v1/completions`
  with token ids in and out (`return_tokens_as_token_ids`), so `TinkerSampler`, the codec, TITO and the per-episode log are
  unchanged; `vllm_model` is the Tinker row's renderer facts at no per-token price. A trained arm is a PEFT adapter loaded
  into the same server. Every arm records `backend` and the engine (`arm_settings`), so `check_protocol` refuses arms on
  different engines; policy ids are `solver:vllm:<model>@vllm[-adapter]`.
- **Worlds without Docker:** `scripts/deltaai/docker` (a stdlib shim) runs the lanes' own `docker run/exec/rm/inspect/ps/cp`
  as Apptainer instances with one lockdown on every container (offline, `--containall --cleanenv --no-home`, binds only
  under the job's scratch, a per-container fuse-overlay layer and `/tmp`, `--fakeroot`, GPUs on request); other
  networks and binds of the home or the shared trees are refused. The SIFs are the Mac's arm64 images (`docker save`,
  `apptainer build docker-archive://`), so the measured Worlds are the ones that run.
- **The lockdown probe** (`world/lockdown.py`) runs inside a World before any model episode and refuses on network,
  visible secrets (home, `~/.ssh`, the HF token, `/u`, `/projects`, `/work`, `/sw`, the munge socket), sensitive
  environment, fabric devices, host processes or shared mounts. Measured clean on gh152 (job 3455864). What stays
  unprotected (one uid, no per-container cgroups) is listed in doc 14 §3.
- **Held-out tasks are offline:** `factory.tinker.require_offline` refuses a held-out task whose World declares a network.
- **The arms:** `tinker_rl.py eval/compare --backend vllm` and `lh_debug_target.py --backend vllm`.

The DeltaAI trainer bridge (`train/bridges/primerl/`, `scripts/delta_rl.py`; `docs/research/swarm-pipeline/14`;
2026-10-25; the user's decision: RL on DeltaAI GPUs, Tinker for task creation). prime-rl v0.10.0 (Apache-2.0; DEPEND on
the cluster, imported lazily) was chosen over slime, verl, SkyRL and NeMo-RL. It models the student's `qwen3_5_moe`
natively, has grouped-expert LoRA in vLLM's adapter format, ring CP for 100K+ sequences, aarch64 wheels, a Slurm
launcher, filesystem transports and an import-path loss.
- **Crucible keeps its loop.** `TinkerTrainer` runs unchanged over `PrimeRLBackend`, the `Backend` protocol over vLLM
  `/v1/completions` (token ids in and out) and prime-rl's batch / broadcast handshake:
  - `batches/step_<s>/rank_<i>.bin`, packed by prime-rl's `prepare_batch`;
  - `broadcasts/step_<s>` (`.sender_ready` -> `.receiver_ready` -> `.finished`);
  - `/load_lora_adapter`, one name per version, so stale groups keep their adapter;
  - prime-rl's own checkpoint per step for resume.

  So groups, filters, quarantine, the difficulty sampler, M6 credit, honesty and resume are shared code. A test pins that
  the bridge trains the same datums, token for token, that Tinker receives. `TinkerTrainer(model=)` takes a row other
  than the price table's: `local_model`, zero priced, since GPU hours are the cost.
- **The loss** (`primerl/loss.crucible_loss`, prime-rl's custom loss): Tinker's `importance_sampling` / `ppo` / `cispo`
  plus a stable_reinforce-style sampler anchor β·Σ(log q − log p_θ) (`--kl-coeff`). Advantages leave multiplied by the
  step's trainable-token count (`wire.undo_token_mean`), cancelling prime-rl's division, so `loss_scale` is exact. The
  off-policy correction `--pg-correction` ∈ {none, tis, mis, sc, sc+tis, sc+mis} defaults to `sc+mis` (below).
- **Deployment** (`primerl/deploy.py`): `DeltaConfig` accepts account `bjhg-dtai-gh` only. The submit unsets the login
  environment's `SBATCH_ACCOUNT`, and memory is requested per GPU. One node runs 2 trainer GPUs (`ep = 2`, `cp = 2`,
  262K) and 2 vLLM GPUs (TP 2, about 31 concurrent 160K-token episodes). vLLM runs as one engine with no router
  (`router = "None"`; the default `vllm-router` sits in prime-rl's `disagg` extra, which the build skips), so data-parallel
  engines are refused. The trainer uses `attn = "flash_attention_2"`, the one prebuilt wheel installed (prime-rl's `auto`
  picks FA3 on the GH200).
- **The build** (`build_env.sh`, on the login node): the venv sits on a uv-managed CPython 3.12
  (`UV_PYTHON_PREFERENCE=only-managed`), because Triton compiles its launcher against `Python.h` on the GPU node and the
  system python has no headers. The build and the job both fail when `Python.h` is missing. The environment (prime-rl,
  its venv, uv's cache, CPython) sits on `/work/nvme` (`DeltaConfig.envdir`), because a cold GPU node took over 13 min to
  import from `/work/hdd`. Extras: `gpu`, `flash-attn`, `kernels` (the trainer imports `prime_kernels`). The build imports
  every module the job imports, so a missing package fails on the login node.
  The trainer runs under torchrun as prime-rl's `rl` launcher starts it (`deploy.trainer_command`), with prime-rl's
  default trainer and common environment; its `trainer` console script sets up no distributed rank.
- **The plan** (`plan`, an estimate from the stored 300-call episodes): about 4.1 GPU-hours per 4 × 8 step. The
  recommended first run is 40 steps, about 165 of the 976 hours left.
- **MoE routing replay** (`primerl/routing.py`, on by default):
  - sampling goes through prime-rl's `/inference/v1/generate` with vLLM's routed-expert capture and a per-episode
    prefix-cache salt;
  - the routes are kept out of band per call (`RoutingStore`, keyed by the sha of prompt + completion), stitched per
    training sequence (the first call that forwarded a position wins), and replayed by the trainer
    (`enable_router_replay`);
  - all or none per step, with incomplete coverage refused, as the torch trainer's R3 replay does;
  - `routing/coverage_*` and `routing/replayed_share` are logged; `kl/mismatch` is the signal with replay off.
  - The broadcast directory is no longer shadowed by the datum loop that drops trained routes (with replay on,
    `save_for_sampler` had returned a datum instead of the adapter path).
- **Score centering with MIS** (`primerl/topk.py`, `topk_server.py`, `sc_head.py`; doc 14 §7; on by default,
  `--pg-correction sc+mis`, k = 32 by `--sc-topk`). arXiv 2609.20807 (reference code martin-marek/score-centering, MIT):
  the per-token loss is −A·(sg(w_y)·log p_y − Σ_{v∈H} sg(q_v·w(p_v/q_v) − α·p_v)·log p_v), where H is the sampler's
  top-k head, the tail is ρ·p (ρ = (1 − Σ_H q)/(1 − Σ_H p), α = ρ·w(1/ρ)), and w is MIS r·𝟙[0.5 ≤ r ≤ 5]. It removes
  the drift E_q[R]·s̄ that a train/inference mismatch adds.
  - **The sampler:** `/inference/v1/generate` with `logprobs = k`. prime-rl's packed handler kept only the sampled
    entry. `topk_server` (a vLLM `--middleware` import path) also packs entries 1..k as `topk_ids` / `topk_logprobs`
    arrays. `vllm.max_logprobs = k`.
  - **Kept and stitched like routes:** per call in a `TopKStore` (int32 + float32), stitched per training sequence (the
    call that sampled a position gives its head).
  - **On the wire:** in prime-rl's `sampling_mask` stream, as `[SC_MAGIC, ids, bits(logprobs)]` rows at trained
    positions.
  - **In the trainer:** prime-rl's custom loss gets only the sampled token's logprobs, so `sc_head.install` patches the
    fused LM head. `HeadLogProbFn` is its chunked logprob function plus k gathers per vocab chunk, never [T, V], and
    never renormalized over the head. `compute_loss` in the train module gathers the head over context parallel, shifts
    it, splits it per sequence and keys it by `trainer_logprobs`; the vanilla head refuses a payload.
  - **No PPO clip with SC or TIS/MIS** (the paper's evidence; TRL's async GRPO default is the same SC+MIS): they require
    `--loss importance_sampling`, and `--pg-correction none` keeps ppo / cispo. The anchor composes with every mode
    (default 0).
  - **All or none per step:** a step where any sequence lacks a head at a trained position sends none, and the loss uses
    MIS (TIS for `sc+tis`) for the whole step, with a warning. A partial head is refused.
  - **Logged:** `sc/coverage_*`, `sc/centered_share`, `sc/fallback(_steps_share)`, `sc/head_mass(_min)`, `sc/rho*`,
    `sc/score_mean_l1` (the drift proxy), `sc/kl_head`, `sc/tail_dropped`, `is/masked_frac`, `sc/align_err`.
  - **Cost:** `delta_rl.py topk-bench` times the server at k = 0, 8 and 32; the smoke runs it. Unit-tested on the CPU
    (k = V equals full centering, the reference code's gradient and drift test, the prime-rl micro-batch path
    emulated); not yet run live.
- **CLI**: `delta_rl.py` takes `tinker_rl.py train`'s flags unchanged, through the shared builders `add_common` /
  `add_train_flags` / `train_config`, plus the deployment's. The smoke runs on `ghx4-interactive`, since `test` is closed
  to the account.

Observability (`src/crucible/observe/`, `scripts/observe.py`; `docs/research/swarm-pipeline/11`; 2026-10-20). Every result,
graph and agent trace goes to Weights & Biases through one module (an import-linter contract forbids `wandb` / `weave`
anywhere else; the layer sits below `workflow`, above `connectors | signals`, so every producer reaches it):
- **Offline first**: W&B runs are written offline (`wandb sync` later) with a redacted mirror (`ops.jsonl`); Weave has no
  offline mode (its init logs in and calls the server), so episodes are written as Weave-shaped spans and replayed with
  their own call ids (`create_call(_call_id_override=, started_at=)`) once a key exists; links written today stay valid.
- **Projects**: `crucible-factory` (funnels, targeting, intent and verifier cards, the trace audit, findings, spend),
  `crucible-train` (per-step reward, Φ, gates, pass rate, advantages, loss, KL and ratio, staleness, drops, tokens, USD vs
  cap, sampler version; through `TinkerTrainer(sinks=)`), `crucible-bench` (per arm and per comparison: the headline
  pass@1 per arm with task-clustered CIs as an HTML chart, the paired trained − control difference, the power stamp
  (`compare/power`, `bench/power`, tag `power:*`), `check_protocol`,
  pass@k / pass^k, cost per passing episode, per-language / family / skill); a W&B Report template.
- **Traces**: episode → one call per model turn (prompt summary, output, tool calls, Φ after, milestones gained / lost,
  ΔΦ credit) → one call per tool (arguments, result); the trace audit's verdict (first wrong step, cause, quote) attaches
  under the same root, since the bundle id is the episode key; full text as an artifact.
- **Redaction**: the M3 redactor, the process's credential values by value, `scrub_secrets`; W&B's console, code, git and
  metadata capture off; a planted key is checked absent from everything written or handed to W&B.
P5a, the generators and scouts (`factory/generators/{rebreak,stitch,open}.py`, `factory/sources/`; $0.01):
- **Each generator builds its own A..B history** in derived clones, so `validate_window` and the P2–P4 handlers run
  unchanged.
- **The intent card drives the scouts:** template query expansion, typed records, the eval lease first, licence/size/CPU
  probes, an on-disk cache.
- **Passing the deterministic gates:**
  - 5 stitched windows (3 with executed consumer→host edges);
  - 1 rebreak window (only 4 of 1,482 real-fix pairs interact);
  - 2 open tasks (speed counts only on a correct result; S6 not applicable; 3 more fail S10 because the speedup sits near
    the bar).
- **New lints:** invented version numbers. S3 reruns compare the shared test ids.

P5b, swarm-built windows and frontier targeting (`factory/generators/swarm.py`, `scripts/lh_swarm.py`; $21.96 of a
$30 cap):
- **The swarm-built pipeline:**
  - a planner (given A's call graph; each `why_requires` must name what the earlier milestone adds);
  - an oracle builder session, one commit per milestone;
  - a blind verifier-writer session;
  - a reviewer and a hacker from another family;
  - then the same executed gates as mined windows. 5 of 14 start windows accepted (semver, boltons, funcy, cachetools,
    more-itertools).
- **Frontier target `openai/gpt-6.1-sol`, counterfactual control at n = 8–9 per arm:** attributed (full 0/8, simplified
  8/8, p ≤ 8e-5, power 0.99) on semver, marshmallow-jsonschema ← marshmallow (stitched), and swarm-built boltons and
  funcy. The frontier fails these windows because of the dependency, which is the intended axis.
- **Too easy for the frontier:** glom stitched, rebreak, and swarm semver, cachetools and more-itertools (4/4 in both
  arms).
- **The band fails at the frontier:** deterministic success or failure means no spread. Attributed windows are benchmark
  items for the frontier and training candidates for weaker rungs.
- **New S7 rule:** every import path the hidden tests use must be importable at A or stated in the prompt.
- **An oracle that skipped its plan:** the cachetools oracle deprecated `MRUCache` instead of removing it, and every gate
  passed it; the auditor and the frontier exposed it. Phase A's S4.5 now catches it.
- `lh_run.py`'s harness masks a lost World container (`InfraWatch`).

Benchmark v2 and the World leak fix (`factory/benchmark.py`, `factory/sealing.py`, `lanes/windows_world.py`; $0):
- **Training cutoff** 2026-07-03T00:00Z. Held-out windows end after it and share no repo or org with any of 1,443
  training sources (835 orgs). Training sources ending after the cutoff are embargoed. Timing tests are never
  milestones.
- **10 sealed tasks** (7 H1, 3 H2). 61 of 72 gradeable windows were independent; the path to 20 is more held-out repos.
- **Leak found and fixed.** One World image was shared by every window of a repo, so the installed copy of the project
  in site-packages could be a *later* version: the solution, inside the agent's World. Confirmed in held-out w3lib and
  pyjwt, and in training arrow-wce69c31.
  - The builder now reinstalls A last.
  - The leak probe `no_installed_project_source` fails every older image.
  - **Every store3 / P5 / P6 World must be refreshed (`benchmark.refresh_world`) before training or targeting.**
  - Earlier "too easy" verdicts may have been inflated by this leak.

Phase A, clean-up (`lanes/windows_world.py`, `lanes/windows_s3.py`, `generators/{stitch,swarm,open}.py`, new
`factory/oracleplan.py`, `factory/root.py`, `factory/traceaudit.py`, `scripts/lh_phase_a.py`; $1.63):
- **Every World refreshed, with evidence.** 65 tasks across every store. Each leak goes through the executed `world_bug`
  acceptance and gets a task version. The old Worlds of 55 tasks held installed source, 10 held another A, and 6 held the
  solution: training rebreak, arrow-wce69c31 and open toolz, plus 3 held-out tasks. No stored solver trail read it.
  - 47 Worlds were refreshed and pass S3.
  - The 8 whose rebuilt World fails S3 keep the old image and must not be trained on. 7 of them fail S3 before the
    rebuild too.
- **The fix had gaps, now closed:**
  - an unstated project name (semver) dropped nothing and the probe passed blind. The probe now also finds the
    installed copy through A's own modules;
  - stitched Worlds had no fix at all;
  - "A reinstalled last" broke python-dotenv. A PEP 517 fallback fixes it.
- **The too-easy verdicts hold.** On the refreshed Worlds the frontier still solves rebreak 5/5 and swarm semver and
  more-itertools 4/4. Cachetools and glom were not re-checked: the OpenRouter account ran out of credit.
- **S4.5, the oracle against its plan** (`oracleplan.py`; the swarm machine's `oracle_plan` state). It executes each
  planned removal, deprecation, addition and option on A and on the oracle, and tells deprecation apart from removal.
  A test of A that uses a removed name is a defect, never a regression check. Of 10 stored swarm windows, it catches
  only cachetools: `MRUCache` and `mru_cache` were deprecated, not removed, and 23 contradicting tests still pass.
- **Root curator:** per-family task repo, in-flight announcements checked before a lease, free abstention, and the 40%
  herding guard at the S17 merge.
- **Auditor forced final-verdict turn:** a harness switch that only `AuditorHarness.refine` can enable. Verdicts on
  capped sessions rose from 0/16 to 13/17. `evaluate` rejected it twice, both times for exogenous noise, so it stays
  off.

P4, failure targeting (`factory/targeting.py`; `lanes/windows_s3.py`; `lanes/windows_world.py` recipe 3; `windows.py`
`simplify('locate')`; $2.05):
- **World fixes (S3, the World expert).** Recipe 3 installs what a World needs beyond packaging, each with its reason:
  a developer baseline (make, hexdump, xxd, patch, file), setuptools and wheel, the commands the repo's own Makefile, tox,
  CI and scripts run, every non-doc extra, and known optional imports. Unknown names are recorded, never installed.
  When A's own tests fail in the World, an executed version search walks the implicated distribution's releases from
  the pin. S3 = builds, A's suite runs, leak probes, 60 clean reruns, verifier unchanged (oracle 1, start 0).
- **Counterfactual control** on the target milestone (success with its prerequisites): Fisher one-sided, a bootstrap CI
  on the target-set Φ gain when Φ is graded, power at the reference effect; verdicts attributed / not_attributed (S2) /
  underpowered (flag). Plus the prerequisite probe, paraphrase stability, band and cause mix; S15 as a machine handler.
- **World results:** 7 World fixes accepted as task versions: make ×2, setuptools ×3, hexdump, and PyYAML version
  skew in glom (found by S3 and repaired with 5.1.2). P3's glom "yaml missing" no longer reproduced; toolz's 3.12-only
  failures need an older interpreter.
- **Targeting results** (qwen3.7-flash, 4 per arm):
  - the seeded tooling-only failure is `not_attributed` (0/3 vs 0/4) and fails the probe;
  - parse, glom and humanize are `not_attributed` (target 0–1/4 even with prerequisites applied);
  - python-semver is `underpowered`: 0/4 vs 3/4, Fisher p 0.071, power 0.41;
  - parse's paraphrases are stable.

Kernel-migration family (`docs/research/swarm-pipeline/06-kernel-family.md`; `factory/lanes/kernel*.py`, `scripts/lh_kernel.py`,
`scripts/lh_kernel_gpu.py`; $0.36):
- **Ground truth:** real GPU-migration commits (texture/surface references, legacy warp intrinsics, removed cuSPARSE calls)
  mined by pickaxe from permissive repos; the oracle is the commit. PCL (task04's repo) is sealed from training.
- **Grader** (`lanes/kernel_plan.py`, a kit plan): the grader owns the nvcc command (`sm_<arch>` SASS only, arch from
  `CRUCIBLE_CUDA_ARCH`); `compile` milestones (CPU) and `run` milestones (GPU: binary check, CUPTI launch count, the
  program's observation, i.e. stdout, device-to-host bytes via link wraps and written files, equal to the original built
  with CUDA 11.8 as compute_60 PTX, fresh `srand` seed; sanitizer and grader-clock perf for L4).
- **Result:** 40 windows, 21 pass every CPU gate (S3, S4, original, S6 compile edges, S7, S8 through the kit); 20 are L1,
  1 has an L3 shape; GPU gates are staged, not run (lab unreachable). The 7B student compiled 0/26 L1 episodes before
  OpenRouter's credits ran out: L1 is above its band.
- **Fixed while running:** CRLF sources gave unappliable oracle patches; the grader now always links `libcuda`; a masked
  episode (provider 402) was read as a 0 by the targeting script.

Student and builder (`docs/research/swarm-pipeline/08-student-and-builder.md`; `actors/claude_code.py`,
`factory/student.py`, `factory/failures.py`, `scripts/student_serve.sh`; the MVP loop: run the student, mine its traces,
build tasks that target its failures):
- **Role map** (`models.DEFAULT_ROLES`, preset `tinker`): Claude Code builds, and Tinker samples every other role.
  - S9: DeepSeek-V4.1-Flash.
  - Reviewer and the student's trace auditor: GLM-5.3.
  - Judges and cause labels: GLM-5.3-Flash.
  - Hackers: DeepSeek and Nemotron.
  - Rewriters: DeepSeek and GLM.
  - Frontier: Qwen3.5-397B and DeepSeek.
  - Ladder: Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B.
  - Target student: Qwen3.6-35B-A3B.

  `tinker` is a registry slot (`register_backend`) that the Tinker bridge fills, and OpenRouter is a fallback preset
  only. The registry refuses a judge, reviewer, hacker or auditor on a builder's family or on a student's family, and
  requires the hackers to span two families.
- **Builder:** the builder roles (planner, oracle builder, verifier writer, prompt author, author, World expert) run on
  headless Claude Code on the user's plan (`ClaudeCodeClient`, `claude-code:opus-5.5`, family `anthropic`). JSON roles
  are `claude -p --output-format json` with no tools and a validated `--json-schema`. Agentic roles run Claude Code in
  the session's private clone, with file tools confined to it and code run only through `docker exec` into the offline
  sandbox (`SessionPool`). Plan usage (turns, duration, API-equivalent cost) goes to receipts, never to `spend.usd`.
  `ModelBook` raises `SameFamilyError` for any independent role (reviewer, realism, second verifier, auditor, hacker,
  tie-break, judge) on Claude Code or on the builder's family.
- **The Tinker swarm (2026-10-23, `swarm-pipeline/13`; the user: new tasks come from the swarm, not from Claude):**
  - **Preset `swarm`** (`models.SWARM_ROLES`, no Claude Code role, refused if one is set): builders over three families.
    The oracle builder and oracle author are on DeepSeek-V4.1-Flash. The planner and the first verifier writer are on
    GLM-5.3. The prompt writer, World expert and mutant builder are on Kimi-K2.6, its 128K variant. Every checker is on
    a family nothing builds: Nemotron-3-Ultra (reviewer, second verifier, auditor), Nemotron-3-Super (hacker,
    rewriter) and gpt-oss-120b (judge, realism, second hacker, tie-break). The fairness solvers are Qwen3.5-397B and
    Nemotron-Ultra. The strict presets (`STRICT_PRESETS`) also refuse a fairness solver of a builder's family, and an
    oracle builder and first verifier writer of one family (V2 by family).
  - **Provenance** (`factory/provenance.py`): every artifact carries the families that produced it, transitively over
    `inputs[].from` and declared service reads. A checker refuses (`SameFamilyError`) an input its own family produced.
    It is checked before a run (`static_problems`) and at every activation, with the family of the client that runs.
  - **Builder sessions on the RLM harness** (`factory/sessions.py`): a session may declare `potential`.
    - **Φ:** the checkpoint sidecar grades the builder's worktree out of band (`factory/builder_potential.py`:
      `prompt_draft`, `hidden_tests`, `oracle_commits`, each the static half of the gate its artifact faces). The policy
      is probed, so compaction points count.
    - **Context:** the window fits the model's context (`context_window`) and the rebuilt context (`window_floor`).
    - **Budget:** the state's dollars stop a session mid-way (`StateBudgetClient`, a budget stop is the task's failure).
      Its spend is re-booked into the machine's scope.
    - **Budget stops:** a session may `settle_on_budget`, so its captures still reach the gate.
  - **Journal, credit, diagnosis** (`factory/swarm_journal.py`, `scripts/swarm.py`): sessions, verdicts and failures are
    logged with provenance. A failure is blamed on the producing role, and a harness stop on `harness`. Per-turn
    builder credit runs through `credit.compose.turn_credit` over each state's attempts. Yield and cost are tabulated per
    role and model. `diagnose` ranks the bottlenecks and lists proposals, never applied. A remote host's outages
    (`hosts.jsonl`, fed by `world.hostreach.listen`) are counted per host (`tables()["hosts"]`), never blamed on a role.
  - **Lanes:** `write-prompts` in `lh_debug.py`, `lh_kernel_debug.py` and `lh_kernel_debug_migrate.py`
    (`factory/swarm_prompts.py`). Per window, a `prompt_author` session drafts the report, the lane's own lint and
    ledger decide, and the reviewer reads it once. `lh_swarm.py --preset swarm` fills `SwarmRunner`'s slots (now also
    journaled, with builder Φ). `lh_debug_target.py --preset swarm` uses the swarm's judges and frontier.
  - **Difficulty interface:** `factory/difficulty_hooks.py` (`KnobAction`, `Controller`, `Reviser`, `Gates`,
    `revise_and_gate`; a held-out task is never revised). One knob vocabulary with the controller (2026-10-23, the
    pilot): K1-K6, C1 (checkpoint start), H1-H3 of `factory.difficulty`; the first draft's distractor knob is gone (it
    asked for behaviour no check grades, V5).
  - **The difficulty loop on the swarm** (2026-10-23, `swarm-pipeline/12` §7, 13 §9): `factory/difficulty_bridge.py`
    (`ControlController`: `difficulty.control` as a hook `Controller`; `choose_checkpoint`, the level whose remaining
    milestones' measured conditional rates multiply closest to 1/2 in band; `k1_target`; `stats_from_targeting` with
    re-pathed frontier verdicts) and `factory/revise_debug.py` (`DebugReviser`, `DebugGates`: a variant is an ordinary
    stored window `<parent>-<knob>-<key>` with `revision = {parent, knob, applied, pre_applied, level}`; its report is
    written by the swarm's `prompt_author` under a knob instruction; World knobs re-run the start check and every executed
    gate, report knobs inherit the parent's gate verdicts only when the start tree and the hidden tests are byte-identical
    by sha256). Driver: `scripts/lh_difficulty_pilot.py propose|revise|measure`.
  - **The training monitor** (2026-10-25, `swarm-pipeline/11` §4.4, 13 §11): `factory/train_monitor.py` (analysis) and
    `observe/train_monitor.py` (W&B tables and lines, the markdown and HTML report; plain dicts in, so `observe` stays below
    `factory`), driven by `scripts/train_monitor.py report|watch`.
    - **Ingest:** every attempt under the roots (Runner layout `.../samples/<s>/a<n>`), read through
      `traceaudit.load_bundle` plus the record's honesty block and the grader's check statuses. A training run's
      `steps.jsonl` maps groups to steps; other roots are cohorts. Backend-neutral (Tinker or DeltaAI write the same
      directories).
    - **General modes** (`GENERAL_MODES`, deterministic): stalls after a partial fix, claims verified work with no test run
      after the last edit, overclaims in STATUS.json, slow first edit, budget with Φ flat, loops on a failing command,
      breaks pass-to-pass tests, loses Φ after a compaction, environment misuse; `symptom_only` comes from a capped judge
      pass (never Qwen, a verbatim quote or the label is rejected, training failures only). A mode is general at >= 3
      tasks over >= 2 languages or lanes (Wilson CI); the rest, and each task's stuck milestones, are task-specific.
    - **Learning:** per step mode prevalence and health (reward, full pass, Φ, budget stops, hacks, path violations, tool
      calls, compactions, Brier; the trainer's KL, loss, entropy and zero-spread groups); first vs last third gives
      `learned`, `regressing`, `stuck`. Unordered cohorts (arms of one run) get no trend. Alerts: hacks up, honesty down,
      reward up with full passes flat, KL spike, entropy collapse, flat groups, the band emptying.
    - **Proposals and intents:** each persistent general mode maps to framework levers (`LEVERS`), journaled,
      `applied: False`; `swarm_intents` feed `swarm_journal.diagnose(monitor=...)` (`scripts/swarm.py diagnose --monitor`).
    - **Held-out discipline:** benchmark episodes are aggregated only (placeholders); `proposals`, `swarm_intents` and
      `judge` raise `HeldOutUse` on a held-out episode.
  - **Framework health** (2026-10-25, `swarm-pipeline/11` §4.5, 13 §12): `factory/framework_health.py`, driven by
    `scripts/train_monitor.py health|fix` and inside `watch`; rendered by `observe/train_monitor.py` (`health_markdown`,
    `log_health`). It is the framework side of the monitor.
    - **Detectors:** deterministic, each labelled agent or framework and given a component (World, harness, grader,
      sandbox, sampler, trainer, task spec). They cover masked failures, unmeasured checkpoints, World and tool faults
      (core toolchain against convenience tools), host resources, token truncation, provider errors, nondeterministic
      grades, the sidecar and final grader disagreeing, re-pathed false positives, stale visible tests, protected-edit
      consensus, unreachable milestones, spec defects, codec suspects, S9b false fails, trainer numerics, job-log
      errors and unreadable attempts. A remote host that stopped answering (`host_unreachable`, 2026-10-26) is one
      sandbox cluster per host: HIGH for the attempts it masked (in place of the generic mask), LOW for those that paused
      and resumed (`meta.host_outages`); a call it cut is not counted as a tool error.
    - **Clusters:** grouped by a normalized signature, each with episodes, tasks, first occurrence, a reproduction, an
      impact and a severity.
      - **Impact:** the reward moved or at risk, the advantage distortion per rollout group (`sum |Δ advantage|`, sign
        flips), and the samples and groups a masked defect cost.
      - **Severity:** CRITICAL corrupts reward or advantage, HIGH wastes episodes, LOW is cosmetic.
    - **First CRITICAL fixed** (2026-10-25, `swarm-pipeline/11` §4.5): the checkpoint sidecar and the final grader
      disagreed on one episode's end patch, so Φ and per-turn credit missed +0.5 the outcome counted.
      - `CheckpointSidecar.reconcile` makes the final grade's vector the end state's one verdict (`WorkspaceHarness` by
        patch, `NativeHarness` for the end checkpoint), rebuilding the series, rows and summary.
      - A test run killed from outside is masked, never a failure of the tests it did not reach (`executable.killed`,
        the Go profile's package-level `signal: killed`).
      - `tests/unit/test_credit_grade_agreement.py`.
    - **Triage:** `pause` while a CRITICAL framework defect is open (the CLI exits 3).
      - **Fixes:** recorded with `record_fix` in `<run>/framework_fixes.jsonl`, which sets a defect to `fixed@<version>`,
        or to `recurring` if it shows up after the fix step.
      - **Regrades:** `regrade_plan` names the offline tool (repath, or `factory.tinker.regrade` with
        `versioning.OfflineGrade` rows that never enter training).
      - **Versions:** `versions` gives the path-rules and reward-rule versions per step; a mix with no fix record is a
        `version_mix` defect.
    - **Metadata and tuning:** per lane and step, `metadata` covers tool failure, latency, tokens, context use,
      compaction Φ, checkpoint cost, mask rate, GPU and World minutes, budget stops, first edit, and useful samples per
      dollar and per GPU-hour. `tuning` turns these into proposals that are never applied: cadence, compaction,
      truncation, output tokens, concurrency, group size from `1 - p^k - (1-p)^k`, tool affordances, and the band.
    - **Held-out:** `redact` keeps component, kind, severity and counts, and withholds tasks, episodes, reproductions
      and content-bearing signatures. Tuning reads training traces only.
  - **Controller fixes from the pilot:** a budget-bound task gets the horizon-shortening knobs (C1, K1; the 300-call
    budget never moves); a root milestone's rate is no checkpoint signal; a conditional rate above the band is not in band.
  - **Pilot fixes:** `sampler.ensure_cap` (every `write-prompts` stopped at the shared meter's 1-USD default whatever its
    `--cap`); `lh_debug_target` refuses unknown task ids before any episode (a missing candidate raised inside the gather
    and killed the episodes in flight).
  - **Reviser and report fixes from the pilot** (`swarm-pipeline/13` §9.2):
    - A knob changes only what it targets: C1, K1, K2 and K4 keep every kept requirement word for word; K3 appends an
      area; K6 rewrites one milestone. A free rewrite dropped a fact the hidden tests check.
    - K1 drops only a fault at a measured rate of 0.7 or less.
    - A revision refused for an infrastructure reason is an infra row, with no blame on the reviser.
    - Findings group by what they say.
    - A report before `mine` marks failures `unmined`, never `off_intent`.
    - The install verdict names an attempted fetch.
  - **Round 2 fixes** (`swarm-pipeline/13` §10.3):
    - K6 must carry an executed repro (S7r verified) for its milestone.
    - A checkpoint start's pass-to-pass tests are graded at the new start, and those failing there leave the
      regression check.
    - Kernel lint `CAUSE` refuses a diagnosis in a symptom report.
    - Stale prompt runs (another machine, or a new hand-off under the same id) move their run, session state and
      hand-off branch aside.
    - K4/K6 writers get 120 turns.
    - `lh_kernel_debug.py build --subsets` (`kdm.connected_subsets`).
    - Diagnosed knob overrides in the pilot.
  - **Pilot result:** 1 kept training task, `db-python-black-ae5830b7-c1-516b23`. It is a C1 checkpoint start written
    by the swarm. It cost $129.7 of Tinker over 8 measured variants. Step 2 found 0 new kernel tasks: llama.cpp's
    unused fixes do not revert together, and there is no harness for the other CUDA repositories yet.
  - **Smoke fixes:** the Kimi native tool section is read by the text protocol, the sidecar probe was added for
    compaction points, the capped client's spend was lost, and a window floor was set.
- **Student:** the target is `tinker:Qwen/Qwen3.6-35B-A3B`, the model that is post-trained, so tasks are calibrated
  against exactly it. The local dev model is `student:Qwen/Qwen3.5-4B`, for harness work and the pipeline smoke only.
  - The local path uses native OpenAI tool calls (the `qwen3_coder` parser on mlx-lm) with thinking off. Thinking is
    sent with every request and recorded with every run.
  - Local serving (mlx-lm on the Mac, vLLM on CUDA) is dev and calibration only.
  - The backend is stamped into the policy id (`@mlx`). Only a target model on a final backend (`tinker`) can produce an
    S15 verdict or a benchmark report (`assert_countable`).
- **Failure mining:** a finding per failing episode, keyed by the student model id. It holds the milestones missed and
  lost, the first wrong step, the cause and a verbatim quote, and is built from the trace audit's own pieces. Skill modes
  cluster per student into builder-intent requests: an
  `IntentCard` candidate, the swarm planner's intent, or a root `Cell` after a human approves it. Harness, World, budget
  and infrastructure failures are routed to their owners.
- **Fixed:** `lh_run.policy_ref` declared no `config`, so the Runner refused every solvers, probe and arms stage at
  start (M10 R-4 `Undeclared`).

Kernel debugging lane (`docs/research/swarm-pipeline/10-kernel-debug-lane.md`; `factory/langs/cuda.py`,
`factory/lanes/kernel_harness.py`, `kernel_debug{,_mine,_ops,_world,_plan,_migrate}.py`, `scripts/lh_kernel_debug{,_migrate,_gpu}.py`;
$0, no model calls, no GPU; student Qwen/Qwen3.6-35B-A3B; builder `claude-code:opus-5.5`):
- **Families:** `kernel-bug` (2-4 real CUDA kernel bug fixes reverted together at B, connected through shared ops),
  `kernel-perf` (real kernel speed-ups reverted: results still correct, a per-GPU perf bar fails), `kernel-migration`
  (2-4 real toolkit-migration commits reverted at B: CUDA 12 -> 13 / CCCL 3 removals, CUDA 11 -> 12 texture/warp changes,
  merged into multi-program windows on the kernel-migration machinery). Training: ggml-org/llama.cpp and the two
  migration repos; held out: ikawrakow/ik_llama.cpp (disjoint repository and commits, same harness type), sealed by the lease.
- **Harness profiles (2026-10-24, doc 10 §4.1):** a repository plugs in through a `KernelHarness` profile (`factory/lanes/kernel_profiles.py`: build, case listing, per-case run and dump, reference and tolerance as `dump` or `status` comparison, sanitizer run, perf, seeded or fixed inputs). `GgmlHarness` delegates to `kernel_harness` unchanged (same environment, commands and verdicts; the ggml reference identity unchanged); `Catch2Harness` runs self-checking Catch2 GPU tests, first CCCL's CUB (`catch2-cccl`, training), smoked on the 5090 (ReduceByKey `6ad0be74`: B 3/3, revert 2/3, the fix's own case broken). Held-out repositories are named only in the gitignored `data/kernel_debug/harness_repos.json`.
- **Harness and grader:** the repository's own op test (`test-backend-ops`) as of B, owned by the grader and patched in
  two layouts (seeded inputs, case selection and listing, dumps, exact repetitions; anchored repairs where a fork's file
  no longer compiles); the grader builds the candidate's CUDA backend only (everything else from the start tree) with
  SASS-only flags, compares each case with the reference build's CPU output on fixed and fresh seeds outside the
  candidate's process, requires kernel launches, runs compute-sanitizer for race/barrier/bounds faults and grader-clock
  perf bars with a measured noise floor; no GPU or no calibration -> masked, never a 0.
- **CPU now, GPU bundled:** synthesis (history-ordered chain, B' == B), compile of every touched unit at A' and B for
  sm_90 and sm_120, the harness built and its case pools listed and seeding executed on the CPU backend, World probes,
  floors and mutants (apply and build), prompt lint + ledger + fairness lint, intent cards; the bundle
  (`data/kernel_debug/gpu_bundle`) calibrates case ownership, executed edges, sanitizer and perf bars, then runs S5, S8
  and S10 on the lab 5090 (Docker) or Marlowe (Apptainer). Results: `10` §8.
- **Hardware reach (2026-10-23, `kernel_debug_reach`):** a mined fault qualifies only when the GH200 (`sm_90`) or the
  RTX 5090 (`sm_120`) compiles and dispatches the changed code (each changed file resolved as that GPU sees it: its
  `__CUDA_ARCH__` or the host pass, HIP/MUSA/NCCL undefined, `common.cuh`'s feature macros and `*_available(cc)`
  predicates, runtime `cc` tests folded three-valued, code after a `return` dropped, identical branches merged; anything
  undecided counts as reached), the bug family's change is not speed-only, launch-shape-only or a precision-only kernel
  selection by its message, no hardware we lack is named, and the op test builds one of its ops. Mining drops the rest
  with the reason; records carry `targets` per fault and `reach` per milestone. `retire` moves a record to
  `data/kernel_debug/retired/<id>/` with its reason (reversible; the report lists it); `retire --auto` once every target
  GPU has run and fewer than two milestones are gradable. Retired: llama.cpp 7ac249cc, 4b8c156e, b377258d; held-out
  ik_llama.cpp 5b19ab0c, 4da3f014 (`10` §3.1).
- **GPU calibration rules (lab 5090 run):** S1 executed starts with a sweep of A' (every fault in) over every case of every
  pool on the fixed seeds; the cases it fails join the 1,000-case sample the rest of the calibration grades; if A' passes
  every pool case and every sanitizer check, no fault shows on this GPU and the task drops; a milestone owns the pool cases its single revert
  breaks (or, when the revert does not apply at B, the cases its fix alone repairs); the oracle is re-graded three times on
  the calibrated cases with fresh seeds and any case B fails leaves (input-sensitive); half-fix mutants are graded over
  the pool: one that breaks cases adds them to its milestone, one that is whitespace-only or breaks nothing is recorded as
  equivalent on the op tests and leaves S8, as does one that does not build. The grader builds statically (sealing removes
  symlinks), names nvcc, splits outputs into chunks of 96 cases, and drops a case the reference crashes on; gate runs share
  a ccache (task grades never do); x86-only CPU gates (ik_llama.cpp) run on the GPU host first (`cpu_host.json`).
  The migration windows run on the kernel-migration grader (`kernel_plan`), which now runs the original and the candidate
  under the same program name (a program that prints its argv[0] failed the oracle).
- **GPU gate fixes and corrections (2026-10-22, `10` §8.2):** grader and runner bugs found by re-reading the failed runs:
  `pool_cap=0` reached the plan as the integer 0 and fell back to the 1,000-case stride, so the S1 sweep "of every pool
  case" covered 10-20% of the large pools (`kernel_debug_plan.pool_cap`); a cached pool grade's reuse key now includes the
  grader kit (`kit_digest`), so a grade from before a grader fix is never read back; the S1 start drop skips perf
  milestones (a perf start is correct by design) and pool grades never run the perf check (no bar exists yet: its infra
  verdict masked every perf task's oracle pool grade); perf calibration times every candidate case on its own inside
  one grade (`perf_each`: three grades per milestone, not three per case); S10 counts only grades that measured (a
  masked grade is no pass), runs a perf task's sixty grades one at a time, and runs only for a task that passed S5 and
  S8; the start of a correctness calibration no longer waits for an idle GPU (only timed steps do). New calibration
  inputs: each sanitizer milestone runs its tool on `sanitizer_cases`, pool cases whose processes launch the fault's
  kernels (`kernel_debug_ops.fault_kernels` at staging -> `fault_kernels.json`; a CUPTI launch profile of up to 192
  pool cases on the reference build), not on the pool's first cases; a stability grade drops an owned case B passes
  above half its tolerance (`case_ratio`, `MARGIN`); a record may carry grader-owned cases (`harness_extra`: only
  `test_cases.emplace_back(new test_X(...));` lines of the file's own classes, at the shapes its report describes,
  `kernel_harness.add_cases`), which join their milestone's pool as added cases and are disclosed in the prompt
  (`EPILOGUE_EXTRA`). Migration grader: `compare.d2h = "none"` grades a self-checking program on its own report (its
  copies back were 0.7-9.9 GB), and a window with no timed milestone does not wait for an idle GPU. The shared gate
  ccache is capped at 12 GB (shared disk). Perf timing: the candidate's timed runs no longer carry the CUPTI launch hook (launches are counted on separate untimed runs; the traced build timed 4-14% slower than itself), and a passing perf check's ratios are read where it reports them. Blackwell consumer GPUs build as `12xa-real` (FP4 MMQ units failed ptxas at `120-real`).
- **Apptainer GPU hosts (DeltaAI GH200, 2026-10-22, `10` §8.3):** both GPU runners (`run_gpu.py`, the migration
  `stage_gpu.py`) run their containers as SIFs (`--mode apptainer`; `--build-sif` converts the bundle's Dockerfiles with
  `kernel_plan.apptainer_def` on the GPU host, apt as root in the build) through one argv builder,
  `kernel_plan.apptainer_argv` (`--nv --containall`, no network, the host's node-local disk bound at /tmp: a CUDA build
  does not fit the session tmpfs). DeltaAI has no subuid range, so `--fakeroot` maps uid 0 only and the grader cannot
  drop to its candidate uid: `CRUCIBLE_APPTAINER_ROOT=user` runs the grader as the invoking user without `--require-root`
  and every result records `runtime.trusted: false` (fit for the gates, which grade the lane's own patches; grading a
  policy there needs a subuid range or another sandbox). The Docker path is unchanged. Apptainer `--nv` reaches the driver's
  libraries only through the container's LD_LIBRARY_PATH, which the grader's rebuilt environment dropped (every CUDA
  program saw no device, "Unified Memory not supported"): `Trusted.run` appends `/.singularity.d/libs` when it exists.
  `ingest-gpu --hardware-only` writes a second GPU model's gates to `gates.gpu[<GPU>]` and its calibrated milestones and
  bars to `gpu_specs[<GPU>]` without touching the record's milestones; `report` adds `gpu_by_hardware` per task and
  `gpu_pass_by_hardware` per family. The S3 start build now also compiles `start_dependents` (units no fault touches that
  use a name a fault's header change takes away at A'), which would have caught `kd-perf-llamacpp-3ad8a68c`.
  Perf calibration (GH200, 2026-10-24): a perf milestone's golden check now runs on cases B passes (its revert breaks none
  by design; with no cases every grade masked), and a perf candidate case is kept only when B against itself stays within
  0.5-2x and the revert is at most 20x slower (`plausible_timing`: the GH200 read ratios of 0.0 and thousands).
- **Grader gaps closed on the existing tasks (2026-10-23, `10` §8.4):**
  - **A case's inputs no longer depend on the cases run before it.** The harness drew every input from one
    process-wide counter, so a case's inputs changed with its position in a chunk: the calibration (96-case chunks) and
    a later grade of the same case saw different inputs at the same seed, and a revert's break seen in the calibration
    did not repeat at S8 (`cef66d4a` m2, `a45699aa` m3, held-out `35d83401` m1). A selected case now restarts the
    stream from (seed, case name) (`crucible_case_start`); the 2024 layout's static per-thread engines and the 2025
    layout's first-tensor thread count are rebuilt per call. Checked on the 5090: the same cases dump identical outputs
    alone and inside a 35-case chunk, in both dialects.
  - **Ownership on fixed-seed breaks.** Pool grades report `per_case_fixed` (the spec's fixed seeds only); a revert owns
    a case, and an executed edge is drawn, only on a break there (one fresh-seed break no longer owns a case S8 then
    misses).
  - **An exact attention reference where B's own CPU kernel is wrong.** ik_llama.cpp's IQK CPU flash attention (taken
    when `max_bias == 0`) disagrees with ggml's semantics on the 2024 op test's inputs, so B failed 237 of its own 455
    FLASH_ATTN_EXT cases. The reference build can replace the CPU column of a listed case with an exact computation
    (`crucible_fa_exact`: scale, ALiBi slopes, softcap, additive mask, softmax and GQA in double from the CPU graph's
    inputs, dequantised by the file's own `tensor_to_float`; `CRUCIBLE_FA_REF_CASES`). The calibration lists a case
    only on evidence (`KDGrader.fa_probe`, `fa_reference_verdict`, `fa_exact_choice`): B's GPU must match the exact
    result within half the tolerance, the CPU kernel must not, B's GPU must be bitwise stable across two processes, and
    the exact computation must agree with both on at least 8 cases where they agree with each other; otherwise nothing
    switches. Recorded per task as `calibration.fa_reference`.
  - **The reference's identity includes the harness hooks** (`harness_digest`): a hook change rebuilds every reference.
  - **Migration S8 read an empty field.** It took per-gate verdicts from the reward document's `gates`, which
    `kernel_plan` leaves empty, so no fault mutant ever counted as killed. Gates now come from the report (or the
    `check/<gate>` components) and a mutant is killed when a milestone of the program it targets fails
    (`gates_of`, `mutant_targets`, `mutant_killed`). A window may list mutants shown equivalent on the program's own
    inputs (`equivalent_mutants`: the reason citing the code), listed apart as `S8_mutants_excluded`; a window whose
    every mutant is excluded fails S8. The migration S10 leaves masked grades out of the denominator (`flake_verdict`,
    as the bug lane does).
  - **A sanitizer tool keeps a milestone only when it discriminates** (clean on B and dirty on the revert,
    `sanitizer_discriminates`); a tool dirty on both had kept milestones with nothing left to grade them.
  - A grade that leaves no reward logs the container's error in the run log.
- **GPU episodes (2026-10-24, `10` §9, `factory/lanes/kernel_episode.py`, `scripts/lh_kernel_target.py`):** a student,
  frontier or swarm episode on a kernel task now runs on a GPU host, not in a CPU-only World.
  - **The World** is one container on the host (the lab 5090's daemon through the shared SSH tunnel), with
    `--runtime=nvidia`, `--network none` and CPU, memory and PID limits. A new World backend, `container_exec` (registry
    `world_backend`, `world/backends/container.py`), routes every App's file tools, shell, git and `fs` state snapshots
    into that running container, so nothing of the World lives on this machine.
  - **The start tree** is the gated A' with history cut (one `base` commit) and a CUDA build of A' already configured and
    built in `/ws/build` (gitignored). The template is made once per task, arch and flags on the host, under its build
    lock. The agent builds with nvcc, runs the op tests on the GPU and uses compute-sanitizer.
  - **Collection:** the harness collects from a local mirror pulled out of the container (tracked and untracked, not
    ignored, files and `.git`; `workspace`); the checkpoint sidecar snapshots inside the container.
  - **Grading stays out of the agent's reach:** every checkpoint and final grade is `run_gpu.py --grade-one` (kernel
    bugs) or `stage_gpu.py --grade-one` (migrations), run on the host from an episode bundle. That bundle is this
    checkout's runner and kit, with the gate bundle's tasks, hooks and cache linked in. Each grade is a fresh grading
    container on the pristine start tree plus the patch, with the calibrated hidden cases, on the GPU model the task was
    calibrated on. The episode grade is the verifier's `GradeReport` with the window lane's path gates and path reward
    (`check_paths`, `path_reward`: a hack-severity violation scores 0). The World liveness probe asks the World's own
    daemon (`envs.world_alive` reads `_crucible_env`).
  - **Smoke (`10` §9.2):** on `ca21f3a1` the oracle grades 1, the empty patch 0 and `revert-m1` 0.25, at 15–20 s each. A
    student and a DeepSeek episode at 40 calls built, ran the op tests and compute-sanitizer on the GPU, edited code, and
    were graded at their checkpoints and at the end. The four smoke runs cost $1.64 of Tinker.
  - **The gate cache, read-only:** kernel-bug episode grades read the gate runs' build cache read-only (`ccache_ro`:
    mounted read-only, `CCACHE_READONLY=1`), so unchanged units are reused and a candidate can neither write nor poison it.
  - **Locks:** the host's GPU lock (`flock`) is shared for correctness grades and exclusive for a perf task's grades and
    its open World, and a perf task's checkpoints are deferred to the end (`episode_checkpoint`, read by
    `envs.make_env`). Grades and template builds take the build lock one at a time, because parallel nvcc builds ran
    the shared 5090 out of memory.
  - **Masking:** an unreachable host or a missing spec masks the grade (V8).
  - **Host outages (2026-10-26, `world/hostreach.py`):** when this Mac lost its network, every in-flight episode's tool
    calls hung until the 9,000 s attempt clock, and all 8 episodes of a measurement were masked after $29.5 of sampling.
    Every call to a remote host now runs under that host's `HostReach`, shared by the process (every `DockerExec` on a
    remote daemon, so the lab's task sandboxes too, `SSHExec`, `kernel_episode.remote`, the workspace pull, the liveness
    check). The local daemon's calls are unchanged.
    - **Fail fast:** a call still running after 15 s has the host probed (`docker version` through the shared tunnel, or
      `ssh ... true`), and two failed probes in a row cut it with the typed `HostUnreachable` (at most 50 s). A transport
      error the call reports itself (no daemon, no tunnel, ssh rc 255) is confirmed by one probe, never read from its words
      alone. SSH options are `ConnectTimeout=15`, `ServerAliveInterval=15`, `ServerAliveCountMax=4`. The workspace pull
      runs off the event loop and is killed after 60 s without data (on the loop, one stalled stream held every episode).
    - **Pause, then retry:** the episode pauses with every registered clock held (the Runner's attempt clock, the agent
      clock, the World's tool deadline, a grade's scoring budget), and probes with backoff for up to 30 min per attempt,
      never past 30 min after the host went down. Back: the call runs again if nothing reached the host or it is
      replayable (framework calls, read-only tools); a policy's write or shell call cut mid-way returns `interrupted` to
      the agent. Model calls already made are never sampled again. Still down: `transport_unreachable` with detail
      `host_unreachable`, an infra mask (a masked final grade is regraded from the stored patch, never re-sampled).
    - **Fleet stop:** `open` starts no World while the host is known to be down, and `lh_kernel_target.py` launches no job
      (`launch_gate`; skipped with the reason, picked up by the next run of the stage).
    - **Diagnostics:** each attempt's outages are in `record.meta.host_outages` and a `host_outage` run-journal event; the
      host's events go to `<out>/journal/hosts.jsonl` (swarm journal); framework health clusters them per host.
  - **Targeting:** `lh_kernel_target.py` runs the debug lane's targeting stages (pilot, student, frontier, audit, mine,
    report; the same meter, keep rule, trace audit and presets) over these tasks, and records per-GPU grade minutes. The
    `fairness` (S9b), `disclose`, `worldfix` and `regrade` stages are refused for kernel tasks: there is no CUDA
    alternative writer or second verifier yet.
  - **DeltaAI is declared, not built:** the runner has to run on the compute node itself.

Multi-language debugging lane (`docs/research/swarm-pipeline/07-debug-lane.md`; `factory/langs/`, `factory/generators/debug.py`,
`scripts/lh_debug.py`; $0, no model calls; student Qwen/Qwen3.6-35B-A3B):
- **Language layer:** a `LangProfile` per language (python adapter, go, rust, node for TypeScript/JavaScript under vitest
  or jest, java for single-module Maven): path rules, test definitions and hiding, suite targets, a pinned World recipe
  that compiles dependencies only, a copy-in `--network none` sandbox with machine reports (`go test -json`, libtest
  lines, vitest/jest JSON, surefire XML), per-language leak probes. The window machinery asks the active profile
  (`langs.base.use`, a `contextvars` scope); no profile or Python keeps the original path, so the Python lanes are
  unchanged. The protocol is the stable contract other lanes (the CUDA debugging lane) build on.
- **The lane:** rebreak per language: fixes after 2026-05-01 (the student's release; 2025-09-01 only as a marked
  fallback), executed at the head; executed interactions; A' with source, test hunks, test data and changelog lines
  reverted (strip mode when that does not build); pre-checks (the start tree and the hidden tests build); the window
  validation; no hidden check may fail visibly in the agent's tree; S6/S5/S8/S3/S10 in the profile's scope.
- **Prompts and intent:** symptom-only bug reports written by the builder (`claude-code:opus-5.5`, recorded), linted for
  faulty function names, hidden test names, paths and versions, two-way ledger; an intent card per task (skills:
  multi-fault localization, interacting faults, reproduce from report, regression safety; why long; milestones; expected
  failures; the trace evidence that attributes a failure to the skill). Held-out debug repositories (disjoint orgs)
  are fixed before mining.
- **First run (CPU, about 11 h):** 98 repositories scouted, 56 mined, 1,379 fix units, 571 real fixes, 44 interacting
  groups, 13 candidates through every CPU gate and S7 in Go, TypeScript, Python and Rust (3 held out); Java built, no
  candidate. Yield is limited by reverts that no longer apply, rare interactions and coarse table tests (`07` §5).
- **Second run (held-out draw 3):** 80 more clones scouted, 16 held-out repositories (orgs with nothing in training)
  mined: 4 more held-out candidates (excelize, carbon), so 17 through every CPU gate and S7 (10 training, 7 held out).
  Intent cards carry `fairness_flags` for review (hidden tests named after internal functions, strip mode's visible test
  data). The node runner carries the repository's own `--experimental-vm-modules` flag (jest ESM). The store is
  `CRUCIBLE_DEBUG_ROOT` when the lane runs from a worktree.
- **Targeting, S15 (`07` §6, `scripts/lh_debug_target.py`; $201 on Tinker, 2026-10-20):** the target student
  Qwen3.6-35B-A3B (k = 8) and the two frontier solvers (k = 2) in exactly the RL environment (`factory.tinker.benchmark`:
  `make_env`, the task's World, the hidden verifier, thinking off, 150 tool calls), one hard Tinker cap for the phase
  (`benchmark(meter=)`); every failing student episode gets the deterministic facts, the first wrong step, the
  honest-trace checks (`traceaudit.honest_trace_facts`: STATUS over-claims, claimed-but-unrun tests, protected or hidden
  paths), the GLM-5.3 auditor's and GLM-5.3-Flash judge's labels with seeded controls, and its intent skill
  (`generators.debug.failure_skill`); the contract's keep rule is `targeting.keep_verdict` (training: pass rate in
  [0.1, 0.6], >= 50% on-intent skill gap (70% until the lead lowered it, 2026-10-27; `targeting.SKILL_MIN`), a frontier pass; benchmark: <= 0.5 (< 0.3 until the lead raised it, 2026-10-27: pass@1 is reported either way; `targeting.BENCH_HI`), same conditions; an open World issue sends
  the task back). Builder-intent requests come from training tasks only. Result: no task kept: the 150-call budget binds
  (67 of 104 student and 21 of 52 frontier episodes stop there, routed as `budget`; DeepSeek solves a "hard" task 2/2
  at 300 calls), echo is too easy, h3 has an open World issue. Fixed on the way: two path-check false positives (heredoc
  bodies, `go mod init`) with re-grading of stored episodes (`factory.tinker.regrade`), the node policy World
  (`langs.runtime.policy_world`), concurrent-arm accounting, the benchmark policy id the family guard reads, the judges'
  quote retry (`AuditConfig.quote_retries`), a `world_fault` fact for World tools that lack a command.
- **Env budget 300 tool calls (`generators.debug.ENV_BUDGET`, user decision 2026-10-20):** the keep rule, the benchmark
  and RL run debug tasks at 300 calls (agent 5400 s, rollout 9000 s, Φ every 20 calls, cap 15). Every phase records
  `settings.env_budget`, and a re-measurement uses fresh seeds so its trace keys are new. S15 at 300 (`07` §6.1, $326
  of $450, 11 of 15 tasks before the user stopped it): still no task kept. The budget no longer binds the frontier
  (0 of 54 limit stops, against 21 of 52), and binds the student less (24 of 88, against 67 of 104).
  black-90117fa0 enters the band (1/8, a frontier pass) but stays budget-bound (6 of 8 at the limit). Two held-out
  excelize tasks are spec defects: a hidden milestone also checks reverted fixes that have no milestone and that the
  report never describes. The judges, who never see hidden tests, labelled those failures skill gaps.
  `generators.debug.disclosure_check` grades B without the silent units through the task's own grader. The keep rule
  takes `spec_issue` (verdict `spec_defect`, owner the builder). The node World's pnpm runs offline without its pre-run
  install (`node.PNPM_OFFLINE`, recipe `node-5`; h3's `pnpm test` hang). Committed summaries seal held-out issue text and
  verdict reasons.
- **Easing support (`07` §6.2, 2026-10-21, $0; patterns from rl-env-use, MIT):** the builder's brief opens with a
  calibration block (`factory/calibration.py`): measured training tasks too easy, in band, too hard and unfair for the
  student, with pass rate, mean Φ, per-milestone rates and budget stops from the S15 records, plus the knobs K1-K6;
  held-out records are refused by split, repository and lease, and the text is checked against every held-out id.
  Gate S7r (`generators/debug_repro.py`, `lh_debug.py repro`): a prompt's optional `repro` per requirement (knob K4) is
  linted with the leak rules (never a hidden test), shown to the agent, and executed in the task World, where it must
  show the bug at A' and pass at the oracle; a missing repro is `unverified`, never a failure. The keep rule reports
  whether its band decision is settled by the attempts run (`targeting.band_power`, a flag).

- **The benchmark, sealed for real (2026-10-24, `swarm-pipeline/00` §10, `07` §4.1):** the debug held-out tasks went
  public on 2026-10-22 through a `git add -f` of `crucible/data/debug`, so sealing is now measured and guarded, not assumed.
  - **Contamination audit** (`factory/benchaudit.py`, `scripts/bench_audit.py`, `runs/benchmark_contamination.json`):
    every held-out task (debug and kernel) and every kept benchmark variant, against our public tree (`git grep -F` of
    its id, report sentences, hidden test names and the lines its fix and hidden tests add; the commit that first
    published each file) and against every training source (`bench.contamination`: 13-grams, code substrings, near
    duplicates, containment; sentences in three or more reports are template, and 13-gram hits spread over three or more
    sources or under five shared n-grams are weak). Verdicts `sealed`, `contaminated`, `unknown`, `not_heldout`, every
    reason listed; the report holds counts, paths and commits only. Result: all five benchmark tasks are contaminated.
  - **Seal guard** (`factory/sealguard.py`, `scripts/seal_guard.py`, `.githooks/pre-commit`): refuses sealed paths
    (`crucible/data/sealed/`, any `heldout` directory, `bench.key`, `sealed.json`), held-out ids and `owner/repo` names in
    added lines (the three registries exempt from the name rule only), hidden content (16-character windows of every
    fix line, hidden-test line and hidden test name, hashed; eight shared windows, or all of a short name, match) and
    the bench key. `--i-know "<reason>"` / `$CRUCIBLE_SEAL_OVERRIDE` let a commit through, journaled without the text.
    `check --tree --baseline` is a ratchet against `runs/seal_baseline.json` (the exposures public before the guard), and
    a unit test runs it on HEAD. `check --range A..B` reads what B's commits add since the merge base (`A...B`): a
    two-dot tree diff showed a branch behind main re-adding ids main had since redacted (2026-10-27).
  - **Held-out supply off git** (`factory/heldout_supply.py`, `scripts/heldout_supply.py`): candidates from `gh api`
    search, owners disjoint from every registered source and from anything our history names, permissive licences, the
    lanes' languages, post-cutoff fixes that change a test and a source file counted on a blobless shallow clone, the
    language manifest at HEAD. The list and the chosen draw live under `windows.SEALED_DIR` (`$CRUCIBLE_SEALED_DIR`,
    default `crucible/data/sealed`); `windows.local_draw` reads `heldout_draw.json` at import, so the lanes, stores and
    the eval lease hold the draw out without its names ever reaching git.
  - **Seal** (`factory/benchseal.py`, `lh_debug.py seal`): refuses unless every chosen task is held out, gated and
    `sealed` by an audit of the named ref; writes a hashed manifest, fresh per-task seeds (HMAC of a new master seed;
    `public.json` carries only its commitment), `sealed.json` for the eval lease (repositories and orgs; no time cut by
    default, since these lanes split by owner) and the `bench.power` plan (MDE for the chosen n x k; `--require-power`).
    **Accepted exposure** (the lead, 2026-10-25): `--accept-exposure TASK=REASON` lets one `contaminated` task through,
    and only that verdict; the manifest records `exposed_public: <first-publishing commits> (lead accepted <date>:
    <reason>)` per task so a paper can disclose it, `public.json` counts them, and `exposure_accepted.jsonl` under the
    sealed directory journals each one. Every other refusal still holds.
  - **Offline benchmark World** (`windows_world.require_offline`, called by `WindowTask.open`): `open_world` reads the
    container's network mode back with `docker inspect` (never assumed from the run command), and a held-out task refuses
    any World whose mode is not `none` (`OnlineWorldRefused`).
  - **Start-tree residue** (2026-10-25, from the co-authors' fixing notes): a release changelog entry that only repeats a
    fix's subject (no PR, issue or sha; echo 48128ab) is stripped through `langs.base.subject_ref` (4 words and 20
    characters minimum). In strip mode the Go profile also removes the fixes' whole table entries from package-level test
    tables (`golang.strip_table_entries`: contiguous runs of fix-added lines at the table's element level, brackets
    balanced outside literals; mvdan-sh dc951aeb-e1 kept its then/do cases readable), recorded as
    `hidden.table_entries_stripped`. `golang._code_lines` blanks literals by character, because `code_mask` shifts on
    multi-byte runes (an open defect of `code_mask` itself).
  - **S3 reruns under load** (2026-10-26, k8sgpt): a rerun the host killed at its time limit with no result is masked
    (`windows_s3.rerun_verdict`; V8) and replaced while masks stay under a fifth; the record keeps every flipped test
    (`reruns.flips`). The flaky-test rule counts flips per test function (subtests of one flaky test are one test), at
    most 3 functions, none a milestone's, each excluded with its flip count. TestInterplexCache starts its gRPC server in
    a goroutine without waiting for it (fails every cold single run, 4 of 20 warm), a non-milestone test in all four
    k8sgpt tasks.
  - **A fix check the host did not run is not evidence** (2026-10-27, `generators.debug.fix_check`, `Debug._executed`):
    a suite at B that timed out or ran under a masked sandbox marks each fix's evidence `infra`. That evidence is never
    cached, and `subset` says the check did not run (retry on a quieter host), not "not a real fix". Under load, a
    held-out rebuild's suite at B timed out at 900 s for all eight fixes, five of which had built the task before. The
    cached failure would have answered every retry.
  - **A grade the host timed out is graded once more** (2026-10-27, `windows.timed_out_empty`, `WindowTask.grade`): a
    run the sandbox killed at its limit with no test reported is no verdict on the tree. The grade runs once more, and a
    second such timeout is the tree's (a hang, a fail, never masked, so a hang cannot buy a mask). A held-out DeepSeek
    episode was graded all-fail on such a run under load. Its regrade passed three of four milestones in 500 s.
  - **S10 names what flipped** (2026-10-27): `FlakeReport.false_negative_detail` counts the failing checks and the
    grader's failing tests over the false negatives (`graded_from` keeps a GradeReport's failing tests), and the S10
    evidence carries it. A held-out task failed S10 on 1 false negative in 59, and its record could not say which
    milestone or test flipped. `runs` lists each false negative (its failing checks, how many tests failed, the first
    five): the task's rerun had 12 in 60, 10 failing everything (the host killed the whole suite under load) and 2
    failing only the regression check, and the totals alone could not tell the two apart.
  - **S3 masks a rerun the host failed or whose test binary died** (2026-10-27, `windows_s3.rerun_verdict`,
    `run_level`, `golang.parse_report`). A rerun is masked and replaced (V8) when:
    - its sandbox never started or was killed from outside (`infra == "masked"`);
    - the Go profile names go test's own timeout, an OOM or a kill;
    - a whole package's passing tests (at least half, and 10) failed at once with no test-level cause.

    A panic is the panicking test's flip only, so the per-function flaky rule applies to it. Masks stay under a fifth
    of the reruns, so a package that keeps dying still fails the bound. An unclean rerun records every differing test
    (`n_differs`; the flip counts read all of them, not the first 10) and its output tail; `masked_why` lists the mask
    reasons. Two kept tasks failed S3 on one rerun in 60 each: one sandbox did not start (exit 125), and one root
    package failed as a whole with infra ok.
  - **Gate records name the tree they graded** (`window_a`); a rebuild moves the old record aside, and the gates and the
    seal ignore a record of another tree.
  - **Agent-caused tool errors** (2026-10-26, from mvdan-sh dc951aeb-e1's budget stops, 13-14 failed calls each): an
    `old_string` the file does not hold, a regex that does not compile and a path that does not exist are
    `apps.base.PolicyToolError` (error `tool_error`, so no format penalty in credit shaping; fault `policy`, never read
    as a World fault). The workspace edit's refusal lists the file's closest lines with their numbers, and an unknown
    tool name (`shell`, `execcmd`) gets the nearest valid tool and the full list. Messages only: no alias, nothing runs.
  - **A renamed test's old copy is a stale visible test** (2026-10-27, `generators.debug.stale_tests`). The build's
    stale check (A' tests that pass in A' and fail on B's source) looked only at milestone tests. A fix that renames
    its test leaves the old copy in A' under the old name, graded by nothing. Now every ungraded test of a test file the
    fixes touched (not a milestone test, not p2p) counts too, and is stripped from A'. Grading is unchanged: the grader
    writes B's test files back. The case: a held-out Go task kept the old copy, which asserts the fault's behaviour,
    while the renamed hidden test asserts the fix's. The only frontier full pass was lost to `path:protected` for
    editing it, so the task read `unfair`. `_locate_defs` keeps a key B does not define (the old name lives only in
    A'); it dropped it, and the second build round refused the window for the same stale test.
  - **Output texts a hidden check looks for go in the report** (2026-10-27, `generators.debug.asserted_messages`,
    `milestone_messages`, `unstated_messages`). The string argument of a containment or error-text assertion on a
    fix's added test lines is a fact the user must state. Patterns: Go testify `Contains`/`ErrorContains`/`EqualError`,
    `strings.Contains`, JS `toContain`/`toThrow`/`toMatch`, pytest `match=`, `assertIn`, Rust `should_panic(expected=)`,
    `.contains(...)`. A regex, a format or a text with a refused name is left out. `write-prompts` gives each
    milestone's texts to the writer (`messages`), whose brief says to quote them word for word. Its lint adds
    `unstated_message` for a requirement that does not quote one. `author` is unchanged, so earlier reports keep
    their verdicts. The case: a held-out Go task's milestone asserts two error-body texts. The writer had been told
    never to quote the notes, the report named only the status code, and no model passed that milestone. The items of
    a literal list count too when a loop checks each one by containment (`for _, want := range []string{...}` then
    `strings.Contains(out, want)`; Python and JS loops alike). A second held-out milestone checked five structured-log
    keys this way; its report named none, and the student and DeepSeek failed it in all six runs. A text the start
    tree's source already holds still counts: the first case's texts were used by other handlers. So does a sentence a
    `want`/`expected` variable or field holds (`want := "..."`, a table row's `expectedErrors: []string{...}`), whose
    comparison is often outside the hunk. The list's opener may be a kept line when the fix adds only a row. A word, a
    format verb or an escape is left out. Two held-out analyzer milestones compared exact finding texts no
    report stated, and every student and frontier run failed them.
  - **The report covers every case the hidden tests check** (2026-10-27, `swarm_prompts.WRITER_BRIEF`,
    `REVIEW_SYSTEM`). The writer is told that an input variant or edge case a test or table row names (a line ending
    such as CRLF, an empty or null value, a stale status, a boundary) must appear in its requirement as the user sees
    it. The reviewer blocks a report that leaves one out and names the case. A held-out milestone's table checked a
    CRLF row its report never mentioned: DeepSeek and three student runs failed only that row. The write state has
    one retry: a writer that answers in prose ends its session (the harness reads it as done) without `prompt.json`;
    two held-out runs paused there ("missing declared output 'prompt'") and now get one fresh session.

## 8. Layer 4: verification and isolation

Every gate declares its kind (M7 R6): `gate`, a rule the instruction states (its failure zeroes the reward and the
attempt's per-turn credit); `outcome`, a correctness, consumer, incident or perf check; `score`, a graded bar. There is no
default; a graded report with an undeclared gate raises, naming it.

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
6. Perf grading runs only when the node is quiet (load guard) and records the spread across rounds; rounds contaminated on the original's side are re-graded, not scored (Appendix B.5). Noise on the candidate's side alone only slows it down, so a failing best-of-k there is a fail (M7 audit: a slow, jittery candidate was masked).
7. **Trusted channels only (P26).** Candidate code runs unprivileged in the verifier too, with privileges dropped *before* any candidate code (its `setup.py`, its import side effects) runs. The test harness, the benchmark and the reference build are root-owned copies made before the candidate runs. Verdicts come from artifacts compared by a trusted process, never from exit codes or the candidate's stdout. Timing is taken outside the candidate's process. Every timed repetition gets fresh inputs, and every candidate invocation a fresh HOME and TMPDIR (no memoization, no replay of cached results). Consumers come before the candidate on `PYTHONPATH`. Any grader exception masks the grade: no `reward.json`/`reward.txt`, and `crucible_reward.json` with `reward: null` (M7 audit: this line still described the old pre-written 0).

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
| Per-uid firewall (M1 item 5, 2026-10-12) | root sets iptables owner-match rules for the policy uid (Karotte): in allowlist worlds only localhost, the sandbox's own addresses and the allowlisted hosts' IPs; in every world the localhost ports of App servers and `blocked_ports` are closed to it; `firewall: required` refuses sandboxes that cannot enforce them (no CAP_NET_ADMIN, gVisor) | an allowlist world whose provider has no egress proxy; calling an App's server directly to bypass the tool gate, its ACL and RBAC |
| Text | instructions, simulated colleagues' replies and CI output are leak-scrubbed (Repo2RLEnv patterns + ours); actors' citations are kept in the trace but hidden from the policy | PR links, fix commits, "fixed in 1.2.3" in a reply |
| Hidden CI | checks live in a separate runner sandbox, run unprivileged by default, re-staged every run, leftovers killed; the policy gets `name: PASS/FAIL` only, under a run budget that is world state | reading, overfitting or rewriting the hidden tests |
| Trusted git | verifier-side git reads only a root-owned copy of `.git` taken before the policy ran; refs are validated and quoted | git-ref shell injection; hooks, filters, fsmonitor, external diff or `info/exclude` planted to run code as root or hide files |
| Fresh account, no escalation | a new uid (never an image account), `PR_SET_NO_NEW_PRIVS` on the policy tree, root's `PATH` and interpreter audited for policy-writable components | the image's uid-1000 user with sudo, setuid-root binaries, a planted `git` on root's `PATH` |
| Episode hygiene | quiesce before restore; restore through directory fds, owned by the policy; `.git` reset; `/tmp`, `/dev/shm`, home, SysV IPC emptied; App-owned state root `0600` | a symlinked parent redirecting a root write; stashing state for the next episode; reading other people's mail from the database file |
| Egress self-test, answer sources | 17 code hosts, registries and mirrors (GitHub incl. `codeload`/`raw`, GitLab, PyPI incl. `files.pythonhosted.org`, npm, crates, Go proxy, conda, Hugging Face, `web.archive.org`, …) probed as the policy uid with a TLS/SNI handshake (a transparent egress proxy accepts every TCP connect); allowlisted hosts are excluded and checked as working lanes; audit `network.answer_sources` fails any allowlist that opens one of them (added 2026-10-09) | `pip download` of the fixed release through an allowlisted index; a mirror or archive after GitHub is blocked |
| Answer-key probes (P27) | every policy tool call is scanned for history, registry, code-host, grader-file, root and `/proc` probes; an exposure masks the episode (`crucible:answer_exposed`) | a pass reached by reading the published fix |

**Built in M1 build item 1 (2026-10-12): a separate verifier World for every World task (`WorldSpec.verifier`, `world/verifier.py`, `world/custody.py`).** Layer 4 previously held only for the coding kit (patch graded in a verifier image). Any World can now set `verifier.mode: separate`. The harness then quiesces and snapshots the policy's World, checks custody of the snapshot from the store, closes the policy's World, restores the snapshot into a fresh World built from the same spec (an optional verifier backend or image, no responders, no interceptors) and grades there. The fingerprints must match. Tested by `tests/unit/test_verifier_separate.py`.

| Mechanism | Attack it defeats |
|---|---|
| Grading in a restored copy, after the policy's World is closed | a process left behind that rewrites the answer during grading (shown to beat the in-place grader on the local backend) |
| Artifact custody over the snapshot's tar (`artifacts: ["<app>:<glob>"]`): regular files and directories only, symlinks only with `allow_symlinks` and only inside the tree, no hard links, FIFOs, sockets or devices, size, count and depth limits | a symlinked answer to a host file, a FIFO that blocks the grader, an oversize or deeply nested hand-in |
| Protected paths (`protected: ["<app>:<glob>"]`) must equal the episode-start snapshot: no change, no removal, no new entry | rewriting or deleting a hidden test; planting a `conftest.py` next to it |
| Artifacts extracted into the attempt dir with root-only modes, `O_EXCL`/`O_NOFOLLOW` and an openat walk, with a Harbor-style `manifest.json` | a hand-off that writes outside its directory |
| Shared state roots refused: two worlds on one host whose App state resolves to one absolute path | separation in name only |

A custody violation, like an unkillable process at quiesce, is `crucible:policy_misbehavior`: scored 0, never masked, and the grader never runs.

**Built in M7 (2026-10-09): the grader-side trusted channels (layer 7), `crucible.verify.runtime`, standard library only, staged read-only into any verifier image and run as root with `--network none`. Tested by `tests/redteam/test_grader_attacks.py`: the honest solution scores 1, 11 cheat mutants score 0 at their intended gate, and 6 of them are shown to beat a seed-style grader (exit-code verdicts, candidate-first `PYTHONPATH`, one shared HOME, no sealing):**

| Mechanism | Attack it defeats |
|---|---|
| Patch applied and every candidate command run as a fresh uid (new session, `PR_SET_NO_NEW_PRIVS`, no groups) before its first instruction | `setup.py`/CMake/import side effects as root; writing the benchmark, references or reward files |
| Fresh HOME/TMPDIR per invocation, removed afterwards; `/dev/shm` and SysV IPC of the uid cleared | memoizing or replaying results across timed repetitions (`memoize_in_home`) |
| Cohort killed **the moment the main process exits** (not after the pipe-drain grace), outputs sealed root-owned read-only before comparison | a detached child rewriting outputs after the run (`background_fixup`; found by the control: a 1 s grace was enough) |
| Python runs `-I` with grader paths first and the candidate **after the standard library** | shadowing the stdlib (`json.py`), the consumer or the test runner (`pytest/`) |
| Test verdicts from a grader-opened results fd: a finish record, the expected count or exact node ids (collected against the original), every test passed, no skips; exit code ignored. Plugin autoload off, outcomes taken from the call's exception before any other hook, `unittest` assertions snapshotted (M7 audit) | `os._exit(0)` under pytest, `pytest.skip` from library code, fake "4 passed" on stdout, a `pytest11` entry point rewriting reports, a library replacing `assertEqual` |
| Static scan of the pruned tree that gets built: macro expansion, token pasting, dynamic imports, native binaries under any suffix, runner references | banned constructs behind `#define`, `##`, `importlib.import_module("cty" + "pes")`, skipped directories |
| Hidden material under `/root` (0700) or root-only copies; references computed at grade time into 0700 dirs | delegating to the original, reading or rewriting hidden tests |
| Perf: load guard, round-spread check (original contaminated → infra, re-graded), wall time from the grader's clock (`wall_ratio_max`, record mode included, or K/5K differential runs with a positive, plausible per-step time), timings positive and finite, end-state sanity | timers forged in-process; zero or negative timings; start-up that shrinks with the step count; scoring another user's GPU job as a perf failure |
| Every reward file removed before grading (host and container, and rLLM's agent-writable `/tmp/rllm/reward.json`); any grader exception → masked `grader_exception`, no reward file; candidate crashes while judged, candidate content that breaks `setup`, and a policy-broken schema → fail, never masked; a certain hard-gate fail decides the reward even when another check is infra | turning a 0 into a masked grade by crashing the grader, a FIFO in the tree, a dropped table or a jittery run; a stale or agent-written reward file read as this grade's |

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

**Multi-step tasks (M1 build item 4, 2026-10-12; `TaskStep`, `NativeHarness._run_steps`; ported from Harbor multi-step trials).** A task may declare `steps`, which all run in one World and one trace. Each step:

1. runs an optional `setup(world)` (new records arrive, for example);
2. opens with its instruction (`World.begin_task_step`, which marks the trace with a `task_step` event);
3. gives the policy a segment of the episode, with the same policy objects (and so the same context) unless `fresh_context` is set;
4. is graded by its own grader, in place or in a separate verifier World (the policy's World stays open between steps and closes only for the last one);
5. passes or fails its `min_reward` gate. A float gates `reward`; a dict gates keys of the flat reward, and a missing key counts as below. A step below its gate ends the task.

The reward (`step_reward`) is one of:
- `mean` over every declared step, the default. A step never reached scores 0, so stopping early cannot raise the mean. Harbor's own mean excludes steps that never ran.
- `final`: the last declared step.
- `mean_executed`: Harbor-compatible.

Run records carry `task_steps` (per-step reward, gate and abort reason) and `step/<name>` gate bits.

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
- **Every failure has a blame** (`infra`, `harness`, `task`, `policy`, `unknown`). Only `policy` failures are scored (reward 0). All others are masked: reward `None`, excluded from mean, pass@k and coverage, never read as 0. A masked sample is retried (never a policy failure) up to `RetryRule`'s limit; a verification-stage mask only as a re-grade of the same work (M10 lab audit R-3).
- **Stop rules:** budget (SQLite reserve/settle ledger; a crashed reservation becomes *uncertain* spend), wall clock, masked share > 20%, unknown-blame share > 5%. Running out of credits stops the run as resumable, not as failed samples. An outage of a dependency (a provider endpoint, a host, or a (task, harness) pair) opens its breaker (M10 lab audit R-5): its samples are deferred, one probe runs after a cool-down, and a key still failing after T is quarantined (journaled, its samples left owed) while the run continues; an infra-masked attempt on an open key is never terminal. On a resume the samples an outage left masked-terminal before the verification phase (a retryable infra failure at the admission, seed or agent stage) are dispatched again with a fresh retry budget, and the stop rules count only the samples that end in the current run window (M10 lab audit R-5).
- **Boundaries normalize foreign errors** (Harbor exception names, MiMo categories, provider failures) into the taxonomy. A verifier `infra` dict is classified into `node_contention`, `stale_image`, `stale_build` or `verifier_error`.
- **Workspace harness:** the patch is collected with the tolerant hook. An empty patch from a changed tree is `crucible:empty_collection`; the workspace is preserved and the retry *re-collects only*, without re-running the agent.
- **Docker grading** (`verify/stage.py`) pins the verifier image digest (mismatch → masked `stale image`, before anything runs), names and labels the container, and turns a timeout into `docker rm -f` plus masked `verifier_timeout`. `runtime/gpu.py` preflights PID 1 (no init → `crucible:zombie_wait`) and the GPU arch.
- **Control channels** run with a sentinel; a signal death without it is `crucible:harness_self_kill`, and a lint rejects self-matching full-command-line `pkill`/`pgrep` patterns.
- **Every B.5 class has an injector** and a test showing it is masked, plus policy controls (agent broke the environment, step limit, context overflow, no change) showing they are scored.
- **Retries reuse what is still valid.** A grade masked for infrastructure is retried as a re-grade of the stored patch, and an empty collection as a re-collect of the preserved tree; neither re-runs the agent, which would replace the measured solution with a new sample. `Runner.regrade` re-grades exhausted samples later from their artifacts. Correction (M10 lab audit R-3, 2026-10-17): until then only the workspace harness honoured this; the native harness's retry of a masked grade was a full agent re-run (hidden by deterministic oracles), so a policy whose answer hung the tests escaped its 0. Now `RetryRule` retries a verification mask only for persisted work: a native separate-verifier grade is re-graded from its persisted, quiesced hand-off (`verifier/handoff_ref.json`), an in-place grade is re-graded inside the attempt before its World closes and the attempt then ends masked, and a failure the answer can cause (a verifier timeout, a checker outage, a non-finite reward) is graded again at once on the same work. One that repeats (or whose first grade ran out the scoring budget: flagged `not_repeated`, and its control must grade within that budget) is decided in the attempt, whatever `RetryRule` allows, by the task's control (`grade_control`: the oracle, a reference answer, graded by the same grader under its own budget): control passes → `crucible:policy_error` / `grader_failed_on_answer`, scored 0 (the hang or the checker failure is the answer's); control fails → a task-blamed mask; no control → a mask with flag `no_control`. Every masked failure carries `evidence = {probe, control, status}`. `PerfGuard` runs one perf grade per host and only on a quiet host (no foreign GPU memory, low utilization and load); a host that never quiets yields `node_contention` before anything runs.
- **Trajectories join.** Every model call has a `call_id` and the ordered list of actions it produced. The ATIF trajectory (Harbor's models) carries `call_id`, `crucible_seq` (the trace event) and per-call `Metrics` with token ids and logprobs; a call that produced no action (a blank reply the policy nudged past) gets its own step. `steps.jsonl` rows get `atif_step_id` and `call_id` from the M5 steps' seqs. A proxy capture log can be cross-checked against the trajectory (`crucible:capture_incomplete` on mismatch).
- **Harbor task directories: two modes (corrected 2026-10-10, m13 audit G1).**
  - **Patch mode** (`verify_dir.HarborDirTask`) is valid only for Crucible-authored tasks, whose verifier kit applies
    `/logs/artifacts/model.patch`.
  - Foreign Harbor tasks grade the container state the agent left. Through patch mode an oracle and a no-op got the
    same grade, a silent wrong reward.
  - **In-place mode** (`interop/harbor/inplace.HarborInPlaceTask`):
    - the policy acts in the task container through `harbor/terminal` (`exec`, `read_file`, `write_file`);
    - Harbor's own `OracleAgent` runs `solve.sh`;
    - Harbor's own `Verifier` grades the same container under Harbor's phase network plan;
    - `reward.json` resolves to `reward`, else the single key, else `reward_key`, else it masks as ambiguous;
    - a missing reward, a timeout or an unapplicable network phase masks, never scores 0.
  - **Fixed alongside:** `HarborBackend` now passes Harbor's `/logs` bind mounts (`Trial._agent_env_mounts`). Without
    them every verifier redirect into `/logs` failed. On a remote daemon, `/logs` lives in the container and is
    downloaded instead.
  - **Tests** (`tests/unit/test_harbor_inplace.py`, local Docker): oracle 1.0, no-op 0.0, a scripted policy through
    the tools 1.0, multi-key with and without `reward_key`, no reward → masked.
  - **A-3:**
    - `HarborBackend` passes the World's network policy and any phase policies to Harbor's `DockerEnvironment` when
      egress control is needed. Before, allowlist worlds could not run on Docker.
    - The overlay's blanket `network_mode: none` steps aside in that case, and the backend fails closed if the daemon
      cannot enforce the policy.
    - In-place tasks declare their agent and verifier phase policies, so a verifier network phase is applied
      dynamically (the fixture grades 1.0) instead of masking.
    - Allowlist test: `pypi.org` reachable, `example.com` blocked.
  - **Real data:** terminal-bench 2.1 `adaptive-rejection-sampler` grades oracle 1.0 and no-op 0.0 through the in-place
    path (`docs/modules/reports/m13_tb21_inplace.json`).
- **Patch mode details.** `interop/harbor/verify_dir.HarborDirTask` grades a patch with the task's own `[verifier].command` in a fresh container of the pinned task image, with Harbor's layout (`/tests`, `/logs/artifacts`, `/logs/verifier`), declared network, limits and GPU, on any Docker host (the lab over `DOCKER_HOST=ssh://`). It can pin the task to a git ref, so a historical run is graded by the grader it was graded with. `ImportedHarness` turns a recorded run's patch into an attempt, so historical runs and dataset submissions pass through the same journal, classification and re-grading as live ones.
- **Every attempt writes `rollout.json`** (`crucible.rollout/v1`, schema in `core/schema/rollout.py`): per call, token ids, behaviour logprobs, policy version (stamped by `LLMClient.policy_version` at capture time), sampling, and the call's credit; per rollout, reward (None when masked), gates, truncation, failure and blame. In `rollout_mode="train"` a rollout without complete token records is masked, never guessed. Blank usage in tolerant mode books a worst-case estimate as uncertain spend instead of $0.

**As built, M11 part 1 (2026-10-10; research `docs/research/m11-m12/05`).**
- `interop/canonical.py`: canonical JSON, derived `call_id`s, one tool naming across runtimes.
- `interop/replay.py`: runtime-independent grades and dense channels.
  - Every runtime's transcript (World trace, Harbor ATIF, NeMo-Gym Responses items, OpenEnv `CallToolAction`s, agentenv gateway events) is normalized to canonical actions.
  - The actions are replayed in a fresh World and graded through the single reward producer.
  - Progress is measured online over the task's subgoals or its plan's checks.
- Exports:
  - **NeMo-Gym resources server** (`interop/nemo_gym/server.py`): grades the live World session from its state, once per session; an infra fault in the World or a grading exception masks. Masked grades go out as `mask_sample` in both NeMo-RL locations (NeMo-RL still counts their 0.0 in group baselines: the M12 bridge fills them). Every route's reply validates against NeMo-Gym's own pydantic models in its Python 3.13 env (M11 audit; before it, only the unmasked verify body did).
  - **OpenEnv `Environment`** (`interop/openenv/export.py`): on OpenEnv's own base and MCP types. Per-step rewards Ψ-telescope to the final reward, with Ψ₋₁ = 0 because a plan's invariants can hold at the start. A masked grade yields reward None.
  - **verifiers taskset** (`interop/verifiers/taskset.py`): subclasses verifiers' `HarborTask`/`HarborTaskset`. A masked grade raises a typed `TaskError`, which prime-rl drops; components are recorded as `crucible/` metrics.
- **Framework environments:** `crucible/.envs/{openenv,verifiers,nemo}`. verifiers pins mcp 2, agentenv and OpenEnv mcp < 2, and NeMo-Gym Python ≥ 3.13.14. Each path runs in its own environment, and the gate compares the outputs.
- **agentenv plugin** (`interop/agentenv/plugin/`, registered by entry points):
  - GPU flags (`--runtime=nvidia --init`) as an opt-in on agentenv's builtin local provider (`gpu_docker.local_gpu`). **Corrected 2026-10-10 (m13 audit C1):**
    - the first version registered a `gpu_docker` provider whose sandboxes agentenv's type guard always destroyed (they
      reported `type="local"`);
    - a distinct type would break agentenv's `sandbox_type == "local"` URL externalization and CA trust;
  - a `crucible_world` step: the World behind a real agentenv gateway, published as a `DeployedGatewayEnv` with its env
    card and MCP URL. **Corrected (C2):**
    - the first version omitted both, so stock steps such as `modify_env_tool_access` and `deploy_agent` failed;
    - a stock `modify_env_tool_access` step is now tested against it;
    - the gateway bind host is `gateway_bind`;
  - a `crucible_verify` step: the single reward producer, unclamped; steps from replaying the gateway's own trajectory; a masked grade records skipped rows and raises; **Fixed (C3):** boolean gate results (agentenv aggregators test `is True`), replay of only this episode's slice of the gateway trajectory, the router's signed `__actor` argument stripped, the replay task rebuilt with its kwargs.
  - a `crucible_ref` artifact type.
- **MiMo round trip** (`interop/mimo/{verify,export}.py`):
  - The import records a sha256 manifest of the source env.
  - Each env's own `verify()` grades a World run in a fresh grading root: the source's verifier files plus the run's `system/*.db` and `workspace/`.
  - A crash or an unavailable judge is masked (MiMo's own contract), never 0. Items map to `check/<id>` channels.
  - MiMo's verifier output is captured and scrubbed: it prints a judge-key prefix.
  - The export emits the original bytes and checks them against the manifest.
  - **Gate ✅:** one env imported, run, graded (masked without a judge; 0.5 with a cheap judge for a placeholder answer) and re-exported byte-identical. The re-export copies the source bytes: it proves the import left the source untouched, not a World round trip (M11 audit; a World → MiMo writer is open).
- **Canonical episode key** `"<task id>|<episode>"`: derived `call_id`s agree across runtimes.
- **Gate (toy) ✅** (`tests/unit/test_interop_gate.py`): the oracle trajectory gives byte-identical `reward.json` and `steps.jsonl` through native, Harbor ATIF, agentenv (real gateway REST transport), NeMo-Gym and OpenEnv, with the same final state fingerprint. verifiers keeps the reward and raises a typed error on a mask. What each leg proves (M11 audit): the Harbor and OpenEnv legs compare replays of the same oracle actions, so they test the transcript formats, not the runtimes; the NeMo-Gym leg drives Crucible's server with a test client (NeMo-Gym's own environment server is not run); the verifiers leg runs verifiers' classes on a stub runtime (the plugin now loads through verifiers' own loader); no leg covers task02.

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
- **OpenEnv's own environments (2026-10-10):** grid_world, connect4, maze and wildfire (a stochastic fire simulation) run
  through the generic importer with native rewards and termination. A mid-episode restore of the simulation replays
  deterministically.
  - **Bug found on the way:** OpenEnv's base `reset_async` / `step_async` forward `seed` / `episode_id` / `timeout_s` to
    the sync method, which crashed every environment declaring a bare `reset()`.
  - **Fix:** the importer calls an async method only when the class overrides it, and passes only the keyword
    arguments the target's signature accepts.
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
- Adapters, each checked by tests that run the pinned upstream code (verl 5eb4826, TRL 72cf37d, SkyRL 3d0ae22, rLLM
  4e828c8, verifiers ae819de; M12 audit: none of the four bridges ran against its upstream before):
  - **verl** (`verl_plugin.py`, `verl_agent_loop.py`, `verl_bridge.py`): a `verl.plugins` entry point registers the
    `crucible_delta` estimator and the `crucible_agent` loop on `import verl` in every Ray actor (registration in the
    driver alone never reached the trainer actor). Turn credit travels in `rm_scores`, the only per-token field verl
    v1's advantage path reads: credit on each call's sampled tokens, the row summing to R (verl's convention, so a row
    ending on tool tokens keeps its R). `CrucibleAgentLoop` runs one World episode per session through verl's
    `LLMServerClient.generate`; with `turn_weight` a uid's sessions meet for M6's group credit. Masked sessions are
    placeholder rows with no trainable token. v1 trainer only; `use_kl_in_reward` and truncated sampling are refused.
  - **TRL** (`trl_bridge.py`): one episode per TRL row (TRL already repeats prompts), the sampled token's logprob from
    TRL's top-k lists, one batched `generate` per round on one thread (ranks in lockstep where `generate` is a
    collective), and masked episodes as placeholder rows (`env_mask=[0]`, reward None → NaN) so rewards never shift onto
    another completion.
  - **SkyRL** (`skyrl_bridge.py`): every `GeneratorOutput` key, one row per input or step-wise rows with
    `trajectory_ids` / `is_last_step`, stop reasons from truncation; masked rollouts never enter a group as 0.
  - **rLLM** (`rllm_bridge.py`): `Step.from_model_output` with `weight_version`; a scalar advantage from
    `Trajectory.reward` for its verl backend, per-token advantages for tinker.
- `flatten` refuses token-incomplete records (missing logprobs were zero-filled, i.e. trained as π_rollout = 1).

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
  The honest status table is in `docs/modules/M12-train.md` §9: 24 done, 23 partial, 5 missing (updated as items close). #34 then closed: a π_θold correction with no source refuses rollouts that span several versions. #40 closed: asynchronous reward. A `grading` pool status (untrained, not selectable). A late outcome is counted but stays pending; a hard timeout or a crashing grader masks the rollout; a late auxiliary credit channel falls back to outcome-only advantage with `credit_fallback`. #37 closed: top-p sampling records each token's nucleus (`Turn.candidates`), and the trainer renormalizes π_θ over it; a record sampled with top_p < 1 but without candidates is invalid for training; the vLLM HTTP engine refuses top_p < 1 until it can report candidates. #36 closed: MoE routing replay (R3). A CPU MoE policy whose full-bf16 engine copy flips about 11% of router choices; engines record the experts per token; `routing_replay="r3"` replays them (refusing incomplete coverage) and more than halves the logprob gap on flipped tokens; routing confidence Σ_k is logged. New engine numerics `bf16_full` (bf16 weights and activations), the realistic inference gap. #50 closed: prefix sharing. A GRPO group's prompt is encoded once (GRU state, or packing with restarted positions and a 4D float mask for transformers); loss and gradients equal the unpacked step within 1e-6. #7 closed: offload. Trainer state goes to CPU or NVMe during rollouts and comes back with Adam moments intact (bitwise on CPU); on MPS at most 5% of the training peak stays resident; KV modes are validated per engine. No checklist item is missing now. Then the monitor and test gaps closed (#5, #6, #13, #20, #24, #27, #51, #52), leaving 38 done and 14 partial. One bug was found on the way: evictions made at a barrier that then blocked were never reported.

**As built, part 7 (M12 critic mode, 2026-10-10; `train/critic.py`; checklist #16, #46–#48).**
- **Value models:** `TinyValue` and `HFValue` (an HF backbone plus a zero-initialized scalar head; `freeze_attention`
  trains only the MLP/MoE projections, as in SAO).
- **Advantages:** `TorchTrainer(critic=...)` replaces group-relative advantages with token-level GAE per rollout:
  - the chain holds completion tokens only (skip-observation);
  - compaction segments share the outcome, scaled by (γλ)^{N>s};
  - truncated rollouts bootstrap from V(s_T);
  - λ_policy = 1 − 1/(1.5·l), λ_critic = 1.
- **Critic training:** TTUR, K critic steps per policy step; explained variance logged.
- **Result:** with n = 1, group-relative advantages are identically zero (GRPO cannot move, tested), while critic mode
  learns (reward 0.04 → 0.21, explained variance 0 → 0.80).
- **Under DP:** the critic trains on the full batch on every rank.

**As built, part 8 (M12 state plane, 2026-10-10; `train/state.py`; checklist #1).**
- **Checkpoint:** taken at the sample barrier (`checkpoint_every`). It captures trainer and critic weights with their
  optimizer moments, the published version, scheduler state, source queues, the rollout seed counter, every pool tag and
  group, the records of finished-untrained rollouts, and the monitor and report counters.
- **Writes:** atomic (temp dir, then rename) with a sha256 manifest; a corrupt file is refused.
- **Resume:** `Runtime.resume()` restores everything and re-publishes the restored weights at the restored version
  (`publisher.restore`: no new version, no optimizer reset). In-flight rollouts restart and are counted.
- **Tests:** synchronous mode resumes **bit for bit** (4 uninterrupted steps = 2 + checkpoint + resume + 2, with and
  without the critic); partial rollout continues the version sequence and never draws a prompt twice.

**Lab full-stack campaign (2026-10-10; `crucible_pack_toy.labstack`).** Every built module (M1–M12) runs on the lab
RTX 5090 with a small locally served model, on the same environments as the CPU gates.
- **Setup:** vLLM serves Qwen3-4B-Instruct with tool calling. A lab-only config maps every model alias (policy,
  simulator, judge, smoke) to it, with `request_defaults` asking vLLM for logprobs and token ids on every call.
- **Phases:**
  - the M1–M9 smoke;
  - the M2 actor gate;
  - the M7 golden check;
  - M8 calibration and world synthesis;
  - the M9 accept gate;
  - M10: `Runner` samples in eval and train mode (train mode needs complete token records);
  - M11: the native cross-runtime gate, plus replay identity of an LLM-produced trajectory;
  - the long-horizon gate;
  - the full unit suite on Linux + CUDA;
  - the M12 GRPO gate.
- **Report:** one JSON report.
- **Gap it exposed:** a provider had no way to request token capture from a self-hosted engine. Fixed with
  `LLMProvider.request_defaults`, merged into every request with call-level `extra_body` keys winning.
- **Run 3 findings (fixed):**
  - **Exact TITO on Qwen3:** the chat template gives only the *last* assistant turn an empty `<think></think>`, so the
    old suffix (render with and without a dummy assistant turn, take the difference) was wrong once a turn stopped being
    last. `HFChatCodec.suffix` now renders a sentinel assistant turn followed by the new messages and takes the tokens
    after the EOS that closes the sentinel. A multi-turn test checks it against a full render on a Qwen3-style template.
  - **Pipeline bubbles:** `EnginePool` counted time spent paused (weight publish) as busy. Busy time now accrues only
    while the pool is active and not paused.
  - **Offload on CUDA:** `TorchTrainer.offload` left 64 MiB of cuBLAS workspace resident; it now clears cuBLAS
    workspaces before `empty_cache`.
  - **Phase order:** smoke, M8 and M9 read the M3 trace store, which the container doesn't carry. A new
    `m3_toy_ingest` phase ingests the toy ticket logs first, twice, and checks idempotency.
  - **Unit suite in the container:** docker-marked tests skip when no daemon is reachable (`tests/conftest.py`);
    data-dependent tests skip when their data isn't staged; git is installed in the container.
  - **Trusted as root:** the demoted candidate uid could not traverse pytest's 0700 temp root, so every candidate write
    failed and looked like a policy error. `Trusted.run(candidate=True)` now refuses (InfraError) a work dir whose
    ancestors lack o+x, instead of producing spurious agent failures. The check runs only when a candidate actually
    executes demoted: run 4 had it in the constructor and broke every grader that never runs a candidate under a
    `mkdtemp` dir.
- **Run 4 findings (fixed):**
  - **Gradient checkpointing was silently off.** `from_pretrained` returns eval mode and HF recomputes layers only in
    train mode, so `HFPolicy(gradient_checkpointing=True)` kept every layer's activations. Once exact TITO let real
    multi-turn segments through, the first 8k-token step of Qwen3-0.6B OOMed the 32 GB card (27 GB allocated).
    `HFPolicy` now enables train mode with every dropout at zero (modules, config, attention). The forward stays
    deterministic and equal to eval mode. A test counts each layer running again in backward.
  - **Lab config:** carries the repo config's `[gpu_hosts]` tables, so the sandbox tests resolve `lab5090`.
- **Run 5 findings (fixed):**
  - **One malformed completion aborted a forge plan.** Qwen3-4B returned unparseable JSON to the LLM world-synthesis
    Test writer, and the exception escaped `Orchestrator._run_node`. `complete_json` now raises `LLMOutputError` (a
    content failure, `failure_kind="malformed_output"`, and it says when the output was cut off at `max_tokens`). The
    orchestrator counts it, and a pydantic `ValidationError`, as a content attempt and passes the parse error to the
    next attempt as feedback. After `max_attempts` the node fails closed (`rejected`) instead of crashing the plan.
  - **The M9 accept gate hung for an hour: lost child exits.** The gate sat idle in `epoll_wait` with a zombie `mkdir`
    child. `/proc` showed it still held the zombie's pidfd, but that pidfd was not in the loop's epoll set.
    - **Cause:** Python 3.12 on Linux watches children with pidfds via `add_reader`. If any fd is closed while
      registered with the loop, the selector keeps a stale key. A later pidfd that reuses the number then only
      replaces the key's data (`selectors.modify` makes no syscall when the events are unchanged), so the exit is
      never seen.
    - **Reproduced** in the lab container: register a pipe, close it behind asyncio's back, run `LocalExec` — it hangs.
    - **Fix:** the World exec path (`world/exec._run_subprocess`, used by local and CLI-based backends) no longer
      depends on the child watcher. It runs a blocking `Popen.communicate` on a wide dedicated thread pool, where
      `waitpid` cannot lose an exit. A cancelled await kills the command's session instead of orphaning it.
    - **Tests:** the stale-registration scenario completes (it hung before, on Linux), and timeout and cancel kill the
      session.
  - **A10 rejected on noise.** Every task Qwen3-4B failed twice was rejected on n=2, although 0/2 has a Wilson
    interval of [0, 0.66], which overlaps the train band [0.1, 0.7]. A10 is now sequential:
    - a point estimate inside the band, or an interval clear of it, decides at once;
    - otherwise the check tops up with more frontier runs, doubling, up to `Policy.a10_max_runs` (8), through
      `calibrate(k=…)`, which resumes the same journaled run;
    - a rejection at the cap says the interval still overlapped.
- **Run 6 findings (fixed):**
  - **Every M10 train-mode rollout was masked as `capture_incomplete`.** A train-valid call needs a `policy_version`,
    and an `LLMClient` pointed at an external vLLM never had one. The reason string only said "token record
    incomplete".
    - `LLMProvider.policy_version` declares the served checkpoint's version (`LLMClient` starts from it; a trainer
      bridge still overrides it per update), and the lab config sets it to 0.
    - `Turn.missing()` names every gap (prompt ids, logprobs, policy_version, prefix_break, spans, candidates), and
      `valid_for_training` reports it.
    - The M10 phase now counts train-valid rollouts with the trainer's own test.
    - Checked offline: Qwen3-4B-Instruct's template does preserve the sampled tool-call prefix across chat-API turns.
  - **A10 top-ups errored.** The runner refuses to resume a run whose spec changed, and `k` is part of it. The toy
    task's `calibrate(k)` now runs only the missing samples as a new run (`…-n<have>`) and returns the latest `k`.
  - **Root-only test assumptions:**
    - a demoted candidate run reports `trusted=True` as root;
    - the empty-collection injection relies on `chmod 000`, which root reads anyway, so it skips as root.
  - **M12 GRPO on the lab passes:** 20 steps, 20 publishes, 0 masked, `stop_reason=steps`, 13.2 GB peak (OOM at 27 GB
    before the checkpointing fix), Qwen3-0.6B on the long-horizon backlog worlds.
- **Run 7 finding (structural, fixed): chat-API capture is not training-grade.** With the version stamp in place,
  M10's train rollouts still failed, now named precisely: `prefix_break` from the second call on. A new lab phase,
  `tito_drift`, decodes the first divergence:
  - Qwen3-4B sampled `open.\n\n<tool_call>`, where `.\n\n` is one token;
  - vLLM's hermes parser kept `\n\n` in `content`, and the chat template adds its own `\n` before `<tool_call>`;
  - so every re-rendered prompt read `open.\n\n\n<tool_call>`: 23 of 24 calls broke.

  The parser + template round trip is lossy by construction, and patching one whitespace case would only move the
  drift. The fix applies M12's design to every LLM consumer:
  - **`LLMProvider.tito`:** `LLMClient` routes through `train.world_env.EngineLLM` over the engine's token-id
    completions endpoint (`VLLMServerEngine.generate`). It renders with the model's own template
    (`template_kwargs`), extends each served conversation token for token (`prev prompt + completion + suffix(new)`),
    parses tool calls locally, and stamps the provider's `policy_version`.
  - **`EngineLLM`** keeps a bounded table of served conversations, so one client serves many concurrent episodes
    (Runner `concurrency`, k samples).
  - **Lab config:** only the policy alias uses `tito` (it produces training data); simulators, judges and forge roles
    keep the chat API with JSON mode.
  - **Tests** (`test_llm_tito.py`): two interleaved conversations through one client extend exactly, and the same
    history re-rendered chat-style does not, which reproduces the lab bug on Qwen3's template.
  - **Limit:** local tool-call parsing is the hermes `<tool_call>` format (the Qwen family); other formats need
    their parser in the codec.
- **Run 8: every phase passes its own verdict.**
  - **Capture:** `tito_drift` shows 0 breaks in 15 calls; M10 eval and train each score 4/4 with 4/4 train-valid
    rollouts.
  - **M11:** LLM replay identity holds.
  - **M12 GRPO:** 20/20 steps, 13.2 GB peak; 2 transient `infra:ReadError`s from vLLM were masked, not scored.
    Runtime failure examples now fall back to `repr` (httpx errors stringify to "").
  - **Unit suite:** 333 passed / 32 skipped on Linux + CUDA as root.
  - **M9:** 5/11 accepted. All other checks pass on all 11 tasks; A10 rejects the 6 tasks Qwen3-4B (the frontier
    stand-in) solves too rarely, after sequential top-ups.
- **False passes in the campaign:**
    - run 4's M10 phase exited 0 with every rollout masked; the phase now reports `pass` only when both modes score
      and train mode yields token-complete rollouts;
    - M9's verdict with a 4B model is expected to stay below 11/11: A10 asks whether the *frontier* model lands in
      the band, and the small stand-in mostly does not. Every other check (A1–A9, A11–A13, power tests 13/13 proven)
      passes on all 11 tasks.
  - **Tests:**
    - the CLI seed test skips when the seeds' Harbor side is not staged;
    - the offload test checks onload against parameters plus optimizer state, not the pre-offload peak, which also held
      the gradients and cuBLAS's workspace.

**C7 (2026-10-10): any A2A agent as a harness.** `interop/agentenv/harness.py:A2AHarness` runs an agentenv A2A agent
(served per attempt from a factory, or at a URL) against the task's World. The World must be served over MCP.
- **mcp-config:** registers the World's MCP URL with `AgentEnv-Role: <actor>`.
- **agent-config:** passes the agent's own fields. `${capture.base_url}` and `${capture.key}` are replaced by the
  capture proxy and a per-attempt session key; the upstream key stays on the proxy.
- **Prompt:** the task instruction as an A2A message. The agent is the policy, so its own `agent_error` is scored
  (blame `policy`), while `infra_error`, `canceled` and `rejected` are masked (blame `infra`).
- **Evidence:** the trajectory/v1 native trajectory is saved, and the capture log becomes the actor's model calls.
- **Token records:** a call gets one only when the upstream returned prompt and completion token ids with logprobs
  (vLLM style), stamped with the harness's `policy_version`. A train rollout without complete records is masked
  `capture_incomplete` by the record builder.
- **Grading:** the task's own, as in the native harness.
- **Finding:** without `policy_version`, a train rollout is masked even when the token ids are complete. The builder
  rightly refuses a token record that cannot say which checkpoint produced it.
- **Tests:** a scripted A2A agent on the toy helpdesk behind a real agentenv gateway makes one model call through the
  capture proxy (fake upstream) and the gold tool calls over MCP:
  - with token ids, the train rollout is complete (turn tokens and logprobs present) and graded 1.0, and the proxy
    log holds no upstream key;
  - with a hosted upstream it is masked `capture_incomplete`;
  - an agent-reported failure is `agent_run_error` with blame `policy`;
  - token-record parsing (ids or `token_id:N`).
- **Real agent (2026-10-10):** `interop/agentenv/agents.py:LLMToolAgent` is a minimal function-calling A2A agent.
  - It lists the tools of the first registered MCP server and runs an OpenAI-compatible chat loop over them through
    the capture proxy. `llm_api_key` is write-only.
  - It caps tool output, stops on a reply with no tool call or at `max_turns`, and returns Usage plus its native
    trajectory.
  - `FibWorldTask` (toy pack) is a coding World: an agentenv workspace app with a shell and git, graded by running
    `python3 fib.py`.
  - Live through OpenRouter, GPT-6 Luna and Claude Sonnet 5.5 each solved it (reward 1.0, about $0.01 in total).
  - `tests/unit/test_llm_tool_agent.py`: a scripted upstream (write_file → shell → answer). The eval rollout scores
    1.0 in 3 turns; the train rollout is masked `capture_incomplete` (a hosted upstream has no token ids).
- **Pending:** snapshot/v1 mid-episode restore.

**C9 (2026-10-10): agentenv env images as Apps.** App template `agentenv/mcp_env`
(`interop/agentenv/mcp_env.py`, which DEPENDs on `agentenv_protocol`'s data-plane client) runs an agentenv-protocol env
image as one loopback-published container per World.
- **Tools:** they come from the MCP interface the card names. Every tool the card advertises must be served (a
  drifted image fails at start). Served but uncarded tools are reported, and hidden with `card_tools_only`.
- **State (adapter `agentenv_data`):** the fingerprint and snapshot are `data/get`'s parts as canonical JSON.
  Restore is `data/reset` + `data/add` of those parts, verified by fingerprint. The env must advertise all three
  operations.
- **Seed:** `seed` parts or `seed_file`, loaded at start.
- **Not covered:** the legacy `base_mcp` images (`slack_mcp`, `email_mcp`) have no v1 card or data plane. Their
  `/api/reset` can only reload a mock file, not restore a snapshot, so they are not loadable as snapshot-capable Apps.
- **Tests (agentenv's own v1 test env, built from its sources):**
  - card tools are served and `list_items` is reported as uncarded;
  - the seed loads, and reset → mutate → snapshot → mutate → restore gives equal fingerprints; reset returns to the
    seed;
  - agentenv's core-protocol probe sees `data/get` and `data/add` supported;
  - the container is removed at close, and `card_tools_only` hides uncarded tools.

**M14 (2026-10-10): the RLM agent.** The user asked for an RLM agent "that manages and improves the tasks, especially
for very long horizon", after studying Prime Agent. The study is in `docs/research/m14-rlm/01`, the design in `02`, the
module doc in `docs/modules/M14-rlm.md`, and the code in `src/crucible/actors/rlm/`.
- **Kernel.**
  - A stdlib-only REPL process, with an NDJSON protocol in the pattern of Prime's v3.
  - **A per-cell deadline**, and a wedged kernel is killed (Prime has neither).
  - Every effect goes through one host bridge.
  - **Exact restore by replaying the journal.** Cells are re-executed with their recorded host replies, the namespace
    digest is checked, and a divergence raises. Prime pickles the namespace with dill (best effort) and loses its
    factory state.
  - Runs locally with a credential-free environment, or inside the World's sandbox (`docker exec -i`).
- **`RLMPolicy`.**
  - Prompt-as-variable, for real: `TASK`, `TOOLS`, `OBS`, `INBOX` live in the kernel. Output is capped at 8192
    characters.
  - **Each `world.call` is one World step**, through a coroutine bridge between `act()` and the model loop.
  - `rlm.query/batch` sub-LLMs are recorded as `kind="sub"` model calls with `parent`.
  - `rlm.finish`, a goal continuation, and compaction with Prime's kernel note plus the live names.
  - Stall signatures from Continual Harness, and timeout restore.
  - A mid-cell checkpoint resumes through `checkpoint.resume` → `policy.recover(world)`, which finds the result of the
    already-stepped call in the trace.
  - `run_episode` now honors a policy's own `too_many_errors` stop.
- **Training-record finding.** `runtime.record.build` treated every call of an actor as one linear token chain, so an
  RLM sub-call (a fresh prompt) was a "prefix break" and masked the whole train rollout. Sub-calls are now separate tree
  nodes that share their parent turn's advantage (`build`, `full_prompts`).
- **Continual harness.**
  - Prime's five typed kinds, with whole-state versions and `rollback_to`.
  - **Edits are kept only after a same-seed A/B** (primary metric with a margin, guards with a tolerance). Prime stores
    `expectedOutcome` and never checks it.
  - A ranked digest, usage counters, and retirement of entries that never help.
  - Base-prompt edits and skills that do not compile are refused.
- **Task manager.**
  - `RLMAgent` (proxy objects with `help()`, `rlm.spawn` with a depth limit, a budget wrap-up).
  - `ForgeWorkspace`: versioned tasks; the verifier is M9 and reports its evidence; commit needs a passing
    verification; the pack supplies the vocabulary and the policy documents.
  - `TaskFSM`: Planner → Implementer (child RLM) → Reviewer (a different model, given the check meanings and the rule
    semantics) → objective Verifier. It loops back with findings, and its state is journaled and resumable.
  - `RLMTaskManager` drives it through its `forge` proxy.
- **Tests:** kernel 3, policy 8, manager 6. They cover the B.15 repair with a loop-back, a crash mid-cell → the same
  final fingerprint, a train rollout complete with sub-calls, and the kernel in a no-network container.
- **Soundness finding (live).** A candidate that dropped checks passed A16, because the property audit is built from
  the task's own checks. Now `ForgeWorkspace.verify` sets the committed version's checks as an **audit floor**
  (`ToyForgedTask.audit_checks`), so a candidate can only strengthen its audit.
- **A7 in the default verifier set** (A1, A7, A16). In the last live run a candidate that dropped every scope check
  passed A1 and A16: the hacker never makes collateral edits, so A16 cannot see a missing scope gate even with the audit
  floor. A7 ("every stated property is graded") fails it objectively. Before this, only the reviewer caught it.
- **Vocabulary conformance.** `add_check` now rejects kinds, rules and spec keys the pack grammar does not understand
  (the implementer had invented fields that the grader silently ignored), and rejects duplicate ids.
- **Live smoke (about $0.20).**
  - The policy works mechanically: 0 on the toy helpdesk, the same as native function calling with GPT-6 Luna.
  - Six live FSM runs exposed and fixed: reasoning budgets, reviewer error vs. rejection, a lossy `show`, evidence-free
    findings, opaque proxies, the vocabulary's actor-id rule, a reviewer without grounding, and an implementer out of
    turns.
  - **Seventh run:** verification and review rejected all three candidates for real reasons (dropped scope checks, the
    hack still open, scope invariants that rejected the oracle). The FSM works; the remaining weak link is the
    implementer model (GPT-6 Luna over-edits instead of making the one-check fix).
- **Stronger models (about $0.46 more).**
  - Implementer Claude Sonnet 5.5 (alias `implementer-strong`), reviewer GPT-6.1 Sol (`reviewer-strong`, a different
    family). Every candidate in both runs passed A1, A7 and A16; the implementer is no longer the weak link.
  - Run 1 (GPT-6 Luna reviewer): the reviewer rejected the candidates over ambiguities in the vocabulary. The text now
    says the rule only examines in-episode calls, that the supervisor only answers the agent, and that `approver` is an
    actor id.
  - Run 2 (Sol reviewer): it rejected all three candidates. Two of its reasons were real:
    - **`approval_grounded` matched approvals by amount alone.** An approval asked for another order at the same amount,
      or one approval reused for a second refund, grounded the refund. Now an approval is bound to the request it
      answers (the latest earlier message to the approver must name the refund's order or ticket id) and is consumed.
      Every oracle already names both. Test: `test_approval_is_bound_to_its_request_and_used_once`.
    - **The reviewer could not tell a final-state check from a delta check**, because it did not know the seed. The
      vocabulary's `start_facts` (scalar SQL) are evaluated on the seeded start state and shown in `forge.show` and the
      reviewer input as `start_state`.

**M14 (2026-10-10): the RLM harness for RL trajectories.** The user asked for an RLM-based harness in which context
for long-horizon agent trajectories is maintained by the RLM. `runtime/harness/rlm.py:RLMHarness` puts the policy's model
inside the RLM scaffold:
- **Policies:** `PolicyRef.make` may return a bare LLM client per actor.
- **Harness id:** `RLMConfig` is hashed into `rlm:<digest>@1`, so the compaction mode, budgets and output cap separate
  GRPO groups.
- **Rollouts:** `train.world_env.rlm_world_rollout` is the M12 rollout (a `make_policy` hook on `world_rollout`).
- **Context lives in the kernel.** At each compaction, the turns leaving the prompt are archived verbatim into
  `HISTORY`, next to `OBS` and `INBOX`, and the prompt points at them.
- **Training records now match `LLMToolPolicy`.** RLM calls carry `step` and linear capture (`prompt_delta`), and an
  engine re-render starts a segment. Previously a thinking-template re-render after a CONTINUE, a stall notice or an
  inbox notice was a prefix break, which masked the rollout.
- **No compaction with nothing to drop.** A compaction before the first turn archived an empty segment and summarized
  nothing.
- **`aclose()`** ends the driver and the kernel after every episode. Before, a step-limit stop left the kernel process
  alive.
- **Evidence:** `rlm.<actor>.json` holds the config, counters, segments, notes and the kernel journal.
- **Engine:** `VLLMServerEngine` takes `api_key`, which the TITO path passes from the provider, for a keyed cloud
  endpoint. `scripts/modal_vllm.py` serves vLLM on Modal with processed logprobs behind a bearer key.
- **Live (2026-10-11, Modal L4, vLLM 0.11 serving Qwen3-4B-Instruct-2507 over TITO, 2 train rollouts, under $0.20):**
  - Both rollouts were train-valid: real engine token ids and logprobs, `policy_version` stamped, 10–12 delta-chained
    turns, and no prefix breaks.
  - Reward was 0. The 4B model never made a World call: it searched the host's working directory for `refund_policy.md`
    and messaged the supervisor.
  - That surfaced two harness defects, both fixed:
    - **The kernel ran on the host.** The kernel is now a `DockerLauncher` container (no network, read-only root, all
      capabilities dropped, uid 65534, memory and pid limits), the `RLMConfig.kernel` default. `LocalLauncher` starts
      in an empty directory and is for trusted code only.
    - **The model never read `TOOLS`.** A one-line-per-tool index now follows the task (`tool_index`).
  - Also found: vLLM 0.11 breaks on transformers 5 (pinned in `scripts/modal_vllm.py`).
- **Tests:** `tests/unit/test_rlm_harness.py` (4; the docker test checks the sandbox from inside the kernel).

**M13 bench and M15 studio (2026-10-11).**

The user asked for both, after a study of every local framework (`docs/research/bench-studio/01-framework-study.md`).

What the study found:
- **Missing everywhere.** No checkout implements progress AUC, first irrecoverable step, recovery rate, paired
  significance tests, per-step heatmaps, 2PL IRT, MinHash/n-gram decontamination, per-model temporal cutoffs, or DPO
  pairs from agent traces. Those are built here.
- **Errors scored as 0.** Harbor, verl, rLLM, SWE-smith, EnterpriseOps-Gym and NeMo-RL score infra errors as 0;
  NeMo-Gym alone masks correctly, and Crucible follows it.

**M13 (`src/crucible/bench/`):**
- every §11.1 metric from an M10 run, plus pass^k and max@k;
- task-level comparison (exact McNemar over task flips, t interval and sign-flip test on per-task deltas; NeMo-Gym's
  Welch interval is unpaired, and per-sample pairs pseudo-replicate);
- validity: oracle and no-op controls (or M9 / import evidence) exclude broken and vacuous tasks; exposures score 0;
- lineage-keyed HMAC splits with a seal;
- 2PL MAP IRT with fixed-anchor calibration (the gdpval anchored-BT idea, item-level) and drift quarantine;
- contamination: canary, 13-gram, code substring, MinHash LSH, tree fingerprint, leak regex;
- dynamic benchmarks: HMAC item seeds with commit–reveal and a validation chain;
- a deterministic single-file HTML report.

**M15 (`src/crucible/studio/`, `studio/datasets.py`, `cli/`):**
- `crucible label`: pack or run → TRL-canonical SFT (with prime-rl `message_loss_mask` and Φ weights), DPO (best vs
  worst; correction pivots) and KTO, with a closed curation grammar;
- `crucible build`: ingest → forge → M9 model-free checks → M11 canonical export → M13 bench;
- `pack.yaml` (`PackManifest`), `crucible pack new` (a conflict-safe scaffold) and `pack check` (static + model-free
  semantic, severity policy);
- a lazy click CLI;
- the README quickstart.

**Findings:**
- the toy forge read a hard-coded M3 store path (`mine_signals(store)` now);
- forged World tasks are not Harbor-native directories, so build's export is the canonical replay;
- ATIF tool names needed `openai_name` in dataset rows;
- the toy logs' correction pivots were all fixed by non-policy actors.

**Tests:** `test_bench.py` (4), `test_studio.py` (3). The module docs are `docs/modules/M13-bench.md` and
`M15-studio.md`.

**M13–M15 completed, paid models and GPUs, and an improvement study (2026-10-11).**

**M14 packs:**
- `pack.yaml` ingest now takes any registered connector (`init`, `scope`, `discover`, `redaction_env`).
- Hooks `fixtures` and `model` were added; `pack check --level fixture` grades designed cases exactly; `requires` gates
  only the semantic level.
- The **coding pack** gets a manifest: the recorded runs and PRs as ingest sources, credit for recorded runs, and CPU
  fixtures over the dense reward on all four seeds.
- **`finance_ap`** is a real accounts-payable World (3-way match, fraud controls, a per-world tolerance, a controls gate,
  Φ per invoice).

**M15 studio:**
- `crucible model` and `crucible export` (canonical, ATIF validated by Harbor, NeMo-Gym, OpenEnv);
- the agentenv explorer plugin.

**M13 with real models:**
- Toy: Sonnet 5.5, Sol 6.1 and Qwen3-235B, $2.15. Sonnet under the RLM harness solved the 12-ticket backlog.
- Coding: from recorded frontier runs, $0.

**Live engine and run findings:**
- **SGLang** (first live run, Modal L4, 0.5.21): the real server rejected nothing once prompts were token lists. But
  released weights were not restored on resume without a publish, so greedy output changed. The engine now reloads the
  last weights. Generate, abort, barrier and weight load all pass.
- **Cost attribution:** concurrent attempts sharing a client were charged each other's spend. Cost now comes from the
  attempt's own calls.

**M12:**
- The GPU gate gained a learnable task (`WordFileTask`) and held-out eval before and after training.
- The trainer logs grad norm and lr.

**Recorded runs:** the coding runs' per-step Φ cannot be replayed (python and `sed` edits), so they report end-state Φ
only.

**Improvement study.** One per module group, in `docs/research/improvements/` (index `00-index.md`). Correctness problems
it found, now queued:
- a candidate can turn a perf failure into a masked grade (M7);
- dense credit ignores gates, with live evidence on finance_ap (M6);
- judges read agent text unfenced;
- circular M4 reliability (addressed in the M4 audit: verdicts use only evidence the mapping rule did not);
- uncontained harness exceptions;
- broken layering with no CI;
- the `SandboxPool` loop;
- uncapped RLM sub-calls;
- an id-only bench seal.

**C8 (2026-10-10): the full trigger language.** `EventSpec` now expresses the rest of urn:agentenv:triggers/v1, and
Crucible's step-mode scheduler runs it with agentenv's semantics. The pure helpers are ported from the gateway into
`core/triggers.py` (Apache-2.0); core may not import agentenv.
- **Language:**
  - `where` predicates (regex, equals, exists over `args.*`, `result.*` or bare argument names);
  - `${args.*}` / `${result.*}` / `${binding}` templating of tool-action args, where an optional `?` drops the key and a
    missing required value fails the action;
  - `state` triggers: check pipelines (tool, args, extract with match/take, bind, predicate), evaluated after every
    policy tool call, firing once;
  - `after_event` + `offset` anchoring (the mark is the anchor's firing instant plus the offset);
  - `every` as a duration or `{dist: exp, mean}` with a required `seed`. The k-th draw is recomputed from the seed,
    and `PendingEvent.draws` records how many were used, so arrivals reproduce after any restore;
  - `until`, `count`, per-action `verify` (logged as `event_verify`), and `barrier: provoking_call`.
- **Semantics change:** an errored tool call provokes no `on_tool` event, as in agentenv.
- **Hash neutrality:** the new fields default to `None` (`where`, `PendingEvent.draws`). Canonical JSON drops `None`,
  so existing specs and WorldRefs keep their content hashes. The suite's frozen helpdesk hash caught a `{}` default.
- **Barrier in step mode:** on_tool events always run before the step's observation, so the barrier holds by
  construction there; in wall mode it is delegated.
- **Translation:** `AgentEnvBackend._trigger` maps every feature to the gateway's JSON (tool names, check steps, ISO
  durations, absolute `until`). `World.firing_log()` gives id, due instant and type for each firing.
- **Tests:**
  - helper parity;
  - in step mode: `where`, templating with `${result.id}`, `verify`, the barrier, a state trigger with a binding that
    fires once, an anchored follow-up exactly 10 min after its anchor, and about 24 seeded arrivals in 2 h that stop
    at `until`, are exponential, and replay identically after a restore at minute 40;
  - every spec is accepted by agentenv's own `TriggerEngine._normalize_trigger`;
  - live, the same acknowledgement event on a real agentenv gateway produces the same inbox as step mode, with
    `fired` and `verify_ok` in the gateway's log.
- **Open:** the full 20-spec firing-log conformance across clocks. Wall-mode instants depend on the 1 s driver poll,
  so time triggers are compared by spec acceptance and marks, not by live instants.

**C5 (2026-10-10): agentenv sandbox adapter breadth.**
- **Provider building:** each provider is built through agentenv's registry, `from_config` with the
  `[sandbox.providers.<name>]` config plus our overrides: Modal's GPU, and Sail's `auto_sleep_min_idle_seconds` from
  `SandboxSpec.auto_sleep_idle_s` for human waits.
- **Fallback chains:** `provider: "modal,e2b"` is a chain. The first provider that can enforce the network policy and
  serves the World wins; the others' errors are kept in `chain_errors`.
- **Create path:** `create_container` when the provider has its own (the image runs as the container); otherwise
  `create_sandbox` (the image is the sandbox).
- **Attribution and lifetime:** attribution is `{world, …SandboxSpec.attribution}` (run_id, sample_key, harness).
  `timeout_s` is validated in [60 s, 24 h], so long-horizon Worlds can go above 2 h.
- **Tests:**
  - with fakes, "modal,e2b": modal raises quota-exceeded and e2b serves through `create_sandbox`, with the full tags
    and an 8 h timeout;
  - Sail gets its auto-sleep override; an all-broken chain raises; a GPU World refuses non-Modal providers;
    `timeout_s` above 24 h is rejected.
- **Pending:** the Sail 30-minute idle sleep and resume needs Sail credentials.

**S4 (2026-10-10): the `ssh_exec` backend.** For Docker-less GPU hosts: a Runpod pod is itself a container, with no
user namespaces, so no container runtime can run there.
- **`world/exec.py:SSHExec`:** one multiplexed SSH connection (`ControlMaster`/`ControlPersist`), files by `scp`, the
  in-host deadline, and `ulimit` rlimits (nproc, as_mb, fsize_mb, cpu_s, nofile). It cannot switch users.
- **`world/backends/ssh.py:SSHExecBackend`** (registry key `ssh_exec`):
  - each World gets its own directory under `~/crucible_worlds`, removed at teardown;
  - the host needs python3 and git for the state adapters;
  - the capability reports `isolation="none"`, `gpus=True`.
- **Refusals:** a World that gives the policy a shell, unless `allow_untrusted_shell` is set; and a `no-network`
  World, which host processes cannot enforce.
- **Tests:** against a throwaway sshd container with its own key and known_hosts:
  - mailbox tool calls and a snapshot/restore round trip work, and the world directory is removed afterwards;
  - `as_mb=200` turns a 400 MB allocation into `MemoryError`, and `nproc=64` holds;
  - both refusals fire.
- **Pending:** the Runpod acceptance (sm_89 preflight, verifier score equals a manual run) needs a running pod. The
  recorded pod refused connections on 2026-10-10.

**S4b (2026-10-19): the `apptainer` backend** (`world/backends/apptainer.py`; HPC hosts without Docker, first Marlowe:
`docs/research/swarm-pipeline/05-marlowe.md`).
- **`ApptainerSandbox`:** the `DockerSandbox` contract over one unprivileged instance per World
  (`--containall --cleanenv --no-home --net --network none --writable-tmpfs`, the workspace bound at its host path,
  `--nv` for GPUs). `ApptainerExec` runs `apptainer exec --cleanenv instance://...` (an exec without `--cleanenv` carries the
  host environment in); files travel on stdin and back as base64. `ExecSandbox` over it is slime's four-method `Sandbox`;
  a vanished instance raises `ExecError`, which becomes `SandboxTerminatedError` (V8).
- **Measured isolation (login node):** loopback only, a public connect fails; no host home, code, venv or `/cm/shared`;
  no scheduler or credential variables; own PID namespace; `instance stop` ends what the policy left. Not achieved:
  cgroup limits (refused on the login node, so `cgroups="auto"` starts without them and says so; the Slurm job's cgroup
  bounds the World) and a demoted `policy_user` (commands run as the invoking user; there is no root to demote from).
- **Also:** `ApptainerBackend` (registry key `apptainer`), `open_workspace_world` (the window lane's World with the tools
  in the instance), `run_once` (a fresh container for grading). `APPTAINER_CONFIGDIR` moves instance state off a full home.
- **Tests:** `tests/unit/test_marlowe_apptainer.py` (fake runner: flags, cgroup fallback, files, the slime contract, a dead
  instance, the stage driver's guards and TERM forwarding); a real rollout through it on the Marlowe login node.

**S5 (2026-10-10): the gVisor tier.** `SandboxSpec.runtime` takes three values:
- `runc` (the default);
- `runsc`, when a daemon has gVisor registered (`--runtime=runsc`);
- `runsc-in-container`, which needs no daemon change (`world/sandbox.py:GVisorSandbox`).

How `runsc-in-container` works:
- **Toolbox:** the pinned toolbox image (`docker/gvisor`: release 20261005.0, sha512-verified; the tarball now
  carries `runsc` plus `gvisor-bin/` sidecars) is mounted read-only with `--volumes-from` into the World's image,
  which runs as a privileged outer container.
- **Sandbox:** one long-lived gVisor sandbox runs inside it, with `--platform=systrap`, `--network=none` for
  no-network worlds, `--overlay2=none --file-access=shared` and the outer root as its root. Every command goes
  through `runsc exec` (`GVisorExec`; user names are resolved to uid:gid), and files go through the outer container.
- **Spec fixes found on the way:**
  - gVisor's own `/tmp` tmpfs hid pushed files, so the outer `/tmp` is bind-mounted;
  - `runsc spec`'s minimal capabilities made root unable to `chown`, so the spec gets Docker's default set plus
    `SYS_RESOURCE`;
  - runsc state, the bundle and `/opt/gvisor` are masked with empty tmpfs.
- **Scope:** the outer container keeps Docker's pids, memory and cpu limits. There are no GPUs (nvproxy is not
  configured) and no pooling. The capability reports `isolation="gvisor"`.
- **Tests (Mac):**
  - a hardened agentenv World runs under gVisor: the dmesg banner and `uname` show it, the policy is demoted, App
    state is unreadable, and the egress probe is isolated;
  - a snapshot / restore round trip works.
- **Lab (amd64):** the toolbox builds and `runsc do` shows the banner.
- **Incident:** a cleanup on the lab ran `docker container prune`, which acts machine-wide. Docker's event log
  shows it removed only our own toolbox container. Machine-wide prunes are now banned in the lab notes.

**S6 (2026-10-10): pause / resume.** `World.pause()` / `resume()` free the CPU while the trainer holds the GPU.
- **Backend:** a backend with `capabilities.pausable` freezes the World's processes. `DockerSandbox` uses
  `docker pause` (the cgroup freezer), and so do pooled sandboxes and the agentenv backend with a docker sandbox.
- **Clock:** a wall clock freezes at pause and continues from the frozen time, with no jump. Step clocks never move by
  themselves. On resume, the agentenv gateway's clock is reset to the frozen time.
- **Tests:**
  - a wall clock at 3600× shows no drift over a 0.2 s pause;
  - a busy sandbox drops to 0.0% CPU while paused, keeps its files, and its process continues after resume;
  - a sandboxed agentenv World with a 600× wall clock, paused for 3 s, keeps its state fingerprints, its gateway time
    stays within 5 virtual minutes of the frozen time, and the World works afterwards.

**S10 (2026-10-10): the sandbox warm pool.** `world/pool.py` keeps `SandboxSpec.pool` pre-started containers per
launch signature: image plus every hardening flag (network, pids, GPUs, cpus, memory, capabilities). It refills in the
background, and `open_sandbox` returns a `PooledSandbox` when `pool > 0`.
- **Fresh mode (default):** a released container is destroyed and replaced, so no rollout inherits another's
  processes, files or the guard's policy user.
- **Reuse mode (`pool_reuse`):** hygiene runs in Python inside the container. It kills every process except PID 1,
  the keeper (the earliest-started child of PID 1: `sleep <ttl>` or the renewable life loop, and its children; M10 lab audit
  R-6) and its own exec session's tree; empties `/tmp`, `/var/tmp`, `/dev/shm` and the
  workdir; removes added users and homes; and clears System V IPC. A verify step must then report
  `procs=0 files=0 users=0`; otherwise the container is destroyed.
- **Shell-hygiene finding:** a first shell version killed its own `docker exec` wrapper (exit 137), and its verify
  step counted its own processes. The session-tree rule fixes both.
- **Pool lifecycle (M10 lab audit R-6):** pooled containers belong to the run, never to an attempt. They carry run-scoped
  labels (`crucible.run`, `crucible.lease`, owner, expiry; no `crucible.attempt`), run the renewable life loop and are
  renewed on hand-out (and by the borrowing attempt while it lives); one whose renewal fails is dead and replaced. Pools
  are keyed by (the run's lease, signature) (a run id is unique only within a root), and every `Runner.run` and `regrade`
  closes its run's pools, warm and handed out, before the run's lease is released (`close_pools(owner=lease)`), in bounded
  time: a refill still in `docker run` is not waited for past `POOL_CLOSE_WAIT_S` and removes its own container later.
- **Measured on the Mac:** cold `docker run` p50 0.16 s vs warm acquire p50 about 20 µs.
- **Tests:**
  - the warm acquire is under 10% of cold, fresh mode never repeats a container, and the refill keeps up;
  - in reuse mode, a planted process, file and user are all gone, the keeper survives the hygiene, and a container that
    cannot be verified clean is destroyed;
  - `open_sandbox` shares one pool per signature.

**G-13 / G-14 (2026-10-10): the Judge and family specialists.**
- **`forge/judge.py:select`** ranks only M9-accepted variants. Its output is an order, so no verdict can change; a
  property test over 200 random pools checks this.
  - Quality: L2's closeness to the middle of the train band, plus small terms for graded gates and composition depth.
  - Diversity: MMR against the picks so far (same operator or seed costs 0.5), and picks that would break the
    YieldBook quota are skipped. This follows GitSwarm, where a selector gained and a synthesizer lost.
- **`forge/specialists.py`:**
  - `FamilyMemory` (jsonl) keeps the strategies of confirmed A16 hacks per family. Strategies are kept, not actions,
    which are task-specific.
  - `SpecialistHacker` first replays the known strategies over the new task's solver trajectories, at zero hacker
    episodes. Only when the replay has nothing to try does it call the expensive hacker.
- **Tests:**
  - the Judge never changes verdicts, prefers a diverse pick over a near-duplicate, and respects the quota;
  - on B.15-weakened t2 the specialist spends one fallback episode and finds `skip_interactions`; remembered, it
    re-finds the hack on t4 with 0 episodes.

**Perturb and G-tighten (2026-10-10).**
- **`forge/perturb.py`:** a perturbation changes the task (World or start state), not the solution. Kinds:
  distractor, decoy, policy_amendment, clock_event, injected_bug.
  - Cosmetic: the unperturbed oracle must still pass.
  - Semantic: the oracle is re-derived (A1 again), the instruction gets the new property sentence (A7), and the
    unperturbed oracle must fail, at the perturbation's gate when named.
  - **A18 perturbation effect** decides this, and catches mislabels in both directions. NA reason `not_perturbed`.
- **Toy (`crucible_pack_toy.forge.perturbed`):**
  - a distractor ticket (cosmetic): the World's fingerprint changes, the old oracle passes;
  - the FIN-7a amendment, "duplicate-charge refunds need approval whatever the amount" (semantic): the re-derived
    oracle asks the supervisor and records the approver, a new invariant grades it, and the old oracle fails at
    `policy`.
- **`forge/tighten.py`:** a tightened bar must be fair and separating.
  - Fair: the oracle keeps A1's 15% headroom. `calibrate.py` still said 10%; that is fixed.
  - Separating: the solution that solved the task before fails it.
  - `tightest_bar` = oracle / (1 − 0.15), rounded up. `add_gate` handles categorical gates (concurrency, sanitizer
    tools): the oracle passes every run (≥ 3), and the frontier solution fails one.
- **Tests:**
  - A1 and A18 pass on both toy perturbations, and both mislabels are caught;
  - the tighten fairness, separation and "not stricter" refusals.

**Forge GPU run (2026-10-10): two structural findings, built into G-span.** The span grades on the lab gave 21/23.
The failures were not kit bugs; they were derivation bugs.
- **Closure under the grader's dependencies.** The KinFu-LS oracle failed `consumers`. The pristine `kinfu_replay`
  consumer links both forks (`find_package(PCL COMPONENTS gpu_kinfu gpu_kinfu_large_scale)`), while each KinFu span
  left its fork sibling unported, so no solution could pass.
  - The KinFu span's grades were masked in this run, but it has the same defect.
  - `derive.py:closure_violations` reads every consumer's required components and flags a span that leaves one
    unported (CPU acceptance row `closure_consumers`).
  - The fork pair becomes one span, `kinfu_family`, with a new grader scope in `verify/task04.py` (both roots, both
    workload sets, both perf bars).
- **Alternatives must share the span start's outside state.** Astra's people/NCV ports failed `build`: they depend
  on Astra's own shared-helper changes (`texture_binder.hpp`, `initialization.h`), not the oracle's helpers that the
  span start carries. `alternative_admissible` excludes an alternative whose changes outside the span differ from the
  oracle's, recorded under `excluded_alternatives`. This also removes Astra's KinFu "slower than the bars" mutant,
  which could never have reached `perf`.
- **Re-derivation:** people and ncv are byte-identical (their 17 verdicts stand). The spans are now people, ncv and
  kinfu_family, all CPU-ok. The 3-span stitch still reproduces v3 (56/56).
- **Lab runner:** keeps verdicts of unchanged spans, drops masked grades and unstaged spans, and waits for the task03
  runner. The 4 `kinfu_family` grades are queued. (2026-10-11: all 17 grades pass on the lab RTX 5090; the kinfu_family solution regraded clean at 1.0.)

**G-11 (2026-10-10): sandboxed roles and the sandboxed hacker.** `interop/harbor/role_runner.py` runs a forge role
as a Harbor installed agent inside a wrapper task. It DEPENDs on Harbor and PORTs the structure of
`analyze/checker.py` (Apache-2.0, credited in THIRD_PARTY.md).
- **Wrapper layout:**
  - `environment/` is exactly the role's visible files: a prebuilt image with no Dockerfile, which Harbor uploads to
    `/app`;
  - forbidden strings (zone canaries, grader material) refuse the wrapper before it runs;
  - the instruction holds the role prompt, the file tree and "`/app/<role>-out.json` is your only deliverable",
    with the JSON schema;
  - a stdlib `tests/validate.py` checks the required keys and top-level types (reward 1 means well-formed, not
    correct), and the full pydantic parse happens on our side.
- **Network:** the environment baseline and the verifier are `no-network`; the agent phase is an allowlist of the
  model API.
- **`HarborRole`:** the orchestrator `Role` adapter. Staged inputs become `inputs/*`. An invalid deliverable raises
  `LLMOutputError` (retried with feedback); a trial failure raises `RoleInfraError`; spend is accumulated.
- **`forge/hacker.py:SandboxedHacker`:** the LLM hacker behind `propose(view)`, which A16 now awaits. Its sandbox
  holds the agent-visible files and the solver trajectories. Trajectories carry what a policy saw in earlier
  episodes; that cross-episode memory is the B.15 threat, so the hacker may see it. Canaries are forbidden.
- **Finding:** a TCP connect from inside the allowlisted sandbox to `example.com:443` succeeds. The egress filter
  accepts the connection and drops the request. Network isolation must therefore be measured at the request level.
  The e2e test does exactly that: `https://example.com` is blocked and the model API is reached.
- **Tests:**
  - the wrapper holds only the visible files, and Harbor's own `Task` parses its network policy;
  - leak and path-escape refusals;
  - the validator on good, wrong-type, missing-key, non-JSON and absent deliverables;
  - the `HarborRole` failure mapping.
  - With Docker:
    - a real Harbor job (the oracle agent in place of an LLM) sees exactly the visible files and reaches only the
      model API;
    - a sandboxed hacker's proposal against B.15-weakened t2 is graded by A16 as one confirmed hack.

**G-12 (2026-10-10): the Curator.** `forge/curator.py`.
- **Dedup:** the `forge/dedup.py` layers run against every kept task.
- **Placement from the ladder:** an inversion rejects (`fix_grader`); `bench` when the frontier's upper bound is
  ≤ 0.2; `train` when L2 is in [0.1, 0.7]; otherwise `hold`, with the ladder's steer.
- **Lineage sealing:** every task carries its seeds, and a composite carries all of its components' seeds. A
  lineage's first placement seals it to that split; a task whose lineage is sealed to the other split is held
  (`lineage_sealed`). Bench material therefore never reaches training through a sibling or a composite.
- **`recompose`:** a task whose training L2 rises above 0.7 is retired and becomes a seed. `compose(task, round)`
  builds one more composition and measures it; the Curator places it, up to `max_rounds`, until a composite lands in
  the band.
- **Tests:**
  - t1 at L2 0.92 is held (`compose_one_more`), then retired;
  - round 0 (t1 + t3, both easy) is still above the band and held;
  - round 1 (t1 + t5) lands in the band and goes to train. Its L2 comes from a simulated policy solving each component
    with its own skill, graded by the real grader;
  - bench sealing holds a train task that shares a bench lineage;
  - duplicates, too-hard tasks and inversions are placed as specified.

**G-10 (2026-10-10): stitch.** `forge/stitch.py` inverts G-span, file by file (`path -> diff`).
- **Owned files:** each span's oracle files. Two spans may own one file only with the identical diff; that is a
  nested span, recorded.
- **Consistency:** a span's start may pre-apply another span's file only with that span's oracle diff.
- **Residual:** files that every start pre-applies and no span owns must agree. They go into the stitched oracle
  (`residual="oracle"`, the seed's base start) or stay pre-applied (`"start"`, an easier variant, reported as such).
- **Errors:** files only some starts pre-apply, or conflicting diffs.
- **Gates:** the union of the spans' gates.
- **Reproduction claim:** a stitched task whose files equal the seed oracle's, under the seed's grader, has the seed's
  verdicts.
- **task04 (`crucible_pack_coding.forge.stitch_task04`, report `docs/modules/reports/m13_stitch.json`):** people,
  ncv and kinfu_family (originally the 4 spans incl. kinfu / kinfu_large_scale) stitch back to task04 v3:
  - 56/56 files with identical diffs, an empty start, and the seed's 11 gates in its order;
  - NCV is nested in People;
  - 7 shared helpers (gpu/containers, gpu/utils) are taught by no span and return to the oracle.

  The golden runner's v3 verdicts therefore carry over without a GPU grade.
- **Tests:**
  - synthetic spans reproduce the seed (nested merge, residual into the oracle or pre-applied);
  - each error case raises;
  - the real task04 check skips when the local seed data is absent.

**G-9 (2026-10-10): fuse (cross-World) and nest.**
- **`forge/fuse.py:fuse_worlds(a, b)`** builds one World from two.
  - Apps are unioned. A shared app id is renamed `b_<id>`, and b's tool patterns, event tool actions, grant/revoke
    patterns and policies are renamed to match.
  - Two apps may never share a state path or a bundle destination.
  - Same-id policy actors merge their tools. Any other shared actor must be identical: two different people under
    one name is an error.
  - Clock, backend and network must agree.
- **`compose.fuse(a, b, cross=…, renames=…)`:** independent components (A17 applies) plus cross-App checks with gate
  `x.<gate>`, which no component owns (`Composite.cross`). `rename_apps` renames a task's checks and oracle with the
  same map.
- **`compose.nest(outer, inner, at)`:** the inner oracle runs after outer's first `at` steps. Outer's actions are
  therefore not contiguous, and outer's own negatives get the inner oracle inserted at `at`. The order is graded
  only when a check observes it (pass such checks as `cross`).
- **Cross check:** `verify/state.py:cross_check` compares two Apps' row sets (`equal` or `subset`). It is the
  `CheckSpec.kind="cross"` used by the toy plan.
- **Toy second App:** a warehouse (`inventory`: shipped items, stock, returns). `crucible_pack_toy.forge.fused`
  generates its World from the helpdesk scenario. The fused task is t1_duplicate ⊕ "log the refunded order's items
  as damaged (quarantined, not restocked)", with `x.consistency`: refunded orders = returned orders.
- **Compose fix:** a drop-component negative now expects a gate with a positive (predicate) check. Before, it expected
  the alphabetically first owned gate, which on the fused task was an invariant that a no-op passes.
- **Tests:**
  - the fused WorldSpec validates, and two fresh starts give the same fingerprint;
  - the oracle scores 1.0; each part alone keeps its own gates but fails `x.consistency`, so the interaction is real;
  - every negative fails at its gate;
  - A14 and A17 pass;
  - fusion conflicts (state path, two actors under one name, clocks) raise;
  - nest interleaves as specified and its oracle scores 1.0.

**G-8 (2026-10-10): the swarm stage runner.** `forge/swarm.py` runs one role's homogeneous workers against the zoned
repo, GitSwarm-style.
- **Scheduling:** at most C in flight (default 5), with replacement on exit, until B charged episodes are spent. An
  episode is never started if the budget could not pay for every episode in flight.
  - Charged: a contribution (committed, invalid, or a zone breach) or a nomination.
  - A voluntary abstention is free. `max_episodes` (4·B) bounds a stage whose workers keep abstaining.
- **Announce:** `Gap | Deliverable | Success | Abort` before editing. Same key, or gap-token Jaccard ≥ 0.8 against an
  in-flight or contributed announcement, is a collision. The worker is cancelled before it edits, and a collision is
  counted but not charged.
- **Episode end:** `Contribute`, `Nominate` or `Abstain`. A contribution passes the publication validator
  (requirements, not correctness) and the repo's zone rule.
- **Dashboard:** budget, in-flight announcements, contributions, collisions, and the nomination tally labelled "not
  correctness evidence". It assigns no work.
- **Readout (`winner`):** most nominations; ties go to the smallest sha; no valid nomination means the root.
- **Repo speed-up:** tree reads are now one `ls-tree` plus one `cat-file --batch`, cached per immutable commit. This
  took the swarm test from 49 s to 3.5 s.
- **Tests:**
  - a Proposer swarm at C=5, B=30 over 8 seed tasks gives 30 distinct chain plans, 0 collisions out of 30
    announcements, and 8 free abstentions (38 episodes);
  - workers that ignore the dashboard and all want one plan: 1 commit, 19 collisions cancelled, 1 charged;
  - breach and invalid contributions are charged and rejected;
  - nominations of unknown commits are rejected, and ties break on the smallest sha;
  - an abstain-only stage stops at 40 episodes with nothing charged.

**G-7 (2026-10-10): the zoned forge repo.** GitSwarm's memory is a git DAG that every worker reads. A forge cannot
have that hole: a Verifier-writer that reads the oracle writes checks that encode it. `forge/repo.py` keeps one git
repo per family, split into visibility zones.
- **Zones:** `public/`, `oracle/`, `checks/` and `solver/`, with per-role read sets (the table in audit 06 §4.1).
- **Commits:** GitSwarm's format, one parent plus a mandatory `informed_by` set, message
  `[<type>|informed_by=<sha>,…] <summary>`. Commits are built with git plumbing (no worktree, so concurrent authors
  are safe) and kept alive by `refs/forge/<sha>`.
- **Breach scan:** runs before anything is written; a rejected commit leaves no object and is recorded in `breaches`.
  It rejects:
  - a write into a zone the author cannot read;
  - an added line that carries a zone canary (`<zone>/CANARY`);
  - an added line that quotes a line (≥ 20 characters, normalized) of a zone the author cannot read, indexed over every
    commit ever made. A line that also appears in a readable zone is not evidence. Quoting the oracle is the leak
    that matters, and it rarely carries a canary;
  - an `informed_by` citation of an unknown commit, or of a commit that touched an unreadable zone.
- **Views:** `view(role, sha)` and `materialize(role, sha, dir)` export only the readable zones, as read-only files
  (the worker's mount). The index lives in `.git/forge-index.json`, so a reopened repo scans the same way.
- **Tests:**
  - a Verifier-writer quoting an oracle line, or the oracle canary, is a breach and nothing is written;
  - a line the oracle shares with the public spec passes;
  - citations into `oracle/` are rejected for the Verifier-writer and allowed for the Adversary;
  - the black-box hacker's mount holds only `public/`, read-only;
  - a reopened repo still catches the quote.

**G-6 (2026-10-10): forge economics.** The ledger caps spend but says nothing about what the spend buys.
`forge/economy.py` tracks dollars per *accepted* task, which is Repo2RLEnv's reference number.
- **Book:** `YieldBook` keeps one row per plan run (generator, operator, seed, family, construction state, spend).
  M9 resolves each row later (accepted, confirmed A16 hacks). The book persists to `economy.jsonl`, so a resumed forge
  keeps its counts and its pauses.
- **Orchestrator:** with `book=`, `run_plan` records every run. A plan of a paused generator is not run (state
  `paused`). `Plan` gains `operator`, `seed` and `family`.
- **Node cost fix:** a node's cost now sums every attempt, including retries, parse failures and infra errors.
  Previously only the successful attempt counted, which understated $/accepted.
- **Yield** counts decided attempts only: rejected at construction, or decided by M9. A task pending M9 or abandoned
  (budget, infra) does not count against its generator.
- **Stop rules:** a generator is paused when its yield is below 10% after 20 decided attempts. It is also paused when
  its $/accepted exceeds its family's cap; with nothing accepted, once its spend exceeds the cap. Each pause files one
  issue. `resume` (a person, after fixing the generator) starts its counts over.
- **Diversity quota:** `admit` defers an accepted task that would put its operator or its seed above 30% of the
  family. The quota is enforced once the family has 10 accepted tasks. `release` admits deferred tasks once the family
  has diversified.
- **Report:** per operator, generator and family: attempts, decided, accepted, yield, $, $/accepted and hack-found
  rate; per-stage yield; the paused generators and the deferred tasks.
- **Tests:**
  - a 0%-yield generator runs exactly 20 plans (concurrency 1), the remaining 10 come back `paused`, and one issue is
    filed;
  - the report gives `chain` 0.02 $/accepted and a 1/6 hack-found rate, and `fuse` ∞ (60 paid attempts);
  - the book reloads with its pause intact;
  - pending and abandoned rows do not count against yield;
  - both forms of the cost cap pause;
  - the quota defers a 4th `chain` at 31% and releases it at 25%;
  - a 3-attempt rejected node costs 0.03.

**G-5 (2026-10-10): the adaptive red team (A16).** Audit 06 found red-teaming static: A13 replays authored attacks,
while the hack that matters is the one nobody wrote down. `forge/hacker.py` searches for reward without the work.
- **Search:** `MutationHacker` is black-box. It sees the instruction and honest solver trajectories (agent-level
  material) and mutates them toward less work, by defensive category: skip the interactions with other actors (asking
  for an approval), skip the read-only calls, keep only the state writes, drop each single step. Attempts are
  deduplicated and capped. An LLM or white-box hacker plugs in behind the same `propose(view)`.
- **Classify:** each attempt is graded by the task's grader and, when it scores, checked by the task's independent
  `property_audit(actions)`. Reward with a failed audit is a `reward_hack`; the other labels are
  `legitimate_success`, `legitimate_failure` and `infra` (masked grade).
- **A16 (`accept/checks/redteam.py`, cpu):** runs the hacker within `Policy.a16_attempts` (32) and fails on any
  confirmed hack. The metrics carry each hack's actions. A task without a property audit is NA (`no_property_audit`):
  its hacks could not be told from successes.
- **Promotion:** `promote` turns confirmed hacks into A13 `Attack`s (replayed on every later version) and registry
  defect candidates (`detector.check: A13`). The task goes back to the Verifier-writer.
- **Toy property audit:** the task's checks plus the pack's rules for its stated properties. A policy property always
  needs an in-episode approval, even when a check set dropped it. So the audit catches only what the pack's rules know;
  coding packs need their own audit (oracle differential, held-out tests).
- **Tests:**
  - with B.15 re-introduced (the event check dropped from t2_approval), the hacker's `skip_interactions` attempt is
    the confirmed hack (1 of 4 attempts; the other 3 fail legitimately) and A16 fails;
  - the promoted attacks score 0 at `policy` on the fixed task;
  - the fixed task passes A16 with 0 confirmed hacks (4/4 legitimate failures).

**G-4 (2026-10-10): the solver ladder.** `forge/ladder.py` measures difficulty with policies of increasing strength under
the training harness.
- **Rungs:** L0 floors; L1 the cheapest model; L2 the training policy; L3 the frontier, single attempt; L4 frontier
  pass@k; L5 a frontier swarm (optional). Each `Rung(name, run, k)` is sampled sequentially with
  `learnability.measure` (4 → 8 → 16, Wilson, stop when decided). pass@k uses the unbiased estimator
  `1 − C(n−c, k)/C(n, k)`.
- **Monotonicity:** a weaker rung's lower bound above a stronger rung's upper bound is an inversion, meaning a flaky
  grader or a check that grades style. A10 records the ladder of a task that provides one (`ladder()`), and an
  inversion fails A10.
- **Decisions:**
  - `accept`: train needs L2 in [0.1, 0.7]; bench needs L3's upper bound ≤ 0.2;
  - `steer`: `compose_one_more` (too easy), `drop_outermost` (too hard), `fix_grader` (inversion); instruction
    obscurity is never a knob;
  - `retire`: L2 above 0.7 in training sends the task back as a seed for one more composition.
- **Tests:**
  - pass@k matches the closed form;
  - a ladder on a real forged toy task (rungs solve with probability p, graded by the real grader) is monotone, accepts
    by L2's band, and steers too-easy and too-hard tasks;
  - an always-passing cheap rung against a failing frontier rung is an inversion that blocks bench acceptance and A10.

**G-2 / G-3 (2026-10-10): composition operators and the composition rigor checks.** Audit 06's first finding: no
source shows that swarms make tasks "exponentially" harder. Composition multiplies *candidates*, so every composite has
to prove that it is real.

- **Composites (`forge/compose.py`):** a composite is a task (TaskSpec, CheckSet, oracle, negatives) plus its component
  map: which oracle actions and which namespaced gates (`c1.<gate>`) each component owns, and the component's source
  hash. Composites therefore compose again (`depth`).
  - `chain(T1, T2, …)` runs the parts in order (`dependent` declares state flow).
  - `fan_in([T1..Tk], J)` adds an integration task.
  - Each operator emits a drop-component negative per component (A3 then keeps proving every part matters) and carries
    the components' own negatives inside the composite.
- **Found on the first toy chain:** each component's scope gate ("nothing else changed") failed the other component's
  legitimate actions, so the composite oracle scored 0.
  - Scope checks now carry their structure (`CheckSpec.spec["scope"]`: template plus allow list; the toy Test writer is
    `toy-3`).
  - Composition widens every scope to the union of the components' allow lists per template (`render_scope`).
  - Scope checks without structure (the LLM Test writer's, for now) leave the composite oracle failing A1, so such a
    composite is rejected, never silently wrong.
- **Checks (`accept/checks/composition.py`):** not applicable (`not_composite`) to ordinary tasks.
  - **A14 necessity:** the composite oracle without a component's actions must fail a gate that component owns.
  - **A15 complexity gain:** `p_composite ≤ min_i p_i − a15_delta` (0.1) on the frontier runs A10 uses. At the bench
    frontier (parts at p ≈ 0), a failing gate no component run fails is required instead. Pending without runs.
  - **A17 non-interference** (independent composites): each component's own actions still pass its own gates inside
    the composite.
- **Tests (forged toy/helpdesk_v2 tasks):**
  - a real chain: the composite oracle scores 1, A14 and A17 pass, and the drop-component negatives are killed at the
    dropped component's gate;
  - a padded component (only "nothing else changed" checks) is rejected by A14;
  - A15 passes a harder composite, rejects an equally hard one as decomposable, and is pending without runs;
  - ordinary tasks are not applicable, and a chain of a chain composes (depth 2).

**G-1 (2026-10-10): the concurrent forge orchestrator.** `Orchestrator.run_plan` awaited nodes one at a time, and one
plan ran at a time (audit 06: "a swarm in name only").
- **Concurrency:** nodes of the same dependency level (`Plan.levels()`) run concurrently under `node_concurrency`.
  `run_plans(plans, concurrency=C)` runs many plans at once (C defaults to 5, as in GitSwarm). The shared reserve /
  settle ledger is synchronous between awaits, so concurrent reservations still never exceed the cap.
- **Per-plan state:** canaries and visibility are kept per plan. Concurrent plans reuse node ids (`oracle`, `checks`),
  and a global map would have checked, or overwritten, another plan's canary.
- **Found while testing:** with level parallelism, a node can finish before a *same-level* canaried node registers its
  canary, so the in-node breach scan missed a leak the sequential order had caught.
  - After each level, every finished node's output is re-scanned against all canaries the plan now knows from nodes
    it cannot see. A late breach fails the node.
  - It also revokes the node's journal entry (a `breach` event carrying the input hash), so a resumed run never reuses
    the leaked output.
- **Tests:**
  - 5 concurrent plans give the same states and outputs as one-node-at-a-time, in under half the wall time;
  - a same-level leak is caught and stays caught on resume;
  - 20 concurrent reservations under a $0.25 cap construct exactly 5 tasks, with committed spend ≤ cap.
- **Also:** the G-tighten text now requires A1's 15% headroom, not 10%.

**F-3 / F-4 (2026-10-10): SGLang engine; top-p candidate sets over vLLM.**
- **`train/sglang_http.SGLangServerEngine`:** SGLang's native API, as slime drives it.
  - **Generation:** `/generate` with `input_ids` and `return_logprob` gives token ids and behaviour logprobs from
    `meta_info.output_token_logprobs`. With `routed_experts=True` (`--enable-return-routed-experts`), the per-token,
    per-layer top-k experts become `GenResult.routing` for R3 replay.
  - **Control:** per-call abort through the call's SGLang `rid`; `/pause_generation` / `/continue_generation`;
    `/release_memory_occupation` / `/resume_memory_occupation`, with the vLLM engine's order (a publish wakes only the
    weights; the KV cache wakes at `resume`).
  - **Weights:** `/update_weights_from_disk` with `success` checked (a refused load raises and keeps the version),
    then `/flush_cache`.
  - Token-level resume re-sends prompt plus kept tokens and records version spans.
- **`VLLMServerEngine(nucleus_k=K)`:** top-p sampling over vLLM was refused, since vLLM returns no nucleus. Each
  position's nucleus is now rebuilt from the top-K processed logprobs (`return_tokens_as_token_ids`): the smallest
  probability-sorted prefix reaching `top_p`. That is exact whenever the top-K mass reaches `top_p`. When it doesn't,
  or the sampled token falls outside the rebuilt nucleus (raw logprobs on the server), the call fails rather than guess.
- **Not built:** vLLM routed experts over HTTP (the capability says `routing: False`; SGLang provides it).
- **Tests (mock servers):**
  - SGLang: request fields, token ids, logprobs, routing flattening, resume spans, the abort `rid`, the full lifecycle
    order, refused weights;
  - vLLM: nucleus reconstruction, refusal without `nucleus_k`, insufficient top-K mass, and an out-of-nucleus sample.

**F-1 / F-2 (2026-10-10): engine capabilities and a real HF engine.**
- **`train/hf_engine.HFGenerateEngine`:** serves a Hugging Face causal LM (CPU / MPS / CUDA) with `TorchEngine`'s
  contract:
  - token ids in, token ids plus the sampled token's processed log-prob (after temperature and top-p) out;
  - version pinned at call start, token-level resume with `version_spans`, top-p candidate sets;
  - `abort_all` plus per-call `abort(call_id)`, and pause / resume;
  - KV-cache decoding (O(1) positions per token);
  - `load_weights(state | checkpoint dir)` loads into a copy while calls run and in place when idle. It accepts
    `HFPolicy`'s `model.`-prefixed state dicts and refuses a partial load (only a tied `lm_head` may be absent).

  Real models now run end to end through M12 without vLLM.
- **Capabilities (F-1):** every engine declares `capabilities()` (token ids, logprob convention, abort granularity,
  pause modes, sleep, top-p candidates, routing, weight-load form). `EnginePool.capabilities()` offers what every
  engine offers.
- **Logprob probe:** `probe_logprobs_mode(engine, reference)` *measures* the convention by sampling at T = 2 and
  comparing with a reference model at T = 2 and T = 1. The off-policy corrections assume processed logprobs, and vLLM
  defaults to raw ones.
- **Tests (random small Qwen3, CPU):**
  - engine logprobs equal `HFPolicy`'s at T ∈ {1, 0.7, 1.3} within 1e-4, so KV-cache decode equals a full forward;
  - the probe returns "processed" for the HF engine and "raw" for a raw-logprob engine;
  - a per-call abort stops only that call;
  - a publish during a running call leaves it on version 0 while new calls get version 1;
  - a bad state dict is refused, and the pool's capabilities are the intersection;
  - the M12 runtime trains 4 steps on the HF engine with train/inference mismatch < 1e-3.

**E: OpenEnv client mode and image fields (2026-10-10; audit 03 G7).** The OpenEnv importer ran environments in process
only, and pixels in non-MCP observations were serialized as JSON text.

- **Client mode:** `OpenEnvTask(client={"base_url": ...} | {"docker_image": ..., "environment": {...}}, action_schema=...)`
  reaches the environment's own server through OpenEnv's `GenericEnvClient`, as OpenEnv's clients do.
  - `ClientEnv` and `_ObsView` present the client's step results through the in-process surface, so journal,
    replay-restore (a fresh server session, reset, then re-issue with digest checks), rewards and termination are
    unchanged. This opens browsergym, atari, carla, unity and other Docker-only environments.
- **`image_fields`:** observation keys holding images (base64 or data URLs) become `ToolResult.images` (stored,
  digested into the replay), and the policy's text gets `<image:field>` instead of base64.
- **Tests (OpenEnv env):**
  - OpenEnv's real connect4 server in a separate process, through the client: native rewards and termination, and a
    mid-game restore replays over the wire;
  - a pixel environment: one PNG per step as an image, a placeholder in the text, and frames verified on restore.

**E: AWM-1K tasks with AWM's own verifiers (2026-10-10; audit 03 G12).** AWM scenarios loaded as Worlds (100/100) but
produced no reward. `interop/openenv/awm_task.AwmTask(scenario, task_idx)`:
- the instruction is the dataset's task;
- `open` copies the freshly seeded session DB to `initial.db`;
- `grade` runs that task's **code-mode** verifier (`gen_verifier.pure_code.jsonl`, 10,010 deterministic
  `verify_task_completion(initial_db, final_db, final_answer)`) on the initial DB and a copy of the final DB, both
  read-only.

It runs in AWM's own out-of-process runner, vendored verbatim (BSD-3) as `_vendor/openenv_awm/verifier_runner.py`:
an import allowlist, an `os` shim exposing only `os.path`, and CPU / memory / file / process limits. `complete` → 1,
`others` → 0, and a runner error is masked. The LLM-judged SQL mode is not used.

Tests:
- real AWM `social_network_8` task 0 through its generated service: the correct `update_profile_summary` call scores 1,
  a wrong summary 0, a no-op 0;
- the runner rejects a disallowed import as an execution error, which the task masks.

**E: two-player OpenEnv games as multi-agent Worlds (2026-10-10; audit 03 pick 8).** The OpenEnv importer had a single
`agent` actor.

- **Seats:** `OpenEnvTask(players=[...], turn_from=..., turn_map=...)` makes every seat a policy actor.
  - The App tracks whose move it is from each observation's metadata (connect4: `next_player`), after reset, every
    step and every replay.
  - A call from another seat is refused (`not_your_turn`, a policy fault, never executed or journaled), and the
    journal records each step's mover.
- **Game end:** `ToolResult.ends_for` lets a terminal observation end every seat, not only the mover.
- **Rewards:** `actor_rewards(journal)` attributes the outcome per seat. With `zero_sum`, the terminal reward goes to its
  mover and its negation to the others; otherwise each seat sums its own step rewards. The grade's check details carry
  it.
- **Test** (in the OpenEnv env), 100 seeded connect4 games of random legal moves, with two scripted seats in
  `run_episode` round robin:
  - final reward and termination equal the native environment's;
  - movers alternate in the journal, and the game's end ends both seats;
  - per-seat rewards are zero-sum, with the winning mover +1;
  - an out-of-turn first move by p2 is refused;
  - a mid-game restore replays with verified digests and restores whose turn it is.

**E: verifiers v1 import (2026-10-10; audit 03 G6, pick 6).** Rather than port each v1 environment's internals,
Crucible lets verifiers run its own episode (harness, toolsets, user simulator, `@reward` rubric) unchanged in its own
Python (`crucible/.envs/verifiers`), and supplies only the model.

- **Token exactness:** verifiers' `TrainClient` renders prompts to token ids and calls the vLLM
  `/inference/v1/generate` protocol, so its traces carry exact token records. Crucible serves that protocol from its
  engines (`gateway.generate_app`: sticky on verifiers' session header, aborts re-issued, versions stamped).
- **`interop/verifiers/v1_import.py`:**
  - `trace_to_record`: a v1 trace is a token graph (nodes: `token_ids`, trainable `mask`, `logprobs`; a prompt is the
    concatenation of its ancestors). Each sampled node becomes a `Turn` (the prompt delta since the previous
    completion, the masked span as the completion). The reward is the weighted rubric, and trace errors mask the record.
  - `RolloutGenerateServer` routes `/r/<key>/inference/v1/generate` to one rollout's `gen` (`Runtime._call`), so
    versions, barrier aborts and staleness are the runtime's own.
  - `vf_rollout(env_id, model)` is the runtime's `rollout_fn`: the prompt is a task index, run with `vf-eval
    --select.include.idx`.
  - `run_vf_eval` evaluates a verifiers env against any engine pool.
- **Found while testing:**
  - `generate_app` declared `req: Request` with `Request` imported inside the function. With postponed annotations,
    FastAPI read it as a query parameter (422). Signatures now use builtins only.
  - Runtime rollout ids (`glossary:0@0/2`) contain `/`, which broke the per-rollout route (404, every rollout masked).
    Routes now use a hash key.
- **Tests (real verifiers `glossary` env, local tool server, Qwen3 renderer, scripted Qwen3-token model):**
  - `vf-eval` on Crucible engines scores 5/5 on verifiers' own rubric;
  - the records are train-valid, and the reconstructed last prompt equals verifiers' token graph exactly;
  - the M12 runtime trains 2 steps on the verifiers env (8 records, 0 masked, versions stamped).
- **Not done:** tau2/tau3 need the upstream `sierra-research/tau2-bench` repo and a user-simulator model;
  BrowseComp-Plus needs its search corpus; a v0 (`load_environment`) import is still open.

**E: NeMo-Gym import (2026-10-10; audit 03 G5, pick 7).** NeMo-Gym's resources servers become Crucible Worlds
(`interop/nemo_gym/import_server.py`). The pattern is verifiers' `v1/tasksets/nemo_gym`: seed with the row, verify with
a rebuilt Responses object.

- **Server:** `NemoGymServer` runs one resources-server class in the NeMo-Gym Python (`crucible/.envs/nemo`, 3.13;
  `serve_resource.py`, which imports only `nemo_gym`) or attaches to a URL.
- **App (`nemo_gym/resources`):**
  - posts the dataset row to `/seed_session`; the session is the server's signed cookie, kept in the App's client;
  - exposes the row's function tools, each call being `POST /<tool>`;
  - **state** is a replay journal (calls, result digests, outputs): restore re-seeds a fresh session and re-issues the
    calls, verifying each digest.
- **Task:**
  - the instruction is the row's input messages;
  - `grade` rebuilds the Responses object, taking function calls and outputs from the App journal plus the final
    message from the trace, and posts `/verify` with the row. The server's reward is the score gate, and
    `mask_sample` masks;
  - `oracle_actions()` replays `ground_truth` function calls.
- **Found while testing:** `World.restore` does not rewind the trace (the log keeps abandoned branches), so a response
  built from the trace re-sent pre-restore calls and a correct restored episode scored 0. The response now comes from
  the App journal, which is exactly the calls applied to the current server session.
- **Tests (real NeMo-Gym servers):**
  - workplace_assistant: oracle 1.0 and no-op 0.0 on all 5 example rows;
  - MRCR and RULER: NeMo-Gym's recorded rollouts, replayed through Crucible Worlds, reproduce NeMo's own rewards
    (including MRCR's continuous 0.976);
  - snapshot/restore re-seeds and replays to a correct 1.0;
  - `mask_sample` masks.
- **Not covered:** `math_with_autograder` and the other judge-graded servers need a judge model server (gap G10).

**Kit grader: OOM attributed to "node contention" was the candidate's own memory (fixed 2026-10-10).** In task04 golden
re-grades, `astra_v3b` and `oob` had their `concurrency` gate masked as `node_contention` ("other processes held
28 GB"). History records both gates as passing, and the only process on the GPU was the grade's own `kinfu_replay`.

- **Cause:** `cand()` read the "foreign" GPU memory right after killing the candidate cohort. Device memory of a
  process that just exited is released asynchronously, so the reading still counted the candidate's previous run. The
  next run then OOMed on memory still being freed, and the OOM was masked as another user's job. The 2026-10-09
  `astra_v3b` "contention" was probably the same race.
- **Fix:**
  - `gpu_memory_settled()` polls until the reading stops falling, both before a candidate run and after a failed one;
  - `oom_under_contention(..., after_mib=...)` calls an OOM the node's fault only if the memory is *still* held once
    the candidate's processes are dead; otherwise the candidate's own OOM is scored.
- **Tests:** settling waits through a falling sequence, and an OOM whose memory disappears with the candidate is
  scored, not masked.
- **Pending:** re-grade `astra_v3b` and `oob` on the lab after the forge and task03 runners finish (one GPU job at a
  time).

**D-4 (2026-10-10): the launcher.** `train/launch.py` follows prime-rl's Ray-free process model.
- **One command:** `run_job(spec)` (or `python -m crucible.train.launch job.toml`) starts engine-server processes behind
  a `GatewayPool`, env-server processes, rollout-worker processes and the controller (runtime plus `LeaseController`,
  with a JSONL sink) in the launching process.
- **Supervision:** a dead engine server, worker or env server is restarted on its port, up to `max_restarts`. The rest
  of the system absorbs each outage:
  - the gateway turns the dead engine's calls into re-issued aborts and revives the restarted engine after replaying
    the current weights;
  - leases of a dead worker expire and are re-leased;
  - a dead env server's episodes are masked.

  Every child's process group is killed when the job ends, however it ends.
- **Found while testing:** the controller hung on shutdown. Workers' in-flight `/gen` handlers sat inside
  `Runtime._call`, re-issuing aborts forever once the engines were gone, and uvicorn waited on them. `/gen` handlers
  are now tracked per rollout and cancelled when their rollout ends or the controller stops, with a 2 s graceful
  timeout. The D-2 kill test also now kills a worker while it provably holds leases (with small-batch pulls it
  sometimes held none).
- **Test:** a 30-step job with 2 engines, 2 workers and 1 env server. Worker 0, engine 0 and env 0 are SIGKILLed after
  step 2.
  - All three are restarted, and the gateway evicts engine 0 and revives it through weight catch-up.
  - All 30 steps and publishes finish, with 0 masked rollouts, nothing trained twice, and 30 metric lines on disk.
- **Not built:** the Ray launcher (audit item G).

**D-3 (2026-10-10): remote environments over the NeMo-Gym resources protocol.**
- **Server (`interop/nemo_gym/server.py`):** previously unbounded, leaking abandoned sessions, and crashing with a
  `KeyError` on an unknown one. The admission and reaper patterns follow NeMo-Gym's `base_environment_server.py` and
  OpenEnv's `http_server.py`.
  - **Admission:** `max_concurrent_episodes` slots. A `seed_session` that waits past `queue_timeout_s` gets
    `503 {"admission": "queue_timeout"}`, a non-terminal failure.
  - **Reaper:** a lifespan task closes sessions idle past `idle_timeout_s`, so an abandoned client cannot leak Worlds.
  - **Unknown sessions** return 404. `/health` reports sessions, capacity and counters.
  - **CLI:** `python -m crucible.interop.nemo_gym.server --tasks module:factory --port P`.
- **Client (`interop/nemo_gym/client.py`):**
  - `remote_episode(urls, task_id, policy, rid=...)` drives any Crucible `Policy`: seed, tool calls as
    `POST /<app>__<tool>`, verify (graded from the World's state on the server), close. It reuses `LLMToolPolicy` as is,
    so context management, compaction and token capture are the in-process ones.
  - **Affinity:** session by `crc32(rid)`. A 503 masks the episode as `env_admission_timeout`, a lost server as
    `env_unreachable`, a reaped session as `env_session_lost`; never a 0.
  - `remote_world_rollout(urls, task_for, codec)` is the M12 `rollout_fn` over remote Worlds, with `EngineLLM` exact
    TITO, recorded as `crucible.rollout/v1`.
- **Tests (localhost, multi-process):**
  - two env-server processes give rewards identical to the in-process World on 30 seeds of scripted episodes, with
    sessions on both servers;
  - a server with one slot masks the second episode, reaps the abandoned slot, then serves again;
  - the M12 runtime trains two steps on remote Worlds, with train-valid records (byte codec, scripted engine emitting
    real tool calls).

**D-2 (2026-10-10): remote rollout workers (Ray-free, leased).** `train/workers.py`, following the slime leased queue,
verifiers EnvServer and prime-rl's process model.

- **Controller (`LeaseController`):** an HTTP app in the runtime's event loop. `Runtime(rollout_fn=lc.rollout_fn())`
  turns each rollout into a job.
  - **Endpoints:** workers `/lease` jobs, renew them with `/heartbeat` and post `/result`. Model calls come back via
    `/gen`, which goes through `Runtime._call`, so versions, sample-barrier aborts, pause, staleness and
    partial-rollout accounting are the in-process ones.
  - **Expiry:** an expired lease is re-leased first in line as a new attempt. `lc.attach(runtime)` resets the pool tag
    so the dead attempt's versions don't count toward staleness. After `max_attempts` the job is masked, never scored 0.
  - **No double training:** every request carries `(rid, attempt)`. `/gen` from a superseded attempt gets 409, its
    `/result` is ignored, and a heartbeat for it revokes the lease, so the zombie cancels its task.
- **Worker (`RolloutWorker`, `python -m crucible.train.workers`):**
  - leases small batches (`prefetch`) as capacity frees;
  - heartbeats from its own thread;
  - retries only requests that were never sent (connect errors), so a `/gen` is never generated twice;
  - reports environment exceptions as masked results.

  `env_rollout(env_factory)` is the runtime's token-env loop as a rollout function.
- **Found while testing:**
  - **Greedy leasing** (a worker took `concurrency` jobs at once) left two of four workers idle: 1.9× scaling.
    Small-batch pulls fixed it, to 3.7×.
  - **Heartbeats on the event loop** were starved by CPU-bound episodes, and connects timed out under load.
    Heartbeats moved to their own thread, with longer connect timeouts.
- **Tests:**
  - lease semantics: expiry, re-attempt, a revoked zombie, a stale result never applied, masking at
    `max_attempts`;
  - throughput 1 → 4 worker processes with a CPU-bound env step (`CopyTask(cpu_s=...)`): 3.3–3.8× over three runs,
    against the audit target of 3.2×;
  - a real runtime with two worker processes, one SIGKILLed after step 2: all 8 steps finish, leases expire and are
    re-leased, no rollout id is trained twice, every trained record is scored.

**D-1 (2026-10-10): the model gateway.** `EnginePool` had no health checks or eviction. Its crc32 routing spilled to
the least-loaded engine as soon as the pinned one reached `capacity`, which broke KV affinity under load, and one
stuck engine pinned the pool's version (audit 05 H4).

- **`train/gateway.GatewayPool`** is a drop-in `EnginePool`, following the rLLM `StickyLeastLoaded` + health-loop
  pattern and verl's per-server cache clearing.
  - **Routing:** a rollout stays on its engine while that engine has fewer than `spill_factor × capacity` calls in
    flight, so it queues rather than spills. New sessions go to the least-loaded live engine.
  - **Failures:** a call that raises comes back as `abort`, which `Runtime._call` already re-issues (with the partial
    output under token-level switching). The engine is evicted after `failure_threshold` failures, or at once on a
    connection error.
  - **Eviction** drops only that engine's sessions. rLLM clears every session on any worker change.
  - **Health loop:** evicts failing engines and revives recovered ones only after replaying the last
    `load_weights`, so a stale engine never serves.
  - **Version:** the minimum over *live* engines. `affinity()` audits the share of calls served by their session's
    first engine.
- **Remote engines:** `engine_app(engine)` serves any engine (FastAPI: generate / abort / pause / resume /
  load_weights from a checkpoint directory / health), `HTTPEngine` is its client, and
  `python -m crucible.train.gateway engine` starts a TinyPolicy engine server.
- **Acceptance (localhost, multi-process):** three engine-server processes sit behind the gateway, with a real trainer
  publishing through `CheckpointPublisher`. One engine is SIGKILLed after step 3, with calls in flight.
  - The run finishes all 10 steps and 10 publishes, with 0 masked rollouts and no recorded failures.
  - Exactly one eviction, and affinity on the surviving engines ≥ 99%. Stable over 5 repeats, about 3 s each.
- **Unit tests:** queue-then-spill, eviction scoped to the dead engine's sessions, a failed call as an abort, a dead
  engine not pinning the version, and revival with weight catch-up.

**C-8 (2026-10-10): shaping (A2-19, A2-21, A2-24..26).**
- **Incomplete trajectories:** `LimitPolicy` takes `"penalty:<c>"` for a limit stop (steps, timeout, context window);
  the reward becomes c in the M10 harness and in M12's `world_rollout(limit_policy=...)`.
- **Tool-format penalty:** `credit/shaping.format_errors_by_call` joins each model call's `actions` (`tool:<id>`) to
  the World's `tool_result` errors (`invalid_arguments`, `unknown_tool`, `bad_tool_name`, `unknown_app`) and stores the
  count on that call (`Turn.format_errors`). `TrainerConfig.tool_format_penalty` subtracts penalty × count from *that
  call's tokens only*, in call and segment items.
- **Speed term:** `speed_term(t, T) = 1/(1+t/T)`. `with_speed(reward_fn, beta, t_base)` adds `beta·h` to passing
  rollouts only, so a fast failure earns nothing. A terminal outcome already reaches every call's advantage, which is
  reward-to-go for that case.
- **Verbosity budget:** `verbosity_group_fn(sigma, l0_of=...)` is a group stage. A member longer than σ·ℓ₀ loses
  (reward 0), with ℓ₀ the task's reference length (default: the median length of the passing members).
- **Grader compute:** each rollout records `timings["grade_s"]`, and a reward function may report
  `reward_components["grader/usd"]`. Steps report `grader/s`, `grader/usd` and `grader/share` (of grading plus
  rollout time).
- **Tests:**
  - a real Runner at max_steps under `penalty:-0.5` scores −0.5;
  - malformed calls are attributed to the issuing call, and the penalty touches only that call's tokens (call and
    segment items) and nothing at penalty 0;
  - the speed term is monotone and gated on success, and the terminal reward reaches every call;
  - the verbosity budget uses a strict "longer than" boundary;
  - the grader is accounted on every step.

**C-7 (2026-10-10): context strategies, compaction as an action, thinking-history modes (A2-46..48).**
- **Strategies (`ContextConfig.strategy`, mode `compact`):** each keeps system + task verbatim and starts a new segment
  recorded with its `strategy`.
  - `summary`: the policy writes the summary, a recorded, trained call (as before).
  - `keep_recent`: GLM, the last k turns; no model call.
  - `discard_all`: system + task + a note.
  - `tiered`: HCM. Older tool outputs shrink to `tier_tool_chars` first (`tiered:shrink`), with a summary only if still
    over (`tiered:summary`).
  - `hybrid`: LongCat. `hybrid_summaries` summaries, then discard-all keeping the latest summary.
- **Compaction as a policy action (`mode="action"`, MiniMax):** the policy gets a `context_compact(summary,
  keep_recent_turns)` tool and compacts only when it chooses to; there is no automatic trigger.
  - The summary is that call's own output, so the call is recorded as `kind="compaction"` and trains with the episode
    advantage, with no extra LLM call.
  - Other tool calls in the same reply are not executed.
  - More than two compactions in a row without acting ends the episode as `too_many_compactions`.
- **Thinking history (`EngineLLM(thinking=...)`):**
  - `preserved` extends token for token. `interleaved` extends inside a tool loop and re-renders the history when a
    new user message arrives (the deployed template drops earlier reasoning there). `strip` re-renders every call.
  - The codec probes its own template (`thinking_behaviour()`: Qwen3 = interleaved, Qwen2.5 and the byte codec =
    preserved). A mode the template does not implement is refused, because training contexts would then differ from
    deployment.
  - A re-render is marked `new_segment` on the call's token record, and the policy starts a segment
    (`strategy="rerender"`). Exact TITO therefore holds within every segment.
- **Tests:**
  - each strategy on a scripted transcript: what is kept, whether a model call happened, the segment and its strategy;
  - hybrid discards after its summaries and keeps the latest one; tiered falls back to a summary;
  - action mode: no automatic trigger even far over budget, the compaction call is recorded and segmented, the next
    call sees the summary instead of the history; repeated compaction stops;
  - template probing and refusal;
  - interleaved on Qwen3: the tool loop extends exactly, and a user turn re-renders into a new segment.

**C-6 (2026-10-10): multi-domain runtime (A2-53..55).**
- **Per-source maps:** `Runtime` used to take one `rollout_fn`, one `reward_fn` and one harness. Now `env_factory`,
  `harness`, `rollout_fn`, `reward_fn` and `aux_fn` each accept `{source: value, "*": default}`, resolved by the
  rollout's scheduler source, so token envs, World tasks and judge-graded sources share one run. Every record carries
  `source` and `timings["active_s"]` (rollout wall time).
- **Per-source staleness (`SchedulerConfig.max_staleness_by_source`):**
  - `drop` evicts against the group's own τ.
  - `wait` needed more: it blocks only on *live* rollouts, while a finished, untrained group of a tight-τ source kept
    aging in the buffer (simulator: τ = 1 source trained at staleness 4). With a per-source map, groups at their
    bound go first in the batch, and a group already past its bound is evicted as stale, never trained over it.
  - Without a map, behaviour is unchanged. Selection and the simulator now report staleness per source.
- **Per-source metrics:** `reward/mean/<src>`, `len/{mean,p99}/<src>`, `active_s/{mean,p99}/<src>` and `records/<src>`,
  whenever a batch has several sources (or `per_source_metrics`). MiMo saw 90× length and 66× time spreads across
  sources.
- **Tests:**
  - a τ = 1 source never trains staleness > 1 next to a τ = 4 source, under `drop` and `wait`, and does exceed 1
    without the map;
  - one runtime with a token-env source, a `rollout_fn` source and a judge-graded source: every record is tagged with
    its source and its own harness, the judge runs only for its source, and the per-source metrics are present.

**C-5 (2026-10-10): teacher pool, on-policy distillation and the KL reference (route ④; A2-9..A2-12, A2-14).**
- **Teacher pool (`train/teachers.py`):** a teacher scores `(prompt, completion)` token ids.
  - **Implementations:** `TorchTeacher` (a local model under `no_grad`, in a thread) and `VLLMTeacher`
    (`/v1/completions` with `prompt_logprobs`, keeping the completion positions; a missing sampled token is an error,
    never a guess).
  - **Routing:** `TeacherPool(route={source: teacher})` with a default (MOPD: one expert per domain). An optional
    `reference` scorer fills `Turn.ref_logprobs`.
  - **Tokenizer gate:** `check_vocab(vocab_signature(tokenizer))` refuses a student/teacher vocabulary mismatch at
    `Runtime` construction, before any rollout.
- **Runtime:** `Runtime(teacher_pool=...)` scores each finished rollout before it enters the pool (status `grading`),
  overlapping other rollouts.
  - It sets `Turn.teacher_logprobs`, `RolloutRecord.teacher_id` and `RolloutRecord.source` (the scheduler source,
    now on every record).
  - Past `teacher_deadline_s` or on error: `teacher_on_miss="fallback"` trains the record on the RL signal only
    (`opd_fallback`), and `"mask"` masks it.
  - Metrics: `teacher/scored`, `teacher/fallback`, `teacher/seconds`.
- **Trainer:**
  - **Token OPD:** the per-token advantage `rl_coef·A + β(log π_T − log π_behaviour)` (Kimi / MiMo MOPD as a dense
    reward through the same correction and aggregation). A teacher identical to the policy contributes exactly 0;
    `rl_coef=0` is pure distillation.
  - **Full-vocabulary OPD (`opd_mode="full_vocab"`, DeepSeek):** `β·KL(π_θ‖π_T)` summed over the vocabulary at every
    trained position, with a local `TorchTrainer(teacher=...)`. The policies gained `completion_logprob_rows`.
  - **KL to the reference:** k3 `exp(r−l) − (r−l) − 1` on sampled tokens, with `kl_coef` and `kl_coef_by_source`
    (call and segment items).
  - With every coefficient at its neutral value, the step is the RL step bit for bit.
- **Tests:**
  - zero weights reproduce the RL step bitwise (call and segment items);
  - a teacher equal to the policy gives a zero gradient, and a different teacher a nonzero one;
  - the token advantage is checked by hand;
  - full-vocab KL equals a hand computation within 1e-6;
  - k3 golden value, 0 at the reference, per-source 0 honoured;
  - through the runtime: 100% token coverage with routed `teacher_id`, and a vocabulary mismatch refused;
  - a late teacher falls back on every record, or masks (never trained as 0);
  - `VLLMTeacher` against a mock server.
- **Not done (P2):** teacher offload (A2-13) and the actor serving as the judge (A2-27).

**Critic mode: advantages collided across rollouts (fixed 2026-10-10).** `TorchTrainer._gae_advantages` keyed per-token
advantages by `call_id`. World episodes all number their calls `agent:c1`, `agent:c2`…, so in a batch of World
rollouts each record overwrote the previous one's GAE: every rollout trained on the last record's advantages. The
critic tests passed because the toy task's call ids are unique per rollout. Advantages are now keyed by
`(rollout_id, call_id)` in GAE, per-call items and segment items. A regression test gives two records with identical
call ids and different rewards, and checks that they keep distinct advantages.

**C-4 (2026-10-10): online hack interception and the two-level detector (A2-28/29, checklist #41).** Before this,
shortcuts were only blocked by the environment (network policy) or noticed after the fact (probes), and
`hack/confirmed` came from the full-tier audit alone.

- **Interception (`world/intercept.py`):** at `World._call_tool`, after argument validation and before any execution,
  policy calls go through interceptors. A hit is not executed: the policy gets a refusal (or `action="dummy"`, a
  neutral result), an `intercept` event is logged, and the episode continues. It is not an infra fault, so it stays
  trainable and the shortcut does not pay.
  - **Default rules** (defensive categories, matched against any string argument):
    - `remote_code_fetch`: `git clone|fetch|pull <remote>`, `curl|wget <url>`; hosts on the world's allowlist pass;
    - `grader_paths`: `/logs/verifier`, `/solution`.
  - **`WorldSpec.intercept`** selects rules: `None` = defaults, `[]` = off. A public-network Harbor task, whose job may
    be to download, keeps only `grader_paths`.
  - `World.intercepts` is derived from the log (it follows resets and snapshots), and the trace outcome lists them.
- **Level 1 (`credit/hack_detector.rule_flags`):** high-recall suspicions over the trace: intercepts, writes to test or
  verifier files, exit-status spoofing written into code, and test-skipping flags. Both the M10 runner
  (`HarnessResult.flags`) and M12's `world_rollout` attach them to `RolloutRecord.flags`, with a
  `hack/intercepted` count.
- **Level 2 (`detector_group_fn(judge)`):** a group stage that judges flagged rollouts only, so cost scales with
  suspicion. The verdict must be exactly `{"hack": bool, …}`; anything else is `hack/judge_unparseable` and never
  confirms. `llm_hack_judge(llm)` is the LLM implementation, and `chain_group_fns(detector, gar)` zeroes confirmed hacks.
- **Alarm:** `StepMonitor.max_confirmed_hack_rate` (2%, MiMo's bound) raises an alarm, which feeds the existing full-tier
  re-audit hook.
- **Tests:**
  - on a real workspace World, `git clone` of a remote and a `curl` download are refused with the fingerprint unchanged;
    reading the verifier's files is refused; benign git runs; the episode is not masked; flags match;
  - allowlisted hosts and custom rules work, and spec rule selection (defaults / grader-only / off) behaves as set;
  - level 1 finds test writes, `os._exit(0)` and `--deselect`;
  - the detector confirms only parsed verdicts, and GAR zeroes the confirmed member while the group stays centred;
  - no judge means nothing is confirmed, and the hack-rate alarm fires.

**C-2 / C-3 (2026-10-10): the group grading stage, GAR and GRS.** `Runtime._grade` scored each record on its own, so
group-level reward operators (MiMo GAR, tournaments, zeroing hacks and recomputing the group mean) had nowhere to run.
- **The stage:** `Runtime(group_fn=...)` calls `group_fn(gid, members)` with deep copies of all n scored members.
  - **When:** it starts the moment a group turns terminal (`_complete`), so it overlaps the remaining rollouts.
  - **At the barrier:** it gets what is left of `RuntimeConfig.group_deadline_s`. A late stage is cancelled and a
    raising one is counted as a harness failure (outside the breaker window); either way the members train on raw
    advantages with `group_credit_fallback=True`.
  - **Results:** a stage returns its members in order. It may rewrite `reward` (the raw value is kept as
    `reward_components["raw"]`) and set `RolloutRecord.advantage`.
  - **Training:** the trainer uses those advantages for a group only when every scored member has one; otherwise it
    uses the group statistic.
  - **Lifecycle:** evicted groups cancel their stage, and groups restored from a checkpoint (stage tasks are not
    checkpointed) get theirs launched at the barrier.
  - **Metrics:** `group/graded`, `group/fallback`, `group/barrier_wait_s`.
- **GAR (`credit/gar.py`):**
  1. zero confirmed hacks and recompute the mean;
  2. in a mixed group, redistribute the passing members' positive mass M by quality factors f ∈ (0, 1]:
     `A' = A + λ(M·f/Σf − A)`. Mass is conserved exactly for any λ, and λ ∈ [0, 1] caps how far quality moves an
     advantage;
  3. leave failures unchanged and all-pass / all-fail groups untouched, so the group stays centred.

  `gar_group_fn(quality, hack=...)` adapts it as a stage. `quality` is the groupwise grader (sync or async), and the
  default hack verdict is `reward_components["hack/confirmed"]`, which the A2-29 detector will produce.
- **GRS:** `grs(R_test, S_sol, S_beh)` is multiplicative, so failures stay 0. `check_rubric_grounding` rejects rubric
  items that cite a rollout.
- **Tests:**
  - the hand-computed golden case, λ capping, and hack zeroing with a recomputed mean;
  - 500 random groups: passing mass conserved to 1e-9, failures unchanged, centred, masked stays masked;
  - through the runtime: every trained group graded and centred;
  - the trainer honours full overrides only;
  - a late stage falls back on all 8 groups, and a crashing stage falls back with 6 harness failures while the run
    completes.

**C-1 (2026-10-10): metric and record sinks.** `train/sinks.py` ports verl / SkyRL `Tracking` (one object fanning out
to backends) with two changes for long agentic runs:
- **Failure isolation:** a backend that raises is disabled (one warning, listed as `sinks_failed` in the summary) and
  the run continues. verl only guards `finish`.
- **Records as well as scalars:** every trained `crucible.rollout/v1` record (`record_rollouts`) and every failure
  (not only the two examples per kind in `RunReport`) streams to record sinks.

Backends:
- **`JsonlSink`:** line-buffered `metrics.jsonl` / `records.jsonl` and an atomically rewritten `summary.json`, so a
  killed run keeps everything up to its last step.
- **`TensorBoardSink`:** scalars.
- **`WandbSink`:** offline by default.
- **`ConsoleSink`.**

Wiring:
- `from_spec("jsonl:DIR,tensorboard:DIR,wandb:PROJECT[:online],console")` builds the sinks.
- `RuntimeConfig.sinks` logs at every step barrier, records at training and failure time, and writes the summary
  plus finish when the run ends, whatever the outcome.
- The M12 GPU gate takes `--sinks` (default: JSONL under its output dir).

Test: a real tiny runtime with a JSONL sink and a sink that raises finishes all steps. Metrics land at every step;
rollout records equal the trained records; failure rows equal `RunReport.failures`; and the summary names the
disabled sink. The golden and forge stagers also now treat an ssh session held open by the detached runner as
started.

**Kit grader: child interpreters lost the import order (2026-10-10, task03 lab golden).** Every task03 grade failed the
`api` gate where history passed:
- `test_pallas_kernels_are_used` traced no Pallas kernels;
- `test_cli_matches_api` differed by 4e-8.

`Trusted.python` applies the B.10 import order (grader code first, the candidate after the stdlib) with `python -I`
plus a bootstrap. That reaches the parent interpreter only: children spawned by the tests (`sys.executable -c PROBE`,
`python -m mdlite.cli`) got the default path and imported the image's own `mdlite` 1.3.0. The old runner had put the
candidate on `PYTHONPATH`, which children inherit, which is why history passed.

`Trusted.python` now also gives children a grader-owned, root-owned, read-only `sitecustomize` on `PYTHONPATH` that
applies the same order. The candidate stays after the stdlib in children too, and a test checks both properties. All
task03 grades and the task04 re-grades are re-staged with the fixed bundle.

**A-4 (2026-10-10): vLLM colocated wake order and the engine's first tests.** `VLLMServerEngine.load_weights` used to wake
the KV cache inside the publish, so the GPU held the full trainer and the full engine at once, defeating sleep mode.
- **Fix:** sleep state is explicit (`asleep ⊆ {weights, kv_cache}`):
  - the publish wakes only the weights, loads them and resets the prefix cache;
  - `resume()` wakes the KV cache after the runtime has offloaded the trainer.
- **Tests** (`tests/unit/test_vllm_engine.py`, mock vLLM server):
  - token-id completions with version stamps;
  - the exact HTTP sequence (pause → sleep; publish → wake weights / collective_rpc / reset_prefix_cache; resume →
    wake KV / resume);
  - generation waits while paused; abort returns `abort`;
  - split placement never sleeps;
  - the runtime order is publish → offload → resume.

**A-5 (2026-10-10): pool hot path and bounded memory.**
- **Admission:** `DataPool` keeps per-source indexes of untrained and live rollouts, maintained by one status setter (the
  scheduler's direct eviction goes through it too). Admission is O(1) per group: before, it scanned every tag ever
  created, O(N²) over a run.
- **Release:** trained and evicted groups are released with their tags and values; a group with a running straggler
  waits until it finishes.
- **Runtime:** finished asyncio tasks are pruned, and `RunReport.trained_records` keeps the newest `keep_records`.
- **Bug found on the way:** group ids used `len(pool.groups)`. Once groups are released, ids, rollout ids and engine
  sampling seeds collided and learning stopped. Ids now use the monotonic submit index.
- **Tests:** admission cost flat after 50k groups of history; a 60-step run holds only open work (tags ≤ 2× the cap,
  ≥ 240 groups released, tasks ≤ cap).

**A-6 (2026-10-10): the confirmed-hack metric is real.** `confirmed_hack_rate` read `reward_components["hack/confirmed"]`,
which nothing wrote, so it was always 0.
- **Audit:** `Runtime(audit_fn=..., audit_rate=0.05)` re-grades a deterministic sample of trained rollouts at full tier
  (`verify.tiers.audit_sample`; the policy cannot know which rollouts are sampled), asynchronously.
- **Confirmed hacks:** fast > 0 with full = 0 is a confirmed hack. It goes into `RunReport.confirmed_hacks`, the
  per-step `audit/*` metrics, the monitor's hack rate, and the record.
- **Probes:** World rollouts carry `probe/attempted` (blocked shortcut attempts stay trainable; an exposure already
  masks the episode), logged as `probe/attempt_share`.
- **Bug found by the test:** audit failures went through the rollout failure path and tripped the masked-share breaker.
  Failures outside rollouts are now recorded outside the breaker window.

**B-1 (2026-10-10): segment-level trainer items.** The reference trainer made every model call an item with its full
prompt, so a long multi-turn rollout cost O(turns²) forward tokens. The audit estimated ~300× redundancy on the
long-horizon gate's shape.
- **Change:** `TrainerConfig.item_unit = "segment"` (the new default) builds one causal sequence per (rollout, actor,
  compaction segment) with `bridges.common.flatten`:
  - sampled tokens are trained and observation tokens masked;
  - critic GAE advantages sit on their calls' spans.
- **Exception:** records needing per-call inputs (top-p candidates, R3 routing) keep per-call items.
- **Tests:** loss and gradient equal the per-call items within 1e-6 (group advantages + turn credit, critic GAE,
  compaction segments); doubling the turns doubles the forward tokens (per-call quadruples them).

**B-2 (2026-10-10): memory-bounded log-probs.** `HFPolicy` computed full-vocabulary fp32 logits, and their log-softmax,
for every position of every sequence, prompt included. That is ~78 GB at 128k tokens with Qwen3's vocabulary.
- **Change:** it now takes the decoder's last hidden state (the base model, no LM head) and gathers rows only at the
  positions predicting trained tokens. The FP32 head runs on those rows in checkpointed chunks of `logit_chunk`, and
  backward recomputes each chunk.
- **Tests:** values and gradients equal the full-logits computation within 1e-5 at chunk sizes 1, 7 and 4096; on MPS
  the memory held for backward drops by at least the full logits.

**B-3 (2026-10-10): linear token capture.** Every captured model call kept its full prompt token ids, in memory and in
every episode `checkpoint.json`, which is O(turns²).
- **Change:** `LLMToolPolicy._compress` stores a call whose prompt extends the previous call's prompt plus completion
  (same segment) as `prompt_delta` plus `prompt_prefix_len`. `runtime.record.build` rebuilds the full prompts and
  still flags prefix breaks. The policy's checkpoint carries the one running full sequence.
- **Test:** over 25 turns, 196 stored prompt tokens against about 4,600 with full-prompt capture, and the rebuilt
  record equals the full-prompt record.
- **Status:** Phase B (vertical scale) is complete.

**Lab acceptance runs, 2026-10-10 (M7 / M8, task04).**
- **M7 golden, GPU:** 7 of 8 submissions graded, all agree with history where measured. `sol_v3` and `oob` had their
  concurrency gate **masked** as node contention (another user's job took ~28 GB mid-grade), which is the correct infra
  classification. The comparison used to report a masked gate as a mismatch; it now lists it under `needs_regrade`.
  `astra_v3b` (the perf gate) is pending.
- **M8 forge spans, GPU:** all 25 grades masked, correctly, as `grader_exception`, because the stager shipped empty start
  patches. The empty-patch negative is named `start.patch`, and copied next to the span's own `start.patch` it
  overwrote it.
- **Forge staging fixes:** negatives are staged under `neg/` with an explicit file column; empty start or solution
  patches are refused; AppleDouble files are stripped; remote sizes are verified; masked grades show as masked in
  `collect`. Re-staged after the labstack campaign.

**Design finding (B.16).** A cap on rollouts *in flight* (Ling's S × B) does not bound staleness when finished rollouts
can wait in the buffer: they age too. In simulation with τ = 2 and `wait`, rollouts were trained at staleness 12. The
admission cap therefore counts every **untrained** rollout, in flight or buffered. With oldest-first selection a
rollout is trained within about S steps, and S = 1 is exactly synchronous.

**M12-infra batch A (2026-10-12; `docs/research/m12-infra/00-option-matrix.md`).** The training stack offers synchronous and asynchronous operation as orthogonal axes, the article's route-② point. Placement is set by `topology`: `colocated_sync`, `colocated_partial`, `disaggregated_stream`, and the new `elastic` (LongCat DORA: a resident generation group keeps generating while an elastic group is held for the trainer). Synchronization is set by the scheduler's staleness knobs. Batch A adds:
- partial-rollout round boundaries: B groups, Kimi K3's completion fraction λ, or Ring-1T C3PO++'s token budget Φ;
- Ring's retention σ (`max_cuts`);
- KV-preserving token-level resume (DeepSeek-V4.1, `resume_keeps_kv`).

Each runs in both the runtime and the simulator.

Three findings came out of building it:
- **B.20.** The simulator let a colocated training step start before the previous publish finished, and the publish re-enabled the engine mid-training. Colocated engine down-time was understated by about half; it is now serialized.
- **B.21.** `wait` watched only live rollouts. A group's finished members age while it waits for its last runner, so every topology trained at τ + 1. `wait` now checks open-group staleness, groups at the bound train first, and `bound_overflow` (`batch` by default) trains every group at the bound when more than B are selectable.
- **B.22.** The simulator measured staleness against the version engines serve. In the async topologies the trainer's committed version is ahead by the publish in flight, and staleness is now measured against it.

Across 24 seed and topology runs, no trained rollout exceeds τ.

**M12-infra batch B (2026-10-12): algorithms and credit.** Additions:
- **Estimators:** `remax` (greedy-rollout baseline from `reward_components["remax/baseline"]`) and `reinforce_pp` (global-batch normalization, no per-prompt baseline), next to mean, Dr.GRPO and RLOO.
- **Reward-to-go credit:** `credit_unit=reward_to_go`, built from `Turn.step_reward` minus the group mean of G_0 (MiniMax M2's (s_t, a_t) units).
- **Speed shaping** (MiniMax M2.5): h(T / T_base), gated on success, with the baseline taken over the group's correct solutions.
- **Latency shaping:** Kimi PARL critical steps, or the DeepSeek DAG critical path through `Turn.depends_on`.
- **Frozen sub-agents** in the reference trainer. Only the bridges honoured `frozen_actors` before, so a PARL-style run would have trained its sub-agents.
- **Entropy-adaptive mimo4 bands** (MiMo).
- **Group operators** (`train/group_ops.py`) for `group_fn`: hack zeroing, GRS (multiplicative rubric anchored to the test), GAR (positive advantage mass conserved, λ capped and re-centred), pairwise GenRM against a reference (O(n)) and round-robin tournaments (O(n²)).
- **QDQ** (`weights.QDQ`: OCP MXFP4 with power-of-two block scales, or FP8-E4M3). The QDQ'd values are written back into the trainer's own parameters, so the trainer and engine hold bit-identical weights, as MiMo does. Quantizing only the published copy would leave the mismatch in place.
- **Value pretraining** (`critic.pretrain_value`).

**M12-infra batch C (2026-10-11): scale-out.** Additions:
- **FSDP2 trainer** (`train/fsdp.py`, `FSDPTrainer`): parameters, gradients and AdamW moments sharded over a device mesh. A 2D mesh `(replicate, shard)` is HSDP. The loss stays each micro batch's share of the global token mean and the reduce-scatter sums, so one sharded update equals the single-process full-batch update (checked to 1e-6 on 2 and 4 gloo ranks and on a 3-block HF Llama).
  - FSDP issues collectives per forward, so every rank must run the same sequence. `reshard="step"` gathers once per update, accumulates locally and reduce-scatters once through a zero-weighted probe backward; it works for any model path. `reshard="layer"` (ZeRO-3 peak memory) gathers per unit per forward. Ranks pad to the same micro-batch count with zero-loss probe micro batches, and the π_θold recompute is chunked the same way.
  - `state_dict()` is a collective that gathers HF-named weights to rank 0 for publishing. `save_sharded` / `load_sharded` write and read each rank's shards with DCP, and the run checkpoint uses them (`train/state.py`).
  - Base-trainer hooks `_gather_params`, `_release_params`, `_old_logprobs`, `_equalize` and `_idle_micro_batch` are no-ops for the other trainers. The pre-clip gradient norm reads DTensor gradients (`full_tensor`).
- **Single controller over SPMD ranks** (`train/spmd.py`): `SPMDDriver` wraps the trainer on rank 0 and broadcasts each runtime call (step, state_dict, reset_optimizer, offload, onload, save/load_sharded) before running it; `follow` mirrors it on the other ranks. The real `Runtime` ran 16 steps over 4 HSDP ranks, learned, checkpointed shards, and a fresh runtime resumed at step 8 with exactly the weights published at version 8.
- **Placement planner** (`train/placement.py`): `ClusterSpec` × `ModelSpec` × `Workload` → topology, engine size (node-local TP, else whole nodes for 1T-class MoE), rollout / trainer / elastic split, FSDP / HSDP mesh, env workers, a GPU-by-GPU layout, `RuntimeConfig` overrides and a `SimConfig`. It uses a first-order cost model with every term exposed: fluid rollout under continuous batching, waves × tail for synchronous rounds, 6·N_active·tokens FLOPs, publish over NVLink or the network, and the colocated role swap over PCIe.
- **Launchers** (`train/cluster.py`): `process_specs` turns the layout into per-process environments (RANK / WORLD_SIZE / NODE_RANK / CUDA_VISIBLE_DEVICES / ENGINE…). `LocalLauncher` runs subprocesses; `RayLauncher` creates one `STRICT_PACK` placement group per node and pins each actor to its bundle. `smoke` is a pre-run placement check. `train/launch.py` stays the single-job supervisor of the HTTP process model.
- **Control plane at 10⁴–10⁵ live rollouts.** The simulator at 1,024 engines and 32k environments showed the scheduler scanning every live group per decision. The pool now indexes live groups, masked rids, pending rids, terminal groups, and each group's oldest version (cached, invalidated by the writes that change it). A lazy per-source min-heap answers the `wait` rule without a scan. The runtime's launch loop reads the pending index. Simulated outcomes are identical; the scheduler is 4–8× faster at that scale. Results: `docs/research/m12-infra/01-scale-sweeps.md`.

**M12-infra live validation (2026-10-11; `docs/research/m12-infra/02-live-validation.md`).** The multi-process scenarios re-ran over real NCCL on Modal (4 × L4, `scripts/modal_infra_check.py`), with soaks. Four defects were invisible on gloo / CPU:
- **B.23.** NCCL takes device tensors only, and the control tensors were on the CPU: the DP trainer's scalar all-reduce, FSDP's micro-batch count, and the weight sync's header, plan check, error code and free-memory reduce. The NCCL weight-sync path had never actually run. All now ride on the device under NCCL.
- **B.24.** FSDP2 step mode with a GRU: on CUDA, `nn.GRU.flatten_parameters()` re-points the weights into a cuDNN buffer, away from FSDP's gathered storage. After the first optimizer step the module read stale weights (0.016 off after two updates while the gradient norms matched). `FSDPTrainer` disables RNN weight flattening.
- **B.25.** `TorchEngine` and `TinyMoEPolicy` built their inputs on the CPU, so any GPU-resident policy broke the engine. Inputs now follow the parameters, and sampling stays on the CPU generator.
- **B.26.** Modal's sandbox rejects NCCL's cuMem allocator ("Cuda failure 801"). Set `NCCL_CUMEM_ENABLE=0` there.

- **B.27.** Instruction noise typo'd the exact word a task asked for, and the agent faithfully wrote the typo (live: GPT-6 Luna, 2 of 10 word-file episodes at full intensity). Value tokens are now never touched: words the task protects (`NoiseConfig.protect`), plus anything with digits or symbols, quoted text and capitalized names.
- **Rollout workers** now exit (code 75) after `orphan_after_s` without contact with their controller, instead of polling forever. A controller restarted as a new process (a new id) makes them drop the dead leases and continue.

**Live stress findings (2026-10-11; `docs/research/m12-infra/02-live-validation.md`).** Real backends: Modal multi-GPU, the lab node's Docker and KVM, and OpenRouter.
- **B.29.** Every Docker CLI call over `DOCKER_HOST=ssh://` opened its own SSH connection. Under concurrency, sshd's MaxStartups throttling left 5 of 24 `docker run` calls hung for over 11 minutes with no error. `world/docker_tunnel.py` now keeps one SSH tunnel per host (local Unix socket → remote `docker.sock`) for every CLI call. 40 concurrent calls take 0.3 s; 48 cold sandboxes start with p95 0.57 s.
- **B.30.** The warm sandbox pool's reuse mode refilled on every acquire, so the pool was full by release time and every cleaned container was destroyed: reuse never happened. Containers on loan are now counted, and 224 of 256 cycles reused with 0 leaks.
- **B.31.** `docker_grade` against a remote daemon masked grades as `no_reward_json`. Bind-mount sources resolve on the daemon's host, so the verifier saw empty mounts. A probe now names the real cause. The red-team attacks pass on the local daemon (27/27) and in a container on the lab's own daemon (22/22 without Harbor).
- **B.32.** The planner's memory model counted state only. Qwen3-1.7B ran out of memory at 8192-token micro batches on 2 × A10G although the planner said it fit. It now models activations and the fp32 LM-head chunk, with a fitted 1.25 overhead, which lands within 10% of the measured peaks (14.4 GB on A10G, 65 GB on H100). It also uses measured decode-versus-batch curves and calibrated presets (`placement.CALIBRATED`).
  - Measured MFU is 3.4% (A10G) and 16.7% (H100) for this stack, against the 0.35 default.
  - The weight-sync broadcast measures 1.1 GB/s on PCIe A10G and 7.6 GB/s on H100, against the assumed 450 GB/s NVLink.
- **B.33.** Remote simulators (Unity through OpenEnv's server, found live on the lab):
  - ControlApp re-opened a session at every reset, hitting the server's one-session cap; it now keeps one session for its life.
  - A stochastic simulator's reset failed the start-digest check; with `restore: none` a fresh start is now accepted.
  - The 20 s WebSocket keepalive dropped a server that blocks while loading a scene; `ping_interval_s` is now configurable.
  - Passing `env_id` on every reset reloaded the Unity binary each episode, so the scene is set once at server start (`reset_kwargs` stays for servers that switch cheaply).

- **B.34.** Two OpenRouter models playing Nim as two plain `LLMToolPolicy` seats both resigned after one move. A seat was never shown the opponent's move or its legal actions. When its turn came back, `run_episode` handed it its own stale observation (`legal` from when the other seat was to move), and a text reply on its own turn ended its game. Three fixes:
  - `run_episode` wraps the seats of a multi-seat turn-based World with `SeatView` + `MoveLog` (moved from `train/seats.py` to `actors/seat_view.py`);
  - `to_move` / `legal` are refreshed from the turn manager before a seat acts;
  - a text reply on one's own turn is nudged (`TURN_NUDGE`), not taken as Done.
  
  Five live games now reach a terminal, zero-sum state with no refused moves.

- **B.35.** MCP SDK 2.x (shipped in the vLLM image; pinned by the `verifiers` extra) broke every `common/sqlite_app` World there: FastMCP was renamed, `Tool.run` takes a context, results became objects, fields went snake_case, the streamable-HTTP client yields two streams over `httpx2`, `call_tool` timeouts became seconds, and tool exceptions are hidden unless raised as `ToolError`. `world/packaging/mcp_compat.py` (`http_streams`, `mcp_field`, `call_timeout`) plus version branches in the in-process module and the vendored server make the in-process path, the HTTP server and every client work on both 1.x and 2.x. The suite was run on both.
- **B.36.** Two defects surfaced on the way, found live on Modal:
  - a vLLM 400 for an over-long context crashed a driver as an HTTP error; it is now a `length` finish (the episode ends truncated), and other engine errors carry the server's reason;
  - the SPMD command channel waited inside NCCL, where a long non-training phase would trip the 10-minute watchdog; it now uses its own gloo group with a long timeout.

**M12 records for games, control and vision (2026-10-11).**
- **Games.** A turn-based multi-agent World trains as one `crucible.rollout/v1` record per trained seat (`train/seats.py`, `world_rollout(seats=SeatConfig)`).
  - Each record holds that seat's turns only, with `RolloutRecord.actor` set and the seat's own outcome from the game App. A missing outcome is masked, never scored 0.
  - The runtime keeps the episode as one pool rollout and expands its seat records into groups `<group>#seat=<id>` when it builds the batch. A zero-sum game's structural edge therefore never becomes credit (SPIRAL / prime-rl RAE).
  - Opponents' executed moves reach a seat as inbox observations on its next turn (`MoveLog` + `SeatView`), in the exact-TITO suffix with mask 0. Simultaneous moves are revealed only after the round resolves.
  - Frozen seats and fixed opponents play but emit nothing.
- **Control episodes** get a sibling schema, `crucible.control_rollout/v1` (`train/control_record.py`). It keeps the rollout/v1 envelope, but its steps carry the executed and sampled action, joint log-prob, value, observation digest, reward, termination and ticks; token fields could not hold these without looking train-valid to the LM trainer. `ppo(recorder=)` writes them, `record_from_world` covers LLM runs, and `replay` re-runs a record through a fresh ControlApp, checking every digest.
- **Vision-language models** use `HFVisionChatCodec`.
  - The processor expands image placeholders exactly as the engine's does, and the sentinel-cut suffix keeps TITO exact across image turns.
  - `Turn.media` stores content-addressed refs (offset, length, grid), never pixels.
  - `VisionHFPolicy` rebuilds the pixel inputs and recomputes log-probs equal to the engine's forward. `TorchTrainer` segment items do not carry media yet.

**M11 importers: EnterpriseOps-Gym and Harbor multi-step (2026-10-11).**
- **EnterpriseOps-Gym.** `interop/eog` turns a task (a `data/revised` file or an HF row) into an `EogTask`.
  - Each gym server becomes a `common/sqlite_app` seeded from its `gym_dbs.zip` dump. The dumps hold only INSERTs, so the schema is inferred from their column lists and the verifiers' queries; a unique integer `*_id` first column becomes `INTEGER PRIMARY KEY`, so new rows get the ids the verifiers hard-code.
  - `now` is pinned to the task's timestamp. The policy gets `sql_query` / `sql_execute`, because the named tools ship only in the benchmark's Docker images, plus an identity from `user_info` / `context`.
  - Each `database_state` verifier is a `SqlAssertion` in one outcome gate. All-zero-count or empty verifiers add a hard `unchanged` gate. `oracle_actions()` derives a witness from single-table verifiers.
  - Of the 13 local tasks, 12 import, all score 0 for doing nothing, and the 7 with single-table verifiers score 1. `response_check` and `tool_execution` verifiers are refused.
- **Harbor multi-step.** `interop/harbor/multi_step.py` maps each Harbor `[[steps]]` entry to a `TaskStep`.
  - Setup is Harbor's step preparation, grading is Harbor's `Verifier(step_name=...)`, and the gate is Harbor's own `min_reward` rule.
  - Steps that need a separate verifier environment are refused. Harbor's hello multi-step examples score 1 end to end on Docker.

**M12-infra batch D (2026-10-11): environment noise.** Additions:
- **Flaky-task admission** (`train/admission.py`, MiMo's rule, the same as acceptance check A4): `screen(sources, rerun, k=8)` reruns each prompt's reference k times and keeps only the stable ones. The verdicts are `flaky` (disagreeing), `broken` (identical and failing) and `infra` (too many infrastructure failures, retried later). It stops early at the first disagreement and reports the Clopper–Pearson bound for the kept set.
- **World noise** (`world/noise.py`, `World(noise=...)`): instruction typos, dropout, case flips and distractors at the first observation. Retryable transient tool errors are scored, never masked. Successful results get truncation, typos, empty output, shuffled lines or a dropped JSON field. The noise changes only the policy's view, never state or the verifier's view. Each perturbation is a `noise` trace event, seeded per episode.
- **Noise curriculum** (`train/noise.py`, LongCat): one intensity knob with a hashed clean control group. It moves on the clean / noisy reward gap over a window and counts only rollouts assigned at the current level. It is a sink (`RuntimeConfig.sinks`) that reads `noise/*` reward components; token environments get them through `Step.info["reward_components"]`, which the runtime now merges into the record.

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

### 10.5 The very-long-horizon RLM harness (M16, built 2026-10-13)

§10.4 made long episodes possible for a transcript policy. M16 does it for the RLM policy, where the model's work lives in a persistent Python kernel. It was built after three studies (`docs/research/long-horizon-rlm/01`–`03`): 17 frameworks, the papers and articles, and a stress test of the m14 harness to 7,161 steps. The stress test found that the m14 prompt ran 1.42× over its budget, overflows and kernel kills were masked, resumed rollouts never trained, restores slowed with every cell, checkpoints held the whole episode, and episode-wide counters ended honest long runs. Module doc: `docs/modules/M16-long-horizon-rlm.md`.

1. **Three tiers of state.** The prompt holds the system prompt, the task verbatim, a summary, a python-state row and the recent tail. The kernel holds variables. An append-only ledger on the host (`actors/rlm/ledger.py`) holds every message, the full output of every cell and every World result, stored once. The kernel reads the ledger with `history.*` and `OUT(i, cursor)`. A compaction therefore loses nothing; the summary is an index.
2. **Bounded prompt by construction** (`actors/rlm/context.py`). Tokens are the last call's provider usage plus an estimate of what was appended since, scaled by the measured ratio. The threshold is `max(W/2, min(0.95·W, W − max_out − reserve))`, never above `seq_len − max_out`, and the trigger includes the summary request's own size. A raised or silent overflow discards the reply, compacts and retries. The summarizer sees tool outputs (head + tail), retries up to 5 times, and a total failure is a clean truncation stop.
3. **The rebuilt context:** system, the task verbatim, a schema summary in update mode (rules in force with when each changed, every supervisor message verbatim, pending items with their triggers, the next step), the python-state row, a boundary marker that names the history API, and about 20k tokens of recent turns cut only before an assistant row.
4. **Durable kernel** (`kernel.py`, `repl.py`). dill snapshots every 200 cells and at each compaction; restore = snapshot + journal suffix (flat 0.5–0.7 s at any length), with full and lenient replay as fallbacks. History, `CONTEXT` and World replies are journaled as small markers and rebuilt from the ledger. A checkpoint is one context window, the state of each component, and offsets.
5. **Child agents** (`tree.py`). Persistent children up to depth 3 (default 1) with their own kernel, ledger and context; World access only when granted, then serialized through the root's action queue. One supervisor enforces tree-wide budgets with concurrency reserved per depth, so nesting cannot deadlock.
6. **Records that stay correct.** Chains are keyed `(actor, agent, segment)`. Sub-calls and summary attempts are nodes that never break a chain (the m14 records forwarded 7–20× the entered tokens; now 1×). A summary is trained only when accepted. A resume opens a segment on every agent, so a resumed rollout trains. Loss weight is per logical rollout, whatever the number of segments. Over-length rows are dropped and counted, never cut.
7. **Stops are typed.** Outcome: `agent_done`, `agent_gave_up`, `stalled`, and `context_overflow` when compaction is off. Truncation: `compaction_failed`, `budget:<kind>`. Masked: infra faults with their kind. Stall detection is windowed and uses only what the policy observes, never Φ.

**Gate** (`docs/research/long-horizon-rlm/06-validation.md`): on the backlog ladder with a 32k window and real Qwen2.5 tokens, a scripted model that keeps no hidden memory scores 1.0 at 5,000 World calls with the default config (m14: 0.227). No call goes above the window (m14: 237). Restore takes 0.69 s (m14: 9.8 s), and the checkpoints total 1.9 MB (m14: 44.8 MB). A model whose summary drops a rule still loses 15–27% of the reward, so the ladder measures summary quality. Every injected fault is scored and train-valid.

Also measured (`07-campaign.md`):
- 10,000 World calls at 1.0;
- child-agent trees of depth 1 and 2 at 1.0, every World step attributed once;
- with Qwen2.5-0.5B, the model's summaries accepted while its calls were corrected (it never wrote a cell).

A red-team hunt found 12 defects and fixed all of them. Among them, a policy could mask its own episode, a no-World child could grant the World, a stray task's World call was credited to the wrong cell, and the ledger had no disk cap (`ledger_max_bytes`, 2 GiB).

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

**M0 audit (2026-10-12).** Canonical JSON is one function (`core/canonical.py`: finite floats, `-0.0` folded, NFC);
content hashes carry a scheme number (`core/ids.py:HASH_SCHEME`) and reference ids are pinned by a test, so a schema edit
that would silently change existing ids fails CI. Every persisted document is versioned and migrates through upcasters;
no nullable field may carry a non-`None` default (a property test enforces it), because canonical JSON drops `None`. The
store is multi-process safe (staged, fsynced writes under flock, WAL) and self-repairing. Details:
`docs/audit/2026-10-12/M00-core.md`.

**M2 audit (2026-10-12).** The grounding gate compares facts as whole tokens (dates first), judges claims behind a
greeting, checks polarity against the cited sentence, and caps citations at two that must each support the sentence;
records are tokenized in every script and untimed records are hidden in replays. The LLM budget reserves each call's
worst case before it starts, and an unpriced hosted model pays a fallback price. `run_episode` takes the harness's
`sample_key`, reports every policy stop label, stops at the first masking failure, and by default ends once every
simulated person has ended. Simulated people get read-only tools unless a scenario allows writes, and their end markers
are judged. A failed simulator ends its conversation. Human and peer failures, silences and cancellations are masked,
and a restored A2A peer gets a fresh context opened with the restored conversation. The CI bot runs through the `ci`
App. A raising responder is a masked failure at the World's choke point. Details: `docs/audit/2026-10-12/M02-actors.md`.

**M3 audit (2026-10-12).** Redaction covers every string of a trace (ids, URLs, targets, dict keys), masks values under
credential-named keys, removes URL userinfo, short passwords, Basic auth, cookies, cut private keys, cards, IBANs and
SSNs, rewrites known people's names in text, runs in linear time, and pseudonymizes with an HMAC under a per-tenant key
(the public default salt let a guessed address be reversed). Stored traces are owner-only, stamped with the rules that
redacted them, purgeable, and keep M4's signals when a grown source is re-ingested. Ingest isolates failing sources.
Captures keep upstream error statuses, record interrupted streams, and tell side calls from context resets. Probes and
write detection read shell commands, argv arrays and harness sentences correctly. CI logs are never `good` when the
runner's summary says otherwise. PR history follows renames and matches reverts by exact PR number. Details:
`docs/audit/2026-10-12/M03-connectors.md`.

**M4 audit (2026-10-12).** The coding M4 gate had silently broken under the M3 fixes: hand labels named events by
position, and the outcome rule read a fetch time M3 had removed, so every merged PR was "unknown". Events now carry
stable keys (the source's row id) that signal ids, labels and carried signals use, and the PR connector records what it
saw of the 30-day window. The gate fails closed (missing traces, extra signals, a step graded where the label says none),
needs 50 labels and reports intervals. Reliability is judged only on evidence the mapping rule did not use, per family,
type and human or automated source, fitted on traces the gate does not score. Attribution treats `file:` objects as
paths and respects the App; business-App calls record the records they change; agent sessions yield interrupts and
rejections without a pack mapper; CI logs have a mapper. The toy pack's labels are scored for the first time. A revert
is never the "chosen" side of a preference pair. Details: `docs/audit/2026-10-12/M04-signals.md`.

**M5 audit (2026-10-12).** Before it, every locally cloned framework and dataset was read again for M5-M7 and the best
picked (`docs/audit/2026-10-12/M05-M07-study-sweep.md`). Replay now fails closed: every path the shared write detector
names is applied or reported, nothing is guessed (`git restore --staged`, `mv f dir`, conditionals, `printf`, expanding
here-docs), refused edits are skipped, and files an unreplayed command wrote are stale, so milestones on them are
unmeasured rather than scored on the wrong content. Mined coding milestones are measurable on a live World (the fs
adapter's grep takes `paths`, and `block_changed` exists there). Mining keeps C pointer lines, skips prose and forward
declarations, and CI mining counts added tests. Rediscovery is 4 of 6 families with every family hand-audited (it was
reported as 5 of 6). S1 is 0.919 on the audited replayer. Details: `docs/audit/2026-10-12/M05-workflow.md`.

**M6 audit (2026-10-12).** Dense credit now honours gates everywhere it leaves M6: a gate-failed or hacked episode gets
no positive turn credit, a group without outcome spread gets none, the per-turn term is centred within each episode and
never flips the sign of a turn's outcome advantage, and the live trainer passes every call's ΔΦ through that rule (it
added raw ΔΦ). `adv_signed` (ported from Xiaomi's verl fork) is available for marked calls. R1-R7 can now fail:
untested properties are reported, crafted negatives must hit a gate while honest failures keep κΦ (DESIGN's R3 is
corrected to say so), and R7 bounds farmable progress. C6 is wired: an unmeasured milestone is not a zero, and a masked
reward is `None`. Rubric items parse their verdict from the last line, ignore unmeasured items, and need fixtures and
data to validate. NeMo-Gym's reward components sum to the reward. Details: `docs/audit/2026-10-12/M06-credit.md`.

**M7 audit (2026-10-12).** A verdict can no longer be forged or turned into a mask by the candidate. Hidden tests: a
`pytest11` entry point in the candidate's install rewrote every outcome (a wrong library scored 1 as root); plugin
autoload is off, outcomes are recorded from the call before any other hook, `unittest` assertion tampering is reported,
and exact node ids can be required. Masks: a certain hard-gate fail now decides the reward even when another check is
unmeasured; candidate-side perf noise, a missing row, a policy-dropped table, an `ABSTAIN` directory and a FIFO copied
in `setup` are fails, not masks. Perf: timings must be positive and finite, record mode keeps the whole-process wall
bar, and task03's outside-clock differencing rejects a start-up that shrinks with the step count. Static: Python
imports are parsed (`import os, ctypes`), builtin `exec`/`eval`/`compile` and `__builtins__` lookups, `.pyc`/zip/`ar`
files and `pytest11` entry points are flagged, skipped directories are checked for binaries, C continuations are
spliced. Reward files: stale files are removed on the host, NaN masks, a score gate averages its checks, and under
runtimes that cannot mask, the grader clears rLLM's agent-writable reward file and `interop/harbor/metrics` provides a
masked-mean Harbor metric (stock `Mean` counts a mask as 0; wiring it into the M11 export is open). Leak audit: every top-level directory, files of any size, refs to blobs and trees. Collect:
the base is the commit recorded before the agent ran, and only root-level build trees are dropped. Details:
`docs/audit/2026-10-12/M07-verify.md`.

**M8 audit (2026-10-12).** Asymmetric information now holds in the forge's own memory: the zoned repo enforces an
information-flow rule (nothing written into a zone may come from a zone one of its readers cannot read), with
shingle-based quote scans and a parent-tree exemption, so an adversary can no longer copy oracle lines into `public/`.
The M9 subset's A5 sees the oracle-only values and the ticket data the agent reads; A7 needs the whole property
sentence and A3 builds its own wording-only alternative, so a check on oracle wording fails; A8 has a closed provenance
grammar. Spend is attributed per node (`actors/llm.SpendScope`), a settle never aborts a run, and one role's or worker's
failure ends only its node or episode. Masked rollouts leave learnability, the ladder and calibration (outcomes
`bool | None`). A16 needs measured evidence and searches past a specialist's memory. Coding trap tasks state the seed's
requirement and are gated by the seed grader's behavioural gate; C2 checks are Φ milestones only, kept when they
separate code, not comments. Details: `docs/audit/2026-10-12/M08-forge.md`.

**M9 audit (2026-10-12).** Acceptance is one verdict on every path: `crucible accept` and `crucible build` run the full
contract (`accept_task`) and report accepted, provisional (nothing failed; only checks that need what the run lacks are
pending) or rejected; the model-free subset is never called acceptance. The agent-image leak check reads the leak
report's findings (it could never fail). A3 fails with zero scored mutants, and only a dropped read may be "equivalent"
(a dropped approval message leaves the state alone and still breaks the process). The acceptance key hashes every
module that decides a verdict. Task facts require their capabilities (a reference build needs self-consistency, an
image needs a pinned digest); red-team checks need controls; N/A comes from facts, not adapter assertions. Repair
verifies A2, A3 and A13 too. A10 samples a fixed n, counts record-beating runs as successes, shares the ladder's bench
band, and only counts calibration runs bound to the task's current grading environment. Grade store and cache writes
are atomic and keyed by content; backfill quarantines anything but a pass. Details: `docs/audit/2026-10-12/M09-accept.md`.

**M10 audit (2026-10-12).** A policy can no longer turn its own failure into a mask (P16): a repository it broke is
`agent_broke_env`, files it made unreadable are its work product, a data error in its own code is
`crucible:policy_error` (scored 0), and its background jobs die with its World, which closes before any grade. Socket
and connection errors are transport (masked), and only the episode deadline is a scored `timeout`. Every attempt is
contained (a harness, record or rollout exception is a masked harness failure), with an optional per-attempt deadline
and backoff outside the concurrency slot (jittered, capped and never below a provider's Retry-After; M10 lab audit R-5). Retries re-grade or re-collect the work of the attempt that ran the agent,
with its recorded base and its model calls; nothing re-runs the agent. Reservations hold real money (a share of the
cap, at least the dearest settled attempt); uncertain spend above tolerance stops the run. Resume chains across
crashes and process deaths and charges only new calls. Credentials are scrubbed at the source (`LLMError`) and at the
record boundary (`core/secrets.py`). Registered pack kinds keep their blame; unmapped names are `unknown`; stale images,
stale builds, 4xx rejections and masked limit stops are not retried. Details: `docs/audit/2026-10-12/M10-runtime.md`.

**M11 audit (2026-10-12).** Every foreign path follows the M10 rule: masks travel as masks, policy failures are scored.
An infrastructure fault recorded in the World masks the NeMo-Gym export's `/verify`, the OpenEnv export's final step
and agentenv's `crucible_verify`; App servers carry the fault label across the MCP hop. An imported framework's own
failure (a NeMo-Gym 500 or dead server, a missing, non-finite or out-of-range reward, a NaN OpenEnv reward, a MiMo score
outside [0, 1]) masks. The policy's own failures are scored, with a health check where it is ambiguous: a raising
OpenEnv action on an environment whose journal replays exactly, a 4xx NeMo-Gym tool reply, an answer that breaks an AWM
verifier which runs on the untouched database, a Harbor container in which the verifier user can no longer write its
reward. The policy no longer reaches the grader or the control plane: one grade per NeMo-Gym session ends it (the model
could call `verify` as a tool), agents get a role-pinned `/mcp`-only proxy instead of the agentenv gateway root, Harbor
in place empties `/logs/verifier` and `/tests` before grading and refuses separate-verifier tasks, the verifiers export
removes grader-only files before `test.sh`, and policy SQL runs under an SQLite authorizer (no ATTACH, VACUUM INTO or
setting PRAGMAs). Outside agents' gateway calls are World steps (choke point, log, clock, events). Framework semantics
are the framework's: Harbor's reward rules everywhere (`interop/harbor/rewards.py`), its verifier network phase,
env and user; NeMo-Gym's pydantic models for every route; verifiers' `ok`, branches and our exact logprobs; EOG's
`_compare_values` and typed columns; MiMo's `detail.passed`, with `verify.py` in an isolated subprocess. A score check
may report `success`, so an imported environment's 0 is no longer an outcome pass. Corrections to earlier claims in this
document: NeMo-RL counts a masked sample's 0.0 in its GRPO group baselines (`grpo.py:3458-3499`), so the M12 NeMo-RL
bridge must fill masked rows first; NeMo-Gym's `harbor_agent` scores a missing reward 0 (unsupported path); the MiMo
"byte-identical re-export" proves only that the import left its source untouched; the verifiers gate runs verifiers'
classes on a stub runtime. Details: `docs/audit/2026-10-12/M11-interop.md`.

**M15 audit (2026-10-13): the studio.** The datasets `crucible label` wrote trained the wrong things: it rebuilt rows
from the ATIF transcript and the bench reward, never the runner's training view, so oracle, no-op and adversary
episodes became SFT positives and DPO chosen (every toy pair was oracle against no-op), held-out benchmark items and
flagged or invalid episodes were kept, RLM episodes were rewritten into chats the model never produced, DPO and KTO
completions carried the rest of the episode (61% of the tokens TRL would train were World output), and no SFT file loaded
with HF `datasets`. Now label reads `rollout.json`, `validity.json` and the sealed `splits.json`, takes positives only from
the model's own unflagged episodes under one rule, emits one policy turn per completion in formats TRL, verl and
prime-rl train correctly (checked with their own preprocessing and the real Qwen and gemma templates), and the runner
writes the World's tool cards into ATIF. Build and export fail loudly: an oracle crash, a rejected-everything forge or an
empty bench is a non-zero exit, rebuilds are staged, train-split items are never benched, and the contamination step runs
in build. Exports copy an allowlist and scan for secrets (they shipped the pack's `.env` and raw logs, and could delete the
source pack). The explorer contains every file read and serves reports under a sandbox CSP (a symlink escaped its root
and a planted report ran script on its origin). Every command records its resolved config with provenance (version,
git sha, pack hash) and refuses a changed rerun before writing. The public site follows the copy rules (no allusions to
other work, no module codes in anchors, impersonal), enforced by a test. Details: `docs/audit/2026-10-13/M15-studio.md`.

**M14 audit (2026-10-13): packs and the RLM agent.** A pack is accepted only if it can fail: `crucible pack check`
passed 19 of 25 deliberately broken packs and 17 of 21 broken finance_ap copies (a grader that always pays once fixtures
were missing, NaN and out-of-range rewards, a self-declared `requires` that turned the semantic level into a silent skip,
a misspelled `redaction_env` that stored the secret). Now missing or empty fixtures and tasks fail, every reward must be
finite and in range, fixtures need a case at each end of the range, seeded packs are re-graded on fresh Worlds, each pack
declares adversary policies with an `expect_max` bound that run at the semantic level, skips are loud, each pack is
checked in its own process, licences are SPDX, placeholders are scanned everywhere, and `--level accept` runs the M9
contract (finance_ap had never been through it). The packs' own graders were hackable: an out-of-range integer argument
masked a failing episode (argument-binding errors are now the policy's scored error, in core), a typed supervisor name
passed the toy's approval gate, a split refund dodged its threshold, comment stuffing outranked honest coding failures in
dense Φ, and finance_ap was solvable without reading the policy (a memorized 2% tolerance solved 73% of worlds; its v3
generator draws tolerance and rule order per world, forces compound, near-miss, restated-duplicate and aliased-vendor
cases, and is all-or-nothing; every fixed shortcut now solves at most the share of worlds that happen to match it). Packs install as wheels. A validation campaign on the fixed tree (≈4,750 episodes) found 14 more defects, fixed before the
commit, among them a lone surrogate in a tool argument that masked episodes and coding Φ fooled by string literals and
dead code (now a lexer / parser code view). The RLM agent runs model code in the isolated Docker kernel by default (it ran
on the host with the parent's credentials), speaks its protocol on a private fd with validated frames (cell code could
forge frames and turn its failure into an infra mask), never trains sub-LLM calls answered by another model, masks
sub-call and sandbox faults, budgets sub-calls and spawns, and keeps a harness edit only on paired, interleaved,
multi-seed evidence. Details: `docs/audit/2026-10-13/M14-packs-rlm.md`.

**M13 audit (2026-10-13).** Benchmarks now measure what they claim. A hack never outranks an honest model: exposed
answers score 0 with a flag in bench views, and in Harbor in place a grader-toolchain manifest taken before the policy acts
is checked host-side at grade time, so stubbing the grader's own tools is the policy's scored 0 (`crucible:grader_tampered`;
it scored 1.0 on every Harbor set before). Infra faults mask by host-side evidence (a lost container or a grade-time
network drop scored 0 before). Only valid tasks count: a validity manifest (oracle and no-op controls by default, or M9
acceptance) removes broken, flaky and no-op-passing tasks from every metric, IRT and compare. Every number states its
denominator: the headline is the mean of per-task means, one task set serves every k, planned / run / owed and the stop
reason are reported, spend includes masked retries, CIs are t intervals over tasks with the task count shown, and
comparisons pair per task (McNemar on task flips, a sign-flip test on per-task deltas). Splits close over lineage
components and the seal covers every label; dynamic item ids never derive from seeds and need a secret key; the IRT fit is
a converged joint MAP with anchor-centred priors and honest SEs, and drift is robust with multiplicity control; the
contamination tools know the real canaries in the local datasets and subtract template boilerplate. Details:
`docs/audit/2026-10-13/M13-bench.md`.

**M13 audit: Harbor in place, OpenEnv runs, environment starts (2026-10-13; bench campaign B1, B2, B8, B12).** P23 now
holds in place too: Harbor's verifier runs in the agent's container, so the grader's toolchain is held to the pristine
image (`interop/harbor/grader_guard.py`). At `open`, before the policy acts, the container is trusted: the verifier
user's `PATH`/`HOME`, every command the test scripts run (plus `sh`/`bash`/`env` and install chains: `apt-get` →
`dpkg`, `curl | sh` → coreutils) resolved with real paths and interpreters, the venvs they name, each grader Python's
stdlib and `sys.path`, and a host-side `docker diff` baseline are recorded. At the grade only the host is trusted
(`docker diff`, `docker cp`): a changed grader tool, a new executable ahead of one on `PATH`, anything in a verifier venv
or grader stdlib, a Python startup hook, `/etc/ld.so.preload`, a root pytest config, an apt hook or local apt source, an
index-redirecting pip/uv config, or a replaced `PATH` directory is `crucible:grader_tampered` (policy blame, scored 0,
rollout flag `grader_tampered` with the paths). A change a package manager explains (dpkg md5sums or a pip `RECORD` of a
package the policy installed) is a suspicion flag (`grader_tool_repackaged`), and a grader-only tool the image lacks
that the policy pre-placed is moved aside for the grade, whose own install provides it (`grader_tool_preinstalled`). The
verifier runs with the pristine `PATH` and a fresh `PYTHONPYCACHEPREFIX` (planted bytecode is never read), and the check
runs again after it. That covered the toolchain only. Since M10 lab audit R-2 (2026-10-15) no in-place grade starts
until a host-side quiesce receipt is clean (`interop/harbor/quiesce.py`, `<workdir>/../verifier/quiesce.json`): a
process baseline taken at `open` before the policy acts (PID 1's host pid, a start mark, `docker top`), then at each
grade (single-step, every multi-step step, R-1's shared-verifier seam) every later process killed by a builtin-only root
`sh` and confirmed gone by the host's `docker top -eo ...` taken with the container paused (a fork-and-die chain is
listed too; a multi-threaded zombie leader is a target), before the reset of `/logs/verifier`. Only the host listing
decides. A process still listed after the bounded rounds is `crucible:policy_misbehavior` (scored 0, flag
`policy_process_survived`); one killed and confirmed gone is not blamed but flagged `policy_process_killed`
(non-scoring: the reward is kept, but a passing attempt counts in bench's hack `rate` as a flagged pass and studio
skips it; this reading of R-2's Acceptance 1 was chosen on 2026-10-17); a host that cannot list the container,
or a helper exec the daemon never started, masks `crucible:backend_error`; a later multi-step step's `setup.sh` that
left a process running masks `crucible:harness_unsupported` (unknown blame, never retried: a root policy can make that
setup start one, so the 5% unknown-blame stop bounds it), and a task whose later setup leaves one running in a pristine
preflight container, before any policy, is refused at `open` with harness blame (review round 3); what a clean step
grade left running (its `test.sh`) and its descendants are listed as `leftovers`, killed at the next step's grade and
never flagged; a target in state D (its leader or, from a per-thread `docker top -eLo`, any thread) is never paused
for (the freezer cannot take it), so the state-D wait stays reachable, and a pause refused while a target turned D
ends the quiesce `uninterruptible` (the cgroup may still be freezing: never the policy's). Non-Docker providers
are refused in place outright (`crucible:harness_unsupported`; `grader_integrity="off"` no longer admits them).
Accepted trade-off: a task whose tests need a process the agent left running fails in place. Residual: shared
libraries, a forged package database, project config in the agent's workdir, `.pth` files of a system Python the
grader shares with the policy by design, a pristine (pre-baseline) service the policy took over that forks after the
final listing, a pristine process's host pid reused within the clock slack, a pristine service the policy sets
respawning, which masks the attempt with unknown blame instead of scoring it (bounded by the 5% unknown-blame stop),
and a process the policy's code under test
starts during the verifier (the receipt covers processes alive when the grade starts). Infrastructure is told apart by host-side
evidence: a container that is gone, stopped through the daemon API, or whose daemon is unreachable masks
`crucible:env_container_lost` (mid-episode too: `HarborExec` raises and the task's `infra_cause` names it); one that
exited from inside is `agent_broke_env`; a failed grade whose test script needs the network, with the container
detached or the host offline, masks `crucible:grader_network_error`. A grade's `meta.flags` join the rollout's flags
and a policy-blamed `meta.failure` becomes the attempt's failure (`runtime/harness/native._take_grade_findings`).
`credit/hack_detector` adds `shadows_grader_tool`. Image pulls and Dockerfile builds that fail are
`crucible:env_image_unavailable` / `crucible:env_build_failed` (infra, not retried when deterministic) with the cause,
the tail and the full log (`env_start_error.log`). OpenEnv tasks run through the runner: ATIF gets the World's rendered
instruction, and Harbor's ATIF modules load lazily (skipped where Harbor is absent). Tests:
`tests/unit/test_m13_audit_harbor.py`.

**M12 audit (2026-10-13).** The per-token loss math already matched verl, MiMo-verl, NeMo-RL, SkyRL, prime-rl, TRL
and AReaL to 1e-8; the defects were in what surrounds it. Numerics: log ratios clamp at ±20, gated tokens multiply a
finite zero weight, non-finite behaviour logprobs drop their token, and a non-finite gradient norm skips the step (one
behaviour logprob of −9999 used to NaN every parameter). The loss unit is the rollout: every aggregation is an exact
per-rollout row weight (`objectives.row_weights`), so the sequence, prompt and HisPO means micro-batch and run under
DP, and GSPO / MIS-PO / HisPO statistics span a rollout's rows; `loss_agg="auto"` gives GSPO and HisPO their own unit.
The trainer trains only what was sampled as it was sampled: token-incomplete records are masked, a temperature mismatch
is refused, images reach vision policies, teachers and references score at the turn's temperature and nucleus. One GAR
(MiMo's redistribution), with its pass set from the raw outcome, so GRS / BT / tournament compose with it. The critic
replaces a truncated rollout's outcome with V(s_T) and sees shaped rewards. The scheduler's bounds are hard (the
off-policy cap and the FIFO window no longer train past τ or deadlock; `allow` ≤ τ + 1; refills and zero-variance
resampling are bounded); every rollout exception masks under a named kind. Exact TITO per conversation, stop at the
codec's end-of-turn, a seed per call; vLLM drains before a weight load; the gateway snapshots weights; SGLang R3 follows
slime's row contract; the vLLM nucleus is the finite processed entries. Data parallelism agrees on metric keys and
used parameters, so an empty rank or a rank-local parameter no longer hangs or diverges. Scale-out: Slurm multi-node
launch and a REPACSS recipe (4 × H100 NVL per node, so 8 GPUs = two nodes), meta-device init with rank-0 broadcast,
FSDP CPU offload, node-local shard groups, a store-backed SPMD channel with heartbeats, resume on every rank,
checkpoint garbage collection. The verl, TRL, SkyRL and rLLM bridges are tested against the pinned upstream code
(none ran before); turn credit reaches verl through a `verl.plugins` entry point and `rm_scores`. QDQ keeps an fp32
master. Corrections: MiMo's T = 4 is Crucible τ = 3; AReaL's η is S − 1 and bounds capacity, not staleness; the
§9 items 25–28 rested on vacuous tests. Details: `docs/audit/2026-10-12/M12-train.md`.

## 14. Repository layout

Superseded by `ARCHITECTURE.md` §3 (project structure) and §4 (modules M0–M15), which also define the domain-pack layout that makes the core open-ended.

**Website (`site/`, 2026-10-11).** A one-page tour of the framework: a hero model, the pipeline as a clickable board, one stepped 3D figure per module (M0–M14, long horizons, the recursive agent, the task manager), the studio flow with its real CLI output, and results from the framework's own reports.
- **Engine (`assets/kit.js`):** one WebGL renderer draws every figure into its panel's rectangle (scissor and viewport) under a fixed canvas, with HTML labels projected on top. A figure is a spec: `setup`, `update`, and `steps` of `{say, go}`. Figures build on first sight and pause offscreen.
- **Framing:** each figure frames itself from the settled first step. Meshes, flow paths and measured label boxes are projected onto the camera plane, and the orthographic view fits them at any aspect ratio. Objects that appear later opt in with `userData.fit`.
- **Steppers:** jumping to a step (dots, back) replays the earlier steps instantly, so the figure always shows the state the story reached. Step timers (`v.later`) are cancelled on every step change.
- **Copy rules:** no other framework, environment suite or model is named; models are "frontier A/B" and "open model". Every number comes from a gate report or a recorded run.

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
| World traces recorded no `writes` | M4 cannot credit the agent (or peer) that made an edit | `writes` per edit call; fixed. Business-App calls (`tickets.update_ticket`, `erp.approve_invoice`) still wrote nothing until the M4 audit: they now record the records and fields they change |
| Docker Hub and RunPod key formats were not redacted; human actor ids were not pseudonymized | real credentials and identities in stored traces | M3 redaction; fixed |

**B.11 Found while building M5/M6 (2026-10-09).** Each is fixed, or recorded as a design limit, and covered by a test or a gate report.

| Defect / finding | Where it bites | Check |
|---|---|---|
| The editor classifier read JSON-string arguments as empty, so a `view` counted as an edit | segmentation over-counts edits | `test_workflow.py` |
| A keyword family table matched substrings ("round" in "stereo_ground", "stream" in "PointStream") | rediscovery reported 6/6 families that were not there | construct-only matching plus a hand audit (`m5_rediscovery.json`). The M5 audit found the "5 of 6" that followed still counted block names and added APIs and audited two families: 4 of 6 |
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
| `mismatch._quantiles` built its quantile tensor on CPU | **every accelerator trainer step crashed** at metric time (the staged M12 GPU gate would have died at step 1); CPU tests could not see it | tensor on the data's device; MPS offload tests now run the trainer on an accelerator; fixed |
| Offloading moved parameters and moments but a parameter-sized residue stayed on the device | `gc` reference cycles from the last step's graph pinned device tensors (31 % of the peak resident) | `gc.collect()` before emptying the cache; ≤ 5 % measured on MPS; fixed |
| **M12 GPU gate, first full run:** with transformers 5, `apply_chat_template(tokenize=True)` returns a `BatchEncoding` (a `UserDict`, so the `isinstance(dict)` guard missed it); the codec took its *keys* as prompt token ids | every real-tokenizer World rollout failed in record building; the runtime masked 228,384 rollouts in 600 s and trained 0 steps **without saying why** (CPU tests use the byte codec) | `return_dict=False` + Mapping check + int validation; test with the real Qwen3 tokenizer; every escaping exception recorded by blame (`infra` / `harness`) and kind with tracebacks; a circuit breaker (the M10 rules: masked > 20 %, harness bugs > 5 % of the last 200) stops a broken run with `stop_reason`; the gate fails on any stop reason but `steps`; fixed |

