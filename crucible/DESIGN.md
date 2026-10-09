# Crucible — design document (v0.1, 2026-10-09)

Crucible turns **workflow traces that carry expert signal** into three products:

1. **RL environments** with dense, step-level rewards, through credit assignment.
2. **Tasks** at several horizons, calibrated so that frontier models fail.
3. **Benchmarks**, including out-of-distribution (OOD) ones, that score each step of a workflow.

The first vertical is **coding**, and the first workflow family is **legacy code modernization and migration of GPU kernels**: the family behind tasks 01–04 in `../migration_tasks/`.

This document is the contract for what we build. Where it borrows from existing frameworks, it names the file it borrows from (Appendix A). Where it encodes a lesson we learned the hard way, it names the incident (Appendix B).

Status: design only. No code exists yet. Section 13 lists the open decisions.

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
- **G1** Ingest coding workflow traces with expert signals into a typed trace store.
- **G2** Build a Workflow Graph for one task family (legacy GPU migration): stages, milestones, and the dependencies between them.
- **G3** Turn expert signals into dense rewards that are **outcome-anchored** and **cannot be farmed**.
- **G4** Synthesize 20–50 tasks at horizons H1–H4 for that family. Every task must pass the Acceptance Contract (§9), and frontier models must fail at a rate inside a target band.
- **G5** Run an RL experiment on an open model: sparse reward vs dense reward, with equal compute.
- **G6** Ship an OOD benchmark with step-level metrics, plus a version that is regenerated dynamically.

### 2.2 Non-goals (MVP)
- A new RL algorithm. We use vanilla GRPO (thesis principle "boring optimizer"); all novelty goes into data, environments and credit.
- Horizontal and non-coding connectors. These are designed for (§12.3) but not built.
- A hosted SaaS. Everything runs locally or in the customer's cloud account; nothing leaves the building.

### 2.3 Success criteria

These are the MVP gates. Each one is a measurement, not a judgement.

| ID | Criterion | Threshold |
|---|---|---|
| S1 | Replay fidelity: replaying the expert's actions in the environment reproduces the expert's end state and CI outcomes | ≥ 90% of sampled traces |
| S2 | Validity of the dense reward: the progress potential at step t predicts the episode's outcome (AUC on held-out traces), and beats an LLM-judge baseline | AUC ≥ 0.75 and > the judge's AUC |
| S3 | Every shipped task passes the Acceptance Contract | 100% |
| S4 | Frontier calibration: best frontier pass@1 on the benchmark split | ≤ 20% |
| S5 | Dense reward beats sparse on the RL run: success after N steps, at equal compute | statistically significant (bootstrap 95% CI) |
| S6 | OOD transfer: gain on the held-out repos / migration types / horizon split | > 0 with 95% CI, and no rise in the hack rate |
| S7 | Reward-hack audit: share of top-decile reward rollouts that fail a manual audit | ≤ 2% |

---

## 3. Design principles

The thesis's ten principles hold: the LLM proposes and records decide; rubric items must predict outcomes; experts audit rather than author; hindsight tasks first; generate at the frontier; grade with an agent; every defect becomes a check; train where the agent will live; boring optimizer; nothing leaves the building.

Eight more come from building tasks 01–04 and surveying nine frameworks:

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
```json
{"reward": 0.0,
 "gates": {"build": 1, "static": 1, "binary": 1, "correctness": 1, "concurrency": 1, "sanitizer": 1, "perf": 0},
 "outcome": 0.0,
 "progress": 0.86,
 "rubric": {"validated_items_met": 0.75},
 "efficiency": 0.4,
 "infra": {"status": "ok"}}
```
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

Per-turn advantage for multi-turn GRPO. Let A^GRPO be the usual group-normalized episode advantage. The turn-level advantage is:

  A_t = A^GRPO + η · ( Σ_{u≥t} r_u^sh − mean_group ) / std_group

This adds a normalized shaping-return-to-go. It can be injected through:
- verl's post-GRPO advantage operator: the fork's `verl/trainer/ppo/arvo_penalties.py` already rewrites turn-level advantages from `llm_turn_spans`;
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

### 7.4 Generators (all parameterized)

These follow Frontier Learning's procedural-generator idea: each generator's parameters form a search space.

- **G-hindsight:** replay a finished trace. Inputs as at the start, the answer hidden, graded against what really happened and held up (thesis principle 4).
- **G-span:** take a sub-span (i, j) of a trace, as in §7.3. Difficulty grows with the span length and the number of traps inside it.
- **G-trap:** inject a review-derived trap into the task's surface. Example: the task's consumer exercises the boundary condition a reviewer caught.
- **G-transplant:** apply the family's migration pattern to a *different* repo that has the same legacy construct, found by static search. The oracle is built and verified by the oracle builder against the original. This is the main source of **OOD** tasks.
- **G-compose:** union of compatible subtasks; harder through interactions (SWE-smith `bug_gen/combine/`).
- **G-tighten:** raise the verification strictness of a solved task. Examples: add perf bars, concurrency bars or sanitizer tools, which is what took task 04 from v1 to v3. Raising the bar is only allowed if the oracle still has ≥10% headroom.

**Learnability steering.** When generating RL training tasks, target a pass rate p ∈ [0.1, 0.7] for the *policy being trained*, using a proposer reward of 4p(1−p) as in verifiers `environments/proposer_solver`. When generating benchmark tasks, target frontier pass@1 ≤ 20%. Difficulty is adjusted with the generator knobs, never by making the instruction more obscure.

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

---

## 9. The Acceptance Contract (quality gate for every task)

A task enters the pool only if every check passes. The `crucible accept <task>` command automates all of them and writes `calibration.json`.

| # | Check | Requirement |
|---|---|---|
| A1 | Oracle | reward = 1 on ≥ 3 independent full grades; every perf bar has ≥ 10% headroom |
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

**Defect registry.** Every defect found in any task, by any audit, model run or reviewer, becomes a new automated check that A1–A12 run on *all* tasks from then on. The registry starts with the 14 defects in Appendix B.

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

---

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
| P3 Forge | 5–9 | swarm roles; generators (hindsight, span, trap, transplant, compose, tighten); calibrator; curator | 20–50 F1 tasks over H1–H4 with 100% A1–A12; frontier pass@1 ≤ 20% on the bench split |
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

**Open decisions for the user:**
1. The name ("Crucible").
2. The first open model and trainer for P4. prime-rl and verifiers v1 have the cleanest advantage hook; verl matches the MiMo recipes.
3. Whether to target the agentenv-framework or Harbor as the primary runner in P0 (both are supported via the canonical format).
4. Which public upstream repos form the F1 trace corpus.
5. The policy for using our own construction transcripts.

---

## 14. Repository layout (planned)

```
crucible/
  DESIGN.md                      # this document
  pyproject.toml
  crucible/
    schema/        trace.py task.py milestones.py card.py reward.py   # pydantic; JSON-schema export
    connectors/    base.py github.py ci.py vcs_history.py agent_sessions.py capture_proxy.py policy_docs.py
    workflow/      segment.py replay.py mine_milestones.py stages.py tracker.py
    credit/        potential.py defect_to_check.py branch_value.py validity.py prm.py compose.py tests/
    forge/         orchestrator.py roles/{miner,architect,env_builder,oracle,test_writer,adversary,auditor,calibrator,curator}.py
                   generators/{hindsight,span,trap,transplant,compose,tighten}.py
    verify/        patterns/{differential,hidden_tests,sanitizers,concurrency,fresh,perf,static}.py isolation.py calibrate.py
    accept/        contract.py registry.py checks/A01..A12.py
    runtime/       agentenv/{gpu_provider.py,separate_verifier_step.py} adapters/{harbor,verifiers,nemo_gym,openenv}.py
    train/         rollout_bridge.py advantages/{verl_hook.py,verifiers_hook.py} sft_export.py curriculum.py
    bench/         splits.py metrics.py dynamic.py irt.py report.py
    cli.py         # crucible ingest | model | credit | forge | accept | export | bench
  families/
    legacy_gpu_migration/        # workflow graph, milestone library, verifier patterns, seed tasks 01–04
```

---

## Appendix A — What we take from each surveyed framework

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

**Known defects in surveyed frameworks that we must not inherit:**
- SWE-smith instance branches keep the bug and test commits in history (`swesmith/profiles/base.py` `get_container` → `git checkout`).
- NeMo-Gym's history scrub is best-effort (logs on failure).
- OpenEnv's kernrl has no sandboxing inside the container and does nothing about timing noise.
- Harbor's pass@k assumes binary rewards.

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
