# The Crucible task factory: the workflow for automated, rigorous creation of long-horizon coding RL tasks

**Scope.** Coding tasks only. Each task is long-horizon and wrapped as an RL environment, built by an agent swarm, with a
measured verifier and a sealed held-out benchmark.

**Sources.** Every rule below comes from one of these, cited by short tag:
- **Our studies:**
  - `swarm-pipeline/01` (Prime Agent, read at HEAD 7cbffa77b);
  - `02` (Sakana MASS);
  - `03` (THUDM slime);
  - `task-factory/01–11` (factory design and our runs).
- **The two articles:**
  - "How to make a good RL task" (`RLT`);
  - "Building RL environments for superintelligence" (`PM`).
- **The agentic-evals guide** (`EVALS`).
- **The agent-tracing survey** (`TRACE`).

**Status.** This is the plan to build against. Everything marked **exists** is on main (01a5627). Everything else is
built in §15's phases.

---

## 0. The goal, and the rules that never bend

**The product** is accepted long-horizon tasks per dollar, at a measured verifier error and a hack-free rate. A task is
accepted only when every gate below has passed **with executed evidence**.

**Invariants.** Each is enforced in code, not in a prompt:

| # | Invariant | Why (source) |
|---|---|---|
| V1 | **Ground truth first.** Every gating check cites a real observation: a real commit, test, fix or end state, executed. | the user's rule; `RLT`; task-factory/05 §1 |
| V2 | **Generation and verification never share a context.** Whoever writes the verifier never sees a solution. Reviewers come from another model family, enforced by config. | Rust rewrite (`swarm-pipeline/01` §1); `EVALS` L4 |
| V3 | **Grade what changed, not what was said.** The end state (hidden tests on the repo) comes first; the path second (protected paths, forbidden operations, step budget); the agent's text last, never on its own. | `EVALS` L1 |
| V4 | **The grader is out of the agent's reach.** It runs after the agent, outside its process, in a fresh container, on a pristine tree plus the agent's patch. Hidden tests and the specification are never in the agent's World. | `EVALS` L7: reachable graders gave 28.57% hacked "passes", unreachable ones 0.56% |
| V5 | **Don't lie to the model.** Everything the prompt asks for is checked. Nothing the grader cannot check is asked; it is made impossible in the World instead. | `PM`; task-factory/08 §3 |
| V6 | **The swarm proposes; executed gates decide.** No model verdict ever gates on its own. A judge is a second column with a published κ. | GitSwarm (`task-factory/01` §3.2); `EVALS` L4 |
| V7 | **Every loop exhaustion escalates; it is never `done`.** Prime's executor ends unapproved loops `done`, and a test pins it (`swarm-pipeline/01` fact 7). Ours escalates. | — |
| V8 | **Infrastructure failure is masked, never a 0.** This covers the World, a sandbox death and a provider error. Truncation is a limit stop, not a failure. | slime trains infra deaths as 0 (`03` fact 8); we do not |
| V9 | **The held-out benchmark is built and sealed first.** No factory role that writes training tasks ever reads it. | `task-factory/12` §0 |
| V10 | **Every model-calling stage has a dollar and token cap** in the USD ledger, and is off unless the run config enables it. | the user's budget rule; Prime has no $ budget (`01` fact 2) |
| V11 | **Every stage is journaled and resumable,** and every decision is stored with its reason. | `task-factory/05` §10 |
| V12 | **A process signal (trace labels, judge opinions) is never reward.** Reward comes only from the executed end state. | `PM`; `TRACE` (semantic enrichment drives improvement, not reward) |

---

**Realism vs environment awareness, reconciled** (`HUD` vs `PM`):
- **The request** is written in a real user's voice:
  - no benchmark framing;
  - no rubric, weights or success criteria;
  - real repos and real state.
- **A fixed, truthful harness note** (sandbox, offline, hidden checks exist, how to give up, STATUS.json) is kept apart
  from the request. It never states what is checked beyond "the behaviour you were asked for".
- The grader lives where the agent never interacts.

## 1. What a task is: the `LongTask` contract (one contract for every generator)

| Field / method | Content | Status |
|---|---|---|
| `world` | a workspace of 1..N repos at the start state; Docker image, offline, history cut, executed leak probes | exists (windows) |
| `prompt`, `requirements` | user-voice request; two-way ledger requirement ↔ milestone; lint; required public names stated | exists |
| `milestones` | hidden checks on the end state; `requires` edges **executed** (fail alone, pass with the prerequisite) | exists |
| `path_rules` | protected paths restored and checked, forbidden operations (test edits, network, history reads), step budget | partly (protected paths); **new** for the path checks below |
| `grade_patch(patch)` | the grader owns the build and the run; reward = 0.5·Φ_final + 0.5·[all gates]; GradeReport with per-milestone and per-path results | exists |
| `milestone_status(patch)` | per milestone, for the checkpoint sidecar | exists |
| `oracle()`, `floors()`, `mutants()`, `alternatives()` | executed ground truth: must score 1 / 0 / fail its milestone / 1 | exists (alternatives partial) |
| `simplify(kind)` | the difficulty-removed variant for the counterfactual control (§7) | **new** |
| `provenance` | ObservationRecords binding every check to a real observation | exists |
| `card` | TaskCard: intent, generator, axes coordinates, verifier card, yield record, split | exists (M17 cards) |

**Path checks** (`EVALS` L1; "lucky pass" = right end state, wrong path, 10.7% in AgentLens):
- no edits to protected or test paths;
- no network attempts;
- no reads of history or the grader;
- required actions where the task states them, for example "add a migration", checked from the diff;
- the step budget.

They are graded on the trail and the diff, never on the agent's account.

---

## 2. Ground truth and the generators (how a task becomes long)

Every generator emits a `LongTask`. **Long-horizon means intrinsic dependency, not length:**
- at least one executed `requires` edge;
- the concatenation and shortcut tests (§6).

| Generator | Ground truth | Long because | Status |
|---|---|---|---|
| **mined window** | a repo's own history from release A to B; per-commit fail-to-pass tests; the oracle is the repo at B | dependent real commits | exists (26 train / 10 held-out dependent windows) |
| **stitched window** | host repo + consumer repos; real changes in the host whose effects the consumers' tests check | the consumer milestones require the host change | new |
| **swarm-built window** | a real start repo + real requests: issues, PRs, changelogs of the repo or its ecosystem, found by scouts. The swarm plans a milestone DAG, then **builds the oracle commit by commit** in a worktree. Blind verifier writers write per-milestone tests; reviewers and hackers attack them | the planned DAG, with executed dependencies | new |
| **rebreak** | N real fix commits reverted in one tree. Each fault is a real bug, caught by the test its fix added | faults that interact through shared code | new |
| **open (Type B)** | a real metric (speed, memory, size) under the full test suite; the best known version proves headroom only | no ceiling: rewards doing better | new |

The **swarm-built window** is how tasks are made where no history fits:
- the swarm writes the "future" of a real repo;
- every milestone must then pass the same executed gates as a mined commit;
- the start repo and the requests are real, and the oracle is executed, so V1 holds.

**Scouts** (new, generic, driven by the intent card) search GitHub (code, commits, issues, PRs) with query expansion and
snowballing (forks, dependents, linked PRs), and produce typed source records:
- `real_change`;
- `real_fix`;
- `real_perf`;
- `host_repo`;
- `consumer`;
- `request` (an issue or feature text).

Filters run before any model call: licence, size, CPU build-and-test probe. The eval lease is taken first.

---

## 3. The swarm: topology, roles, information, communication

**3.1 Topology.**
- A **tree for control** (`SWARM`: with tens of agents a tree is fine). The root curator writes no tasks: it leases cells
  of the coverage matrix, monitors, merges accepted tasks, and talks to the human.
- A **shared task repo as memory**, a mesh (GitSwarm):
  - one repo per family;
  - immutable commits;
  - `informed_by=` links;
  - in-flight announcements with scope and abort criteria, so duplicate work is visible before it starts (GitSwarm:
    about 1 in 6 workers was redundant without them);
  - nominations are preferences, never verification.

**3.2 Roles.** The column "session" is the key change from v1. Builder roles are **agentic sessions in their own
worktree and sandbox**, not single calls (`swarm-pipeline/01` fact 10).

| Role | Session | Model family | Sees | Never sees | Produces |
|---|---|---|---|---|---|
| Root curator | persistent agent | strong, any | coverage matrix, yield ledger, decisions | task internals it doesn't need | leases, merges, escalations |
| Scout (many) | short agent | cheap | intent card, search results | eval store | source records |
| Intent librarian | persistent | reviewer family | failure clusters, trace catalogue | oracles | intent candidates (human approves) |
| Planner | agent | author family | intent, source records, World catalogue | eval store | milestone DAG, requirements, verifier plan, expected causes |
| World expert (one per repo, persistent) | agent + sandbox | author family | the repo at A, CI | B, hidden tests | Dockerfile, test command, leak probes |
| Oracle builder (swarm-built and stitched windows) | agent + worktree | author family | plan, the World | hidden tests | the oracle, **one commit per milestone** |
| Verifier writer (one per family, persistent) | agent + sandbox | author family, separate context | prompt, spec, intent, the World at A | oracle, solutions | hidden tests per milestone |
| Second verifier | agent + sandbox | reviewer family | prompt, spec | oracle, first tests | shadow tests (independent checks) |
| Prompt author | call | author family | spec, commit text, diff summary | tests | prompt + ledger |
| Realism reviewer | call | reviewer family | the maintainer text, the repo | oracle, tests | hint findings |
| Reviewer (adversarial) | agent | reviewer family | everything except the oracle | — | reasons the task is wrong |
| Hacker (one per family, persistent) | agent + World | reviewer family | prompt, verifier code, intent | oracle | exploit attempts → replay library |
| Solvers | agent in the RL harness | target + rungs | prompt + World only | everything else | trajectories |
| Auditor | call / agent | reviewer family | traces + environment evidence + verdicts | — | first wrong step, cause, defects |
| Tie-break | call | third family | the disputed item | — | decision |

**3.2.1 Families since 2026-10-23 (the Tinker swarm, `08` §1.1, `13`).** "Author family" became three families: the
oracle builder is DeepSeek, the planner and the first verifier writer are GLM (V2 by family), and the prompt writer,
World expert and mutant builder are Kimi. "Reviewer family" became the families nothing builds: Nemotron and
gpt-oss. The rule is checked per artifact: every artifact carries the families that produced it (`factory/provenance.py`),
and a checker refuses an input its own family produced. Builder sessions run on the M16 RLM harness with the checkpoint
sidecar, so a builder's Φ, per-turn credit and honesty are journaled like a student's.

**3.3 Communication rules** (`EVALS` L6):
- Hand-offs lose specification: prose costs about 8.5 points per hand-off, structured about 2.8. Sending the full
  original spec loses least.
- **Every hand-off is a typed artifact:** a JSON schema per port plus the full original spec attached, never a summary
  alone.
- Every intermediate artifact is stored, for attribution: hand-off loss, actor error or grader error.
- Messages between agents carry questions and answers; work products go to the task repo (Prime: "children write files
  for fan-in").

**3.4 Swarm hygiene** (GitSwarm measurements):
- **Herding guard:** no single parent idea or template may exceed 40% of a family's accepted tasks. Building on one
  adapter rose from 17% to about 90% in 9 h without it.
- **Abstention is free:** "nothing new to add" is a valid outcome.
- **Cheap evaluations misrank,** so promotion uses a funnel (cheap gate → full gate), never the cheap gate alone.

---

## 4. The per-task machine (states, guards, loop-backs)

Built on our `factory/machine.py`, which journals runs, caps dollars, escalates when a loop is exhausted, enforces the
cross-family rule and screens with a canary. It adds three things from Prime HEAD:
- **infra vs task failure:** a dead worker is re-checked after a short wait and respawned once for free, never counted
  as a task attempt (`01` fact 5);
- **an output that never captures fails its producer,** not its consumer (`01` fact 6);
- **a declared recovery posture:** every state that mutates a tree declares `recovery: reconcile | reset`, and the
  validator refuses a mutating state without one (`01` fact 6, made enforceable).

| # | State | Owner | Guard to advance (executed unless noted) | On failure → |
|---|---|---|---|---|
| S0 | Lease | root | the cell is open; no in-flight duplicate in the task repo | — |
| S1 | Source | scout / miner | the record passes the filters; the CPU probe passes | drop |
| S2 | Plan | planner | DAG well-formed; each milestone names its observation type; ledger draft | S1 |
| S3 | World | World expert | builds; existing tests run; leak probes fail as required; 60 clean reruns of the existing suite | S3 ×2, then escalate |
| S4 | Oracle | oracle builder (or the mined B) | each milestone commit makes its tests pass and keeps pass-to-pass tests passing | S2 or S4 |
| S5 | Verifier | verifier writer | one check group per requirement; **solvability from the clean World:** the oracle passes every check | S5 / S2 |
| S6 | Dependencies | deterministic | executed `requires`; ≥ 1 edge (or labelled short and routed out) | S2 |
| S7 | Prompt | author → realism → ledger → lint | ledger is two-way; required names stated; no ungradable asks; no hints | S7 ×3 → S2 |
| S8 | Floors & mutants | deterministic | null agent, start state, stubs and copy-original all score 0 (Φ ≈ 0); every mutant fails its milestone; alternatives pass | S5 |
| S9 | Second verifier | second verifier | shadow tests: pass on oracle and alternatives, fail on mutants. A disagreement is a defect assigned by the auditor | S5 |
| S10 | Flake | deterministic | 60 oracle grades at 2 concurrency levels; Clopper–Pearson bound ≤ 5% | S3 / S5 |
| S11 | Review | reviewer | no blocking finding (a finding becomes a guard the next time it repeats) | owner of the finding |
| S12 | Hack | hacker | every attempt scores 0; a confirmed hack becomes a replay for every task, and the task returns | S5 / S3 |
| S13 | Solve | solvers (§11 harness) | at least 2 policies × ≥ 5 attempts with checkpoint credit on | — |
| S14 | Audit | auditor (§8) | cause mix ≥ 70% skill gap; no World bug or spec defect; FP/FN = 0 | owner by cause |
| S15 | Target | deterministic (§7) | band; counterfactual control; prerequisite probe; paraphrase stability | S2 (wrong difficulty) / drop |
| S16 | Horizon | deterministic (§6) | concatenation and shortcut tests where powered; compaction-boundary metric recorded | S2 |
| S17 | Accept & publish | root | all of the above; family quotas (herding guard) | — |

**The funnel order is fixed:** cheap deterministic states (S6–S10) run before any solver tokens (S13). `RLT`: the alpha
lives in the recipe.

---

## 5. Verifier rigor (the grader is the product)

1. **State first:**
   - the hidden tests per milestone run on the grader-owned build in a fresh container (V4);
   - no exit code or stdout is trusted;
   - results come from a JUnit file at a random per-grade path (exists);
   - candidate code runs unprivileged.
2. **Path second:** the §1 path checks. **Text last:** STATUS.json claims are scored for honesty only (§11), never as
   evidence of work.
3. **Null agent and clean-World solvability** on every task, re-run whenever the task changes (`EVALS` L1: 2 of 20
   example tasks were unsolvable, and no pass rate showed it).
4. **Mutants:** commit-revert mutants (exist), plus mined fault classes for rebreak. **Valid alternatives:** solver passes
   confirmed by the second verifier, plus one model rewrite in a different style.
5. **The second verifier** comes from a different family, built from the prompt alone. Differential runs over every end
   state; each disagreement is a defect with an owner.
6. **Gold set per family:**
   - ≥ 36 items per class for a false-pass/false-fail bound ≤ 10%, ≥ 73 for ≤ 5%, with zero errors;
   - at least half are real solver outputs;
   - a human spot-checks 10%.
7. **Judges only for side axes** (prompt realism, report quality):
   - the judge gets the trail, the state diff and the spec;
   - a different family from the agent;
   - both orders, neutral labels (MASS's judge saw "baseline" vs "design-vK" labels: `02` fact 6);
   - κ vs the code grader or human published every run; below 0.6 the judge's verdict does not count (`EVALS` L4).

---

8. **Fairness rules** (`HUD`, "What is a good task?"):
   - **Functional checks only.** Never keyword or line matching. HUD's attention-fix grader rejected 4 working fixes because
     it checked for a variable name. The lint refuses checks that read source text where behaviour can be tested.
   - **The ten-competent-people test.** Every gating check must pass the oracle **and** every valid alternative. In S8
     this needs ≥ 2 independent alternatives, at least one built by another family.
   - **Only grade what the prompt asks for:** the ledger is two-way (exists). A check with no requirement is refused.
   - **A correct answer must never crash or trip the grader.** HUD's anti-hallucination rule flagged "ug/m3" as a file
     path. False-fail candidates from the audit are re-run on the alternatives before any check is kept.
   - **S9b, live (2026-10-23; `factory/alternatives_gate.py`, `lh_debug_target.py fairness`).** The alternatives are
     solver end states written from the bug report alone (stored frontier and student runs, plus S9b's own sessions
     by families other than the prompt writer's and the suite's), one per end state. Each is judged by the grader, by
     the blind second verifier (now in every debug-lane language: `verifier2` under the language profile) when its
     family differs from the state's, and by the report's repros. Classes: accepted, `false_fail` candidate (the grader
     rejects what every independent check passes: blocks the task until the audit resolves it), `lucky_pass` (a
     `verifier_gap` candidate), wrong, unknown. κ between the two verifiers is published per task; disagreements go to
     the tie-break. `lh_audit.py --fairness` feeds the accepted alternatives and the qualified second verifier into the
     trace audit's acceptance (the fakes are gone there).
   - **Path rules never punish confined work (2026-10-23).** `path:install` excuses an install in a scratch module
     outside the repo (`cd /tmp/r && go mod tidy`) when the project's manifests and lockfiles are unchanged and the
     World is offline: the mvdan-sh false fail (DeepSeek at Φ = 1 lost its pass). A tool installed on the machine
     (`go install`, `cargo install`, `npm -g`), `pip` anywhere, or an install in the repo still fails. A bare reinstall
     of declared dependencies (`pnpm install`, `npm ci`) is its own class, off by default (`allow_declared_reinstall`).
     `factory/repath.py` re-applies path rules to stored episodes; every path report carries `RULES_VERSION`.

## 6. Long-horizon guarantees

- **Executed dependency** (S6): textual conflicts are only ordering hints (run 2: 51 → 26 windows once edges had to be
  executed).
- **Test ownership:** each test belongs to the first unit after whose prefix it passes. **Import-tolerant grading**
  keeps per-test milestones when a test file imports names the window adds late (F8).
- **Concatenation test:** in-context milestone rates vs isolated rates. Report its power, and do not gate when it is
  unpowered (run 1: n = 4 detects only a 4/4 vs 0/4 split).
- **Shortcut probe:** no passing path in under half the oracle's steps.
- **Compaction is recorded per episode:**
  - milestones lost right after a compaction boundary vs elsewhere (run 2: 0.040 vs 0.022 per span, small n);
  - re-reads of the ledger after compaction (M16 health metrics).

---

## 7. Failure targeting: the task fails for the intended reason

A task is pool-eligible only if:
1. **solvable:** the oracle scores 1, the floors 0, every mutant fails;
2. **in band:** the target rung's F1 reward has spread across its group (otherwise GRPO and our turn credit get nothing);
   benchmark items may sit below band;
3. **cause mix:** ≥ 70% `skill_gap` on the intent's behaviour (auditor with seeded controls);
4. **counterfactual control:** the same policy on `simplify()` succeeds significantly more often (Fisher one-sided on
   success, or a bootstrap CI on the ΔΦ, with power reported). Simplify kinds:
   - prerequisites pre-applied (windows);
   - host change pre-applied (stitch);
   - faulty function named (rebreak);
   - bar at the baseline (open).

   If the policy still fails the simplified task, the failure is not on the intended axis, and the task goes back;
5. **prerequisite probe:** the policy can build the World and run the tests;
6. **paraphrase stability:** two paraphrases of the prompt (same requirements, same ledger) give a pass rate inside the
   original's CI (`EVALS` L2: 46% of meaning-preserving paraphrases changed generated code).

**Difficulty control** (swarm-pipeline/12; exists, 2026-10-23). Measured difficulty drives the generator:
- `factory.difficulty.control` turns a measured rate, the cause labels and the failure modes into one typed knob action: ease (K1–K6, or C1 a checkpoint start), harden (H1–H3), keep, sample_more, fix_task or budget_bound. Each action carries R-Zero's generator reward `1 − 2|p − ½|` and is journaled. Held-out tasks are refused.
- A cheap advisory predictor orders the expensive k=8 measurements. It never keeps or rejects a task.
- `conditional` separates full-task difficulty from p(m | prerequisites).
- Checkpoint starts (`simplify_units`, kind `checkpoint`) re-run every gate.

At training time, `train.curriculum.DifficultySampler` filters by online p̂ and runs the reverse curriculum.

---

7. **Difficulty is real, never contrived** (`HUD`):
   - Difficulty comes from work in the World: dependent edits across files, configs and tests, conflicting evidence. It
     never comes from a vaguer prompt, a stricter grader or an over-weighted criterion. The realism reviewer and the
     ledger refuse those.
   - **A hardness claim names its model, harness, scoring and k.** One success in a handful of runs is noise, so hardness
     is stated as pass^k with a CI.
   - Benchmark items aim at a strong-frontier pass@1 ≤ 50%, measured only with approval because it is paid.

## 8. Traces and the RLM trace-audit loop (the factory improves its verifiers and intents)

**8.1 What every episode records** (`TRACE`: no single instrumentation point sees everything, so we record at all of
them):

| Layer | Content | Status |
|---|---|---|
| Model | every turn's token ids, logprobs, sampling sent, usage, latency (TITO) | exists |
| Harness | every tool call with arguments, result and wall time; the M16 ledger (`history`, `OUT`); compaction segments; child agents | exists |
| Environment evidence | the patch hash and milestone vector at every checkpoint (`progress.jsonl`); the final diff; test reports; leak-probe results | exists |
| Grade | check-by-check results, path checks, the GradeReport, the verifier version | exists (path checks new) |
| Spend | tokens, $ and seconds per episode | exists |

**8.2 The audit pipeline**, run by an M16 RLM agent: the trace ledger is too long for one context, and the RLM reads it
through code.
1. **Facts first, no judge.** Masked → `env_flake`; limit stop → `budget`; path violations; reply-before-action (STATUS
   claims done with no milestone change).
2. **First wrong step** (`EVALS` L3): the earliest turn after which Φ never recovers, or the first divergence from passing
   episodes of the same task, located from the checkpoint series and diffs.
3. **Cause label** from the closed taxonomy:
   - every label quotes the trace verbatim (exists);
   - seeded controls under opaque ids, so an auditor that mislabels them is not trusted (exists; 69/69 so far);
   - judges over-blame the model and experts agree on the root cause only 16.2% of the time (`EVALS` L3), so a 10% human
     sample reports κ.
4. **Defect proposals**, typed:
   - `verifier_gap`: a lucky or false pass, or special-casing;
   - `false_fail`;
   - `spec_defect` / `unstated_format`;
   - `world_bug`;
   - `intent_drift`: failures cluster on something other than the intent.

**8.3 Acceptance of a proposal is executed, never an opinion.** This is the self-improvement rule.

| Proposal | Accepted only when |
|---|---|
| New or changed check (`verifier_gap`) | a **demonstrating end state** exists (the trace's patch) that the current verifier passes and should fail; the new check fails it, passes the oracle and every valid alternative, and keeps every mutant failing; flake stays ≤ 5% |
| Relaxed check (`false_fail`) | the end state is shown correct by the second verifier plus the oracle diff; the relaxed check still fails every mutant and floor |
| Prompt edit (`spec_defect`) | the missing fact comes from the failing test's own source and parameters, never an auditor's guess (D8); the ledger and lint pass; the counterfactual is re-run |
| World fix (`world_bug`) | the reproduction runs in the World; the fix passes S3 again |
| Intent change (`intent_drift`) | a cluster of ≥ 3 traces from ≥ 2 model families → an intent candidate; **a human approves** |

Every accepted change creates a new verifier/task version with lineage. Already-trained episodes are re-graded only
offline, for reporting.

**8.4 The auditor is audited:** seeded controls every batch, a human κ sample, and same-family judging forbidden.
Harness edits to the auditor itself go through `HarnessState.evaluate` (exists: a paired A/B on the same seeds,
alternating order, a noise re-run, guards).

**8.5 Audit the passes too** (`HUD`): QA agents read a frozen snapshot of every attempt, **perfect scores included**. A perfect score with a short path, files touched outside the task, or a path-check near miss is a `verifier_gap` candidate (lucky passes: `EVALS` L1).

**8.6 Trace datasets** (`TRACE`):
- failed solver traces become regression cases for the verifier (the gold set) and intent evidence;
- successful ones become alternatives;
- held-out traces return only intent-level counts (V9).

---

## 9. Measurement and reporting (every batch, every family)

- **Capability and reliability:** pass@k and pass^k with k = 1..5 from n ≥ 5 attempts, CIs that resample **tasks**, not
  episodes, broken out by category (generator × intent) (`EVALS` L2).
- **Φ distributions:**
  - final Φ;
  - milestones lost per episode;
  - compaction-boundary loss;
  - claim calibration (Brier, ECE).
- **Trail metrics** (`EVALS` L3): steps per episode, tool-error rate, refused or blocked change attempts,
  reply-before-action, empty searches.
- **Cost:**
  - **cost per passing episode** by category, with the max/min ratio across tasks (cost per episode rewards giving up);
  - seconds per episode;
  - factory cost per accepted task, per role.
- **Yield ledger:** candidates → each state → accepted, with the reason at each drop, per generator and family.
- **Verifier card per family:**
  - gold-set false-pass/false-fail bounds;
  - judge κ;
  - mutant catch rate;
  - flake bound;
  - confirmed hacks.

---

## 10. Benchmark first, sealed

1. The intent card is written from observed failures, with the human approving.
2. The **held-out benchmark** is built from sealed sources: repos, orgs and time windows after all training sources. It
   uses the same machine and the same gates, ≥ 20 tasks per intent, pass^k reported.
3. Training scouts and authors refuse sealed sources (`factory/sealing.py`, exists).
4. A rise on the benchmark after training is the transfer test, reported paired by task with the MDE (about 13 points at
   n = 30 when σ_d ≈ 0.25).

---

## 11. The RL environment wrapper (one task → a trainable environment)

**Native (exists):** `LongTask` → `WorkspaceHarness` or the M16 RLM harness, plus:
- the checkpoint sidecar: ΔΦ as `Turn.advantage`, through `turn_credit_map`;
- honesty: STATUS.json Brier score and GIVE_UP, added to the advantage, never to the reward;
- the tool-limit warning plus a final STATUS turn.

Infra failures are masked and truncation follows the limit policy.

**slime bridge** (new, Apache-2.0, DEPEND; `03` §"interface"). This is the scale-out trainer: Megatron, async, 27B+.
1. **A JSONL row per training task:** `prompt` (prompt + epilogue) and `metadata` with:
   - `instance_id`;
   - `image` (the World at A);
   - `workdir`;
   - `agent` (`claude_code` | `codex` | `crucible_rlm`);
   - `crucible_task_ref`;
   - `split=train`.

   Held-out tasks are never written.
2. **A sandbox provider:** our World implements slime's `Sandbox` (`exec`, `read_file`, `write_file`, async context,
   `SandboxTerminatedError` on infra death), with egress only to the adapter.
3. **A generate function** (`--custom-generate-function-path`):
   - opens the World;
   - runs the harness;
   - snapshots for checkpoints in an adapter subclass's `_run_turn`, before each model call;
   - calls `grade_patch` in a fresh container;
   - sets R = 0.5·Φ_final + 0.5·[gates];
   - attaches `turn_credit`, STATUS claims and give-up.

   Masked failures return removed samples (V8). slime would train them as 0.
4. **Credit hooks:** a custom convert function and advantage function carry per-turn ΔΦ, plus the one-line patch in
   slime's data-parallel split (`batch_builder.py:467-487` forwards only a fixed field list). slime itself has no step
   credit: the full reward goes to every segment (`trajectory.py:417-419`, verified).
5. **Filters:**
   - nonzero-std groups (exists in slime);
   - a **staleness cap**: slime only logs staleness (`03` fact 2);
   - fully-async rollout, `--use-tis`, partial rollouts.

**Port from slime** into Crucible's runtime, as Apache-2.0 PORT with attribution:
- the 4-method sandbox protocol;
- the Claude Code / Codex harnesses with the compaction hook (70%, max 2);
- the token-exact adapter;
- `TrajectoryManager` (message tree, compaction branches, re-tokenization drift handling), extended to export per-turn
  token spans.

---

**11.3 Portable task rows and the environment protocol** (`HUD` SDK docs, PATTERN; the licence is not checked, so no code is
copied):
1. **A task is a data row, not live code:** `(env name, template id, bound args, slug, split, runtime_config)`. One
   `LongTask` template (a generator) mints many rows: windows, seeds, simplify variants. The row is the unit published,
   synced, filtered and exported. Our `LongTaskRef` (P0) is that row, and JSONL is the codec, shared by slime's input and
   agentenv's `Task`.
2. **The verifier is a separate, agent-less task** that runs in its own environment: HUD's `Task.verifier`, our grading
   container (V4). Its result replaces the actor task's grade. The actor's environment never contains it.
3. **Capabilities are protocols, not tools.** The World exposes a shell over ssh (sandboxed, unprivileged `shell_uid`, no
   network) plus file tracking, so its diffs become environment evidence. Any harness then drives the same task: our RLM,
   Claude Code / Codex CLIs (slime), or a third-party agent. Extra services (a DB, a browser) are capabilities published
   from an `initialize` hook. A tool is never baked into the task.
4. **Two timeouts:**
   - an **agent timeout** ends the agent phase, and grading still runs on whatever work exists (a limit stop, graded;
     V8);
   - a **rollout timeout** bounds everything, and expiring it is infrastructure, so the attempt is masked.
5. **Groups:** if any sibling in a GRPO group is an infrastructure failure, **the whole group is dropped or refilled**.
   A partial group is never trained (HUD drops it; slime would train the dead rollout as 0). Groups whose rewards don't
   differ are filtered out (no spread, no signal).
6. **The objective is versioned with the checkpoint.** When the trace-audit loop (§8.3) changes a verifier or a task
   mid-training, the reward is a new objective. Training continues from a checkpoint taken **before** the change, or is
   reported as a new run. Rewards from two verifier versions are never mixed in one curve (HUD's checkpoint-tree
   rollback rule).
7. **Export adapters:** Harbor task folders (we have `interop/harbor`) and agentenv `Task`/`Eval`, so a published family
   runs in other harnesses unchanged.

## 12. The outer loop: the factory improves itself

1. **Decision store → criteria → guards** (exists in design): every human or gate decision carries a reason. Repeated
   reasons become named criteria, and each criterion becomes a guard state in the machine.
2. **Gate compilation:** human → agent on another family → classifier, swapped only when agreement on ≥ 50 held-out
   decisions passes the criterion's bar, and re-checked every N batches.
3. **Recipe search** (MASS pattern, Rust hill-climb discipline; `02`, `task-factory/01` §2.6):
   - **What it searches:** the factory recipe (role prompts, loop bounds, model per role, state order).
   - **The comparison:** champion vs a fixed reference, on frozen paired cells, in alternating order, with a noise
     re-run and an MDE.
   - **Executed fitness only:**
     - accepted tasks per $;
     - mutant catch rate;
     - zero confirmed hacks;
     - target-rung band hit rate;
     - gold-set false-pass.

     Never a model's own preference (MASS's weakness: `02` facts 6–8).
   - **No numeric target:** stop on marginal yield.
   - **Two reviewers on different families** per accepted recipe change.
   - **The champion moves only on a significant win.**
4. **Distillation of factory roles** (later): gate-accepted role trajectories, with the recipe text removed, train the
   cheaper factory models. The outcome is again judged only by executed gates.
5. **Exploit guard:** a factory change that raises yield must also pass the hack-replay library and a held-out family.
   The Prime paper saw an exploit persisted as a skill.

---

## 13. Verifying the factory itself (receipts)

Prime verifies its factory by running a real daemon against a scripted model and writing hashed receipts with an
explicit `not_verified` list. A twin test proves the receipt checker rejects wrong outcomes (`01` fact 9). Ours:
- **Seeded-defect runs:** a batch with planted defects:
  - an unsolvable task;
  - a reachable grader;
  - a leaky World;
  - a hinting prompt;
  - a flaky test;
  - a mislabelled cause.

  The factory must catch each one at its designated state. Receipt: the state, the evidence, the hash.
- **A scripted-solver run:** deterministic solver scripts (pass / fail / hack / partial) give expected GradeReports and
  credit series. Any deviation fails the release.
- Every run writes a receipt (versions, image digests, seeds, spend) and a `not_verified` list.

---

## 14. Spend and approvals

| Stage | Model calls | Default cap | Approval |
|---|---|---|---|
| Mining, World, dependencies, mutants, flake | none | $0 | — |
| Scouts | cheap | $3 per run | — |
| Plan, oracle build, verifier writing (swarm-built) | author family | $10 per 10 tasks | first run: yes |
| Review, hack, audit | reviewer family | $5 per 10 tasks | — |
| Solvers (S13, benchmark) | target + rungs | $15 per run | yes, every run |
| GPU (RL training, slime) | — | — | yes, every run |

Every report states the actual spend by role.

---

## 15. Build plan (phases with exit criteria)

| Phase | Builds | Exit criterion (executed) | Spend |
|---|---|---|---|
| **P0 Contract** | `LongTask` Protocol; `WindowTask` conforms; path checks; import-tolerant grading (F8); tool-output cap (D7); test-derived author findings (D8) | the existing window store grades identically; path checks catch seeded test edits and network attempts; all m17 tests pass | $0 |
| **P1 Machine v2** | infra/task split with one checked respawn; capture-failure ownership; the `recovery` field; agentic-session states (worktree + sandbox per builder role); typed hand-off artifacts with the full spec | the seeded-defect receipt run (§13) passes for the machine layer | $0 (scripted models) |
| **P2 Verifier rigor** | second verifier + shadow tests; valid alternatives; gold set per family; judge rules with κ | gold-set bounds computed on the 26 dependent windows; seeded verifier gaps caught | ≤ $5 |
| **P3 Trace audit loop** | RLM auditor over traces + evidence; first wrong step; typed proposals; §8.3 executed acceptance; versioning | seeded `verifier_gap` / `spec_defect` / `world_bug` traces become accepted fixes, and a fake proposal is rejected | ≤ $5 |
| **P4 Targeting** | `simplify()` per generator; counterfactual and prerequisite probes; paraphrase stability | on existing windows, a seeded tooling-only failure is caught by the counterfactual | ≤ $10 (approval) |
| **P5 Generators** | scouts; stitched, rebreak, swarm-built windows, open | each generator yields ≥ 5 tasks passing S0–S12 on one generic intent | ≤ $15 (approval) |
| **P6 Benchmark + env** | sealed held-out benchmark; native env publish; slime bridge (JSONL, Sandbox, generate, credit hooks, staleness cap) | a scripted-solver run through the slime generate path gives identical rewards and credit to native | $0 |
| **P7 Outer loop** | decision store → guards; recipe search with executed fitness; receipts per release | one recipe change accepted or rejected by the paired protocol with a receipt | ≤ $10 (approval) |

Order: P0 → P1 → P2/P3 (parallel) → P4 → P5 → P6 → P7. No solver spend before P2's gates exist.

---

## 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Graders that can't fail (null agent passes) | null agent on every task; path checks; V3 |
| Unsolvable tasks | clean-World solvability at S5; re-checked on every change |
| Reward hacking | V4; hacker state; replay library; scripted hack runs in receipts |
| Hard for the wrong reason | cause mix + counterfactual control + prerequisite probe |
| Homogeneous blind spots (MASS) | cross-family roles; executed fitness only; human κ samples |
| Herding and duplicates (GitSwarm) | 40% quota; in-flight announcements; free abstention |
| Hand-off loss | typed artifacts + full spec |
| Cheap-eval misranking | funnel promotion |
| Benchmark contamination | sealed sources; time split; contamination checks; intent-level counts only |
| Exploits persisted by self-improvement | executed acceptance (§8.3); hack replay + held-out family before any recipe change persists |
| Cost blowups | per-stage caps; cost per passing episode; funnel order |
