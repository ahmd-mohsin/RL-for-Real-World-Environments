# The task factory: design for review (M17)

Inputs: `01-prime-agent-factory.md` (Prime Agent's machine factory, the Rust-rewrite process, the swarm essay, GitSwarm),
`02-task-verifier-practice.md` (how 30+ sources make tasks and validate verifiers; requirements R1–R16), the article
"How to make a good RL task", the Preference Model blog, and M16.

## 0. The goal, stated so it can be measured

> **Maximise accepted tasks per dollar.** A task is accepted only when every protected axis holds and is measured against
> ground truth. A family is accepted only when training on it improves its intent on held-out orthogonal tasks.

Not the goal: task count, raw yield, or pass rate. GitSwarm and the Rust hill-climb both show that a numeric target
becomes a stopping point; the factory stops on marginal yield and diversity instead.

Three facts from the studies set the design:
1. **No source publishes human-labelled verifier accuracy** (no κ, no false-pass or false-fail rates with denominators).
   Expert review is claimed and never quantified. The factory must produce this evidence itself.
2. **"Verified" sets are still broken when someone looks.** Oracle and no-op sweeps found:
   - Terminal-Lego: 8.1% excluded;
   - BeyondSWE: 450/500 silent no-op oracles;
   - a workplace set: 13.5% no-op floor;
   - CodeMidas: solver-trace review found false passes or fails in 20.3% of tasks.
3. **Intent is free text everywhere.** Nobody tests whether a task teaches its named skill. Difficulty is often contrived
   (hidden-only rules), and orthogonality is measured only lexically.

## 1. The four parts of a task, each made an object

| Object | Fields | Why |
|---|---|---|
| **IntentCard** | id; behaviour (named skill); target failure mode; **evidence**: ≥ 1 ObservationRecord of a real model failing this way; protected axes; coverage coordinates (behaviour, surface); held-out probe set id | intent must come from observed failures, not imagined ones (user principle; R8, R9) |
| **ObservationRecord** | type (human-graded outcome, expert end state, real PR + tests, production trace, recorded session, observed model failure, reference-derived, simulator state); pointer; hash; date; who | every gating check must cite one; the last two types are second-class and never gate alone (R3) |
| **RequirementLedger** | requirement id ↔ prompt span ↔ check id(s) ↔ observation record | every sentence of the prompt is checked, and nothing is checked that the prompt does not state; a typed map, not an LLM read (R1; PM blog) |
| **VerifierGoldSet** (per family) | labelled solutions: real solver traces, requirement mutants (each violates one requirement), valid alternatives, hacks; labels by human or proven gate; κ between labellers | the verifier is trusted only through its measured error on this set (R2, R4, R5) |
| **FailureCause** taxonomy (closed) | skill gap, spec defect, unstated format, World bug, verifier defect, budget, env flake | a task is hard only when its failures are skill gaps (R7) |
| **TaskCard** | prompt, WorldSpec, CheckSet, oracle, IntentCard id, ledger, measured stats (floors, flake bound, pass rates with intervals, cause mix, horizon milestones), lineage, recipe version | the unit the factory ships |

## 2. The per-task machine (Rust-rewrite pattern; generation separate from verification)

A declarative, journaled, cyclic machine. It ports Prime Agent's MIT validator and guards. Exhausting a loop is
`escalate`, never `done`, which is Prime's defect. States, owners and what each one sees:

| State | Role | Sees | Produces | Guard to advance | On failure, back to |
|---|---|---|---|---|---|
| 1 Intent | Intent librarian (persistent) | failure-trace catalogue, coverage matrix | IntentCard with evidence | evidence is a real record; the coverage cell is open | drop |
| 2 Spec | Planner | intent, World catalogue | stated requirements, verifier plan, expected failure causes, horizon milestones | ledger well-formed | 1 |
| 3 World + prompt | Author | spec | WorldSpec, start state, prompt | prompt realism check: no hints a real user would not give | 2 |
| 4 Verifier | Verifier writer (separate context; **never sees an oracle or solutions**) | prompt, spec, intent | CheckSet bound to ledger ids | every requirement has a check; every check has a requirement | 3 or 2 |
| 5 Oracle | Oracle author (separate) | prompt, World | reference solution(s) | runs in the declared environment | 3 |
| 6 Solvers | ladder rungs, in parallel, ≥ 2 policies × ≥ 4 attempts | prompt only | traces | — | — |
| 7 Probes | Mutant builder | ledger, oracle | per-requirement mutants and valid alternatives | the verifier rejects every mutant and passes every alternative | 4 |
| 8 Hackers | family hacker (persistent) | prompt, verifier, intent | exploits | none scores > 0; each confirmed hack becomes a replay test for every task | 4 |
| 9 Second verifier | independent verifier writer, different model | prompt, spec | CheckSet′ | differential run on all end states: disagreements are defects | 4 |
| 10 Trace audit | failure auditor (different model) | solver traces + verifier verdicts | FailureCause per failing trace; FP/FN candidates | FP/FN = 0; failures mostly skill gap; no World bug | 3, 4 or 2 by cause |
| 11 Gate | deterministic acceptance (A1–A18) | everything | AcceptReport | floors 0; flake bound; long-horizon milestone tests | owner of the failed check |
| 12 Select | Judge (selector, never a synthesizer) | accepted variants of one seed | the kept variant | diversity quota | — |

- **Cross-model rules are enforced by config, not by a docstring.** The reviewer, auditor and second verifier use a
  different model family from the author.
- **Staging decides access, not prompts.** The verifier writer's context never receives oracle material; canaries and the
  zone scan check every message.
- **Cheap checks run first** (a funnel). Floors and the ledger come before solvers; solvers before trace audit.

## 3. The swarm around the machines

- **The root curator writes no tasks.** It owns the coverage matrix (behaviour rows × surface columns), assigns open cells
  to machines, leases work, watches yield and cost, and merges accepted tasks.
- **Many machines run in parallel.** Each task is independent, one context per machine, with short messages between
  stages. This is the easy case of the swarm essay ("parallel over context").
- **Persistent specialists,** asked by anyone through the forge repo and issue board (a mesh, not a tree):
  - **World expert,** one per World: knows its dynamics, answers authors' questions, reviews World bugs;
  - **family hacker:** remembers which exploits worked (`forge/specialists.py`);
  - **intent librarian:** owns the failure-trace catalogue;
  - **family test writer:** never ingests oracle material.
- **Diversity against herding:** collisions counted at announce time (GitSwarm's 1 in 6), and a per-cell quota.
  GitSwarm saw one adapter's share of commits rise from 17% to 90%.
- **A vote or nomination is coordination, never verification.** Only the gate decides.

## 4. Verifying the verifier (the core)

A verifier version is released for training only with these numbers on its card:
1. **Floors at 0:** no-op, empty action, constant answer, original state, copy of the reference (R6).
2. **Requirement kill rate = 100%** on executed mutants, and **pass rate = 100%** on valid alternatives (R2).
3. **Differential agreement** with the independent second verifier on every solver end state (Rust parity pattern).
4. **False-pass and false-fail rates** on the family's VerifierGoldSet, with denominators and Wilson intervals (R4, R5).
5. **Human κ:** a 10% sample of model-reviewed traces is labelled by a human; the model reviewer is trusted only above
   the agreement bar (proposed κ ≥ 0.7). Below it, the human stays in the loop for that family.
6. **Flake bound:** ≥ 10 oracle reruns at two concurrency levels; the card shows the Clopper–Pearson bound (R12).
7. **Hack library replay:** every confirmed hack from any family scores 0 (R13).

A rubric or judge change is a new version and is recalibrated. Versions never mix in one training pool.

## 5. Is the model learning the intended thing?

- **Before training:**
  - failure causes on solver traces are mostly skill gap (R7);
  - successful traces show the intended behaviour (a process audit, recorded, never reward);
  - the shortcut probe cannot reach the end state in fewer steps than the floor (R11).
- **At family acceptance (R9):** the base policy fails the held-out probe set; a policy trained on the family (or a cheap
  proxy, such as best-of-n distillation or in-context examples) improves on it. The probes share no World, prompt,
  template or verifier code with the family.
- **During training (R14):** zero-variance group share, always-pass / always-fail tasks, reward vs held-out divergence (halt
  on divergence), hack rate (alarm at 2%), behaviour audit (turns, tool mix, early stops).
- **On every policy upgrade (R16):** replay hacks, re-run floors, recompute difficulty; quarantine tasks whose failure
  cause shifts to spec or verifier defect.

## 6. Long horizon and orthogonality

- **The horizon must be intrinsic** (R11):
  - ≥ N milestones checkable from World state;
  - an order and non-redundancy test;
  - a **concatenation test:** if the milestones are independent short tasks, the task is not long-horizon, just long;
  - a shortcut probe;
  - recorded oracle step count and expert time;
  - per-step feedback only from public checks.
  The stitch operator builds long tasks from validated spans; M16 runs them.
- **Orthogonality, measured at four layers** (R10):
  1. prompt (lexical and semantic);
  2. World start state and oracle end signature;
  3. check-set Jaccard;
  4. **behavioural:** correlation of pass and failure-cause vectors across the calibration panel.

  A duplicate is the same intent and the same failure cause, not the same topic.
- **Coverage plan:** behaviour stays fixed while the surface varies. Each behaviour gets tasks across domains, Worlds,
  tools, data shapes and prompt styles, plus a held-out surface for the transfer test.

## 7. The outer loop: the recipe improves

- **Recipe = versioned machine definitions** + role prompts + operator settings + gate thresholds (`MACHINE.md`-style,
  exportable).
- **Decision store:** every human or gate decision is stored with its reason (keep, drop, revise, "the prompt hints at the
  file"). Reasons are clustered into **named criteria**; each criterion gets a guard in the machine.
- **Gate compilation:** an expensive gate (human → agent → classifier) is replaced by a cheaper one only when its agreement
  with the expensive gate on held-out decisions passes the bar. The bar is re-checked on drift.
- **Role hill-climb:** the Rust-rewrite loop. A backlog of hypotheses; paired, alternating-order A/B on the same seeds;
  a noise gate; neighbour-family regression checks; two reviewers on different models; no numeric target. The metric is
  accepted tasks per dollar at a fixed hack-free rate. An edit persists only after the hacker and a held-out family pass
  (the Prime Agent paper saw an exploit persisted as a skill).
- **Yield ledger:** candidates → controls → review → accepted, per authoring mode and family, rejected cost included. Yields
  in the sources range from 1.5% to 96% (R15).

## 8. What exists and what is new

| Exists (`forge/`, `accept/`) | New |
|---|---|
| DAG orchestrator, journal, USD ledger; `SwarmStage`; zoned repo; dedup L0–L3; ladder; learnability; tighten / compose / perturb / stitch / fuse; hacker + family memory; Judge (selector); Curator; YieldBook; A1–A18; `TaskFSM`; M16 harness | IntentCard + ObservationRecord + RequirementLedger + VerifierGoldSet + FailureCause; the declarative cyclic machine (port); the end-to-end driver and root curator; second verifier + differential; trace auditor; mutant builder keyed by requirement type; human-label store with κ; transfer probes; behavioural orthogonality; long-horizon concatenation and shortcut tests; decision store + gate compilation; role hill-climb |

**Main gap (study 01):** the built parts are exercised only by unit tests. No driver runs seed → swarm → accept → ladder →
curate → retire.

## 9. Build plan

1. **Objects and ledger:** IntentCard, ObservationRecord, RequirementLedger, FailureCause; the A7 extension that enforces
   the ledger both ways.
2. **Machine + driver:** port the validator and guards; the per-task machine of §2; the root curator over a coverage matrix;
   one family end to end on the toy pack, then finance_ap.
3. **Verifier validation:** mutant builder, second verifier, trace auditor, the VerifierGoldSet and κ reporting.
4. **Long horizon and orthogonality:** milestone, concatenation and shortcut tests; behavioural orthogonality; coverage
   matrix.
5. **Outer loop:** decision store, criteria, gate compilation, role hill-climb.
6. **Transfer:** probe sets and the proxy transfer test; a real training transfer needs GPU approval.

## 10. What needs the user

- **Your curation method** (you said you would specify it). It slots into §1, §2 and §7.
- **Ground truth:** which real observations we can use per domain (expert solutions, recorded sessions, real failure traces
  of frontier models), and **who labels** the first VerifierGoldSet. The first κ needs a human: you, or an expert you
  choose.
- **Models per role:** the cross-model rule needs at least two model families with API access, and that costs money.
- **Compute for transfer tests:** a small training run per family (GPU) or the cheaper proxies only.
