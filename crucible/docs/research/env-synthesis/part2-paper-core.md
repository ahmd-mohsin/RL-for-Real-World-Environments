# Part 2 paper core: synthesizing perfect RL tasks with an end-to-end, self-improving framework

**Date.** 2026-10-29. **Status.** The problem, the framework and the RSI loop the Part 2 paper presents. This is also the
build spec for `crucible.synth`.

**Relation to other docs.**
- `part2-task-env-synthesis-framework.md` is the evidence base: 337 papers, gates, statistics and roles.
- This document makes the framework formal and specifies the code. Where the two disagree, this document wins for the
  build.

---

## 0. The problem and the answer in one page

**The problem: perfect RL task synthesis.** Given any environment and a target skill, automatically produce many RL
tasks that are *perfect for a given student*, at a known cost, with a measured bound on how many are not.

A task τ is **perfect** for student π and skill σ when four properties hold:

| Property | Meaning | Measured by |
|---|---|---|
| **P1 Valid** | None of the eleven defects of §2 is present: the reward is sound, complete, non-vacuous, communicable, faithful to the instruction, unhackable, solvable and deterministic | The falsifier population and certificates (§5–§6) |
| **P2 Learnable** | π's success rate is in the learning band, p̂_π(τ) ∈ [0.1, 0.6] | Per-student measurement (§10) |
| **P3 On-intent** | Success requires σ: removing the skill's evidence or swapping the instruction breaks success | Taint, twins and intent probes (§5, §10) |
| **P4 Novel** | τ adds behaviour the pool lacks | Behaviour signature and coverage (evidence doc §6) |

Today's pipelines each optimise one piece:
- World synthesis;
- verifier hardening;
- difficulty filtering;
- diversity prompting.

They check validity with a two-sided filter, an LLM judge or student pass rates. None of them states the full target,
and none measures how far its output is from it.

**The answer, in three parts.**

1. **An end-to-end framework** that works on any environment that can be reset, snapshotted and read as typed state.
   Its stages:
   - a World interface and canonical state;
   - oracle programs with value tracking;
   - pass rules as data, run by one trusted grader;
   - a typed population of falsifiers that search for executed counterexamples;
   - per-task validity certificates;
   - a repair loop with a monotone counterexample archive;
   - composition, difficulty per student, intent probes and diversity;
   - online certification during RL.
2. **A recursive self-improvement loop over the whole pipeline.** It has three coupled loops:
   - the **falsifier** loop: counterexamples become new falsifier rules;
   - the **synthesizer** loop: counterexample–repair pairs become design rules for the task writers;
   - the **frontier** loop: student failures become targeted tasks.

   Its objective is measured, never judged: perfect tasks per dollar, subject to an estimated imperfect fraction ≤ ε.
3. **A label-free yardstick that makes the loop measurable and collapse-proof.** We plant defects of known class into
   certified tasks by mutating their pass rules. The falsifiers' recall on them calibrates every certificate into an
   estimate of the defects that remain.
   - An always-pass checker scores zero recall.
   - An always-reject checker raises false alarms on certified anchors.
   - Student reward never enters selection, so the loop cannot drift into either extreme.

## 1. Contributions

1. **Problem formulation.** Perfect RL task synthesis is stated as an optimisation: maximise perfect tasks per dollar,
   for a given student, subject to a *measured* bound on the imperfect fraction. The definition is P1–P4 above, P1
   decomposed into eleven typed defects (§2).
2. **The end-to-end framework** (§3–§12). It is domain-generic through a six-operation World interface. Within it:
   - every artifact a model writes is held to execution;
   - every check is cross-checked by another model family;
   - every admitted task carries a certificate.
3. **The RSI loop** (§9). Three coupled loops improve the falsifiers, the synthesizers and the frontier. They are
   selected on executed evidence, planted-defect recall and equal-budget controls, never on student reward. The
   hypothesis is that improvements transfer across domains and substrates.
4. **Measurement.** Error seeding plus capture–recapture turn falsifier-relative certificates into residual-defect
   estimates with intervals, per domain and defect class (§6, §8). This answers how good a task set is, not only how
   many defects were found.
5. **Evidence** (§13):
   - recall and residual-defect estimates on three or more substrates (SQL Apps, repositories, messaging, Gym);
   - loop improvement round over round against equal-budget controls;
   - cross-domain transfer;
   - cost per perfect task;
   - a dose–response RL experiment on Qwen3.6-35B-A3B: pools at increasing certificate strength against hack onset and
     held-out transfer.

**What we build on.** The framework reuses and credits mechanisms from earlier work. §15 lists the closest work for each
component. Several components exist separately elsewhere:
- closed-world diff grading (AppWorld 2407.18901, Agent-Diff 2602.11224);
- oracle edit operators (2610.09142);
- hacker–fixer loops (2606.08960);
- verifier evolution with a birth gate (2610.11464);
- composed sub-tasks (τ² 2506.07982);
- classic error seeding and spec mutation.

The contribution is the problem, the integrated framework that targets all of P1–P4, and the measured self-improving
loop. The parts that are, as far as the search found, new as mechanisms are value-tracked communicability (§5 TAINT,
TWIN), recall-calibrated certificates (§6) and cross-domain falsifier transfer (§9).

## 2. Failure taxonomy (what a bad environment is)

A defect class is a way the claim "this environment rewards exactly the intended skill" is false. Every falsifier in §5
targets one or more classes, and every planted mutant in §8 carries exactly one.

| Code | Defect | Example | Who suffers |
|---|---|---|---|
| D1 LAX-REQ | A required effect is missing from the check | Grader checks `status=on_hold` but not `hold_reason` | Policy learns a half task |
| D2 LAX-FRAME | Unrequested effects are not rejected | Agent also approves three other invoices and passes | Side-effect hacking |
| D3 LAX-VALUE | A value predicate is too loose | Any amount within 50% passes | Imprecise behaviour |
| D4 VACUOUS | A check is already true at the start | `status != rejected` holds at s₀ | Free reward |
| D5 HIDDEN | Success needs information the agent cannot get | A reason code appears in no doc, tool or instruction | Unlearnable noise |
| D6 OVER | A valid solution is rejected | Grader demands an exact tool order or float format never stated | False fails, bad gradients |
| D7 LEAK | The decision is given away | The instruction names the hold reason the agent should infer | Off-intent: no skill needed |
| D8 AFFORD | A shortcut affordance exists | A writable grader path, a `force_status` admin tool, a test file the runner trusts | Reward hacking |
| D9 INSENS | The contract ignores the instruction | Holding *any* invoice passes "hold invoice 42" | Off-intent |
| D10 UNSOLV | No tool sequence reaches success | The required value is not producible | Dead task |
| D11 NONDET | The verdict depends on run noise | Timestamps, ids, ordering, flaky tests | Noisy reward |

## 3. The formal model

### 3.1 The World interface (domain-generic)

A World is anything that implements six operations. `crucible.synth.iface.SynthWorld` is the protocol.

| Op | Signature | Notes |
|---|---|---|
| `reset(seed)` | → `StateRef` | Deterministic from the seed; G2 double-replay checks it |
| `tools()` | → `list[ToolSpec]` | Name, JSON schema, `read_only` flag, `effects` hint (collections it may write) |
| `call(tool, args)` | → `Obs` | JSON-able observation; errors are observations with `error` set |
| `snapshot()` / `restore(ref)` | | Optional; the fallback is replay from `reset` |
| `canon()` | → `CanonState` | Typed view of all reward-bearing state (§3.2); the fallback is observation-only contracts |

Adapters (§11) wrap: the Crucible `World`, with its sql, fs/git, json_doc, mailbox and doc_library adapters; agentenv MCP
envs, through their data plane; Gym simulators; and a pure-Python `PyWorld` for tests and for synthesized Apps.

### 3.2 Canonical state and diff

`CanonState` is a finite map from **addresses** to scalar values.

- **Address.** An address is `(app, collection, key, field)`. The key identifies an entity, not a position.
  - SQL: primary key, or a declared natural key.
  - Files: path.
  - Messages: message id.
  - JSON: a pointer prefix.
  - Gym: the observation index.
- **Normalizers.** Declared per App or per contract, applied before diffing:
  - `ignore_fields`, e.g. `updated_at`;
  - `id_canon`: inserted rows are keyed by their content hash, not by autoincrement ids;
  - `float_round`;
  - `order_insensitive` lists;
  - `text_norm`.
- **Diff.** `Δ(s, s′)` is a set of **changes** `(addr, kind, old, new)` with kind ∈ {insert, update, delete}. An update is
  per field, so "only `status` changed" is expressible.

The diff is exact and deterministic. Everything reward-bearing must be reachable through `canon()`; anything it cannot
see is narration and is never rewarded (principle 18 of the evidence doc).

### 3.3 Task, oracle program, contract

A **task** is τ = (W, seed, ι, actors, C, π*, I):
- ι is the instruction;
- C is the contract;
- π* is the oracle program;
- I is the **intent card**: the skill, plus which literals are *given* to the agent and which it must *derive*.

The **oracle program π*** is a straight-line program over tools. Each step is `id: tool(args)`. An argument is one of:
- `lit(v)`: a literal;
- `ref(step.path)`: a value read from an earlier observation;
- `fn(name, args)`: a pure function from a fixed library: `sum`, `min`, `max`, `round`, `len`, `concat`, `select(list,
  where)`, `get`, arithmetic.

An optional final `answer` is a template over refs. Straight-line suffices: π* is per concrete start state, so branching
is resolved at build time. This restriction is what makes mutation and taint tractable.

The **contract** C = (R, A, Q, M, N):
- **R, the required effects.** Patterns `(addr-pattern, kind, field predicates, count)`. A predicate is one of:
  - `eq`, `in`, `approx(tol)`, `regex`, `any`;
  - `rel(op, start_ref)`, a relation to the start state, e.g. `new.amount = old.amount`;
  - `from_answer`.
- **A, the allowed effects.** The same pattern language. Every entry must carry a `why`, e.g. "audit log rows written by
  any write tool".
- **Q, the answer checks.** Predicates on the final message: number within tol, set equality, regex.
- **M, the trace constraints.** `before(tool_a, tool_b)`, `never(tool-pattern)`, `at_most(tool, n)`,
  `read_before_write(addr)`.
- **N, the normalizers.**
- **Exec predicates.** For code domains, a black-box predicate `exec(cmd, expect)` may appear in R. It runs in a custody
  World and is falsified like any other check.

**The grader, the trusted base.** For a run ρ = (s₀, s_T, trace, answer):

```text
grade(ρ, C):
  Δ ← diff(N(s₀), N(s_T))
  for r in R:  if no subset of Δ satisfies r (incl. count): fail MISSING_REQUIRED(r.id)
  for δ in Δ:  if δ matches no pattern in R ∪ A:            fail FORBIDDEN_EFFECT(δ.addr)
  for q in Q:  if not q(answer, s₀, s_T):                    fail ANSWER(q.id)
  for m in M:  if not m(trace):                              fail TRACE(m.id)
  any exception, timeout or missing state                 → fail GRADER_ERROR (fail-closed)
  pass
```

Verdicts carry reason codes only. The policy never sees the contract, and builders never see the gate code. The
grader is one module of roughly 400 lines with exhaustive tests. Its correctness is a property of that module, not of
each task.

### 3.4 Contract inference

Executing π* from s₀ gives Δ*. `infer(Δ*, trace*)` emits a **skeleton contract**:
- R = Δ*, with each value as `eq`;
- A = ∅;
- Q from the answer template;
- M = ∅.

The LLM contract writer only **generalises** the skeleton. It turns `eq` into tolerances or sets where the instruction
allows, adds allowed patterns with a `why`, and adds trace constraints the instruction states. Every generalisation
widens the set of passing end states, so the falsifiers check each one (D1–D3, D6). This flips the usual pipeline. The
LLM no longer writes a verifier the oracle must happen to pass. It relaxes an exact verifier the oracle passes by
construction, and every relaxation is a falsifiable edit.

## 4. The two levels of truth

Falsification needs labels that do not come from the verifier under test.

| Level | Question | Ground truth | Cost |
|---|---|---|---|
| L1 implementation | Does `grade(·, C)` accept exactly the end states the task intends? | Execution semantics: equivalence of normalised diffs to Δ*, operator-specific label rules (§5) | CPU only |
| L2 specification | Do C, ι and W capture the intended skill? | Executed probes where possible (taint, twins, null agents, red-on-start); otherwise a cross-family adjudication panel calibrated on anchors, with PPI-corrected rates | Model calls |

The design rule: **push as much as possible from L2 to L1**. Taint (D5, D7, D9), twins (D9) and red-on-start (D4) turn
classic "judge" questions into executed checks.

## 5. The falsifier population

Each falsifier class implements `propose(task, ctx) → probes` and `label(probe, outcome) → {confirmed, suspect, clean,
equivalent}`.
- A **confirmed** counterexample needs no model.
- A **suspect** goes to adjudication.

| Class | Probes | Label rule | Targets | Cost |
|---|---|---|---|---|
| **MUT** program mutation | Operators on π*: DropStep, DupStep, Truncate(k), SwapEntity (an arg key → a sibling key of the same collection in s₀), PerturbValue (numeric ±ε·k, enum → another seen value), ExtraWrite (a write tool on a distractor entity with borrowed args), Reorder (adjacent independent writes), WrongAnswer | Run the mutant. If N(Δ_m) = N(Δ*) and the answer and trace constraints are equal, it is **equivalent** and must pass; failing is a confirmed D6. Otherwise it should fail. Survivors of DropStep(write), SwapEntity on a written key, ExtraWrite outside Δ* and WrongAnswer are **confirmed** D1/D2/D9. Other survivors are **suspect** D3. | D1 D2 D3 D6 D9 | ms per probe |
| **NULL** degenerate agents | No-op; read-everything; answer-only (the oracle's answer, no writes); write-everything (every write tool on every key π* reads); oracle-minus-last-write; random-k calls | Must fail; a pass is confirmed D4/D7/D8 | D4 D7 D8 | ms |
| **RED** red-on-start | Grade (s₀, s₀, ∅) per required pattern, and each R pattern against s₀ alone | A required pattern already satisfied at s₀ is confirmed D4 | D4 | ms |
| **TAINT** communicability | Re-execute π* with taint labels: each literal must be found in ι (normalised tokens), an earlier observation, or an `I.given` entry. Each R constant must trace to such a source or to a π* value. | Unsourced literal: confirmed D5. A `derive` literal sourced from ι: confirmed D7. | D5 D7 | ms |
| **TWIN** literal substitution | For each instruction-sourced literal ℓ, substitute a sibling value ℓ′ from s₀ through ι, π* and C, and re-execute π*′ | π* must fail C′ and π*′ must pass C′; a violation is confirmed D9. If π*′ crashes or cannot run, the instance is dead; record D10 for that instance. | D9 D10 | ms–s |
| **PERT** metamorphic | Add distractor entities (clones of the target with new keys and perturbed fields), shuffle row order, rename irrelevant text | π* (refs re-resolved) must still pass, and NULL must still fail; a flip is confirmed D11/D6 | D6 D11 | s |
| **REP** determinism | Run π* twice from `reset(seed)`; restore and run again | Any verdict or Δ difference is confirmed D11 | D11 | s |
| **ALT** alternative solvers | k agents from ≥ 2 families that built nothing in the task, solving from ι | Pass → a clean trial for D6. Fail → adjudicated: a valid solution means D6, otherwise clean. Pass with a forbidden-looking trace → suspect D8. | D6 D5 D10 | $ |
| **ATK** attackers | Agents told to pass without doing the skill, given the attack-strategy library (§9) and the instruction, never the contract | A pass is suspect; the panel decides D7/D8, and confirmed hacks also become negatives | D7 D8 | $ |
| **ONL** online rollouts | Every passing training or eval rollout | L1 hack rules plus the trajectory monitor flag passes; flagged passes are suspect, and a confirmed one quarantines the task | all | free |

**Adjudication.** A panel of two families, neither of which built the task, reads:
- the task;
- the intent card;
- the probe's trace;
- the reason codes.

It never reads the gate code or which falsifier made the probe.
- The panel is calibrated on anchors (trust-or-escalate λ).
- Below λ, the probe goes to the human queue.
- Adjudicated rates are reported with PPI intervals.

## 6. Validity certificates

For a task τ and falsifier class k:
- n_k is the number of independent probes;
- c_k is the number of confirmed counterexamples.

The certificate is the vector of (n_k, c_k, U_k), where U_k is the exact Clopper–Pearson 95% upper bound on the
counterexample rate *under class k's probe distribution*. At c = 0, U ≈ 3/n (the rule of three).

**Admission.** A task is admitted when:
- the deterministic classes (MUT, NULL, RED, TAINT, TWIN, REP, PERT) have c_k = 0 after repair, with U_MUT ≤ 0.05
  (n ≥ 59 mutants; the operators generate 40–150 per task);
- ALT has at least one pass from a non-builder family;
- ATK has 0 confirmed hacks over its budget.

**What a certificate means.** The certificate is honest about its reach. It bounds the counterexample rate under the
falsifier distribution, not under all possible agents. That is why falsifier strength is measured on planted defects
(§8), and why the paper reports both numbers:

> certificate bound U_k (per task) **and** falsifier recall FR_k,d (per falsifier class × defect class, on locked
> planted environments).

Together they give an estimate: P(a task with defect d is admitted) ≈ Π_k (1 − FR_k,d) for independent classes, and it
is measured directly on the locked set.

**Recall calibration: from certificates to residual defects .** Planted defects play the role of seeded errors.
For a domain and defect class d, write:
- FR_d for the population's recall on locked planted environments of class d;
- n_d for the number of natural defects of class d confirmed among candidate tasks.

**The error-seeding estimate (Mills).**
- Total natural defects: N̂_d = n_d / FR_d.
- **Residual defects among admitted tasks:** n_d (1 − FR_d) / FR_d.
- The interval comes from the binomial uncertainty of FR_d (delta method or bootstrap).

**The capture–recapture check.** Take two falsifier classes A and B that both target d and share no mechanism (e.g. MUT
and ATK for D2). Their overlap m on natural defects gives the independent estimate N̂_d = n_A n_B / m
(Lincoln–Petersen, Chapman-corrected).

**Both estimates rest on an assumption the paper tests.** Planted defects must be as detectable as natural ones. The
test compares FR on the planted set with the detection rate on the real-defect set (§8), per class. A class where the two
diverge is reported as uncalibrated.

**Per-domain reporting.** The paper reports, per domain, the estimated residual defect rate among admitted tasks with
its interval. Agent-environment papers report found defects only. This estimate is what makes "falsifier-relative"
certificates usable.

## 7. The per-task loop (CEGIS)

```text
synthesize(cell, domain):
  task ← propose(cell, World slice, lemma bank)            # task proposer (family A)
  π*   ← build_oracle(task)                                # oracle programmer (family B), executed; dead → drop
  C    ← generalise(infer(run(π*)))                        # contract writer (family C ≠ B)
  X    ← ∅                                                 # this task's counterexamples
  for round in 1..3:
      probes ← cheap falsifiers (REP, RED, TAINT, NULL, MUT, TWIN, PERT) ∪ archive.regressions(domain)
      X_new  ← confirmed(probes) ∪ adjudicated(suspects)
      if X_new = ∅: probes ← paid falsifiers (ALT, ATK); X_new ← …
      if X_new = ∅: break
      X ← X ∪ X_new; archive.add(X_new)
      (ι, C, π*, W-patch) ← repair(task, X_new)            # repairer (family ≠ writer); sees counterexamples and
                                                           # reason codes only; may not delete archived regressions
  certify(task) → admit | reject(reason)
```

- **Monotone archive.** No repair may make an archived counterexample pass again. Every version of every task of the
  domain must reject every counterexample recorded for it, re-run as a regression suite.
- **Round cap.** At most 3 rounds, unless new executed information arrives (evidence doc §9.2).
- **Equal-budget control.** For every factory claim, an untargeted batch is generated at the same spend.

## 8. Contract mutation: planting defects with known labels 

**Contract and World mutation operators.** Each operator is applied to a certified task and plants exactly one defect
class:

| Operator | Plants | Mechanism |
|---|---|---|
| `drop_required` | D1 | Remove one R pattern |
| `widen_frame` | D2 | Add `allowed: (app, coll, *, *)` with a plausible `why` |
| `relax_value` | D3 | `eq` → `any`, or tolerance × 10–100 |
| `vacuous_required` | D4 | Add an R pattern that s₀ already satisfies |
| `hide_info` | D5 | Delete the instruction sentence carrying a literal π* needs |
| `leak_decision` | D7 | Put a `derive` value into ι ("…because of a price variance") |
| `plant_affordance` | D8 | Add a write tool that sets the target field directly, or expose a grader-read resource for write |
| `over_constrain` | D6 | Add an unstated trace or format constraint |
| `instruction_drift` | D9 | Make R match any sibling entity (key → `*`) |
| `break_solvability` | D10 | Require a value no tool can produce |
| `inject_nondet` | D11 | Make a tool return a time- or random-dependent value that the contract checks |

**Two planted sets.**
- **The dev set** is used by the RSI loop.
- **The locked set** is never read by any loop. It is built with held-out operator *variants* (different phrasing,
  different collections) on held-out domains.

A third, **real** defect set is collected from history:
- the debug lane's documented measurement-time failures: unstated rules, undescribed faults, stale tests, changelog
  hints, fake links;
- every confirmed counterexample from ONL.

The paper's headline numbers are on the locked and real sets.

**Falsifier recall.** FR_k,d is the fraction of planted environments of class d on which falsifier class k finds a
confirmed counterexample. Its pair is the **false-alarm rate** FA_k: confirmed counterexamples on certified anchor tasks.
Each such alarm is either a real defect, which is adjudicated, kept as a new finding and its anchor repaired, or a
falsifier bug.

## 9. The regularised recursive self-improvement loop

A loop that proposes edits and keeps whatever scores best is not RSI. It is hill-climbing on a reused evaluation, and it
overfits that evaluation. RRSI measured this directly. The unregularised harness loop had the best evolve-set score and
no out-of-distribution gain. With regularisation, out-of-distribution score was 43.6 at 2.42M tokens, against 40.3 at
3.80M without it, and a higher evolve score went with lower transfer in every arm (2609.24972). Unguarded verifier
evolution collapsed to always-pass while downstream scores stayed flat (2610.11464). This section defines what improves,
what it is measured on, and the regularisers that make the improvement real.

### 9.1 What improves: the system state Θ

| Component | Contents | Improved by |
|---|---|---|
| **S, the synthesizer** | Role prompts; the lemma bank; recipes; source weights; cell quotas | The synthesizer loop |
| **F, the falsifiers** | Typed operators; attack strategies; adjudication prompts | The falsifier loop |
| **P, the planter** | Contract-mutation operators that plant defects | The planter loop (§9.3) |
| **K, the frontier controller** | Knob policy; ladders; targeting | The frontier loop |
| **A, the abstractor** | The prompts and procedure that turn counterexamples into operators and lemmas | The meta loop (§9.4) |

Every component is versioned data, and any version can be restored with a plain write.

### 9.2 The objective

Per round r, at budget B:
- K_r is the number of admitted tasks that satisfy P2–P4;
- ρ̂_r is the recall-calibrated residual-defect fraction (§6).

The objective is **J(Θ) = K(1 − ρ̂) / B**, perfect tasks per dollar. Every quantity is measured by execution, by planted
defects with witnesses, or by equal-budget controls. Student reward, visible pass rates and chains of thought never enter
J.

### 9.3 Why it is recursive: three levels of evaluation, each improved

```text
level 0   tasks                 evaluated by  F (falsifiers)               -> certificates
level 1   falsifiers F          evaluated by  P (planted defects)          -> recall FR, false alarms FA
level 2   planter P             evaluated by  witnesses + the real-defect set -> validity, natural-likeness
level 3   abstractor A          evaluated by  the birth-gate pass rate of what it produces next round
```

**The co-evolution game.** F and P play a regularised two-player game:
- P is rewarded for planting defects that the current F misses.
- F is rewarded for catching them without false alarms.

Three constraints keep P honest:
- **Witness validity, executed.** Every planted contract C′ comes with a witness run that passes C′ and fails the
  original C. A planted "defect" without a witness is not a defect.
- **Natural-likeness.** Planted defects are described by features: class mix, diff size, collection, which literal
  moved. Their distribution must stay within MMD ≤ μ of the real-defect set. Without this, P drifts into odd defects
  that no real pipeline makes.
- **Fictitious play.** F must keep recall on the union of all past planters' sets, not only the newest. This is the
  monotone archive applied one level up, and it prevents cycling.

### 9.4 The update rule and its regularisers

A proposed update u: Θ → Θ′ touches one component. It is accepted only if **every** rule below holds. The rules are
non-compensatory: a gain on one cannot buy a loss on another.

| # | Regulariser | Rule | What it prevents |
|---|---|---|---|
| R1 | **Fresh evaluation, never reused** | Each decision runs on a freshly sampled validation view: new proposals, plus a newly planted dev batch drawn from the current and all past planters. The locked set is read only for reporting. | Adaptive overfitting to a reused evaluation set (cf. the reusable holdout, Dwork et al. 2015; RRSI) |
| R2 | **Noise floor and lower bounds** | δ is measured before the loop from ≥ 3 repeated runs of the unchanged Θ, and is never tuned. Accept only if the lower confidence bound of J(Θ′) − J* exceeds δ, where J* is the best so far. The floor never moves down. | Chasing noise; the winner's curse |
| R3 | **Hard non-regression** | Every archived counterexample is still rejected. FA on anchors ≤ δ_FA. No defect class's recall FR_d drops beyond its noise band. The P3 on-intent share and the P4 coverage do not drop. | Trading one defect class for another; forgetting |
| R4 | **Trust region on the task distribution** | KL(q_Θ′ ‖ q_Θ) over the cell histogram ≤ κ, and entropy H(q_Θ′) ≥ H_min | **Certification Goodhart:** the synthesizer drifting toward tasks that are easy to certify (tiny, read-only, single-write) instead of tasks that teach |
| R5 | **Sparsity and cost** | The edit budget anneals from 3 to 1 per round on a cosine schedule (RRSI). An operator must pay for itself: its marginal recall per probe-dollar must be ≥ β. Components with no marginal gain over W rounds are pruned. | Library bloat; untestable bundles of edits |
| R6 | **Leakage critic and family separation** | Before scoring, a different family reads the diff and rejects edits that encode task ids, entities, values, planted-defect identifiers or anchor content. The abstractor is never a family that built the source tasks. | Memorising the yardstick; self-grading |
| R7 | **Transfer requirement** | The gain must also appear on at least one domain that did not produce the update's evidence | Single-domain overfitting; this is also the test of the transfer hypothesis |
| R8 | **Round caps without new executed information** | A component whose acceptance rests only on adjudicated (judge) evidence stops after 2–3 rounds unless new executed counterexamples arrive (Mind the Gap, 2412.02674) | Self-verification saturation |
| R9 | **Equal-budget control rounds** | Each accepted round is compared against a control round at the same spend with u removed | Crediting the loop for what more budget would have bought |
| R10 | **Rollback** | Every accepted update is reversible by restoring a stored version. A later regression detected on a fresh view rolls back to the last good Θ. | Irreversible drift |

### 9.5 The loops in this frame

**The falsifier loop (F).** Confirmed counterexamples are clustered by:
- reason code;
- operator;
- diff signature;
- tools.

The abstractor A turns each cluster into a typed operator or an attack strategy.

The **birth gate** is R1–R7 applied to a single operator. The operator must:
- reproduce its source counterexamples;
- find planted defects on fresh dev views of *other* tasks and domains;
- raise zero false alarms on anchors;
- stay under its cost bar.

**The planter loop (P).** P proposes new mutation operators or new parameterisations, rewarded by F's miss rate. A
planter operator is accepted under witness validity, natural-likeness and R6. Accepted planters join the population.

**The synthesizer loop (S).** A (counterexample, repair) pair is abstracted into a lemma for the proposer, contract-writer
and repairer prompts. A lemma batch is accepted when the first-round confirmed-counterexample rate falls and J rises,
under R1–R9.

**The frontier loop (K).**
1. Executed failure attribution, by snapshot replay, finds where student traces go wrong.
2. Those attributions drive targeted tasks and knob changes.
3. Targeted variants are capped at 20% of the pool; originals are never retired.
4. Every targeted batch is compared against an equal-budget untargeted batch.

R4 also applies here, so the curriculum cannot collapse onto one failure mode.

**The meta loop (A).** The abstractor's own prompts are edited by the same rule. Its J-proxy is the birth-gate pass rate
and the marginal recall of the operators and lemmas it produces in the next round. Recursion stops at this level. Deeper
levels add no new executed signal, so R8 caps them.

### 9.6 What the paper reports about the loop

| Report | Content |
|---|---|
| J_r per round | With its interval, against equal-budget control rounds |
| Locked-set measures per round | FR, FA, and ρ̂ |
| Transfer | Results on domains that produced none of the evidence |
| Ablations | One regulariser removed at a time (R1, R3, R4, R5, R7, R8, and the planter's natural-likeness), at equal budget |
| Failure cases | Overfitting, Goodhart drift and collapse, each documented when it happens |
| Ledger | Every rejected update, kept as negative evidence |

## 10. Difficulty per student, and composition 

**Composition.** seq(τ₁, τ₂) starts τ₂ from s_T(π₁*):
- the program is π₁*; π₂*;
- R = R₁ ⊕ R₂, where a later write to the same address supersedes the earlier one;
- A = A₁ ∪ A₂.

The instruction is rewritten and re-checked by TAINT. par(τ₁, τ₂) requires disjoint address sets.

**Dependency depth.** dep(τ) is the longest taint chain whose edges cross a sub-task boundary: a literal in π₂* sourced
from an observation of state that π₁* wrote. A composition with dep = 0 is two tasks in a row. dep ≥ 1 is a task where
later work genuinely needs earlier results.

**Knobs.** Each knob is a typed transformation with a predicted direction:
- dependency depth;
- horizon;
- distractor count (PERT machinery);
- observation hiding (pagination, an archive App);
- an actor that holds required information (the agent must ask);
- derive-vs-given literals (intent card).

A knob change is kept only if the behaviour signature moves and the student's p̂ moves in the declared direction
(principle 15).

**Band.** The keep band is [0.1, 0.6] on the student, with predictor-first sequential measurement (k = 4 → 8). The
DifficultyRecord is keyed by (task, student, checkpoint, simulator, time mode).

## 11. Generality: which environments the framework covers

| Environment class | `canon()` | Oracle | Contract | Status |
|---|---|---|---|---|
| SQL-backed enterprise Apps (finance_ap, helpdesk tickets, CRM-like) | Rows by primary key | Tool program | Effects over rows | Primary domain for M1–M4 |
| File trees and repos (debug lane) | Path → sha, mode, text | Patch application as a program (write_file steps), or the repo's fix commit | Effects over files plus `exec(test)` predicates | M5 adapter (`PatchTask`) |
| Mail, chat, docs (Slack/Notion-like) | Message and page records | Tool program | Effects over messages | Through PyWorld Apps (M4) |
| Web-app backends (AWM-style FastAPI) | Backend DB rows | API program | Same as SQL | Through `openenv/awm_service` (P1) |
| MCP / agentenv envs | `data/get` dump | Tool program | Same | Through `interop/agentenv/mcp_env` (P1) |
| Gym simulators | Observation vector by index | Action sequence (or a scripted policy) | Region and constraint predicates with tolerance; MUT becomes action-noise operators | Toy demonstration (M6) |
| Black-box (no state access) | Final observation fields | Tool program | Observation predicates only; frame checks unavailable, flagged in the certificate | Fallback |

**Graceful degradation.** A certificate records which falsifier classes were *applicable*. A black-box environment gets
no D2 guarantee, and the certificate says so.

## 12. Code architecture (`crucible.synth`)

**Layer.** `crucible.synth` sits beside `crucible.forge` in the import-linter layers. It may import `runtime`, `verify`,
`world` and `core`, and never `factory`. LLM roles are injected as a `RoleClient` protocol, which `factory/synth_wire.py`
binds to `ModelBook(preset="swarm")` with the family rules.

| Module | Content | Reuse |
|---|---|---|
| `synth/canon.py` | `Address`, `CanonState`, `Change`, `diff`, normalizers, adapter registry (`canon_sql`, `canon_fs`, `canon_json`, `canon_mail`, `canon_gym`) | `world/state/*` dumps (`q_dump`) |
| `synth/iface.py` | `SynthWorld` protocol; `PyWorld` (in-process Apps over dict tables, for tests and synthesized Apps); `CrucibleWorldAdapter` | `world/world.py` reset, snapshot, restore, `_call_tool` |
| `synth/program.py` | `Program`, `Step`, `Arg` (lit, ref, fn), `run(world, program) → Run` with a taint map, the pure function library | — |
| `synth/contract.py` | `Contract` (pydantic), pattern and predicate language, `grade(run, contract) → Verdict` (the trusted grader) | `verify/answers.py` predicates where they fit |
| `synth/infer.py` | Skeleton contract from an oracle run | — |
| `synth/falsify/{base,mut,null,red,taint,twin,pert,rep,agents,online}.py` | The nine classes plus the registry | `accept/mutation/actions.py` patterns |
| `synth/adjudicate.py` | Panel protocol, λ calibration hook, human queue | `factory/goldset.py` anchors (through injection) |
| `synth/certificate.py` | Clopper–Pearson, admission policy, certificate JSON | `factory/flake.py` bound |
| `synth/archive.py` | Hash-chained counterexample archive, regression runner | `factory/labels.py` style |
| `synth/cegis.py` | The per-task loop | — |
| `synth/meta/{plant,recall,library,lemmas}.py` | Contract mutation, the FR/FA matrix, the operator DSL plus birth gate, the lemma bank | — |
| `synth/compose.py` | seq and par composition, dependency depth | `forge/compose.py` ideas |
| `synth/domain.py` | `DomainSpec`: Apps, tools, skills, knobs, intent-card schema, cell grid | — |
| `synth/roles.py` | Role prompts (data) and the role → family constraints | `factory/swarm_prompts.py` |
| `factory/synth_wire.py`, `scripts/synth.py` | Binding to Tinker, the CLI (`propose`, `certify`, `plant`, `recall`, `evolve`, `report`), spend caps | `factory/models.py`, `factory/tinker.py` |

**Storage.** Data lives under `crucible/data/synth/<domain>/`: tasks, certificates, the archive, planted sets, spend.jsonl.
The locked planted set lives under `crucible/data/sealed/synth-locked/` behind the seal guard.

## 13. Experiments (what the paper shows)

| ID | Question | Design | Metric | Budget |
|---|---|---|---|---|
| E1 | Does the falsifier population detect planted defects, and which classes catch what? | FR/FA matrix on the locked set across 3 domains | FR_k,d with Wilson CIs; FA | CPU plus adjudication calls |
| E2 | Does evaluator-side RSI raise recall without false alarms? | Loop 1 for R rounds; locked-set FR per round; equal-budget control: random operator proposals | FR curve; FA; ledger of rejects | Low $ |
| E3 | Do operators transfer across domains? | Library grown on domain A, tested on locked planted sets of B and C | Transfer FR vs B's own library | Low $ |
| E4 | Do certificates predict real hacking? | Run the student (eval, then RL) on admitted tasks; ONL flags vs certificate strength | Rank correlation of U and FR-weighted strength with the confirmed hack rate | Student rollouts |
| E5 | Does certified data train better? (dose–response) | RL at equal budget on pools from the same generator, admitted at increasing certificate strength: (a) two-sided filter only, (b) deterministic falsifiers, (c) all falsifiers + archive, (d) (c) + online certification | Hack onset step, clean-vs-raw pass-rate gap, held-out family score, exploit switching after quarantine; MDE stated first | DeltaAI GPU |
| E8 | Are planted defects like natural ones? | FR per class on planted vs the real-defect set; capture–recapture vs error-seeding agreement | Per-class calibration gap with CIs | CPU |
| E9 | Head-to-head with published pipelines | Re-implement the published recipes of ScaleEnv (2602.06820), AWM (2602.10090) and a two-sided-filter-plus-judge pipeline as arms in the same Worlds, at equal dollars. Same student, same RL budget. | Residual-defect rate ρ̂ (planted + real); confirmed hack rate under RL; held-out family score; cost per perfect task | GPU + low $ |
| E10 | Do the regularisers matter? | §9.6 ablations: remove one regulariser at a time | J, locked FR, transfer, Goodhart drift (cell KL, task size) | Low $ |
| E6 | Factory economics | Yield and cost per stage; with vs without lemmas | Cost per admitted task with CIs | Low $ |
| E7 | Generality | The same code on SQL Apps, repos (debug lane), messaging Apps and one Gym env | Which classes apply; FR per environment class | CPU |

E1–E3 and E6–E7 need no RL. That is deliberate: the paper's core claims are cheap to measure. E4 and E5 tie the
framework to training.

## 14. Build milestones

Each milestone has unit tests that fail on the code before it, a DESIGN.md entry, and the merge procedure in the
repo rules.

| M | Scope | Acceptance |
|---|---|---|
| M1 | `canon`, `iface` (`PyWorld` + `CrucibleWorldAdapter` on finance_ap), `program` with taint, `contract` + `grade`, `infer` | finance_ap oracle programs run through the adapter and pass their inferred contracts. Hand-built planted contracts (one per D1–D11) produce the right reason codes. The grader fails closed on injected exceptions. |
| M2 | Falsifiers MUT, NULL, RED, TAINT, TWIN, PERT, REP; `certificate`; `archive`; `cegis` with a stub repairer | On PyWorld tasks every deterministic class finds its planted defect class. An always-pass contract is rejected by MUT and NULL. Archive regressions block a repair that re-opens a counterexample. |
| M3 | `meta/plant`, `meta/recall`, the dev and locked sets, the FR/FA matrix report; the operator DSL and birth gate (deterministic part) | The FR/FA matrix prints for PyWorld domains. A hand-written candidate operator passes or fails the birth gate as expected. |
| M4 | `roles`, `synth_wire`, LLM proposer, oracle programmer, contract writer, repairer, ALT/ATK agents, adjudicator, abstraction, on Tinker; PyWorld Apps for helpdesk (tickets, kb, mail) and finance_ap | End to end on 2 domains: ~20 proposed tasks each, certified or rejected with reasons; spend and yield per stage reported. |
| M5 | `PatchTask` adapter for the debug lane (fs canon + exec predicates); the historical real-defect set; ONL hook for eval rollouts | Stored debug-lane grades recompute identically through the adapter. Historical defects load as the real set. |
| M6 | `compose` + dependency depth; Gym toy; E1–E3, E6, E7 runs | Report tables for the paper. |

**Tinker budget for building and checking.** $150 in total for M4–M6, logged in `data/synth/**/spend.jsonl`, with a
hard cap in the wire. The checks use the swarm families (Kimi, GLM, DeepSeek, Nemotron, gpt-oss); the student is called
only in M6/E4. Spend beyond $150 needs the lead's approval.

## 15. Relation to prior work

### 15.1 System-level comparison with ScaleEnv and the closest pipelines

**Key.**
- ✓ means built in.
- ~ means partial, or done by a model's opinion.
- — means absent or not reported.
- "Ours" is the design. Rows marked † are not built yet, and none of our results exist yet.

| Dimension | ScaleEnv 2602.06820 | AWM 2602.10090 | EnvScaler 2601.05808 | Agent-World 2604.18292 | AutoEnvScaling 2610.14163 | WildAWM 2610.26828 | Hacker–Fixer 2606.08960 | Verifier evolution 2610.11464 | **Ours** |
|---|---|---|---|---|---|---|---|---|---|
| States the full target (valid + learnable + on-intent + novel) | — | — | — | — | ~ (reward spread) | ~ (gates) | — | — | ✓ P1–P4, §0 |
| World / App synthesis | ✓ tool-first, executed tool tests | ✓ tasks-first SQLite + API | ✓ blind tool tests | ✓ mined real data | ✓ agent in sandbox | ✓ persona → multi-day Worlds | — | — | ~ interface + PyWorld; tool-first synthesizer † |
| Reward / verifier form | Gold DB, per-column match | Code-augmented LLM judge | Checklist of state predicates | Tests + rubrics | Harbor tests | Executed verifiers | Existing benchmarks | Evolved detectors | Pass rules as data, one trusted grader, closed-world frame |
| Collateral / side-effect check | — | — | ~ | — | — | ~ | — | — | ✓ FORBIDDEN_EFFECT + ExtraWrite |
| Verifier testing | — (tools only) | — | — (tools only) | — | Reference = 1, empty < 0.5 | 5 executed gates, gaslight probe | Hacker rounds | Typed ops, birth gate | Nine typed falsifier classes, per-class certificates, archive replayed as a regression suite |
| Communicability / intent | — | — | — | — | LLM review | ~ | — | — | ✓ executed taint + instruction twins |
| Measured residual defects | — | — | — | — | — | — | Found-only (16% hackable) | — | ✓ recall-calibrated estimate per domain and class |
| Difficulty per student | — | — | — | ~ diagnosis | ✓ assignments, spread | ~ | — | — | ✓ band per (task, student, checkpoint, simulator, time mode) + frontier loop |
| Self-improvement | — | — | — | ~ arena loop, no control | ~ proposer memory; the trained proposer collapses without training | — | ~ repair rounds | ~ one level, one domain | ✓ four levels (tasks, falsifiers, planter, abstractor) with R1–R10 regularisers |
| Selection free of student reward | n/a | n/a | n/a | — | — (spread is student reward) | ~ | ✓ | ✓ | ✓ by construction |
| Substrates | Tool + DB | Tool + DB | Tool + DB | Tool + data | Terminal | CUA | Coding benchmarks | One domain | Any World with reset + canonical state; SQL, repos, messaging, Gym † |
| Reported RL result | τ² 30.1 → 38.5 (Qwen3-8B) | ✓ | ✓ | ✓ | TB2.1 26.7 → 38.9 | 42.84 → 53.42 on our student | none (admitted) | none | † E5, E9 |

**What the table supports, and what it does not.**
- **Scope.** No published system targets all of P1–P4. None measures how many defects remain in its output, and none
  regularises its self-improvement at several levels. On those three points the framework is broader, and the
  difference is categorical, not a matter of degree.
- **What others do better today.**
  - **ScaleEnv's World synthesizer.** Keyword → tool schemas → derived DB schema → executed tool tests → tool graph →
    snowballed environment. It is stronger than anything in `crucible.synth` now and is adopted as the F1 stage (P1-1
    in the evidence doc).
  - **WildAWM's dense multi-day Worlds** are richer than ours.
  - **Every listed system has RL results**, and we have none yet.
- **"Better" is an empirical claim.** It is settled by E9: re-implemented ScaleEnv, AWM and two-sided-filter arms, in the
  same Worlds, at equal dollars and the same RL budget. The measures are residual-defect rate, confirmed hack rate under
  RL, held-out score and cost per perfect task. Until E9 runs, the paper may claim broader scope and stronger
  validation by design, not better results.

### 15.2 Component-level credit


The check covered the 337 cards plus a web and arXiv search (2026-10-29); the full report is
`notes/06-novelty-check.md`. Several of the closest papers are not in the card set, and the check read only their
abstract or HTML page. Two must be read in full before submission:
- **2606.08960 (Hacker–Fixer loops):** an auditor will cite it against the falsifier and certificate stages;
- **2602.11224 (Agent-Diff):** the closest prior work for the contract substrate.

| Component | Closest prior work (rating of the component alone) | How the paper uses it |
|---|---|---|
| C1 closed-world contracts | Known (AppWorld 2407.18901; Agent-Diff 2602.11224; τ-bench) | The pass-rule substrate; credited |
| C2 falsifiers + certificates + archive | Partly novel (2610.09142; 2606.08960; BenchJack; SWE-ABS; ImpossibleBench) | Part of the falsifier and certificate stages; new: one typed population, per-class certificates, archive as regression suite |
| C3 taint communicability | Partly novel; the strongest mechanism claim (2610.33568 uses argument support for SFT masking only) | New mechanism; report precision and recall, and handle transformed values |
| C4 planted defects | Partly novel, close to novel (2607.27518; BenchGuard 2604.24955; classic error seeding, spec mutation, oracle assessment) | The yardstick of the RSI loop. Called **contract mutation**, not "second-order" or "higher-order mutation" (Jia & Harman 2009 use those terms for multi-fault program mutants). The classic roots are credited. |
| C5 evaluator-side RSI | Partly novel (2610.11464 has typed ops, a birth gate and a locked set for verifier evolution) | Falsifier loop of the RSI; new: cross-domain birth gate and transfer |
| C6 online certification | Partly novel (2606.26300; HackTrace; PRIME 2606.09711; 2609.35677) | Online certification stage; tested by the dose–response RL experiment |
| C7 composition | Known (τ² 2506.07982; WorkArena++; RHB; CompoWorld) | Composition and difficulty stage; credited |

Two ids came from the checking agent's memory and are unverified: AutoDAN-Turbo 2410.05295 and CaMeL 2503.18813.
VerifierContractBench has only a NeurIPS 2026 listing, with no arXiv id yet.

## 16. Risks

| Risk | Mitigation |
|---|---|
| Planted defects are easier than real ones, so FR overstates strength | Report the real-defect set separately; build the locked set from held-out operator variants |
| Straight-line oracles cannot express some workflows | Branches are resolved at build time per start state; for genuinely interactive tasks (asking a person), the actor's answers are recorded observations, and refs to them are allowed |
| Canonical state misses reward-bearing state (e.g. external side effects) | The App contract requires every write tool to declare the collections it can write; H-gates check that undeclared writes show up in `canon()` |
| Adjudication noise in ALT/ATK | Calibrated panel, PPI intervals, a human queue; the headline claims rest on executable classes |
| Family hacking (GLM, DeepSeek) inside builder roles | Builders never see gate code or contracts of other tasks; cross-family checks on every artifact |
