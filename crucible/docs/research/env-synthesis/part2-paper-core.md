# Part 2 paper core: synthesizing perfect RL tasks with an end-to-end, self-improving framework

**Date.** 2026-10-29; updated 2026-10-30 with the run docs in `runs/` (rsi-stats, rsi-2-prereg with Amendment 1,
recert-1, policy-1, taskloop-1, worlds-1, worlds-2, compose-2, perfect-1, yield-1, falsifiers-2) and DESIGN §7.7 items
16–24. **Status.** The problem, the framework and the RSI loop the Part 2 paper presents. This is also the build spec for
`crucible.synth`. §9 is rewritten: RSI is the creation-time, per-task regularised loop, and it has nothing to do with RL
training.

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
   - World families and instances, gated by execution (§11.1);
   - policy as data: every blocking decision is a versioned rule with a lawful-remedy procedure (§9.7).

   A training-time online monitor is a separate safety net. It is not part of RSI and never feeds it.
2. **A regularised recursive self-improvement (RSI) loop at task creation.** Each new task is certified, then probed by
   the student (probe rollouts, not training). Its outcomes are classified as intended or unintended passes and fails,
   and the task receives one typed decision per round: ORACLE precondition, then REPAIR, then RESKILL / DESCEND / ASCEND,
   then KEEP / RESERVE / HOLD. Regularisers (sample splitting, a drift alarm, a cell-KL trust region, an entropy floor, a
   RESKILL cap, a ledger and rollback) keep the loop from tuning tasks to the student. The lessons it finds are lifted
   into system-level updates:
   - the **falsifier** loop: counterexamples, the student's PROBE counterexamples included, become new falsifier rules;
   - the **synthesizer** loop: counterexample–repair pairs become design rules for the task writers;
   - the **planter** loop: new planted defects keep the falsifiers' yardstick ahead of them.

   A system-level update is accepted only by a statistically controlled test: an A/A noise floor, a betting e-process,
   alpha spending over the whole run and a reusable holdout. The objective is measured, never judged: perfect tasks per
   dollar, subject to an estimated imperfect fraction ≤ ε.
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
3. **The RSI loop** (§9). A creation-time, per-task regularised loop gives every task one typed decision per round from
   the student's probe outcomes, with an executed ORACLE precondition, evidence-derived re-labelling (RESKILL) and no
   deletion (RESERVE). Its lessons become system-level updates, selected on executed evidence, planted-defect recall and
   equal-budget controls under run-level error control, never on student reward. RSI has nothing to do with RL training.
   The hypothesis is that improvements transfer across domains and substrates.
4. **Policy as data** (§9.7). Every decision that used to go to a person is a versioned rule with its source, and one
   procedure resolves blocking conditions from ten ordered principles. No returned remedy can relax a validity rule.
5. **Measurement.** Error seeding plus capture–recapture turn falsifier-relative certificates into residual-defect
   estimates with intervals, per domain and defect class (§6, §8). This answers how good a task set is, not only how
   many defects were found.
6. **Evidence** (§13, §14.3):
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
- classic error seeding and spec mutation;
- the test-oracle problem (Barr et al., IEEE TSE 2015) and overfitting of automated repair to the tests it sees (Smith
  et al., FSE 2015), behind ORACLE and the held-out mutants;
- tests by betting (Waudby-Smith & Ramdas, JRSS-B 2024), the reusable holdout (Dwork et al., Science 2015) and A/A
  tests (Kohavi et al.), behind the acceptance statistics;
- goal relabelling in hindsight (Andrychowicz et al. 2017), the closest idea to RESKILL;
- student-adaptive curricula (teacher–student curricula, Matiisen et al. 2017; PAIRED, Dennis et al. 2020; Prioritized
  Level Replay, Jiang et al. 2021), behind DESCEND and ASCEND.

The contribution is the problem, the integrated framework that targets all of P1–P4, and the measured self-improving
loop. The parts that are, as far as the search found, new as mechanisms are value-tracked communicability (§5 TAINT,
TWIN), recall-calibrated certificates (§6), cross-domain falsifier transfer (§9.5) and the per-task creation-time loop
with its executed ORACLE precondition (§9.1–§9.4).

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
| L2 specification | Do C, ι and W capture the intended skill? | Executed probes where possible (taint, twins, null agents, red-on-start, the World's executed policy rules for ORACLE); otherwise a cross-family adjudication panel calibrated on anchors, with PPI-corrected rates. A panel verdict is necessary, never sufficient, and an uncalibrated panel abstains | Model calls |

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
| **ATK** attackers | Agents told to pass without doing the skill, given the attack-strategy library (§9.5) and the instruction, never the contract | A pass is suspect; the panel decides D7/D8, and confirmed hacks also become negatives | D7 D8 | $ |
| **MUT-HO** held-out mutants | A fixed share h = 0.2 of each task's distinct mutants, keyed by the task id, never shown to a repair round | As MUT. A held-out survivor rejects the task; held-out trials never count in U | D1 D2 D3 D6 D9 | ms per probe |
| **ORACLE** reference decision | The World's registered executable policy rules recompute each decision the contract pins; independent runs that changed a pinned field to one other value | A rule disagreement is a confirmed D6 (op `policy`). ≥ 2 agreeing independent runs raise a dispute, settled by the rules where they exist, else held | D6 (wrong oracle) | ms |
| **LEAK** humanised leaks | Layered TAINT (separator, number, lemma, label, synonym) and a model leak probe with a calibrated leak panel | Structural layers confirm; a probe quote is a suspect for the leak panel | D7 | ms + $ |
| **ONL** online rollouts | Every passing training or eval rollout | L1 hack rules plus the trajectory monitor flag passes; flagged passes are suspect, and a confirmed one quarantines the task. A training-time safety net: **not an input to RSI** (§9) | all | free |

**Adjudication.** A panel of two families, neither of which built the task, reads:
- the task;
- the intent card;
- the probe's trace;
- the reason codes.

It never reads the gate code or which falsifier made the probe.
- The panel is calibrated on anchors (trust-or-escalate λ).
- Below λ, the probe goes to the human queue.
- Adjudicated rates are reported with PPI intervals.

### 5.1 Precise rules settled by the M1–M2 build

These rules supersede the shorter wording above where the two differ. Each is pinned by a test in
`tests/unit/test_synth_*`.

1. **Required-pattern semantics (§3.3).**
   - **One change, one pattern.** A change satisfies at most one required pattern. "Invoice 42 held" and "any invoice
     held" therefore need two changes.
   - **Counts are exact.** A count is exact unless `count_max` widens it; `any` means at least `count`. A duplicate write
     must be allowed explicitly.
   - **Matching.** The grader runs two maximum bipartite matchings: one fills every required slot, and one covers every
     change that A does not allow. By Mendelsohn–Dulmage these equal one joint assignment. A greedy scan is provably
     wrong here, and a test shows it.
   - **Granularity.** An update is one change per field. An insert or a delete is one change per entity.
   - **State patterns.** A pattern kind `state` counts entities in the end state. Without it a "true at the start"
     requirement (D4) cannot be written.
2. **Equivalence uses stated constraints only.** Two runs are equivalent when they have equal normalised diffs, equal
   answers, and equal truth values on the trace constraints that are *stated*. A stated constraint carries a quote that
   appears in ι. An unstated constraint that rejects an equivalent mutant is how D6 (over-strict) is detected.
3. **Deterministic classes run in order.** REP runs first and gates MUT, TWIN and PERT. In a nondeterministic World every
   mutant looks non-equivalent, and the survivors would be mislabelled.
4. **NULL agents and their defect labels.**

   | Agent | Label when it passes |
   |---|---|
   | No-op, read-all, answer-only | D4 |
   | Oracle-minus-last-write | D1 |
   | Write-all | D8 |
   | Shortcut | D8 |
   | Random-k | D8 |

   - **Shortcut** calls each write tool π* never calls, filled with the values Δ* sets. It catches planted admin tools
     that write-all misses.
   - **Random-k** never borrows π*'s arguments, which would hand it the solution, and aims at entities π* never touches.
   - **Answer-only** is skipped on read-only tasks.
5. **TWIN substitution.**
   - A substituted literal also maps its entity's key, but only when the literal identifies exactly one entity. A shared
     value, such as a customer, is a selector: the keys it selects are re-derived instead.
   - Constants taken from the entity (an amount, a subject) are re-derived from π*′'s arguments.
   - A tool error in π*′ makes the twin dead: it is not counted as a trial, and D10 is noted for it.
6. **PERT counts flips only.** NULL's own findings and an oracle that already fails are not double-counted. Distractor
   clones differ on a field that holds an instruction literal, so a correct contract can tell them apart.
7. **TAINT sources.** A literal may be sourced from:
   - the instruction;
   - earlier observations;
   - `I.given`;
   - the **tool catalogue** (descriptions, enums, defaults), since reason codes are often stated only there.

   Library schema positions are structural and need no source: `get` paths, `pluck` fields, `round` digits and `select`
   field names. `select` values are still traced. Numbers are matched numerically ("$1,200.00" = 1200). Strings are
   matched case-insensitively on word boundaries.
8. **First real finding.** TAINT flagged the finance_ap pack's own oracle. It decides invoice ids it never lists: a D5 in
   the oracle program. Adding `list_invoices(status="pending")` first makes it clean. The finding is the kind of defect
   the paper counts in its real-defect set.

### 5.2 Rules settled by the real runs after M5 (2026-10-29/30)

1. **ORACLE: the reference decision is checked against the World's rules.** yield-1 found that half the finance_ap
   oracles of one proposer were wrong: DeepSeek-V4.1-Flash 16 of 31, Kimi-K2.6 0 of 17, checked against the pack's own
   policy. The oracle and the ALT solvers can share the mistake, so the contract accepted the solver and no other class
   saw it. ORACLE (§5 table, §9.3) closes this where a World registers executable rules. On the stored pool it rejected 7
   stored-certified tasks (exactly the 6 yield-1 had found by hand, plus a frontier task) and confirmed on 16 stored
   rejections (recert-1). A World without executable rules gets a weaker check, and the certificate says so.
2. **Held-out mutants (MUT-HO, h = 0.2).** A fifth of each task's distinct mutants is held out of every repair round, so
   a repair cannot fit the mutants it was shown (cf. overfitting in automated program repair, Smith et al. 2015).
   - U_MUT is computed on the **shown** split only, and held-out trials never count in U.
   - A held-out survivor rejects the task (R-holdout).
   - h is never lowered to raise yield. A task short of shown mutants gets a denser World instead (§11.1, superset
     density).

   Measured (recert-1): 59 stored-certified tasks became provisional only because their about 60 distinct mutants lost
   a fifth to the holdout (their whole neighbourhood gave n 60–65). MUT-HO also found 2 real SwapEntity D9 survivors that
   MUT's 240-mutant cap sample had missed. Densified Worlds then certified 53 of the 60 held-out-blocked tasks
   (taskloop-1, §14.3.9).
3. **The blind spots of the M3 recall matrix are closed** (falsifiers-2): MUT gained read mutants (an unstated
   `never(read)` is now confirmed as D6), and TAINT gained structural layers (separator, number, lemma, label, synonym)
   plus an L2 leak probe with a calibrated leak panel (the LEAK class). Confirmed ORACLE and LEAK findings reject in
   admission.
4. **TWIN anchors.** A `twin_solution` calibration anchor is emitted only when the intent card derives nothing: on a
   decision task, π*′ carries π*'s decision to a sibling whose right decision differs, so the anchor's "valid" label was
   false and the panel was right to reject it (compose-2, issue 10).
5. **Archived agent counterexamples replay their trace.** An archived ALT or ATK counterexample is graded as its
   recorded trace, never re-sampled; a judged-valid trace replays only on versions with the same instruction
   (`instruction_sha`), and legacy ones without it fail closed (recert-1).
6. **Seeds by tag are the default** for every model call (recert-1). Before, every call ran on seed 0 unless a wire asked
   otherwise: in yield-1/2, 120 of 216 Ultra attempt pairs were the same trajectory, and the first learnability pilot's k
   rollouts were one sample copied k times.
7. **A panel is necessary, never sufficient** (P-evidence, §9.7). The rule comes from four measurements:
   - the World realism panel rated all 20 worlds-1 instances realistic, including the 8 the executed gates reject. With
     4 gate-rejected anchors added it is 7 of 11 correct and uncalibrated, so the executed gates decide (worlds-2);
   - a panel judged a D7 leak attempt valid at confidence ≥ 0.99, because it judges against the instruction it was given
     (perfect-1);
   - on repository diffs no panel prompt reached the 0.9 target, so all 10 open MUT suspects abstain to a person
     (falsifiers-2);
   - the pipeline's panel decided while uncalibrated under the M4 rule (policy-1).

   A panel decision therefore counts only when the panel is calibrated (≥ 6 decided anchors of both truths, R-panel),
   and a panel pass never admits on its own.

## 6. Validity certificates

For a task τ and falsifier class k:
- n_k is the number of independent probes;
- c_k is the number of confirmed counterexamples.

The certificate is the vector of (n_k, c_k, U_k), where U_k is the exact Clopper–Pearson 95% upper bound on the
counterexample rate *under class k's probe distribution*. At c = 0, U ≈ 3/n (the rule of three).

**Admission** (R-admission-fixed, §9.7). A task is admitted when:
- the deterministic classes (MUT, MUT-HO, NULL, RED, TAINT, TWIN, REP, PERT, and ORACLE and LEAK where they apply) have
  c_k = 0 after repair;
- no suspect is open (a task whose only findings are undecided suspects is **on hold**, not admitted);
- U_MUT ≤ 0.05 on the shown split, which needs n ≥ 59 shown mutants (held-out trials never count).

The M2 operators produce 13–42 mutants for 1–4-step tasks and 60 for a 6-step task. M3 therefore scales the mutant
neighbourhood with the World:
- ExtraWrite and SwapEntity range over every sibling entity;
- PerturbValue uses several magnitudes.

A task with c = 0 that still has n < 59 is admitted as **provisional**, with its U reported, and is never counted as
**certified**.
- ALT has at least one pass from a non-builder family on the whole task (after the solver ladder, §9.7) and no confirmed
  D6;
- ATK has 0 confirmed hacks over its budget.

Every certificate records the policy version it was decided under; a looser admission policy is recorded as a
deviation, and a deviating certificate is kept out of the pool.

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
- **The dev set** is used by the system-level updates of the RSI loop (§9.5).
- **The locked set** is never read by any loop. It is built with held-out operator *variants* (different phrasing,
  different collections) on held-out domains.

A third, **real** defect set is collected from history:
- the debug lane's documented measurement-time failures: unstated rules, undescribed faults, stale tests, changelog
  hints, fake links;
- every confirmed counterexample from ONL (falsifier-found, so reported separately and never used to calibrate recall;
  ONL is a training-time safety net and not an RSI input).

The paper's headline numbers are on the locked and real sets.

**Falsifier recall.** FR_k,d is the fraction of planted environments of class d on which falsifier class k finds a
confirmed counterexample. Its pair is the **false-alarm rate** FA_k: confirmed counterexamples on certified anchor tasks.
Each such alarm is either a real defect, which is adjudicated, kept as a new finding and its anchor repaired, or a
falsifier bug.

### 8.1 Rules settled by the M3 build, and the first recall matrix

**Witnesses by class.** "Passes C′, fails C" is the witness for loosening defects: D1, D2, D3, D8 and D9. The other
classes need their own executed evidence:

| Class | Witness |
|---|---|
| D6 | A valid solution that passes C and fails C′ (the mirror) |
| D4 | `vacuous_required` replaces a requirement with one already true at s₀ and allows the effect. A pure addition only tightens the contract and has no witness. |
| D5 | The deleted instruction span, and TAINT's unsourced literal |
| D7 | The recorded inserted text |
| D10 | The oracle fails, the planted value is nowhere the agent can read, and no candidate run passes |
| D11 | Two replays that differ |

**Calibration caveat.** For D5, the planting evidence and the detector are the same mechanism, so TAINT's recall on
planted D5 is 1 by construction. D5 stays **uncalibrated** until it is measured on the real-defect set (E8). The same
check applies to any class whose witness and detector share a mechanism.

**Mutant counts** are counted by distinct effect: diff, answer and reason codes. On the fixture tasks a single-step task
has 13 distinct mutants. A task reaches n ≥ 59 only in a World with enough sibling entities, so M3's 16-invoice fixture
World yields 75.

**Update-rule thresholds.**
- δ = max(2·sd, range/2) over at least 3 calibration runs. It can only go up.
- J* is the incumbent, measured on the same fresh view and paired by unit.
- R7 and R9 use the same δ bar.
- A pruning update is accepted if J stays within δ and R3 holds.
- **Fictitious play** constrains changes to F, the falsifiers. Retiring or replacing an operator must keep recall on
  the union of every past planter's sets.

**Not yet implemented.** The `plant_affordance` variant "expose a grader-read resource for writing" is missing. The
admin-tool variant exists.

**The first FR/FA matrix (fixtures).** 34 plants, one per operator variant and domain, against 5 certified anchors:
- **No false alarms** in any falsifier class.
- **Every planted class is caught by at least one plant.**
- **Three real recall gaps.** Each is a natural input for the falsifier loop:

| Gap | Why it is missed |
|---|---|
| An unstated `never(read)` constraint | No MUT mutant *adds* a read |
| A humanised leak ("price variance" for `price_variance`) | TAINT's matcher needs the exact token |
| `eq → any` relaxations | MUT only reaches *suspect*, needing adjudication. A RED mislabel of these as D4 was fixed in 59e00633. |

**The falsifier loop on the first gap.** A hand-written DSL operator that inserts an unused read closes it and passes
the birth gate. An operator that false-alarms on all 5 anchors fails the gate. This is the loop working end to end, with
the abstractor role played by hand. In M4 the swarm abstractor plays it.

## 9. The regularised recursive self-improvement loop (RSI at task creation)

**What RSI is.** RSI is a creation-time, per-task, regularised loop (lead, 2026-10-29 and 2026-10-30; `runs/rsi-stats.md`
§9.5′–§9.5‴, `runs/rsi-2-prereg.md` with Amendment 1). Each round, for every task in the pool:
1. a task is created and certified (§3–§7);
2. the student runs on it as **probe rollouts**;
3. each outcome is classified as an intended pass, an intended fail, an unintended pass, an unintended fail, or
   uncertain (§9.2);
4. the task receives exactly one typed decision. The moves are tried in a fixed order: the ORACLE precondition, then
   REPAIR, then RESKILL / DESCEND / ASCEND, then KEEP / RESERVE / HOLD (§9.3);
5. the move is accepted only under the regularisers of §9.4;
6. the lessons the round found are lifted into system-level updates, which a separate, statistically controlled
   acceptance test decides (§9.5, §9.6).

**What RSI is not.** RSI has nothing to do with RL training.
- Probe rollouts are measurement. No gradient step is taken, no checkpoint changes, and the student receives no reward.
- RSI runs before a task enters a training pool. It is finished when every task has its decision.
- The online monitor that watches rollouts during RL training (ONL, §5) is a separate safety net. Its findings never
  feed RSI, and no training signal enters any RSI decision.
- Student reward never enters J or any gate (P-student, §9.7). Pass rates enter in one place only, as an alarm
  (the drift alarm, §9.4).

**Why it must be regularised.** A loop that proposes edits and keeps whatever scores best is not RSI. It is
hill-climbing on a reused evaluation, and it overfits that evaluation. RRSI measured this directly. The unregularised
harness loop had the best evolve-set score and no out-of-distribution gain. With regularisation, out-of-distribution
score was 43.6 at 2.42M tokens, against 40.3 at 3.80M without it, and a higher evolve score went with lower transfer in
every arm (2609.24972). Unguarded verifier evolution collapsed to always-pass while downstream scores stayed flat
(2610.11464). The per-task loop has its own version of the danger: a loop that edits a task until the student's pass
rate lands in the band tunes the task to one student. The regularisers of §9.4 forbid exactly that.

### 9.1 The loop

```text
taskloop(pool, student, policy):                              # meta/rsi.TaskLoop, one typed decision per lineage per round
  for round in 1..3:
    for each active lineage τ (concurrently, Worlds keyed by task):
      ORACLE(τ)                                               # precondition: the reference decision vs the World's rules
        wrong     → REPAIR (the oracle first)
        disputed  → executed rules settle it; without rules HOLD (panel / human queue)
      P1(τ)                                                   # precondition: the certificate
        rejected  → REPAIR from shown counterexamples (never held-out mutants), else REJECTED (kept with its reasons)
        on hold   → HOLD (open suspects to the panel)
      R ← probe(student, τ, k = 8)                            # diagnosis rollouts; their indices never accept anything
      O ← classify(R)                                         # §9.2
      d ← resolve(condition(τ, O), policy)                    # §9.7: the first lawful move in the order of §9.3
      apply d under T1–T7, G1–G9, RS1–RS6                     # §9.4; fresh rollouts for acceptance; full re-certification
      ledger.append(d, its evidence, the rule, the policy version)
  lift(archive, accepted repairs, World fixes) → candidate system updates → §9.5–§9.6
```

The probe runner is the learnability agent's `StudentProbe` (graded rollouts with a replayable literal program, seeded by
tag). Round 1 reuses stored diagnosis rollouts of the exact task version when they exist (prereg §8.1), so no diagnosis
is bought twice.

### 9.2 Outcome classes

| Class | Meaning | Evidence it needs | What it can drive |
|---|---|---|---|
| **Intended pass** | Success that used the skill | Executed, or the calibrated panel | Nothing; it counts toward the band |
| **Intended fail** | A genuine skill gap | Executed, or the calibrated panel | Nothing. A repair that would make it pass is refused (T2) |
| **Unintended pass** | A shortcut, a leak, a vacuous check, an open frame, a hack path | Executed (the student's own program replayed) or calibrated | A PROBE counterexample in the archive; REPAIR |
| **Unintended fail** | An ambiguous instruction, hidden information, an over-strict check, a World or tool bug, contradictory documents, an unsolvable variant | Executed or calibrated | REPAIR. Demoted to an unattributed fail when an independent ALT family solved the task (T7) |
| **Uncertain** | A routed failure (budget, protocol break), or a leak the panel cannot clear | — | Never a task defect and never evidence of "too hard". A task whose graded rollouts are mostly uncertain is held |

- An executed unintended outcome becomes a counterexample of class PROBE: the student's program with the verdict the
  task should have given. Archive regressions replay it like any other counterexample.
- A judged-only or evidence-free "unintended" fail is demoted to an unattributed fail and treated as a skill gap. A
  label that contradicts the trusted grader's verdict is void.
- Labels decided under a wrong reference decision are void in that decision's repair: the student's "intended fail"
  may be the policy's right decision graded against the wrong oracle (taskloop-1, fix 11).
- The classifier is `synth.outcome` (perfect-1), adapted by `rsi.LearnClassifier`. On 26 known-label fixture cases it
  decided 25 with accuracy 1.00, and it never called a skill gap a defect. These cases were written with the rules in
  view, so they show that the rules work as designed, not their recall on natural defects. On 676 real rollouts it
  gave 424 intended fails, 132 intended passes, 52 unintended fails (hidden information on 7 tasks), 1 unintended pass and
  67 uncertain.

### 9.3 The typed decision

"Works for all tasks" means that every task gets a correct, typed decision, not that every task is admitted. The order
of moves is fixed (Amendment 1): **ORACLE precondition → REPAIR → RESKILL / DESCEND / ASCEND → KEEP / RESERVE / HOLD.**
The policy (§9.7, rule R-too-hard) resolves it; a decision records the rule, the policy version and every rung passed
over.

| Decision | When | Action |
|---|---|---|
| (precondition) **ORACLE** | Before any difficulty move | The reference decision is checked against the World's executed rules and against independent runs (below) |
| (precondition) **P1** | A rejected or on-hold certificate | Rejected: repaired from its confirmed shown counterexamples, else the lineage is REJECTED (kept, with its reasons). On hold: HOLD |
| **REPAIR** | The oracle is wrong, or an unintended outcome has executed or calibrated evidence | CEGIS repair of the oracle, contract, World instance or instruction (§7), under T1–T7 |
| **RESKILL** | A valid task whose evidence shows another skill than its label | The intent card is re-labelled from the task's evidence; the task is kept or reserved by its fresh band (RS1–RS6) |
| **DESCEND** | Intended fails; the band says too hard | One declared, skill-preserving easier knob step → a new variant. The original stays as the harder rung |
| **ASCEND** | Intended passes; the band says too easy | One declared harder knob step (distractors, paginate, actor, derive, depth) → a new variant |
| **KEEP** | In band and valid | The lineage stays in the band pool |
| **RESERVE** | No lawful move in the needed direction; a move against the lineage's direction; 2 refused edits; the round cap | Kept for stronger checkpoints, out of the band pool. **No task is ever deleted** (RESERVE replaces RETIRE) |
| **HOLD** | A disputed reference decision; mostly uncertain rollouts; a band the sequential test cannot decide at k_max; a refused or declined repair of an invalid task | Nothing is edited. The task waits for the panel, a person or more evidence |

**The ORACLE falsifier** (`falsify/oracle.py`). A wrong reference decision is invisible to every other class: the
contract is inferred from the oracle, and the ALT solvers can make the same mistake (yield-1 found admitted finance_ap
tasks whose oracle and solvers agreed on a decision the pack's policy does not reach). ORACLE is a falsifier class with
two probes:
- **Policy truth.** A World may register its policy as executable rules (`synth_wire.finance_ap_truth`). The rules
  recompute every decision the contract pins. A disagreement is a confirmed D6, op `policy`. Without registered rules
  the class is not applicable, and the certificate says so.
- **Decision disagreement.** At least 2 independent runs that *changed* a pinned decision field to one other value
  dispute the reference decision. A dispute is settled by the executed rules where they exist (`overruled_dispute`),
  and otherwise held for the panel or a person.

ORACLE runs in certification (a confirmed ORACLE finding rejects) and as the loop's precondition. A shared mistake of
every agent stays invisible in a World without executable rules; the report states which Worlds have them.

**RESKILL** (Amendment 1; `meta/reskill.py`). A task whose label does not match the skill it exercises is re-labelled
from its evidence. It is kept if it is a good task under the new label. A too-hard task re-labelled to a more advanced
skill is still too hard for this student, so it is reserved under its honest label, not kept. The rules RS1–RS6 are in
§9.4.

**DESCEND's lawful moves** are the skill-preserving easier knobs a variant World makes cheap: undo a recorded harder
knob, `fewer_distractors` and `unpaginate`. Each is kept only if π* still runs and passes the contract on the easier
World. Moving a value from derive to given under the same skill is forbidden: it gives the decision away (D7). That
case is a RESKILL to a lookup-and-apply skill, when the domain has one. Shorter horizons apply to compositions only.

**What the moves found on the first real run** (taskloop-1, 195 tasks; §14.3.9): DESCEND and ASCEND did nothing useful
on finance_ap. Its SQL World cannot be perturbed (no distractors), and pagination did not make single-invoice decisions
harder for this student. Ladders need knobs the World supports.

### 9.4 The regularisers

Every move is checked against every rule below. The rules are non-compensatory: a gain on one cannot buy a loss on
another. The named regularisers of the loop are:

| Regulariser | As built | What it prevents |
|---|---|---|
| **Sample splitting** (G4) | The diagnosis's rollout indices never accept anything. Acceptance draws fresh indices (new seeds) sequentially, in looks of 4, 8, 12 and 16 (k_max). A repair is accepted only if no unintended outcome of the same mechanism recurs; a variant only if its fresh band moved in the declared direction | Accepting an edit on the same noise that motivated it |
| **Drift alarm** | On re-probed repaired tasks, d = p_after − (passes_before − unintended_passes)/n_before. A fixed unintended fail explains a rise of up to unintended_fails/n_before; the unexplained part u = d − clip(d, 0, that bound) is averaged over tasks. The alarm fires when at least 5 tasks give a mean \|u\| above 0.1 with a 95% bootstrap interval that excludes 0. It stops all repairs for review; nothing is accepted because of it | Tuning tasks to the student's pass rate |
| **Cell-KL trust region** (G7, RS5) | KL(the pool's cell histogram with the move ‖ the pool at the start) ≤ κ = 0.1 | Certification Goodhart: drift toward tasks that are easy to certify or easy for this student |
| **Entropy floor** (G7, RS5) | H(cell histogram) ≥ H_min | Collapse of the pool onto a few cells |
| **RESKILL cap** (RS5) | At most 20% of a round's tasks are re-labelled; one attempt per lineage (RS6) | Re-labelling as a yield lever |
| **Ledger and rollback** (G9, RS6) | Every decision is a ledger record with its evidence, rule and policy version. A repair is a new version and a variant a new task; the previous ones, and an old intent card, are restorable (`rollback_reskill`) | Irreversible drift; unauditable decisions |

The full rule sets:

**Repairs (T1–T7).**

| # | Rule | What it prevents |
|---|---|---|
| T1 | **Evidence only.** A repair needs ≥ 1 unintended outcome with executed or calibrated evidence. Judged-only or evidence-free labels are demoted; a label that contradicts the grader is void | Repairs justified by opinion; tuning by pass rate |
| T2 | **Never ease a skill gap.** Every intended (and unattributed) fail's program is replayed on the repaired task; one that now passes refuses the repair | Making tasks easier for the student |
| T3 | **Never over-constrain.** Every intended pass's program is replayed; one that now fails refuses the repair | A repair that adds a D6 |
| T4 | **Close what justified it.** Every unintended pass must now fail; every executed unintended fail of a valid program must now pass | Cosmetic repairs |
| T5 | **Monotone archive.** No archived counterexample re-opens (PROBE and ORACLE ones included); the deterministic classes stay clean | Forgetting |
| T6 | **Caps.** ≤ 3 repairs per task; calibrated-only evidence stops after 2 repairs without new executed evidence | Self-verification saturation (R8) |
| T7 | **Independent attribution.** A failure that claims the task cannot be solved as stated is demoted when an ALT solver of a non-student, non-builder family solved the task | Calling a skill gap a task defect |

**Every decision (G1–G9).**

| # | Regulariser | As built |
|---|---|---|
| G1 | Evidence-gated edits | A REPAIR cites a counted unintended outcome (T1) or the ORACLE counterexample; a move cites a declared knob and its direction |
| G2 | Frozen intent | The variant or repair keeps the skill and the cell and passes the P3 probes (EVID, SWAP, TAINT, TWIN); else refused. The intent changes only through RESKILL |
| G3 | Full re-certification under the monotone archive | Every class, held-out mutants at h = 0.2, archive regressions (PROBE and ORACLE counterexamples included), after every edit |
| G4 | Sample splitting | Above |
| G5 | Bounded movement | One knob step per round; ≤ 3 rounds per lineage; a lineage that moved up never moves down, and the reverse; a refused knob is not re-tried on its lineage |
| G6 | Not tuned to the student | A variant must be solved by an independent ALT family; the drift alarm watches repairs; knob moves are logged as knob calibration (predicted against realised band) |
| G7 | Pool regularisation | The cell-KL trust region and the entropy floor; originals are never retired by a variant |
| G8 | System-level lifts | Archive counterexamples, lemma candidates and World-builder fixes go to the update rule (§9.5) and its statistics (§9.6) |
| G9 | Ledger and rollback | Above |

**RESKILL (RS1–RS6).**

| # | Rule | As built |
|---|---|---|
| RS1 | An existing skill | The new skill is one of the domain's `DomainSpec` skills; no invented skill |
| RS2 | The label comes from the task's evidence | `INTENT_DERIVER`, a non-builder role (the abstractor's family, temperature 0, seeded by tag), reads the oracle program, its taint map and the executed effects. It never sees the student, a rollout, a pass rate, a band or the current label. Every derive value is something π* writes or answers that the instruction does not state |
| RS3 | Full re-verification | P3 under the new card, the leak check with the new derive set, full re-certification under the monotone archive |
| RS4 | Good under the new cell | Novelty against the pool tasks of the new cell; the band on fresh rollouts places the task: in band → KEEP, out of band → RESERVE under its honest label |
| RS5 | Pool regularisation | The move counts in the cell-KL trust region and the entropy floor; the 20% per-round cap |
| RS6 | No ping-pong, reversible | One attempt per lineage; the old card and version are in the ledger; rollback restores them |

**J and every acceptance gate never use the raw student pass rate.** A repair decision is a function of the counted
outcomes and the replays of their programs. DESCEND and ASCEND are steered by the student's band: that is the point of a
curriculum, so the regularisers bound it rather than remove it (one declared step, a direction lock, an independent
solver, fresh rollouts, the trust region). For knob moves the pass rate is supposed to move, so they are logged as knob
calibration; the drift alarm covers repairs, where it should move only as the fixed defects explain.

### 9.5 Lifting lessons into system-level updates

The per-task loop improves tasks. Its lessons improve the system that makes tasks. A lesson becomes a candidate update to
one component of the system state Θ:
- an archived counterexample, through its natural cluster, becomes a candidate falsifier operator;
- an accepted repair becomes a candidate design lemma for the writers;
- a World or tool bug becomes a World-builder fix;
- the knob calibration records (predicted against realised band) inform the knob ladders.

A candidate is accepted only by the update rule below and the statistics of §9.6.

**The system state Θ.**

| Component | Contents | Improved by |
|---|---|---|
| **S, the synthesizer** | Role prompts; the lemma bank; recipes; source weights; cell quotas | Lemma candidates from repairs (the synthesizer loop) |
| **F, the falsifiers** | Typed operators; attack strategies; adjudication prompts | Counterexample clusters, PROBE ones included (the falsifier loop) |
| **P, the planter** | Contract-mutation operators that plant defects | The planter loop (the P/F game) |
| **K, the knob ladders** | Declared knobs, their directions and ladders | Knob calibration from the per-task loop |
| **A, the abstractor** | The prompts and procedure that turn lessons into candidates | The meta loop |

Every component is versioned data, and any version can be restored with a plain write.

**The objective.** Per round r, at budget B, K_r is the number of admitted tasks that satisfy P2–P4 and ρ̂_r is the
recall-calibrated residual-defect fraction (§6). The objective is **J(Θ) = K(1 − ρ̂) / B**, perfect tasks per dollar.
Every quantity is measured by execution, by planted defects with witnesses, or by equal-budget controls. Student reward,
visible pass rates and chains of thought never enter J.

**Three levels of evaluation, each improved.**

```text
level 0   tasks                 evaluated by  F (falsifiers) and the per-task loop -> certificates, typed decisions
level 1   falsifiers F          evaluated by  P (planted defects)                  -> recall FR, false alarms FA
level 2   planter P             evaluated by  witnesses + the real-defect set      -> validity, natural-likeness
level 3   abstractor A          evaluated by  the birth-gate pass rate of what it produces next round
```

**The P/F game produces headroom.** F and P play a regularised two-player game. P is rewarded for planting defects the
current F misses; F is rewarded for catching them without false alarms. Three constraints keep P honest:
- **Witness validity, executed.** Every planted contract C′ comes with a witness run that passes C′ and fails the
  original C. A planted "defect" without a witness is not a defect.
- **Natural-likeness.** Planted defects are described by features (class mix, diff size, collection, which literal
  moved), and their distribution must stay within MMD ≤ μ of the real-defect set.
- **Fictitious play.** F must keep recall on the union of all past planters' sets, not only the newest.

*Headroom* is the share of dev plants the current population misses that some operator expressible in the DSL catches
with no false alarm on the dev anchors (`meta/headroom.py`; 6,206 bodies × 6 label rules = 37,236 operators). It bounds
what one round of DSL operators can add. A planter is rewarded for headroom, not for misses alone: a miss nothing
expressible catches is a DSL blind spot, reported, but it gives F nothing to learn.

**The update rule.** A proposed update u: Θ → Θ′ touches one component. It is accepted only if **every** rule holds
(non-compensatory).

| # | Regulariser | Rule | What it prevents |
|---|---|---|---|
| R1 | **Fresh evaluation, strict** | A share of each domain's dev source tasks is the **decision split**. Decision views draw units without replacement from units no earlier decision consumed; the split is never shown to a proposer, the birth gate or the planter loop. Calibration, A/A and audit views come from the proposal split. Every α spend and unit reservation is written to the ledger before evaluation | Adaptive overfitting to a reused evaluation set (the reusable holdout, Dwork et al. 2015; RRSI) |
| R2 | **Noise floor and a sequential test** | The paired gain must beat the A/A noise floor δ_AA in a test by betting at the level α_t of the run's spender (§9.6). The floor only moves up | Chasing noise; the winner's curse |
| R3 | **Hard non-regression** | Every archived counterexample is still rejected. FA on anchors ≤ δ_FA. No class's recall drops beyond its noise band. The P3 share and the P4 coverage do not drop | Trading one defect class for another; forgetting |
| R4 | **Trust region on the task distribution** | KL(q_Θ′ ‖ q_Θ) over the cell histogram ≤ κ, and H(q_Θ′) ≥ H_min | Certification Goodhart |
| R5 | **Sparsity and cost** | The edit budget anneals from 3 to 1 per round on a cosine schedule (RRSI). An operator must pay for itself (marginal recall per probe-dollar ≥ β); components with no marginal gain over W rounds are pruned (a prune is a non-inferiority test) | Library bloat; untestable bundles of edits |
| R6 | **Leakage critic and family separation** | A different family reads the diff and rejects edits that encode task ids, entities, values, planted-defect identifiers or anchor content. The abstractor is never a family that built the source tasks | Memorising the yardstick; self-grading |
| R7 | **Transfer requirement** | The gain must also appear on at least one domain that produced none of the update's evidence | Single-domain overfitting; the test of the transfer hypothesis |
| R8 | **Round caps without new executed information** | A component whose acceptance rests only on judged evidence stops after 2–3 rounds unless new executed counterexamples arrive (Mind the Gap, 2412.02674) | Self-verification saturation |
| R9 | **Equal-budget control rounds** | Each accepted round is compared against a control round at the same spend with u removed (a second betting test at the same α_t) | Crediting the loop for what more budget would have bought |
| R10 | **Rollback** | Every accepted update is reversible by restoring a stored version. A later regression on a fresh view rolls back to the last good Θ | Irreversible drift |

**The loops in this frame.**
- **The falsifier loop (F).** Confirmed counterexamples, PROBE ones included, are clustered by reason code, operator,
  diff signature and tools. The abstractor turns each cluster into a typed operator or an attack strategy. The **birth
  gate** applies R1–R7 to a single operator: it must reproduce its source counterexamples, confirm a counterexample on a
  proposal-split plant the incumbent misses (`gate:adds_on_proposal`), add catches no candidate already sent to a test
  this round holds (`gate:no_new_catches`), raise zero false alarms on anchors and stay under its cost bar.
- **The planter loop (P).** P proposes new mutation operators or parameterisations, rewarded by headroom. A planter is
  accepted under witness validity, novelty, natural-likeness (MMD ≤ 1.1 × that of the base dev plants to the independent
  real defects) and R6.
- **The synthesizer loop (S).** A (counterexample, repair) pair, from CEGIS or from the per-task loop, is abstracted
  into a lemma. A lemma batch is accepted when the first-round confirmed-counterexample rate falls and J rises, at equal
  budget, under R1–R9.
- **The meta loop (A).** The abstractor's own prompts are edited by the same rule. Its J-proxy is the birth-gate pass
  rate of what it produces next round, tested by a one-sided Fisher exact test with ≥ 4 candidates a side. Recursion
  stops here: deeper levels add no new executed signal, so R8 caps them.

The earlier design had a fourth, "frontier" loop that targeted student failures from training-time traces. It is
replaced by the per-task moves of §9.3, which act on creation-time probes. Its one measured round (perfect-1: the
targeted arm 0 perfect tasks, the control 1, n = 4 per arm, no difference) is reported in §14.3.2.

### 9.6 The statistics of acceptance

RSI-1's update test had five flaws (`runs/rsi-stats.md` §1):
- δ was the spread of J across *different* views, while the decision statistic is a paired difference on one view;
- the "fresh" views were overlapping 60% draws from one pool of 125 plants;
- each candidate was tested at an uncorrected one-sided normal bound, so the family-wise error grew like 1 − 0.95^T;
- the meta loop accepted A′ on 2/2 against 0/2, whose exact one-sided p-value is 1/6;
- the locked set had 1 plant for D1, D3 and D4.

The replacements follow.

**The A/A noise floor.** An update u is evaluated on n decision units (planted tasks). Unit i gives
D_i = Y_i(Θ′, s′) − Y_i(Θ, s) ∈ [−1, 1], with s and s′ two independent evaluator seeds (MUT's sibling sample, NULL's
random calls, TWIN's twins, PERT's perturbations). The null is H_u: E[D] ≤ δ_AA, "the update is no better than the
unchanged Θ is against itself". Before the loop, the unchanged Θ is run against itself on one fresh view with k ≥ 5
disjoint pairs of seeds (A/A replicates); each gives a mean paired difference d̄_r, and δ_AA = max(2·sd(d̄_r), max_r
|d̄_r|), at the decision view's size, never tuned, raise-only. The A/A distribution is exactly the null distribution of
the decision statistic for Θ′ = Θ (cf. A/A tests in online experiments, Kohavi et al.).

**The betting e-process.** R2 holds when a one-sided test by betting (Waudby-Smith & Ramdas, JRSS-B 2024) rejects H_u at
level α_t. On X_i = (D_i + 1)/2 and m = (δ_AA + 1)/2 the wealth K_n = Π_{i≤n} (1 + λ_i (X_i − m)), with predictable bets
λ_i ∈ [0, c/m] (aGRAPA, c = 3/4), is a non-negative supermartingale under H_u. Ville's inequality gives
P(sup_n K_n ≥ 1/α_t) ≤ α_t, so the test is valid at every stopping time. The view runs in chunks of fresh units and stops
as soon as the wealth crosses 1/α_t (accept), or after n_fut units when the mean gain is at or below δ_AA (futility,
which only lowers power). Unevaluated units are released and stay fresh. Each decision reports the e-value, the anytime
lower bound, and the paired effect with bootstrap and cluster-bootstrap (by source task) intervals. R3–R7 compare Θ′ and
Θ at the same seed.

**Alpha spending over a whole run.** Every decision that can accept something is a test t with a level α_t fixed before
its data are drawn, from one spender per arm: online Bonferroni, α_t = α / t_max for t ≤ t_max (preregistered), or
α_t = α·6/(π² t²) when open-ended.

> **Theorem.** If each test is valid conditional on everything before it, P(reject_t | F_{t−1}) ≤ α_t whenever H_t is
> true, then P(some non-improving update is accepted in the run) ≤ Σ_t α_t ≤ α.

The union bound needs no assumption on the dependence between tests, and the candidate of test t may depend on anything
before it. Conditional validity comes from Ville's inequality, given that the units of test t are exchangeable draws that
nothing before t has seen: that is what R1-strict guarantees. Alpha-investing (Foster & Stine 2008) and online e-BH
control the false discovery rate, not this family-wise error, so they are not used. R9, prunes and the meta loop's
Fisher test spend α; audits (they can only roll back) and planter acceptance (executed checks) do not. A write-ahead
ledger record means a crashed round never gets its α or its units back. "Non-improving" means E[D] ≤ δ_AA on the
decision-split population; generalisation is a separate gate.

**Generalisation gates.** A system-level update is accepted only if, beyond R1–R10:
- **R7**: its gain appears on dev domains that produced none of its evidence;
- **G (the locked set) through Thresholdout.** Any decision that reads held-out data goes through a reusable holdout
  (Thresholdout, Dwork et al., Science 2015): the gate receives the dev value unless the locked value differs by more than
  a noisy threshold, and then a noisy locked value, with the budget down by one. Every access is logged without the
  held-out value. A spent budget fails closed;
- **the held-out World family**, which no round's evidence came from, is measured every round and reported. No decision
  reads it. It is the headline measure of generalisation.

**Headroom precondition.** If dev headroom is 0 and the planter adds none in rounds 1–2, the falsifier loop stops for
futility, and H1–H3 are reported as **not testable on this pool**, never as a negative result (prereg §6).

**Power, stated before the run.** Run-level control has a price. Exact one-sided conditional McNemar power by
enumeration (with the measured seed discordance of 0.64% per unit) gives:

| Measure | Value |
|---|---|
| Paired plants for a recall gain of 0.2 at power 0.8 | 33 (71 for a gain of 0.1) |
| MDE of one decision at α_t = 0.005 (t_max 10) | 0.27 at n = 60; 0.15 at 120; 0.094 at 240; 0.061 at 375 |
| Simulated family-wise error of a run of T null candidates | 0.000 for T = 5–40, against 0.204–0.798 for the uncorrected per-test rule |
| Per class on the locked set (Bonferroni over 11 classes) | 112, 72, 53 and 34 plants for gains of 0.1, 0.15, 0.2 and 0.3 |

D1, D3, D4 and D8 yield about one witnessed plant per 7 source tasks, so they stay under-powered in a 10 × 10 pool and
are reported descriptively with Wilson intervals, never as a test. Pooled over classes the primary tests are well
powered (MDE 0.075 on a 10-task held-out family, 0.03 on a 30-task locked set). E10's ablations, with 3 seeds each, are
reported as effect sizes with intervals, not as tests. "Not significant" is always reported with its MDE, never as "no
effect".

**Statistics of the per-task loop.** The band is a sequential test in looks of 4, 8, 12 and 16 rollouts (R-band). Its
stated error at k ≤ 8 is 0.29–0.37 at the band edges (perfect-1), and a task the test cannot decide at k_max is held,
never moved. The drift alarm is a bootstrap test over re-probed repaired tasks (§9.4). A lifted operator faces the same
exact McNemar limits on fresh decision units: at α_t = 0.005 the minimum detectable recall gain is 0.33 at n = 30, 0.20 at
52, 0.17 at 60, 0.085 at 120 and 0.045 at 240 (taskloop-1 §B.4).

**What the paper reports about the loop.**

| Report | Content |
|---|---|
| The per-task loop | Outcome counts, demotions, decisions by move with the rule that decided each, refusals by rule, lifted lessons, the drift alarm, ORACLE's wrong and disputed counts by domain |
| J_r per round | With its interval, against equal-budget control rounds |
| Per test | t, α_t, the e-value, the stopping point, the paired effect with bootstrap and cluster-bootstrap intervals; the run's α spent and decision units left |
| Locked-set and held-out family measures per round | FR, FA, ρ̂; the holdout's access count |
| The P/F game | Headroom, planter rewards, pool class KL and MMD per round |
| δ_AA | Its replicates, beside the between-view spread |
| Ablations (E10) | One regulariser removed at a time, equal budget, ≥ 3 seeds, paired differences to the full loop |
| Failure cases and the ledger | Overfitting, Goodhart drift and collapse; every rejected update, kept as negative evidence |

### 9.7 Policy as data: the lawful-remedy procedure

A synthesis framework meets blocking conditions that its rules do not settle by themselves: a certificate whose bound
needs more probes than the World supports, an adjudication panel that is not calibrated, a long task no reference solver
passes, a task too hard for the student. A pipeline that settles them ad hoc moves its own goalposts: the easiest
remedy for a blocked task is usually to relax the rule that blocked it, and each relaxation raises yield. We make every
such decision data and give one procedure that resolves blocking conditions from stated principles (`synth/policy/`,
`runs/policy-1.md`).

**Principles.** An ordered list Π = (π₁, …, π₁₀). Each principle is a short statement and a test `test_π(m, c, P)` that
returns its objections to a candidate remedy m for a condition c under policy P (empty: no objection).

| # | Principle | Its test refuses |
|---|---|---|
| π₁ | **P-valid.** Never relax a validity or admission rule to raise yield. | A change to a validity parameter in its loose direction; counting held-out trials; segmenting a whole-task rule; admitting with open suspects |
| π₂ | **P-evidence.** Prefer executed evidence; a panel verdict is necessary, never sufficient; an uncalibrated panel abstains. | Admission on a judgement alone; a panel decision while the panel is uncalibrated |
| π₃ | **P-more-evidence-first.** An underpowered decision first gathers more evidence. | (The order of the procedure) |
| π₄ | **P-intent.** The skill and cell are frozen except through an evidence-derived RESKILL. | A skill change outside RESKILL; a RESKILL not derived from executed evidence; derive → given as an easier move |
| π₅ | **P-no-delete.** Nothing is deleted. | Deletion |
| π₆ | **P-student.** Student probes may inform task decisions within caps; student reward never enters J. | Student reward in J; a repair justified by a pass rate; rollouts past the cap |
| π₇ | **P-equal-budget.** Every claimed improvement is compared at equal spend. | A claim without an equal-spend control; a budget change that favours one arm |
| π₈ | **P-family.** No family checks its own work. | A builder's or the student's family as solver, attacker, judge or deriver |
| π₉ | **P-person.** A person decides only budget beyond the total, sealed or held-out data access, irreversible or external actions, and changes to the principles. | An escalation on any other ground; any of those four effects without one |
| π₁₀ | **P-ledger.** Every rule change is a versioned ledger entry; every run records its policy version. | A remedy that changes a rule parameter (except a stage cap inside the total) |

**Rules.** A rule ρ = (id, K_ρ, θ_ρ, L_ρ, Π_ρ, source, version) is the codified form of one decision already taken: the
condition kinds K_ρ it applies to, its parameters θ_ρ (for example h = 0.2), its ladder L_ρ (an ordered list of steps),
the principles Π_ρ it derives from, and its source (the person who decided and the date, or the code and run note an
inferred rule was read from). A step is (a, φ, K′): an action a, a predicate φ over the condition's facts, and optionally
the subset K′ ⊆ K_ρ it serves.

**Closed vocabularies.** Three sets live in code because code implements them, and change only with the code:
- the actions A, each with a rung r(a) ∈ {evidence, repair, relabel, variant, defer, escalate}, its declared effects
  and the call site that carries it out (34 actions);
- the validity parameters V ⊆ (rule × parameter), each with its strict direction (lower, higher, true, false, superset,
  subset, fixed): π₁'s data (37 parameters);
- the escalation grounds G = {budget beyond the total, sealed access, irreversible action, principle change}: π₉'s data.

**Conditions.** A condition c = (kind, subject, F), where F are facts the caller measured: availability and counts
(`can_densify`, `shown_n`, `panel_calibrated`, `rungs_tried`, `planned_usd`), never a judgement. A predicate on a missing
fact is false, so resolution falls toward later, safer rungs (fail closed). 25 condition kinds are defined.

**The procedure.** For a condition c, a policy P and optional proposed remedies X (a model's or a script's suggestion):

```text
resolve(c, P, X):
  R  ← {ρ ∈ P : kind(c) ∈ K_ρ}   or, if empty, {R-default}           # the principles alone
  C  ← [(ρ, s) : ρ ∈ R, s ∈ L_ρ, kind(c) ∈ K′_s] ++ X
  sort C stably by rung: evidence < repair < relabel < variant < defer < escalate
  for m in C:
      if m is a step and not φ_m(F): skip                              # inapplicable on the facts
      build m as a Remedy                                              # relaxing V cannot be built: π₁ by construction
      if ∃ π ∈ Π: test_π(m, c, P) ≠ ∅: record the objections; skip
      return m (with every skipped candidate and why)
  return HOLD                                                          # lawful under every π
```

**Properties.** Each holds by construction and is checked by a test.
1. *No relaxation.* No returned remedy changes a validity parameter in its loose direction: a `Remedy` carrying such a
   change raises at construction, and π₁'s test refuses declared relaxing effects. Across versions, no version relaxes V
   except through a person's answer to a principle-change escalation, so the admission rule a run applies is never looser
   than the one before it unless a person changed a principle, and the ledger names who and when.
2. *Evidence first.* A remedy past the evidence rung is returned only when every evidence step for c was inapplicable on
   F or refused by a principle.
3. *Totality.* `resolve` always returns: HOLD is lawful under every principle.
4. *Bounded escalation.* `resolve` returns an escalation only when its ground in G is borne out by F (for example
   `planned_usd > part2_remaining_usd`); an `Escalation` outside G cannot be constructed.
5. *Auditability.* `resolve` is a pure function of (c, P, X). The remedy records the policy version, the rule, the
   action's call site, and every candidate passed over with its reason.
6. *Closed actions.* A proposed remedy whose action is not in A is refused; a rule whose ladder runs out of rung order or
   names an unknown action is not a rule.

**Escalation and versioning.** An escalation is a typed record (id, ground, condition, question, options, the policy
version it was asked under). Every option is a ledger amendment or none. The person's answer is one ledger entry of kind
`answer`, so even a "no" is a new, recorded version. The answer never becomes a code change.

**The ledger.** Rules and principles live in an append-only, hash-chained ledger; a policy version is a ledger position
(`policy-v<n>@<sha>`), and every certificate and run report records the version it ran under. The ledger refuses:
- an entry that relaxes a validity parameter or changes a principle, unless it is a person's answer to a principle-change
  escalation (this binds the lead too);
- a change to the Part 2 total, unless it answers a budget escalation;
- from an inferred source, anything but adding a rule, tightening a validity parameter, or raising a stage cap within the
  recorded remainder of the total. Rules are retired, never removed, and retiring one is a person's decision.

**What the procedure does not do.** It does not measure: facts come from executed checks, and a weak check stays weak.
It does not run the remedy: the action's call site does, under the same certificate and archive rules. It cannot invent
a remedy: the action vocabulary is closed. A blocking condition no rule names (the R-default case) is held with its
condition recorded, which is the cue for a new rule entry.

**The rules at genesis** (22 rules; the ledger's first 23 entries, `policy-v23@e2834a86285f`).

| Rule | Source | Applies to | Ladder (first applicable, lawful) |
|---|---|---|---|
| R-admission-fixed | lead, 2026-10-30 | Confirmed counterexample | Repair with counterexamples (≤ 3 rounds, monotone archive) → reject |
| R-holdout | lead, 2026-10-30 | Underpowered certificate; held-out survivor | Densify the World and re-certify (≤ 2) → provisional; survivor → reject |
| R-ALT-whole | lead, 2026-10-30 | No solver pass | Check the oracle → solver ladder L0 → L1 → L2 (equal spend, independent seeds) → reject "ALT:no_pass after the solver ladder" |
| R-too-hard | lead, 2026-10-30 | Band too hard / too easy | ORACLE → REPAIR → RESKILL → DESCEND (skill-preserving knobs) or ASCEND → RESERVE |
| R-band | lead, 2026-10-29 | Band undecided | More fresh rollouts to k_max = 16 within the probe cap → HOLD |
| R-uncertain | inferred (outcome.py) | Mostly uncertain rollouts | Gather the task's executed evidence → HOLD |
| R-student-probe | lead, 2026-10-29 | Probe cap reached | HOLD |
| R-panel | inferred (falsifiers-2; compose-2) | Uncalibrated panel; open suspects | Executed check → more anchors → calibrated panel → human queue |
| R-oracle | lead, 2026-10-30 | Wrong / disputed / dead oracle | Executed rules → calibrated panel → repair / retry → reject / drop / human queue |
| R-density | lead, 2026-10-29 | Density gate; other World gate | Re-seed denser → rebuild the stage → not used |
| R-budget | lead, 2026-10-29 | Over budget | Reallocate inside the total (equal across arms) → trim → escalate beyond the total |
| R-scale | lead, 2026-10-29 | Scale shortfall | More families → more synthesis → report the shortfall |
| R-family | inferred (roles.py) | Same-family check | Reassign → HOLD |
| R-equal-budget | inferred (§7, R9) | Unequal comparison | Equal-spend control → no claim |
| R-rsi-power | lead, 2026-10-29 | Underpowered update | Larger fresh view → no update |
| R-reserve | lead, 2026-10-30 | Delete request | RESERVE |
| R-sealed, R-irreversible, R-principles | lead / inferred | The three person grounds | Escalate |
| R-seed, R-patch-scope | lead / inferred | (Constraints) | (Parameters only) |
| R-default | the principles | Any other kind | Executed check → HOLD |

**The solver ladder (R-ALT-whole).** The rule stays: a non-builder family passes the whole task from its instruction.
On `ALT:no_pass`, the ladder runs stronger configurations rung by rung on the whole task: L0 the certifying set (gpt-oss,
Nemotron-3-Super, Inkling-Small; harness v3; 1.5 × horizon + 6 calls), L1 stronger models (Nemotron-3-Ultra v2, Inkling
v3), L2 the long call budget (3 × horizon). Every configuration of a rung gets the same per-task budget (USD 0.12), each
attempt its own tag (independent seeds), builder and student families are skipped, and the ladder stops at the first rung
with a pass. Failed attempts go to the adjudicator: a valid failed attempt is a confirmed D6, an undecided one a
suspect, and the certificate is re-decided under its own admission rule.

**What the build found** (each fixed, with a test; `runs/policy-1.md` §2):
1. the pipeline's panel decided while uncalibrated (the M4 rule set λ to 0.95 when no λ reached the target). R-panel's
   strict rule (≥ 6 decided anchors of both truths, else the panel abstains) is now the default;
2. the pipeline's held-out share defaulted to 0. It now defaults to the ledger's h = 0.2;
3. the certificate's default admission policy lacked ALT, ORACLE and LEAK, so a direct `certify` admitted a confirmed
   ORACLE finding. The defaults are now the rule;
4. looser policies were silent. Certificates now record `deviations` and `policy_version`, and a deviating certificate
   is kept out of the pool;
5. the repository exemption from the ALT rule had no name. It is now R-patch-scope, scoped to `repository`;
6. nine budget constants were scattered over the wire and five scripts. They now read R-budget.

**What the paper reports.** The policy version of every result; the ledger (entries, sources, answers); per run, the
count of each remedy by rule and action, the escalations raised and how they were answered; and the conformance tests
that pin each code constant to its rule.

## 10. Difficulty per student, and composition 

**Composition.** seq(τ₁, τ₂) starts τ₂ from s_T(π₁*):
- the program is π₁*; π₂*;
- R = R₁ ⊕ R₂, where a later write to the same address supersedes the earlier one;
- A = A₁ ∪ A₂.

The instruction is rewritten and re-checked by TAINT. par(τ₁, τ₂) requires disjoint address sets.

**Rules settled by compose-2.**
- **Certify the parts first.** A chain link joins only after it is certified on its own World state, and a composite
  from admitted parts is certified again as a whole. Per-part repair edits only the part that owns a counterexample.
- **Restoration gap.** When one part clears a value and another restores it, a merged allowed pattern could still
  permit the cleared value (a D1 found by MUT DropStep). `close_undone` makes each merged pattern exclude the values the
  composite never leaves.
- **The ALT rule is unchanged.** A non-builder family must pass the whole composite from its instruction; the solver
  ladder (§9.7) may escalate the solver configuration, never segment the task.

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
(principle 15). Knobs are applied by the per-task loop at creation time (§9.3): ASCEND takes one declared harder step,
and DESCEND only a skill-preserving easier one (undo a recorded harder knob, `fewer_distractors`, `unpaginate`). Moving a
value from derive to given is never an easier step: it is a RESKILL to a lookup-and-apply skill. A knob the World cannot
take is skipped for the next declared knob, and a lineage with none is reserved.

**Band.** The keep band is [0.1, 0.6] on the student, measured on probe rollouts by a sequential test in looks of 4, 8, 12
and 16 (R-band), with fresh rollouts for every acceptance (§9.4). The difficulty predictor (static features, ALT pass
share, a proxy model) beats a constant and ranks tasks, but its band accuracy of 0.44 (Wilson 0.34–0.54) is under the
trust bar of a 0.7 lower bound, so it never skips a rollout (perfect-1). The DifficultyRecord is keyed by (task, student,
checkpoint, simulator, time mode).

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
no D2 guarantee, and the certificate says so. The same holds for ORACLE: a World without executable policy rules gets no
check of its reference decisions beyond the disagreement probe.

### 11.1 World families × instances

An *environment* is a **World instance** = (World family × instance config) (DESIGN §7.7 item 16). The framework builds
families; every domain, design, App, test, data generator, document and colleague is written by the swarm.

- **A family** is a frozen App with its schema, code, blind tests, a data generator, documents and colleagues, built in
  stages: spec (`DomainSpec`: Apps, skills, knobs, intent-card schema, typed policy parameters, business rules, the
  permission model) → design (held to a realism surface: a paged list tool, filters, write tools with coded errors, the
  acting user from the session, never a tool argument) → schema → code → blind tests by another family → debugging that
  edits code, never tests → data generator (a closed, sampled data language with planted noise: near-duplicates, stale
  clones, conflicting master copies) → documents → colleagues → freeze (digests of every family file).
- **An instance** costs no model call: a seed and a profile (standard, dense, noisy, clean), a policy variant within the
  spec's constraints, documents rendered with those values, and an acting user chosen by execution (the role whose
  reference oracles cover the most target collections).
- **Seven gates, recorded per instance.** Build, determinism, integrity, surface, obscurity, density and the realism
  panel. An instance failing any gate is not used.
- **The listing rule** (worlds-2). Every collection a task may act on has a paged read tool named for it: page
  parameters, no other required argument, the collection's head noun in its name. It is checked at the design stage
  (back to the designer) and executed at the code stage and in every debug round (every paged read tool paged through,
  unfiltered and per enum filter value). The executed check asks only what the design rule asked, since the coder cannot
  add a tool. In worlds-1 two families failed obscurity for want of it (insurance claims 29–38% reachable, returns
  refunds 18–26%); rebuilt under the rule, insurance claims are 100% reachable.
- **The density gate** is on the **shown** mutants (`MUT(split="shown")`, h = 0.2): every target collection holds
  ≥ K entities and is written by some reference oracle reaching n ≥ 59 shown distinct mutants; held-out counts are
  measured beside it. The data stage measures it before freeze and feeds per-tool reach and refusals back to the data
  writer. A failing density gate raises a policy condition whose remedy re-seeds the instance denser (R-density), up to a
  cap, then rebuilds the stage, then marks it not used.
- **The realism panel is necessary, never sufficient.** It separates toys from plausible Worlds but does not rank
  plausible ones (§5.2 item 7); when uncalibrated it abstains and the executed realism gates decide.
- **Superset density for an existing task** (taskloop-1). Re-drawing a World with more entities makes another World
  (the task's named entity changes). `densify` keeps the original World unchanged and adds k entities from a second
  random stream, on new related records only; π* must pass with the same effect and answer, and every observation must
  be the same or only grown. A changed observation marks the task density-sensitive, a finding and never an edit.

Results are in §14.3.5 and §14.3.9.

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
| `synth/falsify/{base,mut,null,red,taint,twin,pert,rep,agents,online}.py` | The nine classes plus the registry; `MUTHeldOut` (MUT-HO) | `accept/mutation/actions.py` patterns |
| `synth/falsify/{oracle,twinjudge,leak}.py` | ORACLE (policy truth, decision disagreement); inline TWIN adjudication; the LEAK class | — |
| `synth/adjudicate.py` | Panel protocol, λ calibration hook, human queue | `factory/goldset.py` anchors (through injection) |
| `synth/certificate.py` | Clopper–Pearson, admission policy, certificate JSON | `factory/flake.py` bound |
| `synth/archive.py` | Hash-chained counterexample archive, regression runner | `factory/labels.py` style |
| `synth/cegis.py` | The per-task loop | — |
| `synth/meta/{plant,recall,library,lemmas}.py` | Contract mutation, the FR/FA matrix, the operator DSL plus birth gate, the lemma bank | — |
| `synth/compose.py`, `synth/parts.py`, `synth/solvers.py` | seq and par composition, dependency depth, knobs and `VariantWorld`; certified parts and per-part repair; the solver pool and the ALT solver ladder | `forge/compose.py` ideas |
| `synth/{learn,intent,novelty,perfect,outcome}.py` | P2 band (sequential looks, predictor), P3 probes (EVID, SWAP, TAINT, TWIN), P4 signatures and coverage, the perfect-task check, the outcome classifier | — |
| `synth/meta/{rsi,reskill,loopkit,stats,update,headroom,ablate,par}.py` | `TaskLoop` (the per-task RSI loop), RESKILL, the stored-pool kit; the update rule's statistics (A/A, betting, alpha spending, Thresholdout, power), headroom, E10 ablations, the shard pool | — |
| `synth/policy/{model,genesis,ledger,resolve}.py`, `policy/ledger.jsonl` | Principles, rules, closed vocabularies, `resolve`, the hash-chained ledger | — |
| `synth/{families,worldgates,datagen,worldkit,catalogue}.py`, `synth/densify.py` | World families, instances and their seven gates; superset density | F1 World synthesis |
| `synth/recert.py` | Re-certification of stored task versions (phases 1 and 2), the pool manifest | — |
| `synth/domain.py` | `DomainSpec`: Apps, tools, skills, knobs, intent-card schema, cell grid | — |
| `synth/roles.py` | Role prompts (data) and the role → family constraints | `factory/swarm_prompts.py` |
| `factory/synth_wire.py`, `scripts/synth.py` | Binding to Tinker, the CLI (`propose`, `certify`, `plant`, `recall`, `evolve`, `report`), spend caps | `factory/models.py`, `factory/tinker.py` |

**Storage.** Data lives under `crucible/data/synth/<domain>/`: tasks, certificates, the archive, planted sets, spend.jsonl.
The locked planted set lives under `crucible/data/sealed/synth-locked/` behind the seal guard.

## 13. Experiments (what the paper shows)

| ID | Question | Design | Metric | Budget |
|---|---|---|---|---|
| E1 | Does the falsifier population detect planted defects, and which classes catch what? | FR/FA matrix on the locked set across 3 domains | FR_k,d with Wilson CIs; FA | CPU plus adjudication calls |
| E2 | Do system-level updates lifted from RSI raise recall without false alarms? | The falsifier loop for R rounds under §9.6; held-out World family and locked-set FR per round; equal-budget control: random DSL operator proposals | FR curve; FA; ledger of rejects; α spent | Low $ |
| E3 | Do operators transfer across domains? | Library grown on domain A, tested on locked planted sets of B and C | Transfer FR vs B's own library | Low $ |
| E4 | Do certificates predict real hacking? | Run the student (eval, then RL) on admitted tasks; ONL flags vs certificate strength. This evaluates certificates; it is not an RSI input | Rank correlation of U and FR-weighted strength with the confirmed hack rate | Student rollouts |
| E5 | Does certified data train better? (dose–response) | RL at equal budget on pools from the same generator, admitted at increasing certificate strength: (a) two-sided filter only, (b) deterministic falsifiers, (c) all falsifiers + archive, (d) (c) + online certification | Hack onset step, clean-vs-raw pass-rate gap, held-out family score, exploit switching after quarantine; MDE stated first | DeltaAI GPU |
| E8 | Are planted defects like natural ones? | FR per class on planted vs the real-defect set; capture–recapture vs error-seeding agreement | Per-class calibration gap with CIs | CPU |
| E9 | Head-to-head with published pipelines | Re-implement the published recipes of ScaleEnv (2602.06820), AWM (2602.10090) and a two-sided-filter-plus-judge pipeline as arms in the same Worlds, at equal dollars. Same student, same RL budget. | Residual-defect rate ρ̂ (planted + real); confirmed hack rate under RL; held-out family score; cost per perfect task | GPU + low $ |
| E10 | Do the regularisers matter? | §9.6 ablations: remove one regulariser at a time | J, locked FR, transfer, Goodhart drift (cell KL, task size) | Low $ |
| E6 | Factory economics | Yield and cost per stage; with vs without lemmas | Cost per admitted task with CIs | Low $ |
| E7 | Generality | The same code on SQL Apps, repos (debug lane), messaging Apps and one Gym env | Which classes apply; FR per environment class | CPU |
| E11 | Does the per-task loop give every task a correct decision without tuning to the student? | `TaskLoop` on the training-side pool, then the 10 × 10 pool (prereg §8, H5, H7–H11) | Decisions by move and rule; refusals by rule; the drift alarm; accepted variants moving in their declared direction on fresh rollouts and solved by an independent family; RS1–RS6 per accepted RESKILL | Student probes + low $ |

E1–E3, E6–E7 and E11 need no RL. That is deliberate: the paper's core claims are cheap to measure, and RSI itself (E11,
E2) runs at task creation. E4 and E5 tie the framework to training.

## 14. Build milestones

Each milestone has unit tests that fail on the code before it, a DESIGN.md entry, and the merge procedure in the
repo rules.

| M | Scope | Acceptance |
|---|---|---|
| M1 | `canon`, `iface` (`PyWorld` + `CrucibleWorldAdapter` on finance_ap), `program` with taint, `contract` + `grade`, `infer` | finance_ap oracle programs run through the adapter and pass their inferred contracts. Hand-built planted contracts (one per D1–D11) produce the right reason codes. The grader fails closed on injected exceptions. |
| M2 | Falsifiers MUT, NULL, RED, TAINT, TWIN, PERT, REP; `certificate`; `archive`; `cegis` with a stub repairer | On PyWorld tasks every deterministic class finds its planted defect class. An always-pass contract is rejected by MUT and NULL. Archive regressions block a repair that re-opens a counterexample. |
| M3 | `meta/plant` (planter with witness validity and natural-likeness features), `meta/recall`, the dev and locked sets, the FR/FA matrix report; `meta/update.py`: the regularised update rule R1–R10 (fresh views, noise floor, non-regression, cell-KL trust region and entropy floor, annealed edit budget, cost bar and pruning, leakage-critic hook, transfer check, round caps, rollback); the operator DSL and birth gate | The FR/FA matrix prints for PyWorld domains. A candidate operator passes or fails the birth gate as expected. Planted updates that violate each of R1–R10 are each refused. |
| M4 | `roles`, `synth_wire`, LLM proposer, oracle programmer, contract writer, repairer, ALT/ATK agents, adjudicator, abstraction, on Tinker; PyWorld Apps for helpdesk (tickets, kb, mail) and finance_ap | End to end on 2 domains: ~20 proposed tasks each, certified or rejected with reasons; spend and yield per stage reported. |
| M5 | `PatchTask` adapter for the debug lane (fs canon + exec predicates); the historical real-defect set; ONL hook for eval rollouts | Stored debug-lane grades recompute identically through the adapter. Historical defects load as the real set. |
| M6 | `compose` + dependency depth; Gym toy; E1–E3, E6, E7 runs | Report tables for the paper. |

**Tinker budget for building and checking.** $150 in total for M4–M6, logged in `data/synth/**/spend.jsonl`, with a
hard cap in the wire. The checks use the swarm families (Kimi, GLM, DeepSeek, Nemotron, gpt-oss, and Inkling since
falsifiers-2). The student is called only as the creation-time probe (learnability, the per-task loop) and in E4. Spend
beyond $150 needs the lead's approval (R-budget, §9.7).

### 14.1 Status after M1–M4 (2026-10-29)

**Built:** M1–M5 (M5 in §14.2), then DESIGN §7.7 items 16–24 (§14.3): World families, the batch runner, composition
and knobs, learnability / intent / novelty and the outcome classifier, the falsifier blind spots, the RSI statistics and
the per-task loop, re-certification, densify, and policy as data. **Not built:** the Gym demo; the full experiment runs
(E1–E11) on the 10 × 10 pool. *(This line updated 2026-10-30.)*

**The first end-to-end runs (M4).** The whole run was on the Tinker swarm, with no Qwen model in any role and the
student never called. The run reports are in `runs/m4/`.

| Domain | Proposed | Oracle executed | Certified | Provisional | Rejected | Cost |
|---|---|---|---|---|---|---|
| finance_ap (existing pack) | 20 | 20 | 2 | 2 | 16 | $1.07 |
| it_helpdesk (World synthesised from the keyword, $0.48: 10 tools, 6 tables, 25 blind tests) | 20 | 19 | 2 | 1 | 16 | $1.94 |

Total M4 spend, smokes included, was $7.91 of its $60 cap.

**What the falsifiers found in swarm-written tasks:**
- TWIN D9: the contract cannot tell a sibling-category task apart;
- MUT ExtraWrite D2 and SwapEntity D3/D9;
- RED D4: requirements already true at the start;
- NULL D8/D1/D4;
- TAINT D5: ids never listed, and TAINT D7: decisions given away in the instruction;
- one ALT D6: a valid alternative solution the contract rejected.

**The binding rule today** is "ALT passes from a non-builder family". Most rejections are solvers misapplying the
domain policy, and the panel confirms those attempts are invalid.

**Calibration found a real judge flaw.** One panel family rated "read everything, never act" traces valid at 0.95
confidence. The panel prompt was tightened, and λ was set from executed-truth anchors.

**Decisions taken from the M4 findings:**
1. **Admission** also requires zero confirmed ALT counterexamples, i.e. no confirmed D6.
2. **Normalizers** are fixed before skeleton inference, because `id_canon` re-keys inserted rows.
3. **The TAINT free-literal rule.** A literal is not D5 when π* still passes with it replaced. Transformed values,
   such as a normalised invoice number, are still flagged: a known precision gap.
4. **The repairer** differs in family from the contract writer.
5. **What the panel sees.** By default it sees reason codes (§5). The blind-panel variant (no verdict at all) is kept as
   an ablation.
6. **Attackers** see the strategies and the instruction, never the World's failure logs.

**Open items:**

| Item | Detail |
|---|---|
| `cegis` final re-run | Closed in RSI-1: the final pass adjudicates its suspects (`FinalPass`). |
| PPI intervals for adjudicated rates | Closed in RSI-1 (`meta/ppi.py`). The labels are executed-truth anchors, not suspects: the interval assumes the panel errs alike on both. |
| Family exhaustion | Three builder families leave two checker families (Nemotron, gpt-oss) for ALT, ATK, the panel and the abstractor, so the panel judges its own family's ALT traces. falsifiers-2 found and wired a sixth non-Qwen family on Tinker (Inkling, Inkling-Small); the default panel is not switched, because Inkling is a lenient judge of code diffs and needs its own calibration on tool traces first. |
| Short generated tasks | Closed in RSI-1: denser seeded Worlds (`reseed`, `--invoices`) give 22 of 24 new tasks ≥ 59 distinct mutants. The M4 counts were also inflated (a panel-wrapped MUT lost its effect deduplication: 66 reported, 20 distinct). |
| The RSI loop (§9) | First real rounds in `runs/rsi-1.md`: no falsifier update survives R1–R10 in 5 rounds (the best misses R2 by 0.003), the planter loop finds the humanised-leak gap on dev, the DSL cannot close it, the lemma test is refused. Re-tested under §9.6 the best candidate is refused too (§14.3.6); the per-task loop's first real run is §14.3.9. |
| Human queue | 53 panel abstentions on it_helpdesk (`human_queue.jsonl`) await a decision: 51 distinct items in 10 groups; deciding the 3 items of 2 pivotal tasks can make them provisional, and the other 9 tasks stay rejected on confirmed counterexamples (falsifiers-2) |

### 14.2 M5: the repository substrate and the real-defect set (2026-10-29)

**PatchTask.** The debug lane runs through the same contracts and falsifiers.
- **Migration: 12 of 12 training-pool tasks recompute the lane's stored grades bit-identically.** That is 246 rows of
  rewards, verdicts, milestones and mutant vectors. Simplified windows are compared on their own milestones.
- **Certificates, CPU and Docker only (ALT and ATK not run):**
  - no confirmed MUT, NULL, RED or REP counterexample on any task;
  - one confirmed D6 on the swarm-pilot black task: two stale visible tests go red when the bugs are fixed;
  - MUT suspects: fix hunks no hidden test needs, awaiting adjudication. *Correction (falsifiers-2):* the summary file
    `runs/m5/certificates_summary.json` lists **10 suspects** (this text said 11 on 8 tasks);
  - status per the summary file: **4 provisional and 8 rejected** (this text said 5 and 7; n = 15–37 distinct mutants,
    U 0.078–0.15 for the provisional ones). Through the strictly calibrated repository panel every one of the 10
    suspects abstains to a person, and no task changes status (falsifiers-2 §4).

**What changes on repositories (E7).**

| Class | Applies? |
|---|---|
| MUT, NULL, RED, REP, ONL | Yes |
| TAINT | Changed: changelog leaks, stale visible tests and ledgers; literal sourcing is report-only, because π*'s literals are whole file texts |
| TWIN, PERT | No: a symptom report names no entity, and a mined commit has no distractors |

**Spec decisions taken from M5:**
1. **Require vs guard.** Exec checks have a kind. A `require` check must be false at s₀ (RED applies). A `guard`, such as
   a pass-to-pass suite, holds at s₀ by design, and RED skips it.
2. **Code mutants are suspects.** A DropHunk survivor on code is always a *suspect*, never a confirmed D1. Whether
   untested code matters is a judgement.
3. **The admission bound is relative to the World.** Each repository mutant costs a Docker run, and n ≥ 59 is out of
   reach, so repository tasks are provisional at best. The paper reports U per substrate.
4. **D6 covers misleading World feedback.** Stale visible tests and fake links are D6 sub-types. Fake links also touch
   P3, on-intent.
5. **ONL replays end states, not traces.** Eval agents use different tools from the synth World.
6. **Only independent defects calibrate recall.** The real-defect set has 390 records, 351 of them P1 validity defects,
   96 executable. Recall is calibrated **only** on the defects found independently of the falsifiers, by people during
   task fixing: D5 7, D6 21, D7 11, D10 5, D11 3. Falsifier-found defects (archive, ONL) are detected at about 1 by
   construction, and are reported separately.

**First independent recall numbers.**
- **D6: 10/11. D7: 2/3.**
- **The misses are real gaps:**
  - a fake link through one case file, where test ids do not overlap;
  - a changelog line whose fix subject is too short to match;
  - answers leaked through visible test data. No detector exists for this one yet.

### 14.3 Status after the 2026-10-29/30 runs (DESIGN §7.7 items 16–24)

Every number below is from the named run doc in `runs/`. The student (Qwen3.6-35B-A3B on Tinker) ran only as the
creation-time probe; no Qwen model held any other role, and no sealed or benchmark task was read.

#### 14.3.1 Throughput and yield (yield-1; item 17)

| Measure | Result |
|---|---|
| Why proposals failed (M4, RSI-1) | finance_ap: half the oracles were wrong for one proposer (DeepSeek-V4.1-Flash 16 of 31, Kimi-K2.6 0 of 17). 14 of 31 `ALT:no_pass` rejections had a wrong oracle; the other 17 were solver weakness on a deliberately tricky policy |
| Blind oracle programmer (v2, never shown the proposer's derive) | 29 of 31 stored proposals right (v1: 15 of 30) |
| Arms at equal hard caps, pooled over 3 runs | `a1` (v2 prompts, blind oracle with reconcile, pre-filters, ALT v2 harness): 44 admitted for USD 10.10, 4.36 per USD (95% bootstrap 3.23–5.55), 0 admitted with a wrong oracle. `a0` (M4): 23 for USD 10.44, 2.20 per USD (1.38–3.13), 5 with a wrong oracle |
| a1 − a0, admitted per USD | +2.15 (0.71–3.63), P(≤ 0) = 0.002; the gain is finance_ap's, it_helpdesk did not improve |
| The lemma batch | 7 lemmas kept of 9 (the leakage critic refused 2); refused by the lemma test (unequal budget, no fall in the first-round confirmed rate, J did not rise) |
| Speed | 17.0× serial-equivalent in yield-1 (83 admitted per hour, 3 arms side by side); 10 × 10 at the pooled `a1` rate would cost about USD 23 |
| Spend | USD 30.78 of the 35 cap |

Open: admitted finance_ap tasks with a wrong oracle (5 of `a0`), which led to ORACLE (§5.2).

#### 14.3.2 Learnability, intent, novelty, the outcome classifier (perfect-1; item 19)

| Measure | Result |
|---|---|
| Band on the 89-task pool | 47 too hard, 19 in band, 23 too easy; stated error 0.29–0.37 at k ≤ 8 |
| P3 on intent | finance_ap 47 of 54 (0.87), it_helpdesk 12 of 35 (0.34; 21 tasks whose contract accepts the twin's solution) |
| P4 near-duplicate threshold | θ = 0.865, TPR 1.00 (0.98–1.00), FPR 0.008 (0.005–0.011) on 174 known duplicates and 3,401 known distinct pairs |
| Perfect (all four properties) | finance_ap 3 of 54, it_helpdesk 0 of 35 |
| Outcome classifier | 25 of 26 fixture cases decided, accuracy 1.00; on 676 real rollouts 424 intended fail, 132 intended pass, 52 unintended fail, 1 unintended pass, 67 uncertain |
| Frontier round (superseded by the per-task loop) | targeted arm 0 perfect, control 1, n = 4 per arm: no difference |
| Spend | USD 5.55 of the 15 cap |

The first pilot's rollouts were all copies of one sample (seed 0); the seed-0 store was set aside and re-measured with
seeds by tag.

#### 14.3.3 The falsifier blind spots (falsifiers-2; item 20)

| Set | Before (M3 population) | After (read mutants, layered TAINT) |
|---|---|---|
| rsi-2 locked, 72 plants | FR 56/72; D6 0/12; D7 14/18; FA 0/7 | **FR 72/72**; D6 12/12; D7 18/18; FA 0/7 |
| rsi-1 locked, 65 plants | FR 49/65 | **FR 65/65** |
| dev, 125 plants | 118/125 | 118/125 (the only miss: `relax_value:eq_to_any`, D3, MUT suspects only) |

The L2 leak probe confirms 18/18 planted D7 on both locked sets. The repository panel never reached the 0.9 target on
diffs, so all 10 open repository MUT suspects go to a person. Spend: USD 1.93 of the 15 cap.

#### 14.3.4 Composition (compose-2; item 18)

| Measure | Result |
|---|---|
| Long composites with dep ≥ 1 | **0** built: follow-up links 0 of 7 certified on their own World state (TAINT D5, ATK shortcuts, undecided TWIN suspects, ALT failures); first links 2 of 13 |
| dep 0 composites from certified parts | 2 built (23 and 22 calls), 0 confirmed counterexamples in any class; both rejected: one open LEAK suspect (a panel split), and ALT 0/3 |
| Solvers | The v3 harness turns gpt-oss from 0/6 to 5/6 on certified single tasks. No solver passed the four-decision composite; the best attempt decided 3 of 4 invoices right |
| TWIN calibration | With the corrected anchors the stored votes re-score to λ = 0.5; 25 open TWIN suspects become confirmed D9 |
| Spend | USD 7.17 of the 8 cap |

#### 14.3.5 World families (worlds-1, worlds-2; item 16)

| Run | Result |
|---|---|
| worlds-1 | Catalogue of 45 domains; 5 families started, 4 frozen; 20 instances, **12 usable** under all seven gates, from 2 families (AR collections, leave and absence). The realism panel rated all 20 realistic, including the 8 the executed gates reject. USD 24.15 of the 30 cap |
| worlds-2 | The listing rule and density-by-design in the builder. Insurance claims rebuilt: 6 of 6 instances usable, claims 100% reachable (was 29–38%). **18 usable instances from 3 families**; the target of ≥ 4 families is not met (returns and purchase requisition are unfinished for budget). With 4 gate-rejected anchors the realism panel is 7 of 11 and uncalibrated, so the executed gates decide. USD 5.56 of the 6 cap; the shared Part 2 total was USD 106.15 of 150 after this run |

#### 14.3.6 RSI statistics and the RSI-2 pilot (rsi-stats, rsi-2-prereg; item 21)

| Measure | Result |
|---|---|
| A/A noise floor | δ_AA = 0.0196 on the rsi-2 dev pool (125 units), 0.0327 on RSI-1's 75-unit view, against RSI-1's between-view δ of 0.0407; per-unit seed discordance 0.64% |
| RSI-1's best update, re-tested | 7 gains, 0 losses on 75 units (mean 0.093, cluster bootstrap 0.028–0.159); e-value 4.2, refused: the binding constraint was the sample |
| Run-level error (simulation) | 0.000 for 5–40 null candidates, against 0.20–0.80 uncorrected |
| Headroom | Dev 7 of 125 plants (all panel-only D3 `eq_to_any`, all expressible); locked 0 after item 20, so the locked set measures non-regression only |
| Pilot on Tinker | No update accepted, as it must not be: the operators that close the only dev gap can show 3 gains in 64 decision units, under the MDE of 0.27 at n = 60. 14 framework problems found and fixed. The E10 pilot (3 ablations × 3 seeds) is uninformative: every Δ is 0. USD 0.0269 of the 30 pilot cap |

RSI-2 is preregistered (H1–H11) with Amendment 1 (RESKILL and RESERVE) and runs only after the lead's go.

#### 14.3.7 Re-certification of the stored pool (recert-1; item 22)

Four bugs weakened every earlier certificate: calls on seed 0, Ultra and Super sharing one ALT runner name, ORACLE and
LEAK outside admission, and the held-out mutants off.

| 195 stored task versions | Certified | Provisional | On hold | Rejected |
|---|---|---|---|---|
| Stored statuses | 103 | 14 | — | 78 |
| **Re-certified, h = 0.2** | **4** | **90** | **10** | **91** |
| The same evidence at h = 0 (report only) | 66 | 30 | 10 | 89 |

Of 117 stored admissions: 14 hold, 64 become provisional (59 for the held-out share), 10 go on hold, 29 fall (ORACLE 7,
LEAK 5, TWIN 3, layered TAINT 3, among others). 16 stored rejections rise to provisional (a solver that never ran
before passes 10). Spend: USD 5.11 of the 8 cap.

#### 14.3.8 Policy as data (policy-1; item 24)

10 principles, 22 rules, 34 actions, 37 validity parameters and 25 condition kinds; the shipped ledger has 23 entries
(`policy-v23@e2834a86285f`). The build found and fixed six places where code was looser than the lead's decisions
(§9.7). 65 tests pass, among them property tests over 400 random conditions with proposed relaxing remedies: no returned
remedy relaxes validity, fails a principle, or escalates outside the four grounds. No model call, no spend.

#### 14.3.9 Denser Worlds and the first run of the per-task loop (taskloop-1; item 23)

**Part A, superset density.** At MUT's 240-mutant cap density does not raise n (read-step sibling swaps share one
effect), so dense certificates use 480 (recorded, never bound). Of the 60 held-out-blocked tasks, 47 reach 59 shown
mutants at +15 invoices, 8 at +30, 2 at +45, 1 at +60, 1 at +90, and 1 cannot (its neighbourhood is capped by MUT's
sibling sample, not the World). Full re-certification: **53 certified**, 1 provisional, 2 on hold, 4 rejected. The pool
TaskLoop starts from: 57 certified, 32 provisional, 10 on hold, 96 rejected (of 195).

**Part B, the per-task loop** (195 tasks, 3 rounds, 51 minutes, policy-v23).

| Move | Decisions | Accepted | Final lineages |
|---|---|---|---|
| REPAIR | 74 | 3 (rejected tasks repaired from their shown counterexamples) | — |
| RESKILL | 25 accepted (+ 33 refused attempts) | 25 (2 kept, 23 reserved under the new label) | — |
| DESCEND | 0 | 0 | — |
| ASCEND | 52 (27 lineages) | 0 | — |
| KEEP | 17 | 17 | **kept 19** |
| RESERVE | 23 | — | **reserved 71** |
| HOLD | 35 | — | **held 85**, **rejected 20** |

- Refused repairs cite T5 53, G2 25, the repairer declined 16, T4 7, T3 2, T2 1, G1 1 (a decision can cite several).
  An invalid task whose repair is refused is held, never reserved or kept.
- RESKILL refusals: RS3 18 (mostly a re-sampled ATK or ALT suspect), RS4 15 (not novel in the new cell).
- The band before the loop, on the 90 valid tasks that reached it: 10 too hard, 22 in band, 58 too easy.
- **Perfect tasks: 8** (kept, certified, on intent, novel), all densified finance_ap tasks. USD 12.29 for the loop
  (student 7.45, swarm 4.84), USD 1.54 per perfect task; USD 16.60 with Part A, the pilot and the re-runs, of the 28 cap.
- The drift alarm did not fire and is not testable at this size (3 re-probed repaired tasks; it needs ≥ 5). The only
  lifted cluster (52 PROBE counterexamples on 7 tasks, hidden information) is already caught by TAINT, so it is no
  headroom, and the R1–R10 gate could not accept anything at this size.
- 13 framework problems found and fixed. Among them, a wrong-oracle repair refused by T2 because the student's "intended
  fail" was the policy's right decision graded against the wrong oracle (now void in that repair), and a refused variant
  re-tried on fresh rollouts the next round (now never re-tried on its lineage).
- Finding, not changed: a RESKILL re-measures the band. 5 of 7 in-band tasks came out of band on RS4's 16 fresh
  rollouts with the task unchanged, which is the band test's error acting on a second measurement.

#### 14.3.10 Open after these runs

| Item | Detail |
|---|---|
| Ladders need knobs the World supports | finance_ap's SQL World takes no distractors and pagination did not move the band; DESCEND has no lawful move on plain tasks |
| P3 often unmeasured on finance_ap variants | EVID needs a perturbable World; G2 then refuses, fail closed |
| Re-certification re-samples ALT and ATK | A label-only change or a repair can fail on a fresh, undecided suspect |
| Power | The drift alarm and the R1–R10 gate are underpowered at today's sizes; the 10 × 10 pool's decision split supports one or two full-size tests |
| Diversity | 18 usable World instances from 3 families; ≥ 4 families not met |
| Long composites | dep ≥ 1 follow-up links do not certify; no solver passes four decisions in a row |
| Worlds without executable rules | ORACLE can only flag disputes there (3 it_helpdesk disputes held in taskloop-1) |
| MUT's cap sample | Can miss a whole operator site (recert-1 issue 7); `MAX_SIBLINGS` caps one task's neighbourhood. Not changed, because RSI-2's preregistered numbers rest on today's sampling |

## 15. Relation to prior work

### 15.1 System-level comparison with ScaleEnv and the closest pipelines

**Key.**
- ✓ means built in.
- ~ means partial, or done by a model's opinion.
- — means absent or not reported.
- "Ours" is the design as built on 2026-10-30. Entries marked † are not built yet. Measured results exist only for the
  synthesis side (§14.3); there is no RL result yet.

| Dimension | ScaleEnv 2602.06820 | AWM 2602.10090 | EnvScaler 2601.05808 | Agent-World 2604.18292 | AutoEnvScaling 2610.14163 | WildAWM 2610.26828 | Hacker–Fixer 2606.08960 | Verifier evolution 2610.11464 | **Ours** |
|---|---|---|---|---|---|---|---|---|---|
| States the full target (valid + learnable + on-intent + novel) | — | — | — | — | ~ (reward spread) | ~ (gates) | — | — | ✓ P1–P4, §0 |
| World / App synthesis | ✓ tool-first, executed tool tests | ✓ tasks-first SQLite + API | ✓ blind tool tests | ✓ mined real data | ✓ agent in sandbox | ✓ persona → multi-day Worlds | — | — | ✓ World families × instances on the tool-first stages, seven executed gates, the listing rule (§11.1); 18 usable instances from 3 families |
| Reward / verifier form | Gold DB, per-column match | Code-augmented LLM judge | Checklist of state predicates | Tests + rubrics | Harbor tests | Executed verifiers | Existing benchmarks | Evolved detectors | Pass rules as data, one trusted grader, closed-world frame |
| Collateral / side-effect check | — | — | ~ | — | — | ~ | — | — | ✓ FORBIDDEN_EFFECT + ExtraWrite |
| Verifier testing | — (tools only) | — | — (tools only) | — | Reference = 1, empty < 0.5 | 5 executed gates, gaslight probe | Hacker rounds | Typed ops, birth gate | Typed falsifier classes (incl. held-out mutants, ORACLE, LEAK), per-class certificates, archive replayed as a regression suite |
| Communicability / intent | — | — | — | — | LLM review | ~ | — | — | ✓ executed taint + instruction twins |
| Measured residual defects | — | — | — | — | — | — | Found-only (16% hackable) | — | ✓ recall-calibrated estimate per domain and class |
| Difficulty per student | — | — | — | ~ diagnosis | ✓ assignments, spread | ~ | — | — | ✓ band per (task, student, checkpoint, simulator, time mode) + per-task typed decisions at creation |
| Self-improvement | — | — | — | ~ arena loop, no control | ~ proposer memory; the trained proposer collapses without training | — | ~ repair rounds | ~ one level, one domain | ✓ a creation-time per-task loop (typed decisions, T1–T7, G1–G9, RS1–RS6) and system-level updates (falsifiers, planter, synthesizer, abstractor) under R1–R10 with run-level error control |
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
| C4 planted defects | Partly novel, close to novel (2607.27518; BenchGuard 2604.24955; classic error seeding, spec mutation, oracle assessment) | The yardstick of the system-level updates lifted from RSI (§9.5). Called **contract mutation**, not "second-order" or "higher-order mutation" (Jia & Harman 2009 use those terms for multi-fault program mutants). The classic roots are credited. |
| C5 evaluator-side updates | Partly novel (2610.11464 has typed ops, a birth gate and a locked set for verifier evolution) | The falsifier loop of §9.5; new: cross-domain birth gate and transfer, run-level error control |
| C6 online certification | Partly novel (2606.26300; HackTrace; PRIME 2606.09711; 2609.35677) | A training-time safety net, separate from RSI and never an RSI input; tested by the dose–response RL experiment |
| C7 composition | Known (τ² 2506.07982; WorkArena++; RHB; CompoWorld) | Composition and difficulty stage; credited |
| C8 creation-time per-task loop with student probes | Partly novel. Student-adaptive curricula are known (teacher–student curricula, Matiisen et al. 2017; PAIRED, Dennis et al. 2020; Prioritized Level Replay, Jiang et al. 2021; POET, Wang et al. 2019), as is filtering tasks by pass rate (AutoEnvScaling 2610.14163). New: typed outcome classes with executed evidence, repairs that may never ease a skill gap (T2), and a trust region over the pool | §9.1–§9.4; credited |
| C9 ORACLE and held-out mutants | Known ideas, new placement: the test-oracle problem (Barr et al., IEEE TSE 2015); differential checks against a reference; overfitting of repair to visible tests (Smith et al., FSE 2015); held-out test splits | §5, §5.2; credited |
| C10 RESKILL | Close analogue: hindsight goal relabelling (Andrychowicz et al. 2017). New: the label comes from executed evidence by a non-builder family and is re-certified, never from outcomes | §9.3–§9.4; credited |
| C11 acceptance statistics | Known: tests by betting (Waudby-Smith & Ramdas, JRSS-B 2024), Ville's inequality, online Bonferroni and alpha spending (alpha-investing, Foster & Stine 2008, controls FDR and is not used), the reusable holdout (Dwork et al., Science 2015), A/A tests (Kohavi et al.), exact McNemar power | §9.6; credited. New only in combination: run-level error control for a self-improving synthesis loop |
| C12 policy as data | Known ideas: policy as code (e.g. Open Policy Agent), prioritised defeasible rules (Nute 1994), preregistration. New: a closed remedy vocabulary in which a validity relaxation cannot be constructed, and a ledger that admits relaxations only as a person's answer to a principle-change escalation | §9.7; credited |
| C13 World families and gates | Known: procedural generation and domain randomisation (Tobin et al. 2017); World synthesis (ScaleEnv, AWM, EnvScaler, WildAWM). New: the listing rule and a density gate tied to the certificate's mutant bound | §11.1; credited |
| C14 LLM judges | Known limits (Zheng et al. 2023, LLM-as-a-judge). Measured here: panels rate rejected Worlds realistic and judge leaks valid against their own instruction | "Necessary, never sufficient" (§5.2) |

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
| A wrong reference decision hides behind agreement | ORACLE against executed policy rules where the World has them; disputes held otherwise; the report states which Worlds have rules |
| The per-task loop tunes tasks to one student | T2 (never ease a skill gap), sample splitting, the drift alarm, the trust region, independent solvers for variants; pass rates never enter J |
| A panel's pass is taken as proof | Necessary, never sufficient (P-evidence); uncalibrated panels abstain; executed gates decide realism |
| Power is too low to accept any system update | Preregistered MDEs; headroom precondition; "not testable" instead of "no effect"; larger decision splits from the 10 × 10 pool |
| Band error at k ≤ 16 | Undecided bands hold; a RESKILL that re-measures the band is a known finding (§14.3.9) |
| Relaxing a rule to raise yield | P-valid: a relaxing remedy cannot be built, and the ledger refuses a relaxation except as a person's principle-change answer |
