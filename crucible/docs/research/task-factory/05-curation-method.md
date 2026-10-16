# The curation method (M17)

How Crucible curates RL tasks: what is done, by whom, in which order, with which evidence, and when a task or family is
accepted, revised, dropped or retired. It builds on design 03. It is grounded in:
- study 01 (Prime Agent's machine factory and the Rust-rewrite process);
- study 02 (how 30+ local environments and datasets build tasks and validate verifiers, R1–R16);
- study 04 (agentenv).

Thresholds marked *v1* are starting values. They are recalibrated from the first batches; the decision store records why
each one changed.

## 1. Principles (each from evidence)

1. **Ground truth first, autonomy second.** The Rust rewrite let 2,209 agents run unattended only after humans built
   objective checks. Here the objective checks are the ground-truth lanes (§3) and the verifier card (§6).
2. **Every gating check cites a real observation** (R3). Invented checks may screen, never gate. TerminalWorld showed tests
   that encode one reference run; CodeMidas found 20.3% wrong verdicts after six control runs.
3. **Generation and verification never share a context** (Rust rewrite; Prime Agent's reviewer on another model). The
   verifier writer never sees a solution. Reviewers come from a different model family, enforced by config.
4. **Intent comes from observed failures, and learning is judged by transfer** (R8, R9). No source tests whether a task
   teaches its intent; Crucible does.
5. **A swarm only proposes; the deterministic gate decides** (GitSwarm: a nomination is a preference, not verification).
6. **No numeric targets.** Stop on marginal yield and diversity (Rust hill-climb; GitSwarm herding 17% → 90%).

## 2. The unit of curation: a family

A **family** = one intent card + a set of surfaces + one generator recipe + one verifier version + a held-out probe set.
Tasks are accepted one by one. Training uses a family only after the family is accepted (§8).

Lifecycle: **Discover → Specify → Generate → Validate → Calibrate → Accept → Train and monitor → Retire or evolve.**

## 3. Ground-truth lanes

| Lane | Domains | What counts as ground truth | ObservationRecord type |
|---|---|---|---|
| **Executable** | coding, data, anything with a runnable end state | real repositories and real tests; real PRs and commits (a mined fix plus the tests that changed with it); the kernel-migration tasks; executed oracle end states that pass the real tests | real PR + tests; expert end state (executed) |
| **Human** | finance ops, support, legal, any judgement domain | a person supplies the correct end state, answer or graded outcome for a task instance, or grades solver outcomes | human-graded outcome; expert end state |
| **Trace** | all | real model failure traces, which become intent evidence only, never pass/fail labels | observed model failure |

- **The human lane runs through an intake.** An append-only label and decision store with an agentenv explorer plugin
  (study 04: the explorer has no write routes as shipped; a plugin router adds them).
- **Two labellers per item** wherever κ is to be computed. A label carries labeller id, time, item hash and reason.
  Labels are never edited, only superseded.
- **Simulator state and reference-derived checks are second class.** They gate only together with executed mutants (§6.2).

## 4. Discover: where intents come from

1. **Run the policies under study on seed tasks.** On CPU these are cheap OpenRouter models at several rungs, through the
   same harness used for training (native or M16 RLM).
2. **Classify every failure** into the closed FailureCause set: skill gap, spec defect, unstated format, World bug,
   verifier defect, budget, env flake. A model on another family does this (`rubrics_verifier`-style). The human samples
   10%.
3. **Cluster the skill-gap failures by behaviour.** A cluster becomes an **intent candidate** when it has ≥ 3 traces from
   ≥ 2 model families (*v1*).
4. **An IntentCard is written** with:
   - the behaviour named in one sentence;
   - the target failure mode;
   - its evidence (the traces);
   - its difficulty axes (what makes it genuinely harder: longer dependencies, more interacting rules, partial
     observability, distractors);
   - its coverage coordinates.

   A human approves or rejects the card. This is the cheapest high-leverage human decision in the factory.

Intents also come from the human directly ("our analysts fail at reconciling restated invoices"), with traces gathered
afterwards to confirm the failure is real.

## 5. Specify and generate: the per-task machine

The machine of design 03 §2, run for each (intent, surface) cell the root curator assigns.

1. **Spec** (Planner):
   - requirements, each with an id and a prompt sentence;
   - the verifier plan;
   - expected failure causes;
   - for long tasks, the milestones;
   - the surface: domain, World, tools, data shape, prompt style.
2. **World + prompt** (Author):
   - the prompt states every requirement;
   - a realism reviewer (another family) rejects hints a real user would not give.
3. **Verifier** (Verifier writer; sees the prompt, spec and intent, never a solution):
   - one check per requirement id;
   - each check names its ObservationRecord.
4. **Oracle** (separate author): a reference solution; in the executable lane, a real fix where one exists.
5. **Ledger check (deterministic):**
   - every requirement has ≥ 1 check, every check has a requirement, every gating check has a first-class observation;
   - failures go back to step 3 or step 2.

## 6. Validate: the verifier card (per task, then per family)

**6.1 Floors (deterministic, first, cheapest).** All of these must score 0:
- no-op;
- empty action;
- constant answer;
- original state;
- a copy of the reference where copying is not the task.

Failing a floor sends the task back to the verifier writer (R6).

**6.2 Executed probes** (R2). For each requirement, the mutant builder writes:
- ≥ 1 **mutant**, a solution wrong on exactly that requirement and right on all others;
- ≥ 1 **valid alternative**, right but different from the oracle.

They come from a probe library keyed by requirement type: laziness, tolerance, compiled path, behaviour, idempotence,
minimality, side-effect scope. The verifier must reject 100% of the mutants and pass 100% of the alternatives. All are
executed in the declared environment.

**6.3 Hackers** (A13/A16 + the family hacker):
- given the prompt, verifier and intent, the hacker searches for reward without the work;
- a confirmed hack fails the task and becomes a scripted replay that every task in every family must score 0 on.

**6.4 Second verifier and differential:**
- a second verifier is written independently, by another family, from the same prompt and spec;
- both verifiers judge every solver end state;
- any disagreement opens a defect, and the defect's owner (prompt, spec or verifier) is decided by the auditor.

**6.5 Solvers and trace audit** (R4, R7):
- ≥ 2 policies × ≥ 4 attempts per task;
- the auditor labels each failing trace with a FailureCause and flags suspected false passes and false fails;
- any FP or FN is a defect;
- a World bug or spec defect sends the task back.

**6.6 Flake:**
- **30 oracle reruns at each of two concurrency levels** (60 in all);
- the card records the Clopper–Pearson upper bound on the flake rate;
- *v1* gate: the bound is ≤ 5%. With zero failures, 60 clean reruns give a one-sided 95% bound of 4.9%. The earlier "≥ 10
  reruns" could never meet 5%: 20 clean reruns give 13.9%, and the gate needs ≥ 59 clean runs.

**6.7 Long-horizon tests (R11),** for tasks labelled long:
- ≥ N milestones checkable from World state (*v1* N = 5);
- the order and non-redundancy test;
- the **concatenation test:** if the milestones are solved independently as short tasks at the same rate, the task is not
  long-horizon;
- the **shortcut probe:** no path reaches the end state in fewer steps than half the oracle's;
- oracle step count and expert time recorded.

## 7. Calibrate: difficulty that teaches

- **Pass rate per (policy, harness, k)** with Wilson intervals. *v1* window for training tasks: the target policy's pass
  rate is in [0.1, 0.8], which is where 4p(1−p) learnability is useful.
- **Cause mix:** ≥ 70% of failures are skill gap (*v1*). Otherwise the task is hard for the wrong reason and goes back.
- **Making a task harder** (tighten, perturb, stitch) is allowed only along the IntentCard's difficulty axes. A
  realism reviewer rejects added difficulty that is absent from ground truth, such as unrealistic requirement lists or
  hidden-only rules.
- **Batch report:** all-pass / mixed / all-fail counts (CodeMidas: 90 / 4 / 6 shows a wasted batch).

## 8. Accept a family

A family enters training only with:
1. **≥ 20 accepted tasks across ≥ 3 surfaces** (*v1*), and a held-out probe set of ≥ 10 tasks on a surface the family
   never uses. The probes share no World, template or verifier code with the family.
2. **The family VerifierGoldSet**, sized by the bound it must prove. The bound is per class: truly failing end states give
   the false-pass rate, truly passing ones the false-fail rate. Items are real passes, real fails, mutants, valid
   alternatives and confirmed hacks; at least half are real solver outputs.
   - *v1* gate: with zero errors, Wilson upper bounds ≤ 10%. That needs ≥ 36 items per class.
   - Before large-scale training: ≤ 5%, which needs ≥ 73 per class.
   - The earlier "≥ 40 items, ≤ 5%" was impossible. In the executable lane, mutants and valid alternatives make both
     classes cheap. In the human lane, the class sizes are the human labelling budget.
   - In the human lane, model-reviewer vs human κ ≥ 0.7 on a ≥ 10% sample. Below that, the human stays a gate for the
     family.
3. **Orthogonality** (R10), in four layers:
   - prompt;
   - start/end state;
   - check-set Jaccard;
   - behavioural (correlation of pass and cause vectors across the panel).

   Duplicates (same intent and same failure cause) are removed. The batch's top surface is ≤ 40% of tasks (*v1*).
4. **Transfer proxy (CPU):**
   - the base policy is measured on the probes;
   - then the same policy is given K solved family traces in context, or best-of-n with selection by the family verifier,
     and measured again;
   - the family is accepted when the proxy improves the probes and the improvement is not explained by the probes'
     surface.

   This is a proxy. A small training run replaces it on GPU later.

## 9. Train and monitor (when training runs)

- **Weighting:** each intent is weighted by measured failure × learnability.
- **Watched during training** (R14):
  - zero-variance group share;
  - always-pass / always-fail tasks (re-validate always-fail twice);
  - reward vs probe divergence (halt);
  - hack rate (alarm at 2%);
  - behaviour audit.
- **After each round:**
  - solved intents retire or harden along their axes;
  - new failure clusters become intent candidates (§4).

  This closes the intent loop.
- **Policy upgrade** (R16): replay the hack library, re-run floors, recompute difficulty, quarantine tasks whose failure
  cause shifts to a spec or verifier defect.

## 10. The outer loop

- **Decision store:** every keep, drop or revise decision by a human or a gate, with a reason. Reasons are clustered into
  **named criteria**; each criterion gets a guard in the machine. The recipe version increases.
- **Gate compilation:** a cheaper gate (human → agent on another family → classifier) replaces a costlier one when its
  agreement on ≥ 50 held-out decisions (*v1*) passes the criterion's bar. It is re-checked every N batches.
- **Role hill-climb:**
  - paired, alternating-order A/B of role prompts and operators on the same seeds, with a noise gate;
  - neighbour-family regression and two reviewers on different families;
  - metric: accepted tasks per dollar at a fixed hack-free rate;
  - no numeric target.

## 11. Swarm orchestration

- **The root curator writes no tasks.** It holds the coverage matrix (behaviours × surfaces), the backlog and leases. It
  runs as many machines in parallel as the budget allows, and merges only gate-accepted tasks.
- **Persistent specialists:**
  - a World expert per World;
  - a hacker per family;
  - a test writer per family (never ingests oracle material);
  - the intent librarian.

  They are reached through the forge repo and issue board; staging, not prompts, controls what each may see.
- **Funnel:** floors and the ledger come first, then probes, then solvers and audit, then the gold set. A task that fails
  an early, cheap stage never reaches an expensive one.

## 12. Models, budget and publishing (CPU v1)

| Role | Family |
|---|---|
| Author, oracle author, spec | `qwen/qwen3.7-flash` |
| Verifier writer | `qwen/qwen3.7-flash` (separate context) |
| Reviewer, second verifier, auditor, realism | `deepseek/deepseek-v4-flash` |
| Tie-break | `z-ai/glm-4.7-flash` |
| Solver rungs | `openai/gpt-oss-20b`, `meta-llama/llama-3.1-8b-instruct`, `qwen/qwen3.7-flash` |

- **Budget:** hard caps per run (the forge USD ledger, reserve then settle). The first end-to-end validation is capped at
  **$20**, and every report states the actual spend.
- **Publishing:** an accepted TaskCard is published as an agentenv `Task` plus one pinned `Eval` per split (study 04).
  Lineage goes in the bundle ledger.
