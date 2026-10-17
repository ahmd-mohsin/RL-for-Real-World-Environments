# M17 task factory, study 08: long-horizon coding tasks that teach (design for review)

This turns the factory into one concrete MVP: **20–30 long-horizon coding tasks plus a sealed held-out benchmark**, built
by the swarm, run in Crucible's sandbox with the M16 harness, trained with step-level credit. It builds on 03 (design),
05 (method) and 07 (first run). New inputs folded in:

- **"How to make a good RL task"** (already the basis of 03): a task is prompt + World + verifier + intent; three
  alignment axes; revision loops; the outer loop compiles human judgement into cheaper gates.
- **"Building RL environments for superintelligence"** (Preference Model, 2026-08):
  - don't lie to the model: everything in the prompt is checked by the grader, and nothing the grader can't check goes in
    the prompt;
  - grade outcomes, not process;
  - accept that the model knows it is in an environment, and trade realism for security;
  - make tasks deep (non-saturating: reward doing better, not clearing a fixed bar);
  - give the model an honest way to give up.
- **"Agent traces"** (2026-10):
  - environment evidence (diffs, test results, state) beats model-call logs;
  - semantic enrichment (models labelling traces) drives improvement loops;
  - failed traces become eval cases.

## 0. The structural finding: our current coding tasks are short

The executable lane (07) mines **single commits that change 1–2 files**. That is "implement feature X" at its smallest:
- it saturates fast;
- it cannot fail for long-horizon reasons;
- the M16 harness is never exercised by it.

The 210 validated tasks stay useful as **isolated milestones** for the concatenation test, but they are not the MVP.

**The fix: the unit of a long coding task is a window of real development.** The window runs from release A to release B
of a real repository, or across a run of commits that build on each other.

The window is the ground truth, observed, not invented:
- the repository at A is the start state;
- the repository at B is an executed oracle;
- the tests added across the window are the checks;
- each commit's own tests are a milestone, in a known real order.

This is the user's principle (rubrics reverse-engineered from ground-truth observations) applied to time: the repo's own
future grades the task.

**Measured on the six repos we mined** (release-to-release, non-merge commits that change both source and tests;
`scratchpad` scan, 2026-10-17):

| repo | releases | windows with 5–25 such commits | largest |
|---|---|---|---|
| more-itertools | 62 | 27 | 15, 15, 20, 22, 25 |
| boltons | 54 | 17 | 12, 13, 14, 14, 16 |
| humanize | 58 | 6 | 5, 5, 7, 7, 9 |
| python-semver | 35 | 5 | 5–8 |
| cachetools | 78 | 3 | 5, 5, 6 |
| parse | 12 | 1 | 6 |
| **total** | | **59** | |

**Caution: a long window is not a long-horizon task.** A release of a utility library is often N independent functions
(more-itertools adds unrelated recipes). Such a window is "long, not long-horizon" and fails the concatenation test
(05 §6.7). **Coupling inside the windows.** Re-counted on `*.py` diffs only, 47 windows have 5–25 commits. Of these, 31 have ≥ 3
commits that touch a function or class an earlier commit in the same window touched:

| repo | windows | coupled |
|---|---|---|
| more-itertools | 21 | 18 |
| boltons | 14 | 6 |
| cachetools | 3 | 3 |
| humanize | 4 | 2 |
| parse | 1 | 1 |
| python-semver | 4 | 1 |

This is a syntactic **upper bound**. Hunk headers name the enclosing or neighbouring symbol, so commits that only share a
file (for example the many recipes appended to `more.py`) count as coupled. The executed dependency test of §5 will be
lower.

Therefore:
- the window miner must filter by **executed dependency**, not by size;
- the repo pool must grow toward medium projects where work builds on itself (CLIs, parsers, ORMs, servers).

## 1. The task, defined for long-horizon coding

| Part | Long-horizon coding instance | Where it comes from |
|---|---|---|
| **Intent** | One named behaviour that fails over long work (§2), with traces as evidence | discovery on windows (05 §4); human approves |
| **World** | The repo at A in a Docker CPU sandbox: no network, git history truncated at A, a local package mirror frozen at A's date, the repo's own test runner available | window miner + World builder |
| **Prompt** | What a maintainer would ask for the release, in a user's voice, written from the window's commit messages, PRs and changelog. It names behaviour, never files or tests | author; realism reviewer |
| **Verifier** | Hidden fail-to-pass tests from the window (grouped per commit = milestones) + every pre-existing pass-to-pass test, run in a separate grading container on the exported end state | the repo's real tests; ledger |
| **Oracle** | The real repo at B, executed | the repo |

**Two task types:**

- **Type A, history replay (bar).**
  - The window above; reward is the outcome plus step credit (§6).
  - Highly verifiable and cheap to scale.
  - Saturates per task, so the difficulty dial (§7) keeps it in band, and solved tasks retire.
- **Type B, deep (no ceiling).**
  - Outcome-graded open optimisation on a real repo. For example:
    - make the parser N× faster on a hidden corpus;
    - shrink peak memory of a pipeline;
    - port a module to a new backend at measured speed (the kernel-migration family, on GPU later).
  - Every existing and hidden test must pass, or the reward is 0.
  - The score is improvement over the baseline on hidden inputs, log-scaled and uncapped, and it rewards doing better
    rather than clearing a bar (Preference Model).
  - Timing noise goes through the flake gate: repeated timings, and the score's CI must exclude noise.
  - The prompt states the metric exactly and states no method.

**MVP mix:** about 24 Type A + about 6 Type B training tasks.

## 2. Intents for long-horizon coding (proposed; each needs discovery evidence and your approval)

Each intent names a failure that only long work exposes. Each also has an **executed signal** that measures it, so the
intent is checked by outcome, not by reading the trace.

| id | Behaviour (one sentence) | Executed signal |
|---|---|---|
| I1 | Keeps earlier work working while extending it | milestones lost after being reached (Φ drops); pass-to-pass breaks |
| I2 | Builds on its own earlier decisions (later steps use the API it chose earlier) | in-context rate of milestone k below its isolated rate (concatenation test) |
| I3 | Finds its way in an unfamiliar codebase before editing | first-milestone rate vs isolated rate when the prompt names no files |
| I4 | Finishes: meets every requirement instead of stopping early | final Φ < 1 with budget left; early-stop share |
| I5 | Reports its own status truthfully | calibration of its per-requirement claims vs verified results (§7.2) |
| I6 | Keeps state across context compaction (M16) | milestone loss right after a compaction boundary vs elsewhere |

**Orthogonality** = the same intent on different surfaces (repos, repo classes, Type A/B, prompt styles), never the same
repo twice for one intent in the accepted set. The four-layer test of 05 §8 applies unchanged.

## 3. The alignment axes, made executable

Each axis gets an executed guard. A reviewer's opinion alone never gates (05 principle 2).

| Axis | Guard (gating) |
|---|---|
| **Prompt ↔ verifier** | **Ledger, two-way** (exists): every prompt sentence that asks for something maps to a check, and every check maps to a sentence. **Unstated-format sweep:** a check every solver fails that the auditor labels `unstated_format` (exact strings, names, error types not derivable from the prompt or the repo's conventions) is either stated in the prompt or demoted to non-gating. |
| **Don't lie** (new) | **Prompt lint:** a process instruction ("don't use X", "do it by hand", "don't read the history") is refused unless an outcome check enforces it. Otherwise the World makes it impossible (history truncated, no network) and the sentence is deleted. The prompt says plainly that this is a sandbox; that hidden tests check the stated behaviour; that existing tests must keep passing; the budget; and how to give up. |
| **Verifier ↔ intent** | Floors; **commit-revert mutants** (§5); shadow tests; hackers + hack replay; second verifier + differential; the gold set. |
| **Prompt ↔ intent** | Realism reviewer (another family) rejects hints a maintainer would not give, such as file paths or test names. Prompt style is matched to the target usage (an issue- or release-note style, often underspecified in direction but never in outcome). |
| **World sufficiency** | Auditor cause mix ≥ 70% skill gap; World bugs send the task back. **Shortcut probe** (exists): no passing path under half the oracle's steps. **Leak probes:** `pip download <pkg>==B` fails; `git log` shows nothing after A; no wheel or sdist of B exists in the mirror or cache. |

**Security over realism.** The model may know it is in an environment. The grading container never mounts the agent's
filesystem writable. The protected paths (conftest, sitecustomize, `*.pth`, test dirs, packaging config) are restored
from the oracle before grading, which `ExecTask.grade` already does.

## 4. The swarm: who does what, who sees what

The root curator writes no tasks (05 §11). It holds:
- the coverage matrix: intents × surfaces × type;
- leases, the USD ledger, and the stop rules.

It runs one machine per (intent, window) cell. **Staging, not prompts, controls what each role sees.**

| Role | Model (v1) | Sees | Never sees | Output |
|---|---|---|---|---|
| Window miner | none (deterministic) | git history | — | candidate windows, per-commit test groups |
| World builder (one per repo, persistent) | qwen3.7-flash | repo at A, CI config | B, the window's tests | Dockerfile, pinned mirror, test command; passes the 60-rerun flake gate |
| Dependency prober | none | executed results | — | the milestone DAG `requires` from executed runs (§5) |
| Spec + prompt author | qwen3.7-flash | commit messages, PR text, changelog, diff *summary* | the tests | requirements with ids, prompt in a user's voice |
| Realism reviewer | deepseek-v4-flash | prompt, repo at A | oracle, tests | hint findings → back to author |
| Ledger mapper | deterministic + deepseek | prompt, test names and bodies | oracle diff | requirement ↔ test map; unmapped either way → author |
| Verifier writer #2 | deepseek-v4-flash | prompt, spec, repo at A | oracle, real tests | independent shadow tests (§5) |
| Hacker (one per family, persistent) | deepseek-v4-flash | prompt, verifier, intent | oracle | reward-without-work attempts → hack library |
| Solvers | gpt-oss-20b, llama-3.1-8b, qwen3.7-flash (M16 harness) | prompt + World only | everything else | trajectories, checkpoint Φ series |
| Auditor | deepseek-v4-flash | traces + environment evidence | — | FailureCause per failing run, suspected FP/FN |
| Tie-break | glm-4.7-flash | the disputed item | — | decision |

- **Funnel (cheap first):** miner → dependency prober → World build + flake → ledger → floors + mutants → hacker →
  solvers + audit → gold set.
- A window that fails a cheap stage never reaches solvers, which are about 90% of the cost.

**Traces** (the tracing article, applied):
- Every solver run records model calls (M16 ledger) and **environment evidence**: the diff and the hidden-test status at
  each checkpoint, stored as artifacts keyed to the turn that produced them.
- **Semantic enrichment** (auditor labels) feeds intent discovery and task revision. It is **never reward** (grade
  outcomes, not process).
- **Failed traces become regression cases:**
  - only in the training pool;
  - eval traces contribute only intent-level counts (§8).

## 5. Building and validating the verifier for a window

1. **Fail-to-pass and pass-to-pass sets** (the SWE-smith pattern, exists): run the window's tests at A and at B, 30 reruns
   each at two concurrency levels.
   - fail-to-pass tests are candidate checks;
   - pass-to-pass tests are regression checks;
   - anything flaky is dropped;
   - a test that passed at A and fails at B is dropped (the window removed that behaviour).
2. **Milestones = per-commit test groups.**
   - Executed dependency: for commit k, apply only commit k's source change to A and run its tests.
   - If they fail without earlier window commits, milestone k `requires` those commits; the prober finds the minimal set
     by reverting earlier commits one at a time.
   - Windows with no dependency edges are long, not long-horizon, and are dropped from Type A (or kept as short tasks).
3. **Commit-revert mutants (free, real, per requirement).** For each milestone k, the oracle B with only commit k's
   source change reverted (where the revert applies cleanly) is a solution wrong on exactly requirement k.
   - The verifier must fail milestone k and pass the others it does not depend on.
   - This is 05 §6.2's mutant, taken from history instead of written by a model.
4. **Valid alternatives.** Solver end states that pass, that the second verifier accepts, and that the auditor confirms
   as a different implementation. If too few exist, one model rewrites the oracle in a different style (the tests decide).
5. **Shadow tests (overfit detector).** Verifier writer #2 writes property-style tests from the prompt alone, with new
   inputs.
   - A shadow test gates only after it passes on the oracle and on the valid alternatives and fails on the mutants.
   - Until then it only flags: "passes the hidden tests, fails shadow" means special-casing → audit.
6. **Floors:** no-op, start state, deleting the new tests' targets, `pass`-stubs of the new symbols. All must score 0
   (and Φ ≈ 0).
7. **Hackers + replay:** confirmed hacks become scripted replays for every task. Expected first hacks:
   - fetching B from the package index (blocked by no network plus the mirror);
   - reading future history (blocked by truncation);
   - editing tests or conftest (restored at grading);
   - special-casing the inputs (caught by shadow tests).
8. **Gold set per family** (05 §8): real solver end states + commit-revert mutants + valid alternatives + confirmed hacks.
   - Mutants make the "truly failing" class cheap: ≥36 per class gives a false-pass bound ≤ 10% with zero errors.
   - Coding is self-verifying, so labels are executed; a human spot-checks 10% of the alternatives.

## 6. Step-level credit assignment

The reward has two parts, both computed only from verified World state, never from the model's text.

**Outcome.** 1 when every gating check passes (Type A), or the uncapped improvement score (Type B), or 0. This is the
sequence advantage Â, group-relative.

**Progress, potential-based** (M6, exists in `credit/` and `train/credit.py`):

- **The potential.** Φ(s) = Σ_m w_m · pass_m(s) over milestones whose `requires` are all satisfied in s.
  - The R5 property blocks stubbing: a milestone met before its prerequisites adds nothing.
  - w_m comes from the **predictive-validity gate** (`credit/validity.py`): a milestone earns weight only if its
    checkpoint value predicts the final outcome on held-out, later traces (time split, AUC, grouped bootstrap). A
    milestone that doesn't predict success earns nothing, and that is what keeps the shaping honest.
- **Checkpoints.** Φ is measured:
  - at each agent `git commit`;
  - every K tool calls (v1 K = 10);
  - at each M16 compaction boundary;
  - at the end;
  - capped at about 20 evaluations per episode.

  Each measurement snapshots the workspace into the grading sidecar. **The agent never sees hidden-test results.** Its
  in-loop feedback is only what it can run itself: the repo's existing tests, which are public checks (03 §6).
- **Per-turn credit.** ΔΦ between consecutive checkpoints is assigned to the turns in that span as `Turn.advantage`.
  `turn_credit_map` then:
  - centres it within each episode;
  - bounds it by the outcome advantage;
  - gives none to gate-failed or hacked episodes;
  - gives none to a group without outcome spread.

  M16 RLM children take the credit of the root turn that spawned them. Compaction calls share the episode advantage.
- **Lost milestones are negative ΔΦ.** Breaking what worked is penalised at the step where it happened. This is the I1
  and I6 signal, and the main thing a short task can never teach.
- **Why it doesn't distort the goal.** Potential-based shaping with γ = 1 and Φ(terminal) folded into the outcome leaves
  the optimal policy unchanged (Ng et al., 1999). M6's properties R1–R7 are run per family as part of acceptance.

**What step credit must not become:** reward for process (tool choice, reading files, writing tests). Those stay audit
labels.

## 7. Calibration, in both senses

**7.1 Difficulty calibration (the task is in the learnable band).**

- **The dial is window length**, a real axis: longer dependency chains. Contrived axes, such as extra requirements or
  hidden rules, are refused by the realism reviewer.
- Grow or cut the window until the **target policy's** final Φ has spread in its group and its full pass rate is in
  [0.1, 0.8].
- If even one milestone is never reached by any rung, split the window. If a rung passes every attempt, extend it or
  retire it.
- Cause mix ≥ 70% skill gap.
- Type B needs no band: its score has no ceiling. It needs only a non-zero baseline improvement from at least one rung,
  so the gradient exists.

**7.2 Self-calibration (the model reports truthfully; intent I5).**

- **Claims.** At the end the agent writes a status file: for each requirement in the prompt, its probability that the
  requirement is met.
- **Scoring.** Claims are scored against the verified per-requirement result with the **Brier score**, a proper scoring
  rule, so stating its true belief is the best policy. The term has small weight and is computed only for requirements
  the verifier checks: nothing unverifiable is ever rewarded.
- **Giving up.** `GIVE_UP: <reason>` ends the episode honestly. The outcome is scored as it stands, and a correct "not
  done" claim is not penalised. False claims of success cost through the Brier term.
- The prompt says all of this (don't lie: the model knows exactly what is scored).

## 8. The held-out evaluation benchmark

It is built by the same factory, then **sealed**.

| Split | What | Size (v1) | Tests |
|---|---|---|---|
| H1 in-family | Type A windows from repos never used in training | 10 | does training transfer to new codebases |
| H2 transfer surface | the same intents on another repo class (larger apps, other test runners) and prompt styles | 8 | does it learn the behaviour, not the surface |
| H3 deep | Type B tasks on unseen repos | 4 | no ceiling, so it can't saturate |
| Time split | the windows of H1 and H2 are all dated after every training window, and after the base model's cutoff where possible | — | contamination |

**Sealing:**
- separate repos and leases;
- no factory role that writes training tasks is ever given eval repos;
- the eval verifier code is never in a training World;
- `bench/contamination.py` checks the training corpus against eval prompts and diffs;
- published as its own `Eval` split, which `factory/publish.py` refuses to mix with training.

**Metrics, per intent and overall:**
- pass@1 and pass^k (consistency);
- mean final Φ;
- milestones lost per episode (I1);
- compaction-boundary loss (I6);
- calibration error of claims (I5);
- hack rate;
- tokens and dollars per solved task;
- IRT item difficulty and discrimination (`bench/irt.py`), to spot dead items.

**Statistical honesty.**
- The comparison is paired by task: base vs trained, same tasks, k attempts each.
- Assume a per-task paired difference with standard deviation about 0.25 (to be measured on the first run). The minimum
  detectable change at 80% power is then about 2.8 · 0.25 / √n:
  - about 13 points at n = 30;
  - about 16 points at n = 22.
- Mean Φ, being continuous, will usually do better.
- So the v1 benchmark can show large effects only, and reports will say so with CIs.

**The trace loop, without leaks.** Eval failures are labelled by the auditor. Only intent-level counts ("I1 failures
rose") return to the root curator, which opens new training windows elsewhere. No eval task, trace or prompt ever enters
training.

## 9. The MVP plan: 30 training tasks + 22 held-out

1. **Fix option-B leftovers** (spec author `max_tokens`; trial adapter bug).
2. **Window lane** (`factory/lanes/windows.py`):
   - miner;
   - history truncation;
   - frozen package mirror;
   - fail-to-pass / pass-to-pass per commit;
   - dependency prober;
   - commit-revert mutants;
   - milestone DAG into the existing `Milestone` schema.
3. **Repo pool:**
   - keep the 6 small repos for the pipeline bring-up;
   - add about 12 medium Python repos with real releases and coupled work, so the dependency filter leaves ≥ 80
     windows;
   - reserve about 6 repos for H1 and H2 before any mining.
4. **Checkpoint-Φ sidecar** in the World:
   - snapshot → grading container → Φ;
   - wired to `Turn.advantage`;
   - a status-file reader and Brier term;
   - the `GIVE_UP` action.
5. **Run the funnel** on CPU with the cheap models: discovery on 15–20 windows → intent cards for your approval → machines
   on about 80 windows → accept 30 (≥ 3 surfaces per intent, top surface ≤ 40%) → build H1–H3 in sealed leases.
6. **Family acceptance:** gold set (≥ 36 per class), orthogonality, R1–R7, and the transfer proxy on H2.
7. **Then GPU** (your approval): a small GRPO run on the 30, base vs trained on H1–H3.

**Stop rules** (no numeric targets): stop mining a repo when its marginal yield over the last 10 windows is 0, or its
surface exceeds 40% of accepted tasks.

**Cost estimate (to confirm on the first 10 windows):**
- Long episodes are about 100–300k tokens each.
- Solvers: 3 rungs × 4 attempts × about 80 windows ≈ 100M tokens. At the cheap models' prices that is tens of dollars,
  above the current $20 cap.
- I'll measure the first 10 windows within $20 and ask before going further.

## 10. Decisions that need you

1. **Approve the intents** I1–I6, or name others. Discovery on windows will attach evidence before any card is final.
2. **Repo pool:** I propose about 12 medium Python repos; you may have preferred codebases, or enterprise-like ones.
3. **Budget:** $20 covers bring-up and the first 10 windows; the full 80-window funnel needs more.
4. **Held-out ownership:** who, if anyone besides the sealed pipeline, may look at H1–H3 tasks.
