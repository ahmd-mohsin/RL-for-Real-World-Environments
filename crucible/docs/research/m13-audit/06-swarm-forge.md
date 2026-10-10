# M13 audit 06: agent swarms for building harder RL tasks, and the Crucible "Swarm Forge" (2026-10-10)

**Scope.** Can parallel agent swarms produce more complex and more rigorous RL tasks and Worlds, and how should Crucible's forge (M8) adopt them? This audit reads the swarm sources, the task-construction sources, and Crucible's forge (M8), verify (M7) and accept (M9) code.

**Path prefixes:**
- `ART/` = `RL-Environments/07-articles/`
- `GS:` = `ART/gitswarm-2610.04862.txt` line numbers
- `RE/` = `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-research-environments/`
- `SS/` = `…/swe-bench-swe-smith/`
- `R2E/` = `…/huggingface-repo2rlenv/`
- `K/` = `…/preferencemodel-karotte/`
- Crucible paths are relative to the repo root.

**Builds on** `docs/research/m7-m8/02-forge-synthesis.md` §8 (GitSwarm as PATTERN-only). New here: composition operators, an adaptive red-team role, solver ladders, and a gap matrix against the code as built.

## 0. Findings that change requirements (act now)

1. **No source shows that swarms make tasks "exponentially" more complex.**
   - GitSwarm compounds *inference on one task*: 94.7% of ProgramBench contributions are built upon (GS:495-539). It never generates tasks or training data. RL appears only as future work (GS:780-784).
   - The Prime Intellect essay argues for meshes over trees and reports no task-generation mechanism.
   - Locally, `recursive_task_synthesis` and `terminal_lego` are dataset *loaders* with no synthesis code (§2.1).
   - Composition grows the *number of candidates* combinatorially, as in SWE-smith's `get_combos`. It does not raise difficulty by itself. Requirement: **every composite must prove it is harder than its parts, and that every part is necessary** (new checks A14/A15, §4.4). Otherwise composition is padding.
2. **Copying GitSwarm directly would break Crucible's core guarantee.**
   - In GitSwarm, every worker sees the whole repository, the reference included (m7-m8/02 §8.1, GS:3667).
   - Crucible's rigor rests on asymmetric information. The Test writer must not see the oracle: staging in `forge/orchestrator.py:Orchestrator._stage`, canaries in `_breach`.
   - Requirement: a swarm memory must be **partitioned into visibility zones**, with canary scans on every commit (§4.1).
3. **The current forge is a swarm in name only.** `Orchestrator.run_plan` awaits nodes one at a time (`orchestrator.py:216-224`), and one plan runs at a time.
   - There is no concurrency limit C.
   - The shared task repository and persistent specialists in DESIGN §7.2 "Coordination (P28)" (DESIGN.md:522) are not built.
   - G-compose, G-transplant and G-tighten are listed as not done (`docs/modules/M08-forge.md:58`).
4. **Red-teaming is static.** A13 runs an *authored* attack list (`accept/checks/extended.py:a13_attacks`, l.233), and the Adversary is scripted (`packs/_toy/.../forge/roles.py:ScriptedAdversary`). No agent *searches* for reward hacks on a new task, and the sources show this is the failure that matters:
     - Karotte: "a 1-in-10 hack becomes the policy" (`ART/pm-karotte-technical-deep-dive.md:15-19`);
     - Repo2RLEnv: "every green score I got before was contamination" (`ART/hf-repo2rlenv-how-not-to-build-coding-environments.md:46-56`).
5. **A minor inconsistency.** DESIGN §7.4 G-tighten allows raising a bar when the oracle keeps "≥10% headroom" (DESIGN.md:550). The A1 check in M9 requires ≥15% (`docs/modules/M09-accept.md:27`). G-tighten should use the A1 value.

---

## 1. Swarm mechanisms

### 1.1 Prime Intellect, "On the nature of the swarm" (`ART/pi-on-the-nature-of-the-swarm.md`; the local copy is complete, footnotes included)

| Mechanism | Claim | Lines |
|---|---|---|
| Compaction | Raises **total context capacity**, but "every compaction is a bet": the summary must guess what will matter later | 15-23 |
| Sub-agents | Compaction where the summary is written *after* the question is known. They are parallel, and they "forget everything" | 35-49 |
| Persistent sub-agents | Keep the context and ask again later. Total capacity grows to roughly one window per agent ("parallel over context") | 51-63, fn13 l.119 |
| No free lunch | The price is communication: lossy and costly in tokens. "If the hard part of a task needs everything in one place at once, splitting it up doesn't help" | 65-73 |
| Trees to meshes | A tree routes everything through the root, whose context fills up. A mesh lets peers ask the specialist directly. Roles emerge from early random differences | 75-91 |
| Structure | "Seed the swarm with human structure" (shared git repo, issue tracker, Slack), then let agents change it | 95-101 |
| Scale | Only one figure: OpenAI's claim that 10,000 agents solved Navier-Stokes, "which mathematicians are still checking". Also "1M working context compresses quite nicely into 16k" | fn16 l.122, fn3 l.109 |

The essay reports no failure numbers, no verification method and no task generation. It contributes design principles only: persistence, a mesh, and shared state as the single source of truth.

### 1.2 GitSwarm (arXiv 2610.04862)

**There is no harness code.** `veds12-git-swarm` is the project website (`index.html`, `source/build.py`, `app.js`, figure JSON). The episode in `app.js:600-654` uses `git switch -c` and a different commit-message format from the paper, so it is illustrative only. The repo has no licence, so GitSwarm can be used as **PATTERN only**.

**Architecture** (§2, GS:150-224):

| Element | Mechanism | Lines |
|---|---|---|
| Memory | The task spec is the root commit. Each contribution is one immutable commit with one Git parent plus a mandatory `informed_by=` set ("semantic inheritance", cross-branch, no merges). Format: `[<type>\|informed_by=<sha>,…] <summary>` | GS:159-179, 3601-3613 |
| Workers | Identical model and prompt. No planner, no roles. Each runs in an isolated worktree and forks exactly once (`gs_fork`). Peer worktrees are read-only | GS:182-185, 3462-3474 |
| Scheduling | At most C workers run concurrently (C = 5), with replacement on exit, until B charged episodes are spent. Budgets: ProgramBench B = 100–500; proofs B = 20–160 | GS:182-184, 285-289 |
| Live state | `gs_dashboard` (budget, workers in flight, nomination tally "not correctness evidence"). `gs_inflight announce "Gap \| Deliverable \| Success \| Abort"` before editing; `inspect` shows a peer's uncommitted diff. "Neither view assigns work" | GS:198-210, 3377-3382, 3445 |
| Episode end | Exactly one of **Contribute**, **Nominate**, **Abstain**. A voluntary abstention does not consume B | GS:186-196 |
| Validation | "Publication requirements, not solution correctness". For example, every ProgramBench node must have a `compile.sh` that produces `./executable`, plus `MY_THOUGHTS.md` and `EVALUATION.md` | GS:188-189, 3481-3498, 3650-3657 |
| Readout | Most nominations wins. Ties go to the lexicographic full SHA; with no valid nomination, the root is used. Nominations are stored outside the graph. Research tasks are scored on validation instead | GS:212-224, 3336-3338 |

**How work compounds:**
- 99.9% (ProgramBench) and 96.2% (IMO) of eligible episodes cite earlier work.
- Built-upon: 94.7% (ProgramBench) vs 44.4% (IMO).
- The winner's ancestry covers 81.6–92.6% of the graph on ProgramBench but only 21.3–33.0% on IMO. Git parentage alone covers about 25% (GS:495-544, 2262-2265).

**Roles are post-hoc labels** (GS:2563-2601), except NanoChat, which assigned roles at 3:1 (GS:2198-2201).

**Rigor:**
- Final scores are external: hidden tests, an LLM judge for IMO, a protected dev set for research tasks.
- Agents do falsification and replication themselves (GS:2047-2069).
- Anti-shortcut rules are stated in the *prompt*: do not read or decompile `executable_orig`, no internet (GS:3656, 3664-3667).
- 3.1% of charged episodes failed validation (GS:2677-2680).
- Negative results are committed as candidates: 43–58% of commits, with 30–76% cited later (GS:1827-1829, 1994-1995, 2123-2129).

**Ablation** (GS:3236-3251, three seeds, no significance claimed): central selector +2.34 pp; central aggregator −3.87 pp and 16% slower; synchronous waves +1.05 pp and 30% slower.

**Failure modes:**
- About 1 in 6 ProgramBench workers abstain as redundant. In 97.1% of collisions the conflicting commit landed after the worker started, so its compute was already spent (GS:719-721, 2885-2905).
- Votes split across repairs and are correlated (GS:2815-2817, 3404-3406).
- Herding: the share of commits building on the adapter rose from 17% to about 90% (GS:2073-2076).
- Proxy evaluations misranked models (GS:2061-2066); input-token cost grows with repo size (GS:765-770); dollar cost is not reported.

---

## 2. Task-construction sources

### 2.1 `recursive_task_synthesis`, `terminal_lego` (Prime Intellect research-environments)

There are **no synthesis, mutation or composition operators locally.**

**`recursive_task_synthesis`:**
- It loads the dataset `PrimeIntellect/Recursive-Task-Synthesis` @`bd6c1e8f` (`RE/environments/terminal/recursive_task_synthesis/recursive_task_synthesis/source.py:10-11`) and runs each task's Harbor verifier unchanged.
- Release audit: 36,249 tasks and 47 repairs, each re-checked as "gold 1 and no-op 0". It excludes 131 failed builds and 1,104 VM validation failures. "Other verifier failures are not filtered, and this is not a fully gold-verified dataset" (`README.md:60-68`). The hosting audit "does not establish gold-solution correctness" (`README.md:12`).

**`terminal_lego`:**
- 13,825 tasks (`README.md:10`), each with `solution/solve.sh`, which is never run (`taskset.py:4-5`).
- Network is unrestricted (`README.md:16`).
- The solver prompt *asks* the agent not to read the history or answer keys (`taskset.py:49-56`).

The closest local mechanisms:
- **`general_agent` `gold_check`** (`RE/environments/tool_use/general_agent/general_agent/corpus.py:78-105`). Requirements: the gold chain replays; it changes the DB hash; `verify(initial) == 0`; `verify(gold) == 1`. It is "shared by the solver's `validate` hook and the synth's reward": **the proposer is rewarded only for tasks that are solvable and non-trivial.** Families are tiered `_t0.._t4`. A pass-rate band (`min_pass_rate`/`max_pass_rate` on `gpt-5-mini`) defaults to the no-op [0, 1] (`taskset.py:104-112`).
- **Procedural presets.** `prolog` has `DIFFICULTY_PARAMS` and solution uniqueness (sudoku `_count_solutions(limit=2) == 1`). `uuid_ctf` has `DIFFICULTY_PRESETS`. The i3_* solve-rate filters default to [0, 1].

### 2.2 SWE-smith (MIT; PORT)

**Bug generation** has four operator families:
- LM modify (`SS/swesmith/bug_gen/llm/modify.py:gen_bug_from_code_lm`);
- LM rewrite, which stubs the function and re-implements it (`rewrite.py`);
- 17 procedural LibCST operators, `CommonPMs` (`procedural/base.py:58-142`; `DEFAULT_PM_LIKELIHOOD=0.2`);
- **combine**: `combine/same_file.py` and `same_module.py`, `num_patches=2`, `max_combos=100`, and a used patch leaves the pool.

There is also PR mirroring (`mirror/generate.py`; ≤8 files, ≤500 lines).

**Validity:** `harness/gather.py:process_instance` keeps a task only with no timeout, F2P > 0 and P2P > 0 (l.333-336). `repair.py` reruns gold and strips tests that fail under it.

**Gaps:**
- Difficulty is an LM time-estimate rater, not solver pass rates.
- No flake reruns, no determinism or contamination scan.
- History is not scrubbed at checkout (`profiles/base.py:493-521`).

### 2.3 Repo2RLEnv (Apache-2.0; PORT)

**Pipeline:** find candidates → synthesize → gold 1 / no-op 0 → Harbor task. Its `swe_smith` recipe has no LM rewrite and no combine (`R2E/docs/pipelines/repo_mutate.md:15-20`).

**Rigor:**
- History scrub and egress denylist (`pipelines/_env_guard.py:64-91`).
- CVE prompt scrubber (`cve_patches.py:_strip_fix_leaks`).
- **The quality loop** (`docs/concepts/quality.mdx`):
  1. a reviewer that cites quotes;
  2. two probes: a wrong solution that must fail and a valid alternative that must pass;
  3. a **blind rollout classified** as legitimate success/failure, reward hack, task defect or infra;
  4. ≤3 repairs.

  `verified` is bound to the bundle hash. "Solve rate is not the quality criterion" (`quality.mdx:18`).

**Difficulty** is LOC buckets (`pr_runtime.py:_difficulty_bucket`).

**Cost** (`docs/pipelines/economics.md`): $0.159/task for the six repository recipes, $12.99 per accepted Tasksmith task, $0.684–1.00 for FrontierSmith and CodeMidas.

### 2.4 Karotte / Preference Model (PATTERN)

**Construction:** no operators. A suite is the cartesian product of "axes of variation" (`suite.py:register_suite`), and `TaskCalibration` is a placeholder.

**Rigor is custody:**
- `collect_submission` kills the student's processes before collecting;
- `save_submission` has caps;
- `sanitize_paths`, a canary firewall, cgroups;
- a list of hand-found hacks: FIFO, `/dev/zero`, TOCTOU symlink, `sitecustomize.py` (`ART/pm-karotte-technical-deep-dive.md:50-111`);
- hack regression by fake-model replay (`:214-235`).

Crucible already has its principles as P20–P23 and P29.

**Missing in every source:** a measured pass-rate window, flake reruns, and determinism checks. Crucible already has all three (`forge/learnability.py`, `accept/checks/core.py:a4_flake`).

---

## 3. Gap matrix: swarm and construction mechanisms vs Crucible

✅ exists · ◐ partial · ✗ missing

| Mechanism (source) | Crucible | Where / what is missing |
|---|---|---|
| Typed role hand-offs, asymmetric info enforced by staging | ✅ | `forge/orchestrator.py:Orchestrator._stage`, `Node.visible`; `forge/schema.py` (CheckSet has no oracle field) |
| Canary breach detection | ✅ | `Orchestrator._breach`, `Node.canary_out`. Covers node outputs only; trajectories and commits are not scanned |
| Journal and resume, feedback retries, reserve/settle budget (RTS, R2E) | ✅ | `orchestrator.py:Journal`, `_run_node`; `forge/budget.py:Ledger`; `runtime/budget.py` |
| **Concurrent workers, C slots, B episodes** (GitSwarm) | ✗ | `run_plan` runs nodes sequentially (l.216-224); one plan per call |
| **Shared repo memory, `informed_by` provenance** (GitSwarm, PI) | ✗ | designed in DESIGN §7.2 P28 and m7-m8/02 §12.2 `repo.py`; not built |
| Contribute / Nominate / Abstain; announce-time dedup | ✗ / ◐ | `forge/dedup.py:dedup` (L0–L3) runs on finished candidates, not at announce |
| Persistent specialists per family (PI, P28) | ✗ | the defect registry `accept/registry/` exists as memory; no persistent role |
| Selection among candidate variants (GitSwarm central selector) | ✗ | |
| Oracle must pass, replay identity, gold check | ✅ | `accept/checks/core.py:a1_oracle`; `accept/contract.py:a1_oracle` |
| No-op and naive floors = 0 (R2E, RTS, general_agent `gold_check`) | ✅ | `core.py:a2_floors` |
| Defect mutants and equivalence (SWE-smith procedural) | ◐ | `accept/mutation/actions.py:generate` covers action oracles only (drop_step, drop_message, numeric_shift, enum_swap, reorder). Patch-oracle mutants for coding need a pack operator (`a3_mutation` reports "no mutant source") |
| Valid alternatives (R2E probes); determinism / flake (no source has it) | ✅ | `a3_mutation`; `core.py:a4_flake` (Clopper-Pearson, `verify/calibrate.py:flake_rate`) |
| Leak scans: strings, canaries, git history, image (R2E, SWE-smith branch leak) | ✅ | `core.py:a5_leak`; `verify/leak_audit.py:audit_tree/audit_image/audit_git`; `world/leaks.py:leak_tokens` |
| Instruction ↔ check coverage (PM); reference-differential grading | ✅ | `a7_coverage`; `verify/runtime/patterns/differential.py` |
| Static attack suite with controls (Karotte list) | ✅ | `extended.py:a13_attacks`; `tests/redteam/` |
| **Adaptive hack agent** (R2E blind-rollout classification) | ✗ | |
| Learnability band 4p(1−p), Wilson, 4→8→16 | ✅ | `forge/learnability.py:measure/advise/BANDS` (train [0.1, 0.7]; bench hi ≤ 0.2) |
| Frontier calibration ≥ 2 runs, failure taxonomy | ✅ | `extended.py:a10_calibration`; runner `runtime/run.py` |
| **Solver ladder** (several rungs, monotonicity) | ✗ | a10 pools models into one rate (`model=",".join(...)`) |
| Generators: span, trap, hindsight | ✅ | `packs/coding/.../forge/derive.py`; toy `hindsight.py` |
| **G-compose** (SWE-smith combine) / G-tighten / G-transplant | ✗ | `M08-forge.md:58` |
| Cross-World fusion (multi-App, multi-actor) | ◐ | `core/schema/world.py:WorldSpec` already validates multi-app, multi-actor worlds (`_references`); no fusion operator |
| Long-horizon stitching | ◐ | decomposition (H4 → H1–H3) is designed (DESIGN §7.3) and spans are built; stitching in the other direction is missing. World clock and events exist (`world/clock.py`) |
| Sandboxed builder roles (Harbor installed agents) | ✗ | `role_runner.py` not built (`M08-forge.md:60`) |
| Curator: splits, ladder, contamination, IRT | ✗ | DESIGN §7.2, §11.3 |
| Cost per accepted task, yield stop rules | ◐ | the ledger records spend; there is no yield metric and no stop rule in the forge (the runner has stop rules, `runtime/run.py` docstring) |

---

## 4. Design: Swarm Forge

**Principle: the swarm proposes, the contract disposes.**
- Open-ended stages run as a GitSwarm-style swarm: proposing, building variants, solving, hacking.
- Acceptance stays the deterministic DAG plus M9.
- No LLM decides acceptance (engineering rule 3, `contract.py:6`).

### 4.1 Memory: a zoned forge repo per family

There is one git repo per family under the store, with commits as in GitSwarm (`[type|informed_by=…]`). The repo is split into **visibility zones**, each mounted read-only into a worker's sandbox by the orchestrator.

| Zone | Contents | Readable by |
|---|---|---|
| `public/` | seeds, task specs, instructions, World specs (agent-visible material), negative findings, audit issues | all roles |
| `oracle/` | oracle actions or patches, expert end states | Builder, Adversary, Hacker-whitebox, Judge |
| `checks/` | verifier plans, hidden strings | Verifier-writer, Adversary, Judge |
| `solver/` | solver and hacker trajectories with graded outcomes | Calibrator, Curator, Verifier-writer (failure taxonomy only) |

A commit fails as a breach if its diff contains a canary from a zone the author cannot read (reusing `Orchestrator._breach`), or if `informed_by` cites a commit there. This closes GitSwarm's "everyone sees everything" hole with Crucible's staging rule.

### 4.2 Roles

Workers within a role are homogeneous, GitSwarm-style; roles differ because their zones differ. Each worker episode ends in **Contribute** (one validated commit), **Nominate** (Judge stage only) or **Abstain** (free in B, tokens still charged). Before editing, a worker announces `Gap | Deliverable | Success | Abort`.

| Role | Does | Output (existing schema) | Never sees |
|---|---|---|---|
| **Proposer** (Architect + Miner) | proposes base tasks and **composition plans** (§4.3) from seeds and signals | `TaskSpec`, `ComposePlan` (new) | `oracle/`, `checks/` |
| **Builder** | builds the World or start state and the oracle; for composites, derives the composite oracle | `EnvBuild`, `OracleBuild` | `checks/` |
| **Verifier-writer** | writes checks from stated properties and signals; reads solver failures to add separating checks (C2-style) | `CheckSet` | `oracle/` |
| **Adversarial solver** | honest solving at a ladder rung (§4.5); its outcomes calibrate difficulty | runner records | `oracle/`, `checks/` |
| **Red-team hacker** | tries to get reward *without* satisfying the stated properties (§4.6); the black-box variant sees what the agent sees, the white-box variant also sees the grader | `NegativeSpec(kind="cheat")`, registry candidates | black-box: everything hidden |
| **Judge** | ranks candidate *variants* of the same seed for diversity and quality (central selector, GitSwarm +2.34 pp); never accepts | ranking | — |
| **Curator** | dedup, splits, ladder placement, contamination | `CurationDecision` | — |

### 4.3 Composition and intermix operators (`forge/compose/`)

Composition is **closed**: a composite is a task (TaskSpec + WorldSpec + OracleBuild + CheckSet + NegativeSet), so composites compose again. Depth is a knob: `compose_depth ≤ 3` by default, raised only while the ladder says the result is still in band. Every operator must say how it builds the oracle, the checks and the negatives, and must pass A14 (necessity) and A15 (complexity gain).

| Operator | Definition | Composite oracle / checks | Necessity test (A14) |
|---|---|---|---|
| **Chain** `T1 ; T2` | T2's start is T1's end; T2's checks read state T1 produces | oracle = o1 ++ o2; checks = C1 ∪ C2 ∪ the dependency checks | `o2` alone on T1's start fails; `o1` alone fails C2 |
| **Fan-in** `(T1 ‖ … ‖ Tk) → J` | k independent subtasks plus an integration task J that needs all their outputs | oracle = any interleaving of o1..ok, then oJ; checks include J's integration gate | dropping any `oi` fails J's gate (k ablations) |
| **Nest** `T1[T2]` | T2 is a hidden prerequisite inside T1: the instruction states T1's outcome, and the World makes T2 necessary (a data inconsistency, a policy precondition) | oracle o2 ++ o1; T2's checks are stated as outcome properties of T1 (A7 holds) | the oracle without o2 fails; the start state with T2 pre-resolved makes T1 easier (ladder Δ ≥ 0) |
| **Fuse** `W1 ⊕ W2` | union of WorldSpecs: apps, actors and policies namespaced, `WorldSpec._references` re-validated; the task needs both worlds (e.g. a helpdesk refund that requires an ERP credit note) | oracle crosses Apps; at least one check reads state from both Apps | each source task's oracle still passes in the fused World where applicable (non-interference); at least one check fails if either App's part is skipped |
| **Perturb** | changes the *task*, not the solution: distractors, decoys, policy amendments, clock events, an injected bug (SWE-smith operators on the start state) | the oracle is re-derived by the Builder and must pass A1 again; perturbations that change semantics get a new property sentence | the unperturbed oracle fails if the perturbation is semantic; passes if it is cosmetic (labelled) |
| **Stitch** | joins G-span tasks from one trace (or Worlds over clock days) into an H4/H5 episode | oracle = the concatenated spans; milestones feed Φ only | each span's gate stays live in the stitched task |
| **Tighten** (existing design) | adds a perf, concurrency or sanitizer bar | the oracle must keep A1 headroom ≥ 15% | the pre-tighten frontier solution fails the new bar |

Combinatorics are bounded the way SWE-smith does it (`max_combos`, a used component leaves the pool). Proposers sample plans under a diversity quota (§4.7).

### 4.4 New acceptance checks

- **A14 necessity.** For each component, grade the composite oracle with that component's actions or patch removed. Every ablation must score 0 at a gate the component owns.
  - It reuses `accept/mutation/actions.py`: `drop_step` generalizes to `drop_component`.
  - It catches compositions where only the last part is really graded.
- **A15 complexity gain.** On the target ladder rung, `p_composite ≤ min_i p_i − δ` (δ = 0.1), with Wilson intervals from `learnability.estimate`. Otherwise the composite is rejected as "decomposable", meaning no interaction. Compositions at bench frontier p ≈ 0 must instead show a gain in failure taxonomy (new failing gates).
- **A16 adaptive red-team** (§4.6): the hacker budget is spent and no attempt scores > 0 without satisfying the stated properties.
- **A17 non-interference** (Fuse, Perturb-cosmetic): component oracles keep their verdicts.

These register in `accept/checks/` through `@register(...)`, with cost classes, so `accept/plan.py` runs them cheap-first (static → cpu → docker → gpu → paid).

### 4.5 Difficulty control: the solver ladder

The ladder's rungs are policies run through `runtime/run.py:Runner`, under the same harness as training:
- **L0:** no-op and naive scripted (= A2);
- **L1:** cheapest model;
- **L2:** the policy being trained (its current checkpoint);
- **L3:** frontier single attempt;
- **L4:** frontier pass@k (k = 8);
- **L5:** frontier swarm (GitSwarm-style, B ≈ 20). L5 is optional and only for a "beyond frontier" bench tier.

How the ladder is used:
- Each rung's pass rate uses `learnability.measure` (4 → 8 → 16 sequential sampling, Wilson).
- pass@k uses the unbiased estimator 1 − C(n−c, k)/C(n, k).
- **Train acceptance:** L2 p ∈ [0.1, 0.7] (existing `BANDS["train"]`).
- **Bench acceptance:** L3 Wilson upper bound ≤ 0.2 (existing). The record frontier is tracked for score-gated tasks (DESIGN §7.5).
- **Monotonicity:** the rungs L1 ≤ L3 ≤ L4 must not invert beyond their intervals. An inversion means a flaky grader or a check that grades style. It blocks acceptance and files an audit issue.
- **Steering:** `learnability.advise` gains two knobs, composition depth and operator (harder: compose one more; easier: drop the outermost component). Instruction obscurity is never a knob (DESIGN §7.4).
- **Retirement:** when L2 p rises above 0.7 in training, the task goes back to the Proposer as a seed for one more composition. Difficulty compounds with the policy and stays measured.

### 4.6 Rigor guarantees

Every guarantee is a check with evidence, bound to the task hash.

| Guarantee | Implementation |
|---|---|
| Oracle passes, replays identically, changes state | A1 (`core.py:a1_oracle`), plus composite oracles |
| No-op and naive fail | A2 |
| Hack agent fails | **A16.** The black-box hacker (agent view only) and the white-box hacker (grader source too) each get N episodes and a $ cap. Each rollout is classified as legitimate success, legitimate failure, reward hack, task defect or infra (PATTERN R2E quality loop). A confirmed hack is turned into a replayable `Attack` (`extended.py:Attack`) and a registry defect (`accept/registry/defects/*.yaml`). The task returns to the Verifier-writer, and A13 replays the hack forever (Karotte fake-model replay pattern) |
| Reference-differential | `verify/runtime/patterns/differential.py` for coding; state-predicate diff against the oracle end state for Worlds |
| Determinism / flake reruns | A4 (`a4_flake`); masked infra never counts (`runtime/run.py`) |
| Leak scans | A5, `leak_audit.audit_git/audit_image`, zone canaries (§4.1); Repo2RLEnv's three cheats as standing hacker probes |
| Valid alternatives pass | A3; honest solver successes that differ from the oracle become valid alternatives automatically |
| Composition is real | A14, A15, A17 |
| Dedup / contamination | `forge/dedup.py` at **announce time** (GitSwarm's 1-in-6 waste) and at acceptance; bench split sealed (DESIGN §11.2) |

### 4.7 Budget and cost control

- **Two ledgers:** the forge `Ledger` (role LLM spend) and the runner's `runtime/budget.py` (solver and hacker runs), with scopes `forge/<family>/<role>` and per-role caps.
- **Cheap-first gating.**
  1. A5/A7/A14 static checks run before any rollout.
  2. A1/A2/A3 run on CPU before the ladder.
  3. The ladder runs L1 before L3, and stops as soon as the band is decided.
  4. The hacker runs only on tasks that passed A1–A5.
- **Swarm knobs:** C (concurrency, default 5 as in GitSwarm), B (episodes per stage). Abstention is free in B. Announce-time dedup cancels a worker whose `Gap` matches an in-flight one.
- **Yield metrics** (Repo2RLEnv economics as the reference): $ per accepted task per operator, yield per stage, hack-found rate.
- **Stop rule:** if a generator's yield is < 10% after 20 attempts, or $ per accepted task exceeds a family cap, the generator is paused and an issue is filed.
- **Diversity quota:** no operator or seed above 30% of accepted tasks in a family (GitSwarm herding: 17% → 90%, GS:2073-2076).

### 4.8 From swarm output to Crucible artifacts

An accepted composite is written as:
1. a `crucible.world/v1` WorldSpec plus pack files (`packs/_toy/.../forge/env.py:write_pack` pattern);
2. a verifier `Plan` of gates (`verify/runtime/plan.py`) compiled from the `CheckSet`;
3. a `NegativeSet` including the A16 cheats;
4. an `AcceptTask` adapter (`packs/_toy/.../forge/task.py:ToyForgedTask` pattern) whose `AcceptanceReport` is bound to the acceptance key (`accept/key.py:compute`);
5. a task card with ladder pass rates, the composition tree (component hashes, so a component's edit invalidates its composites) and provenance (A8);
6. later, a Harbor `task.toml` export (M11).

### 4.9 Components

DEPEND = call existing code; PORT = copy with licence notice; PATTERN = re-implement the idea.

| # | Component | Acceptance test | P | Effort | Reuse |
|---|---|---|---|---|---|
| 1 | Concurrent orchestrator: level-parallel nodes, many plans, semaphore C | same journal `done` set and accepted task hashes as sequential on the toy world_synth (5 tasks); ledger never exceeds cap under 20 concurrent reservations; wall time < 0.5× sequential | P0 | S | DEPEND `forge/orchestrator.py`, `forge/budget.py` |
| 2 | Compose library: Chain, Fan-in, Perturb (toy first) | 10 toy composites from `toy/helpdesk_v2` pass A1–A8 + A14; a deliberately padded composite (unchecked component) is rejected by A14 | P0 | M | PORT SWE-smith `combine/` loop and pool rule; DEPEND `WorldSpec`, `mutation/actions.py` |
| 3 | A14 necessity, A15 complexity gain, A17 non-interference | fixtures in `accept/registry/fixtures.py` style: a clean and a defective composite each, detector proven by re-injection | P0 | S | DEPEND `accept/checks/base.py:register`, `learnability.estimate` |
| 4 | Solver ladder calibrator + monotonicity | ladder report for the 5 forged toy tasks with L1/L3 intervals; a seeded flaky check produces an inversion that blocks acceptance | P0 | M | DEPEND `runtime/run.py:Runner`, `learnability.measure`; PATTERN `general_agent` pass-rate metadata |
| 5 | Red-team hacker (A16) + hack promotion to A13/registry | with M9 finding B.15 re-introduced (state-only approval check), the hacker finds the forged approval within budget; after promotion A13 kills it; the fixed task has 0 confirmed hacks | P0 | M | PATTERN R2E blind-rollout classification, Karotte hack list; DEPEND `extended.py:Attack`, `accept/registry/` |
| 6 | Yield / cost metrics + stop rules + diversity quota | forge report lists $/accepted task per operator; a generator forced to 0% yield is paused after 20 attempts | P0 | S | DEPEND both ledgers |
| 7 | Zoned forge repo + `informed_by` + commit canary scan | a Verifier-writer commit quoting an oracle line fails as a breach; citations into unreadable zones are rejected | P1 | M | PATTERN GitSwarm (no licence); DEPEND `core/store`, `Orchestrator._breach` |
| 8 | Swarm stage runner: Contribute/Nominate/Abstain, announce dedup, dashboard | Proposer swarm at C=5, B=30 yields ≥ 10 distinct composition plans with ≤ 10% announce collisions; abstentions do not consume B | P1 | L | PATTERN GitSwarm §2 + prompt passages (GS:3338-3625) |
| 9 | Fuse operator (cross-World) + Nest | helpdesk ⊕ a second toy App: fused WorldSpec validates, two fresh starts agree (A9), at least one cross-App check, A17 passes | P1 | M | DEPEND `WorldSpec`, `world/world.py:snapshot/restore` |
| 10 | Stitch operator (coding spans → H4) | the 4 task04 spans stitched back reproduce task04 v3's gates; stitched oracle = v3 oracle verdicts | P1 | M | DEPEND `packs/coding/.../derive.py` |
| 11 | Sandboxed builder and hacker roles via Harbor wrapper tasks | a coding hacker runs in a wrapper task with only agent-visible files; network = model API only | P1 | M | DEPEND Harbor installed agents (m7-m8/02 §7.3 recipe), PORT `assemble_check_task` |
| 12 | Curator: splits, ladder placement, retirement → re-compose loop | a task whose L2 p rises above 0.7 is re-seeded and yields a composite back in band | P1 | M | DEPEND `forge/dedup.py`, ladder |
| 13 | Judge (variant ranking only) | ranking never changes an M9 verdict (property test); diversity quota respected | P2 | S | PATTERN GitSwarm central selector |
| 14 | Persistent per-family specialists (auditor, hacker) with registry memory | the second forge run of a family finds known hacks at 0 extra episodes (replayed from the registry) | P2 | M | PATTERN PI persistent sub-agents; DEPEND `accept/registry/` |
| 15 | G-transplant (OOD) + L5 swarm rung | one transplanted task on an unseen repo accepted; L5 report for one bench task | P2 | L | PATTERN DESIGN §7.4; GitSwarm |

**Order of work:** 1 → 3 → 2 → 4 → 5 → 6 is the P0 core on the toy pack, all CPU-runnable. 7–12 bring the swarm and the coding vertical. The swarm (7–8) comes *after* the composition rigor checks (3–5), because without A14–A16 a swarm only produces more unverified tasks faster.
