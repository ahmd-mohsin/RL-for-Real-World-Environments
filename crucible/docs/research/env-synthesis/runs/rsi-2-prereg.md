# RSI-2 preregistration

Written 2026-10-29, before any RSI-2 run, and committed before it. RSI-2 runs on the merged system (the World families,
the batch runner and yield, composition, learnability, novelty and frontier, the falsifier blind spots) only after the
lead's go. Anything this file does not fix is decided before the run in a dated amendment below, never after seeing
results. The statistics are specified in `runs/rsi-stats.md`; the code is `crucible.synth.meta` at the commit that adds
this file.

## 1. The pool RSI-2 is sized for

The production target: **10 environments × 10 tasks = 100 admitted tasks**.

| role | environments | tasks | what reads it |
|---|---|---|---|
| dev, proposal split | 6 | ~30 (half of each dev environment's tasks, by source task) | proposers, birth gates, the planter loop, A/A and audits |
| dev, decision split | (the same 6) | ~30 | only R2/R9/prune decisions, each unit at most once (R1-strict) |
| locked set | 3 held-out environments | 30 | reports every round; decisions only through the reusable holdout (gate G) |
| held-out World family | 1 environment of a World family no other environment belongs to | 10 | reports only; no decision ever reads it |

Assignment is fixed before the run by a seeded hash of the environment names, with the constraint that the held-out
family shares its World family with none of the other nine. Plants: the dev planter (`base_planter`) on dev sources; the
held-out planter (`held_out_planter_v2`: the M3 held-out variants plus five RSI-2 D3 variants, disjoint from dev, `max_per_op = 6`) on locked and family sources. Expected yields
(from rsi-2: 12.5 dev plants per task; locked yields in §5): about 375 decision units, 375 proposal plants.

## 2. Hypotheses

| # | Hypothesis | Primary metric | Test |
|---|---|---|---|
| H1 | The loop finds falsifier updates that generalise | recall of the final F on the **held-out World family** against the round-0 population, paired on the same plants | exact one-sided McNemar, α = 0.05 |
| H2 | The loop beats the equal-budget random-DSL control | the same, loop arm against control arm, paired on held-out-family plants | exact one-sided McNemar, α = 0.05 |
| H3 | Updates transfer across domains (E3) | locked-set recall of operators grown on one dev domain's evidence, on the other domains | McNemar, α = 0.05, Holm over the domains |
| H4 | The planter game keeps producing headroom | dev headroom per round while natural-likeness holds (MMD ≤ μ) | descriptive, with the trajectory |
| H5 | The per-task probe loop repairs only task defects | share of accepted repairs that re-open no intended fail (T2) and trip no drift alarm; unintended outcomes on re-probe after repair | descriptive; the drift alarm (§6) |
| H6 | The regularisers matter (E10) | Δ against the full loop, paired by seed, in held-out-family recall and Goodhart drift | per ablation, 95% t interval over 3 seeds; reported, not a gate |

No hypothesis uses the student's pass rate. H5 counts outcomes and replays.

## 3. Metrics

- **J proxy of an F update**: recall on decision units (a unit scores 1 when some class confirms a counterexample); FA on
  anchors is a hard 0.
- **Per test**: t, α_t, e-value, stopping point (crossed / futility / end), paired mean gain with its bootstrap and
  cluster-bootstrap (by source task) 95% intervals, the anytime LCB, δ_AA.
- **Per round**: dev recall, locked FR and FA per class with Wilson intervals, held-out family FR, headroom, the P/F
  game (planter rewards, pool class KL against round 0, pool MMD to the 47 independent real defects, plant size), α
  spent, decision units left, Thresholdout budget left, the probe loop's counts.
- **Goodhart drift**: pool class-mix KL, plant size (contract diff, witness length), library size.

## 4. Thresholds and the decision rule

| item | value |
|---|---|
| α (whole run, per arm) | 0.05, online Bonferroni |
| t_max (α-spending tests per arm) | 10, so α_t = 0.005 (§5) |
| A/A | 5 replicates, seed pairs (1,2)…(9,10), one proposal-split view of 240 units; δ_AA = max(2·sd, max abs), raise-only |
| decision view | up to 240 fresh units, chunks of 20, stop on crossing 1/α_t; futility: mean gain ≤ δ_AA after 60 units |
| bet | aGRAPA, c = 0.75 |
| R3 bands | between-view, ≥ 3 calibration views, min band 0.02; FA_max 0 |
| R4 | κ = 0.1, H_min = 0 |
| R5 | edit budget 3 → 1 cosine over the rounds; β = 0; prune window 3 |
| R7 | held-out-domain mean gain > δ_AA |
| G | Thresholdout threshold 0.04, σ 0.01, budget 10; locked gain answer ≥ 0 |
| meta loop | Fisher exact one-sided at α_t; only with ≥ 4 candidates per side |
| planter | witnessed, novel, natural-like (MMD ≤ 1.1 × MMD(base dev plants, the 47 independent real defects)), headroom reward > 0 |
| probe loop | 8 probe rollouts per task; T1–T7 of `runs/rsi-stats.md` §2; ≤ 3 repairs per task; judge cap 2 |
| drift alarm | ≥ 5 re-probed repaired tasks (8 re-probe rollouts each), mean unexplained change > 0.1 with a 95% bootstrap interval excluding 0 |

## 5. Power and minimum detectable effects (stated honestly)

All numbers from `scripts/synth_rsi2.py power` and `meta/stats.py` (exact McNemar enumeration; simulated betting power,
500–800 runs each). Seed noise: the A/A replicates on the rsi-2 dev pool give a per-unit discordance of 0.0064 (p01 =
0.0032 per side) and δ_AA = 0.0196 at 125 units (0.0327 at 75).

**Per decision (R2 at α_t, one-sided, power 0.8).** The minimum detectable paired recall gain on n fresh decision units,
with δ_AA scaled to n:

| n units | δ_AA | MDE at α_t = 0.0025 (t_max 20) | MDE at α_t = 0.005 (t_max 10) | MDE at 0.05 (uncorrected) |
|---|---|---|---|---|
| 40 | 0.035 | 0.42 | — | 0.26 |
| 60 | 0.028 | — | 0.27 | 0.20 |
| 80 | 0.025 | 0.22 | — | 0.15 |
| 120 | 0.020 | — | 0.15 | 0.11 |
| 160 | 0.017 | 0.14 | — | 0.10 |
| 240 | 0.014 | 0.094 | 0.094 | 0.074 |
| 375 | 0.011 | — | 0.061 | 0.051 |

**Consequence (stated before the run).** RSI-1's best candidate gained 7 of 125 dev plants (0.056). No feasible decision
of RSI-2 detects a gain that small under run-level control: it needs ~375 fresh units in one test. RSI-2 can detect a
system update that closes ≥ 10% of the decision units at n = 240. **Decision for RSI-2:** t_max = 10 (α_t = 0.005),
n_max = 240, chunks of 20, futility after 60 units; the decision split (~375 units) supports one or two full-size
tests plus several early stops, and planters that join add units to it. §4 uses these values:

| item | RSI-2 value |
|---|---|
| t_max | 10 |
| decision view | up to 240 fresh units, chunks of 20, futility after 60 |
| A/A view | 240 proposal-split units (δ_AA at the decision size) |

**Per class on the locked set (McNemar, power 0.8, p01 = 0.0032).** Plants needed for a recall gain g: g = 0.1 → 71,
g = 0.15 → 46, g = 0.2 → 34, g = 0.3 → 22 (Bonferroni over 11 classes: 112, 72, 53, 34). The rsi-2 locked set (72
plants, 7 sources) and its rebuild with `max_per_op = 6` (81 plants, `rsi-2c`) show that the per-task cap is not the
constraint: D1, D3 and D4 yield one witnessed plant per 7 source tasks. Projected to RSI-2's 30 locked tasks at
rsi-2's per-task yields:

| class | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plants | 4 | 26 | 4 | 4 | 21 | 51 | 77 | 9 | 26 | 56 | 30 |
| MDE | none | 0.26 | none | none | 0.31 | 0.14 | 0.10 | 0.64 | 0.26 | 0.13 | 0.22 |

**D1, D3, D4 and D8 stay under-powered** in the 10 × 10 pool (no gain is detectable at n = 4). Their per-class results
are reported descriptively with Wilson intervals, never as a test. Reaching ≥ 30 per class needs about 210 source tasks
at the current yield, or new held-out variants for these four classes with ≥ 1 witnessed plant per task (an open item
for the planter, not done here). **Pooled over classes** the primary tests are well powered: the held-out World family
(10 tasks, ~100 plants) has MDE 0.075; the locked set (~310 plants) has MDE 0.03 (0.035 at Holm's first step over 3).

**E10.** With 3 seeds per ablation, a paired difference to the full loop is detectable only when it is large against the
seed-to-seed spread (t interval with 2 degrees of freedom: half-width 4.3 sd/√3). E10 is reported as effect sizes with
intervals, not as tests.

## 6. Rounds and stopping rules

| rule | value |
|---|---|
| rounds | at most 6 per arm |
| candidates per round | 8 operator candidates (4 from A, 4 from A′ when the meta loop runs), 3 planter candidates |
| spend | hard cap set by the lead at the go (the pilot used $⟨pilot⟩) |
| α | stop when t_max tests are spent |
| decision split | stop when fewer than 20 fresh units remain |
| futility | stop after 2 consecutive rounds with no accepted F update and no accepted planter while dev headroom is 0 |
| rollbacks | stop at 2 rollbacks |
| Goodhart | stop when the dev pool's class-mix KL against round 0 exceeds 0.5 |
| drift | the probe loop's repairs stop when the drift alarm fires |
| headroom precondition | measured at round 0 on the proposal split and on the locked set. If dev headroom is 0 and the P loop's headroom-rewarded planters add none in rounds 1–2, the F loop stops (`futility`) and H1–H3 are reported as **not testable on this pool**, never as a negative result. If the locked set's headroom is 0, gate G still runs (non-regression) and H1–H3 rest on the held-out family alone. (On the rsi-2 pool after DESIGN item 20: dev headroom 7/125, all panel-only D3 gaps; locked headroom 0 under 18 held-out variants.) |

## 7. Analysis plan

1. Report every decision of every arm from the ledger, accepted or refused, with its rules (non-compensatory).
2. H1, H2: one McNemar each on the held-out World family plants, primary; the locked set is secondary (it was read
   through the holdout by G, so its final number is reported with the holdout's access count).
3. H3: E3 runs (evidence from one dev domain only), locked recall per held-out environment, Holm-corrected.
4. H4: the P/F game's trajectory table; a headroom of 0 with misses left is reported as a DSL blind spot list.
5. H5: the probe loop's outcome table (counted, demoted by rule, repairs accepted/refused by rule), the drift monitor,
   the lessons lifted (archive counterexamples, lemma candidates, World-builder fixes).
6. H6: E10 at equal budget (same rounds, candidates and seeds), 3 seeds, ablations R1, R3, R4, R5, R7, R8, NL.
7. Effects are reported with intervals; "not significant" is reported with the MDE of §5, never as "no effect".

## 8. Student probes

RSI runs at task-creation time. The per-task probe loop (`meta/rsi.ProbeLoop`) runs on every admitted task of the
100-task pool: the learnability agent's probe runner (`learn.StudentProbe`, 8 rollouts, `WireClient(seed_by_tag=True)`
so the rollouts are independent samples) and its outcome classifier (`synth.outcome`, consumed through
`rsi.LearnClassifier`); repairs through CEGIS under the monotone archive; T1–T7; ≤ 3 repairs per task; a re-probe of 8
rollouts after each accepted repair, for the drift monitor only. The student is the only model probed; probes are not
training. The online monitor during RL training is a separate safety net and is not an input to RSI-2. No pass rate
enters J or any gate. Lessons lifted from the probe loop (archive counterexamples, lemma candidates, World-builder
fixes) become system-level candidates and are decided like every other update (§4), with the generalisation gates.

## 9. Amendments

None yet.
