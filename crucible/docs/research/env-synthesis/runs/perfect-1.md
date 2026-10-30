# PERFECT-1: P2 learnability, P3 on-intent, P4 novelty, the perfect-task check, the frontier and the outcome classifier (2026-10-29)

Code: DESIGN §7.7 item 19. Data: `crucible/data/synth/perfect/` (gitignored): `difficulty.jsonl` (the store),
`<run>-perfect-{pilot2c,novelty3,check3,frontier2,outcome4,outcome-cal-panel3,frontier-round}/`. The seed-0 store of the
first pilot is kept as `difficulty-pilot1-seed0-INVALID.jsonl` (see issue 1). Student: `Qwen/Qwen3.6-35B-A3B` on Tinker,
thinking off, the JSON agent protocol (`json-agent/v2:calls=20:T=1.0:seed=tag`). Proxies: Nemotron-3.5-Lightning-30B-A3B
(the student's size class, another family). Panel: Nemotron-3-Ultra + gpt-oss-120b. Training-side synthesis runs only
(`load_pool` refuses sealed roots); no benchmark or sealed task was read.

## The pool

89 tasks with executable oracles (every oracle passes its own contract): finance_ap 54 (6 certified, 2 provisional,
46 rejected), it_helpdesk 35 (2 certified, 1 provisional, 32 rejected). Rejected tasks were measured too: they calibrate
the predictor and exercise the classifier, and P1 keeps them out of the pool.

## Issues the real runs found, and their fixes

| # | Found by | Issue | Fix |
|---|---|---|---|
| 1 | the first 25 band decisions were all 0/8 or 4/4 | `WireClient` sampled with seed = retry attempt (0): Tinker returned one trajectory per task, so k rollouts were one sample copied k times | `WireClient(seed_by_tag=True)`: seed from the rollout's tag; simulator string `v2:...:seed=tag`; seed-0 store moved aside ($0.50 lost). With seeds, a task that was 0/8 measured 3/4 |
| 2 | EVID on the real pool: 25 confirmed on 16 it_helpdesk tasks | π* copied a ticket's own priority back; no pattern checks it, so blanking it changed nothing and EVID called the task off intent | only contract-constrained values are derived values; a `derive` value π* writes is traced through that write only. After: 30 trials, 0 confirmed |
| 3 | SWAP pool on it_helpdesk: 7 confirmed on 6 tasks | "escalate T-1006 for an SLA breach" legitimately satisfies "raise T-1006 if the policy requires it"; two phrasings of one task passed each other | pool peers are suspects (a judge decides); mutual acceptance is a duplicate, no trial. Enum swaps stay confirmed (executed proof) |
| 4 | novelty calibration: enum-swap negatives 0/5 correct | "decide invoice A" and "decide invoice B" with different outcomes shared a signature | `choices` (π*'s literal enumerated write arguments) in the signature and key; a decision gate ×0.5 when both write choices and they differ |
| 5 | the same | twins lost their choices (their runs had no taint) and ref-copied values counted as decisions | twin runs with taint; choices from literals only |
| 6 | outcome classifier on 664 real rollouts: 14 "leak" passes | the D7 rule fired on "check whether it duplicates" (TAINT's matcher) plus PO/receipts unread, which a duplicate rejection does not need | a leak with unread evidence is the residue, not decisive |
| 7 | fixture calibration with the real panel | the panel judged the D7 leak attempt *valid* (both members, confidence ≥ 0.99): it judges against the instruction it was given | a valid verdict cannot clear a task-level leak: the label stays uncertain |
| 8 | real rollouts of `smoke/t02` (8/8 ANSWER only) | the contract demands an exact JSON object answer that TAINT reports unsourced (D5); the D5 relation ignored answer checks | answer-check constants relate a D5 value to an ANSWER failure: 8 rollouts move from uncertain to unintended_fail (hidden information) |
| 9 | the pilot took ~2.5 min per task | tasks measured one at a time | `--parallel-tasks` (each rollout keeps its own World context) |
| 10 | an interrupted run | a cut JSONL row would break the store; a `&`-launched run died with its shell | the store skips unparsable rows; runs go through the background runner |
| 11 | frontier requests | failures the task caused (hidden information) were becoming curriculum targets | `requests_from(skill_gaps=...)`: only `intended_fail` rollouts are targeted |

## P2: the band

| | too_hard | in_band | too_easy | total |
|---|---|---|---|---|
| all tasks | 47 | 19 | 23 | 89 |
| certified / provisional | 0 | 5 | 6 | 11 |
| finance_ap / it_helpdesk | 27 / 20 | 13 / 6 | 14 / 9 | 54 / 35 |

p̂ histogram: 47 at 0/8, 12 at 4/4 (stopped at the first look), the rest spread 1/8 to 7/8. 15 decisions reach posterior
mass ≥ 0.9; mean posterior error: too_hard 0.19 (0/8 under Jeffreys), in_band 0.28, too_easy 0.16. Stated frequentist
error (operating characteristic, Jeffreys prior, ±0.05 indifference): worst case 0.34 too_hard, 0.37 in_band, 0.29
too_easy. k ≤ 8 cannot do better; that is the price of the lead's k = 4 → 8 rule.

| p (true) | 0.02 | 0.05 | 0.15 | 0.3 | 0.5 | 0.65 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| P(decide too_hard) | 0.85 | 0.66 | 0.27 | 0.06 | 0.00 | 0.00 | 0 | 0 |
| P(decide in_band) | 0.15 | 0.34 | 0.72 | 0.88 | 0.63 | 0.29 | 0.06 | 0.01 |
| P(decide too_easy) | 0 | 0 | 0.00 | 0.06 | 0.37 | 0.71 | 0.94 | 1.00 |
| expected rollouts | 8 | 8 | 8 | 8.0 | 7.8 | 7.3 | 6.4 | 5.4 |

**Cost per band decision** (student rollouts only): 664 rollouts, $2.42, $0.0037 per rollout, $0.027 per decision;
too_hard decisions cost $1.48 (61%: they always take 8 rollouts). Proxies for this run: $0.33.

**Predictor calibration** (leave one task out, 89 tasks; band accuracy against the measured band):

| features | MAE | Brier | band accuracy (Wilson 95%) |
|---|---|---|---|
| constant baseline | 0.313 | | 0.21 |
| static (steps, writes, reads, depth, required, derive, …) | 0.302 | 0.145 | 0.27 (0.19–0.37) |
| ALT pass share (certification's non-student solvers, free) | 0.246 | 0.115 | 0.22 (0.15–0.32) |
| proxy estimate (Lightning, one call) | 0.316 | 0.148 | 0.21 (0.14–0.31) |
| proxy rollouts (Lightning, k = 2) | 0.267 | 0.121 | 0.26 (0.18–0.36) |
| static + ALT + proxies | 0.231 | 0.112 | 0.42 (0.32–0.52) |
| all features | **0.229** | **0.107** | **0.44 (0.34–0.54)** |

LOO calibration by bin (predicted → measured mean p̂): 0.10 → 0.13 (n 53), 0.30 → 0.39 (20), 0.51 → 0.34 (7), 0.74 → 0.83
(5), 0.87 → 0.75 (4). The predictor beats the constant and ranks tasks, but is **not trusted** (bar: Wilson lower
bound ≥ 0.7): it never skipped a rollout. Had it been trusted, the skip rule would have skipped 12 of 89 decisions with
4 wrong (33%). The online call (a predictor fit on the first 6 records, recorded before each measurement) was right on
0.24 (0.16–0.34) of 83 decisions. The one-call proxy estimate carries no signal; ALT history is the best single
feature and costs nothing.

## P3: on intent

| pool | tasks | on intent | share | admitted-only share | EVID | SWAP | TWIN D9 | TAINT D7 |
|---|---|---|---|---|---|---|---|---|
| finance_ap | 54 | 47 | 0.87 | 8/8 | n/a (the SQL adapter cannot perturb) | 186 trials, 0 confirmed | 4 tasks | 3 tasks |
| it_helpdesk | 35 | 12 | 0.34 | 3/3 | 22 tasks, 30 trials, 0 confirmed | 473 trials, 0 confirmed, 4 suspects | 21 tasks | 8 tasks |

it_helpdesk's low share is TWIN: 21 rejected tasks whose contract accepts the twin's (another ticket's) solution.

## P4: novelty and coverage

Near-duplicate threshold, calibrated on 174 known duplicates (85 twins, 55 twins with a proxy paraphrase, 34 paraphrases)
and 3401 known distinct pairs (2 enum swaps, 1509 cross-cell, 1890 cross-domain): **w = 1.0, gate 0.5, θ = 0.865, TPR
1.00 (0.98–1.00), FPR 0.008 (0.005–0.011)**, margin 0.004. All twin, twin+paraphrase, paraphrase and enum-swap pairs are
right; 26 of 1509 cross-cell pairs score as duplicates (the proposer's cells are a loose label, so this is an upper bound
on FPR). Caveat: w = 1 means the signature alone separates; the positives share a signature by construction, so the
instruction score is not tested by these labels. A spot check of the 39 flagged finance_ap pairs found the same invoice
proposed in two runs ("decide whether INV-45783 can be approved" vs "review INV-45783 against the AP policy") and the
"decide invoice X under the AP policy" template with one outcome.

| pool | tasks | distinct signatures | near-duplicate pairs | effective tasks | cells covered (all) | cells covered (admitted) |
|---|---|---|---|---|---|---|
| finance_ap | 54 | 52 | 39 | 30 | 4/4 (19/14/11/10) | 3/4 (6/0/1/1) |
| it_helpdesk | 35 | 35 | 3 | 32 | 4/4 (14/8/7/6) | 2/4 (2/0/1/0) |

## The perfect-task check

| pool | candidates | valid | learnable | on intent | novel | **admitted (all four)** | novel vs earlier candidates |
|---|---|---|---|---|---|---|---|
| finance_ap | 54 | 6 | 13 | 47 | 48 | **3** | 31 |
| it_helpdesk | 35 | 2 | 6 | 12 | 35 | **0** | 32 |

Admitted: `finance_ap/20261029-144119-smoke2/t01`, `…-160300-full2/t02`, `…-165857-rsi-dense/t10`. `…-152544-full/t04`
is valid, in band and on intent, and is refused as a duplicate of smoke2/t01 (score 1.0). The other valid tasks are too
easy (5) for this student.

## The outcome classifier (the RSI loop's probe side)

**Calibration on known labels** (26 cases: 12 clean-task attempts, 14 planted defects; MUT on):

| | execution only | execution + panel (λ = 0.7, calibrated 15/15 on oracle, no-op and minus-last-write anchors) |
|---|---|---|
| decided | 25/26 | 25/26 |
| accuracy on decided | 1.00 | 1.00 |
| skill gap called a defect | 0 | 0 |
| defect called intended | 0 | 0 |
| uncertain | D7 leak (by design: not decisive alone) | D7 leak (the panel said valid; it cannot clear a leak) |

Confusion (rows expected, execution + panel): intended_pass 7/7, intended_fail 8/8, unintended_pass 6/7 (1 uncertain),
unintended_fail 4/4. These cases were written with the rules in view: they show the rules work as designed, not their
recall on natural defects.

**Real rollouts** (676: the 664 pilot rollouts and the frontier round's 12; panel λ = 0.7 calibrated on 30 real anchors,
29 correct, error bound 0.15; $0.25):

| label | rollouts | main causes |
|---|---|---|
| intended_fail | 424 | skill gap: decision (wrong AP or triage action), evidence gathering, completion |
| intended_pass | 132 | 78 by execution, 54 by the panel (passes that skipped a collection π* reads) |
| unintended_fail | 52 | hidden information (TAINT D5 on the failing pattern or answer: an exact free-text note, an exact JSON answer) on 7 tasks |
| unintended_pass | 1 | the panel judged a pass that never read the vendor record invalid |
| uncertain | 67 | 35 D5 on the task but unrelated to the failure, 21 passes with a TAINT D7 leak (the panel cannot clear), 11 routed (budget) |

A spot check of four `intended_fail` traces: each is a wrong decision with the evidence read (e.g. approving an invoice
whose vendor shares a bank account with a blocked vendor). No skill gap was labelled a task defect by construction of
the rule; whether some `intended_fail`s hide an ambiguous instruction is not measurable without labels.

## The frontier (K)

Snapshot-replay attribution of the 517 failed pilot rollouts: irreversible 396, omission 56, forbidden 33, answer 16,
wrong_write 4, no_action 1, budget 11 (routed), unreplayable 0. Modes: decision 249, evidence_gathering 150, completion
56, frame_discipline 33, reporting 16. Targeting only the 419 skill-gap rollouts gives 18 requests; the biggest are
decision in c1 (72 traces, 14 tasks) and c2 (65, 12), both "ease: state one derived value" (their source tasks are
mostly too hard). The admission plan against the 11 admitted tasks admits 2 targeted entries and refuses the rest at the
20% cap; R4 holds (KL 0.005, entropy 0.76 → 0.69 ≥ H_min 0.68).

**One real round** (finance_ap, the two requests above, 4 tasks per arm, the same swarm cap per arm):

| arm | spend | certified | in band | perfect | J (perfect per $) |
|---|---|---|---|---|---|
| targeted | $0.378 | 1 | 0 (the certified task was too easy, 4/4) | 0 | 0 |
| control (default cells) | $0.270 | 1 | 1 | 1 | 3.7 (0–11.1) |

Equal budget (within 35%); verdict: no difference (n = 4 per arm). The control's perfect task was admitted (R4: KL 0.002).
What it shows: the loop runs end to end on real models, and the ease knob for a decision gap ("state one derived value")
overshot to too easy on its one certified task; whether targeting beats untargeted needs tens of tasks per arm.

## Cost

| item | $ |
|---|---|
| invalid seed-0 pilot (issue 1) | 0.60 |
| student smokes | 0.02 |
| the pilot, three runs resumed through the store (pilot2 0.20, pilot2b 1.81 until its shell died, pilot2c 1.73): 664 stored rollouts ($2.42) plus proxies and the rollouts of interrupted tasks | 3.74 |
| paraphrases for the novelty calibration (Lightning, 3 runs) | 0.01 |
| outcome panels (3 real runs, 3 fixture calibrations) | 0.53 |
| frontier round (swarm 0.61, student 0.04) | 0.65 |
| **total, every receipt under a "perfect" path** | **5.55 of the $15 cap** |

The shared Part 2 total after this work: $61.76 of $150.
