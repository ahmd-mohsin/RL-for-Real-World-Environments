# YIELD-1: throughput and yield of the synthesis pipeline (2026-10-29/30)

Data: `crucible/data/synth/yield/` (gitignored): `failures-m4-rsi1.json`, `failures-yield-1.json`, `probe-alt-1/`,
`study-oracle-{1,2}/`, `study-alt-1/`, `lemmas-1/`, `smoke-{1,2}/`, the arms `yield-{1,2,3}/`, the speed runs
`speed-par/`, `speed-ser2/` (`speed-ser/` was cut off: the deadlock below). Swarm only, no Qwen in any role, the student
never called. Builders as in M4 (DeepSeek proposes, writes oracles and repairs; GLM writes contracts); ALT
Nemotron-3-Ultra ×2 + gpt-oss, ATK Nemotron-3-Super + gpt-oss, panel Nemotron-3-Ultra + gpt-oss, abstractor gpt-oss.
Code: `synth/batch.py`, `synth/offload.py`, `synth/yieldkit.py`, `synth/failures.py`; CLI `synth.py batch |
batch-report | failures | lemmas`. Spend: USD 30.78 of the 35 cap (diagnostics 1.72, lemma batch 0.004, smokes 1.33,
yield-1 11.45, yield-2 9.00, yield-3 6.70, speed runs 0.58).

## 1. Why proposals fail (M4 and RSI-1 records)

`synth.py failures` over M4 full and full2 and RSI-1 dense (finance_ap 52 tasks, it_helpdesk 32). The finance_ap oracle
is checked against the pack's own policy decision for every invoice its contract pins (`synth_wire.finance_ap_truth`).

| primary reason (one per failed task) | finance_ap | it_helpdesk |
|---|---|---|
| ALT:no_pass | 31 | 3 |
| TAINT:confirmed | 4 | 8 |
| RED:confirmed | 2 | 9 |
| TWIN:confirmed | 1 | 6 |
| ALT / ATK / MUT suspects open | 4 | 2 |
| oracle:dead | 3 | 1 |
| MUT:upper (provisional) | 2 | 1 |
| certified | 5 | 2 |

**finance_ap: half the oracles were wrong.** 48 tasks have a checkable oracle.

| proposer | oracle right | oracle wrong |
|---|---|---|
| Kimi-K2.6 (M4 full) | 17 | 0 |
| DeepSeek-V4.1-Flash (M4 full2, RSI-1 dense) | 15 | 16 |

14 of the 31 `ALT:no_pass` rejections had a wrong oracle: the solvers that applied the policy correctly failed the
contract, and the panel, which sees the intent card with the proposer's `derive`, judged them invalid. The panel split
(members disagree) on 10 failed attempts of wrong-oracle tasks and on 2 of right-oracle tasks: disagreement is an oracle
error signal. Cause: the oracle programmer is shown the proposer's `derive` and copies it, and DeepSeek's derive misses
the policy's traps (shared-bank-account blocks, normalised duplicate numbers, this World's own rule order, the
tolerance edge).

**Failed ALT attempts by mode** (panel reasons, `failures.alt_mode`; finance_ap, 100 attempts):

| mode | Nemotron (Ultra, run twice) | gpt-oss | oracle right | oracle wrong |
|---|---|---|---|---|
| wrong_decision | 30 | 4 | 22 | 10 |
| no_write (reads only) | 10 | 18 | 15 | 13 |
| no_action (empty trace) | 1 | 14 | 10 | 5 |
| split (panel disagrees) | 10 | 2 | 2 | 10 |
| skipped_rule | 3 | 4 | 3 | 4 |
| not_found, incomplete, other | 3 | 5 | 4 | 2 |
| valid (D6: the contract rejected a valid attempt) | 2 | 0 | 0 | 2 |

On it_helpdesk the cheap classes reject first (RED D4, TAINT, TWIN D9); 5 failed ALT attempts were judged valid
(over-strict contracts, D6).

**Why ALT solvers fail on otherwise valid tasks** (12 stored finance_ap tasks whose oracle is right, one attempt per
model and harness, `study-alt-1`):

| solver | v1 harness: pass | v2 harness: pass | v1: read the policy | v2: read the policy |
|---|---|---|---|---|
| Nemotron-3-Ultra | 4/12 | 4/12 | 6/12 | 12/12 |
| Nemotron-3-Super | 0/12 | 2/12 | 5/12 | 10/12 |
| gpt-oss-120b | 0/12 | 0/12 | 0/12 | 1/12 |

The failures of a correct task are solver weakness on a deliberately tricky policy (the attempts write a wrong decision:
`MISSING_REQUIRED` plus `FORBIDDEN_EFFECT` on the same invoice), not hidden information or an over-strict contract.
gpt-oss decides without reading ("Likely approve") and often finishes with no call. Over all records the order of
causes for an `ALT:no_pass` on finance_ap is: a wrong oracle (14 of 31), then solver weakness on a right oracle (17 of
31). Ambiguity shows as panel splits, mostly on wrong oracles. Over-strict contracts are rare on finance_ap (2 valid
attempts rejected), more frequent on it_helpdesk (5).

## 2. Framework bugs the real runs found (all fixed, with tests)

| # | Found in | Bug | Fix |
|---|---|---|---|
| 1 | M4 records | Ultra and Super shared the runner name `alt_solver-nemotron`; ALT finds runners by name, so Ultra ran twice and Super never ran | unique names, duplicates refused, `SYNTH_ROLES` = what ran (Ultra ×2 + gpt-oss) |
| 2 | smoke-1 | two proposer workers proposed the same cell before either batch landed (6 candidates for a max of 4) | proposer calls reserve their proposals |
| 3 | smoke-1 | calibration swapped `Runner.ready` while builds appended to it: a built task was lost (`error`) | `calibrate_panel(ready=)` |
| 4 | yield-1 | the blind oracle read the purchase order a po_mismatch invoice bills; it does not exist, and the correct oracle was dropped as dead (3 in `a1`) | read-only errors are evidence |
| 5 | yield-1 | the proposer ran 30+ candidates ahead of refine, so the soft spend limit could not act | backpressure (`max_backlog` = build + refine workers) |
| 6 | yield-2 | a reached cap came back as `EngineCallError`: panel members abstained, ALT attempts failed, the arm kept refusing calls instead of stopping | the wire and `Resilient` re-raise the budget refusal |
| 7 | speed-ser | serial mode: the paused proposer left 3 built tasks while calibration waited for 4 anchors (deadlock) | calibrate when nothing is being built |
| 8 | lead (item 19) | every call on seed 0: 120 of 216 Ultra attempt pairs in yield-1/2 were the same trajectory | per-agent ALT/ATK tags, the batch wire seeds by tag |

Bug 8 halves the effective ALT attempts of both arms of yield-1 and yield-2 alike (the same pool, the same seeding), so
their comparison stands, but their absolute ALT pass rates are those of about 2 independent attempts, not 3. With seeds
by tag (yield-3) 27 of 79 pairs still coincide (at temperature 0.3 a short trajectory often repeats; 69 of 79 share the
first call). Proposer
duplicates are not affected (each proposer call has its own cell, batch and `existing` list, so no two prompts repeat).

## 3. Yield improvements, measured

Offline diagnostics first (stored proposals, finance_ap truth):

| change | measurement | result |
|---|---|---|
| `ORACLE_PROGRAMMER_V2`, blind to derive | 31 stored DeepSeek proposals, oracle truth | right 29/31 (v1: 15/30) |
| agreement, proposer derive vs blind oracle | the same 31 | disagree on 18; the oracle is right on 16 of them, so a disagreement re-derives the intent card (`reconcile`) instead of dropping |
| `CLARITY_CHECKER` (GLM) | 19 proposals | drops 1 wrong and 3 right oracles, about USD 0.02 a call: off by default (arm `a1c`) |
| `ALT_SOLVER_V2` + finish nudge | 12 right-oracle tasks × 3 solvers | Ultra 4→4, Super 0→2, policy read 11→22 of 24 |

Arms on fresh proposals at equal hard caps, round-robin over the same cells (finance_ap seeds 0–3 at 30 invoices × 4
cells, it_helpdesk dense × 4 cells; quota high so no cell closes): `a0` = M4 prompts and pipeline; `a1` = v2 prompts,
blind oracle with reconcile, pre-filters (unknown id, leaked derive), ALT v2 harness with the finish nudge;
`a1_lemmas` = `a1` plus the lemma batch. The same falsifier population in every arm of a run (yield-1/2: before main's
item 20; yield-3: with item 20's read mutants and LEAK class).

| run | arm | candidates (unique) | admitted | provisional | USD | admitted per USD (95% bootstrap) | USD per admitted | yield (Wilson) | admitted with a wrong oracle |
|---|---|---|---|---|---|---|---|---|---|
| yield-1 | a0 | 60 (56) | 13 | 2 | 3.41 | 3.82 (2.04–5.74) | 0.26 (0.17–0.49) | 0.23 (0.14–0.36) | 2 of 11 checkable |
| yield-1 | a1 | 60 (55) | 21 | 1 | 3.24 | 6.49 (4.43–8.77) | 0.15 (0.11–0.23) | 0.38 (0.27–0.51) | 0 of 14 |
| yield-1 | a1_lemmas | 60 (57) | 20 | 0 | 4.80 | 4.16 (2.61–5.74) | 0.24 (0.17–0.38) | 0.35 (0.24–0.48) | 1 of 16 |
| yield-2 † | a0 | 47 (45) | 3 | 0 | 3.82 | 0.78 (0.00–1.87) | 1.27 | 0.07 | 1 of 3 |
| yield-2 † | a1 | 65 (58) | 11 | 2 | 3.88 | 2.83 (1.39–4.58) | 0.35 | 0.19 | 0 of 10 |
| yield-3 ‡ | a0 | 42 (40) | 7 | 2 | 3.21 | 2.18 (0.82–3.95) | 0.46 (0.25–1.24) | 0.18 (0.09–0.32) | 2 of 7 |
| yield-3 ‡ | a1 | 44 (43) | 12 | 3 | 2.98 | 4.03 (2.22–6.09) | 0.25 (0.19–0.54) | 0.25 (0.15–0.39) | 0 of 9 |
| **pooled** | **a0** | 141 unique | **23** | | **10.44** | **2.20 (1.38–3.13)** | **0.45** | | **5** |
| **pooled** | **a1** | 156 unique | **44** | | **10.10** | **4.36 (3.23–5.55)** | **0.23** | | **0** |

† yield-2 hit its 4.50 caps while bug 6 was open: only the candidates that finished before the cap was approached are
counted (47 of 65 and 65 of 78), with their own costs. ‡ yield-3 ran after the rebase onto main (item 20's falsifiers)
with seeds by tag; its proposer reached the soft limit before the it_helpdesk cells (a0 0, a1 3 of them).
`a1` hit its cap, so only candidates finished before it are counted.

**a1 − a0, admitted per USD, pooled: +2.15 (95% bootstrap 0.71–3.63, P(≤0) = 0.002); yield difference 0.024–0.210
(Newcombe).** Per run: yield-1 +2.67 (−0.11..5.50), yield-2 +2.05 (0.33..4.03), yield-3 +1.85 (−0.63..4.37). Per
domain, the gain is finance_ap's (yield-1: 20/48 vs 11/48 admitted); it_helpdesk did not improve (1/12 vs 2/12).
Where the money goes (yield-1): ALT 39–44 %, contract 13–19 %, oracle 14–17 %, adjudication 6–9 %, repair 8–12 %.

**The lemma batch** (`synth.py lemmas`): 47 (counterexample, repair) pairs from M4/RSI-1 in 30 clusters → the gpt-oss
abstractor wrote 9 lemmas → the deterministic leakage critic refused 2 (they quote a reason code of the evidence) → 7
kept (3 contract writer, 2 proposer, 1 oracle programmer, 1 repairer). Against `a1` in yield-1 (`meta.lemmas.
lemma_acceptance`): **refused**: unequal budget (4.80 vs 3.24), the first-round confirmed rate did not fall (29/53 vs
26/48, difference interval −0.18..0.19), J did not rise (4.16 vs 6.49 admitted per USD).

**Goodhart check (R4).** The admitted tasks of `a1` are not smaller than `a0`'s (oracle steps median 7 in both in
yield-1; `a1` spans 12 cells, `a0` 10).

**Validity.** No admission rule changed (U_MUT ≤ 0.05, an ALT pass from a non-builder family, no confirmed
counterexample). The finance_ap truth check shows what the certificate cannot see: see §5.

## 4. Speed

Equal output, one arm, finance_ap seeds 0–1, 6 proposals, 3 of them refined (`a1`):

| run | mode | wall s | serial-equivalent s (sum of every candidate's stage times) | ratio |
|---|---|---|---|---|
| speed-ser2 | serial: one candidate, one call at a time, in-process falsifiers | 407 | 374 | 0.92 |
| speed-par | parallel: pipelined pools, 4 falsifier processes, concurrent agents | 215 | 457 | 2.12 |

The serial run's ratio near 1 checks the metric (one thing at a time: the stage times add up to the wall clock). At
equal output the parallel runner was 1.9× faster on this small run (3 candidates cannot fill the pools).

At batch scale (every arm of a run at once):

| run | arms × candidates | wall s | serial-equivalent s | speed-up | admitted | admitted per hour |
|---|---|---|---|---|---|---|
| yield-1 | 3 × 60 | 2 344 | 39 776 | 17.0× | 54 | 83 |
| yield-2 | 2 × 65–78 | 1 197 | 15 551 | 13.0× | 14 (cap-hit tail) | 42 |
| yield-3 | 2 × 42–49 | 843 | 9 292 | 11.0× | 19 | 81 |

Per arm the speed-up is 6.1–6.9×; with arms side by side it multiplies. `a1` alone: 37 admitted per hour and 98 s of
wall clock per admitted task in yield-1, 51 per hour and 70 s in yield-3. Time per stage (yield-1, `a1`, summed over
candidates): propose 1 001 s, build 1 146 s, refine 11 541 s; model latency summed per stage: ALT 2 290 s (934 calls),
oracle 805 s, repair 607 s, adjudication 495 s, propose 331 s. The deterministic falsifiers took 11 200 s of worker
time over the 2 344 s batch on 6 processes: refine is CPU-bound in the falsifier pool, not in the model calls.

**What 10 environments × 10 tasks would take** at the pooled `a1` rate: about USD 23 (0.23 per admitted task) and,
with three arms' worth of pools running side by side as in yield-1, a little over an hour of wall clock, if each
environment admits like finance_ap. it_helpdesk admitted 1 in 12 under `a1`: environments like it need the contract-side
fixes first (§5).

## 5. Open issues

- **Certified tasks with a wrong oracle.** Over yield-1–3, 5 admitted finance_ap tasks of `a0` (and 1 of `a1_lemmas`, 0 of
  `a1`) pin a decision the pack's policy does not reach: compound cases where this World's own rule order decides the
  reason code, and a 1 % tolerance edge. The oracle and the ALT solvers made the same mistake, so the contract accepted
  the solver; no falsifier class checks the reference decision against the World's rules, and the panel judges only
  failing ALT attempts. The same wrong oracles poison calibration (oracle traces are the "valid" anchors: yield-2 `a0`
  fell to λ = 0.95, uncalibrated). Only domains with executed ground truth (finance_ap) can measure this rate.
- **it_helpdesk does not improve** (`a1` 1/12 vs `a0` 2/12 admitted in yield-1): its losses are cheap-class
  counterexamples and repairs that re-open archived ones, which the v2 writer prompt did not move.
- **Run-to-run variance is large** (yield-2 admitted far less than yield-1 in both arms, before its cap). The pooled
  comparison is the claim; single runs are not.
- **The machine was shared** (load average about 30 on 14 cores during every run): wall-clock numbers are upper bounds.
- **The falsifier pool's CPU** bounds refine; more worker processes need more free cores.
