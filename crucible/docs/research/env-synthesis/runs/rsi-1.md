# RSI-1: the first real rounds of the regularised loop (2026-10-29)

Data: `crucible/data/synth/rsi/{rsi-1,rsi-2}/` (gitignored); locked sets `crucible/data/sealed/synth-locked/rsi-{1,2}-locked.json`.
Swarm only (abstractor, planter proposer, guidance editor: gpt-oss-120b; seeder: Kimi-K2.6; task builders as in M4). No Qwen.
rsi-1 is the first run; its E2 is void (δ = 0, see "Runs and fixes"). rsi-2 holds the headline numbers.

## Runs and fixes

| Run | What | Result | Fix (commit) |
|---|---|---|---|
| dense it_helpdesk World | `reseed` 12→40 tickets, 8→18 agents, 10→28 notes, 8→16 KB | accepted, $0.23 | denser Worlds (559d0268) |
| dense certify, 12 tasks each | finance_ap `--invoices 30`; it_helpdesk on the dense World | MUT distinct n: finance_ap 31–65 (11/12 ≥ 59), it_helpdesk 22–163 (11/12 ≥ 59); before 20–41 | — |
| M4 certificates re-counted | the panel-wrapped MUT had lost `finalize` | M4 "certified" tasks: n_MUT 66 reported, 20 distinct → provisional | 559d0268 |
| rsi-1 E2 (4 rounds) | one operator accepted at LCB 0.011 | **void**: δ = 0 (views stratified by domain × defect all had one J) | 60ddf3cf |
| rsi-1 planter loop | 12 planters accepted | copies of existing plants (a factor on eq_to_any, why texts), reward 0 | novelty + reward > 0 (60ddf3cf) |
| rsi-1 locked set | no D2, D3 column | — | `widen_frame:field` one draft, `relax_value:eq_to_in` (60ddf3cf) |
| rsi-2 first E2/E3 start | natural-likeness reference = archive (falsifier-found) | aborted after 1–3 rounds (`aborted-archiveref-*`) | reference = M5's 47 independently found P1 defects |
| rsi-2 E2 round 4 | prune "removed" two never-accepted operators | no effect on F (library empty) | `last_gain` from the incumbent only (this commit) |

## Sources and sets (rsi-2)

| | finance_ap | it_helpdesk | ap-test (fixture) | desk-test (fixture) | total |
|---|---|---|---|---|---|
| admitted sources, re-checked clean | 8 | 3 | 3 | 3 | 17 (0 alarms) |
| dev sources / plants | 4 / 50 | 2 / 24 | 2 / 20 | 2 / 31 | 10 / 125 |
| locked sources / plants | 4 / 40 | 1 / 12 | 1 / 8 | 1 / 12 | 7 / 72 |

## E1: FR/FA on the locked set (rsi-2, 72 plants, 7 anchors; Wilson 95%)

| class | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D11 | FA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REP | 0/1 | 0/6 | 0/1 | 0/1 | 0/5 | 0/12 | 0/18 | 0/2 | 0/6 | 0/13 | 7/7 | 0/7 |
| RED | 1/1 | 0/6 | 0/1 | 1/1 | 0/5 | 0/12 | 0/18 | 0/2 | 0/6 | 0/13 | 0/7 | 0/7 |
| TAINT | 0/1 | 0/6 | 1/1 | 0/1 | 5/5 | 0/12 | 14/18 | 0/2 | 0/6 | 13/13 | 0/7 | 0/7 |
| NULL | 1/1 | 0/6 | 0/1 | 1/1 | 0/5 | 0/12 | 0/18 | 2/2 | 0/6 | 0/13 | 0/7 | 0/7 |
| MUT | 1/1 | 6/6 | 0/1 | 1/1 | 0/5 | 0/12 | 0/18 | 0/2 | 6/6 | 13/13 | 0/7 | 0/7 |
| TWIN | 0/1 | 0/6 | 0/1 | 0/1 | 0/5 | 0/12 | 0/18 | 0/2 | 3/6 | 0/13 | 0/7 | 0/7 |
| PERT | 0/1 | 0/6 | 0/1 | 0/1 | 0/5 | 0/12 | 0/18 | 0/2 | 0/6 | 0/13 | 0/7 | 0/7 |
| **any** | 1/1 | 6/6 | 1/1 | 1/1 | 5/5 | **0/12** | **14/18** | 2/2 | 6/6 | 13/13 | 7/7 | 0/7 |
| any, Wilson | .21–1 | .61–1 | .21–1 | .21–1 | .57–1 | .00–.24 | .55–.91 | .34–1 | .61–1 | .77–1 | .65–1 | .00–.35 |

Population FR 56/72 = 0.78; FA 0/7. Misses: `over_constrain:never` (D6) 12/12, `leak_decision:humanised` (D7) 4/9. Dev set (base
variants, 125 plants): 118/125 = 0.94, FA 0/10; the only dev miss is `relax_value:eq_to_any` (D3) 7/7, MUT suspects only.

| Adjudicated MUT (panel Nemotron-3-Ultra + gpt-oss, λ = 0.7, 21 executed-truth anchors, decided 20/20 correct) | value |
|---|---|
| suspects judged / decided invalid / abstained | 3 / 3 / 0 |
| locked FR with adjudication | 56/72 (unchanged: the 3 sat on plants already caught) |
| PPI share judged valid (naive) | 0.000 (0.013), 95% 0.000–0.031 |
| cost | $0.040 |

Residuals (Mills, n_d = natural defect instances in the archive, FR of that class on the locked set; "upper" = Mills at the Wilson lower bound of FR):

| domain | defect | n_d | FR (scope) | FR Wilson | residual | upper |
|---|---|---|---|---|---|---|
| finance_ap | D2 | 1 | 4/4 | .51–1 | 0 | 1.0 |
| finance_ap | D5 | 4 | 1/1 | .21–1 | 0 | 15.4 |
| finance_ap | D6 | 1 | 0/7 | .00–.35 | ∞ | ∞ |
| finance_ap | D7 | 2 | 8/12 | .39–.86 | 1.0 | 3.1 |
| finance_ap | D9 | 8 | 4/4 | .51–1 | 0 | 7.7 |
| it_helpdesk | D1 | 5 | 1/1 | .21–1 | 0 | 19.2 |
| it_helpdesk | D3 | 6 | 1/1 (pooled) | .21–1 | 0 | 23.0 |
| it_helpdesk | D4 | 14 | 1/1 | .21–1 | 0 | 53.8 |
| it_helpdesk | D5 | 11 | 1/1 | .21–1 | 0 | 42.3 |
| it_helpdesk | D6 | 1 | 0/2 | .00–.66 | ∞ | ∞ |
| it_helpdesk | D7 | 6 | 4/4 | .51–1 | 0 | 5.8 |
| it_helpdesk | D9 | 28 | 6/6 (pooled) | .61–1 | 0 | 17.9 |

| capture–recapture (natural finds, by task) | n_A | n_B | overlap | Chapman | interval |
|---|---|---|---|---|---|
| it_helpdesk D9, MUT vs TWIN | 2 | 26 | 0 | 80.0 | 28.0–170.0 |
| it_helpdesk D1, MUT vs NULL | 4 | 3 | 2 | 5.67 | 5.0–7.7 |
| it_helpdesk D4, RED vs NULL | 14 | 2 | 2 | 14.0 | 14.0–14.0 |
| finance_ap D9, MUT vs TWIN | 1 | 8 | 1 | 8.0 | 8.0–8.0 |

## E2: loop arm vs equal-budget random-DSL control (rsi-2, 5 rounds, 4 candidates per round)

δ (R2, measured) = 0.041 loop, 0.015 control. μ (planter MMD bar) = 0.623 = 1.1 × MMD(base dev plants, 47 independent real defects).

| arm | round | F ops | P planters | dev J proxy (recall) | locked FR | locked D6 / D7 | locked FA | candidates | gate passed | accepted | cum. $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loop | 0 | 0 | 1 | 0.944 | 56/72 | 0/12, 14/18 | 0/7 | — | — | — | 0 |
| loop | 1 | 0 | 2 | 0.905 | 56/72 | 0/12, 14/18 | 0/7 | 4 | 2 | 0 | 0.004 |
| loop | 2 | 0 | 3 | 0.873 | 56/72 | 0/12, 14/18 | 0/7 | 4 | 0 | 0 | 0.007 |
| loop | 3 | 0 | 4 | 0.845 | 56/72 | 0/12, 14/18 | 0/7 | 3 | 0 | 0 | 0.011 |
| loop | 4 | 0 | 5 | 0.821 | 56/72 | 0/12, 14/18 | 0/7 | 3 | 0 | 0 | 0.014 |
| loop | 5 | 0 | 6 | 0.800 | 56/72 | 0/12, 14/18 | 0/7 | 4 | 0 | 0 | 0.018 |
| control | 1–5 | 0 | 1 | 0.944 | 56/72 | 0/12, 14/18 | 0/7 | 4 each | 0 | 0 | 0 (model-free) |

Dev J falls in the loop arm because accepted planters add plants the current F misses (the co-evolution game), not because F lost recall.

| update (loop arm) | gate | J* | J′ | J control | LCB(J′−J*) | δ | refused by |
|---|---|---|---|---|---|---|---|
| r1 `mutate_written_value_other_seen` (mutate_arg *, other_seen, D3, always) | pass | 0.907 | 1.000 | 0.907 | 0.038 | 0.041 | R2, R9 |
| r1 `mutate_written_value_sibling` | pass | 0.947 | 0.960 | 0.947 | −0.009 | 0.041 | R2, R7, R9 |
| r2–r5 (14 candidates: leak probes, value relaxers, drops) | fail: reproduces_sources + finds_on_other_domains (14), + zero_false_alarms (1) | | | | | | gate |
| control r1–r5 (20 random DSL operators) | fail: reproduces_sources + finds_on_other_domains (20), + zero_false_alarms (1) | | | | | | gate |

| planter loop (loop arm) | proposed | accepted | refused: not novel | refused: reward 0 | refused: no witness | accepted reward (miss rate) |
|---|---|---|---|---|---|---|
| r1–r5 | 15 | 5 (all `leak_decision:reason` templates with {h}: the decision in words) | 10 | 10 | 2 | 0.5 each, MMD 0.536–0.547 ≤ μ |

| meta loop (A) | A′ gate passes | A gate passes | decision |
|---|---|---|---|
| E2 r2–r5 | 0/2, 0/1, 0/2, 0/2 | 0/2, 0/2, 0/1, 0/2 | refused (R2: pass rate not higher) ×4 |
| E3-it r2, r3 | 2/2, 2/2 | 0/2, 0/2 | **accepted** ×2 |

| R10 / R5 | result |
|---|---|
| audit (rollback) r1–r5 | nothing to roll back (no F accepted) |
| prune r4 | spurious: removed two never-accepted ops from an empty library (bug, fixed) |

## E3: operators grown on one domain's evidence

| evidence | rounds | gate passed | accepted | refused by |
|---|---|---|---|---|
| it_helpdesk only | 3 (+1 rerun storing specs) | 6 + 3 | 0 | R2 ×9, R9 ×9, R7 ×5 |
| finance_ap only | 3 | 0 | 0 | gate (reproduces_sources, finds_on_other_domains) ×10 |

Post hoc (report only), the gate-passing it_helpdesk-grown operators added to the base population:

| operator (grown on it_helpdesk) | dev: ap-test | dev: desk-test | dev: finance_ap | dev: it_helpdesk | locked FR | locked FA | dev FA |
|---|---|---|---|---|---|---|---|
| base population | 20/20 | 27/31 | 49/50 | 22/24 | 56/72 | 0/7 | 0/10 |
| + `mutate_written_value_other_seen` | 20/20 | 31/31 | 50/50 | 24/24 | 56/72 | 0/7 | 0/10 |
| + `mutate_written_numeric_shift` | 20/20 | 29/31 | 49/50 | 24/24 | 56/72 | 0/7 | 0/10 |
| + `replace_written_with_sibling` | 20/20 | 29/31 | 49/50 | 24/24 | 56/72 | 0/7 | 0/10 |

## S: the lemma test (equal budget, first round only, it_helpdesk dense World, 10 cells per arm)

| arm | lemmas | built | tasks with a round-1 confirmed counterexample | spend | J proxy (clean round-1 tasks per $) |
|---|---|---|---|---|---|
| control | none | 9 | 7/9 | $0.275 | 7.26 |
| treated | 4 contract-writer + 2 proposer lemmas from rsi-1 E2 | 8 | 8/8 | $0.284 | 0.00 |
| decision | | | McNemar p = 1.0 (8 pairs) | equal (3%) | **refused**: rate did not fall; J did not rise |

## Cost

| item | $ |
|---|---|
| dense World reseed (Kimi seeder) | 0.235 |
| dense certify runs (finance_ap, it_helpdesk; 12 tasks each) | 1.502 |
| E1 adjudication (rsi-1, rsi-2) | 0.079 |
| evolve runs (smoke, rsi-1 E2, rsi-2 E2, E3 ×3, aborted ×3) | 0.063 |
| lemma test | 0.560 |
| **total (RSI work, all receipts under paths with "rsi")** | **2.44 of the 40 cap** |
