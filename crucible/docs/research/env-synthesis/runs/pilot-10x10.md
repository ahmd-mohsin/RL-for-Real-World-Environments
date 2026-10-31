# pilot-10x10: 10 World instances × 10 tasks, end to end, with quality and generation speed (2026-10-30/31)

Code: DESIGN §7.7 item 26. Driver and report `scripts/synth_pilot.py` (budget, ladder, manifest, report); the stages are the
framework's own: `synth.py batch` (`--work pilot-10x10 --arms a1v3 --cells 10`), `synth_taskloop.py loop` (`--work`,
`--calibrate-per family`), `synth.solvers.escalate_alt` through `synth_pilot.py ladder`. Tests: `tests/unit/test_synth_pilot.py`
(new), changes in `test_synth_roles.py` and `test_synth_policy.py`.

Data (gitignored), under `crucible/data/synth/pilot-10x10/`: `_budget.json` (the stage funded through the policy), `p1/` and
`p2/` (two batch starts stopped by framework bugs, kept with their receipts), **`p3/`** (the pilot: `a1v3/` the batch arm,
`ladder/`, `manifest.pilot.pool.json`, `items.json`, `archives/`, `loop/` and `loop2/` the per-task loop, `loop-dry/` the
wiring check, `budget-run-stages.jsonl` every per-stage budget decision, `pilot_report.json` the numbers below), and the
logs `p1.log`, `p2.log`, `p3.log`, `p3-ladder.log`, `p3-loop.log`, `p3-loop2.log`, `p3-loop-dry.log`.

Models. Swarm only for every builder, checker and judge role: proposer, oracle programmer, repairer DeepSeek-V4.1-Flash;
contract writer GLM-5.3; the World data and code (seeds) Kimi-K2.6 (worlds-1/2); ALT R-ALT-whole's L0 certifying set
(gpt-oss-120b, Nemotron-3-Super, Inkling-Small, harness v3), the ladder's L1/L2 (Nemotron-3-Ultra v2, Inkling v3); ATK
Super + gpt-oss; panels Nemotron-3-Ultra + gpt-oss; leak probe Inkling-Small; intent deriver and abstractor gpt-oss. The
student Qwen3.6-35B-A3B ran only as the loop's probe; its reward entered no J. Claude wrote no task and no World content.
No sealed data was opened. Every decision records `policy-v30@cb93cf57a9c1`.

## Headline

| measure | result |
|---|---|
| attempted | **100** proposals, 10 per World, 10 World instances of 4 families |
| valid (oracle executed, contract written) | 54 |
| certificates | **0 certified, 1 provisional**, 10 on hold, 43 rejected; 26 filtered before the oracle, 20 dead oracles |
| after the per-task loop | **kept 0**, reserved 1 (too easy: 8/8), held 50, rejected 3 |
| perfect tasks (valid, in band, on intent, novel) | **0** |
| tasks in the band 0.1-0.6 | 0 (the one valid task is too easy; no held or rejected task is probed for a band) |
| wall-clock, batch (active) | 100 attempted in **25.7 min**; first admitted (provisional) task at **19.5 min**; no certified task; the batch 36.4 min |
| throughput, batch | 165 attempted per hour, 89 valid per hour, 1.65 admitted per hour, 0 certified per hour |
| wall-clock, rest | ladder about 3 min (2 tasks); loop round 1 11 min for 28 of 54 lineages (stopped: student cap), the re-run 18 min for 33 |
| USD | **19.11** of the 24 cap: p3 17.02 (batch 10.33 with 0.40 booked for stopped calls, ladder 0.50, loop student 5.21, loop swarm 0.98), p1 1.33 and p2 0.75 (starts stopped by framework bugs) |
| USD per attempted / per admitted / per certified / per perfect | 0.17 / 17.02 / none / none |
| shared Part 2 total | USD 118.05 before, **137.16 after** (of 150; the hard stop 148 never came close) |

The answer to the lead's question ("10 different envs with 10 different tasks, and how fast"): the framework attempts 100
tasks over 10 family Worlds in under half an hour of wall clock for about USD 10 (USD 0.10 per attempt in the batch), but
on these Worlds it certifies none of them, and the one admitted task is too easy for the student. The bottlenecks are
the ALT solvers (no non-builder solver passes 37 of 54 valid tasks), contracts that pin free text and World-generated
values (TAINT D5 on 33), and adjudication panels that cannot be calibrated on so few anchors (suspects stay open on 10).
Eighteen framework causes were found and fixed on the way (§6); the ones that set yield are in §7.

## 1. The Worlds

R-scale: 10 environments from ≥ 4 families. Chosen from the 21 usable instances (worlds-2b), spread over the four families
and the instance profiles:

| World (domain) | family | instance | thin tools (density gate) |
|---|---|---|---|
| insurance-i00 / i01 / i02 | insurance claims intake | standard / dense / noisy | 5 / 4 / 4 of 8 write tools |
| ar-i00 / i01 / i02 | AR collections | standard / dense / noisy | 4 / 3 / 5 of 9 |
| leave-i00 / i02 | leave and absence | standard / noisy | 7 of 9 |
| pr-i01 / pr-i06 | purchase requisition | dense / noisy | 6 of 10 |

finance_ap was not needed (every family has ≥ 2 usable instances). Cells: 10 per World from the instance's DomainSpec
(`InstanceDomain(n_cells=10)`, skills whose tools are all thin left out), one proposal per cell (`--quota 1
--max-per-cell 1 --batch 1`), so the 10 tasks of a World are 10 different cells.

## 2. The funnel per World

| World | proposed | filtered | dead oracle | valid | certified | provisional | on hold | rejected | loop: kept / reserved / held / rejected |
|---|---|---|---|---|---|---|---|---|---|
| ar-i00 | 10 | 4 | 2 | 4 | 0 | 0 | 1 | 3 | 0 / 0 / 3 / 1 |
| ar-i01 | 10 | 2 | 0 | 8 | 0 | 0 | 2 | 6 | 0 / 0 / 7 / 1 |
| ar-i02 | 10 | 5 | 0 | 5 | 0 | 1 | 0 | 4 | 0 / 1 / 4 / 0 |
| insurance-i00 | 10 | 3 | 3 | 4 | 0 | 0 | 2 | 2 | 0 / 0 / 4 / 0 |
| insurance-i01 | 10 | 2 | 2 | 6 | 0 | 0 | 1 | 5 | 0 / 0 / 6 / 0 |
| insurance-i02 | 10 | 1 | 3 | 6 | 0 | 0 | 0 | 6 | 0 / 0 / 5 / 1 |
| leave-i00 | 10 | 3 | 1 | 6 | 0 | 0 | 0 | 6 | 0 / 0 / 6 / 0 |
| leave-i02 | 10 | 2 | 2 | 6 | 0 | 0 | 2 | 4 | 0 / 0 / 6 / 0 |
| pr-i01 | 10 | 0 | 3 | 7 | 0 | 0 | 1 | 6 | 0 / 0 / 7 / 0 |
| pr-i06 | 10 | 4 | 4 | 2 | 0 | 0 | 1 | 1 | 0 / 0 / 2 / 0 |
| **all** | **100** | **26** | **20** | **54** | **0** | **1** | **10** | **43** | **0 / 1 / 50 / 3** |

On hold = no confirmed counterexample and no failed rule, only suspects an uncalibrated panel could not decide (R-panel's
human queue; the recert statuses). It counts the 2 tasks the solver ladder passed (§4). Band: the one valid task is too
easy (8/8 intended passes); no other task reaches the band, because the loop measures a band only on a valid version.

Rejection reasons (a task can have several): ALT:no_pass 37, TAINT:confirmed 33, MUT:upper 29, TWIN suspects 24, ALT
suspects 23, TWIN:confirmed 14, MUT-HO suspects 12, MUT suspects 11, MUT:confirmed 7, ATK 9 (4 confirmed), RED 3, PERT 3,
NULL 2, archive regressions re-opened 41. Filters: 31 leak hits on 26 proposals (a derive value the instruction states),
3 unknown entities. Dead oracles: 20 after 4 attempts each.

## 3. Speed

Active wall clock of the batch (p3) is 2,182 s; 664 s of downtime (six stops to fix framework bugs, gaps over 3 minutes
with no candidate event) are left out. Busy time per stage, summed over candidates:

| stage | candidates | busy s | median s per candidate | per hour over the stage's span |
|---|---|---|---|---|
| propose | 100 | 603 | 5.6 | 183 |
| build (oracle executed, screen, contract) | 74 | 2,914 | 27.3 | 133 |
| refine (falsifiers, ALT/ATK, adjudication, CEGIS repair, certificate) | 54 | 8,218 | 121.8 | 71 |

Serial-equivalent 11,734 s over 2,182 s: 5.4× (8 build and 8 refine workers, 6 falsifier processes, 32 model calls in
flight). Refine is the bound, as in yield-1.

## 4. USD per stage

| stage | USD | share |
|---|---|---|
| propose | 0.56 | 3% |
| oracle | 3.09 | 18% |
| contract | 1.45 | 9% |
| ALT | 1.46 | 9% |
| ATK | 0.20 | 1% |
| adjudicate (panels, calibration) | 1.68 | 10% |
| repair (batch CEGIS) | 1.50 | 9% |
| stopped calls, booked by hand | 0.40 | 2% |
| ladder (R-ALT-whole L1/L2) | 0.50 | 3% |
| loop: student probes | 5.21 | 31% |
| loop: swarm (repairs, re-certifications, panels) | 0.98 | 6% |
| **p3** | **17.02** | |

The oracle stage is the costliest builder stage: 20 proposals died after four programmer attempts on 17-21k-token prompts.
The student probes are 31%: the loop draws its 8 diagnosis rollouts before the certificate check (§7.4), so 488 rollouts
went mostly to tasks whose certificate then decided them.

## 5. What the probes found, and what the loop did

**The loop** (`loop`: all 54 lineages, stopped after 28 decisions when the student cap bound; `loop2`: the 33 lineages the
two loop fixes affect or that `loop` never reached, under per-family calibration). Order as the policy resolves it:
ORACLE → P1 (the certificate) → REPAIR → band → RESKILL / DESCEND / ASCEND → KEEP / RESERVE; HOLD where evidence does not
decide.

| move | decisions | accepted |
|---|---|---|
| REPAIR (P1, from the shown counterexamples) | 42 | 0 |
| HOLD | 17 | — |
| RESERVE | 2 | — |

| condition | rule | action | count |
|---|---|---|---|
| `confirmed_counterexample` | R-admission-fixed | repair_with_counterexamples | 42 |
| `confirmed_counterexample` | R-admission-fixed | reject (no counterexample a repairer may see) | 3 |
| `open_suspects` | R-panel | human_queue (uncalibrated) | 13 |
| `open_suspects` | R-panel | **panel_decide** (calibrated, the fix of §6 #16) | 4 |
| `band_too_easy` | R-too-hard | reskill (the derivation confirms the label), then ascend: no declared harder knob applies, reserve | 2 |

Repairs refused by: T4 74 (the unintended fail the repair had to close still fails on fresh rollouts), T5 32 (the
candidate's re-certification rejects it), G2 16 (P3 probes: TWIN or TAINT on the candidate), the repairer declined 10, T2 4
(a repair that would have made a skill-gap fail pass: refused). No repair, RESKILL or variant was accepted. The R-panel
`panel_decide` on the AR family's calibrated panel re-certified 3 on-hold versions: 2 rejected (the panel judged the open
suspects real defects), 1 stays on hold (a suspect it could not decide at λ 0.7; its second condition is the fourth
`panel_decide` row). The drift alarm needs ≥ 5 re-probed repaired tasks; there were none.

**Probe outcome classes** (488 diagnosis rollouts over 61 decisions): intended pass 115, intended fail 118, unattributed
fail 85, unintended fail 134, suspect 36. Labels the outcome classifier recorded in `loop2` (264 rollouts): intended pass
80, intended fail 43, unintended fail 68, uncertain 73.

**Unintended fails** are all hidden information: the student fails a pattern whose constant TAINT reports unsourced (D5):
an exact note text the contract pins ("Reported 71 days after loss date; exceeds 60-day late-report review threshold. Refer
to Coverage Analyst before opening claim."), a section title, a reason code the documents never print
(`policy_not_active_on_loss_date`), and values the World writes (`decided`, the acting user `u-101`) inside the same
patterns. The loop turned each into a PROBE counterexample and asked the repairer; T4 refused the repairs that left them
failing. **Unintended passes**: 36 rollouts were labelled unintended passes with no executed evidence (T1 demotes them:
never a defect, never repaired). No unintended pass had executed evidence.

## 6. Framework issues the real runs found, and their fixes (each with a test in `test_synth_pilot.py` unless named)

| # | found by | issue | fix |
|---|---|---|---|
| 1 | p1, the first 15 insurance builds (5 dead oracles) | the proposer and the oracle programmer saw `compact(slice, 24000)` of a 56-103k-character slice: the first 25-30% of the tables in key order. The policy documents, `policy_params` and most of LossNotice were never shown; the oracle programmer never saw the rules it had to apply | `families.family_slice`: the kit tables (documents, policy parameters, colleagues) whole, every other table an equal share of the rest, the rest counted; within the limit by construction; `DomainEnv.slice_chars` (family Worlds 48,000) used by the proposer and oracle calls |
| 2 | the cell grid | `InstanceDomain` drew 2 × n cells, then dropped skills whose tools are all thin: a PR instance got 6 of 10 cells | the draw grows with the skill count; repeated texts count once |
| 3 | p2 oracle notes (2 of 8 dead) | a literal list or object holding a `$ref` was refused at parse time; DeepSeek wrote `concat([..., {"$ref"}])` four times per task | refs inside literal lists and objects are resolved (`list(...)`, new `obj(key, value, ...)` with structural keys); a container without refs stays a literal (`test_synth_roles.py` updated) |
| 4 | p1-p2 filters (14 of 48) | the proposer listed in `derive` values its own instruction states (an entity it names, a filter) | PROPOSER_V2: derive holds only what the agent must work out; describe only records in the shown state |
| 5 | reading the batch against R-ALT-whole | the batch certified with Ultra × 2 + gpt-oss (v2), the loop re-certified with the L0 set (v3), and no code mapped the policy's ladder to models, so the ladder was never wired | `synth_wire.LADDER_MODELS`, `ladder_models(rung)`, `ladder_configs(wire)`; arm preset `a1v3` (builders v2, ALT = L0 on v3: `Arm/RunConfig.alt_variant`); `synth_pilot.py ladder` runs `escalate_alt` from L1 through `policy.resolve(no_solver_pass)` and re-decides with `with_ladder` |
| 6 | p3, the oracle stage cap | a stage cap that bound marked every queued candidate `budget`/done; a resumed batch never built or refined them | `budget` candidates are re-opened on resume (built ones refine under a fresh attempt id) |
| 7 | the same | a batch's own per-stage cap had no lawful remedy | `synth_pilot.py budget --run-stage`: `resolve(over_budget)` one level down (the enclosing total is the arm's cap); 5 decisions in `budget-run-stages.jsonl` (oracle 1.5 → 2.99 → 3.86, contract trim then 1.2 → 1.72, repair 1.5 → 1.77) |
| 8 | p3 calibrations (insurance 8/9, 8/10, 8/10 correct) | R-panel's `add_calibration_anchors` was not implemented: the batch calibrated once on 4 tasks | `BatchConfig.calib_max_tasks` (12): uncalibrated → `resolve(uncalibrated_panel)` → more anchors while built tasks can come, then abstain; rows in `calibration_remedies.jsonl` (6 in p3: 3 more-anchor rounds, 3 human queue). 5 of 10 domain panels calibrated in the end |
| 9 | the same, on resume | with no anchor at all, `calibrate_panel` returned a panel at the default λ 0.7: uncalibrated, it decided | no anchor → λ above 1 (abstains) under the policy rule |
| 10 | the first restart with fix 8 | calibration readiness used the global "proposing done": a finished domain's refines waited while other domains proposed | per-domain readiness (no pending build, no open cell) |
| 11 | the second restart | a domain whose tasks were built before a stop resumed straight into refine with no anchor; refine waited for calibration, the proposer for the backlog: a deadlock | on resume, anchors come from the stored tasks (their oracles executed again, CPU) for every domain not yet calibrated |
| 12 | p3 transcripts (32 of 751 ALT turns on the first 23 tasks) | a v3 solver sending its plan as notes-only turns got the generic "neither a call nor a finish" note | notes-only turns are told the notes are kept; `max_malformed` (4) ends a solver attempt that never calls (ALT, ATK and the ladder only; the student probe keeps `None`, so its rollouts are unchanged) |
| 13 | p3 TAINT findings (10 literal dates) | the oracle DSL had no date arithmetic: a rule-derived date ("5 days after the promise") could only be a literal, which TAINT rightly calls unsourced | `date_add(date, days)` and `days_between(a, b)` in `program.LIBRARY` and `PROGRAM_DOC` (applies to builds after it; not re-run) |
| 14 | the funnel | the pipeline's certificate says "rejected" for open suspects only | `recert.stored_rows` and the report hold them (`on_hold`, the recert statuses) |
| 15 | the loop's first round (student cap spent, every panel abstained) | the loop calibrated per World instance on certified and provisional tasks only: with 1 such task over 10 instances, no panel calibrated | `synth_taskloop.py loop --calibrate-per family` (default) and `--anchor-statuses no-confirmed` (certified, provisional, on hold): AR's family panel calibrated (λ 0.7, 10 anchors); insurance (8), leave and PR (4 each) did not |
| 16 | the same | `TaskLoop` raised `open_suspects` with `panel_calibrated` hard-coded False: a calibrated panel never decided an on-hold task | `TaskLoop(panel_calibrated=, readjudicate=)`; R-panel's `panel_decide` drops the stored status and re-certifies with the calibrated panels (`Recertifier.readjudicate`); 4 decisions in `loop2` |
| 17 | wiring for the pilot | stored tasks reached the loop only through a paid re-certification; the taskloop report read PROBE records of finance_ap and it_helpdesk only; the loop and the batch had no stage cap of their own | `recert.stored_rows`, `world_builders`, `synth_pilot.py manifest`; the report reads every domain; `--work` on `synth.py batch` and `synth_taskloop.py loop` (the ledger's stage cap) |
| 18 | the full synth suite | `test_r_budget_caps_...` built a fresh ledger but read the shipped policy's stage caps, so any later stage-cap entry broke it | the test reads its own ledger (`test_synth_policy.py`) |

Measured and reverted: a TAINT free-text test that keeps only a note's sourced words. On the 33 stored final versions it
changed no verdict (22 → 22): the D5 notes there are exact sentences the contracts demand, so the finding is real.

## 7. Open, and what sets the yield (for the lead)

1. **ALT on family Worlds.** No L0 solver passes 37 of 54 valid tasks; of the 74 failed attempts the panel judged, both
   members called 49 invalid, 13 valid (the contract rejected a valid attempt: D6 evidence) and 12 split. The ladder passed both
   tasks it was allowed (L1), so stronger solvers help, but R-ALT-whole only climbs for tasks whose sole failed rule is
   ALT:no_pass (2 here). Most tasks also fail on TAINT or TWIN.
2. **Contracts over-pin free text and World-written values.** TAINT confirmed on 33 valid tasks. Over the archived TAINT
   counterexamples of the 43 tasks' latest attempts: free text the contract checks 39, contract constants the agent cannot
   source 20, a document id read without listing 15, D7 (a derive value in the instruction) 13, a page size 11, a
   rule-derived date 10 (fix 13). The same constants are the student's unintended
   fails and part of the D6 attempts. The contract writer's v2 rule (a regex keeping the facts) is not followed reliably
   by GLM on notes; the repairer did not converge (T4 74, T5 32).
3. **Panels.** At 9-19 anchors per instance 5 of 10 batch panels calibrate (target 0.9, some only at λ 0.95); per family,
   1 of 4 in the loop. Uncalibrated, every MUT/ALT/TWIN suspect stays open (R-panel), so a task needs zero suspects to be
   certified. More anchors need more tasks per family; a family pool of 100 tasks is the remedy the policy already names.
4. **The loop probes before P1.** 488 rollouts (USD ~4) went to on-hold and rejected versions. The rollouts feed ORACLE's
   dispute check and the PROBE counterexamples of P1 repairs, so the order is by design (§9.1's code differs from its
   pseudo-code, which checks P1 first); whether to probe an invalid version is a cost-versus-evidence choice for the
   lead (an inferred rule could order it; it is not changed here).
5. **No difficulty knob on family Worlds.** The one valid task is too easy and no declared harder knob (distractors,
   paginate) applies to these Worlds, so it is reserved. `family_knobs` has an entry for finance_ap only.
6. **Throughput is fine; quality is the gap.** USD 0.10 per attempt and 165 attempts per hour are within yield-1's range;
   the family Worlds' tasks fail the existing, unchanged validity rules. No rule was relaxed.

## 8. Budget through the policy

- The pilot's stage: `synth_pilot.py budget --plan 24 --apply` → `resolve(over_budget)` → `reallocate_within_total`
  (R-budget; stage remaining 0, Part 2 remaining 31.95) → ledger seq 29, `stage_caps_usd["pilot-10x10"] = 24`
  (`policy-v30@cb93cf57a9c1`, inferred).
- The batch's own stage caps: five decisions (fix 7), all inside the arm's 12 hard cap; the arm's cap never bound.
- Every paid run checked the shared Part 2 total (`MILESTONE_CAP_USD=150`) and the stage cap before it started.
- Calls cut off by the six stops were booked by hand as `unbooked_estimate` (USD 0.40 in p3, 0.20 in p1, 0.10 in p2, 0.05
  in the loop), so the caps stayed honest.
