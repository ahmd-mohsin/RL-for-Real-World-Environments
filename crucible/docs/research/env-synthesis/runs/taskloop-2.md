# taskloop-2: the findings of taskloop-1 settled as policy, and TaskLoop re-run on the affected lineages (2026-10-30)

Code: DESIGN §7.7 item 25. `synth/policy/amendments.py` (ledger batch `taskloop-2`), `synth/policy/{model,resolve,ledger}.py`,
`synth/learn.py` (`settled`, `credible_interval`, `beta_ppf`), `synth/meta/rsi.py` (`TaskLoop._placed_band`, `_p3`,
`_family_knobs`, RESKILL placement, the lift hook), `synth/meta/lift.py` (new), `synth/iface.py` (`SqlEvidenceWorld`,
`sql_evidence`), `synth/intent.py` (`EvidenceContext`, the derive-effect guard), `synth/meta/reskill.py`
(`p3_checker(evidence=)`), CLI `scripts/synth_taskloop.py` (`loop --ids --lift-seed --lemma-gate-usd`, `lift`). Tests:
`tests/unit/test_synth_taskloop2.py` (new), additions to `test_synth_iface.py` and `test_synth_policy.py`.

Data (gitignored, under `crucible/data/synth/taskloop/`): `20261030-tl2-taskloop-2/` (the loop on the 10 affected
lineages), `20261030-tl2-lift-taskloop-1/` (the offline lift of taskloop-1's accepted repairs).

Models: the student Qwen3.6-35B-A3B on Tinker, only as the probe. ALT Nemotron-3-Super, gpt-oss-120b, Inkling-Small (v3);
ATK Super + gpt-oss; panels Nemotron-3-Ultra + gpt-oss; leak probe Inkling-Small; repairer DeepSeek (a builder); intent
deriver and abstractor gpt-oss (non-builder). No Qwen model in any builder, checker or judge role. No sealed or
benchmark task was read; no task was edited by hand.

Policy: every decision of the run records `policy-v28@d77cf40eeefe` (genesis v23 plus the five `taskloop-2` entries).

## 1. The rules (ledger batch `taskloop-2`, source: the lead's brief of 2026-10-30)

| seq | entry | what it decides |
|---|---|---|
| 23 | amend `R-band` → v2 | `band_undecided`: settled = confident and the 1 − α interval inside one class; escalate on the same measurement to `k_cap` 32 within `task_rollout_cap` 96, then HOLD; `require_settled`, `place_on_second_alone` false, `pool_diagnosis_into_acceptance` false are validity parameters |
| 24 | amend `R-student-probe` | `k_max` 16 → 32 (the same bound as R-band's `k_cap`; the USD cap unchanged) |
| 25 | amend `R-too-hard` | `family_knobs.finance_ap`: no lawful easier or harder knob, with the reason per knob |
| 26 | add `R-lemma-lift` | a recurring counterexample class becomes a candidate lemma; only the gate admits; underpowered defers |
| 27 | add `R-p3` | P3 unmeasured on a candidate edit is `p3_unmeasured`: executed check, else HOLD the edit; never on intent |

Each entry carries its rationale (the taskloop-1 finding it answers). The batch is data in `policy/amendments.py`,
applied once (`--amend taskloop-2`); applying it again changes nothing (test). No entry relaxes a validity parameter:
the ledger refuses a later change of the band edges, α, `require_settled`, `place_on_second_alone` or
`pool_diagnosis_into_acceptance`, from the lead too, except as a person's answer to a principle-change escalation (test).

### 1.1 Band undecided (the top finding)

taskloop-1 §B.2: 7 in-band tasks were re-labelled; on RS4's fresh band 2 stayed in band and 5 left it, and were
reserved, although only the intent card changed (the student never sees the card, so its pass rate cannot change).
Re-reading those decisions: **6 of the 7 diagnosis bands were forced** (unconfident at the final look of 16, e.g. 8/16
called "in band" with posterior mass 0.62), and the fresh bands that moved them were mostly forced too (11/16, 1/16).
The sequential test always returns a class at its last look; the loop used it as a decision.

The fix makes "decided" a property of the measurement, and the second measurement a check, not a verdict:

- **Settled** (`learn.settled`): the decided class holds posterior mass ≥ 1 − α (α = 0.1) **and** the equal-tailed
  1 − α interval of the same posterior (Jeffreys prior: the Jeffreys interval) lies inside that class. Examples: 4/4
  and 7/8 settle too easy; 4/12 to 6/16 and 7/32 to 14/32 settle in band; 0/16 does not settle too hard (its interval
  reaches 0.11), 0/32 does; 8/16 and 16/32 never settle (they straddle 0.6).
- **`band_undecided`** is raised when a measurement is not settled at k_max, or when it is settled but disagrees with
  an earlier measurement of the same task (a RESKILL's fresh band against the diagnosis band). Facts: rollouts,
  probe_budget_ok (the per-lineage index cap and the probe's cap), measurements, disagree, straddles, lineage rollouts.
- **Ladder**: `more_rollouts` on that same measurement (the escalation look 32: fresh indices; a RESKILL's escalation
  never touches the diagnosis rollouts), else `hold`. A RESKILL whose placement stays undecided keeps its label (the
  label rests on the task's evidence, RS1–RS4, not on the band) and its lineage is **held**, never kept or reserved by
  the second measurement alone. A variant's G4 band is a placement too: unsettled after escalation, the variant is
  refused as before.
- **The principles enforce it for any proposer**: P-more-evidence-first refuses an edit, a relabel or a move on an
  undecided band; P-student refuses rollouts past `k_cap`, also when a script proposes them (`resolve(extra=)`).

Cost of the stricter rule, stated before the run: an in-band task at p ≈ 0.25 settles at 12–32 rollouts; one at the
upper edge (p ≈ 0.5–0.65) does not settle within 32 and is held; a too-hard task settles only at 0/32.

### 1.2 Lemma lift

`lift_lemmas` was a hook nothing filled. `meta/lift.LemmaLift` is wired into the loop and runs offline on an earlier run
(`synth_taskloop.py lift --run`). Steps: record every accepted repair as a (counterexample class, repair) pair; a class
that ≥ 2 lineages repaired becomes a candidate; the abstractor (gpt-oss, a non-builder family; it sees the scrubbed
cluster and the defect taxonomy, never a task id) drafts the rule; the deterministic leakage critic refuses a rule naming
an id, entity or value of its evidence; the candidate raises `lemma_candidate` and the policy decides whether the gate
runs.

**The gate and its power (rsi-stats §9.4″).** The existing lemma gate is `meta.lemmas.lemma_acceptance`: the same cells
built with and without the lemma at equal spend, an exact one-sided McNemar test on the first-round
confirmed-counterexample rate at the run's α_t, and J rising under R1–R10. At α_t = 0.005 (t_max 10, prereg §5), a
target gain of 0.2 and power 0.8 it needs **50 paired cells** (`stats.plants_needed`); at USD 0.08 a cell per arm that is
USD 8.0, more than this work's whole cap. The gate is therefore not run: the policy's remedy is `no_update`, the
candidate is **deferred**, and the bank does not change. Admission without the gate is impossible by construction
(`admit_only_through_gate`, a validity parameter).

### 1.3 P3 on SQL Worlds

`EVID` (evidence removal) needs a World that can edit its start state. finance_ap's World is a Crucible World over
SQLite, and the adapter has no `perturb`, so EVID said "not applicable" on every finance_ap task and P3 was unmeasured
whenever SWAP and TWIN had no trial (taskloop-1: 28 ASCEND and 25 repair refusals cite G2).

**The extractor.** `iface.SqlEvidenceWorld` edits one field of one row of a `sql` App through the App's own guarded SQL
executor after the reset (the next reset restores it), and only `set`: a clone of an AP row would create a policy fact
(a duplicate, a vendor's history) and a shuffle has no meaning for an unordered table; key columns, unknown rows or
columns and non-SQL Apps are refused, and the edit is re-read. It is a wrapper used only by P3
(`intent.EvidenceContext`, `p3_checker(evidence=True)`), so certification, and PERT in particular, does not change on SQL
Worlds.

**A soundness fix found by it.** With the channel on, EVID took `approved` on an intent card (written by
`approve_invoice`, no argument carries it) as evidence in an older invoice's `status` and would have confirmed the task
off intent (π* still passes when that unrelated row is blanked). `evidence_sites` now skips a derive value that π*'s own
write produces (test: with the guard off, the coincidence is a site).

**Measured (CPU, no spend), the pool's 86 valid finance_ap tasks:**

| | P3 on intent | P3 unmeasured | EVID applicable | EVID valid trials |
|---|---|---|---|---|
| plain context | 72 | 14 | 0 | 0 |
| with the evidence channel | 73 | 13 | 3 | 2 (both clean) |

83 of 86 have "no derived value with an evidence site": the decisions (status and reason code) are derived by the
policy's rules from several fields (bank verification, PO and receipt matching, history), and π* writes the code as a
literal it read nowhere. EVID's model (a derived value copied from one observation, removed at its row) has no site
there; extending it to rule-derived decisions would need the policy truth to recompute decisions from the edited rows,
which the pack's truth (drawn from the generator) does not do. So the extractor is sound and wired, but it measures P3
on few finance_ap tasks. The unmeasured case is the named condition `p3_unmeasured` (R-p3): `executed_check` when the
World offers another executed P3 probe, else HOLD the edit, recorded on the decision with its rule. Unmeasured never
counts as on intent (a validity parameter).

### 1.4 DESCEND and ASCEND on finance_ap

No skill-preserving knob is both sound and effective on finance_ap, so R-too-hard records `family_knobs.finance_ap =
{easier: [], harder: []}` with the reason per knob:

| knob | why it is not lawful on finance_ap |
|---|---|
| distractors | a cloned AP row is an input of a policy rule (duplicates, vendor history, PO and receipt matching): the variant's reference decision would no longer be the policy truth's. Unsound |
| paginate | applies, but taskloop-1 measured 0 of 27 lineages moved (20 still too easy on 16 fresh rollouts) and P3 unmeasured on the variants; a named-invoice decision finds the invoice by number |
| densify superset | sound (every base row and decision kept), but not a difficulty knob: more pending invoices on new vendors do not change what the student must do |
| actor, derive | need a writer and a second actor; none is wired for finance_ap, and the decision values are already derived |
| fewer_distractors, unpaginate, undo | need a variant World with clones, pages or a recorded harder knob, which finance_ap never gets |
| derive → given | never an easier move (changes the skill) |

The loop filters a task's declared knobs by its family before resolving, so a finance_ap task out of band goes
RESKILL → RESERVE without applying a knob or re-probing (test: no knob is applied).

## 2. The re-run on the affected lineages

`synth_taskloop.py loop --manifest manifest.pool-2b.pool.json --ids <the 10> --cap 5 --student-cap 3 --rounds 3
--concurrency 6 --lift-seed <taskloop-1>` (`taskloop/20261030-tl2-taskloop-2`), the same pool manifest, models, panels
and stored diagnosis rules as taskloop-1; only the ten lineages the findings touch: the 7 in-band tasks taskloop-1
re-labelled, and the 3 rejected tasks whose accepted repairs were the lemma candidates. A fresh run directory, so every
lineage starts again from its pool-2b version (new seeds by the run's salt: a re-derived label, a re-drawn repair).

### 2.1 Decisions

| task | taskloop-1 | taskloop-2 | why (taskloop-2) |
|---|---|---|---|
| `a0/finance_ap.s3.k0-b0-1` | RESKILL, fresh 11/16 "too easy": reserved | **HOLD** | diagnosis 18/32: unconfident (escalated 16 → 32), straddles 0.6 |
| `a1/finance_ap.s0.k2-b0-2` | RESKILL, fresh 1/16 "too hard": reserved | **HOLD** | diagnosis 2/32: unconfident, straddles 0.1 |
| `a0/finance_ap.s3.k1-b0-2` | RESKILL, fresh 1/16 "too hard": reserved | **HOLD** | diagnosis 5/32: unconfident, straddles 0.1 |
| `yield-1/a0/finance_ap.s1.k2-b0-2` | RESKILL, fresh 11/16 "too easy": reserved | **HOLD** | diagnosis 19/32: straddles 0.6 |
| `yield-2/a1/finance_ap.s2.k2-b0-2` | RESKILL, fresh 4/12 in band: kept | **HOLD** | diagnosis 19/32: straddles 0.6 (taskloop-1 kept it on a fresh 4/12 after a forced 8/16 diagnosis) |
| `a1/finance_ap.s1.k3-b0-2` | RESKILL, fresh 1/16 "too hard": reserved | **RESKILL, held** | diagnosis settled in band at 10/32; the re-labelled task's fresh band 6/32 is in band too, but straddles 0.1 at k_cap: label kept (→ `three_way_match`), lineage held |
| `finance_ap/20261029-152544-full/t20` | RESKILL, kept | **KEEP** | diagnosis settled in band at 4/16 (8 stored rollouts + 8); the re-derived RESKILL refused by `p3_unmeasured` (R-p3: hold the edit), so kept under its label |
| `a1/finance_ap.s3.k0-b0-0` | REPAIR; round 2 RESKILL, reserved | **REPAIR; round 2 RESERVE** | repaired again from its 22 shown D9 counterexamples; round 2 8/8 too easy, RESKILL refused (RS5 trust region), no finance_ap knob: reserved at once |
| `a1_lemmas/finance_ap.s3.k2-b0-0` | REPAIR; rounds 2–3 ASCEND refused twice, reserved | **REPAIR; round 2 RESERVE** | the same, RS5's per-round cap; no paginate attempt (taskloop-1 spent two rounds of fresh rollouts on it) |
| `yield-1/a1/finance_ap.s2.k2-b0-2` | REPAIR; rounds 2–3 ASCEND refused twice, reserved | **REPAIR refused, held** | the re-drawn repair candidate had P3 unmeasured: `p3_unmeasured` → R-p3 hold, recorded (taskloop-1's accepted repair had a measured P3); invalid, so held |

**Before and after (the 10 lineages):** kept 2 → 1, reserved 8 → 2, held 0 → 7 (rejected 0 → 0).
**The pool** (taskloop-1's final lineages with these 10 replaced): kept 19 → 18, reserved 71 → 65, held 85 → 92,
rejected 20 → 20 (195).

What the numbers say: of the 7 re-labelled in-band tasks, none is now placed by a second measurement alone. Five never
reach RESKILL: their own diagnosis, escalated to 32 rollouts, cannot say which side of a band edge they are (18/32,
19/32 ×2 at the 0.6 edge; 2/32 and 5/32 at the 0.1 edge). These are the tasks taskloop-1's forced looks moved in or out
of the band. The one that reached the fresh band (`a1/…s1.k3-b0-2`) is the clearest case of the finding: taskloop-1
reserved it as "too hard" on a fresh 1/16 against a diagnosis 7/16; at 32 each, both measurements say in band (10/32,
6/32), and the second is held only because its interval touches 0.1. One task is kept (t20, settled). The two
lineages kept in taskloop-1 that are now held or re-decided did not lose a property: their band was never decided.

### 2.2 Conditions the policy resolved (all under `policy-v28@d77cf40eeefe`)

| condition | rule | action | count |
|---|---|---|---|
| `band_undecided` | R-band | more_rollouts | 7 (6 diagnoses 16 → 32, 1 RESKILL band 16 → 32) |
| `band_undecided` | R-band | hold | 6 (5 diagnoses, 1 RESKILL band) |
| `p3_unmeasured` | R-p3 | hold | 2 (a RESKILL card, a repair candidate) |
| `band_too_easy` | R-too-hard | reskill, then reserve (no lawful knob: `family_knobs`) | 2 |
| `confirmed_counterexample` | R-admission-fixed | repair_with_counterexamples | 3 |
| `lemma_candidate` | R-lemma-lift | no_update (deferred) | the D9 candidate, each time a repair touched it |

No `band_undecided` disagreement fired in this run (the one RESKILL band agreed in class); the disagreement path is
covered by tests (`test_synth_taskloop2.py`). No diagnosis rollout entered an acceptance band (indices disjoint on every
decision). The drift monitor has 2 re-probed repaired tasks (it needs 5).

### 2.3 The lemma lift

Offline on taskloop-1 (`taskloop/20261030-tl2-lift-taskloop-1`, USD 0.0004) and seeded into the run: the 3 accepted
repairs are 3 pairs; their counterexamples share one class, **D9** (MUT SwapEntity and TWIN: the contract does not pin
the entity the instruction names), on 3 lineages, so D9 is one candidate. The abstractor (gpt-oss) wrote, from the
scrubbed cluster:

> When an instruction refers to a specific entity, the contract's required patterns must explicitly bind that entity
> (e.g., by matching its unique identifier) rather than only checking a subset of its fields; otherwise the contract may
> accept actions on a different entity. (applies to: contract writer)

The leakage critic found nothing (no id, entity or value of the evidence). The gate needs 50 paired cells (α_t 0.005,
gain 0.2, power 0.8); the gate budget is 0, so `units_available` 0 < 50 and R-lemma-lift's remedy is `no_update`: the
candidate is **deferred** and the lemma bank is unchanged (v1, 0 lemmas). The two repairs re-accepted in the run are
the same (task, version) pairs, so the candidate did not grow. To admit it, the gate needs an equal-budget paired build
of ≥ 50 cells per arm (about USD 8 at the yield-1 cost per cell), which is beyond this work's cap: a decision of budget,
not of rule.

### 2.4 Cost and time

| item | USD |
|---|---|
| student rollouts (2,851 calls to Qwen3.6-35B-A3B) | 1.293 |
| swarm (ALT 0.122, panel calibration 0.098, outcome 0.035, adjudication 0.023, repairer 0.016, ATK 0.014, deriver and abstractor 0.005, LEAK 0.001) | 0.314 |
| **the loop** | **1.607** |
| offline lift of taskloop-1 (abstractor) | 0.0004 |
| **this work** | **1.61** of the 6.00 cap |

The shared Part 2 total was USD 115.49 when the loop started (other runs share it) and 117.78 after. Wall-clock: 10.5
minutes for the loop (round 1: 6.6 min for 10 tasks); the P3 pool measurement and the dry runs are CPU only.

## 3. What does not work yet (honest list)

- **The stricter band holds more tasks.** On this student many finance_ap tasks sit near p = 0.6 or below 0.1, where 32
  rollouts cannot settle the band at α = 0.1. They are held, not placed. Settling them needs more rollouts than
  `k_cap` (a rule change through the ledger, with its cost) or a stronger student checkpoint.
- **P3 stays unmeasured on most finance_ap decisions.** The SQL evidence channel is sound but EVID's value-copy model
  has no site for rule-derived decisions (83 of 86). An on-intent probe for them needs the policy truth to recompute a
  decision from edited rows; the pack's truth is computed from the generator. Until then `p3_unmeasured` holds such
  edits (2 in this run, including one repair that taskloop-1 had accepted under a measured P3).
- **The lemma gate is unaffordable at this cap** (50 paired cells per arm). The D9 lemma waits as a deferred candidate.
- **finance_ap has no difficulty ladder.** With no lawful knob, out-of-band finance_ap tasks are reserved; a ladder
  needs knobs built into the pack (for example a second actor or a harder derived value), a design task, not a loop fix.

