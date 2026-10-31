# WORLDS-2: the three failed World families rebuilt, and their causes fixed in the builder (2026-10-30)

Data: `crucible/data/synth/worlds/` (gitignored). New this run: `_worlds2-build{1..4}*.log` (the build runs),
`_worlds2-density-recheck.json` (the 12 worlds-1 usable instances under the shown bar), `_worlds2-regate.log`,
`_worlds2-panel-ins.log`, `_worlds2-calibrate.log`, `realism_calibration.json` (with rejected anchors) beside
`realism_calibration.worlds1-m4rule.json` and `realism_calibration.worlds2-rescored.json`, `<family>/app/datagen_density.jsonl`,
`insurance_claims_intake/instances.worlds1-20261030/` (the old instances, moved aside), `gates.reseed<k>.json` (records
kept when R-density reseeded an instance). Code: DESIGN.md §7.7 item 16 ("worlds-2"), wiring of item 24.

Swarm only: Kimi-K2.6 (design, code, debug, data generator), GLM-5.3 (schema, documents), DeepSeek-V4.1-Flash (blind
tests, colleagues), Nemotron-3-Ultra (test approver; realism panel with gpt-oss-120b). No Qwen model in any role; the
student was never called. All model content (designs, code, tests, generators, documents) was written by the swarm.

## Headline

| | |
|---|---|
| usable instances (all 7 gates) | **18**: insurance claims 6/6 (new), AR collections 6/6, leave and absence 6/6 |
| families with usable instances | **3** (the lead's target of ≥ 4 families is **not met**) |
| returns and refunds | redesigned (list tools for customers, refunds, policy exceptions); code accepted; tests, debug, data, docs pending |
| purchase requisition | redesigned (list tools for purchase orders, invoices, goods receipts); debug passed; data generator stopped at its cap |
| cost | **USD 5.56 of the 6 cap** (Kimi 4.81, panel 0.24, GLM 0.24, DeepSeek 0.22, Nemotron approver 0.05) |
| shared Part 2 total after this run | USD 106.15 of 150 (includes the parallel agent's spend) |

The count target (≥ 10 usable) is met with 18; the diversity target is not. Finishing returns would take about
USD 1 (tests, debug, data, docs) and purchase requisition about USD 0.5 (data, docs), at worlds-1's per-stage costs.

## Per family

| family | this run's USD | wall-clock | outcome | gates (pass / instances) | density reached (shown / held-out, per collection) |
|---|---|---|---|---|---|
| insurance claims intake | 2.13 build + 0.11 panel | 54 min (2902 s + 364 s) | frozen | build, determinism, integrity, surface, obscurity, density, panel: 6/6 each | Claim 71-101 / 11-36, LossNotice 80-101 / 11-32 |
| returns and refunds | 1.18 | 41 min (1265 s + 1183 s) | code accepted on re-check; tests pending | none (not frozen) | not measured |
| purchase requisition to order | 2.00 | 54 min (2888 s + 329 s) | data generator stopped at its cap | none (not frozen) | not measured |
| AR collections (re-gated, $0) | 0 | 5 min | unchanged | 6/6 all gates | invoices 87-99 / 16-34, accounts 64-91 / 17-23 |
| leave and absence (re-gated, $0) | 0 | 5 min | i02 reseeded twice by R-density, then passes | 6/6 all gates | LeaveRequest 59-66 / 11-19, AbsenceRecord 98-130 / 19-33 |

Insurance: 14 tools (6 paged list tools, `list_claims` among them), 10 tables, 37 blind tests, 1 debug round, none
dropped. Its first data-generator attempt under the fixed builder was dense (89 shown, 17 held-out, both
collections) and accepted. Instances: 0.03-0.05 s to generate (plus about 3 s to choose the actor by execution),
78-117 s for the six executed gates on CPU, USD 0.019 per panel rating. Claims are 100% reachable (worlds-1: 29-38%).

Leave and absence sits close to the bar under the shown split (59-66 shown): the held-out 20% costs it 9-15 distinct
mutants per instance. Worlds-1's measure (all mutants) was 68-77.

## Issues found by real calls and gate runs, and their fixes

| # | found by | issue | fix (with a test) |
|---|---|---|---|
| 1 | worlds-1 obscurity (insurance, returns) | a collection tasks act on had no list tool | the listing rule at the design stage (`unlisted_targets`) and executed at code and debug (`listing_problems`) |
| 2 | worlds-1 density (returns) | density found only at the gate, after freezing | measured at the data stage on the shown mutants and fed back with per-tool reach and refusals (`density_probe`, `sibling_reach`) |
| 3 | item 22 | the gate counted held-out mutants a certificate never sees | the bar is on `MUT(split="shown")`; `MUTHeldOut` measured beside it |
| 4 | PR attempt log (32 refusals) | `compile_spec` stopped at the first problem: each round fixed one, made another | every static problem returned at once (up to 15) |
| 5 | PR attempt log | `implies` was invalid syntax; `a.b.c` chains refused; `limit >= 0` on null was a TypeError; stale `inactive` refused six times; a unique template ran out | `implies`, chains (`follow`), SQL nulls (a rule holds unless False), stale markers outside live values on stale rows only, numbered template overflow |
| 6 | PR attempt log | told "use a backref" three times, the writer wrote two refs again | `normalize_spec`: the nullable side becomes a backref (recorded) |
| 7 | real code stage (returns, three coder attempts) | the executed check demanded a lister for a collection only the schema's keys made a target (shipments keyed by tracking number); the coder cannot add a tool | the executed check asks exactly what the design rule asked |
| 8 | real code stage (returns, three more attempts) | `created_at` "2025-12-15T09:00:00Z" read as another table's synthetic string: every listed row refused | the exact synthetic pattern; the last coder App re-checked clean and accepted without a new call (recorded in its stage note) |
| 9 | real data stage (insurance, ~$1) | conflicting-source noise planted on a loss notice's insured name, which a claim's lookup copies: three refusals the writer could not fix | no conflict on a field another lookup copies |
| 10 | real data stage (insurance) | the writer wrote 16 rules as `if A then B` | `if A then B [else C]` |
| 11 | real debug stage (PR t11) | a fixture's `required_by` 2024-06-01 failed "expired" against the clock 2026-01-01 for three rounds; nobody had told the tester or coder the clock | `CLOCK_T0` in both prompts; a date-failure hint to dispute; disputed and dropped by the approver in one round ($0.17) |
| 12 | a restart from design | the old design's generator problems stayed in the attempt log and would have been fed to the new writer | moved aside on a redesign |
| 13 | the gate runner | after a reseed only `--only`'s gates re-ran on a new start state | a reseeded instance is gated in full |

## Policy wiring (DESIGN item 24, `runs/policy-1.md` §4)

- `worldgates.N_MIN`, `families.N_MIN_SHOWN`, `FamilyConfig.k_min` and `density_gate`'s default read R-density (pinned
  by `test_r_scale_r_density_r_family_r_seed_conform`).
- Failing gates raise `density_gate_failing` / `world_gate_failing` through `policy.resolve`; the remedy, rule and policy
  version go in `gates.json`. The gate runner acted on them in this run: leave i02 (58 shown) was reseeded at k_min 60,
  then 90, and passed every gate. `rebuild_stage` is offered only with `--allow-rebuild` (it means a paid family build)
  and is then recorded in `rebuild_requests.jsonl`.
- The realism panel calibrates with `policy.panel_calibration()`. The worlds-1 votes re-scored under it: λ = 0.7
  (7 decided of both truths, all correct). With the rejected anchors added (below) the panel is uncalibrated (λ 1.01).
  It abstains, and R-panel's first remedy, `executed_check`, decides on the executed realism gates, recorded per
  instance as `decided_by`. The 18 usable instances carry that decision now; insurance's six were also rated realistic
  at λ 0.7 before the recalibration.

## The realism panel, measured

The panel was calibrated on 12 anchors: 2 reviewed finance_ap packs, 6 framework degradations of them, and 4 worlds-1
returns instances the obscurity gate rejected (refunds 18-26% reachable, no list tool). It rated all 4 rejected
instances realistic at confidence 0.90-0.98, and named a real flaw in one (VIP and standard return windows both 50
days) while still passing it. It decided the 8 known anchors correctly. At λ 0.7 that is 7 of 11 correct (accuracy
0.64), so under R-panel it is uncalibrated. A panel pass stays necessary but never sufficient. The gate-rejected anchors
show that this panel, reading a summary, does not see reachability; the executed gates do.

## Not done

| item | why | next step | estimate |
|---|---|---|---|
| returns and refunds | budget | `worlds build --keywords "Returns and Refunds"` resumes at tests | about USD 1 |
| purchase requisition | budget; its two data attempts were refused (rules `status ... implies field != null` the generators did not ensure; a filter on a `when`-gated status) | resume at datagen | about USD 0.5 |
| 4th family | the two above | either above, or a new catalogue family under the fixed builder | USD 1-1.5 |
| AR's design | fails the listing rule (no invoice list tool); its instances reach invoices through account links (obscurity 100%) | redesign only if the lead wants the rule applied to frozen families | USD 1.5 |
| panel | uncalibrated with rejected anchors | richer summaries (the tool list per collection), or keep executed gates as the realism verdict | small |

## Cost

| item | USD |
|---|---|
| insurance claims (design → freeze, incl. ~1.0 on the conflict-noise bug) | 2.13 |
| purchase requisition (design → debug pass, data stopped) | 2.00 |
| returns and refunds (design, two code stages: about 0.93 lost to the two listing-check bugs) | 1.18 |
| panel: 6 insurance ratings + calibration with 12 anchors | 0.24 |
| **total (receipts under `worlds/` after 2026-10-30 15:47)** | **5.56 of the 6 cap** |
