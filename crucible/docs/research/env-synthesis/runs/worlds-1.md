# WORLDS-1: World families and instances, first pilot (2026-10-29/30)

Data: `crucible/data/synth/worlds/` (gitignored): `catalogue.json`, `realism_calibration.json`, `human_queue.jsonl`,
`<family>/state.json`, `<family>/app/` (the frozen family; `app.prev-*` are earlier versions copied aside),
`<family>/instances/<id>/` (each a servable App directory with `gates.json`, `instance.json`, `noise.json`,
`actor.json`), `report.json` / `report.md`. Code: DESIGN.md §7.7 item 16.

Swarm only: GLM-5.3 (domain spec, schema, documents), Kimi-K2.6 (design, code, debug, data generator), DeepSeek-V4.1-Flash
(catalogue, blind tests, colleagues), Nemotron-3-Ultra (test approver), Nemotron-3-Ultra + gpt-oss-120b (realism panel).
No Qwen in any role; the student was never called.

## Headline

| | |
|---|---|
| catalogue | 45 domains over 16 function areas, $0.024 |
| families started / frozen | 5 / 4 (purchase requisition: data generator never accepted) |
| instances generated / usable (all 7 gates) | 20 / **12**: AR collections 6/6, leave and absence 6/6 |
| not usable | insurance claims 0/4 (obscurity: no list tool for claims), returns and refunds 0/4 (density, obscurity) |
| cost | **$24.15 of the $30 cap** (shared Part 2 total $79.32 of $150) |

The 12 usable environments come from **2 families**. The lead's target is 10 environments from 4 or 5 families, so the
count is met but the diversity is not. Each further instance of a frozen family costs $0 and about 0.05 s to generate,
plus about $0.017 to rate.

## Gates on the final 20 instances

| gate | pass | |
|---|---|---|
| build | 20/20 | kept blind tests re-run, digests, static check + canaries, sandbox-exec |
| determinism | 20/20 | regenerated seed identical; resets identical; reference programs replay identically |
| integrity | 20/20 | types, references, declared values, dates, rules (incl. cross-record), master copies, aggregates; documents state the variant's values |
| surface | 20/20 | pagination, filters, unknown-id and bad-enum refusals, impersonation probe, a replayed PERMISSION_DENIED |
| density | 16/20 | returns: its Customer and Refund collections reach n = 12–39 at best |
| obscurity | 12/20 | insurance: 29–38% of claims reachable (no list tool for claims, one claim per notice summary); returns: 18–26% of refunds |
| panel | 20/20 | **it also passed all 8 instances the executed gates reject** (see below) |

Density on the usable instances (distinct mutants of the best reference oracle writing each target collection):

| family | targets (entities) | best n per collection | dense write tools | acting user (chosen by execution) |
|---|---|---|---|---|
| AR collections | invoices 80–128, customer accounts 50–55 | invoices 106–130, accounts 83–106 | 3–5 of 8 | AR Supervisor (2), Credit Analyst (4) |
| leave and absence | leave requests 60–96, absence records 55–88 | requests 68–77, absences 127–147 | 2 of 9 | HR Administrator |

Instances: 356–603 rows, 6–31 planted noise items (near-duplicates, stale records; AR also conflicting master copies;
leave and returns have none, because their data generators wrote no master copy). Generation takes 0.03–0.07 s.
All seven gates take 19–68 s per instance on CPU (density dominates).

**The realism panel** was calibrated on the reviewed finance_ap pack (2 seeds, realistic) and 6 framework degradations of
it (not realistic). It decided 7 of 8 anchors correctly (λ = 0.6, the floor). On the generated Worlds it rated every
instance realistic (mean scores 3.8–5.0, confidence ≥ 0.9), including those with unreachable claims and thin refunds. Its
listed problems were real and were missed by the executed checks: leave balances with remaining > entitled; a returns
variant whose VIP and standard windows coincide (that spec predates `param_constraints`); a boolean rendered as "yes is in
effect"; an AR invoice that is written off but still has a promise date. As built, the panel separates toys from
plausible Worlds but does not rank plausible Worlds. Treat its pass as necessary, not sufficient; the executed gates do the
filtering.

## Families: cost and time

Cumulative over every rebuild the issues below forced. The totals include $1.55 of conservative estimates booked by
hand for calls in flight when a run was stopped (`kind: unbooked_estimate`). Five families were built concurrently in
each round.

| family | status | tools | tables | blind tests | $ total | rounds |
|---|---|---|---|---|---|---|
| accounts_receivable_collections | frozen | 11 | 11 | 28 → redone with coded permission tests | 1.88 | 4 |
| leave_and_absence_management | frozen | 14 | 9 | 47 | 3.62 | 5 |
| returns_and_refunds | frozen | 14 | 11 | 38 | 3.98 | 6 |
| insurance_claims_intake | frozen | 11 | 10 | 31 | 6.12 | 7 (four data-generator rounds) |
| purchase_requisition_to_order | data generator never accepted | 12 | 16 | 30 (1 dropped on approval) | 7.75 | 7 |

**One clean pass**, first attempts only, spec to freeze: about $0.8–1.6 and 40–60 minutes per family. Kimi dominates the
time: a design call takes 50 s alone, but debug and data-generator calls take 4–13 min each with 5 families in parallel.
First pass per stage: spec $0.03 / 15–70 s (GLM); design $0.08–0.23 / 1–4 min; schema $0.04 / 5 s (GLM); code
$0.17–0.25 / 3–4 min; tests $0.03 / 35 s (DeepSeek); debug $0.15–0.7 / 2–20 min; data generator $0.15–0.4 / 2–20 min (but
$0.8–1.6 for the two families the language did not fit); documents $0.06 / 20 s; colleagues $0.005 / 10 s.

## Issues found by real calls and gate executions, and their fixes

| # | found by | issue | fix |
|---|---|---|---|
| 1 | catalogue | the swarm wrote a description into `function` for 5 of 16 areas | function = the area asked |
| 2 | dedupe tests | 3-token keywords collided at Jaccard 0.5; AP did not match accounts payable | 0.6, acronyms expanded |
| 3 | a design | all 9 write tools took `acting_user_id`: impersonation around every permission check | design check refuses it |
| 4 | surface gate | it came back through the tests (the argument) and the code (a hidden parameter) | parameter conformance for every World (F1 too), tests may not invent arguments, executed impersonation probe |
| 5 | usage rows | Kimi schema deriver hit 8 192 tokens; every first app-coder attempt hit 20k | schema on GLM; family coder/debugger 64k |
| 6 | a stopped run | calls in flight are never booked | estimates booked by hand |
| 7 | 3 stuck debug rounds | `strptime`/`strftime` raised ImportError inside the sandbox (C-level lazy imports through the child's guard); the App's `except` hid it | internal modules admitted at run time; the source check is unchanged |
| 8 | 3 stuck debug rounds | a test needed a write the design does not declare | debugger hint: dispute it |
| 9 | gates (bugs) | required page params hid list tools; one tool's enum compared with every table's `status`; link-only records counted unreachable; any refusal counted as a permission test; only the first 8 refusal tests replayed | fixed |
| 10 | surface | AR declared PERMISSION_DENIED on 8 tools with no check in code and no test (5/5 families' tests missed it); then tests passed on any refusal | `check_permission_tests` + `ToolTestCall.error` (the refusal must name the code) |
| 11 | data stage | rules broken by independent draws; a lognormal sd in value units overflowed; nested `else` generators; ref filters on absent values | rejection sampling; sd > 3 read as absolute; `else` generators; filters checked |
| 12 | instances | dense profiles exhausted unique pools; policy draws over the whole range (a 50% restocking fee) | caps and numbered overflow; draws scale the default; `param_constraints` |
| 13 | density | loss notices after their policy's expiry: every `open_claim` refused | cross-record rules `<ref>.<field>`, also for `when`, `calc`, `date_offset` |
| 14 | instances | no user held a role a filter needed; a self-referencing hierarchy had no root | choice coverage; empty parent at the root |
| 15 | data check | a spec checked at seed 0 failed at 3 of 4 instance seeds | all four profiles sampled before freeze |
| 16 | density | the acting user was a Finance Clerk (every write refused), then an Employee | the actor chosen by execution (`choose_actor`), kept in the config |
| 17 | density bar | "half of all write tools reach 59" rejected realistic Apps with some rarely-applicable tools | per collection (the task spec's rule); tools below are `thin_tools`, kept out of the batch cells |
| 18 | attempt logs | identical resumed prompts returned the identical refused generator (seed 0); the writer repeated a fixed mistake | logged problems, history feedback; the worlds wire seeds by tag |
| 19 | attempt logs | our own near-duplicate noise wrote typo statuses; two-way links refused as cycles; `null` in rules; equality to a referenced field | choice-aware noise; `backref`; null aliases; "use lookup" hint |

## Not done

| item | why | next step | estimate |
|---|---|---|---|
| insurance claims | no list tool for claims, so obscurity fails | redesign (`--restart-from design`) | $1.5 |
| returns and refunds | Customer/Refund density (12–39) and refunds 18–26% reachable | redesign or redo the data generator | $1–1.5 |
| purchase requisition | 4 data-generator rounds refused (polymorphic refs, `implies`, rules through nullable refs) | one more round with the new language | $1.3 |
| diversity | 12 usable from 2 families | the three above, or 2 to 3 new catalogue families | $1–1.5 each |
| panel | rates every swarm World realistic | harder negative anchors (subtle inconsistencies), more reviewed positives | small |
| certification | density is measured on reference oracles, not on certified tasks | `batch --world name=<instance>` over the 12 usable instances | task budget |
| conflicts | leave and returns have no master copies, so no conflicting-source noise | data writer already asked for lookups; redo data | — |

## Cost

| item | $ |
|---|---|
| catalogue | 0.024 |
| panel calibration + 40 instance ratings | 0.71 |
| five families (every round, incl. $1.55 hand-booked estimates) | 23.4 |
| **total (receipts under `worlds/`)** | **24.15 of the 30 cap** |
