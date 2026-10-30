# Falsifiers-2: the measured blind spots closed, and the adjudication backlog (2026-10-29)

Code: DESIGN §7.7 item 20. CLI: `crucible/scripts/synth_falsifiers2.py`. Data: `crucible/data/synth/falsifiers-2/` (gitignored).
Models: Inkling-Small (leak probe), Nemotron-3-Ultra + gpt-oss-120b (panels), Inkling (one panel experiment). No Qwen model;
the student was never called. Locked sets (`data/sealed/synth-locked/rsi-{1,2}-locked.json`) were read only through
`recall.SetStore(purpose="report")`; every setting was fixed on dev, the fixtures and the natural tasks before a locked run.
Spend: **$1.93 of the $15 cap** (`work_spent(..., "falsifiers-2")`; every receipt is under that directory).

## 1. FR/FA before and after (rsi-2 sets; Wilson 95%)

"Before" is `cheap_m3()` (the population RSI-1/2 measured: no read mutants, exact-token leaks); "after" is `cheap()`.
Both are measured in one run, on the same plants and anchors.

| set | population | FR (all) | D6 | D7 | FA |
|---|---|---|---|---|---|
| locked (72 plants, 7 anchors) | before | 56/72 (.67–.86) | 0/12 (.00–.24) | 14/18 (.55–.91) | 0/7 (.00–.35) |
| locked | after | **72/72** (.95–1) | **12/12** (.76–1) | **18/18** (.82–1) | **0/7** (.00–.35) |
| dev (125 plants, 10 anchors) | before | 118/125 (.89–.97) | 25/25 | 12/12 | 0/10 |
| dev | after | 118/125 (.89–.97) | 25/25 | 12/12 | 0/10 (.00–.28) |

- The "before" row reproduces RSI-2's E1 exactly (56/72; misses `over_constrain:never` 12/12 and `leak_decision:humanised` 4/9).
- After: no miss on the locked set. MUT confirms every `never(read)` plant as D6 (MUT's locked hits 27 → 39); TAINT's
  separator layer confirms the 4 humanised leaks (TAINT 33 → 37). No class alarms on an anchor.
- Dev is unchanged: its only miss is still `relax_value:eq_to_any` (D3, 7/7 plants, MUT suspects only: the third M3 gap,
  which is the panel's, not a falsifier's).
- rsi-1 sets: see §1.1.

### 1.1 rsi-1 sets

| set | population | FR (all) | D6 | D7 | FA |
|---|---|---|---|---|---|
| locked (65 plants, 7 anchors) | before | 49/65 (.64–.84) | 0/12 (.00–.24) | 14/18 (.55–.91) | 0/7 (.00–.35) |
| locked | after | **65/65** (.94–1) | **12/12** (.76–1) | **18/18** (.82–1) | **0/7** (.00–.35) |
| dev (125 plants, 10 anchors) | before / after | 118/125 / 118/125 | 25/25 | 12/12 | 0/10 / 0/10 |

The same two gaps, closed the same way (MUT 21 → 33, TAINT 32 → 36 locked hits); the separator layer again suffices.

### 1.2 TAINT per structural layer (cumulative; locked rsi-2)

| layers | D7 found by TAINT | TAINT FA on anchors |
|---|---|---|
| exact (M2) | 14/18 | 0/7 |
| + sep | 18/18 | 0/7 |
| + number, lemma, label, synonym | 18/18 | 0/7 |

On the planted sets the separator layer closes the gap by itself (the planted humanised variant spells `price_variance` as
"price variance"). The deeper layers find nothing more there and alarm nowhere; their measured use is on natural text (§2.2).

## 2. Humanised leaks: the L2 probe and natural text

### 2.1 L2 as its own layer (`LeakProbe(skip_structural=False)`; leak panel λ = 0.7, calibrated on 57 dev anchors, 0.94 accurate at λ = 0.7, coverage 0.93)

| set | D7 plants confirmed | other plants and anchors: confirmed | suspects (human queue) |
|---|---|---|---|
| rsi-2 locked | 18/18 (.82–1) | 1/52 (.00–.10) | 4 |
| rsi-1 locked | 18/18 (.82–1) | 1/46 (.00–.11) | 9 |
| dev (rsi-2 / rsi-1 draws) | 12/12 (.76–1) | 18/113, 21/113 | 0 |

- Every dev "false alarm" comes from two of the nine dev anchors, inherited by their plants: the fixture `ap/hold-1003`
  ("bills more than its purchase order allows" for `price_variance`) and it_helpdesk `t02` ("because it has exceeded its
  service-level deadline" for `sla_breach`). Both panel families call both leaks in every run. They state the fact that
  decides the reason code, which is how §2 defines D7, but the anchors are certified. These are two alarms for a person to
  settle (anchor 2/9, .06–.55); they are not dropped.
- Spend per L2 pass over dev + locked: $0.11–0.12 (about 107 probe calls and the panel's judgements).

### 2.2 The structural layers on the 89 swarm-written tasks (213 derive values; instruction only: no labels, no options guard)

| layer | new hits beyond exact | leak panel (hard negatives in its calibration) |
|---|---|---|
| exact | 7 values (M2 already) | — |
| sep | 2 | 2 abstain (`INV-11118`, a duplicate check that names the very number; an enumerated `on hold`) |
| lemma | 6 | 4 leak, 2 clean |

The 4 panel-confirmed lemma hits state the deciding fact: "the requester is a VIP" / "flagged VIP" (`vip_request`, two
tasks), "SLA is breached" (`sla_breach`), "hold an invoice when its amount exceeds ..." (`on_hold`; a restated policy rule,
which a person should check). The two cleared hits are restated policy rules ("reject an invoice whose invoice number
duplicates..."). With the World, the options guard removes the enumerations among these.

## 3. Leaks through visible test data (repository substrate)

| version | `visible_data` (static) | `visible_check` (executed) |
|---|---|---|
| mvdan-sh-dc951aeb-e1 (record `...-table-entries`; pool and real-defect artifact are the same files) | clean | **confirmed D7**: `TestPrintTable/#267` (+ `-nl`, `-redo`) fails at A′ on two runs and passes at B |
| echo-61eca05b-e2 (record `echo-perf-bench-visible-data`; the pool build carries the lane's flag) | clean | clean: **miss** |
| black-ae5830b7-e1 | suspect (2 docstring cases of the hidden `docstring.py` held by `docstring2.py`) | clean |
| the other 9 pool tasks | clean | clean |

- **Independent real defects (E8):** leak_other 1/2 executable versions detected (.09–.91); 5e983709-e1/-e2 are lost. With
  M5's changelog finds the independent D7 count is 3/5.
- **False alarms:** `visible_check` 0/11 on the other pool tasks (.00–.26); `visible_data` 0/12 confirmed, 1/12 suspect.
- **New finding:** the current pool task mvdan-sh-dc951aeb-e1 still leaks one fix case through visible test data: the
  lane's table-entry strip (15 entries stripped from `printer_test.go`) left one entry that the fix turns green. Not
  changed here.
- **The echo miss:** the fix's benchmark changes in `perf_bench_test.go` carry no literal or definition of the fix and no
  test that runs; seeing them needs the pre-fix tree, which is not in the agent's view.

## 4. The open MUT suspects of the repository pool through the panel

The M5 text says 11 suspects on 8 tasks; `runs/m5/certificates_summary.json` lists **10** (echo 1, go-toml 1, huma 1,
react-hook-form 2, black-90117fa0 1, black-ae5830b7-c1 1, black-ae5830b7-e1 2, pulldown-cmark 1). All 10 were
re-proposed from the pool (same DropHunk sample) and shown to the panel as the attempt's diff. Calibration anchors carry
executed truth: per task the oracle diff (valid), the empty patch (invalid) and two commit-revert mutants (invalid: each
fails its milestone), 48 in all.

| run | panel | prompt | oracle right (Nemotron / gpt-oss) | revert-one-fix right | accuracy at λ = 0.7 | λ | suspects decided |
|---|---|---|---|---|---|---|---|
| 1 | Nemotron-Ultra + gpt-oss | tool-world `PANEL` | 3/12, 4/12 | 22/24, 19/24 | 0.79 (29 decided) | 0.95 | 0 |
| 2 | same | `PANEL_PATCH` | 8/12, 8/12 | 12/24, 20/24 | 0.73 | 0.95 | 0 |
| 3 | same | per-requirement variant | 9/12, 9/12 | 9/24, 15/24 | 0.55 | 0.95 | 0 |
| 4 | same | `PANEL_PATCH` (replicate) | 8/12, 9/12 | 11/24, 18/24 | 0.73 | 0.95 (1 decided anchor) | 1 (echo: valid) |
| **5** | same | `PANEL_PATCH`, strict calibration | 8/12, 7/12 | 12/24, 18/24 | 0.65 | **uncalibrated** | **0** |
| 6 | Inkling + gpt-oss | `PANEL_PATCH`, strict | 12/12, 8/12 | 7/24, 17/24 | 0.68 | uncalibrated | 0 |

**Verdict (run 5): every one of the 10 suspects abstains** and goes to the human queue (`falsifiers-2/triage/
falsifiers-2-repo-panel-20261029-221459/review.csv`, 10 items in 4 groups). **No task changes status.** Certified: 0;
provisional: carbon ×2, golang-net, mvdan-sh-dc951aeb-e1 (4, the tasks without suspects); rejected: echo, go-toml,
huma, react-hook-form, black-90117fa0, black-ae5830b7-c1 (also its confirmed D6), black-ae5830b7-e1, pulldown-cmark (8).
(The M5 text says 5 provisional and 7 rejected; the summary file says 4 and 8.)

Why: on diffs the panel cannot tell a fix with one milestone's code reverted from the whole fix. Nemotron-Ultra is at
chance on revert-one-fix anchors (9–12/24); gpt-oss is better (15–20/24) but they rarely agree; Inkling calls nearly
every diff valid. No prompt reached the 0.9 target. Run 4's single decision came from a calibration flaw (below).

## 5. The it_helpdesk human queue (53 rows)

`triage` (no decision taken): 53 rows are **51 distinct items** (2 queued twice) in **10 groups**; 24 clusters (same task,
reason and site pattern), whose representatives lead the sheet. Two tasks are **pivotal** (t11, t20: every rejection
reason is an open suspect or a bound missed for want of n, so deciding their 3 items can make them provisional); the
other 9 tasks stay rejected on confirmed counterexamples whatever the queue says. Every item is a two-family disagreement
(no abstention for low confidence alone). Largest groups: MUT/DupStep, Nemotron invalid vs gpt-oss valid (11); MUT/
PerturbValue in both directions (8 + 9); MUT/SwapEntity, Nemotron invalid vs gpt-oss valid (8). Sheet:
`falsifiers-2/triage/it_helpdesk-20261029-160257-full2/review.{csv,md}`; `read_decisions` reads it back.

## 6. Family exhaustion

The live Tinker list (2026-10-29) holds one family beyond DeepSeek, GLM, Kimi, Nemotron (Ultra, Super, Lightning,
Super-VL) and gpt-oss: **Thinking Machines Inkling** (975B/41B MoE) and **Inkling-Small** (276B/12B). Its tokenizer is its
own (200,058 tokens, `<|message_*|>`/`<|content_*|>` specials, no Qwen tokens) and the model card names no base model.
Wired: registry rows with prices, `INKLING`/`INKLING_SMALL`, family `inkling`, roles `leak_probe` (default) and
`SYNTH_ROLES_DISJOINT` (panel Inkling + Kimi: `checker_overlap` is empty, as it cannot be with five families). Smoke:
both models answer the leak-probe JSON ($0.0015). The default panel was not switched: run 6 shows Inkling is a lenient
judge of code diffs, so a switch needs its own calibration on tool traces first.

## 7. Issues the runs found, and their fixes

| # | found by | issue | fix |
|---|---|---|---|
| 1 | Inkling smoke | 2 of 3 replies were only the thinking message: sampling stopped at `<|end_message|>` | stop token `<|content_model_end_sampling|>`; `split_thinking` takes the last `<|content_text|>` message |
| 2 | first visible-data pass on the pool | 5 of 12 tasks alarmed: import lines, shared assertions, standard names (`ReadFrom`, `ReturnType`), a comment quoting `type Secret string` | data lines must be rare (in at most one other file, two or more), identifiers must be defined by the fix, literals need a space and a hidden test comparing them, comments are dropped: 0 confirmed, 1 suspect |
| 3 | repo panel run 1 | the tool-world prompt rejected 10 of 12 oracle fixes ("no STATUS.json", "no test run") | `PANEL_PATCH` judges the code only |
| 4 | repo panel run 3 | a per-requirement prompt was worse on revert-one-fix anchors (9/24 vs 12/24) | reverted, recorded in `roles.py` |
| 5 | repo panel run 4 | λ = 0.95 "calibrated" on one decided no-op anchor, then decided a suspect (echo would have become provisional) | `calibrate(min_decided, require_both, refuse_uncalibrated)`; the leak panel's calibration likewise |
| 6 | L2 on dev | the probe raised ticket ids picked by a stated rule ("every open ticket from bob@...") | prompt clause on selection rules: those raises stopped |
| 7 | natural tasks | the leak panel called restated policy rules leaks (8 of 8) | hard negatives (enumerations, restated rules) in its calibration and a prompt clause: 2 cleared, 2 abstain |
| 8 | the re-created strip version | its executed results equalled the stored version's | the lane runner drops writes to test paths (its custody rule); `pre_table_strip` is for static checks only |
| 9 | the M5 summary | "11 suspects on 8 tasks", "5 provisional, 7 rejected" | the file holds 10 suspects, 4 provisional, 8 rejected (reported; the paper core is not edited here) |

## 8. Open

- The repository panel is not trustworthy on diffs (§4): the 10 suspects need a person, or executed evidence (a
  differential test generated for the dropped hunk) instead of a judge.
- Two certified dev anchors are leaks by L2 + panel (§2.1) and one pool task leaks a fix case (§3): for a person; no task
  was changed.
- echo's benchmark-file leak has no detector that sees only the agent's tree.
- The `label` and `synonym` layers have no planted or independent positives yet; their value is measured on natural text
  only. The synonym map learned from this run's confirmed leaks is empty: the confirmed paraphrases are fact statements,
  not word swaps, and `align` refuses them.
- The `lemma` layer on natural text without the World guard flags restated rules; inside TAINT the options guard and
  catalogue siblings apply, but its precision on natural tasks there is not yet measured.
