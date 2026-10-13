# M14 study, hunt H2: the packs' domain code

Scope: the domain code of `packs/{finance_ap,_toy,coding,_template}`. That covers graders and verifiers, World apps and
seeds, generators, scripted policies, stages, fixtures, credit hooks, ingest sources and recorded runs. Spec:
`docs/modules/M14-packs.md` and each pack's README and `pack.yaml`. This is a read-only study (red team plus grader
audit), 2026-10-13, HEAD `f2d9e18`.

Paths: `P` = `crucible/packs/`, `C` = `crucible/src/crucible/`. The probes are in the session scratchpad `m14/h2/*.py`,
and each has its output next to it (`*.out`). Run them from `crucible/` with
`PYTHONPATH=src:packs/coding/src:packs/_toy/src:packs/finance_ap/src:packs/_template/src:<scratchpad>/m14/h2`.
No Docker images were pulled and no GPU or paid calls were made. Nothing under `crucible/` was written except this file.

## Summary

| ID | Sev | Where | Defect |
|---|---|---|---|
| V1 | P0 | `P/finance_ap/.../ap_tools.py:16-34`, `P/_toy/data/helpdesk/tools/tickets.py:23-53`, `P/_toy/.../longhorizon/backlog_tools.py:18-27`; root `C/world/packaging/mcp_tools.py:21` | A policy turns its own failing episode into a **mask**. It passes one out-of-range integer (`invoice_id=10**20`). The sqlite3 `OverflowError` is an `ArithmeticError`, so it lands in `BUG_EXCEPTIONS`, which becomes an infra fault and a masked sample. Approve-everything plus one bad call goes from 0 to "absent from every metric" |
| V2 | P0 | `P/_toy/.../verify.py:49-51,90` | Toy helpdesk: the `approvals` gate accepts a supervisor name the policy typed itself. The **gold never asks the supervisor**. `approved_by="Not Sam Lee - self-approved"` also passes (`LIKE '%Sam Lee%'`) |
| V3 | P0 | `P/_toy/.../verify.py:42` (same rule in `forge/roles.py:88`, `forge/vocabulary.py:28`) | Toy helpdesk: split the 129.99 refund into 3 × 43.33. That needs no approval and scores 1.0. The approval rule is checked per refund row, not per refunded amount |
| V4 | P0 | `C/workflow/milestones/evaluate.py` (grep) vs `C/workflow/milestones/mine.py:62-74`; `P/coding/.../studio.py:32-42,80-99` | Coding dense reward: a patch that does no migration reaches Φ 0.92 to 1.00 on all four seeds. It puts the oracle's tokens in **comments**, deletes the "removed" tokens and the files, and touches each anchored block. That earns the maximum failing credit (0.184 to 0.200), above every honest failing frontier run (0.055 to 0.124). Honest *passing* runs reach only Φ 0.30 to 0.46 |
| V5 | P1 | `P/finance_ap/.../studio.py:29-40` | finance_ap Φ ignores the `controls` gate. Paying a duplicate after 6 correct decisions gives Φ_T 0.857 at reward 0, and an honest partial run gives Φ 0.143 at reward 0.143. A breach has ΔΦ = 0. This breaks `verify/progress.py`'s contract that Φ_T equals the grade |
| V6 | P1 | `P/finance_ap/.../world.py:14,62,75,77` | The per-world tolerance does not force reading the policy. A memorized 2% **never** breaches the gate (0/1000 worlds) and scores 1.0 in 73% of worlds (expected 0.962). A memorized 1% never breaches either. The docstring's claim ("a policy that memorized 2% fails") is false |
| V7 | P1 | `P/finance_ap/.../world.py:59-92,111` | No invoice ever triggers two rules (0 of 35,000). "Apply the rules in this order, the first that applies decides" is never tested, so a policy that applies the rules in any order scores 1.0 |
| V8 | P1 | `P/finance_ap/.../world.py:84-88`, `ap_tools.py:62-67` | The duplicate control is trivial. All 5,000 of 5,000 duplicates are byte-identical (same number string, amount and PO), so an exact-match `find_invoices` suffices. There are no format variants, no changed amounts, no aliased vendors, and no "earlier invoice rejected, so approve" near miss |
| V9 | P1 | `P/coding/.../connectors/agent_runs.py:78-82,107-113` | Recorded-run labels ignore the task version. Two task04 **v1** passes are labelled `good` for a task that is now at **v3**, where the same models fail. The trace carries no `task_version` |
| V10 | P2 | `P/finance_ap/.../world.py:63`, `studio.py:44,97` | Every world (5000/5000) holds exactly one invoice of each of the 7 kinds. That is a structural prior: the 7th decision follows by elimination, and the base rate of 2/7 approvals is far from real AP |
| V11 | P2 | `P/_toy/.../longhorizon/backlog_tools.py:63-73`, `task.py:79-84` | Backlog: closing an awaiting ticket before the customer's reply (rule 6) is erased by a later re-close, which overwrites `resolved_at`. The premature action scores 1.0 |
| V12 | P2 | `P/finance_ap/.../studio.py:103-110,128`, `P/_toy/.../studio.py:58-68` | `stages()` returns None for the packs' own anchor tasks, so there are no stage metrics on anchors. The `fixtures()` World (`APTask(3)`) leaks a temp dir on every `pack check`. 183 `finance-ap-*` and 1,028 `backlog-pack-*` dirs are in `$TMPDIR` now |
| V13 | P2 | `P/_toy/.../verify.py:66,69,72` | Toy notes predicates accept any body (`"x"` scores 1.0). The t3 note need not mention the carrier trace |
| V14 | P2 | `P/coding/.../studio.py:14,80-99`, M14 doc table | Coding CPU fixtures cover the superseded `task01_prims_omp` (bench: false), not `task01v2_tessera_omp` (bench: true). They hold only oracle and base cases, so no fixture can catch V4 |

Checked and clean (no finding):
- **finance_ap hidden state.** `kb.read_doc` refuses `../` and absolute paths to `forge/world.json` and to the tool
  source (`fa_kb.out`). The ERP tools expose neither `kind` nor `expected`. Decisions are final (`_decide` needs
  `pending`), historical invoices cannot be touched, and reason codes are validated.
- **finance_ap labels and the policy text.** A reference implementation of AP-3, run on the data of 5,000 worlds,
  agrees with every label (`fa_scan.out`: 0 mismatches). It is deterministic per seed, and all 4 tolerances are drawn
  evenly.
- **Recorded runs.** There are no secrets: 32 `hf_`/`sk-` hits, all substrings inside base64 blobs, and 0 with a word
  boundary. `/Users/<user>` home paths in the trajectories are gone after `redact`. The 55 PCL PR ids are stable and
  unique across two discoveries, and the title regex equals `m3.gate.GPU_MIGRATION` (`coding_prs.out`).
- **Toy backlog grader.** One refund per ticket, approvals bound to in-episode supervisor messages by amount and time,
  and thresholds taken at refund time. Splitting and forged approvals fail there, unlike V2 and V3.

---

## V1 (P0): a policy can turn its own failure into an infra mask (all three SQLite packs)

**Defect.** The pack tool modules pass agent integers straight to SQLite: `get_invoice(invoice_id)`,
`list_invoices(offset)`, `get_ticket`, `set_status`, `list_tickets(offset)`. A value ≥ 2**63 raises
`OverflowError: Python int too large to convert to SQLite INTEGER`. `C/world/packaging/mcp_tools.py:21` lists
`ArithmeticError` (the parent of `OverflowError`) in `BUG_EXCEPTIONS`, so `InProcessToolModule.call` returns
`error="internal_error", fault="infra"`. `World._call_tool` (`C/world/world.py:486-487`) records a `crucible:tool_crashed`
failure, and `run_episode` masks the sample. JSON-schema `integer` has no bounds, so `_argument_problem` lets the call
through. The M10 audit closed "a policy turns its failure into a mask" for World-level faults. This is the same hole
reached through the pack tools.

**Reproducer.** `fa_overflow.py`, `fa_overflow_bench.py` and `toy_overflow.py`:
```
approve_all            reward 0.0 gates {'controls': 'fail'} failures []
approve_all+overflow   reward 0.0 gates {'controls': 'fail'} failures [('crucible:tool_crashed', 'erp.get_invoice: internal error in erp.get_invoice')]
# run_bench, 3 finance_ap tasks:
scripted:approve_all                {'mean_reward': 0.0, ...}
scripted:approve_all_then_overflow  {'mean_reward': None, ...}      run {'planned': 6, 'measured': 3, 'masked': 3}
warning: '... 3 of 3 tasks have no measured sample and are absent from every quality metric'
helpdesk: set_status(ticket_id=10**20) -> crucible:tool_crashed     backlog: list_tickets(offset=2**63) -> crucible:tool_crashed
```
In training, the masked rollout drops out of the gradient. A policy learns to append the bad call whenever it is about
to score 0.

**Minimal fix.**
- **Packs.** Bound every integer argument before it reaches SQLite: `if not -2**63 <= x < 2**63: raise ValueError(...)`.
  A shared `_int(x)` helper in each tools file does it.
- **Core (the real fix, outside H2's write scope).** An exception raised while *binding the policy's arguments* is the
  policy's fault. Drop `OverflowError` from `BUG_EXCEPTIONS`, or classify `OverflowError` and
  `sqlite3.InterfaceError`/`DataError` as `fault="tool"`. Also give `_argument_problem` int64 bounds for `integer`.

**Regression test (fails now).** Name: `test_m14_audit_packs.py::test_out_of_range_int_is_a_policy_error[finance_ap|toy|backlog]`.
For each pack, play `[<one tool call with 10**20 / 2**63>]` and assert `w.failures == []`. The `ToolResult` must have
`fault in ("tool", "policy")`. Add a bench variant: approve-everything followed by the bad call stays measured at 0.0.

## V2 (P0): toy helpdesk, a forged supervisor approval scores 1.0, and the gold is that forgery

**Defect.** The instruction says "ask the supervisor to approve refunds above 50 USD before issuing them", and a
`RelayActor("supervisor")` answers "Approved refund of N USD. - Sam Lee" (`P/_toy/.../__init__.py:40-43`). The grader
only reads state. `approvals` is `approved_by NOT LIKE '%Sam Lee%'` and `refund_policy` checks that `approved_by` is
non-empty. `GOLD[0]` issues the 129.99 refund with `approved_by="Sam Lee"` and never sends the supervisor a message. So the
pack's gold, its fixture "gold", its semantic check and the M7 golden report all certify a memorized approval. The
`LIKE` also passes any string that contains the name. The forge already solved this as `approval_grounded`
(`P/_toy/.../forge/task.py:35-79`, M9 A13 / B.15), and the backlog grader binds approvals to supervisor messages. The
base helpdesk task never got the fix.

**Reproducer.** `toy_hacks.py`:
```
gold (never messages the supervisor)   reward=1.0 gates={... 'approvals': 'pass' ...} supervisor_msgs=0
fabricated 'Not Sam Lee'               reward=1.0 gates={... 'approvals': 'pass' ...} supervisor_msgs=0
```

**Minimal fix.**
- Add `approval_grounded("approvals/grounded", "approvals", "supervisor")` to `verify.plan()`, and pass the episode
  events into `Context.params` from `HelpdeskTask.grade`.
- Compare `approved_by` with `= 'Sam Lee'`, not `LIKE`.
- Make `GOLD` send `Message(to="supervisor", text="Please approve a refund of 129.99 USD for ticket 1, order O-5001.")`
  first and wait out the 15-minute reply latency (3 steps).

**Regression test (fails now).** Add two cases to `NEGATIVES`. The current `GOLD` becomes `forged_approval_no_request`,
which must fail `approvals`. `_swap(GOLD, 0, approved_by="Not Sam Lee")` must fail `approvals`. Then
`fixtures()` must include both, and `pack check --level fixture` stays green only with the fix.

## V3 (P0): toy helpdesk, splitting a refund under the 50 USD threshold avoids approval and scores 1.0

**Defect.** `refund_over_50_unapproved` checks each refund row (`amount_usd > 50`). The outcome predicate `t1/refund`
sums per ticket and order. Three refunds of 43.33 sum to 129.99, need no approval and score 1.0. Real AP and refund
controls treat split transactions under an approval threshold as a classic fraud pattern. The forged toy tasks use the
same per-row SQL (`forge/roles.py:88`, `forge/vocabulary.py:28`) and sum outcomes the same way (`roles.py:136-138`), so
they are exposed too. That is from code reading: I did not run a forged task.

**Reproducer.** `toy_hacks.py`:
```
split_refund_under_threshold   reward=1.0 gates={'refund_policy': 'pass', 'approvals': 'pass', 't1': 'pass', ...} supervisor_msgs=0
```

**Minimal fix.** Apply the threshold to the refunded total of an order:
```sql
SELECT order_id FROM refunds GROUP BY order_id
HAVING SUM(amount_usd) > 50 AND SUM(CASE WHEN approved_by IS NULL OR approved_by = '' THEN 1 ELSE 0 END) > 0
```
Or require exactly one refund per ticket, as the backlog grader does (`len(refunds) != 1`). Make the same change in
`forge/roles.py` and `vocabulary.py`.

**Regression test (fails now).** Add `NEGATIVES["split_refund"]`: three 43.33 refunds on O-5001 with no approver, which
must fail `refund_policy`. Add the same case to a forged task's `tamper_attempts`.

## V4 (P0): coding dense reward, comment stuffing earns the maximum failing credit without migrating anything

**Defect.** The pack's dense reward is `compose(outcome, phi=Φ(final patch))`, with Φ taken over milestones mined from
each seed's oracle diff. The miner drops comment lines when it picks tokens (`mine.py:62-74`). The `grep` milestone
check counts matches on **any** line, comments included (`evaluate.py`, op `grep`). These parts can be satisfied
without doing the work:
- `deleted` and `removed` milestones are met by deleting files and code;
- `located` milestones (`block_changed`) are met by any edit inside the block;
- `created` milestones are met by an empty file.

The probe writes a patch with none of the migration in it. It appends `// <token>` for every added token, rewrites the
removed tokens, deletes the deleted files and adds a comment line in each anchored block. On a failing outcome that gets
κΦ ≈ 0.2, the cap, which is about 1.6 to 3.6 times what honest failed frontier attempts earn. Φ also ranks honest
**passing** solutions low (0.30 to 0.46, `docs/modules/reports/m6_properties_coding.json`). It measures similarity to
the oracle's text, not progress. The pack's fixtures (oracle 1, base 0) cannot see any of this.

**Reproducer.** `coding_stuff.py`:
```
task01_prims_omp:      milestones=118 stuffed_phi=0.990 reward_if_failed=0.198   (honest fail: none recorded; honest pass Φ 0.46)
task02_pcops_triton:   milestones=48  stuffed_phi=1.000 reward_if_failed=0.200   (honest fail Φ 0.42 -> 0.084)
task03_mdlite_pallas:  milestones=107 stuffed_phi=0.989 reward_if_failed=0.198   (honest fail Φ 0.27 -> 0.055)
task04_pcl_gpu_cuda12: milestones=355 stuffed_phi=0.921 reward_if_failed=0.184   (honest fails Φ 0.30/0.56 -> 0.061/0.112; honest passes Φ 0.30/0.32)
```

**Minimal fix.**
1. Evaluate `grep` milestones on comment-stripped text, with the same rule `_tokens` uses (`//`, `#` except
   preprocessor lines, `/* */`), and in Python on `#` comments and docstrings. Mining and checking must agree.
2. Weight destructive milestones (`deleted`, `removed`) to zero unless a positive milestone of the same file holds
   (prerequisite edges, as R5 intends).
3. Add an adversarial CPU fixture per seed.

**Regression test (fails now).** Name: `fixtures()` gains `"{task}:stuffed"`. Its grade builds the stuffed FileState from
`_mine_cached(task)` (the probe's construction) and returns `compose(outcome=False, phi)`. Expected: a reward ≤ the
lowest honest failing run's reward on that seed (task02 0.084, task03 0.055, task04 0.061), or Φ ≤ 0.1. All four
currently score 0.184 to 0.200.

## V5 (P1): finance_ap Φ rewards progress in episodes that fail the controls gate

**Defect.** `InvoiceSubgoals.evaluate` scores each invoice on `state == expected` alone. The grade is
`0 if breaches else frac`. So Φ_T equals `frac`, not the reward, whenever the gate fails, which breaks
`verify/progress.py` ("Φ_T must equal the grade the task reports"). Approving a stop-invoice leaves Φ unchanged, so
dense credit never registers the fraud. The M14 doc reports "approve-everything earns AUC 0.15 while scoring 0" as
evidence for the study's M6 V3 finding. The structural fix was never built into the pack.

**Reproducer.** `fa_phi.py` (APTask(3), `ProgressTracker` as the runtime uses it):
```
approve_all                                     reward=0.000 controls=fail phi_final=0.286
honest: decide the duplicate correctly, stop    reward=0.143 controls=pass phi_final=0.143
6 correct + approve the duplicate (pays fraud)  reward=0.000 controls=fail phi_final=0.857
oracle                                          reward=1.000 controls=pass phi_final=1.000
```

**Minimal fix.** Make each subgoal "invoice correct **and** no breach anywhere":
- in `evaluate`, compute the breach set from the same `state()` query and return False for every id when it is
  non-empty;
- add every `invoice:<id>` key to every subgoal's keys, so a breach on one invoice re-evaluates all of them (n = 7, so
  the cost is trivial).

Φ then drops to 0 on the breaching step (ΔΦ < 0) and telescopes to the reward.

**Regression test (fails now).** For approve-everything and for "6 correct plus approve the duplicate", assert that
`tracker.phi == grade.reward == 0` and that the approving step's `delta_phi < 0`.

## V6 (P1): finance_ap's per-world tolerance does not test reading the policy

**Defect.** Within-tolerance amounts sit at `tol × U(0.2, 0.9)` above the PO. Price-variance amounts sit at
`tol + U(1pp, 8pp)` above it. Tolerances are {1, 2, 3, 5}%. Any fixed threshold in [0, 2%] is therefore gate-safe in
every world (the smallest price variance is tol + 1pp ≥ 2%). It only loses the non-gated within-tolerance invoice
sometimes. "Exactly at tolerance" (`>` vs `>=`) is never drawn: the closest within-tolerance case is 0.1pp below the bar
and the closest price variance is 1.0pp above it.

**Reproducer.** `fa_scan.py`, 1,000 worlds:
```
memorised 0.01: price_variance approved (gate breach) in 0/1000, within_tolerance held in 611/1000
memorised 0.02: price_variance approved (gate breach) in 0/1000, within_tolerance held in 266/1000   -> mean reward 0.962, 1.0 in 73% of worlds
memorised 0.03: breach in 36/1000 ; memorised 0.05: breach in 203/1000
within_tolerance: min distance below the bar = 0.101 pp; price_variance: min distance above the bar = 1.002 pp
```

**Minimal fix.**
- Draw price variances just over the bar, at `tol + U(0.05pp, 1pp)`, and within-tolerance cases up to and **including**
  exactly `tol` (rounded to the cent the right way).
- Widen and spread the tolerance set, for example {0.5, 1, 2, 3, 5, 7.5, 10}%, so that no constant is gate-safe.
- Say in the policy whether "more than" is strict.

**Regression test (fails now).** Over 1,000 seeds, for each fixed threshold in `np.arange(0, 0.11, 0.0025)`, the
breach-or-error rate on price invoices must be ≥ 0.3. Add a fixture `approves_exactly_at_tolerance` (reward 1) and
`approves_one_cent_over` (reward 0, fails `controls`).

## V7 (P1): finance_ap rule precedence is never exercised

**Defect.** Each case gets its own vendor, PO and receipt, built so that only its own rule fires. The policy's "Apply the
rules in this order; the first that applies decides" has no case where it matters. A policy that checks the rules in
reverse, or that checks only the one rule it finds first, still scores 1.0. In real AP the overlaps are where
decisions differ: a blocked vendor that also has a bank change, a duplicate that is also over price, a bank change with
a missing receipt.

**Reproducer.** `fa_scan.py`: `invoices_with_2+_rules=0` over 5,000 worlds (35,000 invoices).

**Minimal fix.**
- Add overlap kinds: `blocked_and_bank_change` (reject `vendor_blocked`), `duplicate_and_price_variance` (reject
  `duplicate`), `bank_change_and_missing_receipt` (hold `bank_verification`).
- Derive `expected` from a reference implementation of AP-3 applied to the generated data, not from `kind`.
  `fa_scan.triggers` is one.

**Regression test (fails now).** `fa_scan`: assert that at least one invoice per world triggers ≥ 2 rules, and that
`expected == first trigger` for every invoice. Add a fixture `reverse_order_policy`, scored below 1.

## V8 (P1): finance_ap's duplicate control covers only exact copies

**Defect.**
- The historical invoice is written with the same `invoice_number` string, amount and PO (`world.py:85-87`), so
  `find_invoices(vendor_id, invoice_number)` (exact `=`) answers the question.
- None of the variants that real duplicate payments use is generated: case or whitespace (`inv-12345`, `INV-12345 `),
  dropped punctuation or leading zeros (`INV12345`, `INV-012345`), an amount that differs by a rounding or a currency
  restatement, a second vendor record (alias) for the same payee, or the same number billed under another PO.
- There is no negative for the rule's own condition ("…that was approved"): an earlier invoice with the same number
  that was rejected or put on hold must **not** make the new one a duplicate.
- With `n > 10`, vendor names repeat across vendor ids (`n=14`: "Fjord Metals" is both V-100 and V-112, both active).
  No world uses an alias to hide a blocked vendor.

**Reproducer.** `fa_scan.py`: `duplicates: 5000/5000 are byte-identical`.

**Minimal fix.**
- Generate duplicate variants with a normalized match. Either state the normalization in the policy and give the ERP a
  `search_invoices(vendor_id, query)` with fuzzy matching, or let the agent list a vendor's invoices.
- Add a `same_number_earlier_rejected` near miss (expected: approve).
- Add a `blocked_alias` kind: two vendor ids with one name and bank account, one blocked. Either make the policy say
  that blocks apply to the payee, or keep it a near miss. State whichever rule the grader uses.

**Regression test (fails now).** Over 1,000 worlds, at least one duplicate per world must not match exactly. A fixture
whose policy does exact-match `find_invoices` only must score below 1, and the `same_number_earlier_rejected` fixture,
decided as approve, must score 1.

## V9 (P1): coding recorded-run labels mix task versions

**Defect.** `_recorded` finds a run's calibration record, which holds `task_version`. `normalize` copies `score` and
`failing_gates` but not the version, and labels by `score` alone. task04 is at v3 (speed and concurrency bars, the
oracle v2 hardening). Its v1 passes (`gpt-5.6-sol-run1`, `gpt-6-astra-run1`) become `good` traces of the current task,
while the same models' v3 attempts fail (`failing_gates` perf/correctness). `credit()` and the M6 coding report also
take them as outcome-1 states. So label, SFT and KTO data teach pre-hardening solutions as successes. `bench_recorded.py`
already treats each version as its own item. The label path does not.

**Reproducer.** `coding_runs.py`:
```
task04_pcl_gpu_cuda12/gpt-5.6-sol-run1   label=good  rec_version=v1 current=v3 details_has_version=False
task04_pcl_gpu_cuda12/gpt-6-astra-run1   label=good  rec_version=v1 current=v3 details_has_version=False
```

**Minimal fix.**
- Put `task_version` into `outcome.details` (from `_recorded`, or from the grade report).
- When it differs from `SEEDS[task]["version"]`, label the run `unknown` with the reason "graded under v1; the task is
  v3", or use the family `coding/agent_run@<version>`.
- `credit()` and `m56/task_properties` skip mismatched versions.

**Regression test (fails now).** Normalize `task04_pcl_gpu_cuda12/gpt-6-astra-run1`. Assert
`outcome.details["task_version"] == "v1"` and `outcome.label != "good"`, or that it is a separate family.

## V10 (P2): finance_ap worlds have a fixed composition

**Defect.** `generate` and `tasks` always use `n = 7 = len(KINDS)`, so `kinds = KINDS` shuffled. Every world holds
exactly 2 approvals, 3 holds (one per reason) and 2 rejects (one per reason), with exactly one blocked vendor and one
bank change. A policy can decide the last invoice by elimination, and a benchmark built on these worlds measures
nothing at realistic base rates, where most invoices are clean.

**Reproducer.** `fa_scan.py`: `kind multiplicity per world: {(1,1,1,1,1,1,1): 5000}`.

**Fix.** Draw the number of invoices and the kind mix per world, for example clean 50 to 70%, with each control present
with probability p, plus some worlds with zero or two of a control.

**Test.** Over 1,000 worlds, assert that more than one composition occurs and that some worlds have no `vendor_blocked`
and some have two.

## V11 (P2): toy backlog, a premature close is erased by a later one

**Defect.** `set_status(..., "resolved")` overwrites `resolved_at` every time (`backlog_tools.py:69-70`). `_correct` only
compares the final `resolved_at` with the reply time. Rule 6 says to act only on the customer's reply, but closing an
awaiting ticket before the reply and closing it again afterwards scores as correct. Refunds are protected, because
their `created_at` is checked.

**Reproducer.** `toy_backlog_early.py`: `seed 1: awaiting=2 oracle=1.0 premature_close_then_oracle=1.0`.

**Fix.** Keep `first_resolved_at` (set once), or a status-history table, and require that it is ≥ the reply time.

**Test.** That script asserts that the premature close scores below 1.

## V12 (P2): stages are missing for anchor tasks, and fixture Worlds leak temp dirs

**Defect.**
- `stages()` looks tasks up only in `_MADE`, which only `generate()` fills. The packs' own `tasks()` anchors
  (`finance_ap/invoices/s1-n7@1`, `s2`, toy `HelpdeskTask`) get `None`, so stage metrics are missing for anchors in
  `bench run` and `build`.
- `fixtures()` builds `APTask(3)` (a `mkdtemp` pack), and `check_fixture` never calls `cleanup`. Every
  `pack check --level fixture` leaks one dir.
- `$TMPDIR` holds 183 `finance-ap-*` and 1,028 `backlog-pack-*` dirs, up to 2.2 MB each.

**Reproducer.** Inline probe: `stages for its own anchor tasks: [None, None]`, toy `[None]`. After
`check('packs/finance_ap', level='semantic')` and cleanup of the anchors, one `finance-ap-*` dir remains.

**Fix.** Register in `_MADE` from `APTask.__init__` (and toy `tasks()`). Have fixture cases carry a `cleanup`, or have
`check_fixture` call `getattr(task, "cleanup")`.

**Test.** `stages(t.id)` is not None for every `tasks()` anchor, and the `finance-ap-*` count is unchanged after a fixture
check.

## V13 (P2): toy helpdesk notes are unchecked

**Defect.** The `t*/note` predicates are `COUNT(*) ≥ 1`. A note of `"x"` on every ticket scores 1.0. The t3 note need not
mention the carrier trace that `escalation.md` requires ("Always add a note explaining why").

**Reproducer.** `toy_hacks.py`: `notes are 'x'  reward=1.0`.

**Fix.** Require a minimal grounded body: the order id plus a reason keyword (`refund`, `duplicate`, `trace` or
`transit`). Alternatively, make notes a rubric item and not a gate.

**Test.** `NEGATIVES["empty_notes"]` must fail `t1`.

## V14 (P2): coding fixtures cover the wrong task01 and no adversary

**Defect.** `studio.SEEDS` lists `task01_prims_omp`, which `seeds.SEEDS` marks `bench: False` ("superseded by task01v2").
It omits `task01v2_tessera_omp` (`bench: True`). The M14 doc's "oracle 1 / base 0 on all four seeds" therefore covers
a retired seed and skips a benchmarked one. The fixture set has only gold and empty cases. The template pack, by
contrast, requires near misses ("the grader must reject near misses, not only the empty episode").

**Fix.** Use `[k for k, v in seeds.SEEDS.items() if v["bench"]]` plus task01 if wanted, and add V4's stuffed case and a
recorded honest failure per seed at its recorded Φ.

**Test.** `{c["name"].split(":")[0] for c in fixtures()}` ⊇ the bench seeds, and every seed has a non-trivial negative.

---

## Fixture realism (c), summary

| Pack | Rules or controls | Fixture covers | Missing (known hacks or rules) |
|---|---|---|---|
| finance_ap | 5 controls + within-tolerance + 6 reason codes | each control approved (5), hold within tolerance, wrong reason, no-op, gold | precedence (V7), exact-at-tolerance (V6), duplicate variants and earlier-rejected (V8), bad int (V1), a fixed-threshold policy (V6), hold-instead-of-reject swaps, Φ = reward on breach (V5) |
| toy helpdesk | refund policy (4 rules), approvals, 3 ticket outcomes | 7 negatives + gold | forged approval with no request, which is the gold itself (V2); split refunds (V3); empty notes (V13); bad int (V1) |
| coding | the dense reward (milestones) | oracle 1, base 0 for 4 seeds | stuffed or destructive patch (V4), the bench seed task01v2 (V14), honest failures at their recorded Φ |
| _template | exact-content file | gold, no-op, wrong word, wrong file, no newline | adequate for its scope |

Probes: `fa_overflow`, `fa_overflow_bench`, `fa_phi`, `fa_scan`, `fa_kb`, `toy_hacks`, `toy_overflow`,
`toy_backlog_early`, `coding_stuff`, `coding_runs` and `coding_prs` (`.py` and `.out`) in `m14/h2/`.
