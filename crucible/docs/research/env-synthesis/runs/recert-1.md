# recert-1: seeding by default, and the stored pool re-certified under the fixed framework (2026-10-30)

Code: DESIGN §7.7 item 22. `factory/synth_wire.py` (seeds by tag by default), `synth/falsify/agents.py`, `synth/roles.py`,
`synth/pipeline.py`, `synth/falsify/base.py`, `synth/meta/rsi.py` (`archive_recheck(heldout=)`), `synth/recert.py`, CLI
`scripts/synth_recert.py`, tests `tests/unit/test_synth_recert.py`. Data (gitignored):
`crucible/data/synth/recert/20261030-125829-recert-1/` (`items.json`, `det/`, `paid/`, `certs/`, `manifest.recert-1.json`);
each new certificate is also written once beside its stored one as `<run>/tasks/tNN/certificate.recert-1.json`.

Models: ALT Nemotron-3-Super, gpt-oss-120b and Inkling-Small (v3 harness); ATK Nemotron-3-Super and gpt-oss-120b; panel
and TWIN panel Nemotron-3-Ultra and gpt-oss-120b; leak probe Inkling-Small; leak panel Nemotron-3-Ultra and gpt-oss-120b.
No Qwen model ran in any role, the student was never called, and no stored task was edited (every manifest row reloads
its task and checks the digest).

**Spend: USD 5.11 of the 8 cap** (pilot 0.28, the run 4.83). The shared Part 2 total went from 86.52 to 91.62, under the
150 set as `MILESTONE_CAP_USD`. Phase 1 (deterministic classes) cost nothing.

## Headline

| | tasks | certified | provisional | on hold | rejected |
|---|---|---|---|---|---|
| stored statuses | 195 | 103 | 14 | — | 78 |
| **re-certified (h = 0.2, the manifest)** | 195 | **4** | **90** | **10** | **91** |
| the same evidence without held-out mutants (h = 0, report only) | 195 | 66 | 30 | 10 | 89 |

The 195 tasks are the learnability agent's 89-task pool (`perfect-check3`; M4 and RSI-1 runs, every status), the
M4 runs' admitted tasks (9, all inside the 89), the frontier round's 2 certified tasks, and every admitted task of the
yield runs (104: yield-1/2/3, smoke-1, speed-par, speed-ser2).

**Of the 117 stored admissions (certified or provisional):**

| outcome | tasks | primary cause |
|---|---|---|
| holds (same status) | 14 | — |
| changes (certified → provisional) | 64 | held-out mutants 59; MUT n below 59 on the whole neighbourhood too (the M4 duplicate-count bug, RSI-1) 5 |
| held (→ on hold: undecided suspects) | 10 | seed 6 (re-sampled ALT/ATK attempts and votes the panel did not decide); new classes 4 (LEAK suspects) |
| falls (→ rejected) | 29 | new classes 18 (ORACLE 7, LEAK 5, TWIN 3, layered TAINT 3); ALT set 3 (no pass 2, a valid attempt the contract rejects 1); seed 3 (ATK shortcuts by re-sampled attackers 2, undecided ALT failures 1); archive 3 (legacy D6 traces, §2 issue 6); held-out mutants 2 (D9 survivors, §2 issue 7) |

**Of the 78 stored rejections:** 62 stay rejected and 16 rise to provisional. All 16 are finance_ap pool tasks (15 M4,
1 RSI-1) whose `ALT:no_pass` or open ALT suspects are gone: Super passes 10 of them (**ALT naming**: Super never ran in
M4 or RSI-1), gpt-oss or Inkling-Small on the v3 harness the other 6 (**ALT set**). None rises to certified: the 15 M4
Worlds (8-14 invoices) give MUT n < 59, and the RSI-1 dense task is certified only at h = 0.

By cause over the 117 stored admissions (any cause named, a task can have several): held-out mutants 87, new classes 22,
seed 10, other 9, ALT set 7. ALT naming shows only on rises (10), because Super's passes can only lift a task.

**The binding decision is the holdout.** On the 30-invoice finance_ap Worlds MUT has about 60 distinct effects; V5's
h = 0.2 shows 48-54 of them to the rounds, so U_MUT is 0.052-0.065 and the task is provisional, not certified. Without
the holdout 66 tasks are certified. The manifest carries both (`status`, `status_h0`) and the h each certificate used,
so the lead can choose; see §4.

## 1. Seeding by default (the framework fix)

Every sampled call path now seeds by tag:

| path | before | now |
|---|---|---|
| `WireClient` default | seed = JSON retry index (0) | `tag_seed(tag, attempt, salt)` |
| `SynthWire` default (`synth.py certify/propose`, falsifiers-2, perfect, RSI-2 wires) | seed 0 (RSI-2 set it by hand) | by tag, salted `<run dir>|<role>#<i>` |
| batch wire (`synth.py batch`) | by tag unless `--same-seed` | unchanged, plus the salt |
| worlds, compose-2, solver pool wires | by tag | by tag, plus the salt |
| student wire | by tag (`seed=tag`) | unchanged (empty salt: item 19's formula, so stored records stay comparable) |

Tags that collided across attempts and were fixed: one agent runner's attempts across CEGIS rounds, the final pass and
ATK's feedback rounds (`<tag>@r<k>`); `roles.ask` retries after the client's JSON retries ran out (`~a<k>`); a repair
asked again after a refused one (`repair:<id>#v<version>~<k>`); the pipeline proposer's batches (`propose:batch<b>`).
The salt separates two clients of one model on one tag (Ultra ×2) and two runs (this re-certification does not redraw
the stored runs' samples). Unit tests: k attempts of one role on one task get k seeds; the same tag in one run gives the
same seed, another run another; `ask` retries and repeated repairs draw new seeds; the solver pool's attempts and the
student's rollouts are independent; `--same-seed` restores seed = attempt.

## 2. Framework issues the real runs found, and their fixes

| # | found by | issue | fix |
|---|---|---|---|
| 1 | task brief (items 17, 19) | every model call on seed 0 unless a wire asked for seeds by tag | seeds by tag by default, salted per run and client (§1) |
| 2 | audit of call sites | one agent runner re-ran with one tag across rounds; `ask` retries and refused-then-retried repairs repeated their prompt on one tag | `@r<k>`, `~a<k>`, `#v<version>~<k>` (§1) |
| 3 | collect | an M4-era `cegis.json` says `rejected` for a task its run report calls provisional (MUT bound only): 3 pool tasks | the stored status is the run report's |
| 4 | collect | task ids repeat across batch runs (`a1/finance_ap.s0.k0-b0-0` is a different task in yield-1, -2, -3 and smoke-1: 19 ids) | items are keyed by source; each source's archived counterexamples are copied into its own archive |
| 5 | phase 1, archive regressions | an archived ALT/ATK counterexample "replayed" by running a NEW agent attempt (a model call and a fresh sample; a StopIteration with no agent of that name), never the archived trace: every earlier ALT/ATK regression was a re-sample | `_AgentFalsifier.replay` grades the recorded trace as a literal program |
| 6 | phase 1, 3 stored-certified tasks re-open archived D6 traces | the repair that closed a D6 (a valid alternative the contract rejected) changed the instruction, oracle and contract; the trace was judged valid under the OLD instruction, so it is no regression test of the new one | `Counterexample.of` records `instruction_sha`; a judged-valid trace replays only on versions with that instruction. The 3 legacy counterexamples have no sha and **fail closed** (rejected; a person can clear them) |
| 7 | phase 1 | `MUT-HO` confirms SwapEntity D9 survivors (the contract does not pin which invoice step s8 writes) on 2 stored-certified yield tasks, while MUT on the whole neighbourhood is clean: its 240-mutant cap samples about 240 of 390 mutants and never drew them | **open** (reported, not fixed): MUT's cap sampling can miss a whole operator site; changing it changes RSI-2's preregistered numbers |
| 8 | the pilot (4 tasks, Ultra v2 + Super v3 + gpt-oss v3) | USD 0.062 per task, Ultra 53% of it: the pool would not fit the cap | ALT = compose-2's v3 set (Super, gpt-oss, Inkling-Small); the 4 pilot rows are kept aside (`paid-pilot-ultra-rows/`) and were re-run |
| 9 | the pilot | the leak panel had 6 anchors and stayed uncalibrated (λ > 1), so every LEAK quote stayed open | none needed: the full run calibrated on 40 anchors (λ 0.7, 40 of 40 right) |
| 10 | attribution | `refine`'s last regression re-reads a counterexample the same run just archived (`archive:reopened` next to the confirmed finding) | attribution ignores it; `cegis` unchanged (shared behaviour) |
| 11 | lead (mid-task) | `TaskLoop`'s `archive_recheck` had no held-out wiring | `archive_recheck(..., heldout=h)` runs `population(h)`; tested |

Live calibrations (first live run of compose-2's corrected TWIN anchors): panel λ 0.7 (finance_ap 11 of 11, it_helpdesk
10 of 10 decided anchors right), TWIN panel λ 0.5 (18 of 18 and 17 of 17), leak panel λ 0.7 (40 of 40). Panel decisions
in phase 2: ALT failures 124 invalid, 3 valid (D6), 10 abstain; ATK passes 12 valid, 2 invalid (shortcuts), 3 abstain;
MUT suspects 15 valid, 8 abstain; TWIN suspects 3 valid. 21 items went to the human queue, 10 to the leak queue.

## 3. What the fixed framework finds on stored admissions

- **ORACLE (item 21), 7 stored-certified tasks fall**: the reference decision contradicts the finance_ap pack's own
  policy. Exactly the 6 yield-1 §5 found by hand (5 of `a0` over yield-1/2/3, 1 of `a1_lemmas`), plus one frontier-round
  task. No `a1` task falls on ORACLE. It also confirms on 16 stored rejections.
- **LEAK (item 20 L2) in admission**: 4 confirmed give-aways (3 it_helpdesk, 1 finance_ap) and 5 undecided quotes
  (panel splits: 4 holds, 1 with an ALT failure too).
- **TWIN (item 18)**: 3 falls (the contract accepts the twin's solution).
- **Layered TAINT (item 20)**: 3 falls (`sep` 1, `lemma` 2).
- **Re-sampled attackers (seed)**: 2 confirmed ATK shortcuts (falls), 3 undecided ATK passes (holds).
- **ALT set**: 6 stored-certified tasks get no pass from Super/gpt-oss/Inkling-Small v3 (for 2 it is the primary cause);
  2 ALT attempts were judged valid and rejected by the contract (D6, one task).
- Phase 1 alone (CPU): 68 of 195 rejected, 18 of them stored admissions; 127 went to phase 2.

ALT passes in phase 2 (127 tasks): gpt-oss v3 86, Super v3 79, Inkling-Small v3 79.

## 4. The holdout decision (for the lead)

V5 says the holdout is off by default because it lowers MUT's n on small neighbourhoods. On this pool that is the main
effect: 59 stored-certified tasks are provisional only because 20% of their about 60 distinct mutants are held out (their
whole neighbourhood gives n 60-65 and U 0.045-0.049). The manifest uses V5's own share (h = 0.2, `HOLDOUT`) because
`TaskLoop` will repair these tasks, and the held-out set is what keeps a repair from fitting the shown mutants. The
held-out class also found 2 real D9 survivors that the capped whole-neighbourhood sample missed (§2 issue 7). Two ways
to certify more of the pool under h = 0.2 without changing the rule: denser Worlds (more invoices, so n ≥ 59 on the
shown share), or counting the held-out trials in U once a version is final (a rule change: not made).

## 5. The pool manifest

`crucible/data/synth/recert/20261030-125829-recert-1/manifest.recert-1.json` (`schema crucible.synth.recert/v1`,
`synth.recert.load_manifest`): one row per task with key, task id, version, digest, domain, source, kinds, the stored
`cegis.json` that holds the version (`task_path`, reloaded with a digest check), old status and reasons, new status and
reasons, `status_h0`, the new certificate (beside the old one), h, outcome, causes and ALT passes per agent. For
`TaskLoop`: 94 admitted rows (4 certified, 90 provisional), 10 on hold for a person, 91 rejected with their reasons.

## 6. Per task (old → new; h = 0 is the same evidence without held-out mutants)

Primary cause: the first cause of a change (empty when nothing changed). Reasons are cut to four.

| # | source | task@version | kinds | old | new | h=0 | outcome | new reasons | primary cause |
|---|---|---|---|---|---|---|---|---|---|
| 1 | pool-finance_ap | 20261029-143755-smoke/t01@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `archive:reopened=1`, `MUT:upper=0.1391>0.05` |  |
| 2 | pool-finance_ap | 20261029-143755-smoke/t02@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=2`, `archive:reopened=2`, `MUT:upper=0.1391>0.05` |  |
| 3 | pool-finance_ap | 20261029-144119-smoke2/t01@v1 | pool,m4 | certified | **provisional** | provisional | changes | `MUT:upper=0.1533>0.05` | other |
| 4 | pool-finance_ap | 20261029-144119-smoke2/t02@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1459>0.05` | alt_naming |
| 5 | pool-finance_ap | 20261029-144119-smoke2/t03@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1459>0.05` | alt_naming |
| 6 | pool-finance_ap | 20261029-152544-full/t01@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1459>0.05` | alt_naming |
| 7 | pool-finance_ap | 20261029-152544-full/t02@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1533>0.05` | alt_naming |
| 8 | pool-finance_ap | 20261029-152544-full/t03@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1616>0.05` | alt_naming |
| 9 | pool-finance_ap | 20261029-152544-full/t04@v1 | pool,m4 | certified | **provisional** | provisional | changes | `MUT:upper=0.1459>0.05` | other |
| 10 | pool-finance_ap | 20261029-152544-full/t05@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1533>0.05` | alt_set |
| 11 | pool-finance_ap | 20261029-152544-full/t06@v3 | pool | rejected | **rejected** | rejected | holds | `RED:confirmed=2`, `TWIN:confirmed=2`, `archive:reopened=5`, `MUT:upper=0.1273>0.05` |  |
| 12 | pool-finance_ap | 20261029-152544-full/t07@v2 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `MUT:upper=0.0722>0.05` |  |
| 13 | pool-finance_ap | 20261029-152544-full/t08@v4 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=2`, `archive:reopened=1`, `MUT:upper=0.1391>0.05` |  |
| 14 | pool-finance_ap | 20261029-152544-full/t09@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `archive:reopened=1`, `MUT:upper=0.0894>0.05` |  |
| 15 | pool-finance_ap | 20261029-152544-full/t10@v1 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `MUT:upper=0.0868>0.05` |  |
| 16 | pool-finance_ap | 20261029-152544-full/t13@v4 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `archive:reopened=1`, `MUT:upper=0.1616>0.05` |  |
| 17 | pool-finance_ap | 20261029-152544-full/t14@v1 | pool | rejected | **rejected** | rejected | holds | `LEAK:confirmed=1`, `ALT:no_pass`, `archive:reopened=1`, `MUT:upper=0.1707>0.05` |  |
| 18 | pool-finance_ap | 20261029-152544-full/t16@v1 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `MUT:upper=0.1926>0.05` |  |
| 19 | pool-finance_ap | 20261029-152544-full/t17@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1926>0.05` | alt_naming |
| 20 | pool-finance_ap | 20261029-152544-full/t18@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.2209>0.05` | alt_naming |
| 21 | pool-finance_ap | 20261029-152544-full/t19@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1707>0.05` | alt_set |
| 22 | pool-finance_ap | 20261029-152544-full/t20@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1810>0.05` | alt_set |
| 23 | pool-finance_ap | 20261029-160300-full2/t01@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.1616>0.05` |  |
| 24 | pool-finance_ap | 20261029-160300-full2/t02@v1 | pool,m4 | certified | **on_hold** | on_hold | held | `ALT:suspects=2`, `MUT:upper=0.1533>0.05` | seed |
| 25 | pool-finance_ap | 20261029-160300-full2/t03@v1 | pool,m4 | provisional | **provisional** | provisional | holds | `MUT:upper=0.1533>0.05` |  |
| 26 | pool-finance_ap | 20261029-160300-full2/t04@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.1616>0.05` |  |
| 27 | pool-finance_ap | 20261029-160300-full2/t05@v1 | pool,m4 | provisional | **provisional** | provisional | holds | `MUT:upper=0.1459>0.05` |  |
| 28 | pool-finance_ap | 20261029-160300-full2/t06@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `ORACLE:confirmed=1`, `archive:reopened=1`, `MUT:upper=0.1088>0.05` |  |
| 29 | pool-finance_ap | 20261029-160300-full2/t07@v1 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `LEAK:suspects=1`, `MUT:upper=0.0570>0.05` |  |
| 30 | pool-finance_ap | 20261029-160300-full2/t08@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.1050>0.05` |  |
| 31 | pool-finance_ap | 20261029-160300-full2/t09@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1173>0.05` | alt_set |
| 32 | pool-finance_ap | 20261029-160300-full2/t10@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.0688>0.05` |  |
| 33 | pool-finance_ap | 20261029-160300-full2/t11@v1 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `MUT:upper=0.1810>0.05` |  |
| 34 | pool-finance_ap | 20261029-160300-full2/t12@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1810>0.05` | alt_naming |
| 35 | pool-finance_ap | 20261029-160300-full2/t13@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.1810>0.05` |  |
| 36 | pool-finance_ap | 20261029-160300-full2/t14@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.1926>0.05` |  |
| 37 | pool-finance_ap | 20261029-160300-full2/t15@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.1533>0.05` |  |
| 38 | pool-finance_ap | 20261029-160300-full2/t16@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1707>0.05` | alt_set |
| 39 | pool-finance_ap | 20261029-160300-full2/t17@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1459>0.05` | alt_set |
| 40 | pool-finance_ap | 20261029-160300-full2/t18@v1 | pool | rejected | **provisional** | provisional | rises | `MUT:upper=0.1533>0.05` | alt_naming |
| 41 | pool-finance_ap | 20261029-160300-full2/t19@v1 | pool,m4 | certified | **provisional** | provisional | changes | `MUT:upper=0.1459>0.05` | other |
| 42 | pool-finance_ap | 20261029-160300-full2/t20@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.1616>0.05` |  |
| 43 | pool-finance_ap | 20261029-165857-rsi-dense/t01@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.1173>0.05` |  |
| 44 | pool-finance_ap | 20261029-165857-rsi-dense/t02@v1 | pool | certified | **provisional** | certified | changes | `MUT:upper=0.0540>0.05` | heldout |
| 45 | pool-finance_ap | 20261029-165857-rsi-dense/t03@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.0630>0.05` |  |
| 46 | pool-finance_ap | 20261029-165857-rsi-dense/t04@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.0644>0.05` |  |
| 47 | pool-finance_ap | 20261029-165857-rsi-dense/t05@v4 | pool | rejected | **rejected** | rejected | holds | `MUT:confirmed=26`, `RED:confirmed=1`, `MUT-HO:confirmed=11`, `MUT:upper=0.6213>0.05` … |  |
| 48 | pool-finance_ap | 20261029-165857-rsi-dense/t06@v1 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `MUT:upper=0.0560>0.05` |  |
| 49 | pool-finance_ap | 20261029-165857-rsi-dense/t07@v2 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.0618>0.05` |  |
| 50 | pool-finance_ap | 20261029-165857-rsi-dense/t08@v1 | pool | rejected | **provisional** | certified | rises | `MUT:upper=0.0540>0.05` | alt_naming |
| 51 | pool-finance_ap | 20261029-165857-rsi-dense/t09@v3 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.0618>0.05` |  |
| 52 | pool-finance_ap | 20261029-165857-rsi-dense/t10@v1 | pool | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 53 | pool-finance_ap | 20261029-165857-rsi-dense/t11@v2 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.0593>0.05` |  |
| 54 | pool-finance_ap | 20261029-165857-rsi-dense/t12@v1 | pool | rejected | **rejected** | rejected | holds | `ORACLE:confirmed=1`, `MUT:upper=0.0673>0.05` |  |
| 55 | pool-it_helpdesk | 20261029-150517-smoke2/t01@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `archive:reopened=1` |  |
| 56 | pool-it_helpdesk | 20261029-151753-smoke3/t01@v4 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `archive:reopened=1`, `MUT:upper=0.1391>0.05` |  |
| 57 | pool-it_helpdesk | 20261029-151753-smoke3/t02@v4 | pool | rejected | **rejected** | rejected | holds | `MUT:confirmed=9`, `MUT-HO:confirmed=1`, `MUT:upper=0.5095>0.05`, `archive:reopened=8` |  |
| 58 | pool-it_helpdesk | 20261029-151753-smoke3/t03@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `archive:reopened=1` |  |
| 59 | pool-it_helpdesk | 20261029-160257-full2/t01@v1 | pool,m4 | certified | **provisional** | provisional | changes | `MUT:upper=0.1015>0.05` | other |
| 60 | pool-it_helpdesk | 20261029-160257-full2/t02@v3 | pool,m4 | provisional | **rejected** | rejected | falls | `LEAK:confirmed=1`, `ALT:suspects=1`, `archive:reopened=1`, `MUT:upper=0.1329>0.05` | new_classes |
| 61 | pool-it_helpdesk | 20261029-160257-full2/t03@v1 | pool | rejected | **rejected** | rejected | holds | `TWIN:confirmed=2`, `archive:reopened=2`, `MUT:upper=0.1533>0.05` |  |
| 62 | pool-it_helpdesk | 20261029-160257-full2/t04@v3 | pool | rejected | **rejected** | rejected | holds | `MUT:confirmed=7`, `RED:confirmed=3`, `TWIN:confirmed=2`, `MUT-HO:confirmed=1` … |  |
| 63 | pool-it_helpdesk | 20261029-160257-full2/t05@v1 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `MUT:upper=0.1391>0.05` |  |
| 64 | pool-it_helpdesk | 20261029-160257-full2/t06@v3 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `MUT:suspects=2` |  |
| 65 | pool-it_helpdesk | 20261029-160257-full2/t07@v4 | pool | rejected | **rejected** | rejected | holds | `RED:confirmed=1`, `archive:reopened=1`, `MUT:upper=0.0981>0.05` |  |
| 66 | pool-it_helpdesk | 20261029-160257-full2/t08@v1 | pool | rejected | **rejected** | rejected | holds | `TWIN:confirmed=1`, `MUT:upper=0.0894>0.05` |  |
| 67 | pool-it_helpdesk | 20261029-160257-full2/t09@v4 | pool | rejected | **rejected** | rejected | holds | `MUT:confirmed=8`, `RED:confirmed=3`, `TWIN:confirmed=1`, `MUT-HO:confirmed=1` … |  |
| 68 | pool-it_helpdesk | 20261029-160257-full2/t11@v1 | pool | rejected | **rejected** | rejected | holds | `LEAK:confirmed=1`, `ALT:suspects=2`, `MUT:suspects=1`, `MUT-HO:suspects=1` … |  |
| 69 | pool-it_helpdesk | 20261029-160257-full2/t12@v2 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=2`, `archive:reopened=2` |  |
| 70 | pool-it_helpdesk | 20261029-160257-full2/t13@v1 | pool,m4 | certified | **provisional** | provisional | changes | `MUT:upper=0.0739>0.05` | other |
| 71 | pool-it_helpdesk | 20261029-160257-full2/t14@v4 | pool | rejected | **rejected** | rejected | holds | `RED:confirmed=2`, `TWIN:confirmed=2`, `archive:reopened=2` |  |
| 72 | pool-it_helpdesk | 20261029-160257-full2/t15@v4 | pool | rejected | **rejected** | rejected | holds | `MUT:confirmed=1`, `RED:confirmed=3`, `TAINT:confirmed=1`, `TWIN:confirmed=1` … |  |
| 73 | pool-it_helpdesk | 20261029-160257-full2/t16@v1 | pool | rejected | **rejected** | rejected | holds | `TWIN:confirmed=3`, `archive:reopened=3`, `MUT:upper=0.0798>0.05` |  |
| 74 | pool-it_helpdesk | 20261029-160257-full2/t17@v2 | pool | rejected | **rejected** | rejected | holds | `MUT:confirmed=1`, `NULL:confirmed=5`, `RED:confirmed=4`, `TAINT:confirmed=1` … |  |
| 75 | pool-it_helpdesk | 20261029-160257-full2/t18@v1 | pool | rejected | **rejected** | rejected | holds | `NULL:confirmed=9`, `RED:confirmed=1`, `TWIN:confirmed=2`, `archive:reopened=12` … |  |
| 76 | pool-it_helpdesk | 20261029-160257-full2/t19@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `archive:reopened=1` |  |
| 77 | pool-it_helpdesk | 20261029-160257-full2/t20@v2 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `MUT:upper=0.1329>0.05` |  |
| 78 | pool-it_helpdesk | 20261029-165852-rsi-dense/t01@v1 | pool | rejected | **rejected** | rejected | holds | `ALT:confirmed=1`, `MUT-HO:suspects=1`, `archive:reopened=1` |  |
| 79 | pool-it_helpdesk | 20261029-165852-rsi-dense/t02@v1 | pool | rejected | **rejected** | rejected | holds | `ALT:no_pass`, `MUT:suspects=1`, `MUT-HO:suspects=2` |  |
| 80 | pool-it_helpdesk | 20261029-165852-rsi-dense/t03@v1 | pool | rejected | **rejected** | rejected | holds | `TWIN:confirmed=2`, `archive:reopened=2`, `MUT:upper=0.0530>0.05` |  |
| 81 | pool-it_helpdesk | 20261029-165852-rsi-dense/t04@v3 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `TWIN:confirmed=2`, `archive:reopened=2` |  |
| 82 | pool-it_helpdesk | 20261029-165852-rsi-dense/t05@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `TWIN:confirmed=4`, `archive:reopened=6` |  |
| 83 | pool-it_helpdesk | 20261029-165852-rsi-dense/t06@v3 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=2`, `archive:reopened=1`, `MUT:upper=0.1459>0.05` |  |
| 84 | pool-it_helpdesk | 20261029-165852-rsi-dense/t07@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `TWIN:confirmed=2`, `archive:reopened=3` |  |
| 85 | pool-it_helpdesk | 20261029-165852-rsi-dense/t08@v3 | pool | rejected | **rejected** | rejected | holds | `MUT:confirmed=1`, `NULL:confirmed=1`, `RED:confirmed=1`, `TAINT:confirmed=1` … |  |
| 86 | pool-it_helpdesk | 20261029-165852-rsi-dense/t09@v4 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `TWIN:confirmed=1`, `archive:reopened=1` |  |
| 87 | pool-it_helpdesk | 20261029-165852-rsi-dense/t10@v2 | pool | rejected | **rejected** | rejected | holds | `TWIN:confirmed=2`, `archive:reopened=1` |  |
| 88 | pool-it_helpdesk | 20261029-165852-rsi-dense/t11@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `TWIN:confirmed=2`, `archive:reopened=3`, `MUT:upper=0.0512>0.05` |  |
| 89 | pool-it_helpdesk | 20261029-165852-rsi-dense/t12@v1 | pool | rejected | **rejected** | rejected | holds | `TAINT:confirmed=1`, `TWIN:confirmed=1`, `archive:reopened=2` |  |
| 90 | frontier | 20261029-232517-perfect-frontier-control/t02@v1 | frontier | certified | **provisional** | certified | changes | `MUT:upper=0.0560>0.05` | heldout |
| 91 | frontier | 20261029-232517-perfect-frontier-targeted/t04@v1 | frontier | certified | **rejected** | rejected | falls | `ORACLE:confirmed=1`, `MUT:upper=0.0644>0.05` | new_classes |
| 92 | smoke-1/a0 | a0/finance_ap.s0.k0-b1-1@v4 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0582>0.05` | heldout |
| 93 | smoke-1/a1 | a1/finance_ap.s0.k0-b0-0@v2 | yield | certified | **on_hold** | on_hold | held | `ALT:suspects=2` | seed |
| 94 | smoke-1/a1 | a1/finance_ap.s0.k0-b1-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 95 | smoke-1/a1 | a1/finance_ap.s0.k0-b1-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 96 | smoke-1/a1 | a1/finance_ap.s0.k1-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0618>0.05` | heldout |
| 97 | speed-par/a1 | a1/finance_ap.s0.k1-b0-2@v3 | yield | certified | **certified** | certified | holds |  |  |
| 98 | speed-ser2/a1 | a1/finance_ap.s0.k0-b0-1@v1 | yield | provisional | **on_hold** | on_hold | held | `ATK:suspects=1`, `MUT:upper=0.1129>0.05` | seed |
| 99 | yield-1/a0 | a0/finance_ap.s0.k0-b0-1@v1 | yield | certified | **rejected** | rejected | falls | `ATK:confirmed=1`, `archive:reopened=1`, `MUT:upper=0.0593>0.05` | seed |
| 100 | yield-1/a0 | a0/finance_ap.s0.k0-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0560>0.05` | heldout |
| 101 | yield-1/a0 | a0/finance_ap.s0.k1-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 102 | yield-1/a0 | a0/finance_ap.s1.k0-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0540>0.05` | heldout |
| 103 | yield-1/a0 | a0/finance_ap.s1.k1-b0-2@v1 | yield | certified | **rejected** | rejected | falls | `ORACLE:confirmed=1`, `MUT:upper=0.0593>0.05` | new_classes |
| 104 | yield-1/a0 | a0/finance_ap.s1.k2-b0-2@v2 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0530>0.05` | heldout |
| 105 | yield-1/a0 | a0/finance_ap.s2.k1-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0618>0.05` | heldout |
| 106 | yield-1/a0 | a0/finance_ap.s2.k1-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0570>0.05` | heldout |
| 107 | yield-1/a0 | a0/finance_ap.s2.k1-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 108 | yield-1/a0 | a0/finance_ap.s2.k2-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0618>0.05` | heldout |
| 109 | yield-1/a0 | a0/finance_ap.s3.k0-b1-0@v1 | yield | certified | **rejected** | rejected | falls | `ORACLE:confirmed=1`, `MUT:upper=0.0521>0.05` | new_classes |
| 110 | yield-1/a0 | a0/finance_ap.s3.k1-b0-2@v1 | yield | provisional | **provisional** | provisional | holds | `MUT:upper=0.0705>0.05` |  |
| 111 | yield-1/a0 | a0/finance_ap.s3.k2-b0-2@v1 | yield | provisional | **provisional** | provisional | holds | `MUT:upper=0.0894>0.05` |  |
| 112 | yield-1/a0 | a0/it_helpdesk.s0.k1-b0-2@v1 | yield | certified | **rejected** | rejected | falls | `LEAK:confirmed=1`, `archive:reopened=1` | new_classes |
| 113 | yield-1/a0 | a0/it_helpdesk.s0.k2-b0-2@v1 | yield | certified | **rejected** | rejected | falls | `LEAK:confirmed=1`, `ALT:no_pass`, `archive:reopened=1` | new_classes |
| 114 | yield-1/a1 | a1/finance_ap.s0.k0-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 115 | yield-1/a1 | a1/finance_ap.s0.k0-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 116 | yield-1/a1 | a1/finance_ap.s0.k0-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 117 | yield-1/a1 | a1/finance_ap.s0.k1-b0-1@v2 | yield | certified | **rejected** | rejected | falls | `TWIN:confirmed=1`, `MUT:upper=0.0618>0.05` | new_classes |
| 118 | yield-1/a1 | a1/finance_ap.s0.k1-b0-2@v3 | yield | certified | **certified** | certified | holds |  |  |
| 119 | yield-1/a1 | a1/finance_ap.s0.k2-b0-1@v1 | yield | certified | **on_hold** | on_hold | held | `ATK:suspects=1`, `MUT:upper=0.0582>0.05` | seed |
| 120 | yield-1/a1 | a1/finance_ap.s1.k0-b0-0@v1 | yield | certified | **rejected** | rejected | falls | `TAINT:confirmed=1`, `MUT:upper=0.0521>0.05` | new_classes |
| 121 | yield-1/a1 | a1/finance_ap.s1.k0-b0-1@v1 | yield | certified | **on_hold** | on_hold | held | `LEAK:suspects=1`, `MUT:upper=0.0570>0.05` | new_classes |
| 122 | yield-1/a1 | a1/finance_ap.s1.k1-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0582>0.05` | heldout |
| 123 | yield-1/a1 | a1/finance_ap.s1.k1-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 124 | yield-1/a1 | a1/finance_ap.s1.k3-b0-0@v3 | yield | certified | **rejected** | rejected | falls | `TWIN:confirmed=2`, `MUT:upper=0.0593>0.05` | new_classes |
| 125 | yield-1/a1 | a1/finance_ap.s1.k3-b0-1@v1 | yield | certified | **on_hold** | on_hold | held | `LEAK:suspects=1`, `MUT:upper=0.0593>0.05` | new_classes |
| 126 | yield-1/a1 | a1/finance_ap.s1.k3-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0630>0.05` | heldout |
| 127 | yield-1/a1 | a1/finance_ap.s2.k0-b0-1@v2 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0582>0.05` | heldout |
| 128 | yield-1/a1 | a1/finance_ap.s2.k1-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0550>0.05` | heldout |
| 129 | yield-1/a1 | a1/finance_ap.s2.k2-b0-0@v3 | yield | certified | **rejected** | rejected | falls | `TAINT:confirmed=1`, `MUT:upper=0.0550>0.05` | new_classes |
| 130 | yield-1/a1 | a1/finance_ap.s2.k2-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0570>0.05` | heldout |
| 131 | yield-1/a1 | a1/finance_ap.s2.k2-b0-2@v3 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0521>0.05` | heldout |
| 132 | yield-1/a1 | a1/finance_ap.s3.k0-b0-2@v3 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0618>0.05` | heldout |
| 133 | yield-1/a1 | a1/finance_ap.s3.k1-b0-1@v2 | yield | provisional | **provisional** | provisional | holds | `MUT:upper=0.0894>0.05` |  |
| 134 | yield-1/a1 | a1/finance_ap.s3.k3-b0-0@v3 | yield | certified | **rejected** | certified | falls | `MUT-HO:confirmed=4` | heldout |
| 135 | yield-1/a1 | a1/it_helpdesk.s0.k1-b0-2@v1 | yield | certified | **on_hold** | on_hold | held | `LEAK:suspects=1` | new_classes |
| 136 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s0.k0-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0540>0.05` | heldout |
| 137 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s0.k0-b0-1@v1 | yield | certified | **rejected** | rejected | falls | `ORACLE:confirmed=1`, `MUT:upper=0.0582>0.05` | new_classes |
| 138 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s0.k3-b0-0@v1 | yield | certified | **rejected** | rejected | falls | `ALT:no_pass`, `MUT:upper=0.0605>0.05` | alt_set |
| 139 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s0.k3-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0658>0.05` | heldout |
| 140 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s1.k0-b0-2@v3 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0593>0.05` | heldout |
| 141 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s1.k1-b0-1@v2 | yield | certified | **rejected** | rejected | falls | `ALT:no_pass`, `MUT:upper=0.0540>0.05` | alt_set |
| 142 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s1.k3-b0-2@v1 | yield | certified | **rejected** | rejected | falls | `ALT:no_pass`, `LEAK:suspects=1` | new_classes |
| 143 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s2.k0-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0582>0.05` | heldout |
| 144 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s2.k1-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0530>0.05` | heldout |
| 145 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s2.k2-b0-0@v1 | yield | certified | **rejected** | rejected | falls | `ALT:no_pass`, `ALT:suspects=2`, `MUT:upper=0.0560>0.05` | seed |
| 146 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s2.k2-b0-1@v4 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0530>0.05` | heldout |
| 147 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s2.k3-b0-0@v2 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0570>0.05` | heldout |
| 148 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s2.k3-b0-2@v3 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0593>0.05` | heldout |
| 149 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s3.k0-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0582>0.05` | heldout |
| 150 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s3.k0-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0722>0.05` | heldout |
| 151 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s3.k1-b0-2@v2 | yield | certified | **rejected** | rejected | falls | `archive:reopened=2`, `MUT:upper=0.0530>0.05` | other |
| 152 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s3.k2-b0-0@v3 | yield | certified | **rejected** | rejected | falls | `TWIN:confirmed=1`, `MUT:upper=0.0530>0.05` | new_classes |
| 153 | yield-1/a1_lemmas | a1_lemmas/finance_ap.s3.k3-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0570>0.05` | heldout |
| 154 | yield-1/a1_lemmas | a1_lemmas/it_helpdesk.s0.k2-b0-1@v2 | yield | certified | **certified** | certified | holds |  |  |
| 155 | yield-1/a1_lemmas | a1_lemmas/it_helpdesk.s0.k3-b0-2@v1 | yield | certified | **rejected** | rejected | falls | `TAINT:confirmed=1` | new_classes |
| 156 | yield-2/a0 | a0/finance_ap.s0.k3-b0-0@v1 | yield | certified | **rejected** | rejected | falls | `LEAK:confirmed=1`, `ALT:no_pass`, `archive:reopened=1`, `MUT:upper=0.0605>0.05` | new_classes |
| 157 | yield-2/a0 | a0/finance_ap.s1.k2-b0-2@v1 | yield | certified | **rejected** | rejected | falls | `ORACLE:confirmed=1`, `MUT:upper=0.0582>0.05` | new_classes |
| 158 | yield-2/a0 | a0/finance_ap.s2.k2-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0570>0.05` | heldout |
| 159 | yield-2/a1 | a1/finance_ap.s0.k0-b0-0@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 160 | yield-2/a1 | a1/finance_ap.s0.k0-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0570>0.05` | heldout |
| 161 | yield-2/a1 | a1/finance_ap.s0.k0-b0-2@v1 | yield | provisional | **provisional** | provisional | holds | `MUT:upper=0.1129>0.05` |  |
| 162 | yield-2/a1 | a1/finance_ap.s0.k1-b0-1@v4 | yield | provisional | **rejected** | rejected | falls | `archive:reopened=1`, `MUT:upper=0.1050>0.05` | other |
| 163 | yield-2/a1 | a1/finance_ap.s1.k2-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0618>0.05` | heldout |
| 164 | yield-2/a1 | a1/finance_ap.s1.k2-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0540>0.05` | heldout |
| 165 | yield-2/a1 | a1/finance_ap.s2.k0-b0-1@v1 | yield | certified | **on_hold** | on_hold | held | `ALT:suspects=1`, `MUT:upper=0.0550>0.05` | seed |
| 166 | yield-2/a1 | a1/finance_ap.s2.k1-b0-2@v2 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 167 | yield-2/a1 | a1/finance_ap.s2.k2-b0-1@v2 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0540>0.05` | heldout |
| 168 | yield-2/a1 | a1/finance_ap.s2.k2-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0582>0.05` | heldout |
| 169 | yield-2/a1 | a1/finance_ap.s3.k0-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0570>0.05` | heldout |
| 170 | yield-2/a1 | a1/finance_ap.s3.k0-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0593>0.05` | heldout |
| 171 | yield-2/a1 | a1/finance_ap.s3.k3-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0540>0.05` | heldout |
| 172 | yield-3/a0 | a0/finance_ap.s0.k0-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0605>0.05` | heldout |
| 173 | yield-3/a0 | a0/finance_ap.s0.k0-b0-2@v1 | yield | certified | **rejected** | rejected | falls | `ORACLE:confirmed=1`, `MUT:upper=0.0593>0.05` | new_classes |
| 174 | yield-3/a0 | a0/finance_ap.s1.k0-b0-1@v1 | yield | certified | **rejected** | rejected | falls | `ORACLE:confirmed=1`, `MUT:upper=0.0605>0.05` | new_classes |
| 175 | yield-3/a0 | a0/finance_ap.s1.k0-b0-2@v2 | yield | certified | **rejected** | rejected | falls | `ALT:confirmed=2`, `archive:reopened=2`, `MUT:upper=0.0530>0.05` | alt_set |
| 176 | yield-3/a0 | a0/finance_ap.s2.k2-b0-0@v1 | yield | provisional | **rejected** | rejected | falls | `ATK:confirmed=1`, `archive:reopened=1`, `MUT:upper=0.3930>0.05` | seed |
| 177 | yield-3/a0 | a0/finance_ap.s2.k2-b0-1@v1 | yield | provisional | **provisional** | provisional | holds | `MUT:upper=0.3123>0.05` |  |
| 178 | yield-3/a0 | a0/finance_ap.s2.k2-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0618>0.05` | heldout |
| 179 | yield-3/a0 | a0/finance_ap.s3.k0-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0688>0.05` | heldout |
| 180 | yield-3/a0 | a0/finance_ap.s3.k0-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0582>0.05` | heldout |
| 181 | yield-3/a1 | a1/finance_ap.s0.k0-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0560>0.05` | heldout |
| 182 | yield-3/a1 | a1/finance_ap.s0.k0-b0-2@v1 | yield | provisional | **provisional** | provisional | holds | `MUT:upper=0.1088>0.05` |  |
| 183 | yield-3/a1 | a1/finance_ap.s0.k1-b0-0@v1 | yield | certified | **on_hold** | on_hold | held | `ATK:suspects=1`, `MUT:upper=0.0582>0.05` | seed |
| 184 | yield-3/a1 | a1/finance_ap.s0.k1-b0-1@v3 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0644>0.05` | heldout |
| 185 | yield-3/a1 | a1/finance_ap.s0.k1-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0530>0.05` | heldout |
| 186 | yield-3/a1 | a1/finance_ap.s0.k2-b0-2@v1 | yield | provisional | **provisional** | provisional | holds | `MUT:upper=0.1533>0.05` |  |
| 187 | yield-3/a1 | a1/finance_ap.s0.k3-b0-0@v1 | yield | provisional | **provisional** | provisional | holds | `MUT:upper=0.0739>0.05` |  |
| 188 | yield-3/a1 | a1/finance_ap.s0.k3-b0-1@v2 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0582>0.05` | heldout |
| 189 | yield-3/a1 | a1/finance_ap.s0.k3-b0-2@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0644>0.05` | heldout |
| 190 | yield-3/a1 | a1/finance_ap.s1.k1-b0-0@v1 | yield | certified | **on_hold** | on_hold | held | `LEAK:suspects=1`, `MUT:upper=0.0673>0.05` | new_classes |
| 191 | yield-3/a1 | a1/finance_ap.s1.k1-b0-1@v3 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0560>0.05` | heldout |
| 192 | yield-3/a1 | a1/finance_ap.s2.k3-b0-0@v2 | yield | certified | **rejected** | rejected | falls | `archive:reopened=2`, `MUT:upper=0.0530>0.05` | other |
| 193 | yield-3/a1 | a1/finance_ap.s3.k0-b0-0@v2 | yield | certified | **rejected** | certified | falls | `MUT-HO:confirmed=7` | heldout |
| 194 | yield-3/a1 | a1/finance_ap.s3.k0-b0-1@v1 | yield | certified | **provisional** | certified | changes | `MUT:upper=0.0570>0.05` | heldout |
| 195 | yield-3/a1 | a1/finance_ap.s3.k0-b0-2@v4 | yield | certified | **certified** | certified | holds |  |  |
