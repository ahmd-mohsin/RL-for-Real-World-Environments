# compose-2: certified parts, per-part repair and stronger solvers for long composites (2026-10-30)

Code: DESIGN §7.7 item 18 (compose-2 block): `synth/parts.py`, `synth/solvers.py`, `synth/falsify/twinjudge.py`,
`synth/compose.py` (`close_undone`, `build_chain(certify=, start=)`, `certify_composed(parts=)`), `contract.Undone`,
harness v3 (`roles.ALT_SOLVER_V3`, `falsify.agents.llm_agent`). CLI `scripts/synth_compose2.py` (`parts | chain |
composites | solvers | budget`). Data (gitignored): `crucible/data/synth/{finance_ap,ar_collections,it_helpdesk}/*compose2*`,
`crucible/data/synth/compose2/` (solver pools, `offline-recheck-1.json`, `oracle-check-1.json`, `horizon-src/`).

Roles: the swarm built every task, link and instruction. DeepSeek-V4.1-Flash proposed, wrote oracles, repaired and wrote
instructions; GLM-5.3 wrote contracts. The panel was Nemotron-3-Ultra + gpt-oss-120b, the leak probe Inkling-Small, the
leak panel Nemotron-3-Ultra + gpt-oss. ALT was either the default (Nemotron-3-Ultra ×2, gpt-oss) or the v3 set (gpt-oss,
Nemotron-3-Super, Inkling-Small). The solver pool used Nemotron-3-Ultra, Nemotron-3-Super, gpt-oss-120b, Inkling and
Inkling-Small. No Qwen model ran in any role, and the student was never called.

**The ALT rule did not change.** A non-builder family must pass the whole composite from its instruction. The solver
sees the instruction, the tools and its own history, never the contract, the oracle or the intent card.

**Spend: USD 7.17 of the 8 cap** (receipts with "compose2" in the run path). The shared Part 2 total went from 79.32 to
about 86.5, under the 150 set as `MILESTONE_CAP_USD`.

## Headline

| long composites (≥ 20 calls) | built | certified | provisional | rejected | why |
|---|---|---|---|---|---|
| dep ≥ 1, from certified parts | **0** | 0 | 0 | 0 | no chain got a second link certified (0 of 7 follow-up links tried; table below) |
| dep 0 (horizon only), from certified parts | 2 | 0 | 0 | **2** | 23 calls: every class clean, ALT 3/3, MUT U = 0.048; held by **one open LEAK suspect** (a panel split, now in the human queue). 22 calls: every class clean, MUT U = 0.044; **ALT 0/3** |

- **No long composite with dep ≥ 1 is certified, provisional or even built.** Certifying parts first moved the
  bottleneck from the composite to the follow-up links. On their own World state those links fail certification:
  - TAINT D5: literals the earlier work produced, never read in the link;
  - ATK shortcuts;
  - TWIN suspects the strict panel will not decide;
  - ALT failures.
- **Certifying parts first worked where parts could be certified.** The two composites built from admitted parts have
  **0 confirmed counterexamples in every class**. So the per-part repair path never had a composite counterexample to repair in a real run. It is
  exercised by the unit tests only.
- (compose-1's composites carried 1 to 12 inherited confirmed counterexamples each: §2.)
- **The strongest solvers pass a 23-call composite, but not the 22-call four-part one.**
  - The 23-call composite (3 parts, 3 writes) was solved at equal spend by 5 of 6 pool configurations.
  - The 22-call composite (4 parts, 4 writes) failed 12 of 12 attempts at 1.5 × horizon + 6 calls. With 3 × horizon,
    gpt-oss and Inkling-Small still failed, stopping early on their own. Inkling used all 72 calls and decided 3 of 4
    invoices right; it failed on the fourth with a wrong decision.
  - ORACLE (item 21) confirms the composite's reference decisions are right under the pack's policy, so these are solver
    errors, not a wrong oracle.

## 1. Framework changes the runs forced (each with a test in `tests/unit/test_synth_compose2.py`)

| # | found by | issue | fix |
|---|---|---|---|
| 1 | offline re-check of compose-1's composites | restoration gap: it_helpdesk r1 c0 clears T-1006.known_incident_id in one part and restores it in another; a merged allowed pattern still permitted the cleared value (MUT DropStep D1) | `close_undone`: each merged pattern excludes the values the composite never leaves, both restored and superseded ones (`EffectPattern.unless`; dumped only when set, so other contracts keep their digest). The D1 is gone; nothing else changed (§2) |
| 2 | parts smoke (it_helpdesk) | TWIN calibration found 0 anchors: σ that re-keys a written entity is rare, and a ticket subject shared by two tickets is a selector | anchors from clean twins whose π*′ writes other entities than π* |
| 3 | chain-fin / chain-ar, first start | TWIN λ stayed above 1 (0 anchors), so every TWIN suspect stayed open and blocked the link | strict calibration over the panel's own executed-truth anchors (oracle, no-op, minus last write) plus the twin anchors |
| 4 | solver pool single-1 | Inkling made 0 calls in 12 of 12 attempts: it writes `{"name": <tool>, "args": {...}}`, which `AgentTurn` read as a finish | `AgentTurn` accepts `name`/`args` and `function`. A reply that is neither a call nor a finish gets a format note and does not end the attempt |
| 5 | solver pool single-2 | the run's cap refusal crashed `evaluate` and lost the summary | `evaluate` stops every configuration at the cap and keeps the finished rows; a cut attempt is no trial |
| 6 | chain-fin2 / chain-ar2, restart | link ids (`chain0-link01/t01`) carried no run id, so a restarted run re-opened the stopped run's archived counterexamples (`archive:reopened=1` on a fresh task) | link ids carry the run directory (`<run>-chain0-link01/t01`) |
| 7 | chain-fin5, resumed chain | the resuming run's env named a fresh work directory, so every new link "runs in another World" ($0.28 with no trial) | a resumed chain runs on its start links' base World |
| 8 | ORACLE wiring (main's item 21) | a confirmed ORACLE or LEAK finding did not reject: neither class was in `admission_policy().deterministic` | both added; a class that did not run is skipped |
| 9 | long pool, 22-call composite | most attempts ended at the 39-call budget (1.5 × 22 + 6) | `--per-call` for the pool and `certify_composed(alt_per_call=)`; tried at 3 × (§4) |
| 10 | the TWIN panel's calibration errors (§5) | `twin_solution` anchors were labelled valid on decision tasks, where π*′ carries π*'s decision to an entity whose right decision differs: the panel, correctly, called them invalid, so strict calibration refused | a `twin_solution` anchor only when the intent card derives nothing; the stored votes re-score to λ = 0.5 |

Issues 6 and 7 wasted about USD 0.6 of the stopped and resumed runs; those runs are kept as data and counted in no
result. The first two chain starts (`chain-fin`, `chain-ar`) are not counted either: issue 3 left every TWIN suspect
open there.

## 2. The restoration gap, deterministic re-check (no model)

compose-1's four stored composites were recomposed with compose-2's merge, keeping the stored writer text. The
deterministic population ran at MUT cap 12 × horizon (`compose2/offline-recheck-1.json`).

| composite | calls | restored | closed | MUT confirmed (compose-1 → compose-2) | other confirmed (unchanged) |
|---|---|---|---|---|---|
| it_helpdesk r1 c0 | 49 | T-1006.known_incident_id | that address (restored) | WrongAnswer D3 + **DropStep D1** → WrongAnswer D3 | TAINT D5 8, PERT D11 1 |
| it_helpdesk r1 c1 | 37 | T-1008.known_incident_id | none (no pattern covered the undone value: the frame rule already rejects it) | WrongAnswer D3 | TAINT D5 10, D7 1 |
| finance_ap r2 c0 | 46 | none | none | WrongAnswer D3 | TAINT D5 3 |
| finance_ap r2 c1 | 45 | none | none | WrongAnswer D3 | none |

These composites still cannot be certified, because their defects are their parts'. finance_ap r2 c1's first link pins
a decision the pack's policy does not reach (`finance_ap_truth`: wrong), so its composite is no valid ALT target either.

## 3. Certify the parts before composing

### 3.1 New chains, each link certified before it joins (`build_chain(certify=True)`, 2 candidates per round, ≤ 2 tried, 2 rounds)

| run | World | ALT | link 1 | link 2 | USD |
|---|---|---|---|---|---|
| chain-fin3 | finance_ap, 30 invoices, seed 1 | Ultra ×2 + gpt-oss (v2) | 0 of 4 (ALT no pass on 3; 3 of 4 oracles are right per the pack's policy) | — | 0.53 |
| chain-fin4 | same | gpt-oss + Super + Inkling-Small (v3) | **certified** on the 1st try (the same INV-89033 task that chain-fin3 rejected with ALT:no_pass) | 0 of 3 (ATK confirmed 2, open TWIN suspects, archive re-opens) | 0.53 |
| chain-fin6 (resumes fin4) | same | v3 set | (fin4's) | 0 of 3 (TAINT D5 confirmed in every candidate; repairs did not fix it in 2 rounds) | 0.40 |
| chain-ar3 | AR collections i01-dense | Ultra ×2 + gpt-oss (v2) | provisional (a 1-call figure report) after 3 rejections | 0 of 1 tried; 3 dead oracles | 0.92 |
| chain-it | it_helpdesk dense (40 tickets) | v3 set | 0 of 4 (TWIN D9 confirmed, ALT D6 confirmed, ALT no pass) | — | 0.34 |
| parts-smoke | it_helpdesk, compose-1 r1 chain 0 (stored links) | v2 | rejected: its required fields already hold at s₀ (vacuous state patterns); the repairer swung between two broken fixes and the monotone archive refused the swings | chain cut | 0.16 |

Swarm-built first links: 2 admitted out of 13 tried. Follow-up links: **0 admitted out of 7 tried**.

Why the follow-ups fail on their own World state:
- **TAINT D5.** The follow-up's oracle uses the id or number the earlier work produced instead of reading it, and the
  repairer did not turn the literal into a read in 2 rounds.
- **ATK shortcuts.** "Find the other invoice from that vendor" has an attacker pass.
- **TWIN suspects.** The strict panel does not decide them (§5).

### 3.2 Horizon-only chains: certified single tasks in sequence (`parts`, dep 0)

The tasks are swarm-built yield-1 `a1` finance_ap tasks with right oracles (`compose2/horizon-src`), taken in order on
one World instance. Each part is rebased onto the earlier parts and certified again with the v3 ALT set.

| chain | parts (calls) | outcome | USD |
|---|---|---|---|
| c0 | 6, 5, 7, 3, 7 | part 1 rejected: TAINT confirmed under item 20's layered matcher, which its yield-1 certificate predates; the chain was cut | 0.13 (with c1) |
| c1 | 7, 8, 8 = 23 | certified, provisional, provisional (MUT n 60, 58, 56; ALT 2/3, 2/3, 3/3) | (above) |
| c2 (c0 without its first part) | 5, 7, 3, 7 = 22 | certified, certified, provisional, provisional (ALT 3/3 on each) | 0.12 |

## 4. The long composites and the solver pool

**Certification** (`composites-horizon`, USD 0.34). Both composites were composed under the CertBook (every part
admitted for its exact version), and the writer wrote the instructions (2 attempts each). Certification ran with
per-part repair, inline TWIN adjudication (strict λ), the LEAK class and the v3 ALT set.

| composite | calls | writes | dep | REP RED TAINT NULL TWIN | MUT n (U) | ATK | ALT | LEAK | ORACLE | status |
|---|---|---|---|---|---|---|---|---|---|---|
| horizon c1 | 23 | 3 | 0 | all 0 confirmed | 61 (0.048) | 0/2 | **3/3** | 1 suspect | right | rejected: `LEAK:suspects=1` |
| horizon c2 | 22 | 4 | 0 | all 0 confirmed | 67 (0.044) | 0/2 | 0/3 | clean | right | rejected: `ALT:no_pass` |

The LEAK suspect is a split. On the sentence "Decide the pending invoice INV-50504 … and apply the decision in the
system", gpt-oss called the decision `approved` stated (confidence 1.0), and Nemotron called it clean (0.95). The quote
names no decision. The suspect is in `leak_queue.jsonl` for a person, and the rule is not overridden.

**Solver pool at equal spend on the two composites** (`solvers-long-1`). USD 0.12 per task and configuration, up to 3
attempts, call budget 1.5 × horizon + 6.

| config | c1 (23 calls) | c2 (22 calls) | USD/attempt | calls per attempt |
|---|---|---|---|---|
| Nemotron-3-Ultra v2 | **pass** (1st) | 0/2 | 0.095 | 34 |
| Nemotron-3-Ultra v3 | 0/1 (0 writes in 41 calls) | 0/1 | 0.144 | 40 |
| Inkling v3 | **pass** (1st) | 0/1 | 0.090 | 32 |
| Nemotron-3-Super v3 | **pass** (1st) | 0/2 | 0.063 | 34 |
| gpt-oss-120b v3 | **pass** (2nd) | 0/3 | 0.018 | 19 |
| Inkling-Small v3 | **pass** (1st) | 0/3 | 0.017 | 23 |

On c2 every attempt failed on missing required decisions (`MISSING_REQUIRED:t2.r0`, …), most of them at or near the
39-call budget. With 3 × horizon (72 calls, `solvers-long-2/3`):
- gpt-oss 0/2 and Inkling-Small 0/2 stopped early on their own (24 and 28 calls);
- Inkling (1 attempt, USD 0.30) used all 72 calls, wrote 4 decisions and got 3 right. It rejected invoice 117 as
  `duplicate`, and the contract requires another decision. ORACLE says the reference decision is right;
- the Ultra, Super and second Inkling attempts were cut by the run's cap and are no trials.

**Certified single tasks** (6 finance_ap yield-3 `a1` tasks with right oracles; USD 0.02 per task and config; `solvers-single-1/2`):

| config | pass@1 | solved at equal spend | USD/attempt | calls |
|---|---|---|---|---|
| Nemotron-3-Ultra v2 | 5/6 | 5/6 | 0.0125 | 7.3 |
| Nemotron-3-Ultra v3 | 5/6 | 5/6 | 0.0330 | 8.3 |
| Nemotron-3-Super v3 | 4/6 | 5/6 | 0.0073 | 7.8 |
| gpt-oss-120b v3 | **5/6** | 5/6 | **0.0031** | 7.1 |
| gpt-oss-120b v2 | **0/6** | 0/6 | 0.0019 | 1.8 (quits) |
| Inkling-Small v3 | 4/6 | 4/6 | 0.0043 | 7.0 |
| Inkling v3 / v2 (after fix 4; run cut by its cap) | 3/3 / 3/3 | — | 0.025 / 0.019 | 11–12 |

What the pool shows (the intervals are wide: 6 single tasks, 2 long ones):
- **The harness matters more than the model for cheap solvers.** v3 turns gpt-oss from 0/6 (it quits after about two
  calls) into 5/6 at a quarter of Ultra's cost per attempt. The progress check is the part that keeps it working.
  Certification with the v3 set admitted a finance_ap link that the default set (Ultra ×2 + gpt-oss v2) had rejected
  with ALT:no_pass.
- **v3 does not help Ultra.** It costs 2.6× per attempt on single tasks and is the only configuration that failed c1
  (0 writes in 41 calls: it reads and plans until the budget ends).
- **At equal spend the cheap solvers lead.** On c1 they pass at USD 0.07–0.09 per solved task, against 0.29 for Ultra v2.
  A certifying ALT set of gpt-oss, Super and Inkling-Small (v3) is about 10× cheaper than Ultra ×2 + gpt-oss (v2), and it
  passes at least as often.
- **The binding limit on long composites is per-decision accuracy over many decisions.** No solver passed the four-part
  composite. The best attempt decided 3 of 4 invoices right. At a per-part pass rate near 0.8, four independent
  decisions give about 0.8⁴ ≈ 0.4 per attempt at best. The measured 0 of 12 is below even that, because the attempts also
  run out of calls.

## 5. TWIN suspects and the strict calibration

TWIN suspects now go to the panel inline, judged against the twin task: π* when it passes C′, π*′ in the diverged case.
λ comes from item 20's strict `calibrate` (`min_decided` 6, `require_both`, `refuse_uncalibrated`).

| run | anchors (twin / panel) | decided at λ = 0.5 | correct | λ in the run |
|---|---|---|---|---|
| chain-fin4 link 1 | 6 / 6 | 12 | 10 (0.83) | **> 1** (uncalibrated) |
| composites-horizon | 9 / 9 | 14 | 11 (0.79) | **> 1** |

In the runs no TWIN suspect was decided: every one went to the human queue, and TWIN suspects blocked 9 link candidates.

**The calibration, not the panel, was wrong (issue 10).** Every decided error is a `twin_solution` anchor: 5 of 5 were
called invalid, and **the panel was right**.
- On a decision task, π*′ applies π*'s decision to the sibling invoice.
- The policy's decision for the sibling depends on that invoice's own records.
- So π*′ is no twin solution, and the anchor's "valid" label was false.

`twin_anchors` now emits a `twin_solution` anchor only when the intent card derives nothing. Re-scoring the stored votes
without those anchors (no model call):

| run | anchors | decided at λ = 0.5 | correct | strict λ |
|---|---|---|---|---|
| chain-fin4 link 1 | 10 | 10 | 10 | 0.5 |
| composites-horizon | 15 | 11 | 11 | 0.5 |

Re-deciding the stored TWIN suspect votes at λ = 0.5 (fin3, fin4, fin6, ar3, it, the horizon parts):
- **25 become confirmed D9**: the contract does not move with the literal, for example a figure's `as_of_date` the
  contract never checks;
- 1 is cleared;
- 8 stay open (the members split).

The blocked links would therefore still be rejected, but on confirmed counterexamples the repairer can act on, not on
open suspects.

## 6. The LEAK class on composites (item 20's L2)

The leak panel (Nemotron-3-Ultra + gpt-oss) was calibrated on anchors built from the certified parts: dev templates
with a derive value are leaks; the parts' own sentences, enumerations and restated rules are not. It reached λ = 0.7
(calibrated). Inkling-Small probed both composites for USD 0.0005:
- c2: clean;
- c1: 1 suspect, the split above.

Main's ORACLE class now runs with the population wherever the World registers policy truth (`synth_compose2.population`,
finance_ap only). On compose-2's tasks (`compose2/oracle-check-1.json`):
- **right:** 4 first parts and both composites;
- **not applicable:** 6 rebased parts. A prefix-variant World has no registered truth, and the pack's expected decision is
  for the base start state only.

## 7. Open

- **Follow-up links (dep ≥ 1) do not certify on their own World state** (0 of 7). The repairer does not turn an
  earlier result's literal into a read (TAINT D5). The next step is a follow-up oracle prompt that must read what the
  earlier work produced, or a deterministic pre-filter that drops follow-ups whose literals no read in the link shows.
- **The corrected TWIN calibration is measured on stored votes only** (10/10, 11/11 → λ = 0.5). A live run with the
  fixed anchors has not been made; there, most TWIN suspects become confirmed D9s for the repairer.
- **No solver passes a four-decision composite.** The pool's best attempt gets 3 of 4. Admission of longer composites
  under the unchanged rule needs a solver whose per-decision accuracy is well above 0.9.
- **Per-part repair has no real-run evidence.** Composites from certified parts had nothing to repair; the path is
  covered by the unit tests only.
- **Policy truth for prefix-variant Worlds is missing.** ORACLE covers only tasks on the base World. A part after other
  parts is covered only through its composite.
- **AR collections.** This was the first task synthesis on that family: 3 dead oracles of 6 at link 2, and a certified
  first link that was a 1-call figure report.
