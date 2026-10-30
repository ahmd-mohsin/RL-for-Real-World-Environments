# 14 · Tax World (SARA): design

Status: draft for approval, 2026-10-26. No code yet.

## Goal

Long-horizon tasks the student (`Qwen/Qwen3.6-35B-A3B`) usually fails:

- **Student pass rate 0.1–0.3 per task at k=8** (1 or 2 passes out of 8), the low end of the training band in
  [12-difficulty.md](12-difficulty.md). A task at 0 gives no learning signal.
- **A frontier model passes sometimes** (12-difficulty's fairness rule: a task no frontier model solves is fixed, not kept).
- **About 40–60 tool calls per task**: the agent finds the law, gathers each case's facts, decides, records.
- **Hard for legal reasons only**: never because a fact is unstated, a label is wrong, or a tool is awkward. At least
  70% of failures must be skill gaps (12-difficulty's `fix_task` rule).

## The dataset

**SARA** (StAtutory Reasoning Assessment), from Johns Hopkins.

- **Content:** statutory reasoning in US federal tax law.
  - **The law:** excerpts of 9 sections of the Internal Revenue Code (§§1, 2, 63, 68, 151, 152, 3301, 3306, 7703),
    trimmed to the subsections the cases need.
  - **The cases:** 376 short, hand-written fact patterns (Alice, Bob, ...). 276 ask whether a provision applies
    (entailment). **100 ask how much tax a person owes**: those are the ones this World uses.
- **The Prolog program:** the authors translated every section into Prolog rules and every case into Prolog facts.
  Running it computes each answer, and it reproduces all 100 numeric labels (we checked: the report's "Prolog oracle" section).
- **Splits:** SARA splits the numeric cases 80 train / 20 test. LegalBench repackages them as `sara_numeric`, split
  4 train / 96 test.
- **Licence:** MIT-style.

Links:

- SARA project page: <https://nlp.jhu.edu/law/>; download (statutes, cases, Prolog, splits): <https://nlp.jhu.edu/law/sara/sara.tar.gz>
- LegalBench `sara_numeric`: <https://huggingface.co/datasets/nguha/legalbench> (config `sara_numeric`)
- Papers: Holzenberger, Blair-Stanek, Van Durme, *A Dataset for Statutory Reasoning in Tax Law Entailment and Question
  Answering* (NLLP 2020, <https://ceur-ws.org/Vol-2645/paper5.pdf>); Holzenberger, Van Durme, *Factoring Statutory
  Reasoning as Language Understanding Challenges* (2021, SARA v2, <https://arxiv.org/abs/2105.07903>)

## What we know (crucible/data/sara_check/report.md)

- **Single-turn**, all statutes pasted in the prompt: exact pass@1 **0.60**, pass@8 0.91 (96 LegalBench items). Too easy
  as it stands.
- **Failure modes** (hand-checked against the statute):
  - the §3301 employer tax left out;
  - head of household assumed without a household-cost fact;
  - a missed exception (a nonresident alien's zero standard deduction);
  - the scope of a cross-reference (§7703(b) applies to §1, not §2(b));
  - long phaseout arithmetic.
- **Oracle**: SARA's Prolog program reproduces all 100 numeric labels. Each case is event facts (who, whom, amount,
  dates, purpose), 4–68 per case. 22 cases generate facts with small rules. It runs only on SWI-Prolog ≤ 8.0
  (Docker `swipl:8.0.3`, amd64).

## 1. The task

A tax officer's queue. One task holds **5 cases** (a knob: 3–8). Each case is one person and one tax year, drawn from
different SARA case files. The instruction (every graded property is a sentence of it, as in `finance_ap`):

> You are a tax assessment officer. 5 cases are open in the case system. For each case, assess the total tax the person
> owes for the year under the legislation in the library, and record it with `assess_case`, exact to the dollar: round
> each tax to the nearest dollar, then add them. The case system is the complete record: anything it does not show did
> not happen. Reply DONE when every case is assessed.

The instruction does not name which sections apply. Choosing them is the skill being trained.

## 2. The World (the `finance_ap` pattern)

**`law`** (`common/doc_library`; tools `list_docs`, `read_doc`, `search_docs`):

- **The 9 sections**, from SARA's `statutes/source`, with the year-limited paragraphs moved out: §63(c)(7), §151(d)(5)
  and §68(f).
- **An amendment act** containing those paragraphs, with its effective years (taxable years 2018–2025). 23 of the 100
  cases fall in those years. Same law as SARA, so the oracle needs no new rules.
- **An assessment manual**: what an assessment covers, the rounding rule, and the order of authority (statutes and acts
  are law; guidance is not).
- **A guidance note**: non-binding, dated, partly out of date or oversimplified. For example, "an unmarried taxpayer
  living with a dependent files as head of household". It tests checking against the statute, and the manual says
  which wins.

**`cases`** (`common/sqlite_app`): SARA's facts turned into records. Each table answers one kind of question:

| table | holds (from the Prolog predicates) |
|---|---|
| `cases` | case id, person, tax year, status, assessed amount |
| `people` | birth and death dates, citizenship, nonresident-alien periods, blindness |
| `family_links` | parent, child, sibling and in-law links |
| `marriages` | spouses, start, end, how it ended (death, divorce, legal separation) |
| `residences`, `household_costs` | who lived where and when; who paid what share of a home's costs per year |
| `incomes` | gross income by person and year |
| `filings` | joint returns, itemized deductions |
| `employment`, `payments` | services (kind, place, dates); payments with payee or fund, purpose, plan, and the service they pay for |
| `life_events` | retirement, disability, enrollment, incarceration, hospital stays |

Tools: `list_cases`, `get_case`, `get_person`, `list_family`, `list_marriages`, `list_residences`,
`list_household_costs`, `list_incomes`, `list_filings`, `list_employment`, `list_payments`, `list_life_events`,
`assess_case(case_id, amount)`.

Facts the SARA text left implicit become explicit fields, for example whether a retirement-fund payment pays for Bob's
service (case 29 vs 47). The agent sees exactly the facts the oracle used.

**Horizon.** About 11 law reads per task, plus 6–10 lookups and one assessment per case, so 40–60 calls for 5 cases.
That is about 30–40K tokens of context. The `:peft:262144` variant the debug lane uses leaves room for 8 cases.

## 3. Labels

- **Labels are computed offline, never at run time.** A build script turns each file's Prolog into records and runs the
  oracle. Labels are stored in the world's `forge/world.json`, so `pack check` and training never need Prolog.
- **A world is kept only if** `tax/3` gives exactly one value, the run finishes within a time limit, and a round trip
  holds: records → Prolog facts → oracle → the same label.
- **The converter rejects any predicate it cannot map.** That catches dead facts such as case 25's `patient(` typo.
  It first expands the generator rules by loading the case and dumping every derived fact.

## 4. Grading and progress

- **Grade from the final `cases` table.** A case is correct when its assessed amount equals the label exactly.
  Reward is 1 when every case is correct; `outcome` keeps the share correct.
- **Φ moves per case** and ends at that share (`finance_ap`'s `InvoiceSubgoals` pattern), which gives a denser signal on
  a long task.

## 5. Difficulty knobs

| knob | v0 | later range |
|---|---|---|
| cases per task | 5 | 3–8 |
| years | as drawn | require at least one case in 2018–2025 and one before 2018 (the act decides) |
| amendment layout | one act | the provisions split across 2–3 acts with their own effective dates |
| guidance note | on | off, or more wrong statements |
| employer cases (§3301) | as drawn | require at least one |
| fact variation (step 4) | none: the 100 originals | incomes, dates, filing choices, events added or removed; relabelled by the oracle |
| cross-case links | none | cases sharing one household (each case's tax depends on another person's status) |

## 6. Shortcuts the grader must hold near 0

Scripted adversaries for `pack check --level semantic`, each with an `expect_max`:

- income tax only (no §3301);
- the amendment act ignored, or applied to every year;
- head of household whenever a dependent lives with the person (the guidance-note rule);
- figures copied from the guidance note;
- the total rounded once instead of per tax;
- $0 for every case.

## 7. Plan and measurement

1. **v0 build from the original cases.**
   - The 80 SARA-train numeric cases give 16 tasks of 5. The 20 SARA-test cases give 4 held-out tasks.
   - No fact variation yet; every label is one of the 100 already checked.
2. **First measurement on 8 tasks.**
   - Student at k=8; one frontier model (DeepSeek-V4.1-Flash) at k=2.
   - Expected cost about $10 (cached prefill; ~$0.1 per student episode), confirmed by a one-task dry run first.
   - Read every failure: skill gap, or a World problem.
3. **Then step 4** (generated facts, oracle labels) **and step 5** (`pack check --level semantic` and `--level fixture`,
   adversaries, tuning the knobs toward 0.1–0.3).

## 8. Decisions for you

1. **`refer_case`.** The oracle always returns an amount, so a referral needs a trigger of our own, such as records
   that conflict (a control gate, as in `finance_ap`). I'd leave it out of v0 so that failures stay legal ones, and add
   it later as a knob if the band needs it.
2. **A `calculate` tool.** I'd include one. Arithmetic slips are the least interesting failure, and a tax officer has a
   calculator.
3. **An independent label check before step 4.** Generated facts reach oracle paths SARA's authors never tested.
   `finance_ap` checks its labels against an independent implementation over 1,000 seeds. The equivalent here is a
   short Python implementation of the 9 sections compared with the Prolog on every generated file. Not needed for v0.
4. **Name.** `tax_assessment` (a domain pack like `finance_ap`).

SARA (MIT-style licence, © 2020 Nils Holzenberger): the pack ships its statutes and case facts with the notice.
