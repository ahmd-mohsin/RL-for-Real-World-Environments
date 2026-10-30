# SARA check on the student model

Run 2026-10-26. Student: Qwen3.6-35B-A3B on Tinker, thinking off.

The student gets 60% of single tries exactly right, but only 16 of 96 cases land in our target band of 1–2 passes out
of 8. The answers are sound, since SARA's own program reproduces all of them. SARA is mostly too easy per case, and too
small to train on.

## What the benchmark is

[SARA](https://nlp.jhu.edu/law/sara/) (StAtutory Reasoning Assessment, Johns Hopkins) asks how much tax a person owes,
given excerpts of the US tax code and a short story. We used the `sara_numeric` version in
[LegalBench](https://github.com/HazyResearch/legalbench).

- **What the model gets:** about 6,900 tokens of statute text, the same for every case. It covers 9 sections of the
  Internal Revenue Code (§1, §2, §63, §68, §151, §152, §3301, §3306, §7703). After the statute come the case facts and
  a question.
- **What it outputs:** its reasoning, then one dollar amount alone on the last line.
- **How it's graded:** the last number must equal the answer exactly, to the dollar.
- **Where the answers come from:** the authors encoded each case as facts in Prolog. Their program over the 9 sections
  computes the tax.
- **Size:** 96 test cases (100 numeric cases in total).

A real case (tax_case_34):

```
Alice's gross income for the year 2017 is $22895. Alice takes the standard deduction.
How much tax does Alice have to pay in 2017?
```

The answer is $2,684, and the student got it in 6 of 8 tries.

## What we ran

- All 96 test cases, 8 tries each, temperature 1, one seed per try.
- The prompt is the statute, the facts and the question, followed by "final answer as a single dollar amount, in whole
  dollars, alone on the last line".
- At our factory's default cap of 4,096 tokens, 26% of replies were cut off mid-reasoning. We continued those to 8,192
  tokens (the same try, just a longer cap). 10 tries were still cut at 8,192.
- Mean reply length was about 2,700 tokens. Cost: $4.40.

## Results

| Scoring | pass@1 | pass@8 | Never solved | In band (1–2 of 8) | Always solved |
|---|---|---|---|---|---|
| Exact, 8,192-token cap | **0.60** | 0.91 | 9 | **16** | 24 |
| Exact, 4,096-token cap | 0.49 | 0.79 | 20 | 17 | 17 |

pass@1 is the share of single tries that are correct; pass@8 is the share of cases solved at least once in 8 tries.
LegalBench's own scorer accepts any answer within 10% of the right one. On that basis pass@1 would be 0.87, which is
far too loose for training.

## Are the answers right?

Yes.

- **Reproduced:** we ran SARA's Prolog program on all 100 numeric cases, and it reproduces every answer. It only runs
  on SWI-Prolog 8.0.x or older, so we ran it in Docker.
- **Strict by design:** any fact the story doesn't state counts as false or zero. For example, head of household needs
  an explicit fact that the person pays over half the cost of the home.
- **Separate rounding:** each tax is rounded before the taxes are added, which causes some $1 differences.
- **One judgment call:** cases 29 and 47 differ on whether a retirement payment counts as wages. That depends on how
  the authors encoded the story, not on anything the text says.

## Where the student goes wrong

We hand-checked the 9 cases it never solves against the statute:

- **Leaves out a whole tax:** it omits the employer tax under §3301 in cases 26, 29 and 47, and in case 10 in 6 of 8
  tries.
- **Assumes facts the story doesn't state:** it claims head of household without the home-cost fact, in cases 61, 64
  and 39. Case 61 says only that Alice's father "has the same principal place of abode as Alice since 2012". The
  answer is $25,055 at the single rate, and all 8 tries said $22,959 at the head-of-household rate.
- **Uses a section the excerpt leaves out:** case 78.
- **Arithmetic slips are rare:** they caused 1 of the 5 failed tries we traced.

## What it means for us

- **Too easy per case:** 24 cases are always solved and only 16 are in the band.
- **Small and hard to extend:** a new case means writing a new Prolog encoding by hand.
- **Useful failure patterns:** the student misses rules and assumes facts. Both are patterns for a tax World to probe.

## Files

`crucible/data/sara_check/` (gitignored, not in the repo):
- `report.md`: full method and per-case notes.
- `run_sara.py`: sampling and grading.
- `oracle_run.py`: the Prolog re-run.
