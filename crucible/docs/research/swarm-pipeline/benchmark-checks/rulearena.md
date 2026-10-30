# RuleArena tax check on the student model

Run 2026-10-27. Student: Qwen3.6-35B-A3B on Tinker, thinking off.

Level 3 puts 48 of 100 cases in our target band of 1–2 passes out of 8, and RuleArena includes a generator for new
cases. Before any use, though, 27 of the 300 shipped answers are wrong and need a one-line fix in its answer program.

## What the benchmark is

[RuleArena](https://github.com/skyriver-2000/RuleArena) (ACL 2025) tests whether a model can follow long rule sets. We
used its tax domain, in which the model fills in one person's 2023 federal tax return by hand.

- **What the model gets:** one message of about 7,000–11,000 tokens.
  - **Facts about the person:** name, age, filing status, whether they're blind, and number of dependents.
  - **The 2023 IRS forms as text:** the person's own numbers are already filled in, and every line to compute is shown
    as `[__]`.
  - **The instructions and tables those lines need:** the standard deduction chart, the tax table and the worksheets.
- **What it outputs:** every blank, step by step, ending with "The total tax owed is $X." or "The total tax overpaid is
  $X."
- **How it's graded:** the first such sentence must match the answer to within about a cent per $1,000. Answers keep
  their cents.
- **Where the answers come from:** a Python program in the repo that fills in the same forms.
- **Size:** 100 cases at each of 3 levels, with more forms per level:
  - **Level 1:** Form 1040 and Schedules 1–3, about 150 blanks.
  - **Levels 2 and 3:** these add itemized deductions (Schedule A), self-employment (Schedules C and SE), education
    credits (Form 8863) and the child tax credit (Schedule 8812). That's about 200 and 270 blanks.

Part of a real Level 1 case (L1-000):

```
Name: John
Age (on January 2, 2024): 77
Age (on January 2, 2024) of Your Spouse: 75
Filing Status: married filing separately
* You were born before January 2, 1959: [__]
* You were blind: True
* You spouse was born before January 2, 1959: [__]
...
Line 1a - Total amount from Form(s) W-2, box 1: $36,236
Line 1z - Add lines 1a through 1c: [__]
...
You can only check the boxes for your spouse if your spouse is age 65 or older
or blind and you file a joint return.
...
Line 12 - Standard deduction or itemized deductions (from Schedule A): [__]
...
Line 37 - Subtract line 33 from line 24. This is the amount you owe: [__]
```

The answer is "The total tax overpaid is $16,734." The student got it in 6 of 8 tries. One wrong try checked the
spouse's 65+ box even though John files separately, which made the refund $330 too big.

## What we ran

- All 300 cases, 8 tries each, temperature 1, one seed per try.
- The prompt is RuleArena's own default, with no worked example, rebuilt word for word.
- Cap of 16,384 tokens; no reply was cut.
- Mean reply length was 3,800, 6,800 and 9,500 tokens for Levels 1, 2 and 3. Cost: $24.79.

## Results

Scored against the corrected answers (next section):

| Level | pass@1 | pass@8 | Never solved | In band (1–2 of 8) | Always solved |
|---|---|---|---|---|---|
| 1 | 0.60 | 0.88 | 12 | 10 | 14 |
| 2 | 0.37 | 0.88 | 12 | 36 | 2 |
| 3 | **0.15** | 0.62 | 38 | **48** | 0 |

pass@1 is the share of single tries that are correct; pass@8 is the share of cases solved at least once in 8 tries.
With the shipped answers and RuleArena's own scorer, pass@1 is 0.57, 0.31 and 0.13. The paper's best 0-shot model,
o1-preview, scored 0.72, 0.28 and 0.19. So the student, with thinking off, is about level with it on Levels 2 and 3.

## Are the answers right?

Mostly, but 9% are wrong as shipped. We compared the answer program with the forms line by line: every Form 1040
line, the schedules, the tax table and every worksheet.

- **27 wrong (Level 1: 3, Level 2: 11, Level 3: 13).** The forms say Form 1040 line 22 is "zero or less, enter -0-",
  but the program lets it go negative.
  - Example: in L1-017 the shipped answer is overpaid $387, while the forms give owed $703.
  - On these 27 cases, the student matches the corrected answer in 96 of 216 tries, and the shipped one only once.
  - 2 of them also get a negative child tax credit. In L3-077 that produces a total tax of −$2,420.
- **21 ambiguous.** The forms never state the $400 threshold for self-employment tax, and 20 of these cases are
  business losses.
- **9 ambiguous by one day.** "Age exactly 65" and "born before January 2, 1959" disagree.
- **Unrealistic but consistent:** 92 cases have qualified dividends larger than ordinary dividends. The forms and the
  answers agree on them, so they aren't wrong.

Both real problems are one-line fixes: floor line 22 in the program, and add the $400 rule to the forms.

## Where the student goes wrong

We traced the 36 cases where at least 3 of the 8 tries agree on the same wrong answer. None of them was a label error
beyond the 27 above.

- **18: spouse boxes.** It checks the spouse's 65+ or blind box on a return that isn't joint, which the forms forbid.
- **About 8: arithmetic.** These are $1,000 slips in long sums, multiplication errors and cents drift.
- **A few: form rules.** Examples are routing on Form 8863 and turning a usable child tax credit into the refundable
  kind on Schedule 8812.

## What it means for us

- **The best training source we tested.** Levels 2 and 3 already sit in the band, the generator makes unlimited new
  cases, and the label fixes are one line each.
- **A calculator would make it much easier.** Much of the difficulty is mental arithmetic. A multi-turn version with a
  calculator tool would need its difficulty to come from the rules and from longer cases, with more forms and more
  people.
- **Fix the program and the forms before any use.**

## Files

`crucible/data/rulearena_check/` (gitignored, not in the repo):
- `report.md`: full method and every traced case.
- `run_rulearena.py`: sampling and grading.
- `label_audit.py`: the answer checks.
- `ra_src/`: RuleArena's files, unchanged, MIT-licensed.
