# MathConstruct check on the student model

Run 2026-10-27. Student: Qwen3.6-35B-A3B on Tinker, thinking off.

The student passes 33% of single tries, and 28 of 121 problems land in our target band of 1–2 passes out of 8. But 8
of the 24 checkers we read have holes, and 3 of those hand out a reward for no math at all. Every checker needs an
audit before we train on it.

## What the benchmark is

[MathConstruct](https://github.com/eth-sri/mathconstruct) ([paper](https://arxiv.org/abs/2502.10197), ETH Zurich,
2025) asks for constructions: build a mathematical object that meets a competition problem's conditions.

- **What the model gets:** the problem with its numbers filled in, plus a required answer format. Prompts are short,
  with a median of 470 characters.
- **What it outputs:** the object inside `\boxed{}`. It can be a matrix, a list, a coloring or a function.
- **How it's graded:** each problem has its own hand-written Python checker that tests the conditions. Many different
  answers can be correct, so there is no single answer to compare against. The authors' parser reads the last
  `\boxed{}`.
- **Size:** 121 problems from 17 sources, mostly olympiads: 29 from the IMO shortlist, 8 from USAMO and 4 from Putnam.
  Each problem has a generator for new sizes; we used the original competition sizes.

A real problem (imc-2012-2):

```
Find an 7 × 7 matrix of rank at most 3 which has zeros along the main diagonal and
strictly positive real numbers off the main diagonal.
```

One answer is the matrix with entries (i − j)². The checker tests the size, the zero diagonal, the positive entries
and the rank. The student got it in 1 of 8 tries, so this problem is in the band.

## What we ran

- All 121 problems at their original sizes, 8 tries each, temperature 1, one seed per try.
- The prompt is the authors' own, with no system prompt.
- One turn only. The paper also allows up to 2 rounds of format feedback, which we didn't give.
- Cap of 16,384 tokens. The 8 replies cut at that cap were continued to 32,768 tokens; 4 were still cut and count as
  no answer.
- The checker code ran only inside a Docker container with no network access. The authors' 155 tests pass there.
- Mean reply length was 6,800 tokens. Cost: $8.89.

## Results

| Problems | Count | pass@1 | pass@8 | Never solved | In band (1–2 of 8) | Always solved |
|---|---|---|---|---|---|---|
| All | 121 | **0.33** | 0.60 | 49 | **28** | 18 |
| Combinatorics | 41 | 0.30 | 0.51 | 20 | 6 | 3 |
| Algebra | 22 | 0.56 | 0.77 | 5 | 3 | 8 |
| Number theory | 54 | 0.27 | 0.61 | 21 | 19 | 6 |
| Geometry | 4 | 0.25 | 0.25 | 3 | 0 | 1 |

pass@1 is the share of single tries that are correct; pass@8 is the share of problems solved at least once in 8 tries.

For scale, the paper reports QwQ-32B at 16%, o1-mini at 27% and o3-mini at 61%, and a later update gives GPT-5 82%.
Those numbers average 4 sizes per problem and allow format feedback, so they aren't directly comparable with ours.

## Are the checkers right?

Not all of them. We read 24 of the 121 checkers against their problems: all the Bulgarian, Putnam and IMC ones, plus
some from the IMO shortlist, USAMO, Konhauser and Swiss competitions. We then tested each suspected hole with an answer
built to hit it.

- **13 are correct, and 3 have minor gaps.**
- **8 have serious holes.** 7 accept wrong answers and 3 reject right ones; two problems have both kinds.

| Problem | Hole | Kind |
|---|---|---|
| bulgarian-ifym-2022-d1-p6-8th | 343 followed by 19 zeros passes, crossing out one zero per step, although only non-zero digits may be crossed out | Wrong answer passes, for free |
| usamo-2005-1 | Placing n itself between every two divisors passes, because only the set of indices is checked | Wrong answer passes, for free |
| swiss-2024-8-selection | `0` and `-1/x` pass, although the function has to take positive values | Wrong answer passes, for free |
| bulgarian-pms-10-4 | A self-loop or a repeated edge pads the edge count | Wrong answer passes |
| bulgarian-pms-2022-10-p3 | (0, 0) and (3, 0) pass, although p and q must be natural numbers | Wrong answer passes |
| imo-shortlist-2001-c6 | Strings are never checked to be balanced, so 14 junk strings can pass | Wrong answer passes |
| usamo-2001-1 | Color pairs are stored in order, so two boxes with the same colors reversed pass | Wrong answer passes |
| bulgarian-pms-10-4 | An edge written as (j, i) instead of (i, j) fails | Right answer fails |
| bulgarian-ifym-2022-d1-p6-8th | Crossing out two digits in one step fails, although the problem says "some" digits | Right answer fails |
| imc-2018-6 | It tests the element-by-element product instead of the dot product, so valid answers whose columns overlap fail | Right answer fails |

**None of these changed our numbers.** We re-graded the student's replies on these problems with our own checks
written from the problem statements, and found no false passes and no false fails. An RL policy trained against
these checkers would find holes like these, though.

## Where the student goes wrong

- **The wrong construction, in most failed tries:** 429 of 968 tries break one of the problem's conditions.
- **The wrong answer format, in about 22% of tries:**
  - 58 can't be parsed. 30 of these write `\dots` instead of the full object, and 11 have no `\boxed{}`.
  - 102 give the wrong number of items.
  - 55 are tagged as format errors. Some of these are really wrong answers.
- **Of the 49 problems never solved,** 36 fail mostly on the construction and 13 mostly on the format.

## What it means for us

- **121 distinct ideas, 28 already in the band.** A new size usually needs the same construction, so it is closer to
  121 tasks than to an unlimited supply.
- **Memorization is untested.** These are the published competition versions, and the repo includes each problem's
  solution code. One run on generated sizes, about $9, would show how much of the 0.33 is recall.
- **Every checker needs an audit before training,** since about 1 in 3 of those we read has a hole.
- **A Python tool would change the task.** It would fix most format failures and make some problems searchable by
  brute force. A multi-turn version would need sizes where brute force fails, so the difficulty stays in the
  construction idea.

## Files

`crucible/data/mathconstruct_check/` (gitignored, not in the repo):
- `report.md`: full method and every audited checker.
- `run_mathconstruct.py`: sampling and grading.
- `audit_checkers.py`: the hole tests.
- `strict_recheck.py`: our own checks.
- `Dockerfile`: the grader container.
- `mc_src/`: the authors' problems and checkers, unchanged, MIT-licensed.
