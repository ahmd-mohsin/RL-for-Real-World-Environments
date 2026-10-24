# carbon task a783fb3a: too easy, made harder, now kept for training

**Original:** `db-go-carbon-a783fb3a` (Ahmed, held out)
**Fixed version:** `db-go-carbon-a783fb3a-h5` (Uzair, 2026-10-23)
**Result:** kept for training

## What the task is

The agent gets **carbon** (a Go library for dates and times) with real bug fixes undone, plus a bug report written like a
message from a user. It has **300 tool calls** to fix every bug. Hidden tests check each bug.

A **training** task is kept when the student passes **10–60%** of 8 attempts, a strong model passes, and most failures are
on the skills the task trains.

## Before: the original task

| # | Bug, as the user sees it |
|---|---|
| m01 | Formatting with a layout that ends in a lone backslash crashes |
| m02 | With the week starting on another day than Monday, weekday names come out wrong |

| Result (Ahmed's measurement) | |
|---|---|
| Student | 7/8 at 300 calls |
| Status | too easy for the benchmark |

## What we changed

1. **Moved carbon to training.** Both carbon tasks passed above the benchmark bar (3/7 and 7/8), so on Ahmed's call
   carbon left the held-out list (commit `f086d3c5`).
2. **Added bugs, one step at a time, measuring each step** (Ahmed's two paragraphs kept word for word):

| Version | Added | Student |
|---|---|---|
| -h1 | lunar date a day early after midnight | 3/3: too easy |
| -h2 | + hour boundaries in a repeated DST hour | 0/3: too hard (DeepSeek 1/1) |
| -h3 | + minute boundaries in a repeated DST hour (instead of hour) | 7/8: too easy |
| -h2b | -h2 + the Kathmandu rule written out | 0/3: still too hard |
| **-h5** | -h3 + SQL NULL scan keeps the old date + same-decade check ignores the century | **1/8: kept** |

3. Each new paragraph describes what a user sees, with the exact cases the hidden tests check, including "keep today's
   answer" rules a natural fix would break (Monrovia's offset with seconds). No file, function or test names.
4. **Found and fixed a grader bug** on the way: DeepSeek solved -h3 fully but lost the pass because the install check
   misread a nested `$(...)` (commit `1ab930ac`; a re-check of 115 stored attempts gained 2 passes, lost none).
5. Checked: no hidden tests or stale tests in the start code, no changelog line; all five gates pass.

## After: the fixed task (6 bugs, measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| m01 format crash on a trailing backslash | 7/8 |
| m02 SQL NULL keeps the previous date | 2/8 |
| m03 weekday names with another week start | 7/8 |
| m04 minute boundaries in a repeated DST hour | 8/8 |
| m05 same decade ignores the century | 8/8 |
| m06 lunar date a day early after midnight | 2/8 |

| Result | |
|---|---|
| Student full passes | **1/8 (12.5%)**, inside 10–60% |
| Qwen3.5-397B | 1/2 full passes |
| DeepSeek | 2/2 full passes |
| Failures on the skills the task trains | 7/7 |
| Fairness check (S9b) | pass |
| Verdict | **kept for training** |

## Where things are

- Task files: `crucible/data/debug-uzair/gen/store/train/tasks/db-go-carbon-a783fb3a-h5/`
- Bug report: `crucible/data/debug-uzair/prompts/db-go-carbon-a783fb3a-h5.json`
- Results: `crucible/data/targeting/uzair-full-carbon-h5/report.json` ($11.00); earlier steps in `uzair-qp-carbon-*`
  and `uzair-full-carbon-h3`
