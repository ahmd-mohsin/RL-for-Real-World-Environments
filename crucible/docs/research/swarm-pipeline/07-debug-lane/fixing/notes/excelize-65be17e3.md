# excelize task 65be17e3 (held out): one undescribed bug removed, one missing rule added, now kept for the benchmark

**Original:** `db-go-excelize-65be17e3` (Ahmed)
**Fixed version:** `db-go-excelize-65be17e3-d1` (Uzair, 2026-10-23)
**Result:** kept for the benchmark

## What the task is

The agent gets **excelize** (a Go library that reads and writes Excel files, with a formula engine) with real bug fixes
undone, plus a bug report written like a message from a user. It has **300 tool calls** to fix every bug. Hidden tests
check each bug.

This is a **held-out (benchmark)** task: the student never trains on it. It is kept when the student passes **under 30%**
of attempts and a strong model can solve it. Held-out tasks are never made easier; only real defects are fixed.

## Before: the original task

| # | Bug, as the user sees it |
|---|---|
| m01 | ISNUMBER says TRUE for a cell holding TRUE; OR mishandles ranges, arrays and text |
| (m02) | **Undone, but no paragraph:** VALUE gives wrong results for empty and range arguments |
| m03 | MDETERM is wrong for a 1×1 matrix (and so MINVERSE for small matrices) |
| m04 | IF with no third argument returns the wrong thing when the condition is false |

| Result (Ahmed's measurement) | |
|---|---|
| Student | 0/8 at 300 calls |
| Status | "spec defect: fix the report, re-measure" |

## The defects we found

1. **An undescribed bug.** The VALUE fix was undone but no paragraph mentioned it.
2. **An unstated rule.** The OR fix also makes OR return a real TRUE/FALSE value instead of the text "TRUE", and the
   hidden tests check that (`OR(1)*1` gives 1, `ISTEXT(OR(1))` is FALSE). The report never said so. In our first
   quick pass, 2 of 3 student attempts failed exactly there.

## What we changed

1. **Applied the VALUE fix in the start code.** The task now has 3 bugs, each with its own paragraph.
2. **Added one sentence to the OR paragraph:** "OR's result must also be a logical value, as in Excel, not the text TRUE:
   OR(1)*1 must give 1 and ISTEXT(OR(1)) must be FALSE." It describes behaviour, not where the code is.
3. Everything else is Ahmed's wording, renumbered m01–m03.
4. Checked: the start code has no hidden tests, no stale test that expects a bug, no changelog line; all five gates pass.
   The last bug's test (IF) is the big formula table, so it also needs the other two bugs fixed: a real link.

## After: the fixed task (measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| m01 ISNUMBER / OR | 8/8 |
| m02 MDETERM / MINVERSE | 8/8 |
| m03 IF, graded by the whole formula table (also checks every OR case) | 0/8 |

| Result | |
|---|---|
| Student full passes | **0/8**, under the 30% limit |
| Qwen3.5-397B | 0/2 full passes |
| DeepSeek | 1/2 full passes |
| Failures on the skills the task tests | 8/8 |
| Verdict | **kept for the benchmark** |

- Every student fixes IF itself; what fails is the harder OR cases the report states: text such as "text" must give the
  wrong-type error even after a true argument, and a false range must not stop later arguments from counting.

## Where things are

- Task files: `crucible/data/debug-uzair/gen/store/heldout/tasks/db-go-excelize-65be17e3-d1/`
- Bug report: `crucible/data/debug-uzair/prompts/db-go-excelize-65be17e3-d1.json`
- Results: `crucible/data/targeting/uzair-full-excelize-d1/report.json` (shared with excelize-824ac2ce-d1; $20.20 for both)
