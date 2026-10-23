# excelize task 824ac2ce (held out): two undescribed bugs removed, now kept for the benchmark

**Original:** `db-go-excelize-824ac2ce` (Ahmed)
**Fixed version:** `db-go-excelize-824ac2ce-d1` (Uzair, 2026-10-23)
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
| (m01) | **Undone, but no paragraph:** wrong argument count for a function nested in parentheses |
| (m02) | **Undone, but no paragraph:** errors pushed as extra function arguments |
| m03 | The exclusive percentile formula (PERCENTILE.EXC) crashes for a fraction outside the valid range |
| m04 | COUNT counts text and TRUE/FALSE from cell references and arrays, unlike Excel |

| Result (Ahmed's measurement) | |
|---|---|
| Student | 0/8 at 300 calls |
| Status | "spec defect: fix the report, re-measure" |

## The defect we found

Four fixes were undone, but the report described only two. The other two bugs were real and in the code the agent had to
fix, but nobody told the agent about them, and their tests were mixed into m03's. So no attempt could pass by doing what
the report asked.

## What we changed

1. **Applied the two undescribed fixes in the start code** (the remedy in Ahmed's own plan). The task now has 2 bugs,
   each with its own paragraph and tests.
2. **Kept Ahmed's two paragraphs word for word**, renumbered m01 and m02.
3. The bugs are linked: COUNT's hidden test can only run once the PERCENTILE.EXC crash is fixed.
4. Checked: the start code has no hidden tests, no stale test that expects a bug, no changelog line; all five gates pass.

## After: the fixed task (measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| m01 PERCENTILE.EXC crashes | 8/8 |
| m02 COUNT differs from Excel | 2/8 |

| Result | |
|---|---|
| Student full passes | **2/8 (25%)**, under the 30% limit |
| Qwen3.5-397B | 1/2 full passes |
| DeepSeek | 2/2 full passes |
| Failures on the skills the task tests | 6/6 |
| Verdict | **kept for the benchmark** |

- The student's misses on COUNT are on a rule the report states: values inside `{...}` arrays are counted like
  referenced cells (text and TRUE/FALSE not counted). Both strong models got it.

## Where things are

- Task files: `crucible/data/debug-uzair/gen/store/heldout/tasks/db-go-excelize-824ac2ce-d1/`
- Bug report: `crucible/data/debug-uzair/prompts/db-go-excelize-824ac2ce-d1.json`
- Results: `crucible/data/targeting/uzair-full-excelize-d1/report.json` (shared with excelize-65be17e3-d1; $20.20 for both)
