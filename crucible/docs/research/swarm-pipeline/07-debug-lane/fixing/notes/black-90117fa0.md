# black task: one bug swapped, now kept for training

**Original:** `db-python-black-90117fa0` (Ahmed)
**Fixed version:** `db-python-black-90117fa0-e1` (Uzair, 2026-10-23)
**Result:** kept for training

## What the task is

The agent gets **Black** (the Python code formatter) with a few real bug fixes undone, plus a bug report written like a
message from a user. It has **300 tool calls** to find and fix every bug. Hidden tests check each bug separately. The
student is Qwen3.6-35B-A3B; a **full pass** means every bug fixed in one attempt.

## Before: the original task (3 bugs)

| # | Bug, as the user sees it | Student fixed it |
|---|---|---|
| m01 | A long string that is a one-item tuple (trailing comma) makes Black fail with string processing on | 5/8 |
| m02 | Strings in parentheses followed by `# type: ignore` get merged into one line far over the limit | 6/8 |
| m03 | A long string as a dict value with `# type: ignore` makes Black fail ("the comment would be lost") | 2/8 |

**Linked bugs:** m03's correct output needs m02's fix (we checked: with m02 undone, m03's own case comes out wrong).

| Result at 300 tool calls | |
|---|---|
| Student full passes | 1/8 |
| Attempts that ran out of calls | **6 of 8** |
| Verdict | not kept: most failures were running out of calls |

**Why it failed:** the task was too long. m01 is the biggest of the three fixes, so the student spent most of its calls
before reaching the linked pair.

## What we changed

1. **Swapped m01.** The original m01 (the one-item tuple crash) is not part of the link, so we replaced it with a
   smaller, quicker real Black fix: a crash when an f-string follows `# fmt: off` inside brackets. The bug count (3) and
   the real link (m02 → m03) stay the same.
2. **Kept Ahmed's wording.** m02 and m03 are his paragraphs unchanged. The new m01 paragraph is also his, taken from
   another of his Black tasks (`ae5830b7`, set aside) where he described this exact bug.
3. **One small edit to the opening.** It said "long string literals trip it up in a few ways"; the new m01 is not a
   long-string bug, so we removed "long string literals". This adds no information about the bugs.
4. **Checked the task:** the new bug also crashes with preview string processing on (as the opening says), no hidden
   test cases or expected outputs are visible to the agent, no changelog line names the fixes, and all 576 visible tests
   pass on the fixed code.

## After: the fixed task (3 bugs)

| Bug | Student fixed it |
|---|---|
| m01 f-string after `# fmt: off` (new) | **8/8** |
| m02 strings merged past the limit | 7/8 |
| m03 `# type: ignore` crash (needs m02) | 3/8 |

| Result at 300 tool calls | Before | After |
|---|---|---|
| Student full passes | 1/8 (12%) | **3/8 (38%)** |
| Attempts that ran out of calls | 6/8 | **0/8** |
| Average progress | 0.50 | **0.75** |
| DeepSeek | — | 2/2 full passes |
| Qwen3.5-397B | — | 0/2 |
| Verdict | not kept | **kept for training** (target: 10–60%) |

- All 5 failed attempts failed on what the task is meant to teach: the linked bugs (4) and finding several bugs (1).
- Cost: $27.52 to measure it ($10.04 more for the quick check shared with three other tasks).

## Where things are

- Task files: `crucible/data/debug-uzair/gen/store/train/tasks/db-python-black-90117fa0-e1/`
- Bug report: `crucible/data/debug-uzair/prompts/db-python-black-90117fa0-e1.json`
- Results: `crucible/data/targeting/uzair-black-e1/report.json`
- Built at Black commit `8d5a2d9` (2026-09-28), the same point Ahmed's task was built from
