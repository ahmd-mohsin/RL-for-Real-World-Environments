# pulldown-cmark task: made easier, still too hard, set aside

**Original:** `db-rust-pulldown-cmark-e072a230` (Ahmed)
**Easier version:** `db-rust-pulldown-cmark-e072a230-e1` (Uzair, 2026-10-22)
**Result:** not kept. It is fair (a strong model solves it every time), but the student never fully passes it.

## What the task is

The agent gets **pulldown-cmark** (a Rust library that turns Markdown into HTML) with a few real bug fixes undone, plus a
bug report written like a message from a user. It has **300 tool calls** to find and fix every bug. Hidden tests check
each bug separately. The student is Qwen3.6-35B-A3B; a **full pass** means every bug fixed in one attempt.

## Before: the original task (4 bugs)

| # | Bug, as the user sees it | Student fixed it (150 calls) |
|---|---|---|
| m01 | An opening `:::` fence at the very end of the input gives positions past the end of the text | 3/8 |
| m02 | Two hard line break cases are wrong (a tab counted as a break; a backslash at the end of link text not counted) | 1/8 |
| m03 | Old-style CR line endings in HTML and code blocks crash or lose the last newline | 1/8 |
| m04 | In an unclosed code or HTML block, a blank last line is dropped | **0/8** |

**Linked bugs:** m04's correct output needs m03's fix (every last line gets its newline). This is the task's only link.

| Result | |
|---|---|
| Student full passes | 0/8 (at 150 calls; the 300-call run was cancelled) |
| Average progress | 0.16 |
| Verdict | too hard |

## What we changed

1. **Dropped m02 (hard line breaks), not m04.** Ahmed's plan said to drop m04, but m04 is half of the only link, so
   dropping it would leave no link. His own rule says to drop the next-weakest bug instead.
2. **Kept the report wording.** The other three paragraphs are Ahmed's text, unchanged.
3. **Checked the start code:** no hidden test names, no expected outputs, no changelog, and all 1,418 visible tests pass
   on the fixed code.

Renumbered: old m01 stays m01, old m03 is now m02, old m04 is now m03.

## After: the easier task (3 bugs, measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| m01 container fence at end of input | **11/11** |
| m02 CR line endings | 3/11 |
| m03 blank last line of an unclosed block (needs m02) | **0/11** |

| Result | Before | After |
|---|---|---|
| Student full passes | 0/8 | **0/11** |
| Average progress | 0.16 | 0.42 |
| DeepSeek | not measured | **3/3 full passes** |
| Qwen3.5-397B | not measured | 0/2 (stopped early) |
| Verdict | too hard | **too hard** (needs 10–60%) |

## Why it still fails

- **The student is slow to start.** Its first fix usually lands around call 160–260 of 300 (Rust builds and test runs
  take many steps), so little is left for the harder bugs. 2 of 8 attempts ran out of calls.
- **It misses the link.** Even when it fixes m02, it never gets m03, which needs m02's change. The audit labels these
  failures "stalled on interacting faults".
- **It stops too early.** One attempt tested the exact m03 example, got the wrong output (the blank line missing) and
  marked it "OK".

## Why we stopped

- Dropping another bug would leave 2; the minimum is 3.
- We don't add hints to the report.
- We searched all 16 pulldown-cmark fixes for another bug that really links to m02, to swap in for m03: there is none.

So the task is set aside. It is fair and could become useful once the student gets stronger.

## Where things are

- Task files: `crucible/data/debug-uzair/gen/store/train/tasks/db-rust-pulldown-cmark-e072a230-e1/`
- Bug report: `crucible/data/debug-uzair/prompts/db-rust-pulldown-cmark-e072a230-e1.json`
- Results: `crucible/data/targeting/uzair-pulldown/report.json` (total cost $41.06)
- Link search: `crucible/data/debug-uzair/search/pulldown-cmark-links-0d.json`
