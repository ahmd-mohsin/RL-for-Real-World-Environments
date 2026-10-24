# mvdan-sh task 02a801cd: misleading tests removed, bugs rebalanced, now kept for training

**Original:** `db-go-mvdan-sh-02a801cd` (Ahmed)
**Fixed version:** `db-go-mvdan-sh-02a801cd-e10` (Uzair, 2026-10-24)
**Result:** kept for training

## What the task is

The agent gets **mvdan/sh** (a Go shell parser, formatter and interpreter) with real bug fixes undone, plus a bug report
written like a message from a user. It has **300 tool calls** to fix every bug. Hidden tests check each bug.

A **training** task is kept when the student passes **10–60%** of 8 attempts, a strong model passes, and most failures are
on the skills the task trains (running out of tool calls does not count).

## Before: the original task

| # | Bug | Student (Ahmed) |
|---|---|---|
| m01 | Several syntax nodes report the wrong end position | 5/8 |
| m02 | zsh rejects the `;|` case terminator | 7/8 |
| m03 | The formatter does not space nested closing parentheses `( (foo) )` | 0/8 |
| m04 | A `<<-` heredoc's inline comment is printed twice | 0/8 |

Full passes: 0/8. Verdict: too hard.

## The defects we found

1. **Tests that expected the bugs.** The start code still had the old versions of two tests: one asserted the wrong end
   positions and even said `// TODO: want 22` (giving the answer away), others expected `( (foo))`. A correct fix made
   these visible tests fail.
2. **Unfair grading** (fixed earlier in the pipeline): one big test held cases of several bugs, so m03/m04 looked
   unsolvable.
3. **A false install failure** in the grader made DeepSeek lose a full pass (fixed by Ahmed, plus our follow-up fix).

## What we changed (and what we tried)

| Version | Bugs | Student | Problem |
|---|---|---|---|
| -d1 | all 4, misleading tests hidden with the whole test tables | 0/3 | agent could not see its own regressions |
| -e1 | dropped the parens bug; only the misleading end-position test hidden | 0/8 | end positions too hard (1/8) |
| -e5 | `;|`, heredoc + 2 new bugs | 3/8 | most attempts ran out of tool calls on the heredoc bug |
| -e8 | no slow or hard bug | 7/8 | too easy |
| **-e10** | `;|`, end positions + 2 small new bugs | **3/8** | **kept** |

The final task keeps two of Ahmed's bugs with his wording, and adds two real mvdan-sh bugs:

| # | Bug, as the user sees it |
|---|---|
| m01 | zsh rejects the `;|` case terminator (Ahmed's) |
| m02 | The interpreter's `-nt` / `-ot` tests are always false when one file is missing (new) |
| m03 | Several syntax nodes report the wrong end position (Ahmed's) |
| m04 | `type -o echo` reports `command:` instead of `type:` in its error (new) |

Checked: the start code's visible tests pass with the real fix (no test expects a bug), no hints, no changelog line; all
five gates pass.

## After: the fixed task (measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| m01 `;|` in zsh | 8/8 |
| m02 `-nt` / `-ot` with a missing file | 8/8 |
| m03 wrong end positions | 3/8 |
| m04 `type` error prefix | 8/8 |

| Result | |
|---|---|
| Student full passes | **3/8 (37.5%)**, inside 10–60% |
| Qwen3.5-397B | 2/2 full passes |
| DeepSeek | 2/2 full passes |
| Failures on the skills the task trains | 4/5 (one attempt ran out of calls) |
| Verdict | **kept for training** |

## Where things are

- Task files: `crucible/data/debug-uzair/gen/store/train/tasks/db-go-mvdan-sh-02a801cd-e10/`
- Bug report: `crucible/data/debug-uzair/prompts/db-go-mvdan-sh-02a801cd-e10.json`
- Results: `crucible/data/targeting/uzair-full-mvdan-e10/report.json` ($17.65); earlier versions in `uzair-*-mvdan-*`
