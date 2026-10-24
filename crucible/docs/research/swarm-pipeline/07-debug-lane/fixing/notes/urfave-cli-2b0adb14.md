# urfave-cli task (held out): two defects fixed, now kept for the benchmark

**Original:** `db-go-urfave-cli-2b0adb14` (Ahmed)
**Fixed version:** `db-go-urfave-cli-2b0adb14-d1` (Uzair, 2026-10-23)
**Result:** kept for the benchmark

## What the task is

The agent gets **urfave/cli** (a Go library for building command-line tools) with a few real bug fixes undone, plus a
bug report written like a message from a user. It has **300 tool calls** to fix every bug. Hidden tests check each bug.

This is a **held-out (benchmark)** task: the student never trains on it. It is used to measure the student before and
after training, so it is kept when the student passes **under 30%** of attempts and a strong model can solve it. Held-out
tasks are never made easier; only real defects are fixed.

## Before: the original task

| # | Bug, as the user sees it |
|---|---|
| m01 | Running a command with an empty argument list panics |
| m02 | A nested subcommand's help does not show persistent flags inherited from its parents |
| (m03) | **Undone, but no paragraph and no test of its own** (partial flag completion after a positional argument) |
| m04 | Completing a partial flag name (`--pa`) leaves out inherited persistent flags |

| Result (Ahmed's measurement) | |
|---|---|
| Student | 0/8 at 150 calls |
| Strong models | 0 |
| Status | never measured at 300 calls |

## The defects we found

1. **A silent bug.** One of the four undone fixes had no paragraph in the report and no test of its own: its tests had
   been absorbed into m04. So to pass m04 the agent also had to fix a bug nobody described.
2. **An unstated rule in m04.** The hidden test expects inherited flags in a set order: from the root command down to the
   parent. The report only said "the command's own flags followed by the inherited persistent flags". DeepSeek listed
   them nearest-parent first, a fair reading, and failed exactly those cases.

## What we changed

1. **Applied the silent fix in the start code**, the remedy Ahmed's own plan gives for this kind of defect. The task now
   has 3 bugs, each with its own paragraph and tests, and keeps its real link (m03 needs m02).
2. **Added the missing order to m03** (old m04): "…the inherited persistent flags that apply to it, *listed from the root
   command down to its parent*…". It describes the expected behaviour, not where the code is.
3. Everything else is Ahmed's wording. Checked: no hidden tests or hints in the start code; the changelog is old and names
   none of the fixes.

## After: the fixed task (measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| m01 empty argument list panics | 6/8 |
| m02 nested help misses inherited flags | 5/8 |
| m03 partial flag completion misses inherited flags (needs m02) | 0/8 |

| Result | |
|---|---|
| Student full passes | **0/8** (under the 30% limit) |
| Qwen3.5-397B | 1/2 full passes |
| DeepSeek | 1/2 full passes |
| Failures on the skills the task tests | 8/8 |
| Verdict | **kept for the benchmark** |

## Where things are

- Task files: `crucible/data/debug-uzair/gen/store/heldout/tasks/db-go-urfave-cli-2b0adb14-d1/`
- Bug report: `crucible/data/debug-uzair/prompts/db-go-urfave-cli-2b0adb14-d1.json`
- Results: `crucible/data/targeting/uzair-heldout-full/report.json` (shared with h3; $26.51 for both)
