# mvdan-sh dc951aeb: made easier, now kept for training

**Original:** `db-go-mvdan-sh-dc951aeb` (Ahmed, 4 bugs; set aside on 2026-10-22 for its fake link)
**Easier version:** `db-go-mvdan-sh-dc951aeb-e1` (Haseeb, 2026-10-23, built with `--no-link` after the lead allowed tasks
without an executed interaction)
**Result:** kept for training (`train`: student 1/8, DeepSeek 2/2, all failures on the intended skills)

## What the task is

The agent gets **mvdan/sh** (a Go shell-script parser, formatter and interpreter) with real bug fixes undone, plus a bug
report written like a user's message. It has **300 tool calls** to fix every bug. Hidden tests check each bug separately.

## Before: Ahmed's task (4 bugs)

| # | Bug, as the user sees it | Student fixed it |
|---|---|---|
| m01 | `$-` (the shell's option flags) expands to nothing in the interpreter | 7/8 |
| m02 | A `<<-` heredoc nested 3+ blocks deep is mis-indented | 7/8 |
| m03 | With "operators at the start of the line" on, `&&`/`\|\|` in `[[ ]]` stay at the end | **0/8** |
| m04 | A multi-line `if`/`while`/`for` condition loses `then`/`do` on its own line | **0/8** |

| Result at 300 tool calls | |
|---|---|
| Student full passes | 0/8 |
| Average progress | 0.44 |
| Strong model (DeepSeek) | 1/2 |
| Link between bugs | fake (one Go test held cases of several bugs; see `mvdan-sh.md`) |

## What we changed

1. **Dropped m03** (`cfbc33c`, BinaryNextLine for `[[ ]]` and arithmetic): its fix stays applied in the start code and its
   paragraph is removed from the report. The student never fixed it, so a full pass was impossible.
2. **Built with `--no-link`**: mvdan-sh has no real link between any of its fixes, which the lead now allows. The
   milestones follow commit history: `$-` (6622086) = m01, the heredoc (3c1ff2a) = m02, then/do (a135720) = m03.
3. **Kept Ahmed's wording.** The opening and the three paragraphs are his text, re-keyed to the new numbering.
4. Build at B = `c2cf6f25` (2026-10-19), strip mode (the changed test functions are removed from the visible tree).

Checks before measuring: author lint 0 findings; no hidden test name, no `.git` and no changelog line about the undone
fixes in the start tree (the `$-` entry, #1398, is stripped); the whole visible suite passes with the correct fix applied
(no stale assertion); gates S6 S5 S8 S3 S10 all pass (S6 with 0 edges, `allow_independent`).

## After: the easier task (3 bugs)

| Bug | Before | After (8 attempts) |
|---|---|---|
| `$-` in the interpreter (m01) | 7/8 | 8/8 |
| `<<-` heredoc indentation (m02) | 7/8 | 8/8 |
| `then`/`do` on their own line (was m04, now m03) | 0/8 | 8/8 |

| Result at 300 tool calls | Before | After |
|---|---|---|
| Student full passes | 0/8 (0%) | **1/8 (12.5%)**, ci 2–47% |
| Average progress | 0.44 | 1.00 |
| Strong model | DeepSeek 1/2 | DeepSeek **2/2**, Qwen3.5-397B 0/2 |
| Budget stops | — | 0 (105–293 calls) |
| Failure labels | — | 7 skill_gap, on-intent 6/7 (0.86) |
| Verdict | too hard, fake link | **train (kept)** |

**Why only 1 of 8 is a full pass when every attempt fixed all three bugs:** six attempts failed the **regression
check**: their then/do change also kept the newline when the condition is a single line (14 hidden table cases,
`TestPrintTable/#048, #119, #156, #282, #285–#287`), which the report rules out ("Single-line conditions should still
collapse as before"). One attempt ran a real `go get` (an install, a path violation). Both judges called every miss a
skill gap. Note for the lead: those regression cases live in the stripped table test, so the student can only get them
right from the report's words, not by running tests.

**Visible test data (strip mode), for the lead.** Strip removes the changed test *functions* (`TestPrintTable`,
`TestPrintBinaryNextLine`, `TestRunnerRun` are gone from the start tree) but leaves the package-level test *tables* at B,
so the fixes' new and changed entries stay readable in `syntax/printer_test.go` (the `if foo && … then` / `while … do`
cases) and `interp/interp_test.go` (the two `$-` cases), with their expected outputs. They cannot be run. Revert mode
would be worse here: a135720 and 6622086 *modify* existing entries, so revert leaves live tests asserting the old, buggy
output (checked: the revert-mode build of the sibling task fails four visible test functions with the correct fix
applied). The pipeline records this condition as the `visible_test_data` fairness flag when the changes sit in a file it did
not strip (it fired for the sibling task's `filetests_test.go` and for echo's `perf_bench_test.go`), but **not here**: the
entries sit in files whose test functions were stripped, so the flag logic took them as hidden. Two pipeline items,
then: the flag should also cover table variables left in stripped files, and entry-level stripping would remove the
caveat altogether. The report states the same expectations in words, and the measured failures (over-applied fixes, a `go get`)
show no sign of the entries being exploited. Entry-level stripping of Go table tests would remove the caveat.

Quick pass before the full run: student 2/3, DeepSeek 1/1, $5.77. Full run $17.26.

## Where things are (Haseeb's machine, gitignored)

- Task: `crucible/data/debug-haseeb-fixing/gen/store/train/tasks/db-go-mvdan-sh-dc951aeb-e1`
- Gates: `crucible/data/debug-haseeb-fixing/gates/db-go-mvdan-sh-dc951aeb-e1.json`
- Report: `fixing/prompts/db-go-mvdan-sh-dc951aeb-e1.json` (copied from the store's `prompts/`)
- Measurements: `crucible/data/targeting/haseeb-dc951aeb-e1-quick`, `…/haseeb-dc951aeb-e1-full` (W&B online, entity cruciblee)
