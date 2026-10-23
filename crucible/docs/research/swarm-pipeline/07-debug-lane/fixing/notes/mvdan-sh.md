# mvdan-sh tasks: set aside, no real link between bugs

**Original tasks:** `db-go-mvdan-sh-dc951aeb` (the one we worked on), plus `5e983709`, `04056a74` and `02a801cd` (Ahmed)
**Result:** not kept. Under our rules these tasks can't be fixed, so they are set aside.

## What the task is

The agent gets **mvdan/sh** (a Go shell-script parser, formatter and interpreter) with 4 real bug fixes undone, plus a
bug report written like a user's message. It has **300 tool calls** to fix all 4. Hidden tests check each bug separately.

## Before: Ahmed's task dc951aeb (4 bugs)

| # | Bug, as the user sees it | Student fixed it |
|---|---|---|
| m01 | `$-` (the shell's option flags) expands to nothing in the interpreter | 7/8 |
| m02 | A `<<-` heredoc nested 3+ blocks deep is mis-indented | 7/8 |
| m03 | With "operators at the start of the line" on, `&&`/`||` in `[[ ]]` stay at the end | **0/8** |
| m04 | A multi-line `if`/`while`/`for` condition loses `then`/`do` on its own line | **0/8** |

| Result | |
|---|---|
| Student full passes | 0/8 |
| Average progress | 0.44 |
| Strong model (DeepSeek) | 1/2 full passes (at 150 calls) |
| Pipeline said | m04 depends on m02 and m03, so the bugs are linked |

## What we found

1. **The link was fake.** In Go, one test can hold many small cases. `TestPrintTable` holds both m02's case and m04's
   cases, and it only passes when every case passes. The pipeline gave the whole test to m04, so m04 looked like it
   needed m02 and m03. We applied m04's fix with m02 and m03 still broken, and all of m04's own cases passed.
   - m04's 0/8 was forced by m03's 0/8, not by m04 itself.
   - The other three mvdan-sh tasks have the same fake link (through `TestPrintFiles` and `TestParseFiles`).
2. **mvdan-sh has no real link for these bugs.** We undid every fix since May 2026 (66 fixes, at 3 versions of the
   repo) and looked for two fixes that break the same test case. None of Ahmed's 9 mvdan-sh fixes links to anything.
3. **A hint in the start code.** The repo's `CHANGELOG.md` comes from the fixed version, so it still says "Implement …
   `$-`" (m01). For `04056a74` it lists two of its bugs too.

## What we changed

- **Pipeline: fair grading for Go.** A test that holds cases from different bugs no longer counts for any bug, and two
  bugs only count as linked when they break the same case (commit `416a29e7`).
- **Pipeline: no links from compile errors.** When undoing a fix stops a whole test package from compiling, its tests
  no longer count as shared.
- With these fixes, the pipeline now reports "no executed interaction" for the mvdan-sh tasks.

## Why we stopped

Ahmed's rules say every task needs **at least one real link** between bugs and that **a gate is never relaxed**. Right
now we only make tasks easier by dropping a bug, and dropping a bug can't create a link. So none of the four mvdan-sh
tasks can be made valid.

| | Before | After |
|---|---|---|
| Link between bugs | yes (fake) | none |
| Passes Ahmed's rules | looked like it | no |
| Verdict | too hard, ease | **set aside** |

## Where things are

- Rebuilt tasks: `crucible/data/debug-uzair/gen/store/train/tasks/db-go-mvdan-sh-*`
- Bug-by-bug summaries: `crucible/data/debug-uzair/briefs/`
- Link search results: `crucible/data/debug-uzair/search/mvdan-sh-links*.json`
- Pipeline fixes: branch `tasks/uzair`
