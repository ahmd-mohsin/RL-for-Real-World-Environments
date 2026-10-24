# black task: one bug dropped and two stale tests hidden, now kept for training

**Original:** `db-python-black-ae5830b7` (Ahmed)
**Easier version:** `db-python-black-ae5830b7-e1` (Azaan, 2026-10-23)
**Result:** kept for training

## What the task is

The agent gets **Black** (the Python code formatter) with a few real bug fixes undone, plus a bug report written like a
message from a user. It has **300 tool calls** to find and fix every bug. Hidden tests check each bug separately. The
student is Qwen3.6-35B-A3B; a **full pass** means every bug fixed in one attempt.

## Before: the original task (4 bugs, measured at 300 calls)

| # | Bug, as the user sees it | Student fixed it |
|---|---|---|
| m01 | Black crashes when an f-string follows `# fmt: off` inside brackets | 8/8 |
| m02 | a `case case if ...` guard cannot be formatted at a very small line length | **1/8** |
| m03 | leading tabs in docstrings expand to 8 columns instead of 4 | 5/8 |
| m04 | a long multi-line module docstring gets an extra blank line before its closing quotes | 1/8 |

Student 0/8 full passes, no strong model passed. Its "link" (m04 needs m03) is not a real one: both fixes only add
cases to the same case file, `tests/data/cases/docstring.py`, which is graded as one test.

## What we changed

1. **Dropped m02 (the case guard, `13e97b4`),** the bug the student almost never fixes, with `subset --no-link`. Its fix
   stays applied in the start code. The other three fixes and Ahmed's paragraphs for them are unchanged.
2. **B is Black `287d5709` (2026-10-20), not today's head.** At head, a 2026-10-22 commit (`d1deb4af`, comments that end a
   `fmt: off` region inside brackets) covers the f-string crash, so `fc33c25` no longer counts as a real fix there.
3. **Two stale visible tests hidden (a pipeline fix, see below).** In revert mode the start code had the old copies of two
   hidden tests that still expect the bug: `test_lines_with_leading_tabs_expanded` asserting 8-column tabs, and the old
   `tests/data/cases/docstring.py` case. With Black's correct fix in, both failed in the agent's own tree, so a correct fix
   made its own tests go red (the go-toml-e1 trap). Ahmed's original has the same two, which may explain part of its low
   m03 and m04 numbers.
4. **Report:** Ahmed's opening and his m01, m03 and m04 paragraphs, word for word, keyed m01, m02, m03. Lint `findings=0`.
5. **Checks before measuring:** no hidden test name, case file or changelog line in the start code; all 674 visible
   tests pass on the fixed code (2 failed before the fix in 3); gates S6 S5 S8 S3 S10 pass (3.7 h on this machine: Black's
   suite takes about 2 minutes per run in a container).

**The pipeline fix** (`generators/debug.py`, on `tasks-fixing/azaan`): `precheck` also runs the agent's tests on B's
source; a milestone test that passes in A' and fails there expects the bug and is hidden in round 2, by its definition,
or, for a data-driven case like `test_simple_format[docstring]`, by removing that case file from A' (the grader writes
B's file back). Uzair's `subset --hide-tests` does the same by hand; the two work together.

## After: e1 (3 bugs, measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| m01 f-string after `# fmt: off` inside brackets | 8/8 |
| m02 leading tabs in docstrings | 8/8 |
| m03 module docstring blank line (needs m02: the shared case file) | **2/8** |

| Result | Before | After |
|---|---|---|
| Student full passes | 0/8 | **2/8 (25%)** |
| Average progress | 0.47 | 0.75 |
| Attempts that ran out of calls | | **0/8** (mean 203 calls) |
| DeepSeek-V4.1-Flash | 0 | **2/2 full passes** |
| Qwen3.5-397B | 0 | 0/2 |
| Audit of the failed attempts | | **6/6 skill gap** (GLM-5.3; GLM-5.3-Flash 5/6) |
| Verdict | not kept | **kept for training** (target 10-60%) |

- Every failed attempt fixed m01 and m02 and missed m03: the stall is the last dependent step.
- Cost: $23.72 for the full measurement (student $11.69, strong models $8.95, audit $3.08).
- `fc33c25` (m01 here) is also m01 of the kept `db-python-black-90117fa0-e1`. If the reviewers mind the overlap, swap it
  for another Black fix (`azaan.md`).

## Where things are

- Task files: `crucible/data/debug-agent2/gen/store/train/tasks/db-python-black-ae5830b7-e1/`
- Bug report: `crucible/data/debug-agent2/prompts/db-python-black-ae5830b7-e1.json` (copy in `../prompts/`)
- Gate record: `crucible/data/debug-agent2/gates/db-python-black-ae5830b7-e1.json`
- Results: `crucible/data/targeting/agent2-black-e1-full/report.json`; W&B run offline under `crucible/data/observe/`
