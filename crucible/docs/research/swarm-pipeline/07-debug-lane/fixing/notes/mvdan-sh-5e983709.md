# mvdan-sh 5e983709: two eased versions measured, neither kept yet

**Original:** `db-go-mvdan-sh-5e983709` (Ahmed, 4 bugs; set aside on 2026-10-22 for its fake link)
**Versions:** `-e1` (m04 dropped; Haseeb, 2026-10-23), `-e2` (the backquote bug swapped for a same-line control-flow bug; 2026-10-24)
**Result:** `-e1` is in band twice (student 3/8) but budget-bound on one bug; `-e2` is too easy by one pass (5/8). Set
aside with the numbers below; both builds, reports and measurements are kept.

## What the task is

**mvdan/sh** (Go shell parser, formatter, interpreter) with real fixes undone, a bug report in a user's voice, 300 tool
calls, hidden tests per bug.

## Before: Ahmed's task (4 bugs)

| # | Bug, as the user sees it | Student fixed it |
|---|---|---|
| m01 | `$[incomp` is not recovered with `RecoverErrors` on, while `$((incomp` is | 6/8 |
| m02 | inside backquotes inside double quotes, `\"` is kept literally (interpreter runs the wrong command, formatter changes the meaning) | 2/8 |
| m03 | `(( ! 0 ))` is printed as `((!0))`, which history expansion breaks | 5/8 |
| m04 | a lone trailing backslash at the end of a file is not idempotent | **0/8** |

0/8 full passes; the link between bugs was fake (`mvdan-sh.md`).

## `-e1`: m04 dropped, `--no-link`, strip mode (B `c2cf6f25`)

Milestones follow commit history: m01 = `!` spacing (58a8c02), m02 = `$[` recovery (ed1e963), m03 = backquotes
(27c134b). Ahmed's paragraphs re-keyed, word for word. Checks and gates clean.

| Run | Student | Strong models | Failures | Verdict |
|---|---|---|---|---|
| quick | 1/3 | DeepSeek 1/1 | one at the 300-call limit | go on |
| full 1 (`haseeb-5e983709-e1-full`, $22.13) | **3/8**, m01 0.63 m02 0.88 m03 0.50 | DeepSeek 2/2, 397B 0/2 | 5 labelled: **3 at the 300-call limit**, all with the backquote bug undone; 2 skill gaps | `off_intent` (on-intent 0.4; flag "budget-bound … re-measure at a larger limit") |
| full 2, report eased (`…-r2-full`, $22.54) | **3/8**, m01 0.75 m02 0.88 m03 0.50 | DeepSeek 1/2 (a `path:install`), 397B 1/2 | again **3 of 5 at the limit** with the backquote bug undone | `off_intent` (0.4) |

The easing (K6: exact observed and expected output for the interpreter and the formatter, the parse equivalence with
`"$(echo "foobar")"`, what must not change; K2: "three separate problems") changed nothing about the budget: the cost is
in the bug (a lexer and parser change with new state), not in the wording. The limit stays at 300 by the rules.

## `-e2`: the backquote bug swapped for `5beb6ded`, strip mode

Fixes 58a8c02, ed1e963, 5beb6ded ("keep trailing control-flow commands on the same line": `main "$@"; exit $?` split
onto two lines breaks scripts that shells read line by line; 12 cases, local printer fix, no changelog citation). The
search for a swap (`lh_debug_links.py mvdan-sh go 2026-05-01 0`): 66 fixes, 25 usable, 0 real links; the other small
printer candidates were 2255122b (a heredoc's inline comment printed twice, 1 case) and 3aa9ec1f (closing-paren
spacing, cited as #876 in the changelog, which the number rule strips). New paragraph in Ahmed's style; lint 0.

The first `-e2` build came out in revert mode and had stale assertions (four visible test functions fail with the
correct fix applied, e.g. the table still expecting `cat <<DOC; exit 1` split); rebuilt in strip mode: clean (the only
oracle-time failures are `chmod 0000; cd` cases that fail as root, which the pipeline already excludes).

| Run | Student | Strong models | Failures | Verdict |
|---|---|---|---|---|
| full (`haseeb-5e983709-e2-full`, $17.91) | **5/8**, ci 31–86%; m01 0.63 m02 0.88 m03 0.88; calls 159–260, no budget stops | DeepSeek 1/2 (the other masked), 397B 2/2 | 3 labelled, all on-intent skill gaps; one "miss" is a dud attempt that stopped after one call | **`too_easy`** (0.625 > 0.6) |

## Where this leaves the task

- `-e1` has the right difficulty (3/8 twice, both strong models pass) and fails only the on-intent rule because budget
  stops count against it. If the lead ever allows a larger limit for mvdan-sh, or relabels budget stops on a bug this
  size, `-e1` is kept as measured.
- `-e2` is one pass over the ceiling with a dud attempt among the misses; a re-measurement could land either side.
- Next knobs if the lead wants a kept version: swap the trivial `$[` recovery (m02, 4-line fix, 7/8) for a moderate
  printer bug such as 3aa9ec1f, keeping three bugs (adding a fourth would push the 159–260 call counts into the limit).

Strip-mode caveat as in `mvdan-sh-dc951aeb-e1.md`: the fixes' table entries stay readable in `syntax/filetests_test.go`
and `syntax/printer_test.go`; the test functions are stripped. The `visible_test_data` flag fired for this task.

## Where things are (Haseeb's machine, gitignored; records in `fixing/results/`)

- Tasks: `crucible/data/debug-haseeb-fixing/gen/store/train/tasks/db-go-mvdan-sh-5e983709-e1` and `…-e2`
  (`subset --repo mvdan-sh --fixes ed1e963,27c134b,58a8c02 --id db-go-mvdan-sh-5e983709-e1 --parent db-go-mvdan-sh-5e983709 --no-link`;
  `subset --repo mvdan-sh --fixes 58a8c02,ed1e963,5beb6ded --id db-go-mvdan-sh-5e983709-e2 --parent db-go-mvdan-sh-5e983709 --no-link --hide strip`)
- Reports: `fixing/prompts/db-go-mvdan-sh-5e983709-e1.json` (eased text), `…-e2.json`
- Gates and measurements: `fixing/results/db-go-mvdan-sh-5e983709-e1/` (two full runs), `…-e2/`
