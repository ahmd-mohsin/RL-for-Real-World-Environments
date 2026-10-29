# react-hook-form task: a new task, the third one

**Task:** `db-node-react-hook-form-399578e2-e1` (Uzair, 2026-10-29), built from `db-node-react-hook-form-399578e2`
**Repository:** react-hook-form/react-hook-form (new: never used for a debug task before)
**Result:** **kept for training** (Uzair, 2026-10-29) on two runs together: student 2/8, failures on skill, DeepSeek-1017 1 clean pass. The full run's own `report.json` says `unfair`, because in that run no strong model passed cleanly (see "Why it is kept").

## What the task is

The agent gets **react-hook-form** (the React form library: `useForm`, `register`, `setValue`, `reset`, field arrays)
with three real bug fixes undone, and a bug report written like a message from a user. It has **300 tool calls** to
find and fix every bug. Hidden tests check each bug separately.

All fixes are from October 2026, after the student was released (no memorised fix).

## The bugs

| # | Bug, as the user sees it | Student fixed it (full run) |
|---|---|---|
| m01 | When a validate rule, the resolver or `onInvalid` throws during a submit, the form stays in the submitting state forever (`isSubmitting` never goes back to false) | 6/8 |
| m02 | `setValue` on an object that contains a field array updates the values but leaves the old rows on screen | 7/8 |
| m03 | After `reset()`, a field array stops enforcing its own rules (`required`, `minLength` on the array) when the rules object is a constant, so the form submits | 7/8 |

**No link** between these three bugs (built with `--no-link`). See "How it was built" for why.

## How it was built

1. Mined about 35 repositories never used before (fixes since 2026-05-01). Only react-hook-form had a real link: its
   fix "drop the async result of a field unregistered while it validates" (#13912) has a hidden test that unregisters a
   field with `keepValue` and registers it again, and that only passes once the `keepValue` fix (#13891) is in too.
   yq's links were fake (two fixes' cases side by side in one scenario table); the rest had none.
2. **The linked task** `db-node-react-hook-form-399578e2` = the `setValue` bug above + `keepValue` + the async bug
   (m03 needs m02). All five gates passed, report lint 0. **Too hard:** student 0/3 in the quick look and **0/8** in
   the full run; the async bug was never fixed (the cases where a field is registered again under the same name, and
   the resolver path of `trigger`, failed in 9 of 9 attempts). Qwen3.5-397B 0/1. The full run was stopped after the
   student arm to save budget (the verdict could only be "too hard").
3. As for golang/net and isort, the link could not reach the band, so the task was rebuilt **without the link**: both
   linked fixes are left applied (the async fix alone would leave visible tests failing in the start code, which gives
   the other bug away), and two other October fixes were added (submit state when validation throws; field array rules
   after reset).
4. A 4-bug version (`-e2`, adding "isValid stays false on valid defaults when a child subscribes first") was 0/3 in the
   quick look, so the 3-bug version was measured.
5. Checks before measuring: lint 0 findings (the report avoids the names of the functions the fixes changed, such as
   the submit handler and the field array hook, and describes what the user sees); gates S6, S5, S8, S3 (60 clean
   reruns) and S10 (0 false failures in 60) all pass; no hidden test in the start code; the fixed code passes every
   visible test (no stale test); no fairness flags.
6. **Pipeline fix:** react-hook-form's CHANGELOG.md named every fix in reworded form ("`setValue` not updating field
   array rows when a parent object is set"), and the changelog strip missed them. It now also matches stemmed content
   words and a subject that starts with a gitmoji (`langs/base.py`, one test). On this changelog it removes exactly the
   task's entries.

## Results at 300 tool calls

| Quick look | Linked (`399578e2`) | No link (`-e1`) | 4 bugs (`-e2`) |
|---|---|---|---|
| Student | 0/3 | 2/3 (the failure broke a regression test) | 0/3 |
| Qwen3.5-397B | 0/1 | 0/1 (missed the `setValue` bug) | not run |

| Full run (`-e1`) | |
|---|---|
| Student full passes | **2/8 (25%)**, inside the 10-60% training band |
| Average progress (0 = nothing fixed, 1 = all fixed) | 0.83 |
| Attempts that ran out of tool calls | 0 (all 8 stopped on their own, 94 calls on average) |
| Failures on the skill the task trains | **6/6 labelled skill gap** (5 counted on intent); 5 of the 6 broke a regression check the report describes (no state published after a reset supersedes a pending submit; a thrown falsy value is kept) |
| Qwen3.5-397B | 0/2 (it missed the `setValue` bug, as in every Qwen attempt on this repository) |
| DeepSeek-V4.1-Flash-1017 | 0/2 in the full run, but one attempt **fixed all three bugs with no regression** and lost the full pass only for leaving three reproduction tests in `src/__tests__/` (the protected-path gate; the prompt says test changes cost the full-completion credit) |
| Verdict in `report.json` | `unfair`: "no frontier solver passes it" |
| Fresh DeepSeek sample (`uzair-rhf-e1-fair2`, `--seed-offset 1`, frontier only) | **1/2 clean full pass** (all three bugs, no regression, no gate failed); the other attempt ran out of steps |
| Cost | linked task $6.26 quick + $13.16 full; `-e1` $2.93 quick + $17.99 full + $9.05 fresh DeepSeek; `-e2` $5.41 quick; **$54.80 in all** (of the $60 allowed) |

## Why it is kept

Every rule for a training task holds: the student is in the band (2/8), its failures are on the skill, and a strong
model solves the task cleanly. The clean strong-model pass came in a second, frontier-only run
(`uzair-rhf-e1-fair2`, its episodes in `results/db-node-react-hook-form-399578e2-e1/fresh-deepseek-episodes.jsonl`), so
no single `report.json` says `train`. Uzair counted the two runs together (2026-10-29); the student arm was not
re-sampled. A single-report confirmation (student x8 + DeepSeek-1017 x2 in a new folder) would cost about $15.

The task also breaks the link rule (built with `--no-link`, like golang/net and isort): the linked version was
measured and is too hard (0/8).

## Where things are

- Tasks: `crucible/data/debug-uzair/gen/store/train/tasks/db-node-react-hook-form-399578e2{,-e1,-e2}/`
- Bug reports: `crucible/data/debug-uzair/prompts/db-node-react-hook-form-399578e2{,-e1,-e2}.json`
- Gates: `crucible/data/debug-uzair/gates/db-node-react-hook-form-399578e2{,-e1,-e2}.json`
- Results: `crucible/data/targeting/uzair-rhf-quick/`, `uzair-rhf-full/` (linked), `uzair-rhf-e1-quick/`,
  `uzair-rhf-e1-full/report.json`, `uzair-rhf-e2-quick/`
