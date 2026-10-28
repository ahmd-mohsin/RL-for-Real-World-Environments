# huma task: a new task, kept for training

**Task:** `db-go-huma-815e1ff8` (Uzair, 2026-10-28)
**Repository:** danielgtaylor/huma (new: never used for a debug task before)
**Result:** kept for training on the first full run

## What the task is

The agent gets **huma** (a Go framework for HTTP APIs: you declare inputs as structs with tags, and it parses requests,
validates them and writes the OpenAPI document) with four real bug fixes undone. It also gets a bug report written like
a message from a user. It has **300 tool calls** to find and fix every bug. Hidden tests check each bug separately.

All four fixes are from Jul-Oct 2026, after the student was released (no memorised fix).

The model we train (the "student") is Qwen3.6-35B-A3B. It tries the task 8 times. A **full pass** means it fixed every
bug in one attempt.

## The bugs

| # | Bug, as the user sees it | Student fixed it |
|---|---|---|
| m01 | A query parameter typed as a named slice of numbers (`type IDs []ID`) crashes the request with 500; its `minimum` / `uniqueItems` checks never run | 6/8 |
| m02 | With both a `Body` and a `RawBody` field, the OpenAPI document lists two request content types instead of one | 7/8 |
| m03 | An exploded query array (`query:"x,explode"`) ignores its default when the parameter is absent | 8/8 |
| m04 | A JSON-array default (`default:"[1,2]"`) on an array parameter is rejected with 422 | 6/8 |

**Linked bugs:** m04's hidden test also checks an exploded query array with a JSON default, so it only passes once m03
is fixed too. Checked by hand: with m04 fixed and only m03 still broken, m04's own test fails on exactly that line.

## How it was built

1. Mined huma's 44 fixes since 2026-05-01 with the pipeline (`lh_debug.py mine`); one group had a real link.
2. Wrote the bug report from what the hidden tests actually check and from the failures seen on the start code
   (public API only, no file, function or test names; the two error texts the validation check looks for are quoted).
3. Checks before measuring: lint 0 findings; gates S6, S5, S8, S3 (60 clean reruns of 953 existing tests) and S10
   (0 false failures in 60 grades) all pass; no hidden test in the start code; no changelog or docs hint; the fixed
   code passes every visible test (no stale test).

## Results at 300 tool calls

| Quick look | |
|---|---|
| Student | 1/3 full passes |
| DeepSeek | 1/1 |

| Full run | |
|---|---|
| Student full passes | **4/8 (50%)**, inside the 10-60% training band |
| Average progress (0 = nothing fixed, 1 = all fixed) | 0.84 |
| Attempts that ran out of tool calls | 0 (all 8 stopped on their own, 168 calls on average) |
| Failures on the skill the task trains | **4/4** (all "skill gap": the student said "done" with one or two bugs left) |
| Qwen3.5-397B | 2/2 full passes |
| DeepSeek | 0/2: both stalled on Tinker's side (calls went from ~5 s to no reply) and hit the 90-minute limit; not the task |
| Verdict | **train (kept)** |
| Cost | $2.88 quick look + $22.53 full run |

## Where things are

- Task: `crucible/data/debug-uzair/gen/store/train/tasks/db-go-huma-815e1ff8/`
- Bug report: `crucible/data/debug-uzair/prompts/db-go-huma-815e1ff8.json`
- Gates: `crucible/data/debug-uzair/gates/db-go-huma-815e1ff8.json`
- Results: `crucible/data/targeting/uzair-huma-815e1ff8-quick/`, `crucible/data/targeting/uzair-huma-815e1ff8-full/report.json`
