# unstorage task (held out): defects fixed, but too easy for the benchmark

**Task:** `db-node-unstorage-d7aee0b7` (Ahmed; same id, rebuilt by Uzair 2026-10-23)
**Result:** not kept (too easy)

## What the task is

The agent gets **unstorage** (a Node.js key-value storage library) with two real bug fixes undone in its Cloudflare cache
driver, plus Ahmed's bug report. It is a **held-out (benchmark)** task, so it is kept only if the student passes **under
30%** of attempts. Held-out tasks are never made harder or easier.

| # | Bug, as the user sees it |
|---|---|
| m01 | Storing a value through the driver throws a TypeError |
| m02 | Keys go into cache URLs unescaped, so one key can overwrite another's entry (needs m01) |

## Before

Ahmed's measurement: student 3/8 at 150 calls, already above the 30% limit, on the broken Node environment.

## What we changed

The same fixes as for h3: the changelog lines naming both fixes were removed from the start code, and the two Node
environment bugs (path quoting, dependencies lost on reset) were fixed in the pipeline. The task and report are unchanged.

## After (quick check at 300 calls)

| Result | |
|---|---|
| Student full passes | **3/3**, in 46–75 tool calls |
| DeepSeek | 1/1 |
| Verdict | **too easy for the benchmark**; set aside |

With a working environment the student solves it every time, so it cannot show improvement after training.
