# go-toml task: made easier, now kept for training

**Original:** `db-go-go-toml-a02114e3` (Ahmed)
**Easier version:** `db-go-go-toml-a02114e3-e1b` (Uzair, 2026-10-21)
**Result:** kept for training (the first kept task)

## What the task is

The agent gets the **go-toml** library (Go code that reads and writes TOML config files) with a few real bug fixes
undone. It also gets a bug report written like a message from a user. It has **300 tool calls** to find and fix every
bug. Hidden tests check each bug separately.

The model we train (the "student") is Qwen3.6-35B-A3B. It tries the task 8 times. A **full pass** means it fixed every
bug in one attempt.

## Before: the original task (4 bugs)

| # | Bug, as the user sees it | Student fixed it |
|---|---|---|
| m01 | A commented-out value that spans several lines gives TOML that does not parse back | 5/8 |
| m02 | Putting a table where the struct cannot hold one gives a vague error, with no position or key | **0/8** |
| m03 | Decoding ignores a custom `UnmarshalText` on named types (strings, numbers, bools, dates) | 6/8 |
| m04 | Encoding ignores `MarshalText` on named string types, and writes invalid UTF-8 as the wrong escape | 4/8 |

**Linked bugs:** m04's tests encode a value and then decode it back, so they only pass once m03 (decoding) is fixed
too. Every task needs at least one link like this.

| Result at 300 tool calls | |
|---|---|
| Student full passes | **0/8** |
| Average progress (0 = nothing fixed, 1 = all fixed) | 0.47 |
| Strong model (DeepSeek) | 2/2 full passes |
| Verdict | too hard |

**Why it failed:** the student never fixed m02, so a full pass was impossible.

## What we changed

1. **Dropped m02.** Its fix is left in the starting code, and its paragraph is removed from the bug report. The other
   three bugs and the m03 → m04 link stay.
2. **Kept the report wording.** The other three paragraphs are Ahmed's text, unchanged.
3. **Fixed a defect in our first build.** The first easier build (`-e1`) still had an old test that expected the buggy
   output, so a correct fix looked wrong. DeepSeek edited that test and lost the pass. We rebuilt it as `-e1b` with
   that outdated test removed.

The bugs are renumbered: old m03 is now m02, old m04 is now m03.

## After: the easier task (3 bugs)

| Bug | Before | After |
|---|---|---|
| Commented multi-line values (m01) | 5/8 | 5/8 |
| Table placement errors (was m02) | 0/8 | dropped |
| `UnmarshalText` on named types (was m03, now m02) | 6/8 | 5/8 |
| `MarshalText` and invalid UTF-8 (was m04, now m03) | 4/8 | 5/8 |

| Result at 300 tool calls | Before | After |
|---|---|---|
| Student full passes | 0/8 (0%) | **3/8 (38%)** |
| Average progress | 0.47 | 0.54 |
| Strong model (DeepSeek) | 2/2 | 2/2 |
| Verdict | too hard | **kept for training** (target: 10–60%) |

- Every attempt finished on its own within 105–228 tool calls. None ran out of calls or broke a rule.
- All 5 failed attempts failed on what the task is meant to teach: finding several bugs, the linked bugs, or
  reproducing a bug from the report.
- Another strong model, Qwen3.5-397B, got 0/2 (it stopped early).
- Cost: $19.40 to measure it ($25.87 including the first `-e1` build).

## Where things are

- Task files: `crucible/data/debug-uzair/gen/store/train/tasks/db-go-go-toml-a02114e3-e1b/`
- Bug report: `crucible/data/debug-uzair/prompts/db-go-go-toml-a02114e3-e1b.json`
- Results: `crucible/data/targeting/uzair/report.json`
- Code changes: branch `tasks/uzair` (`b3764cd` adds the command that builds an easier version, `295becd` fixes the
  old-test defect)
- Agent traces (W&B): https://forge.coreweave.com/wandb/cruciblee/crucible-factory/runs/d17818718fd7
