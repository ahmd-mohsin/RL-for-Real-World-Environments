# pulldown-cmark task: the bug nobody fixed swapped out, now kept for training

**Original:** `db-rust-pulldown-cmark-e072a230` (Ahmed)
**Earlier easier version:** `db-rust-pulldown-cmark-e072a230-e1` (Uzair, set aside: see `pulldown-cmark-e072a230.md`)
**New version:** `db-rust-pulldown-cmark-e072a230-e2` (Azaan, 2026-10-23)
**Result:** kept for training

## What the task is

The agent gets **pulldown-cmark** (a Rust library that turns Markdown into HTML) with a few real bug fixes undone, plus a
bug report written like a message from a user. It has **300 tool calls** to find and fix every bug. Hidden tests check
each bug separately. The student is Qwen3.6-35B-A3B; a **full pass** means every bug fixed in one attempt.

## Before: e1 (Uzair's 3-bug version, measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| container fence at end of input | 11/11 |
| CR line endings in HTML and code blocks | 3/11 |
| blank last line of an unclosed block (needs the CR fix) | **0/11** |

Student 0/11 full passes, DeepSeek 3/3: fair but too hard. Uzair found no other pulldown-cmark fix that links to the CR
fix, so the blank-last-line bug could not be swapped for a linked one.

## What we changed

1. **Swapped the blank-last-line bug (`def3889`) for an easier real fix, `5b0ef8c`** (block quote continuation after a
   tab, from `azaan.md`'s list). Built with `subset --no-link` (the lead allows tasks of separate bugs since 2026-10-23).
   It changes `parse.rs`, which neither other fix touches, and is quick to reproduce.
2. **`def3889` now stays applied in the start code.** Its two tests still fail until the CR fix is back (they need it),
   so they belong to the CR milestone now. The CR paragraph therefore carries Ahmed's blank-last-line paragraph too: a
   hidden test may only check what the report says.
3. **B is pulldown-cmark `baa6ac7` (2026-10-19), not today's head.** At head, undoing the CR fix alone no longer compiles
   (the 2026-10-23 commit `862d8eb` builds on its code), so it would not count as a real fix. The build used strip mode
   (the reverts conflict in revert mode).
4. **Report:** Ahmed's opening and paragraphs word for word (m01 = his CR paragraph followed by his blank-last-line
   paragraph, m03 = his container-fence paragraph) and one new paragraph in his style for the block quote bug (m02),
   written against its two hidden tests. Lint: `findings=0`.
5. **Checks before measuring:** no hidden test name, `.git` or changelog line in the start code; all 1,416 visible tests
   pass on the fixed code (and on the start code); gates S6 S5 S8 S3 S10 pass (10 min).

Milestones (with `--no-link` they follow commit history): m01 CR line endings (+ blank last line), m02 block quote after
a tab (new), m03 container fence at end of input.

## After: e2 (3 bugs, measured at 300 calls)

| Bug | Student fixed it |
|---|---|
| m01 CR line endings in HTML and code blocks, last line kept | **1/8** |
| m02 block quote continuation after a tab (new) | 5/8 |
| m03 container fence at end of input | 8/8 |

| Result | e1 (before) | e2 (after) |
|---|---|---|
| Student full passes | 0/11 | **1/8 (12.5%)** |
| Average progress | 0.42 | 0.58 |
| Attempts that ran out of calls | 2/8 | **0/8** (mean 194 calls) |
| DeepSeek-V4.1-Flash | 3/3 | **1/2** (the other fixed all 3 bugs but edited a protected file) |
| Qwen3.5-397B | 0/2 | 0/2 (both missed m01) |
| Audit of the failed attempts | | **7/7 skill gap** (GLM-5.3 and GLM-5.3-Flash agree) |
| Verdict | too hard | **kept for training** (target 10-60%) |

- 1/8 is near the bottom of the band (95% interval 2-47%); the attempts spread over 1, 2 and 3 bugs fixed, which is
  what step-wise credit needs.
- Cost: $26.23 for the full measurement (student $13.81, strong models $9.52, audit $2.89).

## Where things are

- Task files: `crucible/data/debug-agent2/gen/store/train/tasks/db-rust-pulldown-cmark-e072a230-e2/`
- Bug report: `crucible/data/debug-agent2/prompts/db-rust-pulldown-cmark-e072a230-e2.json` (copy in `../prompts/`)
- Gate record: `crucible/data/debug-agent2/gates/db-rust-pulldown-cmark-e072a230-e2.json`
- Results: `crucible/data/targeting/agent2-pcm-e2-full/report.json`; W&B run offline under `crucible/data/observe/`
- The Rust World image took a hand build with a patient `cargo fetch` (the pipeline's own Dockerfile, the fetch retried
  with a longer timeout): the slow route made the stock build's `cargo fetch` fail.
