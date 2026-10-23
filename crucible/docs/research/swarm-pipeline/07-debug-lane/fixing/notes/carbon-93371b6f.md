# carbon-93371b6f: moved from the benchmark to training, kept

**Task:** `db-go-carbon-93371b6f` (Ahmed; the same three fixes and the same report, rebuilt in the training store by
Haseeb on 2026-10-23)
**Result:** kept for training (`train`: student 3/8, DeepSeek 1/2, on-intent 0.8)

## What the task is

The agent gets **dromara/carbon** (a Go date/time library) with three real bug fixes undone, all in the date-difference
methods, plus a bug report written like a user's message. It has **300 tool calls** to fix all three. Hidden tests check
each bug separately (the suite's subtests only, never the suite parent).

| # | Bug, as the user sees it | Fix |
|---|---|---|
| m01 | Differences in days and weeks are computed from elapsed seconds, so across a daylight-saving change they disagree with adding days | `515cf63` |
| m02 | The difference in seconds ignores fractions, so a 0.2 s gap across a second boundary counts as a full second | `5c6dffc` |
| m03 | Differences in years and months compare each date's own fields, so two time zones give the wrong answer | `93491ca` |

## Before

Ahmed built carbon as a **held-out (benchmark)** repository. Measured as held out: student **3/7** (one attempt masked),
DeepSeek 1/2, Qwen3.5-397B 0/2, every labelled failure a skill gap. That is inside the training band (10–60%) but above the
benchmark's ceiling (under 30%), so the verdict was `out_of_band`. Its "link" between bugs was also fake (a testify suite
parent held every bug's cases).

## What we changed

1. **carbon left the held-out draw** (the lead's call, 2026-10-23): `"carbon"` removed from `HELD_OUT_REPOS` in
   `crucible/src/crucible/factory/lanes/windows.py` (it stays a debug-lane source). Done in its own commit with the reason.
   Consequence for the lead: a held-out store that still holds carbon windows now fails `benchmark.py`'s `bad_split` check.
2. **Rebuilt in the training store with `--no-link`** (`subset --repo carbon --fixes 515cf63,5c6dffc,93491ca --no-link`),
   B = head `7aa45ed` (2026-10-20), revert mode. Same id, since the fixes are the same.
3. **Report unchanged**: Ahmed's `db-go-carbon-93371b6f.json`, verbatim.

Checks before measuring: author lint 0 findings; the hidden subtest names are not in the start tree; no `.git`, no
changelog; the whole visible suite passes with the correct fix applied; gates S6 S5 S8 S3 S10 all pass (S6 with 0 edges).
The quick pass was skipped (Ahmed's measurement of the identical task stands).

## After (8 attempts at 300 tool calls)

| Bug | Student fixed it |
|---|---|
| m01 calendar days across DST | 3/8 |
| m02 whole seconds | 6/8 |
| m03 years/months across time zones | 8/8 |

| Result | As held out (Ahmed) | As training (now) |
|---|---|---|
| Student full passes | 3/7 | **3/8 (37.5%)**, ci 14–69% |
| Average progress | 0.43 | 0.71 |
| Strong model | DeepSeek 1/2, 397B 0/2 | DeepSeek **1/2**, 397B 0/2 |
| Budget stops | 0 | 0 (60–158 calls) |
| Failure labels | 4 skill_gap | 4 skill_gap + 1 disputed (judge: spec_defect, auditor: skill_gap); on-intent 0.8 |
| Verdict | out_of_band (benchmark rule) | **train (kept)** |

Notes for the lead:
- **The disputed label.** One student counted pure calendar dates, so Aug 5 → Aug 4 13:14:59 gave −1 where the
  repository's own example expects 0, and it concluded the report and the existing tests were "incompatible". The auditor
  calls it a skill gap (the report says "the way months are counted": add whole days and back off when that overshoots,
  which three students did). If the lead wants the paragraph tighter, one clause would do ("whole days actually elapsed:
  a day that has not fully passed counts 0, in either direction"); not changed here, to stay with Ahmed's text.
- **DeepSeek's miss** was a rules violation, not a wrong fix: it added four `scratch_*_test.go` files inside the repository
  (`path:protected`) at 45 calls. Its other attempt passed cleanly.

Full run $15.36.

## Where things are (Haseeb's machine, gitignored)

- Task: `crucible/data/debug-haseeb-fixing/gen/store/train/tasks/db-go-carbon-93371b6f`
- Gates: `crucible/data/debug-haseeb-fixing/gates/db-go-carbon-93371b6f.json`
- Report: `crucible/data/debug/prompts/db-go-carbon-93371b6f.json` (Ahmed's, unchanged)
- Measurement: `crucible/data/targeting/haseeb-carbon-full` (W&B online, entity cruciblee)
