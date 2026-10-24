# echo-61eca05b: made harder (2 bugs → 4), now kept for training

**Original:** `db-go-echo-61eca05b` (Ahmed, 2 bugs; too easy: student 7/8)
**Harder version:** `db-go-echo-61eca05b-e2` (Haseeb, 2026-10-23; a 3-bug `-e1` in between was still too easy)
**Result:** kept for training (`train`: student 4/8, DeepSeek 2/2, Qwen3.5-397B 2/2, every failure on the intended skills)

## What the task is

The agent gets **labstack/echo** (the Go web framework, v5 main branch) with real bug fixes undone, plus a bug report
written like a message from an API team. It has **300 tool calls** to fix every bug. Hidden tests check each bug separately.

## Before: Ahmed's task (2 bugs)

| # | Bug, as the user sees it | Student fixed it |
|---|---|---|
| m01 | A bind error for a bad query/form value does not say which field failed | — |
| m02 | JSON bodies are decoded leniently: trailing data is accepted, a truncated body gives "unexpected EOF" | — |

Student full passes **7/8**: too easy (the band's top is 60%). Its "link" between the bugs was not a real one under the
corrected rules either (no echo fix links to another, at today's head or 30 days back).

## What we changed

1. **Base commit moved back to `3d084be9` (2026-09-21).** At today's head Ahmed's JSON fix can no longer be undone (a later
   change rewrote the same lines), so every build uses that B; all fixes below are usable there.
2. **Round 1 (`-e1`): added a third bug**, `df5edd9` (the RFC 9457 error handler writes its defaults into a *shared*
   problem-details value, altering it for the whole process). Still too easy: 7/8, every attempt fixed all three; the bug
   is found by searching for its default text.
3. **Round 2 (`-e2`): added a fourth bug**, `6a390cb` (for WebSocket upgrade requests the proxy middleware *sets*
   `X-Forwarded-For` only when none came in, instead of *appending* its own peer address to the incoming chain, so
   downstream services pick the wrong client address). Two other candidates were rejected: `8148ea9` (a group's second
   `Use` panics, and a panic aborts the whole Go test binary, so its milestone claimed 768 tests) and `48128ab` (its
   release-changelog line is not stripped because the commit message carries no issue/PR number; see the pipeline note
   below).
4. **Report:** Ahmed's two paragraphs word for word, re-keyed (the milestones follow commit history: proxy = m01, field
   name = m02, JSON = m03, shared error = m04), two new paragraphs in the same style (symptoms and expected behaviour,
   public API only), the opening extended by one clause. Author lint 0 findings (literal IP addresses had to become
   placeholders: the lint reads them as version numbers).
5. Built with `--no-link --hide strip`. Checks before measuring: no hidden test name, no `.git`, no changelog line about the
   undone fixes in the start tree; the whole visible suite passes with the correct fixes applied; gates S6 S5 S8 S3 S10
   all pass (S6 with 0 edges, `allow_independent`).

## After: the harder task (4 bugs, 8 attempts at 300 tool calls)

| Bug | Student fixed it |
|---|---|
| Proxy X-Forwarded-For for WebSocket requests (m01) | 4/8 |
| Field name in bind errors (m02, Ahmed's m01) | 8/8 |
| Strict JSON bodies (m03, Ahmed's m02) | 6/8 |
| Shared problem-details error (m04) | 7/8 |

| Result | Before (2 bugs) | `-e1` (3 bugs) | **`-e2` (4 bugs)** |
|---|---|---|---|
| Student full passes | 7/8 | 7/8 | **4/8 (50%)**, ci 22–78% |
| Average progress | — | 1.00 | 0.78 |
| Strong models | — | DeepSeek 2/2, 397B 2/2 | DeepSeek **2/2**, 397B **2/2** |
| Calls used | — | 112–290 | 74–143 |
| Budget stops | — | 0 | 0 |
| Failure labels | — | 1 skill_gap | **4 skill_gap, on-intent 4/4** |
| Verdict | too easy | too easy | **train (kept)** |

Quick passes: `-e1` 2/3 students solved it but lost the full pass to a grading rule (below); `-e2` 2/3, DeepSeek 1/1.
Spend: `-e1` quick $4.89 + full $8.55; `-e2` quick $1.33 + full $8.24.

## Two things for the lead (pipeline, not this task)

- **`go mod tidy` counts as a package install** (`factory/pathcheck.py`, `GO_MOD_FETCH`). Students reproduce a bug by
  writing a scratch program under `/tmp` with its own `go.mod` (`replace github.com/labstack/echo/v5 => /ws`) and run
  `go mod tidy` there; with `--network none` and `GOPROXY=off` it fetches nothing, but the gate-severity flag removes the
  full-pass bonus. It cost `-e1` 2 of 3 quick-pass solves and 1 of 8 in its full run (none in `-e2`'s full run). Same
  class as the pnpm false positive in the README. **Already fixed on dev** by the scratch rule in `pathcheck.py` (a package-manager run inside a scratch module under `/tmp` is classed `scratch`, not `install`); these measurements predate it, and the stored attempts can be regraded under the new rule at no cost.
- **Changelog stripping misses fixes whose message has no number** (`langs/base.py`, `fix_refs`): it matches entries by
  the `#numbers` of the commit message and the short sha, so `48128ab` ("fix(static): preserve matched handler 404s",
  no number anywhere in the message) stays announced in the start tree's `CHANGELOG.md` with its PR URL. Matching the
  commit subject too would close it.

## Where things are (Haseeb's machine, gitignored; small records copied to `fixing/results/`)

- Task: `crucible/data/debug-haseeb-fixing/gen/store/train/tasks/db-go-echo-61eca05b-e2`
  (build: `subset --repo echo --fixes ec9515a,cc07fa9,df5edd9,6a390cb --b 3d084be902 --id db-go-echo-61eca05b-e2 --parent db-go-echo-61eca05b --no-link --hide strip`)
- Report: `fixing/prompts/db-go-echo-61eca05b-e2.json`
- Gates and measurement: `fixing/results/db-go-echo-61eca05b-e2/` (W&B online, entity cruciblee)
