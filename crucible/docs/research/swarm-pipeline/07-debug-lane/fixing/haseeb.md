# Haseeb: Go tasks to fix

Read `README.md` first (rules, setup, the loop, the checklist). Per-milestone numbers below are Ahmed's measurements
from before our grading fixes, so treat them as a guide and re-measure.

## 1. mvdan-sh 5e983709: drop m04

- Repo `mvdan-sh` (`mvdan/sh`), cutoff `2026-05-01` (the default), B = head (`c2cf6f2`, 2026-10-19).
- Fixes (Ahmed's milestone order): m01 `ed1e963` (`$[...]` missing `]` recovery), m02 `27c134b` (backquotes in double
  quotes), m03 `58a8c02` (space after `!` in arithmetic), m04 `d3da88a` (lone trailing backslash at end of file).
- Student @300: m01 6/8, m02 2/8, m03 5/8, **m04 0/8**; 0/8 full passes.
- Its only "link" was fake (a Go parent test), so it needs `--no-link`.
- Plan: `subset --repo mvdan-sh --fixes ed1e963,27c134b,58a8c02 --id db-go-mvdan-sh-5e983709-e1 --parent db-go-mvdan-sh-5e983709 --no-link`.
  Uzair checked this builds and passes S6. With no link the milestones follow commit history (`58a8c02` becomes m01):
  map Ahmed's paragraphs by bug.
- Check the start code's `CHANGELOG.md`: it listed several of these fixes (e.g. #1083); the pipeline now strips entries
  that cite them, so confirm none is left.

## 2. mvdan-sh dc951aeb: drop m03

- Fixes: m01 `6622086` (`$-` in the interpreter), m02 `3c1ff2a` (`<<-` heredoc indentation), m03 `cfbc33c`
  (BinaryNextLine for `[[ ]]` / arithmetic), m04 `a135720` (keep `then`/`do` on their own line).
- Student: m01 7/8, m02 7/8, **m03 0/8, m04 0/8**. m04's 0/8 was forced by a grading bug (its tests included a parent test
  holding m03's cases), now fixed, so m04's real rate is unknown.
- Plan: drop m03 with `--no-link`: `--fixes 6622086,3c1ff2a,a135720 --id db-go-mvdan-sh-dc951aeb-e1 --parent db-go-mvdan-sh-dc951aeb`.
- Its changelog said "as well as `$-` - #1398" (m01): confirm it is stripped.
- Keep only these two mvdan-sh variants: 04056a74 shares three fixes with 5e983709, and 02a801cd's fix list is unknown.

## 3. carbon-93371b6f: move carbon to training

- Repo `carbon` (`dromara/carbon`), cutoff `2026-05-01`. Fixes: `515cf63`, `5c6dffc`, `93491ca` (three "difference"
  bugs: DiffInDays, DiffInSeconds, years/months comparison).
- Ahmed measured it as held out: student **3/7 at 300 calls**, already in the training range but too easy for the
  benchmark. Its link was fake (a testify suite parent), so it needs `--no-link`.
- Plan: (a) remove `"carbon"` from `HELD_OUT_REPOS` in `crucible/src/crucible/factory/lanes/windows.py` (line ~169) **in
  its own commit**, explaining why (the lead's call: carbon cannot be a benchmark task); (b) rebuild in the training store:
  `subset --repo carbon --fixes 515cf63,5c6dffc,93491ca --no-link`; (c) Ahmed's report: `crucible/data/debug/prompts/db-go-carbon-93371b6f.json`;
  (d) measure.
- carbon-a783fb3a (`cf685dc`, `d9a6370`, 7/8) is too easy and has 2 bugs; leave it.

## 4. echo-61eca05b: make it harder

- Repo `echo` (`labstack/echo`), cutoff `2026-05-01`. Fixes: `ec9515a`, `cc07fa9` (a real link). Student 7/8: too easy.
- Plan: add a third echo fix. List usable fixes and links with `scripts/lh_debug_links.py echo go 2026-05-01 0 ec9515a,cc07fa9`;
  pick one whose bug a user could report by its symptoms. Build `--fixes ec9515a,cc07fa9,<new>`, keep Ahmed's two
  paragraphs (`crucible/data/debug/prompts/db-go-echo-61eca05b.json`) and write one new paragraph in the same style
  (symptoms and expected behaviour only; no file, function or test names). Aim for the student at 10–60%.

## Status (2026-10-24, Haseeb)

| Task | Done | Verdict |
|---|---|---|
| carbon-93371b6f | moved to training, rebuilt `--no-link`, report unchanged; `notes/carbon-93371b6f.md` | **train** (3/8, DeepSeek 1/2, on-intent 0.8) |
| mvdan-sh dc951aeb-e1 | m03 dropped, `--no-link`; `notes/mvdan-sh-dc951aeb-e1.md` | **train** (1/8, DeepSeek 2/2, on-intent 0.86) |
| echo-61eca05b-e2 | two bugs added (df5edd9, 6a390cb) at B 3d084be9; `notes/echo-61eca05b-e2.md` | **train** (4/8, DeepSeek 2/2, 397B 2/2, on-intent 1.0) |
| mvdan-sh 5e983709 | `-e1` (m04 dropped): two full runs 3/8 but budget-bound on the backquote bug (off_intent 0.4); `-e2` swaps it for 5beb6ded (same-line control flow): built, lint 0, checklist clean; gates pending | in progress |

Build commands (store `crucible/data/debug-haseeb-fixing`, clones in `src/`):
- `subset --repo carbon --fixes 515cf63,5c6dffc,93491ca --no-link`
- `subset --repo mvdan-sh --fixes 6622086,3c1ff2a,a135720 --id db-go-mvdan-sh-dc951aeb-e1 --parent db-go-mvdan-sh-dc951aeb --no-link`
- `subset --repo echo --fixes ec9515a,cc07fa9,df5edd9,6a390cb --b 3d084be902 --id db-go-echo-61eca05b-e2 --parent db-go-echo-61eca05b --no-link --hide strip`
- `subset --repo mvdan-sh --fixes 58a8c02,ed1e963,5beb6ded --id db-go-mvdan-sh-5e983709-e2 --parent db-go-mvdan-sh-5e983709 --no-link --hide strip`

Gates and measurement reports of the kept tasks: `results/<id>/`. Spend: about $100 in total (W&B online, entity cruciblee).
