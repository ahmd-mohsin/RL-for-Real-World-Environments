# Azaan: Rust, Python and Node tasks to fix

Read `README.md` first (rules, setup, the loop, the checklist). Per-milestone numbers are Ahmed's or Uzair's measurements;
re-measure after any change.

## 1. pulldown-cmark e072a230: swap the bug the student never fixes

- Repo `pulldown-cmark` (`pulldown-cmark/pulldown-cmark`), cutoff **`2025-09-01`** (pass `--since 2025-09-01`), B = head.
- Ahmed's fixes: m01 `671cf7c` (container fence at end of input), m02 `7e01b77` (hard line breaks), m03 `0424329` (CR
  line endings), m04 `def3889` (blank last line of an unclosed block; needs m03).
- Uzair already dropped m02 (`db-rust-pulldown-cmark-e072a230-e1`, see `notes/pulldown-cmark-e072a230.md`): student
  0/11, with m01 11/11, CR endings 3/11, **blank last line 0/11**; DeepSeek 3/3, so it is fair but too hard.
- Plan: replace `def3889` (blank last line) with an easier pulldown-cmark fix, using `--no-link` (no other fix links to the
  CR fix). Usable fixes at head (`scripts/lh_debug_links.py pulldown-cmark rust 2025-09-01 0` lists them again):
  `68afb08` wikilink overflow, `07bae24` task-list item, `9670b08` closing fence with a tab, `10e55b3` code span indent,
  `c61583e` container-extensions interrupt, `5b0ef8c` block quote indent, `953e44a` inline HTML container prefix,
  `ad1622b` inline HTML scanner, `e3e5670` link title escape after newline (plus the four above).
  Pick one whose bug is easy to report by symptoms, e.g. `--fixes 671cf7c,0424329,<new> --id db-rust-pulldown-cmark-e072a230-e2 --parent db-rust-pulldown-cmark-e072a230 --since 2025-09-01 --no-link`.
- Report: Ahmed's m01 and m03 paragraphs (`crucible/data/debug/prompts/db-rust-pulldown-cmark-e072a230.json`) plus one
  new paragraph in his style. If the student is still near 0, try a second easier swap before giving up.
- Rust attempts are slow (the first fix lands around call 160–260), so prefer a bug that is quick to reproduce.

## 2. black-ae5830b7: drop m02

- Repo `black` (`psf/black`), cutoff **`2025-09-01`**, B = head. Fixes: m01 `fc33c25` (f-string after `# fmt: off`),
  m02 `13e97b4` (case guard at small line lengths), m03 `650983f` (leading tabs in docstrings), m04 `c9b1148` (module
  docstring blank line).
- Student @300: m01 8/8, **m02 1/8**, m03 5/8, m04 1/8; 0/8 full passes.
- Its "link" (m04 needs m03) is fake: both bugs only add cases to the same case file `tests/data/cases/docstring.py`, which
  is graded as one test. So m04 can only pass with m03 too; that is fine now (no link required), just know it.
- Plan: `--fixes fc33c25,650983f,c9b1148 --id db-python-black-ae5830b7-e1 --parent db-python-black-ae5830b7 --since 2025-09-01 --no-link`,
  **default hide mode (revert)**: strip mode leaves black's new test cases and their expected output visible.
- Skip black-ec71b918: it is 3 of these same 4 fixes.
- Note fc33c25 is also m01 of the kept `db-python-black-90117fa0-e1`; if the reviewers mind the overlap, swap it for
  another black fix (`scripts/lh_debug_links.py black python 2025-09-01 0`).

## 3. unstorage-d7aee0b7 (held out): make it harder

- Repo `unstorage` (`unjs/unstorage`), cutoff **`2025-09-01`**, held out (builds go to the held-out store
  automatically). Fixes: `172b615` (setItem TypeError), `884502a` (unescaped cache keys); a real link.
- After the Node fixes on `tasks/uzair` the student solves it **3/3** (46–75 calls), so it is too easy for the benchmark
  (needs under 30%). See `notes/unstorage-d7aee0b7.md`.
- Plan: add one or two more unstorage fixes (list them with `scripts/lh_debug_links.py unstorage node 2025-09-01 0 172b615,884502a`),
  keep Ahmed's two paragraphs (`crucible/data/debug/prompts/db-node-unstorage-d7aee0b7.json`) and write the new ones in
  his style. Measure: student must stay **under 30%** and a strong model must fully pass.
- Node tasks need `tasks/uzair`: without its two Node fixes every attempt either fails to start or cannot run its tests.
