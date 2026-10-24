# unstorage task (held out): made harder, now just above the benchmark bar

**Original:** `db-node-unstorage-d7aee0b7` (Ahmed; defects fixed by Uzair, see `unstorage-d7aee0b7.md`)
**Harder versions:** `db-node-unstorage-d7aee0b7-h1` (4 bugs) and `db-node-unstorage-d7aee0b7-h2` (6 bugs) (Azaan, 2026-10-23)
**Result:** not kept yet. **The student is very close to the 30% bar (3/8 = 37.5%)**: the task can either stay a
benchmark task after one more change, or become a training task, where it is already in the band. That is the lead's
call (see the end).

Held out: the bug reports, hidden tests and traces stay in `crucible/data/` (never committed); this note describes the
bugs only as a user sees them.

## What the task is

The agent gets **unstorage** (a Node.js key-value storage library) with real bug fixes undone and a bug report written
like a message from a user, 300 tool calls. It is a **held-out (benchmark)** task: kept only if the student fully passes
**under 30%** of attempts and a strong model fully passes. `azaan.md`: make it harder by adding real unstorage fixes.

## Before

Ahmed's 2 bugs in the Cloudflare cache driver (storing a value throws a TypeError; keys go into cache URLs unescaped, the
second needs the first). After Uzair's Node fixes the student solved it 3/3 in 46-75 calls: too easy.

## What we changed

1. **h1: added 2 real fixes from other parts of the library** (Ahmed's two kept, with their real link):
   the storage server does not decode paths returned by a custom `resolvePath`; the memory driver drops an item at once
   when its ttl is longer than a timer can wait (about 24.8 days).
2. **h2: added 2 more** (the server fix `5ab2f71` was tried first and refused: it is not a real fix on its own at B, a later
   fix rewrote its line): watching a prefixed storage reports keys with the prefix and events from outside it; a prefix
   base containing `?` is accepted.
3. **Report:** Ahmed's two paragraphs word for word; one new paragraph per added bug in his style, written against its
   hidden tests (the memory one names the timer limit and that no timer may be left, because the hidden tests check
   both); the opening lists the parts of the library in use. Lint `findings=0` (the first h2 draft named two changed
   functions and was reworded).
4. **Checks:** no hidden test title or `.git` in the start code, the changelog lines citing the undone fixes removed by
   the pipeline; the visible suite on the fixed code fails only the same offline-only tests (GitHub and MongoDB drivers) as
   the start code. Gates S6 S5 S8 S3 S10 pass for both (45 and 60 min).

## After (300 calls, k = 8 student, 2 + 2 strong)

| Result | Original (Uzair) | h1 (4 bugs) | h2 (6 bugs) |
|---|---|---|---|
| Student full passes | 3/3 | 4/8 (50%) | **3/8 (37.5%)** |
| 95% interval | | 22-78% | **14-69%** |
| Average progress | | 0.78 | 0.83 |
| Mean tool calls (of 300) | 46-75 | 55 | 62; every attempt stopped by itself ("done") |
| DeepSeek-V4.1-Flash | 1/1 | 1/2 | **1/2** |
| Qwen3.5-397B | | 1/2 | 0/2 |
| Audit of the failed attempts | | 3/4 skill gap | **5/5 skill gap** |
| Verdict | too easy | out of band | **out of band** (needs < 30%) |

h2 per bug (student of 8 / strong models of 4):

| Bug, as the user sees it | Student | Strong |
|---|---|---|
| watch on a prefixed storage reports prefixed and outside keys | 8 | 4 |
| a prefix base containing `?` is accepted | 8 | 4 |
| server: paths from a custom `resolvePath` are not decoded | 7 | 4 |
| **memory driver: a ttl longer than ~24.8 days expires at once** | **5** | **1** |
| Cloudflare cache: storing a value throws a TypeError (Ahmed's) | 6 | 4 |
| Cloudflare cache: keys not escaped in cache URLs (Ahmed's, needs the one above) | 6 | 4 |

- The two bugs added last are fixed in every attempt: they made the task longer, not harder. The memory-driver bug is
  the one that fails attempts (3 of 5 student failures, 3 of 4 strong attempts).
- The student stops early with most of its budget unused; its failures are missed bugs, not budget.
- Cost: $7.56 (h1) and $11.80 (h2).

## Benchmark or training: the lead's call

- **Stay a benchmark task:** 3/8 is one pass above the bar and 8 attempts cannot settle it (interval 14-69%). Either
  re-measure on fresh seeds (k = 8 more), or swap the two bugs everyone fixes for two harder real unstorage fixes (h3) and
  test it the cheap way (cheap gates, quick pass) before a full run.
- **Become a training task:** as training it is **in the band already** (37.5% is inside 10-60%, DeepSeek passes, 5/5
  failures on the intended skill). But the held-out rule puts all of an owner's repositories on one side: moving `unjs`
  to training also moves `db-node-h3-a6cd4654`, which is kept as a benchmark task (it is on the lead's open-decision list:
  "whether `dromara` (carbon) and `unjs` (unstorage, h3) move to training").

## Where things are

- Task files: `crucible/data/debug-agent2/gen/store/heldout/tasks/db-node-unstorage-d7aee0b7-h1/` and `-h2/`
- Bug reports (private): `crucible/data/debug-agent2/prompts/db-node-unstorage-d7aee0b7-h1.json`, `-h2.json`
- Gate records: `crucible/data/debug-agent2/gates/`
- Results: `crucible/data/targeting/agent2-uns-h1-full/`, `crucible/data/targeting/agent2-uns-h2-full/` (`report.json`)
