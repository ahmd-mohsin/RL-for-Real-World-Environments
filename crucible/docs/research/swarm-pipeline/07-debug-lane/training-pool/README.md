# The debug lane's training pool (2026-10-29)

`pool.json` is the canonical list of the tasks RL trains on: owner, source note, recipe, how the recipe was recovered,
the built start commit and World image, the gates and the band. `targeting.json` is its targeting records (what
`delta_rl.py train --targeting` reads). The tasks themselves are in `crucible/data/debug-pool` (gitignored), a debug
root of training-split windows only, with its own clones.

```bash
# check the pool on disk against the manifest (add --images to compare the World image ids too)
.venv/bin/python scripts/lh_debug_pool.py check
# rewrite the targeting records from the manifest
.venv/bin/python scripts/lh_debug_pool.py targeting
```

## How the tasks got there

Most coauthor tasks were built on the coauthors' machines and only their notes, reports and gate records were pushed.
Each was rebuilt here with `lh_debug.py subset`, with the coauthor's own bug report, then linted (`author`: 0
findings for all 12) and gated again (S6, S5, S8, S3, S10: all pass). Nothing was authored, eased or changed.

- When the notes do not list the fixes, they come from the task id or the gate record. A mined id is the hash of its
  fixes in prober order, so huma's 815e1ff8 names exactly one ordered group of huma's fixes. The start commit A' is
  deterministic, so a recorded `window_a` pins B: huma, golang/net and react-hook-form reproduce theirs exactly.
- carbon-93371b6f and echo-e2 have no recorded start commit, but their gates now report the same existing suite (same
  test counts) and the same grading as Haseeb's records.
- carbon-a783fb3a-h5: the note gives no fix shas. Each added bug matches exactly one carbon fix, and their history order
  is the note's and the report's m01 to m06. B is forced (carbon has no commit after 2026-10-20). It is ready, but there
  is no recorded start commit to compare with.
- mvdan-sh-dc951aeb-e1 and black-ae5830b7-c1-516b23 were already built and gated in the main store. They were copied
  unchanged. dc951aeb-e1's tree is the main store's 2026-10-25 rebuild, and its band is the 2026-10-26 re-measure of
  that tree.

## Status

| Task | Owner | Status | Student | Band source |
|---|---|---|---|---|
| db-go-huma-815e1ff8 | Uzair | ready (start commit as recorded) | 4/8 | report.json |
| db-go-golang-net-6c36e4c8-e2 | Uzair | ready (start commit as recorded) | 4/8 | report.json |
| db-node-react-hook-form-399578e2-e1 | Uzair | ready (start commit as recorded) | 2/8 | report.json (verdict `unfair`; kept by Uzair on two runs) |
| db-go-go-toml-a02114e3-e1b | Uzair | ready | 3/8 | note |
| db-python-black-90117fa0-e1 | Uzair | ready | 3/8 | note |
| db-go-carbon-a783fb3a-h5 | Uzair | ready (recipe recovered, no recorded start commit) | 1/8 | note |
| db-go-mvdan-sh-02a801cd-e10 | Uzair | **blocked**: B not recorded (c2cf6f25 or f6d205ab), nor the `--hide-tests` id | 3/8 | note |
| db-go-carbon-93371b6f | Haseeb | ready (gates as recorded) | 3/8 | report.json |
| db-go-echo-61eca05b-e2 | Haseeb | ready (gates as recorded) | 4/8 | report.json |
| db-go-mvdan-sh-dc951aeb-e1 | Haseeb | ready (main store's build) | 1/8 | main store re-measure |
| db-rust-pulldown-cmark-e072a230-e2 | Azaan | ready | 1/8 | note |
| db-python-black-ae5830b7-e1 | Azaan | ready | 2/8 | note |
| db-python-black-ae5830b7-c1-516b23 | swarm | ready (main store's build) | 1/8 | swarm pilot record |

Unblocking mvdan-sh-02a801cd-e10 needs one fact from Uzair: B (the mvdan/sh commit his clone was at), or his
`gen/store/train/tasks/db-go-mvdan-sh-02a801cd-e10/window.json` or `gates/<id>.json` (both record the start commit).

## DeltaAI (not synced yet)

`pool.json` (`deltaai`) has the plan. In short: rsync the pool (about 0.25 GB, without `logs/` and `gen/work/`) to a new
`$W/data/debug-pool`, and pass `--data-src` to it. Then convert the 10 World images (22.2 GB uncompressed, arm64) and
`crucible-lh-base:py312` to SIFs the way `scripts/deltaai/bench.py bundle --images` does. Skip any image whose SIF is
already there with the same image id.
