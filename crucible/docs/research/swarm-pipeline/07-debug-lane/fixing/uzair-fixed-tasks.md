# Debug tasks fixed by Uzair

I worked on 10 of Ahmed's debugging tasks, none of them usable at first. **8 are now fixed and kept; 2 were dropped.**

- **Training task:** kept when the small model (the student) fully solves it 10–60% of the time and a strong model solves it.
- **Benchmark task:** kept when the student solves it under 30% of the time and a strong model solves it.

## Kept (8)

| Task | Type | What was wrong | What we did | Student now |
|---|---|---|---|---|
| go-toml-a02114e3 | Training | Too hard: one bug was never solved | Removed that bug | 3/8 |
| black-90117fa0 | Training | Most attempts ran out of steps | Swapped a slow bug for a quicker one | 3/8 |
| urfave-cli-2b0adb14 | Benchmark | One bug was not in the bug report; one rule was missing | Removed that bug; added the rule | 0/8 |
| h3-a6cd4654 | Benchmark | The changelog gave away a fix; the test environment was broken | Removed the hint; fixed the environment | 1/8 |
| excelize-824ac2ce | Benchmark | Two bugs were not in the bug report | Removed those two bugs | 2/8 |
| excelize-65be17e3 | Benchmark | One bug was not in the bug report; one rule was missing | Removed that bug; added the rule | 0/8 |
| carbon-a783fb3a | Training | Too easy (7/8) | Moved it to training; added four bugs | 1/8 |
| mvdan-sh-02a801cd | Training | Too hard; two tests still expected the bugs (one even gave the answer) | Hid the misleading test; kept 2 of its bugs, replaced 2 with other real bugs | 3/8 |

A strong model solved every task above.

## Dropped (2)

| Task | Type | Why not | Status |
|---|---|---|---|
| mvdan-sh 04056a74 | Training | Mostly the same bugs as another mvdan-sh task | Dropped |
| black-ec71b918 | Training | All its bugs are also in another black task | Dropped |
