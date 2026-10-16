# M17 task factory, study 07: first end-to-end run on CPU (partial, stopped)

- **Run:** 2026-10-14/15, CPU-only Mac, Docker CPU sandboxes, OpenRouter.
  - Models: `qwen/qwen3.7-flash` (author), `deepseek/deepseek-v4-flash` (auditor, reviewer); solver rungs
    `meta-llama/llama-3.1-8b-instruct`, `openai/gpt-oss-20b`, `qwen/qwen3.7-flash`.
- **Spend:** $0.79 of the $20 cap ($0.53 discovery, $0.26 trial run).
- **Status:** mining and discovery completed. The per-task machine ran on 1 trial task. Family acceptance did **not** run.
  The user stopped the run to save tokens. Raw results are in `runs/`.

## 1. Executable lane: mining real fix + test commits (`factory/lanes/executable.py`)

| | Count |
|---|---|
| Candidate commits in 11 local Python repos | 295 |
| **Validated tasks:** new tests fail on the parent and pass on the real fix, in a Docker CPU sandbox | **210** |
| Dropped: no fail-to-pass test | 75 |
| Dropped: tests did not collect after the fix | 8 |
| Dropped: environment mismatch (49 tests fail after the fix) | 1 |
| Dropped: a test passed before and fails after | 1 |

Validated per repo:

| Repo | Tasks |
|---|---|
| boltons | 39 |
| python-semver | 37 |
| more-itertools | 36 |
| parse | 33 |
| cachetools | 19 |
| python-tabulate | 19 |
| humanize | 16 |
| python-slugify | 6 |
| inflection | 2 |
| jmespath.py | 2 |
| isodate | 1 |

The ground truth for every task is an `ObservationRecord` of type `real_pr_tests` (the commit sha plus a hash of the test
files).

## 2. Discovery: who fails, and why (05 §4)

- **65 tasks entered discovery.** 6 were dropped at spec authoring: the author model's JSON was cut off at `max_tokens`
  (`finish=length`). This is a factory setting, not a model weakness. Spec authoring needs a larger output budget.
- **The remaining 59 tasks** were run by 3 solver rungs × 2 attempts = 354 attempts:

| Solver | Pass | Fail | Masked |
|---|---|---|---|
| `meta-llama/llama-3.1-8b-instruct` | 1 | 117 | 0 |
| `openai/gpt-oss-20b` | 28 | 90 | 0 |
| `qwen/qwen3.7-flash` | 21 | 97 | 0 |

- **The auditor** (a different family) classified every failure. It was trusted on all 59 tasks: seeded controls (a World
  bug, a lazy solution, an unstated format) were recalled 177/177.

| Failure cause | Count (of 304) |
|---|---|
| skill gap | 193 |
| budget | 40 |
| unlabelled (the auditor abstained) | 39 |
| spec defect | 13 |
| world bug | 8 |
| unstated format | 7 |
| env flake | 2 |
| verifier defect | 2 |

- **What it shows:**
  - The rungs separate: llama-3.1-8b solves 1/118 attempts, gpt-oss-20b 28/118, so the tasks have a usable difficulty spread
    for small models.
  - 30 of 304 failures (spec defect, world bug, unstated format, verifier defect) are task defects, not model failures.
    These are exactly the failures the machine must route back to the author or verifier, or drop. Without the auditor,
    they would train the model on noise.
  - 39 abstentions show that the auditor refuses to guess when the trace gives no evidence, as designed.
- **Intent candidates** were written with status `candidate` and recorded as pending in the decision store. No intent was
  auto-approved. **They await the user's approval.**

## 3. Per-task machine (trial)

- **1 task** (python-semver) entered the full machine with `provisional_intent=True`. The run reached a state with no
  matching transition and **paused with a decision request** (escalate). The machine treats this as designed: it never
  ends as `done`.
- The agent was fixing an adapter bug exposed there when the run was stopped. A fake client did not accept extra
  keyword arguments.
- **Not yet shown on real tasks:** a full pass through probes, hackers, the second verifier, flake (30 reruns × 2 levels),
  the gate and select; family acceptance (gold set, orthogonality, transfer proxy).

## 4. Next (option B, deferred)

1. Raise the spec-authoring `max_tokens` and fix the adapter bug.
2. Run the machine on ~10 tasks across 3 repos.
3. Run family acceptance with a v1 gold set of ≥ 36 items per class, filled by mutants, valid alternatives and real solver
   outputs.
4. Report the funnel per stage, the verifier cards, and the spend per role.
