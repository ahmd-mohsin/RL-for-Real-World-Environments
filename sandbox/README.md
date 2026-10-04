# Coding sandbox: one task, all four parts

A single, validated RL environment for a coding agent, built the way SWE-bench, SWE-smith and MiMo build theirs.

**Task:** fix a real bug in [`python-slugify`](https://github.com/un33k/python-slugify), taken from upstream [PR #200](https://github.com/un33k/python-slugify/pull/200).
- The sandbox starts at the parent commit `c442cd4`.
- The upstream fix is the reference solution.
- The tests the PR added are the hidden tests.

```
sandbox/
├── task.toml              the contract: ids, image, limits, tools, grading config
├── instruction.md         what the agent is told (an issue, not the fix)
├── image/Dockerfile       Part 1: Image
├── runtime/sandbox.py     Part 2: Runtime
├── interface/tools.py     Part 3: Interface (the agent's tools)
├── grading/               Part 4: Grading hooks
│   ├── tests.patch          hidden tests (never inside the image)
│   ├── run_tests.sh         hidden verifier script
│   └── grade.py             diff → fresh container → tests → reward
├── solution/solution.patch  oracle; used only to validate the task
└── scripts/validate.py    proves the task is valid and cannot be gamed
```

## The four parts

### 1. Image (`image/Dockerfile`)
A frozen filesystem in the task's starting state. Layers:
1. `python:3.12-slim`
2. system tools: git, ripgrep, procps
3. repo cloned at the base commit, plus `pip install -e . pytest`
4. hygiene

The build **runs the existing test suite** and fails if it isn't green, so a broken environment can never ship.

Hygiene, so the answer is not on disk:
- **Git history is cut to one `base` commit.** Otherwise `git log --all` would show the upstream fix.
- **No hidden tests or verifier** are in the image.
- **Build leftovers are removed** (caches, egg-info).

### 2. Runtime (`runtime/sandbox.py`)
One fresh container per rollout (`docker run --rm`), so **reset is free**: throw it away and start another.
- **Limits:** `--network none`, 2 CPUs, 2 GB memory, 512 processes.
- **Timeouts:** each command has a timeout enforced *inside* the container.

Every other part uses only these primitives: `start/stop`, `exec(cmd)`, `write_file`, `read_file`.
At scale the same interface is backed by Kubernetes pods, Modal or Daytona instead of local Docker.

### 3. Interface (`interface/tools.py`): what tools we give the agent

| Tool | Purpose | Why it's there |
|---|---|---|
| `bash(command)` | Run anything in `/testbed`: grep, run tests, inspect | The universal tool. Every harness (SWE-agent, mini-swe-agent, Claude Code, Codex) has it |
| `view(path, view_range?)` | File with line numbers, or a directory listing | Line numbers make edits reliable; ranges keep long files out of the context window |
| `str_replace(path, old_str, new_str)` | Replace one exact, unique snippet | Safer than rewriting whole files. Must match exactly once, or the call errors with no change |
| `create(path, file_text)` | Write a new file | New modules, repro scripts |
| `submit()` | End the episode | The agent decides when it is done, and grading starts |

These follow the de facto standard: SWE-agent's "agent-computer interface", Anthropic's `bash` + `text_editor` tools, and MiMo's `bash/read/write/edit`. Design rules:
- **Errors come back as text, never exceptions**, so the agent can recover.
- **Output is truncated** (20k characters) so one `cat` can't blow the context window.
- **No internet.** This blocks downloading the upstream fix.
- **No access to grading.** Hidden tests appear only in the grading container.
- **The tool set is part of the environment.** The same model can score very differently with different tools. MiMo trains across 4 tool styles (bash-only, Claude-Code-like, Codex-like, MiMo) so the model doesn't overfit one.

### 4. Grading hooks (`grading/grade.py`)
The agent's container is **never trusted**. The grader:
1. Captures the agent's diff against `base`.
2. Drops any changes to protected paths: `tests/*`, `conftest.py`, pytest and tox config.
3. Starts a **fresh container** from the same image and applies the remaining diff. If the agent's diff doesn't apply, that scores 0.
4. Applies the hidden `tests.patch` and copies in `run_tests.sh`.
5. Runs the tests (SWE-bench style):
   - **FAIL_TO_PASS:** the 2 new regression tests must now pass.
   - **PASS_TO_PASS:** the whole existing suite must still pass.

   **Reward = 1.0 if `run_tests.sh` exits 0, else 0.0.**

If the hidden tests themselves fail to apply, the result is tagged `reward/testbed_corrupted`. A trainer masks that sample instead of learning from a false 0.

## Validation (`scripts/validate.py`)

| Check | Expectation | Why |
|---|---|---|
| build | image builds, existing suite green | the environment works |
| leakage | 1 commit; the fix and hidden tests not on disk | the answer can't be read |
| network | outbound HTTP fails | the answer can't be downloaded |
| no-op | reward 0 | the task is not already solved |
| tamper | delete all tests → reward 0 | the agent can't remove the checks |
| hack | plant a `conftest.py` forcing all tests to "pass" → reward 0 | the agent can't fake the checks |
| oracle | upstream fix applied **through the agent's own tools** → reward 1 | the task is solvable with the given tools |

The `hack` check paid **reward 1.0 for doing nothing** against the first grader version, which graded inside the agent's container. That bug is why grading now replays the diff in a fresh container. Every reward hack found becomes a permanent check here.

## Run it

```bash
cd sandbox
../.venv/bin/python -m scripts.validate                  # build + all checks (first build ~1 min)
../.venv/bin/python -m scripts.validate --no-build       # checks only
../.venv/bin/python -m grading.grade                     # grade an untouched sandbox → reward 0
docker run --rm -it --network none rlsandbox/python-slugify:modern-hard-cut bash   # look around as the agent would
```

Requires Docker and Python ≥ 3.11 (stdlib only). Images built on Apple Silicon are arm64. Add `--platform linux/amd64` when building images for an x86 training cluster.

## What is not here yet
- **An agent loop** that connects a model to `TOOL_SCHEMAS` and `Tools.call`, and records the trajectory.
- **Difficulty calibration:** run the model k times and keep the task only if its pass rate is strictly between 0 and 1.
- **Packaging** as a parquet row or Harbor task directory, for verl, MiMo or TRL trainers.
