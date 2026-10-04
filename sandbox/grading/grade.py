"""Part 4: Grading hooks. Turn the agent's final state into a reward.

The agent's container is never trusted. Inside it the agent could have planted a conftest.py
that forces every test to pass, edited pytest in site-packages, or deleted tests. Those are real
reward hacks (scripts/validate.py checks for them). So, as SWE-bench does, grading only takes
the agent's *diff* out of its container and replays it in a fresh one:

  1. capture the agent's diff against the base commit, in the rollout container
  2. drop every file diff that touches a protected path (tests, conftest.py, pytest config)
  3. start a fresh container from the same image; apply the filtered diff
  4. apply the hidden tests patch and copy in the verifier script
  5. run the verifier; reward = 1.0 iff it exits 0

If the agent's own diff does not apply, that is the agent's fault and scores 0. If the hidden tests
do not apply, that is an infrastructure fault: it is reported with error_category so a trainer can
mask the sample instead of training on a false 0.
"""

from __future__ import annotations

import fnmatch
import re
import time

from runtime.sandbox import SANDBOX_DIR, Sandbox

TESTBED_CORRUPTED = "reward/testbed_corrupted"
_MODEL_PATCH_REMOTE = "/tmp/_model.patch"
_TESTS_PATCH_REMOTE = "/tmp/_hidden_tests.patch"
_SCRIPT_REMOTE = "/testbed/run_tests.sh"


def split_patch(patch: str) -> list[tuple[str, str]]:
    """Split a unified git diff into (path, file_diff) chunks."""
    out = []
    for chunk in re.split(r"(?m)^(?=diff --git )", patch):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", chunk)
        if m:
            out.append((m.group(2), chunk))
    return out


def filter_protected(patch: str, protected: list[str]) -> tuple[str, list[str]]:
    kept, dropped = [], []
    for path, chunk in split_patch(patch):
        if any(fnmatch.fnmatch(path, pat) for pat in protected):
            dropped.append(path)
        else:
            kept.append(chunk)
    return "".join(kept), dropped


def capture_model_patch(sb: Sandbox) -> str:
    sb.exec("git add -A")
    return sb.exec("git diff --cached --binary base").output


def grade(rollout: Sandbox) -> dict:
    g = rollout.task["grading"]
    tests_patch = (SANDBOX_DIR / g["tests_patch"]).read_text()
    script = (SANDBOX_DIR / "grading" / "run_tests.sh").read_text()

    # 1-2. take only the agent's source changes out of its container
    model_patch = capture_model_patch(rollout)
    graded_patch, dropped = filter_protected(model_patch, g["protected_paths"])
    result = {"model_patch": model_patch, "dropped_protected_paths": dropped}

    # 3-5. replay in a clean container
    with Sandbox(rollout.task) as clean:
        if graded_patch.strip():
            clean.write_file(_MODEL_PATCH_REMOTE, graded_patch)
            applied = clean.exec(f"git apply --binary {_MODEL_PATCH_REMOTE}")
            if applied.returncode != 0:
                return {**result, "reward": 0.0, "resolved": False,
                        "error": "model_patch_does_not_apply", "detail": applied.output}

        clean.write_file(_TESTS_PATCH_REMOTE, tests_patch)
        clean.write_file(_SCRIPT_REMOTE, script)
        applied = clean.exec(f"git apply {_TESTS_PATCH_REMOTE}")
        if applied.returncode != 0:
            return {**result, "reward": 0.0, "error_category": TESTBED_CORRUPTED,
                    "error": "apply_tests_failed", "detail": applied.output}

        start = time.time()
        res = clean.exec(g["test_command"], timeout=g["verifier_timeout_sec"])
        return {
            **result,
            "reward": 1.0 if res.returncode == 0 else 0.0,
            "resolved": res.returncode == 0,
            "verifier_returncode": res.returncode,
            "test_duration_sec": round(time.time() - start, 2),
            "test_output": res.output[-5000:],
        }


if __name__ == "__main__":
    # From sandbox/: python -m grading.grade
    # Grades an untouched sandbox, which should give reward 0.0 (the bug is still there).
    import json

    with Sandbox() as sb:
        print(json.dumps({k: v for k, v in grade(sb).items() if k != "test_output"}, indent=2))
