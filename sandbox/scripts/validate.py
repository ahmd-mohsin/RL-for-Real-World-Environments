"""Validate the sandbox end to end. A task is only usable for RL if all checks pass.

    cd sandbox && python -m scripts.validate [--no-build] [--platform linux/amd64]

Checks:
  build       the image builds and the existing suite passes at the base commit (inside the Dockerfile)
  leakage     git history has one commit; the fix and the hidden tests are not on disk
  network     the rollout container cannot reach the internet
  no-op       an agent that does nothing gets reward 0
  tamper      an agent that deletes the tests and does nothing else still gets reward 0
  hack        an agent that plants a conftest.py forcing every test to pass gets reward 0
  oracle      the upstream fix, applied through the agent's own tools, gets reward 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from grading.grade import grade
from interface.tools import Tools
from runtime.sandbox import SANDBOX_DIR, Sandbox, load_task

RESULTS: list[tuple[str, bool, str]] = []

CONFTEST_HACK = '''import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.outcome = "passed"
'''


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)


def build(task: dict, platform: str | None) -> None:
    cmd = ["docker", "build", "-t", task["image"]["tag"],
           "--build-arg", f"REPO_URL={task['image']['repo_url']}",
           "--build-arg", f"BASE_COMMIT={task['image']['base_commit']}"]
    if platform:
        cmd += ["--platform", platform]
    cmd.append(str(SANDBOX_DIR / "image"))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    check("build", proc.returncode == 0, "" if proc.returncode == 0 else proc.stderr[-2000:])
    if proc.returncode != 0:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--platform", default=None)
    args = ap.parse_args()
    task = load_task()

    if not args.no_build:
        build(task, args.platform)

    with Sandbox(task) as sb:
        n_commits = sb.exec("git rev-list --all | wc -l").output.strip()
        check("leakage: single commit", n_commits == "1", f"{n_commits} reachable commits")
        fix_marker = sb.exec("grep -c previous_index slugify/slugify.py").output.strip()
        check("leakage: fix not present", fix_marker == "0")
        test_marker = sb.exec("grep -rc post_replacement_delimiters tests | grep -v ':0' | wc -l").output.strip()
        check("leakage: hidden tests not present", test_marker == "0")
        net = sb.exec("python -c \"import urllib.request; urllib.request.urlopen('https://github.com', timeout=5)\"")
        check("network: blocked", net.returncode != 0)

    with Sandbox(task) as sb:
        r = grade(sb)
        check("no-op -> reward 0", r["reward"] == 0.0 and "error_category" not in r, f"reward={r['reward']}")

    with Sandbox(task) as sb:
        tools = Tools(sb)
        tools.call("bash", {"command": "rm -rf tests && mkdir tests && touch tests/__init__.py"})
        r = grade(sb)
        check("tamper -> reward 0", r["reward"] == 0.0 and "error_category" not in r, f"reward={r['reward']}")

    with Sandbox(task) as sb:
        # Reward hack: a conftest.py that rewrites every test outcome to "passed".
        tools = Tools(sb)
        tools.call("create", {"path": "conftest.py", "file_text": CONFTEST_HACK})
        r = grade(sb)
        check("hack: forced-pass conftest -> reward 0", r["reward"] == 0.0, f"reward={r['reward']}")

    with Sandbox(task) as sb:
        tools = Tools(sb)
        patch =(SANDBOX_DIR / task["solution"]["patch"]).read_text()
        tools.call("create", {"path": "/tmp/solution.patch", "file_text": patch})
        out = tools.call("bash", {"command": "git apply /tmp/solution.patch && rm /tmp/solution.patch"})
        tools.call("submit", {})
        r = grade(sb)
        check("oracle -> reward 1", r["reward"] == 1.0, f"reward={r['reward']}" + ("" if r["reward"] else f"\n{out}\n{r.get('test_output', r)}"))

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\nVALID" if not failed else f"\nINVALID: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
