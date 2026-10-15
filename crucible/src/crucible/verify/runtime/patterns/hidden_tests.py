"""Hidden tests with verdicts from a results channel, not exit codes (DESIGN §8.1, P26, A13). Standard library only.

The seed graders judged pytest and doctest suites by return code (task01/01v2/03), so ``os._exit(0)`` when a runner is
loaded, or a PUBLIC ``DOCTEST_CONFIG_DISABLE`` that compiles every test away, passed. Here:

- the suite is a **root-owned copy** made before any candidate code runs; the interpreter runs isolated (``-I``) with
  grader-owned code (plugin, tests, consumers) first on ``sys.path`` and the candidate's ``candidate_path`` appended
  *after the standard library* (``Trusted.python``), so candidate modules shadow neither the suite nor the stdlib;
- the runner writes one record per test outcome to a file descriptor **opened by the grader** and inherited by the
  child (``pass_fds``), plus a final ``finish`` record written at session end. ``os._exit`` skips the finish record;
  a disabled suite runs too few tests. Both fail: the verdict requires the finish record, ``collected >= min_count``
  (or ``== expected``), every collected test passed, and at most ``allowed_skips`` skips;
- the return code is logged, never used;
- in-process forgery of records needs code that targets the runner; ``static`` flags runner references in candidate
  code (``runner_tamper``), and the expected-count manifest is computed by the grader on the oracle at authoring time.

``pytest_suite`` and ``doctest_suite`` are the two runners the seed tasks need; both return a ``Verdict``.

M7 audit (2026-10-12):
- pytest auto-loaded ``pytest11`` entry points from the candidate's install, and such a plugin's
  ``pytest_runtest_logreport`` could set every report to "passed" (a wrong library scored 1 in a root container).
  Autoload is off (``PYTEST_DISABLE_PLUGIN_AUTOLOAD``) and outcomes are recorded from ``call.excinfo`` in an outermost
  ``pytest_runtest_makereport`` wrapper, not from the mutable report;
- a candidate that replaced ``unittest.TestCase.assert*`` passed unittest-style suites with a wrong implementation:
  the plugin snapshots those methods and reports tampering (repo2rlenv ``test_driver.py:14-37``);
- a count can be met by a different test set: ``expected_ids`` checks the exact passed node ids (repo2rlenv
  ``grade.py:187-188``), and ``collect_pytest_ids`` writes that manifest on the oracle.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from crucible.verify.runtime.plan import Verdict

PLUGIN = '''"""Crucible verifier pytest plugin: per-test outcomes to a grader-owned file descriptor."""
import json, os, unittest
import pytest

_FD = int(os.environ.pop("CRUCIBLE_VERIFY_FD", "-1"))  # runs in the candidate's pytest, without crucible; declared in crucible.core.env
_seen = {"collected": 0}
_ASSERTS = {n: getattr(unittest.TestCase, n) for n in dir(unittest.TestCase) if n.startswith(("assert", "fail"))}
_CODES = {n: getattr(f, "__code__", None) for n, f in _ASSERTS.items()}


def _emit(rec):
    if _FD >= 0:
        os.write(_FD, (json.dumps(rec) + "\\n").encode())


def pytest_collection_modifyitems(session, config, items):
    _seen["collected"] = len(items)
    _emit({"event": "collected", "n": len(items), "ids": [i.nodeid for i in items]})


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    # the outcome from the call itself, before any other plugin can touch the report
    exc = call.excinfo
    if exc is None:
        outcome = "passed"
    elif exc.errisinstance(pytest.skip.Exception):
        outcome = "skipped"
    else:
        outcome = "failed"
    xfail = exc is not None and hasattr(pytest, "xfail") and exc.errisinstance(pytest.xfail.Exception)
    yield
    if call.when == "call" or outcome != "passed":
        _emit({"event": "test", "nodeid": item.nodeid, "when": call.when, "outcome": outcome, "wasxfail": bool(xfail)})


def pytest_sessionfinish(session, exitstatus):
    changed = sorted(n for n, f in _ASSERTS.items() if getattr(unittest.TestCase, n, None) is not f
                     or getattr(getattr(unittest.TestCase, n, None), "__code__", None) is not _CODES[n])
    if changed:
        _emit({"event": "tamper", "what": changed[:20]})
    _emit({"event": "finish", "exitstatus": int(exitstatus), "collected": _seen["collected"]})
'''


def _results_fd(work: Path, name: str) -> tuple[int, Path]:
    d = Path(work) / "_results"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    p = d / f"{name}.jsonl"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o600)
    return fd, p


def install_plugin(trusted, where: Path) -> Path:
    """Write the plugin into a root-owned directory the candidate can read but not change."""
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    (where / "crucible_verify_pytest.py").write_text(PLUGIN)
    trusted.seal(where)
    return where


def verdict_from_records(records: list[dict], *, expected: int | None, min_count: int, allowed_skips: int, rc: int, log: str,
                         expected_ids: list[str] | None = None) -> Verdict:
    finish = next((r for r in records if r.get("event") == "finish"), None)
    collected = max([r.get("n", 0) for r in records if r.get("event") == "collected"] + [finish.get("collected", 0) if finish else 0])
    tests = [r for r in records if r.get("event") == "test"]
    passed = {r["nodeid"] for r in tests if r["when"] == "call" and r["outcome"] == "passed" and not r.get("wasxfail")}
    failed = sorted({r["nodeid"] for r in tests if r["outcome"] == "failed" or (r["when"] != "call" and r["outcome"] not in ("passed", "skipped"))})
    skipped = sorted({r["nodeid"] for r in tests if r["outcome"] == "skipped"})
    reasons = []
    if finish is None:
        reasons.append("the test session did not finish (process exited early)")
    if tamper := [r for r in records if r.get("event") == "tamper"]:
        reasons.append(f"the candidate changed the test framework's assertions: {tamper[0].get('what')}")
    if expected_ids is not None and passed != set(expected_ids):
        missing, extra = sorted(set(expected_ids) - passed), sorted(passed - set(expected_ids))
        reasons.append(f"the passed tests are not the expected set ({len(missing)} missing, {len(extra)} unexpected)")
    if expected is not None and collected != expected:
        reasons.append(f"collected {collected} tests, expected {expected}")
    if collected < min_count:
        reasons.append(f"collected {collected} tests, fewer than {min_count}")
    if failed:
        reasons.append(f"{len(failed)} failed")
    if len(skipped) > allowed_skips:
        reasons.append(f"{len(skipped)} skipped (allowed {allowed_skips})")
    if finish is not None and len(passed) + len(skipped) < collected:
        reasons.append(f"only {len(passed)} of {collected} tests reported passing")
    return Verdict(not reasons, {"collected": collected, "passed": len(passed), "failed": failed[:40], "skipped": skipped[:20],
                                 "reasons": reasons, "returncode_ignored": rc, "log": "\n".join((log or "").splitlines()[-60:])})


def pytest_suite(ctx, *, name: str, args: list, cwd: Path, pythonpath: list, plugin_dir: Path, candidate_path: tuple = (), python: str = "python3",
                 expected: int | None = None, min_count: int = 1, allowed_skips: int = 0, timeout: float = 3600, env: dict | None = None,
                 expected_ids: list[str] | None = None) -> Verdict:
    t = ctx.trusted
    fd, path = _results_fd(t.work, name)
    try:
        # no plugin from anyone's install loads (a candidate's ``pytest11`` entry point forged every outcome)
        rr = t.python(module="pytest", args=["-q", "-p", "no:cacheprovider", "-p", "crucible_verify_pytest", "-W", "ignore", *args],
                      first=[plugin_dir, *pythonpath], last=list(candidate_path), python=python, cwd=cwd,
                      env={**(env or {}), "CRUCIBLE_VERIFY_FD": str(fd), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, timeout=timeout, candidate=True,
                      pass_fds=(fd,))
    finally:
        os.close(fd)
    records = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"event": "garbage"})
    return verdict_from_records(records, expected=expected, min_count=min_count, allowed_skips=allowed_skips, rc=rr.rc, log=rr.log,
                                expected_ids=expected_ids)


def doctest_suite(ctx, *, name: str, binary: Path, cwd: Path, expected: int | None = None, min_count: int = 1, timeout: float = 3600,
                  env: dict | None = None, extra: tuple = ()) -> Verdict:
    """A grader-built doctest binary; results through doctest's XML reporter into a grader-opened fd."""
    t = ctx.trusted
    fd, path = _results_fd(t.work, name)
    try:
        rr = t.run([binary, "--reporters=xml", f"--out=/proc/self/fd/{fd}", "--no-colors=true", *extra], cwd=cwd, env=env or {},
                   timeout=timeout, candidate=True, pass_fds=(fd,))
    finally:
        os.close(fd)
    text = Path(path).read_text(errors="replace")
    reasons, cases, fails = [], 0, []
    try:
        root = ET.fromstring(text[text.find("<"):]) if "<" in text else None
    except ET.ParseError as ex:
        root = None
        reasons.append(f"unparseable results: {ex}")
    if root is None:
        reasons.append("no results document (the binary exited before writing it)")
    else:
        overall = root.find("OverallResultsTestCases")
        if overall is None:
            reasons.append("results document has no summary (the run did not finish)")
        else:
            cases = int(overall.get("successes", 0)) + int(overall.get("failures", 0))
            if int(overall.get("failures", 0)):
                reasons.append(f"{overall.get('failures')} test cases failed")
        fails = [tc.get("name") for tc in root.iter("TestCase") if tc.find("OverallResultsAsserts") is not None
                 and int(tc.find("OverallResultsAsserts").get("failures", 0)) > 0][:40]
        skipped = int(overall.get("skipped", 0)) if overall is not None else 0
        if skipped:
            reasons.append(f"{skipped} test cases skipped")
    if expected is not None and cases != expected:
        reasons.append(f"ran {cases} test cases, expected {expected}")
    if cases < min_count:
        reasons.append(f"ran {cases} test cases, fewer than {min_count}")
    return Verdict(not reasons, {"cases": cases, "failed": fails, "reasons": reasons, "returncode_ignored": rr.rc,
                                 "log": "\n".join(rr.log.splitlines()[-40:])})


def count_pytest(ctx, *, args: list, cwd: Path, pythonpath: list, plugin_dir: Path, python: str = "python3", timeout: float = 600) -> int:
    """The expected-count manifest: collect the suite (run as the grader, against the oracle) at authoring time."""
    rr = ctx.trusted.run([python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args], cwd=cwd, timeout=timeout,
                         pythonpath=[plugin_dir, *pythonpath])
    m = re.search(r"(\d+) tests? collected", rr.log)
    return int(m.group(1)) if m else len([ln for ln in rr.stdout.splitlines() if "::" in ln])


def collect_pytest_ids(ctx, *, args: list, cwd: Path, pythonpath: list, plugin_dir: Path, python: str = "python3", timeout: float = 600) -> list[str]:
    """The exact-ids manifest: the node ids the suite collects (run as the grader, against the oracle)."""
    rr = ctx.trusted.run([python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args], cwd=cwd, timeout=timeout,
                         pythonpath=[plugin_dir, *pythonpath], env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    return sorted(ln.strip() for ln in rr.stdout.splitlines() if "::" in ln)
