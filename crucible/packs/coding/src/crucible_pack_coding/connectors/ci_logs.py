"""CI test reports and logs -> traces of ``check`` events (coding pack ``ci_logs`` connector, M3; DESIGN §5.2
"CI result per commit").

One report or log file is one CI run. The format is detected from the content:

  JUnit XML (pytest ``--junitxml``, ``ctest --output-junit``, gtest ``--gtest_output=xml``, Gradle, Azure / GitHub
  test publishers), CTest, GoogleTest, pytest (verbose lines and the short summary), Ninja / Make build failures,
  and the migration tasks' ``grade_report.json`` (one check per gate, failed tests parsed from the gate's pytest list).

PORT: the CTest, GoogleTest and pytest line patterns come from SWE-smith ``swesmith/profiles/cpp.py`` (MIT). Changed:
no synthetic test names when only a summary is printed (the counts go into ``ci_run`` instead), and ``data.message``
keeps the first failure line.

A run becomes a ``ci_run`` event (counts, ``failure_kind``) plus one ``check`` event per test or build target.
``failure_kind`` follows P16: ``infra`` when the log shows the runner failed rather than the code (disk full, OOM
kill, lost network, no GPU, runner timeout); such a run's outcome is ``unknown`` (masked, never scored 0). The
outcome is otherwise ``good`` iff nothing failed.

``flaky(traces)``: tests whose status differs between runs of the same commit. SWE-smith ``harness/repair.py``
drops such tests from FAIL_TO_PASS / PASS_TO_PASS; here they are listed so M4 can discount them.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

FAMILY = "coding/ci_run"
PASS, FAIL, SKIP, ERROR = "pass", "fail", "skip", "error"
INFRA = re.compile(
    r"No space left on device|Killed\s+signal|oom-kill|Out of memory: Killed process|Cannot allocate memory"
    r"|Could not resolve host|Connection (?:reset|refused|timed out)|Temporary failure in name resolution"
    r"|no CUDA-capable device is detected|CUDA driver version is insufficient|NVIDIA-SMI has failed"
    r"|The job running on runner .* has exceeded the maximum execution time|##\[error\]The operation was canceled"
    r"|rate limit exceeded|503 Service Unavailable|docker: Error response from daemon",
    re.IGNORECASE,
)


def parse_junit(text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(text)
    out = []
    for case in root.iter("testcase"):
        cls, name = case.get("classname") or "", case.get("name") or ""
        status, message = PASS, None
        for tag, st in (("failure", FAIL), ("error", ERROR), ("skipped", SKIP)):
            el = case.find(tag)
            if el is not None:
                status, message = st, (el.get("message") or (el.text or "")).strip().splitlines()[:1]
                message = message[0] if message else None
                break
        t = case.get("time")
        out.append({"name": f"{cls}::{name}" if cls else name, "status": status, "message": message,
                    "duration_s": float(t) if t not in (None, "") else None})
    return out


CTEST = re.compile(r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\s*(\*{0,3}\w[\w ]*?)\s+([\d.]+)\s+sec", re.MULTILINE)
CTEST_FAILED = re.compile(r"The following tests FAILED:\n((?:\s+\d+\s+-\s+\S+.*\n?)+)")


def parse_ctest(text: str) -> list[dict[str, Any]]:
    res: dict[str, dict[str, Any]] = {}
    for m in CTEST.finditer(text):
        word = m.group(2).strip("* ").lower()
        st = PASS if word == "passed" else (SKIP if "not run" in word or "disabled" in word or "skipped" in word else FAIL)
        res[m.group(1)] = {"name": m.group(1), "status": st, "message": None if st == PASS else m.group(2).strip("* "), "duration_s": float(m.group(3))}
    sec = CTEST_FAILED.search(text)
    for line in (sec.group(1).splitlines() if sec else []):
        m = re.search(r"\d+\s+-\s+(\S+)(?:\s+\((.*)\))?", line)
        if m:
            res.setdefault(m.group(1), {"name": m.group(1), "status": FAIL, "message": m.group(2), "duration_s": None})
            if res[m.group(1)]["status"] == PASS:
                res[m.group(1)]["status"] = FAIL
    return list(res.values())


GTEST = re.compile(r"^\[\s*(OK|FAILED|SKIPPED|DISABLED)\s*\]\s+([\w/.:<>,]+)(?:\s+\((\d+)\s*ms\))?\s*$", re.MULTILINE)


def parse_gtest(text: str) -> list[dict[str, Any]]:
    res: dict[str, dict[str, Any]] = {}
    for m in GTEST.finditer(text):
        name = m.group(2)
        if name.isdigit():  # summary lines ("[  FAILED  ] 2 tests, listed below:")
            continue
        st = {"OK": PASS, "FAILED": FAIL}.get(m.group(1), SKIP)
        if name in res and m.group(3) is None:  # the summary's re-listing of a failure
            continue
        res[name] = {"name": name, "status": st, "message": None, "duration_s": int(m.group(3)) / 1000 if m.group(3) else None}
    return list(res.values())


PYTEST_LINE = re.compile(r"^(\S+::\S+?)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b", re.MULTILINE)
PYTEST_SHORT = re.compile(r"^(FAILED|ERROR)\s+(\S+::\S+?)(?:\s+-\s+(.*))?$", re.MULTILINE)
PYTEST_SUMMARY = re.compile(r"(\d+) (failed|passed|skipped|errors?|xfailed|xpassed|deselected)")


def parse_pytest(text: str) -> list[dict[str, Any]]:
    res: dict[str, dict[str, Any]] = {}
    for m in PYTEST_LINE.finditer(text):
        st = {"PASSED": PASS, "XFAIL": PASS, "FAILED": FAIL, "XPASS": FAIL, "ERROR": ERROR}.get(m.group(2), SKIP)
        res[m.group(1)] = {"name": m.group(1), "status": st, "message": None, "duration_s": None}
    for m in PYTEST_SHORT.finditer(text):
        st = FAIL if m.group(1) == "FAILED" else ERROR
        res[m.group(2)] = {**res.get(m.group(2), {"name": m.group(2), "duration_s": None}), "status": st, "message": m.group(3)}
    return list(res.values())


BUILD_FAILED = re.compile(r"^FAILED: (\S+)", re.MULTILINE)  # ninja
MAKE_FAILED = re.compile(r"^(?:g?make(?:\[\d+\])?): \*\*\* \[(?:\S+: )?(\S+)\] Error \d+", re.MULTILINE)
COMPILE_ERROR = re.compile(r"^(\S+?)[:(](\d+)\)?(?::\d+)?:? (?:fatal )?error(?: #\w+-?\w*)?: ?(.*)$", re.MULTILINE)  # gcc/clang and nvcc


def parse_build(text: str) -> list[dict[str, Any]]:
    """One check per failed build target; its message is the first compiler error printed under that target."""
    marks = sorted([(m.start(), m.group(1)) for m in BUILD_FAILED.finditer(text)] + [(m.start(), m.group(1)) for m in MAKE_FAILED.finditer(text)])
    out: dict[str, dict[str, Any]] = {}
    for i, (pos, target) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        err = COMPILE_ERROR.search(text, pos, end) or COMPILE_ERROR.search(text, max(0, pos - 20000), pos)  # make prints errors first
        msg = f"{err.group(1)}:{err.group(2)}: {err.group(3).strip()}" if err else None
        out.setdefault(target, {"name": f"build:{target}", "status": FAIL, "message": msg, "duration_s": None})
    return list(out.values())


def summary_counts(text: str) -> dict[str, int]:
    """Counts printed by the runner itself (pytest "6 failed, 132 passed", ctest "95% tests passed, 2 tests failed out of 40")."""
    out: dict[str, int] = {}
    m = re.search(r"(\d+)% tests passed, (\d+) tests? failed out of (\d+)", text)
    if m:
        return {"failed": int(m.group(2)), "passed": int(m.group(3)) - int(m.group(2))}
    for line in text.splitlines():
        if re.search(r"\d+ (passed|failed)", line) and (" in " in line or line.startswith("=")):
            out = {}
            for n, k in PYTEST_SUMMARY.findall(line):
                out[k.rstrip("s") if k.startswith("error") else k] = int(n)
    return out


def detect(text: str) -> str:
    head = text.lstrip()[:400]
    if head.startswith(("<?xml", "<testsuite")):
        return "junit"
    if re.search(r"Test\s+#\d+:", text) or "tests passed," in text:
        return "ctest"
    if "[ RUN      ]" in text or "[==========]" in text:
        return "gtest"
    if PYTEST_LINE.search(text) or PYTEST_SHORT.search(text) or re.search(r"=+ .*\d+ (passed|failed).* in [\d.]+s", text):
        return "pytest"
    if BUILD_FAILED.search(text) or MAKE_FAILED.search(text):
        return "build"
    return "unknown"


PARSERS = {"junit": parse_junit, "ctest": parse_ctest, "gtest": parse_gtest, "pytest": parse_pytest, "build": parse_build}


def parse(text: str, fmt: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    fmt = fmt or detect(text)
    tests = PARSERS[fmt](text) if fmt in PARSERS else []
    if fmt not in ("build", "junit"):
        tests += [t for t in parse_build(text) if t["name"] not in {x["name"] for x in tests}]
    return fmt, tests


def grade_report_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The migration tasks' hidden-verifier report: one check per gate, then the tests each gate listed as failed."""
    out = []
    for gate, res in (report.get("gates") or {}).items():
        res = res if isinstance(res, dict) else {"pass": bool(res)}
        failed = [parse_pytest(f)[0] if PYTEST_SHORT.search(f) else {"name": f, "status": FAIL, "message": None} for f in res.get("failed") or []]
        out.append({"name": f"gate:{gate}", "status": PASS if res.get("pass") else FAIL, "duration_s": None,
                    "message": "; ".join(res.get("summary") or []) or (f"{len(res.get('violations') or [])} violations" if res.get("violations") else None)})
        out += [{**f, "name": f"{gate}/{f['name']}", "duration_s": None} for f in failed]
    return out


class CILogConnector:
    """scope: {"glob": "**/junit.xml", "commit": optional sha, "family": optional} -- one trace per file.

    ``commit`` (or a ``<sha>`` directory component of 40 hex chars) ties the run to the state it tested, which is
    what M4 needs to grade the step that produced that commit.
    """

    name = "ci_logs"

    def __init__(self, root: Path | None = None, privacy: str = "internal"):
        self.root, self.privacy = Path(root or "."), privacy

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        for p in sorted(self.root.glob(scope["glob"])):
            if p.is_file():
                sha = scope.get("commit") or next((x for x in p.parts if re.fullmatch(r"[0-9a-f]{40}", x)), None)
                rel = p.relative_to(self.root).as_posix()
                yield SourceRef(connector=self.name, key=rel, meta={"path": str(p), "commit": sha, "family": scope.get("family", FAMILY)})

    def fetch(self, ref: SourceRef) -> RawRecord:
        return RawRecord(ref=ref, payload={"text": Path(ref.meta["path"]).read_text(errors="replace")}, privacy=self.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        text, path = raw.payload["text"], raw.ref.meta["path"]
        reward = None
        if path.endswith("grade_report.json"):
            report = json.loads(text)
            fmt, tests, reward = "grade_report", grade_report_checks(report), report.get("reward")
            counted = [t for t in tests if t["name"].startswith("gate:")]  # the run passes or fails by its gates
        else:
            fmt, tests = parse(text)
            counted = tests
        counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
        if counted:
            for t in counted:
                counts[{PASS: "passed", FAIL: "failed", SKIP: "skipped", ERROR: "error"}[t["status"]]] += 1
        else:
            counts.update(summary_counts(text))
        infra = INFRA.search(text)
        failing = counts["failed"] + counts["error"]
        kind = "infra" if infra else ("test" if failing else None)
        commit = raw.ref.meta.get("commit")
        run = {"format": fmt, **counts, "failure_kind": kind, **({"reward": reward} if reward is not None else {}), **({"infra_evidence": infra.group(0)} if infra else {}), **({"commit": commit} if commit else {})}
        events = [TraceEvent(seq=0, kind="check", actor="ci", type="ci_run", text=f"{fmt}: {counts['passed']} passed, {failing} failed",
                             state_ref=f"git:{commit}" if commit else None, data=run)]
        for t in tests:
            events.append(TraceEvent(seq=len(events), kind="check", actor="ci", type="test_case", text=f"{t['name']}: {t['status']}",
                                     reply_to=0, data={k: v for k, v in t.items() if v is not None}))
        if infra:
            outcome = TraceOutcome(label="unknown", label_source=f"infra failure (P16): {infra.group(0)}", details=run)
        elif not tests and not any(counts.values()):
            outcome = TraceOutcome(label="unknown", label_source=f"no test results found ({fmt})", details=run)
        else:
            outcome = TraceOutcome(label="bad" if failing else "good", label_source=f"{failing} failing of {sum(counts.values())}", details=run)
        return Trace(trace_id=f"ci:{raw.ref.key}", family=raw.ref.meta.get("family", FAMILY),
                     source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy),  # type: ignore[arg-type]
                     actors=[TraceActor(id="ci", role="ci", kind="system")], events=events, outcome=outcome)


def flaky(traces: Iterable[Trace]) -> dict[str, list[str]]:
    """commit -> tests that both passed and failed on that commit (infra-failed runs excluded)."""
    seen: dict[str, dict[str, set[str]]] = {}
    for t in traces:
        run = t.events[0].data if t.events else {}
        if not run.get("commit") or run.get("failure_kind") == "infra":
            continue
        for e in t.events[1:]:
            seen.setdefault(run["commit"], {}).setdefault(e.data["name"], set()).add(e.data["status"])
    return {c: sorted(n for n, st in tests.items() if PASS in st and (st & {FAIL, ERROR})) for c, tests in seen.items()
            if any(PASS in st and (st & {FAIL, ERROR}) for st in tests.values())}
