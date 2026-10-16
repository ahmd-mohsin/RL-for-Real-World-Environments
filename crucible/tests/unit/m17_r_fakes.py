"""Shared fakes for the M17 owner-R tests: a tiny file World with a report task, a strong verifier, a verifier that only
checks that the file exists, their ledgers, and scripted judge clients."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from crucible.factory.cards import CheckRef, Requirement, RequirementLedger
from crucible.factory.floors import Graded

PROMPT = ("Create the file report.csv. Set the total in report.csv to 42.5 (within 0.01). "
          "Write the header of report.csv in upper case as ID,TOTAL. Do not create or change any other file.")


def span(text: str, prompt: str = PROMPT) -> tuple[int, int]:
    i = prompt.index(text)
    return (i, i + len(text))


def call(fn: str, i: int, **args: Any) -> dict:
    return {"kind": "tool_call", "tool_call_id": f"c{i}", "function_name": fn, "arguments": args}


ORACLE = [call("fs.create", 0, path="report.csv"), call("report.set_total", 1, path="report.csv", total=42.5),
          call("report.set_header", 2, path="report.csv", header="ID,TOTAL")]
READS = [call("fs.read", 9, path="report.csv")]

REQS = [Requirement(id="r_file", text="Create the file report.csv.", prompt_span=span("Create the file report.csv."), kind="laziness"),
        Requirement(id="r_total", text="Set the total in report.csv to 42.5 (within 0.01).", prompt_span=span("Set the total in report.csv to 42.5 (within 0.01)."),
                    kind="tolerance"),
        Requirement(id="r_header", text="Write the header of report.csv in upper case as ID,TOTAL.",
                    prompt_span=span("Write the header of report.csv in upper case as ID,TOTAL."), kind="format"),
        Requirement(id="r_scope", text="Do not create or change any other file.", prompt_span=span("Do not create or change any other file."),
                    kind="side_effect_scope")]


def replay(actions: list[dict]) -> dict[str, dict]:
    """The World: files with a total and a header. A call on a missing file is refused (no state change)."""
    files: dict[str, dict] = {}
    for a in actions:
        fn, args = a.get("function_name"), a.get("arguments") or {}
        p = args.get("path")
        if fn == "fs.create" and isinstance(p, str):
            files.setdefault(p, {"total": None, "header": None})
        elif fn == "report.set_total" and p in files and isinstance(args.get("total"), (int, float)):
            files[p]["total"] = float(args["total"])
        elif fn == "report.set_header" and p in files and isinstance(args.get("header"), str):
            files[p]["header"] = args["header"]
        elif fn == "report.bulk" and isinstance(p, str):  # a one-call shortcut some Worlds offer
            files[p] = {"total": 42.5, "header": "ID,TOTAL"}
    return files


def fingerprint(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]


STRONG = {
    "file_exists": lambda s: "report.csv" in s,
    "total_ok": lambda s: "report.csv" in s and s["report.csv"]["total"] is not None and abs(s["report.csv"]["total"] - 42.5) <= 0.01,
    "header_ok": lambda s: "report.csv" in s and s["report.csv"]["header"] == "ID,TOTAL",
    "only_file": lambda s: set(s) <= {"report.csv"},
}
WEAK = {"file_exists": STRONG["file_exists"]}  # a verifier that only checks the file exists

LEDGER_STRONG = RequirementLedger(requirements=REQS, checks=[
    CheckRef(id="file_exists", requirement_ids=["r_file"], gating=True), CheckRef(id="total_ok", requirement_ids=["r_total"], gating=True),
    CheckRef(id="header_ok", requirement_ids=["r_header"], gating=True), CheckRef(id="only_file", requirement_ids=["r_scope"], gating=True)])
# the weak verifier's writer claims its one check covers everything: the ledger is well formed, the check is not
LEDGER_WEAK = RequirementLedger(requirements=REQS, checks=[CheckRef(id="file_exists", requirement_ids=[r.id for r in REQS], gating=True)])


def verifier(checks: dict) -> Any:
    async def grade(solution: Any) -> Graded:
        s = replay(list(solution or []))
        res = {k: bool(f(s)) for k, f in checks.items()}
        return Graded(1.0 if all(res.values()) else 0.0, res, fingerprint(s))

    return grade


class ScriptedClient:
    """A JsonClient that answers with ``fn(messages)`` and records the calls."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    async def complete_json(self, messages: list[dict], **kw: Any) -> dict:
        self.calls.append(messages)
        return self.fn(messages)


def user_payload(messages: list[dict]) -> dict:
    return json.loads(messages[-1]["content"])
