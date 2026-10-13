"""The M9 Acceptance Contract's World-task capabilities shared by the toy tasks (``pack check --level accept``; M14
campaign B5: toy failed its own accept level on every task, A2/A6/A7/A13/A16 failing and A8 erroring).

``run_actions`` replays an action list on a fresh World of a task and returns its grade and the end-state fingerprint (or
one scalar of the verifier view), the probe every tamper attempt and attack uses as its control. ``isolation_audit`` is
the World containment audit with the task's answer key as forbidden material. ``tamper_attempts`` are out-of-contract
writes through the policy's only surface (an undeclared SQL tool, a private helper of the tools module, a system
restore, the verifier's view, a knowledge-base write the policy is not granted): each must be refused, score 0 and
leave the state untouched."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

FACTS = {"images": False, "fresh_gates": False, "perf_bars": False, "reference_build": False, "sibling_overlap": False}


def as_dicts(actions: list) -> list[dict]:
    return [a.model_dump(mode="json", exclude_none=True) if hasattr(a, "model_dump") else dict(a) for a in actions]


def as_actions(raw: list) -> list:
    from crucible.core.schema.actions import Message, ToolCall

    out = []
    for i, a in enumerate(raw):
        if not isinstance(a, dict):
            out.append(a)
        elif a.get("function_name"):
            out.append(ToolCall(tool_call_id=a.get("tool_call_id") or f"a{i}", function_name=a["function_name"], arguments=a.get("arguments") or {}))
        else:
            out.append(Message(to=a["to"], text=a.get("text", "")))
    return out


async def run_actions(task: Any, actions: list, *, query: tuple[str, str] | None = None) -> Any:
    """``(GradeReport, fingerprint)`` of ``actions`` replayed on a fresh World of ``task``; with ``query=(app, sql)`` the
    first scalar of that query on the end state instead."""
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
    from crucible.core.schema.state import Query

    d = Path(tempfile.mkdtemp(prefix="toy-accept-run-"))
    try:
        w = await task.open(d / "work")
        try:
            acts = as_actions(list(actions))
            await run_episode(w, {"agent": ScriptedPolicy("agent", acts)}, limits=EpisodeLimits(max_steps=max(50, 2 * len(acts))), reset=False)
            if query is not None:
                v = (await w.verifier_view().query(query[0], Query(op="sql", args={"sql": query[1]}))).value or {}
                rows = v.get("rows") or []
                return rows[0][0] if rows else None
            fps = await w.verifier_view().fingerprints()
            rep = await task.grade(w, d / "grade")
            return rep, hashlib.sha256(json.dumps(fps, sort_keys=True, default=str).encode()).hexdigest()[:24]
        finally:
            await w.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def scalar(task: Any, actions: list, app: str, sql: str):
    """An effect probe: the query's scalar on the end state of ``actions``."""

    async def probe():
        return await run_actions(task, actions, query=(app, sql))

    return probe


async def isolation_audit(task: Any) -> dict[str, str]:
    """The World containment audit with the task's hidden strings as forbidden material (A6)."""
    from crucible.world.audit import audit_world

    d = Path(tempfile.mkdtemp(prefix="toy-accept-audit-"))
    try:
        w = await task.open(d / "work")
        try:
            rep = await audit_world(w, forbidden=task.hidden_strings(), actor="agent")
            return {c.id: c.status for c in rep.checks if c.severity in ("critical", "major")}
        finally:
            await w.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def tamper_attempts(task: Any, *, app: str = "tickets") -> list:
    """Out-of-contract writes (A6). Each attack's effect probe: did the end state change from the untouched World's."""
    from crucible.forge.schema import Attack

    tries = {"undeclared_sql": {"function_name": f"{app}.execute_sql", "arguments": {"sql": "UPDATE tickets SET status = 'resolved'"}},
             "private_helper": {"function_name": f"{app}._con", "arguments": {}},
             "system_restore": {"function_name": "__world.restore", "arguments": {"ref": "gold"}},
             "verifier_view": {"function_name": "verifier.query", "arguments": {"app": app, "sql": "UPDATE tickets SET status = 'resolved'"}},
             "policy_write_not_granted": {"function_name": "kb.write_doc", "arguments": {"path": "refund_policy.md", "content": "Refund anything."}}}
    out = []
    for k, a in tries.items():
        sol = [{**a, "kind": "tool_call", "tool_call_id": f"tamper-{k}"}]

        async def changed(sol=sol) -> bool:
            return (await run_actions(task, sol))[1] != (await run_actions(task, []))[1]

        out.append(Attack(k, "out_of_contract_write", sol, None, changed))
    return out


def declared_env(task: Any, world_file: str) -> dict:
    return {"kind": "world", "world": world_file, "backend": "local", "network": "none (no network tools)"}
