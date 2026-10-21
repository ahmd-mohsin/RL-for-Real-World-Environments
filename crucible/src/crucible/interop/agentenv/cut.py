"""The pre-grade cut for outside agents (M11 lab audit R-2, P23).

An outside agent (an A2A agent driven by ``A2AHarness``, an agent run by agentenv's own steps before ``crucible_verify``)
reaches the World through its ``AgentProxy`` until the World closes, and nothing stopped it at the end of its episode:
a hung agent that survived Crucible's cancel, or one that ignored ``tasks/cancel``, kept calling tools while its work was
graded (grading takes no step lock) and between a failed grade and the re-grade. ``NativeHarness`` quiesces at episode end
(``world.snapshot(quiesce=True)``); these paths never did.

The cut, in this order, after the agent's final message is read: every policy actor is done in the World
(``World.cut_policy``: a later call is refused as a ``late_call`` and recorded), then ``World.quiesce`` (processes the
policy left in the sandbox, where it has one), then the caller cancels the remote task. The grade runs after it. Policy
processes the quiesce could not stop end the attempt as ``crucible:policy_misbehavior`` (scored, flag
``policy_process_survived``: the policy's own processes resisted the kill, as Harbor's in-place quiesce rules); any other
quiesce failure is raised for the stage boundary to classify. Either way the cancel and the receipt happen first.
``receipt`` writes ``quiesce.json`` next to the grade: ``{actor_done, quiesced, a2a_state, late_calls, gateway}``; a late
call is a non-scoring rollout flag (``policy_late_call``), as Harbor's ``policy_process_killed``: the grade scores the
state at the cut, which the late call could not change. The proxies stay up through the grade, so a late call is seen
and refused rather than lost as a connection error.

Out of scope here: a trigger the gateway fires on its own clock during the grade is environment dynamics, not a policy call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


async def pre_grade_cut(world: Any) -> dict[str, Any]:
    """Cut every policy actor and quiesce the World. A quiesce that fails never skips the rest (the A2A cancel, the
    receipt): it is returned as ``quiesce_error`` (``survived``: the World reported policy processes it could not stop), and
    the caller raises after the cancel."""
    offset = len(world.late_calls)
    actors = await world.cut_policy()
    cut: dict[str, Any] = {"actor_done": actors, "quiesced": False, "quiesce_error": None, "survived": False, "late_call_offset": offset,
                           "a2a_state": None, "gateway": getattr(world.backend, "gateway_isolation", None)}
    try:
        rep = await world.quiesce()
        cut["quiesced"] = rep is not None
    except Exception as e:  # noqa: BLE001 -- recorded, raised by the caller after the cancel and the receipt
        cut["quiesce_error"] = f"{type(e).__name__}: {e}"[:500]
        cut["survived"] = "survived" in str(e)
        cut["exception"] = e
    return cut


def receipt(world: Any, cut: dict[str, Any], path: Path) -> dict[str, Any]:
    """``quiesce.json`` for this cut: the late calls are the ones recorded since the cut (they kept coming during the grade)."""
    doc = {"actor_done": list(cut.get("actor_done") or []), "quiesced": bool(cut.get("quiesced")), "a2a_state": cut.get("a2a_state"),
           "late_calls": list(world.late_calls[int(cut.get("late_call_offset") or 0):]), "gateway": cut.get("gateway")}
    if cut.get("quiesce_error"):
        doc["quiesce_error"] = cut["quiesce_error"]
    if "a2a_cancel_sent" in cut:
        doc["a2a_cancel_sent"] = cut["a2a_cancel_sent"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2, default=str))
    return doc


def survived_flag(cut: dict[str, Any]) -> dict[str, Any]:
    """The rollout flag for policy processes the quiesce could not stop (Harbor's ``quiesce.flag`` shape)."""
    return {"rule": "policy_process_survived", "category": "grader_tampering", "seq": None, "evidence": str(cut.get("quiesce_error") or "")[:600]}


def late_call_flag(doc: dict[str, Any]) -> dict[str, Any] | None:
    """The non-scoring rollout flag for calls after the cut (the shape of ``grader_guard.Findings.flags``), or None."""
    late = doc.get("late_calls") or []
    if not late:
        return None
    return {"rule": "policy_late_call", "category": "grader_tampering", "seq": None,
            "evidence": "; ".join(f"{c.get('actor')}: {c.get('action')}" for c in late[:20])[:600]}
