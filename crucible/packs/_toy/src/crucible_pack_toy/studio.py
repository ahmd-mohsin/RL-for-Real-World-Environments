"""Studio hooks of the toy pack (declared in ``pack.yaml``; read by ``crucible label | build | bench | pack check``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_MADE: dict[str, Any] = {}  # generated benchmark tasks by id (stages and policies look them up)


async def credit(traces: list) -> dict[str, dict[str, Any]]:
    """Milestones mined from the good dev tickets (M5), then per ticket: Φ after every event (M6 C1), the gated reward
    (no unapproved refund over 50, no closure without a retention offer; outcome resolved) and the outcome label (CSAT /
    reopen rule)."""
    from crucible.credit.operators.potential import trajectory
    from crucible_pack_toy.m56 import pipeline as p

    dev = sorted((t for t in traces if t.source.ref in p.DEV), key=lambda t: t.source.ref)  # mined on dev only; test tickets are held out
    graph, _ = await p.build_graph(dev)
    ms = list(graph.milestones)
    out = {}
    for t in traces:
        st = p.toy_states(t)
        series = await p.series_of(ms, t)
        traj = trajectory(ms, series.values)
        phi_by_seq = {seq: round(s.phi, 6) for seq, s in zip(series.steps, traj)}
        case = t.source.ref
        rj = await p.reward(case, st[-1][1].objects if st else {}, ms)
        out[t.trace_id] = {"reward": rj.reward, "phi_by_seq": phi_by_seq, "gates": rj.gates, "outcome": p.outcome(t)}  # the M4 toy guideline
    return out


def tasks() -> list:
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    return [(HelpdeskTask(), list(GOLD))]


def generate(seed: int):
    from crucible_pack_toy.longhorizon.task import BacklogTask

    t = BacklogTask(seed=seed % 100_000, n=12)
    _MADE[t.id] = t
    return t


def stages(task_id: str):
    t = _MADE.get(task_id)
    if t is None or not hasattr(t, "subgoals"):
        return None
    ids = [g.id for g in t.subgoals.subgoals]
    return [(f"q{q}", ids[: len(ids) * q // 4]) for q in (1, 2, 3, 4)]


def _actions(task) -> list:
    from crucible.core.schema.actions import Message, ToolCall

    if hasattr(task, "oracle_actions"):
        raw = task.oracle_actions()
    elif hasattr(task, "oracle"):
        raw = task.oracle
    else:
        from crucible_pack_toy.verify import GOLD

        return list(GOLD)
    return [ToolCall(tool_call_id=a.get("tool_call_id") or f"o{i}", function_name=a["function_name"], arguments=a.get("arguments") or {})
            if a.get("kind", "tool_call") == "tool_call" else Message(to=a["to"], text=a["text"]) for i, a in enumerate(raw)]


def policies() -> list:
    from crucible.actors.policy import ScriptedPolicy
    from crucible.runtime.harness.native import PolicyRef

    def half(a):
        acts = _actions(a.task)
        return acts[: len(acts) // 2]

    return [PolicyRef(id="scripted:oracle", make=lambda a: {"agent": ScriptedPolicy("agent", _actions(a.task))}),
            PolicyRef(id="scripted:half", make=lambda a: {"agent": ScriptedPolicy("agent", half(a))}),
            PolicyRef(id="scripted:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])})]


async def forge(out: Path, store: Path) -> list:
    """world_synth with scripted roles (no paid calls) on the ingested ticket traces: constructed tasks + signal resolvers."""
    from crucible_pack_toy.forge.world_synth import forge as world_synth

    col: list = []
    await world_synth(out=Path(out), collect=col, store=Path(store))
    return col
