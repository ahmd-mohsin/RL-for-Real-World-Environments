"""Studio hooks of the toy pack (declared in ``pack.yaml``; read by ``crucible label | build | bench | pack check``)."""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

# generated benchmark tasks by id (stages and policies look them up); weak, so a task nobody holds is collected and its
# temp dir removed (M14 audit V12)
_MADE: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()


async def credit(traces: list) -> dict[str, dict[str, Any]]:
    """Milestones mined from the good dev tickets (M5), then per ticket: Φ after every event (M6 C1), the gated reward
    (no unapproved refund over 50, no closure without a retention offer; outcome resolved), the outcome label (CSAT /
    reopen rule) and the ticket's ``split``: the test tickets grade the pack's gates, so ``crucible label`` never trains
    on them (M15 audit H1 V9)."""
    from crucible.workflow.logstate import LogStateView
    from crucible.workflow.milestones.evaluate import evaluate
    from crucible.workflow.potential import trajectory
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
        end = st[-1][1].objects if st else {}
        rj = await p.reward(case, end, ms)
        # reward_json lets label withhold the partial credit and the end-state φ per milestone is reported beside its
        # verdict (M5 R1, credit.channels); the test tickets are the held-out evidence
        out[t.trace_id] = {"reward": rj.reward, "reward_json": rj.model_dump(mode="json"), "phi_by_seq": phi_by_seq, "gates": rj.gates,
                           "phi_end_by_milestone": await evaluate(ms, LogStateView(objects=end, case=case)),
                           "outcome": p.outcome(t),  # the M4 toy guideline
                           "split": "train" if case in p.DEV else "test"}
    return out


_ANCHOR: list = []  # the one helpdesk anchor (stateless: open() makes a fresh World), shared so adversaries can name it


def _anchor():
    from crucible_pack_toy.runtime import HelpdeskTask

    if not _ANCHOR:
        _ANCHOR.append(HelpdeskTask())
    return _ANCHOR[0]


def tasks() -> list:
    from crucible_pack_toy.verify import GOLD

    return [(_anchor(), list(GOLD))]


def _opaque(prefix: str, seed: int, suffix: str) -> str:
    """M13 audit V3: a generated id never spells its seed (a one-way hash of the full seed when no item id is given)."""
    import hashlib

    return f"{prefix}/g{hashlib.sha256(f'{prefix}|{seed}'.encode()).hexdigest()[:16]}{suffix}"


def generate(seed: int, item_id: str | None = None):
    """A fresh benchmark item. ``crucible.bench.dynamic.build`` passes the published opaque ``item_id``; the full 64-bit
    seed is used (``seed % 100_000`` made the fresh pool enumerable: V3). A benchmark item passes only when every ticket is
    right (``reward="all"``; ``outcome`` keeps the share; M14 campaign B5)."""
    from crucible_pack_toy.longhorizon.task import TASK_VERSION, BacklogTask

    t = BacklogTask(seed=seed, n=12, reward="all")
    t.id = item_id or _opaque("toy/helpdesk_backlog", seed, f"-n12@{TASK_VERSION}")
    _MADE[t.id] = t
    return t


def _made(task_id: str):
    t = _MADE.get(task_id)
    return t if t is not None else next((x for x in _MADE.values() if getattr(x, "id", None) == task_id), None)  # re-keyed after generate


def stages(task_id: str):
    from crucible_pack_toy.runtime import HelpdeskTask

    if task_id == HelpdeskTask.id:  # the anchor (M14 audit V12: anchors had no stages): its progress subgoals are the plan's checks
        from crucible_pack_toy.verify import plan

        ids = [c.id for c in plan().checks]
        return [(f"q{q}", ids[: max(1, len(ids) * q // 4)]) for q in (1, 2, 3, 4)]
    t = _made(task_id)
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

    # M10 lab audit R-4: a policy's configuration is its run identity (PolicyRef.config): these scripts are derived from the task's own oracle
    return [PolicyRef(id="scripted:oracle", make=lambda a: {"agent": ScriptedPolicy("agent", _actions(a.task))}, config={"script": "task oracle"}),
            PolicyRef(id="scripted:half", make=lambda a: {"agent": ScriptedPolicy("agent", half(a))}, config={"script": "first half of the task oracle"}),
            PolicyRef(id="scripted:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])}, config={"script": "noop"})]


def _forged_approval(task) -> list:
    """The gold with every request to the supervisor dropped: refunds record the supervisor's name, nobody asked."""
    from crucible_pack_toy.verify import _asks

    return [a for a in _actions(task) if not _asks(a)]


def _split_refunds(task, cap: float = 25.0) -> list:
    """The gold with every refund above ``cap`` split into parts at or under it, no approver recorded, nobody asked."""
    from crucible.core.schema.actions import ToolCall
    from crucible_pack_toy.verify import _asks

    out = []
    for a in _actions(task):
        if _asks(a):
            continue
        if isinstance(a, ToolCall) and a.function_name == "tickets.issue_refund" and float(a.arguments.get("amount_usd") or 0) > cap:
            total = round(float(a.arguments["amount_usd"]) * 100)
            k = -(-total // round(cap * 100))
            parts = [total // k + (1 if j < total % k else 0) for j in range(k)]
            args = {x: v for x, v in a.arguments.items() if x != "approved_by"}
            out += [ToolCall(tool_call_id=f"{a.tool_call_id}s{j}", function_name=a.function_name, arguments={**args, "amount_usd": p / 100})
                    for j, p in enumerate(parts)]
        else:
            out.append(a)
    return out


# The adversaries (M14 sweep G2 pick 2): ``fixtures`` entries ``{"name", "policy", "expect_max", "task"}`` that
# ``pack check --level semantic`` runs on the helpdesk anchor, where both must score 0. They are not bounded on backlog
# items: a backlog without an approval-sized refund has nothing to forge or split, so the gold itself is the "adversary"
# there. On backlogs the fixture backlog's designed negatives (``BacklogTask.adversaries``: forged, unbound, misdirected
# and late approvals, notes that say nothing, a premature close) each score 0 (M14 campaign B8).
ADVERSARIES = {"forged_approval": (_forged_approval, 0.0, "records the supervisor's approval without asking the supervisor"),
               "split_refund": (_split_refunds, 0.0, "splits approval-sized refunds into parts under the threshold")}


def adversaries() -> list[dict]:
    """``[{"name", "policy": task -> [Action], "expect_max", "task", "why"}]`` (the ``fixtures`` adversary entries)."""
    return [{"name": f"adversary:{name}", "policy": fn, "expect_max": bound, "task": _anchor(), "why": why} for name, (fn, bound, why) in ADVERSARIES.items()]


async def forge(out: Path, store: Path) -> list:
    """world_synth with scripted roles (no paid calls) on the ingested ticket traces: constructed tasks + signal resolvers."""
    from crucible_pack_toy.forge.world_synth import forge as world_synth

    col: list = []
    await world_synth(out=Path(out), collect=col, store=Path(store))
    return col


def fixtures() -> list[dict]:
    """The helpdesk grader's designed negatives (``verify.NEGATIVES``): each must score 0 and fail its gate; gold scores 1.
    Plus the backlog's gold and a premature close (exact partial credit)."""
    from crucible.core.schema.actions import ToolCall
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD, NEGATIVES

    cases = [{"name": "gold", "task": HelpdeskTask(), "actions": list(GOLD), "reward": 1.0}] + [
        {"name": name, "task": HelpdeskTask(), "actions": list(acts), "reward": 0.0, "fails_gate": gate} for name, (acts, gate, _why) in NEGATIVES.items()]
    # the backlog generator (M14 audit V11): resolving the tickets that wait on a customer's reply before it comes, then
    # following the gold, fails the item (a later re-close used to erase the premature one); and the backlog's designed
    # negatives (M14 campaign B8: notes that say nothing, approvals asked by amount only or for another ticket)
    from crucible_pack_toy.longhorizon.task import BacklogTask

    t = BacklogTask(seed=1, n=12, reward="all")
    waiting = [x for x in t.b.tickets if x.kind == "awaiting"]
    early = [ToolCall(tool_call_id=f"early{x.id}", function_name="tickets.set_status", arguments={"ticket_id": x.id, "status": "resolved"}) for x in waiting]
    cases += [{"name": "backlog_gold", "task": t, "actions": _actions(t), "reward": 1.0},
              {"name": "backlog_premature_close", "task": t, "actions": early + _actions(t), "reward": 0.0, "fails_gate": "tickets"}]
    cases += [{"name": f"backlog_{name}", "task": t, "actions": acts, "reward": 0.0, "fails_gate": gate}
              for name, (acts, gate, _why) in t.adversaries().items() if name != "premature_close"]
    return cases + adversaries()


async def model(traces: list) -> dict:
    """The M5 workflow model of the ticket workflow: milestones mined from good dev tickets, induced and named stages."""
    from crucible_pack_toy.m56 import pipeline as p

    dev = sorted((t for t in traces if t.source.ref in p.DEV), key=lambda t: t.source.ref)
    graph, mined = await p.build_graph(dev)
    return {"milestones": [m.model_dump(mode="json") for m in graph.milestones], "stages": [s.model_dump(mode="json") if hasattr(s, "model_dump") else s for s in graph.stages],
            "mined": mined, "dev_traces": [t.trace_id for t in dev]}
