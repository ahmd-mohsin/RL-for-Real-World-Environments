"""Runtime-independent grades and dense channels by deterministic replay (research m11-m12/05 §2.4).

No foreign runtime carries the per-step World state Crucible's dense channel needs. So every runtime's transcript is
normalized to canonical actions (``canonical``), re-issued into a fresh World with a scripted policy, and graded and
measured there: ``reward.json`` through the single producer (``verify.runtime.plan.write_reward_files``) and
``steps.jsonl`` from online progress (``verify.progress``) over the task's subgoals — or, for a task graded by an M7
plan over the verifier view, over that plan's own checks (``PlanSubgoals``). ``call_id``s are derived from
(episode key, model-call ordinal), so two runtimes' transcripts of the same episode produce identical rows.

Transcript parsers: ``from_world_trace`` (Crucible), ``from_atif`` (Harbor), ``from_responses`` (NeMo-Gym Responses
``function_call`` items), ``from_openenv_steps`` (OpenEnv ``CallToolAction``s), ``from_gateway_jsonl`` (agentenv).
"""

from __future__ import annotations

import inspect
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.core.schema.reward import StepRow
from crucible.interop.canonical import call_id, canonical_args, canonical_tool
from crucible.verify.progress import ProgressTracker, Subgoal


@dataclass
class Action:
    tool: str | None  # canonical "<app>.<tool>"; None for a message
    args: dict
    to: str | None = None  # message recipient
    text: str | None = None


class PlanSubgoals:
    """Every check of an M7 plan graded over the verifier view, as a subgoal (evaluated each step; small plans)."""

    def __init__(self, plan: Any, *, workdir: Path):
        self.plan, self.workdir = plan, Path(workdir)
        self.subgoals = [Subgoal(c.id, frozenset()) for c in plan.checks]

    async def evaluate(self, world: Any, ids: Iterable[str]) -> dict[str, bool]:
        from crucible.verify.runtime.plan import Context, Verdict
        from crucible.verify.runtime.trusted import Trusted

        want = set(ids)
        ctx = Context(Trusted(self.workdir), params={"view": world.verifier_view(), "events": list(world.log)})
        out = {}
        for c in self.plan.checks:
            if c.id not in want:
                continue
            try:
                v = c.fn(ctx)
                if inspect.isawaitable(v):
                    v = await v
                out[c.id] = bool(v.passed if isinstance(v, Verdict) else (v.get("pass") if isinstance(v, dict) else v))
            except Exception:  # noqa: BLE001 -- a check that cannot be evaluated on a partial state counts as not met
                out[c.id] = False
        return out


class _AllKeys(ProgressTracker):
    """PlanSubgoals have no entity keys: re-evaluate every subgoal after every step."""

    async def on_step(self, world, step, actor, action, result):
        self.full_every = 1
        return await super().on_step(world, step, actor, action, result)


async def replay(task: Any, actions: Sequence[Action], out_dir: Path, *, episode_key: str, harness: str = "replay@1", plan: Any = None,
                 max_steps: int = 500) -> dict:
    """Re-issue ``actions`` in a fresh World of ``task`` (M10 RunTask), write the canonical reward files and steps.jsonl."""
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
    from crucible.core.schema.actions import Message, ToolCall
    from crucible.credit.steps import write as write_steps
    from crucible.verify.runtime.plan import write_reward_files

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="replay-"))
    world = await task.open(d / "work")
    try:
        script = [ToolCall(tool_call_id=f"r{i}", function_name=a.tool, arguments=a.args) if a.tool else Message(to=a.to or "", text=a.text or "")
                  for i, a in enumerate(actions)]
        sg = getattr(task, "subgoals", None) or PlanSubgoals(plan or task.plan_for_progress(), workdir=d / "p")
        tracker = (ProgressTracker if hasattr(task, "subgoals") else _AllKeys)(sg, out=d / "progress.jsonl", full_every=1)
        ep = await run_episode(world, {"agent": ScriptedPolicy("agent", script)}, limits=EpisodeLimits(max_steps=max_steps, max_tool_errors=None),
                               reset=False, observers=[tracker])
        rep = await task.grade(world, d / "grade")
        rj = rep.reward_json()
        if ep.masked:
            rj["infra"] = {"status": "masked", "failure_kind": next((f.get("kind") for f in ep.failures if f.get("kind")), "agent_run_error"), "reason": "episode masked"}
        write_reward_files(out_dir, rj, rep.to_dict())
        from crucible.verify.progress import read_rows

        rows = [StepRow(step=r["step"], call_id=call_id(episode_key, i), harness=harness, actor="agent", phi=r["phi"], delta_phi=r["delta_phi"],
                        milestones_new=r["gained"], milestones_lost=r["lost"])
                for i, r in enumerate(x for x in read_rows(d / "progress.jsonl") if x["step"] >= 0)]
        write_steps(rows, out_dir / "steps.jsonl")
        return {"reward": None if rj.get("infra", {}).get("status") == "masked" else rep.reward, "steps": len(rows),
                "fingerprint": await _fingerprint(world)}
    finally:
        await world.close()


async def _fingerprint(world: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(await world.verifier_view().fingerprints(), sort_keys=True).encode()).hexdigest()[:24]


# ---- transcript parsers -------------------------------------------------------------------------------------------
def from_world_trace(trace: Any, actor: str = "agent") -> list[Action]:
    out = []
    for e in trace.events:
        if e.actor != actor:
            continue
        if e.type == "tool_call":
            out.append(Action(canonical_tool(e.data["function"]), canonical_args(e.data.get("arguments"))))
        elif e.type == "message":
            out.append(Action(None, {}, to=e.data.get("to"), text=e.text))
    return out


def from_atif(traj: Any, *, known: Iterable[str] = ()) -> list[Action]:
    out = []
    for s in traj.steps:
        if s.source != "agent":
            continue
        for tc in s.tool_calls or []:
            out.append(Action(canonical_tool(tc.function_name, known=known), canonical_args(tc.arguments)))
        to = (s.extra or {}).get("to")
        if to and not s.tool_calls:
            out.append(Action(None, {}, to=to, text=s.message if isinstance(s.message, str) else ""))
    return out


def from_responses(output_items: Sequence[dict], *, known: Iterable[str] = ()) -> list[Action]:
    """NeMo-Gym / OpenAI Responses ``output`` items: ``{"type": "function_call", "name", "arguments"}``."""
    out = []
    for it in output_items:
        if it.get("type") == "function_call":
            name = it.get("name", "")
            if name == "send_message":
                a = canonical_args(it.get("arguments"))
                out.append(Action(None, {}, to=a.get("to"), text=a.get("text")))
            else:
                out.append(Action(canonical_tool(name, known=known), canonical_args(it.get("arguments"))))
    return out


def from_openenv_steps(actions: Sequence[dict], *, known: Iterable[str] = ()) -> list[Action]:
    """OpenEnv MCP ``CallToolAction``s: ``{"tool_name", "arguments"}``."""
    return [Action(canonical_tool(a["tool_name"], known=known), canonical_args(a.get("arguments"))) for a in actions]


def from_gateway_jsonl(events: Sequence[dict], *, known: Iterable[str] = ()) -> list[Action]:
    """agentenv gateway ``GET /trajectory`` events: ``{"event_type": "tool_call", "tool_call": {"function_name", "arguments"}}``
    (``agent_env/env/gateway/gateway.py`` ``_log_event``); flat ``tool_name``/``arguments`` also accepted."""
    out = []
    for e in events:
        if e.get("event_type") != "tool_call":
            continue
        tc = e.get("tool_call") or e
        args = canonical_args(tc.get("arguments"))
        args.pop("__actor", None)  # agentenv world_backend.ACTOR_ARG: the World router's signed actor id rides inside the arguments: never a tool argument (C3)
        out.append(Action(canonical_tool(tc.get("function_name") or tc.get("tool_name") or "", known=known), args))
    return out
