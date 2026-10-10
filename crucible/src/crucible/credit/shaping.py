"""Reward shaping operators (m13 C-8; article §3.6.3.1-2: Qwen, MiniMax, Kimi; A2-19, A2-21, A2-24, A2-25).

- **Tool-format penalty (A2-25)**: ``format_errors_by_call(trace, calls)`` counts, per model call, the tool calls it
  issued that the World rejected as malformed (``invalid_arguments``, ``unknown_tool``, ``bad_tool_name``,
  ``unknown_app``). The record stores the count on that call (``Turn.format_errors``) and the trainer subtracts
  ``TrainerConfig.tool_format_penalty`` × count from *that call's tokens only* (MiniMax r_process, Qwen).
- **Speed term (A2-21, MiniMax)**: ``speed_term(t, t_base) = 1 / (1 + t / t_base)``, strictly decreasing, 1 at t = 0,
  0.5 at the baseline. ``with_speed(reward_fn, beta, t_base)`` adds ``beta * h`` to *passing* rollouts only, so a fast
  failure earns nothing. The outcome itself already reaches every step's advantage (the sequence advantage is broadcast to
  every token): reward-to-go for a terminal reward.
- **Verbosity budget (A2-19, Kimi)**: ``verbosity_group_fn(sigma)`` is a group stage: a member longer than
  ``sigma * l0`` loses (reward 0), where ``l0`` is the task's reference length (``l0_of(task)``, default: the median
  length of the group's passing members).
- **Incomplete-trajectory penalty (A2-24, Qwen)**: ``runtime.infra.LimitPolicy`` takes ``"penalty:<c>"`` for a limit
  stop (steps, timeout, context window): the reward becomes ``c``.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Callable
from typing import Any

FORMAT_ERRORS = frozenset({"invalid_arguments", "unknown_tool", "bad_tool_name", "unknown_app"})


def format_errors_by_call(trace: Any, calls: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """{model call id: malformed tool calls it issued}, from the World trace's ``tool_result`` events and each model
    call's ``actions`` (``"tool:<tool_call_id>"``)."""
    bad = {str(e.data.get("id")) for e in trace.events if e.type == "tool_result" and e.data.get("error") in FORMAT_ERRORS}
    out: dict[str, int] = {}
    for cs in calls.values():
        for c in cs:
            n = sum(1 for a in c.get("actions") or [] if isinstance(a, str) and a.startswith("tool:") and a[5:] in bad)
            if n:
                out[c["call_id"]] = n
    return out


def speed_term(t: float, t_base: float) -> float:
    if t_base <= 0:
        raise ValueError("t_base must be positive")
    return 1.0 / (1.0 + max(0.0, float(t)) / t_base)


def with_speed(reward_fn: Callable | None, *, beta: float, t_base: float | Callable[[Any], float], pass_threshold: float = 1.0) -> Callable:
    """A ``reward_fn`` adding ``beta * speed_term(active_s, t_base)`` to passing rollouts (``reward_components["speed"]``).
    ``reward_fn=None``: the record's own reward is the base."""

    async def fn(rec: Any) -> float | None:
        base = rec.reward
        if reward_fn is not None:
            base = reward_fn(rec)
            base = await base if asyncio.iscoroutine(base) else base
        if base is None:
            return None
        if base < pass_threshold:
            return float(base)
        tb = t_base(rec) if callable(t_base) else t_base
        h = speed_term(float(rec.timings.get("active_s", 0.0)), tb)
        rec.reward_components["speed"] = h
        return float(base) + beta * h

    return fn


def _length(rec: Any) -> int:
    return sum(len(t.completion_token_ids or []) for t in rec.turns)


def verbosity_group_fn(sigma: float, *, l0_of: Callable[[str], float | None] | None = None, pass_threshold: float = 1.0) -> Callable:
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    def fn(gid: str, members: list[Any]) -> list[Any]:
        scored = [m for m in members if m.reward is not None]
        if not scored:
            return members
        l0 = l0_of(scored[0].task) if l0_of is not None else None
        if l0 is None:
            ref = [_length(m) for m in scored if m.reward >= pass_threshold] or [_length(m) for m in scored]
            l0 = statistics.median(ref)
        for m in scored:
            if _length(m) > sigma * l0:
                m.reward_components["verbosity/over_budget"] = float(_length(m) / max(l0, 1))
                m.reward = 0.0
        return members

    return fn
