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


def verbosity_group_fn(sigma: float, *, l0_of: Callable[[str], float | None] | None = None, pass_threshold: float = 1.0,
                       max_penalty: float = 0.2, exponent: float = 1.5, anchor_quantile: float = 0.3, min_pass_rate: float = 0.5) -> Callable:
    """Over-length successes lose a bounded amount (PORT of Xiaomi's verl ``length_penalty.py:232-329``, chosen in the
    M5-M7 sweep). Only passing members are penalised; the anchor is a low quantile of the passing lengths (or ``l0_of``);
    a member over ``sigma`` x the anchor loses ``max_penalty`` x t^``exponent``, t the excess saturating at 2 x the
    budget; groups whose pass rate is at most ``min_pass_rate`` are left alone. ``max_penalty`` stays below 1 - κ, so a
    success always out-scores every failure (M6 audit: the old rule set an over-budget success to 0, below a failure's
    κΦ = 0.18)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 <= max_penalty < 0.7:
        raise ValueError("max_penalty must stay below 1 - max κ (0.7)")

    def fn(gid: str, members: list[Any]) -> list[Any]:
        scored = [m for m in members if m.reward is not None]
        passed = [m for m in scored if m.reward >= pass_threshold]
        if not passed or len(passed) / len(scored) <= min_pass_rate:
            return members
        l0 = l0_of(scored[0].task) if l0_of is not None else None
        if l0 is None:
            lens = sorted(_length(m) for m in passed)
            l0 = lens[min(len(lens) - 1, int(anchor_quantile * (len(lens) - 1) + 0.5))]
        budget = sigma * max(l0, 1)
        for m in passed:
            excess = _length(m) / budget - 1.0
            if excess > 0:
                pen = max_penalty * min(excess, 1.0) ** exponent
                m.reward_components["verbosity/over_budget"] = float(_length(m) / max(l0, 1))
                m.reward_components["verbosity/penalty"] = pen
                m.reward = float(m.reward) - pen
        return members

    return fn
