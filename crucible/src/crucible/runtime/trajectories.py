"""Trajectories that join (research m9-m10/02 §1.4, hole H8): one ATIF v1.8 document per policy actor in which
every model call, World action and ``steps.jsonl`` row can be found from the others.

- ``to_atif`` builds on ``interop.harbor.atif.episode_to_atif`` (DEPEND: Harbor's ATIF models) and adds, per agent
  step: ``extra.call_id`` (the model call that produced it), ``llm_call_count`` (1 on a call's first step, 0 on the
  rest), and ``Metrics`` (tokens, cost, token ids and logprobs when the endpoint returned them) on the first step.
  A call that produced no action (a blank or cut-off reply the policy nudged past) becomes its own agent step with
  ``extra.no_action``: before this, such calls had no ATIF step and the three sequences could not be joined.
  ``FinalMetrics`` totals the calls.
- ``align`` sets ``StepRow.atif_step_id`` and ``call_id`` from the M5 steps' trace seqs.
- ``cross_check`` compares the trajectory to an independent capture log (proxy records; OpenEnv's proxy-vs-ATIF
  check): a mismatch is ``crucible:capture_incomplete`` (infra; masks a training rollout, ignored in evaluation).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from harbor.models.trajectories import FinalMetrics, Metrics, Step, Trajectory

from crucible.core.schema.reward import StepRow
from crucible.core.schema.run import Failure
from crucible.core.schema.trace import Trace
from crucible.interop.harbor.atif import episode_to_atif


def _action_key(step: Step) -> str | None:
    if step.source != "agent":
        return None
    if step.tool_calls:
        return f"tool:{step.tool_calls[0].tool_call_id}"
    ex = step.extra or {}
    if ex.get("done"):
        return "done"
    if "to" in ex:
        return "message"
    return None


def _metrics(call: dict[str, Any]) -> Metrics:
    u, tok = call.get("usage") or {}, call.get("tokens") or {}
    return Metrics(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"), cached_tokens=u.get("cached_tokens"),
                   cost_usd=call.get("cost_usd"), prompt_token_ids=tok.get("prompt_token_ids"), completion_token_ids=tok.get("completion_token_ids"),
                   logprobs=tok.get("logprobs"),
                   extra={k: v for k, v in {"policy_version": tok.get("policy_version"), "finish": call.get("finish"), "latency_s": call.get("latency_s"),
                                            "blank_usage": bool(u.get("blank")), "reasoning_tokens": u.get("reasoning_tokens")}.items() if v not in (None, False)} or None)


def to_atif(trace: Trace, actor: str, *, calls: Sequence[dict[str, Any]] = (), instruction: str | None = None, harness: str | None = None,
            model_name: str | None = None, session_id: str | None = None) -> Trajectory:
    base = episode_to_atif(trace, actor, instruction=instruction, model_name=model_name, session_id=session_id)
    expected = [(c, a) for c in calls for a in (c.get("actions") or [])]
    pending_empty = [c for c in calls if not c.get("actions")]
    out: list[Step] = []
    unjoined: list[int] = []
    seen: set[int] = set()
    i = 0

    def emit_empty_before(call: dict[str, Any] | None) -> None:
        # calls with no action that were made before ``call`` (in call order)
        order = {id(c): n for n, c in enumerate(calls)}
        limit = order[id(call)] if call is not None else len(calls)
        while pending_empty and order[id(pending_empty[0])] < limit:
            c = pending_empty.pop(0)
            out.append(Step(step_id=1, source="agent", message="", model_name=c.get("model") or model_name, llm_call_count=1, metrics=_metrics(c),
                            extra={"call_id": c.get("call_id"), "no_action": True, "crucible_world_step": c.get("step")}))

    for s in base.steps:
        key = _action_key(s)
        if key is not None and i < len(expected) and expected[i][1] == key:
            c = expected[i][0]
            i += 1
            emit_empty_before(c)
            first = id(c) not in seen
            seen.add(id(c))
            s.extra = {**(s.extra or {}), "call_id": c.get("call_id"), "crucible_world_step": c.get("step")}
            s.llm_call_count = 1 if first else 0
            if first:
                s.metrics = _metrics(c)
        elif key not in (None, "done") and calls:  # a harness-issued done (submit observed) has no call of its own
            unjoined.append(len(out))  # an action no recorded call produced (scripted step, or a capture gap)
        out.append(s)
    emit_empty_before(None)
    for n, s in enumerate(out, 1):
        s.step_id = n
    tot = lambda k: sum(((c.get("usage") or {}).get(k) or 0) for c in calls)
    fm = FinalMetrics(total_prompt_tokens=tot("prompt_tokens") if calls else None, total_completion_tokens=tot("completion_tokens") if calls else None,
                      total_cached_tokens=tot("cached_tokens") if calls else None, total_cost_usd=round(sum(c.get("cost_usd") or 0.0 for c in calls), 6) if calls else None,
                      total_steps=len(out),
                      extra={"model_calls": len(calls), "unjoined_steps": [out[k].step_id for k in unjoined], "unconsumed_actions": len(expected) - i}
                      if calls else None)
    return Trajectory(session_id=base.session_id, agent=base.agent, steps=out, final_metrics=fm,
                      extra={**(base.extra or {}), **({"harness": harness} if harness else {})})


def step_index(traj: Trajectory) -> dict[int, Step]:
    """trace seq -> ATIF step."""
    return {s.extra["crucible_seq"]: s for s in traj.steps if s.extra and "crucible_seq" in s.extra}


def align(rows: Sequence[StepRow], steps: Sequence[Any], traj: Trajectory) -> list[StepRow]:
    """Set ``atif_step_id`` and ``call_id`` on ``steps.jsonl`` rows: row i belongs to M5 step i, whose trace seqs
    locate its ATIF steps; the last agent step among them is the one the row's credit attaches to."""
    by_seq = step_index(traj)
    by_index = {s.index: s for s in steps}
    out = []
    for r in rows:
        st = by_index.get(r.step)
        hit = [by_seq[q] for q in (st.seqs if st else []) if q in by_seq and by_seq[q].source == "agent"]
        last = hit[-1] if hit else None
        out.append(r.model_copy(update={"atif_step_id": str(last.step_id) if last else None,
                                        "call_id": (last.extra or {}).get("call_id") if last else r.call_id}))
    return out


def cross_check(traj: Trajectory, capture: Sequence[dict[str, Any]], *, tolerance: int = 0) -> Failure | None:
    """Compare the ATIF trajectory's model calls with an independent capture (proxy) log: same number of calls, and
    per call the same completion-token count (within ``tolerance``). ``capture`` rows: ``{completion_tokens}``."""
    mine = [s for s in traj.steps if s.llm_call_count == 1]
    if len(mine) != len(capture):
        return Failure(kind="crucible:capture_incomplete", stage="collect", reason=f"trajectory has {len(mine)} model calls, capture has {len(capture)}")
    for n, (s, c) in enumerate(zip(mine, capture, strict=True), 1):
        a, b = (s.metrics.completion_tokens if s.metrics else None), c.get("completion_tokens")
        if a is not None and b is not None and abs(a - b) > tolerance:
            return Failure(kind="crucible:capture_incomplete", stage="collect", reason=f"call {n}: {a} completion tokens in the trajectory, {b} in the capture")
    return None
