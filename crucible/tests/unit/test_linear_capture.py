"""Linear token capture (m13 B-3): the policy stores each call's prompt as a delta of the previous call's prompt +
completion (same segment, exact TITO), so captured tokens grow O(T), not O(turns²); the rollout record rebuilt from the
deltas equals the one built from full prompts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import LLMToolPolicy
from crucible.core.schema.actions import Observation, ToolResult
from crucible.core.schema.run import RunRecord
from crucible.runtime.record import build


class GrowingLLM:
    """Returns tool calls with token records whose prompts extend the previous prompt + completion (an exact-TITO engine)."""

    model = "fake"
    policy_version = 0

    def __init__(self):
        self.full: list[int] = list(range(100))  # the first prompt
        self.spend = SimpleNamespace(usd=0.0, uncertain_usd=0.0)

    async def complete(self, messages, **kw):
        comp = [7, 8, 9]
        tok = {"prompt_token_ids": list(self.full), "completion_token_ids": comp, "logprobs": [-0.1] * 3, "policy_version": 0}
        self.full = self.full + comp + [1, 2, 3, 4]  # the next prompt: + this completion + the rendered tool result
        return LLMResponse(text="", tool_calls=[{"id": f"t{len(self.full)}", "name": "tickets__get_ticket", "arguments": {"id": 1}}],
                           usage={"prompt_tokens": 1, "completion_tokens": 3}, model="fake", raw_finish="tool_calls", tokens=tok)


async def _episode(turns: int):
    pol = LLMToolPolicy("agent", GrowingLLM())
    obs = Observation(actor="agent", step=0, instruction="go")
    for i in range(turns):
        a = await pol.act(obs)
        obs = Observation(actor="agent", step=i + 1, tool_results=[ToolResult(tool_call_id=a.tool_call_id, function_name="tickets.get_ticket", ok=True, content="{}")])
    return pol


def test_capture_is_linear_and_rebuilds_exactly():
    pol = asyncio.run(_episode(25))
    calls = pol.model_calls
    stored = sum(len(c["tokens"].get("prompt_token_ids") or c["tokens"].get("prompt_delta") or []) for c in calls)
    full = sum(100 + 7 * i for i in range(25))  # what full-prompt capture stored
    assert calls[0]["tokens"]["prompt_token_ids"] is not None and all(c["tokens"]["prompt_token_ids"] is None for c in calls[1:])
    assert stored == 100 + 4 * 24 and stored * 10 < full  # first prompt + each observation tail
    rec = RunRecord(run_id="r", task="t", world="w", policy="p", started_at="2026-01-01T00:00:00Z", steps=25, outcome="completed", reward=1.0, attempt="r-a1")
    ro = build(rec, calls={"agent": calls})
    expanded = [{**c, "tokens": {**c["tokens"], "prompt_token_ids": list(range(100)) + [7, 8, 9, 1, 2, 3, 4] * i}} for i, c in enumerate(calls)]
    ref = build(rec, calls={"agent": expanded})
    assert [t.prompt_token_ids for t in ro.turns] == [t.prompt_token_ids for t in ref.turns] and ro.valid_for_training()[0]
    assert not any(t.prefix_break for t in ro.turns)
