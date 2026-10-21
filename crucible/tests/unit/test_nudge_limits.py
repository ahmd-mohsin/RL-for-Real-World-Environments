"""M2 R-5 (audit 2026-10-15): once the nudge allowance is spent, a cut-off reply's tool calls are still never run, and a
stall is not a voluntary finish. The third cut-off reply's call reached the World and reset the allowance, and three blank
replies ended ``agent_done``, untruncated (probes ss05, pv05, v1_ss3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
from crucible.core.schema.actions import Done, Observation
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

SPEC = {"world": "test/harness", "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}},
                                           "params": {"shell": True, "allow_unsafe_local_shell": True}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]}


class Scripted:
    """Replies with ``finish`` each time; ``write`` adds a write_file call (cut mid-reply when finish is length)."""

    model = "fake"

    def __init__(self, finish: str, *, write: bool = False, text: str = ""):
        self.finish, self.write, self.text, self.n = finish, write, text, 0

    async def complete(self, messages, *, tools=None, **kw):
        self.n += 1
        calls = []
        if self.write:
            name = next(t["function"]["name"] for t in tools if t["function"]["name"].endswith("write_file"))
            calls = [{"id": f"call_{self.n}", "name": name, "arguments": {"path": f"cut{self.n}.txt", "content": "partial"}}]
        return LLMResponse(text=self.text, tool_calls=calls, usage={"prompt_tokens": 10, "completion_tokens": 1200}, raw_finish=self.finish)


@pytest.mark.parametrize(("llm", "stop", "truncated"), [
    (Scripted("length", write=True), "max_output_tokens", True),  # was: cut3.txt written, agent_done
    (Scripted("length"), "max_output_tokens", True),  # blank, reasoning spent the budget: was agent_done, untruncated
    (Scripted("length", text="I will now write the fi"), "max_output_tokens", True),
    (Scripted("stop"), "format_error", False),  # blank with a clean finish: a format failure, terminal
])
async def test_a_spent_allowance_ends_labelled_and_runs_nothing(tmp_path: Path, llm, stop, truncated):
    w = World(WorldSpec.model_validate(SPEC), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    w.instructions = {"agent": "Write files."}
    await w.start()
    try:
        pol = LLMToolPolicy("agent", llm)
        res = await run_episode(w, {"agent": pol}, limits=EpisodeLimits(max_steps=6))
    finally:
        await w.close()
    assert not [e for e in res.trace.events if e.type == "tool_call"]  # no cut-off call ever reaches the World
    assert (llm.n, res.stop_condition, res.truncated, res.masked) == (3, stop, truncated, False)
    calls = res.model_calls["agent"]
    assert [c.get("nudged") for c in calls] == ["cut_off" if llm.finish == "length" and (llm.write or llm.text) else "blank"] * 2 + [None]
    assert calls[-1]["actions"] == ["done"]


async def test_a_single_turn_answer_cut_off_is_still_the_answer():
    pol = LLMToolPolicy("agent", Scripted("length", text="The answer is 4"), max_nudges=0)  # type: ignore[arg-type]
    done = await pol.act(Observation(actor="agent", step=0, instruction="2+2?"))
    assert isinstance(done, Done) and done.reason == "The answer is 4" and pol.stop == "max_output_tokens"
