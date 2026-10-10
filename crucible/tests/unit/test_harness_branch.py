"""The native harness (BashLoopPolicy, mini-swe-agent protocol) and C3 branch values on a real local World with a
scripted model: format-error recovery, explicit submission, cut-off replies, and V̂ from K restored continuations."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.actors.harness import SUBMIT, BashLoopPolicy, format_observation
from crucible.actors.llm import LLMResponse
from crucible.actors.policy import EpisodeLimits, run_episode
from crucible.core.schema.state import Query
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.credit.operators.branch_value import advantages, branch_value, correlation, wilson
from crucible.world.world import World


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []
        self.model = "fake"

    async def complete(self, messages, **kw):
        self.prompts.append([dict(m) for m in messages])
        return self.replies.pop(0)


def say(text, finish="stop"):
    return LLMResponse(text=text, raw_finish=finish)


def spec() -> WorldSpec:
    return WorldSpec.model_validate({"world": "test/harness", "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}},
                                                                         "params": {"shell": True, "allow_unsafe_local_shell": True}}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})


@pytest.fixture
async def world(tmp_path: Path):
    w = World(spec(), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    w.instructions = {"agent": "Create hello.txt containing the word hi."}
    await w.start()
    yield w
    await w.close()


async def hello_exists(w: World) -> float:
    r = await w.verifier_view().query("ws", Query(op="read", args={"path": "hello.txt"}))
    return 1.0 if r.ok and "hi" in (r.value or {}).get("text", "") else 0.0


async def test_bash_loop_recovers_from_format_errors_and_submits(world: World):
    llm = FakeLLM([say("I will create the file now."),  # no block: format error
                   say("THOUGHT: write it\n```bash\necho hi > hello.txt", finish="length"),  # cut off
                   say("THOUGHT: write it\n```bash\necho hi > hello.txt\n```"),
                   say(f"THOUGHT: done\n```bash\necho {SUBMIT}\n```")])
    pol = BashLoopPolicy("agent", llm)
    res = await run_episode(world, {"agent": pol}, limits=EpisodeLimits(max_steps=10))
    assert await hello_exists(world) == 1.0
    assert pol.stop == "agent_done" and res.stop_condition == "agent_done"
    sent = [m["content"] for m in llm.prompts[1] if m["role"] == "user"]
    assert "EXACTLY ONE action" in sent[-1] and "cut off" in [m["content"] for m in llm.prompts[2] if m["role"] == "user"][-1]
    assert sum(e.type == "tool_call" for e in res.trace.events) == 2  # the write and the submit; format errors never reach the World


async def test_three_format_errors_end_the_episode(world: World):
    pol = BashLoopPolicy("agent", FakeLLM([say("no block")] * 3))
    await run_episode(world, {"agent": pol}, limits=EpisodeLimits(max_steps=5))
    assert pol.stop == "format_error" and pol.format_errors == 3


def test_observation_truncation():
    out = format_observation(0, "x" * 30000, max_output=1000)
    assert "<output_head>" in out and "29000 characters elided" in out and len(out) < 2500


async def test_branch_values_from_restored_snapshots(world: World):
    await world.reset()
    ref0 = await world.snapshot()

    def factory(i: int):
        if i % 2 == 0:
            replies = [say("```bash\necho hi > hello.txt\n```"), say(f"```bash\necho {SUBMIT}\n```")]
        else:
            replies = [say("```bash\necho nope > other.txt\n```"), say(f"```bash\necho {SUBMIT}\n```")]
        return BashLoopPolicy("agent", FakeLLM(replies))

    b0 = await branch_value(world, ref0, step=0, policy_factory=factory, verify=hello_exists, k=4, harness="crucible-bash-loop@1")
    assert b0.v_hat == 0.5 and b0.values == [1.0, 0.0, 1.0, 0.0] and b0.masked == 0 and b0.harness == "crucible-bash-loop@1"
    # from a state where the file already exists, every continuation succeeds
    await world.restore(ref0)
    await world.step("agent", {"kind": "tool_call", "tool_call_id": "x", "function_name": "ws.write_file", "arguments": {"path": "hello.txt", "content": "hi"}})
    ref1 = await world.snapshot()
    b1 = await branch_value(world, ref1, step=1, policy_factory=factory, verify=hello_exists, k=4)
    assert b1.v_hat == 1.0
    assert advantages([b0, b1]) == [None, 0.5]
    lo, hi = wilson(2, 4)
    assert lo < 0.5 < hi
    c = correlation([None, 0.5, 0.0, 0.2], [0.0, 1.0, 0.0, 0.3], ["investigate", "edit", "investigate", "edit"])
    assert c["all"]["n"] == 3
