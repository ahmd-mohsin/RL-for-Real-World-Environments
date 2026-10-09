"""The World's tool-call choke point and the policy loop (findings from the verifiers / NeMo-Gym / OpenEnv /
EnterpriseOps / MiMo survey): every malformed or failing call becomes a typed result, infra faults mask the
episode while policy faults are scored, histories stay valid under parallel calls, and stop conditions are typed."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.actors.llm import ContextOverflow, LLMResponse
from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, ScriptedPolicy, openai_name, run_episode
from crucible.core.schema.actions import Message, ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World


def spec(**limits) -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/loop",
        "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}},
                 {"id": "mail", "template": "common/mailbox", "state": {"adapter": "sql", "params": {"path": "mail.db"}}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*", "mail.*"]}, {"id": "boss", "kind": "simulated", "tools": []}],
        "limits": limits or {},
    })


@pytest.fixture
async def world(tmp_path: Path):
    w = World(spec(max_tool_output_chars=300), store=Store(tmp_path / "s"), workdir=tmp_path / "w")

    class Boss:
        actor_id = "boss"

        async def respond(self, turn):
            return [Message(to="agent", text="thanks, noted")] if turn.inbox else []

        def state_dict(self):
            return {}

        def load_state_dict(self, d):
            pass

    w.attach(Boss())
    await w.start()
    await w.reset()
    yield w
    await w.close()


async def call(w: World, name: str, args):
    r = await w.step("agent", ToolCall(tool_call_id="c", function_name=name, arguments=args) if isinstance(args, dict)
                     else ToolCall.model_construct(tool_call_id="c", function_name=name, arguments=args))
    return r.observation.tool_results[0]


async def test_choke_point_types_every_failure(world: World):
    r = await call(world, "ws.read_file", {"path": 5})  # crashed World.step before (AttributeError)
    assert not r.ok and r.error == "invalid_arguments" and r.fault == "policy" and "path" in r.content
    r = await call(world, "ws.read_file", ["a.txt"])
    assert r.error == "invalid_arguments" and "JSON object" in r.content
    r = await call(world, "ws.read_file", {"__unparseable__": "{bad"})
    assert r.error == "invalid_arguments" and "not valid JSON" in r.content
    r = await call(world, "ws.raed_file", {"path": "x"})
    assert r.error == "unknown_tool" and "ws.read_file" in r.content  # near-match suggestion
    r = await call(world, "mail.send_email", {"to": "bob@x", "subject": "s", "body": "b"})  # stored "b","o","b"... before
    assert r.error == "invalid_arguments" and "to" in r.content
    r = await call(world, "ws.read_file", {"path": "missing.txt"})
    assert not r.ok and r.fault == "tool" and r.content.startswith("FileNotFoundError")  # a domain error: scored, not masked
    assert not world.failures
    # bounded output: one head+tail cut with a banner; the full text is in the store
    await call(world, "ws.write_file", {"path": "big.txt", "content": "".join(f"line {i}\n" for i in range(400))})
    r = await call(world, "ws.read_file", {"path": "big.txt"})
    assert r.truncated and "characters truncated" in r.content and r.content_ref and "line 399" in r.content
    assert "line 399" in world.store.get_bytes(r.content_ref).decode()
    # an App that crashes is an environment bug: contained, recorded, the episode is masked
    async def boom(**kw):
        raise KeyError("internal state")

    world.apps["ws"].tool_list_dir = boom
    r = await call(world, "ws.list_dir", {})
    assert not r.ok and r.error == "internal_error" and r.fault == "infra" and "KeyError" not in r.content
    assert world.failures and world.failures[-1].kind == "crucible:tool_crashed"
    assert any(e.type == "infra_fault" and "KeyError" in str(e.data.get("traceback")) for e in world.log)  # traceback: trace only


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts, self.keys = list(replies), [], []
        self.model = "fake"

    async def complete(self, messages, *, tools=None, sample_key=None, **kw):
        self.prompts.append([dict(m) for m in messages])
        self.keys.append(sample_key)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def tc(i, name, args):
    return {"id": f"t{i}", "name": name, "arguments": args}


async def test_parallel_calls_keep_a_valid_history_and_reasoning(world: World):
    llm = FakeLLM([
        LLMResponse(text="", tool_calls=[tc(1, "send_message", {"to": "boss", "text": "starting"}), tc(2, "ws__write_file", {"path": "a.txt", "content": "x"}),
                                         tc(3, "ws__read_file", {"path": "a.txt"})], raw_message={"reasoning": "plan: tell boss, write, check"}),
        LLMResponse(text="done", raw_message={"reasoning": "all good"}),
    ])
    p = LLMToolPolicy("agent", llm)
    res = await run_episode(world, {"agent": p}, reset=False)
    assert res.stop_condition == "agent_done" and not res.masked and res.model_calls["agent"]
    msgs = llm.prompts[-1]
    roles = [m["role"] for m in msgs]
    i = next(k for k, m in enumerate(msgs) if m.get("tool_calls"))
    assert msgs[i]["reasoning"] == "plan: tell boss, write, check"  # provider reasoning replayed verbatim
    assert roles[i + 1 : i + 4] == ["tool", "tool", "tool"]  # every result of the batch before ...
    assert msgs[i + 4]["role"] == "user" and "thanks, noted" in msgs[i + 4]["content"]  # ... the boss's reply that arrived mid-batch
    assert llm.keys[0] == world.episode_id  # rollouts are distinct cache entries


async def test_stop_conditions(world: World):
    res = await run_episode(world, {"agent": LLMToolPolicy("agent", FakeLLM([ContextOverflow("prompt is too long")]))}, reset=False)
    assert res.stop_condition == "context_overflow" and res.truncated and not res.masked  # scored, never hidden
    bad = [ToolCall(tool_call_id=f"b{i}", function_name="ws.read_file", arguments={"path": 5}) for i in range(10)]
    res = await run_episode(world, {"agent": ScriptedPolicy("agent", bad)}, limits=EpisodeLimits(max_tool_errors=3))
    assert res.stop_condition == "too_many_errors" and res.steps == 3 and res.truncated and not res.masked
    from crucible.actors.llm import BudgetExceeded

    res = await run_episode(world, {"agent": LLMToolPolicy("agent", FakeLLM([BudgetExceeded("out of credits")]))})
    assert res.masked and res.stop_condition == "crucible:credits_exhausted"


def test_openai_names():
    assert openai_name("ws.read_file") == "ws__read_file"
    long = "erp_" + "x" * 80 + ".do.it"
    n = openai_name(long)
    assert len(n) <= 64 and n != openai_name(long + "2") and all(c.isalnum() or c in "_-" for c in n)
