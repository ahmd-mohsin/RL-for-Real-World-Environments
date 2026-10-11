"""m13 C7 follow-up: ``LLMToolAgent``, a real LLM function-calling agent behind agentenv's A2A protocol, driven by the
A2A harness on a coding World served over MCP (the toy fib task). With a scripted model upstream behind the capture
proxy: the agent lists the World's tools, writes fib.py, runs it, and stops; the World-state grader scores 1.0; the
proxy recorded every model call; a train rollout without token ids is masked (hosted models give none)."""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("agent_env")
from crucible_pack_toy.runtime import FibWorldTask

from crucible.connectors.capture_proxy import CaptureProxy, Upstream
from crucible.core.schema.rollout import RolloutRecord
from crucible.interop.agentenv.agents import LLMToolAgent
from crucible.interop.agentenv.harness import A2AHarness
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.run import Runner, RunSpec

FIB = "a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n"


def scripted_upstream() -> httpx.MockTransport:
    turns = [{"tool_calls": [{"id": "t1", "type": "function", "function": {"name": "ws__write_file", "arguments": json.dumps({"path": "fib.py", "content": FIB})}}]},
             {"tool_calls": [{"id": "t2", "type": "function", "function": {"name": "ws__shell", "arguments": json.dumps({"command": "python3 fib.py"})}}]},
             {"content": "fib.py prints the first 10 Fibonacci numbers."}]
    state = {"i": 0}

    def handle(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert {t["function"]["name"] for t in body["tools"]} >= {"ws__write_file", "ws__shell"}  # the World's tools reached the model
        msg = {"role": "assistant", "content": None, **turns[min(state["i"], len(turns) - 1)]}
        state["i"] += 1
        return httpx.Response(200, json={"id": f"r{state['i']}", "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    return httpx.MockTransport(handle)


async def test_llm_tool_agent_solves_a_coding_world(tmp_path):
    from tests.unit.test_a2a_harness import serve

    proxy = CaptureProxy(tmp_path / "cap", openai=Upstream(base="http://upstream.invalid/v1", key="sk-upstream"), transport=scripted_upstream())
    base, task, server = await serve(proxy.app())
    try:
        h = A2AHarness(agent=LLMToolAgent, capture=proxy, capture_base_url=base,
                       agent_config={"model": "scripted", "llm_base_url": "${capture.base_url}/v1", "llm_api_key": "${capture.key}", "max_turns": 8})
        for mode, run_id in (("eval", "e"), ("train", "t")):
            await Runner(tmp_path).run(RunSpec(run_id=run_id, tasks=[FibWorldTask()], policies=[PolicyRef(id="a2a:llm-tool-agent", make=lambda a: {})],
                                               harnesses=[h], k=1, rollout_mode=mode))
        e = RolloutRecord.model_validate_json((min((tmp_path / "e" / "samples").glob("*/a*")) / "rollout.json").read_text())
        t = RolloutRecord.model_validate_json((min((tmp_path / "t" / "samples").glob("*/a*")) / "rollout.json").read_text())
        assert e.reward == 1.0 and not e.mask_sample and len(e.turns) == 3, (e.failure_kind, e.stop)
        assert t.mask_sample and t.failure_kind == "crucible:capture_incomplete"  # no token ids from this upstream: never trained on
        traj = json.loads((min((tmp_path / "e" / "samples").glob("*/a*")) / "agent_trajectory.json").read_text())
        assert "ws__write_file" in json.dumps(traj)
    finally:
        server.should_exit = True
        await task
