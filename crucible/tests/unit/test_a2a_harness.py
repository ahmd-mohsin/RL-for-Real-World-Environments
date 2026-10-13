"""m13 C7: any agentenv A2A agent as a Crucible harness. An A2A agent served per attempt gets the World's MCP URL with
its role header (mcp-config) and the capture proxy in its agent-config; it calls the World's tools over MCP and its model
through the proxy. With an upstream that returns token ids the train rollout is complete and graded 1.0; with a hosted
upstream (no token ids) the train rollout is masked capture_incomplete; a failed A2A task is a typed agent failure."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

pytest.importorskip("agent_env")
from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    TRAJECTORY_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    WriteOnly,
    a2a_agent,
)
from crucible_pack_toy.runtime import HelpdeskTask
from crucible_pack_toy.verify import GOLD

from crucible.connectors.capture_proxy import CaptureProxy, Upstream
from crucible.core.schema.rollout import RolloutRecord
from crucible.interop.agentenv.harness import A2AHarness, token_record
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.run import Runner, RunSpec


class ToolAgentConfig(AgentConfig):
    model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: WriteOnly[str | None] = None
    fail: bool = False


@a2a_agent(identity=AgentIdentity(name="crucible-test-tool-agent", description="calls the model once, then the gold actions over MCP", version="1"),
           config=ToolAgentConfig, extensions=(MCP_CONFIG_V1, TRAJECTORY_V1))
class ToolAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[ToolAgentConfig]) -> TaskResult:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        cfg = request.config
        if cfg.fail:
            return TaskResult.builder().failed("refused", "model refused").build()
        async with httpx.AsyncClient() as c:  # one model call, through whatever base URL the harness configured
            r = await c.post(f"{cfg.llm_base_url}/chat/completions", headers={"authorization": f"Bearer {cfg.llm_api_key}"},
                             json={"model": cfg.model, "messages": [{"role": "user", "content": request.parts[0].text}]})
            r.raise_for_status()
        srv = next(iter(request.mcp_servers.values()))  # one server per attempt, named world-<tag>
        async with streamablehttp_client(srv["url"], headers=srv.get("headers") or {}) as (rd, wr, _), ClientSession(rd, wr) as s:
            await s.initialize()
            for a in GOLD:
                app, _, tool = a.function_name.partition(".")
                await s.call_tool(f"{app}__{tool}", a.arguments)
        return (TaskResult.builder().succeeded().parts([TextPart(text="all tickets handled")])
                .native_trajectory(format="crucible-test-tool-agent/v1", payload=[{"type": "tool", "name": a.function_name} for a in GOLD]).build())


def upstream(with_tokens: bool) -> httpx.MockTransport:
    def handle(req: httpx.Request) -> httpx.Response:
        ch: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        body: dict[str, Any] = {"id": "x", "choices": [ch], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        if with_tokens:
            ch["token_ids"], ch["logprobs"] = [7, 8], {"content": [{"token": "o", "logprob": -0.1}, {"token": "k", "logprob": -0.2}]}
            body["prompt_token_ids"] = [1, 2, 3]
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handle)


async def serve(app) -> tuple[str, asyncio.Task, Any]:
    import uvicorn

    from crucible.world.packaging.sidecar import free_port

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    return f"http://127.0.0.1:{port}", task, server


async def run(tmp_path, *, with_tokens: bool, fail: bool = False, run_id="r", mode="train"):
    proxy = CaptureProxy(tmp_path / f"cap-{run_id}", openai=Upstream(base="http://upstream.invalid/v1", key="sk-upstream"), transport=upstream(with_tokens))
    base, task, server = await serve(proxy.app())
    try:
        h = A2AHarness(agent=ToolAgent, capture=proxy, capture_base_url=base, policy_version=3,
                       agent_config={"model": "m", "llm_base_url": "${capture.base_url}/v1", "llm_api_key": "${capture.key}", "fail": fail})
        s = await Runner(tmp_path).run(RunSpec(run_id=run_id, tasks=[HelpdeskTask(backend="agentenv")], policies=[PolicyRef(id="a2a:tool-agent", make=lambda a: {})],
                                               harnesses=[h], k=1, rollout_mode=mode))
        adir = min((tmp_path / run_id / "samples").glob("*/a*"))
        return s, adir, proxy
    finally:
        server.should_exit = True
        await task


async def test_a2a_agent_complete_train_rollout(tmp_path):
    _s, adir, _proxy = await run(tmp_path, with_tokens=True, run_id="tok")
    ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text())
    assert not ro.mask_sample and ro.reward == 1.0, (ro.failure_kind, ro.stop)
    assert ro.turns and ro.turns[0].completion_token_ids == [7, 8] and ro.turns[0].logprobs == [-0.1, -0.2]
    assert (adir / "agent_trajectory.json").exists() and (adir / "a2a_reply.txt").read_text() == "all tickets handled"
    log = next((tmp_path / "cap-tok").glob("*.jsonl")).read_text()
    assert "sk-upstream" not in log and "Bearer" not in log  # the upstream key stays on the proxy


async def test_hosted_upstream_masks_train_rollout(tmp_path):
    _s, adir, _ = await run(tmp_path, with_tokens=False, run_id="hosted")
    ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text())
    assert ro.mask_sample and ro.failure_kind == "crucible:capture_incomplete" and ro.blame == "infra"


async def test_failed_a2a_task_is_a_typed_agent_failure(tmp_path):
    _s, adir, _ = await run(tmp_path, with_tokens=True, fail=True, run_id="fail", mode="eval")  # (train: no calls -> capture_incomplete)
    rec = json.loads((adir / "record.json").read_text())
    assert rec["failure"]["kind"] == "agent_run_error" and "model refused" in rec["failure"]["reason"]
    assert rec["failure"]["blame"] == "policy" and not rec.get("mask_sample")  # the agent's own failure is its outcome


def test_token_record_parsing():
    assert token_record({"prompt_token_ids": [1], "choices": [{"token_ids": [2], "logprobs": {"content": [{"token": "a", "logprob": -1.0}]}}]}) == \
        {"prompt_token_ids": [1], "completion_token_ids": [2], "logprobs": [-1.0]}
    assert token_record({"prompt_token_ids": [1], "choices": [{"logprobs": {"content": [{"token": "token_id:5", "logprob": -0.5}]}}]})["completion_token_ids"] == [5]
    assert token_record({"choices": [{"message": {"content": "x"}}]}) is None
