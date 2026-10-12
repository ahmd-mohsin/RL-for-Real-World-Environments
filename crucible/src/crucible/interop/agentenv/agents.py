"""A real LLM agent behind agentenv's A2A protocol (m13 C7 follow-up): ``LLMToolAgent``.

It is a minimal function-calling agent in the mini-swe-agent / tool-loop style, packaged as an ``AgentEnvAgent`` so
Crucible's ``A2AHarness`` (or any agentenv runner) can drive it:

- **Tools** come from the MCP servers registered through ``urn:agentenv:mcp-config/v1``, e.g. the Crucible World's gateway
  with the actor's role header. They are listed once per task and offered to the model as OpenAI tools.
- **The model** is any OpenAI-compatible chat endpoint set through ``urn:agentenv:agent-config/v1``: ``llm_base_url``,
  ``llm_api_key`` (write-only, redacted on read-back) and ``model``. Pointed at Crucible's capture proxy, every call is
  recorded and the upstream key never reaches the agent.
- **The loop** calls the model, runs every tool call over MCP, appends the results (capped at ``max_tool_output``
  characters), and stops when the model answers without a tool call or ``max_turns`` is reached.
- **The native trajectory** (each model turn and tool call) goes to ``urn:agentenv:trajectory/v1``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    TRAJECTORY_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    WriteOnly,
    a2a_agent,
)

SYSTEM = ("You are a careful software agent. Use the tools to complete the task in the workspace. Check your work by running it. "
          "When the task is complete, reply with a short summary and no tool call.")


class LLMToolAgentConfig(AgentConfig):
    model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: WriteOnly[str | None] = None
    max_turns: int = 30
    max_tokens: int = 2000
    max_tool_output: int = 8000
    temperature: float = 0.0
    system_prompt: str | None = None


def _text(result: Any) -> str:
    parts = [getattr(c, "text", "") for c in (getattr(result, "content", None) or []) if getattr(c, "type", None) == "text"]
    out = "\n".join(p for p in parts if p)
    return ("[tool error] " if getattr(result, "isError", False) else "") + (out or "(no output)")


@a2a_agent(identity=AgentIdentity(name="crucible-llm-tool-agent", description="An LLM function-calling agent over the World's MCP tools", version="1"),
           config=LLMToolAgentConfig, extensions=(MCP_CONFIG_V1, TRAJECTORY_V1))
class LLMToolAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[LLMToolAgentConfig]) -> TaskResult:
        from mcp import ClientSession

        from crucible.world.packaging.mcp_compat import http_streams, mcp_field

        cfg = request.config
        if not (cfg.llm_base_url and cfg.model):
            return TaskResult.builder().failed("not_configured", "agent-config needs llm_base_url and model").build()
        prompt = "\n".join(p.text for p in request.parts if isinstance(p, TextPart))
        servers = dict(request.mcp_servers or {})
        if not servers:
            return TaskResult.builder().failed("no_tools", "no MCP server registered (mcp-config)").build()
        srv = next(iter(servers.values()))
        trajectory: list[dict[str, Any]] = []
        usage = {"in": 0, "out": 0, "tools": 0}
        final = ""
        async with http_streams(srv["url"], headers=srv.get("headers") or {}, timeout_s=30.0, read_timeout_s=300.0) as (rd, wr), ClientSession(rd, wr) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            tools = [{"type": "function", "function": {"name": t.name, "description": (t.description or "")[:1000],
                                                       "parameters": mcp_field(t, "inputSchema") or {"type": "object", "properties": {}}}} for t in listed]
            messages: list[dict[str, Any]] = [{"role": "system", "content": cfg.system_prompt or SYSTEM}, {"role": "user", "content": prompt}]
            headers = {"authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {}
            async with httpx.AsyncClient(timeout=300) as http:
                for turn in range(cfg.max_turns):
                    r = await http.post(f"{cfg.llm_base_url.rstrip('/')}/chat/completions", headers=headers,
                                        json={"model": cfg.model, "messages": messages, "tools": tools, "max_tokens": cfg.max_tokens,
                                              "temperature": cfg.temperature})
                    if r.status_code != 200:
                        return TaskResult.builder().failed("model_error", f"model endpoint {r.status_code}: {r.text[:300]}", error_type="infra_error").build()
                    body = r.json()
                    u = body.get("usage") or {}
                    usage["in"] += int(u.get("prompt_tokens") or 0)
                    usage["out"] += int(u.get("completion_tokens") or 0)
                    msg = (body.get("choices") or [{}])[0].get("message") or {}
                    calls = msg.get("tool_calls") or []
                    messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
                    trajectory.append({"turn": turn, "text": (msg.get("content") or "")[:2000], "tool_calls": [c["function"]["name"] for c in calls]})
                    if not calls:
                        final = msg.get("content") or ""
                        break
                    for c in calls:
                        try:
                            args = json.loads(c["function"].get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        res = _text(await session.call_tool(c["function"]["name"], args))
                        usage["tools"] += 1
                        if len(res) > cfg.max_tool_output:
                            res = res[: cfg.max_tool_output] + f"\n[... {len(res) - cfg.max_tool_output} characters truncated ...]"
                        messages.append({"role": "tool", "tool_call_id": c.get("id") or "", "content": res})
                        trajectory.append({"turn": turn, "tool": c["function"]["name"], "args": args, "result": res[:1000]})
                else:
                    final = "[max_turns reached]"
        return (TaskResult.builder().succeeded().parts([TextPart(text=final or "(no summary)")])
                .usage(Usage(tool_call_count=usage["tools"], input_tokens=usage["in"], output_tokens=usage["out"]))
                .native_trajectory(format="crucible-llm-tool-agent/v1", payload=trajectory).build())
