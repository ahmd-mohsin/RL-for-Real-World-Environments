"""A2A on agentenv's protocol (DEPEND: ``agentenv_protocol.a2a_agent``, ``agent_env.a2a_agent.protocol``).

- ``serve_agent`` runs any ``AgentEnvAgent`` (agentenv's typed A2A agent base) on the caller's event
  loop via its own ``create_app()``.
- ``A2AClient`` sends a message and waits for the terminal reply with agentenv's ``send_a2a_message`` /
  ``poll_a2a_task`` / ``TerminalResponse`` (the same calls ``prompt_agent`` uses).
- ``RelayAgent`` is the executor the agentenv gateway calls for ``nl`` trigger actions; it hands each
  instruction to a Python callback (the World's inbox).
- ``A2ATransport`` lets M2 human / peer actors live behind any A2A endpoint (a human console,
  a deployed enterprise agent, another LLM agent).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agent_env.a2a_agent.protocol import TerminalResponse, poll_a2a_task, send_a2a_message
from agentenv_protocol.a2a_agent import AgentEnvAgent, AgentIdentity, TaskRequest, TaskResult, TextPart, a2a_agent

from crucible.world.packaging.sidecar import free_port


class ServedAgent:
    def __init__(self, agent: AgentEnvAgent, host: str = "127.0.0.1", port: int = 0):
        self.agent = agent
        self.host = host
        self.port = port or free_port()
        self._server = None
        self._task: asyncio.Task | None = None

    @property
    def url(self) -> str:
        """Base URL — agentenv convention: clients and the gateway executor POST to ``<url>/a2a``."""
        return f"http://{self.host}:{self.port}"

    async def start(self) -> ServedAgent:
        import uvicorn

        self._server = uvicorn.Server(uvicorn.Config(self.agent.create_app(), host=self.host, port=self.port, log_level="warning"))
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(400):
            if self._server.started:
                return self
            if self._task.done():
                raise RuntimeError(f"A2A agent failed to start: {self._task.exception()}")
            await asyncio.sleep(0.025)
        raise TimeoutError("A2A agent did not start")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, 5)


async def serve_agent(agent: AgentEnvAgent, *, host: str = "127.0.0.1", port: int = 0) -> ServedAgent:
    return await ServedAgent(agent, host, port).start()


class A2AClient:
    def __init__(self, url: str, *, timeout_s: int = 600, poll_s: int = 1):
        self.url = url
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.context_id: str | None = None

    async def ask(self, text: str) -> TerminalResponse:
        task_id, ctx = await send_a2a_message(self.url, [{"kind": "text", "text": text}], uuid.uuid4().hex, self.context_id, self.timeout_s)
        self.context_id = ctx or self.context_id
        task = await poll_a2a_task(self.url, task_id, self.timeout_s, self.poll_s)
        return TerminalResponse.from_message((task.get("status") or {}).get("message") or {})


Deliver = Callable[[dict[str, Any]], Awaitable[None]]


def make_relay_agent(deliver: Deliver) -> AgentEnvAgent:
    """An A2A agent that forwards each instruction (JSON {to, from, text} or plain text) to ``deliver``."""

    @a2a_agent(identity=AgentIdentity(name="crucible-relay", description="Delivers gateway nl-trigger instructions into a Crucible World.", version="1"))
    class RelayAgent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            text = "".join(p.text for p in request.parts if isinstance(p, TextPart))
            try:
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError
            except ValueError:
                payload = {"text": text}
            await deliver(payload)
            return TaskResult.builder().succeeded().add_text("delivered").build()

    return RelayAgent()


class A2ATransport:
    """Transport for human / peer actors (registry kind ``transport``, name ``a2a``)."""

    def __init__(self, url: str, **kw: Any):
        self.client = A2AClient(url, **kw)

    async def send(self, text: str) -> str:
        return (await self.client.ask(text)).response_text

    def state_dict(self) -> dict[str, Any]:
        return {"context_id": self.client.context_id}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        self.client.context_id = st.get("context_id")
