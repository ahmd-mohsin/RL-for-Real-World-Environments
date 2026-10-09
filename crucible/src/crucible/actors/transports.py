"""Transports for actors that live outside the World process: humans and peer agents.

- ``CallableTransport``: an async function (tests, in-process peers);
- ``FileQueueTransport``: question/answer JSON files in a directory, so any UI (or a person with an
  editor) can answer — the minimal human-in-the-loop channel;
- ``A2ATransport`` (``crucible.interop.agentenv.a2a``): any A2A endpoint — a deployed enterprise agent,
  another LLM agent, or agentenv's human A2A peer (``deploy_human_agent``).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


class CallableTransport:
    def __init__(self, fn: Callable[[str], Awaitable[str]]):
        self.fn = fn

    async def send(self, text: str) -> str:
        return await self.fn(text)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        pass


class FileQueueTransport:
    """Writes ``q-<n>.json``; waits for ``a-<n>.json`` ({"text": ...}) up to ``timeout_s``."""

    def __init__(self, directory: str | Path, *, timeout_s: float = 3600, poll_s: float = 0.5):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s, self.poll_s = timeout_s, poll_s
        self.n = 0

    async def send(self, text: str) -> str:
        self.n += 1
        q = self.dir / f"q-{self.n:04d}.json"
        a = self.dir / f"a-{self.n:04d}.json"
        q.write_text(json.dumps({"n": self.n, "text": text, "asked_at": time.time()}, ensure_ascii=False))
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if a.exists():
                return str(json.loads(a.read_text()).get("text", ""))
            await asyncio.sleep(self.poll_s)
        raise TimeoutError(f"no answer in {a} after {self.timeout_s}s")

    def state_dict(self) -> dict[str, Any]:
        return {"n": self.n}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        self.n = int(st.get("n", 0))
