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
    """Writes ``q-<n>-<id>.json``; waits for ``a-<n>-<id>.json`` ({"text": ...}) up to ``timeout_s``.

    Every question gets a fresh id, so an answer file can only answer the question it names. (M2 audit 2026-10-12: with
    ``q-<n>`` names, a World restored to before question n, or a second episode sharing the directory, read the old
    answer to a different question: "Yes, PO 77 approved" came back for "Approve PO 99 for 1M USD?".)"""

    def __init__(self, directory: str | Path, *, timeout_s: float = 3600, poll_s: float = 0.5):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s, self.poll_s = timeout_s, poll_s
        self.n = 0

    async def send(self, text: str) -> str:
        import uuid

        self.n += 1
        qid = f"{self.n:04d}-{uuid.uuid4().hex[:10]}"
        q = self.dir / f"q-{qid}.json"
        a = self.dir / f"a-{qid}.json"
        tmp = self.dir / f".q-{qid}.tmp"
        tmp.write_text(json.dumps({"n": self.n, "id": qid, "text": text, "asked_at": time.time()}, ensure_ascii=False))
        tmp.replace(q)  # a UI never reads a half-written question
        deadline = time.monotonic() + self.timeout_s
        try:
            while time.monotonic() < deadline:
                if a.exists():
                    try:
                        reply = str(json.loads(a.read_text()).get("text", ""))
                    except json.JSONDecodeError:  # being written: read again on the next poll
                        await asyncio.sleep(self.poll_s)
                        continue
                    return reply
                await asyncio.sleep(self.poll_s)
            raise TimeoutError(f"no answer in {a} after {self.timeout_s}s")
        except BaseException:  # timed out or the episode was cancelled: withdraw the question, so nobody answers a dead one
            q.unlink(missing_ok=True)
            raise

    def state_dict(self) -> dict[str, Any]:
        return {"n": self.n}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        self.n = int(st.get("n", 0))
