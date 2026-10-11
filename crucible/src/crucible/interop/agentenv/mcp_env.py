"""agentenv-protocol environment images as Crucible Apps (m13 C9; audit 02). DEPEND on ``agentenv_protocol``'s data-plane
client.

An agentenv env image (an ``MCPServerEnv``'s ``DockerImageArtifact``: an MCP server plus the v1 data plane at
``/agentenv`` and its card at ``/.well-known/agent-env.json``) runs as one container per World and becomes App
``agentenv/mcp_env``:

- **Tools** come from the env's MCP endpoint (the card's ``additionalInterfaces`` ``mcp`` URL), over one persistent
  session (``common/mcp_server``). They are **cross-checked against the card**: every tool the card advertises under
  ``capabilities.tools`` must be served (a drifted image fails at start); tools served but not on the card are
  reported (``uncarded``) and hidden when ``card_tools_only`` is set.
- **State** (adapter ``agentenv_data``) goes over the data plane: the fingerprint and snapshot are ``data/get``'s
  parts (canonical JSON), restore is ``data/reset`` + ``data/add`` of the snapshot's parts, verified by fingerprint.
  The env must advertise all three operations (``capabilities.operations``).
- **Seed:** ``seed`` (a list of parts) or ``seed_file`` (pack-relative JSON, added as one data part) is loaded with
  ``data/reset`` + ``data/add`` at start.

Params: ``image``, ``port`` (default 18765, the agentenv base-service port), ``env``, ``seed`` / ``seed_file``,
``card_tools_only``, ``start_timeout_s``. The container runs on this machine's Docker daemon, published on loopback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from typing import Any, ClassVar

import httpx

from crucible.core.ids import sha256_text
from crucible.core.schema.actions import ToolSpec
from crucible.core.schema.state import StateDiff, StateRef
from crucible.world.apps.mcp_server import McpServerApp
from crucible.world.exec import _run_subprocess
from crucible.world.state.base import SnapshotMismatch, StateAdapter

REQUIRED_OPS = ("data/reset", "data/add", "data/get")


def _canon(parts: list[dict]) -> str:
    return json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)


class AgentEnvDataAdapter(StateAdapter):
    kind = "agentenv_data"
    OPS: ClassVar[dict[str, str]] = {"parts": "the env's data/get parts", "data": "the first data part's payload"}

    def __init__(self, app_id, exec_, params, store, *, app: AgentEnvMcpApp | None = None):
        super().__init__(app_id, exec_, params, store)
        self.env_app = app

    async def _parts(self) -> list[dict]:
        from agentenv_protocol.client import get_data

        r = await get_data(self.env_app.base_url)
        return [p.model_dump(mode="json", exclude_none=True) for p in r.parts]

    async def fingerprint(self) -> str:
        return sha256_text(_canon(await self._parts()))

    async def snapshot(self) -> StateRef:
        parts = await self._parts()
        text = _canon(parts)
        return StateRef(app=self.app, adapter=self.kind, fingerprint=sha256_text(text), blob=self.store.put_bytes(text.encode()), meta={"parts": len(parts)})

    async def restore(self, ref: StateRef) -> None:
        parts = json.loads(self.store.get_bytes(ref.blob or ""))
        await self.env_app.load(parts)
        now = await self.fingerprint()
        if now != ref.fingerprint:
            raise SnapshotMismatch(f"{self.app}: the env's data/get differs after data/reset + data/add of the snapshot ({now[:12]} vs {ref.fingerprint[:12]})")

    async def diff(self, a: StateRef, b: StateRef) -> StateDiff:
        pa, pb = (json.loads(self.store.get_bytes(r.blob or "")) for r in (a, b))
        return StateDiff(app=self.app, adapter=self.kind, before=a.fingerprint, after=b.fingerprint,
                         detail={} if a.fingerprint == b.fingerprint else {"before_parts": pa[:3], "after_parts": pb[:3]})

    async def q_parts(self) -> list[dict]:
        return await self._parts()

    async def q_data(self) -> Any:
        return next((p.get("data") for p in await self._parts() if p.get("kind") == "data"), None)


class AgentEnvMcpApp(McpServerApp):
    template = "agentenv/mcp_env"

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.state = AgentEnvDataAdapter(spec.id, ctx.exec, dict(spec.state.params), ctx.store, app=self)
        self.container = f"crucible-aemcp-{spec.id}-{uuid.uuid4().hex[:8]}"
        self.base_url = ""
        self.card: dict = {}
        self.uncarded: list[str] = []

    async def _docker(self, *args: str, timeout_s: float = 120):
        return await _run_subprocess(["docker", *args], cwd=None, env=None, timeout_s=timeout_s, stdin=None)

    async def load(self, parts: list[dict]) -> None:
        from agentenv_protocol.client import add_data, reset_data
        from agentenv_protocol.types import DataPart, FilePart

        await reset_data(self.base_url)
        if parts:
            typed = [DataPart.model_validate(p) if p.get("kind") == "data" else FilePart.model_validate(p) for p in parts]
            await add_data(self.base_url, typed)

    def _seed(self) -> list[dict]:
        p = self.spec.params
        if p.get("seed") is not None:
            return list(p["seed"])
        if p.get("seed_file"):
            return [{"kind": "data", "data": json.loads(self.ctx.host_seed(p["seed_file"]).read_text())}]
        return []

    async def start(self) -> None:
        p = self.spec.params
        if not self.ctx.exec.is_local:
            raise RuntimeError("agentenv/mcp_env starts a container on this machine's Docker daemon; use the local or agentenv backend")
        port = int(p.get("port", 18765))
        argv = ["run", "-d", "--rm", "--name", self.container, "--label", "crucible.sandbox=1", "-p", f"127.0.0.1::{port}"]
        for k, v in (p.get("env") or {}).items():
            argv += ["-e", f"{k}={v}"]
        out = await self._docker(*argv, p["image"], timeout_s=600)
        if not out.ok:
            raise RuntimeError(f"{self.id}: env image {p['image']} failed to start: {(out.stderr or out.stdout)[-500:]}")
        mapped = (await self._docker("port", self.container, str(port))).stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        self.base_url = f"http://127.0.0.1:{mapped}"
        deadline = time.monotonic() + float(p.get("start_timeout_s", 120))
        async with httpx.AsyncClient() as client:
            while True:
                with contextlib.suppress(httpx.HTTPError, ValueError):
                    r = await client.get(f"{self.base_url}/.well-known/agent-env.json", timeout=5)
                    if r.status_code == 200:
                        self.card = r.json()
                        break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self.id}: no env card at {self.base_url} (is {p['image']} an agentenv-protocol env?)")
                await asyncio.sleep(0.3)
        ops = (self.card.get("capabilities") or {}).get("operations") or []
        missing = [o for o in REQUIRED_OPS if o not in ops]
        if missing:
            raise RuntimeError(f"{self.id}: the env does not advertise {missing}; the agentenv_data state adapter needs data/get, data/reset and data/add")
        mcp = next((i["url"] for i in self.card.get("additionalInterfaces") or [] if i.get("transport") == "mcp"), "/mcp")
        self.spec = self.spec.model_copy(update={"params": {**p, "url": self.base_url + mcp}})
        await super().start()
        served = {t.name for t in self._tools}
        carded = [t["name"] for t in (self.card.get("capabilities") or {}).get("tools") or []]
        absent = sorted(set(carded) - served)
        if absent:
            raise RuntimeError(f"{self.id}: the env card advertises tools the server does not serve: {absent}")
        self.uncarded = sorted(served - set(carded))
        if p.get("card_tools_only") and carded:
            self._tools = [t for t in self._tools if t.name in carded]
        await self.load(self._seed())

    async def list_tools(self) -> list[ToolSpec]:
        return self._tools

    async def stop(self) -> None:
        await super().stop()
        await self._docker("rm", "-f", self.container)
