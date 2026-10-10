"""Any NeMo-Gym resources server as Crucible Worlds (import; m13 E / audit 03 G5). The export side (``server.py``) serves
Crucible tasks *to* NeMo-Gym; this is the reverse, so NeMo-Gym's servers and datasets (workplace_assistant, ruler/mrcr,
math, ...) get Crucible's contract: World state and snapshot, the M10 runtime, the M12 trainer, the acceptance checks.

Pattern: verifiers ``v1/tasksets/nemo_gym`` (``taskset.py`` seeds with the row and verifies with a rebuilt Responses
object; ``response.py`` ``trace_to_nemo_response``).

- **Server** (``NemoGymServer``): launches one resources-server class in the NeMo-Gym Python (``crucible/.envs/nemo``,
  Python 3.13; ``serve_resource.py``) and waits for it, or attaches to a running URL.
- **App** (template ``nemo_gym/resources``): ``start`` posts the dataset row to ``/seed_session`` (the session is the
  server's signed cookie, kept in the App's cookie jar); the row's function tools (``responses_create_params.tools``)
  become the App's tools; each call is ``POST /<tool>`` with the arguments and returns the server's ``output``.
  **State** is a replay journal (json_doc: the row, every call and the digest of its result): ``restore`` re-seeds a fresh
  session and re-issues the calls, verifying each digest (a non-deterministic server raises ``SnapshotMismatch``).
- **Task** (``NemoGymTask``, an M10 RunTask): the instruction is the row's input messages; ``grade`` rebuilds the
  Responses ``response`` from the World trace (``function_call`` / ``function_call_output`` items in order, then the
  policy's final message as ``output_text``) and posts ``/verify`` with the row: the server's own reward is the score
  gate; ``mask_sample = true`` masks the episode (never a 0). ``oracle_actions()`` replays the row's ``ground_truth``
  function calls when the dataset has them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.apps.base import App
from crucible.world.state.base import SnapshotMismatch

NEMO_ROOT = Path(__file__).resolve().parents[5] / "RL-Environments" / "06-frameworks-and-env-code" / "nvidia-nemo-gym"
NEMO_PY = Path(__file__).resolve().parents[4] / ".envs" / "nemo" / "bin" / "python"


class NemoGymServer:
    def __init__(self, resource_server: str | None = None, *, url: str | None = None, python: Path | str = NEMO_PY, nemo_root: Path | str = NEMO_ROOT):
        self.resource_server, self.url, self.python, self.nemo_root = resource_server, url, str(python), Path(nemo_root)
        self.proc: subprocess.Popen | None = None

    def start(self, timeout_s: float = 120.0) -> str:
        if self.url is not None:
            return self.url
        import httpx

        port_file = Path(tempfile.mkdtemp(prefix="nemo-gym-")) / "port"
        env = {**os.environ, "NEMO_GYM_RESOURCE_SERVER": self.resource_server, "PYTHONPATH": str(self.nemo_root)}
        self.proc = subprocess.Popen([self.python, str(Path(__file__).with_name("serve_resource.py")), str(port_file)], cwd=str(self.nemo_root), env=env,
                                     stdout=subprocess.DEVNULL, stderr=open(port_file.with_suffix(".log"), "wb"), start_new_session=True)  # noqa: SIM115
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"NeMo-Gym server exited: {port_file.with_suffix('.log').read_text(errors='replace')[-2000:]}")
            if port_file.exists() and port_file.read_text().strip():
                url = f"http://127.0.0.1:{port_file.read_text().strip()}"
                try:
                    httpx.get(url, timeout=1)
                    self.url = url
                    return url
                except httpx.HTTPError:
                    pass
            time.sleep(0.3)
        raise TimeoutError("NeMo-Gym server did not start")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            import signal

            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class NemoGymApp(App):
    template = "nemo_gym/resources"
    PATH_PARAMS = ("path",)

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.client: Any = None
        self.journal: dict = {}

    def _row(self) -> dict:
        return dict(self.spec.params["row"])

    async def _seed(self) -> None:
        import httpx

        if self.client is not None:
            await self.client.aclose()
        self.client = httpx.AsyncClient(base_url=self.spec.params["url"].rstrip("/"), timeout=self.spec.params.get("timeout_s", 120.0),
                                        headers=dict(self.spec.params.get("headers") or {}))
        r = await self.client.post("/seed_session", json=self._row())
        r.raise_for_status()

    async def start(self) -> None:
        await self._seed()
        self.journal = {"row_digest": _digest(json.dumps(self._row(), sort_keys=True)), "calls": []}
        await self._write()

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _write(self) -> None:
        path = self.state_params["path"]
        await self.ctx.exec.run(f"mkdir -p {shlex.quote(posixpath.dirname(path))}")
        await self.ctx.exec.write_text(path, json.dumps(self.journal, sort_keys=True, separators=(",", ":")))

    async def list_tools(self) -> list[ToolSpec]:
        out = []
        for t in self._row()["responses_create_params"].get("tools") or []:
            if t.get("type") == "function":
                out.append(ToolSpec(app=self.id, name=t["name"], description=t.get("description") or "", input_schema=t.get("parameters") or {"type": "object"}))
        return out

    async def _issue(self, tool: str, args: dict) -> str:
        r = await self.client.post(f"/{tool}", json=args)
        r.raise_for_status()
        d = r.json()
        out = d.get("output", d)
        return out if isinstance(out, str) else json.dumps(out, sort_keys=True, default=str)

    async def call(self, tool: str, args: dict, *, actor: str, call_id: str) -> ToolResult:
        text = await self._issue(tool, args)
        self.journal["calls"].append({"tool": tool, "args": args, "digest": _digest(text), "id": call_id, "output": text[:65536]})
        await self._write()
        return ToolResult(tool_call_id=call_id, function_name=f"{self.id}.{tool}", ok=True, content=text)

    async def after_restore(self) -> None:
        doc = json.loads(await self.ctx.exec.read_text(self.state_params["path"]))
        await self._seed()
        for i, c in enumerate(doc["calls"]):
            if _digest(await self._issue(c["tool"], c["args"])) != c["digest"]:
                raise SnapshotMismatch(f"{self.id}: replayed call {i} ({c['tool']}) gave a different result: the server is not deterministic")
        self.journal = doc

    async def verify(self, response: dict) -> dict:
        r = await self.client.post("/verify", json={**self._row(), "response": response})
        r.raise_for_status()
        return r.json()


def trace_to_response(trace: Any, row: dict, *, actor: str = "agent", calls: list[dict] | None = None) -> dict:
    """A NeMo-Gym Responses object (port of verifiers ``trace_to_nemo_response``): the tool calls and their outputs in
    order, then the actor's final message. ``calls`` (the App journal: exactly the calls applied to the current server
    session, which follows ``World.restore``) take precedence over the trace, whose log keeps abandoned branches."""
    out: list[dict] = []
    final = None
    if calls is not None:
        for i, c in enumerate(calls):
            cid = str(c.get("id") or f"call_{i}")
            out.append({"type": "function_call", "call_id": cid, "name": c["tool"], "arguments": json.dumps(c.get("args") or {}), "id": f"fc_{i}",
                        "status": "completed"})
            out.append({"type": "function_call_output", "call_id": cid, "output": c.get("output", "")})
    for e in trace.events:
        if e.actor not in (actor, None):
            continue
        if calls is None and e.type == "tool_call" and e.actor == actor:
            name = str(e.data.get("function", "")).split(".", 1)[-1]
            out.append({"type": "function_call", "call_id": str(e.data.get("id")), "name": name, "arguments": json.dumps(e.data.get("arguments") or {}),
                        "id": f"fc_{len(out)}", "status": "completed"})
        elif calls is None and e.type == "tool_result" and e.actor == actor:
            out.append({"type": "function_call_output", "call_id": str(e.data.get("id")), "output": e.text or ""})
        elif e.type == "done" and e.actor == actor and e.text and e.text not in ("environment_terminal", "script finished"):
            final = e.text
    if final:
        out.append({"id": f"msg_{len(out)}", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": final, "annotations": []}]})
    params = row.get("responses_create_params") or {}
    return {"id": "resp_crucible", "created_at": int(time.time()), "model": str(params.get("model") or "crucible"), "object": "response",
            "output": out, "parallel_tool_calls": params.get("parallel_tool_calls", True), "tool_choice": params.get("tool_choice", "auto"),
            "tools": params.get("tools") or [], "status": "completed"}


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


class NemoGymTask:
    def __init__(self, id: str, url: str, row: dict, *, app: str = "gym", tool_timeout_s: float = 120.0):
        self.id, self.world_id, self.url, self.row, self.app, self.tool_timeout_s = id, f"nemo_gym/{id.split('/')[1] if '/' in id else id}", url, row, app, tool_timeout_s

    @property
    def instruction(self) -> str:
        msgs = (self.row.get("responses_create_params") or {}).get("input") or []
        if isinstance(msgs, str):
            return msgs
        parts = []
        for m in msgs:
            c = m.get("content")
            text = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
            parts.append(f"[{m.get('role', 'user')}] {text}" if m.get("role") == "system" else text)
        return "\n\n".join(parts)

    def spec(self, workdir: Path):
        from crucible.core.schema.world import WorldSpec

        return WorldSpec.model_validate({"world": self.world_id, "limits": {"tool_timeout_s": self.tool_timeout_s},
                                         "apps": [{"id": self.app, "template": "nemo_gym/resources",
                                                   "state": {"adapter": "json_doc", "params": {"path": str(Path(workdir).resolve() / "nemo" / "journal.json")}},
                                                   "params": {"url": self.url, "row": self.row}}],
                                         "actors": [{"id": "agent", "kind": "policy", "tools": [f"{self.app}.*"]}]})

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world.world import World

        workdir = Path(workdir)
        w = World(self.spec(workdir), instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
        await w.start()
        await w.reset()
        return w

    def oracle_actions(self) -> list:
        """The row's ``ground_truth`` function calls as World actions (datasets that ship them, e.g. workplace_assistant)."""
        from crucible.core.schema.actions import ToolCall

        gt = self.row.get("ground_truth")
        if not isinstance(gt, list):
            return []
        out = []
        for i, c in enumerate(gt):
            args = c.get("arguments", {})
            out.append(ToolCall(tool_call_id=f"gt{i}", function_name=f"{self.app}.{c['name']}", arguments=json.loads(args) if isinstance(args, str) else args))
        return out

    def plan(self):
        from crucible.verify.runtime.plan import Check, Gate, Plan, Verdict

        async def nemo_verify(ctx) -> Verdict:
            world = ctx.params["world"]
            app = world.apps[self.app]
            doc = await app.verify(trace_to_response(world.to_trace(), self.row, calls=list(app.journal.get("calls", []))))
            if doc.get("mask_sample") or (doc.get("instance_config") or {}).get("mask_sample"):
                from crucible.verify.runtime.trusted import InfraError

                raise InfraError(f"nemo-gym masked the sample: {doc.get('failure_kind') or 'mask_sample'}")
            r = float(doc.get("reward", 0.0))
            return Verdict(True, {"score": r, "normalized": min(1.0, max(0.0, r)), "normalized_reward": min(1.0, max(0.0, r)),
                                  "components": {k: v for k, v in doc.items() if isinstance(v, int | float) and k != "reward"}})

        return Plan(self.id, [Gate("nemo_reward", kind="score", description="the NeMo-Gym server's own /verify reward")],
                    [Check("nemo_reward", "nemo_reward", nemo_verify, tier="fast", provenance="nemo-gym:/verify")])

    async def grade(self, world, workdir: Path):
        from crucible.verify.runtime.plan import Context
        from crucible.verify.runtime.trusted import Trusted

        return await self.plan().arun(Context(Trusted(Path(workdir)), params={"view": world.verifier_view(), "world": world}))


async def wait(coro):  # pragma: no cover - convenience for scripts
    return await asyncio.wait_for(coro, timeout=600)
