"""Remote World client over the NeMo-Gym resources protocol (m13 D-3; audit 05 §2.2 E): rollouts drive Worlds served by
env-server processes (``interop/nemo_gym/server.py``, any host) instead of in-process.

- ``remote_episode(urls, task_id, policy, rid=...)`` drives any Crucible ``Policy`` (``act(Observation) -> Action``):
  ``/seed_session`` gives the tools and the instruction; each ``ToolCall`` becomes ``POST /<app>__<tool>``; the episode ends
  when the policy is done or at ``max_steps``; ``/verify`` grades **from the World's state** on the server;
  ``/close_session`` frees the slot. ``LLMToolPolicy`` therefore runs remotely unchanged: its context management,
  compaction and token capture (``model_calls``) are the in-process ones.
- **Session affinity**: a rollout's session lives on one server (``crc32(rid) % len(urls)``), and every call carries
  the session id (``X-Resources-Session-Id``; the server also reads the cookie NeMo-Gym's own clients send).
- **Admission**: a server at capacity answers 503 (``queue_timeout``); the episode is **masked**
  (``env_admission_timeout``), never scored 0. A lost server (connection error) masks it as ``env_unreachable``.
- ``remote_world_rollout(urls, task_for, codec)`` is the M12 runtime's ``rollout_fn`` over remote Worlds (the remote
  analogue of ``train.world_env.world_rollout``): an ``EngineLLM`` (exact token-in-token-out) inside an
  ``LLMToolPolicy``, recorded as ``crucible.rollout/v1``.
"""

from __future__ import annotations

import time
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.actions import Done, Message, Observation, ToolCall, ToolResult, ToolSpec


@dataclass
class RemoteEpisode:
    reward: float | None
    masked: bool
    failure_kind: str | None
    steps: int
    stop: str
    verify: dict[str, Any] = field(default_factory=dict)
    server: str = ""


def pick(urls: list[str], rid: str) -> str:
    return urls[zlib.crc32(rid.encode()) % len(urls)]


async def remote_episode(urls: list[str], task_id: str, policy: Any, *, rid: str | None = None, max_steps: int = 30, client: Any = None,
                         actor: str = "agent") -> RemoteEpisode:
    import httpx

    rid = rid or uuid.uuid4().hex
    url = pick(list(urls), rid).rstrip("/")
    sid = f"{rid}-{uuid.uuid4().hex[:6]}"
    own = client is None
    c = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
    hdr = {"X-Resources-Session-Id": sid}
    try:
        try:
            r = await c.post(url + "/seed_session", json={"resources_session_id": sid, "task_id": task_id}, headers=hdr)
        except httpx.HTTPError as e:
            return RemoteEpisode(None, True, "env_unreachable", 0, f"seed failed: {e}", server=url)
        if r.status_code == 503:
            return RemoteEpisode(None, True, "env_admission_timeout", 0, "admission", server=url)
        r.raise_for_status()
        seed = r.json()
        tools = []
        for t in seed.get("tools", []):
            app, _, name = str(t["name"]).partition("__")
            tools.append(ToolSpec(app=app, name=name, description=t.get("description", ""), input_schema=t.get("parameters") or {"type": "object"}))
        obs = Observation(actor=actor, step=0, instruction=seed.get("instruction") or None, tools=tools)
        steps, stop = 0, "max_steps"
        while steps < max_steps:
            a = await policy.act(obs)
            steps += 1
            if isinstance(a, Done):
                stop = "agent_done"
                break
            results = []
            if isinstance(a, ToolCall):
                try:
                    tr = await c.post(f"{url}/{a.function_name.replace('.', '__', 1)}", json=a.arguments, headers=hdr)
                except httpx.HTTPError as e:
                    return RemoteEpisode(None, True, "env_unreachable", steps, f"tool call failed: {e}", server=url)
                if tr.status_code == 404:  # reaped or lost on the server: infrastructure, masked
                    return RemoteEpisode(None, True, "env_session_lost", steps, "session lost", server=url)
                tr.raise_for_status()
                d = tr.json()
                results.append(ToolResult(tool_call_id=a.tool_call_id, function_name=a.function_name, ok=bool(d.get("ok", True)), content=str(d.get("output", ""))))
            elif isinstance(a, Message):
                pass  # no peers on a resources server: the message is acknowledged by the policy itself
            obs = Observation(actor=actor, step=steps, tool_results=results)
        v = await c.post(url + "/verify", json={"resources_session_id": sid, "responses_create_params": {}, "response": {}}, headers=hdr)
        v.raise_for_status()
        doc = v.json()
        masked = bool(doc.get("mask_sample"))
        return RemoteEpisode(None if masked else float(doc["reward"]), masked, doc.get("failure_kind") if masked else None, steps, stop, doc, url)
    finally:
        try:
            await c.post(url + "/close_session", json={"resources_session_id": sid}, headers=hdr)
        except Exception:  # noqa: BLE001, S110 -- the server's reaper frees it otherwise
            pass
        if own:
            await c.aclose()


def remote_world_rollout(urls: list[str], task_for: Any, codec: Any, *, max_steps: int = 30, max_tokens: int = 512, context: Any = None) -> Any:
    """``task_for(prompt) -> task_id`` on the servers; the runtime's ``rollout_fn``."""
    import httpx

    from crucible.actors.policy import LLMToolPolicy
    from crucible.core.schema.run import RunRecord
    from crucible.runtime.record import build
    from crucible.train.world_env import EngineLLM

    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0), limits=httpx.Limits(max_connections=512))

    async def rollout(rid: str, prompt: str, seed: int, gen: Any, *, temperature: float = 1.0, harness: str = "nemo-gym:remote@1"):
        tid = task_for(prompt)
        pol = LLMToolPolicy("agent", EngineLLM(codec, gen), max_tokens=max_tokens, temperature=temperature, context=context)
        t0 = time.monotonic()
        ep = await remote_episode(urls, tid, pol, rid=rid, max_steps=max_steps, client=client)
        if ep.masked:
            return None
        rec = RunRecord(run_id=rid, task=tid, world=ep.server, policy="engine", started_at="", steps=ep.steps, outcome="completed", reward=ep.reward,
                        attempt=f"{rid}-a1", meta={"harness": harness, "stop": ep.stop, "truncated": ep.stop == "max_steps",
                                                   "gates": {k[5:]: bool(v) for k, v in ep.verify.items() if k.startswith("gate/")}})
        ro = build(rec, calls={"agent": pol.model_calls}, segments={"agent": list(pol.segments)} if pol.segments else None,
                   sampling={"temperature": temperature})
        ro.group, ro.timings["remote_s"] = None, time.monotonic() - t0
        return ro

    return rollout
