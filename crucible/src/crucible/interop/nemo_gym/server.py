"""Crucible tasks as a NeMo-Gym resources server (export; research m11-m12/05 §3.4). Protocol-faithful to
``nemo_gym/base_resources_server.py`` (NeMo-Gym needs Python ≥ 3.13, so this module does not import it; the gate
validates its responses against NeMo-Gym's own pydantic models in a 3.13 environment).

Routes:
- ``POST /seed_session`` ``{resources_session_id, episode_id?, task_id, task_data?}`` → starts a World session for the task
  and returns ``{resources_session_id, resources_tools: {server_name, url_path, transport}, tools: [...]}``;
- ``POST /{tool}`` (the Responses agent's ``function_call`` dispatch, session in the ``resources_session_id`` cookie or the
  ``X-Resources-Session-Id`` header) → the World tool's result;
- ``POST /verify`` ``{responses_create_params, response, resources_session_id?}`` → graded **from the World's state**
  (never from the transcript, N6): ``BaseMultiRewardVerifyResponse``-shaped
  ``{reward, reward_components, mask_sample, failure_kind?, failure_reason?, instance_config: {mask_sample}, gate/<g>,
  crucible_reward, crucible_steps}``. A masked grade has ``mask_sample = true`` at both places NeMo-RL reads it; its
  ``reward`` is 0.0 only because the schema requires a float (NeMo-RL drops masked samples), never a scored 0;
- ``POST /close_session``, ``GET /reverify_mode``, ``POST /aggregate_metrics``, ``GET /health``.

Serving at scale (m13 D-3; NeMo-Gym ``base_environment_server.py`` admission, OpenEnv ``http_server.py`` reaper):
``max_concurrent_episodes`` bounds live sessions; a ``seed_session`` that waits longer than ``queue_timeout_s`` for a slot
gets ``503 {"admission": "queue_timeout"}``, a non-terminal failure the client masks (never a scored 0); sessions idle
longer than ``idle_timeout_s`` are closed by a reaper (an abandoned client cannot leak Worlds); an unknown session is a
404, not a crash.

``python -m crucible.interop.nemo_gym.server --tasks module:factory --port P`` serves the tasks a factory returns
(``{task_id: RunTask}``).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request  # module level: FastAPI resolves the (postponed) annotations from module globals
from fastapi.responses import JSONResponse

from crucible.interop.canonical import canonical_tool, jcs


def make_app(tasks: dict[str, Any], *, max_concurrent_episodes: int = 64, queue_timeout_s: float = 30.0, idle_timeout_s: float = 600.0,
             reap_every_s: float = 5.0):
    """``tasks``: task_id -> M10 RunTask (``open(workdir)``, ``grade(world, workdir)``)."""
    import asyncio
    import time

    from fastapi import HTTPException

    sessions: dict[str, dict[str, Any]] = {}
    slots = asyncio.Semaphore(max_concurrent_episodes)
    stats = {"admitted": 0, "admission_timeouts": 0, "reaped": 0, "closed": 0}

    async def _close(sid: str) -> bool:
        s = sessions.pop(sid, None)
        if s is None:
            return False
        try:
            await s["world"].close()
        finally:
            slots.release()
        return True

    async def reaper() -> None:
        while True:
            await asyncio.sleep(reap_every_s)
            now = time.monotonic()
            for sid in [k for k, v in sessions.items() if now - v["used"] > idle_timeout_s]:
                if await _close(sid):
                    stats["reaped"] += 1

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        t = asyncio.create_task(reaper())
        try:
            yield
        finally:
            t.cancel()
            for sid in list(sessions):
                await _close(sid)

    app = FastAPI(title="crucible-nemo-gym", lifespan=lifespan)
    app.state.stats, app.state.sessions = stats, sessions

    def sid_of(req: Request, body: dict | None = None) -> str:
        return (body or {}).get("resources_session_id") or req.headers.get("x-resources-session-id") or req.cookies.get("resources_session_id") or ""

    def session(sid: str) -> dict[str, Any]:
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(status_code=404, detail=f"unknown or expired session {sid!r}")
        s["used"] = time.monotonic()
        return s

    @app.get("/health")
    async def health():
        return {"ok": True, "sessions": len(sessions), "capacity": max_concurrent_episodes, "stats": dict(stats)}

    @app.post("/seed_session")
    async def seed_session(req: Request):
        body = await req.json()
        sid, tid = body["resources_session_id"], body["task_id"]
        task = tasks[tid]
        try:
            await asyncio.wait_for(slots.acquire(), timeout=queue_timeout_s)
        except TimeoutError:
            stats["admission_timeouts"] += 1
            return JSONResponse({"admission": "queue_timeout", "detail": f"no episode slot within {queue_timeout_s}s"}, status_code=503)
        try:
            d = Path(tempfile.mkdtemp(prefix="nemo-sess-"))
            world = await task.open(d / "work")
        except BaseException:
            slots.release()
            raise
        stats["admitted"] += 1
        sessions[sid] = {"task": task, "world": world, "dir": d, "task_id": tid, "calls": 0, "used": time.monotonic()}
        tools = [{"type": "function", "name": t.qualified.replace(".", "__", 1), "description": t.description, "parameters": t.input_schema}
                 for t in world.tools_for("agent")]
        resp = JSONResponse({"resources_session_id": sid, "resources_tools": {"server_name": "crucible", "url_path": "/", "transport": "http"}, "tools": tools,
                             "instruction": world.instructions.get("agent", "")})
        resp.set_cookie("resources_session_id", sid)
        return resp

    @app.post("/verify")
    async def verify(req: Request):
        body = await req.json()
        s = session(sid_of(req, body))
        from crucible.interop.replay import from_world_trace, replay

        out = s["dir"] / "verify"
        rep = await s["task"].grade(s["world"], s["dir"] / "grade")
        rj = rep.reward_json()
        from crucible.verify.runtime.plan import write_reward_files

        write_reward_files(out, rj, rep.to_dict())
        steps_dir = s["dir"] / "steps"
        actions = from_world_trace(s["world"].to_trace())
        await replay(s["task"], actions, steps_dir, episode_key=f"{getattr(s['task'], 'id', s['task_id'])}|{body.get('episode_key', 'e0')}", harness="nemo-gym@1")
        rows = [json.loads(x) for x in (steps_dir / "steps.jsonl").read_text().splitlines() if x.strip()]
        masked = (rj.get("infra") or {}).get("status") == "masked"
        flat = json.loads((out / "reward.json").read_text()) if (out / "reward.json").exists() else {}
        comps = {k: v for k, v in flat.items() if k != "reward"}
        doc = {"responses_create_params": body.get("responses_create_params") or {}, "response": body.get("response") or {},  # echoed (BaseVerifyResponse ⊃ request)
               "reward": 0.0 if masked else float(rj["reward"]), "reward_components": comps, "mask_sample": masked,
               "instance_config": {"mask_sample": masked}, **({"failure_kind": rj["infra"].get("failure_kind"), "failure_reason": rj["infra"].get("reason")} if masked else {}),
               **{k: v for k, v in comps.items() if k.startswith("gate/")},
               "crucible_reward": json.loads((out / "crucible_reward.json").read_text()), "crucible_steps": rows}
        return JSONResponse(json.loads(jcs(doc)))

    @app.post("/close_session")
    async def close_session(req: Request):
        body = await req.json()
        ok = await _close(sid_of(req, body))
        stats["closed"] += int(ok)
        return {"closed": ok}

    @app.get("/reverify_mode")
    async def reverify_mode():
        return {"reverify_mode": False}

    @app.post("/aggregate_metrics")
    async def aggregate_metrics(req: Request):
        rows = (await req.json()).get("verify_responses", [])
        rs = [r["reward"] for r in rows if not r.get("mask_sample")]
        return {"mean_reward": sum(rs) / len(rs) if rs else None, "masked": sum(1 for r in rows if r.get("mask_sample")), "n": len(rows)}

    @app.post("/{tool}")
    async def call_tool(tool: str, req: Request):
        from crucible.core.schema.actions import ToolCall

        s = session(sid_of(req))
        args = await req.json() if (await req.body()) else {}
        s["calls"] += 1
        known = [t.qualified for t in s["world"].tools_for("agent")]
        res = await s["world"].step("agent", ToolCall(tool_call_id=f"ng{s['calls']}", function_name=canonical_tool(tool, known=known), arguments=args))
        tr = res.observation.tool_results[0] if res.observation.tool_results else None
        return JSONResponse({"output": tr.content if tr else "", "ok": tr.ok if tr else False})

    return app


def main() -> None:  # pragma: no cover - a server process
    import argparse
    import importlib

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="module:factory returning {task_id: RunTask}")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-concurrent", dest="max_concurrent", type=int, default=64)
    ap.add_argument("--queue-timeout", dest="queue_timeout", type=float, default=30.0)
    ap.add_argument("--idle-timeout", dest="idle_timeout", type=float, default=600.0)
    ap.add_argument("--reap-every", dest="reap_every", type=float, default=5.0)
    a = ap.parse_args()
    mod, _, attr = a.tasks.partition(":")
    tasks = getattr(importlib.import_module(mod), attr)()
    uvicorn.run(make_app(tasks, max_concurrent_episodes=a.max_concurrent, queue_timeout_s=a.queue_timeout, idle_timeout_s=a.idle_timeout,
                         reap_every_s=a.reap_every),
                host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
