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
- ``POST /close_session``, ``GET /reverify_mode``, ``POST /aggregate_metrics``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request  # module level: FastAPI resolves the (postponed) annotations from module globals
from fastapi.responses import JSONResponse

from crucible.interop.canonical import canonical_tool, jcs


def make_app(tasks: dict[str, Any]):
    """``tasks``: task_id -> M10 RunTask (``open(workdir)``, ``grade(world, workdir)``)."""
    app = FastAPI(title="crucible-nemo-gym")
    sessions: dict[str, dict[str, Any]] = {}

    def sid_of(req: Request, body: dict | None = None) -> str:
        return (body or {}).get("resources_session_id") or req.headers.get("x-resources-session-id") or req.cookies.get("resources_session_id") or ""

    @app.post("/seed_session")
    async def seed_session(req: Request):
        body = await req.json()
        sid, tid = body["resources_session_id"], body["task_id"]
        task = tasks[tid]
        d = Path(tempfile.mkdtemp(prefix="nemo-sess-"))
        world = await task.open(d / "work")
        sessions[sid] = {"task": task, "world": world, "dir": d, "task_id": tid, "calls": 0}
        tools = [{"type": "function", "name": t.qualified.replace(".", "__", 1), "description": t.description, "parameters": t.input_schema}
                 for t in world.tools_for("agent")]
        resp = JSONResponse({"resources_session_id": sid, "resources_tools": {"server_name": "crucible", "url_path": "/", "transport": "http"}, "tools": tools,
                             "instruction": world.instructions.get("agent", "")})
        resp.set_cookie("resources_session_id", sid)
        return resp

    @app.post("/verify")
    async def verify(req: Request):
        body = await req.json()
        s = sessions[sid_of(req, body)]
        from crucible.interop.replay import from_world_trace, replay

        out = s["dir"] / "verify"
        rep = await s["task"].grade(s["world"], s["dir"] / "grade")
        rj = rep.reward_json()
        from crucible.verify.runtime.plan import write_reward_files

        write_reward_files(out, rj, rep.to_dict())
        steps_dir = s["dir"] / "steps"
        actions = from_world_trace(s["world"].to_trace())
        await replay(s["task"], actions, steps_dir, episode_key=f"{s['task_id']}|{body.get('episode_key', 'e0')}", harness="nemo-gym@1")
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
        s = sessions.pop(sid_of(req, body), None)
        if s:
            await s["world"].close()
        return {"closed": bool(s)}

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

        s = sessions[sid_of(req)]
        args = await req.json() if (await req.body()) else {}
        s["calls"] += 1
        known = [t.qualified for t in s["world"].tools_for("agent")]
        res = await s["world"].step("agent", ToolCall(tool_call_id=f"ng{s['calls']}", function_name=canonical_tool(tool, known=known), arguments=args))
        tr = res.observation.tool_results[0] if res.observation.tool_results else None
        return JSONResponse({"output": tr.content if tr else "", "ok": tr.ok if tr else False})

    return app
