"""Crucible tasks as a NeMo-Gym resources server (export; research m11-m12/05 §3.4). Protocol-faithful to
``nemo_gym/base_resources_server.py`` (NeMo-Gym needs Python ≥ 3.13, so this module does not import it; the gate
validates its responses against NeMo-Gym's own pydantic models in a 3.13 environment).

Routes:
- ``POST /seed_session`` ``{resources_session_id, episode_id?, task_id, task_data?}`` (or the legacy agent's dataset row: the
  session id is minted, the task comes from ``responses_create_params.metadata.task_id`` or is the only one served) →
  starts a World session and returns exactly ``ResourcesSeedSessionResponse`` ``{resources_session_id}``, with the
  ``resources_session_id`` cookie. Idempotent: a retried seed of a live id returns it (another task or episode: 409).
  ``GET /crucible/session`` gives Crucible's client the tools and the instruction;
- ``POST /{tool}`` (the Responses agent's ``function_call`` dispatch, session in the ``resources_session_id`` cookie or the
  ``X-Resources-Session-Id`` header) → the World tool's result;
- ``POST /verify`` ``{responses_create_params, response, resources_session_id?}`` → graded **from the World's state**
  (never from the transcript, N6): ``BaseMultiRewardVerifyResponse``-shaped
  ``{reward, reward_components, mask_sample, failure_kind?, failure_reason?, instance_config: {mask_sample}, gate/<g>,
  crucible_reward, crucible_steps}``. A masked grade has ``mask_sample = true`` at both places NeMo-RL reads it; its
  ``reward`` is 0.0 only because the schema requires a float (NeMo-RL drops masked samples), never a scored 0;
- ``POST /close_session``, ``GET /reverify_mode``, ``POST /aggregate_metrics``, ``GET /health``; ``POST /crucible/message`` and
  ``/crucible/done`` carry a Crucible client's other actions into the World.

The grade is **once per session** (M11 audit). NeMo-Gym's agent posts any function name the model emits to ``/{name}``,
so the model can reach ``/verify`` and ``/close_session``: the first grade ends the episode (later tool calls are refused),
and a close before any grade grades first, so neither can be used to search against the grader or to turn the policy's
own episode into a lost, masked session. An infrastructure fault recorded in the World during the episode masks the grade
(``World.failures``, the same rule as ``run_episode``), and a grading exception is a masked ``verifier_error``
(``judge_failsafe``). A masked sample's ``reward`` is 0.0 because the schema requires a float; NeMo-RL zeroes its loss but
counts it in the GRPO group baseline (``grpo.py:3458-3499``), so the M12 NeMo-RL bridge must fill masked rows first.

Serving at scale (m13 D-3; NeMo-Gym ``base_environment_server.py`` admission, OpenEnv ``http_server.py`` reaper):
``max_concurrent_episodes`` bounds live sessions; a ``seed_session`` that waits longer than ``queue_timeout_s`` for a slot
gets ``503 {"admission": "queue_timeout"}``, a non-terminal failure the client masks (never a scored 0); sessions idle
longer than ``idle_timeout_s`` are closed by a reaper (an abandoned client cannot leak Worlds); an unknown session is a
404, not a crash.

NeMo-Gym's session contract (M11 lab audit R-4; NeMo-Gym 4330be2 ``base_resources_server.py:167-239``,
``testing/session_conformance.py``; the module imports none of it, NeMo-Gym needs Python >= 3.13.14):

- **Typed ids.** ``single_agent_turn`` seeds with ``task_id: TaskId {taskset, task_id}`` and ``episode_id: EpisodeId
  {rollout_id, attempt}`` objects. ``str(dict)`` was looked up as a task, so the first seed of NeMo-Gym's default path
  answered 404. A task is now ``TaskId.task_id`` (within ``taskset`` when the server declares one), and an episode is
  ``EpisodeId.capture_key`` (``rollout_id``, or ``<rollout_id>-a<attempt>``), which keys the replay's call ids. Crucible's
  own client (a plain string ``task_id``, no ``episode_id``) and the legacy dataset row still seed.
- **A closed id stays closed.** A close (also of an id never seeded, and a reaped session) leaves a tombstone, and a later
  seed of that id answers 400: a seed after close used to open a new World. Tombstones live ``tombstone_s`` (default the
  longer of an hour and four idle timeouts) and at most ``max_tombstones``, the oldest dropped first.
- **Which session a request names.** On ``/seed_session`` the body's id is the one minted when the request carries no
  session (``single_agent_turn`` sends the id in the body, before any cookie exists); when it carries one (the
  ``resources_session_id`` cookie or the ``X-Resources-Session-Id`` header), a different body id is refused, so a model
  that emits ``seed_session`` with an invented id through NeMo-Gym's agent (which posts any function name with the
  session's cookie) cannot open Worlds until the slots run out. On every other route the cookie or the header wins over
  the body. A close keeps the cookie (a later ``/verify`` returns the stored grade); a seed carrying it is refused.
- **Races.** A close that arrives while a seed of the same id waits for a slot or opens its World wins: the seed is refused
  and its World closed (``single_agent_turn`` registers its cleanup close before it seeds). A second seed of an id whose
  first seed is still opening waits for it and answers as a retry: one World per id.
- The replay's episode key carries ``TaskId.taskset`` when the seed named one, so the call ids of ``crucible_steps`` derive
  from the typed ids. (A full ``single_agent_turn`` episode against the export, the recommendation's first acceptance,
  needs a policy model server and is not in the tests.)

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
             reap_every_s: float = 5.0, taskset: str | None = None, tombstone_s: float | None = None, max_tombstones: int = 100_000):
    """``tasks``: task_id -> M10 RunTask (``open(workdir)``, ``grade(world, workdir)``). ``taskset``: the ``TaskId.taskset``
    this server answers for (None: any). ``tombstone_s`` / ``max_tombstones``: how long and how many closed ids are kept
    refused (module doc)."""
    import asyncio
    import shutil
    import time
    import uuid

    from fastapi import HTTPException

    sessions: dict[str, dict[str, Any]] = {}
    finished: dict[str, tuple[float, dict]] = {}  # sessions closed after a grade: a late /verify returns that grade
    tombstones: dict[str, float] = {}  # closed ids (and closed-before-seed ones), oldest first: a later seed is refused (R-4)
    pending: dict[str, asyncio.Event] = {}  # ids whose seed is opening a World: a concurrent seed of the same id waits (R-4)
    keep_tombstones_s = float(tombstone_s) if tombstone_s is not None else max(3600.0, 4 * idle_timeout_s)
    slots = asyncio.Semaphore(max_concurrent_episodes)
    stats = {"admitted": 0, "admission_timeouts": 0, "reaped": 0, "closed": 0}

    def _tombstone(sid: str) -> None:
        tombstones.pop(sid, None)  # re-inserted: the dict stays oldest first
        tombstones[sid] = time.monotonic()
        while len(tombstones) > max_tombstones:
            tombstones.pop(next(iter(tombstones)))

    async def _close(sid: str) -> bool:
        s = sessions.pop(sid, None)
        if s is None:
            return False
        try:
            await s["world"].close()
        finally:
            slots.release()
            shutil.rmtree(s["dir"], ignore_errors=True)  # the session's World store and grade files (M11 audit H2 V18)
        return True

    async def reaper() -> None:
        while True:
            await asyncio.sleep(reap_every_s)
            now = time.monotonic()
            # a session in a tool call or a grade is never reaped mid-flight (H2 V18: a long grade was reaped and masked)
            for sid in [k for k, v in sessions.items() if now - v["used"] > idle_timeout_s and not v["lock"].locked()]:
                _tombstone(sid)  # an abandoned id is closed for good: a late seed must not reopen it
                if await _close(sid):
                    stats["reaped"] += 1
            for sid in [k for k, (t, _) in finished.items() if now - t > idle_timeout_s]:
                finished.pop(sid, None)
            while tombstones and now - next(iter(tombstones.values())) > keep_tombstones_s:
                tombstones.pop(next(iter(tombstones)))

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
    app.state.stats, app.state.sessions, app.state.tombstones = stats, sessions, tombstones

    def carried(req: Request) -> str:
        """The session the request carries (header, then cookie), or ''."""
        return req.headers.get("x-resources-session-id") or req.cookies.get("resources_session_id") or ""

    def sid_of(req: Request, body: dict | None = None) -> str:
        """Every route but the seed: the carried session wins over the body's id (module doc, M11 lab audit R-4)."""
        named = (body or {}).get("resources_session_id")
        return carried(req) or (named if isinstance(named, str) else "")

    def session(sid: str) -> dict[str, Any]:
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(status_code=404, detail=f"unknown or expired session {sid!r}")
        s["used"] = time.monotonic()
        return s

    async def _body(req: Request) -> Any:
        raw = await req.body()
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def _task_id(body: dict) -> str | None:
        """The seed's task: NeMo-Gym's typed ``TaskId`` (``task_id`` within ``taskset``, M11 lab audit R-4), a plain string
        (Crucible's own client), the legacy agent's row metadata, or the only task served (H2 V11, G3 D3: ``SimpleAgent.run``
        posts the dataset row, which has no session id and no ``task_id``)."""
        raw = body.get("task_id")
        if isinstance(raw, dict):  # TaskId {taskset, task_id}: extra keys forbidden upstream, so only these two are read
            ts, tid = raw.get("taskset"), raw.get("task_id")
            if not isinstance(tid, str) or not tid or not isinstance(ts, str) or not ts:
                raise HTTPException(status_code=400, detail="task_id must be a TaskId {taskset, task_id} of non-empty strings")
            if taskset is not None and ts != taskset:
                raise HTTPException(status_code=404, detail=f"unknown taskset {ts!r} (this server serves {taskset!r})")
            return tid
        tid = raw or ((body.get("responses_create_params") or {}).get("metadata") or {}).get("task_id")
        if tid is None and len(tasks) == 1:
            tid = next(iter(tasks))
        return str(tid) if tid is not None else None

    def _episode(body: dict) -> str | None:
        """The seed's episode key: ``EpisodeId.capture_key`` (``rollout_id``, ``<rollout_id>-a<attempt>`` after a retry;
        NeMo-Gym ``episode_types.py:27-50``), a plain string as given, or None (not named: Crucible's client, legacy rows)."""
        e = body.get("episode_id")
        if e is None:
            return None
        if isinstance(e, dict):
            rid, attempt = e.get("rollout_id"), e.get("attempt", 0)
            if not isinstance(rid, str) or not rid or isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
                raise HTTPException(status_code=400, detail="episode_id must be an EpisodeId {rollout_id, attempt >= 0}")
            return rid if attempt == 0 else f"{rid}-a{attempt}"
        if isinstance(e, str) and e:
            return e
        raise HTTPException(status_code=400, detail="episode_id must be an EpisodeId or a string")

    def _info(s: dict) -> dict:
        world = s["world"]
        return {"tools": [{"type": "function", "name": t.qualified.replace(".", "__", 1), "description": t.description, "parameters": t.input_schema}
                          for t in world.tools_for("agent")],
                "instruction": world.instructions.get("agent", "")}

    @app.get("/health")
    async def health():
        return {"ok": True, "sessions": len(sessions), "capacity": max_concurrent_episodes, "stats": dict(stats)}

    @app.post("/seed_session")
    async def seed_session(req: Request):
        body = await _body(req)
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="seed body must be a JSON object")
        named, held = body.get("resources_session_id"), carried(req)
        if named is not None and (not isinstance(named, str) or not named):
            raise HTTPException(status_code=400, detail="resources_session_id must be a non-empty string")
        if held and named and named != held:
            # a seed from inside a session for another id: a model-emitted seed_session (module doc), never a client's retry
            raise HTTPException(status_code=409, detail=f"this request carries session {held!r}; it cannot seed {named!r}")
        sid = held or named or uuid.uuid4().hex
        tid, episode = _task_id(body), _episode(body)
        raw_tid = body.get("task_id")
        taskset_named = raw_tid.get("taskset") if isinstance(raw_tid, dict) else None
        if tid not in tasks:
            raise HTTPException(status_code=404, detail=f"unknown task {tid!r}")
        while sid in pending:  # a concurrent seed of this id is opening its World: wait for it, then answer as a retry
            await pending[sid].wait()

        def closed() -> HTTPException:  # NeMo-Gym's contract: a seed after close (or after a close of a never-seeded id) is refused
            return HTTPException(status_code=400, detail=f"session {sid!r} was closed")

        if sid in tombstones:
            raise closed()
        if sid in sessions:  # "idempotently initialize" (NeMo-Gym): a retried seed never opens a second World (H2 V10, G3 D4)
            if sessions[sid]["task_id"] != tid or (episode is not None and sessions[sid]["episode"] != episode):
                raise HTTPException(status_code=409, detail=f"session {sid!r} is live for another task or episode")
            resp = JSONResponse({"resources_session_id": sid})
            resp.set_cookie("resources_session_id", sid)
            return resp
        task = tasks[tid]
        pending[sid] = asyncio.Event()
        try:
            try:
                await asyncio.wait_for(slots.acquire(), timeout=queue_timeout_s)
            except TimeoutError:
                stats["admission_timeouts"] += 1
                return JSONResponse({"admission": "queue_timeout", "detail": f"no episode slot within {queue_timeout_s}s"}, status_code=503)
            d = None
            try:
                if sid in tombstones:  # closed while this seed waited for a slot (single_agent_turn's cleanup close): never opened
                    raise closed()
                d = Path(tempfile.mkdtemp(prefix="nemo-sess-"))
                world = await task.open(d / "work")
                if sid in tombstones:  # closed while the World opened: close it again here, the client has given up on it
                    try:
                        await world.close()
                    finally:
                        raise closed()
            except BaseException:
                slots.release()
                if d is not None:
                    shutil.rmtree(d, ignore_errors=True)
                raise
            stats["admitted"] += 1
            finished.pop(sid, None)
            sessions[sid] = {"task": task, "world": world, "dir": d, "task_id": tid, "calls": 0, "used": time.monotonic(), "lock": asyncio.Lock(),
                             "episode": episode or "e0", "taskset": taskset_named, "graded": None}
        finally:
            pending.pop(sid).set()
        # exactly ``ResourcesSeedSessionResponse`` (extra fields are forbidden; no MCP route is served, so no ``resources_tools``:
        # G3 D2). Crucible's own client reads the tools and the instruction from ``GET /crucible/session``.
        resp = JSONResponse({"resources_session_id": sid})
        resp.set_cookie("resources_session_id", sid)
        return resp

    @app.get("/crucible/session")
    async def session_info(req: Request):
        return _info(session(sid_of(req)))

    async def _grade(s: dict, body: dict) -> dict:
        """The verify document, graded once per session. Infra faults recorded during the episode mask it (H2 V1, G3 D1:
        ``/verify`` graded the World without reading ``World.failures``, so a crashed tool scored 0). A grading or replay
        exception masks too (NeMo-Gym ``judge_failsafe``), never a 500 after a finished episode."""
        from crucible.actors.policy import _masking_failure
        from crucible.interop.replay import from_world_trace, replay
        from crucible.verify.runtime.plan import write_reward_files

        echo = {"responses_create_params": body.get("responses_create_params") or {}, "response": body.get("response") or {}}  # BaseVerifyResponse ⊃ request
        try:
            out = s["dir"] / "verify"
            rep = await s["task"].grade(s["world"], s["dir"] / "grade")
            rj = rep.reward_json()
            write_reward_files(out, rj, rep.to_dict())
            steps_dir = s["dir"] / "steps"
            actions = from_world_trace(s["world"].to_trace())
            key = body.get("episode_key") or s["episode"]
            tid_key = getattr(s["task"], "id", s["task_id"])
            tid_key = f"{s['taskset']}:{tid_key}" if s.get("taskset") else tid_key  # the typed TaskId when the seed named one (R-4)
            await replay(s["task"], actions, steps_dir, episode_key=f"{tid_key}|{key}", harness="nemo-gym@1")
            rows = [json.loads(x) for x in (steps_dir / "steps.jsonl").read_text().splitlines() if x.strip()]
            infra = dict(rj.get("infra") or {})
            if infra.get("status") != "masked":
                fault = _masking_failure(s["world"])
                if fault is not None:
                    infra = {"status": "masked", "failure_kind": fault, "reason": "an infrastructure fault during the episode"}
            masked = infra.get("status") == "masked"
            flat = json.loads((out / "reward.json").read_text()) if (out / "reward.json").exists() else {}
            comps = {k: v for k, v in flat.items() if k != "reward"}
            crucible_reward = json.loads((out / "crucible_reward.json").read_text())
        except Exception as ex:  # noqa: BLE001 -- judge_failsafe: a grader crash is the environment's failure
            masked, comps, rows, crucible_reward = True, {}, [], {"reward": None}
            infra = {"status": "masked", "failure_kind": "verifier_error", "reason": f"{type(ex).__name__}: {ex}"[:400]}
        reward = 0.0 if masked else float(rj["reward"])
        if masked:
            crucible_reward = {**crucible_reward, "reward": None}
        # NeMo-RL requires reward == sum(reward_components) (nemo_gym.py:1483-1508; M6 audit: gate bits and progress
        # summed to 4.0 against 1.033, and NeMo-RL raised). The channels stay readable beside it, not as components.
        doc = {**echo, "reward": reward, "reward_components": {"crucible": reward}, "crucible_channels": comps, "mask_sample": masked,
               "instance_config": {"mask_sample": masked}, **({"failure_kind": infra.get("failure_kind"), "failure_reason": infra.get("reason")} if masked else {}),
               **{k: v for k, v in comps.items() if k.startswith("gate/")},
               "crucible_reward": crucible_reward, "crucible_steps": rows}
        return json.loads(jcs(doc))

    @app.post("/verify")
    async def verify(req: Request):
        body = await _body(req)
        body = body if isinstance(body, dict) else {}
        sid = sid_of(req, body)
        if sid in finished and sid not in sessions:
            return JSONResponse(finished[sid][1])
        s = session(sid)
        async with s["lock"]:
            # one grade per session: a ``verify`` the model calls as a tool (NeMo-Gym's agent posts any function name to
            # ``/{name}``) ends its own episode instead of letting it search against the grader (H2 V5)
            if s["graded"] is None:
                s["graded"] = await _grade(s, body)
            return JSONResponse(s["graded"])

    @app.post("/close_session")
    async def close_session(req: Request):
        body = await _body(req)
        body = body if isinstance(body, dict) else {}
        sid = sid_of(req, body)
        if sid:
            _tombstone(sid)  # also for an id never seeded: a racing seed must not create it later (R-4)
        s = sessions.get(sid)
        if s is not None:
            async with s["lock"]:
                if s["graded"] is None:  # closed before a grade (a model calling ``close_session``): graded now, so closing can
                    s["graded"] = await _grade(s, {})  # never turn the policy's own episode into a lost (masked) session
                finished[sid] = (time.monotonic(), s["graded"])
        ok = await _close(sid)
        stats["closed"] += int(ok)
        # ``ResourcesCloseSessionResponse`` (G3 D6). The cookie is kept: NeMo-Gym's agent merges every tool response's cookies
        # into its jar and single_agent_turn hands that jar to /verify, so a cleared cookie lost the stored grade (and freed the
        # next seed to mint a new World); the tombstone already makes a stale cookie harmless
        return {"resources_session_id": sid}

    @app.get("/reverify_mode")
    async def reverify_mode():
        return "unsupported"  # a bare ``ReverifyMode`` (G3 D5: NeMo-Gym parses ``ReverifyMode(json)``); a World grade needs live state

    @app.post("/aggregate_metrics")
    async def aggregate_metrics(req: Request):
        rows = ((await _body(req)) or {}).get("verify_responses", [])
        rs = [r["reward"] for r in rows if not r.get("mask_sample")]
        m = {"mean/reward": sum(rs) / len(rs) if rs else None, "masked": sum(1 for r in rows if r.get("mask_sample")), "n": len(rows)}
        return {"group_level_metrics": [], "agent_metrics": m, "key_metrics": {"mean/reward": m["mean/reward"]}}  # ``AggregateMetrics`` (G3 D6)

    async def _step(s: dict, action) -> Any:
        async with s["lock"]:
            if s["graded"] is not None:
                return None
            return await s["world"].step("agent", action)

    @app.post("/crucible/message")
    async def message(req: Request):
        """A Crucible client's ``Message`` (H2 V12: it was dropped, so responders never ran)."""
        from crucible.core.schema.actions import Message

        b = await _body(req)
        s = session(sid_of(req))
        b = b if isinstance(b, dict) else {}
        res = await _step(s, Message(to=str(b.get("to", "")), text=str(b.get("text", ""))))
        return {"ok": res is not None, "inbox": [m.model_dump(mode="json", by_alias=True) for m in res.observation.inbox] if res is not None else []}

    @app.post("/crucible/done")
    async def done(req: Request):
        """A Crucible client's ``Done`` (H2 V12: it was never sent, so answer-graded tasks graded nothing)."""
        from crucible.core.schema.actions import Done

        b = await _body(req)
        s = session(sid_of(req))
        b = b if isinstance(b, dict) else {}
        res = await _step(s, Done(reason=str(b.get("reason", "")), answer=b.get("answer")))
        return {"ok": res is not None}

    @app.post("/{tool}")
    async def call_tool(tool: str, req: Request):
        from crucible.core.schema.actions import ToolCall

        s = session(sid_of(req))
        args = await _body(req)
        if not isinstance(args, dict):  # malformed arguments are the policy's error, shown to it (H2 V11: a 500)
            return JSONResponse({"output": "invalid arguments: a JSON object is required", "ok": False})
        s["calls"] += 1
        known = [t.qualified for t in s["world"].tools_for("agent")]
        res = await _step(s, ToolCall(tool_call_id=f"ng{s['calls']}", function_name=canonical_tool(tool, known=known), arguments=args))
        if res is None:
            return JSONResponse({"output": "the episode is over (graded)", "ok": False})
        tr = res.observation.tool_results[0] if res.observation.tool_results else None
        # ``fault`` lets a client stop at an infra fault; the grade masks the episode either way (H2 V1)
        return JSONResponse({"output": tr.content if tr else "", "ok": tr.ok if tr else False, **({"fault": tr.fault} if tr is not None and tr.fault else {})})

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
    ap.add_argument("--taskset", default=None, help="the TaskId.taskset this server answers for (default: any)")
    ap.add_argument("--tombstone-s", dest="tombstone_s", type=float, default=None, help="how long a closed id stays refused")
    ap.add_argument("--max-tombstones", dest="max_tombstones", type=int, default=100_000)
    a = ap.parse_args()
    mod, _, attr = a.tasks.partition(":")
    tasks = getattr(importlib.import_module(mod), attr)()
    uvicorn.run(make_app(tasks, max_concurrent_episodes=a.max_concurrent, queue_timeout_s=a.queue_timeout, idle_timeout_s=a.idle_timeout,
                         reap_every_s=a.reap_every, taskset=a.taskset, tombstone_s=a.tombstone_s, max_tombstones=a.max_tombstones),
                host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
