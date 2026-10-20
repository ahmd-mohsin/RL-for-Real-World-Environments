"""M11 lab audit R-4: the NeMo-Gym export accepts typed sessions and refuses a seed after close.

NeMo-Gym's ``single_agent_turn`` seeds with ``TaskId {taskset, task_id}`` and ``EpisodeId {rollout_id, attempt}`` objects
(NeMo-Gym 4330be2 ``episode_types.py:27-66``, ``base_resources_server.py:190-200``); the export looked up ``str(dict)`` and
answered 404. A seed after close (or after a close of a never-seeded id) opened a new World, against NeMo-Gym's session
contract (``testing/session_conformance.py:57-62``). The body's session id won over the cookie, so a model emitting
``seed_session`` with an invented id through NeMo-Gym's agent could open Worlds until the slots ran out.

The in-process tests need FastAPI; the conformance test runs NeMo-Gym's own checker in ``crucible/.envs/nemo``."""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from crucible_pack_toy.runtime import HelpdeskTask  # noqa: E402
from crucible_pack_toy.verify import GOLD  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from crucible.interop.nemo_gym.server import make_app  # noqa: E402

CRUCIBLE = Path(__file__).resolve().parents[2]
TYPED = {"resources_session_id": "c1", "episode_id": {"rollout_id": "r1", "attempt": 0}, "task_id": {"taskset": "crucible", "task_id": "t"},
         "task_data": {}}


def _client(**kw):
    app = make_app({"t": HelpdeskTask()}, **kw)
    return app, TestClient(app)


def test_a_typed_seed_opens_the_task_and_keys_the_episode_by_capture_key():
    app, c = _client()
    r = c.post("/seed_session", json=TYPED)
    assert r.status_code == 200, r.text
    assert r.json() == {"resources_session_id": "c1"} and r.cookies.get("resources_session_id") == "c1"  # single_agent_turn needs both
    assert app.state.sessions["c1"]["task_id"] == "t" and app.state.sessions["c1"]["episode"] == "r1"
    _, c2 = _client()
    r = c2.post("/seed_session", json={**TYPED, "resources_session_id": "c2", "episode_id": {"rollout_id": "r1", "attempt": 2}})
    assert r.status_code == 200 and c2.app.state.sessions["c2"]["episode"] == "r1-a2"  # EpisodeId.capture_key


def test_the_episode_key_reaches_the_replays_call_ids():
    def steps(rollout_id: str) -> list[str]:
        _, c = _client()
        c.post("/seed_session", json={**TYPED, "episode_id": {"rollout_id": rollout_id, "attempt": 0}})
        for a in GOLD[:2]:
            c.post("/" + a.function_name.replace(".", "__", 1), json=a.arguments)
        return [row["call_id"] for row in c.post("/verify", json={}).json()["crucible_steps"]]

    a, b = steps("r1"), steps("r2")
    assert a and len(a) == len(b) and set(a).isdisjoint(b)  # another episode, other call ids


def test_a_typed_seed_is_checked():
    _, c = _client(taskset="crucible")
    assert c.post("/seed_session", json={**TYPED, "task_id": {"taskset": "other", "task_id": "t"}}).status_code == 404
    for bad in ({"taskset": "crucible"}, {"taskset": "crucible", "task_id": ""}, {"task_id": "t"}):
        assert c.post("/seed_session", json={**TYPED, "task_id": bad}).status_code == 400, bad
    for bad in ({"rollout_id": ""}, {"rollout_id": "r", "attempt": -1}, {"rollout_id": "r", "attempt": True}, 7):
        assert c.post("/seed_session", json={**TYPED, "episode_id": bad}).status_code == 400, bad
    assert c.post("/seed_session", json=TYPED).status_code == 200


def test_a_retried_seed_for_another_episode_is_refused():
    app, c = _client()
    c.post("/seed_session", json=TYPED)
    assert c.post("/seed_session", json=TYPED).status_code == 200 and app.state.stats["admitted"] == 1
    assert c.post("/seed_session", json={**TYPED, "episode_id": {"rollout_id": "r1", "attempt": 1}}).status_code == 409


def test_a_seed_after_close_and_after_a_close_of_a_never_seeded_id_is_refused():
    app, c = _client()
    c.post("/seed_session", json=TYPED)
    r = c.post("/close_session", json={"resources_session_id": "c1", "episode_id": TYPED["episode_id"]})
    assert r.status_code == 200 and r.json() == {"resources_session_id": "c1"}
    assert c.post("/seed_session", json=TYPED).status_code == 400  # the cookie is kept, and its id is closed
    other = TestClient(app)  # an Environment Server closing an id it never seeded (no cookie)
    assert other.post("/close_session", json={"resources_session_id": "never"}).status_code == 200
    r = other.post("/seed_session", json={**TYPED, "resources_session_id": "never"})
    assert r.status_code == 400 and "closed" in r.text
    assert app.state.stats["admitted"] == 1 and not app.state.sessions


def test_a_close_the_model_calls_keeps_the_grade_for_the_agents_verify():
    """NeMo-Gym's agent merges each tool response's cookies into its jar and single_agent_turn hands that jar to /verify:
    a close that cleared the cookie lost the stored grade (404) and let the next seed mint a fresh World."""
    app, c = _client()
    jar = dict(c.post("/seed_session", json=TYPED).cookies)
    agent = TestClient(app)  # its own jar is never used: the agent's dict jar travels as a Cookie header, merged after each reply

    def post(path: str, body: dict):
        r = agent.post(path, json=body, headers={"cookie": "; ".join(f"{k}={v}" for k, v in jar.items())})
        jar.update(r.cookies)
        return r

    for a in GOLD:
        post("/" + a.function_name.replace(".", "__", 1), a.arguments)
    post("/close_session", {})  # the model emits close_session as a tool
    assert jar == {"resources_session_id": "c1"}
    v = post("/verify", {"responses_create_params": {}, "response": {}})
    assert v.status_code == 200 and v.json()["reward"] == 1.0 and v.json()["mask_sample"] is False
    r = post("/seed_session", {**TYPED, "resources_session_id": "fresh"})
    assert r.status_code == 409 and app.state.stats["admitted"] == 1  # no fresh World to search against the grader


def _slow_task(delay: float):
    class Slow(HelpdeskTask):
        async def open(self, workdir):
            await asyncio.sleep(delay)
            return await super().open(workdir)

    return Slow()


def test_a_close_that_races_a_seed_wins_and_concurrent_seeds_open_one_world():
    import httpx

    async def go():
        app = make_app({"t": _slow_task(0.5)})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ng") as c:
            seed = asyncio.create_task(c.post("/seed_session", json=TYPED))
            await asyncio.sleep(0.1)  # the seed is opening its World
            close = await c.post("/close_session", json={"resources_session_id": "c1"})
            r = await seed
            assert close.status_code == 200 and r.status_code == 400 and not app.state.sessions and app.state.stats["admitted"] == 0
            body = {**TYPED, "resources_session_id": "c2"}
            a, b = await asyncio.gather(c.post("/seed_session", json=body), c.post("/seed_session", json=body))
            assert a.status_code == b.status_code == 200 and app.state.stats["admitted"] == 1 and list(app.state.sessions) == ["c2"]
            for sid in list(app.state.sessions):
                await app.state.sessions[sid]["world"].close()

    asyncio.run(go())


def test_the_tombstones_are_bounded_in_number_and_in_time():
    app, c = _client(max_tombstones=2)
    for sid in ("a", "b", "c"):
        c.post("/close_session", json={"resources_session_id": sid})
    assert list(app.state.tombstones) == ["b", "c"]  # the oldest is dropped first
    assert c.post("/seed_session", json={**TYPED, "resources_session_id": "a"}).status_code == 200
    app, _ = _client(tombstone_s=0.2, reap_every_s=0.05)
    with TestClient(app) as c:  # the lifespan runs the reaper
        c.post("/close_session", json={"resources_session_id": "x"})
        assert "x" in app.state.tombstones
        time.sleep(0.6)
        assert "x" not in app.state.tombstones


def test_a_seed_from_inside_a_session_cannot_open_another_world():
    """A model that emits ``seed_session`` with an invented id: NeMo-Gym's agent posts it with the session's cookie."""
    app, c = _client()
    c.post("/seed_session", json=TYPED)  # the cookie is now c1's
    r = c.post("/seed_session", json={**TYPED, "resources_session_id": "invented"})
    assert r.status_code == 409 and set(app.state.sessions) == {"c1"}
    r = c.post("/seed_session", json={k: v for k, v in TYPED.items() if k != "resources_session_id"})  # no id: the carried session
    assert r.status_code == 200 and r.json() == {"resources_session_id": "c1"} and app.state.stats["admitted"] == 1
    sibling = TestClient(app)  # another rollout: its own client, no cookie
    assert sibling.post("/seed_session", json={**TYPED, "resources_session_id": "c2"}).status_code == 200


def test_the_carried_session_wins_over_the_body_on_every_other_route():
    app, c = _client()
    c.post("/seed_session", json=TYPED)
    TestClient(app).post("/seed_session", json={**TYPED, "resources_session_id": "victim"})
    c.post("/close_session", json={"resources_session_id": "victim"})  # names another session; the cookie's is closed
    assert "victim" in app.state.sessions and "c1" not in app.state.sessions


def test_crucibles_own_string_client_still_seeds():
    app, c = _client()
    h = {"X-Resources-Session-Id": "mine"}
    r = c.post("/seed_session", json={"resources_session_id": "mine", "task_id": "t"}, headers=h)
    assert r.status_code == 200 and app.state.sessions["mine"]["episode"] == "e0"
    r = c.post("/seed_session", json={"resources_session_id": "next", "task_id": "t"}, headers={"X-Resources-Session-Id": "next"})
    assert r.status_code == 200  # a shared client's stale cookie does not matter: the header names the session


def test_nemo_gyms_own_session_contract_passes():
    """NeMo-Gym's checker over the export's routes, with its typed seed (in ``crucible/.envs/nemo``, Python >= 3.13.14)."""
    py = CRUCIBLE / ".envs" / "nemo" / "bin" / "python"
    if not py.exists():
        pytest.skip("NeMo-Gym env not built (crucible/.envs/nemo)")
    code = """
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.testing.session_conformance import check_resources_session_contract
from crucible_pack_toy.runtime import HelpdeskTask
from crucible.interop.nemo_gym.server import make_app
seed = ResourcesSeedSessionRequest(resources_session_id="conf-1", episode_id=EpisodeId(rollout_id="r1"),
                                   task_id=TaskId(taskset="crucible", task_id="toy/helpdesk"), task_data={})
check_resources_session_contract(make_app({"toy/helpdesk": HelpdeskTask()}, taskset="crucible"), seed, keeps_state=True)
print("CONTRACT OK")
"""
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, check=False, timeout=600, cwd=CRUCIBLE)
    if r.returncode != 0 and ("No module named 'crucible'" in r.stderr or "No module named 'crucible." in r.stderr):
        pytest.skip("crucible is not installed in crucible/.envs/nemo")
    assert r.returncode == 0 and "CONTRACT OK" in r.stdout, r.stderr[-3000:]
