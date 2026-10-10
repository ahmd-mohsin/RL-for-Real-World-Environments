"""M11 exit gate (toy part): the same task and the same trajectory give identical canonical reward.json and steps.jsonl
through every runtime path — native World, Harbor (ATIF transcript), NeMo-Gym (resources server over HTTP) — and the
NeMo-Gym tool transport reproduces the native final state (research m11-m12/05 §2)."""

from __future__ import annotations

import json
from pathlib import Path

from crucible_pack_toy.runtime import HelpdeskTask
from crucible_pack_toy.verify import GOLD

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.interop.canonical import jcs, reward_doc, steps_rows
from crucible.interop.harbor.atif import episode_to_atif
from crucible.interop.replay import from_atif, from_responses, from_world_trace, replay

KEY = "toy/helpdesk|e0"


async def native(tmp: Path):
    t = HelpdeskTask()
    w = await t.open(tmp / "w")
    ep = await run_episode(w, {"agent": ScriptedPolicy("agent", list(GOLD))}, limits=EpisodeLimits(max_steps=50), reset=False)
    trace = w.to_trace()
    known = [x.qualified for x in w.tools_for("agent")]
    await w.close()
    out = tmp / "native"
    info = await replay(HelpdeskTask(), from_world_trace(trace), out, episode_key=KEY)
    return trace, known, out, info, ep


async def test_toy_same_reward_and_steps_across_runtimes(tmp_path):
    trace, known, nat, info, _ = await native(tmp_path)
    assert info["reward"] == 1.0 and info["steps"] == len(GOLD) + 1  # every action + the final done

    # Harbor: the transcript as ATIF (what Harbor trials record), normalized and replayed
    atif = episode_to_atif(trace, "agent")
    hb = tmp_path / "harbor"
    await replay(HelpdeskTask(), from_atif(atif, known=known), hb, episode_key=KEY)

    # NeMo-Gym: the resources server over HTTP; the Responses agent dispatches POST /<app>__<tool>
    from fastapi.testclient import TestClient

    from crucible.interop.nemo_gym.server import make_app

    client = TestClient(make_app({"toy/helpdesk": HelpdeskTask()}))
    r = client.post("/seed_session", json={"resources_session_id": "s1", "task_id": "toy/helpdesk"})
    assert r.status_code == 200 and any(t["name"] == "tickets__issue_refund" for t in r.json()["tools"])
    items = []
    for a in GOLD:
        name = a.function_name.replace(".", "__", 1)
        out = client.post(f"/{name}", json=a.arguments, headers={"X-Resources-Session-Id": "s1"})
        assert out.status_code == 200 and out.json()["ok"]
        items.append({"type": "function_call", "name": name, "arguments": json.dumps(a.arguments)})
    v = client.post("/verify", json={"resources_session_id": "s1", "responses_create_params": {}, "response": {"output": items}, "episode_key": "e0"}).json()
    client.post("/close_session", json={"resources_session_id": "s1"})
    ng = tmp_path / "nemo"
    await replay(HelpdeskTask(), from_responses(items, known=known), ng, episode_key=KEY)

    ref_r, ref_s = reward_doc(nat / "reward.json"), steps_rows(nat / "steps.jsonl")
    for d in (hb, ng):
        assert jcs(reward_doc(d / "reward.json")) == jcs(ref_r)
        assert jcs([{k: v for k, v in r.items() if k != "harness"} for r in steps_rows(d / "steps.jsonl")]) == \
            jcs([{k: v for k, v in r.items() if k != "harness"} for r in ref_s])
    # the server graded its own live World session: same reward, same components, same dense rows, not masked
    assert v["reward"] == ref_r["reward"] and v["mask_sample"] is False and v["instance_config"]["mask_sample"] is False
    assert jcs({k: v["reward_components"][k] for k in v["reward_components"]}) == jcs({k: x for k, x in ref_r.items() if k != "reward"})
    assert [r["phi"] for r in v["crucible_steps"]] == [r["phi"] for r in ref_s] and [r["call_id"] for r in v["crucible_steps"]] == [r["call_id"] for r in ref_s]


async def test_masked_grade_travels_as_mask_never_zero(tmp_path):
    from fastapi.testclient import TestClient

    from crucible.interop.nemo_gym.server import make_app
    from crucible.verify.runtime.plan import Check, Context, Gate, Plan
    from crucible.verify.runtime.trusted import InfraError

    class Broken(HelpdeskTask):
        async def grade(self, world, workdir):
            def boom(_):
                raise InfraError("grader host lost")

            return await Plan("x", [Gate("g")], [Check("g", "g", boom)]).arun(Context(__import__("crucible.verify.runtime.trusted", fromlist=["Trusted"]).Trusted(Path(workdir))))

    client = TestClient(make_app({"t": Broken()}))
    client.post("/seed_session", json={"resources_session_id": "m", "task_id": "t"})
    v = client.post("/verify", json={"resources_session_id": "m", "responses_create_params": {}, "response": {"output": []}}).json()
    assert v["mask_sample"] is True and v["instance_config"]["mask_sample"] is True and v["crucible_reward"]["reward"] is None


def test_nemo_responses_validate_against_nemo_gyms_own_models(tmp_path):
    """Our server's responses parse with NeMo-Gym's pydantic models (in its own Python 3.13 env: crucible/.envs/nemo)."""
    import subprocess
    import sys

    import pytest
    from fastapi.testclient import TestClient

    from crucible.interop.nemo_gym.server import make_app

    py = Path(__file__).resolve().parents[2] / ".envs" / "nemo" / "bin" / "python"
    if not py.exists():
        pytest.skip("NeMo-Gym env not built (crucible/.envs/nemo)")
    c = TestClient(make_app({"toy/helpdesk": HelpdeskTask()}))
    seed = c.post("/seed_session", json={"resources_session_id": "v", "task_id": "toy/helpdesk"}).json()
    rcp = {"input": [{"role": "user", "content": "work the tickets"}]}
    resp = {"id": "resp_1", "created_at": 0.0, "model": "m", "object": "response", "output": [], "parallel_tool_calls": False, "tool_choice": "auto", "tools": []}
    ver = c.post("/verify", json={"resources_session_id": "v", "responses_create_params": rcp, "response": resp}).json()
    (tmp_path / "seed.json").write_text(json.dumps(seed))
    (tmp_path / "verify.json").write_text(json.dumps(ver))
    code = ("import json,sys\nfrom nemo_gym.base_resources_server import BaseMultiRewardVerifyResponse as V\n"
            f"v=V.model_validate(json.load(open('{tmp_path}/verify.json')))\nprint(v.reward, v.mask_sample, sorted(v.reward_components)[:3])")
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[1] == "False"
    _ = sys


def _run_env(env: str, code: str) -> str:
    import subprocess

    import pytest

    py = Path(__file__).resolve().parents[2] / ".envs" / env / "bin" / "python"
    if not py.exists():
        pytest.skip(f"{env} env not built (crucible/.envs/{env})")
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, check=False, timeout=600)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-3000:])
    return r.stdout


def test_verifiers_export_reward_and_mask(tmp_path):
    """verifiers' own HarborTask machinery with Crucible's reward files: a grade returns the plan reward (+ crucible/
    metrics); a masked grade raises TaskError with the failure kind (an errored trace prime-rl drops)."""
    from crucible.verify.runtime.plan import write_reward_files

    ok, masked = tmp_path / "ok", tmp_path / "masked"
    write_reward_files(ok, {"reward": 1.0, "outcome": 1.0, "gates": {"policy": 1}, "infra": {"status": "ok"}})
    write_reward_files(masked, {"reward": 0.0, "gates": {}, "infra": {"status": "masked", "failure_kind": "node_contention", "reason": "spread"}})
    code = f'''
import asyncio, json
from pathlib import Path
from crucible.interop.verifiers.taskset import build
from verifiers.v1.errors import TaskError
T, _ = build()
class R:
    def __init__(self, d): self.d = Path(d)
    async def run(self, cmd, env=None): return type("X", (), {{"exit_code": 0, "stdout": "", "stderr": ""}})()
    async def read(self, path, max_bytes=None):
        p = self.d / Path(path).name
        if not p.exists(): raise OSError(path)
        return p.read_bytes()
class Tr:
    def __init__(self): self.metrics, self.info = {{}}, {{}}
    def record_metrics(self, m): self.metrics.update(m)
t = object.__new__(T); t.data = type("D", (), {{"verifier_env": {{}}}})()
tr = Tr(); out = asyncio.run(t.run_verifier(R("{ok}"), tr))
print("OK", out, sorted(tr.metrics))
try:
    asyncio.run(t.run_verifier(R("{masked}"), Tr())); print("NO-RAISE")
except TaskError as e:
    print("MASKED", str(e)[:60])
'''
    out = _run_env("verifiers", code)
    assert "OK {'reward': 1.0}" in out and "crucible/gate/policy" in out
    assert "MASKED crucible:masked:node_contention" in out and "NO-RAISE" not in out


def test_openenv_export_identical_and_telescoping(tmp_path):
    """The OpenEnv path (its own Environment/MCP types, in crucible/.envs/openenv) gives the native reward.json and
    steps.jsonl byte for byte, and its per-step rewards sum exactly to the final reward."""
    import asyncio

    from crucible_pack_toy.m11.gate import native

    asyncio.run(native(tmp_path / "native"))
    _run_env("openenv", f"import sys; sys.argv=['g','openenv','{tmp_path / 'oe'}']\nfrom crucible_pack_toy.m11.gate import main; main()")
    assert (tmp_path / "native" / "reward.json").read_text() == (tmp_path / "oe" / "reward.json").read_text()
    assert (tmp_path / "native" / "steps.jsonl").read_text() == (tmp_path / "oe" / "steps.jsonl").read_text()
    p = json.loads((tmp_path / "oe" / "path.json").read_text())
    assert abs(p["sum"] - p["final"]) < 1e-9 and p["masked"] is False and p["fingerprint"] == json.loads((tmp_path / "oe" / "path.json").read_text())["fingerprint"]
