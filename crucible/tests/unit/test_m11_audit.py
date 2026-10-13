"""M11 interop audit (docs/audit/2026-10-12/M11-interop.md): regression tests, each failing on the M10 audit commit."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from crucible_pack_toy.runtime import HelpdeskTask
from crucible_pack_toy.verify import GOLD
from fastapi import Request  # module level: FastAPI resolves the (postponed) route annotations from module globals


class FlakyTickets(HelpdeskTask):
    """The tickets App raises once mid-episode (a synthetic lost DB connection): infrastructure, not the policy."""

    async def open(self, workdir):
        w = await super().open(workdir)
        app = w.apps["tickets"]
        orig, state = app.call, {"n": 0}

        async def call(name, args, *, actor, call_id):
            if name == "set_status" and not state["n"]:
                state["n"] = 1
                raise ConnectionError("db connection reset (synthetic infra fault)")
            return await orig(name, args, actor=actor, call_id=call_id)

        app.call = call
        return w


def _client(tasks, **kw):
    from fastapi.testclient import TestClient

    from crucible.interop.nemo_gym.server import make_app

    app = make_app(tasks, **kw)
    return app, TestClient(app)


def _tool(a) -> str:
    return "/" + a.function_name.replace(".", "__", 1)


# --------------------------------------------------------------------------------------------- NeMo-Gym export


def test_nemo_export_masks_an_infra_fault_during_the_episode():
    """H2 V1, G3 D1: a tool crash during an episode driven over the resources server was graded and scored 0."""
    _, c = _client({"t": FlakyTickets()})
    c.post("/seed_session", json={"resources_session_id": "s1", "task_id": "t"})
    for a in GOLD:
        c.post(_tool(a), json=a.arguments, headers={"X-Resources-Session-Id": "s1"})
    v = c.post("/verify", json={"resources_session_id": "s1"}).json()
    assert v["mask_sample"] is True and v["instance_config"]["mask_sample"] is True
    assert v["failure_kind"] == "crucible:tool_crashed" and v["crucible_reward"]["reward"] is None


def test_nemo_export_verify_called_as_a_tool_ends_the_episode():
    """H2 V5: NeMo-Gym's agent posts any function name to /{name}; a model calling ``verify`` read every check mid-episode
    and kept working. The first grade now ends the episode: later tools are refused and the grade does not change."""
    _, c = _client({"t": HelpdeskTask()})
    h = {"X-Resources-Session-Id": "s1"}
    c.post("/seed_session", json={"resources_session_id": "s1", "task_id": "t"})
    for a in GOLD[:3]:
        c.post(_tool(a), json=a.arguments, headers=h)
    peek = c.post("/verify", json={}, headers=h).json()
    after = [c.post(_tool(a), json=a.arguments, headers=h).json() for a in GOLD[3:]]
    assert all(x["ok"] is False for x in after)
    final = c.post("/verify", json={"resources_session_id": "s1"}).json()
    assert final["reward"] == peek["reward"] < 1.0


def test_nemo_export_close_before_verify_grades_not_masks():
    """A model calling ``close_session`` must not turn its own episode into a lost (masked) session."""
    _, c = _client({"t": HelpdeskTask()})
    h = {"X-Resources-Session-Id": "s1"}
    c.post("/seed_session", json={"resources_session_id": "s1", "task_id": "t"})
    c.post("/close_session", json={}, headers=h)
    v = c.post("/verify", json={"resources_session_id": "s1"})
    assert v.status_code == 200 and v.json()["mask_sample"] is False and v.json()["reward"] == 0.0


def test_nemo_export_reseed_is_idempotent_and_leaks_nothing():
    """H2 V10, G3 D4: a retried seed replaced the session without closing its World or releasing its slot."""
    app, c = _client({"t": HelpdeskTask()}, max_concurrent_episodes=2, queue_timeout_s=0.3)
    c.post("/seed_session", json={"resources_session_id": "dup", "task_id": "t"})
    w1 = app.state.sessions["dup"]["world"]
    c.post("/seed_session", json={"resources_session_id": "dup", "task_id": "t"})
    assert app.state.sessions["dup"]["world"] is w1 and app.state.stats["admitted"] == 1
    c.post("/close_session", json={"resources_session_id": "dup"})
    assert c.post("/seed_session", json={"resources_session_id": "x1", "task_id": "t"}).status_code == 200
    assert c.post("/seed_session", json={"resources_session_id": "x2", "task_id": "t"}).status_code == 200


def test_nemo_export_legacy_seed_and_unknown_task():
    """H2 V11, G3 D3: NeMo-Gym's SimpleAgent seeds with its dataset row (no session id, no task_id): it was a 500."""
    _, c = _client({"toy/helpdesk": HelpdeskTask()})
    r = c.post("/seed_session", json={"responses_create_params": {"input": [{"role": "user", "content": "x"}]}})
    assert r.status_code == 200 and r.json()["resources_session_id"] and set(r.json()) == {"resources_session_id"}
    assert c.post("/seed_session", json={"resources_session_id": "z", "task_id": "nope"}).status_code == 404
    bad = c.post("/tickets__add_note", content=b"[1,2]", headers={"X-Resources-Session-Id": r.json()["resources_session_id"], "content-type": "application/json"})
    assert bad.status_code == 200 and bad.json()["ok"] is False


def test_nemo_export_replies_validate_against_nemo_gym_models():
    """G3 D2, D5, D6, D17: only the unmasked verify body was validated; seed, close and reverify_mode failed NeMo-Gym's models."""
    import subprocess

    py = Path(__file__).resolve().parents[2] / ".envs" / "nemo" / "bin" / "python"
    if not py.exists():
        pytest.skip("NeMo-Gym env not built (crucible/.envs/nemo)")
    _, c = _client({"t": FlakyTickets()})
    seed = c.post("/seed_session", json={"resources_session_id": "v", "episode_id": "ep1", "task_id": "t", "task_data": {}}).json()
    for a in GOLD:
        c.post(_tool(a), json=a.arguments, headers={"X-Resources-Session-Id": "v"})
    rcp = {"input": [{"role": "user", "content": "work the tickets"}]}
    resp = {"id": "resp_1", "created_at": 0.0, "model": "m", "object": "response", "output": [], "parallel_tool_calls": False, "tool_choice": "auto", "tools": []}
    ver = c.post("/verify", json={"resources_session_id": "v", "responses_create_params": rcp, "response": resp}).json()
    close = c.post("/close_session", json={"resources_session_id": "v", "episode_id": "ep1"}).json()
    mode = c.get("/reverify_mode").json()
    agg = c.post("/aggregate_metrics", json={"verify_responses": [ver]}).json()
    docs = {"seed": seed, "verify": ver, "close": close, "mode": mode, "agg": agg}
    code = ("import json,sys\nfrom nemo_gym.base_resources_server import BaseMultiRewardVerifyResponse as V, ResourcesSeedSessionResponse as S, "
            "ResourcesCloseSessionResponse as C, ReverifyMode as R\nfrom nemo_gym.config_types import AggregateMetrics as A\n"
            "d=json.loads(sys.stdin.read())\nS.model_validate(d['seed']); C.model_validate(d['close']); R(d['mode'])\n"
            "v=V.model_validate(d['verify']); a=A.model_validate(d['agg'])\nprint(v.mask_sample, v.failure_kind, a.agent_metrics['masked'])")
    r = subprocess.run([str(py), "-c", code], input=json.dumps(docs), capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["True", "crucible:tool_crashed", "1"]


def test_nemo_remote_client_masks_server_failures_and_sends_done():
    """H2 V12: a 404 at /verify raised, and ``Done`` never reached the server's World."""
    from crucible.actors.policy import ScriptedPolicy
    from crucible.core.schema.actions import Done
    from crucible.interop.nemo_gym.client import remote_episode

    app, c = _client({"t": HelpdeskTask()})

    class Shim:  # httpx.AsyncClient over the in-process TestClient
        async def post(self, url, **kw):
            path = url.split("http://srv", 1)[1]
            if path == "/verify":
                app.state.sessions.clear()  # reaped just before the grade
            return c.post(path, **kw)

        async def get(self, url, **kw):
            return c.get(url.split("http://srv", 1)[1], **kw)

    paths = []
    _post = Shim.post

    async def post(self, url, **kw):
        paths.append(url.split("http://srv", 1)[1])
        return await _post(self, url, **kw)

    Shim.post = post
    ep = asyncio.run(remote_episode(["http://srv"], "t", ScriptedPolicy("agent", [*GOLD[:1], Done(answer="42")]), rid="r1", client=Shim()))
    assert ep.masked and ep.failure_kind == "env_session_lost"
    assert "/crucible/done" in paths


# --------------------------------------------------------------------------------------------- NeMo-Gym import


def _fake_resources_server(mode: dict):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/seed_session")
    async def seed(req: Request):
        return JSONResponse({})

    @app.post("/lookup")
    async def lookup(req: Request):
        b = await req.json()
        if b.get("bad"):
            return JSONResponse({"detail": "invalid argument"}, status_code=422)
        return {"output": "42"}

    @app.post("/verify")
    async def verify(req: Request):
        body = await req.json()
        mode["seen"] = body["response"]
        m = mode["verify"]
        if m == "500":
            return JSONResponse({"detail": "judge crashed"}, status_code=500)
        if m == "no_reward":
            return {"response": {}}
        if m == "null":
            return {"reward": None}
        if m == "seven":
            return {"reward": 7.0}
        if m == "zero":
            return {"reward": 0.0}
        return {"reward": 1.0}

    return app


NEMO_ROW = {"responses_create_params": {"input": [{"role": "system", "content": "be brief"}, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "find 42"}],
                                        "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}]}}


async def test_nemo_import_server_failures_mask_and_bad_calls_are_scored(tmp_path):
    """H2 V4, G3 D9/D10, G5: a 500, a missing/null reward, an out-of-range reward or a dead server scored 0 (or clamped);
    a 4xx tool reply masked the episode; the final answer reached /verify cut to 2,000 characters; outcome was 1.0 at reward 0."""
    import uvicorn

    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
    from crucible.core.schema.actions import Done, ToolCall
    from crucible.interop.nemo_gym.import_server import NemoGymTask
    from crucible.world.packaging.sidecar import free_port

    mode = {"verify": "ok"}
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(_fake_resources_server(mode), host="127.0.0.1", port=port, log_level="error"))
    t = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.02)
    task = NemoGymTask("nemo/fake", f"http://127.0.0.1:{port}", NEMO_ROW)
    assert "[assistant] hi" in task.instruction and "[system] be brief" in task.instruction
    got = {}
    for m in ("ok", "zero", "500", "no_reward", "null", "seven"):
        mode["verify"] = m
        w = await task.open(tmp_path / f"w{m}")
        rep = await task.grade(w, tmp_path / f"g{m}")
        await w.close()
        got[m] = (rep.reward, rep.outcome, rep.infra.get("status"))
    assert got["ok"] == (1.0, 1.0, "ok") and got["zero"] == (0.0, 0.0, "ok")
    assert all(got[m][2] == "masked" for m in ("500", "no_reward", "null", "seven"))
    # a 4xx tool reply is shown to the policy and scored; the full final answer reaches /verify
    mode["verify"] = "ok"
    w = await task.open(tmp_path / "wa")
    answer = "x" * 2356
    ep = await run_episode(w, {"agent": ScriptedPolicy("agent", [ToolCall(tool_call_id="c1", function_name="gym.lookup", arguments={"bad": 1}),
                                                                   Done(reason=answer[:2000], answer=answer)])}, limits=EpisodeLimits(max_steps=5), reset=False)
    rep = await task.grade(w, tmp_path / "ga")
    assert not ep.masked and rep.infra["status"] == "ok"
    final = [o for o in mode["seen"]["output"] if o.get("type") == "message"]
    assert final and final[-1]["content"][0]["text"] == answer
    # the server dies between the episode and the grade
    srv.should_exit = True
    await t
    rep = await task.grade(w, tmp_path / "gd")
    await w.close()
    assert rep.infra["status"] == "masked"


# --------------------------------------------------------------------------------------------- verifiers export


def _run_env(env: str, code: str) -> str:
    import subprocess

    py = Path(__file__).resolve().parents[2] / ".envs" / env / "bin" / "python"
    if not py.exists():
        pytest.skip(f"{env} env not built (crucible/.envs/{env})")
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, check=False, timeout=600)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-3000:])
    return r.stdout


def test_verifiers_export_loads_by_id_and_ignores_planted_grader_files(tmp_path):
    """G2 D1/D2, H2 V8: verifiers clears only the reward files, so a planted mask.json masked the episode and planted
    steps.jsonl rode beside a real reward; and no verifiers loader could find the taskset (built at call time)."""
    code = f'''
import asyncio, json, shlex
from pathlib import Path
import verifiers.v1 as vf
from verifiers.v1.errors import TaskError
from verifiers.v1.utils.loaders import import_taskset, _plugin_class
m = import_taskset("crucible-vf")
TS = _plugin_class(m, vf.Taskset, "taskset")
print("LOADED", TS.__name__, "HarborEnv" in m.__all__)
from crucible.interop.verifiers.vf_taskset import CrucibleConfig, CrucibleTask
c = CrucibleConfig.model_fields
print("CONFIG", c["ignore_timeouts"].default, "path" in c)
D = Path("{tmp_path}")
class R:
    def __init__(self, grader): self.grader = grader
    async def run(self, cmd, env=None):
        if cmd[:2] == ["sh", "-c"] and cmd[2].startswith("rm -f"):
            for p in shlex.split(cmd[2])[2:]:
                (D / Path(p).name).unlink(missing_ok=True)
        elif cmd == ["bash", "/tests/test.sh"]:
            self.grader()
        return type("X", (), {{"exit_code": 0, "stdout": "", "stderr": ""}})()
    async def read(self, path, max_bytes=None):
        p = D / Path(path).name
        if not p.exists(): raise OSError(path)
        return p.read_bytes()
class Tr:
    def __init__(self): self.metrics, self.info = {{}}, {{}}
    def record_metrics(self, mm): self.metrics.update(mm)
t = object.__new__(CrucibleTask); t.data = type("X", (), {{"verifier_env": {{}}}})()
(D / "mask.json").write_text(json.dumps({{"mask_sample": True, "failure_kind": "node_contention"}}))
try:
    asyncio.run(t.run_verifier(R(lambda: None), Tr())); print("PLANT-MASK no raise")
except TaskError as e:
    print("PLANT-MASK", "masked" if "crucible:masked" in str(e) else "error")
(D / "steps.jsonl").write_text(json.dumps({{"call_id": "c0", "delta_phi": 1.0}}) + "\\n")
tr = Tr()
out = asyncio.run(t.run_verifier(R(lambda: (D / "reward.json").write_text(json.dumps({{"reward": 0.0}}))), tr))
print("PLANT-STEPS", out["reward"], "crucible_steps" in tr.info)
'''
    out = _run_env("verifiers", code)
    assert "LOADED CrucibleTaskset True" in out and "CONFIG False True" in out
    assert "PLANT-MASK error" in out and "PLANT-STEPS 0.0 False" in out


# --------------------------------------------------------------------------------------------- verifiers v1 import


def _vf_trace(**kw):
    nodes = [{"token_ids": [1, 2], "mask": [False, False], "sampled": False},
             {"token_ids": [3, 4], "mask": [True, True], "sampled": True, "logprobs": [-0.1235, -2.0], "parent": 0}]
    return {"nodes": nodes, "ok": True, "errors": [], "rewards": {"solved": {"score": 1.0, "weight": 1.0}}, **kw}


def test_vf_import_masks_by_ok_and_never_guesses():
    """G2 D3-D6, H2 V13: masked on any error even after a successful retry; ok=False with no errors trained; None rewards
    crashed; a second branch replaced the first; traces.jsonl's 4-decimal logprobs were trained."""
    from crucible.interop.verifiers.v1_import import trace_to_record

    r = trace_to_record(_vf_trace(errors=[{"type": "ProviderError"}]), "r1")  # recovered on retry: ok
    assert not r.mask_sample and r.reward == 1.0
    r = trace_to_record(_vf_trace(ok=False), "r2")
    assert r.mask_sample and r.reward is None
    r = trace_to_record(_vf_trace(ok=False, errors=[{"type": "TaskError"}, {"type": "ProviderError"}]), "r3")
    assert r.failure_kind == "provider_unavailable"
    r = trace_to_record(_vf_trace(rewards={"solved": None}, stop_condition="scoring_timeout"), "r4")
    assert r.mask_sample and r.failure_kind == "verifier_timeout"
    t = _vf_trace()
    t["nodes"].append({"token_ids": [5], "mask": [True], "sampled": True, "logprobs": [-0.5], "parent": 0})
    assert trace_to_record(t, "r5").mask_sample
    t = _vf_trace(_crucible_completions=[([3, 4], [-0.123456789, -2.000049])])
    assert trace_to_record(t, "r6").turns[0].logprobs == [-0.123456789, -2.000049]
    assert trace_to_record(_vf_trace(_crucible_completions=[([9, 9], [-1.0, -1.0])]), "r7").mask_sample


# --------------------------------------------------------------------------------------------- agentenv


async def test_agentenv_backend_keeps_fault_labels(tmp_path):
    """H2 V2: on the agentenv backend every App fault arrived as a plain tool error, so a crash scored 0 (masked locally)."""
    pytest.importorskip("agent_env")
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

    w = await FlakyTickets(backend="agentenv").open(tmp_path / "w")
    try:
        ep = await run_episode(w, {"agent": ScriptedPolicy("agent", list(GOLD))}, limits=EpisodeLimits(max_steps=50), reset=False)
    finally:
        await w.close()
    assert ep.masked and ep.stop_condition == "crucible:tool_crashed"


async def test_agentenv_outside_agent_goes_through_the_world_and_cannot_reach_the_control_plane():
    """H2 V3/V5/V6, G4 D1/D3/D5/D7: an outside agent's calls skipped the World (no log, no events, a crash scored 0); it
    could re-enable a revoked tool through the gateway root; concurrent runs of one env_id shared a World; the verifier
    rows were one per gate (aggregate_verifiers scored the gate mean)."""
    pytest.importorskip("agent_env")
    import httpx
    from agent_env.task_step.context import TaskStepContext

    from crucible.interop.agentenv.plugin.steps import WORLDS, CrucibleMaskedGrade, CrucibleVerifyStep, CrucibleWorldStep, instance_for

    c1 = await CrucibleWorldStep("w", env_id="hd", task="crucible_pack_toy.runtime:HelpdeskTask").execute(TaskStepContext())
    c2 = await CrucibleWorldStep("w", env_id="hd", task="tests.unit.test_m11_audit:FlakyTickets").execute(TaskStepContext())
    i1, i2 = instance_for(c1, "hd"), instance_for(c2, "hd")
    assert i1 != i2 and WORLDS[i1]["world"] is not WORLDS[i2]["world"]
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            mcp1 = c1.deployed_envs[-1].mcp_url
            base = mcp1.removesuffix("/mcp")
            assert not mcp1.startswith(c1.deployed_envs[-1].gateway_url)
            assert (await c.post(f"{base}/tools/enable", json={"role": "agent", "tools": ["tickets__issue_refund"]})).status_code == 404
            assert (await c.get(f"{base}/trajectory")).status_code == 404
            for a in GOLD[:3]:  # run 1, through its agent proxy (role pinned)
                r = await c.post(f"{base}/step", json={"action": "call_tool", "tool_name": a.function_name.replace(".", "__", 1), "arguments": a.arguments},
                                 headers={"AgentEnv-Role": "admin"})
                assert r.status_code == 200, r.text
            base2 = c2.deployed_envs[-1].mcp_url.removesuffix("/mcp")
            for a in GOLD:  # run 2: its World's App crashes once
                await c.post(f"{base2}/step", json={"action": "call_tool", "tool_name": a.function_name.replace(".", "__", 1), "arguments": a.arguments})
        w1 = WORLDS[i1]["world"]
        assert sum(1 for e in w1.log if e.type == "tool_call") == 3  # the outside calls are World steps
        c1 = await CrucibleVerifyStep("v", env_id="hd").execute(c1)
        v1 = c1.metadata["verifications"]["crucible"]
        assert [r["id"] for r in v1["results"]] == ["reward"] and v1["results"][0]["score"] == v1["score"] < 1.0
        assert v1["results"][0]["result"] is False and c1.metadata["score"] == v1["score"] and i1 not in WORLDS
        with pytest.raises(CrucibleMaskedGrade):
            await CrucibleVerifyStep("v", env_id="hd").execute(c2)
        assert c2.metadata["verifications"]["crucible"]["crucible"]["failure_kind"] == "crucible:tool_crashed"
    finally:
        for k in (i1, i2):
            if k in WORLDS:
                await WORLDS.pop(k)["world"].close()


# --------------------------------------------------------------------------------------------- OpenEnv


@pytest.mark.parametrize("case", ["import", "export"])
def test_openenv_import_and_export_cases(case, tmp_path):
    """H3 V1-V5, G1 D1: restore accepted a different hidden state; a raising action masked the policy's own episode; a
    NaN terminal reward was scored; an infra-faulted or masked export episode paid its progress; finish re-paid."""
    import os
    import subprocess

    py = Path(__file__).resolve().parents[2] / ".envs" / "openenv" / "bin" / "python"
    if not py.exists():
        pytest.skip("OpenEnv env not built (crucible/.envs/openenv)")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([str(py), str(Path(__file__).with_name("_m11_openenv_cases.py")), case, str(tmp_path)], capture_output=True, text=True,
                       timeout=600, env=env, check=False)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-2000:], r.stderr[-3000:])


# --------------------------------------------------------------------------------------------- MiMo


def _mimo_env(root: Path, body: str) -> Path:
    env = root / "env"
    (env / "system").mkdir(parents=True)
    (env / "verify.py").write_text(body)
    return env


def test_mimo_grade_isolated_and_reads_mimos_own_fields(tmp_path, monkeypatch):
    """H3 V6-V8, G3 D14: GA_JUDGE_KEY2 from the environment reached the grader with judge=None; an importable ``synthesis``
    package decided the grade; outcome read a field MiMo never returns; out-of-range scores were scored."""
    import sys

    from crucible.interop.mimo.verify import grade

    seen = tmp_path / "seen.txt"
    env = _mimo_env(tmp_path, f'''
import os
def verify(env_path, ground_truth, model_answer, pre_state, post_state, agent_output):
    open({str(seen)!r}, "w").write(repr(sorted(k for k in os.environ if k.startswith("GA_JUDGE"))))
    try:
        import synthesis
        return {{"score": 1.0, "detail": {{"passed": True}}, "results": []}}
    except ImportError:
        pass
    s = float(agent_output or 0.5)
    return {{"score": s, "detail": {{"passed": s == 1.0}}, "results": [{{"id": "a", "passed": s == 1.0}}]}}
''')
    syn = tmp_path / "site" / "synthesis"
    syn.mkdir(parents=True)
    (syn / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.setenv("GA_JUDGE_KEY2", "canary-" + "x" * 8)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "site"))
    run = tmp_path / "run"
    run.mkdir()
    r = grade(env, run, agent_output="1.0", workdir=tmp_path / "g1")
    assert seen.read_text() == "[]"  # no judge configuration reached the grader
    assert r["infra"]["status"] == "ok" and r["reward"] == 1.0 and r["outcome"] == 1.0 and r["mimo"]["passed"] is True
    r = grade(env, run, agent_output="0.5", workdir=tmp_path / "g2")
    assert r["reward"] == 0.5 and r["outcome"] == 0.0  # the stand-in ``synthesis`` package did not decide the grade
    assert grade(env, run, agent_output="1.5", workdir=tmp_path / "g3")["infra"]["status"] == "masked"
    _ = sys


# --------------------------------------------------------------------------------------------- EnterpriseOps-Gym and SQL tools

EOG = Path(__file__).resolve().parents[3] / "RL-Environments/06-frameworks-and-env-code/servicenow-enterpriseops-gym"
CSM = "task_20251205_153330_906_a8eea1c0_8c7a6205"


async def _eog_play(task, actions, d: Path):
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

    w = await task.open(d / "w")
    try:
        await run_episode(w, {"agent": ScriptedPolicy("agent", list(actions))}, limits=EpisodeLimits(max_steps=30), reset=False)
        return await task.grade(w, d / "g")
    finally:
        await w.close()


@pytest.mark.skipif(not (EOG / "gym_dbs.zip").is_file(), reason="EnterpriseOps-Gym checkout not present")
async def test_eog_delete_task_typed_columns_and_eog_semantics(tmp_path):
    """H3 V11-V13, G3 D11/D12: a "drive the count to zero" task became an unwinnable refusal; a number sent as a string
    failed a check the benchmark's typed column passes; date ``greater_than`` could never pass."""
    from crucible.core.schema.actions import Done, ToolCall
    from crucible.interop.eog.importer import import_eog_task, load_task
    from crucible.verify.sql_checks import compare

    cfg, _ = load_task(next((EOG / "data" / "revised").glob(f"*/{CSM}.json")))
    gym = cfg["gym_servers_config"][0]["mcp_server_name"]
    delete = {**cfg, "task_id": "close1", "user_prompt": "Delete case 1.",
              "verifiers": [{"verifier_type": "database_state", "name": "gone", "gym_name": gym,
                             "validation_config": {"query": "SELECT COUNT(*) FROM customer_case WHERE case_id = 1;", "expected_value": 0, "comparison_type": "equals"}}]}
    t = import_eog_task(delete, seeds=EOG / "gym_dbs.zip", cache_dir=tmp_path / "c")
    assert not t.refusal
    rep = await _eog_play(t, [ToolCall(tool_call_id="d", function_name="csm.sql_execute", arguments={"sql": "DELETE FROM customer_case WHERE case_id = 1"}),
                              Done(reason="done")], tmp_path / "del")
    assert rep.reward == 1.0
    t = import_eog_task(cfg, seeds=EOG / "gym_dbs.zip", cache_dir=tmp_path / "c")
    stringly = [a.model_copy(update={"arguments": {**a.arguments, "params": [str(p) if isinstance(p, int) else p for p in a.arguments.get("params", [])]}})
                if isinstance(a, ToolCall) else a for a in t.oracle_actions()]
    assert (await _eog_play(t, stringly, tmp_path / "str")).reward == 1.0
    assert compare("2026-01-02", "2025-12-31", "gt", "eog") and not compare("1233", 1233, "equals", "eog")


async def test_sql_tools_cannot_write_outside_the_world_and_see_world_time(tmp_path):
    """H3 V14/V15: ``VACUUM INTO`` (even from the read-only tool) and ``ATTACH`` created files anywhere the process could
    write; the tools saw the host's clock, not the World's."""
    from crucible.core.schema.actions import ToolCall
    from crucible.core.schema.world import WorldSpec
    from crucible.core.store import Store
    from crucible.world.world import World

    schema = tmp_path / "s.sql"
    schema.write_text("CREATE TABLE t (x);")
    spec = WorldSpec.model_validate({"world": "t/sql", "clock": {"t0": "2025-12-05T15:33:30Z"},
                                     "apps": [{"id": "db", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "db.sqlite"}},
                                               "params": {"schema_sql": str(schema), "sql_tools": True, "sql_write": True}}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["db.*"]}]})
    w = World(spec, store=Store(tmp_path / "st"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        outside = tmp_path / "outside"
        outside.mkdir()
        for i, (tool, sql) in enumerate([("sql_execute", f"VACUUM INTO '{outside}/a.db'"), ("sql_query", f"VACUUM INTO '{outside}/b.db'"),
                                         ("sql_execute", f"ATTACH '{outside}/c.db' AS z"), ("sql_execute", "PRAGMA journal_mode=DELETE")]):
            r = await w.step("agent", ToolCall(tool_call_id=f"c{i}", function_name=f"db.{tool}", arguments={"sql": sql}))
            assert not r.observation.tool_results[0].ok, sql
        assert not list(outside.iterdir())
        r = await w.step("agent", ToolCall(tool_call_id="now", function_name="db.sql_query", arguments={"sql": "SELECT date('now')"}))
        assert "2025-12-05" in r.observation.tool_results[0].content
    finally:
        await w.close()


# --------------------------------------------------------------------------------------------- AWM


async def test_awm_verifier_name_rule_and_policy_caused_errors(tmp_path):
    """H3 V16/V17: a long final answer crashed a verifier and masked the episode the policy chose; a verifier whose
    function is not ``verify_task_completion`` was masked on every sample."""
    import sqlite3
    from types import SimpleNamespace

    from crucible.core.schema.trace import TraceEvent
    from crucible.interop.openenv.awm_task import AwmTask, function_name

    code = '''
def verify_task_complete(initial_db_path, final_db_path, final_answer):
    if final_answer and len(final_answer) > 10:
        raise ValueError("cannot parse a long answer")
    return {"result": "complete" if final_answer == "42" else "others"}
'''
    assert function_name(code) == "verify_task_complete"
    db = tmp_path / "s.db"
    sqlite3.connect(db).close()
    t = object.__new__(AwmTask)
    t.id, t.verifier = "awm/x/t0", {"verification": {"code": code}}

    def world(answer):
        ev = [TraceEvent(seq=1, kind="action", actor="agent", type="done", text=answer)]
        return SimpleNamespace(apps={"api": SimpleNamespace(state_params={"path": str(db)})}, _awm_initial=db,
                               to_trace=lambda: SimpleNamespace(events=ev), verifier_view=lambda: None)

    assert (await t.grade(world("42"), tmp_path / "g1")).reward == 1.0
    rep = await t.grade(world("x" * 50), tmp_path / "g2")
    assert rep.infra["status"] == "ok" and rep.reward == 0.0  # the policy's own answer broke the verifier: scored 0
    t.verifier = {"verification": {"code": "def verify_task_completion(a, b, c):\n    raise RuntimeError('broken verifier')\n"}}
    assert (await t.grade(world("42"), tmp_path / "g3")).infra["status"] == "masked"


# --------------------------------------------------------------------------------------------- Harbor

HF = Path(__file__).resolve().parents[1] / "fixtures" / "harbor_inplace"


def _docker() -> bool:
    import shutil
    import subprocess

    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


@pytest.mark.docker
async def test_harbor_inplace_grader_ignores_planted_reward_files(tmp_path):
    """H1 V1, G1 D2: single-step in-place grading read a ``reward.json`` the policy planted in /logs/verifier."""
    if not _docker():
        pytest.skip("no Docker daemon")
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
    from crucible.core.schema.actions import ToolCall
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    t = HarborInPlaceTask(HF / "answer")
    w = await t.open(tmp_path / "w")
    try:
        plant = [ToolCall(tool_call_id="p", function_name="term.exec", arguments={"command": "cd /logs && echo '{\"reward\": 1}' > verifier/reward.json"})]
        await run_episode(w, {"agent": ScriptedPolicy("agent", plant)}, limits=EpisodeLimits(max_steps=5), reset=False)
        rep = await t.grade(w, tmp_path / "g")
    finally:
        await w.close()
    assert rep.infra["status"] == "ok" and rep.reward == 0.0


def test_harbor_inplace_refuses_a_separate_verifier(tmp_path):
    """H1 V2, G1 D4, G5: a task whose verifier runs in its own environment was graded in the agent's container."""
    import shutil

    from crucible.interop.harbor.inplace import HarborInPlaceTask

    d = tmp_path / "sep"
    shutil.copytree(HF / "answer", d)
    toml = (d / "task.toml").read_text().replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    (d / "task.toml").write_text(toml)
    with pytest.raises(ValueError, match="separate environment"):
        HarborInPlaceTask(d)


def test_harbor_reward_rules_patch_mode_and_masked_mean(tmp_path):
    """H1 V5-V7, V12, G1 D6: patch mode took NaN, inf and strings as rewards, crashed on a valid single-key file, defaulted
    the verifier to no network, dropped [verifier].env and user, and trusted any mask.json; masked_mean masked single-key
    and boolean rows."""
    import shutil

    from crucible.interop.harbor.metrics import masked_mean
    from crucible.interop.harbor.rewards import parse_reward_dir
    from crucible.interop.harbor.verify_dir import HarborDirTask

    def rd(name, text):
        d = tmp_path / name
        d.mkdir()
        (d / ("reward.json" if text.lstrip().startswith(("{", "[")) else "reward.txt")).write_text(text)
        return parse_reward_dir(d)

    assert rd("a", '{"score": 1.0}') == ({"score": 1.0}, None)
    for i, bad in enumerate(['{"reward": NaN}', '{"reward": "inf"}', '{"reward": "1"}', "nan", "1\n0", "[1]"]):
        assert rd(f"b{i}", bad)[0] is None, bad
    d = tmp_path / "task"
    shutil.copytree(HF / "answer", d)
    (d / "task.toml").write_text((d / "task.toml").read_text().replace('network_mode = "no-network"', 'network_mode = "public"')
                                 .replace("[verifier]\n", '[verifier]\nuser = "tester"\nenv = { TOKEN = "${M11_CANARY:-dflt}" }\n'))
    t = HarborDirTask(d, image="x")
    assert t.network == "public" and t.user == "tester" and t.verifier_env == {"TOKEN": "dflt"} and t.trust_mask is False
    m = masked_mean([{"score": 1}, {"reward": True}, {"a": 1, "b": 0}, None, {"reward": float("nan")}])
    assert m["graded"] == 2 and m["masked_mean"] == 1.0 and m["masked"] == 2 and m["ambiguous"] == 1


def test_harbor_docker_host_is_reference_counted(monkeypatch):
    """H1 V8: the first of two Worlds on one remote daemon to tear down unset DOCKER_HOST under the other."""
    import os

    from crucible.interop.harbor import world_backend as wb

    monkeypatch.delenv("DOCKER_HOST", raising=False)
    wb._acquire_docker_host("ssh://lab")
    wb._acquire_docker_host("ssh://lab")
    wb._release_docker_host("ssh://lab")
    assert os.environ.get("DOCKER_HOST") == "ssh://lab"
    wb._release_docker_host("ssh://lab")
    assert "DOCKER_HOST" not in os.environ


async def test_replay_masks_incomplete_transcripts_and_carries_the_final_answer(tmp_path):
    """H1 V9, G5: a transcript cut by max_steps or holding an unmapped tool name graded as a real 0; replay dropped the
    final answer, so answer-graded tasks replayed to 0."""
    from crucible.interop.replay import Action, from_responses, replay

    full = [Action(a.function_name, dict(a.arguments)) for a in GOLD]
    assert (await replay(HelpdeskTask(), full, tmp_path / "ok", episode_key="t|e"))["reward"] == 1.0
    assert (await replay(HelpdeskTask(), full, tmp_path / "cut", episode_key="t|e", max_steps=len(full) - 1))["reward"] is None
    bare = [Action("issue_refund" if a.tool == "tickets.issue_refund" else a.tool, a.args) for a in full]
    assert (await replay(HelpdeskTask(), bare, tmp_path / "bare", episode_key="t|e"))["reward"] is None
    acts = from_responses([{"type": "function_call", "name": "tickets__get_ticket", "arguments": "{}"},
                           {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "the answer"}]}])
    assert acts[-1].final and acts[-1].text == "the answer"


def test_m11_failure_names_normalize_and_atif_keeps_the_answer():
    """G5 D10, D2b: M11's own failure names normalized to unmapped (unknown blame, counted by the stop rule); the ATIF
    export cut the final answer to the 2,000-character reason."""
    from types import SimpleNamespace

    from crucible.core.schema.trace import TraceEvent
    from crucible.interop.harbor.atif import episode_to_atif
    from crucible.interop.verifiers.v1_import import trace_to_record
    from crucible.runtime.infra import failure

    for name in ("ambiguous_reward", "no_reward", "env_unreachable", "env_session_lost", "env_admission_timeout", "env_seed_failed",
                 "env_server_error", "crucible:replay_incomplete", "crucible:replay_divergence", "AddTestsDirError"):
        assert failure(name, "verification").who != "unknown", name
    answer = "y" * 2500
    tr = SimpleNamespace(trace_id="t1", family=None, events=[TraceEvent(seq=1, kind="action", actor="agent", type="done", text=answer[:2000], data={"answer": answer})])
    assert episode_to_atif(tr, "agent").steps[-1].message == answer
    assert trace_to_record(_vf_trace(stop_condition="max_turns"), "r").truncated


def test_optional_framework_rule_reads_imports_not_strings():
    """The suite's optional-framework rule matched ``from verifiers ...`` inside code strings, so test_interop_gate.py and
    this file were never collected in a full run."""
    from tests.conftest import _top_imports

    text = 'import json\ncode = """\nfrom verifiers.v1 import x\nimport openenv\n"""\nfrom pathlib import Path\n'
    assert _top_imports(text) == {"json", "pathlib"}
    assert _top_imports("import torch.nn as nn\nfrom openenv.core import y\n") == {"torch", "openenv"}
