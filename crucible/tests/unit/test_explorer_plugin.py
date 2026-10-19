"""M15 UI seam: the agentenv explorer plugin serves packs, runs, M13 summaries and reports, episodes and label datasets
under /api/v1/crucible, and refuses run ids that would leave its data root."""

from __future__ import annotations

import pytest

pytest.importorskip("agent_env")
pytest.importorskip("fastapi")


async def test_explorer_routes(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from crucible.actors.policy import ScriptedPolicy
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.run import Runner, RunSpec
    from crucible.studio.explorer import CrucibleExplorerPlugin

    await Runner(tmp_path / "runs").run(RunSpec(run_id="demo-run", tasks=[HelpdeskTask()], policies=[PolicyRef(id="scripted:gold", make=lambda a: {"agent": ScriptedPolicy("agent", list(GOLD))}, config={"script": "gold"})],
                                                harnesses=[NativeHarness()], k=2))
    (tmp_path / "lab").mkdir()
    (tmp_path / "lab" / "summary.json").write_text('{"command": "label", "datasets": {"sft": 3}, "pack": "toy"}')
    plugin = CrucibleExplorerPlugin(root=tmp_path)
    assert plugin.type == "crucible"
    app = FastAPI()
    app.include_router(plugin.router)
    c = TestClient(app)
    assert {p["dir"] for p in c.get("/api/v1/crucible/packs").json()} >= {"_toy", "coding", "finance_ap"}
    assert c.get("/api/v1/crucible/packs/toy/check").json()["ok"]
    runs = c.get("/api/v1/crucible/runs").json()
    # a run id is the run's path under the data root (unique; M15 audit G2 D5)
    assert [r["run_id"] for r in runs] == ["runs/demo-run"] and runs[0]["results"] == 2 and runs[0]["name"] == "demo-run"
    s = c.get("/api/v1/crucible/runs/runs/demo-run/summary").json()
    assert s["summary"]["models"][0]["overall"]["deterministic"]["pass@1"] == 1.0
    rep = c.get("/api/v1/crucible/runs/runs/demo-run/report")
    assert "<!doctype html>" in rep.text and rep.headers["content-security-policy"] == "sandbox"  # stored HTML never runs script (G2 D3)
    assert not (tmp_path / "runs" / "demo-run" / "bench" / "report.html").exists()  # rendered in memory: a GET writes nothing (H2 V14)
    assert c.get("/api/v1/crucible/runs/runs/demo-run/episodes", params={"limit": 1}).json()["total"] == 2
    assert c.get("/api/v1/crucible/runs/runs/demo-run/episodes", params={"offset": -3}).json()["offset"] == 0
    assert c.get("/api/v1/crucible/datasets").json() == [{"path": "lab", "datasets": {"sft": 3}, "source": "toy"}]
    # the plugin's own checks answer (400 / 404 with its message), not the router (G2 D13)
    r = c.get("/api/v1/crucible/runs/%2E%2E/summary")
    assert r.status_code == 400 and r.json()["detail"] == "invalid run id"
    r = c.get("/api/v1/crucible/runs/nope/summary")
    assert r.status_code == 404 and str(tmp_path) not in r.text and "no run" in r.json()["detail"]
