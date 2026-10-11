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

    await Runner(tmp_path / "runs").run(RunSpec(run_id="demo-run", tasks=[HelpdeskTask()], policies=[PolicyRef(id="scripted:gold", make=lambda a: {"agent": ScriptedPolicy("agent", list(GOLD))})],
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
    assert [r["run_id"] for r in runs] == ["demo-run"] and runs[0]["results"] == 2
    s = c.get("/api/v1/crucible/runs/demo-run/summary").json()
    assert s["summary"]["models"][0]["overall"]["deterministic"]["pass@1"] == 1.0
    assert "<!doctype html>" in c.get("/api/v1/crucible/runs/demo-run/report").text
    assert c.get("/api/v1/crucible/runs/demo-run/episodes", params={"limit": 1}).json()["total"] == 2
    assert c.get("/api/v1/crucible/datasets").json() == [{"path": "lab", "datasets": {"sft": 3}, "source": "toy"}]
    assert c.get("/api/v1/crucible/runs/..%2Fetc/summary").status_code in (400, 404)
    assert c.get("/api/v1/crucible/runs/nope/summary").status_code == 404
