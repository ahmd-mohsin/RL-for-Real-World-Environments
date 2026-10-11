"""M14 packs and the rest of M15 studio:
- every pack under packs/ passes `pack check --level fixture` (the coding pack only when its recorded data is present);
- finance_ap passes the semantic level and its controls gate zeroes approve-everything;
- generic ingest takes any registered connector (coding agent runs);
- `crucible model` mines the toy workflow model;
- `crucible export` writes canonical replays, ATIF trajectories (Harbor-validated) and a NeMo-Gym launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from crucible.cli.main import cli

CRUCIBLE = Path(__file__).resolve().parents[2]
MT = CRUCIBLE.parent / "migration_tasks"


def run(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def test_finance_ap_pack_semantic_and_controls(tmp_path):
    import asyncio

    from crucible.studio.pack import check

    rep = asyncio.run(check("finance_ap", level="semantic"))
    assert rep.ok, rep.to_json()["problems"]
    assert {r.id: r.status for r in rep.results}["fixture.cases"] == "pass"
    r = run("bench", "run", "finance_ap", "--out", str(tmp_path / "runs"), "--fresh", "1", "--k", "1", "--json")
    s = json.loads(r.output)["summary"]
    by = {m["policy"]: m["overall"]["deterministic"] for m in s["models"]}
    assert by["scripted:oracle"]["mean_reward"] == 1.0
    assert by["scripted:approve_all"]["mean_reward"] == 0.0 and by["scripted:approve_all"]["gate_profile"]["gates"]["controls"]["pass_rate"] == 0.0


async def test_all_packs_fixture_level():
    from crucible.studio.pack import all_packs, check

    names = {p.name for p in all_packs()}
    assert {"_toy", "coding", "finance_ap"} <= names
    for p in all_packs():
        if p.name == "coding" and not (MT / "task02_pcops_triton" / "solution" / "solution.patch").exists():
            continue  # the coding fixtures read the seed repositories
        rep = await check(p, level="fixture")
        assert rep.ok, (p.name, rep.to_json()["problems"])


@pytest.mark.skipif(not (MT / "task02_pcops_triton" / "runs").exists(), reason="needs ../migration_tasks recorded runs")
async def test_generic_ingest_coding_agent_runs(tmp_path):
    from crucible.studio.label import ingest
    from crucible.studio.pack import find_pack, load_manifest

    d = find_pack("coding")
    m = load_manifest(d)
    m.ingest[:] = [s for s in m.ingest if s.connector == "agent_runs"]
    traces = ingest(d, m, tmp_path / "store")
    assert len(traces) == 11 and all(t.family == "coding/agent_run" and t.trace_id.startswith("run:") for t in traces)


def test_model_and_export(tmp_path):
    r = run("model", "toy", "--out", str(tmp_path / "model"), "--json")
    assert r.exit_code == 0 and json.loads(r.output)["milestones"] >= 3
    r = run("export", "--to", "canonical", "--pack", "finance_ap", "--out", str(tmp_path / "can"), "--json")
    can = json.loads(r.output)
    assert can["tasks"] and all(v["reward"] == 1.0 and v["steps"] > 0 for v in can["tasks"].values())
    assert all((Path(v["dir"]) / "reward.json").exists() and (Path(v["dir"]) / "steps.jsonl").exists() for v in can["tasks"].values())
    r = run("export", "--to", "nemo-gym", "--pack", "toy", "--out", str(tmp_path / "nemo"), "--json")
    assert r.exit_code == 0 and (tmp_path / "nemo" / "nemo_tasks.py").exists()
    run("bench", "run", "finance_ap", "--out", str(tmp_path / "runs"), "--fresh", "0", "--k", "1")
    r = run("export", "--to", "atif", "--run", str(tmp_path / "runs" / "bench-finance_ap-bench"), "--out", str(tmp_path / "atif"), "--json")
    a = json.loads(r.output)
    assert a["trajectories"] == 6 and not a["invalid"]
