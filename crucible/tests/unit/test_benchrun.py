"""A sealed benchmark run as one arm (factory.benchrun, scripts/deltaai/bench.py): the seal check, the harness match
against the keep rule's measurements, the off-machine bundle, the arm's episodes and numbers, GPU-hours, the stale
policy-World guard and the shim's CPU pinning. Synthetic tasks only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from crucible.bench.lh_metrics import LHEpisode
from crucible.factory import benchrun
from crucible.factory.benchseal import _sha, task_hashes

ROOT = Path(__file__).resolve().parents[2]
POLICY = "solver:vllm:Qwen/Qwen3.6-35B-A3B@vllm"


def _store(debug: Path, tid: str, *, image: str = "crucible-lh-d:go-x-1", digest: str = "sha256:aa") -> Path:
    d = debug / "gen" / "store" / "heldout" / "tasks" / tid
    (d / "base").mkdir(parents=True)
    (d / "base" / "a.go").write_text("package a\n")
    rec = {"milestones": [{"tests": ["TestA", "TestB"]}], "oracle_files": {"a.go": "package a // fixed\n"}, "world": {"image": image, "digest": digest}}
    (d / "window.json").write_text(json.dumps(rec))
    (debug / "prompts").mkdir(parents=True, exist_ok=True)
    (debug / "prompts" / f"{tid}.json").write_text(json.dumps({"prompt": "a report"}))
    return d


def _manifest(debug: Path, ids: list[str]) -> dict:
    tasks = {t: {"store": str(debug / "gen" / "store" / "heldout" / "tasks" / t), "repo": "o/x", "seed": 7 + i,
                 "hashes": task_hashes(debug / "gen" / "store" / "heldout" / "tasks" / t, debug / "prompts" / f"{t}.json")} for i, t in enumerate(ids)}
    m = {"name": "bench-t", "k": 2, "tasks": tasks, "master_seed": 1}
    m["manifest_sha256"] = _sha(m)
    return m


def test_verify_passes_the_sealed_tasks_and_names_a_changed_one(tmp_path):
    debug = tmp_path / "debug"
    for t in ("db-go-x-00000001", "db-go-x-00000002"):
        _store(debug, t)
    m = _manifest(debug, ["db-go-x-00000001", "db-go-x-00000002"])
    assert benchrun.verify(m, debug=debug)["ok"]
    w = debug / "gen" / "store" / "heldout" / "tasks" / "db-go-x-00000002" / "window.json"
    rec = json.loads(w.read_text())
    rec["oracle_files"]["a.go"] = "package a // other\n"
    w.write_text(json.dumps(rec))
    v = benchrun.verify(m, debug=debug)
    assert not v["ok"] and any("db-go-x-00000002" in p and "oracle" in p for p in v["problems"])
    m2 = {**m, "k": 3}  # an edited manifest no longer matches its own sha
    assert any("manifest sha256" in p for p in benchrun.verify(m2, debug=debug)["problems"])


def _metrics(root: Path, phase: str, tid: str, harness: str, tool_limit: int) -> None:
    p = root / phase / "arms" / "student" / tid / "metrics.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"settings": {"harness": harness, "tool_limit": tool_limit}}))


def test_the_eval_harness_must_be_the_one_the_keep_rule_measured(tmp_path):
    _metrics(tmp_path, "p1", "db-go-x-00000001", "native:workspace@1#aaa", 300)
    _metrics(tmp_path, "p2", "db-go-x-00000002", "native:workspace@1#aaa", 300)  # the parent of the -d1 variant
    _metrics(tmp_path, "p0", "db-go-x-00000001", "native:workspace@1#old", 150)  # another budget: not this benchmark's
    meas = benchrun.measured_harness(tmp_path, ["db-go-x-00000001", "db-go-x-00000002-d1"], tool_limit=300)
    assert meas == {"native:workspace@1#aaa": ["p1/db-go-x-00000001", "p2/db-go-x-00000002"]}
    assert benchrun.harness_problems({"db-go-x-00000001": "native:workspace@1#aaa"}, meas) == []
    assert benchrun.harness_problems({"db-go-x-00000001": "native:workspace@1#bbb"}, meas)
    assert benchrun.harness_problems({"t": "h"}, {}) and benchrun.harness_problems({"t": "h"}, {"h": ["a"], "g": ["b"]})


def test_bundle_holds_the_sealed_tasks_only_and_relocates(tmp_path):
    debug = tmp_path / "debug"
    ids = ["db-go-x-00000001"]
    _store(debug, ids[0])
    _store(debug, "db-go-x-00000009")  # held out but not in this benchmark
    (debug / "candidates.json").write_text(json.dumps([{"id": ids[0], "repo": "x", "split": "heldout"}, {"id": "db-go-x-00000009", "repo": "x"},
                                                       {"id": "db-go-y-00000003", "repo": "y", "split": "train"}]))
    for sub in ("authored", "gates"):
        (debug / sub).mkdir()
        for t in (ids[0], "db-go-y-00000003"):
            (debug / sub / f"{t}.json").write_text("{}")
    src = debug / "src" / "o__x"
    (src / ".git" / "objects").mkdir(parents=True)
    lane = debug / "gen" / "clones" / "x" / ".git" / "objects" / "info"
    lane.mkdir(parents=True)
    (lane / "alternates").write_text(f"{src}/.git/objects\n")
    m = _manifest(debug, ids)
    out = tmp_path / "bundle"
    res = benchrun.bundle(m, debug=debug, out=out, clones={"x": src})
    assert res["tasks"] == ids and res["repos"] == ["x"]
    assert [r["id"] for r in json.loads((out / "candidates.json").read_text())] == ids
    assert not (out / "gen" / "store" / "heldout" / "tasks" / "db-go-x-00000009").exists() and not (out / "authored" / "db-go-y-00000003.json").exists()
    alt = (out / "gen" / "clones" / "x" / ".git" / "objects" / "info" / "alternates").read_text().strip()
    assert not alt.startswith("/") and (out / "gen" / "clones" / "x" / ".git" / "objects" / alt).resolve() == (out / "src" / "o__x" / ".git" / "objects").resolve()
    assert benchrun.verify(json.loads((out / "manifest.json").read_text()), debug=out, relocated=True)["ok"]
    benchrun.bundle(m, debug=debug, out=out, clones={"x": src})  # a re-bundle replaces, never fails on what is there
    assert benchrun.world_images(m) == {"crucible-lh-d:go-x-1": {"digest": "sha256:aa", "tasks": ids}}
    with pytest.raises(ValueError):
        benchrun.bundle(m, debug=debug, out=tmp_path / "b2", clones={})


def _ep(task: str, passed: bool, masked: bool = False) -> LHEpisode:
    return LHEpisode(task=task, passed=passed, masked=masked, series=[(10, 1.0 if passed else 0.5, ["m1"], [], True)])


def test_episodes_round_trip_and_the_arm_summary(tmp_path):
    eps = [_ep("a", True), _ep("a", False), _ep("b", False), _ep("b", False, masked=True)]
    p = tmp_path / "lh_episodes.jsonl"
    assert benchrun.save_episodes(p, eps) == 4
    back = benchrun.load_episodes(p)
    assert [(e.task, e.passed, e.masked, e.final_phi) for e in back] == [(e.task, e.passed, e.masked, e.final_phi) for e in eps]
    rows = [{"stop": "done", "masked": False}, {"stop": "timeout", "masked": False}, {"stop": "done", "masked": False},
            {"stop": None, "masked": True, "failure": "crucible:env_container_lost"}]
    s = benchrun.arm_summary(back, rows, k=2, policy=POLICY, n_boot=200)
    agg = s["aggregate"]
    assert agg["episodes"] == 4 and agg["masked"] == 1 and agg["host_failed"] == 1 and agg["tasks"] == 2
    assert agg["pass@1"][0] == pytest.approx(0.25)  # task a 1/2, task b 0/1 live: the mean over tasks
    assert agg["pass@2"][0] == pytest.approx(1.0)  # only task a has k=2 live attempts
    assert agg["tasks_short_of_k"] == 1 and agg["stops"] == {"done": 2, "none": 1, "timeout": 1}
    assert s["per_task"]["a"]["pass_rate"] == 0.5 and s["per_task"]["b"]["pass_rate"] == 0.0
    from crucible.factory.benchmark import paired_report

    assert paired_report(back, back, policies=[POLICY, POLICY])["pass"]  # a later trained arm pairs with this one as stored
    assert not benchrun.host_failed({"masked": True, "failure": "agent_run_error"})  # the harness's, not the host's


def test_gpu_hours_from_sacct():
    out = benchrun.gpu_hours("123|01:30:00|billing=2000,cpu=36,gres/gpu=2,mem=110G,node=1|COMPLETED\n124|1-00:00:30|cpu=8,mem=8G|FAILED\n")
    assert out["jobs"][0]["gpu_hours"] == 3.0 and out["jobs"][0]["billed_gpu_hours"] == 3.0 and out["jobs"][1]["gpus"] == 0
    assert out["gpu_hours"] == 3.0 and out["billed_gpu_hours"] == 3.0


def _script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_policy_world_fix_built_on_an_older_image_is_not_used():
    T = _script("lh_debug_target_bench_t", ROOT / "scripts" / "lh_debug_target.py")
    w = {"world": {"image": "crucible-lh-d:node-x-new"}}
    assert T.worldfix_applies({"accepted": True, "fixed": "crucible-lh-d:node-x-new-pw1", "pinned": "crucible-lh-d:node-x-new"}, w)
    assert not T.worldfix_applies({"accepted": True, "fixed": "crucible-lh-d:node-x-old-pw1", "pinned": "crucible-lh-d:node-x-old"}, w)
    assert not T.worldfix_applies({"accepted": False, "fixed": "f", "pinned": "crucible-lh-d:node-x-new"}, w) and not T.worldfix_applies(None, w)


def test_shim_pins_each_world_to_its_own_cores(tmp_path, monkeypatch):
    d = _script("docker_shim_pin_t", ROOT / "scripts" / "deltaai" / "docker_shim.py")
    monkeypatch.setenv("CRUCIBLE_SHIM_STATE", str(tmp_path))
    monkeypatch.setattr(d, "job_cpus", lambda: [4, 5, 6, 7])
    assert d.alloc_cores("w1", "2") is None  # off unless asked for
    monkeypatch.setenv("CRUCIBLE_SHIM_PIN", "1")
    a, b = d.alloc_cores("w1", "2"), d.alloc_cores("w2", "1.5")
    assert a == [4, 5] and b == [6, 7]  # disjoint while cores are free; ceil(1.5) = 2
    c = d.alloc_cores("w3", "2")
    assert len(c) == 2 and set(c) <= {4, 5, 6, 7}  # oversubscribed: the least-held cores, shared
    d.free_cores("w1")
    d.free_cores("w3")
    assert d.alloc_cores("w4", "2") == [4, 5] and d.alloc_cores("w5", None) is None
    assert d.pinned(None) is None and callable(d.pinned([4]))
