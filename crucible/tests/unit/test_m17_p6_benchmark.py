"""P6 sealed held-out benchmark (``factory.benchmark``; swarm-pipeline/00 §10, V9): sealing by repository, organisation,
time split and contamination; the training gates on held-out windows (no prompt: ``prompt_pending``); the manifest with
content hashes and its verification; held-out ids refused by the training rows; publication refused while prompts are
pending; the report plan through ``bench.lh_metrics``.

The stores are built from the window lane's toy window (``test_m17_w_task.make_task``), graded on the host interpreter."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m17_w_miner import SB
from test_m17_w_task import make_task

from crucible.bench.lh_metrics import LHEpisode
from crucible.factory.benchmark import BuildConfig, GateConfig, build, paired_report, publish, report, sealed_ids, verify
from crucible.factory.lanes.windows import WindowStore
from crucible.factory.sealing import SealError
from crucible.train.bridges.slime.rows import SealedRowError, slime_row, write_rows

FAST = GateConfig(gates=("s5", "s6", "s8"), concurrency=2)


def _put(store: WindowStore, rec: dict, base: Path, **upd) -> dict:
    r = {**json.loads(json.dumps(rec)), "stage": "mutants_valid", "independent": False, **upd}
    d = store.dir(r["id"])
    d.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base, d / "base", symlinks=True)
    store.save_window(r)
    return r


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p6bench")
    (tmp / "src").mkdir()
    task, rec, _ = make_task(tmp / "src")
    base = Path(task.base_dir)
    train = WindowStore(tmp / "train", split="train")
    other = {**rec["oracle_files"], "pkg/zeta.py": "def zeta_transform(values, factor):\n    return [v * factor + len(values) for v in values if v % 3]\n"}
    other = {k: v for k, v in other.items() if k not in ("pkg/gamma.py",)}
    _put(train, rec, base, id="click-wtrain", repo="click", b_date="2024-06-01T00:00:00+00:00", oracle_files=other, split="train")
    held = WindowStore(tmp / "heldout", split="heldout")
    late = "2025-03-01T00:00:00+00:00"
    _put(held, rec, base, id="voluptuous-wlate", repo="voluptuous", b_date=late, split="heldout")
    _put(held, rec, base, id="sqlparse-wearly", repo="sqlparse", b_date="2023-01-01T00:00:00+00:00", split="heldout")
    _put(held, rec, base, id="jinja-worg", repo="jinja", b_date=late, split="heldout")  # pallets, the org of the training click window
    _put(held, rec, base, id="tomlkit-wleak", repo="tomlkit", b_date=late, split="heldout", oracle_files=other)  # the training solution
    return SimpleNamespace(tmp=tmp, train=tmp / "train", heldout=tmp / "heldout", rec=rec)


def _cfg(stores, out: str, **kw) -> BuildConfig:
    return BuildConfig(heldout_root=stores.heldout, train_root=stores.train, out_dir=stores.tmp / out, gates=FAST, sandboxes=lambda _img: SB(), **kw)


@pytest.fixture(scope="module")
def built(stores):
    logs: list[str] = []
    m = asyncio.run(build(_cfg(stores, "bench"), log=logs.append))
    return SimpleNamespace(m=m, logs=logs, out=stores.tmp / "bench")


def test_sealing_keeps_only_tasks_apart_by_repo_org_time_and_content(built) -> None:
    m = built.m
    by = {t["id"]: t for t in m["tasks"]}
    assert m["split_members"] == ["voluptuous-wlate"] and m["sealed"] and not m["partial"]
    assert by["sqlparse-wearly"]["excluded"] == ["sealing:time_ok"]
    assert by["jinja-worg"]["excluded"] == ["sealing:org_ok"] and m["sealing_summary"]["shared_orgs"] == ["pallets"]
    assert by["tomlkit-wleak"]["excluded"][0] == "sealing:contamination_ok"  # (its swapped oracle also fails S5/S8)
    assert by["tomlkit-wleak"]["sealing"]["contamination"][0]["source"] == "click-wtrain:diff"
    assert m["sealing_summary"]["latest_train"]["window"] == "click-wtrain"


def test_gates_run_without_a_prompt_and_the_manifest_hashes_everything(built) -> None:
    t = next(x for x in built.m["tasks"] if x["id"] == "voluptuous-wlate")
    assert t["gates"] == {"s5": True, "s6": True, "s8": True} and t["prompt_status"] == "prompt_pending" and t["executed_edges"] == 1
    assert set(t["hashes"]) == {"window_record", "a_tree", "oracle_patch", "hidden_tests", "checkset", "verifier_version", "world_digest"}
    assert t["category"] == "mined_window/H1/unassigned" and t["horizon_class"] == "H1"
    g = json.loads((built.out / "gates" / "voluptuous-wlate.json").read_text())["gates"]
    assert g["s5"]["clean_world"]["reward"] == 1.0 and g["s5"]["null_agent"]["ok"] and g["s5"]["lint"] == "deferred: prompt_pending"
    assert g["s8"]["ok"] and "revert_m02" in g["s8"]["rows"] and g["s6"]["executed_edges"] == 1
    (row,) = [json.loads(x) for x in (built.out / "tasks.jsonl").read_text().splitlines()]
    assert row["split"] == "heldout" and row["args"]["ref"]["split"] == "heldout" and row["runtime_config"] == {"prompt_status": "prompt_pending"}
    assert built.m["prompt_authoring"]["status"] == "pending" and "metrics_plan" in built.m and built.m["not_verified"]


def test_verify_detects_tampering(stores, built, tmp_path: Path) -> None:
    out = tmp_path / "copy"
    shutil.copytree(built.out, out)
    assert verify(out, sandboxes=lambda _i: SB()) == []
    m = json.loads((out / "manifest.json").read_text())
    m["split_members"].append("sqlparse-wearly")
    (out / "manifest.json").write_text(json.dumps(m))
    assert "manifest.json was edited (sha256 mismatch)" in verify(out, sandboxes=lambda _i: SB())
    f = stores.heldout / "tasks" / "voluptuous-wlate" / "base" / "pkg" / "core.py"
    old = f.read_text()
    f.write_text(old + "\n# changed\n")
    try:
        probs = verify(built.out, sandboxes=lambda _i: SB())
        assert any(p.startswith("voluptuous-wlate: ['a_tree'") for p in probs), probs
    finally:
        f.write_text(old)


def test_held_out_ids_never_reach_training_rows(built, tmp_path: Path) -> None:
    sealed = sealed_ids(built.out / "manifest.json")
    assert sealed == {"voluptuous-wlate", "sqlparse-wearly", "jinja-worg", "tomlkit-wleak"}
    ref = {"lane": "window", "task_id": "jinja-worg", "source_id": "jinja-worg", "store": "/s", "split": "train", "prompt_sha256": "p",
           "checkset_sha256": "c", "verifier_version": "v"}
    row = slime_row(instance_id="jinja-worg", prompt="x", image="i", workdir="/w", agent="codex", crucible_task_ref=ref)
    with pytest.raises(SealedRowError):
        write_rows([row], tmp_path / "rows.jsonl", sealed=sealed)


def test_publication_waits_for_authored_prompts(built) -> None:
    with pytest.raises(SealError, match="prompt_pending"):
        publish(built.m, cards=[])


def test_time_policy_report_and_gate_reuse(stores, built) -> None:
    logs: list[str] = []
    out = stores.tmp / "bench"
    m = asyncio.run(build(BuildConfig(heldout_root=stores.heldout, train_root=stores.train, out_dir=out, gates=FAST, sandboxes=lambda _i: SB(),
                                      time_policy="report"), log=logs.append))
    assert sum("gates reused" in x for x in logs) == 4 and not any("gates running" in x for x in logs)
    assert set(m["split_members"]) == {"voluptuous-wlate", "sqlparse-wearly"} and not m["sealed"]
    asyncio.run(build(_cfg(stores, "bench"), log=lambda _m: None))  # restore the strict manifest for the other tests


def test_a_failing_gate_excludes_the_task(stores) -> None:
    cfg = BuildConfig(heldout_root=stores.heldout, train_root=stores.train, out_dir=stores.tmp / "flake", sandboxes=lambda _i: SB(),
                      gates=GateConfig(gates=("s10",), flake_reruns_per_level=10, concurrency=2), only=("voluptuous-wlate",))
    m = asyncio.run(build(cfg, log=lambda _m: None))
    (t,) = m["tasks"]
    g = json.loads((stores.tmp / "flake" / "gates" / "voluptuous-wlate.json").read_text())["gates"]["s10"]
    assert g["n"] == 20 and g["false_negatives"] == 0 and g["upper"] > 0.05  # 20 clean reruns bound the rate at 13.9%: not enough
    assert t["excluded"] == ["gate:s10"] and m["split_members"] == [] and m["partial"]


def test_report_plan_by_category_with_task_clustered_cis() -> None:
    eps = []
    for t, c, n_pass, cost in (("a", "H1", 5, 1.0), ("b", "H1", 0, 1.0), ("c", "H2", 3, 2.0), ("d", "H2", 1, 0.5)):
        for i in range(5):
            eps.append(LHEpisode(task=t, intent=c, passed=i < n_pass, series=[(5, 1.0 if i < n_pass else 0.25, [], [], True)], cost_usd=cost))
    eps.append(LHEpisode(task="a", intent="H1", masked=True))
    r = report(eps, category_of=lambda t: "H1" if t in "ab" else "H2", n_boot=500)
    h1 = r["H1"]
    assert h1["tasks"] == 2 and h1["pass@1"][0] == pytest.approx(0.5) and h1["pass^5"][0] == pytest.approx(0.5) and h1["summary"]["masked"] == 1
    assert h1["pass@1"][1] <= 0.5 <= h1["pass@1"][2] and r["overall"]["tasks"] == 4
    cpp = r["H2"]["usd_per_passing_episode"]
    assert cpp["by_task"] == {"c": pytest.approx(10 / 3), "d": pytest.approx(2.5)} and cpp["max_min_ratio"] == pytest.approx((10 / 3) / 2.5)
    assert r["H2"]["mean_final_phi"][0] == pytest.approx(((3 + 2 * 0.25) / 5 + (1 + 4 * 0.25) / 5) / 2)
    trained = [LHEpisode(task=e.task, intent=e.intent, passed=True, series=e.series) for e in eps if not e.masked]
    pr = paired_report(eps, trained, n_boot=500)
    assert pr["pass"]["n_tasks"] == 4 and pr["pass"]["mean_diff"] == pytest.approx(1 - (5 + 0 + 3 + 1) / 20) and pr["pass"]["mde"] is not None
