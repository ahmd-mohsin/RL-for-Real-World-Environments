"""M15 audit: the studio's build pipeline, exports, explorer, CLI (docs/audit/2026-10-13/M15-study-defects-H2.md V1-V21,
M15-sweep-G2.md D1-D13) and the public site's copy rules (H2 V22-V25).

Every test here fails on the M14 commit (076ab65) and passes after the fix. No GPU, no paid call, no network. Secrets are
fake canaries."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest
from click.testing import CliRunner

from crucible.cli.main import cli

CRUCIBLE = Path(__file__).resolve().parents[2]
PACKS = CRUCIBLE / "packs"
FAKE_KEY = "sk-proj-FAKECANARY000000000000000000000001"  # shaped like a provider key; not one
FAKE_EMAIL = "jane.canary@example.com"


def run(*args, env=None):
    return CliRunner().invoke(cli, [str(a) for a in args], env=env)


def out_json(r):
    return json.loads(r.stdout)


def toy_copy(tmp_path: Path, name: str = "toycopy", edit=None) -> Path:
    d = tmp_path / name
    shutil.copytree(PACKS / "_toy", d, ignore=shutil.ignore_patterns("__pycache__"))
    if edit:
        y = d / "pack.yaml"
        y.write_text(edit(y.read_text()))
    return d


FIXED_KEY = "m15-audit-fixture-key-a-0123456789abcdefghij"


@pytest.fixture(scope="module")
def good_build(tmp_path_factory):
    """One clean `crucible build toy` (human output), shared by the read-only checks and copied by the rebuild tests."""
    out = tmp_path_factory.mktemp("good") / "build"
    # a fixed key: the bench split is drawn from it, and a fresh random key can seal a bench whose items all match the
    # training corpus, which build correctly refuses (an open limit of the small toy pack, M15 audit report §4)
    r = run("build", "toy", "--out", out, env={"CRUCIBLE_BENCH_KEY": FIXED_KEY})
    assert r.exit_code == 0, r.output
    return out, r


def _clean_env():
    return {"CRUCIBLE_BENCH_KEY": None, "CRUCIBLE_BENCH_KEY_FILE": None}


# ---- V1: an oracle replay crash is a failed build, and the anchor is invalid ---------------------------------------------
def test_v1_oracle_replay_crash_fails_the_build(tmp_path, monkeypatch):
    import crucible.interop.replay as rp

    async def boom(*a, **k):
        raise RuntimeError("replay crashed")

    monkeypatch.setattr(rp, "replay", boom)
    out = tmp_path / "b"
    r = run("build", "toy", "--out", out, "--json", env=_clean_env())
    assert r.exit_code == 1, r.output
    s = out_json(r)
    assert s["ok"] is False and any(f["stage"] == "export" for f in s["failures"])
    v = json.loads((out / "runs" / "bench-toy-build" / "validity.json").read_text())
    anchors = {k: x for k, x in v.items() if k.startswith("toy/helpdesk_v2/")}
    assert anchors and all(x["valid"] is False and "oracle_not_replayed" in x["reasons"] for x in anchors.values())


# ---- V2: export never deletes what it did not write -------------------------------------------------------------------
def test_v2_export_refuses_to_replace_the_source_pack_or_a_foreign_dir(tmp_path):
    src = toy_copy(tmp_path / "x", "pack")
    r = run("export", "--to", "nemo-gym", "--pack", src, "--out", tmp_path / "x")
    assert r.exit_code != 0 and (src / "pack.yaml").is_file() and len(list(src.rglob("*"))) > 50
    n = len(list(src.rglob("*")))
    r = run("export", "--to", "nemo-gym", "--pack", src, "--out", src / "sub")  # an --out inside the pack: nothing written there
    assert r.exit_code == 4 and "inside the source pack" in r.output and len(list(src.rglob("*"))) == n and not (src / "sub").exists()
    other = tmp_path / "y"
    (other / "pack").mkdir(parents=True)
    (other / "pack" / "mine.txt").write_text("user data")
    r = run("export", "--to", "nemo-gym", "--pack", "toy", "--out", other)
    assert r.exit_code != 0 and (other / "pack" / "mine.txt").read_text() == "user data"
    r = run("export", "--to", "nemo-gym", "--pack", "toy", "--out", tmp_path / "z")
    assert r.exit_code == 0, r.output
    r = run("export", "--to", "nemo-gym", "--pack", "toy", "--out", tmp_path / "z")  # its own earlier bundle is replaced
    assert r.exit_code == 0 and (tmp_path / "z" / "pack" / "pack.yaml").is_file()


# ---- V3 / G2 D1: the bundle is an allowlist and is scanned for secrets ---------------------------------------------------
def test_v3_bundle_ships_no_env_raw_logs_or_secrets(tmp_path):
    d = toy_copy(tmp_path)
    pkg = d / "src" / "crucible_pack_toy"
    (d / "src" / ".env").write_text(f"OPENAI_API_KEY={FAKE_KEY}\n")
    (pkg / ".env.local").write_text(f"OPENAI_API_KEY={FAKE_KEY}\n")
    (d / "data" / "server.key").write_text("not a real key\n")
    (d / "data" / "ticket_logs" / "raw_dump.jsonl").write_text(json.dumps({"customer_email": FAKE_EMAIL, "card": "4111 1111 1111 1111"}) + "\n")
    out = tmp_path / "out"
    r = run("export", "--to", "nemo-gym", "--pack", d, "--out", out)
    assert r.exit_code == 0, r.output
    files = [p for p in (out / "pack").rglob("*") if p.is_file()]
    names = {p.relative_to(out / "pack").as_posix() for p in files}
    assert not any(n.endswith((".env", ".env.local", ".key")) for n in names) and not any(n.startswith("data/ticket_logs/") and n.endswith(".jsonl") for n in names)
    assert "src/crucible_pack_toy/runtime.py" in names and "pack.yaml" in names
    blob = "\n".join(p.read_text(errors="replace") for p in files)
    assert FAKE_KEY not in blob and FAKE_EMAIL not in blob
    (d / "data" / "notes.txt").write_text(f"deploy with {FAKE_KEY}\n")  # a credential in a shipped data file refuses the export
    r = run("export", "--to", "nemo-gym", "--pack", d, "--out", tmp_path / "out2")
    assert r.exit_code != 0 and "credential" in r.output and not (tmp_path / "out2" / "pack").exists()
    # a dotenv the pack names as redaction_env: its values are never shipped, wherever they appear
    (d / "data" / "notes.txt").unlink()
    (d / "data" / "secrets.env").write_text("SERVICE_TOKEN=plainvalue-canary-7781\n")
    (d / "data" / "kb" / "leak.md").write_text("the token is plainvalue-canary-7781\n")
    y = d / "pack.yaml"
    y.write_text(y.read_text().replace("extra: [status, team, score, amount, rule, sla, plan]}", "extra: [status, team, score, amount, rule, sla, plan], redaction_env: data/secrets.env}"))
    r = run("export", "--to", "openenv", "--pack", d, "--out", tmp_path / "out3")
    assert r.exit_code != 0 and "credential" in r.output


# ---- V4: nothing to bench is a failure, with the stage named ----------------------------------------------------------
def test_v4_build_fails_when_m9_rejects_everything(tmp_path, monkeypatch):
    import crucible.accept.run as ar

    orig = ar.verdict
    monkeypatch.setattr(ar, "verdict", lambda rep: {**orig(rep), "state": "rejected", "blocking": {"A1": "fail"}})
    r = run("build", "toy", "--out", tmp_path / "b", "--json", env=_clean_env())
    assert r.exit_code == 1 and out_json(r)["error"]["stage"] == "accept"


def test_v4_build_fails_when_forge_or_ingest_makes_nothing(tmp_path, monkeypatch):
    import crucible.studio.pack as sp

    orig = sp.resolve

    def res(d, ref):
        f = orig(d, ref)
        if ref.endswith(":forge"):
            async def empty(*a, **k):
                await f(*a, **k)
                return []
            return empty
        return f

    monkeypatch.setattr(sp, "resolve", res)
    r = run("build", "toy", "--out", tmp_path / "b", "--json", env=_clean_env())
    assert r.exit_code == 1 and out_json(r)["error"]["stage"] == "forge"
    monkeypatch.setattr(sp, "resolve", orig)
    d = toy_copy(tmp_path, edit=lambda t: t.replace("glob: data/ticket_logs/*.jsonl", "glob: data/ticket_logs/none-*.jsonl"))
    r = run("build", d, "--out", tmp_path / "b2", "--json", env=_clean_env())
    assert r.exit_code == 1 and out_json(r)["error"]["stage"] == "ingest"


# ---- V5: build applies bench run's preflight ---------------------------------------------------------------------------
def test_v5_build_enforces_requires_and_orientation(tmp_path):
    d = toy_copy(tmp_path, "req", edit=lambda t: t.replace("determinism: seeded", "determinism: seeded\nrequires: [paid, network]"))
    r = run("build", d, "--out", tmp_path / "b1")
    assert r.exit_code != 0 and "requires paid, network" in r.output and not (tmp_path / "b1" / "summary.json").exists()
    d2 = toy_copy(tmp_path, "low", edit=lambda t: t.replace("higher_is_better: true", "higher_is_better: false"))
    r = run("build", d2, "--out", tmp_path / "b2")
    assert r.exit_code != 0 and "higher_is_better: false" in r.output and not (tmp_path / "b2" / "summary.json").exists()


# ---- V6: the oracle bar is the top of reward.range -----------------------------------------------------------------------
def test_v6_anchor_oracle_must_reach_the_range_top(tmp_path):
    from crucible.studio.build import oracle_reasons

    assert oracle_reasons(1.0, 10.0) == ["broken_oracle"] and oracle_reasons(10.0, 10.0) == []
    assert oracle_reasons(float("nan"), 1.0) == ["broken_oracle"] and oracle_reasons(None, 1.0) == ["oracle_not_replayed"]
    d = toy_copy(tmp_path, "ten", edit=lambda t: t.replace("range: [0.0, 1.0]", "range: [0.0, 10.0]"))
    out = tmp_path / "b"
    r = run("build", d, "--out", out, "--fresh", "0", "--json", env=_clean_env())
    v = json.loads((out / "runs" / "bench-toy-build" / "validity.json").read_text())
    assert v and all(x["valid"] is False and "broken_oracle" in x["reasons"] for x in v.values()), r.output
    s = json.loads((out / "summary.json").read_text())  # no valid bench item: no metric means anything, so the build fails
    assert r.exit_code == 1 and s["ok"] is False and any("none of the" in f["reason"] for f in s["failures"])


def test_v8_bench_split_is_never_empty_and_is_reproducible():
    """The toy's five single-lineage anchors at bench_frac 0.3 left the bench empty for ~17% of fresh keys (a flaky build)."""
    import os as _os

    from crucible.bench.splits import SplitItem
    from crucible.studio.build import assign_splits

    items = [SplitItem(f"t{i}", lineage=(f"t{i}",), axes={}, source_kind="synthetic") for i in range(5)]
    for _ in range(600):  # both sides must be non-empty: no bench was 17% of keys, no train 0.24%
        key = _os.urandom(32)
        sp = assign_splits(items, key, 0.3)
        assert sp["counts"].get("bench") and sp["counts"].get("train") and assign_splits(items, key, 0.3)["seal"] == sp["seal"]


# ---- V7 / G2 D6: a failed or refused rebuild leaves the previous outputs whole -------------------------------------------
def _snapshot(out: Path) -> dict[str, str]:
    snap = {n: (out / n).read_text() for n in ("summary.json", "accept.json", "splits.json", "configs/resolved/build.json")}
    snap["export"] = json.dumps(sorted(p.name for p in (out / "export").iterdir()))
    return snap


def test_v7_failed_rebuild_keeps_the_previous_build(good_build, tmp_path, monkeypatch):
    import crucible.accept.run as ar

    out = tmp_path / "b"
    shutil.copytree(good_build[0], out, symlinks=True)
    before = _snapshot(out)
    orig = ar.verdict
    monkeypatch.setattr(ar, "verdict", lambda rep: {**orig(rep), "state": "rejected", "blocking": {"A1": "fail"}})
    r = run("build", "toy", "--out", out, env={**_clean_env(), "CRUCIBLE_BENCH_KEY": FIXED_KEY})  # the same build identity
    assert r.exit_code == 1
    assert _snapshot(out) == before and not list(out.glob(".build-*"))
    s, a = json.loads((out / "summary.json").read_text()), json.loads((out / "accept.json").read_text())
    assert s["accept"]["provisional"] == sum(x["state"] == "provisional" for x in a.values())


def test_d6_rerun_with_other_flags_is_refused_before_any_write(good_build, tmp_path):
    out = tmp_path / "b"
    shutil.copytree(good_build[0], out, symlinks=True)
    before = _snapshot(out)
    mt = (out / "accept.json").stat().st_mtime_ns
    r = run("build", "toy", "--out", out, "--k", "3", env=_clean_env())
    assert r.exit_code == 4 and "use a new --out" in r.output and "k" in r.output
    assert _snapshot(out) == before and (out / "accept.json").stat().st_mtime_ns == mt
    assert json.loads((out / "configs" / "resolved" / "build.json").read_text())["k"] == 1


# ---- V8: only the bench split is benched; the seal is bound and verified; contamination runs ------------------------------
def test_v8_bench_split_seal_and_contamination(good_build):
    out, _ = good_build
    s = json.loads((out / "summary.json").read_text())
    sp = json.loads((out / "splits.json").read_text())
    bench_ids = {k for k, v in sp["splits"].items() if v == "bench"}
    rows = [json.loads(x) for x in (out / "runs" / "bench-toy-build" / "results.jsonl").read_text().splitlines() if x.strip()]
    assert rows and {x["task"] for x in rows} <= bench_ids
    assert sum(s["bench"]["benched"].values()) == sp["counts"]["bench"] == len(bench_ids)
    saved = json.loads((out / "runs" / "bench-toy-build" / "bench" / "summary.json").read_text())
    assert saved["splits"]["seal"] == sp["seal"] == s["bench"]["splits"]["seal"] and s["bench"]["splits"]["verified"] is True
    assert saved["provenance"]["pack"]["content_sha256"] == s["provenance"]["pack"]["content_sha256"]  # G2 D12: the run dir says what made it
    assert "contamination" in s and s["contamination"]["scanned"] == len(bench_ids) and (out / "contamination.json").is_file()
    assert s["contamination"]["corpus"]["traces"] == 24


def test_v8_contaminated_bench_item_is_invalid():
    from crucible.studio.build import contamination_step

    text = " ".join(f"word{i}" for i in range(40))
    res = contamination_step({"b1": text, "b2": "an unrelated instruction about invoices and approvals"}, {"t1": text}, [], "x")
    assert res["contaminated"] == ["b1"]
    assert contamination_step({"b1": text}, {}, [], "x")["corpus"] is None  # said, not skipped silently



def test_v8_family_boilerplate_is_not_contamination():
    """Template subtraction over the whole benchmark: items sharing a family preamble or a generator template are not
    contaminated by their siblings in the train split (a bench-only template flagged every toy bench item; the build
    then had no valid bench item for ~3 of 4 keys)."""
    from crucible.studio.build import contamination_step

    pre = " ".join(f"policy{i}" for i in range(30))
    items = {f"t{i}": f"{pre} case {i} " + " ".join(f"detail{i}x{j}" for j in range(20)) for i in range(5)}
    res = contamination_step({"t4": items["t4"]}, {k: v for k, v in items.items() if k != "t4"}, [], "x")
    assert res["contaminated"] == [], res["hits"]
    leaked = contamination_step({"t4": items["t4"]}, {**{k: v for k, v in items.items() if k != "t4"}, "t9": items["t4"]}, [], "x")
    assert leaked["contaminated"] == ["t4"]  # an exact copy in the corpus is still found
    same = contamination_step({"g1": items["t4"]}, {"g2": items["t4"], **{k: v for k, v in items.items() if k != "t4"}}, [], "x", generated={"g1", "g2"})
    assert same["contaminated"] == []  # two fresh items (distinct seeds) may share their instruction word for word


# ---- G2 D8: build prints its warnings ------------------------------------------------------------------------------------
def test_d8_build_human_output_shows_warnings_and_splits(good_build, tmp_path, monkeypatch):
    out, r = good_build
    s = json.loads((out / "summary.json").read_text())
    assert s["ok"] is True and "splits" in r.stdout and "contamination" in r.stdout
    assert all(f"warning: {w}" in r.stdout for w in s["bench"]["warnings"])
    # a bench warning reaches the human output (the fixed-key build happens to have none, so inject one)
    import crucible.studio.build as sb

    async def with_warning(*a, **k):
        return {**s, "bench": {**s["bench"], "warnings": ["only 1 bench item: the CI is wide"]}}

    monkeypatch.setattr(sb, "build", with_warning)
    r2 = run("build", "toy", "--out", tmp_path / "w", env=_clean_env())
    assert "warning: only 1 bench item: the CI is wide" in r2.stdout


# ---- V9: task ids are safe, distinct directory names -----------------------------------------------------------------
def test_v9_task_ids_never_escape_or_collide(tmp_path):
    from crucible.studio.export import _canonical
    from crucible.studio.pack import find_pack, load_manifest, resolve

    d = find_pack("toy")
    m = load_manifest(d)
    (t, gold), = resolve(d, m.hooks.tasks)()[:1]
    t.id = ".."
    base = tmp_path / "b"
    base.mkdir()
    asyncio.run(_canonical([(t, gold)], base / "export"))
    assert sorted(p.name for p in base.iterdir()) == ["export"]
    import copy

    (t1, g1), = resolve(d, m.hooks.tasks)()[:1]
    t2, g2 = copy.copy(t1), g1
    t1.id, t2.id = "acme/t1", "acme_t1"
    r = asyncio.run(_canonical([(t1, g1), (t2, g2)], tmp_path / "e2"))
    assert len({v["dir"] for v in r["tasks"].values()}) == 2 and len(list((tmp_path / "e2").iterdir())) == 2


# ---- V10: async hooks.tasks is awaited, in export and in the generated launcher ---------------------------------------
def test_v10_async_tasks_hook_exports(tmp_path):
    from crucible.studio.pack import REPO_ROOT

    if not (REPO_ROOT / "migration_tasks").is_dir():
        pytest.skip("needs ../migration_tasks (the coding pack's task data)")
    for t in ("nemo-gym", "openenv"):  # coding's hooks.tasks is async: export iterated the coroutine (TypeError)
        r = run("export", "--to", t, "--pack", "coding", "--out", tmp_path / t)
        assert r.exit_code == 0, r.output
        assert len(json.loads((tmp_path / t / "crucible_export.json").read_text())["tasks"]) >= 2
    # the generated factory awaits the hook too, outside and inside a running event loop (OpenEnv calls it from its loop).
    # The coding pack reads its task data beside the checkout, so the factory is pointed at the pack in place here.
    src = (tmp_path / "nemo-gym" / "nemo_tasks.py").read_text()
    src = src.replace("Path(__file__).resolve().parent / 'pack'", repr(str(PACKS / "coding")).join(("Path(", ")")))
    ns: dict = {"__name__": "nemo_tasks_test"}
    exec(compile(src, "nemo_tasks.py", "exec"), ns)  # noqa: S102 - the exported launcher
    ids = sorted(ns["tasks"]())

    async def inside():
        return ns["make_task"](1).id

    assert len(ids) >= 2 and asyncio.run(inside()) in ids


# ---- V11: openenv.yaml's app is a servable ASGI app ---------------------------------------------------------------------
def test_v11_openenv_manifest_app_is_servable(tmp_path):
    py = CRUCIBLE / ".envs" / "openenv" / "bin" / "python"
    if not py.exists():
        pytest.skip("OpenEnv venv not built (crucible/.envs/openenv)")
    out = tmp_path / "oe"
    r = run("export", "--to", "openenv", "--pack", "finance_ap", "--out", out)
    assert r.exit_code == 0, r.output
    code = r'''
import importlib, sys, yaml
from fastapi.testclient import TestClient
man = yaml.safe_load(open("openenv.yaml"))
assert man["port"] == 8000, man
mod, attr = man["app"].split(":")
app = getattr(importlib.import_module(mod), attr)
assert callable(app) and not isinstance(app, type), app
c = TestClient(app)
r = c.post("/reset", json={})
assert r.status_code == 200, r.text
print("OK", r.json().get("done"))
'''
    paths = [CRUCIBLE / "src", *(PACKS / p / "src" for p in ("_toy", "finance_ap", "coding")), out]
    p = subprocess.run([str(py), "-c", code], cwd=out, capture_output=True, text=True, timeout=300, check=False,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))})
    assert p.returncode == 0 and "OK" in p.stdout, (p.stdout + p.stderr)[-2000:]


# ---- V12 / G2 D7 / D11: atif refuses nothing; sidecars carry attribution ------------------------------------------------
def test_v12_atif_export_fails_on_nothing_and_without_a_validator(tmp_path, monkeypatch, good_build):
    (tmp_path / "empty").mkdir()
    r = run("export", "--to", "atif", "--run", tmp_path / "empty", "--out", tmp_path / "a1", "--json")
    assert r.exit_code != 0 and "no samples" in r.stdout
    run_dir = good_build[0] / "runs" / "bench-toy-build"
    monkeypatch.setitem(sys.modules, "harbor.utils.trajectory_validator", None)  # Harbor not installed
    r = run("export", "--to", "atif", "--run", run_dir, "--out", tmp_path / "a2")
    assert r.exit_code != 0 and "--no-validate" in r.output
    r = run("export", "--to", "atif", "--run", run_dir, "--out", tmp_path / "a3", "--no-validate", "--json")
    assert r.exit_code == 0 and out_json(r)["validated"] is False


def test_d11_atif_trajectories_carry_task_policy_and_reward(tmp_path, good_build):
    pytest.importorskip("harbor")
    run_dir = good_build[0] / "runs" / "bench-toy-build"
    r = run("export", "--to", "atif", "--run", run_dir, "--out", tmp_path / "a", "--json")
    assert r.exit_code == 0, r.output
    side = sorted((tmp_path / "a").glob("*.result.json"))
    assert len(side) == out_json(r)["trajectories"] > 0
    docs = [json.loads(p.read_text()) for p in side]
    assert all(d["task"] and d["policy"] and d["reward"] is not None and d["run_identity"] for d in docs)
    assert {d["policy"] for d in docs} >= {"scripted:oracle", "scripted:noop"}


def test_d7_canonical_export_of_nothing_fails(tmp_path):
    demo = tmp_path / "demo"
    assert run("pack", "new", "demo", "--dir", demo).exit_code == 0
    import crucible.studio.export as ex

    res = asyncio.run(ex._canonical([], tmp_path / "e"))
    assert res["ok"] is False and res["reason"]
    r = run("build", "toy", "--k", "0", "--out", tmp_path / "k0")
    assert r.exit_code == 2 and not (tmp_path / "k0").exists()


# ---- V13 / G2 D4: pack names only, checked in a subprocess ---------------------------------------------------------------
def _client(root: Path):
    pytest.importorskip("agent_env")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from crucible.studio.explorer import CrucibleExplorerPlugin

    app = FastAPI()
    app.include_router(CrucibleExplorerPlugin(root=root).router)
    return TestClient(app)


def test_v13_explorer_pack_names_are_index_names_and_never_run_here(tmp_path, monkeypatch):
    c = _client(tmp_path / "root")
    evil = tmp_path / "cwd" / "evilpk"
    shutil.copytree(PACKS / "_template", evil, ignore=shutil.ignore_patterns("__pycache__"))
    marker = tmp_path / "ran.txt"
    for f in evil.rglob("__init__.py"):
        f.write_text(f"open({str(marker)!r}, 'w').write(str(__import__('os').getpid()))\n" + f.read_text())
    monkeypatch.chdir(tmp_path / "cwd")
    for name in ("%2E%2E", "%2E", "evilpk"):
        r = c.get(f"/api/v1/crucible/packs/{name}/check")
        assert r.status_code in (400, 404), (name, r.status_code, r.text[:200])
    assert c.get("/api/v1/crucible/packs/%2E%2E/check").status_code == 400
    assert not marker.exists()  # the pack's code never ran in the explorer process
    r = c.get("/api/v1/crucible/packs/toy/check")
    assert r.status_code == 200 and r.json()["ok"]
    assert not marker.exists() or marker.read_text() != str(os.getpid())


# ---- V14 / G2 D2 D3 D5: runs by path, contained reads, sandboxed HTML, no writes on GET ----------------------------------
def test_v14_explorer_runs_reads_and_reports(tmp_path, good_build, monkeypatch):
    root = tmp_path / "root"
    for sub in ("a", "b"):
        shutil.copytree(good_build[0] / "runs" / "bench-toy-build", root / sub / "dup", symlinks=True)
    shutil.copytree(good_build[0] / "runs" / "bench-toy-build", root / "x" / "accept" / "h" / "a11" / "sub", symlinks=True)  # M9 sub-run
    (tmp_path / "outside.txt").write_text("OUTSIDE-ROOT-CANARY-123")
    rep = root / "b" / "dup" / "bench" / "report.html"
    rep.unlink()
    rep.symlink_to(tmp_path / "outside.txt")
    (root / "a" / "dup" / "bench" / "report.html").write_text('<script>fetch("/api/v1/runs",{method:"POST"})</script>')
    c = _client(root)
    ids = [r["run_id"] for r in c.get("/api/v1/crucible/runs").json()]
    assert ids == ["a/dup", "b/dup"]  # unique, addressable; the accept sub-run is hidden
    assert c.get("/api/v1/crucible/runs/a/dup/summary").json()["run_id"] == "a/dup"
    assert c.get("/api/v1/crucible/runs/b/dup/summary").status_code == 403  # it links outside the root
    r = c.get("/api/v1/crucible/runs/b/dup/report")
    assert r.status_code == 403 and "OUTSIDE-ROOT-CANARY" not in r.text
    r = c.get("/api/v1/crucible/runs/a/dup/report")
    assert r.status_code == 200 and r.headers["content-security-policy"] == "sandbox" and r.headers["x-content-type-options"] == "nosniff"
    (root / "a" / "dup" / "bench" / "report.html").unlink()
    r = c.get("/api/v1/crucible/runs/a/dup/report")
    assert r.status_code == 200 and not (root / "a" / "dup" / "bench" / "report.html").exists()  # a GET writes nothing
    r = c.get("/api/v1/crucible/runs/zzz/summary")
    assert r.status_code == 404 and str(tmp_path) not in r.text
    (root / "a" / "dup" / "results.jsonl").write_text("not json\n")
    assert c.get("/api/v1/crucible/runs/a/dup/summary").status_code == 422
    shutil.copytree(good_build[0] / "runs" / "bench-toy-build", root / "c" / "dup", symlinks=True)
    (root / "c" / "dup" / "results.jsonl").unlink()
    (root / "c" / "dup" / "results.jsonl").symlink_to(root / "a" / "dup" / "results.jsonl")
    import crucible.studio.explorer as ex

    monkeypatch.setattr(ex, "CACHE_S", 0.0)  # the listing re-walks
    listed = {x["run_id"]: x["results"] for x in c.get("/api/v1/crucible/runs").json()}
    assert listed.get("c/dup", 0) is None and "a/dup" in listed  # one unreadable run never fails the listing


# ---- V15 / G2 D9: key files are owner-only; an invalid env key is an error ------------------------------------------------
def test_v15_key_file_permissions_and_d9_invalid_env_key(tmp_path, monkeypatch):
    from crucible.bench.run import bench_key

    monkeypatch.delenv("CRUCIBLE_BENCH_KEY", raising=False)
    monkeypatch.delenv("CRUCIBLE_BENCH_KEY_FILE", raising=False)
    k = bench_key(tmp_path)
    f = tmp_path / "bench.key"
    assert oct(f.stat().st_mode & 0o777) == "0o600" and bench_key(tmp_path) == k
    f.chmod(0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        bench_key(tmp_path)
    kf = tmp_path / "shared.key"
    kf.write_text(os.urandom(32).hex())
    kf.chmod(0o666)
    with pytest.raises(PermissionError):
        bench_key(tmp_path / "o", key_file=kf)
    kf.chmod(0o600)
    link = tmp_path / "link.key"
    link.symlink_to(kf)
    with pytest.raises(PermissionError, match="symlink"):
        bench_key(tmp_path / "o2", key_file=link)
    monkeypatch.setenv("CRUCIBLE_BENCH_KEY", "deadbeef")
    with pytest.raises(ValueError):
        bench_key(tmp_path / "o3")
    assert not (tmp_path / "o3" / "bench.key").exists()


# ---- V16: --json errors, exit codes, internal errors ----------------------------------------------------------------
def test_v16_cli_error_contract(tmp_path, monkeypatch):
    for args in (("build", "nosuchpack"), ("export", "--to", "nemo-gym", "--pack", "nosuchpack"), ("pack", "check", "nosuchpack"),
                 ("bench", "run", "nosuchpack"), ("accept", "nosuchpack")):
        r = run(*args, "--json")
        assert r.exit_code == 4, (args, r.exit_code)
        e = out_json(r)["error"]
        assert e["kind"] == "user" and "no pack" in e["message"]
    demo = tmp_path / "demo"
    run("pack", "new", "demo", "--dir", demo)
    (demo / "pack.yaml").write_text((demo / "pack.yaml").read_text().replace("license: Apache-2.0", "license: unknown"))
    assert run("pack", "check", demo).exit_code == 1  # a failed check is 1, an unknown pack 4
    import crucible.studio.pack as sp

    def broken(ref):
        raise KeyError("internal_field")

    monkeypatch.setattr(sp, "find_pack", broken)
    r = run("build", "toy", "--json")
    assert r.exit_code == 3 and out_json(r)["error"]["kind"] == "internal" and "Traceback" in r.stderr


# ---- V17 / G2 D12: every command records its resolved config, scrubbed and stamped ---------------------------------
def test_v17_resolved_configs_with_provenance(tmp_path):
    r = run("export", "--to", "canonical", "--pack", "finance_ap", "--out", tmp_path / "can")
    assert r.exit_code == 0, r.output
    r = run("ingest", "toy", "--out", tmp_path / "store")
    assert r.exit_code == 0
    for f in (tmp_path / "can" / "configs" / "resolved" / "export-canonical.json", tmp_path / "store" / "configs" / "resolved" / "ingest.json"):
        doc = json.loads(f.read_text())
        assert {"crucible", "git_sha", "python"} <= set(doc["provenance"]) and doc["provenance"]["pack"]["content_sha256"]
    man = json.loads((tmp_path / "can" / "crucible_export.json").read_text())
    assert man["provenance"]["pack"]["name"] == "finance_ap"
    from crucible.studio.runs import record

    p = record(tmp_path / "r", "demo", {"api_key": FAKE_KEY, "k": 1})
    assert FAKE_KEY not in p.read_text()


# ---- V18: hidden answers never on argv, never echoed -------------------------------------------------------------------
def test_v18_world_validate_forbidden_from_a_file_and_masked(tmp_path):
    f = tmp_path / "forbidden.txt"
    f.write_text("HIDDEN-ANSWER-4242\n")
    r = run("world", "validate", PACKS / "_toy" / "worlds" / "helpdesk.yaml", "--pack-root", PACKS / "_toy", "--forbidden", f,
            "--instruction", "Resolve the ticket. HIDDEN-ANSWER-4242", "--out", tmp_path / "rep.json")
    assert "<forbidden#0>" in r.output and "HIDDEN-ANSWER-4242" not in r.output
    assert "HIDDEN-ANSWER-4242" not in (tmp_path / "rep.json").read_text()


# ---- V19: train refuses what it cannot honour --------------------------------------------------------------------------
def test_v19_train_refuses_unsupported_specs(tmp_path):
    job = tmp_path / "job.toml"
    job.write_text('[model]\nname = "some-org/some-7b-model"\n[run]\nsteps = 1\nmax_wall_s = 5\n')
    r = run("train", job, "--work", tmp_path / "w")
    assert r.exit_code == 4 and "toy launcher" in r.output and "[model]" in r.output and not (tmp_path / "w").exists()


# ---- V20: canonical export goes on past a failing task ----------------------------------------------------------------
def test_v20_canonical_export_reports_a_failing_task_and_continues(tmp_path, monkeypatch):
    import crucible.interop.replay as rp

    orig, calls = rp.replay, []

    async def flaky(task, *a, **k):
        calls.append(task.id)
        if len(calls) == 1:
            raise RuntimeError("replay crashed")
        return await orig(task, *a, **k)

    monkeypatch.setattr(rp, "replay", flaky)
    r = run("export", "--to", "canonical", "--pack", "finance_ap", "--out", tmp_path / "can", "--json")
    assert r.exit_code == 1
    j = out_json(r)
    assert len(j["errors"]) == 1 and len(j["tasks"]) == 1 and "replay crashed" in next(iter(j["errors"].values()))


# ---- G2 D10: next-step hints are paths, never bare names ----------------------------------------------------------------
def test_d10_pack_new_hints_are_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = run("pack", "new", "bar", "--dir", "toy")
    assert r.exit_code == 0 and "next: crucible pack check ./toy" in r.output


# ---- V21: docs say what the code does -----------------------------------------------------------------------------------
def test_v21_docs_match_the_code():
    doc = (CRUCIBLE / "docs" / "modules" / "M15-studio.md").read_text()
    readme = (CRUCIBLE / "README.md").read_text()
    assert "5 accepted" not in doc and "provisional" in doc
    assert "model-free checks" not in readme and "full contract" in readme


# ==== the public site (H2 V22-V25) ========================================================================================
# M15 audit (H2 V22-V25): the public site follows the user's rules for public material, and stays that way.
# 
# Every visible string of ``site/index.html``, ``site/page.html`` and ``site/assets/*.js`` (text nodes outside script and
# style, aria-label / alt / title / placeholder attributes, the meta description, CSS ``content:`` strings, and the JS
# string literals the figures draw) is checked for:
# - no other framework, environment suite, protocol, algorithm or third-party model, named or alluded to;
# - ASD-STE100 copy: no colon, semicolon, ``--`` or em / en dash; no sentence over 25 words;
# - the formal register: no first or second person;
# - no module codes (M0-M15, A1-A18, RLM) anywhere a reader can see them, including URL anchors, scene ids and JS comments;
# - one attack count (the copy states none: the suites differ in size per pack);
# - ``index.html`` is exactly ``site/build.py``'s output, with ``<title>`` and the meta description in ``<head>``.
# Code shown as code (``<pre>``, ``<code>``, the terminal replay) is exempt from the punctuation and length rules only.
SITE = Path(os.environ.get("CRUCIBLE_SITE_DIR") or Path(__file__).resolve().parents[2] / "site")
_SITE_ONLY = pytest.mark.skipif(not (SITE / "page.html").exists(), reason="site/ not present")

DENY = [  # other frameworks, suites, protocols, algorithms, models and companies, and the known allusions to them
    r"\bMCP\b", r"model context protocol", r"standard tool protocol", r"tool protocol", r"\bGRPO\b", r"group[- ]relative", r"\bPPO\b",
    r"\bDPO\b", r"\bKTO\b", r"\bSFT\b", r"\bRLM\b", r"recursive language model", r"recursive agent", r"recursive self", r"sub-?models?\b",
    r"sub-?calls?\b", r"on-policy distillation", r"\bOpenAI\b", r"\bGPT", r"\bClaude\b", r"\bAnthropic\b", r"\bQwen", r"\bLlama",
    r"\bGemini\b", r"\bDeepSeek", r"\bMistral", r"\bHarbor\b", r"\bNeMo", r"\bOpenEnv\b", r"\bverifiers\b", r"Terminal[- ]Bench",
    r"SWE[- ]bench", r"\btau-?bench", r"\bagentenv\b", r"\bverl\b", r"\bTRL\b", r"\bSkyRL\b", r"\brLLM\b", r"\bslime\b", r"\bAReaL\b",
    r"\bPrime\b", r"\bOpenRLHF\b", r"\bvLLM\b", r"\bSGLang\b", r"\bPyTorch\b", r"\bDocker\b", r"\bKubernetes\b", r"\bHugging ?Face\b",
    r"\blaborator", r"\blabs?\b", r"\bWebGL\b", r"\bGym\b", r"\bMiMo\b", r"\bKimi\b", r"\bpaper\b", r"et al\b",
    r"software engineering",
]
CASED = {r"\bPrime\b", r"\bGym\b"}  # plain words in lower case
PRONOUNS = re.compile(r"\b(we|our|ours|us|you|your|yours)\b", re.IGNORECASE)
CODES = re.compile(r"\bM(?:1[0-5]|[0-9])\b|\bA(?:1[0-8]|[1-9])\b|\bRLM\b|\brlm\b")
SCENE_CODE = re.compile(r"^(?:m\d{1,2}|rlm|fsm|lh)$")
PUNCT = re.compile(r"[:;—–]|--")


class _Page(HTMLParser):
    """Visible strings with their kind; ``code`` marks text inside pre / code (exempt from punctuation and length)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[tuple[int, str, str, bool]] = []
        self.skip = 0
        self.code = 0
        self.ids: list[tuple[int, str, str]] = []
        self.head_open = self.in_head = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        line = self.getpos()[0]
        if tag in ("style", "script"):
            self.skip += 1
        if tag in ("pre", "code"):
            self.code += 1
        if tag == "meta" and a.get("name") == "description":
            self.out.append((line, "meta", a.get("content") or "", False))
        for k in ("alt", "aria-label", "title", "placeholder"):
            if a.get(k):
                self.out.append((line, k, a[k], False))
        for k in ("id", "data-scene"):
            if a.get(k):
                self.ids.append((line, k, a[k]))
        if a.get("href", "").startswith("#"):
            self.ids.append((line, "href", a["href"][1:]))

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.skip = max(0, self.skip - 1)
        if tag in ("pre", "code"):
            self.code = max(0, self.code - 1)

    def handle_data(self, data):
        if self.skip:
            return
        t = " ".join(data.split())
        if t:
            self.out.append((self.getpos()[0], "text", t, bool(self.code)))


def _html(path: Path) -> _Page:
    p = _Page()
    src = path.read_text()
    p.feed(src)
    for i, ln in enumerate(src.splitlines(), 1):
        for m in re.finditer(r'content:\s*"([^"]+)"', ln):
            p.out.append((i, "css", m.group(1), False))
    return p


def _js(src: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """(string and template literals, comments) of a JS file, with lines; regex literals are skipped."""
    lits, comments = [], []
    i, n, line, prev = 0, len(src), 1, ""
    while i < n:
        c = src[i]
        if c == "\n":
            line, i = line + 1, i + 1
            continue
        if c in " \t\r":
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            comments.append((line, src[i:j]))
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2) + 2
            comments.append((line, src[i:j]))
            line += src[i:j].count("\n")
            i = j
            continue
        if c in "'\"`":
            j, buf, start = i + 1, [], line
            while j < n and src[j] != c:
                if src[j] == "\\":
                    buf.append(src[j + 1])
                    j += 2
                    continue
                if c == "`" and src.startswith("${", j):
                    k, d = j + 2, 1
                    while k < n and d:
                        d += {"{": 1, "}": -1}.get(src[k], 0)
                        k += 1
                    buf.append(" X ")
                    j = k
                    continue
                if src[j] == "\n":
                    line += 1
                buf.append(src[j])
                j += 1
            lits.append((start, "".join(buf)))
            i, prev = j + 1, "a"
            continue
        if c == "/" and (prev == "" or prev in "(,=:[!&|?{};+-*%<>~^"):
            j, cls = i + 1, False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    cls = True
                elif src[j] == "]":
                    cls = False
                elif src[j] == "/" and not cls:
                    break
                j += 1
            i, prev = j + 1, "a"
            continue
        if c.isalnum() or c in "_$":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            prev = "(" if src[i:j] in ("return", "typeof", "case", "in", "of") else "a"
            i = j
            continue
        prev = c if c not in ")]" else "a"
        i += 1
    return lits, comments


def _prose(s: str) -> bool:
    """A JS literal a reader sees as words (labels, captions, keys), not code (selectors, ids, materials, paths)."""
    s = s.strip()
    if not re.search(r"[A-Za-z]{2,}", s) or re.fullmatch(r"[\w.#\-/:\[\]=\"'^>*@ ]*", s) and " " not in s:
        return False
    if s.startswith(("http", "./", "three", "(", "rgba", "translate", "&")) or re.fullmatch(r"[.#\w\-\[\]=\"^ ]+", s) and not re.search(r"[A-Z]?[a-z]+ [a-z]+", s):
        return False
    return not re.search(r"\$\{|=>|\bconst\b|class=|<span|</", s)


def _strip_tags(t: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", t).split())


def _terminal_lines(src: str) -> set[int]:
    """Lines of main.js inside the SESSION replay (real CLI output, shown as code)."""
    a = src.find("const SESSION")
    if a < 0:
        return set()
    b = src.find("];", a)
    return set(range(src[:a].count("\n") + 1, src[:b].count("\n") + 2))


def visible() -> list[tuple[str, int, str, str, bool]]:
    """(file, line, kind, text, is_code) for every visible string."""
    out = []
    for name in ("page.html", "index.html"):
        if (SITE / name).exists():
            out += [(name, ln, k, t, code) for ln, k, t, code in _html(SITE / name).out]
    for f in sorted((SITE / "assets").glob("*.js")):
        src = f.read_text()
        term = _terminal_lines(src) if f.name == "main.js" else set()
        for ln, s in _js(src)[0]:
            t = _strip_tags(s)
            if _prose(t):
                out.append((f"assets/{f.name}", ln, "js", t, ln in term))
    return out


def _sentences(t: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def _fail(hits: list[str], what: str) -> None:
    assert not hits, f"{len(hits)} {what}:\n" + "\n".join(hits[:60])


def _all_js_literals() -> list[tuple[str, int, str, str, bool]]:
    """Every JS string literal, prose or not (a figure label such as a bare algorithm acronym is one token)."""
    return [(f"assets/{f.name}", ln, "js-literal", s, False) for f in sorted((SITE / "assets").glob("*.js")) for ln, s in _js(f.read_text())[0]]


@_SITE_ONLY
def test_site_names_no_other_framework_suite_protocol_or_model():
    hits = [f"{f}:{ln} [{rx}] {t!r}" for f, ln, _, t, _ in visible() + _all_js_literals() for rx in DENY
            if re.search(rx, t, 0 if rx in CASED else re.IGNORECASE)]
    _fail(hits, "references to other frameworks, suites, protocols, algorithms or models")


@_SITE_ONLY
def test_site_copy_has_no_banned_punctuation():
    hits = []
    for f, ln, k, t, code in visible():
        if code:
            continue
        bare = re.sub(r"https?://\S+", "", t)
        if PUNCT.search(bare):
            hits.append(f"{f}:{ln} ({k}) {t!r}")
    _fail(hits, "visible strings with a colon, semicolon, -- or a dash")


@_SITE_ONLY
def test_site_copy_has_no_first_or_second_person():
    _fail([f"{f}:{ln} ({k}) {t!r}" for f, ln, k, t, _ in visible() if PRONOUNS.search(t)], "visible strings in the first or second person")


@_SITE_ONLY
def test_site_sentences_are_at_most_25_words():
    hits = []
    for f, ln, k, t, code in visible():
        if code:
            continue
        for s in _sentences(t):
            n = len(re.findall(r"[A-Za-z0-9][\w'’%.\-]*", s))
            if n > 25:
                hits.append(f"{f}:{ln} ({n} words) {s!r}")
    _fail(hits, "sentences over 25 words")


@_SITE_ONLY
def test_site_has_no_module_codes_anywhere_a_reader_can_see():
    hits = [f"{f}:{ln} ({k}) {t!r}" for f, ln, k, t, _ in visible() if CODES.search(t)]
    for name in ("page.html", "index.html"):
        if (SITE / name).exists():
            hits += [f"{name}:{ln} {k}={v!r}" for ln, k, v in _html(SITE / name).ids if SCENE_CODE.match(v) or CODES.search(v)]
    for f in sorted((SITE / "assets").glob("*.js")):
        lits, comments = _js(f.read_text())
        hits += [f"assets/{f.name}:{ln} comment {c.strip()[:100]!r}" for ln, c in comments if CODES.search(c) or re.search(r"\bM\d+\s*[–-]\s*M\d+", c)]
        hits += [f"assets/{f.name}:{ln} id {s!r}" for ln, s in lits if SCENE_CODE.match(s) or re.fullmatch(r"A(?:1[0-8]|[1-9])", s)]
    _fail(hits, "module codes in public material")


@_SITE_ONLY
def test_site_states_one_attack_count():
    counts = {(m.group(1), f) for f, _, _, t, _ in visible() for m in re.finditer(r"(\d+)(?:\s+known)?\s+attack", t, re.IGNORECASE)}
    counts |= {(m.group(1), f) for f, _, _, t, _ in visible() for m in re.finditer(r"more than (\d+) attacks", t, re.IGNORECASE)}
    assert len({c for c, _ in counts}) <= 1, f"the site states different attack counts: {sorted(counts)}"


@_SITE_ONLY
def test_site_index_is_the_build_output_with_head_metadata_in_head():
    spec = importlib.util.spec_from_file_location("_site_build", SITE / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "render"), "site/build.py has no render(): index.html cannot be checked against it"
    idx = (SITE / "index.html").read_text()
    assert idx == mod.render(), "site/index.html is stale: run python site/build.py"
    head, _, body = idx.partition("</head>")
    assert "<title>" in head and 'name="description"' in head, "<title> and the meta description must sit in <head>"
    assert "<title>" not in body and 'name="description"' not in body and "<link" not in body, "head elements left in <body>"
