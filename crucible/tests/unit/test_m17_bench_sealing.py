"""Benchmark v2 (``factory.benchmark``, ``factory.sealing.seal_sources``): one training cutoff, every training source.

A held-out window must end after the training cutoff; its repository and organisation must be apart from every training
source (every window of every training store, the scouted repositories the lease let through, the stitched pairs, clone
directories, the training repository list); training sources that end after the cutoff are the embargo, sealed in the
manifest so the training rows refuse them. S10's empirical timing rule: a checked test whose verdict moves on one end
state is load-dependent, the window is validated again without it and gated once more.

The stores are the window lane's toy window (``test_m17_w_task.make_task``), graded on the host interpreter."""

from __future__ import annotations

import asyncio
import itertools
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m17_w_miner import SB, make_repo
from test_m17_w_task import make_task

from crucible.factory import benchmark as bm
from crucible.factory.benchmark import BuildConfig, GateConfig, build, heldout_windows, run_gates, sealed_ids, training_sources
from crucible.factory.lanes.windows import WindowStore
from crucible.factory.lanes.windows_timing import drop_tests
from crucible.factory.sealing import Source, org_from, repo_key, seal_sources
from crucible.train.bridges.slime.rows import SealedRowError, slime_row, write_rows

FAST = GateConfig(gates=("s5", "s6", "s8"), concurrency=2)
CUT = "2025-01-01T00:00:00+00:00"
LEAK = {"pkg/zeta.py": "def zeta_transform(values, factor):\n    return [v * factor + len(values) for v in values if v % 3]\n"}
URLS = {"voluptuous": "alecthomas/voluptuous", "click": "pallets/click", "jinja": "pallets/jinja"}


def test_repo_keys_and_orgs() -> None:
    assert repo_key("voluptuous", URLS) == "alecthomas/voluptuous" and org_from("voluptuous", URLS) == "alecthomas"
    assert repo_key("fuhrysteve__marshmallow-jsonschema") == "fuhrysteve/marshmallow-jsonschema" and org_from("Textualize/trogon") == "textualize"
    assert org_from("unknownname") == "unknownname"


def test_seal_sources_orgs_repos_cutoff_and_embargo() -> None:
    held = [{"id": "v-late", "repo": "voluptuous", "date": "2025-03-01T00:00:00+00:00"},
            {"id": "s-late", "repo": "sqlparse", "date": "2025-03-01T00:00:00+01:00"},
            {"id": "s-edge", "repo": "sqlparse", "date": "2025-01-01T00:30:00+01:00"},  # 2024-12-31T23:30Z: not after the cutoff
            {"id": "j-late", "repo": "jinja", "date": "2025-03-01"}]
    src = [Source("click", "pallets", "2024-06-01T00:00:00+00:00", "window_store", "train", "click-w1"),
           Source("click", "pallets", "2025-02-01T00:00:00+00:00", "window_store", "train", "click-w2"),
           Source("alecthomas/kong", "alecthomas", None, "scout:consumer", "records.json"),
           Source("someone/sqlparse-ext", "someone", None, "pairs", "pairs.json")]
    rep = seal_sources(held, src, cutoff=CUT, urls=URLS)
    per = rep.per_heldout
    assert per["v-late"]["org_ok"] is False and per["v-late"]["org_sources"] == ["scout:consumer:alecthomas/kong"]
    assert per["s-late"] == {**per["s-late"], "org_ok": True, "repo_ok": True, "time_ok": True}
    assert per["s-edge"]["time_ok"] is False and rep.heldout_not_after_cutoff[0]["heldout"] == "s-edge"
    assert per["j-late"]["org_ok"] is False and set(rep.shared_orgs) == {"alecthomas", "pallets"}
    assert [e["id"] for e in rep.embargo] == ["click-w2"] and rep.counts["embargo"] == 1 and not rep.ok
    # a repository name is matched by owner/name; an owner-less source matches by bare name
    rep2 = seal_sources([{"id": "x", "repo": "sqlparse", "date": "2026-01-01"}], [Source("sqlparse", "sqlparse")], cutoff=CUT, urls=URLS)
    assert rep2.shared_repos == ["sqlparse"] and rep2.problems()


def _put(store: WindowStore, rec: dict, base: Path, **upd) -> dict:
    r = {**json.loads(json.dumps(rec)), "stage": "mutants_valid", "independent": False, **upd}
    d = store.dir(r["id"])
    d.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base, d / "base", symlinks=True)
    store.save_window(r)
    return r


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("benchv2")
    (tmp / "src").mkdir()
    task, rec, _ = make_task(tmp / "src")
    base = Path(task.base_dir)
    train = WindowStore(tmp / "train", split="train")
    _put(train, rec, base, id="click-wtrain", repo="click", b_date="2024-06-01T00:00:00+00:00", split="train", oracle_files={"pkg/zeta.py": "def zeta():\n    return 9\n"})
    _put(train, rec, base, id="click-wafter", repo="click", b_date="2025-02-01T00:00:00+00:00", split="train", stage="leak_ok",
         oracle_files={"pkg/zeta.py": "def zeta():\n    return 8\n"})
    gen = WindowStore(tmp / "gen", split="train")  # another training store (a generator's), its oracle is a held-out solution
    _put(gen, rec, base, id="rb-boltons-1", repo="boltons", b_date="2024-07-01T00:00:00+00:00", split="train", oracle_files=LEAK)
    scout = tmp / "records.json"
    scout.write_text(json.dumps([
        {"type": "consumer", "repo": "alecthomas/kong", "lease": {"ok": True}, "id": "c1"},
        {"type": "consumer", "repo": "andialbrecht/other", "lease": {"ok": False, "rule": "sealed_org"}, "id": "c2"}]))
    held = WindowStore(tmp / "heldout", split="heldout")
    late = "2025-03-01T00:00:00+00:00"
    _put(held, rec, base, id="sqlparse-wlate", repo="sqlparse", b_date=late, split="heldout")
    _put(held, rec, base, id="sqlparse-wearly", repo="sqlparse", b_date="2024-12-01T00:00:00+00:00", split="heldout")
    _put(held, rec, base, id="voluptuous-wlate", repo="voluptuous", b_date=late, split="heldout")
    _put(held, rec, base, id="jinja-worg", repo="jinja", b_date=late, split="heldout")
    _put(held, rec, base, id="sqlparse-wleak", repo="sqlparse", b_date=late, split="heldout", oracle_files=LEAK)
    return SimpleNamespace(tmp=tmp, train=tmp / "train", gen=tmp / "gen", scout=scout, heldout=tmp / "heldout", rec=rec)


def test_training_sources_read_every_kind(stores) -> None:
    src, corpus = training_sources([stores.train, stores.gen, stores.scout])
    kinds = {(s.kind, s.repo) for s in src}
    assert ("window_store", "click") in kinds and ("window_store", "boltons") in kinds and ("scout:consumer", "alecthomas/kong") in kinds
    assert not any(s.repo == "andialbrecht/other" for s in src)  # the lease refused it: never usable, never a source
    assert ("repo_list", "typer") in kinds and not any(s.repo == "sqlparse" for s in src)  # held-out names never enter the list
    assert {c["rec"]["id"] for c in corpus} == {"click-wtrain", "click-wafter", "rb-boltons-1"}
    with pytest.raises(ValueError, match="heldout store"):
        training_sources([stores.heldout])


@pytest.fixture(scope="module")
def built(stores):
    cfg = BuildConfig(heldout_root=stores.heldout, train_root=stores.train, out_dir=stores.tmp / "bench", gates=FAST, sandboxes=lambda _i: SB(),
                      cutoff=CUT, train_sources=(stores.gen, stores.scout), selection={"selected": {"sqlparse": "test"}})
    return asyncio.run(build(cfg, log=lambda _m: None))


def test_v2_build_seals_against_every_source_and_the_cutoff(built) -> None:
    m = built
    by = {t["id"]: t for t in m["tasks"]}
    assert by["sqlparse-wearly"]["excluded"] == ["sealing:time_ok"]
    assert "sealing:org_ok" in by["voluptuous-wlate"]["excluded"]  # alecthomas: a scouted consumer the lease let through
    assert by["voluptuous-wlate"]["sealing"]["org_sources"] == ["scout:consumer:alecthomas/kong"]
    assert "sealing:org_ok" in by["jinja-worg"]["excluded"]
    # the generator store's oracle is a held-out solution: contamination against every training store, not just the train store
    assert by["sqlparse-wleak"]["excluded"][0] == "sealing:contamination_ok"
    assert {h["source"] for h in by["sqlparse-wleak"]["sealing"]["contamination"]} == {"rb-boltons-1:diff"}
    assert m["split_members"] == ["sqlparse-wlate"] and m["sealed"]
    assert m["policy"]["training_cutoff"] == CUT and m["sealing_summary"]["embargoed_sources"] == 1
    assert m["training_embargo"] == ["click-wafter"] and m["selection"] == {"selected": {"sqlparse": "test"}}
    assert m["sources"]["contamination_corpus_windows"] == 3 and m["heldout_funnel"]["mutants_valid"] == 5


def test_embargoed_training_ids_are_sealed_from_the_rows(stores, built, tmp_path: Path) -> None:
    sealed = sealed_ids(stores.tmp / "bench" / "manifest.json")
    assert "click-wafter" in sealed and "sqlparse-wlate" in sealed and "click-wtrain" not in sealed
    ref = {"lane": "window", "task_id": "click-wafter", "source_id": "click-wafter", "store": "/s", "split": "train", "prompt_sha256": "p",
           "checkset_sha256": "c", "verifier_version": "v"}
    row = slime_row(instance_id="click-wafter", prompt="x", image="i", workdir="/w", agent="codex", crucible_task_ref=ref)
    with pytest.raises(SealedRowError):
        write_rows([row], tmp_path / "rows.jsonl", sealed=sealed)


def test_heldout_windows_end_after_the_cutoff(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    got = heldout_windows(repo, "toyrepo", cutoff="2024-01-03T00:00:00+00:00", min_units=2)
    assert [w.b_ref for w in got] == ["v0.2"] and got[0].units[0].body is not None
    assert heldout_windows(repo, "toyrepo", cutoff="2024-01-06T00:00:00+00:00", min_units=2) == []


class _MovingTask:
    """An oracle whose check ``t::b`` fails every 7th grade: one end state, a verdict that moves."""

    task_id = "moving"

    def __init__(self) -> None:
        self.n = itertools.count()
        self.milestones: list = []
        self.check_tests = {"c_m01": ["t::a", "t::b"]}

    def oracle_patch(self) -> str:
        return "patch"

    async def grade_patch(self, patch: str, trail=None) -> SimpleNamespace:
        fail = ["t::b"] if next(self.n) % 7 == 3 else []
        return SimpleNamespace(infra={"status": "ok"}, reward=0.0 if fail else 1.0, gates={"c_m01": "fail" if fail else "pass"}, meta={"failing_tests": fail})


def test_s10_names_load_dependent_tests() -> None:
    out = asyncio.run(run_gates(_MovingTask(), {}, cfg=GateConfig(gates=("s10",), flake_reruns_per_level=14, concurrency=2)))
    s10 = out["s10"]
    assert not s10["ok"] and set(s10["load_dependent"]) == {"t::b"} and s10["load_dependent"]["t::b"] == "S10: failed 4 of 28 oracle grades of one end state"


def test_a_load_dependent_test_is_dropped_and_the_window_gated_again(stores, monkeypatch) -> None:
    calls: list[str] = []
    victim = stores.rec["milestones"][0]["tests"][0]

    async def fake_gates(task, rec, *, cfg=None, log=lambda _m: None):
        calls.append(rec.get("revalidated") and "after" or "before")
        if not rec.get("revalidated"):
            return {"ok": False, "s10": {"ok": False, "load_dependent": {victim: "S10: failed 2 of 60"}}}
        return {"ok": True, **{g: {"ok": True} for g in cfg.gates}}

    async def fake_revalidate(rec, store, repo_dir, exclude, **_kw):
        new = json.loads(json.dumps(rec))
        drop_tests(new, exclude)
        new["revalidated"] = {"exclude": dict(exclude)}
        store.save_window(new)
        return new

    monkeypatch.setattr(bm, "run_gates", fake_gates)
    monkeypatch.setattr(bm, "revalidate", fake_revalidate)
    held = WindowStore(stores.tmp / "held2", split="heldout")
    _put(held, stores.rec, Path(stores.heldout / "tasks" / "sqlparse-wlate" / "base"), id="tomlkit-wmove", repo="tomlkit",
         b_date="2025-03-01T00:00:00+00:00", split="heldout")
    cfg = BuildConfig(heldout_root=stores.tmp / "held2", train_root=stores.train, out_dir=stores.tmp / "bench2", gates=FAST, sandboxes=lambda _i: SB(),
                      cutoff=CUT, repo_dirs={"tomlkit": stores.tmp}, train_sources=())
    m = asyncio.run(build(cfg, log=lambda _m: None))
    (t,) = m["tasks"]
    assert calls == ["before", "after"] and t["revalidated"] and t["in_split"], t
    assert victim not in {x for ms in held.window("tomlkit-wmove")["milestones"] for x in ms["tests"]}
    g = json.loads((stores.tmp / "bench2" / "gates" / "tomlkit-wmove.json").read_text())
    assert g["revalidated"] == {"load_dependent": {victim: "S10: failed 2 of 60"}}


def test_a_one_line_idiom_is_recorded_not_blocking() -> None:
    from crucible.factory.benchmark import seal

    idiom = "    return get_type_hints(obj, globalns, localns, include_extras=True)\n"
    held_sol = "def resolve_hints(obj, globalns, localns):\n    cache_key = (id(obj), bool(globalns))\n" + idiom + "    # end of resolver\n"
    copied = "def render_signature(app, what, name, obj):\n    params = list(collect_parameters(obj))\n    for p in params:\n        annotate(p)\n" * 3
    train_sol = "def typer_hints(obj, globalns=None, localns=None):\n    seen = set()\n" + idiom + "    # done\n"
    train = [{"id": "typer-w1", "repo": "typer", "b_date": "2024-01-01"}, {"id": "typer-w2", "repo": "typer", "b_date": "2024-01-02"}]
    held = [{"id": "sat-w1", "repo": "sphinx-autodoc-typehints", "b_date": "2026-08-01"}, {"id": "sat-w2", "repo": "sphinx-autodoc-typehints", "b_date": "2026-08-01"}]
    diffs = {"typer-w1": train_sol, "typer-w2": copied + "x = 1\n", "sat-w1": held_sol, "sat-w2": "y = 2\n" + copied}
    strict = seal(train, held, diffs=diffs, cutoff="2026-07-03T00:00:00+00:00")
    assert not strict["per_task"]["sat-w1"]["contamination_ok"] and not strict["per_task"]["sat-w2"]["contamination_ok"]  # P6's rule: both block
    v2 = seal(train, held, diffs=diffs, cutoff="2026-07-03T00:00:00+00:00", idiom_chars=100)
    one, two = v2["per_task"]["sat-w1"], v2["per_task"]["sat-w2"]
    assert one["contamination_ok"] and one["contamination_idioms"][0]["source"] == "typer-w1:diff" and one["contamination_idioms"][0]["shared_chars"] < 100
    assert not two["contamination_ok"] and two["contamination"][0]["source"] == "typer-w2:diff"  # a copied block still blocks
