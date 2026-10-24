"""The benchmark seal (factory.benchseal) and the held-out supply ranking (factory.heldout_supply)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.factory import benchseal as bs
from crucible.factory import heldout_supply as hs
from crucible.factory.sources.lease import EvalLease

NOISE = {"metric": "pass", "n_tasks": 11, "k": 8, "sigma2": 0.0472, "control_mean": 0.13}


def store(tmp: Path, ids: list[str]) -> dict[str, Path]:
    out = {}
    for i in ids:
        d = tmp / "heldout" / "tasks" / i
        d.mkdir(parents=True)
        (d / "window.json").write_text(json.dumps({"id": i, "repo": "zzlib", "oracle_files": {"a.go": "x"},
                                                   "milestones": [{"tests": [f"T{i}a", f"T{i}b"]}]}))
        out[i] = d
    return out


def audit(rows: dict[str, str], ref: str = "origin/main") -> dict:
    return {"ref": ref, "tasks": [{"id": i, "verdict": v, "why": [] if v == "sealed" else ["public report"]} for i, v in rows.items()]}


def test_seal_refuses_with_every_reason(tmp_path: Path) -> None:
    ids = ["db-go-zzlib-00000001", "db-go-zzlib-00000002", "db-go-zzlib-00000003"]
    loc = store(tmp_path, ids[:2])
    with pytest.raises(bs.SealRefused) as e:
        bs.seal("v3", ids, audit({ids[0]: "sealed", ids[1]: "contaminated"}, ref="HEAD"), out_root=tmp_path / "sealed", locate=loc,
                gates_ok={ids[0]: True, ids[1]: True}, repos={}, fix_dates={}, noise=NOISE, ref="origin/main")
    msg = str(e.value)
    for want in ("audit ran on 'HEAD'", f"{ids[1]}: contaminated", f"{ids[2]}: not in the audit", f"{ids[2]}: in no held-out store",
                 f"{ids[2]}: its gates have not all passed"):
        assert want in msg, msg
    assert not (tmp_path / "sealed").exists()  # a refusal writes nothing


def test_seal_writes_hashes_fresh_seeds_the_lease_and_the_power_plan(tmp_path: Path) -> None:
    ids = [f"db-go-zzlib-0000000{n}" for n in range(1, 4)]
    loc = store(tmp_path, ids)
    res = bs.seal("v3", ids, audit(dict.fromkeys(ids, "sealed")), out_root=tmp_path / "sealed", locate=loc, gates_ok=dict.fromkeys(ids, True),
                  repos=dict.fromkeys(ids, "acme/zzlib"), fix_dates={i: ["2026-06-01"] for i in ids}, noise=NOISE, ref="origin/main", master=12345)
    out = Path(res["dir"])
    man = json.loads((out / "manifest.json").read_text())
    raw = (out / "manifest.json").read_text()
    assert "Tdb-go-zzlib-00000001a" not in raw and man["tasks"][ids[0]]["hashes"]["n_hidden_tests"] == "2"  # hashes, not tests
    seeds = {t["seed"] for t in man["tasks"].values()}
    assert len(seeds) == 3 and seeds == {bs.task_seed(12345, i) for i in ids}
    pub = json.loads((out / "public.json").read_text())
    assert "master_seed" not in pub and pub["seed_commitment"] and pub["power"]["verdict"] == "underpowered"  # 3 tasks < 10
    lease = EvalLease.default(out / "sealed.json")
    assert not lease.check({"repo": "acme/zzlib"})["ok"] and not lease.check({"repo": "acme/other"})["ok"]  # the org too
    assert lease.after is None and lease.check({"repo": "someone/else", "created": "2026-09-01"})["ok"]  # no time cut by default
    with pytest.raises(bs.SealRefused, match="sealed already"):
        bs.seal("v3", ids, audit(dict.fromkeys(ids, "sealed")), out_root=tmp_path / "sealed", locate=loc, gates_ok=dict.fromkeys(ids, True),
                repos={}, fix_dates={}, noise=NOISE, ref="origin/main")


def test_seal_states_the_mde_and_can_require_power(tmp_path: Path) -> None:
    p = bs.plan(14, 8, NOISE)
    assert p["mde"] < bs.plan(10, 8, NOISE)["mde"] and "MDE" in p["headline"]
    ids = ["db-go-zzlib-00000001"]
    loc = store(tmp_path, ids)
    with pytest.raises(bs.SealRefused, match="underpowered"):
        bs.seal("v4", ids, audit(dict.fromkeys(ids, "sealed")), out_root=tmp_path / "s", locate=loc, gates_ok=dict.fromkeys(ids, True), repos={},
                fix_dates={}, noise=NOISE, require_power=True)


# ---- the supply ---------------------------------------------------------------------------------------------------------
LOG = """@@aaaaaaaaaaaa\tfix(parser): handle empty input
src/parse.go
src/parse_test.go
@@bbbbbbbbbbbb\tfix: typo in README
README.md
@@cccccccccccc\tfeat: add option
src/opt.go
src/opt_test.go
@@dddddddddddd\tFix crash when the config is missing
lib/config.py
tests/test_config.py
@@eeeeeeeeeeee\tfix(docs): wrong example
docs/a.md
"""


def test_fix_commits_count_only_fixes_that_change_tests_and_sources() -> None:
    commits, fixes, with_tests, shas = hs.fix_commits(LOG)
    assert (commits, fixes, with_tests) == (5, 3, 2) and shas == ["aaaaaaaaaa", "dddddddddd"]


def test_refusal_rules() -> None:
    seen = {"usedowner", "other/lib"}
    def c(slug: str, lic: str = "mit") -> hs.Candidate:
        return hs.Candidate(slug, "go", lic, 1000, "2026-10-20")
    owners = {"registered"}
    assert hs.refuse(c("fresh/lib", "gpl-3.0"), owners=owners, seen_fn=seen.__contains__).startswith("licence")
    assert hs.refuse(c("registered/x"), owners=owners, seen_fn=seen.__contains__) == "owner of a registered source"
    assert hs.refuse(c("usedowner/x"), owners=owners, seen_fn=seen.__contains__) == "named in our git history"
    assert hs.refuse(c("fresh/lib"), owners=owners, seen_fn=seen.__contains__) is None


def test_search_results_keep_the_lanes_languages_and_drop_forks() -> None:
    items = [{"full_name": "a/b", "language": "Go", "license": {"spdx_id": "MIT"}, "stargazers_count": 900, "pushed_at": "x"},
             {"full_name": "c/d", "language": "C++", "license": {"spdx_id": "MIT"}},
             {"full_name": "e/f", "language": "Rust", "license": {"spdx_id": "Apache-2.0"}, "fork": True}]
    got = hs.from_search(items)
    assert [(x.slug, x.lang, x.licence) for x in got] == [("a/b", "go", "mit")]


def test_seen_many_finds_owners_in_files_and_commit_messages(tmp_path: Path) -> None:
    import subprocess

    r = tmp_path / "r"
    r.mkdir()
    for a in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(r), *a], check=True)
    (r / "notes.md").write_text("we looked at Acme/ZZlib and the owner widgetco\n")
    (r / "data" / "llm_cache").mkdir(parents=True)
    (r / "data" / "llm_cache" / "x.json").write_text('{"text": "see cachedowner/lib"}')
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "commit", "-qm", "mention msgowner here"], check=True)
    got = hs.seen_many(r, ["acme/zzlib", "widgetco", "msgowner", "cachedowner/lib", "nobody", "widget"])
    assert got == {"acme/zzlib", "widgetco", "msgowner", "cachedowner/lib"}  # whole words: 'widget' is not 'widgetco'


def test_draw_only_grows_and_feeds_the_local_draw(tmp_path: Path) -> None:
    from crucible.factory.lanes import windows

    p = tmp_path / "heldout_draw.json"
    hs.draw(p, [hs.Candidate("acme/zzlib", "go", "mit", 1, "")])
    hs.draw(p, [hs.Candidate("acme2/qq", "rust", "mit", 1, "")])
    assert windows.local_draw(p) == {"zzlib": "acme/zzlib", "qq": "acme2/qq"}
