"""P5 shared machinery (swarm-pipeline/00 §1, §4): the Generator Protocol, synthetic history, placeholder requirements,
the deterministic gate order, authoring in a generator's voice, and the funnel."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_m17_p5_rebreak import make_repo

from crucible.factory.generators.base import (
    AuthorStyle,
    Candidate,
    Funnel,
    GateConfig,
    Generator,
    author_generated,
    combine_trees,
    compose_prompt,
    deterministic_gates,
    gate_requirements,
    git,
    init_mono,
    out,
    tree_of,
    with_files,
    write_chain,
)
from crucible.factory.generators.open import OpenGen
from crucible.factory.generators.rebreak import STYLE, Rebreak
from crucible.factory.generators.stitch import Stitch
from crucible.factory.lanes.windows_git import RepoGit, first_parent_units
from crucible.factory.sources.lease import EvalLease


def test_every_generator_is_a_generator(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    f, lease = Funnel(tmp_path / "f.jsonl"), EvalLease.default()
    gens = [Rebreak(tmp_path / "rb", {"mylib": repo}, funnel=f, lease=lease), Stitch(tmp_path / "st", tmp_path, funnel=f, lease=lease),
            OpenGen(tmp_path / "op", {"mylib": repo}, funnel=f, lease=lease)]
    assert all(isinstance(g, Generator) for g in gens) and [g.name for g in gens] == ["rebreak", "stitched_window", "open"]
    assert Candidate("c", "rebreak", "mylib").to_json()["params"] == {}


def test_synthetic_history_trees_and_commits(tmp_path: Path) -> None:
    repo, shas = make_repo(tmp_path)
    t0 = tree_of(repo, shas["init"])
    t1 = with_files(repo, t0, {"mylib/new.py": "X = 1\n", "CHANGES.rst": None})
    assert out(git(repo, "show", f"{t1}:mylib/new.py")) == "X = 1" and git(repo, "show", f"{t1}:CHANGES.rst", check=False).returncode != 0
    commits = write_chain(repo, [t0, t1, t0], messages=["a", "b", "c"], dates=["2024-01-01T00:00:00+00:00"] * 3, parent=None, ref="refs/crucible/t/x")
    units = first_parent_units(repo, commits[0], commits[-1])
    assert [u.subject for u in units] == ["b", "c"] and out(git(repo, "rev-parse", "refs/crucible/t/x")) == commits[-1]
    again = write_chain(repo, [t0, t1, t0], messages=["a", "b", "c"], dates=["2024-01-01T00:00:00+00:00"] * 3, parent=None, ref="refs/crucible/t/y")
    assert again == commits  # deterministic identity and dates: the same history twice is the same commits
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    (other / "o.py").write_text("o = 1\n")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "o")
    mono = init_mono(tmp_path / "mono", [repo, other])
    t = combine_trees(mono, {"host": shas["init"], "consumer": out(git(other, "rev-parse", "HEAD"))})
    assert out(git(mono, "show", f"{t}:consumer/o.py")) == "o = 1" and out(git(mono, "show", f"{t}:host/mylib/core.py")).startswith("def parse")
    built, _ = RepoGit(mono).build(t, [], tmp_path / "wt")
    assert built and (tmp_path / "wt" / "host" / "mylib" / "core.py").exists()


def test_placeholder_requirements_bind_the_checks_authoring_keeps() -> None:
    rec = {"milestones": [{"id": "m01", "subject": "Fix a", "tests": ["t::a"]}, {"id": "m03", "subject": "Fix b", "tests": ["t::b"]}], "p2p": ["t::c"]}
    reqs = gate_requirements(rec)
    assert [(r["id"], r["milestone"]) for r in reqs] == [("R1", "m01"), ("R2", "m03"), ("r_regress", None)]
    assert gate_requirements({**rec, "p2p": [], "collateral": []})[-1]["id"] == "R2"


class FailingS6:
    """A task with no executed edge: S6 fails, and with stop_on_fail no later gate runs (nothing here could run them)."""

    task_id = "t"
    generator = "rebreak"
    def __init__(self) -> None:
        self.window: dict[str, Any] = {}
        self.milestones = [{"id": "m01", "tests": ["t::a"], "requires": []}, {"id": "m02", "tests": ["t::b"], "requires": []}]


def test_gates_run_cheap_first_and_stop_at_the_first_failure() -> None:
    res = asyncio.run(deterministic_gates(FailingS6(), cfg=GateConfig()))
    assert list(res["gates"]) == ["S6", "S5", "S8", "S3", "S10"] and res["gates"]["S6"]["ok"] is False and not res["ok"]
    assert all(res["gates"][g]["ok"] is None and res["gates"][g]["findings"] == ["not_run: an earlier gate failed"] for g in ("S5", "S8", "S3", "S10"))
    assert res["gates"]["S6"]["evidence"]["executed_edges"] == 0


def test_s6_lets_a_window_built_for_separate_bugs_go_without_an_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """subset --no-link (the lead's call, 2026-10-23): a window recorded allow_independent passes S6 with no edge."""
    # the contract check reduced to its edge rule: "short" only when an edge is required
    monkeypatch.setattr("crucible.factory.longtask.check_longtask", lambda task, long_horizon=True, need_prompt=True: ["short: no edge"] if long_horizon else [])
    strict = asyncio.run(deterministic_gates(FailingS6(), cfg=GateConfig(gates=("S6",))))["gates"]["S6"]
    free_task = FailingS6()
    free_task.window = {"allow_independent": True}
    free = asyncio.run(deterministic_gates(free_task, cfg=GateConfig(gates=("S6",))))["gates"]["S6"]
    assert strict["ok"] is False and strict["findings"] == ["short: no edge"]
    assert free["ok"] is True and free["findings"] == [] and free["evidence"] == {"executed_edges": 0, "milestones": 2, "allow_independent": True}


def test_author_writes_in_the_generators_voice_and_the_lane_states_required_names(tmp_path: Path) -> None:
    repo, shas = make_repo(tmp_path)
    gen = Rebreak(tmp_path / "rb", {"mylib": repo}, funnel=Funnel(tmp_path / "f.jsonl"), lease=EvalLease.default())
    b = out(git(repo, "rev-parse", "v1.1^{commit}"))
    win, _ = gen.synthesize(Candidate("rb-x", "rebreak", "mylib", params={"b": b, "b_ref": "v1.1", "b_date": "2024-02-06T00:00:00+00:00", "b_version": "1.1",
                                                                          "fixes": [shas["fix_neg"], shas["fix_ws"]]}))
    rg = RepoGit(gen.clones["mylib"])
    base = tmp_path / "base"
    rg.build(win.a, [], base)
    tests_b = {"tests/test_core.py": rg.show(win.b, "tests/test_core.py")}
    rec = {"repo": "mylib", "a": win.a, "b": win.b, "units": [json.loads(json.dumps(u.__dict__)) for u in win.units], "tests_b": tests_b,
           "milestones": [{"id": f"m0{i + 1}", "unit": u.sha, "tests": [f"tests/test_core.py::t{i}"], "subject": u.subject} for i, u in enumerate(win.units)],
           "p2p": ["tests/test_core.py::test_parse"], "collateral": [], "id": "rb-x"}

    class Client:
        def __init__(self, reply: dict[str, Any]):
            self.reply, self.msgs = reply, None

        async def complete_json(self, msgs: Any, **kw: Any) -> dict[str, Any]:
            self.msgs = msgs
            return self.reply

    good = Client({"opening": "Users report two parsing problems.", "requirements": [{"id": "m01", "text": "Parsing '~3' must give -3."},
                                                                                    {"id": "m02", "text": "Parsing ' ~4 ' must ignore the spaces."}]})
    res = asyncio.run(author_generated(good, rec, rg, base, style=STYLE))
    assert res["findings"] == [] and "What is broken:" in res["prompt"] and "- [R1] Parsing '~3' must give -3." in res["prompt"]
    assert "ROLE: author" in good.msgs[0]["content"] and "bug report" in good.msgs[0]["content"]
    assert json.loads(good.msgs[1]["content"])["changes"][0]["message"].startswith("Fix parse of negative numbers")
    assert [r["id"] for r in res["requirements"]] == ["R1", "R2", "r_regress"]
    bad = Client({"opening": "See mylib/core.py.", "requirements": [{"id": "m01", "text": "Fix it in test_parse_negative."}]})
    res = asyncio.run(author_generated(bad, rec, rg, base, style=AuthorStyle("author", "brief", "H:")))
    assert any(f.startswith("lint file_path") for f in res["findings"]) and any(f.startswith("lint test_name") for f in res["findings"])
    assert "unmapped_milestone: change m02 has no requirement" in res["findings"]
    assert compose_prompt("O", [{"id": "R1", "text": "t"}], heading="H:").startswith("O\n\nH:\n- [R1] t")


def test_funnel_keeps_the_last_row_per_candidate_and_stage(tmp_path: Path) -> None:
    f = Funnel(tmp_path / "f.jsonl")
    f.note("rebreak", "c1", "S6", False, "independent: no edge")
    f.note("rebreak", "c1", "S6", True)
    f.note("rebreak", "c2", "S6", False, "independent: no edge")
    f.note("open", "o1", "S4_oracle", False, "headroom_unproven: P's speedup 1.01")
    s = f.summary()
    assert s["rebreak"]["S6"] == {"in": 2, "pass": 1, "reasons": {"independent": 1}}
    assert s["open"]["S4_oracle"]["reasons"] == {"headroom_unproven": 1}
    assert SimpleNamespace(x=1).x == 1
