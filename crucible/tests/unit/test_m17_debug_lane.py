"""The debug lane (swarm-pipeline/07) on a synthetic Go repository: fix detection under the Go profile, owned tests, the
start tree A' (source, test hunks and changelog lines of the group reverted; strip mode; visible tests hidden), B' == B,
the task's profile scope, the bug-report authoring checks, and ``subset`` (the candidate of named fixes). No Docker."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from crucible.factory.generators import debug as debug_lane
from crucible.factory.generators.base import Candidate, Funnel, git, out, tree_of
from crucible.factory.generators.debug import Debug, author_debug, group_id, owned_tests, scope
from crucible.factory.generators.rebreak import fix_units
from crucible.factory.lanes.windows_git import RepoGit
from crucible.factory.sources.lease import EvalLease

CORE0 = '''package calc

func Parse(s string) int {
\tn := 0
\tfor _, c := range s {
\t\tn = n*10 + int(c-'0')
\t}
\treturn n
}

func Fmt(n int) string {
\treturn itoa(n)
}
'''
TEST0 = '''package calc

import "testing"

func TestParse(t *testing.T) {
\tif Parse("12") != 12 {
\t\tt.Fatal("parse")
\t}
}
'''


def _commit(repo: Path, msg: str, files: dict[str, str], date: str) -> str:
    for p, t in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(t)
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg, env=env)
    return out(git(repo, "rev-parse", "HEAD"))


def make_repo(tmp: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp / "calc"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    shas = {"base": _commit(repo, "init", {"go.mod": "module example.com/calc\n\ngo 1.22\n", "calc.go": CORE0, "calc_test.go": TEST0,
                                           "CHANGELOG.md": "# Changes\n", "util.go": "package calc\n\nfunc itoa(n int) string { return \"\" }\n"},
                                   "2026-04-01T00:00:00+00:00")}
    core1 = CORE0.replace("\tn := 0\n", "\tneg := len(s) > 0 && s[0] == '-'\n\tif neg {\n\t\ts = s[1:]\n\t}\n\tn := 0\n").replace(
        "\treturn n\n}\n\nfunc Fmt", "\tif neg {\n\t\treturn -n\n\t}\n\treturn n\n}\n\nfunc Fmt")
    test1 = TEST0 + '\nfunc TestParseNegative(t *testing.T) {\n\tif Parse("-3") != -3 {\n\t\tt.Fatal("neg")\n\t}\n}\n'
    shas["fix_neg"] = _commit(repo, "Fix parsing of negative numbers", {"calc.go": core1, "calc_test.go": test1, "CHANGELOG.md": "# Changes\n- negative numbers parse\n"},
                              "2026-05-10T00:00:00+00:00")
    shas["feat"] = _commit(repo, "Add docs", {"README.md": "calc\n"}, "2026-05-11T00:00:00+00:00")
    util = 'package calc\n\nimport "strconv"\n\nfunc itoa(n int) string { return strconv.Itoa(n) }\n'
    test2 = test1 + '\nfunc TestFmt(t *testing.T) {\n\tif Fmt(-7) != "-7" {\n\t\tt.Fatal("fmt")\n\t}\n}\n'
    shas["fix_fmt"] = _commit(repo, "fix: Fmt returned an empty string", {"util.go": util, "calc_test.go": test2,
                                                                       "CHANGELOG.md": "# Changes\n- negative numbers parse\n- Fmt works\n"},
                              "2026-05-12T00:00:00+00:00")
    shas["bump"] = _commit(repo, "fix: bump go version", {"go.mod": "module example.com/calc\n\ngo 1.23\n"}, "2026-05-13T00:00:00+00:00")
    return repo, shas


def test_fix_units_under_the_go_profile(tmp_path: Path) -> None:
    repo, shas = make_repo(tmp_path)
    with scope("go"):
        fixes = fix_units(repo, "HEAD", since="2026-05-01")
    assert [u.sha for u in fixes] == [shas["fix_neg"], shas["fix_fmt"]]  # docs-only and go.mod changes are not fixes
    with scope("go"):
        u = fixes[0]
        assert owned_tests(RepoGit(repo), u, ["calc_test.go::TestParseNegative", "calc_test.go::TestParse", "x_test.go::TestY"]) == [
            "calc_test.go::TestParseNegative"]


def _gen(tmp: Path, repo: Path) -> Debug:
    return Debug(tmp / "dbg", {"calc": repo}, langs_of={"calc": "go"}, funnel=Funnel(tmp / "f.jsonl"), lease=EvalLease.default())


def test_synthesize_revert_and_strip_modes(tmp_path: Path) -> None:
    repo, shas = make_repo(tmp_path)
    gen = _gen(tmp_path, repo)
    b = out(git(repo, "rev-parse", "HEAD"))
    cand = Candidate("db-go-calc-x", "debug", "calc", params={"lang": "go", "b": b, "b_date": "2026-05-13T00:00:00+00:00",
                                                             "fixes": [shas["fix_neg"], shas["fix_fmt"]]})
    d = gen.clones["calc"]
    win, info = gen.synthesize(cand)
    assert win is not None and info["hide_mode"] == "revert" and info["b_prime_equals_b"] and len(win.units) == 2
    a_test, a_util, a_log = (out(git(d, "show", f"{win.a}:{p}")) for p in ("calc_test.go", "util.go", "CHANGELOG.md"))
    assert "TestParseNegative" not in a_test and "TestFmt" not in a_test and "func TestParse(" in a_test
    assert "strconv" not in a_util and "negative numbers" not in a_log and "Fmt works" not in a_log  # the changelog does not announce the fixes
    assert tree_of(d, win.b) == tree_of(repo, b) and out(git(repo, "for-each-ref", "refs/crucible")) == ""
    win2, info2 = gen.synthesize(cand, mode="strip", extra_hide={"calc_test.go": {"TestParse"}})
    gone = git(d, "show", f"{win2.a}:calc_test.go", check=False)
    assert info2["hide_mode"] == "strip" and gone.returncode != 0  # every test of the file hidden: the file is removed from A'
    assert "func TestParse(" in out(git(d, "show", f"{win2.b}:calc_test.go"))
    assert info2["hidden"]["visible_stripped"] == {"calc_test.go": ["TestParse"]} and tree_of(d, win2.b) == tree_of(repo, b)


def test_task_intent_and_cards() -> None:
    from crucible.factory.generators.debug import SKILLS, intent_cards, task_intent

    rec = {"lang": "go", "p2p": ["a_test.go::TestA"], "units": [{"sha": "u1", "files": ["a/x.go", "a/x_test.go"]}, {"sha": "u2", "files": ["b/y.go"]},
                                                              {"sha": "u3", "files": ["a/z.go"]}],
           "milestones": [{"id": "m01", "unit": "u1", "tests": ["t1"]}, {"id": "m02", "unit": "u2", "tests": ["t2"], "requires": ["m01"]},
                          {"id": "m03", "unit": "u3", "tests": ["t3", "t4"], "requires": ["m02"]}]}
    it = task_intent(rec)
    assert it["skills"] == ["multi_fault_localization", "interacting_faults", "reproduce_from_report", "regression_safety"]
    assert it["level"] == "hard" and it["modules"] == ["a", "b"] and "depth 3" in it["why_long"] and it["milestones"][0]["files"] == ["a/x.go"]
    assert set(it["trace_evidence"]) == set(it["skills"]) and set(SKILLS) == {c.coverage["behaviour"] for c in intent_cards()}


def test_author_debug_checks_the_bug_report(tmp_path: Path) -> None:
    repo, shas = make_repo(tmp_path)
    gen = _gen(tmp_path, repo)
    b = out(git(repo, "rev-parse", "HEAD"))
    win, _ = gen.synthesize(Candidate("db-go-calc-y", "debug", "calc", params={"lang": "go", "b": b, "b_date": "2026-05-13T00:00:00+00:00",
                                                                                "fixes": [shas["fix_neg"], shas["fix_fmt"]]}))
    rec = {**win.to_json(), "lang": "go", "p2p": ["calc_test.go::TestParse"], "collateral": [],
           "milestones": [{"id": "m01", "unit": win.units[0].sha, "tests": ["calc_test.go::TestParseNegative"]},
                          {"id": "m02", "unit": win.units[1].sha, "tests": ["calc_test.go::TestFmt"], "requires": ["m01"]}]}
    good = {"builder": "claude-code:opus-5.5", "opening": "Two things in the calculator package are off.",
            "requirements": {"m01": "Reading \"-3\" gives a large positive number instead of minus three.",
                             "m02": "Formatting a number always produces an empty string."}}
    res = author_debug(rec, good, RepoGit(gen.clones["calc"]))
    assert res["findings"] == [] and res["provenance"]["builder"] == "claude-code:opus-5.5"
    assert "What I am seeing:" in res["prompt"] and "go.work files or a vendor directory" in res["prompt"] and res["requirements"][-1]["id"] == "r_regress"
    bad = {**good, "builder": "", "requirements": {"m01": "Parse breaks on negatives; see TestParseNegative in calc_test.go."}}
    found = " ".join(author_debug(rec, bad, RepoGit(gen.clones["calc"]))["findings"])
    assert "provenance" in found and "faulty_name" in found and "hidden_test" in found and "file_path" in found and "unmapped_milestone" in found


def _fake_fix_check(caught: dict[str, list[str]]):
    """``fix_check`` without Docker: a fix is real when the revert catches a test, and the caught ids decide the edges."""

    async def fake(repo, b, units, sandbox_for, images, repo_name, work, *, lang, full_suite=False):
        return {u.sha: {"ok": bool(caught.get(u.sha)), "caught": caught.get(u.sha, []), "why": ""} for u in units}, {"ok": True, "image": "img-b"}

    return fake


def test_subset_is_mines_candidate_of_the_named_fixes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, shas = make_repo(tmp_path)
    neg, fmt = shas["fix_neg"], shas["fix_fmt"]
    both = "calc_test.go::TestParse"  # caught by either revert alone: the executed edge
    monkeypatch.setattr(debug_lane, "fix_check", _fake_fix_check({neg: [both, "calc_test.go::TestParseNegative"], fmt: [both, "calc_test.go::TestFmt"]}))
    gen = _gen(tmp_path, repo)
    [mined] = asyncio.run(gen.candidates_for("calc"))
    with scope("go"):
        units = fix_units(repo, "HEAD", since="2026-05-01")
    assert mined.id == group_id("go", "calc", units) and mined.params["needs"] == {neg: [fmt], fmt: [neg]}
    # named in any order and by prefix, the fixes give mine's candidate: same id (the fixes' hash), same params
    same = asyncio.run(gen.subset("calc", [fmt[:8], neg[:8]]))
    assert same.id == mined.id and same.params == mined.params and "parent" not in same.params and "hide_modes" not in same.params
    eased = asyncio.run(gen.subset("calc", [neg[:8], fmt[:8]], cid="db-go-calc-e1", parent=mined.id, hide="strip"))
    assert eased.id == "db-go-calc-e1" and eased.params["parent"] == mined.id and eased.params["fixes"] == mined.params["fixes"]
    assert eased.params["hide_modes"] == ["strip"] and "hide_tests" not in eased.params
    named = asyncio.run(gen.subset("calc", [neg[:8], fmt[:8]], cid="db-go-calc-e2", hide_tests=["calc_test.go::TestFmt", "calc_test.go::TestFmt"]))
    assert named.params["hide_tests"] == ["calc_test.go::TestFmt"]


def test_build_strips_the_named_visible_tests_from_the_first_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """subset --hide-tests: a stale visible test (mvdan-sh's TestNodeEndPos) is stripped from A' before round 2 could."""
    repo, _ = make_repo(tmp_path)
    gen = _gen(tmp_path, repo)
    seen: list[dict[str, set[str]]] = []
    monkeypatch.setattr(Debug, "synthesize", lambda self, cand, *, mode="revert", extra_hide=None: (seen.append(dict(extra_hide or {})), (None, {}))[1])
    asyncio.run(gen.build(Candidate("db-go-calc-h", "debug", "calc", params={"lang": "go", "hide_tests": ["calc_test.go::TestParse/neg"]})))
    asyncio.run(gen.build(Candidate("db-go-calc-n", "debug", "calc", params={"lang": "go"})))
    assert seen == [{"calc_test.go": {"TestParse"}}, {}]


def test_build_hides_the_tests_in_the_named_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = make_repo(tmp_path)
    gen = _gen(tmp_path, repo)
    seen: list[str] = []
    monkeypatch.setattr(Debug, "synthesize", lambda self, cand, *, mode="revert", extra_hide=None: (seen.append(mode), (None, {}))[1])
    for params, first in (({}, "revert"), ({"hide_modes": ["strip"]}, "strip")):  # mined: revert first; subset --hide strip: strip only
        rec = asyncio.run(gen.build(Candidate(f"db-go-calc-{first}", "debug", "calc", params={"lang": "go", **params})))
        assert rec["stage"] == "source" and seen[-1] == first


def test_subset_refuses_what_mine_would_not_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, shas = make_repo(tmp_path)
    neg, fmt = shas["fix_neg"], shas["fix_fmt"]
    monkeypatch.setattr(debug_lane, "fix_check", _fake_fix_check({neg: ["calc_test.go::TestParseNegative"], fmt: ["calc_test.go::TestFmt"]}))
    gen = _gen(tmp_path, repo)
    for fixes, b, why in (([neg, shas["feat"]], "HEAD", "no fix unit"), ([neg, neg[:7]], "HEAD", "at least two fixes"),
                          ([neg, fmt], "no-such-ref", "is not a commit"), ([neg, fmt], "HEAD", "no executed interaction")):
        with pytest.raises(ValueError, match=why):
            asyncio.run(gen.subset("calc", fixes, b=b))
    with pytest.raises(ValueError, match="not revert or strip"):
        asyncio.run(gen.subset("calc", [neg, fmt], hide="delete"))
    other = tmp_path / "other"  # a fresh work dir: the evidence above is cached per pool
    other.mkdir()
    monkeypatch.setattr(debug_lane, "fix_check", _fake_fix_check({neg: ["calc_test.go::TestParseNegative"]}))
    with pytest.raises(ValueError, match=rf"not a real fix .*{fmt[:10]}"):
        asyncio.run(_gen(other, repo).subset("calc", [neg, fmt]))
