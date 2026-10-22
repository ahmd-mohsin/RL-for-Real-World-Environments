"""P5 rebreak (swarm-pipeline/00 §2): fix detection, the executed-interaction screen, grouping in prober order, hidden fix
tests, and the synthetic history A' (fixes reverted) .. B' (== B) on a synthetic git repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from crucible.factory.generators.base import Funnel, apply_chain, full_patch, git, out, tree_of
from crucible.factory.generators.rebreak import (
    Rebreak,
    allowed_file,
    changed_functions,
    fix_test_names,
    fix_units,
    group_fixes,
    interaction_edges,
    is_fix,
    order_fixes,
    release_window_of,
    strip_tests,
)
from crucible.factory.lanes.windows_git import RepoGit, Unit, version_tags
from crucible.factory.sources.lease import EvalLease, SealedSourceError

CORE0 = '''def parse(s):
    return int(s)


def fmt(x):
    return str(x)


def clamp(x):
    return x
'''
TESTS0 = '''from mylib.core import fmt, parse


def test_parse():
    assert parse("1") == 1


def test_fmt():
    assert fmt(2) == "2"
'''


def _commit(repo: Path, msg: str, files: dict[str, str], date: str, tag: str | None = None) -> str:
    for p, t in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(t)
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg, env=env)
    if tag:
        git(repo, "tag", "-a", tag, "-m", tag, env=env)
    return out(git(repo, "rev-parse", "HEAD"))


def make_repo(tmp: Path) -> tuple[Path, dict[str, str]]:
    r = tmp / "mylib"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    shas = {"init": _commit(r, "init", {"pyproject.toml": "[project]\nname = 'mylib'\n", "mylib/__init__.py": "", "mylib/core.py": CORE0,
                                          "tests/test_core.py": TESTS0, "CHANGES.rst": "Changes\n=======\n"}, "2024-01-01T00:00:00+00:00", "v1.0")}
    core = CORE0.replace("    return int(s)", "    return int(s.replace('~', '-'))")
    tests = TESTS0 + '\n\ndef test_parse_negative():\n    assert parse("~3") == -3\n'
    shas["fix_neg"] = _commit(r, "Fix parse of negative numbers", {"mylib/core.py": core, "tests/test_core.py": tests,
                                                                   "CHANGES.rst": "Changes\n=======\n- negatives\n"}, "2024-02-01T00:00:00+00:00")
    shas["docs"] = _commit(r, "Fix typo in README", {"README.md": "mylib\n"}, "2024-02-02T00:00:00+00:00")
    core2 = core.replace("    return str(x)", "    return str(round(x, 2))")
    tests2 = tests + '\n\ndef test_fmt_round():\n    assert fmt(1.234) == "1.23"\n'
    shas["fix_fmt"] = _commit(r, "Fix fmt rounding bug", {"mylib/core.py": core2, "tests/test_core.py": tests2}, "2024-02-03T00:00:00+00:00")
    core3 = core2.replace("    return str(round(x, 2))", "    return format(round(x, 2), 'g')")  # later code rewrites fmt's fix: its revert conflicts
    shas["refactor"] = _commit(r, "Refactor fmt", {"mylib/core.py": core3}, "2024-02-04T00:00:00+00:00")
    shas["pkg"] = _commit(r, "Fix packaging metadata", {"pyproject.toml": "[project]\nname = 'mylib'\nversion = '1.1'\n", "mylib/core.py": core3 + "\n",
                                                        "tests/test_core.py": tests2 + "\n"}, "2024-02-05T00:00:00+00:00")
    core4 = core3.replace("int(s.replace('~', '-'))", "int(s.strip().replace('~', '-'))")
    tests4 = tests2 + '\n\ndef test_parse_space():\n    assert parse(" ~4 ") == -4\n'
    shas["fix_ws"] = _commit(r, "Fix parse whitespace bug", {"mylib/core.py": core4 + "\n", "tests/test_core.py": tests4 + "\n"},
                             "2024-02-06T00:00:00+00:00", "v1.1")
    return r, shas


def _unit(repo: Path, sha: str) -> Unit:
    from crucible.factory.generators.base import changed_paths, commit_meta

    m = commit_meta(repo, sha)
    return Unit(sha=sha, parent=m["parent"], merge=False, date=m["date"], subject=m["subject"], files=changed_paths(repo, m["parent"], sha))


def test_fix_detection_reads_messages_and_files(tmp_path: Path) -> None:
    repo, shas = make_repo(tmp_path)
    b = out(git(repo, "rev-parse", "v1.1^{commit}"))
    got = [u.sha for u in fix_units(repo, b, since="2023-01-01")]
    assert got == [shas["fix_neg"], shas["fix_fmt"], shas["fix_ws"]]  # not the README typo, not the packaging change
    assert not is_fix(_unit(repo, shas["pkg"])) and not allowed_file("pyproject.toml") and not allowed_file("tests/conftest.py")
    assert allowed_file("CHANGES.rst") and allowed_file("tests/test_x.py") and allowed_file("mylib/core.py")
    tags = version_tags(repo)
    assert release_window_of(repo, tags, set(got)) == dict.fromkeys(got, 1)
    rg = RepoGit(repo)
    assert fix_test_names(rg, _unit(repo, shas["fix_neg"])) == {"tests/test_core.py::test_parse_negative"}
    assert changed_functions(rg, _unit(repo, shas["fix_neg"])) == {"mylib/core.py::parse"}


def test_interactions_are_shared_caught_test_cases() -> None:
    ev = {"a": {"ok": True, "caught": ["t.py::test_x[1]", "t.py::test_y"]}, "b": {"ok": True, "caught": ["t.py::test_y"]},
          "c": {"ok": True, "caught": ["t.py::test_x[2]"]}, "d": {"ok": False, "caught": ["t.py::test_y"]}}
    assert interaction_edges(ev) == {"a": {"b"}, "b": {"a"}}  # [1] and [2] are different cases; d's own revert is not caught
    # Go reports a parent test besides its subtests: a parent that fails because each revert breaks a different subtest
    # links nothing (mvdan-sh dc951aeb), a subtest both reverts break does
    go = {"a": {"ok": True, "caught": ["p_test.go::TestT", "p_test.go::TestT/#1"]}, "b": {"ok": True, "caught": ["p_test.go::TestT", "p_test.go::TestT/#2"]},
          "c": {"ok": True, "caught": ["p_test.go::TestT", "p_test.go::TestT/#1", "p_test.go::TestT/#3"]}}
    assert interaction_edges(go) == {"a": {"c"}, "c": {"a"}}
    # a test that only stopped building (its compiled package errors as a whole) links nothing either (mvdan-sh's zsh fixes)
    built = {"a": {"ok": True, "caught": ["p_test.go::TestB", "p_test.go::TestA"], "caught_unbuilt": ["p_test.go::TestB"]},
             "b": {"ok": True, "caught": ["p_test.go::TestB", "p_test.go::TestC"], "caught_unbuilt": ["p_test.go::TestB", "p_test.go::TestC"]},
             "c": {"ok": True, "caught": ["p_test.go::TestA"]}}
    assert interaction_edges(built) == {"a": {"c"}, "c": {"a"}}
    # a case file both fixes changed checks both behaviours side by side (black ae5830b7's docstring.py): no link; a test
    # one fix owns that the other's revert breaks is one (go-toml's round trip)
    case = {"a": {"ok": True, "caught": ["t.py::test_fmt[doc]"], "caught_own": ["t.py::test_fmt[doc]"]},
            "b": {"ok": True, "caught": ["t.py::test_fmt[doc]", "t.py::test_rt"], "caught_own": ["t.py::test_fmt[doc]"]},
            "c": {"ok": True, "caught": ["t.py::test_rt", "t.py::test_c"], "caught_own": ["t.py::test_rt", "t.py::test_c"]}}
    assert interaction_edges(case) == {"b": {"c"}, "c": {"b"}}
    us =[Unit(sha=s, parent="p", merge=False, date="", subject=s, files=[f"{s}.py", "tests/test_t.py"]) for s in "abcd"]
    assert [u.sha for u in order_fixes(us, {"a": {"c"}})] == ["b", "c", "a", "d"]  # a needs c: c first, else history order
    groups = group_fixes(us, {"a": {"f"}, "c": {"f"}}, need={"a": {"b"}, "b": {"a"}}, max_size=3)
    # the interacting pair, filled with c (shares function f with a); a fix with no pending prerequisite goes first, a cycle keeps history order
    assert [[u.sha for u in g] for g in groups] == [["c", "a", "b"]]


def test_strip_tests_hides_the_fix_tests_only() -> None:
    text = TESTS0 + "\n\nclass TestK:\n    def test_one(self):\n        pass\n\n\ndef helper():\n    return 1\n"
    out_ = strip_tests(text, {"test_fmt", "TestK::test_one", "helper"})
    assert "def test_fmt" not in out_ and "def test_parse" in out_ and "def helper" in out_  # helpers stay: other tests may use them
    assert "class TestK:\n    pass" in out_  # an emptied class stays valid Python
    compile(out_, "t.py", "exec")


def test_synthesize_reverts_the_group_and_ends_at_b(tmp_path: Path) -> None:
    repo, shas = make_repo(tmp_path)
    gen = Rebreak(tmp_path / "rb", {"mylib": repo}, funnel=Funnel(tmp_path / "f.jsonl"), lease=EvalLease.default())
    b = out(git(repo, "rev-parse", "v1.1^{commit}"))
    from crucible.factory.generators.base import Candidate

    cand = Candidate("rb-mylib-x", "rebreak", "mylib", params={"b": b, "b_ref": "v1.1", "b_date": "2024-02-06T00:00:00+00:00", "b_version": "1.1",
                                                              "fixes": [shas["fix_neg"], shas["fix_fmt"], shas["fix_ws"]]})
    win, info = gen.synthesize(cand)
    assert win is not None and [d["sha"] for d in info["dropped"]] == [shas["fix_fmt"]]  # its revert conflicts with the later rewrite of fmt
    assert info["b_prime_equals_b"] and len(win.units) == 2 and [f["subject"] for f in info["fixes"]] == ["Fix parse of negative numbers",
                                                                                                           "Fix parse whitespace bug"]
    d = gen.clones["mylib"]
    a_core = out(git(d, "show", f"{win.a}:mylib/core.py"))
    a_tests = out(git(d, "show", f"{win.a}:tests/test_core.py"))
    assert "replace('~', '-')" not in a_core and "strip()" not in a_core and "format(round" in a_core  # both faults back, the rewrite kept
    assert "test_parse_negative" not in a_tests and "test_parse_space" not in a_tests and "test_fmt_round" in a_tests  # fix tests hidden
    mid = out(git(d, "show", f"{win.units[0].sha}:mylib/core.py"))
    assert "replace('~', '-')" in mid and "strip()" not in mid
    assert tree_of(d, win.b) == tree_of(repo, b)
    assert out(git(repo, "for-each-ref", "refs/crucible")) == ""  # the source clone is never written
    assert out(git(d, "rev-parse", "refs/crucible/rebreak/rb-mylib-x")) == win.b
    with pytest.raises(SealedSourceError):
        Rebreak(tmp_path / "rb2", {"tomlkit": repo}, funnel=Funnel(tmp_path / "f.jsonl"), lease=EvalLease.default())


def test_apply_chain_reports_the_step_that_conflicts(tmp_path: Path) -> None:
    repo, shas = make_repo(tmp_path)
    b = out(git(repo, "rev-parse", "v1.1^{commit}"))
    p = full_patch(repo, shas["fix_fmt"] + "^", shas["fix_fmt"], ["mylib/core.py"])
    trees, bad = apply_chain(repo, b, [(full_patch(repo, shas["fix_ws"] + "^", shas["fix_ws"], ["mylib/core.py"]), True), (p, True)])
    assert len(trees) == 1 and bad == 1
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=False).stdout == ""  # work tree untouched
