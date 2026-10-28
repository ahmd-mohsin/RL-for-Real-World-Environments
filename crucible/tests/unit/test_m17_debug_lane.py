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


def test_a_stale_case_is_hidden_by_removing_its_case_file(tmp_path: Path) -> None:
    from crucible.factory.generators.debug import hide_key

    changed = ["src/black/strings.py", "tests/data/cases/docstring.py", "tests/test_black.py"]
    assert hide_key("tests/test_format.py::test_simple_format[docstring]", changed) == ("tests/data/cases/docstring.py", "*")
    assert hide_key("tests/test_format.py::test_simple_format[fmtonoff]", changed) == ("tests/test_format.py", "test_simple_format")
    assert hide_key("tests/test_black.py::BlackTestCase::test_tabs", changed) == ("tests/test_black.py", "BlackTestCase::test_tabs")
    repo, shas = make_repo(tmp_path)
    gen = _gen(tmp_path, repo)
    b = out(git(repo, "rev-parse", "HEAD"))
    cand = Candidate("db-go-calc-z", "debug", "calc", params={"lang": "go", "b": b, "b_date": "2026-05-13T00:00:00+00:00",
                                                             "fixes": [shas["fix_neg"], shas["fix_fmt"]]})
    win, info = gen.synthesize(cand, extra_hide={"calc_test.go": {"*"}})
    d = gen.clones["calc"]
    assert info["hide_mode"] == "revert" and git(d, "show", f"{win.a}:calc_test.go", check=False).returncode != 0  # gone from A'
    assert "func TestFmt(" in out(git(d, "show", f"{win.b}:calc_test.go")) and tree_of(d, win.b) == tree_of(repo, b)
    assert info["hidden"]["visible_stripped"] == {"calc_test.go": ["*"]}


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


def test_a_renamed_tests_old_copy_that_asserts_the_fault_is_stale() -> None:
    """A fix renamed its test, A' kept the old copy (asserting the fault's behaviour) while m01's hidden test asserts
    the fix's. The old copy is in a touched test file, graded by nothing, passes in A' and fails on
    B's source: stale. An untouched file's test, a p2p test and one that also passes on B's source are not."""
    from crucible.factory.generators.debug import scope, stale_tests

    rec = {"milestones": [{"id": "m01", "tests": ["app_test.go::TestRun_sends", "diff/parse_test.go::TestParseErr"]}],
           "p2p": ["app_test.go::TestRun_p2p"]}
    view = {"app_test.go::TestRun_sends": "failed", "diff/parse_test.go::TestParseErr": "passed",
            "app_test.go::TestRun_old_name": "passed", "app_test.go::TestRun_p2p": "passed",
            "app_test.go::TestRun_other": "passed", "cmd/main_test.go::TestRun_untouched": "passed"}
    oracle = {"diff/parse_test.go::TestParseErr": "failed", "app_test.go::TestRun_old_name": "failed",
              "app_test.go::TestRun_p2p": "failed", "app_test.go::TestRun_other": "passed",
              "cmd/main_test.go::TestRun_untouched": "failed"}
    touched = ["app.go", "app_test.go", "diff/parse.go", "diff/parse_test.go"]
    with scope("go"):  # the lane's profile decides what a test file is (build runs under it)
        assert stale_tests(rec, view, oracle, touched) == ["app_test.go::TestRun_old_name", "diff/parse_test.go::TestParseErr"]
        assert stale_tests(rec, view, {}, touched) == []  # an unusable oracle run says nothing


def test_the_output_texts_a_hidden_test_checks_must_be_quoted_in_the_report() -> None:
    """A held-out task's m02 asserts two error-body texts; its report named only the status code and no model passed
    m02. The texts go to the writer and the write-prompts
    lint refuses a requirement that does not quote them. Patterns, formats and removed lines are not texts to quote."""
    from crucible.factory.generators.debug import asserted_messages, unstated_messages

    diff = "\n".join([  # noqa: FLY002 - one diff line per entry
        "+++ b/api/items_test.go",
        '+\tassert.Contains(t, w.Body.String(), "ids must match")',
        '+\trequire.ErrorContains(t, err, "wrong owner")',
        '-\tassert.Contains(t, w.Body.String(), "old text")',
        '+\tassert.Contains(t, out, "%s failed")',
        "+    with pytest.raises(ValueError, match=r\"invalid \\d+\"):",
        "+    with pytest.raises(ValueError, match='bad header'):",
        "+  expect(() => parse(x)).toThrow(`unexpected token`)",
        '+    self.assertIn("no such key", str(e))',
        '+#[should_panic(expected = "index out of range")]',
        '+\tassert.Contains(t, out, "ids must match")',
    ])
    msgs = asserted_messages(diff)
    assert msgs == ["ids must match", "wrong owner", "bad header", "unexpected token", "no such key", "index out of range"]
    reqs = {"m02": "The API should reject it with HTTP 400 and the message  ids must match  instead of saving it."}
    assert unstated_messages(reqs, {"m02": ["ids must match", "wrong owner"]}) == [
        "unstated_message: m02 must quote 'wrong owner' (a hidden check looks for this exact text in the output)"]


def test_a_stale_test_defined_only_in_the_start_tree_is_still_stripped(tmp_path: Path) -> None:
    """The old name of a renamed test is defined in A' only. ``_locate_defs`` looked its key up at B, found nothing and
    dropped it, so the second build round met the same stale test and refused the window."""
    from crucible.factory.generators.debug import Debug, scope
    from crucible.factory.lanes.windows_git import RepoGit

    repo, _ = make_repo(tmp_path)
    b = out(git(repo, "rev-parse", "HEAD"))
    with scope("go") as prof:
        got = Debug._locate_defs(RepoGit(repo), b, prof, {"calc_test.go": {"TestParse", "TestOldName"}})
    assert got == {"calc_test.go": {"TestParse", "TestOldName"}}


def test_a_fix_check_the_host_did_not_run_is_neither_cached_nor_a_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under load the suite at B timed out (900 s) for every fix of a held-out rebuild: the evidence was cached, subset
    said "not a real fix" for eight fixes that had built tasks before, and every retry read the cache. A host failure
    is now reported as one and never cached; the next run checks again."""
    repo, shas = make_repo(tmp_path)
    neg, fmt = shas["fix_neg"], shas["fix_fmt"]
    calls = []

    async def timed_out(repo_, b, units, sandbox_for, images, repo_name, work, *, lang, full_suite=False):
        calls.append(1)
        why = "the suite does not run at B (timeout after 900.0s)"
        return {u.sha: {"ok": False, "why": why, "infra": True} for u in units}, {"ok": True, "image": "img-b"}

    monkeypatch.setattr(debug_lane, "fix_check", timed_out)
    gen = _gen(tmp_path, repo)
    for _ in range(2):
        with pytest.raises(ValueError, match="did not run .*host failure"):
            asyncio.run(gen.subset("calc", [neg, fmt], independent=True))
    assert len(calls) == 2 and not list((tmp_path / "dbg" / "work").rglob("evidence-*.json"))


def test_a_checked_literal_list_is_a_set_of_texts_to_quote() -> None:
    """A held-out milestone looped over five structured-log keys and checked each with strings.Contains; the report
    named none of them and no model passed it. The items of a literal list count when the loop's item is the needle of a
    containment check; a list of inputs whose output is checked for something else does not."""
    from crucible.factory.generators.debug import asserted_messages

    keys = "\n".join([  # noqa: FLY002 - one diff line per entry
        "+\tfor _, want := range []string{",
        '+\t\t`msg="cache: summary"`,',
        "+\t\t`entry-mode=strict`,",
        '+\t\t"hits=1",',
        "+\t} {",
        "+\t\tif !strings.Contains(logs.String(), want) {",
        '+\t\t\tt.Errorf("missing %q", want)',
        "+\t\t}",
        "+\t}",
    ])
    assert asserted_messages(keys) == ['msg="cache: summary"', "entry-mode=strict", "hits=1"]
    inputs = "\n".join([  # noqa: FLY002
        '+\tfor _, in := range []string{"null", "~"} {',
        "+\t\terr := Parse(in)",
        '+\t\tassert.ErrorContains(t, err, "bad value")',
        "+\t}",
    ])
    assert asserted_messages(inputs) == ["bad value"]
    py = "+    for want in ['first line', 'second line']:\n+        assert want in out\n"
    assert asserted_messages(py) == ["first line", "second line"]


def test_an_expected_text_in_a_want_variable_or_table_field_is_a_text_to_quote() -> None:
    """Two held-out milestones compared an analyzer's exact finding text (``want := "..."`` then ``want == got.Text``;
    a table row's ``expectedErrors: []string{...}``) that no report stated; every student and frontier run failed them.
    The list's opener may be a kept line (the fix adds only a row); only added lines give texts, and a word, a format
    or an escape is not one."""
    from crucible.factory.generators.debug import asserted_messages

    one = "\n".join([  # noqa: FLY002 - one diff line per entry
        "+\twant := \"Pool 'ns/p' has state Ready=False, cause Drained: no members left\"",
        "+\twantKind := \"Pool\"",
        '+\texpected := "%s failed"',
    ])
    assert asserted_messages(one) == ["Pool 'ns/p' has state Ready=False, cause Drained: no members left"]
    table = "\n".join([  # noqa: FLY002
        " \t\t\texpectedErrors: []string{",
        ' \t\t\t\t"Queue ns/a has no consumer group.",',
        '+\t\t\t\t"Queue ns/a routes to the fallback topic ns/t which is unknown.",',
        " \t\t\t},",
        '+\t\t\twant: "line one\\nline two",',
    ])
    assert asserted_messages(table) == ["Queue ns/a routes to the fallback topic ns/t which is unknown."]
