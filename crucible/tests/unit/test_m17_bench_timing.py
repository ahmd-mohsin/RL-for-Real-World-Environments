"""Benchmark v2: the timing-test rule (``factory.lanes.windows_timing``). A test whose verdict depends on wall time is never
a milestone or a regression check: found statically at mining (name, marker, fixture, a body that measures elapsed time)
and empirically at S10 (one end state, a verdict that moves). P6's mistune windows are the cases: ``..._are_near_linear``
(``assert_near_linear``) and ``..._does_not_scale_quadratically`` (``perf_counter() - start < 2.0``)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from test_m17_w_miner import SB, _commit, _tag

from crucible.factory.lanes.windows import f2p_stage, window_files
from crucible.factory.lanes.windows_git import RepoGit, release_windows
from crucible.factory.lanes.windows_timing import drop_tests, load_dependent, name_says_timing, timing_reason, timing_tests

MISTUNE = '''from time import perf_counter
import time

import pytest

from tests.perf_utils import assert_near_linear


class TestMiscCases:
    def test_deeply_indented_list_does_not_scale_quadratically(self):
        start = perf_counter()
        render("x" * 10)
        self.assertLess(perf_counter() - start, 2.0)

    def test_dense_emphasis(self):
        _small, html = assert_near_linear(self, render, 4000, 8000)

    def test_angle_link_escaped_backslash(self):
        assert render("<a\\\\b>") == "x"

    def test_stamp(self):
        stamp = time.time()
        assert stamp > 0

    def test_render_once(self):
        t0 = time.time()
        render("x")
        assert time.time() - t0 < 1


@pytest.mark.benchmark
def test_marked():
    assert True


@pytest.mark.timeout(1)
def test_marked_two():
    assert True


def test_with_fixture(benchmark):
    benchmark(len, "x")


@pytest.mark.flaky(reruns=3)
class TestFlakyClass:
    def test_inner(self):
        assert True


@pytest.mark.parametrize("n", [1, 2])
def test_parse_timedelta(n):
    assert n
'''


def test_names_say_timing_by_segment() -> None:
    yes = ["test_dense_emphasis_is_near_linear", "test_deeply_indented_list_does_not_scale_quadratically", "TestPerformance", "test_timeout",
           "test_benchmarks", "test_unmatched_bracket_input_returns_quickly", "test_latency_budget", "test_elapsed_wall_clock", "TestTiming"]
    # domain nouns are not measurements: voluptuous's decimal precision/scale, mistune's ``speedup`` plugin (first v2 run)
    no = ["test_parse_timedelta", "test_scaled_font", "test_angle_link", "test_benchmarking_helper_imports", "test_fastapi", "test_speedometer_label",
          "test_number_validation_with_invalid_precision_invalid_scale", "test_speedup_plugin_is_compat_noop", "test_scale_factor"]
    assert all(name_says_timing(n) for n in yes), [n for n in yes if not name_says_timing(n)]
    assert not any(name_says_timing(n) for n in no), [n for n in no if name_says_timing(n)]


def test_static_rule_reads_names_markers_fixtures_and_bodies() -> None:
    files = {"tests/test_misc.py": MISTUNE}
    ids = [f"tests/test_misc.py::{q}" for q in (
        "TestMiscCases::test_deeply_indented_list_does_not_scale_quadratically", "TestMiscCases::test_dense_emphasis",
        "TestMiscCases::test_angle_link_escaped_backslash", "TestMiscCases::test_stamp", "TestMiscCases::test_render_once", "test_marked",
        "test_marked_two", "test_with_fixture", "TestFlakyClass::test_inner", "test_parse_timedelta[1]")]
    got = timing_tests(ids, files)
    short = {k.split("::")[-1]: v for k, v in got.items()}
    assert short == {
        "test_deeply_indented_list_does_not_scale_quadratically": "name: test_deeply_indented_list_does_not_scale_quadratically",
        "test_dense_emphasis": "body calls the timing helper assert_near_linear()",
        "test_render_once": "body subtracts time.time()",
        "test_marked": "marker: benchmark",
        "test_marked_two": "marker: timeout",
        "test_with_fixture": "fixture: benchmark",
        "test_inner": "marker: flaky",
    }
    # a timestamp is not a measurement; a parametrised name is read without its parameters
    assert timing_reason("tests/test_misc.py::TestMiscCases::test_stamp", files) is None
    assert timing_reason("tests/test_misc.py::test_x[perf]") is None
    # without the files only the name is read
    assert timing_reason("tests/test_misc.py::TestMiscCases::test_dense_emphasis") is None
    assert timing_tests(["t.py::test_a"], {"t.py": "pytestmark = [pytest.mark.perf]\n\ndef test_a():\n    pass\n"}) == {"t.py::test_a": "module marker: perf"}


def test_load_dependent_is_a_verdict_that_moves_on_one_end_state() -> None:
    grades = [["a::m12"], [], ["a::m12", "a::always"], None, ["a::always"]]
    moving = load_dependent(grades, ["a::m12", "a::always", "a::never"])
    assert set(moving) == {"a::m12", "a::always"} and moving["a::m12"] == "S10: failed 2 of 4 oracle grades of one end state"
    assert load_dependent([["a::x"], ["a::x"]], ["a::x"]) == {}  # always failing is a wrong oracle, not load: S10's rate catches it
    assert load_dependent([["a::x"], None], ["a::x"]) == {}  # one measured grade proves nothing
    assert load_dependent([[], ["a::other"]], ["a::x"]) == {}  # an unchecked test is not a check


def test_drop_tests_removes_checks_and_milestones() -> None:
    rec = {"milestones": [{"id": "m01", "tests": ["t::a", "t::slow"], "requires": []}, {"id": "m02", "tests": ["t::fast"], "requires": ["m01"]},
                          {"id": "m03", "tests": ["t::c"], "requires": ["m02"], "order_hints": ["m02", "m01"]}],
           "p2p": ["t::p", "t::fast"], "collateral": ["t::slow"], "excluded": {"flaky": []}}
    out = drop_tests(rec, {"t::slow": "name: slow", "t::fast": "S10"})
    assert out == {"dropped_tests": 2, "milestones_gone": ["m02"], "milestones_touched": ["m01", "m02"]}
    assert [m["id"] for m in rec["milestones"]] == ["m01", "m03"] and rec["milestones"][0]["tests"] == ["t::a"]
    assert rec["milestones"][1]["requires"] == [] and rec["milestones"][1]["order_hints"] == ["m01"]
    assert rec["p2p"] == ["t::p"] and rec["collateral"] == [] and rec["excluded"]["timing"] == ["t::fast", "t::slow"]
    assert rec["timing_rule"] == {"t::slow": "name: slow", "t::fast": "S10"}


def _timing_repo(root: Path) -> Path:
    """v0.1 -> unit 1 adds ``alpha`` with a correctness test and a timing test, unit 2 adds ``beta``, unit 3 adds
    ``gamma`` on top of beta -> v0.2."""
    repo = root / "timingrepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    core = "def base():\n    return 0\n"
    _commit(repo, {"pkg/__init__.py": "", "pkg/core.py": core, "tests/test_core.py": "from pkg.core import base\n\n\ndef test_base():\n    assert base() == 0\n"},
            "start", date="2024-01-01T00:00:00")
    _tag(repo, "v0.1", "2024-01-01T00:00:00")
    _commit(repo, {"pkg/core.py": core + "\n\ndef alpha():\n    return 1\n",
                   "tests/test_core.py": ("from time import perf_counter\n\nfrom pkg.core import alpha, base\n\n\ndef test_base():\n    assert base() == 0\n\n\n"
                                          "def test_alpha():\n    assert alpha() == 1\n\n\ndef test_alpha_quick():\n    start = perf_counter()\n"
                                          "    alpha()\n    assert perf_counter() - start < 5\n")}, "Add alpha", date="2024-01-02T00:00:00")
    _commit(repo, {"pkg/beta.py": "def beta():\n    return 2\n", "tests/test_beta.py": "from pkg.beta import beta\n\n\ndef test_beta():\n    assert beta() == 2\n"},
            "Add beta", date="2024-01-03T00:00:00")
    _commit(repo, {"pkg/gamma.py": "from pkg.beta import beta\n\n\ndef gamma():\n    return beta() + 1\n",
                   "tests/test_gamma.py": "from pkg.gamma import gamma\n\n\ndef test_gamma():\n    assert gamma() == 3\n"}, "Add gamma", date="2024-01-04T00:00:00")
    _tag(repo, "v0.2", "2024-01-05T00:00:00")
    return repo


def _f2p(tmp: Path, **kw) -> dict:
    repo_dir = _timing_repo(tmp)
    win = release_windows(repo_dir, "timingrepo", since="2000-01-01", min_units=2)[0]
    rec = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "timings": {}, "split": "heldout"}
    repo = RepoGit(repo_dir)
    base = tmp / "base"
    assert repo.build(win.a, [], base)[0]
    window_files(rec, repo)
    rec["why"] = asyncio.run(f2p_stage(rec, repo, base, SB(), reruns=1, **kw))
    return rec


def test_mining_never_makes_a_timing_test_a_check(tmp_path: Path) -> None:
    rec = _f2p(tmp_path)
    assert rec["why"] is None
    checked = {t for m in rec["milestones"] for t in m["tests"]} | set(rec["p2p"]) | set(rec["collateral"])
    assert "tests/test_core.py::test_alpha_quick" not in checked and "tests/test_core.py::test_alpha" in checked
    assert rec["excluded"]["timing"] == ["tests/test_core.py::test_alpha_quick"]
    assert rec["timing_rule"] == {"tests/test_core.py::test_alpha_quick": "body calls perf_counter()"} and rec["validation"]["timing"] == 1


def test_mining_honours_s10_exclusions(tmp_path: Path) -> None:
    rec = _f2p(tmp_path, exclude={"tests/test_beta.py::test_beta": "S10: failed 3 of 60 oracle grades of one end state"})
    assert rec["why"] is None
    ms = {t for m in rec["milestones"] for t in m["tests"]}
    assert "tests/test_beta.py::test_beta" not in ms and "tests/test_gamma.py::test_gamma" in ms
    assert rec["timing_rule"]["tests/test_beta.py::test_beta"].startswith("S10:")
