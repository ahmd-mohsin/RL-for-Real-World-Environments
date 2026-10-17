"""Window lane (study 08 §5; 09): the miner, trees from subsets of units, test owners, the dependency prober and the
commit-revert mutants, on a tiny synthetic repository built in tmp.

The repository: ``v0.1`` has ``base()``; unit 1 adds ``alpha()``, unit 2 adds ``beta()`` in its own module, unit 3 adds
``gamma()`` which calls ``beta()`` (its test needs unit 2), unit 4 is a docs-only unit (no milestone), ``v0.2`` tags the
end. Every source unit also appends to ``CHANGES.rst`` (a changelog line every unit touches must not read as a
dependency). Tests run on the host interpreter (``LocalSandbox``: the Docker sandbox runs the same pytest command)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from crucible.factory.lanes.executable import LocalSandbox
from crucible.factory.lanes.windows import COLLECT_ALL, closure, ddmin, f2p_stage, probe_dependencies, revert_mutants, window_files
from crucible.factory.lanes.windows_git import RepoGit, Unit, key_of_test, mine_windows, owners_of_tests, release_windows


def SB() -> LocalSandbox:
    return LocalSandbox(extra_args=COLLECT_ALL)


GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]


def _commit(repo: Path, files: dict[str, str], msg: str, *, date: str) -> str:
    for p, text in files.items():
        f = repo / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    subprocess.run([*GIT, "add", "-A"], cwd=repo, check=True, capture_output=True)
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    subprocess.run([*GIT, "commit", "-qm", msg], cwd=repo, check=True, capture_output=True, env=env)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _tag(repo: Path, tag: str, date: str) -> None:
    env = {"GIT_COMMITTER_DATE": date, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    subprocess.run([*GIT, "tag", "-a", tag, "-m", tag], cwd=repo, check=True, capture_output=True, env=env)


def make_repo(root: Path) -> Path:
    repo = root / "toyrepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    core = "def base():\n    return 0\n"
    _commit(repo, {"pkg/__init__.py": "", "pkg/core.py": core, "tests/test_core.py": "from pkg.core import base\n\n\ndef test_base():\n    assert base() == 0\n",
                   "CHANGES.rst": "Changes\n=======\n", "README.md": "toy library\n"}, "start", date="2024-01-01T00:00:00")
    _tag(repo, "v0.1", "2024-01-01T00:00:00")
    _commit(repo, {"pkg/core.py": core + "\n\ndef alpha():\n    return 1\n",
                   "tests/test_core.py": "from pkg.core import base, alpha\n\n\ndef test_base():\n    assert base() == 0\n\n\ndef test_alpha():\n    assert alpha() == 1\n",
                   "CHANGES.rst": "Changes\n=======\n- alpha\n"}, "Add alpha", date="2024-01-02T00:00:00")
    _commit(repo, {"pkg/beta.py": "def beta():\n    return 2\n", "tests/test_beta.py": "from pkg.beta import beta\n\n\ndef test_beta():\n    assert beta() == 2\n",
                   "CHANGES.rst": "Changes\n=======\n- alpha\n- beta\n"}, "Add beta", date="2024-01-03T00:00:00")
    _commit(repo, {"pkg/gamma.py": "from pkg.beta import beta\n\n\ndef gamma():\n    return beta() + 1\n",
                   "tests/test_gamma.py": "from pkg.gamma import gamma\n\n\ndef test_gamma():\n    assert gamma() == 3\n",
                   "CHANGES.rst": "Changes\n=======\n- alpha\n- beta\n- gamma\n"}, "Add gamma on top of beta", date="2024-01-04T00:00:00")
    _commit(repo, {"README.md": "toy library, now with gamma\n"}, "Docs", date="2024-01-05T00:00:00")
    _commit(repo, {"tests/test_extra.py": "from pkg.core import alpha\n\n\ndef test_alpha_again():\n    assert alpha() == 1\n"}, "More tests",
            date="2024-01-05T12:00:00")
    _tag(repo, "v0.2", "2024-01-05T00:00:00")
    return repo


def built_record(tmp: Path) -> tuple[dict, RepoGit, Path]:
    """The record ``validate_window`` builds, without the Docker stages (image, leak probes): files, F2P, milestones."""
    repo_dir = make_repo(tmp)
    win = release_windows(repo_dir, "toyrepo", since="2000-01-01", min_units=2)[0]
    rec = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "timings": {}, "split": "train"}
    repo = RepoGit(repo_dir)
    base = tmp / "base"
    assert repo.build(win.a, [], base)[0]
    window_files(rec, repo)
    why = asyncio.run(f2p_stage(rec, repo, base, SB(), reruns=1))
    assert why is None, why
    return rec, repo, base


def test_release_window_units_and_qualifying(tmp_path: Path) -> None:
    repo_dir = make_repo(tmp_path)
    ws = release_windows(repo_dir, "toyrepo", since="2000-01-01", min_units=2)
    assert len(ws) == 1
    w = ws[0]
    assert (w.a_ref, w.b_ref, w.b_version, w.kind) == ("v0.1", "v0.2", "0.2", "release")
    assert [u.subject for u in w.units] == ["Add alpha", "Add beta", "Add gamma on top of beta", "Docs", "More tests"]
    assert [u.qualifies for u in w.units] == [True, True, True, False, False]
    assert w.units[1].src_files == ["pkg/beta.py"] and w.units[1].test_files == ["tests/test_beta.py"]
    assert "CHANGES.rst" in w.units[0].writable_files and "tests/test_core.py" not in w.units[0].writable_files
    # the size filter counts qualifying units; the cap keeps the newest
    assert release_windows(repo_dir, "toyrepo", since="2000-01-01", min_units=4) == []
    assert release_windows(repo_dir, "toyrepo", since="2030-01-01", min_units=2) == []
    assert len(mine_windows(repo_dir, "toyrepo", cap=1, since="2000-01-01", min_units=2, run_size=2)) == 1


def test_a_tag_off_the_first_parent_path_is_not_a_window(tmp_path: Path) -> None:
    repo_dir = make_repo(tmp_path)
    subprocess.run(["git", "checkout", "-q", "-b", "side", "v0.1"], cwd=repo_dir, check=True)
    _commit(repo_dir, {"pkg/side.py": "x = 1\n", "tests/test_side.py": "def test_side():\n    pass\n"}, "side", date="2024-02-01T00:00:00")
    _tag(repo_dir, "v0.3", "2024-02-01T00:00:00")
    assert [w.b_ref for w in release_windows(repo_dir, "toyrepo", since="2000-01-01", min_units=1)] == ["v0.2"]


def test_trees_from_unit_subsets_and_source_only_patches(tmp_path: Path) -> None:
    repo_dir = make_repo(tmp_path)
    w = release_windows(repo_dir, "toyrepo", since="2000-01-01", min_units=2)[0]
    repo = RepoGit(repo_dir)
    u1, u2, u3, _, _ = w.units
    ok, bad = repo.build(w.a, [u3], tmp_path / "t3")
    assert not ok and bad == u3.sha  # the changelog hunk needs units 1 and 2: the whole writable diff conflicts
    ok, _ = repo.build(w.a, [u3], tmp_path / "t3s", source_only=True)
    assert ok and (tmp_path / "t3s/pkg/gamma.py").exists() and not (tmp_path / "t3s/pkg/beta.py").exists()
    assert (tmp_path / "t3s/CHANGES.rst").read_text() == "Changes\n=======\n"
    ok, _ = repo.build(w.a, [u1, u2, u3], tmp_path / "t123")
    assert ok and "gamma" in (tmp_path / "t123/CHANGES.rst").read_text()
    ok, _ = repo.build(w.b, [], tmp_path / "rev2", reverse=[u2], source_only=True)
    assert ok and not (tmp_path / "rev2/pkg/beta.py").exists() and (tmp_path / "rev2/pkg/gamma.py").exists()
    assert b"tests/" not in repo.patch(u1)  # tests are never in a unit's oracle diff


def test_test_owners_are_the_last_unit_that_changed_the_function(tmp_path: Path) -> None:
    repo_dir = make_repo(tmp_path)
    w = release_windows(repo_dir, "toyrepo", since="2000-01-01", min_units=2)[0]
    owners = owners_of_tests(RepoGit(repo_dir), w.units, ["tests/test_core.py::test_alpha", "tests/test_core.py::test_base",
                                                      "tests/test_gamma.py::test_gamma[1]"])
    assert owners["tests/test_core.py::test_alpha"] == w.units[0].sha
    assert owners["tests/test_core.py::test_base"] is None  # unchanged: collateral, never a milestone
    assert owners["tests/test_gamma.py::test_gamma[1]"] == w.units[2].sha
    assert key_of_test("t/test_x.py::TestA::test_b[p-1]") == ("t/test_x.py", "TestA::test_b")


def test_f2p_groups_tests_into_ordered_milestones(tmp_path: Path) -> None:
    rec, _, _ = built_record(tmp_path)
    assert [(m["id"], m["tests"]) for m in rec["milestones"]] == [
        ("m01", ["tests/test_core.py::test_alpha"]), ("m02", ["tests/test_beta.py::test_beta"]), ("m03", ["tests/test_gamma.py::test_gamma"])]
    # B's test_core.py imports alpha, so test_base fails at A only through its module: collateral, a regression check
    # a test-only unit is not a milestone: its new test gates as a regression check, like test_base
    assert rec["p2p"] == [] and rec["collateral"] == ["tests/test_core.py::test_base", "tests/test_extra.py::test_alpha_again"]
    assert set(rec["oracle_files"]) == {"pkg/core.py", "pkg/beta.py", "pkg/gamma.py", "CHANGES.rst", "README.md"}
    assert set(rec["tests_b"]) == {"tests/test_core.py", "tests/test_beta.py", "tests/test_gamma.py", "tests/test_extra.py"}


def test_the_prober_finds_that_gamma_needs_beta_and_nothing_else(tmp_path: Path) -> None:
    rec, repo, _ = built_record(tmp_path)
    counts = asyncio.run(probe_dependencies(rec, repo, SB()))
    by = {m["id"]: m for m in rec["milestones"]}
    assert by["m01"]["probe"] == by["m02"]["probe"] == "alone" and by["m01"]["requires"] == []
    assert by["m03"]["probe"] == "minimized"
    assert by["m03"]["requires"] == ["m02"] and by["m03"]["requires_units"] == [rec["units"][1]["sha"]]
    assert by["m03"]["edges"] == {rec["units"][1]["sha"]: "executed"}  # its diff applies alone: the dependency is executed
    assert counts["milestone_edges"] == 1
    assert closure(rec, "m03") == {rec["units"][1]["sha"]}


def test_commit_revert_mutants_fail_their_milestone_and_spare_independent_ones(tmp_path: Path) -> None:
    rec, repo, base = built_record(tmp_path)
    asyncio.run(probe_dependencies(rec, repo, SB()))
    out = asyncio.run(revert_mutants(rec, repo, base, SB()))
    by = {m["milestone"]: m for m in out["mutants"]}
    m2 = by["m02"]  # beta reverted: beta fails, gamma (which needs beta) is not asserted, alpha still passes
    assert m2["status"] == "valid" and m2["observed"] == {"m01": True, "m02": False, "m03": False}
    assert m2["expected"] == {"m01": True, "m02": False, "m03": None}
    assert "pkg/beta.py" not in m2["solution"]["files"] and "pkg/gamma.py" in m2["solution"]["files"]  # the files that differ from A
    assert by["m01"]["status"] == "valid" and by["m01"]["observed"] == {"m01": False, "m02": True, "m03": True}
    assert out["counts"] == {"valid": 3}


@pytest.mark.parametrize("need", [set(), {3}, {1, 6}, {0, 2, 5}])
def test_ddmin_finds_the_minimal_set(need: set[int]) -> None:
    calls = []

    async def ok(s: list[int]) -> bool:
        calls.append(tuple(s))
        return need <= set(s)

    assert asyncio.run(ddmin(list(range(8)), ok)) == sorted(need)
    assert len(calls) <= 40


def test_unit_round_trips_json() -> None:
    u = Unit("a" * 40, "b" * 40, False, "2024-01-01", "s", ["pkg/x.py", "tests/test_x.py"])
    assert Unit(**u.__dict__) == u and u.qualifies
