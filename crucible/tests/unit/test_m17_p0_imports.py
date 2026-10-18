"""P0 F8 (report 11): import-tolerant grading keeps per-test milestones when B's test file imports, at module level, names
the window adds late.

The synthetic repository: ``v0.1`` has ``base()``; unit 1 adds ``alpha()``; unit 2 adds ``beta()`` in its own module,
calling ``alpha`` (an executed dependency); unit 3 adds ``late()``. B's ``tests/test_core.py`` imports all four names at
module level. Strict grading: no B test passes before unit 3, so every test has one owner (the window collapses to one
milestone, as semver's ``cmd_*`` did). Import-tolerant: ``test_base`` passes at A (a regression check), and each other test
belongs to the unit that made it pass."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from test_m17_w_miner import SB, _commit, _tag

from crucible.factory.lanes.windows import WindowLane, WindowStore, f2p_stage, grader_tests, mined_of, probe_dependencies, revert_mutants, window_files
from crucible.factory.lanes.windows_git import RepoGit, release_windows
from crucible.factory.lanes.windows_imports import project_roots, tolerant_source, tolerant_tests

B_TEST = '''"""B's tests."""
from __future__ import annotations

import pytest

from pkg.beta import beta
from pkg.core import alpha, base
from pkg.late import late


def test_base():
    assert base() == 0


def test_alpha():
    assert alpha() == 1


def test_beta():
    assert beta() == 2


def test_late():
    assert late() == 3


@pytest.mark.parametrize("fn", [alpha, late])
def test_callable(fn):
    assert fn() in (1, 3)
'''


def make_late_repo(root: Path) -> Path:
    repo = root / "laterepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    core = "def base():\n    return 0\n"
    _commit(repo, {"pkg/__init__.py": "", "pkg/core.py": core, "tests/test_core.py": "from pkg.core import base\n\n\ndef test_base():\n    assert base() == 0\n"},
            "start", date="2024-01-01T00:00:00")
    _tag(repo, "v0.1", "2024-01-01T00:00:00")
    _commit(repo, {"pkg/core.py": core + "\n\ndef alpha():\n    return 1\n",
                   "tests/test_core.py": "from pkg.core import alpha, base\n\n\ndef test_base():\n    assert base() == 0\n\n\ndef test_alpha():\n    assert alpha() == 1\n"},
            "Add alpha", date="2024-01-02T00:00:00")
    _commit(repo, {"pkg/beta.py": "from pkg.core import alpha\n\n\ndef beta():\n    return alpha() + 1\n",
                   "tests/test_beta_unit.py": "from pkg.beta import beta\n\n\ndef test_beta_unit():\n    assert beta() == 2\n"},
            "Add beta on top of alpha", date="2024-01-03T00:00:00")
    _commit(repo, {"pkg/late.py": "def late():\n    return 3\n", "tests/test_core.py": B_TEST}, "Add late, one test module", date="2024-01-04T00:00:00")
    _tag(repo, "v0.2", "2024-01-05T00:00:00")
    return repo


def record(tmp: Path, *, tolerant: bool) -> tuple[dict, RepoGit, Path, str | None]:
    repo_dir = make_late_repo(tmp)
    win = release_windows(repo_dir, "laterepo", since="2000-01-01", min_units=2)[0]
    rec = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "timings": {}, "split": "train", "import_tolerant": tolerant}
    repo = RepoGit(repo_dir)
    base = tmp / "base"
    assert repo.build(win.a, [], base)[0]
    window_files(rec, repo)
    why = asyncio.run(f2p_stage(rec, repo, base, SB(), reruns=1))
    return rec, repo, base, why


def test_strict_grading_collapses_ownership_to_the_last_name(tmp_path: Path) -> None:
    rec, _, _, why = record(tmp_path, tolerant=False)
    assert why is None
    owned = {m["subject"]: sorted(t.split("::", 1)[1] for t in m["tests"]) for m in rec["milestones"]}
    # every test of the shared module is owned by "Add late" (the semver collapse); alpha owns nothing
    assert owned == {"Add beta on top of alpha": ["test_beta_unit"],
                     "Add late, one test module": ["test_alpha", "test_base", "test_beta", "test_callable[alpha]", "test_callable[late]", "test_late"]}


def test_tolerant_grading_keeps_one_milestone_per_unit(tmp_path: Path) -> None:
    rec, repo, base, why = record(tmp_path, tolerant=True)
    assert why is None, why
    owned = {m["subject"]: sorted(t.split("::", 1)[1] for t in m["tests"]) for m in rec["milestones"]}
    assert owned == {"Add alpha": ["test_alpha", "test_callable[alpha]"],
                     "Add beta on top of alpha": ["test_beta", "test_beta_unit"],
                     "Add late, one test module": ["test_callable[late]", "test_late"]}
    assert "tests/test_core.py::test_base" in rec["p2p"]  # it never needed a late name: a regression check, not fail-to-pass
    assert rec["import_roots"] == ["pkg"]
    # the prober sees the same files: beta needs alpha (executed), late needs nothing
    asyncio.run(probe_dependencies(rec, repo, SB()))
    by = {m["subject"]: m for m in rec["milestones"]}
    assert by["Add beta on top of alpha"]["requires"] == [by["Add alpha"]["id"]]
    assert by["Add beta on top of alpha"]["edges"] == {rec["units"][0]["sha"]: "executed"}
    assert by["Add late, one test module"]["requires"] == []
    out = asyncio.run(revert_mutants(rec, repo, base, SB()))
    assert out["counts"] == {"valid": 3}


def test_a_tolerant_window_grades_each_milestone_on_its_own(tmp_path: Path) -> None:
    rec, repo, base, _ = record(tmp_path, tolerant=True)
    asyncio.run(probe_dependencies(rec, repo, SB()))
    rec.update(asyncio.run(revert_mutants(rec, repo, base, SB())))
    rec["world"] = {"image": "local", "digest": "sha256:" + "0" * 64}
    store = WindowStore(tmp_path / "store")
    d = store.dir(rec["id"])
    d.mkdir(parents=True)
    subprocess.run(["cp", "-R", str(base), str(d / "base")], check=True)
    store.save_window(rec)
    store.save(mined_of(rec, d / "base"))
    lane = WindowLane(store, repo_dirs={"laterepo": str(repo.dir)}, sandboxes=lambda image: SB())
    task = lane.materialize({"world": {"task": rec["id"]}, "prompt": "", "spec": {"requirements": []}, "checkset": {"checks": []}})
    assert task.import_tolerant and task.guarded_imports == {"tests/test_core.py": ["beta", "alpha", "base", "late"],
                                                          "tests/test_beta_unit.py": ["beta"]}
    ids = {m["subject"]: m["id"] for m in rec["milestones"]}
    a, u1 = rec["a"], rec["units"][0]["sha"]
    alpha_only = subprocess.run(["git", "-C", str(repo.dir), "diff", a, u1, "--", "pkg"], capture_output=True, text=True, check=True).stdout
    st = asyncio.run(task.milestone_status(alpha_only))
    assert st == {ids["Add alpha"]: True, ids["Add beta on top of alpha"]: False, ids["Add late, one test module"]: False}
    rep = asyncio.run(task.grade_patch(alpha_only, trail=[]))
    assert rep.meta["phi"] == pytest.approx(1 / 3) and rep.meta["import_tolerant"]
    # the same window graded strictly: the module import fails, nothing passes
    strict = type(task)(lane, task.mined, task.window, task_id="strict", prompt="", requirements=[], spans={}, checkset=task.checkset, import_tolerant=False)
    assert not any(asyncio.run(strict.milestone_status(alpha_only)).values())
    assert all(asyncio.run(task.milestone_status(task.oracle_patch())).values())  # the oracle passes every milestone tolerantly too


def test_tolerant_source_leaves_other_statements_alone() -> None:
    text = ("from __future__ import annotations\nimport os, json\ntry:\n    from pkg import x\nexcept ImportError:\n    x = None\n"
            "from . import helpers\nfrom pkg import *\nimport pkg.sub as s; y = 1\nfrom pkg.core import a as b  # note\n")
    new, names = tolerant_source(text, {"pkg"})
    assert names == ["b"]  # only the lone, absolute, named project import
    assert "import os, json\ntry:\n    from pkg import x\n" in new and "from pkg import *\n" in new and "import pkg.sub as s; y = 1\n" in new
    assert new.index("from __future__") == 0 and "except ImportError:\n    b = _CrucibleMissing('pkg.core', 'a')" in new
    assert tolerant_source(new, {"pkg"}) == (new, [])  # idempotent
    files, guarded = tolerant_tests({"t.py": text, "data.txt": "from pkg import x"}, {"pkg"})
    assert files["data.txt"] == "from pkg import x" and guarded == {"t.py": ["b"]}


def test_grader_tests_follow_the_record_mode(tmp_path: Path) -> None:
    (tmp_path / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "lib" / "__init__.py").write_text("")
    (tmp_path / "setup.py").write_text("")
    assert project_roots(tmp_path, ["newpkg/__init__.py", "src/other/x.py", "README.md"]) == {"lib", "newpkg", "other"}
    rec = {"tests_b": {"tests/test_x.py": "from lib import y\n"}, "oracle_files": {}}
    assert grader_tests(rec, tmp_path) == (rec["tests_b"], {})  # a window mined before P0 (no flag) stays strict
    files, guarded = grader_tests({**rec, "import_tolerant": True}, tmp_path)
    assert guarded == {"tests/test_x.py": ["y"]} and "_CrucibleMissing('lib', 'y')" in files["tests/test_x.py"]
