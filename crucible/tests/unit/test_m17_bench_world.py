"""Benchmark v2 World fixes found while mining the new held-out repositories:

- A is installed again without dependencies after every other install, then its installed copy keeps only its metadata:
  one image serves every window with the same packaging, so site-packages held another window's A (w3lib, pyjwt: a
  later A is the solution; pint ships its tests, and the leak probe caught a later A's tests);
- the pytest pin follows A's ``minversion`` and coverage data goes to /tmp when ``addopts`` turns on pytest-cov
  (streamlink, jsonpickle);

- the World installs A with every extra, else with its test extras alone, else bare (fsspec: its ``full`` extras did not
  install, the bare fallback had no ``requests``, the conftest's ``importorskip`` failed collection);
- a World mismatch at B is tests that fail or error; skipped and xfailed tests are not (w3lib: 345 xfails read as
  354 failures and stopped every window)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from test_m17_w_miner import SB, _commit, _tag

from crucible.factory.lanes.windows import f2p_stage, window_files
from crucible.factory.lanes.windows_git import RepoGit, release_windows
from crucible.factory.lanes.windows_world import DROP_INSTALLED_SOURCES, EXTRAS_FALLBACK, ImageBook, world_dockerfile


def test_extras_fall_back_to_the_test_extras_before_bare() -> None:
    df = world_dockerfile("pkg", ["requests"], ["pytest-mock>=3"], needs={"apt": [], "pip_build": [], "extras": ["full", "s3"], "import_deps": ["PyYAML"]})
    line = next(x for x in df.splitlines() if "pip install --no-cache-dir \".[" in x)
    assert line.index('".[test,tests,testing,full,s3]"') < line.index('".[test,tests,testing]"') < line.rindex(" .")
    lines = df.splitlines()  # A again, without dependencies, after every other install: no other release of A stays installed
    last_a = max(i for i, x in enumerate(lines) if "--no-deps --force-reinstall ." in x)
    assert last_a > max(i for i, x in enumerate(lines) if x.startswith("RUN pip install --no-cache-dir") and "pytest==" not in x)
    assert "\n" not in DROP_INSTALLED_SOURCES and "importlib.metadata" in lines[last_a + 1] and lines[last_a + 2].startswith("RUN pip install --no-cache-dir pytest==8.3.3")


def test_the_fallback_is_part_of_every_world_tag(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "toy"\nversion = "1"\n')
    tag, info = ImageBook().tag_for("toy", tmp_path)
    assert EXTRAS_FALLBACK == "v2:test-extras-fallback,a-reinstalled-last,installed-sources-removed" and tag.startswith("crucible-lh-w:toy-")
    assert info["recipe"] == "3"


def test_the_installed_copy_of_a_keeps_only_its_metadata(tmp_path: Path) -> None:
    """The deletion script on an installed distribution (a dist-info RECORD on the path): one image serves several
    windows, so its installed A may be another window's A (w3lib, pyjwt: a later A is the solution; pint ships its
    tests). Only the metadata and what lies outside site-packages stay."""
    import sys

    site = tmp_path / "site"
    files = {"toyq/__init__.py": "X = 1\n", "toyq/core.py": "SOLUTION = 2\n", "toyq/testsuite/test_new.py": "def test_future():\n    pass\n",
             "toyq/data/table.txt": "rows\n", "other/keep.py": "Z = 3\n"}
    for rel, text in files.items():
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(text)
    di = site / "toyq-1.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: toyq\nVersion: 1\n")
    (di / "RECORD").write_text("".join(f"{rel},,\n" for rel in [*[f for f in files if f.startswith("toyq/")], "toyq-1.dist-info/METADATA",
                                                                  "toyq-1.dist-info/RECORD", "../../../bin/toyq"]))
    subprocess.run([sys.executable, "-c", DROP_INSTALLED_SOURCES, "toyq"], check=True, env={"PYTHONPATH": str(site), "PATH": "/usr/bin:/bin"})
    left = sorted(p.relative_to(site).as_posix() for p in site.rglob("*") if p.is_file())
    assert left == ["other/keep.py", "toyq-1.dist-info/METADATA", "toyq-1.dist-info/RECORD"]
    subprocess.run([sys.executable, "-c", DROP_INSTALLED_SOURCES, "not-installed"], check=True)  # an unknown name is a no-op


def _xfail_repo(root: Path) -> Path:
    repo = root / "xfailrepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    xfails = "import pytest\n\n" + "".join(f"\n@pytest.mark.xfail(strict=True)\ndef test_known_gap_{i}():\n    assert False\n\n" for i in range(30))
    _commit(repo, {"pkg/__init__.py": "", "pkg/core.py": "def base():\n    return 0\n", "tests/test_gaps.py": xfails,
                   "tests/test_core.py": "from pkg.core import base\n\n\ndef test_base():\n    assert base() == 0\n"}, "start", date="2024-01-01T00:00:00")
    _tag(repo, "v0.1", "2024-01-01T00:00:00")
    _commit(repo, {"pkg/a.py": "def a():\n    return 1\n", "tests/test_a.py": "from pkg.a import a\n\n\ndef test_a():\n    assert a() == 1\n"}, "Add a",
            date="2024-01-02T00:00:00")
    _commit(repo, {"pkg/b.py": "from pkg.a import a\n\n\ndef b():\n    return a() + 1\n",
                   "tests/test_b.py": "from pkg.b import b\n\n\ndef test_b():\n    assert b() == 2\n"}, "Add b", date="2024-01-03T00:00:00")
    _tag(repo, "v0.2", "2024-01-04T00:00:00")
    return repo


def test_xfailed_and_skipped_tests_are_not_a_world_mismatch(tmp_path: Path) -> None:
    repo_dir = _xfail_repo(tmp_path)
    win = release_windows(repo_dir, "xfailrepo", since="2000-01-01", min_units=2)[0]
    rec = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "timings": {}, "split": "heldout"}
    repo = RepoGit(repo_dir)
    base = tmp_path / "base"
    assert repo.build(win.a, [], base)[0]
    window_files(rec, repo)
    why = asyncio.run(f2p_stage(rec, repo, base, SB(), reruns=1))
    assert why is None, why  # 30 xfails among 4 passing tests: before the fix, "30 tests fail at B (environment mismatch)"
    assert rec["validation"]["post_hard_fail"] == 0 and sum("test_known_gap" in t for t in rec["excluded"]["post_fail"]) == 30
    assert [m["id"] for m in rec["milestones"]] == ["m01", "m02"]


def test_runner_fixes_from_the_pytest_configuration(tmp_path: Path) -> None:
    from crucible.factory.lanes.windows_world import pytest_needs, world_needs

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "pyproject.toml").write_text('[project]\nname = "a"\n[tool.pytest.ini_options]\nminversion = "7.0"\naddopts = "-ra"\n')
    assert pytest_needs(plain) == {}  # every other World keeps its tag
    sl = tmp_path / "streamlink"
    sl.mkdir()
    (sl / "pyproject.toml").write_text('[project]\nname = "s"\n[tool.pytest.ini_options]\nminversion = "8.4.0"\n')
    (sl / "setup.cfg").write_text("[tool:pytest]\naddopts = --cov=jsonpickle --cov-report=term\n")
    need = pytest_needs(sl)
    assert need == {"pytest": "pytest==8.4.2", "env": {"COVERAGE_FILE": "/tmp/.coverage"}}
    df = world_dockerfile("s", [], [], needs=world_needs(sl))
    assert "pytest==8.4.2 pytest-timeout==2.3.1" in df and "pytest==8.3.3" not in df and "ENV COVERAGE_FILE=/tmp/.coverage" in df
    assert ImageBook().tag_for("s", sl)[0] != ImageBook().tag_for("s", plain)[0]


def test_the_leak_probe_refuses_an_installed_copy_with_source(tmp_path: Path) -> None:
    import sys

    from crucible.factory.lanes.windows_world import LEAK_ASRC, judge_leak

    base = {"PIP_RC": "1", "DIST_FILES": "", "NAME_HITS": "", "GIT_COMMITS": "1", "GIT_TAGS": "base"}
    assert judge_leak({**base, "A_SOURCES": "4"}, version="1.2")["probes"]["no_installed_project_source"] == "fail"
    assert judge_leak({**base, "A_SOURCES": "0"}, version="1.2")["ok"]
    assert "no_installed_project_source" not in judge_leak(base, version="1.2")["probes"]  # a probe script without the count
    site = tmp_path / "site"
    (site / "toyq").mkdir(parents=True)
    (site / "toyq" / "core.py").write_text("SOLUTION = 1\n")
    (site / "toyq-1.dist-info").mkdir()
    (site / "toyq-1.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: toyq\nVersion: 1\n")
    (site / "toyq-1.dist-info" / "RECORD").write_text("toyq/core.py,,\ntoyq-1.dist-info/METADATA,,\n../../../bin/toyq,,\n")
    out = subprocess.run([sys.executable, "-c", LEAK_ASRC, "toyq"], check=True, capture_output=True, text=True, env={"PYTHONPATH": str(site)})
    assert out.stdout.strip() == "1"


def test_tox_testenv_deps_are_installed(tmp_path: Path) -> None:
    from crucible.factory.lanes.windows_world import tox_test_deps

    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup(name='g')\n")
    (tmp_path / "tox.ini").write_text("[tox]\nenvlist = py312\n\n[testenv]\ndeps =\n    pytest>=7.4,<10\n    pytest-asyncio>=0.21,<2\n"
                                      "    pytest-describe>=2,<4\n    py3{7,8,9},pypy39: typing-extensions>=4.1,<5\n    -rrequirements.txt\n"
                                      "commands =\n    pytest tests {posargs}\n\n[testenv:black]\ndeps = black==26.5.1\n")
    assert tox_test_deps(tmp_path) == ["pytest-asyncio", "pytest-describe"]
    tag, info = ImageBook().tag_for("g", tmp_path)
    assert "pytest-describe" in info["requirement_names"]
    (tmp_path / "tox.ini").unlink()
    assert ImageBook().tag_for("g", tmp_path)[0] != tag  # only the Worlds with tox deps change tag


def test_dependency_groups_install_one_by_one() -> None:
    df = world_dockerfile("s", [], ["starlette[full]", "blockbuster>=1.5.23", "mypy>=99.0", "pytest>=9.1.1"])
    lines = [x for x in df.splitlines() if "blockbuster" in x or "mypy" in x]
    assert lines == ["RUN pip install --no-cache-dir 'blockbuster>=1.5.23' || pip install --no-cache-dir blockbuster || true",
                     "RUN pip install --no-cache-dir 'mypy>=99.0' || pip install --no-cache-dir mypy || true"]
    assert "pytest>=9.1.1" not in df  # the runner's pins stay the base's
