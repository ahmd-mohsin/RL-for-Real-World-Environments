"""P2 re-mine support (``lanes.windows``): test ids keyed by the file pytest collected them from (the cachetools/markdown
regression-check defect P0 left open), volatile ids dropped, and the window-for-window re-mine refusing a split mix."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from crucible.factory.lanes.executable import LocalSandbox, parse_junit
from crucible.factory.lanes.windows import COLLECT_ALL, HeldOutError, WindowStore, normalize_ids, remine, run_tests, volatile_ids


def test_normalize_ids() -> None:
    targets = ["tests/test_cache.py", "markdown/test_tools.py", "tests/test_legacy.py"]
    got = normalize_ids({"tests/__init__.py::tests::test_cache::CacheTest::test_defaults": "passed",
                         "markdown/test_tools.py::tests::test_legacy::TestBasic::test_x[a::b-1]": "failed",
                         "tests/test_cache.py::CacheTest::test_y": "passed", "tests/test_cache.py::test_f": "passed",
                         "other/__init__.py::nowhere::Cls::t": "passed"}, targets)
    assert got == {"tests/test_cache.py::CacheTest::test_defaults": "passed", "tests/test_legacy.py::TestBasic::test_x[a::b-1]": "failed",
                   "tests/test_cache.py::CacheTest::test_y": "passed", "tests/test_cache.py::test_f": "passed", "other/__init__.py::nowhere::Cls::t": "passed"}
    assert volatile_ids({"a": "passed"}, ["a", "b[10:51:27]"]) == ["b[10:51:27]"] and volatile_ids(None, ["a"]) == [] and volatile_ids({}, ["a"]) == []


def test_inherited_mixin_tests_get_their_collecting_file(tmp_path: Path) -> None:
    """The real defect, executed: a test class inheriting its methods from a mixin in ``tests/__init__.py`` reports the
    mixin's file in JUnit; after ``run_tests`` the id names the file that was run, so a regression check reproduces."""
    t = tmp_path / "t"
    (t / "tests").mkdir(parents=True)
    (t / "tests" / "__init__.py").write_text("class Mixin:\n    def test_shared(self):\n        assert True\n")
    (t / "tests" / "test_cache.py").write_text("from tests import Mixin\n\n\nclass TestCache(Mixin):\n    def test_own(self):\n        assert True\n")
    subprocess.run(["git", "init", "-q"], cwd=t, check=True)
    sb = LocalSandbox(extra_args=COLLECT_ALL)
    raw = asyncio.run(sb.run(t, ["tests/test_cache.py"]))
    assert any(k.startswith("tests/__init__.py::") for k in raw.results)  # the parser alone keeps the mixin's file
    tr = asyncio.run(run_tests(sb, t, {}, {}, ["tests/test_cache.py"]))
    assert set(tr.results) == {"tests/test_cache.py::TestCache::test_shared", "tests/test_cache.py::TestCache::test_own"}
    assert parse_junit(tmp_path / "missing.xml") is None


def test_remine_refuses_a_split_mix(tmp_path: Path) -> None:
    src = WindowStore(tmp_path / "a", split="train")
    dst = WindowStore(tmp_path / "b", split="heldout")
    with pytest.raises(HeldOutError):
        asyncio.run(remine(src, dst, {}))
    assert asyncio.run(remine(src, WindowStore(tmp_path / "c", split="train"), {})) == {}
