"""P5 open tasks, Type B (swarm-pipeline/00 §2, §7): perf-commit detection, the traced entries of the workload, fresh
inputs per grade, the speedup statistics, the reward shape and floors, and the honest S6 label (gate edges, no executed
dependency). The trace plugin and the bench script run here on the host interpreter; the containers are exercised by
``scripts/lh_gen.py open``."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.factory.generators.base import git, out
from crucible.factory.generators.open import BENCH, TRACE_PLUGIN, OpenTask, hot_functions, perf_units, speedup_stats
from crucible.factory.lanes.windows import WindowStore
from crucible.factory.lanes.windows_git import RepoGit, Unit
from crucible.factory.longtask import check_longtask

CORE = '''def hot(x):
    return sorted(str(x))


def entry(xs, sep=","):
    return sep.join("".join(hot(x)) for x in xs)


class K:
    def method(self, x):
        return hot(x)
'''


def _commit(repo: Path, msg: str, files: dict[str, str]) -> str:
    for p, t in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(t)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg, env={"GIT_AUTHOR_DATE": "2024-01-01T00:00:00+00:00", "GIT_COMMITTER_DATE": "2024-01-01T00:00:00+00:00"})
    return out(git(repo, "rev-parse", "HEAD"))


def test_perf_commits_and_their_hot_functions(tmp_path: Path) -> None:
    r = tmp_path / "r"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    _commit(r, "init", {"pkg/__init__.py": "", "pkg/core.py": CORE, "tests/test_core.py": "def test_x():\n    pass\n"})
    p = _commit(r, "Speed up hot by caching str", {"pkg/core.py": CORE.replace("sorted(str(x))", "sorted(repr(x))")})
    _commit(r, "Faster CI with tox wheels", {"tox.ini": "[tox]\n"})  # CI/tox, and a config file: not a perf source
    _commit(r, "Make the build faster", {"pyproject.toml": "[project]\nname='pkg'\n", "pkg/core.py": CORE})
    _commit(r, "Add docs", {"README.md": "x\n"})
    units = perf_units(r, since="2000-01-01")
    assert [u.sha for u in units] == [p]
    assert hot_functions(RepoGit(r), units[0]) == [("pkg/core.py", "hot")]


def _run_trace(tmp_path: Path) -> dict:
    proj = tmp_path / "proj"
    (proj / "pkg").mkdir(parents=True)
    (proj / "pkg" / "__init__.py").write_text("")
    (proj / "pkg" / "core.py").write_text(CORE)
    (proj / "tests").mkdir()
    (proj / "tests" / "test_core.py").write_text("from pkg.core import entry, K\n\n\ndef test_a():\n    assert entry(['b', 'a1'])\n\n\n"
                                                 "def test_b():\n    assert entry([3, 21], sep='-')\n\n\ndef test_m():\n    assert K().method('zz')\n")
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "p5trace.py").write_text(TRACE_PLUGIN)
    env = {**os.environ, "PYTHONPATH": f"{proj}:{plug}", "P5_ROOTS": "pkg", "P5_HOT": json.dumps([["pkg/core.py", "hot"]])}
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "p5trace", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "tests"], cwd=proj, env=env,
                       capture_output=True, text=True, timeout=120, check=False)
    m = re.search(r"@@P5TRACE@@(.*?)@@P5END@@", r.stdout, re.DOTALL)
    assert m, r.stdout[-800:] + r.stderr[-800:]
    return json.loads(m.group(1))


def test_the_trace_records_entries_that_reach_the_changed_code(tmp_path: Path) -> None:
    calls = _run_trace(tmp_path)
    e = calls["pkg.core:entry"]
    assert e["hot"] == 2 and not e["method"] and len(e["samples"]) == 2
    assert e["samples"][0] == [[["b", "a1"], ","], {"__d__": {}}]  # [args, kwargs], plain data only
    m = calls["pkg.core:K.method"]
    assert m["method"] and m["samples"] == [] and m["unsimple"] == 1  # a method is recorded, never sampled (its receiver is not plain data)


def _bench(tmp_path: Path, seed: int, module_text: str) -> dict:
    d = tmp_path / f"b{seed}{len(module_text)}"
    (d / "pkg").mkdir(parents=True)
    (d / "pkg" / "__init__.py").write_text("")
    (d / "pkg" / "core.py").write_text(module_text)
    spec = {"entries": [{"module": "pkg.core", "function": "entry", "samples": [[[["b", "a1"], ","], {"__d__": {}}], [[[3, 21], "-"], {"__d__": {}}]],
                         "n": 50, "scale": 3}]}
    (d / "spec.json").write_text(json.dumps(spec))
    (d / "bench.py").write_text(BENCH)
    r = subprocess.run([sys.executable, str(d / "bench.py"), str(d / "spec.json"), str(seed)], cwd=d, env={**os.environ, "PYTHONPATH": str(d)},
                       capture_output=True, text=True, timeout=120, check=False)
    m = re.search(r"@@P5BENCH@@(.*?)@@END@@", r.stdout, re.DOTALL)
    assert m, r.stderr[-800:]
    return json.loads(m.group(1))


def test_fresh_inputs_per_grade_and_the_differential(tmp_path: Path) -> None:
    a, a2, b = _bench(tmp_path, 1, CORE), _bench(tmp_path, 1, CORE + "\n"), _bench(tmp_path, 2, CORE)
    assert len(a["hashes"]) == 50 and a["raised"] == 0 and a["t"] > 0
    assert a["hashes"] == a2["hashes"]  # same seed, same inputs, same results
    assert a["hashes"] != b["hashes"]  # another grade draws other inputs: a memo of the last grade cannot pay
    memo = CORE + "\n_c = {}\n_o = entry\n\n\ndef entry(*a, **k):\n    key = repr((a, sorted(k.items())))\n    if key not in _c:\n        _c[key] = _o(*a, **k)\n    return _c[key]\n"
    assert _bench(tmp_path, 1, memo)["hashes"] == a["hashes"]  # memoisation is correct, just not faster on fresh inputs
    wrong = CORE + "\n\ndef entry(*a, **k):\n    return None\n"
    assert _bench(tmp_path, 1, wrong)["hashes"] != a["hashes"]  # a fast wrong answer fails the differential


def test_speedup_statistics() -> None:
    st = speedup_stats([2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0])
    assert st["s"] == pytest.approx(2.0) and st["lo"] == pytest.approx(2.0) and st["cv_ref"] == 0.0
    noisy = speedup_stats([1.0, 1.3, 0.8, 1.1, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert noisy["lo"] < 1.0 < noisy["hi"] and noisy["cv_ref"] > 0.1
    assert speedup_stats([], [])["n"] == 0
    hiccup = speedup_stats([1.3] * 8 + [1.3], [1.0] * 8 + [2.0])  # one slow candidate round out of nine
    assert hiccup["n"] == 7 and hiccup["lo"] > 1.2  # trimmed: one host hiccup never decides the verdict


def _task(tmp_path: Path) -> OpenTask:
    store = WindowStore(tmp_path / "store", split="train")
    base = store.dir("op-x") / "base"
    (base / "pkg").mkdir(parents=True)
    (base / "pkg" / "__init__.py").write_text("")
    (base / "pkg" / "core.py").write_text(CORE)
    spec = {"entries": [{"module": "pkg.core", "function": "entry", "samples": [[[["b"], ","], {"__d__": {}}]], "n": 10, "scale": 2}]}
    rec = {"id": "op-x", "repo": "pkg", "a": "a" * 40, "b": "b" * 40, "world": {"image": "img", "digest": "d"}, "regression": ["tests/test_core.py::test_x"],
           "oracle_files": {"pkg/core.py": CORE.replace("str(x)", "repr(x)")}, "roots": ["pkg"], "sources": ["src-real_perf-1"],
           "open": {"s_bar": 4.0, "spec": spec, "headroom": "proven"}}
    return OpenTask(SimpleNamespace(store=store, sandboxes=None), rec)  # type: ignore[arg-type]


def test_reward_shape_floors_and_the_honest_s6_label(tmp_path: Path) -> None:
    t = _task(tmp_path)
    ok = {"m_correct": True, "m_faster": True, "m_bar": True}
    assert t.phi(ok) == 1.0 and t.phi({"m_correct": True, "m_faster": False, "m_bar": False}) == 0.0  # the correct start state has Phi 0
    assert t.phi({"m_correct": False, "m_faster": True, "m_bar": True}) == 0.0  # speed without correctness never counts
    assert t.reward_of({"milestones": ok, "bench": {"s": 4.0}}) == pytest.approx(1.0)
    assert t.reward_of({"milestones": ok, "bench": {"s": 16.0}}) == pytest.approx(2.0)  # uncapped: past the known best
    assert t.reward_of({"milestones": {"m_correct": True, "m_faster": False}, "bench": {"s": 1.02}}) == 0.0
    assert t.bar_value() == pytest.approx(2.0) and t.simplify("baseline_bar").bar_value() == 1.0
    with pytest.raises(ValueError):
        t.simplify("locate")
    problems = check_longtask(t, long_horizon=True, need_prompt=False)
    assert any(p.startswith("short:") for p in problems)  # gate edges are not executed dependencies: S6 reports n/a, never a pass
    assert not [p for p in problems if not p.startswith("short:")]
    for kind in ("memo", "replay", "fast_wrong"):
        for text in t._wrap(kind).values():
            compile(text, "core.py", "exec")
    assert [m.milestone for m in t.mutants()] == ["m_correct", "m_faster"] and {f.id for f in t.floors()} >= {"no_op", "memo", "replay"}
    with pytest.raises(ValueError):
        t.card()
    assert math.isclose(t.s_bar, 4.0) and t.world["history"] == "cut" and t.world["network"] == "none"


def test_unit_type_is_reused(tmp_path: Path) -> None:
    u = Unit(sha="s", parent="p", merge=False, date="", subject="x", files=["pkg/core.py"])
    assert u.src_files == ["pkg/core.py"]
