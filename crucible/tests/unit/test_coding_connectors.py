"""M3 coding-pack connectors: ci_logs parsers and connector, policy_docs, and (when the local data is present) the
recorded frontier runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from crucible_pack_coding.connectors.ci_logs import CILogConnector, detect, flaky, parse, summary_counts
from crucible_pack_coding.connectors.policy_docs import PolicyDocConnector, config_rules, modality, sections

TASKS = Path(__file__).resolve().parents[3] / "migration_tasks"

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
 <testcase classname="tests.test_a" name="test_ok" time="0.01"/>
 <testcase classname="tests.test_a" name="test_bad" time="0.20"><failure message="assert 1 == 2">trace</failure></testcase>
 <testcase classname="tests.test_b" name="test_boom"><error message="ImportError: cuda"/></testcase>
 <testcase classname="tests.test_b" name="test_gpu"><skipped message="no gpu"/></testcase>
</testsuite></testsuites>"""

CTEST = """Test project /build
    Start 1: test_common
1/3 Test #1: test_common ......................   Passed    0.52 sec
    Start 2: test_kdtree
2/3 Test #2: test_kdtree ......................***Failed    1.10 sec
3/3 Test #3: test_gpu_octree ..................***Not Run   0.00 sec

67% tests passed, 1 tests failed out of 3

The following tests FAILED:
\t  2 - test_kdtree (Failed)
"""

GTEST = """[==========] Running 3 tests from 1 test suite.
[ RUN      ] Octree.Build
[       OK ] Octree.Build (12 ms)
[ RUN      ] Octree.Radius
[  FAILED  ] Octree.Radius (3 ms)
[ RUN      ] Octree.DISABLED_Big
[  SKIPPED ] Octree.DISABLED_Big (0 ms)
[==========] 3 tests from 1 test suite ran. (15 ms total)
[  PASSED  ] 1 test.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] Octree.Radius
"""

PYTEST = """tests/test_ops.py::test_add PASSED                                   [ 33%]
tests/test_ops.py::test_scatter[mean] FAILED                         [ 66%]
tests/test_ops.py::test_gather SKIPPED (needs gpu)                   [100%]
=========================== short test summary info ============================
FAILED tests/test_ops.py::test_scatter[mean] - assert False
=================== 1 failed, 1 passed, 1 skipped in 2.31s ====================
"""

NINJA = """[12/40] Building CUDA object gpu/kinfu/CMakeFiles/pcl_gpu_kinfu.dir/src/cuda/tsdf.cu.o
FAILED: gpu/kinfu/CMakeFiles/pcl_gpu_kinfu.dir/src/cuda/marching_cubes.cu.o
/usr/local/cuda/bin/nvcc -c marching_cubes.cu
/src/gpu/kinfu/src/cuda/marching_cubes.cu(88): error: identifier "__shfl" is undefined
ninja: build stopped: subcommand failed.
"""


def test_detect_and_parse_formats():
    assert [detect(x) for x in (JUNIT, CTEST, GTEST, PYTEST, NINJA, "hello")] == ["junit", "ctest", "gtest", "pytest", "build", "unknown"]
    assert {t["name"]: t["status"] for t in parse(JUNIT)[1]} == {"tests.test_a::test_ok": "pass", "tests.test_a::test_bad": "fail",
                                                                "tests.test_b::test_boom": "error", "tests.test_b::test_gpu": "skip"}
    assert {t["name"]: t["status"] for t in parse(CTEST)[1]} == {"test_common": "pass", "test_kdtree": "fail", "test_gpu_octree": "skip"}
    assert {t["name"]: t["status"] for t in parse(GTEST)[1]} == {"Octree.Build": "pass", "Octree.Radius": "fail", "Octree.DISABLED_Big": "skip"}
    py = {t["name"]: t for t in parse(PYTEST)[1]}
    assert py["tests/test_ops.py::test_scatter[mean]"]["status"] == "fail" and py["tests/test_ops.py::test_scatter[mean]"]["message"] == "assert False"
    assert summary_counts(PYTEST) == {"failed": 1, "passed": 1, "skipped": 1}
    build = parse(NINJA)[1]
    assert build == [{"name": "build:gpu/kinfu/CMakeFiles/pcl_gpu_kinfu.dir/src/cuda/marching_cubes.cu.o", "status": "fail",
                      "message": '/src/gpu/kinfu/src/cuda/marching_cubes.cu:88: identifier "__shfl" is undefined', "duration_s": None}]


def _write(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


def test_ci_connector_outcomes_infra_and_flaky(tmp_path):
    sha = "a" * 40
    _write(tmp_path, f"{sha}/run1/junit.xml", JUNIT)
    _write(tmp_path, f"{sha}/run2/junit.xml", JUNIT.replace('<failure message="assert 1 == 2">trace</failure>', ""))
    _write(tmp_path, "b/ctest.log", CTEST.replace("Passed    0.52", "Passed    0.50") + "\nNo space left on device\n")
    _write(tmp_path, "c/pytest.log", "tests/x.py::test_y PASSED\n==== 1 passed in 0.1s ====\n")
    c = CILogConnector(tmp_path)
    traces = {r.key: c.normalize(c.fetch(r)) for r in c.discover({"glob": "**/*.*"})}
    r1 = traces[f"{sha}/run1/junit.xml"]
    assert r1.outcome.label == "bad" and r1.events[0].data["commit"] == sha and r1.events[0].state_ref == f"git:{sha}"
    assert r1.events[0].data["failure_kind"] == "test" and all(e.reply_to == 0 for e in r1.events[1:])
    infra = traces["b/ctest.log"]
    assert infra.outcome.label == "unknown" and infra.events[0].data["failure_kind"] == "infra"  # P16: masked, never scored 0
    assert traces["c/pytest.log"].outcome.label == "good"
    assert flaky(traces.values()) == {sha: ["tests.test_a::test_bad"]}  # error in both runs is not flaky; fail-then-pass is


def test_ci_grade_report(tmp_path):
    report = {"task": "t", "reward": 0.0, "gates": {"static": {"pass": True, "violations": []},
                                                   "ops": {"pass": False, "summary": ["1 failed, 3 passed in 1s"], "failed": ["FAILED test_h.py::test_z[a] - assert False"]}}}
    _write(tmp_path, "t/runs/r/grade_report.json", json.dumps(report))
    c = CILogConnector(tmp_path)
    t = c.normalize(c.fetch(next(c.discover({"glob": "*/runs/*/grade_report.json"}))))
    assert [e.text for e in t.events[1:]] == ["gate:static: pass", "gate:ops: fail", "ops/test_h.py::test_z[a]: fail"]
    assert t.outcome.label == "bad" and t.events[0].data["reward"] == 0.0 and t.events[0].data["passed"] == 1


STYLE_RST = """Style Guide
===========

1. Naming
---------

All files should be **under_scored**. See `the docs <https://example.org/a.b>`_ for e.g. details.

.. code-block:: cpp

  // you must not see this
  int x;

* Never use ``using namespace std`` in headers.
* Header files have the extension **.h**

2. Checklist
------------

1. Code compiles with -Werror.
"""

CONTRIB_MD = """# Contributing

<!-- you must ignore comments -->

## Pull requests

Pull requests must target `master`. Please run clang-format before pushing.

```bash
# do not parse this
```
"""


def test_policy_sections_and_rules(tmp_path):
    secs = sections(STYLE_RST.splitlines())
    assert [s["heading"] for s in secs] == [["Style Guide"], ["Style Guide", "1. Naming"], ["Style Guide", "2. Checklist"]]
    units = [u[2] for s in secs for u in s["units"]]
    assert "All files should be under_scored." in units and not any("must not see" in u for u in units)
    assert modality("Never use using namespace std in headers.") == "must_not" and modality("Header files have the extension .h") is None
    _write(tmp_path, "docs/STYLE.rst", STYLE_RST)
    _write(tmp_path, "CONTRIBUTING.md", CONTRIB_MD)
    c = PolicyDocConnector()
    traces = {r.key: c.normalize(c.fetch(r)) for r in c.discover({"root": tmp_path, "glob": "**/*.*"})}
    rules = {e.text: e.data["modality"] for t in traces.values() for e in t.events if e.type == "policy_rule"}
    assert rules == {"All files should be under_scored.": "should", "Never use using namespace std in headers.": "must_not",
                     "Code compiles with -Werror.": "must", "Pull requests must target master.": "must",
                     "Please run clang-format before pushing.": "should"}
    style = traces["docs/STYLE.rst"]
    rule = next(e for e in style.events if e.text.startswith("Never"))
    assert style.events[rule.reply_to].text == "Style Guide > 1. Naming" and rule.target.span == [14, 14]


def test_config_rules():
    fmt = config_rules(".clang-format", ["---", "ColumnLimit: 88", "BraceWrapping:", "  AfterFunction: true", "# c: 1"])
    assert [r[1] for r in fmt] == ["ColumnLimit = 88", "BraceWrapping.AfterFunction = true"]
    tidy = config_rules(".clang-tidy", ["Checks: >", "  -*,", "  modernize-use-nullptr,", "  -readability-braces"])
    assert [(r[2]["check"], r[2]["enabled"]) for r in tidy] == [("*", False), ("modernize-use-nullptr", True), ("readability-braces", False)]
    owners = config_rules("CODEOWNERS", ["# owners", "gpu/** @gpu-team @alice"])
    assert owners == [(2, "gpu/** is owned by @gpu-team @alice", {"pattern": "gpu/**", "owners": ["@gpu-team", "@alice"]})]


@pytest.mark.skipif(not (TASKS / "task01_prims_omp" / "runs").exists(), reason="recorded frontier runs not present")
def test_agent_runs_match_recorded_results():
    from crucible_pack_coding.connectors.agent_runs import AgentRunConnector

    c = AgentRunConnector(TASKS)
    refs = list(c.discover())
    assert len(refs) >= 11
    t = c.normalize(c.fetch(next(r for r in refs if r.key == "task01_prims_omp/fable-5.1-run1")))
    assert t.outcome.label == "good" and t.outcome.details["reward"] == 1.0
    assert t.events[-1].type == "grade_gate" and any(e.type == "final_patch" for e in t.events)
    calls = [e for e in t.events if e.type == "tool_call"]
    results = [e for e in t.events if e.type == "tool_result"]
    assert calls and len(results) == sum(1 for r in results if r.reply_to is not None)
