"""M17 owner S: the executable lane (factory/lanes/executable.py). A tiny git repository with a real fix + test commit is
mined, validated (FAIL_TO_PASS before/after with reruns), turned into an accept-contract task and graded on the host
interpreter (``LocalSandbox``: the Docker sandbox runs the same pytest command). Adversarial cases: test edits, a
conftest that forces passes, an atexit hook that rewrites the report, edits that do not apply, paths that escape."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.lanes.executable import (
    GUARD_PLUGIN,
    REGRESSION_REQ,
    REGRESSION_TEXT,
    ExecLane,
    LaneStore,
    LocalSandbox,
    apply_solution,
    candidates,
    compose_prompt,
    edits_from_change,
    fingerprint,
    is_protected,
    junit_from_stdout,
    normalize_solution,
    oracle_solution,
    parse_edits,
    regression_pool,
    validate_candidate,
)

CORE_V1 = "def add(a, b):\n    return a + b\n\n\ndef clamp(x, lo, hi):\n    return x\n"
CORE_V2 = "def add(a, b):\n    return a + b\n\n\ndef clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n"
TEST_V1 = "from pkg.core import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
TEST_V2 = (TEST_V1 + "\n\ndef test_clamp_low():\n    from pkg.core import clamp\n    assert clamp(-5, 0, 10) == 0\n"
           "\n\ndef test_clamp_high():\n    from pkg.core import clamp\n    assert clamp(50, 0, 10) == 10\n")


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "toylib"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg" / "__init__.py").write_text("")
    (r / "pkg" / "core.py").write_text(CORE_V1)
    (r / "tests" / "test_core.py").write_text(TEST_V1)
    _git(r, "init", "-q")
    _git(r, "add", ".")
    _git(r, "commit", "-qm", "init")
    (r / "pkg" / "core.py").write_text(CORE_V2)
    (r / "tests" / "test_core.py").write_text(TEST_V2)
    _git(r, "commit", "-qam", "Fix clamp to bound values (#12)")
    (r / "README.md").write_text("docs\n")
    _git(r, "add", ".")
    _git(r, "commit", "-qm", "docs only")
    return r


@pytest.fixture
def mined(repo: Path, tmp_path: Path) -> Any:
    store = LaneStore(tmp_path / "lane")
    (c,) = candidates(repo, repo="toylib", since="2000-01-01")
    t, why, _ = asyncio_run(validate_candidate(c, store, LocalSandbox(), reruns=2))
    assert t is not None, why
    return store, t


def asyncio_run(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def test_candidates_need_a_source_change_with_tests(repo: Path) -> None:
    cs = candidates(repo, repo="toylib", since="2000-01-01")
    assert [c.subject for c in cs] == ["Fix clamp to bound values (#12)"]  # the docs-only and the root commits are not candidates
    assert cs[0].src_files == ["pkg/core.py"] and cs[0].test_files == ["tests/test_core.py"]


def test_validation_finds_fail_to_pass_and_binds_the_observation(mined: Any) -> None:
    _, t = mined
    assert t.f2p == ["tests/test_core.py::test_clamp_high", "tests/test_core.py::test_clamp_low"]
    assert t.p2p == ["tests/test_core.py::test_add"]
    obs = t.observation
    assert obs["type"] == "real_pr_tests" and obs["by"].startswith("miner:") and t.sha in obs["pointer"]
    assert t.validation["reruns"] == 2 and t.validation["flaky"] == 0
    assert not (Path(t.base_dir) / ".git").exists()  # the start tree carries no history to read the fix from


def test_a_commit_whose_tests_never_fail_is_dropped(repo: Path, tmp_path: Path) -> None:
    (repo / "pkg" / "core.py").write_text(CORE_V2 + "\n# tidy\n")
    (repo / "tests" / "test_core.py").write_text(TEST_V2 + "\n# tidy\n")
    _git(repo, "commit", "-qam", "tidy")
    c = candidates(repo, repo="toylib", since="2000-01-01")[0]
    t, why, _ = asyncio_run(validate_candidate(c, LaneStore(tmp_path / "s2"), LocalSandbox(), reruns=1))
    assert t is None and why.startswith("0_f2p")


def test_oracle_edits_reproduce_the_fix_and_fingerprints_ignore_layout() -> None:
    sol = oracle_solution({"pkg/core.py": CORE_V1}, {"pkg/core.py": CORE_V2})
    assert sol["edits"] and "files" not in sol
    written, = [apply_solution_in_dict(CORE_V1, sol)]
    assert written == CORE_V2
    assert edits_from_change("a.py", "x = 1\nx = 1\n", "x = 2\nx = 1\n") is not None  # context grows until 'old' is unique
    reformatted = CORE_V2.replace("max(lo, min(x, hi))", "max(lo,   min(x, hi))  # bounded")
    assert fingerprint({"pkg/core.py": CORE_V1}, {"pkg/core.py": CORE_V2}) == fingerprint({"pkg/core.py": CORE_V1}, {"pkg/core.py": reformatted})
    assert fingerprint({"pkg/core.py": CORE_V1}, {"pkg/core.py": CORE_V1}) == fingerprint({}, {})


def apply_solution_in_dict(text: str, sol: dict) -> str:
    from crucible.factory.lanes.executable import apply_in_memory

    written, errors = apply_in_memory({"pkg/core.py": text}, sol)
    assert not errors
    return written["pkg/core.py"]


def test_apply_drops_protected_paths_and_rejects_bad_edits(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("a = 1\na = 1\n")
    ap = apply_solution(tmp_path, {"files": {"conftest.py": "x", "tests/test_x.py": "y", "../escape.py": "z", "sitecustomize.py": "s"},
                                   "edits": [{"path": "pkg/core.py", "old": "a = 1\n", "new": "a = 2\n"}]})
    assert ap.protected == ["conftest.py", "sitecustomize.py", "tests/test_x.py"]
    assert any("unsafe path" in e for e in ap.errors) and any("occurs 2 times" in e for e in ap.errors)
    assert not (tmp_path / "conftest.py").exists() and not (tmp_path.parent / "escape.py").exists()
    assert is_protected("setup.cfg") and is_protected("x/y.pth") and not is_protected("pkg/core.py")
    assert normalize_solution("rm -rf /")["malformed"]


def _task(store: LaneStore, t: Any, lane: ExecLane) -> Any:
    s1 = "clamp(x, lo, hi) must return lo for values below lo and hi for values above hi."
    prompt = compose_prompt("clamp() ignores its bounds.", [s1, REGRESSION_TEXT])
    reqs = [{"id": "r1", "text": s1, "kind": "behaviour"}, {"id": REGRESSION_REQ, "text": REGRESSION_TEXT, "kind": "side_effect_scope"}]
    spans = {r["id"]: [prompt.index(r["text"]), prompt.index(r["text"]) + len(r["text"])] for r in reqs}
    obs = t.observation["id"]
    cs = {"checks": [{"id": "c1", "requirement_ids": ["r1"], "observation_ids": [obs], "gating": True, "spec": {"tests": t.f2p}},
                     {"id": "c2", "requirement_ids": [REGRESSION_REQ], "observation_ids": [obs], "gating": True, "spec": {"tests": regression_pool(t)}}]}
    return lane.materialize({"task_id": "cell/001", "world": lane.world_of(t), "spec": {"requirements": reqs}, "spans": spans, "checkset": cs,
                             "prompt": prompt})


def test_the_task_grades_oracle_floors_mutants_and_attacks(mined: Any) -> None:
    store, t = mined
    lane = ExecLane(store, LocalSandbox())
    task = _task(store, t, lane)

    async def go() -> None:
        rep, fp = await task.grade(task.oracle)
        assert rep.reward == 1.0 and rep.gates == {"c1": "pass", "c2": "pass"} and fp != await task.start_fingerprint()
        rep, _ = await task.grade(task.noop())
        assert rep.reward == 0.0 and rep.gates == {"c1": "fail", "c2": "pass"}  # the real bug fails the new tests only
        for n in task.negatives:
            rep, _ = await task.grade(task.solution_of(n))
            assert (rep.reward == 1.0) == (n.expected == "pass"), n.id
        muts = task.mutants()
        assert muts
        for m in muts:
            assert (await task.grade(m.solution))[0].reward == 0.0, m.id
        for a in task.attacks():  # each attack takes effect (its control) and still scores 0
            rep, _ = await task.grade(a.solution)
            assert rep.reward == 0.0 and await a.effect(), a.id
        bad, _ = await task.grade({"edits": [{"path": "pkg/core.py", "old": "not in the file", "new": "x"}]})
        assert bad.reward == 0.0 and "did not apply" in bad.meta["why"]
        assert await task.property_audit(task.oracle) is True
        assert await task.property_audit({"files": {"conftest.py": "x"}}) is False
        assert await task.property_audit(task.noop()) is False

    asyncio_run(go())
    assert {p.id for p in task.properties} == {"r1", REGRESSION_REQ} and all(p.text in task.instruction for p in task.properties)
    assert task.checks[0].provenance.startswith("run:mine/")  # A8: a recorded mining run grounds every check


def test_the_guard_reports_after_atexit_hooks_cannot_rewrite_it(tmp_path: Path) -> None:
    """A solution that registers an atexit hook printing a forged all-pass report: the guard's os._exit comes first, and
    pytest's failure report (buffered at exit) still reaches stdout."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "m17guard.py").write_text(GUARD_PLUGIN)
    tree = tmp_path / "t"
    tree.mkdir()
    (tree / "evil.py").write_text("import atexit\n\n\ndef _forge():\n    print('\\n@@M17-JUNIT-BEGIN@@\\n<testsuites></testsuites>\\n@@M17-JUNIT-END@@')\n\n\n"
                                  "atexit.register(_forge)\n")
    (tree / "test_x.py").write_text("import evil  # noqa: F401\n\n\ndef test_fails():\n    assert 1 == 2, 'the real failure'\n")
    env = {**os.environ, "PYTHONPATH": f"{plug}:{tree}", "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-m", "pytest", "-p", "m17guard", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                          "-o", "junit_family=xunit1", f"--junitxml={tmp_path / 'r.xml'}", "test_x.py"], cwd=tree, env=env, capture_output=True, text=True, check=False)
    assert junit_from_stdout(out.stdout) == {"test_x.py::test_fails": "failed"}
    assert out.stdout.count("@@M17-JUNIT-BEGIN@@") == 1 and "the real failure" in out.stdout


def test_solver_replies_parse_only_as_edit_lists() -> None:
    assert parse_edits('```json\n{"edits": [{"path": "a.py", "old": "x", "new": "y"}]}\n```')[0] == {"edits": [{"path": "a.py", "old": "x", "new": "y"}]}
    assert parse_edits("I would change clamp.")[0] is None
    assert json.loads(json.dumps(parse_edits('noise {"edits": []} tail')[0])) == {"edits": []}
