"""``synth.patch`` and ``synth.falsify.patch``: the repository substrate (part2-paper-core §11, M5) on a fixture repo."""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path

import pytest

from crucible.synth.contract import ExecCheck
from crucible.synth.falsify.base import falsify
from crucible.synth.falsify.patch import PatchMUT, PatchNULL, PatchREP, PatchTAINT, certify_patch, findings_summary, shared_tests
from crucible.synth.falsify.red import RED
from crucible.synth.patch import (
    ExecResult,
    Milestone,
    PatchEnv,
    PatchError,
    ThreadedRunner,
    apply_hunks,
    default_roles,
    migrate,
    parse_patch,
    revert_hunk,
)
from crucible.synth.program import Program
from tests.unit.synth_patch_fixtures import BUGGY, FIXED, TEST, fixture_runner, make_repo, spec_for


def run(c):
    return asyncio.run(c)


def env_for(tmp_path: Path, **kw) -> PatchEnv:
    roles = kw.pop("roles", None)
    base = make_repo(tmp_path, **{k: kw.pop(k) for k in ("changelog", "visible") if k in kw})
    return PatchEnv(spec_for(base, **kw), roles=roles)


def confirmed(res) -> set[tuple[str, str]]:
    return {(f.op, f.defect) for f in res.confirmed}


def test_roles_default():
    assert default_roles("calc.py") == "source"
    assert default_roles("tests/test_calc.py") == "test"
    assert default_roles("pkg/x_test.go") == "test"
    assert default_roles("tests/data/cases/a.py") == "golden"
    assert default_roles("testdata/x.golden") == "golden"
    assert default_roles("vendor/a/b.go") == "vendored"
    assert default_roles("conftest.py") == "plumbing"
    assert default_roles(".git/config") == "vcs"
    assert default_roles(".github/workflows/ci.yml") == "config"
    assert default_roles("go.mod") == "config"
    assert default_roles("CHANGELOG.md") == "doc"


def test_oracle_passes_and_start_fails(tmp_path):
    env = env_for(tmp_path)
    runner = fixture_runner(env)
    task, ctx = env.task(), env.context(runner)
    r = run(ctx.run(task, task.oracle))
    assert ctx.grade(r, task.contract).passed
    r0 = run(ctx.run(task, Program()))
    v0 = ctx.grade(r0, task.contract)
    assert not v0.passed and set(v0.reasons) == {"MISSING_REQUIRED:c_m01", "MISSING_REQUIRED:c_m02"}
    assert env.overlay_of(r.end) == {"calc.py": FIXED}


def test_frame_forbids_tests_and_custody(tmp_path):
    env = env_for(tmp_path)
    runner = fixture_runner(env)
    task, ctx = env.task(), env.context(runner)
    prog = env.program({"calc.py": FIXED, TEST: "def test_x():\n    pass\n"})
    v = ctx.grade(run(ctx.run(task, prog)), task.contract)
    assert not v.passed and v.reasons == [f"FORBIDDEN_EFFECT:repo.files[{TEST}].sha256", f"FORBIDDEN_EFFECT:repo.files[{TEST}].text"]
    r = run(ctx.run(task, task.oracle))
    env.blobs.clear()  # an end state whose content the World never holds: the exec fails closed
    big = r.end.model_copy(deep=True)
    big.apps["repo"]["files"]["calc.py"].pop("text")
    assert ctx.grade(r.model_copy(update={"end": big}), task.contract).reasons == ["GRADER_ERROR:exec"]


def test_apply_patch_round_trip(tmp_path):
    env = env_for(tmp_path)
    patch = "".join(difflib.unified_diff(BUGGY.splitlines(keepends=True), FIXED.splitlines(keepends=True), "a/calc.py", "b/calc.py"))
    w = env.world()
    run(w.reset(0))
    obs = run(w.call("repo.apply_patch", {"patch": "diff --git a/calc.py b/calc.py\n" + patch}))
    assert obs.ok and w.overlay["calc.py"] == FIXED
    bad = run(w.call("repo.apply_patch", {"patch": patch}))  # applied twice: the context no longer matches
    assert bad.error and "does not apply" in bad.error
    new = run(w.call("repo.apply_patch", {"patch": "--- /dev/null\n+++ b/n.txt\n@@ -0,0 +1,2 @@\n+x\n+y\n\\ No newline at end of file\n"}))
    assert new.ok and w.overlay["n.txt"] == "x\ny"
    assert run(w.call("repo.read_file", {"path": "../etc/passwd"})).error
    with pytest.raises(PatchError):
        parse_patch("nothing here")
    assert apply_hunks("a\nb\n", parse_patch("--- a/f\n+++ b/f\n@@ -2 +2 @@\n-b\n+c\n")[0].hunks) == "a\nc\n"


def test_revert_hunk_keeps_other_regions():
    one = revert_hunk(BUGGY, FIXED, 0)
    assert "return a - b" in one and "return a * b" in one


def test_certificate_clean_on_correct_task(tmp_path):
    env = env_for(tmp_path)
    cert, results = run(certify_patch(env, fixture_runner(env)))
    s = findings_summary(results)
    assert s["confirmed"] == [] and s["suspects"] == [], s
    assert cert.status in ("certified", "provisional") and cert.admitted
    by = {r.cls: r for r in results}
    assert not by["TWIN"].applicable and not by["PERT"].applicable and by["MUT"].applicable
    ops = {f.op for f in by["MUT"].findings}
    assert {"Identity", "DropFile", "DropHunk", "RevertFix", "ExtraSource", "ExtraForbidden", "EditTest", "SharedCheck"} <= ops
    assert not any(f.op == "exec" and "c_regress" in f.site for f in by["RED"].findings)  # a guard is not probed by RED


def test_missing_check_is_d1(tmp_path):
    ms = [Milestone(id="m01", tests=[f"{TEST}::test_add"]), Milestone(id="m02", tests=[f"{TEST}::test_mul"])]
    env = env_for(tmp_path, milestones=ms, checks={"c_m01": [f"{TEST}::test_add"], "c_regress": [f"{TEST}::test_ident"]})
    res = run(falsify(env.task(), PatchMUT(env), env.context(fixture_runner(env))))
    assert ("RevertFix", "D1") in confirmed(res)
    assert any(f.op == "DropHunk" and f.defect == "D1" for f in res.suspects)


def test_lax_frame_is_d2(tmp_path):
    env = env_for(tmp_path, roles=lambda p: "source" if p.startswith("tests/") else default_roles(p))
    res = run(falsify(env.task(), PatchMUT(env), env.context(fixture_runner(env))))
    assert ("EditTest", "D2") in confirmed(res) or ("ExtraForbidden", "D2") in confirmed(res)


def test_shared_test_is_d6(tmp_path):
    both = f"{TEST}::test_add"
    ms = [Milestone(id="m01", tests=[both]), Milestone(id="m02", tests=[both, f"{TEST}::test_mul"], requires=["m01"])]
    env = env_for(tmp_path, milestones=ms)
    res = run(falsify(env.task(), PatchMUT(env), env.context(fixture_runner(env))))
    assert ("SharedCheck", "D6") in confirmed(res)
    assert shared_tests(["a.go::TestX"], ["a.go::TestX/case"]) == ["a.go::TestX"]


def test_null_agents_and_plumbing(tmp_path):
    env = env_for(tmp_path)
    res = run(falsify(env.task(), PatchNULL(env), env.context(fixture_runner(env))))
    assert res.c == 0 and {f.op for f in res.findings} >= {"noop", "claims_only", "delete_tests", "stub_tests", "plumbing_shim", "path_shim"}
    lax = env_for(tmp_path / "lax", roles=lambda p: "source" if p == "conftest.py" else default_roles(p))
    res2 = run(falsify(lax.task(), PatchNULL(lax), lax.context(fixture_runner(lax, honour_conftest=True))))
    assert ("plumbing_shim", "D8") in confirmed(res2)


def test_taint_ledger_leak_and_stale(tmp_path):
    env = env_for(tmp_path, changelog="# Changes\n\n- add returned the difference (#123)\n",
                  visible="from calc import add\n\n\ndef test_add_old():\n    assert add(2, 3) == -1\n")
    env.spec.requirements = [r for r in env.spec.requirements if r.milestone != "m02"]
    runner = fixture_runner(env)
    res = run(falsify(env.task(), PatchTAINT(env, runner), env.context(runner)))
    got = confirmed(res)
    assert ("ledger", "D5") in got and ("world_leak", "D7") in got and ("visible_contradicts", "D6") in got
    stale = next(f for f in res.confirmed if f.op == "visible_contradicts")
    assert stale.outcome.data["stale"] == [f"{TEST}::test_add_old"]
    clean = env_for(tmp_path / "ok")
    r2 = run(falsify(clean.task(), PatchTAINT(clean, fixture_runner(clean)), clean.context(fixture_runner(clean))))
    assert r2.c == 0


def test_rep_flake_and_prior(tmp_path):
    env = env_for(tmp_path)
    runner = fixture_runner(env, flaky=True)
    res = run(falsify(env.task(), PatchREP(env, runner, prior={"S10": {"n": 5, "bad": 0}}), env.context(runner)))
    assert ("fresh_oracle", "D11") in confirmed(res) or ("fresh_start", "D11") in confirmed(res)
    assert sum(1 for f in res.findings if f.op == "stored_S10") == 5


def test_red_skips_guards(tmp_path):
    env = env_for(tmp_path)
    task = env.task()
    assert [x.kind for x in task.contract.exec] == ["require", "require", "guard"]
    res = run(falsify(task, RED(), env.context(fixture_runner(env))))
    assert res.c == 0 and not any("c_regress" in f.site for f in res.findings)
    vac = task.model_copy(update={"contract": task.contract.model_copy(update={"exec": [*task.contract.exec[:2], ExecCheck(id="c_regress", name=env.exec_name,
                                                                                                                         args={"check": "c_regress"})]})})
    assert ("exec", "D4") in confirmed(run(falsify(vac, RED(), env.context(fixture_runner(env)))))


def test_migrate_equal_and_detects_drift(tmp_path):
    env = env_for(tmp_path)
    record = {"gates": {"S8": {"evidence": {"rows": {"oracle_patch": {"kind": "oracle", "reward": 1.0}, "no_op": {"kind": "floor", "reward": 0.0},
                                                     "revert_m01": {"kind": "mutant", "reward": 0.25}, "revert_m02": {"kind": "mutant", "reward": 0.25}}}},
                        "S3": {"evidence": {"grading": {"oracle_milestones": {"m01": True, "m02": True},
                                                         "start_any_milestone_test": {"m01": False, "m02": False}}}}}}
    rep = run(migrate(env, fixture_runner(env), record))
    assert rep.equal, rep.differences()
    record["gates"]["S8"]["evidence"]["rows"]["revert_m01"]["reward"] = 0.375
    rep2 = run(migrate(env, fixture_runner(env), record))
    assert [(d.case, d.field) for d in rep2.differences()] == [("revert_m01", "reward")]


def test_threaded_runner_bounds_and_caches():
    live = {"n": 0, "max": 0}

    async def grade(files, mode):
        live["n"] += 1
        live["max"] = max(live["max"], live["n"])
        await asyncio.sleep(0.05)
        live["n"] -= 1
        return ExecResult(checks={"c": bool(files)})

    r = ThreadedRunner(grade, parallel=2)
    try:
        r.prefetch([{f"f{i}": "x"} for i in range(6)])
        assert all(r.result({f"f{i}": "x"}).checks["c"] for i in range(6))
        assert live["max"] <= 2 and len(r.log) == 6
        r.result({"f0": "x"})
        assert len(r.log) == 6  # cached
        r.result({"f0": "x"}, fresh=True)
        assert len(r.log) == 7
    finally:
        r.close()
