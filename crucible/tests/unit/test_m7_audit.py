"""M7 audit regressions (docs/audit/2026-10-12/M07-verify.md): one test per reproduced defect (V1-V22) and per pick.

Every test here fails on the M6 commit (76ec547). Symbols added by the audit are imported inside the tests, so on the
older tree each test fails on its own rather than the module failing to import.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import py_compile
import subprocess
import sys
import types
from pathlib import Path

import pytest

from crucible.verify.runtime.patterns.compare import json_close
from crucible.verify.runtime.patterns.perf import CandidateError, PerfBars
from crucible.verify.runtime.patterns.static import scan
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict, write_reward_files
from crucible.verify.runtime.trusted import InfraError, Trusted


def ctx(tmp: Path, **kw) -> Context:
    return Context(Trusted(tmp / "w"), **kw)


def tree(root: Path, files: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        p = root / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(v) if isinstance(v, bytes) else p.write_text(v)
    return root


def git(ws: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(ws), "-c", "user.email=a@b", "-c", "user.name=a", *a], capture_output=True, text=True,
                          check=True).stdout.strip()


# ---------------------------------------------------------------- verdict forgery


def _site(base: Path, *, plugin: bool = False, tamper: bool = False) -> Path:
    site = tree(base, {"mdlite/__init__.py": "def answer():\n    return 41  # wrong\n"})
    if plugin:  # V1: a pytest11 entry point whose hook rewrites failed reports
        tree(site, {"mdlite_helpers.py": "def pytest_runtest_logreport(report):\n    if report.outcome == 'failed':\n"
                                         "        report.outcome = 'passed'\n        report.longrepr = None\n",
                    "mdlite-0.1.dist-info/METADATA": "Metadata-Version: 2.1\nName: mdlite\nVersion: 0.1\n",
                    "mdlite-0.1.dist-info/entry_points.txt": "[pytest11]\nmdlite_helpers = mdlite_helpers\n"})
    if tamper:  # a library that disarms unittest assertions when imported
        (site / "mdlite" / "__init__.py").write_text("import unittest\nunittest.TestCase.assertEqual = lambda *a, **k: None\n"
                                                     "def answer():\n    return 41\n")
    return site


def _suite(tmp: Path, site: Path, name: str, test: str):
    from crucible.verify.runtime.patterns.hidden_tests import install_plugin, pytest_suite

    hid = tree(tmp / f"hid_{name}", {"test_hidden.py": test})
    c = ctx(tmp)
    plugin = install_plugin(c.trusted, tmp / f"plugin_{name}")
    return pytest_suite(c, name=name, args=["test_hidden.py"], cwd=hid, pythonpath=[hid], plugin_dir=plugin, candidate_path=(site,),
                        python=sys.executable, expected=2, timeout=120)


PLAIN = "import mdlite\n\ndef test_a():\n    assert mdlite.answer() == 42\n\ndef test_b():\n    assert mdlite.answer() + 1 == 43\n"
UNITTEST = ("import unittest, mdlite\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(mdlite.answer(), 42)\n"
            "    def test_b(self):\n        self.assertEqual(mdlite.answer() + 1, 43)\n")


def test_v1_entry_point_plugin_cannot_forge_hidden_tests(tmp_path):
    honest = _suite(tmp_path, _site(tmp_path / "honest"), "honest", PLAIN)
    assert not honest.passed and honest.detail["failed"]
    forged = _suite(tmp_path, _site(tmp_path / "evil", plugin=True), "evil", PLAIN)
    assert not forged.passed and len(forged.detail["failed"]) == 2  # was: passed, collected 2, failed []


def test_pick_unittest_assertion_tamper_is_reported(tmp_path):
    v = _suite(tmp_path, _site(tmp_path / "t", tamper=True), "tamper", UNITTEST)
    assert not v.passed and any("assertions" in r for r in v.detail["reasons"])


def test_pick_expected_ids_reject_a_different_test_set():
    from crucible.verify.runtime.patterns.hidden_tests import verdict_from_records

    recs = [{"event": "collected", "n": 2, "ids": ["t::x", "t::y"]},
            *({"event": "test", "nodeid": i, "when": "call", "outcome": "passed"} for i in ("t::x", "t::y")),
            {"event": "finish", "exitstatus": 0, "collected": 2}]
    assert verdict_from_records(recs, expected=2, min_count=1, allowed_skips=0, rc=0, log="", expected_ids=["t::x", "t::y"]).passed
    v = verdict_from_records(recs, expected=2, min_count=1, allowed_skips=0, rc=0, log="", expected_ids=["t::x", "t::z"])
    assert not v.passed and "expected set" in v.detail["reasons"][0]


def test_v1_static_flags_pytest11_entry_points(tmp_path):
    from crucible_pack_coding.verify.task03 import RULES as T03

    r = scan(tree(tmp_path / "c", {"mdlite/__init__.py": "pallas_call = 1\n", "mdlite/h.py": "def pytest_runtest_logreport(r):\n    pass\n",
                                   "pyproject.toml": "[project.entry-points.pytest11]\nh = 'mdlite.h'\n"}), T03)
    assert not r["pass"] and any(v["rule"] == "runner_tamper" and "pytest11" in v.get("match", "") for v in r["violations"])


def test_v4_task03_perf_start_up_that_shrinks_with_steps_is_not_a_speedup(tmp_path):
    from crucible_pack_coding.verify import task03 as T

    hid = tree(tmp_path / "hid", {"bench_mdlite.py": "CASES = {'water': ('w.gro', {}, 100)}\n"})
    g = T.Task03.__new__(T.Task03)
    g.hid, g.thresholds = hid, {"max_ratio": {"water": 0.5}}
    g.t = types.SimpleNamespace(kill=lambda: None)
    g.ctx = types.SimpleNamespace(params={"no_load_guard": True}, artifacts={"site": tmp_path})
    end = {"pot_per_atom": -10.0, "temperature": 300.0, "steps": 0}

    def timed(case, k, candidate, tag):  # the candidate is 3x slower per step, but sleeps 6000/k s at start-up
        return (2.0 + k * 0.03 + 6000.0 / k, {**end, "steps": 200 + k}) if candidate else (2.0 + k * 0.01, {**end, "steps": 200 + k})

    g._timed = timed
    v = g.perf(types.SimpleNamespace(params={"no_load_guard": True}))
    assert not v.passed and not v.infra  # was: passed, candidate 0.0 ms/step


def test_v5_non_positive_times_and_records():
    from crucible_pack_coding.verify.task04 import record_score

    from crucible.verify.runtime.patterns.record import RecordSpec, record

    for c in (-5.0, 0.0, float("nan")):
        v = PerfBars({"m": 1.0}, lambda k, c=c: {"original": {"m": 10.0}, "candidate": {"m": c}}, rounds=1, remeasure=0)()
        assert not v.passed and not v.infra
    with pytest.raises(InfraError):  # a zero original is the grader's measurement, never the candidate's fail
        PerfBars({"m": 1.0}, lambda k: {"original": {"m": 0.0}, "candidate": {"m": 5.0}}, rounds=1, remeasure=0)()
    assert record(-1000.0, RecordSpec(0.8, 0.571, "min"), valid=True).detail["normalized_reward"] == 0.0
    s = record_score({"kinfu_frame_ms": {"ratio": -1000}}, {"kinfu_frame_ms": 0.8}, {})
    assert s["score"] == 0.0 and not s["new_records"]  # was: score 9.38, a new record


def test_v12_json_close_non_finite_reference_and_bools():
    nan, inf = float("nan"), float("inf")
    assert json_close(123.0, nan) and json_close(0.0, inf, atol=1e-6) and json_close(0.0, inf, rtol=1e-3)
    assert json_close(nan, nan) == [] and json_close(inf, inf) == [] and json_close(-inf, inf)
    assert json_close(1, True) and json_close(True, True) == []


def test_v13_v14_reference_failures_are_infra(tmp_path):
    from crucible.verify.runtime.patterns.differential import References

    t = Trusted(tmp_path / "w")
    with pytest.raises(InfraError, match="produced nothing"):
        References(tmp_path / "refs", lambda n, a, out: out.mkdir(exist_ok=True), t).get("r", None)

    def boom(n, a, out):
        raise ValueError("reference binary crashed")

    with pytest.raises(InfraError, match="reference run"):
        References(tmp_path / "refs2", boom, t).get("r", None)


def test_v14_missing_grader_cwd_is_infra(tmp_path):
    with pytest.raises(InfraError, match="does not exist"):
        Trusted(tmp_path / "w").run(["/usr/bin/true"], cwd=tmp_path / "missing_build_dir", candidate=True)


def test_v17_fast_tier_cannot_pay_what_full_rejects(tmp_path):
    from crucible.verify.tiers import agreement

    p = Plan("t", [Gate("build"), Gate("tests")], [Check("b", "build", lambda c: False), Check("t", "tests", lambda c: True, tier="fast")])
    assert p.fast_unmeasured_hard == ["build"]
    assert p.run(ctx(tmp_path), tier="fast").meta["fast_unmeasured_hard_gates"] == ["build"]
    a = agreement([0.1, 0.5, 0.9, 1.0], [0.0, 0.5, 0.9, 1.0])
    assert a["fast_pass_full_fail"] == 1 and not a["ok"]  # was ok: spearman 1.0


# ---------------------------------------------------------------- masks a policy can trigger


def test_v2_candidate_side_noise_fails_not_masks():
    noisy = iter([{"original": {"m": 100.0}, "candidate": {"m": 300.0}}, {"original": {"m": 100.0}, "candidate": {"m": 600.0}}] * 4)
    v = PerfBars({"m": 1.0}, lambda k: next(noisy), rounds=2, remeasure=1, spread_max=0.25)()
    assert not v.passed and not v.infra


def test_v3_certain_hard_fail_is_scored_not_masked(tmp_path):
    def infra(_):
        raise InfraError("perf node busy")

    p = Plan("t", [Gate("static"), Gate("perf")], [Check("s", "static", lambda c: False), Check("p", "perf", infra)])
    r = p.run(ctx(tmp_path))
    assert r.reward == 0.0 and r.infra["status"] == "ok" and r.infra["unmeasured_checks"] == ["p"]
    one = Plan("t", [Gate("g")], [Check("a", "g", lambda c: False), Check("b", "g", infra)]).run(ctx(tmp_path))
    assert one.gates["g"] == "fail" and one.infra["status"] == "ok"
    only_infra = Plan("t", [Gate("g"), Gate("h")], [Check("a", "g", lambda c: True), Check("b", "h", infra)]).run(ctx(tmp_path))
    assert only_infra.infra["status"] == "masked"  # nothing failed for certain: still masked


def test_v10_missing_row_and_dropped_table_fail(tmp_path):
    from crucible.core.schema.milestone import Expectation
    from crucible.core.schema.state import QueryResult
    from crucible.verify.runtime.patterns.invariants import Invariant
    from crucible.verify.state import Predicate, invariant_check, predicate_check

    class View:
        def __init__(self, dropped):
            self.dropped = dropped

        async def query(self, app, q):
            return QueryResult(ok=False, error="no such table: refunds", exit_code=1) if self.dropped else QueryResult(ok=True, value=None)

    pred = Predicate.sql("t1/refund", "tickets", "SELECT amount_usd FROM refunds WHERE ticket_id = 1", Expectation(kind="between", lo=129.98, hi=130.0))
    plan = Plan("t", [Gate("pol"), Gate("t1", "outcome")],
                [invariant_check("pol", lambda c: c.params["view"], "tickets", [Invariant("x", "SELECT id FROM refunds WHERE amount_usd > 50")]),
                 predicate_check("t1", lambda c: c.params["view"], [pred])])
    for dropped in (False, True):
        r = asyncio.run(plan.arun(ctx(tmp_path, params={"view": View(dropped)})))
        assert r.infra["status"] == "ok" and r.gates["t1"] == "fail" and r.reward == 0.0
    r = asyncio.run(plan.arun(ctx(tmp_path, params={"view": View(True)})))
    assert r.gates["pol"] == "fail"


def test_v16_candidate_content_outside_checks_is_a_fail(tmp_path):
    import shutil

    from crucible.verify.runtime.patterns.record import AbstainPolicy

    def raising(_):
        raise IsADirectoryError("ABSTAIN is a directory")

    r = Plan("t", [Gate("g")], [Check("a", "g", lambda c: False)], abstain=AbstainPolicy(), abstained=raising).run(ctx(tmp_path))
    assert r.infra["status"] == "ok" and r.reward == 0.0
    cand = tree(tmp_path / "cand", {"mylib.py": "x = 1\n"})
    os.mkfifo(cand / "pipe")
    p = Plan("t", [Gate("tests")], [Check("t", "tests", lambda c: True)],
             setup=lambda c: shutil.copytree(c.candidate, c.work / "cand_site", symlinks=True))
    r2 = p.run(ctx(tmp_path, candidate=cand))
    assert r2.infra["status"] == "ok" and r2.reward == 0.0 and r2.gates["tests"] == "fail"

    def grader_broke(_):
        raise RuntimeError("no GPU")

    r3 = Plan("t", [Gate("g")], [Check("a", "g", lambda c: True)], setup=grader_broke).run(ctx(tmp_path, candidate=cand))
    assert r3.infra["failure_kind"] == "grader_exception"  # the grader's own setup failure still masks


# ---------------------------------------------------------------- isolation, audit, collect


def _fake_docker(tmp: Path, monkeypatch) -> Path:
    b = tmp / "bin"
    b.mkdir()
    log = tmp / "docker.log"
    (b / "docker").write_text(f'#!/bin/sh\necho "$@" >> {log}\ncase "$1" in run) echo "invalid mount config" >&2; exit 125;; esac\nexit 0\n')
    (b / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    return log


def test_v9_v21_docker_grade_stale_files_and_params(tmp_path, monkeypatch):
    from crucible.verify.stage import docker_grade, reward_json

    log = _fake_docker(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "crucible_reward.json").write_text(json.dumps({"reward": 1.0, "gates": {"tests": 1}, "outcome": 1.0, "infra": {"status": "ok"}}))
    g = docker_grade(image="img", bundle_dir=tmp_path, plan="p", out_dir=out, candidate="/cand", params={"name": "true", "tag": "[1]"})
    assert g.reward["reward"] is None and g.reward["infra"]["failure_kind"] == "verifier_not_started"  # was: the old 1.0
    assert reward_json(g).reward is None
    calls = log.read_text()
    assert 'name="true"' in calls and 'tag="[1]"' in calls  # strings stay strings inside the container
    assert any(ln.startswith("rm -f") for ln in calls.splitlines())  # the container is removed whatever happened


def test_v11_leak_audit_large_files_and_odd_refs(tmp_path):
    from crucible.verify.leak_audit import audit_git, audit_tree

    big = tree(tmp_path / "big", {"notes.txt": "x" * (5 << 20) + "\nthe hidden expected answer\n"})
    assert not audit_tree(big, strings=["the hidden expected answer"]).ok
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    sha = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input=b"SECRET ORACLE\n", capture_output=True, check=True).stdout.decode().strip()
    git(repo, "tag", "leak", sha)
    r = audit_git(repo)
    assert not r["ok"] and r["non_commit_refs"] == ["refs/tags/leak"]


def test_v11_audit_image_scans_every_top_level_dir(tmp_path, monkeypatch):
    import io
    import tarfile

    from crucible.verify.leak_audit import audit_image

    tarp = tmp_path / "image.tar"
    with tarfile.open(tarp, "w") as tf:
        for name, data in {"tests/test_hidden.py": b"assert solve() == 'XYZ-hidden'\n", "usr/lib/ref/oracle.py": b"CANARY-7f3a\n",
                           "proc/1/environ": b"CANARY-7f3a\n"}.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    b = tmp_path / "bin"
    b.mkdir()
    (b / "docker").write_text(f'#!/bin/sh\ncase "$1" in create) echo cid;; export) cat {tarp};; esac\nexit 0\n')
    (b / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    rep = audit_image("img", canaries=["CANARY-7f3a"], strings=["XYZ-hidden"])
    assert not rep.ok and {f["file"] for f in rep.findings} == {"tests/test_hidden.py", "usr/lib/ref/oracle.py"}


async def test_v20_collect_keeps_committed_work_and_nested_build_sources(tmp_path):
    from crucible.runtime.collect import tree_changed
    from crucible.verify.collect import collect, is_derived

    ws = tree(tmp_path / "ws", {"a.c": "int a;\n"})
    git(ws, "init", "-q")
    git(ws, "add", "-A")
    git(ws, "commit", "-q", "-m", "base")
    base = git(ws, "rev-parse", "HEAD")
    (ws / "a.c").write_text("int a = 1;\n")
    git(ws, "commit", "-qam", "fix")  # the agent committed its work
    tree(ws, {"tools/build/gen.py": "print(1)\n", "build/out.o": b"\x7fELF", "pkg/__pycache__/m.cpython-312.pyc": b"\0"})
    c = await collect(ws, base=base)  # M10 lab audit R-5 item 4: a coroutine (git runs off the event loop)
    assert "a.c" in c["patch"] and "tools/build/gen.py" in c["patch"] and "build/out.o" not in c["patch"]
    assert not is_derived("tools/build/gen.py") and is_derived("build/out.o") and is_derived("pkg/__pycache__/m.pyc")
    clean = tree(tmp_path / "clean", {"a.py": "x = 1\n"})
    git(clean, "init", "-q")
    git(clean, "add", "-A")
    git(clean, "commit", "-q", "-m", "base")
    tree(clean, {"__pycache__/a.cpython-312.pyc": b"\0"})
    assert not await tree_changed(clean)  # a do-nothing agent that ran Python changed nothing the hook collects


async def test_pick_workspace_harness_records_base_sha(tmp_path):
    from crucible.runtime.harness.workspace import _base_sha

    ws = tree(tmp_path / "ws", {"a": "1\n"})
    git(ws, "init", "-q")
    git(ws, "add", "-A")
    git(ws, "commit", "-q", "-m", "b")
    git(ws, "tag", "base")
    assert await _base_sha(ws) == git(ws, "rev-parse", "HEAD")


def test_pick_submission_custody(tmp_path):
    from crucible.verify.runtime.patterns.compare import load_json, safe_read

    secret = tree(tmp_path / "ref", {"ref.json": "[1, 2]"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "result.json").symlink_to(secret / "ref.json")
    with pytest.raises(CandidateError):
        load_json(out / "result.json")  # a link to the root-only reference is not an output
    os.mkfifo(out / "pipe")
    with pytest.raises(CandidateError):
        safe_read(out / "pipe")
    Trusted(tmp_path / "w").seal(out)
    assert not (out / "result.json").is_symlink() and not (out / "pipe").exists()


# ---------------------------------------------------------------- static checks


def test_v6_python_static_bypasses(tmp_path):
    from crucible_pack_coding.verify.task02 import RULES as T02
    from crucible_pack_coding.verify.task03 import RULES as T03

    def rules_hit(rules, files, name):
        return [v["rule"] for v in scan(tree(tmp_path / name, files), rules)["violations"]]

    assert rules_hit(T03, {"mdlite/x.py": "import os, ctypes\npallas_call = 1\n"}, "a")
    assert rules_hit(T02, {"pkg/k.py": "import os, cupy\n@triton.jit\ndef k(): pass\n"}, "b")
    assert "runner_tamper" in rules_hit(T03, {"mdlite/x.py": "import os, pytest\npallas_call = 1\n"}, "c")
    assert "runner_tamper" in rules_hit(T03, {"mdlite/x.py": "import \\\n    pytest\npallas_call = 1\n"}, "d")
    assert "dynamic_exec" in rules_hit(T03, {"mdlite/x.py": "exec('import cty' + 'pes as m')\npallas_call = 1\n"}, "e")
    assert "dynamic_exec" in rules_hit(T03, {"mdlite/x.py": "m = __builtins__.__dict__['__imp' + 'ort__']('cty' + 'pes')\npallas_call = 1\n"}, "f")
    # control: attribute calls named compile are honest code (the CPU golden caught 28 false positives on task03)
    assert "dynamic_exec" not in rules_hit(T03, {"mdlite/x.py": "import re\nP = re.compile('x')\nf = jax.jit(g).lower(1).compile()\npallas_call = 1\n"}, "ctl")
    src = tree(tmp_path / "src", {"evil.py": "import ctypes\n"})
    py_compile.compile(str(src / "evil.py"), cfile=str(src / "evil.pyc"))
    assert "native_binary" in rules_hit(T03, {"mdlite/__init__.py": "pallas_call = 1\n", "mdlite/_k.pyc": (src / "evil.pyc").read_bytes()}, "g")
    assert "native_binary" in rules_hit(T03, {"mdlite/__init__.py": "pallas_call = 1\n", "mdlite/x.dat": b"!<arch>\nfoo.o/  "}, "h")


def test_v7_task01_native_files_in_skipped_dirs(tmp_path):
    from crucible_pack_coding.verify import task01

    files = {"third_party/doctest.h": "// vendored\n", "src/a.cpp": "int a(){return 0;}\n", "CMakeLists.txt": "project(x CXX)\n"}
    base = tree(tmp_path / "base", files)
    for i, extra in enumerate(({"third_party/libk.a": b"!<arch>\n"}, {"third_party/k.so": b"\x7fELF\x02"}, {"third_party/k.inc": "evil();\n"})):
        r = scan(tree(tmp_path / f"c{i}", {**files, **extra}), task01.rules("v2", frozen_base=str(base)))
        assert not r["pass"], extra


def test_v8_task04_includes_and_line_continuations(tmp_path):
    from crucible_pack_coding.verify.task04 import RULES as T04

    assert not scan(tree(tmp_path / "a", {"gpu/a.cu": '#include "a.inc"\n', "gpu/a.inc": "texture<float, 2> tex;\n"}), T04)["pass"]
    assert not scan(tree(tmp_path / "b", {"gpu/a.cu": "cudaBindTex\\\nture2D(0, tex, p, d, w, h, s);\n"}), T04)["pass"]


# ---------------------------------------------------------------- plan, reward files, answers, calibration


def test_v15_score_gate_averages_its_checks(tmp_path):
    p = Plan("t", [Gate("s", "score")], [Check("a", "s", lambda c: Verdict(True, {"normalized_reward": 1.0})),
                                          Check("b", "s", lambda c: Verdict(True, {"normalized_reward": 1.0}))])
    assert p.run(ctx(tmp_path)).reward == 1.0  # was 2.0


def test_v18_non_finite_reward_is_masked(tmp_path):
    write_reward_files(tmp_path, {"reward": float("nan"), "outcome": 1.0, "gates": {}, "components": {"check/s": 1.0}, "infra": {"status": "ok"}})
    assert not (tmp_path / "reward.json").exists()

    def strict(x):
        raise ValueError(x)

    doc = json.loads((tmp_path / "crucible_reward.json").read_text(), parse_constant=strict)
    assert doc["reward"] is None and doc["infra"]["status"] == "masked"


def test_v19_answer_edge_cases():
    from crucible.verify.answers import CheckerSpec, CheckerUnavailable, ParserSpec, choice_letter, json_value

    assert json_value('{"a": {"b": 1}}') == {"a": {"b": 1}} and json_value("[1, [2, 3]]") == [1, [2, 3]]
    assert ParserSpec("json").parse("[" * 20000) is None
    for bad in (float("nan"), "yes"):
        async def judge(row, bad=bad):
            return bad

        with pytest.raises(CheckerUnavailable):
            asyncio.run(CheckerSpec(kind="judge", judge="j").score("x", "y", judges={"j": judge}))
    letters = set("ABCD")
    assert choice_letter("I think the answer is B", letters) == "B" and choice_letter("A good choice is C", letters) == "C"
    assert choice_letter("B) because", letters) == "B"


def test_v22_tolerance_tiers_need_negatives_and_natural_draws():
    from crucible.verify.calibrate import flake_rate, tolerance_tiers

    assert not tolerance_tiers([1e-6] * 40).ok  # no negatives: nothing shows the tolerance kills anything
    assert not tolerance_tiers([1e-6], [1e-6] * 39, [1.0]).ok  # 1 natural draw
    assert tolerance_tiers([1e-6] * 40, (), [1.0]).ok
    f = flake_rate([True, None, True])
    assert f.ok and f.unmeasured == 1 and f.false_negatives == 0


# ---------------------------------------------------------------- picks: masked grades under other runtimes


def test_pick_rllm_reward_file_is_cleared(tmp_path, monkeypatch):
    from crucible.interop.harbor.role_runner import TEST_SH
    from crucible.verify.runtime import plan as plan_mod

    assert "/tmp/rllm/reward.json" in plan_mod.FOREIGN_REWARD_FILES and "rm -f /tmp/rllm/reward.json" in TEST_SH
    agent_written = tree(tmp_path / "rllm", {"reward.json": '{"reward": 1.0}'}) / "reward.json"
    monkeypatch.setattr(plan_mod, "FOREIGN_REWARD_FILES", (str(agent_written),))
    plan_mod.prewrite_zero(tmp_path / "out")
    assert not agent_written.exists()


def test_pick_masked_mean_metric(tmp_path):
    from crucible.interop.harbor.metrics import masked_mean, write_masked_mean

    rows = [{"reward": 1.0}, None, {"reward": 0.0}, {"reward": float("nan")}]
    m = masked_mean(rows)
    assert m["masked_mean"] == 0.5 and m["masked"] == 2 and m["mask_rate"] == 0.5
    script = write_masked_mean(tmp_path / "masked_mean.py")
    (tmp_path / "rows.jsonl").write_text('{"reward": 1.0}\nnull\n{"reward": 0.0}\n')
    subprocess.run([sys.executable, str(script), "-i", str(tmp_path / "rows.jsonl"), "-o", str(tmp_path / "m.json")], check=True)
    out = json.loads((tmp_path / "m.json").read_text())
    assert out["masked_mean"] == 0.5 and math.isclose(out["mask_rate"], 1 / 3)
