"""M7 verifier kit: plan semantics, static evasions, comparators, perf, results-channel verdicts, outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.core.schema.reward import RewardJson
from crucible.verify.runtime.patterns.compare import TwoTier, bitwise_tree, json_close
from crucible.verify.runtime.patterns.hidden_tests import verdict_from_records
from crucible.verify.runtime.patterns.perf import CandidateError, PerfBars
from crucible.verify.runtime.patterns.static import Rule, RuleSet, scan, strip_c, strip_py
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict, flat_reward, prewrite_zero, write_outputs
from crucible.verify.runtime.trusted import InfraError, Trusted


def ctx(tmp: Path) -> Context:
    return Context(Trusted(tmp / "w"))


def test_plan_semantics(tmp_path):
    def boom(_):
        raise ValueError("malformed candidate output")

    def infra(_):
        raise InfraError("reference build failed")

    p = Plan("t", [Gate("a"), Gate("b"), Gate("c"), Gate("o", "outcome")],
             [Check("a1", "a", lambda c: True, tier="fast"), Check("b1", "b", boom), Check("c1", "c", lambda c: True, requires=("b",)),
              Check("o1", "o", lambda c: Verdict(True), tier="fast")])
    r = p.run(ctx(tmp_path))
    assert r.gates == {"a": "pass", "b": "fail", "c": "skip", "o": "pass"}
    assert r.reward == 0.0 and r.infra["status"] == "ok"  # a crash judging candidate output is a fail, never masked
    fast = p.run(ctx(tmp_path), tier="fast")
    assert fast.reward == 1.0 and set(fast.gates) == {"a", "o"}  # gates without fast checks are not measured in the fast tier
    q = Plan("t", [Gate("a")], [Check("a1", "a", infra)])
    r2 = q.run(ctx(tmp_path))
    assert r2.reward == 0.0 and r2.infra["status"] == "masked" and r2.gates["a"] == "infra"

    def bad_setup(_):
        raise RuntimeError("no GPU")

    r3 = Plan("t", [Gate("a")], [Check("a1", "a", lambda c: True)], setup=bad_setup).run(ctx(tmp_path))
    assert r3.reward == 0.0 and r3.infra["failure_kind"] == "grader_exception"
    with pytest.raises(ValueError):
        Plan("t", [Gate("a"), Gate("b")], [Check("a1", "a", lambda c: True)])  # a gate with no checks


async def test_async_checks_and_outcome_gates(tmp_path):
    async def ok(_):
        return Verdict(True)

    p = Plan("t", [Gate("g"), Gate("o1", "outcome"), Gate("o2", "outcome")],
             [Check("g", "g", ok), Check("o1", "o1", ok), Check("o2", "o2", lambda c: False)])
    r = await p.arun(ctx(tmp_path))
    assert r.outcome == 0.0 and r.reward == 0.0 and r.failing_gates() == ["o2"]


def test_outputs_flat_and_full(tmp_path):
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "reward.txt").write_text("1\n")  # a stale / agent-written reward
    prewrite_zero(tmp_path / "o")
    assert not (tmp_path / "o" / "reward.txt").exists() and not (tmp_path / "o" / "reward.json").exists()  # removed, no placeholder 0
    p = Plan("t", [Gate("a"), Gate("b")], [Check("a", "a", lambda c: True), Check("b", "b", lambda c: False)])
    rj = write_outputs(p.run(ctx(tmp_path)), tmp_path / "o")
    flat = json.loads((tmp_path / "o" / "reward.json").read_text())
    assert flat == {"reward": 0.0, "outcome": 0.0, "gate/a": 1.0, "gate/b": 0.0}
    assert all(isinstance(v, float) for v in flat.values())  # Harbor rejects any non-numeric value
    assert (tmp_path / "o" / "reward.txt").read_text().strip() == "0.0"
    full = RewardJson.model_validate({k: v for k, v in rj.items() if k in RewardJson.model_fields})
    assert full.gates == {"a": 1, "b": 0} and full.flat()["gate/b"] == 0.0
    crashed = write_outputs(None, tmp_path / "x", error="boom")
    assert crashed["infra"]["status"] == "masked" and flat_reward(crashed) is None
    x = tmp_path / "x"
    assert not (x / "reward.json").exists() and not (x / "reward.txt").exists()  # masked: no reward file, so no runtime trains a 0
    assert json.loads((x / "mask.json").read_text())["mask_sample"] is True
    assert json.loads((x / "crucible_reward.json").read_text())["reward"] is None


def test_reward_txt_is_exact(tmp_path):
    from crucible.verify.runtime.plan import write_reward_files

    write_reward_files(tmp_path, {"reward": 0.1234567891234, "gates": {}, "outcome": 0.0, "infra": {"status": "ok"}})
    assert float((tmp_path / "reward.txt").read_text()) == 0.1234567891234  # :g would round to 6 digits


def test_static_evasions(tmp_path):
    root = tmp_path / "src"
    (root / "gpu").mkdir(parents=True)
    (root / "skipped").mkdir()
    (root / "gpu" / "a.cu").write_text("// texture<float> t; in a comment is fine\n#define TEX texture<float, 2>\nTEX tex_ref;\n")
    (root / "gpu" / "b.cu").write_text("#define CAT(a, b) a##b\nCAT(cudaBind, Texture)(x);\n")
    (root / "skipped" / "c.cu").write_text("texture<int> hidden;\n")
    (root / "gpu" / "d.py").write_text("import importlib\nm = importlib.import_module('cty' 'pes')\n")
    (root / "gpu" / "e.bin").write_bytes(b"\x7fELF\x02\x01")
    rules = RuleSet(forbid=[Rule("texref", r"\btexture\s*<[^;{}()]*>\s*[A-Za-z_]\w*\s*[;,=]"), Rule("bind", r"\bcudaBindTexture\w*\s*\(")],
                    banned_modules={"ctypes"}, forbid_native=True, exclude=(".git",), runner_tamper=False)
    r = scan(root, rules)
    kinds = {v["rule"] for v in r["violations"]}
    assert "texref" in kinds  # skipped/ is scanned: what is built is what is scanned
    assert "texref:alias" in kinds and "bind:token_paste" in kinds and "dynamic_import" in kinds and "native_binary" in kinds
    pruned = scan(root, RuleSet(forbid=rules.forbid, exclude=("skipped",), runner_tamper=False))
    assert "skipped" in pruned["pruned"] and not (root / "skipped").exists()  # excluded paths are deleted, never built
    assert "texture" not in strip_c("int a; // texture<float> t;\n/* texture<x> y; */")
    assert strip_py("x = '#not a comment'  # a comment") == "x = '#not a comment'  "
    (root / "gpu" / "f.py").write_text("import pytest\nimport os\nos._exit(0)\n")
    tamper = scan(root, RuleSet(exclude=(), runner_tamper=True))
    assert sum(v["rule"] == "runner_tamper" for v in tamper["violations"]) >= 2


def test_comparators(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "x.bin").write_bytes(b"123")
    assert bitwise_tree(a, b) == []
    (b / "x.bin").write_bytes(b"124")
    (a / "extra").write_text("")
    assert {f["error"] for f in bitwise_tree(a, b)} == {"differs", "unexpected"}
    assert json_close({"v": 1.0001, "n": [1, 2]}, {"v": 1.0, "n": [1, 2]}, atol=1e-3) == []
    assert json_close({"v": float("nan")}, {"v": 1.0}, atol=1.0)[0]["error"] == "not a finite number"
    tt = TwoTier(strict=lambda c, r: [] if abs(c - r) < 0.01 else ["strict"], relaxed=lambda c, r: [] if abs(c - r) < 0.1 else ["relaxed"],
                 bifurcated=lambda c, r: r > 10)
    assert tt(5.0, 5.001) == ([], "strict") and tt(5.0, 5.05) == (["strict"], "strict") and tt(20.0, 20.05) == ([], "relaxed")


def test_perf_bars():
    seq = iter([{"original": {"m": 10.0}, "candidate": {"m": 6.0}}] * 3)
    v = PerfBars({"m": 0.65}, lambda k: next(seq), rounds=3, remeasure=0)()
    assert v.passed and v.detail["metrics"]["m"]["ratio"] == 0.6
    noisy = iter([{"original": {"m": 10.0}, "candidate": {"m": 9.0}}, {"original": {"m": 10.0}, "candidate": {"m": 90.0}}] * 4)
    v2 = PerfBars({"m": 0.65}, lambda k: next(noisy), rounds=2, remeasure=1, spread_max=0.25)()
    assert v2.infra and "contaminated" in v2.infra  # a noisy failing measurement is re-graded, never scored

    def broken(k):
        raise CandidateError("computed a different result")

    assert not PerfBars({"m": 1.0}, broken)().passed
    walls = iter([{"original": {"m": 10.0}, "candidate": {"m": 5.0}, "wall": {"original": 1.0, "candidate": 3.0}}] * 3)
    v3 = PerfBars({"m": 0.65}, lambda k: next(walls), wall_ratio_max=1.0, remeasure=0)()
    assert not v3.passed and v3.detail["wall"]["ratio"] == 3.0  # an in-process timer claims 2x faster; the grader's clock says 3x slower


def test_results_channel_verdicts():
    ok = [{"event": "collected", "n": 2}, *({"event": "test", "nodeid": f"t{i}", "when": "call", "outcome": "passed"} for i in range(2)),
          {"event": "finish", "exitstatus": 0, "collected": 2}]
    assert verdict_from_records(ok, expected=2, min_count=1, allowed_skips=0, rc=0, log="").passed
    early = ok[:-1]
    assert "did not finish" in verdict_from_records(early, expected=2, min_count=1, allowed_skips=0, rc=0, log="").detail["reasons"][0]
    assert not verdict_from_records(ok, expected=3, min_count=1, allowed_skips=0, rc=0, log="").passed
    skipped = [{"event": "collected", "n": 1}, {"event": "test", "nodeid": "t", "when": "call", "outcome": "skipped"}, {"event": "finish", "collected": 1}]
    assert not verdict_from_records(skipped, expected=None, min_count=1, allowed_skips=0, rc=0, log="").passed
    assert not verdict_from_records([], expected=None, min_count=1, allowed_skips=0, rc=0, log="4 passed").passed  # stdout is ignored


def test_trusted_unprivileged_mode(tmp_path):
    import os
    import tempfile

    if hasattr(os, "geteuid") and os.geteuid() == 0:  # as root the candidate is demoted: pytest's root temp dir is 0700
        with pytest.raises(InfraError, match="cannot reach the work dir"):
            Trusted(tmp_path / "unreachable").run(["true"], candidate=True)
        tmp_path = Path(tempfile.mkdtemp(prefix="crucible-trusted-", dir="/tmp"))
        os.chmod(tmp_path, 0o755)
    t = Trusted(tmp_path / "w")
    r = t.run(["sh", "-c", "echo $HOME; echo x > $HOME/f"], candidate=True)
    assert r.rc == 0 and not r.trusted and r.wall_s >= 0
    assert not Path(r.stdout.strip()).exists()  # a fresh HOME dies with its invocation
    r2 = t.python(script=None, module="json.tool", args=["--help"], first=[], last=[])
    assert r2.rc == 0
    with pytest.raises(InfraError):
        Trusted(tmp_path / "w2", require_root=True) if not t.is_root else (_ for _ in ()).throw(InfraError("root"))


async def test_toy_golden():
    from crucible_pack_toy.verify import golden

    rep = await golden()
    assert rep["pass"], {k: v for k, v in rep["negatives"].items() if not v["as_designed"]}


def test_record_scoring_and_abstention(tmp_path):
    from crucible.verify.runtime.patterns.record import AbstainPolicy, RecordSpec, record

    spec = RecordSpec(baseline=28, best_known=30)  # elliptic-curve rank: 28 was easy, 30 is the record
    assert spec.reward(29) == 0.5 and spec.reward(30) == 1.0 and spec.reward(31) > 1.0 and spec.reward(27) == 0.0
    assert RecordSpec(baseline=10.0, best_known=5.0, direction="min").reward(2.5) > 1.0  # a faster kernel than the best known
    assert record(40, spec, valid=False).detail["normalized_reward"] == 0.0  # an unverifiable "record" scores nothing

    def plan(score, abstain=None, gave_up=False):
        return Plan("deep", [Gate("valid"), Gate("rank", "score")],
                    [Check("valid", "valid", lambda c: True), Check("rank", "rank", lambda c: record(score, spec, valid=True))],
                    abstain=abstain, abstained=(lambda c: gave_up) if abstain else None)

    r31 = plan(31).run(ctx(tmp_path))
    assert r31.reward > 1.0 and r31.meta["score"]["rank"]["new_record"]
    assert plan(29).run(ctx(tmp_path)).reward == 0.5
    rj = plan(29).run(ctx(tmp_path)).reward_json()
    assert rj["gates"] == {"valid": 1} and rj["components"]["check/rank"] == 1.0
    assert plan(29, AbstainPolicy("unknown"), gave_up=True).run(ctx(tmp_path)).reward == 0.05
    infeasible = AbstainPolicy("infeasible")
    assert plan(29, infeasible, gave_up=True).run(ctx(tmp_path)).reward == 1.0
    assert plan(31, infeasible, gave_up=False).run(ctx(tmp_path)).reward == 0.0  # claiming a solution to an impossible task
    assert plan(31, AbstainPolicy("feasible"), gave_up=True).run(ctx(tmp_path)).reward == 0.0
    full = write_outputs(plan(29, infeasible, gave_up=True).run(ctx(tmp_path)), tmp_path / "o")
    RewardJson.model_validate({k: v for k, v in full.items() if k in RewardJson.model_fields})  # consistent: skipped checks are not failed gates


def test_calibrate_and_tiers():
    from crucible.verify.calibrate import bifurcations, clopper_pearson_upper, flake_rate, perf_headroom, self_consistency, tolerance_tiers
    from crucible.verify.tiers import agreement, audit_sample, spearman

    f = flake_rate([True] * 80, [True] * 40)
    assert f.ok and f.draws == 120 and abs(f.upper_95 - (1 - 0.05 ** (1 / 120))) < 1e-6
    assert not flake_rate([True] * 39 + [False]).ok and 0.025 < clopper_pearson_upper(1, 40) < 0.13
    nat = [0.001 * (i % 7) for i in range(80)]
    ind = [0.02 + 0.001 * (i % 3) for i in range(40)]
    t = tolerance_tiers(nat, ind, negatives=[0.5, 0.9])
    assert t.ok and t.bimodal and t.strict < t.relaxed < 0.5
    assert not tolerance_tiers(nat, ind, negatives=[0.025]).kills_negatives  # the relaxed tier would pass a negative
    assert len(bifurcations([1, 1.1, 1.2, 9, 9.1])) == 2 and len(bifurcations([1, 2, 3, 4])) == 1
    h = perf_headroom({"kinfu": 0.571, "people": 0.342}, {"kinfu": 0.65, "people": 0.42})
    assert not h["ok"] and h["min"] == round(1 - 0.571 / 0.65, 4)  # task04 v3 KinFu: 12% headroom, below A1's 15%
    assert self_consistency([{"a": 1}] * 3)["ok"] and not self_consistency([{"a": 1}, {"a": 1}, {"a": 2}])["ok"]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and spearman([1, 1], [1, 2]) is None
    assert agreement([0, 1, 1, 0, 1], [0, 1, 1, 0, 1])["ok"]
    rate = sum(audit_sample(f"r{i}") for i in range(4000)) / 4000
    assert 0.03 < rate < 0.07 and audit_sample("r1") == audit_sample("r1")


def test_leak_audit_and_collect(tmp_path):
    import subprocess

    from crucible.verify.collect import collect
    from crucible.verify.leak_audit import audit_git, audit_tree, hidden_strings, oracle_added_lines

    secret = "assert compute_tsdf_volume(frame, guard_band=1024) == expected_reference_digest"
    strings = hidden_strings([secret, "import os", "x = 1"])
    assert strings == [secret]
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "a.py").write_text("print('hi')\n")
    assert audit_tree(ws, canaries=["CANARY-7f3a"], strings=strings).ok
    (ws / "notes.txt").write_text(f"found it: {secret}\nCANARY-7f3a\n")
    (ws / "solution").mkdir()
    (ws / "solution" / "solution.patch").write_text("x")
    kinds = {f["kind"] for f in audit_tree(ws, canaries=["CANARY-7f3a"], strings=strings).findings}
    assert kinds == {"hidden_string", "canary", "hidden_artifact"}
    assert oracle_added_lines("+++ b/x\n+new line\n-old\n") == ["new line"]
    g = lambda *a: subprocess.run(["git", "-C", str(ws), *a], capture_output=True, check=True)
    g("init", "-q")
    g("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "base")
    base = g("rev-parse", "HEAD").stdout.decode().strip()
    assert audit_git(ws, base=base)["ok"]
    g("checkout", "-q", "-b", "fix")
    g("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "the fix")
    g("checkout", "-q", "-")
    assert not audit_git(ws, base=base)["ok"]  # a branch holding the future fix (SWE-smith's leak)
    (ws / "build").mkdir()
    (ws / "build" / "big.o").write_bytes(b"\0" * 10)
    (ws / "src" / "b.py").write_text("print('new')\n")
    c = collect(ws, base=base)
    assert "src/b.py" in c["patch"] and "big.o" not in c["patch"] and "build/" in c["excluded"]


async def test_host_grader_composes(tmp_path):
    from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck
    from crucible.core.schema.state import Query
    from crucible.verify.grader import grade_episode
    from crucible.verify.runtime.plan import Check, Gate, Plan

    ms = [Milestone(id=f"m{i}", check=MilestoneCheck(app="x", query=Query(op="get", args={}), expect=Expectation())) for i in range(4)]
    p = Plan("t", [Gate("policy"), Gate("done", "outcome")], [Check("policy", "policy", lambda c: True), Check("done", "done", lambda c: False)])
    out = await grade_episode(p, ctx(tmp_path), out_dir=tmp_path / "o", milestones=ms, series=[{"m0": 0.0}, {"m0": 1.0, "m1": 1.0}])
    assert out.reward.outcome == 0.0 and 0 < out.reward.reward <= 0.3  # failed but half the milestones: bounded partial credit
    flat = json.loads((tmp_path / "o" / "reward.json").read_text())
    assert flat["gate/policy"] == 1.0 and all(isinstance(v, float) for v in flat.values())


def test_rows_vs_clock():
    from crucible.verify.runtime.patterns.perf import rows_vs_clock

    honest = rows_vs_clock({"a": 4.0, "b": 6.0}, wall_short_s=5.0, wall_long_s=5.0 + 0.011 * 20, extra_reps=20)
    assert honest["pass"]  # rows claim 10 ms/rep, the clock measured 11
    forged = rows_vs_clock({"a": 0.4, "b": 0.6}, wall_short_s=5.0, wall_long_s=5.0 + 0.011 * 20, extra_reps=20)
    assert not forged["pass"]  # rows claim 1 ms/rep, the clock measured 11


def test_static_frozen_skipped_dirs(tmp_path):
    base, cand = tmp_path / "base", tmp_path / "cand"
    for d in (base, cand):
        (d / "third_party" / "lib").mkdir(parents=True)
        (d / "third_party" / "lib" / "a.cpp").write_text("int a() { return 1; }\n")
        (d / "docs").mkdir()
        (d / "docs" / "x.md").write_text("old\n")
    (cand / "docs" / "x.md").write_text("edited docs are fine\n")
    rs = RuleSet(forbid=[Rule("cuda", r"__global__")], skip=("third_party", "docs"), frozen_base=str(base), runner_tamper=False)
    assert scan(cand, rs)["pass"]
    (cand / "third_party" / "lib" / "hidden.cu.cpp").write_text("__global__ void k() {}\n")  # banned code hidden in an unscanned dir
    r = scan(cand, rs)
    assert not r["pass"] and r["violations"][0]["rule"] == "frozen_dir_changed"


def test_task04_record_mode_scoring():
    from crucible_pack_coding.verify.task04 import BEST_KNOWN, record_score

    bars = {"kinfu_frame_ms": 0.65, "people_ms": 0.42}
    oracle = record_score({"kinfu_frame_ms": {"ratio": BEST_KNOWN["kinfu_frame_ms"]}, "people_ms": {"ratio": BEST_KNOWN["people_ms"]}}, bars, BEST_KNOWN)
    assert oracle["score"] == 1.0 and oracle["valid"]
    faster = record_score({"kinfu_frame_ms": {"ratio": 0.40}, "people_ms": {"ratio": 0.30}}, bars, BEST_KNOWN)
    assert faster["score"] > 1.0 and set(faster["new_records"]) == {"kinfu_frame_ms", "people_ms"}  # no ceiling
    slow = record_score({"kinfu_frame_ms": {"ratio": 0.70}, "people_ms": {"ratio": 0.30}}, bars, BEST_KNOWN)
    assert slow["score"] == 0.0 and slow["below_bar"] == ["kinfu_frame_ms"]  # every bar still holds by default
    assert record_score({"kinfu_frame_ms": {"ratio": 0.70}, "people_ms": {"ratio": 0.30}}, bars, BEST_KNOWN, floor="none")["score"] > 0


def test_seed_kit_plans_build(tmp_path):
    """Every seed task's kit plan constructs (gates, checks, prerequisites) without its verifier image."""
    import sys
    import types

    mt = Path(__file__).resolve().parents[3] / "migration_tasks"
    if not mt.exists():
        import pytest

        pytest.skip("needs the seed tasks (../migration_tasks)")
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    from crucible_pack_coding.verify import task01, task02, task03, task04

    def ctx(**p):
        return Context(Trusted(tmp_path / f"w{len(p)}{sorted(p)}".replace("'", "")), params=p)

    t4 = task04.plan(ctx(tests=str(mt / "task04_pcl_gpu_cuda12" / "tests")))
    assert [g.id for g in t4.gates][-1] == "perf" and len(t4.checks) == 11
    rec = task04.plan(ctx(tests=str(mt / "task04_pcl_gpu_cuda12" / "tests"), perf_mode="record"))
    assert next(g for g in rec.gates if g.id == "perf").kind == "score"
    assert [g.id for g in task04.plan(ctx(tests=str(mt / "task04_pcl_gpu_cuda12" / "tests"), scope="ncv")).gates] == [
        "static", "build", "binary", "consumers", "ncv", "fresh", "sanitizer"]
    t3 = task03.plan(ctx(tests=str(mt / "task03_mdlite_pallas" / "tests")))
    assert [g.id for g in t3.gates] == task03.GATES and next(c for c in t3.checks if c.id == "perf").requires[-1] == "fresh"
    t2 = task02.plan(ctx(tests=str(mt / "task02_pcops_triton" / "tests")))
    assert [g.id for g in t2.gates] == task02.GATES
    assert [g.id for g in task01.plan(ctx(version="v1")).gates] == ["static"]


def test_oom_under_contention_is_the_nodes():
    from crucible.verify.runtime.patterns.perf import oom_under_contention

    log = "Error: out of memory\t.../device_memory.cpp:220:create"
    assert oom_under_contention(log, foreign_mib=14000)  # another user's job held the GPU (golden astra_v3b, 2026-10-09)
    assert not oom_under_contention(log, foreign_mib=300)  # an idle GPU: the candidate's own OOM is its failure
    assert not oom_under_contention("segfault", foreign_mib=14000)
