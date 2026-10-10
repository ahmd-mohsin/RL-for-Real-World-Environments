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
    prewrite_zero(tmp_path / "o")
    assert (tmp_path / "o" / "reward.txt").read_text().strip() == "0"
    p = Plan("t", [Gate("a"), Gate("b")], [Check("a", "a", lambda c: True), Check("b", "b", lambda c: False)])
    rj = write_outputs(p.run(ctx(tmp_path)), tmp_path / "o")
    flat = json.loads((tmp_path / "o" / "reward.json").read_text())
    assert flat == {"reward": 0.0, "outcome": 0.0, "gate/a": 1.0, "gate/b": 0.0, "infra/masked": 0.0}
    assert all(isinstance(v, float) for v in flat.values())  # Harbor rejects any non-numeric value
    full = RewardJson.model_validate({k: v for k, v in rj.items() if k in RewardJson.model_fields})
    assert full.gates == {"a": 1, "b": 0} and full.flat()["gate/b"] == 0.0
    crashed = write_outputs(None, tmp_path / "x", error="boom")
    assert crashed["infra"]["status"] == "masked" and (tmp_path / "x" / "reward.txt").read_text().strip() == "0"
    assert flat_reward(crashed)["infra/masked"] == 1.0


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
