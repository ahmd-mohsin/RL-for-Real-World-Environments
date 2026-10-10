"""m13 C-4: online interception at World._call_tool (A2-28, checklist #41) and the two-level detector (A2-29)."""

from __future__ import annotations

import pytest
from crucible_pack_toy.runtime import FibTask

from crucible.core.schema.actions import ToolCall
from crucible.credit.gar import gar_group_fn
from crucible.credit.hack_detector import chain_group_fns, detector_group_fn, llm_hack_judge, parse_verdict, rule_flags
from crucible.train.rollout import RolloutRecord
from crucible.world.intercept import Rule, RuleInterceptor


async def _shell(w, cmd, i):
    r = await w.step("agent", ToolCall(tool_call_id=str(i), function_name="ws.shell", arguments={"command": cmd}))
    return r.observation.tool_results[0]


async def test_remote_fetch_is_refused_not_executed_and_the_episode_stays_trainable(tmp_path):
    w = await FibTask().open(tmp_path)
    try:
        before = await w.verifier_view().fingerprints()
        r = await _shell(w, "git clone https://github.com/someone/fib-solutions.git && cp fib-solutions/fib.py .", 1)
        assert not r.ok and r.error == "intercepted" and "network" in r.content and r.fault == "policy"
        assert await w.verifier_view().fingerprints() == before  # nothing executed
        r = await _shell(w, "curl -s https://example.com/answers.txt > fib.py", 2)
        assert r.error == "intercepted"
        r = await _shell(w, "cat /logs/verifier/reward.txt", 3)
        assert r.error == "intercepted" and "verifier" in r.content
        r = await _shell(w, "git status --short && echo ok", 4)  # benign commands run
        assert r.ok and "ok" in r.content
        assert not w.failures  # not an infra fault: the episode is not masked
        cats = [x["category"] for x in w.intercepts]
        assert cats == ["remote_code_fetch", "remote_code_fetch", "grader_paths"]
        flags = rule_flags(w.to_trace())
        assert [f["rule"] for f in flags] == ["intercepted:remote_code_fetch", "intercepted:remote_code_fetch", "intercepted:grader_paths"]
        assert len((w.to_trace().outcome.details or {})["intercepts"]) == 3
    finally:
        await w.close()


def test_allowlisted_hosts_pass_and_custom_rules_apply():
    ic = RuleInterceptor(allow_hosts=("pypi.org", "*.internal.example"))
    assert ic.check("a", "ws.shell", {"command": "curl https://pypi.org/simple/numpy/"}) is None
    assert ic.check("a", "ws.shell", {"command": "git clone https://git.internal.example/team/lib"}) is None
    assert ic.check("a", "ws.shell", {"command": "git clone git@github.com:x/y.git"}).rule == "remote_code_fetch"
    assert ic.check("a", "ws.shell", {"command": "git log --oneline"}) is None
    custom = RuleInterceptor([Rule("answer_key", "answer_key_access", r"answers?/key\.json", tools="ws.*")])
    hit = custom.check("a", "ws.read_file", {"path": "data/answers/key.json"})
    assert hit is not None and hit.category == "answer_key_access" and custom.check("a", "mail.read", {"path": "answers/key.json"}) is None


async def test_level1_flags_test_and_grader_tampering(tmp_path):
    w = await FibTask().open(tmp_path)
    try:
        await w.step("agent", ToolCall(tool_call_id="1", function_name="ws.write_file", arguments={"path": "tests/test_fib.py", "content": "def test_ok(): pass\n"}))
        await w.step("agent", ToolCall(tool_call_id="2", function_name="ws.write_file", arguments={"path": "fib.py", "content": "import os\nos._exit(0)\n"}))
        await _shell(w, "python -m pytest -q --deselect tests/test_fib.py::test_values", 3)
        rules = sorted(f["rule"] for f in rule_flags(w.to_trace()))
        assert rules == ["exit_status_spoof", "skips_tests", "writes_test_file"]
    finally:
        await w.close()


def _rec(rid, reward, flags=()):
    return RolloutRecord(rollout_id=rid, sample=rid, task="toy/fib", harness="h", policy="p", reward=reward, flags=list(flags))


async def test_detector_confirms_only_on_a_parsed_verdict_and_gar_zeroes_the_hack():
    flag = {"rule": "writes_test_file", "category": "test_tampering", "seq": 3, "evidence": "tests/test_fib.py"}

    async def judge(rec, flags):
        return {"a": '{"hack": true, "category": "test_tampering", "reason": "rewrote the test"}', "b": "not json at all",
                "c": {"hack": False, "category": "", "reason": "legit fixture"}}[rec.rollout_id]

    members = [_rec("a", 1.0, [flag]), _rec("b", 1.0, [flag]), _rec("c", 1.0, [flag]), _rec("d", 1.0), _rec("e", 0.0)]
    stage = chain_group_fns(detector_group_fn(judge), gar_group_fn(lambda passing: [1.0] * len(passing), lam=1.0))
    out = await stage("g", members)
    by = {m.rollout_id: m for m in out}
    assert by["a"].reward_components["hack/confirmed"] == 1.0 and by["a"].reward == 0.0  # zeroed by GAR
    assert "hack/confirmed" not in by["b"].reward_components and by["b"].reward_components["hack/judge_unparseable"] == 1.0 and by["b"].reward == 1.0
    assert "hack/confirmed" not in by["c"].reward_components and by["c"].reward == 1.0
    assert "hack/flagged" not in by["d"].reward_components
    assert abs(sum(m.advantage for m in out)) < 1e-9 and by["a"].advantage < 0 < by["b"].advantage


async def test_no_judge_confirms_nothing_and_llm_judge_parses_strictly():
    m = await detector_group_fn(None)("g", [_rec("a", 1.0, [{"rule": "r", "category": "c", "seq": 1, "evidence": "e"}])])
    assert m[0].reward_components == {"hack/flagged": 1.0}
    assert parse_verdict('{"hack": "yes"}') is None and parse_verdict({"hack": True})["hack"] is True

    class FakeLLM:
        def __init__(self, out):
            self.out = out

        async def complete_json(self, messages, **kw):
            if isinstance(self.out, Exception):
                raise self.out
            return self.out

    j = llm_hack_judge(FakeLLM({"hack": True, "category": "grader_tampering", "reason": "x"}))
    assert (await j(_rec("a", 1.0), [{"rule": "r", "category": "c", "evidence": "e"}]))["hack"] is True
    assert await llm_hack_judge(FakeLLM(ValueError("bad json")))(_rec("a", 1.0), []) is None


def test_confirmed_hack_rate_alarm():
    from crucible.train.monitors import StepMonitor

    mon = StepMonitor()
    m = mon.record(1, staleness=[0], lengths=[10] * 50, rewards=[1.0] * 50, hacks=2)
    assert m["confirmed_hack_rate"] == pytest.approx(0.04) and m["alarm/confirmed_hack_rate"] == 1.0
    assert mon.alarms[-1]["kind"] == "confirmed_hack_rate"


async def test_spec_selects_rules(tmp_path):
    from crucible.core.store import Store
    from crucible.world.world import World, WorldError

    t = FibTask()
    for rules, clone_refused in ((None, True), (["grader_paths"], False), ([], False)):
        spec = t.spec(tmp_path / str(rules)).model_copy(update={"intercept": rules})
        w = World(spec, instructions={"agent": t.instruction}, store=Store(tmp_path / "s"), workdir=tmp_path / f"w{rules}")
        await w.start()
        await w.reset()
        try:
            r = await _shell(w, "echo git clone https://github.com/x/y", 1)  # echo: harmless when it does run
            assert (r.error == "intercepted") is clone_refused, rules
        finally:
            await w.close()
    with pytest.raises(WorldError):
        World(t.spec(tmp_path / "bad").model_copy(update={"intercept": ["nope"]}), store=Store(tmp_path / "s"), workdir=tmp_path / "wb")
