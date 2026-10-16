"""M17 agent Q: the declarative cyclic machine (factory/machine.py): validator, guard language, executor semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from crucible.factory.machine import GuardSyntaxError, MachineError, MachineRun, Settle, StateContext, canonicalize, guard_passes, load_machine, parse_guard, validate_machine


def loop_machine(**run: Any) -> dict[str, Any]:
    """review <-> fix, the shape of Prime Agent's pr-manager (machines/pr-manager/MACHINE.md)."""
    return {"run": {"max_transitions": 20, **run},
            "states": [{"id": "start", "entry": True, "fn": "start", "inputs": [{"name": "goal", "type": "text", "from": "params.goal"}],
                        "outputs": [{"name": "plan", "type": "text"}]},
                       {"id": "review", "role": "reviewer", "max_entries": 3, "different_family_from": ["fix"],
                        "inputs": [{"name": "plan", "type": "text", "from": "start.plan"},
                                   {"name": "report", "type": "json", "from": "fix.report", "optional": True}],
                        "outputs": [{"name": "verdict", "type": "json"}]},
                       {"id": "fix", "role": "fixer", "max_entries": 3, "outputs": [{"name": "report", "type": "json"}],
                        "inputs": [{"name": "verdict", "type": "json", "from": "review.verdict"}]},
                       {"id": "done", "terminal": "done"}],
            "transitions": [{"from": "start", "to": "review"},
                            {"from": "review", "to": "done", "when": "verdict.approved == true"},
                            {"from": "review", "to": "fix", "when": {"output": "verdict", "path": "approved", "op": "eq", "value": False},
                             "carry": "verdict.findings"},
                            {"from": "fix", "to": "review"}]}


def handlers(approve_after: int, seen: list[StateContext]) -> dict[str, Any]:
    async def start(ctx: StateContext) -> dict:
        return {"plan": f"do {ctx.inputs['goal']}"}

    async def review(ctx: StateContext) -> Settle:
        seen.append(ctx)
        ok = ctx.entry > approve_after
        return Settle({"verdict": {"approved": ok, "findings": [] if ok else [f"finding {ctx.entry}"]}})

    async def fix(ctx: StateContext) -> dict:
        seen.append(ctx)
        return {"report": {"fixed": list(ctx.findings)}}

    return {"start": start, "review": review, "fix": fix}


FAM = {"reviewer": "deepseek", "fixer": "qwen"}


async def test_loop_back_carries_findings_and_optional_self_rebinding(tmp_path: Path) -> None:
    seen: list[StateContext] = []
    run = MachineRun(loop_machine(), run_dir=tmp_path / "r", handlers=handlers(2, seen), params={"goal": "x"}, models=FAM)
    res = await run.run()
    assert res.status == "done" and res.entries == {"start": 1, "review": 3, "fix": 2}
    fixes = [c for c in seen if c.state_id == "fix"]
    assert fixes[0].findings == ["finding 1"] and fixes[1].findings == ["finding 2"]
    reviews = [c for c in seen if c.state_id == "review"]
    assert reviews[0].inputs["report"] is None and reviews[1].inputs["report"] == {"fixed": ["finding 1"]}


async def test_exhaustion_is_escalate_not_done(tmp_path: Path) -> None:
    run = MachineRun(loop_machine(), run_dir=tmp_path / "r", handlers=handlers(99, []), params={"goal": "x"}, models=FAM)
    res = await run.run()
    # Prime Agent finalises this exact shape as done (factory.py:3248-3277); here it pauses with a decision request
    assert res.status == "paused" and res.decision_request["kind"] == "exhausted" and res.decision_request["state"] == "review"
    assert res.entries == {"start": 1, "review": 3, "fix": 3} and res.decision_request["findings"] == []
    res = await run.resume({"action": "revise", "actor": "human:x", "extra_entries": 1})
    assert res.status == "paused" and res.entries["fix"] == 4  # the grant covered review and fix; still not approved
    res = await run.resume({"action": "reject", "actor": "human:x", "reason": "no"})
    assert res.status == "rejected"


async def test_failing_state_retries_then_escalates_and_fail_fast_fails(tmp_path: Path) -> None:
    calls = {"n": 0}

    async def boom(ctx: StateContext) -> dict:
        calls["n"] += 1
        raise ValueError("bad output")

    h = {**handlers(0, []), "review": boom}
    m = loop_machine()
    m["states"][1]["retries"] = 2
    res = await MachineRun(m, run_dir=tmp_path / "a", handlers=h, params={"goal": "x"}).run()
    assert res.status == "paused" and res.decision_request["kind"] == "failed" and calls["n"] == 3
    m["states"][1]["failure_policy"] = "fail_fast"
    res = await MachineRun(m, run_dir=tmp_path / "b", handlers=h, params={"goal": "x"}).run()
    assert res.status == "failed"


async def test_outputs_are_typed_and_complete(tmp_path: Path) -> None:
    async def bad(ctx: StateContext) -> dict:
        return {"plan": 3}

    res = await MachineRun(loop_machine(), run_dir=tmp_path / "r", handlers={**handlers(0, []), "start": bad}, params={"goal": "x"}).run()
    assert res.status == "paused" and "must be text" in res.reason

    async def nan(ctx: StateContext) -> dict:
        return {"verdict": {"approved": float("nan")}}

    res = await MachineRun(loop_machine(), run_dir=tmp_path / "n", handlers={**handlers(0, []), "review": nan}, params={"goal": "x"}).run()
    assert res.status == "paused" and "finite JSON" in res.reason


def test_validator_rules() -> None:
    assert validate_machine(loop_machine(), models=FAM) == []
    m = loop_machine()
    del m["states"][1]["max_entries"]
    assert any("must declare max_entries" in e for e in validate_machine(m))
    m = loop_machine()
    m["states"].append({"id": "orphan", "fn": "x", "outputs": []})
    m["transitions"].append({"from": "orphan", "to": "done"})
    assert any("unreachable" in e for e in validate_machine(m))
    m = loop_machine()
    m["transitions"].append({"from": ["review", "fix"], "to": "done"})
    assert any("joins" in e for e in validate_machine(m))
    m = loop_machine()
    m["states"][2]["inputs"].append({"name": "me", "type": "json", "from": "fix.report"})
    assert any("cannot require itself" in e for e in validate_machine(m))
    m = loop_machine()
    m["states"][1]["inputs"][0]["type"] = "json"
    assert any("cannot read output" in e for e in validate_machine(m))
    m = loop_machine()
    m["transitions"][1]["when"] = "verdict.approved =="
    assert any("does not parse" in e for e in validate_machine(m))
    m = loop_machine()
    m["transitions"][1]["when"] = "nope.approved == true"
    assert any("not a declared output" in e for e in validate_machine(m))
    m = loop_machine()
    m["states"][1]["role"] = "fixer"
    assert any("cross-family" in e for e in validate_machine(m, models=FAM))
    assert any("no model assignment" in e for e in validate_machine(loop_machine(), models={"reviewer": "qwen"}))
    m = loop_machine()
    m["states"][2]["isolated_from"] = ["start"]
    assert any("isolated from ['start']" in e for e in validate_machine(m))
    m = loop_machine()
    m["states"][0]["role"] = "x"
    assert any("exactly one of role, fn, terminal" in e for e in validate_machine(m))
    m = loop_machine()
    m["states"][3]["terminal"] = "rejected"
    assert any("no 'done' terminal" in e for e in validate_machine(m))
    m = loop_machine()
    m["transitions"] = [t for t in m["transitions"] if t["from"] != "fix"]
    assert any("fix has no outgoing transition" in e for e in validate_machine(m))
    with pytest.raises(MachineError):
        canonicalize({"states": []})


def test_guard_language_is_parsed_never_evaluated() -> None:
    g = parse_guard("verdict.ok == true and not (verdict.owner == 'spec' or exists verdict.hack) and stats.p >= 0.1")
    out = {"verdict": {"ok": True, "owner": "author"}, "stats": {"p": 0.5}}
    assert guard_passes(g, out)
    assert not guard_passes(g, {**out, "verdict": {"ok": True, "owner": "spec"}})
    assert not guard_passes(g, {**out, "verdict": {"ok": True, "hack": 1}})
    assert guard_passes(parse_guard('tags contains ["a", "b"]'), {"tags": ["b", "a", "c"]})
    assert guard_passes(parse_guard("v contains 'x'"), {"v": "axb"})
    # JSON-strict equality (Prime factory.py:1252): a bool never equals a number; a missing port fails all but exists
    assert not guard_passes(parse_guard("v == 1"), {"v": True})
    assert not guard_passes(parse_guard("v != 1"), {})
    assert not guard_passes(parse_guard("exists v"), {"v": None})
    for bad in ("__import__('os').system('x')", "v == ", "v === 1", "(v == 1", "v == [1, 2", "v ~ 1", "not"):
        with pytest.raises(GuardSyntaxError):
            parse_guard(bad)


def test_yaml_machine(tmp_path: Path) -> None:
    text = """
run: {max_transitions: 5}
states:
  - {id: a, entry: true, fn: a, outputs: [{name: v, type: json}]}
  - {id: ok, terminal: done}
  - {id: drop, terminal: rejected}
transitions:
  - {from: a, to: ok, when: "v.n > 1"}
  - {from: a, to: drop, when: "v.n <= 1"}
"""
    p = tmp_path / "m.yaml"
    p.write_text(text)
    m = load_machine(p)
    assert canonicalize(m)["transitions"][0]["when"] == {"output": "v", "path": "n", "op": "gt", "value": 1}
    assert load_machine(text) == m
    with pytest.raises(MachineError):
        load_machine("- just\n- a list\n")


async def test_no_matching_transition_escalates_and_revise_reruns(tmp_path: Path) -> None:
    m = load_machine("""
states:
  - {id: a, entry: true, fn: a, max_entries: 2, outputs: [{name: v, type: json}]}
  - {id: ok, terminal: done}
transitions:
  - {from: a, to: ok, when: "v.n > 1"}
""")
    n = {"k": 0}

    async def a(ctx: StateContext) -> dict:
        n["k"] += 1
        return {"v": {"n": n["k"]}}

    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"a": a})
    res = await run.run()
    assert res.status == "paused" and res.decision_request["kind"] == "no_transition"
    res = await run.resume({"action": "revise", "actor": "human:x"})
    assert res.status == "done" and n["k"] == 2
