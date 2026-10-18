"""P2 second verifier (``factory.verifier2``): the payload holds only the prompt, the requirements and the repository at A
(V2, proven by ``leaks``); the scripted writer's reply becomes one shadow module per requirement with one repair round;
the runner's verifier-1 verdict equals ``WindowTask.grade``; qualification and the ownership rules; the S5/S8/S9 handlers
run in ``factory.machine``; the session form of the writer validates."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_m17_p2_seeded import SHADOW, alternatives_of, end_states, seeded_task, shadow_suite, toy, with_oracle

from crucible.factory.machine import MachineRun, validate_machine
from crucible.factory.verifier2 import (
    EndStateRec,
    P2Gates,
    ShadowSuite,
    WindowRunner,
    assign_owner,
    gate_handlers,
    gate_machine,
    leaks,
    parse_shadow,
    prompt_names,
    qualify,
    repo_context,
    shadow_path,
    shadow_payload,
    shadow_session_state,
    truth_for,
    unsound_defects,
    write_shadow,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return toy(tmp_path_factory.mktemp("p2v2"))


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen: list = []

    async def complete_json(self, messages, **kw):
        self.seen.append(messages)
        return self.replies.pop(0)


def test_payload_is_prompt_requirements_and_a_only(built) -> None:
    task, rec, _ = built
    assert prompt_names("Add `alpha()` and pkg.beta.beta; make gamma_value( work; CamelCaseThing") [:4] == ["alpha", "pkg", "beta", "gamma_value"]
    p = shadow_payload(task.prompt, task.requirements, task.base_dir)
    assert [r["id"] for r in p["requirements"]] == ["R1", "R2", "R3"] and "How this works" not in p["request"]
    ctx = p["repository"]
    assert "pkg/core.py" in ctx["files"] and "pkg/beta.py" not in ctx["files"]  # A has no beta module
    assert all("alpha" not in body for body in ctx["modules"].values())  # nothing of B
    assert leaks(p, oracle_files=rec["oracle_files"], hidden_tests=rec["tests_b"], base_dir=task.base_dir) == []
    planted = {**p, "hint": "def test_gamma(): ..."}
    assert leaks(planted, oracle_files=rec["oracle_files"], hidden_tests=rec["tests_b"], base_dir=task.base_dir) == ["hidden test name test_gamma"]
    long_line = "    return some_really_long_expression_name_from_the_oracle(alpha, beta) + 1"
    planted2 = {**p, "x": long_line.strip()}
    assert leaks(planted2, oracle_files={"pkg/gamma.py": long_line + "\n"}, hidden_tests={}, base_dir=task.base_dir)[0].startswith("oracle line")


def test_repo_context_outlines_large_modules(tmp_path) -> None:
    (tmp_path / "lib").mkdir()
    big = "\n".join(f"def f{i}(x):\n    '''Doc {i}.'''\n    return x + {i}\n" for i in range(2000))
    (tmp_path / "lib" / "big.py").write_text(big)
    (tmp_path / "lib" / "small.py").write_text("def wanted_name(a):\n    return a\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    ctx = repo_context(tmp_path, "Make wanted_name and f777() better")
    assert ctx["modules"]["lib/small.py"].startswith("def wanted_name") and ctx["modules"]["lib/big.py"].startswith("# outline")
    assert "tests/test_x.py" not in ctx["files"]


def test_write_shadow_parses_repairs_and_refuses_a_leaking_payload(built) -> None:
    task, rec, _ = built
    bad = {"tests": [{"requirement": "R1", "code": SHADOW["R1"]}, {"requirement": "R2", "code": "def test_r2(:\n"}, {"requirement": "R9", "code": "x"}]}
    fix = {"tests": [{"requirement": "R2", "code": SHADOW["R2"]}, {"requirement": "R3", "code": SHADOW["R3"]}]}
    client = Scripted(bad, fix)
    suite = asyncio.run(write_shadow(client, prompt=task.prompt, requirements=task.requirements, base_dir=task.base_dir, model="m", family="deepseek",
                                     guard={"oracle_files": rec["oracle_files"], "hidden_tests": rec["tests_b"]}))
    assert sorted(suite.requirements) == ["R1", "R2", "R3"] and suite.files[shadow_path("R1")] == SHADOW["R1"]
    assert suite.problems == ["unknown requirement id 'R9'"]  # the repair round filled R2 and R3; R9 is dropped
    assert len(client.seen) == 2 and "fix_these_modules" in client.seen[1][-1]["content"]
    sent = json.loads(client.seen[0][1]["content"])
    assert set(sent) == {"request", "requirements", "repository"}
    files, _, probs = parse_shadow({"tests": [{"requirement": "R1", "code": "x = 1\n"}]}, ["R1"])
    assert files == {} and probs == ["R1: no test function", "R1: no shadow module"]
    with pytest.raises(ValueError, match="leaks"):
        asyncio.run(write_shadow(Scripted(bad), prompt=task.prompt + " def test_gamma():", requirements=task.requirements, base_dir=task.base_dir,
                                 guard={"oracle_files": {}, "hidden_tests": rec["tests_b"]}))
    assert ShadowSuite.from_json(suite.to_json()).sha256 == suite.sha256
    # many requirements are asked in groups, each with the whole request
    two = Scripted({"tests": [{"requirement": "R1", "code": SHADOW["R1"]}, {"requirement": "R2", "code": SHADOW["R2"]}]},
                   {"tests": [{"requirement": "R3", "code": SHADOW["R3"]}]})
    grouped = asyncio.run(write_shadow(two, prompt=task.prompt, requirements=task.requirements, base_dir=task.base_dir, chunk=2))
    assert sorted(grouped.requirements) == ["R1", "R2", "R3"] and len(two.seen) == 2
    assert [r["id"] for r in json.loads(two.seen[1][1]["content"])["requirements"]] == ["R3"] and "[R1]" in json.loads(two.seen[1][1]["content"])["request"]


def test_runner_matches_the_grader_and_runs_the_shadow_apart(built) -> None:
    task, _, _ = built
    runner = WindowRunner(task)
    for sol in (task.oracle_patch(), "", alternatives_of(task)["alt_module"], next(iter(task.mutants())).solution):
        rep, _ = asyncio.run(task.grade(sol))
        v1 = asyncio.run(runner.hidden(sol))
        assert v1["reward"] == rep.reward and v1["milestones"] == rep.meta["milestones"] and v1["phi"] == rep.meta["phi"]
    sh = asyncio.run(runner.shadow(shadow_suite(), task.oracle_patch()))
    assert all(t.startswith("crucible_shadow/") for t in sh) and sh[f"{shadow_path('R3')}::test_r3_gamma_value_guess"] != "passed"
    st = asyncio.run(runner.strict(task.oracle_patch()))
    assert st["pass"] and asyncio.run(runner.strict(""))["pass"] is False
    assert asyncio.run(runner.shadow(shadow_suite(), "not a patch")) == {}  # a solution that does not apply: no test passes


def test_constructed_truth() -> None:
    ms = {"R1": "m01", "R2": "m02", "R3": "m03"}
    valid = {"milestone": "m02", "status": "valid", "expected": {"m01": True, "m02": False, "m03": None}}
    assert truth_for("mutant", list(ms), milestone_of=ms, mutant=valid) == {"R1": True, "R2": False, "R3": None}
    assert truth_for("mutant", list(ms), milestone_of=ms, mutant={**valid, "status": "broad"}) == {"R1": None, "R2": False, "R3": None}
    assert truth_for("floor", ["R1"], milestone_of=ms) == {"R1": False} and truth_for("solver", ["R1"], milestone_of=ms) == {"R1": None}


def test_ownership_rules() -> None:
    assert assign_owner(v1=True, v2=False, truth=False, shadow_gates=False, spec_flagged=False)[:3] == ("verifier1", "verifier_gap", True)
    assert assign_owner(v1=False, v2=True, truth=True, shadow_gates=False, spec_flagged=False)[:3] == ("verifier1", "false_fail", True)
    assert assign_owner(v1=True, v2=False, truth=True, shadow_gates=True, spec_flagged=False)[:3] == ("verifier2", "unsound_shadow", True)
    assert assign_owner(v1=False, v2=True, truth=False, shadow_gates=True, spec_flagged=False)[:3] == ("verifier2", "weak_shadow", True)
    assert assign_owner(v1=True, v2=False, truth=None, shadow_gates=True, spec_flagged=False)[:3] == ("verifier1", "verifier_gap", False)
    assert assign_owner(v1=True, v2=False, truth=None, shadow_gates=False, spec_flagged=False)[:3] == ("verifier2", "unsound_shadow", False)
    assert assign_owner(v1=False, v2=True, truth=None, shadow_gates=True, spec_flagged=True)[:3] == ("spec", "unstated_format", False)
    assert assign_owner(v1=False, v2=True, truth=None, shadow_gates=True, spec_flagged=False)[:3] == ("verifier2", "weak_shadow", False)


def test_qualification_flags_until_sound_and_discriminating(built) -> None:
    task, _, _ = built
    runner = WindowRunner(task)
    suite = shadow_suite()
    weak = ShadowSuite({**suite.files, shadow_path("R1"): "def test_r1_base():\n    from pkg.core import base\n    assert base() == 0\n"}, suite.requirements)

    async def go(s):
        alts = {n: await runner.shadow(s, p) for n, p in alternatives_of(task).items()}
        muts = {str(m.id): (str(m.milestone), await runner.shadow(s, m.solution)) for m in task.mutants()}
        return qualify(s, oracle=await runner.shadow(s, task.oracle_patch()), alternatives=alts, start=await runner.shadow(s, ""), mutants=muts,
                       req_milestone={"R1": "m01", "R2": "m02", "R3": "m03"})

    q = asyncio.run(go(suite))
    assert q.gating() == ["R1", "R2", "R3"] and q.to_json()["counts"]["fails_oracle"] == 1
    d = unsound_defects("w", q, suite, {"R3": "m03"})
    assert len(d) == 1 and d[0].owner == "verifier2" and d[0].defect == "unsound_shadow" and d[0].confirmed
    qw = asyncio.run(go(weak))
    assert "R1" not in qw.gating() and "pass the start state" in " ".join(qw.requirements["R1"]["why"])
    assert qw.verdict(None) == {"R1": None, "R2": None, "R3": None}


def _gates(task, rec, **kw) -> P2Gates:
    alts = alternatives_of(task)
    return P2Gates(task=task, runner=WindowRunner(task), alternatives=alts, suite=shadow_suite(), states=asyncio.run(end_states(task, alts)), **kw)


def test_gate_handlers_accept_the_clean_toy_and_send_a_weak_check_back(built, tmp_path) -> None:
    task, rec, _ = built
    assert validate_machine(gate_machine(), handlers=gate_handlers()) == []
    g = _gates(task, rec)
    res = asyncio.run(MachineRun(gate_machine(), run_dir=tmp_path / "clean", handlers=gate_handlers(), services=g, run_id="clean").run())
    assert res.status == "done", res
    assert g.results["s8"]["false_fail"] == [] and g.results["s9"]["differential"]["confirmed_by_owner"]["verifier1"] == 0
    weak = seeded_task(task, rec, test_name="test_gamma_exists", body="def test_gamma_exists():\n    from pkg import gamma as g\n    assert callable(g.gamma)\n",
                       replace=True)
    gw = _gates(weak, rec)
    wrong = with_oracle(weak, {"pkg/gamma.py": "def gamma():\n    return 0\n"})
    gw.states.append(EndStateRec("mutant_wrong_gamma", "mutant", wrong, {"R1": True, "R2": True, "R3": False}))
    res2 = asyncio.run(MachineRun(gate_machine(), run_dir=tmp_path / "weak", handlers=gate_handlers(), services=gw, run_id="weak").run())
    assert res2.status != "done"
    v = gw.results["s9"]["differential"]
    assert v["confirmed_by_owner"]["verifier1"] == 1 and v["defects"][0]["defect"] == "verifier_gap"
    # a task nobody can solve stops at S5
    dead = seeded_task(task, rec, test_name="test_never", body="def test_never():\n    assert False\n")
    gd = _gates(dead, rec)
    res3 = asyncio.run(MachineRun(gate_machine(), run_dir=tmp_path / "dead", handlers=gate_handlers(), services=gd, run_id="dead").run())
    assert res3.status == "rejected" and not gd.results["s5"]["clean_world"]["ok"]


def test_session_form_of_the_writer_validates() -> None:
    m = {"run": {"max_transitions": 10},
         "states": [{"id": "spec", "fn": "spec", "entry": True, "outputs": [{"name": "spec", "type": "json"}]},
                    {"id": "author", "fn": "author", "outputs": [{"name": "prompt", "type": "text"}], "inputs": [{"name": "spec", "type": "json", "from": "spec.spec"}]},
                    {"id": "oracle", "fn": "oracle", "outputs": [{"name": "oracle", "type": "json"}]},
                    shadow_session_state(), {"id": "done", "terminal": "done"}],
         "transitions": [{"from": "spec", "to": "author"}, {"from": "author", "to": "oracle"}, {"from": "oracle", "to": "second_verifier"},
                         {"from": "second_verifier", "to": "done"}]}
    assert validate_machine(m) == []
