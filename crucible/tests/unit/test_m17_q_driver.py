"""M17 agent Q: the per-task machine (factory/driver.py) on fake role clients and fake validation stages (no paid calls).

Covers: a full run to an accepted, validated TaskCard; the loop-backs (the verifier fails mutants -> back to the verifier
writer with the findings; a World bug found in the audit -> back to the author); oracle isolation (the verifier writer
never receives oracle material, and findings that quote it are withheld); exhaustion -> escalate (never done) and the
decision that answers it; resume from the journal after a crash mid-state and between a settle and its transition; the
budget cap stopping a run cleanly; the wall-clock and transition caps; the cross-family rule refused at validation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from crucible.actors.llm import BudgetExceeded, book_scoped
from crucible.factory import driver
from crucible.factory.cards import TaskCard
from crucible.factory.machine import MachineError, MachineRun
from crucible.factory.models import ModelBook
from crucible.forge.budget import Ledger

ORACLE_SECRET = "ORACLE-SECRET alpha beta gamma delta epsilon zeta eta theta iota kappa"
REQS = [{"id": "r1", "text": "refund the duplicate invoice INV-77", "kind": "behaviour"},
        {"id": "r2", "text": "notify the vendor by email", "kind": "side_effect_scope"}]
PROMPT = "Please refund the duplicate invoice INV-77. Then notify the vendor by email."
INTENT = {"id": "intent-dup-refund", "behaviour": "refund restated duplicates", "failure_mode": "pays twice",
          "difficulty_axes": ["distractors", "aliased vendors"]}
OBS = [{"id": "obs1", "type": "expert_end_state", "pointer": "file://x", "sha256": "0" * 64, "created": "2026-10-14T00:00:00Z",
        "by": "human:ann"}]


def cell(key: str = "refund--ap", surface: str = "ap") -> dict[str, Any]:
    return {"key": key, "behaviour": "refund", "surface": surface}


def checks() -> dict[str, Any]:
    return {"checks": [{"id": "c1", "requirement_ids": ["r1"], "observation_ids": ["obs1"], "gating": True, "spec": {"rule": "refunded"}},
                       {"id": "c2", "requirement_ids": ["r2"], "observation_ids": ["obs1"], "gating": True, "spec": {"rule": "emailed"}}]}


def author(_: dict) -> dict:
    spans = {r["id"]: [PROMPT.index(r["text"]), PROMPT.index(r["text"]) + len(r["text"])] for r in REQS}
    return {"world": {"invoices": [{"id": "INV-77", "amount": 97.31}]}, "prompt": PROMPT, "spans": spans}


class FakeLLM:
    """One fake for every role: answers by the ``ROLE:`` line of the system message; records each payload."""

    def __init__(self, usd: float = 0.0, overrides: dict[str, Any] | None = None):
        self.usd = usd
        self.calls: list[tuple[str, dict]] = []
        self.script: dict[str, Any] = {
            "planner": lambda p: {"requirements": REQS, "verifier_plan": "one check per requirement", "expected_causes": ["skill_gap"],
                                  "surface": {"domain": "finance", "world": "ap"}, "long": False},
            "author": author,
            "realism": lambda p: {"ok": True, "findings": []},
            "oracle_author": lambda p: {"solution": [{"kind": "tool_call", "function_name": "refund", "arguments": {"memo": ORACLE_SECRET}}]},
            "verifier_writer": lambda p: checks(),
            "second_verifier": lambda p: checks(),
            "tie_break": lambda p: {"owner": "verifier", "reason": "x"},
            **(overrides or {}),
        }

    async def complete_json(self, messages: list[dict], **kw: Any) -> dict:
        role = messages[0]["content"].split("\n", 1)[0].removeprefix("ROLE: ")
        payload = json.loads(messages[1]["content"])
        self.calls.append((role, payload))
        if self.usd:
            book_scoped(self.usd)  # what LLMClient books from OpenRouter's usage.cost
        r = self.script[role]
        if isinstance(r, Exception):
            raise r
        return r(payload)

    def payloads(self, role: str) -> list[dict]:
        return [p for r, p in self.calls if r == role]


OK = {
    "floors": driver.verdict(True, measured={"floors": {"no_op": 0.0, "empty_action": 0.0}}),
    "probes": driver.verdict(True, measured={"mutant_kill": [2, 2], "alternatives_pass": [2, 2]}),
    "hackers": driver.verdict(True, measured={"hacks_confirmed": 0}),
    "differential": driver.verdict(True, measured={"differential_disagreements": [0, 8]}),
    "audit": driver.verdict(True, measured={"causes": [{"trace": f"t{i}", "cause": "skill_gap"} for i in range(4)], "fp": [0, 8], "fn": [0, 8]}),
    "flake": driver.verdict(True, measured={"flake_upper": 0.03}),
    "horizon": driver.verdict(True, measured={"horizon": None}),
    "gate": driver.verdict(True, measured={"accept": {"state": "accepted", "blocking": {}}}),
}


class Crash(BaseException):
    """A process crash (not an Exception: the executor must not swallow it)."""


class FakeStages:
    """Scripted validation stages: ``plan[stage]`` is a list of verdicts (or exceptions) consumed in order; the stage's
    OK verdict once the list is empty. Solvers: 2 policies x 4 attempts, half passing."""

    def __init__(self, plan: dict[str, list[Any]] | None = None, *, solver_delay: float = 0.0):
        self.plan = {k: list(v) for k, v in (plan or {}).items()}
        self.calls: dict[str, int] = {}
        self.drafts: dict[str, list[dict]] = {}
        self.solver_delay = solver_delay

    def _fn(self, name: str) -> Any:
        async def run(draft: dict, env: driver.FactoryEnv, ctx: Any) -> dict:
            self.calls[name] = self.calls.get(name, 0) + 1
            self.drafts.setdefault(name, []).append(draft)
            todo = self.plan.get(name) or []
            if todo:
                v = todo.pop(0)
                if isinstance(v, BaseException):
                    raise v
                return v
            return OK[name]

        return run

    async def solvers(self, draft: dict, env: driver.FactoryEnv, ctx: Any) -> dict:
        self.calls["solvers"] = self.calls.get("solvers", 0) + 1
        if self.solver_delay:
            await asyncio.sleep(self.solver_delay)
        return {"traces": [{"id": f"{p}#{i}", "policy": p, "passed": i % 2 == 0, "end_state": {"i": i}} for p in ("gpt-oss-20b", "llama-8b")
                           for i in range(4)]}

    def stages(self) -> driver.Stages:
        return driver.Stages(**{n: self._fn(n) for n in OK}, solvers=self.solvers)


def env_of(stages: FakeStages, **kw: Any) -> driver.FactoryEnv:
    return driver.FactoryEnv(stages=stages.stages(), **kw)


def run_of(tmp: Path, llm: FakeLLM, stages: FakeStages, *, name: str = "task-1", models: Any = None, ledger: Ledger | None = None,
           machine: dict | None = None, **envkw: Any) -> MachineRun:
    clients = dict.fromkeys(["planner", "author", "realism", "oracle_author", "verifier_writer", "mutant_builder", "hacker",
                             "second_verifier", "auditor"], llm)
    return driver.task_run(run_dir=tmp / name, intent=INTENT, cell=cell(), env=env_of(stages, tie_break=llm, **envkw), clients=clients,
                           models=models if models is not None else ModelBook().roles, observations=OBS, ledger=ledger, machine=machine)


def events(run: MachineRun) -> list[dict]:
    return [json.loads(x) for x in run.journal.path.read_text().splitlines() if x.strip()]


def fail(owner: str | None, *findings: str) -> dict:
    return driver.verdict(False, owner=owner, findings=findings)


# ---- the full run -----------------------------------------------------------------------------------------------------
async def test_full_run_with_loop_backs_reaches_an_accepted_task_card(tmp_path: Path) -> None:
    llm = FakeLLM()
    stages = FakeStages({
        "probes": [fail("verifier", "mutant for r1 (refund twice) passed every check")],
        "audit": [fail("author", "world_bug: the vendor table has no row for INV-77's vendor")],
    })
    run = run_of(tmp_path, llm, stages)
    res = await run.run()
    assert res.status == "done", (res.reason, res.decision_request)
    # verifier: first entry, the mutant loop-back, then the re-run after the author fixed the World
    assert res.entries["verifier"] == 3 and res.entries["author"] == 2 and res.entries["spec"] == 1
    v_payloads = llm.payloads("verifier_writer")
    assert "mutant for r1 (refund twice) passed every check" in v_payloads[1]["findings_to_address"]
    assert "findings_to_address" not in v_payloads[0]
    a_payloads = llm.payloads("author")
    assert any("world_bug" in f for f in a_payloads[1]["findings_to_address"])
    card = TaskCard.model_validate(res.outputs["select"]["card"])
    assert card.status == "accepted" and card.intent == INTENT["id"] and card.verifier.mutant_kill == (2, 2)
    assert card.verifier.fp == (0, 8) and card.verifier.flake_upper == 0.03 and card.stats["target"] == "gpt-oss-20b"
    assert card.ledger_problems({o["id"]: __import__("crucible.factory.cards", fromlist=["x"]).ObservationRecord.model_validate(o) for o in OBS}) == []
    kinds = [e["event"] for e in events(run)]
    assert kinds.count("transition") == res.transitions and kinds[-1] == "status"


async def test_the_verifier_writers_never_see_the_oracle(tmp_path: Path) -> None:
    llm = FakeLLM()
    stages = FakeStages({"probes": [fail("verifier", f"mutant kept memo {ORACLE_SECRET} and passed", "mutant for r2 passed")]})
    res = await run_of(tmp_path, llm, stages).run()
    assert res.status == "done"
    for role in ("verifier_writer", "second_verifier"):
        for p in llm.payloads(role):
            text = json.dumps(p)
            assert "ORACLE-SECRET" not in text and "CRUCIBLE-CANARY" not in text and "solution" not in text
    second = llm.payloads("verifier_writer")[1]["findings_to_address"]
    assert "mutant for r2 passed" in second and any("withheld" in f for f in second)
    # the stages that may see the oracle got it, canary included
    assert stages.drafts["floors"][0]["oracle"]["_canary"].startswith("CRUCIBLE-CANARY-")


async def test_validator_refuses_a_verifier_that_reads_the_oracle() -> None:
    m = driver.driver_machine()
    ver = next(s for s in m["states"] if s["id"] == "verifier")
    ver["inputs"].append({"name": "oracle", "type": "json", "from": "oracle.oracle"})
    with pytest.raises(MachineError, match="isolated from"):
        MachineRun(m, run_dir=Path("/nonexistent/never-created"), handlers=driver.handlers())


async def test_verifier_output_carrying_the_oracle_canary_is_a_breach(tmp_path: Path) -> None:
    llm = FakeLLM()
    run = run_of(tmp_path, llm, FakeStages())

    def leaky(p: dict) -> dict:
        c = checks()
        c["checks"][0]["spec"] = {"expected": run.canaries.get("oracle")}
        return c

    llm.script["verifier_writer"] = leaky
    res = await run.run()
    assert res.status == "paused" and res.decision_request["kind"] == "failed" and "breach" in res.reason


# ---- exhaustion is escalate, never done ---------------------------------------------------------------------------------
async def test_exhausted_loop_escalates_and_a_decision_answers_it(tmp_path: Path) -> None:
    llm = FakeLLM()
    stages = FakeStages({"probes": [fail("verifier", "mutant for r1 passed")] * 20})
    run = run_of(tmp_path, llm, stages)
    res = await run.run()
    assert res.status == "paused" and res.status != "done"
    req = res.decision_request
    assert req["kind"] == "exhausted" and req["state"] == "verifier" and "revise" in req["options"] and "drop" in req["options"]
    assert res.entries["verifier"] == 4 and "mutant for r1 passed" in req["findings"]
    # a person grants one more round: the loop runs once more and escalates again
    res = await run.resume({"action": "revise", "actor": "human:ann", "reason": "one more try", "extra_entries": 1})
    assert res.status == "paused" and res.entries["verifier"] == 5
    # a fresh process sees the same pause from the journal, and a drop ends the run rejected
    again = run_of(tmp_path, FakeLLM(), stages)
    assert again.status == "paused" and again.decision_request["kind"] == "exhausted"
    res = await again.resume({"action": "drop", "actor": "human:ann", "reason": "the intent is not checkable"})
    assert res.status == "rejected"
    with pytest.raises(RuntimeError, match="only a paused or stopped run resumes"):
        await again.resume({"action": "revise"})


async def test_unexplained_gate_failure_escalates(tmp_path: Path) -> None:
    stages = FakeStages({"gate": [fail(None, "gate A12: error")]})
    res = await run_of(tmp_path, FakeLLM(), stages).run()
    assert res.status == "paused" and res.decision_request["kind"] == "no_transition" and res.decision_request["state"] == "gate"


async def test_gate_failure_with_an_owner_loops_back(tmp_path: Path) -> None:
    stages = FakeStages({"gate": [fail("oracle", "gate A1: the reference fails in a fresh World")]})
    res = await run_of(tmp_path, FakeLLM(), stages).run()
    assert res.status == "done" and res.entries["oracle"] == 2


async def test_audit_tie_goes_to_the_tie_break(tmp_path: Path) -> None:
    llm = FakeLLM()
    stages = FakeStages({"audit": [driver.verdict(False, owner="tie", findings=["two verifiers disagree on t3"])]})
    res = await run_of(tmp_path, llm, stages).run()
    assert res.status == "done" and llm.payloads("tie_break") and res.entries["verifier"] == 2


async def test_calibration_outside_the_window_goes_back_to_the_author(tmp_path: Path) -> None:
    stages = FakeStages()
    n = {"k": 0}

    async def solvers(draft: dict, env: Any, ctx: Any) -> dict:
        n["k"] += 1
        win = n["k"] > 1  # first batch: everything passes (too easy)
        return {"traces": [{"id": f"{p}{i}", "policy": p, "passed": (i % 2 == 0) if win else True} for p in ("a", "b") for i in range(4)]}

    stages.solvers = solvers  # type: ignore[method-assign]
    llm = FakeLLM()
    res = await run_of(tmp_path, llm, stages).run()
    assert res.status == "done" and res.entries["author"] == 2
    assert any("harder" in f and "distractors" in f for f in llm.payloads("author")[1]["findings_to_address"])


async def test_too_few_solver_runs_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    stages = FakeStages()

    async def solvers(draft: dict, env: Any, ctx: Any) -> dict:
        return {"traces": [{"id": "x", "policy": "a", "passed": True}]}

    stages.solvers = solvers  # type: ignore[method-assign]
    res = await run_of(tmp_path, FakeLLM(), stages).run()
    assert res.status == "paused" and res.decision_request["kind"] == "failed" and "05 §6.5" in res.reason


# ---- resume from the journal --------------------------------------------------------------------------------------------
async def test_resume_from_the_journal_after_a_crash_mid_state(tmp_path: Path) -> None:
    llm = FakeLLM()
    stages = FakeStages({"flake": [Crash()], "probes": [fail("verifier", "mutant for r1 passed")]})
    with pytest.raises(Crash):
        await run_of(tmp_path, llm, stages).run()
    before = len(llm.calls)
    llm2 = FakeLLM()
    run = run_of(tmp_path, llm2, stages)
    assert run.status == "running" and run.pending["state"] == "flake"
    res = await run.run()
    assert res.status == "done"
    assert llm2.calls == []  # nothing before the crash is redone: every model state came from the journal
    assert before > 0 and stages.calls["flake"] == 2 and stages.calls["probes"] == 2
    flake_enters = [e for e in events(run) if e["event"] == "enter" and e["state"] == "flake"]
    assert [e["entry"] for e in flake_enters] == [1, 1] and flake_enters[1]["resumed"] is True
    assert res.entries["verifier"] == 2  # the loop count survived the restart


async def test_resume_between_a_settle_and_its_transition(tmp_path: Path) -> None:
    llm = FakeLLM()
    full = run_of(tmp_path, llm, FakeStages(), name="full")
    await full.run()
    lines = full.journal.path.read_text().splitlines()
    cut = next(i for i, x in enumerate(lines) if '"event": "settle"' in x and '"state": "verifier"' in x)
    part = tmp_path / "part"
    part.mkdir()
    (part / "journal.jsonl").write_text("\n".join(lines[: cut + 1]) + "\n" + '{"event": "transit')  # a torn tail too
    llm2 = FakeLLM()
    res = await run_of(tmp_path, llm2, FakeStages(), name="part").run()
    assert res.status == "done" and not llm2.payloads("verifier_writer") and llm2.payloads("second_verifier")


async def test_a_journal_from_another_machine_is_refused(tmp_path: Path) -> None:
    await run_of(tmp_path, FakeLLM(), FakeStages()).run()
    with pytest.raises(MachineError, match="belongs to machine"):
        run_of(tmp_path, FakeLLM(), FakeStages(), machine=driver.driver_machine(max_verifier=5))


# ---- caps ---------------------------------------------------------------------------------------------------------------
async def test_budget_cap_stops_the_run_cleanly_and_a_larger_cap_continues_it(tmp_path: Path) -> None:
    llm = FakeLLM(usd=0.02)
    ledger = Ledger(cap_usd=0.05)
    run = run_of(tmp_path, llm, FakeStages(), ledger=ledger)
    res = await run.run()
    assert res.status == "stopped" and res.reason.startswith("budget")
    assert res.usd == pytest.approx(ledger.committed) and ledger.committed == pytest.approx(0.06) and not ledger.reserved
    assert "oracle" not in res.entries  # nothing started after the cap was spent
    ledger.cap_usd = 10.0
    res = await run.run()
    assert res.status == "done" and res.usd == pytest.approx(ledger.committed)


async def test_a_client_out_of_credits_stops_the_run(tmp_path: Path) -> None:
    llm = FakeLLM(overrides={"oracle_author": BudgetExceeded("LLM budget 0.10 USD exhausted")})
    res = await run_of(tmp_path, llm, FakeStages()).run()
    assert res.status == "stopped" and "budget" in res.reason and res.entries["oracle"] == 1


async def test_wall_clock_and_transition_caps(tmp_path: Path) -> None:
    res = await run_of(tmp_path, FakeLLM(), FakeStages(), name="t", machine=driver.driver_machine(max_transitions=3)).run()
    assert res.status == "paused" and res.decision_request["kind"] == "transition_cap" and res.transitions == 3
    t = {"now": 0.0}

    def clock() -> float:
        t["now"] += 40.0
        return t["now"]

    clients = dict.fromkeys(["planner", "author", "realism", "oracle_author", "verifier_writer", "second_verifier"], FakeLLM())
    run = MachineRun(driver.driver_machine(budget_s=100), run_dir=tmp_path / "w", handlers=driver.handlers(), clients=clients,
                     params={"intent": INTENT, "cell": cell(), "observations": OBS}, services=env_of(FakeStages()), clock=clock)
    res = await run.run()
    assert res.status == "stopped" and res.reason.startswith("wall clock") and res.seconds >= 100


# ---- cross-family ---------------------------------------------------------------------------------------------------------
async def test_cross_family_assignment_is_refused_at_validation(tmp_path: Path) -> None:
    book = ModelBook({"second_verifier": "openrouter:qwen/qwen3.7-flash"})
    with pytest.raises(MachineError, match="cross-family rule: state second_verifier"):
        run_of(tmp_path, FakeLLM(), FakeStages(), models=book.roles)
    assert not (tmp_path / "task-1" / "journal.jsonl").exists()  # refused before anything ran
    # a family published under two prefixes is still one family
    book = ModelBook({"realism": "thudm/glm-4-9b", "author": "z-ai/glm-4.7-flash"})
    assert any("realism" in e and "glm" in e for e in book.check(driver.driver_machine()))
    assert ModelBook().check(driver.driver_machine()) == []


# ---- the default adapters onto agent R's modules, on a toy task ------------------------------------------------------------
class ToyTask:
    """A tool-call task: c1 passes when INV-77 is refunded, c2 when the vendor is emailed (the checkset decides which
    checks exist). Graded deterministically, with an end-state fingerprint."""

    def __init__(self, draft: dict):
        self.instruction = draft["prompt"]
        self.oracle = draft["oracle"]["solution"]
        self.checks = [c["id"] for c in draft["checkset"]["checks"]]
        self.negatives: list = []

    def noop(self) -> list:
        return []

    @staticmethod
    def _state(sol: Any) -> dict:
        acts = sol if isinstance(sol, list) else []
        return {"refunded": any(a.get("function_name") == "refund" and (a.get("arguments") or {}).get("invoice") == "INV-77" for a in acts
                                if isinstance(a, dict)),
                "emailed": any(isinstance(a, dict) and a.get("function_name") == "send_email" for a in acts)}

    async def grade(self, sol: Any) -> dict:
        st = self._state(sol)
        checks = {c: st["refunded"] if c == "c1" else st["emailed"] for c in self.checks}
        return {"reward": 1.0 if all(checks.values()) else 0.0, "checks": checks, "fingerprint": json.dumps(st, sort_keys=True)}

    def property_audit(self, actions: Any) -> bool:
        return all(self._state(actions).values())


ORACLE_ACTS = [{"kind": "tool_call", "tool_call_id": "1", "function_name": "get_invoice", "arguments": {"invoice": "INV-77"}},
               {"kind": "tool_call", "tool_call_id": "2", "function_name": "refund", "arguments": {"invoice": "INV-77"}},
               {"kind": "tool_call", "tool_call_id": "3", "function_name": "send_email", "arguments": {"to": "vendor"}}]


class Judge:
    """A fake auditor on R's Auditor protocol: quotes the trace, labels a planted tool error a World bug."""

    async def complete_json(self, messages: list[dict], **kw: Any) -> dict:
        u = json.loads(messages[1]["content"])
        text = u["trace"]
        cause = "world_bug" if "TOOL ERROR" in text else "skill_gap"
        return {"cause": cause, "solution_correct": u["verifier_verdict"] == "pass", "evidence": text[:20], "rationale": "r"}


async def test_default_adapters_run_on_agent_rs_modules(tmp_path: Path) -> None:
    from crucible.factory.audit import SeededCase, Trace
    from crucible.factory.machine import StateContext

    draft = {"task_id": "t1", "intent": INTENT, "cell": cell(), "observations": OBS, "spec": {"requirements": REQS, "milestones": []},
             "world": {}, "prompt": PROMPT, "spans": author({})["spans"], "oracle": {"solution": ORACLE_ACTS}, "checkset": checks(),
             "checkset2": checks()}
    traces = [{"id": f"a#{i}", "policy": "a", "passed": i % 2 == 0, "solution": ORACLE_ACTS if i % 2 == 0 else ORACLE_ACTS[:2],
               "text": f"step get_invoice INV-77 then {'refund and email' if i % 2 == 0 else 'stopped'} (#{i})"} for i in range(4)]
    draft["runs"] = {"traces": traces}
    env = driver.FactoryEnv(materialize=ToyTask, seeded_controls=lambda d: [SeededCase(Trace("s1", "TOOL ERROR: refund() returned 500", False),
                                                                                     "world_bug")])
    ctx = StateContext("t1", "x", 1, 1, {}, [], {}, client=Judge())
    floors = await driver._default_floors(draft, env, ctx)
    assert floors["ok"] and floors["measured"]["floors"]["no_op"] == 0.0
    probes = await driver._default_probes(draft, env, StateContext("t1", "probes", 1, 1, {}, [], {}))
    assert len(probes["measured"]["mutant_kill"]) == 2 and probes["owner"] in (None, "verifier")
    hackers = await driver._default_hackers(draft, env, ctx)
    assert hackers["ok"] and hackers["measured"]["hacks_confirmed"] == 0 and hackers["measured"]["attempts"] > 0
    diff = await driver._default_differential(draft, env, ctx)
    assert diff["ok"] and diff["measured"]["differential_disagreements"] == [0, 4]
    aud = await driver._default_audit({**draft, "differential": diff}, env, ctx)
    assert aud["ok"], aud["findings"]
    assert aud["measured"]["fn"] == [0, 2] and {c["cause"] for c in aud["measured"]["causes"]} == {"skill_gap"}
    from crucible.factory.flake import FlakeConfig

    flake = await driver._default_flake(draft, driver.FactoryEnv(materialize=ToyTask, flake_config=FlakeConfig(reruns=10, upper_max=0.5)), ctx)
    assert flake["ok"] and 0 < flake["measured"]["flake_upper"] <= 0.5
    # a verifier that disagrees with the first on a requirement: the differential reports it and the audit makes it a tie
    loose = {"checks": [checks()["checks"][0]]}
    diff = await driver._default_differential({**draft, "checkset2": loose}, env, ctx)
    assert not diff["ok"] and diff["measured"]["differential_disagreements"][0] > 0
    aud = await driver._default_audit({**draft, "differential": diff}, env, ctx)
    assert aud["owner"] == "tie"
    # a broken World in a real trace goes back to the author
    bad = [{**traces[1], "text": "TOOL ERROR: refund() returned 500"}]
    aud = await driver._default_audit({**draft, "runs": {"traces": bad}, "differential": {"ok": True}}, env, ctx)
    assert not aud["ok"] and aud["owner"] == "author"
