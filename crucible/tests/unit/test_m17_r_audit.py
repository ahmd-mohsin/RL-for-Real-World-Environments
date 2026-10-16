"""M17 owner R: the FailureCause auditor. Adversarial cases: a judge that always says skill_gap is caught by a seeded World
bug; a judge that invents its evidence or answers outside the taxonomy is rejected; FP/FN candidates come from auditor vs
verifier disagreement; the cause mix carries its denominator and interval."""

from __future__ import annotations

from crucible.factory.audit import CAUSES, AuditConfig, Auditor, SeededCase, Trace, audit, human_sample, validate
from tests.unit.m17_r_fakes import ScriptedClient, user_payload

WORLD_BUG = "step 3: tickets.issue_refund -> ERROR internal: sqlite3.OperationalError: no such table: refunds_v2"
SKILL = "step 2: tickets.set_status(ticket_id=41, status='closed') -> ok\nfinal: I closed the ticket."
FORMAT = "final answer: total = 42,50 EUR (the checker wanted 42.50)"
CORRECT = "step 1: fs.create(report.csv) ok; step 2: set_total 42.5 ok; step 3: set_header ID,TOTAL ok"


def always(cause: str):
    """A degenerate judge: always the same cause, quoting the first line of the trace (valid evidence)."""

    def reply(messages):
        trace = user_payload(messages)["trace"]
        return {"cause": cause, "solution_correct": False, "evidence": trace.splitlines()[0][:40], "rationale": "x"}

    return ScriptedClient(reply)


def reader():
    """A judge that reads the trace."""

    def reply(messages):
        p = user_payload(messages)
        t = p["trace"]
        if "ERROR internal" in t:
            return {"cause": "world_bug", "solution_correct": False, "evidence": "ERROR internal: sqlite3.OperationalError", "rationale": "tool crashed"}
        if "checker wanted" in t:
            return {"cause": "unstated_format", "solution_correct": True, "evidence": "total = 42,50 EUR", "rationale": "format"}
        if "set_header ID,TOTAL ok" in t:
            return {"cause": "verifier_defect" if p["verifier_verdict"] == "fail" else None, "solution_correct": True,
                    "evidence": "set_header ID,TOTAL ok", "rationale": "all done"}
        if "FAKE PASS" in t:
            return {"cause": None, "solution_correct": False, "evidence": "FAKE PASS", "rationale": "claimed only"}
        return {"cause": "skill_gap", "solution_correct": False, "evidence": t.splitlines()[0][:30], "rationale": "wrong status"}

    return ScriptedClient(reply)


SEEDS = [SeededCase(Trace("seed_world_bug", WORLD_BUG, False), "world_bug"),
         SeededCase(Trace("seed_verifier", CORRECT, False), "verifier_defect", solution_correct=True)]


def batch(n_skill: int = 8, extra: tuple = ()) -> list[Trace]:
    ts = [Trace(f"s{i}", f"attempt {i}\n{SKILL}", False, policy="p1" if i % 2 else "p2", failing_checks=["outcome/status_41"]) for i in range(n_skill)]
    return ts + list(extra)


async def test_always_skill_gap_judge_is_caught_by_the_seeded_world_bug():
    rep = await audit(batch(), Auditor(always("skill_gap"), model="m-a"), prompt="p", seeded=SEEDS)
    assert rep.auditor_trusted is False and not rep.ok
    miss = {s["id"]: s for s in rep.seeded}
    assert miss["seed_world_bug"]["ok"] is False and miss["seed_world_bug"]["got"] == "skill_gap"
    assert any("seeded controls" in r for r in rep.reasons())
    # its real-batch numbers would have looked perfect: 100% skill gap
    assert rep.skill_gap_share.pair() == (8, 8)


async def test_reading_judge_is_trusted_and_the_mix_has_intervals():
    extra = (Trace("w1", f"attempt w\n{WORLD_BUG}", False), Trace("f1", FORMAT, False), Trace("ok1", CORRECT, True), Trace("fp1", "FAKE PASS: claimed done", True),
             Trace("fn1", CORRECT, False))
    client = reader()
    rep = await audit(batch(extra=extra), Auditor(client, model="m-b"), prompt="p", seeded=SEEDS)
    assert rep.auditor_trusted is True
    assert rep.failing == 11 and rep.mix["skill_gap"].pair() == (8, 11)
    lo, hi = rep.mix["skill_gap"].interval
    assert lo < 8 / 11 < hi
    assert rep.fp_candidates == ["fp1"] and rep.fn_candidates == ["f1", "fn1"]
    assert rep.send_back == {"world": ["w1"]}
    assert not rep.ok and any("back to world" in r for r in rep.reasons())
    # seeded ids never reach the judge (opaque), and the batch order is not the input order
    seen = [user_payload(m)["trace"] for m in client.calls]
    assert len(seen) == 15 and seen != [t.text for t in batch(extra=extra)] + [s.trace.text for s in SEEDS]


async def test_skill_gap_share_below_v1_fails():
    extra = tuple(Trace(f"f{i}", FORMAT, False) for i in range(5))
    rep = await audit(batch(n_skill=5, extra=extra), Auditor(reader()), prompt="p", seeded=SEEDS)
    assert rep.skill_gap_share.pair() == (5, 10)
    assert any("hard for the wrong reason" in r for r in rep.reasons())
    ok = await audit(batch(n_skill=8), Auditor(reader()), prompt="p", seeded=SEEDS)
    assert ok.ok, ok.reasons()


def test_validate_rejects_invented_evidence_and_out_of_taxonomy_causes():
    t = Trace("t", SKILL, False)
    assert validate(t, {"cause": "skill_gap", "evidence": "status='closed'"}).status == "labelled"
    assert validate(t, {"cause": "skill_gap", "evidence": "STATUS='CLOSED'   "}).status == "labelled"  # case and whitespace folded
    bad = validate(t, {"cause": "skill_gap", "evidence": "the agent refunded twice"})
    assert bad.status == "rejected" and bad.cause is None and "verbatim" in bad.why
    assert validate(t, {"cause": "laziness", "evidence": "status='closed'"}).status == "rejected"
    assert validate(t, {"cause": "skill_gap", "evidence": ""}).status == "rejected"
    ab = validate(t, {"cause": "abstain", "evidence": ""})
    assert ab.status == "abstained" and ab.cause is None
    assert set(CAUSES) == {"skill_gap", "spec_defect", "unstated_format", "world_bug", "verifier_defect", "budget", "env_flake"}


async def test_facts_are_not_judged_and_a_broken_client_abstains():
    def boom(messages):
        raise RuntimeError("provider down")

    client = ScriptedClient(boom)
    a = Auditor(client)
    masked = await a.label(Trace("m", "x", None), prompt="p")
    budget = await a.label(Trace("b", "x", False, meta={"stop": "max_steps"}), prompt="p")
    assert (masked.cause, masked.status) == ("env_flake", "deterministic") and (budget.cause, budget.status) == ("budget", "deterministic")
    assert client.calls == []
    err = await a.label(Trace("e", "x", False), prompt="p")
    assert err.status == "rejected" and "provider down" in err.why
    rep = await audit([Trace("e", "x", False)], a, prompt="p")
    assert rep.abstained == ["e"] and rep.auditor_trusted is None and not rep.ok
    off = Auditor(client, config=AuditConfig(deterministic=False))
    assert (await off.label(Trace("m", "x", None), prompt="p")).status == "rejected"


def test_human_sample_is_ten_percent_reproducible():
    ids = [f"t{i}" for i in range(30)]
    s = human_sample(ids)
    assert len(s) == 3 and s == human_sample(list(reversed(ids)))
    assert len(human_sample(ids[:3])) == 1 and human_sample([]) == []
