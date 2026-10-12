"""M8 audit regressions (docs/audit/2026-10-12/M08-forge.md): one test per reproduced defect (V1-V22) and per pick.

Every test fails on the M7 audit tree (526fdb5). Symbols the audit added are imported inside the tests, so on the older
tree each test fails on its own.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from crucible.core.store import Store
from crucible.forge.budget import BudgetExhausted, Ledger
from crucible.forge.orchestrator import Journal, Node, Orchestrator, Plan, RoleInfraError
from crucible.forge.repo import ForgeRepo, ZoneBreach


class Out(BaseModel):
    text: str = ""


SPENT = [0.0]


class Role:
    version = "1"

    def __init__(self, usd: float = 0.0, *, text="ok", delay: float = 0.02, raise_: BaseException | None = None, scoped: bool = False):
        self.usd, self.text, self.delay, self.raise_, self.scoped = usd, text, delay, raise_, scoped

    async def run(self, staged, *, feedback, knobs):
        await asyncio.sleep(self.delay)
        SPENT[0] += self.usd
        if self.scoped:
            from crucible.actors.llm import book_scoped

            book_scoped(self.usd)
        await asyncio.sleep(self.delay)
        if self.raise_:
            raise self.raise_
        return Out(text=self.text(staged) if callable(self.text) else self.text)


def orch(d: Path, roles, cap: float = 1.0, **kw):
    SPENT[0] = 0.0
    led = Ledger(cap_usd=cap, observe=lambda: SPENT[0])
    return Orchestrator(Store(d / "s"), Journal(d / "j.jsonl"), led, roles, **kw), led


ORACLE = "tickets.issue_refund(ticket_id=38, order_id='O-7108', amount_usd=20.90, approved_by=None)"


# ---------------------------------------------------------------- the zoned repo (V1, V2)


def test_v1_no_write_down_of_oracle_material(tmp_path):
    r = ForgeRepo(tmp_path / "repo", family="toy")
    r.commit("builder", {"oracle/t1.txt": ORACLE + "\n"}, type_="oracle", summary="oracle")
    with pytest.raises(ZoneBreach, match="oracle"):  # the adversary reads oracle/ but public/ readers do not
        r.commit("adversary", {"public/notes/t1.md": "a hint: " + ORACLE + "\n"}, type_="note", summary="notes")
    with pytest.raises(ZoneBreach, match="canary"):
        r.commit("judge", {"public/x.md": r._index["canaries"]["oracle"] + "\n"}, type_="note", summary="judge")


def test_v1_exemption_comes_from_the_parent_tree(tmp_path):
    r = ForgeRepo(tmp_path / "repo", family="toy")
    o = r.commit("builder", {"oracle/t1.txt": ORACLE + "\n"}, type_="oracle", summary="oracle")
    with pytest.raises(ZoneBreach):  # a verifier writer cannot read oracle/ (a copy elsewhere unlocked quoting everywhere)
        r.commit("verifier_writer", {"checks/t.sql": f"-- {ORACLE}\n"}, type_="checks", summary="c", parent=o.sha)


def test_v2_prefix_split_and_short_numeric_lines(tmp_path):
    r = ForgeRepo(tmp_path / "repo", family="toy")
    r.commit("builder", {"oracle/t1.txt": ORACLE + "\namount: 97.31\n"}, type_="oracle", summary="oracle")
    half = len(ORACLE) // 2
    for content in (f"-- {ORACLE}", ORACLE[:half] + "\n" + ORACLE[half:], "x = 1\namount: 97.31"):
        with pytest.raises(ZoneBreach):
            r.commit("verifier_writer", {"checks/t.sql": content + "\n"}, type_="checks", summary="c")
    r.commit("verifier_writer", {"checks/ok.sql": "SELECT COUNT(*) FROM refunds WHERE ticket_id = 38\n"}, type_="checks", summary="ok")


# ---------------------------------------------------------------- orchestrator and budget (V3-V6, V18, picks)


async def test_v3_concurrent_nodes_are_charged_their_own_spend(tmp_path):
    o, led = orch(tmp_path, {"a": Role(0.01, scoped=True), "b": Role(0.01, scoped=True)})
    r = await o.run_plan(Plan("t", "g", [Node("a", "a", estimate_usd=0.01), Node("b", "b", estimate_usd=0.01)]))
    assert round(led.committed, 6) == 0.02 and {k: round(v.cost_usd, 6) for k, v in r.nodes.items()} == {"a": 0.01, "b": 0.01}
    o2, led2 = orch(tmp_path / "2", {"a": Role(0.01), "b": Role(0.01)})  # unscoped roles: deltas, clamped to the observed total
    await o2.run_plan(Plan("t", "g", [Node("a", "a", estimate_usd=0.01), Node("b", "b", estimate_usd=0.01)]))
    assert led2.committed <= SPENT[0] + 1e-9


async def test_v4_overspend_and_role_errors_never_abort_the_run(tmp_path):
    o, led = orch(tmp_path, {"a": Role(0.5), "b": Role(0.01, delay=0.1)}, cap=0.1)
    res = await o.run_plans([Plan("t1", "g", [Node("a", "a", estimate_usd=0.05)]), Plan("t2", "g", [Node("b", "b", estimate_usd=0.01)])])
    assert [x.state for x in res] and led.exhausted  # returned, no BudgetExhausted
    o, led = orch(tmp_path / "2", {"a": Role(raise_=BudgetExhausted("client cap")), "b": Role()}, cap=0.1)
    await o.run_plan(Plan("t1", "g", [Node("a", "a", estimate_usd=0.08)]))
    assert led.reserved == {} and (await o.run_plan(Plan("t2", "g", [Node("b", "b", estimate_usd=0.05)]))).state == "constructed"
    o, _ = orch(tmp_path / "3", {"a": Role(raise_=KeyError("bad key")), "b": Role(delay=0.1)})
    r1, r2 = await o.run_plans([Plan("t1", "g", [Node("a", "a")]), Plan("t2", "g", [Node("b", "b")])])
    assert r1.state == "abandoned" and "KeyError" in r1.reason and r2.state == "constructed"


async def test_v5_v6_validators_and_canary_flag_invalidate_reuse(tmp_path):
    o, _ = orch(tmp_path, {"a": Role(text="SELECT 1; DROP TABLE x")})
    await o.run_plan(Plan("t", "g", [Node("a", "a")]))

    def strict(out, staged):
        return ["no DROP"] if "DROP" in out.text else []

    o2, _ = orch(tmp_path, {"a": Role(text="SELECT 1")})
    r = await o2.run_plan(Plan("t", "g", [Node("a", "a", validators=[strict])]))
    assert not r.nodes["a"].reused and r.outputs["a"]["text"] == "SELECT 1"
    o3, _ = orch(tmp_path, {"a": Role(text="SELECT 1")})
    r = await o3.run_plan(Plan("t", "g", [Node("a", "a", validators=[strict], canary_out=True)]))
    assert "_canary" in r.outputs["a"] and o3.canaries.get("a")


async def test_v18_no_op_rerun_keeps_accepted(tmp_path):
    o, _ = orch(tmp_path, {"a": Role()})
    await o.run_plan(Plan("t", "g", [Node("a", "a")]))
    o.set_state("t", "accepted", task_hash="h1")
    r = await o.run_plan(Plan("t", "g", [Node("a", "a")]))
    assert r.state == "accepted" and o.journal.state("t")["state"] == "accepted"


async def test_pick_node_timeout_is_infra_with_backoff(tmp_path):
    o, _ = orch(tmp_path, {"slow": Role(delay=5.0)})
    t0 = time.monotonic()
    r = await o.run_plan(Plan("t", "g", [Node("x", "slow", timeout_s=0.05, backoff_s=0.01)]))
    assert r.nodes["x"].status == "abandoned" and "timed out" in r.nodes["x"].findings[0] and time.monotonic() - t0 < 3


def test_pick_forge_journal_survives_a_torn_tail(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text(json.dumps({"event": "state", "task_id": "t", "state": "draft"}) + '\n{"event": "node", "task_id"')
    j = Journal(p)
    j.append({"event": "state", "task_id": "t", "state": "constructed"})
    assert j.torn == 1 and Journal(p).state("t")["state"] == "constructed"


# ---------------------------------------------------------------- acceptance subset and red team (V7-V11)


@pytest.fixture(scope="module")
def toy():
    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    return {t.task_id.split("/")[-1]: (t, res) for t, res in col}


def _clone(t, **kw):
    from crucible_pack_toy.forge.task import ToyForgedTask

    spec = t.spec.model_copy(update={k: v for k, v in kw.items() if k == "instruction"})
    return ToyForgedTask(spec, kw.get("checks", t.cs), t.oracle, t.negatives, pack_root=t.pack_root, world_file=t.world_file, scenario=t.sc, canary=t.canary)


async def test_v8_v9_v11_contract_on_the_toy_tasks(toy):
    from crucible.accept.contract import a3_negatives, a5_leak, a7_coverage, a8_grounding, accept
    from crucible.forge.schema import CheckSpec

    t1, resolve = toy["t1_duplicate"]
    assert (await accept(t1, resolve=resolve)).ok  # no false positives on a sound task
    assert a5_leak(_clone(t1, instruction=t1.instruction + "\nReference: " + json.dumps(t1.oracle))).status == "fail"  # V9
    assert "tickets.sql" in t1.agent_visible()  # the ticket rows the agent reads (V9)
    p = t1.properties[0]
    cut = t1.instruction.replace(p.text, p.text[:40])
    assert a7_coverage(_clone(t1, instruction=cut)).status == "fail"  # V8: the whole sentence
    bad = t1.cs.model_copy(update={"checks": [c.model_copy(update={"provenance": "x"}) for c in t1.cs.checks]})
    assert a8_grounding(_clone(t1, checks=bad), resolve).status == "fail"  # V11
    note = next(c for c in t1.cs.checks if c.id.startswith("outcome/note"))
    fin7 = note.model_copy(update={"id": "outcome/note_cites_fin7", "sql": "SELECT COUNT(*) FROM notes WHERE body LIKE '%FIN-7%'",
                                   "expect": {"kind": "between", "lo": 1, "hi": 1000}})
    internals = t1.cs.model_copy(update={"checks": [*t1.cs.checks, CheckSpec.model_validate(fin7.model_dump())]})
    assert (await a3_negatives(_clone(t1, checks=internals))).status == "fail"  # V8: a check on oracle wording


async def test_v7_a16_needs_measured_evidence_and_searches_past_memory(toy, tmp_path):
    from crucible.accept.checks.redteam import a16_red_team
    from crucible.accept.model import Status
    from crucible.accept.policy import Policy
    from crucible.forge.hacker import MutationHacker, classify
    from crucible.forge.specialists import FamilyMemory, SpecialistHacker

    assert classify(1.0, None) == "undecided"
    t2, _ = toy["t2_approval"]

    class Ctx:
        policy = Policy()

        async def grade(self, t, key, sol, run=0):
            rep, fp = await t.grade(sol)
            rep.infra = {"status": "masked", "failure_kind": "probe"}
            return rep, fp

    assert (await a16_red_team(t2, Ctx())).status == Status.ERROR  # every grade masked: no evidence
    weak = t2.cs.model_copy(update={"checks": [c for c in t2.cs.checks if not c.id.startswith("outcome/note")]})
    w2 = _clone(t2, checks=weak)
    w2.audit_floor = list(t2.cs.checks)
    mem = FamilyMemory(tmp_path / "mem.jsonl")
    mem.remember("toy/helpdesk", "t_old", ["skip_interactions"])
    sp = SpecialistHacker("toy/helpdesk", mem, fallback=MutationHacker())
    w2.hacker = lambda: sp

    class Real:
        policy = Policy()

        async def grade(self, t, key, sol, run=0):
            return await t.grade(sol)

    out = await a16_red_team(w2, Real())
    assert out.status == Status.FAIL and sp.episodes == 1 and sp.source == "memory+fallback"


def test_v10_role_runner_scans_json_escaped_values(tmp_path):
    from crucible.interop.harbor.role_runner import LeakRefused, _leaks, assemble_role_task

    line = 'refund(order="O-7213", amount=97.31)'
    assert _leaks(json.dumps({"o": line}, indent=1), [line])
    with pytest.raises(LeakRefused):
        assemble_role_task("tw", {"inputs/x.json": json.dumps({"o": line}, indent=1)}, "write checks", Out, tmp_path / "w", forbidden=[line])


def test_v19_failed_trial_keeps_its_cost():
    from crucible.interop.harbor.role_runner import RoleTrialFailed

    assert RoleTrialFailed("x", cost_usd=0.4).cost_usd == 0.4


# ---------------------------------------------------------------- composition, coding traps (V12-V14, V20, V22)


def test_v14_cross_checks_need_a_stated_property(toy):
    from crucible.forge.compose import Parts, fuse
    from crucible.forge.schema import CheckSpec

    t1, _ = toy["t1_duplicate"]
    cross = CheckSpec(id="consistency/x", gate="consistency", gate_kind="gate", kind="cross", states=["x"], provenance="kb:x")
    with pytest.raises(ValueError, match="no cross property"):
        fuse(Parts.of(t1), Parts.of(toy["t3_in_transit"][0]), task_id="f", cross=[cross])


def test_v12_v13_code_only_c2_and_span_gates():
    from crucible_pack_coding.forge.derive import _CodeOnly, trap_property

    class FS:
        def read(self, p):
            return '"""This module defines the schemas."""\nx = 1\n' if p.endswith(".py") else "// defines\nint x;\n"

    assert "defines" not in _CodeOnly(FS()).read("m.py") and "defines" not in _CodeOnly(FS()).read("m.cu")
    prop = trap_property("# T\n\n## Requirements\n\n- Keep the ncv results bit-exact.\n- Be fast.\n\n## Grading\n", "ncv")
    assert prop.text == "Keep the ncv results bit-exact."
    rep = json.loads((Path(__file__).parents[2] / "docs" / "modules" / "reports" / "m8_coding_derive.json").read_text())
    assert not [c for v in rep["spans"].values() for c in v["checks"] if c.startswith("trap/")]  # C2 are Φ milestones, not gates
    assert all(v.get("instruction_states_gate") for v in rep["traps"].values() if "skipped" not in v)


def test_v20_stitch_rejects_a_span_that_owns_its_own_start():
    from crucible.forge.stitch import Span, stitch

    a = Span("a", start={"f.c": "BASE->START"}, oracle={"f.c": "START->ORACLE"}, gates=["g"])
    b = Span("b", start={}, oracle={"g.c": "x"}, gates=["g"])
    with pytest.raises(ValueError, match="pre-applies and owns"):
        stitch([a, b])


def test_v22_tighten_needs_a_reliable_frontier_failure():
    from crucible.forge.tighten import add_gate

    assert not add_gate("concurrency", [True] * 3, [True, False]).ok
    assert add_gate("concurrency", [True] * 3, [False, False, True]).ok


# ---------------------------------------------------------------- economy, swarm, curator, dedup (V15-V17, V21)


def test_v15_yieldbook_replays_resume_and_keeps_verdicts(tmp_path):
    from crucible.forge.economy import StopRules, YieldBook

    p = tmp_path / "book.jsonl"
    b = YieldBook(StopRules(min_attempts=3, min_yield=0.5), path=p)
    for i in range(3):
        b.record_construction(f"t{i}", "gen", state="rejected")
    assert b.paused("gen")
    b.resume("gen")
    assert not YieldBook(StopRules(min_attempts=3, min_yield=0.5), path=p).paused("gen")
    b.record_construction("ok", "gen2")
    b.record_acceptance("ok", True, hacks=1)
    b.record_construction("ok", "gen2", state="accepted")
    assert b.rows["ok"].accepted is True and b.rows["ok"].hacks == 1


async def test_v16_swarm_protocol_and_worker_failures(tmp_path):
    from crucible.forge.swarm import Cancelled, Contribute, Nominate, SwarmStage

    r = ForgeRepo(tmp_path / "repo", family="toy")
    outs = iter(["boom", "no-announce", "ignore-cancel", "ok"])

    def make(wid):
        async def run(ep):
            kind = next(outs)
            if kind == "boom":
                raise RuntimeError("worker crashed")
            if kind == "ignore-cancel":
                try:  # the second announcement collides with the first: a worker that swallows Cancelled
                    ep.announce("plan A", key="A")
                    ep.announce("plan A again", key="A")
                except Cancelled:
                    pass
                return Contribute({"public/plans/c.json": "{}"}, "plan", "c")
            if kind == "no-announce":
                return Contribute({"public/plans/b.json": "{}"}, "plan", "b")
            ep.announce("plan Z", key="Z")
            return Contribute({"public/plans/z.json": "{}"}, "plan", "z")

        return run

    res = await SwarmStage(r, "proposer", make, C=1, B=10, max_episodes=4).run()
    kinds = sorted(x["kind"] for x in res.rejected)
    assert kinds == ["error", "protocol", "protocol"] and len(res.commits) == 1 and not res.dashboard["in_flight"]
    own = res.commits[0]
    st = SwarmStage(r, "proposer", lambda wid: (lambda ep: _nom(own)), C=1, B=1, max_episodes=1)
    st.authors[own] = "proposer-1"
    res2 = await st.run()
    assert res2.nominations == {} and "own commit" in res2.rejected[0]["reasons"][0]
    assert Nominate


async def _nom(sha):
    from crucible.forge.swarm import Nominate

    return Nominate(sha)


def test_v17_curator_lineage_is_durable_and_transitive(tmp_path):
    from crucible.forge.curator import Curator
    from crucible.forge.dedup import Candidate
    from crucible.forge.ladder import LadderReport
    from crucible.forge.learnability import estimate

    def rep(**r):
        x = LadderReport()
        x.rates = {k: estimate([True] * a + [False] * (b - a), model=k, harness="t") for k, (a, b) in r.items()}
        return x

    p = tmp_path / "cur.jsonl"
    c = Curator(path=p)
    assert c.place("b1", rep(L3=(0, 16)), lineage=["seed:A"], prefer="bench").action == "bench"
    c2 = Curator(path=p)  # the next forge run
    assert c2.place("sib", rep(L2=(6, 16)), lineage=["seed:A"]).action == "hold"
    assert c2.place("child", rep(L2=(6, 16)), lineage=["b1"]).action == "hold"  # transitive: b1's lineage is seed:A
    hit = Curator(path=tmp_path / "x.jsonl", contamination=lambda tid, text: [{"source": "pretrain", "kind": "13-gram"}])
    d = hit.place("b2", rep(L3=(0, 16)), lineage=["seed:Z"], prefer="bench", candidate=Candidate("b2", instruction="refund ticket 7"))
    assert d.action == "hold" and d.steer == "decontaminate"
    assert Candidate


def test_v21_dedup_l1_reads_the_whole_instruction():
    from crucible.forge.dedup import Candidate, duplicate_of

    intro = "You are a helpdesk agent for an online store. Use the ticket tools. " * 3
    a = Candidate("a", instruction=intro + "Refund the duplicate charge on ticket 38.", footprint=["refunds"])
    b = Candidate("b", instruction=intro + "Escalate ticket 41 for a carrier trace, do not refund.", footprint=["refunds"])
    assert duplicate_of(b, [a]) is None
    assert duplicate_of(Candidate("c", instruction=a.instruction.upper(), footprint=["refunds"]), [a]) == ("a", "L1_fingerprint")


# ---------------------------------------------------------------- picks: masked outcomes


async def test_pick_masked_outcomes_are_not_failures():
    from crucible.forge.ladder import Rung, accept, run_ladder, steer
    from crucible.forge.learnability import BANDS, advise, estimate

    r = estimate([True, None, False, None], model="m", harness="h")
    assert (r.n, r.k, r.masked) == (2, 1, 2)
    allmasked = estimate([None] * 8, model="m", harness="h")
    assert advise(allmasked, BANDS["train"])[0].direction == "sample_more"

    async def masked(n):
        return [None] * n

    rep = await run_ladder([Rung("L2", masked)])
    assert rep.rates["L2"].n == 0 and steer(rep) == "keep" and not accept(rep)[0]
    assert RoleInfraError
