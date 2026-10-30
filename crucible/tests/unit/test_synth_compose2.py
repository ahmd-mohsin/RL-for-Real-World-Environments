"""compose-2 (DESIGN §7.7 item 18): undone values closed in the merge, certified parts before composing, per-part repair,
TWIN adjudication, the repairer's variant catalogue and the solver harness. Fixture Worlds; every model role is a fake."""

from __future__ import annotations

import json

import pytest

from crucible.synth.compose import CompositionError, VariantSpec, build_chain, compose
from crucible.synth.contract import EffectPattern, grade
from crucible.synth.falsify import MUT, FalsifyContext
from crucible.synth.pipeline import DomainEnv, RunConfig, Runner, RunRoles
from crucible.synth.program import Program, ref
from crucible.synth.roles import contract_doc, program_doc
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient, const, section, seq
from tests.unit.test_synth_compose import HD, S, after, mk


async def raise_lower(allowed_in: int = 2, back_to: str = "normal"):
    t1 = await mk("raise", "Set the priority of ticket T-1 to high.", [S("s1", "desk.set_priority", ticket_id="T-1", priority="high")],
                  world=HD, norm=fx.HD_N)
    ins2 = "Ticket T-1 was raised by mistake: set its priority back to normal." if back_to == "normal" else f"Set ticket T-1's priority to {back_to}."
    t2 = await mk("lower", ins2, [S("s1", "desk.get_ticket", ticket_id="T-1"), S("s2", "desk.set_priority", ticket_id="T-1", priority=back_to)],
                  world=after(t1, base=HD), norm=fx.HD_N)
    loose = EffectPattern(id="a_any", app="desk", coll="tickets", key="*", field="priority", why="priority edits are routine")
    if allowed_in == 1:
        t1 = t1.model_copy(update={"contract": t1.contract.model_copy(update={"allowed": [loose]})})
    else:
        t2 = t2.model_copy(update={"contract": t2.contract.model_copy(update={"allowed": [loose]})})
    return t1, t2, loose


# ---------------------------------------------------------------------------------------------------------- undone values

async def test_a_restored_value_is_closed_in_every_merged_pattern():
    """compose-1's restoration gap: part 1 raises T-1's priority, part 2 sets it back, and part 2's allowed pattern (any
    priority update) let a run without the restoring step pass. The merge now excludes the undone value."""
    t1, t2, loose = await raise_lower()
    c = await compose([t1, t2])
    assert c.comp.restored == ["desk.tickets[T-1].priority"]
    assert c.comp.closed == ["desk.tickets[T-1].priority (restored)"]
    a = next(p for p in c.task.contract.allowed if p.id == "t2.a_any")
    assert a.unless and a.unless[0].values == ["high"]
    assert grade(c.run, c.task.contract).passed
    ctx = FalsifyContext()
    r = await ctx.run(c.task, Program(steps=[s for s in c.task.oracle.steps if s.id != "t2_s2"]))
    v = grade(r, c.task.contract)
    assert not v.passed and any(x.startswith("FORBIDDEN_EFFECT:desk.tickets[T-1].priority") for x in v.reasons)
    # an unrelated priority change is still allowed: the exclusion is one value at one address
    other = Program(steps=[*c.task.oracle.steps, S("x", "desk.set_priority", ticket_id="T-3", priority="high")])
    assert grade(await ctx.run(c.task, other), c.task.contract).passed
    # no MUT mutant that leaves the undone value survives
    for p in await MUT().propose(c.task, ctx):
        o = await MUT().execute(p, c.task, ctx)
        r2 = await ctx.run(c.task, p.program)
        if r2.end is not None and (r2.end.entity("desk", "tickets", "T-1") or {}).get("priority") == "high":
            assert not o.passed, p.site
    # a pattern without exclusions dumps as before (no "unless" key), so every other contract keeps its digest
    assert "unless" not in loose.model_dump(mode="json") and "unless" in a.model_dump(mode="json")
    await ctx.close()


async def test_a_superseded_value_is_closed_where_the_net_change_differs():
    t1, t2, _ = await raise_lower(allowed_in=1, back_to="low")
    c = await compose([t1, t2])
    assert c.comp.closed == ["desk.tickets[T-1].priority"]
    ctx = FalsifyContext()
    r = await ctx.run(c.task, Program(steps=[s for s in c.task.oracle.steps if s.id != "t2_s2"]))
    assert not grade(r, c.task.contract).passed  # left at the superseded "high"
    await ctx.close()




# ---------------------------------------------------------------------------------------------------------- certified parts

T_PRIO = "Set the priority of ticket T-4 to high."
T_RES = "Resolve ticket T-3 with the id of the knowledge base article on refund policy as the resolution."


class Book:
    """What a scripted solver and judge know: per instruction, the oracle's concrete calls (tool, args, read-only)."""

    def __init__(self):
        self.calls: dict[str, list[tuple[str, dict, bool]]] = {}
        self.answers: dict[str, object] = {}

    async def learn(self, task):
        ctx = FalsifyContext()
        try:
            r = await ctx.oracle(task)
        finally:
            await ctx.close()
        self.calls[task.instruction] = [(c.tool, dict(c.args), c.read_only) for c in r.calls]
        self.answers[task.instruction] = r.answer

    def agent(self):
        def f(msgs, _n):
            ins, hist = section(msgs, "instruction"), section(msgs, "history")
            calls = self.calls.get(ins, [])
            done = 0 if isinstance(hist, str) else sum(1 for h in hist if "call" in h)
            if done < len(calls):
                return {"call": {"tool": calls[done][0], "args": calls[done][1]}}
            return {"done": True, "final": self.answers.get(ins)}
        return f

    def judge(self):
        def f(msgs, _n):
            task, tr = section(msgs, "task"), section(msgs, "trace")
            want = sorted(json.dumps([t, a], sort_keys=True) for t, a, ro in self.calls.get(task, []) if not ro)
            got = sorted(json.dumps([x["tool"], x["args"]], sort_keys=True) for x in (tr if isinstance(tr, list) else [])
                         if "error" not in x and x["tool"] in ("desk.set_priority", "desk.resolve_ticket", "desk.admin_update", "mail.send", "desk.claim"))
            return {"valid": bool(want) and want == got, "confidence": 0.95, "why": ""}
        return f


def part_roles(book: Book, repairer=None) -> RunRoles:
    return RunRoles(proposer=FakeClient("kimi", const({"tasks": []})), oracle_programmer=FakeClient("deepseek", const({"steps": []})),
                    contract_writer=FakeClient("glm", const({"edits": []})), repairer=repairer or FakeClient("deepseek", const({"give_up": True})),
                    alt=[("alt-n", FakeClient("nemotron", book.agent())), ("alt-o", FakeClient("openai", book.agent()))],
                    atk=[("atk-n", FakeClient("nemotron", seq({"done": True, "final": None})))],
                    panel=[("p-n", FakeClient("nemotron", book.judge())), ("p-o", FakeClient("openai", book.judge()))])


def hd_env() -> DomainEnv:
    return DomainEnv(name="desk-compose", description="An IT helpdesk.", seeds=(0,), normalizers=fx.HD_N, world=lambda s: HD)


async def two_parts(book: Book, *, lax_second: bool = False):
    p1 = await mk("prio", T_PRIO, [S("s1", "desk.set_priority", ticket_id="T-4", priority="high")], world=HD, norm=fx.HD_N)
    p2 = await mk("res", T_RES, [S("s1", "desk.search_kb", query="refund policy"), S("s2", "desk.resolve_ticket", ticket_id="T-3", resolution=ref("s1.articles.0.id"))],
                  world=after(p1, base=HD), norm=fx.HD_N)
    p1, p2 = p1.model_copy(update={"domain": "desk-compose"}), p2.model_copy(update={"domain": "desk-compose"})
    if lax_second:  # the resolution value is not checked: a wrong article passes (MUT finds it)
        req = [r.model_copy(update={"value": None}) if r.field == "resolution" else r for r in p2.contract.required]
        p2 = p2.model_copy(update={"contract": p2.contract.model_copy(update={"required": req})})
    for t in (p1, p2):
        await book.learn(t)
    return p1, p2


async def test_composition_uses_only_parts_certified_on_their_own_world_state(tmp_path):
    from crucible.synth.parts import CertBook, PartCert

    book = Book()
    p1, p2 = await two_parts(book)
    certs = CertBook(tmp_path / "certs.json")
    with pytest.raises(CompositionError, match="certified or provisional"):
        await compose([p1, p2], certs=certs)
    certs.put(PartCert(task_id=p1.id, version=1, digest=p1.digest(), status="certified"))
    certs.put(PartCert(task_id=p2.id, version=1, digest=p2.digest(), status="rejected"))
    with pytest.raises(CompositionError, match="part 2"):
        await compose([p1, p2], certs=certs)
    certs.put(PartCert(task_id=p2.id, version=1, digest=p2.digest(), status="provisional"))
    c = await compose([p1, p2], certs=CertBook(tmp_path / "certs.json"))  # read back from disk
    assert [s.status for s in c.comp.segments] == ["certified", "provisional"]
    # a certificate is for one exact version: another digest is not certified
    other = p2.model_copy(update={"instruction": p2.instruction + " Thanks."})
    with pytest.raises(CompositionError, match="no certificate for this version"):
        await compose([p1, other], certs=certs)


async def test_certify_links_certifies_in_order_and_cuts_the_chain_at_a_rejection(tmp_path):
    from crucible.synth.parts import CertBook, certify_links

    book = Book()
    p1, p2 = await two_parts(book, lax_second=True)
    runner = Runner(hd_env(), part_roles(book), RunConfig(max_rounds=2), tmp_path / "parts", tmp_path / "arch", run_id="p")
    await runner.survey()
    certs, log = CertBook(tmp_path / "certs.json"), []
    ctx = FalsifyContext()
    got = await certify_links([p1, p2], runner, certs, ctx, log=log)
    await ctx.close()
    assert [x["status"] for x in log] == ["provisional", "rejected"]  # the lax link fails MUT (a wrong article passes)
    assert [t.id for t in got] == [p1.id] and certs.admitted(p1) and not certs.admitted(p2)
    assert any(r.startswith("MUT") for r in log[1]["reasons"])


async def test_build_chain_certifies_each_link_before_it_joins(tmp_path):
    from crucible.synth.parts import CertBook

    book = Book()
    p1, p2 = await two_parts(book)
    by_ins = {t.instruction: t for t in (p1, p2)}

    def proposer(msgs, n):
        dom = section(msgs, "domain")
        return {"tasks": [{"skill": "x", "instruction": T_RES if "Earlier work" in dom else T_PRIO, "derive": []}]}

    def programmer(msgs, n):
        return program_doc(by_ins[section(msgs, "task")].oracle)

    rl = part_roles(book)
    rl.proposer, rl.oracle_programmer = FakeClient("deepseek", proposer), FakeClient("deepseek", programmer)
    certs = CertBook(tmp_path / "certs.json")
    got, log = await build_chain(hd_env(), rl, length=2, run_dir=tmp_path / "chain", archive_root=tmp_path / "arch", seed=0, candidates=1,
                                 cfg=RunConfig(max_rounds=2), certify=True, book=certs, retries=0)
    assert [t.instruction for t in got] == [T_PRIO, T_RES]
    assert all(certs.admitted(t) for t in got)
    # link ids carry the run directory: a second run with the same chain name never shares an archive key
    assert all(f"{tmp_path.name}-chain-link0" in t.id for t in got)
    # a resumed chain starts after its certified links: one new link proposed, on the World after them
    calls = len(rl.proposer.calls)
    # (the resuming run's own env names another World, as a fresh work dir does: the new links still use the start's base)
    other = DomainEnv(name="desk-compose", description="An IT helpdesk.", seeds=(0,), normalizers=fx.HD_N,
                      world=lambda s: HD.model_copy(update={"params": {"admin": False}}))
    got2, log2 = await build_chain(other, rl, length=2, run_dir=tmp_path / "chain2", archive_root=tmp_path / "arch", seed=0, candidates=1,
                                   cfg=RunConfig(max_rounds=2), certify=True, book=certs, retries=0, start=got[:1])
    assert got2[0].id == got[0].id and len(got2) == 2 and [c.link for c in log2] == [2] and len(rl.proposer.calls) == calls + 1
    with pytest.raises(CompositionError, match="certified"):
        await build_chain(hd_env(), rl, length=2, run_dir=tmp_path / "chain3", archive_root=tmp_path / "arch", seed=0, certify=True,
                          book=CertBook(tmp_path / "empty.json"), start=got[:1])
    assert all(row.get("cert", {}).get("status") in ("certified", "provisional") for ch in log for row in ch.candidates if "cert" in row)
    assert (tmp_path / "chain" / "link01" / "calibration.json").exists()  # the panel is calibrated once, on the first link
    assert not (tmp_path / "chain" / "link02" / "calibration.json").exists()
    c = await compose(got, certs=certs)
    assert len(c.comp.segments) == 2


async def test_composite_counterexamples_are_attributed_and_translated_to_their_parts():
    from crucible.synth.falsify import Counterexample, Finding, Outcome, Probe
    from crucible.synth.parts import attribute, part_slice, reproduces, to_part

    book = Book()
    p1, p2 = await two_parts(book, lax_second=True)
    c = await compose([p1, p2])
    wrong = Program(steps=[*c.task.oracle.steps[:2], S("t2_s2", "desk.resolve_ticket", ticket_id="T-3", resolution="KB-1")])
    f = Finding(cls="MUT", op="PerturbValue", site="t2_s2.resolution=KB-1", label="confirmed", defect="D3",
                probe=Probe(cls="MUT", op="PerturbValue", site="t2_s2.resolution=KB-1", program=wrong),
                outcome=Outcome(passed=True, reasons=[]))
    cx = Counterexample.of(c.task, f)
    at = await attribute(cx, c.task, c.comp)
    assert at.parts == [1] and at.by == "program"
    sl = part_slice(wrong, c.task.oracle, c.comp, 1)
    assert [s.id for s in sl.steps] == ["s1", "s2"] and sl.steps[1].args["resolution"].v == "KB-1"
    pcx = to_part(cx, c.task, c.comp, 1, p2)
    assert pcx.task_id == p2.id and pcx.site == "s2.resolution=KB-1" and pcx.probe.data["from_composite"] == cx.id
    ctx = FalsifyContext()
    assert await reproduces(pcx, p2, ctx)  # the part's own contract also passes the wrong article: a part-level defect
    # with the exact part contract the slice fails: that counterexample would be the merge's, not the part's
    p2x = await mk("res", T_RES, list(p2.oracle.steps), world=p2.world, norm=fx.HD_N)
    assert not await reproduces(to_part(cx, c.task, c.comp, 1, p2x), p2x, ctx)
    # reason codes name their parts
    cx2 = cx.model_copy(update={"probe": cx.probe.model_copy(update={"program": None}), "site": "x", "reasons_seen": ["MISSING_REQUIRED:t1.r0"]})
    assert (await attribute(cx2, c.task, c.comp)).parts == [0]
    await ctx.close()


async def test_per_part_repair_repairs_only_the_owner_part_and_recomposes(tmp_path):
    from crucible.synth.compose import certify_composed
    from crucible.synth.parts import CertBook, certify_part

    book = Book()
    p1, p2 = await two_parts(book, lax_second=True)
    fixed = await mk("res", T_RES, list(p2.oracle.steps), world=p2.world, norm=fx.HD_N)
    seen: list[str] = []

    def repair(msgs, _n):
        seen.append(msgs[1]["content"])
        return {"contract": contract_doc(fixed.contract)}

    rl = part_roles(book, repairer=FakeClient("deepseek", repair))
    runner = Runner(hd_env(), rl, RunConfig(max_rounds=2), tmp_path / "parts", tmp_path / "arch", run_id="p")
    await runner.survey()
    certs = CertBook(tmp_path / "certs.json")
    from crucible.synth.parts import PartCert

    _, c1 = await certify_part(runner, p1, certs, n=1)
    assert c1.status in ("certified", "provisional")
    certs.put(PartCert(task_id=p2.id, version=1, digest=p2.digest(), status="provisional", note="test: admitted as if certified"))
    comp = await compose([p1, p2], certs=certs, task_id="desk-compose/c")
    await book.learn(comp.task)
    rep = await certify_composed([comp], hd_env(), rl, run_dir=tmp_path / "cert", archive_root=tmp_path / "arch", run_id="c",
                                 cfg=RunConfig(max_rounds=2), parts={comp.task.id: [p1, p2]}, book=certs, part_runner=runner, twin=False)
    rec = rep.tasks[0]
    assert rec.repairs_accepted >= 1, (rec.status, rec.reasons)
    rows = [json.loads(x) for x in (tmp_path / "cert" / "tasks" / "t01" / "part_repairs.jsonl").read_text().splitlines()]
    assert rows[0]["result"] == "candidate" and [p["part"] for p in rows[0]["parts"]] == [2]
    # the repairer saw part 2 in its own frame: its program, never the merged one (no t1_/t2_ step names)
    assert seen and all("t1_" not in s and "t2_" not in s for s in seen)
    final = json.loads((tmp_path / "cert" / "tasks" / "t01" / "composed-final.json").read_text())
    assert final["task"]["id"] == comp.task.id and final["task"]["version"] >= 2
    assert final["parts"][0]["version"] == 1 and final["parts"][1]["version"] >= 2  # part 1 untouched, part 2 repaired


async def test_twin_suspects_are_decided_by_the_panel_on_the_twin_task():
    from crucible.synth.falsify import Outcome, Probe
    from crucible.synth.falsify.agents import Panel, PanelMember
    from crucible.synth.falsify.twinjudge import TwinAdjudicated, calibrate_twin_panel, twin_task

    t = fx.t1_hold()
    seen: list[str] = []

    def judge(msgs, _n):
        seen.append(section(msgs, "task"))
        return {"valid": False, "confidence": 0.95, "why": "acts on the invoice the instruction no longer names"}

    class Stub:
        name = "TWIN"

        async def applies(self, task, ctx):
            return None

        async def propose(self, task, ctx):
            return [Probe(cls="TWIN", op="twin", site="literal:x", instruction="Hold invoice INV-1004 instead.", expect="property",
                          data={"sigma": [["INV-1003", "INV-1004"]], "reaches": False})]

        async def execute(self, probe, task, ctx):
            return Outcome(data={"oracle_passes_twin": True, "twin_passes_twin": False, "g1": [], "g2": []})

        def label(self, probe, outcome):
            return "suspect", "D9"

        async def replay(self, cx, task, ctx):
            return False

    panel = Panel(members=[PanelMember("a", FakeClient("nemotron", judge)), PanelMember("b", FakeClient("openai", judge))], lam=0.7)
    ctx = FalsifyContext()
    w = TwinAdjudicated(Stub(), panel)
    from crucible.synth.falsify import falsify

    r = await falsify(t, w, ctx)
    assert r.c == 1 and r.confirmed[0].defect == "D9"  # π* passes C′ but is no solution of ι′: the contract ignores the literal
    assert set(seen) == {"Hold invoice INV-1004 instead."} and len(seen) == 2 and panel.log[0]["judged"] == "oracle"
    assert twin_task(t, (await Stub().propose(t, ctx))[0]).instruction.endswith("instead.")
    # strict calibration: no anchors, or too few, decides nothing
    cal = await calibrate_twin_panel(panel, [])
    assert not cal.calibrated and panel.lam > 1
    await ctx.close()


async def test_twin_anchors_carry_executed_truth_from_clean_twins():
    from crucible.synth.falsify.agents import Panel, PanelMember
    from crucible.synth.falsify.twinjudge import calibrate_twin_panel, twin_anchors

    ctx = FalsifyContext()
    decided = fx.t1_hold()
    assert decided.intent.derive  # the hold reason is derived from the invoice's records
    # a derived decision does not carry to the sibling: π*′ is no twin solution, so only the invalid anchors are kept
    assert {a.kind for a in await twin_anchors([decided], ctx)} == {"original_on_twin", "noop"}
    anchors = await twin_anchors([decided.model_copy(update={"intent": decided.intent.model_copy(update={"derive": []})})], ctx)
    await ctx.close()
    kinds = {a.kind: a.valid for a in anchors}
    assert kinds == {"twin_solution": True, "original_on_twin": False, "noop": False}
    assert all(a.task.instruction != fx.t1_hold().instruction for a in anchors)  # judged against the twin instruction

    def judge(msgs, _n):  # a judge that is right on every anchor: a twin solution writes, π* on the twin task is wrong
        tr = section(msgs, "trace")
        task = section(msgs, "task")
        ok = isinstance(tr, list) and any(x["tool"] == "ap.hold_invoice" and x["args"].get("invoice_id") is not None for x in tr) and \
            all(x["args"]["number"] in task for x in tr if x["tool"] == "ap.find_invoice")
        return {"valid": ok, "confidence": 0.95, "why": ""}

    panel = Panel(members=[PanelMember("a", FakeClient("nemotron", judge)), PanelMember("b", FakeClient("openai", judge))], lam=0.7)
    cal = await calibrate_twin_panel(panel, anchors, min_decided=len(anchors))
    assert cal.calibrated and panel.lam <= 0.95
    cal = await calibrate_twin_panel(panel, anchors, min_decided=len(anchors) + 1)  # too few decided anchors: decides nothing
    assert not cal.calibrated and panel.lam > 1


async def test_the_repairer_sees_the_variant_worlds_tool_catalogue(tmp_path):
    from crucible.synth.compose import ActorSpec, PageSpec

    seen: list[str] = []

    def repair(msgs, _n):
        seen.append(section(msgs, "tools"))
        return {"give_up": True}

    t = fx.t1_hold()
    spec = VariantSpec(base=t.world, actors=[ActorSpec(name="requester", facts={"number": "INV-1003"})],
                       paginate={"ap.list_invoices": PageSpec(field="invoices", size=2)})
    t = t.model_copy(update={"world": spec.ref()})
    rl = part_roles(Book(), repairer=FakeClient("deepseek", repair))
    env = DomainEnv(name="synth-test", description="AP.", seeds=(0,), normalizers=fx.AP_N, world=lambda s: t.world.model_copy() if False else spec.base)
    runner = Runner(env, rl, RunConfig(), tmp_path / "r", tmp_path / "a", run_id="r")
    await runner.survey()
    from crucible.synth.pipeline import TaskRecord

    ctx = FalsifyContext()
    out = await runner.repairer(TaskRecord(n=1, id=t.id, seed=0), ctx)(t, [])
    await ctx.close()
    assert out is None and seen
    names = {x["name"]: x for x in seen[0]}
    assert "actors.ask" in names and "page" in json.dumps(names["ap.list_invoices"])


# ---------------------------------------------------------------------------------------------------------- solvers

async def test_harness_v3_keeps_a_scratchpad_checks_progress_and_windows_the_history():
    from crucible.synth.falsify.agents import PROGRESS_CHECK, AgentBudget, AgentView, llm_agent
    from crucible.synth.roles import ALT_SOLVER, ALT_SOLVER_V3, RoleLeak

    seen: list[str] = []

    def agent(msgs, n):
        seen.append(msgs[1]["content"])
        if n < 30:
            return {"call": {"tool": "desk.get_ticket", "args": {"ticket_id": "T-1"}}, **({"notes": "1. read T-1 (plan)"} if n == 0 else {})}
        return {"done": True, "final": "ok"}

    ctx = FalsifyContext()
    t = fx.t4_priority()
    w = await ctx.world(t)
    await w.reset(0)
    run = llm_agent(FakeClient("nemotron", agent), ALT_SOLVER_V3, budget=AgentBudget(max_calls=40, max_turns=50), progress_check=True, window=8, block=4,
                    short_chars=20)
    out = await run(AgentView(instruction=t.instruction, tools=await w.tools()), w)
    await ctx.close()
    assert out == "ok"
    assert "1. read T-1 (plan)" in seen[5] and "## notes" in seen[5]  # the scratchpad comes back every turn
    assert PROGRESS_CHECK in seen[-1] and sum(PROGRESS_CHECK in s for s in seen) == 1  # one check, then the finish stands
    assert "shortened" in seen[-1]  # old observations are cut; the recent ones are whole
    # the cut moves in blocks: consecutive prompts inside a block share the shortened prefix
    assert seen[20].count("shortened") == seen[21].count("shortened")
    # the solver role still never sees the contract, the oracle or the intent card
    for k in ("contract", "oracle", "intent"):
        with pytest.raises(RoleLeak):
            ALT_SOLVER_V3.render(instruction="x", tools=[], history=[], **{k: "y"})
    assert ALT_SOLVER_V3.never == ALT_SOLVER.never


async def test_agent_turns_in_other_shapes_are_calls_and_bare_arguments_are_no_finish():
    from crucible.synth.falsify.agents import AgentBudget, AgentView, llm_agent
    from crucible.synth.roles import AgentTurn

    t = AgentTurn.model_validate({"name": "desk.get_ticket", "args": {"ticket_id": "T-1"}})  # Inkling's shape (compose-2)
    assert t.call == {"tool": "desk.get_ticket", "args": {"ticket_id": "T-1"}} and not t.is_final()
    assert AgentTurn.model_validate({"function": {"name": "x.y", "arguments": {}}}).call == {"tool": "x.y", "args": {}}
    replies = [{"ticket_id": "T-1"}, {"call": {"tool": "desk.get_ticket", "args": {"ticket_id": "T-1"}}}, {"done": True, "final": 1}]
    seen: list[str] = []

    def f(msgs, n):
        seen.append(msgs[1]["content"])
        return replies[min(n, 2)]

    ctx = FalsifyContext()
    task = fx.t4_priority()
    w = await ctx.world(task)
    await w.reset(0)
    out = await llm_agent(FakeClient("inkling", f), budget=AgentBudget(max_calls=5, max_turns=6))(AgentView(instruction="x", tools=await w.tools()), w)
    await ctx.close()
    assert out == 1 and "neither a tool call" in seen[1]  # the bare-argument turn was answered, not taken as the finish


async def test_solver_pool_compares_configurations_at_equal_spend(tmp_path):
    from crucible.synth.solvers import SolverConfig, evaluate, report_md, summarize

    book = Book()
    p1, _ = await two_parts(book)
    spend: dict[str, float] = {}

    class Costed(FakeClient):
        def __init__(self, fam, script, usd):
            super().__init__(fam, script)
            self.usd = usd

        async def complete_json(self, messages, **kw):
            spend[kw.get("tag", "")] = spend.get(kw.get("tag", ""), 0.0) + self.usd
            return await super().complete_json(messages, **kw)

    def flaky():  # fails its first attempt (finishes at once), passes the second
        good = book.agent()

        def f(msgs, n):
            return {"done": True, "final": None} if any("~1" in x for x in tags[-1:]) else good(msgs, n)
        return f

    tags: list[str] = []
    cheap = Costed("openai", flaky(), 0.001)
    orig = cheap.complete_json

    async def tagged(messages, **kw):
        tags.append(kw.get("tag", ""))
        return await orig(messages, **kw)

    cheap.complete_json = tagged  # type: ignore[method-assign]
    dear = Costed("nemotron", book.agent(), 0.05)
    cfgs = [SolverConfig("cheap", cheap, harness="v2"), SolverConfig("dear", dear, harness="v3")]
    rows = await evaluate([("single", p1, 1)], cfgs, budget_usd=0.02, max_attempts=3, cost_of=lambda tag: sum(v for k, v in spend.items() if k.startswith(tag)),
                          out=tmp_path)
    by = {r["config"]: r for r in summarize(rows, 0.02)}
    assert by["cheap"]["attempts"] == 2 and by["cheap"]["pass_first"] == 0 and by["cheap"]["solved_at_budget"] == 1  # a second try fits the budget
    assert by["dear"]["attempts"] == 1 and by["dear"]["pass_first"] == 1
    assert len({r.k for r in rows if r.config == "cheap"}) == 2 and all("~" in t for t in tags)  # one tag per attempt: seeds differ
    assert (tmp_path / "attempts.jsonl").exists() and "| single | cheap |" in report_md(summarize(rows, 0.02), "pool")


async def test_a_confirmed_oracle_or_leak_finding_rejects(tmp_path):
    """compose-2: ORACLE (item 21) and LEAK are in the admission policy, so a wrong reference decision rejects."""
    from crucible.synth.archive import Archive
    from crucible.synth.cegis import refine
    from crucible.synth.falsify import cheap
    from crucible.synth.falsify.oracle import ORACLE
    from crucible.synth.pipeline import admission_policy

    assert {"ORACLE", "LEAK"} <= set(admission_policy().deterministic)

    async def no_repair(task, cxs):
        return None

    ctx = FalsifyContext()
    rep = await refine(fx.t1_hold(), repair=no_repair, ctx=ctx, archive=Archive(tmp_path), falsifiers=[*cheap(), ORACLE([lambda t: "wrong"])],
                       policy=admission_policy(), max_rounds=1)
    await ctx.close()
    assert rep.status == "rejected" and "ORACLE:confirmed=1" in rep.reasons
