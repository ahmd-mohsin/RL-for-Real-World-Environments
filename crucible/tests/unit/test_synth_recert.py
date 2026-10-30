"""DESIGN §7.7 item 22: seeds by tag are the default for every sampled call path, and the stored pool is re-certified
under the fixed framework without editing a task (``synth/recert.py``). Scripted chat clients only: no model is called."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.synth_wire import SynthWire, Usage, WireClient, tag_seed
from crucible.synth.archive import Archive
from crucible.synth.certificate import certify
from crucible.synth.falsify import ALT, MUT, FalsifyContext, cheap, falsify
from crucible.synth.falsify.agents import AgentBudget, AgentSpec, llm_alt, run_agent
from crucible.synth.falsify.base import Counterexample, Finding, Outcome, Probe
from crucible.synth.recert import (
    Item,
    ManifestRow,
    OldCert,
    Source,
    attribute,
    beside,
    certify_paid,
    collect,
    copy_archive,
    det_job,
    det_verdict,
    load_manifest,
    paid_rows,
    status_of,
    work_world,
    write_new,
)
from crucible.synth.task import Task, WorldRef
from tests.unit import synth_fixtures as fx


class _R:
    def __init__(self, text: str):
        self.text, self.cost_usd, self.raw_finish = text, 0.0, "stop"


class SeedChat:
    """An LLMClient-shaped chat client that records every call's seed and finishes at once (after the agent's nudge)."""

    def __init__(self, log: list[tuple[str, int]], name: str = ""):
        self.log, self.name = log, name

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> _R:
        self.log.append((self.name, kw["seed"]))
        return _R('{"done": true, "final": null}')


# ---------------------------------------------------------------------------------------------------------- seeds by default

async def test_k_attempts_of_one_role_on_one_task_get_k_seeds(tmp_path: Path) -> None:
    """The bug: every model call used seed 0, so k ALT attempts (and k rounds of one agent, and Ultra x2) were copies."""
    log: list[tuple[str, int]] = []
    w = SynthWire(tmp_path / "run-a", cap_usd=1.0, make=lambda model, role: SeedChat(log))  # no seed argument: the default
    assert w.seed_by_tag and all(c.seed_by_tag for c in w.clients("alt_solver"))
    (n0, c0), (n1, c1) = w.run_roles().alt[:2]  # Nemotron-3-Ultra twice (SYNTH_ROLES)
    t, ctx = fx.t1_hold(), FalsifyContext()
    a0 = llm_alt(n0, c0, tag=f"ALT:{t.id}#{n0}")
    for _ in range(3):  # one runner, three attempts on one unchanged instruction: CEGIS rounds, the final pass
        await run_agent(a0, t, ctx)
    await run_agent(llm_alt(n1, c1, tag=f"ALT:{t.id}#{n1}"), t, ctx)
    seeds = [x for _, x in log]  # every turn of one attempt shares the attempt's tag, so one seed per attempt
    assert len(seeds) == 8 and len(set(seeds)) == 4, seeds
    # two clients of one model on the very same tag still draw two streams (the client salt)
    log.clear()
    await c0.complete_json([], tag="ALT:same")
    await c1.complete_json([], tag="ALT:same")
    assert log[0][1] != log[1][1]
    await ctx.close()


async def test_seeds_are_deterministic_per_run_and_independent_across_runs(tmp_path: Path) -> None:
    async def seeds(run: str) -> list[int]:
        log: list[tuple[str, int]] = []
        w = SynthWire(tmp_path / run, cap_usd=1.0, make=lambda model, role: SeedChat(log))
        for tag in ("ALT:t#a", "ALT:t#a", "ATK:t#b", "adjudicate:t"):
            await w.client("alt_solver").complete_json([], tag=tag)
        return [s for _, s in log]

    s1, s2, s3 = await seeds("run-1"), await seeds("run-1"), await seeds("run-2")
    assert s1 == s2  # the same tag in the same run gives the same seed (a resumed run redraws its own samples)
    assert s1[0] == s1[1] and len({s1[0], s1[2], s1[3]}) == 3
    assert not set(s1) & set(s3)  # a re-certification run does not redraw the stored run's samples
    assert tag_seed("student:t@1:0") == WireClient(None, role="r", model="m", family="f", usage=Usage())._seed("student:t@1:0", 0)  # item 19's formula
    assert WireClient(None, role="r", model="m", family="f", usage=Usage(), seed_by_tag=False)._seed("x", 2) == 2  # --same-seed


async def test_role_retries_and_repairs_draw_new_seeds(tmp_path: Path) -> None:
    """``ask``'s retries after the client's JSON retries ran out, and a repair asked again on an unchanged version, used
    one tag: the same seed, the same reply."""
    from crucible.synth.pipeline import RunConfig, Runner, TaskRecord
    from crucible.synth.roles import ABSTRACTOR, RoleFailed, ask
    from tests.unit.test_synth_pipeline import env, roles

    log: list[tuple[str, int]] = []

    class Junk(SeedChat):
        async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> _R:
            self.log.append(("", kw["seed"]))
            return _R("no json here")

    c = WireClient(Junk(log), role="abstractor", model="m", family="openai", usage=Usage(), salt="run")
    with pytest.raises(RoleFailed):
        await ask(c, ABSTRACTOR, tag="abstract:run", retries=1, clusters=[], defects="")
    assert len(log) == 6 and len({s for _, s in log}) == 6
    rl = roles()
    r = Runner(env(), rl, RunConfig(), tmp_path / "run", tmp_path / "arch", run_id="r")
    ctx = FalsifyContext()
    rep = r.default_repairer(TaskRecord(n=1, id="x", seed=0), ctx)
    t = fx.t3_resolve_email()
    await rep(t, [])
    await rep(t, [])
    assert len(set(rl.repairer.tags)) == 2 and all(x.startswith(f"repair:{t.id}#v1~") for x in rl.repairer.tags)
    await ctx.close()


async def test_solver_pool_attempts_and_student_rollouts_get_their_own_seeds(tmp_path: Path) -> None:
    from crucible.factory.synth_student_wire import StudentWire
    from crucible.synth.solvers import SolverConfig, evaluate

    log: list[tuple[str, int]] = []
    w = SynthWire(tmp_path / "pool", cap_usd=1.0, make=lambda model, role: SeedChat(log))
    cfg = SolverConfig(name="gptoss-v3", client=w.client("alt_solver", 2), harness="v3")
    rows = await evaluate([("single", fx.t1_hold(), 3)], [cfg], budget_usd=1.0, max_attempts=3, cost_of=lambda tag: 0.0)
    assert len(rows) == 3
    assert len({x for _, x in log}) == 3
    slog: list[tuple[str, int]] = []
    sw = StudentWire(tmp_path / "stu", cap_usd=1.0, make=lambda model, role: SeedChat(slog))
    probe = sw.probe(FalsifyContext, concurrency=1)
    await probe.rollouts(fx.t1_hold(), 3)
    assert len({x for _, x in slog}) == 3


# ---------------------------------------------------------------------------------------------------------- framework fixes

async def test_an_archived_alt_counterexample_replays_its_trace_without_a_model_call(tmp_path: Path) -> None:
    """Found by the re-certification: ALT's inherited replay ran a NEW agent attempt (a model call) instead of the
    archived trace."""
    t = fx.t1_hold()
    calls: list[int] = []

    async def oracle_like(view: Any, world: Any) -> Any:
        calls.append(1)
        for s in [("ap.find_invoice", {"number": "INV-1003"}), ("ap.hold_invoice", {"invoice_id": 103, "reason": "price_variance"})]:
            await world.call(*s)
        return None

    alt = ALT([AgentSpec(name="a", family="nemotron", runner=oracle_like)])
    ctx = FalsifyContext()
    res = await falsify(t, alt, ctx)
    f = res.findings[0]
    assert f.outcome.passed and calls == [1]
    # archive it as if the panel had called it a D6 the contract rejected: expect "pass" on a version that fails it
    cx = Counterexample.of(t, f.model_copy(update={"label": "confirmed", "defect": "D6", "outcome": f.outcome.model_copy(update={"passed": False})}))
    assert cx.probe.program is not None and cx.probe.expect == "pass"
    assert await ALT([]).replay(cx, t, ctx) is False  # the trace passes the task: not re-opened, and no agent was needed
    assert calls == [1]
    strict = t.model_copy(update={"contract": t.contract.model_copy(update={"required": [*t.contract.required, *fx._hold(key="104", pid="x")]})})
    assert await ALT([]).replay(cx, strict, ctx) is True  # a version that rejects the valid trace re-opens it
    # ... unless the repair changed the instruction the trace was judged valid under (recorded from item 22 on)
    assert await ALT([]).replay(cx, strict.model_copy(update={"instruction": strict.instruction + " Then email the vendor."}), ctx) is False
    legacy = cx.model_copy(update={"probe": cx.probe.model_copy(update={"data": {k: v for k, v in cx.probe.data.items() if k != "instruction_sha"}})})
    assert await ALT([]).replay(legacy, strict.model_copy(update={"instruction": strict.instruction + " Then email the vendor."}), ctx) is True  # fails closed
    await ctx.close()


async def test_archive_recheck_runs_the_shown_mutants_when_a_holdout_is_set() -> None:
    from crucible.synth.falsify.mut import bucket
    from crucible.synth.meta.rsi import archive_recheck

    t = fx.planted("D2")
    every = [x.site for x in (await falsify(t, MUT(), FalsifyContext())).confirmed]
    assert every
    h = min(1.0, max(bucket(t.id, "ExtraWrite", s) for s in every) + 1e-9)  # every survivor held out
    arch = Archive(Path("/nonexistent-archive-root"))
    plain = await archive_recheck(arch, FalsifyContext())(t)
    held = await archive_recheck(arch, FalsifyContext(), heldout=h)(t)
    assert "MUT confirms a counterexample" in plain and "MUT confirms a counterexample" not in held


# ---------------------------------------------------------------------------------------------------------- recert

def _store_run(root: Path, name: str, tasks: list[tuple[Task, str]]) -> Path:
    """A stored synthesis run as the pipeline writes it: report.json and tasks/tNN/cegis.json."""
    run = root / name
    recs = []
    for n, (t, st) in enumerate(tasks, 1):
        cert = certify(t, []).model_dump(mode="json")
        doc = {"task_id": t.id, "domain": t.domain, "status": {"certified": "admitted"}.get(st, st), "reasons": [], "rounds": [],
               "certificate": cert, "task": t.model_dump(mode="json")}
        p = run / "tasks" / f"t{n:02d}" / "cegis.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(doc))
        recs.append({"n": n, "id": t.id, "seed": 0, "status": st})
    (run / "report.json").write_text(json.dumps({"domain": "synth-test", "run_id": name, "started": 0, "tasks": recs,
                                                 "families": {"proposer": [{"family": "deepseek"}], "contract_writer": [{"family": "glm"}]}}))
    return run


def _items(tmp_path: Path) -> list[Item]:
    dom = tmp_path / "synth-test"
    _store_run(dom, "20261029-000000-full", [(fx.t1_hold(), "certified"), (fx.planted("D2"), "certified")])
    src = [Source(name="pool", kind="pool", dir=str(dom), archive_root=str(tmp_path), statuses=["certified", "provisional", "rejected"]),
           Source(name="m4", kind="m4", dir=str(dom), archive_root=str(tmp_path))]
    return collect(src)


async def test_collect_merges_sources_and_keeps_the_stored_version(tmp_path: Path) -> None:
    items = _items(tmp_path)
    assert len(items) == 2 and all(i.kinds == ["pool", "m4"] for i in items) and items[0].builders == ["deepseek", "glm"]
    assert items[0].task.digest() == fx.t1_hold().digest() and items[0].old.status == "certified"
    arch = Archive(tmp_path)
    t = fx.t1_hold()
    probe = Probe(cls="MUT", op="DropStep", site="s3", program=t.oracle, expect="fail")
    cx = Counterexample.of(t, Finding(cls="MUT", op="DropStep", site="s3", label="confirmed", defect="D1", probe=probe, outcome=Outcome(passed=True)))
    arch.add(cx)
    copied = copy_archive(items, tmp_path / "recert" / "archives")
    assert copied == {"pool": 1}
    assert [c.id for c in Archive(tmp_path / "recert" / "archives" / "pool").records("synth-test")] == [cx.id]
    assert len(arch.records("synth-test")) == 1  # the source archive is only read


async def test_work_world_moves_the_work_directory_only(tmp_path: Path) -> None:
    t = fx.t1_hold().model_copy(update={"world": WorldRef(factory=f"{fx.MOD}:ap_world", params={})})
    assert await work_world(tmp_path)(t) is not None  # a World without a work parameter builds as it is
    seen: dict[str, Any] = {}
    import tests.unit.synth_fixtures as m

    def probe_world(work: str, extra: int = 0) -> Any:
        seen["work"] = work
        return m.ap_world(extra)

    m.recert_probe_world = probe_world  # type: ignore[attr-defined]
    t2 = t.model_copy(update={"world": WorldRef(factory=f"{fx.MOD}:recert_probe_world", params={"work": "/old/run/work", "extra": 1})})
    await work_world(tmp_path / "w")(t2)
    assert seen["work"] == str(tmp_path / "w") and t2.world.params["work"] == "/old/run/work"


async def test_det_phase_then_paid_phase_on_scripted_agents(tmp_path: Path) -> None:
    items = _items(tmp_path)
    good, _bad = items
    for it in items:
        doc = json.loads(await asyncio.to_thread(det_job, it.task.model_dump_json(), str(tmp_path / "work"), 240, 0.2, [], str(tmp_path / "arch")))
        st, why, _ = det_verdict(it.task, doc)
        if it is good:
            assert st == "paid" and any(r.startswith("MUT:upper") for r in why) and doc["mut_all"] is not None
            gdoc = doc
        else:
            assert st == "rejected" and any(r.startswith(("MUT:confirmed", "MUT-HO:confirmed")) for r in why)
            assert attribute(it, "rejected", why, doc)["outcome"] == "falls"
    assert {r["cls"] for r in gdoc["results"]} >= {"REP", "MUT", "MUT-HO", "ORACLE", "TWIN"}

    async def solves(view: Any, world: Any) -> Any:
        await world.call("ap.find_invoice", {"number": "INV-1003"})
        await world.call("ap.hold_invoice", {"invoice_id": 103, "reason": "price_variance"})
        return None

    async def idle(view: Any, world: Any) -> Any:
        return None

    alts = [AgentSpec(name="alt_solver-super-v3", family="nemotron", runner=solves), AgentSpec(name="alt_solver-gptoss-v3", family="openai", runner=idle)]
    atks = [AgentSpec(name="attacker-openai", family="openai", runner=idle)]
    ctx = FalsifyContext()
    rep, sink = await certify_paid(good, gdoc, ctx=ctx, archive=Archive(tmp_path / "arch"), holdout=0.2, alt=alts, atk=atks)
    assert status_of(rep)[0] == "on_hold" and "ALT:suspects=1" in rep.reasons  # the idle attempt is a suspect nobody decided
    from crucible.synth.falsify.agents import Panel, PanelMember
    from tests.unit.synth_fakes import FakeClient, const

    judge = const({"valid": False, "confidence": 0.9, "why": "no hold was written"})
    panel = Panel(members=[PanelMember("p-n", FakeClient("nemotron", judge)), PanelMember("p-o", FakeClient("openai", judge))], lam=0.7)
    rep, sink = await certify_paid(good, gdoc, ctx=ctx, archive=Archive(tmp_path / "arch"), holdout=0.2, alt=alts, atk=atks, panel=panel)
    await ctx.close()
    st, why = status_of(rep)
    assert rep.task is not None and rep.task.digest() == good.task.digest()  # never edited
    assert st == "provisional" and [r for r in why if not r.startswith("MUT:upper")] == []
    rows = paid_rows(rep, sink)
    assert {r["agent"] for r in rows if r["cls"] == "ALT" and r["passed"]} == {"alt_solver-super-v3"}
    assert "MUT-HO" in rep.certificate.classes
    att = attribute(good, st, why, gdoc, rows)
    assert att["outcome"] == "changes" and att["primary"] in ("heldout", "other")


def test_statuses_outcomes_and_never_overwritten_certificates(tmp_path: Path) -> None:
    from crucible.synth.cegis import CegisReport

    hold = CegisReport(domain="d", status="rejected", reasons=["TWIN:suspects=2", "MUT:upper=0.07>0.05"])
    assert status_of(hold)[0] == "on_hold"
    assert status_of(CegisReport(domain="d", status="rejected", reasons=["ALT:no_pass", "ALT:suspects=1"]))[0] == "rejected"
    it = Item(key="k", source="s", kinds=["pool"], run_dir=str(tmp_path), n=3, task=fx.t1_hold(), old=OldCert(status="rejected", reasons=["ALT:no_pass"]),
              alt_names_shared=True)
    att = attribute(it, "certified", [], None, [{"cls": "ALT", "agent": "alt_solver-super-v3", "passed": True, "decisive": True}])
    assert att["outcome"] == "rises" and att["primary"] == "alt_naming"
    it2 = it.model_copy(update={"alt_names_shared": False})
    assert attribute(it2, "certified", [], None, [{"cls": "ALT", "agent": "alt_solver-ultra-v2", "passed": True}])["primary"] == "seed"
    p = beside(it, "recert-1")
    assert p == tmp_path / "tasks" / "t03" / "certificate.recert-1.json"
    write_new(p, {"a": 1})
    with pytest.raises(FileExistsError):
        write_new(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}


def test_manifest_rows_load_the_exact_stored_version(tmp_path: Path) -> None:
    items = _items(tmp_path)
    it = items[0]
    row = ManifestRow(key=it.key, task_id=it.task.id, version=it.task.version, digest=it.task.digest(), domain=it.task.domain, source=it.source,
                      kinds=it.kinds, task_path=it.old.path, old_status="certified", status="provisional", heldout=0.2)
    (tmp_path / "m.json").write_text(json.dumps({"schema": "crucible.synth.recert/v1", "rows": [row.model_dump(mode="json")]}))
    got = load_manifest(tmp_path / "m.json", statuses=["provisional"])
    assert got[0].load_task().digest() == it.task.digest()
    bad = row.model_copy(update={"digest": "0" * 32})
    with pytest.raises(ValueError):
        bad.load_task()
    assert cheap  # the population module stays importable alongside


def test_recert_never_imports_factory() -> None:
    import ast

    import crucible.synth.recert as r

    tree = ast.parse(Path(r.__file__).read_text())
    mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in mods if m and m.startswith("crucible.factory")]


def test_budget_of_agent_attempts_is_unchanged() -> None:
    assert AgentBudget().max_calls == 24


async def test_status_without_held_out_mutants_reads_the_whole_neighbourhood() -> None:
    from crucible.synth.recert import status_h0

    big = fx.t1_hold().model_copy(update={"world": WorldRef(factory=f"{fx.MOD}:ap_world", params={"extra": 30})})
    whole = (await falsify(big, MUT(), FalsifyContext())).model_dump(mode="json")
    assert status_h0("provisional", ["MUT:upper=0.0600>0.05"], whole) == "certified"  # n >= 59 once the held-out share counts
    assert status_h0("rejected", ["ALT:no_pass", "MUT:upper=0.06>0.05"], whole) == "rejected"
    assert status_h0("on_hold", ["TWIN:suspects=1"], whole) == "on_hold"
    small = (await falsify(fx.t1_hold(), MUT(), FalsifyContext())).model_dump(mode="json")
    assert status_h0("provisional", ["MUT:upper=0.2>0.05"], small) == "provisional"
