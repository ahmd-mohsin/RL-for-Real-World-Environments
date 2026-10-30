"""Part 1 V5 (2026-10-30, docs/research/env-synthesis/part1-framework-improvements.md §2(c); GateTruth 2608.12635: testbenches
revised to pass a mutant gate overfit it): a held-out mutant set that synth's repair loop never sees. The repairer used to
see counterexamples from the same MUT neighbourhood the certificate then counted, so a contract repaired to kill exactly
the shown mutants passed. ``MUT(split="shown")`` feeds the rounds, ``MUTHeldOut`` (class ``MUT-HO``) runs once on the
certified version, and a held-out survivor rejects it."""

from __future__ import annotations

import json

from crucible.synth.archive import Archive
from crucible.synth.cegis import refine
from crucible.synth.certificate import AdmissionPolicy, certify
from crucible.synth.contract import EffectPattern
from crucible.synth.falsify import GATED_BY_REP, MUT, FalsifyContext, MUTHeldOut, cheap, falsify, population
from crucible.synth.falsify.base import ClassResult, Counterexample, Finding, Outcome, Probe
from crucible.synth.falsify.mut import bucket
from crucible.synth.offload import FalsifyPool, _job
from crucible.synth.task import Task
from tests.unit import synth_fixtures as fx

LENIENT = AdmissionPolicy(u_max={}, require_alt_pass=False)


def _sib(site: str) -> str:
    return site.rsplit("+", 1)[1]  # ExtraWrite sites: "<step>.<arg>+<distractor key>"


async def _survivors(task: Task, f: MUT) -> list[str]:
    return sorted(x.site for x in (await falsify(task, f, FalsifyContext())).confirmed)


def _holdout_between(task: Task, sites: list[str]) -> float:
    """A holdout that splits the ExtraWrite survivors: the middle of their buckets."""
    b = sorted(bucket(task.id, "ExtraWrite", s) for s in sites)
    return (b[len(b) // 2 - 1] + b[len(b) // 2]) / 2


def overfit_repairer(seen: list[str]):
    """Narrows the frame for exactly the distractor invoices its counterexamples name, and no other: every invoice it was
    not shown stays writable."""

    async def repair(task: Task, cxs: list[Counterexample]) -> Task | None:
        seen.extend(c.site for c in cxs)
        named = {_sib(s) for s in seen if "+" in s}
        keys = [k for k in ("101", "102", "103", "104", "105", "106", "107", "108") if k not in named]
        allowed = [EffectPattern(id=f"a_inv_{k}", app="ap", coll="invoices", key=k, kind="any", why="invoice edits are routine") for k in keys]
        allowed.append(EffectPattern(id="a_log", app="ap", coll="audit_log", kind="any", why="audit rows are routine"))
        return task.next_version(contract=task.contract.model_copy(update={"allowed": allowed}))

    return repair


async def test_the_split_partitions_each_neighbourhood_and_the_default_is_unchanged() -> None:
    t = fx.planted("D2")
    every = await _survivors(t, MUT())
    assert len(every) == 5 and every == await _survivors(t, MUT(split="all"))  # the default neighbourhood, as before
    h = _holdout_between(t, every)
    shown, held = await _survivors(t, MUT(split="shown", holdout=h)), await _survivors(t, MUTHeldOut(holdout=h))
    assert shown and held and set(shown).isdisjoint(held) and sorted(shown + held) == every
    ctx = FalsifyContext()
    for f in (MUT(split="shown", holdout=h), MUTHeldOut(holdout=h)):
        assert next(p.op for p in await f.propose(t, ctx)) == "Identity"  # the oracle check is in both
    assert bucket(t.id, "ExtraWrite", every[0]) == bucket(t.next_version().id, "ExtraWrite", every[0])  # a repair never moves a mutant
    assert MUTHeldOut().name == "MUT-HO" and "MUT-HO" in GATED_BY_REP


async def test_a_repair_fitted_to_the_shown_mutants_is_caught_by_the_held_out_ones(tmp_path) -> None:
    t = fx.planted("D2")
    h = _holdout_between(t, await _survivors(t, MUT()))
    # no split: the repairer sees all five distractors, fixes all five, and the version is certified
    seen_all: list[str] = []
    rep = await refine(t, repair=overfit_repairer(seen_all), ctx=FalsifyContext(), archive=Archive(tmp_path / "a"), policy=LENIENT)
    assert rep.status == "admitted" and len({_sib(s) for s in seen_all}) == 5
    # the split: the repairer sees the shown distractors only; a held-out one still passes on the certified version
    seen: list[str] = []
    archive = Archive(tmp_path / "b")
    rep = await refine(t, repair=overfit_repairer(seen), ctx=FalsifyContext(), archive=archive, falsifiers=population(h), policy=LENIENT,
                       heldout=MUTHeldOut(holdout=h))
    held = await _survivors(t, MUTHeldOut(holdout=h))
    assert rep.status == "rejected" and any(r.startswith("MUT-HO:confirmed=") for r in rep.reasons)
    assert not {_sib(s) for s in seen} & {_sib(s) for s in held}  # the repairer never saw a held-out site
    assert {c.cls for c in archive.records("synth-test", t.id)} == {"MUT"}  # held-out findings are never archived
    ho = rep.final.heldout
    assert ho["c"] >= 1 and ho["kill_rate_shown"] == 1.0 and ho["kill_rate_heldout"] < 1.0  # the gap the doc wants measured
    assert rep.certificate is not None and rep.certificate.classes["MUT-HO"].c == ho["c"]


def _result(cls: str, labels: list[str]) -> ClassResult:
    fs = [Finding(cls=cls, op="ExtraWrite", site=f"s{i}", label=lab, defect="D2" if lab == "confirmed" else None,  # type: ignore[arg-type]
                  probe=Probe(cls="MUT", op="ExtraWrite", site=f"s{i}"), outcome=Outcome(passed=lab != "clean")) for i, lab in enumerate(labels)]
    return ClassResult(cls=cls, findings=fs)


def test_a_held_out_survivor_rejects_and_no_held_out_class_changes_nothing() -> None:
    t = fx.t1_hold()
    pol = AdmissionPolicy(u_max={}, require_alt_pass=False, require_applicable=[])
    assert certify(t, [_result("MUT", ["clean", "clean"])], pol).status == "certified"
    c = certify(t, [_result("MUT", ["clean", "clean"]), _result("MUT-HO", ["clean", "confirmed"])], pol)
    assert c.status == "rejected" and c.reasons == ["MUT-HO:confirmed=1"]


def test_the_rounds_population_and_the_worker_pool_carry_the_split() -> None:
    assert [type(f).__name__ for f in population(0.0)] == [type(f).__name__ for f in cheap()]  # off: exactly cheap()
    mut = next(f for f in population(0.2) if f.name == "MUT")
    assert (mut.split, mut.holdout) == ("shown", 0.2)  # type: ignore[attr-defined]
    pool = FalsifyPool(1, holdout=0.2)
    try:
        assert pool.holdout == 0.2 and [f.name for f in pool.falsifiers()] == [f.name for f in cheap()]
    finally:
        pool.close()
    t = fx.planted("D2")
    every = json.loads(_job(t.model_dump_json(), 240, ["MUT"]))["results"][0]
    shown = json.loads(_job(t.model_dump_json(), 240, ["MUT"], 0.5))["results"][0]
    assert len(shown["findings"]) < len(every["findings"])  # a worker runs the shown mutants only
