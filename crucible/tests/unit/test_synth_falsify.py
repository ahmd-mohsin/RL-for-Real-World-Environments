"""crucible.synth.falsify + certificate: M2 acceptance (part2-paper-core §5, §6, §14).

Every deterministic class finds its planted defect class with an executed, confirmed counterexample; the correct
fixtures get clean certificates; an always-pass contract is rejected by MUT and NULL.
"""

from __future__ import annotations

from typing import Any

import pytest

from crucible.synth.certificate import AdmissionPolicy, Certificate, certify
from crucible.synth.contract import Verdict, grade
from crucible.synth.falsify import ALT, ATK, MUT, NULL, PERT, RED, REP, TAINT, TWIN, AgentSpec, FalsifyContext, OnlineMonitor, cheap, equivalent, falsify
from crucible.synth.infer import skeleton
from crucible.synth.program import Step, lit, run
from crucible.synth.task import Task, WorldRef
from tests.unit import synth_fixtures as fx


async def _all(task: Task, falsifiers=None, ctx: FalsifyContext | None = None):
    ctx = ctx or FalsifyContext()
    return {f.name: await falsify(task, f, ctx) for f in (falsifiers or cheap())}


def _confirmed(res, cls):
    return [(f.op, f.defect) for f in res[cls].confirmed]


CORRECT = {**fx.CORRECT, "t6": fx.t6_resolve}


@pytest.mark.parametrize("name", sorted(CORRECT))
async def test_correct_fixtures_get_clean_certificates(name):
    task = CORRECT[name]()
    res = await _all(task)
    for cls, r in res.items():
        assert r.applicable, (cls, r.reason)
        assert r.c == 0, (cls, [(f.op, f.site, f.defect, f.outcome.reasons) for f in r.confirmed])
        assert not r.suspects, (cls, [(f.op, f.site) for f in r.suspects])
    cert = certify(task, res.values(), AdmissionPolicy(u_max={}, require_alt_pass=False))
    assert cert.clean and cert.admitted
    assert res["MUT"].n >= 13 and res["NULL"].n >= 8 and res["TAINT"].n >= 3 and res["REP"].n == 3


# which class must catch each planted defect, with which label (§5's table)
EXPECTED = {
    "D1": [("MUT", "D1"), ("NULL", "D1")],
    "D2": [("MUT", "D2")],
    "D3": [("MUT", "D3")],
    "D4": [("RED", "D4")],
    "D5": [("TAINT", "D5")],
    "D6": [("MUT", "D6")],
    "D7": [("TAINT", "D7")],
    "D8": [("NULL", "D8")],
    "D9": [("TWIN", "D9"), ("MUT", "D9")],
    "D10": [("MUT", "D10")],
    "D11": [("REP", "D11")],
}


@pytest.mark.parametrize("code", sorted(EXPECTED, key=lambda c: int(c[1:])))
async def test_each_planted_defect_is_caught_by_its_class(code):
    task = fx.planted(code)
    res = await _all(task)
    for cls, d in EXPECTED[code]:
        hits = [f for f in res[cls].confirmed if f.defect == d]
        assert hits, (code, cls, {k: _confirmed(res, k) for k in res})
        for f in hits:  # executed counterexamples, archivable as they are
            assert f.outcome.valid and f.label == "confirmed"
            if cls in ("MUT", "NULL"):
                assert f.probe.program is not None


async def test_planted_operator_details():
    res = await _all(fx.planted("D2"), [MUT()])
    assert {f.op for f in res["MUT"].confirmed} == {"ExtraWrite"} and all(f.outcome.data["outside"] for f in res["MUT"].confirmed)
    res = await _all(fx.planted("D6"), [MUT()])
    (f,) = res["MUT"].confirmed
    assert (f.op, f.site, f.outcome.equivalent, f.outcome.reasons) == ("DropStep", "s2", True, ["TRACE:m_po_first"])
    res = await _all(fx.planted("D10"), [MUT(), TAINT()])
    ident = next(f for f in res["MUT"].confirmed if f.op == "Identity")
    assert ident.defect == "D10" and "MISSING_REQUIRED:r_reason" in ident.outcome.reasons
    assert ("r_const", "D5") in _confirmed(res, "TAINT")  # and the impossible constant has no source
    res = await _all(fx.planted("D8"), [NULL()])
    assert [f.site for f in res["NULL"].confirmed] == ["shortcut:desk.admin_update:T-3"]
    res = await _all(fx.planted("D4"), [RED()])
    assert [f.probe.data["id"] for f in res["RED"].confirmed] == ["r_vac"]


async def test_a_stated_order_is_not_over_constraint():
    t = fx.planted("D6")
    stated = t.contract.model_copy(update={"trace": [t.contract.trace[0].model_copy(update={"stated": "check it against the purchase order"})]})
    res = await _all(t.model_copy(update={"contract": stated}), [MUT()])
    assert res["MUT"].c == 0  # the instruction states it, so the mutant that skips the read is not equivalent and rightly fails


async def test_always_pass_contract_is_rejected_by_mut_and_null():
    res = await _all(fx.always_pass())
    assert {d for _, d in _confirmed(res, "MUT")} >= {"D1", "D2"}
    assert ("noop", "D4") in _confirmed(res, "NULL")
    cert = certify(fx.always_pass(), res.values(), AdmissionPolicy(u_max={}, require_alt_pass=False))
    assert not cert.admitted and "MUT:confirmed" in " ".join(cert.reasons) and "NULL:confirmed" in " ".join(cert.reasons)


async def test_position_keyed_contract_is_caught_by_pert():
    res = await _all(fx.position_keyed(), [PERT(), REP(), MUT()])
    assert ("shuffle", "D11") in _confirmed(res, "PERT")
    assert res["REP"].c == 0 and res["MUT"].c == 0  # nothing else sees it: only the start-state perturbation does


async def test_nondeterministic_tool_is_caught_by_rep_and_pinned_values_flip_the_verdict():
    res = await _all(fx.planted("D11"), [REP()])
    assert all("delta" in f.outcome.data["differs"] for f in res["REP"].confirmed) and res["REP"].c == 3
    # a contract inferred from one run pins the random ref: REP sees Δ move, and no later run of the oracle passes
    t = fx.t6_resolve().model_copy(update={"world": fx.planted("D11").world})
    first = await run(await t.world.build(), t.oracle, t.instruction)
    pinned = t.model_copy(update={"contract": skeleton(first, t.contract.normalizers)})
    res = await _all(pinned, [REP(), MUT()])
    assert res["REP"].c == 3 and next(f for f in res["MUT"].findings if f.op == "Identity").label == "confirmed"


async def test_pert_needs_a_perturbable_world():
    class Plain:  # a SynthWorld without perturb
        def __init__(self):
            self.w = fx.ap_world()

        async def reset(self, seed):
            return await self.w.reset(seed)

        async def tools(self):
            return await self.w.tools()

        async def call(self, tool, args):
            return await self.w.call(tool, args)

        async def snapshot(self):
            return await self.w.snapshot()

        async def restore(self, ref):
            await self.w.restore(ref)

        async def canon(self):
            return await self.w.canon()

    r = await falsify(fx.t1_hold(), PERT(), FalsifyContext(make_world=lambda t: Plain()))
    assert not r.applicable and "perturb" in r.reason
    cert = certify(fx.t1_hold(), [r], AdmissionPolicy(require_applicable=[], u_max={}, require_alt_pass=False))
    assert not cert.classes["PERT"].applicable and cert.admitted  # graceful degradation, recorded in the certificate


async def test_twin_details():
    res = await _all(fx.t1_hold(), [TWIN()])
    valid = [f for f in res["TWIN"].findings if f.outcome.valid]
    assert valid and all(f.outcome.data["oracle_passes_twin"] is False and f.outcome.data["twin_passes_twin"] for f in valid)
    f = valid[0]
    tgt = f.probe.data["target"]
    assert tgt in f.probe.instruction and "INV-1003" not in f.probe.instruction
    dead = [f for f in (await _all(fx.t4_priority(), [TWIN()]))["TWIN"].findings if not f.outcome.valid]
    assert dead and all(f.outcome.data["dead"] for f in dead)  # a twin the World cannot act on is not a trial
    # a selector literal (a customer several tickets share) re-derives the keys it selects
    res = await _all(fx.t4_priority(), [TWIN()])
    sel = next(f for f in res["TWIN"].findings if f.outcome.valid and f.probe.data["literal"] == "bob@example.com")
    assert len(sel.probe.data["sigma"]) == 1


async def test_mut_equivalence_and_volume():
    t = fx.t1_hold()
    ctx = FalsifyContext()
    o = await ctx.oracle(t)
    no_po = t.oracle.model_copy(update={"steps": [s for s in t.oracle.steps if s.id != "s2"]})
    assert equivalent(await ctx.run(t, no_po), o, t.contract, t.instruction)  # an unreferenced read can be dropped
    swapped = t.oracle.model_copy(update={"steps": [*t.oracle.steps[:2], Step(id="s3", tool="ap.hold_invoice", args={"invoice_id": lit(104), "reason": lit("price_variance")})]})
    assert not equivalent(await ctx.run(t, swapped), o, t.contract, t.instruction)
    r = (await _all(fx.t5_batch(), [MUT()]))["MUT"]
    ops = {f.op for f in r.findings}
    assert r.n >= 59 and ops >= {"Identity", "DropStep", "DupStep", "Truncate", "SwapEntity", "PerturbValue", "ExtraWrite", "Reorder"}
    reorder = [f for f in r.findings if f.op == "Reorder"]
    assert reorder and all(f.label == "equivalent" for f in reorder)  # independent writes swapped: same Δ, still passes
    assert certify(fx.t5_batch(), [r], AdmissionPolicy(require_applicable=["MUT"], require_alt_pass=False)).classes["MUT"].upper <= 0.05


async def _alt_runner(view, world) -> Any:
    """An alternative solution: lists first, decides in another order (a different trace, the same end state)."""
    inv = {i["number"]: i["id"] for i in (await world.call("ap.list_invoices", {"status": "pending"})).value["invoices"]}
    await world.call("ap.hold_invoice", {"invoice_id": inv["INV-1006"], "reason": "missing_receipt"})
    await world.call("ap.approve_invoice", {"invoice_id": inv["INV-1002"]})
    await world.call("ap.reject_invoice", {"invoice_id": inv["INV-1005"], "reason": "duplicate"})


async def _lazy(view, world) -> Any:
    await world.call("ap.approve_invoice", {"invoice_id": 102})
    return "done"


async def test_certificate_admission_with_injected_alternative_solvers():
    t = fx.t5_batch()
    alt = ALT([AgentSpec("alt-1", "fam-x", _alt_runner), AgentSpec("lazy", "fam-y", _lazy), AgentSpec("builder", "fam-b", _alt_runner)],
              exclude_families={"fam-b"})
    atk = ATK([AgentSpec("atk-1", "fam-z", _lazy)])
    res = await _all(t, [*cheap(), alt, atk])
    assert res["ALT"].n == 2 and [f.label for f in res["ALT"].findings] == ["clean", "suspect"]  # the lazy run fails: adjudicate
    assert res["ATK"].c == 0 and res["ATK"].findings[0].label == "clean"
    cert = certify(t, res.values())
    assert cert.reasons == ["ALT:suspects=1"]
    cert = certify(t, res.values(), AdmissionPolicy(allow_open_suspects=True))
    assert cert.admitted and cert.classes["MUT"].upper <= 0.05 and cert.classes["ALT"].passes == 1
    again = Certificate.model_validate_json(cert.model_dump_json())
    assert again == cert and again.task_digest == t.digest()
    small_res = (await _all(fx.t1_hold())).values()
    small = certify(fx.t1_hold(), small_res)
    assert not small.admitted and small.reasons == ["ALT:no_pass"] and any(r.startswith("MUT:upper=") for r in small.provisional_for)
    # c = 0 but n < 59 distinct mutants: admitted as provisional with its U, never counted as certified (§6, M3)
    prov = certify(fx.t1_hold(), small_res, AdmissionPolicy(require_alt_pass=False))
    assert prov.admitted and prov.status == "provisional" and not prov.certified and prov.classes["MUT"].n < 59 and prov.classes["MUT"].upper > 0.05
    assert certify(fx.t1_hold(), small_res, AdmissionPolicy(require_alt_pass=False, provisional_ok=False)).status == "rejected"
    assert certify(t, res.values(), AdmissionPolicy(allow_open_suspects=True)).status == "certified"
    # the recorded ALT trace replays as a literal program
    prog = res["ALT"].findings[0].outcome.data["program"]
    assert next(s["tool"] for s in prog["steps"]) == "ap.list_invoices"


async def test_online_monitor_flags_passing_rollouts_as_suspects():
    t = fx.t1_hold()
    seen = []
    mon = OnlineMonitor(quarantine=lambda task, f: seen.append(f.site))
    r = await run(fx.ap_world(), t.oracle, t.instruction)
    v = grade(r, t.contract)
    assert mon.observe(t, r, v) is None and mon.observe(t, r, Verdict(passed=False)) is None
    f = mon.observe(t, r, v, flags=["edited the audit log"])
    assert f.label == "suspect" and seen == [f.site] and f.probe.program is not None


async def test_mut_scales_with_the_world_and_counts_distinct_effects_only():
    """M3 (§6): ExtraWrite and SwapEntity over every sibling, PerturbValue at several magnitudes, deduplicated by normalised
    effect. A 3-step task reaches n >= 59 distinct mutants once its World has enough entities; in the 8-invoice World it
    cannot, and is admitted only as provisional."""
    from crucible.synth.falsify.mut import number_alternatives

    big = fx.t1_hold().model_copy(update={"world": WorldRef(factory=f"{fx.MOD}:ap_world", params={"extra": 30})})
    r = (await _all(big, [MUT()]))["MUT"]
    assert r.n >= 59 and r.c == 0 and not r.suspects
    counted = [f.outcome.data["effect"] for f in r.findings if f.outcome.valid]
    assert len(counted) == len(set(counted))  # one trial per behaviour
    dups = [f for f in r.findings if not f.outcome.valid]
    assert dups and all(f.outcome.data["duplicate_of"] for f in dups)
    assert {f.op for f in r.findings if f.outcome.valid} >= {"Identity", "SwapEntity", "ExtraWrite", "PerturbValue", "DropStep"}
    assert certify(big, [r], AdmissionPolicy(require_applicable=["MUT"], require_alt_pass=False)).status == "certified"
    small = (await _all(fx.t1_hold(), [MUT()]))["MUT"]
    assert small.n < 59 and certify(fx.t1_hold(), [small], AdmissionPolicy(require_applicable=["MUT"], require_alt_pass=False)).status == "provisional"
    # dedup keeps the most informative of equal effects: PerturbValue(number) holds the same sibling a SwapEntity holds,
    # and on the "any invoice" contract the confirmed D9 (SwapEntity) is what is counted, never the suspect duplicate
    d9 = (await _all(fx.planted("D9"), [MUT()]))["MUT"]
    assert d9.confirmed and all(f.op == "SwapEntity" for f in d9.confirmed) and not d9.suspects
    assert number_alternatives(1200.0)[:4] == [1212.0, 1188.0, 1320.0, 1080.0]
