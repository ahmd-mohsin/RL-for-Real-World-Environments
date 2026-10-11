"""m13 G-2 / G-3: composition operators and the composition rigor checks (A14 necessity, A15 complexity gain, A17
non-interference) on forged toy/helpdesk_v2 tasks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from crucible.accept.policy import Policy
from crucible.accept.run import accept_task
from crucible.forge.compose import Parts, chain
from crucible.forge.schema import CheckSet


@pytest.fixture(scope="module")
def forged():
    import asyncio

    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    tasks = {t.task_id: t for t, _ in col}
    assert len(tasks) >= 3
    return tasks


def _pick(forged, *names):
    return [next(t for tid, t in forged.items() if tid.endswith(n)) for n in names]


async def _accept(task, only):
    pol = Policy(required=list(only))
    return await accept_task(task, policy=pol, cache_root=Path(tempfile.mkdtemp()), only=only, reuse=False)


async def test_real_chain_passes_necessity_and_non_interference(forged):
    from crucible_pack_toy.forge.composed import ToyComposedTask

    a, b = _pick(forged, "t1_duplicate", "t3_in_transit")
    comp = chain(Parts.of(a), Parts.of(b), task_id="toy/helpdesk_v2/chain_t1_t3")
    assert [c.id for c in comp.components] == ["c1", "c2"] and comp.oracle == list(a.oracle) + list(b.oracle) and comp.depth == 1
    assert all(g.startswith(("c1.", "c2.")) for g in {c.gate for c in comp.checks.checks})
    t = ToyComposedTask(comp, [a, b])
    rep, _ = await t.grade(comp.oracle)
    assert rep.reward == 1.0  # the composite oracle passes every component's gates
    r = await _accept(t, ["A14", "A17"])
    st = {k: str(v.status) for k, v in r.results.items() if k in ("A14", "A17")}  # (A1 runs as their dependency)
    assert st == {"A14": "pass", "A17": "pass"}, {k: v.reasons for k, v in r.results.items()}
    for n in comp.negatives:  # the generated drop-component negatives are killed at the dropped component's gate
        if n.provenance == "compose:drop_component":
            rr, _ = await t.grade(n.actions)
            assert rr.reward == 0.0 and n.expected_gate in rr.failing_gates()


async def test_padded_composite_is_rejected_by_a14(forged):
    from crucible_pack_toy.forge.composed import ToyComposedTask

    a, b = _pick(forged, "t1_duplicate", "t3_in_transit")
    pb = Parts.of(b)
    # padding: the component keeps only "nothing else changed" scope checks, nothing that requires its work
    padded = Parts(pb.spec, CheckSet(task_id=pb.spec.task_id, checks=[c for c in pb.checks.checks if c.gate == "scope"]), pb.oracle, [], "pad")
    comp = chain(Parts.of(a), padded, task_id="toy/helpdesk_v2/padded")
    r = await _accept(ToyComposedTask(comp, [a, b]), ["A14"])
    assert str(r.results["A14"].status) == "fail" and "padding" in r.results["A14"].reasons[0]


async def test_a15_complexity_gain(forged):
    from crucible_pack_toy.forge.composed import ToyComposedTask

    a, b = _pick(forged, "t1_duplicate", "t3_in_transit")
    comp = chain(Parts.of(a), Parts.of(b), task_id="toy/helpdesk_v2/chain_t1_t3")

    def runs(k, n):
        return [{"policy": "llm:m", "reward": 1.0 if i < k else 0.0, "failing": [] if i < k else ["outcome"]} for i in range(n)]

    for src in (a, b):
        src.calibration_runs = lambda: runs(6, 10)  # each part p = 0.6
    harder = ToyComposedTask(comp, [a, b], calibration=runs(2, 10))  # composite p = 0.2 <= 0.6 - 0.1
    assert str((await _accept(harder, ["A15"])).results["A15"].status) == "pass"
    flat = ToyComposedTask(comp, [a, b], calibration=runs(6, 10))  # no harder than its parts: decomposable
    r = await _accept(flat, ["A15"])
    assert str(r.results["A15"].status) == "fail" and "decomposable" in r.results["A15"].reasons[0]
    assert str((await _accept(ToyComposedTask(comp, [a, b], calibration=[]), ["A15"])).results["A15"].status) == "pending"


async def test_non_composites_are_not_applicable_and_composites_compose(forged):
    (a, b, c) = _pick(forged, "t1_duplicate", "t3_in_transit", "t2_approval")
    r = await _accept(a, ["A14", "A15", "A17"])
    assert {k: v.na_reason for k, v in r.results.items() if k != "A1"} == {"A14": "not_composite", "A15": "not_composite", "A17": "not_composite"}
    inner = chain(Parts.of(a), Parts.of(b), task_id="toy/inner")
    outer = chain(inner, Parts.of(c), task_id="toy/outer")
    assert outer.depth == 2 and len(outer.oracle) == len(inner.oracle) + len(c.oracle)
    assert {cc.source for cc in outer.components} == {"toy/inner", c.task_id}
