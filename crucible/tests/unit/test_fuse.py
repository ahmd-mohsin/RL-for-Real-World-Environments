"""m13 G-9: fuse (cross-World) and nest. helpdesk ⊕ warehouse: the fused WorldSpec validates, two fresh starts agree
(A9's property), the cross-App check is the interaction neither part has, and A14 / A17 pass; World-fusion conflicts
are errors; nest puts the inner task inside the outer one."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from crucible.accept.policy import Policy
from crucible.accept.run import accept_task
from crucible.core.schema.world import WorldSpec
from crucible.forge.compose import Parts, nest, rename_apps
from crucible.forge.fuse import fuse_worlds


@pytest.fixture(scope="module")
def forged():
    import asyncio

    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    return {t.task_id.split("/")[-1]: t for t, _ in col}


@pytest.fixture(scope="module")
def fused(forged, tmp_path_factory):
    from crucible_pack_toy.forge.fused import build_fused

    return build_fused(forged["t1_duplicate"], tmp_path_factory.mktemp("fused") / "root")


async def test_fused_world_and_task(fused):
    w = WorldSpec.load(fused.pack_root / fused.world_file)
    assert {a.id for a in w.apps} == {"tickets", "kb", "inventory"} and set(w.actor("agent").tools) >= {"tickets.*", "inventory.*"}
    assert await fused.start_fingerprint() == await fused.start_fingerprint()  # two fresh starts agree
    comp = fused.composite
    assert comp.operator == "fuse" and not comp.dependent and comp.cross == ["x.consistency"]
    rep, _ = await fused.grade(comp.oracle)
    assert rep.reward == 1.0, rep.failing_gates()
    only_helpdesk, _ = await fused.grade(comp.component_oracle("c1"))
    only_wh, _ = await fused.grade(comp.component_oracle("c2"))
    # each part alone keeps its own gates but fails the cross-App consistency: the interaction is real
    assert "x.consistency" in only_helpdesk.failing_gates() and not any(g.startswith("c1.") for g in only_helpdesk.failing_gates())
    assert "x.consistency" in only_wh.failing_gates() and not any(g.startswith("c2.") for g in only_wh.failing_gates())
    for n in comp.negatives:
        r, _ = await fused.grade(n.actions)
        assert r.reward == 0.0 and (n.expected_gate is None or n.expected_gate in r.failing_gates()), (n.id, r.failing_gates())
    acc = await accept_task(fused, policy=Policy(required=["A14", "A17"]), cache_root=Path(tempfile.mkdtemp()), only=["A14", "A17"], reuse=False)
    assert {k: str(v.status) for k, v in acc.results.items() if k in ("A14", "A17")} == {"A14": "pass", "A17": "pass"}, \
        {k: v.reasons for k, v in acc.results.items()}


def _w(world, apps, actors, **kw):
    return WorldSpec(world=world, apps=apps, actors=actors, **kw)


def _app(i, path):
    return {"id": i, "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": path}}}


def test_fuse_worlds_renames_and_conflicts():
    a = _w("p/a", [_app("mail", "a/mail.db")], [{"id": "agent", "kind": "policy", "tools": ["mail.*"]}, {"id": "boss", "kind": "system"}])
    b = _w("p/b", [_app("mail", "b/mail.db")], [{"id": "agent", "kind": "policy", "tools": ["mail.send"]}, {"id": "boss", "kind": "system"}],
           events=[{"id": "e", "when": {"on_tool": "mail.send"}, "actor": "boss", "action": {"type": "permission", "to": "agent", "revoke": ["mail.send"]}}])
    f, ren = fuse_worlds(a, b, world="p/ab")
    assert ren == {"mail": "b_mail"} and [x.id for x in f.apps] == ["mail", "b_mail"]
    assert f.actor("agent").tools == ["mail.*", "b_mail.send"] and f.events[0].action.revoke == ["b_mail.send"]
    with pytest.raises(ValueError, match="state path"):
        fuse_worlds(a, _w("p/c", [_app("db", "a/mail.db")], [{"id": "agent", "kind": "policy"}]), world="x")
    with pytest.raises(ValueError, match="two different actors"):
        fuse_worlds(a, _w("p/d", [_app("db", "d/db")], [{"id": "agent", "kind": "policy"}, {"id": "boss", "kind": "simulated"}]), world="x")
    with pytest.raises(ValueError, match="clocks differ"):
        fuse_worlds(_w("p/e", [_app("x", "e")], [{"id": "agent", "kind": "policy"}], clock={"mode": "step", "t0": "2026-01-01T00:00:00Z"}),
                    _w("p/f", [_app("y", "f")], [{"id": "agent", "kind": "policy"}], clock={"mode": "step", "t0": "2026-02-01T00:00:00Z"}), world="x")


def test_rename_apps_rewrites_checks_and_oracle(forged):
    p = rename_apps(Parts.of(forged["t1_duplicate"]), {"tickets": "b_tickets"})
    assert all(a["function_name"].startswith("b_tickets.") for a in p.oracle if a.get("function_name"))
    assert all(c.app in ("b_tickets", None) for c in p.checks.checks)


async def test_nest_places_inner_inside_outer(forged):
    from crucible_pack_toy.forge.composed import ToyComposedTask

    outer, inner = forged["t5_compose"], forged["t1_duplicate"]
    comp = nest(Parts.of(outer), Parts.of(inner), task_id="toy/helpdesk_v2/nest_t5_t1", at=2)
    n_in = len(inner.oracle)
    assert comp.oracle[:2] == outer.oracle[:2] and comp.oracle[2:2 + n_in] == inner.oracle and comp.oracle[2 + n_in:] == outer.oracle[2:]
    o = next(c for c in comp.components if c.id == "outer")
    assert o.actions == [0, 1, *range(2 + n_in, len(comp.oracle))] and comp.dependent
    t = ToyComposedTask(comp, [outer, inner])
    rep, _ = await t.grade(comp.oracle)
    assert rep.reward == 1.0, rep.failing_gates()
    with pytest.raises(ValueError, match="strictly inside"):
        nest(Parts.of(outer), Parts.of(inner), task_id="x", at=0)
