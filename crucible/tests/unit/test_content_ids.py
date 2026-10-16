"""Content ids name the scheme that made them (M0 audit 2026-10-14 R-21)."""

from __future__ import annotations

import pytest

from crucible.core import canonical, ids
from crucible.core.schema.reward import RewardJson
from crucible.core.store import Store
from crucible.world.world import World, WorldError
from tests.unit.test_world_local import Supervisor, _materialize_tools, _world, pack, spec  # noqa: F401 (pack is a fixture)


def test_scheme_1_ids_are_the_bare_digest_and_later_schemes_are_prefixed():
    doc = RewardJson(reward=1.0)
    hexd = doc.content_hash()
    assert ids.parse_content_id(hexd) == (1, hexd) and doc.matches_id(hexd)  # every id made so far stays what it was
    assert ids.format_content_id(hexd, 2) == f"c2:{hexd}" and ids.parse_content_id(f"c2:{hexd}") == (2, hexd)
    for bad in (f"c1:{hexd}", "abc", f"sha256:{hexd}", hexd.upper()):
        with pytest.raises(ValueError):
            ids.parse_content_id(bad)
    with pytest.raises(ValueError, match="does not know"):
        doc.matches_id(f"c9:{hexd}")
    assert not RewardJson(reward=0.5).matches_id(hexd)


def _scheme_2(monkeypatch):
    monkeypatch.setitem(canonical.SCHEMES, 2, lambda o: "scheme-2|" + canonical.dumps(o))
    monkeypatch.setattr(ids, "HASH_SCHEME", 2)


async def test_a_world_ref_from_before_a_scheme_bump_still_restores(pack, tmp_path, monkeypatch):  # noqa: F811
    """The audit's case: HASH_SCHEME entered no hash, so a canonical change broke World.restore of every stored ref."""
    w1 = await _world(pack, tmp_path)
    try:
        await w1.reset()
        ref = await w1.snapshot()
    finally:
        await w1.close()
    _scheme_2(monkeypatch)
    w2 = World(spec(), store=Store(tmp_path / "store"), pack_root=pack, workdir=tmp_path / "w2")
    _materialize_tools(w2, pack)
    w2.attach(Supervisor())
    await w2.start()
    try:
        assert w2.spec_hash.startswith("c2:") and ids.parse_content_id(ref.world)[0] == 1
        await w2.restore(ref)  # matched under scheme 1, the scheme of the ref's id
        after = await w2.snapshot()  # the same state; its ref names the world by its scheme-2 id
        assert {k: a.fingerprint for k, a in after.apps.items()} == {k: a.fingerprint for k, a in ref.apps.items()} and after.world == w2.spec_hash
        with pytest.raises(WorldError, match="belongs to world"):
            await w2.restore(ref.model_copy(update={"world": "c2:" + "0" * 64}))
    finally:
        await w2.close()


async def test_a_world_whose_spec_changed_after_it_was_built_restores_its_own_refs(pack, tmp_path):  # noqa: F811
    """A World may adjust its spec once running (the browser World does): its refs name the spec it was built from, so
    restore checks the id against that spec, as it compared the stored id before R-21, not against the live one."""
    w = await _world(pack, tmp_path)
    try:
        await w.reset()
        ref = await w.snapshot()
        w.spec = w.spec.model_copy(update={"world": w.spec.world + "-adjusted"})
        await w.restore(ref)
    finally:
        await w.close()
