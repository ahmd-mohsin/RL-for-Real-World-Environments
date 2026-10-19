"""The local backend and network rules it cannot enforce (M1 audit 2026-10-14 R-13): a host allowlist is refused unless
the world accepts running unenforced, and audit_world names an unenforced network as a SKIP instead of implying a pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.audit import audit_world
from crucible.world.world import World, WorldError


def world(tmp_path: Path, backend_params: dict | None = None, **spec) -> World:
    s = WorldSpec.model_validate({"world": "test/r13", "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}}],
                                  "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}], **spec})
    return World(s, store=Store(tmp_path / "s"), workdir=tmp_path / "w", backend_params=backend_params)


async def checks(w: World) -> dict:
    await w.start()
    try:
        return {c.id: c for c in (await audit_world(w)).checks}
    finally:
        await w.close()


async def test_a_host_allowlist_is_refused_on_the_local_backend(tmp_path: Path):
    with pytest.raises(WorldError, match="'allowlist' cannot be enforced on the local backend"):
        await world(tmp_path, network="allowlist", allowed_hosts=["api.anthropic.com"]).start()


async def test_accepting_it_unenforced_runs_and_the_audit_says_so(tmp_path: Path):
    c = await checks(world(tmp_path, network="allowlist", allowed_hosts=["api.anthropic.com"], backend_params={"accept_unenforced": True}))
    assert c["network.enforced"].status == "skip" and "does not enforce network 'allowlist'" in c["network.enforced"].detail["reason"]


async def test_the_default_no_network_runs_and_is_reported_unenforced(tmp_path: Path):
    c = await checks(world(tmp_path))
    assert c["network.enforced"].status == "skip" and c["network.answer_sources"].status == "pass"
    assert "network.enforced" not in await checks(world(tmp_path / "pub", network="public"))  # nothing to enforce
