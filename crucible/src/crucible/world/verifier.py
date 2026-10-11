"""The separate verifier World (M1; Harbor's separate verifier environment, Karotte's submission custody).

``VerifierSpec.mode == "separate"`` grades a copy of the policy's World, never the World it acted in::

    quiesce + snapshot (the hand-off point: no policy process survives it)
      -> custody check of the snapshot (``world/custody.py``): artifacts and protected paths, from the store
      -> close the policy's World (nothing it left can run while the grader reads)
      -> fresh World from the same spec (verifier backend, no responders, no interceptors)
      -> restore the snapshot (fingerprints must match) -> grade there -> close

The verifier World carries the policy's trace (read-only copy) and episode-start ref, so trace-based checks and
``VerifierView.unchanged`` work as they do in place. Its state roots must be its own: two worlds on one host whose
App state points at the same absolute path would share files, and the hand-off would be separation in name only, so
that configuration is refused.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.schema.state import WorldRef
from crucible.world.custody import CustodyViolation, check_snapshot, collect_artifacts
from crucible.world.world import World, WorldError


@dataclass
class Handoff:
    """What crossed from the policy's World to the verifier's (written to ``verifier/handoff.json`` by the harness)."""

    world_ref: str
    snapshot_step: int
    apps: dict[str, str]  # app -> state fingerprint handed over
    artifacts: dict[str, Any] | None = None  # collect_artifacts manifest
    backend: str = ""
    checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"world_ref": self.world_ref, "snapshot_step": self.snapshot_step, "apps": self.apps, "artifacts": self.artifacts,
                "backend": self.backend, "checks": self.checks}


def wants_separate(world: World) -> bool:
    v = world.spec.verifier
    return v is not None and v.mode == "separate"


async def hand_off(world: World, *, artifacts_dir: Path | None = None) -> tuple[WorldRef, Handoff]:
    """Quiesce and snapshot the policy's World and check custody. Raises ``CustodyViolation`` (the policy's outcome)."""
    spec = world.spec.verifier
    if spec is None:
        raise WorldError("hand_off needs WorldSpec.verifier")
    try:
        await world.quiesce()
    except WorldError as e:  # a policy process the guard could not kill would still be writing during the hand-off
        raise CustodyViolation(str(e)) from e
    final = await world.snapshot()
    start = world.episode_start or world.initial
    problems = check_snapshot(spec, start, final, world.store)
    if problems:
        raise CustodyViolation(problems)
    manifest = None
    if spec.artifacts and artifacts_dir is not None:
        manifest = collect_artifacts(spec, final, world.store, artifacts_dir)
    checks = [c for c, on in (("artifacts", bool(spec.artifacts)), ("protected", bool(spec.protected))) if on]
    return final, Handoff(world_ref=final.fingerprint, snapshot_step=final.step, apps={a: r.fingerprint for a, r in final.apps.items()},
                          artifacts=manifest, backend=spec.backend or world.spec.backend, checks=checks)


def _shared_roots(a: World, b: World) -> list[str]:
    out = []
    for app_id, app in a.apps.items():
        other = b.apps.get(app_id)
        if other is None or not (app.ctx.exec.is_local and other.ctx.exec.is_local):
            continue
        for key in app.PATH_PARAMS:
            x, y = app.state_params.get(key), other.state_params.get(key)
            if isinstance(x, str) and x == y:
                out.append(f"{app_id}.{key}={x}")
    return out


@asynccontextmanager
async def verifier_world(world: World, final: WorldRef, *, workdir: Path, close_source: bool = True) -> AsyncIterator[World]:
    """A fresh World holding ``final`` for the grader. Closes the policy's ``world`` first, except between the steps
    of a multi-step task (``close_source=False``): the policy is quiesced and the grader still reads only the copy."""
    spec = world.spec.verifier
    if spec is None:
        raise WorldError("verifier_world needs WorldSpec.verifier")
    log, start, episode = list(world.log), world.episode_start or world.initial, world.episode_id
    if close_source:
        await world.close()
    vw = World(world.spec, store=world.store, workdir=Path(workdir), pack_root=world.pack_root,
               backend=spec.backend or world.spec.backend, backend_params=spec.backend_params or None, interceptors=[])
    try:
        await vw.start()
        shared = _shared_roots(world, vw)  # the closed World's Apps still hold the roots they resolved
        if shared:
            raise WorldError(f"separate verification needs per-world state roots; these are shared with the policy's World: {shared}"
                             " (use relative roots, or a container backend)")
        await vw.restore(final)
        got = await vw.verifier_view().fingerprints()
        bad = {a: (fp[:12], final.apps[a].fingerprint[:12]) for a, fp in got.items() if fp != final.apps[a].fingerprint}
        if bad:
            raise WorldError(f"verifier World does not hold the handed-over state: {bad}")
        vw.log, vw.episode_start, vw.episode_id = log, start, episode
        if start is not None:
            vw.initial = start
        yield vw
    finally:
        await vw.close()
