"""World fusion for the ``fuse`` operator (m13 G-9; audit 06 §4.3): two Worlds' specs as one World.

``fuse_worlds(a, b, world=...)`` returns the fused ``WorldSpec`` and the app renames applied to ``b``:

- **Apps:** the union. An app id ``b`` shares with ``a`` is renamed ``<prefix><id>`` (and ``b``'s tool patterns, event
  tool actions and policies follow); two apps may never share a state path or a bundle destination, so a fused World
  never has two Apps writing one file (``ValueError``).
- **Actors:** policy actors with the same id merge (union of tool patterns, so the agent sees both systems); any other
  actor id ``b`` shares with ``a`` must be identical (one supervisor), otherwise it is an error: two different people
  under one name.
- **Clock / backend / network:** must agree (a fused episode has one clock and one sandbox); a missing clock takes
  the other's. Events and policies are the union, renamed with the apps.
- ``backend_params.bundles`` concatenate; other backend params must agree.

``crucible.forge.compose.fuse`` builds the task (components, cross-App checks); ``rename_apps`` renames a task's checks
and oracle with the same map.
"""

from __future__ import annotations

from typing import Any

from crucible.core.schema.world import WorldSpec


def _rn_tool(pat: str, renames: dict[str, str]) -> str:
    app, dot, rest = pat.partition(".")
    return f"{renames[app]}{dot}{rest}" if app in renames else pat


def fuse_worlds(a: WorldSpec, b: WorldSpec, *, world: str, prefix: str = "b_", description: str | None = None) -> tuple[WorldSpec, dict[str, str]]:
    if a.backend != b.backend:
        raise ValueError(f"backends differ ({a.backend} vs {b.backend}): a fused episode runs in one sandbox")
    if a.network != b.network or sorted(a.allowed_hosts) != sorted(b.allowed_hosts):
        raise ValueError("network policies differ: fuse Worlds with the same network policy")
    if a.clock and b.clock and a.clock.model_dump() != b.clock.model_dump():
        raise ValueError("clocks differ: a fused episode has one clock")
    a_ids = {x.id for x in a.apps}
    renames = {x.id: f"{prefix}{x.id}" for x in b.apps if x.id in a_ids}
    bad = [v for v in renames.values() if v in a_ids]
    if bad:
        raise ValueError(f"renamed app ids still collide: {bad}")

    def path(app) -> str | None:
        return (app.state.params or {}).get("path") or (app.state.params or {}).get("root")

    a_paths = {path(x) for x in a.apps} - {None}
    clash = [x.id for x in b.apps if path(x) in a_paths]
    if clash:
        raise ValueError(f"apps {clash} would share a state path with the other World")
    apps = [*a.apps, *[x.model_copy(update={"id": renames.get(x.id, x.id)}) for x in b.apps]]

    actors = {x.id: x for x in a.actors}
    for x in b.actors:
        x = x.model_copy(update={"tools": [_rn_tool(t, renames) for t in x.tools]})
        if x.id not in actors:
            actors[x.id] = x
        elif x.kind == "policy" and actors[x.id].kind == "policy":
            actors[x.id] = actors[x.id].model_copy(update={"tools": [*actors[x.id].tools, *[t for t in x.tools if t not in actors[x.id].tools]]})
        elif x.model_dump() != actors[x.id].model_dump():
            raise ValueError(f"actor {x.id!r} differs between the Worlds: two different actors under one name")

    events = list(a.events)
    for e in b.events:
        act = e.action
        act = act.model_copy(update={"tool": _rn_tool(act.tool, renames) if act.tool else act.tool,
                                     "grant": [_rn_tool(t, renames) for t in act.grant], "revoke": [_rn_tool(t, renames) for t in act.revoke]})
        if any(x.id == e.id for x in events):
            raise ValueError(f"event id {e.id!r} exists in both Worlds")
        events.append(e.model_copy(update={"action": act}))
    pols = list(a.policies) + [p.model_copy(update={"app": renames.get(p.app, p.app)}) for p in b.policies if not any(q.id == p.id for q in a.policies)]

    bp: dict[str, Any] = dict(a.backend_params)
    for k, v in b.backend_params.items():
        if k == "bundles":
            dests = {x.get("dest") for x in bp.get("bundles", [])}
            if any(x.get("dest") in dests for x in v):
                raise ValueError("bundle destinations collide")
            bp["bundles"] = [*bp.get("bundles", []), *v]
        elif k in bp and bp[k] != v:
            raise ValueError(f"backend param {k!r} differs between the Worlds")
        else:
            bp[k] = v
    fused = WorldSpec(world=world, description=description or f"fusion of {a.world} and {b.world}", backend=a.backend, backend_params=bp,
                      apps=apps, actors=list(actors.values()), clock=a.clock or b.clock, events=events, policies=pols, network=a.network,
                      allowed_hosts=list(a.allowed_hosts), intercept=a.intercept, isolation=a.isolation, limits=a.limits, seed=a.seed)
    return fused, renames
