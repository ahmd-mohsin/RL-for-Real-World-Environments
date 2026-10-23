"""Artifact provenance and the producer-family rule (swarm-pipeline/13 §2; 00 §3.2, V2).

The role-level family rule (``models.ModelBook.family_problems``) says no checker role shares a family with any builder
role. With the builders spread over three families (the ``swarm`` preset) that is still the rule, but it is checked on
roles, not on what a checker actually reads: a reviewer could be handed an artifact a later override built on its own
family. Here every artifact carries the families that produced it, and the rule is checked on the artifact.

- **Producers.** A machine state whose role is a builder (``models.BUILDER_ROLES``) is a producer: its settled outputs are
  stamped ``Producer(role, family, model, state)``. The provenance of an artifact is its producer's stamp plus the
  provenance of every input it read (``inputs[].from``), transitively, so a hidden test written from a plan carries the
  planner's family too.
- **Checkers.** A state whose role is a checker (``models.CHECKER_ROLES``: reviewer, realism, judge, second verifier,
  auditor, hackers, tie-break, the alternatives) refuses to run (``SameFamilyError``) when its own family is in the
  provenance of any input. ``static_problems`` checks a machine against a role -> family map before any run (the same
  closure over ``from``); ``ProvenanceLedger.check`` checks again at each activation with the family of the client that
  will actually run, so a later override cannot slip past.
- **V2.** ``isolated_from`` already keeps the verifier writer from reading the oracle's states; ``v2_problems`` adds the
  family half: the oracle builder and the first verifier writer are two families (the ``swarm`` preset).

Nothing here calls a model. The stamps are JSON (``Provenance.to_json``) and travel with the session records and the
journal (``factory.swarm_journal``), where gate failures are blamed on the producing role.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from crucible.factory.models import BUILDER_ROLES, CHECKER_ROLES, SameFamilyError, family_of


@dataclass(frozen=True)
class Producer:
    role: str
    family: str
    model: str = ""
    state: str = ""

    def to_json(self) -> dict[str, str]:
        return {"role": self.role, "family": self.family, "model": self.model, "state": self.state}


@dataclass(frozen=True)
class Provenance:
    producers: tuple[Producer, ...] = ()

    @property
    def families(self) -> set[str]:
        return {p.family for p in self.producers if p.family}

    def merge(self, *others: Provenance) -> Provenance:
        seen: dict[tuple[str, str, str, str], Producer] = {}
        for p in [*self.producers, *(q for o in others for q in o.producers)]:
            seen.setdefault((p.role, p.family, p.model, p.state), p)
        return Provenance(tuple(seen.values()))

    def to_json(self) -> dict[str, Any]:
        return {"families": sorted(self.families), "producers": [p.to_json() for p in self.producers]}

    @classmethod
    def from_json(cls, d: Mapping[str, Any] | None) -> Provenance:
        return cls(tuple(Producer(**p) for p in (d or {}).get("producers") or []))


def family_of_client(c: Any) -> str | None:
    """The family of a client: its ``family`` attribute (Tinker clients, ``ClaudeCodeClient``), a wrapped client's, else the
    family of its model id; ``None`` when nothing names one."""
    for obj in (c, getattr(c, "_client", None), getattr(c, "inner", None)):
        fam = getattr(obj, "family", None) if obj is not None else None
        if isinstance(fam, str) and fam:
            return fam
    model = str(getattr(c, "model", "") or "")
    if not model:
        return None
    from crucible.factory.audit import policy_model

    return family_of(policy_model(model))


def _from_state(ref: str) -> str | None:
    """``"plan.plan"`` -> ``"plan"``; ``"params.spec"`` -> None (a run parameter has no producer)."""
    head = str(ref).split(".", 1)[0]
    return None if head in ("params", "") else head


def direct_reads(machine: Mapping[str, Any], state_id: str, reads: Mapping[str, Iterable[str]] | None = None) -> list[str]:
    """The states ``state_id`` reads: its ``inputs[].from`` plus ``reads[state_id]`` (a handler that reads another state's
    results through services, not ports: the swarm machine's prompt, gates, second and hack read the build)."""
    by_id = {s["id"]: s for s in machine.get("states") or []}
    out = [src for inp in (by_id.get(state_id) or {}).get("inputs") or [] if (src := _from_state(inp.get("from", ""))) and src in by_id]
    return [*out, *(r for r in (reads or {}).get(state_id) or [] if r in by_id and r not in out)]


def upstream(machine: Mapping[str, Any], state_id: str, reads: Mapping[str, Iterable[str]] | None = None) -> set[str]:
    """Every state whose outputs flow into ``state_id``, transitively (not the state itself)."""
    out: set[str] = set()
    todo = [state_id]
    while todo:
        for src in direct_reads(machine, todo.pop(), reads):
            if src not in out:
                out.add(src)
                todo.append(src)
    return out


def static_problems(machine: Mapping[str, Any], families: Mapping[str, str | None], *, builder_roles: Iterable[str] = BUILDER_ROLES,
                    checker_roles: Iterable[str] = CHECKER_ROLES, reads: Mapping[str, Iterable[str]] | None = None) -> list[str]:
    """Every checker state of ``machine`` that reads (transitively) an artifact a builder state of its own family produced,
    under ``families`` (role -> family)."""
    builders, checkers = set(builder_roles), set(checker_roles)
    by_id = {s["id"]: s for s in machine.get("states") or []}
    out = []
    for sid, s in by_id.items():
        role = s.get("role")
        mine = families.get(role) if role else None
        if role not in checkers or not mine:
            continue
        for up in sorted(upstream(machine, sid, reads)):
            r = by_id[up].get("role")
            if r in builders and families.get(r) == mine:
                out.append(f"provenance: checker state {sid} (role {role!r}, family {mine!r}) reads {up}'s artifact, built by role {r!r} of its own family")
    return out


def v2_problems(families: Mapping[str, str | None]) -> list[str]:
    ob, vw = families.get("oracle_builder"), families.get("verifier_writer")
    return [f"V2: the oracle builder and the first verifier writer share family {ob!r}"] if ob and ob == vw else []


@dataclass
class ProvenanceLedger:
    """The run's provenance: state -> the provenance of its latest settled outputs. ``family_for(role)`` gives the family of
    the client that will run a role (``family_of_client`` of the actual client)."""

    machine: Mapping[str, Any]
    family_for: Callable[[str], str | None]
    model_for: Callable[[str], str] = lambda role: ""
    reads: Mapping[str, Iterable[str]] = field(default_factory=dict)
    states: dict[str, Provenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id = {s["id"]: s for s in self.machine.get("states") or []}

    def inputs_provenance(self, state_id: str) -> Provenance:
        """The union of the recorded provenance of the states ``state_id`` reads directly (each already transitive)."""
        prov = Provenance()
        for src in direct_reads(self.machine, state_id, self.reads):
            if src in self.states:
                prov = prov.merge(self.states[src])
        return prov

    def check(self, state_id: str, role: str | None) -> Provenance:
        """Raise ``SameFamilyError`` when a checker's family produced any of its inputs; return the inputs' provenance."""
        prov = self.inputs_provenance(state_id)
        if role in CHECKER_ROLES:
            mine = self.family_for(role)
            if mine and mine in prov.families:
                who = sorted({f"{p.role}@{p.state}" for p in prov.producers if p.family == mine})
                raise SameFamilyError(f"provenance: the {role} of state {state_id} (family {mine!r}) would check an artifact its own family "
                                      f"produced ({', '.join(who)})")
        return prov

    def record(self, state_id: str, role: str | None) -> Provenance:
        """After a state settles: a builder adds its own stamp to its inputs' provenance; any other state passes its inputs'
        provenance through (a deterministic gate's verdict carries what it judged)."""
        prov = self.inputs_provenance(state_id)
        if role in BUILDER_ROLES:
            prov = prov.merge(Provenance((Producer(role, self.family_for(role) or "", self.model_for(role), state_id),)))
        self.states[state_id] = prov
        return prov

    def guard(self, handlers: Mapping[str, Callable[[Any], Awaitable[Any]]]) -> dict[str, Callable[[Any], Awaitable[Any]]]:
        """``handlers`` wrapped: every activation is checked before it runs and recorded after it settles."""

        def wrap(fn: Callable[[Any], Awaitable[Any]]) -> Callable[[Any], Awaitable[Any]]:
            @functools.wraps(fn)
            async def run(ctx: Any) -> Any:
                self.check(ctx.state_id, ctx.role)
                res = await fn(ctx)
                self.record(ctx.state_id, ctx.role)
                return res

            return run

        return {k: wrap(v) for k, v in handlers.items()}

    def to_json(self) -> dict[str, Any]:
        return {sid: p.to_json() for sid, p in sorted(self.states.items())}
