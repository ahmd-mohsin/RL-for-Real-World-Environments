"""Composition operators for the Swarm Forge (m13 G-2; audit 06 §4.3; SWE-smith ``combine/`` pool rule as the pattern).

A composite is a task: a ``TaskSpec`` + ``CheckSet`` + oracle actions + ``NegativeSet``, plus its **component map**
(which oracle actions and which gates each component owns), so composites compose again and every composite can be
checked for *real* composition:

- **A14 necessity** (``accept/checks/composition.py``): dropping any component's actions must fail a gate it owns,
  otherwise the component is padding;
- **A15 complexity gain**: the composite must be measurably harder than its hardest component on the target rung;
- **A17 non-interference** (independent components): each component's oracle keeps its verdict in the composite.

Audit 06's first finding: no source shows that swarms make tasks "exponentially" harder. Composition multiplies
*candidates*, not difficulty, so these checks are what keep composition honest.

Operators:

- ``chain(T1, T2, …)``: one episode, the components in order. The instruction states each step; the checks are the
  union of the components' checks, namespaced per component (``c1.<gate>``); the oracle is ``o1 ++ o2 ++ …``.
  ``dependent=True`` declares that later components read state earlier ones produce (A17 then does not apply).
- **Scopes**: a component's "nothing else changed" invariants would forbid the other components' legitimate work; scope
  checks that carry their structure (``CheckSpec.spec["scope"]``: template + allow list) are widened to the union of
  every component's allow list (found on the first toy chain: each component's scope gate failed the other's actions).
- ``fan_in([T1..Tk], J)``: k independent subtasks plus an integration task J whose checks need all their outputs; the
  oracle is ``o1 ++ … ++ ok ++ oJ``; J owns its gates.

Each operator also emits a ``drop-component`` negative per component (the composite oracle with that component removed,
expected to fail at one of its gates): a property of the composite that A3 grades, so the grader keeps proving every
part matters. Components carry their source task hash, so editing a source invalidates its composites.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from crucible.forge.schema import CheckSet, CheckSpec, NegativeSpec, StatedProperty, TaskSpec


@dataclass
class Parts:
    spec: TaskSpec
    checks: CheckSet
    oracle: list[dict]
    negatives: list[NegativeSpec] = field(default_factory=list)
    source_hash: str = ""

    @classmethod
    def of(cls, task: Any) -> Parts:
        """From a forged task (``spec``, ``cs`` / ``checks``, ``oracle``, ``negatives``, ``task_hash``)."""
        cs = getattr(task, "cs", None) or CheckSet(task_id=task.task_id, checks=list(task.checks))
        return cls(task.spec, cs, list(task.oracle), list(getattr(task, "negatives", []) or []), getattr(task, "task_hash", ""))


@dataclass
class Component:
    id: str
    source: str
    source_hash: str
    actions: list[int]  # indices into the composite oracle
    gates: list[str]  # namespaced gates this component owns


@dataclass
class Composite(Parts):
    components: list[Component] = field(default_factory=list)
    operator: str = "chain"
    dependent: bool = True
    depth: int = 1

    def without(self, component_id: str) -> list[dict]:
        drop = set(next(c for c in self.components if c.id == component_id).actions)
        return [a for i, a in enumerate(self.oracle) if i not in drop]

    def component_oracle(self, component_id: str) -> list[dict]:
        keep = set(next(c for c in self.components if c.id == component_id).actions)
        return [a for i, a in enumerate(self.oracle) if i in keep]

    def tree(self) -> dict:
        return {"operator": self.operator, "depth": self.depth, "components": [{"id": c.id, "source": c.source, "hash": c.source_hash} for c in self.components]}


def _ns_checks(cs: CheckSet, prefix: str) -> list[CheckSpec]:
    return [c.model_copy(update={"id": f"{prefix}.{c.id}", "gate": f"{prefix}.{c.gate}", "states": [f"{prefix}.{s}" for s in c.states]}) for c in cs.checks]


def _ns_props(spec: TaskSpec, prefix: str) -> list[StatedProperty]:
    return [p.model_copy(update={"id": f"{prefix}.{p.id}"}) for p in spec.properties]


def render_scope(scope: dict) -> str:
    """A scope invariant's SQL from its structure: ``template`` with ``{allow}`` replaced by the allow list (quoted when
    ``quote``; ``empty`` stands in for an empty list so ``NOT IN ()`` is never produced)."""
    allow = list(scope.get("allow") or [])
    if not allow:
        allow = [scope.get("empty", "-" if scope.get("quote") else -1)]
    items = ", ".join(("'" + str(x).replace("'", "''") + "'") if scope.get("quote") else str(int(x)) for x in allow)
    return scope["template"].replace("{allow}", items)


def _widen_scopes(checks: list[CheckSpec]) -> list[CheckSpec]:
    """Scope invariants ("nothing outside the task changed") of different components forbid each other's legitimate work.
    Per template, the union of every component's allow list replaces each one's (structured ``spec.scope``)."""
    union: dict[str, list] = {}
    for c in checks:
        sc = (c.spec or {}).get("scope")
        if sc:
            u = union.setdefault(sc["template"], [])
            u += [x for x in sc.get("allow") or [] if x not in u]
    out = []
    for c in checks:
        sc = (c.spec or {}).get("scope")
        if sc:
            wide = {**sc, "allow": union[sc["template"]]}
            c = c.model_copy(update={"sql": render_scope(wide), "spec": {**c.spec, "scope": wide}})
        out.append(c)
    return out


def _depth(p: Parts) -> int:
    return p.depth if isinstance(p, Composite) else 0


def _assemble(task_id: str, operator: str, parts: list[tuple[str, Parts]], *, steps: list[str], dependent: bool, title: str) -> Composite:
    oracle: list[dict] = []
    comps: list[Component] = []
    checks: list[CheckSpec] = []
    props: list[StatedProperty] = []
    literals: list[str] = []
    provenance: list[str] = []
    for cid, p in parts:
        start = len(oracle)
        oracle += [dict(a) for a in p.oracle]
        cs = _ns_checks(p.checks, cid)
        checks += cs
        props += _ns_props(p.spec, cid)
        literals += [x for x in p.spec.literals if x not in literals]
        provenance += [x for x in p.spec.provenance if x not in provenance]
        comps.append(Component(cid, p.spec.task_id, p.source_hash or _hash(p), list(range(start, len(oracle))), sorted({c.gate for c in cs})))
    checks = _widen_scopes(checks)
    spec = TaskSpec(task_id=task_id, title=title, instruction="\n\n".join(steps), properties=props, literals=literals, provenance=provenance,
                    trap="; ".join(p.spec.trap for _, p in parts if p.spec.trap) or None, horizon=max((p.spec.horizon for _, p in parts), default="H1"),
                    difficulty={"composition": {"operator": operator, "components": [c for c, _ in parts]}})
    comp = Composite(spec, CheckSet(task_id=task_id, checks=checks), oracle, [], "", components=comps, operator=operator, dependent=dependent,
                     depth=1 + max(_depth(p) for _, p in parts))
    negs: list[NegativeSpec] = []
    for c in comps:  # every part must matter: the composite oracle without it fails at a gate it owns
        negs.append(NegativeSpec(id=f"{task_id}:drop-{c.id}", task_id=task_id, kind="defect_mutant", actions=comp.without(c.id),
                                 expected_gate=c.gates[0] if c.gates else None, provenance="compose:drop_component",
                                 description=f"the composite oracle without component {c.id} ({c.source})"))
    for cid, p in parts:  # the components' own negatives, inside the composite (the other components' oracle around them)
        own = next(c for c in comps if c.id == cid)
        before = [a for i, a in enumerate(oracle) if i < own.actions[0]] if own.actions else []
        after = [a for i, a in enumerate(oracle) if own.actions and i > own.actions[-1]]
        for n in p.negatives:
            if n.kind == "valid_alternative" or n.actions is None:
                continue
            negs.append(n.model_copy(update={"id": f"{task_id}:{cid}:{n.id}", "task_id": task_id, "actions": before + list(n.actions) + after,
                                             "expected_gate": f"{cid}.{n.expected_gate}" if n.expected_gate else None}))
    comp.negatives = negs
    comp.source_hash = _hash(comp)
    return comp


def _hash(p: Parts) -> str:
    return hashlib.sha256(json.dumps([p.spec.model_dump(mode="json"), p.checks.model_dump(mode="json"), p.oracle], sort_keys=True, default=str).encode()).hexdigest()[:24]


def chain(*parts: Parts, task_id: str, dependent: bool = False) -> Composite:
    if len(parts) < 2:
        raise ValueError("chain needs at least two components")
    named = [(f"c{i + 1}", p) for i, p in enumerate(parts)]
    steps = [f"Step {i + 1}: {p.spec.instruction}" for i, (_, p) in enumerate(named)]
    return _assemble(task_id, "chain", named, steps=["Do all of the following, in order.", *steps], dependent=dependent,
                     title=" ; ".join(p.spec.title for p in parts))


def fan_in(parts: list[Parts], join: Parts, *, task_id: str) -> Composite:
    if len(parts) < 2:
        raise ValueError("fan_in needs at least two independent components")
    named = [(f"c{i + 1}", p) for i, p in enumerate(parts)] + [("join", join)]
    steps = [f"Part {i + 1}: {p.spec.instruction}" for i, (_, p) in enumerate(named[:-1])] + [f"Finally: {join.spec.instruction}"]
    return _assemble(task_id, "fan_in", named, steps=["Complete every part, then the final step.", *steps], dependent=True,
                     title=" || ".join(p.spec.title for p in parts) + f" -> {join.spec.title}")
