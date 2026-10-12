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
    cross: list[str] = field(default_factory=list)  # gates no single component owns (fuse: the cross-App interaction)

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


def _assemble(task_id: str, operator: str, parts: list[tuple[str, Parts]], *, steps: list[str], dependent: bool, title: str,
              cross: list[CheckSpec] = (), order: list[int] | None = None, place: Any = None,
              cross_props: list[StatedProperty] = ()) -> Composite:
    """``cross``: checks no single component owns (gate ``x.<gate>``: the fused interaction). ``order``: a permutation of
    the concatenated oracle (nest). ``place(cid, actions)``: a component's negative inside the composite (default: the
    other components' oracle around it)."""
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
    if order is not None:
        if sorted(order) != list(range(len(oracle))):
            raise ValueError("order must be a permutation of the composite oracle")
        new_at = {old: new for new, old in enumerate(order)}
        oracle = [oracle[i] for i in order]
        for c in comps:
            c.actions = sorted(new_at[i] for i in c.actions)
    # a cross check grades a property the composite states (M8 audit V14: the fused task's cross check graded nothing
    # the instruction said, so it could never pass A7): its sentence joins the instruction, its states are namespaced
    stated = {p.id for p in cross_props}
    missing = sorted({x for c in cross for x in (c.states or ["<none>"])} - stated)
    if cross and missing:
        raise ValueError(f"cross checks grade properties no cross property states: {missing}")
    props += [p.model_copy(update={"id": f"x.{p.id}"}) for p in cross_props]
    steps = [*steps, *(p.text if p.text.endswith(".") else p.text + "." for p in cross_props)]
    checks = _widen_scopes(checks + [c.model_copy(update={"id": f"x.{c.id}", "gate": f"x.{c.gate}", "states": [f"x.{x}" for x in c.states]})
                                     for c in cross])
    spec = TaskSpec(task_id=task_id, title=title, instruction="\n\n".join(steps), properties=props, literals=literals, provenance=provenance,
                    trap="; ".join(p.spec.trap for _, p in parts if p.spec.trap) or None, horizon=max((p.spec.horizon for _, p in parts), default="H1"),
                    difficulty={"composition": {"operator": operator, "components": [c for c, _ in parts]}})
    comp = Composite(spec, CheckSet(task_id=task_id, checks=checks), oracle, [], "", components=comps, operator=operator, dependent=dependent,
                     depth=1 + max(_depth(p) for _, p in parts))
    negs: list[NegativeSpec] = []
    for c in comps:  # every part must matter: the composite oracle without it fails at a gate it owns
        # the gate expected to fail is one with a positive outcome check (a predicate needs the dropped actions; an invariant
        # such as "nothing restocked" holds on a no-op, found on the first fused task)
        positive = sorted({k.gate for k in checks if k.gate in c.gates and k.kind == "predicate"})
        negs.append(NegativeSpec(id=f"{task_id}:drop-{c.id}", task_id=task_id, kind="defect_mutant", actions=comp.without(c.id),
                                 expected_gate=(positive or c.gates or [None])[0], provenance="compose:drop_component",
                                 description=f"the composite oracle without component {c.id} ({c.source})"))

    def around(cid: str, actions: list[dict]) -> list[dict]:
        own = next(c for c in comps if c.id == cid)
        before = [a for i, a in enumerate(oracle) if own.actions and i < own.actions[0]]
        after = [a for i, a in enumerate(oracle) if own.actions and i > own.actions[-1]]
        return before + list(actions) + after

    for cid, p in parts:  # the components' own negatives, inside the composite (the other components' oracle around them)
        for n in p.negatives:
            if n.kind == "valid_alternative" or n.actions is None:
                continue
            negs.append(n.model_copy(update={"id": f"{task_id}:{cid}:{n.id}", "task_id": task_id, "actions": (place or around)(cid, list(n.actions)),
                                             "expected_gate": f"{cid}.{n.expected_gate}" if n.expected_gate else None}))
    comp.negatives = negs
    comp.cross = sorted({c.gate for c in comp.checks.checks if c.gate.startswith("x.")})
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


def rename_apps(p: Parts, renames: dict[str, str]) -> Parts:
    """A task's app ids renamed (``fuse`` when two Worlds share an app id): its checks' ``app`` and its oracle's
    ``<app>.<tool>`` calls, negatives included."""
    if not renames:
        return p

    def acts(xs):
        out = []
        for a in xs or []:
            fn = a.get("function_name") or ""
            app, _, tool = fn.partition(".")
            out.append({**a, "function_name": f"{renames[app]}.{tool}"} if tool and app in renames else dict(a))
        return out

    cs = p.checks.model_copy(update={"checks": [c.model_copy(update={"app": renames.get(c.app, c.app)}) if c.app else c for c in p.checks.checks]})
    negs = [n.model_copy(update={"actions": acts(n.actions)}) if n.actions is not None else n for n in p.negatives]
    return Parts(p.spec, cs, acts(p.oracle), negs, p.source_hash)


def fuse(a: Parts, b: Parts, *, task_id: str, cross: list[CheckSpec] = (), renames: dict[str, str] | None = None,
         cross_props: list[StatedProperty] = ()) -> Composite:
    """Two tasks from different Worlds, in one episode of the fused World (``forge.fuse.fuse_worlds``). The components
    are independent (A17 applies); ``cross`` checks span both Apps (gate ``x.<gate>``): the interaction a fused task adds
    and neither part has."""
    b = rename_apps(b, renames or {})
    return _assemble(task_id, "fuse", [("c1", a), ("c2", b)], dependent=False, cross=list(cross), cross_props=list(cross_props),
                     steps=["Complete both of the following; they touch different systems.", f"Part 1: {a.spec.instruction}", f"Part 2: {b.spec.instruction}"],
                     title=f"{a.spec.title} (+) {b.spec.title}")


def nest(outer: Parts, inner: Parts, *, task_id: str, at: int, cross: list[CheckSpec] = (), cross_props: list[StatedProperty] = ()) -> Composite:
    """``inner`` done in the middle of ``outer``: after outer's first ``at`` oracle steps (a subtask that interrupts a
    larger one). Dependent by construction; the outer component's actions are not contiguous."""
    if not 0 < at < len(outer.oracle):
        raise ValueError(f"nest point {at} must fall strictly inside the outer oracle (1..{len(outer.oracle) - 1})")
    n_out, n_in = len(outer.oracle), len(inner.oracle)
    order = list(range(at)) + list(range(n_out, n_out + n_in)) + list(range(at, n_out))
    inner_oracle = [dict(x) for x in inner.oracle]
    outer_oracle = [dict(x) for x in outer.oracle]

    def place(cid: str, actions: list[dict]) -> list[dict]:
        if cid == "outer":
            k = min(at, len(actions))
            return actions[:k] + inner_oracle + actions[k:]
        return outer_oracle[:at] + actions + outer_oracle[at:]

    return _assemble(task_id, "nest", [("outer", outer), ("inner", inner)], dependent=True, cross=list(cross), order=order, place=place,
                     cross_props=list(cross_props),
                     steps=[f"Main task: {outer.spec.instruction}", f"Partway through (after the first {at} steps of the main task), also: {inner.spec.instruction}"],
                     title=f"{outer.spec.title} [{inner.spec.title}]")
