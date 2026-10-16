"""Executed requirement probes (05 §6.2, R2): for each requirement of the ledger, ≥ 1 **mutant** (wrong on exactly that
requirement, right on the others) and ≥ 1 **valid alternative** (right, but not the oracle), all run under the verifier.
The verifier must kill every mutant and pass every alternative. A survivor names the requirement whose check is weak;
a rejected alternative names the checks that grade the oracle's internals instead of the stated property.

**Probe library** (``LIBRARY``), keyed by ``Requirement.kind``:

| kind | mutant operators (tool-call oracles) | alternative operators |
|---|---|---|
| ``laziness`` | drop a footprint write / message (``accept.mutation.actions``: ``drop_step``, ``drop_message``) | extra read, reworded free text |
| ``tolerance`` | shift a footprint quantity beyond the tolerance (toward zero) | shift it within the declared tolerance (toward zero) |
| ``behaviour`` | swap an enum / id argument, act before the message that must precede (``enum_swap``, ``reorder``, id ``numeric_shift``) | extra read, reworded free text |
| ``idempotence`` | repeat a footprint write | repeat a read |
| ``minimality`` | an extra change on the footprint entity | extra read |
| ``side_effect_scope`` | the same write on another entity, appended | a read of another entity |
| ``format`` | the footprint's string reformatted (case, separators) | model-authored only |
| ``compiled_path`` | model-authored only | model-authored only |

The **footprint** of a requirement is the oracle steps it is about: ``ProbeContext.footprints`` when the spec states it,
else the steps whose structured arguments and tool name share the requirement's wording, weighted by rarity across steps
(an id every step carries counts little; free text and message bodies are not matched). Where an operator yields too
little, or the oracle is not a tool-call list (patches, answers), the **LLM author** role (``LLMProbeAuthor``, an
injectable ``JsonClient``) writes the probes; it sees the ledger and the oracle, never the verifier.

Mutants whose operator can be inert (a repeated idempotent write, a write on an entity that does not exist) and whose
end state equals the oracle's are *equivalent*: excluded from the kill rate, as A3 excludes ``drop_read``.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from crucible.factory.floors import Frac, Graded, JsonClient, call, graded_from, is_actions, requirements_failed

PROBE_KINDS = ("laziness", "tolerance", "compiled_path", "behaviour", "idempotence", "minimality", "side_effect_scope", "format")
Role = Literal["mutant", "alternative"]


@dataclass
class Probe:
    id: str
    requirement_id: str
    role: Role
    operator: str
    solution: Any
    description: str = ""
    source: Literal["deterministic", "llm"] = "deterministic"
    can_be_equivalent: bool = False  # an inert application leaves the oracle's end state: equivalent, not a survivor


@dataclass
class ProbeContext:
    oracle: Any
    prompt: str = ""
    requirements: Sequence[Any] = ()  # every requirement (the LLM author is told what must stay right)
    tolerances: dict[str, float] = field(default_factory=dict)  # requirement id -> declared absolute tolerance
    footprints: dict[str, list[int]] = field(default_factory=dict)  # requirement id -> oracle step indices (spec-stated)
    other_ids: dict[str, list] = field(default_factory=dict)  # argument name -> other entities in the World (scope probes)
    read_calls: list[dict] = field(default_factory=list)  # read-only calls the World offers (alternatives)


@dataclass
class ProbeConfig:
    min_mutants: int = 1  # per requirement (05 §6.2)
    min_alternatives: int = 1
    max_per_requirement: int = 6  # per role, deterministic operators
    kill_at: float = 0.0  # a mutant is killed when its reward is <= this (A3: reward 0)
    pass_at: float = 1.0  # an alternative passes at full reward
    kill_rate_min: float = 1.0  # v1: 100% of mutants killed
    alternative_rate_min: float = 1.0  # v1: 100% of alternatives pass
    footprint_min_score: float = 1.0


# --- footprints -----------------------------------------------------------------------------------------------------


def _free_text(v: Any) -> bool:
    return isinstance(v, str) and len(v.split()) >= 4


def _is_id(name: str) -> bool:
    return name == "id" or name.endswith(("_id", "_ids"))


def _is_target(name: str) -> bool:
    """An argument naming the entity a call acts on (an id or a path)."""
    return _is_id(name) or name in ("path", "file", "filename")


def _step_tokens(a: dict) -> set[str]:
    from crucible.factory.ledger import tokens

    out: set[str] = set()
    fn = str(a.get("function_name") or "")
    if fn:
        out |= tokens(" ".join(re.split(r"[._]", fn)))
    elif a.get("to"):
        out |= tokens(str(a["to"]))
    for v in (a.get("arguments") or {}).values():
        if isinstance(v, (str, int, float)) and not isinstance(v, bool) and not _free_text(v):
            out |= tokens(str(v))
    return out


def _writes(oracle: Sequence[dict]) -> list[int]:
    from crucible.forge.hacker import _is_read

    return [i for i, a in enumerate(oracle) if a.get("function_name") and not _is_read(a)]


def footprint(req: Any, ctx: ProbeContext, *, min_score: float = 1.0) -> list[int]:
    """The oracle steps a requirement is about (see the module doc)."""
    if req.id in ctx.footprints:
        return list(ctx.footprints[req.id])
    oracle = ctx.oracle
    if not (isinstance(oracle, list) and is_actions(oracle)):
        return []
    if req.kind == "side_effect_scope":
        return _writes(oracle)
    from crucible.factory.ledger import tokens

    want = tokens(req.text)
    toks = [_step_tokens(a) for a in oracle]
    df: dict[str, int] = {}
    for ts in toks:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    scores = [sum(1.0 / df[t] for t in ts & want) for ts in toks]
    top = max(scores, default=0.0)
    if top < min_score:
        return []
    return [i for i, s in enumerate(scores) if s >= max(min_score, top / 2)]


# --- deterministic operators (tool-call oracles) --------------------------------------------------------------------


def _step_of(mutant_id: str) -> set[int]:
    """Oracle step indices a mutant of ``accept.mutation.actions`` touched (``drop_step/3``, ``reorder/4-before-2``)."""
    rest = mutant_id.split("/", 1)[1] if "/" in mutant_id else ""
    return {int(x) for x in re.findall(r"\d+", rest.split("/")[0])}


def _from_accept(req: Any, oracle: list[dict], fp: list[int], families: set[str], *, id_args: bool | None = None) -> list[Probe]:
    from crucible.accept.mutation.actions import generate

    out = []
    for m in generate(oracle, limit=10**6):
        if m.family not in families or not (_step_of(m.id) & set(fp)):
            continue
        if m.family == "numeric_shift" and id_args is not None:
            arg = m.id.split("/")[-1]
            if _is_id(arg) != id_args:
                continue
        out.append(Probe(f"{req.id}/{m.id}", req.id, "mutant", m.family, m.solution, m.description, can_be_equivalent=m.family == "drop_read"))
    return out


def _read_for(oracle: list[dict], ctx: ProbeContext, *, other_entity: bool = False) -> list[dict]:
    from crucible.forge.hacker import _is_read

    reads = [copy.deepcopy(a) for a in oracle if a.get("function_name") and _is_read(a)] + [copy.deepcopy(a) for a in ctx.read_calls]
    if other_entity:
        out = []
        for r in reads:
            for k, v in (r.get("arguments") or {}).items():
                if _is_target(k):
                    alt = _other_id(k, v, oracle, ctx)
                    if alt is not None:
                        r2 = copy.deepcopy(r)
                        r2["arguments"][k] = alt
                        out.append(r2)
                        break
        return out
    return reads


def _other_id(name: str, v: Any, oracle: list[dict], ctx: ProbeContext) -> Any:
    pool = [x for x in ctx.other_ids.get(name, []) if x != v]
    if pool:
        return pool[0]
    used = sorted({a.get("arguments", {}).get(name) for a in oracle if isinstance(a.get("arguments", {}).get(name), type(v))} - {v}, key=repr)
    if used:
        return used[0]
    if isinstance(v, int) and not isinstance(v, bool):
        return v + 1
    return None


def _with_id(a: dict, i: int) -> dict:
    a = copy.deepcopy(a)
    if a.get("function_name"):
        a["tool_call_id"] = f"probe{i}"
    return a


def _alt_extra_read(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    oracle = ctx.oracle
    reads = _read_for(oracle, ctx)
    if not reads or not fp:
        return []
    at = fp[0]
    sol = oracle[:at] + [_with_id(reads[0], 900)] + oracle[at:]
    return [Probe(f"{req.id}/alt/extra_read/{at}", req.id, "alternative", "extra_read", sol, f"read before step {at}")]


def _alt_reword(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    oracle, out = ctx.oracle, []
    for i in fp:
        args = oracle[i].get("arguments") or {}
        for k in sorted(args):
            if _free_text(args[k]):
                sol = copy.deepcopy(oracle)
                sol[i]["arguments"][k] = f"{req.text.strip().rstrip('.')}: done."
                out.append(Probe(f"{req.id}/alt/reword/{i}/{k}", req.id, "alternative", "reword", sol, f"{k} reworded from the requirement's own words"))
                break
    return out


def _generic_alternatives(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    return _alt_reword(req, ctx, fp) + _alt_extra_read(req, ctx, fp)


def _quantities(a: dict) -> list[str]:
    return [k for k, v in sorted((a.get("arguments") or {}).items()) if isinstance(v, (int, float)) and not isinstance(v, bool) and not _is_id(k)]


def _shift(v: float, d: float) -> float:
    return round(v + d, 6)


def _toward_zero(v: float, d: float) -> float:
    """Shift a quantity by ``d`` toward zero: a stated amount is often also a cap (a refund cannot exceed the order), so
    moving down keeps the call valid and leaves only the tolerance in question."""
    return _shift(v, -d if v > 0 else d)


def m_laziness(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    return _from_accept(req, ctx.oracle, fp, {"drop_step", "drop_message"})


def m_tolerance(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    tol = float(ctx.tolerances.get(req.id, 0.0))
    out = []
    for i in fp:
        for k in _quantities(ctx.oracle[i]):
            v = ctx.oracle[i]["arguments"][k]
            sol = copy.deepcopy(ctx.oracle)
            sol[i]["arguments"][k] = _toward_zero(float(v), max(10 * tol, 0.1 * abs(float(v)), 0.01))
            out.append(Probe(f"{req.id}/beyond_tolerance/{i}/{k}", req.id, "mutant", "beyond_tolerance", sol, f"{k}: {v} -> {sol[i]['arguments'][k]} (tolerance {tol})"))
    return out


def a_tolerance(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    if req.id not in ctx.tolerances:  # no declared tolerance: what "within" means is the model author's call
        return []
    tol = float(ctx.tolerances[req.id])
    out = []
    for i in fp:
        for k in _quantities(ctx.oracle[i]):
            v = ctx.oracle[i]["arguments"][k]
            sol = copy.deepcopy(ctx.oracle)
            sol[i]["arguments"][k] = _toward_zero(float(v), tol / 2)
            out.append(Probe(f"{req.id}/alt/within_tolerance/{i}/{k}", req.id, "alternative", "within_tolerance", sol, f"{k}: {v} -> {sol[i]['arguments'][k]}"))
    return out


def m_behaviour(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    return _from_accept(req, ctx.oracle, fp, {"enum_swap", "reorder"}) + _from_accept(req, ctx.oracle, fp, {"numeric_shift"}, id_args=True)


def m_idempotence(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    writes = set(_writes(ctx.oracle))
    out = []
    for i in fp:
        if i in writes:
            sol = ctx.oracle[:i + 1] + [_with_id(ctx.oracle[i], 800 + i)] + ctx.oracle[i + 1:]
            out.append(Probe(f"{req.id}/repeat_write/{i}", req.id, "mutant", "repeat_write", sol, f"step {i} applied twice", can_be_equivalent=True))
    return out


def a_idempotence(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    reads = _read_for(ctx.oracle, ctx)
    if not reads:
        return []
    at = fp[0] if fp else len(ctx.oracle)
    sol = ctx.oracle[:at] + [_with_id(reads[0], 901), _with_id(reads[0], 902)] + ctx.oracle[at:]
    return [Probe(f"{req.id}/alt/repeat_read/{at}", req.id, "alternative", "repeat_read", sol, "a read repeated")]


def _altered(v: Any) -> Any:
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return _shift(float(v), max(0.1 * abs(float(v)), 1.0)) if isinstance(v, float) else v + 1
    if isinstance(v, str):
        return f"{v} (also changed)" if _free_text(v) else (v[::-1] if len(v) > 1 else v + "x")
    return v


def m_minimality(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    writes = set(_writes(ctx.oracle))
    out = []
    for i in fp:
        if i not in writes:
            continue
        a = _with_id(ctx.oracle[i], 700 + i)
        changed = [k for k in sorted(a.get("arguments") or {}) if not _is_target(k)]
        if not changed:
            continue
        a["arguments"][changed[0]] = _altered(a["arguments"][changed[0]])
        sol = ctx.oracle[:i] + [a] + ctx.oracle[i:]  # an extra change on the same entity, before the oracle's own step
        out.append(Probe(f"{req.id}/extra_change/{i}", req.id, "mutant", "extra_change", sol, f"an extra {a['function_name']} ({changed[0]})", can_be_equivalent=True))
    return out


def m_scope(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    writes = set(_writes(ctx.oracle))
    out = []
    for i in fp:
        if i not in writes:
            continue
        a = _with_id(ctx.oracle[i], 600 + i)
        ids = [k for k in sorted(a.get("arguments") or {}) if _is_target(k)]
        if not ids:
            continue
        k = ids[0]
        alt = _other_id(k, a["arguments"][k], ctx.oracle, ctx)
        if alt is None and isinstance(a["arguments"][k], str):
            alt = "other_" + a["arguments"][k]
        if alt is None:
            continue
        a["arguments"][k] = alt
        out.append(Probe(f"{req.id}/collateral_write/{i}", req.id, "mutant", "collateral_write", ctx.oracle + [a], f"{a['function_name']} also on {k}={alt!r}",
                         can_be_equivalent=True))
    return out


def a_scope(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    reads = _read_for(ctx.oracle, ctx, other_entity=True)
    if not reads:
        return []
    return [Probe(f"{req.id}/alt/read_elsewhere", req.id, "alternative", "read_elsewhere", [_with_id(reads[0], 903)] + ctx.oracle, "a read of another entity")]


def _reformat(v: str) -> str:
    out = re.sub(r"[-_,.:/ ]", "", v).swapcase() if not _free_text(v) else v.upper().replace(".", "").replace(",", "")
    return out if out != v else v + " "


def m_format(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    out = []
    for i in fp:
        args = ctx.oracle[i].get("arguments") or {}
        for k in sorted(args):
            if isinstance(args[k], str) and not _is_target(k):
                sol = copy.deepcopy(ctx.oracle)
                sol[i]["arguments"][k] = _reformat(args[k])
                out.append(Probe(f"{req.id}/reformat/{i}/{k}", req.id, "mutant", "reformat", sol, f"{k}: {args[k]!r} -> {sol[i]['arguments'][k]!r}", can_be_equivalent=True))
    return out


def _none(req: Any, ctx: ProbeContext, fp: list[int]) -> list[Probe]:
    return []


@dataclass(frozen=True)
class Operator:
    kind: str
    mutants: Callable[[Any, ProbeContext, list[int]], list[Probe]]
    alternatives: Callable[[Any, ProbeContext, list[int]], list[Probe]]


LIBRARY: dict[str, Operator] = {
    "laziness": Operator("laziness", m_laziness, _generic_alternatives),
    "tolerance": Operator("tolerance", m_tolerance, a_tolerance),
    "behaviour": Operator("behaviour", m_behaviour, _generic_alternatives),
    "idempotence": Operator("idempotence", m_idempotence, a_idempotence),
    "minimality": Operator("minimality", m_minimality, _alt_extra_read),
    "side_effect_scope": Operator("side_effect_scope", m_scope, a_scope),
    "format": Operator("format", m_format, _none),
    "compiled_path": Operator("compiled_path", _none, _none),
}


# --- the LLM author role --------------------------------------------------------------------------------------------

AUTHOR_SYSTEM = """You write probe solutions that test a task's verifier. You never see the verifier.
- A MUTANT is a solution that is wrong on exactly the named requirement and right on every other requirement.
- A VALID ALTERNATIVE is a solution that satisfies every requirement but differs from the reference solution.
Solutions have the same shape as the reference solution (the same JSON structure: a list of tool calls and messages, a
patch, or an answer). Reply with JSON: {"probes": [{"solution": ..., "description": "what is wrong or different"}]}."""


class LLMProbeAuthor:
    """The mutant builder's model role for requirements the deterministic library cannot probe."""

    def __init__(self, client: JsonClient, *, max_tokens: int = 4000):
        self.client, self.max_tokens = client, max_tokens
        self.rejected: list[dict] = []  # replies that were not usable probes, with why

    async def author(self, req: Any, ctx: ProbeContext, role: Role, n: int) -> list[Probe]:
        others = [{"id": r.id, "kind": r.kind, "text": r.text} for r in ctx.requirements if r.id != req.id]
        user = json.dumps({"want": f"{n} {'mutant' if role == 'mutant' else 'valid alternative'} solution(s)", "task_prompt": ctx.prompt,
                           "requirement": {"id": req.id, "kind": req.kind, "text": req.text}, "other_requirements": others,
                           "reference_solution": ctx.oracle}, default=str)
        try:
            reply = await self.client.complete_json([{"role": "system", "content": AUTHOR_SYSTEM}, {"role": "user", "content": user}], max_tokens=self.max_tokens)
        except Exception as e:  # noqa: BLE001 - a failed role call is recorded, never a crash of the stage
            self.rejected.append({"requirement": req.id, "role": role, "why": f"client error: {e}"[:300]})
            return []
        out = []
        for j, p in enumerate((reply or {}).get("probes") or []):
            sol = p.get("solution") if isinstance(p, dict) else None
            why = _shape_problem(sol, ctx.oracle)
            if why:
                self.rejected.append({"requirement": req.id, "role": role, "why": why})
                continue
            out.append(Probe(f"{req.id}/{'alt/' if role == 'alternative' else ''}llm/{j}", req.id, role, "llm", sol, str(p.get("description", ""))[:300], source="llm",
                             can_be_equivalent=role == "mutant"))
            if len(out) >= n:
                break
        return out


def _shape_problem(sol: Any, oracle: Any) -> str | None:
    if sol is None:
        return "no solution"
    if isinstance(oracle, list) and is_actions(oracle):
        return None if isinstance(sol, list) and (sol == [] or is_actions(sol)) else "not a tool-call list like the reference"
    if isinstance(oracle, str) and not isinstance(sol, str):
        return "not text like the reference"
    if sol == oracle:
        return "identical to the reference"
    return None


# --- build and run --------------------------------------------------------------------------------------------------


async def build_probes(ledger: Any, ctx: ProbeContext, *, author: LLMProbeAuthor | None = None, library: dict[str, Operator] | None = None,
                       config: ProbeConfig | None = None) -> tuple[list[Probe], dict[str, list[str]]]:
    """Probes for every requirement of ``ledger``, deterministic first, the model author for what is short. Returns the
    probes and per-requirement notes (no operator for the kind, short of the minimum, ...)."""
    cfg, lib = config or ProbeConfig(), library if library is not None else LIBRARY
    if not ctx.requirements:
        ctx.requirements = list(ledger.requirements)
    actions = isinstance(ctx.oracle, list) and is_actions(ctx.oracle)
    probes: list[Probe] = []
    notes: dict[str, list[str]] = {}
    seen = {repr(ctx.oracle)}
    for req in ledger.requirements:
        op = lib.get(req.kind)
        note = notes.setdefault(req.id, [])
        if op is None:
            note.append(f"no probe operator for kind {req.kind!r}")
        fp = footprint(req, ctx, min_score=cfg.footprint_min_score) if actions else []
        if actions and op is not None and not fp:
            note.append("empty footprint: no oracle step matches the requirement")
        for role, want in (("mutant", cfg.min_mutants), ("alternative", cfg.min_alternatives)):
            got = []
            if actions and op is not None and fp:
                for p in (op.mutants if role == "mutant" else op.alternatives)(req, ctx, fp):
                    if repr(p.solution) not in seen and len(got) < cfg.max_per_requirement:
                        seen.add(repr(p.solution))
                        got.append(p)
            if len(got) < want and author is not None:
                for p in await author.author(req, ctx, role, want - len(got)):  # type: ignore[arg-type]
                    if repr(p.solution) not in seen:
                        seen.add(repr(p.solution))
                        got.append(p)
            if len(got) < want:
                note.append(f"{len(got)} {role}s built, need {want}")
            probes += got
    return probes, {k: v for k, v in notes.items() if v}


@dataclass
class RequirementProbes:
    requirement_id: str
    kill: Frac
    alternatives: Frac
    survivors: list[dict] = field(default_factory=list)
    rejected_alternatives: list[dict] = field(default_factory=list)
    equivalent: list[str] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)
    off_target: list[dict] = field(default_factory=list)  # killed, but by checks of other requirements only


@dataclass
class ProbeReport:
    per_requirement: dict[str, RequirementProbes]
    oracle_ok: bool | None
    notes: dict[str, list[str]] = field(default_factory=dict)
    config: ProbeConfig = field(default_factory=ProbeConfig)

    @property
    def mutant_kill(self) -> tuple[int, int]:
        return (sum(r.kill.k for r in self.per_requirement.values()), sum(r.kill.n for r in self.per_requirement.values()))

    @property
    def alternatives_pass(self) -> tuple[int, int]:
        return (sum(r.alternatives.k for r in self.per_requirement.values()), sum(r.alternatives.n for r in self.per_requirement.values()))

    @property
    def weak_requirements(self) -> list[str]:
        """Requirements with a surviving mutant: their checks are too weak."""
        return sorted(r for r, x in self.per_requirement.items() if x.survivors)

    @property
    def strict_requirements(self) -> list[str]:
        """Requirements with a rejected valid alternative: a check grades something the requirement does not state."""
        return sorted(r for r, x in self.per_requirement.items() if x.rejected_alternatives)

    def reasons(self) -> list[str]:
        c, out = self.config, []
        if self.oracle_ok is False:
            out.append("the oracle does not pass the verifier: probe results mean nothing")
        for rid, x in sorted(self.per_requirement.items()):
            if x.kill.n < c.min_mutants:
                out.append(f"requirement {rid}: {x.kill.n} scored mutants, need {c.min_mutants} (unprobed)")
            elif (x.kill.rate or 0.0) < c.kill_rate_min:
                out.append(f"requirement {rid}: mutant kill {x.kill.k}/{x.kill.n}; the check is weak: "
                           + "; ".join(f"{s['id']} ({s['operator']}) reward {s['reward']}" for s in x.survivors[:3]))
            if x.alternatives.n < c.min_alternatives:
                out.append(f"requirement {rid}: {x.alternatives.n} measured valid alternatives, need {c.min_alternatives}")
            elif (x.alternatives.rate or 0.0) < c.alternative_rate_min:
                out.append(f"requirement {rid}: valid alternatives pass {x.alternatives.k}/{x.alternatives.n}; rejected by "
                           + "; ".join(f"{a['id']} failing {a['failing']}" for a in x.rejected_alternatives[:3]))
        return out

    @property
    def ok(self) -> bool:
        return not self.reasons()

    def to_dict(self) -> dict:
        return {"mutant_kill": self.mutant_kill, "alternatives_pass": self.alternatives_pass, "oracle_ok": self.oracle_ok, "ok": self.ok,
                "weak_requirements": self.weak_requirements, "strict_requirements": self.strict_requirements, "notes": self.notes,
                "per_requirement": {r: {"kill": x.kill.to_dict(), "alternatives": x.alternatives.to_dict(), "survivors": x.survivors,
                                        "rejected_alternatives": x.rejected_alternatives, "equivalent": x.equivalent, "unmeasured": x.unmeasured,
                                        "off_target": x.off_target} for r, x in self.per_requirement.items()},
                "reasons": self.reasons()}


async def run_probes(probes: Sequence[Probe], grade: Callable[[Any], Awaitable[Any]], *, ledger: Any = None, oracle: Any = None,
                     requirement_ids: Sequence[str] | None = None, notes: dict[str, list[str]] | None = None,
                     config: ProbeConfig | None = None) -> ProbeReport:
    """Run every probe under the verifier ``grade(solution)``. ``oracle`` (graded first) gives the reference end state for
    the equivalence filter and must pass. ``requirement_ids`` (default: the ledger's) lists requirements that must be
    probed: one without probes is reported, never skipped."""
    cfg = config or ProbeConfig()
    ids = list(requirement_ids) if requirement_ids is not None else [r.id for r in getattr(ledger, "requirements", [])]
    for p in probes:
        if p.requirement_id not in ids:
            ids.append(p.requirement_id)
    og: Graded | None = graded_from(await call(grade, oracle)) if oracle is not None else None
    oracle_ok = None if og is None else bool(og.passed(cfg.pass_at))
    acc = {r: {"k": 0, "n": 0, "ak": 0, "an": 0, "surv": [], "rej": [], "eq": [], "un": [], "off": []} for r in ids}
    for p in probes:
        g = graded_from(await call(grade, p.solution))
        a = acc[p.requirement_id]
        if g.masked or not g.finite:
            a["un"].append(p.id)
            continue
        failed = sorted(requirements_failed(g, ledger)) if ledger is not None else []
        if p.role == "mutant":
            if g.reward <= cfg.kill_at:  # type: ignore[operator]
                a["k"] += 1
                a["n"] += 1
                if ledger is not None and g.checks and p.requirement_id not in failed:
                    a["off"].append({"id": p.id, "killed_by_requirements": failed})
            elif p.can_be_equivalent and og is not None and g.fingerprint is not None and g.fingerprint == og.fingerprint:
                a["eq"].append(p.id)
            else:
                a["n"] += 1
                a["surv"].append({"id": p.id, "operator": p.operator, "what": p.description, "reward": g.reward, "source": p.source})
        else:
            a["an"] += 1
            if g.reward >= cfg.pass_at:  # type: ignore[operator]
                a["ak"] += 1
            else:
                a["rej"].append({"id": p.id, "operator": p.operator, "reward": g.reward, "failing": g.failing(), "rejected_by_requirements": failed})
    per = {r: RequirementProbes(r, Frac(a["k"], a["n"]), Frac(a["ak"], a["an"]), a["surv"], a["rej"], a["eq"], a["un"], a["off"]) for r, a in acc.items()}
    return ProbeReport(per, oracle_ok, dict(notes or {}), cfg)


async def probe_verifier(ledger: Any, ctx: ProbeContext, grade: Callable[[Any], Awaitable[Any]], *, author: LLMProbeAuthor | None = None,
                         config: ProbeConfig | None = None, library: dict[str, Operator] | None = None) -> ProbeReport:
    """Build and run: the 05 §6.2 stage in one call."""
    probes, notes = await build_probes(ledger, ctx, author=author, library=library, config=config)
    return await run_probes(probes, grade, ledger=ledger, oracle=ctx.oracle, notes=notes, config=config)
