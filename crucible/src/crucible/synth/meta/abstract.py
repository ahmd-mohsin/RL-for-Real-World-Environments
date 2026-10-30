"""The abstractor's side of the loop (part2-paper-core §9.5): what it is shown, what it may answer, and the equal-budget
control that replaces it.

**What the abstractor sees.** Clusters, never tasks. Two kinds:
- *natural* clusters of the archive: confirmed counterexamples grouped by falsifier class, operator, defect and the
  heads of the grader's reason codes (``natural_clusters``);
- *miss* clusters of the current dev pool: planted defects that the current falsifier population did not confirm,
  grouped by defect, planting operator, what the executed witness did relative to the reference program and what the
  population said about it (``miss_clusters``; the witness is described in words: no tool, field, entity, id or value).
Each cluster has an opaque id (``n3``, ``m5``) the reply cites, so a candidate's sources are known to the loop without
the abstractor ever seeing them.

**What it answers** (``OperatorsOut``): candidate operators in the typed DSL of ``library`` (JSON, validated by
``OperatorSpec``), attack strategies in words (recorded; ATK is not run in the loop) and lemmas for the task writers.
``OPERATOR_ABSTRACTOR`` is a ``roles.Role``: it renders only ``sees`` and refuses ``never`` (anchors, the locked set,
task ids, gate code). ``PLANTER_PROPOSER`` proposes new planter parameterisations (the planter loop) and
``GUIDANCE_EDITOR`` proposes an edit to the abstractor's own guidance (the meta loop, A).

**The control** (``random_operator``): a uniformly random draw from the same DSL grammar, so an equal-budget control
round evaluates as many candidates as the loop does, without a model (E2: "operator proposals are random DSL samples").
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crucible.synth.falsify.base import DEFECTS, Counterexample
from crucible.synth.meta.library import (
    AgentCalls,
    DropCalls,
    InsertCall,
    LabelRule,
    MutateArg,
    OperatorSpec,
    SAnyWriteTool,
    SConst,
    SFieldValues,
    SObservationOf,
    SReadKey,
    SReadTool,
    SSiblingOf,
    STool,
    SWrittenKey,
    TOtherSeen,
    TScale,
    TShift,
    TSuffix,
)
from crucible.synth.meta.plant import OPS, Plant, base_planter
from crucible.synth.roles import BUILDERS, DEFECTS_DOC, Lemma, Role

# ---------------------------------------------------------------------------------------------------------- descriptions

WITNESS_WORDS: dict[str, str] = {
    "DropStep": "the reference program without one of its steps",
    "Truncate": "the reference program stopped before its end",
    "DupStep": "the reference program with one step repeated",
    "SwapEntity": "the reference program with the entity a write acts on replaced by a sibling entity of the same collection",
    "PerturbValue": "the reference program with one written value changed (another value seen for that argument, or a number moved)",
    "ExtraWrite": "the reference program plus a write on an entity it does not touch",
    "Reorder": "the reference program with two adjacent independent writes swapped",
    "WrongAnswer": "the reference program with a wrong final answer",
    "noop": "an agent that does nothing",
    "read_all": "an agent that only reads",
    "answer_only": "an agent that only reports the reference answer",
    "minus_last_write": "the reference program without its last write",
    "write_all": "an agent that calls every write tool on every key the reference reads",
    "shortcut": "an agent that calls the write tools the reference never calls, with the values the reference sets",
    "random_k": "an agent making a few random calls on entities the reference never touches",
    "planted": "one call of a tool that exists only in the planted World (a shortcut tool)",
}
KIND_WORDS: dict[str, str] = {
    "lenient": "a run that the planted task ACCEPTS and the original task rejects",
    "strict": "a valid solution (same effects as the reference) that the planted task REJECTS",
    "hidden": "the reference program needs a value that the planted instruction no longer gives",
    "leak": "the planted instruction states the decision the agent should work out",
    "unsolvable": "no run reaches a value the planted task requires",
    "nondet": "two replays of the reference program from the same start give different results",
}


def witness_words(p: Plant) -> str:
    w = p.witness
    base = KIND_WORDS.get(w.kind, w.kind)
    if w.source_op:
        return f"{base}: {WITNESS_WORDS.get(w.source_op, w.source_op)}"
    return base


class Cluster(BaseModel):
    """One cluster as the abstractor reads it, and (``members``, never shown) what it stands for."""

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: str  # natural | miss
    defect: str
    view: dict[str, Any]
    members: list[str] = Field(default_factory=list)  # plant ids (miss) or counterexample ids (natural)
    domains: list[str] = Field(default_factory=list)

    def shown(self) -> dict[str, Any]:
        return {"cluster": self.id, "kind": self.kind, "defect": f"{self.defect} {DEFECTS.get(self.defect, '')}".strip(), **self.view,
                "size": len(self.members), "domains": len(self.domains)}


def _heads(reasons: Iterable[str]) -> list[str]:
    return sorted({r.split(":", 1)[0] for r in reasons})


def natural_clusters(cxs: Sequence[Counterexample], *, limit: int = 20) -> list[Cluster]:
    by: dict[tuple[str, str, str, tuple[str, ...]], list[Counterexample]] = {}
    for c in cxs:
        by.setdefault((c.cls, c.op, c.defect or "?", tuple(_heads(c.reasons_seen))), []).append(c)
    rows = sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:limit]
    out = []
    for i, ((cls, op, d, heads), v) in enumerate(rows):
        view = {"found_by": cls, "operator": WITNESS_WORDS.get(op, op), "should": v[0].probe.expect, "reason_codes": list(heads),
                "tasks": len({c.task_id for c in v})}
        out.append(Cluster(id=f"n{i}", kind="natural", defect=d, view=view, members=[c.id for c in v], domains=sorted({c.domain for c in v})))
    return out


class PlantVerdict(BaseModel):
    plant_id: str
    hit: list[str] = Field(default_factory=list)
    suspect_only: list[str] = Field(default_factory=list)


def miss_clusters(plants: Sequence[Plant], verdicts: Mapping[str, PlantVerdict], *, limit: int = 24) -> list[Cluster]:
    """Plants no class confirmed, grouped by (defect, planting operator, witness, what moved, who only suspected it)."""
    by: dict[tuple[str, str, str, str, tuple[str, ...]], list[Plant]] = {}
    for p in plants:
        v = verdicts.get(p.id)
        if v is None or v.hit:
            continue
        k = (p.defect, f"{p.op.name}:{p.op.variant}", witness_words(p), str(p.change.get("target", "?")), tuple(v.suspect_only))
        by.setdefault(k, []).append(p)
    rows = sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:limit]
    out = []
    for i, ((d, op, wit, tgt, sus), v) in enumerate(rows):
        view = {"planted_by": op, "what_moved": tgt, "witness": wit,
                "falsifiers_said": ("only suspected by " + ", ".join(sus) + " (needs a judge, so not a confirmed counterexample)") if sus
                else "no falsifier class noticed it"}
        out.append(Cluster(id=f"m{i}", kind="miss", defect=d, view=view, members=[p.id for p in v], domains=sorted({p.domain for p in v})))
    return out


# ---------------------------------------------------------------------------------------------------------- the roles

DSL_DOC = """OPERATOR DSL (JSON). An operator turns a task's reference program (a straight-line list of tool calls) into probe
programs; each probe is executed and graded by the task's checker; the label rule says which verdict is a counterexample.
{"name": "<snake_case>", "version": 1, "why": "<the clusters it generalises, in words>", "max_probes": <=48,
 "body": <body>, "label": {"expect": "fail"|"pass", "defect": "D1".."D11", "confirm": "always"|"outside_delta"|"missing_delta"|"never"}}
<body> is one of:
 {"kind": "mutate_arg", "tool": "<glob over tool names, * = any>", "write_only": true|false, "arg": "<glob over argument names>",
  "key_only": true|false, "values": <selector>|null, "transform": <transform>|null}
     each value of the selector (or the transformed original value) replaces the argument: one probe per value
 {"kind": "insert_call", "tool": <tool selector>, "args": {"<arg>": <selector>|"borrow"}, "borrow_rest": true|false,
  "position": "start"|"end"|"after_writes"}      adds a call of each selected tool
 {"kind": "drop_calls", "tool": "<glob>", "write_only": true|false}      removes each matching call (one probe each)
 {"kind": "agent", "calls": [<insert_call body>, ...]}      a program of these calls only, from scratch
<selector>: {"sel": "written_key"} keys of entities the reference changes | {"sel": "read_key"} keys the reference reads |
 {"sel": "sibling_of", "of": <selector>, "limit": n} other keys of the same collection the reference does not change |
 {"sel": "observation_of", "tool": "<glob>", "path": "<field>"|null} values the reference observed |
 {"sel": "field_values", "field": "<field>"} start-state values of a field | {"sel": "const", "values": [...]}
<tool selector>: {"sel": "any_write_tool", "unused": bool} | {"sel": "read_tool", "unused": bool} | {"sel": "tool", "glob": "<glob>"}
<transform>: {"tf": "shift", "by": x} | {"tf": "scale", "by": x} | {"tf": "suffix", "s": "..."} | {"tf": "other_seen"} (another value seen for that argument)
A transform goes in "transform" (key "tf"), a selector in "values" (key "sel"): {"tf": "other_seen"} is never a selector.
Label rule: expect "fail" = the checker must reject the probe, so a probe it ACCEPTS is a counterexample; expect "pass" = it
must accept, so a rejection is one. confirm "always" = such a probe is a confirmed counterexample (no judge); "outside_delta" =
only if the probe changed something the reference did not; "missing_delta" = only if it lacks part of the reference's
changes; "never" = it only becomes a suspect for a judge. A probe with exactly the reference's effects must pass whatever the rule.
An operator is accepted only if it reproduces the clusters it cites, finds planted defects in OTHER domains, raises NO alarm
on correct tasks, and stays cheap. Use only generic names: no tool, field, id or value of any particular application, unless
the glob is generic ("*", "*.get_*")."""


class OperatorsOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    operators: list[dict[str, Any]] = Field(default_factory=list)
    cites: list[list[str]] = Field(default_factory=list)  # per operator: the cluster ids it generalises
    strategies: list[str] = Field(default_factory=list)
    lemmas: list[Lemma] = Field(default_factory=list)
    notes: str = ""


OPERATOR_ABSTRACTOR = Role(
    name="abstractor", stage="abstract", output=OperatorsOut, temperature=0.4, max_tokens=6144,
    sees=("clusters", "misses", "defects", "dsl", "guidance", "library", "count"),
    never=("anchors", "locked_set", "task_ids", "gate_code", "tasks"),
    differs_from=BUILDERS, independent=True,
    system=("You improve an automatic checker of RL tasks. Falsifier operators generate probe runs that expose defects in a task's "
            "verifier. You are shown clusters of defects the current falsifiers confirmed (natural) and clusters of planted defects "
            "they MISSED (misses), described in general terms. Write new operators in the DSL that would confirm the missed clusters "
            "without raising alarms on correct tasks. Prefer operators that would work in any application. " + DEFECTS_DOC + "\n"
            'Reply {"operators": [<operator JSON>, ...], "cites": [["m0", ...] for each operator], "strategies": ["<attack strategy '
            'in words>", ...], "lemmas": [{"applies_to": "proposer|contract_writer|repairer|oracle_programmer", "rule": "...", '
            '"evidence": ["<cluster id>"]}], "notes": "..."} with at most the requested count of operators.'))


class PlanterOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    planters: list[dict[str, Any]] = Field(default_factory=list)  # each {"name", "variant", "params"}
    notes: str = ""


def dev_variants() -> dict[str, list[str]]:
    """The operator variants a dev planter may use: the base planter's (the locked set's held-out variants are never
    shown to, or accepted from, the planter loop)."""
    out: dict[str, list[str]] = {}
    for o in base_planter().ops:
        if o.variant not in out.setdefault(o.name, []):
            out[o.name].append(o.variant)
    return out


PLANTER_DOC = ("PLANT OPERATORS (each plants one defect into a correct task by editing its checker, instruction or World; "
               "every plant must come with an executed witness): " + "; ".join(
                   f"{k} ({OPS[k].defect}; variants {', '.join(v)})" for k, v in dev_variants().items()) + ". Parameters you may set: "
               "drop_required/widen_frame: why (the reason text the edited checker carries); relax_value: factor (tolerance multiplier); "
               "hide_info: phrase (the vague words that replace a literal, variant redact); leak_decision: template (the sentence "
               "appended to the instruction, with {v} for the decision value or {h} for it in words); break_solvability: suffix; "
               "inject_nondet: field (the stamped field name).")

PLANTER_PROPOSER = Role(
    name="planter_proposer", stage="plant", output=PlanterOut, temperature=0.7, max_tokens=3072,
    sees=("operators", "miss_rates", "natural_mix", "count", "history"),
    never=("anchors", "locked_set", "task_ids", "gate_code", "tasks"),
    differs_from=BUILDERS, independent=True,
    system=("You design planted defects that test an automatic checker of RL tasks: realistic defects of the kinds real task "
            "writers make, which the current falsifiers miss. You see the plant operators, how often the falsifiers miss each one, and "
            "the mix of defect classes found in real tasks, and (history) how your earlier proposals fared: a planter is kept only "
            "when it plants NEW defects (a parameter that does not change the planted checker or instruction plants nothing new) "
            "and the falsifiers miss some of them. Propose new parameterisations of existing operators (new phrasings, "
            "tolerances, reasons). Never name a particular application, entity, id or value. " + PLANTER_DOC + "\n"
            'Reply {"planters": [{"name": "<operator>", "variant": "<variant>", "params": {...}}, ...], "notes"}.'))


class GuidanceOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    add: list[str] = Field(default_factory=list)
    remove: list[int] = Field(default_factory=list)
    notes: str = ""


GUIDANCE_EDITOR = Role(
    name="guidance_editor", stage="abstract", output=GuidanceOut, temperature=0.3, max_tokens=2048,
    sees=("guidance", "gate_outcomes", "dsl"),
    never=("anchors", "locked_set", "task_ids", "gate_code", "tasks"),
    differs_from=BUILDERS, independent=True,
    system=("You edit the guidance given to an author of checker operators. You see the current guidance (numbered) and why its "
            "last operators were refused by the acceptance gate (counts per failed check). Propose at most two short general rules to "
            "add and the numbers of rules to remove. Never mention a particular application, entity or value. "
            'Reply {"add": ["..."], "remove": [<n>], "notes"}.'))


# ---------------------------------------------------------------------------------------------------------- parsing

def parse_operators(out: OperatorsOut, *, prefix: str, limit: int) -> tuple[list[tuple[OperatorSpec, list[str]]], list[str]]:
    """The valid operators of a reply (renamed ``<prefix>_<i>_<name>`` so names never collide), with their citations,
    and the errors of the invalid ones."""
    ops, errs = [], []
    for i, doc in enumerate(out.operators[:limit]):
        try:
            d = dict(doc)
            d["name"] = f"{prefix}_{i}_{d.get('name') or 'op'!s}"[:60].replace(" ", "_")
            d["max_probes"] = min(int(d.get("max_probes") or 48), 48)
            spec = OperatorSpec.model_validate(d)
        except (ValueError, TypeError) as e:
            errs.append(f"operator {i}: {str(e)[:300]}")
            continue
        cites = [c for c in (out.cites[i] if i < len(out.cites) else []) if isinstance(c, str)]
        ops.append((spec, cites))
    return ops, errs


# ---------------------------------------------------------------------------------------------------------- the control

_GLOBS = ("*", "*.get_*", "*.list_*", "*.set_*", "*.update_*")
_ARGS = ("*", "*id", "*_id", "status", "reason", "*amount*", "priority")


def _selector(rng: random.Random, depth: int = 0) -> Any:
    k = rng.choice(["written_key", "read_key", "sibling_of", "observation_of", "field_values", "const"] if depth == 0 else ["written_key", "read_key"])
    if k == "written_key":
        return SWrittenKey()
    if k == "read_key":
        return SReadKey()
    if k == "sibling_of":
        return SSiblingOf(of=_selector(rng, 1), limit=rng.choice([4, 8, 16]))
    if k == "observation_of":
        return SObservationOf(tool=rng.choice(_GLOBS), path=rng.choice([None, "id", "status"]))
    if k == "field_values":
        return SFieldValues(field=rng.choice(["status", "id", "priority", "reason"]))
    return SConst(values=rng.choice([[0], [1], ["x"], [""], [None]]))


def _transform(rng: random.Random) -> Any:
    k = rng.choice(["shift", "scale", "suffix", "other_seen"])
    if k == "shift":
        return TShift(by=rng.choice([-1, 1, 10, 0.01]))
    if k == "scale":
        return TScale(by=rng.choice([0, 2, 10, 0.5]))
    if k == "suffix":
        return TSuffix(s=rng.choice(["-x", " ", "_1"]))
    return TOtherSeen()


def _tool_sel(rng: random.Random) -> Any:
    k = rng.choice(["any_write_tool", "read_tool", "tool"])
    if k == "any_write_tool":
        return SAnyWriteTool(unused=rng.random() < 0.5)
    if k == "read_tool":
        return SReadTool(unused=rng.random() < 0.5)
    return STool(glob=rng.choice(_GLOBS))


def _insert(rng: random.Random) -> InsertCall:
    return InsertCall(tool=_tool_sel(rng), args={}, borrow_rest=True, position=rng.choice(["start", "end", "after_writes"]))


def random_operator(rng: random.Random, name: str) -> OperatorSpec:
    """One uniformly random operator of the DSL grammar (the equal-budget control's proposal)."""
    kind = rng.choice(["mutate_arg", "insert_call", "drop_calls", "agent"])
    if kind == "mutate_arg":
        use_sel = rng.random() < 0.5
        body: Any = MutateArg(tool=rng.choice(_GLOBS), write_only=rng.random() < 0.7, arg=rng.choice(_ARGS), key_only=rng.random() < 0.3,
                              values=_selector(rng) if use_sel else None, transform=None if use_sel else _transform(rng))
    elif kind == "insert_call":
        body = _insert(rng)
    elif kind == "drop_calls":
        body = DropCalls(tool=rng.choice(_GLOBS), write_only=rng.random() < 0.5)
    else:
        body = AgentCalls(calls=[_insert(rng) for _ in range(rng.choice([1, 2]))])
    label = LabelRule(expect=rng.choice(["fail", "fail", "fail", "pass"]), defect=rng.choice(list(DEFECTS)),
                      confirm=rng.choice(["always", "outside_delta", "missing_delta", "never"]))
    return OperatorSpec(name=name, body=body, label=label, max_probes=48, why="random DSL sample (equal-budget control)")


def defect_mix(cxs: Sequence[Counterexample]) -> dict[str, int]:
    return dict(Counter(c.defect or "?" for c in cxs).most_common())
