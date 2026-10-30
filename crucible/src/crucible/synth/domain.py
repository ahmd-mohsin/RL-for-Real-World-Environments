"""``DomainSpec`` (part2-paper-core §12): what a domain is, as data the proposer samples from.

A domain spec holds the Apps (collections and tools), the **skills** (the kinds of work the domain teaches), the
**knobs** (the dimensions a task varies along: evidence, distractors, authority, scope, plus the domain's own), the
**intent-card schema** (what a proposal must say about given and derived values), the policy parameters and business
rules, the permission model, and the colleagues. The **cell grid** is skill × knob settings; ``sample_cells`` draws a
balanced, seeded sample of it and ``cell_text`` is what the proposer is shown for one cell.

Sources:
- ``finance_ap_spec()``: the existing pack, written by hand (tools bound from the World with ``bind_tools``);
- ``spec_from_world(dir)``: a ``worldgen`` World (its ``domain.json`` if a family build wrote one, else derived from
  ``design.json``: one skill per write tool, the generic knobs);
- ``generate_spec(keyword, client)``: the swarm writes it (the ``domain_spec_writer`` role, a non-Qwen builder family
  bound in ``factory/synth_wire.py``), checked here, with the problems fed back.

The generic knobs (``GENERIC_KNOBS``) are the framework's: every family supports them because the kit and the instance
generator plant the evidence (documents, colleagues), the distractors (near-duplicates, stale records, conflicting
sources) and the acting user's authority.
"""

from __future__ import annotations

import itertools
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from crucible.synth.iface import ToolInfo
from crucible.synth.roles import Role, RoleClient, ask


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _ident(v: str) -> str:
    v = re.sub(r"\W+", "_", str(v).strip().lower()).strip("_")
    if not v or not v.isidentifier():
        raise ValueError(f"{v!r} is not an identifier")
    return v


class SkillSpec(_M):
    id: str
    description: str
    tools: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _ident(v)


class KnobSpec(_M):
    name: str
    description: str = ""
    values: list[str]

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _ident(v)


class PolicyParam(_M):
    name: str
    type: Literal["number", "integer", "string", "boolean", "ref"] = "number"
    default: Any = None
    choices: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    unit: str | None = None
    meaning: str = ""
    table: str | None = None  # type ref: a key of this table (e.g. the acting user)
    where: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _ident(v)


class EntitySpec(_M):
    name: str
    description: str = ""
    target: bool = True  # tasks act on it


class RoleLimit(_M):
    role: str
    may: list[str] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)


class PermissionSpec(_M):
    user_table: str | None = None  # the table of people who use the App (the acting user is one of them)
    roles: list[RoleLimit] = Field(default_factory=list)
    acting_param: str = "acting_user_id"


class ColleagueSpec(_M):
    title: str
    knows: list[str] = Field(default_factory=list)


class AppSpec(_M):
    name: str
    collections: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    read_only: list[str] = Field(default_factory=list)
    writes: dict[str, list[str]] = Field(default_factory=dict)  # tool -> collections


GENERIC_KNOBS: tuple[KnobSpec, ...] = (
    KnobSpec(name="evidence", description="where the facts the decision needs live",
             values=["the target record alone", "two or more linked records", "a policy document and the records", "a colleague and the records"]),
    KnobSpec(name="distractor", description="what makes the wrong entity look right",
             values=["none", "a near-duplicate entity", "a stale or superseded record", "two sources that disagree"]),
    KnobSpec(name="authority", description="whether the acting user may do it",
             values=["within the acting user's authority", "beyond it: the agent must escalate or route it, not act"]),
    KnobSpec(name="scope", description="how many entities the task touches", values=["one entity", "three to six entities chosen by a stated rule"]),
)

INTENT_SCHEMA: dict[str, str] = {
    "skill": "the id of one of the domain's skills",
    "given": "values handed to the agent outside the instruction (usually [])",
    "derive": "the values the agent must work out itself, bare, as the system stores them",
    "knobs": "the cell's knob settings the task realises",
}


class Cell(_M):
    skill: str
    knobs: dict[str, str] = Field(default_factory=dict)


class DomainSpec(_M):
    name: str
    keyword: str = ""
    function: str = ""
    description: str
    users: list[str] = Field(default_factory=list)
    entities: list[EntitySpec] = Field(default_factory=list)
    skills: list[SkillSpec]
    knobs: list[KnobSpec] = Field(default_factory=list)
    policy_params: list[PolicyParam] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    param_constraints: list[str] = Field(default_factory=list)  # expressions over policy parameter names every variant satisfies
    permissions: PermissionSpec = Field(default_factory=PermissionSpec)
    colleagues: list[ColleagueSpec] = Field(default_factory=list)
    apps: list[AppSpec] = Field(default_factory=list)
    intent_schema: dict[str, str] = Field(default_factory=lambda: dict(INTENT_SCHEMA))
    common_practice: bool = True
    rationale: str = ""
    generic_knobs: bool = True

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _ident(v)

    # -- the cell grid ----------------------------------------------------------------------------------------------
    def all_knobs(self) -> list[KnobSpec]:
        own = {k.name for k in self.knobs}
        return [*self.knobs, *(k for k in GENERIC_KNOBS if self.generic_knobs and k.name not in own)]

    def grid_size(self) -> int:
        n = len(self.skills)
        for k in self.all_knobs():
            n *= max(1, len(k.values))
        return n

    def cells(self, limit: int = 100_000) -> list[Cell]:
        """Every skill × knob-setting cell (the full grid, up to ``limit``)."""
        ks = self.all_knobs()
        out: list[Cell] = []
        for s in self.skills:
            for combo in itertools.product(*[k.values for k in ks]):
                out.append(Cell(skill=s.id, knobs={k.name: v for k, v in zip(ks, combo, strict=True)}))
                if len(out) >= limit:
                    return out
        return out

    def sample_cells(self, n: int, seed: int = 0) -> list[Cell]:
        """A balanced, seeded sample: skills round-robin, and each knob's values cycled from a shuffled order, so every
        value of every knob appears about equally often (not the full product, which grows multiplicatively)."""
        rng = random.Random(f"cells/{self.name}/{seed}")
        ks = self.all_knobs()
        skills = [s.id for s in self.skills]
        rng.shuffle(skills)
        cycles = {}
        for k in ks:
            vals = list(k.values)
            rng.shuffle(vals)
            cycles[k.name] = vals
        out = []
        for i in range(n):
            knobs = {k.name: cycles[k.name][(i + j * 7 + rng.randrange(len(cycles[k.name]))) % len(cycles[k.name])] for j, k in enumerate(ks)}
            out.append(Cell(skill=skills[i % len(skills)], knobs=knobs))
        return out

    def cell_text(self, c: Cell) -> str:
        sk = next((s for s in self.skills if s.id == c.skill), None)
        knobs = "; ".join(f"{k}: {v}" for k, v in c.knobs.items())
        return f"skill {c.skill}: {sk.description if sk else ''} | {knobs}"

    def cell_texts(self, n: int = 12, seed: int = 0) -> list[str]:
        return [self.cell_text(c) for c in self.sample_cells(n, seed)]

    def environment_text(self) -> str:
        """The description a proposer reads (who uses the system, what it holds, its rules)."""
        rules = " ".join(self.rules[:8])
        return (f"{self.description} Users: {', '.join(self.users[:6]) or 'staff'}. Business rules: {rules} Policy and procedure documents are in "
                "the document library; colleagues can be asked for facts not in the records.").strip()

    def targetable(self) -> list[str]:
        """Collections tasks may act on: the collections write tools write, plus entities marked as targets."""
        out = {c for a in self.apps for cs in a.writes.values() for c in cs}
        return sorted(out)


# ---------------------------------------------------------------------------------------------------------- checks

def check_spec(s: DomainSpec) -> None:
    """The generator's checks (each problem goes back to the swarm)."""
    if len(s.skills) < 5:
        raise ValueError(f"list at least 5 skills, not {len(s.skills)}")
    if len({x.id for x in s.skills}) != len(s.skills):
        raise ValueError("skill ids repeat")
    if len(s.knobs) < 2:
        raise ValueError("list at least 2 domain-specific knobs (task dimensions besides evidence, distractor, authority and scope)")
    for k in s.knobs:
        if len(k.values) < 2:
            raise ValueError(f"knob {k.name} needs at least 2 values")
    if len([e for e in s.entities if e.target]) < 2 or len(s.entities) < 4:
        raise ValueError("list at least 4 entities, at least 2 of them task targets")
    if not 3 <= len(s.policy_params) <= 12:
        raise ValueError(f"list 3 to 12 policy parameters, not {len(s.policy_params)}")
    if len({p.name for p in s.policy_params}) != len(s.policy_params):
        raise ValueError("policy parameter names repeat")
    for p in s.policy_params:
        if p.type in ("number", "integer"):
            if not isinstance(p.default, int | float) or isinstance(p.default, bool):
                raise ValueError(f"policy parameter {p.name}: a numeric default is needed")
            if p.min is not None and p.max is not None and not p.min <= p.default <= p.max:
                raise ValueError(f"policy parameter {p.name}: the default {p.default} is outside [{p.min}, {p.max}]")
        if p.type in ("string", "boolean") and p.choices and p.default not in p.choices:
            raise ValueError(f"policy parameter {p.name}: the default {p.default!r} is not one of its choices")
        if p.type == "ref" and not p.table:
            raise ValueError(f"policy parameter {p.name}: a ref parameter names its table")
    if not s.permissions.user_table or len(s.permissions.roles) < 2:
        raise ValueError("permissions need the user table and at least 2 roles with what each may do and its limits")
    if not any(p.name == s.permissions.acting_param for p in s.policy_params):
        raise ValueError(f"add the policy parameter {s.permissions.acting_param} (type ref, table {s.permissions.user_table}): the acting user")
    from crucible.synth.datagen import DataSpecError, eval_expr, expr_names

    defaults = {p.name: p.default for p in s.policy_params}
    for c in s.param_constraints:
        try:
            bad = expr_names(c) - set(defaults)
            if bad:
                raise ValueError(f"param constraint {c!r} names unknown parameters {sorted(bad)}")
            if not eval_expr(c, defaults):
                raise ValueError(f"param constraint {c!r} does not hold for the defaults")
        except DataSpecError as e:
            raise ValueError(f"param constraint {c!r}: {e}") from None
    if len(s.rules) < 3:
        raise ValueError("state at least 3 business rules")
    if not s.common_practice:
        raise ValueError("the domain must be common enterprise practice")


# ---------------------------------------------------------------------------------------------------------- the generator role

DOMAIN_SPEC_WRITER = Role(
    name="domain_spec_writer", stage="W0", output=DomainSpec, temperature=0.3, max_tokens=8192,
    sees=("keyword", "function", "guidance"),
    never=("tests", "tasks", "contract"),
    system=("You specify an enterprise software domain so that a realistic application and many training tasks can be built for it. "
            "Describe it as it works in a real mid-sized company: who uses the system, the records it holds, the written business "
            "rules and policy limits, who may do what, and the colleagues people ask. Prefer common, well-known practice over exotic "
            "variants: a new hire in this function would recognise everything.\n"
            "Reply with one JSON object: {\"name\": \"<lowercase identifier>\", \"keyword\", \"function\", \"description\" (2 sentences), "
            "\"users\": [<job titles>], \"entities\": [{\"name\", \"description\", \"target\": true|false}] (4 to 8), "
            "\"skills\": [{\"id\", \"description\"}] (6 to 10: the kinds of work, e.g. a decision under policy, a reconciliation, a "
            "routing or escalation, a correction of a record, a figure for a manager), "
            "\"knobs\": [{\"name\", \"description\", \"values\": [...]}] (2 to 4 domain dimensions a task varies along), "
            "\"policy_params\": [{\"name\", \"type\": \"number|integer|string|boolean|ref\", \"default\", \"choices\", \"min\", \"max\", "
            "\"unit\", \"meaning\", \"table\"}] (4 to 8: thresholds, limits, windows, required approvals; include acting_user_id of "
            "type ref whose table is the user table), "
            "\"rules\": [<business rules in words, naming the parameters>], \"param_constraints\": [<expressions over parameter names that "
            "any policy variant must keep, e.g. \"vip_window_days >= standard_window_days\">], "
            "\"permissions\": {\"user_table\", \"roles\": [{\"role\", \"may\": [...], \"limits\": {...}}], \"acting_param\": \"acting_user_id\"}, "
            "\"colleagues\": [{\"title\", \"knows\": [<kinds of facts not in the records>]}] (1 to 3), "
            "\"common_practice\": true, \"rationale\"}."))


async def generate_spec(keyword: str, client: RoleClient, *, function: str = "", guidance: str = "", tag: str = "family:spec") -> DomainSpec:
    """The swarm writes the spec for ``keyword`` (checked by ``check_spec``; problems go back as feedback)."""
    spec: DomainSpec = await ask(client, DOMAIN_SPEC_WRITER, tag=tag, check=check_spec, retries=2, keyword=keyword, function=function or None,
                                 guidance=guidance or None)
    return spec.model_copy(update={"keyword": spec.keyword or keyword, "function": spec.function or function})


# ---------------------------------------------------------------------------------------------------------- binding and loaders

def bind_design(spec: DomainSpec, design: Mapping[str, Any], schema: Mapping[str, Any] | None = None) -> DomainSpec:
    """Fill ``apps`` from a World design (tool names, read-only flags, writes) and tie each skill to the write tools
    whose name or description shares its words (best effort: a skill's tools are hints for the proposer)."""
    tools = list(design.get("tools") or [])
    colls = sorted({t["name"] for t in (schema or {}).get("tables", [])}) if schema else sorted({c for t in tools for c in t.get("writes") or []})
    app = AppSpec(name=str(design.get("app", spec.name)), collections=colls, tools=[t["name"] for t in tools],
                  read_only=[t["name"] for t in tools if t.get("read_only")], writes={t["name"]: list(t.get("writes") or []) for t in tools if not t.get("read_only")})
    skills = []
    for s in spec.skills:
        words = set(re.findall(r"[a-z]{4,}", (s.id + " " + s.description).lower()))
        hits = [t["name"] for t in tools if not t.get("read_only") and words & set(re.findall(r"[a-z]{4,}", (t["name"] + " " + t.get("description", "")).lower()))]
        skills.append(s.model_copy(update={"tools": s.tools or hits[:4], "collections": s.collections or sorted({c for h in hits for c in app.writes.get(h, [])})}))
    return spec.model_copy(update={"apps": [app], "skills": skills})


def bind_tools(spec: DomainSpec, tools: Iterable[ToolInfo]) -> DomainSpec:
    """Fill ``apps`` from a live World's tools (one App per qualified-name prefix)."""
    by: dict[str, AppSpec] = {}
    for t in tools:
        app, _, name = t.name.partition(".")
        a = by.setdefault(app, AppSpec(name=app))
        a.tools.append(name)
        if t.read_only:
            a.read_only.append(name)
        else:
            a.writes[name] = [w.split(".", 1)[1] for w in (t.writes or []) if "." in w]
            a.collections = sorted({*a.collections, *a.writes[name]})
    return spec.model_copy(update={"apps": list(by.values())})


def finance_ap_spec() -> DomainSpec:
    """The finance_ap pack as a DomainSpec (hand-written from the pack's policy; its tools are bound from the World)."""
    return DomainSpec(
        name="finance_ap", keyword="accounts payable invoice processing", function="finance",
        description=("Accounts payable in an ERP: pending supplier invoices are approved, put on hold or rejected under the AP policy after "
                     "checking purchase orders, goods receipts, vendors and earlier invoices."),
        users=["AP clerk", "AP manager", "controller"],
        entities=[EntitySpec(name="invoice"), EntitySpec(name="purchase order"), EntitySpec(name="goods receipt"), EntitySpec(name="vendor"),
                  EntitySpec(name="payment", target=False)],
        skills=[SkillSpec(id="three_way_match", description="approve or hold an invoice by matching it to its purchase order and goods receipt"),
                SkillSpec(id="duplicate_check", description="reject an invoice that duplicates an earlier one"),
                SkillSpec(id="vendor_risk", description="hold invoices of blocked vendors or vendors with changed bank details"),
                SkillSpec(id="policy_threshold", description="route invoices above the approval threshold"),
                SkillSpec(id="ap_figure", description="report a figure from the records (totals, counts, ageing)"),
                SkillSpec(id="batch_review", description="decide several pending invoices by a stated rule")],
        knobs=[KnobSpec(name="mismatch", values=["none", "price", "quantity", "missing receipt"]),
               KnobSpec(name="vendor_state", values=["active", "blocked", "bank details changed"])],
        rules=["Invoices are approved only when they match their purchase order and goods receipt within tolerance.",
               "Duplicate invoices are rejected.", "Invoices from blocked vendors are held."],
        permissions=PermissionSpec(user_table=None, roles=[]), generic_knobs=True,
        rationale="the existing hand-built pack (crucible_pack_finance_ap)")


def spec_from_world(world_dir: str | Path) -> DomainSpec:
    """The spec of a generated World: its ``domain.json`` when a family build wrote one, else derived from the design."""
    d = Path(world_dir)
    if (d / "domain.json").exists():
        return DomainSpec.model_validate_json((d / "domain.json").read_text())
    design = json.loads((d / "design.json").read_text())
    schema = json.loads((d / "schema.json").read_text()) if (d / "schema.json").exists() else None
    skills = [SkillSpec(id=t["name"], description=t.get("description", ""), tools=[t["name"]], collections=list(t.get("writes") or []))
              for t in design.get("tools", []) if not t.get("read_only")]
    spec = DomainSpec(name=str(design.get("app", d.name)), description=str(design.get("description", "")), skills=skills, generic_knobs=True)
    return bind_design(spec, design, schema)


def load_spec(name: str, world_dir: str | Path | None = None, tools: Sequence[ToolInfo] | None = None) -> DomainSpec:
    """The loader for existing domains: ``finance_ap`` (the pack) or a generated World directory."""
    if name == "finance_ap":
        s = finance_ap_spec()
        return bind_tools(s, tools) if tools else s
    if world_dir is None:
        raise ValueError(f"domain {name!r} needs its World directory")
    return spec_from_world(world_dir)


__all__ = [
    "DOMAIN_SPEC_WRITER",
    "GENERIC_KNOBS",
    "INTENT_SCHEMA",
    "AppSpec",
    "Cell",
    "ColleagueSpec",
    "DomainSpec",
    "EntitySpec",
    "KnobSpec",
    "PermissionSpec",
    "PolicyParam",
    "SkillSpec",
    "bind_design",
    "bind_tools",
    "check_spec",
    "finance_ap_spec",
    "generate_spec",
    "load_spec",
    "spec_from_world",
]
