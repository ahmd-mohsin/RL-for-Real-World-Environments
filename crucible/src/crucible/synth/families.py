"""World families and World instances (part2-paper-core §6 density, §11 synthesized Apps, §12 ``DomainSpec``).

An *environment* is a **World instance** = (World family × instance config).

- A **family** is generated once per domain keyword by the swarm, reviewed by the gates, and frozen by hash:
  the ``DomainSpec`` (skills, knobs, policy parameters, permissions, colleagues), the tool design, the schema, the App
  code, the blind tests (frozen at write time, never edited), the data generator (``datagen.DataSpec``), the policy and
  SOP document templates, and the colleague fact templates. The framework kit (``worldkit``: documents, policy
  parameters, colleagues) is added to every family as trusted code.
- An **instance** is the family plus a seed, a data profile (size and noise rates), a policy variant (the policy
  parameter values, which the documents are rendered with) and an actor config (the acting user, the colleagues present
  and which records their facts are about). ``make_instance`` builds one with **no model call** and no code change, so a
  family yields as many distinct, dense environments as wanted.

    build_family(keyword):              resumable: every stage is recorded in state.json and skipped when done
      spec     ← domain_spec_writer       DomainSpec, checked (domain.check_spec)
      design   ← family_designer          tool API with the realism surface (pagination, filters, error codes, permissions)
      schema   ← schema_deriver           + the kit tables
      code     ← app_coder                reads policy_params, never hard-codes a policy value
      tests    ← blind_tester             a different family, never sees the code; frozen
      debug    ← debugger / test_approver worldgen.test_and_debug: code-only fixes, disputed tests dropped on approval
      datagen  ← data_spec_writer         sampled, integrity-checked, loaded and smoke-tested; problems fed back
      docs     ← doc_writer               templates with {{param}} placeholders: every parameter stated, a data-standards doc
      actors   ← actor_writer             colleague fact templates over records (optional)
      base     instance 0 generated, loaded and smoke-tested
      freeze   manifest of digests (world.json); the family is now read-only

Layout under ``<worlds>/<family>/``: ``state.json``, ``app/`` (the frozen family files), ``instances/<id>/`` (each a full
App directory ``SandboxWorld`` can serve: the family's files copied and re-hashed, plus ``seed.json``, ``noise.json``,
``instance.json``), and the wire's ``spend.jsonl``/``usage.jsonl``. Nothing under it is committed (data).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crucible.synth.datagen import DATAGEN_FORMAT, PROFILES, TODAY, DataSpec, DataSpecError, Generated, generate, integrity
from crucible.synth.domain import DomainSpec, PolicyParam, bind_design, generate_spec
from crucible.synth.roles import (
    APP_API_DOC,
    APP_CODER,
    BLIND_TESTER,
    DEBUGGER,
    SCHEMA_DERIVER,
    TEST_FORMAT_DOC,
    AppCodeOut,
    Role,
    RoleClient,
    RoleFailed,
    SchemaOut,
    TableD,
    ToolDesignOut,
    ToolTestsOut,
    ask,
)
from crucible.synth.sandbox import SANDBOX_KIND, SandboxError, check_code
from crucible.synth.worldgen import WorldgenReport, check_design, check_schema, check_tests, smoke_world, test_and_debug
from crucible.synth.worldkit import KIT_API_DOC, KIT_NAMES, kit_tables, placeholders, render

FAMILY_ROLES = ("domain_spec_writer", "family_designer", "schema_deriver", "app_coder", "blind_tester", "debugger", "test_approver",
                "data_spec_writer", "doc_writer", "actor_writer")
STAGES = ("spec", "design", "schema", "code", "tests", "debug", "datagen", "docs", "actors", "base", "freeze")
FROZEN_FILES = ("design.json", "schema.json", "app.py", "tests.json", "test_report.json", "kit.json", "datagen.json", "docs.json", "actors.json",
                "domain.json")
PAGE_PARAMS = ({"page", "page_size"}, {"limit", "offset"}, {"cursor", "limit"}, {"page", "per_page"})
PROFILE_CYCLE = ("standard", "dense", "noisy", "clean")


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jwrite(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def _jread(path: Path) -> Any:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------------------------------------- the realism surface

REALISM_SURFACE = """REALISM REQUIREMENTS (the API must feel like a real internal enterprise system)
- Read tools: at least one list/search tool pages (parameters page, starting at 1, and page_size, at most 50; the result has
  items, page, total and next_page) and filters (two or more optional filter parameters such as status, owner, a date range
  or a text query). Get tools return one record by its id with its linked records summarised.
- Write tools carry the domain's real decisions and corrections (at least three). Each states its preconditions and refuses
  with an error whose message starts with a code: NOT_FOUND (an unknown id), INVALID_ARGUMENT (a bad or unknown enum value, a
  missing reason), INVALID_STATE (the record is in the wrong status), PERMISSION_DENIED (the acting user lacks the role or is
  over a limit), CONFLICT (a duplicate). List the codes a tool raises in its errors.
- The acting user is the policy parameter acting_user_id (a row of the user table): the logged-in session, read at run time.
  NO tool takes the acting user (or any "performed by" user) as an argument: a caller could then act as someone else. Write
  tools check the acting user's role and limits as the permissions say, and refuse with PERMISSION_DENIED otherwise.
- Policy thresholds, windows and limits are the domain's policy parameters (read at run time, never hard-coded).
- Enumerated values (statuses, reason codes, categories) are listed in the parameter's enum and in the description.
- Writes record who and when where a real system would (the acting user, the time).
- Do NOT design document or colleague tools: the platform provides list_documents, read_document, search_documents,
  list_colleagues and ask_colleague."""


IMPERSONATION = ("acting_user", "acting_user_id", "actor_id", "performed_by", "requested_by_user", "current_user", "user_id_acting", "approver_id",
                 "agent_id_acting", "operator_id", "session_user")


def realism_design_problems(d: ToolDesignOut, acting_param: str = "acting_user_id") -> list[str]:
    """What the design misses of ``REALISM_SURFACE`` (each problem goes back to the designer). A tool that takes the acting
    user as an argument is refused outright: it is an impersonation shortcut around every permission check (pilot 1:
    Kimi's design for returns and refunds passed ``acting_user_id`` to all nine write tools)."""
    out = []
    clash = sorted({t.name for t in d.tools} & KIT_NAMES)
    if clash:
        out.append(f"the platform provides {clash}: remove them")
    imp = sorted({f"{t.name}.{p.name}" for t in d.tools for p in t.params if p.name == acting_param or p.name in IMPERSONATION})
    if imp:
        out.append(f"tools take the acting user as an argument {imp[:6]}: remove it (the acting user is the session, the policy parameter "
                   f"{acting_param}; a caller must not choose who it acts as)")
    ro = [t for t in d.tools if t.read_only]
    rw = [t for t in d.tools if not t.read_only]
    if len(rw) < 3:
        out.append(f"design at least three write tools for the domain's decisions, not {len(rw)}")
    if not any(any(s <= {p.name for p in t.params} for s in PAGE_PARAMS) for t in ro):
        out.append("no read tool pages: give a list/search tool page and page_size parameters (result: items, page, total, next_page)")
    if not any(sum(1 for p in t.params if not p.required) >= 2 for t in ro):
        out.append("no read tool filters: give a list/search tool two or more optional filter parameters")
    errs = " ".join(" ".join(t.errors) for t in rw).upper()
    if "PERMISSION_DENIED" not in errs:
        out.append("no write tool refuses with PERMISSION_DENIED (the acting user's role or limit)")
    silent = [t.name for t in rw if not any(c in " ".join(t.errors).upper() for c in ("NOT_FOUND", "INVALID_ARGUMENT", "INVALID_STATE"))]
    if silent:
        out.append(f"write tools without NOT_FOUND / INVALID_ARGUMENT / INVALID_STATE errors: {silent}")
    return out


# Pilot 1: a family App (9 to 14 tools with permission checks and policy parameters) does not fit F1's 20k-token coder
# budget once Kimi's reasoning is counted (every first attempt ended at the limit, $0.13 each). Same roles, same
# family rules, larger budgets.
FAMILY_APP_CODER = APP_CODER.model_copy(update={"max_tokens": 64000})
FAMILY_DEBUGGER = DEBUGGER.model_copy(update={"max_tokens": 64000})

FAMILY_DESIGNER = Role(
    name="family_designer", stage="W1", output=ToolDesignOut, temperature=0.3, max_tokens=32768,
    sees=("domain_spec", "realism", "guidance"),
    never=("tests", "tasks"),
    system=("You design the tool API of a self-contained internal enterprise application from its domain specification, tool first: "
            "each tool has a name, a description an agent can act on (state every enum value and reason code it accepts), typed "
            "parameters, whether it is read-only, the collections (tables) it writes, its preconditions, its postconditions (the exact "
            "state change), the errors it raises, and the shape of what it returns (an example object). 9 to 14 tools covering the "
            "domain's skills. Be compact: a description of at most two sentences, at most three short pre and post items, a small "
            "returns example.\n"
            'Reply {"app": "<lowercase identifier>", "description": "...", "tools": [{"name", "description", "params": [{"name", "type", '
            '"required", "description", "enum", "default"}], "read_only", "writes", "pre", "post", "errors", "returns"}]}.'))


class DocTemplate(_M):
    doc_id: str
    title: str
    kind: str = "policy"
    tags: list[str] = Field(default_factory=list)
    body: str


class DocsOut(_M):
    documents: list[DocTemplate]


class FactTemplate(_M):
    topic: str
    keywords: list[str]
    table: str
    text: str


class ColleagueTemplate(_M):
    colleague_id: str
    name: str
    title: str
    topics: list[str] = Field(default_factory=list)
    facts: list[FactTemplate] = Field(default_factory=list)


class ActorsOut(_M):
    colleagues: list[ColleagueTemplate]


DATA_SPEC_WRITER = Role(
    name="data_spec_writer", stage="W2", output=DataSpec, temperature=0.4, max_tokens=48000,
    sees=("domain_spec", "design", "schema", "datagen_format", "feedback"),
    never=("tests", "tasks", "contract"),
    system=("You write the data generator of an enterprise application: for every table of the schema (except the platform tables "
            "documents, policy_params, colleagues), a generator per field, in the format given. The data must look like a real "
            "mid-sized company's: realistic names, codes, amounts with realistic distributions, statuses with realistic weights, dates "
            "in order (history before today), every reference pointing at a row that exists, and enough rows that tasks can choose among "
            "many similar records (tables tasks act on: 40 to 80 rows; reference tables: 8 to 30). Give each table people search by a "
            "near_duplicate rule and each table with a status a stale rule. Rows must satisfy the tools' preconditions often enough "
            "that most records can be acted on (e.g. most invoices pending), and the acting user's table must hold users of every role. "
            "Where a real record keeps a copy of a master value (a customer's name or tier on a request, a supplier's name on an "
            "order, an employee's department on a leave request), generate the copy with lookup, as real systems denormalise. "
            "State every precondition the tools check as a rule, including ones across records (a loss date within the referenced "
            "policy's period: loss_date <= policy_id.expiry_date); the generator redraws rows that break a rule, so most records "
            "end up valid for the tools that act on them.\n"
            "Reply with the generator JSON object (keys tables, notes)."))

DOC_WRITER = Role(
    name="doc_writer", stage="W3", output=DocsOut, temperature=0.4, max_tokens=24000,
    sees=("domain_spec", "design", "policy_params", "data_notes", "doc_format"),
    never=("tests", "tasks", "contract"),
    system=("You write the internal documents of a company's department: the policy, the standard operating procedures, an FAQ and a "
            "data-standards note, as a real company would write them (headings, numbered steps, who approves what, examples). State every "
            "policy parameter where its rule is explained, written as {{parameter_name}} (the system fills in the current value; never write "
            "the value itself). The data-standards note says how to recognise and treat near-duplicate records, stale or superseded "
            "records, and which source wins when two records disagree (the master record).\n"
            'Reply {"documents": [{"doc_id": "DOC-...", "title", "kind": "policy|sop|faq|data_standards", "tags": [...], "body": "<markdown>"}]} '
            "with 4 to 8 documents."))

ACTOR_WRITER = Role(
    name="actor_writer", stage="W4", output=ActorsOut, temperature=0.5, max_tokens=8192,
    sees=("domain_spec", "schema", "documents", "actor_format"),
    never=("tests", "tasks", "contract"),
    system=("You write the colleagues an employee can ask in this department, and the facts each knows that are NOT in the records: "
            "phone calls, verbal approvals, context about a customer or supplier, the reason behind an exception. Each fact is a template "
            "over one row of a table: {field} is that row's field (only fields of that table), {today} is today's date. Facts must be "
            "useful for real work (a task may need one) and consistent with the documents.\n"
            'Reply {"colleagues": [{"colleague_id": "C-...", "name", "title", "topics": [...], "facts": [{"topic", "keywords": [<2 to 4 words '
            'a question would contain>], "table": "<table>", "text": "<template>"}]}]} with 1 to 3 colleagues and 2 to 4 facts each.'))

DOC_FORMAT = ("Each body is markdown, 150 to 700 words. Write a parameter as {{name}} only (e.g. 'Invoices above {{approval_threshold}} "
              "USD need manager approval'). Use only the parameter names given. kinds: policy, sop, faq, data_standards (one of each at least).")
ACTOR_FORMAT = "A fact text like 'I spoke with {contact_name} at {name} on the phone: they will send a corrected invoice.' Only {fields of its table} and {today}."


# ---------------------------------------------------------------------------------------------------------- checks

def design_doc(design: ToolDesignOut, spec: DomainSpec) -> dict[str, Any]:
    """What the code, test and debug roles are shown as "design": the tools plus the policy parameters, the permissions
    and the platform notes (the roles' prompts are shared with ``worldgen``)."""
    return {**design.model_dump(mode="json"), "policy_params": [p.model_dump(mode="json", exclude_none=True) for p in spec.policy_params],
            "permissions": spec.permissions.model_dump(mode="json"), "business_rules": spec.rules, "platform": KIT_API_DOC,
            "platform_tables": kit_tables(["docs", "policy", "actors"])}


def with_kit(schema: SchemaOut, kinds: list[str]) -> SchemaOut:
    kt = kit_tables(kinds)
    tables = [t for t in schema.tables if t.name not in kt]
    tables += [TableD(name=n, key=v["key"], fields=dict(v["fields"]), description=v["description"]) for n, v in kt.items()]
    return SchemaOut(tables=tables)


def schema_map(schema: SchemaOut | Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tables = schema.tables if isinstance(schema, SchemaOut) else [TableD.model_validate(t) for t in schema["tables"]]
    return {t.name: {"key": t.key, "fields": dict(t.fields)} for t in tables}


def check_docs(docs: DocsOut, spec: DomainSpec, *, stale_values: list[str] | None = None) -> None:
    names = {p.name for p in spec.policy_params}
    want = {p.name for p in spec.policy_params if p.type != "ref"}
    used: set[str] = set()
    if not 4 <= len(docs.documents) <= 10:
        raise ValueError(f"write 4 to 8 documents, not {len(docs.documents)}")
    if len({d.doc_id for d in docs.documents}) != len(docs.documents):
        raise ValueError("document ids repeat")
    for d in docs.documents:
        ph = placeholders(d.body)
        bad = sorted(ph - names)
        if bad:
            raise ValueError(f"document {d.doc_id} uses unknown parameters {bad} (only {sorted(names)})")
        used |= ph
        if len(d.body) < 400:
            raise ValueError(f"document {d.doc_id} is too short to be a real policy or SOP")
    missing = sorted(want - used)
    if missing:
        raise ValueError(f"no document states the parameters {missing} as {{{{name}}}}")
    kinds = {d.kind for d in docs.documents}
    for k in ("policy", "sop", "data_standards"):
        if k not in kinds:
            raise ValueError(f"no document of kind {k}")
    ds = " ".join(d.body.lower() for d in docs.documents if d.kind == "data_standards")
    for w in ("duplicate", "stale" if not stale_values else stale_values[0].lower()):
        if w not in ds:
            raise ValueError(f"the data_standards document does not say how to treat {w} records")
    if not any(w in ds for w in ("master", "source of truth", "takes precedence", "prevails", "authoritative")):
        raise ValueError("the data_standards document does not say which source wins when two records disagree (the master record)")


def _log_attempt(app: Path, attempt: int, problem: str) -> None:
    """Every refused data generator, with its problem (``app/datagen_attempts.jsonl``): the record of why a stage spent."""
    with (app / "datagen_attempts.jsonl").open("a") as fh:
        fh.write(json.dumps({"t": time.time(), "attempt": attempt, "problem": problem[:3000]}) + "\n")


def _history(app: Path, latest: str, k: int = 4) -> str:
    """Feedback for a fresh data-generator conversation: the last ``k`` distinct logged problems plus ``latest`` (pilot 1:
    with only the last problem, the writer repeated an earlier mistake, a stale marker outside its choices, three times)."""
    seen: list[str] = []
    p = app / "datagen_attempts.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                prob = str(json.loads(line).get("problem", ""))
            except ValueError:
                continue
            prob = prob.removeprefix("An earlier generator was refused: ")
            if prob and prob not in seen:
                seen.append(prob)
    probs = [x[-600:] for x in seen[-k:]]
    head = [latest] if latest else []
    return "\n".join([*head, "Problems earlier generators had (avoid every one):", *[f"- {x}" for x in probs]]) if probs else latest


def check_permission_tests(tests: ToolTestsOut, d: ToolDesignOut, spec: DomainSpec) -> None:
    """The blind tests must drive the permission model: for at least half of the write tools that declare
    PERMISSION_DENIED (and at least one), a test whose fixture sets the acting user (policy_params) to a user without the
    authority and expects the refusal. Pilot 1: an AR App declared PERMISSION_DENIED on all eight write tools, no test
    exercised it, and the code had no permission check at all."""
    perm = {t.name for t in d.tools if not t.read_only and "PERMISSION_DENIED" in " ".join(t.errors).upper()}
    if not perm:
        return
    acting = spec.permissions.acting_param
    covered = set()
    for t in tests.tests:
        sets_actor = any(r.get("name") == acting for r in t.fixture.get("policy_params", []))
        if not sets_actor:
            continue
        covered |= {c.tool.split(".")[-1] for c in t.calls if c.expect == "error" and c.tool.split(".")[-1] in perm and "PERMISSION_DENIED" in str(c.error or "")}
    need = max(1, (len(perm) + 1) // 2)
    if len(covered) < need:
        raise ValueError(f"write permission refusals are untested: add tests for at least {need} of {sorted(perm)} (covered {sorted(covered)}) whose "
                         f"fixture has a policy_params row {{\"name\": \"{acting}\", \"value\": <a user id>}} for a user whose role or limit does not "
                         "allow the call (the fixture holds every other row the call needs, so only the permission is wrong), and whose call has "
                         "\"expect\": \"error\", \"error\": \"PERMISSION_DENIED\"")


def check_actors(a: ActorsOut, schema: Mapping[str, Mapping[str, Any]]) -> None:
    import re

    if not 1 <= len(a.colleagues) <= 4:
        raise ValueError("write 1 to 3 colleagues")
    if len({c.colleague_id for c in a.colleagues}) != len(a.colleagues):
        raise ValueError("colleague ids repeat")
    for c in a.colleagues:
        if not c.facts:
            raise ValueError(f"colleague {c.colleague_id} knows no facts")
        for f in c.facts:
            if f.table not in schema or f.table in ("documents", "policy_params", "colleagues"):
                raise ValueError(f"fact {f.topic!r}: table {f.table!r} is not a data table of the schema")
            if not f.keywords:
                raise ValueError(f"fact {f.topic!r} has no keywords")
            bad = sorted(set(re.findall(r"\{(\w+)\}", f.text)) - set(schema[f.table]["fields"]) - {"today"})
            if bad:
                raise ValueError(f"fact {f.topic!r}: {bad} are not fields of {f.table}")


def data_notes(spec: DataSpec) -> dict[str, Any]:
    """What the doc writer is told about the data: enumerations, lookups (master copies), stale markers."""
    enums, lookups, stale = {}, [], []
    for t in spec.tables:
        for f, g in t.fields.items():
            if g.get("kind") == "choice":
                enums[f"{t.table}.{f}"] = g["values"]
            if g.get("kind") == "lookup":
                ref = t.fields[str(g["ref"])]
                lookups.append(f"{t.table}.{f} copies {ref['table']}.{g['field']} (the {ref['table']} record is the master)")
        if t.stale and t.stale.set:
            stale.append(f"{t.table}: outdated records have {t.stale.set}")
    return {"enumerations": enums, "copied_fields": lookups, "stale_markers": stale}


def targets(design: Mapping[str, Any], schema: Mapping[str, Any] | None = None) -> list[str]:
    """Collections tasks may act on: those the family's write tools write **and** name by key (a write tool takes the
    table's key field, or ``<table>_id`` / ``<singular>_id``, as a parameter). An audit or history table the tools only
    append to is written but not a target. Without a schema: every written collection."""
    kit = {"documents", "policy_params", "colleagues"}
    written = {c for t in design.get("tools", []) if not t.get("read_only") for c in t.get("writes") or []} - kit
    if schema is None:
        return sorted(written)
    keys = {t["name"]: t["key"] for t in schema.get("tables", [])}
    params = {p["name"] for t in design.get("tools", []) if not t.get("read_only") for p in t.get("params", [])}
    out = set()
    for c in written:
        names = {keys.get(c, ""), f"{c}_id", f"{c.rstrip('s')}_id", f"{c[:-3]}y_id" if c.endswith("ies") else ""}
        if names & params:
            out.add(c)
    return sorted(out or written)


# ---------------------------------------------------------------------------------------------------------- state

class StageRecord(_M):
    status: str = "pending"  # done | failed
    started: float | None = None
    elapsed_s: float = 0.0
    usd: float = 0.0
    attempts: int = 0
    note: str = ""


class FamilyState(_M):
    family: str
    keyword: str
    function: str = ""
    status: str = "building"  # building | frozen | rejected:<stage>
    reasons: list[str] = Field(default_factory=list)
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    families: dict[str, str] = Field(default_factory=dict)  # role -> model family
    sandbox: str = SANDBOX_KIND
    tests_written: int = 0
    tests_dropped: list[str] = Field(default_factory=list)
    pass_history: list[float] = Field(default_factory=list)
    debug_rounds: int = 0
    tools: int = 0
    tables: int = 0
    elapsed_s: float = 0.0
    usd: float = 0.0


class FamilyConfig(_M):
    max_debug_rounds: int = 3
    max_drop_share: float = 0.25
    k_min: int = 40  # rows per targetable collection in every instance (density)
    data_attempts: int = 3
    actors: bool = True
    guidance: str = ""


def family_id(keyword: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")[:48]


def load_state(fdir: Path) -> FamilyState | None:
    p = fdir / "state.json"
    return FamilyState.model_validate_json(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------------------------------------------------- build

async def build_family(keyword: str, fdir: str | Path, clients: Mapping[str, RoleClient], *, function: str = "", cfg: FamilyConfig | None = None,
                       spent: Callable[[], float] | None = None, restart_from: str | None = None, redo: Sequence[str] = ()) -> FamilyState:
    """Build (or resume) the family of ``keyword`` under ``fdir`` (module docstring). ``clients``: one ``RoleClient`` per
    ``FAMILY_ROLES`` entry (the wire checks their families). ``spent()`` is the run meter's total (stage costs).
    ``restart_from``: rebuild that stage and every later one (the old App directory is copied aside first, never
    deleted; the earlier stages' files are kept), e.g. after a check is added that an earlier build would fail."""
    cfg = cfg or FamilyConfig()
    fdir = Path(fdir)
    app = fdir / "app"
    st = load_state(fdir) or FamilyState(family=fdir.name, keyword=keyword, function=function)
    if (restart_from is not None or redo) and st.stages:
        if restart_from is not None and restart_from not in STAGES:
            raise ValueError(f"unknown stage {restart_from!r}")
        bad = sorted(set(redo or ()) - set(STAGES))
        if bad:
            raise ValueError(f"unknown stages {bad}")
        if app.exists():
            shutil.copytree(app, fdir / f"app.prev-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}")
        reset = list(STAGES[STAGES.index(restart_from):]) if restart_from is not None else list(redo or ())
        for s in reset:
            if s in st.stages:
                st.stages[s].status = "pending"
        st.status, st.reasons = "building", []
        if (app / "world.json").exists():  # no longer frozen until the freeze stage runs again (the copy aside keeps it)
            (app / "world.json").replace(app / "world.json.superseded")
    app.mkdir(parents=True, exist_ok=True)
    st.families = {r: getattr(clients[r], "family", "?") for r in FAMILY_ROLES if r in clients}
    if st.status.startswith("rejected"):  # a resumed rejected family restarts at the failed stage
        failed = next((s for s, r in st.stages.items() if r.status == "failed"), None)
        if failed:
            st.stages[failed].status = "pending"
        st.status, st.reasons = "building", []
    meter = spent or (lambda: 0.0)
    t_all = time.monotonic()

    def save() -> None:
        st.usd = round(sum(r.usd for r in st.stages.values()), 6)
        _jwrite(fdir / "state.json", st.model_dump(mode="json"))

    def done(stage: str) -> bool:
        r = st.stages.get(stage)
        return r is not None and r.status == "done"

    async def stage(name: str, fn: Callable[[], Awaitable[None]]) -> bool:
        if done(name):
            return True
        r = st.stages.setdefault(name, StageRecord())
        r.started, r.attempts = time.time(), r.attempts + 1
        t0, u0 = time.monotonic(), meter()
        try:
            await fn()
            r.status = "done"
        except Exception as e:
            if type(e).__name__ in ("BudgetExceeded", "TinkerCapReached"):
                raise
            r.status, r.note = "failed", f"{type(e).__name__}: {str(e)[:1200]}"
            st.status = f"rejected:{name}"
            st.reasons.append(r.note)
        finally:
            r.elapsed_s = round(r.elapsed_s + time.monotonic() - t0, 2)
            r.usd = round(r.usd + meter() - u0, 6)
            save()
        return r.status == "done"

    kinds = ["docs", "policy", *(["actors"] if cfg.actors else [])]
    box: dict[str, Any] = {}

    def spec() -> DomainSpec:
        return DomainSpec.model_validate_json((app / "domain.json").read_text())

    def design() -> ToolDesignOut:
        return ToolDesignOut.model_validate(_jread(app / "design.json"))

    def schema() -> SchemaOut:
        return SchemaOut.model_validate(_jread(app / "schema.json"))

    async def s_spec() -> None:
        sp = await generate_spec(keyword, clients["domain_spec_writer"], function=function, guidance=cfg.guidance)
        _jwrite(app / "domain.json", sp.model_dump(mode="json"))

    async def s_design() -> None:
        sp = spec()

        def chk(d: ToolDesignOut) -> None:
            check_design(d)
            probs = realism_design_problems(d, sp.permissions.acting_param)
            if probs:
                raise ValueError("; ".join(probs))

        d: ToolDesignOut = await ask(clients["family_designer"], FAMILY_DESIGNER, tag="family:design", check=chk, retries=2,
                                     domain_spec=sp.model_dump(mode="json", exclude={"apps"}), realism=REALISM_SURFACE, guidance=cfg.guidance or None)
        _jwrite(app / "design.json", d.model_dump(mode="json"))
        _jwrite(app / "kit.json", {"kinds": kinds})
        st.tools = len(d.tools)

    async def s_schema() -> None:
        d, sp = design(), spec()
        user_table = sp.permissions.user_table

        def chk(s: SchemaOut) -> None:
            s = SchemaOut(tables=[t for t in s.tables if t.name not in ("documents", "policy_params", "colleagues")])
            check_schema(s, d)
            if user_table and user_table not in {t.name for t in s.tables}:
                raise ValueError(f"the user table {user_table!r} (the acting user's) must be a table")

        s: SchemaOut = await ask(clients["schema_deriver"], SCHEMA_DERIVER, tag="family:schema", check=chk, retries=2, design=design_doc(d, sp))
        full = with_kit(s, kinds)
        _jwrite(app / "schema.json", full.model_dump(mode="json"))
        st.tables = len(full.tables)

    async def s_code() -> None:
        d, sp, s = design(), spec(), schema()

        def code_ok(c: AppCodeOut) -> None:
            probs = check_code(c.code)
            if probs:
                raise ValueError("the module fails the sandbox's static check: " + "; ".join(probs[:8]))

        c: AppCodeOut = await ask(clients["app_coder"], FAMILY_APP_CODER, tag="family:code", check=code_ok, retries=2, design=design_doc(d, sp),
                                  schema=s.model_dump(mode="json"), app_api=APP_API_DOC + "\n" + KIT_API_DOC)
        (app / "app.py").write_text(c.code)

    async def s_tests() -> None:
        d, sp, s = design(), spec(), schema()
        def chk(t: ToolTestsOut) -> None:
            check_tests(t, d, s)
            check_permission_tests(t, d, sp)

        t: ToolTestsOut = await ask(clients["blind_tester"], BLIND_TESTER, tag="family:tests", check=chk, retries=2,
                                    design=design_doc(d, sp), schema=s.model_dump(mode="json"), test_format=TEST_FORMAT_DOC)
        _jwrite(app / "tests.json", t.model_dump(mode="json"))
        st.tests_written = len(t.tests)
        box["frozen"] = _sha(app / "tests.json")
        st.stages.setdefault("tests", StageRecord()).note = f"frozen {box['frozen']}"

    async def s_debug() -> None:
        d, sp, s = design(), spec(), schema()
        tests = ToolTestsOut.model_validate(_jread(app / "tests.json"))
        frozen = st.stages["tests"].note.removeprefix("frozen ").strip() or _sha(app / "tests.json")
        if _sha(app / "tests.json") != frozen:
            raise ValueError("the frozen tests file changed since it was written")
        rep = WorldgenReport(keyword=keyword, app=d.app, families=st.families)
        dropped = await test_and_debug(app, d, design_doc(d, sp), s.model_dump(mode="json"), AppCodeOut(code=(app / "app.py").read_text()), tests, frozen,
                                       clients, rep, max_debug_rounds=cfg.max_debug_rounds, max_drop_share=cfg.max_drop_share,
                                       exclude_tools=KIT_NAMES, app_api=APP_API_DOC + "\n" + KIT_API_DOC, tag="family", debugger_role=FAMILY_DEBUGGER)
        st.tests_dropped, st.pass_history, st.debug_rounds = dropped, list(rep.pass_history), rep.debug_rounds
        if rep.status.startswith("rejected"):
            raise ValueError("; ".join(rep.reasons))

    async def s_datagen() -> None:
        d, sp, s = design(), spec(), schema()
        sm = schema_map(s)
        data_tables = {k: v for k, v in sm.items() if k not in ("documents", "policy_params", "colleagues")}
        tg = targets(d.model_dump(mode="json"), s.model_dump(mode="json"))
        feedback = ""
        log = app / "datagen_attempts.jsonl"
        if log.exists() and log.read_text().strip():  # a resumed stage: start from the logged problems (the wire samples at a fixed seed,
            feedback = _history(app, "")  # so the same prompt would return the same refused generator: pilot 1, insurance, twice)
        for attempt in range(cfg.data_attempts):
            def chk(x: DataSpec, attempt: int = attempt) -> None:
                missing = sorted(set(data_tables) - {t.table for t in x.tables})
                if missing:
                    raise ValueError(f"no generator for tables {missing}")
                x = DataSpec(tables=[t for t in x.tables if t.table in data_tables], notes=x.notes)
                # every instance profile at its own seed (pilot 1: a spec that sampled at seed 0 failed at 3 of 4 instance seeds)
                for sd, pname in ((0, "standard"), (1, "dense"), (2, "noisy"), (3, "clean")):
                    try:
                        base = PROFILES[pname]
                        g = generate(x, data_tables, seed=sd, profile=base.model_copy(update={"min_rows": {t: round(cfg.k_min * max(1.0, base.scale)) for t in tg}}))
                        probs = integrity(x, data_tables, g)
                        if probs:
                            raise DataSpecError(f"the sampled data ({pname} profile) breaks its own checks: " + "; ".join(probs[:12]))
                        for p in (p for p in sp.policy_params if p.type == "ref"):
                            if p.table not in data_tables:
                                raise DataSpecError(f"policy parameter {p.name} refers to table {p.table}, which has no generator")
                            if not _ref_candidates(g, data_tables, p):
                                raise DataSpecError(f"policy parameter {p.name}: no {p.table} row matches {p.where}")
                    except DataSpecError as e:
                        _log_attempt(app, attempt, f"{pname}: {e}")
                        raise

            try:
                ds: DataSpec = await ask(clients["data_spec_writer"], DATA_SPEC_WRITER, tag="family:datagen", check=chk, retries=2,
                                         domain_spec=sp.model_dump(mode="json", exclude={"apps"}), design=design_doc(d, sp), schema=s.model_dump(mode="json"),
                                         datagen_format=DATAGEN_FORMAT, feedback=feedback or None)
            except RoleFailed as e:  # a fresh conversation with the last problem (pilot 1: one long repair thread kept failing)
                feedback = _history(app, f"An earlier generator was refused: {str(e)[-1200:]}")
                _log_attempt(app, attempt, feedback)
                if attempt == cfg.data_attempts - 1:
                    raise ValueError(f"data generator: {e}") from None
                continue

            ds = DataSpec(tables=[t for t in ds.tables if t.table in data_tables], notes=ds.notes)
            _jwrite(app / "datagen.json", ds.model_dump(mode="json"))
            # loaded and smoke-tested with the kit tables of a default instance (no documents yet: an empty library)
            probe = fdir / "work" / f"dg-{uuid.uuid4().hex[:8]}"
            _materialize(app, probe, ds, sp, s, InstanceConfig(family=st.family, index=0, seed=0, profile="standard"), k_min=cfg.k_min, docs=None, actors=None)
            try:
                probs = await smoke_world(probe, probe / "work")
            except (SandboxError, ValueError) as e:
                probs = [f"the start state does not load: {str(e)[:600]}"]
            if not probs:
                return
            feedback = "The sampled start state broke the application: " + "; ".join(probs[:6])
            _log_attempt(app, attempt, feedback)
        raise ValueError(f"data generator: {feedback}")

    async def s_docs() -> None:
        d, sp = design(), spec()
        ds = DataSpec.model_validate(_jread(app / "datagen.json"))
        stale_vals = [str(v) for t in ds.tables if t.stale for v in t.stale.set.values() if isinstance(v, str)]
        docs: DocsOut = await ask(clients["doc_writer"], DOC_WRITER, tag="family:docs", check=lambda x: check_docs(x, sp, stale_values=stale_vals[:1]),
                                  retries=2, domain_spec=sp.model_dump(mode="json", exclude={"apps"}), design={"app": d.app, "tools": [t.name for t in d.tools]},
                                  policy_params=[p.model_dump(mode="json", exclude_none=True) for p in sp.policy_params], data_notes=data_notes(ds),
                                  doc_format=DOC_FORMAT)
        _jwrite(app / "docs.json", docs.model_dump(mode="json"))

    async def s_actors() -> None:
        sp, s = spec(), schema()
        if not cfg.actors or not sp.colleagues:
            _jwrite(app / "actors.json", {"colleagues": []})
            return
        docs = DocsOut.model_validate(_jread(app / "docs.json"))
        sm = schema_map(s)
        a: ActorsOut = await ask(clients["actor_writer"], ACTOR_WRITER, tag="family:actors", check=lambda x: check_actors(x, sm), retries=2,
                                 domain_spec=sp.model_dump(mode="json", include={"name", "description", "colleagues", "entities"}),
                                 schema={k: list(v["fields"]) for k, v in sm.items() if k not in ("documents", "policy_params", "colleagues")},
                                 documents=[{"doc_id": x.doc_id, "title": x.title} for x in docs.documents], actor_format=ACTOR_FORMAT)
        _jwrite(app / "actors.json", a.model_dump(mode="json"))

    async def s_base() -> None:
        inst = fdir / "work" / f"base-{uuid.uuid4().hex[:8]}"
        cfg0 = InstanceConfig(family=st.family, index=0, seed=0, profile="standard")
        _materialize(app, inst, DataSpec.model_validate(_jread(app / "datagen.json")), spec(), schema(), cfg0, k_min=cfg.k_min,
                     docs=DocsOut.model_validate(_jread(app / "docs.json")), actors=ActorsOut.model_validate(_jread(app / "actors.json")))
        probs = await smoke_world(inst, inst / "work")
        if probs:
            raise ValueError("the base instance breaks read tools: " + "; ".join(probs[:5]))

    async def s_freeze() -> None:
        sp = bind_design(spec(), _jread(app / "design.json"), _jread(app / "schema.json"))
        _jwrite(app / "domain.json", sp.model_dump(mode="json"))
        files = {f: _sha(app / f) for f in FROZEN_FILES if (app / f).exists()}
        man = {"family": st.family, "keyword": keyword, "function": function, "files": files, "tests_digest": files.get("tests.json"),
               "tests": st.tests_written, "dropped": st.tests_dropped, "debug_rounds": st.debug_rounds, "families": st.families,
               "sandbox": SANDBOX_KIND, "kit": kinds, "k_min": cfg.k_min, "frozen_at": time.time()}
        _jwrite(app / "world.json", man)
        rep = WorldgenReport(keyword=keyword, app=_jread(app / "design.json")["app"], status="accepted", tools=st.tools, tables=st.tables,
                             tests_written=st.tests_written, tests_dropped=st.tests_dropped, debug_rounds=st.debug_rounds, pass_history=st.pass_history,
                             tests_digest=files.get("tests.json"), families=st.families, dir=str(app))
        _jwrite(app / "worldgen_report.json", rep.model_dump(mode="json"))

    for name, fn in (("spec", s_spec), ("design", s_design), ("schema", s_schema), ("code", s_code), ("tests", s_tests), ("debug", s_debug),
                     ("datagen", s_datagen), ("docs", s_docs), ("actors", s_actors), ("base", s_base), ("freeze", s_freeze)):
        if not await stage(name, fn):
            break
    else:
        st.status = "frozen"
    st.elapsed_s = round(st.elapsed_s + time.monotonic() - t_all, 2)
    save()
    return st


# ---------------------------------------------------------------------------------------------------------- instances

class InstanceConfig(_M):
    family: str
    index: int
    seed: int
    profile: str = "standard"
    params: dict[str, Any] = Field(default_factory=dict)  # the policy variant (resolved values; empty = drawn from the seed)
    colleagues: list[str] = Field(default_factory=list)  # the actor config (empty = drawn from the seed)

    @property
    def id(self) -> str:
        return f"i{self.index:02d}-{self.profile}-s{self.seed}"


def instance_config(family: str, index: int, *, base_seed: int = 0, profile: str | None = None) -> InstanceConfig:
    """The default config of instance ``index``: a seed derived from (family, index, base_seed), the profiles cycled."""
    seed = int(hashlib.sha256(f"{family}/{index}/{base_seed}".encode()).hexdigest()[:8], 16) % 1_000_000
    return InstanceConfig(family=family, index=index, seed=seed, profile=profile or PROFILE_CYCLE[index % len(PROFILE_CYCLE)])


def _nice(v: float) -> float:
    if v == 0:
        return 0
    import math

    mag = 10 ** math.floor(math.log10(abs(v)))
    for step in (1, 2, 2.5, 5, 10):
        if abs(v) <= step * mag:
            return math.copysign(step * mag, v)
    return v


def _ref_candidates(g: Generated, schema: Mapping[str, Mapping[str, Any]], p: PolicyParam) -> list[Any]:
    rows = g.tables.get(str(p.table), [])
    kf = schema[str(p.table)]["key"]
    noise = {str(n.key) for n in g.noise if n.table == p.table and n.kind in ("near_duplicate", "stale")}
    return [r[kf] for r in rows if str(r[kf]) not in noise and all((r.get(a) in b) if isinstance(b, list) else r.get(a) == b for a, b in (p.where or {}).items())]


def _operator_candidates(spec: DomainSpec, g: Generated, schema: Mapping[str, Mapping[str, Any]], p: PolicyParam, cands: list[Any], index: int) -> list[Any]:
    """The actor config's acting user: someone who does the domain's main work. The spec's roles are ranked by how
    much they may do; every instance acts as a user of the top role (instances differ in which user), drawn among
    the eligible users holding that role. Pilot 1: a seeded draw made a returns instance act as a Finance Clerk, so
    every write tool refused with PERMISSION_DENIED and no task could act."""
    roles = sorted(spec.permissions.roles, key=lambda r: -len(r.may))
    if not roles:
        return []
    want = roles[0].role  # pilot 1: cycling to the second role made a leave instance act as an Employee (submit only)
    rows = g.tables.get(str(p.table), [])
    kf = schema[str(p.table)]["key"]
    allowed = {str(c) for c in cands}

    def norm(x: Any) -> str:
        return str(x).strip().lower().replace("_", " ")

    hit = [r[kf] for r in rows if str(r[kf]) in allowed and any(norm(v) == norm(want) for v in r.values() if isinstance(v, str))]
    return hit


def policy_variant(spec: DomainSpec, cfg: InstanceConfig, g: Generated, schema: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The instance's policy parameter values: the spec's defaults for instance 0, else a seeded draw (numbers: the
    default scaled by 0.5 to 2, rounded to a nice value, clipped to [min, max]; choices uniformly; booleans flipped half
    the time), redrawn until the spec's ``param_constraints`` hold (else the defaults); ref parameters (the acting user)
    always a seeded eligible row. ``cfg.params`` overrides. Pilot 1: draws spread over the whole range gave a 50%
    restocking fee and windows out of order."""
    from crucible.synth.datagen import DataSpecError, eval_expr

    rng = random.Random(f"policy/{cfg.family}/{cfg.seed}")
    for _ in range(30):
        out = _draw_params(spec, cfg, g, schema, rng)
        try:
            if all(eval_expr(c, out) for c in spec.param_constraints):
                return out
        except DataSpecError:
            break
    return {**_draw_params(spec, cfg.model_copy(update={"index": 0}), g, schema, rng), **cfg.params}


def _draw_params(spec: DomainSpec, cfg: InstanceConfig, g: Generated, schema: Mapping[str, Mapping[str, Any]], rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in spec.policy_params:
        if p.name in cfg.params:
            out[p.name] = cfg.params[p.name]
            continue
        if p.type == "ref":
            cands = _ref_candidates(g, schema, p)
            if p.name == spec.permissions.acting_param:
                cands = _operator_candidates(spec, g, schema, p, cands, cfg.index) or cands
            out[p.name] = rng.choice(cands) if cands else p.default
            continue
        if cfg.index == 0:
            out[p.name] = p.default
            continue
        if p.type in ("number", "integer") and isinstance(p.default, int | float):
            lo = p.min if p.min is not None else -float("inf")
            hi = p.max if p.max is not None else float("inf")
            cand = sorted({min(hi, max(lo, _nice(p.default * f))) for f in (0.5, 0.75, 1.25, 1.5, 2.0)} | {p.default})
            v = rng.choice(cand)
            out[p.name] = round(v) if p.type == "integer" or isinstance(p.default, int) else v
        elif p.choices:
            out[p.name] = rng.choice(list(p.choices))
        elif p.type == "boolean" and isinstance(p.default, bool):
            out[p.name] = (not p.default) if rng.random() < 0.5 else p.default
        else:
            out[p.name] = p.default
    return out


def _render_docs(docs: DocsOut | None, params: dict[str, Any], cfg: InstanceConfig, spec: DomainSpec) -> list[dict[str, Any]]:
    if docs is None:
        return []
    units = {p.name: p.unit for p in spec.policy_params}
    rng = random.Random(f"docs/{cfg.family}/{cfg.seed}")
    out = []
    for d in docs.documents:
        eff = (TODAY - timedelta(days=rng.randint(20, 400))).strftime("%Y-%m-%d")
        out.append({"doc_id": d.doc_id, "title": d.title, "kind": d.kind, "tags": list(d.tags), "body": render(d.body, params, units),
                    "version": f"{rng.randint(1, 4)}.{rng.randint(0, 9)}", "effective_date": eff})
    return out


def _bind_actors(actors: ActorsOut | None, g: Generated, schema: Mapping[str, Mapping[str, Any]], cfg: InstanceConfig) -> list[dict[str, Any]]:
    if actors is None or not actors.colleagues:
        return []
    rng = random.Random(f"actors/{cfg.family}/{cfg.seed}")
    pool = list(actors.colleagues)
    chosen = [c for c in pool if c.colleague_id in cfg.colleagues] if cfg.colleagues else pool if len(pool) <= 2 else rng.sample(pool, max(2, len(pool) - 1))
    today = TODAY.strftime("%Y-%m-%d")
    noise = {(n.table, str(n.key)) for n in g.noise if n.kind in ("near_duplicate", "stale")}
    out = []
    for c in chosen:
        facts = []
        for f in c.facts:
            kf = schema[f.table]["key"]
            rows = [r for r in g.tables.get(f.table, []) if (f.table, str(r[kf])) not in noise]
            if not rows:
                continue
            row = rng.choice(rows)
            try:
                text = f.text.format(**{**{k: ("" if v is None else v) for k, v in row.items()}, "today": today})
            except (KeyError, IndexError, ValueError):
                continue
            facts.append({"topic": f.topic, "keywords": list(f.keywords), "text": text, "about": {"table": f.table, "key": row[kf]}})
        out.append({"colleague_id": c.colleague_id, "name": c.name, "title": c.title, "topics": list(c.topics or [f.topic for f in c.facts]),
                    "facts": facts})
    return out


def _materialize(app: Path, out: Path, ds: DataSpec, spec: DomainSpec, schema: SchemaOut, cfg: InstanceConfig, *, k_min: int,
                 docs: DocsOut | None, actors: ActorsOut | None) -> dict[str, Any]:
    """Write one instance App directory: the family's files copied, then the generated start state (module docstring)."""
    out.mkdir(parents=True, exist_ok=True)
    for f in ("design.json", "schema.json", "app.py", "tests.json", "test_report.json", "kit.json", "domain.json"):
        if (app / f).exists():
            shutil.copyfile(app / f, out / f)
    sm = schema_map(schema)
    data_tables = {k: v for k, v in sm.items() if k not in ("documents", "policy_params", "colleagues")}
    design = _jread(app / "design.json")
    base = PROFILES.get(cfg.profile, PROFILES["standard"])
    prof = base.model_copy(update={"min_rows": {t: round(k_min * max(1.0, base.scale)) for t in targets(design, schema.model_dump(mode="json")) if t in data_tables}})
    g = generate(ds, data_tables, seed=cfg.seed, profile=prof)
    params = policy_variant(spec, cfg, g, data_tables)
    tables: dict[str, list[dict[str, Any]]] = dict(g.tables)
    if "policy_params" in sm:
        tables["policy_params"] = [{"name": p.name, "value": params.get(p.name, p.default), "unit": p.unit, "description": p.meaning} for p in spec.policy_params]
    if "documents" in sm:
        tables["documents"] = _render_docs(docs, params, cfg, spec)
    if "colleagues" in sm:
        tables["colleagues"] = _bind_actors(actors, g, data_tables, cfg)
    _jwrite(out / "seed.json", {"tables": tables, "notes": f"instance {cfg.id} of {cfg.family} (profile {prof.name})"})
    _jwrite(out / "noise.json", [n.model_dump(mode="json") for n in g.noise])
    return {"params": params, "rows": {t: len(r) for t, r in tables.items()}, "noise": _noise_counts(g), "capped": g.capped}


def _noise_counts(g: Generated) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in g.noise:
        out[n.kind] = out.get(n.kind, 0) + 1
    return out


class InstanceReport(_M):
    id: str
    family: str
    dir: str
    config: InstanceConfig
    params: dict[str, Any] = Field(default_factory=dict)
    rows: dict[str, int] = Field(default_factory=dict)
    noise: dict[str, int] = Field(default_factory=dict)
    seed_digest: str = ""
    capped: dict[str, list[int]] = Field(default_factory=dict)
    elapsed_s: float = 0.0


def frozen_manifest(fdir: Path) -> dict[str, Any]:
    p = Path(fdir) / "app" / "world.json"
    if not p.exists():
        raise ValueError(f"family {Path(fdir).name} is not frozen")
    man = _jread(p)
    for f, h in man["files"].items():
        if _sha(Path(fdir) / "app" / f) != h:
            raise ValueError(f"family {Path(fdir).name}: {f} changed since it was frozen")
    return man


def make_instance(fdir: str | Path, cfg: InstanceConfig, *, k_min: int | None = None) -> InstanceReport:
    """Generate one instance of a frozen family (module docstring): no model call, the family's files re-hashed."""
    t0 = time.monotonic()
    fdir = Path(fdir)
    man = frozen_manifest(fdir)
    app = fdir / "app"
    out = fdir / "instances" / cfg.id
    info = _materialize(app, out, DataSpec.model_validate(_jread(app / "datagen.json")), DomainSpec.model_validate_json((app / "domain.json").read_text()),
                        SchemaOut.model_validate(_jread(app / "schema.json")), cfg, k_min=int(k_min or man.get("k_min", 40)),
                        docs=DocsOut.model_validate(_jread(app / "docs.json")), actors=ActorsOut.model_validate(_jread(app / "actors.json")))
    for f in ("app.py", "tests.json", "design.json", "schema.json"):
        if _sha(out / f) != man["files"][f]:
            raise ValueError(f"instance {cfg.id}: copied {f} differs from the frozen family")
    rep0 = _jread(app / "worldgen_report.json")
    _jwrite(out / "worldgen_report.json", {**rep0, "dir": str(out), "reasons": [f"instance {cfg.id} of family {cfg.family}"]})
    rep = InstanceReport(id=cfg.id, family=cfg.family, dir=str(out), config=cfg, params=info["params"], rows=info["rows"], noise=info["noise"], capped=info["capped"],
                         seed_digest=_sha(out / "seed.json"), elapsed_s=round(time.monotonic() - t0, 3))
    _jwrite(out / "instance.json", rep.model_dump(mode="json"))
    _jwrite(out / "world.json", {**man, "instance": cfg.model_dump(mode="json"), "files": {**man["files"], "seed.json": rep.seed_digest},
                                 "parent": str(app)})
    return rep


def make_instances(fdir: str | Path, n: int, *, base_seed: int = 0, start: int = 0, k_min: int | None = None) -> list[InstanceReport]:
    fam = Path(fdir).name
    return [make_instance(fdir, instance_config(fam, i, base_seed=base_seed), k_min=k_min) for i in range(start, start + n)]


def list_instances(fdir: str | Path) -> list[Path]:
    d = Path(fdir) / "instances"
    return sorted(p for p in d.iterdir() if (p / "instance.json").exists()) if d.exists() else []


BUILDER_ROLES = ("domain_spec_writer", "family_designer", "schema_deriver", "app_coder", "debugger", "blind_tester", "data_spec_writer", "doc_writer",
                 "actor_writer")


class InstanceDomain:
    """A World instance as the batch runner takes it (``synth.batch.batch_domains``: an object with ``env`` and
    ``cells``): the instance's ``DomainEnv`` (its DomainSpec's description, the family's builder families as
    ``built_by``) and the DomainSpec's sampled cells, without the cells whose skill only uses thin tools (the density
    gate's ``thin_tools``: no reference oracle on them reached n >= 59)."""

    def __init__(self, inst: str | Path, work: str | Path, *, n_cells: int = 24, seed: int = 0, name: str | None = None):
        from crucible.synth.pipeline import DomainEnv
        from crucible.synth.task import WorldRef

        inst = Path(inst).resolve()
        spec = DomainSpec.model_validate_json((inst / "domain.json").read_text())
        man = _jread(inst / "world.json")
        gates = _jread(inst / "gates.json") if (inst / "gates.json").exists() else {}
        thin = set(((gates.get("gates") or {}).get("density") or {}).get("details", {}).get("thin_tools") or [])
        self.usable = bool(gates.get("usable"))
        self.thin = sorted(thin)
        cells = [c for c in spec.sample_cells(max(n_cells * 2, n_cells), seed)
                 if not (thin and (sk := next((s for s in spec.skills if s.id == c.skill), None)) and sk.tools and set(sk.tools) <= thin)]
        self.cells = [spec.cell_text(c) for c in cells[:n_cells]]
        d, w = str(inst), str(Path(work).resolve())
        self.env = DomainEnv(name=name or f"{man.get('family', inst.parent.parent.name)}-{inst.name}", description=spec.environment_text(),
                             world=lambda s: WorldRef(factory="crucible.synth.worldgen:open_world", params={"dir": d, "work": w}), seeds=(0,),
                             cells=tuple(self.cells),
                             built_by=frozenset(f for r, f in (man.get("families") or {}).items() if r in BUILDER_ROLES))


def usable_instances(worlds: str | Path) -> list[Path]:
    """Every instance under a worlds root whose gates all passed (``gates.json`` usable)."""
    out = []
    for g in sorted(Path(worlds).glob("*/instances/*/gates.json")):
        try:
            if json.loads(g.read_text()).get("usable"):
                out.append(g.parent)
        except ValueError:
            continue
    return out


WORLD_BUILD_ROLES: dict[str, Role] = {r.name: r for r in (FAMILY_DESIGNER, DATA_SPEC_WRITER, DOC_WRITER, ACTOR_WRITER)}


__all__ = [
    "ACTOR_WRITER",
    "DATA_SPEC_WRITER",
    "DOC_WRITER",
    "FAMILY_DESIGNER",
    "FAMILY_ROLES",
    "REALISM_SURFACE",
    "STAGES",
    "WORLD_BUILD_ROLES",
    "ActorsOut",
    "DocsOut",
    "FamilyConfig",
    "FamilyState",
    "InstanceConfig",
    "InstanceReport",
    "build_family",
    "check_actors",
    "check_docs",
    "design_doc",
    "family_id",
    "frozen_manifest",
    "instance_config",
    "list_instances",
    "load_state",
    "make_instance",
    "make_instances",
    "policy_variant",
    "realism_design_problems",
    "schema_map",
    "targets",
    "with_kit",
]
