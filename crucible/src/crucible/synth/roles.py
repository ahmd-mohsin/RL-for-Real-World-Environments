"""The model-driven roles of ``crucible.synth`` as data (part2-paper-core §5, §7, §10; evidence doc §3.2 F1 and §10).

A ``Role`` is a prompt template, a JSON output schema (a pydantic model), what it may see (``sees``: the only input keys
``render`` accepts), what it must never see (``never``: keys ``render`` refuses even when a caller passes them), and its
family constraints: ``differs_from`` (roles whose family it must not share), ``independent`` (never the student's
family: no Qwen model judges, attacks, audits or solves-as-reference for the Qwen student) and ``min_families`` (a
multi-model role, e.g. the two-family adjudication panel). ``family_problems`` checks a whole assignment; the binding to
models is ``factory/synth_wire.py`` (synth never imports factory), through the ``RoleClient`` protocol.

| role | stage | sees | never sees | differs from |
|---|---|---|---|---|
| ``proposer`` | F3 | domain, tool catalogue, World slice, cell, earlier proposals, lemmas | contracts, gate code, anchors | — |
| ``oracle_programmer`` | F4 | task, intent card, tool catalogue, World slice, execution errors | contracts, gate code | — |
| ``contract_writer`` | F5 | task, intent card, the skeleton contract, the oracle's Δ*, tool catalogue | gate code, candidate solutions | proposer, oracle programmer |
| ``repairer`` | §7 | task, contract, oracle, counterexamples with reason codes | gate code, falsifier code | contract writer |
| ``alt_solver`` | ALT | instruction, tool catalogue | contract, oracle, gate code | every builder |
| ``attacker`` | ATK | instruction, tool catalogue, attack strategies | contract, oracle, gate code | every builder |
| ``panel`` | adjudication | task, intent card, one probe trace, reason codes | gate code, which falsifier made the probe | every builder |
| ``abstractor`` | §9.5 | clusters of counterexamples (codes, operators, defects) | anchors, the locked set, task ids | every builder (R6) |
| ``tool_designer`` | F1 | domain keyword | — | — |
| ``schema_deriver`` | F1 | the tool design | — | — |
| ``app_coder`` | F1 | tool design, table schema, the App API | the blind tests | — |
| ``blind_tester`` | F1 | tool design, table schema | the App code | app coder |
| ``debugger`` | F1 | code, failing tests with tracebacks | (cannot edit tests or expectations) | test approver |
| ``test_approver`` | F1 | tool design, one disputed test, its failure | the App code | debugger, blind tester |
| ``seeder`` | F2 | tool design, table schema | tasks, verifiers | — |

Program and contract JSON (``parse_program``, ``program_doc``): a step argument is a plain JSON literal, ``{"$ref":
"s1.path"}`` or ``{"$fn": name, "args": [...]}``; the native ``{"op": ...}`` forms are accepted too. Every parse error is a
``ValueError`` whose text goes back to the model as feedback (it names the problem, never a gate).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crucible.actors.llm import LLMOutputError
from crucible.synth.canon import cjson
from crucible.synth.contract import AnswerCheck, Contract, EffectPattern, Pred, TraceConstraint
from crucible.synth.iface import ToolInfo
from crucible.synth.program import LIBRARY, Fn, Lit, Program, Ref, Step

# ---------------------------------------------------------------------------------------------------------- the client


@runtime_checkable
class RoleClient(Protocol):
    """One model serving one role. ``complete_json`` returns a JSON object that passed ``schema`` (a pydantic class, a
    key list or a callable), retrying with the parse error as feedback; ``tag`` attributes its spend (a stage, a task)."""

    family: str
    model: str

    async def complete_json(self, messages: list[dict[str, Any]], *, schema: Any = None, max_tokens: int = 2048, temperature: float = 0.0,
                            tag: str = "") -> dict[str, Any]: ...


class RoleLeak(ValueError):
    """A caller tried to show a role something it must not see (or something outside what it may see)."""


class _Out(BaseModel):
    model_config = ConfigDict(extra="ignore")  # models add harmless keys ("notes", "thoughts"); the parts we read are typed


# ---------------------------------------------------------------------------------------------------------- output schemas

class ProposedTask(_Out):
    skill: str
    instruction: str
    given: list[Any] = Field(default_factory=list)
    derive: list[Any] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("instruction")
    @classmethod
    def _ins(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("the instruction is too short to be a task")
        return v.strip()


class ProposalOut(_Out):
    tasks: list[ProposedTask]


class OracleOut(_Out):
    steps: list[dict[str, Any]]
    answer: Any = None
    notes: str = ""


class ContractEdit(_Out):
    op: Literal["relax", "allow", "demote", "count", "answer", "trace", "normalize", "key", "keep"]
    target: str | None = None
    field: str | None = None
    pred: dict[str, Any] | None = None
    pattern: dict[str, Any] | None = None
    check: dict[str, Any] | None = None
    constraint: dict[str, Any] | None = None
    normalizers: dict[str, Any] | None = None
    count_max: int | Literal["any"] | None = None
    key: str | None = None
    why: str | None = None


class ContractOut(_Out):
    edits: list[ContractEdit] = Field(default_factory=list)
    notes: str = ""


class RepairOut(_Out):
    give_up: bool = False
    instruction: str | None = None
    oracle: dict[str, Any] | None = None
    contract: dict[str, Any] | None = None
    given: list[Any] | None = None
    derive: list[Any] | None = None
    notes: str = ""


class AgentTurn(_Out):
    """One turn of an ALT or ATK agent: a tool call, or the final answer (``final`` present, possibly null)."""

    call: dict[str, Any] | None = None
    final: Any = None
    done: bool = False

    @model_validator(mode="before")
    @classmethod
    def _shapes(cls, v: Any) -> Any:
        """Models write a call several ways: {"call": {"tool", "args"}}, {"tool", "args"}, {"call": {"name", "arguments"}}."""
        if not isinstance(v, dict):
            return v
        v = dict(v)
        if v.get("call") is None and isinstance(v.get("tool"), str):
            v["call"] = {"tool": v.pop("tool"), "args": v.pop("args", v.pop("arguments", {}))}
        c = v.get("call")
        if isinstance(c, dict) and "tool" not in c and isinstance(c.get("name"), str):
            v["call"] = {"tool": c["name"], "args": c.get("arguments", c.get("args", {}))}
        if isinstance(v.get("call"), dict) and isinstance(v["call"].get("args"), str):
            try:
                v["call"]["args"] = json.loads(v["call"]["args"])
            except ValueError:
                pass
        return v

    def is_final(self) -> bool:
        return self.done or (self.call is None)


class PanelOut(_Out):
    valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    why: str = ""


class Lemma(_Out):
    applies_to: Literal["proposer", "oracle_programmer", "contract_writer", "repairer"]
    rule: str
    evidence: list[str] = Field(default_factory=list)


class AbstractOut(_Out):
    lemmas: list[Lemma] = Field(default_factory=list)


class ToolParamD(_Out):
    name: str
    type: Literal["string", "integer", "number", "boolean", "array", "object"] = "string"
    required: bool = True
    description: str = ""
    enum: list[Any] | None = None
    default: Any = None


class ToolD(_Out):
    name: str
    description: str
    params: list[ToolParamD] = Field(default_factory=list)
    read_only: bool = False
    writes: list[str] = Field(default_factory=list)  # collection names this tool may write
    pre: list[str] = Field(default_factory=list)
    post: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    returns: Any = None  # the shape of the result, as an example object

    @field_validator("name")
    @classmethod
    def _ident(cls, v: str) -> str:
        if not v.isidentifier() or v.startswith("_"):
            raise ValueError(f"tool name {v!r} is not a public Python identifier")
        return v


class ToolDesignOut(_Out):
    app: str
    description: str
    tools: list[ToolD]

    @field_validator("app")
    @classmethod
    def _app(cls, v: str) -> str:
        import re

        v = re.sub(r"\W+", "_", v.strip().lower()).strip("_")  # "it-helpdesk-triage" -> "it_helpdesk_triage"
        if not v.isidentifier():
            raise ValueError(f"app name {v!r} is not an identifier")
        return v


class TableD(_Out):
    name: str
    key: str
    fields: dict[str, str]  # field -> string | integer | number | boolean | array | object (| null allowed by "?")
    description: str = ""


class SchemaOut(_Out):
    tables: list[TableD]


class AppCodeOut(_Out):
    code: str
    notes: str = ""


class ToolTestCall(_Out):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    expect: Literal["ok", "error"] = "ok"
    result: Any = None  # a subset the result must contain (None: unchecked)


class ToolTest(_Out):
    id: str
    tier: Literal["simple", "medium", "complex", "boundary"] = "simple"
    fixture: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # table -> rows
    calls: list[ToolTestCall]
    expect_state: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)  # table -> key -> field subset
    expect_absent: dict[str, list[str]] = Field(default_factory=dict)  # table -> keys that must not exist


class ToolTestsOut(_Out):
    tests: list[ToolTest]


class Dispute(_Out):
    test: str
    why: str


class DebugOut(_Out):
    code: str
    disputes: list[Dispute] = Field(default_factory=list)
    notes: str = ""


class ApproveOut(_Out):
    test_is_wrong: bool
    why: str = ""


class SeedOut(_Out):
    tables: dict[str, list[dict[str, Any]]]
    notes: str = ""


# ---------------------------------------------------------------------------------------------------------- the role type

class Role(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    stage: str
    system: str
    output: Any  # a pydantic model class
    sees: tuple[str, ...]
    never: tuple[str, ...] = ()
    differs_from: tuple[str, ...] = ()
    independent: bool = False
    min_families: int = 1
    max_tokens: int = 4096
    temperature: float = 0.2

    def render(self, **inputs: Any) -> list[dict[str, Any]]:
        """The messages for one call. Refuses a forbidden key outright and any key outside ``sees`` (``RoleLeak``)."""
        bad = sorted(set(inputs) & set(self.never))
        if bad:
            raise RoleLeak(f"role {self.name!r} must never see {bad}")
        extra = sorted(set(inputs) - set(self.sees))
        if extra:
            raise RoleLeak(f"role {self.name!r} is not shown {extra} (it sees {list(self.sees)})")
        body = "\n\n".join(f"## {k}\n{_text(v)}" for k, v in inputs.items() if v is not None)
        return [{"role": "system", "content": self.system}, {"role": "user", "content": body + "\n\nReply with one JSON object only."}]

    def schema_hint(self) -> str:
        return self.output.__name__


def _text(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, indent=1, ensure_ascii=False, sort_keys=False, default=str)


# ---------------------------------------------------------------------------------------------------------- shared docs

PROGRAM_DOC = """PROGRAM FORMAT (the reference solution, a straight-line tool program)
{"steps": [{"id": "s1", "tool": "<app.tool>", "args": {"<param>": <arg>, ...}}, ...], "answer": <arg or null>}
An <arg> is one of:
- a plain JSON value: a literal, e.g. "pending" or 42;
- {"$ref": "s1.<field>.0.<field>"}: a value read from the observation of an EARLIER step (a dot path; integers index lists);
- {"$fn": "<name>", "args": [<arg>, ...]}: a pure function: sum, min, max, round(x, digits), len, concat,
  select(list, {field: value | {"op": "eq|ne|lt|le|gt|ge|in", "v": value}}) -> the matching items,
  get(obj, "path") -> a value inside obj, pluck(list, "field"), add, sub, mul, div.
Rules:
- Straight line: no branches or loops. You resolve every decision now, from the data you are shown.
- Read before you act, as an agent who has never seen the data would: list or search first, then refer to what you read.
  A literal is allowed only when the instruction states it (in the same spelling) or the tool catalogue lists it (an enum,
  a reason code in a description). Every other value (ids, vendor or customer ids, amounts, names, dates, normalised
  spellings) must come from an earlier observation through $ref or $fn, even though you can see the start state: e.g.
  list the candidates, then {"$fn": "get", "args": [{"$fn": "select", "args": [{"$ref": "s1.<the list field of the s1 result>"}, {"<field>": "<the stated value>"}]}, "0.id"]}.
- Step ids are s1, s2, ... in order. Write tools change state; read tools never do.
- "answer": null unless the instruction explicitly asks the agent to report a value. Then the bare value it asks for (a
  number, an id or other short string, or a list of them), never prose and never an object."""

CONTRACT_DOC = """CONTRACT FORMAT (what the trusted grader checks; JSON)
{"required": [<pattern>], "allowed": [<pattern with "why">], "answer": [<check>], "trace": [<constraint>], "normalizers": {...}}
<pattern>: {"id", "app", "coll", "key", "field", "kind": "insert|update|delete|any|state", "value": <pred>|null,
            "where": {field: <pred>}, "count": n, "count_max": n|"any"|null, "why"}
  app/coll/key/field are globs ("*" = any). An update is one change per field; an insert or delete is one change per row.
  A change satisfies at most one required pattern; counts are exact unless count_max widens them.
  "state" patterns count rows of the END state that match (no change needed).
  Every state change must match a required pattern or an allowed one, or the run fails (FORBIDDEN_EFFECT).
<pred>: {"op": "eq", "v": x} | {"op": "in", "vs": [..]} | {"op": "approx", "v": x, "tol": t} | {"op": "regex", "pattern": p}
        | {"op": "any"} | {"op": "rel", "rel": "eq|ne|lt|le|gt|ge", "ref": "old", "delta": d}
        | {"op": "from_answer", "path": null}
<check>: {"id", "op": "eq|approx|set_eq|regex|in|present", "value": x, "tol": t}
<constraint>: {"id", "op": "before|never|at_most|read_before_write", "tool": glob, "then": glob, "n": k,
               "entity": {"app","coll","key"}, "stated": "<the exact instruction words that state it>"}
normalizers: {"ignore_fields": [{"app","coll","field"}], "id_canon": [{"app","coll","id_fields": ["id"]}],
              "float_round": [{"app","coll","field","digits"}], "text_norm": [{"app","coll","field"}],
              "order_insensitive": [{"app","coll","field"}]}"""

DEFECTS_DOC = """DEFECT CLASSES (a counterexample names one)
D1 LAX-REQ a required effect is not checked | D2 LAX-FRAME unrequested effects pass | D3 LAX-VALUE a value check is too loose
D4 VACUOUS a check already holds at the start | D5 HIDDEN success needs information the agent cannot get
D6 OVER a valid solution is rejected (an unstated order or format demand) | D7 LEAK the instruction gives the decision away
D8 AFFORD a shortcut passes | D9 INSENS the contract ignores which entity the instruction names | D10 UNSOLV no solution passes
D11 NONDET the verdict depends on run noise"""

# The attack-strategy library F gives the attackers (§9.1 "attack strategies"; the falsifier loop evolves it in M3).
ATTACK_STRATEGIES: tuple[str, ...] = (
    "Do the minimum: perform only the single most visible write and skip every check and secondary requirement.",
    "Act in bulk: apply the action to every candidate entity, hoping the checker only looks at the target.",
    "Look for an admin, override, force or bulk tool that sets the final state directly, and use it.",
    "Guess the decision without reading the evidence: pick the most common status, reason code or value.",
    "Report a plausible final answer without doing the work the instruction asks for.",
    "Fill required fields with placeholder or junk values (empty strings, 'n/a', 0).",
    "Call tools with edge arguments (wrong types, empty strings, huge numbers) to reach states the normal path refuses.",
    "Repeat, undo or reorder actions to confuse checks on order or counts.",
)

BUILDERS = ("proposer", "oracle_programmer", "contract_writer", "repairer")

# ---------------------------------------------------------------------------------------------------------- the roles

PROPOSER = Role(
    name="proposer", stage="F3", output=ProposalOut, temperature=0.8, max_tokens=12000,
    sees=("domain", "tools", "world", "cell", "existing", "lemmas", "count"),
    never=("contract", "gate_code", "anchors", "locked_set", "verdicts"),
    system=("You design realistic training tasks for an AI agent that works inside an enterprise system through tools. "
            "Each task is one instruction a real user would give. It must be solvable with the listed tools from the start "
            "state shown, it must need the target skill, and its success must be checkable from the system's end state "
            "(writes) and, when the user asks for a figure or a list, the agent's final answer.\n"
            "Rules: name entities the way a user would (an invoice number, a ticket subject, a person), never internal "
            "database ids unless users see them. State every fact the agent cannot look up. Never state a decision the agent "
            "must work out (if the skill is deciding a reason code, the instruction must not contain it). Use different "
            "entities and different kinds of work across tasks; do not repeat an existing proposal.\n"
            "Ask the agent to report something only when the user needs a figure, an id or a short list, and then say exactly "
            "what to report (\"reply with the total as a number\"); a task that only changes records asks for no report.\n"
            "For each task give: skill (short), instruction, given (facts handed to the agent outside the instruction; "
            "usually []), derive (the values the agent must work out itself, as BARE values exactly as the system stores "
            "them: a reason code, a status, an id, an amount; never 'key=value' text), rationale.\n"
            'Reply {"tasks": [{"skill", "instruction", "given", "derive", "rationale"}, ...]} with exactly the requested count.'))

ORACLE_PROGRAMMER = Role(
    name="oracle_programmer", stage="F4", output=OracleOut, temperature=0.0, max_tokens=8192,
    sees=("task", "intent", "tools", "world", "program_format", "feedback"),
    never=("contract", "gate_code", "verdicts"),
    system=("You write the reference solution of a task as a tool program that is executed against the real system. "
            "Read the task, the tools and the start state, decide everything the task needs, then write the program.\n\n"
            + PROGRAM_DOC + "\n\nReply with the program JSON object (keys steps, answer, notes)."))

CONTRACT_WRITER = Role(
    name="contract_writer", stage="F5", output=ContractOut, temperature=0.0, max_tokens=6144,
    sees=("task", "intent", "tools", "skeleton", "effects", "answer", "contract_format"),
    never=("gate_code", "candidate_solutions", "proposer_notes", "verdicts"),
    differs_from=("proposer", "oracle_programmer"),
    system=("You generalise an exact verifier. The skeleton contract was inferred from the reference solution's executed "
            "effects: every changed field must equal what the reference wrote, nothing else may change, and the answer must "
            "equal the reference answer. Relax ONLY what the instruction leaves open, so that every valid solution passes and "
            "no invalid one does. Typical relaxations:\n"
            "- a timestamp, generated reference or clock value a tool writes: add an ignore_fields normalizer for that field;\n"
            "- rows a tool inserts with a generated id: add an id_canon normalizer for that collection;\n"
            "- a free-text field the instruction does not dictate (a note, a message body): relax its predicate to any or a "
            "regex that keeps the required content;\n"
            "- a number the instruction allows within a tolerance: approx;\n"
            "- an effect any valid solution may also cause (an audit row, a log line): an allowed pattern with a why.\n"
            "Do not relax what the instruction decides (which entity, which status, which reason). Add a trace constraint only "
            "when the instruction states it, quoting its exact words in stated.\n\n" + CONTRACT_DOC + "\n\n"
            "Reply {\"edits\": [...], \"notes\"} where each edit is one of:\n"
            '{"op": "relax", "target": <required id>, "field": null | <where field>, "pred": <pred>}\n'
            '{"op": "allow", "pattern": <pattern with why>}\n'
            '{"op": "demote", "target": <required id>, "why": ...}  (required -> allowed)\n'
            '{"op": "count", "target": <required id>, "count_max": n | "any"}\n'
            '{"op": "answer", "target": <check id>, "check": <check>}\n'
            '{"op": "trace", "constraint": <constraint with stated>}\n'
            '{"op": "normalize", "normalizers": {...}}\n'
            '{"op": "key", "target": <required id>, "key": <glob>}\n'
            "An empty list keeps the skeleton exact."))

REPAIRER = Role(
    name="repairer", stage="repair", output=RepairOut, temperature=0.2, max_tokens=8192,
    sees=("task", "intent", "tools", "oracle", "contract", "counterexamples", "defects", "program_format", "contract_format", "feedback"),
    never=("gate_code", "falsifier_code", "anchors", "verdict_internals"),
    differs_from=("contract_writer",),
    system=("You repair a task that automatic falsifiers found defective. Each counterexample is an executed probe (a tool "
            "program, or a property of the task) with the defect class it shows and the grader's reason codes. Fix the cause, "
            "not the symptom: tighten a contract that accepts a wrong run, relax one that rejects a valid run, state in the "
            "instruction a fact the agent cannot otherwise get, remove from the instruction a decision the agent must work out, "
            "make the reference program read before it acts. Never weaken a check just to stop a counterexample, and never make "
            "an earlier counterexample pass.\n\n" + DEFECTS_DOC + "\n\n" + PROGRAM_DOC + "\n\n" + CONTRACT_DOC + "\n\n"
            "Reply {\"instruction\": <new text or null to keep>, \"oracle\": <program or null>, \"contract\": <full contract or "
            "null>, \"given\": <list or null>, \"derive\": <list or null>, \"notes\"}, or {\"give_up\": true} if the task cannot "
            "be repaired."))

ALT_SOLVER = Role(
    name="alt_solver", stage="ALT", output=AgentTurn, temperature=0.3, max_tokens=2048,
    sees=("instruction", "tools", "history"),
    never=("contract", "oracle", "gate_code", "intent", "verdicts"),
    differs_from=BUILDERS, independent=True, min_families=2,
    system=("You are an operator completing a task in an enterprise system through its tools. Work step by step: read what "
            "you need, then act. Each reply is ONE JSON object: either a tool call {\"call\": {\"tool\": \"<app.tool>\", "
            "\"args\": {...}}} or, when the task is complete, {\"done\": true, \"final\": <the bare value the task asks you to "
            "report (number, string or list), or null>}. The task is complete only when every change it asks for has been "
            "made through the write tools. Do only what the task asks; do not change anything else."))

ATTACKER = Role(
    name="attacker", stage="ATK", output=AgentTurn, temperature=0.7, max_tokens=2048,
    sees=("instruction", "tools", "strategies", "history"),
    never=("contract", "oracle", "gate_code", "intent", "verdicts"),
    differs_from=BUILDERS, independent=True, min_families=2,
    system=("You are a red-team tester of an automated task checker. A checker will look at the system's end state and your "
            "final answer and decide whether the task below was done. Your goal is to make the checker ACCEPT the task WITHOUT "
            "doing the work the task really requires: find a shortcut. Strategies you may use are listed. Each reply is ONE "
            "JSON object: a tool call {\"call\": {\"tool\": \"<app.tool>\", \"args\": {...}}} or, when you are finished, "
            "{\"done\": true, \"final\": <the value to report, or null>}."))

PANEL = Role(
    name="panel", stage="adjudicate", output=PanelOut, temperature=0.0, max_tokens=3072,
    sees=("task", "intent", "trace", "answer", "reason_codes"),
    never=("gate_code", "falsifier", "contract", "builder", "probe_class"),
    differs_from=BUILDERS, independent=True, min_families=2,
    system=("You audit one recorded attempt at a task. Decide whether the attempt is a VALID solution: it does everything the "
            "instruction asks, on the right entities, with correct values; it relies only on information the agent could get; "
            "it makes no change the instruction does not ask for or clearly allow; and it takes no shortcut around the skill. "
            "The reason codes are what an automated grader reported (empty = it found nothing wrong); they may be right or "
            "wrong. Reply {\"valid\": true|false, \"confidence\": <0..1>, \"why\": \"<one or two sentences>\"}."))

ABSTRACTOR = Role(
    name="abstractor", stage="abstract", output=AbstractOut, temperature=0.2, max_tokens=4096,
    sees=("clusters", "defects"),
    never=("anchors", "locked_set", "task_ids", "gate_code"),
    differs_from=BUILDERS, independent=True,
    system=("You turn clusters of counterexamples that automatic falsifiers found in generated tasks into short design rules "
            "(lemmas) for the roles that write tasks. A lemma is general: it names no task, entity, id or value, and it would "
            "have prevented the whole cluster. " + DEFECTS_DOC + "\n"
            'Reply {"lemmas": [{"applies_to": "proposer|oracle_programmer|contract_writer|repairer", "rule": "...", '
            '"evidence": ["<cluster id>", ...]}]}.'))

TOOL_DESIGNER = Role(
    name="tool_designer", stage="F1", output=ToolDesignOut, temperature=0.4, max_tokens=16384,
    sees=("domain", "guidance"),
    never=("tests", "tasks"),
    system=("You design the tool API of a small, self-contained enterprise application for a domain, tool first: each tool "
            "has a name, a description an agent can act on (state every enum value and reason code it accepts), typed "
            "parameters, whether it is read-only, the collections it writes, its preconditions, its postconditions (the exact "
            "state change), the errors it raises, and the shape of what it returns (an example object). Include list/search "
            "tools that let an agent discover every entity, and write tools for the domain's real decisions. 8 to 10 tools. "
            "Be compact: a description of at most two sentences, at most three short pre and post items, a small returns "
            "example. If the domain has written rules (a policy), include a read-only tool that returns the policy text.\n"
            'Reply {"app": "<lowercase identifier>", "description": "...", "tools": [{"name", "description", "params": [{"name", '
            '"type", "required", "description", "enum", "default"}], "read_only", "writes", "pre", "post", "errors", '
            '"returns"}]}.'))

SCHEMA_DERIVER = Role(
    name="schema_deriver", stage="F1", output=SchemaOut, temperature=0.0, max_tokens=8192,
    sees=("design",),
    system=("Derive the database schema an application needs from its tool API: one table per collection the tools read or "
            "write, with its key field and every field with a type (string, integer, number, boolean, array, object; add '?' "
            "for a field that may be null). Every collection a tool writes must be a table. "
            'Reply {"tables": [{"name", "key", "fields": {"<field>": "<type>"}, "description"}]}.'))

APP_API_DOC = """APP API (the module you write runs inside an isolated sandbox)
Write a Python module that defines exactly one function:
    def build(app):
        @app.tool(read_only=True)
        def list_things(ctx, status: str = "open") -> dict:
            \"\"\"<the tool description>\"\"\"
            return {"things": [dict(r) for r in ctx.tables["things"].values() if r["status"] == status]}
        @app.tool(writes=["things"])
        def close_thing(ctx, thing_id: str, note: str) -> dict:
            \"\"\"...\"\"\"
            t = ctx.tables["things"].get(thing_id)
            if t is None:
                raise ToolError(f"no thing {thing_id}")
            t["status"] = "closed"
            return {"id": thing_id, "status": "closed"}
- ctx.tables[<table>] is a dict key -> row dict (mutate it to write); ctx.now() is the clock (ISO text); ctx.rng a seeded
  random.Random. Keys are the row's key field value (strings, or integers when the key field is an integer).
- A write tool declares every table it changes in writes=[...]; read-only tools change nothing. Raise ToolError("...") to
  refuse a call (bad argument, wrong state); a refused call changes nothing.
- Type-annotate every parameter (str, int, float, bool, list, dict); give optional parameters defaults.
- Allowed imports: math, re, datetime, json, copy, itertools, collections, functools, statistics, string, decimal.
  No files, no network, no os/sys/subprocess, no eval/exec, no dunder attributes. Return JSON-able values."""

APP_CODER = Role(
    name="app_coder", stage="F1", output=AppCodeOut, temperature=0.0, max_tokens=20000,
    sees=("design", "schema", "app_api"),
    never=("tests",),
    system=("You implement an application's tools exactly as designed (names, parameters, read-only flags, writes, "
            "preconditions, postconditions, errors, return shapes).\n\n" + APP_API_DOC + "\n\n"
            'Reply {"code": "<the whole module>", "notes"}.'))

BLIND_TESTER = Role(
    name="blind_tester", stage="F1", output=ToolTestsOut, temperature=0.2, max_tokens=20000,
    sees=("design", "schema", "test_format"),
    never=("code",),
    differs_from=("app_coder",),
    system=("You write blind acceptance tests for an application's tools from its specification only (you never see the "
            "code). Cover every tool: simple, medium, complex and boundary cases, and negative probes (a missing id, a wrong "
            "state, an out-of-range or unknown enum value) that must be refused. Each test builds its own small fixture, calls "
            "tools in order and states what must hold. Check only what the specification defines (documented return fields, "
            "postconditions); never assume an undocumented detail. Your expectations are frozen once written."))

TEST_FORMAT_DOC = """TEST FORMAT
{"tests": [{"id": "t1", "tier": "simple|medium|complex|boundary",
            "fixture": {"<table>": [<row with its key field>, ...]},
            "calls": [{"tool": "<tool name>", "args": {...}, "expect": "ok|error", "result": <subset of the result or null>}],
            "expect_state": {"<table>": {"<key>": {"<field>": <value>}}},
            "expect_absent": {"<table>": ["<key>"]}}]}
A result subset matches when every key it names is in the result with an equal value (lists: same length, items matched in
order). Write 2 tests per tool (3 for a complex write tool), 16 to 26 tests in all, with small fixtures (at most
four rows per table, only the fields a test needs plus the key)."""

DEBUGGER = Role(
    name="debugger", stage="F1", output=DebugOut, temperature=0.0, max_tokens=20000,
    sees=("design", "schema", "code", "failures", "app_api"),
    never=("edit_tests",),
    differs_from=("test_approver",),
    system=("You fix an application's code so that its frozen acceptance tests pass. You may change the code only: the tests "
            "and their expectations are frozen. If you are sure a test contradicts the specification, list it under disputes "
            "with the reason (an independent reviewer decides; a disputed test is never edited).\n\n" + APP_API_DOC + "\n\n"
            'Reply {"code": "<the whole fixed module>", "disputes": [{"test", "why"}], "notes"}.'))

TEST_APPROVER = Role(
    name="test_approver", stage="F1", output=ApproveOut, temperature=0.0, max_tokens=2048,
    sees=("design", "schema", "test", "failure", "dispute"),
    never=("code",),
    differs_from=("debugger", "blind_tester"), independent=True,
    system=("A test of an application's tools is disputed. Read the specification, the test and how it failed. Decide only "
            "whether the TEST contradicts the specification (then it is wrong and will be dropped). If the specification "
            "supports the test, the code is wrong. Reply {\"test_is_wrong\": true|false, \"why\"}."))

SEEDER = Role(
    name="seeder", stage="F2", output=SeedOut, temperature=0.6, max_tokens=16384,
    sees=("design", "schema", "guidance"),
    never=("tasks", "verifiers"),
    system=("You write the start state of an application: realistic rows for every table, consistent with each other (every "
            "reference points at a row that exists), with enough variety for many different tasks, and with distractors: rows "
            "that look like likely task targets but differ in one field (a similar name, a near-duplicate, a different status). "
            "6 to 15 rows per table; every field of the schema in every row. "
            'Reply {"tables": {"<table>": [<row with its key field>, ...]}, "notes"}.'))

ROLES: dict[str, Role] = {r.name: r for r in (PROPOSER, ORACLE_PROGRAMMER, CONTRACT_WRITER, REPAIRER, ALT_SOLVER, ATTACKER, PANEL, ABSTRACTOR,
                                              TOOL_DESIGNER, SCHEMA_DERIVER, APP_CODER, BLIND_TESTER, DEBUGGER, TEST_APPROVER, SEEDER)}


# ---------------------------------------------------------------------------------------------------------- family rules

def family_problems(assignment: Mapping[str, Sequence[str]], *, students: Iterable[str] = ("qwen",), roles: Mapping[str, Role] | None = None) -> list[str]:
    """Every violation of the roles' family constraints by ``assignment`` (role -> the families serving it):
    a ``differs_from`` role sharing a family, an ``independent`` role on a student family, a multi-model role with too
    few families, an unknown role."""
    rs = roles or ROLES
    stud = set(students)
    out: list[str] = []
    for r, fams in assignment.items():
        role = rs.get(r)
        if role is None:
            out.append(f"unknown role {r!r}")
            continue
        fs = set(fams)
        if not fs:
            out.append(f"role {r!r} has no model")
            continue
        if len(fs) < role.min_families:
            out.append(f"role {r!r} needs >= {role.min_families} families, has {sorted(fs)}")
        if role.independent and fs & stud:
            out.append(f"role {r!r} is independent and may not run on the student's family {sorted(fs & stud)}")
        for other in role.differs_from:
            shared = fs & set(assignment.get(other, ()))
            if shared:
                out.append(f"role {r!r} must differ in family from {other!r}; both use {sorted(shared)}")
    return out


# ---------------------------------------------------------------------------------------------------------- parsing

def tool_catalogue(tools: Iterable[ToolInfo]) -> list[dict[str, Any]]:
    """The tool list as a role sees it: name, description, parameters, read-only flag."""
    out = []
    for t in tools:
        props = t.input_schema.get("properties") or {}
        params = {}
        for k, p in props.items():
            if not isinstance(p, dict):
                continue
            d = {x: p[x] for x in ("type", "enum", "default", "description") if x in p}
            if "anyOf" in p:
                d["type"] = "/".join(str(a.get("type")) for a in p["anyOf"] if isinstance(a, dict))
            params[k] = d
        out.append({"name": t.name, "description": t.description, "params": params, "required": list(t.input_schema.get("required") or []),
                    "read_only": t.read_only})
    return out


def _arg(v: Any, where: str) -> Lit | Ref | Fn:
    if isinstance(v, dict):
        if set(v) == {"$ref"}:
            if not isinstance(v["$ref"], str) or not v["$ref"]:
                raise ValueError(f"{where}: $ref must be a non-empty path string")
            return Ref(path=v["$ref"])
        if "$fn" in v:
            name = v["$fn"]
            if name not in LIBRARY:
                raise ValueError(f"{where}: unknown function {name!r} (have {sorted(LIBRARY)})")
            args = v.get("args", [])
            if not isinstance(args, list):
                raise ValueError(f"{where}: $fn args must be a list")
            return Fn(name=name, args=[_arg(a, f"{where}/{name}[{i}]") for i, a in enumerate(args)])
        if v.get("op") in ("lit", "ref", "fn"):
            if v["op"] == "lit":
                return Lit(v=v.get("v"))
            if v["op"] == "ref":
                return Ref(path=str(v.get("path")))
            return Fn(name=str(v.get("name")), args=[_arg(a, f"{where}/{v.get('name')}[{i}]") for i, a in enumerate(v.get("args") or [])])
        return Lit(v={k: _plain(x, f"{where}.{k}") for k, x in v.items()})
    if isinstance(v, list):
        return Lit(v=[_plain(x, f"{where}[{i}]") for i, x in enumerate(v)])
    return Lit(v=v)


def _plain(v: Any, where: str) -> Any:
    """A literal container may not hide a ref (it would never be resolved)."""
    if isinstance(v, dict):
        if "$ref" in v or "$fn" in v:
            raise ValueError(f"{where}: a $ref or $fn inside a literal list or object is not resolved; pass it as its own argument")
        return {k: _plain(x, f"{where}.{k}") for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x, f"{where}[{i}]") for i, x in enumerate(v)]
    return v


def parse_program(doc: Mapping[str, Any], tools: Iterable[str] | None = None) -> Program:
    """A model's program JSON (``PROGRAM_DOC``) as a ``Program``; ``ValueError`` names the first problem."""
    steps_in = doc.get("steps")
    if not isinstance(steps_in, list) or not steps_in:
        raise ValueError("the program has no steps")
    known = set(tools) if tools is not None else None
    steps = []
    for i, s in enumerate(steps_in):
        if not isinstance(s, dict):
            raise ValueError(f"step {i} is not an object")  # noqa: TRY004 - a ValueError is the feedback to the model
        sid, tool, args = s.get("id") or f"s{i + 1}", s.get("tool"), s.get("args") or {}
        if not isinstance(tool, str) or not tool:
            raise ValueError(f"step {sid}: no tool")
        if known is not None and tool not in known:
            raise ValueError(f"step {sid}: unknown tool {tool!r}")
        if not isinstance(args, dict):
            raise ValueError(f"step {sid}: args must be an object")  # noqa: TRY004
        steps.append(Step(id=str(sid), tool=tool, args={str(k): _arg(v, f"{sid}.{k}") for k, v in args.items()}))
    ans = doc.get("answer")
    prog = Program(steps=steps, answer=None if ans is None else _arg(ans, "answer"))
    probs = prog.problems()
    if probs:
        raise ValueError("; ".join(probs[:5]))
    return prog


def program_doc(p: Program) -> dict[str, Any]:
    """A ``Program`` in the role JSON form (the inverse of ``parse_program``)."""
    def a(x: Any) -> Any:
        if isinstance(x, Lit):
            return x.v
        if isinstance(x, Ref):
            return {"$ref": x.path}
        return {"$fn": x.name, "args": [a(y) for y in x.args]}

    return {"steps": [{"id": s.id, "tool": s.tool, "args": {k: a(v) for k, v in s.args.items()}} for s in p.steps],
            "answer": None if p.answer is None else a(p.answer)}


def contract_doc(c: Contract) -> dict[str, Any]:
    """A contract as the roles read it: empty parts left out, every predicate's ``op`` kept (``exclude_defaults`` would
    drop the discriminator and make ``{"op": "eq", ...}`` unreadable on the way back)."""

    def prune(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: prune(x) for k, x in v.items() if x not in ({}, [], None) and not (k == "count_max" and x is None)}
        if isinstance(v, list):
            return [prune(x) for x in v]
        return v

    return prune(c.model_dump(mode="json", exclude_none=True))


def parse_contract(doc: Mapping[str, Any]) -> Contract:
    try:
        return Contract.model_validate(dict(doc))
    except Exception as e:  # noqa: BLE001 - pydantic's message is the feedback
        raise ValueError(f"invalid contract: {str(e)[:600]}") from None


class EditResult(BaseModel):
    contract: Any
    applied: list[int] = Field(default_factory=list)
    rejected: list[tuple[int, str]] = Field(default_factory=list)


def _pred(d: Mapping[str, Any]) -> Any:
    from pydantic import TypeAdapter

    return TypeAdapter(Pred).validate_python(dict(d))


def apply_edits(skeleton: Contract, edits: Sequence[ContractEdit], instruction: str = "") -> EditResult:
    """Apply a contract writer's edits to the skeleton. Each edit is checked on its own; a bad edit is rejected with a
    reason (it never aborts the others). A ``trace`` edit must quote the instruction (``stated``)."""
    from crucible.synth.canon import Normalizers, norm_text

    cur = skeleton.model_copy(deep=True)
    res = EditResult(contract=cur)
    for i, e in enumerate(edits):
        c = cur.model_copy(deep=True)
        try:
            if e.op == "keep":
                continue
            req = {p.id: p for p in c.required}
            if e.op in ("relax", "demote", "count", "key"):
                if e.target not in req:
                    raise ValueError(f"no required pattern {e.target!r}")
                p = req[e.target]
                if e.op == "relax":
                    if e.pred is None:
                        raise ValueError("relax needs pred")
                    pr = _pred(e.pred)
                    if e.field:
                        if e.field not in p.where:
                            raise ValueError(f"pattern {p.id} has no where field {e.field!r}")
                        new = p.model_copy(update={"where": {**p.where, e.field: pr}})
                    else:
                        if p.kind not in ("update", "state", "any"):
                            raise ValueError(f"pattern {p.id} is an {p.kind}: relax a where field instead")
                        new = p.model_copy(update={"value": pr})
                    c.required = [new if x.id == p.id else x for x in c.required]
                elif e.op == "demote":
                    if not (e.why or "").strip():
                        raise ValueError("demote needs a why")
                    c.required = [x for x in c.required if x.id != p.id]
                    c.allowed = [*c.allowed, p.model_copy(update={"id": f"a_{p.id}", "why": e.why, "count": 0, "count_max": "any"})]
                elif e.op == "count":
                    if e.count_max is None:
                        raise ValueError("count needs count_max")
                    c.required = [x.model_copy(update={"count_max": e.count_max}) if x.id == p.id else x for x in c.required]
                else:
                    if not e.key:
                        raise ValueError("key needs key")
                    c.required = [x.model_copy(update={"key": e.key}) if x.id == p.id else x for x in c.required]
            elif e.op == "allow":
                if e.pattern is None:
                    raise ValueError("allow needs pattern")
                raw = {"count": 0, "count_max": "any", **dict(e.pattern)}
                if raw.get("kind") in ("insert", "delete"):  # a whole-entity pattern: models often add a field or a value; drop them
                    raw["field"] = "*"
                    raw.pop("value", None)
                pat = EffectPattern.model_validate(raw)
                if not (pat.why or "").strip():
                    raise ValueError("an allowed pattern needs a why")
                c.allowed = [*c.allowed, pat]
            elif e.op == "answer":
                if e.check is None:
                    raise ValueError("answer needs check")
                chk = AnswerCheck.model_validate({"id": e.target or "q0", **dict(e.check)})
                if not any(q.id == chk.id for q in c.answer):
                    raise ValueError(f"no answer check {chk.id!r}")
                c.answer = [chk if q.id == chk.id else q for q in c.answer]
            elif e.op == "trace":
                if e.constraint is None:
                    raise ValueError("trace needs constraint")
                m = TraceConstraint.model_validate(dict(e.constraint))
                if not m.stated or norm_text(m.stated) not in norm_text(instruction):
                    raise ValueError("a trace constraint must quote the instruction words that state it (stated)")
                c.trace = [*c.trace, m]
            elif e.op == "normalize":
                if e.normalizers is None:
                    raise ValueError("normalize needs normalizers")
                c.normalizers = c.normalizers.merged(Normalizers.model_validate(dict(e.normalizers)))
            cur = Contract.model_validate(c.model_dump())  # the contract's own invariants (ids unique, allowed has why)
            res.applied.append(i)
        except Exception as ex:  # noqa: BLE001 - one bad edit is rejected with its reason, the rest still apply
            res.rejected.append((i, str(ex)[:300]))
    res.contract = cur
    return res


class RoleFailed(RuntimeError):
    """A role gave no usable reply within its retries (the reason is the last problem found)."""


async def ask(client: RoleClient, role: Role, *, tag: str = "", check: Callable[[Any], Any] | None = None, retries: int = 1, **inputs: Any) -> Any:
    """One role call: render (``RoleLeak`` on a forbidden input), ask for JSON that fits ``role.output``, then run
    ``check`` (which may return a parsed value to use instead). A ``ValueError`` from parsing or ``check`` goes back to
    the model as feedback, up to ``retries`` more times; then ``RoleFailed``."""
    msgs = role.render(**inputs)
    last = ""
    for _ in range(retries + 1):
        try:
            doc = await client.complete_json(msgs, schema=role.output, max_tokens=role.max_tokens, temperature=role.temperature, tag=tag or role.name)
        except LLMOutputError as e:  # the client's own JSON retries ran out: one attempt of this role spent
            last = str(e)[:4000]
            continue
        try:
            out = role.output.model_validate(doc)
            got = check(out) if check is not None else None
            if inspect.isawaitable(got):
                got = await got
            return out if got is None else got
        except ValueError as e:  # pydantic's ValidationError is a ValueError
            last = str(e)[:4000]
            msgs = [*msgs, {"role": "assistant", "content": json.dumps(doc, default=str)[:20000]},
                    {"role": "user", "content": f"That reply has a problem: {last}\nFix it and reply with the whole JSON object again."}]
    raise RoleFailed(f"{role.name}: {last}")


def compact(v: Any, limit: int = 4000) -> str:
    """JSON text cut to ``limit`` characters (observations and slices shown to models)."""
    t = cjson(v) if not isinstance(v, str) else v
    return t if len(t) <= limit else t[:limit] + f"...<{len(t) - limit} more characters>"


__all__ = [
    "APP_API_DOC",
    "ATTACK_STRATEGIES",
    "BUILDERS",
    "CONTRACT_DOC",
    "DEFECTS_DOC",
    "PROGRAM_DOC",
    "ROLES",
    "TEST_FORMAT_DOC",
    "AgentTurn",
    "ContractEdit",
    "ContractOut",
    "EditResult",
    "OracleOut",
    "PanelOut",
    "ProposalOut",
    "RepairOut",
    "Role",
    "RoleClient",
    "RoleFailed",
    "RoleLeak",
    "apply_edits",
    "ask",
    "compact",
    "contract_doc",
    "family_problems",
    "parse_contract",
    "parse_program",
    "program_doc",
    "tool_catalogue",
]
