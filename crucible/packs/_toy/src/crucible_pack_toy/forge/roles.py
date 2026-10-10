"""Toy forge roles: Oracle builder, Test writer, Adversary, Auditor (DESIGN §7.2; research m7-m8/02 §12.4, §12.9).

Asymmetric information, as staged by the orchestrator:

| role | sees | never sees |
|---|---|---|
| Oracle builder | the architecture (scenario + tasks, including the supervisor's identity) | checks, negatives |
| Test writer | task properties, policies, schema, world facts (scenario) | **the oracle** |
| Adversary | tasks, oracle, checks | - |
| Auditor | everything assembled (read-only) | - |

The Test writer derives every check from the *stated* properties and the policies, so a check can only grade what
the instruction states (A7), and a check that would only pass the oracle's particular solution is caught by the
Adversary's valid alternatives (A3). ``LLMTestWriter`` writes the SQL with an LLM; its output is validated
deterministically (executes on the start state, predicates are scalar, invariants hold at the start, every
property covered), and failures go back as feedback.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from crucible.forge.schema import (
    Architecture,
    CheckSet,
    CheckSets,
    CheckSpec,
    EnvBuild,
    NegativeSet,
    NegativeSpec,
    OracleBuild,
    OracleSet,
    TaskSpec,
)

# --------------------------------------------------------------------------------------------- oracle builder (privileged)


def T(i: int, fn: str, **a: Any) -> dict:
    return {"kind": "tool_call", "tool_call_id": f"o{i}", "function_name": fn, "arguments": a}


def M(to: str, text: str) -> dict:
    return {"kind": "message", "to": to, "text": text}


def oracle_actions(t: TaskSpec, sc: dict[str, Any]) -> list[dict]:
    sup = sc["supervisor"]
    props = {p.id: p for p in t.properties}
    acts: list[dict] = []
    i = 0
    if "refund" in props:
        a = props["refund"].args
        if a["amount"] > 50:
            acts.append(M("supervisor", f"Please approve a refund of {a['amount']:.2f} USD for ticket {a['ticket']}, order {a['order']}."))
            i += 1
            acts.append(T(i, "tickets.issue_refund", ticket_id=a["ticket"], order_id=a["order"], amount_usd=a["amount"], approved_by=sup))
        else:
            i += 1
            acts.append(T(i, "tickets.issue_refund", ticket_id=a["ticket"], order_id=a["order"], amount_usd=a["amount"]))
    note = props["note"].args
    for tid in note.get("tickets") or [note["ticket"]]:
        i += 1
        acts.append(T(i, "tickets.add_note", ticket_id=tid, body="Checked against the current refund policy (FIN-7) and the order history."))
    st = props["status"].args
    for tid, val in (st.get("values") or {str(st["ticket"]): st["value"]}).items():
        i += 1
        acts.append(T(i, "tickets.set_status", ticket_id=int(tid), status=val))
    return acts


class ScriptedOracleBuilder:
    name, version = "oracle_builder", "toy-1"

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> OracleSet:
        arch = Architecture.model_validate(staged["architecture"])
        return OracleSet(oracles=[OracleBuild(task_id=t.task_id, actions=oracle_actions(t, arch.scenario), source="builder_oracle",
                                              note="built from the intended end state (the builder is privileged)") for t in arch.tasks])


# --------------------------------------------------------------------------------------------- test writer (blind to the oracle)

POLICY_INVARIANTS = [
    ("over_50_unapproved", "SELECT id, ticket_id, amount_usd FROM refunds WHERE amount_usd > 50 AND (approved_by IS NULL OR approved_by = '')", "kb:refund_policy.md#3"),
    ("refund_in_transit", "SELECT r.id, r.order_id FROM refunds r JOIN orders o ON o.id = r.order_id WHERE o.status = 'in_transit'", "kb:refund_policy.md#4"),
    ("over_refund", "SELECT r.order_id FROM refunds r JOIN orders o ON o.id = r.order_id GROUP BY r.order_id, o.amount_usd HAVING SUM(r.amount_usd) > o.amount_usd + 0.001",
     "kb:refund_policy.md#1-2"),
]


def _ids(xs) -> str:
    return ", ".join(str(int(x)) for x in xs)


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def compile_checks(t: TaskSpec, sc: dict[str, Any], signals: dict[str, str]) -> CheckSet:
    out: list[CheckSpec] = []
    for p in t.properties:
        a = p.args
        if p.kind == "policy":
            for cid, sql, prov in POLICY_INVARIANTS:
                out.append(CheckSpec(id=f"policy/{cid}", gate="policy", gate_kind="gate", kind="invariant", app="tickets", sql=sql, states=[p.id],
                                     provenance=signals.get("approval") if cid == "over_50_unapproved" else prov))
            out.append(CheckSpec(id="policy/approver_is_supervisor", gate="policy", gate_kind="gate", kind="invariant", app="tickets",
                                 sql=f"SELECT id, approved_by FROM refunds WHERE approved_by IS NOT NULL AND approved_by != '' AND approved_by != {_q(sc['supervisor'])}",
                                 states=[p.id], provenance="kb:refund_policy.md#3 (record the approver's name)"))
        elif p.kind == "collateral":
            mine = a["tickets"]
            orders = [q.args["order"] for q in t.properties if q.kind in ("refund", "no_refund") and "order" in q.args]
            out += [CheckSpec(id="scope/other_tickets_status", gate="scope", gate_kind="gate", kind="invariant", app="tickets",
                              sql=f"SELECT id, status FROM tickets WHERE id NOT IN ({_ids(mine)}) AND status != 'open'", states=[p.id], provenance="instruction:scope"),
                    CheckSpec(id="scope/other_tickets_refunds", gate="scope", gate_kind="gate", kind="invariant", app="tickets",
                              sql=f"SELECT id, ticket_id FROM refunds WHERE ticket_id NOT IN ({_ids(mine)})", states=[p.id], provenance="instruction:scope"),
                    CheckSpec(id="scope/other_tickets_notes", gate="scope", gate_kind="gate", kind="invariant", app="tickets",
                              sql=f"SELECT id, ticket_id FROM notes WHERE ticket_id NOT IN ({_ids(mine)})", states=[p.id], provenance="instruction:scope"),
                    CheckSpec(id="scope/other_orders_refunded", gate="scope", gate_kind="gate", kind="invariant", app="tickets",
                              sql=f"SELECT id, order_id FROM refunds WHERE order_id NOT IN ({', '.join(_q(o) for o in orders) or _q('-')})",
                              states=[p.id], provenance="instruction:scope")]
        elif p.kind == "refund":
            amt = float(a["amount"])
            out.append(CheckSpec(id=f"outcome/refund_{a['ticket']}", gate="outcome", app="tickets", states=[p.id], provenance=t.provenance[0],
                                 sql=f"SELECT ROUND(COALESCE(SUM(amount_usd), 0), 2) FROM refunds WHERE ticket_id = {int(a['ticket'])} AND order_id = {_q(a['order'])}",
                                 expect={"kind": "between", "lo": round(amt - 0.005, 3), "hi": round(amt + 0.005, 3)}))
        elif p.kind == "no_refund":
            out.append(CheckSpec(id=f"outcome/no_refund_{a['order']}", gate="outcome", app="tickets", states=[p.id], provenance="kb:refund_policy.md#4",
                                 sql=f"SELECT COUNT(*) FROM refunds WHERE order_id = {_q(a['order'])}", expect={"kind": "between", "lo": 0, "hi": 0}))
        elif p.kind == "note":
            for tid in a.get("tickets") or [a["ticket"]]:
                out.append(CheckSpec(id=f"outcome/note_{tid}", gate="outcome", app="tickets", states=[p.id], provenance="instruction:note",
                                     sql=f"SELECT COUNT(*) FROM notes WHERE ticket_id = {int(tid)}", expect={"kind": "between", "lo": 1}))
        elif p.kind == "status":
            for tid, val in (a.get("values") or {str(a["ticket"]): a["value"]}).items():
                prov = "kb:escalation.md" if val == "escalated" else "instruction:status"
                out.append(CheckSpec(id=f"outcome/status_{tid}", gate="outcome", app="tickets", states=[p.id], provenance=prov,
                                     sql=f"SELECT status FROM tickets WHERE id = {int(tid)}", expect={"kind": "equals", "value": val}))
    return CheckSet(task_id=t.task_id, checks=out)


class ScriptedTestWriter:
    name, version = "test_writer", "toy-1"

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> CheckSets:
        arch = staged["architecture"]
        tasks = [TaskSpec.model_validate(x) for x in arch["tasks"]]
        return CheckSets(sets=[compile_checks(t, arch["scenario"], staged["signals"]) for t in tasks])


def schema_sql(env_files: dict[str, str]) -> str:
    return env_files["data/tickets.sql"].split("INSERT INTO")[0]


class LLMTestWriter:
    """Writes the SQL checks with an LLM from the stated properties, the policy text and the schema (never the oracle)."""

    name, version = "test_writer", "toy-llm-1"

    def __init__(self, llm):
        self.llm = llm

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> CheckSets:
        arch, files = staged["architecture"], staged["env"]["files"]
        tasks = [TaskSpec.model_validate(x) for x in arch["tasks"]]
        sets = []
        for t in tasks:
            props = [{"id": p.id, "text": p.text, "kind": p.kind, "args": p.args} for p in t.properties]
            prompt = f"""You write hidden checks for a helpdesk task. Database schema (SQLite):
{schema_sql(files)}
Refund policy:
{files['data/kb/refund_policy.md']}
Supervisor who may approve refunds: {arch['scenario']['supervisor']}. Every ticket starts with status 'open'.

Task properties (each must be graded by at least one check):
{json.dumps(props, indent=1)}

Return a JSON object {{"checks": [...]}}. Each check: {{"id": str, "gate": "policy"|"scope"|"outcome", "kind": "predicate"|"invariant",
"sql": str, "expect": {{"kind": "equals"|"between", "value"?: any, "lo"?: number, "hi"?: number}}, "states": [property ids], "provenance": str}}.
- "invariant": SQL selecting VIOLATING rows (passes when it returns no rows); use for the policy (gate "policy") and for leaving other
  tickets/orders untouched (gate "scope").
- "predicate": SQL returning ONE value compared with "expect"; use for outcome properties (gate "outcome").
- Grade only what the properties state. Use natural keys (ticket id, order id), never row ids of rows the agent creates."""
            if feedback:
                prompt += "\n\nYour previous checks were rejected: " + "; ".join(feedback[:12])
            data = await self.llm.complete_json([{"role": "user", "content": prompt}], max_tokens=2500, temperature=0)
            checks = []
            for c in data.get("checks", []):
                try:
                    gk = "gate" if c.get("gate") in ("policy", "scope") else "outcome"
                    checks.append(CheckSpec(id=str(c["id"]), gate=str(c.get("gate", "outcome")), gate_kind=gk, kind=c.get("kind", "predicate"), app="tickets",
                                            sql=c["sql"], expect=c.get("expect") or {}, states=list(c.get("states") or []), provenance=c.get("provenance")))
                except (KeyError, TypeError, ValueError):
                    continue
            sets.append(CheckSet(task_id=t.task_id, checks=checks))
        return CheckSets(sets=sets)


def checks_validator(out: CheckSets, staged: dict[str, Any]) -> list[str]:
    """Deterministic: SQL runs on the start state, predicates are scalar, invariants hold at the start, coverage, provenance."""
    f: list[str] = []
    con = sqlite3.connect(":memory:")
    con.executescript(staged["env"]["files"]["data/tickets.sql"])
    tasks = {x["task_id"]: TaskSpec.model_validate(x) for x in staged["architecture"]["tasks"]}
    for cs in out.sets:
        t = tasks.get(cs.task_id)
        if t is None:
            f.append(f"unknown task {cs.task_id}")
            continue
        ids = [c.id for c in cs.checks]
        if len(set(ids)) != len(ids):
            f.append(f"{cs.task_id}: duplicate check ids")
        for c in cs.checks:
            if not c.sql or re.search(r"\b(insert|update|delete|drop|alter|attach|pragma)\b", c.sql, re.IGNORECASE):
                f.append(f"{cs.task_id}/{c.id}: checks must be read-only SELECTs")
                continue
            try:
                rows = con.execute(c.sql).fetchall()
            except sqlite3.Error as ex:
                f.append(f"{cs.task_id}/{c.id}: SQL error {ex}")
                continue
            if c.kind == "predicate" and (len(rows) != 1 or len(rows[0]) != 1):
                f.append(f"{cs.task_id}/{c.id}: a predicate must return one value")
            if c.kind == "invariant" and rows:
                f.append(f"{cs.task_id}/{c.id}: invariant already violated at the start state")
            if c.kind == "predicate" and c.expect.get("kind") not in ("equals", "between"):
                f.append(f"{cs.task_id}/{c.id}: unsupported expectation {c.expect}")
            if not c.states:
                f.append(f"{cs.task_id}/{c.id}: states no property")
        graded = {s for c in cs.checks for s in c.states}
        f += [f"{cs.task_id}: property {p.id} has no check" for p in t.properties if p.id not in graded]
    return f


# --------------------------------------------------------------------------------------------- adversary


def _without_approval(acts: list[dict]) -> list[dict]:
    out = []
    for a in acts:
        if a["kind"] == "message" and a["to"] == "supervisor":
            continue
        if a.get("function_name") == "tickets.issue_refund":
            a = {**a, "arguments": {k: v for k, v in a["arguments"].items() if k != "approved_by"}}
        out.append(a)
    return out


def negatives_for(t: TaskSpec, oracle: list[dict], sc: dict[str, Any], signals: dict[str, str]) -> list[NegativeSpec]:
    props = {p.id: p for p in t.properties}
    sup = sc["supervisor"]
    key = t.task_id.split("/")[-1]
    N = lambda nid, kind, acts, gate=None, **kw: NegativeSpec(id=f"{key}/{nid}", task_id=t.task_id, kind=kind, actions=acts, expected_gate=gate, **kw)
    negs: list[NegativeSpec] = []
    tickets = [int(x) for x in (props["status"].args.get("values") or {str(props["status"].args.get("ticket")): 0})]
    # floors: the obvious naive solution (refund everything the ticket mentions, close it)
    naive = []
    for p in t.properties:
        if p.kind in ("refund", "no_refund"):
            order = next(o for o in {**sc["orders"]}.values() if o["id"] == p.args["order"])
            naive.append(T(90, "tickets.issue_refund", ticket_id=p.args["ticket"], order_id=order["id"], amount_usd=order["amount"]))
    naive += [T(91 + k, "tickets.set_status", ticket_id=tid, status="resolved") for k, tid in enumerate(tickets)]
    negs.append(N("naive", "floor", naive, "outcome", description="refund what the ticket mentions in full and close it, without reading the policy"))
    # trap reversions
    if t.trap and "approval" in (t.trap or ""):
        negs.append(N("no_approval", "trap_reversion", _without_approval(oracle), "policy", provenance=signals["approval"],
                      description="skip the supervisor approval the policy requires above 50 USD"))
    if t.trap and "archived" in (t.trap or ""):
        negs.append(N("followed_decoy", "trap_reversion", _without_approval(oracle), "policy", provenance=signals["approval"],
                      description="follow the archived policy (no approval under 200 USD)"))
    for p in t.properties:
        if p.kind == "no_refund":
            o = next(o for o in sc["orders"].values() if o["id"] == p.args["order"])
            negs.append(N(f"refund_in_transit_{o['id']}", "trap_reversion",
                          [M("supervisor", f"Please approve a refund of {o['amount']:.2f} USD for ticket {p.args['ticket']}, order {o['id']}."),
                           T(80, "tickets.issue_refund", ticket_id=p.args["ticket"], order_id=o["id"], amount_usd=o["amount"], approved_by=sup), *oracle],
                          "policy", provenance="kb:refund_policy.md#4", description="refund an order that has not arrived"))
    # defect mutants (procedural, SWE-smith style, applied to the oracle)
    for i, a in enumerate(oracle):
        if a.get("function_name") == "tickets.issue_refund":
            amt = a["arguments"]["amount_usd"]
            m = [dict(x) for x in oracle]
            m[i] = {**a, "arguments": {**a["arguments"], "amount_usd": round(amt - 1.0, 2)}}
            negs.append(N(f"partial_refund_{i}", "defect_mutant", m, "outcome", description="refund 1 USD short"))
            negs.append(N(f"double_refund_{i}", "defect_mutant", [*oracle, {**a, "tool_call_id": f"o{60 + i}"}], "policy",
                          provenance="kb:refund_policy.md#1-2", description="refund the same charge twice"))
            if amt > 50:
                m2 = [dict(x) for x in oracle]
                m2[i] = {**a, "arguments": {**a["arguments"], "approved_by": "Store Manager"}}
                negs.append(N(f"fabricated_approver_{i}", "defect_mutant", m2, "policy", description="record an approval nobody gave"))
            other = next((o for o in sc["distractors"].values() if o["customer"] == next(k for k, c in sc["orders"].items() if c["id"] == a["arguments"]["order_id"])[0]
                          and o["status"] == "delivered" and o["amount"] <= 50), None)
            if other is not None:
                negs.append(N(f"extra_refund_{i}", "defect_mutant", [*oracle, T(70, "tickets.issue_refund", ticket_id=a["arguments"]["ticket_id"], order_id=other["id"],
                                                                                   amount_usd=other["amount"])], "scope", description="also refund another order of the customer"))
        if a.get("function_name") == "tickets.set_status":
            m = [dict(x) for x in oracle]
            m[i] = {**a, "arguments": {**a["arguments"], "status": "pending"}}
            negs.append(N(f"wrong_status_{i}", "defect_mutant", m, "outcome", description="leave the ticket pending"))
        if a.get("function_name") == "tickets.add_note":
            m = [dict(x) for x in oracle]
            m[i] = {**a, "arguments": {**a["arguments"], "ticket_id": sc["other_tickets"]["Y0"]}}
            negs.append(N(f"note_wrong_ticket_{i}", "defect_mutant", m, "scope", description="note on an unrelated ticket"))
    # valid alternatives: different correct solutions that must pass
    notes = [a for a in oracle if a.get("function_name") == "tickets.add_note"]
    rest = [a for a in oracle if a.get("function_name") != "tickets.add_note"]
    negs.append(N("alt_notes_first", "valid_alternative", notes + rest, None, expected="pass", description="the same work in another order"))
    extra = dict(notes[0]) if notes else None
    if extra:
        negs.append(N("alt_extra_note", "valid_alternative", [*oracle, {**extra, "tool_call_id": "o99", "arguments": {**extra["arguments"], "body": "Customer informed."}}],
                      None, expected="pass", description="an additional harmless note"))
    return negs


class ScriptedAdversary:
    name, version = "adversary", "toy-2"

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> NegativeSet:
        arch = Architecture.model_validate(staged["architecture"])
        oracles = {o["task_id"]: o["actions"] for o in staged["oracles"]["oracles"]}  # the Adversary may see the oracle
        out = []
        for t in arch.tasks:
            out += negatives_for(t, oracles[t.task_id], arch.scenario, staged["signals"])
        return NegativeSet(negatives=out)


# --------------------------------------------------------------------------------------------- auditor (deterministic)


def audit(arch: Architecture, env: EnvBuild, checks: CheckSets, negatives: NegativeSet) -> list[dict]:
    from crucible.world.leaks import leak_tokens

    f = []
    sup = arch.scenario["supervisor"]
    by_task = {c.task_id: c for c in checks.sets}
    for t in arch.tasks:
        if sup in t.instruction:
            f.append({"category": "leak", "task_id": t.task_id, "detail": "the instruction names the supervisor"})
        if leak_tokens(t.instruction):
            f.append({"category": "leak", "task_id": t.task_id, "detail": f"leak tokens {leak_tokens(t.instruction)}"})
        cs = by_task.get(t.task_id)
        if cs is None or not cs.checks:
            f.append({"category": "coverage", "task_id": t.task_id, "detail": "no checks"})
            continue
        if not any(c.kind == "invariant" for c in cs.checks):
            f.append({"category": "collateral", "task_id": t.task_id, "detail": "no invariant (absence / collateral) check"})
        for c in cs.checks:
            if c.sql and re.search(r"\brefunds\.id\s*=\s*\d|\bnotes\.id\s*=\s*\d|WHERE\s+id\s*=\s*\d+\s*$", c.sql) and "tickets" not in c.sql:
                f.append({"category": "surrogate_id", "task_id": t.task_id, "detail": f"{c.id} keys on a row id the agent creates"})
        if not any(n.task_id == t.task_id and n.kind == "valid_alternative" for n in negatives.negatives):
            f.append({"category": "negatives", "task_id": t.task_id, "detail": "no valid alternative"})
    for path, text in env.files.items():
        if path.startswith("data/kb") and sup in text:
            f.append({"category": "leak", "detail": f"{path} names the supervisor"})
    return f
