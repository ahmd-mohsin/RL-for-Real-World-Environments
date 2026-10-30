"""A scripted World family for the family / instance / gate tests: every model reply is a fixture (no model is
called). The domain is a small service desk: tickets, staff, an audit log; three write tools with permission checks
that read the policy parameters."""

from __future__ import annotations

from typing import Any

from tests.unit.synth_fakes import FakeClient, const

SPEC = {
    "name": "service_desk", "keyword": "IT service desk ticket handling", "function": "IT",
    "description": "An internal IT service desk where agents assign, prioritise and close employee tickets under an SLA policy.",
    "users": ["service desk agent", "team lead", "service desk manager"],
    "entities": [{"name": "ticket", "target": True}, {"name": "staff member", "target": True}, {"name": "audit entry", "target": False},
                 {"name": "requester", "target": False}],
    "skills": [{"id": "assign", "description": "assign an open ticket to an agent with capacity"},
               {"id": "prioritise", "description": "set a ticket's priority under the SLA policy"},
               {"id": "close", "description": "close a resolved ticket with the right resolution"},
               {"id": "dedupe", "description": "close a duplicate ticket as duplicate"},
               {"id": "escalate", "description": "route a ticket the acting user may not change to a manager"},
               {"id": "report", "description": "report a count or list from the queue"}],
    "knobs": [{"name": "queue_load", "values": ["light", "heavy"]}, {"name": "requester_vip", "values": ["no", "yes"]}],
    "policy_params": [{"name": "acting_user_id", "type": "ref", "table": "staff", "where": {"role": ["lead", "manager"]}, "meaning": "who is acting"},
                      {"name": "max_open_per_agent", "type": "integer", "default": 6, "min": 3, "max": 12, "unit": "tickets",
                       "meaning": "an agent holds at most this many open tickets"},
                      {"name": "critical_needs_manager", "type": "boolean", "default": True, "meaning": "only a manager sets critical"},
                      {"name": "sla_hours_high", "type": "integer", "default": 8, "min": 4, "max": 24, "unit": "hours", "meaning": "SLA of a high ticket"}],
    "rules": ["An agent holds at most max_open_per_agent open tickets.", "Only a manager sets critical priority when critical_needs_manager.",
              "Closed tickets cannot be reassigned."],
    "permissions": {"user_table": "staff", "roles": [{"role": "agent", "may": ["close"]}, {"role": "lead", "may": ["assign", "prioritise", "close"]},
                                                    {"role": "manager", "may": ["assign", "prioritise", "close"], "limits": {"critical": True}}],
                    "acting_param": "acting_user_id"},
    "colleagues": [{"title": "desk lead", "knows": ["which tickets block payroll"]}],
    "common_practice": True, "rationale": "every company runs a service desk",
}

T = {"name": "ticket_id", "type": "string", "description": "the ticket id"}
DESIGN = {"app": "desk", "description": "The IT service desk.", "tools": [
    {"name": "list_tickets", "description": "Tickets, filtered by status (open, in_progress, closed, archived) and priority; paginated.", "read_only": True,
     "params": [{"name": "status", "type": "string", "required": False}, {"name": "priority", "type": "string", "required": False},
                {"name": "page", "type": "integer", "required": False, "default": 1}, {"name": "page_size", "type": "integer", "required": False, "default": 10}],
     "returns": {"items": [{"ticket_id": "T-1"}], "page": 1, "total": 1, "next_page": None}},
    {"name": "get_ticket", "description": "One ticket.", "read_only": True, "params": [T], "errors": ["NOT_FOUND"]},
    {"name": "list_staff", "description": "Staff with their role (agent, lead, manager) and open ticket count.", "read_only": True,
     "params": [{"name": "role", "type": "string", "required": False}]},
    {"name": "assign_ticket", "description": "Assign an open or in_progress ticket to an agent with capacity.", "writes": ["tickets", "audit"],
     "params": [T, {"name": "staff_id", "type": "string"}], "errors": ["NOT_FOUND", "INVALID_STATE", "PERMISSION_DENIED"]},
    {"name": "set_priority", "description": "Set a ticket's priority: low, medium, high or critical (critical needs a manager).", "writes": ["tickets", "audit"],
     "params": [T, {"name": "priority", "type": "string", "enum": ["low", "medium", "high", "critical"]}],
     "errors": ["NOT_FOUND", "INVALID_ARGUMENT", "PERMISSION_DENIED"]},
    {"name": "close_ticket", "description": "Close a ticket with a resolution: fixed, wontfix or duplicate.", "writes": ["tickets", "audit"],
     "params": [T, {"name": "resolution", "type": "string", "enum": ["fixed", "wontfix", "duplicate"]}],
     "errors": ["NOT_FOUND", "INVALID_ARGUMENT", "INVALID_STATE"]},
]}
SCHEMA = {"tables": [
    {"name": "tickets", "key": "ticket_id", "fields": {"ticket_id": "string", "subject": "string", "status": "string", "priority": "string",
                                                     "requester": "string", "assignee_id": "string?", "assignee_name": "string?", "resolution": "string?",
                                                     "created_at": "string", "due_at": "string"}},
    {"name": "staff", "key": "staff_id", "fields": {"staff_id": "string", "name": "string", "role": "string", "team": "string"}},
    {"name": "audit", "key": "audit_id", "fields": {"audit_id": "integer", "ticket_id": "string", "action": "string", "actor": "string", "at": "string"}},
]}

CODE = '''
def build(app):
    def param(ctx, name, default):
        p = ctx.tables["policy_params"].get(name)
        return p["value"] if p else default

    def actor(ctx):
        uid = param(ctx, "acting_user_id", None)
        s = ctx.tables["staff"].get(uid)
        if s is None:
            raise ToolError("PERMISSION_DENIED: no acting user")
        return s

    def log(ctx, ticket_id, action):
        nid = max(ctx.tables["audit"], default=0) + 1
        ctx.tables["audit"][nid] = {"audit_id": nid, "ticket_id": ticket_id, "action": action, "actor": actor(ctx)["staff_id"], "at": ctx.now()}

    def ticket(ctx, ticket_id):
        t = ctx.tables["tickets"].get(ticket_id)
        if t is None:
            raise ToolError(f"NOT_FOUND: no ticket {ticket_id}")
        return t

    @app.tool(read_only=True)
    def list_tickets(ctx, status: str = "", priority: str = "", page: int = 1, page_size: int = 10) -> dict:
        """Tickets, filtered and paginated."""
        if page < 1 or not 1 <= page_size <= 50:
            raise ToolError("INVALID_ARGUMENT: page or page_size")
        rows = [dict(r) for k, r in sorted(ctx.tables["tickets"].items()) if (not status or r["status"] == status) and (not priority or r["priority"] == priority)]
        lo = (page - 1) * page_size
        return {"items": rows[lo:lo + page_size], "page": page, "total": len(rows), "next_page": page + 1 if lo + page_size < len(rows) else None}

    @app.tool(read_only=True)
    def get_ticket(ctx, ticket_id: str) -> dict:
        """One ticket."""
        return dict(ticket(ctx, ticket_id))

    @app.tool(read_only=True)
    def list_staff(ctx, role: str = "") -> dict:
        """Staff and their open ticket count."""
        out = []
        for k, s in sorted(ctx.tables["staff"].items()):
            if role and s["role"] != role:
                continue
            n = sum(1 for t in ctx.tables["tickets"].values() if t.get("assignee_id") == k and t["status"] in ("open", "in_progress"))
            out.append({**s, "open_tickets": n})
        return {"staff": out}

    @app.tool(writes=["tickets", "audit"])
    def assign_ticket(ctx, ticket_id: str, staff_id: str) -> dict:
        """Assign a ticket."""
        t = ticket(ctx, ticket_id)
        s = ctx.tables["staff"].get(staff_id)
        if s is None:
            raise ToolError(f"NOT_FOUND: no staff {staff_id}")
        if actor(ctx)["role"] not in ("lead", "manager"):
            raise ToolError("PERMISSION_DENIED: only a lead or manager assigns")
        if t["status"] not in ("open", "in_progress"):
            raise ToolError("INVALID_STATE: the ticket is not open")
        n = sum(1 for x in ctx.tables["tickets"].values() if x.get("assignee_id") == staff_id and x["status"] in ("open", "in_progress"))
        if n >= param(ctx, "max_open_per_agent", 6):
            raise ToolError("INVALID_STATE: the agent is at capacity")
        t["assignee_id"], t["assignee_name"] = staff_id, s["name"]
        log(ctx, ticket_id, "assign")
        return {"ticket_id": ticket_id, "assignee_id": staff_id}

    @app.tool(writes=["tickets", "audit"])
    def set_priority(ctx, ticket_id: str, priority: str) -> dict:
        """Set a priority."""
        t = ticket(ctx, ticket_id)
        if priority not in ("low", "medium", "high", "critical"):
            raise ToolError("INVALID_ARGUMENT: unknown priority")
        if priority == "critical" and param(ctx, "critical_needs_manager", True) and actor(ctx)["role"] != "manager":
            raise ToolError("PERMISSION_DENIED: only a manager sets critical")
        t["priority"] = priority
        log(ctx, ticket_id, "priority")
        return {"ticket_id": ticket_id, "priority": priority}

    @app.tool(writes=["tickets", "audit"])
    def close_ticket(ctx, ticket_id: str, resolution: str) -> dict:
        """Close a ticket."""
        t = ticket(ctx, ticket_id)
        if resolution not in ("fixed", "wontfix", "duplicate"):
            raise ToolError("INVALID_ARGUMENT: unknown resolution")
        if t["status"] not in ("open", "in_progress"):
            raise ToolError("INVALID_STATE: already closed")
        t["status"], t["resolution"] = "closed", resolution
        log(ctx, ticket_id, "close")
        return {"ticket_id": ticket_id, "status": "closed"}
'''


def _fx(role: str = "lead") -> dict[str, Any]:
    return {"tickets": [{"ticket_id": "T1", "subject": "VPN down", "status": "open", "priority": "low", "requester": "a@x", "assignee_id": None,
                         "assignee_name": None, "resolution": None, "created_at": "2025-12-01", "due_at": "2025-12-03"},
                        {"ticket_id": "T2", "subject": "Printer", "status": "closed", "priority": "low", "requester": "b@x", "assignee_id": None,
                         "assignee_name": None, "resolution": "fixed", "created_at": "2025-12-01", "due_at": "2025-12-03"}],
            "staff": [{"staff_id": "S1", "name": "Ana Lead", "role": role, "team": "desk"}, {"staff_id": "S2", "name": "Bo Agent", "role": "agent", "team": "desk"}],
            "policy_params": [{"name": "acting_user_id", "value": "S1", "unit": None, "description": ""}]}


TESTS = {"tests": [
    {"id": "list", "fixture": _fx(), "calls": [{"tool": "list_tickets", "args": {"status": "open"}, "result": {"total": 1}}]},
    {"id": "list_page", "fixture": _fx(), "calls": [{"tool": "list_tickets", "args": {"page": 2, "page_size": 1}, "result": {"page": 2}}]},
    {"id": "get", "fixture": _fx(), "calls": [{"tool": "get_ticket", "args": {"ticket_id": "T1"}, "result": {"status": "open"}}]},
    {"id": "get_missing", "fixture": _fx(), "calls": [{"tool": "get_ticket", "args": {"ticket_id": "T9"}, "expect": "error"}]},
    {"id": "staff", "fixture": _fx(), "calls": [{"tool": "list_staff", "args": {"role": "agent"}}]},
    {"id": "assign", "fixture": _fx(), "calls": [{"tool": "assign_ticket", "args": {"ticket_id": "T1", "staff_id": "S2"}}],
     "expect_state": {"tickets": {"T1": {"assignee_id": "S2"}}}},
    {"id": "assign_perm", "fixture": _fx("agent"), "calls": [{"tool": "assign_ticket", "args": {"ticket_id": "T1", "staff_id": "S2"}, "expect": "error", "error": "PERMISSION_DENIED"}]},
    {"id": "assign_closed", "fixture": _fx(), "calls": [{"tool": "assign_ticket", "args": {"ticket_id": "T2", "staff_id": "S2"}, "expect": "error"}]},
    {"id": "prio", "fixture": _fx(), "calls": [{"tool": "set_priority", "args": {"ticket_id": "T1", "priority": "high"}}],
     "expect_state": {"tickets": {"T1": {"priority": "high"}}}},
    {"id": "prio_critical", "fixture": _fx(), "calls": [{"tool": "set_priority", "args": {"ticket_id": "T1", "priority": "critical"}, "expect": "error", "error": "PERMISSION_DENIED"}]},
    {"id": "close", "fixture": _fx(), "calls": [{"tool": "close_ticket", "args": {"ticket_id": "T1", "resolution": "fixed"}}],
     "expect_state": {"tickets": {"T1": {"status": "closed", "resolution": "fixed"}}}},
    {"id": "close_bad", "fixture": _fx(), "calls": [{"tool": "close_ticket", "args": {"ticket_id": "T1", "resolution": "solved"}, "expect": "error"}]},
]}

FIRST = ["Ana", "Bo", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea", "Mo", "Nia", "Omar", "Pia"]
LAST = ["Garcia", "Okafor", "Lindqvist", "Tanaka", "Moreau", "Novak", "Silva", "Kowalski", "Haddad", "Brennan"]
SUBJ = ["VPN drops every hour", "Laptop battery swelling", "Outlook search broken", "Printer on floor 3 jams", "Password reset loop",
        "Teams camera not detected", "Shared drive access", "SAP login error", "Monitor flickers", "New hire laptop setup", "MFA token lost",
        "Excel macro blocked", "Wi-Fi in room 4B", "Payroll portal 500 error", "Docking station dead"]
DATAGEN = {"tables": [
    {"table": "staff", "rows": 14, "fields": {
        "staff_id": {"kind": "seq", "prefix": "S-", "start": 101},
        "name": {"kind": "template", "template": "{first} {last}", "parts": {"first": FIRST, "last": LAST}, "unique": True},
        "role": {"kind": "choice", "values": ["agent", "lead", "manager"], "weights": [8, 3, 2]},
        "team": {"kind": "choice", "values": ["desk", "field", "apps"]}},
     "near_duplicate": {"vary": ["name"]}},
    {"table": "tickets", "rows": 60, "fields": {
        "ticket_id": {"kind": "seq", "prefix": "INC-", "start": 20001},
        "subject": {"kind": "template", "template": "{s} ({r})", "parts": {"s": SUBJ, "r": ["HQ", "Berlin", "Austin", "Pune", "remote"]}},
        "status": {"kind": "choice", "values": ["open", "in_progress", "closed", "archived"], "weights": [7, 2, 2, 0.01]},
        "priority": {"kind": "choice", "values": ["low", "medium", "high", "critical"], "weights": [4, 4, 2, 0.5]},
        "requester": {"kind": "template", "template": "{f}.{l}@acme.example", "parts": {"f": ["ana", "li", "sam", "joe", "kim"], "l": ["ng", "roy", "diaz"]}},
        "assignee_id": {"kind": "ref", "table": "staff", "null_p": 0.6},
        "assignee_name": {"kind": "lookup", "ref": "assignee_id", "field": "name"},
        "resolution": {"kind": "choice", "values": ["fixed", "wontfix", "duplicate"], "when": "status == 'closed'", "else": None},
        "created_at": {"kind": "date", "days_ago": [1, 120]},
        "due_at": {"kind": "date_offset", "from": "created_at", "min_days": 1, "max_days": 5}},
     "rules": ["due_at >= created_at"], "near_duplicate": {"vary": ["subject"]},
     "stale": {"set": {"status": "archived"}, "older": ["created_at", "due_at"]}},
    {"table": "audit", "rows": 10, "fields": {
        "audit_id": {"kind": "seq", "start": 1},
        "ticket_id": {"kind": "ref", "table": "tickets"},
        "action": {"kind": "choice", "values": ["assign", "priority", "close"]},
        "actor": {"kind": "ref", "table": "staff"},
        "at": {"kind": "date", "days_ago": [1, 60], "time": True}}},
]}

_PAD = (" Every agent follows this document; questions go to the desk lead. The steps below are mandatory and audited monthly by the "
        "service desk manager, and exceptions are recorded in the ticket's notes with the reason and the approver's name.")
DOCS = {"documents": [
    {"doc_id": "DOC-1", "title": "Service desk policy", "kind": "policy", "tags": ["policy", "sla"],
     "body": "# Policy\nAn agent holds at most {{max_open_per_agent}} open tickets. High tickets are due within {{sla_hours_high}} hours." + _PAD * 2},
    {"doc_id": "DOC-2", "title": "Priority SOP", "kind": "sop", "tags": ["priority"],
     "body": "# Setting priority\n1. Read the ticket. 2. Critical needs a manager: {{critical_needs_manager}}. 3. Record the reason." + _PAD * 2},
    {"doc_id": "DOC-3", "title": "Desk FAQ", "kind": "faq", "tags": ["faq"], "body": "# FAQ\nQ: Who assigns? A: A lead or a manager." + _PAD * 2},
    {"doc_id": "DOC-4", "title": "Data standards", "kind": "data_standards", "tags": ["data"],
     "body": "# Data\nA near-duplicate ticket (same subject) is closed as duplicate. Archived tickets are stale: never act on them. "
             "When a ticket's assignee name disagrees with the staff record, the staff record is the master." + _PAD * 2},
]}
ACTORS = {"colleagues": [{"colleague_id": "C-1", "name": "Rui Costa", "title": "desk lead", "topics": ["payroll blockers"],
                          "facts": [{"topic": "payroll blocker", "keywords": ["payroll", "blocking"], "table": "tickets",
                                     "text": "Ticket {ticket_id} ({subject}) is blocking payroll, the requester called me on {today}."}]}]}

REALISTIC = {"scores": {"R1": 4, "R2": 4, "R3": 4, "R4": 4, "R5": 5}, "realistic": True, "obscure": False, "confidence": 0.9, "problems": []}
UNREALISTIC = {"scores": {"R1": 1, "R2": 2, "R3": 3, "R4": 1, "R5": 3}, "realistic": False, "obscure": False, "confidence": 0.9, "problems": ["toy"]}


def family_clients(**over: FakeClient) -> dict[str, FakeClient]:
    cl = {"domain_spec_writer": FakeClient("glm", const(SPEC)), "family_designer": FakeClient("kimi", const(DESIGN)),
          "schema_deriver": FakeClient("kimi", const(SCHEMA)), "app_coder": FakeClient("kimi", const({"code": CODE})),
          "blind_tester": FakeClient("deepseek", const(TESTS)), "debugger": FakeClient("kimi", const({"code": CODE})),
          "test_approver": FakeClient("nemotron", const({"test_is_wrong": False})), "data_spec_writer": FakeClient("kimi", const(DATAGEN)),
          "doc_writer": FakeClient("glm", const(DOCS)), "actor_writer": FakeClient("deepseek", const(ACTORS))}
    cl.update(over)
    return cl
