"""Mailbox App: email between actors and the outside world, stored in SQLite (state = ``sql``).

Tools: send_email(to, subject, body, thread_id=None), list_inbox(folder="inbox", unread_only=False),
read_email(id), search_email(query). Timestamps come from the World's virtual clock, so replies
"arrive" at the right virtual time. One mailbox serves many addresses; ``actor`` is the sender.
Params: ``addresses`` (actor id -> email address), ``owner`` (whose inbox list_inbox shows by default).
"""

from __future__ import annotations

import contextvars
from typing import Any

from crucible.core.schema.actions import ToolSpec
from crucible.world.apps.base import App, ToolError

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id TEXT NOT NULL,
  sender TEXT NOT NULL,
  recipients TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL,
  sent_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reads (message_id INTEGER NOT NULL, reader TEXT NOT NULL, PRIMARY KEY (message_id, reader));
"""

_CODE = '''
import sqlite3, json
def main(a):
    con = sqlite3.connect(a["path"]); con.row_factory = sqlite3.Row
    try:
        con.executescript(a["schema"])
        op = a["op"]
        if op == "init":
            return {}
        if op == "send":
            thread = a.get("thread_id") or ("t" + str((con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]) + 1))
            cur = con.execute("INSERT INTO messages(thread_id, sender, recipients, subject, body, sent_at) VALUES (?,?,?,?,?,?)",
                              (thread, a["sender"], json.dumps(sorted(a["to"])), a["subject"], a["body"], a["now"]))
            con.commit(); return {"id": cur.lastrowid, "thread_id": thread}
        if op == "inbox":
            rows = con.execute("SELECT m.id, m.thread_id, m.sender, m.subject, m.sent_at, (r.reader IS NOT NULL) AS read FROM messages m "
                               "LEFT JOIN reads r ON r.message_id = m.id AND r.reader = ? WHERE EXISTS (SELECT 1 FROM json_each(m.recipients) WHERE value = ?) "
                               "ORDER BY m.sent_at, m.id", (a["who"], a["who"])).fetchall()
            out = [dict(r) for r in rows]
            if a.get("unread_only"):
                out = [r for r in out if not r["read"]]
            return {"messages": out}
        if op == "sent":
            return {"messages": [dict(r) for r in con.execute("SELECT id, thread_id, recipients, subject, sent_at FROM messages WHERE sender = ? ORDER BY sent_at, id", (a["who"],))]}
        if op == "read":
            r = con.execute("SELECT * FROM messages WHERE id = ?", (a["id"],)).fetchone()
            if r is None or (a["who"] not in json.loads(r["recipients"]) and r["sender"] != a["who"]):
                raise ValueError("no such message")
            con.execute("INSERT OR IGNORE INTO reads(message_id, reader) VALUES (?, ?)", (a["id"], a["who"])); con.commit()
            d = dict(r); d["recipients"] = json.loads(d["recipients"]); return d
        if op == "search":
            q = "%" + a["query"] + "%"
            rows = con.execute("SELECT id, thread_id, sender, subject, sent_at FROM messages WHERE (subject LIKE ? OR body LIKE ?) "
                               "AND (sender = ? OR EXISTS (SELECT 1 FROM json_each(recipients) WHERE value = ?)) ORDER BY sent_at, id", (q, q, a["who"], a["who"])).fetchall()
            return {"messages": [dict(r) for r in rows]}
        raise ValueError("bad op")
    finally:
        con.close()
'''


_ACTOR: contextvars.ContextVar[str] = contextvars.ContextVar("mailbox_actor")

class MailboxApp(App):
    template = "common/mailbox"
    PATH_PARAMS = ("path",)

    def _addr(self, actor: str) -> str:
        return self.spec.params.get("addresses", {}).get(actor, f"{actor}@world.local")

    def _now(self) -> str:
        return self.ctx.clock.now_iso() if self.ctx.clock else "1970-01-01T00:00:00Z"

    async def _op(self, **a: Any) -> Any:
        try:
            return await self.ctx.exec.run_python(_CODE, {"path": self.state_params["path"], "schema": SCHEMA, **a})
        except Exception as e:
            raise ToolError(str(e).split("python snippet failed: ")[-1]) from e

    async def provision(self) -> None:
        await self.ctx.exec.run(f"mkdir -p $(dirname {self.state_params['path']})")
        await self._op(op="init")
        for msg in self.spec.params.get("seed_messages", []):
            await self._op(op="send", sender=msg["from"], to=msg["to"], subject=msg["subject"], body=msg["body"], thread_id=msg.get("thread_id"), now=msg.get("at", self._now()))

    async def list_tools(self) -> list[ToolSpec]:
        s = {"type": "string"}
        return [
            ToolSpec(app=self.id, name="send_email", description="Send an email.", input_schema={"type": "object", "properties": {"to": {"type": "array", "items": s}, "subject": s, "body": s, "thread_id": s}, "required": ["to", "subject", "body"]}),
            ToolSpec(app=self.id, name="list_inbox", read_only=True, description="List received emails (newest last).", input_schema={"type": "object", "properties": {"unread_only": {"type": "boolean"}}}),
            ToolSpec(app=self.id, name="list_sent", read_only=True, description="List sent emails.", input_schema={"type": "object", "properties": {}}),
            ToolSpec(app=self.id, name="read_email", description="Read one email by id (marks it read).", input_schema={"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}),
            ToolSpec(app=self.id, name="search_email", read_only=True, description="Search your emails by subject/body text.", input_schema={"type": "object", "properties": {"query": s}, "required": ["query"]}),
        ]

    async def call(self, name, args, *, actor, call_id):
        token = _ACTOR.set(actor)  # per call (asyncio tasks copy contexts): concurrent calls never swap senders
        try:
            return await super().call(name, args, actor=actor, call_id=call_id)
        finally:
            _ACTOR.reset(token)

    async def tool_send_email(self, to: list[str], subject: str, body: str, thread_id: str | None = None) -> Any:
        if not to:
            raise ToolError("no recipients")
        return await self._op(op="send", sender=self._addr(_ACTOR.get()), to=to, subject=subject, body=body, thread_id=thread_id, now=self._now())

    async def tool_list_inbox(self, unread_only: bool = False) -> Any:
        return await self._op(op="inbox", who=self._addr(_ACTOR.get()), unread_only=unread_only)

    async def tool_list_sent(self) -> Any:
        return await self._op(op="sent", who=self._addr(_ACTOR.get()))

    async def tool_read_email(self, id: int) -> Any:
        return await self._op(op="read", id=id, who=self._addr(_ACTOR.get()))

    async def tool_search_email(self, query: str) -> Any:
        return await self._op(op="search", query=query, who=self._addr(_ACTOR.get()))

    async def deliver(self, *, sender: str, to: list[str], subject: str, body: str, thread_id: str | None = None) -> dict[str, Any]:
        """System-side delivery (events / simulated actors), bypassing tool permissions."""
        return await self._op(op="send", sender=sender, to=to, subject=subject, body=body, thread_id=thread_id, now=self._now())
