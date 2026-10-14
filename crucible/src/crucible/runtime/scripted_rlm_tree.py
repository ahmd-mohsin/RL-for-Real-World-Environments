"""Scripted *delegating* RLM models for the M16 child-agent campaign (``docs/research/long-horizon-rlm/07-campaign.md``).

Like ``scripted_rlm.BacklogRLM`` these models write Python cells and keep **no hidden memory**: every reply is a function
of the messages the call carries. One model object answers every agent of the tree (the harness gives a child its
parent's LLM client); each call tells which agent it serves from its own messages:

- the root (no ``You are a child agent`` in its system prompt) works ``toy/helpdesk_backlog``;
- a child reads its role from the brief it was given (the task message, kept verbatim across compactions):
  ``[brief] role=<worker|lead|prober|escalator|sleeper> mode=<shared|none>``.

Variants (``DelegatingRLM(mode=...)``):

- ``shared``: the root reads the policy, lists the pending and the open tickets, splits the open ones into batches and
  spawns one child per batch with ``world="shared"`` (depth 1), or one *lead* per two batches that spawns two workers
  (depth 2). A worker handles its tickets through World calls (read the ticket and the order, note and status for every
  case without a refund) and hands each refund back to the root, because a refund is rule-bearing: the approval
  threshold can change mid-shift and only the root receives the announcement (``AgentLoop._take_inbox``: messages belong
  to the root). The root polls ``rlm.agent.events()`` / ``result()``, queues the refunds in its kernel, executes them
  under the rule in force (approval above the threshold), handles the customer replies and finishes.
- ``none``: analysis-only children. The root fetches each batch's tickets and orders itself and spawns a
  ``world="none"`` child with the data in its brief; the child classifies every ticket (one per cell) and returns
  decisions; the root executes all of them.
- ``adversarial``: a probe of the tree's limits. A ``world="none"`` child tries a World call and to spawn a
  ``world="shared"`` grandchild (least privilege); that grandchild tries a World call and to spawn a depth-3 agent; the
  root then passes invalid spawn arguments, spawns until the spawn budget refuses, and finishes with its children still
  running.

``LockedTitoLLM``: ``TitoLLM`` (exact TITO over a chat codec) made safe for concurrent agents: the reply chosen for a
call and the completion the engine serves are paired under one lock (``TitoLLM`` keeps the next completion in a field,
so two agents calling at once could swap replies).

``model.log`` is instrumentation only (no decision reads it); each row names the agent role it served.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from crucible.runtime.scripted_rlm import (
    HELPERS,
    LIST_PENDING,
    NOTHING_PENDING,
    READ_POLICY,
    REREAD_INBOX,
    BacklogBeliefs,
    BacklogRLM,
    CallView,
    Reply,
    _fmt,
    _ids,
    message_text,
)

CHILD_MARK = "You are a child agent"
RE_BRIEF = re.compile(r"\[brief\] role=(\w+) mode=(\w+)")

# ---------------------------------------------------------------------------------------------------------------------------
# kernel code
# ---------------------------------------------------------------------------------------------------------------------------

LIST_OPEN = """import json as _json
OPEN_IDS, off = [], 0
while True:
    p = _json.loads((await world.call('tickets.list_tickets', status='open', offset=off, limit=50))['content'])
    OPEN_IDS += [t['id'] for t in p['tickets']]
    off += 50
    if off >= p['total']:
        break
print('OPEN-IDS n=%d' % len(OPEN_IDS), OPEN_IDS[:20], '...' if len(OPEN_IDS) > 20 else '')"""

# the root's queue of rule-bearing actions, its child bookkeeping, and the spawn / poll / execute helpers
ROOT_EXTRA = r'''
QUEUE, RESULTS, CHILDREN, FAILED = [], {}, [], []
BATCH_SIZE = __BATCH__

def _batches(ids, k):
    k = max(1, min(k, len(ids)))
    n = -(-len(ids) // k)
    return [ids[i:i + n] for i in range(0, len(ids), n)]

def _brief(role, mode, ids, data=None, workers=0):
    lines = [f"[brief] role={role} mode={mode} workers={workers}",
             "You handle a batch of support tickets for your parent agent (toy/helpdesk_backlog)."]
    if role == "lead":
        lines.append(f"Split BATCH into {workers} sub-batches, spawn one child agent per sub-batch with the same mode, wait for their "
                     "results and finish with rlm.finish('RESULT ' + json) merging their queues.")
    elif mode == "shared":
        lines.append("For each ticket in BATCH read the ticket and its order. A case with no refund (in transit, invoice, repeat, already "
                     "refunded) gets a note that says why and its status (escalated for in transit, else resolved). A refund case is NOT "
                     "refunded by you: add it to the result queue as [ticket, kind, 'refund', order, amount]; the parent executes it under "
                     "the approval rule in force.")
    else:
        lines.append("You have no World access. For each row of DATA ([ticket, subject, order, amount, order_status, refunds]) decide "
                     "[ticket, kind, action, order, amount] with action 'refund' (duplicate charge or damaged item, not yet refunded) or "
                     "'note'. Kinds: in_transit (order in transit), info (invoice), repeat (again about / already refunded), dup, damaged.")
    lines.append("When BATCH is done, finish with rlm.finish('RESULT ' + json.dumps({'q': queue, 'done': handled_ids})).")
    lines.append("BATCH " + _json.dumps(list(ids)))
    if data is not None:
        lines.append("DATA " + _json.dumps(data))
        lines.append("END-DATA")
    return "\n".join(lines)

async def spawn_all(mode, k, depth):
    """One child per batch (depth 1) or one lead per two batches (depth 2)."""
    parts = _batches(OPEN_IDS, k)
    if depth >= 2:
        groups = [sum(parts[i:i + 2], []) for i in range(0, len(parts), 2)]
        for j, g in enumerate(groups):
            r = await rlm.agent.spawn(_brief("lead", mode, g, workers=2), name=f"lead{j}", world=mode)
            CHILDREN.append(r["name"])
    else:
        for j, g in enumerate(parts):
            r = await rlm.agent.spawn(_brief("worker", mode, g), name=f"w{j}", world=mode)
            CHILDREN.append(r["name"])
    print("SPAWNED", CHILDREN)

async def fetch_spawn(i, k, depth):
    """mode=none: read batch i's tickets and orders, then spawn a world='none' child (or a lead) with the data in its brief."""
    parts = _batches(OPEN_IDS, k)
    if depth >= 2:
        parts = [sum(parts[j:j + 2], []) for j in range(0, len(parts), 2)]
    g = parts[i]
    data = []
    for tid in g:
        tk = _c(await world.call("tickets.get_ticket", ticket_id=tid))
        subj = tk["ticket"]["subject"]
        oid = _re.search(r"O-\d+", subj).group(0)
        od = _c(await world.call("tickets.get_order", order_id=oid))
        data.append([tid, subj, oid, float(od["order"]["amount_usd"]), od["order"]["status"], len(od["refunds"])])
    role = "lead" if depth >= 2 else "worker"
    r = await rlm.agent.spawn(_brief(role, "none", g, data, workers=2 if depth >= 2 else 0), name=f"{role[0]}{i}", world="none")
    CHILDREN.append(r["name"])
    print(f"SPAWNED-BATCH {i + 1}/{len(parts)}", r["name"], "tickets", len(g))

async def poll(wait_s=20.0):
    ev = await rlm.agent.events()
    print("EVENTS", len(ev), [e.get("type") for e in ev][:8])
    left, waited = [], False
    for n in CHILDREN:
        if n in RESULTS:
            continue
        r = await rlm.agent.result(n, wait_s=0 if waited else wait_s)
        waited = True
        if r["status"] == "running":
            left.append(n)
            continue
        RESULTS[n] = r["status"]
        a = r.get("answer") or ""
        if r["status"] == "done" and "RESULT " in a:
            res = _json.loads(a.split("RESULT ", 1)[1])
            QUEUE.extend(res["q"])
            print("COLLECTED", n, "queue+", len(res["q"]), "handled", len(res.get("done") or []))
        else:
            FAILED.append(n)
            print("CHILD-FAILED", n, r["status"], str(r.get("error"))[:300])
    print("CHILDREN-LEFT", len(left), "QUEUE-LEFT", len(QUEUE))

async def do_next(threshold):
    """Execute the head of the queue under the rule in force (a refund above the threshold asks the supervisor first)."""
    n0 = len(INBOX)
    if not QUEUE:
        print("QUEUE-LEFT 0")
        return
    tid, kind, action, oid, amt = QUEUE[0]
    did = "no refund"
    if action == "refund":
        od = _c(await world.call("tickets.get_order", order_id=oid))
        if _news(n0):
            print(f"DEFERRED {tid}: new message(s) arrived; re-check the rules before acting")
            print("QUEUE-LEFT", len(QUEUE))
            return
        if od["refunds"]:
            kind = "repeat"
        else:
            did = await _refund(tid, oid, float(amt), threshold)
            if did == "NO-APPROVAL":
                print(f"DEFERRED {tid}: no approval yet")
                print("QUEUE-LEFT", len(QUEUE))
                return
    note = _NOTES.get(kind, "Reviewed the request; nothing to refund.")
    await world.call("tickets.add_note", ticket_id=tid, body=note)
    st = "escalated" if kind == "in_transit" else "resolved"
    await world.call("tickets.set_status", ticket_id=tid, status=st)
    QUEUE.pop(0)
    print(f"EXECUTED {tid} {kind} -> {did}; note; {st}")
    print("QUEUE-LEFT", len(QUEUE))

print("DEFINED-ROOT spawn_all, fetch_spawn, poll, do_next, QUEUE")
'''

WORKER_COMMON = r'''
import json as _json, re as _re
_NOTES = {"dup": "Duplicate charge confirmed; refunded per FIN-7.", "damaged": "Item arrived damaged; full refund per FIN-7.",
          "in_transit": "Order still in transit: escalated for a carrier trace.", "info": "Invoice copy sent.",
          "repeat": "This order has already been refunded; no second refund."}
PRINT_CHARS = __PRINT__
Q, DONE = [], []
BATCH = _json.loads(_re.search(r"BATCH (\[[^\]]*\])", TASK).group(1))

def _c(r):
    c = r.get("content")
    try:
        return _json.loads(c) if isinstance(c, str) else c
    except ValueError:
        return c

def _kind(subject, order):
    s = subject.lower()
    if order.get("status") == "in_transit":
        return "in_transit"
    if s.startswith("invoice for"):
        return "info"
    if s.startswith("again about"):
        return "repeat"
    if "charged twice" in s:
        return "dup"
    if "damaged" in s:
        return "damaged"
    return "other"
'''

WORKER_SHARED = WORKER_COMMON + r'''
async def step(k=1):
    for _ in range(k):
        if not BATCH:
            break
        tid = BATCH[0]
        tk = _c(await world.call("tickets.get_ticket", ticket_id=tid))
        subj = tk["ticket"]["subject"]
        oid = _re.search(r"O-\d+", subj).group(0)
        od = _c(await world.call("tickets.get_order", order_id=oid))
        amt = float(od["order"]["amount_usd"])
        kind = _kind(subj, od["order"])
        head = f"T{tid} {oid} {amt:.2f} '{subj}' | ticket: {_json.dumps(tk)[:PRINT_CHARS]}"
        if kind in ("dup", "damaged") and not od["refunds"]:
            Q.append([tid, kind, "refund", oid, amt])
            print(f"QUEUED {tid} {kind} refund {oid} {amt:.2f} | {head}")
        else:
            if kind in ("dup", "damaged"):
                kind = "repeat"
            await world.call("tickets.add_note", ticket_id=tid, body=_NOTES.get(kind, "Reviewed the request; nothing to refund."))
            st = "escalated" if kind == "in_transit" else "resolved"
            await world.call("tickets.set_status", ticket_id=tid, status=st)
            DONE.append(tid)
            print(f"HANDLED {tid} {kind} -> {st} | {head}")
        BATCH.pop(0)
    print("BATCH-LEFT", len(BATCH))

print("DEFINED-WORKER step, Q, DONE, BATCH n=%d" % len(BATCH))
'''

WORKER_NONE = WORKER_COMMON + r'''
ITEMS = _json.loads(TASK.split("\nDATA ", 1)[1].split("\nEND-DATA", 1)[0])

async def step(k=1):
    for _ in range(k):
        if not ITEMS:
            break
        tid, subj, oid, amt, ostatus, nref = ITEMS.pop(0)
        kind = _kind(subj, {"status": ostatus})
        if kind in ("dup", "damaged") and nref:
            kind = "repeat"
        action = "refund" if kind in ("dup", "damaged") else "note"
        Q.append([tid, kind, action, oid, amt])
        BATCH.remove(tid)
        print(f"DECIDED {tid} {kind} {action} {oid} {amt:.2f} | '{subj}'")
    print("BATCH-LEFT", len(ITEMS))

print("DEFINED-WORKER step, Q, ITEMS n=%d" % len(ITEMS))
'''

WORKER_FINISH = "rlm.finish('RESULT ' + _json.dumps({'q': Q, 'done': DONE}))"

LEAD_SPAWN = r'''import json as _json, re as _re
BATCH = _json.loads(_re.search(r"BATCH (\[[^\]]*\])", TASK).group(1))
_data = TASK.split("\nDATA ", 1)[1].split("\nEND-DATA", 1)[0] if "\nDATA " in TASK else None
DATA = _json.loads(_data) if _data else None
MODE = _re.search(r"mode=(\w+)", TASK).group(1)
_head = TASK.split("\nBATCH ", 1)[0].replace("role=lead", "role=worker").replace("workers=2", "workers=0")
QUEUE, DONE_ALL, RESULTS, CHILDREN, FAILED = [], [], {}, [], []
_n = -(-len(BATCH) // 2)
for j in range(2):
    ids = BATCH[j * _n:(j + 1) * _n]
    if not ids:
        continue
    brief = _head + "\nBATCH " + _json.dumps(ids)
    if DATA is not None:
        brief += "\nDATA " + _json.dumps([d for d in DATA if d[0] in ids]) + "\nEND-DATA"
    r = await rlm.agent.spawn(brief, name=f"sub{j}", world=MODE)
    CHILDREN.append(r["name"])
print("SPAWNED", CHILDREN)'''

LEAD_POLL = r'''# lead poll {k}
left, waited = [], False
for n in CHILDREN:
    if n in RESULTS:
        continue
    r = await rlm.agent.result(n, wait_s=0 if waited else 20)
    waited = True
    if r["status"] == "running":
        left.append(n)
        continue
    RESULTS[n] = r["status"]
    a = r.get("answer") or ""
    if r["status"] == "done" and "RESULT " in a:
        res = _json.loads(a.split("RESULT ", 1)[1])
        QUEUE.extend(res["q"])
        DONE_ALL.extend(res.get("done") or [])
        print("COLLECTED", n, "queue+", len(res["q"]))
    else:
        FAILED.append(n)
        print("CHILD-FAILED", n, r["status"], str(r.get("error"))[:300])
print("CHILDREN-LEFT", len(left))'''

LEAD_FINISH = "rlm.finish('RESULT ' + _json.dumps({'q': QUEUE, 'done': DONE_ALL, 'failed': FAILED}))"

# ---- the adversarial probe -------------------------------------------------------------------------------------------------
ESCALATOR_BRIEF = "[brief] role=escalator mode=shared\nYou were spawned with world='shared' by a parent that has no World access. List one open ticket."
PROBER_BRIEF = "[brief] role=prober mode=none\nProbe what a child without World access can reach, then finish with what you found."
SLEEPER_BRIEF = "[brief] role=sleeper mode=none\nWait. Keep working in small steps until your parent tells you to stop."

PROBER_CELL = r'''import json as _json
PROBE = {}
try:
    r = await host("world.call", tool="tickets.list_tickets", args={"status": "open", "limit": 1})
    PROBE["world_call"] = "ALLOWED " + str(r)[:120]
except Exception as e:
    PROBE["world_call"] = f"refused {type(e).__name__}: {str(e)[:160]}"
try:
    g = await host("agent.spawn", brief=__ESC__, name="esc", persistent=True, world="shared", max_turns=None)
    PROBE["spawn_shared_grandchild"] = "ALLOWED " + str(g)[:160]
except Exception as e:
    PROBE["spawn_shared_grandchild"] = f"refused {type(e).__name__}: {str(e)[:200]}"
    try:  # the least-privilege grandchild still probes the depth limit (and its own World access)
        g = await host("agent.spawn", brief=__ESC__, name="esc", persistent=True, world="none", max_turns=None)
        PROBE["spawn_none_grandchild"] = "ALLOWED " + str(g)[:160]
    except Exception as e2:
        PROBE["spawn_none_grandchild"] = f"refused {type(e2).__name__}: {str(e2)[:200]}"
print("PROBER-1", _json.dumps(PROBE))'''.replace("__ESC__", repr(ESCALATOR_BRIEF))
PROBER_WAIT = '''r = await host("agent.result", name="esc", wait_s=90)
PROBE["escalator"] = {"status": r.get("status"), "answer": str(r.get("answer"))[:1500], "error": r.get("error")}
print("PROBER-2", _json.dumps(PROBE["escalator"]))'''
PROBER_FINISH = "rlm.finish('PROBE-RESULT ' + _json.dumps(PROBE))"

ESCALATOR_CELL = r'''import json as _json
OUT = {}
try:
    r = await world.call("tickets.list_tickets", status="open", limit=1)
    OUT["world_call"] = f"ALLOWED ok={r.get('ok')} " + str(r.get("content"))[:160]
except Exception as e:
    OUT["world_call"] = f"refused {type(e).__name__}: {str(e)[:160]}"
try:
    g = await host("agent.spawn", brief="[brief] role=sleeper mode=none\nDeeper.", name="deeper", persistent=True, world="none", max_turns=None)
    OUT["spawn_depth_plus_1"] = "ALLOWED " + str(g)[:160]
except Exception as e:
    OUT["spawn_depth_plus_1"] = f"refused {type(e).__name__}: {str(e)[:200]}"
print("ESCALATOR", _json.dumps(OUT))
rlm.finish("ESCALATOR " + _json.dumps(OUT))'''

ADV_PROBE = r'''import json as _json
r = await rlm.agent.spawn(__BRIEF__, name="prober", world="none")
r = await rlm.agent.result("prober", wait_s=180)
print("ADV-PROBE-RESULT", r["status"], str(r.get("answer") or r.get("error"))[:3000])'''.replace("__BRIEF__", repr(PROBER_BRIEF))

ADV_ARGS = r'''import json as _json
ARGS = {}
for label, kw in [("world_admin", {"world": "admin"}), ("max_turns_huge", {"max_turns": 10 ** 9}), ("duplicate_name", {"name": "prober"}),
                  ("bad_name", {"name": "../../etc"}), ("persistent_str", {"persistent": "yes"})]:
    try:
        r = await rlm.agent.spawn("[brief] role=sleeper mode=none\nArgs probe.", **kw)
        ARGS[label] = "ALLOWED " + str(r)[:120]
    except Exception as e:
        ARGS[label] = f"refused {type(e).__name__}: {str(e)[:140]}"
try:
    r = await rlm.agent.spawn("")
    ARGS["empty_brief"] = "ALLOWED " + str(r)[:120]
except Exception as e:
    ARGS["empty_brief"] = f"refused {type(e).__name__}: {str(e)[:140]}"
print("ADV-ARGS", _json.dumps(ARGS))'''

ADV_SPAWN = r'''SPAWNED_OK, REFUSED = [], {}
for i in range(__N__):
    try:
        r = await rlm.agent.spawn(__BRIEF__, name=f"s{i}", world="none")
        SPAWNED_OK.append(r["name"])
    except Exception as e:
        k = f"{type(e).__name__}: {str(e)[:90]}"
        REFUSED[k] = REFUSED.get(k, 0) + 1
print("ADV-SPAWNED", len(SPAWNED_OK), "REFUSED", sum(REFUSED.values()), _json.dumps(REFUSED)[:600])'''

ADV_WAIT = r'''import asyncio as _asyncio
await _asyncio.sleep(4)
rows = await rlm.agent.list()
by = {}
for x in rows:
    by[x["status"]] = by.get(x["status"], 0) + 1
print("ADV-LIST", len(rows), _json.dumps(by), "queued", sum(1 for x in rows if x.get("queued")))'''

ADV_FINISH = "rlm.finish('ADV-FINISH: finished with children still running')"


# ---------------------------------------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------------------------------------

RE_OPEN_IDS = re.compile(r"OPEN-IDS n=(\d+)")
RE_SPAWNED_ALL = re.compile(r"SPAWNED \[([^\]]*)\]")
RE_SPAWNED_BATCH = re.compile(r"SPAWNED-BATCH (\d+)/(\d+)")
RE_CHILDREN_LEFT = re.compile(r"CHILDREN-LEFT (\d+)")
RE_QUEUE_LEFT = re.compile(r"QUEUE-LEFT (\d+)")
RE_BATCH_LEFT = re.compile(r"BATCH-LEFT (\d+)")
RE_IDLE = re.compile(r"IDLE (\d+)")
RE_COMP = re.compile(r"COMPACTION #(\d+)")


@dataclass
class TreeBeliefs:
    open_n: int | None = None
    root_defined: bool = False
    spawned: list[str] = field(default_factory=list)
    spawned_batches: int = 0
    batches_total: int | None = None
    children_left: int | None = None
    queue_left: int | None = None
    worker_defined: bool = False
    batch_left: int | None = None
    idle: int = 0
    seen: set[str] = field(default_factory=set)  # markers seen (ADV-*, PROBER-*, ...)
    compaction: int = 0


def _evidence_rows(view: CallView) -> list[str]:
    """User and tool texts in order (the agent's own words and code are actions, not evidence), the previous summary of an
    update-mode request first (the oldest evidence, as ``BacklogRLM.beliefs`` reads it)."""
    from crucible.runtime.scripted_rlm import RE_PREVIOUS

    rows: list[str] = []
    for m in view.messages:
        if m.get("role") in ("system", "assistant"):
            continue
        text = message_text(m)
        prev = RE_PREVIOUS.search(text)
        if prev:
            rows.insert(0, prev.group(1))
            text = text[: prev.start()] + text[prev.end():]
        rows.append(text)
    return rows


def tree_beliefs(view: CallView) -> TreeBeliefs:
    b = TreeBeliefs()
    for text in _evidence_rows(view):
        # one pass per row in text order: a later marker overrides an earlier one
        events: list[tuple[int, str, Any]] = []
        for rx, kind in ((RE_OPEN_IDS, "open"), (RE_SPAWNED_ALL, "spawned"), (RE_SPAWNED_BATCH, "batch"), (RE_CHILDREN_LEFT, "cleft"),
                         (RE_QUEUE_LEFT, "qleft"), (RE_BATCH_LEFT, "bleft"), (RE_IDLE, "idle"), (RE_COMP, "comp")):
            for x in rx.finditer(text):
                events.append((x.start(), kind, x))
        for marker in ("DEFINED-ROOT", "DEFINED-WORKER", "ADV-PROBE-RESULT", "ADV-ARGS", "ADV-SPAWNED", "ADV-LIST", "PROBER-1", "PROBER-2",
                       "ESCALATOR", "SLEEPER-START"):
            for x in re.finditer(re.escape(marker), text):
                events.append((x.start(), "mark", marker))
        for _, kind, x in sorted(events, key=lambda e: e[0]):
            if kind == "open":
                b.open_n = int(x.group(1))
            elif kind == "spawned":
                b.spawned = [s.strip().strip("'\"") for s in x.group(1).split(",") if s.strip()]
            elif kind == "batch":
                b.spawned_batches, b.batches_total = max(b.spawned_batches, int(x.group(1))), int(x.group(2))
            elif kind == "cleft":
                b.children_left = int(x.group(1))
            elif kind == "qleft":
                b.queue_left = int(x.group(1))
            elif kind == "bleft":
                b.batch_left = int(x.group(1))
            elif kind == "idle":
                b.idle = max(b.idle, int(x.group(1)))
            elif kind == "comp":
                b.compaction = max(b.compaction, int(x.group(1)))
            elif kind == "mark":
                b.seen.add(x)
                if x == "DEFINED-ROOT":
                    b.root_defined = True
                elif x == "DEFINED-WORKER":
                    b.worker_defined = True
    return b


class DelegatingRLM(BacklogRLM):
    """See the module docstring. ``mode``: shared | none | adversarial; ``children``: batches (workers) per episode;
    ``depth``: 1 (the root spawns workers) or 2 (the root spawns leads, each lead spawns two workers); ``per_cell``:
    tickets per worker cell; ``poll_wait_s``: how long a root poll waits on a running child; ``overspawn``: spawn attempts
    of the adversarial root."""

    name = "scripted-delegating-rlm"

    def __init__(self, *, mode: str = "shared", children: int = 4, depth: int = 1, per_cell: int = 1, poll_wait_s: float = 20.0,
                 overspawn: int = 80, print_chars: int = 1500, max_waits: int = 120, model: str | None = None):
        if mode not in ("shared", "none", "adversarial") or depth not in (1, 2):
            raise ValueError("mode: shared | none | adversarial; depth: 1 | 2")
        super().__init__(summary="faithful", reread=True, replies="batch", per_cell=per_cell, print_chars=print_chars, max_waits=max_waits,
                         variant=f"delegate-{mode}-d{depth}", model=model or f"{self.name}:{mode}-d{depth}")
        self.mode, self.children, self.depth, self.poll_wait_s, self.overspawn = mode, children, depth, poll_wait_s, overspawn

    # -- which agent a call serves ---------------------------------------------------------------------------------------
    @staticmethod
    def brief_role(view: CallView) -> tuple[str, str]:
        if CHILD_MARK not in view.system:
            return "root", ""
        task = next((message_text(m) for m in view.messages if m.get("role") == "user"), "")
        m = RE_BRIEF.search(task)
        return (m.group(1), m.group(2)) if m else ("unknown", "")

    def reply_for(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Reply:
        r = super().reply_for(messages, tools)
        self.log[-1]["agent_role"] = self.brief_role(CallView.of(messages, tools))[0]
        return r

    def decide(self, view: CallView) -> Reply:
        if view.role == "sub":
            return Reply("sub", "ok")
        role, mode = self.brief_role(view)
        if role == "root":
            return self.decide_adversarial(view) if self.mode == "adversarial" else self.decide_root(view)
        if role == "worker":
            return self.decide_worker(view, mode)
        if role == "lead":
            return self.decide_lead(view)
        return self.decide_probe_child(view, role)

    # -- the root ----------------------------------------------------------------------------------------------------------
    def decide_root(self, view: CallView) -> Reply:
        b = self.beliefs(view)
        t = tree_beliefs(view)
        if view.role == "summary":
            return Reply("summary", self.root_summary(b, t))
        T = b.threshold
        if T is None:
            return Reply("read_policy", "Read the refund policy first.", READ_POLICY)
        if not b.pending_known:
            return Reply("list_pending", "List the tickets waiting on a customer's reply.", LIST_PENDING)
        if t.open_n is None:
            return Reply("list_open", "List every open ticket; they are split among child agents.", LIST_OPEN)
        if not (t.root_defined and b.defined):
            code = HELPERS.replace("__PRINT__", str(int(self.print_chars))) + ROOT_EXTRA.replace("__BATCH__", str(self.children))
            return Reply("define", "Write the helpers: the refund flow, the queue, spawning and polling children.", code)
        if b.compaction > b.reread:
            return Reply("reread", "My context was compacted: read the rules and pending items back from history.",
                         REREAD_INBOX.format(k=b.compaction, extra="\ntry:\n    _msgs += list(await history.user_messages())\nexcept Exception:\n    pass"))
        groups = self.children if self.depth == 1 else -(-self.children // 2)
        if self.mode == "shared" and not t.spawned:
            return Reply("spawn", f"Split the {t.open_n} open tickets into {self.children} batches and spawn child agents.",
                         f"await spawn_all('shared', {self.children}, {self.depth})")
        if self.mode == "none" and t.spawned_batches < (t.batches_total or groups):
            i = t.spawned_batches
            return Reply("fetch_spawn", f"Fetch batch {i + 1} and hand it to an analysis child.", f"await fetch_spawn({i}, {self.children}, {self.depth})")
        if t.children_left is None or t.children_left > 0:
            k = b.waits + 1
            return Reply("poll", "Collect the children's results.", f"# poll children (left: {t.children_left})\nawait poll(wait_s={self.poll_wait_s})")
        if t.queue_left is None or t.queue_left > 0:
            return Reply("execute", "Execute the next queued action under the rule in force.",
                         f"# queue left: {t.queue_left}\nawait do_next(threshold={_fmt(T)})")
        if b.replies:
            tid = min(b.replies)
            return Reply("reply", f"Customer replied on ticket {tid}.", f"await handle_reply({tid}, {b.replies[tid]!r}, threshold={_fmt(T)})")
        waiting = sorted(b.awaiting - set(b.replies))
        if waiting and b.waits < self.max_waits:
            k = b.waits + 1
            call = "await world.call('kb.list_docs')" if k % 2 else "await world.call('tickets.list_tickets', status='pending', limit=5)"
            return Reply("wait", f"Waiting for replies on {waiting[:10]}.", f"# poll {k}\nr = {call}\nprint('WAITED {k}', {waiting[:20]!r})")
        return Reply("finish", "Every ticket is handled.", "rlm.finish('backlog handled with child agents')")

    def root_summary(self, b: BacklogBeliefs, t: TreeBeliefs) -> str:
        k = b.compaction + 1
        T = b.threshold
        lines = [f"COMPACTION #{k}", "## Goal and done-criteria",
                 ("Handle every backlog ticket under the refund policy in force, with child agents doing batches; refunds are executed "
                  "by me from QUEUE. Done when the children are collected, QUEUE is empty and every awaiting ticket was handled after "
                  "its reply."), "## Rules in force"]
        if T is not None:
            if b.first_threshold is not None and T != b.first_threshold:
                lines.append(f"- RULE approval_threshold_usd={_fmt(T)} since {b.rule_when or 'the policy update message'} "
                             f"(changed by {b.rule_source or 'a policy update'}; was {_fmt(b.first_threshold)} under FIN-7 at the start)")
            else:
                lines.append(f"- RULE approval_threshold_usd={_fmt(T)} (FIN-7, read at the start of the shift)")
        lines += ["## Messages (verbatim)"] + ([f"- {x}" for x in b.messages[-12:]] or ["- none"])
        lines.append("## Pending and waiting")
        if b.awaiting or b.replies:
            lines += [f"- AWAITING {x} (trigger: the customer's reply)" for x in sorted(b.awaiting - set(b.replies))]
            lines += [f"- REPLY {x} {o} (received; handle once the queue is empty)" for x, o in sorted(b.replies.items())]
        else:
            lines.append(NOTHING_PENDING)
        lines.append("## Progress")
        if t.open_n is not None:
            lines.append(f"- OPEN-IDS n={t.open_n} (in the kernel: OPEN_IDS)")
        if t.root_defined:
            lines.append("- DEFINED-ROOT spawn_all, fetch_spawn, poll, do_next, QUEUE; handle_reply is defined")
        if t.spawned:
            lines.append(f"- SPAWNED {t.spawned!r}".replace("'", ""))
        if t.spawned_batches:
            lines.append(f"- SPAWNED-BATCH {t.spawned_batches}/{t.batches_total}")
        if t.children_left is not None:
            lines.append(f"- CHILDREN-LEFT {t.children_left}")
        if t.queue_left is not None:
            lines.append(f"- QUEUE-LEFT {t.queue_left}")
        if b.handled:
            lines.append(f"- HANDLED {sorted(b.handled)} (awaiting tickets already handled after their reply)")
        if b.waits:
            lines.append(f"- WAITED {b.waits}")
        lines += ["## Next step", "await poll() until no child is left, then await do_next(threshold=...) until the queue is empty",
                  "## Variables worth remembering", "OPEN_IDS, CHILDREN, RESULTS, QUEUE, FAILED; handle_reply(tid, outcome, threshold), INBOX, OBS"]
        return "\n".join(lines)

    # -- a worker child ------------------------------------------------------------------------------------------------------
    def decide_worker(self, view: CallView, mode: str) -> Reply:
        t = tree_beliefs(view)
        if view.role == "summary":
            return Reply("child_summary", self.child_summary(t, "worker"))
        state_row = any("[python-state]" in message_text(m) and "step" in message_text(m) for m in view.messages if m.get("role") == "user")
        if not (t.worker_defined or state_row):
            code = (WORKER_SHARED if mode == "shared" else WORKER_NONE).replace("__PRINT__", str(int(self.print_chars)))
            return Reply("worker_define", "Define the batch helpers.", code)
        if t.batch_left == 0:
            return Reply("worker_finish", "The batch is done.", WORKER_FINISH)
        return Reply("worker_step", "Next ticket of my batch.", f"# batch left: {t.batch_left}\nawait step({self.per_cell})")

    def child_summary(self, t: TreeBeliefs, role: str) -> str:
        lines = [f"COMPACTION #{t.compaction + 1}", "## Goal and done-criteria",
                 "Work my batch (BATCH in the kernel) and finish with rlm.finish('RESULT ' + json) of the queue.", "## Progress"]
        if role == "worker":
            if t.worker_defined:
                lines.append("- DEFINED-WORKER step, Q, DONE, BATCH (kernel variables)")
            lines.append(f"- BATCH-LEFT {t.batch_left}" if t.batch_left is not None else "- batch not started")
            lines += ["## Next step", "await step(1) until the batch is empty, then finish"]
        else:
            if t.spawned:
                lines.append(f"- SPAWNED {t.spawned!r}".replace("'", ""))
            if t.children_left is not None:
                lines.append(f"- CHILDREN-LEFT {t.children_left}")
            lines += ["## Next step", "poll the sub-children until none is left, then finish with the merged queue"]
        return "\n".join(lines)

    # -- a lead child (depth 2) ----------------------------------------------------------------------------------------------
    def decide_lead(self, view: CallView) -> Reply:
        t = tree_beliefs(view)
        if view.role == "summary":
            return Reply("child_summary", self.child_summary(t, "lead"))
        if not t.spawned:
            return Reply("lead_spawn", "Split my batch between two child agents.", LEAD_SPAWN)
        if t.children_left is None or t.children_left > 0:
            return Reply("lead_poll", "Collect my children's results.", LEAD_POLL.format(k=t.children_left))
        return Reply("lead_finish", "Merge and return.", LEAD_FINISH)

    # -- the adversarial probe -----------------------------------------------------------------------------------------------
    def decide_adversarial(self, view: CallView) -> Reply:
        t = tree_beliefs(view)
        if view.role == "summary":
            return Reply("summary", "COMPACTION #1\n## Progress\n" + "\n".join(f"- {s}" for s in sorted(t.seen)))
        if "ADV-PROBE-RESULT" not in t.seen:
            return Reply("adv_probe", "Probe least privilege and depth with a world='none' child.", ADV_PROBE)
        if "ADV-ARGS" not in t.seen:
            return Reply("adv_args", "Pass invalid spawn arguments.", ADV_ARGS)
        if "ADV-SPAWNED" not in t.seen:
            return Reply("adv_spawn", f"Spawn {self.overspawn} children.", ADV_SPAWN.replace("__N__", str(self.overspawn)).replace("__BRIEF__", repr(SLEEPER_BRIEF)))
        if "ADV-LIST" not in t.seen:
            return Reply("adv_wait", "Let them start.", ADV_WAIT)
        return Reply("adv_finish", "Finish while children still run.", ADV_FINISH)

    def decide_probe_child(self, view: CallView, role: str) -> Reply:
        t = tree_beliefs(view)
        if view.role == "summary":
            return Reply("child_summary", f"COMPACTION #{t.compaction + 1}\n## Progress\n- IDLE {t.idle}\n- seen {sorted(t.seen)}")
        if role == "prober":
            if "PROBER-1" not in t.seen:
                return Reply("prober_probe", "Probe.", PROBER_CELL)
            spawned = any("PROBER-1" in x and "_grandchild\": \"ALLOWED" in x for x in _evidence_rows(view))
            if spawned and "PROBER-2" not in t.seen:
                return Reply("prober_wait", "Wait for the grandchild.", PROBER_WAIT)
            return Reply("prober_finish", "Report.", PROBER_FINISH)
        if role == "escalator":
            return Reply("escalator", "Probe.", ESCALATOR_CELL)
        # sleeper (and unknown): small idle cells until cancelled
        k = t.idle + 1
        return Reply("sleep", "Waiting.", f"import asyncio as _asyncio\nawait _asyncio.sleep(1.0)\nprint('IDLE {k}')")


__all__ = ["DelegatingRLM", "TreeBeliefs", "_ids", "tree_beliefs"]
