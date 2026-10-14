"""Scripted RLM models for the long-horizon ladders (M16 design §4; study 02 H12).

A scripted model here is an ``LLMClient`` (``complete(messages, tools, max_tokens, temperature, sample_key, seed)``) that
solves a World task by **writing Python cells** for the RLM harness, and that keeps **no hidden memory**: every reply is
a pure function of the messages it is sent. What it knows after a compaction is what its visible context says (the
task, the summary text, the python-state row, the recent tail, tool outputs) or what its own cells read back from the
kernel (``OBS`` / ``INBOX`` / ``HISTORY`` today; ``history.*`` / ``OUT`` when the system prompt names them). So a summary
that drops a rule really loses the rule, which ``ReplayLLM`` (an action list indexed by tool-call ids) can never show
(study 02 §8.1: "no decision depends on what the summary says").

Two deliveries of the same decisions:

- ``ScriptedRLM.complete``: an ``LLMResponse`` with estimated usage (chars/4, ``usage_source="estimate"``) and no token
  record. Fast; unit tests and smoke runs.
- ``TitoLLM(model, codec)``: exact token-in/token-out through ``train.world_env.EngineLLM`` over a chat codec (the real
  Qwen2.5 tokenizer via ``HFChatCodec``, or ``ByteChatCodec`` in tests): the decision is taken from the messages, its
  reply is encoded with the codec and served as the engine's sampled completion, so prompts, deltas, segments and the
  rollout's token records are exactly what a served model would produce (the method of study 03's ``stress.py``).

The backlog model (``BacklogRLM``) works ``toy/helpdesk_backlog``. Its beliefs come from evidence, not from its own
earlier code: the approval threshold from the policy document it read, the ops announcement, a summary's ``RULE`` line
or a re-read; awaiting tickets from the pending listing it printed, the summary or a re-read; customer replies from the
INBOX announcements, the summary or a re-read. Its kernel helpers (``work``, ``handle``, ``handle_reply``) take the
threshold as an argument, so the rule in force lives in the context, which is what a compaction can lose. Knobs:

- ``summary``: ``"faithful"`` writes every schema section (rules in force with when each changed, every message
  verbatim, pending and awaiting ids with their triggers, progress, next step, variables); ``"forgetful"`` keeps the
  rule as first read (it omits the 50 -> 25 amendment) and writes "nothing pending" (it omits the awaiting ids and the
  unhandled replies).
- ``reread``: after each compaction, one cell reads the past back from the kernel (rule messages, awaiting ids,
  replies) before acting. ``False`` = never consults history.
- ``replies``: ``"batch"`` (default) handles customer replies once the open queue is empty, so a reply must survive
  compactions in the model's memory; ``"immediate"`` handles each reply on the next turn.

Presets (``BACKLOG_VARIANTS``): ``faithful`` (faithful + reread), ``forgetful`` (forgetful, no reread), ``lazy``
(faithful, never consults history) and ``forgetful_reread`` (a forgetful summary repaired from history).

``model.log`` is instrumentation for the ladder (what each call was, prompt sizes); no decision reads it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from crucible.actors.llm import LLMResponse

PY = "python"


def _approx(text: str) -> int:
    return max(1, len(text) // 4)


def message_text(m: dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return "" if c is None else str(c)


def cell_code(m: dict[str, Any]) -> str:
    """The python code an assistant message ran (OpenAI-style ``tool_calls`` with JSON arguments)."""
    out = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments") if fn else tc.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if isinstance(args, dict):
            out.append(str(args.get("code") or ""))
    return "\n".join(out)


@dataclass
class Reply:
    kind: str  # cell kind (work, reread, ...) | summary | sub | text
    text: str = ""
    code: str | None = None


@dataclass
class CallView:
    """What a call is, read from its messages and tools alone."""

    role: str  # root | summary | sub
    system: str
    messages: list[dict[str, Any]]

    @classmethod
    def of(cls, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> CallView:
        system = next((message_text(m) for m in messages if m.get("role") == "system"), "")
        if not system:
            return cls("sub", "", messages)  # an RLM sub-LLM call: one user message, no system prompt
        if not tools:
            return cls("summary", system, messages)  # the compaction request: no tools offered
        return cls("root", system, messages)

    def has_history_api(self) -> bool:
        return "history." in self.system

    def has_context_api(self) -> bool:
        return "CONTEXT" in self.system


class ScriptedRLM:
    """Base: ``decide(view) -> Reply``; subclasses implement it from the visible messages only."""

    name = "scripted-rlm"

    def __init__(self, *, model: str | None = None):
        self.model = model or self.name
        self.log: list[dict[str, Any]] = []  # instrumentation only: never read by decide()
        self.calls = 0

    # -- decision ----------------------------------------------------------------------------------------------------------
    def decide(self, view: CallView) -> Reply:  # pragma: no cover - abstract
        raise NotImplementedError

    def reply_for(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Reply:
        view = CallView.of(messages, tools)
        r = self.decide(view)
        self.log.append({"call": self.calls, "role": view.role, "kind": r.kind, "messages": len(messages),
                         "prompt_chars": sum(len(message_text(m)) + len(cell_code(m)) for m in messages)})
        self.calls += 1
        return r

    @staticmethod
    def hermes(r: Reply) -> str:
        """The reply as a completion in the hermes tool-call format both chat codecs parse."""
        if r.code is None:
            return r.text
        call = "<tool_call>\n" + json.dumps({"name": PY, "arguments": {"code": r.code}}) + "\n</tool_call>"
        return (r.text + "\n" if r.text else "") + call

    # -- LLMClient ---------------------------------------------------------------------------------------------------------
    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, max_tokens: int = 2000,
                       temperature: float = 0.0, sample_key: str | None = None, seed: int | None = None, **_: Any) -> LLMResponse:
        r = self.reply_for(messages, tools)
        prompt = sum(_approx(message_text(m) + cell_code(m)) for m in messages)
        completion = _approx(self.hermes(r))
        self.log[-1].update(prompt_tokens=prompt, completion_tokens=completion, usage_source="estimate")
        calls = [] if r.code is None else [{"id": f"py{len(messages)}", "name": PY, "arguments": {"code": r.code}}]
        return LLMResponse(text=r.text, tool_calls=calls, usage={"prompt_tokens": prompt, "completion_tokens": completion},
                           model=self.model, raw_finish="stop")


class TitoLLM:
    """Exact TITO for a scripted model: ``EngineLLM`` over ``codec`` serves the model's reply as the sampled completion."""

    def __init__(self, model: ScriptedRLM, codec: Any, *, name: str | None = None):
        from crucible.train.world_env import EngineLLM

        self.model, self.codec = model, codec
        self._next: list[int] = []
        self.engine = EngineLLM(codec, self._gen, model=name or model.model)
        self.spend = self.engine.spend

    def _encode(self, text: str) -> list[int]:
        enc = getattr(self.codec, "encode_text", None)
        ids = enc(text) if callable(enc) else self.codec.tok.encode(text, add_special_tokens=False)
        return list(ids) + [self.codec.eos_ids[0]]

    async def _gen(self, prompt: list[int], max_tokens: int, temperature: float, *, seed: int | None = None, stop_ids: Any = ()) -> Any:
        from crucible.train.engine import GenResult

        ids = self._next
        if self.model.log:
            self.model.log[-1].update(prompt_tokens=len(prompt), completion_tokens=len(ids), usage_source="tito")
        return GenResult(completion_ids=ids, logprobs=[-0.1] * len(ids), version=0, finish="stop", prompt_tokens=len(prompt))

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
        r = self.model.reply_for(messages, tools)
        self._next = self._encode(self.model.hermes(r))
        return await self.engine.complete(messages, tools=tools, **kw)


def qwen_codec(name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Any:
    """The real Qwen2.5 chat codec from the local Hugging Face cache (offline)."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    from crucible.train.chat import HFChatCodec

    return HFChatCodec(AutoTokenizer.from_pretrained(name))


# ---------------------------------------------------------------------------------------------------------------------------
# The backlog model
# ---------------------------------------------------------------------------------------------------------------------------

# FIN-7's "why" notes (the backlog oracle's wording, ``crucible_pack_toy.longhorizon.world.NOTES``)
HELPERS = r'''
import json as _json, re as _re
_NOTES = {"dup": "Duplicate charge confirmed; refunded per FIN-7.", "damaged": "Item arrived damaged; full refund per FIN-7.",
          "in_transit": "Order still in transit: escalated for a carrier trace.", "info": "Invoice copy sent.",
          "repeat": "This order has already been refunded; no second refund."}
PRINT_CHARS = __PRINT__

def _c(r):
    c = r.get("content")
    try:
        return _json.loads(c) if isinstance(c, str) else c
    except ValueError:
        return c

def _news(n0):
    return [m for m in INBOX[n0:] if m.get("from") != "supervisor"]

async def _refund(tid, oid, amt, threshold):
    ab = ""
    if amt > threshold:
        k = len(INBOX)
        await world.send("supervisor", f"Please approve a refund of {amt:.2f} USD for ticket {tid}, order {oid}.")
        ok = [m for m in INBOX[k:] if m.get("from") == "supervisor" and f"{amt:.2f} USD" in str(m.get("text"))]
        if not ok:  # the answer comes with the next observation
            await world.call("tickets.get_order", order_id=oid)
            ok = [m for m in INBOX[k:] if m.get("from") == "supervisor" and f"{amt:.2f} USD" in str(m.get("text"))]
        if not ok:
            return "NO-APPROVAL"
        ab = str(ok[-1]["text"]).rsplit("- ", 1)[-1].strip()
    r = await world.call("tickets.issue_refund", ticket_id=tid, order_id=oid, amount_usd=amt, approved_by=ab)
    if not r.get("ok"):
        return "REFUND-FAILED " + str(r.get("error"))[:80]
    return f"refunded {amt:.2f}" + (f" approved_by={ab}" if ab else " (no approval needed)")

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

async def handle(tid, threshold, n0):
    tk = _c(await world.call("tickets.get_ticket", ticket_id=tid))
    subj = tk["ticket"]["subject"]
    oid = _re.search(r"O-\d+", subj).group(0)
    od = _c(await world.call("tickets.get_order", order_id=oid))
    head = f"T{tid} {oid} {od['order']['amount_usd']:.2f} '{subj}' | ticket: {_json.dumps(tk)[:PRINT_CHARS]}"
    if _news(n0):
        return f"DEFERRED {tid}: new message(s) arrived; re-check the rules before acting | {head}"
    kind = _kind(subj, od["order"])
    did = ""
    if kind in ("dup", "damaged") and not od["refunds"]:
        did = await _refund(tid, oid, float(od["order"]["amount_usd"]), threshold)
        if did == "NO-APPROVAL":
            return f"DEFERRED {tid}: no approval yet | {head}"
    elif kind in ("dup", "damaged"):
        kind = "repeat"
    note = _NOTES.get(kind, "Reviewed the request; nothing to refund.")
    await world.call("tickets.add_note", ticket_id=tid, body=note)
    st = "escalated" if kind == "in_transit" else "resolved"
    await world.call("tickets.set_status", ticket_id=tid, status=st)
    return f"DONE {tid} {kind} -> {did or 'no refund'}; note; {st} | {head}"

async def work(threshold, k=1):
    n0 = len(INBOX)
    page = _c(await world.call("tickets.list_tickets", status="open", limit=k))
    done = 0
    for t in page["tickets"]:
        out = await handle(t["id"], threshold, n0)
        print(out)
        if out.startswith("DEFERRED"):
            break
        done += 1
    print("OPEN-LEFT", max(0, page["total"] - done))

async def handle_reply(tid, outcome, threshold):
    n0 = len(INBOX)
    tk = _c(await world.call("tickets.get_ticket", ticket_id=tid))
    oid = _re.search(r"O-\d+", tk["ticket"]["subject"]).group(0)
    od = _c(await world.call("tickets.get_order", order_id=oid))
    if _news(n0):
        print(f"DEFERRED-REPLY {tid}: new message(s) arrived; re-check the rules before acting")
        return
    did = "closed"
    if outcome == "lost":
        if od["refunds"]:
            did = "already refunded"
        else:
            did = await _refund(tid, oid, float(od["order"]["amount_usd"]), threshold)
            if did == "NO-APPROVAL":
                print(f"DEFERRED-REPLY {tid}: no approval yet")
                return
        await world.call("tickets.add_note", ticket_id=tid, body="Customer replied: package lost; refunded the order.")
    else:
        await world.call("tickets.add_note", ticket_id=tid, body="Customer replied: package found; closing the request.")
    await world.call("tickets.set_status", ticket_id=tid, status="resolved")
    print(f"HANDLED-REPLY {tid} {outcome} -> {did}; note; resolved")

print("DEFINED work, handle, handle_reply")
'''

READ_POLICY = "r = await world.call('kb.read_doc', path='refund_policy.md')\nprint(r['content'])"
LIST_PENDING = """import json as _json
ids, off = [], 0
while True:
    p = _json.loads((await world.call('tickets.list_tickets', status='pending', offset=off, limit=50))['content'])
    ids += [t['id'] for t in p['tickets']]
    off += 50
    if off >= p['total']:
        break
print('PENDING-IDS', ids)"""

# after a compaction: read the past back from the kernel (rule messages, awaiting ids, replies), then act
REREAD_INBOX = """import json as _json, re as _re
print('REREAD #{k}')
_msgs = [m for m in INBOX]{extra}
for m in _msgs:
    t = str(m.get('text', '')) if isinstance(m, dict) else str(m)
    if 'refunds above' in t.lower() and '(was' in t:
        print('RULE-MSG', (m.get('at') if isinstance(m, dict) else None), t[:240])
_pend = sorted({{x['id'] for o in OBS if o.get('tool') == 'tickets.list_tickets' and (o.get('args') or {{}}).get('status') == 'pending'
               and o.get('ok') for x in _json.loads(o['content'])['tickets']}})
_done = {{(o.get('args') or {{}}).get('ticket_id') for o in OBS if o.get('tool') == 'tickets.set_status' and o.get('ok')}}
print('AWAITING', [t for t in _pend if t not in _done])
for m in _msgs:
    t = str(m.get('text', '')) if isinstance(m, dict) else str(m)
    h = _re.search(r'Ticket (\\d+): ', t)
    if h and int(h.group(1)) not in _done:
        print('REPLY-MSG', t[:240])
print('HANDLED', sorted(t for t in _pend if t in _done))"""
HISTORY_EXTRA = "\ntry:\n    _msgs += list(await history.user_messages())\nexcept Exception:\n    pass"
# M16 (agent F): a re-read whose output was cut at the harness's output cap is read on in pages from the ledger copy the cut
# names (``OUT(i, cursor=...)``), with an overlap so no line is split between pages (E's 5,000 rung lost 53 replies to the cut)
PAGE_CELL = "_p = await OUT({i}, cursor={c}, max_chars={n})\nprint('PAGE {i}', {c}, _p['next'])\nprint(_p['text'])"
PAGE_CHARS, PAGE_OVERLAP = 6_000, 300
RE_PREVIOUS = re.compile(r"<previous-summary>(.*?)</previous-summary>", re.DOTALL)
RE_CUT = re.compile(r"characters truncated here; full output: await OUT\((\d+), cursor=(\d+)\)")
RE_PAGE = re.compile(r"PAGE (\d+) (\d+) (\d+|None)")

RE_POLICY = re.compile(r"Refunds above (\d+(?:\.\d+)?) USD need the supervisor's approval")
RE_AMENDED = re.compile(r"FIN-7a \(current, amended ([0-9T:\-]+)\)")
RE_ANNOUNCE = re.compile(r"refunds above (\d+(?:\.\d+)?) USD \(was (\d+(?:\.\d+)?)\)")
RE_RULE_MSG = re.compile(r"RULE-MSG (\S+) ")
RE_RULE = re.compile(r"RULE approval_threshold_usd=(\d+(?:\.\d+)?)(?: since ([^\n(;]+))?(?:\([^)\n]*?was (\d+(?:\.\d+)?))?")
RE_PENDING = re.compile(r"PENDING-IDS \[([\d, ]*)\]")
RE_AWAITING_LIST = re.compile(r"AWAITING \[([\d, ]*)\]")
RE_AWAITING_ONE = re.compile(r"- AWAITING (\d+)")
RE_REPLY = re.compile(r"Ticket (\d+): (I found the package|It is definitely lost)")
RE_REPLY_SUM = re.compile(r"- REPLY (\d+) (lost|found)")
RE_HANDLED = re.compile(r"HANDLED-REPLY (\d+)")
RE_HANDLED_LIST = re.compile(r"HANDLED \[([\d, ]*)\]")
RE_OPEN = re.compile(r"OPEN-LEFT (\d+)")
RE_COMPACTION = re.compile(r"COMPACTION #(\d+)")
RE_REREAD = re.compile(r"REREAD #(\d+)")
RE_WAITED = re.compile(r"WAITED (\d+)")
RE_MESSAGE = re.compile(r"from (ops|supervisor|customer[^:]*): ([^\n;]{1,240})")
NOTHING_PENDING = "- nothing pending"


def _ids(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", s)]


def _fmt(x: float) -> str:
    return f"{x:g}"


@dataclass
class BacklogBeliefs:
    threshold: float | None = None
    first_threshold: float | None = None
    rule_when: str | None = None  # when the rule in force took effect (as the evidence states it)
    rule_source: str | None = None
    pending_known: bool = False
    awaiting: set[int] = field(default_factory=set)
    replies: dict[int, str] = field(default_factory=dict)
    handled: set[int] = field(default_factory=set)
    open_left: int | None = None
    defined: bool = False
    compaction: int = 0
    reread: int = 0
    waits: int = 0
    messages: list[str] = field(default_factory=list)


class BacklogRLM(ScriptedRLM):
    name = "scripted-backlog-rlm"

    def __init__(self, *, summary: str = "faithful", reread: bool = True, replies: str = "batch", per_cell: int = 1, print_chars: int = 1500,
                 max_waits: int = 60, variant: str | None = None, model: str | None = None):
        if summary not in ("faithful", "forgetful") or replies not in ("batch", "immediate"):
            raise ValueError("summary: faithful | forgetful; replies: batch | immediate")
        self.summary, self.reread, self.replies = summary, reread, replies
        self.per_cell, self.print_chars, self.max_waits = per_cell, print_chars, max_waits
        self.variant = variant or f"{summary}{'+reread' if reread else ''}"
        super().__init__(model=model or f"{self.name}:{self.variant}")

    # -- beliefs from the visible messages ---------------------------------------------------------------------------------
    def beliefs(self, view: CallView) -> BacklogBeliefs:
        b = BacklogBeliefs()
        # M16 (agent F): a summary request in update mode (context.UPDATE_PROMPT) carries the previous checkpoint inside
        # <previous-summary> in its last message; it is the oldest evidence, so it is read first and the work since then
        # (earlier in the message list) updates it, as the prompt asks
        rows: list[tuple[str, str]] = []
        for m in view.messages:
            if m.get("role") in ("system", "assistant"):
                continue  # its own earlier words and code are actions, not evidence (module docstring)
            text = message_text(m)
            prev = RE_PREVIOUS.search(text)
            if prev:
                rows.insert(0, ("user", prev.group(1)))
                text = text[: prev.start()] + text[prev.end():]
            rows.append((str(m.get("role")), text))
        for _role, text in rows:
            events: list[tuple[int, str, Any]] = []
            for x in RE_POLICY.finditer(text):
                am = RE_AMENDED.search(text)
                events.append((x.start(), "rule", (float(x.group(1)), am.group(1) if am else None, "the refund policy document", None)))
            for x in RE_ANNOUNCE.finditer(text):
                at = None
                for y in RE_RULE_MSG.finditer(text[: x.start()]):
                    at = y.group(1) if y.group(1) != "None" else None
                events.append((x.start(), "rule", (float(x.group(1)), at, "the ops policy update message", float(x.group(2)))))
            for x in RE_RULE.finditer(text):
                was = float(x.group(3)) if x.group(3) else None
                events.append((x.start(), "rule", (float(x.group(1)), (x.group(2) or "").strip() or None, "the policy update" if was else "FIN-7", was)))
            for x in RE_PENDING.finditer(text):
                events.append((x.start(), "pending", _ids(x.group(1))))
            for x in RE_AWAITING_LIST.finditer(text):
                events.append((x.start(), "pending", _ids(x.group(1))))
            for x in RE_AWAITING_ONE.finditer(text):
                events.append((x.start(), "pending", [int(x.group(1))]))
            if "## Pending and waiting" in text:
                events.append((text.index("## Pending and waiting"), "pending", []))
            for x in RE_REPLY.finditer(text):
                events.append((x.start(), "reply", (int(x.group(1)), "found" if "found" in x.group(2) else "lost")))
            for x in RE_REPLY_SUM.finditer(text):
                events.append((x.start(), "reply", (int(x.group(1)), x.group(2))))
            for x in RE_HANDLED.finditer(text):
                events.append((x.start(), "handled", [int(x.group(1))]))
            for x in RE_HANDLED_LIST.finditer(text):
                events.append((x.start(), "handled", _ids(x.group(1))))
            for x in RE_OPEN.finditer(text):
                events.append((x.start(), "open", int(x.group(1))))
            for x in RE_WAITED.finditer(text):
                events.append((x.start(), "waited", int(x.group(1))))
            for x in RE_MESSAGE.finditer(text):
                events.append((x.start(), "message", f"from {x.group(1)}: {x.group(2).strip()}"))
            for x in RE_COMPACTION.finditer(text):
                b.compaction = max(b.compaction, int(x.group(1)))
            for x in RE_REREAD.finditer(text):
                b.reread = max(b.reread, int(x.group(1)))
            if re.search(r"\bhandle_reply\b", text):
                b.defined = True
            for _, kind, data in sorted(events, key=lambda e: e[0]):
                self._apply(b, kind, data)
        b.awaiting -= b.handled
        for t in list(b.replies):
            if t in b.handled:
                del b.replies[t]
        return b

    @staticmethod
    def _apply(b: BacklogBeliefs, kind: str, data: Any) -> None:
        if kind == "rule":
            value, when, source, was = data
            if b.first_threshold is None:
                b.first_threshold = was if was is not None else value
            if value != b.threshold:
                b.rule_when, b.rule_source = when, source
            elif when:
                b.rule_when = when
            b.threshold = value
        elif kind == "pending":
            b.pending_known = True
            b.awaiting |= set(data)
        elif kind == "reply":
            tid, outcome = data
            if tid not in b.handled:
                b.replies[tid] = outcome
                b.awaiting.add(tid)  # a reply comes only for a ticket that waits on it
        elif kind == "handled":
            b.handled |= set(data)
            for t in data:
                b.replies.pop(t, None)
                b.awaiting.discard(t)
        elif kind == "open":
            b.open_left = data
        elif kind == "waited":
            b.waits = data
        elif kind == "message" and data not in b.messages and not data.startswith("from supervisor: Approved refund"):
            b.messages.append(data)

    # -- decisions --------------------------------------------------------------------------------------------------------
    def decide(self, view: CallView) -> Reply:
        if view.role == "sub":
            return Reply("sub", "ok")
        b = self.beliefs(view)
        if view.role == "summary":
            return Reply("summary", self.write_summary(b))
        T = b.threshold
        if T is None:
            return Reply("read_policy", "Read the refund policy first.", READ_POLICY)
        if not b.pending_known:
            return Reply("list_pending", "List the tickets waiting on a customer's reply.", LIST_PENDING)
        if not b.defined:
            return Reply("define", "Write the helpers that handle a ticket.", HELPERS.replace("__PRINT__", str(int(self.print_chars))))
        # it pages a cut re-read only when its checkpoint did not keep the pending items (it relies on the re-read then)
        lacks = any(NOTHING_PENDING in message_text(m) for m in view.messages if m.get("role") == "user")
        page = self.next_page(view) if self.reread and lacks and view.has_history_api() else None
        if page is not None:
            return Reply("page", "Read the rest of that output from history.", PAGE_CELL.format(i=page[0], c=page[1], n=PAGE_CHARS))
        if self.reread and b.compaction > b.reread:
            extra = HISTORY_EXTRA if view.has_history_api() else ""
            return Reply("reread", "My context was compacted: read the rules and pending items back from history.",
                         REREAD_INBOX.format(k=b.compaction, extra=extra))
        if self.replies == "immediate" and b.replies:
            tid = min(b.replies)
            return Reply("reply", f"Customer replied on ticket {tid}.", f"await handle_reply({tid}, {b.replies[tid]!r}, threshold={_fmt(T)})")
        if b.open_left is None or b.open_left > 0:
            left = "?" if b.open_left is None else b.open_left  # the progress note keeps consecutive cells distinct (as a model's would)
            return Reply("work", "Next open ticket.", f"# open tickets left: {left}\nawait work(threshold={_fmt(T)}, k={self.per_cell})")
        if b.replies:
            tid = min(b.replies)
            return Reply("reply", f"Customer replied on ticket {tid}.", f"await handle_reply({tid}, {b.replies[tid]!r}, threshold={_fmt(T)})")
        waiting = sorted(b.awaiting - set(b.replies))
        if waiting and b.waits < self.max_waits:
            k = b.waits + 1
            call = "await world.call('kb.list_docs')" if k % 2 else "await world.call('tickets.list_tickets', status='pending', limit=5)"
            return Reply("wait", f"Waiting for replies on {waiting[:10]}.", f"# poll {k}\nr = {call}\nprint('WAITED {k}', {waiting[:20]!r})")
        return Reply("finish", "Every ticket is handled.", "rlm.finish('backlog handled')")

    @staticmethod
    def next_page(view: CallView) -> tuple[int, int] | None:
        """(ledger index, cursor) of the next page of a cut re-read, read from the last tool row alone."""
        last = next((m for m in reversed(view.messages) if m.get("role") == "tool"), None)
        text = message_text(last) if last is not None else ""
        pages = RE_PAGE.findall(text)
        if pages:
            i, _, nxt = pages[-1]
            return None if nxt == "None" else (int(i), max(0, int(nxt) - PAGE_OVERLAP))
        cut = RE_CUT.search(text) if "REREAD #" in text else None
        return (int(cut.group(1)), max(0, int(cut.group(2)) - PAGE_OVERLAP)) if cut else None

    def write_summary(self, b: BacklogBeliefs) -> str:
        k = b.compaction + 1
        faithful = self.summary == "faithful"
        T = b.threshold if faithful else (b.first_threshold if b.first_threshold is not None else b.threshold)
        lines = [f"COMPACTION #{k}", "## Goal and done-criteria",
                 ("Handle every backlog ticket under the refund policy in force; done when no open ticket is left and every awaiting "
                  "ticket was handled after its customer's reply. The kernel keeps running; history keeps everything."),
                 "## Rules in force"]
        if T is not None:
            if faithful and b.first_threshold is not None and T != b.first_threshold:
                lines.append(f"- RULE approval_threshold_usd={_fmt(T)} since {b.rule_when or 'the policy update message'} "
                             f"(changed by {b.rule_source or 'a policy update'}; was {_fmt(b.first_threshold)} under FIN-7 at the start)")
            else:
                lines.append(f"- RULE approval_threshold_usd={_fmt(T)} (FIN-7, read at the start of the shift)")
        lines += ["- One approval covers one refund: ask the supervisor naming ticket, order and amount; record approved_by.",
                  "- A ticket waiting for the customer's reply stays pending until the reply; every handled ticket gets a note that says why.",
                  "## Messages (verbatim)"]
        msgs = [x for x in b.messages if faithful or "Policy update" not in x]
        msgs = [x for x in msgs if faithful or not RE_REPLY.search(x)]
        lines += [f"- {x}" for x in msgs[-12:]] or ["- none"]
        lines.append("## Pending and waiting")
        if faithful and (b.awaiting or b.replies):
            lines += [f"- AWAITING {t} (trigger: the customer's reply)" for t in sorted(b.awaiting - set(b.replies))]
            lines += [f"- REPLY {t} {o} (received; handle once the open queue is empty)" for t, o in sorted(b.replies.items())]
        else:
            lines.append(NOTHING_PENDING)
        lines += ["## Progress", f"- OPEN-LEFT {b.open_left}" if b.open_left is not None else "- open queue not listed yet"]
        if faithful and b.handled:
            lines.append(f"- HANDLED {sorted(b.handled)} (awaiting tickets already handled after their reply)")
        lines += ["## Next step", f"await work(threshold={_fmt(T) if T is not None else '?'}, k={self.per_cell})",
                  "## Variables worth remembering", "work(threshold, k), handle(tid, threshold, n0), handle_reply(tid, outcome, threshold), OBS, INBOX"]
        return "\n".join(lines)


BACKLOG_VARIANTS: dict[str, dict[str, Any]] = {
    "faithful": {"summary": "faithful", "reread": True},
    "forgetful": {"summary": "forgetful", "reread": False},
    "lazy": {"summary": "faithful", "reread": False},
    "forgetful_reread": {"summary": "forgetful", "reread": True},
}


def backlog_model(variant: str = "faithful", **kw: Any) -> BacklogRLM:
    if variant not in BACKLOG_VARIANTS:
        raise ValueError(f"variant: {' | '.join(BACKLOG_VARIANTS)} (got {variant!r})")
    return BacklogRLM(variant=variant, **{**BACKLOG_VARIANTS[variant], **kw})


# ---------------------------------------------------------------------------------------------------------------------------
# Fault injection for the M16 campaign (design §4): each helper is a function or wrapper the campaign calls; each records
# that it fired (``fired``), so a run can prove the fault happened before it judges the harness's reaction.
# ---------------------------------------------------------------------------------------------------------------------------


class FaultLLM:
    """Wraps an LLM client. ``overflow_at``: call indices (0-based, every call the client serves) that raise
    ``ContextOverflow``; ``overflow_when(view)``: a predicate on the call (``CallView``: role root | summary | sub) that
    overflows, at most ``overflow_times`` times (e.g. every compaction attempt: ``lambda v: v.role == "summary"``);
    ``silent``: the overflow is a silent one (an empty reply with ``finish="length"`` and the window's prompt size, which
    some providers return instead of an error); ``crash_at``: a call index that raises ``exc`` (a lost worker)."""

    def __init__(self, inner: Any, *, overflow_at: tuple[int, ...] = (), overflow_when: Any = None, overflow_times: int | None = None,
                 silent: bool = False, window: int = 32_000, crash_at: int | None = None, exc: BaseException | None = None):
        self.inner, self.overflow_at, self.overflow_when, self.overflow_times = inner, set(overflow_at), overflow_when, overflow_times
        self.silent, self.window, self.crash_at = silent, window, crash_at
        self.exc = exc if exc is not None else ConnectionError("worker lost (injected)")
        self.spend = getattr(inner, "spend", None)
        self.calls = 0
        self.fired: list[dict[str, Any]] = []

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        from crucible.actors.llm import ContextOverflow

        i = self.calls
        self.calls += 1
        view = CallView.of(messages, kw.get("tools"))
        if self.crash_at is not None and i == self.crash_at:
            self.fired.append({"call": i, "fault": "crash", "role": view.role})
            raise self.exc
        hit = i in self.overflow_at or (self.overflow_when is not None and self.overflow_when(view)
                                         and (self.overflow_times is None or sum(f["fault"] == "overflow" for f in self.fired) < self.overflow_times))
        if hit:
            self.fired.append({"call": i, "fault": "overflow", "role": view.role, "silent": self.silent})
            if self.silent:
                return LLMResponse(text="", usage={"prompt_tokens": self.window, "completion_tokens": 0}, raw_finish="length", model="fault")
            raise ContextOverflow(f"prompt exceeds the {self.window}-token context window (injected at call {i})")
        return await self.inner.complete(messages, **kw)


def crash_then_resume_ref(make_llm: Any, *, crash_at_call: int, policy_id: str = "scripted:crash-resume",
                          on_attempts: tuple[int, ...] = (1,)) -> tuple[Any, list[FaultLLM]]:
    """(a ``PolicyRef`` whose attempt ``on_attempts`` loses its worker at model call ``crash_at_call`` (an infra fault, masked);
    with the harness's ``checkpoint_every`` the runner resumes the next attempt from the last checkpoint
    (``runtime.checkpoint``). ``make_llm(attempt)`` builds a fresh client per attempt (the model is stateless, so the resumed
    attempt continues from its restored context), the wrappers that fired)."""
    from crucible.runtime.harness.native import PolicyRef

    made: list[FaultLLM] = []

    def make(attempt: Any) -> dict[str, Any]:
        llm = make_llm(attempt)
        if getattr(attempt, "n", 1) in on_attempts:
            llm = FaultLLM(llm, crash_at=crash_at_call)
            made.append(llm)
        return {"agent": llm}

    return PolicyRef(id=policy_id, make=make), made


async def kill_kernel(kernel: Any) -> bool:
    """SIGKILL the kernel's process group inside its sandbox (Docker or host), leaving the client side to find it dead: what
    an OOM kill or a crashed interpreter looks like to the harness. Returns whether the launcher confirmed the kill."""
    launcher = getattr(kernel, "launcher", None)
    if kernel is None or getattr(kernel, "proc", None) is None or launcher is None or not hasattr(launcher, "kill"):
        return False
    return bool(await launcher.kill(kernel.proc, kernel.pgid))


def kill_mid_cell(policy: Any, *, at_world_call: int) -> dict[str, Any]:
    """Kill the policy's kernel while a cell is running: on the ``at_world_call``-th World call a cell makes (counted from
    now), the kernel is killed before the host answers, so the cell dies mid-flight. Works on any policy that serves
    kernel host requests through ``_host(data)`` (``RLMPolicy`` today). Returns the record the fault fills in."""
    rec: dict[str, Any] = {"at_world_call": at_world_call, "fired": False, "seen": 0}
    inner = policy._host

    async def host(data: dict[str, Any]) -> Any:
        if data.get("type") == "world.call" and not rec["fired"]:
            rec["seen"] += 1
            if rec["seen"] >= at_world_call:
                rec["fired"] = True
                rec["killed"] = await kill_kernel(policy.kernel)
        return await inner(data)

    policy._host = host
    return rec


# a catastrophic-backtracking pattern and an input that makes it run for ages (nested quantifiers, a failing suffix)
PATHOLOGICAL_REGEX = r"(a+)+$"
PATHOLOGICAL_INPUT = "a" * 40 + "!"
PATHOLOGICAL_CELL = f"import re\nprint(bool(re.search({PATHOLOGICAL_REGEX!r}, {PATHOLOGICAL_INPUT!r})))"
PATHOLOGICAL_HISTORY_CELL = f"r = await history.search({PATHOLOGICAL_REGEX!r}, regex=True)\nprint(len(r))"
BIG_OUTPUT_BYTES = 50_000_000
BIG_OUTPUT_CELL = f"print('x' * {BIG_OUTPUT_BYTES})"


def pathological_ledger_search(ledger: Any, *, budget_s: float = 2.0) -> dict[str, Any]:
    """Plant an event the pathological regex backtracks on, search the ledger with it, and time the search (the ledger must
    bound it: ≤ 2 s per call, design §3.1). Returns {"s", "bounded", "result" | "error"}."""
    import time

    ledger.append("tool", text=PATHOLOGICAL_INPUT * 200)
    t0 = time.perf_counter()
    try:
        res: Any = ledger.search(PATHOLOGICAL_REGEX, regex=True)
        err = None
    except Exception as e:  # noqa: BLE001 -- a refusal or a timeout error is a bounded outcome
        res, err = None, f"{type(e).__name__}: {e}"
    s = time.perf_counter() - t0
    return {"s": round(s, 3), "bounded": s <= budget_s + 1.0, "result": res, "error": err}
