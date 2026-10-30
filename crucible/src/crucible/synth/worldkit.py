"""The trusted kit every World family gets on top of its swarm-written tools (part2-paper-core §11; evidence doc
principle 18: everything reward-bearing is in ``canon()``).

Real enterprise work needs facts that are not in the transactional records: the written policy and SOPs, and the
colleagues who know things nobody wrote down. The kit provides both as framework code (Claude builds framework; the
*content* is the swarm's and differs per instance):

| kit | tables (in the App, so in ``canon()``) | tools (read-only) |
|---|---|---|
| ``docs`` | ``documents`` (doc_id, title, kind, tags, body, version, effective_date) | ``list_documents(query, kind, page, page_size)``, ``read_document(doc_id)``, ``search_documents(query)`` |
| ``policy`` | ``policy_params`` (name, value, unit, description) | none: rules reach the agent through documents only, as in a real company |
| ``actors`` | ``colleagues`` (colleague_id, name, title, topics, facts) | ``list_colleagues()``, ``ask_colleague(colleague_id, question)`` |

``policy_params`` is what the family's own App code reads (``ctx.tables["policy_params"][name]["value"]``), and what an
instance's policy variant changes; the instance's documents are rendered from the family's templates with the same values,
so the documents always agree with the behaviour (the realism gate re-checks it by execution). A colleague answers
deterministically: the facts whose keywords appear in the question, else what they can help with.

This module is imported by the sandbox child (``_sandbox_child.py``) to register the kit into a synthesized App, so it
imports only ``iface``.
"""

from __future__ import annotations

import re
from typing import Any

from crucible.synth.iface import App, ToolError

KIT_TABLES: dict[str, dict[str, Any]] = {
    "documents": {"key": "doc_id", "fields": {"doc_id": "string", "title": "string", "kind": "string", "tags": "array", "body": "string",
                                              "version": "string", "effective_date": "string"},
                  "description": "The company's policy, SOP, FAQ and data-standards documents (kit)."},
    "policy_params": {"key": "name", "fields": {"name": "string", "value": "any", "unit": "string?", "description": "string"},
                      "description": "The policy parameters the App's rules read (kit); documents state them in words."},
    "colleagues": {"key": "colleague_id", "fields": {"colleague_id": "string", "name": "string", "title": "string", "topics": "array",
                                                     "facts": "array"},
                   "description": "Simulated colleagues and the facts each one holds (kit)."},
}
KIT_TOOLS: dict[str, tuple[str, ...]] = {"docs": ("list_documents", "read_document", "search_documents"), "actors": ("list_colleagues", "ask_colleague")}
KIT_KINDS = ("docs", "policy", "actors")
KIT_NAMES = frozenset(n for v in KIT_TOOLS.values() for n in v)


def kit_tables(kinds: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    want = {"docs": ["documents"], "policy": ["policy_params"], "actors": ["colleagues"]}
    return {t: KIT_TABLES[t] for k in kinds for t in want.get(k, [])}


def kit_tool_names(kinds: list[str] | tuple[str, ...]) -> set[str]:
    return {n for k in kinds for n in KIT_TOOLS.get(k, ())}


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 2}


def _page(items: list[Any], page: int, page_size: int) -> dict[str, Any]:
    if page < 1:
        raise ToolError("INVALID_ARGUMENT: page starts at 1")
    if not 1 <= page_size <= 50:
        raise ToolError("INVALID_ARGUMENT: page_size must be 1 to 50")
    lo = (page - 1) * page_size
    chunk = items[lo:lo + page_size]
    return {"items": chunk, "page": page, "page_size": page_size, "total": len(items), "next_page": page + 1 if lo + page_size < len(items) else None}


def register(app: App, kinds: list[str] | tuple[str, ...]) -> None:
    """Register the kit tools of ``kinds`` into ``app`` (its tables come from the instance's start state)."""
    if "docs" in kinds:
        @app.tool(read_only=True)
        def list_documents(ctx: Any, query: str = "", kind: str = "", page: int = 1, page_size: int = 10) -> dict:
            """List the company's documents (policies, SOPs, FAQs, data standards), newest version of each, optionally
            filtered by words in the title or tags and by kind (policy, sop, faq, data_standards). Paginated."""
            docs = sorted(ctx.tables.get("documents", {}).values(), key=lambda d: str(d.get("doc_id")))
            q = _words(query)
            out = []
            for d in docs:
                if kind and d.get("kind") != kind:
                    continue
                if q and not q & (_words(str(d.get("title", ""))) | _words(" ".join(map(str, d.get("tags") or [])))):
                    continue
                out.append({"doc_id": d.get("doc_id"), "title": d.get("title"), "kind": d.get("kind"), "tags": d.get("tags") or [],
                            "version": d.get("version"), "effective_date": d.get("effective_date")})
            return _page(out, page, page_size)

        @app.tool(read_only=True)
        def read_document(ctx: Any, doc_id: str) -> dict:
            """The full text of one document."""
            d = ctx.tables.get("documents", {}).get(doc_id)
            if d is None:
                raise ToolError(f"NOT_FOUND: no document {doc_id}")
            return dict(d)

        @app.tool(read_only=True)
        def search_documents(ctx: Any, query: str) -> dict:
            """Full-text search over every document body: the documents whose text contains every word of the query, with
            the matching lines."""
            q = [w for w in re.findall(r"[a-z0-9.%$]+", (query or "").lower()) if len(w) > 1]
            if not q:
                raise ToolError("INVALID_ARGUMENT: the query has no words")
            hits = []
            for d in sorted(ctx.tables.get("documents", {}).values(), key=lambda d: str(d.get("doc_id"))):
                body = str(d.get("body", ""))
                if all(w in body.lower() or w in str(d.get("title", "")).lower() for w in q):
                    lines = [ln.strip() for ln in body.splitlines() if any(w in ln.lower() for w in q)][:5]
                    hits.append({"doc_id": d.get("doc_id"), "title": d.get("title"), "lines": lines})
            return {"results": hits[:20], "total": len(hits)}

    if "actors" in kinds:
        @app.tool(read_only=True)
        def list_colleagues(ctx: Any) -> dict:
            """The colleagues you can ask, with their title and the topics they can help with."""
            cs = sorted(ctx.tables.get("colleagues", {}).values(), key=lambda c: str(c.get("colleague_id")))
            return {"colleagues": [{"colleague_id": c.get("colleague_id"), "name": c.get("name"), "title": c.get("title"),
                                    "can_help_with": c.get("topics") or []} for c in cs]}

        @app.tool(read_only=True)
        def ask_colleague(ctx: Any, colleague_id: str, question: str) -> dict:
            """Ask a colleague a question in plain words; they answer what they know about it."""
            c = ctx.tables.get("colleagues", {}).get(colleague_id)
            if c is None:
                raise ToolError(f"NOT_FOUND: no colleague {colleague_id}")
            q = _words(question)
            if not q:
                raise ToolError("INVALID_ARGUMENT: ask a question")
            said = [f.get("text") for f in (c.get("facts") or []) if q & {w.lower() for k in (f.get("keywords") or []) for w in _words(str(k))}]
            if not said:
                return {"from": c.get("name"), "reply": f"Sorry, I don't know about that. I can help with: {', '.join(c.get('topics') or [])}."}
            return {"from": c.get("name"), "reply": " ".join(str(s) for s in said)}


# ---------------------------------------------------------------------------------------------------------- rendering

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def fmt_value(v: Any, unit: str | None = None) -> str:
    """A parameter value as a document states it (the realism gate looks for exactly this text)."""
    if isinstance(v, bool):
        s = "yes" if v else "no"
    elif isinstance(v, float) and v.is_integer():
        s = f"{int(v):,}"
    elif isinstance(v, int):
        s = f"{v:,}"
    elif isinstance(v, float):
        s = f"{v:,.2f}"
    elif isinstance(v, list):
        s = ", ".join(str(x) for x in v)
    else:
        s = str(v)
    return s


def render(template: str, params: dict[str, Any], units: dict[str, str | None] | None = None) -> str:
    """Fill ``{{name}}`` placeholders; an unknown name raises ``KeyError`` (a template/param mismatch)."""
    def sub(m: re.Match[str]) -> str:
        n = m.group(1)
        if n not in params:
            raise KeyError(n)
        return fmt_value(params[n], (units or {}).get(n))

    return PLACEHOLDER.sub(sub, template)


def placeholders(template: str) -> set[str]:
    return set(PLACEHOLDER.findall(template or ""))


KIT_API_DOC = """PLATFORM (provided by the framework; do not implement or test these)
- Table policy_params: rows keyed by parameter name, {"name", "value", "unit", "description"}. Business rules that depend on
  a policy parameter MUST read it: p = ctx.tables["policy_params"].get("<name>"); value = p["value"] if p else <the default
  in design.policy_params>. Never hard-code a policy value that is a parameter.
- Tables documents and colleagues and their tools (list_documents, read_document, search_documents, list_colleagues,
  ask_colleague) are provided. Do not define tools with these names."""


__all__ = ["KIT_API_DOC", "KIT_KINDS", "KIT_NAMES", "KIT_TABLES", "KIT_TOOLS", "fmt_value", "kit_tables", "kit_tool_names", "placeholders", "register", "render"]
