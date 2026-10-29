"""Scripted stand-ins for model clients (``synth.roles.RoleClient``) in the M4 tests: no model is called."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


class FakeClient:
    """``RoleClient`` whose reply is ``script(messages, n)`` (``n`` = this client's call count, from 0)."""

    def __init__(self, family: str, script: Callable[[list[dict[str, Any]], int], dict[str, Any]], model: str = "fake/model"):
        self.family, self.model, self.script = family, model, script
        self.calls: list[list[dict[str, Any]]] = []
        self.tags: list[str] = []

    async def complete_json(self, messages: list[dict[str, Any]], *, schema: Any = None, max_tokens: int = 2048, temperature: float = 0.0,
                            tag: str = "") -> dict[str, Any]:
        n = len(self.calls)
        self.calls.append(messages)
        self.tags.append(tag)
        return self.script(messages, n)


def user_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


def section(messages: list[dict[str, Any]], name: str) -> Any:
    """The ``## name`` section of the (first) user message, JSON-decoded when it is JSON."""
    t = messages[1]["content"]
    head = f"## {name}\n"
    i = t.find(head)
    if i < 0:
        return None
    body = t[i + len(head):]
    j = body.find("\n\n## ")
    body = body[:j] if j >= 0 else body.split("\n\nReply with one JSON object only.")[0]
    try:
        return json.loads(body)
    except ValueError:
        return body


def const(doc: dict[str, Any]) -> Callable[[list[dict[str, Any]], int], dict[str, Any]]:
    return lambda _m, _n: doc


def seq(*docs: dict[str, Any]) -> Callable[[list[dict[str, Any]], int], dict[str, Any]]:
    return lambda _m, n: docs[min(n, len(docs) - 1)]
