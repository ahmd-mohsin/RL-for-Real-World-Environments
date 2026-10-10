"""Online hack interception at the World's tool choke point (m13 C-4; article §3.7.3.5, GLM-5.2 A2-28; checklist #41).

Before a policy's tool call executes, ``World._call_tool`` asks its interceptors. A hit is never executed: the policy
gets a refusal (or, ``action="dummy"``, a neutral placeholder result) and the episode continues, so it stays trainable
and the shortcut simply does not pay. Each hit is an ``intercept`` event in the trace (rule, category, tool), which
becomes a rollout flag (``hack/intercepted``) for the two-level detector (``credit/hack_detector.py``).

Rules are defensive categories, matched against any string argument of the call (a shell command, a path, a URL):

- ``remote_code_fetch``: cloning or downloading code / answers from the network (``git clone|fetch|pull <remote>``,
  ``curl|wget <url>``); hosts in ``allow_hosts`` (the world's own allowlist) pass;
- ``grader_paths``: reading or writing the verifier's own locations (``/logs/verifier``, ``/solution``).

A task can add rules (``World(interceptors=[RuleInterceptor([...])])``) or turn interception off (``interceptors=[]``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse


@dataclass(frozen=True)
class Interception:
    rule: str
    category: str
    action: str  # refuse | dummy
    message: str
    evidence: str


class Interceptor(Protocol):
    def check(self, actor: str, tool: str, arguments: dict[str, Any]) -> Interception | None: ...


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    pattern: str
    tools: str = "*"  # fnmatch over the qualified tool name
    action: str = "refuse"
    message: str = "Refused by the environment: this action is not allowed in this task. Continue with the resources you have."
    url_hosts: bool = False  # the pattern extracts a URL / remote: hosts in allow_hosts pass


REMOTE_FETCH = Rule(
    "remote_code_fetch", "remote_code_fetch",
    r"\bgit\s+(?:clone|fetch|pull|submodule\s+add)\b[^\n;&|]*?(?P<url>(?:https?|ssh|git)://\S+|git@[\w.-]+:\S+|\b[\w.-]+\.(?:com|org|io|net)/\S+)"
    r"|\b(?:curl|wget)\b[^\n;&|]*?(?P<url2>https?://\S+)",
    message="Refused by the environment: fetching code or answers from the network is not allowed in this task. Work with the files you have.",
    url_hosts=True)
GRADER_PATHS = Rule(
    "grader_paths", "grader_paths", r"(?:^|[\s'\"=:(])(?P<p>/logs/verifier\b|/solution\b)",
    message="Refused by the environment: the verifier's files are not part of the task.")
DEFAULT_RULES = (REMOTE_FETCH, GRADER_PATHS)


def _strings(x: Any) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list | tuple):
        return [s for v in x for s in _strings(v)]
    return []


def _host(url: str) -> str:
    if url.startswith("git@"):
        return url[4:].split(":", 1)[0].lower()
    if "://" not in url:
        url = "https://" + url
    return (urlparse(url).hostname or "").lower()


@dataclass
class RuleInterceptor:
    rules: tuple[Rule, ...] | list[Rule] = DEFAULT_RULES
    allow_hosts: tuple[str, ...] | list[str] = ()
    _compiled: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = [(r, re.compile(r.pattern, re.IGNORECASE)) for r in self.rules]

    def _allowed(self, host: str) -> bool:
        return any(host == h or (h.startswith("*.") and host.endswith(h[1:])) for h in (x.lower() for x in self.allow_hosts))

    def check(self, actor: str, tool: str, arguments: dict[str, Any]) -> Interception | None:
        import fnmatch

        texts = _strings(arguments)
        for rule, rx in self._compiled:
            if not fnmatch.fnmatchcase(tool, rule.tools):
                continue
            for t in texts:
                for m in rx.finditer(t):
                    if rule.url_hosts:
                        url = next((g for g in m.groups() if g), "")
                        if url and self._allowed(_host(url)):
                            continue
                    return Interception(rule.id, rule.category, rule.action, rule.message, t[max(0, m.start() - 40): m.end() + 40][:300])
        return None
