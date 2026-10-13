"""Online hack interception at the World's tool choke point (m13 C-4; article §3.7.3.5, GLM-5.2 A2-28; checklist #41).

Before a policy's tool call executes, ``World._call_tool`` asks its interceptors. A hit is never executed: the policy
gets a refusal (or, ``action="dummy"``, a neutral placeholder result) and the episode continues, so it stays trainable
and the shortcut simply does not pay. Each hit is an ``intercept`` event in the trace (rule, category, tool), which
becomes a rollout flag (``hack/intercepted``) for the two-level detector (``credit/hack_detector.py``).

Rules are defensive categories, matched against any string argument of the call (a shell command, a path, a URL):

- ``remote_code_fetch``: cloning or downloading code / answers from the network. Qwen's rule (article Q:L808, M12 audit
  G1 H7): **a call that carries a network link and a network keyword anywhere in its strings** is intercepted. The
  keywords are git's network operations (``clone``, ``fetch``, ``pull``, ``ls-remote``, ``remote add|set-url``,
  ``submodule add``, ``config remote.<name>.url``), ``curl``, ``wget``, a ``git+<scheme>://`` requirement and a git
  config ``[remote "<name>"]`` section. So the git back-link (re-add a remote, then fetch it:
  ``git remote add origin https://github.com/o/r && git fetch origin``, ``git remote add up …; git pull up main``) is
  refused, and so is the split form, since the ``remote add`` call already carries the link and the keyword. Hosts in
  ``allow_hosts`` (the world's own allowlist) and loopback hosts (the world's own services) pass. Local git use is never
  touched: no link (``git remote add backup ../mirror && git fetch backup``, ``git pull . feature``) or no network
  keyword (``git commit -m "fixes https://…"``). The egress guard stays the network-level backstop (a remote
  configured before the episode and fetched by name carries no link);
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
    url_hosts: bool = False  # the pattern extracts a URL / remote: hosts in allow_hosts (and loopback hosts) pass
    # Qwen's rule: when set, ``pattern`` extracts links from all of the call's strings joined, and a link counts only
    # when this keyword pattern matches somewhere in the same call
    requires: str | None = None


# a network link: a URL with a network scheme, an scp-style git address, or a bare ``host.tld/path`` (M12 audit G1 H7)
_LINK_END = r"[^\s'\"<>;|&)`]+"
LINK = (rf"(?P<url>(?:https?|ssh|git|ftp|rsync)://{_LINK_END}|\b[\w.-]+@[\w.-]+\.[a-z]{{2,}}:{_LINK_END}"
        rf"|\b(?:[\w-]+\.)+(?:com|org|io|net|dev|ai|co|me|xyz|app|cn|uk|de)/{_LINK_END})")
# a network keyword (Qwen: git / curl / wget), git's network subcommands after any global options
NETWORK_KEYWORD = (r"\bgit\b(?:\s+(?:-[Cc]\s+\S+|--?[\w.-]+(?:=\S+)?))*\s+(?:clone|fetch|pull|ls-remote|remote\s+(?:add|set-url)"
                   r"|submodule\s+add|config\b[^\n;&|]*?\bremote\.[^\s.]+\.(?:push)?url)\b"
                   r"|\b(?:curl|wget)\b|\bgit\+(?:https?|ssh|git)://|\[remote\s+\"[^\"]*\"\]|\bgh\s+repo\s+clone\b")
REMOTE_FETCH = Rule(
    "remote_code_fetch", "remote_code_fetch", LINK,
    message="Refused by the environment: fetching code or answers from the network is not allowed in this task. Work with the files you have.",
    url_hosts=True, requires=NETWORK_KEYWORD)
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
    if "://" not in url and re.match(r"^[\w.-]+@[\w.-]+:", url):  # scp-style git address: user@host:path
        return url.split("@", 1)[1].split(":", 1)[0].lower()
    if "://" not in url:
        url = "https://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:  # a malformed bracketed host: not provably local
        return ""


def _loopback(host: str) -> bool:
    """The world's own services (a local API, a test server) are not the network."""
    return host in ("localhost", "::1", "0.0.0.0", "ip6-localhost") or host.endswith(".localhost") or host.startswith("127.")


@dataclass
class RuleInterceptor:
    rules: tuple[Rule, ...] | list[Rule] = DEFAULT_RULES
    allow_hosts: tuple[str, ...] | list[str] = ()
    _compiled: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = [(r, re.compile(r.pattern, re.IGNORECASE), re.compile(r.requires, re.IGNORECASE) if r.requires else None)
                          for r in self.rules]

    def _allowed(self, host: str) -> bool:
        return any(host == h or (h.startswith("*.") and host.endswith(h[1:])) for h in (x.lower() for x in self.allow_hosts))

    def check(self, actor: str, tool: str, arguments: dict[str, Any]) -> Interception | None:
        import fnmatch

        texts = _strings(arguments)
        for rule, rx, req in self._compiled:
            if not fnmatch.fnmatchcase(tool, rule.tools):
                continue
            scan = texts
            if req is not None:  # Qwen's rule: link + keyword anywhere in the call (all of its strings)
                joined = "\n".join(texts)
                if not req.search(joined):
                    continue
                scan = [joined]
            for t in scan:
                for m in rx.finditer(t):
                    if rule.url_hosts:
                        url = next((g for g in m.groups() if g), "")
                        host = _host(url) if url else ""
                        if url and (self._allowed(host) or _loopback(host)):
                            continue
                    return Interception(rule.id, rule.category, rule.action, rule.message, t[max(0, m.start() - 40): m.end() + 40][:300])
        return None
