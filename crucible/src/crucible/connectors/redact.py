"""Redaction inside the tenant (DESIGN §5.1): runs on every trace before it is stored.

Removes credentials (API keys and tokens in the formats providers issue, private keys, JWTs), replaces e-mail
addresses and internal host names / private IPs with stable pseudonyms (the same input always maps to the same
token, so threads stay linkable), pseudonymizes actor handles, and drops any extra terms the policy names. The
gate is canary-based: every planted canary must be gone (``tests/unit/test_connectors.py``).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.trace import Trace

SECRET_PATTERNS = [
    r"sk-(?:or-v1-|ant-|proj-)?[A-Za-z0-9_\-]{20,}",  # OpenAI / OpenRouter / Anthropic
    r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{40,}",  # GitHub
    r"hf_[A-Za-z0-9]{30,}",  # Hugging Face
    r"(?:AKIA|ASIA)[0-9A-Z]{16}",  # AWS access key id
    r"xox[abposr]-[A-Za-z0-9-]{10,}",  # Slack
    r"glpat-[A-Za-z0-9_\-]{20}",  # GitLab
    r"AIza[0-9A-Za-z_\-]{35}",  # Google API key
    r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    r"(?i:(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"]?[^\s'\"]{8,})",
]
EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
PRIVATE_IP = r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.\d{1,3}\.\d{1,3}\b"
INTERNAL_HOST = r"\b[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)*\.(?:internal|corp|local|lan|intra|ts\.net)\b"


class RedactionPolicy(CrucibleModel):
    secrets: bool = True
    emails: bool = True
    internal_hosts: bool = True
    pseudonymize_actors: bool = True
    terms: list[str] = Field(default_factory=list)  # extra strings to remove (customer names, project code names)
    salt: str = "crucible"  # per-tenant salt for pseudonyms
    keep_emails: list[str] = Field(default_factory=list)  # e.g. a world's synthetic addresses


def pseudonym(kind: str, value: str, salt: str) -> str:
    return f"<{kind}:{hashlib.sha256(f'{salt}|{value}'.encode()).hexdigest()[:10]}>"


class Redactor:
    def __init__(self, policy: RedactionPolicy):
        self.p = policy
        self._secret = re.compile("|".join(f"(?:{s})" for s in SECRET_PATTERNS))
        self._email = re.compile(EMAIL)
        self._host = re.compile(f"{PRIVATE_IP}|{INTERNAL_HOST}", re.IGNORECASE)
        self._terms = re.compile("|".join(re.escape(t) for t in sorted(policy.terms, key=len, reverse=True)), re.IGNORECASE) if policy.terms else None
        self.hits: dict[str, int] = {}

    def _count(self, k: str) -> None:
        self.hits[k] = self.hits.get(k, 0) + 1

    def text(self, s: str) -> str:
        if self.p.secrets:
            s = self._secret.sub(lambda m: (self._count("secret"), "<secret>")[1], s)
        if self.p.emails:
            s = self._email.sub(lambda m: m.group(0) if m.group(0) in self.p.keep_emails else (self._count("email"), pseudonym("email", m.group(0).lower(), self.p.salt))[1], s)
        if self.p.internal_hosts:
            s = self._host.sub(lambda m: (self._count("host"), pseudonym("host", m.group(0).lower(), self.p.salt))[1], s)
        if self._terms is not None:
            s = self._terms.sub(lambda m: (self._count("term"), "<redacted>")[1], s)
        return s

    def value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, dict):
            return {k: self.value(x) for k, x in v.items()}
        return v


def redact(trace: Trace, policy: RedactionPolicy | None = None) -> Trace:
    r = Redactor(policy or RedactionPolicy())
    d = trace.model_dump(mode="json", by_alias=True)
    for e in d["events"]:
        if e.get("text"):
            e["text"] = r.text(e["text"])
        e["data"] = r.value(e.get("data") or {})
    for s in d.get("signals", []):
        if s.get("text"):
            s["text"] = r.text(s["text"])
        s["meta"] = r.value(s.get("meta") or {})
    if r.p.pseudonymize_actors:
        for a in d["actors"]:
            if a.get("handle"):
                a["handle"] = pseudonym("user", a["handle"], r.p.salt)
    d["outcome"]["details"] = r.value(d["outcome"].get("details") or {})
    out = Trace.model_validate(d)
    out.source.ref = r.text(out.source.ref)
    return out
