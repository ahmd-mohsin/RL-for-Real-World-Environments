"""Redaction inside the tenant (DESIGN §5.1): runs on every trace before it is stored.

Removes credentials (API keys and tokens in the formats providers issue, private keys, JWTs), replaces e-mail
addresses and internal host names / private IPs with stable pseudonyms (the same input always maps to the same
token, so threads stay linkable), pseudonymizes actor handles, and drops any extra terms the policy names. The
gate is canary-based: every planted canary must be gone (``tests/unit/test_connectors.py``).

M3 audit (2026-10-12): every string field of the trace is redacted (ids, refs, targets, URLs and dict keys kept
e-mails and passwords); credentials in URLs, short passwords, Azure keys, card numbers and phone numbers are removed;
prefix patterns need a word boundary ("task-04-gpu-migration" lost its tail to the ``sk-`` rule); ``threading.local()``
and ``icon@2x.png`` are no longer a host and an e-mail; pseudonyms are an HMAC under a per-tenant secret (the public
default salt "crucible" let anyone reverse a guessed address); redacting twice changes nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets as _secrets
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.trace import Trace
from crucible.core.secrets import _B, _CODE_VALUE, _KEYWORD, ASSIGNMENT, PRIVATE_KEY, SECRET_PATTERNS, URL_USERINFO  # noqa: F401 -- one list

_TRAILER = re.compile(r"(?im)^(\s*(?:Signed-off-by|Co-authored-by|Reviewed-by|Acked-by|Tested-by|Reported-by|Suggested-by)\s*:\s*)([^<\r\n]+?)(?=\s*<|\s*$)")
IBAN = r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"
SSN = r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
# data keys whose string value is a person (git authors, ticket assignees): pseudonymized like an actor handle
IDENTITY_KEYS = {"author", "committer", "git_author", "git_committer", "author_name", "committer_name", "assignee", "reporter", "username",
                 "user_name", "login", "reviewer", "approver", "requester", "owner_name"}
_SECRET_KEY_PARTS = ("password", "passwd", "passphrase", "secret", "credential", "cookie", "authorization", "bearer", "api_key", "apikey",
                     "api-key", "access_key", "accesskey", "private_key", "privatekey", "session_id", "sessionid", "auth_token")


def secret_key(key: object) -> bool:
    """A dict key whose value is a credential whatever it looks like (NeMo-Gym ``secret_utils`` idea, narrowed for
    traces: ``max_tokens``, ``token_count``, ``sort_key`` and ``author`` are data, not secrets)."""
    k = str(key).lower()
    if k.startswith("<") and k.endswith(">"):  # a placeholder left by redaction ("<secret>"), not a credential name
        return False
    return (any(x in k for x in _SECRET_KEY_PARTS) or k in ("pwd", "pass", "token", "auth")
            or k.endswith(("_pass", "_pwd", "_token", "-token", "_secret")))
# a private key, also when a connector cut the text before its END line (the body was stored). Removed in its own pass
# first: a greedy rule elsewhere that ate the BEGIN line left the body unrecognizable (found by the gate's long canary)
# anchored at a token start and bounded (an unanchored, unbounded local part made a 200k-character run take 18 s);
# not an e-mail: "icon@2x.png", "x@self.weights" (matrix product), "git@github.com:org/repo" (an ssh remote)
EMAIL = (r"(?<![A-Za-z0-9._%+\-])(?!git@)[A-Za-z0-9._%+\-]{1,64}@(?![0-9]+x\.(?:png|jpe?g|gif|svg|webp)\b)(?!self\.)"
         r"[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){0,8}\.[A-Za-z]{2,24}\b")
_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
PRIVATE_IP = rf"\b(?:10\.{_OCT}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.{_OCT}\.{_OCT}\b"
GENERIC_ACTORS = {"user", "agent", "system", "author", "reviewer", "ci"}
# a host name, not a call or attribute in code: "threading.local()" and "x.internal_" are code
INTERNAL_HOST = r"\b[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)*\.(?:internal|corp|local|lan|intra|ts\.net)\b(?![(\[_])"
CARD = r"\b(?:\d[ \-]?){12,18}\d\b"  # checked with Luhn before it is removed
PHONE = r"(?<![\w+])\+\d{1,3}[\s.\-]?(?!\d{4}-\d{2}-\d{2})(?:\(\d{1,4}\)[\s.\-]?)?\d{2,4}(?:[\s.\-]?\d{2,4}){1,3}\b"


_CARD_IIN = re.compile(r"4|5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720)|3[47]|6(?:011|5|4[4-9])|35(?:2[89]|[3-8])")


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    n = "".join(str(int(c, 36)) for c in s[4:] + s[:4])
    return int(n) % 97 == 1


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total, alt = total + d, not alt
    return total % 10 == 0


def salt_path() -> Path:
    from crucible.core.config import get_config

    return get_config().data_path() / "redaction.salt"


def tenant_salt() -> str:
    """The tenant's pseudonym key: ``CRUCIBLE_REDACTION_SALT``, else a random key created once in the data directory
    (mode 0600). Pseudonyms stay stable within the tenant and cannot be recomputed outside it."""
    from crucible.core import env as knobs

    env = knobs.get("CRUCIBLE_REDACTION_SALT")
    if env:
        return env
    p = salt_path()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(_secrets.token_hex(32))
    return p.read_text().strip()


class RedactionPolicy(CrucibleModel):
    secrets: bool = True
    emails: bool = True
    internal_hosts: bool = True
    pseudonymize_actors: bool = True
    terms: list[str] = Field(default_factory=list)  # extra strings to remove (customer names, project code names)
    # pseudonym key; None = the tenant's (``tenant_salt``). Excluded from serialization like the secret values.
    salt: str | None = Field(default=None, exclude=True, repr=False)
    keep_emails: list[str] = Field(default_factory=list)  # e.g. a world's synthetic addresses
    # Exact secret values the tenant knows it holds (its .env / secret store): removed whatever their format. Never
    # serialized, so a stored policy cannot carry them.
    secret_values: list[str] = Field(default_factory=list, exclude=True, repr=False)

    @classmethod
    def with_env_file(cls, path: str | Path, *, missing_ok: bool = False, **kw: Any) -> RedactionPolicy:
        """A policy that also removes every credential-like value (``*TOKEN*``, ``*KEY*``, ``*SECRET*``, ``*PASSWORD*``,
        ``*PAT*``) defined in a dotenv file. A path that does not exist raises ``FileNotFoundError`` unless ``missing_ok``
        (M14 audit V5: a misspelled path was a silent no-op and the secret it named was stored verbatim)."""
        vals = []
        p = Path(path)
        if not p.is_file() and not missing_ok:
            raise FileNotFoundError(f"redaction env file {p} does not exist (its secret values could not be redacted)")
        for line in p.read_text().splitlines() if p.is_file() else []:
            k, _, v = line.strip().removeprefix("export ").partition("=")
            v = v.strip()
            if v[:1] in "'\"" and v[:1] and v[1:].find(v[0]) >= 0:  # a quoted value ends at its closing quote
                v = v[1:1 + v[1:].find(v[0])]
            else:  # an unquoted value ends before an inline comment (the comment kept the real secret from matching)
                v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
            ku = k.strip().upper()
            if (re.search(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|COOKIE|AUTH(?!OR)|DSN|DATABASE_URL|\bPAT\b|_PAT$|_PASS$|_PWD$", ku)
                    or re.search(r"://[^/@\s]+:[^/@\s]+@", v)) and len(v) >= 8:
                vals.append(v)
        return cls(secret_values=vals, **kw)


def pseudonym(kind: str, value: str, salt: str) -> str:
    return f"<{kind}:{hmac.new(salt.encode(), f'{kind}|{value}'.encode(), hashlib.sha256).hexdigest()[:12]}>"


_PSEUDO = re.compile(r"<(?:user|actor|email|host|phone):[0-9a-f]{10,12}>")


class Redactor:
    def __init__(self, policy: RedactionPolicy):
        self.p = policy
        self.salt = policy.salt or tenant_salt()
        known = [re.escape(v) for v in sorted(set(policy.secret_values), key=len, reverse=True)]
        self._secret = re.compile("|".join([*known, *(f"(?:{s})" for s in SECRET_PATTERNS)]))
        self._userinfo = re.compile(URL_USERINFO)
        self._pem = re.compile(PRIVATE_KEY)
        self._card, self._phone = re.compile(CARD), re.compile(PHONE)
        self._assign = re.compile(ASSIGNMENT)
        self._iban, self._ssn = re.compile(IBAN), re.compile(SSN)
        self.people: dict[str, str] = {}  # lower-cased name / handle of a human in the trace -> its pseudonym (set by redact)
        self._people_rx: re.Pattern | None = None
        self._email = re.compile(EMAIL)
        self._host = re.compile(f"{PRIVATE_IP}|{INTERNAL_HOST}", re.IGNORECASE)
        self._terms = re.compile("|".join(re.escape(t) for t in sorted(policy.terms, key=len, reverse=True)), re.IGNORECASE) if policy.terms else None
        self.hits: dict[str, int] = {}

    def _count(self, k: str) -> None:
        self.hits[k] = self.hits.get(k, 0) + 1

    def _card_sub(self, m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        # issuer prefix + length + Luhn: a millisecond timestamp or a 16-digit id is not a card
        if len(digits) in (15, 16, 19) and _CARD_IIN.match(digits) and _luhn(digits):
            self._count("card")
            return "<card>"
        return m.group(0)

    def _assign_sub(self, m: re.Match) -> str:
        value = m.group("qv") if m.group("qv") is not None else m.group("v")
        if value == "<secret>" or _CODE_VALUE.fullmatch(value):
            return m.group(0)
        self._count("secret")
        q = m.group("q") or ""
        return f"{m.group('key')}{m.group('sep')}{q}<secret>{q}"

    def set_people(self, names: dict[str, str]) -> None:
        """Names and handles of the trace's humans: replaced in every text (git author names, @mentions, sign-offs
        survived the actor pseudonym)."""
        self.people = {k.lower(): v for k, v in names.items() if len(k) >= 3 and not k.startswith("<")}
        alts = "|".join(re.escape(k) for k in sorted(self.people, key=len, reverse=True))
        self._people_rx = re.compile(rf"(?i)(?<![\w@.])@?(?:{alts})(?![\w@]|\.\w)") if alts else None  # not inside an address or path

    def person(self, name: str) -> str:
        return self.people.get(name.lower()) or pseudonym("user", name.strip().lower(), self.salt)

    def text(self, s: str) -> str:
        if self.p.secrets:
            s = self._pem.sub(lambda m: (self._count("secret"), "<secret>")[1], s)
            s = self._userinfo.sub(lambda m: (self._count("secret"), "<secret>@")[1], s)
            s = self._secret.sub(lambda m: (self._count("secret"), "<secret>")[1], s)
            s = self._assign.sub(self._assign_sub, s)
            s = self._card.sub(self._card_sub, s)
            s = self._iban.sub(lambda m: (self._count("iban"), "<iban>")[1] if _iban_ok(m.group(0)) else m.group(0), s)
            s = self._ssn.sub(lambda m: (self._count("ssn"), "<ssn>")[1], s)
        if self.p.emails:
            s = self._email.sub(lambda m: m.group(0) if m.group(0) in self.p.keep_emails else (self._count("email"), pseudonym("email", m.group(0).lower(), self.salt))[1], s)
            s = self._phone.sub(lambda m: (self._count("phone"), pseudonym("phone", re.sub(r"\D", "", m.group(0)), self.salt))[1], s)
        if self.p.internal_hosts:
            s = self._host.sub(lambda m: (self._count("host"), pseudonym("host", m.group(0).lower(), self.salt))[1], s)
        if self.p.pseudonymize_actors:
            s = _TRAILER.sub(lambda m: m.group(1) + self.person(m.group(2)), s)
            if self._people_rx is not None:
                s = self._people_rx.sub(lambda m: (self._count("person"), self.people[m.group(0).lstrip("@").lower()])[1], s)
        if self._terms is not None:
            s = self._terms.sub(lambda m: (self._count("term"), "<redacted>")[1], s)
        return s

    def value(self, v: Any, *, secret: bool = False, identity: bool = False) -> Any:
        """``secret``: under a credential-named key, every string is removed whatever it looks like (``{"password":
        "hunter2hunter2"}`` and ``{"Cookie": ...}`` were kept). ``identity``: under ``author`` / ``assignee``-like keys,
        a string is a person."""
        if isinstance(v, str):
            if secret and self.p.secrets and v and not _PSEUDO.fullmatch(v) and v != "<secret>" and not v.isdigit():
                self._count("secret")
                return "<secret>"
            if identity and self.p.pseudonymize_actors and v and not v.startswith("<"):
                return self.person(v)
            return self.text(v)
        if isinstance(v, list):
            return [self.value(x, secret=secret, identity=identity) for x in v]
        if isinstance(v, dict):  # keys too: a dict keyed by e-mail or by a token kept them
            return {self.text(k) if isinstance(k, str) else k: self.value(x, secret=secret or secret_key(k), identity=str(k).lower() in IDENTITY_KEYS)
                    for k, x in v.items()}
        return v


def _target(r: Redactor, t: dict | None) -> dict | None:
    if t:
        t = {**t, **{k: r.text(t[k]) for k in ("app", "object") if isinstance(t.get(k), str)}}
    return t


def redact(trace: Trace, policy: RedactionPolicy | None = None) -> Trace:
    """Every string in the trace except structural names (kinds, types, roles, enum values) goes through the
    redactor: ids, refs, URLs, targets and dict keys as well as texts and data."""
    r = Redactor(policy or RedactionPolicy())
    d = trace.model_dump(mode="json", by_alias=True)
    if r.p.pseudonymize_actors:
        people = {}
        for a in d["actors"]:
            if a["kind"] == "human":
                pid = a["id"] if a["id"].startswith("<") else pseudonym("actor", a["id"].lower(), r.salt)
                for name in (a["id"], a.get("handle")):
                    if name and name not in GENERIC_ACTORS and not name.startswith("<"):
                        people[name] = pid
        r.set_people(people)
    d["trace_id"] = r.text(d["trace_id"])
    src = d["source"]
    for k in ("ref", "url"):
        if isinstance(src.get(k), str):
            src[k] = r.text(src[k])
    for e in d["events"]:
        if e.get("text"):
            e["text"] = r.text(e["text"])
        for k in ("state_ref", "diff_ref", "atif_ref"):
            if isinstance(e.get(k), str):
                e[k] = r.text(e[k])
        e["target"] = _target(r, e.get("target"))
        e["data"] = r.value(e.get("data") or {})
    for s in d.get("signals", []):
        if s.get("text"):
            s["text"] = r.text(s["text"])
        s["target"] = _target(r, s.get("target"))
        s["meta"] = r.value(s.get("meta") or {})
    if r.p.pseudonymize_actors:
        # A human's id is as identifying as their handle (file_drop uses the exported name / e-mail as the id), so
        # both become salted pseudonyms; every reference to the id is rewritten. Role placeholders stay readable.
        ids: dict[str, str] = {}
        for a in d["actors"]:
            # humans only (an agent's handle is its model id); case-folded so "Alice" and "alice" stay one person;
            # redacting twice changes nothing
            if a["kind"] == "human" and a.get("handle") and not _PSEUDO.fullmatch(a["handle"]):
                a["handle"] = pseudonym("user", a["handle"].lower(), r.salt)
            if a["kind"] == "human" and a["id"] not in GENERIC_ACTORS and not a["id"].startswith("<"):
                ids[a["id"]] = pseudonym("actor", a["id"].lower(), r.salt)
                a["id"] = ids[a["id"]]
            elif (new := r.text(a["id"])) != a["id"]:  # an agent or system id that embeds an address or host
                ids[a["id"]] = new
                a["id"] = new
        for x in [*d["events"], *d.get("signals", [])]:
            who = x.get("actor")
            x["actor"] = ids.get(who, r.text(who) if isinstance(who, str) and not who.startswith("<") else who)
    d["outcome"]["details"] = r.value(d["outcome"].get("details") or {})
    if isinstance(d["outcome"].get("label_source"), str):
        d["outcome"]["label_source"] = r.text(d["outcome"]["label_source"])
    return Trace.model_validate(d)


def _version() -> str:
    """Hash of the rules: a trace stored under another version may hold what today's rules remove (``TraceStore.stale``)."""
    rules = "\n".join([PRIVATE_KEY, *SECRET_PATTERNS, ASSIGNMENT, URL_USERINFO, EMAIL, PRIVATE_IP, INTERNAL_HOST, CARD, PHONE, IBAN, SSN, _TRAILER.pattern,
                       *sorted(_SECRET_KEY_PARTS), *sorted(IDENTITY_KEYS)])
    return "redact-" + hashlib.sha256(rules.encode()).hexdigest()[:12]


REDACTION_VERSION = _version()
