"""Credential patterns and a scrubber for text that leaves the process (M10 audit V12: a credential a provider echoed in
an error message reached 14 record and trace files). The patterns are the M3 redaction's (``connectors/redact.py``
imports them from here, so both stay one list); ``scrub_secrets`` is for the lower layers (the LLM client, failure
reasons, journals) that cannot import connectors."""

from __future__ import annotations

import re

_B = r"(?<![A-Za-z0-9_\-])"  # prefix-format keys start at a word boundary: "task-04-gpu-migration" is not an "sk-" key
SECRET_PATTERNS = [
    # OpenAI / OpenRouter / Anthropic; a bare "sk-" key must carry a digit (prose like "sk-learn-compatible-..." does not)
    _B + r"sk-(?:(?:or-v1-|ant-|proj-|svcacct-|admin-)[A-Za-z0-9_\-]{20,}|(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{20,})",
    _B + r"gh[pousr]_[A-Za-z0-9]{30,}", _B + r"github_pat_[A-Za-z0-9_]{40,}",  # GitHub
    _B + r"hf_[A-Za-z0-9]{30,}",  # Hugging Face
    _B + r"(?:AKIA|ASIA)[0-9A-Z]{16}",  # AWS access key id
    _B + r"xox[abposr]-[A-Za-z0-9-]{10,}",  # Slack
    _B + r"glpat-[A-Za-z0-9_\-]{20}",  # GitLab
    _B + r"AIza[0-9A-Za-z_\-]{35}",  # Google API key
    r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
    _B + r"dckr_pat_[A-Za-z0-9_\-]{20,}",  # Docker Hub
    _B + r"rpa_[A-Za-z0-9]{30,}",  # RunPod
    _B + r"npm_[A-Za-z0-9]{36}", r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{50,}",  # npm, PyPI
    _B + r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}",  # Stripe
    r"SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}",  # SendGrid
    _B + r"r8_[A-Za-z0-9]{30,}", _B + r"gsk_[A-Za-z0-9]{40,}", _B + r"xai-[A-Za-z0-9]{40,}",  # Replicate, Groq, xAI
    _B + r"tskey-[a-z]+-[A-Za-z0-9\-]{20,}", _B + r"dapi[0-9a-f]{32}",  # Tailscale, Databricks
    _B + r"ya29\.[A-Za-z0-9_\-]{20,}", _B + r"hv[sbr]\.[A-Za-z0-9_\-]{20,}",  # Google OAuth access token, Vault
    r"https://hooks\.slack\.com/services/[A-Za-z0-9/]{20,}", r"https://(?:\w+\.)?discord(?:app)?\.com/api/webhooks/[0-9]+/[A-Za-z0-9_\-]+",
    r"(?i:\bbearer\s+[A-Za-z0-9._~+/\-]{20,}=*)",  # any bearer token in a header or a curl command
    r"(?i:\bbasic\s+[A-Za-z0-9+/]{8,}={0,2})",  # HTTP basic credentials
    r"(?i:(?<=cookie:)\s*(?:[^\s;=]{1,128}=[^\s;]{1,4096};?\s?){1,64})",  # cookie headers carry session ids (pairs only)
    # Azure storage / service bus connection strings and SAS signatures
    r"(?i:(?<=AccountKey=)|(?<=SharedAccessKey=)|(?<=SharedAccessSignature=)|(?<=[?&]sig=))[A-Za-z0-9+/%]{20,}={0,2}",
]
# KEY = VALUE where the key names a credential (env, YAML, JSON, CLI, URL-encoded, escaped quotes). Only the value goes;
# a value that is code (``tokenizer.encode(x)``, ``os.getcwd()``), a number or a reference (``${DB_PASS}``) stays.
_KEYWORD = r"(?:passw(?:or)?d|passphrase|pwd|secret|api[_-]?key|access[_-]?key|private[_-]?key(?:[_-]?id)?|auth[_-]?token|credentials?|token|pass)"
ASSIGNMENT = (r"(?i)(?<![A-Za-z0-9_.\-])(?P<key>[A-Za-z0-9_.\-]{0,40}?" + _KEYWORD + r"[A-Za-z0-9_\-]{0,40})"
              r"(?P<sep>\\?[\"']?\s{0,3}(?::|=>|=|%3D)\s{0,3})"
              r"(?:(?P<q>\\?[\"'])(?P<qv>[^\"'\\\n]{4,300})(?P=q)|(?P<v>[^\s'\",;)\\<>]{4,300}))")
_CODE_VALUE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+.*|[A-Za-z_][\w.]*\(.*|\d+|(?i:none|null|true|false|undefined|\*+|x+)|[$%{].*|env:.*")
PRIVATE_KEY = r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----|(?:[A-Za-z0-9+/=\s]|\\[nr])*)"
# user:password@ in any URL (git remotes, DSNs): the userinfo goes, the host stays
URL_USERINFO = r"(?<=://)[^\s/:@<>\"']{1,256}:[^\s/@<>\"']{1,256}@"
# a credential in a URL query (``?key=...``, ``&access_token=...``): httpx errors include the request URL
QUERY_SECRET = r"(?i)(?<=[?&])(?:key|api[_-]?key|apikey|token|access[_-]?token|auth|sig|signature|secret|password)=[^&\s\"'<>]+"
_SECRETS = [re.compile(x) for x in (*SECRET_PATTERNS, PRIVATE_KEY, URL_USERINFO, QUERY_SECRET)]
_ASSIGN = re.compile(ASSIGNMENT)


def scrub_secrets(text: str) -> str:
    """Every credential-shaped substring replaced by ``<secret>``; a ``key = value`` assignment keeps its key."""
    if not text:
        return text
    out = text
    for rx in _SECRETS:
        out = rx.sub("<secret>", out)

    def assign(m: re.Match) -> str:
        v = m.group("qv") or m.group("v") or ""
        if _CODE_VALUE.fullmatch(v):
            return m.group(0)
        q = m.group("q") or ""
        return f"{m.group('key')}{m.group('sep')}{q}<secret>{q}"

    return _ASSIGN.sub(assign, out)


def scrub_obj(obj: object) -> object:
    """Every string inside a JSON-like structure scrubbed (keys and values); structure untouched. Use this, not
    ``scrub_secrets`` on serialized JSON: the assignment rule would rewrite ``"token_ids": [1, 2`` as a credential."""
    if isinstance(obj, str):
        return scrub_secrets(obj)
    if isinstance(obj, dict):
        return {scrub_secrets(k) if isinstance(k, str) else k: scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_obj(v) for v in obj]
    return obj


def scrub_json(text: str, **dump_kw: object) -> str:
    """A serialized JSON document with every string scrubbed (re-serialized with ``dump_kw``)."""
    import json

    return json.dumps(scrub_obj(json.loads(text)), **dump_kw)  # type: ignore[arg-type]
