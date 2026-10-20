"""Redaction of everything that leaves for W&B or Weave (swarm-pipeline/11 §5): the M3 redaction, not a new one.

Every value logged through ``crucible.observe`` passes ``Scrubber.value`` first:

- ``connectors.redact.Redactor`` (credential formats, private keys, JWTs, ``KEY=value`` assignments, cards, IBANs; e-mail
  addresses, phone numbers and internal hosts become HMAC pseudonyms under the tenant salt, ``data/redaction.salt`` or
  ``CRUCIBLE_REDACTION_SALT``; a value under a credential-named key is removed whatever it looks like);
- the exact values of this process's credential environment variables (``*_API_KEY``, ``*TOKEN*``, ``*SECRET*``, ...,
  ``WANDB_API_KEY`` and ``TINKER_API_KEY`` included) are removed by value: a key with no vendor prefix matches no pattern.
  The values are read from ``os.environ`` only; ``.env`` is never opened here;
- ``core.secrets.scrub_secrets`` last (the secrets the process registered, ``core.secrets.register_secret``).

``truncate`` cuts long text for a trace view; the full text goes to a W&B artifact, redacted the same way.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

from crucible.connectors.redact import RedactionPolicy, Redactor
from crucible.core.secrets import MIN_SECRET_LEN, scrub_obj, scrub_secrets

# the variable names whose values are credentials (the M14 rule of ``RedactionPolicy.with_env_file``)
SECRET_ENV = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|COOKIE|AUTH(?!OR)|DSN|DATABASE_URL|\bPAT\b|_PAT$|_PASS$|_PWD$")
NOT_SECRET_ENV = {"WANDB_KEY_PROMPT", "KEYCHAIN", "SSH_AUTH_SOCK"}


def env_secret_values(env: dict[str, str] | None = None) -> list[str]:
    """The values of the credential-named variables of the process environment (longest first)."""
    env = dict(os.environ if env is None else env)
    vals = {v for k, v in env.items() if SECRET_ENV.search(k.upper()) and k.upper() not in NOT_SECRET_ENV and isinstance(v, str) and len(v) >= MIN_SECRET_LEN}
    return sorted(vals, key=len, reverse=True)


class Scrubber:
    """One redaction pass for observability payloads (strings, and dicts/lists of them, keys included)."""

    def __init__(self, *, salt: str | None = None, extra_values: Iterable[str] = (), env: dict[str, str] | None = None):
        vals = [*env_secret_values(env), *(v for v in extra_values if isinstance(v, str) and len(v) >= MIN_SECRET_LEN)]
        self.policy = RedactionPolicy(secret_values=vals, salt=salt, pseudonymize_actors=False)
        self.r = Redactor(self.policy)

    @property
    def hits(self) -> dict[str, int]:
        return dict(self.r.hits)

    def text(self, s: str) -> str:
        return scrub_secrets(self.r.text(s)) if s else s

    def value(self, v: Any) -> Any:
        """Strings redacted, structure kept; tuples become lists; non-JSON leaves are left to the caller."""
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, tuple):
            v = list(v)
        if isinstance(v, dict | list):
            return scrub_obj(self.r.value(v))
        return v


def truncate(s: Any, n: int) -> tuple[str, bool]:
    """``(text cut to n characters with a marker, whether it was cut)``."""
    t = s if isinstance(s, str) else ("" if s is None else str(s))
    if len(t) <= n:
        return t, False
    return t[:n] + f"\n... [{len(t) - n} more characters in the episode artifact]", True


_default: Scrubber | None = None


def default_scrubber() -> Scrubber:
    """The process's scrubber (built once; the environment is read when it is built)."""
    global _default
    if _default is None:
        _default = Scrubber()
    return _default


def reset_default() -> None:
    global _default
    _default = None
