"""Text that UTF-8 cannot encode: lone surrogates, such as ``"\\udcff"`` from ``os.fsdecode`` of a non-UTF-8 file name or
from a model writing ``"\\udcff"`` inside its tool-call JSON. Python holds them, JSON writers and hashes refuse them.

Two policies, chosen per field. Free text that comes from the environment (a failure reason naming a file) is escaped to
``\\udcff`` text with ``safe_text`` or the ``SafeText`` field type: lossless to read, always writable (R-10; PATTERN:
karotte ``src/karotte/text.py:escape_surrogates``, MIT). Text the policy wrote is left as it is, so the World can find it
with ``unencodable`` and refuse the call with policy blame (M14 campaign B1, ``world.World._call_tool``).
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator


def _encodable(s: str) -> bool:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def unencodable(x: Any, path: str = "") -> str | None:
    """Where ``x`` holds a string that UTF-8 cannot encode (a lone surrogate such as ``"\\udcff"``, which a model emits as
    plain ASCII inside tool-call JSON), else None. M14 campaign B1: such an argument made the trace write raise
    ``PydanticSerializationError`` inside ``World.step``, which masked the policy's own failing episode (unknown blame)."""
    if isinstance(x, str):
        return None if x.isascii() or _encodable(x) else (path or "(value)")
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(k, str) and unencodable(k) is not None:
                return f"{path}/{safe_text(k)}" if path else safe_text(k)
            if (p := unencodable(v, f"{path}/{k}" if path else str(k))) is not None:
                return p
    elif isinstance(x, list | tuple):
        for i, v in enumerate(x):
            if (p := unencodable(v, f"{path}/{i}" if path else str(i))) is not None:
                return p
    return None


def safe_text(s: str) -> str:
    """``s`` with every code point UTF-8 cannot encode escaped as ``\\udcff`` text (lossless to read, always writable)."""
    return s if s.isascii() or _encodable(s) else s.encode("utf-8", "backslashreplace").decode("utf-8")


def safe_json(x: Any) -> Any:
    """``x`` with every string (keys included) passed through ``safe_text``; the same object when nothing needs escaping."""
    if isinstance(x, str):
        return safe_text(x)
    if unencodable(x) is None:
        return x
    if isinstance(x, dict):
        return {safe_text(k) if isinstance(k, str) else k: safe_json(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [safe_json(v) for v in x]
    return x


# A free-text field that is always writable: lone surrogates are escaped at validation.
SafeText = Annotated[str, AfterValidator(safe_text)]
