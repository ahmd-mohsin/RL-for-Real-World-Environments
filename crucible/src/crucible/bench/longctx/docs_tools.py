"""Long-context ``docs`` app tools (copied into the forged World as ``longctx/tools/docs.py``). Plain functions over the
task's document, read-only; the document is ``longctx/corpus/context.txt`` next to this file. Every reply is bounded."""
import json
from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent / "corpus" / "context.txt"
_PARTS = Path(__file__).resolve().parent.parent / "corpus" / "parts.json"
MAX_READ = 65_536
MAX_HITS = 50
_cache: dict = {}


def _text() -> str:
    if "t" not in _cache:
        _cache["t"] = _DOC.read_text()
        _cache["low"] = _cache["t"].lower()
    return _cache["t"]


def info() -> dict:
    """The document's size in characters and its named parts (name, offset, chars), if it has any."""
    t = _text()
    parts = json.loads(_PARTS.read_text()) if _PARTS.exists() else []
    return {"chars": len(t), "approx_tokens": len(t) // 4, "parts": parts[:200], "max_read": MAX_READ}


def read(offset: int = 0, n: int = 4096) -> dict:
    """Read n characters (at most 65,536) of the document starting at character offset."""
    if isinstance(offset, bool) or not isinstance(offset, int) or isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("offset and n must be integers")
    t = _text()
    offset = max(0, min(offset, len(t)))
    n = max(0, min(n, MAX_READ))
    return {"offset": offset, "text": t[offset: offset + n], "next": min(len(t), offset + n), "total": len(t)}


def search(q: str, limit: int = 20, context: int = 120) -> dict:
    """Find a substring (case-insensitive) in the document: the first `limit` hits (at most 50), each with its offset and
    `context` characters around it (at most 400)."""
    if not isinstance(q, str) or not q:
        raise ValueError("q must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int) or isinstance(context, bool) or not isinstance(context, int):
        raise TypeError("limit and context must be integers")
    t = _text()
    low = _cache["low"]
    ql = q.lower()
    limit, context = max(1, min(limit, MAX_HITS)), max(0, min(context, 400))
    hits, i, total = [], low.find(ql), 0
    while i != -1:
        total += 1
        if len(hits) < limit:
            hits.append({"offset": i, "snippet": t[max(0, i - context): i + len(q) + context]})
        i = low.find(ql, i + 1)
    return {"q": q, "total": total, "hits": hits}
