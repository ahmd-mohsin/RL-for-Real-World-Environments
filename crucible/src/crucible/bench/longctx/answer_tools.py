"""Long-context ``answer`` app tool (copied into the forged World as ``longctx/tools/answer.py``): the final answer is
World state, graded after the episode (the last submission counts)."""
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "db" / "answers.db"
MAX_CHARS = 200_000


def submit(text: str) -> dict:
    """Submit your final answer (the last submission counts)."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(text) > MAX_CHARS:
        raise ValueError(f"an answer is at most {MAX_CHARS} characters")
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    with sqlite3.connect(_DB) as c:
        n = c.execute("INSERT INTO answers(text, at) VALUES (?, ?)", (text, now)).lastrowid
    return {"ok": True, "submission": n, "chars": len(text)}
