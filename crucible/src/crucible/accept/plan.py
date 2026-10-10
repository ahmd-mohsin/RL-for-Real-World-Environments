"""Planning (research m9-m10/01 §15.3): select the required checks, order them by dependencies and then by cost class
(static -> cpu -> docker -> gpu -> paid -> human), so cheap failures surface before expensive grades are spent."""

from __future__ import annotations

from collections.abc import Sequence

from crucible.accept.checks.base import CheckDef
from crucible.accept.model import COST_CLASSES


def order(checks: dict[str, CheckDef], selected: Sequence[str]) -> list[CheckDef]:
    want = set(selected)
    for c in list(want):  # dependencies of a selected check are planned too
        stack = [c]
        while stack:
            for d in checks[stack.pop()].depends_on:
                if d not in want:
                    want.add(d)
                    stack.append(d)
    out: list[CheckDef] = []
    done: set[str] = set()
    pending = sorted((checks[c] for c in want), key=lambda c: (COST_CLASSES.index(c.cost_class), c.id))
    while pending:
        ready = [c for c in pending if set(c.depends_on) <= done]
        if not ready:
            raise ValueError(f"dependency cycle among {[c.id for c in pending]}")
        nxt = ready[0]
        out.append(nxt)
        done.add(nxt.id)
        pending.remove(nxt)
    return out
