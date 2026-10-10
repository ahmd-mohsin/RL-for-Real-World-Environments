"""Operators on tool-call sequences (World tasks: the SQL/toy family of research §15.6).

Each operator removes or corrupts one property of the oracle:

- ``drop_step``: omit one call (a missing refund, note, status change or approval request);
- ``numeric_shift``: change one numeric argument by a small amount (wrong amount, wrong id);
- ``enum_swap``: replace a string argument with another value the oracle uses for that argument name elsewhere
  (wrong status, wrong order id);
- ``swap_target``: point a call at another entity id the oracle touches (refund on the wrong ticket);
- ``drop_message``: omit one message to another actor (skipping the approval request);
- ``reorder``: move a write before the message that should precede it (acting before the approval arrives).

The validity and equivalence filters live in A3: a mutant whose end state equals the oracle's is equivalent (it changed
nothing the task is about) and is excluded from the score.
"""

from __future__ import annotations

import copy
from typing import Any

from crucible.accept.mutation.base import Mutant


def _name(a: dict) -> str:
    return a.get("function_name") or (f"message:{a.get('to')}" if a.get("to") else "?")


def generate(oracle: list[dict], *, limit: int = 40) -> list[Mutant]:
    out: list[Mutant] = []
    calls = [i for i, a in enumerate(oracle) if a.get("function_name")]
    msgs = [i for i, a in enumerate(oracle) if not a.get("function_name") and a.get("to")]
    values: dict[str, set] = {}
    for a in oracle:
        for k, v in (a.get("arguments") or {}).items():
            if isinstance(v, str) and len(v) < 40:
                values.setdefault(k, set()).add(v)
    for i in calls:
        out.append(Mutant(f"drop_step/{i}", "drop_step", oracle[:i] + oracle[i + 1:], f"omit {_name(oracle[i])}"))
    for i in msgs:
        out.append(Mutant(f"drop_message/{i}", "drop_message", oracle[:i] + oracle[i + 1:], f"omit the message to {oracle[i].get('to')}"))
    for i in calls:
        for k, v in sorted((oracle[i].get("arguments") or {}).items()):
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                m = copy.deepcopy(oracle)
                m[i]["arguments"][k] = round(v + (1 if isinstance(v, int) else max(0.01, abs(v) * 0.1)), 2)
                out.append(Mutant(f"numeric_shift/{i}/{k}", "numeric_shift", m, f"{_name(oracle[i])}.{k}: {v} -> {m[i]['arguments'][k]}"))
            elif isinstance(v, str) and len(values.get(k, ())) > 1:
                alt = min(values[k] - {v})
                m = copy.deepcopy(oracle)
                m[i]["arguments"][k] = alt
                out.append(Mutant(f"enum_swap/{i}/{k}", "enum_swap", m, f"{_name(oracle[i])}.{k}: {v!r} -> {alt!r}"))
    for i in msgs:
        later = [j for j in calls if j > i]
        if later:
            j = later[0]
            m = list(oracle)
            m.insert(i, m.pop(j))
            out.append(Mutant(f"reorder/{j}-before-{i}", "reorder", m, f"{_name(oracle[j])} before the message to {oracle[i].get('to')}"))
    seen, uniq = set(), []
    for x in out:
        k = repr(x.solution)
        if k not in seen and x.solution != oracle:
            seen.add(k)
            uniq.append(x)
    return uniq[:limit]


def is_actions(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(a, dict) and ("function_name" in a or "to" in a) for a in x)
