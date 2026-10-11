"""The rest of the agentenv trigger language for Crucible's own scheduler (m13 C8). PORT of the pure helpers of
agentenv ``env/gateway/triggers.py`` (Apache-2.0, credited in THIRD_PARTY.md): predicates, dot paths with ``[n]`` /
``[*]``, ``extract`` with ``match``/``take``, and ``${args.*}`` / ``${result.*}`` / ``${binding}`` templating with the
optional ``?`` marker. The semantics are agentenv's, so an event means the same in step mode (Crucible's scheduler) and in
wall mode (delegated to the gateway).
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

PREDICATE_KEYS = ("regex", "equals", "exists")
CTX_PREFIXES = ("args", "result")
_VAR_RE = re.compile(r"\$\{([\w.\[\]*]+\??)\}")
_DROP = object()


class TemplateError(LookupError):
    """A required ``${args.*}`` / ``${result.*}`` placeholder had no value in the provoking call."""


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def eval_predicate(pred: dict, value: Any) -> bool:
    if "regex" in pred:
        return re.search(pred["regex"], as_text(value)) is not None
    if "equals" in pred:
        return value == pred["equals"]
    non_empty = value is not None and value != "" and value != []
    return non_empty == pred["exists"]


def validate_predicate(pred: Any, where: str) -> None:
    if not isinstance(pred, dict) or sum(k in pred for k in PREDICATE_KEYS) != 1:
        raise ValueError(f"{where}: a predicate has exactly one of {PREDICATE_KEYS}")
    if "regex" in pred:
        re.compile(pred["regex"])
    if "exists" in pred and not isinstance(pred["exists"], bool):
        raise ValueError(f"{where}: exists must be a boolean")


def apply_path(data: Any, path: str) -> Any:
    current = data
    for segment in path.split("."):
        m = re.fullmatch(r"(\w+)((?:\[(?:\d+|\*)\])*)", segment)
        if m is None:
            raise ValueError(f"invalid path segment {segment!r}")
        key, indices = m.group(1), m.group(2)
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
        for idx in re.findall(r"\[(\d+|\*)\]", indices):
            if idx == "*":
                continue
            if not isinstance(current, list) or int(idx) >= len(current):
                return None
            current = current[int(idx)]
    return current


def apply_extract(extract: Any, data: Any, raw_text: str) -> Any:
    if extract is None:
        return raw_text
    if isinstance(extract, str):
        return apply_path(data, extract)
    items = apply_path(data, extract["path"])
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and all(eval_predicate(p, item.get(f)) for f, p in extract.get("match", {}).items()):
            return item.get(extract["take"]) if extract.get("take") else item
    return None


def result_data(text: str | None) -> Any:
    """A tool result's text parsed as JSON, the raw text when it is not JSON, or None."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def _ctx_path(name: str) -> tuple[str | None, str]:
    head, _, path = name.partition(".")
    return (head, path) if head in CTX_PREFIXES else (None, "")


def _lookup(name: str, bindings: dict, ctx: dict | None) -> tuple[bool, Any]:
    name = name.rstrip("?")
    if name in bindings:
        return True, bindings[name]
    root_name, path = _ctx_path(name)
    if root_name is None or ctx is None:
        return False, None
    root = ctx.get(root_name)
    if not path:
        return root is not None, root
    value = apply_path(root, path) if isinstance(root, dict) else None
    return value is not None, value


def _resolve(name: str, literal: str, bindings: dict, ctx: dict | None) -> tuple[bool, Any]:
    found, value = _lookup(name, bindings, ctx)
    if not found and not name.endswith("?") and _ctx_path(name.rstrip("?"))[0] is not None:
        raise TemplateError(f"unresolved placeholder {literal}")
    return found, value


def template(args: Any, bindings: dict | None = None, ctx: dict | None = None) -> Any:
    """Substitute placeholders recursively. A string that is one placeholder takes the raw value; an embedded one is
    stringified. Unresolved: a required args/result placeholder raises, a bare name stays literal, an optional whole
    value is dropped (so the tool applies its own default)."""
    bindings = bindings or {}

    def sub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: r for k, v in value.items() if (r := sub(v)) is not _DROP}
        if isinstance(value, list):
            return [r for v in value if (r := sub(v)) is not _DROP]
        if not isinstance(value, str):
            return value
        whole = _VAR_RE.fullmatch(value)
        if whole is not None:
            name = whole.group(1)
            found, resolved = _resolve(name, value, bindings, ctx)
            if not found:
                return _DROP if name.endswith("?") else value
            return resolved if _ctx_path(name.rstrip("?"))[0] else str(resolved)

        def repl(m: re.Match) -> str:
            found, resolved = _resolve(m.group(1), m.group(0), bindings, ctx)
            if not found:
                return "" if m.group(1).endswith("?") else m.group(0)
            return resolved if isinstance(resolved, str) else json.dumps(resolved, default=str)

        return _VAR_RE.sub(repl, value)

    out = sub(args)
    return args if out is _DROP else out


def where_value(ctx: dict, field: str) -> Any:
    root_name, path = _ctx_path(field)
    if root_name is not None and path:
        root = ctx.get(root_name)
        return apply_path(root, path) if isinstance(root, dict) else None
    args = ctx.get("args")
    return args.get(field) if isinstance(args, dict) else None


def where_matches(where: dict[str, dict], ctx: dict) -> bool:
    return all(eval_predicate(p, where_value(ctx, f)) for f, p in where.items())


def validate_check(check: Any, where: str) -> dict:
    if not isinstance(check, dict):
        raise ValueError(f"{where}: check must be an object")  # noqa: TRY004 (pydantic validators surface ValueError only)
    steps = check.get("steps", [check] if "tool" in check else None)
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{where}: check needs 'steps' (or the single-step tool/predicate form)")
    for i, step in enumerate(steps):
        if not isinstance(step.get("tool"), str) or not step["tool"]:
            raise ValueError(f"{where}.steps[{i}]: tool must be a non-empty string")
        if "predicate" in step:
            validate_predicate(step["predicate"], f"{where}.steps[{i}].predicate")
    if "predicate" not in steps[-1]:
        raise ValueError(f"{where}: the last check step carries a predicate")
    return {"steps": steps}


def exp_interval(seed: int, mean_s: float, k: int) -> float:
    """The k-th (0-based) exponential inter-arrival of a seeded stream: agentenv draws sequentially from
    ``random.Random(seed)``, so the k-th draw is recomputed from the seed (reproducible after any restore)."""
    rng = random.Random(seed)
    v = 0.0
    for _ in range(k + 1):
        v = rng.expovariate(1.0 / mean_s)
    return v
