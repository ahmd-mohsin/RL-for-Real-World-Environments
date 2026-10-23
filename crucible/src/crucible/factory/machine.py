"""The declarative cyclic machine of the task factory (M17; docs/research/task-factory/03 §2, 06 "Machine (Q)").

A machine is data (a dict, or YAML): states with typed ports, guarded transitions, bounded cycles. It is a PORT (MIT) of
Prime Agent's factory validator, guards and canonicalizer (``prime-agent-runtime/src/rlm/factory.py``; file:line in the
comments below, PA = that file), with the defects study 01 found removed:

- **Exhaustion is ``escalate``, never ``done``.** Prime ends a loop whose last re-entry is blocked as a silent success
  (PA:3248-3277). Here a transition onto a state at ``max_entries``, a settle with no passing transition, the
  transition cap and a failed state all *pause* the run with a decision request (``RunResult.decision_request``); a
  person (or a compiled gate) answers through ``resume(decision)``.
- **The run state is journaled to disk** (``journal.jsonl``, the forge ``Journal``: append-only, torn-tail repair) and
  rebuilt from it on resume. Prime keeps it in kernel memory (PA:19-23, :1422-1424).
- **Run caps**: USD through the forge ``Ledger`` (reserve an estimate, settle what the role's LLM clients booked in a
  ``SpendScope``), transitions, wall clock (seconds of work, summed across resumes). Prime has no dollar cap (PA:2483).
- **Loop-back carries findings to the owning state**: the settle that fires a transition hands its findings (the
  ``carry`` port, or the settle's own findings) to the next entry of the target state (Prime: a typed port only,
  PA:4640-4709; retries never carry findings, PA:2984-3027, and the same holds here).
- **Cross-family rule enforced by config, not by a docstring** (Prime: a convention, study 01 §1.3): a state declares
  ``different_family_from: [state, ...]``; the validator refuses a model assignment that gives both the same family.
- **Staging decides access**: a state declares ``isolated_from: [state, ...]``. The validator refuses any input that
  (transitively) reads those states; at run time a ``canary: true`` state's outputs carry a canary (forge orchestrator
  pattern), findings carried into an isolated state are scanned for the canary and for quoted shingles of the source's
  outputs (``forge/repo._marks``) and withheld when they match, and an isolated state whose output carries the canary is a
  breach (a failed attempt).

One token runs at a time (a per-task machine is a chain with loop-backs; parallelism lives inside state functions and
across machines in ``root.py``). So joins and ``foreach`` are refused, and the first passing transition in declared order
fires (Prime fires every passing guard, PA:2322-2430).

Machine shape::

    {"run": {"max_transitions": 80, "budget_s": 3600, "budget_usd": 2.0, "failure_policy": "escalate"},
     "states": [{"id": "verifier", "role": "verifier_writer", "max_entries": 4, "isolated_from": ["oracle"],
                 "inputs": [{"name": "prompt", "type": "text", "from": "author.prompt"},
                            {"name": "intent", "type": "json", "from": "params.intent"}],
                 "outputs": [{"name": "checkset", "type": "json"}], "retries": 1, "estimate_usd": 0.01},
                {"id": "ledger", "fn": "ledger_check", ...},
                {"id": "accepted", "terminal": "done"}, {"id": "dropped", "terminal": "rejected"}],
     "transitions": [{"from": "ledger", "to": "verifier", "when": "verdict.owner == 'verifier'", "carry": "verdict.findings"}]}

Guards: Prime's leaf form ``{"output", "path"?, "op", "value"?}`` with ops ``eq ne gt gte lt lte exists contains``
(PA:60, :1270-1325), combined with ``{"all": [...]}``, ``{"any": [...]}``, ``{"not": g}``, or a string in a small
expression language compiled to that form by a tokenizer and recursive-descent parser (never ``eval``):
``verdict.ok == true and not (verdict.owner == "spec")``, ``exists report.hacks``, ``stats.p >= 0.1``.

**Machine v2 (P1; docs/research/swarm-pipeline/00 §4, 01 facts 5-6).** Ported from Prime Agent at HEAD 7cbffa77b
(``F`` = ``prime-agent-runtime/src/rlm/factory.py`` there, MIT):

- **Infra vs task failure** (``F:1095-1116``, ``:2876-2895``, ``:2990-3084``). A state attempt that dies of infrastructure
  (``is_infra_error``: an ``InfraError``, a failure kind whose blame is ``infra``, a provider outage, a lost container,
  Prime's "child worker unreachable" / "supervisor link" markers) is re-checked after a quiet period (the injected
  ``sleep``, so tests never wait) through the run's ``infra_probe(state, error)`` (``"alive"``: the sandbox answered, the
  same session is re-run; ``"dead"``: the zombie is torn down first, so the respawn never runs beside it in the same
  worktree), then re-run **once for free**: the attempt number, ``retries`` and ``max_entries`` are not spent, and the dead
  attempt's spend row keeps its error. A second infra death takes the normal retry path; if the retries end on infra, the
  run pauses with kind ``infra`` (V8: masked, a decision for a person, never a task failure).
- **A missing capture fails its producer** (``F:2969-2988``, ``:3123-3222``). ``artifact`` ports carry ``{"path",
  "sha256"}`` of a file or directory under the run's ``artifact_root`` (the task repo). The producer's settle verifies each
  one exists and hashes right, after one bounded re-capture window; otherwise the attempt fails (retries, then policy)
  and its outputs are kept as ``needs_verify`` (provisional). A consumer whose bound artifact has since vanished or
  changed does not fail: the run pauses with kind ``capture_failed`` naming the **producer**, and ``resume`` re-runs it.
- **A declared recovery posture** (``F:5099-5107``: a prose rule there, "validation cannot judge prose"; a field here).
  A state with ``mutates: true`` must declare ``recovery: reconcile | reset`` or the validator refuses the machine; a
  ``session`` state that writes its worktree must declare ``mutates: true``. The run's ``trees(state)`` gives the state's
  ``MutableTree``; a snapshot is taken (and journaled) before the first attempt of each entry. Every re-run of that entry
  (a retry, an infra respawn, a resumed failed entry, a crash replay) first recovers: ``reset`` restores the snapshot;
  ``reconcile`` hands the attempt ``ctx.recovery = {"mode", "diff", "findings"}``: the earlier attempt's diff against the
  snapshot and why it failed, for the retry prompt.
- **Exhaustion escalates** (V7) as before; Prime still ends it ``done`` at 7cbffa77b (``F:2469-2483``, ``:3515-3541``,
  pinned by ``TF:5450-5477``). The regression test ``test_m17_p1_machine`` contrasts the two.
- **Dollar and token budgets per state and per run** (V10; Prime has neither, ``F:5071-5098``). ``run.budget_usd`` /
  ``run.budget_tokens`` stop the run; a state's ``budget_usd`` / ``budget_tokens`` cap the spend of all its attempts and
  entries: an attempt that ends over the cap fails with no retry and the run pauses with kind ``state_budget`` (``resume``
  may grant ``extra_usd`` / ``extra_tokens``). Tokens are booked with ``book_tokens(n)`` (or ``Settle.tokens``) inside a
  state; ``ctx.budget`` tells the handler what is left.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from crucible.actors.llm import SpendScope
from crucible.core.schema.run import default_blame
from crucible.core.serialize import load_yaml
from crucible.forge.budget import BudgetExhausted, Ledger
from crucible.forge.orchestrator import Journal
from crucible.forge.repo import _marks

# ---- constants (PA:56-82, adapted) ------------------------------------------------------------------------------------
FAILURE_POLICIES: tuple[str, ...] = ("fail_fast", "escalate")  # PA:56 also has "continue"; one token has no dependents to continue
PORT_TYPES: tuple[str, ...] = ("text", "json", "artifact")  # PA:57 + artifact (P1: files in the task repo, by path + hash)
RECOVERY: tuple[str, ...] = ("reconcile", "reset")  # F:5099-5107 (P1: a field, not prose)
# what a session state may configure (``factory/sessions.py`` reads it; validated here so a bad machine fails at load)
SESSION_KEYS: tuple[str, ...] = ("brief", "captures", "specialist", "repo", "base", "image", "network", "read_only", "max_turns", "wall_s",
                                "potential", "checkpoint_every", "checkpoint_cap", "settle_on_budget")  # potential: the builder's Φ (sessions.SessionPool)
CAPTURE_KINDS: dict[str, str] = {"json": "json", "text": "text", "file": "artifact", "dir": "artifact", "patch": "artifact", "commits": "artifact"}
# F:1095-1116 (``_INFRA_ERROR_MARKERS``), plus the runner errors study 01 §6.3 maps to infra: a lost or unreachable
# container, a docker daemon that does not answer, a provider 5xx / connection fault
INFRA_MARKERS: tuple[str, ...] = (
    "child worker unreachable", "supervisor link",  # F:1095-1098, verbatim
    "no such container", "is not running", "cannot connect to the docker daemon", "container not found", "error during connect",
    "connection reset", "connection refused", "server disconnected", "bad gateway", "service unavailable", "gateway timeout",
)
# kinds whose default blame is infra but which are not deaths: a spent budget stops the run instead (``_run_state``)
NOT_RESPAWNED_KINDS: frozenset[str] = frozenset({"crucible:credits_exhausted"})
INFRA_KINDS_EXTRA: frozenset[str] = frozenset({"provider_unavailable", "crucible:env_container_lost", "crucible:backend_error"})
INFRA_QUIET_S = 2.0  # F:1089 (one collect-poll slice, POLL_TIMEOUT_MS / 1000)
RECAPTURE_WAIT_S = 1.0  # one bounded re-capture window before a missing artifact fails its producer (F:3171-3222)
RECONCILE_DIFF_CAP = 60_000  # characters of the earlier attempt's diff handed to a reconcile retry
GUARD_OPS: tuple[str, ...] = ("eq", "ne", "gt", "gte", "lt", "lte", "exists", "contains")  # PA:60
TERMINALS: tuple[str, ...] = ("done", "rejected")
MAX_STATES = 256
MAX_RETRIES = 10  # PA:63
MAX_ENTRIES_CAP = 64
MAX_TRANSITIONS_CAP = 10_000  # PA:70
TRANSITIONS_PER_STATE_DEFAULT = 10  # PA:71
STATE_MAX_ENTRIES_DEFAULT = 1  # PA:82
MAX_GUARD_VALUE_DEPTH = 64  # PA:96 (256 there)
MAX_GUARD_DEPTH = 32
OUTPUT_CAP_BYTES = 1_000_000  # PA:1016 binds 8,192 chars of a child's answer; drafts here carry Worlds
QUOTE_SHINGLE = 8  # tokens per quote mark when scanning carried findings (forge/repo.py uses 6 over zone files)
_STATE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")  # PA:84 allows [a-z0-9-]; underscores read better in role names
PARAMS = "params"  # input source namespace for run parameters (Prime's entry states read nothing, PA:633)

Status = str  # running | paused | done | rejected | failed | stopped


class MachineError(ValueError):
    """The machine (or its model assignment) is invalid; ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


# ---- P1: infra vs task failure, tokens, artifacts, trees -------------------------------------------------------------
class InfraError(RuntimeError):
    """A state attempt died of infrastructure (a sandbox, a worker, a provider), not of the task."""

    def __init__(self, message: str, failure_kind: str = "crucible:backend_error"):
        super().__init__(message)
        self.failure_kind = failure_kind


def is_infra_error(err: BaseException | None) -> bool:
    """PORT of ``_is_infra_error`` (F:1109-1116) over our runner's errors: an ``InfraError``; an exception whose
    ``failure_kind`` is infra-blamed (``core.schema.run.DEFAULT_BLAME``) or a provider / container kind; else Prime's
    message markers. A spent budget, a malformed model reply, a timeout and every other exception are the task's."""
    if err is None:
        return False
    if isinstance(err, InfraError):
        return True
    kind = getattr(err, "failure_kind", None)
    if isinstance(kind, str) and kind:
        if kind in NOT_RESPAWNED_KINDS:
            return False
        return kind in INFRA_KINDS_EXTRA or default_blame(kind) == "infra"
    text = str(err).lower()
    return any(m in text for m in INFRA_MARKERS)


_TOKEN_SCOPE: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar("crucible_factory_tokens", default=None)


def book_tokens(n: int) -> None:
    """Book ``n`` model tokens (prompt + completion) to the state attempt running in this task (no-op outside one)."""
    acc = _TOKEN_SCOPE.get()
    if acc is not None and n > 0:
        acc[0] += int(n)


class TokenScope:
    """``with TokenScope() as t: ...; t.tokens`` (the token twin of ``actors.llm.SpendScope``)."""

    def __enter__(self) -> TokenScope:  # noqa: PYI034
        self._acc = [0]
        self._tok = _TOKEN_SCOPE.set(self._acc)
        return self

    def __exit__(self, *exc: object) -> None:
        _TOKEN_SCOPE.reset(self._tok)

    @property
    def tokens(self) -> int:
        return self._acc[0]


def artifact_digest(path: Path) -> str:
    """sha256 of a file's bytes; of a directory: of its sorted (relative path, file sha256) rows (symlinks by target)."""
    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file() or x.is_symlink()):
        rel = p.relative_to(path).as_posix()
        body = ("link:" + str(p.readlink())).encode() if p.is_symlink() else p.read_bytes()
        h.update(f"{rel}\0{hashlib.sha256(body).hexdigest()}\n".encode())
    return h.hexdigest()


def artifact_problem(root: Path | None, value: Any) -> str | None:
    """Why ``value`` is not a live artifact under ``root`` (None: it is)."""
    if not isinstance(value, dict) or not _nonempty_str(value.get("path")) or not _nonempty_str(value.get("sha256")):
        return "an artifact must be {'path': <relative path>, 'sha256': <hex>}"
    if root is None:
        return "the run has no artifact_root to resolve artifacts against"
    base = Path(root).resolve()
    p = (base / value["path"]).resolve()
    if p != base and base not in p.parents:
        return f"artifact path {value['path']!r} escapes the artifact root"
    if not p.exists():
        return f"artifact {value['path']!r} does not exist"
    if artifact_digest(p) != value["sha256"]:
        return f"artifact {value['path']!r} does not hash to its recorded sha256 (changed after capture)"
    return None


@runtime_checkable
class MutableTree(Protocol):
    """A state's working tree (P1 recovery posture): ``snapshot`` returns a token, ``restore`` puts the tree back to it,
    ``diff`` is the tree's change since it (``factory/sessions.GitWorkspace`` implements it with git)."""

    async def snapshot(self, label: str) -> str: ...

    async def restore(self, token: str) -> None: ...

    async def diff(self, token: str) -> str: ...


class InfraProbe(Protocol):
    async def __call__(self, state_id: str, err: BaseException) -> str:  # "alive" | "dead"
        ...


class CaptureLost(RuntimeError):
    """A consumer's bound artifact vanished or changed after its producer settled: the producer's failure."""

    def __init__(self, producer: str, message: str):
        super().__init__(message)
        self.producer = producer


# ---- small predicates (PA:100-170) ------------------------------------------------------------------------------------
def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v != ""


def _value_is_finite(value: Any, _seen: frozenset[int] | None = None, _depth: int = 0) -> bool:
    """PORT of ``_value_is_finite`` (PA:109-166): finite JSON data, string keys, no cycle, bounded depth."""
    if _depth > MAX_GUARD_VALUE_DEPTH:
        return False
    seen = _seen or frozenset()
    if isinstance(value, (list, dict)):
        if id(value) in seen:
            return False
        seen = seen | {id(value)}
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return True
    if isinstance(value, list):
        return all(_value_is_finite(x, seen, _depth + 1) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _value_is_finite(x, seen, _depth + 1) for k, x in value.items())
    return False


def _ports(state: Mapping[str, Any], key: str) -> list[Any]:
    raw = state.get(key)
    return raw if isinstance(raw, list) else []


def _port_types(state: Mapping[str, Any], key: str) -> dict[str, str]:
    """PORT of ``_declared_port_types`` (PA:178-187)."""
    out: dict[str, str] = {}
    for p in _ports(state, key):
        if isinstance(p, dict) and _nonempty_str(p.get("name")) and p.get("type") in PORT_TYPES:
            out[p["name"]] = p["type"]
    return out


# ---- the guard language -----------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"""\s*(?:
    (?P<num>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<dq>"(?:[^"\\]|\\.)*")
  | (?P<sq>'(?:[^'\\]|\\.)*')
  | (?P<op>==|!=|>=|<=|>|<)
  | (?P<punct>[()\[\],])
  | (?P<word>[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*)
)""", re.VERBOSE)
_SYM_OPS = {"==": "eq", "!=": "ne", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}
_KEYWORDS = {"and", "or", "not", "exists", "contains", "true", "false", "null"}


class GuardSyntaxError(ValueError):
    pass


def _tokenize(src: str) -> list[tuple[str, Any]]:
    toks: list[tuple[str, Any]] = []
    pos, n = 0, len(src)
    while pos < n:
        if src[pos:].strip() == "":
            break
        m = _TOKEN.match(src, pos)
        if not m or m.end() == pos:
            raise GuardSyntaxError(f"unexpected character at {pos}: {src[pos:pos + 12]!r}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group(kind)
        if kind in ("num", "dq"):
            toks.append(("lit", json.loads(text)))
        elif kind == "sq":
            toks.append(("lit", text[1:-1].replace("\\'", "'").replace("\\\\", "\\")))
        elif kind == "op":
            toks.append(("op", _SYM_OPS[text]))
        elif kind == "punct":
            toks.append((text, text))
        elif text in ("true", "false", "null"):
            toks.append(("lit", {"true": True, "false": False, "null": None}[text]))
        elif text in _KEYWORDS:
            toks.append((text, text))
        else:
            toks.append(("ref", text))
    return toks


class _Parser:
    """Recursive descent: ``or := and ("or" and)*; and := unary ("and" unary)*; unary := "not" unary | "(" or ")" |
    "exists" REF | REF OP LITERAL | REF "contains" LITERAL``. Produces the dict form; never evaluates code."""

    def __init__(self, src: str):
        self.toks, self.i, self.depth = _tokenize(src), 0, 0

    def peek(self) -> tuple[str, Any] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, kind: str | None = None) -> tuple[str, Any]:
        t = self.peek()
        if t is None or (kind is not None and t[0] != kind):
            raise GuardSyntaxError(f"expected {kind or 'a token'} at token {self.i}, got {t[1] if t else 'end of input'!r}")
        self.i += 1
        return t

    def parse(self) -> dict[str, Any]:
        g = self.or_()
        if self.peek() is not None:
            raise GuardSyntaxError(f"unexpected {self.peek()[1]!r} at token {self.i}")
        return g

    def _nest(self) -> None:
        self.depth += 1
        if self.depth > MAX_GUARD_DEPTH:
            raise GuardSyntaxError(f"guard nests deeper than {MAX_GUARD_DEPTH}")

    def or_(self) -> dict[str, Any]:
        parts = [self.and_()]
        while self.peek() and self.peek()[0] == "or":
            self.take()
            parts.append(self.and_())
        return parts[0] if len(parts) == 1 else {"any": parts}

    def and_(self) -> dict[str, Any]:
        parts = [self.unary()]
        while self.peek() and self.peek()[0] == "and":
            self.take()
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else {"all": parts}

    def unary(self) -> dict[str, Any]:
        t = self.peek()
        if t is None:
            raise GuardSyntaxError("unexpected end of guard")
        if t[0] == "not":
            self.take()
            self._nest()
            return {"not": self.unary()}
        if t[0] == "(":
            self.take()
            self._nest()
            g = self.or_()
            self.take(")")
            return g
        if t[0] == "exists":
            self.take()
            return {**self.ref(), "op": "exists"}
        ref = self.ref()
        nt = self.peek()
        if nt and nt[0] == "op":
            self.take()
            return {**ref, "op": nt[1], "value": self.literal()}
        if nt and nt[0] == "contains":
            self.take()
            v = self.literal()
            return {**ref, "op": "contains", "value": v if isinstance(v, list) else [v]}
        raise GuardSyntaxError(f"expected an operator after {ref}")

    def ref(self) -> dict[str, Any]:
        _, text = self.take("ref")
        output, _, path = text.partition(".")
        return {"output": output, **({"path": path} if path else {})}

    def literal(self) -> Any:
        t = self.peek()
        if t and t[0] == "lit":
            return self.take()[1]
        if t and t[0] == "[":
            self.take()
            out = []
            if self.peek() and self.peek()[0] == "]":
                self.take()
                return out
            while True:
                out.append(self.literal())
                nt = self.take()
                if nt[0] == "]":
                    return out
                if nt[0] != ",":
                    raise GuardSyntaxError(f"expected ',' or ']' in a list, got {nt[1]!r}")
        raise GuardSyntaxError(f"expected a literal at token {self.i}, got {t[1] if t else 'end of input'!r}")


def parse_guard(src: str) -> dict[str, Any]:
    """Compile a guard expression to the dict form (raises ``GuardSyntaxError``)."""
    return _Parser(src).parse()


def _compile_guard(when: Any) -> Any:
    return parse_guard(when) if isinstance(when, str) else when


def _validate_guard(when: Any, where: str, src_types: dict[str, str], errors: list[str], depth: int = 0) -> None:
    """PORT of ``_validate_guard`` (PA:440-487), plus the all/any/not combinators."""
    if depth > MAX_GUARD_DEPTH:
        errors.append(f"{where} guard nests deeper than {MAX_GUARD_DEPTH}")
        return
    if not isinstance(when, dict):
        errors.append(f"{where} when must be an object or a guard expression string")
        return
    for comb in ("all", "any"):
        if comb in when:
            parts = when[comb]
            if len(when) != 1 or not isinstance(parts, list) or not parts:
                errors.append(f"{where} '{comb}' must be the only key and hold a non-empty list")
                return
            for p in parts:
                _validate_guard(p, where, src_types, errors, depth + 1)
            return
    if "not" in when:
        if len(when) != 1:
            errors.append(f"{where} 'not' must be the only key")
            return
        _validate_guard(when["not"], where, src_types, errors, depth + 1)
        return
    output = when.get("output")
    if not _nonempty_str(output):
        errors.append(f"{where} when requires a non-empty output")
    elif output not in src_types:
        errors.append(f"{where} when.output {output!r} is not a declared output of the from-state")
    else:
        path = when.get("path")
        if path is not None:
            if not _nonempty_str(path):
                errors.append(f"{where} when.path must be a non-empty dotted path")
            elif src_types[output] not in ("json", "artifact"):
                errors.append(f"{where} when.path requires a json or artifact output, got text output {output!r}")
    op = when.get("op")
    if op not in GUARD_OPS:
        errors.append(f"{where} when.op must be one of {list(GUARD_OPS)}, got {op!r}")
        return
    if op == "exists":
        return
    value = when.get("value")
    if op in ("gt", "gte", "lt", "lte") and not _is_number(value):
        errors.append(f"{where} when.op {op!r} requires a numeric value")
    elif op == "contains" and (not isinstance(value, list) or not value):
        errors.append(f"{where} when.op 'contains' requires a non-empty list value")
    elif op in ("eq", "ne") and not _is_scalar(value):
        errors.append(f"{where} when.op {op!r} requires a scalar value")
    if not _value_is_finite(value):
        errors.append(f"{where} when.value must be finite JSON data")


def _json_equal(actual: Any, expected: Any) -> bool:
    """PORT of ``_json_equal`` (PA:1252-1267): a bool never equals a number; numbers compare exactly."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual is expected
    if actual is None or expected is None:
        return actual is None and expected is None
    if _is_number(actual) and _is_number(expected):
        return actual == expected
    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected
    return False


def guard_passes(when: Mapping[str, Any] | None, outputs: Mapping[str, Any]) -> bool:
    """PORT of ``_guard_passes`` (PA:1270-1325) plus combinators. ``None`` always passes. A missing port fails every op
    except ``exists`` (false then)."""
    if when is None:
        return True
    if "all" in when:
        return all(guard_passes(g, outputs) for g in when["all"])
    if "any" in when:
        return any(guard_passes(g, outputs) for g in when["any"])
    if "not" in when:
        return not guard_passes(when["not"], outputs)
    port = when.get("output")
    value: Any = None
    found = False
    path = when.get("path")
    if path:
        current = outputs.get(port)
        if isinstance(current, dict):
            found = True
            for part in path.split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    found = False
                    break
            value = current
    elif port in outputs:
        found, value = True, outputs[port]
    op = when.get("op")
    if op == "exists":
        return found and value is not None
    if not found:
        return False
    expected = when.get("value")
    if op == "eq":
        return _json_equal(value, expected)
    if op == "ne":
        return not _json_equal(value, expected)
    if op in ("gt", "gte", "lt", "lte"):
        if not _is_number(value) or not _is_number(expected):
            return False
        return {"gt": value > expected, "gte": value >= expected, "lt": value < expected, "lte": value <= expected}[op]
    if op == "contains":
        if not isinstance(expected, list) or not expected:
            return False
        if isinstance(value, list):
            return all(item in value for item in expected)
        if isinstance(value, str):
            return all(isinstance(item, str) and item in value for item in expected)
    return False


def _resolve_path(outputs: Mapping[str, Any], ref: str) -> Any:
    port, _, path = ref.partition(".")
    cur: Any = outputs.get(port)
    for part in [p for p in path.split(".") if p]:
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


# ---- loading, validation, canonical form ------------------------------------------------------------------------------
def load_machine(src: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """A machine from a dict, a YAML/JSON file path or YAML text (``load_yaml``: no object construction, no duplicate keys)."""
    if isinstance(src, Mapping):
        return copy.deepcopy(dict(src))
    source = "<machine>"
    if isinstance(src, Path) or (isinstance(src, str) and "\n" not in src and Path(src).suffix in (".yaml", ".yml", ".json")):
        source, src = str(src), Path(src).read_text()
    data = load_yaml(src, source=source)
    if not isinstance(data, dict):
        raise MachineError(["a machine must be a mapping"])
    return data


def _family(models: Mapping[str, Any] | None, role: str) -> str | None:
    if models is None or role not in models:
        return None
    m = models[role]
    return m if isinstance(m, str) else getattr(m, "family", None)


def _sccs(nodes: list[str], edges: dict[str, set[str]]) -> list[set[str]]:
    """Tarjan's strongly connected components (iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    out: list[set[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(sorted(edges.get(root, ()))))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            nxt = next(it, None)
            if nxt is not None:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on.add(nxt)
                    work.append((nxt, iter(sorted(edges.get(nxt, ())))))
                elif nxt in on:
                    low[v] = min(low[v], index[nxt])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = set()
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.add(w)
                    if w == v:
                        break
                out.append(comp)
    return out


SESSION_HANDLER = "session"  # the handler of every session state without its own ``handler`` (``sessions.SessionPool.handler``)


def handler_key(s: Mapping[str, Any]) -> str:
    return s.get("fn") or s.get("handler") or (SESSION_HANDLER if "session" in s else s["id"])


def _validate_p1_fields(sid: str, s: Mapping[str, Any], kind: str, errors: list[str]) -> None:
    """P1 state fields: the recovery posture, per-state budgets, the session block and its captures."""
    if "mutates" in s and not isinstance(s["mutates"], bool):
        errors.append(f"state {sid} mutates must be a boolean")
    if "recovery" in s and s["recovery"] not in RECOVERY:
        errors.append(f"state {sid} recovery must be one of {list(RECOVERY)}, got {s['recovery']!r}")
    if s.get("mutates") is True and "recovery" not in s:
        errors.append(f"state {sid} mutates its tree and must declare recovery: reconcile | reset (a retry re-runs into a changed world)")
    if "budget_usd" in s and not (_is_number(s["budget_usd"]) and math.isfinite(s["budget_usd"]) and s["budget_usd"] > 0):
        errors.append(f"state {sid} budget_usd must be a positive number")
    if "budget_tokens" in s and not (_is_int(s["budget_tokens"]) and s["budget_tokens"] > 0):
        errors.append(f"state {sid} budget_tokens must be a positive integer")
    if "session" not in s:
        return
    sess = s["session"]
    if kind != "role":
        errors.append(f"state {sid} declares a session but has no role (a session is model-backed: its role names the model)")
    if not isinstance(sess, dict):
        errors.append(f"state {sid} session must be an object")
        return
    unknown = sorted(set(sess) - set(SESSION_KEYS))
    if unknown:
        errors.append(f"state {sid} session has unknown keys {unknown} (allowed: {list(SESSION_KEYS)})")
    for key in ("network", "read_only"):
        if key in sess and not isinstance(sess[key], bool):
            errors.append(f"state {sid} session.{key} must be a boolean")
    if sess.get("network") is True:
        errors.append(f"state {sid} session.network: sessions are offline (a World expert's builds run from a prepared image)")
    if not sess.get("read_only") and s.get("mutates") is not True:
        errors.append(f"state {sid} is a session that writes its worktree: declare mutates: true (and its recovery), or session.read_only")
    out_types = _port_types(s, "outputs")
    caps = sess.get("captures", [])
    if not isinstance(caps, list):
        errors.append(f"state {sid} session.captures must be a list")
        caps = []
    captured: set[str] = set()
    for j, c in enumerate(caps):
        if not isinstance(c, dict) or not _nonempty_str(c.get("output")) or c.get("kind") not in CAPTURE_KINDS:
            errors.append(f"state {sid} session.captures[{j}] needs an output and a kind in {sorted(CAPTURE_KINDS)}")
            continue
        if c["kind"] != "commits" and not _nonempty_str(c.get("path")):
            errors.append(f"state {sid} session.captures[{j}] ({c['kind']}) needs a path in the worktree")
        elif _nonempty_str(c.get("path")) and (c["path"].startswith("/") or ".." in Path(c["path"]).parts):
            errors.append(f"state {sid} session.captures[{j}] path must be relative to the worktree")
        want = CAPTURE_KINDS[c["kind"]]
        if c["output"] not in out_types:
            errors.append(f"state {sid} session.captures[{j}] captures undeclared output {c['output']!r}")
        elif out_types[c["output"]] != want:
            errors.append(f"state {sid} session.captures[{j}] kind {c['kind']!r} fills a {want} port, but {c['output']!r} is {out_types[c['output']]}")
        captured.add(c["output"])
    missing = sorted(set(out_types) - captured)
    if missing:
        errors.append(f"state {sid} session outputs {missing} have no capture (a session's outputs are files it writes, never its messages)")


def validate_machine(machine: Any, *, models: Mapping[str, Any] | None = None, handlers: Mapping[str, Any] | None = None) -> list[str]:
    """Every problem with ``machine`` (empty = valid). Ports Prime's ``validate_factory_machine`` (PA:487-688) and
    ``_validate_state_fields`` (PA:258-433) and adds: state kinds (role / fn / terminal), one entry, reachability from the
    entry, a reachable ``done`` terminal, every non-terminal state has an exit, every state on a cycle declares its
    ``max_entries``, ``isolated_from`` taint, and (with ``models``: role -> spec with a ``family``, or a family string)
    the ``different_family_from`` rule. With ``handlers``, every state's handler must exist."""
    if not isinstance(machine, dict):
        return ["a machine must be an object"]
    errors: list[str] = []
    run = machine.get("run", {})
    if not isinstance(run, dict):
        errors.append("run must be an object")
        run = {}
    for key in ("budget_s", "budget_usd"):
        if key in run and not (_is_number(run[key]) and run[key] > 0 and math.isfinite(run[key])):
            errors.append(f"run {key} must be a positive number")
    if "budget_tokens" in run and not (_is_int(run["budget_tokens"]) and run["budget_tokens"] > 0):
        errors.append("run budget_tokens must be a positive integer")
    if "max_transitions" in run and not (_is_int(run["max_transitions"]) and 0 < run["max_transitions"] <= MAX_TRANSITIONS_CAP):
        errors.append(f"run max_transitions must be a positive integer no greater than {MAX_TRANSITIONS_CAP}")
    if "failure_policy" in run and run["failure_policy"] not in FAILURE_POLICIES:
        errors.append(f"run failure_policy must be one of {list(FAILURE_POLICIES)}")
    states = machine.get("states")
    if not isinstance(states, list) or not 1 <= len(states) <= MAX_STATES:
        return [*errors, f"a machine needs a states list of 1..{MAX_STATES} states"]
    by_id: dict[str, dict[str, Any]] = {}
    for i, s in enumerate(states):
        if not isinstance(s, dict):
            errors.append(f"states[{i}] must be an object")
            continue
        sid = s.get("id")
        if not _nonempty_str(sid) or not _STATE_ID.fullmatch(sid) or sid == PARAMS:
            errors.append(f"states[{i}] id must match ^[a-z0-9][a-z0-9_-]{{0,63}}$ and not be {PARAMS!r}, got {sid!r}")
        elif sid in by_id:
            errors.append(f"states[{i}] duplicates state id {sid!r}")
        else:
            by_id[sid] = s

    for sid, s in by_id.items():
        kinds = [k for k in ("role", "fn", "terminal") if k in s]
        if len(kinds) != 1:
            errors.append(f"state {sid} must declare exactly one of role, fn, terminal (got {kinds or 'none'})")
            continue
        if "terminal" in s:
            if s["terminal"] not in TERMINALS:
                errors.append(f"state {sid} terminal must be one of {list(TERMINALS)}")
            for k in ("inputs", "outputs", "entry", "different_family_from", "isolated_from", "mutates", "recovery", "session"):
                if s.get(k):
                    errors.append(f"terminal state {sid} cannot declare {k}")
            continue
        if not _nonempty_str(s.get(kinds[0])):
            errors.append(f"state {sid} {kinds[0]} must be a non-empty string")
        if handlers is not None and handler_key(s) not in handlers:
            errors.append(f"state {sid} has no handler {handler_key(s)!r}")
        if "entry" in s and not isinstance(s["entry"], bool):
            errors.append(f"state {sid} entry must be a boolean")
        if "max_entries" in s and not (_is_int(s["max_entries"]) and 1 <= s["max_entries"] <= MAX_ENTRIES_CAP):
            errors.append(f"state {sid} max_entries must be an integer in 1..{MAX_ENTRIES_CAP}")
        if "retries" in s and not (_is_int(s["retries"]) and 0 <= s["retries"] <= MAX_RETRIES):
            errors.append(f"state {sid} retries must be an integer between 0 and {MAX_RETRIES}")
        for key in ("timeout_s", "estimate_usd"):
            if key in s and not (_is_number(s[key]) and math.isfinite(s[key]) and (s[key] > 0 if key == "timeout_s" else s[key] >= 0)):
                errors.append(f"state {sid} {key} must be a {'positive' if key == 'timeout_s' else 'non-negative'} number")
        if "failure_policy" in s and s["failure_policy"] not in FAILURE_POLICIES:
            errors.append(f"state {sid} failure_policy must be one of {list(FAILURE_POLICIES)}")
        if "canary" in s and not isinstance(s["canary"], bool):
            errors.append(f"state {sid} canary must be a boolean")
        _validate_p1_fields(sid, s, kinds[0], errors)
        for key in ("foreach", "wait", "lifecycle"):
            if key in s:
                errors.append(f"state {sid}: {key} is not supported (one token runs at a time; parallelism lives inside a state)")
        # ports (PA:337-433)
        seen: set[str] = set()
        for j, out in enumerate(_ports(s, "outputs")):
            if not isinstance(out, dict) or not _nonempty_str(out.get("name")):
                errors.append(f"state {sid} outputs[{j}] requires a non-empty name")
                continue
            if out["name"] in seen:
                errors.append(f"state {sid} declares duplicate output name {out['name']!r}")
            seen.add(out["name"])
            if out.get("type") not in PORT_TYPES:
                errors.append(f"state {sid} output {out['name']!r} type must be one of {list(PORT_TYPES)}")
            if "schema" in out:
                from jsonschema import Draft202012Validator
                from jsonschema.exceptions import SchemaError

                try:
                    Draft202012Validator.check_schema(out["schema"])
                except SchemaError as e:
                    errors.append(f"state {sid} output {out['name']!r} schema is not a JSON schema: {e.message}"[:300])
        if s.get("outputs") is not None and not isinstance(s.get("outputs"), list):
            errors.append(f"state {sid} outputs must be a list")
        seen = set()
        for j, inp in enumerate(_ports(s, "inputs")):
            if not isinstance(inp, dict) or not _nonempty_str(inp.get("name")):
                errors.append(f"state {sid} inputs[{j}] requires a non-empty name")
                continue
            name, ptype, source = inp["name"], inp.get("type"), inp.get("from")
            if name in seen:
                errors.append(f"state {sid} declares duplicate input name {name!r}")
            seen.add(name)
            if name == "findings":
                errors.append(f"state {sid} input name 'findings' is reserved for loop-back findings")
            if "optional" in inp and not isinstance(inp["optional"], bool):
                errors.append(f"state {sid} input {name!r} optional must be a boolean")
            if ptype not in PORT_TYPES:
                errors.append(f"state {sid} input {name!r} type must be one of {list(PORT_TYPES)}")
            if not isinstance(source, str) or "." not in source:
                errors.append(f"state {sid} input {name!r} requires 'from' of the form '<state>.<output>' or 'params.<name>'")
                continue
            src_id, _, src_out = source.partition(".")
            if src_id == PARAMS:
                continue
            if src_id not in by_id:
                errors.append(f"state {sid} input {name!r} references unknown state {src_id!r}")
                continue
            if src_id == sid and not inp.get("optional"):  # PA:671-683: a required self-input can never bind
                errors.append(f"state {sid} input {name!r} cannot require itself: mark the self-input optional")
            src_types = _port_types(by_id[src_id], "outputs")
            if src_out not in src_types:
                errors.append(f"state {sid} input {name!r} references output {src_out!r} that state {src_id!r} does not declare")
            elif ptype in PORT_TYPES and src_types[src_out] != ptype:
                errors.append(f"state {sid} input {name!r} of type {ptype!r} cannot read output {src_out!r} of type {src_types[src_out]!r}")
        if s.get("inputs") is not None and not isinstance(s.get("inputs"), list):
            errors.append(f"state {sid} inputs must be a list")

    entries = [sid for sid, s in by_id.items() if s.get("entry") is True]
    if by_id and len(entries) != 1:
        errors.append(f"a machine needs exactly one entry state, got {entries}")
    elif entries:
        bad = [i.get("from") for i in _ports(by_id[entries[0]], "inputs") if isinstance(i, dict)
               and not str(i.get("from", "")).startswith(PARAMS + ".") and not i.get("optional")]
        if bad:
            errors.append(f"entry state {entries[0]} may only require params inputs, got {bad}")

    # transitions (PA:584-688)
    transitions = machine.get("transitions") or []
    if not isinstance(transitions, list):
        return [*errors, "transitions must be a list"]
    edges: dict[str, set[str]] = {sid: set() for sid in by_id}
    for i, t in enumerate(transitions):
        where = f"transitions[{i}]"
        if not isinstance(t, dict):
            errors.append(f"{where} must be an object")
            continue
        src, dst = t.get("from"), t.get("to")
        if isinstance(src, list):
            errors.append(f"{where}: joins (a list of from-states) are not supported; one token runs at a time")
            continue
        if src not in by_id:
            errors.append(f"{where} references unknown from-state {src!r}")
            continue
        if dst not in by_id:
            errors.append(f"{where} references unknown to-state {dst!r}")
            continue
        if "terminal" in by_id[src]:
            errors.append(f"{where} cannot leave terminal state {src!r}")
            continue
        edges[src].add(dst)
        src_types = _port_types(by_id[src], "outputs")
        when = t.get("when")
        if when is not None:
            try:
                when = _compile_guard(when)
            except GuardSyntaxError as e:
                errors.append(f"{where} guard does not parse: {e}")
                when = None
            if when is not None:
                _validate_guard(when, where, src_types, errors)
        carry = t.get("carry")
        if carry is not None and (not _nonempty_str(carry) or carry.partition(".")[0] not in src_types):
            errors.append(f"{where} carry {carry!r} must name a declared output (optionally .path) of state {src!r}")

    # graph rules (new: Prime has no reachability rule and lets a dead state sit unreachable)
    if entries and len(entries) == 1:
        seen_r, todo = {entries[0]}, [entries[0]]
        while todo:
            for n in edges.get(todo.pop(), ()):
                if n not in seen_r:
                    seen_r.add(n)
                    todo.append(n)
        unreachable = sorted(set(by_id) - seen_r)
        if unreachable:
            errors.append(f"states unreachable from the entry: {unreachable}")
        if not any(by_id[n].get("terminal") == "done" for n in seen_r):
            errors.append("no 'done' terminal state is reachable from the entry")
    for sid, s in by_id.items():
        if "terminal" not in s and not edges.get(sid):
            errors.append(f"state {sid} has no outgoing transition (a settle there could only escalate)")
    for comp in _sccs(sorted(by_id), edges):
        cyclic = len(comp) > 1 or any(n in edges.get(n, ()) for n in comp)
        if cyclic:
            loose = sorted(n for n in comp if "max_entries" not in by_id[n])
            if loose:
                errors.append(f"states {loose} are on a cycle {sorted(comp)} and must declare max_entries")

    # staging: an isolated state reads nothing that (transitively) read its sources
    reads = {sid: {str(i.get("from", "")).partition(".")[0] for i in _ports(s, "inputs") if isinstance(i, dict)} - {PARAMS}
             for sid, s in by_id.items()}
    for sid, s in by_id.items():
        iso = s.get("isolated_from")
        if iso is None:
            continue
        if not isinstance(iso, list) or not all(x in by_id for x in iso):
            errors.append(f"state {sid} isolated_from must list existing states")
            continue
        seen_t, todo = set(), list(reads.get(sid, ()))
        while todo:
            n = todo.pop()
            if n in seen_t or n == sid:
                continue
            seen_t.add(n)
            todo.extend(reads.get(n, ()))
        leak = sorted(seen_t & set(iso))
        if leak:
            errors.append(f"state {sid} is isolated from {leak} but reads them through its inputs")

    # the cross-family rule (new: Prime leaves it to convention)
    for sid, s in by_id.items():
        dff = s.get("different_family_from")
        if dff is None:
            continue
        if "role" not in s:
            errors.append(f"state {sid} declares different_family_from but has no role (only model-backed states have a family)")
            continue
        if not isinstance(dff, list) or not all(x in by_id and "role" in by_id[x] for x in dff):
            errors.append(f"state {sid} different_family_from must list role states")
            continue
        if models is None:
            continue
        mine = _family(models, s["role"])
        if mine is None:
            errors.append(f"state {sid} role {s['role']!r} has no model assignment")
            continue
        for other in dff:
            theirs = _family(models, by_id[other]["role"])
            if theirs is None:
                errors.append(f"state {other} role {by_id[other]['role']!r} has no model assignment")
            elif theirs == mine:
                errors.append(f"cross-family rule: state {sid} (role {s['role']!r}) and state {other} (role {by_id[other]['role']!r}) "
                              f"are both assigned model family {mine!r}")
    return errors


def canonicalize(machine: Any, *, models: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and normalise (PORT of ``_canonicalize_machine`` PA:827-875 and ``canonicalize_factory_spec`` PA:878-895):
    defaults applied, guard strings compiled to dicts. Raises ``MachineError``."""
    errors = validate_machine(machine, models=models)
    if errors:
        raise MachineError(errors)
    run_in = machine.get("run") or {}
    policy = run_in.get("failure_policy", "escalate")
    run: dict[str, Any] = {"failure_policy": policy,
                           "max_transitions": run_in.get("max_transitions", min(TRANSITIONS_PER_STATE_DEFAULT * len(machine["states"]), MAX_TRANSITIONS_CAP))}
    for key in ("budget_s", "budget_usd", "budget_tokens"):
        if key in run_in:
            run[key] = run_in[key]
    states_out = []
    for s in machine["states"]:
        if "terminal" in s:
            states_out.append({"id": s["id"], "terminal": s["terminal"]})
            continue
        out = {"id": s["id"], "entry": bool(s.get("entry", False)), "max_entries": s.get("max_entries", STATE_MAX_ENTRIES_DEFAULT),
               "retries": s.get("retries", 0), "failure_policy": s.get("failure_policy", policy), "estimate_usd": float(s.get("estimate_usd", 0.0)),
               "canary": bool(s.get("canary", False)), "inputs": copy.deepcopy(_ports(s, "inputs")), "outputs": copy.deepcopy(_ports(s, "outputs"))}
        if s.get("mutates"):  # only when set: a v1 machine keeps its hash (its journals still resume)
            out["mutates"] = True
        for key in ("role", "fn", "handler", "timeout_s", "different_family_from", "isolated_from", "doc", "recovery", "budget_usd", "budget_tokens",
                    "session"):
            if key in s:
                out[key] = copy.deepcopy(s[key])
        states_out.append(out)
    transitions = []
    for t in machine.get("transitions") or []:
        o = {"from": t["from"], "to": t["to"]}
        if t.get("when") is not None:
            o["when"] = _compile_guard(t["when"])
        if t.get("carry"):
            o["carry"] = t["carry"]
        transitions.append(o)
    return {"run": run, "states": states_out, "transitions": transitions}


def machine_hash(canonical: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


# ---- executor ---------------------------------------------------------------------------------------------------------
@dataclass
class Settle:
    """What a state function returns: its declared outputs and the findings it hands to a loop-back target. ``tokens``:
    model tokens the state used that were not booked with ``book_tokens``."""

    outputs: dict[str, Any]
    findings: list[str] = field(default_factory=list)
    tokens: int = 0


@dataclass
class StateContext:
    run_id: str
    state_id: str
    entry: int  # 1-based activation number of this state
    attempt: int  # 1-based try within this entry (retries re-run the same inputs, never new findings)
    inputs: dict[str, Any]
    findings: list[str]  # carried by the transition that entered this state (loop-back)
    params: dict[str, Any]
    role: str | None = None
    model: Any = None  # the ModelSpec of the role (models.py), when assigned
    client: Any = None  # the role's LLM client (real or fake)
    services: Any = None  # whatever the machine's owner injects (the driver's env)
    run_dir: Path | None = None
    # ---- P1
    state: dict[str, Any] = field(default_factory=dict)  # the canonical state (its session block, captures, budgets)
    recovery: dict[str, Any] | None = None  # a re-run of a mutating state: {"mode", "snapshot", "findings", "diff"?}
    budget: dict[str, Any] = field(default_factory=dict)  # what is left: {"usd": float | None, "tokens": int | None}
    respawned: bool = False  # this is the one free re-run after a verified infra death
    artifact_root: Path | None = None  # where artifact ports resolve (the task repo)


Handler = Callable[[StateContext], Awaitable["Settle | dict[str, Any]"]]


@dataclass
class RunResult:
    run_id: str
    status: Status
    reason: str | None
    outputs: dict[str, dict[str, Any]]  # state id -> latest settled outputs
    decision_request: dict[str, Any] | None
    transitions: int
    usd: float
    seconds: float
    entries: dict[str, int]
    path: Path
    tokens: int = 0
    needs_verify: list[dict[str, Any]] = field(default_factory=list)  # provisional outputs of attempts whose capture failed


class StateFailed(RuntimeError):
    pass


async def _no_sleep(_s: float) -> None:
    return None


class MachineRun:
    """One run of a canonical machine, journaled under ``run_dir``. ``await run()`` starts it, or continues it from the
    journal after a crash; ``await resume(decision)`` answers a pause. A different machine (hash) on the same journal
    is refused.

    P1 hooks: ``trees`` (a mapping or a callable, sync or async: state id -> ``MutableTree``) for mutating states;
    ``artifact_root`` for artifact ports; ``infra_probe(state, error) -> "alive" | "dead"`` (dead: the probe has torn the
    zombie down); ``sleep`` (injected: the infra quiet period and the re-capture window; default ``asyncio.sleep``)."""

    def __init__(self, machine: Mapping[str, Any], *, run_dir: Path, handlers: Mapping[str, Handler], run_id: str | None = None,
                 params: Mapping[str, Any] | None = None, models: Mapping[str, Any] | None = None, clients: Mapping[str, Any] | None = None,
                 ledger: Ledger | None = None, services: Any = None, clock: Callable[[], float] = time.monotonic,
                 trees: Mapping[str, MutableTree] | Callable[[str], Any] | None = None, artifact_root: Path | None = None,
                 infra_probe: InfraProbe | None = None, sleep: Callable[[float], Awaitable[None]] | None = None,
                 infra_quiet_s: float = INFRA_QUIET_S, recapture_wait_s: float = RECAPTURE_WAIT_S):
        self.machine = canonicalize(machine, models=models)
        errors = validate_machine(self.machine, models=models, handlers=handlers)
        if errors:
            raise MachineError(errors)
        self.hash = machine_hash(self.machine)
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.handlers, self.models, self.clients, self.services, self.clock = dict(handlers), models, dict(clients or {}), services, clock
        self.trees, self.artifact_root, self.infra_probe = trees, (Path(artifact_root) if artifact_root else None), infra_probe
        self.sleep = sleep or asyncio.sleep
        self.infra_quiet_s, self.recapture_wait_s = infra_quiet_s, recapture_wait_s
        self.states = {s["id"]: s for s in self.machine["states"]}
        self.transitions = self.machine["transitions"]
        self.runcfg = self.machine["run"]
        self.journal = Journal(self.run_dir / "journal.jsonl")
        self.own_ledger = ledger is None
        self.ledger = ledger if ledger is not None else Ledger(cap_usd=self.runcfg.get("budget_usd", math.inf))
        self.params = dict(params or {})
        self._replay()
        if not self.started:
            self._log({"event": "start", "machine_hash": self.hash, "run_id": self.run_id, "params": self.params})
            self.started = True

    # ---- journal replay -----------------------------------------------------------------------------------------------
    def _events(self) -> list[dict[str, Any]]:
        p = self.journal.path
        out = []
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _replay(self) -> None:
        self.started = False
        self.status: Status = "running"
        self.reason: str | None = None
        self.decision_request: dict[str, Any] | None = None
        self.entries: dict[str, int] = {}
        self.grants: dict[str, int] = {}
        self.extra_transitions = 0
        self.latest: dict[str, dict[str, Any]] = {}
        self.latest_findings: dict[str, list[str]] = {}
        self.canaries: dict[str, str] = {}
        self.fired = 0
        self.usd = 0.0
        self.seconds = 0.0
        self.tokens = 0
        self.state_usd: dict[str, float] = {}
        self.state_tokens: dict[str, int] = {}
        self.state_grants: dict[str, dict[str, float]] = {}
        self.snapshots: dict[str, str] = {}  # "<state>#<entry>" -> tree snapshot token
        self.fails: dict[str, list[str]] = {}  # "<state>#<entry>" -> failed attempts' errors (reconcile findings)
        self.needs_verify: list[dict[str, Any]] = []
        entry = next(s for s in self.states.values() if s.get("entry"))
        self.pending: dict[str, Any] | None = {"state": entry["id"], "findings": [], "via": None}
        self.inflight: dict[str, Any] | None = None
        self.unrouted: str | None = None
        for ev in self._events():
            kind = ev.get("event")
            if kind == "start":
                if ev.get("machine_hash") != self.hash:
                    raise MachineError([f"journal {self.journal.path} belongs to machine {ev.get('machine_hash')}, not {self.hash}"])
                self.started = True
                self.params = ev.get("params") or {}  # the journal's params win: a resumed run is the same run
            elif kind == "enter":
                self.entries[ev["state"]] = max(self.entries.get(ev["state"], 0), ev["entry"])
                self.inflight = {"state": ev["state"], "entry": ev["entry"], "findings": ev.get("findings", []), "via": ev.get("via")}
                self.pending = None
            elif kind == "settle":
                self.latest[ev["state"]] = ev["outputs"]
                self.latest_findings[ev["state"]] = ev.get("findings", [])
                if ev.get("canary"):
                    self.canaries[ev["state"]] = ev["canary"]
                self.inflight = None
                self.unrouted = ev["state"]
            elif kind == "spend":
                self._count(ev.get("state"), ev.get("usd", 0.0), ev.get("seconds", 0.0), ev.get("tokens", 0))
            elif kind == "snapshot":
                self.snapshots[f"{ev['state']}#{ev['entry']}"] = ev["token"]
            elif kind == "fail":
                self.fails.setdefault(f"{ev['state']}#{ev['entry']}", []).append(str(ev.get("error", "")))
            elif kind == "capture_failed":
                self.needs_verify.append({k: ev.get(k) for k in ("state", "entry", "attempt", "error", "outputs")})
            elif kind == "transition":
                self.fired += 1
                self.pending = {"state": ev["to"], "findings": ev.get("findings", []), "via": ev["from"]}
                self.unrouted = None
            elif kind == "status":
                self.status, self.reason, self.decision_request = ev["status"], ev.get("reason"), ev.get("decision_request")
                self.pending = ev.get("pending")
                if ev["status"] != "running":
                    self.inflight, self.unrouted = None, None
            elif kind == "grant":
                self._apply_grant(ev)
            elif kind == "decision":
                self.decision_request = None
        if self.inflight is not None:  # a crash mid-state: that entry runs again (same entry number, same findings)
            self.pending = {**self.inflight, "resume_entry": self.inflight["entry"]}
            self.inflight = None
        if self.own_ledger:
            self.ledger.committed = self.usd

    def _count(self, sid: str | None, usd: float, secs: float, tokens: int) -> None:
        self.usd += usd
        self.seconds += secs
        self.tokens += int(tokens or 0)
        if sid:
            self.state_usd[sid] = self.state_usd.get(sid, 0.0) + usd
            self.state_tokens[sid] = self.state_tokens.get(sid, 0) + int(tokens or 0)

    def _apply_grant(self, ev: Mapping[str, Any]) -> None:
        for k, v in (ev.get("entries") or {}).items():
            self.grants[k] = self.grants.get(k, 0) + v
        self.extra_transitions += ev.get("transitions", 0)
        for k, v in (ev.get("state_budget") or {}).items():
            g = self.state_grants.setdefault(k, {"usd": 0.0, "tokens": 0})
            g["usd"] += float(v.get("usd", 0.0))
            g["tokens"] += int(v.get("tokens", 0))

    def _log(self, ev: dict[str, Any]) -> None:
        self.journal.append(ev)

    def _set_status(self, status: Status, reason: str | None = None, *, request: dict[str, Any] | None = None) -> None:
        self.status, self.reason, self.decision_request = status, reason, request
        self._log({"event": "status", "status": status, "reason": reason, "decision_request": request, "pending": self.pending})

    def _escalate(self, kind: str, reason: str, *, state: str | None, findings: list[str] | None = None,
                  options: tuple[str, ...] = ("revise", "drop")) -> None:
        req = {"id": f"{self.run_id}:{kind}:{self.fired}", "subject": self.run_id, "kind": kind, "state": state, "reason": reason,
               "options": list(options), "findings": list(findings or [])[:20], "machine_hash": self.hash}
        self._set_status("paused", reason, request=req)

    # ---- public API ---------------------------------------------------------------------------------------------------
    def result(self) -> RunResult:
        return RunResult(self.run_id, self.status, self.reason, copy.deepcopy(self.latest), copy.deepcopy(self.decision_request),
                         self.fired, round(self.usd, 6), round(self.seconds, 3), dict(self.entries), self.run_dir, self.tokens,
                         copy.deepcopy(self.needs_verify))

    def status_view(self) -> dict[str, Any]:
        r = self.result()
        return {"run_id": r.run_id, "status": r.status, "reason": r.reason, "pending": self.pending, "entries": r.entries,
                "transitions": r.transitions, "usd": r.usd, "tokens": r.tokens, "seconds": r.seconds, "decision_request": r.decision_request,
                "needs_verify": r.needs_verify}

    async def run(self) -> RunResult:
        if self.status == "stopped" and self._caps_ok() is None:  # a stop whose cause is gone (a larger cap): continue
            self._set_status("running", "caps raised")
        if self.status == "running":
            await self._loop()
        return self.result()

    async def resume(self, decision: Mapping[str, Any] | None = None) -> RunResult:
        """Answer a pause. ``decision`` follows ``cards.Decision`` (``action`` keep|drop|revise|approve|reject, ``actor``,
        ``reason``); ``drop``/``reject`` end the run ``rejected``; anything else grants ``extra_entries`` (default 1) to
        the blocked state, or ``extra_transitions`` (default 10) after the transition cap, or (``state_budget``)
        ``extra_usd`` / ``extra_tokens`` (default: one more of the state's budget), and continues. A ``failed`` or
        ``infra`` pause re-runs the same entry with fresh retries and a fresh free respawn; ``capture_failed`` re-runs
        the producer."""
        if self.status not in ("paused", "stopped"):
            raise RuntimeError(f"run {self.run_id} is {self.status}; only a paused or stopped run resumes")
        d = dict(decision or {"action": "revise", "actor": "system", "reason": "resume"})
        req = self.decision_request or {}
        self._log({"event": "decision", "request": req.get("id"), "decision": d})
        if d.get("action") in ("drop", "reject"):
            self.pending = None
            self._set_status("rejected", f"decision: {d.get('action')} by {d.get('actor')}: {d.get('reason', '')}"[:300])
            return self.result()
        grant: dict[str, Any] = {"event": "grant", "entries": {}, "transitions": 0}
        if req.get("kind") in ("exhausted", "capture_failed") and req.get("state"):
            # the blocked state and every state it can reach get the extra entries: a re-entered owner re-runs the states
            # downstream of it, whose bounds were sized for the original loop
            for sid in self._reachable(req["state"]):
                grant["entries"][sid] = int(d.get("extra_entries", 1))
        if req.get("kind") == "transition_cap":
            grant["transitions"] = int(d.get("extra_transitions", 10))
        if req.get("kind") == "no_transition" and req.get("state"):
            grant["entries"][req["state"]] = int(d.get("extra_entries", 1))  # the state runs again
        if req.get("kind") == "state_budget" and req.get("state"):
            st = self.states[req["state"]]
            grant["state_budget"] = {req["state"]: {"usd": float(d.get("extra_usd", st.get("budget_usd", 0.0))),
                                                    "tokens": int(d.get("extra_tokens", st.get("budget_tokens", 0)))}}
        # a "failed" / "infra" pause re-runs the same entry (resume_entry): no grant needed
        self._log(grant)
        self._apply_grant(grant)
        self._set_status("running", f"resumed by {d.get('actor')}")
        await self._loop()
        return self.result()

    # ---- the loop -----------------------------------------------------------------------------------------------------
    def _reachable(self, sid: str) -> list[str]:
        seen, todo = {sid}, [sid]
        while todo:
            cur = todo.pop()
            for t in self.transitions:
                if t["from"] == cur and t["to"] not in seen and "terminal" not in self.states[t["to"]]:
                    seen.add(t["to"])
                    todo.append(t["to"])
        return sorted(seen)

    def _max_entries(self, sid: str) -> int:
        return self.states[sid]["max_entries"] + self.grants.get(sid, 0)

    def _caps_ok(self) -> str | None:
        if self.ledger.exhausted:
            return f"budget: the USD cap {self.ledger.cap_usd:.4f} is spent ({self.ledger.committed:.4f} committed)"
        if "budget_tokens" in self.runcfg and self.tokens >= self.runcfg["budget_tokens"]:
            return f"budget: the token cap {self.runcfg['budget_tokens']} is spent ({self.tokens} tokens)"
        if "budget_s" in self.runcfg and self.seconds >= self.runcfg["budget_s"]:
            return f"wall clock: {self.seconds:.1f}s of work reached the run budget {self.runcfg['budget_s']}s"
        return None

    def _state_caps(self, state: Mapping[str, Any]) -> tuple[float | None, int | None]:
        g = self.state_grants.get(state["id"], {})
        usd = state["budget_usd"] + g.get("usd", 0.0) if "budget_usd" in state else None
        tok = state["budget_tokens"] + int(g.get("tokens", 0)) if "budget_tokens" in state else None
        return usd, tok

    def _state_budget_spent(self, state: Mapping[str, Any]) -> str | None:
        sid = state["id"]
        usd, tok = self._state_caps(state)
        if usd is not None and self.state_usd.get(sid, 0.0) >= usd - 1e-12:
            return f"state {sid} spent its budget_usd {usd:.4f} ({self.state_usd.get(sid, 0.0):.4f})"
        if tok is not None and self.state_tokens.get(sid, 0) >= tok:
            return f"state {sid} spent its budget_tokens {tok} ({self.state_tokens.get(sid, 0)})"
        return None

    def _budget_left(self, state: Mapping[str, Any]) -> dict[str, Any]:
        sid = state["id"]
        usd, tok = self._state_caps(state)
        left_usd = None if usd is None else max(0.0, usd - self.state_usd.get(sid, 0.0))
        if math.isfinite(self.ledger.cap_usd):
            run_left = max(0.0, self.ledger.cap_usd - self.ledger.committed)
            left_usd = run_left if left_usd is None else min(left_usd, run_left)
        left_tok = None if tok is None else max(0, tok - self.state_tokens.get(sid, 0))
        if "budget_tokens" in self.runcfg:
            run_left_t = max(0, self.runcfg["budget_tokens"] - self.tokens)
            left_tok = run_left_t if left_tok is None else min(left_tok, run_left_t)
        return {"usd": left_usd, "tokens": left_tok}

    def _bind(self, state: Mapping[str, Any]) -> dict[str, Any]:
        bound: dict[str, Any] = {}
        for inp in state["inputs"]:
            src, _, out = inp["from"].partition(".")
            if src == PARAMS:
                val, ok = self.params.get(out), out in self.params
            else:
                ok = src in self.latest and out in self.latest[src]
                val = self.latest[src][out] if ok else None
            if not ok and not inp.get("optional"):
                raise StateFailed(f"input {inp['name']!r} from {inp['from']} is not bound (its source has not settled)")
            if ok and val is not None and inp["type"] == "artifact" and src != PARAMS:
                why = artifact_problem(self.artifact_root, val)
                if why:  # the producer's capture is gone: its failure, not this consumer's (F:2969-2988)
                    raise CaptureLost(src, f"input {inp['name']!r} from {inp['from']}: {why}")
            bound[inp["name"]] = val
        return bound

    def _screen_findings(self, sid: str, findings: list[str]) -> list[str]:
        """Findings carried into an isolated state lose every line that carries an isolated source's canary or quotes
        its outputs (staging, not prompts, decides access: forge orchestrator canaries and forge/repo quote marks)."""
        iso = self.states[sid].get("isolated_from") or []
        if not iso or not findings:
            return findings
        marks: set[str] = set()
        canaries = []
        for src in iso:
            if src in self.latest:
                marks |= _marks(json.dumps(self.latest[src], sort_keys=True, default=str), QUOTE_SHINGLE)
                marks |= _marks("\n".join(str(v) for v in self.latest[src].values() if isinstance(v, str)), QUOTE_SHINGLE)
            if src in self.canaries:
                canaries.append(self.canaries[src])
        own = set()
        for v in (self._bind_quiet(sid) or {}).values():  # material the state may see anyway is not a leak
            own |= _marks(v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str), QUOTE_SHINGLE)
        kept, withheld = [], []
        for f in findings:
            text = str(f)
            if any(c in text for c in canaries) or (_marks(text, QUOTE_SHINGLE) & (marks - own)):
                withheld.append(text[:200])
            else:
                kept.append(text)
        if withheld:
            self._log({"event": "breach_withheld", "state": sid, "count": len(withheld), "sources": iso})
            kept.append(f"{len(withheld)} finding(s) withheld: they quoted material this state may not see")
        return kept

    def _bind_quiet(self, sid: str) -> dict[str, Any] | None:
        try:
            return self._bind(self.states[sid])
        except (StateFailed, CaptureLost):
            return None

    def _check_outputs(self, state: Mapping[str, Any], outputs: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        """(problems, capture problems): a missing port or a dead artifact is a capture failure of this producer."""
        problems, capture = [], []
        for out in state["outputs"]:
            name = out["name"]
            if name not in outputs:
                capture.append(f"missing declared output {name!r}")
                continue
            v = outputs[name]
            if out["type"] == "text" and not isinstance(v, str):
                problems.append(f"output {name!r} must be text, got {type(v).__name__}")
            elif out["type"] in ("json", "artifact"):
                try:
                    size = len(json.dumps(v, allow_nan=False))
                except (TypeError, ValueError) as e:
                    problems.append(f"output {name!r} is not finite JSON: {e}")
                    continue
                if size > OUTPUT_CAP_BYTES:
                    problems.append(f"output {name!r} is {size} bytes (cap {OUTPUT_CAP_BYTES})")
                    continue
                if isinstance(out.get("schema"), dict):  # a declared port schema is checked where the value is made
                    from jsonschema import Draft202012Validator

                    errs = [e.message for e in Draft202012Validator(out["schema"]).iter_errors(v)]
                    if errs:
                        problems.append(f"output {name!r} breaks its schema: {'; '.join(errs[:3])}"[:500])
                if out["type"] == "artifact":
                    why = artifact_problem(self.artifact_root, v)
                    if why:
                        capture.append(f"output {name!r}: {why}")
        extra = sorted(set(outputs) - {o["name"] for o in state["outputs"]})
        if extra:
            problems.append(f"undeclared outputs {extra}")
        return problems, capture

    async def _loop(self) -> None:
        if self.unrouted is not None and self.status == "running":  # a crash between a settle and its transition
            sid = self.unrouted
            self._route(sid, Settle(self.latest[sid], self.latest_findings.get(sid, [])))
        while self.status == "running":
            why = self._caps_ok()
            if why:
                self._set_status("stopped", why)
                return
            if self.pending is None:
                self._set_status("failed", "nothing pending in a running machine (dead configuration)")
                return
            sid = self.pending["state"]
            state = self.states[sid]
            if "terminal" in state:
                self.pending = None
                self._set_status(state["terminal"], f"reached terminal state {sid}")
                return
            entry = self.pending.get("resume_entry") or self.entries.get(sid, 0) + 1
            if entry > self._max_entries(sid):  # defense in depth: transitions check this before firing
                self._escalate("exhausted", f"state {sid} would exceed max_entries {self._max_entries(sid)}", state=sid,
                               findings=self.pending.get("findings"))
                return
            findings = self._screen_findings(sid, list(self.pending.get("findings") or []))
            rerun = bool(self.pending.get("resume_entry"))
            self.entries[sid] = entry
            self._log({"event": "enter", "state": sid, "entry": entry, "findings": findings, "via": self.pending.get("via"),
                       "resumed": rerun})
            self.pending = None
            settle = await self._run_state(state, entry, findings, rerun=rerun)
            if self.status != "running":
                return
            if settle is not None:
                self._route(sid, settle)

    def _route(self, sid: str, settle: Settle) -> None:
        """Fire the first transition out of ``sid`` whose guard passes on its settle; every blocked case escalates."""
        self.unrouted = None
        cap = self.runcfg["max_transitions"]
        fired = None
        for i, t in enumerate(self.transitions):
            if t["from"] == sid and guard_passes(t.get("when"), settle.outputs):
                fired = (i, t)
                break
        if fired is None:
            self.pending = {"state": sid, "findings": settle.findings, "via": sid}
            self._escalate("no_transition", f"no transition out of {sid} matched its settle", state=sid, findings=settle.findings)
            return
        i, t = fired
        carried = settle.findings
        if t.get("carry"):
            c = _resolve_path(settle.outputs, t["carry"])
            carried = [str(x) for x in c] if isinstance(c, list) else ([str(c)] if c not in (None, "") else [])
        to = t["to"]
        if self.fired >= cap + self.extra_transitions:
            self.pending = {"state": to, "findings": carried, "via": sid}
            self._escalate("transition_cap", f"run reached max_transitions {cap + self.extra_transitions}", state=to, findings=carried)
            return
        if "terminal" not in self.states[to] and self.entries.get(to, 0) >= self._max_entries(to):
            # Prime records this as an event and later finalises the run as done (F:2469-2483, :3515-3541; TF:5450-5477
            # pins it at 7cbffa77b): the defect V7 forbids
            self.pending = {"state": to, "findings": carried, "via": sid}
            self._escalate("exhausted", f"loop exhausted: {sid} -> {to}, but {to} used all {self._max_entries(to)} entries",
                           state=to, findings=carried)
            return
        self.fired += 1
        self._log({"event": "transition", "from": sid, "to": to, "index": i, "findings": carried})
        self.pending = {"state": to, "findings": carried, "via": sid}

    # ---- P1: trees, recovery, infra -------------------------------------------------------------------------------------
    async def _tree(self, sid: str) -> MutableTree:
        t: Any = None
        if isinstance(self.trees, Mapping):
            t = self.trees.get(sid)
        elif callable(self.trees):
            t = self.trees(sid)
            if asyncio.iscoroutine(t):
                t = await t
        if t is None:
            raise StateFailed(f"state {sid} mutates its tree, but the run gives it no tree (MachineRun(trees=...))")
        return t

    async def _recover(self, state: Mapping[str, Any], entry: int, rerun: bool) -> dict[str, Any] | None:
        """Snapshot before an entry's first attempt; on a re-run apply the declared posture (F:5099-5107, enforced)."""
        if not state.get("mutates"):
            return None
        sid, key = state["id"], f"{state['id']}#{entry}"
        tree = await self._tree(sid)
        snap = self.snapshots.get(key)
        findings = list(self.fails.get(key, []))[-5:]
        if snap is None:
            snap = await tree.snapshot(f"{self.run_id}/{key}")
            self.snapshots[key] = snap
            self._log({"event": "snapshot", "state": sid, "entry": entry, "token": snap})
            # a crash between the entry and its snapshot leaves nothing to recover to: the attempt starts from the tree
            return None if not rerun else {"mode": state["recovery"], "snapshot": snap, "findings": findings, "diff": "",
                                           "note": "no snapshot was taken before the interrupted attempt"}
        if not rerun:
            return None
        if state["recovery"] == "reset":
            await tree.restore(snap)
            self._log({"event": "recovery", "state": sid, "entry": entry, "mode": "reset", "snapshot": snap})
            return {"mode": "reset", "snapshot": snap, "findings": findings}
        diff = await tree.diff(snap)
        self._log({"event": "recovery", "state": sid, "entry": entry, "mode": "reconcile", "snapshot": snap, "diff_chars": len(diff),
                   "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()})
        cut = diff if len(diff) <= RECONCILE_DIFF_CAP else diff[:RECONCILE_DIFF_CAP] + f"\n... [{len(diff) - RECONCILE_DIFF_CAP} chars cut]"
        return {"mode": "reconcile", "snapshot": snap, "findings": findings, "diff": cut}

    async def _recheck_infra(self, sid: str, err: BaseException) -> str:
        """F:2990-3017 adapted: a quiet period, then one re-check. ``alive``: the sandbox answered (re-run the same
        session); ``dead`` (or no probe, or a probe that fails): the probe has torn the zombie down (F:3036-3046)."""
        await self.sleep(self.infra_quiet_s)
        if self.infra_probe is None:
            return "dead"
        try:
            v = await self.infra_probe(sid, err)
        except Exception:  # noqa: BLE001 -- a probe that cannot answer is as dead as the error said
            return "dead"
        return v if v in ("alive", "dead") else "dead"

    def _capture_lost(self, sid: str, entry: int, err: CaptureLost) -> None:
        self._log({"event": "capture_lost", "state": sid, "entry": entry, "producer": err.producer, "error": str(err)[:500]})
        self.pending = {"state": err.producer, "findings": [f"capture lost: {err}"[:500]], "via": sid}
        self._escalate("capture_failed", f"the producer {err.producer} must run again: {err}"[:500], state=err.producer,
                       findings=[str(err)[:500]])

    async def _run_state(self, state: Mapping[str, Any], entry: int, findings: list[str], *, rerun: bool = False) -> Settle | None:
        sid = state["id"]
        role = state.get("role")
        handler = self.handlers[handler_key(state)]
        attempts = state["retries"] + 1
        key = f"{sid}#{entry}"
        last, blame, over_budget = "", "task", None
        respawned = False
        attempt = 1
        while attempt <= attempts:
            try:
                inputs = self._bind(state)
            except CaptureLost as e:
                self._capture_lost(sid, entry, e)
                return None
            except StateFailed as e:
                last = str(e)
                break
            over_budget = self._state_budget_spent(state)
            if over_budget:
                last = over_budget
                break
            who = f"{self.run_id}/{sid}#{entry}.{attempt}{'r' if respawned else ''}"
            try:
                self.ledger.reserve(who, state["estimate_usd"])
            except BudgetExhausted as e:
                self.pending = {"state": sid, "findings": findings, "via": None, "resume_entry": entry}
                self._set_status("stopped", f"budget: {e}")
                return None
            t0 = self.clock()
            err: BaseException | None = None
            out: Any = None
            recovery: dict[str, Any] | None = None
            with SpendScope() as scope, TokenScope() as tscope:
                try:
                    recovery = await self._recover(state, entry, rerun or attempt > 1 or respawned)
                    ctx = StateContext(self.run_id, sid, entry, attempt, inputs, list(findings), dict(self.params), role,
                                       (self.models or {}).get(role) if role else None, self.clients.get(role) if role else None,
                                       self.services, self.run_dir, copy.deepcopy(dict(state)), recovery, self._budget_left(state),
                                       respawned, self.artifact_root)
                    coro = handler(ctx)
                    out = await (asyncio.wait_for(coro, state["timeout_s"]) if state.get("timeout_s") else coro)
                except BudgetExhausted as e:
                    err = e
                except TimeoutError:
                    err = StateFailed(f"timed out after {state['timeout_s']}s")
                except Exception as e:  # noqa: BLE001 -- a failing state is an attempt, never a crashed run
                    err = e
            usd = self.ledger.settle(who, scope.usd)
            secs = max(0.0, self.clock() - t0)
            tokens = tscope.tokens + (int(out.tokens) if isinstance(out, Settle) else 0)
            self._count(sid, usd, secs, tokens)
            self._log({"event": "spend", "state": sid, "entry": entry, "attempt": attempt, "usd": round(usd, 6), "seconds": round(secs, 4),
                       "tokens": tokens, **({"respawn": True} if respawned else {})})
            if isinstance(err, CaptureLost):  # the handler found an input its producer settled broken (a hand-off check)
                self._capture_lost(sid, entry, err)
                return None
            if isinstance(err, BudgetExhausted) or getattr(err, "failure_kind", None) == "crucible:credits_exhausted":
                self.pending = {"state": sid, "findings": findings, "via": None, "resume_entry": entry}
                self._set_status("stopped", f"budget: {err}")
                return None
            if err is not None and is_infra_error(err) and not respawned:
                # F:2876-2895 / :3019-3084: verified, then ONE free re-run of the same attempt (no retry, no entry spent)
                verdict = await self._recheck_infra(sid, err)
                self._log({"event": "infra_respawn", "state": sid, "entry": entry, "attempt": attempt, "verdict": verdict,
                           "error": f"{type(err).__name__}: {err}"[:500]})
                respawned = True
                continue
            capture: list[str] = []
            if err is None:
                settle = out if isinstance(out, Settle) else Settle(dict(out or {}))
                problems, capture = self._check_outputs(state, settle.outputs)
                if capture and not problems and all("artifact" in c for c in capture):
                    await self.sleep(self.recapture_wait_s)  # one bounded re-capture window (F:3171-3222)
                    problems, capture = self._check_outputs(state, settle.outputs)
                if not problems and not capture and state.get("isolated_from"):
                    text = json.dumps(settle.outputs, default=str)
                    hit = [s for s in state["isolated_from"] if s in self.canaries and self.canaries[s] in text]
                    if hit:
                        problems.append(f"breach: output carries the canary of {hit}, which this state may not see")
                if not problems and not capture:
                    outputs = dict(settle.outputs)
                    canary = None
                    if state["canary"]:
                        canary = self.canaries.get(sid) or "CRUCIBLE-CANARY-" + hashlib.sha256(f"{self.run_id}|{sid}".encode()).hexdigest()[:16]
                        outputs = {k: ({**v, "_canary": canary} if isinstance(v, dict) else v) for k, v in outputs.items()}
                        self.canaries[sid] = canary
                    self.latest[sid] = outputs
                    self.latest_findings[sid] = [str(f) for f in settle.findings]
                    self._log({"event": "settle", "state": sid, "entry": entry, "attempt": attempt, "outputs": outputs,
                               "findings": self.latest_findings[sid], "canary": canary})
                    return Settle(outputs, self.latest_findings[sid])
                last = "; ".join(problems + capture)
                blame = "task"
                if capture:  # the producer's task failure; its outputs stay provisional for needs_verify (F:3123-3169)
                    nv = {"state": sid, "entry": entry, "attempt": attempt, "error": "; ".join(capture)[:500],
                          "outputs": json.loads(json.dumps(settle.outputs, default=str))}
                    self.needs_verify.append(nv)
                    self._log({"event": "capture_failed", **nv})
            else:
                last = f"{type(err).__name__}: {err}"[:500]
                blame = "infra" if is_infra_error(err) else "task"
            self._log({"event": "fail", "state": sid, "entry": entry, "attempt": attempt, "error": last, "blame": blame})
            self.fails.setdefault(key, []).append(last)
            over_budget = self._state_budget_spent(state)
            if over_budget:  # a spent state budget: no retry (F:2925-2939 does the same for budget_ms)
                break
            attempt += 1
        self.pending = {"state": sid, "findings": findings, "via": None, "resume_entry": entry}
        if over_budget:
            self._escalate("state_budget", f"{over_budget}; last: {last}"[:500], state=sid, options=("revise", "drop"))
        elif blame == "infra":  # V8: an infra failure is masked, a decision, never the task's 0 (fail_fast does not apply)
            self._escalate("infra", f"state {sid}: infrastructure failed after the free respawn and {attempts} attempt(s): {last}", state=sid)
        elif state["failure_policy"] == "fail_fast":
            self._set_status("failed", f"state {sid} failed: {last}")
        else:
            self._escalate("failed", f"state {sid} failed after {attempts} attempt(s): {last}", state=sid, options=("revise", "drop"))
        return None
