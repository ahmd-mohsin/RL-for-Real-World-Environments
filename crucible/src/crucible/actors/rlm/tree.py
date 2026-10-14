"""``tree.py`` (m16): tree-wide budgets and persistent child agents for the very-long-horizon RLM harness.

Design: ``docs/research/long-horizon-rlm/04-design.md`` §3.3 (requirements R7, R8 of ``01-frameworks.md`` §7, H6 of
``02-papers-articles.md`` §9, break point B6 / gap G6 of ``03-crucible-stress.md``).

**One budget for the whole agent tree** (``TreeSupervisor``). Every agent of an episode (the root RLM at depth 0, its
children at depth 1, grandchildren at depth 2) charges the same supervisor: model turns, completion tokens (summary calls
count as tokens, never as turns; nano-rlm ``engine.py:1276-1281``), spawns, sub-LLM calls and their tokens. A spent budget
raises ``BudgetExhausted`` whose ``kind`` is the typed stop (``budget:<name>``), which the loop records as a truncation.
Ported from nano-rlm (MIT) ``src/rlm/config.py:51-93`` (tree-total caps, ``max_concurrent_subagents >= max_depth``) and
``src/rlm/supervisor.py:439-456`` (a spawn is refused at the depth, call, turn or token cap). M14's ``accept/repair.py``
``TreeBudget`` (audit H3 V11: depth was the only limit, one cell spawned 25 children) is the in-repo precedent; this one
adds per-depth accounting, concurrency and the persistent-children API. Episode-wide limits that do not scale with the
horizon were B6 (``max_turns=500``, 256 sub-calls): the defaults here are thousands of turns and 10M tokens.

**Per-depth reserved concurrency** (``TreeSupervisor.slot``). A child holds a slot while it runs, including while it
waits for its own children. With one shared pool, ``max_concurrent`` depth-1 children that each wait for a depth-2 child
take every slot and the tree deadlocks. nano-rlm (``supervisor.py:146-158``, ``depth_capacities``) partitions the pool
per depth. Here a depth-``d`` child may start only while fewer than ``cap(d) = max_concurrent - reserve_per_depth *
(max_depth - d)`` children run tree-wide, so ``reserve_per_depth`` slots per deeper level are always held back (the
deepest level may use the whole pool). Proof sketch: in a stuck state let ``D`` be the deepest running depth. ``D <
max_depth`` (deepest-level children never wait on a slot), so at most ``cap(D) < cap(D + 1)`` children run and the
pending depth-``D + 1`` child they wait for can start. ``tests/unit/test_m16_c_tree.py`` shows the naive shared semaphore
deadlocking on the same tree.

**Persistent children** (``ChildRegistry``; the Prime Agent ``rlm.spawn`` / ``rlm.collect`` API,
``prime-agent-runtime/src/rlm/__init__.py:179-210,416-453``, and nano-rlm's ``rlm.agent`` op set,
``supervisor.py:1076-1135``, both MIT). A child is an asynchronous handle with its own kernel, context and ledger
(built by the factory F supplies). ``spawn`` returns at once; ``result(name, wait_s)`` polls or waits; ``send`` queues a
message (and wakes a finished persistent child for a follow-up round, nano-rlm ``status="idle"``,
``supervisor.py:1230``); ``steer`` interrupts a running child; ``cancel`` cascades to every descendant; ``watch``
subscribes to progress; ``events`` drains the parent's inbox (``child.completed`` with the last 4 KiB of the answer,
nano-rlm ``supervisor.py:55,1256-1262``; ``child.failed``; ``child.cancelled``; ``child.progress``).

- **Hard limits, not cooperative ones.** ``child_wall_s`` and the per-child turn cap are enforced by the registry: the
  runner is told to stop, then its task is cancelled (OpenEnv's ``repl_env`` timeouts were cooperative only, the
  "what to avoid" list of ``01-frameworks.md``). A child stopped by a budget is reported with status ``budget`` and the
  typed kind in ``error``, never as a crash.
- **Least privilege.** ``world`` is ``"none"`` (default: the child computes over its brief, files handed to it and
  sub-calls) or ``"shared"`` (World calls through the root's action queue). The grant is passed to the factory and
  recorded in every row.
- **Bounded messages.** The inbox keeps ``max_events`` (drop-oldest, with a counted ``events.dropped`` notice);
  every event fits ``max_event_bytes``; ``result`` returns at most 8,000 answer characters with a pointer to the
  child's ledger, which keeps the full text.
- **Usage accounting.** A child's turns and tokens are charged to the tree (the shared supervisor) by the child's own
  loop, never to the parent's context meter (Prime Agent ``SE/rlm_usage.rs:32-38``): the parent's context grows only
  by what the parent reads (events, results), which these caps bound.
- **Resume (v1).** ``TreeSupervisor.state_dict`` keeps the counters. ``ChildRegistry.state_dict`` keeps each child's
  row and capped result. Children are **not** resumed mid-flight: on ``load_state_dict`` a child that was running is
  reported as ``failed`` with error ``"lost at resume"`` and a ``child.failed`` event that says so (nano-rlm has no
  supervisor crash recovery either, README "Compaction"). Its spend stays charged.

Kernel side: ``AGENT_API`` (Python source run in the kernel after ``BOOTSTRAP``) defines ``rlm.agent`` whose methods are
host requests ``"agent.<op>"``, served by ``ChildRegistry.handle``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import json
import math
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

RESULT_ANSWER_CHARS = 8_000  # result() answer cap (H6: a returned answer of about 2k tokens)
COMPLETED_TAIL_BYTES = 4 * 1024  # child.completed carries the answer's last 4 KiB (nano-rlm supervisor.py:55)
PROGRESS_LAST_BYTES = 2 * 1024  # child.progress carries the tail of the child's last output
MAX_BRIEF_CHARS = 1_000_000  # a brief is prompt text for the child; large inputs go in files handed to it
BRIEF_HEAD_CHARS = 500  # what a registry checkpoint keeps of each brief
LOST_AT_RESUME = "lost at resume"
WORLD_GRANTS = ("none", "shared")
RUN_STATUSES = ("done", "failed", "cancelled", "budget")
NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
BUDGET_KINDS = ("budget:new_tokens", "budget:turns", "budget:spawns", "budget:depth", "budget:sub_calls", "budget:sub_tokens",
                "budget:child_wall_s", "budget:child_turns")

# host request "agent.<op>" -> {argument: required}
_OPS: dict[str, dict[str, bool]] = {
    "spawn": {"brief": True, "name": False, "persistent": False, "world": False, "max_turns": False},
    "result": {"name": True, "wait_s": False},
    "send": {"name": True, "text": True},
    "steer": {"name": True, "text": True},
    "cancel": {"name": True},
    "watch": {"name": True, "every_turns": False, "every_tokens": False},
    "events": {},
    "list": {},
}


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class TreeLimits:
    """Tree-wide limits (nano-rlm ``config.py:51-93``; Crucible defaults sized for 1-10M-token episodes, B6)."""

    max_total_new_tokens: int = 10_000_000  # completion tokens (+ summary calls) across the whole tree
    max_total_turns: int = 20_000  # model turns across the tree (summary calls excluded)
    max_spawns: int = 64
    max_depth: int = 1  # 0 = no children; <= 3
    max_concurrent: int = 8  # children running at once, tree-wide
    reserve_per_depth: int = 1  # slots held back for deeper levels so nesting cannot deadlock
    child_wall_s: float = 3_600.0
    child_max_turns: int = 2_000
    max_sub_calls: int = 4_096  # rlm.query / batch, tree-wide
    max_sub_tokens: int = 20_000_000
    max_events: int = 10_000  # child-event inbox per parent (nano-rlm supervisor.py:88)
    max_event_bytes: int = 65_536  # one event's JSON size (nano-rlm supervisor.py:89)

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == "child_wall_s":
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise ValueError(f"TreeLimits.child_wall_s must be a positive finite number, got {v!r}")
                continue
            if not _is_int(v):
                raise TypeError(f"TreeLimits.{f.name} must be an int, got {type(v).__name__}")
            low = 0 if f.name in ("max_depth", "reserve_per_depth") else 1
            if v < low:
                raise ValueError(f"TreeLimits.{f.name} must be >= {low}, got {v}")
        if self.max_depth > 3:
            raise ValueError(f"TreeLimits.max_depth must be <= 3, got {self.max_depth}")
        if self.max_depth >= 2 and self.reserve_per_depth < 1:
            raise ValueError("TreeLimits.reserve_per_depth must be >= 1 when max_depth >= 2: without reserved capacity nested children "
                             "can deadlock (nano-rlm supervisor.py:146-158)")
        if self.max_depth >= 1 and self.cap(1) < 1:
            raise ValueError(f"TreeLimits.max_concurrent ({self.max_concurrent}) must exceed reserve_per_depth * (max_depth - 1) "
                             f"({self.reserve_per_depth * (self.max_depth - 1)}) so depth-1 children get a slot")
        if self.max_event_bytes < 1024:
            raise ValueError("TreeLimits.max_event_bytes must be >= 1024")

    def cap(self, depth: int) -> int:
        """How many children may run tree-wide while a depth-``depth`` child starts (see the module docstring)."""
        return self.max_concurrent - self.reserve_per_depth * (self.max_depth - depth)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> TreeLimits:
        """The fields present in ``m`` (e.g. an ``RLMConfig`` dict); other keys are ignored."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in m.items() if k in names})


class BudgetExhausted(Exception):
    """A tree budget is spent. ``kind`` is the typed stop, one of ``BUDGET_KINDS`` (``"budget:new_tokens"`` ...)."""

    def __init__(self, kind: str, message: str = ""):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind

    def __reduce__(self) -> tuple[Any, ...]:
        msg = str(self)
        return (BudgetExhausted, (self.kind, msg[len(self.kind) + 2:] if msg.startswith(self.kind + ": ") else ""))


_COUNTERS = ("turns", "new_tokens", "estimated_new_tokens", "prompt_tokens", "summary_calls", "summary_tokens", "spawns", "sub_calls",
             "sub_tokens", "refused")
_DEPTH_COUNTERS = ("turns", "new_tokens", "prompt_tokens", "summary_calls", "spawns")


class TreeSupervisor:
    """The tree's budgets and concurrency, shared by every agent of one episode (see the module docstring).

    ``charge_turn`` / ``charge_sub`` / ``check_spawn`` refuse *before* the work (nothing is charged on a refusal).
    ``charge_tokens`` / ``charge_sub_tokens`` record usage that already happened (always recorded, so the totals stay
    honest), then raise when the total now exceeds its cap; the caller may still keep the reply it holds, and the next
    ``charge_turn`` / ``charge_sub`` is refused anyway. Callers pass engine usage; ``estimate=True`` marks a number that
    is not (it is counted, and ``usage()`` says so)."""

    def __init__(self, limits: TreeLimits | None = None):
        self.limits = limits if limits is not None else TreeLimits()
        self._c: dict[str, int] = dict.fromkeys(_COUNTERS, 0)
        self._by_depth: dict[int, dict[str, int]] = {d: dict.fromkeys(_DEPTH_COUNTERS, 0) for d in range(self.limits.max_depth + 1)}
        self._running = 0
        self._running_by_depth: dict[int, int] = {d: 0 for d in range(1, self.limits.max_depth + 1)}
        self._peak_running = 0
        self._waiters: list[asyncio.Future[None]] = []

    # ---- checks ---------------------------------------------------------------------------------------------------------
    def _depth(self, depth: Any, *, child: bool = False) -> int:
        if not _is_int(depth) or depth < (1 if child else 0) or depth > self.limits.max_depth:
            raise ValueError(f"depth must be an int in [{1 if child else 0}, {self.limits.max_depth}], got {depth!r}")
        return depth

    def _refuse(self, kind: str, message: str) -> BudgetExhausted:
        self._c["refused"] += 1
        return BudgetExhausted(kind, message)

    def _spent(self) -> BudgetExhausted | None:
        lim = self.limits
        if self._c["turns"] >= lim.max_total_turns:
            return self._refuse("budget:turns", f"the agent tree's turn budget ({lim.max_total_turns}) is spent")
        if self._c["new_tokens"] >= lim.max_total_new_tokens:
            return self._refuse("budget:new_tokens", f"the agent tree's token budget ({lim.max_total_new_tokens}) is spent")
        return None

    @staticmethod
    def _count(v: Any, what: str) -> int:
        if not _is_int(v) or v < 0:
            raise ValueError(f"{what} must be a non-negative int, got {v!r}")
        return v

    # ---- charges --------------------------------------------------------------------------------------------------------
    def charge_turn(self, depth: int) -> None:
        """Before a model turn (a work-loop call; summary calls are not turns). Raises when turns or tokens are spent."""
        d = self._depth(depth)
        err = self._spent()
        if err is not None:
            raise err
        self._c["turns"] += 1
        self._by_depth[d]["turns"] += 1

    def charge_tokens(self, depth: int, completion: int, prompt: int = 0, *, summary: bool = False, estimate: bool = False) -> None:
        """After a model call: ``completion`` counts toward ``max_total_new_tokens``; ``prompt`` is recorded only."""
        d = self._depth(depth)
        completion, prompt = self._count(completion, "completion"), self._count(prompt, "prompt")
        self._c["new_tokens"] += completion
        self._c["prompt_tokens"] += prompt
        self._by_depth[d]["new_tokens"] += completion
        self._by_depth[d]["prompt_tokens"] += prompt
        if estimate:
            self._c["estimated_new_tokens"] += completion
        if summary:
            self._c["summary_calls"] += 1
            self._c["summary_tokens"] += completion
            self._by_depth[d]["summary_calls"] += 1
        if self._c["new_tokens"] > self.limits.max_total_new_tokens:
            raise self._refuse("budget:new_tokens", f"the agent tree's token budget ({self.limits.max_total_new_tokens}) is exceeded "
                                                    f"({self._c['new_tokens']})")

    def charge_sub(self, n: int) -> None:
        """Before ``n`` sub-LLM calls (``rlm.query`` / ``batch``): all or nothing, so a refused batch makes no call."""
        if not _is_int(n) or n < 1:
            raise ValueError(f"n must be a positive int, got {n!r}")
        lim = self.limits
        if self._c["sub_calls"] + n > lim.max_sub_calls:
            raise self._refuse("budget:sub_calls", f"the agent tree's sub-LLM call budget ({lim.max_sub_calls}) would be exceeded "
                                                   f"({self._c['sub_calls']} used, {n} asked)")
        if self._c["sub_tokens"] >= lim.max_sub_tokens:
            raise self._refuse("budget:sub_tokens", f"the agent tree's sub-LLM token budget ({lim.max_sub_tokens}) is spent")
        self._c["sub_calls"] += n

    def charge_sub_tokens(self, n: int) -> None:
        """After a sub-LLM call (prompt + completion of that call)."""
        self._c["sub_tokens"] += self._count(n, "n")
        if self._c["sub_tokens"] > self.limits.max_sub_tokens:
            raise self._refuse("budget:sub_tokens", f"the agent tree's sub-LLM token budget ({self.limits.max_sub_tokens}) is exceeded "
                                                    f"({self._c['sub_tokens']})")

    def check_spawn(self, depth: int) -> None:
        """May an agent at ``depth`` spawn a child (``depth + 1 <= max_depth``, spawns, turns and tokens left)?"""
        d = self._depth(depth)
        if d + 1 > self.limits.max_depth:
            raise self._refuse("budget:depth", f"recursion depth limit {self.limits.max_depth} reached (this agent is at depth {d})")
        if self._c["spawns"] >= self.limits.max_spawns:
            raise self._refuse("budget:spawns", f"the agent tree's spawn budget ({self.limits.max_spawns}) is spent")
        err = self._spent()
        if err is not None:
            raise err

    def charge_spawn(self, depth: int) -> None:
        """``check_spawn`` and count the spawn (what ``ChildRegistry.spawn`` calls)."""
        self.check_spawn(depth)
        self._c["spawns"] += 1
        self._by_depth[depth]["spawns"] += 1

    def _uncharge_spawn(self, depth: int) -> None:  # the factory failed: the child never existed
        self._c["spawns"] -= 1
        self._by_depth[depth]["spawns"] -= 1

    # ---- concurrency ----------------------------------------------------------------------------------------------------
    @contextlib.asynccontextmanager
    async def slot(self, depth: int) -> AsyncIterator[None]:
        """Hold one of ``max_concurrent`` tree-wide slots while a depth-``depth`` child runs (per-depth reservation)."""
        d = self._depth(depth, child=True)
        cap = self.limits.cap(d)
        while self._running >= cap:
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
            try:
                await fut
            finally:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(fut)
        self._running += 1
        self._running_by_depth[d] += 1
        self._peak_running = max(self._peak_running, self._running)
        try:
            yield
        finally:  # synchronous: a second cancellation during cleanup cannot leak the slot
            self._running -= 1
            self._running_by_depth[d] -= 1
            for w in list(self._waiters):
                if not w.done():
                    w.set_result(None)

    # ---- reporting and checkpoint ---------------------------------------------------------------------------------------
    def usage(self) -> dict[str, Any]:
        lim, c = self.limits, self._c
        left = {"turns": lim.max_total_turns - c["turns"], "new_tokens": lim.max_total_new_tokens - c["new_tokens"],
                "spawns": lim.max_spawns - c["spawns"], "sub_calls": lim.max_sub_calls - c["sub_calls"],
                "sub_tokens": lim.max_sub_tokens - c["sub_tokens"]}
        fraction = max(c["turns"] / lim.max_total_turns, c["new_tokens"] / lim.max_total_new_tokens, c["spawns"] / lim.max_spawns,
                       c["sub_calls"] / lim.max_sub_calls, c["sub_tokens"] / lim.max_sub_tokens)
        return {**c, "new_tokens_source": "usage" if c["estimated_new_tokens"] == 0 else "usage+estimate",
                "running": self._running, "running_by_depth": dict(self._running_by_depth), "peak_running": self._peak_running,
                "by_depth": {d: dict(v) for d, v in self._by_depth.items()}, "left": {k: max(0, v) for k, v in left.items()},
                "fraction": fraction, "limits": dataclasses.asdict(lim)}

    def state_dict(self) -> dict[str, Any]:
        """Counters only: running children are not checkpointed (v1, see ``ChildRegistry.load_state_dict``)."""
        return {"version": 1, "counters": dict(self._c), "by_depth": {str(d): dict(v) for d, v in self._by_depth.items()},
                "peak_running": self._peak_running, "limits": dataclasses.asdict(self.limits)}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore the counters (the current ``limits`` stay: a resumed run may change them)."""
        if not isinstance(d, dict) or d.get("version") != 1:
            raise ValueError("not a TreeSupervisor state (version 1)")
        counters = d.get("counters")
        if not isinstance(counters, dict) or set(counters) != set(_COUNTERS):
            raise ValueError(f"TreeSupervisor state counters must have exactly {sorted(_COUNTERS)}")
        for k, v in counters.items():
            self._count(v, k)
        by_depth: dict[int, dict[str, int]] = {dd: dict.fromkeys(_DEPTH_COUNTERS, 0) for dd in range(self.limits.max_depth + 1)}
        for k, row in (d.get("by_depth") or {}).items():
            dd = int(k)
            if dd not in by_depth or not isinstance(row, dict) or set(row) != set(_DEPTH_COUNTERS):
                raise ValueError(f"TreeSupervisor state by_depth row {k!r} does not fit max_depth {self.limits.max_depth}")
            by_depth[dd] = {kk: self._count(vv, kk) for kk, vv in row.items()}
        self._c = dict(counters)
        self._by_depth = by_depth
        self._peak_running = self._count(d.get("peak_running", 0), "peak_running")


@runtime_checkable
class ChildRunner(Protocol):
    """One child agent (F's ``AgentLoop``). ``run`` returns ``{"status": done|failed|cancelled|budget, "answer": str,
    "error": str | None}`` (raising ``BudgetExhausted`` counts as ``budget``, any other exception as ``failed``); for a
    persistent child it may be called again after ``deliver`` (a follow-up round). ``deliver`` kind: ``"message"`` (read at
    the next turn boundary) or ``"steer"`` (interrupts the current turn). ``cancel`` asks it to stop (the registry cancels
    the task too). ``progress`` returns ``{"turns", "tokens", "last"}`` (cumulative). An optional ``async aclose()`` frees
    the child's kernel; the registry calls it when the child is finished with."""

    name: str

    async def run(self) -> dict: ...

    def deliver(self, text: str, kind: str) -> None: ...

    def cancel(self) -> None: ...

    def progress(self) -> dict: ...


@dataclass
class _Child:
    name: str
    depth: int
    world: str
    persistent: bool
    max_turns: int
    brief_head: str
    brief_chars: int
    registry: ChildRegistry | None = None
    runner: Any = None
    status: str = "starting"  # starting | running | done | failed | cancelled | budget
    queued: bool = True
    answer: str = ""
    answer_chars: int = 0
    truncated: bool = False  # restored rows: ``answer`` already holds the capped view
    error: str | None = None
    elapsed: float = 0.0
    round_t0: float | None = None
    rounds: int = 0
    task: asyncio.Task[None] | None = None
    run_task: asyncio.Future[Any] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_requested: bool = False
    stop_kind: str | None = None
    every_turns: int | None = None
    every_tokens: int | None = None
    next_turns: int = 0
    next_tokens: int = 0
    last_progress: dict[str, Any] = field(default_factory=lambda: {"turns": 0, "tokens": 0, "last": ""})
    runner_closed: bool = False
    cleanup_error: str | None = None
    restored: bool = False
    stuck: bool = False  # the run task ignored repeated cancellation


def _tail_bytes(s: str, n: int) -> str:
    b = s.encode("utf-8")
    return s if len(b) <= n else b[-n:].decode("utf-8", errors="ignore")


def _json_size(obj: Any) -> int:
    return len(json.dumps(obj, default=repr, ensure_ascii=False).encode("utf-8"))


def _cap_answer(name: str, answer: str) -> tuple[str, bool]:
    """At most ``RESULT_ANSWER_CHARS`` characters: head + a pointer to the full answer in the child's ledger + tail."""
    if len(answer) <= RESULT_ANSWER_CHARS:
        return answer, False
    pointer = (f"\n[... answer truncated: {len(answer):,} characters in total; the full answer is in the child's ledger: "
               f"history.child({name!r}) ...]\n")
    avail = RESULT_ANSWER_CHARS - len(pointer)
    head = avail * 3 // 4
    return answer[:head] + pointer + answer[len(answer) - (avail - head):], True


async def _hard_cancel(task: asyncio.Future[Any], grace_s: float, tries: int = 3) -> bool:
    """Cancel ``task`` and wait for it; repeat for a task that swallows a cancellation. True when it ended."""
    for _ in range(tries):
        if task.done():
            return True
        task.cancel()
        await asyncio.wait({task}, timeout=grace_s)
    return task.done()


class ChildRegistry:
    """The children of one agent at ``depth`` (the root's registry has ``depth=0``); see the module docstring.

    ``factory(*, name, brief, depth, world, persistent, max_turns, path, registry)`` returns a ``ChildRunner`` (or an
    awaitable of one). ``depth`` is the child's depth, ``path`` its tree path (``"root/scout/reader"``, for its ledger
    directory), ``registry`` the child's own ``ChildRegistry`` (same supervisor, ``depth + 1``) for its grandchildren:
    the parent's registry owns it, so ``cancel`` reaches every descendant even when a child's code ignores the request.
    ``max_turns`` is the child's own turn cap (the runner should stop itself there; the registry stops it hard one turn
    past it). ``poll_s`` is the monitor period (wall, turn cap, watch events); ``cancel_grace_s`` how long a cancelled
    task gets per attempt."""

    def __init__(self, supervisor: TreeSupervisor, factory: Callable[..., Any], *, depth: int, path: str = "root",
                 poll_s: float = 0.25, cancel_grace_s: float = 5.0):
        self.supervisor = supervisor
        self.factory = factory
        self.depth = supervisor._depth(depth)
        self.path = path
        self.poll_s = float(poll_s)
        self.cancel_grace_s = float(cancel_grace_s)
        self._children: dict[str, _Child] = {}
        self._inbox: deque[dict[str, Any]] = deque()
        self._dropped = 0
        self._seq = 0
        self._auto = 0
        self._closed = False
        self._closing: asyncio.Future[None] | None = None

    @property
    def limits(self) -> TreeLimits:
        return self.supervisor.limits

    # ---- validation -----------------------------------------------------------------------------------------------------
    def _text(self, v: Any, what: str) -> str:
        if not isinstance(v, str) or not v:
            raise ValueError(f"{what} must be a non-empty string")
        if len(v.encode("utf-8")) > self.limits.max_event_bytes:
            raise ValueError(f"{what} exceeds {self.limits.max_event_bytes} bytes; hand large inputs over as files")
        return v

    def _get(self, name: Any) -> _Child:
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ValueError(f"child name must match [A-Za-z0-9_-]{{1,64}}, got {name!r}")
        c = self._children.get(name)
        if c is None:
            raise ValueError(f"no child named {name!r} (children: {', '.join(self._children) or 'none'})")
        return c

    def _new_name(self, name: Any) -> str:
        if name is None:
            while True:
                self._auto += 1
                cand = f"child-{self._auto}"
                if cand not in self._children:
                    return cand
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ValueError(f"child name must match [A-Za-z0-9_-]{{1,64}}, got {name!r}")
        if name in self._children:
            raise ValueError(f"a child named {name!r} already exists; names are unique among siblings")
        return name

    # ---- events ---------------------------------------------------------------------------------------------------------
    def _bound(self, ev: dict[str, Any]) -> dict[str, Any]:
        """Fit ``max_event_bytes``: cut the longest string field (keeping the tail of ``*_tail`` / ``last`` fields)."""
        cap = self.limits.max_event_bytes
        for _ in range(64):
            size = _json_size(ev)
            if size <= cap:
                return ev
            strs = [(len(v.encode("utf-8")), k) for k, v in ev.items() if isinstance(v, str) and v]
            if not strs:
                break
            n, k = max(strs)
            keep = max(0, n - (size - cap) - 64)
            b = ev[k].encode("utf-8")
            ev[k] = (b[len(b) - keep:] if (k.endswith("_tail") or k == "last") else b[:keep]).decode("utf-8", errors="ignore")
            ev["truncated"] = True
        return {"type": ev.get("type"), "name": ev.get("name"), "seq": ev.get("seq"), "truncated": True}

    def _emit(self, ev: dict[str, Any]) -> None:
        self._seq += 1
        ev = self._bound({**ev, "seq": self._seq, "at": time.time()})
        if len(self._inbox) >= self.limits.max_events:
            self._inbox.popleft()  # drop-oldest, counted (nano-rlm caps the inbox at 10,000, supervisor.py:88,545)
            self._dropped += 1
        self._inbox.append(ev)

    def events(self) -> list[dict[str, Any]]:
        """Drain the inbox: ``child.completed`` | ``child.failed`` | ``child.cancelled`` | ``child.progress``, after an
        ``events.dropped`` notice when the inbox overflowed since the last drain."""
        out: list[dict[str, Any]] = []
        if self._dropped:
            out.append({"type": "events.dropped", "count": self._dropped,
                        "note": f"the inbox holds at most {self.limits.max_events} events; the {self._dropped} oldest were dropped. "
                                "Use rlm.agent.list() and rlm.agent.result(name) for the current state."})
            self._dropped = 0
        out.extend(self._inbox)
        self._inbox.clear()
        return out

    # ---- views ----------------------------------------------------------------------------------------------------------
    def _progress(self, c: _Child) -> dict[str, Any]:
        if c.runner is None:
            return c.last_progress
        try:
            p = c.runner.progress()
        except Exception:  # noqa: BLE001 - a broken progress() must not take the monitor down
            return c.last_progress
        if not isinstance(p, dict):
            return c.last_progress
        turns, tokens, last = p.get("turns"), p.get("tokens"), p.get("last")
        c.last_progress = {"turns": turns if _is_int(turns) and turns >= 0 else c.last_progress["turns"],
                           "tokens": tokens if _is_int(tokens) and tokens >= 0 else c.last_progress["tokens"],
                           "last": last if isinstance(last, str) else ""}
        return c.last_progress

    def _elapsed(self, c: _Child) -> float:
        return c.elapsed + (time.monotonic() - c.round_t0 if c.round_t0 is not None else 0.0)

    def _row(self, c: _Child) -> dict[str, Any]:
        p = self._progress(c)
        return {"name": c.name, "status": "running" if c.status == "starting" else c.status, "queued": c.status == "running" and c.queued,
                "depth": c.depth, "world": c.world, "persistent": c.persistent, "max_turns": c.max_turns, "rounds": c.rounds,
                "idle": c.status == "done" and c.persistent and not c.restored and not c.runner_closed,
                "turns": p["turns"], "tokens": p["tokens"], "elapsed_s": round(self._elapsed(c), 3), "error": c.error,
                "restored": c.restored}

    def list(self) -> list[dict[str, Any]]:
        return [self._row(c) for c in self._children.values()]

    def _result_view(self, c: _Child) -> dict[str, Any]:
        row = self._row(c)
        if c.status in ("starting", "running"):
            return {**row, "status": "running", "last": _tail_bytes(c.last_progress["last"], PROGRESS_LAST_BYTES)}
        if c.restored:
            answer, truncated = c.answer, c.truncated
        else:
            answer, truncated = _cap_answer(c.name, c.answer)
        return {**row, "answer": answer, "answer_chars": c.answer_chars, "truncated": truncated}

    # ---- the API --------------------------------------------------------------------------------------------------------
    async def spawn(self, brief: str, *, name: str | None = None, persistent: bool = True, world: str = "none",
                    max_turns: int | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("this agent's children are closed")
        if not isinstance(brief, str) or not brief.strip():
            raise ValueError("brief must be a non-empty string: a self-contained task for the child (it sees none of your context)")
        if len(brief) > MAX_BRIEF_CHARS:
            raise ValueError(f"brief exceeds {MAX_BRIEF_CHARS:,} characters; hand large inputs over as files")
        if not isinstance(persistent, bool):
            raise TypeError(f"persistent must be a bool, got {persistent!r}")
        if world not in WORLD_GRANTS:
            raise ValueError(f"world must be one of {WORLD_GRANTS}, got {world!r}")
        cap = self.limits.child_max_turns
        if max_turns is None:
            max_turns = cap
        elif not _is_int(max_turns) or not 1 <= max_turns <= cap:  # a model may lower the child's cap, never raise it
            raise ValueError(f"max_turns must be an int in [1, {cap}], got {max_turns!r}")
        name = self._new_name(name)
        self.supervisor.charge_spawn(self.depth)  # raises BudgetExhausted (depth, spawns, turns, tokens)
        c = _Child(name=name, depth=self.depth + 1, world=world, persistent=persistent, max_turns=max_turns,
                   brief_head=brief[:BRIEF_HEAD_CHARS], brief_chars=len(brief))
        self._children[name] = c  # reserved before any await: concurrent spawns cannot share a name
        sub = ChildRegistry(self.supervisor, self.factory, depth=c.depth, path=f"{self.path}/{name}",
                            poll_s=self.poll_s, cancel_grace_s=self.cancel_grace_s)
        c.registry = sub
        try:
            runner = self.factory(name=name, brief=brief, depth=c.depth, world=world, persistent=persistent, max_turns=max_turns,
                                  path=sub.path, registry=sub)
            if inspect.isawaitable(runner):
                runner = await runner
            if not isinstance(runner, ChildRunner):
                raise TypeError(f"the child factory returned {type(runner).__name__}, not a ChildRunner")
            if self._closed:
                raise RuntimeError("this agent's children were closed while the child was being created")
        except BaseException:
            self._children.pop(name, None)
            self.supervisor._uncharge_spawn(self.depth)
            sub._closed = True
            raise
        c.runner = runner
        if c.cancel_requested:  # cancelled while the factory ran
            await self._close_runner(c)
            self._settle(c, {"status": "cancelled", "answer": "", "error": "cancelled before it started"})
            return {"name": name, "status": "cancelled", "depth": c.depth, "world": world, "persistent": persistent, "max_turns": max_turns}
        self._start(c)
        return {"name": name, "status": "running", "depth": c.depth, "world": world, "persistent": persistent, "max_turns": max_turns}

    def _start(self, c: _Child) -> None:
        c.status, c.queued, c.stop_kind, c.cancel_requested, c.run_task = "running", True, None, False, None
        c.rounds += 1
        c.done.clear()
        c.task = asyncio.create_task(self._drive(c), name=f"crucible-child:{self.path}/{c.name}")

    async def _invoke(self, runner: Any) -> Any:
        return await runner.run()

    async def _drive(self, c: _Child) -> None:
        """One round of one child: wait for a slot, run under the wall / turn monitor, settle, emit the event."""
        outcome: dict[str, Any]
        try:
            async with self.supervisor.slot(c.depth):
                c.queued = False
                c.round_t0 = time.monotonic()
                c.run_task = asyncio.ensure_future(self._invoke(c.runner))
                try:
                    await self._monitor(c)
                finally:
                    c.elapsed += time.monotonic() - c.round_t0
                    c.round_t0 = None
                outcome = self._outcome(c)
        except asyncio.CancelledError:  # cancelled while queued for a slot (cancel() stops a running round through run_task)
            if c.run_task is not None and not c.run_task.done():
                c.run_task.cancel()
            outcome = {"status": "cancelled", "answer": "", "error": "cancelled before it started" if c.run_task is None else "cancelled"}
            self._settle(c, outcome)
            return
        except Exception as e:  # noqa: BLE001 - an infra fault of the registry itself: the child failed, the parent goes on
            outcome = {"status": "failed", "answer": "", "error": f"{type(e).__name__}: {e}"[:2000]}
        keep = outcome["status"] == "done" and c.persistent and not self._closed
        try:
            if not keep:  # a finished child's descendants and kernel go with it (a persistent idle child keeps them)
                if c.registry is not None:
                    await c.registry.aclose()
                await self._close_runner(c)
        finally:  # settled even when this cleanup is itself cancelled (aclose's last resort)
            self._settle(c, outcome)

    async def _monitor(self, c: _Child) -> None:
        lim = self.limits
        assert c.run_task is not None
        while True:
            wall_left = lim.child_wall_s - self._elapsed(c)
            if wall_left <= 0:
                c.stop_kind = "budget:child_wall_s"
                break
            done, _ = await asyncio.wait({c.run_task}, timeout=min(self.poll_s, wall_left))
            if done:
                return
            p = self._progress(c)
            self._watch_tick(c, p)
            if p["turns"] > c.max_turns:
                c.stop_kind = "budget:child_turns"
                break
        await self._stop_run(c)

    async def _stop_run(self, c: _Child) -> None:
        """Ask the runner to stop, then cancel its task (hard: the cooperative request alone is not trusted)."""
        try:
            c.runner.cancel()
        except Exception as e:  # noqa: BLE001 - the hard cancel below does not depend on it
            c.cleanup_error = f"cancel(): {type(e).__name__}: {e}"[:500]
        if c.run_task is not None and not await _hard_cancel(c.run_task, self.cancel_grace_s):
            c.stuck = True

    def _outcome(self, c: _Child) -> dict[str, Any]:
        t = c.run_task
        assert t is not None
        salvage = self._progress(c).get("last") or ""  # a capped stop salvages the child's last output (nano-rlm engine.py:740-756)
        if c.cancel_requested:
            return {"status": "cancelled", "answer": salvage, "error": "cancelled by the parent"}
        if c.stop_kind is not None:
            what = (f"wall-clock budget ({self.limits.child_wall_s:g} s)" if c.stop_kind == "budget:child_wall_s"
                    else f"turn budget ({c.max_turns} turns)")
            return {"status": "budget", "answer": salvage, "error": f"{c.stop_kind}: the child exceeded its {what} and was stopped"}
        if not t.done() or t.cancelled():
            return {"status": "cancelled", "answer": salvage, "error": "the child's run was cancelled"}
        exc = t.exception()
        if isinstance(exc, BudgetExhausted):
            return {"status": "budget", "answer": salvage, "error": str(exc)}
        if exc is not None:
            return {"status": "failed", "answer": salvage, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        res = t.result()
        if not isinstance(res, dict) or res.get("status") not in RUN_STATUSES:
            return {"status": "failed", "answer": salvage, "error": f"the child returned an invalid result: {res!r}"[:2000]}
        answer = res.get("answer")
        error = res.get("error")
        return {"status": res["status"], "answer": answer if isinstance(answer, str) else "",
                "error": None if error is None else str(error)[:2000]}

    def _settle(self, c: _Child, outcome: dict[str, Any]) -> None:
        c.status, c.answer, c.error = outcome["status"], outcome["answer"] or "", outcome["error"]
        c.answer_chars = len(c.answer)
        if c.stuck:
            c.error = (c.error or "") + " (the child's task ignored cancellation)"
        p = c.last_progress
        base = {"name": c.name, "status": c.status, "turns": p["turns"], "tokens": p["tokens"], "round": c.rounds}
        if c.status == "done":
            self._emit({"type": "child.completed", **base, "answer_tail": _tail_bytes(c.answer, COMPLETED_TAIL_BYTES),
                        "answer_chars": c.answer_chars, "idle": c.persistent and not c.runner_closed})
        elif c.status == "cancelled":
            self._emit({"type": "child.cancelled", **base, "error": c.error})
        else:
            self._emit({"type": "child.failed", **base, "error": c.error, "answer_tail": _tail_bytes(c.answer, COMPLETED_TAIL_BYTES)})
        c.done.set()

    def _watch_tick(self, c: _Child, p: dict[str, Any]) -> None:
        fire = False
        if c.every_turns and p["turns"] >= c.next_turns:
            fire, c.next_turns = True, (p["turns"] // c.every_turns + 1) * c.every_turns
        if c.every_tokens and p["tokens"] >= c.next_tokens:
            fire, c.next_tokens = True, (p["tokens"] // c.every_tokens + 1) * c.every_tokens
        if fire:
            self._emit({"type": "child.progress", "name": c.name, "status": "running", "turns": p["turns"], "tokens": p["tokens"],
                        "elapsed_s": round(self._elapsed(c), 3), "last": _tail_bytes(p["last"], PROGRESS_LAST_BYTES)})

    async def result(self, name: str, *, wait_s: float = 0.0) -> dict[str, Any]:
        """``running`` (with progress) or the settled result: ``answer`` at most 8,000 characters (``truncated`` and a
        pointer to the child's ledger when longer). ``wait_s`` (capped at ``child_wall_s``) waits for the child to settle."""
        c = self._get(name)
        if isinstance(wait_s, bool) or not isinstance(wait_s, (int, float)) or not math.isfinite(wait_s) or wait_s < 0:
            raise ValueError(f"wait_s must be a non-negative number of seconds, got {wait_s!r}")
        wait_s = min(float(wait_s), self.limits.child_wall_s)
        if c.status in ("starting", "running") and wait_s > 0:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(wait_s):
                    await c.done.wait()
        return self._result_view(c)

    async def send(self, name: str, text: str) -> dict[str, Any]:
        """Queue a message for the child's next turn boundary; a finished persistent child starts a follow-up round."""
        c = self._get(name)
        text = self._text(text, "text")
        if c.status in ("starting", "running"):
            c.runner.deliver(text, "message")
            return {"name": c.name, "delivered": True, "status": "running"}
        if c.status != "done" or not c.persistent or c.restored or c.runner_closed or self._closed:
            why = LOST_AT_RESUME if c.restored else f"it is {c.status}" + ("" if c.persistent else " and not persistent")
            raise RuntimeError(f"child {c.name!r} cannot take messages: {why}; spawn a new child")
        err = self.supervisor._spent()
        if err is not None:
            raise err
        if c.elapsed >= self.limits.child_wall_s:
            raise BudgetExhausted("budget:child_wall_s", f"child {c.name!r} has used its wall-clock budget ({self.limits.child_wall_s:g} s)")
        c.runner.deliver(text, "message")
        self._start(c)
        return {"name": c.name, "delivered": True, "status": "running", "round": c.rounds}

    async def steer(self, name: str, text: str) -> dict[str, Any]:
        """Interrupt a running child with an instruction (delivered at its next tool or answer boundary)."""
        c = self._get(name)
        text = self._text(text, "text")
        if c.status not in ("starting", "running"):
            raise RuntimeError(f"child {c.name!r} is {c.status}, not running; use send() for a follow-up to a persistent child")
        c.runner.deliver(text, "steer")
        return {"name": c.name, "delivered": True, "status": "running"}

    async def cancel(self, name: str) -> dict[str, Any]:
        """Stop the child and every descendant (hard), close its kernel, return its final row."""
        c = self._get(name)
        if c.status in ("starting", "running"):
            c.cancel_requested = True  # a "starting" child (factory still running) is settled by spawn()
            if c.run_task is not None:
                await self._stop_run(c)
            elif c.task is not None:
                c.task.cancel()  # queued for a slot
        if c.registry is not None:
            await c.registry.aclose()  # cascade: grandchildren are cancelled by this registry, not by the child's code
        if c.task is not None and not c.task.done():
            await asyncio.wait({c.task}, timeout=self.cancel_grace_s)
            if not c.task.done() and not await _hard_cancel(c.task, self.cancel_grace_s):
                c.stuck = True
        if not c.done.is_set() and c.status in ("starting", "running") and (c.task is None or c.task.done()):
            # a task cancelled before its first step never enters _drive, so nothing settled it
            self._settle(c, {"status": "cancelled", "answer": "", "error": "cancelled before it started"})
        await self._close_runner(c)
        return {"name": c.name, "status": c.status, "error": c.error}

    def watch(self, name: str, *, every_turns: int | None = None, every_tokens: int | None = None) -> dict[str, Any]:
        """Emit ``child.progress`` every ``every_turns`` turns and / or ``every_tokens`` tokens (both ``None``: stop)."""
        c = self._get(name)
        for k, v in (("every_turns", every_turns), ("every_tokens", every_tokens)):
            if v is not None and (not _is_int(v) or v < 1):
                raise ValueError(f"{k} must be a positive int or None, got {v!r}")
        p = self._progress(c)
        c.every_turns, c.every_tokens = every_turns, every_tokens
        c.next_turns = (p["turns"] // every_turns + 1) * every_turns if every_turns else 0
        c.next_tokens = (p["tokens"] // every_tokens + 1) * every_tokens if every_tokens else 0
        return {"name": c.name, "every_turns": every_turns, "every_tokens": every_tokens}

    async def _close_runner(self, c: _Child) -> None:
        if c.runner_closed or c.runner is None:
            return
        c.runner_closed = True
        fn = getattr(c.runner, "aclose", None)
        if not callable(fn):
            return
        try:
            r = fn()
            if inspect.isawaitable(r):
                await asyncio.wait_for(r, timeout=max(self.cancel_grace_s, 1.0))
        except Exception as e:  # noqa: BLE001 - a cleanup fault is recorded, never raised into the parent
            c.cleanup_error = f"{type(e).__name__}: {e}"[:500]

    async def aclose(self) -> None:
        """Cancel every child (cascading), await them, close their kernels. Idempotent; concurrent callers share one close."""
        self._closed = True
        if self._closing is None:
            self._closing = asyncio.ensure_future(self._aclose())
        await asyncio.shield(self._closing)

    async def _aclose(self) -> None:
        await asyncio.gather(*(self.cancel(n) for n in list(self._children)), return_exceptions=True)
        for c in self._children.values():
            for t in (c.run_task, c.task):
                if t is not None and not t.done() and not await _hard_cancel(t, self.cancel_grace_s):
                    c.stuck = True

    # ---- kernel host requests -------------------------------------------------------------------------------------------
    def handle(self, request: dict) -> Awaitable[Any]:
        """Serve a kernel host request ``{"type": "agent.<op>", ...}`` (strict: op allowlist, exact keys, types)."""
        return self._handle(request)

    async def _handle(self, request: dict) -> Any:
        if not isinstance(request, dict):
            raise TypeError("an agent request must be a dict")
        t = request.get("type")
        if not isinstance(t, str) or not t.startswith("agent."):
            raise ValueError(f"not an agent request: {t!r}")
        op = t[len("agent."):]
        spec = _OPS.get(op)
        if spec is None:
            raise ValueError(f"unknown agent op {op!r}; allowed: {', '.join(sorted(_OPS))}")
        args = {k: v for k, v in request.items() if k != "type"}
        extra = sorted(set(args) - set(spec))
        if extra:
            raise ValueError(f"agent.{op} got unexpected arguments {extra}; allowed: {sorted(spec)}")
        missing = sorted(k for k, req in spec.items() if req and args.get(k) is None)
        if missing:
            raise ValueError(f"agent.{op} needs {missing}")
        if op == "spawn":
            brief = args.pop("brief")
            if args.get("persistent") is None:
                args.pop("persistent", None)
            if args.get("world") is None:
                args.pop("world", None)
            return await self.spawn(brief, **args)
        if op == "result":
            wait_s = args.get("wait_s")
            return await self.result(args["name"], wait_s=0.0 if wait_s is None else wait_s)
        if op == "send":
            return await self.send(args["name"], args["text"])
        if op == "steer":
            return await self.steer(args["name"], args["text"])
        if op == "cancel":
            return await self.cancel(args["name"])
        if op == "watch":
            return self.watch(args["name"], every_turns=args.get("every_turns"), every_tokens=args.get("every_tokens"))
        if op == "events":
            return self.events()
        return self.list()

    # ---- checkpoint -----------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """This level's rows (capped results) and inbox. Running children are not checkpointed (v1)."""
        rows = []
        for c in self._children.values():
            settled = c.status not in ("starting", "running")
            answer, truncated = (c.answer, c.truncated) if c.restored else _cap_answer(c.name, c.answer) if settled else ("", False)
            rows.append({"name": c.name, "depth": c.depth, "world": c.world, "persistent": c.persistent, "max_turns": c.max_turns,
                         "status": "running" if not settled else c.status, "answer": answer, "answer_chars": c.answer_chars,
                         "truncated": truncated, "error": c.error, "turns": c.last_progress["turns"], "tokens": c.last_progress["tokens"],
                         "elapsed_s": self._elapsed(c), "rounds": c.rounds, "brief_head": c.brief_head, "brief_chars": c.brief_chars})
        return {"version": 1, "path": self.path, "depth": self.depth, "auto": self._auto, "seq": self._seq, "dropped": self._dropped,
                "events": list(self._inbox), "children": rows}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore rows and inbox into a fresh registry. A child that was running is ``failed`` with error ``"lost at
        resume"`` and a ``child.failed`` event (children are not resumed mid-flight in v1); settled children keep their
        capped result but cannot take follow-ups."""
        if self._children or self._inbox:
            raise RuntimeError("load_state_dict needs a fresh ChildRegistry")
        if not isinstance(d, dict) or d.get("version") != 1 or not isinstance(d.get("children"), list):
            raise ValueError("not a ChildRegistry state (version 1)")
        self._auto = int(d.get("auto", 0))
        self._seq = int(d.get("seq", 0))
        self._dropped = int(d.get("dropped", 0))
        for ev in d.get("events") or []:
            if isinstance(ev, dict):
                self._inbox.append(ev)
        lost = []
        for r in d["children"]:
            name = r.get("name")
            if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name in self._children:
                raise ValueError(f"bad child row in ChildRegistry state: {name!r}")
            c = _Child(name=name, depth=int(r["depth"]), world=str(r["world"]), persistent=bool(r["persistent"]),
                       max_turns=int(r["max_turns"]), brief_head=str(r.get("brief_head", "")), brief_chars=int(r.get("brief_chars", 0)),
                       status=str(r["status"]), answer=str(r.get("answer") or ""), answer_chars=int(r.get("answer_chars", 0)),
                       truncated=bool(r.get("truncated")), error=r.get("error"), elapsed=float(r.get("elapsed_s", 0.0)),
                       rounds=int(r.get("rounds", 0)), queued=False, restored=True, runner_closed=True)
            c.last_progress = {"turns": int(r.get("turns", 0)), "tokens": int(r.get("tokens", 0)), "last": ""}
            if c.status not in RUN_STATUSES:
                c.status, c.error = "failed", LOST_AT_RESUME
                lost.append(c)
            c.done.set()
            self._children[name] = c
        for c in lost:
            self._emit({"type": "child.failed", "name": c.name, "status": "failed", "error": LOST_AT_RESUME,
                        "turns": c.last_progress["turns"], "tokens": c.last_progress["tokens"], "round": c.rounds,
                        "note": "The episode resumed from a checkpoint while this child was running; children are not resumed "
                                "mid-flight, so its work in progress is gone (its spend stays charged). Its brief began: "
                                f"{c.brief_head[:200]!r}. Spawn a new child if the work is still needed."})


AGENT_API = r'''
class _Agents:
    """rlm.agent: persistent child agents. A child has its own kernel, context and ledger; it sees only the brief you
    give it (none of your variables or conversation) and runs while you keep working. Its usage counts against the
    shared tree budget, not against your context. Pass a name or the dict spawn() returned."""

    @staticmethod
    def _n(x):
        return x.get("name") if isinstance(x, dict) else x

    async def spawn(self, brief, name=None, persistent=True, world="none", max_turns=None):
        """Start a child on a self-contained brief; returns {"name", "status": "running", ...} at once.
        world: "none" (default: no World access) or "shared" (World calls through your episode)."""
        return await host("agent.spawn", brief=brief, name=name, persistent=persistent, world=world, max_turns=max_turns)

    async def result(self, name, wait_s=0):
        """The child's state: "running" (with progress), or done / failed / cancelled / budget with its answer
        (at most 8,000 characters; the full text is in history.child(name)). wait_s > 0 waits up to that long."""
        return await host("agent.result", name=self._n(name), wait_s=wait_s)

    async def send(self, name, text):
        """Queue a message for the child (a finished persistent child starts a follow-up round)."""
        return await host("agent.send", name=self._n(name), text=text)

    async def steer(self, name, text):
        """Interrupt a running child with an instruction."""
        return await host("agent.steer", name=self._n(name), text=text)

    async def cancel(self, name):
        """Stop the child and all of its own children."""
        return await host("agent.cancel", name=self._n(name))

    async def watch(self, name, every_turns=None, every_tokens=None):
        """Get a child.progress event every N turns and / or N tokens of the child (both None: stop watching)."""
        return await host("agent.watch", name=self._n(name), every_turns=every_turns, every_tokens=every_tokens)

    async def events(self):
        """Drain your children's events: child.completed (last 4 KiB of the answer), child.failed, child.cancelled,
        child.progress, and events.dropped when the inbox overflowed."""
        return await host("agent.events")

    def __repr__(self):
        return "<rlm.agent: spawn, result, send, steer, cancel, watch, events, list>"

    async def list(self):
        """One row per child: name, status, depth, world, turns, tokens, elapsed_s, error."""
        return await host("agent.list")

rlm.agent = _Agents()
'''
