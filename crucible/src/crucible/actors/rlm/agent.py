"""``AgentLoop`` (M16): the very-long-horizon RLM agent loop, one model loop over one kernel, one ledger, one context
manager and one stall detector. The root ``RLMPolicy`` (``policy.py``) is an ``AgentLoop`` with the World ``act()``
bridge; every child agent of the tree is a plain ``AgentLoop`` serving ``tree.ChildRunner``
(``docs/research/long-horizon-rlm/04-design.md`` §3.6, §5).

The three tiers (design §1): the prompt (L1) is bounded by ``context.ContextSpec`` and rebuilt at each compaction; the
kernel (L2, ``kernel.py`` / ``repl.py``, a Docker container by default) keeps every variable across compactions and is
snapshotted every ``snapshot_every`` cells; the ledger (L3, ``ledger.py``) keeps every message, the full output of every
cell and every World result, outside the kernel, readable from cells through ``history`` / ``OUT`` host requests. The
m14 in-kernel ``HISTORY`` archive (capped outputs, ``_archive`` cells that grew the kernel, B10) is gone.

Per turn (§3.6): (1) drain child events, INBOX and messages from the parent (news resets the stall streak); (2) the
pre-send check: the usage-anchored ``ContextMeter`` at or above ``spec.threshold()`` (or a pending ``rlm.compact()``)
compacts first, so no call (root, compaction, child) is sent above the threshold (M16 stress B1: the chars/4 trigger let
prompts reach 1.42x the budget); (3) the call; an overflow, raised or silent, discards the reply, compacts and retries
without charging a turn (B2: a root overflow was masked); with ``context_mode="none"`` it is the scored stop
``context_overflow``; (4) the meter is anchored to the accepted call's usage and the tree is charged
(``TreeSupervisor``); (5) the cell runs; its full output goes to the ledger and a capped head + tail view naming
``OUT(i)`` to the context, with the ``<context_usage>`` footer from ``footer_from`` of the threshold (H15, B9); (6) the
stall verdict (notice, forced reflection, scored stop ``stalled``; B6: a windowed detector replaces the episode-wide
counter) and the 0.9 wrap-up notices.

Compaction (R3, R4): ``context.Compactor`` inside a ``try`` (an escaped exception is a provider fault: masked with its
kind); every attempt is a recorded model call, ``kind="compaction"``, edge ``compaction_attempt``, a tree node in the old
segment that never moves the linear-capture anchor (accepted: trained; rejected: never trained; D's contract, §3.5), and
the first turn after it is edge ``compaction`` with its whole prompt; all attempts failing is the clean typed stop
``compaction_failed`` (a truncation). The rebuilt context is system + task + summary + a capped ``[python-state]`` row +
a boundary marker naming the ledger window + the recent tail (``context.rebuild``); a kernel snapshot is taken and a
ledger window opened at each one; continuations are per segment (B6).

Kernel faults (R12, B10): a cell that times out or a kernel killed mid-cell (an OOM, a crash: the old loop masked it as
``agent_run_error``, E's fault campaign) is restored in the episode from the last snapshot plus the journal suffix
(``Kernel.replay`` falls back to full, then lenient replay) and the model is told what was lost and that the World
actions the cell took still happened; a kernel lost between cells is restored and the cell run again (nothing of it
ran). Journal records are small (B5): ``history.*``, ``context.*`` and ``world.call`` replies are journaled as markers
(``kernel.JournalReply``) and rebuilt from the ledger or the context file on replay.

Training records (design §3.5): every call carries ``agent`` ("" root, a child's path), ``depth``, ``edge``,
``segment`` (per agent), ``window`` and ``ledger_index``; a chain call stores only its prompt delta within a segment
(m13 B-3); compaction attempts and sub-calls are nodes with whole prompts; a child's first call names the root turn
that spawned it (``parent``), so it shares that turn's credit. Every token number names its source (B8).

``RLMConfig`` (the harness id, ``runtime.harness.rlm``) lives here so the actor never imports the runtime.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.actors.llm import BudgetExceeded, LLMError
from crucible.actors.policy import approx_tokens
from crucible.actors.rlm.context import (
    Compactor,
    ContextMeter,
    ContextSpec,
    boundary_marker,
    footer,
    is_overflow,
    is_silent_overflow,
    python_state_row,
    rebuild,
    split_tail,
    summary_request,
    wrap_up_notice,
)
from crucible.actors.rlm.kernel import (
    CellResult,
    HostAbort,
    JournalEntry,
    JournalReply,
    Kernel,
    KernelDivergence,
    KernelError,
    KernelStartError,
    lifetime_for_episode,
)
from crucible.actors.rlm.ledger import _KIND as _LEDGER_KIND
from crucible.actors.rlm.ledger import HISTORY_API, Ledger, LedgerError
from crucible.actors.rlm.stall import StallDetector
from crucible.actors.rlm.tree import AGENT_API, BudgetExhausted, ChildRegistry, TreeLimits, TreeSupervisor

PYTHON_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python code in your persistent REPL. Top-level await works. Variables persist between calls. "
                   "Printed output and the value of a final expression are returned (long output is cut; the full copy is kept as OUT(i)).",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "the Python code to run"}}, "required": ["code"]}}}

# The system prompt (B10 / the real-model finding: Qwen2.5-0.5B called World tools by name and never ran a cell, so the
# prompt says the only tool is python and ends with a worked first cell; E's ladder: the 8,192 cap broke history
# re-reads, so it teaches paging with a cursor). {context}, {compact}, {agents}, {child} are filled per agent.
RLM_SYSTEM = """You are an agent working in a simulated workplace through a persistent Python REPL: the `python` tool, your only tool. Everything is code.
- `TASK` is your instruction. `TOOLS` maps each World tool name ("app.tool") to its description and JSON schema. World tools are called from code, never as tools: `r = await world.call("app.tool", **args)` (or `await world.app.tool(**args)`); `r` is a dict {ok, content, error}. Every result is also appended to `OBS`.
- Message a person: `await world.send(to, text)`. Messages to you are appended to `INBOX` (dicts with from, text, at); new ones are announced.
- Sub-LLMs: `await rlm.query(prompt, context=None)` returns a string; `await rlm.batch([prompt, ...])` runs prompts in parallel. Use them to read, filter or analyse large data instead of printing it.{context}{child}
- Output: what a cell prints is cut to {max_output} characters (head and tail). A cut output names its full copy, e.g. `OUT(12)`: read it in pages, `p = await OUT(12, cursor=0, max_chars=6000)`, `print(p["text"])`, then `cursor=p["next"]` until `p["next"]` is None. Keep large data in variables and print only what you need.
- History: every message, the full output of every cell and every tool result stay in a ledger outside the kernel, also after your context is compacted: `await history.search("text")`, `await history.messages(window=-1)`, `await history.windows()`, `await history.user_messages()`, `await history.get(i)`, `await history.read(i, cursor=0)`. Lists come in pages (`.next`).
- Long work: when your context fills it is compacted: you get the summary you write, a [python-state] row and your recent turns. The kernel keeps running: variables, functions and imports stay.{compact} `await rlm.budget()` shows what is left of your turns, tokens and time.{agents}
- Each python call has a {timeout} s deadline for its own computation (time spent waiting on tools and sub-LLMs does not count). A call that exceeds it is stopped and the kernel is restored to before that call. One python call per reply.
- When the task is complete, call `rlm.finish("<short summary>")`; check the state against every requirement first. If it cannot be done, call `rlm.give_up("<reason>", evidence="<what you found>")`.
A first cell: `print(TASK[:3000]); print(sorted(TOOLS)[:50])`. A World call: `r = await world.call(name, **args); print(r["ok"], str(r["content"])[:2000])`."""

SYSTEM_CONTEXT = ("\n- `CONTEXT` holds a large input outside your prompt and this kernel: `CONTEXT.info()` (its size), `await CONTEXT.read(offset, n)` "
                  "(a string of n characters from a character offset), `await CONTEXT.search(pattern)` (offsets and snippets). Read it in chunks into variables.")
SYSTEM_COMPACT = " `await rlm.compact(instructions=\"\")` writes a checkpoint now and frees context: compact at a natural boundary."
SYSTEM_AGENTS = ("\n- Child agents: `await rlm.agent.spawn(brief, name=None, world=\"none\")` starts a child with its own kernel, context and "
                 "ledger on a self-contained brief (it sees none of your variables; world=\"shared\" lets it call World tools); `await "
                 "rlm.agent.result(name, wait_s=60)`, `await rlm.agent.send(name, text)`, `await rlm.agent.events()`, `await rlm.agent.list()`, "
                 "`await rlm.agent.cancel(name)`. Children share your budget.")
SYSTEM_CHILD_NO_WORLD = "\n- You are a child agent with no World access: `world` calls are refused. Work from your brief, sub-LLMs and your variables."
SYSTEM_CHILD = "\n- You are a child agent: `TASK` is the brief your parent agent gave you; finish with `rlm.finish(answer)`, which is what your parent receives."

# the design's ledger kind for a summary attempt (§3.1); ledger.py's kind pattern allows 16 characters, so until it takes
# "compaction_attempt" (18) the attempts are logged as "compaction_try" (requested of the ledger's owner)
ATTEMPT_KIND = "compaction_attempt" if _LEDGER_KIND.fullmatch("compaction_attempt") else "compaction_try"
GIVE_UP_MARK = "\x1erlm:gave_up\x1e"  # rlm.give_up() travels in FINISHED with this prefix (a model cannot type it by accident)

BOOTSTRAP = r'''
import json as _json
TASK = __TASK__
TOOLS = __TOOLS__
OBS = []
INBOX = []

class _Result(dict):
    __getattr__ = dict.get

class _App:
    def __init__(self, app):
        self._app = app
    def __getattr__(self, tool):
        async def call(**args):
            return await world.call(f"{self._app}.{tool}", **args)
        return call

class _World:
    async def call(self, tool, **args):
        r = await host("world.call", tool=tool, args=args)
        INBOX.extend(r.pop("inbox", []))
        res = _Result(r)
        OBS.append({"tool": tool, "args": args, **res})
        return res
    async def send(self, to, text):
        r = await host("world.send", to=to, text=text)
        INBOX.extend(r.pop("inbox", []))
        return _Result(r)
    def __getattr__(self, app):
        if app.startswith("_"):
            raise AttributeError(app)
        return _App(app)

class _RLM:
    async def query(self, prompt, context=None, max_tokens=None):
        return await host("llm.query", prompt=str(prompt), context=None if context is None else str(context), max_tokens=max_tokens)
    async def batch(self, prompts, max_tokens=None):
        items = [p if isinstance(p, (list, tuple)) else (p, None) for p in prompts]
        return await host("llm.batch", items=[[str(p), None if c is None else str(c)] for p, c in items], max_tokens=max_tokens)
    def finish(self, summary=""):
        globals()["FINISHED"] = str(summary)
        return "finished"
    def give_up(self, reason, evidence=""):
        """End the episode: the task cannot be done (state why, with evidence). Scoring is the task's."""
        globals()["FINISHED"] = "\x1erlm:gave_up\x1e" + _json.dumps({"reason": str(reason)[:2000], "evidence": str(evidence)[:4000]})
        return "gave up"
    async def compact(self, instructions=""):
        """Write a checkpoint of your work now and free context (before your next turn)."""
        return await host("rlm.compact", instructions=str(instructions)[:2000])
    async def budget(self):
        """What is left of your turns, tokens, sub-calls and time, and how full your context is."""
        return await host("rlm.budget")
    async def note(self, text):
        return await host("rlm.note", text=str(text)[:512])

world = _World()
rlm = _RLM()
def _deliver(msgs):
    INBOX.extend(msgs)
'''

CONTEXT_API = r'''
class _Context:
    """CONTEXT: a large input kept outside your prompt and this kernel, read in pieces through host requests."""
    def __init__(self, info):
        self._info = dict(info)
    def info(self):
        return dict(self._info)
    def __len__(self):
        return int(self._info.get("chars", 0))
    async def read(self, offset=0, n=65536):
        """n characters from a character offset (a string; empty past the end)."""
        return await host("context.read", offset=int(offset), n=int(n))
    async def search(self, pattern, regex=False, limit=20, context_chars=200):
        """{"items": [{"offset", "snippet"}], "truncated", "timed_out"}: substring (regex=True for a regex), bounded."""
        return await host("context.search", pattern=str(pattern), regex=bool(regex), limit=int(limit), context_chars=int(context_chars))
    def __repr__(self):
        return (f"<CONTEXT: {self._info.get('chars', 0):,} characters; CONTEXT.info(), await CONTEXT.read(offset, n), "
                "await CONTEXT.search(pattern)>")

CONTEXT = _Context(__INFO__)
'''

CONTINUE = ("[goal: continuation] You stopped without calling rlm.finish(). The task is not marked complete. Check the current state "
            "against every requirement in TASK (use your variables and the tools), finish what is missing, then call rlm.finish(summary).")

# A stop is typed (design §3.6). Truncations: the episode ran out of something, its partial outcome is not the policy's
# final answer (critics bootstrap V(s_T), M12); scored: the policy's own outcome; ``policy_error``: masked (infra kinds).
TRUNCATING = ("compaction_failed", "too_many_compactions", "context_overflow")
# a World-acting root (one that has stepped the World) stops "stalled" when this many threshold compactions in a row saw no
# World action between them: a context loop (re-reading its own past, compacting, re-reading...) that no cell-level stall
# signature sees, because each round's cells differ (M16 validation: 2,814 compactions at the 5,000 rung)
NO_PROGRESS_COMPACTIONS = 8


def truncating_stop(stop: str | None) -> bool:
    """A typed stop that is a truncation: ``budget:<kind>``, ``compaction_failed``, ``too_many_compactions``,
    ``context_overflow``."""
    return bool(stop) and (stop.startswith("budget:") or stop in TRUNCATING)


# ---- the config (the harness id) ----------------------------------------------------------------------------------------
STALL_FIELDS = {"stall_repeat_cells": "repeat_cells", "stall_same_call_window": "same_call_window", "stall_same_call_hits": "same_call_hits",
                "stall_static_window": "static_results_window", "stall_max_cell_errors": "max_cell_errors",
                "stall_escalate_after": "escalate_after", "stall_stop_after": "stop_after"}
# the m14 fields: a config that sets only these keeps one digest whatever M16 fields are added later at their defaults
M14_FIELDS = ("max_output_chars", "cell_timeout_s", "max_tokens", "sub_max_tokens", "temperature", "max_turns", "max_continuations", "max_stalls",
              "tool_index", "kernel", "kernel_image", "allow_host", "max_sub_calls", "max_sub_calls_per_cell", "sub_concurrency", "max_sub_tokens",
              "budget_tokens", "trigger", "keep_recent_turns", "summary_max_tokens")


@dataclass(frozen=True)
class RLMConfig:
    """The RLM harness's configuration; its digest is the harness id (``rlm:<digest>@<version>``), so one GRPO group never
    mixes configurations. JSON-serialisable, frozen.

    m14 fields keep their names: ``budget_tokens`` is the window (``window_tokens`` overrides it; 0 turns compaction off,
    ``context_mode="none"``); ``trigger`` and ``keep_recent_turns`` are ignored (the threshold is ``ContextSpec.threshold()``
    on usage-anchored tokens, B1, and the kept tail is ``tail_tokens``) and a non-default value is recorded in the
    harness notes; ``max_stalls``, ``max_sub_calls`` and ``max_sub_tokens`` (0 = unset) override ``stall_stop_after``,
    ``tree_max_sub_calls`` and ``tree_max_sub_tokens``. ``max_turns`` 0 = derived (the tree's turn budget; B6: the m14
    default 500 ended honest episodes); ``max_continuations`` is per segment (B6).

    M16 fields: the context (``window_tokens``, ``seq_len``, ``tail_tokens``, ``reserve_tokens`` (None: min(16,384,
    window // 8), so a small window is not halved by a reserve sized for 128k), ``max_compaction_attempts``,
    ``max_compactions``, ``context_mode`` auto | action | none, ``footer_from``), kernel snapshots (``snapshot_every``
    cells; 0 = only at compactions), the episode wall budget ``wall_s`` (the container's lifetime, the footer's time
    left, the stop ``budget:wall_s``), the stall thresholds (``stall_*``, ``stall.StallDetector``) and the tree limits
    (``tree_*``, ``tree.TreeLimits``)."""

    max_output_chars: int = 8192  # the RLM default: large data stays in variables
    cell_timeout_s: float = 120.0
    max_tokens: int = 2000
    sub_max_tokens: int = 1000
    temperature: float = 1.0  # RL samples; set 0 for a deterministic evaluation
    max_turns: int = 0
    max_continuations: int = 2
    max_stalls: int = 0
    tool_index: bool = True  # a one-line-per-tool index after the task (schemas stay in TOOLS)
    # where model-written code runs: "docker" (a no-network, read-only, capability-free container; the default for any
    # rollout) or "local" (a host subprocess: trusted code and unit tests only, refused unless ``allow_host``; M14 audit H3 V1)
    kernel: str = "docker"
    kernel_image: str = "python:3.12-slim"
    allow_host: bool = False
    max_sub_calls: int = 0
    max_sub_calls_per_cell: int = 64
    sub_concurrency: int = 4
    max_sub_tokens: int = 0
    budget_tokens: int = 128_000
    trigger: float = 0.8
    keep_recent_turns: int = 3
    summary_max_tokens: int = 2000
    # ---- M16: context
    window_tokens: int = 0
    seq_len: int | None = None
    tail_tokens: int = 20_000
    reserve_tokens: int | None = None
    max_compaction_attempts: int = 5
    max_compactions: int | None = None
    context_mode: str = "auto"
    footer_from: float = 0.8
    # ---- M16: kernel and episode
    snapshot_every: int = 200
    wall_s: float | None = None
    # ---- M16: stall detection (stall.StallDetector)
    stall_repeat_cells: int = 3
    stall_same_call_window: int = 10
    stall_same_call_hits: int = 6
    stall_static_window: int = 100
    stall_max_cell_errors: int = 4
    stall_escalate_after: int = 3
    stall_stop_after: int = 6
    # ---- M16: the agent tree (tree.TreeLimits)
    tree_max_total_new_tokens: int = 10_000_000
    tree_max_total_turns: int = 20_000
    tree_max_spawns: int = 64
    tree_max_depth: int = 1
    tree_max_concurrent: int = 8
    tree_reserve_per_depth: int = 1
    tree_child_wall_s: float = 3_600.0
    tree_child_max_turns: int = 2_000
    tree_max_sub_calls: int = 4_096
    tree_max_sub_tokens: int = 20_000_000
    tree_max_events: int = 10_000
    tree_max_event_bytes: int = 65_536
    # ---- M16 hunt H9: the ledger's disk cap for the whole tree (bytes of text logged; past it a text field keeps its head)
    ledger_max_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        if self.context_mode not in ("auto", "action", "none"):
            raise ValueError(f"RLMConfig.context_mode: auto | action | none (got {self.context_mode!r})")
        if self.snapshot_every < 0:
            raise ValueError("RLMConfig.snapshot_every must be >= 0")

    def digest(self) -> str:
        """sha256 of the m14 fields plus every M16 field that differs from its default (so later fields at their default
        never change an id)."""
        d = asdict(self)
        base = RLMConfig.__dataclass_fields__
        keep = {k: v for k, v in d.items() if k in M14_FIELDS or v != base[k].default}
        return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]

    # ---- derived parts
    def window(self) -> int:
        return int(self.window_tokens or self.budget_tokens or 128_000)

    def mode(self) -> str:
        return "none" if not (self.window_tokens or self.budget_tokens) else self.context_mode

    def max_out(self) -> int:
        """The turn's max_tokens, never more than a quarter of the window (a reply as large as the window cannot fit)."""
        return max(1, min(int(self.max_tokens), self.window() // 4))

    def context_spec(self) -> ContextSpec:
        w = self.window()
        reserve = self.reserve_tokens if self.reserve_tokens is not None else min(16_384, w // 8)
        return ContextSpec(window_tokens=w, max_out=self.max_out(), seq_len=self.seq_len, reserve=reserve, tail_tokens=self.tail_tokens,
                           summary_max_tokens=self.summary_max_tokens, max_compaction_attempts=self.max_compaction_attempts,
                           max_compactions=self.max_compactions, mode=self.mode(), footer_from=self.footer_from)

    def tree_limits(self) -> TreeLimits:
        m = {k[5:]: v for k, v in asdict(self).items() if k.startswith("tree_")}
        if self.max_sub_calls:
            m["max_sub_calls"] = self.max_sub_calls
        if self.max_sub_tokens:
            m["max_sub_tokens"] = self.max_sub_tokens
        return TreeLimits.from_mapping(m)

    def stall_kwargs(self) -> dict[str, int]:
        kw = {v: getattr(self, k) for k, v in STALL_FIELDS.items()}
        if self.max_stalls:
            kw["stop_after"] = self.max_stalls
        kw["escalate_after"] = min(kw["escalate_after"], kw["stop_after"])
        return kw

    def agent_max_turns(self) -> int:
        return int(self.max_turns or self.tree_max_total_turns)

    def notes(self) -> list[str]:
        """What the config asked for that this harness maps or ignores (recorded in the evidence)."""
        out = []
        if self.trigger != 0.8:
            out.append(f"trigger={self.trigger} ignored: the compaction threshold is ContextSpec.threshold() on usage-anchored tokens (M16 B1)")
        if self.keep_recent_turns != 3:
            out.append(f"keep_recent_turns={self.keep_recent_turns} ignored: the kept tail is tail_tokens={self.tail_tokens}")
        if self.max_out() < self.max_tokens:
            out.append(f"max_tokens={self.max_tokens} clamped to {self.max_out()} (a quarter of the {self.window()}-token window)")
        return out

    def launcher(self) -> Any:
        from crucible.actors.rlm.kernel import DockerLauncher, LocalLauncher

        if self.kernel == "docker":
            # M16 B9: the container lives as long as the episode's wall budget (+ margin) and is renewed while it runs
            return DockerLauncher(self.kernel_image, lifetime_s=lifetime_for_episode(self.wall_s), renewable=True)
        if self.kernel == "local":
            if not self.allow_host:
                raise ValueError("RLMConfig.kernel='local' runs model-written code on the host: refused unless allow_host=True "
                                 "(trusted code only; rollouts and evaluations use kernel='docker')")
            return LocalLauncher(allow_host=True)
        raise ValueError(f"RLMConfig.kernel: docker | local (got {self.kernel!r})")

    def policy_kwargs(self) -> dict[str, Any]:
        return {"config": self}


def clone_launcher(launcher: Any) -> Any:
    """A fresh launcher of the same kind for a child agent's own kernel (a child never shares its parent's process)."""
    from crucible.actors.rlm.kernel import DockerLauncher, LocalLauncher

    if hasattr(launcher, "clone") and callable(launcher.clone):
        return launcher.clone()
    if isinstance(launcher, DockerLauncher):
        return DockerLauncher(launcher.image, memory=launcher.memory, pids=launcher.pids, cpus=launcher.cpus, docker=launcher.docker,
                              lifetime_s=launcher.lifetime_s, renewable=launcher.renewable)
    if isinstance(launcher, LocalLauncher):
        return LocalLauncher(launcher.python, allow_host=launcher.allow_host)
    raise KernelStartError(f"a child agent needs its own kernel, and {type(launcher).__name__} cannot make another "
                           "(pass child_launcher=<factory> to RLMPolicy)", "crucible:harness_unsupported")


# ---- shared state of one agent tree ---------------------------------------------------------------------------------------
LEDGER_CUT_HEAD = 4_096  # past the ledger cap, a text field keeps this many characters
# World results are what a replay rebuilds a cell's ``world.call`` reply from (``_world_reply``): never cut
LEDGER_UNCUT = {"world_call": ("content", "error", "inbox"), "world_send": ("inbox",)}


class LedgerBudget:
    """(M16 hunt H9) The tree's ledger disk cap. Every cell's full output (up to 1 MiB per stream), every sub-call and
    every World argument went to disk with no bound: a policy printing 3 MiB per cell over 20,000 turns writes ~60 GB of
    host disk, and distinct 16 MiB ``world.call`` arguments defeat the blob dedup. Past ``max_bytes`` (counted over the
    UTF-8 text the loop logs, before dedup, so an over-estimate) each text field keeps its first ``LEDGER_CUT_HEAD``
    characters and a marker; World results are kept whole (a replay reads them back). The cap is the policy's scored
    business (its own outputs are cut), never an infra fault."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self.used = 0
        self.cut_fields = 0
        self.cut_chars = 0

    @staticmethod
    def _size(v: Any) -> int:
        if isinstance(v, str):
            return len(v.encode("utf-8", "replace"))
        if isinstance(v, (dict, list, tuple)):
            return len(json.dumps(v, ensure_ascii=False, default=str).encode("utf-8", "replace"))
        return 16

    def fit(self, fields: dict[str, Any], kind: str = "") -> dict[str, Any]:
        size = sum(self._size(v) for v in fields.values())
        if self.used + size > self.max_bytes:
            keep = LEDGER_UNCUT.get(kind, ())
            out = {}
            for k, v in fields.items():
                if isinstance(v, str) and len(v) > LEDGER_CUT_HEAD and k not in keep:
                    out[k] = (v[:LEDGER_CUT_HEAD] + f"\n[... {len(v) - LEDGER_CUT_HEAD} characters not kept: the episode's ledger reached "
                              f"its {self.max_bytes}-byte cap ...]")
                    self.cut_fields += 1
                    self.cut_chars += len(v) - LEDGER_CUT_HEAD
                else:
                    out[k] = v
            fields, size = out, sum(self._size(v) for v in out.values())
        self.used += size
        return fields

    def stats(self) -> dict[str, int]:
        return {"used": self.used, "max_bytes": self.max_bytes, "cut_fields": self.cut_fields, "cut_chars": self.cut_chars}


class CallLog:
    """The episode's model calls (every agent's, one list: the record builder keys chains by agent). ``path``: a stream
    file the checkpoint appends settled calls to, so a checkpoint holds an offset instead of every token record (B5)."""

    def __init__(self, actor: str, path: Path | None = None):
        self.actor, self.path = actor, Path(path) if path else None
        self.items: list[dict[str, Any]] = []
        self.n = 0
        self.flushed = 0

    def new_id(self) -> str:
        self.n += 1
        return f"{self.actor}:c{self.n}"

    def append(self, call: dict[str, Any]) -> dict[str, Any]:
        self.items.append(call)
        return call

    def checkpoint(self, keep_from: int) -> dict[str, Any]:
        """Settled calls (index < ``keep_from``) go to the stream; the rest travel inline."""
        keep_from = max(self.flushed, min(keep_from, len(self.items)))
        if self.path is None:
            return {"n": self.n, "path": None, "streamed": 0, "inline": self.items}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.flushed else "w"
        with open(self.path, mode) as f:
            f.writelines(json.dumps(c, default=str) + "\n" for c in self.items[self.flushed:keep_from])
        self.flushed = keep_from
        return {"n": self.n, "path": str(self.path), "streamed": keep_from, "inline": self.items[keep_from:]}

    def load(self, d: dict[str, Any]) -> None:
        items: list[dict[str, Any]] = []
        if d.get("path") and d.get("streamed"):
            with open(d["path"]) as f:
                for k, line in enumerate(f):
                    if k >= int(d["streamed"]):
                        break
                    items.append(json.loads(line))
            if len(items) != int(d["streamed"]):
                raise KernelStartError(f"the model-call stream holds {len(items)} of the {d['streamed']} calls the checkpoint names",
                                       "crucible:backend_error")
        self.items = items + list(d.get("inline") or [])
        self.n = int(d.get("n") or len(self.items))
        self.flushed = 0  # the next checkpoint writes the whole list to this attempt's own stream


@dataclass
class Tree:
    """What every agent of one episode shares: the config, the supervisor (one budget), the call log, the segment list,
    the agents by path (``root``, ``root/scan``...), the World step, and the first infra fault any agent hit (a child's
    provider or sandbox fault masks the episode like the root's)."""

    actor: str
    cfg: RLMConfig
    spec: ContextSpec
    supervisor: TreeSupervisor
    calls: CallLog
    launcher_factory: Callable[[], Any]
    agents: dict[str, AgentLoop] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    fault: tuple[str, str] | None = None
    trajectory_tokens: int = 0
    trajectory_sources: set[str] = field(default_factory=set)
    sample_key: str | None = None
    started: float = field(default_factory=time.monotonic)
    ledger_budget: LedgerBudget | None = None

    def __post_init__(self) -> None:
        if self.ledger_budget is None:
            self.ledger_budget = LedgerBudget(self.cfg.ledger_max_bytes)

    @property
    def root(self) -> AgentLoop:
        return self.agents["root"]

    def make_child(self, *, name: str, brief: str, depth: int, world: str, persistent: bool, max_turns: int, path: str,
                   registry: ChildRegistry) -> AgentLoop:
        """``ChildRegistry``'s factory: a child ``AgentLoop`` with its own kernel (a new launcher), its own ledger (the
        parent's ``ledger.child(name)``) and a fresh context from ``brief``."""
        parent = self.agents[path.rsplit("/", 1)[0]]
        if world == "shared" and parent.depth > 0 and parent.world_grant != "shared":  # (M16 hunt H2) defence in depth
            raise PermissionError("a child agent without World access cannot give World access to its own children")
        agent = path.split("/", 1)[1]
        child = AgentLoop(tree=self, llm=parent.llm, sub_llm=parent.sub_llm, sub_trainable=parent.sub_trainable, launcher=self.launcher_factory(),
                          name=agent, depth=depth, ledger=parent.ledger.child(name), registry=registry, world_grant=world, max_turns=max_turns,
                          seed=zlib.crc32(f"{self.sample_key}:{self.actor}:{agent}".encode()), task=brief)
        child.spawn_parent = parent._current_call["call_id"] if parent._current_call else None
        self.agents[path] = child
        return child


# ---- CONTEXT: a large input as a host file --------------------------------------------------------------------------------
CONTEXT_READ_MAX = 131_072
_SEARCH_WORKER = r'''
import json, re, sys
job = json.loads(sys.stdin.buffer.read())
text = open(job["path"], encoding="utf-8", errors="replace").read()
rx = re.compile(job["pattern"])
out = []
for m in rx.finditer(text):
    p = m.start()
    out.append({"offset": p, "snippet": text[max(0, p - job["ctx"]): m.end() + job["ctx"]][: 2 * job["ctx"] + 2000]})
    if len(out) >= job["limit"]:
        break
sys.stdout.write(json.dumps(out))
'''


class ContextFile:
    """``CONTEXT`` (design §3.6): a large input the kernel reads through ``context.read`` / ``context.search`` host
    requests, served from a host file (character offsets). Substring search is bounded at 2 s; a regex runs in a worker
    process killed at 2 s (Python's ``re`` backtracks without bound and holds the GIL, as in ``ledger.py``)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._text: str | None = None

    def text(self) -> str:
        if self._text is None:
            self._text = self.path.read_text(encoding="utf-8", errors="replace")
        return self._text

    def info(self) -> dict[str, Any]:
        return {"chars": len(self.text()), "bytes": self.path.stat().st_size, "name": self.path.name}

    def preview(self, n: int = 500) -> str:
        return self.text()[:n]

    def read(self, offset: Any, n: Any) -> str:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("CONTEXT.read: offset must be a non-negative int")
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= CONTEXT_READ_MAX:
            raise ValueError(f"CONTEXT.read: n must be an int in [1, {CONTEXT_READ_MAX}]")
        return self.text()[offset: offset + n]

    def search(self, pattern: Any, regex: Any = False, limit: Any = 20, context_chars: Any = 200, *, bound_s: float = 2.0) -> dict[str, Any]:
        if not isinstance(pattern, str) or not 1 <= len(pattern) <= 1000:
            raise ValueError("CONTEXT.search: pattern must be a string of 1..1000 characters")
        limit = int(limit) if isinstance(limit, int) and not isinstance(limit, bool) else 20
        ctx = int(context_chars) if isinstance(context_chars, int) and not isinstance(context_chars, bool) else 200
        limit, ctx = max(1, min(limit, 100)), max(0, min(ctx, 2000))
        if regex:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"CONTEXT.search: invalid regex: {e}") from None
            job = json.dumps({"path": str(self.path), "pattern": pattern, "limit": limit, "ctx": ctx}).encode()
            try:
                r = subprocess.run([sys.executable, "-I", "-S", "-c", _SEARCH_WORKER], input=job, capture_output=True, timeout=bound_s, env={}, check=False)
                return {"items": json.loads(r.stdout or b"[]"), "truncated": False, "timed_out": False}
            except subprocess.TimeoutExpired:
                return {"items": [], "truncated": True, "timed_out": True}
        text, items, start, deadline = self.text(), [], 0, time.monotonic() + bound_s
        while len(items) < limit and time.monotonic() < deadline:
            p = text.find(pattern, start)
            if p < 0:
                break
            items.append({"offset": p, "snippet": text[max(0, p - ctx): p + len(pattern) + ctx]})
            start = p + 1
        return {"items": items, "truncated": len(items) >= limit, "timed_out": time.monotonic() >= deadline}


LEDGER_STRUCT_WIRE = 16_384  # (M16 hunt H1) a dict / list field above this many wire bytes is logged as its JSON text


def _ledger_field(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)) and v:
        try:
            if len(json.dumps(v, default=str)) > LEDGER_STRUCT_WIRE:
                return json.dumps(v, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return repr(v)[:65_536]
    return v


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


# ---- the loop ---------------------------------------------------------------------------------------------------------------
class AgentLoop:
    """One agent of the RLM tree (see the module docstring). The root is ``policy.RLMPolicy``; a child is built by
    ``Tree.make_child`` and serves ``tree.ChildRunner`` (``run``, ``deliver``, ``cancel``, ``progress``, ``aclose``)."""

    def __init__(self, *, tree: Tree, llm: Any, sub_llm: Any = None, sub_trainable: bool | None = None, launcher: Any = None, name: str = "",
                 depth: int = 0, ledger: Ledger | None = None, registry: ChildRegistry | None = None, world_grant: str = "root",
                 max_turns: int | None = None, seed: int | None = None, task: str = "", system: str | None = None):
        cfg = tree.cfg
        self.tree, self.cfg, self.spec = tree, cfg, tree.spec
        self.actor_id = tree.actor
        self.agent, self.depth, self.world_grant = name, depth, world_grant
        self.name = name.rsplit("/", 1)[-1]  # tree.ChildRunner's name: the child's own name under its parent
        self.llm, self.sub_llm = llm, sub_llm or llm
        # sub-calls answered by another model are recorded with that model and never trained (M14 audit G1 D1)
        self.sub_trainable = (sub_llm is None or sub_llm is llm) if sub_trainable is None else bool(sub_trainable)
        self.launcher = launcher
        self.seed = seed
        self.ledger = ledger
        self.registry = registry
        self.max_turns = int(max_turns or cfg.agent_max_turns())
        self.temperature, self.max_out = cfg.temperature, cfg.max_out()
        self.max_output_chars, self.cell_timeout_s = cfg.max_output_chars, cfg.cell_timeout_s
        self.threshold = self.spec.threshold()
        self.meter = ContextMeter()
        self.stall = StallDetector(**cfg.stall_kwargs())
        self.context_file: ContextFile | None = None
        self.task_text = task
        self.system = self._system_prompt(system)
        self.messages: list[dict[str, Any]] = []
        self.kernel: Kernel | None = None
        self.stop: str | None = None
        self.stop_reason = ""
        self.error: str | None = None
        self.failure_kind: str | None = None
        self.answer: str | None = None
        self.notes: list[str] = []
        self.segment = 0
        self.spawn_parent: str | None = None
        self.counters: dict[str, Any] = {
            "turns": 0, "cells": 0, "continuations": 0, "stalls": 0, "world_calls": 0, "world_sends": 0, "sub_calls": 0, "sub_tokens": 0,
            "sub_refused": 0, "refused_calls": 0, "misrouted_calls": 0, "restores": 0, "inexact_restores": 0, "lost_between_cells": 0,
            "killed_mid_cell": 0, "protocol_errors": 0, "compactions": 0, "compaction_attempts": 0, "compaction_failures": 0,
            "overflow_retries": 0, "history_reads": 0, "history_reads_after_compaction": 0, "context_reads": 0, "snapshots": 0,
            "snapshot_failures": 0, "renewals": 0, "agent_requests": 0, "completion_tokens": 0, "discarded_cost_usd": 0.0,
            "stray_requests": 0, "compact_requests_refused": 0, "snapshots_skipped": 0}
        self.health: dict[str, Any] = {"turns_between_compactions": [], "chars_dropped": [], "summary_chars": [], "stall_verdicts": {},
                                       "restores_by_mode": {}, "largest_prompt": {}, "overflows": []}
        self._continuations = 0  # per segment (B6)
        self._last_summary: str | None = None
        self._compact_request: str | None = None
        self._turn_paid = False
        self._overflows_in_row = 0
        self._turn_at_compaction = 0
        self._wrapped: set[str] = set()
        self._lost_names: list[str] = []
        self._current_call: dict[str, Any] | None = None
        self._cell_msgs: list[dict[str, Any]] = []
        self._cell_sub_calls = 0
        self._cell_world = 0
        self._sub_sem: asyncio.Semaphore | None = None
        self._last_full: list[int] | None = None
        self._last_seg: int | None = None
        self._chain_calls_in_segment = 0
        self._prev_total: int | None = None  # the previous chain call's prompt + completion (B8 accounting)
        self._anchor_n: int | None = None  # messages covered by the meter's usage anchor
        self._next_edge: str | None = None
        self._delivered: list[tuple[str, str]] = []
        self._cancelled = False
        self._last_output = ""
        self._snapshots_on = True
        self._snap_paths: list[Path] = []
        self._ckpt_snapshot: str | None = None
        self._cell_elapsed: float | None = None
        self._rounds = 0
        self._parent_sent = False
        self._steps_at_compaction: list[int] = []
        self._ledger_opened = False

    # ---- the prompt pieces --------------------------------------------------------------------------------------------
    def _system_prompt(self, override: str | None) -> str:
        cfg = self.cfg
        child = ""
        if self.depth > 0:
            child = SYSTEM_CHILD + (SYSTEM_CHILD_NO_WORLD if self.world_grant != "shared" else "")
        agents = SYSTEM_AGENTS if self.depth < self.tree.supervisor.limits.max_depth else ""
        text = (override or RLM_SYSTEM).replace("{context}", "{__ctx__}").replace("{child}", child).replace("{agents}", agents)
        text = text.replace("{compact}", SYSTEM_COMPACT if self.spec.mode != "none" else "")
        return text.replace("{max_output}", str(cfg.max_output_chars)).replace("{timeout}", f"{cfg.cell_timeout_s:g}")

    def mount_context(self, path: Path | str | None) -> None:
        """Serve ``path`` as the kernel's ``CONTEXT`` (before the episode starts)."""
        if path is None:
            return
        self.context_file = ContextFile(path)

    def _final_system(self) -> str:
        return self.system.replace("{__ctx__}", SYSTEM_CONTEXT if self.context_file is not None else "")

    def _tools_view(self) -> dict[str, Any]:
        return {} if self.depth > 0 and self.world_grant != "shared" else dict(self.tree.root._tools)

    def _task_message(self) -> str:
        """The task as the model first sees it: the instruction, a one-line-per-tool index (the full schemas stay in
        ``TOOLS``: in the m14 live smoke a 4B model never read ``TOOLS``), and the ``CONTEXT`` size and a 500-character
        preview when one is mounted."""
        text = self.task_text
        tools = self._tools_view()
        if self.cfg.tool_index and tools:
            lines = [f"- {name}: {(spec.get('description') or '').strip().splitlines()[0][:100] if spec.get('description') else ''}"
                     for name, spec in sorted(tools.items())]
            text += ("\n\n[tools] Everything in this workplace (tickets, orders, the knowledge base) is reached through these tools, never "
                     "through local files: `await world.call(name, **args)`. Schemas: TOOLS[name][\"schema\"].\n" + "\n".join(lines))
        if self.context_file is not None:
            info = self.context_file.info()
            text += (f"\n\n[context] CONTEXT holds {info['chars']:,} characters (read it with `await CONTEXT.read(offset, n)`). It begins:\n"
                     + self.context_file.preview(500))
        return text

    # ---- small helpers ------------------------------------------------------------------------------------------------
    @property
    def model_calls(self) -> list[dict[str, Any]]:
        return self.tree.calls.items

    @property
    def segments(self) -> list[dict[str, Any]]:
        return [s for s in self.tree.segments if s.get("agent") == self.agent]

    @property
    def sample_key(self) -> str | None:
        k = self.tree.sample_key
        return k if not self.agent else f"{k}/{self.agent}"

    def _log(self, kind: str, **fields: Any) -> int:
        """One ledger event at the current World step; a ledger fault is infra (masked, ``crucible:backend_error``).

        (M16 hunt H1) A structured field the policy controls (``world.call`` arguments, a restore's lost names) whose
        ASCII-escaped JSON is large is stored as its JSON text: the ledger's history replies cut text fields to fit
        their 256 KiB cap but cannot cut a dict or a list, so ``history.get`` of a call with 60,000 CJK characters in
        its arguments raised ``LedgerError`` and masked the policy's own episode as ``crucible:backend_error``.
        (M16 hunt H9) Then the tree's ledger disk cap (``LedgerBudget``)."""
        fields = {k: _ledger_field(v) for k, v in fields.items()}
        if kind in ("restore", "notice", "snapshot"):
            # (M16 hunt H6) harness texts (a restore's fallback reasons, a failed snapshot) named host paths ("the snapshot file is
            # missing: /var/.../ledger/root/.snapshots/..."), and the kernel reads every ledger event through history.*
            fields = {k: self._unpath(v) for k, v in fields.items()}
        fields = self.tree.ledger_budget.fit(fields, kind)
        try:
            return self.ledger.append(kind, step=self.tree.step, **fields)
        except LedgerError as e:
            raise HostAbort(f"the episode ledger failed: {e}", "crucible:backend_error") from e
        except ValueError:  # a field that is not JSON (a provider's raw object): kept as text
            return self.ledger.append(kind, step=self.tree.step, **{k: v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)[:65_536]
                                                                   for k, v in fields.items()})

    def _unpath(self, v: Any) -> Any:
        """(M16 hunt H6) The ledger root's host path replaced by ``<ledger>`` in a harness-written field."""
        root = self.tree.agents.get("root")
        led = root.ledger if root is not None and root.ledger is not None else self.ledger
        if led is None:
            return v
        if isinstance(v, str):
            for base in {str(led.dir.parent), str(led.dir.parent.resolve())}:
                v = v.replace(base, "<ledger>")
            return v
        if isinstance(v, list):
            return [self._unpath(x) for x in v]
        return v

    def _window(self) -> int:
        return self.ledger.window if self.ledger is not None else 0

    def _compress(self, tok: dict[str, Any] | None) -> dict[str, Any] | None:
        """Linear token capture (m13 B-3, ``LLMToolPolicy._compress``) per agent chain: a call extending the previous one in
        this segment stores only its prompt delta and the prefix length."""
        if not tok or tok.get("prompt_token_ids") is None:
            return tok
        full = list(tok["prompt_token_ids"])
        out = dict(tok)
        if self._last_full is not None and self._last_seg == self.segment and full[: len(self._last_full)] == self._last_full:
            out["prompt_token_ids"] = None
            out["prompt_delta"] = full[len(self._last_full):]
            out["prompt_prefix_len"] = len(self._last_full)
        self._last_full = full + list(tok.get("completion_token_ids") or [])
        self._last_seg = self.segment
        return out

    def _budgets(self) -> dict[str, tuple[int, int]]:
        u = self.tree.supervisor.usage()
        lim = self.tree.supervisor.limits
        b = {"turns": (self.max_turns - self.counters["turns"], self.max_turns), "tree_turns": (u["left"]["turns"], lim.max_total_turns),
             "new_tokens": (u["left"]["new_tokens"], lim.max_total_new_tokens), "sub_calls": (u["left"]["sub_calls"], lim.max_sub_calls)}
        if self.cfg.wall_s:
            b["wall_s"] = (int(self.cfg.wall_s - (time.monotonic() - self.tree.started)), int(self.cfg.wall_s))
        return b

    def _end(self, stop: str, reason: str = "") -> None:
        """A typed stop (design §3.6). ``policy_error`` carries ``failure_kind`` (masked); a child's infra fault is the
        tree's (it masks the episode as the root's would)."""
        if self.stop is not None:
            return
        self.stop, self.stop_reason = stop, reason
        if stop == "policy_error" and self.depth > 0 and self.tree.fault is None:
            self.tree.fault = (self.failure_kind or "provider_unavailable", f"child agent {self.agent}: {self.error}")
        if self.ledger is not None:
            with contextlib.suppress(Exception):
                self._log("notice", text=f"[stop: {stop}] {reason}"[:4000], stop=stop)

    def _provider_fault(self, e: BaseException, what: str) -> None:
        self.failure_kind = self.failure_kind or getattr(e, "failure_kind", None) or "provider_unavailable"
        self.error = self.error or f"{self.failure_kind}: {what}: {type(e).__name__}: {str(e)[:500]}"
        self._end("policy_error", self.error)

    # ---- start --------------------------------------------------------------------------------------------------------
    def _make_ledger(self) -> Ledger:
        raise KernelStartError("no ledger for this agent", "crucible:backend_error")

    def _open_ledger(self) -> None:
        if self.ledger is None:
            self.ledger = self._make_ledger()
        if self.ledger.window < 0:
            self.ledger.open_window("start")
        elif self.depth > 0 and not self._ledger_opened and len(self.ledger):
            # (M16 hunt H10) a child spawned after a resume under the name of a child the crashed attempt ran: its ledger holds
            # that attempt's (abandoned) events, so this child starts in a resume window of its own
            self.ledger.open_window("resume", step=self.tree.step)
        self._ledger_opened = True

    async def _ensure_kernel(self) -> None:
        """The kernel, started and bootstrapped; a kernel lost between cells (or closed by a failed snapshot) is restored
        from its snapshot + journal first (G's note: restore before the next cell)."""
        self._open_ledger()
        if self.kernel is not None and self.kernel.proc is not None:
            return
        if self.kernel is not None and self.kernel.journal:
            await self._restore(drop_last=False, why="the kernel was lost between cells")
            return
        self.kernel = Kernel(self.launcher, seed=self.seed or 0, cell_timeout_s=self.cell_timeout_s)
        self.kernel.resolve = self._resolve_record
        r = await self.kernel.execute(self._bootstrap_code(), self._host, timeout_s=60, bootstrap=True)
        if not r.ok:  # no model code has run yet: the sandbox or the harness failed (masked)
            raise KernelStartError(f"RLM bootstrap failed: {r.error}", "crucible:backend_error")

    def _bootstrap_code(self) -> str:
        """BOOTSTRAP + the ledger's ``HISTORY_API`` + ``AGENT_API`` (when this agent may spawn) + ``CONTEXT``: one host-written
        cell, journaled with the bootstrap flag (re-run before a snapshot is loaded, §3.4)."""
        code = BOOTSTRAP.replace("__TASK__", repr(self.task_text)).replace("__TOOLS__", repr(self._tools_view())) + HISTORY_API
        if self.registry is not None and self.depth < self.tree.supervisor.limits.max_depth:
            code += AGENT_API
        if self.context_file is not None:
            code += CONTEXT_API.replace("__INFO__", repr(self.context_file.info()))
        return code

    async def _start_fresh(self) -> None:
        """A new agent: kernel and bootstrap, the system prompt and the task, the ledger's first window."""
        self._open_ledger()
        self.system = self._final_system()
        await self._ensure_kernel()
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": self._task_message()}]
        self._log("system", text=self.system)
        self._log("user", text=self.messages[1]["content"], role="task")
        if self.depth > 0:
            self._next_edge = "subagent_call"

    # ---- host requests from cells ---------------------------------------------------------------------------------------
    def _take_inbox(self) -> list[dict[str, Any]]:
        """The root's undelivered INBOX messages (children never take them: messages belong to the root agent)."""
        return []

    async def _host(self, data: dict[str, Any]) -> Any:
        t = data.get("type")
        if not isinstance(t, str):
            raise ValueError("a host request needs a type")  # noqa: TRY004 -- a bad request is the cell's ValueError
        if t in ("world.call", "world.send"):
            return await self._host_world(t, data)
        if t.startswith("history."):
            return await self._host_history(data)
        if t.startswith("context."):
            return await self._host_context(data)
        if t.startswith("agent."):
            return await self._host_agent(data)
        if t in ("llm.query", "llm.batch"):
            return await self._host_sub(t, data)
        if t == "rlm.note":
            text = data.get("text")
            if not isinstance(text, str):
                raise ValueError("rlm.note needs a string")
            self.notes.append(text[:512])  # bounded here too: a direct host("rlm.note", ...) bypassed the wrapper's cap (H3 V13)
            return "noted"
        if t == "rlm.compact":
            if self.spec.mode == "none":
                raise ValueError("compaction is off in this harness (context_mode='none')")
            if self.counters["compactions"] and self.counters["turns"] - self._turn_at_compaction < 2:
                # (M16 hunt H12) rlm.compact() every turn bought a summary call and a snapshot per turn: refused under two turns
                self.counters["compact_requests_refused"] += 1
                return "compaction refused: too soon (fewer than 2 turns since the last compaction)"
            ins = data.get("instructions")
            self._compact_request = ins[:2000] if isinstance(ins, str) else ""
            return "a checkpoint will be written before your next turn"
        if t == "rlm.budget":
            return self._budget_view()
        raise ValueError(f"unknown host request {t!r}")

    def _budget_view(self) -> dict[str, Any]:
        used = self.meter.tokens(self.messages)
        out: dict[str, Any] = {k: {"left": max(0, v[0]), "limit": v[1]} for k, v in self._budgets().items()}
        out["context"] = {"used": used, "threshold": self.threshold, "window": self.spec.window_tokens, "source": self.meter.source}
        return out

    def _need_cell(self, what: str) -> None:
        if self._current_call is None:
            raise PermissionError(f"{what} are only allowed from your own python calls")

    async def _host_world(self, t: str, data: dict[str, Any]) -> Any:
        """A World action from a cell: one World step through the root's ``act()`` bridge, attributed to this agent's
        model call. A child without the ``shared`` grant is refused (least privilege, §3.3)."""
        root = self.tree.root
        if self.depth > 0 and self.world_grant != "shared":
            raise PermissionError("this child agent has no World access (spawned with world='none')")
        if self._current_call is None and getattr(root, "_replaying_pending", None) is None:
            # outside the model's own cells (the bootstrap, _deliver: functions model code can redefine) a World action would
            # be stepped but attributed to no model call, shifting per-call credit (H3 V13)
            raise PermissionError("World actions are only allowed from your own python calls")
        if t == "world.call":
            tool, args = data.get("tool"), data.get("args") if data.get("args") is not None else {}
            if not isinstance(tool, str) or not tool or len(tool) > 200:
                raise ValueError("world.call needs a tool name string (\"app.tool\")")
            if not isinstance(args, dict):
                raise ValueError("world.call arguments must be keyword arguments (a dict)")
            res = await root._world_step(self, "tool", tool=tool, args=args)
            inbox = self._take_inbox()
            i = self._log("world_call", tool=tool, args=args, ok=res["ok"], content=res["content"], error=res["error"], inbox=inbox,
                          call_id=(self._current_call or {}).get("call_id"))
            self.stall.on_world(tool, args, _digest(json.dumps([res["ok"], res["content"], res["error"]], default=str)))
            if inbox:
                self.stall.note_progress("inbox")
            self.counters["world_calls"] += 1
            self._cell_world += 1
            return JournalReply({**res, "inbox": inbox}, {"world": i})
        to, text = data.get("to"), data.get("text")
        if not isinstance(to, str) or not isinstance(text, str):
            raise ValueError("world.send needs to (str) and text (str)")  # noqa: TRY004 -- a bad request is the cell's ValueError
        await root._world_step(self, "message", to=to, text=text)
        inbox = self._take_inbox()
        i = self._log("world_send", to=to, text=text, inbox=inbox, call_id=(self._current_call or {}).get("call_id"))
        self.stall.on_world(to, {"text": text}, _digest(text), kind="send")
        if inbox:
            self.stall.note_progress("inbox")
        self.counters["world_sends"] += 1
        self._cell_world += 1
        return JournalReply({"ok": True, "inbox": inbox}, {"world": i})

    async def _host_history(self, data: dict[str, Any]) -> Any:
        """``history.*`` (the ledger, A): served from the view at this moment and journaled as a < 1 KiB marker that a
        replay rebuilds exactly (B5)."""
        try:
            reply, marker = await self.ledger.ahandle_recorded(data)
        except LedgerError as e:
            raise HostAbort(f"the episode ledger failed: {e}", "crucible:backend_error") from e
        self.counters["history_reads"] += 1
        if self.counters["compactions"]:
            self.counters["history_reads_after_compaction"] += 1
        return JournalReply(reply, marker)

    async def _host_context(self, data: dict[str, Any]) -> Any:
        if self.context_file is None:
            raise ValueError("no CONTEXT is mounted for this task")
        self.counters["context_reads"] += 1
        if data.get("type") != "context.search":
            return JournalReply(self._serve_context(data), {"context": data})
        # (M16 hunt H4) a search runs in a thread: its 2 s bound (a regex worker killed at the deadline) blocked the host's event
        # loop, so every cell could stall every other agent and episode of the process for 2 s per call, without bound. Its
        # record keeps where the live call stopped (the time bound is not replayable): a replay re-scans to the same hit count
        # and reports the same flags, instead of re-running the bounded search, which answered a timed-out call with hits
        reply = await asyncio.to_thread(self._serve_context, data)
        stop = {"n": len(reply.get("items") or []), "truncated": bool(reply.get("truncated")), "timed_out": bool(reply.get("timed_out"))}
        return JournalReply(reply, {"context": data, "stop": stop})

    def _serve_context(self, data: dict[str, Any]) -> Any:
        assert self.context_file is not None
        t = data.get("type")
        if t == "context.read":
            return self.context_file.read(data.get("offset"), data.get("n"))
        if t == "context.search":
            return self.context_file.search(data.get("pattern"), data.get("regex", False), data.get("limit", 20), data.get("context_chars", 200))
        raise ValueError(f"unknown context request {t!r}")

    def _replay_context(self, record: dict[str, Any]) -> Any:
        """(M16 hunt H4) A journaled CONTEXT reply: a search re-scans to the live call's hit count, without its time bound."""
        stop, data = record.get("stop"), record["context"]
        if not isinstance(stop, dict) or data.get("type") != "context.search":
            return self._serve_context(data)
        flags = {"truncated": bool(stop.get("truncated")), "timed_out": bool(stop.get("timed_out"))}
        n = int(stop.get("n") or 0)
        if n == 0:
            return {"items": [], **flags}
        assert self.context_file is not None
        r = self.context_file.search(data.get("pattern"), data.get("regex", False), n, data.get("context_chars", 200), bound_s=60.0)
        return {"items": r["items"][:n], **flags}

    async def _host_agent(self, data: dict[str, Any]) -> Any:
        """``agent.*`` (the tree, C): only from a cell (PermissionError otherwise); a refused spawn's ``BudgetExhausted``
        is the cell's error, not a stop."""
        self._need_cell("child agent requests")
        if self.registry is None or self.depth >= self.tree.supervisor.limits.max_depth:
            raise ValueError("child agents are off at this depth (tree_max_depth)")
        if data.get("type") == "agent.spawn" and data.get("world") == "shared" and self.depth > 0 and self.world_grant != "shared":
            # (M16 hunt H2) least privilege is inherited: a child spawned with world="none" granted world="shared" to its own
            # child, which then stepped the World (the grant was checked per agent, never against the parent's)
            raise PermissionError("a child agent without World access cannot give World access to its own children")
        self.counters["agent_requests"] += 1
        reply = await self.registry.handle(data)
        op = str(data.get("type"))[6:]
        if op == "spawn" and isinstance(reply, dict):
            self._log("child_spawn", name=reply.get("name"), text=str(data.get("brief"))[:4000], world=reply.get("world"),
                      call_id=(self._current_call or {}).get("call_id"))
        elif op == "result" and isinstance(reply, dict) and reply.get("status") not in ("running", None):
            self._log("child_return", name=reply.get("name"), status=reply.get("status"), text=str(reply.get("answer") or "")[:65_536],
                      error=reply.get("error"))
        if self.tree.fault is not None:  # a child's provider or sandbox fault ends the episode as infra
            raise HostAbort(self.tree.fault[1], self.tree.fault[0])
        return reply

    async def _host_sub(self, t: str, data: dict[str, Any]) -> Any:
        self._need_cell("sub-LLM calls")
        mt = data.get("max_tokens")
        if mt is not None and (not isinstance(mt, int) or isinstance(mt, bool) or mt < 1):
            raise ValueError("max_tokens must be a positive integer")
        if t == "llm.query":
            p, c = data.get("prompt"), data.get("context")
            if not isinstance(p, str) or (c is not None and not isinstance(c, str)):
                raise ValueError("rlm.query needs a prompt string (and an optional context string)")
            self._reserve_subs(1)
            return await self._sub(p, c, mt)
        items = data.get("items")
        if not isinstance(items, list) or not all(isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and (x[1] is None or isinstance(x[1], str))
                                                  for x in items):
            raise ValueError("rlm.batch needs a list of prompts (or (prompt, context) pairs)")
        self._reserve_subs(len(items))
        tasks = [asyncio.ensure_future(self._sub(p, c, mt)) for p, c in items]
        try:
            return list(await asyncio.gather(*tasks))
        finally:
            for x in tasks:
                x.cancel()

    def _reserve_subs(self, n: int) -> None:
        """The sub-call budget, checked before any call starts so a refused batch makes no call: per cell
        (``max_sub_calls_per_cell``) and tree-wide (``TreeSupervisor.charge_sub``; B6: the m14 counters were per episode
        and fixed at 256)."""
        if n < 1:
            return
        if self._cell_sub_calls + n > self.cfg.max_sub_calls_per_cell:
            self.counters["sub_refused"] += n
            raise ValueError(f"one python call may make at most {self.cfg.max_sub_calls_per_cell} sub-LLM calls ({self._cell_sub_calls} made, {n} asked)")
        try:
            self.tree.supervisor.charge_sub(n)
        except BudgetExhausted as e:
            self.counters["sub_refused"] += n
            raise ValueError(str(e)) from None
        self.counters["sub_calls"] += n
        self._cell_sub_calls += n

    async def _sub(self, prompt: str, context: str | None, max_tokens: int | None) -> str:
        content = prompt if context is None else f"{prompt}\n\n<context>\n{context}\n</context>"
        parent = self._current_call["call_id"] if self._current_call else None
        if self._sub_sem is None:
            self._sub_sem = asyncio.Semaphore(max(1, int(self.cfg.sub_concurrency)))
        mt = min(int(max_tokens or self.cfg.sub_max_tokens), int(self.cfg.sub_max_tokens))  # the model may lower the cap, never raise it
        base = {"kind": "sub", "parent": parent, "agent": self.agent, "depth": self.depth, "edge": "sub", "segment": self.segment, "step": self.tree.step,
                "window": self._window(), "trainable": self.sub_trainable, "actions": []}
        async with self._sub_sem:
            try:
                r = await self.sub_llm.complete([{"role": "user", "content": content}], tools=None, max_tokens=mt, temperature=self.temperature,
                                                sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)
            except (LLMError, BudgetExceeded) as e:
                if is_overflow(e):
                    raise ValueError(f"the sub-LLM prompt does not fit its context window: {e}") from None  # the policy's own prompt: the cell's error
                # a provider fault is infra: the episode is masked as on the root path, with a record of the failed call (H3 V5, G1 D2)
                kind = getattr(e, "failure_kind", None) or "provider_unavailable"
                self.tree.calls.append({"call_id": self.tree.calls.new_id(), **base, "model": None, "failed": True,
                                        "error": f"{type(e).__name__}: {str(e)[:500]}", "tokens": None})
                self.failure_kind = self.failure_kind or kind
                self.error = self.error or f"{kind}: a sub-LLM call failed: {e}"
                if self.depth > 0 and self.tree.fault is None:
                    self.tree.fault = (kind, f"child agent {self.agent}: {self.error}")
                raise HostAbort(f"sub-LLM call failed: {e}", kind) from e
        u = r.usage or {}
        n = int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
        self.counters["sub_tokens"] += n
        with contextlib.suppress(BudgetExhausted):  # recorded; the next charge_sub is refused
            self.tree.supervisor.charge_sub_tokens(n)
        cid = self.tree.calls.new_id()
        i = self._log("sub_call", prompt=content, text=r.text or "", call_id=cid, parent=parent)
        self.tree.calls.append({"call_id": cid, **base, "ledger_index": i, "model": r.model, **r.finish_fields(), "usage": r.usage,
                                "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3), "cached": r.cached, "tokens": r.tokens})
        return r.text or ""

    def _resolve_record(self, request: dict[str, Any], record: Any) -> Any:
        """``Kernel.resolve``: rebuild a journaled reply from its record (B5): a ``history`` marker from the ledger
        (``Ledger.replay_reply``), a ``context`` request from the context file, a ``world`` index from the ledger event."""
        if isinstance(record, dict) and isinstance(record.get("history"), dict):
            return self.ledger.replay_reply(record)
        if isinstance(record, dict) and "context" in record:
            return self._replay_context(record)
        if isinstance(record, dict) and isinstance(record.get("world"), int):
            return self._world_reply(record["world"])
        raise LookupError(f"no resolver for journal record {str(record)[:80]}")

    def _read_field(self, i: int, name: str) -> Any:
        ev = self.ledger.get(i)
        if name not in (ev.get("more") or {}):
            return ev.get(name)
        out, cursor = [], 0
        while cursor is not None:
            r = self.ledger.read(i, name, cursor, 131_072)
            out.append(r["text"])
            cursor = r["next"]
        return "".join(out)

    def _world_reply(self, i: int) -> dict[str, Any]:
        ev = self.ledger.get(i)
        inbox = ev.get("inbox") or []
        if isinstance(inbox, str):  # (M16 hunt H1) a large inbox was logged as its JSON text
            inbox = json.loads(self._read_field(i, "inbox"))
        if ev.get("kind") == "world_send":
            return {"ok": True, "inbox": inbox}
        if ev.get("kind") != "world_call":
            raise LookupError(f"ledger event {i} is not a World call")
        return {"ok": ev.get("ok"), "content": self._read_field(i, "content"), "error": ev.get("error"), "inbox": inbox}

    # ---- the cell ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _format(r: CellResult) -> str:
        parts = []
        if r.stdout:
            parts.append(r.stdout.rstrip("\n"))
        if r.stderr:
            parts.append("[stderr]\n" + r.stderr.rstrip("\n"))
        if r.result is not None:
            parts.append(r.result)
        if r.error:
            parts.append(r.error.rstrip("\n"))
        return "\n".join(parts) or "(no output)"

    @staticmethod
    def _cap(text: str, n: int, i: int | None) -> str:
        """Head (3/4) + tail (1/4) of ``text`` in ``n`` characters, the cut named with the ledger copy and where to resume."""
        if len(text) <= n:
            return text

        def marker(cut: int, h: int) -> str:
            where = f"; full output: await OUT({i}, cursor={h})" if i is not None else ""
            return f"\n[... {cut} characters truncated here{where} ...]\n"

        keep = max(0, n - len(marker(len(text), len(text))))
        h = keep * 3 // 4
        return text[:h] + marker(len(text) - keep, h) + (text[-(keep - h):] if keep - h > 0 else "")

    def _announce(self, msgs: list[dict[str, Any]], room: int) -> str:
        """New INBOX messages, bounded (G1 D15: 50 messages put ~11k characters past a 200-character output cap)."""
        head = f"[{len(msgs)} new message(s) appended to INBOX]"
        lines = "; ".join(f"from {m['from']}: {str(m['text'])[:200]}" for m in msgs)
        text = f"{head} {lines}"
        if len(text) > room:
            tail = " ... (read INBOX for the rest)"
            text = text[: max(len(head), room - len(tail))] + tail
        return text

    async def _execute(self, code: str, entry: JournalEntry | None, *, timeout_s: float | None = None) -> tuple[CellResult, bool]:
        """Run a cell; a kernel that died mid-cell (its channel reset while the host answered) is a killed cell
        (E's fault campaign: ``ConnectionResetError`` escaped as a masked ``agent_run_error``). Returns (result, died)."""
        assert self.kernel is not None
        try:
            return await self.kernel.execute(code, self._host, entry=entry, timeout_s=timeout_s), False
        except OSError as e:
            with contextlib.suppress(Exception):
                await self.kernel.close()
            return CellResult(killed=True, error=f"the kernel process died while this cell ran ({type(e).__name__}: {e})"), True

    async def _run_cell(self, call: dict[str, Any], tc_id: str, code: str, entry: JournalEntry | None = None, *, rerun: str = "") -> bool:
        """Execute one cell (or continue a resumed one); True when the agent's episode is over."""
        await self._ensure_kernel()
        self._current_call = call
        self.counters["cells"] += 1
        self._cell_msgs, self._cell_sub_calls, self._cell_world = [], 0, 0
        t0 = time.monotonic()
        try:
            r, died = await self._execute(code, entry)
        finally:
            self._current_call = None
        self._cell_elapsed = time.monotonic() - t0
        if r.lost_between_cells and not rerun:
            # nothing of this cell ran: restore the kernel and run it again (B10: the old notice said it timed out)
            self.counters["lost_between_cells"] += 1
            self.counters["cells"] -= 1
            how = await self._restore(drop_last=True, why="the kernel was lost before this cell ran")
            return await self._run_cell(call, tc_id, code, rerun=f"[the kernel was lost before this cell ran (its code did not run); {how}; "
                                                                 "the cell then ran normally]")
        notes = [rerun] if rerun else []
        if r.protocol_errors:  # the cell forged or garbled kernel protocol frames: dropped; its own (scored) problem (H3 V2)
            self.counters["protocol_errors"] += r.protocol_errors
            notes.append(f"[kernel: {r.protocol_errors} invalid protocol frame(s) from this cell were ignored]")
        if r.stray_requests:  # (M16 hunt H11) once per cell: a task of an earlier cell asked the host during this one
            self.counters["stray_requests"] += r.stray_requests
            note = (f"[kernel: {r.stray_requests} host request(s) from a task an earlier python call started were refused: they are not "
                    "part of this call; await your tasks inside the cell that starts them]")
            notes.append(note)
            self._log("notice", text=note, stray_requests=r.stray_requests)
        if r.timed_out or r.killed:
            notes.append(await self._recovery_notice(r, died))
        out = self._format(r)
        ann = self._announce(self._cell_msgs, max(80, self.max_output_chars // 4)) if self._cell_msgs else ""
        tail = "\n".join(notes)
        used = len(tail) + (len(ann) + 1 if ann else 0)
        room = max(50, self.max_output_chars - used - 1) if used else self.max_output_chars  # announcements share the cap (G1 D15)
        # the full output goes to the ledger; the context gets the capped view naming its index (A's tool event)
        i = self._log("tool", text=out, shown_chars=min(len(out), room), call_id=call.get("call_id"), cell=self.counters["cells"], ok=r.ok,
                      elapsed_s=round(self._cell_elapsed, 3), world_calls=self._cell_world)
        view = self._cap(out, room, i)
        self._last_output = out[-4096:]
        msg = {"role": "tool", "tool_call_id": tc_id, "content": view + ("\n" + tail if tail else "")}
        self.messages.append(msg)
        if ann:
            # messages from people are a user row of their own, after the result: the summarizer's view keeps user rows whole
            # and cuts tool rows to head + tail, which cut an announcement out of a long output (M16: a faithful summary lost
            # the policy amendment)
            msg = {"role": "user", "content": ann}
            self.messages.append(msg)
        self._add_footer(msg)
        self._wrap_up()
        if self.tree.fault is not None and self.depth == 0:
            self.failure_kind, self.error = self.tree.fault
            self._end("policy_error", self.error)
            return True
        if r.finished is not None:
            fin = str(r.finished)
            call.setdefault("actions", []).append("done")
            if fin.startswith(GIVE_UP_MARK):
                try:
                    g = json.loads(fin[len(GIVE_UP_MARK):])
                except ValueError:
                    g = {"reason": fin[len(GIVE_UP_MARK):]}
                self.answer = f"gave up: {g.get('reason')}"
                self._log("notice", text=json.dumps(g)[:8000], stop="agent_gave_up")
                self._end("agent_gave_up", f"gave up: {g.get('reason')}"[:2000])
            else:
                self.answer = fin
                self._end("agent_done", fin[:2000])
            return True
        self.stall.on_cell(code, r.ok, output_digest=_digest(out))
        if self._check_stall():
            return True
        if self.cfg.snapshot_every and self.counters["cells"] % self.cfg.snapshot_every == 0:
            await self._snapshot("cadence")
        return False

    def _add_footer(self, msg: dict[str, Any]) -> None:
        """The ``<context_usage>`` footer from ``footer_from`` × threshold (H15; B9: the model never saw its deadline)."""
        used = self.meter.tokens(self.messages)
        if used < self.spec.footer_from * self.threshold:
            return
        msg["content"] += "\n" + footer(used=used, threshold=self.threshold, window=self.spec.window_tokens, step=self.tree.step,
                                        turns=self.counters["turns"], budgets=self._budgets(), elapsed_s=self._cell_elapsed, source=self.meter.source,
                                        compact_hint=self.spec.mode == "action")

    def _wrap_up(self) -> None:
        """Once per budget, at 0.9 used: finish now, do not compact (H3)."""
        for kind, (left, limit) in self._budgets().items():
            if limit > 0 and kind not in self._wrapped and left <= 0.1 * limit:
                self._wrapped.add(kind)
                text = wrap_up_notice(kind, max(0, left))
                self.messages.append({"role": "user", "content": text})
                self._log("notice", text=text, budget=kind)

    def _check_stall(self) -> bool:
        v = self.stall.check()
        if v is None:
            return False
        self.counters["stalls"] += 1
        sv = self.health["stall_verdicts"]
        sv[f"{v.signature}:{v.action}"] = sv.get(f"{v.signature}:{v.action}", 0) + 1
        text = v.message()
        self._log("notice", text=text, stall=v.signature, action=v.action)
        if v.action == "stop":
            self._end("stalled", text)
            return True
        self.messages.append({"role": "user", "content": text})  # a notice, or the forced reflection the next turn answers
        return False

    async def _recovery_notice(self, r: CellResult, died: bool) -> str:
        """Restore the kernel to before this cell and say exactly what happened (G's notes 4, 5; B10)."""
        if died or (r.killed and not r.timed_out):
            self.counters["killed_mid_cell"] += 1
            what = "the kernel process died while this cell ran (it was killed or ran out of memory)"
        else:
            what = f"the cell did not finish in its {self.cell_timeout_s:g} s deadline and was stopped"
        how = await self._restore(drop_last=True, why=what)
        acted = (f" The {self._cell_world} World action(s) it took still happened: check their results (`await history.since({self.tree.step})`, OBS "
                 "lacks them) before repeating anything; never blindly repeat the interrupted cell.") if self._cell_world else \
            " World actions it took still happened."
        return f"[{what}; {how}.{acted}]"

    async def _restore(self, *, drop_last: bool, why: str) -> str:
        """Rebuild the kernel from its last snapshot + the journal suffix (full, then lenient replay as fallbacks:
        ``Kernel.replay``), without the failed cell. Returns the notice text."""
        from crucible.world.state.base import SnapshotMismatch

        k = self.kernel
        assert k is not None
        entries = [e for e in (k.journal[:-1] if drop_last and k.journal else k.journal)]
        entries = [e for e in entries if e.done]
        snap = k.last_snapshot
        self.counters["restores"] += 1
        t0 = time.monotonic()
        try:
            try:
                await k.replay(entries, snapshot=snap, offset=k.journal_offset if snap is not None else 0)
            except (SnapshotMismatch, KernelDivergence):  # no snapshot and a strict replay diverged: the lenient replay
                await k.replay(entries, strict=False)
        except (KernelStartError, HostAbort):
            raise
        except (KernelError, SnapshotMismatch):
            # (M16 hunt H3) every restore path failed (a namespace object the policy planted cannot be unpickled, e.g. a
            # __reduce__ that raises, and the journal starts at the snapshot so no full replay exists): the KernelError
            # escaped the driver and masked the policy's own episode as agent_run_error. The kernel restarts from its
            # bootstrap and the model is told every variable is gone (scored: its own state broke the restore)
            return await self._restart_kernel(why)
        lr = dict(k.last_restore or {})
        mode = lr.get("mode") or "full"
        self.health["restores_by_mode"][mode] = self.health["restores_by_mode"].get(mode, 0) + 1
        lost = list(lr.get("lost") or [])
        self._lost_names = lost
        self._log("restore", text=why, mode=mode, cells=lr.get("cells"), seconds=round(time.monotonic() - t0, 3), lost=lost,
                  divergences=lr.get("divergences"), fallback=lr.get("fallback"))
        with contextlib.suppress(ValueError):
            self.ledger.open_window("restore", mode=mode)
        parts = ["the kernel state was restored to before this cell" if drop_last else "the kernel state was restored"]
        if mode == "lenient":
            self.counters["inexact_restores"] += 1
            parts.append(f"but {lr.get('divergences') or 0} replayed cell(s) produced different values (they read the clock, random ids or other "
                         "changing state): check the variables you rely on")
        if lost:
            parts.append(f"variables lost: {', '.join(lost[:20])}{' ...' if len(lost) > 20 else ''} (they could not be saved): redefine them before use")
        return "; ".join(parts)

    async def _restart_kernel(self, why: str) -> str:
        """(M16 hunt H3) A fresh kernel with only the bootstrap: the last resort when no restore path works."""
        k = self.kernel
        assert k is not None
        await k.close()
        k.journal, k.journal_offset, k.last_snapshot, k._boot = [], 0, None, []
        self._lost_names = ["(every variable)"]
        r = await k.execute(self._bootstrap_code(), self._host, timeout_s=60, bootstrap=True)
        if not r.ok:
            raise KernelStartError(f"RLM bootstrap failed: {r.error}", "crucible:backend_error")
        self.counters["inexact_restores"] += 1
        self.health["restores_by_mode"]["restart"] = self.health["restores_by_mode"].get("restart", 0) + 1
        self._log("restore", text=why, mode="restart", cells=0, lost=["(every variable)"])
        with contextlib.suppress(ValueError):
            self.ledger.open_window("restore", mode="restart")
        return ("the kernel state could not be restored (an object in it cannot be reloaded), so the kernel was restarted: every "
                "variable, function and import you defined is lost (TASK, TOOLS, world, rlm and history are back; OBS and INBOX are empty): "
                "redefine what you need")

    async def _snapshot(self, why: str) -> None:
        """A kernel snapshot between cells (every ``snapshot_every`` cells and at each compaction, §3.4): restores and
        resumes then replay only the cells after it (B4, B5). No dill in the kernel: journal replay stays the restore."""
        k = self.kernel
        if not self._snapshots_on or k is None or not k.alive() or (k.journal and not k.journal[-1].done):
            return
        n = self.counters["snapshots"] + 1
        dest = self.ledger.dir / ".snapshots" / f"snap-{n:05d}.bin"
        try:
            snap = await k.snapshot(dest)
        except KernelError as e:
            self.counters["snapshot_failures"] += 1
            self._log("notice", text=f"[snapshot failed: {e}]"[:2000])
            if "dill" in str(e):
                self._snapshots_on = False
            return
        self.counters["snapshots"] += 1
        self._snap_paths.append(dest)
        self._log("snapshot", reason=why, bytes=snap.bytes, cells=snap.journal_len, seconds=snap.seconds, lost=snap.lost)
        for p in self._snap_paths[:-1]:  # keep the newest and the one the last checkpoint names
            if str(p) != self._ckpt_snapshot:
                with contextlib.suppress(OSError):
                    p.unlink()
        self._snap_paths = [p for p in self._snap_paths if p.exists()]

    # ---- the boundary -----------------------------------------------------------------------------------------------------
    async def _boundary(self) -> None:
        """Turn boundary (§3.6 step 1): child events, messages from the parent, INBOX; news resets the stall streak."""
        if self.registry is not None:
            evs = self.registry.events()
            if evs:
                self.stall.note_progress("child events")
                for ev in evs:
                    self._log("child_event", name=str(ev.get("name")), type=str(ev.get("type")), text=json.dumps(ev, default=str)[:65_536])
                room = max(200, self.max_output_chars // 4)
                lines = []
                for ev in evs:
                    tail = ev.get("answer_tail") or ev.get("error") or ev.get("last") or ev.get("note") or ""
                    lines.append(f"{ev.get('type')} {ev.get('name') or ''} ({ev.get('status', '')}): {str(tail)[-300:]}")
                text = f"[{len(evs)} child event(s); `await rlm.agent.result(name)` for a full answer] " + "; ".join(lines)
                self.messages.append({"role": "user", "content": text[:room]})
        if self._delivered:
            batch, self._delivered = self._delivered, []
            for kind, text in batch:
                row = f"[{'steer' if kind == 'steer' else 'message'} from your parent agent] {text}"
                self.messages.append({"role": "user", "content": row})
                self._log("user", text=row, role=kind)
            self._next_edge = self._next_edge or "agent_message"
            self.stall.note_progress("message")

    # ---- compaction -------------------------------------------------------------------------------------------------------
    def _size(self) -> int:
        return self._size_of(self.messages)

    def _size_of(self, messages: list[dict[str, Any]]) -> int:
        """The pre-send size of a context: the usage-anchored meter plus a margin for the part it estimates (10 % of the
        unanchored suffix + 32), so the call sent is not above the threshold when the estimate runs low (B1)."""
        used = self.meter.tokens(messages)
        n = self._anchor_n if self._anchor_n is not None and self._anchor_n <= len(messages) else 0
        return used + int(0.1 * approx_tokens(messages[n:]) * max(1.0, self.meter.ratio)) + 32

    def _summary_overhead(self) -> int:
        """What the summary request adds to the context: its prompt, with the previous summary in update mode (B1)."""
        text = summary_request(self._last_summary, instructions=self._compact_request, max_tokens=self.spec.summary_tokens())
        return int((len(text) / 4 + 8) * max(1.0, self.meter.ratio))

    def _over_threshold(self) -> bool:
        """Compact when the context plus the summary request's own prompt reaches the threshold: the summary call then
        carries the whole context (no hollowing in the common case) and is still not above the threshold (B1)."""
        return self.spec.mode != "none" and self._size() + self._summary_overhead() >= self.threshold

    def _summary_spec(self) -> ContextSpec:
        """The spec the Compactor runs under: its request (the summary view + the prompt) is bounded at 95 % of the
        compaction threshold (calibrated chars/4 units run a few percent off on a template's markup), so no call of the
        loop, the summary call included, is sent above the threshold (B1: the m14 summary call was each segment's largest
        prompt). The view is hollowed only when one turn pushed the context past the trigger by more than the margin.
        ``seq_len`` is the knob ``summary_input_tokens()`` reads."""
        s = self.spec.summary_tokens()
        cap = int(0.95 * self.threshold) + s
        return dataclasses.replace(self.spec, seq_len=min(self.spec.seq_len or cap, cap))

    async def _compact(self, reason: str) -> bool:
        """One compaction (R3, R4; see the module docstring). False when the agent stopped."""
        spec = self.spec
        if spec.max_compactions is not None and self.counters["compactions"] >= spec.max_compactions:
            self._end("too_many_compactions", f"{self.counters['compactions']} compactions (max_compactions)")
            return False
        before = self.meter.tokens(self.messages)
        src = self.meter.source
        before_chars = sum(len(json.dumps(m, default=str)) for m in self.messages)
        instructions, self._compact_request = self._compact_request, None
        attempts: list[dict[str, Any]] = []

        def on_call(resp: Any, attempt: int, accepted: bool | None, input_tokens: int) -> None:
            rec: dict[str, Any] = {"call_id": self.tree.calls.new_id(), "kind": "compaction", "agent": self.agent, "depth": self.depth,
                                   "edge": "compaction_attempt", "accepted": bool(accepted), "attempt": attempt,
                                   "trainable": accepted is True, "segment": self.segment, "window": self._window(), "step": self.tree.step,
                                   "context_tokens": input_tokens, "reason": reason, "actions": []}
            if isinstance(resp, BaseException):
                rec.update(model=None, failed=True, error=f"{type(resp).__name__}: {str(resp)[:500]}", tokens=None)
                text = rec["error"]
            else:
                u = resp.usage or {}
                rec.update(model=resp.model, **resp.finish_fields(), usage=resp.usage, cost_usd=resp.cost_usd, latency_s=round(resp.latency_s, 3),
                           cached=resp.cached, tokens=resp.tokens)
                text = resp.text or ""
                self._largest("compaction", int(u.get("prompt_tokens") or 0))
                with contextlib.suppress(BudgetExhausted):  # recorded; the next charge_turn is refused
                    self.tree.supervisor.charge_tokens(self.depth, int(u.get("completion_tokens") or 0), int(u.get("prompt_tokens") or 0),
                                                       summary=True, estimate=not u)
            rec["ledger_index"] = self._log(ATTEMPT_KIND, attempt=attempt, accepted=bool(accepted), input_tokens=input_tokens, text=text,
                                            call_id=rec["call_id"])
            self.tree.calls.append(rec)
            attempts.append(rec)
            self.counters["compaction_attempts"] += 1

        try:
            out = await Compactor(meter=self.meter).run(self.llm, self.messages, self._summary_spec(), previous_summary=self._last_summary, on_call=on_call,
                                                        sample_key=self.sample_key, temperature=self.temperature, instructions=instructions)
        except (LLMError, BudgetExceeded) as e:  # a provider fault escaped the Compactor (B2): masked with its kind
            self._provider_fault(e, "a compaction call failed")
            return False
        seg = {"index": self.segment + 1, "agent": self.agent, "strategy": "summary", "boundary_reason": "compaction", "attempts": out.attempts,
               "failed": not out.ok, "context_before": before, "tokens_source": src, "reason": reason, "step": self.tree.step,
               "summary_call": next((a["call_id"] for a in attempts if a["accepted"]), None)}
        if not out.ok:
            self.counters["compaction_failures"] += 1
            self.tree.segments.append({**seg, "boundary_reason": "compaction_failed", "context_after": before})
            with contextlib.suppress(ValueError):
                self.ledger.open_window("compaction_failed", attempts=out.attempts, detail=out.detail)
            salvage = next((m.get("content") for m in reversed(self.messages) if m.get("role") == "assistant" and m.get("content")), "") or ""
            self.answer = salvage or None
            self._end("compaction_failed", f"every summary attempt failed ({out.detail}); {salvage[:1500]}")
            return False
        n = self.counters["compactions"] + 1
        tail_budget = min(spec.tail_tokens, self.threshold // 4)
        _, tail = split_tail(self.messages[2:], int(tail_budget / max(1.0, self.meter.ratio)))
        entries = await self._describe()
        state = python_state_row(entries, lost=self._lost_names, max_chars=max(400, min(2_000, self.threshold * 4 // 10)))
        self._lost_names = []
        w = self.ledger.open_window("compaction", n=n, before=before, source=src)
        self._log("summary", text=out.summary, compaction=n, call_id=seg["summary_call"])
        system, task = self.messages[0], self.messages[1]
        self.meter.reset()
        self._last_summary = out.summary
        self._anchor_n = None

        def build(tail: list[dict[str, Any]]) -> list[dict[str, Any]]:
            size = self.meter.tokens(rebuild(system, task, out.summary, state, "", tail))
            return rebuild(system, task, out.summary, state, boundary_marker(n, before, size, w), tail)

        msgs = build(tail)
        # room for work: the rebuilt context plus the next summary request must sit well under the threshold
        if tail and self._size_of(msgs) + self._summary_overhead() >= 0.75 * self.threshold:
            tail = []  # the tail alone keeps it full: drop it (the ledger has it)
            msgs = build(tail)
        after = self.meter.tokens(msgs)
        if self._size_of(msgs) + self._summary_overhead() >= self.threshold:
            self.counters["compaction_failures"] += 1
            self.tree.segments.append({**seg, "boundary_reason": "compaction_failed", "failed": True, "context_after": after})
            self._end("compaction_failed", f"the rebuilt context ({after} tokens: system, task, summary and state) and the next summary request "
                                           f"leave no room under the compaction threshold ({self.threshold}); raise window_tokens")
            return False
        self.messages = msgs
        # (M16 hunt H12) a snapshot (up to 256 MiB, seconds of kernel time) at a compaction only when it freed context (at least a
        # tenth of the threshold, and the rebuilt context is under 90 % of it) or the cadence is due; a compaction a cell asked
        # for right after the last one takes none (its restore replays the few cells since the last snapshot)
        cells_since = self.counters["cells"] - (self.kernel.last_snapshot.journal_len if self.kernel is not None and self.kernel.last_snapshot else 0)
        freed = before - after >= 0.1 * self.threshold and after <= 0.9 * self.threshold
        if freed or (self.cfg.snapshot_every and cells_since >= self.cfg.snapshot_every):
            await self._snapshot("compaction")
        else:
            self.counters["snapshots_skipped"] += 1
        self.health["turns_between_compactions"].append(self.counters["turns"] - self._turn_at_compaction)
        self.health["chars_dropped"].append(before_chars - sum(len(json.dumps(m, default=str)) for m in msgs))
        self.health["summary_chars"].append(len(out.summary))
        self._turn_at_compaction = self.counters["turns"]
        self.counters["compactions"] = n
        self._next_edge = "compaction"  # D's contract: the first turn after an accepted summary (whole prompt, new segment)
        self._new_segment({**seg, "context_after": after, "tokens_source": src, "after_source": self.meter.source, "window": w, "kept_messages": len(tail)})
        return True

    def _new_segment(self, seg: dict[str, Any]) -> None:
        """A new token sequence for this agent: its first call stores its whole prompt; continuations are per segment."""
        self.segment = seg["index"]
        self.tree.segments.append(seg)
        self._continuations = 0
        self._last_full = None
        self._chain_calls_in_segment = 0
        self._prev_total = None

    async def _describe(self) -> list[dict[str, Any]]:
        if self.kernel is None or not self.kernel.alive():
            return []
        try:
            entries = await self.kernel.describe()
        except KernelError:
            return []
        unit = {"items": "len", "chars": "chars", "bytes": "bytes"}
        return [{"name": e["name"], "type": e["type"], unit.get(e.get("unit"), "size"): e["size"]} for e in entries]

    # ---- the model call ---------------------------------------------------------------------------------------------------
    def _largest(self, kind: str, prompt: int) -> None:
        lp = self.health["largest_prompt"]
        lp[kind] = max(lp.get(kind, 0), prompt)

    async def _call_model(self) -> Any:
        """§3.6 steps 2-4. Returns the accepted reply, or None (an overflow was compacted, or the agent stopped)."""
        if not self._turn_paid:
            try:
                self.tree.supervisor.charge_turn(self.depth)
            except BudgetExhausted as e:
                self._end(e.kind, str(e))
                return None
            self._turn_paid = True  # an overflow retry is the same turn (no turn charged twice)
        try:
            r = await self.llm.complete(self.messages, tools=[PYTHON_TOOL], max_tokens=self.max_out, temperature=self.temperature,
                                        sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)
        except asyncio.CancelledError:
            raise
        except (LLMError, BudgetExceeded) as e:
            if is_overflow(e):
                return await self._on_overflow(f"{type(e).__name__}: {e}", None)
            self._provider_fault(e, "a model call failed")
            return None
        except Exception as e:
            if is_overflow(e):
                return await self._on_overflow(f"{type(e).__name__}: {e}", None)
            raise  # a lost worker / transport fault: the driver's exception (masked by run_episode, agent_run_error)
        if is_silent_overflow(r.usage, r.finish, self.spec.window_tokens, r.text, r.tool_calls):
            return await self._on_overflow("silent overflow (the provider returned an empty reply for an over-long prompt)", r)
        self._turn_paid = False
        self._overflows_in_row = 0
        return r

    async def _on_overflow(self, what: str, r: Any) -> None:
        """B2: the reply is discarded (never recorded as a trajectory call; its cost is counted), the context compacted,
        the call retried. Compaction off: the scored stop ``context_overflow`` (no failure kind: never masked)."""
        self.counters["overflow_retries"] += 1
        self._overflows_in_row += 1
        if r is not None:
            self.counters["discarded_cost_usd"] += float(r.cost_usd or 0.0)
        self.health["overflows"].append({"turn": self.counters["turns"], "what": what[:200], "context": self.meter.tokens(self.messages)})
        self._log("notice", text=f"[overflow] {what}"[:2000])
        if self.spec.mode == "none":
            self._end("context_overflow", what)
            return
        if self._overflows_in_row > 3:
            self._end("context_overflow", f"the context still overflowed after {self._overflows_in_row - 1} compactions: {what}")
            return
        await self._compact("overflow")

    def _record_turn(self, r: Any, sent: int, sent_source: str) -> dict[str, Any]:
        """Append the assistant row, anchor the meter, charge the tree, record the call (§3.5 fields)."""
        if (r.tokens or {}).get("new_segment") and self._chain_calls_in_segment:
            # the engine re-rendered the history (a thinking template dropped earlier reasoning): a new segment, not a break (H3 V9)
            w = self.ledger.open_window("rerender")
            self._new_segment({"index": self.segment + 1, "agent": self.agent, "strategy": "rerender", "boundary_reason": "rerender", "summary_call": None,
                               "context_before": sent, "context_after": sent, "tokens_source": sent_source, "window": w, "step": self.tree.step})
        u = r.usage or {}
        prompt, completion = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        self.counters["turns"] += 1
        self.counters["completion_tokens"] += completion
        self._largest("turn" if self.depth == 0 else "child_turn", prompt)
        with contextlib.suppress(BudgetExhausted):  # recorded; the next charge_turn is refused (C's contract)
            self.tree.supervisor.charge_tokens(self.depth, completion, prompt, estimate=not u.get("prompt_tokens"))
        # B8: the tokens that entered the context, from engine usage (prompt delta + completion) where the engine reports it
        if prompt:
            entered = completion + (prompt - self._prev_total if self._prev_total is not None and prompt >= self._prev_total else prompt)
            self._prev_total = prompt + completion
            self.tree.trajectory_sources.add("usage")
        else:
            entered = approx_tokens(self.messages[-2:]) + completion
            self.tree.trajectory_sources.add("estimate")
        self.tree.trajectory_tokens += entered
        edge, self._next_edge = self._next_edge or "continuation", None
        call = {"call_id": self.tree.calls.new_id(), "kind": "turn", "agent": self.agent, "depth": self.depth, "edge": edge, "segment": self.segment,
                "window": self._window(), "step": self.tree.step, "context_tokens": sent, "context_source": sent_source, "model": r.model,
                **r.finish_fields(), "usage": r.usage, "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3), "cached": r.cached,
                "actions": [], "trainable": True, "tokens": self._compress(r.tokens)}
        if self.spawn_parent and not self._parent_sent:  # a child's first call names the call whose cell spawned it (D)
            call["parent"], self._parent_sent = self.spawn_parent, True
        self._chain_calls_in_segment += 1
        for sg in reversed(self.tree.segments):  # the segment's first call
            if sg.get("agent") == self.agent and sg["index"] == self.segment:
                sg.setdefault("start_call", call["call_id"])
                break
        return call

    async def _turn(self) -> bool:
        """One turn: the model call and what its reply asks for. True when the agent's episode is over."""
        sent = self._size()
        src = self.meter.source
        r = await self._call_model()
        if r is None:
            return self.stop is not None
        call = self._record_turn(r, sent, src)
        ids, tcs = [], []
        for k, tc in enumerate(r.tool_calls):
            tid = tc.get("id") or f"py{self.counters['cells'] + 1}_{k}"
            ids.append(tid)
            args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
            tcs.append({"id": tid, "type": "function", "function": {"name": tc.get("name"), "arguments": json.dumps(args, default=str)}})
        msg = {**(r.raw_message or {}), "role": "assistant", "content": r.text or ""}
        if tcs:
            msg["tool_calls"] = tcs
        self.messages.append(msg)
        self.meter.anchor(r.usage, len(self.messages))  # never on a discarded reply or a compaction call (B's note 1)
        if (r.usage or {}).get("prompt_tokens"):
            self._anchor_n = len(self.messages)
        py = [k for k, tc in enumerate(r.tool_calls) if tc.get("name") == "python"]
        code = str((r.tool_calls[py[0]].get("arguments") or {}).get("code") or "") if py else ""
        call["ledger_index"] = self._log("assistant", text=r.text or "", code=code, call_id=call["call_id"],
                                         tools=[tc.get("name") for tc in r.tool_calls])
        self.tree.calls.append(call)
        if not py:
            if r.tool_calls:  # B10: a World tool called by name: a precise correction, no continuation used
                self._correct(r, ids, call)
                self.stall.on_cell(f"<called {', '.join(str(tc.get('name')) for tc in r.tool_calls)} directly>", False)
                return self._check_stall()
            if self._continuations < self.cfg.max_continuations:
                self._continuations += 1
                self.counters["continuations"] += 1
                self.messages.append({"role": "user", "content": CONTINUE})
                self._log("user", text=CONTINUE, role="continuation")
                return False
            call["actions"].append("done")
            self.answer = r.text or ""
            self._end("agent_done", (r.text or "")[:2000])
            return True
        # one python call per reply; the others are kept in the message (they are in the sampled tokens and in the model's
        # context) and answered, never silently dropped (G1 D14)
        done = await self._run_cell(call, ids[py[0]], code)
        extra = [k for k in range(len(ids)) if k != py[0]]
        if extra:
            self.counters["refused_calls"] += len(extra)
            call["refused_calls"] = len(extra)
            for k in extra:
                tc = r.tool_calls[k]
                why = (self._correction(tc) if tc.get("name") != "python" else
                       "[not run: one python call per reply; this reply's first python call ran, put everything in one call]")
                self.messages.append({"role": "tool", "tool_call_id": ids[k], "content": why})
        return done

    def _correction(self, tc: dict[str, Any]) -> str:
        name = str(tc.get("name") or "")
        args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
        if name in self._tools_view() or re.fullmatch(r"[A-Za-z_][\w-]*\.[A-Za-z_]\w*", name):
            line = f"r = await world.call({name!r}, **{json.dumps(args, default=str)})"
            return (f"[not run: `{name}` is a World tool, not a tool you can call. Your only tool is `python`; call it from code:\n{line}\n"
                    "print(r)]")
        return f"[not run: there is no tool `{name}`. Your only tool is `python`: write Python code (World tools via `await world.call(name, **args)`).]"

    def _correct(self, r: Any, ids: list[str], call: dict[str, Any]) -> None:
        self.counters["misrouted_calls"] += len(r.tool_calls)
        call["misrouted_calls"] = len(r.tool_calls)
        for tid, tc in zip(ids, r.tool_calls, strict=True):
            self.messages.append({"role": "tool", "tool_call_id": tid, "content": self._correction(tc)})
        self._log("notice", text=f"[correction] {len(r.tool_calls)} tool call(s) by name: {[tc.get('name') for tc in r.tool_calls]}")

    # ---- the loop ---------------------------------------------------------------------------------------------------------
    async def _renew(self) -> None:
        """Extend a renewable kernel container before it ends by itself (B9)."""
        ln = self.launcher
        if not getattr(ln, "renewable", False) or getattr(ln, "deadline", None) is None or getattr(ln, "name", None) is None:
            return
        if ln.deadline - time.time() > 900:
            return
        try:
            await ln.renew()
            self.counters["renewals"] += 1
        except KernelError as e:
            self._log("notice", text=f"[kernel container renewal failed: {e}]"[:2000])

    async def _loop(self) -> None:
        """Turns until a typed stop (§3.6)."""
        while self.stop is None:
            if self._cancelled:
                self._end("cancelled", "cancelled by the parent agent")
                break
            if self.depth == 0 and self.tree.fault is not None:
                self.failure_kind, self.error = self.tree.fault
                self._end("policy_error", self.error)
                break
            if self.cfg.wall_s and time.monotonic() - self.tree.started >= self.cfg.wall_s:
                self._end("budget:wall_s", f"the episode's wall-clock budget ({self.cfg.wall_s:g} s) is spent")
                break
            if self.counters["turns"] >= self.max_turns:
                self._end("budget:child_turns" if self.depth else "budget:turns", f"{self.counters['turns']} turns (max_turns {self.max_turns})")
                break
            await self._renew()
            await self._boundary()
            if self._compact_request is not None or self._over_threshold():
                gaps = self.health["turns_between_compactions"][-3:] + [self.counters["turns"] - self._turn_at_compaction]
                if self.counters["compactions"] >= 3 and self._compact_request is None and max(gaps) <= 1:
                    # thrash: every compaction leaves room for at most one turn (a livelock that spends the budget on summaries)
                    self._end("compaction_failed", f"the last compactions left room for at most one turn each ({gaps}): the window is too small "
                                                   "for the summary, the state and one turn of work; raise window_tokens")
                    break
                steps = self._steps_at_compaction[-(NO_PROGRESS_COMPACTIONS - 1):] + [self.tree.step]
                if self.depth == 0 and self.tree.step > 0 and len(steps) == NO_PROGRESS_COMPACTIONS and len(set(steps)) == 1:
                    # a context loop: compactions with no World action between them (re-reading its own past, then compacting)
                    self.counters["stalls"] += 1
                    self._end("stalled", f"stalled: {NO_PROGRESS_COMPACTIONS - 1} compactions in a row with no World action between them "
                                         f"(World step {self.tree.step})")
                    break
                self._steps_at_compaction.append(self.tree.step)
                if not await self._compact("request" if self._compact_request is not None else "threshold"):
                    break
                continue
            if self.spec.mode == "none" and self.meter.tokens(self.messages) > self.spec.window_tokens - self.max_out:
                self._end("context_overflow", f"the context ({self.meter.tokens(self.messages)} tokens) no longer fits the window and compaction is off")
                break
            if await self._turn():
                break

    # ---- ChildRunner (tree.py) ---------------------------------------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        """One round of a child agent (a persistent child's follow-up rounds continue the same context)."""
        self._rounds += 1
        try:
            if self.kernel is None:
                await self._start_fresh()
            self.stop, self.stop_reason, self.answer = None, "", None
            await self._loop()
        except (KernelStartError, HostAbort) as e:  # the child's sandbox or provider: infra for the whole episode
            kind = getattr(e, "failure_kind", None) or "crucible:backend_error"
            if self.tree.fault is None:
                self.tree.fault = (kind, f"child agent {self.agent}: {e}")
            return {"status": "failed", "answer": self._last_output, "error": f"{kind}: {e}"[:2000]}
        return self._result()

    def _result(self) -> dict[str, Any]:
        s = self.stop or "agent_done"
        if s == "agent_done":
            return {"status": "done", "answer": self.answer or "", "error": None}
        if s == "cancelled":
            return {"status": "cancelled", "answer": self._last_output, "error": "cancelled"}
        if truncating_stop(s):
            return {"status": "budget", "answer": self.answer or self._last_output, "error": f"{s}: {self.stop_reason}"[:2000]}
        return {"status": "failed", "answer": self.answer or self._last_output, "error": f"{s}: {self.stop_reason}"[:2000]}

    def deliver(self, text: str, kind: str) -> None:
        self._delivered.append((kind, str(text)))
        self.stall.note_progress(kind)

    def cancel(self) -> None:
        self._cancelled = True

    def progress(self) -> dict[str, Any]:
        return {"turns": self.counters["turns"], "tokens": self.counters["completion_tokens"], "last": self._last_output[-2048:]}

    async def aclose(self) -> None:
        """A child's kernel and container (its ledger stays: the root's close closes it)."""
        if self.registry is not None:
            with contextlib.suppress(Exception):
                await self.registry.aclose()
        if self.kernel is not None:
            await self.kernel.close()
        if hasattr(self.launcher, "aclose"):
            await self.launcher.aclose()

    # ---- evidence ---------------------------------------------------------------------------------------------------------
    def context_health(self) -> dict[str, Any]:
        """The context-health metrics of the evidence file (never in the reward)."""
        h = self.health
        return {"num_compactions": self.counters["compactions"], "compaction_attempts": self.counters["compaction_attempts"],
                "compaction_failures": self.counters["compaction_failures"], "turns_between_compactions": h["turns_between_compactions"],
                "chars_dropped": sum(h["chars_dropped"]), "summary_chars": sum(h["summary_chars"]), "chars_dropped_by_compaction": h["chars_dropped"],
                "summary_chars_by_compaction": h["summary_chars"], "history_reads": self.counters["history_reads"],
                "history_reads_after_compaction": self.counters["history_reads_after_compaction"], "stall_verdicts": dict(h["stall_verdicts"]),
                "restores_by_mode": dict(h["restores_by_mode"]), "overflow_retries": self.counters["overflow_retries"], "overflows": h["overflows"][-20:],
                "largest_prompt": dict(h["largest_prompt"]), "threshold": self.threshold, "window": self.spec.window_tokens,
                "largest_prompt_over_threshold": round(max(h["largest_prompt"].values(), default=0) / max(1, self.threshold), 4),
                "misrouted_calls": self.counters["misrouted_calls"], "meter_ratio": round(self.meter.ratio, 4),
                "delegations": self.tree.supervisor.usage()["by_depth"].get(self.depth, {}).get("spawns", 0)}

