"""The agent side of one slime episode: the adapter's turn recorder with the checkpoint hook, and the scripted model and
CLI harness of the P6 equivalence run (swarm-pipeline/00 §11 item 3; 03 §2.1, §4.3).

- **The hook.** While slime's adapter serves a turn, the CLI is blocked on the model, so a snapshot taken in the adapter's
  ``_run_turn`` *before* ``call_sglang_generate`` (``slime/agent/adapters/common.py:423``) sees a quiescent workspace;
  ``debug_callback`` (:377) fires after the reply is flushed, too late (03 §4.3). ``AdapterSession.turn`` runs the
  ``before_model_call`` hook first, then the model, then records the turn: the order slime's subclass must keep. The hook
  reports the *previous* turn's actions to the checkpoint observer, so the state it snapshots is the state after them,
  exactly the state the native harness's sidecar sees after each World step.
- **Turns.** ``TurnRecord`` keeps the fields of slime's ``TurnRecord`` (``slime/agent/trajectory.py:40-54``) the bridge
  needs (prompt and output ids, output log probs, finish reason, weight versions) plus the parsed actions and the
  observation tokens fed back before the next turn. A session linearizes into one segment (no compaction branch: the
  scripted harness never compacts; the ported ``TrajectoryManager`` is P7+).
- **Scripted model and harness** (the equivalence run, $0): ``ScriptedModel`` replays a step list (``("write", path,
  text)``, ``("shell", command)``, ``("final", text)``, ``("chaos", fn)`` runs a host side effect such as killing the
  container) as model turns with deterministic token ids; ``ScriptedCLIHarness`` is the CLI: it executes each turn's
  actions through the ``Sandbox`` protocol only (``write_file``, ``exec``), as Claude Code or Codex would, and stops on a
  turn without actions or when its time budget runs out (``EXIT_TIME_BUDGET_EXCEEDED``: a limit stop, graded).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from crucible.runtime.harness.sandbox import EXIT_TIME_BUDGET_EXCEEDED, Sandbox
from crucible.verify.collect import snapshot_command

# the in-sandbox snapshot (``verify.collect.snapshot_command``) as a ``str.format`` template on ``{base}``: pass the base commit
# recorded at open (``SlimeTaskEnv.open``), never the agent-movable ``base`` tag (harness audit R6, 2026-10-28)
SNAPSHOT_CMD = snapshot_command("{base}", require_sha=False)


def toy_tokens(text: str, *, vocab: int = 50_000) -> list[int]:
    """Deterministic token ids for the scripted path (one id per whitespace word; never trained on)."""
    return [int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % vocab + 1 for w in str(text).split()] or [1]


@dataclass
class ModelTurn:
    text: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)  # {"id", "tool": "shell" | "write_file", "arguments": {...}}
    output_ids: list[int] = field(default_factory=list)
    output_log_probs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"
    weight_version: str | None = None


@dataclass
class TurnRecord:
    index: int
    prompt_ids: list[int]
    output_ids: list[int]
    output_log_probs: list[float]
    finish_reason: str
    weight_versions: list[str]
    actions: list[dict[str, Any]]
    text: str
    obs_ids: list[int] = field(default_factory=list)  # observation tokens fed back before this turn (loss mask 0)


class TurnModel(Protocol):
    async def __call__(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> ModelTurn: ...


class AgentHarness(Protocol):
    """slime's ``BaseHarness.run`` reduced to what the generate function calls (``slime/agent/harness/common.py:81-104``):
    run the agent in ``sb`` against the adapter session until it exits; the exit code (``EXIT_TIME_BUDGET_EXCEEDED`` on
    the time budget)."""

    async def run(self, sb: Sandbox, session: AdapterSession, prompt: str, *, time_budget_s: float) -> int: ...


class AdapterSession:
    """One adapter session (slime: one ``sid``): ``turn`` = hook, model call, record (flush-then-record order of
    ``common.py:441-469``). ``observe`` adds tool output to the next turn's observation tokens."""

    def __init__(self, sid: str, model: TurnModel, *, sampling_params: dict[str, Any] | None = None,
                 before_model_call: Callable[[AdapterSession], Awaitable[None]] | None = None, tokenize: Callable[[str], list[int]] = toy_tokens):
        self.sid, self.model, self.sampling = sid, model, dict(sampling_params or {})
        self.hook = before_model_call
        self.tokenize = tokenize
        self.turns: list[TurnRecord] = []
        self._obs: list[int] = []
        self._prompt: list[int] = []

    async def turn(self, messages: list[dict[str, Any]]) -> ModelTurn:
        if self.hook is not None:
            await self.hook(self)
        if not self.turns:
            self._prompt = self.tokenize(" ".join(str(m.get("content", "")) for m in messages))
        t = await self.model(messages, self.sampling)
        ids = list(t.output_ids) or self.tokenize(t.text or " ".join(a.get("tool", "") for a in t.actions))
        lps = (list(t.output_log_probs) + [0.0] * len(ids))[: len(ids)]
        self.turns.append(TurnRecord(len(self.turns) + 1, list(self._prompt), ids, lps, t.finish_reason, [t.weight_version] if t.weight_version else [],
                                     [dict(a) for a in t.actions], t.text, self._obs))
        self._obs = []
        return t

    def observe(self, text: str) -> None:
        self._obs = self._obs + self.tokenize(text)

    def linearize(self) -> tuple[list[int], list[int], list[int], list[float], list[tuple[int, int]]]:
        """(prompt ids, response ids, loss mask, log probs, per-turn completion spans in response coordinates)."""
        resp: list[int] = []
        mask: list[int] = []
        lps: list[float] = []
        spans = []
        for t in self.turns:
            resp += t.obs_ids
            mask += [0] * len(t.obs_ids)
            lps += [0.0] * len(t.obs_ids)
            s = len(resp)
            resp += t.output_ids
            mask += [1] * len(t.output_ids)
            lps += t.output_log_probs
            spans.append((s, len(resp)))
        return list(self._prompt), resp, mask, lps, spans


def step_turn(st: tuple, k: int, *, weight_version: str | None = "0") -> ModelTurn:
    """One scripted step as the model turn both paths replay (the native twin: ``factory.envs.ScriptedLLM``)."""
    if st[0] == "write":
        return ModelTurn(actions=[{"id": f"t{k}", "tool": "write_file", "arguments": {"path": st[1], "content": st[2]}}], weight_version=weight_version,
                         output_ids=toy_tokens(f"write_file {st[1]} {st[2]}"))
    if st[0] == "shell":
        return ModelTurn(actions=[{"id": f"t{k}", "tool": "shell", "arguments": {"command": st[1]}}], weight_version=weight_version,
                         output_ids=toy_tokens(f"shell {st[1]}"))
    if st[0] == "final":
        return ModelTurn(text=st[1], weight_version=weight_version, output_ids=toy_tokens(st[1]))
    raise ValueError(f"unknown scripted step {st[0]!r}")


class ScriptedSteps:
    """A cursor over scripted steps: ``chaos`` steps run their callable (a host side effect, e.g. killing the World's
    container) before the next turn is returned; past the end the model says ``done``."""

    def __init__(self, steps: Sequence[tuple]):
        self.steps, self.i, self.k = list(steps), 0, 0

    async def next(self, *, weight_version: str | None = "0") -> ModelTurn:
        while self.i < len(self.steps) and self.steps[self.i][0] == "chaos":
            out = self.steps[self.i][1]()
            if asyncio.iscoroutine(out):
                await out
            self.i += 1
        self.k += 1
        if self.i >= len(self.steps):
            return ModelTurn(text="done", weight_version=weight_version, output_ids=toy_tokens("done"))
        st = self.steps[self.i]
        self.i += 1
        return step_turn(st, self.k, weight_version=weight_version)


class ScriptedModel:
    """A model that replays ``steps``, one turn per call (``ScriptedSteps``); ``delay_s`` makes each call slow (the
    agent-timeout tests)."""

    def __init__(self, steps: Sequence[tuple], *, weight_version: str = "0", delay_s: float = 0.0):
        self.cursor = ScriptedSteps(steps)
        self.steps = self.cursor.steps
        self.weight_version, self.delay_s = weight_version, delay_s

    async def __call__(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> ModelTurn:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return await self.cursor.next(weight_version=self.weight_version)


class ScriptedCLIHarness:
    """The CLI of the equivalence run: executes each turn's actions through the sandbox only."""

    def __init__(self, *, shell_timeout_s: int = 600, max_turns: int = 500):
        self.shell_timeout_s, self.max_turns = shell_timeout_s, max_turns
        self.trail: list[dict[str, Any]] = []

    async def run(self, sb: Sandbox, session: AdapterSession, prompt: str, *, time_budget_s: float) -> int:
        deadline = time.monotonic() + time_budget_s
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for _ in range(self.max_turns):
            if time.monotonic() >= deadline:
                return EXIT_TIME_BUDGET_EXCEEDED
            try:
                t = await asyncio.wait_for(session.turn(messages), max(0.001, deadline - time.monotonic()))
            except TimeoutError:
                return EXIT_TIME_BUDGET_EXCEEDED
            messages.append({"role": "assistant", "content": t.text, "actions": t.actions})
            if not t.actions:
                return 0
            for a in t.actions:
                args = dict(a.get("arguments") or {})
                self.trail.append({"function": f"repo.{a['tool']}", "arguments": args})
                if a["tool"] == "write_file":
                    await sb.write_file(str(args["path"]), str(args.get("content", "")))
                    result = f"wrote {len(str(args.get('content', '')).encode())} bytes to {args['path']}"
                elif a["tool"] == "shell":
                    rc, out, err = await sb.exec(str(args["command"]), timeout=self.shell_timeout_s)
                    result = f"exit {rc}\n{out}{err}"
                else:
                    result = f"unknown tool {a['tool']}"
                session.observe(result)
                messages.append({"role": "tool", "content": result})
        return EXIT_TIME_BUDGET_EXCEEDED
