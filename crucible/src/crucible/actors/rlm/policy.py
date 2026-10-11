"""``RLMPolicy`` (m14): a Recursive-Language-Model policy for very long World episodes.

The model's only tool is ``python`` (a persistent REPL in a separate kernel process, ``kernel.py``). Prompt-as-variable for
real: the kernel holds ``TASK``, ``TOOLS``, ``OBS`` (every tool result) and ``INBOX``; the model's context holds the
system prompt, the task and its own turns, whose output is capped at ``max_output_chars`` (8192, the RLM default), so
large observations stay in variables and reach the model only as far as it prints them (RLM blog; Prime Agent asks for
this habit but pre-loads nothing).

**World calls are World steps.** ``await world.call("app.tool", **args)`` inside a cell is a host request; the policy
returns it from ``act()`` as a ``ToolCall``; ``run_episode`` steps the World; the next ``act(obs)`` resolves the awaiting
cell with the result. Step limits, interception, the trace, dense progress and every M9 check see one World action per
call, exactly as for ``LLMToolPolicy``. ``world.send(to, text)`` is a ``Message``.

**Sub-LLMs** (the RLM's depth-1 recursion): ``await rlm.query(prompt, context=None)`` and ``await rlm.batch([...])`` run in
the host (in parallel) and are recorded as model calls ``kind="sub"`` with ``parent`` = the root call whose cell issued
them, token records included: a sub-call shares its root turn's credit.

**Long horizon:** ``rlm.finish(summary)`` ends the episode (answer-as-variable); a reply with no code gets a goal
continuation (audit against every requirement, Prime Agent's goal driver) up to ``max_continuations``; compaction at
``ContextConfig.trigger`` writes a recorded summary whose prompt carries the kernel note and lists live variables after;
stall signatures (identical cells, consecutive errors, identical repeated World calls; Continual Harness failure
signatures) inject a ``[stall-detected]`` notice and stop the episode after ``max_stalls``; a timed-out or wedged cell
restores the kernel by journal replay to before the cell.

**Checkpoint:** ``state_dict`` = context, model calls, the kernel journal (an in-progress cell included) and counters;
``load_state_dict`` replays the kernel; ``recover(world)`` (called by ``runtime.checkpoint.resume``) finds the result of a
World call that was stepped just before the checkpoint in the restored trace.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from crucible.actors.llm import BudgetExceeded, ContextOverflow, LLMClient, LLMError
from crucible.actors.policy import ContextConfig, Policy, approx_tokens
from crucible.actors.rlm.kernel import CellResult, JournalEntry, Kernel
from crucible.core.schema.actions import Action, Done, Message, Observation, ToolCall

PYTHON_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python code in your persistent REPL. Top-level await works. Variables persist between calls. "
                   "Printed output and the value of a final expression are returned (truncated).",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "the Python code to run"}}, "required": ["code"]}}}

RLM_SYSTEM = """You are an agent working in a simulated workplace through a persistent Python REPL: the `python` tool. Everything is code.
- `TASK` is your instruction. `TOOLS` maps each tool name ("app.tool") to its description and JSON schema.
- Call a tool: `r = await world.call("app.tool", **args)` (or `await world.app.tool(**args)`). `r` is a dict {ok, content, error}. Every result is also appended to `OBS`.
- Message a person: `await world.send(to, text)`. Messages to you are appended to `INBOX` (dicts with from, text, at); new ones are announced.
- Sub-LLMs: `await rlm.query(prompt, context=None)` returns a string; `await rlm.batch([prompt, ...])` runs prompts in parallel. Use them to read, filter or analyse large data instead of printing it.
- What you print is truncated to {max_output} characters. Keep large data in variables and print only what you need.
- Variables, functions and imports persist across calls, and across context compaction.
- When the task is complete, call `rlm.finish("<short summary>")`. Before that, check the current state against every requirement."""

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
    async def note(self, text):
        return await host("rlm.note", text=str(text)[:512])

world = _World()
rlm = _RLM()
def _deliver(msgs):
    INBOX.extend(msgs)
'''

COMPACT = ("Your context is nearly full. Write a summary that lets you continue without the earlier history: what is done, what is "
           "pending and why, every rule that changes how you act, and the exact ids and values you still need. The Python kernel keeps "
           "running after this summary: every variable, function and import stays available, but the cells that defined them will not "
           "be visible, so name the variables worth remembering. Reply with the summary only, no tool call.")
SUMMARY_HEAD = "[Summary of your earlier work in this episode; the full history was compacted]\n"
CONTINUE = ("[goal: continuation] You stopped without calling rlm.finish(). The task is not marked complete. Check the current state "
            "against every requirement in TASK (use your variables and the tools), finish what is missing, then call rlm.finish(summary).")


class RLMPolicy(Policy):
    def __init__(self, actor_id: str, llm: LLMClient, *, sub_llm: LLMClient | None = None, launcher: Any = None, seed: int = 0,
                 max_output_chars: int = 8192, cell_timeout_s: float = 120.0, max_tokens: int = 2000, sub_max_tokens: int = 1000,
                 temperature: float = 0.0, context: ContextConfig | None = None, max_continuations: int = 2, max_stalls: int = 3,
                 stall_repeat: int = 3, max_cell_errors: int = 4, max_turns: int = 500, system: str = RLM_SYSTEM):
        self.actor_id, self.llm, self.sub_llm = actor_id, llm, sub_llm or llm
        self.launcher, self.seed = launcher, seed
        self.max_output_chars, self.cell_timeout_s = max_output_chars, cell_timeout_s
        self.max_tokens, self.sub_max_tokens, self.temperature = max_tokens, sub_max_tokens, temperature
        self.context, self.max_continuations, self.max_stalls = context, max_continuations, max_stalls
        self.stall_repeat, self.max_cell_errors, self.max_turns = stall_repeat, max_cell_errors, max_turns
        self.system = system.replace("{max_output}", str(max_output_chars))
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system}]
        self.model_calls: list[dict[str, Any]] = []
        self.kernel: Kernel | None = None
        self.stop: str | None = None
        self.error: str | None = None
        self.failure_kind: str | None = None
        self.sample_key: str | None = None
        self.segment, self.segments = 0, []
        self.trajectory_tokens = 0
        self.notes: list[str] = []
        self.counters = {"turns": 0, "cells": 0, "cell_errors_in_row": 0, "continuations": 0, "stalls": 0, "world_calls": 0, "sub_calls": 0}
        self._recent_cells: list[str] = []
        self._recent_world: list[str] = []
        self._task: asyncio.Task | None = None
        self._out: asyncio.Queue[Action] = asyncio.Queue()
        self._waiting: asyncio.Future | None = None
        self._pending_action: dict[str, Any] | None = None
        self._inbox: list[dict[str, Any]] = []
        self._current_call: dict[str, Any] | None = None
        self._resume: dict[str, Any] | None = None
        self._recovered: Observation | None = None
        self._replaying_pending: dict[str, Any] | None = None
        self._cell_msgs: list[dict[str, Any]] = []
        self._n_actions = 0
        self._instruction = ""
        self._tools: dict[str, Any] = {}

    def begin_episode(self, episode_id: str) -> None:
        self.sample_key = episode_id

    # ---- the act() <-> driver bridge ----------------------------------------------------------------------------------
    async def act(self, obs: Observation) -> Action:
        for m in obs.inbox:
            if m.frm != self.actor_id:
                self._inbox.append({"from": m.frm, "text": m.text, "at": m.at})
        if self._task is None:
            if obs.tools:
                self._tools = {t.qualified: {"description": t.description, "schema": t.input_schema} for t in obs.tools}
            if obs.instruction:
                self._instruction = obs.instruction
            self._task = asyncio.get_running_loop().create_task(self._drive())
        elif self._waiting is not None and not self._waiting.done():
            if self._recovered is not None and not obs.tool_results:
                obs, self._recovered = self._recovered, None
            self._waiting.set_result(obs)
        getter = asyncio.ensure_future(self._out.get())
        await asyncio.wait({getter, self._task}, return_when=asyncio.FIRST_COMPLETED)
        if getter.done():
            return getter.result()
        getter.cancel()
        exc = self._task.exception() if self._task.done() and not self._task.cancelled() else None
        if exc is not None:
            self.failure_kind = self.failure_kind or "crucible:policy_error"
            self.error = self.error or f"{type(exc).__name__}: {exc}"
            await self._close_kernel()
            return Done(reason=f"policy error: {exc}")
        await self._close_kernel()
        return Done(reason=self.stop or "rlm driver ended")

    async def _emit_and_wait(self, action: Action) -> Observation:
        fut = asyncio.get_running_loop().create_future()
        self._waiting = fut
        await self._out.put(action)
        obs = await fut
        self._waiting = None
        return obs

    # ---- host requests from cells --------------------------------------------------------------------------------------
    def _take_inbox(self) -> list[dict[str, Any]]:
        out, self._inbox = self._inbox, []
        self._cell_msgs += out  # announced after the cell: messages must not arrive silently (the live smoke's supervisor replies did)
        return out

    async def _host(self, data: dict[str, Any]) -> Any:
        t = data.get("type")
        if self._replaying_pending is not None and t in ("world.call", "world.send"):
            # resumed inside a cell: this request was already stepped before the checkpoint; answer it from the trace
            self._replaying_pending = None
            self._pending_action = None
            if t == "world.call":
                tr = (self._recovered.tool_results[0] if self._recovered and self._recovered.tool_results else None)
                self._recovered = None
                res = {"ok": bool(tr.ok) if tr else False, "content": tr.content if tr else "",
                       "error": (tr.error if tr else "the result of this call was lost in the resume")}
                return {**res, "inbox": self._take_inbox()}
            return {"ok": True, "inbox": self._take_inbox()}
        if t == "world.call":
            self._n_actions += 1
            self.counters["world_calls"] += 1
            cid = f"{self.actor_id}:rlm{self._n_actions}"
            tool, args = str(data.get("tool") or ""), dict(data.get("args") or {})
            self._pending_action = {"kind": "tool", "id": cid, "tool": tool, "args": args}
            if self._current_call is not None:
                self._current_call["actions"].append(f"tool:{cid}")
            obs = await self._emit_and_wait(ToolCall(tool_call_id=cid, function_name=tool, arguments=args))
            self._pending_action = None
            tr = next((r for r in obs.tool_results if r.tool_call_id == cid), None)
            res = {"ok": bool(tr.ok) if tr else False, "content": tr.content if tr else "", "error": tr.error if tr else "no result returned"}
            key = hashlib.sha256(json.dumps([tool, args, res["content"]], sort_keys=True, default=str).encode()).hexdigest()
            self._recent_world = (self._recent_world + [key])[-self.stall_repeat:]
            return {**res, "inbox": self._take_inbox()}
        if t == "world.send":
            self._n_actions += 1
            if self._current_call is not None:
                self._current_call["actions"].append("message")
            self._pending_action = {"kind": "message", "to": data.get("to"), "text": data.get("text")}
            await self._emit_and_wait(Message(to=str(data.get("to") or ""), text=str(data.get("text") or "")))
            self._pending_action = None
            return {"ok": True, "inbox": self._take_inbox()}
        if t == "llm.query":
            return await self._sub(str(data.get("prompt") or ""), data.get("context"), data.get("max_tokens"))
        if t == "llm.batch":
            items = data.get("items") or []
            return list(await asyncio.gather(*(self._sub(str(p), c, data.get("max_tokens")) for p, c in items)))
        if t == "rlm.note":
            self.notes.append(str(data.get("text") or ""))
            return "noted"
        raise ValueError(f"unknown host request {t!r}")

    async def _sub(self, prompt: str, context: str | None, max_tokens: int | None) -> str:
        content = prompt if context is None else f"{prompt}\n\n<context>\n{context}\n</context>"
        r = await self.sub_llm.complete([{"role": "user", "content": content}], tools=None, max_tokens=int(max_tokens or self.sub_max_tokens),
                                        temperature=self.temperature, sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)
        self.counters["sub_calls"] += 1
        parent = self._current_call["call_id"] if self._current_call else None
        self.model_calls.append({"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "kind": "sub", "parent": parent, "segment": self.segment,
                                 "model": r.model, "finish": r.raw_finish, "usage": r.usage, "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3),
                                 "cached": r.cached, "actions": [], "tokens": r.tokens})
        return r.text or ""

    # ---- kernel ---------------------------------------------------------------------------------------------------------
    async def _ensure_kernel(self) -> None:
        if self.kernel is not None and self.kernel.proc is not None:
            return
        self.kernel = Kernel(self.launcher, seed=self.seed, cell_timeout_s=self.cell_timeout_s)
        code = BOOTSTRAP.replace("__TASK__", repr(self._instruction)).replace("__TOOLS__", repr(self._tools))
        r = await self.kernel.execute(code, self._host, timeout_s=60)
        if not r.ok:
            raise RuntimeError(f"RLM bootstrap failed: {r.error}")

    async def _close_kernel(self) -> None:
        if self.kernel is not None:
            await self.kernel.close()

    async def _restore_before_last(self) -> None:
        """A cell timed out or wedged the kernel: rebuild the kernel from the journal without that cell."""
        assert self.kernel is not None
        journal = [e for e in self.kernel.journal[:-1]]
        await self.kernel.replay(journal)

    def _format(self, r: CellResult) -> str:
        parts = []
        if r.stdout:
            parts.append(r.stdout.rstrip("\n"))
        if r.stderr:
            parts.append("[stderr]\n" + r.stderr.rstrip("\n"))
        if r.result is not None:
            parts.append(r.result)
        if r.error:
            parts.append(r.error.rstrip("\n"))
        text = "\n".join(parts) or "(no output)"
        n = self.max_output_chars
        if len(text) > n:
            h = n * 3 // 4
            text = text[:h] + f"\n[... {len(text) - n} characters truncated: keep large data in variables and print slices ...]\n" + text[-(n - h):]
        return text

    # ---- the model loop -------------------------------------------------------------------------------------------------
    async def _drive(self) -> None:
        if self._resume is not None:
            await self._continue_resumed()
            if self.stop is not None:
                return
        else:
            await self._ensure_kernel()
            self.messages.append({"role": "user", "content": self._instruction})
            self.trajectory_tokens += approx_tokens(self.messages)
        while self.counters["turns"] < self.max_turns:
            await self._maybe_compact()
            news = self._take_inbox()
            if news:
                await self._deliver_inbox(news)
            try:
                r = await self.llm.complete(self.messages, tools=[PYTHON_TOOL], max_tokens=self.max_tokens, temperature=self.temperature,
                                            sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)
            except ContextOverflow as e:
                self.stop = "context_overflow"
                self.error = str(e)
                return
            except (LLMError, BudgetExceeded) as e:
                self.failure_kind = getattr(e, "failure_kind", None) or "provider_unavailable"
                self.error = f"{self.failure_kind}: {e}"
                self.stop = "policy_error"
                return
            self.counters["turns"] += 1
            call = {"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "kind": "turn", "segment": self.segment,
                    "context_tokens": approx_tokens(self.messages), "model": r.model, "finish": r.raw_finish, "usage": r.usage,
                    "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3), "cached": r.cached, "actions": [], "tokens": r.tokens}
            self.model_calls.append(call)
            self.trajectory_tokens += int((r.usage or {}).get("completion_tokens") or 0)
            code_calls = [tc for tc in r.tool_calls if tc["name"] == "python"]
            if not code_calls:
                self.messages.append({**(r.raw_message or {}), "role": "assistant", "content": r.text or ""})
                if self.counters["continuations"] < self.max_continuations:
                    self.counters["continuations"] += 1
                    self.messages.append({"role": "user", "content": CONTINUE})
                    continue
                self.stop = "agent_done"
                call["actions"].append("done")
                await self._out.put(Done(reason=(r.text or "")[:2000]))
                return
            tc = code_calls[0]
            tc_id = tc.get("id") or f"py{self.counters['cells'] + 1}"
            code = str((tc.get("arguments") or {}).get("code") or "")
            self.messages.append({**(r.raw_message or {}), "role": "assistant", "content": r.text or "",
                                  "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}]})
            if await self._run_cell(call, tc_id, code):
                return

    async def _run_cell(self, call: dict[str, Any], tc_id: str, code: str, entry: JournalEntry | None = None) -> bool:
        """Execute one cell (or continue a resumed one); returns True when the episode is over."""
        assert self.kernel is not None
        self._current_call = call
        self.counters["cells"] += 1
        self._cell_msgs = []
        r = await self.kernel.execute(code, self._host, entry=entry)
        self._current_call = None
        out = self._format(r)
        if self._cell_msgs:
            out += "\n[" + str(len(self._cell_msgs)) + " new message(s) appended to INBOX] " + "; ".join(
                f"from {m['from']}: {str(m['text'])[:200]}" for m in self._cell_msgs)
        if r.timed_out or r.killed:
            await self._restore_before_last()
            out += "\n[the cell did not finish in time; the kernel state was restored to before this cell, World actions it took still happened]"
        self.messages.append({"role": "tool", "tool_call_id": tc_id, "content": out})
        self.trajectory_tokens += approx_tokens([self.messages[-1]])
        if r.finished is not None:
            self.stop = "agent_done"
            call["actions"].append("done")
            await self._out.put(Done(reason=str(r.finished)[:2000]))
            return True
        return await self._check_stall(code, r)

    async def _check_stall(self, code: str, r: CellResult) -> bool:
        self.counters["cell_errors_in_row"] = self.counters["cell_errors_in_row"] + 1 if not r.ok else 0
        norm = " ".join(code.split())
        self._recent_cells = (self._recent_cells + [norm])[-self.stall_repeat:]
        sig = None
        if len(self._recent_cells) == self.stall_repeat and len(set(self._recent_cells)) == 1:
            sig = f"the same code ran {self.stall_repeat} times in a row"
        elif self.counters["cell_errors_in_row"] >= self.max_cell_errors:
            sig = f"{self.counters['cell_errors_in_row']} cells in a row failed"
        elif len(self._recent_world) == self.stall_repeat and len(set(self._recent_world)) == 1:
            sig = f"the same World call returned the same result {self.stall_repeat} times"
        if sig is None:
            return False
        self.counters["stalls"] += 1
        self._recent_cells, self._recent_world = [], []
        self.counters["cell_errors_in_row"] = 0
        if self.counters["stalls"] >= self.max_stalls:
            self.stop = "too_many_errors"
            await self._out.put(Done(reason=f"stalled: {sig}"))
            return True
        self.messages.append({"role": "user", "content": f"[stall-detected: {sig}] Your approach is not making progress. Inspect the state "
                                                         "(your variables, OBS, the tool results), state what is wrong, and change the approach."})
        return False

    async def _deliver_inbox(self, msgs: list[dict[str, Any]]) -> None:
        assert self.kernel is not None
        await self.kernel.execute(f"_deliver({msgs!r})", self._host, timeout_s=10)
        lines = "; ".join(f"from {m['from']}: {str(m['text'])[:200]}" for m in msgs)
        self.messages.append({"role": "user", "content": f"[{len(msgs)} new message(s) appended to INBOX] {lines}"})

    async def _maybe_compact(self) -> None:
        cfg = self.context
        if cfg is None or cfg.mode == "none" or approx_tokens(self.messages) < cfg.trigger * cfg.budget_tokens:
            return
        before = approx_tokens(self.messages)
        r = await self.llm.complete([*self.messages, {"role": "user", "content": COMPACT}], tools=None, max_tokens=cfg.summary_max_tokens,
                                    temperature=self.temperature, sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)
        self.model_calls.append({"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "kind": "compaction", "segment": self.segment,
                                 "model": r.model, "finish": r.raw_finish, "usage": r.usage, "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3),
                                 "cached": r.cached, "actions": [], "tokens": r.tokens, "context_tokens": before})
        names = await self.kernel.names() if self.kernel is not None else []
        starts = [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]
        k = cfg.keep_recent_turns
        recent = self.messages[starts[-k]:] if k and len(starts) >= k else []
        self.messages = [self.messages[0], {"role": "user", "content": self._instruction}, {"role": "user", "content": SUMMARY_HEAD + (r.text or "")},
                         {"role": "user", "content": "[python-state] Your kernel kept running. Live names: " + ", ".join(n for n in names if n not in
                                                                                                                         ("world", "rlm", "host", "asyncio"))},
                         *recent]
        self.segments.append({"index": self.segment + 1, "strategy": "summary", "summary_call": self.model_calls[-1]["call_id"],
                              "context_before": before, "context_after": approx_tokens(self.messages), "kept_turns": min(k, len(starts))})
        self.segment += 1

    # ---- checkpoint -----------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        in_cell = self.kernel is not None and bool(self.kernel.journal) and not self.kernel.journal[-1].done
        return {"messages": self.messages, "model_calls": self.model_calls, "kernel": self.kernel.state() if self.kernel else None,
                "counters": dict(self.counters), "stop": self.stop, "segment": self.segment, "segments": self.segments,
                "trajectory_tokens": self.trajectory_tokens, "notes": self.notes, "n_actions": self._n_actions, "instruction": self._instruction,
                "tools": self._tools, "pending_action": self._pending_action, "inbox": self._inbox, "recent_cells": self._recent_cells,
                "recent_world": self._recent_world, "sample_key": self.sample_key, "in_cell": in_cell}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.messages, self.model_calls, self.counters = d["messages"], d["model_calls"], dict(d["counters"])
        self.stop, self.segment, self.segments, self.trajectory_tokens = d.get("stop"), d["segment"], d["segments"], d["trajectory_tokens"]
        self.notes, self._n_actions, self._instruction, self._tools = d["notes"], d["n_actions"], d["instruction"], d["tools"]
        self._pending_action, self._inbox = d.get("pending_action"), list(d.get("inbox") or [])
        self._recent_cells, self._recent_world, self.sample_key = d["recent_cells"], d["recent_world"], d.get("sample_key")
        self._resume = {"kernel": d["kernel"], "in_cell": d.get("in_cell")}

    def recover(self, world: Any) -> None:
        """After a resume: the World call the policy was waiting on was stepped before the checkpoint; find its result."""
        pa = self._pending_action
        if not pa or pa.get("kind") != "tool":
            return
        from crucible.core.schema.actions import ToolResult

        ev = next((e for e in reversed(world.log) if e.kind == "observation" and e.type == "tool_result" and (e.data or {}).get("id") == pa["id"]), None)
        if ev is not None:
            tr = ToolResult(tool_call_id=pa["id"], function_name=pa["tool"], ok=bool(ev.data.get("ok")), content=ev.text or "", error=ev.data.get("error"))
            self._recovered = Observation(actor=self.actor_id, step=world.step_index, tool_results=[tr])

    async def _continue_resumed(self) -> None:
        """Rebuild the kernel from the journal; if the checkpoint was taken inside a cell, finish that cell (its recorded
        replies first, then live: the pending World call's result comes from ``recover``)."""
        st, self._resume = self._resume, None
        self.kernel = Kernel(self.launcher, seed=st["kernel"]["seed"], cell_timeout_s=st["kernel"]["cell_timeout_s"])
        pending = await self.kernel.replay(st["kernel"]["journal"])
        if pending is None:
            return
        call = next((c for c in reversed(self.model_calls) if c["kind"] == "turn"), None) or {"call_id": None, "actions": []}
        tc = next((m for m in reversed(self.messages) if m["role"] == "assistant" and m.get("tool_calls")), None)
        tc_id = tc["tool_calls"][0]["id"] if tc else "resumed"
        self._replaying_pending = self._pending_action
        self.counters["cells"] -= 1  # the interrupted cell is counted once
        await self._run_cell(call, tc_id, pending.code, entry=pending)
