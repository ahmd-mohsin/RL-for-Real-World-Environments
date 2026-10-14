"""``RLMPolicy`` (m14, rebuilt in M16): a Recursive-Language-Model policy for very long World episodes: the root
``AgentLoop`` of the agent tree (``agent.py``) plus the World ``act()`` bridge.

The model's only tool is ``python`` (a persistent REPL in a separate kernel process, ``kernel.py``, a no-network Docker
container by default). The kernel holds ``TASK``, ``TOOLS``, ``OBS`` (every tool result), ``INBOX``, ``history`` / ``OUT``
(the ledger, every message and full cell output, on disk outside the kernel), ``rlm`` (sub-LLMs, ``rlm.agent`` children,
``rlm.compact``, ``rlm.budget``, ``rlm.finish`` / ``rlm.give_up``) and, for a task that gives one, ``CONTEXT`` (a large
input read in pieces). The loop, the compaction, the kernel restores and the records are ``AgentLoop``'s
(``docs/research/long-horizon-rlm/04-design.md`` §3.6).

**World calls are World steps.** ``await world.call("app.tool", **args)`` inside a cell is a host request; the policy
returns it from ``act()`` as a ``ToolCall``; ``run_episode`` steps the World; the next ``act(obs)`` resolves the awaiting
cell with the result. Step limits, interception, the trace, dense progress and every M9 check see one World action per
call, exactly as for ``LLMToolPolicy``. ``world.send(to, text)`` is a ``Message``. A child agent granted ``world="shared"``
sends its World actions through the same bridge, one at a time under a lock, each attributed to the child's model call.

**Checkpoint (M16 stress B3, B5).** ``state_dict`` = the current context (one window), counters, the meter, stall,
supervisor and child-registry states, the ledger's path and length, the kernel's last snapshot and the journal *since*
it, and the model calls as an offset into ``calls.<actor>.jsonl`` (streamed when the harness gives a directory; the
in-flight call travels inline). The m14 checkpoint held the whole episode (16.9 MB per save at 1,914 steps).
``load_state_dict`` + ``recover(world)`` (``runtime.checkpoint.resume``) reopen the ledger (a ``resume`` window), restore
the kernel from its snapshot + suffix, finish a cell the checkpoint interrupted (the World call stepped before the
checkpoint is answered from the restored trace), and open a new segment (``boundary_reason="resume"``) so the first call
after a resume stores its whole prompt: a resumed rollout is train-valid (B3: it was masked ``capture_incomplete``).
Running children are not resumed (``tree.ChildRegistry``: reported failed "lost at resume").
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import shutil
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

from crucible.actors.policy import ContextConfig, Policy
from crucible.actors.rlm.agent import (
    BOOTSTRAP,
    PYTHON_TOOL,
    RLM_SYSTEM,
    AgentLoop,
    CallLog,
    RLMConfig,
    Tree,
    clone_launcher,
    truncating_stop,
)
from crucible.actors.rlm.kernel import JournalEntry, Kernel, KernelDivergence, Snapshot, default_launcher, lifetime_for_episode
from crucible.actors.rlm.ledger import Ledger
from crucible.actors.rlm.tree import ChildRegistry, TreeSupervisor
from crucible.core.schema.actions import Action, Done, Message, Observation, ToolCall

__all__ = ["BOOTSTRAP", "PYTHON_TOOL", "RLM_SYSTEM", "RLMConfig", "RLMPolicy"]

# the m14 constructor keywords: each maps onto an RLMConfig field
_LEGACY = {"stall_repeat": "stall_repeat_cells", "max_cell_errors": "stall_max_cell_errors"}


def _config(config: RLMConfig | None, legacy: dict[str, Any]) -> RLMConfig:
    """``config`` (default: ``RLMConfig(temperature=0.0)``) with the m14 keyword arguments applied."""
    base = config or RLMConfig(temperature=0.0)
    names = {f.name for f in dataclasses.fields(RLMConfig)}
    upd: dict[str, Any] = {}
    for k, v in legacy.items():
        if k == "context":
            cc: ContextConfig | None = v
            if cc is None:
                continue
            upd["budget_tokens"] = 0 if cc.mode == "none" else cc.budget_tokens
            upd["summary_max_tokens"] = cc.summary_max_tokens
            if cc.mode == "action":
                upd["context_mode"] = "action"
        elif k in _LEGACY:
            upd[_LEGACY[k]] = v
        elif k in names:
            upd[k] = v
        else:
            raise TypeError(f"RLMPolicy got an unexpected keyword argument {k!r}")
    return dataclasses.replace(base, **upd) if upd else base


class RLMPolicy(AgentLoop, Policy):
    def __init__(self, actor_id: str, llm: Any, *, sub_llm: Any = None, launcher: Any = None, child_launcher: Any = None, seed: int | None = None,
                 config: RLMConfig | None = None, system: str | None = None, sub_trainable: bool | None = None, ledger_dir: Path | str | None = None,
                 stream_dir: Path | str | None = None, context_path: Path | str | None = None, **legacy: Any):
        """``launcher``: where the root kernel runs (default: an isolated container, refused where Docker is missing; never
        shared between episodes); ``child_launcher``: a factory of launchers for child agents (default: a fresh one of the
        root's kind). ``seed``: the kernel's seed (default: derived from the episode id in ``begin_episode``). ``config``:
        ``RLMConfig`` (the m14 keyword arguments ``max_output_chars``, ``max_stalls``, ``context=ContextConfig`` ... still
        work and override it). ``ledger_dir``: where the ledger and kernel snapshots live (default: a temporary directory
        removed by ``aclose``); ``stream_dir``: where checkpoints stream the model calls (``calls.<actor>.jsonl``);
        ``context_path``: a host file mounted as ``CONTEXT``. ``sub_trainable``: whether sub-calls are the trained
        policy's samples (default: True only when no separate ``sub_llm`` answers them, G1 D1)."""
        cfg = _config(config, legacy)
        if launcher is None:
            default_launcher()  # model code never silently runs on the host (H3 V1): refused where Docker is missing
            from crucible.actors.rlm.kernel import DockerLauncher

            launcher = DockerLauncher(cfg.kernel_image, lifetime_s=lifetime_for_episode(cfg.wall_s), renewable=True)
        sup = TreeSupervisor(cfg.tree_limits())
        calls = CallLog(actor_id, Path(stream_dir) / f"calls.{actor_id}.jsonl" if stream_dir else None)
        tree = Tree(actor=actor_id, cfg=cfg, spec=cfg.context_spec(), supervisor=sup, calls=calls,
                    launcher_factory=child_launcher or (lambda: clone_launcher(self.launcher)))
        self._tools: dict[str, Any] = {}
        super().__init__(tree=tree, llm=llm, sub_llm=sub_llm, sub_trainable=sub_trainable, launcher=launcher, depth=0,
                         registry=ChildRegistry(sup, tree.make_child, depth=0), seed=seed, system=system)
        tree.agents["root"] = self
        self._seed_fixed = seed is not None
        self._ledger_dir = Path(ledger_dir) if ledger_dir is not None else None
        self._owned_dir: Path | None = None
        self.mount_context(context_path)
        self.harness_notes = cfg.notes()
        self._task: asyncio.Task | None = None
        self._out: asyncio.Queue[Action] = asyncio.Queue()
        self._waiting: asyncio.Future | None = None
        self._pending_action: dict[str, Any] | None = None
        self._inbox: list[dict[str, Any]] = []
        self._resume: dict[str, Any] | None = None
        self._resume_notice: str | None = None
        self._recovered: Observation | None = None
        self._replaying_pending: dict[str, Any] | None = None
        self._n_actions = 0
        self._instruction = ""
        self._world_lock = asyncio.Lock()

    def begin_episode(self, episode_id: str) -> None:
        self.tree.sample_key = episode_id
        if not self._seed_fixed:  # one seed per episode (H3 V12: every rlm_world_rollout kernel was seeded 0)
            self.seed = zlib.crc32(f"{episode_id}:{self.actor_id}".encode())

    # ---- what the harness and the record builder read --------------------------------------------------------------
    @property
    def segments(self) -> list[dict[str, Any]]:  # every agent's (each row names its agent)
        return self.tree.segments

    @property
    def trajectory_tokens(self) -> int:
        return self.tree.trajectory_tokens

    @property
    def context_spec(self) -> Any:
        return self.spec

    # ---- the ledger ---------------------------------------------------------------------------------------------------
    def _make_ledger(self) -> Ledger:
        root = self._ledger_dir
        if root is None:
            root = self._owned_dir = Path(tempfile.mkdtemp(prefix="crucible-rlm-ledger-"))
        return Ledger(root, name="root")

    # ---- the act() <-> driver bridge ----------------------------------------------------------------------------------
    async def act(self, obs: Observation) -> Action:
        self.tree.step = obs.step
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
            # an exception out of the driver is the harness's or the sandbox's, never the policy's (H3 V6, G1 D6): a kernel or
            # sandbox that cannot start or was lost carries its infra kind; anything else is a harness fault (masked, counted)
            self.failure_kind = self.failure_kind or getattr(exc, "failure_kind", None) or "agent_run_error"
            self.error = self.error or f"{type(exc).__name__}: {exc}"
            await self._close_kernel()
            return Done(reason=f"policy error: {exc}")
        await self._close_kernel()
        return Done(reason=self.stop or "rlm driver ended")

    async def _emit_and_wait(self, action: Action, pending: dict[str, Any]) -> Observation:
        fut = asyncio.get_running_loop().create_future()
        self._pending_action, self._waiting = pending, fut
        await self._out.put(action)
        obs = await fut
        self._waiting, self._pending_action = None, None
        return obs

    async def _world_step(self, owner: AgentLoop, kind: str, *, tool: str | None = None, args: dict[str, Any] | None = None, to: str | None = None,
                          text: str | None = None) -> dict[str, Any]:
        """One World action for ``owner`` (this root or a child granted ``shared``), one at a time."""
        if self._replaying_pending is not None:
            # resumed inside a cell: this request was already stepped before the checkpoint; answer it from the trace
            self._replaying_pending = self._pending_action = None
            if kind == "tool":
                tr = self._recovered.tool_results[0] if self._recovered and self._recovered.tool_results else None
                self._recovered = None
                return {"ok": bool(tr.ok) if tr else False, "content": tr.content if tr else "",
                        "error": tr.error if tr else "the result of this call was lost in the resume"}
            return {"ok": True}
        async with self._world_lock:
            self._n_actions += 1
            acts = owner._current_call["actions"] if owner._current_call is not None else None
            if kind == "tool":
                cid = f"{self.actor_id}:rlm{self._n_actions}"
                if acts is not None:
                    acts.append(f"tool:{cid}")
                obs = await self._emit_and_wait(ToolCall(tool_call_id=cid, function_name=str(tool), arguments=dict(args or {})),
                                                {"kind": "tool", "id": cid, "tool": tool, "args": args, "agent": owner.agent})
                tr = next((r for r in obs.tool_results if r.tool_call_id == cid), None)
                return {"ok": bool(tr.ok) if tr else False, "content": tr.content if tr else "", "error": tr.error if tr else "no result returned"}
            if acts is not None:
                acts.append("message")
            await self._emit_and_wait(Message(to=str(to or ""), text=str(text or "")), {"kind": "message", "to": to, "text": text, "agent": owner.agent})
            return {"ok": True}

    def _take_inbox(self) -> list[dict[str, Any]]:
        out, self._inbox = self._inbox, []
        for m in out:
            self._log("inbox", **{"from": str(m.get("from"))}, text=str(m.get("text")), at=m.get("at"))
        self._cell_msgs += out  # announced after the cell: messages must not arrive silently (the live smoke's supervisor replies did)
        return out

    async def _boundary(self) -> None:
        await super()._boundary()
        news = self._take_inbox()
        if news:
            await self._deliver_inbox(news)

    async def _deliver_inbox(self, msgs: list[dict[str, Any]]) -> None:
        """Messages that arrived between cells: into ``INBOX`` by a host-written cell, announced (bounded) in the context."""
        await self._ensure_kernel()
        r, _ = await self._execute(f"_deliver({msgs!r})", None, timeout_s=10)
        if r.timed_out or r.killed:  # model code redefined _deliver to block: restore like any wedged cell
            await self._restore(drop_last=True, why="a message delivery cell blocked")
        self.messages.append({"role": "user", "content": self._announce(msgs, max(80, self.max_output_chars // 4))})
        self.stall.note_progress("inbox")

    # ---- the episode ------------------------------------------------------------------------------------------------------
    async def _drive(self) -> None:
        if self._resume is not None:
            await self._continue_resumed()
        else:
            self.task_text = self._instruction
            await self._start_fresh()
        if self.stop is None:
            await self._loop()
        with contextlib.suppress(Exception):
            await self.registry.aclose()  # the children end with the root's episode
        await self._out.put(Done(reason=(self.stop_reason or self.stop or "rlm driver ended")[:2000]))

    async def _close_kernel(self) -> None:
        if self.kernel is not None:
            await self.kernel.close()

    async def aclose(self) -> None:
        """End of the episode however it ended: an episode stopped by a limit leaves the driver waiting inside a cell and
        the kernel process alive. Children, kernels, containers and the ledger are closed; the journal stays readable
        (``kernel.state()``) and the ledger stays on disk (the harness's attempt directory)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)  # the episode is over; its outcome was recorded
        with contextlib.suppress(Exception):
            await self.registry.aclose()
        await self._close_kernel()
        if hasattr(self.launcher, "aclose"):
            await self.launcher.aclose()  # the kernel's container or scratch directory
        if self.ledger is not None:
            self.ledger.close()
        if self._owned_dir is not None:
            shutil.rmtree(self._owned_dir, ignore_errors=True)

    # ---- checkpoint -------------------------------------------------------------------------------------------------------
    def _kernel_state(self) -> dict[str, Any] | None:
        """The kernel as a B5 checkpoint: its last snapshot and the journal since it (the whole journal when none)."""
        k = self.kernel
        if k is None:
            return None
        snap = k.last_snapshot
        if snap is None:
            return {"seed": k.seed, "cell_timeout_s": k.cell_timeout_s, "journal": [e.to_dict() for e in k.journal], "journal_offset": k.journal_offset,
                    "snapshot": None}
        cut = max(0, snap.journal_len - k.journal_offset)
        return {"seed": k.seed, "cell_timeout_s": k.cell_timeout_s, "journal": [e.to_dict() for e in k.journal[cut:]], "journal_offset": snap.journal_len,
                "snapshot": snap.to_dict()}

    def state_dict(self) -> dict[str, Any]:
        k = self.kernel
        in_cell = k is not None and bool(k.journal) and not k.journal[-1].done
        items = self.tree.calls.items
        busy = [a._current_call for a in self.tree.agents.values() if a._current_call is not None]
        keep_from = min((j for j in range(len(items) - 1, -1, -1) for c in busy if items[j] is c), default=len(items))
        ks = self._kernel_state()
        self._ckpt_snapshot = ks["snapshot"]["path"] if ks and ks.get("snapshot") else None
        return {"version": 2, "messages": self.messages, "calls": self.tree.calls.checkpoint(keep_from), "kernel": ks, "in_cell": in_cell,
                "counters": dict(self.counters), "health": self.health, "stop": self.stop, "stop_reason": self.stop_reason, "segment": self.segment,
                "segments": self.tree.segments, "trajectory_tokens": self.tree.trajectory_tokens, "trajectory_sources": sorted(self.tree.trajectory_sources),
                "notes": self.notes, "n_actions": self._n_actions, "instruction": self._instruction, "task_text": self.task_text, "tools": self._tools,
                "system": self.system, "pending_action": self._pending_action, "inbox": self._inbox, "sample_key": self.tree.sample_key, "step": self.tree.step,
                "seed": self.seed, "continuations": self._continuations, "last_summary": self._last_summary, "compact_request": self._compact_request,
                "turn_at_compaction": self._turn_at_compaction, "wrapped": sorted(self._wrapped), "lost_names": self._lost_names,
                "chain_calls": self._chain_calls_in_segment, "meter": self.meter.state_dict(), "stall": self.stall.state_dict(),
                "supervisor": self.tree.supervisor.state_dict(), "registry": self.registry.state_dict(),
                # (M16 hunt H10) every ledger of the tree's event count, so a resume hides what a crashed attempt wrote after it
                "ledger": None if self.ledger is None else {"root": str(self.ledger.dir.parent), "name": self.ledger.name,
                                                            "lengths": self.ledger.lengths()},
                "ledger_budget_used": self.tree.ledger_budget.used,
                "context_path": str(self.context_file.path) if self.context_file is not None else None,
                "elapsed_s": time.monotonic() - self.tree.started}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d.get("version") != 2:
            raise ValueError("not an M16 RLMPolicy checkpoint (version 2); an m14 checkpoint cannot resume under the M16 harness")
        self.messages, self.counters, self.health = d["messages"], dict(d["counters"]), dict(d["health"])
        self.tree.calls.load(d["calls"])
        # (M16 hunt H7) a checkpoint taken during a compaction (a shared child's World step) holds that compaction's attempts
        # but not the rebuilt context: the resume compacts again from the old one, so an accepted attempt no segment continued
        # from was trained although the episode never ran on its summary. It is recorded, never trained
        used = {sg.get("summary_call") for sg in d["segments"]}
        for c in self.tree.calls.items:
            if c.get("kind") == "compaction" and c.get("accepted") and c["call_id"] not in used:
                c.update(accepted=False, trainable=False, abandoned="resume")
        self.stop, self.stop_reason, self.segment = d.get("stop"), d.get("stop_reason") or "", int(d["segment"])
        self.tree.segments[:] = list(d["segments"])
        self.tree.trajectory_tokens, self.tree.trajectory_sources = int(d["trajectory_tokens"]), set(d.get("trajectory_sources") or [])
        self.notes, self._n_actions, self._instruction, self.task_text = list(d["notes"]), int(d["n_actions"]), d["instruction"], d["task_text"]
        self._tools, self.system = dict(d["tools"]), d["system"]
        self._pending_action, self._inbox = d.get("pending_action"), list(d.get("inbox") or [])
        self.tree.sample_key, self.tree.step = d.get("sample_key"), int(d.get("step") or 0)
        self.tree.started = time.monotonic() - float(d.get("elapsed_s") or 0.0)
        self.seed, self._seed_fixed = d.get("seed"), True
        self._continuations, self._last_summary, self._compact_request = int(d["continuations"]), d.get("last_summary"), d.get("compact_request")
        self._turn_at_compaction, self._wrapped, self._lost_names = int(d["turn_at_compaction"]), set(d.get("wrapped") or []), list(d.get("lost_names") or [])
        self.meter.load_state_dict(d["meter"])
        self.stall.load_state_dict(d["stall"])
        self.tree.supervisor.load_state_dict(d["supervisor"])
        self.registry.load_state_dict(d["registry"])
        self.tree.ledger_budget.used = int(d.get("ledger_budget_used") or 0)
        if d.get("context_path") and self.context_file is None:
            self.mount_context(d["context_path"])
        self._resume = {"kernel": d["kernel"], "in_cell": d.get("in_cell"), "ledger": d.get("ledger")}

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

    async def _replay(self, kernel: Kernel, journal: list[Any], snap: Snapshot | None, offset: int) -> JournalEntry | None:
        from crucible.world.state.base import SnapshotMismatch

        try:
            return await kernel.replay(journal, snapshot=snap, offset=offset)
        except (SnapshotMismatch, KernelDivergence):  # no snapshot and a strict replay diverged: lenient (the model is told)
            return await kernel.replay(journal, strict=False)

    async def _continue_resumed(self) -> None:
        """Reopen the ledger, restore the kernel from its snapshot + journal suffix, open the ``resume`` segment, and finish
        a cell the checkpoint interrupted (its recorded replies first, then live: the pending World call's result comes
        from ``recover``)."""
        st, self._resume = self._resume, None
        led = st["ledger"]
        self.ledger = Ledger.open(led["root"], name=led["name"])
        # (M16 hunt H10) events a crashed attempt wrote after this checkpoint (in the root's and every child's ledger) describe
        # a future the restore undid: history.* showed the model World results and outputs that never happened in this
        # attempt. They are abandoned (hidden from every kernel view; the log stays append-only) before any replay
        lengths = led.get("lengths") if isinstance(led.get("lengths"), dict) else {"": int(led.get("len") or 0)}
        gone = self.ledger.rewind_to(lengths)
        abandoned = sum(gone.values())
        w = self.ledger.open_window("resume", step=self.tree.step, checkpoint_events=int(lengths.get("", 0)), abandoned_events=abandoned)
        ks = st["kernel"]
        self.kernel = Kernel(self.launcher, seed=ks["seed"], cell_timeout_s=ks["cell_timeout_s"])
        self.kernel.resolve = self._resolve_record
        snap = Snapshot.from_dict(ks["snapshot"]) if ks.get("snapshot") else None
        pending = await self._replay(self.kernel, ks["journal"], snap, int(ks.get("journal_offset") or 0))
        lr = dict(self.kernel.last_restore or {})
        self.counters["restores"] += 1
        self.health["restores_by_mode"][lr.get("mode") or "full"] = self.health["restores_by_mode"].get(lr.get("mode") or "full", 0) + 1
        self._log("restore", text="resume from a checkpoint", mode=lr.get("mode"), cells=lr.get("cells"), seconds=lr.get("seconds"),
                  lost=lr.get("lost") or [], divergences=lr.get("divergences"), abandoned_events=abandoned)
        if lr.get("mode") == "lenient" or lr.get("lost"):
            self.counters["inexact_restores"] += lr.get("mode") == "lenient"
            self._resume_notice = (f"[the episode resumed from a checkpoint; the kernel was rebuilt ({lr.get('mode')})"
                                   + (f"; variables lost: {', '.join((lr.get('lost') or [])[:20])}" if lr.get("lost") else "")
                                   + "; check the variables you rely on]")
        before = self.meter.tokens(self.messages)
        self._new_segment({"index": self.segment + 1, "agent": self.agent, "strategy": "resume", "boundary_reason": "resume", "summary_call": None,
                           "context_before": before, "context_after": before, "tokens_source": self.meter.source, "window": w, "step": self.tree.step,
                           "restore_mode": lr.get("mode"), "restore_s": lr.get("seconds")})
        if pending is not None:
            call = next((c for c in reversed(self.model_calls) if c["kind"] == "turn" and not c.get("agent")), None) or {"call_id": None, "actions": []}
            tc = next((m for m in reversed(self.messages) if m["role"] == "assistant" and m.get("tool_calls")), None)
            tc_id = tc["tool_calls"][0]["id"] if tc else "resumed"
            # the World call stepped before the checkpoint is answered from the trace, if it was the root's own (a child's
            # was lost with the child: children are not resumed)
            pa = self._pending_action
            self._replaying_pending = pa if pa and not pa.get("agent") else None
            if self._replaying_pending is None:
                self._pending_action = self._recovered = None
            self.counters["cells"] -= 1  # the interrupted cell is counted once
            try:
                await self._run_cell(call, tc_id, pending.code, entry=pending)
            except KernelDivergence as e:
                # (M16 hunt H8) the re-run of the interrupted cell asked the host for something other than its record (it read
                # state a replay does not reproduce: a file, the C clock): the KernelDivergence escaped the driver and masked the
                # resumed attempt too (agent_run_error, retried again). The kernel is restored to before the cell and the model is
                # told what happened, as for a cell killed mid-flight; the World actions it took before the checkpoint stand
                self._replaying_pending = self._pending_action = self._recovered = None
                self._current_call = None
                how = await self._restore(drop_last=True, why=f"the resumed cell could not be continued: {str(e)[:300]}")
                self.messages.append({"role": "tool", "tool_call_id": tc_id, "content": (
                    f"[the episode resumed from a checkpoint inside this cell, and the cell could not be continued (re-running it took "
                    f"another path than before); {how}. The World actions it took still happened: check their results "
                    f"(`await history.since({self.tree.step})`) before repeating anything; never blindly repeat the interrupted cell.]")})
        if self._resume_notice and self.stop is None:
            self.messages.append({"role": "user", "content": self._resume_notice})
            self._resume_notice = None

    async def restore_probe(self) -> dict[str, Any]:
        """Time a restore of the final kernel into a fresh sandbox, the way a resume would do it (the last snapshot + the
        journal since it; the whole journal when no snapshot was taken). For the ladders (E's request 4)."""
        st = self._kernel_state()
        if st is None:
            return {"method": None, "error": "no kernel"}
        snap = Snapshot.from_dict(st["snapshot"]) if st.get("snapshot") else None
        reopened = self.ledger is not None and self.ledger._closed
        if reopened:
            self.ledger = Ledger.open(self.ledger.dir.parent, name=self.ledger.name)
        launcher = self.tree.launcher_factory()
        fresh = Kernel(launcher, seed=st["seed"], cell_timeout_s=st["cell_timeout_s"])
        fresh.resolve = self._resolve_record
        t0 = time.perf_counter()
        err = None
        try:
            await self._replay(fresh, st["journal"], snap, int(st.get("journal_offset") or 0))
        except Exception as e:  # noqa: BLE001 -- a failed probe is a measurement
            err = f"{type(e).__name__}: {str(e)[:300]}"
        s = round(time.perf_counter() - t0, 3)
        lr = dict(fresh.last_restore or {})
        with contextlib.suppress(Exception):
            await fresh.close()
        with contextlib.suppress(Exception):
            await launcher.aclose()
        if reopened:
            self.ledger.close()
        return {"restore": "snapshot" if snap is not None else "journal_replay", "restore_s": s, "mode": lr.get("mode"), "cells": len(st["journal"]),
                "snapshot_cells": snap.journal_len if snap else 0, "snapshot_bytes": snap.bytes if snap else 0,
                "journal_bytes": len(json.dumps(st["journal"], default=str)), "fallback": lr.get("fallback"), "error": err}

    # ---- evidence ---------------------------------------------------------------------------------------------------------
    def evidence(self) -> dict[str, Any]:
        """``rlm.<actor>.json``'s body (never in the reward): counters (the root's and the tree's), segments, context health,
        the ledger's stats, the tree's usage, the kernel journal (what the code did between World steps) and the stop."""
        agents = {p: {"counters": a.counters, "stop": a.stop, "health": a.context_health()} for p, a in self.tree.agents.items() if p != "root"}
        tree_counters: dict[str, Any] = {}
        for a in self.tree.agents.values():
            for k, v in a.counters.items():
                if isinstance(v, (int, float)):
                    tree_counters[k] = tree_counters.get(k, 0) + v
        return {"counters": self.counters, "tree_counters": tree_counters, "segments": self.tree.segments, "notes": self.notes,
                "harness_notes": self.harness_notes, "stop": self.stop, "stop_reason": self.stop_reason[:2000], "truncated": truncating_stop(self.stop),
                "context_health": self.context_health(), "children": agents, "tree": self.tree.supervisor.usage(),
                "ledger": self.ledger.stats() if self.ledger is not None else None, "ledger_budget": self.tree.ledger_budget.stats(),
                "trajectory_tokens": {"tokens": self.tree.trajectory_tokens, "source": "+".join(sorted(self.tree.trajectory_sources)) or None},
                "kernel": self.kernel.state() if self.kernel is not None else None,
                "last_restore": self.kernel.last_restore if self.kernel is not None else None}
