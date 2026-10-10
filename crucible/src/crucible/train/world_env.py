"""Crucible Worlds as M12 environments: a World episode driven by ``LLMToolPolicy`` whose model calls go to the M12
engines with exact token-in-token-out (``chat``), graded by the task's own grader (M7), recorded as
``crucible.rollout/v1`` by M10's record builder (linear delta tokens, versions, spans, segments, per-call credit).

``EngineLLM`` is a drop-in for ``actors.llm.LLMClient`` (the ``complete`` surface ``LLMToolPolicy`` uses). Per episode it
tracks the conversation it has served: when the policy's next message list *extends* it (the assistant turn it produced
+ new messages), the prompt is ``prev_prompt + completion (+EOS) + codec.suffix(new)``; otherwise (compaction rebuilt
the context, or a harness rewrote history) it renders the whole list and a new segment begins.

``world_rollout(...)`` returns the runtime's ``rollout_fn``: open the task's World, run the episode, grade it; a masked
episode or masked grade returns ``None`` (the runtime masks the rollout, never scores it 0); an agent-caused failure is
scored by the grader.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from crucible.actors.llm import LLMResponse
from crucible.train.chat import ChatCodec
from crucible.train.engine import GenResult


class _Spend:
    def __init__(self) -> None:
        self.usd, self.uncertain_usd, self.calls = 0.0, 0.0, 0


class EngineLLM:
    def __init__(self, codec: ChatCodec, gen: Callable[[list[int], int, float], Awaitable[GenResult]], *, model: str = "policy"):
        self.codec, self.gen, self.model = codec, gen, model
        self.spend = _Spend()
        self._served: list[dict[str, Any]] | None = None  # the message list of the last call (without its assistant reply)
        self._prev_ids: list[int] | None = None  # last prompt + completion (+EOS)
        self.prefix_breaks = 0

    def _prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[int]:
        s = self._served
        if s is not None and self._prev_ids is not None and len(messages) > len(s) and messages[: len(s)] == s and messages[len(s)].get("role") == "assistant":
            return self._prev_ids + self.codec.suffix(messages[len(s) + 1:], tools)
        if s is not None:
            self.prefix_breaks += 1  # rebuilt context (compaction): a new segment, rendered whole
        return self.codec.render(messages, tools)

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, temperature: float = 1.0,
                       max_tokens: int = 512, **_: Any) -> LLMResponse:
        prompt = self._prompt(messages, tools)
        t0 = time.monotonic()
        res = await self.gen(prompt, max_tokens, temperature)
        self.spend.calls += 1
        comp = list(res.completion_ids)
        tail = comp if (comp and comp[-1] in self.codec.eos_ids) else comp + list(self.codec.eos_ids[:1])
        self._served = [dict(m) for m in messages]
        self._prev_ids = prompt + tail
        text, calls = self.codec.parse(comp)
        finish = "stop" if res.finish == "stop" else ("length" if res.finish == "length" else res.finish)
        return LLMResponse(text=text, tool_calls=calls, usage={"prompt_tokens": len(prompt), "completion_tokens": len(comp)}, model=self.model,
                           raw_finish=finish, latency_s=time.monotonic() - t0,
                           tokens={"prompt_token_ids": prompt, "completion_token_ids": comp, "logprobs": list(res.logprobs),
                                   "policy_version": res.version, "version_spans": res.version_spans})


def world_rollout(task_for: Callable[[str], Any], codec: ChatCodec, *, max_steps: int = 30, max_tokens: int = 512, context: Any = None,
                  workroot: Path | None = None) -> Callable:
    """``task_for(prompt) -> RunTask`` (M10 contract: ``open(workdir)``, ``grade(world, workdir)``)."""
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.core.schema.run import RunRecord
    from crucible.runtime.record import build

    async def rollout(rid: str, prompt: str, seed: int, gen: Callable[[list[int], int, float], Awaitable[GenResult]], *, temperature: float = 1.0,
                      harness: str = "native:world@1"):
        task = task_for(prompt)
        d = Path(tempfile.mkdtemp(prefix="m12-world-", dir=workroot))
        world = await task.open(d / "work")
        try:
            llm = EngineLLM(codec, gen)
            pol = LLMToolPolicy("agent", llm, max_tokens=max_tokens, temperature=temperature, context=context)
            ep = await run_episode(world, {"agent": pol}, limits=EpisodeLimits(max_steps=max_steps), reset=True)
            if ep.masked:
                return None
            rep = await task.grade(world, d / "grade")
            if rep.infra.get("status") == "masked":
                return None
            rec = RunRecord(run_id=rid, task=getattr(task, "id", prompt), world=getattr(task, "world_id", ""), policy="engine", started_at="", steps=ep.steps,
                            outcome="completed", reward=rep.reward, attempt=f"{rid}-a1",
                            meta={"harness": harness, "stop": ep.stop_condition, "truncated": ep.truncated, "gates": rep.gate_bits()})
            ro = build(rec, calls=ep.model_calls, segments={"agent": list(pol.segments)} if pol.segments else None, sampling={"temperature": temperature})
            ro.group = None
            return ro
        finally:
            await world.close()

    return rollout
