"""Crucible Worlds as M12 environments: a World episode driven by ``LLMToolPolicy`` whose model calls go to the M12
engines with exact token-in-token-out (``chat``), graded by the task's own grader (M7), recorded as
``crucible.rollout/v1`` by M10's record builder (linear delta tokens, versions, spans, segments, per-call credit).

``EngineLLM`` is a drop-in for ``actors.llm.LLMClient`` (the ``complete`` surface ``LLMToolPolicy`` uses). It tracks
the conversations it has served (a bounded table, so one instance can serve many concurrent episodes): when a message
list *extends* a served one (the assistant turn it produced + new messages), the prompt is ``prev_prompt + completion
(+EOS) + codec.suffix(new)``; otherwise (a new conversation, compaction rebuilt the context, a harness rewrote history)
it renders the whole list and a new segment begins. ``LLMProvider.tito`` routes ``LLMClient`` through it, because a
chat API re-renders history through its tool parser and template, which is lossy (lab: Qwen3 + hermes re-rendered
``open.\n\n<tool_call>`` as ``open.\n\n\n<tool_call>``, a prefix break on 23 of 24 calls).

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
    def __init__(self, codec: ChatCodec, gen: Callable[[list[int], int, float], Awaitable[GenResult]], *, model: str = "policy",
                 max_conversations: int = 4096, thinking: str | None = None):
        """``thinking`` (m13 C-7, A2-48): how earlier reasoning is shown, matching the deployment template:
        "preserved" (extend token for token), "interleaved" (extend inside a tool loop; a new user message re-renders the
        history, which drops reasoning: a new segment), "strip" (every call renders the history: each call its own
        segment). Default: the codec's own template behaviour; a mode the template does not implement is refused."""
        self.codec, self.gen, self.model = codec, gen, model
        probe = getattr(codec, "thinking_behaviour", None)
        actual = probe() if callable(probe) else "preserved"
        self.thinking = thinking or actual
        if self.thinking not in ("preserved", "interleaved", "strip"):
            raise ValueError(f"thinking: preserved | interleaved | strip (got {self.thinking!r})")
        if self.thinking != actual:
            raise ValueError(f"thinking={self.thinking!r} but the chat template is {actual!r}: training contexts would differ from deployment")
        self.rerenders = 0
        self._rerendered = False
        self.spend = _Spend()
        # served conversations: (message list of a call without its reply, that call's prompt + completion (+EOS))
        self._convs: list[tuple[list[dict[str, Any]], list[int]]] = []
        self.max_conversations = max_conversations
        self.prefix_breaks = 0

    def _prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[int]:
        self._rerendered = False
        best = None
        for i, (s, _ids) in enumerate(self._convs):
            if len(messages) > len(s) and messages[len(s)].get("role") == "assistant" and messages[: len(s)] == s and (best is None or len(s) > len(self._convs[best][0])):
                best = i
        if best is not None:
            s, ids = self._convs.pop(best)  # the conversation moves on: its next entry replaces this one
            new = messages[len(s) + 1:]
            if self.thinking == "strip" or (self.thinking == "interleaved" and any(m.get("role") == "user" for m in new)):
                self.rerenders += 1  # the deployed template re-renders (and drops) earlier reasoning here: so do we, in a new segment
                self._rerendered = True
                return self.codec.render(messages, tools)
            return ids + self.codec.suffix(new, tools)
        if any(m.get("role") == "assistant" for m in messages):
            self.prefix_breaks += 1  # mid-conversation but extends nothing served (compaction rewrote it): a new segment, rendered whole
        return self.codec.render(messages, tools)

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, temperature: float = 1.0,
                       max_tokens: int = 512, **_: Any) -> LLMResponse:
        prompt = self._prompt(messages, tools)
        t0 = time.monotonic()
        res = await self.gen(prompt, max_tokens, temperature)
        self.spend.calls += 1
        comp = list(res.completion_ids)
        tail = comp if (comp and comp[-1] in self.codec.eos_ids) else comp + list(self.codec.eos_ids[:1])
        self._convs.append(([dict(m) for m in messages], prompt + tail))
        del self._convs[: max(0, len(self._convs) - self.max_conversations)]
        text, calls = self.codec.parse(comp)
        finish = "stop" if res.finish == "stop" else ("length" if res.finish == "length" else res.finish)
        return LLMResponse(text=text, tool_calls=calls, usage={"prompt_tokens": len(prompt), "completion_tokens": len(comp)}, model=self.model,
                           raw_finish=finish, latency_s=time.monotonic() - t0,
                           tokens={"prompt_token_ids": prompt, "completion_token_ids": comp, "logprobs": list(res.logprobs),
                                   "policy_version": res.version, "version_spans": res.version_spans,
                                   **({"new_segment": True} if self._rerendered else {})})


def world_rollout(task_for: Callable[[str], Any], codec: ChatCodec, *, max_steps: int = 30, max_tokens: int = 512, context: Any = None,
                  workroot: Path | None = None, limit_policy: Any = None, make_policy: Callable[[Any, float], Any] | None = None,
                  harness_id: str = "native:world@1") -> Callable:
    """``task_for(prompt) -> RunTask`` (M10 contract: ``open(workdir)``, ``grade(world, workdir)``). ``make_policy(llm,
    temperature)`` replaces the default ``LLMToolPolicy`` (``rlm_world_rollout``: the RLM harness)."""
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.core.schema.run import RunRecord
    from crucible.runtime.record import build

    async def rollout(rid: str, prompt: str, seed: int, gen: Callable[[list[int], int, float], Awaitable[GenResult]], *, temperature: float = 1.0,
                      harness: str = harness_id):
        from crucible.core.prov import utc_now

        started_at = utc_now()
        task = task_for(prompt)
        d = Path(tempfile.mkdtemp(prefix="m12-world-", dir=workroot))
        world = await task.open(d / "work")
        try:
            llm = EngineLLM(codec, gen)
            pol = make_policy(llm, temperature) if make_policy else LLMToolPolicy("agent", llm, max_tokens=max_tokens, temperature=temperature, context=context)
            try:
                ep = await run_episode(world, {"agent": pol}, limits=EpisodeLimits(max_steps=max_steps), reset=True)
            finally:
                if hasattr(pol, "aclose"):
                    await pol.aclose()
            if ep.masked:
                return None
            rep = await task.grade(world, d / "grade")
            if rep.infra.get("status") == "masked":
                return None
            reward = rep.reward
            pen = limit_policy.penalty(ep.stop_condition) if limit_policy is not None else None
            if pen is not None:  # m13 C-8 (A2-24): an incomplete trajectory gets the configured penalty
                reward = pen
            rec = RunRecord(run_id=rid, task=getattr(task, "id", prompt), world=getattr(task, "world_id", ""), policy="engine", started_at=started_at, steps=ep.steps,
                            outcome="completed", reward=reward, attempt=f"{rid}-a1",
                            meta={"harness": harness, "stop": ep.stop_condition, "truncated": ep.truncated, "gates": rep.gate_bits()})
            ro = build(rec, calls=ep.model_calls, segments={"agent": list(pol.segments)} if pol.segments else None, sampling={"temperature": temperature})
            ro.group = None
            trace = world.to_trace()
            from crucible.credit.hack_detector import rule_flags

            ro.flags = rule_flags(trace)  # m13 C-4 level 1: judged (level 2) in the group stage
            from crucible.credit.shaping import format_errors_by_call

            fe = format_errors_by_call(trace, ep.model_calls)  # m13 C-8: malformed tool calls, charged to the issuing call
            for t in ro.turns:
                t.format_errors = int(fe.get(t.call_id, 0))
            if world.intercepts:
                ro.reward_components["hack/intercepted"] = float(len(world.intercepts))
            probes = (trace.outcome.details or {}).get("probes") or {}
            attempted = sum(int(v) for v in (probes.get("attempted") or {}).values()) if isinstance(probes.get("attempted"), dict) else 0
            if attempted:  # blocked shortcut attempts: the episode stays trainable (an *exposure* already masked it)
                ro.reward_components["probe/attempted"] = float(attempted)
            return ro
        finally:
            await world.close()

    return rollout


def rlm_world_rollout(task_for: Callable[[str], Any], codec: ChatCodec, config: Any = None, *, max_steps: int = 10_000, workroot: Path | None = None,
                      limit_policy: Any = None, launcher: Any = None) -> Callable:
    """The M12 rollout under the RLM harness (m14): the engine's model drives an ``RLMPolicy`` configured by ``RLMConfig``;
    root, sub and compaction calls carry the engine's token records; the harness label is ``rlm:<config digest>@1``."""
    from crucible.actors.rlm.policy import RLMPolicy
    from crucible.runtime.harness.rlm import RLMConfig

    c = config or RLMConfig()

    def make(llm: Any, temperature: float) -> Any:
        return RLMPolicy("agent", llm, launcher=launcher or c.launcher(), max_output_chars=c.max_output_chars, cell_timeout_s=c.cell_timeout_s, max_tokens=c.max_tokens,
                         sub_max_tokens=c.sub_max_tokens, temperature=temperature, context=c.context(), max_continuations=c.max_continuations,
                         max_stalls=c.max_stalls, max_turns=c.max_turns, tool_index=c.tool_index)

    return world_rollout(task_for, codec, max_steps=max_steps, workroot=workroot, limit_policy=limit_policy, make_policy=make,
                         harness_id=f"rlm:{c.digest()}@1")


def tito_llm(provider: Any, model: str, *, timeout_s: float, seed: int | None = None) -> EngineLLM:
    """``LLMProvider.tito``: an ``EngineLLM`` over a vLLM server's token-id completions endpoint, rendered with the
    model's own chat template. ``actors.llm.LLMClient`` resolves this through a pointer (it may not import train)."""
    from transformers import AutoTokenizer

    from crucible.core.config import resolve_secret
    from crucible.train.chat import HFChatCodec
    from crucible.train.engine import GenRequest
    from crucible.train.vllm_http import VLLMServerEngine

    codec = HFChatCodec(AutoTokenizer.from_pretrained(model), template_kwargs=provider.template_kwargs)
    eng = VLLMServerEngine(provider.base_url.rstrip("/").removesuffix("/v1"), model, timeout_s=timeout_s,
                           api_key=resolve_secret(provider.api_key) or None)
    calls = [0]

    async def gen(prompt: list[int], n: int, temp: float):
        calls[0] += 1
        return await eng.generate(GenRequest(rid="llm", call_id=f"c{calls[0]}", prompt_ids=prompt, max_tokens=n, temperature=temp,
                                             stop_ids=codec.eos_ids, seed=seed))

    return EngineLLM(codec, gen, model=model)
