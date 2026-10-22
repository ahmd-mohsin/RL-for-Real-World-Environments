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
scored by the grader. With ``seats=SeatConfig(...)`` (``train.seats``) it plays a turn-based game with every seat and
returns one record per trained seat.

Vision-language codecs (``chat.HFVisionChatCodec``) report the images each render / suffix expanded (``last_media``):
``EngineLLM`` makes their placeholder offsets absolute in the full prompt, keeps every image of a served conversation
(handed to the engine as ``gen(..., media=[...])``: an engine for a VL model needs all of them), and records the images a
call adds in ``tokens["media"]`` (``Turn.media``).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import json
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.actors.llm import ContextOverflow, LLMError, LLMResponse
from crucible.train.chat import ChatCodec
from crucible.train.engine import GenResult


class _Spend:
    def __init__(self) -> None:
        self.usd, self.uncertain_usd, self.calls = 0.0, 0.0, 0


REPLY_KEY = "crucible_reply"  # per-reply nonce in ``LLMResponse.raw_message``: policies replay it on the assistant turn


class EngineCallError(LLMError):
    """The engine (or the chat codec) failed a model call: an infrastructure / harness failure. It carries a named
    ``failure_kind`` so the episode is *masked* (``LLMToolPolicy`` and ``run_episode`` read it), never scored 0 as the
    policy's own error (M12 audit: a ``ValueError`` from the engine path reached the policy-error branch)."""

    def __init__(self, msg: str, *, failure_kind: str = "provider_unavailable"):
        super().__init__(msg)
        self.failure_kind = failure_kind


@dataclass
class _Conv:
    key: str | None  # the caller's ``sample_key`` (episode id): conversations of different episodes never extend each other
    msgs: list[dict[str, Any]]  # the call's message list, without its reply
    ids: list[int]  # that call's prompt + completion (+EOS)
    media: list[dict[str, Any]]  # every image in that prompt, absolute offsets
    reply: str  # the nonce handed out with the reply (``raw_message[REPLY_KEY]``)
    text: str
    calls: list[tuple[str, str]]  # (name, canonical json arguments) of the parsed tool calls


def _canon_calls(calls: Any) -> list[tuple[str, str]]:
    out = []
    for c in calls or []:
        f = c.get("function") if isinstance(c, dict) and isinstance(c.get("function"), dict) else c
        if not isinstance(f, dict):
            return [("?", repr(c))]
        args = f.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {"__unparseable__": args}
        out.append((str(f.get("name", "")), json.dumps(args, sort_keys=True, default=str)))
    return out


def _accepts(fn: Callable, name: str) -> bool:
    """``fn`` names the keyword itself (a ``**kw`` pass-through, e.g. ``EnvWorkerPool.bridge``, may wrap a ``gen`` that
    does not take it)."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def cut_at_stop(res: GenResult, stop_ids: Sequence[int]) -> GenResult:
    """The completion up to and including its first stop token (M12 audit V10: with no stop ids the engine sampled past
    ``<|im_end|>`` into invented turns, which were trained and parsed). A prefix of what was sampled, with its own
    behaviour log-probs, candidates, routes and version spans: still exactly what the engine produced."""
    ids = list(res.completion_ids)
    k = next((i + 1 for i, t in enumerate(ids) if t in stop_ids), None)
    if k is None or k == len(ids):
        return res
    spans = [(int(s), int(v)) for s, v in (res.version_spans or []) if s < k] or None
    return dataclasses.replace(res, completion_ids=ids[:k], logprobs=list(res.logprobs)[:k], finish="stop",
                               version=spans[-1][1] if spans else res.version, version_spans=spans if spans and len(spans) > 1 else None,
                               candidates=list(res.candidates)[:k] if res.candidates is not None else None,
                               routing=list(res.routing)[:k] if res.routing is not None else None)


class EngineLLM:
    def __init__(self, codec: ChatCodec, gen: Callable[..., Awaitable[GenResult]], *, model: str = "policy",
                 max_conversations: int = 4096, thinking: str | None = None):
        """``thinking`` (m13 C-7, A2-48): how earlier reasoning is shown, matching the deployment template:
        "preserved" (extend token for token), "interleaved" (extend inside a tool loop; a new user message re-renders the
        history, which drops reasoning: a new segment), "strip" (every call renders the history: each call its own
        segment). Default: the codec's own template behaviour; a mode the template does not implement is refused.

        A conversation is extended only when the caller's ``sample_key`` is the one that served it *and* the assistant
        turn that follows is the reply this engine produced for it (the ``crucible_reply`` nonce the policy replays from
        ``raw_message``, else the same tool calls and text). M12 audit V1: matching on the message prefix alone spliced
        one GRPO member's sampled completion into another member's next prompt."""
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
        self._convs: list[_Conv] = []
        self._media_new: list[dict[str, Any]] = []
        self._media_all: list[dict[str, Any]] = []
        self.max_conversations = max_conversations
        self.prefix_breaks = 0
        self._gen_seed, self._gen_stop = _accepts(gen, "seed"), _accepts(gen, "stop_ids")

    def _rendered(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[int]:
        ids = self.codec.render(messages, tools)
        self._media_new = list(getattr(self.codec, "last_media", None) or [])  # a whole render: offsets are already absolute
        self._media_all = list(self._media_new)
        return ids

    @staticmethod
    def _is_reply(m: dict[str, Any], c: _Conv) -> bool:
        if REPLY_KEY in m:
            return m[REPLY_KEY] == c.reply
        content = m.get("content")
        return _canon_calls(m.get("tool_calls")) == c.calls and (not c.text or (isinstance(content, str) and content.startswith(c.text)))

    def _prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, key: str | None) -> list[int]:
        self._rerendered = False
        # served conversations this call strictly extends (equal message lists are another sample of the same opening: a
        # GRPO member, a retry; rendered whole, and the served one stays for its own continuation)
        mine = [c for c in self._convs if c.key == key and len(messages) > len(c.msgs) and messages[: len(c.msgs)] == c.msgs]
        ext = [c for c in mine if messages[len(c.msgs)].get("role") == "assistant" and self._is_reply(messages[len(c.msgs)], c)]
        top = [c for c in ext if len(c.msgs) == max(len(x.msgs) for x in ext)]
        if len(top) == 1:
            c = top[0]
            self._convs.remove(c)  # the conversation moves on: its next entry replaces this one
            new = messages[len(c.msgs) + 1:]
            if self.thinking == "strip" or (self.thinking == "interleaved" and any(m.get("role") == "user" for m in new)):
                self.rerenders += 1  # the deployed template re-renders (and drops) earlier reasoning here: so do we, in a new segment
                self._rerendered = True
                return self._rendered(messages, tools)
            suf = self.codec.suffix(new, tools)
            self._media_new = [{**m, "offset": int(m["offset"]) + len(c.ids)} for m in getattr(self.codec, "last_media", None) or []]
            self._media_all = [*c.media, *self._media_new]
            return c.ids + suf
        if mine:
            # it continues a served conversation, but not through the reply (a nudge after an empty reply, a rewritten or
            # ambiguous assistant turn): rendered whole, and the caller is told it is a new segment (M12 audit V15: a
            # nudge re-rendered in the same segment was flagged prefix_break and the exact episode was masked)
            self.rerenders += 1
            self._rerendered = True
            for c in mine:
                self._convs.remove(c)
        elif any(m.get("role") == "assistant" for m in messages):
            self.prefix_breaks += 1  # mid-conversation but extends nothing served (compaction rewrote it): a new segment, rendered whole
        return self._rendered(messages, tools)

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, temperature: float = 1.0,
                       max_tokens: int = 512, sample_key: str | None = None, seed: int | None = None, **_: Any) -> LLMResponse:
        try:
            prompt = self._prompt(messages, tools, sample_key)
        except Exception as e:
            raise EngineCallError(f"chat codec: {type(e).__name__}: {e}", failure_kind="crucible:harness_unsupported") from e
        # read the per-call results of ``_prompt`` before the first await: one engine serves concurrent episodes, and another
        # episode's ``_prompt`` resets them while this call waits on ``gen`` (Tinker smoke 2026-10-20: a re-render lost its
        # new_segment flag, so the record saw a prefix break and train mode masked the episode)
        media_new, media_all, rerendered = self._media_new, self._media_all, self._rerendered
        t0 = time.monotonic()
        kw: dict[str, Any] = {}
        if media_all:
            kw["media"] = media_all
        if self._gen_seed and seed is not None:
            kw["seed"] = seed  # per call (M12 audit V12): a greedy call's seed must never pin later sampled calls
        if self._gen_stop:
            kw["stop_ids"] = list(self.codec.eos_ids)
        try:
            res = await self.gen(prompt, max_tokens, temperature, **kw)
        except asyncio.CancelledError:
            raise
        except LLMError:
            raise
        except Exception as e:
            raise EngineCallError(f"engine: {type(e).__name__}: {e}", failure_kind=getattr(e, "failure_kind", None) or "provider_unavailable") from e
        if res.finish == "abort":  # the runtime re-issues its aborts; one reaching here (a direct engine) is not a reply
            raise EngineCallError(f"engine aborted the call after {len(res.completion_ids)} tokens", failure_kind="provider_unavailable")
        res = cut_at_stop(res, self.codec.eos_ids)
        self.spend.calls += 1
        if res.finish == "length" and not res.completion_ids:
            # the prompt left no room: an engine refusal (``VLLMServerEngine`` turns vLLM's 400 into this) or a full window.
            # It was an empty reply, nudged twice and then labelled agent_done, untruncated (M2 R-1, probe tb02)
            raise ContextOverflow(f"{self.model}: finish=length with 0 completion tokens ({len(prompt)} prompt tokens, max_tokens {max_tokens})")
        comp = list(res.completion_ids)
        tail = comp if (comp and comp[-1] in self.codec.eos_ids) else comp + list(self.codec.eos_ids[:1])
        text, calls = self.codec.parse(comp)
        nonce = uuid.uuid4().hex
        self._convs.append(_Conv(sample_key, [dict(m) for m in messages], prompt + tail, media_all, nonce, text, _canon_calls(calls)))
        del self._convs[: max(0, len(self._convs) - self.max_conversations)]
        finish = "stop" if res.finish == "stop" else ("length" if res.finish == "length" else res.finish)
        return LLMResponse(text=text, tool_calls=calls, usage={"prompt_tokens": len(prompt), "completion_tokens": len(comp), "source": "engine"}, model=self.model,
                           raw_finish=finish, latency_s=time.monotonic() - t0, raw_message={REPLY_KEY: nonce},
                           tokens={"prompt_token_ids": prompt, "completion_token_ids": comp, "logprobs": list(res.logprobs),
                                   "policy_version": res.version, "version_spans": res.version_spans,
                                   **({"media": media_new} if media_new else {}), **({"new_segment": True} if rerendered else {})})


def attach_call_fields(ro: Any, model_calls: dict[str, list[dict[str, Any]]]) -> None:
    """Per-call fields ``runtime.record.build`` does not carry: the images a call's prompt added (``Turn.media``)."""
    media = {c["call_id"]: (c.get("tokens") or {}).get("media") for cs in model_calls.values() for c in cs}
    for t in ro.turns:
        if media.get(t.call_id):
            t.media = [dict(m) for m in media[t.call_id]]


def codec_capture(codec: Any) -> Any:
    """The tokenizer and chat-template arguments a chat codec renders with (R-17), or None for a codec without a
    tokenizer (scripted codecs). Computed once per rollout function: it hashes the whole vocabulary."""
    from crucible.core.schema.rollout import Capture, TokenizerId

    tok = getattr(codec, "tok", None)
    if tok is None or not hasattr(tok, "get_vocab"):
        return None
    return Capture(tokenizer=TokenizerId.of(tok), template_kwargs=dict(getattr(codec, "kw", None) or {}))


def world_rollout(task_for: Callable[[str], Any], codec: ChatCodec, *, max_steps: int = 30, max_tokens: int = 512, context: Any = None,
                  workroot: Path | None = None, limit_policy: Any = None, make_policy: Callable[[Any, float], Any] | None = None,
                  harness_id: str = "native:world@1", seats: Any = None, track_progress: bool = True) -> Callable:
    """``task_for(prompt) -> RunTask`` (M10 contract: ``open(workdir)``, ``grade(world, workdir)``). ``make_policy(llm,
    temperature)`` replaces the default ``LLMToolPolicy`` (``rlm_world_rollout``: the RLM harness).

    ``track_progress``: a task that declares ``subgoals`` records per-call ΔΦ (``Turn.advantage``) and a progress summary
    (``RolloutRecord.progress``), as the M10 native harness does; the trainer's turn credit needs it (M6 R3: without it
    ``turn_weight > 0`` trained outcome-only). False skips the per-step subgoal checks for an outcome-only run.

    ``seats`` (``train.seats.SeatConfig``): a multi-agent game. Every seat of the World's turn order plays (engine seats
    each through their own ``EngineLLM``: two seats with the same opening messages must never extend each other's
    conversation), opponents' moves reach each seat on its next observation, and the rollout returns
    ``list[RolloutRecord]``, one per trained seat (``train.seats.split_seats``; group ``<prompt>#seat=<id>``), or None
    when the episode or its grade is masked. ``train.seats.single_seat`` adapts it to the runtime."""
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.core.schema.run import RunRecord
    from crucible.runtime.record import build

    capture = codec_capture(codec)

    def policies(world: Any, gen: Callable, temperature: float) -> tuple[dict[str, Any], list[Any], list[str]]:
        if seats is None:
            llm = EngineLLM(codec, gen)
            pol = make_policy(llm, temperature) if make_policy else LLMToolPolicy("agent", llm, max_tokens=max_tokens, temperature=temperature, context=context)
            return {"agent": pol}, [], ["agent"]
        from crucible.train.seats import MoveLog, SeatView

        game = getattr(world, "turns", None)
        ids = list(game.seats) if game is not None else [a.id for a in world.spec.policy_actors]
        trained = seats.trained_seats(ids)
        log = MoveLog(ids)
        out: dict[str, Any] = {}
        for sid in ids:
            if sid in seats.opponents:
                inner = seats.opponents[sid](sid)
            else:
                llm = EngineLLM(codec, gen)
                inner = seats.make_policy(sid, llm, temperature) if seats.make_policy else \
                    LLMToolPolicy(sid, llm, max_tokens=max_tokens, temperature=temperature, context=context)
            out[sid] = SeatView(inner, sid, log, reveal=seats.reveal, show_legal=seats.show_legal)
        return out, [log], trained

    async def rollout(rid: str, prompt: str, seed: int, gen: Callable[[list[int], int, float], Awaitable[GenResult]], *, temperature: float = 1.0,
                      harness: str = harness_id):
        from crucible.core.prov import utc_now

        started_at = utc_now()
        task = task_for(prompt)
        d = Path(tempfile.mkdtemp(prefix="m12-world-", dir=workroot))
        world = await task.open(d / "work")
        try:
            pols, observers, trained = policies(world, gen, temperature)
            tracker, prog = None, d / "progress.jsonl"
            if track_progress and getattr(task, "subgoals", None) is not None:  # per-turn progress (ΔΦ), as the M10 native harness records it (M6 R3)
                from crucible.verify.progress import ProgressTracker

                tracker = ProgressTracker(task.subgoals, out=prog)
                observers = [*observers, tracker]
            try:
                ep = await run_episode(world, pols, limits=EpisodeLimits(max_steps=max_steps), reset=True, observers=observers)
            finally:
                # M14 campaign B4: a rollout cancelled mid-flight (a stopped trainer, a process ending) still closes every
                # policy's kernel and container: each close is shielded from the cancellation and run whatever the others do
                closes = [asyncio.ensure_future(pol.aclose()) for pol in pols.values() if hasattr(pol, "aclose")]
                if closes:
                    try:
                        await asyncio.shield(asyncio.gather(*closes, return_exceptions=True))
                    except asyncio.CancelledError:
                        with contextlib.suppress(BaseException):
                            await asyncio.wait_for(asyncio.gather(*closes, return_exceptions=True), 120)
                        raise
            if ep.masked:
                return None
            rep = await task.grade(world, d / "grade")
            if rep.infra.get("status") == "masked":
                return None
            reward = rep.reward
            progress = None
            if tracker is not None:  # M7 R7: a tracker that raised leaves the episode graded, its progress unavailable
                from crucible.verify.progress import observed_progress

                progress = observed_progress(ep.observed[observers.index(tracker)], prog)
            pen = limit_policy.penalty(ep.stop_condition) if limit_policy is not None else None
            if pen is not None:  # m13 C-8 (A2-24): an incomplete trajectory gets the configured penalty
                reward = pen
            from crucible.actors.rlm.agent import truncating_stop

            # an RLM policy's budget:<kind> / compaction_failed stops are truncations too (M16 §3.6)
            truncated = ep.truncated or any(truncating_stop(getattr(p, "stop", None)) for p in pols.values())
            rec = RunRecord(run_id=rid, task=getattr(task, "id", prompt), world=getattr(task, "world_id", ""), policy="engine", started_at=started_at, steps=ep.steps,
                            outcome="completed", reward=reward, attempt=f"{rid}-a1",
                            meta={"harness": harness, "stop": ep.stop_condition, "truncated": truncated, "gates": rep.gate_bits(),
                                  "gate_kinds": rep.gate_kinds(),
                                  **({"progress": progress} if progress is not None else {})})
            segs = {a: list(p.segments) for a, p in pols.items() if getattr(p, "segments", None)}
            ro = build(rec, calls=ep.model_calls, segments=segs or None, sampling={"temperature": temperature},
                       progress=prog if tracker is not None and prog.exists() else None)
            ro.group = None
            ro.capture = capture
            attach_call_fields(ro, ep.model_calls)
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
            if seats is None:
                return ro
            from crucible.core.schema.rollout import Segment
            from crucible.train.seats import game_rewards, split_seats

            outcome = game_rewards(world)
            rewards = {s: (seats.reward(world, rep, s) if seats.reward is not None else outcome.get(s)) for s in pols}
            if pen is not None:  # the limit policy's penalty replaces every trained seat's outcome, as it replaces the reward
                rewards = {s: (pen if s in trained else r) for s, r in rewards.items()}
            per_seat = {a: [Segment(index=g["index"], summary_call=g.get("summary_call"), context_before=g.get("context_before"),
                                    context_after=g.get("context_after")) for g in gs] for a, gs in segs.items()}
            return split_seats(ro, rewards=rewards, trained=trained, group=prompt, segments=per_seat)
        finally:
            await world.close()

    rollout.codec = codec  # the runtime defaults its stop ids to codec.eos_ids (M12 audit H3 V10): no decoding past end-of-turn
    return rollout


def rlm_world_rollout(task_for: Callable[[str], Any], codec: ChatCodec, config: Any = None, *, max_steps: int = 10_000, workroot: Path | None = None,
                      limit_policy: Any = None, launcher: Callable[[], Any] | None = None) -> Callable:
    """The M12 rollout under the RLM harness (m14; the M16 loop): the engine's model drives an ``RLMPolicy`` configured by
    ``RLMConfig``; root, child, sub and compaction calls carry the engine's token records; the harness label is
    ``rlm:<config digest>@1``. The ledger lives in a temporary directory the policy removes at ``aclose``.

    ``launcher``: a factory called once per rollout (M14 audit H3 V12, G1 D9: one shared instance gave concurrent rollouts
    one container, and one rollout's ``aclose`` removed it under the others); default ``RLMConfig.launcher``. Each rollout's
    kernel is seeded from its rollout id and seed (every kernel used seed 0). Model code never runs on the host here unless
    ``RLMConfig(kernel="local", allow_host=True)`` says so; without Docker the default is refused when the rollout is built."""
    import contextvars
    import shutil
    import zlib

    from crucible.actors.rlm.policy import RLMPolicy
    from crucible.runtime.harness.rlm import RLMConfig

    c = config or RLMConfig()
    if launcher is not None and (hasattr(launcher, "start") or not callable(launcher)):
        raise TypeError("rlm_world_rollout(launcher=...) takes a factory (called once per rollout), not a launcher instance")
    if launcher is None and c.kernel == "docker" and not shutil.which("docker"):
        from crucible.actors.rlm.kernel import IsolationRefused

        raise IsolationRefused("rlm_world_rollout: no isolated kernel (docker not found); pass launcher=<factory> (e.g. an ExecLauncher "
                               "into the World's sandbox) or RLMConfig(kernel='local', allow_host=True) for trusted code only")
    if launcher is None:
        c.launcher()  # refuses kernel='local' without allow_host now, not at the first rollout
    current: contextvars.ContextVar[tuple[str, int] | None] = contextvars.ContextVar("rlm_rollout", default=None)

    def make(llm: Any, temperature: float) -> Any:
        rid, seed = current.get() or ("", 0)
        factory = launcher if launcher is not None else c.launcher  # the root's kernel and every child agent's (M16 §3.3)
        return RLMPolicy("agent", llm, launcher=factory(), child_launcher=factory, temperature=temperature,
                         seed=zlib.crc32(f"{rid}:{seed}".encode()), **c.policy_kwargs())

    inner = world_rollout(task_for, codec, max_steps=max_steps, workroot=workroot, limit_policy=limit_policy, make_policy=make,
                          harness_id=f"rlm:{c.digest()}@1")

    async def rollout(rid: str, prompt: str, seed: int, gen: Callable, **kw: Any) -> Any:
        tok = current.set((rid, seed))
        try:
            return await inner(rid, prompt, seed, gen, **kw)
        finally:
            current.reset(tok)

    rollout.codec = codec
    return rollout


def tito_llm(provider: Any, model: str, *, timeout_s: float, seed: int | None = None) -> EngineLLM:
    """``LLMProvider.tito``: an ``EngineLLM`` over a vLLM server's token-id completions endpoint, rendered with the
    model's own chat template. ``actors.llm.LLMClient`` resolves this through a pointer (it may not import train).

    The seed travels with each call (``EngineLLM.complete(seed=...)``); ``seed`` here is ignored (kept for callers):
    it pinned the first call's seed for the client's life, so after one greedy call every sampled GRPO member drew the
    same completion (M12 audit V12). Conversations are keyed by the caller's ``sample_key`` (V1)."""
    from transformers import AutoTokenizer

    from crucible.core.config import resolve_secret
    from crucible.train.chat import HFChatCodec
    from crucible.train.engine import GenRequest
    from crucible.train.vllm_http import VLLMServerEngine

    codec = HFChatCodec(AutoTokenizer.from_pretrained(model), template_kwargs=provider.template_kwargs)
    eng = VLLMServerEngine(provider.base_url.rstrip("/").removesuffix("/v1"), model, timeout_s=timeout_s,
                           api_key=resolve_secret(provider.api_key) or None)
    calls = [0]

    async def gen(prompt: list[int], n: int, temp: float, *, seed: int | None = None, stop_ids: Sequence[int] = ()):
        calls[0] += 1
        return await eng.generate(GenRequest(rid="llm", call_id=f"c{calls[0]}", prompt_ids=prompt, max_tokens=n, temperature=temp,
                                             stop_ids=list(stop_ids) or codec.eos_ids, seed=seed))

    return EngineLLM(codec, gen, model=model)
