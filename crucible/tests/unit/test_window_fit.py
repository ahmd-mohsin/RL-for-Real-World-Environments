"""M2 R-1 (audit 2026-10-15): one window rule at every model-call boundary. On the engine path a refused prompt came back
as an empty ``length`` reply, was nudged twice and ended ``agent_done``, untruncated (probe tb02); and ``max_tokens`` was
never fitted to the room left, so a prompt with room for a reply was refused (probe tc01)."""

from __future__ import annotations

from pathlib import Path

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import WINDOW_MARGIN, EpisodeLimits, LLMToolPolicy, run_episode
from crucible.core.schema.actions import Done, Observation
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.train.chat import ByteChatCodec
from crucible.train.engine import GenResult
from crucible.train.world_env import EngineLLM
from crucible.world.world import World


def _world(tmp: Path) -> World:
    spec = WorldSpec.model_validate({"world": "t/window", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})
    return World(spec, instructions={"agent": "Say you are done."}, store=Store(tmp / "s"), workdir=tmp / "w")


async def test_an_engine_refusal_ends_context_overflow_truncated_without_nudges(tmp_path):
    seen: list[tuple[int, int]] = []

    async def gen(prompt, n, temp, **kw):  # a 400-token engine: VLLMServerEngine turns its 400 into "length" with no tokens
        seen.append((len(prompt), n))
        if len(prompt) + n > 400:
            return GenResult([], [], 0, "length", prompt_tokens=len(prompt))
        return GenResult([*b"all done", 256], [-0.1] * 9, 0, "stop", prompt_tokens=len(prompt))

    w = _world(tmp_path)
    await w.start()
    try:
        pol = LLMToolPolicy("agent", EngineLLM(ByteChatCodec(), gen), max_tokens=300)
        res = await run_episode(w, {"agent": pol}, limits=EpisodeLimits(max_steps=5))
    finally:
        await w.close()
    assert (res.stop_condition, res.truncated, res.masked) == ("context_overflow", True, False)  # was agent_done, untruncated
    assert len(seen) == 1 and not any(m.get("content", "").startswith("Your last reply") for m in pol.messages)  # no nudges


class Windowed:
    """A model whose client knows its window; records the max_tokens of every call."""

    model = "windowed"

    def __init__(self, window: int | None):
        self.window, self.sent = window, []

    async def context_window(self):
        return self.window

    async def complete(self, messages, *, tools=None, max_tokens=0, **kw):
        self.sent.append(max_tokens)
        return LLMResponse(text="done", raw_finish="stop")


async def test_a_reply_is_fitted_to_the_room_left_and_both_values_are_recorded():
    llm = Windowed(1_000)
    pol = LLMToolPolicy("agent", llm, max_tokens=8_000)  # type: ignore[arg-type]
    obs = Observation(actor="agent", step=0, instruction="x" * 2_400)
    used = None

    async def spy():
        nonlocal used
        used = pol.context_tokens()
        return 1_000

    llm.context_window = spy  # type: ignore[method-assign]
    assert isinstance(await pol.act(obs), Done)
    assert llm.sent == [1_000 - used - int(1_000 * WINDOW_MARGIN)]  # was 8,000: refused with room left
    call = pol.model_calls[-1]
    assert call["max_tokens_requested"] == 8_000

    roomy = Windowed(100_000)
    pol = LLMToolPolicy("agent", roomy, max_tokens=8_000)  # type: ignore[arg-type]
    await pol.act(Observation(actor="agent", step=0, instruction="hi"))
    assert roomy.sent == [8_000] and "max_tokens_requested" not in pol.model_calls[-1]
    blind = Windowed(None)
    pol = LLMToolPolicy("agent", blind, max_tokens=8_000)  # type: ignore[arg-type]
    await pol.act(Observation(actor="agent", step=0, instruction="x" * 2_400))
    assert blind.sent == [8_000]  # an unknown window: as before


async def test_no_room_at_all_ends_the_episode_before_the_call_is_paid_for():
    llm = Windowed(500)
    pol = LLMToolPolicy("agent", llm)  # type: ignore[arg-type]
    done = await pol.act(Observation(actor="agent", step=0, instruction="x" * 4_000))
    assert isinstance(done, Done) and pol.stop == "context_overflow" and llm.sent == []
