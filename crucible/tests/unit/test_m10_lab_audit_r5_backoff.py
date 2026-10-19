"""M10 lab audit R-5, item 3: a jittered, capped retry backoff that never undercuts a provider's Retry-After.

``Failure.retry_after`` is filled from the capture proxy's evidence (``agent_evidence``) and from the native LLM client
(``LLMError.retry_after`` -> ``LLMToolPolicy`` -> ``run_episode`` -> ``episode_failure``); ``RetryRule.backoff`` is the curve
``Runner`` sleeps (outside the slot) before a retry. The new ``RetryRule`` caps are operational (``identity.DENY``).

Nothing here needs a network, Docker or a paid call.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from crucible.core.schema.run import Failure
from crucible.runtime import identity as ident
from crucible.runtime.agent_evidence import UpstreamEvidence, classify_agent_end
from crucible.runtime.harness.base import HarnessResult
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.infra import RetryRule, episode_failure
from crucible.runtime.run import Runner, RunSpec


@dataclass
class _T:
    id: str = "t@1"


@dataclass
class _P:
    id: str = "p"
    estimate_usd: float = 0.0


def test_backoff_is_jittered_capped_and_zero_without_a_base():
    r = RetryRule()
    assert r.backoff(None, 1, 0.25, rand=lambda: 0.0) == 0.125  # base * 2**0 scaled by 0.5 + 0
    assert r.backoff(None, 2, 0.25, rand=lambda: 1.0) == 0.75  # base * 2**1 scaled by 0.5 + 1
    assert r.backoff(None, 30, 0.25, rand=lambda: 1.0) == r.backoff_cap_s  # the doubling and the jitter are both capped
    assert r.backoff(None, 3, 0.0) == 0.0  # a base of 0 keeps immediate retries
    # the jitter is real: the same call is not one constant
    assert len({r.backoff(None, 2, 1.0) for _ in range(20)}) > 1


def test_backoff_honours_retry_after_up_to_its_cap():
    f = Failure(kind="provider_unavailable", stage="agent", retry_after=12.0)
    assert RetryRule().backoff(f, 1, 0.25, rand=lambda: 0.0) == 12.0  # a Retry-After is a floor under the jittered wait
    assert RetryRule().backoff(f, 1, 0.0) == 12.0  # even when the run opted out of waiting
    assert RetryRule(retry_after_cap_s=5.0).backoff(f, 1, 0.25, rand=lambda: 0.0) == 5.0  # a provider asking for an hour is capped
    # a long jittered wait is not shortened by a short Retry-After
    short = Failure(kind="provider_unavailable", stage="agent", retry_after=1.0)
    assert RetryRule().backoff(short, 30, 0.25, rand=lambda: 1.0) == RetryRule().backoff_cap_s
    with pytest.raises(ValidationError):
        Failure(kind="provider_unavailable", stage="agent", retry_after=-1)
    assert "retry_after" not in Failure(kind="provider_unavailable", stage="agent").canonical_json()  # no content hash moves while None
    assert "retry_after" in f.canonical_json()


def test_retry_after_travels_from_the_proxy_evidence_and_the_episode():
    from crucible.actors.llm import _retry_after

    up = UpstreamEvidence(calls=1, last_status=429, verdict="infra", detail="HTTP 429", probe="infra", retry_after=7.0, model="m")
    end = classify_agent_end(upstream=up, error=RuntimeError("x"), agent_is_harness=True)
    assert end.failure is not None and end.failure.kind == "provider_unavailable" and end.failure.retry_after == 7.0
    no_header = classify_agent_end(upstream=UpstreamEvidence(calls=1, last_status=503, verdict="infra", detail="HTTP 503", probe="infra", model="m"),
                                   error=RuntimeError("x"), agent_is_harness=True)
    assert no_header.failure is not None and no_header.failure.retry_after is None

    assert episode_failure({"kind": "provider_unavailable", "reason": "x", "retry_after": 4.0}).retry_after == 4.0
    assert episode_failure({"kind": "provider_unavailable", "reason": "x"}).retry_after is None

    def err(value):
        return SimpleNamespace(response=SimpleNamespace(headers={"retry-after": value}))

    assert _retry_after(err("3")) == 3.0
    assert _retry_after(err("soon")) is None
    assert _retry_after(err("nan")) is None
    assert _retry_after(err("-2")) is None
    assert _retry_after(RuntimeError("no response")) is None


async def test_the_policy_keeps_the_retry_after_of_the_llm_error_that_ended_the_episode():
    from crucible.actors.llm import LLMError, _retry_delay
    from crucible.actors.policy import LLMToolPolicy
    from crucible.core.schema.actions import Done, Observation

    class FakeLLM:
        model = "fake"

        async def complete(self, messages, **kw):
            err = LLMError("fake failed after 6 attempts: HTTP 429")
            err.retry_after = 9.0  # what ``LLMClient._complete_paid`` sets from the last answer's header
            raise err

    p = LLMToolPolicy("agent", FakeLLM())
    assert p.retry_after is None
    a = await p.act(Observation(actor="agent", step=0, instruction="go"))  # the error ends the episode, it does not raise
    assert isinstance(a, Done) and p.failure_kind == "provider_unavailable" and p.retry_after == 9.0
    # the client's own wait honours the same header, capped
    e = SimpleNamespace(status_code=429, response=SimpleNamespace(headers={"retry-after": "500"}))
    assert _retry_delay(e, 0, cap=60.0) == 60.0
    assert _retry_delay(SimpleNamespace(status_code=429, response=SimpleNamespace(headers={"retry-after": "4"})), 0) == 4.0


async def test_the_runner_waits_a_retry_after_before_the_retry(tmp_path):
    class RA:
        id = "ra@1"

        async def run(self, attempt):
            if attempt.n == 1:
                return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="503", retry_after=0.3))
            return HarnessResult(reward=1.0, steps=1)

    s = await Runner(tmp_path).run(RunSpec(run_id="ra", tasks=[_T()], policies=[_P()], harnesses=[RA()], k=1, retry_backoff_s=0.0))
    assert s.scored == 1 and s.attempts == 2
    events = [json.loads(line) for line in (tmp_path / "ra" / "journal.jsonl").read_text().splitlines() if line.strip()]
    first_finished = next(e for e in events if e["event"] == "finished")
    dispatched = [e for e in events if e["event"] == "dispatched"]
    assert len(dispatched) == 2
    assert dispatched[1]["ts"] - first_finished["ts"] >= 0.25  # the Retry-After (0.3 s) floors the wait; the base is 0


async def test_the_native_client_reads_a_429s_retry_after_into_its_error(tmp_path, monkeypatch):
    """``LLMClient._complete_paid`` sets ``err.retry_after`` from the last answer's header (llm.py: the first link of the chain a native
    429 takes to ``Failure.retry_after``); the fake client raises what the openai SDK does: an exception with ``status_code`` and a
    ``response.headers``."""
    from crucible.actors import llm as m
    from crucible.actors.llm import LLMClient, LLMError

    class RateLimited(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "3"})

    class Fake:
        chat = property(lambda self: self)
        completions = property(lambda self: self)

        async def create(self, **kw):
            raise RateLimited("slow down")

    async def no_sleep(s):
        return None

    monkeypatch.setattr(m.asyncio, "sleep", no_sleep)
    c = LLMClient("policy-default", budget_usd=1, cache_mode="off", cache_dir=tmp_path)
    c._client = Fake()
    with pytest.raises(LLMError) as ei:
        await c.complete([{"role": "user", "content": "q"}])
    assert ei.value.retry_after == 3.0


async def test_a_native_retry_after_reaches_the_masked_record(tmp_path):
    """The second and third links: ``LLMToolPolicy`` keeps the error's Retry-After (``p.retry_after``), ``run_episode`` puts it in the
    episode's failure dict and ``episode_failure`` into the ``Failure`` the record carries (the provider breaker's cool-down and the
    retry backoff read it there). One attempt (``max_attempts=1``) and no breaker: the masked row is the evidence."""
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.actors.llm import LLMError
    from crucible.actors.policy import LLMToolPolicy
    from crucible.runtime.inject import FakeLLM, tool_reply

    err = LLMError("fake failed after 6 attempts: HTTP 429")
    err.retry_after = 9.0

    def make(attempt):
        return {"agent": LLMToolPolicy("agent", FakeLLM([tool_reply("tickets__list_tickets", {"status": "open"}, 0)], fail_at=(0, err)))}

    pol = PolicyRef(id="fake:429", make=make, spend=lambda: (0.0, 0.0), estimate_usd=0.001, config={"name": "429"})
    spec = RunSpec(run_id="ra429", tasks=[HelpdeskTask()], policies=[pol], harnesses=[NativeHarness()], k=1, retry=RetryRule(max_attempts=1), breaker=None)
    s = await Runner(tmp_path).run(spec)
    assert s.masked_terminal == 1, s
    rows = [json.loads(line) for line in (tmp_path / "ra429" / "failures.jsonl").read_text().splitlines() if line.strip()]
    assert rows[0]["failure"]["kind"] == "provider_unavailable" and rows[0]["failure"]["retry_after"] == 9.0, rows[0]["failure"]


async def test_a_retry_after_longer_than_the_wall_clock_never_holds_the_run(tmp_path):
    """A 429 with ``Retry-After: 60`` at the start of a 0.3 s wall clock: the backoff sleeps at most until the wall clock, the sample
    sees the stop and the run returns (it held its lease for the whole Retry-After, up to 300 s, before)."""

    class RA:
        id = "ra@1"

        async def run(self, attempt):
            if attempt.n == 1:
                return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="429", retry_after=60.0))
            return HarnessResult(reward=1.0, steps=1)

    spec = RunSpec(run_id="rawall", tasks=[_T()], policies=[_P()], harnesses=[RA()], k=1, retry_backoff_s=0.0, wall_clock_s=0.3, breaker=None)
    t0 = time.monotonic()
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 30)
    assert time.monotonic() - t0 < 10  # not the 60 s Retry-After
    assert s.stopped == "wall_clock" and s.scored == 0 and s.attempts == 1


async def test_a_stop_rule_wakes_a_sample_in_a_retry_backoff(tmp_path):
    """Another sample's terminal masks trip the health rule while ``slow`` sleeps its 60 s Retry-After: the stop wakes it, and the run
    ends at once with the reason (the sleep was a plain ``asyncio.sleep`` nothing woke)."""

    class H:
        id = "h@1"

        async def run(self, attempt):
            if attempt.task.id == "slow":
                return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="429", retry_after=60.0))
            if attempt.task.id == "bad":
                return HarnessResult(failure=Failure(kind="verifier_error", stage="verification", blame="task", retryable=False, reason="grader broke"))
            return HarnessResult(reward=1.0, steps=1)

    # plan order slow, bad, c at concurrency 2: slow fails and sleeps, bad ends terminal-masked (1/1 > 0.2 at min_terminal 1), and the
    # third sample's dispatch reads the stop
    spec = RunSpec(run_id="rastop", tasks=[_T(id="slow"), _T(id="bad"), _T(id="c")], policies=[_P()], harnesses=[H()], k=1, concurrency=2, retry_backoff_s=0.0,
                   min_terminal_for_health=1, max_mask_fraction=0.2, breaker=None)
    t0 = time.monotonic()
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 30)
    assert time.monotonic() - t0 < 10
    assert s.stopped is not None and s.stopped.startswith("health"), s.stopped


def test_new_retry_fields_are_operational():
    def spec(**kw) -> RunSpec:
        return RunSpec(run_id="r", tasks=[SimpleNamespace(id="t")], policies=[PolicyRef(id="p", make=lambda a: {}, config={})],
                       harnesses=[NativeHarness()], k=1, **kw)

    a = ident.fingerprint(spec().resolved_doc())
    assert a == ident.fingerprint(spec(retry=RetryRule(backoff_cap_s=1.0, retry_after_cap_s=2.0)).resolved_doc())  # how long to wait is not what is measured
    assert spec().resolved_doc()["retry"]["backoff_cap_s"] == 30.0  # but it is recorded
    assert a != ident.fingerprint(spec(retry=RetryRule(max_attempts=5)).resolved_doc())  # the attempts a sample gets still are
