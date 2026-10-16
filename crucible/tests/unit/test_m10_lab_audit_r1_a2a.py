"""M10 lab audit R-1 (2026-10-15), slice 2: the A2A harness decides blame from evidence, not from the agent's word.

- An agent's ``failed``/``agent_error`` after the proxy saw the provider fail is ``provider_unavailable`` (masked); with a
  healthy proxy it is the policy's outcome (scored); an unconfirmed ``infra_error``/``canceled`` is ``unknown`` blame.
- An agent that stops answering polls (agentenv ``AgentUnreachableError``, raised when ``poll_a2a_task`` gets a
  ``sandbox_id``) is masked ``transport_unreachable`` instead of scored as a timeout after ``timeout_s``.
- The capture key is dead before verification; spend is settled from the captured usage (``budget_usd`` trips); a
  limit stop follows ``limit_policy``.

Part A runs without agentenv: a fake ``agent_env.a2a_agent.protocol`` in ``sys.modules``, the fake agent's ``/ext``
endpoints behind ``httpx.MockTransport``, the real ``CaptureProxy`` in front of a fake upstream, and the real ``Runner`` for
the budget stop. Part B drives real
agentenv A2A agents (``pytest.importorskip("agent_env")``), as ``test_a2a_harness.py`` does.

No ``from __future__ import annotations`` here: agentenv resolves ``run(self, request: TaskRequest)`` of the agent class
defined inside ``_sleep_agent`` from its annotation object, which a postponed (string) annotation would hide.
"""

import asyncio
import dataclasses
import functools
import inspect
import json
import sys
import time
import types
from types import SimpleNamespace

import httpx
import pytest

import crucible.interop.agentenv.harness as hmod
from crucible.actors.llm import UNPRICED_USD_PER_M
from crucible.connectors.capture_proxy import CaptureProxy, Upstream
from crucible.core.schema.run import Failure
from crucible.interop.agentenv.harness import A2AHarness
from crucible.runtime.agent_evidence import UpstreamEvidence
from crucible.runtime.harness.base import Attempt
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.infra import LimitPolicy, RetryRule
from crucible.runtime.run import Runner, RunSpec
from crucible.verify.runtime.plan import GradeReport

KEY = "sk-ant-api03-" + "Q" * 40  # credential-shaped: an agent echoing it must not reach a stored reason
PRICED = {"m": [1_000_000.0, 1_000_000.0]}  # 3 prompt + 2 completion tokens -> 5.0 USD per call
COMPLETED = {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": "done"}, {"kind": "data", "data": {"tool_call_count": 2}}]}}}


# ------------------------------------------------------------------ part A: fakes (no agentenv)


@pytest.fixture
def proto(monkeypatch):
    """A fake ``agent_env.a2a_agent.protocol`` with agentenv's liveness surface (protocol.py:21-32, 145-204): the
    harness imports it at attempt time, so a test scripts ``poll_a2a_task`` through ``p.script["poll"]``."""
    p = types.ModuleType("agent_env.a2a_agent.protocol")

    class AgentUnreachableError(TimeoutError):
        pass

    @dataclasses.dataclass(frozen=True)
    class TerminalResponse:
        response_text: str = ""
        tool_call_count: int | None = None
        error_type: str | None = None
        error_code: str | None = None
        error_message: str | None = None

        @classmethod
        def from_message(cls, message: dict) -> "TerminalResponse":
            text, data = "", {}
            for part in (message or {}).get("parts", []):
                if part.get("kind") == "text":
                    text = part["text"]
                elif part.get("kind") == "data":
                    data.update(part.get("data") or {})
            return cls(response_text=text, **{k: data.get(k) for k in ("tool_call_count", "error_type", "error_code", "error_message")})

    p.calls = {"send": [], "poll": [], "cancel": []}
    p.script = {"poll": None}

    async def send_a2a_message(a2a_url, parts, message_id, context_id, timeout_seconds):
        p.calls["send"].append(a2a_url)
        return "t1", None

    async def poll_a2a_task(a2a_url, task_id, timeout_seconds, poll_interval_seconds=10, *, sandbox_id=None):
        p.calls["poll"].append({"url": a2a_url, "task_id": task_id, "timeout": timeout_seconds, "sandbox_id": sandbox_id})
        return await p.script["poll"]()

    async def cancel_a2a_task(a2a_url, task_id):
        p.calls["cancel"].append(task_id)

    p.AgentUnreachableError, p.TerminalResponse = AgentUnreachableError, TerminalResponse
    p.send_a2a_message, p.poll_a2a_task, p.cancel_a2a_task = send_a2a_message, poll_a2a_task, cancel_a2a_task
    p.UNREACHABLE_AFTER_SECONDS = 60
    pkg, sub = types.ModuleType("agent_env"), types.ModuleType("agent_env.a2a_agent")
    pkg.a2a_agent, sub.protocol = sub, p
    for name, mod in (("agent_env", pkg), ("agent_env.a2a_agent", sub), ("agent_env.a2a_agent.protocol", p)):
        monkeypatch.setitem(sys.modules, name, mod)
    return p


class FakeWorld:
    def __init__(self):
        self.backend = SimpleNamespace(agent_mcp_url=self._mcp)
        self.closed = False

    async def _mcp(self, actor):
        return "http://world/mcp"

    def stream_trace(self, path):
        pass

    def to_trace(self):
        return SimpleNamespace(model_dump_json=lambda: "{}")

    async def close(self):
        self.closed = True


class FakeTask:
    id, world_id, instruction = "fake/t", "fake", "do it"

    def __init__(self, reward: float = 1.0, on_grade=None):
        self.reward, self.on_grade, self.graded = reward, on_grade, 0

    async def open(self, workdir):
        return FakeWorld()

    async def grade(self, world, workdir):
        self.graded += 1
        if self.on_grade is not None:
            await self.on_grade()
        return GradeReport(task=self.id, tier="full", checks=[], gates={"g": "pass"}, reward=self.reward, outcome=self.reward,
                           infra={"status": "ok"}, trusted=True, elapsed_s=0.0)


class Env:
    """The real capture proxy in front of a fake upstream, and a fake A2A agent's ``/ext`` endpoints: the harness module's
    ``httpx.AsyncClient`` is pointed at them. ``model_call()`` is the agent calling its model with the key it was
    configured with (agent-config)."""

    def __init__(self, tmp_path, monkeypatch, status: int = 200):
        self.tmp, self.status, self.hits = tmp_path, status, 0
        self.holder: dict = {}
        self.proxy = CaptureProxy(tmp_path / "cap", openai=Upstream(base="http://upstream.invalid/v1", key="sk-upstream"),
                                  transport=httpx.MockTransport(self._upstream))
        self.app = self.proxy.app()
        real, agent = httpx.AsyncClient, httpx.MockTransport(self._agent)
        monkeypatch.setattr(hmod, "httpx", SimpleNamespace(AsyncClient=lambda **kw: real(transport=agent, **kw), HTTPError=httpx.HTTPError))

    def _upstream(self, req: httpx.Request) -> httpx.Response:
        self.hits += 1
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json={"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    async def model_call(self) -> int:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://proxy") as c:
            r = await c.post("/v1/chat/completions", headers={"authorization": f"Bearer {self.holder['key']}"},
                             json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        return r.status_code

    async def _agent(self, req: httpx.Request) -> httpx.Response:
        self.holder.setdefault("asked", []).append(req.url.path)
        if req.url.path == "/ext/mcp-config":
            return httpx.Response(200, json={"mcp_servers": {}} if req.method == "GET" else {})
        if req.url.path == "/ext/agent-config":
            self.holder["key"] = json.loads(req.content)["llm_api_key"]
            if self.holder.get("call_on_config"):
                self.holder["config_call"] = await self.model_call()
            return httpx.Response(self.holder.get("config_status", 200), json={})
        if self.holder.get("trajectory_down"):
            raise httpx.ConnectError("connection refused", request=req)
        return httpx.Response(404, json={})  # no trajectory/v1

    def harness(self, **kw) -> A2AHarness:
        kw.setdefault("prices", PRICED)
        return A2AHarness(agent_url="http://agent", capture=self.proxy, capture_base_url="http://proxy",
                          agent_config={"model": "m", "llm_base_url": "${capture.base_url}/v1", "llm_api_key": "${capture.key}"}, **kw)

    async def run(self, harness: A2AHarness, task, n: int = 1):
        adir = self.tmp / f"a{n}"
        adir.mkdir(exist_ok=True)
        return await harness.run(Attempt(sample=None, id=f"s-a{n}", n=n, dir=adir, task=task, policy=None))  # type: ignore[arg-type]


def term(error_type=None, message="model refused", code=None):
    return SimpleNamespace(error_type=error_type, error_code=code, error_message=message, response_text=None, tool_call_count=None)


async def test_ask_catches_unreachable_before_timeout(proto):
    async def poll():
        raise proto.AgentUnreachableError("The agent on sandbox sbx stopped answering: 3 polls in a row over 61s got no reply")

    proto.script["poll"] = poll
    h = A2AHarness(agent_url="http://a", capture=None)
    task_id, state, got, why = await h._ask("http://a", "x", "sbx")
    assert (task_id, state, got) == ("t1", "unreachable", None) and "3 polls" in why
    assert proto.calls["poll"][0]["sandbox_id"] == "sbx"
    assert proto.calls["cancel"] == []  # poll_a2a_task cancelled it already (protocol.py:197-198): never twice
    assert h._liveness_supported()


async def test_ask_without_liveness_support_still_times_out(proto):
    seen = []

    async def poll_a2a_task(a2a_url, task_id, timeout_seconds, poll_interval_seconds=10):  # an agentenv before liveness
        seen.append(a2a_url)
        raise TimeoutError(f"A2A task {task_id} did not complete within {timeout_seconds}s")

    proto.poll_a2a_task = poll_a2a_task
    del proto.AgentUnreachableError
    h = A2AHarness(agent_url="http://a", capture=None)
    task_id, state, got, why = await h._ask("http://a", "x", "sbx")  # sandbox_id is dropped, never passed to an old poll
    assert (state, got, why) == ("timeout", None, None) and seen == ["http://a"]
    assert proto.calls["cancel"] == ["t1"] and not h._liveness_supported()


def test_a2a_outcome_table():
    h = A2AHarness(agent_url="http://a", capture=None)
    # M10 lab audit R-1: a verdict that blames the upstream counts only with Crucible's probe agreeing (probe=...)
    infra = UpstreamEvidence(calls=1, last_status=503, verdict="infra", detail="HTTP 503", probe="infra")
    ok = UpstreamEvidence(calls=1, last_status=200, verdict="ok", detail="HTTP 200")
    bad_request = UpstreamEvidence(calls=1, last_status=400, verdict="client_error", detail="HTTP 400")

    end = h._outcome("failed", term("agent_error"), None, infra)  # the proxy saw the provider fail: not the policy's 0
    assert end.failure.kind == "provider_unavailable" and end.failure.masks and "model refused" in end.failure.reason
    end = h._outcome("failed", term("agent_error"), None, bad_request)
    assert end.failure.who == "policy" and not end.failure.masks and "model refused" in end.failure.reason
    end = h._outcome("failed", term(None), None, UpstreamEvidence())  # no model call at all (test_a2a_harness.py:125)
    assert end.failure.kind == "agent_run_error" and end.failure.who == "policy" and not end.failure.masks
    end = h._outcome("failed", term("infra_error"), None, ok)  # an infra claim the proxy does not confirm
    assert end.failure.who == "unknown" and end.failure.masks
    assert h._outcome("failed", term("infra_error"), None, infra).failure.kind == "provider_unavailable"
    for state in ("canceled", "rejected"):
        end = h._outcome(state, term(None, message=None), None, ok)
        assert end.failure.who == "unknown" and end.failure.masks, state
    end = h._outcome("unreachable", None, "3 polls in a row got no reply", UpstreamEvidence())
    assert end.failure.kind == "transport_unreachable" and end.failure.masks and "3 polls" in end.failure.reason
    end = h._outcome("timeout", None, None, ok)
    assert end.stop == "timeout" and end.failure is None and end.evidenced
    # Crucible's deadline during a provider outage is the provider's (critic amendment 7), not the policy's timeout
    assert h._outcome("timeout", None, None, infra).failure.kind == "provider_unavailable"
    assert h._outcome("timeout", None, None, UpstreamEvidence(calls=1, last_status=402, verdict="credits", probe="credits")).failure.kind == "crucible:credits_exhausted"
    assert h._outcome("timeout", None, None, UpstreamEvidence(calls=1, last_status=401, verdict="auth", probe="auth")).failure.kind == "crucible:harness_unsupported"
    # the agent is the policy and writes its own requests: a 402 Crucible's probe does not see is unknown, never a run stop
    end = h._outcome("failed", term("agent_error"), None, UpstreamEvidence(calls=1, last_status=402, verdict="credits", probe="ok"))
    assert end.failure.who == "unknown" and end.failure.detail == "unconfirmed:credits" and end.failure.kind != "crucible:credits_exhausted"
    end = h._outcome("completed", term(None, message=None), None, ok)
    assert end.stop == "agent_done" and end.failure is None


def test_echoed_key_in_the_agent_error_is_scrubbed():
    h = A2AHarness(agent_url="http://a", capture=None)
    end = h._outcome("failed", term("agent_error", message=f"upstream rejected key {KEY}: invalid"), None, UpstreamEvidence())
    assert KEY not in end.failure.reason and "sk-ant-api03" not in end.failure.reason
    assert "upstream rejected key" in end.failure.reason
    long = h._outcome("failed", term("agent_error", message="x" * 5000), None, UpstreamEvidence())
    assert len(long.failure.reason) <= 500 and "unconfirmed claim" in long.failure.reason  # the evidence part is not cut off


async def test_key_is_revoked_before_verification_and_the_grade_keeps_the_evidence(tmp_path, monkeypatch, proto):
    env = Env(tmp_path, monkeypatch)
    seen: dict = {}

    async def poll():
        seen["agent_phase"] = await env.model_call()
        return COMPLETED

    async def on_grade():
        seen["verification"] = await env.model_call()
        seen["live"] = dict(env.proxy.sessions)

    proto.script["poll"] = poll
    res = await env.run(env.harness(), FakeTask(on_grade=on_grade))
    assert res.failure is None and res.reward == 1.0 and res.stop == "agent_done" and res.steps == 2
    assert seen["agent_phase"] == 200 and seen["verification"] == 401 and seen["live"] == {}
    assert env.hits == 1  # the call made during verification never reached the upstream
    # merged into the grade, never replaced by it (critic amendment 4): the record carries the evidence
    assert res.grade["reward"] == 1.0 and res.grade["agent"]["state"] == "completed" and res.grade["agent"]["upstream"]["verdict"] == "ok"
    assert res.grade["agent"]["liveness"] == "sandbox_id" and proto.calls["poll"][0]["sandbox_id"] == "http://agent"  # default: the agent's URL
    assert env.proxy.sessions == {}


@pytest.mark.parametrize("kw, expect", [
    ({"prices": PRICED}, (5.0, 0.0)),
    ({"prices": {}}, (0.0, 3 / 1e6 * UNPRICED_USD_PER_M[0] + 2 / 1e6 * UNPRICED_USD_PER_M[1])),  # unpriced hosted model: uncertain
    ({"prices": {"*": [2_000_000.0, 0.0]}}, (0.0, 6.0)),
    ({"prices": {}, "policy_version": 3}, (0.0, 0.0)),  # the agent's model server is the trainer's: free
    ({"prices": {}, "self_hosted": True}, (0.0, 0.0)),
], ids=["priced", "unpriced", "star", "policy-version", "self-hosted"])
async def test_spend_settles_from_the_capture(tmp_path, monkeypatch, proto, kw, expect):
    env = Env(tmp_path, monkeypatch)

    async def poll():
        await env.model_call()
        return COMPLETED

    proto.script["poll"] = poll
    res = await env.run(env.harness(**kw), FakeTask())
    assert res.failure is None
    assert (res.cost_usd, res.cost_uncertain_usd) == (pytest.approx(expect[0]), pytest.approx(expect[1]))


async def test_spend_settles_when_the_attempt_fails_before_the_prompt(tmp_path, monkeypatch, proto):
    """Critic amendment 11: the log path is known when the session is, so an agent-config refused after a model call
    still settles what the proxy forwarded."""
    env = Env(tmp_path, monkeypatch)
    env.holder.update(call_on_config=True, config_status=400)
    res = await env.run(env.harness(), FakeTask())
    assert res.failure is not None and res.failure.kind == "agent_run_error" and "agent-config refused" in res.failure.reason
    assert env.holder["config_call"] == 200 and res.cost_usd == pytest.approx(5.0)
    assert proto.calls["poll"] == [] and env.proxy.sessions == {}


async def test_spend_settles_into_the_runner_ledger_and_trips_the_budget(tmp_path, monkeypatch, proto):
    """Acceptance without agentenv (M10 lab audit R-1): each attempt's captured usage is priced (5.0 USD) and settled by
    the Runner's ledger, so a 1.0 USD budget stops the run after one attempt. Part B's twin needs agentenv installed."""
    env = Env(tmp_path, monkeypatch)

    async def poll():
        assert await env.model_call() == 200
        return COMPLETED

    proto.script["poll"] = poll
    s = await Runner(tmp_path / "runs").run(RunSpec(run_id="budget", tasks=[FakeTask()], policies=[PolicyRef(id="a2a:fake", make=lambda a: {})],
                                                    harnesses=[env.harness()], k=3, concurrency=1, budget_usd=1.0, retry=RetryRule(max_attempts=1)))
    rows = [json.loads(x) for x in (tmp_path / "runs" / "budget" / "results.jsonl").read_text().splitlines() if x.strip()]
    assert s.stopped and s.stopped.startswith("budget"), s.stopped
    assert s.attempts == 1 and s.spend["spent_run"] == pytest.approx(5.0) and env.hits == 1
    assert len(rows) == 1 and rows[0]["cost_usd"] == pytest.approx(5.0) and not rows[0]["mask_sample"], rows[0].get("failure")
    assert env.proxy.sessions == {}


async def test_identical_model_calls_each_settle(tmp_path, monkeypatch, proto):
    """M10 lab audit R-1: two identical calls collapse into one exchange for training, but both were forwarded and billed."""
    env = Env(tmp_path, monkeypatch)

    async def poll():
        assert await env.model_call() == 200
        assert await env.model_call() == 200
        return COMPLETED

    proto.script["poll"] = poll
    res = await env.run(env.harness(), FakeTask())
    assert res.failure is None and env.hits == 2 and res.cost_usd == pytest.approx(10.0)


@pytest.mark.parametrize("policy, outcome", [(None, "scored"), (LimitPolicy(on_timeout="mask"), "masked"), (LimitPolicy(on_timeout="penalty:-0.5"), "penalty")])
async def test_deadline_follows_the_limit_policy(tmp_path, monkeypatch, proto, policy, outcome):
    env = Env(tmp_path, monkeypatch)

    async def poll():
        raise TimeoutError("A2A task t1 did not complete within 600s")

    proto.script["poll"] = poll
    env.holder["trajectory_down"] = True  # the trajectory fetch is best effort: its failure never decides the attempt
    task = FakeTask(reward=0.25)
    res = await env.run(env.harness(limit_policy=policy), task)
    assert proto.calls["cancel"] == ["t1"] and res.grade["agent"]["stop"] == "timeout"
    if outcome == "masked":
        assert res.failure.kind == "agent_timeout" and res.failure.masks and res.reward is None and task.graded == 0
    elif outcome == "scored":
        assert res.failure is None and res.stop == "timeout" and res.truncated and res.reward == 0.25
    else:
        assert res.failure is None and res.reward == -0.5 and res.grade["limit_penalty"] == -0.5


async def test_deadline_during_a_provider_outage_is_masked_provider_unavailable(tmp_path, monkeypatch, proto):
    env = Env(tmp_path, monkeypatch, status=503)

    async def poll():
        assert await env.model_call() == 503  # the agent kept retrying a failing provider until Crucible's deadline
        raise TimeoutError("A2A task t1 did not complete within 600s")

    proto.script["poll"] = poll
    task = FakeTask()
    res = await env.run(env.harness(), task)
    assert res.failure.kind == "provider_unavailable" and res.failure.masks and "HTTP 503" in res.failure.reason and task.graded == 0
    assert res.grade["agent"]["upstream"]["last_status"] == 503


async def test_unreachable_agent_is_masked_transport_unreachable(tmp_path, monkeypatch, proto):
    env = Env(tmp_path, monkeypatch)

    async def poll():
        raise proto.AgentUnreachableError("The agent on sandbox sbx-1 stopped answering: 3 polls in a row over 61s got no reply")

    proto.script["poll"] = poll
    task = FakeTask()
    res = await env.run(env.harness(sandbox_id="sbx-1"), task)
    assert res.failure.kind == "transport_unreachable" and res.failure.masks and res.failure.detail == "agent_unreachable"
    assert proto.calls["poll"][0]["sandbox_id"] == "sbx-1" and proto.calls["cancel"] == [] and task.graded == 0
    assert res.grade["agent"]["state"] == "unreachable" and env.proxy.sessions == {}
    assert "/ext/trajectory" not in env.holder["asked"]  # a dead agent is not asked for its trajectory (that raised ConnectError)


async def test_agent_error_after_upstream_5xx_is_masked_and_echoed_key_scrubbed(tmp_path, monkeypatch, proto):
    env = Env(tmp_path, monkeypatch, status=503)

    async def poll():
        await env.model_call()
        return {"status": {"state": "failed", "message": {"parts": [{"kind": "data", "data": {"error_type": "agent_error", "error_code": "llm",
                                                                                                    "error_message": f"provider said no to {KEY}"}}]}}}

    proto.script["poll"] = poll
    res = await env.run(env.harness(), FakeTask())
    assert res.failure.kind == "provider_unavailable" and res.failure.masks and res.failure.who != "policy"
    assert KEY not in res.failure.reason and "provider said no" in res.failure.reason
    assert res.cost_usd == 0.0  # the failed call carried no usage: nothing billed


# ------------------------------------------------------------------ review round 2: the probe's dialect is configured


FAILED = {"status": {"state": "failed", "message": {"parts": [{"kind": "data", "data": {"error_type": "agent_error"}}]}}}


def _upstreams(env: Env, *, anthropic: bool) -> list[tuple[str, str | None]]:
    """Replace the rig's proxy with one whose OpenAI upstream serves chat completions (200) but not Responses (404) and,
    with ``anthropic``, an Anthropic upstream that answers 404 not_found_error for any model. Returns every upstream hit."""
    seen: list[tuple[str, str | None]] = []

    def upstream(req: httpx.Request) -> httpx.Response:
        seen.append((f"{req.url.host}{req.url.path}", json.loads(req.content or b"{}").get("model")))
        if req.url.path.endswith("/messages"):
            return httpx.Response(404, json={"type": "error", "error": {"type": "not_found_error", "message": "model not found"}})
        if req.url.path.endswith("/responses"):
            return httpx.Response(404, json={"error": {"message": "Not Found"}})
        return httpx.Response(200, json={"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    extra = {"anthropic": Upstream(base="http://anthropic.invalid/v1", key="sk-ant-upstream")} if anthropic else {}
    env.proxy = CaptureProxy(env.tmp / "cap2", openai=Upstream(base="http://upstream.invalid/v1", key="sk-upstream"), transport=httpx.MockTransport(upstream),
                             **extra)
    env.app = env.proxy.app()
    return seen


async def _agent_post(env: Env, path: str, body: dict) -> int:
    """The A2A agent (the policy) calling its model on a route it picked, with the key it was configured with."""
    h = {"x-api-key": env.holder["key"], "anthropic-version": "2023-06-01"} if path == "/v1/messages" else {"authorization": f"Bearer {env.holder['key']}"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env.app), base_url="http://proxy") as c:
        return (await c.post(path, json=body, headers=h)).status_code


@pytest.mark.parametrize("api", [None, "openai"])
async def test_a_policy_made_404_on_another_dialect_is_never_confirmed(tmp_path, monkeypatch, proto, api):
    """M10 lab audit R-1 review round 2: the probe's dialect came from the session's first request, which an A2A agent
    (the policy) writes: it called /v1/messages with any model, Anthropic answered 404, Crucible probed Anthropic with the
    configured model, got 404 and masked the failure as ``crucible:harness_unsupported`` (not counted by the 5% stop).
    The dialect is now configured (``api``); with several upstreams and none configured, the verdict stays unconfirmed."""
    env = Env(tmp_path, monkeypatch)
    seen = _upstreams(env, anthropic=True)

    async def poll():
        assert await _agent_post(env, "/v1/messages", {"model": "whatever", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}) == 404
        return FAILED

    proto.script["poll"] = poll
    res = await env.run(env.harness(**({"api": api} if api else {})), FakeTask())
    f = res.failure
    assert f.kind == "agent_run_error" and f.who == "unknown" and f.detail == "unconfirmed:not_found" and f.masks, f
    probes = seen[1:]  # the first hit is the agent's own call
    assert not any(host.startswith("anthropic.invalid") for host, _ in probes)  # the policy's route is never probed
    assert probes == ([] if api is None else [("upstream.invalid/v1/chat/completions", "m")])


async def test_a_policy_route_the_only_upstream_does_not_serve_is_never_confirmed(tmp_path, monkeypatch, proto):
    """With only an OpenAI-compatible upstream that does not serve /v1/responses, a policy that calls it there first and
    last got a 404 the probe of /responses confirmed. The proxy's only upstream is the configured dialect: chat."""
    env = Env(tmp_path, monkeypatch)
    seen = _upstreams(env, anthropic=False)

    async def poll():
        assert await _agent_post(env, "/v1/responses", {"model": "m", "input": "hi"}) == 404
        return FAILED

    proto.script["poll"] = poll
    res = await env.run(env.harness(), FakeTask())
    f = res.failure
    assert f.kind == "agent_run_error" and f.who == "unknown" and f.detail == "unconfirmed:not_found", f
    assert seen == [("upstream.invalid/v1/responses", "m"), ("upstream.invalid/v1/chat/completions", "m")]


def test_the_configured_dialect_is_checked():
    assert A2AHarness(agent_url="http://a", capture=None, api="anthropic").api == "anthropic"
    with pytest.raises(ValueError, match="api must be one of"):
        A2AHarness(agent_url="http://a", capture=None, api="gemini")


# ------------------------------------------------------------------ part B: real agentenv A2A agents

STARTED: list[asyncio.Event] = []  # set when a SleepAgent's task starts running (the prompt reached it)


@functools.cache
def _sleep_agent():
    """An A2A agent that accepts the task and never finishes in time."""
    pytest.importorskip("agent_env")
    from agentenv_protocol.a2a_agent import MCP_CONFIG_V1, TRAJECTORY_V1, AgentEnvAgent, AgentIdentity, TaskRequest, TaskResult, a2a_agent

    from tests.unit.test_a2a_harness import ToolAgentConfig

    @a2a_agent(identity=AgentIdentity(name="crucible-test-sleep-agent", description="accepts the task, then never finishes in time", version="1"),
               config=ToolAgentConfig, extensions=(MCP_CONFIG_V1, TRAJECTORY_V1))
    class SleepAgent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            for ev in STARTED:
                ev.set()
            await asyncio.sleep(30)
            return TaskResult.builder().succeeded().add_text("too late").build()

    return SleepAgent


def _ok(holder: dict, req: httpx.Request) -> httpx.Response:
    holder.setdefault("keys", []).extend(holder["proxy"].sessions)  # the live capture key, seen while the agent's call is in flight
    return httpx.Response(200, json={"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 3, "completion_tokens": 2}})


def _overloaded(holder: dict, req: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"error": {"message": "overloaded", "type": "server_error"}})


async def _run_b(tmp_path, run_id: str, *, handler=_ok, agent=None, agent_url: str | None = None, task=None, k: int = 1,
                 budget: float | None = None, **hkw):
    """One run of a real A2A agent (served per attempt from ``agent``, default ``ToolAgent``; or the running one at
    ``agent_url``) on the toy helpdesk World behind agentenv's gateway, with the capture proxy served over HTTP."""
    pytest.importorskip("agent_env")
    from crucible_pack_toy.runtime import HelpdeskTask

    from tests.unit.test_a2a_harness import ToolAgent, serve

    holder: dict = {}
    proxy = CaptureProxy(tmp_path / f"cap-{run_id}", openai=Upstream(base="http://upstream.invalid/v1", key="sk-upstream"),
                         transport=httpx.MockTransport(functools.partial(handler, holder)))
    holder["proxy"] = proxy
    base, srv, server = await serve(proxy.app())
    try:
        src = {"agent_url": agent_url} if agent_url is not None else {"agent": agent or ToolAgent}
        h = A2AHarness(**src, capture=proxy, capture_base_url=base,
                       agent_config={"model": "m", "llm_base_url": "${capture.base_url}/v1", "llm_api_key": "${capture.key}"}, **hkw)
        t = task(base, holder) if task is not None else HelpdeskTask(backend="agentenv")
        s = await Runner(tmp_path).run(RunSpec(run_id=run_id, tasks=[t], policies=[PolicyRef(id="a2a:tool-agent", make=lambda a: {})], harnesses=[h],
                                               k=k, concurrency=1, budget_usd=budget, retry=RetryRule(max_attempts=1)))
        rows = [json.loads(x) for x in (tmp_path / run_id / "results.jsonl").read_text().splitlines() if x.strip()]
        return s, rows, proxy, t
    finally:
        server.should_exit = True
        await srv


class KeyCheckTask:
    """The toy helpdesk task whose grade first calls the model with the capture key the agent used: it must get 401."""

    def __init__(self, base: str, holder: dict):
        from crucible_pack_toy.runtime import HelpdeskTask

        self.inner = HelpdeskTask(backend="agentenv")
        self.id, self.world_id, self.instruction = self.inner.id, self.inner.world_id, self.inner.instruction
        self.base, self.holder, self.seen = base, holder, []

    async def open(self, workdir):
        return await self.inner.open(workdir)

    async def grade(self, world, workdir):
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{self.base}/v1/chat/completions", headers={"authorization": f"Bearer {self.holder['keys'][-1]}"},
                             json={"model": "m", "messages": []})
        self.seen.append((r.status_code, dict(self.holder["proxy"].sessions)))
        return await self.inner.grade(world, workdir)


async def test_a2a_capture_key_is_rejected_during_verification(tmp_path):
    """Acceptance: the key the agent called its model with is dead before the task is graded."""
    _s, rows, proxy, task = await _run_b(tmp_path, "key", task=KeyCheckTask)
    assert task.holder["keys"]  # the agent's model call went through the proxy with a live key
    assert task.seen == [(401, {})]
    assert rows[0]["reward"] == 1.0 and not rows[0]["mask_sample"], rows[0].get("failure")
    assert proxy.sessions == {}


async def test_a2a_agent_error_after_upstream_5xx_is_masked(tmp_path):
    _s, rows, _proxy, _t = await _run_b(tmp_path, "5xx", handler=_overloaded)
    f = Failure.model_validate(rows[0]["failure"])
    assert f.kind == "provider_unavailable" and rows[0]["mask_sample"] and f.who != "policy", f


async def test_a2a_spend_settles_from_the_capture_and_trips_the_budget(tmp_path):
    """Acceptance: each attempt's captured usage is priced (5.0 USD), so a 1.0 USD budget stops the run after one."""
    s, rows, _proxy, _t = await _run_b(tmp_path, "budget", k=3, budget=1.0, prices=PRICED)
    assert s.stopped and s.stopped.startswith("budget"), s.stopped
    assert s.attempts == 1 and s.spend["spent_run"] >= 5.0
    assert len(rows) == 1 and rows[0]["cost_usd"] == pytest.approx(5.0)


async def test_a2a_timeout_follows_the_limit_policy(tmp_path):
    _s, rows, _p, _t = await _run_b(tmp_path, "tmask", agent=_sleep_agent(), timeout_s=2, limit_policy=LimitPolicy(on_timeout="mask"))
    assert rows[0]["failure"]["kind"] == "agent_timeout" and rows[0]["mask_sample"]
    _s, rows, _p, _t = await _run_b(tmp_path, "tscore", agent=_sleep_agent(), timeout_s=2)
    assert rows[0]["meta"]["stop"] == "timeout" and not rows[0]["mask_sample"] and rows[0]["reward"] is not None  # graded, the policy's own


@pytest.mark.slow
async def test_killed_a2a_server_is_masked_transport_unreachable(tmp_path, monkeypatch):
    """Acceptance: an agent server that dies mid-task is masked transport_unreachable within about a minute
    (UNREACHABLE_AFTER_SECONDS patched to 2 s), not scored as a timeout after timeout_s (600 s)."""
    pytest.importorskip("agent_env")
    import agent_env.a2a_agent.protocol as proto

    if not hasattr(proto, "AgentUnreachableError") or "sandbox_id" not in inspect.signature(proto.poll_a2a_task).parameters:
        pytest.skip("the installed agentenv has no agent liveness (AgentUnreachableError / poll_a2a_task(sandbox_id=))")
    from crucible.interop.agentenv.a2a import ServedAgent

    monkeypatch.setattr(proto, "UNREACHABLE_AFTER_SECONDS", 2)
    served = await ServedAgent(_sleep_agent()()).start()
    started = asyncio.Event()
    STARTED.append(started)

    async def kill():
        await started.wait()  # the prompt reached the agent: everything before it (seed, mcp-config) is done
        await asyncio.sleep(0.5)
        await served.stop()

    killer = asyncio.create_task(kill())
    try:
        t0 = time.monotonic()
        _s, rows, _p, _t = await _run_b(tmp_path, "killed", agent_url=served.url, timeout_s=600)
        wall = time.monotonic() - t0
    finally:
        STARTED.remove(started)
        killer.cancel()
        await served.stop()
    f = Failure.model_validate(rows[0]["failure"])
    assert f.kind == "transport_unreachable" and rows[0]["mask_sample"], f
    assert wall < 90, wall
