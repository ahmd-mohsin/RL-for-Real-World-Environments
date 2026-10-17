"""M10 lab audit R-1 (2026-10-15), slice 4: Harbor installed agents as a harness (``interop/harbor/agent_harness``), the
separate-verifier patch seam (``interop/harbor/patch_task``) and the in-place task hooks.

- The id carries a digest of the configuration; the agent's endpoint variables carry only the capture key, scoped to
  setup and run, and a host credential in what the agent would see refuses the attempt.
- Setup is the harness's (its own deadline, never the policy's); only Crucible's own deadline is a ``timeout`` stop; the
  key is dead before verification; a non-policy blame needs proxy or host evidence (an upstream 503 then a non-zero exit
  is masked, an agent printing "rate limit" with a healthy proxy is scored).
- The proxy's call cap is an evidenced ``max_steps``; mini-swe-agent's own exit statuses are recorded and scored.
- Spend settles from the capture on every exit path (``budget_usd`` trips on a Harbor run).
- The patch seam collects, closes the World, then grades; a masked grade is re-graded from the stored patch with the
  prior attempt's evidence; a collect hook that fails in a live container is the agent's; the reconcile is recorded,
  and a FATAL one masks a train rollout once, never retried.
- Review round 1: the patch is read through the container (a link at its path is refused, a failing read masks without a
  retry); links the policy plants under /logs are never followed; host keys an agent reads at run time reach it blank;
  a forged upstream refusal is unknown (Crucible's probe disagrees); an agent that never reached its model is not the
  policy's; the stop spares the init session and kills the phase's cohort; every record carries the agent's identity.

- Review round 2: in place only the agent's own tree is stopped here (M10 lab audit R-2: the grade's host-side quiesce
  kills the rest, and its survivors are ``crucible:policy_misbehavior``); the stops need
  no procps, fail closed, and their survivors or stalls refuse the collect; the patch read is framed and classified by
  its in-container verdict (Harbor's exec merges stderr and returns docker or ssh failures as exit codes).
- Review round 3: a root policy controls ``bash``, ``sleep`` and every tool the stop, the hooks and the read run, so a
  stop, hook or read that never reports is the policy's only on evidence that can only blame it (its shell killed after
  its marker while a fresh root shell answers, runc's refusal to start it) and otherwise unknown, never infra; the stop
  and the read run in ``/``; each hook is framed with its own verdict; a stall is decided by the time alone; children a
  service older than the phase respawns and processes in uninterruptible sleep are never the policy's.

Part A runs without Harbor: fake environment, task and agent, the real ``CaptureProxy`` behind ``httpx.MockTransport``
and the real ``Runner``. Part B needs Harbor (``pytest.importorskip("harbor")``); its tests that admit the proxy host or
read ``agents.model_connection`` need the 0.24 API and skip on the locked 0.23.0.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from crucible.connectors.capture_proxy import CaptureProxy, Upstream
from crucible.core.schema.run import Failure
from crucible.interop.harbor.agent_harness import AgentLimits, HarborAgentHarness, _agent_env_views, config_digest, mini_swe_exit_status
from crucible.interop.harbor.patch_task import CollectError, HarborPatchTask
from crucible.runtime.agent_evidence import leaked_host_secrets
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.infra import LimitPolicy, RetryRule
from crucible.runtime.run import Runner, RunSpec
from crucible.verify.runtime.plan import GradeReport

F = Path(__file__).resolve().parents[1] / "fixtures" / "harbor_inplace"
PRICED = {"m": [1_000_000.0, 1_000_000.0]}  # 3 prompt + 2 completion tokens -> 5.0 USD per call
HOST_KEY, HOST_ANT = "sk-real-host-123456", "sk-ant-real-123456"


class NonZeroAgentExitCodeError(RuntimeError):  # Harbor agents/installed/base.py:34, matched by name (no Harbor import)
    pass


class ApiRateLimitError(NonZeroAgentExitCodeError):  # a regex over the agent's own output (base.py:46): no evidence
    pass


# ------------------------------------------------------------------ part A: fakes (no Harbor)


class FakeEnv:
    """Harbor's ``BaseEnvironment`` surface the harness uses: ``scoped_exec_env`` overlays, ``with_default_user``, ``exec``.
    ``sent`` records each command with the env it would run with (the scoped overlay wins over its own env, as Harbor's
    ``_merge_env``), its user and its cwd; ``mark`` answers the harness's phase-mark probe; ``kill_raises`` fails the
    cohort kill; ``kill`` scripts the default stop's answer (``rc``, ``stdout``, ``stall``: sleep that long, then raise
    Harbor's deadline error, or ``timeout_now``: a provider's fast TimeoutError). A fresh-shell probe
    (``printf '%s\\n' CRUCIBLE-PROBE-<nonce>``) is answered with its nonce unless ``probe_dead`` (review round 3)."""

    def __init__(self, events: list, sent: list | None = None, mark: int | None = None, kill_raises: bool = False, kill: dict | None = None,
                 probe_dead: bool = False):
        self.events, self.overlays, self.default_user = events, [], None
        self.sent, self.mark, self.kill_raises, self.kill, self.probe_dead = sent, mark, kill_raises, kill, probe_dead
        self.capabilities = SimpleNamespace(mounted=True)

    @contextmanager
    def scoped_exec_env(self, env):
        self.overlays.append(dict(env))
        try:
            yield
        finally:
            self.overlays.pop()

    @contextmanager
    def with_default_user(self, user):
        prev, self.default_user = self.default_user, user
        try:
            yield
        finally:
            self.default_user = prev

    def current_env(self) -> dict:
        merged: dict = {}
        for o in self.overlays:
            merged.update(o)
        return merged

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.events.append(f"exec:{command}")
        if self.sent is not None:
            self.sent.append({"command": command, "env": {**(env or {}), **self.current_env()}, "user": user, "cwd": cwd})
        if command.startswith("cut -d' ' -f22") and self.mark is not None:
            return SimpleNamespace(return_code=0, stdout=f"{self.mark}\n", stderr="")
        if "CRUCIBLE-PROBE-" in command:  # the harness's fresh-shell probe: the nonce back, or nothing
            if self.probe_dead:
                return SimpleNamespace(return_code=255, stdout=CONNECT_ERROR, stderr=None)
            return SimpleNamespace(return_code=0, stdout=command.split()[-1] + "\n", stderr=None)
        if self.kill_raises and self.mark is not None and f"-ge {self.mark}" in command:
            raise RuntimeError("docker exec failed: the container is not responding")
        if self.kill and "killed:" in command:  # the harness's default stop (user, cohort or tree)
            if self.kill.get("stall"):
                await asyncio.sleep(self.kill["stall"])
                raise RuntimeError("Command timed out after 60 seconds")
            if self.kill.get("timeout_now"):
                raise TimeoutError("connect timed out")
            return SimpleNamespace(return_code=self.kill.get("rc", 0), stdout=self.kill.get("stdout", ""), stderr=None)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


class FakeWorld:
    def __init__(self, workdir: Path, task: FakeTask):
        self.task = task
        self.backend = SimpleNamespace(env=FakeEnv(task.events, task.sent, task.mark, task.kill_raises, task.kill, task.probe_dead), params={},
                                       container_lost=self._lost)
        self.logs = Path(workdir) / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    async def _lost(self):
        return self.task.lost

    async def close(self):
        self.task.events.append("close")


class FakeTask:
    id, world_id, instruction = "fake/t", "harbor:fake", "do it"
    agent_user = None
    agent_timeout_s = 5.0
    mark: int | None = None  # what the phase-mark probe answers (None: an empty answer, no mark)
    kill_raises = False
    kill: dict | None = None  # the default stop's scripted answer (FakeEnv)
    probe_dead = False  # the harness's fresh-shell probe gets no answer (FakeEnv)

    def __init__(self, events: list, *, reward: float = 1.0, lost: str | None = None, on_grade=None, admits: bool = True):
        self.events, self.reward, self.lost, self.on_grade, self.admits = events, reward, lost, on_grade, admits
        self.graded: list[dict] = []
        self.sent: list[dict] = []  # every command the fake container ran, with its effective env and user

    def admits_agent_host(self, host):
        return self.admits

    async def open(self, workdir, *, before_agent=None):
        w = FakeWorld(Path(workdir), self)
        if before_agent is not None:
            await before_agent(w)
        self.events.append("agent_phase")
        return w

    async def agent_logs(self, world, *, fetch=False):
        return world.logs

    def report(self, reward: float, infra: dict | None = None) -> GradeReport:
        return GradeReport(task=self.id, tier="full", checks=[], gates={"g": "pass"}, reward=reward, outcome=reward, infra=infra or {"status": "ok"},
                           trusted=True, elapsed_s=0.0)

    async def grade(self, world, gdir):
        self.graded.append(world.backend.env.current_env())
        self.events.append("grade")
        if self.on_grade is not None:
            await self.on_grade()
        return self.report(self.reward)


class FakePatchTask(FakeTask):
    def __init__(self, events: list, *, masked_first: bool = False, collect_error: CollectError | None = None, **kw):
        super().__init__(events, **kw)
        self.masked_first, self.collect_error, self.patch_grades = masked_first, collect_error, 0

    async def collect(self, world, dest):
        self.events.append("collect")
        if self.collect_error is not None:
            raise self.collect_error
        return "diff --git a/x b/x\n"

    async def grade_patch(self, patch, gdir):
        self.patch_grades += 1
        self.events.append("grade_patch")
        if self.masked_first and self.patch_grades == 1:
            return self.report(0.0, {"status": "masked", "failure_kind": "check_infra", "reason": "x"})
        return self.report(self.reward)


class FakeAgent:
    """A Harbor installed agent's surface: ``setup``, ``run``, ``populate_context_post_run``, ``extra_env``,
    ``model_connection``. Its behaviour comes from ``env.hooks``."""

    def __init__(self, env: Env, logs_dir: Path, extra_env: dict, kw: dict, connection_extra: dict):
        self.env_, self.logs_dir, self.extra_env, self.kw = env, Path(logs_dir), dict(extra_env), kw
        self.model_connection = SimpleNamespace(env={**dict(extra_env), **connection_extra})
        self.seen: list[tuple[str, dict]] = []

    async def setup(self, environment):
        self.env_.events.append("setup")
        self.seen.append(("setup", environment.current_env()))
        if self.env_.hooks.get("setup") is not None:
            await self.env_.hooks["setup"](self, environment)

    async def run(self, instruction, environment, context):
        self.env_.events.append("run")
        self.seen.append(("run", environment.current_env()))
        if self.env_.hooks.get("run") is not None:
            await self.env_.hooks["run"](self, environment)

    def populate_context_post_run(self, context):
        self.env_.events.append("populate")

    async def call(self, text: str = "hi", model: str = "m") -> int:
        return await self.env_.call(self.extra_env["OPENAI_API_KEY"], text, model)


class Env:
    """The real capture proxy in front of a fake upstream; a factory of fake agents whose model calls go through it.
    ``by_model``: the status the upstream answers for a model (else ``status``); Crucible's own probe hits it too."""

    def __init__(self, tmp_path: Path, status: int = 200, by_model: dict[str, int] | None = None):
        self.tmp, self.status, self.hits = tmp_path, status, 0
        self.by_model = dict(by_model or {})
        self.models: list[str | None] = []  # the model of every request that reached the upstream
        self.events: list = []
        self.hooks: dict[str, Any] = {"setup": None, "run": None}
        self.agents: list[FakeAgent] = []
        self.proxy = CaptureProxy(tmp_path / "cap", openai=Upstream(base="http://upstream.invalid/v1", key="sk-upstream"),
                                  transport=httpx.MockTransport(self._upstream))
        self.app = self.proxy.app()

    def _upstream(self, req: httpx.Request) -> httpx.Response:
        self.hits += 1
        model = json.loads(req.content or b"{}").get("model")
        self.models.append(model)
        status = self.by_model.get(model, self.status)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "overloaded" if status >= 500 else f"refused ({status})"}})
        return httpx.Response(200, json={"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    async def call(self, key: str, text: str = "hi", model: str = "m") -> int:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://proxy") as c:
            r = await c.post("/v1/chat/completions", headers={"authorization": f"Bearer {key}"},
                             json={"model": model, "messages": [{"role": "user", "content": text}]})
        return r.status_code

    def factory(self, **connection_extra):
        def make(*, logs_dir, model_name, version, extra_env, **kw):
            a = FakeAgent(self, logs_dir, extra_env, kw, connection_extra)
            self.agents.append(a)
            return a

        return make

    def harness(self, **kw) -> HarborAgentHarness:
        kw.setdefault("prices", PRICED)
        kw.setdefault("agent_factory", self.factory())
        return HarborAgentHarness("fake-agent", "1", model="openai/m", proxy=self.proxy, proxy_url="http://proxy",
                                  endpoint_env={"OPENAI_API_KEY": "{key}", "OPENAI_BASE_URL": "{proxy}/v1"}, harbor_version="0.24.0",
                                  context_factory=lambda: SimpleNamespace(), **kw)

    async def run(self, task, run_id: str = "r", *, harness: HarborAgentHarness | None = None, **spec_kw):
        spec_kw.setdefault("retry", RetryRule(max_attempts=1))
        s = await Runner(self.tmp / "runs").run(RunSpec(run_id=run_id, tasks=[task], policies=[PolicyRef(id="harbor:m", make=lambda a: {})],
                                                        harnesses=[harness or self.harness()], **spec_kw))
        rows = [json.loads(x) for x in (self.tmp / "runs" / run_id / "results.jsonl").read_text().splitlines() if x.strip()]
        return s, rows

    def adir(self, run_id: str, row: dict) -> Path:
        sid, n = row["attempt"].rsplit("-a", 1)
        return self.tmp / "runs" / run_id / "samples" / sid / f"a{n}"


async def _one_call(agent: FakeAgent, environment) -> None:
    assert await agent.call() == 200


def _failure(row: dict) -> Failure:
    return Failure.model_validate(row["failure"])


def test_harness_id_digest(tmp_path):
    proxy = CaptureProxy(tmp_path / "cap")
    fac = lambda **kw: None  # noqa: E731 -- never called: the id needs no Harbor

    def h(agent="mini-swe-agent", proxy_url="http://10.1.2.3:8788", **kw):
        kw.setdefault("harbor_version", "0.24.0")
        return HarborAgentHarness(agent, "2.4.6", model=kw.pop("model", "openai/gpt-x"), proxy=proxy, proxy_url=proxy_url, agent_factory=fac, **kw)

    base = h()
    assert re.fullmatch(r"harbor:mini-swe-agent@2\.4\.6#[0-9a-f]{12}", base.id), base.id
    assert h().id == base.id  # the same configuration: the same SampleKeys and GRPO groups
    changed = [h(agent_kwargs={"config": {"agent": {"step_limit": 10}}}), h(limits=AgentLimits(max_model_calls=7)),
               h(limit_policy=LimitPolicy(on_timeout="mask")), h(model="openai/gpt-y"), h(harbor_version="0.23.0"),
               h(endpoint_env={"OPENAI_API_KEY": "{key}", "OPENAI_BASE_URL": "{proxy}/v1"})]
    digests = {base.digest, *(x.digest for x in changed)}
    assert len(digests) == 1 + len(changed)  # every change makes a new id
    assert h("pkg.mod:MyAgent").id.startswith("harbor:MyAgent@2.4.6#")
    # M10 lab audit R-1: two modules' MyAgent share the id's class name but never the digest (SampleKeys, GRPO groups)
    assert h("a.agents:MyAgent").id != h("b.agents:MyAgent").id
    assert config_digest(a=1, b=[1, 2]) == config_digest(b=[1, 2], a=1)  # canonical: key order does not matter
    with pytest.raises(ValueError, match="no host"):
        h(proxy_url="not a url")


async def test_upstream_503_then_nonzero_exit_is_masked_provider_unavailable(tmp_path):
    """Acceptance: the proxy saw the provider fail last, so the agent's non-zero exit is not the policy's 0."""
    env = Env(tmp_path, status=503)
    task = FakeTask(env.events)

    async def run(agent, environment):
        assert await agent.call() == 503
        raise NonZeroAgentExitCodeError("mini-swe-agent exited 1")

    env.hooks["run"] = run
    _s, rows = await env.run(task)
    f = _failure(rows[0])
    assert f.kind == "provider_unavailable" and rows[0]["mask_sample"] and f.who != "policy", f
    assert task.graded == []
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["upstream"]["verdict"] == "infra" and doc["harbor_version"] == "0.24.0"
    assert doc["exception"]["type"] == "NonZeroAgentExitCodeError" and "NonZeroAgentExitCodeError" in doc["exception"]["chain"]


async def test_agent_printing_rate_limit_with_a_healthy_proxy_is_scored(tmp_path):
    """Acceptance: Harbor's ``ApiRateLimitError`` is a regex over the agent's own output; with a healthy proxy it is the
    policy's outcome."""
    env = Env(tmp_path)

    async def run(agent, environment):
        assert await agent.call() == 200
        raise ApiRateLimitError("rate limit")

    env.hooks["run"] = run
    _s, rows = await env.run(FakeTask(env.events))
    f = _failure(rows[0])
    assert f.who == "policy" and not rows[0]["mask_sample"] and rows[0]["reward"] == 0.0, f
    assert "ApiRateLimitError" in f.reason


async def test_capture_key_is_rejected_during_verification(tmp_path):
    """Acceptance: the key the agent called its model with gets 401 while its work is graded; the scoped endpoint
    variables never reach verification."""
    env = Env(tmp_path)
    seen: dict = {}

    async def on_grade():
        seen["status"] = await env.call(env.agents[0].extra_env["OPENAI_API_KEY"])
        seen["live"] = dict(env.proxy.sessions)

    task = FakeTask(env.events, on_grade=on_grade)
    env.hooks["run"] = _one_call
    _s, rows = await env.run(task, harness=env.harness(kill_command="kill-agent-tree"))
    assert rows[0]["reward"] == 1.0 and not rows[0]["mask_sample"], rows[0].get("failure")
    assert seen == {"status": 401, "live": {}} and env.hits == 1  # the verification call never reached the upstream
    assert task.graded == [{}]  # no overlay at grade time
    ev = env.events
    # the agent's processes stopped, then the grade; M10 lab audit R-2: no separate quiesce seam (the guarded grade
    # quiesces the container host-side itself)
    assert ev.index("run") < ev.index("exec:kill-agent-tree") < ev.index("grade")
    assert not any("quiesce" in e for e in ev), ev
    assert env.proxy.sessions == {}


async def test_no_provider_key_in_the_agent_env(tmp_path, monkeypatch):
    """Acceptance: the agent's environment carries the capture key, never a host credential; one that would is refused
    before setup."""
    monkeypatch.setenv("OPENAI_API_KEY", HOST_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", HOST_ANT)
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    _s, rows = await env.run(FakeTask(env.events))
    assert not rows[0]["mask_sample"], rows[0].get("failure")
    agent = env.agents[0]
    key = agent.extra_env["OPENAI_API_KEY"]
    assert key.startswith("cap-") and agent.extra_env["OPENAI_BASE_URL"] == "http://proxy/v1"
    assert [p for p, _ in agent.seen] == ["setup", "run"]
    for _phase, overlay in agent.seen:
        assert overlay["OPENAI_API_KEY"] == key
        assert HOST_KEY not in overlay.values() and HOST_ANT not in overlay.values()
    assert HOST_KEY not in agent.extra_env.values() and HOST_ANT not in agent.extra_env.values()

    leaky = Env(tmp_path / "leaky")
    leaky.hooks["run"] = _one_call
    _s, rows = await leaky.run(FakeTask(leaky.events), harness=leaky.harness(agent_factory=leaky.factory(OPENAI_API_KEY=HOST_KEY)))
    f = _failure(rows[0])
    assert f.kind == "crucible:harness_unsupported" and rows[0]["mask_sample"] and not f.retryable and "model_connection:OPENAI_API_KEY" in f.reason
    assert "setup" not in leaky.events and "run" not in leaky.events and leaky.proxy.sessions == {}


async def test_setup_runs_before_the_agent_phase_under_its_own_deadline(tmp_path):
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    _s, rows = await env.run(FakeTask(env.events))
    assert not rows[0]["mask_sample"]
    assert env.events.index("setup") < env.events.index("agent_phase") < env.events.index("run")

    slow = Env(tmp_path / "slow")

    async def sleepy(agent, environment):
        await asyncio.sleep(1)

    slow.hooks["setup"] = sleepy
    h = slow.harness(limits=AgentLimits(setup_timeout_s=0.05))
    _s, rows = await slow.run(FakeTask(slow.events), harness=h)
    f = _failure(rows[0])
    assert (f.kind, f.stage, f.detail) == ("crucible:backend_error", "seed", "AgentSetupTimeoutError") and rows[0]["mask_sample"], f
    assert "run" not in slow.events and slow.proxy.sessions == {}
    # M10 lab audit R-1 (record per attempt): a setup failure still names the agent, digest, Harbor version and session
    ag = rows[0]["meta"]["agent"]
    assert (ag["agent"], ag["harness"], ag["config_digest"], ag["harbor_version"], ag["model"]) == ("fake-agent@1", h.id, h.digest, "0.24.0", "openai/m")
    assert ag["proxy_session"] and ag["harness"] == rows[0]["meta"]["harness"]

    gone = Env(tmp_path / "gone")

    async def crash(agent, environment):
        raise RuntimeError("install exited 137")

    gone.hooks["setup"] = crash
    _s, rows = await gone.run(FakeTask(gone.events, lost="removed"))
    assert _failure(rows[0]).kind == "crucible:env_container_lost" and rows[0]["mask_sample"]
    crashed = Env(tmp_path / "crashed")
    crashed.hooks["setup"] = crash
    _s, rows = await crashed.run(FakeTask(crashed.events))
    f = _failure(rows[0])
    assert (f.kind, f.detail) == ("crucible:backend_error", "agent_setup:RuntimeError") and rows[0]["mask_sample"]  # never the policy's


async def test_setup_timeout_after_a_model_call_still_settles_spend(tmp_path):
    """Critic amendment 11: the capture-log path is known when the session is, so an install that called the model and
    then timed out still settles what the proxy forwarded."""
    env = Env(tmp_path)

    async def call_then_hang(agent, environment):
        assert await agent.call() == 200
        await asyncio.sleep(1)

    env.hooks["setup"] = call_then_hang
    _s, rows = await env.run(FakeTask(env.events), harness=env.harness(limits=AgentLimits(setup_timeout_s=0.05)))
    assert rows[0]["mask_sample"] and _failure(rows[0]).kind == "crucible:backend_error"
    assert rows[0]["cost_usd"] == pytest.approx(5.0)


async def test_crucible_deadline_is_a_limit_stop_and_only_its_own(tmp_path):
    env = Env(tmp_path)

    async def hang(agent, environment):
        # the policy answered once (an installed agent that never reached its model is not the policy's: see below)
        assert await agent.call() == 200
        await asyncio.sleep(5)

    env.hooks["run"] = hang
    _s, rows = await env.run(FakeTask(env.events, reward=0.25), harness=env.harness(limits=AgentLimits(timeout_s=0.5)))
    assert rows[0]["meta"]["stop"] == "timeout" and rows[0]["meta"]["truncated"] and rows[0]["reward"] == 0.25 and not rows[0]["mask_sample"]

    masked = Env(tmp_path / "masked")
    masked.hooks["run"] = hang
    _s, rows = await masked.run(FakeTask(masked.events), harness=masked.harness(limits=AgentLimits(timeout_s=0.5), limit_policy=LimitPolicy(on_timeout="mask")))
    assert _failure(rows[0]).kind == "agent_timeout" and rows[0]["mask_sample"]
    assert "grade" not in masked.events

    inner = Env(tmp_path / "inner")

    async def raises_timeout(agent, environment):
        assert await agent.call() == 200
        raise TimeoutError("inner")  # the agent's own socket timeout: not Crucible's deadline

    inner.hooks["run"] = raises_timeout
    _s, rows = await inner.run(FakeTask(inner.events))
    f = _failure(rows[0])
    assert rows[0]["meta"]["stop"] != "timeout" and f.who == "unknown" and rows[0]["mask_sample"], f


async def test_proxy_call_cap_is_a_max_steps_stop(tmp_path):
    env = Env(tmp_path)
    statuses: list[int] = []

    async def three(agent, environment):
        for _ in range(3):
            statuses.append(await agent.call())

    env.hooks["run"] = three
    _s, rows = await env.run(FakeTask(env.events), harness=env.harness(limits=AgentLimits(max_model_calls=2)))
    assert env.hits == 2 and statuses == [200, 200, 200]  # the third is the proxy's own stop answer
    assert rows[0]["meta"]["stop"] == "max_steps" and not rows[0]["mask_sample"] and rows[0]["reward"] == 1.0
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["limit_stops"] == 1 and doc["outcome"]["evidenced"]
    # M10 lab audit R-1: both forwarded calls are billed (identical bodies collapse into one exchange, never into one bill);
    # the third answer was the proxy's own stop
    assert rows[0]["cost_usd"] == pytest.approx(10.0)


@pytest.mark.parametrize("status, stop", [("LimitsExceeded", "max_steps"), ("Submitted", "agent_done"), ("TimeExceeded", "timeout")])
async def test_mini_swe_agent_exit_status_mapping(tmp_path, status, stop):
    """mini-swe-agent's own stops are recorded and scored; read from a file the policy can write, they are agent-reported,
    so even a mask policy scores them (critic amendment 6)."""
    env = Env(tmp_path)

    async def write_status(agent, environment):
        (agent.logs_dir / "mini-swe-agent.trajectory.json").write_text(json.dumps({"info": {"exit_status": status}}))

    env.hooks["run"] = write_status
    policy = LimitPolicy(on_timeout="mask", on_steps="mask", on_context_window="mask")
    _s, rows = await env.run(FakeTask(env.events, reward=0.5), harness=env.harness(limit_policy=policy))
    assert rows[0]["meta"]["stop"] == stop and not rows[0]["mask_sample"] and rows[0]["reward"] == 0.5, rows[0].get("failure")
    assert rows[0]["meta"]["agent"]["exit_status"] == status
    assert mini_swe_exit_status(tmp_path) is None  # no file: no status


async def test_container_lost_is_env_container_lost(tmp_path):
    env = Env(tmp_path)

    async def die(agent, environment):
        raise NonZeroAgentExitCodeError("exit 137")

    env.hooks["run"] = die
    _s, rows = await env.run(FakeTask(env.events, lost="the task container was removed"))
    f = _failure(rows[0])
    assert f.kind == "crucible:env_container_lost" and rows[0]["mask_sample"] and "removed" in f.reason


async def test_budget_trips_on_a_harbor_run(tmp_path):
    """Acceptance: each attempt's captured usage is priced (5.0 USD), so a 1.0 USD budget stops the run after one."""
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    s, rows = await env.run(FakeTask(env.events), k=3, concurrency=1, budget_usd=1.0)
    assert s.stopped and s.stopped.startswith("budget"), s.stopped
    assert s.attempts == 1 and len(rows) == 1 and rows[0]["cost_usd"] == pytest.approx(5.0)


async def test_patch_seam_collects_closes_then_grades_and_regrades_the_stored_patch(tmp_path):
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    task = FakePatchTask(env.events, reward=0.75)
    _s, rows = await env.run(task, harness=env.harness(kill_command="kill-agent-tree"))
    assert rows[0]["reward"] == 0.75 and not rows[0]["mask_sample"], rows[0].get("failure")
    ev = env.events
    assert ev.index("run") < ev.index("exec:kill-agent-tree") < ev.index("collect") < ev.index("close") < ev.index("grade_patch")
    assert (env.adir("r", rows[0]) / "artifacts" / "model.patch").read_text() == "diff --git a/x b/x\n"
    assert "grade" not in ev  # a separate-verifier task is never graded in the agent's container

    again = Env(tmp_path / "again")
    again.hooks["run"] = _one_call
    task = FakePatchTask(again.events, reward=0.75, masked_first=True)
    s, rows = await again.run(task, retry=RetryRule(max_attempts=2))
    assert again.events.count("run") == 1 and task.patch_grades == 2  # the agent ran once; its stored patch was re-graded
    assert s.attempts == 2 and rows[-1]["meta"]["stop"] == "regraded" and rows[-1]["reward"] == 0.75 and not rows[-1]["mask_sample"]
    # critic amendment 2: the re-grade keeps the evidence of the attempt that ran the agent
    assert rows[-1]["meta"]["agent"]["harbor_version"] == "0.24.0" and rows[-1]["meta"]["agent"]["regraded_from"] == "a1"
    assert rows[-1]["meta"]["reconcile"]["level"] in ("info", "warn")
    assert (again.adir("r", rows[-1]) / "agent_run.json").exists()


@pytest.mark.parametrize("kind, lost, expect, masked", [
    ("hook", None, "crucible:agent_broke_env", False),  # the agent's repository, its base tag: its outcome (P23)
    ("missing", None, "crucible:agent_broke_env", False),
    ("hook", "the task container was removed", "crucible:env_container_lost", True),
    # M10 lab audit R-1: a read that kept failing in a live container masks, never retried (a retry re-runs the agent)
    ("transport", None, "crucible:backend_error", True),
    ("transport", "the task container was removed", "crucible:env_container_lost", True),  # host evidence: the work is gone
    ("unsafe", None, "crucible:policy_misbehavior", False),  # a symlink or special file at the patch path: custody's rule
    # review round 3: no shell report and nothing showing whose (an ssh blip, or a root policy's replaced shell): unknown
    ("shell", None, "crucible:backend_error", True),
    ("shell", "the task container was removed", "crucible:env_container_lost", True),
])
async def test_collect_failures_follow_host_evidence(tmp_path, kind, lost, expect, masked):
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    task = FakePatchTask(env.events, collect_error=CollectError("collect hook exited 128: fatal: bad revision 'base'", kind=kind), lost=lost)
    s, rows = await env.run(task, retry=RetryRule(max_attempts=3))
    f = _failure(rows[-1])
    assert f.kind == expect and f.stage == "collect" and rows[-1]["mask_sample"] == masked, f
    assert task.patch_grades == 0
    if not masked:
        assert rows[-1]["reward"] == 0.0 and s.attempts == 1
    if kind == "transport" and lost is None:  # an exec that raised fast: host side, infrastructure
        assert s.attempts == 1 and env.events.count("run") == 1 and not f.retryable and f.detail == "collect_transport" and f.who == "infra"
    if kind == "shell" and lost is None:  # review round 3: unknown blame, counted by the 5% stop, never retried
        assert s.attempts == 1 and env.events.count("run") == 1 and not f.retryable and f.detail == "collect_unreported" and f.who == "unknown"


def _atif(n: int) -> dict:
    return {"schema_version": "ATIF-v1.7", "steps": [{"step_id": i + 1, "source": "agent", "message": "x", "metrics": {"completion_tokens": 2}}
                                                      for i in range(n)]}


async def test_reconcile_fatal_is_recorded(tmp_path):
    env = Env(tmp_path)

    async def logs_three_calls_one(agent, environment):
        assert await agent.call() == 200
        (agent.logs_dir / "trajectory.json").write_text(json.dumps(_atif(3)))

    env.hooks["run"] = logs_three_calls_one
    _s, rows = await env.run(FakeTask(env.events))
    assert rows[0]["meta"]["reconcile"]["level"] == "fatal" and not rows[0]["mask_sample"] and rows[0]["reward"] == 1.0  # eval: recorded only
    a = env.adir("r", rows[0])
    assert json.loads((a / "reconcile.json").read_text())["code"] == "atif_calls_missing"
    assert json.loads((a / "agent_trajectory.atif.json").read_text())["steps"][0]["source"] == "agent"
    assert rows[0]["meta"]["artifacts"]["agent_atif"] == "agent_trajectory.atif.json"
    assert not any(k.startswith("trajectory.") for k in rows[0]["meta"]["artifacts"])  # never parsed as Crucible's own ATIF


async def test_fatal_reconcile_masks_a_train_rollout_once(tmp_path):
    """Critic amendment 2: on the patch seam in train mode a FATAL reconcile masks once and is never retried (a retry
    would re-grade the stored patch and launder the mask)."""
    env = Env(tmp_path)

    async def logs_three_calls_one(agent, environment):
        assert await agent.call() == 200
        (agent.logs_dir / "trajectory.json").write_text(json.dumps(_atif(3)))

    env.hooks["run"] = logs_three_calls_one
    s, rows = await env.run(FakePatchTask(env.events), retry=RetryRule(max_attempts=2), rollout_mode="train")
    f = _failure(rows[-1])
    assert s.attempts == 1 and rows[-1]["mask_sample"] and f.kind == "crucible:capture_incomplete" and f.detail == "reconcile_fatal", f


async def test_aux_calls_are_not_trainable(tmp_path):
    env = Env(tmp_path)

    async def two_calls_one_step(agent, environment):
        # two different requests: an identical body without an SDK retry header is a retry by CaptureLogConnector's
        # rule (test_m6_audit.py:250) and would collapse into one captured exchange
        assert await agent.call("one") == 200
        assert await agent.call("two") == 200
        (agent.logs_dir / "trajectory.json").write_text(json.dumps(_atif(1)))

    env.hooks["run"] = two_calls_one_step
    _s, rows = await env.run(FakeTask(env.events))
    rc = rows[0]["meta"]["reconcile"]
    assert rc["code"] == "atif_aux_calls" and rc["aux_call_ids"] == ["agent:cap2"] and not rows[0]["mask_sample"], rc
    calls = json.loads((env.adir("r", rows[0]) / "calls.json").read_text())["agent"]
    assert [c.get("trainable", True) for c in calls] == [True, False]


async def test_records_per_attempt(tmp_path):
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    _s, rows = await env.run(FakeTask(env.events))
    a = env.adir("r", rows[0])
    doc = json.loads((a / "agent_run.json").read_text())
    for k in ("agent", "version", "harness", "config_digest", "harbor_version", "model", "proxy_session", "exit_status", "exception", "upstream",
              "outcome", "processes", "limit_stops", "scrubbed", "agent_network"):
        assert k in doc, k
    # "skipped": an in-place task stops only the agent's own tree, and "fake-agent" has no AGENT_KILL
    assert doc["exception"] is None and doc["processes"]["status"] == "skipped" and doc["harness"] == rows[0]["meta"]["harness"]
    assert doc["processes"]["scope"] == "agent_tree"
    assert (a / "calls.json").exists() and (a / "reconcile.json").exists()
    meta = rows[0]["meta"]
    assert meta["agent"]["harbor_version"] == "0.24.0" and meta["agent"]["agent"] == "fake-agent@1" and meta["agent"]["upstream"] == "ok"
    assert meta["reconcile"]["code"] == "no_atif"
    assert rows[0]["cost_usd"] == pytest.approx(5.0)


async def test_refusals_are_harness_unsupported_and_never_retried(tmp_path):
    env = Env(tmp_path)
    s, rows = await env.run(FakeTask(env.events, admits=False), retry=RetryRule(max_attempts=3))
    f = _failure(rows[0])
    assert f.kind == "crucible:harness_unsupported" and not f.retryable and "agent_allowed_hosts" in f.reason and s.attempts == 1
    assert env.events == [] and env.proxy.sessions == {}
    # M10 lab audit R-1: refused before any key, the record still names the Harbor version (no session yet)
    assert rows[0]["meta"]["agent"]["harbor_version"] == "0.24.0" and rows[0]["meta"]["agent"]["proxy_session"] is None

    class Plain:  # not a Harbor task adapter
        id, world_id, instruction = "plain/t", "plain", "x"

    other = Env(tmp_path / "plain")
    _s, rows = await other.run(Plain())
    assert _failure(rows[0]).kind == "crucible:harness_unsupported"


async def test_a_multi_step_task_toml_is_refused_without_steps(tmp_path):
    """M10 lab audit R-1: a plain adapter on a ``[[steps]]`` task.toml has no ``steps`` (only ``HarborMultiStepTask`` builds
    them) but ``open`` applies no agent phase to it, so it must be refused on ``multi_step`` before any key or setup, even
    when its task-level agent phase admits the proxy."""
    env = Env(tmp_path)
    task = FakeTask(env.events, admits=True)
    task.multi_step = True
    s, rows = await env.run(task, retry=RetryRule(max_attempts=3))
    f = _failure(rows[0])
    assert f.kind == "crucible:harness_unsupported" and not f.retryable and "multi-step" in f.reason and s.attempts == 1
    assert env.events == [] and env.proxy.sessions == {} and env.agents == []


async def test_an_agent_constructor_os_error_is_harness_unsupported_and_never_retried(tmp_path):
    """M10 lab audit R-1: a Harbor agent reads the host files its kwargs name in its constructor (mini-swe-agent's
    config_file); a missing one is configuration, never the transport (``boundary`` maps any OSError to a retried
    ``transport_unreachable``)."""
    env = Env(tmp_path)

    def make(**kw):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'missing.yaml'")

    s, rows = await env.run(FakeTask(env.events), harness=env.harness(agent_factory=make), retry=RetryRule(max_attempts=3))
    f = _failure(rows[0])
    assert f.kind == "crucible:harness_unsupported" and not f.retryable and "FileNotFoundError" in f.reason and s.attempts == 1
    assert "setup" not in env.events and "agent_phase" not in env.events and env.proxy.sessions == {}


async def test_early_failures_carry_the_agent_identity(tmp_path, monkeypatch):
    """M10 lab audit R-1 (record per attempt): a Harbor environment without scoped_exec_env (the 0.23 API) is refused
    before setup, and its record still names agent@version, harness, config digest, Harbor version, model and the
    proxy session, so a by-version query keeps the attempts the version gap broke."""
    monkeypatch.setattr(FakeEnv, "scoped_exec_env", None)
    env = Env(tmp_path)
    h = env.harness()
    _s, rows = await env.run(FakeTask(env.events), harness=h)
    f = _failure(rows[0])
    assert f.kind == "crucible:harness_unsupported" and not f.retryable and "scoped_exec_env" in f.reason
    assert "setup" not in env.events and "run" not in env.events and env.proxy.sessions == {}
    ag = rows[0]["meta"]["agent"]
    assert (ag["agent"], ag["harness"], ag["config_digest"], ag["harbor_version"], ag["model"]) == ("fake-agent@1", h.id, h.digest, "0.24.0", "openai/m")
    assert ag["proxy_session"] and "agent_run" not in rows[0]["meta"]["artifacts"]


async def test_host_keys_an_agent_reads_at_run_time_reach_the_container_blank(tmp_path, monkeypatch):
    """M10 lab audit R-1 (provider keys scrubbed): an agent that reads ``os.environ`` inside ``run`` (cursor-cli's
    CURSOR_API_KEY, claude-code's CLAUDE_CODE_OAUTH_TOKEN) and puts it in a command's own env gets it blank: the scoped
    overlay wins over the command's env. Only the names are recorded, never the values, and nothing is blanked at grade
    time."""
    monkeypatch.setenv("CURSOR_API_KEY", "crsr-real-host-123456")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-real-123456")
    env = Env(tmp_path)

    async def reads_environ(agent, environment):
        await environment.exec("cursor-agent -p x", env={"CURSOR_API_KEY": os.environ["CURSOR_API_KEY"],
                                                         "CLAUDE_CODE_OAUTH_TOKEN": os.environ["CLAUDE_CODE_OAUTH_TOKEN"]})
        assert await agent.call() == 200

    env.hooks["run"] = reads_environ
    task = FakeTask(env.events)
    _s, rows = await env.run(task)
    assert not rows[0]["mask_sample"], rows[0].get("failure")
    [sent] = [x for x in task.sent if x["command"] == "cursor-agent -p x"]
    assert sent["env"]["CURSOR_API_KEY"] == "" and sent["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "" and sent["env"]["OPENAI_API_KEY"].startswith("cap-")
    raw = (env.adir("r", rows[0]) / "agent_run.json").read_text()
    assert {"CURSOR_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"} <= set(json.loads(raw)["scrubbed"])
    assert "crsr-real-host-123456" not in raw and "sk-ant-oat01-real-123456" not in raw
    assert task.graded == [{}]  # no overlay (and no blanks) at grade time


async def test_an_agent_that_fails_before_its_first_model_call_is_not_the_policys(tmp_path):
    """M10 lab audit R-1: an installed agent that exits non-zero before the proxy recorded any exchange (a missing
    binary, a config key it rejects, an unreachable proxy) never let the policy answer: ``unknown``, masked, not a 0."""
    env = Env(tmp_path)

    async def not_found(agent, environment):
        raise NonZeroAgentExitCodeError("Command failed (exit 127): mini-swe-agent: command not found")

    env.hooks["run"] = not_found
    task = FakeTask(env.events)
    _s, rows = await env.run(task)
    f = _failure(rows[0])
    assert f.who == "unknown" and f.detail == "no_upstream_call" and rows[0]["mask_sample"] and task.graded == [], f
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["upstream"]["calls"] == 0 and doc["outcome"]["failure"] == "agent_run_error"


@pytest.mark.parametrize("forged", [404, 402])
async def test_a_forged_upstream_refusal_is_unknown_not_the_upstreams(tmp_path, forged):
    """M10 lab audit R-1: the agent's commands hold the capture key, so the policy can send its own request (a bogus
    model: 404; an unfundable one: 402) and end the agent with that row last. Crucible's probe of the session's first
    model answers 200, so the verdict is unconfirmed: ``unknown``, never ``harness_unsupported`` and never a credits stop
    of the whole run."""
    env = Env(tmp_path, by_model={"bogus": forged})

    async def forge(agent, environment):
        assert await agent.call() == 200
        assert await agent.call("x", model="bogus") == forged
        raise NonZeroAgentExitCodeError("exit 1")

    env.hooks["run"] = forge
    s, rows = await env.run(FakeTask(env.events))
    f = _failure(rows[0])
    verdict = "not_found" if forged == 404 else "credits"
    assert f.who == "unknown" and f.detail == f"unconfirmed:{verdict}" and rows[0]["mask_sample"], f
    assert env.hits == 3 and env.models == ["m", "bogus", "m"]  # the third is Crucible's own probe of the first model
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["upstream"]["probe"] == "ok" and doc["upstream"]["model"] == "m" and doc["upstream"]["verdict"] == verdict
    assert s.stopped is None


async def test_identical_calls_each_settle(tmp_path):
    """M10 lab audit R-1: spend comes from every forwarded row; two identical calls collapse into one exchange for
    training, but both were billed."""
    env = Env(tmp_path)

    async def twice(agent, environment):
        assert await agent.call() == 200
        assert await agent.call() == 200

    env.hooks["run"] = twice
    _s, rows = await env.run(FakeTask(env.events))
    assert env.hits == 2 and rows[0]["cost_usd"] == pytest.approx(10.0)


def _kill_harness(tmp_path, **kw) -> HarborAgentHarness:
    return HarborAgentHarness("mini-swe-agent", "2.4.6", model="openai/m", proxy=CaptureProxy(tmp_path / "cap"), proxy_url="http://10.1.2.3:8788",
                              agent_factory=lambda **k: None, harbor_version="0.24.0", **kw)


def test_the_default_stop_spares_the_init_session_and_marks_the_cohort(tmp_path):
    """M10 lab audit R-1: a non-root agent user's stop never signals the container's init session (a bare ``pkill -u``
    killed a keep-alive running as that user, the container stopped and was graded the policy's ``agent_broke_env``); a
    root agent with a phase mark is stopped by start time; without one, its own tree. Review round 2: the user and cohort
    stops need no procps and no binary from the policy's PATH for /proc, skip zombies, and exit non-zero when processes
    survive (``left:``) or cannot be listed; the tree stop refuses without pgrep; in place only the agent's own tree."""
    from crucible.interop.harbor.agent_harness import AGENT_KILL

    h = _kill_harness(tmp_path)
    cmd = h._kill_command(SimpleNamespace(agent_user="agent"))
    assert "pkill" not in cmd and "pgrep" not in cmd and "cat " not in cmd and '[ "$4" = 1 ]' in cmd
    assert "/proc/[0-9]*" in cmd and "(Uid:)" in cmd and "id -u agent" in cmd and "exit 3" in cmd
    assert "Z|X|x) continue" in cmd and "left:$l" in cmd and cmd.rstrip().endswith('[ -z "$o" ]') and not cmd.rstrip().endswith("true")
    # review round 3: a start marker, then a report line that sorts the survivors (left last); D is uninterruptible
    assert cmd.startswith("printf '%s\\n' CRUCIBLE-STOP-BEGIN;") and 'D) w="$w $p"' in cmd
    assert 'echo "killed:$k respawned:$r uninterruptible:$w left:$l"' in cmd
    assert "u=1000" in h._kill_command(SimpleNamespace(agent_user="1000"))  # a numeric user is its own uid
    # review round 3: the uid resolved before the phase is used, so no id the policy planted runs
    resolved = h._kill_command(SimpleNamespace(agent_user="agent"), uid=1001)
    assert "case 1001 in" in resolved and "u=1001" in resolved and "id -u agent" not in resolved
    assert AGENT_KILL["mini-swe-agent"].startswith("command -v pgrep") and "exit 3" in AGENT_KILL["mini-swe-agent"]
    for root in ("root", 0, None):
        assert h._kill_command(SimpleNamespace(agent_user=root)) == AGENT_KILL["mini-swe-agent"]
        cohort = h._kill_command(SimpleNamespace(agent_user=root), 77)
        assert "-ge 77" in cohort and "case $p in 1|$$|$PPID)" in cohort and "kill -KILL -1" not in cohort
        assert "pgrep" not in cohort and "cat " not in cohort and "Z|X|x) continue" in cohort and cohort.rstrip().endswith('[ -z "$o" ]')
        # review round 3: a survivor's blame follows its parent: one older than the mark (not PID 1) respawned it
        assert "q=$2" in cohort and "/proc/$q/stat" in cohort and '[ "${20:-77}" -lt 77 ]' in cohort and 'r="$r $p"' in cohort
        assert "respawned:$r" in cohort and "left:$l" in cohort and 'D) w="$w $p"; continue' in cohort
    for user in ("agent", None, "root"):  # in place: the agent's own tree, never the user or cohort stop
        assert h._kill_command(SimpleNamespace(agent_user=user), 77, in_place=True) == AGENT_KILL["mini-swe-agent"]
    assert _kill_harness(tmp_path / "never", kill_command="")._kill_command(SimpleNamespace(agent_user="agent"), 77) is None  # "": no stop
    assert _kill_harness(tmp_path / "own", kill_command="stop-it")._kill_command(SimpleNamespace(agent_user="agent"), 77, in_place=True) == "stop-it"


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("sh"), reason="a POSIX shell")
def test_the_stop_commands_parse_in_a_posix_shell(tmp_path):
    """M10 lab audit R-1: the stops run under ``bash -c`` (Harbor) and ``sh -c`` (the redteam containers): each parses
    (``-n``: nothing is executed, so nothing on this host is killed)."""
    from crucible.interop.harbor.agent_harness import AGENT_KILL

    from crucible.interop.harbor.patch_task import _hook_cmd

    h = _kill_harness(tmp_path)
    for cmd in (h._kill_command(SimpleNamespace(agent_user="agent")), h._kill_command(SimpleNamespace(agent_user="1000")),
                h._kill_command(SimpleNamespace(agent_user="agent"), uid=1001), h._kill_command(SimpleNamespace(agent_user=None), 77),
                AGENT_KILL["mini-swe-agent"], _hook_cmd("cd /app && git diff base > /logs/artifacts/model.patch # a trailing comment")):
        for shell in [s for s in ("sh", "bash") if shutil.which(s)]:
            r = subprocess.run([shell, "-n", "-c", cmd], capture_output=True, text=True, timeout=30, check=False)
            assert r.returncode == 0, (shell, cmd, r.stderr)


def test_survivors_are_read_from_the_last_report_line():
    from crucible.interop.harbor.agent_harness import _survivors

    assert _survivors("killed: 12 13 left:") == [] and _survivors("") == [] and _survivors("killed: 5") == []
    assert _survivors("WARN[0000] the attribute `version` is obsolete\nkilled: 12 13 left: 14 15\n") == [14, 15]
    # review round 3: respawned children of an older service and processes in state D are reported apart, never as left
    from crucible.interop.harbor.agent_harness import _respawned, _stop_reported, _uninterruptible
    line = "CRUCIBLE-STOP-BEGIN\nkilled: 4300 respawned: 4330 4331 uninterruptible: 4340 left:"
    assert _survivors(line) == [] and _respawned(line) == [4330, 4331] and _uninterruptible(line) == [4340]
    assert _respawned("killed: 12 left: 14") == [] and _uninterruptible("killed: 12 left: 14") == []
    # a report counts only with the exit the stop gives it: a planted program that prints one and kills the shell does not
    assert _stop_reported(line, 1) and not _stop_reported(line, 137) and not _stop_reported("killed: left:", 1)
    assert _stop_reported("no /proc", 3) and not _stop_reported("CRUCIBLE-STOP-BEGIN\n", 137) and not _stop_reported(CONNECT_ERROR, 255)


async def test_the_phase_mark_skips_merged_warnings(tmp_path):
    """M10 lab audit R-1: Harbor's exec merges stderr into stdout; a warning before the mark no longer drops the cohort
    stop (``int()`` of the whole output raised, and the stop fell back to the agent's tree)."""
    h = _kill_harness(tmp_path)

    def world(stdout: str, rc: int = 0) -> SimpleNamespace:
        async def exec(command, cwd=None, env=None, timeout_sec=None, user=None):
            return SimpleNamespace(return_code=rc, stdout=stdout, stderr=None)

        return SimpleNamespace(backend=SimpleNamespace(env=SimpleNamespace(exec=exec)))

    assert await h._phase_mark(world("bash: warning: setlocale: LC_ALL: cannot change locale (xx_XX.UTF-8)\n4242\n")) == 4242
    assert await h._phase_mark(world("4242\n", rc=255)) is None and await h._phase_mark(world("error during connect\n")) is None


async def test_a_root_agent_past_the_deadline_is_stopped_by_its_cohort_before_collect(tmp_path):
    """M10 lab audit R-1: no per-agent pattern was needed: the phase's cohort (by start time) is killed as root before
    the patch is collected, so no work done after Crucible's deadline is collected."""
    env = Env(tmp_path)

    async def late(agent, environment):
        assert await agent.call() == 200
        await asyncio.sleep(5)

    env.hooks["run"] = late
    task = FakePatchTask(env.events)
    task.mark = 4242
    _s, rows = await env.run(task, harness=env.harness(limits=AgentLimits(timeout_s=0.5)))
    assert rows[0]["meta"]["stop"] == "timeout" and not rows[0]["mask_sample"], rows[0].get("failure")
    [kill] = [x for x in task.sent if "-ge 4242" in x["command"]]
    assert kill["user"] == "root" and kill["cwd"] == "/"  # review round 3: never in a working directory the policy can remove
    [mark] = [x for x in task.sent if x["command"].startswith("cut -d' ' -f22")]
    assert mark["cwd"] == "/" and mark["user"] == "root"
    ev = env.events
    assert ev.index("run") < ev.index(f"exec:{kill['command']}") < ev.index("collect")
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["status"] == "ran" and doc["processes"]["mark"] == 4242
    assert doc["processes"]["scope"] == "phase" and doc["processes"]["left"] == []


async def _late(agent, environment):
    assert await agent.call() == 200
    await asyncio.sleep(5)


def _gone_at_the_gate(monkeypatch) -> list[int]:
    """M10 lab audit R-1 review round 3: the host answers "alive" to the evidence read and "gone" to every later ask (the
    container died between the evidence read and the collect gate), so only the gate's own host check, made before it
    blames the stop on the policy, can make the attempt ``crucible:env_container_lost``. A constant answer never reaches
    the gate: a deadline end asks the host at the evidence read first. FakeWorld binds ``container_lost`` to ``_lost`` when
    the world opens, after this patch. Returns the asks."""
    asks: list[int] = []

    async def gone_by_the_gate(self):
        asks.append(1)
        return "the task container was removed" if len(asks) > 1 else None

    monkeypatch.setattr(FakeWorld, "_lost", gone_by_the_gate)
    return asks


async def test_a_patch_is_never_collected_from_an_agent_that_could_not_be_stopped(tmp_path):
    """M10 lab audit R-1: past the deadline with the stop failing, the collect would race a live agent: masked
    ``backend_error`` (not retried), never the policy's, and nothing collected or graded."""
    env = Env(tmp_path)
    env.hooks["run"] = _late
    task = FakePatchTask(env.events)
    task.mark, task.kill_raises = 4242, True
    _s, rows = await env.run(task, harness=env.harness(limits=AgentLimits(timeout_s=0.5)))
    f = _failure(rows[0])
    assert (f.kind, f.detail, f.stage) == ("crucible:backend_error", "agent_stop", "collect") and rows[0]["mask_sample"] and not f.retryable, f
    assert "collect" not in env.events and task.patch_grades == 0


@pytest.mark.parametrize("user, mark, rc, out, who", [
    ("agent", None, 3, "no uid", "infra"),  # the user stop could not resolve the user
    (None, 4242, 3, "no /proc", "infra"),  # the cohort stop could not list the processes
    # review round 3: no report and nothing showing whose (no start marker, no runtime refusal): unknown, never infra
    (None, 4242, 127, "bash: line 1: kill: command not found", "unknown"),
    (None, 4242, 255, CONNECT_ERROR, "unknown"),  # a docker or ssh failure twice running is the same in band as a replaced shell
])
async def test_a_stop_that_could_not_list_the_processes_masks_the_collect(tmp_path, user, mark, rc, out, who):
    """M10 lab audit R-1 review round 2: the stops exited 0 whatever they found (``; true``) and ``_stop_agent`` reported
    any returned exec as "ran", so a user stop without pgrep (a slim image) let the patch be collected from a live agent.
    A non-zero exit is "failed": past the deadline the collect is masked, never retried, never collected. Review round 3:
    one with its own exit-3 reason is infrastructure; one that never reported is run once more and stays unknown."""
    env = Env(tmp_path)
    env.hooks["run"] = _late
    task = FakePatchTask(env.events)
    task.mark, task.agent_user, task.kill = mark, user, {"rc": rc, "stdout": out}
    _s, rows = await env.run(task, harness=env.harness(limits=AgentLimits(timeout_s=0.5)))
    f = _failure(rows[0])
    assert (f.kind, f.detail, f.stage) == ("crucible:backend_error", "agent_stop", "collect") and rows[0]["mask_sample"] and not f.retryable, f
    assert f.who == who and "collect" not in env.events and task.patch_grades == 0
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["status"] == "failed" and doc["processes"]["rc"] == rc and doc["processes"]["left"] == []
    assert sum("killed:" in x["command"] for x in task.sent) == (1 if rc == 3 else 2)  # an unreported stop runs once more


@pytest.mark.parametrize("kill, probe_dead", [
    ({"rc": 137, "stdout": "CRUCIBLE-STOP-BEGIN\n"}, False),  # a planted sleep that kills its shell: started, killed, a root shell answers
    ({"rc": 137, "stdout": ""}, False),  # no marker and no runtime refusal: nothing shows whose (unknown, below)
    ({"rc": 126, "stdout": 'OCI runtime exec failed: exec failed: unable to start container process: exec: "bash": executable file not '
                           "found in $PATH: unknown\n"}, True),  # bash removed: runc's own refusal, whatever the probe
])
async def test_a_root_stop_that_never_reports_in_a_live_container_is_the_agents(tmp_path, monkeypatch, kill, probe_dead):
    """M10 lab audit R-1 review round 3: a root policy controls ``bash`` and the ``sleep`` the stop calls (and could make
    the stop fail fast instead of stalling: a planted ``sleep`` running ``kill -9 $PPID``, or ``rm /bin/bash``), which was
    masked as infra (``agent_stop``), uncounted. A built-in stop that never prints its report is run once more; still
    unreported, it is ``crucible:agent_broke_env`` when its shell started both times while a fresh root shell answers, or
    when runc could not start it; otherwise unknown."""
    env = Env(tmp_path)
    env.hooks["run"] = _late
    task = FakePatchTask(env.events)
    task.mark, task.kill, task.probe_dead = 4242, kill, probe_dead
    _s, rows = await env.run(task, harness=env.harness(limits=AgentLimits(timeout_s=0.5)))
    f = _failure(rows[0])
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert sum("killed:" in x["command"] for x in task.sent) == 2  # run once more first
    if kill["stdout"] == "":  # no marker and no runtime refusal: nothing shows whose, so unknown, never the policy's
        assert (f.kind, f.detail, f.who) == ("crucible:backend_error", "agent_stop", "unknown") and rows[0]["mask_sample"], f
        assert doc["processes"]["status"] == "failed" and doc["processes"]["unreported"]
    else:
        assert (f.kind, f.stage, f.who) == ("crucible:agent_broke_env", "collect", "policy") and not rows[0]["mask_sample"], f
        assert doc["processes"]["status"] == "broken" and rows[0]["reward"] == 0.0
    assert "collect" not in env.events and task.patch_grades == 0

    dead = Env(tmp_path / "dead")  # started, killed, and no fresh shell answers either: unknown
    dead.hooks["run"] = _late
    task = FakePatchTask(dead.events)
    task.mark, task.kill, task.probe_dead = 4242, {"rc": 137, "stdout": "CRUCIBLE-STOP-BEGIN\n"}, True
    _s, rows = await dead.run(task, harness=dead.harness(limits=AgentLimits(timeout_s=0.5)))
    f = _failure(rows[0])
    assert (f.kind, f.detail, f.who) == ("crucible:backend_error", "agent_stop", "unknown") and rows[0]["mask_sample"], f

    gone = Env(tmp_path / "gone")  # already gone at the evidence read: classified there (stage agent), whatever the stop did
    gone.hooks["run"] = _late
    task = FakePatchTask(gone.events, lost="the task container was removed")
    task.mark, task.kill = 4242, {"rc": 137, "stdout": "CRUCIBLE-STOP-BEGIN\n"}
    _s, rows = await gone.run(task, harness=gone.harness(limits=AgentLimits(timeout_s=0.5)))
    assert (_failure(rows[0]).kind, _failure(rows[0]).stage) == ("crucible:env_container_lost", "agent") and rows[0]["mask_sample"]

    asks = _gone_at_the_gate(monkeypatch)  # alive at the evidence read, gone at the collect gate: the gate asks the host first
    late = Env(tmp_path / "late")
    late.hooks["run"] = _late
    task = FakePatchTask(late.events)
    task.mark, task.kill = 4242, {"rc": 137, "stdout": "CRUCIBLE-STOP-BEGIN\n"}
    _s, rows = await late.run(task, harness=late.harness(limits=AgentLimits(timeout_s=0.5)))
    f = _failure(rows[0])
    assert (f.kind, f.stage) == ("crucible:env_container_lost", "collect") and rows[0]["mask_sample"] and "could not be stopped" in f.reason, f
    assert len(asks) == 2 and "collect" not in late.events and task.patch_grades == 0
    doc = json.loads((late.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["status"] == "broken" and doc["container_lost"] is None


async def test_a_stop_that_raises_a_fast_timeout_is_a_failed_stop_not_a_stall(tmp_path):
    """M10 lab audit R-1 review round 3: any TimeoutError was read as the stop outliving its deadline (``stalled``, the
    policy's); a provider SDK's fast connect timeout (aiohttp, a socket) is host side: a failed stop, masked infra."""
    env = Env(tmp_path)
    env.hooks["run"] = _late
    task = FakePatchTask(env.events)
    task.mark, task.kill = 4242, {"timeout_now": True}
    _s, rows = await env.run(task, harness=env.harness(limits=AgentLimits(timeout_s=0.5)))
    f = _failure(rows[0])
    assert (f.kind, f.detail, f.stage, f.who) == ("crucible:backend_error", "agent_stop", "collect", "infra") and rows[0]["mask_sample"], f
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["status"] == "failed" and "TimeoutError" in doc["processes"]["error"]
    assert "collect" not in env.events and task.patch_grades == 0


@pytest.mark.parametrize("deadline", [True, False])
async def test_children_a_pre_phase_service_respawns_are_never_the_policys(tmp_path, deadline):
    """M10 lab audit R-1 review round 3: a service older than the phase (a postmaster's crash restart, an nginx master)
    forks a fresh worker every time the stop kills one; those were ``left`` and scored the honest attempt 0 as
    ``crucible:policy_misbehavior``. They are still killed, reported as ``respawned`` and never blamed: a run that ended
    on its own is collected; one past the deadline masks (the stop did not take effect)."""
    env = Env(tmp_path)
    env.hooks["run"] = _late if deadline else _one_call
    task = FakePatchTask(env.events, reward=0.5)
    task.mark, task.kill = 4242, {"rc": 1, "stdout": "CRUCIBLE-STOP-BEGIN\nkilled: 4300 4310 4320 respawned: 4330 uninterruptible: left:"}
    kw = {"limits": AgentLimits(timeout_s=0.5)} if deadline else {}
    _s, rows = await env.run(task, harness=env.harness(**kw))
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["respawned"] == [4330] and doc["processes"]["left"] == [] and doc["processes"]["status"] == "failed"
    assert sum("killed:" in x["command"] for x in task.sent) == 1  # a reported stop is not run again
    if deadline:
        f = _failure(rows[0])
        assert (f.kind, f.detail, f.who) == ("crucible:backend_error", "agent_stop", "infra") and rows[0]["mask_sample"], f
        assert "collect" not in env.events
    else:
        assert rows[0]["reward"] == 0.5 and not rows[0]["mask_sample"] and "collect" in env.events, rows[0].get("failure")


@pytest.mark.parametrize("out, kind", [
    ("killed: 4300 respawned: uninterruptible: 4302 left:", "crucible:backend_error"),  # only D-state: exiting, masked
    ("killed: 4300 respawned: uninterruptible: 4302 left: 4301", "crucible:policy_misbehavior"),  # a runnable survivor too
])
async def test_processes_still_exiting_in_uninterruptible_sleep_are_not_the_policys(tmp_path, out, kind):
    """M10 lab audit R-1 review round 3: a process that took SIGKILL but is still in state D (a CUDA context's release)
    is exiting, not resisting; it was ``left`` and scored as ``crucible:policy_misbehavior``. The stop waits for it, then
    reports it as ``uninterruptible``: masked, never retried (a retry re-runs the agent), nothing collected."""
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    task = FakePatchTask(env.events)
    task.mark, task.kill = 4242, {"rc": 1, "stdout": out}
    s, rows = await env.run(task, retry=RetryRule(max_attempts=3))
    f = _failure(rows[0])
    assert f.kind == kind and f.stage == "collect" and s.attempts == 1, f
    if kind == "crucible:backend_error":
        assert f.detail == "agent_stop_uninterruptible" and rows[0]["mask_sample"] and not f.retryable and f.who != "policy"
    else:
        assert f.who == "policy" and not rows[0]["mask_sample"] and "4301" in f.reason
    assert "collect" not in env.events and task.patch_grades == 0
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["uninterruptible"] == [4302]


async def test_a_named_agent_users_uid_is_resolved_before_the_phase(tmp_path, monkeypatch):
    """M10 lab audit R-1 review round 3: the user stop called ``id -u <user>`` after the phase, a binary a root policy could
    plant (one that exits non-zero made the stop's "no uid" an infra mask); the uid is resolved as root in ``/`` before the
    agent runs and passed to the stop."""
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    task = FakePatchTask(env.events)
    task.agent_user = "agent"
    real = FakeEnv.exec
    asked: list[tuple] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if command == "id -u agent":
            asked.append((cwd, user))
            self.events.append(f"exec:{command}")
            return SimpleNamespace(return_code=0, stdout="1001\n", stderr=None)
        return await real(self, command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    monkeypatch.setattr(FakeEnv, "exec", exec)
    _s, rows = await env.run(task)
    assert not rows[0]["mask_sample"], rows[0].get("failure")
    ev = env.events
    [stop] = [e for e in ev if "(Uid:)" in e]
    assert asked == [("/", "root")] and ev.index("exec:id -u agent") < ev.index("run") < ev.index(stop)
    assert "case 1001 in" in stop and "id -u agent" not in stop


@pytest.mark.parametrize("deadline", [True, False])
async def test_processes_that_survive_the_stop_are_the_policys_and_nothing_is_collected(tmp_path, deadline):
    """M10 lab audit R-1 review round 2: ``left:`` was printed and never read. Processes of the agent's phase still running
    after the stop's rounds of SIGKILL as root (zombies excluded: a fork chain) are the policy's
    (``crucible:policy_misbehavior``, scored 0), whether or not its run ended on its own, and the patch is not collected."""
    env = Env(tmp_path)
    env.hooks["run"] = _late if deadline else _one_call
    task = FakePatchTask(env.events)
    task.mark, task.kill = 4242, {"rc": 1, "stdout": "killed: 4300 4301 left: 4302"}
    kw = {"limits": AgentLimits(timeout_s=0.5)} if deadline else {}
    s, rows = await env.run(task, harness=env.harness(**kw), retry=RetryRule(max_attempts=3))
    f = _failure(rows[0])
    assert (f.kind, f.stage, f.who) == ("crucible:policy_misbehavior", "collect", "policy") and not rows[0]["mask_sample"], f
    assert rows[0]["reward"] == 0.0 and s.attempts == 1 and "4302" in f.reason
    assert "collect" not in env.events and task.patch_grades == 0
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["left"] == [4302] and doc["processes"]["status"] == "failed"


async def test_a_stop_that_stalls_in_a_live_container_is_the_agents(tmp_path, monkeypatch):
    """M10 lab audit R-1 review round 2: a root policy can make the in-container stop hang (a planted binary, a flood of
    processes); that masked the attempt as infra (``backend_error`` / ``agent_stop``) instead of scoring its 0. A stop that
    outlives its deadline in a live container is ``crucible:agent_broke_env``. Review round 3: so is one that never reports
    (``test_a_root_stop_that_never_reports_in_a_live_container_is_the_agents``); a fast host-side failure stays masked
    (``test_a_stop_that_raises_a_fast_timeout_is_a_failed_stop_not_a_stall``)."""
    from crucible.interop.harbor import agent_harness

    monkeypatch.setattr(agent_harness, "_STOP_TIMEOUT_S", 0.05)
    env = Env(tmp_path)
    env.hooks["run"] = _late
    task = FakePatchTask(env.events)
    task.mark, task.kill = 4242, {"stall": 0.2}
    _s, rows = await env.run(task, harness=env.harness(limits=AgentLimits(timeout_s=0.5)))
    f = _failure(rows[0])
    assert (f.kind, f.stage, f.who) == ("crucible:agent_broke_env", "collect", "policy") and not rows[0]["mask_sample"], f
    assert "collect" not in env.events and task.patch_grades == 0
    assert json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())["processes"]["status"] == "stalled"

    gone = Env(tmp_path / "gone")  # already gone at the evidence read: classified there (stage agent), whatever the stop did
    gone.hooks["run"] = _late
    task = FakePatchTask(gone.events, lost="the task container was removed")
    task.mark, task.kill = 4242, {"stall": 0.2}
    _s, rows = await gone.run(task, harness=gone.harness(limits=AgentLimits(timeout_s=0.5)))
    assert (_failure(rows[0]).kind, _failure(rows[0]).stage) == ("crucible:env_container_lost", "agent") and rows[0]["mask_sample"]

    # review round 3: alive at the evidence read, gone at the collect gate. Only the gate's own host check, made before the
    # stall is blamed on the policy, makes this the infrastructure's (the variant above never reaches the gate)
    asks = _gone_at_the_gate(monkeypatch)
    late = Env(tmp_path / "late")
    late.hooks["run"] = _late
    task = FakePatchTask(late.events)
    task.mark, task.kill = 4242, {"stall": 0.2}
    _s, rows = await late.run(task, harness=late.harness(limits=AgentLimits(timeout_s=0.5)))
    f = _failure(rows[0])
    assert (f.kind, f.stage) == ("crucible:env_container_lost", "collect") and rows[0]["mask_sample"] and "could not be stopped" in f.reason, f
    assert len(asks) == 2 and "collect" not in late.events and task.patch_grades == 0
    doc = json.loads((late.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["status"] == "stalled" and doc["container_lost"] is None


@pytest.mark.parametrize("user", [None, "agent"])
async def test_an_in_place_grade_follows_no_phase_or_user_stop(tmp_path, user):
    """M10 lab audit R-1 review round 2: in place the harness adds no phase (cohort) or user stop before the grade; it
    stops only the agent's own tree. M10 lab audit R-2: the kill of every other process started after ``open`` is the
    guarded grade's own host-side quiesce (``HarborInPlaceTask._verify``), decided by the host's listing; a cohort or
    user stop here would only repeat it with in-container evidence a root policy can forge."""
    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    task = FakeTask(env.events)
    task.mark, task.agent_user = 77, user
    _s, rows = await env.run(task)
    assert rows[0]["reward"] == 1.0 and not rows[0]["mask_sample"], rows[0].get("failure")
    ev = env.events
    assert not any("-ge 77" in e or "(Uid:)" in e or "killed:" in e for e in ev[:ev.index("grade")]), ev
    doc = json.loads((env.adir("r", rows[0]) / "agent_run.json").read_text())
    assert doc["processes"]["scope"] == "agent_tree" and doc["processes"]["status"] == "skipped" and doc["processes"]["mark"] == 77


async def test_r1_harness_maps_a_quiesce_survivor_to_policy_misbehavior(tmp_path):
    """M10 lab audit R-2: the harness's shared-verifier seam is the guarded grade itself, which quiesces first. Its
    survivors report (``HarborInPlaceTask._zero`` with ``meta.failure`` ``crucible:policy_misbehavior`` and the
    ``policy_process_survived`` flag, the shape ``HarborInPlaceTask._quiesce`` returns) becomes the attempt's scored
    failure: reward 0, never masked, never retried, and the flag reaches the rollout's flags (bench's hack metrics)."""
    from crucible.interop.harbor import quiesce
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    receipt = quiesce.Receipt(before=[], killed=[], rounds=quiesce.ROUNDS, method=quiesce.METHOD, verdict="survivors", mark=77, pid1=4000, helper=[],
                              survivors=[{"pid": 4301, "ppid": 1, "stat": "S", "etimes": 3, "args": "sh -c crucible-reward-writer", "why": "left"}])
    summary = receipt.summary(None)

    class SurvivorTask(FakeTask):
        async def grade(self, world, gdir):
            self.events.append("grade")
            return HarborInPlaceTask._zero(self.id, time.monotonic(), "crucible:policy_misbehavior",
                                           "1 process(es) the policy started still ran after 5 host-side kill rounds, so the grade did "
                                           "not start (M10 lab audit R-2, P23): 4301 sh -c crucible-reward-writer",
                                           flags=[quiesce.flag(receipt)], quiesce=summary)

    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    s, rows = await env.run(SurvivorTask(env.events), retry=RetryRule(max_attempts=3))
    f = _failure(rows[0])
    assert (f.kind, f.stage, f.who) == ("crucible:policy_misbehavior", "verification", "policy") and not f.masks, f
    assert not rows[0]["mask_sample"] and rows[0]["reward"] == 0.0 and s.attempts == 1, rows[0]
    a = env.adir("r", rows[0])
    ro = json.loads((a / "rollout.json").read_text())
    assert any(x["rule"] == "policy_process_survived" and "crucible-reward-writer" in x["evidence"] for x in ro.get("flags") or []), ro.get("flags")
    rep = json.loads((a / "verifier" / "report.json").read_text())
    assert rep["meta"]["quiesce"]["verdict"] == "survivors" and rep["meta"]["failure"]["kind"] == "crucible:policy_misbehavior"


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks and FIFOs")
async def test_links_the_policy_plants_under_logs_are_never_followed(tmp_path, monkeypatch):
    """M10 lab audit R-1: /logs/agent is the policy's to write. A link it plants at trajectory.json (Harbor's converter
    writes there) and a FIFO are dropped from the host-only copy the converter and every read work on: the link's host
    target is never written or read, and nothing blocks."""
    victim = tmp_path / "victim.json"
    victim.write_text('{"host": "secret"}')
    env = Env(tmp_path / "run")

    def convert(self, context):
        self.env_.events.append("populate")
        (self.logs_dir / "trajectory.json").write_text(json.dumps(_atif(1)))

    monkeypatch.setattr(FakeAgent, "populate_context_post_run", convert)

    async def plant(agent, environment):
        assert await agent.call() == 200
        (agent.logs_dir / "trajectory.json").symlink_to(victim)
        os.mkfifo(agent.logs_dir / "fifo")

    env.hooks["run"] = plant
    _s, rows = await env.run(FakeTask(env.events))
    assert not rows[0]["mask_sample"], rows[0].get("failure")
    assert victim.read_text() == '{"host": "secret"}'
    a = env.adir("r", rows[0])
    assert env.agents[0].logs_dir == a / "agent_logs"
    atif = (a / "agent_trajectory.atif.json").read_text()
    assert len(json.loads(atif)["steps"]) == 1 and "secret" not in atif
    notes = " ".join(json.loads((a / "agent_run.json").read_text())["notes"])
    assert "trajectory.json" in notes and "fifo" in notes


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks")
def test_copy_untrusted_logs_copies_files_and_drops_links(tmp_path):
    from crucible.interop.harbor.agent_harness import copy_untrusted_logs

    src, victim = tmp_path / "src", tmp_path / "victim"
    (src / "sub").mkdir(parents=True)
    victim.mkdir()
    (src / "a.txt").write_text("a")
    (src / "sub" / "b.txt").write_text("b")
    (src / "big").write_bytes(b"x" * 64)
    (src / "link").symlink_to(victim / "x")
    (src / "dirlink").symlink_to(victim)
    dest = tmp_path / "dest"
    dropped = copy_untrusted_logs(src, dest, max_bytes=32)
    assert (dest / "a.txt").read_text() == "a" and (dest / "sub" / "b.txt").read_text() == "b"
    assert not (dest / "link").exists() and not (dest / "dirlink").exists() and not (dest / "big").exists()
    assert {d.split(":")[0] for d in dropped} == {"big", "dirlink", "link"}
    assert copy_untrusted_logs(src, dest, max_bytes=32) == dropped  # an existing copy is replaced, not merged
    assert not list(victim.iterdir())


def test_env_views_name_every_source():
    agent = SimpleNamespace(extra_env={"OPENAI_API_KEY": "cap-x"}, model_connection=SimpleNamespace(env={"OPENAI_API_KEY": HOST_KEY}))
    views = _agent_env_views(agent)
    assert views == {"extra_env:OPENAI_API_KEY": "cap-x", "model_connection:OPENAI_API_KEY": HOST_KEY}  # one source never hides another
    assert leaked_host_secrets(views, allow={"cap-x"}, environ={"OPENAI_API_KEY": HOST_KEY}) == ["model_connection:OPENAI_API_KEY"]


# ------------------------------------------------------------------ part A: the patch task's logs and collect, both branches


class _TrialPaths:
    def __init__(self, root: Path):
        self.agent_dir, self.artifacts_dir = root / "agent", root / "artifacts"

    def mkdir(self):
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


NOISE = ("bash: warning: setlocale: LC_ALL: cannot change locale (xx_XX.UTF-8)\n"
         "WARN[0000] /t/docker-compose.yaml: the attribute `version` is obsolete, it will be ignored, please remove it\n")
CONNECT_ERROR = 'error during connect: Get "http://docker.example/v1.47/containers/x/json": ssh: connect to host lab port 22: Connection reset\n'
# runc's refusals to start a process in the container's present state (review round 3): bash removed, the working directory removed
OCI_NO_BASH = ('OCI runtime exec failed: exec failed: unable to start container process: exec: "bash": executable file not found in $PATH: '
               "unknown\n")
OCI_NO_CWD = ('OCI runtime exec failed: exec failed: unable to start container process: chdir to cwd ("/app") set in config.json failed: '
              "no such file or directory: unknown\n")


def _framed(content: str | None, rc: int = 0) -> str:
    """What ``_patch_read_cmd`` prints in the container: the frame, the size and base64 lines (76 columns, as base64(1))
    for verdict 0, and the verdict."""
    from crucible.interop.harbor.patch_task import _BEGIN, _END, _SIZE

    if rc != 0 or content is None:
        return f"\n{_BEGIN}\n\n{_END} rc={rc}\n"
    raw = content.encode()
    return f"\n{_BEGIN}\n{_SIZE} {len(raw)}\n{base64.encodebytes(raw).decode()}\n{_END} rc=0\n"


class _HarborEnv:
    """The download / exec surface of a Harbor environment, recording every call. Like Harbor's Docker exec
    (docker.py:742-743, 1441-1446) it returns stderr merged into stdout (``stderr=None``) and a docker or ssh failure as a
    non-zero exit, not an exception. The patch is read through the container (``base64 <``, as root): ``content`` is what
    the container holds; ``symlink`` plants a link at the patch path (also on the host, pointing at ``victim``: the host
    must never read it); ``noise`` comes before every output (compose and shell warnings); ``read_raises`` makes that
    many reads raise; ``read_rc_fault`` makes that many reads return rc 255 with the docker CLI's error and no verdict;
    ``read_stalls`` sleeps that long, then raises Harbor's deadline error; ``read_garbage`` returns verdict 0 around a body
    that is not the patch (a replaced ``base64``); ``hook_raises``: "deadline" sleeps past the hook's deadline then raises
    Harbor's RuntimeError, "now" raises at once, "timeout_now" raises a provider's fast TimeoutError; ``hook_rc_fault``
    returns rc 255 before the hook's shell started.

    Review round 3: a hook is framed (``_hook_cmd``: its marker, then its own ``rc`` verdict) and found by its command
    inside the frame. ``hook_cut`` returns rc 255 with the marker already streamed and the docker CLI's error after it (an
    ssh reset mid-hook); ``hook_killed`` rc 137 after the marker (its shell killed by a signal in the container);
    ``read_timeouts`` makes that many reads raise a fast TimeoutError; ``read_killed`` makes that many reads print the
    frame's start and size, then return 137 (a planted ``base64`` that kills its shell); ``shell_replaced`` answers every
    exec rc 1 with nothing (a stub ``bash``); ``no_bash`` answers every exec with runc's refusal (``bash`` removed);
    ``workdir_gone`` answers every exec outside ``/`` with runc's chdir refusal (the policy removed the working directory);
    ``ctx_broken`` answers every exec outside ``/`` rc 1 with nothing (the hook's context broken, a root shell in ``/``
    fine). A fresh-shell probe (``printf '%s\\n' CRUCIBLE-PROBE-<nonce>``) gets its nonce back under those rules, unless
    ``probe_dead``. Every exec is recorded as ``("exec", command, user, cwd)``."""

    def __init__(self, root: Path, *, hook_rc: int = 0, patch_exists: bool = True, content: str = "the-diff\n", symlink: bool = False,
                 read_raises: int = 0, hook_raises: str | None = None, victim: Path | None = None, noise: str = "", read_rc_fault: int = 0,
                 read_stalls: float = 0.0, read_garbage: bool = False, hook_rc_fault: bool = False, hook_cut: bool = False,
                 hook_killed: bool = False, read_timeouts: int = 0, read_killed: int = 0, shell_replaced: bool = False, no_bash: bool = False,
                 workdir_gone: bool = False, ctx_broken: bool = False, probe_dead: bool = False):
        self.trial_paths = _TrialPaths(root)
        self.calls: list[tuple] = []
        self.hook_rc, self.patch_exists, self.content, self.symlink = hook_rc, patch_exists, content, symlink
        self.read_raises, self.hook_raises, self.victim = read_raises, hook_raises, victim
        self.noise, self.read_rc_fault, self.read_stalls, self.read_garbage, self.hook_rc_fault = noise, read_rc_fault, read_stalls, read_garbage, hook_rc_fault
        self.hook_cut, self.hook_killed, self.read_timeouts, self.read_killed = hook_cut, hook_killed, read_timeouts, read_killed
        self.shell_replaced, self.no_bash, self.workdir_gone, self.ctx_broken, self.probe_dead = shell_replaced, no_bash, workdir_gone, ctx_broken, probe_dead

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from crucible.interop.harbor.patch_task import _BEGIN, _END, _HOOK_BEGIN, _HOOK_END, _SIZE

        self.calls.append(("exec", command, user, cwd))
        if self.no_bash:  # runc cannot start bash: before any shell runs, whatever the command
            return SimpleNamespace(return_code=126, stdout=self.noise + OCI_NO_BASH, stderr=None)
        if self.workdir_gone and cwd != "/":
            return SimpleNamespace(return_code=126, stdout=self.noise + OCI_NO_CWD, stderr=None)
        if self.shell_replaced or (self.ctx_broken and cwd != "/"):
            return SimpleNamespace(return_code=1, stdout="", stderr=None)
        if "CRUCIBLE-PROBE-" in command:
            if self.probe_dead:
                return SimpleNamespace(return_code=255, stdout=self.noise + CONNECT_ERROR, stderr=None)
            return SimpleNamespace(return_code=0, stdout=self.noise + command.split()[-1] + "\n", stderr=None)
        if "base64 <" in command:
            if self.read_stalls:
                await asyncio.sleep(self.read_stalls)
                raise RuntimeError("Command timed out after 300 seconds")
            if self.read_raises > 0:
                self.read_raises -= 1
                raise RuntimeError("docker exec failed")
            if self.read_timeouts > 0:
                self.read_timeouts -= 1
                raise TimeoutError("connect timed out")
            if self.read_killed > 0:
                self.read_killed -= 1
                return SimpleNamespace(return_code=137, stdout=f"{self.noise}\n{_BEGIN}\n{_SIZE} 9\n", stderr=None)
            if self.read_rc_fault > 0:
                self.read_rc_fault -= 1
                return SimpleNamespace(return_code=255, stdout=self.noise + CONNECT_ERROR, stderr=None)
            if self.read_garbage:
                return SimpleNamespace(return_code=0, stdout=f"{self.noise}\n{_BEGIN}\n{_SIZE} 9\na\n\n{_END} rc=0\n", stderr=None)
            if self.symlink:
                return SimpleNamespace(return_code=4, stdout=self.noise + _framed(None, 4), stderr=None)
            if not self.patch_exists:
                return SimpleNamespace(return_code=3, stdout=self.noise + _framed(None, 3), stderr=None)
            return SimpleNamespace(return_code=0, stdout=self.noise + _framed(self.content), stderr=None)
        if "make-patch" in command:  # review round 3: the hook inside its frame (_hook_cmd)
            if self.hook_raises == "deadline":
                await asyncio.sleep(0.1)
                raise RuntimeError("Command timed out after 1 seconds")
            if self.hook_raises == "now":
                raise RuntimeError("docker compose exec: connection reset")
            if self.hook_raises == "timeout_now":
                raise TimeoutError("connect timed out")
            if self.hook_rc_fault:
                return SimpleNamespace(return_code=255, stdout=self.noise + CONNECT_ERROR, stderr=None)
            if self.hook_cut:
                return SimpleNamespace(return_code=255, stdout=f"{self.noise}{_HOOK_BEGIN}\n{CONNECT_ERROR}", stderr=None)
            if self.hook_killed:
                return SimpleNamespace(return_code=137, stdout=f"{self.noise}{_HOOK_BEGIN}\n", stderr=None)
            if self.symlink and self.victim is not None:  # what docker cp or a bind mount would show the host
                self.trial_paths.mkdir()
                (self.trial_paths.artifacts_dir / "model.patch").symlink_to(self.victim)
            return SimpleNamespace(return_code=self.hook_rc, stdout=f"{self.noise}{_HOOK_BEGIN}\n" + ("fatal: bad revision 'base'\n" if self.hook_rc else "")
                                   + f"\n{_HOOK_END} rc={self.hook_rc}\n", stderr=None)
        return SimpleNamespace(return_code=0, stdout="", stderr=None)

    async def download_dir(self, source_dir, target_dir):
        self.calls.append(("download_dir", source_dir, Path(target_dir)))

    async def download_file(self, source_path, target_path):
        self.calls.append(("download_file", source_path, Path(target_path)))
        Path(target_path).write_text("never-read\n")

    async def prepare_logs_for_host(self):
        self.calls.append(("prepare_logs_for_host",))


def _bare_patch_task(timeout_sec: float = 300.0, *, hooks: bool = True) -> HarborPatchTask:
    """A ``HarborPatchTask`` without Harbor: only what ``agent_logs`` and ``collect`` read (``hooks=False``: no collect hook)."""
    t = object.__new__(HarborPatchTask)
    hook = SimpleNamespace(command="make-patch", timeout_sec=timeout_sec, user=None, service="main")
    t.task = SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace(collect=[hook] if hooks else [])))
    t._env_paths = lambda: SimpleNamespace(agent_dir=PurePosixPath("/logs/agent"), artifacts_dir=PurePosixPath("/logs/artifacts"))
    return t


def _world(env: _HarborEnv, **params) -> SimpleNamespace:
    return SimpleNamespace(backend=SimpleNamespace(env=env, params=dict(params)))


@pytest.mark.parametrize("params, bound", [({}, True), ({"docker_host": "ssh://lab"}, False), ({"environment_type": "modal"}, False)])
async def test_logs_are_fetched_unless_bound_and_the_patch_is_read_through_the_container(tmp_path, params, bound):
    """Critic amendment 3: Harbor's Docker environment always says ``mounted``; whether /logs is bound is Crucible's own
    backend's decision (local daemon only), so the lab's logs are downloaded. M10 lab audit R-1: the patch is never
    fetched through a host path (bound or downloaded); it is read through the container as root."""
    t = _bare_patch_task()
    env = _HarborEnv(tmp_path / "trial")
    w = _world(env, **params)
    assert t._logs_bound(w) is bound
    logs = await t.agent_logs(w, fetch=True)
    assert logs == env.trial_paths.agent_dir
    patch = await t.collect(w, tmp_path / "collect")
    kinds = [c[0] for c in env.calls]
    assert patch == "the-diff\n" and "download_file" not in kinds
    [read] = [c for c in env.calls if c[0] == "exec" and "base64 <" in c[1]]
    assert read[2] == "root" and read[3] == "/"  # review round 3: never in a working directory the policy can remove
    if bound:
        assert "download_dir" not in kinds and "prepare_logs_for_host" in kinds
    else:
        assert ("download_dir", "/logs/agent", env.trial_paths.agent_dir) in env.calls
    assert (tmp_path / "collect" / "collect.log").exists()
    assert await t.agent_logs(w) == env.trial_paths.agent_dir and len(env.calls) == len(kinds)  # no fetch: nothing called


@pytest.mark.parametrize("kw, kind, reads", [
    ({"hook_rc": 128}, "hook", 0),
    ({"patch_exists": False}, "missing", 1),
    ({"symlink": True}, "unsafe", 1),  # M10 lab audit R-1: a link at the patch path is refused, never followed
    ({"hook_raises": "deadline"}, "hook", 0),  # outlived its deadline in the agent's repository: the agent's (was an unknown mask)
    ({"hook_raises": "now"}, "transport", 0),
    ({"read_raises": 3}, "transport", 3),  # the read kept raising in a live container
    ({"read_raises": 2}, None, 3),  # a transient blip: the third read returns the patch, nothing is lost
    # review round 2: Harbor's exec merges stderr into stdout and returns a docker or ssh failure as a non-zero exit
    ({"noise": NOISE}, None, 1),  # compose and shell warnings before the frame: the patch exactly (b64decode kept their letters)
    ({"noise": NOISE, "content": "x" * 1000 + "\n"}, None, 1),  # several base64 lines
    ({"noise": NOISE, "patch_exists": False}, "missing", 1),
    ({"noise": NOISE, "hook_rc": 128}, "hook", 0),  # the hook's shell started: its exit is the agent's
    ({"read_rc_fault": 2}, None, 3),  # no verdict (the docker CLI's error): retried, never "missing"
    # never "missing" (was the policy's 0 for an ssh blip); review round 3: nor infra. A returned exec without the shell's
    # report is the same in band as a root policy's replaced shell, which can print the CLI's error itself: unknown
    ({"read_rc_fault": 3}, "shell", 3),
    ({"hook_rc_fault": True}, "shell", 0),  # rc 255 before the hook's shell started: never the agent's, never infra
    ({"read_garbage": True}, "missing", 1),  # verdict 0 around a body that is not the patch: the container's own tools
    ({"read_stalls": 0.2}, "hook", 1),  # outlived its deadline (a planted base64 that hangs): the agent's, never retried
    # ---- review round 3
    # the hook's own verdict decides: an ssh reset mid-hook carried the streamed marker and the CLI's exit (was the policy's 0)
    ({"hook_cut": True}, "shell", 0),
    ({"noise": NOISE, "hook_cut": True}, "shell", 0),
    ({"hook_killed": True}, "hook", 0),  # its shell killed by a signal after its marker (docker exec passes 128+n through)
    # a provider's fast TimeoutError is host side, never the hook's or the read's stall (decided by the time alone)
    ({"hook_raises": "timeout_now"}, "transport", 0),
    ({"read_timeouts": 3}, "transport", 3),
    ({"read_timeouts": 2}, None, 3),  # retried like any fast failure: the third read returns the patch
    # a planted base64 that kills its shell after the frame's start: the agent's while a fresh root shell answers
    ({"read_killed": 3}, "hook", 3),
    ({"read_killed": 2}, None, 3),
    ({"read_killed": 3, "probe_dead": True}, "shell", 3),  # no root shell answers either: nothing shows whose
    # runc refused to start the shell in the container's state (bash or the working directory removed): the agent's
    ({"no_bash": True}, "hook", 0),
    ({"hookless": True, "no_bash": True}, "hook", 3),
    ({"workdir_gone": True}, "hook", 0),
    ({"hookless": True, "workdir_gone": True}, None, 1),  # the read runs in /, which the policy cannot remove
    # the hook's own context fails again while a root shell in / answers (its working directory, user or shell): the agent's
    ({"ctx_broken": True}, "hook", 0),
    ({"ctx_broken": True, "probe_dead": True}, "shell", 0),
    ({"shell_replaced": True}, "shell", 0),  # a stub bash: every exec, the root probe's too, returns nothing: unknown
    ({"hookless": True, "shell_replaced": True}, "shell", 3),
])
async def test_collect_errors_are_typed(tmp_path, monkeypatch, kw, kind, reads):
    from crucible.interop.harbor import patch_task

    monkeypatch.setattr(patch_task, "_READ_BACKOFF_S", 0.0)
    if kw.get("read_stalls"):
        monkeypatch.setattr(patch_task, "_READ_TIMEOUT_S", 0.05)
    kw = dict(kw)
    t = _bare_patch_task(timeout_sec=0.05, hooks=not kw.pop("hookless", False))
    env = _HarborEnv(tmp_path / "trial", **kw)
    if kind is None:
        assert await t.collect(_world(env, docker_host="ssh://lab"), tmp_path / "collect") == kw.get("content", "the-diff\n")
    else:
        with pytest.raises(CollectError) as ei:
            await t.collect(_world(env, docker_host="ssh://lab"), tmp_path / "collect")
        assert ei.value.kind == kind, ei.value
    assert sum(1 for c in env.calls if c[0] == "exec" and "base64 <" in c[1]) == reads
    # review round 3: the read runs in /; the hook keeps its own (Harbor's) working directory
    assert all(c[3] == "/" for c in env.calls if c[0] == "exec" and "base64 <" in c[1])
    assert all(c[3] is None for c in env.calls if c[0] == "exec" and "make-patch" in c[1])


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks")
@pytest.mark.parametrize("params", [{}, {"docker_host": "ssh://lab"}])
async def test_a_planted_patch_link_never_reads_a_host_file(tmp_path, params):
    """M10 lab audit R-1: a link at /logs/artifacts/model.patch (bound, or copied as a link by docker cp) pointed at a host
    file used to be read by the host into the record and shipped to the grader. Now it is ``unsafe`` and the host's copy
    is never opened."""
    victim = tmp_path / "crucible.toml"
    victim.write_text("secret = 'host'\n")
    t = _bare_patch_task()
    env = _HarborEnv(tmp_path / "trial", symlink=True, victim=victim)
    with pytest.raises(CollectError) as ei:
        await t.collect(_world(env, **params), tmp_path / "collect")
    assert ei.value.kind == "unsafe" and "secret" not in str(ei.value)
    assert not any(c[0] == "download_file" for c in env.calls)


@pytest.mark.skipif(sys.platform == "win32" or not all(shutil.which(x) for x in ("sh", "base64", "wc")), reason="a POSIX shell with base64 and wc")
def test_the_patch_read_command_in_a_real_shell(tmp_path):
    """M10 lab audit R-1: the in-container read prints a regular file within the cap as framed base64; a symlink (even to
    a regular file) or a FIFO exits 4 (unsafe), a missing or over-cap file exits 3 (missing), each with the same verdict
    in its frame. Review round 2: the frame is read past whatever Harbor's merged stderr put before it."""
    from crucible.interop.harbor.patch_task import _patch_read_cmd, _payload, _read_verdict

    def run(p: Path, cap: int = 8) -> subprocess.CompletedProcess:
        return subprocess.run(["sh", "-c", _patch_read_cmd(str(p), cap=cap)], capture_output=True, text=True, timeout=30, check=False)

    def verdict(r: subprocess.CompletedProcess) -> int:
        v = _read_verdict(NOISE + r.stdout)
        assert v is not None and v[0] == r.returncode, r.stdout
        return v[0]

    ok = tmp_path / "ok.patch"
    ok.write_bytes(b"diff\n")
    r = run(ok)
    assert r.returncode == 0 and verdict(r) == 0 and _payload(_read_verdict(NOISE + r.stdout)[1]) == b"diff\n"
    empty = tmp_path / "empty.patch"
    empty.write_bytes(b"")
    assert _payload(_read_verdict(run(empty).stdout)[1]) == b""
    big = tmp_path / "big.patch"
    big.write_bytes(bytes(range(256)) * 4)  # several base64 lines, every byte value
    assert _payload(_read_verdict(run(big, cap=1024).stdout)[1]) == bytes(range(256)) * 4
    link = tmp_path / "link.patch"
    link.symlink_to(ok)
    assert verdict(run(link)) == 4
    fifo = tmp_path / "fifo.patch"
    os.mkfifo(fifo)
    assert verdict(run(fifo)) == 4
    assert verdict(run(tmp_path / "missing.patch")) == 3
    assert verdict(run(big)) == 3 and verdict(run(big, cap=1024)) == 0
    assert _read_verdict(NOISE + CONNECT_ERROR) is None  # the docker CLI's own failure carries no verdict
    with pytest.raises(ValueError):  # a body that is not the counted base64 (a replaced base64 or wc)
        _payload(["CRUCIBLE-PATCH-SIZE 5", "a"])
    with pytest.raises(ValueError):
        _payload(["CRUCIBLE-PATCH-SIZE 4", base64.b64encode(b"diff\n").decode()])


async def test_patch_task_grade_is_refused(tmp_path):
    t = object.__new__(HarborPatchTask)
    t.id = "x"
    rep = await t.grade(None, tmp_path)
    assert rep.infra["status"] == "masked" and rep.infra["failure_kind"] == "crucible:harness_unsupported"


# ------------------------------------------------------------------ part B: Harbor installed


def _separate(tmp_path: Path) -> Path:
    """The answer fixture rewritten in task02's shape: schema 1.3, a declared model.patch, a separate verifier, a collect hook."""
    d = tmp_path / "sep"
    shutil.copytree(F / "answer", d)
    toml = (d / "task.toml").read_text()
    toml = re.sub(r'schema_version = "[^"]*"', 'schema_version = "1.3"\nartifacts = ["/logs/artifacts/model.patch"]', toml, count=1)
    toml = toml.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n', 1)
    toml += '\n[[verifier.collect]]\ncommand = "mkdir -p /logs/artifacts && cd /app && git diff > /logs/artifacts/model.patch"\ntimeout_sec = 60.0\n'
    (d / "task.toml").write_text(toml)
    return d


def _harbor_with_agent_hosts() -> None:
    """Part B tests that admit the proxy host need Harbor 0.24's ``TrialAgentConfig.extra_allowed_hosts`` (the lock pins
    0.23.0, where ``_network_plan`` refuses agent-phase hosts): they skip on an older Harbor instead of failing."""
    pytest.importorskip("harbor")
    from harbor.models.trial.config import AgentConfig as TrialAgentConfig

    if "extra_allowed_hosts" not in (getattr(TrialAgentConfig, "model_fields", None) or {}):  # a pydantic model
        pytest.skip("needs Harbor 0.24 (TrialAgentConfig.extra_allowed_hosts); the lock pins 0.23.0")


def test_patch_task_accepts_a_separate_verifier_and_admits_the_proxy_host(tmp_path):
    _harbor_with_agent_hosts()
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    d = _separate(tmp_path)
    t = HarborPatchTask(d, image="x", agent_allowed_hosts=["10.1.2.3"])
    assert t.admits_agent_host("10.1.2.3") and not t.admits_agent_host("evil.example")
    assert t.grader.id == t.id and t.grader_integrity == "off"
    assert len(t.backend_params["phase_network_policies"]) == 1  # the agent's phase only: the verifier's belongs to the grading container
    with pytest.raises(ValueError, match="separate environment"):
        HarborInPlaceTask(d)

    plain = HarborInPlaceTask(F / "answer")
    assert not plain.admits_agent_host("10.1.2.3")  # the fixture's no-network baseline admits nothing
    opened = HarborInPlaceTask(F / "answer", agent_allowed_hosts=["10.1.2.3"])
    phase = opened.plan.agent_phase
    assert getattr(phase.network_mode, "value", str(phase.network_mode)) == "allowlist" and "10.1.2.3" in phase.allowed_hosts
    assert opened.admits_agent_host("10.1.2.3") and opened.agent_timeout_s == 60.0
    with pytest.raises(ValueError, match="shares the agent's environment"):
        HarborPatchTask(F / "answer", image="x")


async def test_a_plain_in_place_adapter_on_a_steps_dir_is_refused(tmp_path):
    """M10 lab audit R-1: ``HarborInPlaceTask`` loads a ``[[steps]]`` dir (only the separate-verifier case is checked when
    steps are absent), its task-level agent phase admits the proxy, and ``open`` would skip the agent phase. The harness
    refuses it (no key registered), and ``open(before_agent=...)`` refuses it before any container starts."""
    _harbor_with_agent_hosts()
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    d = tmp_path / "multi"
    shutil.copytree(F / "answer", d)
    (d / "task.toml").write_text((d / "task.toml").read_text() + '\n\n[[steps]]\nname = "only"\n')
    (d / "steps" / "only").mkdir(parents=True)
    (d / "steps" / "only" / "instruction.md").write_text("Write 42 to /app/answer.txt\n")
    t = HarborInPlaceTask(d, agent_allowed_hosts=["10.1.2.3"])
    assert t.multi_step and getattr(t, "steps", None) is None and t.admits_agent_host("10.1.2.3")
    assert not HarborInPlaceTask(F / "answer").multi_step

    env = Env(tmp_path / "run")
    s, rows = await env.run(t, retry=RetryRule(max_attempts=3))
    f = _failure(rows[0])
    assert f.kind == "crucible:harness_unsupported" and not f.retryable and "multi-step" in f.reason and s.attempts == 1
    assert env.proxy.sessions == {} and env.agents == []

    async def setup(w):
        raise AssertionError("setup must not run")

    with pytest.raises(ValueError, match=r"\[\[steps\]\]"):
        await t.open(tmp_path / "w", before_agent=setup)


def test_harbor_agent_adapters(tmp_path, monkeypatch):
    pytest.importorskip("harbor.agents.model_connection", reason="the 0.24 API (default_endpoint_env, agent.model_connection); the lock pins 0.23.0")
    from crucible.interop.harbor.agent_harness import default_endpoint_env, harbor_agent_class

    cls = harbor_agent_class("mini-swe-agent")
    assert cls.name() == "mini-swe-agent"
    tmpl = default_endpoint_env(cls, "openai/gpt-x")
    assert tmpl == {"MSWEA_API_KEY": "{key}", "OPENAI_API_KEY": "{key}", "OPENAI_BASE_URL": "{proxy}/v1", "OPENAI_API_BASE": "{proxy}/v1"}

    # the real agent built as the harness builds it: what it resolves is the capture key and the proxy, never the host's key
    monkeypatch.setenv("OPENAI_API_KEY", HOST_KEY)
    proxy = CaptureProxy(tmp_path / "cap")
    h = HarborAgentHarness("mini-swe-agent", "2.4.6", model="openai/gpt-x", proxy=proxy, proxy_url="http://10.1.2.3:8788")
    env_vars = {k: v.replace("{key}", "cap-test-key-0000").replace("{proxy}", h.proxy_url) for k, v in h._endpoint_template().items()}
    agent = h._make_agent(tmp_path / "logs", env_vars)
    conn = agent.model_connection
    assert conn.api_key == "cap-test-key-0000" and set(conn.env.values()) <= {"cap-test-key-0000", "http://10.1.2.3:8788/v1"}
    assert leaked_host_secrets(_agent_env_views(agent), allow={"cap-test-key-0000"}) == []
    assert h.harbor_version not in ("", None)
