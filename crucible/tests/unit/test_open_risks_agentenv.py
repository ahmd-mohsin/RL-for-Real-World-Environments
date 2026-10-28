"""Open risks from the harness-reward audit (2026-10-28, §5 / R10): a failed agentenv run leaked its crucible World (the
gateway, the App servers, any paid sandbox) and the step's ``mkdtemp`` directory. Each test fails on the code before the
fix: the World was closed only by ``crucible_verify``."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("agent_env")

OPENED: list[FakeWorld] = []


class FakeBackend:
    def __init__(self, fail: bool):
        self.fail = fail
        self.gateway_url = "http://crucible-fake-gateway.invalid"

    async def agent_mcp_url(self) -> str:
        if self.fail:
            raise RuntimeError("the agent proxy did not start")
        return f"{self.gateway_url}/proxy/mcp"


class FakeWorld:
    def __init__(self, workdir: Path, fail: bool):
        self.workdir, self.closed, self.log = Path(workdir), False, []
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "state.txt").write_text("x")
        self.backend = FakeBackend(fail)

    async def close(self) -> None:
        self.closed = True


class FakeTask:
    fail = False

    def __init__(self, **kw):
        self.kw = kw

    async def open(self, workdir: Path) -> FakeWorld:
        w = FakeWorld(workdir, self.fail)
        OPENED.append(w)
        return w


class FailingTask(FakeTask):
    fail = True


@pytest.fixture
def card(monkeypatch):
    """The step's card GET answered in process (no gateway): httpx.AsyncClient with a mock transport."""
    import httpx

    import crucible.interop.agentenv.plugin.steps  # noqa: F401 -- its imports (mcp) annotate with the real class first

    class Client(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"name": "fake"})), **kw)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    OPENED.clear()
    yield
    OPENED.clear()


async def test_world_step_that_fails_after_open_closes_its_world_and_directory(card):
    from agent_env.task_step.context import TaskStepContext

    from crucible.interop.agentenv.plugin.steps import WORLDS, CrucibleWorldStep

    before = set(WORLDS)
    with pytest.raises(RuntimeError, match="proxy did not start"):
        await CrucibleWorldStep("w", env_id="e", task=f"{__name__}:FailingTask").execute(TaskStepContext())
    [w] = OPENED
    assert w.closed, "the World of a failed crucible_world step was left open"
    assert not w.workdir.parent.exists(), "the step's mkdtemp directory was left behind"
    assert set(WORLDS) == before


async def test_teardown_run_closes_the_world_of_a_run_that_failed_before_verify(card, monkeypatch):
    """A later step (deploy_agent, prompt_agent) failed, so crucible_verify never ran: agentenv's own teardown_run (what
    ``agent-env run`` and the LocalRunner call however a run ends) closes the World through its crucible_world record."""
    from agent_env.providers.sandbox_providers import sandbox_provider
    from agent_env.task.teardown import teardown_run
    from agent_env.task_step.context import TaskStepContext

    from crucible.interop.agentenv.plugin.steps import WORLDS, CrucibleWorldProvider, CrucibleWorldStep

    reg = dict(sandbox_provider._get_sandbox_registry())
    reg["crucible_world"] = {"impl": CrucibleWorldProvider}  # what the plugin's agent_env.sandbox_providers entry point registers
    monkeypatch.setattr(sandbox_provider, "_get_sandbox_registry", lambda: reg)
    ctx = await CrucibleWorldStep("w", env_id="e", task=f"{__name__}:FakeTask").execute(TaskStepContext())
    [w] = OPENED
    d = w.workdir.parent
    assert not w.closed and d.exists()
    report = await teardown_run(ctx)
    assert not report.failed, report.failed
    assert w.closed, "teardown_run left the crucible World open"
    assert not d.exists()
    assert not any(s.get("world") is w for s in WORLDS.values())
    assert not (await teardown_run(ctx)).failed  # a second teardown (or one after verify closed it) is a no-op


async def test_teardown_worlds_and_verify_outputs_kept(card):
    """``teardown_worlds`` releases every World a context opened, for embedders that call ``Task.run`` themselves; a
    directory holding a verify step's outputs keeps them (the run's records name them), only the work tree goes."""
    from agent_env.task_step.context import TaskStepContext

    from crucible.interop.agentenv.plugin.steps import CrucibleWorldStep, release, teardown_worlds

    ctx = TaskStepContext()
    ctx = await CrucibleWorldStep("w1", env_id="a", task=f"{__name__}:FakeTask").execute(ctx)
    ctx = await CrucibleWorldStep("w2", env_id="b", task=f"{__name__}:FakeTask").execute(ctx)
    w1, w2 = OPENED
    (w2.workdir.parent / "verify-crucible").mkdir()
    (w2.workdir.parent / "verify-crucible" / "reward.json").write_text("{}")
    assert len(await teardown_worlds(ctx)) == 2 and w1.closed and w2.closed
    assert not w1.workdir.parent.exists()
    assert (w2.workdir.parent / "verify-crucible" / "reward.json").exists() and not w2.workdir.exists()
    assert await teardown_worlds(ctx) == []
    inst = ctx.deployed_envs[-1].instance_id
    assert await release(inst) is False


def test_provider_creates_nothing():
    from crucible.interop.agentenv.plugin.steps import CrucibleWorldProvider

    async def go():
        with pytest.raises(NotImplementedError):
            await CrucibleWorldProvider().create_sandbox(image_name="x", port=1, env={})

    import asyncio

    asyncio.run(go())
