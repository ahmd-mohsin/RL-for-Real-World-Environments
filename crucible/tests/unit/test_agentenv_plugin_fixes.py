"""agentenv plugin fixes (m13 audit 02): C1 GPU flags as an opt-in on the builtin local provider (no plugin provider whose
sandboxes the type guard destroys); C2 the crucible_world record carries the env card and MCP URL, so stock agentenv
steps work; C3 crucible_verify writes boolean gate results, replays only its episode's slice of the gateway trajectory,
and strips the router's signed actor argument."""

from __future__ import annotations

import pytest

pytest.importorskip("agent_env")


def test_gpu_flags_are_an_opt_in_on_the_local_provider():
    from importlib.metadata import entry_points

    from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider

    from crucible.interop.agentenv.plugin.gpu_docker import GPU_FLAGS, gpu_enabled, local_gpu

    base = LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS
    with local_gpu():
        assert gpu_enabled() and GPU_FLAGS in LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS
        assert LocalSandbox.type == "local"  # sandboxes keep agentenv's local networking (host_url_for, CA trust)
    assert LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS == base
    assert not [e for e in entry_points(group="agent_env.sandbox_providers") if e.value.startswith("crucible")]  # no broken provider registered


async def test_world_record_has_card_and_stock_tool_access_step_works():
    import httpx
    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps.modify_env_tool_access import ModifyEnvToolAccessStep

    from crucible.interop.agentenv.plugin.steps import WORLDS, CrucibleWorldStep

    ctx = await CrucibleWorldStep("w", env_id="hd2", task="crucible_pack_toy.runtime:HelpdeskTask").execute(TaskStepContext())
    env = ctx.deployed_envs[-1]
    try:
        assert env.environment_card and env.environment_card_url.endswith("/.well-known/agent-env.json") and env.mcp_url.startswith(env.gateway_url)
        ctx = await ModifyEnvToolAccessStep("m", None, env_id="hd2", action="disable", role="agent", tools=["tickets__issue_refund"]).execute(ctx)
        async with httpx.AsyncClient(timeout=60) as c:
            tools = (await c.post(f"{env.gateway_url}/step", json={"action": "list_tools"}, headers={"AgentEnv-Role": "agent"})).json()["tools"]
        assert not any(t["name"] == "tickets__issue_refund" for t in tools) and any(t["name"] == "tickets__get_ticket" for t in tools)
    finally:
        await WORLDS.pop("hd2")["world"].close()


async def test_verify_booleans_episode_window_and_router_actor_strip():
    import httpx
    from agent_env.task_step.context import TaskStepContext
    from crucible_pack_toy.verify import GOLD

    from crucible.interop.agentenv.plugin.steps import WORLDS, CrucibleVerifyStep, CrucibleWorldStep

    ctx = await CrucibleWorldStep("w", env_id="hd3", task="crucible_pack_toy.runtime:HelpdeskTask").execute(TaskStepContext())
    env = ctx.deployed_envs[-1]
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            for a in GOLD:
                await c.post(f"{env.gateway_url}/step", json={"action": "call_tool", "tool_name": a.function_name.replace(".", "__", 1), "arguments": a.arguments},
                             headers={"AgentEnv-Role": "agent"})
        ctx = await CrucibleVerifyStep("v", env_id="hd3", episode_key="e0").execute(ctx)
        v1 = ctx.metadata["verifications"]["crucible"]
        assert v1["score"] == 1.0 and v1["actions_replayed"] == len(GOLD)
        assert all(r["result"] is True for r in v1["results"])  # agentenv aggregators test `is True`
        world = WORLDS["hd3"]["world"]
        await world.step("agent", GOLD[0].model_copy(update={"tool_call_id": "again"}))  # through the World's router: a signed __actor argument
        ctx = await CrucibleVerifyStep("v2", env_id="hd3", verifier_id="second", episode_key="e1").execute(ctx)
        v2 = ctx.metadata["verifications"]["second"]
        assert v2["actions_replayed"] == 1  # only this episode's slice of the trajectory, not the first episode again
        assert v2["crucible"]["steps"] and all("__actor" not in str(r) for r in v2["crucible"]["steps"])
    finally:
        await WORLDS.pop("hd3")["world"].close()
