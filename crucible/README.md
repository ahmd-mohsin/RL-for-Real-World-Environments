# Crucible

Workflow traces with expert signal → dense-reward RL environments, frontier-hard tasks, step-level benchmarks.
`DESIGN.md` covers what and why; `ARCHITECTURE.md` covers how (modules M0–M15). `docs/modules/` holds each built module's spec and gate report; `THIRD_PARTY.md` lists what we build on.

**Built so far: M0 core, M1 world, M2 actors.**

## Install
```bash
uv sync --python 3.12 --extra mcp --extra harbor --extra agentenv --extra llm --extra dev --extra openenv-awm --group packs
```
`verifiers` (the prime-rl path) pins `mcp==2.0.0` and cannot share this environment with agentenv; it gets its own environment.

## A World in 20 lines
```python
import asyncio
from crucible.actors.policy import ScriptedPolicy, run_episode
from crucible.core.schema.actions import Message, ToolCall
from crucible.world.world import World
from crucible_pack_toy import INSTRUCTION, PACK_ROOT, helpdesk_spec, responders

async def main():
    w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION}, pack_root=PACK_ROOT)  # backend="agentenv" works too
    for r in responders():            # grounded customers + supervisor bot run inside the world
        w.attach(r)
    await w.start()
    try:
        policy = ScriptedPolicy("agent", [ToolCall(tool_call_id="1", function_name="tickets.list_tickets", arguments={}),
                                          Message(to="customer_lee", text="How much were you charged for O-5002?")])
        res = await run_episode(w, {"agent": policy})
        print(next(e.text for e in res.trace.events if e.actor == "customer_lee"))  # the customer's grounded answer
        ref = await w.snapshot(); await w.restore(ref)         # exact, verified by fingerprints
        print(await w.verifier_view().query("tickets", {"op": "sql", "args": {"sql": "SELECT id, status FROM tickets"}}))
    finally:
        await w.close()

asyncio.run(main())
```

## What else is there
- **Seed coding worlds** (tasks 01–04) on a GPU host through Harbor: `packs/coding/families/legacy_gpu_migration/seeds/*/crucible/world.yaml`. Gate: `CRUCIBLE_GPU=1 pytest tests/gpu -m gpu`.
- **Import any MiMo env or AgentWorldModel-1K scenario as a World:** `crucible.interop.mimo.importer.import_mimo_env`, `crucible.interop.openenv.awm.awm_world`. Sweeps: `python -m crucible.interop.mimo.check <envs_dir>`, `python -m crucible.interop.openenv.awm_check 100`.
- **Serve a World to any MCP harness:** with the agentenv backend, `world.backend.mcp_url` plus the header `AgentEnv-Role: <actor>`.
- **Export an episode as an ATIF trajectory:** `crucible.interop.harbor.atif.episode_to_atif`.
- **Grounded actors:** `GroundedResponder` (verbatim / quote / llm), and the coding pack's `ReviewerActor`, which replays real review comments. Fidelity: `python -m crucible_pack_coding.reviewer_fidelity`.

## Tests
`pytest tests/unit tests/toy_e2e` (CPU, under a minute, no paid calls). `-m docker` needs Docker; GPU gates need `CRUCIBLE_GPU=1` and a `[gpu_hosts]` entry in `crucible.toml`. `lint-imports` checks the layering contracts.
