"""M11 exit gate driver (toy task): run ONE runtime path and write its canonical outputs, so each path can run in the
environment that has its framework installed (OpenEnv, verifiers: ``crucible/.envs/<fw>``; NeMo-Gym: 3.13).

    python -m crucible_pack_toy.m11.gate native  OUT
    python -m crucible_pack_toy.m11.gate openenv OUT     # in crucible/.envs/openenv

Each path writes ``OUT/reward.json`` (absent when masked), ``OUT/steps.jsonl`` and ``OUT/path.json`` (what the runtime
itself reported). The trajectory is the toy helpdesk oracle (``crucible_pack_toy.verify.GOLD``).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from crucible_pack_toy.runtime import HelpdeskTask
from crucible_pack_toy.verify import GOLD

KEY = "toy/helpdesk|e0"


async def native(out: Path) -> dict:
    from crucible.interop.replay import Action, replay

    acts = [Action(a.function_name, dict(a.arguments)) for a in GOLD]
    return await replay(HelpdeskTask(), acts, out, episode_key=KEY)


def openenv(out: Path) -> dict:
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    from crucible.interop.openenv.export import FINISH, make_environment
    from crucible.interop.replay import from_openenv_steps, replay

    env = make_environment(HelpdeskTask)()
    env.reset(seed=0, episode_id=KEY)
    tools = {t.name for t in env.step(ListToolsAction()).tools}
    sent, rewards = [], []
    for a in GOLD:
        name = a.function_name.replace(".", "__", 1)
        assert name in tools, name
        obs = env.step(CallToolAction(tool_name=name, arguments=dict(a.arguments)))
        rewards.append(obs.reward)
        sent.append({"tool_name": name, "arguments": dict(a.arguments)})
    fin = env.step(CallToolAction(tool_name=FINISH, arguments={}))
    rewards.append(fin.reward)
    env.close()
    info = asyncio.run(replay(HelpdeskTask(), from_openenv_steps(sent), out, episode_key=KEY))
    return {**info, "step_rewards": rewards, "sum": sum(r for r in rewards if r is not None), "final": fin.metadata["crucible"].get("psi_final"),
            "masked": fin.metadata["masked"], "env_reward_doc": fin.metadata["crucible_reward"]}


def main() -> None:
    path, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    res = asyncio.run(native(out)) if path == "native" else openenv(out) if path == "openenv" else None
    if res is None:
        raise SystemExit(f"unknown path {path}")
    (out / "path.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "env_reward_doc"}, default=str))


if __name__ == "__main__":
    main()
