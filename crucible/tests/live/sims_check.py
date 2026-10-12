"""Live simulator checks (run inside the ``crucible-sim`` image on the lab; ``scripts/lab_sims.py`` stages and runs them).

    python tests/live/sims_check.py mujoco --out /work/out
    python tests/live/sims_check.py unity --unity-url http://127.0.0.1:8010 --out /work/out

- **mujoco**: real MuJoCo physics (``gymnasium:HalfCheetah-v5``, ``Hopper-v5``) as a World ``common/control`` App.
  Snapshot → restore replays the journal and checks every observation digest (MuJoCo must be deterministic for the
  same seed and actions). A raw-gymnasium rollout is compared with the World rollout (the App adds no drift). PPO
  through ``WorldEnv`` / ``VectorWorldEnv`` on Hopper must beat its random-policy return. 8 parallel Worlds stress the
  App (steps per second recorded).
- **unity**: a Unity ML-Agents environment (3DBall) served by the OpenEnv Unity server in its own Python 3.10
  container, reached through ``ControlApp``'s WebSocket client (``client:``): reset, continuous actions, rewards,
  episode ends; PPO through Worlds over the remote link.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np


def _task(env_id: str, rr, **kw):
    from crucible.runtime.control_task import ControlTask

    return ControlTask(f"live/{env_id}@1", env=f"gymnasium:{env_id}", return_range=rr, journal="lazy", clock=False, **kw)


async def mujoco(out: dict) -> None:
    import gymnasium

    from crucible.train.control import PPOConfig, evaluate, ppo
    from crucible.world.gym import VectorWorldEnv, WorldEnv

    # 1. the World rollout equals a raw gymnasium rollout (same seed, same actions)
    rng = np.random.default_rng(0)
    actions = [rng.uniform(-1, 1, 6).astype(np.float32) for _ in range(200)]
    raw = gymnasium.make("HalfCheetah-v5")
    o_raw, _ = raw.reset(seed=3)
    rr = [float(raw.step(a)[1]) for a in actions]
    env = WorldEnv(_task("HalfCheetah-v5", (-1000, 5000), max_episode_ticks=1000))
    o_w, _ = await env.areset(seed=3)
    rw = [float((await env.astep(a))[1]) for a in actions]
    out["mujoco_world_equals_raw"] = {"obs0_diff": float(np.abs(np.array(o_w) - o_raw).max()), "reward_diff": float(np.abs(np.array(rw) - np.array(rr)).max())}
    # 2. snapshot mid-episode, run on, restore (journal replay with digest checks), run the same actions again
    w = env.world
    ref = await w.snapshot()
    tail = [rng.uniform(-1, 1, 6).astype(np.float32) for _ in range(50)]
    first = [float((await env.astep(a))[1]) for a in tail]
    t0 = time.time()
    await w.restore(ref)
    restore_s = time.time() - t0
    second = [float((await env.astep(a))[1]) for a in tail]
    out["mujoco_snapshot_restore"] = {"replayed_steps": 200, "restore_s": round(restore_s, 3), "max_reward_diff_after_restore": float(np.abs(np.array(first) - np.array(second)).max())}
    await env.aclose()
    # 3. throughput: 8 Worlds in parallel, random actions
    venv = VectorWorldEnv(lambda i: WorldEnv(_task("Hopper-v5", (-100, 4000))), 8, seed=1)
    await venv.areset()
    t0 = time.time()
    n = 0
    for _ in range(250):
        await venv.astep([np.random.uniform(-1, 1, 3) for _ in range(8)])
        n += 8
    out["mujoco_throughput"] = {"worlds": 8, "steps": n, "steps_per_s": round(n / (time.time() - t0), 1)}
    await venv.aclose()
    # 4. PPO through Worlds on Hopper: beats the random policy
    venv = VectorWorldEnv(lambda i: WorldEnv(_task("Hopper-v5", (-100, 4000))), 8, seed=2)
    res = await ppo(venv, PPOConfig(total_steps=int(out.get("_ppo_steps", 200_000)), n_envs=8, rollout=256, epochs=10, minibatches=32, lr=3e-4, gamma=0.99, seed=0))
    await venv.aclose()
    ev = WorldEnv(_task("Hopper-v5", (-100, 4000)))
    greedy = await evaluate(ev, res, episodes=5)
    await ev.aclose()
    out["mujoco_ppo_hopper"] = {"first20": float(np.mean(res.episodes[:20])), "last20": res.history[-1]["mean_return_20"], "greedy": [round(g, 1) for g in greedy],
                                "updates": len(res.history), "episodes": len(res.episodes)}


async def unity(out: dict, url: str) -> None:
    from crucible.runtime.control_task import ControlTask
    from crucible.train.control import PPOConfig, ppo
    from crucible.world.gym import VectorWorldEnv, WorldEnv

    def task(i: int = 0):
        return ControlTask(f"live/unity-3dball-{i}@1", client={"base_url": url, "action_key": "continuous_actions", "observation_key": "vector_observations",
                                                                "done_is": "terminated", "message_timeout_s": 600,
                                                                "ping_interval_s": None},  # the server is started with UNITY_ENV_ID=3DBall
                           return_range=(-1, 100), journal="lazy", clock=False, restore="none",
                           action_space={"kind": "box", "shape": [2], "low": -1.0, "high": 1.0},
                           observation_space={"kind": "box", "shape": [8], "low": -1e9, "high": 1e9}, max_episode_ticks=500)

    env = WorldEnv(task())
    t0 = time.time()
    obs, _ = await env.areset(seed=0)
    out["unity_reset"] = {"obs_len": len(obs), "s": round(time.time() - t0, 1)}
    rets, ret, steps, t0 = [], 0.0, 0, time.time()
    for _ in range(300):
        obs, r, term, trunc, _ = await env.astep(np.random.uniform(-1, 1, 2))
        ret += r
        steps += 1
        if term or trunc:
            rets.append(ret)
            ret = 0.0
            await env.areset(seed=len(rets))
    out["unity_random"] = {"steps": steps, "episodes": len(rets), "mean_return": float(np.mean(rets)) if rets else None, "steps_per_s": round(steps / (time.time() - t0), 1)}
    await env.aclose()
    venv = VectorWorldEnv(lambda i: WorldEnv(task(i)), 1, seed=0)
    res = await ppo(venv, PPOConfig(total_steps=int(out.get("_unity_ppo_steps", 20_000)), n_envs=1, rollout=512, epochs=10, minibatches=8, lr=3e-4, seed=0))
    await venv.aclose()
    out["unity_ppo"] = {"first10": float(np.mean(res.episodes[:10])) if res.episodes else None, "last": res.history[-1]["mean_return_20"] if res.history else None,
                        "episodes": len(res.episodes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["mujoco", "unity"])
    ap.add_argument("--out", default="/work/out")
    ap.add_argument("--unity-url", default="http://127.0.0.1:8010")
    ap.add_argument("--ppo-steps", type=int, default=200_000)
    a = ap.parse_args()
    out: dict = {"_ppo_steps": a.ppo_steps, "_unity_ppo_steps": max(2000, a.ppo_steps // 10)}
    t0 = time.time()
    try:
        asyncio.run(mujoco(out) if a.which == "mujoco" else unity(out, a.unity_url))
    except Exception as e:  # noqa: BLE001 -- record what failed, with the partial results
        import traceback

        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-3000:]
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"sims_{a.which}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
