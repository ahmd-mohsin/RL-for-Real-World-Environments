"""M1 item 10c: Gymnasium-API environments over Worlds (``world/gym.py``) and PPO for neural policies (``train/control.py``).

Every action is a World step (the tool gate, the trace). PPO must learn through that path: CartPole (discrete) and
Pendulum (continuous torque) improve well beyond their random-policy returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from crucible.runtime.control_task import ControlTask
from crucible.train.control import PPOConfig, evaluate, ppo
from crucible.world.gym import VectorWorldEnv, WorldEnv

pytest.importorskip("gymnasium")


def _task(env_id: str, rr, **kw) -> ControlTask:
    return ControlTask(f"t/{env_id}@1", env=f"gymnasium:{env_id}", return_range=rr, journal="lazy", clock=False, **kw)


def test_world_env_sync_api_and_trace():
    env = WorldEnv(_task("Pendulum-v1", (-1600, 0), max_episode_ticks=5))
    try:
        obs, info = env.reset(seed=4)
        assert len(obs) == 3 and info["seed"] == 4 and env.obs_dim == 3
        a_sp, o_sp = env.gym_spaces()
        assert a_sp.shape == (1,) and o_sp.shape == (3,)
        total = 0.0
        for k in range(5):
            obs, r, term, trunc, info = env.step(np.array([9.0]))  # out of bounds: clipped to 2.0, never refused
            total += r
        assert trunc and not term and info["episode"]["ticks"] == 5 and abs(info["episode"]["return"] - total) < 1e-9
        events = env.world.to_trace().events
        assert sum(1 for e in events if e.type == "tool_call") == 5  # each action went through the World's gate
        assert env.app.journal["steps"][0]["action"] == [2.0]
        again, _ = env.reset(seed=4)
        first, _ = WorldEnv(_task("Pendulum-v1", (-1600, 0))).reset(seed=4)
        assert again == first  # a seed reproduces the episode start
    finally:
        env.close()


async def test_vector_env_auto_reset():
    venv = VectorWorldEnv(lambda i: WorldEnv(_task("CartPole-v1", (0, 500), max_episode_ticks=3)), 2, seed=7)
    try:
        obs = await venv.areset()
        assert len(obs) == 2 and len(obs[0]) == 4
        for _ in range(2):
            await venv.astep([0, 1])
        obs, _rew, _term, trunc, infos = await venv.astep([0, 1])
        assert all(trunc) and all("final_obs" in i and "episode" in i for i in infos)
        assert all(e.episode_ticks == 0 for e in venv.envs)  # reset on a fresh seed
    finally:
        await venv.aclose()


async def test_ppo_learns_cartpole_through_worlds():
    venv = VectorWorldEnv(lambda i: WorldEnv(_task("CartPole-v1", (0, 500))), 8, seed=1)
    try:
        res = await ppo(venv, PPOConfig(total_steps=25_000, n_envs=8, rollout=128, epochs=4, minibatches=4, lr=2.5e-4, seed=0))
        first = float(np.mean(res.episodes[:20]))
        last = res.history[-1]["mean_return_20"]
        assert first < 40 and last > 3 * first, (first, last)
        ev = WorldEnv(_task("CartPole-v1", (0, 500)))
        greedy = await evaluate(ev, res, episodes=3)
        await ev.aclose()
        assert np.mean(greedy) > 100, greedy
    finally:
        await venv.aclose()


@pytest.mark.slow
async def test_ppo_learns_pendulum_continuous_control():
    venv = VectorWorldEnv(lambda i: WorldEnv(_task("Pendulum-v1", (-1600, 0))), 8, seed=1)
    try:
        res = await ppo(venv, PPOConfig(total_steps=60_000, n_envs=8, rollout=256, epochs=10, minibatches=32, lr=1e-3, gamma=0.9, seed=0))
        first = float(np.mean(res.episodes[:16]))
        last = res.history[-1]["mean_return_20"]
        assert first < -1000 and last > -700, (first, last)  # random torque scores about -1200; a swing-up policy about -200
    finally:
        await venv.aclose()
