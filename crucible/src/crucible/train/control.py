"""PPO for neural-network policies on control Worlds (M1 item 10c; the non-token trainer path of M12).

The language-model trainer (``train/``) learns over tokens. A control task's policy is a small network over observation
vectors, so it gets its own learner: PPO (clipped surrogate, GAE(lambda), value loss, entropy bonus, minibatch epochs,
gradient clipping, running observation normalization), with a categorical head for discrete actions and a Gaussian head
with a state-independent log standard deviation for box actions. It trains through ``VectorWorldEnv``: every action is
a World step (tool gate, trace, events on simulation time), so what is learned here runs unchanged in a World.

Follows the PPO recipe as CleanRL's ``ppo.py`` / ``ppo_continuous_action.py`` and Stable-Baselines3 implement it
(pattern; no code copied).

Records (optional, off by default): ``ppo(venv, cfg, recorder=ControlRecorder(sink=...))`` writes one
``crucible.control_rollout/v1`` record per finished episode (``train.control_record``): executed and sampled actions, the
joint log-prob and value of each sampled action, observation digests, rewards, terminated / truncated, ticks, seed and env
id, so a run can be audited, replayed through a fresh ``ControlApp`` and streamed to the run's sinks.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class PPOConfig:
    total_steps: int = 50_000
    n_envs: int = 8
    rollout: int = 128  # steps per env per update
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    obs_norm: bool = True
    anneal_lr: bool = True
    seed: int = 0


class RunningNorm:
    """Running mean and variance of observations (Welford over batches); clipped normalized output."""

    def __init__(self, dim: int, clip: float = 10.0):
        self.mean, self.var, self.count, self.clip = np.zeros(dim), np.ones(dim), 1e-4, clip

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}


def _layer(i: int, o: int, std: float = math.sqrt(2)) -> nn.Linear:
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.constant_(lin.bias, 0.0)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_space: Any, hidden: int = 64):
        super().__init__()
        self.kind = action_space.kind
        if self.kind == "discrete":
            self.n_out, self.start = int(action_space.n), action_space.start
        elif self.kind == "box":
            self.n_out = action_space.flat_dim()
            lo, hi = action_space.bounds()
            self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
            self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
            self.log_std = nn.Parameter(torch.zeros(self.n_out))
        else:
            raise ValueError(f"PPO here handles discrete and box actions, not {self.kind}")
        self.critic = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, 1, 1.0))
        self.actor = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, self.n_out, 0.01))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        out = self.actor(obs)
        if self.kind == "discrete":
            return torch.distributions.Categorical(logits=out)
        return torch.distributions.Normal(out, self.log_std.exp().expand_as(out))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def logp_ent(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        if self.kind == "discrete":
            return d.log_prob(act), d.entropy()
        return d.log_prob(act).sum(-1), d.entropy().sum(-1)

    def to_env(self, act: torch.Tensor) -> list[Any]:
        if self.kind == "discrete":
            return [int(a) + self.start for a in act.tolist()]
        return act.tolist()  # WorldEnv clips into the bounds; log-probs stay those of the sampled action

    @torch.no_grad()
    def act_greedy(self, obs: torch.Tensor) -> list[Any]:
        out = self.actor(obs)
        return self.to_env(out.argmax(-1) if self.kind == "discrete" else out)


@dataclass
class PPOResult:
    model: ActorCritic
    norm: RunningNorm | None
    history: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[float] = field(default_factory=list)  # every finished episode's return, in order
    seconds: float = 0.0
    steps: int = 0


async def ppo(venv: Any, cfg: PPOConfig, *, recorder: Any = None) -> PPOResult:
    """Train on a ``VectorWorldEnv`` (``cfg.n_envs`` must match it). ``recorder``: a ``control_record.ControlRecorder``
    (attached before the first reset; episodes still running at the end are flushed as ``stop="run_end"``)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if recorder is not None:
        recorder.attach(venv)
    raw = np.asarray(await venv.areset(), dtype=np.float64)  # opens the Worlds: their spaces are known from here on
    obs_dim = venv.observation_space.flat_dim()
    model = ActorCritic(obs_dim, venv.action_space, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    norm = RunningNorm(obs_dim) if cfg.obs_norm else None
    n, T = venv.n, cfg.rollout
    updates = max(1, cfg.total_steps // (n * T))
    res = PPOResult(model=model, norm=norm)
    t0 = time.monotonic()
    act_dim = () if model.kind == "discrete" else (model.n_out,)
    for u in range(updates):
        if cfg.anneal_lr:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - u / updates)
        obs_b = np.zeros((T, n, obs_dim), np.float32)
        act_b = np.zeros((T, n, *act_dim), np.float32 if act_dim else np.int64)
        logp_b, rew_b, done_b, val_b = (np.zeros((T, n), np.float32) for _ in range(4))
        boot = np.zeros((T, n), np.float32)  # value of the final obs of a truncated episode (not a terminal state)
        for t in range(T):
            if norm is not None:
                norm.update(raw)
            o = norm(raw) if norm is not None else raw
            ot = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                d = model.dist(ot)
                a = d.sample()
                lp = d.log_prob(a) if model.kind == "discrete" else d.log_prob(a).sum(-1)
                v = model.value(ot)
            env_actions = model.to_env(a)
            if recorder is not None:
                recorder.annotate(env_actions if model.kind == "discrete" else a.tolist(), lp.tolist(), v.tolist(), version=u)
            nxt, rew, term, trunc, infos = await venv.astep(env_actions)
            obs_b[t], act_b[t], logp_b[t], val_b[t] = o, a.numpy(), lp.numpy(), v.numpy()
            rew_b[t] = rew
            done_b[t] = [float(x or y) for x, y in zip(term, trunc, strict=True)]
            for i, info in enumerate(infos):
                if trunc[i] and not term[i]:
                    fo = np.asarray(info["final_obs"], dtype=np.float64)
                    fo = norm(fo) if norm is not None else fo
                    with torch.no_grad():
                        boot[t, i] = float(model.value(torch.as_tensor(fo, dtype=torch.float32)))
                if "episode" in info:
                    res.episodes.append(float(info["episode"]["return"]))
            raw = np.asarray(nxt, dtype=np.float64)
        rew_b = rew_b + cfg.gamma * boot
        with torch.no_grad():
            last_v = model.value(torch.as_tensor(norm(raw) if norm is not None else raw, dtype=torch.float32)).numpy()
        adv, gae = np.zeros((T, n), np.float32), np.zeros(n, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_b[t + 1]
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + cfg.gamma * nv * nonterm - val_b[t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val_b
        B = T * n
        bo = torch.as_tensor(obs_b.reshape(B, obs_dim))
        ba = torch.as_tensor(act_b.reshape(B, *act_dim))
        blp, badv, bret = (torch.as_tensor(x.reshape(B)) for x in (logp_b, adv, ret))
        mb = max(1, B // cfg.minibatches)
        stats: dict[str, float] = {}
        for _ in range(cfg.epochs):
            perm = torch.randperm(B)
            for s in range(0, B, mb):
                idx = perm[s:s + mb]
                lp, ent = model.logp_ent(bo[idx], ba[idx])
                ratio = (lp - blp[idx]).exp()
                a_n = (badv[idx] - badv[idx].mean()) / (badv[idx].std() + 1e-8)
                pg = torch.max(-a_n * ratio, -a_n * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                vl = 0.5 * (model.value(bo[idx]) - bret[idx]).pow(2).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                stats = {"policy_loss": float(pg.detach()), "value_loss": float(vl.detach()), "entropy": float(ent.mean().detach()),
                         "clip_frac": float(((ratio - 1).abs() > cfg.clip).float().mean())}
        res.steps = (u + 1) * B
        recent = res.episodes[-20:]
        res.history.append({"update": u + 1, "steps": res.steps, "episodes": len(res.episodes),
                            "mean_return_20": float(np.mean(recent)) if recent else None, **stats})
    res.seconds = time.monotonic() - t0
    if recorder is not None:
        recorder.flush()
    return res


async def evaluate(env: Any, result: PPOResult, *, episodes: int = 5, seed: int = 10_000) -> list[float]:
    """Greedy returns of a trained policy on a ``WorldEnv``."""
    out = []
    for k in range(episodes):
        raw, _ = await env.areset(seed + k)
        total, done = 0.0, False
        while not done:
            o = np.asarray(raw, dtype=np.float64)
            o = result.norm(o) if result.norm is not None else o
            a = result.model.act_greedy(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))[0]
            raw, r, term, trunc, _ = await env.astep(a)
            total += r
            done = term or trunc
        out.append(total)
    return out


def run(venv: Any, cfg: PPOConfig, *, recorder: Any = None) -> PPOResult:
    return asyncio.run(ppo(venv, cfg, recorder=recorder))
