#!/usr/bin/env python
"""Smoke: the RL loop on the CPU, no key, no spend, no Docker.

    python scripts/smoke/rl_loop.py --out <dir>

Real components: ``factory.envs.make_env`` (WorkspaceHarness, the checkpoint-Phi sidecar, InfraWatch, honesty) on the
three-milestone toy window in a host World (the unit-test fixtures, as ``tinker_rl.py smoke-rl`` builds it), the runtime
``Runner``, ``TinkerTrainer`` (groups, filters, datums, staleness, resume, the masked-share stop) over the in-memory
Tinker (``train.bridges.tinker.fake``: the documented losses, checkpoints in a dict), the prime-rl loss's score centering
(``train.bridges.primerl.loss.token_objective``) training a small torch policy against a mismatched sampler.

Checks (exit 1 when one fails):
1. **async**: ``async_groups=1, max_staleness=1``, 3 steps: stop ``done``, each step's ``min_versions`` within the bound,
   the sampler refreshed every step, staleness buckets logged;
2. **resume**: a new trainer on the same run dir and Tinker store continues at step 4 with the optimizer's step count;
3. **masked share**: every attempt's World dies (``infra``): the run stops ``masked_frac`` and never trains;
4. **score centering**: on a mismatched sampler, ``sc+mis`` gradient steps on a toy logits table raise the expected reward
   and the loss stays finite; plain IS with no correction is run beside it for the drift figure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "unit"))
MODEL = "Qwen/Qwen3.6-35B-A3B"


def toy(tmp: Path) -> Any:
    from test_m17_p6_equiv import HostWindow, scripts  # the toy window in a host World and its scripted solvers
    from test_m17_w_task import make_task

    task, rec, _ = make_task(tmp)
    task.__class__ = HostWindow
    return task, scripts(rec), HostWindow


def trainer(task: Any, sc: dict, run_dir: Path, cycle: list[str], fake: Any = None, **cfg: Any) -> tuple[Any, Any]:
    from crucible.factory.checkpoint import CheckpointConfig
    from crucible.factory.envs import Timeouts, make_env, objective_of
    from crucible.train.bridges.tinker.fake import FakeTinker, ScriptedResponder
    from crucible.train.bridges.tinker.trainer import TinkerTrainer, TrainConfig
    from crucible.train.chat import ByteChatCodec

    env = make_env(task, checkpoint=CheckpointConfig(every=2, cap=20, concurrency=1), timeouts=Timeouts(agent_s=300, rollout_s=900))
    codec = ByteChatCodec()
    fake = fake or FakeTinker()
    fake.responder = ScriptedResponder([sc[n] for n in cycle], codec)
    base = {"model": MODEL, "group_size": 2, "steps": 2, "turn_eta": 0.5, "require_band": False, "cap_usd": 1.0}
    t = TinkerTrainer([env], config=TrainConfig(**{**base, **cfg}), backend=fake, run_dir=run_dir, codec=codec, objective_id=objective_of([task]))
    return t, fake


def score_centering(steps: int = 200, seed: int = 0) -> dict[str, Any]:
    """A T x V logits table as the policy; the sampler is the policy plus fixed noise (a stale / mismatched engine); reward 1
    for token 0 at every position. Each step samples from the sampler, takes the top-k head, and descends ``token_objective``."""
    import torch

    from crucible.train.bridges.primerl.loss import Head, token_objective

    out: dict[str, Any] = {}
    for pg in ("sc+mis", "none"):
        g = torch.Generator().manual_seed(seed)
        T, V, k = 8, 16, 4
        z = torch.zeros(T, V, dtype=torch.float64, requires_grad=True)
        noise = 0.8 * torch.randn(T, V, generator=g, dtype=torch.float64)
        opt = torch.optim.SGD([z], lr=0.5)
        losses = []
        p0 = torch.softmax(z.detach(), -1)[:, 0].mean().item()
        for _ in range(steps):
            lp_full = torch.log_softmax(z, -1)
            lq_full = torch.log_softmax(z.detach() + noise, -1)
            y = torch.multinomial(lq_full.exp(), 1, generator=g).squeeze(-1)
            adv = (y == 0).double() - lq_full.exp()[:, 0]  # reward minus the sampler's baseline
            ids = lq_full.topk(k, dim=-1).indices
            head = Head(lp=lp_full.gather(-1, ids), lq=lq_full.gather(-1, ids), valid=torch.ones_like(ids, dtype=torch.bool),
                        present=torch.ones(T, dtype=torch.bool)) if pg != "none" else None
            loss, _m = token_objective(lp_full.gather(-1, y[:, None])[:, 0], lq_full.gather(-1, y[:, None])[:, 0], adv,
                                       torch.ones(T, dtype=torch.bool), pg_correction=pg, head=head)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        p1 = torch.softmax(z.detach(), -1)[:, 0].mean().item()
        out[pg] = {"p_token0_start": round(p0, 4), "p_token0_end": round(p1, 4), "loss_finite": all(abs(x) < 1e9 for x in losses)}
    out["ok"] = out["sc+mis"]["loss_finite"] and out["sc+mis"]["p_token0_end"] > out["sc+mis"]["p_token0_start"] + 0.2
    return out


def main_run(out: Path) -> dict[str, Any]:
    from crucible.train.bridges.tinker.fake import FakeTinker

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "toy").mkdir()
    task, sc, HW = toy(out / "toy")
    res: dict[str, Any] = {}

    HW._dead = False
    rd = out / "run-async"
    t, fake = trainer(task, sc, rd, ["pass", "fail"], steps=3, async_groups=1, max_staleness=1)
    o = asyncio.run(t.run())
    steps = [json.loads(x) for x in (rd / "steps.jsonl").read_text().splitlines()]
    mins = [s.get("min_versions") for s in steps]
    stale_keys = sorted({k for s in steps for k in (s.get("metrics") or {}) if k.startswith("staleness/")})
    samplers = [c["path"] for c in fake.calls if c["op"] == "sampler"]
    res["async"] = {"stop": o["stop"], "steps_done": o["steps_done"], "min_versions": mins, "staleness_keys": stale_keys[:6], "drops": o.get("drops"),
                    "sampler_refreshes": len(samplers),
                    "ok": o["stop"] == "done" and o["steps_done"] == 3 and all(i - min(m) <= 1 for i, m in enumerate(mins) if m)
                    and len(samplers) >= 3 and bool(stale_keys)}

    t2, fake2 = trainer(task, sc, rd, ["pass", "fail"], fake=FakeTinker(store=fake.store), steps=4, async_groups=1, max_staleness=1)
    o2 = asyncio.run(t2.run())
    tr = next(c for c in fake2.calls if c["op"] == "trainer")
    opt = next((c for c in fake2.calls if c["op"] == "optim_step"), {})
    res["resume"] = {"stop": o2["stop"], "steps_done": o2["steps_done"], "resumed_from": tr.get("resume"), "first_optim_step": opt.get("step"),
                     "ok": o2["steps_done"] == 4 and bool(tr.get("resume")) and opt.get("step") == 4}

    HW._dead = False
    rd3 = out / "run-masked"
    t3, fake3 = trainer(task, sc, rd3, ["infra"], steps=2, max_masked_frac=0.5, masked_window=2)
    try:
        o3 = asyncio.run(t3.run())
    finally:
        HW._dead = False
    trained = any(c["op"] in ("forward_backward", "optim_step") for c in fake3.calls)
    res["masked_share"] = {"stop": o3["stop"], "steps_done": o3["steps_done"], "drops": o3.get("drops"), "trained": trained,
                           "ok": o3["stop"] == "masked_frac" and not trained}

    res["score_centering"] = score_centering()
    res["ok"] = all(v.get("ok") for v in res.values() if isinstance(v, dict))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = main_run(Path(a.out))
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
