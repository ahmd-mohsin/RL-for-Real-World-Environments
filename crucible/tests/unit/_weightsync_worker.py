"""One rank of a weight-sync scenario (``test_weightsync.py`` launches WORLD_SIZE of these over gloo):
``python _weightsync_worker.py <scenario> <out_dir>`` with RANK / WORLD_SIZE set (rendezvous: a file store in <out_dir>)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from crucible.train.weightsync import BroadcastWeightSync, DistributedPublisher, EngineReceiver, WeightSyncError


def weights(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"embed.weight": torch.randn(64, 32, generator=g), "layers.0.attn.q": torch.randn(32, 32, generator=g).to(torch.bfloat16),
            "layers.0.attn.k": torch.randn(32, 32, generator=g).to(torch.bfloat16), "layers.0.mlp.up": torch.randn(32, 96, generator=g),
            "layers.0.norm": torch.randn(32, generator=g).half(), "lm_head.weight": torch.randn(64, 32, generator=g), "step": torch.tensor(seed, dtype=torch.int64),
            "mask": torch.rand(7, generator=g) > 0.5, "scalar": torch.tensor(1.5)}


def digest(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {k: hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16] + f"|{v.dtype}|{tuple(v.shape)}"
            for k, v in sorted(state.items())}


BUCKET = 13 * 1024  # forces many buckets (the largest tensor, layers.0.mlp.up, is 12 KiB)


def main() -> None:
    scenario, out = sys.argv[1], Path(sys.argv[2])
    from datetime import timedelta

    # rendezvous through a file store (no TCP port to race for); a 60 s collective timeout turns a hang into an error
    dist.init_process_group("gloo", init_method=f"file://{out / 'store'}", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
                            timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    res: dict = {"rank": rank}
    sync = BroadcastWeightSync(bucket_bytes=BUCKET)
    held = {"v": 0, "state": {}}

    def commit(v, s):
        held["v"], held["state"] = v, s

    if scenario == "broadcast":  # rank 0 = trainer (owner); others receive; two versions through one plan
        owned = weights(1) if rank == 0 else None
        plan = sync.plan(owned)
        res["buckets"] = len(plan.buckets())
        res["aligned"] = all(m.offset % 256 == 0 for m in plan.metas)
        for v, seed in ((1, 1), (2, 2)):
            owned = weights(seed) if rank == 0 else None
            got = sync.update(v if rank == 0 else None, owned, receive=rank != 0, commit=commit)
            res[f"v{v}"] = got
            res[f"d{v}"] = digest(held["state"] if rank else owned)
    elif scenario == "slow_reload":  # the pipeline under a slow loader: a half is never overwritten while being read
        owned = weights(3) if rank == 0 else None
        sync.plan(owned)

        def slow(named):
            time.sleep(0.02)
            staged.update({n: t.clone() for n, t in named})

        staged: dict = {}
        sync.update(1 if rank == 0 else None, owned, receive=rank != 0, load=slow)
        res["d"] = digest(staged if rank else owned)
    elif scenario == "sharded":  # ranks 0 and 1 each own a shard and also receive (colocated); rank 2 receives only
        full = weights(4)
        names = sorted(full)
        owned = {n: full[n] for i, n in enumerate(names) if i % 2 == rank} if rank < 2 else None
        plan = sync.plan(owned)
        res["owners"] = sorted({m.owner for m in plan.metas})
        res["single_owner_buckets"] = all(len({m.owner for m in b}) == 1 for b in plan.buckets())
        sync.update(1 if rank == 0 else None, owned, receive=True, commit=commit)
        res["d"] = digest(held["state"])
        res["full"] = digest(full)
    elif scenario == "fail":  # rank 2 fails on its 3rd bucket: every rank raises, nobody commits, old weights stay
        owned = weights(5) if rank == 0 else None
        sync.plan(owned)
        sync.update(1 if rank == 0 else None, owned, receive=rank != 0, commit=commit)
        before = digest(held["state"]) if rank else None
        calls = {"n": 0}
        staged: dict = {}

        def flaky(named):
            calls["n"] += 1
            if rank == 2 and calls["n"] == 3:
                raise OSError("injected reload failure")
            staged.update({n: t.clone() for n, t in named})

        try:
            sync.update(2 if rank == 0 else None, weights(6) if rank == 0 else None, receive=rank != 0, load=flaky,
                        commit=lambda v, s: commit(v, dict(staged)))
            res["raised"] = None
        except WeightSyncError as e:
            res["raised"] = str(e)
        res["version_after"] = held["v"]
        res["unchanged"] = (digest(held["state"]) == before) if rank else None
    elif scenario == "subgroup":  # only ranks 0 and 2 update (a restarted engine catches up); rank 1 returns at once
        owned = weights(7) if rank == 0 else None
        plan = sync.plan(owned)
        text = plan.to_json()
        late = BroadcastWeightSync(bucket_bytes=BUCKET)
        late.load_plan(text)  # metas exported by a member: same crc, no trainer involvement
        res["same_crc"] = late.plan_.crc() == plan.crc()
        got = sync.update(1 if rank == 0 else None, owned, receive=rank != 0, commit=commit, ranks=[0, 2])
        res["got"] = got
        res["d"] = digest(held["state"]) if rank == 2 else None
        res["full"] = digest(owned) if rank == 0 else None
    elif scenario == "publisher":  # the runtime contract: DistributedPublisher on the trainer, EngineReceiver on engines
        from crucible.train.engine import TinyPolicy, TorchEngine

        async def go():
            if rank == 0:
                model = TinyPolicy(64, 16, seed=0)

                class T:
                    def state_dict(self):
                        return {k: v.detach().clone() for k, v in model.state_dict().items()}

                    def reset_optimizer(self):
                        pass

                pub = DistributedPublisher(T(), sync)
                vs = []
                for _ in range(2):
                    with torch.no_grad():
                        for p in model.parameters():
                            p.add_(0.01)
                    vs.append(await pub.publish())
                return {"versions": vs, "d": digest(model.state_dict())}
            eng = TorchEngine(TinyPolicy(64, 16, seed=1), capacity=2, numerics="fp32")
            sync.plan(None)
            rec = EngineReceiver(sync, eng)
            vs = [await rec.receive_once(), await rec.receive_once()]
            return {"versions": vs, "engine_version": eng.version, "d": digest(eng._serve.state_dict())}

        res.update(asyncio.run(go()))
    elif scenario in ("dp_exact", "dp_broadcast"):  # DistributedTorchTrainer: DP=world equals the single-process full batch
        import copy

        from crucible.train.engine import TinyPolicy
        from crucible.train.mismatch import Correction
        from crucible.train.rollout import RolloutRecord, Turn
        from crucible.train.trainer import DistributedTorchTrainer, TorchTrainer, TrainerConfig

        g = torch.Generator().manual_seed(0)
        base = TinyPolicy(48, 16, seed=0)
        recs, groups = [], []
        for i in range(12):  # long prompts, short completions: placement must count forward tokens
            prompt = torch.randint(0, 48, (int(torch.randint(3, 40, (1,), generator=g)),), generator=g).tolist()
            comp = torch.randint(0, 48, (int(torch.randint(1, 9, (1,), generator=g)),), generator=g).tolist()
            with torch.no_grad():
                lp = (base.token_logprobs(prompt, comp) + 0.05 * torch.randn(len(comp), generator=g)).tolist()  # behaviour ≠ current
            recs.append(RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=float(torch.rand(1, generator=g) > 0.5),
                                      turns=[Turn(call_id=f"r{i}:c", actor="agent", prompt_token_ids=prompt, completion_token_ids=comp, logprobs=lp, policy_version=0)]))
            groups.append(f"g{i // 4}")
        out_res = {}
        for corr in ("dis", "ppo"):
            cfg = {"lr": 1e-2, "correction": Correction(corr), "token_budget": 40}
            ref = TorchTrainer(copy.deepcopy(base), TrainerConfig(**cfg, dp=1))
            ref.step(recs, groups)
            tr = DistributedTorchTrainer(copy.deepcopy(base), TrainerConfig(**cfg, dp=dist.get_world_size()), broadcast_batch=scenario == "dp_broadcast")
            give = (recs, groups) if (scenario == "dp_exact" or rank == 0) else ([], [])
            m = tr.step(*give)
            err = max(float((a - b).abs().max()) for a, b in zip(tr.model.state_dict().values(), ref.model.state_dict().values(), strict=True))
            out_res[corr] = {"max_abs_diff_vs_full_batch": err, "d": digest(tr.model.state_dict()), "micro_batches": m["train/micro_batches"],
                             "imbalance": m["train/dp_imbalance"], "loss": m["train/loss"], "changed": digest(tr.model.state_dict()) != digest(base.state_dict())}
        try:
            DistributedTorchTrainer(copy.deepcopy(base), TrainerConfig(dp=dist.get_world_size(), loss_agg="seq_mean"))
            out_res["refused_seq_mean"] = False
        except ValueError:
            out_res["refused_seq_mean"] = True
        res.update(out_res)
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    (out / f"r{rank}.json").write_text(json.dumps(res, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
