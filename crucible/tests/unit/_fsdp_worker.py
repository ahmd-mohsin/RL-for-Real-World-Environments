"""One rank of an FSDP scenario (``test_fsdp.py`` launches WORLD_SIZE of these over gloo):
``python _fsdp_worker.py <scenario> <out_dir>`` with RANK / WORLD_SIZE set (rendezvous: a file store in <out_dir>).
``CRUCIBLE_DIST_BACKEND=nccl``: every scenario on ``cuda:<LOCAL_RANK>`` over NCCL (``scripts/modal_infra_check.py``)."""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.engine import TinyPolicy
from crucible.train.fsdp import FSDPTrainer, transformer_blocks
from crucible.train.mismatch import Correction
from crucible.train.trainer import TorchTrainer, TrainerConfig


class BatchedTiny(TinyPolicy):
    """``TinyPolicy`` with one forward per micro batch (right padding; the GRU is causal, so padding never leaks back)."""

    def batch_token_logprobs(self, pairs, temperature=1.0, candidates=None):
        seqs = [p + c[:-1] if c else p for p, c in pairs]
        dev = self.emb.weight.device
        ids = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, device=dev)
        lp = torch.log_softmax(self(ids) / temperature, -1)
        return [lp[i, len(p) - 1: len(p) - 1 + len(c)].gather(-1, torch.tensor(c, device=dev).unsqueeze(-1)).squeeze(-1) for i, (p, c) in enumerate(pairs)]


def batch(vocab: int, base: torch.nn.Module, n: int = 12, skew: bool = False):
    """``skew``: one 100-token prompt and short ones, so the ranks get different micro-batch counts (padding runs)."""
    g = torch.Generator().manual_seed(0)
    recs, groups = [], []
    for i in range(n):  # long prompts, short completions; behaviour ≠ current (a real correction signal)
        plen = (100 if i == 0 else 3) if skew else int(torch.randint(3, 40, (1,), generator=g))
        prompt = torch.randint(0, vocab, (plen,), generator=g).tolist()
        comp = torch.randint(0, vocab, (int(torch.randint(1, 9, (1,), generator=g)),), generator=g).tolist()
        with torch.no_grad():
            lp = (base.token_logprobs(prompt, comp).float() + 0.05 * torch.randn(len(comp), generator=g)).tolist()
        recs.append(RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=float(torch.rand(1, generator=g) > 0.5),
                                  turns=[Turn(sampling={"temperature": 1.0}, call_id=f"r{i}:c", actor="agent", prompt_token_ids=prompt, completion_token_ids=comp, logprobs=lp, policy_version=0)]))
        groups.append(f"g{i // 4}")
    return recs, groups


def compare(tr: FSDPTrainer, ref: TorchTrainer, rank: int) -> float:
    full = tr.state_dict()  # collective; rank 0 holds it
    if rank != 0:
        return 0.0
    want = {k: v.detach().cpu() for k, v in ref.model.state_dict().items()}
    assert set(full) == set(want), (sorted(full), sorted(want))
    return max(float((full[k].float() - want[k].float()).abs().max()) for k in want)


def main() -> None:
    scenario, out = sys.argv[1], Path(sys.argv[2])
    backend = os.environ.get("CRUCIBLE_DIST_BACKEND", "gloo")
    # ``CRUCIBLE_RENDEZVOUS=env``: MASTER_ADDR / MASTER_PORT (multi-node under Slurm: ``scripts/slurm_check.py unit``)
    init = "env://" if os.environ.get("CRUCIBLE_RENDEZVOUS") == "env" else f"file://{out / 'store'}"
    dist.init_process_group(backend, init_method=init, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
                            timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    # the node-local device (LOCAL_RANK, else rank modulo the visible GPUs): ``cuda:{rank}`` is invalid on a second node (G5 D6)
    local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count()) if backend == "nccl" else 0))
    dev = torch.device(f"cuda:{local}") if backend == "nccl" else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    mesh_dev = dev.type
    res: dict = {"rank": rank, "backend": backend}
    torch.manual_seed(0)
    if scenario in ("fsdp", "hsdp"):
        base = BatchedTiny(48, 16, seed=0)
        recs, groups = batch(48, base)
        mesh = None
        if scenario == "hsdp":
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh(mesh_dev, (2, world // 2), mesh_dim_names=("replicate", "shard"))
        runs = [(m, c, False) for m in (("step", "layer") if scenario == "fsdp" else ("step",)) for c in ("dis", "ppo")]
        runs += [("layer", "ppo", True)] if scenario == "fsdp" else []
        total = sum(p.numel() for p in base.parameters())
        for mode, corr, skew in runs:
            if skew:
                recs, groups = batch(48, base, skew=True)
            if True:
                cfg = {"lr": 1e-2, "correction": Correction(corr), "token_budget": 40}
                ref = TorchTrainer(copy.deepcopy(base).to(dev), TrainerConfig(**cfg, dp=1))
                r1 = ref.step(recs, groups)
                ref.step(recs, groups)  # two updates: the AdamW moments (sharded) carry over correctly
                tr = FSDPTrainer(copy.deepcopy(base).to(dev), TrainerConfig(**cfg, dp=world), mesh=mesh, reshard=mode,
                                 units=lambda m: [m.emb, m.rnn, m.head])
                give = (recs, groups) if rank == 0 else ([], [])
                tr.broadcast_batch = True  # rank 0 owns the batch; the others receive it
                m1 = tr.step(*give)
                tr.step(*give)
                err = compare(tr, ref, rank)
                at_rest = tr.local_numel()
                res["rnn_flatten_disabled"] = tr.model.rnn.flatten_parameters() is None and "lambda" in repr(tr.model.rnn.flatten_parameters)
                res[f"{mode}/{corr}" + ("/skew" if skew else "")] = {"max_abs_diff": err, "micro_batches": m1["train/micro_batches"], "loss": m1["train/loss"], "ref_grad_norm": r1["train/grad_norm"],
                                         "grad_norm": m1["train/grad_norm"], "params_frac": at_rest["params"] / total,
                                         "moments_frac": at_rest["moments"] / (2 * total), **tr.counters}
        try:
            FSDPTrainer(copy.deepcopy(TinyPolicy(48, 16)).to(dev), TrainerConfig(dp=world), reshard="layer")
            res["refused_unbatched_layer"] = False
        except ValueError:
            res["refused_unbatched_layer"] = True
    elif scenario == "few_items":  # G5 D1 / H4 V1 at the world size of the run: fewer trained rows than ranks
        base = BatchedTiny(48, 16, seed=0)
        n = max(1, world - 1)
        recs, groups = batch(48, base, n=n)
        for mode in ("step", "layer"):
            cfg = {"lr": 1e-2, "correction": Correction("dis"), "token_budget": 40}
            ref = TorchTrainer(copy.deepcopy(base).to(dev), TrainerConfig(**cfg, dp=1))
            ref.step(recs, groups)
            tr = FSDPTrainer(copy.deepcopy(base).to(dev), TrainerConfig(**cfg, dp=world), reshard=mode, units=lambda m: [m.emb, m.rnn, m.head],
                             broadcast_batch=True)
            m1 = tr.step(*((recs, groups) if rank == 0 else ([], [])))
            res[mode] = {"items": n, "max_abs_diff": compare(tr, ref, rank), "loss": m1["train/loss"]}
    elif scenario == "fsdp_debug":  # step mode, one update at a time: where does it leave the reference?
        base = BatchedTiny(48, 16, seed=0)
        recs, groups = batch(48, base)
        cfg = {"lr": 1e-2, "correction": Correction("dis"), "token_budget": 40}
        ref = TorchTrainer(copy.deepcopy(base).to(dev), TrainerConfig(**cfg, dp=1))
        tr = FSDPTrainer(copy.deepcopy(base).to(dev), TrainerConfig(**cfg, dp=world), reshard="step", units=lambda m: [m.emb, m.rnn, m.head],
                         broadcast_batch=True)
        give = (recs, groups) if rank == 0 else ([], [])
        for k in range(3):
            r = ref.step(recs, groups)
            m = tr.step(*give)
            res[f"step{k}"] = {"diff": compare(tr, ref, rank), "gn": m["train/grad_norm"], "ref_gn": r["train/grad_norm"], "loss": m["train/loss"], "ref_loss": r["train/loss"]}
            tr._gather_params()  # gathered copy vs the shards' full tensors: stale after the optimizer step?
            with torch.no_grad():
                lp_g = tr.model.batch_token_logprobs([(recs[0].turns[0].prompt_token_ids, recs[0].turns[0].completion_token_ids)])[0]
                lp_r = ref.model.batch_token_logprobs([(recs[0].turns[0].prompt_token_ids, recs[0].turns[0].completion_token_ids)])[0]
            res[f"step{k}"]["forward_diff"] = float((lp_g - lp_r).abs().max())
            tr._release_params()
    elif scenario == "hf_layer":  # a real HF decoder (random init from a config, no download): blocks as units, ZeRO-3 forwards
        from transformers import LlamaConfig, LlamaForCausalLM

        from crucible.train.hf import HFPolicy

        cfg_hf = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False)
        torch.manual_seed(0)
        base = HFPolicy(LlamaForCausalLM(cfg_hf).float(), lm_head_fp32=True)
        recs, groups = batch(64, base, n=8)
        cfg = {"lr": 1e-3, "correction": Correction("ppo"), "token_budget": 48}
        ref = TorchTrainer(copy.deepcopy(base).to(dev), TrainerConfig(**cfg, dp=1))
        ref.step(recs, groups)
        tr = FSDPTrainer(copy.deepcopy(base).to(dev), TrainerConfig(**cfg, dp=world), reshard="layer", units=transformer_blocks)
        m = tr.step(recs, groups)
        res["units"] = len(transformer_blocks(base))
        res["max_abs_diff"] = compare(tr, ref, rank)
        res["loss"] = m["train/loss"]
        res["probe_micro_batches"] = tr.counters["probe_micro_batches"]
        res["names"] = sorted(ref.model.state_dict())[:3]
    elif scenario == "spmd_run":  # rank 0 drives the real Runtime; every rank is an HSDP trainer shard; checkpoint + resume
        import asyncio

        from torch.distributed.device_mesh import init_device_mesh

        from crucible.train.engine import EnginePool, TorchEngine
        from crucible.train.envs import CopyTask
        from crucible.train.runtime import Runtime, RuntimeConfig
        from crucible.train.scheduler import SchedulerConfig
        from crucible.train.spmd import SPMDDriver, follow
        from crucible.train.weights import InprocPublisher

        mesh = init_device_mesh(mesh_dev, (2, world // 2), mesh_dim_names=("replicate", "shard"))
        ck = out / "ckpt"

        def trainer():
            torch.manual_seed(0)
            return FSDPTrainer(TinyPolicy(32, 48, seed=1).to(dev), TrainerConfig(lr=1e-2, dp=world), mesh=mesh, broadcast_batch=True)

        if rank != 0:
            res["first"] = follow(trainer())
            t2 = trainer()
            res["second"] = follow(t2)
            res["updates_after_run2"] = t2.updates
        else:
            published: dict[int, dict] = {}

            def build(tr, steps, every):
                eng = TorchEngine(TinyPolicy(32, 48, seed=1).to(dev), capacity=32, numerics="bf16", decode_delay_s=0.001)
                pool = EnginePool([eng])
                drv = SPMDDriver(tr)
                pub = InprocPublisher(drv, pool, post_update_transform=lambda st: published.setdefault(pub.version + 1, {k: v.clone() for k, v in st.items()}) and st)
                rt = Runtime(RuntimeConfig(topology="disaggregated_stream", steps=steps, max_turn_tokens=3, checkpoint_every=every, checkpoint_dir=str(ck)),
                             SchedulerConfig(n=4, B=4, max_staleness=2), pool, drv, pub,
                             lambda: CopyTask(turns=2, symbols=8, latency_s=0.001, long_tail=0.2), {"copy": [f"p{i}" for i in range(4000)]})
                return rt, drv

            rt, drv = build(trainer(), 16, 8)
            rep = asyncio.run(rt.run())
            rw = rep.series("reward/mean")
            res.update(steps=len(rep.steps), publishes=rep.publishes, first=sum(rw[:4]) / 4, last=sum(rw[-6:]) / 6,
                       max_staleness=max(rep.series("staleness/max")), checkpoints=[Path(c).name for c in rep.checkpoints], calls=dict(drv.calls))
            drv.stop()
            v8 = published[8]
            rt2, drv2 = build(trainer(), 12, 0)
            start = asyncio.run(rt2.resume(str(ck / "step-000008")))
            got = drv2.state_dict()  # collective: the restored shards, gathered
            res.update(resumed_at=start, resumed_version=rt2.version, restore_diff=max(float((got[k].cpu() - v8[k].cpu()).abs().max()) for k in v8),
                       resumed_updates=drv2.trainer.updates)  # the trainer's own counter (H4 V3: the driver kept a shadow)
            rep2 = asyncio.run(rt2.run())
            res.update(steps2=len(rep2.steps), first_step2=rep2.steps[0].step, updates_after_run2=drv2.trainer.updates)
            drv2.stop()
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    (out / f"r{rank}.json").write_text(json.dumps(res, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
