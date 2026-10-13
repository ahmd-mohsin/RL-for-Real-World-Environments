"""One rank of a ``test_m12_audit_scale.py`` scenario (M12 audit: the multi-GPU / scale-out path on gloo):
``python _m12_scale_worker.py <scenario> <out_dir> '<json args>'`` with RANK / WORLD_SIZE set; rendezvous through a file
store in <out_dir>; each rank writes ``r<rank>.json``."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from _fsdp_worker import BatchedTiny, batch

from crucible.train.mismatch import Correction
from crucible.train.trainer import DistributedTorchTrainer, TorchTrainer, TrainerConfig


class Gated(BatchedTiny):
    """``BatchedTiny`` plus an expert reached only by items whose prompt contains token 0 (data-dependent control flow)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.expert = torch.nn.Linear(16, 48)

    def batch_token_logprobs(self, pairs, temperature=1.0, candidates=None):
        seqs = [p + c[:-1] if c else p for p, c in pairs]
        dev = self.emb.weight.device
        ids = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, device=dev)
        h, _ = self.rnn(self.emb(ids))
        logits = self.head(h)
        routed = torch.tensor([0 in p for p, _ in pairs], device=dev)
        if bool(routed.any()):  # per item: only routed rows get the expert (a micro batch's composition never matters)
            logits = logits + self.expert(h) * routed[:, None, None].to(h.dtype)
        lp = torch.log_softmax(logits / temperature, -1)
        return [lp[i, len(p) - 1: len(p) - 1 + len(c)].gather(-1, torch.tensor(c, device=dev).unsqueeze(-1)).squeeze(-1) for i, (p, c) in enumerate(pairs)]

    def token_logprobs(self, prompt, completion, temperature=1.0, candidates=None):
        return self.batch_token_logprobs([(prompt, completion)], temperature)[0]


def routed_batch(base, *, route: bool):
    """8 items; only item 0 (the longest prompt: rank 0's partition) contains token 0 when ``route``."""
    recs, groups = batch(48, base, n=8)

    def no0(xs):
        return [1 if x == 0 else x for x in xs]

    for i in range(len(recs)):
        t = recs[i].turns[0]
        prompt = no0(t.prompt_token_ids)
        if i == 0:
            prompt = ([0] if route else [1]) + prompt[1:] + [1] * 60
        recs[i] = recs[i].model_copy(update={"turns": [t.model_copy(update={"prompt_token_ids": prompt, "completion_token_ids": no0(t.completion_token_ids)})]})
    return recs, groups


def full_state(tr) -> dict:
    from crucible.train.fsdp import FSDPTrainer

    if isinstance(tr, FSDPTrainer):
        return tr.state_dict()
    return {k: v.detach().cpu() for k, v in tr.model.state_dict().items()}


def diff(a: dict, b: dict) -> float:
    return max(float((a[k].float().cpu() - b[k].float().cpu()).abs().max()) for k in b)


def main() -> None:
    scenario, out = sys.argv[1], Path(sys.argv[2])
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", init_method=f"file://{out / 'store'}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=float(args.get("pg_timeout", 60))))
    res: dict = {"rank": rank, "world": world}
    torch.manual_seed(0)
    try:
        run(scenario, out, args, rank, world, res)
    except Exception as e:  # noqa: BLE001 -- the test reads the error
        res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        res["tb"] = traceback.format_exc()[-2000:]
    (out / f"r{rank}.json").write_text(json.dumps(res, default=str))
    if not args.get("no_destroy"):
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001, S110 -- best effort at exit
            pass


def run(scenario: str, out: Path, args: dict, rank: int, world: int, res: dict) -> None:
    from crucible.train.fsdp import FSDPTrainer

    cfg = {"lr": 1e-2, "correction": Correction(args.get("corr", "dis")), "token_budget": args.get("budget", 4096)}
    if scenario == "unused":  # V2: a parameter only rank 0's item reaches
        base = Gated(48, 16, seed=0)
        recs, groups = routed_batch(base, route=args.get("route", True))
        ref = TorchTrainer(copy.deepcopy(base), TrainerConfig(**cfg, dp=1))
        ref.step(recs, groups)
        units = (lambda m: [m.emb, m.rnn, m.head, m.expert]) if args.get("expert_unit") else (lambda m: [m.emb, m.rnn, m.head])
        kw = {"probe": lambda mdl: mdl.batch_token_logprobs([([2, 2], [2])])[0].sum()} if args.get("probe_no_expert", True) else {}
        tr = FSDPTrainer(copy.deepcopy(base), TrainerConfig(**cfg, dp=world), reshard=args["mode"], units=units, broadcast_batch=True, **kw)
        m = tr.step(*((recs, groups) if rank == 0 else ([], [])))
        res["loss"] = m["train/loss"]
        fs = full_state(tr)
        if rank == 0:
            want = {k: v.detach() for k, v in ref.model.state_dict().items()}
            res["max_abs_diff"] = diff(fs, want)
            res["expert_moved"] = float((fs["expert.weight"] - base.expert.weight.detach()).abs().max())
            res["ref_expert_moved"] = float((want["expert.weight"] - base.expert.weight.detach()).abs().max())
            res["per_key"] = {k: float((fs[k].float() - want[k].float()).abs().max()) for k in want}
        res["zero_filled"] = tr.counters.get("zero_filled_grads")
    elif scenario == "meta_load":  # V4: meta-device model, full state on rank 0 only, no replica broadcast
        from crucible.train.fsdp import meta_params

        base = BatchedTiny(48, 16, seed=0)
        recs, groups = batch(48, base)
        ref = TorchTrainer(copy.deepcopy(base), TrainerConfig(**{**cfg, "token_budget": 40}, dp=1))
        ref.step(recs, groups)
        with meta_params():
            empty = BatchedTiny(48, 16, seed=7)
        res["built_on_meta"] = all(p.is_meta for p in empty.parameters())
        calls = {"broadcast": 0}
        orig_bc = dist.broadcast

        def counting(t, *a, **k):
            calls["broadcast"] += 1
            return orig_bc(t, *a, **k)

        dist.broadcast = counting
        tr = FSDPTrainer(empty, TrainerConfig(**{**cfg, "token_budget": 40}, dp=world), reshard=args.get("mode", "layer"),
                         units=lambda m: [m.emb, m.rnn, m.head], broadcast_batch=True,
                         full_state={k: v.detach().clone() for k, v in base.state_dict().items()} if rank == 0 else None)
        dist.broadcast = orig_bc
        res["init_broadcasts"] = calls["broadcast"]
        res["n_params"] = len(list(base.parameters()))
        res["local_frac"] = tr.local_numel()["params"] / sum(p.numel() for p in base.parameters())
        loaded = full_state(tr)
        if rank == 0:
            res["load_diff"] = diff(loaded, {k: v.detach() for k, v in base.state_dict().items()})
        tr.step(*((recs, groups) if rank == 0 else ([], [])))
        fs = full_state(tr)
        if rank == 0:
            res["max_abs_diff"] = diff(fs, {k: v.detach() for k, v in ref.model.state_dict().items()})
    elif scenario == "meta_hf":  # V4 on a real HF decoder: rotary buffers survive, tied embeddings, HF names
        from transformers import LlamaConfig, LlamaForCausalLM

        from crucible.train.fsdp import meta_params, transformer_blocks
        from crucible.train.hf import HFPolicy

        cfg_hf = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=bool(args.get("tied")))
        torch.manual_seed(0)
        base = HFPolicy(LlamaForCausalLM(cfg_hf).float(), lm_head_fp32=True)
        recs, groups = batch(64, base, n=8)
        hcfg = {"lr": 1e-3, "correction": Correction("ppo"), "token_budget": 48}
        ref = TorchTrainer(copy.deepcopy(base), TrainerConfig(**hcfg, dp=1))
        ref.step(recs, groups)
        with meta_params():
            empty = HFPolicy(LlamaForCausalLM(cfg_hf).float(), lm_head_fp32=True)
        tr = FSDPTrainer(empty, TrainerConfig(**hcfg, dp=world), reshard="layer", units=transformer_blocks, broadcast_batch=True,
                         full_state=base.state_dict() if rank == 0 else None)
        tr.step(*((recs, groups) if rank == 0 else ([], [])))
        fs = full_state(tr)
        if rank == 0:
            want = {k: v.detach() for k, v in ref.model.state_dict().items()}
            res["keys_equal"] = sorted(fs) == sorted(want)
            res["max_abs_diff"] = diff(fs, {k: v for k, v in want.items() if k in fs})
    elif scenario == "offload":  # V7: offload/onload between steps, then the update equals the reference
        base = BatchedTiny(48, 16, seed=0)
        recs, groups = batch(48, base)
        c2 = {**cfg, "token_budget": 40}
        ref = TorchTrainer(copy.deepcopy(base), TrainerConfig(**c2, dp=1))
        tr = FSDPTrainer(copy.deepcopy(base), TrainerConfig(**c2, dp=world), reshard=args.get("mode", "step"),
                         units=lambda m: [m.emb, m.rnn, m.head], broadcast_batch=True)
        give = (recs, groups) if rank == 0 else ([], [])
        ids = [id(p) for p in tr.model.parameters()]
        for _ in range(2):
            ref.step(recs, groups)
            tr.step(*give)
            before = {n: p.to_local().clone() for n, p in tr.model.named_parameters()}
            tr.offload("cpu")
            res["offloaded_device"] = sorted({p.to_local().device.type for p in tr.model.parameters()})
            res["grads_dropped"] = all(p.grad is None for p in tr.model.parameters())
            tr.onload()
            res["roundtrip_exact"] = all(torch.equal(before[n], p.to_local()) for n, p in tr.model.named_parameters())
        res["same_param_objects"] = ids == [id(p) for p in tr.model.parameters()]
        fs = full_state(tr)
        if rank == 0:
            res["max_abs_diff"] = diff(fs, {k: v.detach() for k, v in ref.model.state_dict().items()})
    elif scenario == "plan_kwargs":  # V14: the planner's kwargs build the trainer (HSDP mesh, dp, reshard)
        from crucible.train.placement import ClusterSpec, ModelSpec, Workload, plan

        p = plan(ClusterSpec(1, gpus_per_node=world, gpu_mem_gb=1.7e-4), ModelSpec(1e-5, 2, 1, 4), Workload(8, 16, 16),
                 "colocated_partial")
        res["mesh"] = list(p.mesh)
        base = BatchedTiny(48, 16, seed=0)
        recs, groups = batch(48, base)
        c2 = {**cfg, "token_budget": 40}
        ref = TorchTrainer(copy.deepcopy(base), TrainerConfig(**c2, dp=1))
        ref.step(recs, groups)
        kw = p.trainer_kwargs()
        res["kwargs"] = {k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()}
        tr = FSDPTrainer(copy.deepcopy(base), TrainerConfig(**c2, **p.trainer_config()), units=lambda m: [m.emb, m.rnn, m.head], broadcast_batch=True, **kw)
        res["mesh_ndim"] = tr.mesh.ndim
        tr.step(*((recs, groups) if rank == 0 else ([], [])))
        res["params_frac"] = tr.local_numel()["params"] / sum(q.numel() for q in base.parameters())
        fs = full_state(tr)
        if rank == 0:
            res["max_abs_diff"] = diff(fs, {k: v.detach() for k, v in ref.model.state_dict().items()})
    elif scenario == "mixed_precision":  # P2-a: bf16 gathers for the blocks, fp32 root (embeddings, LM head), fp32 master
        from transformers import LlamaConfig, LlamaForCausalLM

        from crucible.train.fsdp import transformer_blocks
        from crucible.train.hf import HFPolicy

        cfg_hf = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False)
        torch.manual_seed(0)
        base = HFPolicy(LlamaForCausalLM(cfg_hf).float(), lm_head_fp32=True)
        recs, groups = batch(64, base, n=8)
        hcfg = {"lr": 1e-3, "correction": Correction("dis"), "token_budget": 48}
        ref = TorchTrainer(copy.deepcopy(base), TrainerConfig(**hcfg, dp=1))
        ref.step(recs, groups)
        tr = FSDPTrainer(copy.deepcopy(base), TrainerConfig(**hcfg, dp=world), reshard=args.get("mode", "layer"), units=transformer_blocks,
                         mixed_precision="bf16", broadcast_batch=True)
        seen: dict[str, str] = {}
        blk = transformer_blocks(tr.model)[0]
        blk.register_forward_pre_hook(lambda m, a: seen.setdefault("block", str(m.mlp.gate_proj.weight.dtype)) and None)
        orig = tr.model._head_rows

        def spy(h, temperature, _o=orig):
            seen.setdefault("head", str(tr.model.model.get_output_embeddings().weight.dtype))
            return _o(h, temperature)

        tr.model._head_rows = spy
        m = tr.step(*((recs, groups) if rank == 0 else ([], [])))
        res["dtypes"] = seen
        res["master_dtype"] = sorted({str(p.dtype) for p in tr.model.parameters()})
        res["loss"] = m["train/loss"]
        fs = full_state(tr)
        if rank == 0:
            res["max_abs_diff"] = diff(fs, {k: v.detach() for k, v in ref.model.state_dict().items()})
    elif scenario in ("resume_ddp", "resume_fsdp"):  # V3: resume through SPMDDriver
        resume_scenario(scenario, out, args, rank, world, res)
    elif scenario in ("late_driver", "driver_dies", "follower_dies", "offload_path"):
        spmd_scenario(scenario, out, args, rank, world, res)
    else:
        raise SystemExit(f"unknown scenario {scenario}")


def resume_scenario(scenario: str, out: Path, args: dict, rank: int, world: int, res: dict) -> None:
    import asyncio

    from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
    from crucible.train.envs import CopyTask
    from crucible.train.fsdp import FSDPTrainer
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.spmd import SPMDDriver, follow
    from crucible.train.weights import InprocPublisher

    ck = out / "ckpt"

    def trainer():
        torch.manual_seed(0)
        if scenario == "resume_fsdp":
            return FSDPTrainer(TinyPolicy(32, 48, seed=1), TrainerConfig(lr=1e-2, dp=world), broadcast_batch=True)
        return DistributedTorchTrainer(TinyPolicy(32, 48, seed=1), TrainerConfig(lr=1e-2, dp=world), broadcast_batch=True)

    def spread(tr):
        if scenario == "resume_fsdp":
            return 0.0
        vec = torch.cat([p.detach().flatten() for p in tr.model.parameters()])
        allv = [torch.zeros_like(vec) for _ in range(world)]
        dist.all_gather(allv, vec)
        return max(float((v - allv[0]).abs().max()) for v in allv)

    if rank != 0:
        t1 = trainer()
        res["first"] = follow(t1)
        spread(t1)
        t2 = trainer()
        res["second"] = follow(t2)
        res["follower_updates"] = t2.updates
        spread(t2)
        return

    def mk(tr, steps, every):
        eng = TorchEngine(TinyPolicy(32, 48, seed=1), capacity=32, numerics="bf16", decode_delay_s=0.001)
        pool = EnginePool([eng])
        drv = SPMDDriver(tr)
        pub = InprocPublisher(drv, pool)
        rt = Runtime(RuntimeConfig(topology="disaggregated_stream", steps=steps, max_turn_tokens=3, checkpoint_every=every, checkpoint_dir=str(ck)),
                     SchedulerConfig(n=4, B=4, max_staleness=2), pool, drv, pub,
                     lambda: CopyTask(turns=2, symbols=8, latency_s=0.001, long_tail=0.2), {"copy": [f"p{i}" for i in range(4000)]})
        return rt, drv

    t1 = trainer()
    rt, drv = mk(t1, 4, 2)
    asyncio.run(rt.run())
    drv.stop()
    res["spread_after_run1"] = spread(t1)
    t2 = trainer()
    rt2, drv2 = mk(t2, 6, 0)
    asyncio.run(rt2.resume(str(ck / "step-000004")))
    res["updates_after_resume"] = [drv2.updates, t2.updates]
    asyncio.run(rt2.run())
    drv2.stop()
    res["spread_after_resume_run"] = spread(t2)
    res["driver_updates_attr"] = drv2.updates
    res["trainer_updates_attr"] = t2.updates


class Recorder:
    """A stand-in trainer that records the calls ``follow`` makes."""

    broadcast_batch = True

    def __init__(self) -> None:
        self.calls: list = []
        self.updates = 0

    def offload(self, mode="cpu", *, path=None):
        self.calls.append(["offload", mode, path])

    def onload(self):
        self.calls.append(["onload"])

    def step(self, records, groups):
        self.calls.append(["step"])
        return {}


def spmd_scenario(scenario: str, out: Path, args: dict, rank: int, world: int, res: dict) -> None:
    from crucible.train import spmd

    live = {"period_s": float(args.get("period", 0.2)), "stale_s": float(args.get("stale", 2.0))}
    if scenario == "offload_path":  # V13: the offload path reaches every rank
        tr = Recorder()
        if rank == 0:
            drv = spmd.SPMDDriver(tr, **live)
            drv.offload("nvme", path="/shared/offload")
            drv.onload()
            drv.stop()
        else:
            spmd.follow(tr, **live)
        res["calls"] = tr.calls
        return
    base = BatchedTiny(48, 16, seed=0)
    recs, groups = batch(48, base, n=8)
    tr = DistributedTorchTrainer(copy.deepcopy(base), TrainerConfig(lr=1e-2, dp=world), broadcast_batch=True)
    t0 = time.time()
    if scenario == "late_driver":  # V5: followers wait in follow() while the driver is busy for longer than the PG timeout
        ch = spmd.open_channel(**live)  # every rank, right after init_process_group
        if rank == 0:
            time.sleep(float(args.get("delay", 20)))
            drv = spmd.SPMDDriver(tr, channel=ch)
            res["metrics"] = sorted(drv.step(recs, groups))[:3]
            drv.stop()
            res["driver"] = "ok"
        else:
            res["follower"] = spmd.follow(tr, channel=ch)
    elif scenario == "driver_dies":  # V6: followers notice a dead driver
        ch = spmd.open_channel(**live)
        if rank == 0:
            spmd.SPMDDriver(tr, channel=ch)
            time.sleep(1.0)
            (out / "r0.json").write_text(json.dumps({"rank": 0, "driver": "exiting hard"}))
            os._exit(1)
        try:
            spmd.follow(tr, channel=ch)
            res["follower"] = "returned"
        except Exception as e:  # noqa: BLE001
            res["follower"] = f"{type(e).__name__}: {str(e)[:200]}"
    elif scenario == "follower_dies":  # V6: a follower dies mid-step; the driver is blocked in a collective
        ch = spmd.open_channel(**live)
        if rank == 1:
            def boom():
                (out / "r1.json").write_text(json.dumps({"rank": 1, "follower": "dying in the step"}))
                os._exit(1)

            tr._reduce_grads = boom
            spmd.follow(tr, channel=ch)
        else:
            drv = spmd.SPMDDriver(tr, channel=ch)
            (out / "driver_started").write_text(str(time.time()))
            drv.step(recs, groups)
            res["driver"] = "returned"
    res["wait_s"] = round(time.time() - t0, 1)


def fail_or_sleep(spec) -> dict:
    """A ``LocalLauncher`` entry: trainer rank 0 fails at once, every other process would sleep for a minute."""
    if spec.rank == 0:
        raise RuntimeError("rank 0 failed")
    time.sleep(60)
    return {"rank": spec.rank}


if __name__ == "__main__":
    main()
