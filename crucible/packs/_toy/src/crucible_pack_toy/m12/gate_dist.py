"""M12-infra C on real GPUs: a disaggregated run placed by the planner and launched by ``train.cluster``.

One node with G GPUs. ``placement.plan(..., "disaggregated_stream")`` splits them into vLLM engine GPUs and FSDP trainer
ranks; ``cluster.process_specs`` gives every process its devices and torch.distributed environment, and
``LocalLauncher`` starts them with this module's ``entry``:

- an **engine** process serves the policy with vLLM (``--worker-extension-cls`` for checkpoint loads) on port
  8100 + engine index until the driver writes ``done``;
- trainer **rank 0** is the driver: the real ``Runtime`` (``disaggregated_stream``), an ``EnginePool`` over every engine,
  ``SPMDDriver(FSDPTrainer)`` and a ``CheckpointPublisher`` (the collective ``state_dict`` gathers HF-named weights to
  rank 0, bf16 safetensors, every engine loads them);
- trainer ranks > 0 ``follow`` (NCCL, FSDP shards).

The task is ``WordFileTask`` (learnable by a 0.6B policy), with held-out evaluation before and after training.

    python -m crucible_pack_toy.m12.gate_dist --gpus 4 --steps 20 --eval 24 --out /work/out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

ARGS_ENV = "CRUCIBLE_GATE_ARGS"


def _args() -> dict:
    return json.loads(os.environ[ARGS_ENV])


def _engine(spec, a: dict) -> dict:
    ext = str(Path(__file__).parent / "_ext")
    port = 8100 + spec.engine
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1", "PYTHONPATH": f"{ext}:{os.environ.get('PYTHONPATH', '')}"}
    log = Path(a["out"]) / f"vllm{spec.engine}.log"
    proc = subprocess.Popen(["vllm", "serve", a["model"], "--port", str(port), "--gpu-memory-utilization", str(a["mem"]), "--max-model-len", str(a["max_len"]),
                             "--worker-extension-cls", "crucible_vllm_ext.CrucibleWorkerExtension", "--logprobs-mode", "processed_logprobs",
                             "--dtype", "bfloat16"], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    done = Path(a["out"]) / "done"
    while not done.exists():
        if proc.poll() is not None:
            return {"role": "engine", "engine": spec.engine, "error": log.read_text(errors="replace")[-2000:]}
        time.sleep(2)
    proc.terminate()
    proc.wait(timeout=60)
    return {"role": "engine", "engine": spec.engine, "port": port, "ok": True}


def _trainer(a: dict):
    import torch

    from crucible.train.fsdp import FSDPTrainer, transformer_blocks
    from crucible.train.hf import HFPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.trainer import TrainerConfig

    pol = HFPolicy.from_pretrained(a["model"], dtype=torch.float32, device="cuda", lm_head_fp32=True, gradient_checkpointing=True,
                                   autocast_dtype=torch.bfloat16)
    return FSDPTrainer(pol, TrainerConfig(lr=a["lr"], correction=Correction("dis"), token_budget=a["token_budget"], dp=int(os.environ["WORLD_SIZE"])),
                       units=transformer_blocks, reshard=a["reshard"], broadcast_batch=True)


async def _drive(tr, a: dict, ports: list[int]) -> dict:
    import torch
    from transformers import AutoTokenizer

    from crucible.actors.policy import ContextConfig
    from crucible.train.chat import HFChatCodec
    from crucible.train.engine import EnginePool, GenRequest
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.spmd import SPMDDriver
    from crucible.train.vllm_http import VLLMServerEngine
    from crucible.train.weights import CheckpointPublisher
    from crucible.train.world_env import world_rollout
    from crucible_pack_toy.runtime import WordFileTask

    tok = AutoTokenizer.from_pretrained(a["model"])
    codec = HFChatCodec(tok, template_kwargs={"enable_thinking": False})
    engines = [VLLMServerEngine(f"http://127.0.0.1:{p}", a["model"], capacity=128, placement="disaggregated") for p in ports]
    pool = EnginePool(engines)
    drv = SPMDDriver(tr)
    pub = CheckpointPublisher(drv, pool, Path(a["out"]) / "ckpt", post_update_transform=lambda sd: {k: v.to(torch.bfloat16) for k, v in sd.items()})
    tasks: dict[str, object] = {}

    def task_for(prompt: str):
        if prompt not in tasks:
            tasks[prompt] = WordFileTask(seed=int(prompt[1:]))
        return tasks[prompt]

    rollout = world_rollout(task_for, codec, max_steps=a["max_steps"], max_tokens=a["max_tokens"], context=ContextConfig(budget_tokens=a["context"], keep_recent_turns=2))

    async def evaluate(tag: str) -> dict:
        sem = asyncio.Semaphore(32)
        rewards: list[float] = []

        async def one(i: int, j: int) -> None:
            rid, calls = f"eval-{tag}-{i}-{j}", [0]

            async def gen(ids, max_tokens, temperature):
                calls[0] += 1
                return await pool.generate(GenRequest(rid=rid, call_id=f"{rid}:c{calls[0]}", prompt_ids=ids, max_tokens=max_tokens, temperature=temperature,
                                                      stop_ids=tuple(codec.eos_ids), seed=None))

            async with sem:
                ro = await rollout(rid, f"s{1_000_000 + i}", j, gen, temperature=1.0, harness="eval")
            if ro is not None and ro.reward is not None:
                rewards.append(float(ro.reward))

        await asyncio.gather(*(one(i, j) for i in range(a["eval"]) for j in range(4)))
        n = len(rewards)
        return {"mean": sum(rewards) / n if n else None, "n": n, "version": engines[0].version}

    before = await evaluate("before") if a["eval"] else None
    rt = Runtime(RuntimeConfig(topology="disaggregated_stream", steps=a["steps"], max_turn_tokens=a["max_tokens"], temperature=1.0,
                               stop_ids=tuple(codec.eos_ids), warmup_batches=0, checkpoint_every=a["checkpoint_every"],
                               checkpoint_dir=str(Path(a["out"]) / "runckpt")),
                 SchedulerConfig(n=a["n"], B=a["B"], max_staleness=a["tau"]), pool, drv, pub, None, {"wordfile": [f"s{i}" for i in range(100000)]},
                 rollout_fn=rollout, harness="native:world-qwen@1")
    t0 = time.time()
    rep = await rt.run()
    wall = time.time() - t0
    after = await evaluate("after") if a["eval"] else None
    drv.stop()
    r = [s.metrics.get("reward/mean", 0.0) for s in rep.steps]
    out = {"steps": len(rep.steps), "publishes": rep.publishes, "wall_s": round(wall, 1), "stop_reason": rep.stop_reason,
           "max_staleness": max((s.metrics.get("staleness/max", 0) for s in rep.steps), default=None), "masked": rep.masked_rollouts,
           "reward_first3": sum(r[:3]) / max(1, len(r[:3])), "reward_last3": sum(r[-3:]) / max(1, len(r[-3:])), "eval": {"before": before, "after": after},
           "engine_versions": [e.version for e in engines], "checkpoints": [Path(c).name for c in rep.checkpoints], "spmd_calls": dict(drv.calls),
           "peak_mem_gb_rank0": round(torch.cuda.max_memory_allocated() / 2**30, 2), "local_numel": tr.local_numel(),
           "per_step": [{"step": s.step, **{k: v for k, v in s.metrics.items() if k in ("reward/mean", "staleness/max", "mismatch/rollout_probs_diff",
                                                                                          "train/loss", "train/grad_norm", "step/wall_s")}} for s in rep.steps]}
    out["pass"] = bool(out["steps"] == a["steps"] and out["publishes"] == a["steps"] and out["max_staleness"] is not None and out["max_staleness"] <= a["tau"]
                       and all(v == a["steps"] for v in out["engine_versions"]))
    return out


def entry(spec) -> dict:
    a = _args()
    if spec.role == "engine":
        return _engine(spec, a)
    import torch
    import torch.distributed as dist

    from crucible.train.spmd import follow

    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES holds this rank's GPU only
    dist.init_process_group("nccl", init_method="env://")
    tr = _trainer(a)
    if spec.rank != 0:
        counts = follow(tr)
        dist.destroy_process_group()
        return {"role": "trainer", "rank": spec.rank, "follow": counts, "local_numel": tr.local_numel(),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    ports = [int(p) for p in os.environ["CRUCIBLE_ENGINE_PORTS"].split(",")]
    for p in ports:  # every engine healthy before the first rollout
        t0 = time.time()
        while True:
            try:
                if httpx.get(f"http://127.0.0.1:{p}/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() - t0 > 900:
                raise RuntimeError(f"engine on {p} never became healthy")
            time.sleep(3)
    try:
        out = asyncio.run(_drive(tr, a, ports))
    finally:
        (Path(a["out"]) / "done").write_text("1")
    dist.destroy_process_group()
    return {"role": "driver", "rank": 0, **out}


def main() -> None:
    from crucible.train.cluster import LocalLauncher, free_port, process_specs
    from crucible.train.placement import ClusterSpec, ModelSpec, Workload, plan

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-mem", dest="gpu_mem", type=float, default=24.0)
    ap.add_argument("--train-gpus", dest="train_gpus", type=int, default=0, help="force the split (0: the planner's)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=12)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=256)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--token-budget", dest="token_budget", type=int, default=8192)
    ap.add_argument("--reshard", default="layer")
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--max-len", dest="max_len", type=int, default=8192)
    ap.add_argument("--eval", type=int, default=24)
    ap.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=10)
    ap.add_argument("--out", default="/work/out")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    c = ClusterSpec(1, gpus_per_node=a.gpus, gpu_mem_gb=a.gpu_mem)
    m = ModelSpec(0.6, 28, 8, 128)
    p = plan(c, m, Workload(a.n * a.B, 512, a.max_tokens * a.max_steps), "disaggregated_stream")
    if a.train_gpus:  # override the split, keep the planner's engine size
        from crucible.train.placement import _candidate, _layout

        p = _candidate(c, m, Workload(a.n * a.B, 512, a.max_tokens * a.max_steps), "disaggregated_stream", a.gpus - a.train_gpus, a.train_gpus, 0)
        p.layout = _layout(c, p.topology, p.engine_gpus, p.rollout_gpus, p.train_gpus, 0)
    specs = process_specs(p, master_port=free_port())
    ports = sorted({8100 + s.engine for s in specs if s.role == "engine"})
    os.environ[ARGS_ENV] = json.dumps(vars(a))
    t0 = time.time()
    res = LocalLauncher(extra_env={ARGS_ENV: json.dumps(vars(a)), "CRUCIBLE_ENGINE_PORTS": ",".join(map(str, ports))}, timeout_s=7200).run(
        specs, "crucible_pack_toy.m12.gate_dist:entry")
    out = {"placement": p.explain(), "layout": p.layout, "wall_s": round(time.time() - t0, 1), "processes": res}
    (Path(a.out) / "gate_dist.json").write_text(json.dumps(out, indent=1, default=str))
    drv = next(r for r in res if r and r.get("role") == "driver")
    print(json.dumps({"placement": out["placement"], **{k: drv.get(k) for k in ("pass", "steps", "eval", "reward_first3", "reward_last3", "max_staleness", "wall_s")}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
