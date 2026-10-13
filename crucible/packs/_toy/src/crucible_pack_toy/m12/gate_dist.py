"""M12-infra C on real GPUs: a disaggregated run placed by the planner and launched by ``train.cluster``.

``placement.plan(..., "disaggregated_stream")`` splits the cluster into vLLM engine GPUs and FSDP trainer ranks;
``cluster.process_specs`` gives every process its devices and torch.distributed environment, and ``LocalLauncher``
(one node) or ``cluster.run_node`` (one call per node, e.g. ``srun --ntasks-per-node=1``) starts them with this
module's ``entry``:

- an **engine** process serves the policy with vLLM (``--worker-extension-cls`` for checkpoint loads; tensor parallel
  over every GPU it was given) on port 8100 + engine index, writes its URL to ``<out>/engines/<k>.json`` (the output
  directory must be on storage every node sees) and serves until the driver writes ``done``;
- trainer **rank 0** is the driver: the real ``Runtime`` (``disaggregated_stream``), an ``EnginePool`` over every engine,
  ``SPMDDriver(FSDPTrainer)`` and a ``CheckpointPublisher`` (the collective ``state_dict`` gathers HF-named weights to
  rank 0, bf16 safetensors under ``--ckpt-dir``, the last two versions kept, every engine loads them);
- trainer ranks > 0 ``follow`` (NCCL, FSDP shards).

Scale-out (M12 audit H4 V4/V5/V8/V9/V14, G5 D2/D5/D6/D8): every trainer rank builds the model on the meta device and
rank 0 alone reads the checkpoint into host memory (``fsdp.meta_params`` + ``FSDPTrainer(full_state=)``), so no GPU
ever holds the full fp32 model; the trainer uses the planner's mesh; the SPMD channel opens right after the process
group (the followers then outlive engine start-up and the "before" evaluation); ``cluster.init_rank`` sets the device
from ``LOCAL_RANK`` and an explicit process-group timeout.

The task is ``WordFileTask`` (learnable by a 0.6B policy) or the helpdesk backlog (``--task backlog``: multi-ticket
enterprise tool use, hard enough for an 8B policy), with held-out evaluation before and after training.

    python -m crucible_pack_toy.m12.gate_dist --gpus 4 --steps 20 --eval 24 --out /work/out
    srun --nodes=2 --ntasks-per-node=1 python -m crucible_pack_toy.m12.gate_dist --nodes 2 --gpus 4 --cluster repacss --out /lustre/run
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
    import socket

    ext = str(Path(__file__).parent / "_ext")
    port = 8100 + spec.engine
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1", "PYTHONPATH": f"{ext}:{os.environ.get('PYTHONPATH', '')}"}
    if os.environ.get("CRUCIBLE_ENGINE_NCCL_CUMEM_ENABLE"):  # OpenRLHF: NCCL_CUMEM_ENABLE=0 breaks NCCL >= 2.27 under vLLM >= 0.16
        env["NCCL_CUMEM_ENABLE"] = os.environ["CRUCIBLE_ENGINE_NCCL_CUMEM_ENABLE"]
    log = Path(a["out"]) / f"vllm{spec.engine}.log"
    proc = subprocess.Popen(["vllm", "serve", a["model"], "--port", str(port), "--gpu-memory-utilization", str(a["mem"]), "--max-model-len", str(a["max_len"]),
                             "--tensor-parallel-size", str(len(spec.gpus)),  # every GPU of the engine is visible to this process (H4 V10)
                             "--worker-extension-cls", "crucible_vllm_ext.CrucibleWorkerExtension", "--logprobs-mode", "processed_logprobs",
                             "--dtype", "bfloat16"], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)  # noqa: SIM115 -- the server owns the log for its life
    reg = Path(a["out"]) / "engines"
    reg.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname() if a["nodes"] > 1 else "127.0.0.1"
    (reg / f"{spec.engine}.json").write_text(json.dumps({"engine": spec.engine, "url": f"http://{host}:{port}", "node": spec.node}))
    done = Path(a["out"]) / "done"
    while not done.exists():
        if proc.poll() is not None:
            return {"role": "engine", "engine": spec.engine, "error": log.read_text(errors="replace")[-2000:]}
        time.sleep(2)
    proc.terminate()
    proc.wait(timeout=60)
    return {"role": "engine", "engine": spec.engine, "port": port, "ok": True}


def _trainer(a: dict, rank: int, dev):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from crucible.train.fsdp import FSDPTrainer, meta_params, transformer_blocks
    from crucible.train.hf import HFPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.trainer import TrainerConfig

    # every rank: the model on the meta device (no memory); rank 0: the checkpoint in host memory, scattered shard by shard
    with meta_params():
        pol = HFPolicy(AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(a["model"]), dtype=torch.float32), lm_head_fp32=True,
                       gradient_checkpointing=True, autocast_dtype=torch.bfloat16)
    full = AutoModelForCausalLM.from_pretrained(a["model"], dtype=torch.float32).state_dict() if rank == 0 else None
    world = int(os.environ["WORLD_SIZE"])
    tr = FSDPTrainer(pol, TrainerConfig(lr=a["lr"], correction=Correction("dis"), token_budget=a["token_budget"], dp=world),
                     units=transformer_blocks, reshard=a["reshard"], mesh_shape=tuple(a["mesh"]) if a.get("mesh") else None, device=dev,
                     full_state=full, mixed_precision="bf16" if a.get("bf16_gather") else None, broadcast_batch=True)
    del full
    return tr


def _engine_urls(a: dict, n: int, timeout_s: float = 1800.0) -> list[str]:
    reg = Path(a["out"]) / "engines"
    t0 = time.time()
    while True:
        files = sorted(reg.glob("*.json")) if reg.exists() else []
        if len(files) >= n:
            return [json.loads(f.read_text())["url"] for f in sorted(files, key=lambda f: int(f.stem))]
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"{len(files)} of {n} engines registered under {reg}")
        time.sleep(3)


async def _drive(tr, a: dict, urls: list[str], channel) -> dict:
    import torch
    from transformers import AutoTokenizer

    from crucible.actors.policy import ContextConfig
    from crucible.train.chat import HFChatCodec
    from crucible.train.engine import EnginePool, GenRequest
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.sinks import from_spec
    from crucible.train.spmd import SPMDDriver
    from crucible.train.vllm_http import VLLMServerEngine
    from crucible.train.weights import CheckpointPublisher
    from crucible.train.world_env import world_rollout
    from crucible_pack_toy.longhorizon.task import BacklogTask
    from crucible_pack_toy.runtime import WordFileTask

    tok = AutoTokenizer.from_pretrained(a["model"])
    codec = HFChatCodec(tok, template_kwargs={"enable_thinking": False})
    engines = [VLLMServerEngine(u, a["model"], capacity=128, placement="disaggregated") for u in urls]
    pool = EnginePool(engines)
    drv = SPMDDriver(tr, channel=channel)
    pub = CheckpointPublisher(drv, pool, Path(a["ckpt_dir"] or Path(a["out"]) / "ckpt"), keep=2,
                              post_update_transform=lambda sd: {k: v.to(torch.bfloat16) for k, v in sd.items()})
    tasks: dict[str, object] = {}

    def task_for(prompt: str):
        if prompt not in tasks:
            tasks[prompt] = WordFileTask(seed=int(prompt[1:])) if a["task"] == "wordfile" else BacklogTask(seed=int(prompt[1:]), n=a["tickets"])
        return tasks[prompt]

    rollout = world_rollout(task_for, codec, max_steps=a["max_steps"], max_tokens=a["max_tokens"], context=ContextConfig(budget_tokens=a["context"], keep_recent_turns=2))

    async def evaluate(tag: str) -> dict:
        sem = asyncio.Semaphore(32)
        rewards: list[float] = []
        failed: list[str] = []

        async def one(i: int, j: int) -> None:
            rid, calls = f"eval-{tag}-{i}-{j}", [0]

            async def gen(ids, max_tokens, temperature):
                calls[0] += 1
                return await pool.generate(GenRequest(rid=rid, call_id=f"{rid}:c{calls[0]}", prompt_ids=ids, max_tokens=max_tokens, temperature=temperature,
                                                      stop_ids=tuple(codec.eos_ids), seed=None))

            async with sem:
                try:
                    ro = await rollout(rid, f"s{1_000_000 + i}", j, gen, temperature=1.0, harness="eval")
                except Exception as e:  # noqa: BLE001 -- an evaluation episode that fails is masked, as the runtime does, never fatal
                    failed.append(f"{type(e).__name__}: {e}"[:300])
                    return
            if ro is not None and ro.reward is not None:
                rewards.append(float(ro.reward))

        await asyncio.gather(*(one(i, j) for i in range(a["eval"]) for j in range(4)))
        n = len(rewards)
        return {"mean": sum(rewards) / n if n else None, "n": n, "version": engines[0].version, "failed": len(failed), "failures": failed[:3]}

    before = await evaluate("before") if a["eval"] else None
    rt = Runtime(RuntimeConfig(topology="disaggregated_stream", steps=a["steps"], max_turn_tokens=a["max_tokens"], temperature=1.0,
                               stop_ids=tuple(codec.eos_ids), warmup_batches=0, checkpoint_every=a["checkpoint_every"], max_wall_s=a["max_wall_s"],
                               sinks=from_spec(a["sinks"]) if a["sinks"] else [],
                               checkpoint_dir=str(Path(a["out"]) / "runckpt")),
                 SchedulerConfig(n=a["n"], B=a["B"], max_staleness=a["tau"]), pool, drv, pub, None, {a["task"]: [f"s{i}" for i in range(100000)]},
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
           "peak_mem_gb_rank0": round(torch.cuda.max_memory_allocated() / 2**30, 2), "local_numel": tr.local_numel(), "ckpt_deleted": len(pub.deleted),
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

    from crucible.train.cluster import init_rank
    from crucible.train.spmd import follow, open_channel

    rank, _, dev = init_rank("nccl")  # cuda:LOCAL_RANK, explicit timeout (CRUCIBLE_PG_TIMEOUT_S)
    channel = open_channel()  # every rank, now: the followers wait on it while the driver starts engines and evaluates
    tr = _trainer(a, rank, dev)
    if rank != 0:
        counts = follow(tr, channel=channel)
        dist.destroy_process_group()
        return {"role": "trainer", "rank": rank, "follow": counts, "local_numel": tr.local_numel(),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    urls = _engine_urls(a, a["engines"])
    for u in urls:  # every engine healthy before the first rollout
        t0 = time.time()
        while True:
            try:
                if httpx.get(f"{u}/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() - t0 > 900:
                raise RuntimeError(f"engine {u} never became healthy")
            time.sleep(3)
    try:
        out = asyncio.run(_drive(tr, a, urls, channel))
    finally:
        (Path(a["out"]) / "done").write_text("1")
    dist.destroy_process_group()
    return {"role": "driver", "rank": 0, **out}


def main() -> None:
    from crucible.train.cluster import LocalLauncher, process_specs, run_node
    from crucible.train.placement import ClusterSpec, ModelSpec, Workload, plan

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=4, help="GPUs per node")
    ap.add_argument("--cluster", default="", help="a placement.ClusterSpec preset (e.g. repacss); default: --gpus / --gpu-mem")
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
    ap.add_argument("--bf16-gather", dest="bf16_gather", action="store_true", help="FSDP2 MixedPrecisionPolicy: bf16 gathers, fp32 reduce")
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--max-len", dest="max_len", type=int, default=8192)
    ap.add_argument("--eval", type=int, default=24)
    ap.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=10)
    ap.add_argument("--max-wall-s", dest="max_wall_s", type=float, default=7200.0)
    ap.add_argument("--task", choices=["wordfile", "backlog"], default="wordfile")
    ap.add_argument("--sinks", default="", help="train.sinks spec, e.g. jsonl:DIR (step metrics as the run goes)")
    ap.add_argument("--tickets", type=int, default=3)
    ap.add_argument("--params-b", dest="params_b", type=float, default=0.6, help="model size for the planner")  # the runtime's default (600 s) stopped a 1.7B run at step 9
    ap.add_argument("--master-port", dest="master_port", type=int, default=29500)
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="", help="published weights (default <out>/ckpt; every engine node must read it)")
    ap.add_argument("--timeout-s", dest="timeout_s", type=float, default=0.0, help="launcher wall clock (default max-wall-s + 1 h)")
    ap.add_argument("--out", default="/work/out")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    c = ClusterSpec.preset(a.cluster, a.nodes) if a.cluster else ClusterSpec(a.nodes, gpus_per_node=a.gpus, gpu_mem_gb=a.gpu_mem)
    m = ModelSpec(a.params_b, 36 if a.params_b > 4 else 28, 8, 128)
    w = Workload(a.n * a.B, 512, a.max_tokens * a.max_steps, reshard=a.reshard)
    p = plan(c, m, w, "disaggregated_stream")
    if a.train_gpus:  # override the split, keep the planner's engine size
        from crucible.train.placement import _candidate, _layout

        p = _candidate(c, m, w, "disaggregated_stream", c.gpus - a.train_gpus, a.train_gpus, 0)
        p.layout = _layout(c, p.topology, p.engine_gpus, p.rollout_gpus, p.train_gpus, 0)
    args = {**vars(a), "mesh": list(p.mesh), "engines": p.engines}
    os.environ[ARGS_ENV] = json.dumps(args)
    env = {ARGS_ENV: json.dumps(args)}
    t0 = time.time()
    kw = {"extra_env": env, "timeout_s": a.timeout_s or a.max_wall_s + 3600, "log_dir": Path(a.out) / "proc_logs"}
    if a.nodes > 1:  # this node's processes; srun runs main once per node (MASTER_ADDR: the job's first host)
        res = run_node(p, "crucible_pack_toy.m12.gate_dist:entry", master_port=a.master_port, **kw)
    else:
        res = LocalLauncher(**kw).run(process_specs(p, master_port=a.master_port), "crucible_pack_toy.m12.gate_dist:entry")
    out = {"placement": p.explain(), "layout": p.layout, "wall_s": round(time.time() - t0, 1), "processes": res}
    node = os.environ.get("SLURM_NODEID", "0")
    (Path(a.out) / f"gate_dist-node{node}.json").write_text(json.dumps(out, indent=1, default=str))
    drv = next((r for r in res if r and r.get("role") == "driver"), None)
    if drv is not None:
        (Path(a.out) / "gate_dist.json").write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps({"placement": out["placement"], **{k: drv.get(k) for k in ("pass", "steps", "eval", "reward_first3", "reward_last3", "max_staleness", "wall_s")}},
                         indent=1, default=str))


if __name__ == "__main__":
    main()
