"""Calibrate the placement planner's first-order terms on real GPUs (Modal): decode speed per sequence vs batch, trainer
MFU under FSDP, checkpoint publish + engine load time, and NCCL broadcast bandwidth.

    CRUCIBLE_CAL_GPU=A10G:4 modal run scripts/modal_calibrate.py --model Qwen/Qwen3-1.7B
    CRUCIBLE_CAL_GPU=H100:4 modal run scripts/modal_calibrate.py --model Qwen/Qwen3-8B --peak-tflops 989

GPU 0 serves the model with vLLM; GPUs 1..N-1 are FSDP trainer ranks (layer mode, HF blocks as units). Results merge into
``docs/research/m12-infra/02-live-validation.json`` under ``calibration`` and print as the planner's ``Workload`` /
``ClusterSpec`` numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.from_registry("vllm/vllm-openai:latest").entrypoint([])
         .run_commands("ln -sf $(command -v python3) /usr/local/bin/python && python --version")
         .add_local_dir(ROOT / "src", "/root/crucible/src", ignore=["**/__pycache__"], copy=True)
         .add_local_dir(ROOT / "packs" / "_toy", "/root/crucible/packs/_toy", ignore=["**/__pycache__", "**/.venv"], copy=True)
         .add_local_file(ROOT / "pyproject.toml", "/root/crucible/pyproject.toml", copy=True)
         .add_local_file(ROOT / "README.md", "/root/crucible/README.md", copy=True)
         .run_commands("pip install -q -e /root/crucible -e /root/crucible/packs/_toy")
         .env({"NCCL_CUMEM_ENABLE": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
hf_cache = modal.Volume.from_name("crucible-hf-cache", create_if_missing=True)
app = modal.App("crucible-calibrate")

TRAIN_WORKER = r'''
import json, os, sys, time
import torch, torch.distributed as dist
from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.fsdp import FSDPTrainer, transformer_blocks
from crucible.train.hf import HFPolicy
from crucible.train.mismatch import Correction
from crucible.train.trainer import TrainerConfig
from crucible.train.weightsync import BroadcastWeightSync
a = json.loads(os.environ["CAL_ARGS"])
torch.cuda.set_device(0)
dist.init_process_group("nccl", init_method="env://")
rank, world = dist.get_rank(), dist.get_world_size()
pol = HFPolicy.from_pretrained(a["model"], dtype=torch.float32, device="cuda", lm_head_fp32=True, gradient_checkpointing=True, autocast_dtype=torch.bfloat16)
n_params = sum(p.numel() for p in pol.parameters())
tr = FSDPTrainer(pol, TrainerConfig(lr=1e-6, correction=Correction("dis"), token_budget=a["token_budget"], dp=world), units=transformer_blocks, reshard="layer")
g = torch.Generator().manual_seed(rank)
P, C, S = a["prompt"], a["completion"], a["seqs_per_rank"] * world
recs, groups = [], []
for i in range(S):
    ids = torch.randint(100, 30000, (P + C,), generator=g).tolist()
    recs.append(RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=float(i % 2),
                              turns=[Turn(call_id=f"r{i}:c", actor="agent", prompt_token_ids=ids[:P], completion_token_ids=ids[P:], logprobs=[-1.0] * C, policy_version=0)]))
    groups.append(f"g{i // 4}")
times = []
for k in range(a["train_steps"]):
    torch.cuda.synchronize(); t0 = time.time()
    m = tr.step(recs, groups)
    torch.cuda.synchronize(); times.append(time.time() - t0)
tok = S * (P + C)
steady = sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else times[0]
out = {"rank": rank, "world": world, "params": n_params, "step_s": [round(t, 2) for t in times], "steady_step_s": round(steady, 3),
       "tokens_per_step": tok, "train_tok_s": round(tok / steady, 1), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "local_numel": tr.local_numel()}
# trainer-side NCCL broadcast bandwidth: rank 0 owns the bf16 weights, every rank receives (the disaggregated publish path)
full = tr.state_dict()
owned = {k: v.to(torch.bfloat16) for k, v in full.items()} if rank == 0 else None
sync = BroadcastWeightSync(device="cuda")
sync.plan(owned)
torch.cuda.synchronize(); t0 = time.time()
sync.update(1 if rank == 0 else None, owned, receive=rank != 0)
torch.cuda.synchronize(); bt = time.time() - t0
nbytes = sum(m.nbytes for m in sync.plan_.metas)
out.update(broadcast_s=round(bt, 3), broadcast_gb=round(nbytes / 1e9, 3), broadcast_gbps=round(nbytes / 1e9 / bt, 2))
open(f"/work/out/train_r{rank}.json", "w").write(json.dumps(out))
dist.destroy_process_group()
'''


@app.function(image=image, gpu=os.environ.get("CRUCIBLE_CAL_GPU", "A10G:4"), timeout=5400, volumes={"/root/.cache/huggingface": hf_cache})
def calibrate(a: dict) -> dict:
    import asyncio
    import subprocess
    import sys
    import time

    import httpx
    import torch

    out_dir = Path("/work/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = torch.cuda.device_count()
    res: dict = {"gpu": torch.cuda.get_device_name(0), "gpus": gpus, "model": a["model"]}
    # ---- decode speed vs batch (vLLM on GPU 0) --------------------------------------------------------------------
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    log = open(out_dir / "vllm.log", "w")  # noqa: SIM115 -- handed to the server for its lifetime
    srv = subprocess.Popen(["vllm", "serve", a["model"], "--port", "8100", "--gpu-memory-utilization", "0.88", "--max-model-len", "4096", "--dtype", "bfloat16"],
                           env=env, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        try:
            if httpx.get("http://127.0.0.1:8100/health", timeout=5).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if srv.poll() is not None or time.time() - t0 > 1200:
            return {**res, "error": "vllm did not start: " + (out_dir / "vllm.log").read_text()[-2000:]}
        time.sleep(3)
    res["vllm_start_s"] = round(time.time() - t0, 1)

    async def bench(batch: int, out_tok: int) -> dict:
        async with httpx.AsyncClient(timeout=1800) as c:
            async def one(i):
                t = time.time()
                r = await c.post("http://127.0.0.1:8100/v1/completions", json={"model": a["model"], "prompt": f"Story {i}: once upon a time", "max_tokens": out_tok,
                                                                               "min_tokens": out_tok, "temperature": 1.0, "ignore_eos": True})
                r.raise_for_status()
                return time.time() - t, r.json()["usage"]["completion_tokens"]
            t = time.time()
            rs = await asyncio.gather(*(one(i) for i in range(batch)))
            wall = time.time() - t
        toks = sum(n for _, n in rs)
        return {"batch": batch, "per_seq_tok_s": round(sum(n / d for d, n in rs) / len(rs), 1), "aggregate_tok_s": round(toks / wall, 1), "wall_s": round(wall, 1)}

    res["decode"] = [asyncio.run(bench(b, a["decode_tokens"])) for b in a["batches"]]
    # ---- checkpoint publish + engine load (the CheckpointPublisher path) -------------------------------------------
    from crucible.train.hf import HFPolicy

    pol = HFPolicy.from_pretrained(a["model"], dtype=torch.bfloat16, device="cpu")
    from safetensors.torch import save_file

    t1 = time.time()
    ck = out_dir / "ckpt"
    ck.mkdir(exist_ok=True)
    save_file({k: v.detach().clone().contiguous() for k, v in pol.state_dict().items()}, str(ck / "model.safetensors"))  # tied embeddings share storage
    res["checkpoint_write_s"] = round(time.time() - t1, 2)
    res["checkpoint_gb"] = round(sum(f.stat().st_size for f in ck.iterdir()) / 1e9, 2)
    srv.terminate()
    srv.wait(timeout=120)
    del pol
    # ---- FSDP trainer throughput + NCCL broadcast on GPUs 1..N-1 ----------------------------------------------------
    world = gpus - 1
    if world >= 1:
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        procs = []
        for r in range(world):
            e = {**os.environ, "CUDA_VISIBLE_DEVICES": str(r + 1), "RANK": str(r), "WORLD_SIZE": str(world), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                 "CAL_ARGS": json.dumps(a)}
            procs.append(subprocess.Popen([sys.executable, "-c", TRAIN_WORKER], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        errs = []
        for p in procs:
            _, err = p.communicate(timeout=3600)
            if p.returncode:
                errs.append(err[-3000:])
        if errs:
            res["train_error"] = errs
        else:
            ranks = [json.loads((out_dir / f"train_r{r}.json").read_text()) for r in range(world)]
            r0 = ranks[0]
            n = r0["params"]
            peak = a["peak_tflops"] * 1e12
            res["train"] = {**r0, "mfu": round(6 * n * r0["train_tok_s"] / (world * peak), 4), "peak_mem_gb_per_rank": [x["peak_mem_gb"] for x in ranks]}
    return res


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-1.7B", peak_tflops: float = 125.0, prompt: int = 512, completion: int = 512, seqs_per_rank: int = 8, train_steps: int = 4,
         decode_tokens: int = 256, batches: str = "1,8,32,128", token_budget: int = 8192):
    a = {"model": model, "peak_tflops": peak_tflops, "prompt": prompt, "completion": completion, "seqs_per_rank": seqs_per_rank, "train_steps": train_steps,
         "decode_tokens": decode_tokens, "batches": [int(x) for x in batches.split(",")], "token_budget": token_budget}
    res = calibrate.remote(a)
    print(json.dumps(res, indent=1, default=str)[:5000])
    dest = ROOT / "docs" / "research" / "m12-infra" / "02-live-validation.json"
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    prev.setdefault("calibration", []).append({"gpu_spec": os.environ.get("CRUCIBLE_CAL_GPU", "A10G:4"), "args": a, **res})
    dest.write_text(json.dumps(prev, indent=1, default=str))
