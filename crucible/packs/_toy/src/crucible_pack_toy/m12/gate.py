"""M12 exit gate, GPU part (ARCHITECTURE M12: "one colocated async GRPO run on the toy pack on the lab RTX 5090, with
staleness and rollout_probs_diff monitors recorded").

Inside the vLLM image on the lab: a vLLM 0.31 server (fixed memory fraction, sleep mode, Crucible worker extension) and
Crucible's M12 runtime in one container. Policy: Qwen3-0.6B (HF weights for the trainer, fp32 master + bf16 autocast,
FP32 LM head). Environment: small long-horizon backlog Worlds (``longhorizon`` BacklogTask, 4 tickets; reward = share of
tickets correct, so groups have variance). Topology ``colocated_partial``: at every sample barrier the engine's calls
are aborted and the server sleeps; the trainer steps; the checkpoint publisher writes bf16 safetensors and the engine
loads them through ``/collective_rpc``; generation resumes on the new version. Staleness, spans, rollout_probs_diff,
masked/clipped token shares, aborts and publish latency are recorded per step.

    python -m crucible_pack_toy.m12.gate --steps 20 --out /work/out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

TEMPFAIL = 75  # EX_TEMPFAIL: the shared GPU was taken between the idle check and vLLM's memory profile; the runner re-waits
GPU_BUSY = ("free memory on device", "less than desired GPU memory", "CUDA out of memory", "out of memory", "cudaErrorMemoryAllocation")
LOG = Path(os.environ.get("CRUCIBLE_VLLM_LOG", "/work/out/vllm.log"))


def start_server(model: str, port: int, mem: float, ext_dir: str, max_len: int) -> subprocess.Popen:
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1", "PYTHONPATH": f"{ext_dir}:{os.environ.get('PYTHONPATH', '')}"}
    cmd = ["vllm", "serve", model, "--port", str(port), "--gpu-memory-utilization", str(mem), "--enable-sleep-mode", "--max-model-len", str(max_len),
           "--worker-extension-cls", "crucible_vllm_ext.CrucibleWorkerExtension", "--logprobs-mode", "processed_logprobs", "--dtype", "bfloat16"]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)  # full log kept in the out dir


def wait_health(base: str, proc: subprocess.Popen, timeout: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log = LOG.read_text(errors="replace")
            if any(k in log for k in GPU_BUSY):
                print(f"vLLM could not get its memory (shared GPU busy): exit {TEMPFAIL} so the runner waits again", flush=True)
                sys.exit(TEMPFAIL)
            raise RuntimeError("vLLM exited (full log: vllm.log): " + log[-3000:])
        try:
            if httpx.get(base + "/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy")


async def run(args) -> dict:
    import torch
    from transformers import AutoTokenizer

    from crucible.actors.policy import ContextConfig
    from crucible.train.chat import HFChatCodec
    from crucible.train.engine import EnginePool
    from crucible.train.hf import HFPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.vllm_http import VLLMServerEngine
    from crucible.train.weights import CheckpointPublisher
    from crucible.train.world_env import world_rollout
    from crucible_pack_toy.longhorizon.task import BacklogTask

    base = f"http://127.0.0.1:{args.port}"
    tok = AutoTokenizer.from_pretrained(args.model)
    codec = HFChatCodec(tok, template_kwargs={"enable_thinking": False})
    eng = VLLMServerEngine(base, args.model, capacity=128, placement="colocated")
    pool = EnginePool([eng])
    pol = HFPolicy.from_pretrained(args.model, dtype=torch.float32, device="cuda", lm_head_fp32=True, gradient_checkpointing=True,
                                   autocast_dtype=torch.bfloat16)
    tr = TorchTrainer(pol, TrainerConfig(lr=args.lr, correction=Correction("dis"), token_budget=args.token_budget))
    pub = CheckpointPublisher(tr, pool, Path(args.out) / "ckpt", post_update_transform=lambda sd: {k: v.to(torch.bfloat16) for k, v in sd.items()})
    tasks: dict[str, BacklogTask] = {}

    def task_for(prompt: str) -> BacklogTask:
        if prompt not in tasks:
            tasks[prompt] = BacklogTask(seed=int(prompt[1:]), n=args.tickets)
        return tasks[prompt]

    rt = Runtime(RuntimeConfig(topology=args.topology, steps=args.steps, max_turn_tokens=args.max_tokens, temperature=1.0, stop_ids=tuple(codec.eos_ids),
                               warmup_batches=0),
                 SchedulerConfig(n=args.n, B=args.B, max_staleness=args.tau), pool, tr, pub, None, {"backlog": [f"s{i}" for i in range(100000)]},
                 rollout_fn=world_rollout(task_for, codec, max_steps=args.max_steps, max_tokens=args.max_tokens,
                                          context=ContextConfig(budget_tokens=args.context, keep_recent_turns=2)), harness="native:world-qwen@1")
    t0 = time.time()
    rep = await rt.run()
    out = {"gate": "M12 GPU: colocated async GRPO on toy Worlds, lab RTX 5090", "model": args.model, "topology": args.topology, "steps": len(rep.steps),
           "wall_s": round(time.time() - t0, 1), "aborted_calls": rep.aborted_calls, "reprefill_tokens": rep.reprefill_tokens, "publishes": rep.publishes,
           "masked_rollouts": rep.masked_rollouts, "max_untrained": rep.max_untrained, "per_step": [{"step": s.step, "version": s.version, **s.metrics} for s in rep.steps],
           "gpu": torch.cuda.get_device_name(0), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "stop_reason": rep.stop_reason, "failures": rep.failures, "failure_examples": rep.failure_examples[:4]}
    r = [s.metrics.get("reward/mean", 0.0) for s in rep.steps]
    out["reward_first3"], out["reward_last3"] = (sum(r[:3]) / max(len(r[:3]), 1)), (sum(r[-3:]) / max(len(r[-3:]), 1))
    out["pass"] = bool(rep.stop_reason == "steps" and len(rep.steps) == args.steps and rep.publishes == args.steps and all("staleness/max" in s.metrics and "mismatch/rollout_probs_diff" in s.metrics
                                                                                           for s in rep.steps) and max(s.metrics["staleness/max"] for s in rep.steps) <= args.tau)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--topology", default="colocated_partial")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--tickets", type=int, default=4)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=24)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=384)
    ap.add_argument("--context", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--token-budget", dest="token_budget", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--mem", type=float, default=0.35)
    ap.add_argument("--max-len", dest="max_len", type=int, default=8192)
    ap.add_argument("--out", default="/work/out")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ext_dir = str(Path(__file__).resolve().parents[0] / "_ext")
    proc = start_server(a.model, a.port, a.mem, ext_dir, a.max_len)
    try:
        wait_health(f"http://127.0.0.1:{a.port}", proc)
        rep = asyncio.run(run(a))
        (Path(a.out) / "m12_gpu_gate.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
        print(json.dumps({k: v for k, v in rep.items() if k != "per_step"}, indent=1))
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(0)


if __name__ == "__main__":
    main()
