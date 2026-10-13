"""Multi-node GPU checks under Slurm (REPACSS: 2 nodes × 4 H100 NVL), one task per GPU (``srun --ntasks-per-node=4``).
The stages of ``scripts/repacss_train.sbatch`` (repository root) call this module; each task reads its rank from the
Slurm environment (``train.cluster.slurm_env``), initializes NCCL with an explicit timeout (``cluster.init_rank``) and
writes ``<out>/<stage>-r<rank>.json``; rank 0 also writes ``<out>/<stage>.json`` with the verdict.

    srun ... python scripts/slurm_check.py nccl --out $OUT                      # stage 0: rendezvous, topology, bandwidth
    srun ... python scripts/slurm_check.py unit fsdp --out $OUT                 # stage 1: a tests/unit worker scenario on NCCL
    srun ... python scripts/slurm_check.py weightsync --model Qwen/Qwen3-8B     # stage 2: bucketed broadcast + checkpoint write
    srun ... python scripts/slurm_check.py fsdp --model Qwen/Qwen3-1.7B         # stage 3: FSDP2 parity and peak memory
    python scripts/slurm_check.py vllm --model Qwen/Qwen3-1.7B --tp 2           # stage 4: one node, sleep / wake / load cycles

On a laptop, ``CRUCIBLE_CPU_ONLY=1`` with fake ``SLURM_*`` variables runs ``nccl``, ``unit`` and ``fsdp --model tiny`` on
gloo (that is how they were checked before any GPU time was spent).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _setup() -> tuple[int, int, object]:
    from crucible.train.cluster import init_rank, slurm_env

    if "SLURM_PROCID" in os.environ:
        os.environ.update(slurm_env())
    return init_rank()


def _write(out: Path, stage: str, rank: int, res: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stage}-r{rank}.json").write_text(json.dumps(res, indent=1, default=str))


def _gather(obj: object) -> list:
    import torch.distributed as dist

    box: list = [None] * dist.get_world_size()
    dist.all_gather_object(box, obj)
    return box


# ---- stage 0 --------------------------------------------------------------------------------------------------------------
def stage_nccl(a: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    rank, world, dev = _setup()
    local, nodes = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("NNODES", "1"))
    per_node = world // max(1, nodes)
    res: dict = {"rank": rank, "host": socket.gethostname(), "device": str(dev), "local_rank": local,
                 "env": {k: os.environ.get(k) for k in ("NCCL_CUMEM_ENABLE", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_HCA", "CUDA_VISIBLE_DEVICES")}}
    t = torch.tensor([float(rank)], device=dev)
    dist.all_reduce(t)
    res["rank_sum"] = float(t.item())
    if dev.type == "cuda" and local == 0:
        res["topo"] = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=False).stdout
        res["gpu"] = torch.cuda.get_device_name(dev)
        res["host_mem_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        res["cpus"] = os.cpu_count()
    groups = {"world": list(range(world))}
    if per_node >= 2:
        groups["pair"] = [0, 1]  # NVLink-bridged pair on H100 NVL (check the topo matrix: NV12 vs PIX/SYS)
        groups["node0"] = list(range(per_node))
    if per_node >= 4:
        groups["cross_pair"] = [0, 2]  # PCIe between the bridged pairs
    if nodes >= 2:
        groups["inter_node"] = [0, per_node]
    pgs = {k: dist.new_group(v) for k, v in groups.items()}  # every rank creates every group, in order
    sizes = [int(x) for x in a.sizes_mb.split(",")]
    bw: dict = {}
    for name, ranks in groups.items():
        if rank not in ranks:
            continue
        rows = []
        for mb in sizes:
            n = mb * 2**20 // 4
            x = torch.ones(n, dtype=torch.float32, device=dev)
            for _ in range(2):
                dist.all_reduce(x, group=pgs[name])
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(a.iters):
                dist.all_reduce(x, group=pgs[name])
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            dt = (time.perf_counter() - t0) / a.iters
            k = len(ranks)
            rows.append({"mb": mb, "ms": round(dt * 1e3, 3), "busbw_gbs": round(n * 4 * 2 * (k - 1) / k / dt / 1e9, 2)})
        bw[name] = rows
    res["busbw"] = bw
    allr = _gather(res)
    _write(Path(a.out), "stage0", rank, res)
    if rank == 0:
        verdict = {"world": world, "rank_sum_ok": all(r["rank_sum"] == world * (world - 1) / 2 for r in allr),
                   "hosts": sorted({r["host"] for r in allr}), "busbw_largest": {k: v[-1]["busbw_gbs"] for k, v in bw.items()},
                   "nodes": [{k: r.get(k) for k in ("host", "gpu", "host_mem_gb", "cpus", "topo")} for r in allr if r.get("topo")]}
        verdict["pass"] = verdict["rank_sum_ok"] and len(verdict["hosts"]) == nodes
        (Path(a.out) / "stage0.json").write_text(json.dumps(verdict, indent=1))
        print(json.dumps({k: v for k, v in verdict.items() if k != "nodes"}, indent=1))
    dist.destroy_process_group()


# ---- stage 1 --------------------------------------------------------------------------------------------------------------
def stage_unit(a: argparse.Namespace) -> None:
    """Run one ``tests/unit`` worker scenario with the Slurm ranks over NCCL (env:// rendezvous)."""
    from crucible.train.cluster import slurm_env

    if "SLURM_PROCID" in os.environ:
        os.environ.update(slurm_env())
    os.environ.setdefault("CRUCIBLE_DIST_BACKEND", "gloo" if os.environ.get("CRUCIBLE_CPU_ONLY") else "nccl")
    os.environ["CRUCIBLE_RENDEZVOUS"] = "env"
    out = Path(a.out) / f"stage1-{a.scenario}"
    out.mkdir(parents=True, exist_ok=True)
    worker = ROOT / "tests" / "unit" / a.worker
    sys.argv = [str(worker), a.scenario, str(out)]
    sys.path.insert(0, str(worker.parent))
    import runpy

    runpy.run_path(str(worker), run_name="__main__")


def stage_report(a: argparse.Namespace) -> None:
    """Stage 1 verdict: every rank wrote its result; every ``max_abs_diff`` on rank 0 is within ``--tol``."""
    rows = {}
    for d in sorted(Path(a.out).glob("stage1-*")):
        rs = [json.loads(f.read_text()) for f in sorted(d.glob("r*.json"))]
        diffs: list[float] = []

        def walk(x, diffs=diffs):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ("max_abs_diff", "restore_diff") and isinstance(v, int | float):
                        diffs.append(float(v))
                    else:
                        walk(v, diffs)

        walk(rs[0] if rs else {})
        rows[d.name] = {"ranks": len(rs), "max_abs_diff": max(diffs, default=None), "pass": bool(rs) and all(x <= a.tol for x in diffs)}
    verdict = {"scenarios": rows, "pass": bool(rows) and all(r["pass"] for r in rows.values())}
    (Path(a.out) / "stage1.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


# ---- stage 2 --------------------------------------------------------------------------------------------------------------
def _hf_state(model: str, dtype):
    if model == "tiny":
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)
        return LlamaForCausalLM(cfg).to(dtype)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model, dtype=dtype)


def stage_weightsync(a: argparse.Namespace) -> None:
    import hashlib

    import torch
    import torch.distributed as dist

    from crucible.train.weightsync import BroadcastWeightSync

    rank, _, dev = _setup()
    owned = None
    if rank == 0:
        owned = {k: v.detach().to(dev).contiguous() for k, v in _hf_state(a.model, torch.bfloat16).state_dict().items()}
    sync = BroadcastWeightSync(device=dev)
    sync.plan(owned)
    got: dict = {}

    def load(named):
        for n, t in named:
            got[n] = t.to("cpu", copy=True)

    rows = []
    for v in (1, 2):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        sync.update(v if rank == 0 else None, owned, receive=rank != 0, load=load if rank != 0 else None)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        rows.append(round(time.perf_counter() - t0, 3))

    def digest(d: dict) -> str:
        h = hashlib.sha256()
        for k in sorted(d):
            h.update(k.encode())
            h.update(d[k].view(torch.uint8).cpu().numpy().tobytes() if d[k].dtype == torch.bfloat16 else d[k].cpu().numpy().tobytes())
        return h.hexdigest()

    res = {"rank": rank, "seconds": rows, "digest": digest(got) if rank else digest({k: v.cpu() for k, v in owned.items()})}
    if rank == 0:  # the CheckpointPublisher path: write the same state as safetensors to node-local and shared storage
        from safetensors.torch import save_file

        cpu = {k: v.cpu() for k, v in owned.items()}
        for name, d in (("local", Path(a.local_dir)), ("shared", Path(a.out) / "stage2-ckpt")):
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            save_file(cpu, str(d / "model.safetensors"))
            os.sync()
            res[f"write_{name}_s"] = round(time.perf_counter() - t0, 2)
            res[f"write_{name}_gb"] = round((d / "model.safetensors").stat().st_size / 1e9, 2)
    allr = _gather(res)
    _write(Path(a.out), "stage2", rank, res)
    if rank == 0:
        total = sum(t.numel() * t.element_size() for t in owned.values())
        verdict = {"gb": round(total / 1e9, 2), "broadcast_s": rows, "gb_per_s": round(total / 1e9 / rows[-1], 2),
                   "bitwise": len({r["digest"] for r in allr}) == 1, **{k: v for k, v in res.items() if k.startswith("write_")}}
        verdict["pass"] = verdict["bitwise"] and rows[-1] < a.max_publish_s
        (Path(a.out) / "stage2.json").write_text(json.dumps(verdict, indent=1))
        print(json.dumps(verdict, indent=1))
    dist.destroy_process_group()


# ---- stage 3 --------------------------------------------------------------------------------------------------------------
def _records(vocab: int, n: int, plen: int, clen: int, seed: int = 0):
    import torch

    from crucible.core.schema.rollout import RolloutRecord, Turn

    g = torch.Generator().manual_seed(seed)
    recs, groups = [], []
    for i in range(n):
        prompt = torch.randint(5, vocab, (plen,), generator=g).tolist()
        comp = torch.randint(5, vocab, (clen,), generator=g).tolist()
        recs.append(RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=float(i % 2),
                                  turns=[Turn(call_id=f"r{i}:c", actor="agent", prompt_token_ids=prompt, completion_token_ids=comp,
                                              logprobs=[-3.0] * clen, policy_version=0)]))
        groups.append(f"g{i // 4}")
    return recs, groups


def stage_fsdp(a: argparse.Namespace) -> None:
    """Meta-device build + rank-0 load (no full fp32 copy per GPU), then for each mesh two updates on one batch; rank 0
    compares the gathered weights with a single-GPU ``TorchTrainer`` run on the same batch, and reports peak memory."""
    import gc

    import torch
    import torch.distributed as dist

    from crucible.train.fsdp import FSDPTrainer, meta_params, transformer_blocks
    from crucible.train.hf import HFPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    rank, world, dev = _setup()
    full = _hf_state(a.model, torch.float32) if rank == 0 else None
    cfg_hf = full.config if full is not None else None
    cfg_hf = _gather(cfg_hf)[0]
    vocab = cfg_hf.vocab_size
    recs, groups = _records(vocab, a.items, a.prompt, a.completion)
    if rank == 0:  # behaviour log-probs from the initial weights (a real ratio signal; random ones would be masked)
        pol = HFPolicy(full.to(dev), lm_head_fp32=True)
        with torch.no_grad():
            recs = [r.model_copy(update={"turns": [r.turns[0].model_copy(update={"logprobs": [
                float(x) - 0.05 for x in pol.token_logprobs(r.turns[0].prompt_token_ids, r.turns[0].completion_token_ids).float().cpu()]})]})
                for r in recs]
        full.to("cpu")
        del pol
    hcfg = {"lr": a.lr, "correction": Correction("dis"), "token_budget": a.token_budget}
    amp = torch.bfloat16 if dev.type == "cuda" else None
    res: dict = {"rank": rank, "world": world}
    states: dict = {}
    for shape in a.meshes.split(","):
        rep, shard = (int(x) for x in shape.split("x"))
        from transformers import AutoModelForCausalLM

        with meta_params():
            empty = HFPolicy(AutoModelForCausalLM.from_config(cfg_hf, dtype=torch.float32), lm_head_fp32=True, gradient_checkpointing=True, autocast_dtype=amp)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        tr = FSDPTrainer(empty, TrainerConfig(**hcfg, dp=world), mesh_shape=(rep, shard), reshard="layer", units=transformer_blocks, device=dev,
                         broadcast_batch=True, full_state=full.state_dict() if full is not None else None,
                         mixed_precision="bf16" if a.bf16_gather else None)
        load_s = time.perf_counter() - t0
        load_peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None
        t0 = time.perf_counter()
        ms = [tr.step(*((recs, groups) if rank == 0 else ([], []))) for _ in range(2)]
        step_s = (time.perf_counter() - t0) / 2
        peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None
        st = tr.state_dict()
        res[shape] = {"load_s": round(load_s, 2), "load_peak_gb": load_peak and round(load_peak, 2), "step_s": round(step_s, 2),
                      "peak_gb": peak and round(peak, 2), "grad_norms": [m["train/grad_norm"] for m in ms], "local": tr.local_numel()}
        if rank == 0:
            states[shape] = st
        del tr, empty, st
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    allr = _gather(res)
    _write(Path(a.out), f"stage3-{a.tag}", rank, res)
    if rank == 0 and a.no_ref:
        verdict = {"world": world, **{k: {"peak_gb_per_rank": [r[k]["peak_gb"] for r in allr], "load_peak_gb_per_rank": [r[k]["load_peak_gb"] for r in allr],
                                          "step_s": res[k]["step_s"]} for k in states}}
        verdict["pass"] = all(max(x or 0 for x in v["peak_gb_per_rank"]) <= a.max_peak_gb for k, v in verdict.items() if isinstance(v, dict))
        (Path(a.out) / f"stage3-{a.tag}.json").write_text(json.dumps(verdict, indent=1, default=str))
        print(json.dumps(verdict, indent=1, default=str))
    elif rank == 0:
        ref_model = HFPolicy(full.to(dev), lm_head_fp32=True, gradient_checkpointing=True, autocast_dtype=amp)
        ref = TorchTrainer(ref_model, TrainerConfig(**hcfg, dp=1))
        rms = [ref.step(recs, groups) for _ in range(2)]
        want = {k: v.detach().float().cpu() for k, v in ref_model.state_dict().items()}
        verdict: dict = {"world": world, "ref_grad_norms": [m["train/grad_norm"] for m in rms]}
        for shape, st in states.items():
            rel = max(float((st[k].float() - want[k]).norm() / (want[k].norm() + 1e-12)) for k in want)
            gn = [abs(x - y) / max(abs(y), 1e-12) for x, y in zip(res[shape]["grad_norms"], verdict["ref_grad_norms"], strict=True)]
            verdict[shape] = {"max_rel_weight_delta": rel, "grad_norm_rel": max(gn), "peak_gb_per_rank": [r[shape]["peak_gb"] for r in allr],
                              "load_peak_gb_per_rank": [r[shape]["load_peak_gb"] for r in allr], "step_s": res[shape]["step_s"]}
        verdict["pass"] = all(v["max_rel_weight_delta"] <= a.max_rel and v["grad_norm_rel"] <= 1e-3 for k, v in verdict.items() if isinstance(v, dict))
        (Path(a.out) / f"stage3-{a.tag}.json").write_text(json.dumps(verdict, indent=1, default=str))
        print(json.dumps(verdict, indent=1, default=str))
    dist.destroy_process_group()


# ---- stage 4 --------------------------------------------------------------------------------------------------------------
def stage_vllm(a: argparse.Namespace) -> None:
    """One node: ``vllm serve`` with sleep mode at TP ``--tp`` on the first ``tp`` GPUs (a bridged pair for TP2), then
    ``--cycles`` × (sleep → load a checkpoint → wake): GPU memory after each wake must stay within 1 GB of the first."""
    import asyncio

    import httpx
    import torch

    from crucible.train.vllm_http import VLLMServerEngine

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ext = ROOT / "packs" / "_toy" / "src" / "crucible_pack_toy" / "m12" / "_ext"
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(a.tp)),
           "PYTHONPATH": f"{ext}:{os.environ.get('PYTHONPATH', '')}"}
    log = open(out / f"stage4-vllm-tp{a.tp}.log", "w")  # noqa: SIM115 -- the server owns it for its life
    srv = subprocess.Popen(["vllm", "serve", a.model, "--port", str(a.port), "--tensor-parallel-size", str(a.tp), "--enable-sleep-mode",
                            "--gpu-memory-utilization", str(a.mem), "--max-model-len", "4096", "--dtype", "bfloat16", "--logprobs-mode", "processed_logprobs",
                            "--worker-extension-cls", "crucible_vllm_ext.CrucibleWorkerExtension"], env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    t0 = time.time()
    while True:
        try:
            if httpx.get(base + "/health", timeout=5).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if srv.poll() is not None or time.time() - t0 > 1200:
            raise SystemExit("vllm did not start: " + (out / f"stage4-vllm-tp{a.tp}.log").read_text()[-3000:])
        time.sleep(3)

    def used_gb() -> list[float]:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", ",".join(str(i) for i in range(a.tp))],
                           capture_output=True, text=True, check=False).stdout
        return [round(float(x) / 1024, 2) for x in q.split()]

    ck = out / "stage4-ckpt"
    ck.mkdir(exist_ok=True)
    from safetensors.torch import save_file

    save_file({k: v.detach().clone().contiguous() for k, v in _hf_state(a.model, torch.bfloat16).state_dict().items()}, str(ck / "model.safetensors"))

    async def cycles() -> dict:
        eng = VLLMServerEngine(base, a.model, placement="colocated")
        mem = []
        for v in range(1, a.cycles + 1):
            await eng.pause()
            asleep = used_gb()
            await eng.load_weights(str(ck), v)
            await eng.resume()
            mem.append({"asleep": asleep, "awake": used_gb()})
        return {"memory": mem}

    res = asyncio.run(cycles())
    first = sum(res["memory"][0]["awake"])
    res["max_awake_drift_gb"] = max(abs(sum(m["awake"]) - first) for m in res["memory"])
    res["pass"] = res["max_awake_drift_gb"] <= 1.0
    srv.terminate()
    srv.wait(timeout=120)
    (out / f"stage4-tp{a.tp}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "memory"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p0 = sub.add_parser("nccl")
    p0.add_argument("--sizes-mb", dest="sizes_mb", default="1,16,256,1024")
    p0.add_argument("--iters", type=int, default=10)
    p1 = sub.add_parser("unit")
    p1.add_argument("scenario")
    p1.add_argument("--worker", default="_fsdp_worker.py")
    pr = sub.add_parser("report")
    pr.add_argument("--tol", type=float, default=5e-6)
    p2 = sub.add_parser("weightsync")
    p2.add_argument("--model", default="Qwen/Qwen3-8B")
    p2.add_argument("--local-dir", dest="local_dir", default=os.environ.get("TMPDIR", "/tmp") + "/crucible-stage2")
    p2.add_argument("--max-publish-s", dest="max_publish_s", type=float, default=10.0)
    p3 = sub.add_parser("fsdp")
    p3.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p3.add_argument("--meshes", default="1x8,2x4")
    p3.add_argument("--items", type=int, default=16)
    p3.add_argument("--prompt", type=int, default=512)
    p3.add_argument("--completion", type=int, default=512)
    p3.add_argument("--token-budget", dest="token_budget", type=int, default=8192)
    p3.add_argument("--lr", type=float, default=1e-6)
    p3.add_argument("--max-rel", dest="max_rel", type=float, default=1e-4)
    p3.add_argument("--bf16-gather", dest="bf16_gather", action="store_true")
    p3.add_argument("--no-ref", dest="no_ref", action="store_true", help="no single-GPU reference (a model whose AdamW state exceeds one GPU)")
    p3.add_argument("--max-peak-gb", dest="max_peak_gb", type=float, default=85.0)
    p3.add_argument("--tag", default="parity")
    p4 = sub.add_parser("vllm")
    p4.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p4.add_argument("--tp", type=int, default=1)
    p4.add_argument("--cycles", type=int, default=50)
    p4.add_argument("--mem", type=float, default=0.6)
    p4.add_argument("--port", type=int, default=8200)
    for p in (p0, p1, pr, p2, p3, p4):
        p.add_argument("--out", default=os.environ.get("OUT", "out"))
    a = ap.parse_args()
    {"nccl": stage_nccl, "unit": stage_unit, "report": stage_report, "weightsync": stage_weightsync, "fsdp": stage_fsdp, "vllm": stage_vllm}[a.stage](a)


if __name__ == "__main__":
    main()
