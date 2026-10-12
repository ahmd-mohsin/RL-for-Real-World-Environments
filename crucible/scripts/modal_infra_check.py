"""M12-infra on real multi-GPU NCCL (Modal): the gloo multi-process scenarios, re-run on ``cuda:<rank>`` over NCCL.

    modal run scripts/modal_infra_check.py                       # default: L4:4, every scenario
    modal run scripts/modal_infra_check.py --gpu L4:2 --only fsdp,dp_exact
    modal run scripts/modal_infra_check.py --stress 20           # + a 20-round weight-sync / FSDP soak

Each scenario launches WORLD_SIZE worker processes in one container (one per GPU, file-store rendezvous), exactly as
``tests/unit/test_fsdp.py`` and ``test_weightsync.py`` do on gloo, and applies the same assertions. Results print as
JSON and are saved to ``docs/research/m12-infra/02-live-validation.json`` by the local entrypoint.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.1", "numpy>=1.26", "safetensors>=0.4", "transformers>=4.56,<5", "pydantic>=2.12", "pyyaml>=6.0",
                         "tomli-w>=1.0", "httpx>=0.27", "rich>=13", "click>=8.1", "pytest>=8")
         .env({"PYTHONPATH": "/root/crucible/src", "NCCL_DEBUG": "WARN", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"), "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "0")})
         .add_local_dir(ROOT / "src", "/root/crucible/src", ignore=["**/__pycache__"])
         .add_local_dir(ROOT / "tests" / "unit", "/root/crucible/tests/unit", ignore=["**/__pycache__", "**/*.pyc"]))
app = modal.App("crucible-infra-check")

# scenario -> (worker file, world size, check)
SCENARIOS = {
    "broadcast": ("_weightsync_worker.py", 3), "sharded": ("_weightsync_worker.py", 3), "fail": ("_weightsync_worker.py", 3),
    "subgroup": ("_weightsync_worker.py", 3), "publisher": ("_weightsync_worker.py", 2), "slow_reload": ("_weightsync_worker.py", 3),
    "dp_exact": ("_weightsync_worker.py", 2), "dp_broadcast": ("_weightsync_worker.py", 2),
    "fsdp": ("_fsdp_worker.py", 2), "hsdp": ("_fsdp_worker.py", 4), "hf_layer": ("_fsdp_worker.py", 2), "spmd_run": ("_fsdp_worker.py", 4),
    "fsdp_debug": ("_fsdp_worker.py", 2), "fsdp_probe": ("_fsdp_probe.py", 2),
}


def _launch(worker: str, scenario: str, world: int, out: Path, timeout: float = 600) -> list[dict]:
    import subprocess
    import time

    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "WORLD_SIZE": str(world), "CRUCIBLE_DIST_BACKEND": "nccl"}
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, f"/root/crucible/tests/unit/{worker}", scenario, str(out)], env={**env, "RANK": str(r)},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for r in range(world)]
    errs = []
    for p in procs:
        try:
            _, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
            err = "TIMEOUT\n" + err
        if p.returncode:
            errs.append(err[-2500:])
    if errs:
        return [{"error": e} for e in errs]
    rs = [json.loads((out / f"r{r}.json").read_text()) for r in range(world)]
    rs[0]["_wall_s"] = round(time.time() - t0, 2)
    return rs


def _check(scenario: str, rs: list[dict]) -> list[str]:
    """The gloo tests' assertions, as failures (empty = pass)."""
    bad = []

    def need(cond, msg):
        if not cond:
            bad.append(msg)

    if any("error" in r for r in rs):
        return [r["error"] for r in rs if "error" in r]
    need(all(r.get("backend") == "nccl" for r in rs), "not on nccl")
    if scenario == "broadcast":
        need(rs[0]["buckets"] >= 3 and rs[0]["aligned"], "fewer than 3 buckets or misaligned")
        need(all(r["v1"] == 1 and r["v2"] == 2 and r["d1"] == rs[0]["d1"] and r["d2"] == rs[0]["d2"] for r in rs[1:]), "broadcast not bitwise")
        need(rs[0]["d1"] != rs[0]["d2"], "versions identical")
    elif scenario == "sharded":
        need(rs[0]["owners"] == [0, 1] and rs[0]["single_owner_buckets"], "plan owners")
        need(rs[0]["d"] == rs[1]["d"] == rs[2]["d"] == rs[0]["full"], "sharded mismatch")
    elif scenario == "fail":
        need(all(r["raised"] and r["raised"].startswith("rank 2 failed to reload bucket 2 of version 2") for r in rs), "fail not all-rank")
        need(all(r["version_after"] == 1 and r["unchanged"] for r in rs[1:]), "a rank committed after a failure")
    elif scenario == "subgroup":
        need(all(r["same_crc"] for r in rs) and rs[1]["got"] is None and rs[2]["got"] == 1 and rs[2]["d"] == rs[0]["full"], "subgroup mismatch")
    elif scenario == "publisher":
        need(rs[0]["versions"] == [1, 2] == rs[1]["versions"] and rs[1]["engine_version"] == 2 and rs[1]["d"] == rs[0]["d"], "publisher mismatch")
    elif scenario == "slow_reload":
        need(rs[1]["d"] == rs[0]["d"] == rs[2]["d"], "slow reload corrupted a half")
    elif scenario in ("dp_exact", "dp_broadcast"):
        for corr in ("dis", "ppo"):
            a, b = rs[0][corr], rs[1][corr]
            need(a["d"] == b["d"] and a["changed"], f"{corr}: replicas differ")
            need(a["max_abs_diff_vs_full_batch"] < 1e-5, f"{corr}: {a['max_abs_diff_vs_full_batch']} vs full batch")
    elif scenario == "fsdp":
        for key in ("step/dis", "step/ppo", "layer/dis", "layer/ppo", "layer/ppo/skew"):
            a = rs[0][key]
            need(a["max_abs_diff"] < 5e-5, f"{key}: diff {a['max_abs_diff']}")
            need(abs(a["grad_norm"] - a["ref_grad_norm"]) < 1e-4 and a["grad_norm"] > 0, f"{key}: grad norm {a['grad_norm']} vs {a['ref_grad_norm']}")
            need(abs(a["params_frac"] - 0.5) < 0.02, f"{key}: params_frac {a['params_frac']}")
        need(rs[0]["layer/ppo/skew"]["probe_micro_batches"] + rs[1]["layer/ppo/skew"]["probe_micro_batches"] > 0, "no padding ran")
    elif scenario == "hsdp":
        for key in ("step/dis", "step/ppo"):
            need(rs[0][key]["max_abs_diff"] < 5e-5, f"{key}: diff {rs[0][key]['max_abs_diff']}")
    elif scenario == "hf_layer":
        need(rs[0]["units"] == 3 and rs[0]["max_abs_diff"] < 5e-5, f"hf diff {rs[0].get('max_abs_diff')}")
    elif scenario == "spmd_run":
        d = rs[0]
        need(d["steps"] == 16 and d["last"] > d["first"] and d["max_staleness"] <= 2, f"spmd run: {d.get('first')}→{d.get('last')}")
        need(d["restore_diff"] == 0.0 and d["resumed_at"] == 8 and d["steps2"] == 4, "spmd resume")
        need(all(r["first"]["step"] == 16 for r in rs[1:]), "followers missed steps")
    return bad


@app.function(image=image, gpu=os.environ.get("CRUCIBLE_INFRA_GPU", "L4:4"), timeout=3600)
def run(only: list[str], stress: int = 0) -> dict:
    import subprocess

    import torch

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "gpus": torch.cuda.device_count(), "nvidia_smi": smi.strip().splitlines()}
    results = {}
    for sc in only:
        worker, world = SCENARIOS[sc]
        if world > info["gpus"]:
            results[sc] = {"skipped": f"needs {world} GPUs"}
            continue
        rounds = max(1, stress) if sc in ("broadcast", "fsdp", "dp_exact") and stress else 1
        fails, walls = [], []
        for i in range(rounds):
            rs = _launch(worker, sc, world, Path(f"/tmp/{sc}-{i}"))
            fails += [f"round {i}: {f}" for f in _check(sc, rs)]
            walls.append(rs[0].get("_wall_s"))
            if fails:
                break
        results[sc] = {"pass": not fails, "fails": fails[:5], "rounds": len(walls), "wall_s": walls, "sample": {k: v for k, v in rs[0].items() if k != "error"}}
        print(sc, "PASS" if not fails else f"FAIL {fails[:2]}", flush=True)
    return {"info": info, "results": results}


@app.local_entrypoint()
def main(only: str = ",".join(SCENARIOS), stress: int = 0, gpu: str = ""):
    if gpu:
        print("set CRUCIBLE_INFRA_GPU before `modal run` to change the GPU (decorator-time):", gpu)
    out = run.remote([s for s in only.split(",") if s], stress)
    print(json.dumps({"info": out["info"], "summary": {k: v.get("pass", v.get("skipped")) for k, v in out["results"].items()}}, indent=1))
    dest = ROOT / "docs" / "research" / "m12-infra" / "02-live-validation.json"
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    cur = prev.setdefault("modal_nccl", {"info": out["info"], "results": {}})
    cur["info"] = out["info"]
    for k, v in out["results"].items():  # merge per scenario: the latest run of each scenario is kept
        cur["results"][k] = {**v, "gpu": os.environ.get("CRUCIBLE_INFRA_GPU", "L4:4")}
    dest.write_text(json.dumps(prev, indent=1, default=str))
