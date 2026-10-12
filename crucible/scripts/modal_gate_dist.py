"""The disaggregated real-model gate (``crucible_pack_toy.m12.gate_dist``) on a Modal multi-GPU container.

    CRUCIBLE_GATE_GPU=L4:4 modal run scripts/modal_gate_dist.py --steps 20 --eval 24
    CRUCIBLE_GATE_GPU=L4:4 modal run scripts/modal_gate_dist.py --model Qwen/Qwen3-1.7B --train-gpus 3

The vLLM image (the same one the lab gate uses), Crucible and the toy pack from this checkout, the HF cache on a volume.
The result JSON is merged into ``docs/research/m12-infra/02-live-validation.json`` under ``modal_gate_dist``.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.from_registry("vllm/vllm-openai:latest").entrypoint([])
         .run_commands("ln -sf $(command -v python3) /usr/local/bin/python && python --version")  # Modal looks for `python`
         .add_local_dir(ROOT / "src", "/root/crucible/src", ignore=["**/__pycache__"], copy=True)
         .add_local_dir(ROOT / "packs" / "_toy", "/root/crucible/packs/_toy", ignore=["**/__pycache__", "**/.venv"], copy=True)
         .add_local_file(ROOT / "pyproject.toml", "/root/crucible/pyproject.toml", copy=True)
         .add_local_file(ROOT / "README.md", "/root/crucible/README.md", copy=True)
         .run_commands("pip install -q -e /root/crucible -e /root/crucible/packs/_toy")
         .env({"NCCL_CUMEM_ENABLE": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
hf_cache = modal.Volume.from_name("crucible-hf-cache", create_if_missing=True)
app = modal.App("crucible-gate-dist")


@app.function(image=image, gpu=os.environ.get("CRUCIBLE_GATE_GPU", "L4:4"), timeout=5400, volumes={"/root/.cache/huggingface": hf_cache})
def run(argv: list[str]) -> dict:
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "-m", "crucible_pack_toy.m12.gate_dist", *argv, "--out", "/work/out"], capture_output=True, text=True, check=False)
    print(p.stdout[-4000:])
    if p.returncode:
        print(p.stderr[-6000:])
    out = Path("/work/out/gate_dist.json")
    res = json.loads(out.read_text()) if out.exists() else {"error": p.stderr[-6000:]}
    logs = {f.name: f.read_text(errors="replace")[-3000:] for f in Path("/work/out").glob("vllm*.log")}
    return {"result": res, "returncode": p.returncode, "vllm_log_tails": logs if p.returncode else {}}


@app.local_entrypoint()
def main(args: str = "--steps 20 --eval 24"):
    out = run.remote(shlex.split(args))
    res = out["result"]
    drv = next((r for r in res.get("processes", []) if r and r.get("role") == "driver"), {})
    print(json.dumps({"returncode": out["returncode"], "placement": res.get("placement"),
                      **{k: drv.get(k) for k in ("pass", "steps", "eval", "reward_first3", "reward_last3", "max_staleness", "wall_s", "peak_mem_gb_rank0")},
                      "error": res.get("error", "")[-1500:]}, indent=1, default=str))
    if out["returncode"]:
        for k, v in out["vllm_log_tails"].items():
            print("==", k, v[-1500:])
    dest = ROOT / "docs" / "research" / "m12-infra" / "02-live-validation.json"
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    prev.setdefault("modal_gate_dist", []).append({"gpu": os.environ.get("CRUCIBLE_GATE_GPU", "L4:4"), "args": args, **out})
    dest.write_text(json.dumps(prev, indent=1, default=str))
