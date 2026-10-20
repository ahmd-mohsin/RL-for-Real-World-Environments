"""GPU in an Apptainer container, as a kernel-migration World needs it (smoke1 stage ``gpu_container``).

A ``--nv`` instance of the CUDA 12.8 image (``defs/cuda128.def``, the base of migration_tasks/task04) with no network,
through ``ApptainerSandbox``: nvidia-smi sees the job's GPU, nvcc compiles a vector-add kernel into the bound workspace,
the kernel runs and its result is checked; the same in a one-shot container (``run_once``, a grader's fresh environment).
Receipt: ``<out>/gpu_container.receipt.json``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

KERNEL = r"""
#include <cstdio>
#include <cuda_runtime.h>
__global__ void add(const float* a, const float* b, float* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }
int main() {
  const int n = 1 << 20; size_t bytes = n * sizeof(float);
  float *a, *b, *c;
  if (cudaMallocManaged(&a, bytes) != cudaSuccess || cudaMallocManaged(&b, bytes) != cudaSuccess || cudaMallocManaged(&c, bytes) != cudaSuccess) {
    printf("CUDA_ERROR no device memory (no GPU visible?)\n"); return 3; }
  for (int i = 0; i < n; i++) { a[i] = 1.0f * i; b[i] = 2.0f; }
  add<<<(n + 255) / 256, 256>>>(a, b, c, n);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA_ERROR %s\n", cudaGetErrorString(e)); return 2; }
  double err = 0; for (int i = 0; i < n; i++) err += (c[i] - (1.0f * i + 2.0f)) * (c[i] - (1.0f * i + 2.0f));
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("DEVICE %s sm_%d%d ERR %g\n", p.name, p.major, p.minor, err);
  return err == 0 ? 0 : 1;
}
"""


async def main() -> int:
    from crucible.world.backends.apptainer import ApptainerSandbox, run_once

    out = Path(sys.argv[1]).resolve()
    image = os.path.join(os.environ["CRU_SIF"], "cuda128.sif")
    ws = out / "gpu_ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "vadd.cu").write_text(KERNEL)
    r: dict = {"image": image, "image_sha256": Path(image).with_suffix(".sha256").read_text().strip() if Path(image).with_suffix(".sha256").exists() else None,
               "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID")}
    sb = ApptainerSandbox(image, gpus=1, binds=[(str(ws), str(ws), "rw")], label="gpu-probe", configdir=os.environ.get("APPTAINER_CONFIGDIR"),
                          limits={"pids": 4096})
    t0 = time.time()
    await sb.start()
    try:
        r["sandbox"] = sb.description
        r["isolation"] = await sb.isolation_report(must_hide=[str(ROOT)])
        smi = await sb.exec.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -2", timeout_s=120)
        r["nvidia_smi_in_container"] = (smi.stdout + smi.stderr).strip().splitlines()
        cc = await sb.exec.run("nvcc -O2 -arch=sm_90 -o vadd vadd.cu", cwd=str(ws), timeout_s=600)
        r["nvcc"] = {"rc": cc.rc, "tail": (cc.stdout + cc.stderr)[-400:]}
        run = await sb.exec.run("./vadd", cwd=str(ws), timeout_s=120)
        r["kernel"] = {"rc": run.rc, "out": (run.stdout + run.stderr).strip()[-400:]}
    finally:
        await sb.close()
    once = await run_once(image, "./vadd", binds=[(str(ws), "/w", "ro")], cwd="/w", gpus=1, timeout_s=120)
    r["kernel_fresh_container"] = {"rc": once.rc, "out": (once.stdout + once.stderr).strip()[-300:]}
    r["seconds"] = round(time.time() - t0, 1)
    r["pass"] = bool(r["kernel"]["rc"] == 0 and once.rc == 0 and r["isolation"].get("ok"))
    (out / "gpu_container.receipt.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
