"""Kernel family CPU gates on real toolchains (swarm-pipeline/06 §6): a synthetic texture-reference program in the lane's
CUDA 12.8 and 11.8 images. The start fails to build for sm_90, the oracle builds, the original builds as compute_60 PTX,
the run milestone's edge to its compile milestone executes, and fault-class mutants still build (left for the GPU)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from crucible.factory.lanes import kernel as K
from crucible.factory.lanes import kernel_gates as kg
from crucible.factory.lanes import kernel_world as kw

pytestmark = [pytest.mark.docker, pytest.mark.slow]

A = """#include <cstdio>
#include <cuda_runtime.h>
texture<float, 1, cudaReadModeElementType> tex;
__global__ void readout(float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = tex1D(tex, i * 0.37f); }
int main() {
  const int n = 64; float h[n]; for (int i = 0; i < n; ++i) h[i] = i * 0.5f;
  cudaArray* arr; cudaChannelFormatDesc d = cudaCreateChannelDesc<float>(); cudaMallocArray(&arr, &d, n);
  cudaMemcpy2DToArray(arr, 0, 0, h, n * sizeof(float), n * sizeof(float), 1, cudaMemcpyHostToDevice);
  tex.filterMode = cudaFilterModeLinear; tex.addressMode[0] = cudaAddressModeClamp; cudaBindTextureToArray(tex, arr);
  float* o; cudaMalloc(&o, n * sizeof(float)); readout<<<1, 64>>>(o, n); cudaMemcpy(h, o, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i += 8) printf("%d %.6f\\n", i, h[i]);
  return 0;
}
"""
B = """#include <cstdio>
#include <cuda_runtime.h>
__global__ void readout(cudaTextureObject_t tex, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = tex1D<float>(tex, i * 0.37f); }
int main() {
  const int n = 64; float h[n]; for (int i = 0; i < n; ++i) h[i] = i * 0.5f;
  cudaArray* arr; cudaChannelFormatDesc d = cudaCreateChannelDesc<float>(); cudaMallocArray(&arr, &d, n);
  cudaMemcpy2DToArray(arr, 0, 0, h, n * sizeof(float), n * sizeof(float), 1, cudaMemcpyHostToDevice);
  cudaResourceDesc r = {}; r.resType = cudaResourceTypeArray; r.res.array.array = arr;
  cudaTextureDesc t = {}; t.filterMode = cudaFilterModeLinear; t.addressMode[0] = cudaAddressModeClamp;
  cudaTextureObject_t tex = 0; cudaCreateTextureObject(&tex, &r, &t, nullptr);
  float* o; cudaMalloc(&o, n * sizeof(float)); readout<<<1, 64>>>(tex, o, n); cudaMemcpy(h, o, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i += 8) printf("%d %.6f\\n", i, h[i]);
  cudaDestroyTextureObject(tex); return 0;
}
"""


def _images() -> bool:
    return shutil.which("docker") is not None and bool(kw.image_id(kw.BASE_TAG)) and bool(kw.image_id(kw.LEGACY_TAG))


@pytest.mark.skipif(not _images(), reason="needs the lane's images (crucible-k-base, crucible-k-legacy)")
def test_compile_gates_on_a_texture_reference_program(tmp_path: Path):
    a = tmp_path / "a"
    (a / "tex").mkdir(parents=True)
    (a / "tex" / "readout.cu").write_text(A)
    progs = K.discover_programs(a, ["tex/readout.cu"])
    ms = K.build_milestones(["tex/readout.cu"], {"tex/readout.cu": ["tex/readout.cu"]}, progs)
    w = {"id": "k-test-tex", "repo": "t_r", "a": "a" * 40, "b": "b" * 40, "milestones": ms, "migrated_files": ["tex/readout.cu"],
         "programs": {p: {**v.spec(), "main": v.main} for p, v in progs.items()}, "kinds": {"texref": {}}, "oracle_files": {"tex/readout.cu": B}}
    g = kg.compile_gates(w, a, arches=("90",))
    assert g["needs"] == "breaks" and g["S3_needed"] and g["S4"] and g["S3_original"] and g["cpu_pass"]
    assert g["legacy"] == {"tex_readout": "60"} and w["graded_kinds"] == ["texref"]
    assert w["milestones"][1]["requires_evidence"] == {"c01": "executed"} and w["rung"] == "L1"
    built = {f["id"].split("-")[1] for f in g["fault_mutants"] if f["builds"]}
    assert {"addr_clamp_to_border", "filter_linear_to_point"} <= built  # semantic mutants: they build, the GPU must catch them


@pytest.mark.skipif(not _images(), reason="needs the lane's images")
def test_world_workspace_has_one_commit(tmp_path: Path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "k.cu").write_text(A)
    ws = kw.workspace(a, tmp_path / "ws")
    assert kw.history_probe(ws)["ok"]
    assert subprocess.run(["git", "-C", str(ws), "tag"], capture_output=True, text=True, check=True).stdout.split() == ["base"]
