"""Kernel family S1 (swarm-pipeline/06 §3): migration-commit classification, licences, and a real git history."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from crucible.factory.lanes import kernel_mine as km
from crucible.factory.lanes import kernel_plan as kp

TEXREF_A = """#include <cuda_runtime.h>
texture<float, 1> tex;  // a texture reference
__global__ void k(float* o, int n) { int i = threadIdx.x; if (i < n) o[i] = tex1Dfetch(tex, i); }
void run(float* d, float* o, int n) { cudaBindTexture(0, tex, d, n * sizeof(float)); k<<<1, 32>>>(o, n); cudaThreadSynchronize(); }
int main() { return 0; }
"""
TEXREF_B = """#include <cuda_runtime.h>
__global__ void k(cudaTextureObject_t tex, float* o, int n) { int i = threadIdx.x; if (i < n) o[i] = tex1Dfetch<float>(tex, i); }
void run(float* d, float* o, int n) {
  cudaResourceDesc r = {}; r.resType = cudaResourceTypeLinear; r.res.linear.devPtr = d; r.res.linear.sizeInBytes = n * sizeof(float);
  r.res.linear.desc = cudaCreateChannelDesc<float>(); cudaTextureDesc t = {}; cudaTextureObject_t tex = 0;
  cudaCreateTextureObject(&tex, &r, &t, nullptr); k<<<1, 32>>>(tex, o, n); cudaDeviceSynchronize(); cudaDestroyTextureObject(tex); }
int main() { return 0; }
"""
PLAIN_B = TEXREF_A.replace("texture<float, 1> tex;  // a texture reference\n", "").replace("tex1Dfetch(tex, i)", "d_in[i]").replace(
    "cudaBindTexture(0, tex, d, n * sizeof(float)); ", "")
SHFL_A = "__global__ void r(float* o) { float v = o[threadIdx.x]; v += __shfl_down(v, 1); o[threadIdx.x] = v; }\n"
SHFL_B = "__global__ void r(float* o) { float v = o[threadIdx.x]; v += __shfl_down_sync(0xffffffff, v, 1); o[threadIdx.x] = v; }\n"


def _fc(path: str, before: str, after: str, status: str = "M") -> km.FileChange:
    a, b = before.splitlines(), after.splitlines()
    return km.FileChange(path, removed=[x for x in a if x not in b], added=[x for x in b if x not in a], status=status)


def test_texref_and_warp_sync_migrations_are_classified():
    kinds, mig = km.classify([_fc("src/k.cu", TEXREF_A, TEXREF_B), _fc("src/r.cuh", SHFL_A, SHFL_B)])
    assert set(kinds) >= {"texref", "warp_sync", "thread_api"} and mig == ["src/k.cu", "src/r.cuh"]


def test_dead_code_removal_and_new_code_are_not_migrations():
    kinds, mig = km.classify([_fc("a.cu", TEXREF_A, ""), _fc("b.cu", "", TEXREF_B, status="A")])
    assert "texref" not in kinds and mig == []  # emptied file: the legacy is gone but nothing replaced it; b.cu is new code
    kinds, mig = km.classify([_fc("a.cu", TEXREF_A, "", status="D")])
    assert kinds == {} and mig == []


def test_complete_removal_counts_only_with_file_texts():
    fc = _fc("k.cu", TEXREF_A, PLAIN_B)  # texture reference replaced by plain loads (rodinia's kmeans migration)
    assert "texref" not in km.classify([fc])[0]
    kinds, mig = km.classify([fc], lambda p: (TEXREF_A, PLAIN_B))
    assert "texref" in kinds and mig == ["k.cu"]


def test_keeping_the_legacy_form_under_an_ifdef_is_not_a_migration():
    both = SHFL_A + "#if CUDART_VERSION >= 9000\n" + SHFL_B + "#endif\n"
    kinds, _ = km.classify([_fc("r.cuh", SHFL_A.replace("o[threadIdx.x] = v;", "o[threadIdx.x] = v + 0;"), both)])
    assert "warp_sync" not in kinds


def test_licences():
    assert km.licence_of("MIT License\n\nPermission is hereby granted, free of charge, to any person") == "MIT"
    assert km.licence_of("Redistribution and use in source and binary forms ... Neither the name of") == "BSD-3-Clause"
    assert km.licence_of("Redistribution and use in source and binary forms, with or without") == "BSD-2-Clause"
    assert km.licence_of("Apache License\nVersion 2.0, January 2004") == "Apache-2.0"
    assert km.licence_of("GNU GENERAL PUBLIC LICENSE Version 2") is None
    assert km.licence_of("for non-commercial use only. Redistribution and use in source and binary forms") is None


def test_grader_static_patterns_match_the_miners_legacy_patterns():
    samples = {"texref": "texture<float, 2, cudaReadModeElementType> t;", "surfref": "surface<void, 2> s;", "warp_sync": "x = __shfl_xor(x, 1);",
               "thread_api": "cudaThreadSynchronize();", "lib_api": "cusparseScsrmv(h, op, m, n, nnz, &a, d, v, r, c, x, &b, y);"}
    for k, line in samples.items():
        assert km.KINDS[k][0].search(line), k
        assert re.search(kp.LEGACY[k], line), k


def _git(d: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True).stdout


def test_scan_repo_finds_the_migration_commit(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    _git(d, "init", "-q")
    (d / "LICENSE").write_text("MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n")
    (d / "k.cu").write_text(TEXREF_A)
    (d / "README").write_text("x\n")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "start")
    (d / "README").write_text("y\n")
    _git(d, "commit", "-qam", "docs")
    (d / "k.cu").write_text(TEXREF_B)
    _git(d, "commit", "-qam", "migrate to texture objects")
    rows = km.scan_repo(d, "t_r")
    assert len(rows) == 1
    c = rows[0]
    assert c.subject == "migrate to texture objects" and c.migrated_files == ["k.cu"] and c.licence == "MIT" and "texref" in c.kinds
    assert c.parent == _git(d, "rev-parse", "HEAD~1").strip()
