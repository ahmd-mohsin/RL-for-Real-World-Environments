"""Kernel debugging lane, the CPU-gate helpers that need no Docker (swarm-pipeline/10 §5): unit commands from CMake's
compile database (objects to /tmp, no dependency files, one arch for mutants), units that compile each changed file,
the banner filter, hunk splitting for partial-fix mutants, the host-fallback floor, and the GPU backend masking an
uncalibrated task."""

from __future__ import annotations

import asyncio
import importlib.util
import shlex
from pathlib import Path

from crucible.factory.lanes import kernel_debug as kd
from crucible.factory.lanes import kernel_debug_ops as ko
from crucible.factory.lanes import kernel_debug_world as kdw

CMD = ('/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DGGML_USE_CUDA -I/src/ggml/src/../include -O3 -std=c++17 '
       '"--generate-code=arch=compute_90,code=[sm_90]" "--generate-code=arch=compute_120a,code=[sm_120a]" -MD -MT ggml/x.o -MF ggml/x.o.d '
       '-x cu -c /src/ggml/src/ggml-cuda/norm.cu -o ggml/src/CMakeFiles/ggml-cuda.dir/ggml-cuda/norm.cu.o')


def test_unit_command_writes_nothing_into_the_build_dir():
    parts = shlex.split(kdw.unit_command(CMD, 3))
    assert parts[parts.index("-o") + 1] == "/tmp/crucible_unit_3.o"
    assert "-MD" not in parts and "-MF" not in parts and "-MT" not in parts
    assert sum(p.startswith("--generate-code=") for p in parts) == 2
    one = shlex.split(kdw.unit_command(CMD, 0, arch="90"))
    assert [p for p in one if p.startswith("--generate-code=")] == ["--generate-code=arch=compute_90,code=[sm_90]"]


def test_units_for_picks_the_cheapest_includer(tmp_path: Path):
    d = tmp_path / "ggml/src/ggml-cuda"
    (d / "template-instances").mkdir(parents=True)
    (d / "common.cuh").write_text("#pragma once\n")
    (d / "fattn-vec.cuh").write_text('#include "common.cuh"\n')
    (d / "fattn.cu").write_text('#include "fattn-vec.cuh"\n' + "// big\n" * 500)
    (d / "fill.cu").write_text('#include "common.cuh"\n')
    (d / "template-instances/fattn-vec-q4.cu").write_text('#include "../fattn-vec.cuh"\n')
    got = kd.units_for(tmp_path, ["ggml/src/ggml-cuda/fattn-vec.cuh", "ggml/src/ggml-cuda/common.cuh", "ggml/src/ggml-cuda/fill.cu"], ko.GGML_ORG)
    assert got == ["ggml/src/ggml-cuda/fattn.cu", "ggml/src/ggml-cuda/fill.cu"]  # instances only when nothing else includes it


def test_banner_is_stripped():
    out = "==========\n== CUDA ==\n==========\n\nCUDA Version 12.8.1\nContainer image Copyright (c) NVIDIA\nreal line\n"
    assert kdw._strip_banner(out) == "real line"


def load_script():
    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_kernel_debug.py"
    spec = importlib.util.spec_from_file_location("lh_kernel_debug", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hunks_and_the_no_support_floor():
    s = load_script()
    patch = ("diff --git a/x.cu b/x.cu\nindex 1..2 100644\n--- a/x.cu\n+++ b/x.cu\n@@ -1,1 +1,1 @@\n-a\n+b\n@@ -9,1 +9,1 @@\n-c\n+d\n"
             "diff --git a/y.cu b/y.cu\n--- a/y.cu\n+++ b/y.cu\n@@ -1 +1 @@\n-e\n+f\n")
    hs = s.hunks(patch)
    assert len(hs) == 3 and hs[1].startswith("diff --git a/x.cu") and "@@ -9,1" in hs[1] and "@@ -1,1" not in hs[1]
    src = "static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {\n    return true;\n}\n"
    out = s.supports_floor(src, "NORM")
    assert "if (op->op == GGML_OP_NORM) { return false; }" in out
    assert s.supports_floor("GGML_CALL static bool ggml_backend_cuda_supports_op(ggml_backend_t b, const ggml_tensor * op) {\n}", "ADD")
    assert s.supports_floor(src, "gelu") is None


def test_gpu_backend_masks_an_uncalibrated_task(tmp_path: Path):
    rep = asyncio.run(kdw.GpuBackend(tmp_path).grade({"id": "kd-x"}, ""))
    assert rep["infra"]["status"] == "masked" and rep["infra"]["failure_kind"] == "uncalibrated"
