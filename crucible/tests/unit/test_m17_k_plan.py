"""Kernel family grader (``lanes/kernel_plan``, swarm-pipeline/06 §5): the grader-owned build command, the static rule,
observation compares, the sanitizer rule, legacy repair, and the plan run through the kit (compile milestones on the CPU
with a stand-in compiler; GPU milestones masked when not run)."""

from __future__ import annotations

import json
import stat
import struct
from pathlib import Path

from crucible.factory.lanes import kernel_gates as kg
from crucible.factory.lanes import kernel_plan as kp
from crucible.verify.runtime.plan import Context
from crucible.verify.runtime.trusted import Trusted

PROG = {"sources": ["app/k.cu"], "includes": ["lib"], "defines": ["N=4"], "libs": ["cublas"], "std": "c++17"}


def test_candidate_builds_carry_native_code_for_the_arch_only():
    cmd = kp.nvcc_command(PROG, root=Path("/t"), out=Path("/o/p"), arch="120", observe=Path("/h/o.cpp"), nvcc="nvcc")
    assert cmd[cmd.index("-gencode") + 1] == "arch=compute_120,code=sm_120" and "-cudart" in cmd and "static" in cmd
    assert "/t/app/k.cu" in cmd and "/h/o.cpp" in cmd and any(c.startswith("--wrap=cudaMemcpy") for c in cmd) and "-lcublas" in cmd and cmd[-1] == "-lcuda"
    leg = kp.nvcc_command(PROG, root=Path("/t"), out=Path("/o/p"), arch="120", observe=Path("/h/o.cpp"), legacy=True, legacy_arch="60", nvcc="nvcc")
    assert leg[leg.index("-gencode") + 1] == "arch=compute_60,code=compute_60"  # the original: PTX only, driver JIT
    tu = kp.tu_command(PROG, "app/k.cu", root=Path("/t"), out=Path("/o/u.o"), arch="90", nvcc="nvcc")
    assert tu[-4:] == ["-c", "/t/app/k.cu", "-o", "/o/u.o"] and "arch=compute_90,code=sm_90" in tu and "/h/o.cpp" not in tu


def test_static_rule_strips_comments_and_flags_jit_escape(tmp_path: Path):
    (tmp_path / "a.cu").write_text('// texture<float, 1> t;\n/* cudaBindTexture(0, t, p, 4); */\nconst char* s = "texture<float,1> x;";\n')
    (tmp_path / "b.cu").write_text("texture<float, 1> t;\nvoid f() { cuModuleLoadData(&m, ptx); }\n")
    assert kp.static_hits(tmp_path, ["a.cu"], ["texref"]) == []
    hits = kp.static_hits(tmp_path, ["b.cu", "missing.cu"], ["texref"])
    assert {h["kind"] for h in hits} == {"texref", "jit_escape"}


def obs(stdout: str, d2h: bytes = b"", files: dict | None = None) -> dict:
    return {"stdout": stdout, "d2h": d2h, "files": files or {}}


def test_observation_compares():
    ref = obs("result 1.0000 2.5\nTime: 3.2 ms\nPASS\n", struct.pack("<4f", 1, 2, 3, 4))
    assert kp.compare_obs(obs("result 1.0000 2.5\nTime: 9.9 ms\nPASS\n", struct.pack("<4f", 1, 2, 3, 4)), ref, {}) == []  # timing is volatile
    assert kp.compare_obs(obs("result 1.0001 2.5\nPASS\n", struct.pack("<4f", 1, 2, 3, 4)), ref, {})  # numbers exact by default
    near = struct.pack("<4f", 1, 2, 3, 4.0000005)
    assert kp.compare_obs(obs(ref["stdout"], near), ref, {})[0]["d2h"] == "bytes differ"
    assert kp.compare_obs(obs(ref["stdout"], near), ref, {"d2h": "f32", "rtol": 1e-6}) == []
    assert kp.compare_obs(obs(ref["stdout"], struct.pack("<4f", 1, 2, 3, 5)), ref, {"d2h": "f32", "rtol": 1e-6})[0]["count"] == 1
    assert kp.compare_obs(obs(ref["stdout"], ref["d2h"][:8]), ref, {})[0]["d2h"] == "length"
    assert kp.compare_obs(obs(ref["stdout"], ref["d2h"], {"out.txt": b"x"}), {**ref, "files": {"out.txt": b"y"}}, {})
    assert kp.empty_obs(obs("Time: 3 ms\n")) and not kp.empty_obs(obs("", b"\0"))
    # a self-checking program (its copies back are gigabytes): its report is the observation, the copies are not read
    assert kp.compare_obs(obs(ref["stdout"], b""), ref, {"d2h": "none"}) == []
    assert kp.compare_obs(obs("result 1.0000 2.5\nMISMATCH at 7\n"), ref, {"d2h": "none"})  # its own check failing still fails
    rows = {"d2h": "none", "volatile": [kp.DEFAULT_VOLATILE, r"^[-+0-9.\s]+$"]}
    assert kp.compare_obs(obs("Unroll 1 2\n4 21.5 30.1\n"), obs("Unroll 1 2\n4 19.0 33.3\n"), rows) == []  # bandwidth rows are timings


def test_sanitizer_rule_reads_every_summary_line():
    clean = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n========= ERROR SUMMARY: 0 errors\n"
    assert kp.sanitizer_clean(0, clean)[0]
    forged = "========= ERROR SUMMARY: 0 errors\n... ========= RACECHECK SUMMARY: 3 hazards displayed (0 errors, 3 warnings)\n"
    assert not kp.sanitizer_clean(0, forged)[0]  # the task04 grader read only the first summary
    assert not kp.sanitizer_clean(97, clean)[0] and not kp.sanitizer_clean(0, "no summary")[0]


def test_legacy_repair_defines_only_newer_enum_names():
    errs = ['/t/chLib/chError_cuda.h(147): error: identifier "CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC" is undefined',
            '/t/chLib/chError_cuda.h(149): error: identifier "CUDA_ERROR_CONTAINED" is undefined']
    assert kg.legacy_repair(errs) == ["CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC=990001", "CUDA_ERROR_CONTAINED=990002"]
    assert kg.legacy_repair([*errs, "/t/k.cu(3): error: texture is not a template"]) == []


FAKE_NVCC = """#!/usr/bin/env python3
import sys, re
a = sys.argv[1:]
src = a[a.index("-c") + 1]
text = open(src).read()
if re.search(r"texture\\s*<", text):
    print(src + "(1): error: texture is not a template"); sys.exit(1)
open(a[a.index("-o") + 1], "w").write("obj")
"""


def test_plan_grades_compile_milestones_on_the_cpu(tmp_path: Path, monkeypatch):
    nvcc = tmp_path / "nvcc"
    nvcc.write_text(FAKE_NVCC)
    nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(kp, "NVCC", str(nvcc))
    spec = {"task_id": "k-t", "kinds": ["texref"], "programs": {"p": {"sources": ["k.cu"], "includes": [], "cases": {"default": []}}},
            "milestones": [{"id": "c01", "kind": "compile", "program": "p", "tu": "k.cu", "files": ["k.cu"], "programs": ["p"]},
                           {"id": "r01", "kind": "run", "program": "p", "files": ["k.cu"], "programs": ["p"]}]}
    (tmp_path / "kspec.json").write_text(json.dumps(spec))
    rows = {}
    for name, text in (("start", "texture<float, 1> t;\n"), ("migrated", "cudaTextureObject_t t;\n")):
        cand = tmp_path / name
        cand.mkdir()
        (cand / "k.cu").write_text(text)
        work = tmp_path / f"w-{name}"
        ctx = Context(Trusted(work), candidate=cand, params={"spec": str(tmp_path / "kspec.json"), "cpu_only": "1", "arch": "90"})
        rep = kp.plan(ctx).run(ctx)
        rows[name] = rep
    assert rows["start"].gates["c01"] == "fail" and rows["migrated"].gates["c01"] == "pass"
    for rep in rows.values():
        assert rep.gates["r01"] == "infra" and rep.infra["status"] == "masked"  # the GPU milestone was not run: never a 0


def test_rebuilt_environment_keeps_the_apptainer_driver_libraries(tmp_path: Path, monkeypatch):
    import sys

    from crucible.verify.runtime import trusted as tr

    show = [sys.executable, "-c", "import os; print(os.environ.get('LD_LIBRARY_PATH', ''))"]
    monkeypatch.setattr(tr, "APPTAINER_DRIVER_LIBS", str(tmp_path / "absent"))
    t = Trusted(tmp_path / "w")
    assert t.run(show, env={"LD_LIBRARY_PATH": "/k/ref/bin"}).stdout.strip() == "/k/ref/bin"  # Docker hosts: unchanged
    libs = tmp_path / "libs"
    libs.mkdir()
    monkeypatch.setattr(tr, "APPTAINER_DRIVER_LIBS", str(libs))
    assert t.run(show, env={"LD_LIBRARY_PATH": "/k/ref/bin"}).stdout.strip() == f"/k/ref/bin:{libs}"  # the grader's own path first
    assert t.run(show).stdout.strip() == str(libs)
