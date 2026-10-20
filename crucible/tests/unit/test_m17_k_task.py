"""Kernel family windows and the LongTask (swarm-pipeline/06 §2, §4): program discovery from the repository's own build
files, compile/run milestones with claimed edges, rungs from executed evidence, the oracle patch, prompt + ledger + lint,
simplify, conformance, and masking without a GPU."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from crucible.factory.lanes import kernel as K
from crucible.factory.longtask import check_longtask

HDR_A = "#pragma once\n__device__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_down(v, o); return v; }\n"
HDR_B = "#pragma once\n__device__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o); return v; }\n"
PROG = '#include <cstdio>\n#include "../lib/warp.cuh"\n__global__ void k(float* o) { o[threadIdx.x] = warp_sum(o[threadIdx.x]); }\nint main() { return 0; }\n'
TEX_A = "texture<float, 1> t;\n__global__ void g(float* o) { o[threadIdx.x] = tex1Dfetch(t, threadIdx.x); }\nint main() { cudaBindTexture(0, t, 0, 4); return 0; }\n"
TEX_B = ("__global__ void g(cudaTextureObject_t t, float* o) { o[threadIdx.x] = tex1Dfetch<float>(t, threadIdx.x); }\n"
         "int main() { cudaTextureObject_t t = 0; return (int)t; }\n")


def tree(tmp: Path, files: dict[str, str]) -> Path:
    for f, text in files.items():
        p = tmp / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp


@pytest.fixture
def lib_tree(tmp_path: Path) -> Path:
    return tree(tmp_path / "a", {"lib/warp.cuh": HDR_A, "app/scan.cu": PROG, "app/Makefile": "source = scan.cu\nrun: main\n\t./main 1024 7\n",
                                 "tools/sum.cu": PROG.replace("../lib/warp.cuh", "warp.cuh"), "tools/CMakeLists.txt": "include_directories(../lib)\nadd_executable(sum sum.cu)\n"})


def test_programs_come_from_the_repositorys_build_files(lib_tree: Path):
    progs = K.discover_programs(lib_tree, ["app/scan.cu", "tools/sum.cu"])
    scan, tool = progs["app_scan"], progs["tools_sum"]
    assert scan.sources == ["app/scan.cu"] and scan.cases == {"run0": ["1024", "7"]} and "lib/warp.cuh" in scan.closure
    assert tool.sources == ["tools/sum.cu"] and "lib" in tool.includes and "lib/warp.cuh" in tool.closure and not tool.needs


def test_compile_and_run_milestones_with_claimed_edges(lib_tree: Path):
    progs = K.discover_programs(lib_tree, ["app/scan.cu", "tools/sum.cu"])
    ms = K.build_milestones(["lib/warp.cuh"], {"lib/warp.cuh": ["app/scan.cu", "tools/sum.cu"]}, progs)
    comp = [m for m in ms if m["kind"] == "compile"]
    runs = [m for m in ms if m["kind"] == "run"]
    assert {m["tu"] for m in comp} == {"app/scan.cu", "tools/sum.cu"} and all(m["files"] == ["lib/warp.cuh"] for m in comp)
    assert len(runs) == 2 and all(len(r["requires"]) == 1 for r in runs) and all(not r["requires_evidence"] for r in runs)


def test_unsupported_needs_are_left_out(tmp_path: Path):
    t = tree(tmp_path, {"gl/v.cu": "#include <GL/gl.h>\n" + TEX_A})
    progs = K.discover_programs(t, ["gl/v.cu"])
    assert progs["gl_v"].needs == ["gl"] and K.build_milestones(["gl/v.cu"], {"gl/v.cu": ["gl/v.cu"]}, progs) == []


def _window(ms: list[dict], files: list[str], programs: dict) -> dict:
    return {"id": "k-t-1", "repo": "o_r", "a": "a" * 40, "b": "b" * 40, "milestones": ms, "migrated_files": files, "programs": programs,
            "kinds": {"warp_sync": {}}, "oracle_patch": "", "b_date": "2026-01-01T00:00:00+00:00"}


def test_rungs_need_executed_evidence(lib_tree: Path):
    progs = {p: {"main": v.main} for p, v in K.discover_programs(lib_tree, ["app/scan.cu", "tools/sum.cu"]).items()}
    one = _window([{"id": "c01", "kind": "compile", "tu": "app/scan.cu", "files": ["app/scan.cu"], "requires": []},
                   {"id": "r01", "kind": "run", "program": "app_scan", "files": ["app/scan.cu"], "requires": ["c01"], "requires_evidence": {"c01": "executed"}}],
                  ["app/scan.cu"], progs)
    assert K.rung_of(one) == "L1"
    two = _window([{"id": "c01", "kind": "compile", "tu": "app/scan.cu", "files": ["lib/warp.cuh"], "requires": []},
                   {"id": "c02", "kind": "compile", "tu": "tools/sum.cu", "files": ["lib/warp.cuh"], "requires": []},
                   {"id": "r01", "kind": "run", "program": "app_scan", "files": ["lib/warp.cuh"], "requires": ["c01"], "requires_evidence": {"c01": "claimed"}},
                   {"id": "r02", "kind": "run", "program": "tools_sum", "files": ["lib/warp.cuh"], "requires": ["c02"], "requires_evidence": {"c02": "claimed"}}],
                  ["lib/warp.cuh", "app/scan.cu"], progs)
    assert K.rung_of(two) == "L2-short"  # claimed edges do not count
    for m in two["milestones"][2:]:
        m["requires_evidence"] = {m["requires"][0]: "executed"}
    assert K.rung_of(two) == "L3"  # the header's consumers live outside lib/
    assert K.cross_edges(two) == [("r01", "c01"), ("r02", "c02")]


def test_components_split_independent_files(tmp_path: Path):
    t = tree(tmp_path, {"x/a.cu": TEX_A, "x/b.cu": TEX_A, "lib/warp.cuh": HDR_A, "app/scan.cu": PROG})
    progs = K.discover_programs(t, ["x/a.cu", "x/b.cu", "app/scan.cu"])
    aff = {"x/a.cu": ["x/a.cu"], "x/b.cu": ["x/b.cu"], "lib/warp.cuh": ["app/scan.cu"], "app/scan.cu": ["app/scan.cu"]}
    comps = K.components(["x/a.cu", "x/b.cu", "lib/warp.cuh", "app/scan.cu"], progs, aff)
    assert sorted(comps) == [["app/scan.cu", "lib/warp.cuh"], ["x/a.cu"], ["x/b.cu"]]


def test_oracle_patch_applies_to_the_start(tmp_path: Path):
    a = tree(tmp_path / "a", {"lib/warp.cuh": HDR_A, "app/scan.cu": PROG})
    patch = K.oracle_patch_of(a, {"lib/warp.cuh": HDR_B, "new/added.cu": "int x;\n"})
    assert "diff --git a/lib/warp.cuh b/lib/warp.cuh" in patch and "__shfl_down_sync" in patch
    subprocess.run(["git", "init", "-q"], cwd=a, check=True)
    p = subprocess.run(["git", "apply", "--check", "-"], cwd=a, input=patch, text=True, capture_output=True, check=False)
    assert p.returncode == 0, p.stderr


def authored(tmp_path: Path) -> dict:
    a = tree(tmp_path / "a", {"lib/warp.cuh": HDR_A, "app/scan.cu": PROG})
    progs = K.discover_programs(a, ["app/scan.cu"])
    w = _window(K.build_milestones(["lib/warp.cuh"], {"lib/warp.cuh": ["app/scan.cu"]}, progs), ["lib/warp.cuh"],
                {p: {**v.spec(), "main": v.main} for p, v in progs.items()})
    w["oracle_files"] = {"lib/warp.cuh": HDR_B}
    w["oracle_patch"] = K.oracle_patch_of(a, w["oracle_files"])
    w["oracle_new_names"] = K.oracle_new_names(w)
    w["prompt"], w["spans"] = K.compose_prompt("The reduction no longer builds on the current toolkit.", w)
    w["requirements"] = K.requirement_texts(w)
    w["locate"] = K.locate_lines(a, w)
    w["tree_a"] = str(a)
    return w


def test_prompt_ledger_and_lint(tmp_path: Path):
    w = authored(tmp_path)
    led, problems = K.ledger_of(w)
    assert problems == [] and len(led.requirements) == len(w["milestones"])
    assert all(w["prompt"][s:e] == r["text"] for r in w["requirements"] for s, e in [w["spans"][r["id"]]])
    assert K.lint_prompt(w["prompt"], w) == []
    bad = w["prompt"] + "\nSee https://github.com/o/r/commit/0123456789abcdef0123 for the fix.\n"
    assert any(p.startswith("leak") for p in K.lint_prompt(bad, w))
    assert K.scrub("Fix #123 (deadbeefdeadbeef)\nCo-Authored-By: X <x@y>") == "Fix  ()"


class FakeBackend:
    def __init__(self, gates: dict[str, int], infra: dict | None = None):
        self.gates, self.infra = gates, infra

    async def grade(self, window, patch):
        return {"gates": self.gates, "infra": self.infra or {"status": "ok"}}


def test_longtask_conformance_and_grading(tmp_path: Path):
    w = authored(tmp_path)
    for m in w["milestones"]:
        m["requires_evidence"] = dict.fromkeys(m["requires"], "executed")
    task = K.KernelTask(w)
    assert check_longtask(task) == []
    rep = asyncio.run(task.grade_patch(task.oracle_patch()))
    assert rep.infra["status"] == "masked" and rep.infra["failure_kind"] == "no_gpu"  # V8: never a 0
    ids = [m["id"] for m in w["milestones"]]
    full = K.KernelTask(w, FakeBackend(dict.fromkeys(ids, 1)))
    half = K.KernelTask(w, FakeBackend({ids[0]: 1, ids[1]: 0}))
    r1, r2 = asyncio.run(full.grade_patch("x")), asyncio.run(half.grade_patch("x"))
    assert r1.reward == 1.0 and r1.meta["phi"] == 1.0
    assert r2.reward == 0.25 and r2.meta["phi"] == 0.5 and r2.meta["full_pass"] is False
    masked = asyncio.run(K.KernelTask(w, FakeBackend({}, {"status": "masked", "failure_kind": "node_contention"})).grade_patch("x"))
    assert masked.infra["status"] == "masked"


def test_simplify(tmp_path: Path):
    w = authored(tmp_path)
    for m in w["milestones"]:
        m["requires_evidence"] = dict.fromkeys(m["requires"], "executed")
    t = K.KernelTask(w)
    loc = t.simplify("locate")
    assert "lib/warp.cuh" in loc.prompt and "__shfl_down" in loc.prompt and loc.task_id.endswith("~locate")
    pre = t.simplify("prerequisites")
    assert [m["kind"] for m in pre.milestones] == ["run"] and pre.window["start_patch_milestones"] == ["c01"]
    with pytest.raises(ValueError):
        t.simplify("host_change")


def test_fault_mutants_rewrite_the_oracle():
    text = "cudaTextureDesc d = {}; d.addressMode[0] = cudaAddressModeClamp; d.filterMode = cudaFilterModeLinear; x = __shfl_down_sync(0xffffffff, v, 1);"
    mus = K.fault_mutants({"k.cu": text, "doc.md": text}, lambda f: "c01")
    ids = {m["id"].split("-")[1] for m in mus}
    assert {"addr_clamp_to_border", "filter_linear_to_point", "mask_active", "shfl_down_to_up"} <= ids and all(m["file"] == "k.cu" for m in mus)
    assert all(m["content"] != text for m in mus)


def test_oracle_patch_keeps_crlf_sources_applicable(tmp_path: Path):
    """Found by running S5 through the kit on rodinia's kmeans: a text-mode diff of CRLF sources did not apply."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "k.cu").write_bytes(b"texture<float, 1> t;\r\nint f() { return 1; }\r\n")
    new = K.read_raw(a / "k.cu").replace("texture<float, 1> t;", "cudaTextureObject_t t;")
    patch = K.oracle_patch_of(a, {"k.cu": new})
    assert "\r\n" in patch
    subprocess.run(["git", "init", "-q"], cwd=a, check=True)
    p = subprocess.run(["git", "apply", "--check", "-"], cwd=a, input=patch.encode(), capture_output=True, check=False)
    assert p.returncode == 0, p.stderr
