"""Kernel debugging lane, the migration family (swarm-pipeline/10 §3.3): the CUDA 13 kinds and the grader's static scan in
step, the miner's classification of real CUDA 13 changes, grouping by program and kind, the merged multi-program window,
the prompt in the generation's toolkit. No Docker."""

from __future__ import annotations

import re

from crucible.factory.lanes import kernel as K
from crucible.factory.lanes import kernel_debug_migrate as kdm
from crucible.factory.lanes import kernel_mine as km
from crucible.factory.lanes import kernel_plan as kp

TOPK_REMOVED = ["        T global_min = BlockReduceT(temp_storage).Reduce(local_min, cub::Min());"]
TOPK_ADDED = ["struct Min", "{", "  operator()(T &&t, U &&u) const", "        T global_min = BlockReduceT(temp_storage).Reduce(local_min, Min());"]
PREFETCH_REMOVED = ["    CUDACHECK(cudaMemPrefetchAsync(A, numElements*sizeof(float), gpuDeviceId));"]
PREFETCH_ADDED = ["    cudaMemLocation deviceLoc;", "    CUDACHECK(cudaMemPrefetchAsync(A, numElements*sizeof(float), deviceLoc, 0));"]


def test_cuda13_kinds_and_static_scan_agree():
    samples = {"cccl3_functors": "x = BlockReduce(t).Reduce(v, cub::Max());", "memloc_api": "cudaMemPrefetchAsync(p, n, dev, stream);",
               "devprop_removed": "if (prop.computeMode == cudaComputeModeProhibited) {"}
    for k, line in samples.items():
        assert kdm.KINDS13[k][0].search(line), k
        assert re.search(kp.LEGACY[k], line), k
        assert k in K.KIND_TEXT
    assert not re.search(kp.LEGACY["memloc_api"], "cudaMemPrefetchAsync(p, n, deviceLoc, 0);")  # the CUDA 13 form is not legacy
    assert not re.search(kp.LEGACY["cccl3_functors"], "cuda::maximum<>{}")


def test_real_cuda13_changes_classify():
    f1 = km.FileChange("src/topk-cuda/topk_per_row_kernels.h", TOPK_REMOVED, TOPK_ADDED)
    f2 = km.FileChange("src/prefetch-cuda/main.cu", PREFETCH_REMOVED, PREFETCH_ADDED)
    kinds, files = km.classify([f1, f2], kinds=kdm.KINDS13, complete=tuple(kdm.KINDS13))
    assert set(kinds) == {"cccl3_functors", "memloc_api"} and files == [f1.path, f2.path]
    # the default kind table is untouched: the same change is not a CUDA 11 -> 12 migration
    assert km.classify([f1, f2])[0] == {}


def mc(sha: str, date: str, files: list[str], kinds: list[str]) -> km.MigrationCommit:
    return km.MigrationCommit(repo="r", sha=sha, parent=sha + "p", date=date, subject=sha, kinds={k: {"legacy_removed": 1} for k in kinds},
                              migrated_files=files, changed_files=files, lines_added=1, lines_removed=1, licence="BSD-3-Clause")


def test_groups_prefer_new_kinds_and_skip_other_backends():
    cs = [mc("a", "2026-04-02T00:00:01", ["src/topk-cuda/k.h"], ["cccl3_functors"]), mc("b", "2026-04-02T00:00:02", ["src/moe-cuda/k.h"], ["cccl3_functors"]),
          mc("c", "2026-04-02T00:00:03", ["src/prefetch-cuda/main.cu"], ["memloc_api"]), mc("d", "2026-04-02T00:00:04", ["src/atomic-cuda/main.cu"], ["devprop_removed"]),
          mc("h", "2026-04-02T00:00:00", ["src/topk-hip/main.cu"], ["devprop_removed"])]
    gs = kdm.group_commits(cs, max_size=3)
    assert [[c.sha for c in g] for g in gs] == [["a", "c", "d"]]  # new kinds first; the HIP twin never joins
    assert kdm.program_key("texturing/tex1dfetch.cu") == "texturing/tex1dfetch" and kdm.program_key("src/topk-cuda/k.h") == "src/topk-cuda"


def window(wid: str, n: int) -> dict:
    return {"id": wid, "repo": "r", "a": "A", "b": "B", "migrated_files": [f"p{n}/k.cuh"], "kinds": {"cccl3_functors": {}}, "oracle_files": {f"p{n}/k.cuh": "x"},
            "programs": {f"p{n}_main": {"main": f"p{n}/main.cu"}},
            "milestones": [{"id": "c01", "kind": "compile", "tu": f"p{n}/main.cu", "program": f"p{n}_main", "files": [f"p{n}/k.cuh"], "programs": [f"p{n}_main"],
                            "requires": []},
                           {"id": "r01", "kind": "run", "program": f"p{n}_main", "files": [f"p{n}/k.cuh"], "programs": [f"p{n}_main"], "requires": ["c01"]}]}


def test_merge_windows_renumbers_and_remaps_edges():
    w = kdm.merge_windows([window("w1", 1), window("w2", 2)], "km-x")
    assert [m["id"] for m in w["milestones"]] == ["c01", "c02", "r01", "r02"]
    assert [m["requires"] for m in w["milestones"] if m["kind"] == "run"] == [["c01"], ["c02"]]
    assert set(w["programs"]) == {"p1_main", "p2_main"} and w["id"] == "km-x"
    reqs = kdm.migration_requirements({**w, "graded_kinds": ["cccl3_functors"]}, "cuda13")
    assert "CUDA 13" in reqs[0]["text"] and "CUDA 12.8" not in reqs[0]["text"]
    text, spans = kdm.compose("We moved the cluster to the new toolkit.", {**w, "graded_kinds": ["cccl3_functors"]}, "cuda13")
    assert "Only CUDA 13 is installed" in text and all(text[a:b] for a, b in spans.values())
