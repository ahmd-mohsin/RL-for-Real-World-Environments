"""M1 exit gate (coding half): the seed worlds of tasks 01-04 run on the lab RTX 5090 through Harbor.

For each seed: start the World (Harbor GPU Docker on the remote daemon, ``network_mode: none``), probe
the GPU from inside the sandbox, apply the oracle patch (as the verifier would, not the policy),
snapshot, tear down, then restore into 3 freshly started containers — each must reproduce the
snapshot fingerprint exactly (ARCHITECTURE M1 gate: "identical fingerprints, 3 out of 3").

Run: ``CRUCIBLE_GPU=1 pytest tests/gpu -m gpu`` (optionally ``CRUCIBLE_SEEDS=task04_pcl_gpu_cuda12,...``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from crucible.core.schema.state import Query
from crucible.core.store import Store
from crucible.world.world import World

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
if not os.environ.get("CRUCIBLE_GPU"):
    pytest.skip("set CRUCIBLE_GPU=1 to run GPU gates on the lab host", allow_module_level=True)

from crucible_pack_coding import seed_dirs

from crucible.interop.harbor.task_io import load_crucible_task

REPORT = Path(__file__).resolve().parents[2] / "docs" / "modules" / "reports" / "m1_seed_worlds_gpu.json"
SELECT = [s for s in os.environ.get("CRUCIBLE_SEEDS", "").split(",") if s]
SEEDS = [d for d in seed_dirs() if not SELECT or d.name in SELECT]


def _main_repo(world_spec) -> str:
    return next(a.id for a in world_spec.apps if a.params.get("shell"))


@pytest.mark.parametrize("seed", SEEDS, ids=[d.name for d in SEEDS])
async def test_seed_world_snapshot_restore_on_gpu(seed: Path, tmp_path: Path):
    t = load_crucible_task(seed)
    spec = t.world
    main = _main_repo(spec)
    root = spec.app(main).state.params["root"]
    store = Store(tmp_path / "store")
    rec: dict = {"seed": seed.name, "image": spec.backend_params["docker_image"]}
    t0 = time.monotonic()
    w = World(spec, store=store, pack_root=seed, workdir=tmp_path / "w0")
    await w.start()
    try:
        rec["start_s"] = round(time.monotonic() - t0, 1)
        await w.reset()
        gpu = await w.verifier_view().query("gpu", Query(op="gpu"))
        assert gpu.ok, gpu
        rec["gpu"] = gpu.value["gpus"][0]["name"]
        assert "RTX" in rec["gpu"]
        initial = await w.snapshot()
        # Oracle patch applied by the verifier side (system exec), never through the policy's tools.
        ex = w.backend.exec_for(spec.app(main))
        oracle = t.harbor.path / "solution" / "solution.patch"
        await ex.push_file(oracle, "/tmp/.crucible_oracle.patch")
        out = await ex.run(f"cd {root} && git apply --binary /tmp/.crucible_oracle.patch && rm -f /tmp/.crucible_oracle.patch", timeout_s=600)
        assert out.ok, out.stdout[-2000:]
        t1 = time.monotonic()
        ref = await w.snapshot()
        rec["snapshot_s"] = round(time.monotonic() - t1, 1)
        rec["main_repo"] = {"files": ref.apps[main].meta["files"], "bytes": ref.apps[main].meta["bytes"], "tar_bytes": ref.apps[main].size_bytes}
        assert ref.apps[main].fingerprint != initial.apps[main].fingerprint
        diff = await w.apps[main].state.diff(initial.apps[main], ref.apps[main])
        rec["oracle_diff_files"] = len(diff.files)
    finally:
        await w.close()
    rec["restores"] = []
    for i in range(3):
        w2 = World(spec, store=store, pack_root=seed, workdir=tmp_path / f"w{i + 1}")
        await w2.start()
        try:
            t2 = time.monotonic()
            await w2.restore(ref)
            got = await w2.snapshot()
            rec["restores"].append({"fingerprint_equal": got.fingerprint == ref.fingerprint, "restore_s": round(time.monotonic() - t2, 1)})
            assert got.fingerprint == ref.fingerprint
        finally:
            await w2.close()
    rec["total_s"] = round(time.monotonic() - t0, 1)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    data[seed.name] = rec
    REPORT.write_text(json.dumps(data, indent=1, sort_keys=True))
