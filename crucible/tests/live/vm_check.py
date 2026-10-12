"""Live QEMU/KVM checks (``world/apps/vm.py``) against a Docker daemon with ``/dev/kvm`` (the lab, over ``DOCKER_HOST``).

    DOCKER_HOST=ssh://ahmed@100.100.101.36 python tests/live/vm_check.py --pool 4 --resets 12 --out docs/research/m12-infra/live

1. ``VMPool(4).start()``: the first VM cold-boots and saves the golden ``ready`` state; the others resume it.
2. Isolation and reset: a VM writes a file, ``release`` resets it (``loadvm ready``), the file is gone on re-acquire.
3. Mid-episode RAM snapshots through a World: ``before_snapshot`` → ``savevm``; diverge; ``restore`` → ``loadvm``; the
   guest's files and a process started before the snapshot are back exactly.
4. Stress: ``resets`` acquire → exec → release cycles across the pool, concurrently; every exec succeeds, every reset is
   clean (no file from the previous user).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path


async def main(a) -> dict:
    from crucible.core.schema.actions import ToolCall
    from crucible.core.schema.world import WorldSpec
    from crucible.core.store import Store
    from crucible.world.apps.vm import VMPool
    from crucible.world.world import World

    dh = os.environ.get("DOCKER_HOST")
    out: dict = {"docker_host": dh}
    pool = VMPool.named("live", a.pool, mem_mb=1024, cpus=2, docker_host=dh)
    t0 = time.monotonic()
    await pool.start()
    out["pool_start"] = {"vms": a.pool, "wall_s": round(time.monotonic() - t0, 1), "boot_s": pool.stats["boot_s"],
                         "first_vm": {k: round(v, 2) for k, v in pool.vms[0].timings.items()}, "others": [{k: round(v, 2) for k, v in v.timings.items()} for v in pool.vms[1:]]}
    try:
        # 2. isolation and reset
        vm = await pool.acquire()
        w = await vm.exec("echo secret > /root/leak.txt && uname -r && nproc && free -m | sed -n 2p")
        pool.release(vm)
        seen = []
        for _ in range(a.pool):
            v = await pool.acquire()
            seen.append((await v.exec("test -f /root/leak.txt && echo LEAK || echo clean"))["stdout"].strip())
            pool.release(v)
        out["isolation"] = {"guest": w["stdout"].split("\n")[:3], "after_reset": seen}
        await asyncio.sleep(1)

        # 3. a World with the VM App: mid-episode RAM snapshot and restore
        spec = WorldSpec.model_validate({"world": "live/vm", "apps": [{"id": "vm", "template": "common/vm", "state": {"adapter": "json_doc", "params": {"path": "vm.json"}},
                                                                        "params": {"pool": "live", "pool_size": a.pool}}],
                                         "actors": [{"id": "agent", "kind": "policy", "tools": ["vm.*"]}]})
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="crucible-vmcheck-"))
        world = World(spec, instructions={"agent": "use the VM"}, store=Store(tmp / "store"), workdir=tmp / "w")
        await world.start()
        await world.reset()

        async def call(tool, args, i):
            r = await world.step("agent", ToolCall(tool_call_id=f"c{i}", function_name=f"vm.{tool}", arguments=args))
            return r.observation.tool_results[0]

        await call("write_file", {"path": "/root/work/notes.txt", "content": "before snapshot\n"}, 1)
        await call("exec", {"command": "nohup sleep 3600 >/dev/null 2>&1 & echo $! > /root/work/pid"}, 2)
        t1 = time.monotonic()
        ref = await world.snapshot()
        snap_s = time.monotonic() - t1
        await call("write_file", {"path": "/root/work/notes.txt", "content": "diverged\n"}, 3)
        await call("exec", {"command": "kill $(cat /root/work/pid); rm -rf /root/work/extra; mkdir /root/work/extra"}, 4)
        t2 = time.monotonic()
        await world.restore(ref)
        restore_s = time.monotonic() - t2
        notes = (await call("read_file", {"path": "/root/work/notes.txt"}, 5)).content
        alive = (await call("exec", {"command": "kill -0 $(cat /root/work/pid) && echo alive || echo dead; test -d /root/work/extra && echo extra || echo noextra"}, 6)).content
        out["world_snapshot"] = {"savevm_s": round(snap_s, 2), "loadvm_s": round(restore_s, 2), "notes_after_restore": notes.strip(), "process_and_dirs": json.loads(alive)["stdout"].split()}
        await world.close()

        # 4. stress: concurrent acquire → exec → release
        async def cycle(i: int) -> dict:
            v = await pool.acquire()
            leak = (await v.exec("test -f /root/mine && echo LEAK || echo clean"))["stdout"].strip()
            r = await v.exec(f"echo {i} > /root/mine && python3 -c 'print(sum(range(10**6)))'")
            pool.release(v)
            return {"leak": leak, "ok": r["exit_code"] == 0}

        t3 = time.monotonic()
        res = await asyncio.gather(*(cycle(i) for i in range(a.resets)))
        await asyncio.sleep(0)
        out["stress"] = {"cycles": a.resets, "wall_s": round(time.monotonic() - t3, 1), "all_ok": all(r["ok"] for r in res),
                         "leaks": sum(r["leak"] == "LEAK" for r in res), "reset_s_median": statistics.median(pool.stats["reset_s"]) if pool.stats["reset_s"] else None,
                         "reset_s_max": max(pool.stats["reset_s"]) if pool.stats["reset_s"] else None, "resets": pool.stats["resets"]}
    finally:
        await pool.close()
    out["pass"] = bool(out.get("isolation", {}).get("after_reset") == ["clean"] * a.pool and out.get("world_snapshot", {}).get("notes_after_restore") == "before snapshot"
                       and out.get("world_snapshot", {}).get("process_and_dirs") == ["alive", "noextra"] and out.get("stress", {}).get("all_ok") and out["stress"]["leaks"] == 0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--resets", type=int, default=12)
    ap.add_argument("--out", default="docs/research/m12-infra/live")
    a = ap.parse_args()
    t0 = time.time()
    try:
        res = asyncio.run(main(a))
    except Exception as e:  # noqa: BLE001
        import traceback

        res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "vm_check.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:4000])
