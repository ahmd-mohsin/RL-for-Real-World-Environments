"""Docker sandbox stress against a daemon (the lab over ``DOCKER_HOST=ssh://...``, or local).

    DOCKER_HOST=ssh://ahmed@100.100.101.36 python tests/live/docker_stress.py --cold 24 --cycles 96 --conc 16

A. **Cold**: ``cold`` hardened sandboxes (``DockerSandbox``: ``--init``, no network, pids limit) started concurrently;
   each runs a command and closes. Start latency p50 / p95, failures.
B. **Warm pool, fresh mode**: ``cycles`` acquire → exec → release at concurrency ``conc``. Acquire latency, every
   container handed out once (fresh mode never reuses), refill keeping up.
C. **Warm pool, reuse mode with hygiene**: every cycle first checks that it inherited nothing (no planted file, no
   process, no user from the previous holder), then plants all three and releases. Leaks must be 0; dirty containers
   are destroyed, never pooled.
D. **Cleanup**: after the pools close, no ``crucible.sandbox`` container of this run is left on the daemon.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

IMAGE = "python:3.12-slim"


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3)


async def main(a) -> dict:
    from crucible.core.schema.world import SandboxSpec, WorldSpec
    from crucible.world.exec import _run_subprocess
    from crucible.world.pool import PooledSandbox, SandboxPool
    from crucible.world.sandbox import DockerSandbox

    dh = os.environ.get("DOCKER_HOST")
    run_tag = f"stress{int(time.time())}"
    spec = WorldSpec.model_validate({"world": f"live/{run_tag}", "network": "no-network", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})
    sb = SandboxSpec(image=IMAGE, docker_host=dh)
    env = {"DOCKER_HOST": dh} if dh else None
    if not (await _run_subprocess(["docker", "image", "inspect", IMAGE], cwd=None, env=env, timeout_s=60, stdin=None)).ok:
        await _run_subprocess(["docker", "pull", "-q", IMAGE], cwd=None, env=env, timeout_s=900, stdin=None)
    out: dict = {"docker_host": dh, "image": IMAGE}

    # A. cold
    async def cold_one(i: int) -> tuple[float | None, str | None]:
        s = DockerSandbox(sb, spec, pids_limit=256)
        t0 = time.perf_counter()
        try:
            await s.start()
            dt = time.perf_counter() - t0
            r = await s.exec.run(f"python3 -c 'print({i}*{i})'", timeout_s=60)
            return (dt, None) if r.stdout.strip() == str(i * i) else (dt, f"bad output {r.stdout[:80]!r}")
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"[:200]
        finally:
            await s.close()

    t0 = time.perf_counter()
    res = await asyncio.gather(*(cold_one(i) for i in range(a.cold)))
    starts = [d for d, e in res if d is not None and e is None]
    out["A_cold"] = {"n": a.cold, "wall_s": round(time.perf_counter() - t0, 1), "start_p50_s": pct(starts, 0.5), "start_p95_s": pct(starts, 0.95),
                     "failures": [e for _, e in res if e][:5], "failed": sum(1 for _, e in res if e)}

    # B. warm pool, fresh mode
    pool = SandboxPool(sb, spec, size=a.conc, pids_limit=256)
    await pool.warm()
    seen: list[str] = []
    acq: list[float] = []
    sem = asyncio.Semaphore(a.conc)
    errs: list[str] = []

    async def fresh_cycle(i: int) -> None:
        async with sem:
            ps = PooledSandbox(pool)
            try:
                await ps.start()
                acq.append(ps.acquire_s)
                seen.append(ps._sbx.name)
                r = await ps.exec.run("echo ok", timeout_s=30)
                if r.stdout.strip() != "ok":
                    errs.append(f"cycle {i}: {r.stdout[:60]!r}")
            except Exception as e:  # noqa: BLE001
                errs.append(f"cycle {i}: {type(e).__name__}: {e}"[:200])
            finally:
                await ps.close()

    t0 = time.perf_counter()
    await asyncio.gather(*(fresh_cycle(i) for i in range(a.cycles)))
    out["B_warm_fresh"] = {"cycles": a.cycles, "concurrency": a.conc, "wall_s": round(time.perf_counter() - t0, 1), "acquire_p50_s": pct(acq, 0.5),
                           "acquire_p95_s": pct(acq, 0.95), "distinct_containers": len(set(seen)), "reused_any": len(set(seen)) != len(seen),
                           "errors": errs[:5], "stats": dict(pool.stats)}
    await pool.close()

    # C. warm pool, reuse mode: each holder checks it inherited nothing, then plants a file, a process and a user
    rpool = SandboxPool(sb, spec, size=max(2, a.conc // 2), reuse=True, pids_limit=256)
    await rpool.warm()
    leaks: list[str] = []
    rerrs: list[str] = []
    rsem = asyncio.Semaphore(max(2, a.conc // 2))

    async def reuse_cycle(i: int) -> None:
        async with rsem:
            s = await rpool.acquire()
            try:
                chk = await s.exec.run("ls -A /crucible /tmp 2>/dev/null; id planted 2>&1; ps -eo args | grep -v grep | grep -c 'sleep 4242'", timeout_s=30, user="0")
                lines = chk.stdout.split()
                if "secret" in lines or "uid=" in chk.stdout or (lines and lines[-1] != "0"):
                    leaks.append(f"cycle {i} inherited: {chk.stdout[:120]!r}")
                await s.exec.run("useradd -m planted; mkdir -p /crucible && echo x > /crucible/secret && echo y > /tmp/secret; (nohup sleep 4242 >/dev/null 2>&1 &)",
                                 timeout_s=60, user="0")
            except Exception as e:  # noqa: BLE001
                rerrs.append(f"cycle {i}: {type(e).__name__}: {e}"[:200])
            finally:
                await rpool.release(s)

    t0 = time.perf_counter()
    await asyncio.gather(*(reuse_cycle(i) for i in range(a.cycles)))
    out["C_warm_reuse_hygiene"] = {"cycles": a.cycles, "wall_s": round(time.perf_counter() - t0, 1), "leaks": len(leaks), "leak_examples": leaks[:3],
                                   "errors": rerrs[:5], "stats": dict(rpool.stats)}
    await rpool.close()

    # D. nothing of this run is left on the daemon
    await asyncio.sleep(2)
    left = (await _run_subprocess(["docker", "ps", "-a", "-q", "--filter", f"label=crucible.world=live/{run_tag}"], cwd=None, env=env, timeout_s=60, stdin=None)).stdout.split()
    out["D_cleanup"] = {"left_containers": len(left)}
    out["pass"] = bool(out["A_cold"]["failed"] == 0 and not errs and not out["B_warm_fresh"]["reused_any"] and not rerrs and not leaks and not left)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", type=int, default=24)
    ap.add_argument("--cycles", type=int, default=96)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "docs" / "research" / "m12-infra" / "live"))
    a = ap.parse_args()
    t0 = time.time()
    try:
        res = asyncio.run(main(a))
    except Exception as e:  # noqa: BLE001
        import traceback

        res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "docker_stress.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:4000])
