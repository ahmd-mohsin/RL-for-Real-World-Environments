"""m13 D-4: one call starts a job (engine servers behind the gateway, rollout workers, env servers, controller) and
supervises it: a killed worker, engine and env server are restarted; the run completes with nothing trained twice."""

from __future__ import annotations

import asyncio
import os
import signal

from crucible.train.launch import run_job


async def test_job_survives_killed_worker_engine_and_env_server(tmp_path):
    spec = {"run": {"steps": 30, "n": 4, "B": 4, "max_staleness": 2, "lease_s": 1.5, "max_wall_s": 300, "health_interval_s": 0.3},
            "engines": {"count": 2, "capacity": 64, "decode_delay_s": 0.002},
            "workers": {"count": 2, "concurrency": 12, "rollout": "copy"},
            "env_servers": {"count": 1, "max_concurrent": 8}}
    kills = []

    def on_start(la, rt, gw, lc):
        async def chaos():
            while len(rt.report.steps) < 2:
                await asyncio.sleep(0.05)
            for kind in ("worker", "engine", "env"):
                ch = la.by_kind(kind)[0]
                os.killpg(ch.proc.pid, signal.SIGKILL)
                kills.append(ch.name)
                await asyncio.sleep(0.3)

        asyncio.get_running_loop().create_task(chaos())

    out = await run_job(spec, tmp_path / "job", on_start=on_start)
    assert out["steps"] == 30 and out["stop_reason"] == "steps" and out["publishes"] == 30
    assert kills == ["worker0", "engine0", "env0"]
    assert all(out["restarts"][k] >= 1 for k in kills)  # the supervisor restarted every killed child
    assert out["trained"] == out["unique_trained"] and out["masked"] == 0  # nothing trained twice, nothing masked
    assert out["evictions"] >= 1 and out["revived"] >= 1  # the gateway evicted the dead engine and revived it after restart
    assert (tmp_path / "job" / "metrics" / "metrics.jsonl").read_text().count("\n") == 30
