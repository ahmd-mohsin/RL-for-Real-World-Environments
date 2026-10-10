"""Launcher (m13 D-4; audit 05 §2.2, the prime-rl process model): one command starts a whole training job as OS processes —
engine servers behind the gateway, rollout workers, env servers — with the controller (runtime + lease table) in the
launching process, and supervises them.

    python -m crucible.train.launch job.toml

- **Ray-free by default** (works on the lab node: Docker, no Kubernetes, no Ray): children are processes in their own
  sessions, on fixed ports chosen at start.
- **Supervision**: a dead engine server, worker or env server is restarted (same port, same arguments) up to
  ``max_restarts`` per child. The system around it is already built to absorb the outage: the gateway turns the dead
  engine's calls into re-issued aborts and revives the restarted engine only after replaying the current weights
  (``GatewayPool``); a dead worker's leases expire and are re-leased (``LeaseController``); a dead env server masks its
  episodes (``env_unreachable``), never scores them 0.
- **Shutdown**: every child's process group is killed when the job ends, however it ends.

The job spec (TOML) — the toy policy (``TinyPolicy``) and token environments make it runnable on any laptop:

    [run]      steps, n, B, max_staleness, topology, lr, seed, max_wall_s
    [engines]  count, capacity, decode_delay_s
    [workers]  count, concurrency, rollout ("copy" | "module:factory" returning a rollout function), cpu_s
    [env_servers] count, tasks ("module:factory"), max_concurrent

Ray (detached named engine actors, node-affinity workers) is the optional second launcher of the audit (item G); it is
not built here.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Child:
    kind: str
    name: str
    argv: list[str]
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    proc: subprocess.Popen | None = None
    restarts: int = 0

    def start(self, log_dir: Path) -> None:
        log = open(log_dir / f"{self.name}.log", "ab")  # noqa: SIM115 -- handed to the child for its lifetime
        self.proc = subprocess.Popen(self.argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, **self.env})
        log.close()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()


class Launcher:
    def __init__(self, spec: dict[str, Any], *, work: Path, max_restarts: int = 3, supervise_every_s: float = 0.5):
        self.spec, self.work, self.max_restarts, self.supervise_every_s = spec, Path(work), max_restarts, supervise_every_s
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "logs").mkdir(exist_ok=True)
        self.children: list[Child] = []
        self.events: list[dict[str, Any]] = []
        self._sup: asyncio.Task | None = None

    def plan(self, controller_url: str) -> None:
        py = sys.executable
        e, w, s = self.spec.get("engines", {}), self.spec.get("workers", {}), self.spec.get("env_servers", {})
        r = self.spec.get("run", {})
        for i in range(int(e.get("count", 0))):
            port = _port()
            self.children.append(Child("engine", f"engine{i}", [py, "-m", "crucible.train.gateway", "engine", "--port", str(port), "--seed", str(r.get("seed", 1)),
                                                               "--capacity", str(e.get("capacity", 64)), "--delay", str(e.get("decode_delay_s", 0.002))],
                                       url=f"http://127.0.0.1:{port}"))
        env_urls = []
        for i in range(int(s.get("count", 0))):
            port = _port()
            env_urls.append(f"http://127.0.0.1:{port}")
            self.children.append(Child("env", f"env{i}", [py, "-m", "crucible.interop.nemo_gym.server", "--tasks", s.get("tasks", "crucible_pack_toy.runtime:nemo_tasks"),
                                                         "--port", str(port), "--max-concurrent", str(s.get("max_concurrent", 64))], url=env_urls[-1]))
        for i in range(int(w.get("count", 0))):
            self.children.append(Child("worker", f"worker{i}", [py, "-m", "crucible.train.workers", "--controller", controller_url, "--rollout", w.get("rollout", "copy"),
                                                               "--concurrency", str(w.get("concurrency", 16)), "--cpu-s", str(w.get("cpu_s", 0.0)), "--id", f"worker{i}"],
                                       env={"CRUCIBLE_ENV_URLS": ",".join(env_urls)}))

    async def _wait_health(self, kinds: tuple[str, ...], timeout_s: float = 120.0) -> None:
        import httpx

        deadline = time.monotonic() + timeout_s
        async with httpx.AsyncClient(timeout=1.0) as c:
            for ch in [c_ for c_ in self.children if c_.kind in kinds and c_.url]:
                while True:
                    try:
                        if (await c.get(ch.url + "/health")).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline or not ch.alive():
                        raise RuntimeError(f"{ch.name} did not become healthy (see {self.work / 'logs' / (ch.name + '.log')})")
                    await asyncio.sleep(0.2)

    async def start(self, kinds: tuple[str, ...]) -> None:
        for ch in self.children:
            if ch.kind in kinds:
                ch.start(self.work / "logs")
        await self._wait_health(kinds)

    async def _supervise(self) -> None:
        while True:
            await asyncio.sleep(self.supervise_every_s)
            for ch in self.children:
                if ch.proc is not None and not ch.alive():
                    if ch.restarts >= self.max_restarts:
                        continue
                    ch.restarts += 1
                    self.events.append({"t": time.time(), "child": ch.name, "event": "restart", "rc": ch.proc.returncode, "n": ch.restarts})
                    ch.start(self.work / "logs")

    def supervise(self) -> None:
        if self._sup is None:
            self._sup = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        if self._sup is not None:
            self._sup.cancel()
            await asyncio.gather(self._sup, return_exceptions=True)
        for ch in self.children:
            ch.kill()

    def by_kind(self, kind: str) -> list[Child]:
        return [c for c in self.children if c.kind == kind]


async def run_job(spec: dict[str, Any], work: Path, *, on_start: Any = None) -> dict[str, Any]:
    """Run a whole job from a spec; returns a summary (steps, restarts, failures, leases)."""
    from crucible.train.engine import TinyPolicy
    from crucible.train.gateway import GatewayPool, HTTPEngine
    from crucible.train.mismatch import Correction
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.sinks import JsonlSink
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import CheckpointPublisher
    from crucible.train.workers import LeaseController

    r = spec.get("run", {})
    lc = LeaseController(lease_s=float(r.get("lease_s", 5.0)), max_attempts=int(r.get("max_attempts", 3)))
    url = await lc.start()
    la = Launcher(spec, work=work, max_restarts=int(r.get("max_restarts", 3)))
    la.plan(url)
    try:
        await la.start(("engine", "env"))
        gw = GatewayPool([HTTPEngine(c.url, capacity=int(spec.get("engines", {}).get("capacity", 64))) for c in la.by_kind("engine")],
                         health_interval_s=float(r.get("health_interval_s", 0.5)))
        tr = TorchTrainer(TinyPolicy(32, 48, seed=int(r.get("seed", 1))), TrainerConfig(lr=float(r.get("lr", 1e-2)), correction=Correction("dis")))
        rt = Runtime(RuntimeConfig(topology=r.get("topology", "colocated_partial"), steps=int(r.get("steps", 10)), max_turn_tokens=int(r.get("max_turn_tokens", 3)),
                                   max_wall_s=float(r.get("max_wall_s", 600)), sinks=[JsonlSink(work / "metrics")]),
                     SchedulerConfig(n=int(r.get("n", 4)), B=int(r.get("B", 4)), max_staleness=int(r.get("max_staleness", 2))), gw, tr,
                     CheckpointPublisher(tr, gw, work / "ckpt"), None, {"job": [f"p{i}" for i in range(100_000)]}, rollout_fn=lc.rollout_fn())
        lc.attach(rt)
        await la.start(("worker",))
        la.supervise()
        gw.start_health()
        if on_start is not None:
            on_start(la, rt, gw, lc)
        rep = await rt.run()
        await gw.stop_health()
        ids = [x.rollout_id for x in rep.trained_records]
        return {"steps": len(rep.steps), "stop_reason": rep.stop_reason, "publishes": rep.publishes, "masked": rep.masked_rollouts, "failures": dict(rep.failures),
                "restarts": {c.name: c.restarts for c in la.children}, "events": la.events, "evictions": gw.stats["evictions"], "revived": gw.stats["revived"],
                "leases": dict(lc.stats), "trained": len(ids), "unique_trained": len(set(ids))}
    finally:
        await la.stop()
        await lc.stop()


def main() -> None:  # pragma: no cover - CLI
    import argparse
    import json
    import tomllib

    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--work", default="runs/launch")
    a = ap.parse_args()
    spec = tomllib.loads(Path(a.spec).read_text())
    out = asyncio.run(run_job(spec, Path(a.work)))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
