"""Single controller over an SPMD trainer (M12-infra C; art. §2 "controller": one driver process owns the loop, N
trainer ranks run the same program).

The ``Runtime`` is one asyncio process: it calls ``trainer.step`` / ``state_dict`` / ``reset_optimizer`` /
``offload`` / ``onload`` on rank ``src``. A sharded trainer (``FSDPTrainer``, ``DistributedTorchTrainer``) needs every
rank inside each of those calls, in the same order. ``SPMDDriver`` wraps the trainer on rank ``src`` and posts the
command before running it; ``follow`` is the loop on every other rank: receive a command, run the same method
(``step`` receives its batch from ``src`` through ``broadcast_batch``), until ``stop``. Anything else the runtime reads
(``cfg``, ``updates``, ``model``) passes through, and attribute writes (``updates`` on resume) reach the trainer.

Commands travel through the process group's **store** (the c10d ``TCPStore`` / ``FileStore`` the ranks rendezvoused
with), not through a collective: a follower polls for the next command key on the CPU while the driver does anything
that is not training (engine start-up, evaluation, rollouts), so no process-group timeout runs while it waits and no
GPU spins. The batch still rides the trainer's own broadcast.

Liveness (M12 audit H4 V5 / V6). Every rank of a ``Channel`` increments a heartbeat counter in the store from a
background thread (``period_s``); each rank watches the others (the driver all followers, a follower the driver):

- a peer whose counter stops moving for ``stale_s`` (or that never joins within ``join_timeout_s``), or a store that
  stops answering (the driver hosts the ``TCPStore``: its death closes it) is **lost**;
- a rank waiting for a command, or about to send one, raises ``LostPeer``;
- a rank blocked inside a training command (a collective a dead peer will never join) exits the process
  (``os._exit(75)``) after printing why, so the job ends and frees its GPUs instead of holding them until the
  process-group timeout or the allocation's walltime (``srun --kill-on-bad-exit=1`` then stops the other tasks).

Open the channel on every rank right after ``init_process_group`` (``open_channel()``): the driver then heartbeats
while it waits for engines and runs its "before" evaluation, and the followers know it is alive. (Before this, the
command group was a gloo ``new_group`` the followers created at once and the driver only after engine start-up and
evaluation; its rendezvous waits only the default process-group timeout, so followers died whenever start-up took
longer.) Defaults come from ``CRUCIBLE_SPMD_PERIOD_S`` (2), ``CRUCIBLE_SPMD_STALE_S`` (120) and
``CRUCIBLE_SPMD_JOIN_S`` (3600).
"""

from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

from crucible.core import env as knobs
from crucible.core.schema.rollout import RolloutRecord

COMMANDS = ("step", "state_dict", "reset_optimizer", "offload", "onload", "save_sharded", "load_sharded", "load_state", "stop")
EXIT_LOST = 75  # the exit code of a rank that lost a peer while blocked in a command

_SESSIONS: dict[tuple, int] = {}
_LOCK = threading.Lock()


class LostPeer(RuntimeError):
    """A peer of the SPMD channel stopped heartbeating or the store stopped answering."""


def command_group(group: Any = None, *, timeout_s: float = 7 * 24 * 3600) -> Any:
    """A gloo group over the trainer group's ranks (a collective: every rank calls it, in the same order). Kept for
    callers that want a CPU group; the SPMD channel itself no longer uses one."""
    from datetime import timedelta

    import torch.distributed as dist

    ranks = None if group is None else dist.get_process_group_ranks(group)
    return dist.new_group(ranks=ranks, backend="gloo", timeout=timedelta(seconds=timeout_s))


class Channel:
    """The driver → followers command channel of one SPMD session, with heartbeats. Not a collective: each rank creates
    its own; sessions pair up by creation order (the n-th channel of a rank group on every rank)."""

    def __init__(self, group: Any = None, src: int = 0, *, store: Any = None, period_s: float | None = None, stale_s: float | None = None,
                 join_timeout_s: float | None = None):
        import torch.distributed as dist

        ranks = list(dist.get_process_group_ranks(group)) if group is not None else list(range(dist.get_world_size()))
        self.me, self.driver = dist.get_rank(), ranks[src]
        self.is_driver = self.me == self.driver
        self.peers = [r for r in ranks if r != self.me] if self.is_driver else [self.driver]
        self.store = store if store is not None else dist.distributed_c10d._get_default_store()
        with _LOCK:
            _SESSIONS[tuple(ranks)] = n = _SESSIONS.get(tuple(ranks), 0) + 1
        self.prefix = f"crucible/spmd/{'-'.join(map(str, ranks))}/{n}"
        self.period_s = period_s if period_s is not None else knobs.get("CRUCIBLE_SPMD_PERIOD_S")
        self.stale_s = stale_s if stale_s is not None else knobs.get("CRUCIBLE_SPMD_STALE_S")
        self.join_timeout_s = join_timeout_s if join_timeout_s is not None else knobs.get("CRUCIBLE_SPMD_JOIN_S")
        self.seq = 0
        self.busy: str | None = None  # the command this rank is executing (blocked in its collectives)
        self.lost: str | None = None
        self._stop = threading.Event()
        self._seen: dict[int, tuple[int, float]] = {}  # peer -> (last counter, local time it last moved)
        self._t0 = time.monotonic()
        self.store.add(self._hb(self.me), 1)
        self._thread = threading.Thread(target=self._watch, name="crucible-spmd-heartbeat", daemon=True)
        self._thread.start()

    def _hb(self, rank: int) -> str:
        return f"{self.prefix}/hb/{rank}"

    # ---- heartbeat and watchdog ------------------------------------------------------------------------------------
    def _watch(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.store.add(self._hb(self.me), 1)
                now = time.monotonic()
                for p in self.peers:
                    c = int(self.store.add(self._hb(p), 0))
                    last = self._seen.get(p)
                    if last is None or c != last[0]:
                        self._seen[p] = (c, now)
                    elif c > 0 and now - last[1] > self.stale_s:
                        self._lose(f"rank {p} stopped heartbeating {now - last[1]:.0f}s ago")
                    if c == 0 and now - self._t0 > self.join_timeout_s:
                        self._lose(f"rank {p} never joined the SPMD channel within {self.join_timeout_s:.0f}s")
            except Exception as e:  # noqa: BLE001 -- the store is gone: its host (the driver's process) died
                if not self._stop.is_set():
                    self._lose(f"the rendezvous store stopped answering ({type(e).__name__}: {e})")

    def _lose(self, why: str) -> None:
        if self.lost is not None:
            return
        self.lost = why
        if self.busy is not None:  # blocked in a collective the lost peer will never join: free the GPUs now
            print(f"crucible.spmd: rank {self.me} lost a peer during {self.busy!r}: {why}; exiting ({EXIT_LOST})", file=sys.stderr, flush=True)
            os._exit(EXIT_LOST)
        self._stop.set()

    def check(self) -> None:
        if self.lost is not None:
            raise LostPeer(f"rank {self.me}: {self.lost}")

    def close(self) -> None:
        self._stop.set()

    # ---- commands --------------------------------------------------------------------------------------------------
    def send(self, cmd: str, args: list) -> None:
        self.check()
        self.seq += 1
        self.store.set(f"{self.prefix}/cmd/{self.seq}", pickle.dumps([cmd, args]))

    def receive(self, poll_s: float = 0.02) -> tuple[str, list]:
        self.seq += 1
        key = f"{self.prefix}/cmd/{self.seq}"
        wait = poll_s
        while True:
            self.check()
            try:
                if self.store.check([key]):
                    cmd, args = pickle.loads(self.store.get(key))
                    return cmd, args
            except Exception as e:  # noqa: BLE001
                if self.lost is None:
                    self._lose(f"the rendezvous store stopped answering ({type(e).__name__}: {e})")
                self.check()
            time.sleep(wait)
            wait = min(wait * 1.5, 0.25)


def open_channel(group: Any = None, src: int = 0, **kw: Any) -> Channel:
    """Call on every trainer rank right after ``init_process_group``; pass the result to ``SPMDDriver`` / ``follow``."""
    return Channel(group, src, **kw)


class SPMDDriver:
    def __init__(self, trainer: Any, *, group: Any = None, src: int = 0, channel: Channel | None = None, **live: Any):
        if not getattr(trainer, "broadcast_batch", False):
            raise ValueError("SPMDDriver needs the trainer built with broadcast_batch=True (the followers receive the batch)")
        ch = channel if channel is not None else Channel(group, src, **live)
        if not ch.is_driver:
            raise ValueError(f"rank {ch.me} is not the channel's driver (rank {ch.driver})")
        self.__dict__.update(trainer=trainer, _ch=ch, calls={c: 0 for c in COMMANDS})

    def _run(self, cmd: str, *args: Any, local: Any = None) -> Any:
        ch = self._ch
        ch.send(cmd, list(args))
        self.calls[cmd] += 1
        if cmd == "stop":
            ch.close()
            return None
        ch.busy = cmd
        try:
            return local() if local is not None else _apply(self.trainer, cmd, list(args))
        finally:
            ch.busy = None

    def step(self, records: Sequence[RolloutRecord], groups: Sequence[str]) -> dict[str, float]:
        return self._run("step", local=lambda: self.trainer.step(records, groups))

    def state_dict(self) -> dict:
        return self._run("state_dict")

    def reset_optimizer(self) -> None:
        self._run("reset_optimizer")

    def offload(self, mode: str = "cpu", *, path: str | None = None) -> None:
        self._run("offload", mode, path)  # H4 V13: the path reaches every rank (it was dropped)

    def onload(self) -> None:
        self._run("onload")

    def save_sharded(self, path: str) -> None:
        self._run("save_sharded", path)

    def load_sharded(self, path: str) -> None:
        self._run("load_sharded", path)

    def load_state(self, path: str, blob: dict | None = None) -> None:
        """Resume (``state.restore``): every rank loads the trainer part of the run checkpoint at ``path`` (a directory
        the ranks share): weights and optimizer for a replicated trainer, the shards for a sharded one, the critic, the
        counters. The driver uses ``blob`` (already loaded and verified) when given. (H4 V3: only the driver loaded a
        replicated trainer, so the replicas diverged after the first resumed step.)"""
        from crucible.train import state

        local = (lambda: state.load_trainer_state(self.trainer, path)) if blob is None else \
            (lambda: state.apply_trainer_state(self.trainer, blob["trainer"], path))
        self._run("load_state", str(path), local=local)

    def stop(self) -> None:
        self._run("stop")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.trainer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            self.__dict__[name] = value
        else:  # H4 V3: ``tr.updates = …`` on the driver set a shadow attribute here; the trainer kept counting from 0
            setattr(self.trainer, name, value)


def _apply(trainer: Any, cmd: str, args: list) -> Any:
    if cmd == "step":
        return trainer.step([], [])
    if cmd == "offload":
        mode, path = (list(args) + [None])[:2]
        return trainer.offload(mode, path=path)
    if cmd == "load_state":
        from crucible.train import state

        return state.load_trainer_state(trainer, args[0])
    return getattr(trainer, cmd)(*args)


def follow(trainer: Any, *, group: Any = None, src: int = 0, channel: Channel | None = None, **live: Any) -> dict[str, int]:
    """The loop on a non-driver rank: run each command until ``stop``; returns the command counts. Raises ``LostPeer``
    when the driver is lost while this rank waits."""
    ch = channel if channel is not None else Channel(group, src, **live)
    if ch.is_driver:
        raise ValueError(f"rank {ch.me} is the channel's driver: wrap the trainer in SPMDDriver instead")
    counts = {c: 0 for c in COMMANDS}
    try:
        while True:
            cmd, args = ch.receive()
            counts[cmd] += 1
            if cmd == "stop":
                return counts
            ch.busy = cmd
            try:
                _apply(trainer, cmd, args)
            finally:
                ch.busy = None
    finally:
        ch.close()
