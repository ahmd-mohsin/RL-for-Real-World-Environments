"""Single controller over an SPMD trainer (M12-infra C; art. §2 "controller": one driver process owns the loop, N
trainer ranks run the same program).

The ``Runtime`` is one asyncio process: it calls ``trainer.step`` / ``state_dict`` / ``reset_optimizer`` /
``offload`` / ``onload`` on rank ``src``. A sharded trainer (``FSDPTrainer``, ``DistributedTorchTrainer``) needs every
rank inside each of those calls, in the same order. ``SPMDDriver`` wraps the trainer on rank ``src`` and broadcasts
the command before running it; ``follow`` is the loop on every other rank: receive a command, run the same method
(``step`` receives its batch from ``src`` through ``broadcast_batch``), until ``stop``. Anything else the runtime reads
(``cfg``, ``updates``, ``model``) passes through.

Commands are small pickled lists on a dedicated **gloo** group with a long timeout (``command_group``): followers wait
there, on the CPU, while the driver does anything that is not training (rollouts, evaluation). Waiting inside an NCCL
collective instead would spin a GPU and trip NCCL's 10-minute watchdog whenever the driver's non-training work runs long.
The batch still rides the trainer's own broadcast, so the driver adds one tiny CPU collective per call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crucible.core.schema.rollout import RolloutRecord

COMMANDS = ("step", "state_dict", "reset_optimizer", "offload", "onload", "save_sharded", "load_sharded", "stop")


def command_group(group: Any = None, *, timeout_s: float = 7 * 24 * 3600) -> Any:
    """A gloo group over the trainer group's ranks for commands (a collective: every rank calls it, in the same order)."""
    from datetime import timedelta

    import torch.distributed as dist

    ranks = None if group is None else dist.get_process_group_ranks(group)
    return dist.new_group(ranks=ranks, backend="gloo", timeout=timedelta(seconds=timeout_s))


class SPMDDriver:
    def __init__(self, trainer: Any, *, group: Any = None, src: int = 0):
        import torch.distributed as dist

        if not getattr(trainer, "broadcast_batch", False):
            raise ValueError("SPMDDriver needs the trainer built with broadcast_batch=True (the followers receive the batch)")
        self.__dict__.update(trainer=trainer, _dist=dist, _group=command_group(group), _src=src, calls={c: 0 for c in COMMANDS})

    def _send(self, cmd: str, *args: Any) -> None:
        self._dist.broadcast_object_list([cmd, list(args)], src=self._src, group=self._group)
        self.calls[cmd] += 1

    def step(self, records: Sequence[RolloutRecord], groups: Sequence[str]) -> dict[str, float]:
        self._send("step")
        return self.trainer.step(records, groups)

    def state_dict(self) -> dict:
        self._send("state_dict")
        return self.trainer.state_dict()

    def reset_optimizer(self) -> None:
        self._send("reset_optimizer")
        self.trainer.reset_optimizer()

    def offload(self, mode: str = "cpu", *, path: str | None = None) -> None:
        self._send("offload", mode)
        self.trainer.offload(mode)

    def onload(self) -> None:
        self._send("onload")
        self.trainer.onload()

    def save_sharded(self, path: str) -> None:
        self._send("save_sharded", path)
        self.trainer.save_sharded(path)

    def load_sharded(self, path: str) -> None:
        self._send("load_sharded", path)
        self.trainer.load_sharded(path)

    def stop(self) -> None:
        self._send("stop")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.trainer, name)


def follow(trainer: Any, *, group: Any = None, src: int = 0) -> dict[str, int]:
    """The loop on a non-driver rank: run each broadcast command until ``stop``; returns the command counts."""
    import torch.distributed as dist

    group = command_group(group)
    counts = {c: 0 for c in COMMANDS}
    while True:
        box: list[Any] = [None, None]
        dist.broadcast_object_list(box, src=src, group=group)
        cmd, args = box
        counts[cmd] += 1
        if cmd == "stop":
            return counts
        if cmd == "step":
            trainer.step([], [])
        elif cmd == "offload":
            trainer.offload(*args)
        else:
            getattr(trainer, cmd)(*args)
