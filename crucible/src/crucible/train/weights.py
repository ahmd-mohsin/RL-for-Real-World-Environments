"""Weight publishing (research m11-m12/06 §4.1, §4.4; m11-m12/04 checkpoint-engine, TRL; m11-m12/01 verl/slime).

``publish()`` makes the trainer's current weights servable and returns the new **policy version** (+1 per publish,
exactly). The version travels with every engine response; the trainer never infers it from time.

- ``InprocPublisher``: same process / same device (colocated): hand the state dict to every engine's
  ``load_weights`` (verl/MiMo sleep → update → wake; TRL colocate ``load_weights``). Engines must be paused or drained
  by the caller (colocated: aborted at the sample barrier; disaggregated: engines swap at call boundaries).
- ``CheckpointPublisher``: write ``safetensors`` to a versioned directory and ask each engine to reload it (an HTTP
  engine's ``/update_weights_from_disk``-style endpoint). The slow, universal path; NCCL broadcast and checkpoint-engine
  style bucketed P2P are the scale-out path (pattern now, dependency when M12 goes multi-node).
- ``post_update_transform``: optional hook before publish (MiMo QDQ to the rollout quantization grid); identity by default.
- ``reset_optimizer_on_publish``: GLM's optimizer reset after every weight push.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class InprocPublisher:
    def __init__(self, trainer: Any, engines: Any, *, post_update_transform: Callable[[dict], dict] | None = None, reset_optimizer_on_publish: bool = False):
        self.trainer, self.engines, self.transform, self.reset = trainer, engines, post_update_transform, reset_optimizer_on_publish
        self.version = 0
        self.latencies: list[float] = []

    async def publish(self) -> int:
        t0 = time.monotonic()
        state = self.trainer.state_dict()
        if self.transform is not None:
            state = self.transform(state)
        self.version += 1
        await self.engines.load_weights(state, self.version)
        if self.reset:
            self.trainer.reset_optimizer()
        self.latencies.append(time.monotonic() - t0)
        return self.version


    async def restore(self, version: int) -> None:
        """Resume (state plane): serve the restored weights at the restored ``version`` (no new version, no optimizer reset)."""
        state = self.trainer.state_dict()
        if self.transform is not None:
            state = self.transform(state)
        self.version = version
        await self.engines.load_weights(state, version)


class CheckpointPublisher(InprocPublisher):
    def __init__(self, trainer: Any, engines: Any, root: Path, **kw: Any):
        super().__init__(trainer, engines, **kw)
        self.root = Path(root)

    async def publish(self) -> int:
        from safetensors.torch import save_file

        t0 = time.monotonic()
        state = self.trainer.state_dict()
        if self.transform is not None:
            state = self.transform(state)
        self.version += 1
        d = self.root / f"v{self.version}"
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in state.items()}, str(d / "model.safetensors"))
        await self.engines.load_weights(str(d), self.version)
        if self.reset:
            self.trainer.reset_optimizer()
        self.latencies.append(time.monotonic() - t0)
        return self.version

    async def restore(self, version: int) -> None:
        from safetensors.torch import save_file

        state = self.trainer.state_dict()
        if self.transform is not None:
            state = self.transform(state)
        self.version = version
        d = self.root / f"v{version}"
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in state.items()}, str(d / "model.safetensors"))
        await self.engines.load_weights(str(d), version)
