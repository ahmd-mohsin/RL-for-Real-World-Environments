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


# ---- QDQ to the rollout quantization grid (M12-infra B; art. §3.4.3.3: MiMo QDQ of expert weights to the MXFP4 kernel's
# grid after every update, Kimi K3 MXFP4 QAT) ---------------------------------------------------------------------------
FP4_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def qdq_fp8_e4m3(t: Any, *, per_row: bool = True) -> Any:
    """Quantize to FP8 E4M3 with an absmax scale (per row or per tensor) and dequantize back to the input dtype."""
    import torch

    x = t.float()
    amax = x.abs().amax(dim=-1, keepdim=True) if (per_row and x.dim() > 1) else x.abs().amax()
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    return ((x / scale).to(torch.float8_e4m3fn).float() * scale).to(t.dtype)


def qdq_mxfp4(t: Any, *, block: int = 32) -> Any:
    """OCP MX FP4: blocks of ``block`` values along the last dim share a power-of-two scale X = 2^(⌊log2 amax⌋ − 2);
    each value is rounded to the nearest E2M1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6} (saturating), then dequantized."""
    import torch

    x = t.float()
    shape = x.shape
    flat = x.reshape(-1, shape[-1]) if x.dim() > 1 else x.reshape(1, -1)
    n = flat.shape[-1]
    pad = (-n) % block
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    b = flat.reshape(flat.shape[0], -1, block)
    amax = b.abs().amax(dim=-1, keepdim=True)
    exp = torch.floor(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - 2
    scale = torch.pow(2.0, exp)
    v = b / scale
    grid = torch.tensor(FP4_E2M1, dtype=v.dtype, device=v.device)
    mag = v.abs().clamp(max=6.0)
    q = grid[(mag.unsqueeze(-1) - grid).abs().argmin(dim=-1)] * v.sign()
    out = (q * scale).reshape(flat.shape[0], -1)[:, :n]
    return out.reshape(shape).to(t.dtype)


class QDQ:
    """A ``post_update_transform``: QDQ every parameter whose name matches ``include`` (fnmatch; e.g. ``*experts*``).
    With ``model`` set, the trainer's own parameters are overwritten with the QDQ'd values too, so the trainer and the
    engine hold bit-identical weights (MiMo) instead of the engine alone seeing the grid."""

    def __init__(self, kind: str = "mxfp4", *, include: tuple[str, ...] = ("*",), exclude: tuple[str, ...] = ("*norm*", "*bias*", "*embed*"),
                 model: Any = None, block: int = 32):
        if kind not in ("mxfp4", "fp8"):
            raise ValueError("QDQ kind: mxfp4 | fp8")
        self.kind, self.include, self.exclude, self.model, self.block = kind, include, exclude, model, block
        self.applied: list[str] = []

    def selects(self, name: str) -> bool:
        import fnmatch

        return any(fnmatch.fnmatchcase(name, p) for p in self.include) and not any(fnmatch.fnmatchcase(name, p) for p in self.exclude)

    def _q(self, t: Any) -> Any:
        return qdq_mxfp4(t, block=self.block) if self.kind == "mxfp4" else qdq_fp8_e4m3(t)

    def __call__(self, state: dict) -> dict:
        import torch

        out, self.applied = {}, []
        params = dict(self.model.named_parameters()) if self.model is not None else {}
        for name, t in state.items():
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dim() >= 1 and self.selects(name):
                q = self._q(t)
                out[name] = q
                self.applied.append(name)
                if name in params:
                    with torch.no_grad():
                        params[name].data.copy_(q.to(params[name].dtype))
            else:
                out[name] = t
        return out
