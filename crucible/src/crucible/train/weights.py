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

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class InprocPublisher:
    def __init__(self, trainer: Any, engines: Any, *, post_update_transform: Callable[[dict], dict] | None = None, reset_optimizer_on_publish: bool = False):
        self.trainer, self.engines, self.transform, self.reset = trainer, engines, post_update_transform, reset_optimizer_on_publish
        self.version = 0
        self.latencies: list[float] = []
        # the trainer's update count whose weights the engines serve as ``version`` (the state plane checks a checkpoint
        # against it: M12 audit H2 V12); None before the first publish
        self.published_updates: int | None = None

    def _updates(self) -> int | None:
        u = getattr(self.trainer, "updates", None)
        return int(u) if isinstance(u, int | float) else None

    async def publish(self) -> int:
        t0 = time.monotonic()
        state = self.trainer.state_dict()
        if self.transform is not None:
            state = self.transform(state)
        self.version += 1
        self.published_updates = self._updates()
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
        self.published_updates = self._updates()
        await self.engines.load_weights(state, version)


class CheckpointPublisher(InprocPublisher):
    """``keep``: version directories kept on disk (default 2: the version being served and the one before, which an
    engine that restarts or reloads late may still read). Older ones are deleted only after every engine confirmed
    the newer version (``load_weights`` returned), never one that a load in flight may still be reading (M12 audit G5
    D8: ``v{n}/`` was never deleted, 16-33 GB per publish for an 8B model). ``keep=0`` keeps everything."""

    def __init__(self, trainer: Any, engines: Any, root: Path, *, keep: int = 2, **kw: Any):
        super().__init__(trainer, engines, **kw)
        self.root = Path(root)
        if keep < 0:
            raise ValueError("keep >= 0 (0 keeps every version)")
        self.keep = keep
        self._loading: set[Path] = set()  # version dirs an engine may be reading right now
        self.deleted: list[str] = []

    def _write(self, state: dict, version: int) -> Path:
        from safetensors.torch import save_file

        d = self.root / f"v{version}"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "model.safetensors.tmp"
        save_file({k: v.contiguous() for k, v in state.items()}, str(tmp))
        os.replace(tmp, d / "model.safetensors")  # an engine never sees a half-written file
        return d

    async def _serve(self, d: Path, version: int) -> None:
        self._loading.add(d)
        await self.engines.load_weights(str(d), version)  # a failed or cancelled load raises here: nothing is collected
        self._loading.discard(d)
        self._collect(version)

    def _collect(self, current: int) -> None:
        if not self.keep:
            return
        import re
        import shutil

        for d in self.root.glob("v*"):
            m = re.fullmatch(r"v(\d+)", d.name)
            if not m or not d.is_dir() or d in self._loading:
                continue
            if int(m.group(1)) <= current - self.keep:
                shutil.rmtree(d, ignore_errors=True)
                self.deleted.append(d.name)

    async def publish(self) -> int:
        t0 = time.monotonic()
        state = self.trainer.state_dict()
        if self.transform is not None:
            state = self.transform(state)
        self.version += 1
        self.published_updates = self._updates()
        d = self._write(state, self.version)
        await self._serve(d, self.version)
        if self.reset:
            self.trainer.reset_optimizer()
        self.latencies.append(time.monotonic() - t0)
        return self.version

    async def restore(self, version: int) -> None:
        state = self.trainer.state_dict()
        if self.transform is not None:
            state = self.transform(state)
        self.version = version
        self.published_updates = self._updates()
        await self._serve(self._write(state, version), version)


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
    d = (mag.unsqueeze(-1) - grid).abs()
    # round half to even: on an exact tie the code with LSB 0 (even index) wins (0.75 → 1, 1.75 → 2, 3.5 → 4; OCP MX spec)
    odd = (torch.arange(len(FP4_E2M1), device=v.device) % 2).to(v.dtype)
    best = d.min(dim=-1, keepdim=True).values
    q = grid[torch.where(d <= best, odd, torch.full_like(d, 2.0)).argmin(dim=-1)] * v.sign()
    out = (q * scale).reshape(flat.shape[0], -1)[:, :n]
    return out.reshape(shape).to(t.dtype)


class QDQ:
    """A ``post_update_transform``: QDQ every parameter whose name matches ``include`` (fnmatch; default the MoE expert
    weights, which is what MiMo quantizes for its MXFP4 rollout kernels) before the engines get them.

    With ``model`` set, the trainer computes on the same grid as the engine (MiMo: bit-identical expert weights on both
    sides) while an unquantized fp32 **master** copy accumulates the updates, as Megatron's master weights do. At every
    publish: master += (parameter − the grid values it was last set to), i.e. the optimizer's update since then; the
    parameter is then set to QDQ(master), and the engine receives exactly those values. M12 audit: the grid was written
    into the master itself, so an update smaller than half a grid step was rounded away (49 of 28,672 elements ever moved
    at lr 2e-6), and HF policies were never matched at all (their parameter names carry a ``model.`` prefix)."""

    def __init__(self, kind: str = "mxfp4", *, include: tuple[str, ...] = ("*experts*",), exclude: tuple[str, ...] = ("*norm*", "*bias*", "*embed*"),
                 model: Any = None, block: int = 32):
        if kind not in ("mxfp4", "fp8"):
            raise ValueError("QDQ kind: mxfp4 | fp8")
        self.kind, self.include, self.exclude, self.model, self.block = kind, include, exclude, model, block
        self.applied: list[str] = []
        self.master: dict[str, Any] = {}  # name -> fp32 master (model mode)
        self._last: dict[str, Any] = {}  # name -> the grid values the parameter was last set to

    def selects(self, name: str) -> bool:
        import fnmatch

        return any(fnmatch.fnmatchcase(name, p) for p in self.include) and not any(fnmatch.fnmatchcase(name, p) for p in self.exclude)

    def _q(self, t: Any) -> Any:
        return qdq_mxfp4(t, block=self.block) if self.kind == "mxfp4" else qdq_fp8_e4m3(t)

    def _params(self) -> dict[str, Any]:
        """The trainer's parameters under the published (state-dict) names: a wrapper's prefix (``HFPolicy.model``) stripped."""
        out: dict[str, Any] = {}
        for n, p in self.model.named_parameters():
            out.setdefault(n, p)
            for pre in ("model.", "module.", "_fsdp_wrapped_module."):
                if n.startswith(pre):
                    out.setdefault(n[len(pre):], p)
        return out

    def __call__(self, state: dict) -> dict:
        import torch

        out, self.applied = {}, []
        params = self._params() if self.model is not None else {}
        for name, t in state.items():
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dim() >= 1 and self.selects(name):
                prm = params.get(name)
                if prm is not None:
                    with torch.no_grad():
                        cur = prm.detach().float()
                        if name not in self.master:
                            self.master[name] = cur.clone()
                        else:  # the optimizer's update since the last publish, applied to the master
                            self.master[name] += cur - self._last[name]
                        q = self._q(self.master[name]).to(prm.dtype)
                        prm.data.copy_(q)
                        self._last[name] = q.float()
                    out[name] = q.to(t.dtype).to(t.device)
                else:
                    if self.model is not None:
                        raise KeyError(f"QDQ(model=...): published tensor {name!r} has no trainer parameter (the grids would differ)")
                    out[name] = self._q(t)
                self.applied.append(name)
            else:
                out[name] = t
        return out
