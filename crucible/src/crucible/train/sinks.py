"""Metric and record sinks (m13 C-1; article A2-56): where a run's step metrics, trained rollout records, failures and
final summary go.

The pattern is verl / SkyRL ``Tracking`` (one object fanning out to backends: console, TensorBoard, W&B, ...), with
two changes for long agentic runs:

- **failure isolation**: a backend that raises (W&B ``BrokenPipeError``, a full disk) is disabled with one warning and
  counted in ``Tracking.failed``; it never stops training (verl only guards ``finish``);
- **records, not just scalars**: ``crucible.rollout/v1`` records and failure examples stream to record sinks
  (JSONL), so a run's training data and its failure taxonomy survive the process, not only ``RunReport``.

Sinks (``from_spec``: ``"jsonl:DIR,tensorboard:DIR,wandb:PROJECT,console"``):

- ``JsonlSink(dir)``: ``metrics.jsonl`` (one line per step), ``records.jsonl`` (``{"kind", "step", "data"}``),
  ``summary.json`` (rewritten atomically); line-buffered, so a killed run keeps everything up to its last step;
- ``TensorBoardSink(dir)``: numeric scalars through ``torch.utils.tensorboard`` (optional dependency);
- ``WandbSink(project, name, mode)``: ``mode="offline"`` by default (no network unless asked); optional dependency;
- ``ConsoleSink``: one compact line per step.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class Sink(Protocol):
    def log(self, metrics: Mapping[str, float], step: int) -> None: ...

    def record(self, kind: str, data: Any, step: int) -> None: ...

    def summary(self, data: Mapping[str, Any]) -> None: ...

    def finish(self, status: str) -> None: ...


def _scalars(metrics: Mapping[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, int | float) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def _jsonable(x: Any) -> Any:
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    return x


class JsonlSink:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._m = open(self.root / "metrics.jsonl", "a", buffering=1)  # noqa: SIM115 -- held for the run's lifetime
        self._r = open(self.root / "records.jsonl", "a", buffering=1)  # noqa: SIM115

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        self._m.write(json.dumps({"step": step, "t": time.time(), **_scalars(metrics)}) + "\n")

    def record(self, kind: str, data: Any, step: int) -> None:
        self._r.write(json.dumps({"kind": kind, "step": step, "data": _jsonable(data)}, default=str) + "\n")

    def summary(self, data: Mapping[str, Any]) -> None:
        tmp = self.root / ".summary.json.tmp"
        tmp.write_text(json.dumps(dict(data), indent=1, default=str))
        os.replace(tmp, self.root / "summary.json")

    def finish(self, status: str) -> None:
        self._m.close()
        self._r.close()


class TensorBoardSink:
    def __init__(self, root: str | Path):
        from torch.utils.tensorboard import SummaryWriter  # optional: needs the tensorboard package

        self.w = SummaryWriter(log_dir=str(root))

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        for k, v in _scalars(metrics).items():
            self.w.add_scalar(k, v, step)

    def record(self, kind: str, data: Any, step: int) -> None:
        pass  # TensorBoard holds scalars; records go to a JSONL sink

    def summary(self, data: Mapping[str, Any]) -> None:
        self.w.add_text("summary", json.dumps(dict(data), indent=1, default=str))

    def finish(self, status: str) -> None:
        self.w.flush()
        self.w.close()


class WandbSink:
    def __init__(self, project: str, name: str | None = None, *, mode: str = "offline", config: Mapping[str, Any] | None = None):
        import wandb  # optional dependency

        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, mode=mode, config=dict(config or {}))

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        self.run.log(_scalars(metrics), step=step)

    def record(self, kind: str, data: Any, step: int) -> None:
        if kind == "failure":  # rollout bodies are too large for W&B rows; failures are the actionable part
            self.run.log({"failures/last": json.dumps(_jsonable(data), default=str)[:2000]}, step=step)

    def summary(self, data: Mapping[str, Any]) -> None:
        self.run.summary.update({k: v for k, v in data.items() if isinstance(v, int | float | str | bool)})

    def finish(self, status: str) -> None:
        self.run.finish(exit_code=0 if status in ("steps", "ok") else 1)


class ConsoleSink:
    KEYS = ("reward/mean", "train/loss", "staleness/max", "mismatch/rollout_probs_diff", "untrained")

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        s = _scalars(metrics)
        log.info("step %d %s", step, " ".join(f"{k}={s[k]:.4g}" for k in self.KEYS if k in s))

    def record(self, kind: str, data: Any, step: int) -> None:
        pass

    def summary(self, data: Mapping[str, Any]) -> None:
        log.info("run summary %s", json.dumps({k: v for k, v in data.items() if isinstance(v, int | float | str)}, default=str)[:2000])

    def finish(self, status: str) -> None:
        pass


class Tracking:
    """Fan-out to sinks; a sink that raises is disabled (one warning) and the run continues."""

    def __init__(self, sinks: Iterable[Sink] = ()):
        self.sinks: list[Sink] = list(sinks)
        self.failed: dict[str, str] = {}
        self._finished = False

    def _each(self, method: str, *args: Any) -> None:
        for s in list(self.sinks):
            try:
                getattr(s, method)(*args)
            except Exception as e:  # noqa: BLE001 -- a metrics backend must never stop training
                name = type(s).__name__
                self.failed[name] = f"{method}: {type(e).__name__}: {e}"[:500]
                self.sinks.remove(s)
                log.warning("sink %s disabled after %s failed: %s", name, method, e)

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        self._each("log", metrics, step)

    def record(self, kind: str, data: Any, step: int) -> None:
        self._each("record", kind, data, step)

    def records(self, kind: str, items: Iterable[Any], step: int) -> None:
        for x in items:
            self.record(kind, x, step)

    def summary(self, data: Mapping[str, Any]) -> None:
        self._each("summary", data)

    def finish(self, status: str) -> None:
        if not self._finished:
            self._finished = True
            self._each("finish", status)


def from_spec(spec: str, *, run_name: str | None = None) -> list[Sink]:
    """``"jsonl:DIR,tensorboard:DIR,wandb:PROJECT[:online],console"`` → sinks."""
    out: list[Sink] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        kind, _, arg = part.partition(":")
        if kind == "jsonl":
            out.append(JsonlSink(arg or "runs/metrics"))
        elif kind == "tensorboard":
            out.append(TensorBoardSink(arg or "runs/tb"))
        elif kind == "wandb":
            project, _, mode = arg.partition(":")
            out.append(WandbSink(project or "crucible", run_name, mode=mode or "offline"))
        elif kind == "console":
            out.append(ConsoleSink())
        else:
            raise ValueError(f"unknown sink {kind!r} (jsonl | tensorboard | wandb | console)")
    return out
