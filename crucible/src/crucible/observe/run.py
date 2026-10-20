"""The one gateway to W&B (swarm-pipeline/11 §3): every Crucible producer logs through ``ObsRun``.

``ObsRun`` is a W&B run plus a local mirror:

- **W&B** (``wandb``, optional extra ``observe``): ``mode="offline"`` by default. An offline run is written under
  ``<root>/runs/<project>/<name>/wandb/offline-run-*`` and uploaded later with ``wandb sync`` (``scripts/observe.py
  sync``). ``online`` needs ``WANDB_API_KEY`` in the environment (checked for presence only; without it the run falls
  back to offline with one warning). ``disabled``: the mirror only. Console capture, code saving, git and system
  metadata, the requirements file and system stats are off: what W&B gets is what this module logs.
- **The mirror** (``ops.jsonl`` beside it): every operation as one JSON line (``log``, ``table``, ``chart``, ``html``,
  ``artifact``, ``summary``, ``finish``), after redaction. A run logged without ``wandb`` installed is still complete on
  disk, and ``replay`` turns its mirror into a W&B run later.

Every payload goes through ``observe.redact.Scrubber`` (credentials removed, e-mails and hosts pseudonymized); a planted
key never reaches W&B (``tests/unit/test_observe.py``). ``ObsRun`` also satisfies the ``train.sinks.Sink`` protocol
(``log``, ``record``, ``summary``, ``finish``), so it can be one of a ``Tracking``'s sinks: a failure of W&B never stops
the producer (``Tracking`` disables the sink with one warning).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from crucible.observe.naming import digest, entity, flat
from crucible.observe.redact import Scrubber, default_scrubber

log = logging.getLogger(__name__)
MODES = ("offline", "online", "disabled")
KEY_ENV = "WANDB_API_KEY"
RECORD_SKIP = {"rollout"}
RECORD_MAX = 10_000


def default_root() -> Path:
    """``CRUCIBLE_OBSERVE_DIR``, else ``<data dir>/observe`` (durable: never the session temp directory)."""
    if os.environ.get("CRUCIBLE_OBSERVE_DIR"):
        return Path(os.environ["CRUCIBLE_OBSERVE_DIR"])
    from crucible.core.config import get_config

    return get_config().data_path() / "observe"


def has_key() -> bool:
    """Whether a W&B key is in the environment (its value is never read here)."""
    return bool(os.environ.get(KEY_ENV))


@dataclass
class ObserveConfig:
    root: Path = field(default_factory=default_root)
    mode: str = "offline"
    entity: str | None = field(default_factory=entity)
    use_wandb: bool = True  # False: the mirror only, even when wandb is installed (tests, a machine without the extra)
    max_chars: int = 2000  # trace view text; the full text goes to the episode artifact

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.mode not in MODES:
            raise ValueError(f"mode is one of {MODES}")

    @classmethod
    def from_env(cls, root: Path | str | None = None, **kw: Any) -> ObserveConfig:
        """``CRUCIBLE_WANDB_MODE`` (offline | online | disabled; default offline); an explicit ``mode=`` wins."""
        mode = kw.pop("mode", None) or os.environ.get("CRUCIBLE_WANDB_MODE", "offline")
        return cls(**({"root": Path(root)} if root else {}), mode=mode, **kw)

    def effective_mode(self) -> str:
        if self.mode == "online" and not has_key():
            log.warning("CRUCIBLE_WANDB_MODE=online but %s is not set: logging offline (sync later)", KEY_ENV)
            return "offline"
        return self.mode


def _wandb() -> Any:
    try:
        import wandb  # optional extra ``observe``
    except ImportError:
        return None
    return wandb


def _finite(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))


def scalars(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Numeric leaves of a (nested) dict as ``a/b`` keys; booleans as 0/1; non-finite and non-numeric dropped."""
    out = {}
    for k, v in flat(metrics).items():
        if isinstance(v, bool):
            v = float(v)
        if _finite(v):
            out[k] = float(v)
    return out


def _cell(v: Any) -> Any:
    """A table cell W&B accepts: scalars and strings as they are, anything else as compact JSON."""
    if v is None or isinstance(v, str | bool | int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return json.dumps(v, sort_keys=True, default=str)


class ObsRun:
    """One W&B run (offline unless asked) with its local mirror. ``kind`` is for the mirror only."""

    def __init__(self, project: str, name: str, *, group: str | None = None, tags: Sequence[str] = (), config: Mapping[str, Any] | None = None,
                 job_type: str | None = None, notes: str | None = None, cfg: ObserveConfig | None = None, scrubber: Scrubber | None = None,
                 run_id: str | None = None):
        self.cfg = cfg or ObserveConfig.from_env()
        self.s = scrubber or default_scrubber()
        self.project, self.name, self.group = project, name, group
        self.tags = [self.s.text(t) for t in tags]
        self.config = self.s.value(json.loads(json.dumps(dict(config or {}), default=str)))
        self.job_type = job_type
        self.id = run_id or digest([project, name, time.time_ns()], 12)
        self.dir = self.cfg.root / "runs" / project / f"{name}-{self.id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._ops = open(self.dir / "ops.jsonl", "a", buffering=1)  # noqa: SIM115 -- held for the run's lifetime
        self._records: dict[str, list[dict[str, Any]]] = {}
        self._finished = False
        self.mode = self.cfg.effective_mode()
        meta = {"project": project, "name": name, "id": self.id, "group": group, "tags": self.tags, "job_type": job_type, "mode": self.mode,
                "entity": self.cfg.entity, "notes": self.s.text(notes) if notes else None}
        self._op("init", meta=meta, config=self.config)
        self.wb = None
        wandb = _wandb() if self.cfg.use_wandb and self.mode != "disabled" else None
        if wandb is not None:
            settings = wandb.Settings(console="off", save_code=False, disable_code=True, disable_git=True, disable_job_creation=True, x_disable_meta=True,
                                      x_save_requirements=False, x_disable_stats=True, silent=True)
            self.wb = wandb.init(project=project, entity=self.cfg.entity, name=name, group=group, tags=self.tags or None, config=self.config,
                                 job_type=job_type, notes=meta["notes"], dir=str(self.dir), mode=self.mode, id=self.id, settings=settings, reinit="create_new")
        self.wandb = wandb

    # ---- the mirror ----
    def _op(self, op: str, **kw: Any) -> None:
        self._ops.write(json.dumps({"op": op, "t": round(time.time(), 3), **kw}, sort_keys=True, default=str) + "\n")

    # ---- Sink protocol ----
    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        sc = scalars(self.s.value(dict(metrics)))
        if not sc:
            return
        self._op("log", step=step, data=sc)
        if self.wb is not None:
            self.wb.log(sc, step=step)

    def record(self, kind: str, data: Any, step: int) -> None:
        """Records collect into a ``records/<kind>`` table written at ``finish`` (drops, failures, episodes). Rollout bodies
        (``kind="rollout"``) are too large for W&B rows: they stay in the JSONL sink and the trace export."""
        if kind in RECORD_SKIP or len(self._records.get(kind, ())) >= RECORD_MAX:
            return
        row = data.model_dump(mode="json") if hasattr(data, "model_dump") else data
        row = self.s.value(json.loads(json.dumps(row, default=str)))
        self._records.setdefault(kind, []).append({"step": step, **(row if isinstance(row, dict) else {"value": row})})
        if kind == "failure" and self.wb is not None:
            self.wb.log({"failures/last": json.dumps(row, default=str)[:2000]}, step=step)

    def summary(self, data: Mapping[str, Any]) -> None:
        d = self.s.value(json.loads(json.dumps(dict(data), default=str)))
        keep = {k: v for k, v in flat(d).items() if v is None or isinstance(v, int | float | str | bool)}
        self._op("summary", data=keep)
        if self.wb is not None:
            self.wb.summary.update(keep)

    def finish(self, status: str = "ok") -> None:
        if self._finished:
            return
        self._finished = True
        for kind, rows in self._records.items():
            cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
            self.table(f"records/{kind}", cols, [[r.get(c) for c in cols] for r in rows])
        self._op("finish", status=status)
        self._ops.close()
        if self.wb is not None:
            self.wb.finish(exit_code=0 if status in ("ok", "done", "steps") else 1)

    # ---- richer panels ----
    def table(self, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], *, step: int | None = None) -> Any:
        cols = [str(c) for c in columns]
        data = [[_cell(self.s.value(v)) for v in r] for r in rows]
        self._op("table", key=key, step=step, columns=cols, rows=data)
        if self.wb is None:
            return None
        t = self.wandb.Table(columns=cols, data=data, allow_mixed_types=True)
        self.wb.log({key: t}, step=step)
        return t

    def bar(self, key: str, labels: Sequence[str], values: Sequence[float | None], *, title: str, label: str = "label", value: str = "value",
            step: int | None = None) -> None:
        """A native W&B bar chart (its table is logged with it)."""
        rows = [[str(lb), float(v)] for lb, v in zip(labels, values, strict=True) if _finite(v)]
        self._op("chart", kind="bar", key=key, title=title, columns=[label, value], rows=rows, step=step)
        if self.wb is not None:
            t = self.wandb.Table(columns=[label, value], data=rows)
            self.wb.log({key: self.wandb.plot.bar(t, label, value, title=title)}, step=step)

    def lines(self, key: str, xs: Sequence[float], ys: Sequence[Sequence[float | None]], names: Sequence[str], *, title: str, xname: str = "x",
              step: int | None = None) -> None:
        """A native W&B multi-line chart (``wandb.plot.line_series``); a None point is left out of its line."""
        series = [[float(y) if _finite(y) else None for y in line] for line in ys]
        self._op("chart", kind="lines", key=key, title=title, xs=list(xs), ys=series, names=list(names), xname=xname, step=step)
        if self.wb is not None and xs and series:
            pts = [[(x, y) for x, y in zip(xs, line, strict=False) if y is not None] for line in series]
            keep = [(n, p) for n, p in zip(names, pts, strict=True) if p]
            if keep:
                self.wb.log({key: self.wandb.plot.line_series(xs=[[x for x, _ in p] for _, p in keep], ys=[[y for _, y in p] for _, p in keep],
                                                              keys=[n for n, _ in keep], title=title, xname=xname)}, step=step)

    def html(self, key: str, html: str, *, step: int | None = None) -> Path:
        """An HTML panel (a chart with CIs, a trace view); also written to ``<run dir>/media/<key>.html`` for viewing offline."""
        html = self.s.text(html)
        p = self.dir / "media" / f"{key.replace('/', '__')}.html"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html)
        self._op("html", key=key, path=str(p.relative_to(self.dir)), step=step)
        if self.wb is not None:
            self.wb.log({key: self.wandb.Html(html, inject=False, data_is_not_path=True)}, step=step)
        return p

    def artifact(self, name: str, type: str, *, files: Mapping[str, Any] | None = None, paths: Mapping[str, Path | str] | None = None,
                 metadata: Mapping[str, Any] | None = None, aliases: Sequence[str] = ()) -> Path:
        """A W&B artifact. ``files``: name -> JSON-able content (redacted, written as JSON or text); ``paths``: name -> a
        local file, copied after redaction when it is text (a binary file is copied as it is: never pass secrets).
        ``metadata`` holds references W&B cannot fetch (``tinker://`` adapter paths)."""
        from crucible.observe.naming import slug

        aname = slug(name, 120)
        d = self.dir / "artifacts" / aname
        d.mkdir(parents=True, exist_ok=True)
        for fn, content in (files or {}).items():
            red = self.s.value(json.loads(json.dumps(content, default=str))) if not isinstance(content, str) else self.s.text(content)
            (d / fn).write_text(red if isinstance(red, str) else json.dumps(red, indent=1, sort_keys=True, default=str))
        for fn, src in (paths or {}).items():
            src = Path(src)
            try:
                (d / fn).write_text(self.s.text(src.read_text()))
            except UnicodeDecodeError:
                shutil.copyfile(src, d / fn)
        meta = self.s.value(json.loads(json.dumps(dict(metadata or {}), default=str)))
        self._op("artifact", name=aname, type=type, dir=str(d.relative_to(self.dir)), files=sorted(p.name for p in d.iterdir()), metadata=meta,
                 aliases=list(aliases))
        if self.wb is not None:
            art = self.wandb.Artifact(aname, type=type, metadata=meta)
            art.add_dir(str(d))
            self.wb.log_artifact(art, aliases=list(aliases) or None)
        return d

    @property
    def sync_dirs(self) -> list[Path]:
        return sorted((self.dir / "wandb").glob("offline-run-*")) if (self.dir / "wandb").exists() else []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, et: object, ev: object, tb: object) -> None:
        self.finish("ok" if et is None else "error")


class WandbSink(ObsRun):
    """``train.sinks.WandbSink``: the M12 trainer's ``wandb:PROJECT`` sink (``from_spec``), now an ``ObsRun``."""

    def __init__(self, project: str, name: str | None = None, *, mode: str = "offline", config: Mapping[str, Any] | None = None):
        super().__init__(project, name or "run", config=config, job_type="rl", cfg=ObserveConfig.from_env(mode=mode))


def mirror_runs(root: Path | str) -> list[Path]:
    """Every mirrored run directory under ``root/runs`` (those with an ``ops.jsonl``)."""
    return sorted(p.parent for p in Path(root).glob("runs/*/*/ops.jsonl"))


def replay(run_dir: Path | str, *, cfg: ObserveConfig | None = None) -> ObsRun:
    """A W&B run rebuilt from a mirror (a run logged where ``wandb`` was not installed). The mirror is already redacted."""
    run_dir = Path(run_dir)
    ops = [json.loads(x) for x in (run_dir / "ops.jsonl").read_text().splitlines() if x.strip()]
    init = next(o for o in ops if o["op"] == "init")
    m = init["meta"]
    cfg = cfg or ObserveConfig.from_env(root=run_dir.parents[2] / "replayed")
    r = ObsRun(m["project"], m["name"], group=m.get("group"), tags=m.get("tags") or (), config=init.get("config"), job_type=m.get("job_type"), cfg=cfg,
               run_id=m["id"])
    for o in ops:
        if o["op"] == "log":
            r.log(o["data"], o.get("step"))
        elif o["op"] == "table":
            r.table(o["key"], o["columns"], o["rows"], step=o.get("step"))
        elif o["op"] == "chart" and o.get("kind") == "bar":
            r.bar(o["key"], [x[0] for x in o["rows"]], [x[1] for x in o["rows"]], title=o["title"], label=o["columns"][0], value=o["columns"][1])
        elif o["op"] == "chart" and o.get("kind") == "lines":
            r.lines(o["key"], o["xs"], o["ys"], o["names"], title=o["title"], xname=o.get("xname", "x"))
        elif o["op"] == "html":
            r.html(o["key"], (run_dir / o["path"]).read_text())
        elif o["op"] == "artifact":
            src = run_dir / o["dir"]
            r.artifact(o["name"], o["type"], paths={p.name: p for p in sorted(src.iterdir())}, metadata=o.get("metadata"), aliases=o.get("aliases") or ())
        elif o["op"] == "summary":
            r.summary(o["data"])
    r.finish(next((o["status"] for o in ops if o["op"] == "finish"), "ok"))
    return r
