"""``crucible label``: workflows → traces with dense rewards, plus SFT / DPO / KTO data (ARCHITECTURE §4 M15).

Two sources:
- **a pack** (``--pack``). This runs M3 ingest over the pack's ``ingest`` sources, which is redacted and idempotent. Then:
  1. M4 signals and pivots (when the family has a signal mapper);
  2. the pack's ``credit`` hook (M5 milestones → M6 Φ per event and a gated reward);
  3. the conversation from the policy role's side (``credit.datasets.from_trace``).
- **a run** (``--run``): an M10 run directory. Scored episodes become datasets, masked episodes are skipped, and DPO
  pairs come from best vs worst of k on the same task.

Outputs in ``out/``:
- ``configs/resolved/label.json``;
- ``store/`` (the traces);
- ``traces.jsonl`` (the labelled traces: signals, outcome);
- ``rewards.jsonl`` (per trace: reward, gates, Φ per event);
- ``sft.jsonl``, ``dpo.jsonl``, ``kto.jsonl``;
- ``summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.credit import datasets as ds


def ingest(pack_dir: Path, manifest: Any, store_dir: Path) -> list[Any]:
    from crucible.connectors.base import ingest as m3_ingest
    from crucible.connectors.file_drop import FieldMap, FileDropConnector
    from crucible.connectors.trace_store import TraceStore
    from crucible.core.store import Store

    ts = TraceStore(Store(Path(store_dir)))
    keys: list[str] = []
    for spec in manifest.ingest:
        conn = FileDropConnector(FieldMap(family=spec.family, extra=list(spec.extra)))
        refs = list(conn.discover({"glob": str(Path(pack_dir) / spec.glob)}))
        keys += m3_ingest(conn, refs, ts)
    ids = sorted({i for i in ts.ids() if any(i.startswith(s.family + ":") for s in manifest.ingest)})
    return [ts.get(i) for i in ids]


def _resolved(out: Path, cmd: str, cfg: dict[str, Any]) -> None:
    p = out / "configs" / "resolved" / f"{cmd}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1, sort_keys=True, default=str) + "\n")


async def label(*, out: Path, pack: str | Path | None = None, run: Path | None = None, filter: str | None = None, select: str = "correct",
                max_per_task: int | None = None, threshold: float | None = None, margin: float = 0.25, mask_negative: bool = False,
                dry_run: bool = False) -> dict[str, Any]:
    if (pack is None) == (run is None):
        raise ValueError("label needs exactly one source: --pack <pack> or --run <run dir>")
    out = Path(out)
    ds.compile_filter(filter)  # a bad filter fails before any work
    cfg = {"pack": str(pack) if pack else None, "run": str(run) if run else None, "filter": filter, "select": select, "max_per_task": max_per_task,
           "threshold": threshold, "margin": margin, "mask_negative": mask_negative}
    summary: dict[str, Any] = {"command": "label", "config": cfg}
    if pack is not None:
        from crucible.signals.extract import extract, mapper_for
        from crucible.studio.pack import find_pack, load_manifest, resolve

        pack_dir = find_pack(pack)
        m = load_manifest(pack_dir)
        thr = m.label.threshold if threshold is None else threshold
        if not m.ingest:
            raise ValueError(f"pack {m.name} declares no ingest sources: nothing to label (add `ingest:` to pack.yaml)")
        if dry_run:
            files = sorted(str(p) for s in m.ingest for p in pack_dir.glob(s.glob))
            return {**summary, "dry_run": True, "would_ingest": files, "credit_hook": m.hooks.credit}
        out.mkdir(parents=True, exist_ok=True)
        _resolved(out, "label", cfg | {"pack_dir": str(pack_dir), "threshold": thr})
        traces = ingest(pack_dir, m, out / "store")
        labelled_traces = []
        sig_counts: dict[str, int] = {}
        for t in traces:
            if mapper_for(t.family) is not None:
                t = extract(t)
                for s in t.signals:
                    sig_counts[s.type] = sig_counts.get(s.type, 0) + 1
            labelled_traces.append(t)
        credit = await resolve(pack_dir, m.hooks.credit)(labelled_traces) if m.hooks.credit else {}
        for i, t in enumerate(labelled_traces):  # a pack's outcome rule fills in what no registered labeler decided
            lab = (credit.get(t.trace_id) or {}).get("outcome")
            if lab in ("good", "bad") and t.outcome.label == "unknown":
                labelled_traces[i] = t.model_copy(update={"outcome": t.outcome.model_copy(update={"label": lab, "label_source": f"pack {m.name} credit hook"})})
        items = [ds.from_trace(t, policy_roles=m.label.policy_roles, phi_by_seq=(credit.get(t.trace_id) or {}).get("phi_by_seq"),
                               reward=(credit.get(t.trace_id) or {}).get("reward")) for t in labelled_traces]
        (out / "traces.jsonl").write_text("".join(t.model_dump_json() + "\n" for t in labelled_traces))
        ds.write_jsonl([{"trace_id": t.trace_id, "outcome": t.outcome.label, **{k: v for k, v in (credit.get(t.trace_id) or {}).items()}}
                        for t in labelled_traces], out / "rewards.jsonl")
        summary |= {"pack": m.name, "traces": len(traces), "signals": dict(sorted(sig_counts.items())), "credited": len(credit),
                    "outcomes": {k: sum(t.outcome.label == k for t in labelled_traces) for k in ("good", "bad", "unknown")},
                    "pivots": {"pairs": sum(len(x.pivots) for x in items), "without_policy_fix": sum(x.pivots_unpaired for x in items)}}
    else:
        thr = 1.0 if threshold is None else threshold
        items = ds.from_run(Path(run))
        if dry_run:
            return {**summary, "dry_run": True, "episodes": len(items)}
        out.mkdir(parents=True, exist_ok=True)
        _resolved(out, "label", cfg | {"threshold": thr})
        summary |= {"run": str(run), "episodes": len(items)}
    sft = ds.sft_rows(ds.curate(items, filter=filter, select=select, max_per_task=max_per_task, threshold=thr), threshold=thr, mask_negative=mask_negative)
    dpo = ds.dpo_best_vs_worst(items, threshold=thr, margin=margin) + ds.dpo_pivots(items)
    kto = ds.kto_rows(items, threshold=thr)
    summary["datasets"] = {"sft": ds.write_jsonl(sft, out / "sft.jsonl"), "dpo": ds.write_jsonl(dpo, out / "dpo.jsonl"), "kto": ds.write_jsonl(kto, out / "kto.jsonl"),
                           "dpo_kinds": {k: sum(r["kind"] == k for r in dpo) for k in ("best_vs_worst", "correction_pivot")}}
    summary["files"] = {k: str(out / f) for k, f in (("sft", "sft.jsonl"), ("dpo", "dpo.jsonl"), ("kto", "kto.jsonl"), ("summary", "summary.json"))}
    summary["next"] = [f"python -m crucible.train.launch <job.toml>  # train on {out / 'sft.jsonl'}", f"crucible build {pack or '<pack>'}"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    return summary
