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

from crucible.studio import datasets as ds


def connector_for(spec: Any, pack_dir: Path) -> Any:
    from crucible.connectors.file_drop import FieldMap, FileDropConnector
    from crucible.core.registry import get_registry

    if spec.connector == "file_drop":
        return FileDropConnector(FieldMap(family=spec.family, extra=list(spec.extra)))
    target = get_registry().get("connector", spec.connector)
    if isinstance(target, str):
        from crucible.studio.pack import resolve as _resolve

        target = _resolve(pack_dir, target)
    from crucible.studio.pack import ingest_value

    return target(**ingest_value(spec.init, pack_dir, init=True))


def ingest(pack_dir: Path, manifest: Any, store_dir: Path, *, labelled_only: bool = False, redaction: dict[str, Any] | None = None) -> list[Any]:
    """M3 over every ``ingest`` source of the pack (fetch, normalize, redact, store; idempotent). Returns the traces of
    those sources, by the keys the store returned (not by an id prefix: connectors name traces their own way). Every path
    is confined to the pack's allowed roots and a declared ``redaction_env`` must exist, or nothing is stored.
    ``redaction`` (when given) receives, per family, the dotenv used and how many values it loaded (never the values)."""
    from crucible.connectors.base import ingest as m3_ingest
    from crucible.connectors.redact import RedactionPolicy
    from crucible.connectors.trace_store import TraceStore
    from crucible.core.store import Store
    from crucible.studio.pack import ingest_problems, ingest_value

    specs = [s for s in manifest.ingest if s.label or not labelled_only]
    # refuse before anything is stored (M14 audit V5, V17): a missing redaction_env, a path outside the allowed roots
    problems = [p for s in specs for p in ingest_problems(Path(pack_dir), s)[0] if "no file matches" not in p]
    if problems:
        raise ValueError("ingest refused (nothing was stored): " + "; ".join(problems))
    ts = TraceStore(Store(Path(store_dir)))
    keys: list[str] = []
    for spec in specs:
        conn = connector_for(spec, pack_dir)
        if spec.connector == "file_drop":
            # every matched file is confined (a symlink out of the roots was refused above)
            refs = list(conn.discover({"glob": str((Path(pack_dir) / (spec.root or ".")) / spec.glob)}))
        else:
            refs = list(getattr(conn, spec.discover)(ingest_value(spec.scope, pack_dir)) if spec.scope or spec.discover != "discover" else conn.discover())
        policy = (RedactionPolicy.with_env_file(Path(pack_dir) / spec.redaction_env, missing_ok=spec.redaction_env_optional) if spec.redaction_env
                  else RedactionPolicy())
        if redaction is not None:
            redaction[spec.family] = {"env": spec.redaction_env, "values": len(policy.secret_values)} if spec.redaction_env else None
        keys += m3_ingest(conn, refs, ts, policy=policy)
    from crucible.core.schema.trace import Trace

    out, seen = [], set()
    for k in keys:  # the store returns content keys: each is one stored version of a trace
        t = ts.store.get_doc(k, Trace)
        if t.trace_id not in seen:
            seen.add(t.trace_id)
            out.append(ts.get(t.trace_id))  # the latest version of that trace id
    return sorted(out, key=lambda t: t.trace_id)


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
        from crucible.signals.extract import extract
        from crucible.studio.pack import find_pack, load_manifest, resolve

        pack_dir = find_pack(pack)
        m = load_manifest(pack_dir)
        thr = m.label.threshold if threshold is None else threshold
        if not m.ingest:
            raise ValueError(f"pack {m.name} declares no ingest sources: nothing to label (add `ingest:` to pack.yaml)")
        if dry_run:
            from crucible.studio.pack import ingest_problems

            got = [ingest_problems(pack_dir, s) for s in m.ingest if s.connector == "file_drop"]
            return {**summary, "dry_run": True, "would_ingest": sorted(str(p) for _, fs in got for p in fs),
                    "refused": [p for bad, _ in got for p in bad], "credit_hook": m.hooks.credit}
        out.mkdir(parents=True, exist_ok=True)
        _resolved(out, "label", cfg | {"pack_dir": str(pack_dir), "threshold": thr})
        redaction: dict[str, Any] = {}
        traces = ingest(pack_dir, m, out / "store", labelled_only=True, redaction=redaction)
        summary["redaction_env"] = redaction
        labelled_traces = []
        sig_counts: dict[str, int] = {}
        failed: list[dict[str, str]] = []
        for t in traces:
            try:  # one trace a mapper cannot handle is reported, not the end of the run (M4 audit A15)
                t = extract(t)
            except Exception as e:  # noqa: BLE001 - a pack mapper's error, with the trace it failed on
                failed.append({"trace_id": t.trace_id, "error": f"{type(e).__name__}: {e}"[:500]})
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
        summary |= {"pack": m.name, "traces": len(traces), "signals": dict(sorted(sig_counts.items())), "credited": len(credit), "extract_failed": failed,
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
