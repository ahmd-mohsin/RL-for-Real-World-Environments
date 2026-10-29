"""``crucible label``: workflows → traces with dense rewards, plus SFT / DPO / KTO data (ARCHITECTURE §4 M15).

Two sources:
- **a pack** (``--pack``). This runs M3 ingest over the pack's ``ingest`` sources, which is redacted and idempotent. Then:
  1. M4 signals and pivots (when the family has a signal mapper);
  2. the pack's ``credit`` hook (M5 milestones → M6 Φ per event, a gated reward, the outcome and the trace's
     ``split``: only ``train`` traces become data);
  3. the validity gate (``credit.channels``): per family, Φ's two uses (per-turn weights; partial credit in the reward)
     are released only when they predict the outcome on the hook's held-out traces; otherwise the data carries no dense
     credit for that use;
  4. the conversation from the policy role's side (``studio.datasets.from_trace``).
- **a run** (``--run``): an M10 run directory, read through the runner's training view (``rollout.json`` token turns,
  ``mask_sample``, flags), its ``validity.json`` and the build's ``splits.json``: only valid, train-split, unmasked,
  unflagged, model-produced episodes become data, decoded with the policy's own tokenizer and redacted like the pack
  path (``studio.datasets.from_run``).

Rows are one decision each (``studio.datasets``). Outputs in ``out/``, each replaced atomically:
- ``configs/resolved/label.json``;
- ``store/`` (the traces; pack only);
- ``traces.jsonl`` (the labelled traces: signals, outcome; pack only);
- ``rewards.jsonl`` (per trace: reward, gates, Φ per event, split, channel verdicts, ``reward_with_progress`` when the
  partial credit was withheld; pack only);
- ``sft.jsonl``, ``dpo.jsonl``, ``kto.jsonl`` (each loads with ``datasets.load_dataset("json")``);
- ``summary.json`` (counts, what was left out and why, ``credit_channels`` per family).
"""

from __future__ import annotations

import json
from collections import Counter
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


def _write_rows(rows: list[dict[str, Any]], out: Path, name: str) -> tuple[Path | None, int]:
    """One shape per file: TRL decides text vs conversational from a dataset's first row (``is_conversational``), so
    the policy's token rows (text) and chat rows (workflow traces, scripted demonstrations) never share a file. Text rows
    go to ``<name>.jsonl``; chat rows too when there are no text rows, else to ``<name>.chat.jsonl``. Returns the chat
    file (when written) and its row count."""
    chat = [r for r in rows if not isinstance(r.get("prompt", r.get("input_ids")), str) and "input_ids" not in r]
    text = [r for r in rows if not any(r is c for c in chat)]
    side = out / f"{name}.chat.jsonl"
    if text and chat:
        ds.write_jsonl(text, out / f"{name}.jsonl")
        ds.write_jsonl(chat, side)
        return side, len(chat)
    ds.write_jsonl(rows, out / f"{name}.jsonl")
    side.unlink(missing_ok=True)
    return None, 0


async def label(*, out: Path, pack: str | Path | None = None, run: Path | None = None, filter: str | None = None, select: str = "correct",
                max_per_task: int | None = None, threshold: float | None = None, margin: float = 0.25, mask_negative: bool = False,
                dry_run: bool = False, allow_scripted: bool = False, include_bench: bool = False, allow_unknown: bool = False,
                allow_state_shift: bool = False, policies: list[str] | None = None, tokenizer: str | None = None,
                redaction_env: Path | None = None, sft_format: str = "turns", template: str | None = None,
                validity: Path | None = None) -> dict[str, Any]:
    """Label a pack's workflow logs or a run's episodes into SFT / DPO / KTO rows (``studio.datasets``). Only
    train-split, valid, unmasked, unflagged, model-produced trajectories become data (M15 audit H1 V1-V19, G1 D1-D15);
    everything left out is counted by reason in ``summary.json`` ``skipped``. Nothing is written before every input
    check passed, and each output file is replaced atomically."""
    if (pack is None) == (run is None):
        raise ValueError("label needs exactly one source: --pack <pack> or --run <run dir>")
    if max_per_task is not None and max_per_task < 1:
        raise ValueError(f"--max-per-task must be >= 1 (got {max_per_task}); omit it for no cap")
    if margin <= 0:
        raise ValueError(f"--margin must be > 0 (got {margin}): a preference pair needs a reward gap")
    if select not in ("correct", "best", "all"):
        raise ValueError("--select: correct | best | all")
    if sft_format not in ("turns", "chain"):
        raise ValueError("--sft-format: turns | chain")
    out = Path(out)
    ds.compile_filter(filter)  # a bad filter fails before any work
    cfg = {"pack": str(pack) if pack else None, "run": str(run) if run else None, "filter": filter, "select": select, "max_per_task": max_per_task,
           "threshold": threshold, "margin": margin, "mask_negative": mask_negative, "allow_scripted": allow_scripted, "include_bench": include_bench,
           "allow_unknown": allow_unknown, "allow_state_shift": allow_state_shift, "policies": policies, "tokenizer": tokenizer,
           "redaction_env": str(redaction_env) if redaction_env else None, "sft_format": sft_format, "template": template,
           "validity": str(validity) if validity else None}
    summary: dict[str, Any] = {"command": "label", "config": cfg}
    skipped: Counter = Counter()
    from crucible.studio import runs

    if not dry_run:  # a re-run into this --out with other flags is refused before anything is written (M15 G2 D6)
        runs.check(out, "label", cfg)
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
        # the resolved config is the first write: compared with an existing one (``runs.ConfigChanged``), then stamped
        runs.record(out, "label", cfg | {"threshold_used": thr, "pack_dir": str(pack_dir)}, identity=cfg, pack_dir=pack_dir, manifest=m)
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
        # M5 R1: Φ reaches the data only through a held-out verdict per channel (step weights, partial credit); an
        # unreleased channel falls back to "no dense credit" (credit.channels)
        from crucible.credit import channels

        verdicts = {fam: channels.decide(fam, [t for t in labelled_traces if str(t.family) == fam], credit)
                    for fam in sorted({str(t.family) for t in labelled_traces if t.trace_id in credit})}
        trained_phi: dict[str, Any] = {}
        for t in labelled_traces:
            if t.trace_id in credit:
                credit[t.trace_id], trained_phi[t.trace_id] = channels.apply(credit[t.trace_id], verdicts[str(t.family)], hook=str(m.hooks.credit))
        summary["credit_channels"] = {fam: v.to_dict() for fam, v in verdicts.items()}
        # M4 audit R-2: a trace whose extraction raised has no M4 verdict, so nothing in it is graded wrong: it yields no
        # row (a corrected step became an SFT target and KTO-desirable) and stays in traces.jsonl as evidence (PATTERN:
        # AReaL's PRM error_policy="reject", areal/api/cli_args.py:2701-2711)
        unextracted = {f["trace_id"] for f in failed}
        if unextracted:
            skipped["extract_failed"] += len(unextracted)
        all_items = [ds.from_trace(t, policy_roles=m.label.policy_roles, phi_by_seq=trained_phi.get(t.trace_id),
                                   reward=(credit.get(t.trace_id) or {}).get("reward"), tools=(credit.get(t.trace_id) or {}).get("tools"),
                                   split=str((credit.get(t.trace_id) or {}).get("split") or "train")) for t in labelled_traces
                     if t.trace_id not in unextracted]
        # the credit hook's split: held-out (test) traces grade the pack's gates, never train (V9)
        held = sorted(x.episode for x in all_items if x.split != "train" and not include_bench)
        skipped["held_out"] += len(held)
        items = [x for x in all_items if x.split == "train" or include_bench]
        ds_payload = {"traces.jsonl": "".join(t.model_dump_json() + "\n" for t in labelled_traces),
                      "rewards.jsonl": [{"trace_id": t.trace_id, "outcome": t.outcome.label, **dict(credit.get(t.trace_id) or {})} for t in labelled_traces]}
        summary |= {"pack": m.name, "traces": len(traces), "signals": dict(sorted(sig_counts.items())), "credited": len(credit), "extract_failed": failed,
                    "outcomes": {k: sum(t.outcome.label == k for t in labelled_traces) for k in ("good", "bad", "unknown")},
                    "held_out": held, "used": len(items),
                    "pivots": {"pairs": sum(len(x.pivots) for x in items), "without_policy_fix": sum(x.pivots_unpaired for x in items)}}
    else:
        thr = 1.0 if threshold is None else threshold
        from crucible.connectors.redact import RedactionPolicy

        policy = RedactionPolicy.with_env_file(redaction_env) if redaction_env else RedactionPolicy()
        info: dict[str, Any] = {}
        items = ds.from_run(Path(run), tokenizer=tokenizer, policies=policies, allow_scripted=allow_scripted, include_bench=include_bench,
                            redaction=policy, validity=validity, skipped=skipped, info=info)
        summary |= {"run": str(run), "episodes": len(items), "considered": len(items) + sum(skipped.values()), **info,
                    "redaction_env": {"env": str(redaction_env), "values": len(policy.secret_values)} if redaction_env else None}
        ds_payload = {}
        if dry_run:
            return {**summary, "dry_run": True, "skipped": dict(sorted(skipped.items()))}
    sft = ds.sft_rows(ds.curate(items, filter=filter, select=select, max_per_task=max_per_task, threshold=thr, allow_unknown=allow_unknown),
                      threshold=thr, mask_negative=mask_negative, fmt=sft_format, skipped=skipped)
    dpo = (ds.dpo_best_vs_worst(items, threshold=thr, margin=margin, allow_unknown=allow_unknown, skipped=skipped)
           + ds.dpo_pivots(items, allow_state_shift=allow_state_shift, skipped=skipped))
    kto = ds.kto_rows(items, threshold=thr, allow_unknown=allow_unknown, skipped=skipped)
    if template:  # fit the chat rows to a consumer's template (Gemma-2 has no system or tool role) and keep what renders
        tok = ds.load_tokenizer(template)
        roles = ds.template_roles(tok)
        sft, dpo, kto = (ds.check_render(ds.adapt_rows(r, roles), tok, skipped, n) for r, n in ((sft, "sft"), (dpo, "dpo"), (kto, "kto")))
        summary["template"] = {"tokenizer": template, "roles": roles}
    out.mkdir(parents=True, exist_ok=True)
    if pack is None:  # the run path's first write (it read and checked everything above)
        runs.record(out, "label", cfg | {"threshold_used": thr}, identity=cfg)
    for name, payload in ds_payload.items():
        if isinstance(payload, str):
            tmp = out / f".{name}.tmp"
            tmp.write_text(payload)
            tmp.replace(out / name)
        else:
            ds.write_jsonl(payload, out / name)
    files = {n: _write_rows(r, out, n) for r, n in ((sft, "sft"), (dpo, "dpo"), (kto, "kto"))}
    summary["datasets"] = {"sft": len(sft), "dpo": len(dpo), "kto": len(kto), "chat_rows": {n: f[1] for n, f in files.items() if f[1]},
                           "dpo_kinds": {k: sum(r["kind"] == k for r in dpo) for k in ("best_vs_worst", "correction_pivot")},
                           "kto_balance": {"desirable": sum(r["label"] for r in kto), "undesirable": sum(not r["label"] for r in kto)},
                           "sft_format": sft_format,
                           "text_inexact": sum(1 for r in [*sft, *dpo, *kto] if r.get("text_exact") is False),
                           "redacted_rows": sum(1 for r in [*sft, *dpo, *kto] if r.get("redacted"))}
    summary["skipped"] = dict(sorted(skipped.items()))
    summary["files"] = {**{k: str(out / f"{k}.jsonl") for k in ("sft", "dpo", "kto")}, **{f"{k}_chat": str(f[0]) for k, f in files.items() if f[0]},
                        "summary": str(out / "summary.json")}
    n_rows = len(sft) + len(dpo) + len(kto)
    # what consumes these files (G1 D15: ``crucible.train.launch`` reads no dataset): HF datasets -> TRL's trainers
    summary["next"] = [f"datasets.load_dataset('json', data_files='{f}')  # TRL {t}" for k, t, n in
                       (("sft", "SFTTrainer", len(sft)), ("dpo", "DPOTrainer", len(dpo)), ("kto", "KTOTrainer", len(kto)))
                       for f, m in ((out / f"{k}.jsonl", n - files[k][1]), (files[k][0], files[k][1])) if f and m]
    if not n_rows:
        summary["warning"] = "no training rows: every trajectory was left out (see skipped)"
    tmp = out / ".summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(out / "summary.json")
    return summary
