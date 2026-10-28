"""Freezing the chosen held-out set (swarm-pipeline/00 §9; 2026-10-24): ``lh_debug.py seal``.

``seal`` refuses to start unless every chosen task is held out, gated and **sealed** by the latest contamination audit
(``factory.benchaudit``: a ``contaminated``, ``unknown`` or ``not_heldout`` verdict, or a task the audit never saw, stops
it, and so does an audit run on another ref than the one named). Then it writes, under ``windows.SEALED_DIR`` (never git):

- ``manifest.json``: per task the sha256 of its window record, its oracle files, its hidden test ids and its report, and
  the store it came from. Hashes only: the manifest proves later that the benchmark was not changed, without holding it.
- **Fresh seeds** for the final run: a new 63-bit master seed (``secrets``), per-task seeds derived by HMAC-SHA256 of the
  task id, so no earlier run (training, targeting, the audit's) shares a seed with the final one. The master seed stays in
  the sealed manifest; ``public.json`` carries only its sha256 (a commitment that can be published and checked later).
- **The eval lease** (``sealed.json`` for ``sources.lease.EvalLease.default(manifest=...)``): the chosen repositories and
  the orgs they belong to, so no training scout or generator reads them while the benchmark is current. The lease's time
  cut (``after``) stays off unless ``time_cut`` is given: these lanes split by owner (both sides' fixes postdate the
  student's cutoff), and a cut at the first held-out fix would refuse every newer training source.
- **The power plan** (``bench.power.power_gate``): the MDE of ``len(ids)`` tasks x ``k`` attempts per arm against the target
  effect, from the A/A noise of our measured student runs; ``require_power`` makes an underpowered plan a refusal.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.bench import power


class SealRefused(ValueError):
    """The chosen held-out set cannot be sealed as it is (every reason listed)."""


def _sha(x: Any) -> str:
    data = x if isinstance(x, bytes) else json.dumps(x, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def task_hashes(wdir: Path, prompt: Path | None) -> dict[str, str]:
    """Hashes of a held-out debug window (``tasks/<id>/window.json``; a directory without one: its ``record.json``)."""
    wf = wdir / "window.json" if (wdir / "window.json").is_file() else wdir / "record.json"
    rec = json.loads(wf.read_text())
    tests = sorted(t for m in rec.get("milestones") or [] for t in (m.get("tests") or m.get("owned") or m.get("cases") or []))
    out = {"record": _sha(wf.read_bytes()), "oracle": _sha(rec.get("oracle_files") or rec.get("oracle_patch") or {}), "hidden_tests": _sha(tests),
           "n_hidden_tests": str(len(tests))}
    if prompt is not None and prompt.is_file():
        out["report"] = _sha(prompt.read_bytes())
    return out


def task_seed(master: int, task_id: str) -> int:
    return int.from_bytes(hmac.new(master.to_bytes(8, "big"), task_id.encode(), hashlib.sha256).digest()[:8], "big") >> 1


def exposure_commits(row: Mapping[str, Any]) -> list[str]:
    """The commits that first published a task's material (the audit's ``first_public``), oldest first."""
    seen: dict[str, str] = {}
    for v in (row.get("first_public") or {}).values():
        sha, _, day = str(v).partition(" ")
        if sha and (sha not in seen or day < seen[sha]):
            seen[sha] = day
    return [c for c, _ in sorted(seen.items(), key=lambda x: (x[1], x[0]))]


def check(ids: Sequence[str], audit: Mapping[str, Any], *, locate: Mapping[str, Path], gates_ok: Mapping[str, bool], ref: str | None = None,
          accept_exposure: Mapping[str, str] | None = None) -> list[str]:
    """Every reason the set cannot be sealed (empty: it can). ``accept_exposure`` (task id -> the lead's reason) lets a
    ``contaminated`` verdict through for that task only: every other refusal (unknown, not held out, not in the audit,
    no store, gates) still holds, and an acceptance for a task that is not contaminated is itself refused."""
    why = []
    if len(set(ids)) != len(ids):
        why.append("a task is chosen twice")
    if ref is not None and audit.get("ref") != ref:
        why.append(f"the audit ran on {audit.get('ref')!r}, not {ref!r}: run scripts/bench_audit.py --ref {ref} first")
    rows = {r["id"]: r for r in audit.get("tasks", [])}
    for i in ids:
        r = rows.get(i)
        if r is None:
            why.append(f"{i}: not in the audit")
        elif r.get("verdict") == "contaminated" and i in (accept_exposure or {}):
            if not str((accept_exposure or {})[i]).strip():
                why.append(f"{i}: an exposure acceptance needs the lead's reason")
        elif r.get("verdict") != "sealed":
            why.append(f"{i}: {r.get('verdict')} ({'; '.join(r.get('why') or []) or 'see the audit'})")
        if r is not None and i in (accept_exposure or {}) and r.get("verdict") != "contaminated":
            why.append(f"{i}: exposure accepted, but the audit says {r.get('verdict')!r}, not 'contaminated'")
        if i not in locate:
            why.append(f"{i}: in no held-out store on this machine")
        if not gates_ok.get(i):
            why.append(f"{i}: its gates have not all passed")
    return why


def plan(n: int, k: int, noise: Mapping[str, Any], *, target: float = power.TARGET, tau: float = 0.0) -> dict[str, Any]:
    return power.power_gate(noise, n, k, target=target, tau=tau, source="seal")


def seal(name: str, ids: Sequence[str], audit: Mapping[str, Any], *, out_root: Path, locate: Mapping[str, Path], gates_ok: Mapping[str, bool],
         repos: Mapping[str, str], fix_dates: Mapping[str, Sequence[str]], noise: Mapping[str, Any], k: int = 8, target: float = power.TARGET,
         ref: str | None = None, prompts: Mapping[str, Path] | None = None, require_power: bool = False, master: int | None = None,
         time_cut: str | None = None, accept_exposure: Mapping[str, str] | None = None, accepted_on: str | None = None,
         journal: Path | None = None) -> dict[str, Any]:
    """Freeze ``ids`` as benchmark ``name`` (see the module doc). ``repos``: task id -> ``owner/repo``. ``accept_exposure``:
    the lead's explicit acceptance of a task whose material is public (task id -> reason); the manifest records it per task
    as ``exposed_public: <commits> (lead accepted <date>: <reason>)`` so a paper can disclose it, and ``journal`` (a JSONL
    file) gets one line per acceptance."""
    accept = dict(accept_exposure or {})
    stray = sorted(set(accept) - set(ids))
    why = check(ids, audit, locate=locate, gates_ok=gates_ok, ref=ref, accept_exposure=accept)
    if stray:
        why.append(f"exposure accepted for tasks not chosen: {', '.join(stray)}")
    pw = plan(len(ids), k, noise, target=target)
    if require_power and pw.get("verdict") != "powered":
        why.append(f"underpowered: {pw.get('headline')}")
    if why:
        raise SealRefused("; ".join(why))
    out = Path(out_root) / name
    if (out / "manifest.json").exists():
        raise SealRefused(f"{out} is sealed already: a sealed benchmark is never rewritten (choose a new name)")
    master = master if master is not None else secrets.randbits(63)
    tasks = {i: {"store": str(locate[i]), "repo": repos.get(i), "hashes": task_hashes(locate[i], (prompts or {}).get(i)),
                 "seed": task_seed(master, i)} for i in ids}
    day = accepted_on or dt.datetime.now(dt.UTC).date().isoformat()
    rows = {r["id"]: r for r in audit.get("tasks", [])}
    for i, reason in accept.items():
        commits = exposure_commits(rows[i])
        tasks[i]["exposed_public"] = f"{', '.join(commits) or 'unknown commit'} (lead accepted {day}: {reason})"
        if journal is not None:
            Path(journal).parent.mkdir(parents=True, exist_ok=True)
            with Path(journal).open("a") as fh:
                fh.write(json.dumps({"at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "benchmark": name, "task": i,
                                     "commits": commits, "reason": reason, "by": "lead"}, sort_keys=True) + "\n")
    dates = sorted(d for i in ids for d in fix_dates.get(i, []) if d)
    slugs = sorted({repos[i].lower() for i in ids if repos.get(i)})
    # the debug lane splits by owner, not by time (both sides' fixes are after the student's cutoff), so the
    # lease's time cut stays off unless asked for: it would refuse every training source dated after the first fix
    lease = {"repos": slugs, "orgs": sorted({s.split("/")[0] for s in slugs if "/" in s}), "after": time_cut,
             "fix_dates": [dates[0], dates[-1]] if dates else []}
    at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    manifest = {"name": name, "at": at, "k": k, "master_seed": master, "tasks": tasks, "lease": lease, "power": pw,
                "audit": {"ref": audit.get("ref"), "sha256": _sha(audit)}}
    manifest["manifest_sha256"] = _sha({x: y for x, y in manifest.items() if x != "manifest_sha256"})
    public = {"name": name, "at": at, "n": len(ids), "k": k, "manifest_sha256": manifest["manifest_sha256"], "exposed_public": len(accept),
              "seed_commitment": _sha(str(master).encode()), "power": {x: pw.get(x) for x in ("verdict", "mde", "target", "headline")}}
    out.mkdir(parents=True, exist_ok=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (out / "sealed.json").write_text(json.dumps(lease, indent=1, sort_keys=True))
    (out / "public.json").write_text(json.dumps(public, indent=1, sort_keys=True))
    return {"dir": str(out), "public": public, "lease": lease}
