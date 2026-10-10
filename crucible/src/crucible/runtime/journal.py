"""The run journal (research m9-m10/02 §3.4): append-only, fsynced events plus atomic per-attempt records; re-running
the same RunSpec resumes.

Layout under ``data/runs/<run_id>/``:

    run.json        identity: hash of the spec (minus cosmetic fields), code/framework pins; a mismatch refuses to resume
    journal.jsonl   planned | dispatched | phase | budget | finished | retry_scheduled | stop_rule (torn tail tolerated)
    results.jsonl   one RunRecord per terminal sample (scored, policy-failed, or masked with no retry left)
    failures.jsonl  every masked attempt (never read by metrics; read by status, coverage and retry)
    samples/<sample_id>/a<N>/record.json (+ trajectory, steps.jsonl, verifier/, artifacts/)

Resume: done = terminal rows in ``results.jsonl``; owed = planned samples without one; the next attempt index is
1 + the highest attempt seen. Attempt dirs without ``record.json`` move to ``samples/<id>/aborted/`` as evidence.
Patterns: Repo2RLEnv ``EventJournal``/``save_record``, NeMo-Gym failure sidecars and torn-tail repair, FineEnvs identity.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from crucible.core.schema.run import RunRecord


def _append(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a torn tail from a crash mid-write: ignore the fragment
            continue
    return out


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RunJournal:
    def __init__(self, root: Path, run_id: str, identity: str, spec: dict[str, Any]):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "run.json"
        if meta.exists():
            old = json.loads(meta.read_text())
            if old.get("identity") != identity:
                raise ValueError(f"run {run_id} exists with a different identity ({old.get('identity')} != {identity}): refusing to resume a different run")
        else:
            atomic_write(meta, json.dumps({"run_id": run_id, "identity": identity, "spec": spec, "created": time.time()}, indent=1, default=str))
        self.events = self.dir / "journal.jsonl"
        self.results = self.dir / "results.jsonl"
        self.failures = self.dir / "failures.jsonl"

    def event(self, event_type: str, /, **data: Any) -> None:
        _append(self.events, {"ts": round(time.time(), 3), "event": event_type, **data})

    def attempt_dir(self, sample_id: str, n: int) -> Path:
        return self.dir / "samples" / sample_id / f"a{n}"

    def finish(self, rec: RunRecord, *, terminal: bool) -> None:
        sid, n = rec.attempt.rsplit("-a", 1) if rec.attempt else (rec.run_id, "1")
        atomic_write(self.attempt_dir(sid, int(n)) / "record.json", rec.model_dump_json(indent=1))
        if rec.mask_sample:
            _append(self.failures, json.loads(rec.model_dump_json()))
        if terminal:
            _append(self.results, json.loads(rec.model_dump_json()))
        self.event("finished", attempt=rec.attempt, outcome=rec.outcome, reward=rec.scored_reward(), terminal=terminal,
                   failure=rec.failure.kind if rec.failure else None)

    def terminal(self) -> dict[str, RunRecord]:
        out = {}
        for row in _read_jsonl(self.results):
            r = RunRecord.model_validate(row)
            out[(r.attempt or "").rsplit("-a", 1)[0]] = r
        return out

    def next_attempt(self, sample_id: str) -> int:
        d = self.dir / "samples" / sample_id
        if not d.exists():
            return 1
        seen = [int(p.name[1:]) for p in d.iterdir() if p.is_dir() and p.name.startswith("a") and p.name[1:].isdigit()]
        for p in d.iterdir():  # attempts that never wrote a record are kept as evidence, not reused
            if p.is_dir() and p.name.startswith("a") and p.name[1:].isdigit() and not (p / "record.json").exists():
                dest = d / "aborted" / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
        return 1 + max(seen, default=0)

    def records(self) -> list[RunRecord]:
        return list(self.terminal().values())

    def masked_attempts(self) -> list[dict]:
        return _read_jsonl(self.failures)
