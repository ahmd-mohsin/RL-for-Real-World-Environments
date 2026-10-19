"""The run journal (research m9-m10/02 §3.4): append-only, fsynced events plus atomic per-attempt records; re-running
the same RunSpec resumes.

Layout under ``data/runs/<run_id>/``:

    run.lock        the run's lease (M10 lab audit R-4): an exclusive non-blocking OS lock (``core.locks``) held from the
                    moment a ``RunJournal`` opens until it closes, so exactly one writer has the directory; never deleted
    owner.json      who holds it: {host, pid, process_started, started_at}; evidence for ``RunBusy``'s message, not the lock
    run.json        the run's identity (M10 lab audit R-4, ``runtime.identity``): the resolved document v2 (every task's, policy's
                    and harness's configuration and content digests, the retry rule, the sampling, the rollout mode, the
                    operational fields), the code versions, the identity epochs, and ``spec`` (the id lists bench reads). A
                    resume compares every leaf but the operational denylist; a difference is refused naming its dotpaths
                    (``IdentityDrift``) unless it is accepted (``accept_drift``), which, like a code-version change, opens a new
                    epoch; a ``run.json`` from before identity v2 is refused unless ``*`` is accepted
    journal.jsonl   planned | dispatched | phase | budget | finished | retry_scheduled | stop_rule | epoch (torn tail tolerated)
    results.jsonl   one RunRecord per terminal sample (scored, policy-failed, or masked with no retry left)
    failures.jsonl  every masked attempt (never read by metrics; read by status, coverage and retry)
    samples/<sample_id>/a<N>/record.json (+ trajectory, steps.jsonl, verifier/, artifacts/)
    samples/<sample_id>/a<N>/refused.json   a reservation the budget (or the ledger) refused: nothing was dispatched

Resume: done = terminal rows in ``results.jsonl``; owed = planned samples without one; the next attempt index is
1 + the highest index seen live, in ``aborted/`` or refused (M10 lab audit R-4: an index is never reused, so neither is the
ledger op or any sandbox named after the attempt). Attempt dirs without ``record.json`` move to ``samples/<id>/aborted/``
as evidence (to ``aborted/a<N>~<k>`` when ``aborted/a<N>`` exists: a move never nests); a refused reservation
(``refused.json``) keeps its index in place and is no attempt. One writer per run directory: a second ``RunJournal`` on a
held ``run.lock`` raises ``RunBusy`` naming the owner and touches nothing (the dead holder's lock is dropped by the OS, so
there is no stale-lock reclaim; single host, no network filesystems).

Identity epochs (M10 lab audit R-4): ``run.json``'s top-level fields describe the current epoch, ``epochs`` lists them all
(``{epoch, identity, resolved, code, accepted, code_changed, started}``). Epoch 0 is the run's first configuration; a resume
under accepted drift or a different ``code`` document appends the next one and journals an ``epoch`` event. The Runner stamps
records written in an epoch after the first with ``meta.epoch``, so reports can split by it; code versions are never a reason to
refuse (a hotfix resumes), configuration is. A journal opened without a resolved document (direct callers, tests) keeps the
older check on the identity string alone and writes the older ``run.json``.
Patterns: Repo2RLEnv ``EventJournal``/``save_record``, NeMo-Gym failure sidecars and torn-tail repair, FineEnvs identity.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crucible.core.locks import FileLock, process_start_time
from crucible.core.schema.run import RunRecord
from crucible.runtime.identity import IdentityDrift, accepts, compared, diff, normalize

LOCK = "run.lock"
OWNER = "owner.json"
REFUSED = "refused.json"
_LIVE = re.compile(r"a(\d+)")
_SEEN = re.compile(r"a(\d+)(?:~\d+)?")  # a live attempt dir, or an aborted one (``a<N>~<k>`` when ``a<N>`` was taken)


class RunBusy(ValueError):
    """Another process (or another journal of this one) holds the run directory's lease (M10 lab audit R-4). A ValueError,
    so the CLI reports it as a user error."""


def _owner_text(run_dir: Path) -> str:
    """Who ``owner.json`` says holds the run, for a message: evidence only (the file can lag the lock by a moment)."""

    def when(t: Any) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(t)))

    try:
        o = json.loads((run_dir / OWNER).read_text())
        text = f"pid {o['pid']} on {o['host']}"
        times = [f"{label} {when(o[key])}" for label, key in (("process started", "process_started"), ("run opened", "started_at")) if o.get(key) is not None]
        return text + (f" ({', '.join(times)})" if times else "")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "an unknown process"


def _busy(run_id: str, run_dir: Path) -> RunBusy:
    return RunBusy(f"run {run_id} is in use by {_owner_text(run_dir)} ({run_dir / LOCK}): only one process may write a run directory; a second "
                   "writer would move its live attempts to aborted/ and reuse its reservations (M10 lab audit R-4)")


def assert_free(run_dir: Path) -> None:
    """A preflight: ``RunBusy`` when ``run_dir``'s lease is held, nothing else (M10 lab audit R-4). It creates nothing (a
    directory with no ``run.lock`` was never opened), so a caller that does work before ``Runner.run`` (bench's controls, a
    pack's validation episodes under the run dir) refuses a second writer first instead of after it has written beside the
    first. Only a fast-fail: the authority is the lease ``RunJournal`` takes, and a probe that lands on a holder's very
    first moment can see the other side's attempt to open as busy."""
    run_dir = Path(run_dir)
    if not (run_dir / LOCK).exists():
        return
    probe = FileLock(run_dir / LOCK)
    try:
        probe.acquire()
    except BlockingIOError:
        raise _busy(run_dir.name, run_dir) from None
    probe.release()


def _append(path: Path, obj: dict) -> None:
    """Append one JSON line. A torn tail (a crash mid-write) is cut off first (NeMo-Gym's torn-tail truncation; M10 audit
    V10: the next row was concatenated to the fragment and lost, so a finished sample ran again). Credentials are
    scrubbed from the line (V12)."""
    from crucible.core.secrets import scrub_obj

    if path.exists() and path.stat().st_size:
        with path.open("rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                data = path.read_bytes()
                f.truncate(data.rfind(b"\n") + 1)
    with path.open("a") as f:
        f.write(json.dumps(scrub_obj(json.loads(json.dumps(obj, default=str)))) + "\n")
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
    """One writer's view of a run directory. M10 lab audit R-4: constructing it takes the run's lease (``run.lock``,
    ``core.locks.FileLock``) before anything is read or written, and holds it until ``close`` (also ``with`` and garbage
    collection): a second journal on the directory, in this process or another, raises ``RunBusy`` naming the owner. Without
    it a second runner moved the first's live attempts to ``aborted/`` and reused its reservations.

    ``resolved`` (the run's resolved identity document, ``RunSpec.resolved_doc`` normalized), ``code`` (``identity.code_versions``)
    and ``accept_drift`` (dotpath patterns, ``identity.accepts``) make it an identity-v2 journal: the resolved document is
    compared leaf by leaf with the one on disk after the lease is taken, a difference raises ``IdentityDrift`` unless accepted,
    and an accepted difference or a code-version change opens a new epoch (``self.epoch``: 0 for the run's first
    configuration). Without ``resolved`` only the identity string is compared."""

    def __init__(self, root: Path, run_id: str, identity: str, spec: dict[str, Any], *, resolved: dict[str, Any] | None = None,
                 code: dict[str, Any] | None = None, accept_drift: Sequence[str] = ()):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = self.dir / "journal.jsonl"
        self.results = self.dir / "results.jsonl"
        self.failures = self.dir / "failures.jsonl"
        self.live: set[Path] = set()  # attempt dirs running in this process: next_attempt never moves them (M13 audit B7)
        self.epoch = 0
        self.lease = FileLock(self.dir / LOCK)
        try:
            self.lease.acquire()
        except BlockingIOError:
            raise _busy(run_id, self.dir) from None
        try:
            try:  # the owner is evidence for the next opener's message, never correctness: a failed write (win32: os.replace racing a reader) is dropped
                atomic_write(self.dir / OWNER, json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "process_started": process_start_time(),
                                                           "started_at": time.time()}))
            except OSError:
                pass
            meta = self.dir / "run.json"
            if resolved is not None:
                self.epoch = self._open_identity(meta, run_id, identity, spec, normalize(resolved), dict(code or {}), tuple(accept_drift))
            elif meta.exists():
                old = json.loads(meta.read_text())
                if old.get("identity") != identity:
                    raise ValueError(f"run {run_id} exists with a different identity ({old.get('identity')} != {identity}): refusing to resume a different run")
            else:
                atomic_write(meta, json.dumps({"run_id": run_id, "identity": identity, "spec": spec, "created": time.time()}, indent=1, default=str))
        except BaseException:
            self.lease.release()  # a refused resume never leaves the run locked
            raise

    def _open_identity(self, meta: Path, run_id: str, identity: str, spec: dict[str, Any], resolved: dict[str, Any], code: dict[str, Any],
                       accept_drift: tuple[str, ...]) -> int:
        """Identity v2 (M10 lab audit R-4), under the lease: create ``run.json`` with epoch 0, or compare the resolved document
        with the stored one and return the epoch this open belongs to. Configuration that differs is refused naming its
        dotpaths unless ``accept_drift`` covers every one; accepted differences, and a ``code`` that differs from the last
        epoch's (never a reason to refuse), open a new epoch."""
        now = time.time()

        def entry(n: int, accepted: list[str], code_changed: list[str]) -> dict[str, Any]:
            return {"epoch": n, "identity": identity, "resolved": resolved, "code": code, "accepted": accepted, "code_changed": code_changed, "started": now}

        if not meta.exists():
            atomic_write(meta, json.dumps({"run_id": run_id, "identity": identity, "identity_version": 2, "spec": spec, "resolved": resolved, "code": code,
                                           "created": now, "epochs": [entry(0, [], [])]}, indent=1, default=str))
            return 0
        old = json.loads(meta.read_text())
        epochs: list[dict[str, Any]] = list(old.get("epochs") or [])
        if "resolved" not in old:
            # written before identity v2: its configuration was never recorded, so nothing can be compared (the ids it
            # hashed said nothing of max_steps, the sampling or a grader). Resuming it is a decision, not a default
            if "*" not in accept_drift:
                # no dotpath exists to accept (nothing was recorded to differ from): only ``*`` covers it, which the generic advice
                # does not say, and accepting re-keys every sample (harness ids now carry a configuration digest)
                raise IdentityDrift(run_id, ["<identity v1: the configuration was never recorded>"], accepted=accept_drift,
                                    hint="run.json predates identity v2, so there is nothing to compare and no dotpath to accept: resume it only with "
                                         "--accept-drift '*' (RunSpec.accept_drift=('*',)), which opens identity epoch 1 and runs every sample again under "
                                         "the new harness ids (they now carry a digest of the harness configuration), or use a new run id (bench run: a "
                                         "different --out)")
            if not epochs:
                epochs = [{"epoch": 0, "identity": old.get("identity"), "resolved": None, "code": None, "accepted": [], "code_changed": [], "started": old.get("created")}]
            accepted, code_changed = ["<identity v1>"], []
        else:
            last = epochs[-1] if epochs else {}
            old_code = last.get("code") or old.get("code") or {}
            diffs = diff(compared(old["resolved"]), compared(resolved))
            unaccepted = [text for path, text in diffs if not accepts(path, accept_drift)]
            if unaccepted:
                raise IdentityDrift(run_id, unaccepted, accepted=accept_drift)
            accepted = [path for path, _ in diffs]
            code_changed = sorted(k for k in set(old_code) | set(code) if old_code.get(k) != code.get(k))
            if not accepted and not code_changed:
                return int(last.get("epoch", len(epochs) - 1)) if epochs else 0
        n = int(epochs[-1].get("epoch", len(epochs) - 1)) + 1 if epochs else 1
        epochs.append(entry(n, accepted, code_changed))
        atomic_write(meta, json.dumps({**old, "run_id": run_id, "identity": identity, "identity_version": 2, "spec": spec, "resolved": resolved, "code": code,
                                       "epochs": epochs}, indent=1, default=str))
        self.event("epoch", epoch=n, accepted=accepted, code_changed=code_changed)
        return n

    def close(self) -> None:
        """Release the run's lease (idempotent). The Runner calls it when ``run`` / ``regrade`` ends, on every path."""
        self.lease.release()

    def __enter__(self) -> RunJournal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def event(self, event_type: str, /, **data: Any) -> None:
        _append(self.events, {"ts": round(time.time(), 3), "event": event_type, **data})

    def attempt_dir(self, sample_id: str, n: int) -> Path:
        return self.dir / "samples" / sample_id / f"a{n}"

    def finish(self, rec: RunRecord, *, terminal: bool) -> None:
        sid, n = rec.attempt.rsplit("-a", 1) if rec.attempt else (rec.run_id, "1")
        from crucible.core.secrets import scrub_json

        atomic_write(self.attempt_dir(sid, int(n)) / "record.json", scrub_json(rec.model_dump_json(), indent=1))
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
        """The index of the sample's next attempt: 1 + the highest seen live, in ``aborted/`` or refused (M10 lab audit R-4:
        an attempt index, and with it the ledger op ``<sample>-a<N>``, is never reused; a crash that moved ``a1`` to
        ``aborted/`` followed by a refused ``a2`` once gave ``a2`` the index ``a1`` again and overwrote its uncertain row)."""
        d = self.dir / "samples" / sample_id
        if not d.exists():
            return 1
        aborted = d / "aborted"
        seen = [int(m[1]) for p in d.iterdir() if p.is_dir() and (m := _LIVE.fullmatch(p.name))]
        seen += [int(m[1]) for p in aborted.iterdir() if p.is_dir() and (m := _SEEN.fullmatch(p.name))] if aborted.is_dir() else []
        for p in list(d.iterdir()):  # attempts that never wrote a record are kept as evidence, not reused
            # a refused reservation (refused.json) is no attempt and no evidence of a crash: it stays, holding its index
            if p.is_dir() and _LIVE.fullmatch(p.name) and not (p / "record.json").exists() and not (p / REFUSED).exists() and p not in self.live:
                dest = aborted / p.name
                k = 0
                while dest.exists():  # never nested into an earlier aborted copy of this index (pre-R-4 runs left some)
                    k += 1
                    dest = aborted / f"{p.name}~{k}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
        return 1 + max(seen, default=0)

    def refuse(self, adir: Path, aid: str, reason: str) -> None:
        """Mark a reservation the budget (or the ledger) refused, in place: nothing was dispatched, so it is no attempt and has
        no record, but it keeps its index (``next_attempt``) and is not moved to ``aborted/``."""
        atomic_write(Path(adir) / REFUSED, json.dumps({"attempt": aid, "reason": reason, "ts": round(time.time(), 3)}))

    def dispatched(self, sample_id: str, n: int) -> int:
        """How many of the sample's attempts up to index ``n`` were dispatched: ``n`` less the earlier refused reservations.
        The retry rule counts these (M10 lab audit R-4): a budget stop that refused ``a2`` took an index and no attempt, so
        it must not use up one of the sample's retries."""
        d = self.dir / "samples" / sample_id
        return n - sum(1 for k in range(1, n) if (d / f"a{k}" / REFUSED).exists())

    def records(self) -> list[RunRecord]:
        return list(self.terminal().values())

    def masked_attempts(self) -> list[dict]:
        return _read_jsonl(self.failures)
