"""The episode ledger (M16, L3 of the three tiers): every event of an RLM agent, append-only on disk, outside the kernel.

Why (``docs/research/long-horizon-rlm/04-design.md`` §3.1; requirement R1 of ``01-frameworks.md`` §7, H1 of
``02-papers-articles.md``): a 10M-token episode cannot keep its past in the prompt (L1) or in kernel memory (the old
in-kernel ``HISTORY`` held capped outputs and grew the kernel, B10), and the stress test found each tool result stored
3-4 times (``OBS``, the journal, the trace, checkpoints; G3 / B5). The ledger holds each text once, by a stable index,
and the kernel reads it back through ``history`` host requests (G4).

Shape (PATTERN / API shape ported from nano-rlm, MIT: ``session.py:91-156`` the append-only ``messages.jsonl`` with a
stable message index and numbered ``context_window`` records with a reason; ``history.py:12-67`` reading a torn last
record as "a live writer may not have finished"; ``prompt.py:313-323`` ``h.windows`` / ``h.messages`` /
``h.user_messages()``). What is new here, for horizons nano-rlm does not reach:

- **Disk layout.** ``<root>/<name>/events.jsonl`` (one JSON object per line, flushed per event), ``index.bin`` (a
  sidecar of fixed 52-byte records: offset, step, window, kind; written buffered, rebuilt from the log's tail on open),
  ``blobs/<sha256>.txt`` (text fields over 64 KiB, **content-addressed**: an identical text is stored once however many
  events cite it; ``ref(text)`` stores one and returns a reference any later ``append`` can carry), and children in
  ``<root>/<name>/<child>/`` with the same layout (nano-rlm's session tree mirrors the call tree, ``supervisor.py:457``).
- **O(1) addressing.** The index lives in memory as arrays (offset, step, window, kind per event); ``get(i)`` is one
  ``pread``. ``Ledger.open`` reloads the sidecar, checks its last record against the log, scans only the tail it does not
  cover, and drops a torn last line (a crash loses at most the event being written).
- **Windows.** ``open_window(reason)`` writes a ``window`` event; a window is the contiguous index range up to the next
  one. Closed windows never change (nano-rlm ``session.py:94``).
- **Bounded search.** Substring by default (``str.find``, with a raw-line prefilter); ``regex=True`` runs in a separate
  worker process killed at the deadline, so a catastrophic pattern (``(a+)+$``) cannot hang the host. Every call stops at
  2 s wall-clock or 64 MiB scanned and says so (``timed_out``, ``reason``, ``next`` to continue older, ``incomplete``
  for an event it could not finish).
- **Kernel replies** (``handle``): an op allowlist, strict types and ranges (a bad request is a ``ValueError``: the
  cell's exception), JSON-safe replies of at most 256 KiB (as the kernel frames them, ASCII-escaped), never a host path;
  a ledger I/O fault is a ``LedgerError`` (infra: the caller masks it).

Single writer. ``handle`` runs in the writer's event loop; ``ahandle`` runs a search in a thread so a 2 s scan does not
stall the loop (reads see the events that existed when the call began; appends meanwhile are safe).
"""

from __future__ import annotations

import asyncio
import bisect
import codecs
import contextlib
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any, Self

INLINE_CHARS = 65_536  # a text field longer than this goes to blobs/
GET_CHARS = 65_536  # get(): each text field capped at this many chars, the rest named in "more"
READ_MAX = 131_072  # read(): max_chars ceiling (the reply cap may cut further)
REPLY_CAP = 262_144  # a reply to the kernel, as json.dumps frames it (ASCII-escaped)
SEARCH_WALL_S = 2.0  # the bound a call keeps (the scan itself stops _MARGIN_S earlier: killing and parsing take the rest)
_MARGIN_S = 0.1
REPLAY_TIMEOUT_S = 60.0  # a replayed regex search re-scans only events the live call finished: a hang here is a fault
SEARCH_BYTES = 64 * 1024 * 1024
SEARCH_LIMIT = 100
MAX_PATTERN = 1_000
MAX_CONTEXT = 2_000
MAX_PREVIEW = 4_000
PAGE = 200  # items per list reply to the kernel
CHUNK = 4 * 1024 * 1024  # blob bytes per search / skip step

KINDS = ("system", "user", "assistant", "tool", "world_call", "world_send", "inbox", "sub_call", "summary", "compaction_attempt",
         "notice", "window", "child_spawn", "child_event", "child_return", "snapshot", "restore", "abandon")
WINDOW_REASONS = frozenset({"start", "compaction", "compaction_failed", "restore", "rerender", "resume"})
USER_KINDS = frozenset({"user", "inbox"})  # every user / supervisor / INBOX message
HEADER = ("i", "kind", "window", "step", "t", "agent")
_RESERVED_FIELDS = frozenset(HEADER) | {"more", "chars"}
_RESERVED_NAMES = frozenset({"blobs", "events.jsonl", "index.bin"})
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_KIND = re.compile(r"[a-z][a-z_]{0,31}")
_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_SHA = re.compile(r"[0-9a-f]{64}")
_REC = struct.Struct("<QqI32s")  # offset, step, window, kind (ascii, NUL-padded)
_FLUSH_INDEX = 1024  # sidecar records are flushed every N events (the log is the source of truth)


class LedgerError(RuntimeError):
    """The ledger itself failed (I/O, corruption): an infra fault, never the policy's. Messages carry no host paths."""


class Page(list):
    """A list of events (or windows) plus where it stopped. ``next``: the ``start`` (``before`` for search) that continues;
    None when nothing is left. Search also sets ``timed_out``, ``reason`` (limit | scan_cap | timeout), ``scanned_bytes``
    and ``incomplete`` (an event the bound cut mid-scan; it is skipped by ``next``: read it with ``read``)."""

    def __init__(self, items: Any = (), *, start: int = 0):
        super().__init__(items)
        self.start = start
        self.next: int | None = None
        self.timed_out = False
        self.reason: str | None = None
        self.scanned_bytes = 0
        self.incomplete: int | None = None
        self.floor: int | None = None  # search: every target at index >= floor was searched in full (the replay range)


def check_name(name: Any) -> str:
    """A ledger or child name: one path segment of ``[A-Za-z0-9_.-]``, not starting with a dot or dash, not a file the
    ledger itself uses. Refuses traversal (``..``, ``/``, absolute paths) and NUL."""
    if not isinstance(name, str) or not _NAME.fullmatch(name) or name in _RESERVED_NAMES or ".." in name:
        raise ValueError(f"invalid ledger name {name!r}: use 1-64 characters of letters, digits, '_', '.', '-' (one path segment)")
    return name


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _wire(obj: Any) -> int:
    """Size of ``obj`` as the kernel's host-reply frame encodes it (``json.dumps`` default: ASCII-escaped)."""
    return len(json.dumps(obj, default=str))


def _is_ref(v: Any) -> bool:
    return isinstance(v, dict) and "$blob" in v


def _int(v: Any, what: str, lo: int | None = None, hi: int | None = None) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(f"{what} must be an integer, not {type(v).__name__}")  # noqa: TRY004 -- a bad request is the cell's ValueError
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f"{what} must be in [{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}], got {v}")
    return v


def _opt_int(v: Any, what: str, lo: int | None = None, hi: int | None = None) -> int | None:
    return None if v is None else _int(v, what, lo, hi)


def _kinds(v: Any) -> list[str] | None:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) > 32 or not all(isinstance(k, str) and _KIND.fullmatch(k) for k in v):
        raise ValueError("kinds must be a list of up to 32 event kinds, e.g. ['tool', 'user']")
    return list(v)


class Ledger:
    """One agent's append-only event log (see the module docstring). ``Ledger(root, name=...)`` creates a new one and
    refuses a directory that already holds events (two episodes must never mix); ``Ledger.open`` reopens one, after a
    crash too."""

    def __init__(self, root: Path | str, *, name: str = "root", _agent: str | None = "", _recover: bool = False):
        self.name = check_name(name)
        self.dir = Path(root) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "blobs").mkdir(exist_ok=True)
        self._events = self.dir / "events.jsonl"
        self._index = self.dir / "index.bin"
        if not _recover and self._events.exists() and self._events.stat().st_size:
            raise FileExistsError(f"ledger {self.name!r} already holds events: reopen it with Ledger.open")
        self._off = array("Q")
        self._steps = array("q")
        self._wins = array("I")
        self._kind: list[str] = []
        self._kcache: dict[str, str] = {}
        self._windows: list[dict[str, Any]] = []
        self._wstart: list[int] = []  # each window's start index (increasing): windows visible under an upto view
        self._abandoned: list[tuple[int, int, int, str]] = []  # (abandon event index, start, end, reason), in order
        self._blobs: set[str] = set()
        self._blob_bytes = 0
        self.dedup_hits = 0
        self.torn_bytes = 0  # bytes of a torn last line dropped by open()
        self.scanned_events = 0  # events open() indexed by scanning the log (the sidecar did not cover them)
        self._children: dict[str, Ledger] = {}
        self._mono = True  # steps non-decreasing (a resume may rewind the World step)
        self._t_last = 0.0
        self._size = 0
        self._unflushed = 0
        self.agent = _agent or ""
        if _recover:
            self._recover(agent_known=_agent is not None)
        else:
            self._events.touch()
            self._index.write_bytes(b"")
        try:
            self._wf = open(self._events, "ab")  # noqa: SIM115 -- held open for the ledger's life, closed in close()
            self._xf = open(self._index, "ab")  # noqa: SIM115
            self._rfd = os.open(self._events, os.O_RDONLY)
        except OSError as e:
            raise LedgerError(f"cannot open ledger {self.name!r}: {type(e).__name__}") from e
        self._clock0 = time.monotonic() - self._t_last
        self._closed = False

    @classmethod
    def open(cls, root: Path | str, *, name: str = "root") -> Ledger:
        """Reopen (or create) a ledger. A torn last line (a crash mid-write) is dropped; the index continues after the last
        whole event; a missing or inconsistent sidecar is rebuilt from the log."""
        return cls(root, name=name, _agent=None, _recover=True)

    # ---- recovery -------------------------------------------------------------------------------------------------------
    def _recover(self, *, agent_known: bool) -> None:
        if not self._events.exists():
            self._events.touch()
        size = self._events.stat().st_size
        with open(self._events, "r+b") as f:
            cut = self._last_newline(f, size)
            if cut < size:  # torn last line: the writer died mid-event
                self.torn_bytes = size - cut
                f.truncate(cut)
                size = cut
        self._size = size
        raw = self._index.read_bytes() if self._index.exists() else b""
        raw = raw[: len(raw) - len(raw) % _REC.size]
        fd = os.open(self._events, os.O_RDONLY)
        try:
            prev = -1
            for off, step, win, kind in _REC.iter_unpack(raw):
                if off >= size or off <= prev:
                    break
                self._push(off, step, win, kind.rstrip(b"\0").decode("ascii", "replace"))
                prev = off
            pos = 0
            if self._off:  # the sidecar's last record must name the event the log holds at that offset
                last = len(self._off) - 1
                ln = self._line_len(fd, self._off[last], size)
                ok = False
                with contextlib.suppress(ValueError):
                    ev = json.loads(self._pread(fd, self._off[last], ln))
                    ok = isinstance(ev, dict) and ev.get("i") == last and ev.get("kind") == self._kind[last]
                if ok:
                    pos = self._off[last] + ln
                else:
                    self._clear_index()
            n_side = len(self._off)
            with open(self._events, "rb") as f:
                f.seek(pos)
                for line in f:
                    try:
                        ev = json.loads(line)
                        if not isinstance(ev, dict) or ev.get("i") != len(self._off):
                            raise ValueError("index mismatch")
                    except ValueError:
                        if pos + len(line) >= size:  # a whole but unparsable last line: treat as torn
                            self.torn_bytes += size - pos
                            with open(self._events, "r+b") as g:
                                g.truncate(pos)
                            self._size = size = pos
                            break
                        raise LedgerError(f"ledger {self.name!r} is corrupt at event {len(self._off)}") from None
                    self._push(pos, int(ev.get("step") or 0), int(ev.get("window") or 0), str(ev.get("kind")))
                    pos += len(line)
            self.scanned_events = len(self._off) - n_side
            if len(self._off) != n_side or len(raw) != n_side * _REC.size:
                self._index.write_bytes(b"".join(_REC.pack(self._off[j], self._steps[j], self._wins[j], self._kind[j].encode())
                                                 for j in range(len(self._off))))
            n = len(self._off)
            for j in range(n):
                if self._kind[j] == "window":
                    ev = json.loads(self._pread(fd, self._off[j], self._line_len(fd, self._off[j], size)))
                    self._windows.append({"w": ev["window"], "reason": ev.get("reason"), "start_index": j, "meta": ev.get("meta") or {}})
                    self._wstart.append(j)
                elif self._kind[j] == "abandon":
                    ev = json.loads(self._pread(fd, self._off[j], self._line_len(fd, self._off[j], size)))
                    self._abandoned.append((j, int(ev["start"]), int(ev["end"]), str(ev.get("reason") or "")))
            if n:
                tail = json.loads(self._pread(fd, self._off[n - 1], size - self._off[n - 1]))
                self._t_last = float(tail.get("t") or 0.0)
                if not agent_known:
                    first = json.loads(self._pread(fd, 0, self._line_len(fd, 0, size)))
                    self.agent = str(first.get("agent") or "")
        finally:
            os.close(fd)
        bd = self.dir / "blobs"
        for p in bd.iterdir():
            if p.suffix == ".txt" and _SHA.fullmatch(p.stem):
                self._blobs.add(p.stem)
                self._blob_bytes += p.stat().st_size
            elif p.name.endswith(".tmp"):
                p.unlink(missing_ok=True)  # a blob write the crash interrupted (blobs land by rename)

    def _clear_index(self) -> None:
        self._off, self._steps, self._wins, self._kind = array("Q"), array("q"), array("I"), []
        self._mono = True

    @staticmethod
    def _last_newline(f: Any, size: int) -> int:
        pos = size
        while pos > 0:
            n = min(1 << 16, pos)
            f.seek(pos - n)
            buf = f.read(n)
            k = buf.rfind(b"\n")
            if k >= 0:
                return pos - n + k + 1
            pos -= n
        return 0

    @staticmethod
    def _line_len(fd: int, off: int, size: int) -> int:
        n, step = 0, 1 << 16
        while off + n < size:
            buf = os.pread(fd, step, off + n)
            k = buf.find(b"\n")
            if k >= 0:
                return n + k + 1
            n += len(buf)
            step = min(step * 2, 1 << 24)
        return n

    @staticmethod
    def _pread(fd: int, off: int, n: int) -> bytes:
        out = os.pread(fd, n, off)
        while len(out) < n:
            more = os.pread(fd, n - len(out), off + len(out))
            if not more:
                break
            out += more
        return out

    def _push(self, off: int, step: int, win: int, kind: str) -> None:
        if self._steps and step < self._steps[-1]:
            self._mono = False
        self._off.append(off)
        self._steps.append(step)
        self._wins.append(win)
        self._kind.append(self._kcache.setdefault(kind, kind))

    # ---- writing --------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._off)

    @property
    def window(self) -> int:
        """The current window's number (-1 before the first event)."""
        return len(self._windows) - 1

    def append(self, kind: str, /, **fields: Any) -> int:
        """Log one event; returns its index (stable, monotonic). ``step`` (the World step) defaults to the last one. Text
        fields over 64 KiB go to a content-addressed blob; a ``ref(...)`` value is stored as that reference (one copy)."""
        self._check_open()
        if not isinstance(kind, str) or not _KIND.fullmatch(kind) or kind in ("window", "abandon"):
            raise ValueError(f"invalid event kind {kind!r} (lowercase letters and '_', at most 32; windows via open_window, abandon via abandon)")
        step = fields.pop("step", None)
        _opt_int(step, "step", 0, 2**62)
        for k in fields:
            if not _FIELD.fullmatch(k) or k in _RESERVED_FIELDS:
                raise ValueError(f"invalid or reserved field name {k!r}")
        stored = {k: self._store(k, v) for k, v in fields.items()}  # every check before anything is written
        if not self._windows:
            self.open_window("start")
        return self._write(kind, step, stored, len(self._windows) - 1)

    def open_window(self, reason: str, **meta: Any) -> int:
        """Start a context window (reason: start | compaction | compaction_failed | restore | rerender | resume); returns
        its number. ``step=`` in ``meta`` sets the event's World step (a resume may rewind it)."""
        self._check_open()
        if reason not in WINDOW_REASONS:
            raise ValueError(f"window reason must be one of {sorted(WINDOW_REASONS)}, got {reason!r}")
        step = meta.pop("step", None)
        try:
            if len(_dumps(meta)) > INLINE_CHARS:
                raise ValueError("window meta exceeds 64 KiB")
        except (TypeError, ValueError) as e:
            raise ValueError(f"window meta must be small JSON: {e}") from None
        w = len(self._windows)
        i = self._write("window", step, {"reason": reason, "meta": meta}, w)
        self._windows.append({"w": w, "reason": reason, "start_index": i, "meta": meta})
        self._wstart.append(i)
        return w

    def ref(self, text: str) -> dict[str, Any]:
        """Store ``text`` once (content-addressed by sha256) and return a reference to pass as a field value: every event
        that carries it cites one blob (G3: one copy of each observation)."""
        self._check_open()
        if not isinstance(text, str):
            raise ValueError("ref() takes a string")  # noqa: TRY004 -- a bad request is the cell's ValueError
        return self._put_blob(text)

    def abandon(self, start: int, end: int, reason: str) -> None:
        """Hide events ``start <= i < end`` from every kernel view: a crashed attempt wrote them past the checkpoint the
        resume restored, so they describe a future the restore undid. The log stays append-only: an ``abandon`` event
        records the range (it survives ``open``), and views at or after that event exclude it. A view fixed before it (a
        journal marker with ``upto`` <= its index) still sees what it saw, so replay stays exact. An empty range is a no-op."""
        self._check_open()
        n = len(self)
        _int(start, "start", 0, n)
        _int(end, "end", start, n)
        if not isinstance(reason, str) or not 1 <= len(reason) <= 200:
            raise ValueError("abandon needs a reason string of 1..200 characters")
        if start == end:
            return
        if not self._windows:
            self.open_window("start")
        step = self._steps[start - 1] if start else 0  # not the abandoned future's step
        i = self._write("abandon", step, {"start": start, "end": end, "reason": reason}, len(self._windows) - 1)
        self._abandoned.append((i, start, end, reason))

    def lengths(self) -> dict[str, int]:
        """The event count of this ledger and of every child ledger on disk, by agent path (``""`` = this one): what a
        checkpoint stores so ``rewind_to`` can abandon what a crashed attempt wrote after it, in every ledger of the tree."""
        out = {"": len(self)}
        for p in sorted(self.dir.iterdir()):
            if p.is_dir() and p.name != "blobs" and (p / "events.jsonl").exists():
                for k, v in self.child(p.name).lengths().items():
                    out[f"{p.name}/{k}" if k else p.name] = v
        return out

    def rewind_to(self, lengths: dict[str, int], reason: str = "resume") -> dict[str, int]:
        """After a resume: abandon every event written after the checkpoint's ``lengths()``, in this ledger and every
        child ledger (a child the checkpoint did not know is abandoned whole). Returns the abandoned count per agent path."""
        if not isinstance(lengths, dict) or not all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) for k, v in lengths.items()):
            raise ValueError("rewind_to needs the dict a checkpoint saved from lengths()")
        done: dict[str, int] = {}
        for path, now in self.lengths().items():
            led = self._resolve(path or None)
            keep = min(lengths.get(path, 0), now)
            if keep < now:
                led.abandon(keep, now, reason)
                done[path] = now - keep
        return done

    def _ranges(self, n: int) -> list[tuple[int, int]]:
        """The ranges hidden in a view of ``n`` events: those abandoned by an event inside the view."""
        return [(s, e) for at, s, e, _ in self._abandoned if at < n]

    @staticmethod
    def _hidden(j: int, rs: list[tuple[int, int]]) -> bool:
        return any(s <= j < e for s, e in rs)

    def _visible(self, lo: int, hi: int, n: int, ks: Any = None) -> Any:
        """Indices ``lo <= j < hi`` a view of ``n`` events shows (optionally of some kinds): a range when nothing is hidden."""
        rs = [(s, e) for s, e in self._ranges(n) if s < hi and e > lo]
        if not rs and ks is None:
            return range(lo, hi)
        return [j for j in range(lo, hi) if (ks is None or self._kind[j] in ks) and not self._hidden(j, rs)]

    def _write(self, kind: str, step: Any, stored: dict[str, Any], win: int) -> int:
        if step is None:
            step = self._steps[-1] if self._steps else 0
        _int(step, "step", 0, 2**62)
        i = len(self._off)
        t = max(self._t_last, round(time.monotonic() - self._clock0, 3))
        ev: dict[str, Any] = {"i": i, "kind": kind, "window": win, "step": step, "t": t, "agent": self.agent, **stored}
        line = _dumps(ev) + "\n"
        try:
            data = line.encode("utf-8")
        except UnicodeEncodeError:  # lone surrogates (undecodable kernel output): replaced, never a crash
            data = line.encode("utf-8", "replace")
        try:
            self._wf.write(data)
            self._wf.flush()
            self._xf.write(_REC.pack(self._size, step, win, kind.encode()))
            self._unflushed += 1
            if self._unflushed >= _FLUSH_INDEX:
                self._xf.flush()
                self._unflushed = 0
        except OSError as e:
            raise LedgerError(f"ledger {self.name!r}: write failed ({type(e).__name__})") from e
        self._push(self._size, step, win, kind)
        self._size += len(data)
        self._t_last = t
        return i

    def _store(self, k: str, v: Any) -> Any:
        if isinstance(v, str):
            return self._put_blob(v) if len(v) > INLINE_CHARS else v
        if _is_ref(v):
            sha = v.get("$blob")
            if set(v) - {"$blob", "chars", "bytes", "json"} or not isinstance(sha, str) or sha not in self._blobs:
                raise ValueError(f"field {k!r}: not a reference to a blob of this ledger (use ledger.ref(text))")
            return dict(v)
        try:
            s = _dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"field {k!r} is not JSON: {e}") from None
        if len(s) > INLINE_CHARS:
            return {**self._put_blob(s), "json": True}
        return v

    def _put_blob(self, text: str) -> dict[str, Any]:
        try:
            b = text.encode("utf-8")
        except UnicodeEncodeError:
            text = text.encode("utf-8", "replace").decode("utf-8")
            b = text.encode("utf-8")
        sha = hashlib.sha256(b).hexdigest()
        if sha in self._blobs:
            self.dedup_hits += 1
        else:
            p = self.dir / "blobs" / f"{sha}.txt"
            tmp = p.with_name(f"{sha}.{os.getpid()}.tmp")
            try:
                tmp.write_bytes(b)
                os.replace(tmp, p)  # a blob is whole or absent
            except OSError as e:
                raise LedgerError(f"ledger {self.name!r}: blob write failed ({type(e).__name__})") from e
            self._blobs.add(sha)
            self._blob_bytes += len(b)
        return {"$blob": sha, "chars": len(text), "bytes": len(b)}

    def _check_open(self) -> None:
        if self._closed:
            raise LedgerError(f"ledger {self.name!r} is closed")

    def flush(self) -> None:
        self._wf.flush()
        self._xf.flush()
        self._unflushed = 0

    def close(self) -> None:
        if self._closed:
            return
        for c in self._children.values():
            c.close()
        self._closed = True
        with contextlib.suppress(OSError):
            self._wf.close()
        with contextlib.suppress(OSError):
            self._xf.close()
        with contextlib.suppress(OSError):
            os.close(self._rfd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- reading --------------------------------------------------------------------------------------------------------
    def _norm(self, i: Any, n: int) -> int:
        _int(i, "i")
        j = i + n if i < 0 else i
        if not 0 <= j < n:
            raise ValueError(f"no event {i}: the ledger has {n} events (0..{n - 1})")
        if self._hidden(j, self._ranges(n)):
            raise ValueError(f"event {j}: this event was abandoned by a resume")
        return j

    def _raw(self, j: int) -> bytes:
        end = self._off[j + 1] if j + 1 < len(self._off) else self._size
        try:
            return self._pread(self._rfd, self._off[j], end - self._off[j])
        except OSError as e:
            raise LedgerError(f"ledger {self.name!r}: read failed ({type(e).__name__})") from e

    def _event(self, j: int) -> dict[str, Any]:
        try:
            return json.loads(self._raw(j))
        except ValueError:
            raise LedgerError(f"ledger {self.name!r}: event {j} is unreadable") from None

    def _blob_path(self, ref: dict[str, Any]) -> Path:
        sha = ref.get("$blob")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise LedgerError(f"ledger {self.name!r}: a malformed blob reference")
        return self.dir / "blobs" / f"{sha}.txt"

    def _blob_text(self, ref: dict[str, Any], cursor: int, n: int) -> str:
        """``n`` chars of a blob from char ``cursor``: a seek for ASCII blobs, a bounded decode-and-skip otherwise."""
        try:
            with open(self._blob_path(ref), "rb") as f:
                if ref.get("chars") == ref.get("bytes"):
                    f.seek(cursor)
                    return f.read(n).decode("utf-8", "replace")
                dec = codecs.getincrementaldecoder("utf-8")("replace")
                skip, out = cursor, ""
                while True:
                    chunk = f.read(CHUNK)
                    s = dec.decode(chunk, final=not chunk)
                    if skip >= len(s):
                        skip -= len(s)
                    else:
                        out += s[skip: skip + n - len(out)]
                        skip = 0
                    if len(out) >= n or not chunk:
                        return out
        except OSError as e:
            raise LedgerError(f"ledger {self.name!r}: blob unreadable ({type(e).__name__})") from e

    def _render(self, ev: dict[str, Any], cap: int) -> dict[str, Any]:
        out = {k: ev[k] for k in HEADER if k in ev}
        more: dict[str, int] = {}
        for k, v in ev.items():
            if k in out:
                continue
            if isinstance(v, str):
                out[k] = v[:cap]
                if len(v) > cap:
                    more[k] = len(v)
            elif _is_ref(v):
                total = int(v.get("chars") or 0)  # a JSON field this large comes back as its (capped) JSON text
                out[k] = self._blob_text(v, 0, cap)
                if total > cap:
                    more[k] = total
            else:
                out[k] = v
        if more:
            out["more"] = more
        return out

    def _preview(self, j: int, chars: int) -> dict[str, Any]:
        ev = self._event(j)
        out = {k: ev[k] for k in HEADER if k in ev}
        sizes: dict[str, int] = {}
        for k, v in ev.items():
            if k in out:
                continue
            if isinstance(v, str):
                out[k] = v[:chars]
                if len(v) > chars:
                    sizes[k] = len(v)
            elif _is_ref(v):
                out[k] = self._blob_text(v, 0, chars) if chars else ""
                sizes[k] = int(v.get("chars") or 0)
            else:
                s = _dumps(v)
                if len(s) <= chars:
                    out[k] = v
                else:
                    out[k] = s[:chars]
                    sizes[k] = len(s)
        if sizes:
            out["chars"] = sizes
        return out

    def _lim(self, upto: Any) -> int:
        """The view's length: every read sees events ``< upto`` only (None: all). Events below it never change, so a
        reply under a fixed ``upto`` is the same whenever it is computed (journal replay)."""
        return len(self) if upto is None else min(len(self), _int(upto, "upto", 0))

    def get(self, i: int, *, upto: int | None = None) -> dict[str, Any]:
        """Event ``i`` (negative counts from the end): header + fields, each text capped at 65,536 chars; ``more`` names
        the fields that continue (``read`` them)."""
        return self._render(self._event(self._norm(i, self._lim(upto))), GET_CHARS)

    def read(self, i: int, field: str = "text", cursor: int = 0, max_chars: int = 65_536, *, upto: int | None = None) -> dict[str, Any]:
        """``max_chars`` of one field of event ``i`` from char ``cursor``: ``{"text", "next", "total"}`` (``next`` is None at
        the end). A non-text field reads as its JSON."""
        j = self._norm(i, self._lim(upto))
        _int(cursor, "cursor", 0)
        _int(max_chars, "max_chars", 1, READ_MAX)
        ev = self._event(j)
        if not isinstance(field, str) or field in HEADER or field not in ev:
            names = [k for k in ev if k not in HEADER]
            raise ValueError(f"event {j} has no field {field!r}; its fields: {names}")
        v = ev[field]
        if _is_ref(v):
            total = int(v.get("chars") or 0)
            text = self._blob_text(v, cursor, max_chars) if cursor < total else ""
        else:
            s = v if isinstance(v, str) else _dumps(v)
            total, text = len(s), s[cursor: cursor + max_chars]
        end = cursor + len(text)
        return {"text": text, "next": end if end < total else None, "total": total}

    def _range(self, window: Any, n: int) -> tuple[int, int]:
        """The index range of window ``window`` (its number ``w``; negative counts back over the windows the view shows)."""
        if window is None:
            return 0, n
        nw = bisect.bisect_left(self._wstart, n)  # the windows opened within the view
        w = _int(window, "window")
        if nw == 0:
            return 0, 0
        if w < 0:
            vis = self._visible_windows(n)
            if -w > len(vis):
                raise ValueError(f"no window {window}: the ledger shows {len(vis)} windows")
            w = vis[w]
        if not 0 <= w < nw:
            raise ValueError(f"no window {window}: the ledger has windows 0..{nw - 1}")
        if self._hidden(self._wstart[w], self._ranges(n)):
            raise ValueError(f"window {w} was abandoned by a resume")
        return self._wstart[w], (self._wstart[w + 1] if w + 1 < nw else n)

    def _visible_windows(self, n: int) -> list[int]:
        rs = self._ranges(n)
        return [w for w in range(bisect.bisect_left(self._wstart, n)) if not self._hidden(self._wstart[w], rs)]

    def windows(self, *, start: int = 0, limit: int | None = None, upto: int | None = None) -> Page:
        """``[{w, reason, start_index, end_index, meta}]`` (end_index inclusive: the window's last event the view shows;
        windows a resume abandoned are left out, so ``w`` may skip numbers)."""
        n = self._lim(upto)
        nw, rows = bisect.bisect_left(self._wstart, n), []
        for w in self._visible_windows(n):
            lo, hi = self._wstart[w], (self._wstart[w + 1] if w + 1 < nw else n)
            vis = self._visible(lo, hi, n)
            rows.append({**self._windows[w], "end_index": vis[-1] if len(vis) else lo})
        return self._page(rows, start, None, limit, lambda r: r)

    @staticmethod
    def _page(seq: Any, start: int, end: int | None, limit: int | None, make: Any) -> Page:
        s0, s1, _ = slice(start, end).indices(len(seq))
        stop = s1 if limit is None else min(s1, s0 + limit)
        page = Page((make(x) for x in seq[s0:stop]), start=s0)
        page.next = stop if stop < s1 else None
        return page

    def messages(self, *, window: int | None = None, start: int = 0, end: int | None = None, kinds: list[str] | None = None,
                 preview_chars: int = 300, limit: int | None = None, upto: int | None = None) -> Page:
        """Previews of the events of one window (None: all; negative counts from the end), filtered by kind; ``start`` /
        ``end`` slice that selection (Python semantics). Each preview: header + every field's first ``preview_chars``
        chars; ``chars`` gives a cut field's full length."""
        n = self._lim(upto)
        lo, hi = self._range(window, n)
        ks = _kinds(kinds)
        _int(start, "start")
        _opt_int(end, "end")
        _int(preview_chars, "preview_chars", 0, MAX_PREVIEW)
        sel = self._visible(lo, hi, n, ks)
        return self._page(sel, start, end, limit, lambda j: self._preview(j, preview_chars))

    def user_messages(self, *, start: int = 0, limit: int | None = None, upto: int | None = None) -> Page:
        """Every user / supervisor / INBOX message, verbatim (each text up to 65,536 chars; ``more`` for the rest)."""
        n = self._lim(upto)
        sel = self._visible(0, n, n, USER_KINDS)
        return self._page(sel, _int(start, "start"), None, limit, lambda j: self._render(self._event(j), GET_CHARS))

    def since(self, step: int, *, start: int = 0, limit: int | None = None, preview_chars: int = 300, upto: int | None = None) -> Page:
        """Previews of the events at World step >= ``step``, in order."""
        _int(step, "step", 0)
        n = self._lim(upto)
        if self._mono:
            sel: Any = self._visible(bisect.bisect_left(self._steps, step, 0, n), n, n)
        else:
            rs = self._ranges(n)
            sel = [j for j in range(n) if self._steps[j] >= step and not self._hidden(j, rs)]
        _int(preview_chars, "preview_chars", 0, MAX_PREVIEW)
        return self._page(sel, _int(start, "start"), None, limit, lambda j: self._preview(j, preview_chars))

    def child(self, name: str) -> Ledger:
        """The ledger of child agent ``name`` (``<this dir>/<name>/``, same API; reopened if it exists)."""
        self._check_open()
        name = check_name(name)
        c = self._children.get(name)
        if c is None:
            agent = f"{self.agent}/{name}" if self.agent else name
            exists = (self.dir / name / "events.jsonl").exists()
            c = Ledger(self.dir, name=name, _agent=agent, _recover=exists)
            self._children[name] = c
        return c

    def stats(self) -> dict[str, Any]:
        children = sorted(p.name for p in self.dir.iterdir() if p.is_dir() and p.name != "blobs" and (p / "events.jsonl").exists())
        return {"events": len(self), "bytes": self._size, "windows": len(self._windows), "blobs": len(self._blobs),
                "blob_bytes": self._blob_bytes, "dedup_hits": self.dedup_hits, "children": children, "agent": self.agent,
                "abandoned": self._n_hidden(len(self)), "abandons": [{"i": at, "start": s, "end": e, "reason": r} for at, s, e, r in self._abandoned]}

    def _n_hidden(self, n: int) -> int:
        hidden, last = 0, 0
        for s, e in sorted(self._ranges(n)):
            s = max(s, last)
            if e > s:
                hidden, last = hidden + e - s, e
        return hidden

    def _view_stats(self, n: int) -> dict[str, Any]:
        """What the kernel's ``history.stats()`` gets: only numbers fixed by the view (blob and child counts grow later).
        ``events`` counts what the view shows; ``next_index`` is the index the next event gets."""
        return {"events": n - self._n_hidden(n), "next_index": n, "windows": len(self._visible_windows(n)), "agent": self.agent}

    # ---- search ---------------------------------------------------------------------------------------------------------
    def search(self, pattern: str, *, regex: bool = False, kinds: list[str] | None = None, window: int | None = None, limit: int = 20,
               context_chars: int = 200, before: int | None = None, upto: int | None = None, _floor: int | None = None) -> Page:
        """Events whose fields contain ``pattern`` (substring; ``regex=True`` for a Python regex), newest first, one hit per
        event: ``[{i, kind, window, step, agent, field, pos, snippet}]`` (``pos``: char offset in that field, for ``read``).
        Bounded: 2 s and 64 MiB per call; ``Page.next`` (pass as ``before``) continues with older events. ``_floor`` (replay
        only): search exactly the targets at index >= floor, with no time or byte bound."""
        if not isinstance(pattern, str) or not 1 <= len(pattern) <= MAX_PATTERN:
            raise ValueError(f"pattern must be a string of 1..{MAX_PATTERN} characters")
        if not isinstance(regex, bool):
            raise ValueError("regex must be True or False")  # noqa: TRY004 -- a bad request is the cell's ValueError
        ks = _kinds(kinds)
        _int(limit, "limit", 1, SEARCH_LIMIT)
        _int(context_chars, "context_chars", 0, MAX_CONTEXT)
        n = self._lim(upto)
        lo, hi = self._range(window, n)
        if before is not None:
            hi = min(hi, _int(before, "before", 0))
        bounded = _floor is None
        stop_at = lo if bounded else max(lo, _int(_floor, "floor", 0))
        if regex:
            try:
                re.compile(pattern)
            except (re.error, RecursionError, OverflowError) as e:
                raise ValueError(f"invalid regex: {e}") from None
        rs = [(s, e) for s, e in self._ranges(n) if s < hi and e > stop_at]
        targets = (j for j in range(hi - 1, stop_at - 1, -1) if (ks is None or self._kind[j] in ks) and not (rs and self._hidden(j, rs)))
        if regex:
            return self._search_regex(pattern, targets, lo, limit, context_chars, bounded)
        return self._search_sub(pattern, targets, lo, limit, context_chars, bounded)

    def _hit(self, j: int, field: str, pos: int, snippet: str) -> dict[str, Any]:
        return {"i": j, "kind": self._kind[j], "window": self._wins[j], "step": self._steps[j], "agent": self.agent,
                "field": field, "pos": pos, "snippet": snippet}

    def _search_sub(self, pattern: str, targets: Any, lo: int, limit: int, ctx: int, bounded: bool = True) -> Page:
        deadline = time.monotonic() + SEARCH_WALL_S - _MARGIN_S if bounded else float("inf")
        cap = SEARCH_BYTES if bounded else 1 << 62
        page = Page()
        page.floor = lo
        quick = _dumps(pattern)[1:-1] == pattern  # no JSON escapes: the raw line must contain the pattern's bytes
        pb = pattern.encode("utf-8", "replace")
        scanned, count = 0, 0
        for j in targets:
            if len(page) >= limit:
                page.reason, page.next, page.floor = "limit", j + 1, j + 1
                break
            if time.monotonic() > deadline:
                page.reason, page.next, page.timed_out, page.floor = "timeout", j + 1, True, j + 1
                break
            raw = self._raw(j)
            if scanned + len(raw) > cap and count:
                page.reason, page.next, page.floor = "scan_cap", j + 1, j + 1
                break
            scanned += len(raw)
            count += 1
            if quick and pb not in raw and b'"$blob"' not in raw:
                continue
            ev = json.loads(raw)
            for k, v in ev.items():
                if k in HEADER:
                    continue
                if _is_ref(v):
                    hit, scanned, stop = self._find_blob(v, pattern, ctx, scanned, deadline, cap)
                    if stop:
                        page.reason, page.timed_out, page.scanned_bytes, page.floor = stop, stop == "timeout", scanned, j + 1
                        if count == 1:  # this event alone exceeds a call's bound: skipped by next (read it directly)
                            page.incomplete, page.next = j, (j if j > lo else None)
                        else:  # cut mid-event: the next call starts with it
                            page.next = j + 1
                        return page
                else:
                    s = v if isinstance(v, str) else _dumps(v)
                    p = s.find(pattern)
                    hit = (p, s[max(0, p - ctx): p + len(pattern) + ctx]) if p >= 0 else None
                if hit:
                    page.append(self._hit(j, k, *hit))
                    break
        if page.next is not None and page.next <= lo:
            page.next = None
        page.scanned_bytes = scanned
        return page

    def _find_blob(self, ref: dict[str, Any], pattern: str, ctx: int, scanned: int, deadline: float,
                   cap: int) -> tuple[tuple[int, str] | None, int, str | None]:
        keep = len(pattern) - 1 + ctx
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        carry, base = "", 0  # base: char offset of the current chunk's start
        try:
            with open(self._blob_path(ref), "rb") as f:
                while True:
                    if time.monotonic() > deadline:
                        return None, scanned, "timeout"
                    room = cap - scanned
                    if room <= 0:
                        return None, scanned, "scan_cap"
                    chunk = f.read(min(CHUNK, room))
                    scanned += len(chunk)
                    s = dec.decode(chunk, final=not chunk)
                    buf = carry + s
                    p = buf.find(pattern)
                    if p >= 0:
                        return (base - len(carry) + p, buf[max(0, p - ctx): p + len(pattern) + ctx]), scanned, None
                    if not chunk:
                        return None, scanned, None
                    base += len(s)
                    carry = buf[-keep:] if keep else ""
        except OSError as e:
            raise LedgerError(f"ledger {self.name!r}: blob unreadable ({type(e).__name__})") from e

    def _search_regex(self, pattern: str, targets: Any, lo: int, limit: int, ctx: int, bounded: bool = True) -> Page:
        """Regex in a worker process killed at the deadline: Python's ``re`` backtracks without bound and holds the GIL."""
        t0 = time.monotonic()
        wall, cap = (SEARCH_WALL_S - _MARGIN_S, SEARCH_BYTES) if bounded else (REPLAY_TIMEOUT_S, 1 << 62)
        page = Page()
        page.floor = lo
        tl: list[list[int]] = []
        total, cut = 0, None
        for j in targets:
            ln = (self._off[j + 1] if j + 1 < len(self._off) else self._size) - self._off[j]
            if tl and total + ln > cap:
                cut = j
                break
            tl.append([j, self._off[j], ln])
            total += ln
        if not tl:
            return page
        job = json.dumps({"pattern": pattern, "limit": limit, "ctx": ctx, "max_bytes": cap, "targets": tl,
                          "events": str(self._events), "blobs": str(self.dir / "blobs")}).encode()
        with tempfile.TemporaryFile() as prog:  # the worker's progress: the event it is on, written through a shared mmap
            prog.write(b"\0" * 8)
            prog.flush()
            try:
                proc = subprocess.Popen([sys.executable, "-I", "-S", "-c", _REGEX_WORKER, str(prog.fileno())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env={}, pass_fds=(prog.fileno(),))
            except OSError as e:
                raise LedgerError(f"regex search could not start ({type(e).__name__})") from e
            timed_out = False
            try:
                out, _ = proc.communicate(job, timeout=max(0.05, wall - (time.monotonic() - t0)))
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                out, _ = proc.communicate()
            on = int.from_bytes(os.pread(prog.fileno(), 8, 0), "little")
        current, ended, cap_at = (on - 1 if on else None), False, None
        for line in out.splitlines():
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "m" in msg:
                j, field, pos, snip = msg["m"]
                page.append(self._hit(j, field, pos, snip))
            elif "cap" in msg:
                cap_at = msg["cap"]
            elif "end" in msg:
                ended = True
                page.scanned_bytes = msg["end"]
        if (timed_out or not ended) and not bounded:
            raise LedgerError("a replayed regex search did not finish")
        if timed_out or not ended:
            page.timed_out, page.reason = True, "timeout"
            if current is not None and page and page[-1]["i"] == current:  # killed just after a hit: that event was whole
                page.next = page.floor = current
            elif current is not None:
                page.incomplete, page.next, page.floor = current, current, current + 1  # the event it hung on: skipped by next
            else:  # nothing scanned (the worker did not get going): retry from the same place
                page.next = page.floor = tl[0][0] + 1
            if not page.scanned_bytes:
                page.scanned_bytes = sum(x[2] for x in tl if current is None or x[0] >= current)
        elif cap_at is not None:
            page.reason = "scan_cap"
            page.floor = cap_at + 1
            if cap_at == tl[0][0]:
                page.incomplete, page.next = cap_at, cap_at
            else:
                page.next = cap_at + 1
        elif len(page) >= limit:
            page.reason, page.next, page.floor = "limit", page[-1]["i"], page[-1]["i"]
        elif cut is not None:
            page.reason, page.next, page.floor = "scan_cap", cut + 1, cut + 1
        if page.next is not None and page.next <= lo:
            page.next = None
        return page

    # ---- kernel host requests -------------------------------------------------------------------------------------------
    def handle(self, request: dict[str, Any], *, upto: int | None = None) -> Any:
        """Serve one kernel host request ``{"type": "history.<op>", ...}``: validated strictly (``ValueError`` names the
        problem), replied with JSON of at most 256 KiB and no host path. A ledger fault raises ``LedgerError``. ``upto``
        (host-set, never a request key): the addressed ledger's view length; None = everything now."""
        led, op, kw = self._prepare(request)
        return led._serve(op, kw, led._lim(upto))

    async def ahandle(self, request: dict[str, Any], *, upto: int | None = None) -> Any:
        """``handle`` for an event loop: a search (up to 2 s) runs in a thread; everything else inline. The view is fixed
        before the thread starts, so appends meanwhile are never seen."""
        led, op, kw = self._prepare(request)
        n = led._lim(upto)
        if op == "search":
            return await asyncio.to_thread(led._serve, op, kw, n)
        return led._serve(op, kw, n)

    # ---- journal replay (B5): the kernel journal keeps a < 1 KiB marker instead of a reply of up to 256 KiB ------------
    def upto_for(self, request: dict[str, Any]) -> int:
        """The view length to serve ``request`` under now: the addressed (root or child) ledger's event count."""
        led, _, _ = self._prepare(request)
        return len(led)

    async def ahandle_recorded(self, request: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Serve a live request and return ``(reply, journal_marker)``: what the policy journals instead of the reply."""
        upto = self.upto_for(request)
        reply = await self.ahandle(request, upto=upto)
        return reply, self.journal_marker(request, upto, reply)

    def journal_marker(self, request: dict[str, Any], upto: int, reply: Any) -> dict[str, Any]:
        """A small record from which ``replay_reply`` rebuilds ``reply`` exactly: the request, the view length and, for a
        search, where the live call stopped (the time bound is not replayable; where it cut is)."""
        _, op, _ = self._prepare(request)
        marker: dict[str, Any] = {"history": dict(request), "upto": _int(upto, "upto", 0)}
        if op == "search":
            marker["stop"] = {k: reply.get(k) for k in _STOP_KEYS}
        return marker

    def replay_reply(self, marker: dict[str, Any]) -> Any:
        """The reply ``journal_marker`` recorded, rebuilt from the ledger (events below ``upto`` never change). A search
        re-scans exactly the events the live call searched in full (index >= its ``floor``), without a time bound, and
        restores where it stopped."""
        if not isinstance(marker, dict) or not isinstance(marker.get("history"), dict):
            raise LedgerError("not a history journal marker")
        led, op, kw = self._prepare(marker["history"])
        upto = _int(marker.get("upto"), "upto", 0)
        if upto > len(led):
            raise LedgerError(f"a journal marker names {upto} events; the ledger has {len(led)}")
        stop = marker.get("stop") if op == "search" else None
        if op == "search" and (not isinstance(stop, dict) or not isinstance(stop.get("floor"), int)):
            raise LedgerError("a search marker needs its stop record")
        return led._serve(op, kw, upto, stop)

    def _prepare(self, request: Any) -> tuple[Ledger, str, dict[str, Any]]:
        self._check_open()
        if not isinstance(request, dict):
            raise ValueError("a history request must be a dict")  # noqa: TRY004 -- a bad request is the cell's ValueError
        t = request.get("type")
        op = t[8:] if isinstance(t, str) and t.startswith("history.") else None
        if op not in _OPS:
            raise ValueError(f"unknown history request {t!r}; one of {', '.join('history.' + o for o in _OPS)}")
        kw = {k: v for k, v in request.items() if k != "type"}
        bad = set(kw) - _OPS[op] - {"child"}
        if bad:
            raise ValueError(f"history.{op} got unexpected argument(s) {sorted(bad)}; it takes {sorted(_OPS[op])}")
        led = self._resolve(kw.pop("child", None))
        return led, op, kw

    def _resolve(self, path: Any) -> Ledger:
        if path is None:
            return self
        if not isinstance(path, str) or not path or len(path) > 512:
            raise ValueError("child must be a child agent's name (or 'a/b' for a grandchild)")
        led = self
        for seg in path.split("/"):
            check_name(seg)
            if seg not in led._children and not (led.dir / seg / "events.jsonl").exists():
                raise ValueError(f"no child ledger {path!r}")
            led = led.child(seg)
        return led

    def _serve(self, op: str, kw: dict[str, Any], n: int, stop: dict[str, Any] | None = None) -> Any:
        kw = dict(kw)
        try:
            if op == "get":
                return _shrink(self.get(kw.get("i"), upto=n), REPLY_CAP - 64, "more")
            if op == "read":
                cursor = kw.get("cursor", 0)
                r = self.read(kw.get("i"), kw.get("field", "text"), cursor, kw.get("max_chars", 65_536), upto=n)
                return _fit_read(r, cursor)
            if op == "stats":
                return self._view_stats(n)
            if op == "search":
                for k in ("kinds", "window", "before"):
                    kw.setdefault(k, None)
                if not isinstance(kw.get("pattern"), str):
                    raise ValueError("search needs a pattern string")
                page = self.search(kw.pop("pattern"), **kw, upto=n, _floor=None if stop is None else stop["floor"])
                if stop is not None:  # replay: where the live call stopped
                    page.next, page.timed_out, page.reason = stop.get("next"), bool(stop.get("timed_out")), stop.get("reason")
                    page.scanned_bytes, page.incomplete, page.floor = stop.get("scanned_bytes") or 0, stop.get("incomplete"), stop["floor"]
                return _envelope(page, lambda keep: page[keep - 1]["i"], "chars", search=True)
            start, marks = kw.get("start", 0), "chars"
            if op == "windows":
                page = self.windows(start=_int(start, "start"), limit=PAGE, upto=n)
            elif op == "messages":
                page = self.messages(window=kw.get("window"), start=start, end=kw.get("end"), kinds=kw.get("kinds"),
                                     preview_chars=kw.get("preview_chars", 300), limit=PAGE, upto=n)
            elif op == "user_messages":
                page, marks = self.user_messages(start=start, limit=PAGE, upto=n), "more"
            else:  # since
                page = self.since(kw.get("step"), start=start, limit=PAGE, preview_chars=kw.get("preview_chars", 300), upto=n)
            return _envelope(page, lambda keep: page.start + keep, marks)
        except (ValueError, LedgerError):
            raise
        except OSError as e:
            raise LedgerError(f"history.{op}: the ledger could not be read ({type(e).__name__})") from None


_OPS: dict[str, frozenset[str]] = {
    "get": frozenset({"i"}),
    "read": frozenset({"i", "field", "cursor", "max_chars"}),
    "search": frozenset({"pattern", "regex", "kinds", "window", "limit", "context_chars", "before"}),
    "windows": frozenset({"start"}),
    "messages": frozenset({"window", "start", "end", "kinds", "preview_chars"}),
    "user_messages": frozenset({"start"}),
    "since": frozenset({"step", "start", "preview_chars"}),
    "stats": frozenset(),
}
_STOP_KEYS = ("next", "timed_out", "reason", "scanned_bytes", "incomplete", "floor")


def _shrink(d: dict[str, Any], room: int, marks: str) -> dict[str, Any]:
    """Cut the longest fields until ``d`` fits ``room`` wire bytes. A structured field (dict / list: a ``world_call``'s
    args, a result) becomes its JSON text first, cut like a text field; ``marks`` (``more`` / ``chars``) names each cut
    field's full length, so ``read(i, field=...)`` pages the rest. Never fails for a condition the policy causes: the
    header fields are small, and everything else can be cut to 16 chars (H1: a 60k-CJK args dict is 360 KiB on the wire)."""
    while True:
        size = _wire(d)
        if size <= room:
            return d
        best, best_len = None, 16
        for k, v in d.items():
            if k in HEADER or k == marks:
                continue
            if not isinstance(v, str):
                if isinstance(v, (dict, list)):
                    v = _dumps(v)  # the JSON ``read(i, field=k)`` returns
                    d[k] = v
                else:
                    continue
            if len(v) > best_len:
                best, best_len = k, len(v)
        if best is None:
            break
        v = d[best]
        d.setdefault(marks, {}).setdefault(best, len(v))
        d[best] = v[: max(16, min(len(v) // 2, len(v) - (size - room)))]
    raise LedgerError("a history reply could not be fitted under the reply cap (internal: only header fields remained)")


def _fit_read(r: dict[str, Any], cursor: int) -> dict[str, Any]:
    room = REPLY_CAP - 128
    text = r["text"]
    while _wire(text) > room:  # escapes (control chars, non-ASCII) can grow a char to 6 wire bytes
        text = text[: max(1, min(len(text) // 2, len(text) - (_wire(text) - room)))]
    if len(text) < len(r["text"]):
        r = {**r, "text": text, "next": cursor + len(text)}
    return r


def _envelope(page: Page, next_after: Any, marks: str, *, search: bool = False) -> dict[str, Any]:
    env: dict[str, Any] = {"items": [], "next": page.next, "truncated": False}
    if search:
        env.update(timed_out=page.timed_out, reason=page.reason, scanned_bytes=page.scanned_bytes, incomplete=page.incomplete, floor=page.floor)
    room = REPLY_CAP - _wire(env) - 64
    keep, total = 0, 0
    for x in page:
        s = _wire(x) + 2
        if total + s > room:
            if keep == 0:  # one item alone is too large: cut its texts
                x = _shrink(dict(x), room - 2, marks)
                s = _wire(x) + 2
                env["items"].append(x)
                keep, total = 1, s
            break
        env["items"].append(x)
        keep, total = keep + 1, total + s
    if keep < len(page):
        env["truncated"] = True
        env["next"] = next_after(keep)
    return env


# The regex worker (stdlib only; run with -I -S and an empty environment). It reads the targets newest first, writes the
# event it is on into a shared 8-byte mmap (no syscall) so the host knows which one a hung pattern was on, {"m": [...]} per hit,
# {"cap": i} at the byte budget and {"end": scanned} when done. The host kills it at the deadline.
_REGEX_WORKER = r'''
import json, mmap, os, re, sys
job = json.loads(sys.stdin.buffer.read())
prog = mmap.mmap(int(sys.argv[1]), 8)
rx = re.compile(job["pattern"])
ctx, limit, budget = job["ctx"], job["limit"], job["max_bytes"]
HEADER = {"i", "kind", "window", "step", "t", "agent"}
out = sys.stdout
def emit(o):
    out.write(json.dumps(o) + "\n")
    out.flush()
fd = os.open(job["events"], os.O_RDONLY)
scanned, found = 0, 0
for i, off, ln in job["targets"]:
    if scanned + ln > budget and scanned:
        emit({"cap": i})
        break
    prog[:8] = (i + 1).to_bytes(8, "little")
    raw = os.pread(fd, ln, off)
    scanned += len(raw)
    ev = json.loads(raw)
    hit = None
    for k, v in ev.items():
        if k in HEADER:
            continue
        if isinstance(v, dict) and "$blob" in v:
            sha = str(v["$blob"])
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                continue
            n = int(v.get("bytes") or 0)
            if scanned + n > budget:
                emit({"cap": i})
                emit({"end": scanned})
                sys.exit(0)
            with open(os.path.join(job["blobs"], sha + ".txt"), "rb") as f:
                s = f.read().decode("utf-8", "replace")
            scanned += n
        elif isinstance(v, str):
            s = v
        else:
            s = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        m = rx.search(s)
        if m:
            p = m.start()
            hit = [i, k, p, s[max(0, p - ctx): m.end() + ctx][: 2 * ctx + 2000]]
            break
    if hit:
        emit({"m": hit})
        found += 1
        if found >= limit:
            break
emit({"end": scanned})
'''


# The kernel half (F appends it to the RLM bootstrap). ``host`` is the kernel's host-request function.
HISTORY_API = r'''
class _HistoryPage(list):
    """A page of ledger events. .next: pass it back as start= (before= for search) for more; None when nothing is left.
    Search pages also carry .timed_out, .reason (limit | scan_cap | timeout), .scanned_bytes and .incomplete (an event
    the time or size bound cut: read it with OUT(i))."""
    next = None
    truncated = False
    timed_out = False
    reason = None
    scanned_bytes = 0
    incomplete = None


class _HistoryText(dict):
    """A chunk of one event field: ["text"], ["next"] (the cursor for the rest; None at the end), ["total"] chars.
    Printing it shows the text."""
    _i = None
    _field = "text"
    def __str__(self):
        t = self.get("text", "")
        if self.get("next") is not None:
            more = self["total"] - self["next"]
            call = (f"OUT({self._i}, cursor={self['next']})" if self._field == "text"
                    else f"history.read({self._i}, cursor={self['next']}, field={self._field!r})")
            t += f"\n[... {more} more chars: await {call}]"
        return t
    __repr__ = __str__


class _History:
    """Your episode's ledger, on disk outside this kernel: every message, every cell's full output, every World result,
    summary and child event, by a stable index i. Every method is async: `await history.search("refund")`."""
    def __init__(self, child=None):
        self._child = child
    async def _ask(self, op, **kw):
        if self._child is not None:
            kw["child"] = self._child
        return await host("history." + op, **kw)
    @staticmethod
    def _page(r):
        p = _HistoryPage(r.get("items") or [])
        for k in ("next", "truncated", "timed_out", "reason", "scanned_bytes", "incomplete"):
            if k in r:
                setattr(p, k, r[k])
        return p
    async def get(self, i):
        """Event i (negative counts from the end); texts capped at 65,536 chars, "more" names the fields that continue."""
        return await self._ask("get", i=i)
    async def read(self, i, cursor=0, max_chars=65536, field="text"):
        """max_chars of one field of event i from cursor: ["text"], ["next"], ["total"]."""
        r = _HistoryText(await self._ask("read", i=i, field=field, cursor=cursor, max_chars=max_chars))
        r._i, r._field = i, field
        return r
    async def search(self, pattern, regex=False, kinds=None, window=None, limit=20, context_chars=200, before=None):
        """Events containing pattern (substring; regex=True for a regex), newest first: [{i, kind, window, step, field,
        pos, snippet}]. Bounded per call (2 s, 64 MiB): continue older with before=page.next."""
        return self._page(await self._ask("search", pattern=pattern, regex=regex, kinds=None if kinds is None else list(kinds),
                                          window=window, limit=limit, context_chars=context_chars, before=before))
    async def windows(self, start=0):
        """Context windows: [{w, reason, start_index, end_index, meta}]."""
        return self._page(await self._ask("windows", start=start))
    async def messages(self, window=-1, start=0, end=None, kinds=None, preview_chars=300):
        """Previews of one window's events (-1: the current one; None: all), sliced by start/end."""
        return self._page(await self._ask("messages", window=window, start=start, end=end,
                                          kinds=None if kinds is None else list(kinds), preview_chars=preview_chars))
    async def user_messages(self, start=0):
        """Every user / supervisor / INBOX message, verbatim."""
        return self._page(await self._ask("user_messages", start=start))
    async def since(self, step, start=0):
        """Previews of the events at World step >= step."""
        return self._page(await self._ask("since", step=step, start=start))
    async def stats(self):
        return await self._ask("stats")
    def child(self, name):
        """The ledger of child agent `name` (same API)."""
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError("child(name) takes one child agent's name")
        return _History(name if self._child is None else self._child + "/" + name)
    def __repr__(self):
        return ("<history" + ("" if self._child is None else f" of {self._child}") + ": await .search(pattern), .get(i), .read(i), "
                ".messages(), .windows(), .user_messages(), .since(step), .child(name)>")


history = _History()


async def OUT(i, cursor=0, max_chars=65536):
    """The full output of event i (a cell's whole stdout, stderr, result and error), in pages: print(await OUT(i))."""
    return await history.read(i, cursor=cursor, max_chars=max_chars)
'''
