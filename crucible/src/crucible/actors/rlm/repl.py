"""The Crucible RLM kernel (m14): a persistent Python REPL driven over NDJSON. Stdlib only, so it can be pushed into any
sandbox and run with ``python3 repl.py``.

Ported in part from Prime Agent ``prime-agent-runtime/src/rlm/repl.py`` (protocol v3; MIT, Copyright (c) 2025-2026 Prime
Intellect Ltd.; notice in ``crucible/THIRD_PARTY.md``): the I/O isolation (a private dup of fd 1 carries the protocol,
fds 1 and 2 are pipes pumped into the running cell's output, with drain fences), the payload and result caps with
truncation markers, protocol errors that keep the kernel serving, the interrupt, and the owner watchdog.

**I/O isolation (M14 audit H3 V2, G1 D12).** Only kernel code writes protocol frames, to a private descriptor. Cell code
that writes to fd 1 (``print``, ``sys.__stdout__``, ``os.write(1, ...)``, a subprocess) writes into a pipe whose bytes
become the cell's ``stdout``: a forged ``done`` is just text. The host still validates every frame
(``kernel.Kernel._read``), because a cell runs in this process and can reach any descriptor.

**Frame authentication (M14 campaign B10).** A cell read ``_protocol_fd`` and the running cell's id from
``sys.modules['__main__']`` and wrote a well-formed ``done``: the host ended the cell on it and journaled the forged digest.
Now the host hands the kernel a fresh per-session key in its first request (``hello``), before any cell exists, and every
frame carries an HMAC-SHA256 over its canonical JSON; the host drops a frame whose MAC is missing or wrong (a protocol
error on the cell). The key lives only inside the signer's closure (an HMAC object; the key bytes are not kept), no module
global names the signer, the protocol descriptor or the stderr descriptor, and the signer handed to ``host()`` can only
build ``host_request`` frames (which a cell may send anyway). What this does not claim: cell code runs in this process,
so a cell that walks the interpreter (``gc``, frames of suspended tasks) can still reach the full signer. Whatever the
kernel reports is therefore treated as the policy's own output: it can end or spoil only the policy's own episode
(scored), never mask it, and never reach another episode.

Requests (one JSON object per line on the request channel, fd 0 at start; fd 0 is then ``/dev/null`` for cells):
- ``{"type":"execute","id":ID,"code":SRC,"timeout_s":T}``: run a cell (top-level ``await`` allowed, one persistent
  namespace and event loop). Ends with exactly one ``done`` event. ``T`` counts the cell's own time: time spent waiting
  on host requests (World steps, sub-LLMs) is not counted (G1 D14).
- ``{"type":"host_reply","id":RID,"data":ANY}``: the answer to a ``host_request`` the running cell raised.
- ``{"type":"interrupt","id":ID}``: stop the running cell, keeping the namespace (cancel, then SIGINT to a blocked loop).
- ``{"type":"digest","id":ID}`` / ``{"type":"names","id":ID}``: the full namespace digest / live names.
- ``{"type":"describe","id":ID,"limit":N}``: ``[{name,type,size,unit,boot}]`` for the python-state row (M16 §3.4).
- ``{"type":"snapshot","id":ID,"max_bytes":B,"max_var_bytes":V}``: dill the namespace and stream it back (below).
- ``{"type":"restore_chunk","id":ID,"seq":N,"data":B64}`` ... then ``{"type":"restore","id":ID,"sha256":HEX}``: load a
  snapshot blob into this (fresh, bootstrapped) kernel.
- ``{"type":"shutdown"}``: exit, killing the kernel's process group (threads and children a cell started).

``execute`` may carry ``"bootstrap": true`` (a host-written cell: the objects it binds are re-created by re-running it,
never snapshotted by value) and ``"entropy": {...}`` (the clock and random bytes the cell read when it first ran, served
again on replay).

Events: ``ready{protocol,python,pid,pgid}``, ``started{id}`` (the cell's code is about to run: a kernel lost before it is
"lost between cells", M16 B10), ``host_request{id,cell,data}``, ``done{id,stdout,stderr,result,error,timed_out,finished,
digest,entropy}``, ``snapshot_chunk{id,seq,data}``, ``protocol_error{id,error}`` (a malformed request; the kernel keeps
serving).

**Snapshots (M16 §3.4, R12, H11; Prime Agent ``repl.py:40-41,771-875,1070-1092``, MIT).** ``snapshot`` dill-serializes
the namespace into one pickle stream (one memo, so two names that share an object still share it after a restore):
every non-dunder name; ``NS`` itself (a function's globals) and the objects a bootstrap cell bound (functions, classes,
``world``, ``rlm``, ``host``) are persistent references, re-created by re-running the bootstrap before the load. Each
name is bounded at ``max_var_bytes`` (16 MiB; names the bootstrap bound, e.g. ``OBS``, only by the total) and the stream
at ``max_bytes`` (256 MiB); a name that fails or exceeds is rolled back out of the stream (bytes and memo) and reported
in ``lost`` with the reason. The blob travels as ``snapshot_chunk`` frames (4 MiB each, signed like every frame) and the
final ``done`` reports its sha256, the full digest, and the digest over the saved names only. The host writes the bytes
and never unpickles them; only a fresh kernel (the same sandbox) loads them, through an unpickler that refuses to reopen
raw file descriptors (Prime Agent's guard). ``random``'s state travels in the blob; dill is pushed by the host as a
zip that the kernel loads into memory and removes at start (``_load_pylib``). Not in a snapshot: files a cell wrote,
``os.environ`` / cwd changes, other modules' globals, threads and subprocesses (dill re-creates a lock unlocked).

**Digests (M16 B4).** The per-cell ``done.digest`` is a bounded fingerprint (names, types, lengths, scalars, short strings
in full, long top-level strings hashed once per object, containers sampled at both ends, at most ``FP_BUDGET`` nodes):
O(names), not O(everything the episode has seen) (18.6 ms per trivial cell at the end of a 1,914-cell episode, B4). The
full ``digest`` (snapshot points, restore checks, on request) is complete: each name's value is hashed whole (its pickle
stream, or a structural walk for what does not pickle), so a change past the old 100k-character cut is seen.

**Replayable clock and entropy (M16 B4).** While a cell runs, ``time.time``, ``time.time_ns`` and ``os.urandom`` (so
``uuid.uuid4``, ``secrets``, ``random.SystemRandom``, ``tempfile`` names) are recorded in the main thread (at most 4,096
reads and 64 KiB per cell) and travel in ``done.entropy``; a replayed cell is served the recorded values in order, so a
cell that keeps ``time.time()`` or ``uuid4()`` replays exactly. ``datetime.now()`` / ``date.today()`` read the C clock
directly and are not recorded: a cell that keeps them diverges and the host falls back to a lenient replay.

Inside cells: ``await host(type, **payload)`` is the only way out of the kernel (World tools, sub-LLMs, the forge facade
are thin wrappers over it, installed by the host's bootstrap cell). ``FINISHED`` set by ``rlm.finish`` travels in
``done``. The kernel seeds ``random`` from ``CRUCIBLE_RLM_SEED`` so a replayed journal reproduces the namespace.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import codecs
import contextlib
import contextvars
import hashlib
import io
import itertools
import json
import math
import os
import pickle
import random
import re
import signal
import sys
import threading
import time
import traceback
import types

PROTOCOL = 3  # 3: authenticated frames after a ``hello`` key exchange (M14 campaign B10)
MAX_STREAM = 1_048_576  # characters per stream per cell
RESULT_CAP = 1_048_576  # characters of a result repr or an error text
PAYLOAD_CAP = 16 * 1024 * 1024  # one host request's JSON
REQUEST_CAP = 64 * 1024 * 1024  # one request line from the host
OWNER_GRACE_S = 5.0  # after the request channel closes, how long a clean shutdown may take before the group is killed

_write_lock = threading.Lock()
_cell = contextvars.ContextVar("crucible_rlm_cell", default=None)
_cells: dict = {}
_active: dict = {"cid": None, "task": None, "cancel_ran": False, "interrupt": None}
_background: list = []
_pending: dict = {}
_loop = None
_queue = None
_main_thread = threading.main_thread()
# The namespace is a module's dict (M16): dill pickles a cell function's globals as a reference to ``__main__.__dict__``
# when ``sys.modules["__main__"]`` is this module (swapped in only while a snapshot is written or loaded), so a restored
# function sees the live namespace, not a frozen copy. ``sys.modules["__main__"]`` stays the kernel script otherwise.
_NS_MODULE = types.ModuleType("__main__")
NS: dict = _NS_MODULE.__dict__
NS["__builtins__"] = __builtins__
# The kernel's own entropy, never recorded or replayed (request ids, drain fences)
_real_time, _real_time_ns, _real_urandom = time.time, time.time_ns, os.urandom
SNAPSHOT_MAGIC = b"CRUCIBLE-RLM-SNAPSHOT-1\n"
SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024  # Prime Agent repl.py:40 (DEFAULT_SNAPSHOT_MAX_BYTES)
SNAPSHOT_MAX_VAR_BYTES = 16 * 1024 * 1024  # Prime Agent repl.py:41 (DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES)
SNAPSHOT_CHUNK = 4 * 1024 * 1024  # raw bytes per snapshot_chunk frame (base64 inside the frame)
RANDOM_KEY = "__crucible_random_state__"  # a dunder: never a saved name
FP_BUDGET = 2_000  # nodes one per-cell fingerprint may visit
FP_STR = 4_096  # strings up to this length are fingerprinted whole
ENTROPY_MAX_READS = 4_096
ENTROPY_MAX_BYTES = 65_536


class CellInterrupted(BaseException):
    """Raised in a blocked cell by the host's interrupt."""


def _clean(x):
    """No lone surrogates leave the kernel (a forged one broke the host's chat codec, H3 V2)."""
    if isinstance(x, str):
        try:
            x.encode("utf-8")
            return x
        except UnicodeEncodeError:
            return x.encode("utf-8", "replace").decode("utf-8")
    if isinstance(x, dict):
        return {str(_clean(k)): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _finite(x):
    """Non-finite floats as text: the channel carries strict JSON only."""
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def canonical(obj) -> bytes:
    """The bytes a frame's MAC covers (the host computes the same from the parsed frame)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def _make_signer(key: bytes, fd: int, err_fd: int):
    """The only writer of protocol frames: every frame normalized to plain JSON, MAC'd with the session key and written in
    one locked call. The key is consumed into an HMAC object held by this closure only."""
    import hashlib as _hashlib
    import hmac as _hmac

    base = _hmac.new(bytes(key), digestmod=_hashlib.sha256)
    del key

    def send(obj: dict) -> None:
        norm = json.loads(json.dumps(_finite(_clean(obj)), default=repr))  # exactly what the host will parse
        m = base.copy()
        m.update(canonical(norm))
        norm["mac"] = m.hexdigest()
        data = (json.dumps(norm, ensure_ascii=False) + "\n").encode("utf-8")
        with _write_lock:
            view = memoryview(data)
            try:
                while view:
                    view = view[os.write(fd, view):]
            except OSError as e:
                try:
                    os.write(err_fd, f"rlm.repl: dropped protocol frame: {e}\n".encode())
                except OSError:
                    pass

    return send


class _CellState:
    def __init__(self, cid: str):
        self.cid = cid
        self.out = {"stdout": io.StringIO(), "stderr": io.StringIO()}
        self.cut = {"stdout": 0, "stderr": 0}
        self.pending = 0  # outstanding host requests
        self.wait_started = 0.0
        self.host_wait = 0.0  # seconds spent waiting on the host (not counted against the deadline)

    def write(self, stream: str, s: str) -> None:
        b = self.out[stream]
        room = max(MAX_STREAM - b.tell(), 0)
        if room:
            b.write(s[:room])
        if len(s) > room:
            self.cut[stream] += len(s) - room

    def text(self, stream: str) -> str:
        t = self.out[stream].getvalue()
        if self.cut[stream]:
            t += f"\n[... {stream} truncated: {self.cut[stream]} more characters were not kept ...]"
        return t


def _write_out(stream: str, s: str, cid=None) -> None:
    st = _cells.get(cid) if cid is not None else None
    if st is None and _active["cid"] is not None:
        st = _cells.get(_active["cid"])
    if st is not None:
        st.write(stream, s)
    elif sum(len(x) for x in _background) < MAX_STREAM:
        _background.append(s[:MAX_STREAM])


class _Writer(io.TextIOBase):
    """``sys.stdout`` / ``sys.stderr``: text goes to the cell that wrote it (the task context), else the running cell."""

    def __init__(self, stream: str):
        self.stream = stream

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        s = str(s)
        _write_out(self.stream, s, _cell.get())
        return len(s)

    def flush(self) -> None:
        pass


class _Pump:
    """Reads one captured fd pipe (fd 1 or 2 of the kernel) into the running cell's output (Prime Agent ``_Pump``)."""

    def __init__(self, read_fd: int, write_fd: int, stream: str):
        self._read_fd = read_fd
        self._token_fd = os.dup(write_fd)  # private write end: a cell closing fd 1 cannot break the drain fence
        os.set_inheritable(self._token_fd, False)
        self.stream = stream
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._lock = threading.Lock()
        self._watch = None
        self._buf = b""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def drain(self) -> None:
        """Block until every byte written to the fd so far has reached the cell's buffer."""
        token = b"\xff<drain:" + _real_urandom(16).hex().encode() + b">\xff"
        seen = threading.Event()
        with self._lock:
            self._watch = (token, seen)
        try:
            os.write(self._token_fd, token)
            for _ in range(50):
                if seen.wait(0.1) or not self._thread.is_alive():
                    return
        except OSError:
            return
        finally:
            with self._lock:
                self._watch = None

    def _run(self) -> None:
        while True:
            try:
                chunk = os.read(self._read_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            self._feed(chunk)

    def _feed(self, chunk: bytes) -> None:
        data = self._buf + chunk
        self._buf = b""
        with self._lock:
            watch = self._watch
        if watch is None:
            self._emit(data)
            return
        token, seen = watch
        while True:
            i = data.find(token)
            if i == -1:
                break
            self._emit(data[:i])
            seen.set()
            data = data[i + len(token):]
        hold = 0
        for k in range(min(len(data), len(token) - 1), 0, -1):
            if data.endswith(token[:k]):
                hold = k
                break
        if hold:
            self._buf = data[len(data) - hold:]
            data = data[: len(data) - hold]
        self._emit(data)

    def _emit(self, data: bytes) -> None:
        if data:
            text = self._decoder.decode(data)
            if text:
                _write_out(self.stream, text)


_pumps: list = []


def _make_host(send):
    """``host`` for cells, over a sender that can only build ``host_request`` frames."""

    def request(rid: str, cell, data: dict) -> None:
        send({"event": "host_request", "id": str(rid), "cell": cell, "data": data})

    async def host(type: str, **payload):
        return await _host(request, type, **payload)

    host.__doc__ = _host.__doc__
    return host


async def _host(request, type: str, **payload):
    """Ask the host (the policy / manager process) to do something; returns its reply's data (raises on an error reply)."""
    data = {"type": type, **payload}
    try:
        size = len(json.dumps(data, default=repr, allow_nan=False))
    except ValueError as e:
        raise ValueError(f"host request {type!r}: the payload is not JSON (NaN or infinity): {e}") from None
    if size > PAYLOAD_CAP:
        raise ValueError(f"host request {type!r}: the payload exceeds the {PAYLOAD_CAP}-character cap ({size})")
    rid = _real_urandom(16).hex()
    fut = asyncio.get_running_loop().create_future()
    _pending[rid] = fut
    st = _cells.get(_cell.get())
    if st is not None:
        if st.pending == 0:
            st.wait_started = time.monotonic()
        st.pending += 1
    try:
        request(rid, _cell.get(), data)
        reply = await fut
    finally:
        _pending.pop(rid, None)
        if st is not None:
            st.pending -= 1
            if st.pending == 0:
                st.host_wait += time.monotonic() - st.wait_started
    if isinstance(reply, dict) and reply.get("__error__"):
        raise RuntimeError(reply["__error__"])
    return reply


_SKIP = ("__builtins__", "host", "asyncio")
_BOOT: dict = {}  # name -> object a bootstrap cell bound (re-created by re-running the bootstrap; a persistent reference)
_BOOT_IDS: dict = {}  # id(object) -> name
_BOOT_NAMES: set = set()  # every name a bootstrap cell bound (data included: OBS, INBOX, HISTORY, TASK, TOOLS)
_FP_CACHE: dict = {}  # name -> (str/bytes object, its sha256): an immutable top-level value is hashed once per object
_SAFE_REPR = ("datetime", "decimal", "fractions", "uuid", "pathlib", "ipaddress")  # top-level module of the value's type
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _public() -> list:
    return sorted(k for k in NS if not k.startswith("_") and k not in _SKIP)


def names() -> list:
    return _public()


def _nondata(v) -> bool:
    """An object the bootstrap re-creates (code, a class, a module, a harness object): never snapshotted by value."""
    if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, type, types.ModuleType)):
        return True
    return type(v).__module__ == "__main__" and not isinstance(v, (dict, list, set, tuple, str, bytes))


def _register_boot() -> None:
    """After a bootstrap cell: remember what it bound."""
    for k, v in list(NS.items()):
        if k.startswith("__"):
            continue
        _BOOT_NAMES.add(k)
        if _nondata(v):
            _BOOT[k] = v
            _BOOT_IDS[id(v)] = k


def _boot_unchanged(k: str) -> bool:
    b = _BOOT.get(k, _BOOT)
    return b is not _BOOT and NS.get(k) is b


# ---- the per-cell fingerprint (bounded; M16 B4) ----------------------------------------------------------------------
def _sk(h, v, budget: list, depth: int) -> None:
    """A deterministic, bounded sketch of ``v`` (sampled containers; no user ``__repr__`` / ``__len__`` runs)."""
    budget[0] -= 1
    t = type(v)
    h.update(t.__qualname__.encode("utf-8", "replace") + b"|")
    if v is None or t is bool or t is float or t is complex:
        h.update(repr(v).encode())
        return
    if t is int:
        h.update(hex(v).encode())  # no int-to-decimal digit limit, linear
        return
    if t is str:
        n = len(v)
        h.update(b"%d:" % n)
        h.update((v if n <= FP_STR else v[:1024] + v[-1024:]).encode("utf-8", "surrogatepass"))
        return
    if t is bytes or t is bytearray:
        n = len(v)
        h.update(b"%d:" % n)
        h.update(bytes(v) if n <= FP_STR else bytes(v[:1024]) + bytes(v[-1024:]))
        return
    if t.__module__.partition(".")[0] in _SAFE_REPR:  # stdlib value types: their repr is their value (a stored datetime / UUID is seen)
        with contextlib.suppress(Exception):
            h.update(repr(v).encode("utf-8", "replace")[:FP_STR])
        return
    if isinstance(v, (list, tuple)):
        n = (list.__len__ if isinstance(v, list) else tuple.__len__)(v)
        h.update(b"[%d" % n)
        if budget[0] > 0 and depth < 6:
            get = list.__getitem__ if isinstance(v, list) else tuple.__getitem__
            for i in (range(n) if n <= 16 else itertools.chain(range(8), range(n - 8, n))):
                _sk(h, get(v, i), budget, depth + 1)
        return
    if isinstance(v, dict):
        n = dict.__len__(v)
        h.update(b"{%d" % n)
        if budget[0] > 0 and depth < 6:
            items = dict.items(v)
            it = items if n <= 16 else itertools.chain(itertools.islice(items, 8), reversed(list(itertools.islice(reversed(items), 8))))
            for k, x in it:
                _sk(h, k, budget, depth + 1)
                _sk(h, x, budget, depth + 1)
        return
    if isinstance(v, (set, frozenset)):
        n = (set.__len__ if isinstance(v, set) else frozenset.__len__)(v)
        h.update(b"<%d" % n)
        if n <= 32 and budget[0] > 0 and depth < 6:
            parts = []
            for x in v:
                hx = hashlib.sha256()
                _sk(hx, x, budget, depth + 1)
                parts.append(hx.digest())
            for p in sorted(parts):
                h.update(p)
        return
    if t.__module__ == "__main__" and budget[0] > 0 and depth < 6:  # an instance of a cell's class: its attributes
        try:
            d = object.__getattribute__(v, "__dict__")
        except Exception:  # noqa: BLE001 -- no instance dict (slots, a C object)
            d = None
        if type(d) is dict:
            _sk(h, d, budget, depth + 1)


def fingerprint() -> str:
    """The per-cell digest (``done.digest``): O(names + FP_BUDGET), deterministic across kernels with the same seed."""
    h = hashlib.sha256()
    budget = [FP_BUDGET]
    live = _public()
    for k in live:
        v = NS[k]
        h.update(k.encode("utf-8", "surrogatepass") + b"=")
        t = type(v)
        if t is str or t is bytes:
            c = _FP_CACHE.get(k)
            if c is None or c[0] is not v:
                c = _FP_CACHE[k] = (v, hashlib.sha256(v.encode("utf-8", "surrogatepass") if t is str else v).digest())
            h.update(t.__name__.encode() + b"|%d|" % len(v) + c[1])
            continue
        _sk(h, v, budget, 0)
    if len(_FP_CACHE) > len(live):
        keep = set(live)
        for k in [k for k in _FP_CACHE if k not in keep]:
            del _FP_CACHE[k]
    return "f" + h.hexdigest()[:23]


# ---- the full digest (complete; snapshot points and restore checks) ---------------------------------------------------
class _MainSwap:
    """``sys.modules["__main__"]`` is the namespace's module while a snapshot is pickled, loaded or digested (a cell's
    classes and functions resolve by name in it); the kernel script otherwise."""

    def __enter__(self):
        self.old = sys.modules.get("__main__")
        sys.modules["__main__"] = _NS_MODULE

    def __exit__(self, *exc):
        sys.modules["__main__"] = self.old


class _HashSink:
    """A pickle stream fed straight into a hash (no copy of a 20 MB ``OBS`` is kept)."""

    def __init__(self):
        self.h = hashlib.sha256()

    def write(self, b) -> int:
        self.h.update(b)
        return len(b)


def _canon(h, v, depth: int, path: set) -> None:
    """A complete, deterministic encoding of ``v``: its pickle stream when it pickles, else a structural walk (sets
    order-free) with addresses scrubbed from any repr. The pickler runs in fast mode (no memo): the digest is of values,
    not of object sharing, which a restore cannot reproduce for interned strings (a key the bootstrap's code interned is
    one object live, an unpickled copy after a restore)."""
    t = type(v)
    if not isinstance(v, (types.FunctionType, type, types.ModuleType)):
        sink = _HashSink()
        try:
            p = pickle.Pickler(sink, 4)
            p.fast = True  # no memo; a self-referencing value recurses until RecursionError and is walked instead
            p.dump(v)
        except Exception:  # noqa: BLE001 -- not picklable as a whole: walk it
            sink = None
        if sink is not None:
            h.update(b"P" + sink.h.digest())
            return
    name = f"{t.__module__}.{t.__qualname__}".encode("utf-8", "replace")
    if id(v) in path or depth > 100:
        h.update(b"R" + name)
        return
    if isinstance(v, (types.FunctionType, type, types.ModuleType)):
        h.update(b"N" + name + b":" + str(getattr(v, "__qualname__", getattr(v, "__name__", ""))).encode("utf-8", "replace"))
        return
    path.add(id(v))
    try:
        if isinstance(v, (list, tuple)):
            h.update(b"L" + name + b"%d" % len(v))
            for x in (list.__iter__(v) if isinstance(v, list) else tuple.__iter__(v)):
                _canon(h, x, depth + 1, path)
        elif isinstance(v, dict):
            h.update(b"D" + name + b"%d" % dict.__len__(v))
            for k, x in dict.items(v):
                _canon(h, k, depth + 1, path)
                _canon(h, x, depth + 1, path)
        elif isinstance(v, (set, frozenset)):
            parts = []
            for x in v:
                hx = hashlib.sha256()
                _canon(hx, x, depth + 1, path)
                parts.append(hx.digest())
            h.update(b"S" + name + b"%d" % len(parts) + b"".join(sorted(parts)))
        else:
            d = None
            with contextlib.suppress(Exception):
                d = object.__getattribute__(v, "__dict__")
            if type(d) is dict:
                h.update(b"O" + name)
                _canon(h, d, depth + 1, path)
            else:
                try:
                    r = repr(v)
                except BaseException as e:  # noqa: BLE001 -- a broken __repr__: its type still counts
                    r = f"<repr failed: {type(e).__name__}>"
                h.update(b"r" + name + b":" + _ADDR.sub(" at 0x?", r).encode("utf-8", "replace"))
    finally:
        path.discard(id(v))


def _name_hashes() -> dict:
    out = {}
    with _MainSwap():
        for k in _public():
            h = hashlib.sha256()
            _canon(h, NS[k], 0, set())
            out[k] = h.digest()
    return out


def _combine(parts: dict, exclude=()) -> str:
    h = hashlib.sha256()
    for k in sorted(parts):
        if k not in exclude:
            h.update(k.encode("utf-8", "surrogatepass") + b"=" + parts[k])
    return h.hexdigest()[:24]


def digest() -> str:
    """The full namespace digest: every public name's value hashed whole."""
    return _combine(_name_hashes())


# ---- replayable clock and entropy (M16 B4) ---------------------------------------------------------------------------
class _Entropy:
    """Per cell: records what the main thread read from the clock and the OS's random source, or serves a recorded run's
    values in order (a value of another shape, or past the record, is read live and recorded)."""

    def __init__(self):
        self.on = False
        self.rec: dict = {}
        self.src: dict = {}
        self.pos: dict = {}
        self.nbytes = 0
        self.over = False

    def begin(self, recorded) -> None:
        self.rec, self.pos, self.nbytes, self.over = {"t": [], "n": [], "u": []}, {"t": 0, "n": 0, "u": 0}, 0, False
        src = recorded if isinstance(recorded, dict) else {}
        self.src = {k: v for k, v in src.items() if k in ("t", "n", "u") and isinstance(v, list)}
        self.on = True

    def end(self):
        self.on = False
        out = {k: v for k, v in self.rec.items() if v}
        if self.over:
            out["over"] = True
        self.rec, self.src = {}, {}
        return out or None

    def active(self) -> bool:
        return self.on and threading.get_ident() == _main_thread.ident

    def take(self, kind: str):
        src = self.src.get(kind)
        i = self.pos[kind]
        if src is not None and i < len(src):
            self.pos[kind] = i + 1
            return src[i]
        return None

    def keep(self, kind: str, v, size: int = 0) -> None:
        if len(self.rec[kind]) >= ENTROPY_MAX_READS or self.nbytes + size > ENTROPY_MAX_BYTES:
            self.over = True
            return
        self.nbytes += size
        self.rec[kind].append(v)


_ENT = _Entropy()


def _v_time() -> float:
    if not _ENT.active():
        return _real_time()
    v = _ENT.take("t")
    if type(v) is not float:
        v = _real_time()
    _ENT.keep("t", v)
    return v


def _v_time_ns() -> int:
    if not _ENT.active():
        return _real_time_ns()
    v = _ENT.take("n")
    if type(v) is not int:
        v = _real_time_ns()
    _ENT.keep("n", v)
    return v


def _v_urandom(n) -> bytes:
    if not _ENT.active():
        return _real_urandom(n)
    v = _ENT.take("u")
    try:
        b = bytes.fromhex(v) if isinstance(v, str) and len(v) == 2 * int(n) else None
    except (ValueError, TypeError):
        b = None
    if b is None:
        b = _real_urandom(n)
    _ENT.keep("u", b.hex(), len(b))
    return b


def _install_entropy() -> None:
    time.time, time.time_ns, os.urandom = _v_time, _v_time_ns, _v_urandom
    random._urandom = _v_urandom  # random.seed(), SystemRandom, secrets


# ---- describe (the python-state row, M16 §3.4) ------------------------------------------------------------------------
def _size(v) -> tuple:
    """(size, unit): length for builtin containers and text (their own ``__len__``, never a subclass's), ``nbytes`` for
    arrays, else ``sys.getsizeof``."""
    for base, unit in ((str, "chars"), (bytes, "bytes"), (bytearray, "bytes"), (list, "items"), (tuple, "items"), (dict, "items"),
                       (set, "items"), (frozenset, "items")):
        if isinstance(v, base):
            return base.__len__(v), unit
    t = type(v)
    if t.__module__ != "__main__" and isinstance(getattr(t, "nbytes", None), (types.GetSetDescriptorType, types.MemberDescriptorType, property)):
        with contextlib.suppress(Exception):
            nb = v.nbytes
            if isinstance(nb, int):
                return nb, "bytes"
    try:
        return sys.getsizeof(v), "bytes"
    except Exception:  # noqa: BLE001
        return 0, "bytes"


def describe(limit: int = 500) -> dict:
    """``[{name, type, size, unit, boot}]`` for the model's names: private names and the harness objects the bootstrap
    re-creates (``world``, ``rlm``, its classes) are left out; the bootstrap's data (``OBS``, ``INBOX``...) is kept,
    flagged ``boot``. At most ``limit`` entries (sorted by name); ``total`` counts them all."""
    entries = []
    names_ = [k for k in _public() if not _boot_unchanged(k)]
    for k in names_[: max(0, int(limit))]:
        v = NS[k]
        size, unit = _size(v)
        t = type(v)
        entries.append({"name": k, "type": t.__qualname__ if t.__module__ in ("builtins", "__main__") else f"{t.__module__}.{t.__qualname__}",
                        "size": int(size), "unit": unit, "boot": k in _BOOT_NAMES})
    return {"entries": entries, "total": len(names_)}


# ---- snapshot / restore (M16 §3.4; Prime Agent repl.py:731-875,1070-1092, MIT) -----------------------------------------
class _TooBig(Exception):
    pass


class _CappedSink:
    """The snapshot stream: one name may add at most ``var`` bytes past ``mark``, the stream at most ``total`` (Prime
    Agent ``_CappedWriter``)."""

    def __init__(self, total: int):
        self.buf = io.BytesIO()
        self.total, self.mark, self.var, self.why = total, 0, total, ""

    def write(self, b) -> int:
        n = self.buf.tell() + len(b)
        if n > self.total or n - self.mark > self.var:
            self.why = "total" if n > self.total else "var"
            raise _TooBig()
        return self.buf.write(b)


def _pickler_class(dill):
    class SnapPickler(dill.Pickler):
        def persistent_id(self, obj):
            if obj is NS:
                return "ns"
            k = _BOOT_IDS.get(id(obj))
            if k is not None and _BOOT.get(k) is obj:
                return "b:" + k
            return None

    return SnapPickler


def _unpickler_class(dill):
    """Prime Agent ``_snapshot_unpickler`` (repl.py:1070-1092): dill pickles a pipe- or socket-backed file as its fd
    number; reopening it here would close whatever owns that fd now (the protocol pipe). Plus the persistent references."""
    create_filehandle = dill._dill._create_filehandle

    def refuse_raw_fd(name, *args):
        if isinstance(name, int):
            raise pickle.UnpicklingError(f"refusing to reopen raw file descriptor {name} from a snapshot")
        return create_filehandle(name, *args)

    class SnapUnpickler(dill.Unpickler):
        def find_class(self, module, name):
            target = super().find_class(module, name)
            return refuse_raw_fd if target is create_filehandle else target

        def persistent_load(self, pid):
            if pid == "ns":
                return NS
            if isinstance(pid, str) and pid.startswith("b:") and pid[2:] in _BOOT:
                return _BOOT[pid[2:]]
            raise pickle.UnpicklingError(f"the snapshot refers to {str(pid)[:80]!r}, which this kernel's bootstrap did not bind")

    return SnapUnpickler


def snapshot_blob(max_bytes: int = SNAPSHOT_MAX_BYTES, max_var_bytes: int = SNAPSHOT_MAX_VAR_BYTES) -> tuple:
    """(blob, saved names, lost {name: reason}). One pickle stream of a dict, written name by name; a name that fails or
    exceeds its cap is cut back out of the stream and the memo (Prime Agent skips per name; one stream keeps sharing)."""
    import dill

    sink = _CappedSink(max_bytes)
    sink.buf.write(SNAPSHOT_MAGIC)
    p = _pickler_class(dill)(sink, protocol=4, byref=False, recurse=False)
    p._main = _NS_MODULE
    p.write(pickle.PROTO + bytes([4]) + pickle.EMPTY_DICT)
    saved, lost = [], {}
    items = [(k, NS[k]) for k in sorted(NS) if not k.startswith("__")]
    items.append((RANDOM_KEY, random.getstate()))
    with _MainSwap():
        for k, v in items:
            sink.mark, m = sink.buf.tell(), len(p.memo)
            sink.var = max_bytes if (k in _BOOT_NAMES or k == RANDOM_KEY) else max_var_bytes  # harness data: the total only
            try:
                p.save(k)
                p.save(v)
                p.write(pickle.SETITEM)
            except Exception as e:  # noqa: BLE001 -- one name never aborts the snapshot
                sink.buf.seek(sink.mark)
                sink.buf.truncate()
                while len(p.memo) > m:
                    p.memo.popitem()  # the memo is insertion-ordered: the failed name's entries are the newest
                p._postproc.clear()
                if isinstance(e, _TooBig):
                    lost[k] = f"exceeds the snapshot's {max_bytes}-byte total cap" if sink.why == "total" else f"exceeds the {max_var_bytes}-byte per-variable cap"
                else:
                    lost[k] = f"{type(e).__name__}: {str(e)[:200]}"
                continue
            if k != RANDOM_KEY:
                saved.append(k)
        sink.mark, sink.var = sink.buf.tell(), max_bytes
        try:
            p.write(pickle.STOP)
        except _TooBig:
            raise ValueError("the snapshot exceeds its total size cap") from None
    return sink.buf.getvalue(), saved, lost


def _snapshot(req: dict, _send) -> None:
    rid = req["id"]
    try:
        import dill  # noqa: F401 -- pushed by the host (_load_pylib); a kernel without it cannot snapshot
    except Exception as e:  # noqa: BLE001
        _send({"event": "done", "id": rid, "error": f"snapshot: dill is not available in the kernel ({type(e).__name__}: {e})"})
        return
    mb = req.get("max_bytes") if isinstance(req.get("max_bytes"), int) else SNAPSHOT_MAX_BYTES
    mv = req.get("max_var_bytes") if isinstance(req.get("max_var_bytes"), int) else SNAPSHOT_MAX_VAR_BYTES
    try:
        parts = _name_hashes()
        blob, saved, lost = snapshot_blob(min(mb, SNAPSHOT_MAX_BYTES * 4), mv)
    except Exception as e:  # noqa: BLE001
        _send({"event": "done", "id": rid, "error": f"snapshot: {type(e).__name__}: {str(e)[:500]}"})
        return
    sha = hashlib.sha256(blob).hexdigest()
    n = 0
    for n, i in enumerate(range(0, len(blob), SNAPSHOT_CHUNK)):
        _send({"event": "snapshot_chunk", "id": rid, "seq": n, "data": base64.b64encode(blob[i:i + SNAPSHOT_CHUNK]).decode("ascii")})
    _send({"event": "done", "id": rid, "snapshot": {"sha256": sha, "bytes": len(blob), "chunks": n + 1, "digest": _combine(parts),
                                                    "saved_digest": _combine(parts, exclude=set(lost)), "saved": saved, "lost": lost,
                                                    "python": sys.version.split()[0]}})


_restoring: dict = {"id": None, "parts": [], "size": 0, "next": 0}


def _restore_chunk(req: dict, _send) -> None:
    r = _restoring
    if r["id"] != req["id"]:
        r.update(id=req["id"], parts=[], size=0, next=0)
    if req["seq"] != r["next"]:
        r.update(id=None, parts=[], size=0, next=0)
        _protocol_error(_send, req["id"], f"restore_chunk out of order (seq {req['seq']})")
        return
    try:
        b = base64.b64decode(req["data"], validate=True)
    except (ValueError, TypeError):
        r.update(id=None, parts=[], size=0, next=0)
        _protocol_error(_send, req["id"], "restore_chunk: bad base64")
        return
    r["size"] += len(b)
    if r["size"] > SNAPSHOT_MAX_BYTES * 4 + len(SNAPSHOT_MAGIC):
        r.update(id=None, parts=[], size=0, next=0)
        _protocol_error(_send, req["id"], "restore_chunk: the blob exceeds the snapshot cap")
        return
    r["parts"].append(b)
    r["next"] += 1


def _restore(req: dict, _send) -> None:
    """Load a snapshot into this kernel (bootstrapped by the host first): the blob's names replace the namespace's, a
    name the blob lacks is removed (unless the bootstrap bound it: its fresh value stays, so the harness keeps working)."""
    rid = req["id"]
    r = _restoring
    blob = b"".join(r["parts"]) if r["id"] == rid else b""
    r.update(id=None, parts=[], size=0, next=0)
    if hashlib.sha256(blob).hexdigest() != req["sha256"] or not blob.startswith(SNAPSHOT_MAGIC):
        _send({"event": "done", "id": rid, "error": "restore: the blob does not match its sha256 (or is not a snapshot)"})
        return
    try:
        import dill

        with _MainSwap():
            u = _unpickler_class(dill)(io.BytesIO(memoryview(blob)[len(SNAPSHOT_MAGIC):]))
            u._main = _NS_MODULE
            loaded = u.load()
        if not isinstance(loaded, dict):
            raise pickle.UnpicklingError("the snapshot is not a namespace")
    except Exception as e:  # noqa: BLE001 -- a broken blob is a failed restore (the host falls back to replay), never a dead kernel
        _send({"event": "done", "id": rid, "error": f"restore: {type(e).__name__}: {str(e)[:500]}"})
        return
    rstate = loaded.pop(RANDOM_KEY, None)
    for k in [k for k in NS if not k.startswith("__") and k not in loaded and k not in _BOOT_NAMES]:
        del NS[k]
    NS.update(loaded)
    if rstate is not None:
        try:
            random.setstate(rstate)
        except (TypeError, ValueError):
            pass
    _FP_CACHE.clear()
    _send({"event": "done", "id": rid, "restore": {"restored": sorted(loaded), "digest": digest()}})


def _compile(code: str):
    tree = ast.parse(code, "<cell>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    body = compile(tree, "<cell>", "exec", flags=flags) if tree.body else None
    tail = compile(last, "<cell>", "eval", flags=flags) if last is not None else None
    return body, tail


def _cap(text: str, what: str) -> str:
    if len(text) > RESULT_CAP:
        return text[:RESULT_CAP] + f"\n[... {what} truncated at {RESULT_CAP} characters ...]"
    return text


def _safe_repr(v) -> str:
    try:
        return repr(v)
    except BaseException as e:  # noqa: BLE001 -- a broken __repr__ must not kill the kernel
        return f"<repr failed: {type(e).__name__}>"


async def _run_cell(cid: str, code: str) -> dict:
    _cell.set(cid)
    result = None
    body, tail = _compile(code)
    if body is not None:
        r = eval(body, NS)  # this process IS the sandboxed REPL
        if asyncio.iscoroutine(r):
            await r
    if tail is not None:
        v = eval(tail, NS)
        if asyncio.iscoroutine(v):
            v = await v
        if v is not None:
            NS["_"] = v
            result = _cap(_safe_repr(v), "result")
    return {"result": result}


def _error_text(e: BaseException) -> str:
    """The traceback without kernel frames, bounded: the newest lines are kept (the last one names the exception) and each
    line keeps its head (the exception type), with markers (G1 D13: a 3 MB message lost its type)."""
    try:
        if isinstance(e, SyntaxError):
            lines = traceback.format_exception_only(type(e), e)
        else:
            lines = [ln for ln in traceback.format_exception(type(e), e, e.__traceback__)
                     if "crucible_rlm_repl" not in ln and "/actors/rlm/repl.py" not in ln and "repl.py\", line" not in ln]
    except BaseException:  # noqa: BLE001
        lines = [f"{type(e).__name__}: <the error text could not be formatted>\n"]
    lines = [(ln[: RESULT_CAP // 4] + " [... line truncated ...]\n") if len(ln) > RESULT_CAP // 4 else ln for ln in lines]
    kept, room = [], RESULT_CAP
    for ln in reversed(lines):
        if len(ln) > room and kept:
            break
        kept.append(ln)
        room -= len(ln)
    kept.reverse()
    if len(kept) < len(lines):
        kept.insert(0, f"[... traceback truncated: kept the newest {len(kept)} of {len(lines)} lines ...]\n")
    return "".join(kept)


async def _execute(req: dict, _send) -> None:
    cid = str(req["id"])
    _send({"event": "started", "id": cid})  # the code runs now: a kernel lost before this frame lost no cell (M16 B10)
    st = _CellState(cid)
    _cells[cid] = st
    if _background:
        st.write("stdout", "[output from background work since the last cell]\n" + "".join(_background))
        _background.clear()
    NS.pop("FINISHED", None)
    out: dict = {"result": None, "error": None, "timed_out": False}
    ctx = contextvars.copy_context()
    _ENT.begin(req.get("entropy"))
    task = asyncio.get_running_loop().create_task(_run_cell(cid, req.get("code", "")), context=ctx)
    _active.update(cid=cid, task=task, cancel_ran=False, interrupt=None)
    timeout = req.get("timeout_s") or None
    t0 = time.monotonic()
    try:
        while not task.done():
            if timeout is None:
                await asyncio.wait({task})
                break
            now = time.monotonic()
            used = now - t0 - st.host_wait - ((now - st.wait_started) if st.pending else 0.0)
            if st.pending or used < timeout:
                await asyncio.wait({task}, timeout=0.25 if st.pending else max(0.02, timeout - used))
                continue
            task.cancel()
            await asyncio.wait({task}, timeout=2)
            out["timed_out"] = True
            out["error"] = (f"TimeoutError: the cell exceeded {timeout} s, its deadline (time spent waiting on tools and sub-LLMs is "
                            "not counted)")
            break
        if task.done() and not out["timed_out"]:
            if task.cancelled():
                out["timed_out"] = True
                out["error"] = "Interrupted: the host stopped this cell"
            else:
                exc = task.exception()
                if isinstance(exc, CellInterrupted):
                    out["timed_out"] = True
                    out["error"] = "Interrupted: the host stopped this cell (it blocked the kernel past its deadline)"
                elif exc is not None:
                    out["error"] = _error_text(exc)
                else:
                    out.update(task.result())
    except BaseException as e:  # noqa: BLE001 -- the kernel keeps serving whatever the cell did
        out["error"] = _error_text(e)
        out["timed_out"] = out["timed_out"] or isinstance(e, CellInterrupted)
    if _active.get("interrupt") == cid and not out["timed_out"]:
        out["timed_out"] = True
        out["error"] = out["error"] or "Interrupted: the host stopped this cell"
    entropy = _ENT.end()
    _active.update(cid=None, task=None, interrupt=None)
    if req.get("bootstrap") is True and out["error"] is None and not out["timed_out"]:
        _register_boot()
    for p in _pumps:
        p.drain()
    _cells.pop(cid, None)
    fin = NS.get("FINISHED")
    try:
        fp = fingerprint()
    except BaseException as e:  # noqa: BLE001 -- the cell left something that breaks the sketch: its own replay check
        fp = f"x{type(e).__name__}"[:24]
    done = {"event": "done", "id": cid, "stdout": st.text("stdout"), "stderr": st.text("stderr"), **out,
            "finished": None if fin is None else _cap(fin if isinstance(fin, str) else _safe_repr(fin), "finished"), "digest": fp}
    if entropy:
        done["entropy"] = entropy
    _send(done)


PYLIB_NAME = "crucible_pylib.zip"


class _MemImporter:
    """Imports the host-pushed pure-Python packages (dill) from memory: the zip next to this script is read once at start
    and removed, so the sandbox's ``/tmp`` holds only the kernel script (M16; the M14 sandbox probe)."""

    def __init__(self, files: dict):
        self.files = files

    def _key(self, name: str):
        rel = name.replace(".", "/")
        for key, pkg in ((rel + "/__init__.py", True), (rel + ".py", False)):
            if key in self.files:
                return key, pkg
        return None, False

    def find_spec(self, name, path=None, target=None):
        import importlib.util

        key, pkg = self._key(name)
        if key is None:
            return None
        spec = importlib.util.spec_from_loader(name, self, origin=f"<crucible-pylib>/{key}", is_package=pkg)
        spec.has_location = True
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        key, _ = self._key(module.__name__)
        exec(compile(self.files[key], f"<crucible-pylib>/{key}", "exec"), module.__dict__)  # noqa: S102 -- the host's own dill source


def _load_pylib() -> None:
    import zipfile

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PYLIB_NAME)
    if not os.path.isfile(path):
        return
    try:
        with zipfile.ZipFile(path) as z:
            files = {n: z.read(n) for n in z.namelist() if n.endswith(".py")}
    except (OSError, zipfile.BadZipFile):
        files = {}
    with contextlib.suppress(OSError):
        os.unlink(path)
    if files:
        sys.meta_path.append(_MemImporter(files))


def _protocol_error(_send, rid, why: str) -> None:
    _send({"event": "protocol_error", "id": rid if isinstance(rid, str) else None, "error": why[:2000]})


_seen_ids: set = set()


def _interrupt(cid: str) -> None:
    """Reader thread: cancel the running cell; if its code blocks the event loop (the cancel never runs), SIGINT the main
    thread so the cell raises ``CellInterrupted`` (Prime Agent's interrupt)."""
    if _active["cid"] != cid or _loop is None:
        return
    _active["interrupt"] = cid
    task = _active["task"]

    def cancel() -> None:
        _active["cancel_ran"] = True
        if task is not None and not task.done():
            task.cancel()

    _loop.call_soon_threadsafe(cancel)

    def escalate() -> None:
        time.sleep(0.5)
        if _active["cid"] == cid and not _active["cancel_ran"]:
            try:
                signal.pthread_kill(_main_thread.ident, signal.SIGINT)
            except (OSError, ValueError, AttributeError):
                pass

    threading.Thread(target=escalate, daemon=True).start()


def _on_sigint(signum, frame) -> None:
    # Only a blocked cell is interrupted: if the loop got to run the cancel, the main thread may be in loop code
    if _active["interrupt"] is not None and _active["interrupt"] == _active["cid"] and not _active["cancel_ran"]:
        _active["cancel_ran"] = True
        raise CellInterrupted()


def _readline(f):
    """One request line, at most ``REQUEST_CAP`` bytes; a longer line is discarded and reported (b"" = oversize)."""
    line = f.readline(REQUEST_CAP + 1)
    if not line:
        return None
    if len(line) > REQUEST_CAP and not line.endswith(b"\n"):
        while True:
            more = f.readline(1 << 20)
            if not more or more.endswith(b"\n"):
                break
        return b""
    return line


_EXPECTED = {"execute": {"id": str, "code": str}, "host_reply": {"id": str}, "interrupt": {"id": str}, "digest": {"id": str},
             "names": {"id": str}, "describe": {"id": str}, "snapshot": {"id": str}, "restore_chunk": {"id": str, "seq": int, "data": str},
             "restore": {"id": str, "sha256": str}, "shutdown": {}}
_ID_ONCE = ("execute", "digest", "names", "describe", "snapshot", "restore")


def _reader(f, _send) -> None:
    while True:
        try:
            raw = _readline(f)
        except (OSError, ValueError):
            raw = None
        if raw is None:
            break
        if raw == b"":
            _protocol_error(_send, None, f"request line exceeds {REQUEST_CAP} bytes; discarded")
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw.decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001 -- Prime: a ProtocolError reply, never a dead reader (G1 D13: "["*100000)
            _protocol_error(_send, None, f"malformed request: {type(e).__name__}")
            continue
        t = req.get("type") if isinstance(req, dict) else None
        spec = _EXPECTED.get(t) if isinstance(t, str) else None
        if spec is None:
            _protocol_error(_send, req.get("id") if isinstance(req, dict) else None, f"unknown request type {t!r}")
            continue
        bad = [k for k, ty in spec.items() if not isinstance(req.get(k), ty)]
        if t == "execute" and req.get("timeout_s") is not None and not isinstance(req.get("timeout_s"), (int, float)):
            bad.append("timeout_s")
        if bad:
            _protocol_error(_send, req.get("id"), f"{t}: bad or missing fields {bad}")
            continue
        if t == "host_reply":
            fut = _pending.get(req["id"])
            if fut is not None and _loop is not None:
                data = req.get("data")
                _loop.call_soon_threadsafe(lambda f=fut, d=data: f.done() or f.set_result(d))
            continue
        if t == "interrupt":
            _interrupt(req["id"])
            continue
        if t in _ID_ONCE:
            if req["id"] in _seen_ids:
                _protocol_error(_send, req["id"], f"duplicate request id {req['id']!r}")
                continue
            _seen_ids.add(req["id"])
        if _loop is not None:
            asyncio.run_coroutine_threadsafe(_queue.put(req), _loop)
    # the request channel closed: the owner is gone. A blocked cell would keep the queued shutdown from ever running, so the
    # group is killed after a grace (Prime's owner watchdog; H3 V3: kernels outlived their killed ``docker exec`` client)
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_queue.put({"type": "shutdown"}), _loop)
    time.sleep(OWNER_GRACE_S)
    _kill_group()


def _owner_watchdog(ppid: int) -> None:
    while True:
        time.sleep(0.5)
        if os.getppid() != ppid:  # reparented: the process that started the kernel died
            _kill_group()


def _kill_group() -> None:
    """The kernel and everything a cell started in its process group (threads die with the process)."""
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except OSError:
        pass
    os._exit(0)


async def _serve(_send) -> None:
    global _loop, _queue
    _loop = asyncio.get_running_loop()
    _queue = asyncio.Queue()
    _send({"event": "ready", "protocol": PROTOCOL, "python": sys.version.split()[0], "pid": os.getpid(), "pgid": os.getpgid(0)})
    while True:
        req = await _queue.get()
        t = req.get("type")
        if t == "shutdown":
            return
        try:
            if t == "execute":
                await _execute(req, _send)
            elif t == "digest":
                _send({"event": "done", "id": req.get("id"), "digest": digest()})
            elif t == "names":
                _send({"event": "done", "id": req.get("id"), "names": names()})
            elif t == "describe":
                lim = req.get("limit")
                _send({"event": "done", "id": req.get("id"), "describe": describe(lim if isinstance(lim, int) else 500)})
            elif t == "snapshot":
                _snapshot(req, _send)
            elif t == "restore_chunk":
                _restore_chunk(req, _send)
            elif t == "restore":
                _restore(req, _send)
        except BaseException as e:  # noqa: BLE001 -- e.g. a late CellInterrupted: report it, keep serving
            _protocol_error(_send, req.get("id"), f"kernel: {type(e).__name__}: {str(e)[:500]}")


def _setup_fds() -> tuple[int, int, int]:
    """Reserve the original stdout for the protocol; route fds 1/2 through captured pipes; requests from a dup of fd 0.
    Returns (request fd, protocol fd, host stderr fd); no module global keeps the last two (B10)."""
    protocol_fd = os.dup(1)
    os.set_inheritable(protocol_fd, False)
    host_stderr_fd = os.dup(2)
    os.set_inheritable(host_stderr_fd, False)
    req_fd = os.dup(0)
    os.set_inheritable(req_fd, False)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    for fd, stream in ((1, "stdout"), (2, "stderr")):
        r, w = os.pipe()
        os.set_inheritable(r, False)
        os.dup2(w, fd)
        os.close(w)
        _pumps.append(_Pump(r, fd, stream))
    sys.stdout, sys.stderr = _Writer("stdout"), _Writer("stderr")
    sys.stdin = open(os.devnull)  # noqa: SIM115 -- input() sees EOF
    return req_fd, protocol_fd, host_stderr_fd


def _handshake(f) -> bytes:
    """The host's first request: ``{"type": "hello", "key": <hex>}`` (the session's frame key). Anything else ends the
    kernel: frames could not be authenticated."""
    raw = _readline(f)
    try:
        req = json.loads((raw or b"").decode("utf-8", "replace"))
        key = bytes.fromhex(req["key"]) if isinstance(req, dict) and req.get("type") == "hello" and isinstance(req.get("key"), str) else b""
    except (ValueError, KeyError, TypeError):
        key = b""
    if len(key) < 16:
        os._exit(3)
    return key


def main() -> None:
    try:
        os.setsid()  # its own process group, so close kills everything a cell started (a leader already: keep it)
    except OSError:
        pass
    random.seed(int(os.environ.get("CRUCIBLE_RLM_SEED", "0")))
    _load_pylib()
    _install_entropy()
    req_fd, protocol_fd, err_fd = _setup_fds()
    f = os.fdopen(req_fd, "rb", buffering=1 << 16)
    send = _make_signer(_handshake(f), protocol_fd, err_fd)  # before any cell exists; nothing global holds it (B10)
    NS["host"] = _make_host(send)
    NS["asyncio"] = asyncio
    for k in ("host", "asyncio"):  # re-created by every kernel: persistent references in a snapshot
        _BOOT[k] = NS[k]
        _BOOT_IDS[id(NS[k])] = k
    signal.signal(signal.SIGINT, _on_sigint)  # set before asyncio.run, which then keeps it
    threading.Thread(target=_owner_watchdog, args=(os.getppid(),), daemon=True).start()
    threading.Thread(target=_reader, args=(f, send), daemon=True).start()
    try:
        asyncio.run(_serve(send))
    finally:
        _kill_group()


if __name__ == "__main__":
    main()
