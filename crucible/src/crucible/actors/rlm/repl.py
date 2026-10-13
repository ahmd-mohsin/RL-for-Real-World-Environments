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
- ``{"type":"digest","id":ID}`` / ``{"type":"names","id":ID}``: namespace fingerprint / live names.
- ``{"type":"shutdown"}``: exit, killing the kernel's process group (threads and children a cell started).

Events: ``ready{protocol,python,pid,pgid}``, ``host_request{id,cell,data}``, ``done{id,stdout,stderr,result,error,
timed_out,finished,digest}``, ``protocol_error{id,error}`` (a malformed request; the kernel keeps serving).

Inside cells: ``await host(type, **payload)`` is the only way out of the kernel (World tools, sub-LLMs, the forge facade
are thin wrappers over it, installed by the host's bootstrap cell). ``FINISHED`` set by ``rlm.finish`` travels in
``done``. The kernel seeds ``random`` from ``CRUCIBLE_RLM_SEED`` so a replayed journal reproduces the namespace.
"""

from __future__ import annotations

import ast
import asyncio
import codecs
import contextvars
import hashlib
import io
import json
import math
import os
import random
import signal
import sys
import threading
import time
import traceback
import uuid

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
NS: dict = {"__name__": "__main__", "__builtins__": __builtins__}


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
        token = b"\xff<drain:" + uuid.uuid4().hex.encode() + b">\xff"
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
    rid = uuid.uuid4().hex
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


def digest() -> str:
    """Fingerprint of the namespace: names, types and, for plain values, a hash of their repr."""
    h = hashlib.sha256()
    for k in sorted(NS):
        if k.startswith("_") or k in ("__builtins__", "host", "asyncio"):
            continue
        v = NS[k]
        h.update(k.encode("utf-8", "replace"))
        h.update(type(v).__name__.encode())
        if isinstance(v, (int, float, str, bool, bytes, type(None), list, dict, tuple, set)):
            try:
                h.update(repr(v).encode("utf-8", "replace")[:100_000])
            except Exception:  # noqa: BLE001 -- an object whose repr raises: its name and type still count
                h.update(b"<unrepresentable>")
    return h.hexdigest()[:24]


def names() -> list:
    return sorted(k for k in NS if not k.startswith("_") and k not in ("__builtins__", "host", "asyncio"))


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
    st = _CellState(cid)
    _cells[cid] = st
    if _background:
        st.write("stdout", "[output from background work since the last cell]\n" + "".join(_background))
        _background.clear()
    NS.pop("FINISHED", None)
    out: dict = {"result": None, "error": None, "timed_out": False}
    ctx = contextvars.copy_context()
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
    _active.update(cid=None, task=None, interrupt=None)
    for p in _pumps:
        p.drain()
    _cells.pop(cid, None)
    fin = NS.get("FINISHED")
    _send({"event": "done", "id": cid, "stdout": st.text("stdout"), "stderr": st.text("stderr"), **out,
           "finished": None if fin is None else _cap(fin if isinstance(fin, str) else _safe_repr(fin), "finished"), "digest": digest()})


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
             "names": {"id": str}, "shutdown": {}}


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
        if t in ("execute", "digest", "names"):
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
    req_fd, protocol_fd, err_fd = _setup_fds()
    f = os.fdopen(req_fd, "rb", buffering=1 << 16)
    send = _make_signer(_handshake(f), protocol_fd, err_fd)  # before any cell exists; nothing global holds it (B10)
    NS["host"] = _make_host(send)
    NS["asyncio"] = asyncio
    signal.signal(signal.SIGINT, _on_sigint)  # set before asyncio.run, which then keeps it
    threading.Thread(target=_owner_watchdog, args=(os.getppid(),), daemon=True).start()
    threading.Thread(target=_reader, args=(f, send), daemon=True).start()
    try:
        asyncio.run(_serve(send))
    finally:
        _kill_group()


if __name__ == "__main__":
    main()
