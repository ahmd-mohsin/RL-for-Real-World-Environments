"""The Crucible RLM kernel (m14): a persistent Python REPL driven over NDJSON on stdio. Stdlib only, so it can be pushed
into any sandbox and run with ``python3 repl.py`` (pattern: Prime Agent ``prime-agent-runtime/src/rlm/repl.py``,
protocol v3, MIT; re-implemented smaller).

Requests (one JSON object per line on stdin):
- ``{"type":"execute","id":ID,"code":SRC,"timeout_s":T}``: run a cell (top-level ``await`` allowed, one persistent
  namespace and event loop). Ends with exactly one ``done`` event.
- ``{"type":"host_reply","id":RID,"data":ANY}``: the answer to a ``host_request`` the running cell raised.
- ``{"type":"digest","id":ID}`` / ``{"type":"names","id":ID}``: namespace fingerprint / live names.
- ``{"type":"shutdown"}``.

Events (stdout): ``ready``, ``host_request{id,cell,data}``, ``done{id,stdout,stderr,result,error,timed_out,finished,digest}``.

Inside cells: ``await host(type, **payload)`` is the only way out of the kernel (World tools, sub-LLMs, the forge facade
are thin wrappers over it, installed by the host's bootstrap cell). ``FINISHED`` set by ``rlm.finish`` travels in
``done``. The kernel seeds ``random`` from ``CRUCIBLE_RLM_SEED`` so a replayed journal reproduces the namespace.
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import hashlib
import io
import json
import os
import random
import sys
import threading
import traceback
import uuid

MAX_STREAM = 1_048_576
_out = sys.__stdout__
_lock = threading.Lock()
_cell = contextvars.ContextVar("crucible_rlm_cell", default=None)
_buffers: dict[str, dict[str, io.StringIO]] = {}
_pending: dict[str, asyncio.Future] = {}
_loop: asyncio.AbstractEventLoop | None = None
NS: dict = {"__name__": "__main__", "__builtins__": __builtins__}


def _send(obj: dict) -> None:
    line = json.dumps(obj, default=repr, ensure_ascii=False)
    with _lock:
        _out.write(line + "\n")
        _out.flush()


class _Writer(io.TextIOBase):
    def __init__(self, stream: str):
        self.stream = stream

    def write(self, s: str) -> int:
        cid = _cell.get()
        if cid is not None and cid in _buffers:
            b = _buffers[cid][self.stream]
            if b.tell() < MAX_STREAM:
                b.write(s[: MAX_STREAM - b.tell()])
        return len(s)

    def flush(self) -> None:
        pass


async def host(type: str, **payload):
    """Ask the host (the policy / manager process) to do something; returns its reply's data (raises on an error reply)."""
    rid = uuid.uuid4().hex
    fut = asyncio.get_running_loop().create_future()
    _pending[rid] = fut
    _send({"event": "host_request", "id": rid, "cell": _cell.get(), "data": {"type": type, **payload}})
    data = await fut
    if isinstance(data, dict) and data.get("__error__"):
        raise RuntimeError(data["__error__"])
    return data


def digest() -> str:
    """Fingerprint of the namespace: names, types and, for plain values, a hash of their repr."""
    h = hashlib.sha256()
    for k in sorted(NS):
        if k.startswith("_") or k in ("__builtins__", "host", "asyncio"):
            continue
        v = NS[k]
        h.update(k.encode())
        h.update(type(v).__name__.encode())
        if isinstance(v, (int, float, str, bool, bytes, type(None), list, dict, tuple, set)):
            try:
                h.update(repr(v).encode()[:100_000])
            except Exception:  # noqa: BLE001 -- an object whose repr raises: its name and type still count
                h.update(b"<unrepresentable>")
    return h.hexdigest()[:24]


def names() -> list[str]:
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
            result = repr(v)[:MAX_STREAM]
    return {"result": result}


async def _execute(req: dict) -> None:
    cid = str(req["id"])
    _buffers[cid] = {"stdout": io.StringIO(), "stderr": io.StringIO()}
    NS.pop("FINISHED", None)
    out: dict = {"result": None, "error": None, "timed_out": False}
    ctx = contextvars.copy_context()
    task = asyncio.get_running_loop().create_task(_run_cell(cid, req.get("code", "")), context=ctx)
    try:
        r = await asyncio.wait_for(task, timeout=req.get("timeout_s") or None)
        out.update(r)
    except asyncio.TimeoutError:  # noqa: UP041 -- python3 < 3.11 in sandboxes: not yet the builtin
        out["timed_out"] = True
        out["error"] = f"TimeoutError: the cell exceeded {req.get('timeout_s')} s"
    except SyntaxError as e:
        out["error"] = "".join(traceback.format_exception_only(type(e), e))
    except BaseException as e:  # noqa: BLE001 -- the cell's own failure, reported to the model
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        keep = [ln for ln in tb if "crucible_rlm_repl" not in ln and "/actors/rlm/repl.py" not in ln]
        out["error"] = "".join(keep)[-20_000:]
    bufs = _buffers.pop(cid)
    _send({"event": "done", "id": cid, "stdout": bufs["stdout"].getvalue(), "stderr": bufs["stderr"].getvalue(), **out,
           "finished": NS.get("FINISHED"), "digest": digest()})


_stdin = sys.stdin  # the request channel (sys.stdin itself is rebound to /dev/null for cells)


def _reader() -> None:
    for line in _stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        t = req.get("type")
        if t == "host_reply":
            fut = _pending.pop(str(req.get("id")), None)
            if fut is not None and _loop is not None:
                data = req.get("data")
                _loop.call_soon_threadsafe(lambda f=fut, d=data: f.done() or f.set_result(d))
        elif _loop is not None:
            asyncio.run_coroutine_threadsafe(_queue.put(req), _loop)
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_queue.put({"type": "shutdown"}), _loop)


_queue: asyncio.Queue


async def _serve() -> None:
    global _loop, _queue
    _loop = asyncio.get_running_loop()
    _queue = asyncio.Queue()
    threading.Thread(target=_reader, daemon=True).start()
    _send({"event": "ready", "protocol": 1, "python": sys.version.split()[0]})
    while True:
        req = await _queue.get()
        t = req.get("type")
        if t == "shutdown":
            return
        if t == "execute":
            await _execute(req)
        elif t == "digest":
            _send({"event": "done", "id": req.get("id"), "digest": digest()})
        elif t == "names":
            _send({"event": "done", "id": req.get("id"), "names": names()})


def main() -> None:
    random.seed(int(os.environ.get("CRUCIBLE_RLM_SEED", "0")))
    NS["host"] = host
    NS["asyncio"] = asyncio
    sys.stdout, sys.stderr = _Writer("stdout"), _Writer("stderr")
    sys.stdin = open(os.devnull)  # noqa: SIM115 -- input() sees EOF; requests come through the reader thread's handle
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
