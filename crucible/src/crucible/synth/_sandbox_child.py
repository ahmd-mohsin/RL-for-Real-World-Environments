"""The sandbox child (``sandbox.py``): serves one synthesized App as a ``PyWorld`` over line-delimited JSON on stdin and
the original stdout. Started as ``python -I -B _sandbox_child.py --src <dir> --app <dir> --work <dir> --mode world|test``
inside the OS sandbox, with a scrubbed environment. Model code runs only here, under a builtins allowlist and an import
allowlist, after the same static check as the parent's.

Requests: ``reset``, ``tools``, ``call``, ``snapshot``, ``restore``, ``canon``, ``perturb``, ``tables`` (world mode);
``run_tests``, ``info`` (test mode). Every reply is ``{"ok": bool, "value"?, "error"?, "undeclared"?}``.
"""

from __future__ import annotations

import argparse
import asyncio
import builtins
import inspect
import json
import os
import sys
import traceback
from typing import Any


def _harden(work: str, cpu_s: int) -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    except (ImportError, ValueError, OSError):
        pass
    import socket

    def _no_net(*_a: Any, **_k: Any) -> Any:
        raise PermissionError("network access is disabled in the sandbox")

    socket.socket = _no_net  # type: ignore[assignment,misc]
    socket.create_connection = _no_net  # type: ignore[assignment]
    socket.getaddrinfo = _no_net  # type: ignore[assignment]
    os.chdir(work)


SAFE_BUILTINS = ("abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash", "int", "isinstance",
                 "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object", "pow", "range", "repr", "reversed", "round", "set",
                 "slice", "sorted", "str", "sum", "tuple", "type", "zip", "callable", "chr", "ord", "hex", "bin", "oct", "property", "staticmethod",
                 "classmethod", "super", "Exception", "ValueError", "KeyError", "TypeError", "IndexError", "LookupError", "ArithmeticError",
                 "ZeroDivisionError", "RuntimeError", "AttributeError", "StopIteration", "NotImplementedError", "AssertionError", "OverflowError",
                 "True", "False", "None", "NotImplemented", "print", "__build_class__")


def _namespace(allowed: frozenset[str], tool_error: type) -> dict[str, Any]:
    real_import = builtins.__import__

    def guarded_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if level or name.split(".")[0] not in allowed:
            raise ImportError(f"import of {name} is not allowed in the sandbox")
        return real_import(name, globals, locals, fromlist, level)

    b = {k: getattr(builtins, k) for k in SAFE_BUILTINS if hasattr(builtins, k)}
    b["print"] = lambda *a, **k: None  # stdout is the protocol channel
    b["__import__"] = guarded_import
    return {"__builtins__": b, "__name__": "synth_app", "ToolError": tool_error}


class Served:
    def __init__(self, app_dir: str):
        from crucible.synth.sandbox import ALLOWED_IMPORTS, check_code

        self.dir = app_dir
        self.code = open(os.path.join(app_dir, "app.py")).read()  # noqa: SIM115 - the child reads its own App files
        problems = check_code(self.code)
        if problems:
            raise ValueError("code rejected: " + "; ".join(problems[:8]))
        self.design = json.load(open(os.path.join(app_dir, "design.json")))  # noqa: SIM115
        self.schema = json.load(open(os.path.join(app_dir, "schema.json")))  # noqa: SIM115
        seed_path = os.path.join(app_dir, "seed.json")
        self.seed = json.load(open(seed_path)) if os.path.exists(seed_path) else {"tables": {}}  # noqa: SIM115
        kit_path = os.path.join(app_dir, "kit.json")  # the framework kit (worldkit): documents, policy parameters, colleagues
        self.kit = list(json.load(open(kit_path)).get("kinds") or []) if os.path.exists(kit_path) else []  # noqa: SIM115
        self.allowed = ALLOWED_IMPORTS
        self.app_name = self.design["app"]
        self.keys = {t["name"]: (t["key"], str(t.get("fields", {}).get(t["key"], "string")).rstrip("?")) for t in self.schema["tables"]}
        self.last_tb: str | None = None
        self.world: Any = None

    def rows(self, tables: dict[str, list[dict[str, Any]]]) -> dict[str, dict[Any, dict[str, Any]]]:
        out: dict[str, dict[Any, dict[str, Any]]] = {name: {} for name in self.keys}
        for t, rows in (tables or {}).items():
            if t not in self.keys:
                raise ValueError(f"unknown table {t!r}")
            kf, ktype = self.keys[t]
            for r in rows:
                if not isinstance(r, dict) or kf not in r:
                    raise ValueError(f"a row of {t} has no key field {kf!r}")
                k = r[kf]
                if ktype == "integer" and isinstance(k, str) and k.lstrip("-").isdigit():
                    k = int(k)
                out[t][k] = dict(r, **{kf: k})
        return out

    def build(self, tables: dict[str, list[dict[str, Any]]]) -> Any:
        from crucible.synth.iface import App, PyWorld, ToolError

        app = App(self.app_name, self.rows(tables))
        ns = _namespace(self.allowed, ToolError)
        exec(compile(self.code, "app.py", "exec"), ns)  # noqa: S102 - model code, inside the sandbox child only
        build = ns.get("build")
        if not callable(build):
            raise ValueError("the module defines no build(app)")  # noqa: TRY004
        build(app)
        if self.kit:
            from crucible.synth.worldkit import KIT_NAMES, register

            clash = sorted(set(app.tools) & KIT_NAMES)
            if clash:
                raise ValueError(f"the module defines tools the platform provides: {clash}")
            register(app, self.kit)
        design = {t["name"]: t for t in self.design.get("tools", [])}
        for name, td in app.tools.items():
            td.fn = self._wrap(td.fn)
            d = design.get(name)
            if d is not None:  # the catalogue the agent sees: the design's parameter descriptions and enums
                props = td.input_schema.setdefault("properties", {})
                for p in d.get("params", []):
                    if p.get("name") in props:
                        props[p["name"]].update({k: p[k] for k in ("description", "enum") if p.get(k) not in (None, "", [])})
        return PyWorld([app])

    def _wrap(self, fn: Any) -> Any:
        from crucible.synth.iface import ToolError

        def wrapped(*a: Any, **k: Any) -> Any:
            try:
                return fn(*a, **k)
            except ToolError:
                raise
            except Exception:
                self.last_tb = "".join(traceback.format_exc(limit=-4))[-1500:]
                raise

        wrapped.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]  # PyWorld binds arguments against the tool's signature
        wrapped.__doc__ = fn.__doc__
        return wrapped


def _subset(expected: Any, actual: Any) -> str | None:
    """None when ``actual`` contains ``expected`` (dicts by key, lists item by item, numbers numerically)."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"expected an object, got {type(actual).__name__}"
        for k, v in expected.items():
            if k not in actual:
                return f"missing key {k!r}"
            m = _subset(v, actual[k])
            if m:
                return f"{k}: {m}"
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return f"expected a list of {len(expected)}, got {json.dumps(actual, default=str)[:200]}"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            m = _subset(e, a)
            if m:
                return f"[{i}]: {m}"
        return None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return None if expected is actual or expected == actual else f"expected {expected!r}, got {actual!r}"
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return None if abs(expected - actual) <= 1e-6 * max(1.0, abs(expected)) else f"expected {expected!r}, got {actual!r}"
    return None if expected == actual else f"expected {json.dumps(expected, default=str)[:200]}, got {json.dumps(actual, default=str)[:200]}"


async def _run_test(s: Served, t: dict[str, Any]) -> dict[str, Any]:
    from crucible.synth.iface import UndeclaredWrite

    tid = t.get("id")
    try:
        w = s.build(t.get("fixture") or {})
        await w.reset(0)
    except Exception as e:  # noqa: BLE001 - a fixture or load failure fails this test with its reason
        return {"id": tid, "passed": False, "detail": f"setup: {type(e).__name__}: {e}"[:1500]}
    for i, c in enumerate(t.get("calls") or []):
        s.last_tb = None
        tool = c.get("tool", "")
        q = tool if "." in tool else f"{s.app_name}.{tool}"
        try:
            o = await w.call(q, c.get("args") or {})
        except UndeclaredWrite as e:
            return {"id": tid, "passed": False, "detail": f"call {i} {tool}: undeclared write: {e}"[:1500]}
        want = c.get("expect", "ok")
        if want == "ok" and o.error is not None:
            return {"id": tid, "passed": False, "detail": f"call {i} {tool}: expected success, got error {o.error}" + (f"\n{s.last_tb}" if s.last_tb else "")}
        if want == "error" and o.error is None:
            return {"id": tid, "passed": False, "detail": f"call {i} {tool}: expected a refusal, got {json.dumps(o.value, default=str)[:300]}"}
        if want == "error" and s.last_tb:  # a crash is not a refusal: refuse with ToolError
            return {"id": tid, "passed": False, "detail": f"call {i} {tool}: crashed instead of refusing\n{s.last_tb}"}
        if want == "ok" and c.get("result") is not None:
            m = _subset(c["result"], o.value)
            if m:
                return {"id": tid, "passed": False, "detail": f"call {i} {tool}: result {m}"}
    tables = w.tables.get(s.app_name, {})
    for tname, keys in (t.get("expect_state") or {}).items():
        rows = tables.get(tname)
        if rows is None:
            return {"id": tid, "passed": False, "detail": f"state: no table {tname!r}"}
        for k, fields in keys.items():
            row = next((r for kk, r in rows.items() if str(kk) == str(k)), None)
            if row is None:
                return {"id": tid, "passed": False, "detail": f"state: {tname}[{k}] does not exist"}
            m = _subset(fields, row)
            if m:
                return {"id": tid, "passed": False, "detail": f"state: {tname}[{k}] {m}"}
    for tname, keys in (t.get("expect_absent") or {}).items():
        rows = tables.get(tname) or {}
        for k in keys:
            if any(str(kk) == str(k) for kk in rows):
                return {"id": tid, "passed": False, "detail": f"state: {tname}[{k}] should not exist"}
    return {"id": tid, "passed": True, "detail": ""}


async def serve(args: argparse.Namespace, out: Any) -> None:
    from crucible.synth.iface import Perturbation, UndeclaredWrite

    def reply(**kw: Any) -> None:
        out.write(json.dumps(kw, default=str) + "\n")
        out.flush()

    try:
        s = Served(args.app)
        if args.mode == "world":
            s.world = s.build(s.seed.get("tables") or {})
    except Exception as e:  # noqa: BLE001 - the parent shows the reason to the debugger
        reply(ok=False, error=f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=-3)[-1500:])
        return
    reply(ok=True, value="ready")
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        try:
            req = json.loads(line)
            op = req.get("op")
            w = s.world
            if op == "reset":
                val: Any = await w.reset(int(req.get("seed", 0)))
            elif op == "tools":
                val = [t.model_dump(mode="json") for t in await w.tools()]
            elif op == "call":
                try:
                    val = (await w.call(req["tool"], req.get("args") or {})).model_dump(mode="json")
                except UndeclaredWrite as e:
                    reply(ok=False, undeclared=str(e))
                    continue
            elif op == "snapshot":
                val = await w.snapshot()
            elif op == "restore":
                await w.restore(req["ref"])
                val = None
            elif op == "canon":
                c = await w.canon()
                val = {"apps": c.apps, "unobserved": c.unobserved}
            elif op == "perturb":
                await w.perturb(Perturbation.model_validate(req["p"]))
                val = None
            elif op == "tables":
                val = {t: list(rows.values()) for t, rows in w.tables.get(s.app_name, {}).items()}
            elif op == "run_tests":
                val = [await _run_test(s, t) for t in req.get("tests") or []]
            elif op == "info":
                pw = s.build({})
                val = {"tools": [{"name": t.name.split(".", 1)[1], "read_only": t.read_only, "writes": [x.split(".", 1)[1] for x in (t.writes or [])]}
                                 for t in await pw.tools()]}
            else:
                raise ValueError(f"unknown op {op!r}")
            reply(ok=True, value=val)
        except Exception as e:  # noqa: BLE001 - one bad request is answered, the child keeps serving
            reply(ok=False, error=f"{type(e).__name__}: {e}"[:1500])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--app", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--mode", default="world")
    ap.add_argument("--cpu", type=int, default=3600)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # anything the module prints goes to stderr, never into the protocol
    sys.stdout = sys.stderr
    loop = asyncio.new_event_loop()  # before the sockets are disabled: the loop's self-pipe is a local socket pair
    _harden(a.work, a.cpu)
    loop.run_until_complete(serve(a, out))


if __name__ == "__main__":
    main()
