"""Python tool modules (MiMo style: ``tools/<app>.py`` with plain public functions) as tool surfaces.

Two transports with identical semantics:

- **in-process**: the module is loaded with MiMo's own loader and size-cap wrapper
  (``_vendor/mimo_sidecar/mcp_http.py``) and every function becomes a FastMCP ``Tool`` — the same
  object the HTTP server builds, so input schemas, argument validation, result conversion and
  error text are byte-identical to the served version. Fast; used by the local backend and tests.
- **mcp**: the module is served over streamable HTTP by the vendored ``mcp_http`` script and called
  through an MCP client (``crucible.world.packaging.sidecar``). Used when an external harness or the
  agentenv gateway must reach the tools over the network.
"""

from __future__ import annotations

import math
import sqlite3 as _sqlite3

# Exceptions that mean the tool module itself is broken (found by M10 fault injection, 2026-10-09): masked as
# crucible:tool_crashed. Domain errors (ValueError, KeyError, LookupError, PermissionError, RuntimeError raised on
# purpose) stay tool faults the policy sees and is scored on. They are classified only for exceptions raised by the tool
# BODY on arguments that already bound cleanly: a bad argument (out-of-range int, non-finite number, failed coercion) is
# rejected before the body runs, as the policy's error (M14 audit V1).
BUG_EXCEPTIONS = (ArithmeticError, TypeError, AttributeError, NameError, ImportError, AssertionError, NotImplementedError, SyntaxError,
                  RecursionError, MemoryError, _sqlite3.OperationalError, _sqlite3.ProgrammingError)

# Exceptions a tool body raises to refuse the caller's values (an id that does not exist, a reason that does not apply, a
# value outside the allowed set): the policy's malformed call, ``invalid_arguments`` with policy fault (M14 campaign B13;
# they were ``fault: tool``). Other deliberate errors (PermissionError, RuntimeError, ...) stay the tool's.
POLICY_EXCEPTIONS = (ValueError, LookupError)

import json
import sys
import time
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec

INT64 = (-(2**63), 2**63 - 1)  # SQLite INTEGER: a wider Python int raises OverflowError while the tool binds it


def bad_argument(args: Any, path: str = "") -> str | None:
    """Why the policy's arguments cannot be bound by any tool (None when they can): an integer outside int64 (SQLite's
    INTEGER and every C binding), a non-finite number, an unparseable or non-object argument set. M14 audit V1: such a
    value raised ``OverflowError`` inside the tool's SQLite binding, which counted as an environment bug and masked the
    episode, so a policy about to score 0 could append one bad call and drop out of every metric."""
    if not path:
        if not isinstance(args, dict):
            return f"tool arguments must be a JSON object, got {type(args).__name__}"
        if "__unparseable__" in args:
            return "the arguments were not valid JSON"
    if isinstance(args, bool):
        return None
    if isinstance(args, int):
        return None if INT64[0] <= args <= INT64[1] else f"{path or '(arguments)'}: integer {str(args)[:24]}... is outside the 64-bit range"
    if isinstance(args, float):
        return None if math.isfinite(args) else f"{path or '(arguments)'}: {args} is not a finite number"
    if isinstance(args, dict):
        items = list(args.items())
    elif isinstance(args, list | tuple):
        items = list(enumerate(args))
    else:
        return None
    for k, v in items:
        if (why := bad_argument(v, f"{path}/{k}" if path else str(k))) is not None:
            return why
    return None


def _vendored():
    from crucible._vendor.mimo_sidecar import mcp_http

    return mcp_http


def _content_text(blocks: Any) -> str:
    parts: list[str] = []
    for b in blocks or []:
        text = getattr(b, "text", None)
        if text is None and isinstance(b, dict):
            text = b.get("text")
        if text is not None:
            parts.append(text)
        else:
            dump = getattr(b, "model_dump", None)
            parts.append(json.dumps(dump() if dump else b, default=str))
    return "\n".join(parts)


def envelope_error(structured: Any, content: str) -> str | None:
    """The error code of a MiMo-style error envelope (``{"error": {"code": "NOT_FOUND", ...}}``), else None."""
    obj = structured
    if isinstance(obj, dict) and set(obj) == {"result"}:
        obj = obj["result"]
    if obj is None and content.lstrip().startswith("{"):
        try:
            obj = json.loads(content)
        except ValueError:
            return None
    if isinstance(obj, dict) and isinstance(obj.get("error"), dict | str) and not obj.get("ok", False) and not obj.get("success", False):
        err = obj["error"]
        return str(err.get("code") or err.get("type") or "error") if isinstance(err, dict) else "error"
    return None


def _tool_class() -> tuple[Any, bool]:
    """The MCP SDK's function-tool class: mcp 1.x ``fastmcp`` or mcp 2.x ``mcpserver`` (FastMCP renamed to MCPServer;
    ``run`` takes a context and returns a ``CallToolResult``). Found live: the vLLM image ships mcp 2, and every World
    with a ``common/sqlite_app`` failed to start there; the ``verifiers`` extra pins mcp 2 as well."""
    try:
        from mcp.server.fastmcp.tools.base import Tool

        return Tool, False
    except ImportError:  # mcp 2.x: the fastmcp shim module raises on import
        from mcp.server.mcpserver.tools.base import Tool

        return Tool, True


class InProcessToolModule:
    """All public functions of one tools file, as MCP SDK function Tools (mcp 1.x or 2.x), called in this process."""

    def __init__(self, app: str, tools_path: str | Path, *, server_name: str | None = None, env: Any = None):
        Tool, self._mcp2 = _tool_class()

        self.app = app
        self.path = Path(tools_path).resolve()
        mh = _vendored()
        # Unique module name per world copy: two worlds in one process must not share module state.
        before = set(sys.modules)
        module = mh._load_tools_module(self.path)
        self._module_names = set(sys.modules) - before
        self.virtualized: list[str] = []
        if env is not None:  # the World's virtual clock and seed instead of wall time and os.urandom (virtual_env.py)
            from crucible.world.packaging.virtual_env import patch_module, shims

            s = shims(env)
            for n in sorted(self._module_names | {module.__name__}):
                if sys.modules.get(n) is not None:
                    self.virtualized += [f"{n}.{k}" for k in patch_module(sys.modules[n], env, s)]
        self.tools: dict[str, Any] = {}
        for name, fn in mh._public_functions(module):
            self.tools[name] = Tool.from_function(mh._wrap_capped(fn), name=name, description=mh._first_line(fn.__doc__) or name)
        if not self.tools:
            raise ValueError(f"no public tool functions in {self.path}")
        self.server_name = server_name or self.path.stem

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(app=self.app, name=n, description=t.description or n, input_schema=t.parameters) for n, t in sorted(self.tools.items())]

    async def call(self, name: str, args: dict[str, Any], *, call_id: str) -> ToolResult:
        t0 = time.monotonic()
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(tool_call_id=call_id, function_name=f"{self.app}.{name}", ok=False, content=f"Unknown tool: {name}", error="unknown_tool")

        def policy_error(why: str) -> ToolResult:  # the caller's arguments, not the tool, are at fault: scored, never masked
            return ToolResult(tool_call_id=call_id, function_name=f"{self.app}.{name}", ok=False, error="invalid_arguments", fault="policy",
                              content=f"Invalid arguments for {self.app}.{name}: {why}. Resend the call with arguments matching the tool's schema.",
                              duration_s=time.monotonic() - t0)

        if (why := bad_argument(args)) is not None:
            return policy_error(why)
        # Bind the arguments exactly as FastMCP will (JSON pre-parse + the pydantic argument model) before the tool body
        # runs: anything raised here (a ValidationError, a TypeError or OverflowError from a coercion) is the policy's
        # malformed call. Only an exception from the body itself, on arguments that bound cleanly, can be an
        # environment bug (BUG_EXCEPTIONS below).
        meta = getattr(tool, "fn_metadata", None)
        model, parse = getattr(meta, "arg_model", None), getattr(meta, "pre_parse_json", None)
        if model is not None and callable(parse) and callable(getattr(model, "model_validate", None)):  # an SDK without them: the body decides
            try:
                inst = model.model_validate(parse(args))
            except Exception as e:  # noqa: BLE001 - every binding failure is the caller's
                return policy_error(f"{type(e).__name__}: {str(e)[:300]}")
            dump = getattr(inst, "model_dump_one_level", None)
            bound = dump() if callable(dump) else dict(inst)
            if (why := bad_argument(bound, "(bound)")) is not None:
                return policy_error(why)  # a coercion (e.g. "1e30" -> int) produced an unbindable value
            # M14 campaign B13: an argument the tool does not take was dropped silently (get_invoice(vendor_id=...) answered
            # ok), and a float id was coerced to an int (100.0 -> 100); both are the caller's malformed call
            fields = getattr(model, "model_fields", {}) or {}
            known = set(fields) | {f.alias for f in fields.values() if getattr(f, "alias", None)}
            extra = sorted(str(k) for k in args if k not in known)
            if fields and extra:
                return policy_error(f"unknown argument(s) {', '.join(extra)}; {self.app}.{name} takes: {', '.join(sorted(fields)) or '(none)'}")
            for k, raw in args.items():
                v = bound.get(k)
                if isinstance(raw, float | bool) and isinstance(v, int) and not isinstance(v, bool):
                    return policy_error(f"{k}: expected an integer, got {type(raw).__name__} {raw!r}")
        try:
            if self._mcp2:
                r = await tool.run(args, None, convert_result=True)
                result = (list(r.content), r.structured_content)
            else:
                result = await tool.run(args, convert_result=True)
        except Exception as e:  # noqa: BLE001 - FastMCP surfaces tool exceptions as isError text; mirror that exactly
            orig = e.__cause__ or e
            if orig is not e and str(orig) and str(orig) not in str(e):  # mcp 2 keeps the reason on the cause only
                e = type(e)(f"{e}: {orig}")
            if isinstance(orig, BUG_EXCEPTIONS):  # a programming bug in the environment, not a domain answer: mask, don't score
                return ToolResult(tool_call_id=call_id, function_name=f"{self.app}.{name}", ok=False, content=f"internal error in {self.app}.{name}",
                                  error="internal_error", fault="infra", duration_s=time.monotonic() - t0)
            if isinstance(orig, POLICY_EXCEPTIONS):  # the body refused the caller's values (no invoice 99999, status='*'): the policy's
                return ToolResult(tool_call_id=call_id, function_name=f"{self.app}.{name}", ok=False, content=str(e), error="invalid_arguments",
                                  fault="policy", duration_s=time.monotonic() - t0)
            return ToolResult(tool_call_id=call_id, function_name=f"{self.app}.{name}", ok=False, content=str(e), error=type(e).__name__,
                              fault="tool", duration_s=time.monotonic() - t0)
        unstructured = result[0] if isinstance(result, tuple) else result
        structured = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        content = _content_text(unstructured)
        code = envelope_error(structured, content)
        return ToolResult(
            tool_call_id=call_id,
            function_name=f"{self.app}.{name}",
            ok=code is None,  # MiMo convention: {"error": {"code": ...}} returned as a value is still a failed call
            content=content,
            data=structured,
            error=f"tool:{code}" if code else None,
            fault="tool" if code else None,
            duration_s=time.monotonic() - t0,
        )

    def unload(self) -> None:
        for n in self._module_names:
            sys.modules.pop(n, None)
