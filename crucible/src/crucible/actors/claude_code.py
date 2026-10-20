"""The builder client: headless Claude Code (``claude -p``) on the user's Claude plan (docs/research/swarm-pipeline/08).

The task factory's builder roles (planner, oracle builder, verifier writer, prompt author) run on Claude through the
Claude Code CLI, not an API key: every call is a ``claude -p`` subprocess. ``ClaudeCodeClient`` has the ``LLMClient`` /
``JsonClient`` shape the factory roles already take (``complete``, ``complete_json``, ``spend``, ``model``,
``lock_settings``), plus ``run_session`` for agentic roles.

- **Identity.** ``model`` is ``claude-code:opus-5.5`` (the id recorded in receipts, families and policy labels); the CLI
  gets ``--model <cli_model>`` (default ``opus``). The family is ``anthropic``. ``factory.models`` refuses this client for
  any role that must be a different family from the builder (reviewer, realism, second verifier, auditor, hacker,
  tie-break, judge): a misassignment raises at ``ModelBook`` construction and at ``client(role)``.
- **JSON roles** (``complete_json``): ``--output-format json`` with no tools (``--tools ""``), the system message as
  ``--system-prompt`` (Claude Code's own prompt replaced), the conversation on stdin. With ``schema=`` the CLI also gets
  ``--json-schema`` and the reply is validated here (``jsonschema``); a reply that is not JSON or breaks the schema is
  retried ``json_retries`` times with the error fed back, then raises ``LLMOutputError``.
- **Agentic roles** (``run_session``): Claude Code with ``cwd`` = the session's private clone, ``--restricted`` (file
  tools confined to the working directories, user/project settings ignored), ``--permission-mode acceptEdits``,
  ``--permission-prompts none`` (anything not allowed is denied, never asked), and an allowed-tools list: the file tools,
  read/commit ``git`` subcommands, and (when a sandbox container is given) ``docker exec <that container>`` so every
  command that runs code runs in the session's offline sandbox, never on the host.
- **Isolation from the operator's setup.** Every call runs with ``--safe-mode`` (no CLAUDE.md, memory, hooks, skills,
  plugins or MCP servers of the operator's own setup), ``--strict-mcp-config`` and ``--no-session-persistence``; the child
  environment drops ``ANTHROPIC_API_KEY`` (the plan pays, not an API key; ``keep_api_key=True`` keeps it) and the parent
  session's ``CLAUDECODE`` markers.
- **Plan usage, not dollars.** The plan bills no per-call dollars, so ``spend.usd`` stays 0. What Claude Code reports
  (``num_turns``, ``duration_ms``, ``duration_api_ms``, ``total_cost_usd`` as an API-equivalent estimate, ``usage``,
  ``modelUsage``, ``session_id``) is kept per call in ``receipts`` (and appended to ``receipts_path`` as JSONL), summed in
  ``plan`` (``PlanUsage``). ``budget_usd`` caps the API-equivalent total: a call that would start past it raises
  ``BudgetExceeded`` (V10), and each call gets ``--max-budget-usd`` for what is left. A plan limit reached (the CLI's
  "usage limit" / "rate limit" errors) is ``BudgetExceeded`` too: infrastructure, masked, never a task's 0 (V8).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.actors.llm import BudgetExceeded, LLMError, LLMOutputError, LLMResponse, Spend
from crucible.core.schema.usage import Usage

MODEL_ID = "claude-code:opus-5.5"
FAMILY = "anthropic"
CLI_MODEL = "opus"
DEFAULT_BINARY = "~/.local/bin/claude"
# file tools of an agentic session (Claude Code built-ins); Bash is added with its allowed command prefixes only
SESSION_TOOLS: tuple[str, ...] = ("Read", "Edit", "Write", "Glob", "Grep", "Bash")
SESSION_GIT: tuple[str, ...] = ("git status *", "git diff *", "git log *", "git show *", "git add *", "git commit *", "git rm *", "git mv *",
                                "git status", "git diff", "git log")
PLAN_LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "limit reached", "quota", "credit balance", "overloaded_error")
_STRIP_ENV = ("ANTHROPIC_API_KEY", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT")


def resolve_binary(binary: str | None = None) -> str:
    """The ``claude`` executable: the argument, ``$CRUCIBLE_CLAUDE_BIN``, ``~/.local/bin/claude``, then ``PATH``."""
    for cand in (binary, os.environ.get("CRUCIBLE_CLAUDE_BIN"), DEFAULT_BINARY):
        if cand:
            p = Path(cand).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
    found = shutil.which("claude")
    if found:
        return found
    raise LLMError("no claude binary (set CRUCIBLE_CLAUDE_BIN or install Claude Code at ~/.local/bin/claude)")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class PlanUsage:
    """Claude Code's own accounting, summed over a client's calls (plan usage: no dollars are billed per call)."""

    calls: int = 0
    turns: int = 0
    duration_ms: int = 0
    duration_api_ms: int = 0
    cost_equiv_usd: float = 0.0  # Claude Code's ``total_cost_usd``: what the calls would cost at API prices
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    errors: int = 0
    sessions: int = 0

    def add(self, rec: Mapping[str, Any]) -> None:
        u = Usage.model_validate(rec.get("usage") or {}) if rec.get("usage") else Usage(source="none")
        self.calls += 1
        self.sessions += rec.get("kind") == "session"
        self.turns += int(rec.get("num_turns") or 0)
        self.duration_ms += int(rec.get("duration_ms") or 0)
        self.duration_api_ms += int(rec.get("duration_api_ms") or 0)
        self.cost_equiv_usd += float(rec.get("total_cost_usd") or 0.0)
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cache_read_tokens += u.cache_read_tokens
        self.cache_write_tokens += u.cache_write_tokens
        self.errors += bool(rec.get("is_error"))


@dataclass
class SessionResult:
    """One agentic session: ``stop`` is ``agent_done`` | ``error`` | ``timeout`` | ``budget``; ``text`` is the final reply."""

    stop: str
    text: str
    turns: int
    usage: dict[str, Any]
    cost_equiv_usd: float
    duration_ms: int
    session_id: str | None
    receipt: dict[str, Any] = field(default_factory=dict)
    permission_denials: list[Any] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        u = Usage.model_validate(self.usage) if self.usage else Usage(source="none")
        return u.prompt_tokens + u.completion_tokens


def render_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """(system prompt, stdin prompt): system messages joined; a single user message as is, a longer conversation as a
    transcript with role headers (a JSON role's retry carries its earlier reply and the error)."""
    system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system").strip()
    rest = [m for m in messages if m.get("role") != "system"]

    def text(c: Any) -> str:
        if isinstance(c, list):
            return " ".join(str(p.get("text", "")) for p in c if isinstance(p, Mapping))
        return "" if c is None else str(c)

    if len(rest) == 1 and rest[0].get("role") == "user":
        return system, text(rest[0].get("content"))
    return system, "\n\n".join(f"[{m.get('role')}]\n{text(m.get('content'))}" for m in rest)


def parse_json_text(text: str) -> Any:
    """A reply's JSON value (fences and leading prose tolerated, as ``LLMClient.complete_json``); ``ValueError`` if none."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(t[start: end + 1])
    raise ValueError(f"not valid JSON: {t[:300]}")


def schema_errors(value: Any, schema: Mapping[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator

    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in Draft202012Validator(dict(schema)).iter_errors(value)][:20]


class ClaudeCodeClient:
    """See the module doc. ``role`` names the factory role this client serves (set by ``factory.models``)."""

    family = FAMILY

    def __init__(self, model: str = MODEL_ID, *, cli_model: str = CLI_MODEL, binary: str | None = None, budget_usd: float | None = None,
                 timeout_s: float = 1800.0, session_timeout_s: float = 5400.0, json_retries: int = 2, receipts_path: Path | str | None = None,
                 effort: str | None = None, keep_api_key: bool = False, role: str | None = None, extra_args: Sequence[str] = ()):
        self.model = model
        self.cli_model = cli_model
        self._binary = binary
        self.budget_usd = budget_usd
        self.timeout_s, self.session_timeout_s = float(timeout_s), float(session_timeout_s)
        self.json_retries = int(json_retries)
        self.receipts_path = Path(receipts_path) if receipts_path else None
        self.effort, self.keep_api_key, self.role = effort, keep_api_key, role
        self.extra_args = list(extra_args)
        self.spend = Spend()  # dollars stay 0 (plan); tokens and calls are counted
        self.plan = PlanUsage()
        self.receipts: list[dict[str, Any]] = []
        self.policy_version = None

    @property
    def binary(self) -> str:
        return resolve_binary(self._binary)

    def lock_settings(self) -> dict[str, Any]:
        return {"model": self.model, "cli_model": self.cli_model, "provider": "claude-code", "family": FAMILY, "effort": self.effort,
                "cache_mode": "off"}

    # ---- the subprocess -------------------------------------------------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        for k in _STRIP_ENV:
            if k == "ANTHROPIC_API_KEY" and self.keep_api_key:
                continue
            env.pop(k, None)
        return env

    def _remaining(self) -> float | None:
        if self.budget_usd is None:
            return None
        left = self.budget_usd - self.plan.cost_equiv_usd
        if left <= 0:
            raise BudgetExceeded(f"Claude plan budget {self.budget_usd:.4f} USD (API-equivalent) used up ({self.plan.cost_equiv_usd:.4f})")
        return left

    def _common(self) -> list[str]:
        args = ["-p", "--output-format", "json", "--model", self.cli_model, "--no-session-persistence", "--safe-mode", "--strict-mcp-config"]
        if self.effort:
            args += ["--effort", self.effort]
        left = self._remaining()
        if left is not None:
            args += ["--max-budget-usd", f"{left:.4f}"]
        return args + self.extra_args

    async def _invoke(self, argv: list[str], prompt: str, *, cwd: Path | None, timeout_s: float, kind: str) -> dict[str, Any]:
        """Run one ``claude -p`` and return its JSON result, with the receipt booked. Infra failures raise ``LLMError``."""
        t0 = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(self.binary, *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE, cwd=str(cwd) if cwd else None, env=self._env())
        except OSError as e:
            raise LLMError(f"claude-code: cannot start the CLI: {e}") from e
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            self._book({"kind": kind, "is_error": True, "subtype": "timeout", "wall_s": round(time.time() - t0, 3), "cwd": str(cwd) if cwd else None,
                        "prompt_sha256": _sha(prompt)})
            raise LLMError(f"claude-code: no result after {timeout_s:.0f} s") from None
        stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
        try:
            doc = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else None
        except (json.JSONDecodeError, IndexError):
            doc = None
        if not isinstance(doc, dict):
            text = f"{stderr.strip()[-400:]} {stdout.strip()[-200:]}".strip()
            if any(m in text.casefold() for m in PLAN_LIMIT_MARKERS):
                raise BudgetExceeded(f"claude-code: plan limit: {text[:300]}")
            raise LLMError(f"claude-code: exit {proc.returncode} without a JSON result: {text[:400]}")
        rec = {"kind": kind, "model": self.model, "cli_model": self.cli_model, "session_id": doc.get("session_id"), "subtype": doc.get("subtype"),
               "is_error": bool(doc.get("is_error")), "num_turns": doc.get("num_turns"), "duration_ms": doc.get("duration_ms"),
               "duration_api_ms": doc.get("duration_api_ms"), "total_cost_usd": doc.get("total_cost_usd"), "usage": doc.get("usage") or {},
               "model_usage": doc.get("modelUsage") or {}, "exit_code": proc.returncode, "wall_s": round(time.time() - t0, 3),
               "cwd": str(cwd) if cwd else None, "prompt_sha256": _sha(prompt), "result_sha256": _sha(str(doc.get("result") or "")),
               "permission_denials": len(doc.get("permission_denials") or []), "role": self.role, "at": round(t0, 1)}
        self._book(rec)
        if rec["is_error"] and any(m in str(doc.get("result") or "").casefold() for m in PLAN_LIMIT_MARKERS):
            raise BudgetExceeded(f"claude-code: plan limit: {str(doc.get('result'))[:300]}")
        return doc

    def _book(self, rec: dict[str, Any]) -> None:
        self.receipts.append(rec)
        self.plan.add(rec)
        u = Usage.model_validate(rec["usage"]) if rec.get("usage") else None
        self.spend.calls += 1
        if u is not None:
            self.spend.prompt_tokens += u.prompt_tokens
            self.spend.completion_tokens += u.completion_tokens
        if self.receipts_path is not None:
            self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
            with self.receipts_path.open("a") as fh:
                fh.write(json.dumps(rec, default=str, sort_keys=True) + "\n")

    # ---- JSON / text roles ----------------------------------------------------------------------------------------------
    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, schema: Mapping[str, Any] | None = None,
                       timeout_s: float | None = None, **_: Any) -> LLMResponse:
        """One tool-less call. ``tools`` are refused: an agentic builder runs as ``run_session`` (Claude Code's own tools),
        never through the OpenAI tool schema. Sampling parameters (temperature, max_tokens, seed) have no CLI flag."""
        if tools:
            raise LLMError("claude-code: OpenAI-style tools are not supported; use run_session for an agentic role")
        system, prompt = render_messages(messages)
        argv = [*self._common(), "--tools", ""]
        if system:
            argv += ["--system-prompt", system]
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
        t0 = time.monotonic()
        doc = await self._invoke(argv, prompt, cwd=None, timeout_s=timeout_s or self.timeout_s, kind="json" if schema is not None else "call")
        if doc.get("is_error"):
            raise LLMError(f"claude-code: {doc.get('subtype')}: {str(doc.get('result'))[:300]}")
        text = doc.get("structured_output")
        text = json.dumps(text) if text is not None and not isinstance(text, str) else (text or str(doc.get("result") or ""))
        u = doc.get("usage") or {}
        return LLMResponse(text=text, usage=u or Usage(source="none"), cost_usd=0.0, model=self.model, raw_finish=doc.get("stop_reason") or "stop",
                           latency_s=time.monotonic() - t0, response_id=doc.get("session_id"),
                           raw_message={"claude_code": {k: doc.get(k) for k in ("num_turns", "duration_ms", "duration_api_ms", "total_cost_usd", "session_id")}})

    async def complete_json(self, messages: list[dict[str, Any]], *, schema: Mapping[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
        """A JSON object (validated against ``schema`` when given), retried with the error fed back (module doc)."""
        msgs = list(messages)
        if not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": "Reply with one JSON object and nothing else."})
        last = ""
        for attempt in range(self.json_retries + 1):
            r = await self.complete(msgs, schema=schema, **{k: v for k, v in kw.items() if k in ("timeout_s",)})
            try:
                val = parse_json_text(r.text)
            except ValueError as e:
                last = str(e)
            else:
                errs = schema_errors(val, schema) if schema is not None else ([] if isinstance(val, dict) else [f"expected a JSON object, got {type(val).__name__}"])
                if not errs:
                    return val
                last = "the reply breaks the schema: " + "; ".join(errs)
            if attempt < self.json_retries:
                msgs = [*msgs, {"role": "assistant", "content": r.text[:6000]},
                        {"role": "user", "content": f"Your reply was rejected ({last[:600]}). Reply again with one JSON object only."}]
        raise LLMOutputError(f"claude-code: no valid JSON after {self.json_retries + 1} tries: {last[:400]}")

    # ---- agentic roles ---------------------------------------------------------------------------------------------------
    def session_argv(self, *, cwd: Path, add_dirs: Sequence[Path | str] = (), sandbox: str | None = None, tools: Sequence[str] = SESSION_TOOLS,
                     allowed: Sequence[str] | None = None, system_append: str | None = None) -> list[str]:
        """The CLI arguments of one agentic session (exposed for tests and receipts)."""
        if allowed is None:
            allowed = ["Read", "Edit", "Write", "Glob", "Grep", *(f"Bash({g})" for g in SESSION_GIT)]
            if sandbox:
                allowed.append(f"Bash(docker exec {sandbox} *)")
        argv = [*self._common(), "--restricted", "--permission-mode", "acceptEdits", "--permission-prompts", "none",
                "--tools", ",".join(tools), "--allowed-tools", *allowed]
        for d in add_dirs:
            argv += ["--add-dir", str(d)]
        if system_append:
            argv += ["--append-system-prompt", system_append]
        return argv

    async def run_session(self, task: str, *, cwd: Path | str, add_dirs: Sequence[Path | str] = (), sandbox: str | None = None,
                          tools: Sequence[str] = SESSION_TOOLS, allowed: Sequence[str] | None = None, system_append: str | None = None,
                          timeout_s: float | None = None) -> SessionResult:
        """Claude Code working in ``cwd`` (module doc). Infra (no CLI, no result, plan limit) raises; the agent's own end
        (an error result, a budget stop) is a ``SessionResult`` with that ``stop``."""
        cwd = Path(cwd).resolve()
        if not cwd.is_dir():
            raise LLMError(f"claude-code: session directory {cwd} does not exist")
        argv = self.session_argv(cwd=cwd, add_dirs=add_dirs, sandbox=sandbox, tools=tools, allowed=allowed, system_append=system_append)
        try:
            doc = await self._invoke(argv, task, cwd=cwd, timeout_s=timeout_s or self.session_timeout_s, kind="session")
        except LLMError as e:
            if "no result after" in str(e):
                rec = self.receipts[-1] if self.receipts else {}
                return SessionResult("timeout", "", 0, {}, 0.0, int(1000 * float(rec.get("wall_s") or 0)), None, rec)
            raise
        sub = str(doc.get("subtype") or "")
        stop = "agent_done" if not doc.get("is_error") and sub in ("success", "") else ("budget" if "budget" in sub or "max_turns" in sub else "error")
        return SessionResult(stop, str(doc.get("result") or ""), int(doc.get("num_turns") or 0), dict(doc.get("usage") or {}),
                             float(doc.get("total_cost_usd") or 0.0), int(doc.get("duration_ms") or 0), doc.get("session_id"), self.receipts[-1],
                             list(doc.get("permission_denials") or []))

    def plan_usage(self) -> dict[str, Any]:
        """The plan accounting for receipts and provenance."""
        return {"model": self.model, "cli_model": self.cli_model, "family": FAMILY, "role": self.role, **asdict(self.plan)}
