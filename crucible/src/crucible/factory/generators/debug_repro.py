"""S7r: every concrete claim of a bug report is backed by an executed reproduction (swarm-pipeline/07 §3, easing knob K4).

Adapted from browser-use/rl-env-use (MIT; ``rl_env_use/scout.py`` 102-193: a miner's fact is recorded only with a quote
copied verbatim from the live page, and the code checks the quote is there). PATTERN, rewritten for code tasks: the
evidence for a requirement is a command that shows the bug, executed by us in the task's own World.

**The field.** A builder's ``prompts/<id>.json`` may carry ``repro``: requirement (milestone) id ->
``{"command": "...", "expect": {"exit": 0, "contains": ["..."], "equals": "..."}, "timeout_s": 120}``. The command runs
through the World's shell in the repository root; ``expect`` is the observation of the *correct* program (``exit``
default 0; every ``contains`` string in the output; ``equals``: the whole output, whitespace-normalized). A repro may
write scratch files under ``/tmp/`` (a heredoc in the command); it never edits the repository.

**The gate** (``s7r``): each repro runs in a World at the start tree A' and must show the bug (it does not pass: a
non-zero exit or an output other than expected), and in a World at the oracle B (A' with the oracle's source files
written) where it must pass. Per requirement: ``verified``, ``no_bug_at_start`` (fails the gate: the report claims a bug
the command does not show), ``fails_at_oracle`` (fails: the expected observation is wrong or the command needs more than
the fix), ``unmeasured`` (the World did not open or the shell tool failed: infrastructure, no verdict), ``unverified``
(no repro: never a failure, so prompts without the field keep working). The window's status: ``failed`` if any present
repro failed, else ``unmeasured``, ``verified`` (every milestone), ``partial`` or ``unverified``; ``ok`` is False only for
``failed``.

**Leak rules** (``lint_repro``; the repro is shown to the agent in the prompt, ``render``): the debug lane's prompt lint
(no function a fix changed, no hidden test name, no test name of any language, no version, no commit id or link) with
one exception, paths under ``/tmp/`` (the repro's own scratch files); plus no hidden test file's path and no line of
32+ characters copied from the lines the fixes added to the hidden test files. A repro is never a hidden test.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPRO_TIMEOUT_S = 120
COPY_MIN = 32  # a copied hidden-test line at least this long (stripped) is a leak
SCRATCH = "/tmp/"
EXIT_RX = re.compile(r"^\[(?:exit (-?\d+)|(TIMEOUT))\]\n?", re.MULTILINE)


@dataclass(frozen=True)
class Repro:
    req: str
    command: str
    exit: int = 0
    contains: tuple[str, ...] = ()
    equals: str | None = None
    timeout_s: int = REPRO_TIMEOUT_S

    def text(self) -> str:
        return "\n".join([self.command, *self.contains, *([self.equals] if self.equals is not None else [])])


def parse_repros(authored: Mapping[str, Any], milestone_ids: Sequence[str]) -> tuple[dict[str, Repro], list[str]]:
    """The builder's ``repro`` field -> ``{milestone id: Repro}`` and its format problems (an unknown id, no command, a
    malformed ``expect``). A missing field is no repro (``unverified``), never a problem."""
    raw = authored.get("repro") or {}
    out: dict[str, Repro] = {}
    problems: list[str] = []
    if not isinstance(raw, Mapping):
        return {}, ["repro: must map requirement ids to {command, expect}"]
    for mid, spec in raw.items():
        mid = str(mid)
        if mid not in milestone_ids:
            problems.append(f"repro unknown_requirement_id: {mid}")
            continue
        if isinstance(spec, str):
            spec = {"command": spec}
        if not isinstance(spec, Mapping) or not str(spec.get("command") or "").strip():
            problems.append(f"repro {mid}: no command")
            continue
        exp = spec.get("expect")
        exp = {} if exp is None else exp
        if not isinstance(exp, Mapping):
            problems.append(f"repro {mid}: expect must be an object {{exit, contains, equals}}")
            continue
        cont = exp.get("contains") or []
        cont = [cont] if isinstance(cont, str) else list(cont)
        try:
            ex, to = int(exp.get("exit", 0)), int(spec.get("timeout_s") or REPRO_TIMEOUT_S)
        except (TypeError, ValueError):
            problems.append(f"repro {mid}: exit and timeout_s are integers")
            continue
        out[mid] = Repro(mid, str(spec["command"]).strip(), ex, tuple(str(c) for c in cont if str(c)), None if exp.get("equals") is None else str(exp["equals"]),
                         max(1, min(to, 1800)))
    return out, problems


def _norm(s: str) -> str:
    return " ".join(s.split())


def parse_shell(content: str) -> tuple[int | None, bool, str]:
    """The World shell tool's ``[exit N]`` / ``[TIMEOUT]`` header -> (exit code, timed out, output)."""
    m = EXIT_RX.match(content or "")
    if not m:
        return None, False, content or ""
    return (None if m.group(2) else int(m.group(1))), bool(m.group(2)), (content or "")[m.end():]


def judge(r: Repro, rc: int | None, timed_out: bool, output: str) -> tuple[bool, str]:
    """Does the run show the *correct* behaviour ``r`` expects? (False with the reason otherwise.)"""
    if timed_out:
        return False, f"timed out after {r.timeout_s}s"
    if rc != r.exit:
        return False, f"exit {rc}, expected {r.exit}"
    miss = [c for c in r.contains if c not in output]
    if miss:
        return False, f"output lacks {miss[0]!r}"
    if r.equals is not None and _norm(output) != _norm(r.equals):
        return False, "output differs from the expected output"
    return True, "as expected"


def hidden_paths(rec: Mapping[str, Any]) -> set[str]:
    """The hidden test files: every milestone test's file and every test file the grader writes (``tests_b``)."""
    from crucible.factory.generators.debug import scope
    from crucible.factory.lanes.windows_git import key_of_test

    out = set(rec.get("tests_b") or {})
    with scope(str(rec.get("lang") or "python")):
        for m in rec.get("milestones") or []:
            for t in m.get("tests") or []:
                f, _ = key_of_test(t)
                if f:
                    out.add(f)
    return {p for p in out if p}


def hidden_lines(rec: Mapping[str, Any], base_dir: Path | None = None) -> set[str]:
    """Lines (stripped, >= ``COPY_MIN`` characters) of the hidden test files that the agent's tree does not show: lines of
    ``tests_b`` absent from the same file in A' (``base_dir``), or every such line when A' is not at hand."""
    out: set[str] = set()
    for p, txt in (rec.get("tests_b") or {}).items():
        seen: set[str] = set()
        if base_dir is not None and (Path(base_dir) / p).is_file():
            seen = {ln.strip() for ln in (Path(base_dir) / p).read_text(errors="replace").splitlines()}
        out |= {s for ln in str(txt or "").splitlines() if len(s := ln.strip()) >= COPY_MIN and s not in seen}
    return out


def lint_repro(r: Repro, *, faulty: set[str], hidden: set[str], paths: Iterable[str] = (), lines: Iterable[str] = ()) -> list[str]:
    """The leak rules for one repro (module doc)."""
    from crucible.factory.generators.debug import lint_debug

    text = r.text()
    out = []
    for f in lint_debug(text, faulty=faulty, hidden=hidden):
        m = re.match(r"lint file_path: '([^']*)'", f)
        if m and any(p.startswith(SCRATCH) and p in text for p in (m.group(1), "/" + m.group(1))):
            continue  # the repro's own scratch file (the rule's match starts after the leading slash)
        out.append(f"repro {r.req}: {f}")
    out += [f"repro {r.req}: lint repro_hidden_test_file: {p!r} (a hidden test file)" for p in sorted(set(paths)) if p and p in text]
    out += [f"repro {r.req}: lint repro_copies_hidden_test: {ln[:60]!r}..." for ln in sorted(set(lines)) if ln in text]
    return out


def render(repros: Mapping[str, Repro], req_of: Mapping[str, str]) -> str:
    """The prompt's reproduction block (after the requirements): per requirement its command and the expected
    observation. ``req_of``: milestone id -> the prompt's requirement id (R1, R2, ...)."""
    if not repros:
        return ""
    out = ["How to reproduce (each command runs in the repository root; the expected result is what a correct build shows):"]
    for mid in sorted(repros, key=lambda m: (req_of.get(m, m), m)):
        r = repros[mid]
        exp = [f"exit code {r.exit}"] + [f"the output contains {c!r}" for c in r.contains] + (["the output is exactly the text below"] if r.equals is not None else [])
        body = r.command if "\n" not in r.command else "\n" + r.command
        out.append(f"- [{req_of.get(mid, mid)}] {body}\n  expected: {'; '.join(exp)}" + (f"\n{r.equals}" if r.equals is not None else ""))
    return "\n".join(out)


# ---- the gate --------------------------------------------------------------------------------------------------------------------
@dataclass
class Run:
    rc: int | None = None
    timed_out: bool = False
    output: str = ""
    error: str | None = None  # the World or its tool failed (infrastructure)
    passes: bool | None = None
    why: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


async def _shell(world: Any, command: str, timeout_s: int, n: int) -> tuple[Any, bool]:
    from crucible.core.schema.actions import ToolCall

    res = await world.step("agent", ToolCall(tool_call_id=f"s7r-{n}", function_name="repo.shell", arguments={"command": command, "timeout_s": timeout_s}))
    r = res.observation.tool_results[0] if res.observation.tool_results else None
    return r, bool(getattr(r, "ok", False))


async def _write_oracle(world: Any, files: Mapping[str, str | None]) -> list[str]:
    """B in the World: the oracle's source files written over A' (a deleted file removed). Returns the errors."""
    from crucible.core.schema.actions import ToolCall

    errs = []
    for i, (p, txt) in enumerate(sorted(files.items())):
        if txt is None:
            call = ToolCall(tool_call_id=f"s7r-rm-{i}", function_name="repo.shell", arguments={"command": f"rm -f -- '{p}'", "timeout_s": 60})
        else:
            call = ToolCall(tool_call_id=f"s7r-w-{i}", function_name="repo.write_file", arguments={"path": p, "content": txt})
        res = await world.step("agent", call)
        r = res.observation.tool_results[0] if res.observation.tool_results else None
        if not getattr(r, "ok", False):
            errs.append(f"{p}: {str(getattr(r, 'content', '') or getattr(r, 'error', ''))[:200]}")
    return errs


async def _run_state(task: Any, repros: Mapping[str, Repro], state: str, workdir: Path) -> dict[str, Run]:
    out: dict[str, Run] = {}
    try:
        world = await task.open(workdir)
    except Exception as e:  # noqa: BLE001 -- a World that does not open is infrastructure: no verdict
        return {m: Run(error=f"the World did not open at {state}: {type(e).__name__}: {e}"[:300]) for m in repros}
    try:
        if state == "oracle":
            errs = await _write_oracle(world, dict(task.window.get("oracle_files") or {}))
            if errs:
                return {m: Run(error="the oracle's files could not be written: " + "; ".join(errs)[:300]) for m in repros}
        for n, (mid, r) in enumerate(sorted(repros.items())):
            res, ok = await _shell(world, r.command, r.timeout_s, n)
            content = str(getattr(res, "content", "") or "")
            rc, to, text = parse_shell(content)
            if not ok or (rc is None and not to):
                out[mid] = Run(error=f"the shell tool failed: {content[:200] or getattr(res, 'error', '')}")
                continue
            passes, why = judge(r, rc, to, text)
            out[mid] = Run(rc, to, text[-2000:], None, passes, why)
    finally:
        await world.close()
    return out


async def s7r(task: Any, repros: Mapping[str, Repro], *, workdir: Path | None = None) -> dict[str, Any]:
    """Run every repro at A' (must show the bug) and at the oracle B (must pass) in the task's World (module doc)."""
    mids = [m["id"] for m in task.window.get("milestones") or []]
    per: dict[str, dict[str, Any]] = {m: {"status": "unverified"} for m in mids if m not in repros}
    if repros:
        with tempfile.TemporaryDirectory(prefix="s7r-") as td:
            root = Path(workdir) if workdir is not None else Path(td)
            start = await _run_state(task, repros, "start", root / "start")
            oracle = await _run_state(task, repros, "oracle", root / "oracle")
        for mid in repros:
            a, b = start[mid], oracle[mid]
            row = {"command": repros[mid].command, "start": {"exit": a.rc, "timed_out": a.timed_out, "passes": a.passes, "why": a.why, "tail": a.output[-400:]},
                   "oracle": {"exit": b.rc, "timed_out": b.timed_out, "passes": b.passes, "why": b.why, "tail": b.output[-400:]}}
            if a.error or b.error:
                row.update(status="unmeasured", why=a.error or b.error)
            elif a.passes:
                row.update(status="no_bug_at_start", why=f"at the start tree the command already shows the expected result ({a.why}): it does not show the bug")
            elif not b.passes:
                row.update(status="fails_at_oracle", why=f"at the oracle: {b.why}")
            else:
                row.update(status="verified", why=f"start: {a.why}; oracle: as expected")
            per[mid] = row
    st = [per[m]["status"] for m in per]
    if any(s in ("no_bug_at_start", "fails_at_oracle") for s in st):
        status = "failed"
    elif "unmeasured" in st:
        status = "unmeasured"
    elif st and all(s == "verified" for s in st):
        status = "verified"
    elif "verified" in st:
        status = "partial"
    else:
        status = "unverified"
    return {"task": str(getattr(task, "task_id", "")), "gate": "S7r", "status": status, "ok": status != "failed",
            "verified": sorted(m for m in per if per[m]["status"] == "verified"), "unverified": sorted(m for m in per if per[m]["status"] == "unverified"),
            "failed": sorted(m for m in per if per[m]["status"] in ("no_bug_at_start", "fails_at_oracle")), "per_requirement": per}
