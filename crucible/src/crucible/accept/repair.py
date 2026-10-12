"""The RLM task manager (m14): an RLM that manages and improves forge tasks over long horizons.

Pieces (study: ``docs/research/m14-rlm/01-prime-agent-study.md``; design ``02-rlm-agent-design.md``):

- ``RLMAgent``: the RLM loop without a World (a persistent kernel, one ``python`` tool, ``rlm.query/batch/spawn/finish``,
  goal continuation, output cap). Its namespace gets proxy objects (``forge``, ``harness``, ...) whose calls are host
  requests answered by Python handlers in this process, so every effect on the forge is decided, journaled and auditable
  here. ``rlm.spawn(role, prompt)`` runs a child ``RLMAgent`` synchronously (depth-limited; Prime's children are
  asynchronous sessions, ours return their answer, which keeps the manager's journal deterministic).
- ``ForgeWorkspace``: versioned tasks. Edits (``add_check``, ``drop_check``) create a *candidate version*; ``verify`` runs
  the M9 checks on a version in fresh Worlds; ``commit`` refuses a version without a passing verification. The verifier
  is objective (the M9 contract), never an LLM.
- ``TaskFSM``: Planner -> Implementer -> Reviewer -> Verifier per task (the Rust-rewrite swarm's machine, RUST:62-72):
  generation and verification never share an agent; the reviewer is a different model in a fresh context; a failed review
  or verification loops back to the implementer with the findings, up to ``max_rounds``; the FSM state is data, written to
  a journal after every phase, so a crashed run resumes where it stopped (Prime's factory keeps run state in kernel memory).
- ``RLMTaskManager``: the long-horizon orchestrator: an ``RLMAgent`` whose ``forge`` proxy can list and inspect tasks,
  verify them, and run the FSM on the ones that fail; lessons go to the continual harness only through measured
  evaluation (``harness_state.HarnessState.evaluate``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.actors.rlm.kernel import Kernel
from crucible.actors.rlm.policy import PYTHON_TOOL

AGENT_SYSTEM = """You are {name}, an agent that works through a persistent Python REPL: the `python` tool. Everything is code.
- `TASK` is your instruction. Objects in your namespace: {objects}. Call their methods with `await obj.method(**kwargs)`; `await obj.help()` lists each method's signature.
- Sub-LLMs: `await rlm.query(prompt, context=None)` -> str, `await rlm.batch([...])` in parallel. A sub-agent with its own REPL: `await rlm.spawn(role, prompt)` -> its final answer.
- What you print is truncated to {max_output} characters; keep data in variables and print what you need.
- When done, call `rlm.finish("<answer>")`. Before that, check your result against every requirement in TASK."""

AGENT_BOOTSTRAP = r'''
TASK = __TASK__
class _Proxy:
    def __init__(self, ns):
        self._ns = ns
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        async def call(**kw):
            return await host(f"{self._ns}.{name}", **kw)
        return call
class _RLM:
    async def query(self, prompt, context=None, max_tokens=None):
        return await host("llm.query", prompt=str(prompt), context=None if context is None else str(context), max_tokens=max_tokens)
    async def batch(self, prompts, max_tokens=None):
        items = [p if isinstance(p, (list, tuple)) else (p, None) for p in prompts]
        return await host("llm.batch", items=[[str(p), None if c is None else str(c)] for p, c in items], max_tokens=max_tokens)
    async def spawn(self, role, prompt):
        return await host("rlm.spawn", role=str(role), prompt=str(prompt))
    def finish(self, answer=""):
        globals()["FINISHED"] = str(answer)
        return "finished"
rlm = _RLM()
'''

CONTINUE = ("[goal: continuation] You stopped without calling rlm.finish(). Check your work against every requirement in TASK, "
            "finish what is missing, then call rlm.finish(answer).")


@dataclass
class AgentRun:
    answer: str | None
    finished: bool
    turns: int
    model_calls: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    stop: str


class RLMAgent:
    def __init__(self, name: str, llm: Any, *, handlers: dict[str, Callable[..., Awaitable[Any]]] | None = None, objects: tuple[str, ...] = (),
                 sub_llm: Any = None, roles: dict[str, dict[str, Any]] | None = None, depth: int = 0, max_depth: int = 2, max_turns: int = 40,
                 max_output_chars: int = 8192, cell_timeout_s: float = 120.0, max_continuations: int = 2, launcher: Any = None, seed: int = 0,
                 system: str | None = None, max_tokens: int = 2000, extra_context: str = ""):
        self.name, self.llm, self.sub_llm = name, llm, sub_llm or llm
        self.handlers, self.objects, self.roles = dict(handlers or {}), objects, dict(roles or {})
        self.depth, self.max_depth, self.max_turns = depth, max_depth, max_turns
        self.max_output_chars, self.cell_timeout_s, self.max_continuations = max_output_chars, cell_timeout_s, max_continuations
        self.launcher, self.seed, self.max_tokens, self.extra_context = launcher, seed, max_tokens, extra_context
        objs = ", ".join(f"`{o}`" for o in objects) or "none besides `rlm`"
        self.system = (system or AGENT_SYSTEM).format(name=name, objects=objs, max_output=max_output_chars)
        self.model_calls: list[dict[str, Any]] = []
        self._call: dict[str, Any] | None = None

    async def _host(self, data: dict[str, Any]) -> Any:
        t = str(data.pop("type", ""))
        if t == "llm.query":
            return await self._sub(data.get("prompt", ""), data.get("context"), data.get("max_tokens"))
        if t == "llm.batch":
            import asyncio

            return list(await asyncio.gather(*(self._sub(p, c, data.get("max_tokens")) for p, c in data.get("items") or [])))
        if t == "rlm.spawn":
            return await self._spawn(str(data.get("role") or "worker"), str(data.get("prompt") or ""))
        if t.endswith(".help"):  # the proxies are opaque in the kernel (``dir`` shows nothing): help lists real signatures and docs
            ns = t[: -len(".help")]
            import inspect

            return {k.split(".", 1)[1]: f"{k.split('.', 1)[1]}{inspect.signature(v)}: {(inspect.getdoc(v) or '').splitlines()[0] if inspect.getdoc(v) else ''}"
                    for k, v in sorted(self.handlers.items()) if k.startswith(ns + ".")}
        fn = self.handlers.get(t)
        if fn is None:
            raise ValueError(f"unknown call {t!r}; available: {sorted(self.handlers)}")
        return await fn(**data)

    async def _sub(self, prompt: str, context: str | None, max_tokens: int | None) -> str:
        content = prompt if context is None else f"{prompt}\n\n<context>\n{context}\n</context>"
        r = await self.sub_llm.complete([{"role": "user", "content": content}], tools=None, max_tokens=int(max_tokens or 1000), temperature=0.0)
        self.model_calls.append({"call_id": f"{self.name}:c{len(self.model_calls) + 1}", "agent": self.name, "kind": "sub",
                                 "parent": self._call["call_id"] if self._call else None, "model": r.model, "usage": r.usage, "cost_usd": r.cost_usd,
                                 "tokens": r.tokens})
        return r.text or ""

    async def _spawn(self, role: str, prompt: str) -> str:
        if self.depth + 1 > self.max_depth:
            raise RuntimeError(f"recursion depth limit {self.max_depth} reached")
        spec = self.roles.get(role, {})
        child = RLMAgent(f"{self.name}/{role}", spec.get("llm") or self.llm, handlers=spec.get("handlers", self.handlers),
                         objects=tuple(spec.get("objects", self.objects)), sub_llm=self.sub_llm, roles=self.roles, depth=self.depth + 1,
                         max_depth=self.max_depth, max_turns=spec.get("max_turns", self.max_turns), max_output_chars=self.max_output_chars,
                         cell_timeout_s=self.cell_timeout_s, launcher=self.launcher, seed=self.seed, system=spec.get("system"),
                         extra_context=spec.get("instructions", ""))
        run = await child.run(prompt)
        self.model_calls += run.model_calls
        if not run.finished:
            raise RuntimeError(f"sub-agent {role} stopped without an answer ({run.stop})")
        return run.answer or ""

    def _format(self, r: Any) -> str:
        parts = [x for x in (r.stdout.rstrip("\n") if r.stdout else "", ("[stderr]\n" + r.stderr.rstrip("\n")) if r.stderr else "",
                              r.result or "", (r.error or "").rstrip("\n")) if x]
        text = "\n".join(parts) or "(no output)"
        n = self.max_output_chars
        return text if len(text) <= n else text[: n * 3 // 4] + f"\n[... {len(text) - n} characters truncated ...]\n" + text[-(n - n * 3 // 4):]

    async def run(self, prompt: str) -> AgentRun:
        kernel = Kernel(self.launcher, seed=self.seed, cell_timeout_s=self.cell_timeout_s)
        boot = AGENT_BOOTSTRAP.replace("__TASK__", repr(prompt)) + "".join(f"\n{o} = _Proxy({o!r})" for o in self.objects)
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": prompt + (f"\n\n{self.extra_context}" if self.extra_context else "")}]
        turns, conts, stop = 0, 0, "max_turns"
        try:
            r0 = await kernel.execute(boot, self._host, timeout_s=60)
            if not r0.ok:
                raise RuntimeError(f"agent bootstrap failed: {r0.error}")
            while turns < self.max_turns:
                if turns == self.max_turns - 2:  # Prime's budget wrap-up: the m14 live implementer reached a right answer and ran out of turns
                    messages.append({"role": "user", "content": "[budget] 2 turns left. Stop exploring: call rlm.finish(...) now with your best result."})
                r = await self.llm.complete(messages, tools=[PYTHON_TOOL], max_tokens=self.max_tokens, temperature=0.0)
                turns += 1
                call = {"call_id": f"{self.name}:c{len(self.model_calls) + 1}", "agent": self.name, "kind": "turn", "model": r.model,
                        "usage": r.usage, "cost_usd": r.cost_usd, "tokens": r.tokens}
                self.model_calls.append(call)
                cells = [tc for tc in r.tool_calls if tc["name"] == "python"]
                if not cells:
                    messages.append({"role": "assistant", "content": r.text or ""})
                    if conts < self.max_continuations:
                        conts += 1
                        messages.append({"role": "user", "content": CONTINUE})
                        continue
                    return AgentRun(r.text, False, turns, self.model_calls, messages, "no_finish")
                tc = cells[0]
                tid = tc.get("id") or f"py{turns}"
                code = str((tc.get("arguments") or {}).get("code") or "")
                messages.append({"role": "assistant", "content": r.text or "", "tool_calls": [
                    {"id": tid, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}]})
                self._call = call
                res = await kernel.execute(code, self._host)
                self._call = None
                messages.append({"role": "tool", "tool_call_id": tid, "content": self._format(res)})
                if res.finished is not None:
                    return AgentRun(str(res.finished), True, turns, self.model_calls, messages, "finished")
            return AgentRun(None, False, turns, self.model_calls, messages, stop)
        finally:
            await kernel.close()


# ---- the forge workspace: versioned tasks, objective verification ------------------------------------------------------
@dataclass
class Version:
    task: Any
    parent: int | None
    note: str
    verification: dict[str, Any] | None = None


class ForgeWorkspace:
    """``rebuild(task, checks) -> task``: the pack's way to make a task with a different CheckSet (``CheckSpec`` list)."""

    # A7 belongs in the default set: in the m14 live run a candidate that dropped every scope check passed A1 + A16 (the hacker
    # never makes collateral edits); only "every stated property is graded" catches an ungraded property objectively
    # A2, A3 and A13 too (M9 audit V4: a repair that dropped the check guarding a stated bound passed A1 + A7 + A16 and was
    # committed; A3's authored negative would have caught it)
    def __init__(self, tasks: dict[str, Any], *, rebuild: Callable[[Any, list[Any]], Any],
                 checks: tuple[str, ...] = ("A1", "A2", "A3", "A7", "A13", "A16"),
                 cache_root: Path | None = None, vocabulary: dict[str, Any] | None = None):
        self.versions: dict[str, list[Version]] = {k: [Version(t, None, "original")] for k, t in tasks.items()}
        self.vocab = dict(vocabulary or {})  # the pack's check grammar: kinds, event rules, examples (an implementer cannot guess them)
        self._start: dict[str, dict[str, Any]] = {}
        self.rebuild, self.checks = rebuild, checks
        self.cache_root = cache_root
        self.committed: dict[str, int] = dict.fromkeys(tasks, 0)

    def task(self, task_id: str, version: int | None = None) -> Any:
        vs = self.versions[task_id]
        return vs[self.committed[task_id] if version is None else int(version)].task

    async def tasks(self) -> list[dict[str, Any]]:
        """Every task: id, committed version, number of versions, title."""
        return [{"task_id": k, "committed": self.committed[k], "versions": len(v), "title": getattr(v[0].task.spec, "title", "")} for k, v in self.versions.items()]

    async def vocabulary(self) -> dict[str, Any]:
        """The check grammar: fields, kinds, event rules and examples."""
        return self.vocab

    async def show(self, task_id: str, version: int | None = None) -> dict[str, Any]:
        """A task version's instruction, stated properties and checks (default: the committed version)."""
        t = self.task(task_id, version)
        docs = {}
        if hasattr(t, "agent_visible"):  # the policy documents the task is graded against (planner and reviewer read them)
            docs = {k: v[:4000] for k, v in t.agent_visible().items() if k != "instruction" and k.endswith((".md", ".txt"))}
        return {"task_id": task_id, "version": self.committed[task_id] if version is None else int(version), "instruction": t.instruction,
                "documents": docs, "start_state": await self.start_state(task_id),
                "properties": [{"id": p.id, "kind": p.kind, "text": p.text} for p in t.spec.properties],
                "checks": [c.model_dump(mode="json", exclude_none=True, exclude_defaults=True) | {"id": c.id, "gate": c.gate, "kind": c.kind}
                           for c in t.cs.checks],
                "oracle_actions": len(t.oracle)}

    async def start_state(self, task_id: str) -> dict[str, Any]:
        """The vocabulary's ``start_facts`` (name -> scalar SQL) evaluated on the task's seeded start state. Every episode starts
        from it, so a final-state check over rows absent here grades in-episode work; the m14 live reviewer rejected sound
        candidates because it could not know that."""
        facts = self.vocab.get("start_facts") or {}
        t = self.task(task_id, 0)
        if not facts or not hasattr(t, "_run"):
            return {}
        if task_id not in self._start:
            out = {}
            for name, q in facts.items():
                app, sql = q if isinstance(q, (list, tuple)) else ("tickets", q)
                out[name] = await t._run([], query=(app, sql))
            self._start[task_id] = {"note": "every episode starts from this seeded state", **out}
        return self._start[task_id]

    def _new(self, task_id: str, base: int, checks: list[Any], note: str) -> int:
        t = self.rebuild(self.task(task_id, base), checks)
        self.versions[task_id].append(Version(t, base, note))
        return len(self.versions[task_id]) - 1

    async def add_check(self, task_id: str, check: dict[str, Any], base: int | None = None, note: str = "") -> int:
        """Add one check (a dict in the vocabulary's grammar) to version ``base`` (default: committed); returns the new version."""
        from crucible.forge.schema import CheckSpec

        b = self.committed[task_id] if base is None else int(base)
        spec = CheckSpec.model_validate(check)
        problems = self._conform(spec)
        if problems:  # the m14 live implementer invented spec fields the grader silently ignored
            raise ValueError("check rejected by the vocabulary: " + "; ".join(problems))
        if spec.id in {c.id for c in self.task(task_id, b).cs.checks}:
            raise ValueError(f"check {spec.id} already exists in version {b}; drop it first or use a new id")
        return self._new(task_id, b, [*self.task(task_id, b).cs.checks, spec], note or f"add {spec.id}")

    def _conform(self, spec: Any) -> list[str]:
        v = self.vocab
        out = []
        kinds = v.get("kinds")
        if kinds and spec.kind not in kinds:
            out.append(f"kind {spec.kind!r} is not one of {sorted(kinds)}")
        if spec.kind == "event":
            rule = (spec.spec or {}).get("rule")
            rules = v.get("event_rules")
            if rules is not None and rule not in rules:
                out.append(f"event rule {rule!r} is not one of {sorted(rules)}")
            allowed = (v.get("spec_keys") or {}).get(rule)
            if allowed is not None:
                bad = sorted(set(spec.spec or {}) - set(allowed))
                if bad:
                    out.append(f"spec keys {bad} are not understood by rule {rule!r} (allowed: {allowed})")
        elif spec.spec and (v.get("spec_keys") or {}).get(spec.kind) is not None:
            bad = sorted(set(spec.spec) - set(v["spec_keys"][spec.kind]))
            if bad:
                out.append(f"spec keys {bad} are not understood by {spec.kind} checks")
        if spec.kind in ("predicate", "invariant") and not spec.sql:
            out.append(f"a {spec.kind} check needs sql")
        if spec.kind == "predicate" and not spec.expect:
            out.append("a predicate check needs expect")
        return out

    async def drop_check(self, task_id: str, check_id: str, base: int | None = None, note: str = "") -> int:
        """Remove a check from version ``base`` (default: committed); returns the new version."""
        b = self.committed[task_id] if base is None else int(base)
        return self._new(task_id, b, [c for c in self.task(task_id, b).cs.checks if c.id != check_id], note or f"drop {check_id}")

    async def verify(self, task_id: str, version: int | None = None, checks: list[str] | None = None) -> dict[str, Any]:
        """Run the objective acceptance checks (M9) on a version in fresh Worlds: {passed, checks, findings}."""
        import tempfile

        from crucible.accept.policy import Policy
        from crucible.accept.run import accept_task

        v = self.committed[task_id] if version is None else int(version)
        only = list(checks or self.checks)
        cand = self.task(task_id, v)
        cand.audit_floor = list(self.task(task_id).cs.checks)  # the committed version's checks stay in the audit (no self-weakening)
        r = await accept_task(cand, policy=Policy(required=only), cache_root=self.cache_root or Path(tempfile.mkdtemp()),
                              only=only, reuse=False)
        rows = {k: {"status": str(x.status), "reasons": list(x.reasons)[:5]} for k, x in r.results.items() if k in only}
        passed = all(x["status"] == "pass" for x in rows.values())
        findings = [f"{k}: {reason}" for k, x in rows.items() if x["status"] != "pass" for reason in x["reasons"]]
        for k, x in r.results.items():  # the evidence, not just the verdict: e.g. the exact actions of each A16 hack
            if k in only and str(x.status) != "pass" and k == "A16":
                for hk in (x.measured or {}).get("hacks", [])[:3]:
                    acts = [a.get("function_name") or f"message to {a.get('to')}" for a in hk.get("actions") or []]
                    findings.append(f"A16 evidence: {hk.get('strategy')} earned reward {hk.get('reward')} with only these actions: {acts}")
        out = {"task_id": task_id, "version": v, "passed": passed, "checks": rows, "findings": findings}
        self.versions[task_id][v].verification = out
        return out

    async def commit(self, task_id: str, version: int) -> dict[str, Any]:
        ver = self.versions[task_id][int(version)]
        if not (ver.verification and ver.verification.get("passed")):
            raise PermissionError(f"version {version} of {task_id} has no passing verification: run forge.verify first")
        self.committed[task_id] = int(version)
        return {"task_id": task_id, "committed": int(version)}


# ---- the per-task FSM ----------------------------------------------------------------------------------------------------
@dataclass
class FSMState:
    task_id: str
    phase: str = "verify_base"  # verify_base | plan | implement | review_verify | committed | healthy | failed
    round: int = 0
    plan: str = ""
    candidate: int | None = None
    findings: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    review_errors: int = 0


class ImplementerFailed(RuntimeError):
    """The implementer produced no usable candidate (a content failure: a finding for the next round, not a crash)."""


# What each acceptance check proves (reviewers and planners judge with these; m14 live run: a reviewer that did not know A1
# meant "the reference solution passes every gate" rejected three verified candidates for "unproven" checks)
CHECK_MEANING = {
    "A1": "the reference solution passes every gate, new checks included, in repeated fresh Worlds (valid solutions are not rejected)",
    "A2": "doing nothing scores exactly 0",
    "A7": "every property the instruction states is graded by at least one check, and every check grades a stated property", "A3": "known-wrong solutions (negatives) fail at their expected gates",
    "A13": "authored attacks on the grader are caught",
    "A16": "an adversarial hacker that mutates honest solutions toward less work finds no submission that earns reward without the stated "
           "properties",
    "A18": "a perturbation has the effect its label claims",
}


def _json_block(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


class TaskFSM:
    """``planner(report) -> plan``; ``implementer(task_id, plan, findings, base) -> candidate version``; ``reviewer(summary) ->
    {"approved", "findings"}``; the verifier is ``workspace.verify``. The state is journaled after every phase."""

    def __init__(self, workspace: ForgeWorkspace, *, planner: Callable[..., Awaitable[str]], implementer: Callable[..., Awaitable[int]],
                 reviewer: Callable[..., Awaitable[dict[str, Any]]], max_rounds: int = 3, journal: Path | None = None):
        self.ws, self.planner, self.implementer, self.reviewer = workspace, planner, implementer, reviewer
        self.max_rounds, self.journal = max_rounds, Path(journal) if journal else None

    def _load(self, task_id: str) -> FSMState:
        if self.journal and self.journal.exists():
            for line in reversed(self.journal.read_text().splitlines()):
                d = json.loads(line)
                if d["task_id"] == task_id:
                    return FSMState(**d)
        return FSMState(task_id)

    def _save(self, st: FSMState, event: str, **info: Any) -> None:
        st.history.append({"phase": st.phase, "round": st.round, "event": event, **info})
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a") as f:
                f.write(json.dumps(asdict(st), default=str) + "\n")

    async def _review_input(self, task_id: str, st: FSMState, ver: dict[str, Any]) -> dict[str, Any]:
        """What the reviewer judges: the change (a check diff), the task's stated properties, the verification and the plan."""
        before, after = await self.ws.show(task_id), await self.ws.show(task_id, st.candidate)
        b = {c["id"]: c for c in before["checks"]}
        a = {c["id"]: c for c in after["checks"]}
        added = [a[k] for k in a if k not in b]
        rules = {(c.get("spec") or {}).get("rule") for c in added} - {None}
        vocab = self.ws.vocab
        return {"task_id": task_id, "instruction": after["instruction"], "properties": after["properties"], "documents": after.get("documents", {}),
                "start_state": after.get("start_state", {}),
                "check_grammar": {"rules_used": {r: (vocab.get("event_rules") or {}).get(r) for r in rules}, "fields": vocab.get("check_fields")},
                "what_the_verification_proves": {k: CHECK_MEANING.get(k, "") for k in ver["checks"]},
                "change": {"added_checks": [a[k] for k in a if k not in b], "removed_checks": [k for k in b if k not in a],
                           "changed_checks": [a[k] for k in a if k in b and a[k] != b[k]]},
                "verification": {"passed": ver["passed"], "checks": ver["checks"]}, "plan": st.plan[:4000]}

    async def run(self, task_id: str) -> FSMState:
        st = self._load(task_id)
        while st.phase not in ("committed", "healthy", "failed"):
            if st.phase == "verify_base":
                rep = await self.ws.verify(task_id)
                if rep["passed"]:
                    st.phase = "healthy"
                    self._save(st, "base passes every check")
                    break
                st.findings, st.phase = rep["findings"], "plan"
                self._save(st, "base fails", findings=rep["findings"])
            elif st.phase == "plan":
                st.plan = await self.planner(await self.ws.show(task_id), st.findings)
                st.phase = "implement"
                self._save(st, "planned")
            elif st.phase == "implement":
                if st.round >= self.max_rounds:
                    st.phase = "failed"
                    self._save(st, f"no passing candidate after {self.max_rounds} rounds (escalate to a person)")
                    break
                st.round += 1
                try:
                    st.candidate = int(await self.implementer(task_id, st.plan, st.findings, self.ws.committed[task_id]))
                except ImplementerFailed as e:
                    st.findings = [*st.findings[-8:], f"implementer: {e}"]
                    self._save(st, "no candidate", reason=str(e)[:500])
                    continue
                st.phase = "review_verify"
                self._save(st, "implemented", candidate=st.candidate)
            elif st.phase == "review_verify":
                ver = await self.ws.verify(task_id, st.candidate)
                rev = await self.reviewer(await self._review_input(task_id, st, ver))
                if rev.get("error"):  # the reviewer did not answer usably: that is not a rejection; review again (bounded)
                    st.review_errors += 1
                    self._save(st, "reviewer error", reason=str(rev["error"])[:300])
                    if st.review_errors >= 3:
                        st.phase = "failed"
                        self._save(st, "reviewer unavailable (escalate to a person)")
                    continue
                if ver["passed"] and not rev.get("approved") and not rev.get("findings") and st.review_errors < 3:
                    st.review_errors += 1  # a passing candidate rejected with no reason gets one fresh review before it is discarded
                    self._save(st, "rejected without findings: re-review")
                    continue
                if ver["passed"] and rev.get("approved"):
                    await self.ws.commit(task_id, st.candidate)
                    st.phase = "committed"
                    self._save(st, "verified and approved", verification=ver["checks"])
                else:
                    st.findings = [*ver["findings"], *[f"reviewer: {f}" for f in rev.get("findings") or []]]
                    if ver["passed"] and not rev.get("findings"):
                        st.findings.append("reviewer: rejected without findings")
                    st.phase = "implement"
                    self._save(st, "loop back", findings=st.findings, verified=ver["passed"], approved=bool(rev.get("approved")))
        return st


async def _complete_text(llm: Any, messages: list[dict[str, Any]], *, max_tokens: int, want_json: bool = False) -> str:
    """One completion with one retry when a reasoning model spends the budget thinking (empty text) or skips the JSON."""
    r = await llm.complete(messages, max_tokens=max_tokens, temperature=0.0)
    text = r.text or ""
    if not text.strip() or (want_json and not _json_block(text)):
        r = await llm.complete([*messages, {"role": "user", "content": "Reply now, briefly" + (", with the JSON object only." if want_json else ".")}],
                               max_tokens=max_tokens * 2, temperature=0.0)
        text = r.text or ""
    return text


def llm_planner(llm: Any, *, max_tokens: int = 4000) -> Callable[..., Awaitable[str]]:
    async def plan(task: dict[str, Any], findings: list[str]) -> str:
        return await _complete_text(llm, [{"role": "system", "content": "You plan repairs to RL task graders. Be specific: which check to add, change "
                                                                        "or drop, and why it closes each finding without rejecting a valid solution."},
                                          {"role": "user", "content": json.dumps({"task": task, "findings": findings}, default=str)[:30000]}],
                                    max_tokens=max_tokens)
    return plan


def llm_reviewer(llm: Any, *, max_tokens: int = 4000) -> Callable[..., Awaitable[dict[str, Any]]]:
    """An adversarial reviewer (use a different model than the implementer, in a fresh context): JSON {approved, findings};
    an unusable reply is ``{"error": ...}`` (the FSM re-reviews; it is not a rejection)."""
    async def review(summary: dict[str, Any]) -> dict[str, Any]:
        text = await _complete_text(llm, [{"role": "system", "content": "You review a change to an RL task's grader adversarially, looking for "
                                                                        "reasons it is wrong: does it grade a property the instruction does not "
                                                                        "state, reject a valid solution, or leave a shortcut open? The verification "
                                                                        "results come from the objective acceptance checks. Reply with JSON "
                                                                        "{\"approved\": bool, \"findings\": [str]} only."},
                                          {"role": "user", "content": json.dumps(summary, default=str)[:30000]}], max_tokens=max_tokens, want_json=True)
        d = _json_block(text)
        if not d or "approved" not in d:
            return {"approved": False, "findings": [], "error": f"no JSON verdict in the reviewer's reply: {text[:200]!r}"}
        return {"approved": bool(d.get("approved")), "findings": [str(x) for x in d.get("findings") or []]}
    return review


def rlm_implementer(llm: Any, workspace: ForgeWorkspace, **agent_kw: Any) -> Callable[..., Awaitable[int]]:
    """A child RLM agent with the ``forge`` proxy; it must finish with the candidate version number."""
    async def implement(task_id: str, plan: str, findings: list[str], base: int) -> int:
        handlers = {"forge.show": workspace.show, "forge.add_check": workspace.add_check, "forge.drop_check": workspace.drop_check,
                    "forge.vocabulary": workspace.vocabulary}
        agent = RLMAgent("implementer", llm, handlers=handlers, objects=("forge",), **{"max_turns": 20, **agent_kw})
        prompt = (f"Repair the grader of task {task_id!r} (committed version {base}). Plan:\n{plan}\n\nFindings to close:\n" + "\n".join(f"- {f}" for f in findings)
                  + "\n\nUse `await forge.vocabulary()` for the check grammar, `await forge.show(task_id=...)` to inspect the task, and `await forge.add_check(task_id=..., check={...}, base=...)` / "
                  "`await forge.drop_check(...)` to make a candidate version (each returns the new version number). Finish with "
                  "rlm.finish(str(version)) for the candidate to verify.")
        run = await agent.run(prompt)
        implement.runs.append(run)
        if not run.finished or not str(run.answer or "").strip().isdigit():
            raise ImplementerFailed(f"no candidate version returned (answer {str(run.answer)[:200]!r}, stop {run.stop}); "
                                    "finish with rlm.finish(str(version)) after forge.add_check returns the version")
        return int(str(run.answer).strip())
    implement.runs = []
    return implement


class RLMTaskManager:
    """The long-horizon manager: an RLM agent whose ``forge`` proxy can list, inspect and verify tasks and run the repair
    FSM; ``manage(goal)`` returns its run plus the FSM states it produced."""

    def __init__(self, llm: Any, workspace: ForgeWorkspace, fsm: TaskFSM, *, harness: Any = None, **agent_kw: Any):
        self.llm, self.ws, self.fsm, self.harness, self.kw = llm, workspace, fsm, harness, agent_kw
        self.fsm_results: dict[str, FSMState] = {}

    async def _fsm(self, task_id: str) -> dict[str, Any]:
        st = await self.fsm.run(task_id)
        self.fsm_results[task_id] = st
        return {"task_id": task_id, "phase": st.phase, "rounds": st.round, "committed": self.ws.committed[task_id], "findings": st.findings[-5:]}

    async def manage(self, goal: str, *, family: str = "*") -> AgentRun:
        handlers = {"forge.tasks": self.ws.tasks, "forge.show": self.ws.show, "forge.verify": self.ws.verify, "forge.repair": self._fsm}
        context = self.harness.digest(goal, family=family) if self.harness is not None else ""
        agent = RLMAgent("manager", self.llm, handlers=handlers, objects=("forge",), extra_context=context, **self.kw)
        return await agent.run(goal + "\n\nTools: `await forge.tasks()`, `await forge.show(task_id=...)`, `await forge.verify(task_id=...)`, "
                                      "`await forge.repair(task_id=...)` (runs plan -> implement -> review -> verify until the task passes).")
