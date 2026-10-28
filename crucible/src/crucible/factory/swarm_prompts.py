"""The swarm's prompt writer for the rebreak lanes (debug, kernel bug, kernel migration): the S7 report, written by the
``prompt_author`` role on the RLM harness and checked by the lane's own gates (swarm-pipeline/13 §4).

Until 2026-10-23 the bug reports of these lanes were written by the Claude builder into ``prompts/<id>.json`` by hand.
Now one small machine per window writes them, with no Claude in any role:

| state | owner | what decides |
|---|---|---|
| ``write`` | ``prompt_author`` (an M16 RLM session in a private clone of the start tree A', offline sandbox) | it writes ``prompt.json`` (``{opening, requirements: {mid: text}, closing, repro?}``) and ``STATUS.json``; the checkpoint sidecar grades the draft out of band (potential ``prompt_draft``: the lane's lint per requirement), so the session's Φ and per-call ΔΦ are journaled like a student's |
| ``lint`` | deterministic: the lane's check (``lint_debug`` + ledger, ``kernel_debug.author``, the migration author) | findings go back to ``write`` (up to ``max_writes``), then ``dropped`` |
| ``review`` | ``reviewer`` (a call, a family nothing built: provenance-checked) | blocking findings go back to ``write`` once; after that they are advisory (V6) and recorded next to the prompt |

The lane's ``author`` step then runs unchanged on the written file (V6: executed gates decide). Every row (sessions,
verdicts, failures, the run's end) goes to the ``SwarmJournal``, so ``scripts/swarm.py diagnose`` can blame and rank.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WRITER_BRIEF = """You are writing the bug report a real user of this repository would file. The repository in your worktree is at the
state the user has: several bugs are present. The hand-off holds the maintainers' notes for each bug (the fix's message
and diff, the tests that catch it): they are for your understanding only. Never quote them, never name the faulty
functions, kernels, files, test names, commit hashes, versions or links, never say how to fix anything, and never say
why it goes wrong inside the code (no diagnosis: report what you observe, as a user would).

Write `prompt.json` in the worktree root (when the hand-off gives a `format`, follow it instead):
{"opening": "<one or two sentences: who you are and what you were doing>",
 "requirements": {<one entry per milestone id listed in the hand-off>: "<what goes wrong, observed from the outside: inputs, outputs, error messages, and what must hold instead>"},
 "closing": "<optional one sentence>"}
When the hand-off has a `revision`, this is a revised report: follow its `instruction`, and use its `current_report` (what the
user's previous report said) for everything the instruction does not change. The hand-off's `forbidden_strings` must not appear anywhere in the file. When a milestone in the hand-off lists `messages`, the hidden checks look for each of those exact texts in the program's output (an error message, a printed line): quote every one word for word in that milestone's requirement, as what the program must print or return. Cover every case the hidden tests check: an input variant or edge case a test or a table row names (a line ending such as CRLF, an empty or null value, a stale status, a boundary) must appear in its requirement as what the user sees. Each requirement must be specific enough that a competent engineer can reproduce the symptom, and must only state
behaviour the hidden checks test. Use the repository (read the code, run what you can in the sandbox) to get the user's
view right. Keep `STATUS.json` in the worktree root up to date: {"<milestone id>": <your probability that its requirement
is right and complete>}. When the file is written, call rlm.finish("<one line>")."""

REVIEW_SYSTEM = ("You review a bug report written for a debugging task. You see the maintainers' notes (the real fixes) and the "
                 "report. Block it when a requirement leaks the fix (names a faulty function, file, kernel or test, or says how to fix), "
                 "when a requirement does not describe an observable symptom, when it asks for behaviour the notes do not show is "
                 "checked, or when the notes' tests check a case the requirement never mentions (an input variant, a line ending, an "
                 "edge case, a table row's condition): name the case. Reply with one JSON object: {\"ok\": true|false, \"findings\": [\"<milestone id>: <problem>\", ...]}.")


def start_repo(tree: Path | str, dest: Path | str, *, tag: str = "A") -> Path:
    """A git repository holding ``tree`` (a plain start tree, e.g. a debug window's ``base``) as one commit tagged ``tag``,
    made once (the session's ``GitWorkspace`` fetches that commit into a private clone)."""
    dest = Path(dest)
    if (dest / ".git").exists():
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(tree, tmp, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    env = {"GIT_AUTHOR_NAME": "crucible", "GIT_AUTHOR_EMAIL": "crucible@local", "GIT_COMMITTER_NAME": "crucible", "GIT_COMMITTER_EMAIL": "crucible@local",
           "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A", "-f"], ["git", "commit", "-qm", "start"], ["git", "tag", tag]):
        subprocess.run(cmd, cwd=tmp, check=True, capture_output=True, env=env, timeout=600)
    tmp.rename(dest)
    return dest


def prompt_machine(*, budget_usd: float = 1.5, max_turns: int = 60, max_writes: int = 3, checkpoint_every: int = 5) -> dict[str, Any]:
    j = "json"
    v = [{"name": "verdict", "type": j}]
    states = [
        {"id": "write", "entry": True, "role": "prompt_author", "mutates": True, "recovery": "reconcile", "max_entries": max_writes + 1,
         "inputs": [{"name": "brief", "type": j, "from": "params.spec"}],
         "session": {"brief": WRITER_BRIEF, "captures": [{"output": "prompt", "kind": "json", "path": "prompt.json"}], "potential": "prompt_draft",
                     "max_turns": max_turns, "checkpoint_every": checkpoint_every, "checkpoint_cap": 12, "settle_on_budget": True},
         "outputs": [{"name": "prompt", "type": j}]},
        {"id": "lint", "fn": "lint", "max_entries": max_writes + 1, "inputs": [{"name": "prompt", "type": j, "from": "write.prompt"}], "outputs": v},
        {"id": "review", "role": "reviewer", "handler": "review", "max_entries": max_writes + 1, "different_family_from": ["write"],
         "inputs": [{"name": "prompt", "type": j, "from": "write.prompt"}, {"name": "brief", "type": j, "from": "params.spec"}], "outputs": v},
        {"id": "written", "terminal": "done"}, {"id": "dropped", "terminal": "rejected"}]
    t = [{"from": "write", "to": "lint"},
         {"from": "lint", "to": "review", "when": "verdict.ok == true"},
         {"from": "lint", "to": "write", "when": "verdict.ok == false and verdict.retry == true", "carry": "verdict.findings"},
         {"from": "lint", "to": "dropped", "when": "verdict.ok == false"},
         {"from": "review", "to": "write", "when": "verdict.ok == false and verdict.retry == true", "carry": "verdict.findings"},
         {"from": "review", "to": "written"}]
    return {"run": {"max_transitions": 4 * (max_writes + 1), "budget_usd": budget_usd}, "states": states, "transitions": t}


def archive_stale_run(root: Path, run_id: str, machine: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> list[Path]:
    """A run whose journal belongs to another machine (the brief or the states changed since) or to other parameters (a
    revision's new hand-off under the same id) cannot be resumed: resuming replays the old outputs, and the session's
    hand-off branch hands the writer its old ``prompt.json`` back. Its run directory, its session state and its hand-off
    branch are moved aside (``<dir>.prev-<n>``, never deleted) so the new run starts fresh. Returns what was moved."""
    from crucible.factory.machine import canonicalize, machine_hash

    root = Path(root)
    j = root / "runs" / run_id / "journal.jsonl"
    if not j.exists():
        return []
    first = next((json.loads(x) for x in j.read_text().splitlines() if x.strip()), {})
    same_machine = first.get("machine_hash") in (None, machine_hash(canonicalize(machine)))
    same_params = params is None or "params" not in first or json.dumps(first["params"], sort_keys=True, default=str) == json.dumps(dict(params), sort_keys=True, default=str)
    if same_machine and same_params:
        return []
    n = 1
    while Path(f"{root / 'runs' / run_id}.prev-{n}").exists():
        n += 1
    moved = []
    for d in (root / "runs" / run_id, root / "sessions" / "runs" / run_id, root / "family" / "runs" / run_id):  # SessionPool keeps sessions/runs/<id>
        if d.exists():
            dest = Path(f"{d}.prev-{n}")
            d.rename(dest)
            moved.append(dest)
    return moved


@dataclass
class PromptJob:
    """One window's report. ``brief``: the hand-off (milestone ids, the maintainers' notes per milestone, the lane's rules);
    ``tree``: the start tree A' (plain directory); ``lint(draft) -> findings``: the lane's S7 check of a whole draft;
    ``lint_text(text) -> findings``: the same lint on one line (the potential); ``out``: where ``prompts/<id>.json`` goes."""

    task_id: str
    milestone_ids: Sequence[str]
    brief: Mapping[str, Any]
    tree: Path
    lint: Callable[[Mapping[str, Any]], list[str]]
    lint_text: Callable[[str], Sequence[str]]
    out: Path


async def write_prompt(job: PromptJob, *, book: Any, root: Path | str, journal: Any, kernel: str = "sandbox", budget_usd: float = 1.5,
                       max_turns: int = 60, max_writes: int = 3, rlm_config: Any = None) -> dict[str, Any]:
    """Run the prompt machine for ``job`` with ``book``'s ``prompt_author`` and ``reviewer``. Writes ``job.out`` on
    ``written`` (the builder's ref, the provenance, the review column) and returns the run's summary."""
    from crucible.factory.builder_potential import prompt_draft
    from crucible.factory.handoff import TaskRepo
    from crucible.factory.machine import MachineRun, Settle
    from crucible.factory.provenance import ProvenanceLedger, family_of_client
    from crucible.factory.sessions import SessionPool

    root = Path(root)
    run_id = f"prompt-{job.task_id}"
    src = start_repo(job.tree, root / "starts" / job.task_id)
    machine = prompt_machine(budget_usd=budget_usd, max_turns=max_turns, max_writes=max_writes)
    clients = {r: book.client(r) for r in ("prompt_author", "reviewer")}
    params = {"spec": dict(job.brief), "repo": str(src), "base": "A", "task_id": job.task_id}  # the hand-off always carries the spec
    repo = TaskRepo(root / "family", "prompts")
    pool = SessionPool(root / "sessions", repo=repo, machine=machine, run_id=run_id, params=params, llms=clients, kernel=kernel,
                       potentials={"prompt_draft": lambda ctx, ws: prompt_draft(job.task_id, ws.path, list(job.milestone_ids), lint=job.lint_text)},
                       on_session=journal.session, rlm_config=rlm_config)
    ledger = ProvenanceLedger(machine, family_for=lambda role: family_of_client(clients.get(role)), model_for=lambda role: str(getattr(clients.get(role), "model", "")))
    review_log: list[dict[str, Any]] = []

    async def lint(ctx: Any) -> Any:
        draft = ctx.inputs.get("prompt") or {}
        found = job.lint(draft)
        return Settle({"verdict": {"ok": not found, "findings": found[:30], "owner": "write", "retry": ctx.entry <= max_writes}}, found[:30])

    async def review(ctx: Any) -> Any:
        draft = ctx.inputs.get("prompt") or {}
        msgs = [{"role": "system", "content": REVIEW_SYSTEM},
                {"role": "user", "content": json.dumps({"notes": job.brief, "report": draft}, default=str)[:60_000]}]
        try:
            d = await clients["reviewer"].complete_json(msgs, schema=["ok", "findings"], max_tokens=2000)
        except Exception as e:  # noqa: BLE001 -- a reviewer that cannot answer is advisory-absent, never a pass or a block
            review_log.append({"entry": ctx.entry, "ok": None, "error": f"{type(e).__name__}: {e}"[:300]})
            return Settle({"verdict": {"ok": True, "findings": [], "owner": "", "retry": False, "advisory": True, "error": str(e)[:300]}})
        ok, found = bool(d.get("ok")), [str(x) for x in d.get("findings") or []][:20]
        review_log.append({"entry": ctx.entry, "ok": ok, "findings": found})
        blocking = not ok and ctx.entry == 1  # once: after that the reviewer's verdict is a column, not a gate (V6)
        return Settle({"verdict": {"ok": ok, "findings": found, "owner": "write" if blocking else "", "retry": blocking}}, found if blocking else [])

    handlers = journal.guard({"session": pool.handler, "lint": lint, "review": review}, ledger)
    run_dir = root / "runs" / run_id
    archive_stale_run(root, run_id, machine, params)
    run = MachineRun(machine, run_dir=run_dir, run_id=run_id, params=params, handlers=handlers, clients=clients, **pool.machine_kwargs())
    try:
        res = await run.run()
    finally:
        await pool.close()
    usd = float(getattr(run.ledger, "committed", 0.0) or 0.0)
    written = res.status == "done"
    summary = {"task": job.task_id, "status": res.status, "reason": res.reason, "usd": usd, "sessions": len(pool.records),
               "phi_final": [r.phi_final for r in pool.records], "review": review_log, "provenance": ledger.to_json()}
    if written:
        draft = dict((res.outputs.get("write") or {}).get("prompt") or {})
        spec = book.spec("prompt_author")
        draft.update(builder=spec.ref, swarm={"run": run_id, "provenance": ledger.states.get("write").to_json() if ledger.states.get("write") else {},
                                               "review": review_log, "usd": round(usd, 6)})
        job.out.parent.mkdir(parents=True, exist_ok=True)
        job.out.write_text(json.dumps(draft, indent=1))
    journal.run_end(run=run_id, status=res.status, reason=res.reason, usd=usd, task=job.task_id, accepted=written)
    return summary
