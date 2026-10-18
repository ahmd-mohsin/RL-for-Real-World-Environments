"""Agentic-session states (P1; docs/research/swarm-pipeline/00 §3.2, 01 fact 10 and §6.2).

A machine state with a ``session`` block is backed by an **M16 RLM agent** (``actors.rlm.policy.RLMPolicy``: a persistent
kernel, a ledger, compaction, the agent tree) working in **its own git worktree and its own Docker sandbox**, instead of
one model call. Prime Agent's factory has none of this: its children share the parent's filesystem view and nothing
gives a state a worktree or a sandbox (study 01 §3.4); the blog's "dedicated worktree" implementers and "fresh sandbox"
verifiers are process description.

- **Worktree** (``GitWorkspace``): a private repository holding only the source repo's ``base`` commit (``git fetch
  <source> <base>`` into a fresh ``git init``, tagged ``base``), one per (run, state). Not a ``git worktree``: worktrees
  share the object store and refs, so a verifier writer's ``git log --all`` would list the oracle builder's commits (V2:
  generation and verification never share a context). It is the machine's ``MutableTree``: ``snapshot`` writes the
  tree (untracked files too) as a commit under ``refs/crucible/snapshots/``, ``restore`` puts HEAD and the files back,
  ``diff`` is the change since a snapshot (the reconcile hand-off). Ignored files are not snapshotted.
- **Sandbox** (``Sandbox``): a ``crucible-p1-*`` container, ``--network none`` (sessions are offline; the validator
  refuses ``session.network: true``), the worktree bind-mounted at its own absolute path and the hand-off directory
  read-only at ``/handoff``. The agent's kernel runs **inside** it (``ExecLauncher`` over ``docker exec``), and its tool
  surface is the World's ``common/git_repo`` App (``GitRepoApp``: list/read/write/edit/search files, ``shell``,
  ``git_status``, ``git_diff``) whose commands run in the same container (the windows lane's ``open_world`` pattern).
- **Hand-off in, files out.** The session's TASK is its brief plus the typed hand-off (``handoff.build_handoff``: every
  input port against its schema, the full original spec attached), stored in the task repo first. Outputs are
  **captures**: files the agent writes in its worktree, copied into the task repo under
  ``runs/<run>/<state>/<entry>.<attempt>/`` and committed with ``informed_by=<hand-off commit>`` (Prime: "have children
  write files and read those files for fan-in"). Kinds: ``json`` / ``text`` (a file parsed into a json / text port),
  ``file`` / ``dir`` (an artifact port), ``patch`` (the worktree's diff against ``base``, untracked files included) and
  ``commits`` (``git format-patch base..HEAD``: the oracle builder's one commit per milestone). A capture that finds
  nothing leaves its port unbound, so the machine fails the producing state (P1 capture rule).
- **Infra vs task.** A sandbox that cannot start, a kernel or container lost mid-episode, a provider outage raise
  ``InfraError`` (the machine re-checks with ``probe`` and re-runs once for free); the agent's ``give_up``, a budget
  stop, a stall are task failures (``StateFailed``).
- **Budgets.** ``ctx.budget`` (the state's and the run's remaining dollars and tokens) bounds the agent's tree tokens;
  the session's model tokens are booked to the state (``machine.book_tokens``) and its dollars through the LLM client's
  ``SpendScope``.
- **Persistent specialists** (``SpecialistBook``): ``session.specialist`` (a template over params, e.g.
  ``"world_expert:{repo_name}"``) names a specialist that outlives a machine run: one World expert per repo, one verifier
  writer or hacker per family. Its ledger lives under ``specialists/<id>/`` and every session of it opens a new window
  in the same ledger, so its context across runs is its ledger (``history.windows()``, ``history.search()``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.handoff import HandoffError, TaskRepo, build_handoff
from crucible.factory.machine import PARAMS, CaptureLost, InfraError, Settle, StateContext, StateFailed, artifact_digest, book_tokens

PREFIX = "crucible-p1-"
SESSION_IMAGE = "crucible-p1-session:py312"
SESSION_DOCKERFILE = """FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*
RUN git config --system safe.directory "*"
ENV PYTHONDONTWRITEBYTECODE=1
"""
HANDOFF_MOUNT = "/handoff"
BRIEF_HANDOFF_CHARS = 24_000  # the hand-off JSON inline in TASK (the full file is always at /handoff/handoff.json)
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe(s: str) -> str:
    return _SAFE.sub("-", str(s)).strip("-") or "x"


def docker(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)


def docker_infra_kind(text: str) -> str:
    from crucible.actors.rlm.kernel import classify_sandbox_error

    return classify_sandbox_error(text)


def ensure_image(tag: str = SESSION_IMAGE) -> str:
    """The session image (Python 3.12 + git), built once from ``SESSION_DOCKERFILE``; returns its id."""
    r = docker("image", "inspect", "--format", "{{.Id}}", tag, timeout=60)
    if r.returncode == 0:
        return r.stdout.strip()
    import tempfile

    with tempfile.TemporaryDirectory(prefix="crucible-p1-img-") as d:
        (Path(d) / "Dockerfile").write_text(SESSION_DOCKERFILE)
        b = docker("build", "-q", "-t", tag, d, timeout=1800)
    if b.returncode != 0:
        raise InfraError(f"building {tag} failed: {b.stderr[-400:]}", "crucible:env_build_failed")
    return docker("image", "inspect", "--format", "{{.Id}}", tag, timeout=60).stdout.strip()


def own_containers() -> list[str]:
    r = docker("ps", "-a", "--filter", f"name={PREFIX}", "--format", "{{.Names}}", timeout=60)
    return [x for x in r.stdout.split() if x.startswith(PREFIX)]


# ---- the worktree -------------------------------------------------------------------------------------------------------
async def _git(cwd: Path, *args: str, env: Mapping[str, str] | None = None, check: bool = True, timeout: float = 300) -> str:
    def go() -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
                              check=False, env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})})

    r = await asyncio.to_thread(go)
    if check and r.returncode != 0:
        raise StateFailed(f"git {' '.join(args[:2])} in {cwd.name}: {r.stderr.strip()[-400:]}")
    return r.stdout


class GitWorkspace:
    """A session's private repository at the source's ``base`` commit (see the module doc); the machine's ``MutableTree``."""

    def __init__(self, path: Path, *, source: Path, base: str):
        self.path, self.source, self.base = Path(path).resolve(), Path(source).resolve(), base
        self._n = 0

    @property
    def exists(self) -> bool:
        return (self.path / ".git").is_dir()

    async def create(self) -> GitWorkspace:
        if self.exists:
            return self
        self.path.mkdir(parents=True, exist_ok=True)
        await _git(self.path, "init", "-q", "-b", "work")
        await _git(self.path, "fetch", "-q", "--no-tags", str(self.source), self.base)
        await _git(self.path, "checkout", "-q", "-B", "work", "FETCH_HEAD")
        await _git(self.path, "tag", "-f", "base")
        await _git(self.path, "config", "user.email", "session@crucible.local")
        await _git(self.path, "config", "user.name", "crucible-session")
        return self

    async def head(self) -> str:
        return (await _git(self.path, "rev-parse", "HEAD")).strip()

    async def _tree_commit(self, parent: str, message: str) -> str:
        """The working tree (tracked + untracked, minus ignored) as a commit, through a throwaway index."""
        idx = self.path / ".git" / f"crucible-{uuid.uuid4().hex[:8]}.idx"
        env = {"GIT_INDEX_FILE": str(idx)}
        try:
            await _git(self.path, "add", "-A", env=env)
            tree = (await _git(self.path, "write-tree", env=env)).strip()
        finally:
            idx.unlink(missing_ok=True)
        return (await _git(self.path, "commit-tree", tree, "-p", parent, "-m", message)).strip()

    async def snapshot(self, label: str) -> str:
        head = await self.head()
        snap = await self._tree_commit(head, f"crucible snapshot {label}")
        self._n += 1
        await _git(self.path, "update-ref", f"refs/crucible/snapshots/{safe(label)}-{uuid.uuid4().hex[:6]}", snap)
        return f"{head}:{snap}"

    async def restore(self, token: str) -> None:
        head, snap = token.split(":", 1)
        await _git(self.path, "reset", "-q", "--hard", head)
        await _git(self.path, "clean", "-fdq")
        await _git(self.path, "read-tree", "-u", "--reset", snap)
        await _git(self.path, "reset", "-q")  # the index back to HEAD; the files stay as the snapshot had them

    async def diff(self, token: str) -> str:
        head, snap = token.split(":", 1)
        now = await self._tree_commit(await self.head(), "crucible diff")
        commits = (await _git(self.path, "log", "--format=%h %s", f"{head}..HEAD")).strip()
        body = await _git(self.path, "diff", "--no-color", snap, now)
        return (f"# commits since the snapshot:\n{commits}\n" if commits else "") + body

    async def patch(self, ref: str = "base") -> str:
        """The worktree's change against ``ref`` (untracked files included)."""
        now = await self._tree_commit(await self.head(), "crucible patch")
        return await _git(self.path, "diff", "--no-color", "--binary", ref, now)

    async def commits(self, ref: str = "base") -> tuple[str, list[str]]:
        subjects = [s for s in (await _git(self.path, "log", "--reverse", "--format=%s", f"{ref}..HEAD")).splitlines() if s]
        mbox = await _git(self.path, "format-patch", "--stdout", "--no-signature", f"{ref}..HEAD") if subjects else ""
        return mbox, subjects

    def remove(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


# ---- the sandbox --------------------------------------------------------------------------------------------------------
class Sandbox:
    """One ``crucible-p1-*`` container, offline, with the worktree and the read-only hand-off directory mounted."""

    def __init__(self, *, workspace: Path, handoff_dir: Path, image: str = SESSION_IMAGE, memory: str = "2g", cpus: str = "2"):
        self.workspace, self.handoff_dir, self.image, self.memory, self.cpus = Path(workspace), Path(handoff_dir), image, memory, cpus
        self.name: str | None = None

    async def start(self) -> str:
        name = f"{PREFIX}{uuid.uuid4().hex[:12]}"
        ws, ho = str(self.workspace.resolve()), str(self.handoff_dir.resolve())
        r = await asyncio.to_thread(docker, "run", "-d", "--name", name, "--network", "none", "--memory", self.memory, "--cpus", self.cpus,
                                    "--pids-limit", "512", "-v", f"{ws}:{ws}", "-v", f"{ho}:{HANDOFF_MOUNT}:ro", "-w", ws, "-e", "HOME=/tmp",
                                    "--label", "crucible.p1=session", self.image, "sleep", "infinity")
        if r.returncode != 0:
            await asyncio.to_thread(docker, "rm", "-f", name, timeout=60)
            raise InfraError(f"the session sandbox did not start: {r.stderr.strip()[-400:]}", docker_infra_kind(r.stderr))
        self.name = name
        return name

    async def alive(self) -> bool:
        if not self.name:
            return False
        r = await asyncio.to_thread(docker, "inspect", "-f", "{{.State.Running}}", self.name, timeout=60)
        return r.returncode == 0 and r.stdout.strip() == "true"

    async def remove(self) -> None:
        if self.name:
            await asyncio.to_thread(docker, "rm", "-f", self.name, timeout=120)
        self.name = None

    def exec_prefix(self) -> list[str]:
        assert self.name
        return ["docker", "exec", "-i", self.name]


# ---- persistent specialists ---------------------------------------------------------------------------------------------
@dataclass
class Specialist:
    id: str
    role: str
    ledger_dir: str
    sessions: list[dict[str, Any]] = field(default_factory=list)


class SpecialistBook:
    """Specialists that outlive a machine run (``specialists.json`` under ``root``, a ledger directory each)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "specialists.json"

    def _load(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self, d: Mapping[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, sort_keys=True, indent=1))
        tmp.replace(self.path)

    def get(self, sid: str) -> Specialist | None:
        d = self._load().get(sid)
        return Specialist(**d) if d else None

    def resume(self, sid: str, role: str) -> Specialist:
        """The specialist ``sid`` (created on first use): its ledger is its context across runs."""
        d = self._load()
        if sid not in d:
            d[sid] = {"id": sid, "role": role, "ledger_dir": str(self.root / safe(sid)), "sessions": []}
            self._save(d)
        return Specialist(**d[sid])

    def record(self, sid: str, row: Mapping[str, Any]) -> None:
        d = self._load()
        d[sid]["sessions"].append(dict(row))
        self._save(d)

    def list(self) -> list[Specialist]:
        return [Specialist(**v) for v in self._load().values()]


def _session_policy_class() -> Any:
    from crucible.actors.rlm.ledger import Ledger
    from crucible.actors.rlm.policy import RLMPolicy

    class SessionPolicy(RLMPolicy):
        """An ``RLMPolicy`` whose ledger is a persistent specialist's: reopened, with a new window for this session."""

        def __init__(self, *a: Any, specialist_ledger: Path | None = None, session_meta: Mapping[str, Any] | None = None, **kw: Any):
            super().__init__(*a, **kw)
            self._spec_ledger = specialist_ledger
            self._session_meta = dict(session_meta or {})

        def _make_ledger(self) -> Ledger:
            if self._spec_ledger is None:
                return super()._make_ledger()
            root = Path(self._spec_ledger)
            ev = root / "root" / "events.jsonl"
            if ev.exists() and ev.stat().st_size:
                led = Ledger.open(root, name="root")
                led.open_window("start", kind="specialist_session", **{k: str(v) for k, v in self._session_meta.items()})
                return led
            return Ledger(root, name="root")

    return SessionPolicy


# ---- the pool: one machine run's sessions -------------------------------------------------------------------------------
@dataclass
class SessionRecord:
    state: str
    entry: int
    attempt: int
    stop: str | None
    turns: int
    tokens: int
    usd: float
    container: str | None
    handoff_commit: str
    capture_commit: str | None
    specialist: str | None = None


def _resolve(ref: Any, params: Mapping[str, Any]) -> Any:
    if isinstance(ref, str) and ref.startswith(PARAMS + "."):
        key = ref.partition(".")[2]
        if key not in params:
            raise StateFailed(f"session reference {ref!r} names a run parameter the run does not have")
        return params[key]
    return ref


class SessionPool:
    """Sessions of one machine run. ``handler`` serves every session state; ``tree`` / ``probe`` / ``artifact_root`` go to
    ``MachineRun`` (``machine_kwargs()``). ``llms`` maps a role to its model client (or ``llm_for(role)``); ``kernel``:
    ``"sandbox"`` (default: the kernel inside the session's container) or ``"local"`` (trusted scripted code only:
    unit tests without Docker)."""

    def __init__(self, root: Path, *, repo: TaskRepo, machine: Mapping[str, Any], run_id: str, params: Mapping[str, Any],
                 llms: Mapping[str, Any] | None = None, llm_for: Callable[[str], Any] | None = None, image: str = SESSION_IMAGE,
                 specialists: SpecialistBook | None = None, kernel: str = "sandbox", max_steps: int = 400, rlm_config: Any = None,
                 sandbox_factory: Callable[..., Sandbox] | None = None, task_id: str | None = None):
        from crucible.factory.machine import canonicalize

        self.root = Path(root).resolve()
        self.repo, self.run_id, self.params = repo, run_id, dict(params)
        self.machine = canonicalize(machine)
        self.states = {s["id"]: s for s in self.machine["states"]}
        self.llms, self.llm_for = dict(llms or {}), llm_for
        self.image, self.kernel, self.max_steps, self.rlm_config = image, kernel, max_steps, rlm_config
        self.specialists = specialists or SpecialistBook(self.root / "specialists")
        self.sandbox_factory = sandbox_factory or Sandbox
        self.task_id = task_id or str(self.params.get("task_id") or run_id)
        self.workspaces: dict[str, GitWorkspace] = {}
        self.sandboxes: dict[str, Sandbox] = {}
        self.records: list[SessionRecord] = []
        self.dir = self.root / "runs" / safe(run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        if kernel not in ("sandbox", "local"):
            raise ValueError("SessionPool.kernel: sandbox | local")

    # ---- MachineRun hooks ---------------------------------------------------------------------------------------------
    def machine_kwargs(self) -> dict[str, Any]:
        return {"trees": self.tree, "infra_probe": self.probe, "artifact_root": self.repo.root}

    def _session(self, sid: str) -> dict[str, Any]:
        st = self.states.get(sid)
        if st is None or "session" not in st:
            raise StateFailed(f"state {sid} is not a session state")
        return st["session"]

    def workspace(self, sid: str) -> GitWorkspace:
        if sid not in self.workspaces:
            sess = self._session(sid)
            source = _resolve(sess.get("repo", "params.repo"), self.params)
            base = _resolve(sess.get("base", "params.base"), self.params)
            self.workspaces[sid] = GitWorkspace(self.dir / "ws" / safe(sid), source=Path(source), base=str(base))
        return self.workspaces[sid]

    async def tree(self, sid: str) -> GitWorkspace:
        return await self.workspace(sid).create()

    async def probe(self, sid: str, err: BaseException) -> str:
        """The infra re-check (machine ``infra_probe``): ``alive`` when the state's sandbox still answers; otherwise the
        zombie is removed (a respawn never runs beside it in the same worktree, F:3036-3046) and ``dead``."""
        sb = self.sandboxes.get(sid)
        if sb is not None and await sb.alive():
            return "alive"
        if sb is not None:
            await sb.remove()
            self.sandboxes.pop(sid, None)
        return "dead"

    async def close(self) -> None:
        for sb in list(self.sandboxes.values()):
            with contextlib.suppress(Exception):
                await sb.remove()
        self.sandboxes.clear()

    # ---- one session ----------------------------------------------------------------------------------------------------
    def _llm(self, role: str, ctx: StateContext) -> Any:
        if role in self.llms:
            return self.llms[role]
        if self.llm_for is not None:
            return self.llm_for(role)
        if ctx.client is not None and hasattr(ctx.client, "complete"):
            return ctx.client
        raise StateFailed(f"no model client for role {role!r}")

    async def _sandbox(self, sid: str, ws: GitWorkspace, handoff_dir: Path) -> Sandbox:
        sb = self.sandboxes.get(sid)
        if sb is not None and await sb.alive():
            return sb
        if sb is not None:
            await sb.remove()
        sb = self.sandbox_factory(workspace=ws.path, handoff_dir=handoff_dir, image=self.image)
        await sb.start()
        self.sandboxes[sid] = sb
        return sb

    def _stage(self, ctx: StateContext, handoff_json: str, attempt_dir: Path) -> Path:
        """The read-only ``/handoff`` directory of the state (one per state: the sandbox mounts it at start, so it is
        emptied and refilled in place, never replaced): the hand-off and a copy of each input artifact, nothing else."""
        d = self.dir / safe(ctx.state_id) / "handoff"
        d.mkdir(parents=True, exist_ok=True)
        for child in d.iterdir():
            (shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink())
        (attempt_dir / "handoff.json").write_text(handoff_json)
        (d / "handoff.json").write_text(handoff_json)
        for inp in ctx.state.get("inputs", []):
            v = ctx.inputs.get(inp["name"])
            if inp["type"] != "artifact" or not isinstance(v, dict):
                continue
            src = self.repo.root / v["path"]
            dst = d / safe(inp["name"]) / Path(v["path"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree(src, dst, symlinks=True) if src.is_dir() else shutil.copy2(src, dst))
        return d

    def _brief(self, ctx: StateContext, sess: Mapping[str, Any], handoff_json: str, spec: Any) -> str:
        role = ctx.role or ctx.state_id
        caps = sess.get("captures", [])
        ws_line = ("[workspace] Your git worktree is the World app `repo` (tools repo.*; repo.shell runs in your offline sandbox). "
                   "It holds only the start commit, tagged `base`.")
        ho_line = (f"[hand-off] The typed hand-off (JSON: every input with its schema, and the full original spec) is the file "
                   f"{HANDOFF_MOUNT}/handoff.json (read-only); input artifacts are under {HANDOFF_MOUNT}/<input>/. It reads:")
        body = handoff_json if len(handoff_json) <= BRIEF_HANDOFF_CHARS else handoff_json[:BRIEF_HANDOFF_CHARS] + "\n... (cut: read the file)"
        lines = [f"ROLE: {role}", f"(factory state {ctx.state_id}, entry {ctx.entry}, attempt {ctx.attempt}; run {ctx.run_id})", "",
                 str(sess.get("brief") or f"You are the {role}.").strip(), "", ws_line, "", ho_line, body]
        if caps:
            lines += ["", "[outputs] Your work products are files in your worktree, collected when you finish (never your messages):"]
            for c in caps:
                what = {"json": "a JSON file", "text": "a text file", "file": "a file", "dir": "a directory",
                        "patch": "your change against `base` (every file you edit)", "commits": "your commits on top of `base` (git commit each one)"}[c["kind"]]
                lines.append(f"- {c['output']}: {what}" + (f" at `{c['path']}`" if c.get("path") else ""))
        if ctx.findings:
            lines += ["", "[findings] This entry was sent back to you with these findings:"] + [f"- {f}" for f in ctx.findings]
        rec = ctx.recovery
        if rec:
            why = [f"- {f}" for f in rec.get("findings", [])]
            if rec["mode"] == "reset":
                lines += ["", ("[recovery: reset] An earlier attempt of this step failed; your worktree was reset to how it was before "
                               "it. Why it failed:"), *why]
            else:
                lines += ["", ("[recovery: reconcile] An earlier attempt of this step failed and left the worktree changed. Start from "
                               "`git status` and `git diff`: keep the changes that already serve the task, revert half-finished edits, "
                               "never redo finished work. Why it failed:"), *why]
                if rec.get("diff"):
                    lines += ["The earlier attempt's change:", rec["diff"]]
        b = ctx.budget or {}
        if b.get("tokens") is not None or b.get("usd") is not None:
            lines += ["", f"[budget] tokens left: {b.get('tokens')}, USD left: {b.get('usd')}"]
        lines += ["", "When every output is written, call rlm.finish(\"<one-line summary>\"). If the step cannot be done, call rlm.give_up(reason)."]
        return "\n".join(lines)

    def _world(self, ws: GitWorkspace, sb: Sandbox | None, brief: str, attempt_dir: Path) -> Any:
        from crucible.core.schema.world import WorldSpec
        from crucible.core.store import Store
        from crucible.world.world import World

        spec = WorldSpec.model_validate({
            "world": "factory/session", "network": "no-network",
            "apps": [{"id": "repo", "template": "common/git_repo", "state": {"adapter": "fs", "params": {"root": str(ws.path)}},
                      "params": {"shell": True, "init_git": False, "max_output": 30000,
                                 **({} if sb is not None else {"allow_unsafe_local_shell": True})}}],
            "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}]})
        return World(spec, instructions={"agent": brief}, store=Store(attempt_dir / "store"), workdir=attempt_dir / "world")

    async def handler(self, ctx: StateContext) -> Settle:
        """The machine handler of every session state."""
        from crucible.actors.policy import EpisodeLimits, run_episode
        from crucible.actors.rlm.agent import RLMConfig
        from crucible.actors.rlm.kernel import ExecLauncher, LocalLauncher
        from crucible.world.exec import DockerExec

        sid = ctx.state_id
        sess = self._session(sid)
        attempt_dir = self.dir / safe(sid) / f"{ctx.entry}.{ctx.attempt}{'r' if ctx.respawned else ''}-{uuid.uuid4().hex[:6]}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if sess.get("read_only"):  # a fresh tree for every attempt: nothing to recover, so no recovery is declared
            self.workspace(sid).remove()
        ws = await self.workspace(sid).create()
        try:
            h = build_handoff(ctx, self.states)
        except HandoffError as e:  # a producer settled a value that breaks its port schema: the producer runs again
            if e.producers:
                raise CaptureLost(e.producers[0], f"hand-off into {sid}: {e}") from e
            raise StateFailed(str(e)) from e
        hj = h.to_json()
        rel = f"runs/{safe(self.run_id)}/{safe(sid)}/{attempt_dir.name}"  # unique: a resumed entry restarts its attempt numbers
        upstream = [v.get("commit") for v in ctx.inputs.values() if isinstance(v, dict) and v.get("commit")]
        hcommit = self.repo.commit({f"{rel}/handoff.json": hj + "\n"}, f"hand-off into {sid} #{ctx.entry}.{ctx.attempt}",
                                   informed_by=upstream, trailers={"run": self.run_id, "state": sid, "handoff": h.id, "role": str(ctx.role)})
        handoff_dir = self._stage(ctx, hj, attempt_dir)
        brief = self._brief(ctx, sess, hj, ctx.params.get("spec"))
        sb = await self._sandbox(sid, ws, handoff_dir) if self.kernel == "sandbox" else None
        spec_id = None
        spec_ledger = None
        if sess.get("specialist"):
            spec_id = str(sess["specialist"]).format(**{k: v for k, v in self.params.items() if isinstance(v, (str, int, float))})
            spec_ledger = Path(self.specialists.resume(spec_id, str(ctx.role)).ledger_dir)
            prior = len(self.specialists.get(spec_id).sessions)  # type: ignore[union-attr]
            if prior:
                brief += (f"\n\n[memory] You are the persistent specialist `{spec_id}`. Your ledger holds your {prior} earlier session(s): "
                          "`await history.windows()`, `await history.search(\"...\")`.")
        budget_tokens = (ctx.budget or {}).get("tokens")
        cfg = self.rlm_config or RLMConfig(temperature=0.0, tree_max_depth=0, wall_s=float(sess.get("wall_s", 3600)),
                                           max_turns=int(sess.get("max_turns", 200)))
        if budget_tokens is not None:
            import dataclasses

            cfg = dataclasses.replace(cfg, tree_max_total_new_tokens=max(1, int(budget_tokens)))
        if sb is not None:
            dex = DockerExec(sb.name)  # type: ignore[arg-type]

            def launcher() -> Any:
                return ExecLauncher(sb.exec_prefix(), push=dex.push_file)
        else:
            def launcher() -> Any:
                return LocalLauncher(allow_host=True)

        policy = _session_policy_class()("agent", self._llm(str(ctx.role), ctx), launcher=launcher(), child_launcher=launcher, config=cfg,
                                         ledger_dir=attempt_dir / "ledger", stream_dir=attempt_dir, specialist_ledger=spec_ledger,
                                         session_meta={"run": self.run_id, "state": sid, "entry": ctx.entry, "attempt": ctx.attempt})
        world = self._world(ws, sb, brief, attempt_dir)
        ep = None
        try:
            try:
                await world.start()
                if sb is not None:
                    world.apps["repo"].ctx.extra["policy_exec"] = DockerExec(sb.name)  # type: ignore[arg-type]
                ep = await run_episode(world, {"agent": policy}, limits=EpisodeLimits(max_steps=self.max_steps))
            except (InfraError, StateFailed):
                raise
            except Exception as e:  # a World that cannot start or step: infra (the sandbox), never the agent's failure
                raise InfraError(f"session {sid}: {type(e).__name__}: {e}"[:500], docker_infra_kind(str(e))) from e
        finally:
            with contextlib.suppress(Exception):
                await world.close()
            with contextlib.suppress(Exception):
                await policy.aclose()
        calls = [c for c in (ep.model_calls.get("agent") or []) if isinstance(c, dict)] if ep else []
        tokens = sum(int((c.get("usage") or {}).get("prompt_tokens") or 0) + int((c.get("usage") or {}).get("completion_tokens") or 0) for c in calls)
        book_tokens(tokens)
        usd = sum(float(c.get("cost_usd") or 0.0) for c in calls)
        stop = policy.stop
        rec = SessionRecord(sid, ctx.entry, ctx.attempt, stop, policy.counters.get("turns", 0), tokens, usd, sb.name if sb else None, hcommit,
                            None, spec_id)
        self.records.append(rec)

        def write_record() -> None:
            (attempt_dir / "session.json").write_text(json.dumps({**rec.__dict__, "stop_reason": policy.stop_reason[:2000],
                                                                  "failure_kind": policy.failure_kind, "steps": ep.steps if ep else 0,
                                                                  "masked": ep.masked if ep else None}, default=str, indent=1))

        write_record()
        if spec_id:
            self.specialists.record(spec_id, {"run": self.run_id, "state": sid, "entry": ctx.entry, "attempt": ctx.attempt, "stop": stop,
                                              "turns": rec.turns, "at": time.time()})
        kinds = [k for k in [policy.failure_kind, *[f.get("kind") for f in (ep.failures if ep else []) if isinstance(f, dict)]] if k]
        why = f"{policy.error or policy.stop_reason or ''}"[:300]
        if sb is not None and not await sb.alive():
            raise InfraError(f"session {sid}: the sandbox {sb.name} died during the session ({', '.join(kinds) or 'no failure kind'}): {why}"[:500],
                             "crucible:env_container_lost")
        infra = [k for k in kinds if session_blame(k) in ("infra", "harness")]
        if infra:  # the sandbox, the kernel, the provider or the harness failed: never the task's failure
            raise InfraError(f"session {sid}: {infra[0]}: {why}"[:500], infra[0])
        if ep is not None and ep.masked and not [k for k in kinds if k in SESSION_IGNORED_KINDS]:
            raise InfraError(f"session {sid}: the episode was masked ({', '.join(kinds) or 'no failure kind'}): {why}"[:500], "crucible:backend_error")
        if stop == "agent_gave_up":
            raise StateFailed(f"the {ctx.role} gave up: {policy.stop_reason}"[:500])
        if stop != "agent_done":
            raise StateFailed(f"the {ctx.role} session ended without finishing ({stop}: {policy.stop_reason})"[:500])
        outputs, files = await self._capture(ctx, sess, ws, rel)
        if files:
            ccommit = self.repo.commit(files, f"{sid} #{ctx.entry}.{ctx.attempt} outputs", informed_by=[hcommit],
                                       trailers={"run": self.run_id, "state": sid, "role": str(ctx.role)})
            rec.capture_commit = ccommit
            for v in outputs.values():
                if isinstance(v, dict) and "sha256" in v and "path" in v:
                    v["commit"] = ccommit
            write_record()
        return Settle(outputs)

    async def _capture(self, ctx: StateContext, sess: Mapping[str, Any], ws: GitWorkspace, rel: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Each capture's port value and the files it adds to the task repo. A capture that finds nothing binds no value
        (the machine then fails this producer)."""
        outputs: dict[str, Any] = {}
        files: dict[str, Any] = {}
        for c in sess.get("captures", []):
            name, kind = c["output"], c["kind"]
            src = ws.path / c["path"] if c.get("path") else None
            dest = f"{rel}/out/{safe(name)}"
            if kind in ("json", "text"):
                if src is None or not src.is_file():
                    continue
                text = src.read_text(errors="replace")
                if kind == "json":
                    try:
                        outputs[name] = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                else:
                    outputs[name] = text
                files[f"{dest}/{src.name}"] = text
            elif kind == "file":
                if src is None or not src.is_file():
                    continue
                files[f"{dest}/{src.name}"] = src.read_bytes()
                outputs[name] = {"path": f"{dest}/{src.name}", "sha256": artifact_digest(src), "kind": "file"}
            elif kind == "dir":
                if src is None or not src.is_dir() or not any(src.rglob("*")):
                    continue
                files[f"{dest}/{src.name}"] = src
                outputs[name] = {"path": f"{dest}/{src.name}", "sha256": artifact_digest(src), "kind": "dir"}
            elif kind == "patch":
                text = await ws.patch("base")
                if not text.strip():
                    continue
                files[f"{dest}/change.patch"] = text
                outputs[name] = {"path": f"{dest}/change.patch", "sha256": _sha_text(text), "kind": "patch"}
            elif kind == "commits":
                mbox, subjects = await ws.commits("base")
                if not subjects:
                    continue
                files[f"{dest}/commits.mbox"] = mbox
                outputs[name] = {"path": f"{dest}/commits.mbox", "sha256": _sha_text(mbox), "kind": "commits", "subjects": subjects,
                                 "head": await ws.head()}
        return outputs, files


# solver-episode exposure probes (core/agent_events): a builder session reads and writes hidden tests by design (its World is
# the factory's, not a solver's), so they are recorded in the session's evidence and never fail it
SESSION_IGNORED_KINDS: frozenset[str] = frozenset({"crucible:answer_exposed"})


def session_blame(kind: str) -> str:
    """The blame of a session failure kind: ``core.schema.run.default_blame``, with the provider's outage infra."""
    from crucible.core.schema.run import default_blame
    from crucible.factory.machine import INFRA_KINDS_EXTRA

    return "infra" if kind in INFRA_KINDS_EXTRA else default_blame(kind)


def _sha_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def session_states(machine: Mapping[str, Any]) -> list[str]:
    return [s["id"] for s in machine.get("states", []) if isinstance(s, dict) and "session" in s]
