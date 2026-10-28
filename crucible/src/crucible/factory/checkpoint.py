"""Checkpoint Φ: step credit for long-horizon coding tasks (study 08 §6; docs/research/task-factory/10-checkpoint-credit.md).

During an episode the sidecar snapshots the workspace as a **patch against the start commit** and grades it out of band
with the task's hidden per-milestone tests (``MilestoneTask.milestone_status``, a separate grading container). Nothing it
measures ever reaches the agent: the harness discards what ``on_step`` returns, grades run as background tasks, and every
file is written only after the agent loop has ended.

- **When.** At each agent ``git commit`` (a shell tool call), every ``every`` tool calls (K = 10), at each compaction
  boundary of a policy's root context (the state when the new segment starts: the sidecar sees the action chosen right
  after the compaction before the World applies it), and at the end. At most ``cap`` gradings per episode (20); the end
  always has a slot. When a new snapshot would exceed the cap, the lowest-priority snapshot not yet graded is dropped
  (periodic < compaction < commit; among equals the one closest to its predecessor), so the end and the commit points
  survive. Identical patches share one grading; the start state (the empty patch) is graded once per task, outside the cap.
- **Φ.** ``Φ(s) = Σ_m w_m · pass_m · Π_{r ∈ requires(m)} pass_r`` with Σ w = 1 (C1, ``workflow.potential``: a milestone met
  before its prerequisites adds nothing, M6 R5). Weights: ``CheckpointConfig.weights``, else the task's own ``weights``
  mapping, else uniform; ``weights_from_validity`` derives them from the predictive-validity gate
  (``credit.validity.validate``'s kept items, weight max(0, 2(AUC−½))).
- **Unmeasured.** A grading that raises, times out or returns no verdict for some milestone is *unmeasured*: Φ is carried
  from the last measured checkpoint and the point is flagged; it is never read as 0. A snapshot dropped by the cap is not a
  checkpoint at all (its span merges into the next).
- **Credit.** The series [(turn, Φ, newly passed, lost)] becomes ``progress.jsonl`` rows (the ``verify.progress`` format):
  one row per World step, Φ carried between checkpoints, and the span's ΔΦ on the row of the step that **closes** the span
  (``assign="close"``; ``"spread"`` divides it evenly over the span's steps). The runner's record builder maps the k-th row
  of an actor to the call that issued its k-th action (``runtime.record.credit_from_progress``), which sets
  ``Turn.advantage``; ``train.credit.turn_credit_map`` -> ``token_credit`` then apply unchanged (centred, bounded by the
  outcome advantage, none on gate-failed or hacked episodes, RLM children take the root turn's credit, compaction calls the
  episode's). Lost milestones give negative ΔΦ at the step where they were lost.
- **One verdict on the end state** (``reconcile``). The harness grades the collected patch after the episode; when that
  patch is a checkpoint's patch, the final grade's milestone vector replaces the sidecar's own for it, and the series, the
  rows and the summary are rebuilt, so Φ, per-turn credit and the outcome reward read the same verdict by construction. A
  sidecar grading of that patch that ran under the episode's load (2026-10-25: an excelize run whose own ``go test`` was
  killed in the World graded one milestone failed that the final grade passed) can no longer leave the credit short of the
  outcome. Each replacement is recorded (``summary["reconciled"]``, ``checkpoints.jsonl`` ``source``).
- **Evidence** (next to the episode, environment facts only): ``checkpoints.jsonl`` (turn, triggers, patch sha256 and size,
  milestone vector, Φ, gained, lost, grading time), ``checkpoints/<sha>.patch`` and the summary in ``res.grade["progress"]``
  (``kind="checkpoint"``), which also carries the give-up and the STATUS.json honesty score (``factory.honesty``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from crucible.factory import honesty as _hon

PRIORITY = {"periodic": 0, "compaction": 1, "commit": 2, "end": 3}
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's empty tree: the base of a repository without commits


@runtime_checkable
class MilestoneTask(Protocol):
    """The contract a long-horizon coding task offers the sidecar (``factory.lanes.windows``' WindowTask)."""

    milestones: Sequence[Any]  # each with ``.id`` and ``.requires`` (prerequisite ids), or a mapping with those keys
    requirement_ids: Sequence[str]

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        """The hidden tests of every milestone on A + patch, in a separate grading container. Raise (or return no verdict
        for a milestone) when grading is masked: the checkpoint is then unmeasured, never 0."""
        ...


@dataclass(frozen=True)
class CheckpointConfig:
    every: int = 10  # a periodic checkpoint every K tool calls (0 = none)
    cap: int = 20  # gradings per episode (the end always has a slot; the start state is outside the cap)
    on_commit: bool = True
    on_compaction: bool = True
    assign: str = "close"  # "close": the span's ΔΦ on the step that closes it; "spread": evenly over the span's steps
    weights: Mapping[str, float] | None = None  # milestone id -> weight (normalised); None = the task's ``weights``, else uniform
    grade_start: bool = True  # grade the start state (empty patch) once per task; else Φ0 = 0
    concurrency: int = 2
    grade_timeout_s: float = 1800.0
    defer: bool = False  # grade nothing until the episode ends (fully deterministic thinning)
    keep_patches: bool = True
    honesty: bool = True  # score STATUS.json at the end when the task has requirement ids
    honesty_lambda: float = _hon.DEFAULT_LAMBDA
    snapshot: Callable[[Any], Awaitable[str]] | None = None  # world -> patch vs the start commit (default: the task's workspace)

    def __post_init__(self) -> None:
        if self.assign not in ("close", "spread"):
            raise ValueError("assign is 'close' or 'spread'")
        if self.cap < 1 or self.every < 0 or self.concurrency < 1:
            raise ValueError("cap >= 1, every >= 0 and concurrency >= 1")


# ---- Φ ---------------------------------------------------------------------------------------------------------------------


def _mid(m: Any) -> str:
    """A milestone's id: an object with ``.id`` (``core.schema.milestone.Milestone``) or a mapping (the window lane's)."""
    return str(m["id"] if isinstance(m, Mapping) else m.id)


def _requires(m: Any) -> list[str]:
    r = m.get("requires") if isinstance(m, Mapping) else getattr(m, "requires", None)
    return [str(x) for x in r or ()]


def _ids(milestones: Iterable[Any]) -> list[str]:
    return [_mid(m) for m in milestones]


def uniform_weights(milestones: Sequence[Any]) -> dict[str, float]:
    ids = _ids(milestones)
    return {i: 1.0 / len(ids) for i in ids} if ids else {}


def normalized(weights: Mapping[str, float], milestones: Sequence[Any]) -> dict[str, float]:
    """Weights over every milestone summing to 1 (missing = 0). All zero: Φ ≡ 0 (no milestone earned weight: no shaping)."""
    ids = set(_ids(milestones))
    if bad := sorted(set(weights) - ids):
        raise ValueError(f"weights for unknown milestones {bad}")
    if any(not (math.isfinite(float(w)) and float(w) >= 0) for w in weights.values()):
        raise ValueError("weights must be finite and >= 0")
    total = sum(float(w) for w in weights.values())
    return {i: (float(weights.get(i, 0.0)) / total if total > 0 else 0.0) for i in _ids(milestones)}


def weights_from_validity(validities: Iterable[Any], milestones: Sequence[Any]) -> dict[str, float]:
    """Weights from the predictive-validity gate (``credit.validity.validate`` results or their ``to_dict()``): a kept
    milestone weighs max(0, 2(AUC − ½)), a dropped one 0. Normalised; items that are not milestones are ignored."""
    ids = set(_ids(milestones))
    w: dict[str, float] = {}
    for v in validities:
        d = v.to_dict() if hasattr(v, "to_dict") else dict(v)
        if d.get("item") in ids:
            w[str(d["item"])] = float(d.get("weight") or 0.0) if d.get("kept") else 0.0
    return normalized(w, milestones)


def credited(status: Mapping[str, bool], milestones: Sequence[Any]) -> set[str]:
    """Milestones that count in ``status``: passing, and every direct prerequisite passing (C1)."""
    return {_mid(m) for m in milestones if status.get(_mid(m)) and all(status.get(r) for r in _requires(m))}


def potential(status: Mapping[str, bool], milestones: Sequence[Any], weights: Mapping[str, float]) -> float:
    return sum(weights.get(i, 0.0) for i in credited(status, milestones))


# ---- triggers ----------------------------------------------------------------------------------------------------------------

# ``git`` in command position (line start or after ; & | ( { ` with optional VAR=value prefixes), its global options, then
# the ``commit`` subcommand (not ``commit-tree``)
_COMMIT = re.compile(r"(?m)(?:^|[;&|(){}`])\s*(?:\w+=\S*\s+)*git(?:\s+(?:-[cC]\s+\S+|--?[\w.-]+(?:=\S+)?))*\s+commit(?![\w-])")
_CMD_KEYS = ("command", "cmd", "script", "code")


def is_commit(action: Any) -> bool:
    """A tool call that runs ``git commit`` in a shell (``git -c k=v commit``, ``git -C dir commit``, after ``&&`` ...)."""
    if getattr(action, "kind", None) != "tool_call":
        return False
    args = getattr(action, "arguments", None) or {}
    return any(isinstance(args.get(k), str) and _COMMIT.search(args[k]) for k in _CMD_KEYS)


def root_compactions(policy: Any) -> int:
    """Context rebuilds of a policy's root agent (``segments`` of ``LLMToolPolicy`` and ``RLMPolicy``; a resume is not one)."""
    segs = getattr(policy, "segments", None) or []
    return sum(1 for s in segs if isinstance(s, dict) and not s.get("agent") and (s.get("boundary_reason") or s.get("strategy")) != "resume")


# ---- one change, two renderings ----------------------------------------------------------------------------------------------
_DIFF_HEAD = re.compile(r"(?m)^diff --git ")


def _unquote(p: str) -> str:
    """A path as git prints it, unquoted (``core.quotepath``: ``"b/caf\\303\\251.txt"`` -> ``b/café.txt``)."""
    p = p.rstrip("\t")
    if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
        import codecs

        raw = codecs.escape_decode(p[1:-1].encode("latin-1", "backslashreplace"))[0]
        return raw.decode("utf-8", "replace")
    return p


def _block_path(block: str) -> str:
    lines = block.split("\n")
    for prefix, side in (("+++ ", "b/"), ("--- ", "a/")):
        line = next((ln for ln in lines[1:] if ln.startswith(prefix)), None)
        if line is not None and line[4:].strip() != "/dev/null":
            return _unquote(line[4:]).removeprefix(side)
    head = lines[0]
    cut = head.rfind(' "b/') if head.endswith('"') else head.rfind(" b/")
    return _unquote(head[cut + 1:]).removeprefix("b/") if cut >= 0 else head


def canonical_change(patch: str) -> list[tuple[str, tuple[str, ...]]] | None:
    """A git patch as the change it makes: per file (unquoted path), its lines without the path lines (``diff --git``,
    ``---``, ``+++``), sorted by path; files the harness's collect drops as derived (``verify.collect.is_derived``) left out.
    None for text that is not a git patch (no ``diff --git`` block and not empty): it has no canonical form."""
    from crucible.verify.collect import is_derived

    parts = _DIFF_HEAD.split(patch or "")
    if parts[0].strip():
        return None  # text before the first block (or no block at all): not a git patch
    out = []
    for block in parts[1:]:
        path = _block_path(block)
        if is_derived(path):
            continue
        body = tuple(ln for ln in block.split("\n")[1:] if not ln.startswith(("--- ", "+++ ")))
        out.append((path, body))
    return sorted(out)


def same_change(a: str | bytes, b: str | bytes) -> bool:
    """Whether two renderings of a workspace's diff make the same change (``canonical_change``)."""
    da = a.decode("utf-8", "replace") if isinstance(a, bytes) else a
    db = b.decode("utf-8", "replace") if isinstance(b, bytes) else b
    ca, cb = canonical_change(da), canonical_change(db)
    return ca is not None and cb is not None and ca == cb


# ---- snapshots ---------------------------------------------------------------------------------------------------------------


async def _sh(cmd: str, timeout: float = 600.0) -> tuple[int, str, str]:
    p = await asyncio.create_subprocess_exec("bash", "-c", cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout)
    except TimeoutError:
        p.kill()
        raise
    return p.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def start_commit(root: Path | str) -> str:
    """The commit the agent starts from (tag ``base``, else HEAD, else the empty tree), resolved before it acts: the tag
    lives in the agent's repository and can be moved later (M7 audit)."""
    from crucible.world.state.fs import git_shell

    g = git_shell(str(root))
    for ref in ("base^{commit}", "HEAD^{commit}"):
        rc, out, _ = await _sh(f"{g} rev-parse --verify -q {ref!r}", 60)
        if rc == 0 and out.strip():
            return out.strip()
    return EMPTY_TREE


def workspace_snapshot(root: Path | str, base: str) -> Callable[[Any], Awaitable[str]]:
    """The working tree (untracked included, ignored excluded) against ``base`` through a throwaway index, with git's hooks,
    fsmonitor and global config off (``world.state.fs.git_diff_command``): the agent's index and repository are untouched,
    and nothing enters the World's trace."""
    from crucible.verify.collect import filter_drivers, filter_flags, repo_hazards
    from crucible.world.state.fs import git_diff_command

    async def snap(_world: Any) -> str:
        # a redirected repository (gitdir file or symlink, commondir, core.worktree, core.bare) would make host git read another
        # repository or tree: refused as the final collect refuses it, an unmeasured checkpoint (2026-10-28, the R6 follow-up)
        if hazards := await repo_hazards(Path(root)):
            raise RuntimeError(f"snapshot refused: {hazards[0]}")
        # harness-reward audit R6 (2026-10-28): a filter driver the policy wrote into its repository's config would run on the
        # host at ``git add`` and could put any content in the snapshot: each one is a pass-through at command scope
        fl = filter_flags(await filter_drivers(Path(root)))
        cmd = git_diff_command(str(root), base, config=dict(x.split("=", 1) for x in fl[1::2]))
        rc, out, err = await _sh(cmd)
        if rc:
            raise RuntimeError(f"snapshot diff failed: {err.strip()[-300:]}")
        return out

    return snap


# ---- the series --------------------------------------------------------------------------------------------------------------


@dataclass
class Point:
    """One checkpoint of the series: the state after World step ``turn``."""

    turn: int
    phi: float
    new: list[str]
    lost: list[str]
    measured: bool
    triggers: list[str]
    index: int = 0


def series(checkpoints: Sequence[tuple[int, list[str], Mapping[str, bool] | None]], milestones: Sequence[Any], weights: Mapping[str, float],
           *, start: Mapping[str, bool] | None = None) -> tuple[float, list[Point]]:
    """(Φ0, points) from (turn, triggers, status or None = unmeasured), in turn order. An unmeasured point carries the last
    measured Φ (flagged); gains and losses compare credited sets of consecutive measured points."""
    prev = credited(start or {}, milestones)
    phi0 = potential(start or {}, milestones, weights)
    cur = phi0
    out = []
    for k, (turn, trig, st) in enumerate(sorted(checkpoints, key=lambda c: c[0])):
        if st is None:
            out.append(Point(turn, cur, [], [], False, list(trig), k))
            continue
        now = credited(st, milestones)
        cur = potential(st, milestones, weights)
        out.append(Point(turn, cur, sorted(now - prev), sorted(prev - now), True, list(trig), k))
        prev = now
    return phi0, out


def progress_rows(phi0: float, points: Sequence[Point], steps: Sequence[tuple[int, str | None, str | None]], *, assign: str = "close",
                  start_credited: Iterable[str] = ()) -> list[dict[str, Any]]:
    """``verify.progress`` rows: step 0 (the start state), one row per World step ``(step, actor, tool_call_id)`` with Φ
    carried between checkpoints, and a ``step=-1`` row for ΔΦ no step can carry (a checkpoint at turn 0). The rows'
    ΔΦ telescope to Φ_final − Φ0."""
    rows: list[dict[str, Any]] = [{"step": 0, "phi": round(phi0, 9), "delta_phi": 0.0, "gained": sorted(start_credited), "lost": [], "tool_call_id": None,
                                   "actor": None, "drift": []}]
    by_step = {s: (a, c) for s, a, c in steps}
    order = sorted(by_step)
    delta: dict[int, float] = dict.fromkeys(order, 0.0)
    marks: dict[int, dict[str, Any]] = {}
    orphan, prev_turn, prev_phi = 0.0, 0, phi0
    for p in points:
        d = p.phi - prev_phi
        span = [s for s in order if prev_turn < s <= p.turn]
        if not span:
            orphan += d
        elif assign == "spread":
            for s in span:
                delta[s] += d / len(span)
        else:
            delta[span[-1]] += d
        if span and (p.new or p.lost or p.triggers):
            m = marks.setdefault(span[-1], {"gained": [], "lost": [], "checkpoint": [], "measured": True})
            m["gained"] += p.new
            m["lost"] += p.lost
            m["checkpoint"] += p.triggers
            m["measured"] = m["measured"] and p.measured
        elif not span and (p.new or p.lost):
            marks.setdefault(-1, {"gained": [], "lost": []})
            marks[-1]["gained"] += p.new
            marks[-1]["lost"] += p.lost
        prev_turn, prev_phi = max(prev_turn, p.turn), p.phi
    phi = phi0
    for s in order:
        phi += delta[s]
        a, c = by_step[s]
        m = marks.get(s, {})
        row = {"step": s, "phi": round(min(1.0, max(0.0, phi)), 9), "delta_phi": round(delta[s], 9), "gained": sorted(m.get("gained", [])),
               "lost": sorted(m.get("lost", [])), "tool_call_id": c, "actor": a, "drift": []}
        if m.get("checkpoint"):
            row["checkpoint"], row["measured"] = sorted(set(m["checkpoint"])), m["measured"]
        rows.append(row)
    if abs(orphan) > 1e-12 or marks.get(-1):
        phi += orphan
        rows.append({"step": -1, "phi": round(min(1.0, max(0.0, phi)), 9), "delta_phi": round(orphan, 9), "gained": sorted(marks.get(-1, {}).get("gained", [])),
                     "lost": sorted(marks.get(-1, {}).get("lost", [])), "tool_call_id": None, "actor": None, "drift": []})
    return rows


# ---- the sidecar -------------------------------------------------------------------------------------------------------------


@dataclass
class _Slot:
    """One distinct patch and its grading (checkpoints with the same patch share it)."""

    sha: str
    patch: str
    order: int
    state: str = "pending"  # pending | running | done | unmeasured | skipped
    status: dict[str, bool] | None = None
    reason: str = ""
    graded_s: float | None = None
    source: str = "sidecar"  # sidecar | final_grade (``CheckpointSidecar.reconcile``)


@dataclass
class _Cp:
    index: int
    turn: int
    triggers: list[str]
    sha: str | None  # None: the snapshot itself failed (unmeasured)
    nbytes: int = 0
    reason: str = ""

    @property
    def priority(self) -> int:
        return max(PRIORITY[t] for t in self.triggers)


@dataclass
class _Run:
    steps: list[tuple[int, str | None, str | None]] = field(default_factory=list)
    tool_calls: int = 0
    compaction_turns: list[int] = field(default_factory=list)
    give_up: str | None = None
    faults: list[str] = field(default_factory=list)


class CheckpointSidecar:
    """``runtime.harness.native.EpisodeSidecar`` for a ``MilestoneTask``. ``out_dir``: the attempt directory (evidence and
    ``progress.jsonl``). ``baseline``: a per-task cache of the start state's milestone vector (``checkpoint_sidecar``)."""

    def __init__(self, task: Any, out_dir: Path | str, config: CheckpointConfig | None = None, *, baseline: dict[str, Any] | None = None):
        self.task, self.out = task, Path(out_dir)
        self.cfg = config or CheckpointConfig()
        self.milestones = list(task.milestones)
        self.ids = _ids(self.milestones)
        w = self.cfg.weights if self.cfg.weights is not None else getattr(task, "weights", None)  # the config's, else the task's own
        self.weights = normalized(w, self.milestones) if isinstance(w, Mapping) else uniform_weights(self.milestones)
        self._baseline_cache = baseline if baseline is not None else {}
        self._snap: Callable[[Any], Awaitable[str]] | None = self.cfg.snapshot
        self._root: Path | None = None
        if self._snap is None and not hasattr(task, "snapshot_patch") and not hasattr(task, "workspace"):
            raise ValueError("checkpoint credit needs a snapshot: CheckpointConfig.snapshot, task.snapshot_patch(world) or task.workspace(world)")
        self.run = _Run()
        self.cps: list[_Cp] = []
        self.slots: dict[str, _Slot] = {}
        self._queue: list[str] = []
        self._tasks: set[asyncio.Task] = set()
        self._running = 0
        self._ended = False
        self._segs: dict[str, int] = {}
        self._start: dict[str, bool] | None = None
        self._start_reason = ""
        self.summary: dict[str, Any] = {}
        self.points: list[Point] = []
        self.reconciled: list[dict[str, Any]] = []
        self._claims: Any = None  # STATUS.json as read at the end (the World is gone by the time the final grade exists)
        self._req_verified: dict[str, bool | None] | None = None  # task.requirement_status's verdicts, when the task has one

    # -- the observer protocol (run_episode) -------------------------------------------------------------------------------

    async def on_start(self, world: Any) -> None:
        if self._snap is None:
            if hasattr(self.task, "snapshot_patch"):
                self._snap = self.task.snapshot_patch
            else:
                self._root = Path(self.task.workspace(world))
                self._snap = workspace_snapshot(self._root, await start_commit(self._root))
        if self.cfg.grade_start:
            key = str(getattr(self.task, "id", id(self.task)))
            fut = self._baseline_cache.get(key)
            if fut is None or fut.get_loop() is not asyncio.get_running_loop():  # A is the same for every attempt of a task
                self._baseline_cache[key] = asyncio.ensure_future(self._status(""))
            self._start_future = self._baseline_cache[key]

    async def before_action(self, world: Any, actor: str, policy: Any, action: Any) -> None:
        """The policy has chosen its next action; the World has not applied it. A new root segment since the last action
        means a compaction boundary: the state now is the state the new segment starts from."""
        if not self.cfg.on_compaction:
            return
        n = root_compactions(policy)
        if n > self._segs.get(actor, 0):
            self._segs[actor] = n
            turn = len(self.run.steps)
            self.run.compaction_turns.append(turn)
            await self._checkpoint(world, turn, ["compaction"])

    async def on_step(self, world: Any, step: int, actor: str, action: Any, result: Any) -> None:
        try:
            self.run.steps.append((step, actor, getattr(action, "tool_call_id", None)))
            trig = []
            if getattr(action, "kind", None) == "tool_call":
                self.run.tool_calls += 1
                if self.cfg.every and self.run.tool_calls % self.cfg.every == 0:
                    trig.append("periodic")
                if self.cfg.on_commit and is_commit(action):
                    trig.append("commit")
            g = _hon.give_up_of_action(action)
            if g is not None:
                self.run.give_up = g
            if trig:
                await self._checkpoint(world, len(self.run.steps), trig)
        except Exception as e:  # noqa: BLE001 -- the sidecar never ends the agent's episode
            self.run.faults.append(f"on_step {step}: {type(e).__name__}: {e}"[:300])
        # returns nothing: nothing measured ever travels back toward the agent

    async def on_end(self, world: Any) -> dict[str, Any]:
        self._ended = True
        await self._checkpoint(world, len(self.run.steps), ["end"])
        self._pump()
        while True:  # gradings started by finishing gradings join the wait
            pending = [t for t in self._tasks if not t.done()]
            if not pending:
                break
            await asyncio.wait(pending)
        if self.cfg.grade_start:
            st, why = await self._start_future
            self._start, self._start_reason = st, why
        hon = await self._honesty(world)
        self.summary = self._finish(hon)
        return self.summary

    # -- snapshots and the grading queue -------------------------------------------------------------------------------------

    async def _checkpoint(self, world: Any, turn: int, triggers: list[str]) -> None:
        last = self.cps[-1] if self.cps else None
        if last is not None and last.turn == turn and last.sha is not None:  # no World step since: the same state
            last.triggers = sorted(set(last.triggers) | set(triggers))
            if "end" in triggers:
                self._admit(last.sha, end=True)
            return
        cp = _Cp(len(self.cps), turn, sorted(set(triggers)), None)
        self.cps.append(cp)
        try:
            patch = await self._snap(world)  # type: ignore[misc]
        except Exception as e:  # noqa: BLE001 -- a failed snapshot is an unmeasured checkpoint
            cp.reason = f"snapshot: {type(e).__name__}: {e}"[:300]
            return
        cp.sha, cp.nbytes = hashlib.sha256(patch.encode()).hexdigest(), len(patch.encode())
        if cp.sha not in self.slots:
            self.slots[cp.sha] = _Slot(cp.sha, patch, len(self.slots))
        self._admit(cp.sha, end="end" in cp.triggers)

    def _live(self) -> list[_Slot]:
        empty = hashlib.sha256(b"").hexdigest()
        return [s for s in self.slots.values() if s.state != "skipped" and not (self.cfg.grade_start and s.sha == empty)]

    def _admit(self, sha: str, *, end: bool = False) -> None:
        """Queue a slot for grading within the cap: the empty patch is the start state (graded once per task); the end is
        always admitted; else, over the cap, the lowest-priority pending slot (or this one) is dropped."""
        slot = self.slots[sha]
        if self.cfg.grade_start and sha == hashlib.sha256(b"").hexdigest():
            slot.state = "start"
            return
        if slot.state not in ("pending", "skipped") or (slot.state == "pending" and sha in self._queue):
            return
        slot.state = "pending"
        self._queue.append(sha)
        budget = self.cfg.cap if end else self.cfg.cap - 1
        while len(self._live()) > budget:
            cand = [self.slots[s] for s in self._queue if s != sha or not end]
            if not cand:
                break
            victim = min(cand, key=lambda s: (self._slot_priority(s), self._gap(s), s.order))
            if victim.sha == sha and end:
                break
            victim.state = "skipped"
            self._queue.remove(victim.sha)
        self._pump()

    def _slot_priority(self, s: _Slot) -> int:
        return max((cp.priority for cp in self.cps if cp.sha == s.sha), default=0)

    def _gap(self, s: _Slot) -> int:
        """Turns from the slot's latest checkpoint back to the previous checkpoint still kept: thinning drops the densest."""
        mine = [cp.turn for cp in self.cps if cp.sha == s.sha]
        kept = sorted(cp.turn for cp in self.cps if cp.sha and cp.sha != s.sha and self.slots[cp.sha].state != "skipped")
        t = max(mine, default=0)
        before = [k for k in kept if k <= t]
        return t - (before[-1] if before else 0)

    def _pump(self) -> None:
        if self.cfg.defer and not self._ended:
            return
        while self._running < self.cfg.concurrency and self._queue:
            slot = self.slots[self._queue.pop(0)]
            slot.state = "running"
            self._running += 1
            self._tasks.add(asyncio.ensure_future(self._grade(slot)))

    async def _grade(self, slot: _Slot) -> None:
        t0 = time.monotonic()
        try:
            slot.status, slot.reason = await self._status(slot.patch)
            slot.state = "done" if slot.status is not None else "unmeasured"
        finally:
            slot.graded_s = round(time.monotonic() - t0, 3)
            self._running -= 1
            self._pump()

    async def _status(self, patch: str) -> tuple[dict[str, bool] | None, str]:
        """The milestone vector for ``patch``, or (None, why) when grading did not measure every milestone."""
        try:
            st = await asyncio.wait_for(self.task.milestone_status(patch), self.cfg.grade_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- masked grading: unmeasured, never 0
            return None, f"{type(e).__name__}: {e}"[:300]
        if not isinstance(st, Mapping):
            return None, f"milestone_status returned {type(st).__name__}"
        missing = [i for i in self.ids if not isinstance(st.get(i), bool)]
        if missing:
            return None, f"no verdict for {missing[:5]}"
        return {i: bool(st[i]) for i in self.ids}, ""

    # -- the end ---------------------------------------------------------------------------------------------------------------

    async def _honesty(self, world: Any) -> _hon.HonestyScore | None:
        reqs = list(getattr(self.task, "requirement_ids", None) or [])
        end = self.cps[-1] if self.cps else None
        if not (self.cfg.honesty and reqs and end is not None and end.sha is not None):
            return None
        patch = self.slots[end.sha].patch
        self._claims = _hon.status_from_tree(self._root) if self._root is not None else _hon.status_from_patch(patch)
        if hasattr(self.task, "requirement_status"):
            self._req_verified = {}
            try:
                got = await asyncio.wait_for(self.task.requirement_status(patch), self.cfg.grade_timeout_s)
                self._req_verified = {r: (bool(got[r]) if isinstance(got.get(r), bool) else None) for r in reqs}
            except Exception as e:  # noqa: BLE001 -- unverified: nothing is scored
                self.run.faults.append(f"requirement_status: {type(e).__name__}: {e}"[:300])
        return self._honesty_score()

    def _honesty_score(self) -> _hon.HonestyScore | None:
        """The honesty score from the claims read at the end, verified by ``requirement_status`` or, for requirements graded
        through milestones (``requirement_milestones`` {req: [milestone ids]}, else same-named ones), by the end slot's vector
        as it stands (after ``reconcile``, the final grade's)."""
        reqs = list(getattr(self.task, "requirement_ids", None) or [])
        end = self.cps[-1] if self.cps else None
        if not (self.cfg.honesty and reqs and end is not None and end.sha is not None):
            return None
        if self._req_verified is not None:
            verified = dict(self._req_verified)
        else:
            st = self.slots[end.sha].status if self.slots[end.sha].state != "start" else self._start
            by = getattr(self.task, "requirement_milestones", None) or {r: [r] for r in reqs}
            verified = {r: (all(st.get(m) for m in by[r]) if st is not None and by.get(r) and all(m in st for m in by[r]) else None) for r in reqs}
        return _hon.score(self._claims, verified, requirement_ids=reqs, lam=self.cfg.honesty_lambda, give_up=self.run.give_up)

    def reconcile(self, patch: str | bytes | None, final: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """After the harness graded the collected ``patch``: its milestone vector ``final`` (``GradeReport.meta['milestones']``
        of an unmasked grade) becomes the vector of every checkpoint with that patch, and the series, ``progress.jsonl``,
        ``checkpoints.jsonl`` and the summary are rebuilt (module doc, "One verdict on the end state"). ``patch=None``: the
        harness graded the World's end state itself (``NativeHarness``), which is the end checkpoint's. Returns the new
        summary, or None when nothing changes: the episode has not ended, ``final`` lacks a verdict for some milestone (a
        masked final grade is never a vector of zeros), no checkpoint holds this patch, or the vectors already agree."""
        if not self._ended or not isinstance(final, Mapping) or not all(isinstance(final.get(i), bool) for i in self.ids):
            return None
        end_sha = self.cps[-1].sha if self.cps and "end" in self.cps[-1].triggers else None
        if patch is None:
            sha = end_sha
        else:
            sha = hashlib.sha256(patch if isinstance(patch, bytes) else patch.encode()).hexdigest()
            if sha not in self.slots and end_sha in self.slots and same_change(self.slots[end_sha].patch, patch):
                # harness-reward audit 2026-10-28: the end snapshot and the harness's collect build the same change with other
                # flags (the collect drops derived files such as an unignored __pycache__, quotes non-ASCII paths), so a byte
                # match missed the end checkpoint and its sidecar verdict stood against the final grade
                sha = end_sha
        slot = self.slots.get(sha) if sha else None
        if slot is None:
            return None
        new = {i: bool(final[i]) for i in self.ids}
        if slot.state == "start":
            if self._start == new:
                return None
            self.reconciled.append({"patch_sha256": sha, "state": "start", "from": self._start, "to": new, "reason": self._start_reason})
            self._start, self._start_reason = new, ""
        else:
            if slot.state == "done" and slot.status == new:
                return None
            self.reconciled.append({"patch_sha256": sha, "state": slot.state, "from": slot.status, "to": new, "reason": slot.reason})
            slot.status, slot.state, slot.reason = new, "done", ""
        slot.source = "final_grade"
        self.summary = self._finish(self._honesty_score())
        return self.summary

    def _finish(self, hon: _hon.HonestyScore | None) -> dict[str, Any]:
        rows_in = []
        for cp in self.cps:
            slot = self.slots.get(cp.sha) if cp.sha else None
            if slot is not None and slot.state == "skipped":
                continue
            if slot is not None and slot.state == "start":
                st = self._start
            else:
                st = slot.status if slot is not None else None
            rows_in.append((cp, st))
        phi0, pts = series([(cp.turn, cp.triggers, st) for cp, st in rows_in], self.milestones, self.weights, start=self._start)
        for p, (cp, _) in zip(pts, sorted(rows_in, key=lambda x: x[0].turn), strict=True):
            p.index = cp.index
        self.points = pts
        rows = progress_rows(phi0, pts, self.run.steps, assign=self.cfg.assign, start_credited=credited(self._start or {}, self.milestones))
        self._write(rows, pts, rows_in)
        final = pts[-1] if pts else None
        graded = [s for s in self.slots.values() if s.state in ("done", "unmeasured")]
        out: dict[str, Any] = {
            "kind": "checkpoint", "phi0": round(phi0, 9), "phi_final": round(final.phi if final else phi0, 9),
            "sum_delta": round(sum(r["delta_phi"] for r in rows), 9), "rows": len(rows), "checkpoints": len(rows_in),
            "graded": len(graded), "unmeasured": sum(1 for p in pts if not p.measured), "skipped_cap": sum(1 for cp in self.cps if cp.sha and self.slots[cp.sha].state == "skipped"),
            "final_measured": bool(final and final.measured), "milestones_lost": sum(len(p.lost) for p in pts),
            "compaction_turns": list(self.run.compaction_turns), "steps": len(self.run.steps), "tool_calls": self.run.tool_calls,
            "series": [[p.turn, round(p.phi, 9), p.new, p.lost, p.measured, p.triggers] for p in pts],
            "satisfied": len(credited(self.slots[fs].status or {}, self.milestones)) if (fs := self._final_sha()) else 0,
            "subgoals": len(self.ids), "give_up": self.run.give_up, "start_measured": self._start is not None or not self.cfg.grade_start,
        }
        if self._start_reason:
            out["start_reason"] = self._start_reason
        if hon is not None:
            out["honesty"] = hon.to_dict()
        if self.run.faults:
            out["faults"] = self.run.faults[:20]
        if self.reconciled:
            out["reconciled"] = list(self.reconciled)
        return out

    def _final_sha(self) -> str | None:
        for cp in reversed(self.cps):
            if cp.sha and self.slots[cp.sha].state == "done":
                return cp.sha
        return None

    def _write(self, rows: list[dict[str, Any]], pts: list[Point], rows_in: list[tuple[_Cp, Any]]) -> None:
        """Evidence, written once the agent loop has ended (a local-shell agent can never read it mid-episode)."""
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "progress.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        by_index = {p.index: p for p in pts}
        lines = []
        for cp in self.cps:
            slot = self.slots.get(cp.sha) if cp.sha else None
            p = by_index.get(cp.index)
            state = "start" if slot is not None and slot.state == "start" else (slot.state if slot is not None else "unmeasured")
            lines.append({"index": cp.index, "turn": cp.turn, "triggers": cp.triggers, "patch_sha256": cp.sha, "patch_bytes": cp.nbytes, "state": state,
                          "milestones": (self._start if state == "start" else (slot.status if slot else None)), "reason": cp.reason or (slot.reason if slot else ""),
                          "phi": None if p is None else round(p.phi, 9), "measured": bool(p and p.measured), "gained": p.new if p else [],
                          "lost": p.lost if p else [], "graded_s": slot.graded_s if slot else None, "source": slot.source if slot else None})
        (self.out / "checkpoints.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in lines))
        if self.cfg.keep_patches:
            d = self.out / "checkpoints"
            d.mkdir(exist_ok=True)
            for s in self.slots.values():
                if s.state in ("done", "unmeasured", "start"):
                    (d / f"{s.sha}.patch").write_text(s.patch)

    @property
    def phi_series(self) -> list[tuple[int, float, list[str], list[str]]]:
        """[(turn index, Φ, newly passed, lost)] after ``on_end``."""
        return [(p.turn, p.phi, p.new, p.lost) for p in self.points]


def checkpoint_sidecar(config: CheckpointConfig | None = None) -> Callable[[Any], CheckpointSidecar | None]:
    """``WorkspaceHarness(sidecar=checkpoint_sidecar())`` / ``RLMHarness(sidecar=...)``: a sidecar for every attempt whose task
    is a ``MilestoneTask`` (others run unchanged). The start state's grading is cached per task across attempts."""
    cache: dict[str, Any] = {}

    def make(attempt: Any) -> CheckpointSidecar | None:
        t = attempt.task
        if not isinstance(t, MilestoneTask):
            return None
        return CheckpointSidecar(t, attempt.dir, config, baseline=cache)

    return make
