"""Verifier and task versions with lineage, and versioned training objectives (swarm-pipeline/00 §8.3, §11.3 item 6, P3).

**Versions.** Every change the trace-audit loop accepts (``factory.traceaudit``: a new or relaxed check, a prompt edit, a
World fix) makes a new version of the task's verifier or of the task itself. A version is content-addressed and carries
its lineage:
- ``parent``: the version it changes (``None`` only for a root: the version a task was published with);
- ``proposal_id``: the accepted proposal that made it;
- ``evidence_hash``: the sha256 of the executed evidence the acceptance recorded (the demonstrating end state's verdicts,
  the oracle, alternative and mutant runs, the flake bound, the lint and ledger results, the World reproduction).

``VersionStore`` is append-only JSONL. Each line carries the hash of the line before it, so an edited or dropped line
breaks ``verify()``. A derived version needs a proposal id and non-empty evidence, and its parent must be the subject's
current head: two changes never fork one verifier silently.

**Objectives** (HUD's checkpoint-tree rollback rule). A training run's reward is defined by the verifier version of every
task it trains on. ``objective_id`` hashes that mapping. ``TrainingObjective`` pins it at the start of a run:
- ``record`` refuses a reward graded by any other verifier version (``ObjectiveMixError``);
- ``after_change`` is the only way forward when a verifier changes mid-run. It returns a new objective, which either
  resumes from a checkpoint taken **before** the change (a new run id with a ``@<step>`` suffix) or starts a new run;
- ``curve`` and ``merge_rewards`` refuse rows from two objectives. A curve never mixes rewards of two verifier versions.

**Offline re-grades.** ``regrade_offline`` grades past episodes under a new version for reporting only. Its rows are
``OfflineGrade`` objects, which ``TrainingObjective.record`` refuses, so a re-grade can never re-enter training.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Kind = Literal["verifier", "task"]
GENESIS = "0" * 64


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str, separators=(",", ":"))


def sha(x: Any) -> str:
    return hashlib.sha256((x if isinstance(x, str) else canonical(x)).encode()).hexdigest()


class VersionError(ValueError):
    pass


class ObjectiveMixError(ValueError):
    """Rewards from two verifier versions (two objectives) were about to be mixed."""


@dataclass(frozen=True)
class Version:
    id: str
    subject: str  # the task id
    kind: Kind
    parent: str | None
    content: dict[str, Any]  # the whole versioned content (checks, prompt, World fixes), not a diff
    content_hash: str
    proposal_id: str | None
    evidence_hash: str | None
    change: dict[str, Any]  # what changed against the parent, typed: add_check | relax_check | prompt_edit | world_fix | root
    created_at: float
    note: str = ""

    @staticmethod
    def make_id(subject: str, kind: str, parent: str | None, content_hash: str, proposal_id: str | None, evidence_hash: str | None) -> str:
        return sha({"subject": subject, "kind": kind, "parent": parent, "content": content_hash, "proposal": proposal_id,
                    "evidence": evidence_hash})[:16]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class VersionStore:
    """Append-only, hash-chained version records (see the module doc). ``path`` is the JSONL file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = []
        if self.path.exists():
            self._rows = [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    # -- reading --------------------------------------------------------------------------------------------------------
    def all(self) -> list[Version]:
        return [Version(**{k: v for k, v in r.items() if k not in ("prev", "line_hash")}) for r in self._rows]

    def get(self, vid: str) -> Version:
        for v in self.all():
            if v.id == vid:
                return v
        raise VersionError(f"no version {vid}")

    def head(self, subject: str, kind: Kind) -> Version | None:
        mine = [v for v in self.all() if v.subject == subject and v.kind == kind]
        return mine[-1] if mine else None

    def lineage(self, vid: str) -> list[Version]:
        """Root first, ``vid`` last."""
        out = []
        cur: str | None = vid
        while cur is not None:
            v = self.get(cur)
            out.append(v)
            cur = v.parent
        return out[::-1]

    def verify(self) -> list[str]:
        """Every problem in the store: a broken hash chain, an id that does not recompute, a missing parent, a derived
        version without a proposal or evidence."""
        out = []
        prev = GENESIS
        ids: set[str] = set()
        for i, r in enumerate(self._rows):
            body = {k: v for k, v in r.items() if k not in ("prev", "line_hash")}
            if r.get("prev") != prev:
                out.append(f"line {i}: the chain is broken (prev {str(r.get('prev'))[:12]} != {prev[:12]})")
            if r.get("line_hash") != sha({**body, "prev": r.get("prev")}):
                out.append(f"line {i}: the line was edited (its hash does not recompute)")
            if r["content_hash"] != sha(r["content"]):
                out.append(f"line {i}: the content hash does not recompute")
            if r["id"] != Version.make_id(r["subject"], r["kind"], r["parent"], r["content_hash"], r["proposal_id"], r["evidence_hash"]):
                out.append(f"line {i}: the version id does not recompute")
            if r["parent"] is not None and r["parent"] not in ids:
                out.append(f"line {i}: parent {r['parent']} is not an earlier version")
            if r["parent"] is not None and not (r["proposal_id"] and r["evidence_hash"]):
                out.append(f"line {i}: a derived version without a proposal id or evidence")
            ids.add(r["id"])
            prev = r.get("line_hash") or prev
        return out

    # -- writing --------------------------------------------------------------------------------------------------------
    def _append(self, v: Version) -> Version:
        body = v.to_json()
        prev = self._rows[-1]["line_hash"] if self._rows else GENESIS
        row = {**body, "prev": prev}
        row["line_hash"] = sha({**body, "prev": prev})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(canonical(row) + "\n")
        self._rows.append(row)
        return v

    def root(self, subject: str, kind: Kind, content: Mapping[str, Any], *, note: str = "published") -> Version:
        """The subject's root version (idempotent: the same content returns the existing root)."""
        ch = sha(dict(content))
        for v in self.all():
            if v.subject == subject and v.kind == kind and v.parent is None and v.content_hash == ch:
                return v
        if self.head(subject, kind) is not None:
            raise VersionError(f"{subject}/{kind} already has versions; derive from its head instead")
        vid = Version.make_id(subject, kind, None, ch, None, None)
        return self._append(Version(vid, subject, kind, None, dict(content), ch, None, None, {"type": "root"}, time.time(), note))

    def derive(self, parent_id: str, *, content: Mapping[str, Any], change: Mapping[str, Any], proposal_id: str, evidence: Mapping[str, Any],
               note: str = "") -> Version:
        """A new version from an accepted proposal. ``parent_id`` must be the subject's head; ``evidence`` is the executed
        evidence of the acceptance (hashed into the version)."""
        parent = self.get(parent_id)
        head = self.head(parent.subject, parent.kind)
        if head is None or head.id != parent_id:
            raise VersionError(f"{parent_id} is not the head of {parent.subject}/{parent.kind} (head {head.id if head else None}): no forks")
        if not proposal_id:
            raise VersionError("a derived version needs the accepted proposal's id")
        if not evidence:
            raise VersionError("a derived version needs the acceptance's executed evidence")
        ch = sha(dict(content))
        if ch == parent.content_hash:
            raise VersionError("the change leaves the content unchanged")
        eh = sha(dict(evidence))
        vid = Version.make_id(parent.subject, parent.kind, parent_id, ch, proposal_id, eh)
        return self._append(Version(vid, parent.subject, parent.kind, parent_id, dict(content), ch, proposal_id, eh, dict(change), time.time(), note))


# ---- objectives ------------------------------------------------------------------------------------------------------------
def objective_id(verifier_versions: Mapping[str, str]) -> str:
    """The training objective defined by task -> verifier version id."""
    if not verifier_versions:
        raise ObjectiveMixError("an objective needs at least one task's verifier version")
    return "obj-" + sha(dict(sorted(verifier_versions.items())))[:16]


@dataclass(frozen=True)
class RewardRow:
    run_id: str
    objective_id: str
    step: int
    episode: str
    task: str
    verifier_version: str
    reward: float


@dataclass(frozen=True)
class OfflineGrade:
    """A past episode re-graded under another verifier version: reporting only, never a training reward."""

    episode: str
    task: str
    verifier_version: str
    reward: float | None
    original_reward: float | None
    original_version: str | None
    for_reporting_only: bool = True


@dataclass
class TrainingObjective:
    """A training run's objective (see the module doc)."""

    run_id: str
    versions: dict[str, str]
    parent: str | None = None  # the objective this one follows after a verifier change
    resumed_from_step: int | None = None
    rows: list[RewardRow] = field(default_factory=list)

    @property
    def objective_id(self) -> str:
        return objective_id(self.versions)

    def record(self, *, episode: str, task: str, verifier_version: str, reward: float, step: int) -> RewardRow:
        if not isinstance(task, str) or not isinstance(verifier_version, str):
            raise ObjectiveMixError("a reward row names its task and verifier version")
        if task not in self.versions:
            raise ObjectiveMixError(f"task {task} is not part of objective {self.objective_id}")
        if self.versions[task] != verifier_version:
            raise ObjectiveMixError(f"reward for {task} graded by verifier {verifier_version}; objective {self.objective_id} pins "
                                    f"{self.versions[task]}: a changed verifier is a new objective (after_change)")
        if self.resumed_from_step is not None and step <= self.resumed_from_step:
            raise ObjectiveMixError(f"step {step} is not after the checkpoint this objective resumed from ({self.resumed_from_step})")
        row = RewardRow(self.run_id, self.objective_id, int(step), episode, task, verifier_version, float(reward))
        self.rows.append(row)
        return row

    def accept(self, row: Any) -> RewardRow:
        """A row from elsewhere (an offline re-grade is refused)."""
        if isinstance(row, OfflineGrade) or getattr(row, "for_reporting_only", False):
            raise ObjectiveMixError("an offline re-grade is for reporting only and never a training reward")
        return self.record(episode=row.episode, task=row.task, verifier_version=row.verifier_version, reward=row.reward, step=row.step)

    def after_change(self, new_versions: Mapping[str, str], *, change_step: int, resume_from_step: int | None = None) -> TrainingObjective:
        """The objective after a verifier change at ``change_step``. ``resume_from_step``: a checkpoint taken before the
        change (training continues from it, as run ``<run>@<step>``); None: a new run (``<run>+<objective>``)."""
        merged = {**self.versions, **dict(new_versions)}
        nxt = objective_id(merged)
        if nxt == self.objective_id:
            raise ObjectiveMixError("the new versions leave the objective unchanged")
        if resume_from_step is not None:
            if resume_from_step >= change_step:
                raise ObjectiveMixError(f"resume from step {resume_from_step} is not before the change at step {change_step}")
            return TrainingObjective(f"{self.run_id}@{resume_from_step}", merged, self.objective_id, resume_from_step)
        return TrainingObjective(f"{self.run_id}+{nxt[4:12]}", merged, self.objective_id, None)


def curve(rows: Sequence[RewardRow]) -> list[tuple[int, float]]:
    """(step, mean reward) for one objective's rows; rows from two objectives raise."""
    if any(isinstance(r, OfflineGrade) or getattr(r, "for_reporting_only", False) for r in rows):
        raise ObjectiveMixError("offline re-grades never enter a training curve")
    objs = {r.objective_id for r in rows}
    if len(objs) > 1:
        raise ObjectiveMixError(f"rows from {len(objs)} objectives ({sorted(objs)}): rewards of two verifier versions are never mixed")
    by: dict[int, list[float]] = {}
    for r in rows:
        by.setdefault(r.step, []).append(r.reward)
    return [(s, sum(v) / len(v)) for s, v in sorted(by.items())]


def merge_rewards(*groups: Iterable[RewardRow]) -> list[RewardRow]:
    rows = [r for g in groups for r in g]
    curve(rows)  # raises on a mix
    return rows


async def regrade_offline(episodes: Sequence[Mapping[str, Any]], grade: Callable[[Any], Awaitable[float | None]], *, version: str) -> list[OfflineGrade]:
    """Re-grade stored episodes (``{"episode", "task", "solution", "reward"?, "verifier_version"?}``) under ``version``
    for reporting only."""
    out = []
    for e in episodes:
        r = await grade(e["solution"])
        out.append(OfflineGrade(str(e["episode"]), str(e["task"]), version, None if r is None else float(r), e.get("reward"), e.get("verifier_version")))
    return out
