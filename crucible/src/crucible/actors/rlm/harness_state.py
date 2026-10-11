"""The continual harness (m14), Crucible's version of Prime Agent's harness state + ``/refine`` (PA §3; CH), fixing what
the study found missing there: **edits are measured before they are kept**.

- **Typed entries** (Prime's five kinds): ``note`` (supplemental prompt; the base prompt is immutable), ``memory`` (facts,
  decisions, failures), ``skill`` (Python code importable into the RLM kernel), ``subagent`` (a role spec), ``fsm`` (a
  workflow spec). Each entry is scoped ``(family, task | None)`` and carries usage counters (``shown``, ``used``,
  ``helped``) so entries that never help can be retired (CH: 99 skills created, 5 ever succeeded).
- **Versions:** the store is an append-only list of whole-state versions with content hashes; ``rollback_to(v)`` restores a
  recorded state exactly (Prime's per-edit inverse can overwrite later edits).
- **Measured edits:** ``propose(edits)`` validates (kind, required fields, no base-prompt edits, skills must compile) and
  returns a candidate state; ``evaluate(candidate, fn, seeds)`` runs ``fn(state, seed) -> {metric: value}`` for the current
  and the candidate state on the **same seeds** and accepts only if the primary metric improves by ``margin`` and no guard
  metric regresses by more than ``tolerance``; the decision and every metric are recorded either way (Prime stores
  ``expectedOutcome`` and never checks it).
- **Digest:** ``digest(query, per_kind)`` ranks entries by term overlap with the query (goal + recent text), Prime's ranked
  ``[harness-digest]``, and counts what was shown.
- Persistence: one JSON file, atomic write; loading a corrupt file raises (no silent empty store for a manager that must
  not forget it forgot).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("note", "memory", "skill", "subagent", "fsm")


@dataclass
class Entry:
    id: str
    kind: str
    title: str
    content: str
    family: str = "*"
    task: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    shown: int = 0
    used: int = 0
    helped: int = 0
    created_at: float = 0.0


def _hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]{3,}", text.lower())]


class HarnessError(ValueError):
    pass


class HarnessState:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.entries: dict[str, Entry] = {}
        self.versions: list[dict[str, Any]] = []  # [{"version", "hash", "entries", "event"}]
        self.events: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text())
            self.entries = {k: Entry(**v) for k, v in d["entries"].items()}
            self.versions, self.events = d["versions"], d["events"]
        if not self.versions:
            self._commit({"kind": "init"})

    # ---- persistence and versions ---------------------------------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any]:
        return {k: asdict(v) for k, v in sorted(self.entries.items())}

    def _content(self) -> dict[str, Any]:
        return {k: {f: v[f] for f in ("kind", "title", "content", "family", "task", "metadata", "version")} for k, v in self._snapshot().items()}

    @property
    def version(self) -> int:
        return self.versions[-1]["version"]

    @property
    def fingerprint(self) -> str:
        return _hash(self._content())

    def _commit(self, event: dict[str, Any]) -> None:
        v = (self.versions[-1]["version"] + 1) if self.versions else 0
        self.versions.append({"version": v, "hash": self.fingerprint, "entries": self._snapshot(), "event": event})
        self._save()

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": self._snapshot(), "versions": self.versions, "events": self.events}, default=str))
        tmp.replace(self.path)

    def rollback_to(self, version: int) -> None:
        rec = next((v for v in self.versions if v["version"] == version), None)
        if rec is None:
            raise HarnessError(f"no harness version {version}")
        self.entries = {k: Entry(**e) for k, e in rec["entries"].items()}
        self.events.append({"kind": "rollback", "to": version, "at": time.time()})
        self._commit({"kind": "rollback", "to": version})

    # ---- edits ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _validate(edit: dict[str, Any], entries: dict[str, Entry]) -> list[str]:
        why = []
        act, kind, eid = edit.get("action"), edit.get("kind"), edit.get("id")
        if act not in ("create", "update", "delete"):
            why.append(f"unknown action {act!r}")
        if kind not in KINDS:
            why.append(f"unknown kind {kind!r} (one of {KINDS})")
        if not eid:
            why.append("an edit needs an id")
        if eid == "base_system_prompt" or (kind == "note" and str(eid).startswith("base")):
            why.append("the base system prompt is immutable; notes only supplement it")
        if act == "create" and eid in entries:
            why.append(f"{eid} already exists")
        if act in ("update", "delete") and eid not in entries:
            why.append(f"{eid} does not exist")
        if act in ("create", "update") and not (edit.get("title") or (eid in entries)) :
            why.append("create needs a title")
        if act in ("create", "update") and kind == "skill" and edit.get("content"):
            try:
                compile(edit["content"], f"<skill {eid}>", "exec", flags=0x2000)  # PyCF_ALLOW_TOP_LEVEL_AWAIT
            except SyntaxError as e:
                why.append(f"skill {eid} does not compile: {e}")
        return why

    def propose(self, edits: Sequence[dict[str, Any]]) -> dict[str, Entry]:
        """The candidate entries after ``edits`` (validated; nothing is committed)."""
        cand = copy.deepcopy(self.entries)
        problems = []
        for e in edits:
            why = self._validate(e, cand)
            if why:
                problems.append(f"{e.get('id')}: {'; '.join(why)}")
                continue
            eid = e["id"]
            if e["action"] == "delete":
                del cand[eid]
            elif e["action"] == "create":
                cand[eid] = Entry(id=eid, kind=e["kind"], title=e["title"], content=e.get("content", ""), family=e.get("family", "*"),
                                  task=e.get("task"), metadata=dict(e.get("metadata") or {}), created_at=time.time())
            else:
                old = cand[eid]
                cand[eid] = Entry(**{**asdict(old), **{k: e[k] for k in ("title", "content", "family", "task", "metadata") if k in e},
                                     "version": old.version + 1})
        if problems:
            raise HarnessError("; ".join(problems))
        return cand

    async def evaluate(self, edits: Sequence[dict[str, Any]], fn: Callable[[dict[str, Entry], int], Awaitable[dict[str, float]]],
                       seeds: Sequence[int], *, primary: str, margin: float = 0.0, guards: Sequence[str] = (), tolerance: float = 0.0,
                       higher_is_better: dict[str, bool] | None = None, trigger: str = "", evidence: str = "") -> dict[str, Any]:
        """A/B on the same seeds: accept iff the candidate improves ``primary`` by > ``margin`` and no guard regresses by more
        than ``tolerance``. Records the event and commits the candidate only when accepted."""
        cand = self.propose(edits)
        hib = {primary: True, **(higher_is_better or {})}
        base_runs = [await fn(self.entries, s) for s in seeds]
        cand_runs = [await fn(cand, s) for s in seeds]

        def mean(rs, k):
            vals = [r[k] for r in rs if k in r and r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
            return sum(vals) / len(vals) if vals else float("nan")

        metrics = {k: {"base": mean(base_runs, k), "candidate": mean(cand_runs, k)} for k in {primary, *guards}}

        def gain(k):
            m = metrics[k]
            d = m["candidate"] - m["base"]
            return d if hib.get(k, True) else -d

        reasons = []
        if not gain(primary) > margin:
            reasons.append(f"{primary} did not improve by more than {margin} ({metrics[primary]['base']:.4g} -> {metrics[primary]['candidate']:.4g})")
        for g in guards:
            if gain(g) < -tolerance:
                reasons.append(f"guard {g} regressed ({metrics[g]['base']:.4g} -> {metrics[g]['candidate']:.4g})")
        accepted = not reasons
        event = {"kind": "refinement", "trigger": trigger, "evidence": evidence[:2000], "edits": list(edits), "seeds": list(seeds),
                 "metrics": metrics, "accepted": accepted, "reasons": reasons, "base_version": self.version, "at": time.time()}
        self.events.append(event)
        if accepted:
            self.entries = cand
            self._commit({"kind": "refinement", "accepted_event": len(self.events) - 1})
        else:
            self._save()
        return event

    def apply_unmeasured(self, edits: Sequence[dict[str, Any]], *, reason: str) -> None:
        """For edits that cannot be measured (a fact the user stated): recorded as unmeasured, so an audit can find them."""
        self.entries = self.propose(edits)
        self.events.append({"kind": "unmeasured", "edits": list(edits), "reason": reason, "at": time.time()})
        self._commit({"kind": "unmeasured"})

    # ---- digest and counters ----------------------------------------------------------------------------------------
    def select(self, family: str, task: str | None = None, kinds: Sequence[str] = KINDS) -> list[Entry]:
        return [e for e in self.entries.values() if e.kind in kinds and e.family in ("*", family) and (e.task is None or e.task == task)]

    def digest(self, query: str, *, family: str = "*", task: str | None = None, per_kind: int = 3, chars: int = 300) -> str:
        q = set(_terms(query))
        lines = ["[harness-digest] supplemental notes, memories, skills and roles (the base prompt is immutable):"]
        for kind in KINDS:
            pool = self.select(family, task, (kind,))
            pool.sort(key=lambda e: (-len(q & set(_terms(e.title + " " + e.content))), -e.helped, e.id))
            for e in pool[:per_kind]:
                e.shown += 1
                lines.append(f"- {kind}:{e.id} {e.title}: {e.content[:chars]}")
        return "\n".join(lines)

    def mark(self, entry_id: str, *, used: bool = False, helped: bool = False) -> None:
        e = self.entries.get(entry_id)
        if e is not None:
            e.used += int(used)
            e.helped += int(helped)
            self._save()

    def retire_candidates(self, *, min_shown: int = 10) -> list[str]:
        """Entries shown at least ``min_shown`` times that never helped (create-and-forget)."""
        return sorted(e.id for e in self.entries.values() if e.shown >= min_shown and e.helped == 0)
