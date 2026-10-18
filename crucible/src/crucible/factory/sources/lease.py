"""The eval lease (swarm-pipeline/00 §2, §10; V9): taken before any filter or model call on a source. A training scout
or generator never reads a sealed source.

Sealed are: the held-out repositories (``windows.HELD_OUT_REPOS``, by local name and by ``owner/name`` through
``REPO_URLS``), every organisation that owns a held-out repository and no training repository (an org that owns both,
such as ``pallets``, is sealed repository by repository), any extra repositories, orgs and the time cut of a sealed
benchmark's manifest (``sealed.json``: ``{"repos": [...], "orgs": [...], "after": "<ISO date>"}``), and, with ``after``
set, any source dated after it (the held-out benchmark's time window lies after every training source).

``EvalLease.check(record)`` returns the decision ``{"ok", "reason", "rule"}``; ``take(record)`` writes it on the record
and returns whether it was granted. ``guard_repo(name)`` raises ``SealedSourceError`` (the generators call it before
cloning or building anything).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.lanes.windows import HELD_OUT_REPOS, REPO_URLS


class SealedSourceError(PermissionError):
    """A sealed (held-out) source offered to a training generator."""


def _norm(name: str) -> str:
    n = name.strip().lower().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    return n.replace("__", "/")  # a scouted clone is named owner__name


@dataclass
class EvalLease:
    repos: set[str] = field(default_factory=set)  # local names and owner/name, lower case
    orgs: set[str] = field(default_factory=set)
    after: str | None = None  # ISO date: a source dated after it belongs to the sealed time window

    @classmethod
    def default(cls, manifest: Path | None = None, *, extra_repos: Iterable[str] = (), extra_orgs: Iterable[str] = ()) -> EvalLease:
        repos = {_norm(n) for n in HELD_OUT_REPOS} | {_norm(REPO_URLS[n]) for n in HELD_OUT_REPOS if n in REPO_URLS}
        train_orgs = {_norm(REPO_URLS[n]).split("/")[0] for n in REPO_URLS if n not in HELD_OUT_REPOS}
        orgs = {_norm(REPO_URLS[n]).split("/")[0] for n in HELD_OUT_REPOS if n in REPO_URLS} - train_orgs
        after = None
        if manifest is not None and Path(manifest).exists():
            m = json.loads(Path(manifest).read_text())
            repos |= {_norm(r) for r in m.get("repos") or []}
            orgs |= {_norm(o) for o in m.get("orgs") or []}
            after = m.get("after")
        return cls(repos | {_norm(r) for r in extra_repos}, orgs | {_norm(o) for o in extra_orgs}, after)

    def check(self, rec: Any) -> dict[str, Any]:
        repo = _norm(str(rec.get("repo") if isinstance(rec, Mapping) else getattr(rec, "repo", "")))
        created = str((rec.get("created") if isinstance(rec, Mapping) else getattr(rec, "created", "")) or "")
        names = {repo, repo.split("/")[-1]}
        if names & self.repos:
            return {"ok": False, "rule": "sealed_repo", "reason": f"{repo} is a held-out repository"}
        if "/" in repo and repo.split("/")[0] in self.orgs:
            return {"ok": False, "rule": "sealed_org", "reason": f"{repo.split('/')[0]} owns held-out repositories only"}
        ev = (rec.get("evidence") if isinstance(rec, Mapping) else getattr(rec, "evidence", None)) or {}
        for other in [ev.get("host"), ev.get("consumer")] if isinstance(ev, Mapping) else []:
            if other and {_norm(str(other)), _norm(str(other)).split("/")[-1]} & self.repos:
                return {"ok": False, "rule": "sealed_repo", "reason": f"{other} (paired with {repo}) is a held-out repository"}
        if self.after and created and created[:10] > self.after[:10]:
            return {"ok": False, "rule": "sealed_time", "reason": f"dated {created[:10]}, after the sealed cut {self.after[:10]}"}
        return {"ok": True, "rule": "", "reason": ""}

    def take(self, rec: Any) -> bool:
        d = self.check(rec)
        if isinstance(rec, dict):
            rec["lease"] = d
        else:
            rec.lease = d
        return bool(d["ok"])

    def guard_repo(self, name: str) -> None:
        d = self.check({"repo": name})
        if not d["ok"]:
            raise SealedSourceError(d["reason"])

    def to_json(self) -> dict[str, Any]:
        return {"repos": sorted(self.repos), "orgs": sorted(self.orgs), "after": self.after}
