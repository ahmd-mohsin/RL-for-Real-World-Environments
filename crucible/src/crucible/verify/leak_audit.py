"""Leak audit (DESIGN §8.3 layers 1-2, A5): nothing the agent can read may contain hidden material.

- ``hidden_strings``: distinctive lines from hidden material (tests, the oracle's added lines, reference outputs) to
  search for; short or generic lines are skipped so the audit has no false alarms on boilerplate;
- ``audit_tree``: walks a directory (an agent workspace, an exported agent image) for hidden artifact names, canary
  strings (OpenEnv RFC 008) and hidden strings; never follows symlinks, skips special files, caps file size;
- ``audit_image``: the same over ``docker export`` of a container made from the agent image, streamed;
- ``audit_git``: the repo's history may contain nothing beyond the base (``rev-list --all`` within the base's ancestry,
  no unreachable objects, forbidden SHAs absent) - SWE-smith's per-instance branches leak the fix and its tests.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import tarfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

HIDDEN_NAMES = ("tests/hidden", "solution", "solution.patch", "grade.py", "fingerprints.json", "perf_thresholds.json", "reference*",
                "*.oracle", "crucible_reward.json", "milestones.yaml")
GENERIC = re.compile(r"^[\s{}()\[\];,.#/*=+-]*$|^\s*(import|from|#include|#pragma|using namespace)\b")


def hidden_strings(texts: Iterable[str], *, min_len: int = 32, limit: int = 400) -> list[str]:
    out: list[str] = []
    seen = set()
    for t in texts:
        for line in t.splitlines():
            s = line.strip().lstrip("+").strip()
            if len(s) < min_len or GENERIC.match(s) or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                return out
    return out


def oracle_added_lines(patch: str) -> list[str]:
    return [ln[1:] for ln in patch.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


@dataclass
class LeakReport:
    findings: list[dict] = field(default_factory=list)
    scanned_files: int = 0

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict:
        return {"ok": self.ok, "scanned_files": self.scanned_files, "findings": self.findings[:100]}


def _scan_text(rel: str, data: bytes, canaries: Sequence[str], strings: Sequence[str], rep: LeakReport) -> None:
    text = data.decode("utf-8", "ignore")
    for c in canaries:
        if c and c in text:
            rep.findings.append({"file": rel, "kind": "canary", "match": c[:60]})
    for s in strings:
        if s in text:
            rep.findings.append({"file": rel, "kind": "hidden_string", "match": s[:80]})
            break


def _hidden_name(rel: str, names: Sequence[str]) -> str | None:
    return next((n for n in names if fnmatch.fnmatch(rel, f"*{n}") or fnmatch.fnmatch(Path(rel).name, n) or f"/{n}/" in f"/{rel}/"), None)


def audit_tree(root: Path, *, canaries: Sequence[str] = (), strings: Sequence[str] = (), names: Sequence[str] = HIDDEN_NAMES,
               max_bytes: int = 4 << 20, skip: Sequence[str] = (".git/objects",)) -> LeakReport:
    rep = LeakReport()
    root = Path(root)
    for p in root.rglob("*"):
        rel = p.relative_to(root).as_posix()
        if any(rel.startswith(s) for s in skip) or p.is_symlink() or not p.is_file():
            continue
        if (n := _hidden_name(rel, names)) is not None:
            rep.findings.append({"file": rel, "kind": "hidden_artifact", "match": n})
        rep.scanned_files += 1
        if p.stat().st_size <= max_bytes:
            _scan_text(rel, p.read_bytes(), canaries, strings, rep)
    return rep


def audit_image(image: str, *, roots: Sequence[str] = ("workspace", "app", "root", "home", "opt", "tmp", "srv"), canaries: Sequence[str] = (),
                strings: Sequence[str] = (), names: Sequence[str] = HIDDEN_NAMES, max_bytes: int = 4 << 20, docker_host: str | None = None) -> LeakReport:
    import os

    env = {**os.environ, **({"DOCKER_HOST": docker_host} if docker_host else {})}
    cid = subprocess.run(["docker", "create", image], capture_output=True, text=True, check=True, env=env).stdout.strip()
    rep = LeakReport()
    try:
        p = subprocess.Popen(["docker", "export", cid], stdout=subprocess.PIPE, env=env)
        with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
            for m in tf:
                rel = m.name.lstrip("./")
                if not m.isfile() or rel.split("/", 1)[0] not in roots:
                    continue
                if (n := _hidden_name(rel, names)) is not None:
                    rep.findings.append({"file": rel, "kind": "hidden_artifact", "match": n})
                rep.scanned_files += 1
                if m.size <= max_bytes and (f := tf.extractfile(m)) is not None:
                    _scan_text(rel, f.read(), canaries, strings, rep)
        p.wait()
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False, env=env)
    return rep


def audit_git(repo: Path, *, base: str | None = None, forbidden: Sequence[str] = ()) -> dict:
    def git(*a: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=False)

    if git("rev-parse", "--git-dir").returncode != 0:
        return {"ok": True, "repo": False}
    head = git("rev-parse", "HEAD").stdout.strip()
    base = base or head
    all_commits = set(git("rev-list", "--all").stdout.split())
    ancestry = set(git("rev-list", base).stdout.split())
    beyond = sorted(all_commits - ancestry)
    unreachable = [ln for ln in git("fsck", "--unreachable", "--no-reflogs").stdout.splitlines() if ln.startswith("unreachable")]
    present = [s for s in forbidden if git("cat-file", "-e", s).returncode == 0]
    refs = git("for-each-ref", "--format=%(refname)").stdout.split()
    return {"ok": not beyond and not unreachable and not present, "commits": len(all_commits), "beyond_base": beyond[:20], "unreachable": unreachable[:20],
            "forbidden_present": present, "refs": refs[:20]}
