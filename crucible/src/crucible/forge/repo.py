"""The zoned forge repo (m13 G-7; audit 06 §4.1; GitSwarm as PATTERN, no licence): swarm memory with Crucible's staging rule.

GitSwarm's memory is a git DAG every worker reads; that is the hole a forge cannot have (a Verifier-writer that reads the
oracle writes checks that encode it). Here there is one git repo per family, split into **visibility zones**:

| zone | contents | readable by (default) |
|---|---|---|
| ``public/`` | seeds, specs, instructions, World specs, findings | every role |
| ``oracle/`` | oracle actions / patches, expert end states | builder, adversary, hacker_whitebox, judge |
| ``checks/`` | verifier plans, hidden strings | verifier_writer, adversary, judge |
| ``solver/`` | solver and hacker trajectories with outcomes | calibrator, curator, verifier_writer |

- **Commits** follow GitSwarm: one immutable commit with one parent and a mandatory ``informed_by`` set, message
  ``[<type>|informed_by=<sha>,…] <summary>``. Each commit is kept alive by ``refs/forge/<sha>``; there are no merges.
- **Writes**: an author writes only into zones it can read (a zone it cannot read is one it would write blind).
- **Breach scan** on every commit, before anything is written. The material a commit adds to zone Z must not come
  from a **protected** zone: one the author cannot read, or one that some reader of Z cannot read (information flow:
  an adversary that reads ``oracle/`` must not copy it into ``public/``, which the black-box hacker reads; M8 audit
  V1). A commit fails if the added text carries a protected zone's canary (each zone holds ``<zone>/CANARY``) or
  *quotes* it: any run of ``shingle`` tokens of the added text (prefixes, splits and re-wrapping do not hide it; V2),
  or a short line that carries a number or identifier, indexed per zone over every commit ever made. Material that is
  already in the parent tree in a zone both the author and every reader of Z can read is not evidence (computed from
  the parent, not a global index: a note on an unrelated branch unlocked quoting everywhere). Quoting the oracle is the
  leak that matters, and it rarely carries a canary.
- **Citations**: ``informed_by`` naming an unknown commit or a commit that touched a zone the author cannot read is
  rejected (citing it means it was read).
- **Views**: ``view(role, sha)`` and ``materialize(role, sha, dir)`` export only the readable zones (read-only files),
  so a worker's sandbox is mounted with what its role may see.

Rejected commits leave no object behind and are recorded in ``breaches``. The index (zone line hashes, commit zones)
lives in ``.git/forge-index.json``, so a reopened repo scans exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

ZONES = ("public", "oracle", "checks", "solver")
DEFAULT_READS: dict[str, set[str]] = {
    "proposer": {"public"},
    "builder": {"public", "oracle"},
    "verifier_writer": {"public", "checks", "solver"},
    "adversary": {"public", "oracle", "checks"},
    "hacker_blackbox": {"public"},
    "hacker_whitebox": {"public", "oracle"},
    "calibrator": {"public", "solver"},
    "curator": {"public", "solver"},
    "judge": {"public", "oracle", "checks"},
}
MSG = re.compile(r"^\[(?P<type>[A-Za-z0-9_-]+)\|informed_by=(?P<inf>[0-9a-f,]*)\] (?P<summary>.*)$")


class ZoneBreach(RuntimeError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class Commit:
    sha: str
    type: str
    author: str
    parent: str | None
    informed_by: list[str]
    summary: str
    zones: list[str] = field(default_factory=list)


def _norm(line: str) -> str:
    return " ".join(line.split())


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:20]


_TOK = re.compile(r"\w+|[^\w\s]")


def _marks(text: str, k: int) -> set[str]:
    """Quote evidence of a text: hashes of every run of ``k`` tokens (across line breaks), plus each short line that
    carries a number or an identifier (``amount: 97.31``, ``O-7213``) as one mark."""
    toks = _TOK.findall(text)
    out = {_h("s|" + " ".join(toks[i:i + k])) for i in range(len(toks) - k + 1)}
    for line in text.splitlines():
        lt = _TOK.findall(line)
        if 0 < len(lt) < k and any(any(c.isdigit() for c in t) for t in lt) and len(_norm(line)) >= 5:
            out.add(_h("l|" + " ".join(lt)))
    return out


class ForgeRepo:
    def __init__(self, path: Path, *, family: str, reads: dict[str, set[str]] | None = None, min_line: int = 20, shingle: int = 6):
        self.path, self.family, self.min_line, self.shingle = Path(path), family, min_line, shingle
        self.reads = {k: set(v) for k, v in (reads or DEFAULT_READS).items()}
        self._lock = threading.Lock()
        self._trees: dict[str, dict[str, str]] = {}
        self.breaches: list[dict] = []
        self._ix_path = self.path / ".git" / "forge-index.json"
        if not (self.path / ".git").exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self._git("init", "-q")
            self._index: dict = {"lines": {z: {} for z in ZONES}, "marks": {z: {} for z in ZONES}, "canaries": {}, "commits": {}}
            canaries = {z: f"CRUCIBLE-CANARY-{z}-{_h(family + z + os.urandom(8).hex())[:16]}" for z in ZONES}
            self._index["canaries"] = canaries
            self.root = self._write_commit({f"{z}/CANARY": canaries[z] + "\n" for z in ZONES}, parent=None, type_="root", author="forge",
                                           informed_by=[], summary=f"{family} forge repo")
        else:
            self._index = json.loads(self._ix_path.read_text())
            self.root = next(s for s, c in self._index["commits"].items() if c["type"] == "root")
            if "marks" not in self._index:  # an index from before the shingle scan: rebuild the marks from every commit
                self._index["marks"] = {z: {} for z in ZONES}
                for sha in self._index["commits"]:
                    for p, content in self._tree_files(sha).items():
                        for m in _marks(content, self.shingle):
                            self._index["marks"][p.split("/", 1)[0]].setdefault(m, sha)

    # ---- git ---------------------------------------------------------------------------------------------------------
    def _git(self, *args: str, input: bytes | None = None, env: dict | None = None) -> str:
        e = {**os.environ, "GIT_AUTHOR_NAME": "crucible-forge", "GIT_AUTHOR_EMAIL": "forge@crucible.local", "GIT_COMMITTER_NAME": "crucible-forge",
             "GIT_COMMITTER_EMAIL": "forge@crucible.local", **(env or {})}
        r = subprocess.run(["git", "-C", str(self.path), *args], input=input, capture_output=True, env=e, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout.decode(errors="replace").strip()

    def _tree_files(self, sha: str) -> dict[str, str]:
        """Every file of a commit's tree: one ``ls-tree`` plus one ``cat-file --batch`` (commits are immutable: cached)."""
        hit = self._trees.get(sha)
        if hit is not None:
            return dict(hit)
        entries = [ln.split("\t", 1) for ln in self._git("ls-tree", "-r", sha).splitlines()]
        blobs = [(meta.split()[2], path) for meta, path in entries]
        out: dict[str, str] = {}
        if blobs:
            r = subprocess.run(["git", "-C", str(self.path), "cat-file", "--batch"], input="\n".join(b for b, _ in blobs).encode() + b"\n",
                               capture_output=True, check=True)
            buf, i = r.stdout, 0
            for _, path in blobs:
                nl = buf.index(b"\n", i)
                size = int(buf[i:nl].split()[2])
                out[path] = buf[nl + 1:nl + 1 + size].decode(errors="replace")
                i = nl + 1 + size + 1
        self._trees[sha] = out
        return dict(out)

    def _write_commit(self, files: dict[str, str], *, parent: str | None, type_: str, author: str, informed_by: list[str], summary: str) -> str:
        """Builds the commit with plumbing (no worktree, no checkout): parent tree + files -> tree -> commit."""
        idx_env = {"GIT_INDEX_FILE": str(self.path / ".git" / f"forge-index-{os.getpid()}-{threading.get_ident()}")}
        try:
            if parent:
                self._git("read-tree", parent, env=idx_env)
            else:
                self._git("read-tree", "--empty", env=idx_env)
            for p, content in sorted(files.items()):
                blob = self._git("hash-object", "-w", "--stdin", input=content.encode())
                self._git("update-index", "--add", "--cacheinfo", f"100644,{blob},{p}", env=idx_env)
            tree = self._git("write-tree", env=idx_env)
        finally:
            Path(idx_env["GIT_INDEX_FILE"]).unlink(missing_ok=True)
        msg = f"[{type_}|informed_by={','.join(informed_by)}] {summary}\n\nauthor-role: {author}\n"
        sha = self._git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", msg)
        self._git("update-ref", f"refs/forge/{sha}", sha)
        zones = sorted({p.split("/", 1)[0] for p in files})
        self._index["commits"][sha] = {"type": type_, "author": author, "parent": parent, "informed_by": informed_by, "summary": summary, "zones": zones}
        for p, content in files.items():
            z = p.split("/", 1)[0]
            for line in content.splitlines():
                n = _norm(line)
                if len(n) >= self.min_line or (p.endswith("/CANARY") and n):
                    self._index["lines"][z][_h(n)] = sha
            for m in _marks(content, self.shingle):
                self._index["marks"].setdefault(z, {}).setdefault(m, sha)
        tmp = self._ix_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._index))
        os.replace(tmp, self._ix_path)
        return sha

    # ---- the zone rule ---------------------------------------------------------------------------------------------
    def readable(self, role: str) -> set[str]:
        if role not in self.reads:
            raise KeyError(f"unknown role {role!r} (zones: {sorted(self.reads)})")
        return self.reads[role] | {"public"}

    def readers(self, zone: str) -> list[str]:
        return sorted(r for r in self.reads if zone in self.readable(r))

    def protected(self, role: str, zone: str) -> set[str]:
        """Zones whose material must not appear in what ``role`` writes into ``zone``: the zones it cannot read, and the
        zones some reader of ``zone`` cannot read."""
        out = set(ZONES) - self.readable(role)
        for r in self.readers(zone):
            out |= set(ZONES) - self.readable(r)
        return out - {zone} if zone in self.readable(role) else out

    def _parent_marks(self, parent: str | None, zones: set[str]) -> set[str]:
        files = self._tree_files(parent) if parent else {}
        return {m for p, v in files.items() if p.split("/", 1)[0] in zones for m in _marks(v, self.shingle)}

    def scan(self, role: str, files: dict[str, str], informed_by: list[str], parent: str | None = None) -> list[str]:
        can = self.readable(role)
        reasons = []
        for p in files:
            z = p.split("/", 1)[0]
            if z not in ZONES:
                reasons.append(f"{p}: not in a zone ({', '.join(ZONES)})")
            elif z not in can:
                reasons.append(f"{p}: {role} cannot read zone {z}/, so it cannot write there")
        old = self._tree_files(parent) if parent else {}
        for p, content in files.items():
            z = p.split("/", 1)[0]
            if z not in ZONES:
                continue
            prot = sorted(self.protected(role, z))
            if not prot:
                continue
            prev = {_norm(x) for x in old.get(p, "").splitlines()}
            added = "\n".join(x for x in content.splitlines() if _norm(x) and _norm(x) not in prev)
            if not added:
                continue
            flat, squeezed = _norm(added), "".join(added.split())
            # already readable by the author and by every reader of z, in the parent tree: not evidence of a leak
            common = can.intersection(*(self.readable(r) for r in self.readers(z))) if self.readers(z) else can
            exempt = self._parent_marks(parent, common)
            for y in prot:
                c = self._index["canaries"][y]
                if c in flat or c in squeezed:
                    reasons.append(f"{p}: carries the {y}/ canary")
                    continue
                hit = next((m for m in _marks(added, self.shingle) if m in self._index["marks"].get(y, {}) and m not in exempt), None)
                if hit:
                    why = "it cannot read" if y not in can else "a reader of " + z + "/ cannot read"
                    reasons.append(f"{p}: quotes a line of {y}/ ({why}; commit {self._index['marks'][y][hit][:10]})")
        for s in informed_by:
            c = self._index["commits"].get(s)
            if c is None:
                reasons.append(f"informed_by {s[:10]}: unknown commit")
            elif set(c["zones"]) - can:
                reasons.append(f"informed_by {s[:10]}: cites a commit in {sorted(set(c['zones']) - can)} that {role} cannot read")
        if parent and parent not in self._index["commits"]:
            reasons.append(f"parent {parent[:10]}: unknown commit")
        return reasons

    def commit(self, role: str, files: dict[str, str], *, type_: str, summary: str, informed_by: list[str] | None = None,
               parent: str | None = None) -> Commit:
        """One contribution. Raises ``ZoneBreach`` (and writes nothing) when the zone rule is broken."""
        inf = list(informed_by or [])
        with self._lock:
            parent = parent or self.root
            reasons = self.scan(role, files, inf, parent)
            if reasons:
                self.breaches.append({"role": role, "type": type_, "summary": summary, "reasons": reasons})
                raise ZoneBreach(reasons)
            sha = self._write_commit(files, parent=parent, type_=type_, author=role, informed_by=inf, summary=summary)
        return self.get(sha)

    # ---- reading -----------------------------------------------------------------------------------------------------
    def get(self, sha: str) -> Commit:
        c = self._index["commits"][sha]
        return Commit(sha, c["type"], c["author"], c["parent"], list(c["informed_by"]), c["summary"], list(c["zones"]))

    def commits(self) -> list[Commit]:
        return [self.get(s) for s in self._index["commits"]]

    def parse_message(self, sha: str) -> dict:
        m = MSG.match(self._git("log", "-1", "--format=%s", sha))
        if not m:
            raise ValueError(f"{sha[:10]}: not a forge commit message")
        return {"type": m["type"], "informed_by": [x for x in m["inf"].split(",") if x], "summary": m["summary"]}

    def view(self, role: str, sha: str | None = None) -> dict[str, str]:
        can = self.readable(role)
        return {p: v for p, v in self._tree_files(sha or self.root).items() if p.split("/", 1)[0] in can}

    def materialize(self, role: str, sha: str | None, dest: Path) -> Path:
        dest = Path(dest)
        for p, v in self.view(role, sha).items():
            f = dest / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(v)
            f.chmod(0o444)
        return dest
