"""The seal guard (swarm-pipeline/00 §9; 2026-10-24): no held-out content reaches git by accident.

The held-out debug tasks went public on 2026-10-22 through a ``git add -f`` of ``crucible/data/debug`` (their reports,
check sets with the hidden test ids, gate records). ``.gitignore`` cannot stop ``-f``; this guard runs where every
commit passes (a pre-commit hook, ``.githooks/pre-commit``) and where every push is checked (a CI or test run over a
commit range), and refuses:

- **Sealed paths**, whatever they hold: ``crucible/data/sealed/`` (``SEALED_ROOT``: the home of held-out stores,
  reports, the held-out supply list, the local held-out draw and the guard's own fingerprints), any ``heldout`` /
  ``held-out`` / ``held_out`` directory, ``bench.key``, a benchmark ``sealed.json``.
- **Held-out names** in added lines: a held-out task id (``db-<lang>-<repo>-<8 hex>``, ``kd-``/``km-`` ids, the ids of
  every held-out store window), and a held-out repository as ``owner/repo``. The registries that must name the held-out
  repositories (``REGISTRY``) are exempt from this rule only.
- **Hidden content** in added lines: the sha256 of every fix line, hidden-test line and hidden test name of every
  held-out task (``refresh`` builds them from the stores; the fingerprint file holds hashes, never text).
- **The bench key**: a 32+ hex run whose hash is the bench key's.

An override exists because publishing is the lead's call: ``--i-know "<reason>"`` (or ``$CRUCIBLE_SEAL_OVERRIDE`` for
the hook) lets the commit through and journals who, when, the findings (rule, path, line number, never the text) and
the reason to ``crucible/data/sealguard/overrides.jsonl``.

Already-public files are not this guard's business (``factory.benchaudit`` measures them): hooks and ranges check
added lines and added paths only, and ``tree`` mode compares a whole tree against a frozen baseline, a ratchet.
"""

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SEALED_ROOT = "crucible/data/sealed"
FINGERPRINTS = f"{SEALED_ROOT}/guard_fingerprints.json"
JOURNAL = "crucible/data/sealguard/overrides.jsonl"
OVERRIDE_ENV = "CRUCIBLE_SEAL_OVERRIDE"
# files that must name held-out repositories to keep them out of training (the draw, the benchmark's reasons, the lease)
REGISTRY = ("crucible/src/crucible/factory/lanes/windows.py", "crucible/src/crucible/factory/benchmark.py",
            "crucible/src/crucible/factory/sources/lease.py")
SEALED_PATHS = (re.compile(rf"^{re.escape(SEALED_ROOT)}/"), re.compile(r"(^|/)(heldout|held-out|held_out)(/|$)", re.IGNORECASE),
                re.compile(r"(^|/)bench\.key$"), re.compile(r"(^|/)sealed\.json$"))
LINE_CHARS = 40
TEST_CHARS = 16
W = 16  # window width: hidden content is matched as shared 16-character windows, so a copy inside a longer line is found
NEED = 8  # at least this many windows shared with one hidden item (about a 23-character copy)
COVER = 0.6  # and this share of a hidden code line's windows (a common idiom inside a long line is not a copy of it);
#              a hidden test name must be held whole
CAP = 60  # items per kind per task (the longest), as in the audit
HEX = re.compile(r"\b[0-9a-fA-F]{32,}\b")


def digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:20]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _windows(text: str) -> set[str]:
    t = _norm(text)
    return {hashlib.blake2b(t[i:i + W].encode(), digest_size=6).hexdigest() for i in range(max(0, len(t) - W + 1))}


@dataclass(frozen=True)
class Finding:
    rule: str  # sealed_path | heldout_id | heldout_repo | hidden_content | bench_key
    path: str
    line: int = 0
    detail: str = ""  # a rule-level note, never the matched text


@dataclass
class Seal:
    """What the guard refuses: held-out ids and repos, hashed hidden lines, the bench key hash."""

    ids: frozenset[str] = frozenset()
    repos: frozenset[str] = frozenset()  # local names (``h3``)
    slugs: frozenset[str] = frozenset()  # ``owner/repo``, lower case
    windows: Mapping[str, int] = field(default_factory=dict)  # window hash -> hidden item (16-char windows, hashed)
    need: Mapping[int, int] = field(default_factory=dict)  # hidden item -> windows a line must share with it
    key_hashes: frozenset[str] = frozenset()

    def hidden_in(self, text: str) -> bool:
        if not self.windows:
            return False
        got: dict[int, int] = {}
        for h in _windows(text):
            i = self.windows.get(h)
            if i is not None:
                got[i] = got.get(i, 0) + 1
                if got[i] >= self.need.get(i, NEED):
                    return True
        return False

    def id_pattern(self) -> re.Pattern[str] | None:
        names = sorted({re.escape(r) for r in self.repos} | {re.escape(r.replace("_", "").replace(".", "")) for r in self.repos}, key=len, reverse=True)
        return re.compile(rf"\b(?:db|kd|km)-[a-z]+-(?:{'|'.join(names)})-[0-9a-f]{{6,}}") if names else None

    @classmethod
    def load(cls, root: Path, *, held_out: Mapping[str, str] | None = None, urls: Mapping[str, str] | None = None) -> Seal:
        """From the registries (``windows.HELD_OUT_REPOS``, ``REPO_URLS``, the local draw) and ``FINGERPRINTS``."""
        if held_out is None or urls is None:
            try:
                from crucible.factory.lanes.windows import HELD_OUT_REPOS, REPO_URLS
            except Exception:  # noqa: BLE001 - a hook under a python without the project's deps reads the fingerprint file's copy
                HELD_OUT_REPOS, REPO_URLS = {}, {}
            held_out, urls = HELD_OUT_REPOS if held_out is None else held_out, REPO_URLS if urls is None else urls
        fp = Path(root) / FINGERPRINTS
        d = json.loads(fp.read_text()) if fp.is_file() else {}
        slugs = {urls[n].lower() for n in held_out if n in urls} | {s.lower() for s in d.get("slugs", [])}
        return cls(frozenset(d.get("ids", [])), frozenset(held_out) | frozenset(d.get("repos", [])), frozenset(slugs),
                   dict(d.get("windows", {})), {int(k): int(v) for k, v in d.get("need", {}).items()}, frozenset(d.get("key_hashes", [])))


def refresh(root: Path, tasks: Iterable[Any], *, key: bytes | None = None, extra_slugs: Iterable[str] = (),
            held_out: Mapping[str, str] | None = None, urls: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Write ``FINGERPRINTS`` from held-out tasks (``benchaudit.HeldTask``): ids, hashes of fix and hidden-test lines and
    hidden test names, the bench key's hash, and a copy of the held-out repository lists (a hook without the project's
    deps reads them from here). Text of task content never: hashes only."""
    if held_out is None or urls is None:
        from crucible.factory.lanes.windows import HELD_OUT_REPOS, REPO_URLS

        held_out, urls = HELD_OUT_REPOS if held_out is None else held_out, REPO_URLS if urls is None else urls
    extra_slugs = [*extra_slugs, *(urls[n] for n in held_out if n in urls)]
    ids: set[str] = set()
    items: dict[str, float] = {}  # item -> the share of its windows a line must hold
    for t in tasks:
        ids.add(t.id)
        ids.update(getattr(t, "aliases", []) or [])
        for group, n, share in (([*t.fix_lines], LINE_CHARS, COVER), ([*t.test_lines], LINE_CHARS, COVER),
                                ([re.split(r"::|/", x)[-1] for x in t.hidden_tests], TEST_CHARS, 1.0)):
            for x in sorted({_norm(x) for x in group if len(_norm(x)) >= n}, key=len, reverse=True)[:CAP]:
                items[x] = max(items.get(x, 0.0), share)
    windows: dict[str, int] = {}
    need: dict[int, int] = {}
    for i, (x, share) in enumerate(items.items()):
        ws = _windows(x)
        need[i] = max(min(len(ws), NEED), math.ceil(share * len(ws)))
        for h in ws:
            windows.setdefault(h, i)
    out = {"ids": sorted(ids), "windows": windows, "need": need, "key_hashes": [digest(key.hex())] if key else [],
           "slugs": sorted({s.lower() for s in extra_slugs}), "repos": sorted(held_out), "w": W,
           "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    p = Path(root) / FINGERPRINTS
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, sort_keys=True))
    return {"ids": len(ids), "items": len(need), "windows": len(windows), "key": bool(key), "path": str(p)}


def check_path(path: str, seal: Seal | None = None) -> list[Finding]:
    out = [Finding("sealed_path", path, 0, rx.pattern) for rx in SEALED_PATHS if rx.search(path)][:1]
    if not out and seal is not None and path not in REGISTRY:
        idp = seal.id_pattern()
        if (idp and idp.search(path)) or any(i in path for i in seal.ids):
            out.append(Finding("heldout_id", path, 0, "a held-out task id in the path"))
    return out


def check_lines(path: str, lines: Iterable[tuple[int, str]], seal: Seal) -> list[Finding]:
    """Findings in a file's added lines (line number, text)."""
    out: list[Finding] = []
    registry = path in REGISTRY
    idp = seal.id_pattern()
    for n, text in lines:
        low = text.lower()
        if not registry:
            if (idp and idp.search(text)) or any(i in text for i in seal.ids):
                out.append(Finding("heldout_id", path, n, "a held-out task id"))
                continue
            if any(re.search(rf"(?<![\w.-]){re.escape(s)}(?![\w-])", low) for s in seal.slugs):
                out.append(Finding("heldout_repo", path, n, "a held-out repository as owner/repo"))
                continue
        if seal.hidden_in(text):
            out.append(Finding("hidden_content", path, n, "a hidden test name or fix/test line of a held-out task"))
            continue
        if seal.key_hashes and any(digest(h.lower()) in seal.key_hashes for h in HEX.findall(text)):
            out.append(Finding("bench_key", path, n, "the bench key"))
    return out


# ---- git ---------------------------------------------------------------------------------------------------------------
def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.strip()[:300]}")
    return r.stdout


def added(diff: str) -> dict[str, list[tuple[int, str]]]:
    """``path -> [(new line number, text)]`` of a unified diff's added lines (``--unified=0``)."""
    out: dict[str, list[tuple[int, str]]] = {}
    path, n = None, 0
    for ln in diff.splitlines():
        if ln.startswith("+++ "):
            path = None if ln[4:] == "/dev/null" else (ln[6:] if ln.startswith("+++ b/") else ln[4:])
            if path is not None:
                out.setdefault(path, [])
        elif ln.startswith("@@"):
            m = re.search(r"\+(\d+)", ln)
            n = int(m.group(1)) if m else 0
        elif path is not None and ln.startswith("+") and not ln.startswith("+++"):
            out[path].append((n, ln[1:]))
            n += 1
    return out


def scan_diff(diff: str, seal: Seal, *, new_paths: Iterable[str] = ()) -> list[Finding]:
    found: list[Finding] = []
    lines = added(diff)
    for p in sorted(set(lines) | set(new_paths)):
        found += check_path(p, seal)
        found += check_lines(p, lines.get(p, []), seal)
    return found


def staged(repo: Path, seal: Seal) -> list[Finding]:
    diff = _git(repo, "diff", "--cached", "--unified=0", "--no-color", "--no-ext-diff", "--text")
    names = _git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR").split()
    return scan_diff(diff, seal, new_paths=names)


def commit_range(repo: Path, rng: str, seal: Seal) -> list[Finding]:
    diff = _git(repo, "diff", "--unified=0", "--no-color", "--no-ext-diff", "--text", rng)
    names = _git(repo, "diff", "--name-only", "--diff-filter=ACMR", rng).split()
    return scan_diff(diff, seal, new_paths=names)


def tree(repo: Path, ref: str, seal: Seal) -> list[Finding]:
    """Every tracked path of ``ref``, and every line naming a held-out id or repository (ids and names only: hidden
    content in an old tree is the audit's to measure)."""
    found: list[Finding] = []
    for p in _git(repo, "ls-tree", "-r", "--name-only", ref).splitlines():
        found += check_path(p, seal)
    names = sorted(seal.ids | {s for s in seal.slugs})
    if names:
        args = ["grep", "-I", "-n", "-F", "-i"] + [x for nm in names for x in ("-e", nm)] + [ref, "--"]
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
        for ln in r.stdout.splitlines():
            parts = ln.split(":", 3)
            if len(parts) >= 4 and parts[1] not in REGISTRY:
                found.append(Finding("heldout_id" if any(i.lower() in parts[3].lower() for i in seal.ids) else "heldout_repo", parts[1], int(parts[2])))
    return found


def ratchet(found: Sequence[Finding], baseline: Iterable[Mapping[str, Any]]) -> list[Finding]:
    """Findings not in the frozen baseline (matched on rule and path: an exposure already public is the audit's)."""
    known = {(b.get("rule"), b.get("path")) for b in baseline}
    return [f for f in found if (f.rule, f.path) not in known]


def journal(repo: Path, found: Sequence[Finding], reason: str) -> Path:
    if not reason.strip():
        raise ValueError("an override needs a reason")
    p = Path(repo) / JOURNAL
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        who = _git(repo, "config", "user.name").strip()
    except RuntimeError:
        who = getpass.getuser()
    row = {"at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "who": who, "reason": reason.strip(),
           "findings": [asdict(f) for f in found]}
    with p.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return p


def decide(repo: Path, found: Sequence[Finding], *, reason: str | None = None) -> tuple[bool, str]:
    """(allowed, message). An override reason (argument or ``$CRUCIBLE_SEAL_OVERRIDE``) lets findings through, journaled."""
    if not found:
        return True, "seal guard: clean"
    reason = reason if reason is not None else os.environ.get(OVERRIDE_ENV)
    lines = [f"  {f.rule:15} {f.path}{':' + str(f.line) if f.line else ''}  {f.detail}" for f in found[:40]]
    if reason:
        p = journal(repo, found, reason)
        return True, f"seal guard: {len(found)} findings let through by override, journaled to {p}\n" + "\n".join(lines)
    return False, (f"seal guard: refused, {len(found)} findings (held-out content must stay out of git):\n" + "\n".join(lines)
                   + f"\nTo publish anyway: --i-know \"<reason>\" (hook: {OVERRIDE_ENV}=\"<reason>\" git commit ...)")
