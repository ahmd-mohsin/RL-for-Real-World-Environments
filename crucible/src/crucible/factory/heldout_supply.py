"""Held-out supply for the swarm (swarm-pipeline/00 §9; 2026-10-24): fresh repositories for a sealed benchmark.

The benchmark needs 13-15 clean held-out tasks (``bench.power``). A clean source is one whose owner has never touched
training and never appeared in our public repository: the debug held-out tasks went public on 2026-10-22, so every owner
named anywhere in our git history is spent for the benchmark.

``rank`` turns search results into ranked candidates, executed where it is cheap:

1. **Disjoint owners.** An owner of any registered source (``windows.REPO_URLS``, held out or training) is refused, and
   so is an owner or ``owner/repo`` that ``seen`` finds in our history (``git grep`` over the tracked tree of every ref
   and ``git log`` messages).
2. **Licence and language.** Permissive licences only (``LICENCES``), the lanes' languages (``LANGS``), not archived, not a
   fork.
3. **Supply, measured.** A blobless, shallow clone (``--filter=blob:none --shallow-since``) gives every commit and tree
   after the cutoff without file contents; a **fix with tests** is a non-merge commit after ``cutoff`` whose subject reads
   as a fix and that changes at least one test file and one source file. That count is the expected supply (the debug
   lane's own S1 fix check runs later, on the swarm's side, with a full clone).
4. **CPU probe, static.** The language's build and test manifest exists at HEAD (``go.mod``, ``Cargo.toml``,
   ``package.json``, ``pyproject.toml``/``setup.py``/``setup.cfg``, ``pom.xml``/``build.gradle``). The executed build-and-test
   probe is the lane's scout stage, after the draw.

The ranked list is written under ``windows.SEALED_DIR`` (the seal guard refuses that path): it names the next held-out
repositories, so it never goes to git. ``draw`` turns the chosen ones into ``heldout_draw.json``, which
``windows.local_draw`` reads at import, so every lane, store and the eval lease hold them out from then on.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CUTOFF = "2026-05-01"
LICENCES = ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "mpl-2.0", "unlicense", "0bsd")
LANGS = {"Go": "go", "Rust": "rust", "TypeScript": "node", "JavaScript": "node", "Python": "python", "Java": "java"}
MANIFESTS = {"go": ("go.mod",), "rust": ("Cargo.toml",), "node": ("package.json",), "python": ("pyproject.toml", "setup.py", "setup.cfg"),
             "java": ("pom.xml", "build.gradle", "build.gradle.kts")}
FIX = re.compile(r"\b(fix(e[sd])?|bug(fix)?|regression|crash(es)?|panic(s)?|incorrect(ly)?|wrong)\b", re.IGNORECASE)
NOT_FIX = re.compile(r"\b(typo|docs?|readme|ci|chore|bump|deps?|dependabot|lint|format(ting)?)\b", re.IGNORECASE)
TEST = re.compile(r"(^|/)(tests?|__tests__|testdata|spec|src/test)(/|$)|(^|/)(test_[^/]*\.py|[^/]*_test\.(go|py)|[^/]*\.(test|spec)\.[jt]sx?|[^/]*Test\.java)$")
SOURCE = re.compile(r"\.(go|rs|ts|tsx|js|mjs|cjs|py|java|kt)$")


@dataclass
class Candidate:
    slug: str
    lang: str
    licence: str
    stars: int
    pushed: str
    fixes_with_tests: int | None = None
    fixes: int | None = None
    commits: int | None = None
    manifest: bool | None = None
    refused: str | None = None
    evidence: list[str] = field(default_factory=list)  # a few fix-commit shas (never their text)

    @property
    def owner(self) -> str:
        return self.slug.split("/")[0].lower()


def from_search(items: Iterable[Mapping[str, Any]]) -> list[Candidate]:
    out = []
    for it in items:
        lang = LANGS.get(str(it.get("language") or ""))
        lic = str((it.get("license") or {}).get("spdx_id") or (it.get("license") or {}).get("key") or "").lower()
        if not lang or it.get("archived") or it.get("fork"):
            continue
        out.append(Candidate(str(it["full_name"]), lang, lic, int(it.get("stargazers_count") or 0), str(it.get("pushed_at") or "")))
    return out


def registered_owners(urls: Mapping[str, str]) -> set[str]:
    return {v.split("/")[0].lower() for v in urls.values() if "/" in v}


def seen(repo: Path, needle: str, refs: Sequence[str] = ("--all",)) -> bool:
    """Has ``needle`` (an owner or ``owner/repo``) ever appeared in our repository: any ref's tracked tree, any commit
    message? Case-insensitive, whole word."""
    r = subprocess.run(["git", "-C", str(repo), "log", *refs, "-i", "-F", "--grep", needle, "-1", "--format=%h"], capture_output=True, text=True, check=False)
    if r.stdout.strip():
        return True
    for ref in ("HEAD", "origin/main", "origin/dev"):
        g = subprocess.run(["git", "-C", str(repo), "grep", "-I", "-l", "-i", "-w", "-F", "-e", needle, ref, "--"], capture_output=True, text=True, check=False)
        if g.returncode == 0 and g.stdout.strip():
            return True
    return False


def seen_many(repo: Path, needles: Iterable[str], refs: Sequence[str] = ("HEAD", "origin/main", "origin/dev")) -> set[str]:
    """``seen`` for many needles at once: one ``git grep -o`` per ref and one pass over every commit message."""
    want = sorted({n.lower() for n in needles if n})
    if not want:
        return set()
    got: set[str] = set()
    pats = [x for n in want for x in ("-e", n)]
    for ref in refs:
        g = subprocess.run(["git", "-C", str(repo), "grep", "-I", "-h", "-o", "-i", "-w", "-F", *pats, ref, "--"], capture_output=True, text=True, check=False)
        got.update(x.strip().lower() for x in g.stdout.splitlines())
    msgs = subprocess.run(["git", "-C", str(repo), "log", "--all", "--format=%B"], capture_output=True, text=True, check=False).stdout.lower()
    got.update(n for n in want if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", msgs))
    return got & set(want)


def refuse(c: Candidate, *, owners: set[str], seen_fn: Callable[[str], bool]) -> str | None:
    if c.licence not in LICENCES:
        return f"licence {c.licence or 'none'}"
    if c.owner in owners:
        return "owner of a registered source"
    if seen_fn(c.slug) or seen_fn(c.owner):
        return "named in our git history"
    return None


def fix_commits(log: str) -> tuple[int, int, int, list[str]]:
    """(commits, fixes, fixes with tests, shas) from ``git log --no-merges --format=@@%H%x09%s --name-only``."""
    commits = fixes = with_tests = 0
    shas: list[str] = []
    for block in log.split("@@")[1:]:
        lines = [x for x in block.splitlines() if x.strip()]
        if not lines:
            continue
        sha, _, subject = lines[0].partition("\t")
        files = lines[1:]
        commits += 1
        if not FIX.search(subject) or NOT_FIX.search(subject.split(":")[0]):
            continue
        fixes += 1
        if any(TEST.search(f) for f in files) and any(SOURCE.search(f) and not TEST.search(f) for f in files):
            with_tests += 1
            shas.append(sha[:10])
    return commits, fixes, with_tests, shas


def measure(c: Candidate, clones: Path, *, cutoff: str = CUTOFF, since: str = "2026-04-15", timeout: int = 600) -> Candidate:
    """Blobless shallow clone, then the post-cutoff log and the manifest check (no file content is read)."""
    d = Path(clones) / f"{c.slug.replace('/', '__')}.git"
    if not d.exists():
        r = subprocess.run(["git", "clone", "-q", "--bare", "--filter=blob:none", f"--shallow-since={since}", "--single-branch",
                            f"https://github.com/{c.slug}.git", str(d)], capture_output=True, text=True, timeout=timeout, check=False)
        if r.returncode != 0:
            c.refused = f"clone failed: {r.stderr.strip()[:120]}"
            return c
    log = subprocess.run(["git", "-C", str(d), "log", "--no-merges", f"--since={cutoff}", "--format=@@%H%x09%s", "--name-only"],
                         capture_output=True, text=True, check=False).stdout
    c.commits, c.fixes, c.fixes_with_tests, shas = fix_commits(log)
    c.evidence = shas[:5]
    top = subprocess.run(["git", "-C", str(d), "ls-tree", "--name-only", "HEAD"], capture_output=True, text=True, check=False).stdout.split()
    c.manifest = any(m in top for m in MANIFESTS[c.lang])
    return c


def ranked(cands: Sequence[Candidate]) -> list[Candidate]:
    ok = [c for c in cands if not c.refused and c.manifest and (c.fixes_with_tests or 0) > 0]
    return sorted(ok, key=lambda c: (-(c.fixes_with_tests or 0), -c.stars))


def write(path: Path, cands: Sequence[Candidate], *, meta: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "ranked": [asdict(c) for c in ranked(cands)],
                                "refused": [asdict(c) for c in cands if c.refused or not c.manifest or not c.fixes_with_tests]}, indent=1))


def draw(path: Path, chosen: Sequence[Candidate], *, names: Mapping[str, str] | None = None) -> dict[str, str]:
    """Write ``heldout_draw.json`` for ``windows.local_draw``: local name -> owner/repo. The draw only grows: a repository
    already drawn stays held out."""
    have = json.loads(path.read_text()).get("repos", {}) if path.is_file() else {}
    for c in chosen:
        name = (names or {}).get(c.slug) or c.slug.split("/")[1].lower()
        have.setdefault(name, c.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"repos": have}, indent=1, sort_keys=True))
    return have
