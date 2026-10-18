"""Generic scouts (swarm-pipeline/00 §2): intent card -> queries -> typed source records, filtered before any model call.

**Query expansion without a model** (``card_terms``, ``expand``): the intent card's own words (behaviour, failure mode,
difficulty axes, coverage), stopwords dropped, plus adjacent-word bigrams, are put through fixed templates per record
type (``TEMPLATES``); a consumer search fills the templates with a host's names instead. Deterministic: the same card
gives the same queries, so a rerun hits the cache.

**Search** (``Scout``): repositories, commits and issues through ``github.GitHub`` (unauthenticated, cached, paced).
**Snowball**: from a found record to the repositories that depend on it (a consumer's packaging names the host; read
from raw files, never the API quota) and to its forks (``forks``: the API lists them; GitHub's dependents graph has no
REST endpoint, so dependents come from the requirement scan only).

**Filters, in order, before anything costly** (every result is recorded on the record, kept or not):
1. the eval lease (``lease.EvalLease``): sealed repositories, orgs and the sealed time window never pass;
2. licence: an OSI permissive or weak-copyleft licence (``LICENCES``) GitHub detected;
3. size: the repository is at most ``max_size_kb``; not a fork, not archived;
4. the CPU probe (``cpu_probe``, executed): a full clone, the World image of its HEAD (``windows_world.ImageBook``), and
   its own test suite in a ``--network none`` container: at least ``min_pass`` tests pass and at most ``max_fail_frac``
   fail.

No model is called anywhere here.
"""

from __future__ import annotations

import itertools
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.sources.github import GitHub, RateLimited
from crucible.factory.sources.lease import EvalLease
from crucible.factory.sources.records import SourceRecord

LICENCES = frozenset({"mit", "bsd-2-clause", "bsd-3-clause", "apache-2.0", "isc", "psf-2.0", "mpl-2.0", "unlicense", "0bsd", "zlib", "lgpl-3.0",
                      "lgpl-2.1"})
_STOPWORDS = """a an and are as at be been but by can do does for from has have how in into is it its of on or that the their them then
there these this to was were what when where which while who will with without within real python code codebase task tasks agent model
models user users make makes made keep keeps kept so such than too very much more most less least any all each every other
carry out loses lose track earlier later step steps multi uses used using"""
STOP = frozenset(_STOPWORDS.split())
TEMPLATES: dict[str, tuple[str, ...]] = {
    "host_repo": ("{term} library language:python", "{term} in:name,description language:python"),
    "consumer": ("{host} in:name,description language:python", "{host} plugin language:python", "{host} extension language:python",
                 "{host} integration language:python"),
    "real_fix": ("fix {term}", "{term} bug fix", "regression {term}"),
    "real_perf": ("faster {term}", "speed up {term}", "{term} performance"),
    "real_change": ("add {term}", "support {term}"),
    "request": ("{term} is:issue is:closed language:python", "{term} feature request is:issue language:python"),
}
PACKAGING_FILES = ("pyproject.toml", "setup.cfg", "setup.py", "requirements.txt")


# ---- query expansion -------------------------------------------------------------------------------------------------------------
def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9+-]{2,}", text.lower()) if w not in STOP]


def card_terms(card: Any, *, limit: int = 8) -> list[str]:
    """Search terms from an intent card (``cards.IntentCard`` or a mapping): content words of its behaviour, failure mode,
    difficulty axes and coverage, most frequent first, then adjacent bigrams of the behaviour."""
    get = (lambda k: card.get(k)) if isinstance(card, Mapping) else (lambda k: getattr(card, k, None))
    parts = [str(get("behaviour") or ""), str(get("failure_mode") or ""), " ".join(get("difficulty_axes") or []),
             " ".join((get("coverage") or {}).values())]
    counts: dict[str, int] = {}
    for p in parts:
        for w in _words(p):
            counts[w] = counts.get(w, 0) + 1
    singles = sorted(counts, key=lambda w: (-counts[w], w))
    bw = _words(parts[0])
    bigrams = [f"{a} {b}" for a, b in itertools.pairwise(bw)]
    return list(dict.fromkeys([*singles[:limit // 2], *bigrams[:2], *singles]))[:limit]


def expand(card: Any, kind: str, *, hosts: Sequence[str] = (), limit: int = 12) -> list[str]:
    """The queries for one record type: every template of ``kind`` filled with each term (or each host for consumer
    searches), interleaved so the first queries cover every template."""
    tpls = TEMPLATES[kind]
    fills = list(hosts) if "{host}" in tpls[0] else card_terms(card)
    out: list[str] = []
    for f in fills:
        for t in tpls:
            out.append(t.format(term=f, host=f))
    return list(dict.fromkeys(out))[:limit]


# ---- requirement scan ------------------------------------------------------------------------------------------------------------
def depends_on(text: str, dist: str) -> str | None:
    """The requirement line naming ``dist`` in packaging text (pyproject, setup.cfg, setup.py, requirements), or None."""
    name = re.sub(r"[-_.]+", "[-_.]", re.escape(dist).replace("\\-", "-").replace("\\.", ".").replace("\\_", "_"))
    m = re.search(rf"""(?im)^[^#\n]*?(?:^|[\s"',\[(])({name})(?:\[[^\]]*\])?\s*(?:[<>=!~]=?[^"',\n;]*)?(?=["',\s;)\]]|$)""", text or "")
    return m.group(0).strip()[:200] if m else None


# ---- the scout -------------------------------------------------------------------------------------------------------------------
class Scout:
    """Searches for typed source records for one intent card (see the module doc)."""

    def __init__(self, gh: GitHub, lease: EvalLease, *, max_size_kb: int = 30_000, min_stars: int = 10, licences: Iterable[str] = LICENCES,
                 exclude: Iterable[str] = ()):
        self.gh, self.lease = gh, lease
        self.max_size_kb, self.min_stars, self.licences = max_size_kb, min_stars, frozenset(licences)
        self.exclude = {e.lower() for e in exclude}
        self.log: list[dict[str, Any]] = []

    def _repo_record(self, item: Mapping[str, Any], kind: str, query: str, **evidence: Any) -> SourceRecord:
        lic = ((item.get("license") or {}).get("spdx_id") or (item.get("license") or {}).get("key") or "").lower() or None
        return SourceRecord(kind, str(item.get("full_name") or ""), url=str(item.get("html_url") or ""), title=str(item.get("description") or "")[:300],
                            licence=lic, size_kb=item.get("size"), stars=item.get("stargazers_count"), created=str(item.get("pushed_at") or item.get("created_at") or ""),
                            evidence={"default_branch": item.get("default_branch") or "main", "fork": bool(item.get("fork")),
                                      "archived": bool(item.get("archived")), **evidence}, query=query)

    def filter(self, rec: SourceRecord) -> bool:
        """Lease first, then licence and size (no clone, no model). Records the result on ``rec``; True when it passes."""
        if not self.lease.take(rec):
            return False
        lic_ok = (rec.licence or "") in self.licences
        rec.filters["licence"] = {"ok": lic_ok, "value": rec.licence}
        size_ok = rec.size_kb is not None and rec.size_kb <= self.max_size_kb and not rec.evidence.get("fork") and not rec.evidence.get("archived") \
            and (rec.stars or 0) >= self.min_stars
        rec.filters["size"] = {"ok": size_ok, "size_kb": rec.size_kb, "stars": rec.stars, "fork": rec.evidence.get("fork"), "archived": rec.evidence.get("archived")}
        return lic_ok and size_ok

    def repos(self, queries: Sequence[str], kind: str = "host_repo", *, per_query: int = 30) -> list[SourceRecord]:
        out: dict[str, SourceRecord] = {}
        for q in queries:
            try:
                items = self.gh.search_repos(q, per_page=per_query, sort="stars")
            except RateLimited as e:
                self.log.append({"query": q, "error": str(e)})
                break
            self.log.append({"query": q, "results": len(items)})
            for it in items:
                name = str(it.get("full_name") or "")
                if name and name.lower() not in out and name.lower() not in self.exclude:
                    rec = self._repo_record(it, kind, q)
                    self.filter(rec)
                    out[name.lower()] = rec
        return list(out.values())

    def consumers_of(self, host: str, dist: str, queries: Sequence[str], *, per_query: int = 30, max_scan: int = 60) -> list[SourceRecord]:
        """Repositories the queries find whose packaging (raw files of the default branch) requires ``dist``: ``consumer``
        records with ``evidence.host`` and the requirement line (the snowball from a host to its dependents)."""
        found = self.repos(queries, "consumer", per_query=per_query)
        out = []
        scanned = 0
        for rec in found:
            out.append(rec)
            if not rec.passed or scanned >= max_scan:
                continue  # a record that failed a cheaper filter is kept (the funnel counts it), never read further
            scanned += 1
            br = rec.evidence.get("default_branch") or "main"
            line = None
            for f in PACKAGING_FILES:
                try:
                    txt = self.gh.raw(rec.repo, br, f)
                except RateLimited:
                    txt = None
                if txt and (line := depends_on(txt, dist)):
                    break
            rec.evidence.update(host=host, dist=dist, requirement=line)
            rec.filters["depends_on_host"] = {"ok": bool(line), "line": line}
        return out

    def forks(self, rec: SourceRecord, *, limit: int = 10) -> list[SourceRecord]:
        """Forks of a record's repository (snowball), each a record of the same type, leased and filtered."""
        try:
            items = self.gh.forks(rec.repo, per_page=limit)
        except RateLimited:
            return []
        outp = []
        for it in items:
            r = self._repo_record({**it, "fork": False}, rec.type, f"forks:{rec.repo}", via=f"fork_of:{rec.repo}")
            self.filter(r)
            outp.append(r)
        return outp

    def commits(self, queries: Sequence[str], kind: str) -> list[SourceRecord]:
        """``real_fix`` / ``real_perf`` / ``real_change`` records from commit search (leased; licence and size are checked
        on the repository when it is cloned)."""
        out: list[SourceRecord] = []
        for q in queries:
            try:
                items = self.gh.search_commits(q, per_page=30)
            except RateLimited as e:
                self.log.append({"query": q, "error": str(e)})
                break
            for it in items:
                repo = str(((it.get("repository") or {}).get("full_name")) or "")
                rec = SourceRecord(kind, repo, sha=str(it.get("sha") or ""), url=str(it.get("html_url") or ""),
                                   title=str((it.get("commit") or {}).get("message") or "")[:300],
                                   created=str((((it.get("commit") or {}).get("committer") or {}).get("date")) or ""), query=q)
                self.lease.take(rec)
                out.append(rec)
        return out

    def issues(self, queries: Sequence[str]) -> list[SourceRecord]:
        out: list[SourceRecord] = []
        for q in queries:
            try:
                items = self.gh.search_issues(q, per_page=30)
            except RateLimited as e:
                self.log.append({"query": q, "error": str(e)})
                break
            for it in items:
                repo = "/".join(str(it.get("repository_url") or "").rstrip("/").split("/")[-2:])
                rec = SourceRecord("request", repo, url=str(it.get("html_url") or ""), title=str(it.get("title") or "")[:300],
                                   text=str(it.get("body") or "")[:2000], created=str(it.get("created_at") or ""), query=q,
                                   evidence={"state": it.get("state"), "labels": [x.get("name") for x in it.get("labels") or []]})
                self.lease.take(rec)
                out.append(rec)
        return out


# ---- the CPU probe ---------------------------------------------------------------------------------------------------------------
def clone(full_name: str, dest_root: Path) -> Path:
    """A full clone (history is the ground truth) of ``full_name`` as ``<dest_root>/<owner>__<name>``."""
    dest = Path(dest_root) / full_name.replace("/", "__")
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "-q", f"https://github.com/{full_name}.git", str(dest)], capture_output=True, text=True, timeout=900, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"clone {full_name}: {r.stderr[-300:]}")
    return dest


async def cpu_probe(rec: SourceRecord, clones: Path, *, images: Any, sandboxes: Any, min_pass: int = 10, max_fail_frac: float = 0.2) -> bool:
    """Clone, build the World image of HEAD, run the repository's own suite with no network: the record's
    ``filters["cpu_probe"]`` holds the counts; True when it passes."""
    import asyncio
    import shutil
    import tempfile

    from crucible.factory.lanes.windows import normalize_ids, suite_targets
    from crucible.factory.lanes.windows_git import RepoGit

    try:
        d = await asyncio.to_thread(clone, rec.repo, clones)
    except Exception as e:  # noqa: BLE001 - a clone that fails is a failed probe, with its reason
        rec.filters["cpu_probe"] = {"ok": False, "why": str(e)[:300]}
        return False
    with tempfile.TemporaryDirectory(prefix="p5probe-") as td:
        tree = Path(td) / "t"
        ok, bad = await asyncio.to_thread(RepoGit(d).build, "HEAD", [], tree)
        if not ok:
            rec.filters["cpu_probe"] = {"ok": False, "why": f"HEAD does not check out ({bad})"}
            return False
        card = await images.ensure(d.name, tree)
        if not card["ok"]:
            rec.filters["cpu_probe"] = {"ok": False, "why": "the World image does not build", "log": str(card.get("build_log", ""))[-300:]}
            return False
        targets = suite_targets(tree, {})
        tr = await sandboxes(card["image"]).run(tree, targets)
        res = normalize_ids(tr.results, targets) if tr.infra == "ok" else {}
        shutil.rmtree(tree, ignore_errors=True)
    passed = sum(1 for v in res.values() if v == "passed")
    failed = sum(1 for v in res.values() if v in ("failed", "error"))
    ok = passed >= min_pass and failed <= max_fail_frac * max(1, passed + failed)
    rec.filters["cpu_probe"] = {"ok": ok, "passed": passed, "failed": failed, "image": card["image"], "clone": str(d), "infra": tr.infra}
    return ok
