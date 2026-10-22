"""Rebreak (swarm-pipeline/00 §2, §7.4, §15 P5): N real fix commits reverted in one tree at release B. Each fault is a real
bug, caught by the test its fix added or changed; faults that interact through shared code make the task long.

**Fix detection** (``fix_units``): first-parent units before B (``since`` to B) whose message says fix, bug, regression,
crash, wrong or incorrect (not typo, docs, CI, lint, typing or a release bump), that change source ``.py`` and test
``.py`` files and nothing else but prose (no packaging, config or test plumbing).

**Executed at B** (``real_fix_check``, one container run per candidate): the tree B with only the fix's *source* reverted
(3-way) runs the fix's test files as they are at B. The fix is real when at least one test its commit added or changed
(the test function's AST differs across the commit) passes at B and fails after the revert: the revert re-introduces a
bug its own test catches. Each kept fix becomes a ``real_fix`` source record with that evidence.

**Groups** (``group_fixes``): fixes that change the same functions (``windows.changed_defs``), else the same source files,
are connected; each component is cut, in history order, into groups of 2..``max_size``. Shared code is where faults
interact.

**The tree** (``build``): B with the whole group reverted, newest first (every file each fix changed: source, tests,
changelog lines), through ``base.apply_chain``; a fix whose revert conflicts is dropped and the group retried. The
synthetic history is A' (every fix of the group reverted), then one commit per fix re-applied, oldest first, ending at
B's own tree (B' == B, checked). ``windows.validate_window`` then runs on A'..B' unchanged: the World at A', leak probes
(B's release, the fixes' test names), fail-to-pass with reruns, prefix ownership (milestone = the tests a fix's unit
makes pass), the executed dependency prober and the commit-revert mutants.

**Interacting faults** are the prober's executed edges: fix k's tests fail on A' + fix k (the other faults still in) and
pass once fix j is also applied, so reverting both changes k's behaviour against reverting k alone. ``S6`` needs one.

**The task** (``RebreakTask``): oracle = the tree at B; milestones = per-fix test groups; ``simplify('locate')`` (also
``'faulty_function'``) names the faulty functions and files of the target milestone's fix; ``'prerequisites'`` pre-applies
the fixes the target's fix needs.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.generators.base import (
    WORLD_PREFIX,
    AuthorStyle,
    Candidate,
    Funnel,
    GeneratedTask,
    GenLane,
    apply_chain,
    changed_paths,
    commit_meta,
    derived_clone,
    drop_crlf_prose,
    full_patch,
    git,
    out,
    tree_of,
    with_files,
    write_chain,
)
from crucible.factory.lanes.executable import is_protected, is_test_path
from crucible.factory.lanes.windows import (
    ImageBook,
    WindowStore,
    changed_defs,
    normalize_ids,
    subtests_of,
    suite_targets,
    validate_window,
)
from crucible.factory.lanes.windows_git import RepoGit, Unit, Window, _bodies, _defs, first_parent_units, key_of_test, version_tags
from crucible.factory.sources.lease import EvalLease
from crucible.factory.sources.records import SourceRecord

NAME = "rebreak"
FIX_RX = re.compile(r"\b(fix(e[sd])?|fixing|bug(s|fix)?|regression|crash(es|ed)?|incorrect(ly)?|wrong(ly)?|broken|mistaken(ly)?)\b", re.IGNORECASE)
NOT_FIX_RX = re.compile(r"\b(typos?|docs?|documentation|readme|changelog|ci|lint(ing)?|flake8|mypy|pyright|typing|type hints?|annotations?|"
                        r"warnings?|deprecat\w*|release|bump|version|tests? only|coverage|pre-commit|black|isort|ruff)\b", re.IGNORECASE)
PROSE = re.compile(r"\.(md|rst|txt)$|(^|/)(docs?|changes|changelog|history|news)(/|$)|(^|/)(CHANGES|CHANGELOG|HISTORY|NEWS)[^/]*$", re.IGNORECASE)
STYLE = AuthorStyle(
    role="author",
    brief=("You write the bug report a maintainer of this Python library would file for a developer: several things the library does "
           "wrong today. You get, per fault id, the message of the change that once fixed that fault (the fix is gone now), the public "
           "names it touched, the changelog lines, and the library's README head. Write one requirement per fault id, each one or two "
           "plain sentences stating the wrong behaviour a user sees and the correct behaviour expected, in terms of the public API "
           "(inputs, outputs, errors, formats), never how to fix it and never where the code is. Never write file paths, module file "
           "names, test names, commit ids, issue or pull request numbers, links, version numbers, or process instructions (what tools to use or avoid). "
           "Return {opening, requirements: [{id, text}], closing}: opening is two or three sentences of context in the maintainer's voice, "
           "closing is optional, requirements has exactly one entry per change id. Names listed in a change's must_name (or in "
           "must_name_any) are public names the code must expose under exactly that name: write each one, verbatim, in the requirement "
           "of its change (must_name_any: in the one it fits)."),
    heading="What is broken:")


def allowed_file(path: str) -> bool:
    """A file a rebreak fix may change: source, tests (no plumbing), prose. Packaging and config change the World. Under a
    non-Python profile (``factory.langs``): its source and test files, prose, and test data (fixtures, snapshots)."""
    from crucible.factory.langs.base import active

    if (lang := active()) is not None:
        if PROSE.search(path):
            return True
        if lang.is_plumbing(path):
            return False
        return lang.is_test_path(path) or lang.is_source(path)
    if PROSE.search(path):
        return True
    if path.endswith(".py") and is_test_path(path):
        return not path.endswith("conftest.py")
    return path.endswith(".py") and not is_protected(path)


def is_fix(u: Unit) -> bool:
    text = f"{u.subject}\n{u.body}"
    return (bool(FIX_RX.search(text)) and not NOT_FIX_RX.search(u.subject) and bool(u.src_files) and bool(u.test_files)
            and all(allowed_file(p) for p in u.files))


def fix_units(repo_dir: Path, b: str, *, since: str) -> list[Unit]:
    """Fix units on B's first-parent path after ``since`` (an ISO date), oldest first."""
    start = out(git(repo_dir, "rev-list", "--first-parent", "-1", f"--before={since}", b, check=False))
    if not start:
        start = out(git(repo_dir, "rev-list", "--max-parents=0", b)).split()[-1]
    units = first_parent_units(repo_dir, start, b)
    _bodies(repo_dir, units)
    return [u for u in units if is_fix(u)]


def _spans(text: str) -> dict[str, tuple[int, int, bool]]:
    """``Class::name`` -> (first line incl. decorators, last line, inside a class) of every function, 0-based lines."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    outp: dict[str, tuple[int, int, bool]] = {}

    def walk(nodes: Sequence[ast.AST], prefix: str) -> None:
        for n in nodes:
            if isinstance(n, ast.ClassDef):
                walk(n.body, f"{prefix}{n.name}::")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([n.lineno, *[d.lineno for d in n.decorator_list]]) - 1
                outp[prefix + n.name] = (start, (n.end_lineno or n.lineno) - 1, bool(prefix))

    walk(tree.body, "")
    return outp


def strip_tests(text: str, keys: set[str]) -> str:
    """``text`` without the test functions ``keys`` (``Class::name``): the hidden tests a fix added or changed are never
    in the agent's tree (the grader writes B's file back). A class left empty gets ``pass``."""
    spans = {k: v for k, v in _spans(text).items() if k in keys and k.rsplit("::", 1)[-1].startswith("test")}
    if not spans:
        return text
    lines = text.splitlines(keepends=True)
    for pad in (False, True):
        new = list(lines)
        for _k, (a, b, in_class) in sorted(spans.items(), key=lambda kv: -kv[1][0]):
            indent = re.match(r"\s*", lines[a]).group(0) if in_class else ""
            new[a: b + 1] = [f"{indent}pass\n"] if pad and in_class else []
        cand = "".join(new)
        try:
            ast.parse(cand)
            return cand
        except SyntaxError:
            continue
    return text


def release_window_of(repo_dir: Path, tags: Sequence[tuple[str, str, str, str]], shas: set[str]) -> dict[str, int]:
    """sha -> index i of the release tag whose window (tag i-1, tag i] on the first-parent path holds it."""
    found: dict[str, int] = {}
    for i in range(1, len(tags)):
        a, b = tags[i - 1][1], tags[i][1]
        r = git(repo_dir, "rev-list", "--first-parent", f"{a}..{b}", check=False)
        for s in out(r).split():
            if s in shas and s not in found:
                found[s] = i
    return found


def fix_test_names(repo: RepoGit, u: Unit) -> set[str]:
    """``Class::name`` keys of the test functions the fix added or changed."""
    keys: set[str] = set()
    for p in u.test_files:
        a, b = _defs(repo.show(u.parent, p), p), _defs(repo.show(u.sha, p), p)
        keys |= {f"{p}::{k}" for k in b if a.get(k) != b[k]}
    return keys


def changed_functions(repo: RepoGit, u: Unit) -> set[str]:
    return {f"{p}::{q}" for p in u.src_files for q in changed_defs(repo.show(u.parent, p), repo.show(u.sha, p))}


def interaction_edges(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
    """``{k: {j, ...}}`` (symmetric): fixes k and j interact when a test that passes at B fails both with k alone reverted
    and with j alone reverted (``real_fix_check``'s ``caught``, test-case ids): that test passes only once both faults are
    fixed, so in the task its milestone needs the other fix (the prober's executed edge). Only fixes whose own revert is
    caught (``ok``) count. Test-case ids matter: two fixes that each add cases to one parametrized test do not interact, nor
    do two that each break a different subtest of one Go parent test, which fails with either (``subtests_of``)."""
    ok = {s: set(ev.get("caught") or []) for s, ev in evidence.items() if ev.get("ok")}
    need: dict[str, set[str]] = {}
    shas = sorted(ok)
    for i, a in enumerate(shas):
        for b in shas[i + 1:]:
            if any(not subtests_of(t, ok[a] | ok[b]) for t in ok[a] & ok[b]):
                need.setdefault(a, set()).add(b)
                need.setdefault(b, set()).add(a)
    return need


def order_fixes(group: Sequence[Unit], need: Mapping[str, set[str]]) -> list[Unit]:
    """History order, except that a fix another one needs comes first (Kahn's order; a cycle keeps history order), so the
    prober, which looks for prerequisites among earlier units, sees the edge."""
    order = {u.sha: i for i, u in enumerate(group)}
    shas = set(order)
    pre = {u.sha: {j for j in need.get(u.sha, set()) if j in shas} for u in group}
    out_: list[Unit] = []
    left = list(group)
    while left:
        ready = [u for u in left if not (pre[u.sha] - {x.sha for x in out_})]
        nxt = min(ready or left, key=lambda u: order[u.sha])
        out_.append(nxt)
        left.remove(nxt)
    return out_


def group_fixes(fixes: Sequence[Unit], functions: Mapping[str, set[str]], *, need: Mapping[str, set[str]] | None = None, max_size: int = 4,
                min_size: int = 2) -> list[list[Unit]]:
    """Groups of ``min_size..max_size`` fixes. First the components of the executed interaction graph (``need``; each
    such group holds an edge), filled up with fixes that share a changed function or source file with them; then the
    remaining fixes by shared function, else shared file, cut in history order. Each group is in prober order
    (``order_fixes``)."""
    need = need or {}
    order = {u.sha: i for i, u in enumerate(fixes)}
    by = {u.sha: u for u in fixes}

    def shares(u: Unit, v: Unit) -> bool:
        return bool((functions.get(u.sha, set()) & functions.get(v.sha, set())) or (set(u.src_files) & set(v.src_files)))

    def components(edges: list[tuple[str, str]], nodes: Sequence[str]) -> list[list[str]]:
        parent = {n: n for n in nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y in edges:
            parent[find(x)] = find(y)
        comps: dict[str, list[str]] = {}
        for n in nodes:
            comps.setdefault(find(n), []).append(n)
        return sorted((sorted(c, key=order.__getitem__) for c in comps.values()), key=lambda c: order[c[0]])

    used: set[str] = set()
    groups: list[list[Unit]] = []
    inter = [(k, j) for k, js in need.items() for j in js if k in by and j in by]
    in_edges = {x for e in inter for x in e}
    for comp in components(inter, sorted(in_edges, key=order.__getitem__)):
        g = comp[:max_size]
        for u in fixes:  # fill up with fixes sharing code with the group (more milestones, more chances to interact)
            if len(g) >= max_size:
                break
            if u.sha not in g and u.sha not in used and u.sha not in in_edges and any(shares(u, by[x]) for x in g):
                g.append(u.sha)
        if len(g) >= min_size:
            used |= set(g)
            groups.append(order_fixes([by[x] for x in sorted(g, key=order.__getitem__)], need))
    rest = [u for u in fixes if u.sha not in used]
    share_edges = [(u.sha, v.sha) for i, u in enumerate(rest) for v in rest[i + 1:] if shares(u, v)]
    for comp in components(share_edges, [u.sha for u in rest]):
        chunks = [comp[i: i + max_size] for i in range(0, len(comp), max_size)]
        if len(chunks) > 1 and len(chunks[-1]) < min_size and len(chunks[-2]) + len(chunks[-1]) <= max_size + 1:
            tail = chunks.pop()
            chunks[-1] = chunks[-1] + tail
        groups += [order_fixes([by[x] for x in c], need) for c in chunks if len(c) >= min_size]
    return groups


async def real_fix_check(repo: RepoGit, b: str, units: Sequence[Unit], sandbox_for: Callable[[str], Any], images: ImageBook, repo_name: str,
                         work: Path, *, full_suite: bool = False) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Per fix, executed at B: B with only the fix's source reverted runs the test files of *every* fix of the pool (as
    they are at B). ``caught_own``: tests the fix added or changed that pass at B and fail after the revert (``ok`` when
    any); ``caught``: every test of the pool's files that passes at B and fails after the revert (``interaction_edges``
    intersects them). Returns ``({sha: evidence}, world card of B)``."""
    base_b = work / "b"
    if not base_b.exists():
        ok, bad = await asyncio.to_thread(repo.build, b, [], base_b)
        if not ok:
            raise RuntimeError(f"B does not check out ({bad})")
    card = await images.ensure(repo_name, base_b)
    if not card["ok"]:
        return {u.sha: {"ok": False, "why": "the World image of B does not build"} for u in units}, card
    sb = sandbox_for(card["image"])
    files = {u.sha: [p for p in u.test_files if (base_b / p).is_file()] for u in units}
    union = sorted({p for fs in files.values() for p in fs})
    if full_suite:  # every test of B: a test no fix touched can still need two fixes (an integration test)
        union = sorted(set(union) | set(suite_targets(base_b, {})))
    at_b = await sb.run(base_b, union) if union else None
    b_res = normalize_ids(at_b.results, union) if at_b and at_b.infra == "ok" else {}
    own = {u.sha: fix_test_names(repo, u) for u in units}

    def owned(t: str, sha: str) -> bool:
        f, q = key_of_test(t)
        return f"{f}::{q}" in own[sha]

    async def one(u: Unit) -> tuple[str, dict[str, Any]]:
        if not files[u.sha]:
            return u.sha, {"ok": False, "why": "its test files are gone at B"}
        with tempfile.TemporaryDirectory(prefix="p5rb-") as td:
            tree = Path(td) / "t"
            ok, _ = await asyncio.to_thread(repo.build, b, [], tree, reverse=[u], source_only=True)
            if not ok:
                return u.sha, {"ok": False, "why": "its source revert does not apply at B"}
            tr = await sb.run(tree, union)
        if tr.infra != "ok":
            return u.sha, {"ok": False, "why": f"masked: {tr.why}"}
        res = normalize_ids(tr.results, union)
        caught = sorted(t for t in res if b_res.get(t) == "passed" and res[t] in ("failed", "error"))
        caught_own = [t for t in caught if owned(t, u.sha)]
        return u.sha, {"ok": bool(caught_own), "caught": caught, "caught_own": caught_own[:30], "own_tests": len(own[u.sha]),
                       "why": "" if caught_own else ("no own test fails after the revert" if own[u.sha] else "the fix changed no test function")}

    return dict(await asyncio.gather(*[one(u) for u in units])), card


class RebreakTask(GeneratedTask):
    generator = NAME

    def simplify(self, kind: str, *, milestones: Sequence[str] | None = None, root: Path | None = None) -> Any:
        """``locate`` / ``faulty_function``: the prompt names the faulty functions and files of the target milestone's fix
        (``windows.WindowTask._simplify_locate``); ``prerequisites``: the fixes the target's fix needs, pre-applied."""
        if kind in ("locate", "faulty_function"):
            t = self._simplify_locate(milestones=milestones)
        elif kind == "prerequisites":
            t = super().simplify("prerequisites", milestones=milestones, root=root)
        else:
            raise ValueError(f"a rebreak task simplifies by locate | faulty_function | prerequisites, not {kind!r}")
        t.__class__ = type(self)  # the same task kind (adds no state): its card and generator stay rebreak
        return t


class Rebreak:
    """The rebreak generator over local clones of training repositories (``Generator``)."""

    name = NAME

    def __init__(self, root: Path, clones: Mapping[str, Path], *, funnel: Funnel, lease: EvalLease, images: ImageBook | None = None,
                 sandboxes: Any = None, since_years: int = 6, max_size: int = 4, max_groups_per_repo: int = 3, max_tags: int = 10, windows: int = 4,
                 require_interaction: bool = True, full_suite: bool = False, log: Callable[[dict], None] | None = None):
        self.root = Path(root)
        self.funnel, self.lease, self.log = funnel, lease, log
        self.require_interaction, self.full_suite = require_interaction, full_suite
        self.images = images or ImageBook()
        from crucible.factory.generators.base import GRADE_PREFIX
        from crucible.factory.lanes.windows import sandbox_factory

        self.sandboxes = sandboxes or sandbox_factory(parallel=5, memory="1536m", name_prefix=GRADE_PREFIX)
        self.since_years, self.max_size, self.max_groups, self.max_tags, self.windows = since_years, max_size, max_groups_per_repo, max_tags, windows
        self.clones: dict[str, Path] = {}
        for name, src in clones.items():
            lease.guard_repo(name)
            self.clones[name] = derived_clone(Path(src), self.root / "clones" / name)
        self.store = WindowStore(self.root / "store" / "train", split="train")
        self._lane: GenLane | None = None
        self.records: list[SourceRecord] = []

    def lane(self) -> GenLane:
        if self._lane is None:
            self._lane = GenLane(self.store, generator=NAME, task_cls=RebreakTask, repo_dirs={k: str(v) for k, v in self.clones.items()},
                                 sandboxes=self.sandboxes)
        return self._lane

    def task(self, record_id: str, *, authored: Mapping[str, Any] | None = None) -> RebreakTask:
        return self.lane().task(record_id, authored=authored)  # type: ignore[return-value]

    async def candidates_for(self, name: str) -> list[Candidate]:
        """Per release B (newest first, at most ``max_tags``): the fixes of B's own release window and the ``windows - 1``
        windows before it, each executed at B (``real_fix_check``), grouped; a fix joins at most one group. B is the release
        that shipped the fixes (or shortly after), so their reverts still apply: a fix five years before the newest release
        rarely reverts cleanly there (the code moved on)."""
        repo_dir = self.clones[name]
        repo = RepoGit(repo_dir)
        tags = version_tags(repo_dir)
        if not tags:
            self.funnel.note(NAME, name, "S1_source", False, "no release tag")
            return []
        newest = tags[-1]
        since = f"{int(newest[2][:4]) - self.since_years}{newest[2][4:10]}"
        fixes = await asyncio.to_thread(fix_units, repo_dir, newest[1], since=since)
        window_of = await asyncio.to_thread(release_window_of, repo_dir, tags, {u.sha for u in fixes})
        by_sha = {u.sha: u for u in fixes}
        for u in fixes:
            self.funnel.note(NAME, f"{name}:{u.sha[:10]}", "S1_source", u.sha in window_of, "" if u.sha in window_of else "in no release window",
                             subject=u.subject[:120])
        used: set[str] = set()
        out_: list[Candidate] = []
        tried = 0
        for i in range(len(tags) - 1, 0, -1):
            if len(out_) >= self.max_groups or tried >= self.max_tags:
                break
            b_ref, b, b_date, b_version = tags[i]
            pool = [by_sha[s] for s, w in window_of.items() if i - self.windows < w <= i and s not in used and s in by_sha]
            pool.sort(key=lambda u: fixes.index(u))
            if len(pool) < 2:
                continue
            tried += 1
            work = self.root / "work" / name / b[:10]
            key = "|".join(u.sha for u in pool) + ("|full" if self.full_suite else "")
            cache = work / f"evidence-{hashlib.sha256(key.encode()).hexdigest()[:10]}.json"
            if cache.exists():
                evid, card = json.loads(cache.read_text())
            else:
                evid, card = await real_fix_check(repo, b, pool, self.sandboxes, self.images, name, work, full_suite=self.full_suite)
                cache.write_text(json.dumps([evid, {k: v for k, v in card.items() if k != "build_log"}], default=str))
            kept = []
            for u in pool:
                ev = evid[u.sha]
                rec = SourceRecord("real_fix", name, sha=u.sha, title=u.subject, text=u.body[:1500], created=u.date,
                                   evidence={"executed": True, "b": b, "b_ref": b_ref, **ev})
                self.lease.take(rec)
                self.records.append(rec)
                self.funnel.note(NAME, f"{name}:{u.sha[:10]}@{b_ref}", "S1_executed", ev["ok"] and rec.lease["ok"],
                                 ev.get("why") or rec.lease.get("reason", ""))
                if ev["ok"] and rec.lease["ok"]:
                    kept.append(u)
            funcs = {u.sha: changed_functions(repo, u) for u in kept}
            need = interaction_edges({u.sha: evid[u.sha] for u in kept})
            recs = {r.sha: r for r in self.records if r.repo == name and r.evidence.get("b") == b}
            for g in group_fixes(kept, funcs, need=need, max_size=self.max_size):
                if len(out_) >= self.max_groups:
                    break
                cid = f"rb-{name}-{hashlib.sha256('|'.join(u.sha for u in g).encode()).hexdigest()[:8]}"
                inter = any(need.get(u.sha, set()) & {x.sha for x in g} for u in g)
                if self.require_interaction and not inter:
                    # S6 screened cheaply: no test the group's fixes catch is caught by two of them, so the prober would find no
                    # executed edge (12 of 12 such groups came out independent in the first run); never built
                    self.funnel.note(NAME, cid, "S1_executed", False, "no executed interaction at B (no test caught by two reverts)",
                                     fixes=[u.sha[:10] for u in g])
                    continue
                used |= {u.sha for u in g}
                out_.append(Candidate(cid, NAME, name, sources=[recs[u.sha].to_json() for u in g],
                                      params={"b": b, "b_ref": b_ref, "b_date": b_date, "b_version": b_version, "fixes": [u.sha for u in g],
                                              "image_b": card.get("image"),
                                              "needs": {u.sha: sorted(need.get(u.sha, set()) & {x.sha for x in g}) for u in g}}))
        return out_

    async def candidates(self) -> list[Candidate]:
        outs = await asyncio.gather(*[self.candidates_for(n) for n in self.clones])
        return [c for cs in outs for c in cs]

    def synthesize(self, cand: Candidate) -> tuple[Window | None, dict[str, Any]]:
        """A'..B' for the candidate's fixes (see the module doc). Returns the window (None when fewer than two fixes revert
        together) and what happened (dropped fixes, B' == B).

        Tree T_k (k fixes re-applied) = B with the *source* of fixes k+1..n reverted (one 3-way chain, newest first; a fix
        whose source revert conflicts is dropped) and, in each test file they touched, the test functions they added or
        changed removed (``strip_tests``: hidden from the agent; the grader writes B's file back). Prose (changelog lines)
        is kept: reverting several fixes' changelog entries conflicts on adjacent lines."""
        repo_dir = self.clones[cand.repo]
        repo = RepoGit(repo_dir)
        b = cand.params["b"]
        fixes = [commit_meta(repo_dir, s) for s in cand.params["fixes"]]
        info: dict[str, Any] = {"dropped": [], "prose_kept": True}
        for f in fixes:
            f["files"] = changed_paths(repo_dir, f["parent"], f["sha"])
        while len(fixes) >= 2:
            steps = [(full_patch(repo_dir, f["parent"], f["sha"], [p for p in f["files"] if p.endswith(".py") and not is_test_path(p)]), True)
                     for f in reversed(fixes)]
            trees, bad = apply_chain(repo_dir, b, steps)
            if bad is None:
                break
            drop = list(reversed(fixes))[bad]
            info["dropped"].append({"sha": drop["sha"], "why": "its source revert conflicts with the fixes after it"})
            fixes = [f for f in fixes if f["sha"] != drop["sha"]]
        if len(fixes) < 2:
            return None, info
        tb = tree_of(repo_dir, b)
        src_forward = [*reversed(trees), tb]  # source of fixes k+1..n reverted, k = 0..n
        tests_b: dict[str, str] = {}
        keys: dict[str, dict[str, set[str]]] = {}  # fix sha -> test file -> changed test keys
        for f in fixes:
            u = Unit(sha=f["sha"], parent=f["parent"], merge=False, date=f["date"], subject=f["subject"], files=f["files"])
            for p in u.test_files:
                k = {q.split("::", 1)[1] for q in fix_test_names(repo, u) if q.startswith(p + "::")}
                if k and (txt := repo.show(b, p)) is not None:
                    keys.setdefault(f["sha"], {})[p] = k
                    tests_b[p] = txt
        forward = []
        for k, t in enumerate(src_forward):
            pending = fixes[k:]
            files: dict[str, str | None] = {}
            for p, txt in tests_b.items():
                ks = set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in pending]) if pending else set()
                if ks:
                    files[p] = strip_tests(txt, ks)
            forward.append(with_files(repo_dir, t, files))
        msgs = [f"rebreak start: {len(fixes)} fixes of {cand.params['b_ref']} reverted"] + [f"{f['subject']}\n\n{f['body']}" for f in fixes]
        dates = [cand.params["b_date"]] + [f["date"] for f in fixes]
        shas = write_chain(repo_dir, forward, messages=msgs, dates=dates, parent=b, ref=f"refs/crucible/rebreak/{cand.id}")
        a2, b2 = shas[0], shas[-1]
        info["b_prime_equals_b"] = tree_of(repo_dir, b2) == tb
        units = first_parent_units(repo_dir, a2, b2)
        _bodies(repo_dir, units)
        win = Window(id=cand.id, repo=cand.repo, repo_dir=str(repo_dir), kind="rebreak", a=a2, b=b2, a_ref=f"{cand.params['b_ref']}-rebreak",
                     b_ref=cand.params["b_ref"], a_date=cand.params["b_date"], b_date=cand.params["b_date"], b_version=cand.params.get("b_version"),
                     units=units)
        info["fixes"] = [{"sha": f["sha"], "subject": f["subject"], "unit": u.sha} for f, u in zip(fixes, units, strict=True)]
        return win, info

    async def build(self, cand: Candidate) -> dict[str, Any]:
        self.lease.guard_repo(cand.repo)
        if (self.store.dir(cand.id) / "window.json").exists():
            return self.store.window(cand.id)
        win, info = await asyncio.to_thread(self.synthesize, cand)
        if win is None:
            self.funnel.note(NAME, cand.id, "S3_world", False, "fewer than two fixes revert together", **info)
            return {"id": cand.id, "stage": "source", "reason": "fewer than two fixes revert together", **info}
        rec = await validate_window(win, self.store, self.images, self.sandboxes, reruns=3, min_milestones=2, mutants=True, import_tolerant=True,
                                    leak_prefix=f"{WORLD_PREFIX}leak-", log=self.log)
        if rec.get("oracle_files"):
            info["crlf_prose_dropped"] = drop_crlf_prose(rec, self.store.dir(cand.id) / "base")
        units = {u.sha: u for u in win.units}
        by_unit = {m["unit"]: m["id"] for m in rec.get("milestones") or []}
        edges = [{"fix": units[m["unit"]].subject[:100], "needs": [units[s].subject[:100] for s in m.get("requires_units") or []]}
                 for m in rec.get("milestones") or [] if m.get("requires")]
        rec[NAME] = {**info, "b": cand.params["b"], "b_ref": cand.params["b_ref"], "source_repo": cand.repo, "interacting": edges,
                     "fix_milestones": {f["sha"]: by_unit.get(f["unit"]) for f in info["fixes"]}, "sources": [s["id"] for s in cand.sources]}
        self.store.save_window(rec)
        stage = rec.get("stage")
        self.funnel.note(NAME, cand.id, "S3_world", stage not in ("windows",), "" if stage != "windows" else rec.get("reason", ""))
        if stage not in ("windows",):
            ok4 = stage in ("f2p_valid", "dependent", "mutants_valid")
            self.funnel.note(NAME, cand.id, "S4_oracle", ok4, "" if ok4 else rec.get("reason", ""), milestones=len(rec.get("milestones") or []))
            if ok4 and rec.get("stage") == "f2p_valid":  # mined, but no executed edge: S6 drops it here (no gate run needed)
                self.funnel.note(NAME, cand.id, "S6", False, rec.get("reason", ""), executed_edges=0)
        return rec

    def cleanup_work(self) -> None:
        for d in (self.root / "work").glob("*/*/b"):
            shutil.rmtree(d, ignore_errors=True)
