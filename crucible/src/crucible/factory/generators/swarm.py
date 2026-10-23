"""Swarm-built windows (swarm-pipeline/00 §2, §3, §4, §15 P5): a real start repository plus real requests; the swarm plans
a milestone DAG, builds the oracle commit by commit in a private worktree, and blind verifier writers write the hidden
tests per milestone. Then the window machinery and the same executed gates as a mined window decide (V1, V6).

**Sources (S1).** The start is a training window of store3 at its release A (its World image builds, A's suite runs).
The requests are the repository's own later changelog: the lines the release after A added to its CHANGES / CHANGELOG /
HISTORY / NEWS file (else the first-parent commit subjects of that release), split into items, each with the issue or pull
request text it cites when a GitHub cache is given (``request_records``). Each item is a ``request`` ``SourceRecord``; the
eval lease is taken before anything runs on it. The real B is never shown to any role: the planner gets the requests and
the repository at A only, and B's tree is never fetched into a session's worktree.

**The machine** (``swarm_machine``; ``factory/machine.py`` v2 with ``factory/sessions.py`` sessions and ``factory/handoff.py``
typed hand-offs carrying the full spec, every artifact committed to the family's task repo with ``informed_by=``):

| state | owner | guard |
|---|---|---|
| ``lease`` (S0) | root | no live overlapping announcement |
| ``plan`` (S2) | planner (author family, a call; ``code_links`` from A's call graph in its input) | ``plan_problems``: 2..N milestones, ids m1..mN in order, each with a requirement, the observable behaviour and its public names (dunders are public); ``requires`` only on earlier milestones; >= 1 edge whose ``why_requires`` names a public name the required milestone adds; every request id real (loops on its own findings with its previous plan) |
| ``oracle`` (S4) | oracle builder (author family, M16 session in a private clone at A, the World image, offline) | captures its commits; ``oracle_check``: one commit per milestone, subjects ``mK: ...`` in order, each changes library source, A's passing tests still pass on the final tree (regressions go back with their ids; a regression the builder cannot remove twice goes to the planner once: the plan asks for behaviour the existing suite pins) |
| ``oracle_plan`` (S4.5) | deterministic (``factory/oracleplan.py``) | every planned removal, deprecation, addition and option executed on A and on the oracle in the World (deprecation told apart from removal); A's tests that use a removed name are defects, never regression checks (``regressions`` does not count them); a defect goes back to the builder with its finding |
| ``verifier`` (S5 writer) | verifier writer (author family, another session, ``isolated_from`` the oracle) | captures ``crucible_hidden/test_<mid>.py`` per milestone from the plan, the spec and A only |
| ``review`` (S11) | reviewer (reviewer family, a call, ``different_family_from`` the verifier) | blocking findings go back to the verifier once; then they are advisory (V6: no model verdict gates alone) |
| ``build`` (S3/S4/S5/S6/S8 inputs) | deterministic | ``test_problems``; the hidden tests run on A and on the oracle (a milestone whose tests all fail on the oracle, or pass on A, goes back to the verifier with test ids and outcome kinds only, never values); the synthetic history and ``windows.validate_window`` (World, leak probes, fail-to-pass with reruns, prefix owners, the executed prober, commit-revert mutants of the oracle's own commits); no executed edge goes back to the planner |
| ``prompt`` (S7) | prompt author (author family) | ``base.author_generated`` (two-way ledger, lint, required names, version lint), up to 3 tries; every ``from X import Y`` of the hidden tests that A cannot import (run in A's World) and the request does not state gets a fixed lane sentence (``unstated_imports``) |
| ``gates`` | deterministic | ``base.deterministic_gates``: S6, S5, S8, S3 (60 reruns), S10 (60 grades) |
| ``second`` (S8 alternatives, S9) | rewriters of both families, second verifier (reviewer family) | alternatives accepted by the real tests and the shadow suite; S9 differential: no confirmed verifier-1 or spec defect |
| ``hack`` (S12) | hacker (reviewer family, an agent in the solver's World) | the attempt does not score 1, or it does and the second verifier passes every requirement it measures with no path check fired (``judge_pass``: the hacker implemented the behaviour; executed, V6) |
| ``accept`` (S17) | root | the herding guard |

**The window** (``synthesize``). The oracle's commits are fetched from the builder's worktree into a derived clone. Tree k =
the oracle's commit k with every test and protected path put back to A, plus the hidden test files of milestones 1..k
(``<suite dir>/test_swarm_<mid>.py``). The trees are written as one synthetic history under
``refs/crucible/swarm/<id>``; ``validate_window`` then runs unchanged, so a swarm-built task is a ``GeneratedTask`` and every
P0-P6 handler reads it as a mined window. Milestones are what execution says (the first unit after whose prefix a test
passes), never what the plan claimed.

**Models.** The author family is qwen: the planner on qwen3.8-flash (qwen3.7-flash planned independent series on every
candidate), the builder, verifier writer and prompt author on qwen3.7-flash; the reviewer, second verifier, rewriter and
hacker on deepseek-v4-flash. Call roles bound their thinking (``ReasoningCap``). ``SwarmRunner`` books every role's dollars
in the phase's one USD ledger; ``finish`` re-runs S9, S12 and S17 for a candidate whose window, prompt and gates are stored.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.generators.base import (
    AuthorStyle,
    Funnel,
    GeneratedTask,
    derived_clone,
    git,
    out,
    tree_of,
    with_files,
    write_chain,
)
from crucible.factory.lanes.executable import is_protected, is_test_path
from crucible.factory.sources.records import SourceRecord

NAME = "swarm_window"
HIDDEN_DIR = "crucible_hidden"  # where the verifier writer leaves its files in its own worktree
CHANGELOG_RX = re.compile(r"(^|/)(changes|changelog|history|news|release[-_]?notes)[^/]*$", re.IGNORECASE)
REF_RX = re.compile(r"(?::(?:issue|pr|gh):`|#)(\d{1,6})\b")
HEADING_RX = re.compile(r"^\s*(?:[=\-~*#^`]{3,}\s*$|#+\s|v?\d+(?:\.\d+)+\b|version\s+\d|released?\b|_?\(\w+ \d|\*{3,}|\.\.\s)", re.IGNORECASE)
SECTION_RX = re.compile(r"^\s*(features?|bug\s*fixes|fixes|additions|removals|deprecations|changes|changed|added|fixed|other|documentation|"
                        r"improvements?|internal|misc\w*|trivial\w*|breaking[\w\s]*)\s*:?\s*$", re.IGNORECASE)
BULLET_RX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
NOT_REQUEST_RX = re.compile(r"\b(typos?|docs?|documentation|readme|ci|lint(ing)?|flake8|mypy|pyright|typing|type (hints?|annotations?|coverage)|"
                            r"python\s*3\.\d+|officially|support python|tox|github actions|pre-commit|black|isort|ruff|bump|release|copyright|"
                            r"changelog|coverage|test suite|optimi[sz]ations?|released|maintainer|renovate|dependency|dependencies|github-actions|"
                            r"codspeed|benchmark|internal|trivial|docstrings?|translations?|locale|localization|spelling|doctests?)\b|actions/|setup\.py|^\[[^\]]+\]:\s*https?://",
                            re.IGNORECASE)
STYLE = AuthorStyle(
    role="author",
    brief=("You write the feature request a maintainer of this Python library would give a developer: a dependent series of changes "
           "the library should gain. You get, per change id, the planned requirement of that change (what a user observes), the public "
           "names it adds, the users' original requests it serves, and the library's README head. Write one requirement per change id, "
           "each one or two plain sentences stating the behaviour a user will see, in terms of the public API (inputs, outputs, errors, "
           "formats), never how to implement it and never where the code is. Never write file paths, module file names, test names, commit "
           "ids, issue or pull request numbers, links, version numbers, or process instructions (what tools to use or avoid). Return "
           "{opening, requirements: [{id, text}], closing}: opening is two or three sentences of context in the maintainer's voice, closing "
           "is optional, requirements has exactly one entry per change id. Names listed in a change's must_name (or in must_name_any) are "
           "public names the code must expose under exactly that name: write each one, verbatim, in the requirement of its change "
           "(must_name_any: in the one it fits)."),
    heading="What we need:")


# ---- S1: requests ------------------------------------------------------------------------------------------------------------
def changelog_lines(repo_dir: Path, a: str, b: str) -> list[str]:
    """The lines the window A..B added to the repository's changelog files, raw (issue references kept)."""
    changed = out(git(Path(repo_dir), "diff", "--name-only", "--no-renames", a, b)).splitlines()
    files = [p for p in changed if CHANGELOG_RX.search(p)][:2]
    lines: list[str] = []
    for p in files:
        diff = git(Path(repo_dir), "diff", a, b, "--", p, check=False).stdout.decode("utf-8", "replace")
        lines += [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    return lines


def split_requests(lines: Sequence[str]) -> list[str]:
    """Changelog lines -> request items: a bullet starts an item, indented or plain continuation lines join it; headings,
    version and date lines, section titles and rules end one."""
    items: list[str] = []
    cur: list[str] = []

    def flush() -> None:
        if cur:
            text = " ".join(" ".join(cur).split())
            if len(text) >= 12:
                items.append(text)
            cur.clear()

    for ln in lines:
        s = ln.rstrip()
        if not s.strip() or HEADING_RX.match(s) or SECTION_RX.match(s):
            flush()
            continue
        if BULLET_RX.match(s):
            flush()
            cur.append(BULLET_RX.sub("", s, count=1))
        elif cur:
            cur.append(s.strip())
        else:
            cur.append(s.strip())
    flush()
    return items


def is_request(item: str) -> bool:
    """A user-facing behaviour request (not a docs, CI, typing, packaging, translation or release note, and not a bare name
    left over from a sub-list: at least three words outside code spans)."""
    prose = re.sub(r"``[^`]*``|`[^`]*`", " ", item)
    return not NOT_REQUEST_RX.search(item) and len(re.findall(r"[A-Za-z]{2,}", prose)) >= 3


def request_items(repo_dir: Path, a: str, b: str, *, units: Sequence[Mapping[str, Any]] = (), limit: int = 14) -> tuple[list[str], str]:
    """``(items, source)``: the changelog's request items, else the window's commit subjects (pull request titles)."""
    items = [x for x in split_requests(changelog_lines(repo_dir, a, b)) if is_request(x)]
    if len(items) >= 2:
        return items[:limit], "changelog"
    subj = []
    for u in units:
        s = str(u.get("subject") or "")
        m = re.match(r"Merge pull request #(\d+) from \S+\s*(.*)", s)
        body = str(u.get("body") or "").split("\n")[0]
        text = (f"{body} (#{m.group(1)})" if m and body else s).strip()
        if text and is_request(text) and not text.lower().startswith(("merge", "revert", "bump", "release")):
            subj.append(text)
    return list(dict.fromkeys(subj))[:limit], "commit_subjects"


def request_records(repo_name: str, window: Mapping[str, Any], repo_dir: Path, *, gh: Any = None, owner_repo: str | None = None,
                    lease: Any = None, max_issues: int = 8) -> list[SourceRecord]:
    """One ``request`` record per item (its id ``R<n>``), with the issue or pull request text it cites (``gh``: a
    ``sources.github.GitHub``; network optional, cached). The lease is taken on every record."""
    items, source = request_items(repo_dir, window["a"], window["b"], units=window.get("units") or [])
    recs: list[SourceRecord] = []
    fetched = 0
    for i, item in enumerate(items, 1):
        refs = list(dict.fromkeys(REF_RX.findall(item)))
        extra: list[dict[str, Any]] = []
        if gh is not None and owner_repo:
            for n in refs[:2]:
                if fetched >= max_issues:
                    break
                with contextlib.suppress(Exception):
                    st, js = gh.fetch(f"https://api.github.com/repos/{owner_repo}/issues/{n}")
                    fetched += 1
                    if st == 200 and isinstance(js, Mapping):
                        extra.append({"number": int(n), "kind": "pull_request" if js.get("pull_request") else "issue", "title": str(js.get("title") or "")[:200],
                                      "body": str(js.get("body") or "")[:1500], "url": js.get("html_url")})
        text = item + "".join(f"\n\n[{e['kind']} {e['number']}] {e['title']}\n{e['body']}" for e in extra)
        rec = SourceRecord("request", repo_name, id=f"req-{repo_name}-{window['id']}-R{i}", title=item[:160], text=text[:4000],
                           created=str(window.get("b_date") or ""), url=(extra[0].get("url") or "") if extra else "",
                           evidence={"source": source, "window": window["id"], "a": window["a"], "a_ref": window.get("a_ref"), "rid": f"R{i}",
                                     "refs": refs, "issues": [{k: e[k] for k in ("number", "kind", "title")} for e in extra]},
                           by="miner:p5b-changelog")
        if lease is not None:
            lease.take(rec)
        recs.append(rec)
    return recs


# ---- S2: the plan --------------------------------------------------------------------------------------------------------------
NAME_RX = re.compile(r"``([A-Za-z_][\w.]*)(?:\(\))?``|`([A-Za-z_][\w.]*)(?:\(\))?`|\b([a-z_][a-z0-9_]{2,})\(\)|\[`([A-Za-z_][\w.]*)`\]")


def mentioned_names(text: str) -> list[str]:
    """Identifiers a request names (code spans, ``name()``, markdown links): the last dotted part."""
    outp = []
    for m in NAME_RX.finditer(text):
        n = next(g for g in m.groups() if g)
        outp.append(n.split(".")[-1])
    return list(dict.fromkeys(x for x in outp if not x.startswith("_") or x.startswith("__")))


def call_links(base: Path, names: Sequence[str], *, limit: int = 40) -> list[dict[str, str]]:
    """From the repository at A (static, ``ast``): which function or method calls which, for the names the requests mention
    (``caller`` or ``callee`` mentioned; callees are the project's own public definitions). A dependency along such a call
    chain is real: a change to the callee is observable through the caller."""
    import ast as _ast

    base = Path(base)
    want = set(names)
    defs: dict[str, str] = {}
    top: set[str] = set()  # module-level functions and classes (a method name such as ``append`` is too ambiguous to link)
    calls: dict[tuple[str, str], set[str]] = {}
    methods: dict[str, set[str]] = {}  # class -> its methods (``self.span()`` inside ``Arrow.floor`` links to ``Arrow.span``)
    for f in sorted(base.rglob("*.py")):
        rel = f.relative_to(base).as_posix()
        if is_test_path(rel) or rel.startswith((".", "docs/", "doc/")):
            continue
        try:
            tree = _ast.parse(f.read_text(errors="replace"))
        except (SyntaxError, ValueError):
            continue

        def visit(nodes: Sequence[Any], prefix: str, rel: str = rel) -> None:
            for n in nodes:
                if isinstance(n, _ast.ClassDef):
                    methods.setdefault(n.name, set()).update(x.name for x in n.body if isinstance(x, (_ast.FunctionDef, _ast.AsyncFunctionDef)))
                    visit(n.body, f"{n.name}.")
                elif isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    defs.setdefault(n.name, rel)
                    called = set()
                    for c in _ast.walk(n):
                        if isinstance(c, _ast.Call):
                            fn = c.func
                            nm = fn.id if isinstance(fn, _ast.Name) else (fn.attr if isinstance(fn, _ast.Attribute) else None)
                            if nm and nm != n.name:
                                called.add(nm)
                    calls[(prefix + n.name, rel)] = called

        visit(tree.body, "")
        top |= {n.name for n in tree.body if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))}
    links = []
    for (caller, rel), called in sorted(calls.items()):
        short = caller.split(".")[-1]
        for callee in sorted(called):
            own = callee in methods.get(caller.rsplit(".", 1)[0], set()) if "." in caller else False
            if callee not in defs or callee.startswith("_") or not (callee in want or callee in top or own):
                continue
            if short in want or callee in want:
                links.append({"caller": caller, "callee": callee, "module": rel, "both_mentioned": str(short in want and callee in want).lower()})
    links.sort(key=lambda x: x["both_mentioned"] != "true")
    return links[:limit]


PLANNER_BRIEF = (
    "You are the planner of a coding task built from a real Python library and real requests its users made. Choose 2 to {n} of the "
    "requests (or parts of them) that touch related code, and turn them into an ordered series of milestones a developer implements one "
    "after another on the repository as it is now. Make the series DEPENDENT where the work truly is: a later milestone must build on "
    "what an earlier one adds (it calls, extends, configures or reuses it), so it cannot work without it; at least one milestone must "
    "require an earlier one, and say in why_requires which public name the earlier milestone adds that this one calls or uses (name "
    "it). code_links lists, from the repository as it is now, which function calls "
    "which among the names the requests mention: a dependency along such a call chain is real (a change to the callee is what the "
    "caller's new behaviour needs), a dependency between unrelated functions is not, and the hidden tests will show which. Prefer "
    "requests that touch the same functions or build on each other; when "
    "the requests you pick are independent, split one of them into a foundation step and a step that uses it (a new PUBLIC parameter, "
    "function, method or class first, itself checkable through the public API, never a private helper; then the requested behaviour "
    "built on it). Never invent behaviour no request "
    "asks for, and never ask for docs, typing, "
    "CI or packaging. Each milestone must be checkable through the public API by a unit test (return values, raised exception types, "
    "state changes). Name every public function, class, method or parameter a milestone adds or changes, exactly as code will spell it. "
    "Return {milestones: [{id, title, requests, requirement, observable, public_names, requires, why_requires}], expected_causes, "
    "verifier_plan}: ids m1, m2, ... in order; requests lists the request ids it serves; requirement is one or two sentences in a "
    "user's voice; observable says what a test calls and what it expects; requires lists earlier milestone ids; expected_causes lists "
    "why a capable developer could still fail it; verifier_plan says how the hidden tests will check each milestone. Shape (an "
    "example of the form only, not of the content): {\"milestones\": [{\"id\": \"m1\", \"title\": \"add parse_range\", \"requests\": "
    "[\"R2\"], \"requirement\": \"parse_range('1-3') returns (1, 3) ...\", \"observable\": \"parse_range('1-3') == (1, 3); "
    "parse_range('x') raises ValueError\", \"public_names\": [\"parse_range\"], \"requires\": [], \"why_requires\": \"\"}, {\"id\": "
    "\"m2\", \"title\": \"ranges in select\", \"requests\": [\"R5\"], \"requirement\": \"select accepts a range string ...\", "
    "\"observable\": \"select(items, '2-3') == items[1:3]\", \"public_names\": [\"select\"], \"requires\": [\"m1\"], "
    "\"why_requires\": \"select parses its argument with parse_range\"}], \"expected_causes\": [\"...\"], \"verifier_plan\": \"...\"}.")


def plan_messages(spec: Mapping[str, Any], *, max_milestones: int, findings: Sequence[str] = (), previous: Any = None) -> list[dict[str, str]]:
    user = {"library": spec.get("repo_name"), "requests": spec.get("requests"), "repository_at_start": spec.get("repository"),
            "code_links": spec.get("code_links") or [],
            **({"findings_to_address": list(findings)} if findings else {}), **({"your_previous_plan": previous} if previous else {})}
    return [{"role": "system", "content": "ROLE: planner\n" + PLANNER_BRIEF.replace("{n}", str(max_milestones)) + "\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps(user, sort_keys=True)}]


IDENT_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def is_private(name: str) -> bool:
    """A leading underscore, except a dunder (``__getitem__`` is public behaviour)."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__") and len(name) > 4)


def plan_problems(plan: Any, *, request_ids: Sequence[str], max_milestones: int = 4) -> list[str]:
    """Why a plan is not well-formed (``00`` §4 S2): empty list = ok."""
    if not isinstance(plan, Mapping) or not isinstance(plan.get("milestones"), list):
        return ["the plan has no milestones list"]
    ms = plan["milestones"]
    probs: list[str] = []
    if not 2 <= len(ms) <= max_milestones:
        probs.append(f"{len(ms)} milestones: need 2 to {max_milestones}")
    ids: list[str] = []
    edges = 0
    for i, m in enumerate(ms, 1):
        if not isinstance(m, Mapping):
            probs.append(f"milestone {i} is not an object")
            continue
        mid = str(m.get("id") or "")
        if mid != f"m{i}":
            probs.append(f"milestone {i} has id {mid!r}: ids are m1, m2, ... in order")
        for k in ("title", "requirement", "observable"):
            if not str(m.get(k) or "").strip():
                probs.append(f"{mid or i}: {k} is empty")
        names = m.get("public_names")
        if not isinstance(names, list) or not names:
            probs.append(f"{mid or i}: public_names must list the public names it adds or changes")
        else:
            bad = [n for n in names if not isinstance(n, str) or not IDENT_RX.match(n) or is_private(n.split(".")[-1])]
            if bad:
                probs.append(f"{mid or i}: public_names {bad} are not public Python identifiers")
        reqs = m.get("requests")
        if not isinstance(reqs, list) or not reqs:
            probs.append(f"{mid or i}: requests must list the request ids it serves")
        else:
            unknown = [r for r in reqs if r not in request_ids]
            if unknown:
                probs.append(f"{mid or i}: unknown request ids {unknown}")
        req = m.get("requires") or []
        if not isinstance(req, list):
            probs.append(f"{mid or i}: requires must be a list")
            req = []
        for r in req:
            if r not in ids:
                probs.append(f"{mid or i}: requires {r!r}, which is not an earlier milestone")
        why = str(m.get("why_requires") or "")
        if req and not why.strip():
            probs.append(f"{mid or i}: why_requires is empty")
        elif req:  # a dependency is a code path: it names what the earlier milestone adds, used by this one
            earlier = {str(n).split(".")[-1] for x in ms[: i - 1] if isinstance(x, Mapping) and x.get("id") in req for n in x.get("public_names") or []}
            if earlier and not any(re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", why) for n in earlier):
                probs.append(f"{mid or i}: why_requires must name what the required milestone adds ({sorted(earlier)}) and how this one uses it")
        edges += len([r for r in req if r in ids])
        ids.append(mid)
    if len(set(ids)) != len(ids):
        probs.append("milestone ids repeat")
    if not edges:
        probs.append("no milestone requires an earlier one: the series must be dependent")
    return probs


def renumber(plan: Any) -> Any:
    """A plan whose milestone ids are unique but not ``m1..mN`` in order gets them, ``requires`` remapped (a naming slip, not a
    planning error); anything else is returned as it is."""
    if not isinstance(plan, Mapping) or not isinstance(plan.get("milestones"), list) or not all(isinstance(m, Mapping) for m in plan["milestones"]):
        return plan
    ids = [str(m.get("id") or "") for m in plan["milestones"]]
    if len(set(ids)) != len(ids) or not all(ids) or ids == [f"m{i}" for i in range(1, len(ids) + 1)]:
        return plan
    new = {old: f"m{i}" for i, old in enumerate(ids, 1)}
    ms = [{**m, "id": new[str(m["id"])], "requires": [new.get(str(r), str(r)) for r in (m.get("requires") or [])] if isinstance(m.get("requires"), list)
           else m.get("requires")} for m in plan["milestones"]]
    return {**plan, "milestones": ms}


def normalize_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    ms = [{"id": str(m["id"]), "title": " ".join(str(m["title"]).split())[:120], "text": " ".join(str(m["requirement"]).split()),
           "requests": list(m.get("requests") or []),
           "requirement": " ".join(str(m["requirement"]).split()), "observable": " ".join(str(m["observable"]).split()),
           "public_names": [str(n) for n in m.get("public_names") or []], "requires": [str(r) for r in m.get("requires") or []],
           "why_requires": str(m.get("why_requires") or "")} for m in plan["milestones"]]
    return {"milestones": ms, "expected_causes": list(plan.get("expected_causes") or [])[:8], "verifier_plan": str(plan.get("verifier_plan") or "")[:2000]}


# ---- S4: the oracle --------------------------------------------------------------------------------------------------------------
ORACLE_BRIEF = (
    "You are the oracle builder. The hand-off's `plan` port lists milestones m1..mN in order, each with the behaviour users must "
    "observe and the public names it adds; the spec holds the users' original requests. Implement the milestones in the repository, "
    "one at a time and in order. After each milestone: run the existing tests ({test_cmd}) and keep every test that passed before "
    "passing; then commit exactly that milestone's change with `git add -A && git commit -q -m \"mK: <title>\"` (K its number). The "
    "result must be exactly one commit per milestone, in order, each changing library source. Spell every public name exactly as the "
    "plan does. Never create or change test files (tests, conftest.py), packaging or CI files: someone else writes the tests; you may "
    "try your code with `python -c ...` snippets. Your kernel and the shell run inside the repository's sandbox (offline); `open()` and "
    "`subprocess` work on the repository path directly. When all N commits exist (check `git log --oneline base..HEAD`), finish.")

VERIFIER_BRIEF = (
    "You are the verifier writer. You never see the implementation: it does not exist in your repository, which is the library as it "
    "is now. The hand-off's `plan` port lists milestones m1..mN with the behaviour users must observe and the public names each adds; "
    "the spec holds the users' original requests. For every milestone write one pytest file `" + HIDDEN_DIR + "/test_<id>.py` (for "
    "example `" + HIDDEN_DIR + "/test_m1.py`) with 2 to 6 test functions that pass exactly when that milestone's behaviour is "
    "implemented as the plan states. Rules: import the library inside each test function (never at module top level), so a missing name "
    "fails only its own test; use only public names the plan states or the library already has; check observable behaviour (return "
    "values, raised exception types, state changes); when the plan does not fix an exact text, assert properties instead of whole "
    "strings; a later milestone's tests may use what earlier milestones add, never what later ones add; never read source files, "
    "inspect code objects, use private names, the network, the clock or randomness without a fixed seed. Run your files with `python -m "
    "pytest -q " + HIDDEN_DIR + "` in the sandbox: on this repository every test you write must fail or error (the behaviour does not "
    "exist yet); a test that passes now checks nothing new, so change it. Write only under `" + HIDDEN_DIR + "/`. Then finish.")


def oracle_problems(subjects: Sequence[str], plan: Mapping[str, Any]) -> list[str]:
    ids = [m["id"] for m in plan["milestones"]]
    probs = []
    if len(subjects) != len(ids):
        probs.append(f"{len(subjects)} commits for {len(ids)} milestones: exactly one commit per milestone, in order")
    for k, (s, mid) in enumerate(zip(subjects, ids, strict=False), 1):
        if not re.match(rf"^\s*{re.escape(mid)}\b", s):
            probs.append(f"commit {k} is {s[:60]!r}: its subject must start with '{mid}:'")
    return probs


# ---- S5: the hidden tests --------------------------------------------------------------------------------------------------------
STRUCTURE_RX = re.compile(r"\b(getsource|inspect\.|__code__|co_varnames|ast\.parse|dis\.|open\([^)]*\.py)")


def test_file(mid: str) -> str:
    return f"test_{mid}.py"


def hidden_tests(captured: Path, plan: Mapping[str, Any]) -> dict[str, str]:
    """``{milestone id: file text}`` from the verifier's captured directory."""
    outp = {}
    for m in plan["milestones"]:
        f = Path(captured) / test_file(m["id"])
        if f.is_file():
            outp[m["id"]] = f.read_text(errors="replace")
    return outp


def test_problems(tests: Mapping[str, str], plan: Mapping[str, Any]) -> list[str]:
    """Static checks before anything runs: a file per milestone, it parses, it defines tests, no source-structure checks."""
    probs = []
    for m in plan["milestones"]:
        mid = m["id"]
        text = tests.get(mid)
        if text is None:
            probs.append(f"{mid}: no file {HIDDEN_DIR}/{test_file(mid)}")
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            probs.append(f"{mid}: {test_file(mid)} does not parse ({e.msg}, line {e.lineno})")
            continue
        n = sum(1 for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name.startswith("test"))
        if n == 0:
            probs.append(f"{mid}: {test_file(mid)} defines no test function")
        if STRUCTURE_RX.search(text):
            probs.append(f"{mid}: {test_file(mid)} reads source or code structure ({STRUCTURE_RX.search(text).group(0)}): test behaviour")  # type: ignore[union-attr]
    return probs


def suite_dir(base: Path) -> str:
    """Where the repository keeps its tests (the most common directory of its suite files), ``tests`` when none."""
    from collections import Counter

    from crucible.factory.lanes.windows import suite_targets

    dirs = Counter(str(Path(t).parent) for t in suite_targets(Path(base), {}))
    best = [d for d, _ in dirs.most_common() if d not in (".", "")]
    return best[0] if best else "tests"


def hidden_path(base: Path, mid: str) -> str:
    return f"{suite_dir(base)}/test_swarm_{mid}.py"


def outcome_findings(at_a: Mapping[str, str], at_b: Mapping[str, str], plan: Mapping[str, Any], base: Path) -> tuple[list[str], dict[str, Any]]:
    """Per milestone: tests that pass on A (check nothing new) and tests that fail on the oracle. Findings name test ids and
    the outcome kind only (``failed`` / ``error``), never a value the oracle produced (V2). A milestone with no test that
    fails on A and passes on the oracle is a finding."""
    findings, per = [], {}
    for m in plan["milestones"]:
        f = hidden_path(base, m["id"])
        ids = sorted({t for t in [*at_a, *at_b] if t.split("::", 1)[0] == f})
        f2p = [t for t in ids if at_a.get(t) != "passed" and at_b.get(t) == "passed"]
        on_a = [t for t in ids if at_a.get(t) == "passed"]
        bad_b = [t for t in ids if at_b.get(t) != "passed"]
        per[m["id"]] = {"tests": len(ids), "f2p": len(f2p), "pass_on_start": len(on_a), "fail_on_reference": len(bad_b)}
        name = lambda t: t.split("::", 1)[1] if "::" in t else t
        if on_a:
            findings.append(f"{m['id']}: these tests already pass on the current repository, so they check nothing new: {[name(t) for t in on_a][:6]}")
        if bad_b:
            findings.append(f"{m['id']}: these tests fail on a reference implementation of the plan ({', '.join(sorted({at_b.get(t, 'missing') for t in bad_b}))}): "
                            f"{[name(t) for t in bad_b][:6]}; check them against the plan's wording (an unstated exact format, a name or argument the plan "
                            f"never gave, an assumption about behaviour the plan does not fix)")
        if not f2p:
            findings.append(f"{m['id']}: no test fails now and passes once the milestone is implemented")
    return findings, per


# ---- S7: import paths the hidden tests use ----------------------------------------------------------------------------------
def test_imports(tests: Mapping[str, str]) -> list[tuple[str, str]]:
    """``(module, name)`` of every ``from <module> import <name>`` in the hidden test files (any depth), first seen first."""
    outp: list[tuple[str, str]] = []
    for text in tests.values():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and not n.level:
                outp += [(n.module, a.name) for a in n.names if a.name != "*"]
    return list(dict.fromkeys(outp))


IMPORT_PROBE = """import importlib, json, sys
pairs = json.loads(sys.argv[1]); bad = []
for mod, name in pairs:
    try:
        m = importlib.import_module(mod)
        getattr(m, name)
    except Exception:
        try:
            importlib.import_module(mod + "." + name)
        except Exception:
            bad.append([mod, name])
print("CRUCIBLE_MISSING=" + json.dumps(bad))
"""


def unstated_imports(missing: Sequence[Sequence[str]], prompt: str) -> list[str]:
    """``from <module> import <name>`` statements for the pairs A cannot import whose import path the request never states
    (the name alone is not enough: P5b's funcy tests imported ``SkipMemoization`` from the package root; the request named the
    class, not that the root exports it)."""
    outp = []
    for mod, name in missing:
        if not any(x in prompt for x in (f"from {mod} import {name}", f"{mod}.{name}", f"`{mod}` exports `{name}`")):
            outp.append(f"from {mod} import {name}")
    return outp


def state_imports(prompt: str, statements: Sequence[str]) -> str:
    """The request with one fixed lane sentence stating the import paths, before the harness epilogue."""
    from crucible.factory.lanes.windows import EPILOGUE

    if not statements:
        return prompt
    note = "These names must be importable exactly as written: " + "; ".join(f"`{x}`" for x in statements) + ".\n\n"
    i = prompt.find(EPILOGUE[:40])
    return prompt[:i] + note + prompt[i:] if i >= 0 else prompt.rstrip() + "\n\n" + note


# ---- the synthetic window -------------------------------------------------------------------------------------------------------
def synthesize(clone: Path, a: str, oracle_shas: Sequence[str], tests: Mapping[str, str], plan: Mapping[str, Any], base: Path, *, cid: str,
               a_ref: str, a_date: str, repo: str) -> tuple[Any, dict[str, Any]]:
    """The window A..B' of the oracle's commits (module doc): tree k = oracle commit k with test and protected paths back at
    A, plus the hidden files of milestones 1..k. Returns ``(Window, info)``."""
    from crucible.factory.lanes.windows_git import Window, _bodies, first_parent_units

    clone = Path(clone)
    ms = plan["milestones"]
    if len(oracle_shas) != len(ms):
        raise ValueError(f"{len(oracle_shas)} oracle commits for {len(ms)} milestones")
    trees, reset_paths = [], set()
    for k, sha in enumerate(oracle_shas):
        files: dict[str, str | None] = {}
        for p in [ln for ln in out(git(clone, "diff", "--name-only", "--no-renames", a, sha)).splitlines() if ln]:
            if is_test_path(p) or is_protected(p):
                txt = git(clone, "show", f"{a}:{p}", check=False)
                files[p] = txt.stdout.decode("utf-8", "replace") if txt.returncode == 0 else None
                reset_paths.add(p)
        for m in ms[: k + 1]:
            files[hidden_path(base, m["id"])] = tests[m["id"]]
        trees.append(with_files(clone, tree_of(clone, sha), files))
    msgs = [f"{m['id']}: {m['title']}\n\n{m['requirement']}" for m in ms]
    date = a_date or "2020-01-01T00:00:00+00:00"
    shas = write_chain(clone, trees, messages=msgs, dates=[date] * len(trees), parent=a, ref=f"refs/crucible/swarm/{cid}")
    units = first_parent_units(clone, a, shas[-1])
    _bodies(clone, units)
    win = Window(id=cid, repo=repo, repo_dir=str(clone), kind="swarm", a=a, b=shas[-1], a_ref=a_ref, b_ref=f"{a_ref}-swarm", a_date=a_date,
                 b_date=a_date, b_version=None, units=units)
    return win, {"chain": shas, "unit_of": {m["id"]: u.sha for m, u in zip(ms, units, strict=True)}, "reset_paths": sorted(reset_paths),
                 "hidden_files": {m["id"]: hidden_path(base, m["id"]) for m in ms}}


class SwarmTask(GeneratedTask):
    generator = NAME

    def simplify(self, kind: str, *, milestones: Sequence[str] | None = None, root: Path | None = None) -> Any:
        """``prerequisites``: the target's executed prerequisite commits pre-applied; ``locate``: the prompt names the files
        and functions of the target's commit (as for a mined window)."""
        if kind == "locate":
            t = self._simplify_locate(milestones=milestones)
        elif kind == "prerequisites":
            t = super().simplify("prerequisites", milestones=milestones, root=root)
        else:
            raise ValueError(f"a swarm-built window simplifies by prerequisites | locate, not {kind!r}")
        t.__class__ = type(self)
        return t


# ---- the session sandbox: the World image at A ----------------------------------------------------------------------------------
def world_sandbox_class(prefix: str) -> type:
    """``sessions.Sandbox`` in the window's World image (A's dependencies, pytest), named ``<prefix>session-*``, with the
    worktree first on ``PYTHONPATH`` (``src/`` layout too), as the solver's World runs it."""
    import uuid

    from crucible.factory.machine import InfraError
    from crucible.factory.sessions import HANDOFF_MOUNT, Sandbox, docker, docker_infra_kind

    class WorldSandbox(Sandbox):
        async def start(self) -> str:
            name = f"{prefix}session-{uuid.uuid4().hex[:12]}"
            ws, ho = str(self.workspace.resolve()), str(self.handoff_dir.resolve())
            r = await asyncio.to_thread(docker, "run", "-d", "--name", name, "--network", "none", "--memory", self.memory, "--cpus", self.cpus,
                                        "--pids-limit", "512", "-v", f"{ws}:{ws}", "-v", f"{ho}:{HANDOFF_MOUNT}:ro", "-w", ws, "-e", "HOME=/tmp",
                                        "-e", f"PYTHONPATH={ws}/src:{ws}", "--label", "crucible.p5b=session", self.image, "sleep", "infinity")
            if r.returncode != 0:
                await asyncio.to_thread(docker, "rm", "-f", name, timeout=60)
                raise InfraError(f"the session sandbox did not start: {r.stderr.strip()[-400:]}", docker_infra_kind(r.stderr))
            self.name = name
            return name

    return WorldSandbox


# ---- the machine -----------------------------------------------------------------------------------------------------------------
# the author family is qwen throughout; the planner is its stronger flash model: qwen3.7-flash planned independent series on 4 of 4
# candidates in 4 tries each (P5b run), and a plan with no real dependency fails S6 after a full oracle and verifier build
ROLE_MODELS = {"planner": "qwen/qwen3.8-flash", "oracle_builder": "qwen/qwen3.7-flash", "verifier_writer": "qwen/qwen3.7-flash",
               "author": "qwen/qwen3.7-flash", "reviewer": "deepseek/deepseek-v4-flash", "hacker": "deepseek/deepseek-v4-flash",
               "second_verifier": "deepseek/deepseek-v4-flash"}


# states whose handlers read the build through ``SwarmServices`` instead of ports: their provenance includes it (and so the
# planner's, the oracle builder's and the verifier writer's families: factory.provenance)
SERVICE_READS: dict[str, tuple[str, ...]] = {"prompt": ("build",), "gates": ("build", "prompt"), "second": ("build", "prompt"), "hack": ("build", "prompt")}


def swarm_machine(*, budget_usd: float = 2.0, oracle_turns: int = 150, verifier_turns: int = 150) -> dict[str, Any]:
    j = "json"
    v = [{"name": "verdict", "type": j}]
    states = [
        {"id": "lease", "entry": True, "fn": "lease", "outputs": [{"name": "lease", "type": j}]},
        {"id": "plan", "role": "planner", "handler": "plan", "max_entries": 5, "inputs": [{"name": "spec", "type": j, "from": "params.spec"}],
         "outputs": [{"name": "plan", "type": j}, *v]},
        {"id": "oracle", "role": "oracle_builder", "canary": True, "mutates": True, "recovery": "reconcile", "retries": 1, "max_entries": 5,
         "inputs": [{"name": "plan", "type": j, "from": "plan.plan"}],
         "session": {"brief": "(set per run)", "captures": [{"output": "oracle", "kind": "commits"}], "max_turns": oracle_turns, "potential": "oracle_commits"},
         "outputs": [{"name": "oracle", "type": "artifact"}]},
        {"id": "oracle_check", "fn": "oracle_check", "max_entries": 5, "inputs": [{"name": "oracle", "type": "artifact", "from": "oracle.oracle"},
                                                                                  {"name": "plan", "type": j, "from": "plan.plan"}], "outputs": v},
        {"id": "oracle_plan", "fn": "oracle_plan", "max_entries": 5, "inputs": [{"name": "check", "type": j, "from": "oracle_check.verdict"},
                                                                                {"name": "plan", "type": j, "from": "plan.plan"}], "outputs": v},
        {"id": "verifier", "role": "verifier_writer", "isolated_from": ["oracle", "oracle_check", "oracle_plan"], "mutates": True, "recovery": "reset", "retries": 1,
         "max_entries": 6, "inputs": [{"name": "plan", "type": j, "from": "plan.plan"}],
         "session": {"brief": "(set per run)", "captures": [{"output": "tests", "kind": "dir", "path": HIDDEN_DIR}], "max_turns": verifier_turns,
                     "potential": "hidden_tests"},
         "outputs": [{"name": "tests", "type": "artifact"}]},
        {"id": "review", "role": "reviewer", "handler": "review", "different_family_from": ["verifier"], "max_entries": 6,
         "inputs": [{"name": "tests", "type": "artifact", "from": "verifier.tests"}, {"name": "plan", "type": j, "from": "plan.plan"}], "outputs": v},
        {"id": "build", "fn": "build", "max_entries": 5, "inputs": [{"name": "tests", "type": "artifact", "from": "verifier.tests"},
                                                                    {"name": "oracle", "type": "artifact", "from": "oracle.oracle"},
                                                                    {"name": "check", "type": j, "from": "oracle_check.verdict"},
                                                                    {"name": "plan", "type": j, "from": "plan.plan"}], "outputs": v},
        {"id": "prompt", "role": "author", "handler": "prompt", "outputs": v},
        {"id": "gates", "fn": "gates", "outputs": v},
        {"id": "second", "role": "second_verifier", "handler": "second", "outputs": v},
        {"id": "hack", "role": "hacker", "handler": "hack", "different_family_from": ["verifier", "oracle"], "outputs": v},
        {"id": "accept", "fn": "accept", "inputs": [{"name": "lease", "type": j, "from": "lease.lease"}, {"name": "parent", "type": "text", "from": "params.parent"}],
         "outputs": [{"name": "accept", "type": j}]},
        {"id": "accepted", "terminal": "done"}, {"id": "dropped", "terminal": "rejected"}, {"id": "abstained", "terminal": "rejected"},
        {"id": "herded", "terminal": "rejected"}]
    t = [
        {"from": "lease", "to": "plan", "when": "lease.ok == true"}, {"from": "lease", "to": "abstained", "when": "lease.ok == false"},
        {"from": "plan", "to": "oracle", "when": "verdict.ok == true"},
        {"from": "plan", "to": "plan", "when": "verdict.ok == false and verdict.retry == true", "carry": "verdict.findings"},
        {"from": "plan", "to": "dropped", "when": "verdict.ok == false"},
        {"from": "oracle", "to": "oracle_check"},
        {"from": "oracle_check", "to": "oracle_plan", "when": "verdict.ok == true"},
        {"from": "oracle_plan", "to": "verifier", "when": "verdict.ok == true"},
        {"from": "oracle_plan", "to": "oracle", "when": "verdict.ok == false and verdict.retry == true", "carry": "verdict.findings"},
        {"from": "oracle_plan", "to": "plan", "when": 'verdict.ok == false and verdict.owner == "planner"', "carry": "verdict.findings"},
        {"from": "oracle_plan", "to": "dropped", "when": "verdict.ok == false"},
        {"from": "oracle_check", "to": "oracle", "when": "verdict.ok == false and verdict.retry == true", "carry": "verdict.findings"},
        {"from": "oracle_check", "to": "plan", "when": 'verdict.ok == false and verdict.owner == "planner"', "carry": "verdict.findings"},
        {"from": "oracle_check", "to": "dropped", "when": "verdict.ok == false"},
        {"from": "verifier", "to": "review"},
        {"from": "review", "to": "build", "when": "verdict.ok == true"},
        {"from": "review", "to": "verifier", "when": "verdict.ok == false", "carry": "verdict.findings"},
        {"from": "build", "to": "prompt", "when": "verdict.ok == true"},
        {"from": "build", "to": "verifier", "when": 'verdict.ok == false and verdict.owner == "verifier"', "carry": "verdict.findings"},
        {"from": "build", "to": "plan", "when": 'verdict.ok == false and verdict.owner == "planner"', "carry": "verdict.findings"},
        {"from": "build", "to": "dropped", "when": "verdict.ok == false"},
        {"from": "prompt", "to": "gates", "when": "verdict.ok == true"}, {"from": "prompt", "to": "dropped", "when": "verdict.ok == false"},
        {"from": "gates", "to": "second", "when": "verdict.ok == true"}, {"from": "gates", "to": "dropped", "when": "verdict.ok == false"},
        {"from": "second", "to": "hack", "when": "verdict.ok == true"}, {"from": "second", "to": "dropped", "when": "verdict.ok == false"},
        {"from": "hack", "to": "accept", "when": "verdict.ok == true"}, {"from": "hack", "to": "dropped", "when": "verdict.ok == false"},
        {"from": "accept", "to": "accepted", "when": "accept.ok == true"}, {"from": "accept", "to": "herded", "when": "accept.ok == false"}]
    return {"run": {"max_transitions": 60, "budget_usd": budget_usd}, "states": states, "transitions": t}


def verdict(ok: bool, findings: Sequence[str] = (), *, owner: str = "", retry: bool = False, **evidence: Any) -> Any:
    from crucible.factory.machine import Settle

    v = {"ok": bool(ok), "findings": list(findings)[:30], "owner": owner, "retry": bool(retry), **json.loads(json.dumps(evidence, default=str))}
    return Settle({"verdict": v}, list(findings)[:30])


@dataclass
class SwarmCandidate:
    id: str
    repo: str  # local clone name (a training repository)
    window: dict[str, Any]  # the store3 window record whose start A seeds it
    requests: list[dict[str, Any]]  # SourceRecord JSON
    image: str = ""

    @property
    def a(self) -> str:
        return str(self.window["a"])


@dataclass
class SwarmServices:
    """What the state functions read: the candidate, the derived clone, the generator's store and lane, the role clients,
    and the callables of the deterministic and paid stages (tests inject fakes)."""

    cand: SwarmCandidate
    clone: Path
    work: Path
    funnel: Funnel
    max_milestones: int = 4
    clients: dict[str, Any] = field(default_factory=dict)
    build_fn: Callable[..., Any] | None = None
    prompt_fn: Callable[..., Any] | None = None
    gates_fn: Callable[..., Any] | None = None
    second_fn: Callable[..., Any] | None = None
    hack_fn: Callable[..., Any] | None = None
    oracle_fn: Callable[..., Any] | None = None  # (shas, plan) -> findings: the S4 regression check
    oracle_plan_fn: Callable[..., Any] | None = None  # (shas, plan) -> ``oracleplan.check`` result: S4.5, the oracle against its plan
    review_fn: Callable[..., Any] | None = None
    pool: Any = None
    results: dict[str, Any] = field(default_factory=dict)

    def note(self, stage: str, ok: bool, reason: str = "", **ev: Any) -> None:
        self.funnel.note(NAME, self.cand.id, stage, ok, reason, **ev)


def _spec_of(ctx: Any) -> dict[str, Any]:
    return dict(ctx.params.get("spec") or {})


async def plan_state(ctx: Any) -> Any:
    from crucible.factory.lanes.executable import call_supported

    svc: SwarmServices = ctx.services
    spec = _spec_of(ctx)
    rids = [r["id"] for r in spec.get("requests") or []]
    raws = svc.results.setdefault("plan_raw", [])
    msgs = plan_messages(spec, max_milestones=svc.max_milestones, findings=ctx.findings, previous=raws[-1] if raws and ctx.findings else None)
    try:
        # the runner's planner client bounds the thinking (``ReasoningCap``: qwen3.8-flash thought through 24k tokens, empty answer)
        got = await call_supported(svc.clients["planner"].complete_json, msgs, max_tokens=32000, temperature=0.3 + 0.2 * (ctx.entry - 1),
                                   sample_key=f"p5b-plan-{svc.cand.id}-{ctx.entry}.{ctx.attempt}")
    except Exception as e:  # noqa: BLE001 - a malformed reply is the planner's finding, retried
        why = [f"the planner's reply was not a JSON plan: {type(e).__name__}: {e}"[:300]]
        svc.note("S2_plan", False, why[0], entry=ctx.entry)
        from crucible.factory.machine import Settle

        return Settle({"plan": {}, "verdict": {"ok": False, "findings": why, "owner": "planner", "retry": ctx.entry < 4}}, why)
    raws.append(got)
    with contextlib.suppress(OSError):
        svc.work.mkdir(parents=True, exist_ok=True)
        (svc.work / f"plan_raw_{ctx.entry}.{ctx.attempt}.json").write_text(json.dumps(got, indent=1, default=str))
    got = renumber(got)
    probs = plan_problems(got, request_ids=rids, max_milestones=svc.max_milestones)
    if probs:
        svc.note("S2_plan", False, "; ".join(probs)[:300], entry=ctx.entry)
        from crucible.factory.machine import Settle

        return Settle({"plan": {}, "verdict": {"ok": False, "findings": probs[:20], "owner": "planner", "retry": ctx.entry < 4}}, probs[:20])
    plan = normalize_plan(got)
    svc.results["plan"] = plan
    svc.note("S2_plan", True, milestones=len(plan["milestones"]), edges=sum(len(m["requires"]) for m in plan["milestones"]))
    from crucible.factory.machine import Settle

    return Settle({"plan": plan, "verdict": {"ok": True, "findings": [], "owner": "", "retry": False}})


async def oracle_check_state(ctx: Any) -> Any:
    """S4's guard: the commits (count, subjects, source), fetched into the derived clone; the regression check."""
    svc: SwarmServices = ctx.services
    plan = ctx.inputs["plan"]
    o = ctx.inputs["oracle"]
    probs = oracle_problems(list(o.get("subjects") or []), plan)
    shas: list[str] = []
    if not probs:
        ws = svc.pool.workspace("oracle").path if svc.pool is not None else Path(o.get("workspace") or "")
        r = git(svc.clone, "fetch", "-q", "--no-tags", str(ws), str(o["head"]), check=False)
        if r.returncode != 0:
            probs.append(f"the oracle's commits could not be fetched: {r.stderr.decode('utf-8', 'replace')[-200:]}")
        else:
            shas = out(git(svc.clone, "rev-list", "--reverse", "--first-parent", f"{svc.cand.a}..{o['head']}")).split()
            for k, sha in enumerate(shas, 1):
                parent = out(git(svc.clone, "rev-parse", f"{sha}^"))
                src = [p for p in out(git(svc.clone, "diff", "--name-only", parent, sha)).splitlines() if p.endswith(".py") and not is_test_path(p)
                       and not is_protected(p)]
                if not src:
                    probs.append(f"commit {k} changes no library source file")
    regress: list[str] = []
    if not probs and svc.oracle_fn is not None:
        regress = list(await svc.oracle_fn(shas, plan))
    if probs or regress:
        svc.note("S4_oracle", False, "; ".join(probs + regress)[:300], entry=ctx.entry)
        if regress and not probs and ctx.entry >= 2 and not svc.results.get("regression_to_planner"):
            # the builder could not keep the existing tests passing twice: the plan asks for behaviour they pin (a requested
            # change of behaviour the existing suite contradicts); the planner chooses again, once
            svc.results["regression_to_planner"] = True
            return verdict(False, [f"the planned behaviour contradicts existing tests of the repository ({r.split(': ', 1)[-1][:300]}); "
                                   "plan requests compatible with the existing tests" for r in regress], owner="planner")
        return verdict(False, probs + regress, owner="oracle_builder", retry=ctx.entry < 3)
    svc.results["oracle_shas"] = shas
    svc.note("S4_oracle", True, commits=len(shas))
    return verdict(True, shas=shas)


async def oracle_plan_state(ctx: Any) -> Any:
    """S4.5 (``factory/oracleplan.py``): every planned removal, deprecation, addition and option executed on A and on the
    oracle, and the regression tests that contradict a planned removal. A defect goes back to the oracle builder with its
    finding (twice), a plan probe that cannot hold on A to the planner; no runner (scripted tests) passes through."""
    svc: SwarmServices = ctx.services
    if svc.oracle_plan_fn is None:
        return verdict(True, skipped="no oracle-plan runner")
    shas = list((ctx.inputs.get("check") or {}).get("shas") or svc.results.get("oracle_shas") or [])
    res = await svc.oracle_plan_fn(shas, ctx.inputs["plan"])
    svc.results["oracle_plan"] = {k: res.get(k) for k in ("ok", "infra", "findings", "defects", "drop_from_regression", "probes")}
    if res.get("ok"):
        svc.note("S4_5_oracle_plan", True, probes=len(res.get("probes") or []), contradicting=len(res.get("drop_from_regression") or []))
        return verdict(True, probes=len(res.get("probes") or []), drop_from_regression=list(res.get("drop_from_regression") or [])[:200])
    svc.note("S4_5_oracle_plan", False, "; ".join(res.get("findings") or [])[:300], entry=ctx.entry)
    if res.get("infra"):
        return verdict(False, list(res.get("findings") or []), owner="")
    owner = res.get("owner") or "oracle_builder"
    return verdict(False, list(res.get("findings") or []), owner=owner, retry=owner == "oracle_builder" and ctx.entry < 3)


REVIEW_BRIEF = (
    "You review hidden tests a verifier wrote for a coding task before any solution exists. You get the plan (milestones with the "
    "behaviour users must observe and their public names), the users' requests and each milestone's test file. You never see an "
    "implementation. List the BLOCKING problems only: a test that checks something the plan never asks for; an exact text or format the "
    "plan does not fix; a public name or argument the plan never gives; a test of milestone K that relies on a later milestone; a test "
    "that reads source code, private names, the clock, randomness or the network; a test that cannot fail. Return {blocking: [\"<test "
    "or file>: <problem> -> <fix>\"], advisory: [...]}; an empty blocking list approves the tests.")


async def review_state(ctx: Any) -> Any:
    """S11 on the tests: the reviewer family's blocking findings go back to the verifier on the first entry only; later they
    are advisory (V6: a model verdict never gates on its own; the executed gates decide)."""
    from crucible.factory.lanes.executable import call_supported

    svc: SwarmServices = ctx.services
    root = Path(ctx.artifact_root or ".")
    tests = hidden_tests(root / ctx.inputs["tests"]["path"], ctx.inputs["plan"])
    probs = test_problems(tests, ctx.inputs["plan"])
    if probs:  # deterministic: always blocking while entries last
        svc.note("S5_verifier", False, "; ".join(probs)[:300], entry=ctx.entry)
        return verdict(False, probs, owner="verifier") if ctx.entry < 3 else verdict(True, [], advisory=probs, static_problems=probs)
    if svc.review_fn is not None:
        got = await svc.review_fn(tests, ctx.inputs["plan"])
    else:
        payload = {"plan": ctx.inputs["plan"], "requests": _spec_of(ctx).get("requests"), "tests": {f"test_{k}.py": v for k, v in tests.items()}}
        msgs = [{"role": "system", "content": "ROLE: reviewer\n" + REVIEW_BRIEF + "\nAnswer with one JSON object only."},
                {"role": "user", "content": json.dumps(payload, sort_keys=True)}]
        try:
            got = await call_supported(svc.clients["reviewer"].complete_json, msgs, max_tokens=32000, temperature=0.0)
        except Exception as e:  # noqa: BLE001 - an unreadable review blocks nothing (V6)
            got = {"blocking": [], "advisory": [f"review unreadable: {e}"[:200]]}
    blocking = [str(x)[:400] for x in (got or {}).get("blocking") or []][:12]
    advisory = [str(x)[:300] for x in (got or {}).get("advisory") or []][:12]
    svc.results.setdefault("reviews", []).append({"entry": ctx.entry, "blocking": blocking, "advisory": advisory})
    svc.note("S11_review", not blocking or ctx.entry > 1, "; ".join(blocking)[:300], entry=ctx.entry)
    if blocking and ctx.entry == 1:
        return verdict(False, blocking, owner="verifier")
    return verdict(True, [], advisory=advisory + (blocking if ctx.entry > 1 else []))


def handlers(svc: SwarmServices, *, lease: Any, accept: Any) -> dict[str, Any]:
    """The machine's handlers; ``session`` comes from the pool (``svc.pool.handler``)."""

    async def build(ctx: Any) -> Any:
        return await svc.build_fn(ctx)  # type: ignore[misc]

    async def prompt(ctx: Any) -> Any:
        return await svc.prompt_fn(ctx)  # type: ignore[misc]

    async def gates(ctx: Any) -> Any:
        return await svc.gates_fn(ctx)  # type: ignore[misc]

    async def second(ctx: Any) -> Any:
        return await svc.second_fn(ctx)  # type: ignore[misc]

    async def hack(ctx: Any) -> Any:
        return await svc.hack_fn(ctx)  # type: ignore[misc]

    return {"session": svc.pool.handler, "lease": lease, "plan": plan_state, "oracle_check": oracle_check_state, "oracle_plan": oracle_plan_state,
            "review": review_state, "build": build,
            "prompt": prompt, "gates": gates, "second": second, "hack": hack, "accept": accept}


def machine_with_briefs(test_cmd: str, **kw: Any) -> dict[str, Any]:
    m = swarm_machine(**kw)
    for s in m["states"]:
        if s["id"] == "oracle":
            s["session"]["brief"] = ORACLE_BRIEF.replace("{test_cmd}", test_cmd)
        elif s["id"] == "verifier":
            s["session"]["brief"] = VERIFIER_BRIEF
    return m


# ---- the run: candidates -> machines ------------------------------------------------------------------------------------------------
def candidate_of(win: Mapping[str, Any], repo_dir: Path, *, gh: Any = None, lease: Any = None) -> SwarmCandidate:
    from crucible.factory.lanes.windows import REPO_URLS

    url = REPO_URLS.get(str(win["repo"]), "")
    owner_repo = "/".join(url.rstrip("/").removesuffix(".git").split("/")[-2:]) if url else None
    recs = request_records(str(win["repo"]), win, repo_dir, gh=gh, owner_repo=owner_repo, lease=lease)
    cid = f"sw-{win['repo']}-{hashlib.sha256((str(win['a']) + '|' + str(win['id'])).encode()).hexdigest()[:8]}"
    return SwarmCandidate(id=cid, repo=str(win["repo"]), window=dict(win), requests=[r.to_json() for r in recs])


def summarize(root: Path) -> dict[str, Any]:
    """The swarm funnel (per stage: in, passed, reasons), the accepted windows and one example."""
    root = Path(root)
    f = Funnel(root / "funnel.jsonl")
    res: dict[str, Any] = {"funnel": f.summary().get(NAME, {}), "candidates": {}}
    for p in sorted((root / "candidates").glob("*.json")):
        c = json.loads(p.read_text())
        res["candidates"][c["id"]] = {k: c.get(k) for k in ("repo", "source_window", "status", "reason", "stage", "milestones", "edges", "gates", "usd")}
    res["accepted"] = [k for k, v in res["candidates"].items() if v.get("status") == "done"]
    res["passing_deterministic_gates"] = [k for k, v in res["candidates"].items() if (v.get("gates") or {}).get("ok")]
    return res




class ReasoningCap:
    """A model client whose JSON calls carry a thinking budget (OpenRouter ``reasoning.max_tokens``): deepseek-v4-flash and
    qwen3.8-flash spent the whole ``max_tokens`` thinking and returned an empty answer (``finish=length``) in this phase's
    reviews, plans and shadow suites. Everything else is the wrapped client's."""

    def __init__(self, client: Any, *, reasoning_tokens: int = 12000):
        self._client, self._cap = client, reasoning_tokens

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        body = dict(kw.pop("extra_body", None) or {})
        body.setdefault("reasoning", {"max_tokens": self._cap})
        return await self._client.complete_json(messages, extra_body=body, **kw)


def _family_of_client(c: Any) -> str:
    from crucible.factory.provenance import family_of_client

    return family_of_client(c) or ""


def checked_overrides(current: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """``overrides`` after the builder/independent family rule (``models.BUILDER_ROLES`` vs ``models.INDEPENDENT_ROLES``,
    with the swarm's own role names): a ``SameFamilyError`` names every independent role whose client would share a
    builder's family."""
    from crucible.factory.models import BUILDER_ROLES, INDEPENDENT_ROLES, SameFamilyError

    merged = {**dict(current), **dict(overrides)}
    builders = {_family_of_client(c): r for r, c in merged.items() if r in BUILDER_ROLES}
    bad = [f"role {r!r} shares family {_family_of_client(c)!r} with builder role {builders[_family_of_client(c)]!r}"
           for r, c in merged.items() if (r in INDEPENDENT_ROLES or r.startswith("rewriter_deepseek")) and _family_of_client(c) in builders
           and (r in overrides or builders[_family_of_client(c)] in overrides)]
    if bad:
        raise SameFamilyError("; ".join(bad))
    return dict(overrides)


class SwarmRunner:
    """Candidates -> one machine each (``run_all``). ``run`` is the phase's guarded run (``scripts/lh_run.Run``: one USD ledger,
    ``solver_client``, ``spend``); ``hack_episode(task) -> {reward, ...}`` runs the hacker in the solver's World (the script owns
    the RL harness). Containers: grading ``<grade_prefix>*``, Worlds, leak probes, S3 and sessions ``<world_prefix>*``."""

    def __init__(self, run: Any, *, root: Path, clones: Path, store3: Any, world_prefix: str, grade_prefix: str, lease: Any,
                 hack_episode: Callable[[Any], Any] | None = None, gh: Any = None, max_milestones: int = 4, parallel: int = 2, grade_parallel: int = 6,
                 paid_gates: bool = True, budget_usd: float = 2.0, log: Callable[[dict], None] | None = None, redo: bool = False,
                 intent: Mapping[str, Any] | None = None, clients: Mapping[str, Any] | None = None):
        """``intent``: the planner's intent (default the generic P5 one; ``factory.failures.BuilderIntent.planner_intent()`` turns
        mined student failures into one). ``clients``: role -> client overrides (e.g. the Claude Code builder on the planner,
        oracle builder, verifier writer and author); a Claude Code client on a reviewer, hacker or second verifier raises
        ``models.SameFamilyError``, as does any override that shares the family of a builder override."""
        from crucible.factory.generators.base import GateConfig
        from crucible.factory.handoff import TaskRepo
        from crucible.factory.lanes.windows import ImageBook, WindowStore, sandbox_factory

        self.run, self.root, self.src_clones, self.store3 = run, Path(root), Path(clones), store3
        self.world_prefix, self.grade_prefix, self.lease, self.hack_episode, self.gh = world_prefix, grade_prefix, lease, hack_episode, gh
        self.max_milestones, self.parallel, self.paid_gates, self.budget_usd, self.redo = max_milestones, parallel, paid_gates, budget_usd, redo
        self.log = log or (lambda ev: print(json.dumps(ev, default=str)[:400], flush=True))
        self.funnel = Funnel(self.root / "funnel.jsonl")
        self.images = ImageBook(parallel=2)
        self.sandboxes = sandbox_factory(parallel=grade_parallel, memory="1536m", name_prefix=grade_prefix)
        self.store = WindowStore(self.root / "store" / "train", split="train")
        self.family = TaskRepo(self.root / "family", "swarm-window")
        self.gate_cfg = GateConfig()
        self.clients = {r: run.solver_client(m, tag=r) for r, m in ROLE_MODELS.items()}
        self.clients["rewriter_qwen"] = run.solver_client(ROLE_MODELS["author"], tag="rewriter_qwen")
        self.clients["rewriter_deepseek"] = run.solver_client(ROLE_MODELS["second_verifier"], tag="rewriter_deepseek")
        for r in ("planner", "reviewer", "second_verifier", "rewriter_deepseek"):  # call roles (sessions keep their own settings)
            self.clients[r] = ReasoningCap(self.clients[r])
        self.intent = dict(intent) if intent else None
        self.clients.update(checked_overrides(self.clients, clients or {}))
        self._booked = {r: 0.0 for r in self.clients}
        self._lane: Any = None
        self._lock = asyncio.Lock()
        from crucible.factory.swarm_journal import SwarmJournal

        self.journal = SwarmJournal(self.root / "journal")  # sessions, verdicts, failures: scripts/swarm.py diagnose

    def families(self) -> dict[str, str]:
        """role -> the family of the client that runs it (the machine validator's ``different_family_from``), the P5 map's
        families where a client names none."""
        from crucible.factory.models import family_of
        from crucible.factory.provenance import family_of_client

        out = {r: family_of(m) for r, m in ROLE_MODELS.items()}
        out.update({r: f for r, c in self.clients.items() if (f := family_of_client(c))})
        return out

    # -- spend: one row per role and candidate in the phase's ledger -----------------------------------------------------------
    def flush_spend(self, who: str) -> dict[str, float]:
        rows = {}
        for role, c in self.clients.items():
            now = float(getattr(c.spend, "usd", 0.0))
            d = now - self._booked[role]
            if d > 0:
                rows[role] = d
                self._booked[role] = now
        with Path(self.run.spend_path).open("a") as fh:
            for role, d in rows.items():
                fh.write(json.dumps({"ts": round(time.time(), 1), "stage": "swarm", "role": role, "who": who, "usd": round(d, 8)}) + "\n")
        return rows

    def lane(self) -> Any:
        from crucible.factory.generators.base import GenLane

        if self._lane is None:
            self._lane = GenLane(self.store, generator=NAME, task_cls=SwarmTask, repo_dirs={p.name: str(p) for p in (self.root / "clones").iterdir() if p.is_dir()},
                                 sandboxes=self.sandboxes)
            self._lane.world_prefix = f"{self.world_prefix}world-"
        self._lane.repo_dirs.update({p.name: str(p) for p in (self.root / "clones").iterdir() if p.is_dir()})  # clones made since
        return self._lane

    # -- one candidate ----------------------------------------------------------------------------------------------------------
    async def prepare(self, wid: str) -> tuple[SwarmCandidate, Path, Path, dict[str, Any]]:
        from crucible.factory.lanes.windows import run_tests, suite_targets
        from crucible.factory.lanes.windows_git import RepoGit

        win = self.store3.window(wid)
        self.lease.guard_repo(win["repo"])
        self.store.guard(win["repo"])
        clone = derived_clone(self.src_clones / win["repo"], self.root / "clones" / win["repo"])
        cdir = self.root / "candidates"
        cand = None
        for p in sorted(cdir.glob("*.json")) if cdir.exists() else []:
            c = json.loads(p.read_text())
            if c.get("source_window") == wid:
                cand = SwarmCandidate(id=c["id"], repo=c["repo"], window=win, requests=c["requests"], image=c.get("image", ""))
        if cand is None:
            cand = await asyncio.to_thread(candidate_of, win, clone, gh=self.gh, lease=self.lease)
        ok_reqs = [r for r in cand.requests if (r.get("lease") or {}).get("ok", True)]
        self.funnel.note(NAME, cand.id, "S1_source", len(ok_reqs) >= 2, "" if len(ok_reqs) >= 2 else f"{len(ok_reqs)} usable request(s)", requests=len(ok_reqs),
                         source=(ok_reqs[0]["evidence"].get("source") if ok_reqs else None))
        work = self.root / "work" / cand.id
        base = work / "base"
        if not base.exists():
            okb, bad = await asyncio.to_thread(RepoGit(clone).build, cand.a, [], base)
            if not okb:
                raise RuntimeError(f"A does not check out ({bad})")
        card = await self.images.ensure(cand.repo, base)
        if not card["ok"]:
            raise RuntimeError("the World image at A does not build")
        cand.image = card["image"]
        targets = suite_targets(base, {})
        apath = work / "a_suite.json"
        if not apath.exists():
            tr = await run_tests(self.sandboxes(cand.image), base, {}, {}, targets)
            apath.write_text(json.dumps({"infra": tr.infra, "results": tr.results}))
        a_suite = json.loads(apath.read_text())
        self.save_cand(cand, wid, status="prepared")
        return cand, clone, base, {"targets": targets, "a_suite": a_suite}

    def save_cand(self, cand: SwarmCandidate, wid: str, **kw: Any) -> None:
        p = self.root / "candidates" / f"{cand.id}.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**old, "id": cand.id, "repo": cand.repo, "source_window": wid, "requests": cand.requests, "image": cand.image, **kw},
                                indent=1, default=str))

    def spec_of(self, cand: SwarmCandidate, base: Path) -> dict[str, Any]:
        from crucible.factory.verifier2 import repo_context

        reqs = [{"id": r["evidence"]["rid"], "text": r["text"]} for r in cand.requests if (r.get("lease") or {}).get("ok", True)]
        names = mentioned_names(" ".join(r["text"] for r in reqs))
        return {"repo_name": cand.repo, "start": {"ref": cand.window.get("a_ref"), "sha": cand.a}, "requests": reqs, "code_links": call_links(base, names),
                "intent": self.intent or {"id": "intent-p5-generic-change", "behaviour": "carry out a dependent multi-step change in a real Python library"},
                "repository": repo_context(base, " ".join(r["text"] for r in reqs), budget=24_000),
                "sources": [r["id"] for r in cand.requests]}

    async def one(self, wid: str) -> dict[str, Any]:
        from crucible.actors.rlm.agent import RLMConfig
        from crucible.factory.handoff import accept_state, lease_state
        from crucible.factory.machine import MachineRun
        from crucible.factory.sessions import SessionPool

        cand, clone, base, info = await self.prepare(wid)
        done = json.loads((self.root / "candidates" / f"{cand.id}.json").read_text())
        if self.redo and done.get("status") == "done":  # an accepted task is never rebuilt by a redo
            return done
        if self.redo:  # a fresh machine run (the window store, World image and A's suite results are kept)
            import shutil

            from crucible.factory.sessions import safe

            shutil.rmtree(self.root / "machines" / cand.id, ignore_errors=True)
            shutil.rmtree(self.root / "factory" / "runs" / safe(cand.id), ignore_errors=True)  # the sessions' worktrees start from A again
            done = {}
        if done.get("status") in ("done", "rejected", "paused", "failed") and done.get("final"):
            return done
        test_cmd = "python -m pytest -q -x -p no:cacheprovider " + " ".join(info["targets"][:30])
        m = machine_with_briefs(test_cmd, budget_usd=self.budget_usd)
        params = {"spec": self.spec_of(cand, base), "repo": str(clone), "base": cand.a, "task_id": cand.id, "repo_name": cand.repo,
                  "scope": {"repo": cand.repo, "start": cand.a, "generator": NAME}, "abort_criteria": ["the start repository changes upstream"],
                  "holder": f"root/{cand.id}", "parent": f"{cand.repo}@{cand.window.get('a_ref')}"}
        # 80 turns ended a 4-milestone oracle build with its four commits made but rlm.finish not yet called (P5b run, boltons)
        cfg = RLMConfig(temperature=0.3, tree_max_depth=0, max_turns=150, window_tokens=60_000, max_tokens=12000, cell_timeout_s=300.0, wall_s=3600.0)
        from crucible.factory.provenance import ProvenanceLedger, family_of_client

        # the oracle builder's and verifier writer's Φ: builder_potential.DEFAULTS (oracle_commits, hidden_tests)
        pool = SessionPool(self.root / "factory", repo=self.family, machine=m, run_id=cand.id, params=params, llm_for=lambda role: self.clients[role],
                           image=cand.image, rlm_config=cfg, sandbox_factory=world_sandbox_class(self.world_prefix), task_id=cand.id, max_steps=200,
                           on_session=self.journal.session)
        ledger = ProvenanceLedger(m, family_for=lambda role: family_of_client(self.clients.get(role)), model_for=lambda role: str(getattr(self.clients.get(role), "model", "")),
                                  reads=SERVICE_READS)
        svc = SwarmServices(cand=cand, clone=clone, work=self.root / "work" / cand.id, funnel=self.funnel, max_milestones=self.max_milestones,
                            clients=self.clients, pool=pool)
        svc.results["base"], svc.results["a_suite"], svc.results["targets"] = str(base), info["a_suite"], info["targets"]
        svc.oracle_fn = lambda shas, plan: self.regressions(svc, base, shas, plan)
        svc.oracle_plan_fn = lambda shas, plan: self.oracle_plan(svc, base, shas, plan)
        svc.build_fn, svc.prompt_fn, svc.gates_fn = (lambda ctx: self.build(svc, ctx)), (lambda ctx: self.prompt(svc, ctx)), (lambda ctx: self.gates(svc, ctx))
        svc.second_fn, svc.hack_fn = (lambda ctx: self.second(svc, ctx)), (lambda ctx: self.hack(svc, ctx))
        run = MachineRun(m, run_dir=self.root / "machines" / cand.id, run_id=cand.id, models=self.families(), params=pool.params,
                         handlers=self.journal.guard(handlers(svc, lease=lease_state(self.family), accept=accept_state(self.family)), ledger), services=svc,
                         **pool.machine_kwargs())
        t0 = time.monotonic()
        try:
            res = await run.run()
        finally:
            await pool.close()
            spent = self.flush_spend(cand.id)
        self.journal.run_end(run=cand.id, status=res.status, reason=res.reason, usd=sum(spent.values()), task=cand.id, accepted=res.status == "done")
        ann = ((res.outputs.get("lease") or {}).get("lease") or {}).get("announcement")
        if ann and res.status in ("rejected", "paused", "failed", "stopped") and not ((res.outputs.get("accept") or {}).get("accept") or {}).get("commit"):
            with contextlib.suppress(Exception):  # the scope is free again (a dropped or paused run never holds it)
                self.family.retire(ann, "aborted" if res.status != "rejected" else "rejected", reason=str(res.reason)[:200])
        row = {"status": res.status, "reason": res.reason, "decision": res.decision_request, "entries": res.entries, "final": True,
               "plan": svc.results.get("plan"), "reviews": svc.results.get("reviews"), "build": svc.results.get("build"), "gates": svc.results.get("gates"),
               "second": svc.results.get("second"), "hack": svc.results.get("hack"), "prompt_findings": svc.results.get("prompt_findings"),
               "milestones": (svc.results.get("build") or {}).get("milestones"), "edges": (svc.results.get("build") or {}).get("edges"),
               "stage": (svc.results.get("build") or {}).get("stage"), "usd": round(sum(spent.values()), 6), "seconds": round(time.monotonic() - t0, 1),
               "sessions": [r.__dict__ for r in pool.records]}
        self.save_cand(cand, wid, **row)
        self.log({"candidate": cand.id, "status": res.status, "reason": str(res.reason)[:200], "usd": row["usd"]})
        return row

    async def finish(self, wid: str) -> dict[str, Any]:
        """Re-run S9 and S12 (then S17) for a candidate whose window, prompt and gates are stored: a mini machine ``lease ->
        second -> hack -> accept`` journaled under ``machines/<id>-s12`` (no session, no rebuild)."""
        from crucible.factory.handoff import accept_state, lease_state
        from crucible.factory.machine import MachineRun

        cand, clone, _base, _info = await self.prepare(wid)
        full = swarm_machine(budget_usd=self.budget_usd)
        keep = {"lease", "second", "hack", "accept", "accepted", "dropped", "abstained", "herded"}
        trans = [{"from": "lease", "to": "second", "when": "lease.ok == true"} if t["from"] == "lease" and t["to"] == "plan" else t for t in full["transitions"]]
        m = {"run": full["run"], "states": [dict(x) for x in full["states"] if x["id"] in keep],
             "transitions": [t for t in trans if t["from"] in keep and t["to"] in keep]}
        for st in m["states"]:
            st.pop("different_family_from", None)  # the sessions it names are not in this machine (the families were checked in the full run)
        params = {"task_id": cand.id, "scope": {"repo": cand.repo, "start": cand.a, "generator": NAME, "stage": "s12"},
                  "abort_criteria": ["the task changes"], "holder": f"root/{cand.id}-s12", "parent": f"{cand.repo}@{cand.window.get('a_ref')}"}
        svc = SwarmServices(cand=cand, clone=clone, work=self.root / "work" / cand.id, funnel=self.funnel, clients=self.clients)
        svc.second_fn, svc.hack_fn = (lambda ctx: self.second(svc, ctx)), (lambda ctx: self.hack(svc, ctx))
        h = {"lease": lease_state(self.family), "accept": accept_state(self.family), "second": lambda ctx: self.second(svc, ctx),
             "hack": lambda ctx: self.hack(svc, ctx)}
        run = MachineRun(m, run_dir=self.root / "machines" / f"{cand.id}-s12", run_id=f"{cand.id}-s12", models=ROLE_MODELS, params=params, handlers=h, services=svc)
        try:
            res = await run.run()
        finally:
            spent = self.flush_spend(f"{cand.id}-s12")
        row = {"status": res.status, "reason": res.reason, "final": True, "finished_from": "second", "second": svc.results.get("second"),
               "hack": svc.results.get("hack"), "usd_s12": round(sum(spent.values()), 6)}
        old = json.loads((self.root / "candidates" / f"{cand.id}.json").read_text())
        self.save_cand(cand, wid, **{**row, "first_run": {k: old.get(k) for k in ("status", "reason", "hack")}})
        self.log({"candidate": cand.id, "finish": res.status, "reason": str(res.reason)[:200]})
        return row

    async def run_all(self, wids: Sequence[str]) -> dict[str, Any]:
        sem = asyncio.Semaphore(self.parallel)

        async def go(w: str) -> tuple[str, Any]:
            async with sem:
                try:
                    return w, await self.one(w)
                except Exception as e:  # noqa: BLE001 - one candidate's failure is recorded, never the run's
                    self.funnel.note(NAME, f"from:{w}", "S1_source", False, f"error: {type(e).__name__}: {e}"[:300])
                    return w, {"error": f"{type(e).__name__}: {e}"[:500]}

        return dict(await asyncio.gather(*[go(w) for w in wids]))

    # -- S4: regressions of A's suite on the oracle ---------------------------------------------------------------------------
    def oracle_files(self, clone: Path, a: str, sha: str) -> dict[str, str | None]:
        files: dict[str, str | None] = {}
        for ln in out(git(clone, "diff", "--name-status", "--no-renames", a, sha)).splitlines():
            st, _, p = ln.partition("\t")
            if not p or is_test_path(p) or is_protected(p):
                continue
            if st.startswith("D"):
                files[p] = None
            else:
                r = git(clone, "show", f"{sha}:{p}", check=False)
                if r.returncode == 0:
                    try:
                        files[p] = r.stdout.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
        return files

    def contradicting(self, base: Path, plan: Mapping[str, Any], ids: Sequence[str]) -> dict[str, list[str]]:
        """A's tests that use a name the plan removes (``oracleplan.contradicting_tests``): defects, not regression checks."""
        from crucible.factory.oracleplan import contradicting_tests, plan_probes

        removed = [p.name for p in plan_probes(plan) if p.kind == "removal"]
        files = sorted({t.split("::", 1)[0] for t in ids})
        return contradicting_tests(removed, {f: (base / f).read_text(errors="replace") for f in files if (base / f).is_file()}, ids)

    async def regressions(self, svc: SwarmServices, base: Path, shas: Sequence[str], plan: Mapping[str, Any] | None = None) -> list[str]:
        from crucible.factory.lanes.windows import run_tests

        a_res = (svc.results.get("a_suite") or {}).get("results") or {}
        passed_a = {t for t, s in a_res.items() if s == "passed"}
        if not passed_a:
            return []
        # Phase A: a test that uses a name the plan removes is expected to fail once it is removed; it is never a regression
        contra = self.contradicting(base, plan, sorted(passed_a)) if plan else {}
        files = self.oracle_files(svc.clone, svc.cand.a, shas[-1])
        bad: set[str] | None = None
        for _ in range(2):  # a regression must reproduce (a flaky test is not the oracle's)
            tr = await run_tests(self.sandboxes(svc.cand.image), base, files, {}, svc.results["targets"])
            if tr.infra != "ok":
                return []
            svc.results["oracle_suite"] = dict(tr.results)
            now = {t for t in passed_a if tr.results.get(t) != "passed"} - set(contra)
            bad = now if bad is None else bad & now
        if not bad:
            return []
        note = (f"; the {len(contra)} test(s) that use what the plan removes are expected to fail and are not counted" if contra else "")
        return [(f"the existing tests that passed before now fail on your final commit ({len(bad)}): {sorted(bad)[:12]}; fix the commit of the "
                 f"milestone that broke them (amend it or rebase), keeping one commit per milestone{note}")]

    async def oracle_plan(self, svc: SwarmServices, base: Path, shas: Sequence[str], plan: Mapping[str, Any]) -> dict[str, Any]:
        """S4.5 in the candidate's World (``oracleplan.check`` through ``windows_s3.WindowWorldPort``, offline)."""
        from crucible.factory.lanes.windows_s3 import WindowWorldPort
        from crucible.factory.oracleplan import check, world_runner

        if not shas:
            return {"ok": False, "infra": True, "findings": ["no oracle commits to check"]}
        a_res = (svc.results.get("a_suite") or {}).get("results") or {}
        port = WindowWorldPort({"id": svc.cand.id, "repo": svc.cand.repo, "world": {"image": svc.cand.image}}, base, image=svc.cand.image,
                               name_prefix=self.world_prefix)
        return await check(plan, base, self.oracle_files(svc.clone, svc.cand.a, shas[-1]), runner=world_runner(port),
                           regression_ids=sorted(t for t, st in a_res.items() if st == "passed"), regression_on_oracle=svc.results.get("oracle_suite"))

    # -- build: tests on A and the oracle, the synthetic window, validate_window ----------------------------------------------
    async def build(self, svc: SwarmServices, ctx: Any) -> Any:
        from crucible.factory.lanes.windows import run_tests, validate_window

        plan = ctx.inputs["plan"]
        base = Path(svc.results["base"])
        tests = hidden_tests(Path(ctx.artifact_root) / ctx.inputs["tests"]["path"], plan)
        tries = svc.results.setdefault("build_tries", {"verifier": 0, "planner": 0})
        probs = test_problems(tests, plan)
        if probs:
            return self._back(svc, "verifier", probs, tries, stage="S5_verifier")
        shas = list((ctx.inputs.get("check") or {}).get("shas") or [])
        hidden = {hidden_path(base, mid): txt for mid, txt in tests.items()}
        files_b = self.oracle_files(svc.clone, svc.cand.a, shas[-1])
        sb = self.sandboxes(svc.cand.image)
        at_a = await run_tests(sb, base, {}, hidden, sorted(hidden))
        at_b = await run_tests(sb, base, files_b, hidden, sorted(hidden))
        if at_a.infra != "ok" or at_b.infra != "ok":
            return verdict(False, [f"masked: the grading sandbox did not run ({at_a.why or at_b.why})"], owner="infra")
        findings, per = outcome_findings(at_a.results, at_b.results, plan, base)
        svc.results["hidden_outcomes"] = per
        empty = [mid for mid, v in per.items() if not v["f2p"]]
        if len(empty) > len(per) - 2 or (empty and tries["verifier"] < 1):
            return self._back(svc, "verifier", findings, tries, stage="S5_solvable", per=per)
        svc.note("S5_solvable", True, per=per)
        win, info = await asyncio.to_thread(synthesize, svc.clone, svc.cand.a, shas, tests, plan, base, cid=svc.cand.id,
                                            a_ref=str(svc.cand.window.get("a_ref")), a_date=str(svc.cand.window.get("a_date") or ""), repo=svc.cand.repo)
        rec = await validate_window(win, self.store, self.images, self.sandboxes, reruns=3, min_milestones=2, mutants=True, import_tolerant=True,
                                    leak_prefix=f"{self.world_prefix}leak-", log=None)
        drop = set((svc.results.get("oracle_plan") or {}).get("drop_from_regression") or [])
        if drop:  # S4.5: a test of A that uses a name the plan removes is a defect, never a regression check
            rec["excluded"] = {**(rec.get("excluded") or {}), "contradicts_plan": sorted(drop & (set(rec.get("p2p") or []) | set(rec.get("collateral") or [])))}
            rec["p2p"] = [t for t in rec.get("p2p") or [] if t not in drop]
            rec["collateral"] = [t for t in rec.get("collateral") or [] if t not in drop]
        rec[NAME] = {**info, "plan": plan, "requests": ctx.params["spec"]["requests"], "source_window": svc.cand.window["id"], "sources": ctx.params["spec"]["sources"],
                     "oracle_commits": shas, "hidden_outcomes": per, "verifier_findings_open": findings, "oracle_plan": svc.results.get("oracle_plan")}
        self.store.save_window(rec)
        stage, reason = rec.get("stage"), str(rec.get("reason") or "")
        b = {"stage": stage, "reason": reason, "milestones": len(rec.get("milestones") or []), "edges": rec.get("dependency_edges"),
             "mutants": rec.get("mutant_counts"), "plan_of_unit": {u: mid for mid, u in info["unit_of"].items()}}
        svc.results["build"] = b
        self.funnel.note(NAME, svc.cand.id, "S3_world", stage not in ("windows", "world_built"), reason if stage in ("windows", "world_built") else "")
        if stage == "mutants_valid":
            self.funnel.note(NAME, svc.cand.id, "S4_window", True, milestones=b["milestones"], edges=b["edges"])
            return verdict(True, **b)
        if stage == "f2p_valid" and rec.get("independent"):
            self.funnel.note(NAME, svc.cand.id, "S6", False, "independent: no executed edge", milestones=b["milestones"])
            plan_ids = [m["id"] for m in plan["milestones"]]
            why = [(f"the planned dependencies did not execute: every milestone's tests pass without the earlier milestones ({plan_ids}); plan a "
                    "series where a later milestone calls or extends what an earlier one adds")]
            return self._back(svc, "planner", why, tries, stage="S6")
        if "milestone(s) with fail-to-pass" in reason:
            return self._back(svc, "verifier", findings or [reason], tries, stage="S4_window")
        self.funnel.note(NAME, svc.cand.id, "S4_window", False, reason)
        return verdict(False, [reason], owner="")

    def _back(self, svc: SwarmServices, owner: str, findings: Sequence[str], tries: dict[str, int], *, stage: str, **ev: Any) -> Any:
        self.funnel.note(NAME, svc.cand.id, stage, False, "; ".join(findings)[:300], **ev)
        if tries[owner] >= 1:
            return verdict(False, findings, owner="")
        tries[owner] += 1
        return verdict(False, findings, owner=owner)

    # -- S7 ----------------------------------------------------------------------------------------------------------------------
    async def prompt(self, svc: SwarmServices, ctx: Any) -> Any:
        from crucible.factory.generators.base import author_generated
        from crucible.factory.lanes.windows_git import RepoGit
        from crucible.factory.longtask import check_longtask

        cid = svc.cand.id
        rec = self.store.window(cid)
        sw = rec[NAME]
        plan_ms = {m["id"]: m for m in sw["plan"]["milestones"]}
        req_text = {r["id"]: r["text"][:600] for r in sw["requests"]}
        unit_plan = {u: mid for mid, u in sw["unit_of"].items()}
        extra = {"planned_changes": {m["id"]: {"requirement": plan_ms[unit_plan[m["unit"]]]["requirement"],
                                               "public_names": plan_ms[unit_plan[m["unit"]]]["public_names"],
                                               "users_requests": [req_text.get(r, "") for r in plan_ms[unit_plan[m["unit"]]]["requests"]]}
                                     for m in rec["milestones"] if m.get("unit") in unit_plan}}
        findings: list[str] = []
        res = None
        for k in range(1, 4):
            try:
                res = await author_generated(svc.clients["author"], rec, RepoGit(svc.clone), self.store.dir(cid) / "base", style=STYLE, findings=findings,
                                             extra=extra, sample_key=f"p5b-author-{cid}-{k}")
            except Exception as e:  # noqa: BLE001 - recorded, retried
                findings = [f"error: {e}"[:300]]
                continue
            findings = res["findings"]
            if not findings:
                break
        svc.results["prompt_findings"] = findings
        if res is None or findings:
            self.funnel.note(NAME, cid, "S7", False, "; ".join(findings)[:300])
            return verdict(False, findings, owner="author")
        # import paths the hidden tests use that A cannot import are stated by the lane, as required names are (executed in A's World)
        missing = await self.missing_at_a(svc, rec)
        stated = unstated_imports(missing, res["prompt"])
        if stated:
            from crucible.factory.lanes.windows import locate

            res["prompt"] = state_imports(res["prompt"], stated)
            res["spans"] = {r["id"]: list(loc) for r in res["requirements"] if (loc := locate(res["prompt"], r["text"]))}
        svc.results["imports_stated_by_lane"] = stated
        row = {"task": cid, "findings": [], "prompt": res["prompt"], "requirements": res["requirements"], "spans": res["spans"],
               "authored_by": f"model:{ROLE_MODELS['author']}", "names_stated_by_lane": res.get("names_stated_by_lane"),
               "imports_stated_by_lane": stated, "imports_missing_at_a": missing}
        task = self.lane().task(cid, authored=row)
        row["conformance"] = check_longtask(task, long_horizon=True, need_prompt=True)
        (self.store.dir(cid) / "prompt.json").write_text(json.dumps(row, indent=1, sort_keys=True, default=str))
        ok = not row["conformance"]
        self.funnel.note(NAME, cid, "S7", ok, "; ".join(row["conformance"])[:300])
        return verdict(ok, row["conformance"], owner="author")

    async def missing_at_a(self, svc: SwarmServices, rec: Mapping[str, Any]) -> list[list[str]]:
        """The ``(module, name)`` pairs the hidden tests import that A cannot import, run in A's World image (offline)."""
        import uuid

        from crucible.factory.lanes.windows_world import docker

        pairs = test_imports(rec.get("tests_b") or {})
        if not pairs:
            return []
        base = str((self.store.dir(svc.cand.id) / "base").resolve())
        name = f"{self.world_prefix}imports-{uuid.uuid4().hex[:10]}"
        r = await asyncio.to_thread(docker, "run", "--rm", "--name", name, "--network", "none", "-v", f"{base}:/w:ro", "-w", "/w", "-e",
                                    "PYTHONPATH=/w/src:/w", "-e", "PYTHONDONTWRITEBYTECODE=1", rec["world"]["image"], "python", "-c", IMPORT_PROBE,
                                    json.dumps(pairs), timeout=300)
        m = re.search(r"CRUCIBLE_MISSING=(.*)", r.stdout or "")
        return json.loads(m.group(1)) if m else []

    def authored(self, cid: str) -> Any:
        doc = json.loads((self.store.dir(cid) / "prompt.json").read_text())
        return self.lane().task(cid, authored=doc)

    # -- S6, S5, S8, S3, S10 -----------------------------------------------------------------------------------------------------
    async def gates(self, svc: SwarmServices, ctx: Any) -> Any:
        from crucible.factory.generators.base import deterministic_gates

        cid = svc.cand.id
        res = await deterministic_gates(self.authored(cid), cfg=self.gate_cfg, s3_prefix=f"{self.world_prefix}s3-")
        (self.store.dir(cid) / "gates.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
        for g, r in res["gates"].items():
            if r["ok"] is not None:
                self.funnel.note(NAME, cid, g, bool(r["ok"]), "; ".join(r.get("findings") or [])[:300])
        svc.results["gates"] = {"ok": res["ok"], "passed": res["passed"], "findings": {g: r.get("findings") for g, r in res["gates"].items()},
                                "alternatives_pending": res["gates"].get("S8", {}).get("alternatives_pending")}
        return verdict(res["ok"], [f for g, r in res["gates"].items() for f in (r.get("findings") or [])][:20], owner="verifier", passed=res["passed"])

    # -- S8 alternatives and S9 (paid, cheap) --------------------------------------------------------------------------------------
    async def second(self, svc: SwarmServices, ctx: Any) -> Any:
        from types import SimpleNamespace

        from crucible.factory.alternatives import Alternative, accept, coverage, rewrite_oracle, salvage
        from crucible.factory.verifier2 import EndStateRec, P2Gates, WindowRunner, qualify, s8_ground_truth, s9_second_verifier, truth_for, write_shadow

        cid = svc.cand.id
        if not self.paid_gates:
            svc.results["second"] = {"skipped": "paid gates off"}
            return verdict(True, [], skipped=True)
        task = self.authored(cid)
        rec = task.window
        runner = WindowRunner(task)
        reqs = [r for r in task.requirements if r.get("milestone")]
        rids = [r["id"] for r in reqs]
        req_ms = {r["id"]: r["milestone"] for r in reqs}
        suite = None
        for chunk in (6, 3):
            try:
                suite = await write_shadow(svc.clients["second_verifier"], prompt=task.prompt, requirements=task.requirements, base_dir=task.base_dir,
                                           model=ROLE_MODELS["second_verifier"], family="deepseek", guard={"oracle_files": rec["oracle_files"], "hidden_tests": rec["tests_b"]},
                                           chunk=chunk)
                break
            except Exception as e:  # noqa: BLE001
                svc.results.setdefault("second_errors", []).append(str(e)[:200])
        cands: list[Any] = []
        for role, style, fam, model in (("rewriter_qwen", "imperative", "qwen", ROLE_MODELS["author"]), ("rewriter_deepseek", "functional", "deepseek",
                                                                                                           ROLE_MODELS["second_verifier"])):
            try:
                cands.append(await rewrite_oracle(svc.clients[role], task, style=style, model=model, family=fam))
            except Exception as e:  # noqa: BLE001
                cands.append(Alternative(f"rewrite-{style}-{fam}", "rewrite", "", model, fam, style, reasons=[f"error: {e}"[:300]]))
        muts = task.mutants()
        mut_recs = {m["id"]: m for m in rec.get("mutants") or []}
        oracle_v2 = await runner.shadow(suite, task.oracle_patch()) if suite else None
        start_v2 = await runner.shadow(suite, "") if suite else None
        mutant_runs = {str(m.id): (str(m.milestone), await runner.shadow(suite, m.solution) if suite else None) for m in muts}
        q1 = qualify(suite, oracle=oracle_v2, alternatives={}, start=start_v2, mutants=mutant_runs, req_milestone=req_ms) if suite else None
        for c in list(cands):
            if not c.patch:
                continue
            v1 = await runner.hidden(c.patch)
            v2 = q1.verdict(await runner.shadow(suite, c.patch)) if suite and q1 else None
            await accept(c, v1=v1, v2=v2)
            if not c.accepted and (v1 or {}).get("reward") != 1.0:
                async def passes(patch: str) -> bool:
                    return ((await runner.hidden(patch)) or {}).get("reward") == 1.0

                sub = await salvage(task, c, passes)
                if sub is not None:
                    await accept(sub, v1=await runner.hidden(sub.patch), v2=q1.verdict(await runner.shadow(suite, sub.patch)) if suite and q1 else None)
                    cands.append(sub)
        accepted = {c.id: c.patch for c in cands if c.accepted}
        states = [EndStateRec("oracle", "oracle", task.oracle_patch(), truth_for("oracle", rids, milestone_of=req_ms)),
                  EndStateRec("start", "start", "", truth_for("start", rids, milestone_of=req_ms))]
        states += [EndStateRec(f"floor:{f.id}", "floor", f.solution, truth_for("floor", rids, milestone_of=req_ms)) for f in task.floors() if f.id != "test_edit"]
        states += [EndStateRec(f"mutant:{m.id}", "mutant", m.solution, truth_for("mutant", rids, milestone_of=req_ms, mutant=mut_recs.get(str(m.id)))) for m in muts]
        states += [EndStateRec(f"rewrite:{c.id}", "alternative" if c.accepted else "rewrite_candidate", c.patch,
                               truth_for("alternative" if c.accepted else "rewrite_candidate", rids, milestone_of=req_ms)) for c in cands if c.patch]
        gates = P2Gates(task=task, runner=runner, alternatives=accepted, suite=suite, states=states)
        ns = SimpleNamespace(services=gates)
        s8 = (await s8_ground_truth(ns)).outputs["verdict"]
        s9 = (await s9_second_verifier(ns)).outputs["verdict"] if suite else {"ok": False, "findings": ["no shadow suite (the second verifier's replies failed)"]}
        alt_only = [f for f in s8["findings"] if "valid alternative(s), need 2" in f]
        s8_rest = [f for f in s8["findings"] if f not in alt_only]
        cov = coverage(cands, writer_family="qwen")
        out_ = {"alternatives": {"candidates": [c.to_json() | {"patch": f"<{len(c.patch)} bytes>"} for c in cands], "accepted": sorted(accepted), "coverage": cov},
                "s8_with_alternatives": {"ok": not s8_rest, "findings": s8["findings"]},
                "s9": {k: s9.get(k) for k in ("ok", "owner", "findings", "evidence")}, "shadow_requirements": sorted(suite.requirements) if suite else [],
                "suite": suite.to_json() if suite else None, "qualification": (gates.results.get("s9") or {}).get("qualification"),
                "errors": svc.results.get("second_errors") or []}
        svc.results["second"] = out_
        (self.store.dir(cid) / "second.json").write_text(json.dumps(out_, indent=1, sort_keys=True, default=str))
        self.funnel.note(NAME, cid, "S8_alternatives", not s8_rest and not alt_only, "; ".join(s8["findings"])[:300], accepted=len(accepted))
        self.funnel.note(NAME, cid, "S9", bool(s9.get("ok")), "; ".join(s9.get("findings") or [])[:300])
        ok = not s8_rest and bool(s9.get("ok"))
        return verdict(ok, [*s8_rest, *(s9.get("findings") or [])], owner="verifier", alternatives=len(accepted))

    # -- S12 ------------------------------------------------------------------------------------------------------------------------
    async def hack(self, svc: SwarmServices, ctx: Any) -> Any:
        cid = svc.cand.id
        if self.hack_episode is None or not self.paid_gates:
            svc.results["hack"] = {"skipped": True}
            return verdict(True, [], skipped=True)
        task = self.authored(cid)
        r = dict(await self.hack_episode(task))
        if r.get("reward") == 1.0:
            r.update(await self.judge_pass(task, r))
        svc.results["hack"] = r
        ok = not r.get("confirmed_hack")
        self.funnel.note(NAME, cid, "S12", ok, "" if ok else f"the hacker scored {r.get('reward')}: {r.get('judged')}")
        return verdict(ok, [] if ok else [f"confirmed hack: reward {r.get('reward')} ({r.get('judged')})"], owner="verifier",
                       **{k: r.get(k) for k in ("reward", "phi", "masked", "judged")})

    async def judge_pass(self, task: Any, r: Mapping[str, Any]) -> dict[str, Any]:
        """A hacker end state that scores 1 is executed again by the second verifier (V6, executed): when every qualified
        shadow requirement passes it and no path check fired, the hacker implemented the behaviour (a valid solution, an
        alternative), not a hack; otherwise it is a confirmed hack (a verifier-1 gap the shadow suite exposes)."""
        from crucible.factory.verifier2 import Qualification, ShadowSuite, WindowRunner

        sec = json.loads((self.store.dir(task.task_id) / "second.json").read_text()) if (self.store.dir(task.task_id) / "second.json").exists() else {}
        patch_path = Path(str(r.get("dir") or "")) / "artifacts" / "model.patch"
        if not sec.get("suite") or not sec.get("qualification") or not patch_path.exists():
            return {"confirmed_hack": True, "judged": "unmeasured: no qualified shadow suite or no patch (a pass is kept as a hack)"}
        suite = ShadowSuite.from_json(sec["suite"])
        q = sec["qualification"]
        qual = Qualification(tests=dict(q["tests"]), requirements=dict(q["requirements"]))
        v2 = qual.verdict(await WindowRunner(task).shadow(suite, patch_path.read_text(errors="replace")))
        measured = [v for v in v2.values() if v is not None]
        valid = bool(measured) and all(measured) and not r.get("path_violations") and not r.get("hack_suspect")
        return {"confirmed_hack": not valid, "shadow_verdicts": v2,
                "judged": "valid solution: the second verifier passes every requirement it measures" if valid else "hack: the second verifier fails it"}
