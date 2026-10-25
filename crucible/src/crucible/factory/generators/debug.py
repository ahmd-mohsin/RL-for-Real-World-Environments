"""The debugging lane (swarm-pipeline/07): rebreak over every language profile. N real fix commits, made after the
student model's release, are reverted in one tree; the agent gets a user's bug report of what they observe and debugs
the interacting faults; each fix's own fail-to-pass tests are its hidden milestone.

**Per language** (``factory.langs``). Every step runs inside ``langs.base.use(profile)``, so the window machinery's
predicates (source, test, protected, test definitions, suite targets, leak probes) answer for the repository's
language; Python keeps the rebreak behaviour unchanged. World images are the profile's (``crucible-lh-d:<lang>-...``),
grading runs in ``langs.runtime.LangSandbox`` (Python: ``windows.NamedSandbox``), containers are ``crucible-lh-d-*``.

**Fixes** (``fix_units`` of rebreak under the profile): first-parent units dated on or after ``since`` (default
2026-05-01, after the target model's release; ``FALLBACK_SINCE`` 2025-09-01 is a marked fallback: each candidate carries
``cutoff`` and ``contamination`` = ``post-release`` when every fix is after ``SINCE``, else ``fallback``) whose message says fix and which change source and test files of the language
and nothing else but prose and test data.

**Real-fix check** (``fix_check``, executed at B, the newest release or the default branch's head): B with only the fix's
source reverted runs the test files of every fix of the pool (or B's whole suite). A test the fix *owns* is one in its
test files that the profile's definition diff proposes (added or changed by the fix); when the profile proposes none for
a file (titles built at run time), every test of that file the revert catches is owned: the executed fail-to-pass set is
the truth, the text only proposes. A compiled language counts only ``failed`` tests: a revert that stops a test file
from building (a removed API) is not a behavioural fault. Interactions, grouping and the prober's executed edges are
rebreak's (``interaction_edges``, ``group_fixes``). ``Debug.subset`` takes a named list of fixes instead of a group: a
mined task rebuilt, or eased by leaving one of its fixes applied in A'.

**The start tree** (``synthesize``). A' = B with every fix of the group reverted: source **and test data** (fixtures,
snapshots, golden files: the expected output is the fix), newest first, 3-way; the tests each fix added or changed are
removed from its test files (``profile.strip_tests``; a file left with no test is removed; a Go file loses the imports
only the removed tests used); the changelog lines the fixes added are removed (a start tree whose changelog announces
the fix hints the answer). B' == B, checked. Then ``windows.validate_window`` (World, leak probes, fail-to-pass with
reruns, prefix owners, the executed prober, revert mutants), after two cheap pre-checks: B's test files build against
A', and the agent's own view of A' builds (``precheck``).

**The task** (``DebugTask``): ``RebreakTask`` with every method inside its profile's scope, the profile's epilogue (its
own test-plumbing names), its World setup (node: the dependency tree linked into the workspace). Prompts are symptom
reports (``author_debug``): what a user observes and how to reproduce it, never the fix, the faulty function names or
the hidden test names; the builder is recorded on every prompt (``BUILDER``), and the lint adds those rules to the
window lane's.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import inspect
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from crucible.factory import langs
from crucible.factory.generators.base import (
    Candidate,
    Funnel,
    GateConfig,
    GenLane,
    apply_chain,
    changed_paths,
    commit_meta,
    compose_prompt,
    deterministic_gates,
    full_patch,
    git,
    out,
    tree_of,
    with_files,
    write_chain,
)
from crucible.factory.generators.rebreak import (
    PROSE,
    Rebreak,
    RebreakTask,
    changed_functions,
    fix_test_names,
    fix_units,
    group_fixes,
    interaction_edges,
    order_fixes,
)
from crucible.factory.lanes.executable import is_test_path
from crucible.factory.lanes.windows import (
    EPILOGUE,
    WindowStore,
    lint_prompt,
    locate,
    normalize_ids,
    sandbox_factory,
    suite_targets,
    validate_window,
    window_checkset,
    window_ledger,
)
from crucible.factory.lanes.windows_git import RepoGit, Unit, Window, _bodies, first_parent_units, key_of_test, version_tags
from crucible.factory.lanes.windows_world import ImageBook
from crucible.factory.langs.base import CHANGELOG_FILE, fix_refs, strip_added_lines, strip_ref_entries, subject_ref, use
from crucible.factory.langs.runtime import IMAGE_REPO, PREFIX, lang_sandbox_factory, world_setup
from crucible.factory.sources.lease import EvalLease
from crucible.factory.sources.records import SourceRecord

NAME = "debug"
# the target model (Qwen/Qwen3.6-35B-A3B) was released 2026-04-15: fixes after SINCE limit contamination; FALLBACK_SINCE is a
# clearly marked fallback where a repository's post-release supply holds no interacting group (every candidate records its cutoff)
SINCE = "2026-05-01"
FALLBACK_SINCE = "2025-09-01"
BUILDER = "claude-code:opus-5.5"
HEADING = "What I am seeing:"
COMPILED = ("go", "rust", "java")
B_STEPS = (20, 45, 75, 110, 150)  # days before the head: the earlier B points of ``candidates_for``
FULL_SUITE_MAX = 400  # test files: above it the interaction screen runs the pool's own test files only



@dataclass(frozen=True)
class EnvBudget:
    """The lane's environment budget: what the keep rule (S15), the benchmark and RL all run a debug task under. The
    timeouts and the checkpoint-Phi cadence scale with the tool limit. 150 calls (S15 first run): agent 2700 s, rollout
    4500 s, Phi every 10 calls, cap 12. 67 of 104 student and 21 of 52 frontier episodes stopped at that limit, so it
    measured the budget, not the skills (07 §6). Raised to 300 by the user on 2026-10-20."""

    tool_limit: int = 300
    agent_s: float = 5400.0
    rollout_s: float = 9000.0
    checkpoint_every: int = 20  # the same points per budget as every 10 at 150 calls
    checkpoint_cap: int = 15  # the periodic points reach call 280; the end always has a slot (``factory.checkpoint``)
    decided: str = "2026-10-20"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


ENV_BUDGET = EnvBudget()

def profile_of(lang: str) -> Any:
    return langs.get(lang)


@contextlib.contextmanager
def scope(lang: str) -> Iterator[Any]:
    with use(profile_of(lang)) as p:
        yield p


class PyDebugImages(ImageBook):
    """The Python World book under the debug lane's image name (``crucible-lh-d:python-<repo>-<hash>``)."""

    def tag_for(self, repo: str, base_dir: Path) -> tuple[str, dict[str, Any]]:
        tag, info = super().tag_for(repo, base_dir)
        return f"{IMAGE_REPO}:python-{tag.split(':', 1)[1]}", info


def debug_sandboxes(*, parallel: int = 3, prefix: str = f"{PREFIX}grade-") -> Callable[[str], Any]:
    """One factory for every language: a profile's ``LangSandbox`` by image tag, the Python lane's pytest sandbox for a
    Python World. Containers are ``<prefix>*`` (always under ``crucible-lh-d-``)."""
    lang_f = lang_sandbox_factory(parallel=parallel, name_prefix=prefix)
    py_f = sandbox_factory(parallel=parallel, memory="1536m", name_prefix=f"{prefix}py-")

    def get(image: str) -> Any:
        return py_f(image) if image.startswith(f"{IMAGE_REPO}:python-") or not image.startswith(f"{IMAGE_REPO}:") else lang_f(image)

    return get


def epilogue_for(lang: str) -> str:
    p = profile_of(lang)
    words = getattr(p, "plumbing_words", "")
    return EPILOGUE.replace("(conftest.py, sitecustomize.py, usercustomize.py or .pth files)", f"({words})") if words else EPILOGUE


# ---- the real-fix check ----------------------------------------------------------------------------------------------------
def owned_tests(repo: RepoGit, u: Unit, caught: Sequence[str], *, written: bool = False) -> list[str]:
    """The caught tests the fix owns (module doc): in its test files, proposed by the definition diff, or any test of a
    file for which the diff proposes none; or a test whose id names a case or golden file the fix added or changed.
    ``written``: only the tests the fix wrote (a changed definition or case file), not every test of a file it touched,
    which is all a test file whose definitions the profile cannot read gives (h3's test/app.test.ts)."""
    keys = fix_test_names(repo, u)
    files = set(u.test_files)
    # a file proposes tests only through a changed definition that is a caught test: the functions of a case file are its
    # input (black ae5830b7's 650983f edited docstring.py's docstring_with_inline_tabs_* and so never owned
    # test_simple_format[docstring])
    ran = {"::".join(key_of_test(t)) for t in caught}
    proposed = {k.split("::", 1)[0] for k in keys if k in ran}
    # data-driven suites: a case or golden file the fix added or changed (black's ``tests/data/cases/<case>.py``) owns the
    # caught test whose id names it (``test_simple_format[<case>]``)
    data = {Path(p).stem for p in u.files if is_test_path(p) and p not in proposed and len(Path(p).stem) >= 4}
    out = []
    for t in caught:
        f, q = key_of_test(t)
        by_def = f in files and (f"{f}::{q}" in keys or (f not in proposed and not written))
        by_case = bool(data) and any(re.search(rf"(?<![A-Za-z0-9]){re.escape(d)}(?![A-Za-z0-9])", t.split("::", 1)[-1]) for d in data)
        if by_def or by_case:
            out.append(t)
    return out


async def fix_check(repo: RepoGit, b: str, units: Sequence[Unit], sandbox_for: Callable[[str], Any], images: Any, repo_name: str, work: Path, *,
                    lang: str, full_suite: bool = False) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """``rebreak.real_fix_check`` per language (module doc). Returns ``({sha: evidence}, world card of B)``."""
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
    suite = suite_targets(base_b, {}) if full_suite else []
    if full_suite and len(suite) <= (0 if lang == "java" else FULL_SUITE_MAX):  # a JVM full suite per revert: the pool's classes only  # a suite of thousands of files (prettier) overruns the sandbox: the pool's files only
        union = sorted(set(union) | set(suite))
    at_b = await sb.run(base_b, union) if union else None
    b_res = normalize_ids(at_b.results, union) if at_b and at_b.infra == "ok" else {}
    bad_states = ("failed",) if lang in COMPILED else ("failed", "error")
    if union and sum(1 for v in b_res.values() if v == "passed") == 0:  # a World mismatch, not a property of the fixes
        why = f"the suite does not run at B ({(at_b.why if at_b else '')[:160]})"
        return {u.sha: {"ok": False, "why": why} for u in units}, card

    async def one(u: Unit) -> tuple[str, dict[str, Any]]:
        if not files[u.sha]:
            return u.sha, {"ok": False, "why": "its test files are gone at B"}
        with tempfile.TemporaryDirectory(prefix="lhd-fix-") as td:
            tree = Path(td) / "t"
            ok, _ = await asyncio.to_thread(repo.build, b, [], tree, reverse=[u], source_only=True)
            if not ok:
                return u.sha, {"ok": False, "why": "its source revert does not apply at B"}
            tr = await sb.run(tree, union)
        if tr.infra != "ok":
            return u.sha, {"ok": False, "why": f"masked: {tr.why}"}
        res = normalize_ids(tr.results, union)
        caught = sorted(t for t in res if b_res.get(t) == "passed" and res[t] in ("failed", "error"))
        own = owned_tests(repo, u, [t for t in caught if res[t] in bad_states])
        written = owned_tests(repo, u, caught, written=True)
        why = ""
        if not own:
            why = ("its revert only stops tests from building" if any(res[t] == "error" for t in caught) and lang in COMPILED
                   else "no own test fails after the revert")
        # a compiled test package that no longer builds reports every test of it as ``error``: those tests are caught by any
        # revert that breaks the build, which links nothing (mvdan-sh: four zsh fixes "shared" the syntax package's 63 tests)
        unbuilt = [t for t in caught if res[t] == "error"] if lang in COMPILED else []
        return u.sha, {"ok": bool(own), "caught": caught, "caught_unbuilt": unbuilt, "caught_own": own[:30], "caught_written": written,
                       "build_notes": tr.why[:200], "why": why}

    return dict(await asyncio.gather(*[one(u) for u in units])), card


# ---- the task ------------------------------------------------------------------------------------------------------------
def setup_after_reset(world: Any, setup: Callable[[], Any]) -> None:
    """Run the World setup again after every ``world.reset``: the episode resets the World once more before the agent's
    first step (``actors.policy.run_episode``), and a reset re-seeds the workspace without what the setup added (h3: the
    node_modules link gone, so the agent could run no test and tried pnpm install, which the install gate fails)."""
    inner = world.reset

    async def reset(*a: Any, **kw: Any) -> Any:
        obs = await inner(*a, **kw)
        await setup()
        return obs

    world.reset = reset


class DebugTask(RebreakTask):
    """A rebreak task of any language: every bound method runs inside its profile's scope (``langs.base.use``)."""

    generator = NAME

    def __init__(self, lane: Any, mined: Any, window: Mapping[str, Any], **kw: Any):
        with scope(str(window.get("lang") or "python")):
            super().__init__(lane, mined, window, **kw)

    def __getattribute__(self, name: str) -> Any:
        attr = object.__getattribute__(self, name)
        if name.startswith("__") or not inspect.ismethod(attr):
            return attr
        try:
            lang = str(object.__getattribute__(self, "window").get("lang") or "python")
        except AttributeError:
            return attr
        if inspect.iscoroutinefunction(attr):
            @functools.wraps(attr)  # the signature stays visible (``longtask.check_longtask`` reads it)
            async def run_async(*a: Any, **k: Any) -> Any:
                with scope(lang):
                    return await attr(*a, **k)
            return run_async

        @functools.wraps(attr)
        def run(*a: Any, **k: Any) -> Any:
            with scope(lang):
                return attr(*a, **k)
        return run

    @property
    def lang(self) -> str:
        return str(self.window.get("lang") or "python")

    async def open(self, workdir: Path) -> Any:
        world = await super().open(workdir)
        try:
            await world_setup(world._crucible_container, world._crucible_ws, profile_of(self.lang))
        except BaseException:
            await world.close()
            raise
        setup_after_reset(world, lambda: world_setup(world._crucible_container, world._crucible_ws, profile_of(self.lang)))
        return world

    def card(self, **kw: Any) -> Any:
        c = super().card(**kw)
        prov = dict(self.window.get("prompt_provenance") or {})
        return c.model_copy(update={"surface": {**c.surface, "domain": f"{self.lang} library", "prompt_style": "bug report"},
                                    "lineage": {**c.lineage, "lang": self.lang, "prompt_builder": prov.get("builder"),
                                                "intent": (self.window.get(NAME) or {}).get("intent") or task_intent(self.window)}})


# ---- authoring (the builder writes the bug report; the lane checks it) -----------------------------------------------------------
def hidden_names(rec: Mapping[str, Any]) -> set[str]:
    """Leaf names of the hidden tests (every milestone test), as the lint looks for them."""
    with scope(str(rec.get("lang") or "python")) as p:
        out = set()
        for m in rec.get("milestones") or []:
            for t in m["tests"]:
                _, k = key_of_test(t)
                leaf = p.leaf(k) if hasattr(p, "leaf") else k.rsplit("::", 1)[-1]
                out.add(leaf)
    return {n for n in out if n}


def lang_changed_functions(repo: RepoGit, u: Unit, prof: Any) -> set[str]:
    """``rebreak.changed_functions`` through the profile's own definition diff (Python: unchanged)."""
    if prof.name == "python":
        return changed_functions(repo, u)
    return {f"{p}::{q}" for p in u.src_files for q in prof.changed_defs(repo.show(u.parent, p), repo.show(u.sha, p), p)}


def hide_key(test_id: str, changed: Iterable[str], key_of: Callable[[str], tuple[str, str]] = key_of_test) -> tuple[str, str]:
    """``(file, key)`` that hides a visible test in A': its definition (``key_of``), or for a data-driven case whose
    parameter names a case file the fixes changed (black's ``tests/test_format.py::test_simple_format[docstring]`` and
    ``tests/data/cases/docstring.py``) that case file, removed whole (key ``*``): stripping ``test_simple_format`` would
    hide every case. The grader writes B's file back."""
    m = re.search(r"\[([^\]]+)\]$", test_id)
    hit = sorted(p for p in changed if is_test_path(p) and Path(p).stem == m.group(1)) if m else []
    return (hit[0], "*") if hit else key_of(test_id)


def visible_test_changes(rec: Mapping[str, Any], repo: RepoGit) -> list[str]:
    """Strip mode's residue, re-derived from a stored window: test files the group's fixes changed that the start tree
    does not hide (not among the grader's files, not stripped as visible tests)."""
    info = rec.get(NAME) or {}
    hidden = info.get("hidden") or {}
    if hidden.get("mode") != "strip":
        return []
    from crucible.factory.generators.base import changed_paths, commit_meta

    shown = set(rec.get("tests_b") or {}) | set(hidden.get("visible_stripped") or {}) | set(hidden.get("table_entries_stripped") or {})
    out: set[str] = set()
    with scope(str(rec.get("lang") or "python")) as prof:
        for f in info.get("fixes") or []:
            meta = commit_meta(repo.dir, f["sha"])
            out |= {p for p in changed_paths(repo.dir, meta["parent"], f["sha"]) if prof.is_test_file(p) and p not in shown}
    return sorted(out)


def faulty_names(rec: Mapping[str, Any], repo: RepoGit) -> set[str]:
    """Names of the functions the group's fixes changed (the lint refuses them in a prompt)."""
    names: set[str] = set()
    with scope(str(rec.get("lang") or "python")) as prof:
        for u in rec.get("units") or []:
            unit = Unit(**u)
            for q in lang_changed_functions(repo, unit, prof):
                leaf = re.split(r"[.:]+", q.split("::", 1)[-1])[-1]
                if len(leaf) >= 4:
                    names.add(leaf)
    return names


# test names per language the window lane's ``test_\w+`` rule does not see (Go ``TestX``/``ExampleX``, JUnit ``testX(``)
LANG_TEST_NAME = re.compile(r"\b(?:Test|Example|Fuzz|Benchmark)[A-Z_]\w*|\btest[A-Z]\w*\s*\(")


def lint_debug(text: str, *, faulty: set[str], hidden: set[str]) -> list[str]:
    """``windows.lint_prompt`` plus the debug lane's rules: no function a fix changed (by name), no hidden test name, no
    test name of any language, no version number."""
    out = lint_prompt(text)
    out += [f"lint test_name: {m.group(0)!r}" for m in LANG_TEST_NAME.finditer(text)]
    for n in sorted(faulty):
        if re.fullmatch(r"[a-z]+", n) and len(n) < 10:
            continue  # a plain lower-case word ("command", "reset") names nothing a user would not write anyway
        if re.search(rf"(?<![\w$]){re.escape(n)}(?![\w$])", text):
            out.append(f"lint faulty_name: {n!r} (a function the fix changed: describe what the user sees instead)")
    for n in sorted(hidden):
        if len(n) >= 6 and n in text:
            out.append(f"lint hidden_test: {n!r} (a hidden test's name)")
    from crucible.factory.generators.base import VERSION_RX

    out += [f"lint version: {m.group(0)!r}" for m in VERSION_RX.finditer(text)]
    return out


def author_debug(rec: Mapping[str, Any], authored: Mapping[str, Any], repo: RepoGit, *, base_dir: Path | None = None) -> dict[str, Any]:
    """The prompt from the builder's bug report ``authored`` = ``{builder, opening, requirements: {milestone id: text},
    closing, repro?}``: one requirement per milestone, the regression requirement when the record has regression tests,
    the reproduction block when the builder gave ``repro`` (``debug_repro``: easing knob K4, executed by gate S7r), the
    profile's epilogue; then the lint (window lane + ``lint_debug``, and ``debug_repro.lint_repro`` per repro: never a
    hidden test) and the two-way ledger. ``findings`` is empty when the prompt is publishable. ``base_dir`` (A'): the
    copied-hidden-line rule then looks only at lines the agent's tree does not show."""
    from crucible.factory.generators import debug_repro as dr
    from crucible.factory.lanes.executable import REGRESSION_REQ, REGRESSION_TEXT

    problems: list[str] = []
    if not str(authored.get("builder") or ""):
        problems.append("provenance: no builder recorded")
    got = {str(k): " ".join(str(v).split()) for k, v in (authored.get("requirements") or {}).items()}
    ids = [m["id"] for m in rec["milestones"]]
    problems += [f"unknown_requirement_id: {k}" for k in got if k not in ids]
    problems += [f"unmapped_milestone: {k} has no requirement" for k in ids if not got.get(k)]
    reqs = [{"id": f"R{i}", "milestone": mid, "text": got[mid], "kind": "behaviour"} for i, mid in enumerate(ids, 1) if got.get(mid)]
    opening, closing = str(authored.get("opening") or "").strip(), str(authored.get("closing") or "").strip()
    if not opening:
        problems.append("author.opening is empty")
    text = "\n".join([opening, closing, *[r["text"] for r in reqs]])
    faulty, hidden = faulty_names(rec, repo), hidden_names(rec)
    problems += lint_debug(text, faulty=faulty, hidden=hidden)
    repros, rp = dr.parse_repros(authored, ids)
    problems += rp
    if repros:
        paths, lines = dr.hidden_paths(rec), dr.hidden_lines(rec, base_dir)
        for r in repros.values():
            problems += dr.lint_repro(r, faulty=faulty, hidden=hidden, paths=paths, lines=lines)
    block = dr.render(repros, {r["milestone"]: r["id"] for r in reqs if r.get("milestone")})
    regress = bool(set(rec.get("p2p") or []) | set(rec.get("collateral") or []))
    if regress:
        reqs.append({"id": REGRESSION_REQ, "milestone": None, "text": REGRESSION_TEXT, "kind": "side_effect_scope"})
    tail = "\n\n".join(x for x in (block, closing, REGRESSION_TEXT if regress else "") if x)
    prompt = compose_prompt(opening, [r for r in reqs if r["id"] != REGRESSION_REQ], tail, heading=HEADING, epilogue=epilogue_for(str(rec.get("lang") or "python")))
    spans = {r["id"]: list(loc) for r in reqs if (loc := locate(prompt, r["text"]))}
    _, ledger = window_ledger(rec, reqs, spans, prompt)
    problems += ledger
    return {"prompt": prompt, "requirements": reqs, "spans": spans, "checkset": window_checkset(rec, reqs), "findings": problems,
            "repro": {m: asdict(r) for m, r in repros.items()}, "repro_status": "unverified" if not repros else "unexecuted",
            "provenance": {"builder": str(authored.get("builder") or ""), "kind": "symptom bug report", "sha256": hashlib.sha256(prompt.encode()).hexdigest()}}


def builder_calibration(window_id: str, split: str, records: Mapping[str, Mapping[str, Any]], *, model: str | None = None, per_level: int = 3) -> str:
    """The calibration block of the builder's brief (``factory.calibration.brief_block``; rl-env-use's drafter
    calibration, MIT): training tasks the target student found too easy, in band, too hard or unfair, with their measured
    numbers from the S15 records, this window's own numbers when it is a measured training task, and the easing knobs
    K1-K6. Held-out records never enter it; a held-out window gets none."""
    from crucible.factory.calibration import brief_block
    from crucible.factory.student import TARGET_MODEL

    return brief_block(window_id, split, records, model=model or TARGET_MODEL, per_level=per_level)


# ---- intent (00 §7: the task fails for the intended reason) -------------------------------------------------------------------
SKILLS: dict[str, dict[str, str]] = {
    "multi_fault_localization": {
        "behaviour": "finds several faults from user-visible symptoms alone, across files and modules, without a failing test to start from",
        "failure_mode": "fixes the first fault found and stops, or edits code near the symptom instead of the faulty module",
        "evidence": "milestone vector at each checkpoint (progress.jsonl): some milestones never pass; files the agent edited vs the faults' "
                    "source files (task lineage, never shown to the agent)"},
    "interacting_faults": {
        "behaviour": "keeps debugging after a correct fix whose symptom persists because a second fault masks it (an executed requires edge)",
        "failure_mode": "reverts or abandons a correct fix because its tests still fail, or declares success after the first fix",
        "evidence": "a prerequisite milestone passes while its dependent never does; the dependent's fault file untouched; a revert of a "
                    "change that had made a milestone pass in the checkpoint series; STATUS.json claims vs the hidden tests"},
    "reproduce_from_report": {
        "behaviour": "turns a symptom report into a reproduction and checks each fix against it (the checks are hidden, none fails visibly)",
        "failure_mode": "edits without ever running a reproduction, or claims a fix it never exercised",
        "evidence": "commands in the trail before the first edit (a test, a script or a REPL run touching the reported API); STATUS.json "
                    "probabilities vs verified milestones (Brier)"},
    "regression_safety": {
        "behaviour": "fixes the faults without breaking behaviour the existing suite covers",
        "failure_mode": "a fix that passes its milestone but fails pass-to-pass tests",
        "evidence": "the regression check's failing tests in the GradeReport; whether the agent ran the existing suite before finishing"},
}


def fairness_flags(rec: Mapping[str, Any], faulty: Iterable[str] = ()) -> list[str]:
    """What a reviewer must look at before the task trains or benchmarks (``00`` §5.8; never a gate by itself): a hidden
    milestone test named after a function a fix changed (it likely calls that internal function directly, so a correct
    fix made elsewhere fails it), and the fixes' test-file changes left visible in the agent's tree (strip mode)."""
    out = []
    lang = str(rec.get("lang") or "python")
    # exported Go names are the package's public API; a test of one checks behaviour a correct fix must keep
    norm = {re.sub(r"[^a-z0-9]", "", n.lower()): n for n in faulty
            if len(n) >= 6 and not (lang == "go" and n[:1].isupper())}
    for m in rec.get("milestones") or []:
        for t in m["tests"]:
            name = t.split("::")[-1].split("[")[0].split("/")[0]
            leaf = re.sub(r"^(test|example|benchmark)", "", re.sub(r"[^a-z0-9]", "", name.lower()))
            hit = next((n for k, n in norm.items() if k and leaf.startswith(k)), None)
            if hit:
                out.append(f"internal_function_check: {m['id']} test {name} is named after the changed function {hit}")
    vis = ((rec.get(NAME) or {}).get("hidden") or {}).get("visible_test_changes") or []
    if vis:
        out.append(f"visible_test_data: the fixes' changes to {', '.join(vis)} stay in the agent's tree (strip mode)")
    return sorted(set(out))


def silent_units(rec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The window's reverted fixes without a milestone of their own (strip mode: every check of theirs was visible). They
    stay reverted in A', so the report must describe them wherever another milestone's tests check them."""
    own = {m.get("unit") for m in rec.get("milestones") or []}
    return [u for u in rec.get("units") or [] if u.get("sha") not in own]


async def disclosure_check(task: Any) -> dict[str, Any]:
    """Executed check for silent faults (S15 at 300 calls, 2026-10-20). Grade B without the silent units: the oracle's unit
    steps minus those units (``task.oracle``, through the task's own grader, any language). A milestone that fails there
    needs a reverted fix that has no milestone, and its report must describe that fix. In excelize, every student and
    frontier episode failed such a milestone, and both judges, who never see the hidden tests, labelled the failures
    skill gaps. The result goes to review: a builder may already have folded the fix into a requirement's text (urfave-cli's
    R3 does), so it is never a gate by itself.

    Returns ``{measured, silent, needs, milestones, why}``. ``needs`` maps a milestone id to the silent units' subjects
    when it fails without them."""
    rec = task.window
    silent = silent_units(rec)
    out: dict[str, Any] = {"task": str(task.task_id), "silent": [{"sha": u["sha"], "subject": u.get("subject")} for u in silent], "needs": {},
                           "milestones": None, "measured": True, "why": ""}
    if not silent:
        out["why"] = "every reverted fix has a milestone"
        return out
    skip = {u["sha"] for u in silent}
    steps = list(task.oracle)
    if not steps or any(s.get("function_name") != "apply_unit" for s in steps):
        return {**out, "measured": False, "why": "the oracle is not unit steps (no clone, or a simplified start)"}
    rep, _ = await task.grade([s for s in steps if (s.get("arguments") or {}).get("unit") not in skip])
    meta = rep.meta or {}
    if (rep.infra or {}).get("status") != "ok" or meta.get("apply_errors"):
        return {**out, "measured": False, "why": "B without the silent units does not apply or grade: " + "; ".join(meta.get("apply_errors") or [])[:300]}
    status = dict(meta.get("milestones") or {})
    out["milestones"] = status
    out["needs"] = {m["id"]: [u.get("subject") or u["sha"] for u in silent] for m in rec["milestones"] if not status.get(m["id"])}
    out["why"] = ("every milestone passes without the silent units" if not out["needs"] else
                  f"{', '.join(out['needs'])} fail(s) without the silent units: the report must describe them")
    return out


def task_intent(rec: Mapping[str, Any], faulty: Iterable[str] = ()) -> dict[str, Any]:
    """The task's intent card (``00`` §7): skills (``SKILLS``), language, why it is long-horizon, its milestones, the
    expected failure modes and the trace evidence that attributes a failure to the skill. Derived from the executed
    window (faults, files, edges), never from the prompt; stored with the record and on the TaskCard lineage.
    ``fairness_flags`` lists what a reviewer must check (``faulty``: the names of the functions the fixes changed)."""
    lang = str(rec.get("lang") or "python")
    ms = list(rec.get("milestones") or [])
    units = {u["sha"]: u for u in rec.get("units") or []}
    with scope(lang) as prof:
        files = {m["id"]: sorted(p for p in units.get(m["unit"], {}).get("files") or [] if prof.is_source(p)) for m in ms}
    all_files = sorted({f for v in files.values() for f in v})
    modules = sorted({f.rsplit("/", 1)[0] if "/" in f else "." for f in all_files})
    edges = [(m["id"], r) for m in ms for r in m.get("requires") or []]
    skills = ["reproduce_from_report"]
    if len(ms) >= 2:
        skills.insert(0, "multi_fault_localization")
    if edges:
        skills.insert(1, "interacting_faults")
    if set(rec.get("p2p") or []) | set(rec.get("collateral") or []):
        skills.append("regression_safety")
    depth = 0
    req = {m["id"]: list(m.get("requires") or []) for m in ms}

    def d(x: str, seen: frozenset[str] = frozenset()) -> int:
        return 1 + max((d(y, seen | {x}) for y in req.get(x, []) if y not in seen), default=0)

    depth = max((d(m["id"]) for m in ms), default=0)
    level = "hard" if len(ms) >= 3 and len(all_files) >= 2 else "easy" if len(ms) <= 2 else "medium"
    return {"skills": skills, "lang": lang, "level": level, "faults": len(ms), "source_files": all_files, "modules": modules,
            "why_long": (f"{len(edges)} executed requires edge(s) ({', '.join(f'{a} needs {b}' for a, b in edges)}): a dependent fault's checks "
                         f"fail until its prerequisite is fixed; prerequisite chain depth {depth}"),
            "milestones": [{"id": m["id"], "requires": list(m.get("requires") or []), "tests": len(m["tests"]), "files": files[m["id"]]} for m in ms],
            "expected_failures": {k: SKILLS[k]["failure_mode"] for k in skills},
            "trace_evidence": {k: SKILLS[k]["evidence"] for k in skills},
            "attribution_rule": "a failure counts toward a skill gap only when its trace evidence is present; S15 needs >= 70% of failures so attributed",
            "fairness_flags": fairness_flags(rec, faulty)}


def failure_skill(finding: Mapping[str, Any], intent: Mapping[str, Any], *, ran_before_edit: bool | None = None,
                  honest: Sequence[str] = ()) -> dict[str, Any]:
    """Which of the task's intent skills (``SKILLS``) a failing episode's trace evidence points at (``00`` §7: a failure
    counts toward the intent only when its trace evidence is present). ``finding``: a ``failures.FailureFinding.to_json()``;
    ``intent``: the task's card (``task_intent``); ``ran_before_edit``: a test or reproduction command ran before the first
    edit; ``honest``: the episode's honest-trace fact kinds (``traceaudit.honest_trace_facts``).

    In order (the first that holds): a routed failure (World, harness, limits, infrastructure, a non-skill cause) has no
    skill; **interacting_faults** when a milestone was lost after being reached, or a prerequisite of an executed edge
    passes while its dependent never does; **regression_safety** when every milestone passes but the run still fails (the
    existing suite broke); **reproduce_from_report** when the work was claimed done or verified without the evidence
    (``claimed_done_without_progress``, ``claimed_unrun_tests``) or no reproduction ran before the first edit and nothing
    passed; **multi_fault_localization** when milestones are missed (some faults never found). ``on_intent``: the skill is
    one the task's card names."""
    skills = list(intent.get("skills") or [])
    if finding.get("routed"):
        return {"skill": None, "on_intent": False, "why": f"routed to {finding.get('routed')} ({finding.get('mode')}, cause {finding.get('cause')})"}
    req = {m["id"]: list(m.get("requires") or []) for m in intent.get("milestones") or []}
    missed = set(finding.get("milestones_missed") or [])
    lost = list(finding.get("milestones_lost") or [])
    all_ms = set(req)
    passing = all_ms - missed
    skill, why = None, ""
    edge = next(((m, r) for m, rs in req.items() for r in rs if r in passing and m in missed), None)
    if lost or finding.get("mode") == "lost_progress":
        skill, why = "interacting_faults", f"milestone(s) lost after being reached: {lost or 'Phi fell and never recovered'}"
    elif edge:
        skill, why = "interacting_faults", f"prerequisite {edge[1]} passes while its dependent {edge[0]} never does"
    elif all_ms and not missed:
        skill, why = "regression_safety", "every milestone passes but the run fails: behaviour the existing suite covers broke"
    elif finding.get("mode") == "claimed_done_without_progress" or "claimed_unrun_tests" in honest:
        skill, why = "reproduce_from_report", "the work was claimed done or verified without the evidence"
    elif ran_before_edit is False and missed == all_ms:
        skill, why = "reproduce_from_report", "no reproduction ran before the first edit and no milestone ever passed"
    elif missed:
        skill, why = "multi_fault_localization", f"faults never found: {sorted(missed)}"
    if skill is None:
        return {"skill": None, "on_intent": False, "why": "no intent evidence in the trace"}
    return {"skill": skill, "on_intent": skill in skills, "why": why}


def intent_cards() -> list[Any]:
    """The family-level ``cards.IntentCard`` per skill (``candidate``: no model failure observed yet; S13/S14 add the
    evidence and a human approves)."""
    from crucible.factory.cards import IntentCard

    return [IntentCard(id=f"intent-debug-{k.replace('_', '-')}", behaviour=v["behaviour"], failure_mode=v["failure_mode"],
                       difficulty_axes=["faults", "modules", "edge depth", "language"], coverage={"behaviour": k, "surface": "debugging, any language"})
            for k, v in SKILLS.items()]


# ---- the generator ------------------------------------------------------------------------------------------------------------
def group_id(lang: str, name: str, group: Sequence[Unit]) -> str:
    """A mined candidate's id: the hash of its fix shas in prober order, so the same fixes at any B give the same id."""
    return f"db-{lang}-{name}-{hashlib.sha256('|'.join(u.sha for u in group).encode()).hexdigest()[:8]}"


class Debug(Rebreak):
    """The debugging lane over local clones (``Generator``): ``clones`` name -> path, ``langs`` name -> profile name."""

    name = NAME

    def __init__(self, root: Path, clones: Mapping[str, Path], *, langs_of: Mapping[str, str], funnel: Funnel, lease: EvalLease, split: str = "train",
                 since: str = SINCE, max_size: int = 4, max_groups_per_repo: int = 4, max_pool: int = 40, full_suite: bool = True,
                 parallel: int = 3, log: Callable[[dict], None] | None = None):
        self.langs_of = dict(langs_of)
        self.images_by_lang: dict[str, Any] = {}
        sb = debug_sandboxes(parallel=parallel)
        super().__init__(root, clones, funnel=funnel, lease=lease, images=None, sandboxes=sb, max_size=max_size, max_groups_per_repo=max_groups_per_repo,
                         require_interaction=True, full_suite=full_suite, log=log)
        self.store = WindowStore(Path(root) / "store" / split, split=split)
        self.since, self.max_pool = since, max_pool
        self.views: dict[str, dict[str, str]] = {}
        self.oracle_views: dict[str, dict[str, str]] = {}  # A' tests on B's source (``precheck``)

    def images_for(self, lang: str) -> Any:
        if lang not in self.images_by_lang:
            self.images_by_lang[lang] = PyDebugImages(parallel=2) if lang == "python" else profile_of(lang).images(parallel=2)
        return self.images_by_lang[lang]

    def lane(self) -> GenLane:
        if self._lane is None:
            self._lane = GenLane(self.store, generator=NAME, task_cls=DebugTask, repo_dirs={k: str(v) for k, v in self.clones.items()},
                                 sandboxes=self.sandboxes)
            self._lane.world_prefix = f"{PREFIX}world-"
        return self._lane

    def note(self, cand: str, stage: str, ok: bool, reason: str = "", **ev: Any) -> None:
        self.funnel.note(NAME, cand, stage, ok, reason, **ev)

    async def candidates_for(self, name: str, *, since: str | None = None) -> list[Candidate]:
        """The pool = every fix since ``since`` (newest ``max_pool``) on the default branch. B is the head, then (for the
        fixes not yet grouped) the first-parent commit ``B_STEPS`` days earlier: a fix whose revert no longer applies at
        the head often still applies closer to its own date. At each B the pool is executed (``fix_check``) and grouped as
        rebreak groups it (an executed interaction in every group); a fix joins at most one group."""
        lang = self.langs_of[name]
        with scope(lang):
            repo_dir = self.clones[name]
            from crucible.factory.generators.base import git, out

            head = out(git(repo_dir, "rev-parse", "HEAD"))
            head_date = out(git(repo_dir, "log", "-1", "--format=%cI", head))
            since = since or self.since
            fixes = (await asyncio.to_thread(fix_units, repo_dir, head, since=since))[-self.max_pool:]
            for u in fixes:
                self.note(f"{lang}/{name}:{u.sha[:10]}", "S1_source", True, "", subject=u.subject[:120], lang=lang)
            if len(fixes) < 2:
                self.note(f"{lang}/{name}", "S1_source", False, f"{len(fixes)} fix unit(s) since {since}", lang=lang)
                return []
            import datetime as dt

            points = [(head, "HEAD", head_date)]
            for days in B_STEPS:
                cut = (dt.datetime.fromisoformat(head_date) - dt.timedelta(days=days)).isoformat()
                c = out(git(repo_dir, "rev-list", "--first-parent", "-1", f"--before={cut}", head, check=False))
                if c and cut[:10] > since and c not in {x[0] for x in points}:
                    points.append((c, f"HEAD-{days}d", out(git(repo_dir, "log", "-1", "--format=%cI", c))))
            out_: list[Candidate] = []
            used: set[str] = set()
            stats = {"kept": 0, "interacting_fixes": 0}
            for b, b_ref, b_date in points:
                pool = [u for u in fixes if u.sha not in used and git(repo_dir, "merge-base", "--is-ancestor", u.sha, b, check=False).returncode == 0]
                if len(pool) < 2 or len(out_) >= self.max_groups:
                    continue
                got = await self._groups_at(name, lang, b, b_ref, b_date, pool, since, stats)
                for c in got:
                    if len(out_) < self.max_groups:
                        out_.append(c)
                        used |= set(c.params["fixes"])
            self.note(f"{lang}/{name}", "S1_groups", bool(out_), "" if out_ else "no interacting group", lang=lang, fixes=len(fixes),
                      groups=len(out_), points=[x[1] for x in points], **stats)
            return out_

    async def _groups_at(self, name: str, lang: str, b: str, b_ref: str, b_date: str, fixes: list[Unit], since: str,
                         stats: dict[str, int]) -> list[Candidate]:
        kept, need, card = await self._executed(name, lang, b, b_ref, fixes)
        repo = RepoGit(self.clones[name])
        funcs = {u.sha: lang_changed_functions(repo, u, profile_of(lang)) for u in kept}
        stats["kept"] += len(kept)
        stats["interacting_fixes"] += len(need)
        out_: list[Candidate] = []
        for g in group_fixes(kept, funcs, need=need, max_size=self.max_size):
            if not any(need.get(u.sha, set()) & {x.sha for x in g} for u in g):
                self.note(group_id(lang, name, g), "S1_executed", False, "no executed interaction at B (no test caught by two reverts)", lang=lang,
                          fixes=[u.sha[:10] for u in g])
                continue
            out_.append(self._candidate(name, lang, g, need, card, b=b, b_ref=b_ref, b_date=b_date, since=since))
        return out_

    async def subset(self, name: str, shas: Sequence[str], *, b: str = "HEAD", since: str | None = None, cid: str = "",
                     parent: str = "", hide: str = "", independent: bool = False, hide_tests: Sequence[str] = ()) -> Candidate:
        """The candidate of exactly the fixes ``shas`` (sha prefixes of fix units on B's first-parent path since ``since``)
        at ``b``: ``candidates_for``'s executed check, edges, prober order and params, without its grouping. A mined task is
        rebuilt from its fixes; an eased one (TASKS_CREDIT_AND_EASING §5.5) is its parent's fixes minus the dropped one, which
        stays applied in A'. ``cid`` names it (default the group hash, as mined); ``parent`` is recorded in the window.
        ``hide`` = ``strip`` builds A' in strip mode only: revert mode restores the old version of a test a fix changed, and
        an old assertion of the faulty behaviour (go-toml-e1's ``TestMarshalStringEscapes`` expecting ``\\u00FF``) then fails
        on a correct fix in the agent's own tree.
        ``independent``: the fixes need not interact (the lead's call, 2026-10-23, for tasks of separate bugs); the build
        then keeps a window whose milestones need no earlier one and S6 passes it without an edge.
        ``hide_tests`` (test ids, ``file::Test``): visible tests stripped from A' from the first round, as round 2 strips
        the milestone tests that fail there. Revert mode keeps the table drivers a strip build loses, but a test a fix
        changed comes back with its old assertions; one that still passes on A' and fails on the fixed source (mvdan-sh's
        ``TestNodeEndPos`` with its ``// TODO: want 22`` offsets) is named here.
        ``ValueError`` when a sha is not such a fix, a fix's own tests are not caught at B, or no two of the fixes interact
        (unless ``independent``)."""
        if hide not in ("", "revert", "strip"):
            raise ValueError(f"hide {hide!r} is not revert or strip")
        lang = self.langs_of[name]
        with scope(lang):
            from crucible.factory.generators.base import git, out

            repo_dir = self.clones[name]
            since = since or self.since
            r = git(repo_dir, "rev-parse", "--verify", "--quiet", f"{b}^{{commit}}", check=False)
            if r.returncode != 0:
                raise ValueError(f"B {b!r} is not a commit of {name}")
            b_full, head = out(r), out(git(repo_dir, "rev-parse", "HEAD"))
            pool = await asyncio.to_thread(fix_units, repo_dir, b_full, since=since)
            picked: list[Unit] = []
            for s in shas:
                hits = [u for u in pool if u.sha.startswith(s)]
                if len(hits) != 1:
                    raise ValueError(f"{s}: {'no' if not hits else 'more than one'} fix unit on B's first-parent path since {since}")
                if hits[0] not in picked:
                    picked.append(hits[0])
            if len(picked) < 2:
                raise ValueError("a task needs at least two fixes")
            b_ref = "HEAD" if b_full == head else "pinned"
            kept, need, card = await self._executed(name, lang, b_full, b_ref, picked)
            if lost := [u.sha[:10] for u in picked if u not in kept]:
                raise ValueError(f"not a real fix at B (no test of its own fails when it alone is reverted): {lost}")
            g = order_fixes(sorted(picked, key=pool.index), need)
            if not independent and not any(need.get(u.sha, set()) & {x.sha for x in g} for u in g):
                raise ValueError("no executed interaction at B (no test caught by two reverts)")
            b_date = out(git(repo_dir, "log", "-1", "--format=%cI", b_full))
            return self._candidate(name, lang, g, need, card, b=b_full, b_ref=b_ref, b_date=b_date, since=since, cid=cid,
                                   **({"parent": parent} if parent else {}), **({"hide_modes": [hide]} if hide else {}),
                                   **({"allow_independent": True} if independent else {}),
                                   **({"hide_tests": sorted(set(hide_tests))} if hide_tests else {}))

    async def _executed(self, name: str, lang: str, b: str, b_ref: str, fixes: Sequence[Unit]
                        ) -> tuple[list[Unit], dict[str, set[str]], dict[str, Any]]:
        """``fix_check`` of ``fixes`` at B (cached per pool): the real fixes the lease allows, their executed interaction
        edges and B's World card. Every fix's source record joins ``records``."""
        repo = RepoGit(self.clones[name])
        work = self.root / "work" / name / b[:10]
        work.mkdir(parents=True, exist_ok=True)
        key = "|".join(u.sha for u in fixes) + ("|full" if self.full_suite else "") + "|unbuilt|written"  # evidence before caught_unbuilt / caught_written is run again
        cache = work / f"evidence-{hashlib.sha256(key.encode()).hexdigest()[:10]}.json"
        if cache.exists():
            evid, card = json.loads(cache.read_text())
        else:
            evid, card = await fix_check(repo, b, fixes, self.sandboxes, self.images_for(lang), name, work, lang=lang, full_suite=self.full_suite)
            cache.write_text(json.dumps([evid, {k: v for k, v in card.items() if k != "build_log"}], default=str))
        if not card.get("ok"):
            self.note(f"{lang}/{name}", "S1_executed", False, "the World image of B does not build", lang=lang, log=str(card.get("build_log", ""))[-600:])
        kept = []
        for u in fixes:
            ev = evid[u.sha]
            rec = SourceRecord("real_fix", name, sha=u.sha, title=u.subject, text=u.body[:1500], created=u.date,
                               evidence={"executed": True, "b": b, "b_ref": b_ref, "lang": lang, **ev})
            self.lease.take(rec)
            self.records.append(rec)
            ok = bool(ev["ok"] and rec.lease["ok"])
            if ok or b_ref == "HEAD":  # an earlier B only adds what the head could not execute
                self.note(f"{lang}/{name}:{u.sha[:10]}", "S1_executed", ok, ev.get("why") or rec.lease.get("reason", ""), lang=lang, b_ref=b_ref)
            if ok:
                kept.append(u)
        return kept, interaction_edges({u.sha: evid[u.sha] for u in kept}), card

    def _candidate(self, name: str, lang: str, g: Sequence[Unit], need: Mapping[str, set[str]], card: Mapping[str, Any], *, b: str, b_ref: str,
                   b_date: str, since: str, cid: str = "", **params: Any) -> Candidate:
        recs = {r.sha: r for r in self.records if r.repo == name and r.evidence.get("b") == b}
        return Candidate(cid or group_id(lang, name, g), NAME, name, sources=[recs[u.sha].to_json() for u in g],
                         params={"lang": lang, "b": b, "b_ref": b_ref, "b_date": b_date, "b_version": None, "fixes": [u.sha for u in g],
                                 "image_b": card.get("image"), "tags": len(version_tags(self.clones[name])), "cutoff": since,
                                 "fix_dates": {u.sha: u.date for u in g},
                                 "contamination": "post-release" if min(u.date[:10] for u in g) >= SINCE else "fallback",
                                 "needs": {u.sha: sorted(need.get(u.sha, set()) & {x.sha for x in g}) for u in g}, **params})

    def synthesize(self, cand: Candidate, *, mode: str = "revert", extra_hide: Mapping[str, set[str]] | None = None
                   ) -> tuple[Window | None, dict[str, Any]]:
        """A'..B' (module doc): rebreak's chain with test data reverted with the source, hidden tests stripped per
        language, the fixes' changelog lines removed."""
        lang = cand.params["lang"]
        with scope(lang) as prof:
            repo_dir = self.clones[cand.repo]
            repo = RepoGit(repo_dir)
            b = cand.params["b"]
            fixes = [commit_meta(repo_dir, s) for s in cand.params["fixes"]]
            info: dict[str, Any] = {"dropped": [], "lang": lang}
            for f in fixes:
                f["files"] = changed_paths(repo_dir, f["parent"], f["sha"])

            def reverted(f: Mapping[str, Any], tests: bool) -> list[str]:  # source + non-code test data (+ test code)
                return [p for p in f["files"] if prof.is_source(p) or (is_test_path(p) and not PROSE.search(p) and (tests or not prof.is_test_file(p)))]

            # mode "revert": the fixes' test hunks reverted too (a table row, a changed expectation, a new test: all gone from
            # A'); mode "strip" when that conflicts: only the changed test definitions removed (``profile.strip_tests``)
            while len(fixes) >= 2:
                steps = [(full_patch(repo_dir, f["parent"], f["sha"], reverted(f, mode == "revert")), True) for f in reversed(fixes)]
                trees, bad = apply_chain(repo_dir, b, steps)
                if bad is None:
                    break
                if mode == "revert":
                    mode = "strip"
                    continue
                drop = list(reversed(fixes))[bad]
                info["dropped"].append({"sha": drop["sha"], "why": "its revert conflicts with the fixes after it"})
                fixes = [f for f in fixes if f["sha"] != drop["sha"]]
            if len(fixes) < 2:
                return None, info
            info["hide_mode"] = mode
            tb = tree_of(repo_dir, b)
            src_forward = [*reversed(trees), tb]
            tests_b: dict[str, str] = {}
            keys: dict[str, dict[str, set[str]]] = {}
            prose_added: dict[str, dict[str, list[str]]] = {}
            for f in fixes:
                u = Unit(sha=f["sha"], parent=f["parent"], merge=False, date=f["date"], subject=f["subject"], files=f["files"])
                names = fix_test_names(repo, u)
                for p in u.test_files:
                    k = {q.split("::", 1)[1] for q in names if q.startswith(p + "::")}
                    if k and (txt := repo.show(b, p)) is not None:
                        keys.setdefault(f["sha"], {})[p] = k
                        tests_b[p] = txt
                for p in f["files"]:
                    if PROSE.search(p) and not prof.is_source(p):
                        diff = full_patch(repo_dir, f["parent"], f["sha"], [p]).decode("utf-8", "replace")
                        prose_added.setdefault(f["sha"], {})[p] = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
            prose_b = {p: t for p in {q for d in prose_added.values() for q in d} if (t := repo.show(b, p)) is not None}
            # a changelog written at release time cites the fixes although their diffs never added the line (mvdan-sh, h3, unstorage)
            logs_b = {p: t for p in out(git(repo_dir, "ls-tree", "-r", "--name-only", b)).splitlines() if CHANGELOG_FILE.search(p) and (t := repo.show(b, p)) is not None}
            refs = {f["sha"]: fix_refs(f["subject"], f["body"], f["sha"]) | ({sr} if (sr := subject_ref(f["subject"])) else set()) for f in fixes}
            cited: dict[str, int] = {}
            tables: dict[str, int] = {}
            _added: dict[tuple[str, str], list[list[str]]] = {}

            def table_added(f: Mapping[str, Any], p: str) -> list[list[str]]:
                """The fix's runs of consecutive added lines in ``p`` (a table entry it added or changed)."""
                if (f["sha"], p) not in _added:
                    runs: list[list[str]] = [[]]
                    for ln in full_patch(repo_dir, f["parent"], f["sha"], [p]).decode("utf-8", "replace").splitlines():
                        if ln.startswith("+") and not ln.startswith("+++"):
                            runs[-1].append(ln[1:])
                        elif runs[-1]:
                            runs.append([])
                    _added[(f["sha"], p)] = [r for r in runs if r]
                return _added[(f["sha"], p)]

            extra = self._locate_defs(repo, b, prof, extra_hide or {})
            other_tests = sorted({p for f in fixes for p in f["files"] if prof.is_test_file(p) and p not in tests_b and p not in extra})
            forward = []
            for k, t in enumerate(src_forward):
                pending = fixes[k:]
                files: dict[str, str | None] = {}
                for p, ks in extra.items() if pending else ():  # tests that fail visibly in the agent's tree: hidden until B'
                    if (mode == "strip" and p in tests_b) or "*" in ks:
                        continue  # stripped together with the fixes' own tests below; a whole case file removed below
                    cur = repo.show(t, p)
                    if cur is not None:
                        files[p] = prof.strip_tests(cur, ks, p)
                for p, txt in tests_b.items() if mode == "strip" else ():
                    ks = set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in pending]) if pending else set()
                    if ks:
                        files[p] = prof.strip_tests(txt, ks | (extra.get(p, set()) if pending else set()), p)
                    tbl = getattr(prof, "strip_table_entries", None)
                    if tbl and pending and files.get(p, txt) is not None:  # Go: the fixes' table entries, not only their functions
                        cut, n = tbl(files.get(p, txt), [r for f in pending for r in table_added(f, p)])
                        if n:
                            files[p] = cut
                            tables.setdefault(p, n)  # A' first: every fix still pending
                for p in (other_tests if mode == "strip" and pending else ()):  # a table-only test file the fixes changed
                    tbl = getattr(prof, "strip_table_entries", None)
                    cur = files.get(p, repo.show(b, p))
                    if tbl and cur is not None:
                        cut, n = tbl(cur, [r for f in pending for r in table_added(f, p)])
                        if n:
                            files[p] = cut
                            tables.setdefault(p, n)
                for p, ks in extra.items() if pending else ():  # a data-driven case (``*``): its case file is not in A'
                    if "*" in ks and repo.show(t, p) is not None:
                        files[p] = None
                for p, txt in prose_b.items():
                    added = [ln for f in pending for ln in prose_added.get(f["sha"], {}).get(p, [])]
                    if added:
                        files[p] = strip_added_lines(txt, added)
                for p, txt in logs_b.items() if pending else ():
                    cut, n = strip_ref_entries(files.get(p, txt), set().union(*[refs[f["sha"]] for f in pending]))
                    if n:
                        files[p] = cut
                        cited.setdefault(p, n)  # A' first: every fix still pending
                forward.append(with_files(repo_dir, t, files))
            msgs = [f"debug start: {len(fixes)} fixes reverted"] + [f"{f['subject']}\n\n{f['body']}" for f in fixes]
            dates = [cand.params["b_date"]] + [f["date"] for f in fixes]
            shas = write_chain(repo_dir, forward, messages=msgs, dates=dates, parent=b, ref=f"refs/crucible/debug/{cand.id}")
            a2, b2 = shas[0], shas[-1]
            info["b_prime_equals_b"] = tree_of(repo_dir, b2) == tb
            units = first_parent_units(repo_dir, a2, b2)
            _bodies(repo_dir, units)
            win = Window(id=cand.id, repo=cand.repo, repo_dir=str(repo_dir), kind="debug", a=a2, b=b2, a_ref=f"{cand.params.get('b_ref', 'HEAD')}-debug", b_ref=str(cand.params.get("b_ref", "HEAD")),
                         a_date=cand.params["b_date"], b_date=cand.params["b_date"], b_version=None, units=units)
            info["fixes"] = [{"sha": f["sha"], "subject": f["subject"], "unit": u.sha} for f, u in zip(fixes, units, strict=True)]
            # strip mode leaves the fixes' other test-file changes in A' (a table row a driver test reads): recorded as a risk
            vis_data = sorted({p for f in fixes for p in f["files"] if prof.is_test_file(p) and p not in tests_b and p not in extra
                               and p not in tables}) if mode == "strip" else []
            info["hidden"] = {"mode": mode, "prose": sorted(prose_b), "changelog_cited": cited, "visible_stripped": {p: sorted(ks) for p, ks in extra.items()},
                              "table_entries_stripped": tables,
                              "visible_test_changes": vis_data,
                              **({"stripped": {p: sorted(set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in fixes])) for p in tests_b}}
                                 if mode == "strip" else {"reverted": sorted({p for f in fixes for p in f["files"] if is_test_path(p)})})}
            return win, info

    @staticmethod
    def _locate_defs(repo: RepoGit, b: str, prof: Any, keys: Mapping[str, set[str]]) -> dict[str, set[str]]:
        """``{file: keys}`` of tests to strip, each key moved to the file that defines it: a test binary's id names its
        root file (Rust ``tests/lib.rs::suite::spec::t``) while the function lives in a module file
        (``tests/suite/spec.rs``); the defining file is the test file whose definition key is a suffix of the id's key."""
        from crucible.factory.generators.base import git, out

        out_: dict[str, set[str]] = {}
        listing: list[str] | None = None
        for p, ks in keys.items():
            text = repo.show(b, p)
            defs = set(prof.test_defs(text, p)) if text is not None else set()
            for k in ks:
                if k in defs or k == "*":  # ``*``: the whole case file (``hide_key``)
                    out_.setdefault(p, set()).add(k)
                    continue
                if listing is None:
                    listing = [f for f in out(git(repo.dir, "ls-tree", "-r", "--name-only", b)).splitlines() if prof.is_test_file(f)]
                root = p.rsplit("/", 1)[0] if "/" in p else ""
                for f in listing:
                    if root and not f.startswith(root + "/"):
                        continue
                    # the id's last segments are the definition's (module and name)
                    hit = next((dk for dk in prof.test_defs(repo.show(b, f), f)
                                if k.split("::")[-min(2, len(dk.split("::"))):] == dk.split("::")[-min(2, len(dk.split("::"))):]), None) if f != p else None
                    if hit:
                        out_.setdefault(f, set()).add(hit)
                        break
        return out_

    async def precheck(self, win: Window, lang: str) -> str:
        """Why the window cannot be graded fairly ("" when it can): the agent's own view of A' must build, and B's test files
        must build against A' (a compiled language's ``build failed`` notes). The view's results are kept
        (``self.views``): after validation no milestone test may fail in the agent's own tree (``visible_checks``).
        The agent's tests also run on B's source (``self.oracle_views``): a milestone test that passes in A' but fails
        there is an old copy of a hidden test that asserts the fault (revert mode restores black ae5830b7's
        ``test_lines_with_leading_tabs_expanded`` expecting 8-column tabs and its old ``docstring.py`` case), so a correct
        fix makes the agent's own tests fail (``stale_checks``)."""
        repo = RepoGit(Path(win.repo_dir))
        with tempfile.TemporaryDirectory(prefix="lhd-pre-") as td:
            a = Path(td) / "a"
            ok, bad = await asyncio.to_thread(repo.build, win.a, [], a)
            if not ok:
                return f"A' does not check out ({bad})"
            card = await self.images_for(lang).ensure(win.repo, a)
            if not card["ok"]:
                return "the World image of A' does not build"
            sb = self.sandboxes(card["image"])
            view = await sb.run(a, suite_targets(a, {}))
            self.views[win.id] = dict(view.results) if view.infra == "ok" else {}
            if "build failed" in view.why:
                return f"the agent's start tree does not build ({view.why[:200]})"
            changed = repo.changed(win.a, win.b)
            o = Path(td) / "o"
            if (await asyncio.to_thread(repo.build, win.a, [], o))[0]:
                for st, p in changed:
                    if is_test_path(p):
                        continue
                    if st == "D":
                        (o / p).unlink(missing_ok=True)
                    elif (txt := repo.show(win.b, p)) is not None:
                        (o / p).parent.mkdir(parents=True, exist_ok=True)
                        (o / p).write_text(txt)
                ov = await sb.run(o, suite_targets(o, {}))
                # a run that did not build or passed nothing says nothing about single tests
                usable = ov.infra == "ok" and "build failed" not in ov.why and any(v == "passed" for v in ov.results.values())
                self.oracle_views[win.id] = dict(ov.results) if usable else {}
            for st, p in changed:
                if st != "D" and is_test_path(p) and (txt := repo.show(win.b, p)) is not None:
                    (a / p).parent.mkdir(parents=True, exist_ok=True)
                    (a / p).write_text(txt)
            files = sorted({p for st, p in changed if profile_of(lang).is_test_file(p)})
            hid = await sb.run(a, files) if files and lang in COMPILED else None
            if hid is not None and "build failed" in hid.why:
                return f"hidden tests do not build at the start (a fix adds API: {hid.why[:200]})"
        return ""

    async def build(self, cand: Candidate) -> dict[str, Any]:
        lang = cand.params["lang"]
        self.lease.guard_repo(cand.repo)
        if (self.store.dir(cand.id) / "window.json").exists():
            return self.store.window(cand.id)
        with scope(lang) as prof:
            rec: dict[str, Any] = {}
            extra: dict[str, set[str]] = {}
            for t in cand.params.get("hide_tests") or ():  # subset --hide-tests: visible tests that expect the bug
                f, k = prof.key_of_test(t)
                extra.setdefault(f, set()).add(k)
            for _round in range(2):  # round 2: the milestone tests that failed in the agent's own tree are stripped too
                why = ""
                for mode in cand.params.get("hide_modes") or ("revert", "strip"):  # a reverted test hunk can take a helper later tests use: then strip only
                    win, info = await asyncio.to_thread(self.synthesize, cand, mode=mode, extra_hide=extra)
                    if win is None:
                        self.note(cand.id, "S3_world", False, "fewer than two fixes revert together", lang=lang, **info)
                        return {"id": cand.id, "stage": "source", "reason": "fewer than two fixes revert together", **info}
                    why = await self.precheck(win, lang)
                    if not why.startswith("the agent's start tree does not build") or info.get("hide_mode") == "strip":
                        break
                if why:
                    self.note(cand.id, "S3_world", False, why, lang=lang)
                    rec = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "split": self.store.split, "stage": "windows", "reason": why,
                           "lane": "window", "lang": lang, NAME: info}
                    self.store.save_window(rec)
                    return rec
                shutil.rmtree(self.store.dir(cand.id), ignore_errors=True)  # a second round validates a new start tree
                rec = await validate_window(win, self.store, self.images_for(lang), self.sandboxes, reruns=3, min_milestones=2, mutants=True,
                                            import_tolerant=lang == "python", leak_prefix=f"{PREFIX}leak-", log=self.log,
                                            allow_independent=bool(cand.params.get("allow_independent")))
                view, oracle = self.views.pop(win.id, {}), self.oracle_views.pop(win.id, {})
                vis = sorted(t for m in rec.get("milestones") or [] for t in m["tests"] if view.get(t) in ("failed", "error"))
                stale = sorted(t for m in rec.get("milestones") or [] for t in m["tests"]
                               if view.get(t) == "passed" and oracle.get(t) in ("failed", "error"))
                info["visible_checks"], info["stale_checks"] = vis[:20], stale[:20]
                if not (vis or stale) or rec.get("stage") == "windows":
                    break
                with scope(lang) as prof:
                    touched = {p for u in win.units for p in u.files}
                    for t in [*vis, *stale]:
                        f, k = hide_key(t, touched, prof.key_of_test)
                        extra.setdefault(f, set()).add(k)
                if _round == 1:
                    rec["stage"], rec["reason"] = "windows", (f"hidden checks visible: {len(vis)} milestone test(s) fail in the agent's own tree, "
                                                              f"{len(stale)} fail there on the fixed source")
            rec["lang"] = lang
            units = {u.sha: u for u in win.units}
            by_unit = {m["unit"]: m["id"] for m in rec.get("milestones") or []}
            edges = [{"fix": units[m["unit"]].subject[:100], "needs": [units[s].subject[:100] for s in m.get("requires_units") or []]}
                     for m in rec.get("milestones") or [] if m.get("requires")]
            rec[NAME] = {**info, "b": cand.params["b"], "source_repo": cand.repo, "interacting": edges,
                         **{k: cand.params.get(k) for k in ("cutoff", "contamination", "fix_dates")},
                         **({"parent": cand.params["parent"]} if cand.params.get("parent") else {}),
                         "fix_milestones": {f["sha"]: by_unit.get(f["unit"]) for f in info["fixes"]}, "sources": [s["id"] for s in cand.sources]}
            if rec.get("milestones"):
                rec[NAME]["intent"] = task_intent(rec, faulty_names(rec, RepoGit(Path(win.repo_dir))))
            rec["rebreak"] = rec[NAME]  # GeneratedTask.oracle_steps reads ``window[generator]``; rebreak tooling reads this
            self.store.save_window(rec)
            stage = rec.get("stage")
            self.note(cand.id, "S3_world", stage != "windows", "" if stage != "windows" else rec.get("reason", ""), lang=lang)
            if stage != "windows":
                ok4 = stage in ("f2p_valid", "dependent", "mutants_valid")
                self.note(cand.id, "S4_oracle", ok4, "" if ok4 else rec.get("reason", ""), lang=lang, milestones=len(rec.get("milestones") or []))
                if ok4 and stage == "f2p_valid":
                    self.note(cand.id, "S6", False, rec.get("reason", ""), lang=lang, executed_edges=0)
            return rec

    async def gates(self, wid: str, *, cfg: GateConfig | None = None) -> dict[str, Any]:
        """S6, S5, S8, S3, S10 on one stored window (``base.deterministic_gates``) inside its profile's scope, with the
        lane's own S3 containers; each gate's verdict goes to the funnel."""
        rec = self.store.window(wid)
        lang = str(rec.get("lang") or "python")
        task = self.task(wid)
        with scope(lang):
            res = await deterministic_gates(task, cfg=cfg, s3_sandboxes=debug_sandboxes(parallel=4, prefix=f"{PREFIX}s3-"), s3_prefix=f"{PREFIX}s3-")
        s3 = res["gates"].get("S3") or {}
        failing = [k for k, v in ((s3.get("evidence") or {}).get("checks") or {}).items() if not v]
        flipped = sorted({t for u in (((s3.get("evidence") or {}).get("reruns") or {}).get("unclean") or []) for t in u.get("differs") or []})
        checked = {t for m in rec.get("milestones") or [] for t in m["tests"]}
        s5 = res["gates"].get("S5") or {}
        refused = [re.search(r"on c_regress \(([^)]*)\)", f) for f in s5.get("findings") or []]
        if s5.get("ok") is False and refused and all(refused) and not rec.get("s5_regress_excluded"):
            # the fairness lint refuses a *regression* test that reads source structure (a formatter's test parsing its own
            # output): removing a regression test only removes a check, so the task is re-gated without those tests
            from crucible.factory.lanes.windows_timing import drop_tests

            drop = {m.group(1): "S5 fairness: the test reads source structure; never a regression check" for m in refused if m}
            drop_tests(rec, drop)
            rec["s5_regress_excluded"] = sorted(drop)
            self.store.save_window(rec)
            self.lane()._made.clear()
            return await self.gates(wid, cfg=cfg)
        s8 = res["gates"].get("S8") or {}
        if s8.get("ok") is False and not rec.get("s8_flaky_excluded") and all("ground truth fails: oracle" in f for f in s8.get("findings") or []):
            # the oracle as one patch and as its unit steps are one end state: a regression test that fails in one grade and
            # passes in the other moved with the load (``windows_timing.load_dependent``), never with the code
            with scope(lang):
                grades = [await task.grade_patch(sol, trail=[]) for sol in (task.oracle_patch(), task.oracle, task.oracle_patch(), task.oracle)]
            fails = [set((g.meta or {}).get("failing_tests") or []) for g in grades if g.infra.get("status") == "ok"]
            moved = sorted(t for t in set().union(*fails) - set.intersection(*fails) if t not in checked) if len(fails) >= 2 else []
            if moved and len(moved) <= 3:
                from crucible.factory.lanes.windows_timing import drop_tests

                drop_tests(rec, {t: "S8: failed in some grades of one oracle end state and passed in others" for t in moved})
                rec["s8_flaky_excluded"] = moved
                self.store.save_window(rec)
                self.lane()._made.clear()
                return await self.gates(wid, cfg=cfg)
        if failing == ["reruns_clean"] and flipped and len(flipped) <= 3 and not set(flipped) & checked and not rec.get("s3_flaky_excluded"):
            # benchmark v2's empirical rule (``windows_timing.load_dependent``) for the existing suite: a test that flips
            # between reruns of one unchanged tree is the host's load, never a check; excluded with its reason, gates re-run once
            from crucible.factory.lanes.windows_timing import drop_tests

            drop_tests(rec, {t: "S3: flipped between reruns of the unchanged start tree" for t in flipped})
            rec["s3_flaky_excluded"] = flipped
            self.store.save_window(rec)
            self.lane()._made.clear()
            return await self.gates(wid, cfg=cfg)
        for g, v in res["gates"].items():
            if v.get("ok") is not None:
                self.note(wid, g, bool(v["ok"]), "; ".join(v.get("findings") or [])[:300], lang=lang)
        return res


def funnel_by_lang(funnel: Funnel) -> dict[str, Any]:
    """Per language and stage: in, passed, drop reasons (the yield ledger of ``00`` §9, split by language)."""
    out: dict[str, Any] = {}
    for r in funnel.rows():
        if r["generator"] != NAME:
            continue
        lang = (r.get("evidence") or {}).get("lang") or r["candidate"].split("/", 1)[0].removeprefix("db-").split("-", 1)[0]
        s = out.setdefault(lang, {}).setdefault(r["stage"], {"in": 0, "pass": 0, "reasons": {}})
        s["in"] += 1
        if r["ok"]:
            s["pass"] += 1
        else:
            k = re.split(r"[:(]", r.get("reason") or "fail")[0].strip()[:70] or "fail"
            s["reasons"][k] = s["reasons"].get(k, 0) + 1
    return out
