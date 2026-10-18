"""Fairness rules for checks (P2; swarm-pipeline/00 §5.8, ``HUD`` "What is a good task?").

A check is fair when every competent solution passes it and it grades only what the prompt asks for. Four rules, the
first three static (a lint over the hidden tests' own source), the last executed:

| rule | finding kind | severity | what happens |
|---|---|---|---|
| functional checks only: a test that reads source text (``inspect.getsource``, opening a ``.py`` file, ``grep``) or source structure (``ast.parse`` of code, ``__code__``, ``co_varnames``, ``dis``, ``locals()``) where behaviour can be tested | ``source_text`` / ``source_structure`` | refuse | the check is refused (HUD's attention-fix grader rejected 4 working fixes for a variable name) |
| a test that asserts a private name exists (``hasattr(x, "_cache")``, ``"_x" in dir(m)``, ``vars(x)["_y"]``) | ``private_name`` | refuse | the check is refused (an implementation detail) |
| exact-string assertions on free text (a message, an output line) the prompt never states | ``unstated_format`` | prompt | a ``spec_defect``: the prompt must state the format, from the test's own source (D8); the check is kept |
| only grade what the prompt asks for: a gating check with no requirement | ``no_requirement`` | refuse | the check is refused (the ledger is two-way) |
| the ten-competent-people test: every gating check passes the oracle **and** every valid alternative | ``false_fail`` | refuse | the check is refused; false-fail candidates from the audit are re-run on the alternatives before any check is kept |

``lint_checks`` runs the static rules over a window's check set; ``ten_competent_people`` judges executed results;
``admit_check`` is the executed admission of a new or changed check (§8.3: passes the oracle and every valid
alternative, every mutant keeps failing its milestone); ``recheck_false_fail`` re-runs suspected false-fail checks on
the alternatives.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Kind = Literal["source_text", "source_structure", "private_name", "unstated_format", "no_requirement", "false_fail", "weak_check"]
SEVERITY: dict[str, str] = {"source_text": "refuse", "source_structure": "refuse", "private_name": "refuse", "unstated_format": "prompt",
                            "no_requirement": "refuse", "false_fail": "refuse", "weak_check": "refuse"}
OWNER: dict[str, str] = {"source_text": "verifier", "source_structure": "verifier", "private_name": "verifier", "unstated_format": "spec",
                         "no_requirement": "verifier", "false_fail": "verifier", "weak_check": "verifier"}
SOURCE_CALLS = frozenset({"getsource", "getsourcelines", "getsourcefile", "getfile", "findsource", "getcomments"})
STRUCTURE_ATTRS = frozenset({"__code__", "co_varnames", "co_names", "co_consts", "co_code", "co_cellvars", "co_freevars", "func_code", "f_locals"})
SHELL_TEXT = re.compile(r"\b(grep|rg|ag|ack|sed|awk)\b")
FREE_TEXT_MIN_WORDS = 3
FREE_TEXT_MIN_CHARS = 16
TEXT_CONTEXT = re.compile(r"\b(str|repr|format|message|msg|args|output|stdout|stderr|text|err|out|value|excinfo|exc|error|warning|caplog|capsys|"
                          r"capfd|records|getvalue|readouterr|render|describe|help)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    check: str
    test: str
    kind: str
    severity: str
    owner: str
    evidence: str
    requirement_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)
    tests_seen: int = 0
    tests_unresolved: list[str] = field(default_factory=list)  # ids whose source is not in the test files (inherited, generated)

    def refused_checks(self) -> list[str]:
        return sorted({f.check for f in self.findings if f.severity == "refuse"})

    def spec_defects(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "prompt"]

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.kind] = out.get(f.kind, 0) + 1
        return out

    @property
    def ok(self) -> bool:
        return not self.refused_checks()

    def to_json(self) -> dict[str, Any]:
        return {"findings": [f.to_json() for f in self.findings], "by_kind": self.by_kind(), "refused_checks": self.refused_checks(),
                "spec_defects": len(self.spec_defects()), "tests_seen": self.tests_seen, "tests_unresolved": len(self.tests_unresolved), "ok": self.ok}


# ---- the static rules -------------------------------------------------------------------------------------------------
def _parse(text: str) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(text)
    except SyntaxError:
        return None


def test_node(tree: ast.Module, test_id: str) -> ast.AST | None:
    """The function (or class method) a pytest id names, in its own file's tree."""
    names = [re.sub(r"\[.*$", "", n) for n in test_id.split("::")[1:]]
    body: list[ast.stmt] = tree.body
    node: Any = None
    for n in names:
        node = next((x for x in body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and x.name == n), None)
        if node is None:
            return None
        body = getattr(node, "body", [])
    return node


def _name(f: ast.AST) -> str:
    return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def is_free_text(s: str) -> bool:
    """A string literal that is prose (a message or an output line), not data: several words, mostly letters."""
    words = re.findall(r"[A-Za-z]{2,}", s)
    letters = sum(c.isalpha() for c in s)
    return len(s) >= FREE_TEXT_MIN_CHARS and len(words) >= FREE_TEXT_MIN_WORDS and " " in s.strip() and letters >= 0.5 * len(s.replace(" ", ""))


def _str_consts(n: ast.AST) -> list[str]:
    return [c.value for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)]


def _helpers(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level helper functions a test may call (``def check_source(f): ...``): their bodies count as the test's."""
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("test")}


def _call_kind(n: ast.Call) -> str | None:
    """The fairness kind of one call in a test body, or None."""
    f = _name(n.func)
    full = ast.unparse(n.func)
    consts = _str_consts(n)
    reads_py = any(c.endswith(".py") for c in consts) or any(isinstance(a, ast.Attribute) and a.attr == "__file__" for a in ast.walk(n))
    if (f in SOURCE_CALLS and ("inspect" in full or f.startswith("getsource"))) or (f in ("open", "read_text", "read_bytes") and reads_py) \
            or (f in ("run", "check_output", "call", "Popen", "system", "getoutput") and any(SHELL_TEXT.search(c) for c in consts)):
        return "source_text"
    if (f == "parse" and full.endswith("ast.parse") and n.args and not isinstance(n.args[0], ast.Constant)) or full.startswith("dis."):
        return "source_structure"  # (``locals()`` in a test reads the test's own frame, not the code under test)
    if f in ("hasattr", "getattr") and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
        name = n.args[1].value
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            return "private_name"
    return None


def lint_test(node: ast.AST, prompt: str, *, helpers: Mapping[str, ast.AST] | None = None) -> list[tuple[str, str]]:
    """``(kind, evidence)`` for one test's body (its decorators are parameters, not checks)."""
    out: list[tuple[str, str]] = []
    stmts = list(getattr(node, "body", []))
    seen_helpers: set[str] = set()
    todo = list(stmts)
    nodes: list[ast.AST] = []
    while todo:  # the body, plus the module helpers it calls (one level of indirection is where a source grep hides)
        s = todo.pop()
        for n in ast.walk(s):
            nodes.append(n)
            if isinstance(n, ast.Call) and helpers and (h := _name(n.func)) in helpers and h not in seen_helpers:
                seen_helpers.add(h)
                todo += list(getattr(helpers[h], "body", []))
    p = _norm(prompt)

    def ev(n: ast.AST) -> str:
        u = ast.unparse(n)
        return u if len(u) <= 200 else u[:197] + "..."

    in_assert = {id(c) for a in nodes if isinstance(a, ast.Assert) for c in ast.walk(a)}
    inputs = {_norm(c.value) for c in nodes if isinstance(c, ast.Constant) and isinstance(c.value, str) and id(c) not in in_assert}
    for n in nodes:
        if isinstance(n, ast.Call) and (kind := _call_kind(n)):
            out.append((kind, ev(n)))
        elif isinstance(n, ast.Attribute) and n.attr in STRUCTURE_ATTRS:
            out.append(("source_structure", ev(n)))
        elif isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
            right = n.comparators[0]
            if isinstance(n.left, ast.Constant) and isinstance(n.left.value, str) and n.left.value.startswith("_") \
                    and not n.left.value.startswith("__") and isinstance(right, ast.Call) and _name(right.func) in ("dir", "vars"):
                out.append(("private_name", ev(n)))
        if isinstance(n, ast.Assert):  # a literal the test itself feeds in (``echo("Pause to stderr")``) is echoed, not a format
            out += [("unstated_format", f"{ev(n.test)} (text {s!r} is not in the prompt)") for s in _unstated(n.test, p)
                    if not any(_norm(s) in i or (len(i) >= 8 and i in _norm(s)) for i in inputs)]
        elif isinstance(n, ast.Call) and _name(n.func) == "raises":
            for kw in n.keywords:
                if kw.arg == "match" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    lit = re.sub(r"\\(.)", r"\1", kw.value.value).strip("^$")
                    if is_free_text(lit) and _norm(lit) not in p:
                        out.append(("unstated_format", f"pytest.raises(match={kw.value.value!r}) (the message is not in the prompt)"))
    return list(dict.fromkeys(out))


def _unstated(test: ast.AST, prompt_norm: str) -> list[str]:
    """Free-text literals an assertion compares a text value against (``==``, ``in``, ``startswith``) that the prompt
    does not contain. Only text-valued sides count (``str(e)``, ``result.output``, ``caplog.text``, ...)."""
    out = []
    for n in ast.walk(test):
        lits: list[str] = []
        other: list[ast.AST] = []
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
            for side, rest in ((n.left, n.comparators[0]), (n.comparators[0], n.left)):
                if isinstance(side, ast.Constant) and isinstance(side.value, str):
                    lits.append(side.value)
                    other.append(rest)
        elif isinstance(n, ast.Call) and _name(n.func) in ("startswith", "endswith") and n.args and isinstance(n.args[0], ast.Constant) \
                and isinstance(n.args[0].value, str) and isinstance(n.func, ast.Attribute):
            lits.append(n.args[0].value)
            other.append(n.func.value)
        for lit, o in zip(lits, other, strict=True):
            if is_free_text(lit) and TEXT_CONTEXT.search(ast.unparse(o)) and _norm(lit) not in prompt_norm:
                out.append(lit)
    return out


def lint_checks(checks: Sequence[Mapping[str, Any]], tests: Mapping[str, str], prompt: str) -> LintReport:
    """The static rules over a check set (``{"id", "requirement_ids", "gating", "spec": {"tests": [...]}}`` per check)
    whose tests live in ``tests`` (path -> source; the hidden test files). A test whose source is not found (inherited
    from a mixin, generated) is listed as unresolved, never passed silently."""
    rep = LintReport()
    trees: dict[str, ast.Module | None] = {}
    for c in checks:
        cid = str(c["id"])
        reqs = tuple(str(r) for r in c.get("requirement_ids") or [])
        if c.get("gating", True) and not reqs:
            rep.findings.append(Finding(cid, "", "no_requirement", SEVERITY["no_requirement"], OWNER["no_requirement"],
                                        "a gating check that enforces no requirement of the prompt"))
        for tid in (c.get("spec") or {}).get("tests") or []:
            rep.tests_seen += 1
            path = tid.split("::", 1)[0]
            if path not in trees:
                trees[path] = _parse(tests[path]) if path in tests else None
            tree = trees[path]
            node = test_node(tree, tid) if tree is not None else None
            if node is None:
                rep.tests_unresolved.append(tid)
                continue
            for kind, evidence in lint_test(node, prompt, helpers=_helpers(tree)):  # type: ignore[arg-type]
                rep.findings.append(Finding(cid, tid, kind, SEVERITY[kind], OWNER[kind], evidence, reqs))
    rep.findings = list(dict.fromkeys(rep.findings))
    return rep


# ---- the executed rules -----------------------------------------------------------------------------------------------
def check_passes(statuses: Mapping[str, str], tests: Sequence[str]) -> bool:
    return bool(tests) and all(statuses.get(t) == "passed" for t in tests)


def ten_competent_people(check_tests: Mapping[str, Sequence[str]], positives: Mapping[str, Mapping[str, str] | None],
                         gating: Mapping[str, bool] | None = None) -> list[Finding]:
    """Every gating check must pass every end state that is a correct solution (``positives``: name -> the test
    statuses of that end state, ``None`` when the run was unmeasured: then nothing is concluded from it). A check that
    fails any of them is a ``false_fail`` finding."""
    out = []
    for cid, tests in check_tests.items():
        if gating is not None and not gating.get(cid, True):
            continue
        bad = [name for name, st in positives.items() if st is not None and not check_passes(st, tests)]
        if bad:
            failing = sorted({t for name in bad for t in tests if (positives[name] or {}).get(t) != "passed"})
            out.append(Finding(cid, failing[0] if failing else "", "false_fail", SEVERITY["false_fail"], OWNER["false_fail"],
                               f"fails the valid end state(s) {bad}; failing tests {failing[:5]}"))
    return out


RunFn = Callable[[Any, Mapping[str, Sequence[str]]], Awaitable[Mapping[str, str] | None]]


async def admit_check(check_id: str, tests: Sequence[str], run: RunFn, *, oracle: Any, alternatives: Mapping[str, Any],
                      mutants: Mapping[str, Any] | None = None, must_fail: Sequence[str] = ()) -> dict[str, Any]:
    """The executed admission of a new or changed check (§8.3 and §5.8): ``run(solution, {check: tests})`` gives the
    test statuses. The check is admitted only when it passes the oracle and every valid alternative and, for each
    mutant named in ``must_fail``, fails it. Returns ``{admitted, findings, rows}``."""
    rows: dict[str, Any] = {}
    pos: dict[str, Mapping[str, str] | None] = {"oracle": await run(oracle, {check_id: tests})}
    for name, sol in alternatives.items():
        pos[name] = await run(sol, {check_id: tests})
    findings = ten_competent_people({check_id: tests}, pos)
    for name, st in pos.items():
        rows[name] = None if st is None else check_passes(st, tests)
    for name in must_fail:
        st = await run((mutants or {})[name], {check_id: tests})
        rows[name] = None if st is None else check_passes(st, tests)
        if st is not None and check_passes(st, tests):
            findings.append(Finding(check_id, "", "weak_check", SEVERITY["weak_check"], OWNER["weak_check"],
                                        f"mutant {name} passes the check: it does not test its requirement's behaviour"))
    unmeasured = [n for n, v in rows.items() if v is None]
    return {"admitted": not findings and not unmeasured, "findings": [f.to_json() for f in findings], "rows": rows, "unmeasured": unmeasured}


async def recheck_false_fail(candidates: Mapping[str, Sequence[str]], run: RunFn, alternatives: Mapping[str, Any]) -> dict[str, Any]:
    """False-fail candidates (check id -> its tests) from an audit, re-run on every valid alternative before any of them
    is kept: a check an alternative fails is refused; one every alternative passes is kept (the audit's suspicion is
    not confirmed by execution). ``{"refused": [...], "kept": [...], "findings": [...]}``."""
    pos = {name: await run(sol, candidates) for name, sol in alternatives.items()}
    findings = ten_competent_people(candidates, pos)
    refused = sorted({f.check for f in findings})
    unmeasured = sorted(n for n, st in pos.items() if st is None)
    return {"refused": refused, "kept": sorted(set(candidates) - set(refused)) if not unmeasured else [],
            "unmeasured": unmeasured, "findings": [f.to_json() for f in findings]}
