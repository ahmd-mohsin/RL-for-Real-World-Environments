"""The timing-test rule (benchmark v2; P6 finding): a test whose verdict depends on wall time is never a milestone or a
regression check.

P6 measured why. mistune's milestones m12/m13 were ``..._are_near_linear`` tests (``assert_near_linear`` times two input
sizes): 12 of 60 oracle grades failed under load (S10), and ``..._does_not_scale_quadratically`` (``perf_counter() -
start < 2.0``) passed at A on an idle host, so the empty patch scored Φ 0.083 (S3). Such a test grades the host's load,
not the agent's code.

Two ways a test is found, both recorded with the reason:

- **statically** (``timing_tests``, at mining, before ownership): its name (function or class, parameters ignored)
  names timing (``timing``, ``timeout``, ``perf``, ``benchmark``, ``near_linear``, ``quadratic``, ``scale_linearly``,
  ``quickly``, ``latency``, ``throughput``, ``elapsed``, ``wall_clock``, ``deadline``...); a marker or fixture says so
  (``pytest.mark.benchmark|perf|performance|timing|timeout|flaky``, a ``benchmark`` fixture, a module ``pytestmark``);
  or its body measures elapsed time (``perf_counter``, ``monotonic``, ``process_time``, ``timeit``, ``default_timer``,
  ``time.time() - t0``) or calls a helper whose name names timing (``assert_near_linear``);
- **empirically** (``load_dependent``, at S10): the oracle's end state is graded at concurrency 1 and 4; a test that
  fails in some grades and passes in others (one end state, so the verdict moved with the load) is excluded the same way
  and the window is validated again without it.

Excluding a test only removes a check (a milestone whose tests are all excluded goes; ownership, the prober and the
mutants run on what is left), so the rule can never accept a task the gates would refuse.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# words that name a measurement, never a domain noun: ``scale`` and ``speedup`` were dropped after the first v2 run
# excluded voluptuous's decimal ``precision_scale`` tests and mistune's ``speedup`` plugin test (both deterministic);
# scaling is named only as ``scale_linearly`` / ``scale_quadratically``
TIMING_WORDS = (r"timing|timed|timeouts?|perf|performance|benchmarks?|bench|near_?linear|linear_?time|quadratic(?:ally)?|complexity|"
                r"scales?_(?:linearly|quadratically|exponentially)|quickly|latency|throughput|elapsed|wall_?clock|deadline")
# a name segment (split on ``_`` and case changes) that names timing: ``test_dense_emphasis_is_near_linear``,
# ``TestPerformance::test_x``; never a substring of another word (``test_scaled_font`` stays, ``test_benchmarks`` goes)
TIMING_NAME = re.compile(rf"(?i)(?:^|_)(?:{TIMING_WORDS})(?=_|$)")
TIMING_MARKS = frozenset({"benchmark", "perf", "performance", "timing", "timeout", "flaky"})
TIMING_FIXTURES = frozenset({"benchmark", "benchmark_pedantic", "aio_benchmark"})
CLOCK_CALLS = frozenset({"perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns", "process_time", "process_time_ns", "thread_time",
                         "timeit", "default_timer", "Timer"})


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)


def name_says_timing(name: str) -> bool:
    return bool(TIMING_NAME.search(_snake(name)))


def _test_parts(test_id: str) -> tuple[str, list[str]]:
    """``file::Class::name[param]`` -> (file, [Class, name])."""
    parts = test_id.split("::")
    return parts[0], [re.sub(r"\[.*$", "", p) for p in parts[1:]]


def _mark_names(decorators: Iterable[ast.expr]) -> set[str]:
    out: set[str] = set()
    for d in decorators:
        node = d.func if isinstance(d, ast.Call) else d
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            chain.append(node.id)
        chain.reverse()  # ["pytest", "mark", "benchmark"]
        if "mark" in chain[:-1]:
            out.add(chain[-1])
    return out


def _module_marks(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets):
            vals = n.value.elts if isinstance(n.value, (ast.List, ast.Tuple)) else [n.value]
            out |= _mark_names(vals)
    return out


def _callee(c: ast.Call) -> str:
    f = c.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _is_wall_time(c: ast.Call) -> bool:
    f = c.func
    return isinstance(f, ast.Attribute) and f.attr == "time" and isinstance(f.value, ast.Name) and f.value.id == "time"


def body_reasons(fn: ast.AST) -> list[str]:
    """Why a test function's body times itself (empty: it does not)."""
    out: list[str] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            name = _callee(n)
            if name in CLOCK_CALLS:
                out.append(f"body calls {name}()")
            elif name and name_says_timing(name) and not name.startswith("test"):
                out.append(f"body calls the timing helper {name}()")
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub):
            if any(isinstance(s, ast.Call) and _is_wall_time(s) for s in (n.left, n.right)):
                out.append("body subtracts time.time()")
    return sorted(set(out))


def _index(text: str) -> tuple[dict[str, tuple[ast.AST, list[ast.ClassDef]]], set[str]] | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    out: dict[str, tuple[ast.AST, list[ast.ClassDef]]] = {}

    def walk(nodes: Iterable[ast.stmt], prefix: str, classes: list[ast.ClassDef]) -> None:
        for n in nodes:
            if isinstance(n, ast.ClassDef):
                walk(n.body, f"{prefix}{n.name}::", [*classes, n])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + n.name] = (n, classes)

    walk(tree.body, "", [])
    return out, _module_marks(tree)


def timing_reason(test_id: str, tests: Mapping[str, str] | None = None, *, _cache: dict[str, Any] | None = None) -> str | None:
    """Why ``test_id`` is a timing test, or None. ``tests``: path -> text of the test files (B's, as the grader writes
    them); without them only the name is read."""
    path, quals = _test_parts(test_id)
    for q in quals:
        if name_says_timing(q):
            return f"name: {q}"
    text = (tests or {}).get(path)
    if not text:
        return None
    cache = _cache if _cache is not None else {}
    if path not in cache:
        cache[path] = _index(text)
    idx = cache[path]
    if idx is None:
        return None
    funcs, module_marks = idx
    if module_marks & TIMING_MARKS:
        return f"module marker: {min(module_marks & TIMING_MARKS)}"
    hit = funcs.get("::".join(quals))
    if hit is None:
        return None
    fn, classes = hit
    marks = _mark_names(getattr(fn, "decorator_list", []))
    for c in classes:
        marks |= _mark_names(c.decorator_list)
    if marks & TIMING_MARKS:
        return f"marker: {min(marks & TIMING_MARKS)}"
    args = getattr(fn, "args", None)
    fixtures = {a.arg for a in (args.args + args.kwonlyargs)} if args else set()
    if fixtures & TIMING_FIXTURES:
        return f"fixture: {min(fixtures & TIMING_FIXTURES)}"
    why = body_reasons(fn)
    return why[0] if why else None


def timing_tests(test_ids: Iterable[str], tests: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{test id: reason}`` for every timing test among ``test_ids`` (see the module doc)."""
    cache: dict[str, Any] = {}
    out = {}
    for t in test_ids:
        why = timing_reason(t, tests, _cache=cache)
        if why:
            out[t] = why
    return out


def load_dependent(failing_by_grade: Sequence[Iterable[str] | None], checked: Iterable[str]) -> dict[str, str]:
    """The empirical rule. ``failing_by_grade``: the failing test ids of each oracle grade of one end state (None: a
    masked grade, skipped); ``checked``: every test a check reads. A checked test that failed in some grades and passed
    in others changed its verdict with nothing but the load: ``{test id: reason}``."""
    grades = [set(g) for g in failing_by_grade if g is not None]
    if len(grades) < 2:
        return {}
    out = {}
    for t in sorted(set(checked)):
        fails = sum(1 for g in grades if t in g)
        if 0 < fails < len(grades):
            out[t] = f"S10: failed {fails} of {len(grades)} oracle grades of one end state"
    return out


def drop_tests(rec: dict[str, Any], drop: Mapping[str, str]) -> dict[str, Any]:
    """Remove ``drop`` from every check of a window record in place (milestone tests, pass-to-pass, collateral) and
    record them in ``excluded['timing']`` with ``timing_rule`` reasons. A milestone left with no test goes (and leaves
    every ``requires`` / ``order_hints``). Returns what changed. Edges and mutants are not re-derived here: a window that
    loses tests is validated again (``windows.validate_window(exclude=...)``)."""
    drop = {t: w for t, w in drop.items()}
    gone_ms: list[str] = []
    touched: list[str] = []
    keep = []
    for m in rec.get("milestones") or []:
        left = [t for t in m["tests"] if t not in drop]
        if len(left) != len(m["tests"]):
            touched.append(m["id"])
        if left:
            keep.append({**m, "tests": left})
        else:
            gone_ms.append(m["id"])
    for m in keep:
        m["requires"] = [r for r in m.get("requires") or [] if r not in gone_ms]
        m["order_hints"] = [r for r in m.get("order_hints") or [] if r not in gone_ms]
    rec["milestones"] = keep
    for k in ("p2p", "collateral"):
        rec[k] = [t for t in rec.get(k) or [] if t not in drop]
    ex = rec.setdefault("excluded", {})
    ex["timing"] = sorted(set(ex.get("timing") or []) | set(drop))
    rec.setdefault("timing_rule", {}).update(drop)
    return {"dropped_tests": len(drop), "milestones_gone": gone_ms, "milestones_touched": touched}
