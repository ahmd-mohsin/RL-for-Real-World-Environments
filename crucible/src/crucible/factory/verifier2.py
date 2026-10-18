"""The second verifier, shadow tests and the differential (P2; swarm-pipeline/00 §0 V2/V3/V6, §3.2, §4 S5/S8/S9, §5.4-5.5).

**Who writes it.** The second verifier is written by the reviewer family (``models.DEFAULT_ROLES["second_verifier"]``,
deepseek) while the window's checks come from the repository and its prompt from the author family (qwen). It sees only
the prompt's maintainer text, the requirements and the repository **at A** (``repo_context``: the file list, the README
head, and the source of the modules that define the names the request mentions). It never sees the oracle, B, the
hidden tests or any solution (V2): ``shadow_payload`` is built from those inputs alone and ``leaks`` proves it on every
call (no line of an oracle-only file, no hidden test name).

**Single call, not a session (justified).** The machine can run the writer as a P1 agentic session
(``shadow_session_state``). P2 runs it as one call with repository context plus one syntax-repair round, because of
cost: a session explores with 20 to 40 turns over a growing context (about 0.5 to 1 M tokens a window), so 26 windows
would take most of P2's $5 cap before any rewrite or prompt, while one call is about 20 k tokens. The call is offline
of the oracle in the same way (the payload holds only the prompt, the requirements and A).

**Shadow tests flag until they qualify.** Each requirement gets one pytest module (``crucible_shadow/test_shadow_<rid>.py``),
run by the grader in its own run (never beside the hidden tests) on A plus the solution, import-tolerant. A test is
*sound* once it passes the oracle and every valid alternative; a requirement's shadow check *gates* only when its sound
tests fail the start state and every mutant of its milestone. Until then it only flags (``Qualification``).

**Differential.** Both verifiers judge every end state (oracle, floors, mutants, alternatives, rewrite candidates,
solver end states). Verifier 1's requirement verdict is its milestone's raw pass; verifier 2's is "every sound shadow
test of the requirement passes". Every disagreement is a ``Defect`` whose owner is assigned by rule (``assign_owner``):
- **truth known** (oracle and alternatives pass every requirement, floors and the start state fail every one, a mutant
  fails its milestone and follows its executed ``expected`` map elsewhere): the verifier that contradicts the truth owns
  it (verifier 1: ``verifier_gap`` / ``false_fail``; verifier 2: ``weak_shadow`` / ``unsound_shadow``). Confirmed.
- **truth unknown** (solver end states, rewrite candidates): verifier 1 passes and verifier 2 fails: verifier 1
  (``verifier_gap`` candidate) when the shadow check gates, else verifier 2. Verifier 1 fails and verifier 2 passes: the
  spec (``unstated_format``) when the fairness lint flags a failing hidden test of the requirement, else verifier 2 (the
  hidden tests are real observations, V1). Candidates go to the trace audit (P3); they never gate on their own (V6).

**Machine handlers** (``gate_handlers``): ``s5_solvability`` (clean-World solvability + the null agent + the fairness
lint), ``s8_ground_truth`` (floors, mutants, alternatives, the ten-competent-people test) and ``s9_second_verifier``
(shadow qualification + the differential), for ``factory/machine.py`` with a ``P2Gates`` services object.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.lanes.executable import call_supported, sha256_text
from crucible.factory.lanes.windows import maintainer_text, run_tests
from crucible.factory.lanes.windows_imports import project_roots, tolerant_tests

SHADOW_DIR = "crucible_shadow"
SHADOW_BRIEF = (
    "You are an independent verifier for a coding task. A developer was given the request below and a copy of the "
    "repository as it is now. You write hidden pytest tests that decide whether each requirement was met, from the request "
    "and the current repository only (you never see a solution). Rules: one pytest module per requirement id; import the "
    "library inside each test function, never at module top level; use only public names the request states or that the "
    "repository already has; test observable behaviour (return values, raised exception types, state changes, output "
    "properties); when the request does not fix an exact text or format, assert properties (contains the number, is a "
    "str, is sorted) instead of exact strings; never read source files, never inspect code objects or private names, no "
    "network, files only under tmp_path; deterministic and fast; 2 to 6 focused tests per requirement, named "
    "test_<requirement id lower case>_<what>. Return JSON {\"tests\": [{\"requirement\": id, \"code\": module source}], "
    "\"notes\": str}.")
CONTEXT_CHARS = 60_000
FILE_CHARS = 14_000
_IDENT = re.compile(r"`([^`]{1,80})`|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[a-z]+_[a-z0-9_]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"
                    r"|\b([A-Za-z_][A-Za-z0-9_]{2,})\(")


def shadow_path(rid: str) -> str:
    return f"{SHADOW_DIR}/test_shadow_{re.sub(r'[^a-z0-9_]', '_', rid.lower())}.py"


# ---- the payload: prompt + requirements + the repository at A, nothing else (V2) -------------------------------------------
def prompt_names(text: str) -> list[str]:
    """Identifiers the request mentions (in backticks, dotted, snake_case, CamelCase, or called)."""
    out: list[str] = []
    for m in _IDENT.finditer(text):
        raw = m.group(1) or m.group(2) or m.group(3) or ""
        for part in re.split(r"[^A-Za-z0-9_.]", raw):
            for name in part.split("."):
                if len(name) > 2 and name not in out:
                    out.append(name)
    return out


def _source_files(base_dir: Path) -> list[Path]:
    skip = re.compile(r"(^|/)(\.|tests?/|testing/|docs?/|examples?/|benchmarks?/|build/|dist/)|(^|/)(setup|noxfile|conftest)\.py$|(^|/)test_[^/]*\.py$")
    return sorted(p for p in Path(base_dir).rglob("*.py") if p.is_file() and not skip.search(p.relative_to(base_dir).as_posix()))


def outline(text: str) -> str:
    """Signatures and first docstring lines of a module (for a file too large to send whole)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return text[:FILE_CHARS]
    lines = text.splitlines()
    out: list[str] = []

    def walk(body: list[ast.stmt], indent: str) -> None:
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                head = lines[n.lineno - 1].strip()
                j = n.lineno
                while not head.rstrip().endswith(":") and j < len(lines) and j < n.lineno + 6:
                    head += " " + lines[j].strip()
                    j += 1
                out.append(indent + head)
                doc = ast.get_docstring(n)
                if doc:
                    out.append(indent + "    # doc: " + doc.strip().splitlines()[0][:160])  # a comment: never mistaken for a source line
                if isinstance(n, ast.ClassDef):
                    walk(n.body, indent + "    ")

    walk(tree.body, "")
    return "\n".join(out)


def repo_context(base_dir: Path, prompt: str, *, budget: int = CONTEXT_CHARS) -> dict[str, Any]:
    """The repository at A as the second verifier sees it: every source path, the README head, and the modules that
    define (or are named by) the identifiers the request mentions, whole when small, else as an outline. Chosen from the
    prompt and A only: nothing here depends on B, the oracle or the hidden tests."""
    base_dir = Path(base_dir)
    files = _source_files(base_dir)
    names = prompt_names(prompt)
    score: dict[Path, int] = {}
    for f in files:
        text = f.read_text(errors="replace")
        rel = f.relative_to(base_dir).as_posix()
        s = sum(3 for n in names if re.search(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(n)}\b|^{re.escape(n)}\s*=", text, re.MULTILINE))
        s += sum(2 for n in names if n in rel.replace("/", ".").removesuffix(".py").split("."))
        if f.name == "__init__.py" and rel.count("/") <= 2:
            s += 1
        if s:
            score[f] = s
    modules: dict[str, str] = {}
    left = budget
    for f in sorted(score, key=lambda p: (-score[p], p.as_posix())):
        text = f.read_text(errors="replace")
        body = text if len(text) <= FILE_CHARS else "# outline (the module is large)\n" + outline(text)[:FILE_CHARS]
        if len(body) > left:
            body = body[:left]
        if not body.strip() or left <= 200:
            continue
        modules[f.relative_to(base_dir).as_posix()] = body
        left -= len(body)
    readme = next(iter(sorted(base_dir.glob("README*"))), None)
    return {"files": [f.relative_to(base_dir).as_posix() for f in files][:400], "readme_head": readme.read_text(errors="replace")[:1500] if readme else "",
            "modules": modules, "names": names[:60]}


def shadow_payload(prompt: str, requirements: Sequence[Mapping[str, Any]], base_dir: Path) -> dict[str, Any]:
    """Everything the second verifier gets: the request (maintainer text, no harness epilogue), the behaviour requirements
    and the repository at A."""
    text = maintainer_text(prompt)
    reqs = [{"id": str(r["id"]), "text": str(r["text"])} for r in requirements if r.get("milestone")]
    return {"request": text, "requirements": reqs, "repository": repo_context(base_dir, text)}


def leaks(payload: Mapping[str, Any], *, oracle_files: Mapping[str, str | None], hidden_tests: Mapping[str, str], base_dir: Path) -> list[str]:
    """V2 proven per call: a payload line that exists only in B (a line of an oracle file that no file of A has: code
    the window moved between files is A's) or a hidden test's function name. Empty = clean."""
    blob = json.dumps(payload)
    strings: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            strings.append(x)
        elif isinstance(x, Mapping):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(payload)
    sent = {ln.strip() for x in strings for ln in x.splitlines()}
    out = []
    a_lines = {ln.strip() for f in Path(base_dir).rglob("*") if f.is_file() and f.suffix in (".py", ".md", ".rst", ".txt", ".cfg", ".toml", "")
               for ln in f.read_text(errors="replace").splitlines()} if Path(base_dir).is_dir() else set()
    for p, t in oracle_files.items():
        if t is None or not p.endswith(".py"):
            continue
        for ln in t.splitlines():
            s = ln.strip()
            if len(s) >= 40 and s not in a_lines and s in sent:  # a whole line: a B line can be a substring of an A line
                out.append(f"oracle line from {p}: {s[:80]}")
                break
    names = {n for txt in hidden_tests.values() for n in re.findall(r"^\s*def (test_\w+)", txt, re.MULTILINE)}
    a_text = "\n".join(f.read_text(errors="replace") for f in Path(base_dir).rglob("test*.py") if f.is_file())
    out += [f"hidden test name {n}" for n in sorted(names) if n not in a_text and re.search(rf"\b{re.escape(n)}\b", blob)]
    return out


# ---- the suite -------------------------------------------------------------------------------------------------------------
@dataclass
class ShadowSuite:
    files: dict[str, str]  # path -> module source
    requirements: dict[str, str]  # requirement id -> path
    model: str = ""
    family: str = ""
    notes: str = ""
    problems: list[str] = field(default_factory=list)
    usd: float = 0.0

    @property
    def sha256(self) -> str:
        return sha256_text(json.dumps(self.files, sort_keys=True))

    def requirement_of(self, test_id: str) -> str | None:
        path = test_id.split("::", 1)[0]
        return next((r for r, p in self.requirements.items() if p == path), None)

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "sha256": self.sha256}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> ShadowSuite:
        return cls(**{k: d[k] for k in ("files", "requirements", "model", "family", "notes", "problems", "usd") if k in d})


def parse_shadow(out: Any, requirement_ids: Sequence[str]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """The model's JSON -> (files, requirement -> path, problems). A module that does not parse, an unknown requirement
    id or a module with no test function is a problem (and left out)."""
    files: dict[str, str] = {}
    reqs: dict[str, str] = {}
    problems: list[str] = []
    rows = out.get("tests") if isinstance(out, Mapping) else None
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        rid, code = str(row.get("requirement") or ""), str(row.get("code") or "")
        if rid not in requirement_ids:
            problems.append(f"unknown requirement id {rid!r}")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(code)
        except SyntaxError as e:
            problems.append(f"{rid}: syntax error line {e.lineno}: {e.msg}")
            continue
        if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test") for n in ast.walk(tree)):
            problems.append(f"{rid}: no test function")
            continue
        path = shadow_path(rid)
        files[path] = (files[path] + "\n\n" + code) if path in files else code
        reqs[rid] = path
    problems += [f"{r}: no shadow module" for r in requirement_ids if r not in reqs]
    return files, reqs, problems


async def write_shadow(client: Any, *, prompt: str, requirements: Sequence[Mapping[str, Any]], base_dir: Path, model: str = "", family: str = "",
                       max_tokens: int = 32000, repair: bool = True, guard: Mapping[str, Any] | None = None, chunk: int = 6) -> ShadowSuite:
    """One call (plus one syntax-repair call when a module does not parse) of the second verifier's model; a request with
    more than ``chunk`` requirements is asked in groups of ``chunk`` (one reply for 13 requirements overran 32 k tokens),
    each group with the whole request and the same repository context. ``guard`` (``{"oracle_files", "hidden_tests"}``)
    runs ``leaks`` on the payload first and refuses to send a leaking one."""
    behaviour = [r for r in requirements if r.get("milestone")]
    if len(behaviour) > chunk:
        parts = [await write_shadow(client, prompt=prompt, requirements=behaviour[i:i + chunk], base_dir=base_dir, model=model, family=family,
                                    max_tokens=max_tokens, repair=repair, guard=guard, chunk=chunk) for i in range(0, len(behaviour), chunk)]
        return ShadowSuite({k: v for x in parts for k, v in x.files.items()}, {k: v for x in parts for k, v in x.requirements.items()}, model=model,
                           family=family, notes=" | ".join(x.notes for x in parts)[:2000], problems=sorted({p for x in parts for p in x.problems}))
    payload = shadow_payload(prompt, requirements, base_dir)
    if guard is not None:
        bad = leaks(payload, oracle_files=guard.get("oracle_files") or {}, hidden_tests=guard.get("hidden_tests") or {}, base_dir=base_dir)
        if bad:
            raise ValueError(f"the second verifier's payload leaks: {bad[:3]}")
    ids = [r["id"] for r in payload["requirements"]]
    msgs = [{"role": "system", "content": f"ROLE: second_verifier\n{SHADOW_BRIEF}\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps(payload)}]
    out = await call_supported(client.complete_json, msgs, max_tokens=max_tokens, temperature=0.0)
    files, reqs, problems = parse_shadow(out, ids)
    broken = [p for p in problems if "syntax error" in p or "no test function" in p]
    if repair and broken:
        fix = msgs + [{"role": "assistant", "content": json.dumps(out)[:20000]},
                      {"role": "user", "content": json.dumps({"fix_these_modules": broken, "return": "the same JSON shape, every requirement"})}]
        out2 = await call_supported(client.complete_json, fix, max_tokens=max_tokens, temperature=0.0)
        f2, r2, p2 = parse_shadow(out2, ids)
        for rid, path in r2.items():
            if rid not in reqs:
                files[path], reqs[rid] = f2[path], path
        problems = [p for p in p2 if not any(p.startswith(f"{r}:") for r in reqs)] + [p for p in problems if not any(p.startswith(f"{r}:") for r in reqs)]
    return ShadowSuite(files, reqs, model=model, family=family, notes=str((out or {}).get("notes") or "")[:2000], problems=sorted(set(problems)))


# ---- running both verifiers on one end state -------------------------------------------------------------------------------
class WindowRunner:
    """The window's grader runs for P2: the hidden tests (verifier 1, ``task.grade``), the shadow suite (verifier 2, its own
    run, import-tolerant, never beside the hidden tests) and the strict full suite (the executed gold label). Results are
    cached by (kind, end-state fingerprint)."""

    def __init__(self, task: Any):
        self.task = task
        self.cache: dict[tuple[str, str], Any] = {}
        self.roots = set(task.window.get("import_roots") or []) or project_roots(task.base_dir, list(task.window.get("oracle_files") or {}))

    def key(self, solution: Any) -> str:
        return self.task.end_fingerprint(solution)

    async def hidden(self, solution: Any, checks: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any] | None:
        """Verifier 1 exactly as ``WindowTask.grade`` runs it (the grader's test files, a fresh tree, the check tests'
        files as targets), keeping every test's status: ``{reward, milestones, phi, statuses, gates, apply_errors}``, or
        None when the run was masked. ``reward`` is the end-state verdict (1 = every gating check passes)."""
        from crucible.factory.lanes.windows import passed_group

        t = self.task
        check_tests = dict(checks) if checks is not None else dict(t.check_tests)
        k = ("v1", self.key(solution) + json.dumps(check_tests, sort_keys=True))
        if k not in self.cache:
            files, _, errors = t._files(solution)
            statuses: dict[str, str] = {}
            if not errors:
                targets = sorted({x.split("::", 1)[0] for ts in check_tests.values() for x in ts})
                tr = await run_tests(t.sandbox, t.base_dir, files, t.grader_tests, targets, t.window.get("tests_deleted") or ())
                if tr.infra != "ok":
                    self.cache[k] = None
                    return None
                statuses = tr.results
            gates = {c: ("pass" if ts and all(statuses.get(x) == "passed" for x in ts) else "fail") for c, ts in check_tests.items()}
            gating = [c for c in check_tests if t.gating.get(c, True)]
            ms = {m["id"]: passed_group(statuses, m["tests"]) for m in t.milestones}
            want = {x for ts in check_tests.values() for x in ts}
            self.cache[k] = {"reward": 1.0 if gating and all(gates[c] == "pass" for c in gating) else 0.0, "milestones": ms, "phi": t.phi(ms),
                             "gates": gates, "statuses": {x: statuses.get(x, "missing") for x in want}, "apply_errors": list(errors)}
        return self.cache[k]

    async def shadow(self, suite: ShadowSuite, solution: Any) -> dict[str, str] | None:
        """Verifier 2: shadow test id -> status, or None (masked / the solution does not apply)."""
        k = ("v2:" + suite.sha256, self.key(solution))
        if k not in self.cache:
            files, _, errors = self.task._files(solution)
            if errors:
                self.cache[k] = {}
            else:
                tests, _ = tolerant_tests(suite.files, self.roots)
                tr = await run_tests(self.task.sandbox, self.task.base_dir, files, tests, sorted(suite.files))
                self.cache[k] = tr.results if tr.infra == "ok" else None
        return self.cache[k]

    async def strict(self, solution: Any) -> dict[str, Any] | None:
        """The executed gold label: B's test files as written (strict imports), every suite target, in a fresh run; pass
        when every test that was stable-passing at B (milestone tests, pass-to-pass and collateral) passes."""
        k = ("strict", self.key(solution))
        if k not in self.cache:
            w = self.task.window
            files, _, errors = self.task._files(solution)
            want = sorted({t for m in w["milestones"] for t in m["tests"]} | set(w.get("p2p") or []) | set(w.get("collateral") or []))
            if errors:
                self.cache[k] = {"pass": False, "failing": ["(the solution does not apply)"], "n": len(want)}
            else:
                tr = await run_tests(self.task.sandbox, self.task.base_dir, files, dict(w.get("tests_b") or {}), list(w["targets"]), w.get("tests_deleted") or ())
                if tr.infra != "ok":
                    self.cache[k] = None
                else:
                    bad = [t for t in want if tr.results.get(t) != "passed"]
                    self.cache[k] = {"pass": not bad, "failing": bad[:30], "n": len(want)}
        return self.cache[k]


# ---- qualification ---------------------------------------------------------------------------------------------------------
@dataclass
class Qualification:
    tests: dict[str, str]  # shadow test id -> sound | fails_oracle | fails_alternative:<name> | unmeasured
    requirements: dict[str, dict[str, Any]]  # rid -> {sound, discriminates, mutants, state: gating | flag, why}

    def sound(self, rid: str) -> list[str]:
        return list((self.requirements.get(rid) or {}).get("sound") or [])

    def gating(self) -> list[str]:
        return sorted(r for r, v in self.requirements.items() if v["state"] == "gating")

    def verdict(self, statuses: Mapping[str, str] | None) -> dict[str, bool | None]:
        """Verifier 2's requirement verdicts on one end state: every sound test passes (None: no sound test, or the run
        was unmeasured)."""
        if statuses is None:
            return dict.fromkeys(self.requirements)
        return {r: (all(statuses.get(t) == "passed" for t in v["sound"]) if v["sound"] else None) for r, v in self.requirements.items()}

    def to_json(self) -> dict[str, Any]:
        return {"tests": self.tests, "requirements": self.requirements, "gating": self.gating(),
                "counts": {s: sum(1 for v in self.tests.values() if v.split(":")[0] == s) for s in ("sound", "fails_oracle", "fails_alternative", "unmeasured")}}


def qualify(suite: ShadowSuite, *, oracle: Mapping[str, str] | None, alternatives: Mapping[str, Mapping[str, str] | None], start: Mapping[str, str] | None,
            mutants: Mapping[str, tuple[str, Mapping[str, str] | None]], req_milestone: Mapping[str, str]) -> Qualification:
    """Sound tests pass the oracle and every valid alternative; a requirement gates when its sound tests fail the start
    state and every mutant of its milestone (``mutants``: name -> (milestone, statuses)), at least one such mutant
    measured. Everything else flags."""
    ids = sorted(oracle or {}) if oracle else []
    tests: dict[str, str] = {}
    for t in ids:
        if oracle is None or oracle.get(t) != "passed":
            tests[t] = "fails_oracle"
            continue
        bad = next((n for n, st in alternatives.items() if st is not None and st.get(t) != "passed"), None)
        tests[t] = f"fails_alternative:{bad}" if bad else "sound"
    if oracle is None:
        tests = {}
    reqs: dict[str, dict[str, Any]] = {}
    for rid in suite.requirements:
        sound = sorted(t for t, s in tests.items() if s == "sound" and suite.requirement_of(t) == rid)
        disc = None if start is None or not sound else not all(start.get(t) == "passed" for t in sound)
        ms = req_milestone.get(rid)
        caught = {n: (None if st is None or not sound else not all(st.get(t) == "passed" for t in sound)) for n, (m, st) in mutants.items() if m == ms}
        why = []
        if oracle is None:
            why.append("oracle unmeasured")
        if not sound:
            why.append("no sound test (each fails the oracle or a valid alternative, or none collected)")
        if disc is False:
            why.append("the sound tests pass the start state (they do not test new behaviour)")
        if not caught:
            why.append("no mutant of its milestone to fail")
        elif any(v is False for v in caught.values()):
            why.append(f"a mutant passes: {sorted(n for n, v in caught.items() if v is False)}")
        elif any(v is None for v in caught.values()):
            why.append("a mutant run unmeasured")
        reqs[rid] = {"sound": sound, "discriminates": disc, "mutants": caught, "state": "flag" if why else "gating", "why": why,
                     "tests": sorted(t for t in tests if suite.requirement_of(t) == rid)}
    return Qualification(tests, reqs)


# ---- the differential ------------------------------------------------------------------------------------------------------
TRUTH_PASS_KINDS = ("oracle", "alternative")
TRUTH_FAIL_KINDS = ("floor", "start")


@dataclass
class EndStateRec:
    """One end state both verifiers judge: ``kind`` oracle | alternative | floor | start | mutant | solver | hacker |
    rewrite_candidate; ``truth`` requirement -> known verdict (None = unknown) from construction."""

    id: str
    kind: str
    solution: Any
    truth: dict[str, bool | None] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def truth_for(kind: str, requirements: Sequence[str], *, milestone_of: Mapping[str, str], mutant: Mapping[str, Any] | None = None) -> dict[str, bool | None]:
    """The constructed truth per requirement: oracle/alternative pass, floor/start fail, a ``valid`` mutant follows its
    executed ``expected`` map (its own milestone fails; independent milestones pass; dependents unknown), a ``broad`` one
    is known only on its own milestone; real end states unknown."""
    if kind in TRUTH_PASS_KINDS:
        return dict.fromkeys(requirements, True)
    if kind in TRUTH_FAIL_KINDS:
        return dict.fromkeys(requirements, False)
    if kind == "mutant" and mutant is not None:
        own = mutant.get("milestone")
        exp = mutant.get("expected") or {own: False}
        if mutant.get("status") == "broad":  # the revert broke more than its unit: only its own milestone is known
            exp = {own: False}
        return {r: exp.get(milestone_of.get(r)) for r in requirements}
    return dict.fromkeys(requirements)


@dataclass
class Defect:
    id: str
    window: str
    end_state: str
    kind: str  # the end state's kind
    requirement: str
    milestone: str
    v1: bool
    v2: bool
    truth: bool | None
    owner: str  # verifier1 | verifier2 | spec
    defect: str  # verifier_gap | false_fail | weak_shadow | unsound_shadow | unstated_format
    confirmed: bool  # truth known: the owner is certain; else a candidate for the audit
    rule: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def assign_owner(*, v1: bool, v2: bool, truth: bool | None, shadow_gates: bool, spec_flagged: bool) -> tuple[str, str, bool, str]:
    """(owner, defect, confirmed, rule) for one disagreement (``v1 != v2``)."""
    if truth is not None:
        if v1 != truth:
            return "verifier1", ("verifier_gap" if v1 else "false_fail"), True, "truth known: verifier 1 contradicts it"
        return "verifier2", ("weak_shadow" if v2 else "unsound_shadow"), True, "truth known: verifier 2 contradicts it"
    if v1 and not v2:
        if shadow_gates:
            return "verifier1", "verifier_gap", False, "truth unknown, verifier 1 passes, a gating shadow check fails"
        return "verifier2", "unsound_shadow", False, "truth unknown, verifier 1 passes, the failing shadow check does not gate"
    if spec_flagged:
        return "spec", "unstated_format", False, "truth unknown, verifier 1 fails on a hidden test the fairness lint flags (unstated format)"
    return "verifier2", "weak_shadow", False, "truth unknown, verifier 1 (a real observation) fails, verifier 2 passes"


@dataclass
class DiffReport:
    defects: list[Defect] = field(default_factory=list)
    compared: int = 0  # (end state, requirement) pairs with both verdicts
    agree: int = 0
    unmeasured: list[str] = field(default_factory=list)
    states: int = 0

    def by_owner(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.defects:
            out[d.owner] = out.get(d.owner, 0) + 1
        return out

    def confirmed(self, owner: str | None = None) -> list[Defect]:
        return [d for d in self.defects if d.confirmed and (owner is None or d.owner == owner)]

    def to_json(self) -> dict[str, Any]:
        return {"defects": [d.to_json() for d in self.defects], "compared": self.compared, "agree": self.agree, "states": self.states,
                "unmeasured": self.unmeasured, "by_owner": self.by_owner(),
                "confirmed_by_owner": {o: len(self.confirmed(o)) for o in ("verifier1", "verifier2", "spec")}}


async def run_differential(runner: WindowRunner, suite: ShadowSuite, qual: Qualification, states: Sequence[EndStateRec], *,
                           requirements: Sequence[Mapping[str, Any]], spec_flags: Mapping[str, Sequence[str]] | None = None) -> DiffReport:
    """Both verifiers on every end state, requirement by requirement. ``spec_flags``: requirement -> hidden test ids the
    fairness lint flags ``unstated_format``."""
    task = runner.task
    rep = DiffReport()
    req_ms = {str(r["id"]): str(r["milestone"]) for r in requirements if r.get("milestone")}
    ms_tests = {m["id"]: m["tests"] for m in task.milestones}
    wid = str(task.window.get("id"))
    for s in states:
        v1 = await runner.hidden(s.solution)
        st2 = await runner.shadow(suite, s.solution)
        if v1 is None or st2 is None:
            rep.unmeasured.append(s.id)
            continue
        rep.states += 1
        v2 = qual.verdict(st2)
        for rid, mid in req_ms.items():
            a, b = v1["milestones"].get(mid), v2.get(rid)
            if a is None or b is None:
                continue
            rep.compared += 1
            if a == b:
                rep.agree += 1
                continue
            truth = s.truth.get(rid)
            failing1 = [t for t in ms_tests.get(mid, []) if v1["statuses"].get(t) != "passed"]
            flagged = bool(set(failing1) & set((spec_flags or {}).get(rid) or []))
            owner, kind, conf, rule = assign_owner(v1=bool(a), v2=bool(b), truth=truth, shadow_gates=rid in qual.gating(), spec_flagged=flagged)
            failing2 = [t for t in qual.sound(rid) if st2.get(t) != "passed"]
            rep.defects.append(Defect(f"{wid}:{s.id}:{rid}", wid, s.id, s.kind, rid, mid, bool(a), bool(b), truth, owner, kind, conf, rule,
                                      {"v1_failing": failing1[:10], "v2_failing": failing2[:10]}))
    return rep


def unsound_defects(window: str, qual: Qualification, suite: ShadowSuite, req_milestone: Mapping[str, str]) -> list[Defect]:
    """Shadow tests that fail the oracle or a valid alternative: confirmed verifier-2 defects (they never gate)."""
    out = []
    for t, s in qual.tests.items():
        if s == "sound":
            continue
        rid = suite.requirement_of(t) or ""
        where = "oracle" if s == "fails_oracle" else s.split(":", 1)[-1]
        out.append(Defect(f"{window}:{where}:{t}", window, where, "oracle" if s == "fails_oracle" else "alternative", rid, req_milestone.get(rid, ""),
                          True, False, True, "verifier2", "unsound_shadow", s != "unmeasured", "a shadow test fails a correct end state", {"test": t}))
    return out


# ---- machine handlers (S5, S8, S9) -----------------------------------------------------------------------------------------
@dataclass
class P2Gates:
    """What the S5/S8/S9 handlers read (``ctx.services``): the task (a ``LongTask``, ``WindowTask`` for the window lane),
    its runner, the accepted alternatives (name -> patch), the shadow suite and the end states for the differential."""

    task: Any
    runner: Any = None
    alternatives: dict[str, Any] = field(default_factory=dict)
    suite: ShadowSuite | None = None
    states: list[EndStateRec] = field(default_factory=list)
    lint: Any = None  # a fairness.LintReport over the task's checks (S5 computes it when absent)
    results: dict[str, Any] = field(default_factory=dict)  # each handler's full evidence


def _verdict(ok: bool, owner: str, findings: Sequence[str], evidence: Mapping[str, Any]) -> Any:
    from crucible.factory.machine import Settle

    v = {"ok": bool(ok), "owner": owner, "findings": list(findings)[:40], "evidence": json.loads(json.dumps(evidence, default=str))}
    return Settle({"verdict": v}, list(findings)[:40])


async def s5_solvability(ctx: Any) -> Any:
    """S5: the oracle passes every check from the clean World (``longtask.clean_world_solvable``), the null agent scores
    0, and the fairness lint refuses no check. Owner on failure: the verifier."""
    from crucible.factory.fairness import lint_checks
    from crucible.factory.longtask import clean_world_solvable, null_agent_score

    g: P2Gates = ctx.services
    t = g.task
    cw = await clean_world_solvable(t)
    null = await null_agent_score(t)
    lint = g.lint or lint_checks(t.checkset["checks"], t.window.get("tests_b") or {}, t.prompt)
    g.lint = lint
    findings = ([] if cw["ok"] else [f"unsolvable: the oracle scores reward {cw.get('reward')} phi {cw.get('phi')} from the clean World"])
    findings += [] if null["ok"] else [f"null agent scores above 0: {null['failed']}"]
    findings += [f"fairness {f.kind} on {f.check} ({f.test}): {f.evidence}" for f in lint.findings if f.severity == "refuse"]
    g.results["s5"] = {"clean_world": cw, "null_agent": null, "lint": lint.to_json()}
    return _verdict(not findings, "verifier", findings, {"clean_world_ok": cw["ok"], "null_ok": null["ok"], "refused_checks": lint.refused_checks(),
                                                         "spec_defects": len(lint.spec_defects())})


async def s8_ground_truth(ctx: Any) -> Any:
    """S8: floors 0 with Phi 0, every mutant fails its milestone, the oracle and every accepted alternative score 1, and
    every gating check passes the oracle and every alternative (the ten-competent-people test)."""
    from crucible.factory.fairness import ten_competent_people
    from crucible.factory.longtask import ground_truth

    g: P2Gates = ctx.services
    t = g.task
    gt = await ground_truth(t)
    runner = g.runner or WindowRunner(t)
    pos: dict[str, Any] = {"oracle": (await runner.hidden(t.oracle_patch()) or {}).get("statuses")}
    for name, sol in g.alternatives.items():
        pos[name] = (await runner.hidden(sol) or {}).get("statuses")
        r = (await t.grade_patch(sol, trail=[]))
        gt["rows"][name] = {"kind": "alternative", "reward": r.reward, "phi": (r.meta or {}).get("phi"), "ok": r.reward == 1.0}
        if r.reward != 1.0:
            gt["failed"].append(name)
    ff = ten_competent_people(t.check_tests, pos, gating=t.gating)
    findings = [f"ground truth fails: {x}" for x in gt["failed"]] + [f"false_fail: {f.check}: {f.evidence}" for f in ff]
    if len(g.alternatives) < 2:
        findings.append(f"{len(g.alternatives)} valid alternative(s), need 2 (one from another family)")
    g.results["s8"] = {"ground_truth": gt, "false_fail": [f.to_json() for f in ff]}
    return _verdict(not findings, "verifier", findings, {"failed": gt["failed"], "false_fail_checks": sorted({f.check for f in ff}),
                                                         "alternatives": sorted(g.alternatives)})


async def s9_second_verifier(ctx: Any) -> Any:
    """S9: shadow tests qualified (sound on the oracle and every alternative; gating when they fail the start state and the
    mutants) and the differential over every end state. Confirmed verifier-1 or spec defects send the task back to S5;
    verifier-2 defects only flag; unconfirmed candidates go to the audit."""
    g: P2Gates = ctx.services
    t = g.task
    if g.suite is None:
        return _verdict(False, "second_verifier", ["no shadow suite"], {})
    runner = g.runner or WindowRunner(t)
    reqs = list(t.requirements)
    req_ms = {str(r["id"]): str(r["milestone"]) for r in reqs if r.get("milestone")}
    oracle = await runner.shadow(g.suite, t.oracle_patch())
    alts = {n: await runner.shadow(g.suite, s) for n, s in g.alternatives.items()}
    start = await runner.shadow(g.suite, "")
    muts = {str(m.id): (str(m.milestone), await runner.shadow(g.suite, m.solution)) for m in t.mutants()}
    qual = qualify(g.suite, oracle=oracle, alternatives=alts, start=start, mutants=muts, req_milestone=req_ms)
    flags: dict[str, list[str]] = {}
    for f in getattr(g.lint, "findings", []) or []:
        if f.kind == "unstated_format":
            for r in f.requirement_ids:
                flags.setdefault(r, []).append(f.test)
    diff = await run_differential(runner, g.suite, qual, g.states, requirements=reqs, spec_flags=flags)
    blocking = [d for d in diff.defects if d.confirmed and d.owner in ("verifier1", "spec")]
    findings = [f"{d.owner} {d.defect} on {d.requirement} at {d.end_state}: v1={d.v1} v2={d.v2}" for d in blocking]
    g.results["s9"] = {"qualification": qual.to_json(), "differential": diff.to_json()}
    return _verdict(not blocking, "verifier" if blocking else "second_verifier", findings,
                    {"gating": qual.gating(), "defects": diff.by_owner(), "confirmed": {o: len(diff.confirmed(o)) for o in ("verifier1", "verifier2", "spec")},
                     "candidates": sum(1 for d in diff.defects if not d.confirmed)})


def gate_handlers() -> dict[str, Any]:
    """``{"s5_solvability", "s8_ground_truth", "s9_second_verifier"}`` for ``MachineRun(handlers=...)``."""
    return {"s5_solvability": s5_solvability, "s8_ground_truth": s8_ground_truth, "s9_second_verifier": s9_second_verifier}


def gate_machine(*, loop_to: str = "s5_solvability") -> dict[str, Any]:
    """S5 -> S8 -> S9 -> accepted; S8 and S9 failures go back to S5 (owner: the verifier), bounded; S5 failure escalates
    (the verifier writer's loop is S5 itself)."""
    v = [{"name": "verdict", "type": "json"}]
    states = [{"id": "s5_solvability", "fn": "s5_solvability", "outputs": v, "entry": True, "max_entries": 2},
              {"id": "s8_ground_truth", "fn": "s8_ground_truth", "outputs": v, "max_entries": 2},
              {"id": "s9_second_verifier", "fn": "s9_second_verifier", "outputs": v, "max_entries": 2},
              {"id": "accepted", "terminal": "done"}, {"id": "refused", "terminal": "rejected"}]
    trans = [{"from": "s5_solvability", "to": "s8_ground_truth", "when": "verdict.ok == true"},
             {"from": "s5_solvability", "to": "refused", "when": "verdict.ok == false"},
             {"from": "s8_ground_truth", "to": "s9_second_verifier", "when": "verdict.ok == true"},
             {"from": "s8_ground_truth", "to": loop_to, "when": "verdict.ok == false", "carry": "verdict.findings"},
             {"from": "s9_second_verifier", "to": "accepted", "when": "verdict.ok == true"},
             {"from": "s9_second_verifier", "to": loop_to, "when": "verdict.ok == false", "carry": "verdict.findings"}]
    return {"run": {"max_transitions": 20}, "states": states, "transitions": trans}


def shadow_session_state(*, sid: str = "second_verifier", isolated_from: Sequence[str] = ("oracle",), max_turns: int = 40) -> dict[str, Any]:
    """The second verifier as a P1 agentic session (``factory/sessions.py``): a read-only worktree of the repository at A,
    its own sandbox, the hand-off (prompt + requirements + spec), the shadow suite captured as ``shadow.json``. Not run in
    P2 (cost; see the module docstring); the machine validator accepts it."""
    return {"id": sid, "role": "second_verifier", "isolated_from": list(isolated_from), "mutates": True, "recovery": "reset",
            "inputs": [{"name": "prompt", "type": "text", "from": "author.prompt"}, {"name": "spec", "type": "json", "from": "spec.spec"}],
            "outputs": [{"name": "shadow", "type": "json"}],
            "session": {"brief": SHADOW_BRIEF + " Write shadow.json at the worktree root.", "captures": [{"output": "shadow", "kind": "json", "path": "shadow.json"}],
                        "max_turns": max_turns}}
