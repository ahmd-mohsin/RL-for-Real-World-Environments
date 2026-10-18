"""Valid alternatives (P2; swarm-pipeline/00 §1 ``alternatives()``, §5.4, §5.8 "the ten-competent-people test").

A check is fair only if every correct solution passes it, so the factory needs correct solutions other than the
oracle: at least 2 per window, at least one built by a model family other than the verifier writer's. Two sources:

- **(a) solver end states** that score full on the real tests *and* that the second verifier confirms
  (``from_solver_states``);
- **(b) a model rewrite of the oracle in a different style** (``rewrite_oracle``): the functions and methods the window
  changed (``changed_defs``: B's definitions whose source differs from A's, or that A lacks) are sent, with a style
  brief, to a rewriter model; each rewritten definition is spliced back over B's (re-indented, the names kept) and the
  result must parse and differ from B. A candidate is **accepted only if the real tests and the second verifier both
  pass it** (``accept``); a candidate the real tests fail is kept as a ``rewrite_candidate`` end state for the
  differential (a false-fail candidate if the second verifier passes it).

Two style briefs (``STYLES``) push the rewrites apart: an imperative one and a functional one, both renaming locals.
The rewriter sees the oracle (it is a builder role); the second verifier never sees a rewrite (V2).
"""

from __future__ import annotations

import ast
import json
import textwrap
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.lanes.executable import call_supported, sha256_text

STYLES: dict[str, str] = {
    "imperative": ("Rewrite each definition in a plainly imperative style: explicit loops instead of comprehensions where it reads "
                   "naturally, early returns, intermediate local variables with new descriptive names. Rename every local variable."),
    "functional": ("Rewrite each definition in a compact functional style: comprehensions, generator expressions, helpers from the "
                   "standard library (itertools, functools, operator) where they fit, fewer intermediate variables, new short names "
                   "for every local variable."),
}
REWRITE_BRIEF = ("You refactor Python code without changing its behaviour. You get definitions (functions and methods) from a "
                 "library, each with its qualified name and its source. Return each one rewritten in the requested style: the same "
                 "name, the same parameters (names, defaults, order, *args/**kwargs), the same decorators, the same return values, "
                 "raised exception types and messages, side effects and docstring meaning; the same public behaviour for every "
                 "input. Change the implementation's form, not what it does. Do not add imports unless the definition imports "
                 "inside itself. Return JSON {\"defs\": [{\"qualname\": str, \"source\": str}]}, each source starting at column 0 "
                 "(a method without its class).")
MAX_DEF_CHARS = 5000
REWRITE_CHARS = 12_000  # 24 k overran a 16 k-token reply on python-semver (40 definitions)


@dataclass(frozen=True)
class DefSpan:
    path: str
    qualname: str  # "func" or "Class.method"
    start: int  # 1-based first line (decorators included)
    end: int  # 1-based last line
    indent: int
    source: str  # dedented


@dataclass
class Alternative:
    id: str
    source: str  # rewrite | solver | oracle_steps
    patch: str
    model: str = ""
    family: str = ""
    style: str = ""
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    usd: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Alternative:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _parse(text: str) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(text)
    except SyntaxError:
        return None


def _defs(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = n
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{n.name}.{m.name}"] = m
    return out


def _dedent(lines: Sequence[str], indent: int) -> str:
    """Remove the definition's own indentation from each line (at most ``indent`` spaces: a docstring line with less
    indentation keeps its text; ``textwrap.dedent`` would leave the whole block indented and unparsable)."""
    return "\n".join(ln[min(indent, len(ln) - len(ln.lstrip(" "))):] for ln in lines)


def _span(node: Any) -> tuple[int, int]:
    start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
    return start, node.end_lineno or node.lineno


def changed_defs(path: str, a_text: str | None, b_text: str) -> list[DefSpan]:
    """B's functions and methods (one level into top-level classes) whose source differs from A's, or that A lacks."""
    b_tree = _parse(b_text)
    if b_tree is None:
        return []
    a_tree = _parse(a_text) if a_text else None
    a_src = {}
    if a_tree is not None:
        a_lines = (a_text or "").splitlines()
        for q, n in _defs(a_tree).items():
            s, e = _span(n)
            a_src[q] = _dedent(a_lines[s - 1:e], len(a_lines[s - 1]) - len(a_lines[s - 1].lstrip()))
    lines = b_text.splitlines()
    out = []
    for q, n in _defs(b_tree).items():
        s, e = _span(n)
        indent = len(lines[s - 1]) - len(lines[s - 1].lstrip())
        src = _dedent(lines[s - 1:e], indent)
        if a_src.get(q) == src:
            continue
        out.append(DefSpan(path, q, s, e, indent, src))
    return out


def pick_defs(oracle_files: Mapping[str, str | None], base_dir: Path, *, budget: int = REWRITE_CHARS, min_lines: int = 3) -> list[DefSpan]:
    """The changed definitions to rewrite, within ``budget`` characters: small ones skipped (one-liners say nothing about
    style), oversized ones skipped, the rest in order of size (largest first) until the budget is spent."""
    cands: list[DefSpan] = []
    for p, t in sorted(oracle_files.items()):
        if t is None or not p.endswith(".py"):
            continue
        a = Path(base_dir) / p
        cands += changed_defs(p, a.read_text(errors="replace") if a.is_file() else None, t)
    cands = [d for d in cands if d.source.count("\n") + 1 >= min_lines and len(d.source) <= MAX_DEF_CHARS]
    out, left = [], budget
    for d in sorted(cands, key=lambda d: (-len(d.source), d.path, d.qualname)):
        if len(d.source) <= left:
            out.append(d)
            left -= len(d.source)
    return sorted(out, key=lambda d: (d.path, d.start))


def splice(text: str, spans: Sequence[DefSpan], new: Mapping[str, str]) -> tuple[str, list[str]]:
    """``text`` with each span's lines replaced by its rewritten source (re-indented). Returns (text, problems): a
    rewrite that does not parse, renames the definition or changes its parameters is skipped (kept as B's)."""
    lines = text.splitlines(keepends=True)
    problems: list[str] = []
    for d in sorted(spans, key=lambda d: -d.start):
        src = new.get(f"{d.path}::{d.qualname}")
        if src is None:
            continue
        src = src.strip("\n") + "\n"
        if src[:1] in (" ", "\t"):
            src = textwrap.dedent(src)
        tree = _parse(src)
        fn = next((n for n in (tree.body if tree else []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        old = _parse(d.source)
        old_fn = next((n for n in (old.body if old else []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if fn is None or old_fn is None:
            problems.append(f"{d.path}::{d.qualname}: the rewrite is not one function")
            continue
        if fn.name != old_fn.name or ast.dump(fn.args) != ast.dump(old_fn.args) or [ast.dump(x) for x in fn.decorator_list] != [
                ast.dump(x) for x in old_fn.decorator_list]:
            problems.append(f"{d.path}::{d.qualname}: the rewrite renames it or changes its parameters or decorators")
            continue
        pad = " " * d.indent  # the inverse of ``_dedent`` (only docstring whitespace can differ, which ``same_code`` ignores)
        block = "".join((pad + ln) if ln.strip() else ln for ln in src.splitlines(keepends=True))
        lines[d.start - 1:d.end] = [block]
    out = "".join(lines)
    if _parse(out) is None:
        return text, [*problems, "the spliced module does not parse"]
    return out, problems


def _code_dump(tree: ast.Module) -> str:
    for n in ast.walk(tree):  # docstrings out: re-indenting a method can change only their whitespace
        body = getattr(n, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body[0] = ast.Pass()
    return ast.dump(tree)


def same_code(a: str, b: str) -> bool:
    """Equal up to formatting, comments and docstrings (the AST dump)."""
    ta, tb = _parse(a), _parse(b)
    return ta is not None and tb is not None and _code_dump(ta) == _code_dump(tb)


async def rewrite_oracle(client: Any, task: Any, *, style: str, model: str = "", family: str = "", max_tokens: int = 32000,
                         min_lines: int = 3) -> Alternative:
    """One rewrite candidate (not yet accepted): the oracle with its changed definitions rewritten in ``style``."""
    files = dict(task.window["oracle_files"])
    spans = pick_defs(files, task.base_dir, min_lines=min_lines)
    aid = f"rewrite-{style}-{family or 'model'}"
    if not spans:
        return Alternative(aid, "rewrite", "", model, family, style, reasons=["no changed definition to rewrite"])
    payload = {"style": STYLES[style], "defs": [{"qualname": f"{d.path}::{d.qualname}", "source": d.source} for d in spans]}
    msgs = [{"role": "system", "content": f"ROLE: rewriter\n{REWRITE_BRIEF}\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps(payload)}]
    out = await call_supported(client.complete_json, msgs, max_tokens=max_tokens, temperature=0.2)
    got = {str(r.get("qualname")): str(r.get("source") or "") for r in (out or {}).get("defs") or [] if isinstance(r, Mapping)}
    new_files, problems = rebuild(files, spans, got)
    changed = [p for p in new_files if new_files[p] != files[p] and not same_code(new_files[p] or "", files[p] or "")]
    if not changed:
        return Alternative(aid, "rewrite", "", model, family, style, reasons=["the rewrite is the oracle (no definition changed)", *problems[:5]])
    patch = task.patch_of(new_files)
    sent = {f"{d.path}::{d.qualname}" for d in spans}
    keep = {k for k in got if k in sent and not any(pr.startswith(k + ":") for pr in problems)}  # the definitions actually spliced
    return Alternative(aid, "rewrite", patch, model, family, style, reasons=problems[:10],
                       evidence={"defs_sent": len(spans), "defs_returned": len(got), "files_changed": changed, "patch_sha256": sha256_text(patch),
                                 "rewritten": {k: v for k, v in got.items() if k in keep}})


def rebuild(files: Mapping[str, str | None], spans: Sequence[DefSpan], rewritten: Mapping[str, str]) -> tuple[dict[str, str | None], list[str]]:
    """The oracle's files with ``rewritten`` (``path::qualname`` -> source) spliced in."""
    out = dict(files)
    problems: list[str] = []
    for p in sorted({d.path for d in spans}):
        text, probs = splice(files[p] or "", [d for d in spans if d.path == p], rewritten)
        problems += probs
        out[p] = text
    return out, problems


async def salvage(task: Any, alt: Alternative, passes: Any, *, chunks: int = 4, min_lines: int = 3) -> Alternative | None:
    """A rewrite the real tests fail may still hold correct rewrites of most definitions: split its rewritten definitions
    into ``chunks`` and add each chunk to a growing set while the real tests (``passes(patch) -> bool``) keep passing (at
    most ``chunks`` grader runs, no model call). Returns the largest such candidate (a new id, ``...-subset``), or None."""
    rewritten = dict(alt.evidence.get("rewritten") or {})
    if len(rewritten) < 2:
        return None
    files = dict(task.window["oracle_files"])
    spans = [d for d in pick_defs(files, task.base_dir, min_lines=min_lines) if f"{d.path}::{d.qualname}" in rewritten]
    keys = sorted(rewritten)
    size = -(-len(keys) // chunks)
    good: dict[str, str] = {}
    runs = 0
    for i in range(0, len(keys), size):
        trial = {**good, **{k: rewritten[k] for k in keys[i:i + size]}}
        new_files, _ = rebuild(files, spans, trial)
        if same_code("".join(str(v) for v in new_files.values()), "".join(str(v) for v in files.values())):
            continue
        runs += 1
        if await passes(task.patch_of(new_files)):
            good = trial
    if not good:
        return None
    new_files, _ = rebuild(files, spans, good)
    patch = task.patch_of(new_files)
    return Alternative(f"{alt.id}-subset", "rewrite", patch, alt.model, alt.family, alt.style, reasons=[f"salvaged {len(good)}/{len(keys)} definitions"],
                       evidence={"defs_sent": alt.evidence.get("defs_sent"), "defs_returned": len(good), "salvage_runs": runs, "parent": alt.id,
                                 "patch_sha256": sha256_text(patch), "rewritten": good})


async def accept(alt: Alternative, *, v1: Mapping[str, Any] | None, v2: Mapping[str, bool | None] | None) -> Alternative:
    """Accept a candidate only when the real tests pass it (``v1["reward"] == 1``) and the second verifier passes it
    (every requirement with a verdict True, at least one verdict)."""
    reasons = []
    if v1 is None:
        reasons.append("the real tests' run was unmeasured")
    elif v1.get("reward") != 1.0:
        failing = sorted(k for k, ok in (v1.get("milestones") or {}).items() if not ok)
        reasons.append(f"the real tests fail it (milestones {failing}; regression {v1.get('gates', {}).get('c_regress')})")
    verdicts = [v for v in (v2 or {}).values() if v is not None]
    if v2 is None or not verdicts:
        reasons.append("the second verifier has no verdict on it")
    elif not all(verdicts):
        reasons.append(f"the second verifier fails it on {sorted(k for k, v in (v2 or {}).items() if v is False)}")
    alt.accepted = not reasons
    alt.reasons = [*alt.reasons, *reasons]
    alt.evidence.update(v1_reward=None if v1 is None else v1.get("reward"), v2=dict(v2 or {}))
    return alt


def from_solver_states(states: Sequence[Mapping[str, Any]]) -> list[Alternative]:
    """(a) Solver end states as alternative candidates: ``{"id", "patch", "policy", "v1", "v2"}`` each; accepted when the
    real tests and the second verifier both pass it (the same rule as a rewrite)."""
    out = []
    for s in states:
        policy = str(s.get("policy") or "")
        fam = policy.split(":", 1)[-1].split("/", 1)[0] if policy else ""
        a = Alternative(f"solver-{s['id']}", "solver", str(s.get("patch") or ""), policy, fam)
        v2 = s.get("v2")
        v1 = s.get("v1")
        a.accepted = bool(v1 and v1.get("reward") == 1.0 and v2 and [x for x in v2.values() if x is not None] and all(x for x in v2.values() if x is not None))
        if not a.accepted:
            a.reasons.append("not a confirmed pass (real tests and second verifier)")
        out.append(a)
    return out


def coverage(alts: Sequence[Alternative], *, writer_family: str) -> dict[str, Any]:
    """The S8 alternative rule: >= 2 accepted, >= 1 from a family other than ``writer_family`` (the family that wrote the
    task: its prompt author). A salvaged subset counts once with its parent (``evidence.parent``)."""
    acc = [a for a in alts if a.accepted]
    roots: dict[str, Alternative] = {}
    for a in acc:
        roots.setdefault(str(a.evidence.get("parent") or a.id), a)
    other = [a for a in roots.values() if a.family and a.family != writer_family]
    fams = sorted({a.family for a in roots.values() if a.family})
    return {"accepted": len(roots), "other_family": len(other), "families": fams, "ok": len(roots) >= 2 and len(other) >= 1, "ids": [a.id for a in acc]}
