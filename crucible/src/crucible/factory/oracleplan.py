"""S4.5: the oracle against its plan, executed (swarm-pipeline/00 §4 S4-S5, §5.3; Phase A).

P5b's swarm-built cachetools window passed every gate with an oracle that skipped part of its plan: R3 asked to remove
``MRUCache`` and ``mru_cache``; the oracle only dropped them from ``__all__`` (they stayed, deprecated). The verifier's
removal tests were right and failed on that oracle, so the build kept only the two that passed, and 23 MRU tests stayed in
the regression set. A correct removal then scored Phi 1 with reward 0.5. Two rules close it, both executed:

1. **Plan probes** (``plan_probes``, deterministic, no model): each planned milestone's own text and public names give
   executable claims, run on A and on the oracle in the World (``observe``):

   | kind | from the plan | evidence on A | evidence on the oracle |
   |---|---|---|---|
   | ``removal`` | remove / delete / drop / "no longer" + a public name | the name exists in a module of A | the name exists in none of those modules (``from M import N`` fails) |
   | ``deprecation`` | deprecate + a public name (and no removal) | the name works, without a deprecation warning | the name still works **and** using it warns |
   | ``addition`` | a public name A has nowhere | absent | present in some module of the package |
   | ``option`` | parameter / option / argument / keyword + a name that is not module-level, on the milestone's callables | not a parameter of the callable | a parameter of the callable |

   Deprecation is told apart from removal by execution: the probe reads the name with warnings recorded and calls it with
   0, 1 or 2 small arguments; a ``DeprecationWarning`` / ``FutureWarning`` on the oracle where the plan says *remove* is
   ``deprecated_not_removed`` (the cachetools seed), a missing name where the plan says *deprecate* is
   ``removed_not_deprecated``. A behaviour change of an existing name has no deterministic probe: the verifier writer's
   hidden tests for it (fail on A, pass on the oracle) are the evidence, as before.

2. **Contradiction** (``contradicting_tests``): a test of A that uses a name the plan removes contradicts the plan. It is
   a defect, not a regression check: it leaves the regression set with its requirement (``drop_from_regression``), the
   builder is told it is expected to fail, and when it still **passes** on the oracle that is executed evidence the removal
   was not done.

A defect sends the oracle back to its builder (owner ``oracle_builder``); a probe that cannot run on A (the plan removes a
name A does not have) is the planner's. ``check`` returns ``{ok, probes, defects, findings, contradicting,
drop_from_regression}``. Runners: ``host_runner`` (tests: the host interpreter on a temporary tree) and ``world_runner``
(``windows_s3.WindowWorldPort.run`` in the window's World image, offline).
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import shlex
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REMOVE_RX = re.compile(r"\b(remov(?:e|es|ed|al|ing)|delet(?:e|es|ed|ion|ing)|drop(?:s|ped|ping)?|no longer (?:exists?|available|importable|exported|"
                       r"provided|present|appears?)|get rid of)\b", re.IGNORECASE)
DEPRECATE_RX = re.compile(r"\bdeprecat(?:e|es|ed|ion|ing)\b", re.IGNORECASE)
OPTION_RX = re.compile(r"\b(parameters?|options?|arguments?|keywords?|kwargs?|flags?)\b", re.IGNORECASE)
FROM_IMPORT_RX = re.compile(r"\bfrom\s+([A-Za-z_][\w.]*)\s+import\s+([A-Za-z_]\w*)")
MAX_MODULES = 120

Runner = Callable[[Mapping[str, str | None], str], Awaitable[tuple[int, str]]]


@dataclass
class Probe:
    milestone: str
    requests: list[str]
    kind: str  # removal | deprecation | addition | option
    name: str
    modules: list[str] = field(default_factory=list)  # import paths the plan states (``from M import N``)
    callables: list[str] = field(default_factory=list)  # option: the names whose signature gains it
    option_talk: bool = False  # the milestone speaks of a parameter / option / argument


def _text(m: Mapping[str, Any]) -> str:
    return " ".join(str(m.get(k) or "") for k in ("title", "requirement", "text", "observable"))


MARKER_RX = re.compile(r"^[`'\"]*\s*(?:\w+\s+){0,4}(?:is|are|be|being)?\s*(?:removed|deleted|dropped|deprecated|no longer)\b", re.IGNORECASE)
CONTAINER_RX = re.compile(r"\b(from|of|on|in)\s+[`'\"]*$", re.IGNORECASE)


def _targets(text: str, names: Sequence[str], verb: re.Pattern[str], *, window: int = 90) -> list[str]:
    """The names a sentence of ``text`` makes the object of ``verb``: named within ``window`` characters after the verb (not
    as the container in ``from N`` / ``of N`` / ``on N``), or just before "removed" / "deprecated" / "no longer"."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.;])\s+|\n", text):
        hits = list(verb.finditer(sent))
        if not hits:
            continue
        for n in names:
            for m in re.finditer(rf"(?<![\w.]){re.escape(n)}(?!\w)", sent):
                after_verb = any(0 <= m.start() - h.end() <= window for h in hits) and not CONTAINER_RX.search(sent[max(0, m.start() - 8):m.start()])
                marked = bool(MARKER_RX.match(sent[m.end():m.end() + 60]))
                if (after_verb or marked) and n not in out:
                    out.append(n)
    return out


def plan_probes(plan: Mapping[str, Any]) -> list[Probe]:
    """The executable claims of each planned milestone (module doc table). A public name (``name`` or ``Class.attr``) is a
    ``removal`` (or ``deprecation``) target when a sentence of the milestone makes it the object of the verb
    (``_targets``, on its last part); every other public name is a candidate ``addition``, decided after the observation
    (``judge``): present at A is a change of an existing name (the hidden tests are its evidence); absent from A and
    present in the oracle (a module, or a class of the package) is an addition; absent from both is checked as a
    parameter of the milestone's other names (an option), then against the sources (a runtime attribute)."""
    out: list[Probe] = []
    for m in plan.get("milestones") or []:
        text = _text(m)
        names = list(dict.fromkeys(str(n).strip() for n in m.get("public_names") or [] if str(n).strip()))
        stated: dict[str, list[str]] = {}
        for mod, nm in FROM_IMPORT_RX.findall(text):
            stated.setdefault(nm, []).append(mod)
        last = {n: n.split(".")[-1] for n in names}
        removed = [n for n in names if last[n] in _targets(text, list(last.values()), REMOVE_RX)]
        deprecated = [n for n in names if last[n] in _targets(text, list(last.values()), DEPRECATE_RX) and n not in removed]
        for n in names:
            kind = "removal" if n in removed else "deprecation" if n in deprecated else "addition"
            others = [x for x in names if x != n and x not in removed]
            out.append(Probe(str(m["id"]), [str(r) for r in m.get("requests") or []], kind, n, sorted(set(stated.get(last[n], []))),
                             others if kind in ("removal", "addition") else [], bool(OPTION_RX.search(text))))
    return out


# ---- what the package has: modules, names, warnings, signatures (executed) ---------------------------------------------------
def package_modules(tree: Path, extra_files: Sequence[str] = ()) -> list[str]:
    """Dotted names of the project's own modules under its top-level packages (``windows_world.workspace_modules``), at
    the root or under ``src/``, plus modules the oracle adds (``extra_files``); test modules out; at most ``MAX_MODULES``."""
    from crucible.factory.lanes.windows_world import workspace_modules

    tree = Path(tree)
    tops = set(workspace_modules(tree))
    mods: set[str] = set()
    paths = {str(f.relative_to(tree)) for f in tree.rglob("*.py") if f.is_file() and ".git" not in f.parts} | {p for p in extra_files if p.endswith(".py")}
    for rel in sorted(paths):
        parts = rel[:-3].split("/")
        if parts[0] == "src":
            parts = parts[1:]
        if not parts or parts[0] not in tops or any(p.startswith("test") or p in ("tests", "testing", "conftest") for p in parts):
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts and all(p.isidentifier() for p in parts):
            mods.add(".".join(parts))
    return sorted(mods, key=lambda x: (x.count("."), x))[:MAX_MODULES]


PROBE_SCRIPT = r'''
import importlib, inspect, json, os, sys, warnings
spec = json.loads(sys.argv[1])
root = os.getcwd()
sys.path[:0] = [os.path.join(root, "src"), root]
out = {"modules": {}, "names": {}, "use": {}, "signatures": {}}
loaded = {}
for mod in spec["modules"]:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            loaded[mod] = importlib.import_module(mod)
            out["modules"][mod] = "ok"
        except BaseException as e:
            out["modules"][mod] = type(e).__name__

def classes(mod, m):
    return [(k, c) for k, c in list(vars(m).items()) if isinstance(c, type) and getattr(c, "__module__", "") == mod]

def find(name):
    head, _, tail = name.partition(".")
    where = []
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        for mod, m in loaded.items():
            try:
                if tail:
                    c = vars(m).get(head)
                    if isinstance(c, type) and getattr(c, "__module__", "") == mod and tail in vars(c):
                        where.append(mod + ":" + head)
                elif name in vars(m) or hasattr(m, name):
                    where.append(mod)
            except BaseException:
                pass
        if not where and not tail:  # a method or class attribute of a class the package defines
            for mod, m in loaded.items():
                where += [mod + ":" + k for k, c in classes(mod, m) if name in vars(c)]
    return where

def resolve(name, where):
    mod, _, cls = where[0].partition(":")
    m = loaded[mod]
    return getattr(getattr(m, cls), name.split(".")[-1]) if cls else getattr(m, name)

for name in spec["names"]:
    out["names"][name] = find(name)
for name in spec["use"]:
    where = out["names"].get(name) or []
    if not where:
        out["use"][name] = {"present": False}
        continue
    row = {"present": True, "where": where[0], "warned": [], "called": None}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            obj = resolve(name, where)
            for args in ((), (2,), (2, 2)):
                if not callable(obj):
                    break
                try:
                    obj(*args)
                    row["called"] = len(args)
                    break
                except TypeError:
                    continue
                except BaseException as e:
                    row["called"] = "raised " + type(e).__name__
                    break
        except BaseException as e:
            row["error"] = type(e).__name__
        row["warned"] = sorted({c.category.__name__ for c in w if issubclass(c.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning))})
    out["use"][name] = row
for name, opt in spec["options"]:
    sig = {}
    where = out["names"].get(name) or []
    if where:
        try:
            obj = resolve(name, where)
            ps = inspect.signature(obj).parameters
            sig = {"where": where[0], "params": list(ps), "var_kw": any(p.kind == p.VAR_KEYWORD for p in ps.values())}
        except (TypeError, ValueError, AttributeError):
            sig = {}
    out["signatures"][name + "(" + opt + ")"] = sig
print("ORACLEPLAN " + json.dumps(out))
'''


def probe_command(spec: Mapping[str, Any]) -> str:
    return f"python -c {shlex.quote(PROBE_SCRIPT)} {shlex.quote(json.dumps(spec, sort_keys=True))}"


def parse_probe(out: str) -> dict[str, Any] | None:
    for ln in reversed(out.splitlines()):
        if ln.startswith("ORACLEPLAN "):
            return json.loads(ln[len("ORACLEPLAN "):])
    return None


def host_runner(base: Path, *, python: str = sys.executable, timeout_s: float = 120.0) -> Runner:
    """Runs a shell command on A (``base``) plus files, with the host interpreter (tests and local kernels)."""

    async def run(files: Mapping[str, str | None], command: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory(prefix="oracleplan-") as d:
            t = Path(d) / "t"
            shutil.copytree(base, t, symlinks=True, ignore=shutil.ignore_patterns(".git"))
            _overlay(t, files)
            cmd = command.replace("python -c", f"{shlex.quote(python)} -c", 1) if command.startswith("python -c") else command
            p = await asyncio.create_subprocess_shell(cmd, cwd=t, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                o, _ = await asyncio.wait_for(p.communicate(), timeout=timeout_s)
            except TimeoutError:
                p.kill()
                return -1, "timeout"
            return p.returncode or 0, o.decode(errors="replace")

    return run


def world_runner(port: Any, *, image: str | None = None) -> Runner:
    """Runs a shell command on A plus files in the window's World (``windows_s3.WindowWorldPort``: ``/w``, offline)."""

    async def run(files: Mapping[str, str | None], command: str) -> tuple[int, str]:
        return await port.run(f"cd /w && {command}", state={"files": dict(files)}, image=image or port.image)

    return run


def _overlay(tree: Path, files: Mapping[str, str | None]) -> None:
    for p, txt in files.items():
        f = tree / p
        if txt is None:
            f.unlink(missing_ok=True)
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(txt)


async def observe(runner: Runner, files: Mapping[str, str | None], spec: Mapping[str, Any]) -> dict[str, Any]:
    rc, out = await runner(files, probe_command(spec))
    got = parse_probe(out)
    if got is None:
        return {"ok": False, "exit": rc, "tail": out[-800:]}
    return {"ok": True, **got}


# ---- contradiction: tests of A that use a name the plan removes ----------------------------------------------------------------
def _names_in(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            out.add(x.id)
        elif isinstance(x, ast.Attribute):
            out.add(x.attr)
        elif isinstance(x, ast.alias):
            out.add((x.asname or x.name).split(".")[-1])
            out.add(x.name.split(".")[-1])
    return out


def contradicting_tests(removed: Sequence[str], test_sources: Mapping[str, str], test_ids: Sequence[str]) -> dict[str, list[str]]:
    """``{test id: [removed names it uses]}`` for the ids in ``test_ids`` whose test uses a name the plan removes: its own
    function, its class (with every base class defined in the same file), or the module's top level (a module-level import
    of the name fails the whole module once the name is gone)."""
    removed = [r for r in dict.fromkeys(removed) if r]
    if not removed:
        return {}
    by_file: dict[str, dict[str, Any]] = {}
    for path, text in test_sources.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        top = set()
        for n in tree.body:
            if not isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                top |= _names_in(n)
        by_file[path] = {"classes": classes, "funcs": funcs, "top": top}
    out: dict[str, list[str]] = {}
    for tid in test_ids:
        path, _, rest = tid.partition("::")
        f = by_file.get(path)
        if f is None:
            continue
        parts = [p.split("[", 1)[0] for p in rest.split("::") if p]
        used = set(f["top"])
        if len(parts) >= 2 and parts[0] in f["classes"]:
            seen: set[str] = set()
            todo = [parts[0]]
            while todo:
                c = todo.pop()
                if c in seen or c not in f["classes"]:
                    continue
                seen.add(c)
                node = f["classes"][c]
                used |= _names_in(node)
                todo += [b.id for b in node.bases if isinstance(b, ast.Name)]
        elif parts and parts[0] in f["funcs"]:
            used |= _names_in(f["funcs"][parts[0]])
        hit = [r for r in removed if r in used]
        if hit:
            out[tid] = hit
    return out


# ---- the check -----------------------------------------------------------------------------------------------------------------
@dataclass
class Defect:
    milestone: str
    requests: list[str]
    kind: str  # not_removed | deprecated_not_removed | removed_not_deprecated | not_deprecated | not_added | option_missing | contradicted_regression
    name: str
    owner: str  # oracle_builder | planner
    evidence: dict[str, Any]
    finding: str


def _spec(probes: Sequence[Probe], modules: Sequence[str]) -> dict[str, Any]:
    names = sorted({p.name for p in probes} | {c for p in probes for c in p.callables})
    mods = sorted(set(modules) | {m for p in probes for m in p.modules}, key=lambda x: (x.count("."), x))
    use = sorted({p.name for p in probes if p.kind in ("removal", "deprecation")})
    options = sorted({(c, p.name.split(".")[-1]) for p in probes for c in p.callables})
    return {"modules": mods, "names": names, "use": use, "options": [list(o) for o in options]}


def judge(probes: Sequence[Probe], at_a: Mapping[str, Any], at_b: Mapping[str, Any], *, in_source: Mapping[str, tuple[bool, bool]] | None = None
          ) -> tuple[list[dict[str, Any]], list[Defect]]:
    """Verdict per probe (module doc). ``in_source``: name -> (in A's source, in the oracle's source), the last resort
    for a name no module, class or signature exposes (an attribute set at run time: markdown's ``md.toc_tokens``)."""
    rows: list[dict[str, Any]] = []
    defects: list[Defect] = []
    a_names, b_names = at_a.get("names") or {}, at_b.get("names") or {}
    in_source = in_source or {}

    def bad(p: Probe, kind: str, owner: str, finding: str, **ev: Any) -> None:
        defects.append(Defect(p.milestone, p.requests, kind, p.name, owner, ev, finding))

    for p in probes:
        wa, wb = list(a_names.get(p.name) or []), list(b_names.get(p.name) or [])
        row: dict[str, Any] = {**asdict(p), "at_a": wa, "at_b": wb}
        if p.kind == "removal":
            opt = p.name.split(".")[-1]
            sig_a = {k: v for k, v in (at_a.get("signatures") or {}).items() if k.endswith(f"({opt})") and opt in (v.get("params") or [])}
            if not wa and not p.modules and sig_a:  # an option the plan removes from the milestone's callables
                sig_b = {k: v for k, v in (at_b.get("signatures") or {}).items() if k.endswith(f"({opt})") and opt in (v.get("params") or [])}
                row.update(kind="option_removal", verdict="ok" if not sig_b else "option_not_removed")
                if sig_b:
                    bad(p, "option_not_removed", "oracle_builder", f"{p.milestone} ({', '.join(p.requests)}) plans the removal of the option {p.name}; "
                        f"the oracle's {', '.join(k.split('(')[0] for k in sig_b)} still accept it", signatures=sig_b)
            elif not wa and not p.modules:
                row["verdict"] = "unverifiable"
                bad(p, "removal_of_absent_name", "planner", f"{p.milestone} plans the removal of {p.name}, which the repository does not have", at_a=wa)
            else:
                still = [m for m in wb if m in (set(wa) | set(p.modules))]
                use = (at_b.get("use") or {}).get(p.name) or {}
                if still:
                    dep = bool(use.get("warned"))
                    row["verdict"] = "deprecated_not_removed" if dep else "not_removed"
                    how = f" (it now warns: {', '.join(use['warned'])}): deprecated, not removed" if dep else ""
                    bad(p, row["verdict"], "oracle_builder",
                        f"{p.milestone} ({', '.join(p.requests)}) plans the removal of {p.name}, but the oracle still provides it in "
                        f"{', '.join(still)}{how}; remove it (importing it must fail)", still_in=still, use=use)
                else:
                    row["verdict"] = "ok"
        elif p.kind == "deprecation":
            ua, ub = (at_a.get("use") or {}).get(p.name) or {}, (at_b.get("use") or {}).get(p.name) or {}
            if not wb:
                row["verdict"] = "removed_not_deprecated" if wa else "unverifiable"
                if wa:
                    bad(p, "removed_not_deprecated", "oracle_builder", f"{p.milestone} plans to deprecate {p.name}; the oracle removed it instead: keep it "
                        "working and warn", at_a=wa)
            elif ub.get("warned") and not ua.get("warned"):
                row["verdict"] = "ok"
            elif ub.get("called") is None and not ub.get("warned"):
                row["verdict"] = "unexercised"  # the probe could not call it: the hidden tests are the evidence
            else:
                row["verdict"] = "not_deprecated"
                bad(p, "not_deprecated", "oracle_builder", f"{p.milestone} plans to deprecate {p.name}; using it on the oracle gives no deprecation "
                    "warning", use=ub)
        else:  # addition, an option of the milestone's callables, or a runtime attribute
            opt = p.name.split(".")[-1]
            sigs = {k: v for k, v in (at_b.get("signatures") or {}).items() if k.endswith(f"({opt})") and v}
            sig_a = {k: v for k, v in (at_a.get("signatures") or {}).items() if k.endswith(f"({opt})") and v}
            has_b = [k for k, v in sigs.items() if opt in (v.get("params") or [])]
            has_a = [k for k, v in sig_a.items() if opt in (v.get("params") or [])]
            src_a, src_b = in_source.get(p.name, (False, False))
            if wa:
                row["verdict"] = "exists_at_start"  # a change of an existing name: the hidden tests are the evidence
            elif wb:
                row["verdict"] = "ok"
            elif has_b or has_a:
                row.update(kind="option", signatures_b={k: sigs[k] for k in has_b})
                row["verdict"] = "ok" if has_b and not has_a else "exists_at_start"
            elif src_a:
                row["verdict"] = "exists_at_start"  # set at run time (no module, class or signature has it), already in A's source
            elif any(v.get("var_kw") for v in sigs.values()) or src_b:
                row["verdict"] = "unexercised"  # behind **kwargs, or set at run time in the oracle's source: the hidden tests decide
            elif p.callables and p.option_talk:
                row["verdict"] = "option_missing"
            else:
                row["verdict"] = "not_added"
            if row["verdict"] in ("option_missing", "not_added"):
                what = (f"the option {opt} on {', '.join(p.callables)}: no oracle signature has it" if row["verdict"] == "option_missing"
                        else f"the public name {p.name}: no module or class of the oracle provides it, and its source never mentions it")
                bad(p, row["verdict"], "oracle_builder", f"{p.milestone} ({', '.join(p.requests)}) plans {what}", signatures=sigs, at_b=wb)
        rows.append(row)
    return rows, defects


def source_mentions(names: Sequence[str], base: Path, oracle_files: Mapping[str, str | None]) -> dict[str, tuple[bool, bool]]:
    """name -> (A's library source mentions it as an identifier, the oracle's does): tests left out."""
    base = Path(base)
    a = {str(f.relative_to(base)): f.read_text(errors="replace") for f in base.rglob("*.py") if f.is_file() and ".git" not in f.parts}
    b = {**a, **{k: v for k, v in oracle_files.items() if v is not None and k.endswith(".py")}}
    for k, v in oracle_files.items():
        if v is None:
            b.pop(k, None)

    def lib(files: Mapping[str, str]) -> str:
        return "\n".join(t for k, t in files.items() if not any(x.startswith("test") or x == "conftest.py" for x in k.split("/")))

    ta, tb = lib(a), lib(b)
    out = {}
    for n in names:
        rx = re.compile(rf"(?<![\w]){re.escape(n.split('.')[-1])}(?!\w)")
        out[n] = (bool(rx.search(ta)), bool(rx.search(tb)))
    return out


async def check(plan: Mapping[str, Any], base: Path, oracle_files: Mapping[str, str | None], *, runner: Runner, regression_ids: Sequence[str] = (),
                test_sources: Mapping[str, str] | None = None, regression_on_oracle: Mapping[str, str] | None = None) -> dict[str, Any]:
    """S4.5 on one oracle (module doc). ``regression_ids``: A's passing tests the window would keep as regression checks;
    ``test_sources``: their files' text (default: read from ``base``); ``regression_on_oracle``: their statuses on the
    oracle when already run (a contradicting test that still passes is evidence of an undone removal)."""
    base = Path(base)
    probes = plan_probes(plan)
    spec = _spec(probes, package_modules(base, list(oracle_files)))
    at_a = await observe(runner, {}, spec)
    at_b = await observe(runner, oracle_files, spec)
    if not at_a.get("ok") or not at_b.get("ok"):
        why = f"the plan probe did not run ({'A' if not at_a.get('ok') else 'oracle'}): {(at_a if not at_a.get('ok') else at_b).get('tail', '')[-300:]}"
        return {"ok": False, "infra": True, "findings": [why], "defects": [], "probes": [], "contradicting": {}, "drop_from_regression": []}
    rows, defects = judge(probes, at_a, at_b, in_source=source_mentions([p.name for p in probes], base, oracle_files))
    removed = [p.name for p in probes if p.kind == "removal"]
    if test_sources is None:
        files = sorted({t.split("::", 1)[0] for t in regression_ids})
        test_sources = {f: (base / f).read_text(errors="replace") for f in files if (base / f).is_file()}
    contra = contradicting_tests(removed, test_sources, regression_ids)
    still = sorted(t for t in contra if (regression_on_oracle or {}).get(t) == "passed")
    if still:
        by_name: dict[str, list[str]] = {}
        for t in still:
            for n in contra[t]:
                by_name.setdefault(n, []).append(t)
        for n, ts in by_name.items():
            p = next(x for x in probes if x.kind == "removal" and x.name == n)
            defects.append(Defect(p.milestone, p.requests, "contradicted_regression", n, "oracle_builder", {"tests": ts[:20], "n": len(ts)},
                                  f"{len(ts)} existing test(s) that use {n}, which {p.milestone} removes, still pass on the oracle (e.g. "
                                  f"{ts[0].split('::', 1)[-1]}): the removal is not done"))
    findings = list(dict.fromkeys(d.finding for d in defects))
    return {"ok": not defects, "infra": False, "probes": rows, "defects": [asdict(d) for d in defects], "findings": findings,
            "contradicting": contra, "drop_from_regression": sorted(contra), "spec": spec,
            "owner": "planner" if defects and all(d.owner == "planner" for d in defects) else ("oracle_builder" if defects else "")}
