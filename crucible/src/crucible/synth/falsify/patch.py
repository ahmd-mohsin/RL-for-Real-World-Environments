"""The falsifier population on the repository substrate (part2-paper-core §5 on ``synth.patch``; M5, E7).

The classes keep §5's names, so certificates and the archive read the same; their operators are the repo's own. Every
probe is a program of the patch World's tools, graded by the trusted grader through the lane's exec runner, or a
property probe over the task's data. An unmeasured grade (``GRADER_ERROR``: a sandbox that did not start) makes the
probe invalid: it is not a trial.

| Class | Operators | Label rule |
|---|---|---|
| MUT | ``Identity``; ``DropFile`` (one source file of B reverted); ``DropHunk`` (one changed region of a source file reverted, files with 2+ code regions, at most 16 sampled; comment-only regions and non-source files B also changes, such as spec text, are never proposed); ``RevertFix`` (the lane's commit-revert mutant of one milestone); ``ExtraSource`` (B plus an inert comment in a source file B does not touch); ``ExtraForbidden`` (B plus an inert edit to a test, golden, config, vendored, plumbing or VCS path); ``EditTest`` (B plus an edit of a hidden test file); ``SideEffect`` (B plus one flipped comparison or boolean literal on a line at least 3 lines from B's changes, in a source file B changes, else in an untouched source file of its directory: an unrequested behaviour change, Part 1 V1); ``SharedCheck`` (two milestones graded by one test) | Identity must pass (else D10 when a required check fails, D6 otherwise). A DropFile/DropHunk survivor is a **suspect** D1 (the hidden tests do not need that code; whether it matters is a judgement). A RevertFix survivor, or one whose own milestone still passes, is a confirmed D1; one that also fails a milestone that does not require it is a suspect D6. ExtraSource must pass (FORBIDDEN: confirmed D6; another failure: suspect D6). ExtraForbidden and EditTest must fail (a pass: confirmed D2). A SideEffect survivor is a suspect D2 (the hidden tests miss an unrequested behaviour change; a flip in dead code changes nothing, so a judgement decides). A shared test is a confirmed D6 at milestone scope: one milestone's fix is rejected until the other's is in |
| NULL | ``noop``, ``claims_only`` (STATUS.json), ``minus_last_write``, ``delete_tests``, ``stub_tests``, ``test_edit`` (the lane's own floor), ``plumbing_shim`` (conftest.py, go.work, .npmrc, .cargo config), ``path_shim`` (a fake runner on ``bin/``) | must fail; a pass is confirmed D4 (noop, claims), D1 (minus last write) or D8 |
| RED | §5's RED (start state; each ``require`` exec check on s₀). ``guard`` checks (the pass-to-pass suite) are not probed | confirmed D4 |
| TAINT | ``ledger`` (every milestone is asked for by a requirement of the report); ``world_leak`` (a fix's issue reference in a changelog line of A′, or a line holding the subject's content words: all of 5, all but one from 6, the lane's own strip rule); ``visible_contradicts`` (a visible test that passes at A′ fails at B: a stale assertion, executed twice); ``report_literal`` (an output string the fix adds and a hidden test compares, found neither in the report nor anywhere in A′); ``visible_data`` (static, ``visible_data()``: a fix-added literal a hidden test compares, or an identifier the fix defines, already readable in a test, golden or doc file; or two rare hidden-test data lines held by another readable file); ``visible_check`` (executed: a visible test that fails at A′ and passes at B on two runs of A′, the mirror of a stale test) | ledger: confirmed D5; world_leak: confirmed D7; visible_contradicts: confirmed D6; visible_check: confirmed D7; visible_data: confirmed D7 on fix answers, suspect D7 on shared case lines only; report_literal is **report-only** (labelled clean with ``hit``): the literal may be derivable from the code |
| REP | fresh oracle and start-state grades compared with the first (cached) grade; the lane's stored S10 oracle grades and S3 reruns as prior trials when given | any difference: confirmed D11 |

TWIN and PERT do not apply (a symptom report names no entity to substitute, and the start tree is a mined commit with no
distractor entities); ALT and ATK are paid and not run here; ONL applies online (``falsify.online``).
"""

from __future__ import annotations

import asyncio
import difflib
import io
import json
import random
import re
import tokenize
from collections.abc import Collection, Mapping, Sequence
from typing import Any, ClassVar

from crucible.synth.certificate import AdmissionPolicy, Certificate, certify
from crucible.synth.falsify.base import BaseFalsifier, ClassResult, FalsifyContext, Finding, Label, Outcome, Probe, dedupe_programs, run_classes
from crucible.synth.falsify.mut import MUT
from crucible.synth.falsify.red import RED
from crucible.synth.patch import ALLOWED_ROLES, FORBIDDEN_ROLES, ExecRunner, PatchEnv, cosmetic, diff_hunks, files_key, revert_hunk
from crucible.synth.program import Program
from crucible.synth.task import Task

SOURCE_EXT = {"python": (".py",), "go": (".go",), "node": (".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".jsx", ".tsx"), "rust": (".rs",),
              "java": (".java",)}
COMMENT = {"python": "#", "go": "//", "node": "//", "rust": "//", "java": "//"}
MAX_HUNKS = 16
INERT = "synth probe: an inert edit"


def _fails(outcome: Outcome) -> set[str]:
    return {r.split(":", 1)[1] for r in outcome.reasons if r.startswith("MISSING_REQUIRED:")}


def _forbidden(outcome: Outcome) -> bool:
    return any(r.startswith("FORBIDDEN_EFFECT:") for r in outcome.reasons)


def _unmeasured(outcome: Outcome) -> bool:
    return any(r.startswith("GRADER_ERROR") for r in outcome.reasons)


def inert(text: str, comment: str) -> str:
    return text + ("" if text.endswith("\n") or not text else "\n") + f"{comment} {INERT}\n"


# Part 1 V1 (2026-10-30; part1-framework-improvements.md §2(c)): the side-effect negative, B plus an unrequested behaviour
# change. A one-token flip of a comparison or a boolean literal still compiles in every lane language and changes behaviour
# wherever the line runs. Every false positive in 2610.09142 came from such an unrequested change.
FLIP = {"==": "!=", "!=": "==", "===": "!==", "!==": "===", "True": "False", "False": "True", "true": "false", "false": "true"}
_FLIP_RX = re.compile(r"===|!==|==|!=|\b(?:true|false)\b")
_STRING_RX = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`[^`\n]*`')
SIDE_EFFECT_MARGIN = 3  # lines kept clear of B's changed regions: the flip lands in a neighbour, not in the fix


def flip_sites(text: str, lang: str, skip: Collection[int] = ()) -> list[tuple[int, int, str]]:
    """``(line, column, token)`` of every comparison or boolean literal outside comments, strings and the lines in ``skip``
    (0-based lines split at ``\n``, as the tokenizer splits them). Python is tokenized; the C-like languages are read per
    line with strings masked and ``//`` comments cut."""
    out: list[tuple[int, int, str]] = []
    if lang == "python":
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return []
        for t in toks:
            if ((t.type == tokenize.OP and t.string in ("==", "!=")) or (t.type == tokenize.NAME and t.string in ("True", "False"))) \
                    and t.start[0] - 1 not in skip:
                out.append((t.start[0] - 1, t.start[1], t.string))
        return out
    for i, ln in enumerate(io.StringIO(text).readlines()):
        if i in skip or _COMMENT_LINE.match(ln):
            continue
        code = _STRING_RX.sub(lambda m: " " * len(m.group(0)), ln)
        code = code[: code.find("//")] if "//" in code else code
        out += [(i, m.start(), m.group(0)) for m in _FLIP_RX.finditer(code)]
    return out


def flip(text: str, line: int, col: int, token: str) -> str | None:
    """``text`` with ``token`` at ``(line, col)`` flipped (``FLIP``); None when the site does not hold that token."""
    lines = io.StringIO(text).readlines()
    if line >= len(lines) or lines[line][col:col + len(token)] != token:
        return None
    lines[line] = lines[line][:col] + FLIP[token] + lines[line][col + len(token):]
    return "".join(lines)


def changed_lines(old: str, new: str, margin: int = SIDE_EFFECT_MARGIN) -> set[int]:
    """The lines of ``new`` (0-based) within ``margin`` of a region it changed from ``old``."""
    return {j for _, _, j1, j2 in diff_hunks(old, new) for j in range(max(0, j1 - margin), j2 + margin)}


class _PatchClass(BaseFalsifier):
    def __init__(self, env: PatchEnv):
        self.env = env

    def start_text(self, path: str) -> str | None:
        s = self.env.spec.start_files
        return s[path] if path in s else self.env.base_text(path)

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        if probe.program is None:
            return Outcome(data=dict(probe.data))
        out = await super().execute(probe, task, ctx)
        if _unmeasured(out):
            return out.model_copy(update={"valid": False, "error": "unmeasured"})
        return out


def _rng(task: Task, name: str) -> random.Random:
    return random.Random(f"{name}/{task.id}/{task.version}")


class PatchMUT(_PatchClass):
    """MUT over patch hunks (module docstring)."""

    name: ClassVar[str] = "MUT"

    def __init__(self, env: PatchEnv, *, max_hunks: int = MAX_HUNKS, extra_source: int = 2, side_effect: int = 2):
        super().__init__(env)
        self.max_hunks, self.extra_source, self.side_effect = max_hunks, extra_source, side_effect

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        env, spec = self.env, self.env.spec
        rng = _rng(task, "PMUT")
        comment = COMMENT.get(spec.lang, spec.comment)
        oracle = dict(spec.oracle_files)
        out = [Probe(cls="MUT", op="Identity", site="-", program=task.oracle, expect="pass")]
        drops: list[Probe] = []
        hunks: list[Probe] = []
        skipped = 0
        exts = SOURCE_EXT.get(spec.lang, ())
        self.skipped_noncode = sorted(p for p in oracle if exts and not p.endswith(exts))
        for p, t in sorted(oracle.items()):
            if exts and not p.endswith(exts):
                continue  # a spec, doc or data file B also changed: not code the hidden tests could need (recorded, not probed)
            old = self.start_text(p)
            rest = {q: v for q, v in oracle.items() if q != p}
            if t is None or old is None:
                drops.append(Probe(cls="MUT", op="DropFile", site=p, program=env.program(rest), data={"file": p}))
                continue
            a, b = old.splitlines(keepends=True), t.splitlines(keepends=True)
            code = [k for k, (i1, i2, j1, j2) in enumerate(diff_hunks(old, t)) if not cosmetic(a[i1:i2], b[j1:j2], comment)]
            skipped += len(diff_hunks(old, t)) - len(code)
            if not code:
                continue
            drops.append(Probe(cls="MUT", op="DropFile", site=p, program=env.program(rest), data={"file": p}))
            if len(code) >= 2:
                for k in code:
                    hunks.append(Probe(cls="MUT", op="DropHunk", site=f"{p}#{k}", program=env.program({**oracle, p: revert_hunk(old, t, k)}),
                                       data={"file": p, "hunk": k}))
        if len(hunks) > self.max_hunks:
            hunks = sorted(rng.sample(hunks, self.max_hunks), key=lambda x: x.site)
        out += drops + hunks
        for m in spec.mutants:
            out.append(Probe(cls="MUT", op="RevertFix", site=m.milestone, program=env.program(m.files), data={"milestone": m.milestone, "mutant": m.id}))
        base = env.base_rows()
        exts = SOURCE_EXT.get(spec.lang, ())
        sources = sorted(p for p, r in base.items() if r.get("role") == "source" and p.endswith(exts) and p not in oracle and "link" not in r)
        for p in sorted(rng.sample(sources, min(self.extra_source, len(sources)))):
            t = self.start_text(p)
            if t is not None:
                out.append(Probe(cls="MUT", op="ExtraSource", site=p, program=env.program({**oracle, p: inert(t, comment)}), expect="pass", data={"file": p}))
        out += self.side_effects(oracle, sources, rng)
        by_role: dict[str, list[str]] = {}
        for p, r in base.items():
            if r.get("role") in FORBIDDEN_ROLES and p not in oracle and "link" not in r and "text" in r:
                by_role.setdefault(str(r["role"]), []).append(p)
        for role in sorted(by_role):
            p = rng.choice(sorted(by_role[role]))
            out.append(Probe(cls="MUT", op="ExtraForbidden", site=f"{role}:{p}", program=env.program({**oracle, p: inert(str(base[p]["text"]), comment)}),
                             data={"role": role, "file": p}))
        if "vcs" not in by_role:
            out.append(Probe(cls="MUT", op="ExtraForbidden", site="vcs:.git/info/synth-probe", program=env.program({**oracle, ".git/info/synth-probe": "x\n"}),
                             data={"role": "vcs", "file": ".git/info/synth-probe"}))
        for p in sorted(spec.hidden_tests)[:2]:
            t = spec.hidden_tests[p]  # the grader's own version, edited (never the same program as an ExtraForbidden on the visible copy)
            out.append(Probe(cls="MUT", op="EditTest", site=p, program=env.program({**oracle, p: inert(t, comment)}), data={"file": p}))
        ms = spec.milestones
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                shared = shared_tests(ms[i].tests, ms[j].tests)
                edge = ms[i].id in ms[j].requires or ms[j].id in ms[i].requires
                out.append(Probe(cls="MUT", op="SharedCheck", site=f"{ms[i].id}~{ms[j].id}", expect="property",
                                 data={"hit": bool(shared), "shared": shared[:10], "declared_edge": edge}))
        self.skipped_cosmetic = skipped
        return dedupe_programs(out)

    def side_effects(self, oracle: Mapping[str, str | None], sources: Sequence[str], rng: random.Random) -> list[Probe]:
        """Part 1 V1: up to ``side_effect`` probes of B plus one flip (``flip_sites``). Sites come from the source files B
        changes, away from its changed regions; only when those have none, from the untouched source files of their
        directories."""
        spec, exts = self.env.spec, SOURCE_EXT.get(self.env.spec.lang, ())
        near: list[tuple[str, str, int, int, str]] = []
        for p, t in sorted(oracle.items()):
            old = self.start_text(p)
            if t is not None and old is not None and (not exts or p.endswith(exts)):
                near += [(p, t, *s) for s in flip_sites(t, spec.lang, changed_lines(old, t))]
        if not near:
            dirs = {p.rsplit("/", 1)[0] if "/" in p else "" for p in oracle}
            for p in sources:
                if (p.rsplit("/", 1)[0] if "/" in p else "") in dirs and (t := self.start_text(p)) is not None:
                    near += [(p, t, *s) for s in flip_sites(t, spec.lang)]
        out = []
        for p, t, line, col, tok in sorted(rng.sample(near, min(self.side_effect, len(near))), key=lambda x: (x[0], x[2], x[3])):
            if (changed := flip(t, line, col, tok)) is not None:
                out.append(Probe(cls="MUT", op="SideEffect", site=f"{p}:{line + 1}", program=self.env.program({**oracle, p: changed}),
                                 data={"file": p, "line": line + 1, "from": tok, "to": FLIP[tok]}))
        return out

    def finalize(self, findings: list[Finding]) -> list[Finding]:
        return MUT.finalize(self, findings)  # type: ignore[arg-type]  # one trial per normalised effect, the most informative kept

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        op, fails = probe.op, _fails(outcome)
        if op == "SharedCheck":
            return ("confirmed", "D6") if outcome.data.get("hit") else ("clean", None)
        req = {c for c in self.env.spec.gating_checks() if c not in self.env.spec.guards}
        if op == "Identity":
            if outcome.passed:
                return "equivalent", None
            return "confirmed", "D10" if fails & req else "D6"
        if op in ("DropFile", "DropHunk"):
            return ("suspect", "D1") if outcome.passed else ("clean", None)
        if op == "RevertFix":
            if outcome.passed:
                return "confirmed", "D1"
            m = str(probe.data["milestone"])
            if self.env.spec.check_of(m) not in fails:
                return "confirmed", "D1"  # its own milestone passes without its fix
            need = dependents(self.env, m)
            others = {mm.id for mm in self.env.spec.milestones if self.env.spec.check_of(mm.id) in fails} - {m} - need
            return ("suspect", "D6") if others else ("clean", None)
        if op == "ExtraSource":
            if outcome.passed:
                return "equivalent", None
            return ("confirmed", "D6") if _forbidden(outcome) else ("suspect", "D6")
        if op in ("ExtraForbidden", "EditTest"):
            return ("confirmed", "D2") if outcome.passed else ("clean", None)
        if op == "SideEffect":
            return ("suspect", "D2") if outcome.passed else ("clean", None)
        return ("suspect", "D3") if outcome.passed else ("clean", None)


def shared_tests(a: Sequence[str], b: Sequence[str]) -> list[str]:
    """Tests that grade both milestones: the same id, or a parent test and one of its subtests (``T`` and ``T/case``)."""
    out = set(a) & set(b)
    for x in a:
        for y in b:
            if y.startswith((x + "/", x + "::")) or x.startswith((y + "/", y + "::")):
                out.add(x if len(x) < len(y) else y)
    return sorted(out)


def dependents(env: PatchEnv, mid: str) -> set[str]:
    """Milestones that require ``mid``, transitively."""
    req = {m.id: set(m.requires) for m in env.spec.milestones}
    out: set[str] = set()
    grow = True
    while grow:
        grow = False
        for k, rs in req.items():
            if k not in out and (mid in rs or rs & out):
                out.add(k)
                grow = True
    return out


SHIMS = {
    "python": ("conftest.py", ("import pytest\n\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
                              "    outcome = yield\n    outcome.get_result().outcome = \"passed\"\n")),
    "go": ("go.work", "go 1.22\n\nuse .\n"),
    "node": (".npmrc", "ignore-scripts=true\n"),
    "rust": (".cargo/config.toml", "[build]\nrustflags = []\n"),
}
RUNNER = {"python": "pytest", "go": "go", "node": "vitest", "rust": "cargo", "java": "mvn"}
NULL_DEFECT = {"noop": "D4", "claims_only": "D4", "minus_last_write": "D1", "delete_tests": "D8", "stub_tests": "D8", "test_edit": "D8",
               "plumbing_shim": "D8", "path_shim": "D8"}


class PatchNULL(_PatchClass):
    """Degenerate agents on a repository (module docstring)."""

    name: ClassVar[str] = "NULL"

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        env, spec = self.env, self.env.spec
        comment = COMMENT.get(spec.lang, spec.comment)
        out = [Probe(cls="NULL", op="noop", site="noop", program=Program())]
        if "claims_only" in spec.floors:
            out.append(Probe(cls="NULL", op="claims_only", site="claims_only", program=env.patch_program(spec.floors["claims_only"])))
        else:
            ids = [r.id for r in spec.requirements if r.milestone] or [m.id for m in spec.milestones]
            out.append(Probe(cls="NULL", op="claims_only", site="claims_only", program=env.program({"STATUS.json": json.dumps(dict.fromkeys(ids, 1.0), indent=1) + "\n"})))
        if len(task.oracle.steps) > 1:
            out.append(Probe(cls="NULL", op="minus_last_write", site="minus_last_write", program=task.oracle.model_copy(update={"steps": task.oracle.steps[:-1]}, deep=True)))
        files = sorted({t.split("::", 1)[0] for m in spec.milestones for t in m.tests} | set(spec.hidden_tests))
        present = [p for p in files if self.start_text(p) is not None]
        if present:
            out.append(Probe(cls="NULL", op="delete_tests", site="delete_tests", program=env.program(dict.fromkeys(present))))
            out.append(Probe(cls="NULL", op="stub_tests", site="stub_tests", program=env.program({p: f"{comment} {INERT}\n" for p in present})))
        if "test_edit" in spec.floors:
            out.append(Probe(cls="NULL", op="test_edit", site="test_edit", program=env.patch_program(spec.floors["test_edit"])))
        if spec.lang in SHIMS:
            p, text = SHIMS[spec.lang]
            out.append(Probe(cls="NULL", op="plumbing_shim", site=p, program=env.program({p: text})))
        runner = RUNNER.get(spec.lang, "test")
        out.append(Probe(cls="NULL", op="path_shim", site=f"bin/{runner}", program=env.program({f"bin/{runner}": "#!/bin/sh\necho PASS\nexit 0\n"})))
        return dedupe_programs(out)

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("confirmed", NULL_DEFECT.get(probe.op, "D8")) if outcome.passed else ("clean", None)


class PatchREP(_PatchClass):
    """Flake on the repository substrate: fresh grades against the first; the lane's stored reruns as prior trials."""

    name: ClassVar[str] = "REP"

    def __init__(self, env: PatchEnv, runner: ExecRunner, *, replays: int = 2, prior: Mapping[str, Any] | None = None):
        super().__init__(env)
        self.runner, self.replays, self.prior = runner, replays, dict(prior or {})

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        out = [Probe(cls="REP", op="fresh_oracle", site=f"fresh_oracle:{i}", expect="property") for i in range(self.replays)]
        out.append(Probe(cls="REP", op="fresh_start", site="fresh_start", expect="property"))
        for src in ("S10", "S3"):  # the lane's executed reruns: S10 grades B, S3 reruns A′'s existing suite
            p = self.prior.get(src) or {}
            n, bad = int(p.get("n") or 0), int(p.get("bad") or 0)
            out += [Probe(cls="REP", op=f"stored_{src}", site=f"stored_{src}:{i}", expect="property", data={"differs": ["stored"] if i < bad else [],
                                                                                                        "source": f"lane gate record {src}"})
                    for i in range(n)]
        return out

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        if probe.op.startswith("stored_"):
            return Outcome(data=dict(probe.data))
        prog = task.oracle if probe.op == "fresh_oracle" else Program()
        r = await ctx.run(task, prog)
        if r.end is None:
            return Outcome(valid=False, error="no end state")
        files = self.env.overlay_of(r.end)
        first = await asyncio.to_thread(self.runner.result, files)
        again = await asyncio.to_thread(self.runner.result, files, fresh=True)
        if not first.ok or not again.ok:
            return Outcome(valid=False, error="unmeasured")
        differs = sorted(c for c in set(first.checks) | set(again.checks) if first.checks.get(c) != again.checks.get(c))
        return Outcome(data={"differs": differs})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("confirmed", "D11") if outcome.data.get("differs") else ("clean", None)


_ISSUE = re.compile(r"(?<![\w/])#(\d{2,})\b|\(#(\d{2,})\)")
_CHANGELOG = re.compile(r"(^|/)(CHANGELOG|CHANGES|NEWS|HISTORY|RELEASE[_-]?NOTES)[^/]*$", re.IGNORECASE)
_LITERAL = re.compile(r'"((?:[^"\\\n]|\\.){4,200})"|\'((?:[^\'\\\n]|\\.){4,200})\'|`([^`\n]{4,200})`')
STOP = frozenset(["a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "when", "is", "are", "be", "not", "no", "it", "its", "at", "by", "from", "as", "this", "that", "fix", "fixes", "fixed"])


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1}


# ---------------------------------------------------------------------------------------------------------- visible test data

_IDENT = re.compile(r"(?<![\w$])[A-Za-z_][A-Za-z0-9_]{7,}(?![\w$])")
VISIBLE_ROLES = ("test", "golden", "doc")  # what the agent reads to learn expected behaviour (the changelog is world_leak's)
MIN_DATA_LINE = 16


def _structured(ident: str) -> bool:
    """An identifier with structure (snake_case or camelCase), not an English word that happens to be long."""
    return "_" in ident.strip("_") or bool(re.search(r"[a-z][A-Z]", ident))


def _norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _literals(text: str, min_len: int) -> list[str]:
    out = []
    for g in _LITERAL.finditer(text):
        lit = next(x for x in g.groups() if x is not None)
        if len(lit) >= min_len and len(re.findall(r"[A-Za-z]", lit)) >= 3:
            out.append(lit)
    return out


_COMMENT_LINE = re.compile(r"^\s*(//|#|/\*|\*|--|;;)")


def _code_lines(lines: Sequence[str]) -> list[str]:
    """Lines that are not comments (a comment that quotes code is prose: "e.g. `type Secret string`")."""
    return [x for x in lines if not _COMMENT_LINE.match(x)]


def _added_lines(old: str, new: str) -> list[str]:
    a, b = old.splitlines(), new.splitlines()
    out: list[str] = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            out += b[j1:j2]
    return out


_IMPORT_LINE = re.compile(r"^\s*(import|from|use|package|#include|require|export \* from)\b|^\s*[\w.]*\s*\"[^\"\s]+\"\s*$|^\s*\w+\s+\"[\w./-]+\"\s*$")
_DEF = re.compile(r"\b(?:func|def|class|fn|struct|enum|type|const|let|var|interface|trait|impl)\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]{5,})")


def visible_data(env: PatchEnv, start_text: Any) -> dict[str, Any]:
    """Static evidence that files the agent can read hold what the hidden checks need (D7, the strip-mode leak of the
    real-defect set), matched against the hidden tests and the fix diff. Two signals:

    - ``hidden_cases``: at least two *data lines* of one hidden test file (``MIN_DATA_LINE``+ characters with a literal;
      not an import; absent from the agent's copy of that file) held verbatim by one other readable file, where each
      line is rare (in no other readable file): a test case with its expected output kept in a fixture or a sibling;
    - ``fix_answers``: a string literal (8+ characters, with a space) that the fix adds to the source, that A′'s source
      lacks and a hidden test compares, already readable in a test, golden or doc file; or an identifier the fix
      *defines* (``func``/``def``/``class``/... X, 6+ characters, absent from A′'s source) that a hidden test calls and
      a readable file already names: the expected value or the fix's API is in the agent's tree.

    The changelog is TAINT ``world_leak``'s. Returns the evidence (empty lists: nothing found). Measured on the 12 pool
    tasks before it was trusted (DESIGN §7.7 item 20): imports, shared assertions and standard-library names were the
    false alarms of a looser first version."""
    spec = env.spec
    rows = env.base_rows()
    readable: dict[str, str] = {}
    for p, r in rows.items():
        if r.get("role") in VISIBLE_ROLES and "link" not in r and not _CHANGELOG.search(p) and p not in spec.oracle_files:
            t = start_text(p)
            if t:
                readable[p] = t
    norm_lines = {p: {_norm_line(x) for x in t.splitlines()} for p, t in readable.items()}
    src_a = "\n".join((start_text(p) or "") for p, r in rows.items() if r.get("role") == "source" and "link" not in r)
    a_idents = set(_IDENT.findall(src_a)) | set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", src_a))
    hidden_all = "\n".join(spec.hidden_tests.values())
    hidden_cases: list[dict[str, Any]] = []
    for hp, h in sorted(spec.hidden_tests.items()):
        vis = {_norm_line(x) for x in (start_text(hp) or "").splitlines()}
        data = [ln for ln in dict.fromkeys(_norm_line(x) for x in h.splitlines())
                if len(ln) >= MIN_DATA_LINE and ln not in vis and _LITERAL.search(ln) and not _IMPORT_LINE.search(ln)]
        if not data:
            continue
        freq = {ln: sum(1 for p, s in norm_lines.items() if p != hp and ln in s) for ln in data}
        for p, s in sorted(norm_lines.items()):
            if p == hp:
                continue
            got = [ln for ln in data if ln in s and freq[ln] == 1]
            if len(got) >= 2:
                hidden_cases.append({"hidden": hp, "file": p, "lines": len(got), "sample": [g[:120] for g in got[:3]]})
    lits: set[str] = set()
    defs: set[str] = set()
    for p, t in sorted(spec.oracle_files.items()):
        if t is None or rows.get(p, {}).get("role", env.roles(p)) != "source":
            continue
        added = "\n".join(_code_lines(_added_lines(start_text(p) or "", t)))
        lits |= {lit for lit in _literals(added, 8) if " " in lit.strip() and lit not in src_a and lit in hidden_all}
        defs |= {d for d in _DEF.findall(added) if d not in a_idents}
    answers: list[dict[str, Any]] = []
    for p, t0 in sorted(readable.items()):
        t = "\n".join(_code_lines(t0.splitlines()))
        got = sorted([x for x in lits if x in t] + [d for d in defs if re.search(rf"(?<![\w$]){re.escape(d)}(?![\w$])", t)])
        if got:
            answers.append({"file": p, "tokens": [g[:80] for g in got[:8]], "n": len(got), "hidden_copy": p in spec.hidden_tests})
    return {"hidden_cases": hidden_cases, "fix_answers": answers, "fix_literals": len(lits), "fix_definitions": len(defs)}


class PatchTAINT(_PatchClass):
    """Communicability on a repository (module docstring): ledger, World leaks, stale visible tests, report literals."""

    name: ClassVar[str] = "TAINT"

    def __init__(self, env: PatchEnv, runner: ExecRunner | None = None, *, visible: bool = True):
        super().__init__(env)
        self.runner, self.visible = runner, visible

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        env, spec = self.env, self.env.spec
        out: list[Probe] = []
        asked = {r.milestone for r in spec.requirements if r.milestone}
        for m in spec.milestones:
            out.append(Probe(cls="TAINT", op="ledger", site=f"ledger:{m.id}", expect="property", data={"hit": m.id not in asked, "milestone": m.id}))
        logs = {p: self.start_text(p) or "" for p, r in env.base_rows().items() if _CHANGELOG.search(p) and "link" not in r}
        for u in spec.units:
            refs = sorted({a or b for a, b in _ISSUE.findall(u.subject)})
            words = _words(re.sub(r"^\W*\w+(\([^)]*\))?:\s*", "", u.subject))
            hits: list[str] = []
            near: list[str] = []
            for p, text in sorted(logs.items()):
                for i, ln in enumerate(text.splitlines(), 1):
                    if any(re.search(rf"(?<![\w/])#{r}\b", ln) for r in refs):
                        hits.append(f"{p}:{i}")
                    elif len(words) >= 5 and len(words - _words(ln)) <= (1 if len(words) >= 6 else 0):
                        near.append(f"{p}:{i}")
            out.append(Probe(cls="TAINT", op="world_leak", site=f"world_leak:{u.sha[:12] or u.subject[:40]}", expect="property",
                             data={"hit": bool(hits or near), "lines": hits[:5], "near": near[:5], "refs": refs}))
        if self.visible and self.runner is not None:
            out.append(Probe(cls="TAINT", op="visible_contradicts", site="visible_contradicts", expect="property"))
            out.append(Probe(cls="TAINT", op="visible_check", site="visible_check", expect="property"))
        ev = visible_data(env, self.start_text)
        out.append(Probe(cls="TAINT", op="visible_data", site="visible_data", expect="property",
                         data={"hit": bool(ev["hidden_cases"] or ev["fix_answers"]), **ev}))
        corpus = "\n".join((self.start_text(p) or "") for p, r in env.base_rows().items() if "link" not in r).lower()
        hidden = "\n".join(spec.hidden_tests.values())
        ins = spec.instruction.lower()
        seen: set[str] = set()
        for p, t in sorted(spec.oracle_files.items()):
            if t is None:
                continue
            old = self.start_text(p) or ""
            a, b = old.splitlines(), t.splitlines()
            for _tag, _i1, _i2, j1, j2 in (op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"):
                for ln in b[j1:j2]:
                    for g in _LITERAL.finditer(ln):
                        lit = next(x for x in g.groups() if x is not None)
                        if lit in seen or len(re.findall(r"[A-Za-z]", lit)) < 3 or lit not in hidden:
                            continue
                        seen.add(lit)
                        hit = lit.lower() not in ins and lit.lower() not in corpus
                        out.append(Probe(cls="TAINT", op="report_literal", site=f"literal:{lit[:80]}", expect="property",
                                         data={"hit": hit, "report_only": True, "file": p}))
        return out

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        if probe.op not in ("visible_contradicts", "visible_check"):
            return Outcome(data=dict(probe.data))
        assert self.runner is not None
        r = await ctx.run(task, task.oracle)
        if r.end is None:
            return Outcome(valid=False, error="no end state")
        files = self.env.overlay_of(r.end)
        start = await asyncio.to_thread(self.runner.result, dict(self.env.spec.start_files), mode="visible")
        b1 = await asyncio.to_thread(self.runner.result, files, mode="visible")
        if not (start.ok and b1.ok):
            return Outcome(valid=False, error="unmeasured")
        ignored = set(self.env.spec.meta.get("ignored_tests") or [])
        if probe.op == "visible_check":  # the mirror of a stale test: a visible test the fix turns green shows the agent the answer
            f2p = {t for t, s in b1.statuses.items() if s == "passed" and start.statuses.get(t) in ("failed", "error") and t not in ignored}
            if f2p:  # confirmed only when a fresh run of A′ agrees (a flake is REP's)
                s2 = await asyncio.to_thread(self.runner.result, dict(self.env.spec.start_files), mode="visible", fresh=True)
                if not s2.ok:
                    return Outcome(valid=False, error="unmeasured")
                f2p = {t for t in f2p if s2.statuses.get(t) in ("failed", "error")}
            ms = {t for m in self.env.spec.milestones for t in m.tests}
            return Outcome(data={"hit": bool(f2p), "fail_to_pass": sorted(f2p)[:20], "n_f2p": len(f2p), "milestone_tests": sorted(f2p & ms)[:20]})
        bad = {t for t, s in b1.statuses.items() if s in ("failed", "error") and start.statuses.get(t) == "passed" and t not in ignored}
        if bad:  # confirmed only when a fresh run agrees (a flake is REP's)
            b2 = await asyncio.to_thread(self.runner.result, files, mode="visible", fresh=True)
            if not b2.ok:
                return Outcome(valid=False, error="unmeasured")
            bad = {t for t in bad if b2.statuses.get(t) in ("failed", "error")}
        return Outcome(data={"hit": bool(bad), "stale": sorted(bad)[:20], "n_stale": len(bad)})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        if not outcome.data.get("hit") or outcome.data.get("report_only"):
            return "clean", None
        if probe.op == "visible_data" and not outcome.data.get("fix_answers"):  # shared case lines only: whether they matter is a judgement
            return "suspect", "D7"
        return "confirmed", {"ledger": "D5", "world_leak": "D7", "visible_contradicts": "D6", "visible_check": "D7", "visible_data": "D7"}.get(probe.op, "D5")


# ---------------------------------------------------------------------------------------------------------- certificates

PATCH_POLICY = AdmissionPolicy(deterministic=["MUT", "NULL", "RED", "TAINT", "REP"], require_applicable=["MUT", "NULL", "RED", "TAINT", "REP"],
                               require_alt_pass=False)
NOT_APPLICABLE = {
    "TWIN": "a symptom report names no entity to substitute: the instruction's literals are behaviour, not keys",
    "PERT": "the start tree is one mined commit; distractor entities have no meaning for test-graded code",
    "ALT": "paid (alternative solvers): not run in M5; the lane's S8 records alternatives_pending",
    "ATK": "paid (attackers): not run in M5",
    "ONL": "online: applies to eval and training rollouts (falsify.online), none in a certificate run",
}


def patch_classes(env: PatchEnv, runner: ExecRunner, *, prior: Mapping[str, Any] | None = None, max_hunks: int = MAX_HUNKS,
                  visible: bool = True) -> list[Any]:
    return [PatchREP(env, runner, prior=prior), RED(), PatchTAINT(env, runner, visible=visible), PatchNULL(env), PatchMUT(env, max_hunks=max_hunks)]


async def prefetch(task: Task, ctx: FalsifyContext, env: PatchEnv, classes: Sequence[Any], runner: ExecRunner) -> int:
    """Run every program probe in the patch World (cheap) and queue its end state with the runner, so the Docker runs
    happen ``parallel`` at a time instead of one after another. Returns the number of distinct end states queued."""
    seen: set[str] = set()
    states = []
    for f in classes:
        if not isinstance(f, PatchMUT | PatchNULL):
            continue
        for p in await f.propose(task, ctx):
            if p.program is None:
                continue
            r = await ctx.run(task, p.program)
            if r.end is None:
                continue
            files = env.overlay_of(r.end)
            k = files_key(files)
            if k not in seen:
                seen.add(k)
                states.append(files)
    runner.prefetch(states)
    return len(states)


async def certify_patch(env: PatchEnv, runner: ExecRunner, *, prior: Mapping[str, Any] | None = None, policy: AdmissionPolicy | None = None,
                        max_hunks: int = MAX_HUNKS, visible: bool = True) -> tuple[Certificate, list[ClassResult]]:
    """Every applicable class on one patch task, in ``run_classes`` order; the others recorded as not applicable."""
    task = env.task()
    ctx = env.context(runner)
    classes = patch_classes(env, runner, prior=prior, max_hunks=max_hunks, visible=visible)
    await prefetch(task, ctx, env, classes, runner)
    results = await run_classes(task, classes, ctx)
    results += [ClassResult(cls=k, applicable=False, reason=why) for k, why in NOT_APPLICABLE.items()]
    return certify(task, results, policy or PATCH_POLICY), results


def findings_summary(results: Sequence[ClassResult]) -> dict[str, Any]:
    """Confirmed and suspect findings with their op, site, defect and reason codes; TAINT's report-only hits."""
    out: dict[str, Any] = {"confirmed": [], "suspects": [], "report_only": [], "invalid": 0}
    for r in results:
        for f in r.findings:
            if not f.outcome.valid:
                out["invalid"] += 1 if not f.outcome.data.get("duplicate_of") else 0
                continue
            row = {"cls": f.cls, "op": f.op, "site": f.site, "defect": f.defect, "reasons": f.outcome.reasons[:6],
                   **{k: v for k, v in f.outcome.data.items() if k in ("shared", "stale", "lines", "declared_edge", "milestone", "role", "file")}}
            if f.label == "confirmed":
                out["confirmed"].append(row)
            elif f.label == "suspect":
                out["suspects"].append(row)
            elif f.outcome.data.get("report_only") and f.outcome.data.get("hit"):
                out["report_only"].append(row)
    return out


def applicability(results: Sequence[ClassResult]) -> dict[str, str]:
    return {r.cls: ("applies" if r.applicable else f"n/a: {r.reason}") for r in results}


__all__ = ["ALLOWED_ROLES", "NOT_APPLICABLE", "PATCH_POLICY", "PatchMUT", "PatchNULL", "PatchREP", "PatchTAINT", "applicability", "certify_patch",
           "dependents", "findings_summary", "patch_classes", "prefetch", "shared_tests"]
