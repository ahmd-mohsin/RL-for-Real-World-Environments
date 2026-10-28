"""Benchmark contamination audit (swarm-pipeline/00 §9; 2026-10-24): is each held-out task still sealed?

A held-out task can leak in two ways, and the audit measures both, executed, never assumed:

1. **Public exposure** in our own repository. Its tracked tree (any ref, normally ``origin/main``) is grepped for
   fixed-string probes cut from the task: its id, sentences of its bug report, its hidden test names, and the lines its
   real fix and its hidden tests add. ``git grep -F`` matches them exactly. The first commit that added each file holding a
   hit is recorded.
2. **Training overlap.** ``bench.contamination`` scans every training source (training windows' prompts and fix lines,
   training bug reports, swarm variants) against the held-out prompts and fix lines: exact 13-grams, code substrings, near
   duplicates and containment, with template boilerplate subtracted.

The upstream fix and its tests were public on GitHub from the fix's own date, by construction (they are real fix
commits). The cutoff rule (fixes after the student's cutoff) is what keeps them unseen by the frozen student, so it is
recorded per task (``fix_dates``, ``cutoff``) and judged apart from our own repository's exposure.

**Verdict** (``verdict``), reported and never applied by itself (the lead decides):

- ``contaminated``: a training-overlap hit, or our repository holds the task's grading content: at least
  ``MIN_LINES`` fix or hidden-test lines, any hidden test name tied to the task, or at least ``MIN_PROMPT`` report
  sentences. Every reason is listed.
- ``sealed``: no hit of any kind beyond the task id (an id in a results table identifies the task; it holds no content).
- ``unknown``: the task's content could not be loaded (a co-author's variant whose window is on another machine is
  audited through its parent window and its committed report, and says so in ``basis``).

Nothing here writes a probe's text out: the report holds counts, paths and commit hashes only, so publishing the report
cannot re-leak what it measured.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.bench import contamination

MIN_LINES = 3  # fix or hidden-test lines in the public tree that make a task contaminated
MIN_PROMPT = 2  # report sentences in the public tree that make a task contaminated
MIN_GRAMS = 5  # shared 13-grams with one training source that count (fewer is shared phrasing)
SPREAD = 3  # 13-gram hits spread over this many training sources of one task read as a family's template
LINE_CHARS = 40  # a code line shorter than this is too common to be evidence
SENT_CHARS = 60
TEST_CHARS = 16
CAP = 60  # probes per kind (the longest first)
TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|testdata|spec)(/|$)|(^|/)(test_[^/]*|[^/]*_test\.[a-z]+|[^/]*\.(test|spec)\.[a-z]+)$")


@dataclass
class HeldTask:
    id: str
    repo: str
    kind: str  # debug | variant
    prompt: str = ""
    hidden_tests: list[str] = field(default_factory=list)
    fix_lines: list[str] = field(default_factory=list)
    test_lines: list[str] = field(default_factory=list)
    fix_dates: list[str] = field(default_factory=list)
    cutoff: str | None = None
    basis: str = "own window"
    aliases: list[str] = field(default_factory=list)  # a variant's parent id: files naming it name the variant's tests too
    loaded: bool = True


# ---- loading ----------------------------------------------------------------------------------------------------------
def _added(old: str, new: str) -> list[str]:
    return [ln[1:] for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
            if ln.startswith("+") and not ln.startswith("+++") and ln[1:].strip()]


def _prompt_of(p: Path) -> str:
    if not p.is_file():
        return ""
    d = json.loads(p.read_text())
    reqs = d.get("requirements") or []
    texts = [r.get("text", "") if isinstance(r, Mapping) else str(r) for r in (reqs.values() if isinstance(reqs, Mapping) else reqs)]
    return "\n".join([str(d.get("opening") or "")] + texts + [str(d.get("closing") or "")]).strip()


def load_debug(store: Path, prompts: Sequence[Path] = ()) -> list[HeldTask]:
    """Held-out debug windows (``<store>/tasks/<id>/{window.json, base/}``) with their reports from ``prompts`` dirs."""
    out = []
    for wdir in sorted(Path(store).glob("tasks/*")):
        wf = wdir / "window.json"
        if not wf.is_file():
            continue
        w = json.loads(wf.read_text())
        base = wdir / "base"

        def old(p: str, base: Path = base) -> str:
            f = base / p
            return f.read_text(errors="replace") if f.is_file() else ""

        fix = [ln for p, new in sorted((w.get("oracle_files") or {}).items()) if isinstance(new, str) and not TEST_PATH.search(p) for ln in _added(old(p), new)]
        tst = [ln for p, new in sorted((w.get("tests_b") or {}).items()) if isinstance(new, str) for ln in _added(old(p), new)]
        tests = [t for m in w.get("milestones") or [] for t in m.get("tests") or []]
        rb = w.get("rebreak") or {}
        prompt = next((s for s in (_prompt_of(Path(d) / f"{w['id']}.json") for d in prompts) if s), "")
        out.append(HeldTask(w["id"], str(w.get("repo")), "debug", prompt, tests, fix, tst, sorted(map(str, (rb.get("fix_dates") or {}).values()))
                            if isinstance(rb.get("fix_dates"), Mapping) else [str(x) for x in rb.get("fix_dates") or []], rb.get("cutoff")))
    return out


def variants(parents: Sequence[HeldTask], reports: Path, ids: Iterable[str]) -> list[HeldTask]:
    """A co-author's benchmark variant (``<parent>-<tag>``): its committed report, its parent's hidden content (the
    variant's own window lives on the author's machine; a variant applies some of the parent's fixes, so its hidden
    content is a subset of the parent's)."""
    by = {p.id: p for p in parents}
    out = []
    for vid in ids:
        pid = next((p for p in sorted(by, key=len, reverse=True) if vid.startswith(p + "-") or vid == p), None)
        prompt = _prompt_of(Path(reports) / f"{vid}.json")
        if pid is None:
            out.append(HeldTask(vid, "?", "variant", prompt, basis="no parent window here", loaded=False))
            continue
        p = by[pid]
        out.append(HeldTask(vid, p.repo, "variant", prompt or p.prompt, list(p.hidden_tests), list(p.fix_lines), list(p.test_lines), list(p.fix_dates),
                            p.cutoff, basis=f"parent {pid} window; {'own committed report' if prompt else 'parent report'}", aliases=[pid]))
    return out


# ---- probes -----------------------------------------------------------------------------------------------------------
def _distinct(lines: Iterable[str], n: int) -> list[str]:
    seen, out = set(), []
    for ln in sorted({x.strip() for x in lines}, key=len, reverse=True):
        if len(ln) >= n and sum(ch.isalnum() for ch in ln) >= n // 2 and ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out[:CAP]


def _sentences(prompt: str, template: frozenset[str] = frozenset()) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", prompt)
    return [x for x in _distinct((re.sub(r"^\s*-?\s*\[R\d+\]\s*", "", p) for p in parts), SENT_CHARS) if x not in template]


def template_sentences(prompts: Iterable[str], min_df: int = 3) -> frozenset[str]:
    """Sentences in at least ``min_df`` different reports: the harness note and the lanes' shared phrasing, never one
    task's content (a sentence only one or two tasks state is evidence)."""
    df: dict[str, int] = {}
    for t in prompts:
        for x in set(_sentences(t)):
            df[x] = df.get(x, 0) + 1
    return frozenset(x for x, k in df.items() if k >= min_df)


def _leaf(test: str) -> str:
    return re.split(r"::|/|\.", test)[-1] if "::" in test or "/" in test else test


def probes(t: HeldTask, template: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """Fixed strings per evidence kind, each one line stripped at its ends (``git grep`` matches within a line); report
    sentences in ``template`` are boilerplate, not evidence."""
    return {"id": [t.id, *t.aliases], "prompt": _sentences(t.prompt, template), "hidden_test": _distinct((_leaf(x) for x in t.hidden_tests), TEST_CHARS),
            "fix_line": _distinct(t.fix_lines, LINE_CHARS), "test_line": _distinct(t.test_lines, LINE_CHARS)}


# ---- the public tree --------------------------------------------------------------------------------------------------
Grep = Callable[[Sequence[str]], dict[str, list[str]]]


def git_grep(repo: Path, ref: str = "HEAD", *, exclude: Sequence[str] = ()) -> Grep:
    """``probe -> [paths]`` over ``ref``'s tracked tree, exact fixed strings (``-F``: no escaping), binary files skipped."""
    ex = [f":(exclude){x}" for x in exclude]

    def run(pats: Sequence[str]) -> dict[str, list[str]]:
        hits: dict[str, set[str]] = {}
        pats = [p for p in dict.fromkeys(pats) if p and "\n" not in p]
        for i in range(0, len(pats), 200):
            chunk = pats[i:i + 200]
            args = ["git", "-C", str(repo), "grep", "-F", "-I", "-o", "-z"]
            for p in chunk:
                args += ["-e", p]
            r = subprocess.run(args + [ref, "--", ".", *ex], capture_output=True, text=True, check=False)
            if r.returncode not in (0, 1):
                raise RuntimeError(r.stderr.strip()[:300])
            for rec in r.stdout.split("\n"):
                if not rec:
                    continue
                f = rec.split("\0")
                if len(f) >= 2:
                    path = f[0].split(":", 1)[1] if f[0].startswith(ref + ":") else f[0]
                    hits.setdefault(f[-1], set()).add(path)
        return {p: sorted(v) for p, v in hits.items()}

    return run


def first_added(repo: Path, paths: Iterable[str]) -> dict[str, str]:
    """The commit (short sha and date) that added each path."""
    out = {}
    for p in sorted(set(paths))[:20]:
        r = subprocess.run(["git", "-C", str(repo), "log", "--all", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", p], capture_output=True, text=True, check=False)
        lines = r.stdout.strip().splitlines()
        if lines:
            out[p] = lines[-1]
    return out


def exposure(t: HeldTask, grep: Grep, template: frozenset[str] = frozenset()) -> dict[str, Any]:
    pr = probes(t, template)
    hits = grep([x for v in pr.values() for x in v])
    out: dict[str, Any] = {}
    for kind, ps in pr.items():
        got = {p: hits[p] for p in ps if p in hits}
        out[kind] = {"probes": len(ps), "matched": len(got), "files": sorted({f for v in got.values() for f in v})[:12]}
    return out


# ---- training overlap -------------------------------------------------------------------------------------------------
def train_overlap(held: Sequence[HeldTask], corpus: Iterable[tuple[str, str]], template: Iterable[str] = ()) -> dict[str, list[dict[str, Any]]]:
    """``bench.contamination`` over every training source: hits per held-out id (kind, source, overlap); ``template``
    sentences are subtracted as boilerplate."""
    bench = contamination.fingerprints([(t.id, t.prompt, "\n".join(t.fix_lines)) for t in held if t.loaded and (t.prompt or t.fix_lines)],
                                       extra_template=list(template))
    out: dict[str, list[dict[str, Any]]] = {}
    for h in contamination.scan(bench, list(corpus)):
        out.setdefault(str(h.get("task")), []).append({k: h.get(k) for k in ("kind", "source", "overlap") if k in h})
    return out


def training_corpus(train_store: Path | None = None, prompt_dirs: Sequence[Path] = (),
                    skip: Callable[[str], bool] = lambda _id: False) -> list[tuple[str, str]]:
    """(source id, text) for every training source on this machine: training windows' fix lines and training reports
    (``prompt_dirs``; a held-out id there is skipped by ``skip``)."""
    docs: list[tuple[str, str]] = []
    for t in (load_debug(train_store) if train_store and Path(train_store).is_dir() else []):
        docs.append((f"{t.id}:diff", "\n".join(t.fix_lines)))
    for d in prompt_dirs:
        for p in sorted(Path(d).glob("*.json")):
            if not skip(p.stem):
                docs.append((f"{p.stem}:prompt", _prompt_of(p)))
    return [(s, x) for s, x in docs if x.strip()]


# ---- verdict ----------------------------------------------------------------------------------------------------------
def split_hits(hits: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """(strong, weak): code substrings, near duplicates and containment are strong; a 13-gram hit is weak below
    ``MIN_GRAMS`` shared n-grams, or when the task's 13-gram hits spread over ``SPREAD`` or more sources (a lane's shared
    phrasing that the template did not catch)."""
    grams = [h for h in hits if h.get("kind") == "13-gram"]
    spread = len({h.get("source") for h in grams}) >= SPREAD
    strong = [h for h in hits if h.get("kind") != "13-gram" or (not spread and int(h.get("overlap") or 0) >= MIN_GRAMS)]
    return strong, [h for h in hits if h not in strong]


def verdict(t: HeldTask, exp: Mapping[str, Any], train_hits: Sequence[Mapping[str, Any]], *, held: bool = True) -> tuple[str, list[str]]:
    if not held:
        return "not_heldout", [f"{t.repo} left the held-out draw: a training source now (its stale held-out windows are not a benchmark)"]
    if not t.loaded:
        return "unknown", [f"content not loadable ({t.basis})"]
    why = []
    strong, _weak = split_hits(train_hits)
    if strong:
        srcs = sorted({str(h.get("source")) for h in strong})[:4]
        why.append(f"{len(strong)} training-overlap hits ({sorted({str(h.get('kind')) for h in strong})}; sources {srcs})")
    lines = exp["fix_line"]["matched"] + exp["test_line"]["matched"]
    if lines >= MIN_LINES:
        why.append(f"{exp['fix_line']['matched']} fix lines and {exp['test_line']['matched']} hidden-test lines in the public tree")
    if exp["hidden_test"]["matched"] and exp["id"]["matched"]:
        shared = set(exp["hidden_test"]["files"]) & set(exp["id"]["files"])
        if shared:
            why.append(f"{exp['hidden_test']['matched']} hidden test names in public files that also name the task ({len(shared)} files)")
    if exp["prompt"]["matched"] >= MIN_PROMPT:
        why.append(f"{exp['prompt']['matched']} of {exp['prompt']['probes']} report sentences in the public tree")
    return ("contaminated", why) if why else ("sealed", [])


def audit(held: Sequence[HeldTask], grep: Grep, corpus: Iterable[tuple[str, str]], *, repo: Path | None = None, ref: str = "HEAD",
          in_draw: Callable[[HeldTask], bool] = lambda _t: True, min_df: int = 3) -> dict[str, Any]:
    corpus = list(corpus)
    template = template_sentences([t.prompt for t in held] + [x for s, x in corpus if s.endswith(":prompt")], min_df)
    over = train_overlap(held, corpus, template)
    rows = []
    for t in held:
        exp = exposure(t, grep, template) if t.loaded else {}
        v, why = verdict(t, exp, over.get(t.id, []), held=in_draw(t))
        files = {f for k in ("prompt", "hidden_test", "fix_line", "test_line") for f in (exp.get(k) or {}).get("files", [])}
        strong, weak = split_hits(over.get(t.id, []))
        rows.append({"id": t.id, "repo": t.repo, "kind": t.kind, "basis": t.basis, "verdict": v, "why": why, "exposure": exp,
                     "train_overlap": strong[:10], "train_overlap_weak": weak[:10], "first_public": first_added(repo, files) if repo and files else {},
                     "fix_dates": t.fix_dates, "cutoff": t.cutoff})
    count: dict[str, int] = {}
    for r in rows:
        count[r["verdict"]] = count.get(r["verdict"], 0) + 1
    return {"ref": ref, "rules": {"min_lines": MIN_LINES, "min_prompt": MIN_PROMPT, "line_chars": LINE_CHARS, "sentence_chars": SENT_CHARS,
                                  "test_chars": TEST_CHARS, "template_min_df": min_df, "template_sentences": len(template)}, "counts": count, "tasks": rows}
