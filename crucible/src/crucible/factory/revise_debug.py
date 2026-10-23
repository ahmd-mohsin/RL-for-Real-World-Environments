"""The swarm's reviser and gates for the debug lane (swarm-pipeline/12 §3, 13 §5): a typed knob action in, a stored,
gated variant task out. No Claude in any role: the report of every variant is written by the swarm's ``prompt_author``
(``factory.swarm_prompts.write_prompt``) under a knob-specific instruction; the World and the oracle come only from the
window's real fixes; executed gates decide (V6).

**Variants are ordinary windows.** A variant ``<parent>-<knob>-<key>`` is saved into the parent's store (``window.json``,
``record.json``, ``base/``) with ``revision = {parent, knob, applied, pre_applied, level}``, so the lane's own commands
(``gates``, ``author``, ``repro``, ``intent``) and the targeting phase treat it like any other candidate. Its ``applied``
list (the parent's plus this knob) is what the controller's H2 takes back later.

| knob | World / milestones | report |
|---|---|---|
| ``C1`` checkpoint start | the real fixes of the chosen level's milestones pre-applied (``difficulty.checkpoint_start``); their tests guard regressions | rewritten for the milestones left |
| ``K1`` drop a fault | the hardest non-interacting milestone's fix pre-applied (``simplify_units``, kind ``drop``) | rewritten for the milestones left |
| ``K2`` ``K3`` ``K4`` ``K6`` | unchanged (``base/`` hard-linked from the parent) | rewritten under the knob's instruction (K4 adds the ``repro`` field: gate S7r) |

**Gates** (``DebugGates.run``): the lane's S7 lint and ledger on the new report (``author_debug``); for a World knob the
deterministic gates S6/S5/S8/S3/S10 on the variant (``Debug.gates``) plus the start check (``difficulty.check_start``:
pre-applied milestones' tests pass at the new start, the rest fail); for a prompt knob the parent's executed gate
verdicts are inherited only when the start tree and every hidden test are byte-identical (checked by hash, recorded);
for K4 the executed S7r (``debug_repro.s7r``) must not fail. A held-out window is refused before anything is written (V9).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.difficulty import HeldOutRefused, check_start, checkpoint_levels
from crucible.factory.difficulty_bridge import k1_target
from crucible.factory.difficulty_hooks import PROMPT_KNOBS, WORLD_KNOBS, GateResult, KnobAction, Revision

SUPPORTED = frozenset({"C1", "K1", "K2", "K3", "K4", "K6"})  # K5 (order): the debug lane renders requirements in milestone order, so a
# reordered report changes nothing the agent sees; H1-H3 need a new real fault from the pool (the generator's job), not a revision

# Knobs that change the report's frame, never a requirement's facts: the swarm writes the opening and closing (K4: also
# the repro field) and every kept requirement stays word for word as the parent's measured report had it. A free rewrite
# lost facts the hidden tests need (pilot 2026-10-23: go-toml's C1 variant dropped "date and time values" from m03, and
# m03 went from 6/8 to 0/8 although its World and tests were unchanged).
VERBATIM_KNOBS = frozenset({"C1", "K1", "K2", "K4"})
VERBATIM = " Every requirement in `current_report` stays word for word: copy it unchanged."
KNOB_INSTRUCTIONS: dict[str, str] = {
    "C1": "Some of the problems in the current report are already fixed in this version of the repository (they are no longer in the hand-off). "
          "Write the opening and closing for the problems that are left, as the same user." + VERBATIM,
    "K1": "One of the problems in the current report is already fixed in this version of the repository (it is no longer in the hand-off). "
          "Write the opening and closing for the problems that are left, as the same user." + VERBATIM,
    "K2": "Announce the structure: the opening must say how many separate problems you have seen ({n}) and that fixing one of them may change "
          "how another one shows up." + VERBATIM,
    "K3": "Point to the area: each requirement must name the user-visible feature or area of the library it concerns (for example 'the encoder', "
          "'the here-document parser', 'the CLI flag handling'), never a file, a function, a test or the fix.",
    "K4": "Add a reproduction per requirement: a `repro` field mapping each requirement id to {{\"command\": \"...\", \"expect\": {{\"exit\": 0, "
          "\"contains\": [\"...\"]}}, \"timeout_s\": 120}}. The command runs from the repository root, offline, and may write scratch files only under "
          "/tmp/ (a heredoc inside the command); it must show the bug in the repository as it is now and pass once the bug is fixed. `expect` "
          "describes what the CORRECT program prints. Run every command in your sandbox before you finish. Never copy or paraphrase a hidden test."
          + VERBATIM,
    "K6": "Sharpen requirement {milestone}: give a concrete input, the output you observe and the output you expect, observed from the outside. "
          "Every other requirement stays word for word as in `current_report`.",
}


def variant_id(parent: str, knob: str, detail: Sequence[str] = ()) -> str:
    key = hashlib.sha256("|".join((parent, knob, *sorted(detail))).encode()).hexdigest()[:6]
    return f"{parent}-{knob.lower()}-{key}"


def restore_requirements(path: Path, parent_reqs: Mapping[str, str], action: KnobAction) -> list[str]:
    """Keep the knob's scope: the requirements the knob does not target go back to the parent's text (``VERBATIM_KNOBS``:
    all of them; K6: all but its milestone). The writer's own versions stay in the file under ``swarm_requirements`` for
    diagnosis. Returns the milestone ids restored (those whose text differed)."""
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    keep_own = set() if action.knob in VERBATIM_KNOBS else ({action.milestone} if action.knob == "K6" else set(parent_reqs))
    reqs = dict(d.get("requirements") or {})
    changed = [m for m, t in parent_reqs.items() if m not in keep_own and m in reqs and " ".join(str(reqs[m]).split()) != " ".join(str(t).split())]
    if changed:
        d["swarm_requirements"] = {m: reqs[m] for m in changed}
        for m in changed:
            reqs[m] = parent_reqs[m]
        d["requirements"] = reqs
        d["restored_verbatim"] = changed
        path.write_text(json.dumps(d, indent=1))
    return changed


def tree_hash(base: Path) -> str:
    """sha256 over the start tree's relative paths and bytes (``.git`` excluded)."""
    h = hashlib.sha256()
    for p in sorted(x for x in Path(base).rglob("*") if x.is_file() and ".git" not in x.relative_to(base).parts):
        h.update(str(p.relative_to(base)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def hidden_hash(rec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps({"tests_b": rec.get("tests_b"), "milestones": [(m["id"], sorted(m["tests"])) for m in rec.get("milestones") or []],
                                      "p2p": sorted(rec.get("p2p") or [])}, sort_keys=True).encode()).hexdigest()


def _link_copy(src: Path, dst: Path) -> None:
    def link(s: str, d: str) -> None:
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

    shutil.copytree(src, dst, symlinks=True, copy_function=link)


@dataclass
class DebugReviser:
    """``difficulty_hooks.Reviser`` for the debug lane. Injected (the lane script owns them): ``task_of(pid)`` the parent
    as a ``DebugTask`` under its authored prompt; ``store`` the parent's ``WindowStore``; ``mined_of(rec, base)`` in the
    lane's language scope; ``write_report(vid, rec, revision_brief) -> {status, usd, ...}`` the swarm's prompt machine
    writing ``prompts/<vid>.json``; ``prompts`` that directory."""

    store: Any
    prompts: Path
    task_of: Callable[[str], Any]
    mined_of: Callable[[Mapping[str, Any], Path], Any]
    write_report: Callable[[str, Mapping[str, Any], Mapping[str, Any]], Awaitable[Mapping[str, Any]]]
    scratch: Path
    spent: float = 0.0
    log: list[dict[str, Any]] = field(default_factory=list)

    async def revise(self, task: Any, action: KnobAction) -> Revision:
        pid = str(getattr(task, "task_id", task))
        if action.knob not in SUPPORTED:
            raise ValueError(f"the debug reviser does not apply {action.knob}")
        prec = self.store.window(pid)
        if str(getattr(self.store, "split", "train")) != "train":
            raise HeldOutRefused(f"{pid} is held out: no revision (V9)")
        applied = [*((prec.get("revision") or {}).get("applied") or []), action.knob]
        pre: list[str] = []
        level = None
        if action.knob == "C1":
            cp = action.params.get("checkpoint") or {}
            pre, level = list(cp.get("pre") or []), cp.get("level")
            if not pre:
                levels = checkpoint_levels(prec["milestones"])
                pre, level = (levels[0], 1) if levels else ([], None)
        elif action.knob == "K1":
            m = action.milestone or k1_target(prec["milestones"], (action.evidence or {}).get("per_milestone"))
            pre = [m] if m else []
        if action.knob in WORLD_KNOBS and not pre:
            raise ValueError(f"{pid}: {action.knob} found nothing to pre-apply")
        vid = variant_id(pid, action.knob, pre or [action.milestone or ""])
        vdir = Path(self.store.dir(vid))
        if vdir.exists():
            shutil.rmtree(vdir)
        if action.knob in WORLD_KNOBS:
            parent = self.task_of(pid)
            by_id = {m["id"]: m for m in parent.milestones}
            units = {str(by_id[m]["unit"]) for m in pre}
            v = parent.simplify_units(units, kind="checkpoint" if action.knob == "C1" else "drop", root=self.scratch)
            rec = json.loads(json.dumps(v.window, default=str))
            src_base = Path(self.scratch) / str(v.window["id"]) / "base"
            vdir.mkdir(parents=True)
            shutil.copytree(src_base, vdir / "base", symlinks=True)
        else:
            rec = json.loads(json.dumps(prec, default=str))
            vdir.mkdir(parents=True)
            _link_copy(Path(self.store.dir(pid)) / "base", vdir / "base")
        rec["id"] = vid
        rec["stage"] = prec.get("stage")
        rec["revision"] = {"parent": pid, "knob": action.knob, "direction": action.direction, "applied": applied, "pre_applied": sorted(pre),
                           "level": level, "reason": action.reason, "evidence": dict(action.evidence)}
        self.store.save_window(rec)
        self.store.save(self.mined_of(rec, vdir / "base"))
        old = json.loads((self.prompts / f"{pid}.json").read_text()) if (self.prompts / f"{pid}.json").exists() else {}
        keep = {m["id"] for m in rec["milestones"]}
        n = len(rec["milestones"])
        brief = {"knob": action.knob, "instruction": KNOB_INSTRUCTIONS[action.knob].format(n=n, milestone=action.milestone or ""),
                 "current_report": {"opening": old.get("opening", ""), "requirements": {k: v for k, v in (old.get("requirements") or {}).items() if k in keep},
                                    "closing": old.get("closing", "")}}
        if action.knob == "K4":
            brief["format"] = {"opening": "...", "requirements": {m: "..." for m in sorted(keep)}, "closing": "...",
                               "repro": {m: {"command": "...", "expect": {"exit": 0, "contains": ["..."]}, "timeout_s": 120} for m in sorted(keep)}}
        res = dict(await self.write_report(vid, rec, brief))
        self.spent += float(res.get("usd") or 0.0)
        restored = restore_requirements(self.prompts / f"{vid}.json", brief["current_report"]["requirements"], action)
        row = {"variant": vid, "parent": pid, "knob": action.knob, "pre_applied": sorted(pre), "level": level, "report": res.get("status"),
               "report_reason": str(res.get("reason") or "")[:500], "usd": res.get("usd"), "restored_verbatim": restored}
        self.log.append(row)
        return Revision(vid, pid, action, task=rec, notes=json.dumps(row, default=str))


@dataclass
class DebugGates:
    """``difficulty_hooks.Gates`` for the debug lane: see the module docstring. Injected: ``author(vid) -> authored``
    (writes ``authored/<vid>.json``), ``world_gates(vid) -> {ok, passed, gates}`` (``Debug.gates``), ``start_status(vid,
    pre) -> {milestone: bool|None}`` (the ORIGINAL milestones graded at the variant's start), ``repro(vid) -> {ok, status}``
    (S7r), ``gates_dir`` (``gates/<id>.json``)."""

    store: Any
    gates_dir: Path
    prompts: Path
    author: Callable[[str], Mapping[str, Any]]
    world_gates: Callable[[str], Awaitable[Mapping[str, Any]]]
    start_status: Callable[[str, Sequence[str]], Awaitable[Mapping[str, bool | None]]]
    repro: Callable[[str], Awaitable[Mapping[str, Any]]]

    async def run(self, revision: Revision) -> GateResult:
        vid, pid, knob = revision.task_id, revision.parent_id, revision.action.knob
        rec = self.store.window(vid)
        found: list[str] = []
        gates: dict[str, Any] = {}
        if not (self.prompts / f"{vid}.json").exists():
            why = str((json.loads(revision.notes) if revision.notes.startswith("{") else {}).get("report_reason") or "")
            infra = any(k in why for k in ("credits_exhausted", "TinkerCapReached", "budget:", "masked", "infra"))
            return GateResult(False, {"S7": {"ok": False, "why": f"the swarm wrote no report ({why[:200]})"}, "infra": infra},
                              [f"S7: no report written{' (infrastructure: ' + why[:120] + ')' if infra else ''}"])
        authored = self.author(vid)
        gates["S7"] = {"ok": not authored.get("findings"), "findings": list(authored.get("findings") or [])[:10]}
        found += [f"S7: {f}" for f in authored.get("findings") or []][:10]
        if knob in WORLD_KNOBS:
            pre = (rec.get("revision") or {}).get("pre_applied") or []
            parent = self.store.window(pid)
            st = await self.start_status(vid, pre)
            probs = check_start(parent["milestones"], pre, st)
            gates["start"] = {"ok": not probs, "problems": probs, "status": dict(st)}
            found += [f"start: {p}" for p in probs]
            prev = self.gates_dir / f"{vid}.json"
            th, hh = tree_hash(Path(self.store.dir(vid)) / "base"), hidden_hash(rec)
            reuse = prev.exists() and json.loads(prev.read_text()).get("ok") and json.loads(prev.read_text()).get("tree_sha256") == th \
                and json.loads(prev.read_text()).get("hidden_sha256") == hh
            if not probs and not found and reuse:  # the same start tree and hidden tests were gated already (sha256)
                gates["world"] = {"ok": True, "reused": True, "passed": json.loads(prev.read_text()).get("passed")}
            elif not probs and not found:
                g = dict(await self.world_gates(vid))
                gates["world"] = {"ok": bool(g.get("ok")), "passed": g.get("passed"), "error": g.get("error")}
                if not g.get("ok"):
                    found.append(f"world gates: passed {g.get('passed')} {g.get('error') or ''}".strip())
        elif knob in PROMPT_KNOBS:
            parent = self.store.window(pid)
            same_tree = tree_hash(Path(self.store.dir(vid)) / "base") == tree_hash(Path(self.store.dir(pid)) / "base")
            same_tests = hidden_hash(rec) == hidden_hash(parent)
            pg = self.gates_dir / f"{pid}.json"
            if same_tree and same_tests and pg.exists() and json.loads(pg.read_text()).get("ok"):
                inherited = {**json.loads(pg.read_text()), "task": vid, "inherited_from": pid,
                             "inherited_because": "a report-only knob: the start tree and every hidden test are byte-identical (sha256)"}
                (self.gates_dir / f"{vid}.json").write_text(json.dumps(inherited, indent=1, default=str))
                gates["world"] = {"ok": True, "inherited_from": pid}
            else:
                found.append(f"world gates not inheritable (tree {same_tree}, tests {same_tests}, parent gated {pg.exists()})")
            if knob == "K4" and not found:
                r = dict(await self.repro(vid))
                gates["S7r"] = {"ok": bool(r.get("ok")), "status": r.get("status")}
                if not r.get("ok") or r.get("status") in ("unverified", "unmeasured"):
                    found.append(f"S7r: {r.get('status')}")
        return GateResult(not found, gates, found)
