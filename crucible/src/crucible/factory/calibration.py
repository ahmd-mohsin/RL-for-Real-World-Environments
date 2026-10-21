"""Calibration exemplars for the task builder: measured training tasks the target student found too easy, in band, too
hard or unfair, with their numbers, and the easing knobs K1-K6 (swarm-pipeline/07 §6.2).

Adapted from browser-use/rl-env-use (MIT; ``rl_env_use/tasks.py`` 185-208, ``calibration``: the drafter is shown the
environment's own difficulty record, what every model solved and what split the panel and why). PATTERN, rewritten: our
record is S15's (``scripts/lh_debug_target.py``'s ``report.json`` ``tasks``: pass rate, mean Φ, per-milestone pass rates,
stops, the keep verdict), our builder is Claude Code reading ``scripts/lh_debug.py brief``, and the levels are the keep
rule's (``targeting.KeepRule``) rather than a model panel's.

**Held-out tasks never enter a brief** (V9: their numbers, traces, prompts and milestones stay sealed). A record is used
only when three independent checks agree that it is a training task: its own ``split``, ``lanes.windows.split_of`` of
its repository, and the eval lease (``sources.lease.EvalLease.default().check``). ``render`` then asserts that no
held-out task id given to it appears in the text (``HeldOutLeak``); a held-out window's brief gets no calibration block
(``brief_block`` refuses it: held-out tasks are eased only for defects, never with K1-K6 from their traces).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

LEVELS = ("too_easy", "in_band", "too_hard", "unfair")
LEVEL_TEXT = {"too_easy": "TOO EASY (the student passes most attempts: it teaches little)",
              "in_band": "IN BAND (pass rate inside the training band: the shape to aim for)",
              "too_hard": "TOO HARD (the student never passes, a frontier solver does: ease it)",
              "unfair": "UNFAIR (no frontier solver passes either: check the report and the World before easing)"}

# the easing knobs, in the order to try them (handoff TASKS_CREDIT_AND_EASING §5.2, decided 2026-10-21)
KNOBS: tuple[tuple[str, str, str], ...] = (
    ("K1", "Drop one fault", "apply the fix of the hardest non-interacting fault in the start tree (one fewer milestone, the interaction kept)"),
    ("K2", "Announce the structure", "say how many separate problems there are and that fixing one may change how another shows up"),
    ("K3", "Point to the area", "name the user-visible feature or package area of each requirement, never the file, function or fix"),
    ("K4", "A visible reproduction per requirement", "a small command or script that shows the bug (the prompt's `repro` field, gate S7r), never a hidden test"),
    ("K5", "Order the requirements", "list them in dependency order, the prerequisite first"),
    ("K6", "Sharpen the stuck requirement", "for the milestone with the lowest pass rate: concrete input, observed output, expected output"),
)
NEVER = ("name files, functions or hidden tests", "describe the fix", "cut to 2 faults or remove the interaction", "relax a gate",
         "raise the budget past 300 tool calls")


class HeldOutLeak(AssertionError):
    """A held-out task's id (so its numbers, prompt or milestones) reached a builder brief."""


def _base(model: str) -> str:
    return str(model or "").split(":", 1)[0]


def is_training(rec: Mapping[str, Any], *, split_of: Callable[[str], str] | None = None, lease: Any = None) -> bool:
    """A training task by all three checks (module doc); any doubt is held out."""
    if str(rec.get("split") or "") != "train":
        return False
    repo = str(rec.get("repo") or "")
    if not repo:
        return False
    if split_of is None:
        from crucible.factory.lanes.windows import split_of as lane_split

        split_of = lane_split
    if split_of(repo) != "train":
        return False
    if lease is None:
        from crucible.factory.sources.lease import EvalLease

        lease = EvalLease.default()
    return bool(lease.check({"repo": repo})["ok"])


def level_of(rec: Mapping[str, Any], *, train_lo: float = 0.1, train_hi: float = 0.6) -> str | None:
    """The record's calibration level, or None (a World or spec defect: not a difficulty measurement)."""
    keep = rec.get("keep") or {}
    v = str(keep.get("verdict") or "")
    if v in ("world_bug", "spec_defect", "unmeasured", ""):
        return None
    p = rec.get("pass_rate")
    if p is None:
        return None
    if v == "unfair" or keep.get("fair") is False:
        return "unfair"
    if p > train_hi:
        return "too_easy"
    if p < train_lo:
        return "too_hard"
    return "in_band"


def load_records(paths: Iterable[Path | str]) -> dict[str, dict[str, Any]]:
    """S15 ``report.json`` files, later files overriding earlier ones per task; each record keeps the run it came from and
    the run's tool limit (numbers at 150 and at 300 calls are not comparable)."""
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        doc = json.loads(p.read_text())
        limit = (doc.get("settings") or {}).get("tool_limit")
        tasks = doc.get("tasks") or {}
        for tid, rec in (tasks.items() if isinstance(tasks, Mapping) else ((r.get("task"), r) for r in tasks)):
            out[str(tid)] = {**rec, "task": str(tid), "_run": p.parent.name, "_tool_limit": (rec.get("settings") or {}).get("tool_limit", limit)}
    return out


def exemplars(records: Mapping[str, Mapping[str, Any]], *, model: str, per_level: int = 3, split_of: Callable[[str], str] | None = None,
              lease: Any = None, train_lo: float = 0.1, train_hi: float = 0.6) -> dict[str, list[dict[str, Any]]]:
    """Up to ``per_level`` training exemplars per level for ``model`` (its base id), with their measured numbers. Held-out
    records are dropped before anything is read from them."""
    out: dict[str, list[dict[str, Any]]] = {lv: [] for lv in LEVELS}
    for tid in sorted(records):
        rec = records[tid]
        if _base(rec.get("model", "")) != _base(model) or not is_training(rec, split_of=split_of, lease=lease):
            continue
        lv = level_of(rec, train_lo=train_lo, train_hi=train_hi)
        if lv is None or len(out[lv]) >= per_level:
            continue
        n = int(rec.get("n") or 0) - int(rec.get("masked") or 0)
        stops = dict(rec.get("stops") or {})
        out[lv].append({"task": tid, "repo": rec.get("repo"), "lang": rec.get("lang"), "level": rec.get("level"), "faults": rec.get("faults"),
                        "tool_limit": rec.get("_tool_limit"), "run": rec.get("_run"), "passes": rec.get("passes"), "n": n, "pass_rate": rec.get("pass_rate"),
                        "mean_phi": rec.get("mean_phi"), "per_milestone": dict(rec.get("per_milestone") or {}),
                        "budget_stops": int(stops.get("max_steps", 0)), "stops": stops, "verdict": (rec.get("keep") or {}).get("verdict"),
                        "why": (rec.get("keep") or {}).get("why"), "frontier": (rec.get("keep") or {}).get("frontier") or rec.get("fairness"),
                        "modes": dict(rec.get("modes") or {})})
    return out


def _line(e: Mapping[str, Any]) -> str:
    ms = ", ".join(f"{m} {v:.2f}" for m, v in sorted((e.get("per_milestone") or {}).items()))
    fr = e.get("frontier") or {}
    fr_s = "; ".join(f"{k.split('/')[-1]} {sum(map(bool, v))}/{len(v)}" for k, v in sorted(fr.items())) if fr else "not run"
    modes = ", ".join(f"{k} {v}" for k, v in sorted((e.get("modes") or {}).items(), key=lambda kv: -kv[1])[:3])
    return (f"- {e['task']} ({e.get('lang')}, {e.get('repo')}, {e.get('faults')} faults, {e.get('tool_limit')} tool calls): pass {e.get('passes')}/{e.get('n')}"
            f" = {e.get('pass_rate'):.2f}, mean Φ {e.get('mean_phi'):.2f}; per milestone {ms or '-'}; budget stops {e.get('budget_stops')}/{e.get('n')};"
            f" frontier {fr_s}; verdict {e.get('verdict')} ({e.get('why')})" + (f"; failure modes {modes}" if modes else ""))


def render(ex: Mapping[str, Sequence[Mapping[str, Any]]], *, model: str, heldout_ids: Iterable[str] = (), this_task: Mapping[str, Any] | None = None) -> str:
    """The calibration block (plain text). Raises ``HeldOutLeak`` if a held-out id is in it."""
    lines = [f"CALIBRATION for the target student {_base(model)} (S15 records, training tasks only; held-out tasks are never shown):"]
    if this_task:
        lines += ["THIS TASK as measured:", _line(this_task)]
    for lv in LEVELS:
        lines.append(LEVEL_TEXT[lv] + ":")
        lines += [_line(e) for e in ex.get(lv) or []] or ["- (none measured)"]
    lines.append("EASING KNOBS, in the order to try them:")
    lines += [f"- {k} {name}: {what}" for k, name, what in KNOBS]
    lines.append("NEVER: " + "; ".join(NEVER) + ".")
    lines.append("Target for an eased task: full passes 10-60% of attempts, mean Φ 0.4-0.8 with spread, a frontier pass, failures on the intended skill.")
    text = "\n".join(lines)
    assert_no_heldout(text, heldout_ids)
    return text


def assert_no_heldout(text: str, heldout_ids: Iterable[str]) -> None:
    leaked = sorted(h for h in heldout_ids if h and h in text)
    if leaked:
        raise HeldOutLeak(f"held-out task(s) in a builder brief: {leaked}")


def brief_block(window_id: str, split: str, records: Mapping[str, Mapping[str, Any]], *, model: str, per_level: int = 3,
                split_of: Callable[[str], str] | None = None, lease: Any = None) -> str:
    """The calibration block of ``window_id``'s brief: training exemplars, this task's own numbers when it is a measured
    training task, the knobs. A held-out window gets none (its brief must not invite K1-K6 easing)."""
    if split != "train":
        return "CALIBRATION: omitted (a held-out window: eased only for defects, never with K1-K6 or from its traces)."
    heldout = [t for t, r in records.items() if not is_training(r, split_of=split_of, lease=lease)]
    ex = exemplars(records, model=model, per_level=per_level, split_of=split_of, lease=lease)
    mine = records.get(window_id)
    one = exemplars({window_id: mine}, model=model, per_level=1, split_of=split_of, lease=lease) if mine is not None else {}
    this = next((e for lv in LEVELS for e in one.get(lv) or []), None)
    return render(ex, model=model, heldout_ids=heldout, this_task=this)
