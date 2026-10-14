"""Stall and loop detection for very long RLM episodes (M16, design §3.2 ``stall.py``; R11, H9; stress B6).

Only signals the policy itself observes go in: the code of each cell, whether it failed, a digest of its printed output,
and each World call's tool, arguments and result digest. Never Φ, a grader check or hidden World state.

Why windowed: the m14 detector counted stalls over the whole episode and stopped at 3, so three honest polls of a
pending ticket 150 cells apart ended a 1,834-step episode at step 1,698 (03-crucible-stress.md B6). Here a verdict
streak decays (``decay_cells`` without a verdict, or any new information from the World), and waiting is told apart
from looping:

- **waiting**: polling whose result changes (a reply arrived, a clock rose: the digest differs), or polling after the
  policy sent a message (``on_world(..., kind="send")``) until a polled result changes. Repeated polls during a wait are
  not a same-cell / same-call / cycle stall; only the slow static-results signature (≥ ``static_results_window`` calls)
  bounds a wait, and only with notices unless verdicts keep coming.
- **looping**: the same cell (code, output and World results) ``repeat_cells`` times, or a cycle of up to ``max_period``
  cells repeated ``repeat_cells`` times (an A/B/A/B loop that never repeats one cell twice in a row); the same
  ``(tool, args)`` with the same result in ≥ ``same_call_hits`` of the last ``same_call_window`` calls (the browsecomp
  guard, 01-frameworks.md §5); the last ``same_call_window`` calls a cycle of ≤ ``max_period`` identical calls; ≤ 2
  distinct ``(call, result)`` pairs over the last ``static_results_window`` calls (H9's static-state signature, keyed by
  call so a batch of identical "ok" writes to different ids is not static); ``max_cell_errors`` failed cells in a row.
  Sending the same payload again is never waiting (Continual Harness's 842 identical sends, CH:1020-1106).

Escalation (H9): each verdict extends the streak; below ``escalate_after`` it is a notice, from ``escalate_after`` a
forced reflection turn (the loop asks the model to state what is wrong and change approach, a normal recorded turn), at
``stop_after`` the scored stop ``stalled``. A verdict consumes the short-window evidence (recent cells, recent calls,
the error run; the static window only for its own signature), so one loop seen by two signatures counts once and the
next verdict needs new evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

__all__ = ["StallDetector", "StallVerdict"]


@dataclass(frozen=True)
class StallVerdict:
    signature: str   # repeat_cells | cell_cycle | same_call | call_cycle | static_results | cell_errors
    evidence: str
    action: str      # notice | reflect | stop
    streak: int = 1

    def message(self) -> str:
        """The text the loop shows the model (``notice`` / ``reflect``) or records as the stop reason (``stop``)."""
        if self.action == "stop":
            return f"stalled: {self.evidence}"
        if self.action == "reflect":
            return (f"[stall-detected: {self.evidence}; {self.streak} stalls in a row] Before your next cell, reflect in plain text: what you "
                    "are trying to do, why your last attempts changed nothing, and what you will do differently. Then continue with a different "
                    "approach. If you are waiting for someone, do other pending work between checks.")
        return (f"[stall-detected: {self.evidence}] Your approach is not making progress. Inspect the state (your variables, OBS, the tool "
                "results), state what is wrong, and change the approach. If you are waiting for someone, do other pending work between checks.")


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def _call_key(tool: str, args: dict[str, Any] | None) -> str:
    try:
        a = json.dumps(args or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        a = repr(args)
    return f"{tool}#{_h(a)}"


def _periodic(seq: list[Any], p: int) -> bool:
    return all(seq[i] == seq[i - p] for i in range(p, len(seq)))


class StallDetector:
    """Windowed stall detection with escalation. Feed it ``on_world`` for every World call or send made by a cell (in
    order) and ``on_cell`` when the cell ends; call ``check()`` after ``on_cell``. ``note_progress()`` marks other new
    information the policy received (an INBOX message, a child event): it ends a wait and resets the streak."""

    def __init__(self, *, repeat_cells: int = 3, same_call_window: int = 10, same_call_hits: int = 6, static_results_window: int = 100,
                 max_cell_errors: int = 4, escalate_after: int = 3, stop_after: int = 6, max_period: int = 4, decay_cells: int = 50):
        if not (1 <= escalate_after <= stop_after):
            raise ValueError("StallDetector: need 1 <= escalate_after <= stop_after")
        if repeat_cells < 2 or same_call_hits < 2 or same_call_window < same_call_hits or max_period < 1 or static_results_window < 3:
            raise ValueError("StallDetector: thresholds out of range")
        self.repeat_cells, self.same_call_window, self.same_call_hits = repeat_cells, same_call_window, same_call_hits
        self.static_results_window, self.max_cell_errors = static_results_window, max_cell_errors
        self.escalate_after, self.stop_after, self.max_period, self.decay_cells = escalate_after, stop_after, max_period, decay_cells
        self._cells: list[list[Any]] = []        # [fingerprint, had_send, code preview]; the last max_period * repeat_cells
        self._cur: list[list[Any]] = []          # World calls of the running cell: [key, digest, is_send]
        self._calls: list[list[Any]] = []        # [key, digest, is_send]; the last max(static window, same-call window)
        self._seen: dict[str, int] = {}          # "key|digest" -> call number last seen (novelty)
        self._keys: dict[str, int] = {}          # key -> call number last seen (a changed result ends a wait)
        self._n_calls = 0
        self._n_cells = 0
        self._short_mark = 0                     # calls consumed by a same_call / call_cycle verdict
        self._static_mark = 0                    # calls consumed by a static_results verdict
        self._errors = 0
        self._waiting = False
        self.streak = 0
        self._last_verdict_cell: int | None = None
        self.verdicts = 0
        self.by_signature: Counter[str] = Counter()

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def on_world(self, tool: str, args: dict[str, Any] | None, result_digest: str, *, kind: str = "call") -> None:
        """One World call (``kind="call"``) or message to a person (``kind="send"``) and a digest of its full result as the
        policy received it (include timestamps: a rising clock is how a poll shows the World moved)."""
        is_send = kind == "send"
        key = _call_key(tool if not is_send else f"send:{tool}", args)
        digest = str(result_digest)
        pair = f"{key}|{digest}"
        self._n_calls += 1
        if pair not in self._seen or self._n_calls - self._seen[pair] > self.static_results_window:
            self.streak = 0  # new information from the World: whatever looped before has moved on
            if key in self._keys and not is_send:
                self._waiting = False  # a polled result changed: the wait is over
        self._seen[pair] = self._n_calls
        self._keys[key] = self._n_calls
        if is_send:
            self._waiting = True
        entry = [key, digest, is_send]
        self._cur.append(entry)
        self._calls.append(entry)
        keep = max(self.static_results_window, self.same_call_window)
        if len(self._calls) > keep:
            del self._calls[: len(self._calls) - keep]
        if len(self._seen) > 8 * keep:  # bounded memory over thousands of calls: forget pairs outside the window
            floor = self._n_calls - self.static_results_window
            self._seen = {k: v for k, v in self._seen.items() if v > floor}
            self._keys = {k: v for k, v in self._keys.items() if v > floor}

    def on_cell(self, code: str, ok: bool, *, output_digest: str | None = None) -> None:
        """A cell ended. ``output_digest``: a digest of what it printed (optional; without it a repeated cell is judged by
        its code and World results alone)."""
        norm = " ".join((code or "").split())
        fp = _h(json.dumps([norm, output_digest, [c[:2] for c in self._cur]]))
        had_send = any(c[2] for c in self._cur)
        self._cells.append([fp, had_send, norm[:80]])
        cap = self.max_period * self.repeat_cells
        if len(self._cells) > cap:
            del self._cells[: len(self._cells) - cap]
        self._cur = []
        self._errors = 0 if ok else self._errors + 1
        self._n_cells += 1

    def note_progress(self, reason: str = "") -> None:
        """New information the policy observed outside a World result (an INBOX message, a child event)."""
        self.streak = 0
        self._waiting = False

    @property
    def waiting(self) -> bool:
        return self._waiting

    # ---- the verdict -----------------------------------------------------------------------------------------------
    def _short_calls(self) -> list[list[Any]]:
        fresh = self._n_calls - self._short_mark
        return self._calls[-min(fresh, self.same_call_window):] if fresh > 0 else []

    def _find(self) -> tuple[str, str] | None:
        if self._errors >= self.max_cell_errors:
            n = self._errors
            self._errors = 0
            return "cell_errors", f"{n} cells in a row failed"
        cells = self._cells
        for p in range(1, self.max_period + 1):
            k = p * self.repeat_cells
            if len(cells) < k:
                break
            window = cells[-k:]
            if not _periodic([c[0] for c in window], p):
                continue
            if self._waiting and not any(c[1] for c in window):
                break  # polling after a send: waiting, not looping
            self._cells = []
            if p == 1:
                return "repeat_cells", f"the same cell ran {k} times in a row with the same results ({window[-1][2]!r})"
            return "cell_cycle", f"the same {p} cells repeated {self.repeat_cells} times with the same results"
        short = self._short_calls()
        if len(short) >= self.same_call_hits:
            keys = Counter(c[0] for c in short)
            key, hits = keys.most_common(1)[0]
            digests = {c[1] for c in short if c[0] == key}
            sends = any(c[2] for c in short if c[0] == key)
            if hits >= self.same_call_hits and len(digests) == 1 and (sends or not self._waiting):
                self._short_mark = self._n_calls
                return "same_call", f"the same World call {key.split('#')[0]} returned the same result {hits} times in the last {len(short)} calls"
        if len(short) == self.same_call_window:
            pairs = [(c[0], c[1]) for c in short]
            for p in range(1, self.max_period + 1):
                if 2 * p <= len(pairs) and _periodic(pairs, p):
                    if self._waiting and not any(c[2] for c in short):
                        break
                    self._short_mark = self._n_calls
                    tools = sorted({c[0].split("#")[0] for c in short})
                    return "call_cycle", f"the last {len(short)} World calls repeat a cycle of {p} ({', '.join(tools)}) with the same results"
        fresh = self._n_calls - self._static_mark
        if fresh >= self.static_results_window:
            last = self._calls[-self.static_results_window:]
            distinct = {(c[0], c[1]) for c in last}
            if len(distinct) <= 2:
                self._static_mark = self._n_calls
                return "static_results", f"the last {len(last)} World calls returned only {len(distinct)} distinct results"
        return None

    def check(self) -> StallVerdict | None:
        """The verdict after the latest cell, or None. A streak decays after ``decay_cells`` cells without a verdict."""
        if self._last_verdict_cell is not None and self._n_cells - self._last_verdict_cell > self.decay_cells:
            self.streak = 0
        found = self._find()
        if found is None:
            return None
        sig, evidence = found
        # one piece of evidence, one verdict: the cells and the recent calls of a loop are the same loop seen twice
        self._cells = []
        self._short_mark = self._n_calls
        self._errors = 0
        self.streak += 1
        self.verdicts += 1
        self.by_signature[sig] += 1
        self._last_verdict_cell = self._n_cells
        action = "stop" if self.streak >= self.stop_after else "reflect" if self.streak >= self.escalate_after else "notice"
        return StallVerdict(signature=sig, evidence=evidence, action=action, streak=self.streak)

    # ---- checkpoint ------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {"cells": [list(c) for c in self._cells], "cur": [list(c) for c in self._cur], "calls": [list(c) for c in self._calls],
                "seen": dict(self._seen), "keys": dict(self._keys), "n_calls": self._n_calls, "n_cells": self._n_cells,
                "short_mark": self._short_mark, "static_mark": self._static_mark, "errors": self._errors, "waiting": self._waiting,
                "streak": self.streak, "last_verdict_cell": self._last_verdict_cell, "verdicts": self.verdicts, "by_signature": dict(self.by_signature)}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._cells = [list(c) for c in d.get("cells", [])]
        self._cur = [list(c) for c in d.get("cur", [])]
        self._calls = [list(c) for c in d.get("calls", [])]
        self._seen = {str(k): int(v) for k, v in d.get("seen", {}).items()}
        self._keys = {str(k): int(v) for k, v in d.get("keys", {}).items()}
        self._n_calls, self._n_cells = int(d.get("n_calls", 0)), int(d.get("n_cells", 0))
        self._short_mark, self._static_mark = int(d.get("short_mark", 0)), int(d.get("static_mark", 0))
        self._errors, self._waiting, self.streak = int(d.get("errors", 0)), bool(d.get("waiting", False)), int(d.get("streak", 0))
        lv = d.get("last_verdict_cell")
        self._last_verdict_cell = int(lv) if lv is not None else None
        self.verdicts = int(d.get("verdicts", 0))
        self.by_signature = Counter({str(k): int(v) for k, v in d.get("by_signature", {}).items()})
