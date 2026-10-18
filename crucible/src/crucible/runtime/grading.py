"""M10 lab audit R-3: grade-only retries (a failed grade is graded again on the same work; the policy is never sampled
again because a grade failed). PATTERN: verifiers v1 ``grading.py:81-170`` (grader-only retries restored from artifacts,
under one scoring deadline); Harbor's regrade from artifacts.

- ``grade_failure(rep)``: the failure a grade report carries, or None. A masked report goes through
  ``classify_verifier`` (unless its reason quotes checker or judge output, ``reason_untrusted``: then its
  ``failure_kind`` alone decides, since an injection answer steers that text); a non-finite reward is ``verifier_error``
  (detail ``non_finite_reward``): before R-3 a NaN reward reached ``HarnessResult``'s ``allow_inf_nan=False``, became a
  harness crash at the agent stage and was retried as a fresh agent run.
- ``grade_in_place(grade, gdir)``: the same work graded up to ``1 + retries`` times inside one attempt, under one
  optional scoring budget (its expiry during the first grade is ``verifier_timeout``, never ``transport_timeout``; during
  a later grade the earlier try's failure stands, noting the budget and the seconds the cut try had, ``budget_cut_s``,
  review round 3). The budget counts only the grade's own time:
  infrastructure inside a grade (a separate verifier World's start, restore and close; Harbor in place: the liveness
  probe, the R-2 quiesce, the earlier grade's leftovers; ``HarborDirTask``: the image inspect and the host transfers) runs
  under ``unbudgeted()``, which holds the budget while it runs (review round 2: a serial host lock or a slow quiesce that
  ran the budget out was the answer's ``verifier_timeout``, and a control that needs no such wait scored the policy 0)
  and, once it ends, gives the time it held back to the Runner's attempt clock too (review round 3). A host check whose
  length the answer sets (Harbor in place: the grader guard's scan fetches every changed file the policy left) holds it
  at most ``HOLD_CAP_S`` per ``grade_in_place`` call (``unbudgeted(answer=True)``), the attempt clock moved past that
  allowance while it runs; past it the check runs on the scoring budget again.
  An answer-sensitive failure it returns carries ``evidence.tries``: the answer-sensitive grades of this work that ended
  in it (``seen``: those an earlier attempt already ran). Try ``i > 0`` grades into
  ``<gdir.parent>/regrade<i>/<gdir.name>``, so whatever a grader writes next to its grade dir (Harbor in place: the R-2
  quiesce receipt ``<workdir>/../verifier/quiesce.json``) is per try. A ``TimeoutError`` that escapes the grade function
  (the budget's own expiry excluded) is the grader's ``verifier_timeout`` too: through ``boundary("verification")`` it
  would read as a transport timeout of the infrastructure, masked and never answered by the answer's own 0. Any other
  exception is classified as ``boundary("verification")`` would (``infra.classify_exception``) and graded again under
  the same rule as a failure a report carries.
- Which failures are graded again: ``regradable`` (in place: the World is gone after the attempt, so any retryable
  non-policy grade failure is retried here, a raised one included) or ``answer_sensitive`` (a persisted hand-off or
  patch: a contaminated node or a transport failure waits for a grade-only retry in a fresh attempt instead; a failure
  the answer can cause, a hang or a checker outage on this answer, is graded again at once, and an answer-sensitive
  failure that repeats is decided in the attempt, never left to ``RetryRule``).
- ``HANDOFF_REF``: the native separate verifier's persisted hand-off (``verifier/handoff_ref.json``: the quiesced
  ``WorldRef`` and the spec, store, trace length and episode ``verifier_world`` needs), read by ``NativeHarness.regrade``.
  ``deciding_record``: the earlier attempt whose record decides whether attempt ``n`` re-grades (an attempt the process
  died in, or a grade-only retry that reached no verdict, is looked past).
- ``grading_window(attempt, task)``: the grading's own time on the Runner's attempt clock (review round 2): the
  scoring budget defaults to the clock's length, and the clock is moved past that budget, the control's and a margin, so a
  grade that hangs ends inside the attempt as ``verifier_timeout`` and ``settle`` decides it (cut by the clock, it was an
  ``attempt_timeout`` mask the control never saw: a hang escaped its 0). ``agent_clock`` gives a multi-step task's next
  agent phase back the time its clock had left. ``control_window`` (review round 3): ``settle`` moves the clock past the
  control's own budget just before it runs, so infrastructure that ran between the window opening and the control (a
  ``PerfGuard`` slot, the policy World's close) never leaves a decided failure without its control.
- ``settle(task, f)`` (R-3 item 2): the last word on a grade failure no further grade of this work will answer. An
  answer-sensitive failure that repeated on the same work (``decided``: two answer-sensitive grades, or a first grade
  that ran out the whole scoring budget: flag ``not_repeated``, and the control must then grade within that budget; a
  later try cut by what was left of it is no grade of the answer, review round 3) is decided
  by grading the task's **control**
  (the optional ``RunTask.grade_control(workdir)`` hook: the oracle, a reference answer or a canary, graded by the same
  grader, under its own budget ``control_timeout_s``, raised to the task's own ``control_timeout_s`` when it declares a
  longer one; ``control_grade``). Control passes (the grader ran cleanly on known-good work):
  ``crucible:policy_error`` / ``grader_failed_on_answer``, scored 0 (the hang or the checker failure is the answer's;
  before R-3 a policy whose answer hung the tests escaped its 0 and pass@k became conditional on "did not hang").
  Control fails, raises or times out (repo2rlenv runner.py:76-138: a control that does not run cleanly fails it):
  ``verifier_error``, task blame, masked, never retried. No control: the failure stays masked, never retried, with
  flag ``no_control``. An answer-sensitive failure graded only once (``grade_retries=0``, a step graded once, or an
  earlier try that failed for another reason) is masked with flag ``not_repeated`` and never controlled (review round
  2: one such grade says nothing yet about the answer). A dead liveness probe (``liveness``: the task's ``infra_cause``
  named a cause) skips the control (the sandbox, not the answer). Every result carries ``evidence = {probe, control, status}`` (the R-5 task breaker
  reads ``control``). Accepted trade-off (Reasoning): a flaky grader whose control passes by chance scores the policy
  0; the evidence makes each such case auditable.

Only ``core`` and ``runtime.infra`` are imported here: interop harnesses (Harbor in place) use it too.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

from crucible.core.schema.run import NON_RETRYABLE_KINDS, Failure
from crucible.runtime.infra import NORMALIZE, StageFailure, _causes, classify_exception, classify_verifier, failure

HANDOFF_REF = "handoff_ref.json"
# grade failures the answer itself can cause: a hang of the tests on this answer, a checker or judge that fails only on it
# (CheckerUnavailable -> check_infra), a reward the grader cannot turn into a number. They are graded again inside the
# attempt, and (R-3 item 2) one that repeats on the same work is what a control grade decides. A patch-mode non-finite
# reward arrives as a masked report: ``HarborDirTask.grade_patch`` names it ``non_finite_reward`` (a NaN or infinite
# reward file, or the kit's own non-finite mask)
ANSWER_SENSITIVE = frozenset({"verifier_timeout", "VerifierTimeoutError", "check_infra", "non_finite_reward"})
# M10 lab audit R-3 review round 3: the most a grade's answer-dependent host checks (``unbudgeted(answer=True)``) may hold
# the scoring budget, per ``grade_in_place`` call (read at call time, so a test can lower it)
HOLD_CAP_S = 600.0


def finite(x: Any) -> float | None:
    """``x`` as a finite float, else None (a NaN or infinite reward is no reward)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def grade_failure(rep: Any) -> Failure | None:
    """The failure a grade report carries: ``classify_verifier`` of a masked report, ``verifier_error`` /
    ``non_finite_reward`` for a reward that is not a finite number, else None. A masked report whose reason quotes
    checker or judge output (``infra.reason_untrusted``: ``SingleTurnTask``'s ``CheckerUnavailable``) is classified by
    its ``failure_kind`` alone, never sniffed for a node, image or build cause."""
    infra = getattr(rep, "infra", None) or {}
    if infra.get("status") == "masked":
        if infra.get("reason_untrusted"):
            # M10 lab audit R-3: the answer steers that text. An injection answer ("reply with exactly: stale image") turned
            # classify_verifier's sniffing into crucible:stale_image / node_contention, which is not answer_sensitive: the
            # control never ran and the sample was masked instead of scored 0
            return failure(infra.get("failure_kind") or "grader_exception", "verification", infra.get("reason") or "")
        return classify_verifier(infra)
    if finite(getattr(rep, "reward", None)) is None:
        return failure("verifier_error", "verification", f"the grader returned a non-finite reward {getattr(rep, 'reward', None)!r}", detail="non_finite_reward")
    return None


def regradable(f: Failure) -> bool:
    """A grade failure another grade of the same work can answer: retryable, not a kind no retry fixes (a stale image or
    build, an unsupported harness), and not the policy's (scored) or of unknown blame (counted, never retried)."""
    return f.retryable and f.kind not in NON_RETRYABLE_KINDS and f.who in ("infra", "harness", "task")


def answer_sensitive(f: Failure) -> bool:
    """A grade failure the graded answer can cause (``ANSWER_SENSITIVE``): a ``verifier_error`` whose detail is a
    timeout, a checker outage or a non-finite reward."""
    return f.kind == "verifier_error" and (f.detail or "") in ANSWER_SENSITIVE


def decided(f: Failure) -> bool:
    """M10 lab audit R-3 review round 2: an answer-sensitive failure the attempt decides now (``settle``, the control):
    two or more answer-sensitive grades of this work ended in it (``evidence.tries``, ``grade_in_place``), or the first
    grade ran out the whole scoring budget (``budget_s``). One answer-sensitive grade alone stays regradable where the
    work is persisted, so a grade-only retry grades it again (``grade_retries=0`` went straight to the control, which
    scored the policy 0 for a single checker blip). Review round 3: a later try the budget's remainder cut short
    (``budget_cut_s``) is no grade of the answer and decides nothing (it carried ``budget_s``, and one infrastructure
    stall plus a second grade that started with seconds left sent an honest answer to the control)."""
    ev = f.evidence or {}
    return answer_sensitive(f) and (ev.get("tries") != 1 or "budget_s" in ev)


def seen_before(sample_dir: Path, n: int) -> int:
    """M10 lab audit R-3 review round 2: the answer-sensitive grades an earlier attempt already ran on the work attempt
    ``n`` re-grades (``evidence.tries`` of the deciding record's ``verifier_error``, ``deciding_record``), passed to
    ``grade_in_place(seen=)`` so a repeat on the grade-only retry is decided there. 0 when that record is no such
    failure."""
    got = deciding_record(sample_dir, n)
    f = (got[1].get("failure") or {}) if got is not None else {}
    if not isinstance(f, dict) or f.get("kind") != "verifier_error" or (f.get("detail") or "") not in ANSWER_SENSITIVE:
        return 0
    try:
        return max(0, int((f.get("evidence") or {}).get("tries") or 0))
    except (TypeError, ValueError, AttributeError):
        return 0


def try_dir(gdir: Path, i: int) -> Path:
    """The grade dir of try ``i``: ``gdir`` itself, then ``<gdir.parent>/regrade<i>/<gdir.name>``."""
    gdir = Path(gdir)
    return gdir if i == 0 else gdir.parent / f"regrade{i}" / gdir.name


def _raised_timeout(ex: BaseException) -> bool:
    """``ex`` or a link of its ``__cause__`` chain is a ``TimeoutError``, and no earlier link is already classified (a
    ``StageFailure``, a NORMALIZE name such as Harbor's ``EnvironmentStartTimeoutError``: ``boundary``'s first match)."""
    for link in _causes(ex):
        if isinstance(link, StageFailure) or type(link).__name__ in NORMALIZE:
            return False  # classified on the side that raised
        if isinstance(link, TimeoutError):
            return True
    return False


class _Budget:
    """The scoring budget of one try (``grade_in_place``): its ``asyncio.Timeout``, the seconds ``unbudgeted`` held it,
    ``allow`` (a one-element list shared by every try of one ``grade_in_place`` call: what is left of ``HOLD_CAP_S`` for
    answer-dependent checks) and the Runner's attempt clock (``Attempt.clock``, or None)."""

    __slots__ = ("allow", "clock", "cm", "held")

    def __init__(self, cm: asyncio.Timeout, allow: list[float], clock: Any = None) -> None:
        self.cm, self.held, self.allow, self.clock = cm, 0.0, allow, clock


_BUDGET: contextvars.ContextVar[_Budget | None] = contextvars.ContextVar("crucible_scoring_budget", default=None)


def _lend(clock: Any, s: float) -> None:
    """Move the Runner's attempt clock by ``s`` seconds (later when positive), unless there is none, it has no deadline
    or it has already fired (M10 lab audit R-3 review round 3)."""
    if clock is None or not s:
        return
    with contextlib.suppress(RuntimeError, AttributeError):
        when = clock.when()
        if when is not None and not clock.expired():
            clock.reschedule(when + s)


@contextlib.contextmanager
def unbudgeted(*, answer: bool = False) -> Iterator[None]:
    """M10 lab audit R-3 review round 2: infrastructure inside a grade (a separate verifier World's start, restore,
    fingerprint check and close; Harbor in place: ``_liveness``, the R-2 quiesce, ``_regrade_leftovers``;
    ``HarborDirTask``: the image inspect and the host put/get) does not run on the answer's scoring budget. The try's
    deadline is held while it runs and moved later by the time it took, so a budget expiry is always the grade's own and
    an infrastructure wait is bounded by its own timeouts (a serial lock's wait) or the Runner's attempt clock. Before, a
    budget that ran out in a serial host lock or a quiesce round was the answer's ``verifier_timeout``, and a control that
    needed no such wait scored the policy 0 (P16). A no-op outside ``grade_in_place``, without a budget, once the budget
    has expired, and when nested (the outer hold already covers it).

    Review round 3: once a hold ends, the time it held is given to the Runner's attempt clock as well (``grade_in_place``'s
    ``attempt``): ``grading_window`` moved the clock once, so a host wait that finished (a verifier World's start, the
    quiesce, a transfer) used up the time reserved for the grade's later tries and the control, and a decided hang ended
    as an ``attempt_timeout`` mask, never controlled. A wait that never ends is still cut by the clock.

    ``answer=True`` (review round 3): a host check whose length the answer sets (Harbor in place: the grader guard's scan
    fetches every changed file the policy left, one ``docker cp`` each) holds the budget at most what is left of
    ``HOLD_CAP_S`` for this ``grade_in_place`` call; the attempt clock is moved past that allowance while it runs and
    given back what the check did not use. Past the allowance the budget runs again and can expire inside the check (the
    answer's ``verifier_timeout``, decided and controlled); once it is spent the check runs on the budget. Uncapped, enough
    planted files held the budget until the clock fired, and a 0 became a terminal ``attempt_timeout`` mask."""
    b = _BUDGET.get()
    when = None if b is None or b.cm.expired() else b.cm.when()
    if b is None or when is None:
        yield
        return
    cap = b.allow[0] if answer else None
    if cap is not None and cap <= 0:
        yield  # the allowance is spent: the check runs on the scoring budget
        return
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        b.cm.reschedule(None if cap is None else when + cap)
    except RuntimeError:  # the try already ended (a task its grade left behind): there is nothing to hold
        b = None
    if b is None:
        yield
        return
    if cap is not None:
        _lend(b.clock, cap)  # the clock covers the whole allowance while the check runs
    try:
        yield
    finally:
        held = loop.time() - t0
        if cap is not None:
            held = min(held, cap)
            b.allow[0] -= held
            _lend(b.clock, held - cap)  # what the check did not use of the allowance
        else:
            _lend(b.clock, held)
        b.held += held
        if not b.cm.expired():
            with contextlib.suppress(RuntimeError):
                b.cm.reschedule(when + held)


async def grade_in_place(grade: Callable[[Path], Awaitable[Any]], gdir: Path, *, retries: int = 1, budget_s: float | None = None,
                         again: Callable[[Failure], bool] = regradable, out: list[Any] | None = None, seen: int = 0,
                         on_again: Callable[[Failure], None] | None = None, attempt: Any = None) -> tuple[list[Any], Failure | None]:
    """Grade the same work with ``grade(dir)`` up to ``1 + retries`` times (try ``i`` into ``try_dir(gdir, i)``), while
    the last grade failed with a failure that is ``regradable`` and that ``again`` accepts. Returns ``(reports, failure)``: every report that came back,
    in order (empty when the budget ran out during the first), and the failure of the last try, or None.

    ``budget_s``: one scoring budget for every try (a later try is not started once it is spent), counting only the
    grade's own time (``unbudgeted`` holds it during infrastructure). Its expiry during the first grade ends it as
    ``verifier_timeout`` with ``evidence = {budget_s, tries}`` (``tries``: 1 plus ``seen``; ``settle`` flags a 1
    ``not_repeated`` and holds the control to that budget); during a later grade the earlier try's failure stands, its
    reason noting the budget (``tries``: the answer-sensitive grades that ended in it, else the grades before), as when
    the budget runs out between tries: that try used up the budget that cut this one short, which says nothing about the
    answer. Review round 3: that later expiry adds ``budget_cut_s`` (the seconds of its own the cut try had), never
    ``budget_s``, so it neither counts as a repeat nor makes the failure budget-decided (``decided``, ``settle``): the same
    outcome as a budget spent before that try started. A ``TimeoutError`` raised by the
    grade function (or in its ``__cause__`` chain, unless an earlier link is already classified) is ``verifier_timeout``
    as well. Any other exception is classified as the caller's ``boundary("verification")`` would
    (``infra.classify_exception``): graded again when ``regradable`` and ``again`` accept it (in place: a transport
    failure of the grader; any caller: a raised ``VerifierTimeoutError``), else re-raised unchanged (unknown blame, a kind
    no grade fixes, or one left to a fresh attempt). A ``StageFailure`` always propagates (R-2's ``respawned`` /
    ``uninterruptible`` / ``setup_service`` verdicts end the attempt unchanged).

    ``out`` (M10 lab audit R-3): the caller's list, filled with every report that came back even when a later try raises,
    so the reports already graded (and their flags) are kept before it propagates.

    Review round 2: ``seen``, the answer-sensitive grades an earlier attempt already ran on this work (a grade-only
    retry, ``seen_before``); an answer-sensitive failure returned carries ``evidence.tries``, those grades plus this
    call's, so ``settle`` controls only a failure that repeated (one after an infra failure, or with ``retries=0``, was
    controlled as if it had). ``on_again(f)``: called with the last failure just before try ``i > 0`` starts (the persisted
    paths clear ``Attempt.regradable`` there once an answer-sensitive failure is being graded again, so an attempt clock
    firing during that grade ends the attempt instead of scheduling a grade-only retry).

    ``attempt`` (review round 3): the harness's ``Attempt``; its ``clock`` gets back the time ``unbudgeted`` held, and is
    moved past an answer-dependent check's allowance (``HOLD_CAP_S``, one per call, shared by every try) while it runs."""
    reps: list[Any] = [] if out is None else out
    f: Failure | None = None
    hits = max(0, seen)  # the answer-sensitive grades of this work so far
    allow = [HOLD_CAP_S]  # review round 3: read at call time (a module global, never a default argument)
    clock = getattr(attempt, "clock", None)
    deadline = None if budget_s is None else time.monotonic() + max(0.0, budget_s)
    for i in range(1 + max(0, retries)):
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0 and i > 0:
            break  # the budget is spent: the last try's failure stands
        if i > 0 and f is not None and on_again is not None:
            on_again(f)
        cm = asyncio.timeout(left)
        b = _Budget(cm, allow, clock)
        tok = _BUDGET.set(b)  # unbudgeted() inside the grade holds this try's deadline
        try:
            try:
                async with cm:
                    rep = await grade(try_dir(gdir, i))
            except StageFailure:
                raise
            except Exception as ex:
                if isinstance(ex, TimeoutError) and cm.expired():
                    # M10 lab audit R-3: the shared budget ran out. During the first grade, the grade itself ran past it
                    # (verifier_timeout, answer-sensitive). During a later one, the earlier try's failure stands: relabelled
                    # verifier_timeout, an infra failure re-graded in place (a contaminated node, a lost grader network) became
                    # answer-sensitive, and settle's control could score the policy 0 for it (P16)
                    note = f"the scoring budget of {budget_s:g}s ran out during grade {i + 1}"
                    if i == 0 or f is None:
                        f = failure("verifier_timeout", "verification", note, evidence={"budget_s": budget_s, "tries": hits + 1})
                    else:
                        # M10 lab audit R-3 review round 3: the cut try had only `left` s of the grade's own time (unbudgeted
                        # moves both the budget and `deadline`). It is no grade of the answer, so it neither counts as a repeat
                        # nor makes the failure budget-decided (decided and settle read budget_s, which only a first grade that
                        # ran out the whole budget carries): the same outcome as the left <= 0 break above. With budget_s, one
                        # infrastructure stall plus a second grade started with seconds left sent the failure to the control,
                        # held to the whole budget, and an honest answer scored 0
                        tries = hits if answer_sensitive(f) else i
                        f = f.model_copy(update={"reason": f"{f.reason}; {note}"[:500],
                                                 "evidence": {**(f.evidence or {}), "budget_cut_s": round(left or 0.0, 3), "tries": tries}})
                    break
                if _raised_timeout(ex):
                    via = "" if isinstance(ex, TimeoutError) else " (from a timeout)"
                    f = failure("verifier_timeout", "verification", f"the grader raised {type(ex).__name__}{via} during grade {i + 1}: {ex}"[:500])
                else:
                    # M10 lab audit R-3: a grade that raises is a grade failure like a masked report, classified as the caller's
                    # boundary would. It propagated, so an in-place grader's ConnectionResetError was graded once and ended
                    # terminal-masked (an in-place World is never regradable), and a raised VerifierTimeoutError was never
                    # graded again or decided by the control
                    f = classify_exception(ex, "verification")
                    if not (regradable(f) and again(f)):
                        raise
            else:
                reps.append(rep)
                f = grade_failure(rep)
        finally:
            _BUDGET.reset(tok)
            if deadline is not None:
                deadline += b.held  # the time infrastructure held this try's budget is not the grade's
        if f is not None and answer_sensitive(f):
            hits += 1
        if f is None or not (regradable(f) and again(f)):
            break
    if f is not None and answer_sensitive(f) and "tries" not in (f.evidence or {}):
        # M10 lab audit R-3 review round 2: how often this work's grade ended answer-sensitive (settle and decided read it)
        f = f.model_copy(update={"evidence": {**(f.evidence or {}), "tries": hits}})
    return reps, f


def deciding_record(sample_dir: Path, n: int) -> tuple[Path, dict[str, Any]] | None:
    """M10 lab audit R-3: the earlier attempt whose record decides how attempt ``n`` retries, and that record. An attempt
    the process died in has no record (the journal moved it to ``aborted/``, ``RunJournal.next_attempt``) and is looked
    past, as ``checkpoint.attempt_path`` does for a resume; so is a grade-only retry the Runner recorded outside the
    verification stage (a raise around ``regrade``: stage ``agent``), recognised by the hand-off it carried
    (``verifier/HANDOFF_REF`` whose ``source`` is an earlier attempt). Before this only ``a{n-1}``'s record was read, so a
    crash during a re-grade made the next attempt run the agent again. Used by ``NativeHarness``, ``WorkspaceHarness`` and
    the Runner's ``grade_only`` journal flag alike."""
    for k in range(n - 1, 0, -1):
        d = Path(sample_dir) / f"a{k}"
        try:
            r = json.loads((d / "record.json").read_text())
        except (OSError, ValueError):
            continue  # died before its record (moved to aborted/): the decision is an earlier attempt's
        if not isinstance(r, dict):
            continue
        f = r.get("failure") or {}
        if r.get("mask_sample") and f.get("stage") != "verification":
            try:
                carried = json.loads((d / "verifier" / HANDOFF_REF).read_text()).get("source") not in (None, d.name)
            except (OSError, ValueError, AttributeError):
                carried = False
            if carried:
                continue  # a grade-only retry that reached no verdict: still re-grading the same work
        return d, r
    return None


def write_reports(vdir: Path, reps: list[Any]) -> None:
    """``vdir/report.json`` is the last report (the one that decided). When more than one grade ran, every try's report
    is kept as ``report.r<i>.json`` and ``grades.json`` lists each try's dir, infra and reward. Nothing is written when no
    report came back (the scoring budget ran out during the first grade)."""
    if not reps:
        return
    vdir = Path(vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "report.json").write_text(json.dumps(reps[-1].to_dict(), default=str, indent=1))
    if len(reps) < 2:
        return
    rows = []
    for i, r in enumerate(reps):
        (vdir / f"report.r{i}.json").write_text(json.dumps(r.to_dict(), default=str, indent=1))
        rows.append({"try": i, "report": f"report.r{i}.json", "reward": finite(getattr(r, "reward", None)), "infra": getattr(r, "infra", None)})
    (vdir / "grades.json").write_text(json.dumps(rows, default=str, indent=1))


def grade_summary(reps: list[Any], f: Failure | None) -> dict[str, Any]:
    """``HarnessResult.grade``'s ``reward`` / ``gates`` / ``infra`` from the last report (``finite`` reward); with no
    report (the scoring budget ran out during the first grade) the infra is the failure, masked."""
    if not reps:
        return {"reward": None, "gates": {}, "infra": {"status": "masked", "failure_kind": (f.detail or f.kind) if f else "verifier_timeout",
                                                       "reason": f.reason if f else ""}}
    rep = reps[-1]
    out: dict[str, Any] = {"reward": finite(rep.reward), "gates": rep.gate_bits(), "infra": rep.infra}
    if len(reps) > 1:
        out["grades"] = len(reps)
    return out


# ---------------------------------------------------------------------------- the control grade (R-3 item 2)


def control_passed(rep: Any, *, min_outcome: float | None = None) -> bool:
    """Did the control grade run cleanly: a report, not masked, a finite reward and outcome, and no ``meta.failure``
    (the grader's own verdict on the work, e.g. ``agent_broke_env``). The control shows that the grader runs on
    known-good work, not that the oracle is perfect (critic amendment 6: a perf gate or partial credit kept an oracle
    below 1.0 and would turn every answer-caused hang into the task's mask); a task may still ask for an outcome floor
    (``control_min_outcome``)."""
    if rep is None:
        return False
    if (getattr(rep, "infra", None) or {}).get("status") == "masked":
        return False
    outcome = finite(getattr(rep, "outcome", None))
    if finite(getattr(rep, "reward", None)) is None or outcome is None:
        return False
    if (getattr(rep, "meta", None) or {}).get("failure"):
        return False
    return min_outcome is None or outcome >= min_outcome


def control_budget(task: Any, timeout_s: float | None) -> float | None:
    """M10 lab audit R-3: a control grade's budget: the harness's ``timeout_s``, raised to the task's own
    ``control_timeout_s`` when it declares a longer one (what one honest control may take: its grader's own timeout, plus
    a fresh World and the oracle's run in place). A harness budget below the task's grade timeout cut a healthy grader's
    control, counted it failed, and turned every answer-caused hang into a task-blamed mask. None: no cap."""
    need = getattr(task, "control_timeout_s", None)
    return timeout_s if timeout_s is None or need is None else max(float(timeout_s), float(need))


def grading_window(attempt: Any, task: Any, *, budget_s: float | None, control_s: float | None) -> float | None:
    """M10 lab audit R-3 review round 2: the scoring budget of the grading that starts now, on an attempt the Runner
    clocks (``Attempt.clock``, its ``asyncio.Timeout``, and ``Attempt.timeout_s``, its length). Called where a harness
    enters the verification phase. Returns ``budget_s``, or the clock's length when it is None and there is a clock (a
    grader without a timeout of its own is then bounded by the scoring budget, never only by the clock); and moves the
    clock past ``now + scoring budget + control budget (control_budget(task, control_s), the clock's length when None) +
    a margin`` (the clock's length, at most 60 s), never earlier. A hang then ends inside the attempt as
    ``verifier_timeout`` and ``settle`` decides it with the control; before, the clock (sized for the agent and one grade)
    cut the in-attempt re-grades and the control short and recorded an ``attempt_timeout`` mask: an answer that hung the
    tests escaped its 0, or was re-graded until ``max_attempts`` (R-3 Reasoning: masking grader failures rewards hanging).
    The clock stays the backstop for infrastructure (``unbudgeted``) and cleanup. Records what the clock had left for the
    agent (``Attempt.agent_left``, ``agent_clock``). Without a clock: ``budget_s``, nothing moved.

    Review round 3: the clock is moved here once, but not only here: a hold of ``unbudgeted`` gives the clock back the
    time it took once it ends (an answer-dependent check: its ``HOLD_CAP_S`` allowance while it runs), ``control_window``
    moves it past the control's budget just before the control runs, and a caller may call this again once a wait outside
    the grade is over (``WorkspaceHarness._grade``: the ``PerfGuard`` slot). Only an infrastructure wait still running
    when the clock fires ends the attempt (``attempt_timeout``)."""
    clock, total = getattr(attempt, "clock", None), getattr(attempt, "timeout_s", None)
    if clock is None or total is None:
        return budget_s
    scoring = float(total) if budget_s is None else budget_s
    when = clock.when()
    if when is None or clock.expired():
        return scoring
    now = asyncio.get_running_loop().time()
    if getattr(attempt, "agent_left", None) is None:
        attempt.agent_left = max(0.0, when - now)
    ctl = control_budget(task, control_s)
    need = now + scoring + (float(total) if ctl is None else ctl) + min(60.0, float(total))
    if need > when:
        with contextlib.suppress(RuntimeError):  # a clock that has already ended has nothing to move
            clock.reschedule(need)
    return scoring


def control_window(attempt: Any, task: Any, timeout_s: float | None) -> None:
    """M10 lab audit R-3 review round 3: the control's own window on the Runner's attempt clock, opened just before
    ``settle`` grades it: the clock is moved past ``now + control budget (control_budget(task, timeout_s), the clock's
    length when None) + a margin`` (the clock's length, at most 60 s), never earlier. ``grading_window`` reserved that
    time when verification opened, but infrastructure that ran since on the clock (a ``PerfGuard`` slot's lock and
    quiet-host wait, a separate verifier's hand-off, the policy World's close and liveness probe) used it up: the clock
    fired during the control of a decided failure, which ended as a terminal ``attempt_timeout`` mask instead of its 0. A
    no-op without a clock or once it has fired."""
    clock, total = getattr(attempt, "clock", None), getattr(attempt, "timeout_s", None)
    if clock is None or total is None:
        return
    when = clock.when()
    if when is None or clock.expired():
        return
    ctl = control_budget(task, timeout_s)
    need = asyncio.get_running_loop().time() + (float(total) if ctl is None else ctl) + min(60.0, float(total))
    if need > when:
        with contextlib.suppress(RuntimeError):
            clock.reschedule(need)


def agent_clock(attempt: Any) -> None:
    """M10 lab audit R-3 review round 2: a multi-step task's next agent phase gets back what the attempt clock had left
    when the previous step's grading window opened (``grading_window``), so the grading's time never extends an agent
    phase. A no-op without a clock or a recorded remainder."""
    clock, left = getattr(attempt, "clock", None), getattr(attempt, "agent_left", None)
    if clock is None or left is None or clock.expired():
        return
    attempt.agent_left = None
    with contextlib.suppress(RuntimeError):
        clock.reschedule(asyncio.get_running_loop().time() + left)


async def control_grade(task: Any, workdir: Path, *, timeout_s: float | None, budget_s: float | None = None) -> tuple[str, dict[str, Any]]:
    """Grade the task's control into ``workdir``: ``("none", {})`` without a ``grade_control`` hook (or when it returns
    None), else ``("passed" | "failed", info)``. Anything that keeps the control from running cleanly fails it: an
    exception (a ``StageFailure`` included), its budget (``control_budget(task, timeout_s)``) running out, a report that
    cannot be written (repo2rlenv runner.py:76-138). ``budget_s`` (M10 lab audit R-3): the scoring budget the answer's
    grade ran out of; a control whose own grade took longer (``elapsed_s``, else the hook's wall time) shows nothing
    about the grader within that budget and fails. ``info``: ``control_reward`` / ``control_outcome`` /
    ``control_elapsed_s``, or ``control_error`` (with ``control_budget_s``). Never raises but ``CancelledError`` (critic
    amendment 3: the workdir is created here, and the report written inside the guard)."""
    hook = getattr(task, "grade_control", None)
    if hook is None:
        return "none", {}
    workdir = Path(workdir)
    cap = control_budget(task, timeout_s)
    t0 = time.monotonic()
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        async with asyncio.timeout(cap):
            rep = await hook(workdir)
        if rep is None:
            return "none", {}
        (workdir / "report.json").write_text(json.dumps(rep.to_dict(), default=str, indent=1))
    except Exception as e:  # noqa: BLE001 -- a control that does not run cleanly fails the control, never the policy
        return "failed", {"control_error": f"{type(e).__name__}: {e}"[:300], **({"control_budget_s": cap} if cap is not None else {})}
    took = finite(getattr(rep, "elapsed_s", None))
    took = round(time.monotonic() - t0, 3) if took is None else took
    info = {"control_reward": finite(getattr(rep, "reward", None)), "control_outcome": finite(getattr(rep, "outcome", None)), "control_elapsed_s": took}
    if budget_s is not None and took > budget_s:
        return "failed", {**info, "control_error": f"the control graded in {took:g}s, over the scoring budget of {budget_s:g}s the answer's grade had"}
    return ("passed" if control_passed(rep, min_outcome=getattr(task, "control_min_outcome", None)) else "failed"), info


async def liveness(task: Any, world: Any) -> str:
    """The sandbox's liveness before a grade failure is blamed (verifiers v1/harness.py:173-193: probe before blaming
    the sandbox): ``dead`` when the task's ``infra_cause(world)`` names a cause, ``alive`` when it finds none,
    ``unprobed`` without the hook or a World, or when asking failed."""
    probe = getattr(task, "infra_cause", None)
    if probe is None or world is None:
        return "unprobed"
    try:
        return "dead" if await probe(world) is not None else "alive"
    except Exception:  # noqa: BLE001 -- asking the host is best effort
        return "unprobed"


async def settle(task: Any, f: Failure, *, workdir: Path, timeout_s: float | None, probe: str = "unprobed", control: bool = True,
                 attempt: Any = None) -> Failure:
    """M10 lab audit R-3 item 2: the last word on a grade failure ``f`` that no further grade of this work will answer
    (in place after the in-attempt re-grades; on a persisted patch or hand-off once an answer-sensitive failure repeated,
    or once the scoring budget ran out during its first grade).

    A failure that does not mask is returned as it is. A masked one gets ``evidence = {probe, control, status}`` (merged
    over what ``f`` already carries: a budget expiry's ``budget_s`` / ``tries``); one that is not ``answer_sensitive`` (a
    contaminated node, a stale image, a transport failure), or whose probe says the sandbox is ``dead``, is not
    controlled. ``control=False`` (a multi-step step whose task declares ``step_grades_once``: R-2's per-step state is
    used up by its grade) masks it with flag ``no_control``. Otherwise the task's control is graded (``control_grade``,
    own budget): passed -> ``crucible:policy_error`` / ``grader_failed_on_answer`` (policy blame, scored 0, not retried);
    failed -> ``verifier_error`` with ``f``'s detail, task blame, masked, not retried; none -> ``f`` masked, not retried,
    flag ``no_control``. A failure graded only once (``tries == 1``) is flagged ``not_repeated``: when the scoring budget
    ran out during that grade (``budget_s``) its control must grade within that budget (``control_grade(budget_s=)``),
    else it fails (a harness setting is never charged to the policy); otherwise (review round 2: ``grade_retries=0``, a
    step graded once, an answer-sensitive try after a transport failure or a contaminated node, review round 3: a later
    try the budget's remainder cut, ``budget_cut_s``) it is masked, not retried, and never controlled: one such grade says
    nothing yet about the answer. ``attempt`` (review round 3): the control's own window is opened on its clock just
    before the control runs (``control_window``). Never raises but ``CancelledError``."""
    if not f.masks:
        return f
    ev: dict[str, Any] = {**(f.evidence or {}), "probe": probe, "control": "none", "status": f.detail or f.kind}
    flags = [*(ev.pop("flags", None) or []), *(["not_repeated"] if ev.get("tries") == 1 else [])]

    def marked(e: dict[str, Any], *more: str) -> dict[str, Any]:
        fl = [*flags, *more]
        return {**e, "flags": fl} if fl else e

    if not answer_sensitive(f) or probe == "dead":
        return f.model_copy(update={"evidence": marked(ev)})
    if not control:
        return f.model_copy(update={"retryable": False, "evidence": marked(ev, "no_control")})
    if ev.get("tries") == 1 and "budget_s" not in ev:
        # M10 lab audit R-3 review round 2 (item 2: a control decides only a failure that repeats on the same work): a single
        # answer-sensitive grade, after a reset connection or a contaminated node, or with grade_retries=0, went to the control,
        # and a control graded once the host recovered scored the policy 0 for an outage
        return f.model_copy(update={"retryable": False, "evidence": marked(ev)})
    if getattr(task, "grade_control", None) is not None:
        control_window(attempt, task, timeout_s)  # M10 lab audit R-3 review round 3: a decided failure always reaches its control
    got, info = await control_grade(task, workdir, timeout_s=timeout_s, budget_s=ev.get("budget_s"))
    why = f"{f.detail}: {f.reason}"
    if got == "passed":
        return Failure(kind="crucible:policy_error", stage="verification", blame="policy", mask=False, retryable=False, detail="grader_failed_on_answer",
                       reason=f"M10 lab audit R-3: the grader failed on this answer again, and graded the task's control cleanly: the failure "
                              f"is the answer's, scored 0 ({why})"[:500], evidence=marked({**ev, "control": "passed", **info}))
    if got == "failed":
        return Failure(kind="verifier_error", stage="verification", blame="task", retryable=False, detail=f.detail,
                       reason=f"M10 lab audit R-3: the grader failed on this answer again, and on the task's control too: the grader's "
                              f"fault, masked ({why})"[:500], evidence=marked({**ev, "control": "failed", **info}))
    return f.model_copy(update={"retryable": False, "evidence": marked(ev, "no_control")})
