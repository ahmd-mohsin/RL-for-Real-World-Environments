"""Reconcile an agent's ATIF trajectory with the calls the capture proxy recorded (M10 lab audit R-1, 2026-10-15): two
independent measurements of one rollout. The agent (Harbor's converter) counts its own tokens per step; the proxy records
what the upstream answered. Agreeing call by call, they corroborate each other; disagreeing, one of them is wrong and
nothing says which, so a training rollout built on them is not trainable (the Runner masks it; evaluation records it).

PORT (adapted): OpenEnv ``src/openenv/harbor/atif.py`` (``reconcile``, ``_reconcile_eval``, ``_subsequence_gap``,
``agent_steps``, ``atif_turn_lengths``), revision 55347f62e2, BSD-3-Clause, Copyright (c) 2026 Hugging Face, Inc.;
licence: ``LICENSES/BSD-3-Clause-openenv.txt``. Taken: the per-call completion-token comparison; auxiliary calls (calls
the agent made but did not log as steps, e.g. mini-swe-agent logs 5 of 6, atif.py:401) demoted only when ATIF covers at
least half of the captured calls (atif.py:393-416); synthetic ``<synthetic>`` API-error steps and the proxy's own
budget-stop steps excluded (atif.py:253-303); a call ATIF logged that the capture never saw, a per-call disagreement and
subagent trajectories are FATAL; ``llm_call_count > 1`` and a first-prompt disagreement are WARN. Changed:

- the input is Crucible's captured calls (``interop.agentenv.harness.calls_from_capture``: ``usage.completion_tokens``
  or Anthropic ``output_tokens``) instead of an OpenEnv rollout graph; ``aux_call_ids`` name captured ``call_id``s;
- one entry point for train and eval; the consequence is the Runner's (``runtime/run.py``: train masks
  ``crucible:capture_incomplete``, eval records ``meta.reconcile`` and a rollout flag);
- the counts-only comparison (OpenEnv's eval path) is taken whenever a kept call has no token count, and there a call
  ATIF logged that the capture lacks is FATAL (OpenEnv: WARN);
- agent steps with an explicit ``llm_call_count == 0`` are dropped before comparing: ATIF v1.7 defines them as
  deterministic dispatches, not model calls (harbor ``models/trajectories/step.py:74-85``), and Crucible's own ``to_atif``
  writes one per extra action of a call (OpenEnv counts every agent step);
- captured calls that failed upstream (an ``error`` and no completion-token count: a 5xx, a cut stream) are left out:
  they produced no reply for the agent to log (OpenEnv's graph holds sampled turns only);
- no captured call against logged agent steps is FATAL ``atif_calls_missing`` (OpenEnv defers to its own ``no_turns``
  check, which Crucible's eval path does not have: an agent that reached a model around the proxy must show);
- the prompt-length, retry and subagent checks run on every path (OpenEnv's aux and no-token-count paths returned before
  them, so an aux-call rollout with subagents passed);
- the proxy budget stop is Crucible's ``LIMIT_STOP_TEXT`` (``connectors/capture_proxy.py``);
- the ACP partial-usage path (``_partial_usage_matches_by_call_id``) is not ported.

Pure dict code: no Harbor import, so it runs (and is tested) without Harbor installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.connectors.capture_proxy import LIMIT_STOP_TEXT, client_left

Finding = dict  # {"level": one of LEVELS, "code": str, "detail": str}
LEVELS = ("ok", "info", "warn", "fatal")
COVERAGE_FLOOR = 0.5  # aux calls are inferred only when ATIF accounts for at least this share of the captured calls
_SHOW = 40  # values listed in a finding's detail


@dataclass
class ReconcileReport:
    findings: list[dict[str, Any]] = field(default_factory=list)
    aux_call_ids: list[str] = field(default_factory=list)  # captured calls ATIF does not log as agent steps: never trained

    def add(self, level: str, code: str, detail: str) -> None:
        if level not in LEVELS[1:]:
            raise ValueError(f"unknown reconcile level {level!r}")
        self.findings.append({"level": level, "code": code, "detail": detail})

    @property
    def level(self) -> str:
        """The highest level among the findings ("ok" when there are none)."""
        return max((f["level"] for f in self.findings), key=LEVELS.index, default="ok")

    @property
    def code(self) -> str | None:
        """The code of the first finding at the report's level."""
        lv = self.level
        return next((f["code"] for f in self.findings if f["level"] == lv), None)

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "code": self.code, "findings": [dict(f) for f in self.findings], "aux_call_ids": list(self.aux_call_ids)}


def _int(x: Any) -> int | None:
    return int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def agent_steps(atif: dict[str, Any]) -> list[dict[str, Any]]:
    """Steps the agent produced that represent model calls: ``source == "agent"``, without an explicit
    ``llm_call_count == 0`` (a deterministic dispatch; ``None`` is a producer that did not track it, counted as one)."""
    return [s for s in (atif.get("steps") or []) if isinstance(s, dict) and s.get("source") == "agent" and s.get("llm_call_count") != 0]


def _metrics(step: dict[str, Any]) -> dict[str, Any]:
    m = step.get("metrics")
    return m if isinstance(m, dict) else {}


def _tokens(step: dict[str, Any]) -> int:
    """A step's completion tokens; a missing count is 0 (OpenEnv ``atif_turn_lengths``)."""
    return _int(_metrics(step).get("completion_tokens")) or 0


def _usage(call: dict[str, Any]) -> dict[str, Any]:
    u = call.get("usage")
    return u if isinstance(u, dict) else {}


def _call_tokens(call: dict[str, Any]) -> int | None:
    """A captured call's completion tokens (OpenAI ``completion_tokens``, Anthropic and Responses ``output_tokens``), or None."""
    u = _usage(call)
    v = _int(u.get("completion_tokens"))
    return v if v is not None else _int(u.get("output_tokens"))


def _call_prompt(call: dict[str, Any]) -> int | None:
    u = _usage(call)
    v = _int(u.get("prompt_tokens"))
    if v is not None:
        return v
    parts = [_int(u.get(k)) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")]
    return sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None


def _subsequence_gap(needle: list[int], haystack: list[int]) -> list[int] | None:
    """Indices of ``haystack`` skipped when ``needle`` is matched as a subsequence, else None (OpenEnv, unchanged in
    logic). Greedy two-pointer, exact for subsequence membership; the needle must be strictly shorter: a longer needle
    means ATIF logged calls the capture never saw, which is a capture failure, not auxiliary calls."""
    if len(needle) >= len(haystack):
        return None
    skipped: list[int] = []
    j = 0
    for i, value in enumerate(haystack):
        if j < len(needle) and value == needle[j]:
            j += 1
        else:
            skipped.append(i)
    return skipped if j == len(needle) else None


def _show(xs: Sequence[Any]) -> str:
    return str(list(xs[:_SHOW])) + (f" …(+{len(xs) - _SHOW})" if len(xs) > _SHOW else "")


def _excluded(steps: list[dict[str, Any]], limit_stops: int, notes: ReconcileReport) -> list[dict[str, Any]]:
    """Drop the steps that are not model calls although logged as agent steps: synthetic API-error records and up to
    ``limit_stops`` copies of the proxy's limit-stop answer (only exact, zero-token, tool-less copies backed by the proxy's
    own count: a sampled copy of the text, or one too many, stays and fails the comparison). What was dropped goes to
    ``notes``."""
    out, api_errors, stops = [], 0, 0
    for s in steps:
        m, msg = _metrics(s), s.get("message")
        if (s.get("model_name") == "<synthetic>" and isinstance(msg, str) and msg.startswith("API Error: ") and not s.get("tool_calls")
                and m.get("completion_tokens") == 0 and m.get("prompt_tokens") == 0 and m.get("cached_tokens", 0) in (0, None)):
            api_errors += 1
            continue
        if (stops < limit_stops and isinstance(msg, str) and msg.strip() == LIMIT_STOP_TEXT and not s.get("tool_calls")
                and m.get("completion_tokens") in (None, 0) and m.get("prompt_tokens") in (None, 0)):
            stops += 1
            continue
        out.append(s)
    if api_errors:
        notes.add("warn", "atif_synthetic_api_error", f"excluded {api_errors} synthetic zero-token API error step(s); only model responses are compared")
    if stops:
        notes.add("info", "proxy_limit_stops", f"excluded {stops} zero-token proxy limit-stop step(s) of {limit_stops} the proxy answered")
    return out


def _compare(rep: ReconcileReport, theirs_steps: list[dict[str, Any]], kept: list[dict[str, Any]]) -> None:
    theirs = [_tokens(s) for s in theirs_steps]
    ours_raw = [_call_tokens(c) for c in kept]
    if not kept:
        if theirs_steps:
            rep.add("fatal", "atif_calls_missing", f"ATIF logged {len(theirs_steps)} agent step(s) but no model call reached the capture proxy")
        else:
            rep.add("info", "no_calls", "no model call was captured and ATIF logged no agent step: nothing to reconcile")
        return
    if any(t is None for t in ours_raw):
        # counts only (OpenEnv _reconcile_eval): a captured call without a token count gives nothing to compare per call
        only = " (counts only: a captured call carries no completion-token count)"
        if len(kept) < len(theirs):
            rep.add("fatal", "atif_calls_missing", f"ATIF logged {len(theirs)} agent step(s) but only {len(kept)} call(s) were captured{only}")
        elif len(kept) > len(theirs):
            rep.add("info", "atif_extra_calls", f"captured {len(kept)} call(s) against {len(theirs)} ATIF agent step(s); the extra ones are auxiliary{only}")
        else:
            rep.add("info", "counts_match", f"{len(kept)} captured call(s), {len(theirs)} ATIF agent step(s){only}")
        return
    ours: list[int] = [t for t in ours_raw if t is not None]
    if not any(theirs):
        # a converter that fills in no counts (or logs no agent step) gives nothing to compare against: unverified, not wrong
        what = f"{len(theirs)} agent step(s), all with 0 completion tokens" if theirs else "no agent steps"
        rep.add("warn", "atif_no_token_counts", f"ATIF has {what}; the {len(ours)} captured call(s) ({sum(ours)} tokens) stand unverified")
        return
    if ours == theirs:
        rep.add("info", "turns_match", f"all {len(ours)} calls agree token for token (total {sum(ours)})")
        return
    if len(ours) < len(theirs):
        rep.add("fatal", "atif_calls_missing", f"ATIF logged {len(theirs)} agent step(s) but only {len(ours)} call(s) were captured.\n"
                                               f"capture: {_show(ours)}\nATIF   : {_show(theirs)}")
        return
    aux = _subsequence_gap(theirs, ours)
    if aux is not None and len(theirs) < COVERAGE_FLOOR * len(ours):
        rep.add("fatal", "atif_coverage_too_low", f"ATIF accounts for only {len(theirs)} of {len(ours)} captured calls ({len(theirs) / len(ours):.0%}): "
                                                  f"they embed as a subsequence, but at this ratio that is as likely coincidence; refusing rather than "
                                                  f"discarding {len(aux)} calls on a guess.\ncapture: {_show(ours)}\nATIF   : {_show(theirs)}")
        return
    if aux is not None:
        rep.aux_call_ids = [str(kept[i].get("call_id")) for i in aux]
        rep.add("warn", "atif_aux_calls", f"{len(aux)} of {len(ours)} captured calls are absent from ATIF (auxiliary calls the agent did not log as "
                                          f"steps); the other {len(theirs)} agree token for token. Sizes: {_show([ours[i] for i in aux])}. Not trained.")
        return
    rep.add("fatal", "turn_mismatch", f"per-call completion tokens disagree.\ncapture: {_show(ours)}\nATIF   : {_show(theirs)}")


def reconcile(atif: dict[str, Any] | None, calls: Sequence[dict[str, Any]], *, limit_stops: int = 0) -> ReconcileReport:
    """Compare the agent's ATIF trajectory with the captured calls (``calls_from_capture`` rows, in arrival order).
    ``limit_stops``: the proxy's limit-stop answers this session (``UpstreamEvidence.limit_stops``)."""
    rep = ReconcileReport()
    if atif is None:
        rep.add("info", "no_atif", "the agent emitted no ATIF trajectory; cross-check unavailable")
        return rep
    # what was left out of the comparison is noted after its verdict, so the report's code (the first finding at its
    # level) is the comparison's whenever nothing outranks it
    notes = ReconcileReport()
    raw = [s for s in (atif.get("steps") or []) if isinstance(s, dict) and s.get("source") == "agent"]
    dispatch = sum(1 for s in raw if s.get("llm_call_count") == 0)
    if dispatch:
        notes.add("info", "atif_dispatch_steps", f"left out {dispatch} agent step(s) with llm_call_count 0 (deterministic dispatches, not model calls)")
    steps = _excluded(agent_steps(atif), max(0, int(limit_stops or 0)), notes)
    # M3 audit R-2: a call answered after the agent's client left is no step of the agent's (it never received it)
    kept = [c for c in calls if not (c.get("error") and _call_tokens(c) is None) and not client_left(c)]
    if len(kept) < len(calls):
        notes.add("info", "failed_calls", f"left out {len(calls) - len(kept)} captured call(s) that failed upstream with no completion or were "
                                          "answered after the agent's client left (no reply to log)")
    _compare(rep, steps, kept)
    rep.findings += notes.findings
    if steps and kept:
        a, b = _int(_metrics(steps[0]).get("prompt_tokens")), _call_prompt(kept[0])
        if a and b is not None and a != b:
            rep.add("warn", "prompt_len_mismatch", f"first-call prompt tokens: capture {b}, ATIF {a}")
    retried = sum(1 for s in steps if (_int(s.get("llm_call_count")) or 1) > 1)
    if retried:
        rep.add("warn", "atif_retries", f"{retried} ATIF step(s) report llm_call_count > 1: the agent retried (one step, several calls)")
    subs = atif.get("subagent_trajectories") or []
    if subs:
        # FATAL (OpenEnv atif.py:476-494): ATIF says subagents ran but not which captured calls are theirs; a rollout whose
        # reward cannot be attributed to the policy's own calls is not trainable
        rep.add("fatal", "atif_subagents", f"ATIF reports {len(subs)} subagent trajectory(ies); their calls cannot be told from the agent's own")
    return rep
