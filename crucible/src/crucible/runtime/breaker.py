"""Breakers: an outage pauses the run instead of masking its samples (M10 lab audit R-5, item 1).

Before R-5 a provider that returned 503s for ten minutes made every sample burn its three attempts, end terminal-masked, and
trip the health rule that stops the run (``Health``: mask fraction > 0.2): the run measured the outage, not the policy, and
what it had spent on the failing attempts was gone. Here the dependencies an attempt needs each have a breaker, and an
open breaker holds back the samples that need it:

- **Keys.** ``provider:<endpoint>`` (``PolicyRef.endpoint``, else the policy id: the model server the policy calls),
  ``host:<name>`` (the machine or daemon the task's sandbox and grade run on, ``host_name``) and ``task:<task>|<harness>``
  (a task's grader and environment as one harness runs them). ``counted`` names the keys a masked failure counts against by
  its kind: a provider kind counts against the provider key, a transport, session or sandbox kind against the host (and
  the task), anything else of infra, harness or task blame (a grader that fails its own control: ``Failure.evidence.control``)
  against the task. Policy-blamed and unknown-blamed failures, and ``crucible:credits_exhausted`` (a budget stop), count
  against nothing.
- **Open.** A provider key opens after ``failures`` consecutive failing attempts; a host key after as many spanning at least
  two tasks (one broken task is not a broken host); a task key after ``task_samples`` consecutive failing *samples* (a
  sample's own retries count once). A verdict (an attempt that was not masked: the dependencies answered, whatever the
  policy then did) ends the run of failures.
- **Defer.** While a key is open its owed samples wait before dispatch (``Breakers.admit``): they hold no concurrency slot,
  no attempt index, no ledger reservation. The Runner never makes terminal an infra-masked attempt whose key is open
  (``Runner._run``'s ``_attempt``): it is journaled ``deferred`` and does not use up one of the sample's retries
  (``RunJournal.tries``). A harness- or task-blamed failure counts toward opening a key but its own attempt ends as
  ``RetryRule`` says (retried within its budget, else terminal): deferring it would run the policy again for as long as the
  key stays open, until one run happens to finish (survivorship, DESIGN P23).
- **Probe.** After a cool-down (``cooldown_s``, doubling per failed probe, never shorter than the failure's Retry-After,
  never longer than a quarter of ``quarantine_s``) the key is half open: exactly one attempt of one waiting sample runs, and
  its verdict closes the key (a failure opens it again for twice as long). Only the claimed probe, or an attempt admitted
  after the key opened, can close it: a verdict of an attempt already in flight when it opened says nothing of the outage
  that began after it started.
- **Quarantine.** A probe that fails after the key has been open ``quarantine_s`` quarantines it: journaled with its reason,
  its samples are left owed (``quarantined`` events, ``RunSummary.quarantined``), and the rest of the run continues. A run
  never quarantines without a probe having failed: an outage that ends mid-way is found by the next probe.

State is in memory and starts closed on every ``Runner.run``: a resume probes again instead of trusting a verdict of an
earlier process. Every transition is a journal event (``waiting``, ``breaker_open``, ``breaker_probe``, ``breaker_closed``,
``quarantine``), additive to the journal.

PATTERN: NeMo-Gym ``responses_api_models/vllm_model/app.py:1948-2062`` (route around an endpoint that failed; a trial call
reinstates it) and NeMo-RL ``single_controller_utils/config.py:141-168`` (consecutive-drop budgets, which stop the run where
these pause it; Apache-2.0). Deferral and quarantine are Crucible's own. Imports ``crucible.core`` and the stdlib only.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.run import Failure, RunRecord

PROVIDER_KINDS = frozenset({"provider_unavailable", "provider_quota_exhausted", "provider_oom_killed"})
# kinds that name the machine, daemon or session an attempt runs on (its transport, its sandbox's container, the grader's own
# network): they count against the host and, because a task can break its own container, against the task too
HOST_KINDS = frozenset({"transport_unreachable", "transport_peer_drop", "transport_timeout", "transport_local_resource", "session_lost", "session_expired",
                        "crucible:backend_error", "crucible:env_container_lost", "crucible:node_contention", "crucible:grader_network_error",
                        "crucible:env_image_unavailable"})


@dataclass
class BreakerRule:
    """The thresholds (operational: ``identity.DENY``, how a run pauses and not what it measures). ``failures`` should not
    exceed ``RetryRule.max_attempts``: then no sample can use up its retries on a dependency before the dependency's breaker
    opens (the attempt that opens it is itself deferred). The Runner enforces it (M10 lab audit R-5 review round 1): it builds the
    breakers with ``failures`` clamped to ``max(1, RetryRule.max_attempts)``, so a ``RetryRule(max_attempts=1)`` run opens a key at
    the first failure instead of ending its first samples masked-terminal on a key that was still closed."""

    failures: int = 3  # consecutive failing attempts that open a provider key, or a host key across at least two tasks
    task_samples: int = 3  # consecutive failing samples that open a (task, harness) key: a sample's own retries count once
    cooldown_s: float = 30.0  # open -> half open; doubles after each failed probe, capped at a quarter of ``quarantine_s``
    quarantine_s: float = 900.0  # a key whose probe still fails after being open this long is quarantined
    retry_after_cap_s: float = 300.0  # the longest Retry-After the cool-down honours (the same cap ``RetryRule`` waits under)


@dataclass(frozen=True)
class SampleKeys:
    """The keys one attempt needs. ``provider`` is None for an attempt that calls no model (a grade-only retry, M10 lab audit R-3):
    it neither waits on, claims the probe of, nor closes the provider key, so a clean regrade is no evidence that the endpoint is
    back (M10 lab audit R-5 review round 1)."""

    provider: str | None
    host: str
    task: str

    def all(self) -> tuple[str, ...]:
        return (*((self.provider,) if self.provider else ()), self.host, self.task)

    def without_provider(self) -> SampleKeys:
        return SampleKeys(provider=None, host=self.host, task=self.task)


def host_name(task: Any, harness: Any) -> str:
    """The host a sample's sandbox and grade run on, for the host key: the first non-empty ``breaker_host`` or ``host`` of the
    task and then of the harness (a name, or an object with ``.name``: ``GpuHost``), else ``local``. An adapter that runs its
    sandboxes elsewhere declares it this way; the Runner knows nothing of Docker daemons."""
    for o in (task, harness):
        for attr in ("breaker_host", "host"):
            v = getattr(o, attr, None)
            name = v if isinstance(v, str) else getattr(v, "name", None)
            if isinstance(name, str) and name:
                return name
    return "local"


def sample_keys(sk: Any, task: Any, policy: Any, harness: Any) -> SampleKeys:
    """The breaker keys of one planned sample (``sk``: its ``SampleKey``)."""
    return SampleKeys(provider=f"provider:{getattr(policy, 'endpoint', None) or sk.policy}", host=f"host:{host_name(task, harness)}",
                      task=f"task:{sk.task}|{sk.harness}")


def counted(f: Failure | None, keys: SampleKeys) -> tuple[str, ...]:
    """The keys a failure counts against: none for a failure that is not masked, is blamed on the policy or on nobody known
    (unknown blame is a classifier gap, ``Health`` stops on it), or is a budget stop; the provider for a provider kind; the
    host and the task for a transport, session or sandbox kind; else the task (a grader that fails, a harness that cannot run
    it, a stale image: the kinds no retry can change land here too)."""
    if f is None or not f.masks or f.who not in ("infra", "harness", "task") or f.kind == "crucible:credits_exhausted":
        return ()
    if f.kind in PROVIDER_KINDS:
        return (keys.provider,) if keys.provider else ()  # an attempt that called no model counts nothing against the endpoint
    if f.kind in HOST_KINDS:
        return keys.host, keys.task
    return (keys.task,)


def _last(f: Failure) -> str:
    """What a key's quarantine reason says failed last: the kind, its detail, and the control's verdict when the task's own
    control was run (``Failure.evidence.control``: a grader that fails the control grades nothing)."""
    control = (f.evidence or {}).get("control")
    return f.kind + (f" ({f.detail})" if f.detail else "") + (f", control {control}" if control and control != "none" else "")


@dataclass
class _Key:
    name: str
    state: str = "closed"  # closed | open | half_open | quarantined
    streak: int = 0  # consecutive failing attempts while closed
    samples: set[str] = field(default_factory=set)  # the samples that failed in that run
    tasks: set[str] = field(default_factory=set)  # and the (task, harness) keys they belong to (a host key needs two)
    opened_at: float | None = None
    until: float = 0.0  # open: half open from here
    probes_failed: int = 0
    probing: str | None = None  # the sample whose attempt holds the half-open key's one probe
    last: str = ""
    deferred: set[str] = field(default_factory=set)  # samples that waited on it since it opened
    gen: int = 0  # how many times it has opened: an attempt admitted under an older one is stale
    reason: str = ""  # why it was quarantined


class Breakers:
    """The run's breakers (see the module doc). ``event`` is ``RunJournal.event``; ``clock`` is injectable for tests.

    Not thread-safe and not async-aware beyond ``admit``: ``check`` and ``record`` are synchronous and run on the event loop
    between awaits, so a decision and its bookkeeping are one step."""

    def __init__(self, rule: BreakerRule, event: Callable[..., None], clock: Callable[[], float] = time.monotonic):
        self.rule = rule
        self.event = event
        self._clock = clock
        self._keys: dict[str, _Key] = {}
        self._stamp: dict[str, dict[str, int]] = {}  # sample -> {key: generation} at its last admission
        self._changed = asyncio.Event()

    def _key(self, name: str) -> _Key:
        return self._keys.setdefault(name, _Key(name))

    def wake(self) -> None:
        """Every waiter re-checks: something changed (a transition, a released probe, a stop)."""
        self._changed.set()
        self._changed = asyncio.Event()

    def _gap(self, probes_failed: int, retry_after: float) -> float:
        """Seconds until the next probe: ``cooldown_s`` doubled per failed probe, never below the provider's Retry-After, and
        never above a quarter of ``quarantine_s`` (or ``cooldown_s`` itself when that is longer): an outage that ends before the
        quarantine is then found by a probe, and a long Retry-After cannot push the first probe past it."""
        r = self.rule
        return min(max(r.cooldown_s * 2 ** min(probes_failed, 30), retry_after), max(r.cooldown_s, r.quarantine_s / 4))

    # ------------------------------------------------------------------------------------------------------ admission

    def check(self, sid: str, keys: SampleKeys) -> tuple[str, float | None]:
        """May sample ``sid`` dispatch an attempt now? ``("go", None)``; ``("probe", None)``: yes, as the one probe of every
        half-open key it needs (claimed here); ``("wait", seconds | None)``: a key it needs is open (``seconds``: until it is
        half open) or is being probed by another sample (``None``: until something changes); ``("quarantined", None)``: a key it
        needs was quarantined. A passing check stamps the keys' generations, which ``record`` reads."""
        now = self._clock()
        claims: list[_Key] = []
        blocked = False
        waits: list[float] = []
        for k in keys.all():
            st = self._keys.get(k)
            if st is None or st.state == "closed":
                continue
            if st.state == "quarantined":
                return "quarantined", None
            if st.state == "open" and now >= st.until:
                st.state = "half_open"
            if st.state == "half_open" and st.probing in (None, sid):
                claims.append(st)
                continue
            blocked = True
            if sid not in st.deferred:
                st.deferred.add(sid)
                self.event("waiting", sample=sid, key=k)
            if st.state == "open":
                waits.append(max(st.until - now, 0.0))
        if blocked:
            return "wait", min(waits) if waits else None
        self._stamp[sid] = {k: (self._keys[k].gen if k in self._keys else 0) for k in keys.all()}
        for st in claims:
            if st.probing != sid:
                st.probing = sid
                self.event("breaker_probe", key=st.name, sample=sid)
        return ("probe" if claims else "go"), None

    async def admit(self, sid: str, keys: SampleKeys, *, stopped: Callable[[], bool], remaining: Callable[[], float | None] = lambda: None) -> str:
        """Wait, outside any concurrency slot, until ``check`` lets ``sid`` dispatch: "go", "probe", "quarantined", or
        "stopped" (the run's stop rules fired while it waited). The fast path (every key closed) never suspends, so a run
        without an outage dispatches in the order it did before R-5. ``remaining``: seconds to the run's wall clock (None: no
        limit), so a sample that waits wakes to see it expire."""
        while True:
            got, wait = self.check(sid, keys)
            if got != "wait":
                return got
            if stopped():
                return "stopped"
            left = remaining()
            if left is not None:
                wait = max(left, 0.0) if wait is None else min(wait, max(left, 0.0))
            ev = self._changed  # taken in the same step as the check: a change after it is not missed
            try:
                async with asyncio.timeout(wait):
                    await ev.wait()
            except TimeoutError:
                pass

    # ----------------------------------------------------------------------------------------------------- the verdicts

    def record(self, sid: str, keys: SampleKeys, rec: RunRecord) -> list[str]:
        """Account an attempt's record. Returns the keys it counted against that are not closed (open, half open or
        quarantined) after it: an infra failure that returns any is deferred by the Runner, never terminal."""
        now = self._clock()
        stamp = self._stamp.pop(sid, {})
        f = rec.failure
        changed = False
        if not rec.mask_sample:
            # a verdict: the dependencies answered. It ends a closed key's run of failures; it closes an open key only when it is
            # the claimed probe's, or its attempt was admitted after the key opened: a native episode whose model calls all
            # finished before the outage began answers nothing about it
            for k in keys.all():
                st = self._keys.get(k)
                if st is None:
                    continue
                if st.state == "closed":
                    st.streak, st.samples, st.tasks = 0, set(), set()
                elif st.state in ("open", "half_open") and (st.probing == sid or stamp.get(k, -1) >= st.gen):
                    self._close(st, sid, now)
                    changed = True
            if changed:
                self.wake()
            return []
        ours = counted(f, keys)
        ra = min(f.retry_after or 0.0, self.rule.retry_after_cap_s) if f is not None else 0.0
        for k in ours:
            st = self._key(k)
            st.last = _last(f)  # type: ignore[arg-type]
            if st.state == "closed":
                st.streak += 1
                st.samples.add(sid)
                st.tasks.add(keys.task)
                if self._reached(st):
                    self._open(st, now, ra, f)  # type: ignore[arg-type]
                    changed = True
            elif st.probing == sid:  # the probe failed: open again for longer, or quarantine
                st.probes_failed += 1
                st.probing = None
                changed = True
                self.event("breaker_probe", key=k, sample=sid, result="failed")
                if now - (st.opened_at if st.opened_at is not None else now) >= self.rule.quarantine_s:  # a clock reading of 0.0 is a time
                    self._quarantine(st, now)
                else:
                    st.state, st.until = "open", now + self._gap(st.probes_failed, ra)
            # else: a stale failure of an attempt admitted before the key opened: not evidence of a second outage
        for k in keys.all():  # a probe whose attempt failed for another dependency's reason says nothing of this one: the next sample probes
            st = self._keys.get(k)
            if st is not None and k not in ours and st.probing == sid:
                st.probing = None
                changed = True
        if changed:
            self.wake()
        return [k for k in ours if self._keys[k].state != "closed"]

    def release(self, sid: str) -> None:
        """``sid`` holds no probe any more (its attempt did not run to a record: a stop, or a check that failed in the slot)."""
        released = False
        for st in self._keys.values():
            if st.probing == sid:
                st.probing = None
                released = True
        if released:
            self.wake()

    def quarantined(self) -> dict[str, str]:
        return {k: st.reason for k, st in self._keys.items() if st.state == "quarantined"}

    # ------------------------------------------------------------------------------------------------------ transitions

    def _reached(self, st: _Key) -> bool:
        r = self.rule
        if st.name.startswith("provider:"):
            return st.streak >= r.failures
        if st.name.startswith("host:"):
            return st.streak >= r.failures and len(st.tasks) >= 2
        return len(st.samples) >= r.task_samples

    def _open(self, st: _Key, now: float, retry_after: float, f: Failure) -> None:
        st.gen += 1
        st.state, st.opened_at, st.probes_failed, st.probing = "open", now, 0, None
        st.until = now + self._gap(0, retry_after)
        self.event("breaker_open", key=st.name, kind=f.kind, detail=f.detail, control=(f.evidence or {}).get("control"), streak=st.streak,
                   samples=len(st.samples), until_s=round(st.until - now, 3))

    def _close(self, st: _Key, sid: str, now: float) -> None:
        self.event("breaker_closed", key=st.name, sample=sid, open_s=round(now - (st.opened_at or now), 3))
        st.state, st.opened_at, st.probes_failed, st.probing = "closed", None, 0, None
        st.streak, st.samples, st.tasks, st.deferred, st.last = 0, set(), set(), set(), ""

    def _quarantine(self, st: _Key, now: float) -> None:
        open_s = now - (st.opened_at or now)
        st.state = "quarantined"
        st.reason = (f"{st.last}: open {open_s:.0f}s (quarantine after {self.rule.quarantine_s:g}s), {st.probes_failed} failed probes, "
                     f"{len(st.deferred)} samples waited")
        self.event("quarantine", key=st.name, reason=st.reason, open_s=round(open_s, 3), probes_failed=st.probes_failed, deferred=len(st.deferred))
