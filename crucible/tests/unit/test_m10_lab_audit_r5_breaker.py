"""M10 lab audit R-5, item 1 (Acceptance 1 and 3): an outage pauses the run instead of masking its samples.

Breakers keyed by provider endpoint, host and (task, harness) (``runtime.breaker``): N consecutive failures blamed on infra,
harness or task open a key; its owed samples wait before dispatch (no slot, no index, no reservation); after a cool-down one
probe runs and its verdict closes the key; a key whose probe still fails after ``quarantine_s`` is quarantined (journaled, the
samples left owed with the reason) while the rest of the run continues; an infra-masked attempt whose key is open is never
terminal and never uses up a retry. The unit tests drive ``Breakers.check`` / ``record`` with a fake clock; the run-level tests
use small real durations (0.05-0.4 s) with generous margins.

Nothing here needs a network, Docker or a paid call.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from crucible.core.schema.run import Failure, RunRecord
from crucible.runtime import identity as ident
from crucible.runtime.breaker import BreakerRule, Breakers, SampleKeys, counted, host_name, sample_keys
from crucible.runtime.harness.base import HarnessResult
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.infra import RetryRule
from crucible.runtime.journal import RunJournal
from crucible.runtime.run import Runner, RunSpec

KEYS = SampleKeys(provider="provider:p", host="host:local", task="task:t|h@1")


@dataclass
class _T:
    id: str = "t@1"


@dataclass
class _P:
    id: str = "p"
    estimate_usd: float = 0.0


def _f(kind: str = "provider_unavailable", **kw) -> Failure:
    return Failure(kind=kind, stage=kw.pop("stage", "agent"), reason="x", **kw)


def _rec(sid: str, failure: Failure | None = None, reward: float = 1.0) -> RunRecord:
    rec = RunRecord(run_id="r", world="w", policy="p", started_at="2026-10-19T00:00:00Z", attempt=f"{sid}-a1", reward=reward, outcome="completed")
    if failure is not None:
        rec.mark_failure(failure)
    return rec


def _mk(rule: BreakerRule | None = None) -> tuple[Breakers, list, list]:
    """Breakers on a fake clock (``now[0]``) that record their journal events in ``ev``."""
    ev: list = []
    now = [0.0]
    return Breakers(rule or BreakerRule(), lambda t, **d: ev.append((t, d)), clock=lambda: now[0]), ev, now


def _fail(b: Breakers, sid: str, keys: SampleKeys = KEYS, kind: str = "provider_unavailable", **kw) -> list[str]:
    return b.record(sid, keys, _rec(sid, _f(kind, **kw)))


def _open(b: Breakers, **kw) -> None:
    for sid in ("s1", "s2", "s3"):
        _fail(b, sid, **kw)


def _events(run_dir) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "journal.jsonl").read_text().splitlines() if x.strip()]


def _rows(run_dir, name: str) -> list[dict]:
    p = run_dir / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _outage(up_at: float):
    """A stateless harness whose provider is down until ``time.monotonic()`` passes ``up_at``."""

    class Outage:
        id = "outage@1"

        async def run(self, attempt):
            await asyncio.sleep(0.01)
            if time.monotonic() < up_at:
                return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="HTTP 503"))
            return HarnessResult(reward=1.0, steps=1)

    return Outage()


# ------------------------------------------------------------------------------------------------------------ keys and counting


def test_counted_names_the_failing_dependency():
    assert counted(_f("provider_unavailable"), KEYS) == (KEYS.provider,)
    assert counted(_f("provider_quota_exhausted"), KEYS) == (KEYS.provider,)
    assert counted(_f("crucible:env_container_lost"), KEYS) == (KEYS.host, KEYS.task)  # a sandbox kind: the host, and the task that broke it
    assert counted(_f("transport_unreachable"), KEYS) == (KEYS.host, KEYS.task)
    assert counted(_f("verifier_error", stage="verification", blame="task"), KEYS) == (KEYS.task,)
    assert counted(_f("crucible:stale_image", stage="verification"), KEYS) == (KEYS.task,)  # a kind no retry can change lands on the task
    assert counted(_f("agent_run_error", blame="harness"), KEYS) == (KEYS.task,)
    assert counted(_f("agent_run_error", blame="unknown"), KEYS) == ()  # a classifier gap: Health stops on it, no breaker counts it
    assert counted(_f("crucible:policy_error", mask=False), KEYS) == ()  # the policy's own outcome
    assert counted(_f("crucible:credits_exhausted"), KEYS) == ()  # a budget stop, not an outage
    assert counted(None, KEYS) == ()


def test_sample_keys_follow_the_endpoint_and_the_declared_host():
    sk = SimpleNamespace(policy="p", task="t", harness="h@1")
    assert sample_keys(sk, _T(), _P(), SimpleNamespace()) == SampleKeys(provider="provider:p", host="host:local", task="task:t|h@1")
    assert sample_keys(sk, _T(), SimpleNamespace(endpoint="https://m/v1#x"), SimpleNamespace()).provider == "provider:https://m/v1#x"  # PolicyRef.endpoint
    assert host_name(SimpleNamespace(host=SimpleNamespace(name="gpu1")), SimpleNamespace()) == "gpu1"  # a GpuHost-like object
    assert host_name(SimpleNamespace(host="gpu2"), SimpleNamespace(breaker_host="b")) == "gpu2"  # the task before the harness
    assert host_name(_T(), SimpleNamespace(breaker_host="b", host="h")) == "b"  # an explicit declaration before the plain attribute
    assert host_name(SimpleNamespace(host=None), SimpleNamespace(host="")) == "local"
    assert PolicyRef(id="p", make=lambda a: {}, config={}, endpoint="e").endpoint == "e"
    assert "endpoint" not in PolicyRef(id="p", make=lambda a: {}, config={}, endpoint="e").identity_config()  # placement, not what is measured


# ----------------------------------------------------------------------------------------------------------------- opening


def test_a_provider_key_opens_after_n_failing_attempts_and_a_verdict_resets_it():
    b, ev, now = _mk(BreakerRule(failures=3))
    assert _fail(b, "s1") == [] and _fail(b, "s1") == []
    assert b.record("s2", KEYS, _rec("s2")) == []  # a verdict: the provider answered
    assert _fail(b, "s3") == [] and _fail(b, "s3") == []
    assert b.check("s9", KEYS) == ("go", None)  # two failures since the verdict: still closed
    assert _fail(b, "s3") == ["provider:p"]  # the third opens it, and the attempt that opened it is reported as open
    assert b.check("s3", KEYS) == ("wait", 30.0)
    assert any(t == "breaker_open" and d["key"] == "provider:p" and d["kind"] == "provider_unavailable" for t, d in ev)


def test_a_host_key_needs_failures_across_two_tasks():
    b, ev, now = _mk(BreakerRule(failures=3))
    a = SampleKeys(provider="provider:p", host="host:local", task="task:a|h@1")
    c = SampleKeys(provider="provider:p", host="host:local", task="task:b|h@1")
    for _ in range(3):
        assert _fail(b, "s1", a, "crucible:env_container_lost") == []  # one sample, one task: its own container, not the host
    assert b.check("x", a) == ("go", None) and b.check("x", c) == ("go", None)
    assert _fail(b, "s4", c, "crucible:env_container_lost") == ["host:local"]  # a second task fails the same way: the host
    assert b.check("s5", c)[0] == "wait" and b.check("s5", a)[0] == "wait"
    assert [d["key"] for t, d in ev if t == "breaker_open"] == ["host:local"]


def test_a_task_key_counts_samples_not_retries():
    b, ev, now = _mk(BreakerRule(task_samples=2))
    for _ in range(3):
        assert _fail(b, "s1", kind="verifier_error", stage="verification", blame="task") == []  # one sample's retries count once
    assert b.check("x", KEYS) == ("go", None)
    assert _fail(b, "s2", kind="verifier_error", stage="verification", blame="task") == ["task:t|h@1"]


# ---------------------------------------------------------------------------------------------------- probe, cool-down, close


def test_half_open_admits_one_probe_and_its_verdict_closes_the_key():
    b, ev, now = _mk()
    _open(b)
    now[0] = 31
    assert b.check("s4", KEYS) == ("probe", None)  # the cool-down is over: this sample's attempt is the one probe
    assert b.check("s5", KEYS)[0] == "wait"  # the others wait for its verdict
    assert b.check("s4", KEYS) == ("probe", None)  # the same sample asking again (its slot check) is still the prober
    assert [d["sample"] for t, d in ev if t == "breaker_probe"] == ["s4"]
    assert b.record("s4", KEYS, _rec("s4")) == []
    assert b.check("s5", KEYS) == ("go", None)
    assert any(t == "breaker_closed" and d["key"] == "provider:p" and d["sample"] == "s4" for t, d in ev)


def test_a_failed_probe_opens_the_key_again_for_twice_as_long():
    b, ev, now = _mk()
    _open(b)
    now[0] = 31
    assert b.check("s4", KEYS) == ("probe", None)
    assert _fail(b, "s4") == ["provider:p"]  # the probe's own attempt is deferred by the Runner: its key is open
    st = b._keys["provider:p"]
    assert st.state == "open" and st.until == 31 + 60 and st.probing is None
    assert any(t == "breaker_probe" and d.get("result") == "failed" for t, d in ev)
    now[0] = 91
    assert b.check("s5", KEYS) == ("probe", None)  # and another sample probes after it


def test_a_probe_that_failed_for_another_dependency_hands_the_probe_on():
    b, ev, now = _mk()
    _open(b)
    now[0] = 31
    assert b.check("s4", KEYS) == ("probe", None)
    assert _fail(b, "s4", kind="verifier_error", stage="verification", blame="task") == []  # the task's grader, not the provider
    assert b._keys["provider:p"].probing is None and b._keys["provider:p"].state == "half_open"
    assert b.check("s5", KEYS) == ("probe", None)


def test_the_cooldown_honours_retry_after_up_to_a_quarter_of_the_quarantine():
    b, ev, now = _mk()
    _open(b, retry_after=120.0)
    assert b.check("x", KEYS) == ("wait", 120.0)  # never shorter than the provider asked
    b, ev, now = _mk()
    _open(b, retry_after=10_000.0)
    assert b.check("x", KEYS) == ("wait", 225.0)  # capped (BreakerRule.retry_after_cap_s, then a quarter of quarantine_s): a probe is never pushed past it


def test_an_in_flight_verdict_from_before_the_outage_does_not_close_the_key():
    b, ev, now = _mk()
    assert b.check("late", KEYS) == ("go", None)  # admitted while the provider was up, still running
    _open(b)
    assert b.record("late", KEYS, _rec("late")) == []  # its model calls finished before the outage began
    assert b.check("x", KEYS) == ("wait", 30.0)  # the key is still open
    now[0] = 31
    assert b.check("p", KEYS) == ("probe", None)
    assert b.record("p", KEYS, _rec("p")) == []
    assert b.check("x", KEYS) == ("go", None)  # only the probe's verdict closed it


# --------------------------------------------------------------------------------------------------------------- quarantine


def test_a_key_still_failing_after_quarantine_s_is_quarantined_once():
    b, ev, now = _mk(BreakerRule(quarantine_s=100))
    _open(b)
    now[0] = 100.5  # open longer than T: the next probe decides
    assert b.check("s4", KEYS) == ("probe", None)
    assert _fail(b, "s4") == ["provider:p"]
    assert b.check("s5", KEYS) == ("quarantined", None) and b.check("s6", KEYS) == ("quarantined", None)
    assert [d["key"] for t, d in ev if t == "quarantine"] == ["provider:p"]  # exactly once
    assert "provider_unavailable" in b.quarantined()["provider:p"]
    assert b.record("s7", KEYS, _rec("s7")) == []  # a verdict does not lift a quarantine
    assert b.check("s8", KEYS) == ("quarantined", None)


def test_an_outage_that_ends_before_the_quarantine_is_found_by_a_probe_never_quarantined():
    b, ev, now = _mk(BreakerRule(cooldown_s=10, quarantine_s=100))
    _open(b)  # the provider is down until t = 60 (0.6 of T)
    for t, until in ((10, 30), (30, 55), (55, 80)):
        now[0] = t
        assert b.check(f"p{t}", KEYS) == ("probe", None)
        assert _fail(b, f"p{t}") == ["provider:p"]  # still down: the gap doubles but never passes a quarter of T
        assert b._keys["provider:p"].until == until
    now[0] = 80
    assert b.check("p80", KEYS) == ("probe", None)  # the provider is back
    assert b.record("p80", KEYS, _rec("p80")) == []
    assert b.quarantined() == {} and b.check("x", KEYS) == ("go", None)
    assert not any(t == "quarantine" for t, _ in ev)


# ---------------------------------------------------------------------------------------------------------------- admission


async def test_admit_returns_at_once_when_closed_and_wakes_on_a_verdict():
    ev: list = []
    b = Breakers(BreakerRule(cooldown_s=0.1, quarantine_s=30.0), lambda t, **d: ev.append((t, d)))
    assert await asyncio.wait_for(b.admit("s", KEYS, stopped=lambda: False), 1) == "go"  # the fast path: nothing suspends
    _open(b)
    waiters = {asyncio.ensure_future(b.admit(sid, KEYS, stopped=lambda: False)): sid for sid in ("s9", "s10")}
    await asyncio.sleep(0.02)
    assert not any(t.done() for t in waiters)  # the key is open: they wait, holding nothing
    done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED, timeout=2)
    assert len(done) == 1 and len(pending) == 1  # after the cool-down exactly one sample is the probe
    prober = next(iter(done))
    assert prober.result() == "probe"
    await asyncio.sleep(0.02)
    assert not next(iter(pending)).done()  # the other still waits for the probe's verdict
    b.record(waiters[prober], KEYS, _rec(waiters[prober]))  # a verdict closes the key and wakes it
    assert await asyncio.wait_for(next(iter(pending)), 1) == "go"
    assert any(t == "waiting" for t, _ in ev)


async def test_admit_stops_at_the_wall_clock_while_every_key_it_needs_is_open():
    b = Breakers(BreakerRule(cooldown_s=60.0), lambda t, **d: None)
    _open(b)
    deadline = time.monotonic() + 0.2
    t0 = time.monotonic()
    got = await asyncio.wait_for(b.admit("s", KEYS, stopped=lambda: time.monotonic() >= deadline, remaining=lambda: deadline - time.monotonic()), 5)
    assert got == "stopped" and time.monotonic() - t0 < 2.0  # woken by the wall clock, not by the 60 s cool-down


# --------------------------------------------------------------------------------------------------------------------- runs


async def test_a_provider_outage_defers_samples_and_the_run_completes_with_unchanged_thresholds(tmp_path):
    """Acceptance 1: a provider that is down for 0.4 s. Before R-5 the third failing attempt of each sample ended terminal-masked and
    the health rule stopped the run; here the samples wait for a probe, none ends masked, and the run measures every sample."""
    spec = RunSpec(run_id="outage", tasks=[_T()], policies=[_P()], harnesses=[_outage(time.monotonic() + 0.4)], k=6, concurrency=2, retry_backoff_s=0.0,
                   min_terminal_for_health=1, breaker=BreakerRule(failures=3, cooldown_s=0.1, quarantine_s=30.0))
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 60)
    assert s.terminal == 6 and s.scored == 6 and s.masked_terminal == 0 and s.stopped is None and s.quarantined == {}, s
    ev = _events(tmp_path / "outage")
    assert any(e["event"] == "breaker_open" and e["key"] == "provider:p" for e in ev)
    assert any(e["event"] == "breaker_closed" and e["key"] == "provider:p" for e in ev)
    assert any(e["event"] == "waiting" for e in ev)  # a sample held back at admission
    assert any(e["event"] == "deferred" and e["keys"] == ["provider:p"] for e in ev)  # an attempt made non-terminal
    assert not any(r["mask_sample"] for r in _rows(tmp_path / "outage", "results.jsonl"))
    deferred = [r for r in _rows(tmp_path / "outage", "failures.jsonl") if "breaker" in (r.get("meta") or {})]
    assert deferred and all(r["failure"]["kind"] == "provider_unavailable" for r in deferred)
    # a deferred attempt uses up none of the sample's retries: every sample's record shows at most max_attempts - 1 non-deferred failures
    for sdir in (tmp_path / "outage" / "samples").iterdir():
        recs = [json.loads(p.read_text()) for p in sdir.glob("a*/record.json")]
        assert sum(1 for r in recs if r["mask_sample"] and "breaker" not in (r.get("meta") or {})) <= 2


async def test_a_task_whose_grader_always_fails_is_quarantined_and_the_rest_completes(tmp_path):
    """Acceptance 3: one task's grader fails its own control on every sample. Its (task, harness) key opens after two samples, is
    probed and, still failing past ``quarantine_s``, quarantined with the control's verdict in its reason; its remaining samples
    are left owed, the other task's samples all finish and the run is not stopped."""

    class Grader:
        id = "grader@1"

        async def run(self, a):
            if a.task.id == "broken":
                return HarnessResult(failure=Failure(kind="verifier_error", stage="verification", blame="task", retryable=False, detail="verifier_timeout",
                                                     evidence={"probe": "unprobed", "control": "failed", "status": "verifier_timeout"}))
            return HarnessResult(reward=1.0, steps=1)

    spec = RunSpec(run_id="q", tasks=[_T(id="broken"), _T(id="good")], policies=[_P()], harnesses=[Grader()], k=8, concurrency=2, retry_backoff_s=0.0,
                   breaker=BreakerRule(task_samples=2, cooldown_s=0.2, quarantine_s=0.3))  # k=8: more samples than probes before the quarantine
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 60)
    assert list(s.quarantined) == ["task:broken|grader@1"] and "control failed" in s.quarantined["task:broken|grader@1"], s.quarantined
    assert s.stopped is None
    rows = _rows(tmp_path / "q", "results.jsonl")
    good = [r for r in rows if r["task"] == "good"]
    assert len(good) == 8 and all(r["reward"] == 1.0 for r in good)  # the rest of the run continued
    assert s.quarantined_samples >= 1 and s.terminal == 16 - s.quarantined_samples
    ev = _events(tmp_path / "q")
    assert sum(1 for e in ev if e["event"] == "quarantine") == 1
    assert sum(1 for e in ev if e["event"] == "quarantined") == s.quarantined_samples
    assert all(e["reasons"] and "task:broken|grader@1" in e["reasons"] for e in ev if e["event"] == "quarantined")
    assert [e["quarantined"] for e in ev if e["event"] == "summary"][-1] == s.quarantined
    open_ev = next(e for e in ev if e["event"] == "breaker_open")
    assert open_ev["key"] == "task:broken|grader@1" and open_ev["control"] == "failed"  # the grader's own control is in the event
    from crucible.bench.metrics import run_info

    assert run_info(tmp_path / "q")["quarantined"] == s.quarantined


async def test_a_run_whose_every_sample_waits_on_an_open_key_still_stops_at_its_wall_clock(tmp_path):
    spec = RunSpec(run_id="wall", tasks=[_T()], policies=[_P()], harnesses=[_outage(math.inf)], k=6, concurrency=2, retry_backoff_s=0.0, wall_clock_s=0.3,
                   breaker=BreakerRule(failures=3, cooldown_s=60.0, quarantine_s=900.0))
    t0 = time.monotonic()
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 30)
    assert s.stopped == "wall_clock" and time.monotonic() - t0 < 10  # not the 60 s cool-down
    assert s.scored == 0 and s.quarantined == {}


async def test_a_harness_blamed_failure_follows_the_retry_rule_even_while_its_key_is_open(tmp_path):
    """P23: only infra-blamed attempts are deferred. A harness-blamed one counts toward opening the (task, harness) key, but its own
    attempt ends as RetryRule says, so the policy is never re-run without limit for as long as the key stays open."""

    class Flaky:
        id = "flaky@1"

        async def run(self, a):
            return HarnessResult(failure=Failure(kind="agent_run_error", stage="agent", blame="harness", reason="harness bug"))

    spec = RunSpec(run_id="p23", tasks=[_T()], policies=[_P()], harnesses=[Flaky()], k=6, concurrency=2, retry_backoff_s=0.0, max_mask_fraction=1.0,
                   retry=RetryRule(max_attempts=2), breaker=BreakerRule(task_samples=2, cooldown_s=0.05, quarantine_s=0.2))
    await asyncio.wait_for(Runner(tmp_path).run(spec), 60)
    for sdir in (tmp_path / "p23" / "samples").iterdir():
        recs = [json.loads(p.read_text()) for p in sdir.glob("a*/record.json")]
        assert len(recs) <= 2, sdir  # max_attempts per sample, the probes included
        assert all("breaker" not in (r.get("meta") or {}) for r in recs)  # none was deferred
    assert not any(e["event"] == "deferred" for e in _events(tmp_path / "p23"))
    assert any(e["event"] == "breaker_open" for e in _events(tmp_path / "p23"))  # it still counted toward the key


async def test_breaker_none_keeps_the_pre_r5_behaviour(tmp_path):
    spec = RunSpec(run_id="off", tasks=[_T()], policies=[_P()], harnesses=[_outage(math.inf)], k=2, concurrency=2, retry_backoff_s=0.0, max_mask_fraction=1.0,
                   retry=RetryRule(max_attempts=3), breaker=None)
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 60)
    assert s.masked_terminal == 2 and s.attempts == 6 and s.quarantined == {} and s.quarantined_samples == 0
    assert not any(e["event"] in ("breaker_open", "deferred", "waiting") for e in _events(tmp_path / "off"))


# -------------------------------------------------------------------------------------------------- identity and the journal


def test_breaker_settings_are_operational():
    def spec(**kw) -> RunSpec:
        return RunSpec(run_id="r", tasks=[SimpleNamespace(id="t")], policies=[PolicyRef(id="p", make=lambda a: {}, config={})],
                       harnesses=[NativeHarness()], k=1, **kw)

    a = ident.fingerprint(spec().resolved_doc())
    assert a == ident.fingerprint(spec(breaker=BreakerRule(cooldown_s=1, failures=9)).resolved_doc())  # how a run pauses is not what it measures
    assert a == ident.fingerprint(spec(breaker=None).resolved_doc())  # and a run from before R-5 resumes as it was
    assert spec().resolved_doc()["breaker"]["failures"] == 3  # recorded, though not compared


def test_journal_tries_skip_refused_and_deferred_attempts(tmp_path):
    with RunJournal(tmp_path, "r", "id", {}) as j:
        a1 = j.attempt_dir("sid", 1)
        a1.mkdir(parents=True)
        (a1 / "record.json").write_text(json.dumps({"meta": {"breaker": {"deferred": ["provider:p"]}}}))  # deferred by an open breaker
        j.refuse(j.attempt_dir("sid", 2), "sid-a2", "budget")  # refused: nothing was dispatched
        a3 = j.attempt_dir("sid", 3)
        a3.mkdir(parents=True)
        (a3 / "record.json").write_text(json.dumps({"meta": {}}))  # an ordinary failed attempt
        j.attempt_dir("sid", 4).mkdir(parents=True)  # the attempt being decided: no record yet
        assert j.tries("sid", 4) == 2
        assert j.tries("sid", 4, since=3) == 1
        assert j.tries("sid", 1) == 0
        assert j.dispatched("sid", 4) == 3  # unchanged: refused attempts only


# ------------------------------------------------------------------------------------------------------------ review round 1


def test_two_declared_hosts_get_two_host_keys():
    sk = SimpleNamespace(policy="p", task="t", harness="h@1")
    a = sample_keys(sk, SimpleNamespace(id="t", breaker_host="lab-a"), _P(), SimpleNamespace())
    b = sample_keys(sk, SimpleNamespace(id="t", breaker_host="lab-b"), _P(), SimpleNamespace())
    assert (a.host, b.host, sample_keys(sk, _T(), _P(), SimpleNamespace()).host) == ("host:lab-a", "host:lab-b", "host:local")


def test_harbor_tasks_key_the_host_they_run_on():
    """Without ``breaker_host`` every Harbor task keyed ``host:local``: one lab host's outage paused the samples of every other, and a
    local verdict closed its key while the lab was still down. Built without ``__init__`` (it needs a Harbor task directory)."""
    pytest.importorskip("harbor")
    from crucible.interop.harbor.inplace import HarborInPlaceTask
    from crucible.interop.harbor.patch_task import HarborPatchTask
    from crucible.runtime.gpu import GpuHost

    def inplace(**bp) -> HarborInPlaceTask:
        t = object.__new__(HarborInPlaceTask)
        t.backend_params = bp
        return t

    sk = SimpleNamespace(policy="p", task="t", harness="h@1")
    hosts = [sample_keys(sk, t, _P(), SimpleNamespace()).host for t in (inplace(), inplace(docker_host="ssh://lab"), inplace(docker_host="ssh://x", gpu_host="rtx"))]
    assert hosts == ["host:local", "host:ssh://lab", "host:rtx"]  # a GPU host's name before the daemon it derives
    patch = object.__new__(HarborPatchTask)
    patch.backend_params = {}
    patch.grader = SimpleNamespace(host=GpuHost(name="local"))  # HarborDirTask's default host names no machine
    assert host_name(patch, SimpleNamespace()) == "local"
    patch.grader = SimpleNamespace(host=GpuHost(name="lab5", ssh="u@lab5", docker_host="ssh://u@lab5"))
    assert host_name(patch, SimpleNamespace()) == "lab5"  # the grading container's host
    patch.backend_params = {"gpu_host": "rtx"}
    assert host_name(patch, SimpleNamespace()) == "rtx"  # the agent's World host first


def test_a_grade_only_attempt_neither_probes_nor_closes_the_provider_key():
    b, ev, now = _mk()
    _open(b)
    now[0] = 31
    g = KEYS.without_provider()
    assert g.all() == ("host:local", "task:t|h@1")
    assert b.check("g", g) == ("go", None)  # the provider key is not one a regrade needs: it neither waits on it nor claims its probe
    assert b.record("g", g, _rec("g")) == []  # a clean regrade says nothing of an endpoint that is still down
    assert not any(t == "breaker_closed" for t, _ in ev)
    assert b.check("s", KEYS) == ("probe", None)  # the endpoint still needs its own probe
    assert counted(_f("provider_unavailable"), g) == ()  # and an attempt that called no model counts nothing against it


async def test_a_grade_only_retry_never_waits_on_an_open_provider_key(tmp_path):
    """Provider outage with a half-finished sample: ``g``'s first attempt is a masked grade with its work persisted (R-3), so its retry
    is a regrade that calls no model. It must dispatch while the provider key is open (before: it queued behind it, and a clean
    regrade could claim the probe and close the key while the provider was still down)."""

    class H:
        id = "go@1"

        async def run(self, a):
            if a.task.id == "down":
                await asyncio.sleep(0.02)
                return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="503"))
            if a.n == 1:
                await asyncio.sleep(0.3)  # the provider key opens meanwhile
                return HarnessResult(failure=Failure(kind="crucible:node_contention", stage="verification", reason="contaminated"), regradable=True)
            return HarnessResult(reward=1.0, steps=1)

    spec = RunSpec(run_id="go", tasks=[_T(id="g"), _T(id="down")], policies=[_P()], harnesses=[H()], k=2, concurrency=4, retry_backoff_s=0.0, wall_clock_s=1.5,
                   breaker=BreakerRule(failures=2, cooldown_s=60.0, quarantine_s=900.0))
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 30)
    rows = _rows(tmp_path / "go", "results.jsonl")
    assert sorted(r["task"] for r in rows if not r["mask_sample"]) == ["g", "g"] and all(r["reward"] == 1.0 for r in rows if r["task"] == "g")
    assert s.stopped == "wall_clock" and s.scored == 2  # the provider stays down: its samples wait, and are left owed
    ev = _events(tmp_path / "go")
    assert any(e["event"] == "breaker_open" and e["key"] == "provider:p" for e in ev)
    assert not any(e["event"] == "breaker_closed" for e in ev)  # nothing but a call to the model can close it
    assert sum(1 for e in ev if e["event"] == "dispatched" and e.get("grade_only")) == 2


async def test_a_broken_grader_never_stops_a_campaign_through_the_run_global_health(tmp_path):
    """Review round 1: a broken task's masks (a grader that fails its control is task-blamed and never deferred) were fed to the
    run-global Health: the sample that opened its key and every failed probe after it. At the default thresholds (mask fraction 0.2
    after 20 terminal samples) broken graders stopped the campaign. Three broken tasks come first (6 masks at once, half of them the
    samples that opened a key) and the good samples are slower than the cool-down, so ~20 terminal rows exist while the keys are
    still open: without the fix 6 of 20 are masks and the run stops; with it 3 are (the first sample of each task, before its key)."""

    class Grader:
        id = "grader@1"

        async def run(self, a):
            if a.task.id.startswith("broken"):
                return HarnessResult(failure=Failure(kind="verifier_error", stage="verification", blame="task", retryable=False, detail="verifier_timeout",
                                                     evidence={"probe": "unprobed", "control": "failed", "status": "verifier_timeout"}))
            await asyncio.sleep(0.05)
            return HarnessResult(reward=1.0, steps=1)

    spec = RunSpec(run_id="health", tasks=[*[_T(id=f"broken{i}") for i in range(3)], *[_T(id=f"g{i}") for i in range(6)]], policies=[_P()], harnesses=[Grader()],
                   k=5, concurrency=4, retry_backoff_s=0.0,
                   breaker=BreakerRule(task_samples=2, cooldown_s=0.1, quarantine_s=0.3))  # the health thresholds stay at their defaults (20, 0.2)
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 60)
    assert s.stopped is None, s.stopped
    rows = _rows(tmp_path / "health", "results.jsonl")
    assert sum(1 for r in rows if not r["task"].startswith("broken") and r["reward"] == 1.0) == 30  # every good sample is scored
    assert all(k.startswith("task:broken") for k in s.quarantined), s.quarantined  # only a broken task's key can be quarantined
    assert s.breaker_masked >= 3  # the sample that opened each key: not the stop rules' to count
    assert s.terminal + s.quarantined_samples == 45
    assert [e["breaker_masked"] for e in _events(tmp_path / "health") if e["event"] == "summary"][-1] == s.breaker_masked


@pytest.mark.parametrize("kind,stage,retryable", [("crucible:grader_network_error", "verification", True), ("crucible:stale_image", "verification", True),
                                                  ("crucible:stale_image", "agent", True), ("crucible:env_container_lost", "agent", False)])
async def test_an_attempt_no_retry_could_change_is_never_deferred_while_its_key_is_open(tmp_path, kind, stage, retryable):
    """The deferral guard ``RetryRule.allows(f, 0, regrade=...)``: an infra grade failure with no persisted work (not regradable), a kind no
    retry fixes and a non-retryable failure end as the rule says, even on an open key: deferring them would run the policy again after
    a grade (or a setup) failed, for as long as the key stays open (R-3, P23)."""

    class Fails:
        id = "fails@1"

        async def run(self, a):
            return HarnessResult(failure=Failure(kind=kind, stage=stage, reason="x", retryable=retryable))

    spec = RunSpec(run_id="nd", tasks=[_T()], policies=[_P()], harnesses=[Fails()], k=2, concurrency=2, retry_backoff_s=0.0, max_mask_fraction=1.0,
                   breaker=BreakerRule(task_samples=2, cooldown_s=0.05, quarantine_s=0.2))
    await asyncio.wait_for(Runner(tmp_path).run(spec), 30)
    ev = _events(tmp_path / "nd")
    assert any(e["event"] == "breaker_open" for e in ev)  # the second sample opened the (task, harness) key...
    assert not any(e["event"] == "deferred" for e in ev)  # ...and was not deferred
    for sdir in (tmp_path / "nd" / "samples").iterdir():
        recs = [json.loads(p.read_text()) for p in sdir.glob("a*/record.json")]
        assert len(recs) == 1 and recs[0]["mask_sample"] and "breaker" not in (recs[0].get("meta") or {}), sdir  # one attempt, terminal-masked
    assert sum(1 for r in _rows(tmp_path / "nd", "results.jsonl") if r["mask_sample"]) == 2


async def test_deferred_attempts_use_none_of_the_samples_retries(tmp_path):
    """The Runner counts ``RunJournal.tries``, not ``dispatched``: a1..a3 are deferred by the open key (the outage), a4 is the probe, and
    its harness-blamed flake is the sample's first real failure; RetryRule(max_attempts=2) still grants the retry (a5, scored). Counting
    dispatched attempts (4 >= 2) would end the sample terminal-masked, the outage burning its retries."""

    class H:
        id = "seq@1"

        async def run(self, a):
            if a.n <= 3:
                return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="HTTP 503"))
            if a.n == 4:
                return HarnessResult(failure=Failure(kind="agent_run_error", stage="agent", blame="harness", reason="flake"))
            return HarnessResult(reward=1.0, steps=1)

    spec = RunSpec(run_id="tries", tasks=[_T()], policies=[_P()], harnesses=[H()], k=1, concurrency=1, retry_backoff_s=0.0, retry=RetryRule(max_attempts=2),
                   breaker=BreakerRule(failures=1, cooldown_s=0.05, quarantine_s=30.0))
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 30)
    assert s.scored == 1 and s.masked_terminal == 0 and s.stopped is None, s
    sdir = next((tmp_path / "tries" / "samples").iterdir())
    recs = [json.loads(p.read_text()) for p in sorted(sdir.glob("a*/record.json"))]
    assert sum(1 for r in recs if "breaker" in (r.get("meta") or {})) >= 3  # a1..a3 deferred
    assert sum(1 for r in recs if r["mask_sample"] and "breaker" not in (r.get("meta") or {})) == 1  # the one real failure
    assert any(r["attempt"].endswith("-a5") and r["reward"] == 1.0 for r in recs)


async def test_a_retry_rule_below_the_breaker_threshold_still_defers_the_outage(tmp_path):
    """``RetryRule(max_attempts=1)`` with the default ``BreakerRule(failures=3)``: the first samples ended masked-terminal on a key that was
    still closed (the retry was refused, the key not yet open). The Runner clamps ``failures`` to ``max_attempts``, so the first
    failure opens the key and is deferred."""
    spec = RunSpec(run_id="clamp", tasks=[_T()], policies=[_P()], harnesses=[_outage(time.monotonic() + 0.3)], k=3, concurrency=1, retry_backoff_s=0.0,
                   retry=RetryRule(max_attempts=1), breaker=BreakerRule(failures=3, cooldown_s=0.1, quarantine_s=30.0))
    s = await asyncio.wait_for(Runner(tmp_path).run(spec), 30)
    assert s.scored == 3 and s.masked_terminal == 0 and s.stopped is None, s
    ev = _events(tmp_path / "clamp")
    assert next(e for e in ev if e["event"] == "breaker_open")["streak"] == 1  # opened by the first failure, not the third
    assert not any(r["mask_sample"] for r in _rows(tmp_path / "clamp", "results.jsonl"))


async def test_a_resume_runs_the_samples_a_quarantine_left_owed(tmp_path):
    """``RunSummary.quarantined``: the held samples run again on a resume (the breakers start closed, nothing is seeded from the journal)."""
    state = {"broken": True}

    class Grader:
        id = "grader@1"

        async def run(self, a):
            if a.task.id == "broken" and state["broken"]:
                return HarnessResult(failure=Failure(kind="verifier_error", stage="verification", blame="task", retryable=False, detail="verifier_timeout",
                                                     evidence={"probe": "unprobed", "control": "failed", "status": "verifier_timeout"}))
            return HarnessResult(reward=1.0, steps=1)

    spec = RunSpec(run_id="qr", tasks=[_T(id="broken"), _T(id="good")], policies=[_P()], harnesses=[Grader()], k=8, concurrency=2, retry_backoff_s=0.0,
                   breaker=BreakerRule(task_samples=2, cooldown_s=0.2, quarantine_s=0.3))
    s1 = await asyncio.wait_for(Runner(tmp_path).run(spec), 60)
    held = s1.quarantined_samples
    assert s1.quarantined and held >= 1
    state["broken"] = False  # the grader is fixed
    s2 = await asyncio.wait_for(Runner(tmp_path).run(spec), 60)
    assert s2.quarantined == {} and s2.quarantined_samples == 0 and s2.stopped is None, s2
    assert s2.attempts == held  # exactly the held samples were dispatched
    assert s2.terminal == 16 and s2.scored == 8 + held and s2.masked_terminal == 8 - held  # the earlier terminal masks are not reopened (verification)
    ev = _events(tmp_path / "qr")
    assert sum(1 for e in ev if e["event"] == "quarantine") == 1 and sum(1 for e in ev if e["event"] == "breaker_open") == 1  # nothing seeded from them


async def test_a_resume_after_deferred_attempts_gives_the_sample_its_full_retry_budget(tmp_path):
    """The deferred attempts left on disk by a run stopped at its wall clock use up none of the sample's retries on the resume: its first
    attempt of the new call fails once and RetryRule(max_attempts=2) still retries it (``tries`` skips ``meta.breaker`` records;
    counting dispatched attempts, a1 + a2 >= 2, would end it masked-terminal)."""
    mode = {"down": True}
    seen: dict[str, int] = {}

    class H:
        id = "svc@1"

        async def run(self, a):
            if mode["down"]:
                return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="HTTP 503"))
            sid = a.sample.id()
            seen[sid] = seen.get(sid, 0) + 1
            if seen[sid] == 1:  # the first attempt of this call flakes once
                return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="HTTP 503"))
            return HarnessResult(reward=1.0, steps=1)

    def spec(**kw) -> RunSpec:
        return RunSpec(run_id="dres", tasks=[_T()], policies=[_P()], harnesses=[H()], k=2, concurrency=1, retry_backoff_s=0.0, retry=RetryRule(max_attempts=2), **kw)

    s1 = await asyncio.wait_for(Runner(tmp_path).run(spec(wall_clock_s=0.4, breaker=BreakerRule(failures=1, cooldown_s=60.0, quarantine_s=900.0))), 30)
    assert s1.stopped == "wall_clock" and s1.terminal == 0 and s1.scored == 0
    recs = [json.loads(p.read_text()) for p in (tmp_path / "dres" / "samples").glob("*/a*/record.json")]
    assert recs and all("breaker" in (r.get("meta") or {}) for r in recs)  # every attempt on disk was deferred
    mode["down"] = False
    s2 = await asyncio.wait_for(Runner(tmp_path).run(spec(breaker=None)), 30)
    assert s2.stopped is None and s2.scored == 2 and s2.masked_terminal == 0, s2
