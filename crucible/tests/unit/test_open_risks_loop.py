"""The open risks of the two framework audits that the trainer loop and the transcript harness carried (2026-10-28):
harness-reward-audit.md R1, R2, R5, R8, R9 and the masked-share stop; async-rl-audit.md R2, R3, R4, R6, R7.

Every test fails on the code before its fix (checked by reverting the source); each names the item it covers.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from crucible_pack_toy.runtime import FibTask

from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, PolicyOutputError
from crucible.core.schema.rollout import RolloutRecord
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.inject import FakeLLM, tool_reply
from crucible.runtime.run import Runner, RunSpec
from crucible.train.bridges.tinker.backend import TokenDatum
from crucible.train.bridges.tinker.datums import build_datums
from crucible.train.bridges.tinker.fake import FakeTinker
from crucible.train.bridges.tinker.trainer import Group, TinkerTrainer, TrainConfig, staleness_metrics
from crucible.train.chat import ByteChatCodec
from crucible.train.quarantine import QuarantineConfig, TaskQuarantine

MODEL = "Qwen/Qwen3.6-35B-A3B"


async def _run(tmp_path, task, policy, harness, run_id):
    await Runner(tmp_path).run(RunSpec(run_id=run_id, tasks=[task], policies=[policy], harnesses=[harness], k=1))
    adir = min((tmp_path / run_id / "samples").glob("*/a*"))
    return json.loads((adir / "record.json").read_text())


# ---- harness-reward audit R1: a harness exception is the framework's, masked ---------------------------------------------------
async def test_r1_a_harness_bug_inside_the_transcript_policy_is_masked_as_a_framework_fault(tmp_path):
    bug = ValueError("zip() argument 2 is shorter than argument 1")  # a harness bug of the kind R1 names, raised inside act
    pol = PolicyRef(id="fake:bug", make=lambda a: {"agent": LLMToolPolicy("agent", FakeLLM(["done"], fail_at=(0, bug)))}, config={"bug": 1})
    rec = await _run(tmp_path, FibTask(), pol, WorkspaceHarness(), run_id="r1-bug")
    # before the fix: crucible:policy_error, scored 0, a negative signal for a bug the policy did not cause
    assert rec["mask_sample"] and rec["failure"]["kind"] == "crucible:framework_error"
    assert rec["failure"]["detail"] == "framework" and rec["failure"]["blame"] == "harness" and rec["reward"] is None


class _BadOutput(LLMToolPolicy):
    async def act(self, obs):
        raise PolicyOutputError("the model's reply named no action")


class _Scripted:
    actor_id = "agent"

    async def act(self, obs):
        raise ValueError("an out-of-range move")  # a policy whose own logic parses actions: still its outcome


@pytest.mark.parametrize("make", [lambda a: {"agent": _BadOutput("agent", FakeLLM([]))}, lambda a: {"agent": _Scripted()}])
async def test_r1_the_models_own_output_error_stays_the_policys_scored_zero(tmp_path, make):
    rec = await _run(tmp_path, FibTask(), PolicyRef(id="fake:out", make=make, config={"out": 1}), WorkspaceHarness(), run_id="r1-own")
    assert not rec["mask_sample"] and rec["failure"]["kind"] == "crucible:policy_error" and rec["reward"] == 0.0


# ---- harness-reward audit R2: the agent clock counts the agent's own time in training ----------------------------------------
class _SlowLLM:
    """Every model call takes ``delay`` seconds: a sampler queue under load."""

    def __init__(self, inner, delay: float):
        self.inner, self.delay = inner, delay

    async def complete(self, messages, **kw):
        await asyncio.sleep(self.delay)
        return await self.inner.complete(messages, **kw)

    def __getattr__(self, k):
        return getattr(self.inner, k)


@pytest.mark.parametrize("clock,stop", [("wall", "timeout"), ("own", "agent_done")])
async def test_r2_sampler_latency_decides_the_timeout_only_on_the_wall_clock(tmp_path, clock, stop):
    acts = [tool_reply("ws__shell", {"command": "ls"}, 1), tool_reply("ws__shell", {"command": "ls"}, 2), "done"]
    pol = PolicyRef(id=f"fake:slow-{clock}", make=lambda a: {"agent": LLMToolPolicy("agent", _SlowLLM(FakeLLM(list(acts)), 0.35))}, config={"slow": 1})
    h = WorkspaceHarness(limits=EpisodeLimits(max_steps=10, timeout_s=0.6, agent_clock=clock))  # 3 calls x 0.35 s > 0.6 s
    rec = await _run(tmp_path, FibTask(), pol, h, run_id=f"r2-{clock}")
    assert rec["meta"]["stop"] == stop  # before the fix: EpisodeLimits had no agent_clock, and every clock was the wall clock


def test_r2_training_envs_count_own_time_and_mask_timeouts_while_evals_are_unchanged():
    from crucible.factory.envs import make_env
    from crucible.runtime.identity import describe

    task = SimpleNamespace(task_id="t", prompt="p", split="train", verifier_version="v")
    train, ev = make_env(task, training=True), make_env(task)
    assert train.limits.agent_clock == "own" and train.harness.limit_policy.for_stop("timeout") == "mask"
    assert ev.limits.agent_clock == "wall" and ev.harness.limit_policy.for_stop("timeout") == "score"
    assert "agent_clock" not in describe(ev.limits, "limits")  # an eval harness keeps the id it had before the field existed
    assert ev.harness.id != train.harness.id


def test_r2_a_model_call_holds_only_the_agent_clock():
    from crucible.actors import policy as pm

    async def go():
        async with asyncio.timeout(10.0) as agent, asyncio.timeout(20.0) as other:
            tok = pm._MODEL_CLOCK.set(agent)
            try:
                with pm.model_time():
                    assert agent.when() is None and other.when() is not None  # the attempt's own clock keeps running
                    await asyncio.sleep(0.05)
                left = agent.when() - asyncio.get_running_loop().time()
            finally:
                pm._MODEL_CLOCK.reset(tok)
        return left

    assert asyncio.run(go()) > 9.9  # the 50 ms of model time were not charged


# ---- harness-reward audit R5: the format penalty on every backend ------------------------------------------------------------
def _rec(rid: str, reward: float, *, errs: int = 0, version: int = 0, honesty: float | None = None) -> RolloutRecord:
    turns = [{"call_id": f"{rid}:c1", "actor": "agent", "prompt_token_ids": [1, 2], "completion_token_ids": [3, 4], "logprobs": [-0.5, -0.5],
              "policy_version": version, "format_errors": errs},
             {"call_id": f"{rid}:c2", "actor": "agent", "prompt_token_ids": [1, 2, 3, 4, 5], "completion_token_ids": [6, 7], "logprobs": [-0.5, -0.5],
              "policy_version": version}]
    doc = {"rollout_id": rid, "sample": rid, "task": "t", "harness": "h", "policy": "p", "reward": reward, "turns": turns}
    if honesty is not None:
        doc["progress"] = {"honesty": {"term": honesty}}
    return RolloutRecord.model_validate(doc)


def _call_adv(d: TokenDatum, cid: str) -> list[float]:
    off = d.meta["prompt_len"] - 1
    return next(d.advantages[off + a: off + b] for c, a, b in d.meta["call_spans"] if c == cid)


def test_r5_the_format_penalty_reaches_the_tinker_and_primerl_datums():
    recs, groups = [_rec("a", 1.0, errs=2), _rec("b", 0.0)], ["g", "g"]
    plain = {d.meta["rollout"]: d for d in build_datums(recs, groups)}
    pen = {d.meta["rollout"]: d for d in build_datums(recs, groups, format_penalty=0.1)}  # before the fix: no such argument
    assert _call_adv(pen["a"], "a:c1") == pytest.approx([x - 0.2 for x in _call_adv(plain["a"], "a:c1")])  # 0.1 x 2 errors, its own call only
    assert _call_adv(pen["a"], "a:c2") == _call_adv(plain["a"], "a:c2") and pen["b"].advantages == plain["b"].advantages
    signed = {d.meta["rollout"]: d for d in build_datums(recs, groups, format_penalty=0.1, format_mode="signed")}
    assert _call_adv(signed["a"], "a:c1") == [0.0, 0.0]  # a positive rollout stops reinforcing the malformed call
    pos = sum(a for d in signed.values() for a, m in zip(d.advantages, d.mask, strict=True) if m and a > 0)
    assert pos == pytest.approx(sum(a for d in plain.values() for a, m in zip(d.advantages, d.mask, strict=True) if m and a > 0))  # mass kept
    cfg = TrainConfig(model=MODEL, tool_format_penalty=0.1)
    assert cfg.identity()["tool_format_penalty"] == 0.1 and "tool_format_penalty" not in TrainConfig(model=MODEL).identity()


# ---- harness-reward audit R8 / R9: honesty in flat groups; the loss scale ----------------------------------------------------
def _trainer_on(tmp_path: Path, **cfg) -> TinkerTrainer:
    env = SimpleNamespace(task=SimpleNamespace(task_id="t1", split="train", verifier_version="v1"), policy_kwargs=dict)
    base = {"model": MODEL, "group_size": 2, "steps": 1, "require_band": False, "turn_eta": 0.0, "quarantine_q": 0}
    return TinkerTrainer([env], config=TrainConfig(**{**base, **cfg}), backend=FakeTinker(), run_dir=tmp_path, codec=ByteChatCodec(), objective_id="obj")


def test_r8_a_flat_group_whose_honesty_terms_differ_trains(tmp_path):
    t = _trainer_on(tmp_path)
    t._open()
    honest = Group(0, None, "t1", "v1", "g0", [_rec("a", 0.0, honesty=0.0), _rec("b", 0.0, honesty=-0.4)], 0, 0.0)
    flat = Group(1, None, "t1", "v1", "g1", [_rec("c", 0.0, honesty=0.0), _rec("d", 0.0, honesty=0.0)], 0, 0.0)
    assert t._verdict(honest) is None  # before the fix: zero_spread (the rewards are equal), so honesty never trained here
    assert t._verdict(flat) == "zero_spread"


def test_r9_the_default_loss_scale_is_a_global_token_mean():
    assert TrainConfig(model=MODEL).loss_scale == "token_mean"  # before: "sum"


# ---- async-RL audit R2: the trust region per staleness bucket ----------------------------------------------------------------
def test_ar2_mismatch_and_mis_share_are_logged_per_staleness_bucket():
    def d(mv: int, n: int) -> TokenDatum:
        return TokenDatum(input_ids=[1] * n, target_ids=[1] * n, logprobs=[-1.0] * n, advantages=[0.0] * n, mask=[1] * n, meta={"min_version": mv})

    ds = [d(3, 4), d(2, 2)]
    tl = [[-1.0] * 4, [-1.0 + math.log(8.0), -1.0]]  # bucket 1: one token at r = 8 (outside MIS's [0.5, 5]), one at r = 1
    m = staleness_metrics(ds, tl, 3)
    assert m["staleness/tokens_frac/s0"] == pytest.approx(4 / 6) and m["staleness/tokens_frac/s1"] == pytest.approx(2 / 6)
    assert m["is/masked_frac/s0"] == 0.0 and m["is/masked_frac/s1"] == pytest.approx(0.5)
    assert m["kl/mismatch/s0"] == 0.0 and m["kl/mismatch/s1"] == pytest.approx((8.0 - 1.0 - math.log(8.0)) / 2)
    assert "kl/mismatch/s1" not in staleness_metrics(ds, [], 3) and "staleness/tokens_frac/s1" in staleness_metrics(ds, [], 3)  # prime-rl


# ---- the loop with scripted groups (async-RL audit R2, R3, R4; the masked-share stop) -----------------------------------------
class _Runner:
    def __init__(self, mv: int | None):
        self.mv = mv

    def inflight_min_version(self, run_id=None):
        return self.mv


class _Sink:
    def __init__(self):
        self.rows, self.records = [], []

    def log(self, m, step):
        self.rows.append((step, dict(m)))

    def record(self, kind, data, step):
        self.records.append((kind, data))

    def summary(self, data):
        pass

    def finish(self, status):
        pass


class ScriptedTrainer(TinkerTrainer):
    """Groups come from ``plan`` (one entry per launch: seconds to land, or "masked"); a slow group shows its in-flight
    minimum version to the cancel check through a stand-in Runner, as ``Runner.inflight_min_version`` does."""

    def __init__(self, *a, plan, **kw):
        super().__init__(*a, **kw)
        self.plan, self.cancelled = list(plan), []

    async def _group(self, env, gi, *, picked_step=None, probe=None):
        v = self.sampler.version
        delay = self.plan.pop(0) if self.plan else 0.0
        runners = self.__dict__.setdefault("_runners", {})  # (the loop reads it; set here so the old loop runs this script too)
        runners[gi] = _Runner(v)
        try:
            await asyncio.sleep(0.0 if delay == "masked" else delay)
        except asyncio.CancelledError:
            self.cancelled.append(gi)
            raise
        finally:
            runners.pop(gi, None)
        if delay == "masked":
            recs = [RolloutRecord.model_validate({"rollout_id": f"m{gi}{i}", "sample": "s", "task": "t1", "harness": "h", "policy": "p", "mask_sample": True,
                                                  "failure_kind": "crucible:env_container_lost"})
                    for i in range(2)]
        else:
            recs = [_rec(f"g{gi}a", 1.0, version=v), _rec(f"g{gi}b", 0.0, version=v)]
        return Group(gi, env, "t1", "v1", f"g{gi:06d}-t1", recs, v, 0.0)


def _scripted(tmp_path: Path, plan, sinks=(), **cfg) -> ScriptedTrainer:
    env = SimpleNamespace(task=SimpleNamespace(task_id="t1", split="train", verifier_version="v1"), policy_kwargs=dict)
    base = {"model": MODEL, "group_size": 2, "steps": 1, "require_band": False, "turn_eta": 0.0, "quarantine_q": 0, "cap_usd": 100.0}
    return ScriptedTrainer([env], config=TrainConfig(**{**base, **cfg}), backend=FakeTinker(), run_dir=tmp_path, codec=ByteChatCodec(),
                           objective_id="obj", sinks=list(sinks), plan=plan)


def test_ar3_a_group_that_can_no_longer_train_is_cancelled_before_the_weight_swap(tmp_path):
    # A trains step 1; B (sampling v0, slow) is in flight; C trains step 2: under max_staleness 1, v2 - v0 > 1, so B is doomed
    t = _scripted(tmp_path, [0.0, 3.0, 0.05, 0.0], steps=3, async_groups=1, max_staleness=1)
    t0 = time.monotonic()
    out = asyncio.run(t.run())
    assert out["steps_done"] == 3 and t.cancelled == [1]  # before the fix: B ran its 3 s out and was dropped as stale on landing
    assert out["drops"].get("cancelled_stale") == 1 and "stale" not in out["drops"] and time.monotonic() - t0 < 2.5
    row = next(json.loads(x) for x in (tmp_path / "drops.jsonl").read_text().splitlines() if '"cancelled_stale"' in x)
    assert row["min_version"] == 0 and row["group"].startswith("g000001")


def test_ar4_an_error_stop_cancels_the_groups_in_flight(tmp_path):
    class Dies(ScriptedTrainer):
        async def _step(self, batch):
            raise RuntimeError("the trainer died")

    env = SimpleNamespace(task=SimpleNamespace(task_id="t1", split="train", verifier_version="v1"), policy_kwargs=dict)
    t = Dies([env], config=TrainConfig(model=MODEL, group_size=2, steps=2, require_band=False, turn_eta=0.0, quarantine_q=0, async_groups=1,
                                       max_staleness=1, cap_usd=100.0), backend=FakeTinker(), run_dir=tmp_path, codec=ByteChatCodec(), objective_id="obj",
             plan=[0.0, 6.0])
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="trainer died"):
        asyncio.run(t.run())
    assert time.monotonic() - t0 < 3.0 and t.cancelled == [1]  # before the fix: the finally awaited the 6 s group
    assert t.drops == {"cancelled_at_stop:error:RuntimeError": 1}


def test_masked_share_stops_the_run_instead_of_starving_training(tmp_path):
    t = _scripted(tmp_path, ["masked"] * 10, max_groups_per_step=10, masked_window=4)
    out = asyncio.run(t.run())
    # before the fix: no such setting; the run spent its 10 launches and stopped too_many_drops
    assert out["stop"] == "masked_frac" and out["masked_stop"]["kinds"] == {"crucible:env_container_lost": 4}
    assert out["drops"] == {"infra_in_group": 2}


def test_ar2_a_synchronous_step_reports_staleness_zero_and_its_bucket(tmp_path):
    sink = _Sink()
    out = asyncio.run(_scripted(tmp_path, [0.0], sinks=[sink]).run())
    assert out["steps_done"] == 1
    _, m = sink.rows[0]
    assert m["staleness/max"] == 0.0  # before the fix: 1.0 (measured against the version after the step)
    assert m["staleness/tokens_frac/s0"] == 1.0 and m["kl/mismatch/s0"] == pytest.approx(0.0) and m["is/masked_frac/s0"] == 0.0


# ---- async-RL audit R3: the Runner's in-flight minimum version ------------------------------------------------------------------
def test_ar3_the_runner_exposes_each_runs_inflight_minimum_version(tmp_path):
    r = Runner(tmp_path)
    calls = [{"tokens": {"policy_version": 3}}, {"tokens": {"policy_version": 1}, "accepted": False}, {"tokens": {"policy_version": 2}}]
    r.live["a1"] = SimpleNamespace(run_id="g1", policies={"agent": SimpleNamespace(model_calls=calls)})
    r.live["a2"] = SimpleNamespace(run_id="g2", policies={"agent": SimpleNamespace(model_calls=[{"tokens": {"policy_version": 0}}])})
    assert r.inflight_min_version("g1") == 2 and r.inflight_min_version() == 0 and r.inflight_min_version("g3") is None


# ---- async-RL audit R6: the quarantine under async sampling -----------------------------------------------------------------
def test_ar6_only_the_retry_group_decides_a_probation():
    q = TaskQuarantine(["t"], QuarantineConfig(q=1, r=2))
    q.observe("t", [0.0, 0.0], [], 1, picked=1, probe=False)  # flat: quarantined at step 1 until step 3
    assert q.state["t"].until == 3
    q.pick(0, 3)  # probation opens: the retry group is out
    assert q.last_probe
    # a group picked at step 0 (before the quarantine) lands now, with spread: it measured the policy the quarantine judged
    q.observe("t", [1.0, 0.0], [], 3, picked=0, probe=False)
    assert q.state["t"].probation  # before the fix: it released the task in place of the retry group
    q.observe("t", [0.0, 0.0], [], 3, picked=3, probe=True)  # the retry group is flat: quarantined again
    assert [e["event"] for e in q.events] == ["quarantined", "probation", "requarantined"]


# ---- async-RL audit R7: one row per step after a resume ---------------------------------------------------------------------
def test_ar7_a_resume_moves_the_rows_of_retrained_steps_aside(tmp_path):
    t = _trainer_on(tmp_path, save_every=2)
    t._open()
    for s in (1, 2, 3):
        (tmp_path / "steps.jsonl").open("a").write(json.dumps({"step": s}) + "\n")
        (tmp_path / "rewards.jsonl").open("a").write(json.dumps({"step": s, "reward": 1.0}) + "\n")
    (tmp_path / "steps.jsonl").open("a").write('{"step": 4, "tor')  # the crash cut the last line
    t._rollback_logs(2)  # the last save_state was step 2: steps 3 and 4 train again
    assert [json.loads(x)["step"] for x in (tmp_path / "steps.jsonl").read_text().splitlines()] == [1, 2]
    assert [json.loads(x)["step"] for x in (tmp_path / "rewards.jsonl").read_text().splitlines()] == [1, 2]
    assert len((tmp_path / "steps.rolled_back.jsonl").read_text().splitlines()) == 2  # step 3 and the torn line: kept, not deleted


# ---- harness-reward audit R6: the host-side checkpoint snapshot of lanes without their own snapshot ---------------------------
async def test_r6_the_host_checkpoint_snapshot_runs_no_filter_driver_of_the_agent(tmp_path):
    import subprocess

    from crucible.factory.checkpoint import workspace_snapshot

    ws, marker = tmp_path / "ws", tmp_path / "ran"
    ws.mkdir()

    def git(*a):
        return subprocess.run(["git", "-C", str(ws), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    (ws / "lib.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").strip()
    # the policy plants a clean filter: code it chose runs on the host, and the snapshot shows the content it substitutes
    git("config", "filter.evil.clean", f"sh -c 'touch {marker}; sed s/2/9/'")
    (ws / ".gitattributes").write_text("*.py filter=evil\n")
    (ws / "lib.py").write_text("x = 2\n")
    patch = await workspace_snapshot(ws, base)(None)
    assert not marker.exists() and "+x = 2" in patch  # before the fix: the driver ran on the host and the patch said x = 9

