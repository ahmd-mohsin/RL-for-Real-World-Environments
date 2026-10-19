"""The workspace harness: a policy works in a shell workspace World; its work is *collected as a patch* and graded by
the task from a pristine tree (the coding seam; the shape of a Harbor trial with a separate verifier).

Phases: seed (open) -> agent (run_episode; broken-environment check) -> collect (tolerant or legacy; an empty patch from
a changed tree is ``crucible:empty_collection``) -> verification (``task.grade_patch``) -> cleanup.

A collection failure preserves the workspace in the attempt directory. The next attempt of the same sample then
**re-collects only** (tolerant hook, preserved tree) instead of re-running the agent: the agent's work was fine, the
harness lost it (research m9-m10/02 §3.3 row 1). Likewise a grade masked for infrastructure (contention, stale image,
verifier timeout) is retried as a **re-grade of the stored patch** (``regrade``; Harbor's regrade-from-artifacts):
re-running the agent would sample a different solution and silently replace the one being measured.

Tasks that declare ``perf = True`` grade inside a ``PerfGuard`` slot (one perf grade per host, quiet host first).

M17 checkpoint credit (opt-in): the harness's ``sidecar(attempt)`` or the task's ``episode_sidecar(attempt)``
(``native.EpisodeSidecar``; ``factory.checkpoint``) observes the episode, grades workspace snapshots out of band and writes
``progress.jsonl``, which the record builder turns into ``Turn.advantage``. Off (the default): no observer, no change.
"""

from __future__ import annotations

import json
import shutil
import time
from typing import Any

from crucible.actors.policy import run_episode
from crucible.core.schema.run import Failure
from crucible.runtime.collect import Collected, classify, collect, repo_broken
from crucible.runtime.harness.base import Attempt, HarnessResult, settle_usage
from crucible.runtime.harness.native import LIMIT_STOPS, NativeHarness, episode_sidecar, probed, save_episode
from crucible.runtime.infra import StageFailure, boundary, classify_verifier, failure


class WorkspaceHarness(NativeHarness):
    def __init__(self, name: str = "native:workspace", version: str = "1", *, collect_mode: str = "tolerant", perf_guard=None, host: str = "local", **kw):
        super().__init__(name, version, **kw)
        self.collect_mode, self.perf_guard, self.host = collect_mode, perf_guard, host

    def _prior(self, attempt: Attempt) -> tuple[str, object] | None:
        """What an earlier attempt of this sample left that a retry can reuse instead of re-running the agent. The latest
        attempt decides *what* (re-collect or re-grade); the work itself is found by looking back to the attempt that ran
        the agent (M10 audit V1/V3: a re-collect or a train-mode re-grade preserved nothing, so the next retry re-ran the
        agent and silently replaced the solution being measured)."""
        prev = attempt.dir.parent / f"a{attempt.n - 1}"
        rec = prev / "record.json"
        if attempt.n <= 1 or not rec.exists():
            return None
        r = json.loads(rec.read_text())
        f = r.get("failure") or {}
        if not r.get("mask_sample"):
            return None
        for k in range(attempt.n - 1, 0, -1):
            d = attempt.dir.parent / f"a{k}"
            if f.get("kind") == "crucible:empty_collection" and (d / "preserved_ws").exists():
                return "recollect", d / "preserved_ws"
            if f.get("stage") in ("verification", "collect") and f.get("kind") != "crucible:empty_collection" and (d / "artifacts" / "model.patch").exists():
                return "regrade", d
        return None

    @staticmethod
    def _inherit_calls(res: HarnessResult, source: object) -> None:
        """A re-grade or re-collect measures the source attempt's work: its rollout is that attempt's model calls (V3: in
        train mode a call-less re-grade was masked ``capture_incomplete``)."""
        from pathlib import Path

        src = Path(str(source))
        src = src.parent if src.name == "preserved_ws" else src
        f = src / "calls.json"
        if f.exists():
            try:
                res.calls = json.loads(f.read_text())
                res.artifacts["calls_from"] = src.name
            except json.JSONDecodeError:
                pass

    async def regrade(self, attempt: Attempt, prior) -> HarnessResult:
        """Grade the patch stored by an earlier attempt (no agent, no spend)."""
        res = HarnessResult(stop="regraded")
        t: dict[str, float] = {}
        meta = json.loads((prior / "record.json").read_text()).get("meta") or {} if (prior / "record.json").exists() else {}
        patch = (prior / "artifacts" / "model.patch").read_text()
        try:
            await self._grade(attempt, Collected(patch=patch, mode="stored", tree_changed=bool(meta.get("tree_changed", True))), res, t)
        except StageFailure as sf:
            res.failure = sf.failure
        self._inherit_calls(res, prior)
        res.artifacts["regraded_from"] = prior.name
        res.timings = {k: round(v, 3) for k, v in t.items()}
        return res

    async def _grade(self, attempt: Attempt, c: Collected, res: HarnessResult, t: dict) -> None:
        (attempt.dir / "artifacts").mkdir(exist_ok=True)
        (attempt.dir / "artifacts" / "model.patch").write_text(c.patch)
        res.artifacts["patch"] = "artifacts/model.patch"
        res.tree_changed = c.tree_changed
        t0 = time.monotonic()
        if self.perf_guard is not None and getattr(attempt.task, "perf", False):
            async with self.perf_guard.slot(self.host) as g:
                (attempt.dir / "perf_guard.json").write_text(json.dumps(g, default=str))
                with boundary("verification"):
                    rep = await attempt.task.grade_patch(c.patch, attempt.dir / "grade")
        else:
            with boundary("verification"):
                rep = await attempt.task.grade_patch(c.patch, attempt.dir / "grade")
        t["verification"] = time.monotonic() - t0
        (attempt.dir / "verifier").mkdir(exist_ok=True)
        (attempt.dir / "verifier" / "report.json").write_text(json.dumps(rep.to_dict(), default=str, indent=1))
        res.grade = {**res.grade, "reward": rep.reward, "gates": rep.gate_bits(), "infra": rep.infra}
        if rep.infra.get("status") == "masked":
            raise StageFailure(classify_verifier(rep.infra))
        res.reward, res.gates, res.gate_kinds = rep.reward, rep.gate_bits(), rep.gate_kinds()

    async def run(self, attempt: Attempt) -> HarnessResult:
        t: dict[str, float] = {}
        res = HarnessResult()
        world = None
        policies: dict[str, Any] = {}
        spend0 = attempt.policy.spend()
        prior = self._prior(attempt)
        if prior is not None and prior[0] == "regrade":
            return await self.regrade(attempt, prior[1])
        try:
            if prior is not None:  # re-collect only: never re-run an agent whose work the harness lost
                # the base recorded by the attempt that ran the agent (M10 audit V2: this read the *new* attempt's dir and
                # fell back to the agent-movable ``base`` tag, so a committed fix collected as an empty patch)
                base_file = prior[1].parent / "base_sha"
                if not base_file.exists():
                    raise StageFailure(Failure(kind="crucible:empty_collection", stage="collect", retryable=False,
                                               reason="no recorded base commit for the preserved workspace"))
                res.stop = "recollected"
                self._inherit_calls(res, prior[1])
                c = collect(prior[1], base=base_file.read_text().strip(), mode="tolerant")
                f = classify(c)
                if f:
                    raise StageFailure(f)
                await self._grade(attempt, c, res, t)
                return res
            t0 = time.monotonic()
            with boundary("seed"):
                world = await attempt.task.open(attempt.dir / "work")
            t["seed"] = time.monotonic() - t0
            t0 = time.monotonic()
            # the commit the agent starts from, recorded before it runs: the ``base`` tag is in the agent's own repository
            # and can be deleted or moved (M7 audit)
            base_sha = _base_sha(attempt.task.workspace(world))
            if base_sha:
                (attempt.dir / "base_sha").write_text(base_sha + "\n")
            with boundary("agent"):
                policies = attempt.policy.make(attempt)
                side = episode_sidecar(self.sidecar, attempt)  # M17 checkpoint credit (opt-in; None = off, nothing changes)
                ep = await run_episode(world, probed(policies, side, world), limits=self.limits, observers=[side] if side is not None else (),
                                       sample_key=attempt.id)
                if side is not None:
                    res.grade["progress"] = ep.observed[0]
            t["agent"] = time.monotonic() - t0
            res.steps, res.stop, res.truncated = ep.steps, ep.stop_condition, ep.truncated or ep.stop_condition in LIMIT_STOPS
            save_episode(attempt, ep, res, self.id, policies)
            if ep.masked:
                f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
                raise StageFailure(failure(f["kind"], "agent", f.get("reason", "")))
            pe = next((x for x in ep.failures if x.get("kind") == "crucible:policy_error"), None)
            if pe is not None:  # the policy's own code or output raised: its outcome, scored 0 (M10 audit V6)
                raise StageFailure(Failure(kind="crucible:policy_error", stage="agent", reason=pe.get("reason", "")))
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") else []
            ws = attempt.task.workspace(world)
            broken += [b for b in repo_broken(ws, base_sha) if b not in broken]
            if broken:
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason="; ".join(broken)[:400]))
            t0 = time.monotonic()
            c = collect(ws, base=base_sha or "base", mode=self.collect_mode)
            t["collect"] = time.monotonic() - t0
            f = classify(c)
            if f:  # keep the agent's tree as it is (a rename: unreadable files survive) for a re-collect-only retry
                shutil.move(str(ws), str(attempt.dir / "preserved_ws"))
                raise StageFailure(f)
            # the policy's World is closed before grading: a perf grade must not count the attempt's own sandbox or the
            # processes it started as foreign load (M10 audit V5: they masked the grade as node contention)
            closing, world = world, None
            try:
                await closing.close()
            except Exception:  # noqa: BLE001, S110 -- cleanup never overrides a result
                pass
            await self._grade(attempt, c, res, t)
        except StageFailure as sf:
            res.failure = sf.failure
        finally:
            if world is not None:
                try:
                    await world.close()
                except Exception:  # noqa: BLE001, S110 -- cleanup never overrides a result
                    pass
            usd, unc = attempt.policy.spend()
            res.cost_usd, res.cost_uncertain_usd = max(0.0, usd - spend0[0]), max(0.0, unc - spend0[1])
            # the attempt's own model calls when it ran the agent (NativeHarness does the same): a client shared by concurrent
            # attempts makes the spend delta count theirs too (M17 report 11: 10 concurrent attempts booked 7.3x the real cost)
            own = [c for cs in (res.calls or {}).values() for c in cs if c.get("cost_usd") is not None]
            if own and not res.artifacts.get("calls_from"):
                res.cost_usd = round(sum(float(c["cost_usd"]) for c in own), 9)
            if policies:  # an agent ran here: a re-collect's calls are those of the attempt that ran it
                settle_usage(res, res.calls or {a: list(getattr(p, "model_calls", [])) for a, p in policies.items()})
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res


def _base_sha(ws) -> str | None:
    import subprocess

    try:
        r = subprocess.run(["git", "-C", str(ws), "rev-parse", "--verify", "-q", "base^{commit}"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None
