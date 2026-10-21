#!/usr/bin/env python
"""S15 targeting of the multi-language debugging lane on Tinker (swarm-pipeline/07 §6, 08, 09, 11).

The key: ``TINKER_API_KEY`` from the repo-root ``.env``, sourced into this process only (``set -a; . ../.env; set +a``);
never printed. Run from ``crucible/`` of the main checkout (its ``data/`` is the durable data dir):

    set -a; . ../.env; set +a; .venv/bin/python <worktree>/crucible/scripts/lh_debug_target.py pilot --cap 10
    ... student frontier --cap 250        # the 13 candidates: student k=8, two frontier solvers k=2
    ... audit mine report                 # trace audit + cause labels, failure mining, the keep rule, W&B views
    ... cleanup                           # this phase's containers (crucible-lh-d-s15-*) and nothing else

Stages (each resumable; everything under ``--out``, default ``data/targeting/debug``):
- ``pilot``    one training candidate (``--pilot-task``), 2 student attempts + 1 frontier attempt: $/episode, turns, wall time,
               the host's Docker load (``load.jsonl``), and the projection for the full phase against ``--cap``;
- ``student``  every candidate (``data/debug/candidates.json``: through every CPU gate with a clean S7 prompt), the target
               student ``--student`` (``Qwen/Qwen3.6-35B-A3B``, the only student) k = ``--k`` through ``factory.tinker.benchmark``:
               ``make_env`` (``WorkspaceHarness`` + ``LLMToolPolicy``, the tool limit, both timeouts, the checkpoint-Phi
               sidecar), the task's own Docker World and its hidden verifier: exactly the RL environment of ``factory.tinker.train``;
- ``frontier`` the fairness solvers (``models.FRONTIER_SOLVERS``: Qwen3.5-397B-A17B, DeepSeek-V4.1-Flash), ``--frontier-k`` each;
- ``audit``    every failing student episode: deterministic facts, first wrong step, the honest-trace checks
               (``traceaudit.honest_trace_facts``), the GLM-5.3 ``auditor`` and the GLM-5.3-Flash ``judge`` labels with seeded
               controls in every batch (``audit.audit``; ``ModelBook`` refuses a same-family judge), the debug skill of
               each failure (``generators.debug.failure_skill``);
- ``disclose`` per task, ``generators.debug.disclosure_check``: which milestones fail when B lacks the window's silent units
               (reverted fixes without a milestone), for review against the report (no model call);
- ``mine``     failure findings and builder-intent requests per student (``factory.failures``);
- ``report``   the keep rule per task (``targeting.keep_verdict``), the S15 targeting records (``targeting.json``, what
               ``trainer.check_band`` reads), ``report.json``, the W&B ``crucible-factory`` run (offline) with every episode as a
               Weave-shaped trace and its audit attached.

The env budget is the lane's (``generators.debug.ENV_BUDGET``: 300 tool calls, agent 5400 s, rollout 9000 s, Phi every 20
calls, cap 15; the keep rule, the benchmark and RL use the same one); ``settings.env_budget`` records what a phase ran and
whether it was the lane's. A re-measurement passes ``--seed-offset`` so its run ids (the Weave trace keys) are new.

One Tinker meter for the whole phase (``spend.jsonl``; ``CRUCIBLE_TINKER_CAP_USD`` = ``--cap``): every student, frontier,
auditor and judge call books there and is refused past the cap. Episodes run under a host slot pool (``--slots``: Worlds
open at once), so Docker on this Mac is never oversubscribed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

sys.path.insert(0, str(Path(__file__).resolve().parent))

PREFIX = "crucible-lh-d-s15-"  # every container of this phase (World, grading); cleanup removes these and nothing else
STUDENT = "Qwen/Qwen3.6-35B-A3B:peft:262144"
FRONTIER = ("Qwen/Qwen3.5-397B-A17B:peft:262144", "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144")
SEEDS = {"student": 0, "Qwen/Qwen3.5-397B-A17B:peft:262144": 100, "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": 200}
PILOT_TASK = "db-go-echo-61eca05b"
# the pilot (2026-10-20, echo, tool limit 100): USD per episode; DeepSeek-V4.1-Flash projected from its token prices (09 §2)
PILOT_USD = {STUDENT: 0.42, FRONTIER[0]: 1.61, FRONTIER[1]: 0.62}


def data_dir() -> Path:
    from crucible.core.config import get_config

    return Path(os.environ.get("CRUCIBLE_DATA_DIR") or get_config().data_path())


def jload(p: Path, default: Any = None) -> Any:
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    text = json.dumps(x, indent=1, sort_keys=True, default=str)
    key = os.environ.get("TINKER_API_KEY")
    if key and len(key) >= 8:
        text = text.replace(key, "<secret>")
    tmp.write_text(text + "\n")
    tmp.replace(p)


def arm_name(model: str) -> str:
    return model.split(":", 1)[0].replace("/", "_")


# ---- the candidates as tasks -------------------------------------------------------------------------------------------
def candidates(debug: Path) -> list[dict[str, Any]]:
    """Every candidate through every CPU gate with a clean S7 prompt (``candidates.json`` + ``authored/<id>.json``)."""
    out = []
    for r in json.loads((debug / "candidates.json").read_text()):
        a = debug / "authored" / f"{r['id']}.json"
        if r.get("gates_ok") and a.exists() and not json.loads(a.read_text()).get("findings"):
            out.append(r)
    return out


class Tasks:
    """The debug lane's tasks (``generators.debug.DebugTask``) under their authored prompts, in this phase's containers."""

    def __init__(self, debug: Path, *, grade_parallel: int = 2, out: Path | None = None):
        import lh_debug

        from crucible.factory.generators.debug import debug_sandboxes

        lh_debug.ROOT = debug  # the lane's durable data dir (the main checkout's), never the worktree's
        self.debug, self.lh = debug, lh_debug
        self.rows = {r["id"]: r for r in candidates(debug)}
        self.gens: dict[str, Any] = {}
        self.sandboxes = debug_sandboxes(parallel=grade_parallel, prefix=f"{PREFIX}grade-")
        self._tasks: dict[str, Any] = {}
        self.out = out  # the phase dir: accepted policy-World fixes live under <out>/worldfix

    def gen(self, split: str) -> Any:
        if split not in self.gens:
            repos = sorted({r["repo"] for r in self.rows.values() if r["split"] == split})
            g = self.lh.generator(repos, split)
            g.sandboxes = self.sandboxes
            lane = g.lane()
            lane.world_prefix = f"{PREFIX}world-"
            self.gens[split] = g
        return self.gens[split]

    def task(self, tid: str) -> Any:
        if tid not in self._tasks:
            row = self.rows[tid]
            authored = json.loads((self.debug / "authored" / f"{tid}.json").read_text())
            t = self.gen(row["split"]).task(tid, authored=authored)
            t.language, t.family, t.split = row["lang"], row["repo"], row["split"]
            t.intent = "+".join(row.get("skills") or [])
            fix = jload(self.out / "worldfix" / f"{tid}.json") if self.out is not None else None
            if fix and fix.get("accepted"):
                t.world_image = fix["fixed"]  # the accepted policy World (stage worldfix); the grader keeps the pinned image
            self._tasks[tid] = t
        return self._tasks[tid]


# ---- the host: slots and load ----------------------------------------------------------------------------------------------
class Slots:
    """Worlds open at once on this host: a job of ``n`` concurrent episodes takes ``n`` slots for its duration."""

    def __init__(self, n: int):
        self.free, self.n = n, n
        self.cond = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def take(self, k: int) -> Any:
        k = min(k, self.n)
        async with self.cond:
            await self.cond.wait_for(lambda: self.free >= k)
            self.free -= k
        try:
            yield
        finally:
            async with self.cond:
                self.free += k
                self.cond.notify_all()


class LoadMonitor:
    """Every ``every`` seconds: the host's load averages and this phase's containers' CPU / memory (``docker stats``)."""

    def __init__(self, path: Path, every: float = 60.0):
        self.path, self.every = path, every
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def sample(self) -> dict[str, Any]:
        load = os.getloadavg()
        r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"], capture_output=True, text=True,
                           timeout=120, check=False)
        mine, others, cpu, mem = 0, 0, 0.0, 0.0
        for ln in r.stdout.splitlines():
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            if parts[0].startswith(PREFIX):
                mine += 1
                with contextlib.suppress(ValueError):
                    cpu += float(parts[1].rstrip("%"))
                used = parts[2].split("/")[0].strip()
                with contextlib.suppress(ValueError):
                    mem += float(used[:-3]) * {"KiB": 1 / 1024 / 1024, "MiB": 1 / 1024, "GiB": 1.0}.get(used[-3:], 0.0)
            else:
                others += 1
        return {"t": round(time.time(), 1), "load1": round(load[0], 2), "load5": round(load[1], 2), "containers_mine": mine, "containers_other": others,
                "cpu_pct_mine": round(cpu, 1), "mem_gib_mine": round(mem, 2)}

    def _run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                row = self.sample()
                with self.path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            self._stop.wait(self.every)

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a: object) -> None:
        self._stop.set()


def load_summary(path: Path, since: float = 0.0) -> dict[str, Any]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    rows = [r for r in rows if r["t"] >= since]
    if not rows:
        return {"samples": 0}

    def q(k: str) -> dict[str, float]:
        xs = sorted(float(r[k]) for r in rows)
        return {"mean": round(statistics.fmean(xs), 2), "max": xs[-1], "p50": xs[len(xs) // 2]}

    return {"samples": len(rows), "load1": q("load1"), "containers_mine": q("containers_mine"), "cpu_pct_mine": q("cpu_pct_mine"),
            "mem_gib_mine": q("mem_gib_mine"), "containers_other": q("containers_other")}


def env_budget(a: argparse.Namespace) -> dict[str, Any]:
    """The env budget this phase ran under and whether it is the lane's (``generators.debug.ENV_BUDGET``: the keep rule,
    the benchmark and RL run the debug tasks under the same one)."""
    from crucible.factory.generators.debug import ENV_BUDGET

    ran = {"tool_limit": a.tool_limit, "agent_s": a.agent_s, "rollout_s": a.rollout_s, "checkpoint_every": a.ckpt_every, "checkpoint_cap": a.ckpt_cap}
    lane = ENV_BUDGET.to_json()
    return {**ran, "lane": lane, "is_lane_budget": all(lane[k] == v for k, v in ran.items())}


# ---- the arms ----------------------------------------------------------------------------------------------------------------
class Phase:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        os.environ["CRUCIBLE_TINKER_CAP_USD"] = str(a.cap)
        os.environ["CRUCIBLE_TINKER_SPEND"] = str(self.out / "spend.jsonl")
        from crucible.train.bridges.tinker.sampler import shared_meter

        self.meter = shared_meter()  # the phase's one cap (persisted: a restart starts from the booked total)
        self.meter.cap_usd = float(a.cap)
        self.tasks = Tasks(Path(a.debug), grade_parallel=a.grade_parallel, out=self.out)
        self.slots = Slots(a.slots)
        self._codecs: dict[str, Any] = {}
        self._backend: Any = None
        self.inflight = 0.0  # the admitted jobs' projected USD not yet booked
        self.refused: list[dict[str, Any]] = []

    def est(self, model: str) -> float:
        """Projected USD per episode of ``model``: ``--est-usd`` if given, else the mean of its finished episodes in this
        phase (the meter's attribution), else the pilot's measurement scaled to the tool limit (``PILOT_USD``)."""
        given = dict(x.split("=", 1) for x in self.a.est_usd.split(",") if "=" in x)
        if model in given:
            return float(given[model])
        xs: list[float] = []
        for d in (self.out / "arms").glob("*/*/done.json"):
            j = jload(d, {}) or {}
            if j.get("model") == model and not d.parent.parent.name.startswith("pilot"):
                xs += [e["usd"] for e in episodes(self.out, d.parent.parent.name, d.parent.name) if e["usd"] > 0]
        if len(xs) >= 4:
            return statistics.fmean(xs)
        return PILOT_USD.get(model, 3.0) * max(1.0, self.a.tool_limit / 100)

    def codec(self, model: str) -> Any:
        if model not in self._codecs:
            from crucible.train.bridges.tinker.codec import TinkerChatCodec, load_tokenizer
            from crucible.train.bridges.tinker.models import resolve

            m = resolve(model)
            self._codecs[model] = TinkerChatCodec(load_tokenizer(m), m, enable_thinking=self.a.thinking)
        return self._codecs[model]

    def backend(self) -> Any:
        if self._backend is None:
            from crucible.train.bridges.tinker.backend import RealTinker

            self._backend = RealTinker()
        return self._backend

    def env_kw(self) -> dict[str, Any]:
        from crucible.factory.checkpoint import CheckpointConfig
        from crucible.factory.envs import Timeouts

        return {"phi_checkpoint": CheckpointConfig(every=self.a.ckpt_every, cap=self.a.ckpt_cap, concurrency=1), "tool_limit": self.a.tool_limit,
                "timeouts": Timeouts(agent_s=self.a.agent_s, rollout_s=self.a.rollout_s)}

    def settings(self) -> dict[str, Any]:
        a = self.a
        return {"student": a.student, "frontier": list(a.frontier), "k": a.k, "frontier_k": a.frontier_k, "tool_limit": a.tool_limit, "max_tokens": a.max_tokens,
                "temperature": 1.0, "enable_thinking": a.thinking, "agent_s": a.agent_s, "rollout_s": a.rollout_s, "checkpoint_every": a.ckpt_every,
                "checkpoint_cap": a.ckpt_cap, "seed_offset": a.seed_offset, "harness": "WorkspaceHarness+LLMToolPolicy (factory.envs.make_env)", "slots": a.slots,
                "grade_parallel": a.grade_parallel, "world": "task World image, --network none, 2 CPU / 2 GB", "containers": f"{PREFIX}*",
                "env_budget": env_budget(a)}

    def run_dir(self, arm: str, tid: str) -> Path:
        return self.out / "arms" / arm / tid

    async def job(self, model: str, tid: str, *, k: int, seed: int, purpose: str, concurrency: int, tag: str) -> dict[str, Any]:
        from crucible.factory.tinker import benchmark

        rd = self.run_dir(tag, tid)
        done = rd / "done.json"
        if (d := jload(done)) and d.get("k") == k:
            return d
        if self.meter.exhausted:
            return {"task": tid, "arm": tag, "skipped": "cap reached"}
        task = self.tasks.task(tid)
        async with self.slots.take(concurrency):
            proj = k * self.est(model)
            if self.meter.spent + self.inflight + proj > self.a.cap - self.a.audit_reserve:
                why = (f"projected {proj:.2f} USD on {self.meter.spent:.2f} spent + {self.inflight:.2f} in flight passes the cap {self.a.cap} less "
                       f"the audit reserve {self.a.audit_reserve}")
                self.refused.append({"task": tid, "arm": tag, "why": why})
                print(f"[{tag}] {tid}: not started: {why}", flush=True)
                return {"task": tid, "arm": tag, "skipped": why}
            self.inflight += proj
            t0 = time.time()
            print(f"[{tag}] {tid}: k={k} start (spent {self.meter.spent:.3f})", flush=True)
            try:
                rep = await benchmark([task], model=model, k=k, run_dir=rd, cap_usd=self.a.cap, meter=self.meter, backend=self.backend(),
                                      codec=self.codec(model), max_tokens=self.a.max_tokens, enable_thinking=self.a.thinking, concurrency=concurrency,
                                      seed=seed, purpose=purpose, observe=False, **self.env_kw())
            except Exception as e:  # noqa: BLE001 - one task's failure is recorded; the phase goes on
                print(f"[{tag}] {tid}: error {type(e).__name__}: {e}", flush=True)
                return {"task": tid, "arm": tag, "error": f"{type(e).__name__}: {e}"[:500]}
            finally:
                self.inflight -= proj
            rep.pop("episodes", None)
            row = {"task": tid, "arm": tag, "model": model, "k": k, "seed": seed, "purpose": purpose, "wall_s": round(time.time() - t0, 1),
                   "metrics": rep["metrics"], "settings": rep["settings"], "codec": rep.get("codec")}
            jsave(done, row)
            m = rep["metrics"]
            print(f"[{tag}] {tid}: pass@1 {m.get('pass@1')} episodes {m.get('episodes')} masked {m.get('masked')} usd {m.get('usd_total'):.3f} "
                  f"wall {row['wall_s']}s (spent {self.meter.spent:.3f})", flush=True)
            return row


_SPEND: dict[str, list[dict[str, Any]]] = {}


def episode_usd(spend_path: Path, window: tuple[float, float] | None = None) -> dict[str, float]:
    """USD per episode key from the phase's meter receipts (survives restarts, unlike a sampler's in-memory attribution).
    ``window``: only the calls booked inside it (a run's journal span): a re-run of a task under the same run id reuses
    its sample keys, so without the window the two runs' dollars would add up on one key."""
    key = str(spend_path)
    if key not in _SPEND:
        _SPEND[key] = [json.loads(ln) for ln in spend_path.read_text().splitlines() if ln.strip()] if spend_path.exists() else []
    out: Counter[str] = Counter()
    for r in _SPEND[key]:
        if r.get("episode") and (window is None or window[0] <= float(r.get("t") or 0.0) <= window[1]):
            out[str(r["episode"])] += float(r.get("usd") or 0.0)
    return dict(out)


def journal_window(rollouts: Path) -> tuple[float, float] | None:
    """The span of a run's journal events (every run dir under ``rollouts``), padded by a minute."""
    ts = []
    for j in rollouts.glob("*/journal.jsonl"):
        for ln in j.read_text().splitlines():
            if ln.strip():
                try:
                    ts.append(float(json.loads(ln).get("ts") or 0.0))
                except ValueError:
                    continue
    ts = [t for t in ts if t > 0]
    return (min(ts) - 60.0, max(ts) + 60.0) if ts else None


def episodes(out: Path, arm: str, tid: str) -> list[dict[str, Any]]:
    """Per attempt of one (arm, task): reward, pass, Phi, milestones, stop, turns, tool calls, wall time, USD."""
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.factory.traceaudit import regrade_source
    from crucible.observe.traces import attempt_dirs

    usd = episode_usd(out / "spend.jsonl", journal_window(out / "arms" / arm / tid / "rollouts"))
    rows = []
    for adir in attempt_dirs(out / "arms" / arm / tid / "rollouts"):
        rec = jload(adir / "record.json", {}) or {}
        src = regrade_source(adir, rec)  # a re-graded attempt: its grade here, the agent's trail, timing and spend in the source
        srec = rec if src == adir else (jload(src / "record.json", {}) or {})
        rep = jload(adir / "verifier" / "report.json", {}) or {}
        ro_p = src / "rollout.json"
        ro = RolloutRecord.model_validate_json(ro_p.read_text()) if ro_p.exists() else None
        meta = rep.get("meta") or {}
        trace = jload(src / "trace.json", {}) or {}
        calls = sum(1 for e in trace.get("events") or [] if e.get("type") == "tool_call" and (e.get("actor") or "agent") != "system")
        sid = adir.parent.name
        key = ro.rollout_id if ro is not None else f"{sid}-{src.name}"
        masked = bool(rec.get("mask_sample")) or rec.get("reward") is None
        timing = srec.get("timing") or {}
        rows.append({"task": tid, "arm": arm, "episode": f"{rec.get('run_id') or adir.parents[2].name}/{sid}/{adir.name}", "dir": str(adir),
                     "reward": rec.get("reward"), "passed": (not masked) and float(rec.get("reward") or 0.0) >= 1 - 1e-9, "masked": masked,
                     "failure": rec.get("failure"), "stop": (srec.get("meta") or {}).get("stop"), "phi": meta.get("phi"),
                     "milestones": meta.get("milestones"), "full_pass": meta.get("full_pass"), "gates": (rec.get("meta") or {}).get("gates"),
                     "regraded_from": src.name if src != adir else None, "turns": len(ro.turns) if ro is not None else None, "tool_calls": calls,
                     "wall_s": round(sum(float(v) for v in timing.values() if isinstance(v, int | float)), 1) or None,
                     "model_s": timing.get("model"), "policy": rec.get("policy"),
                     "usd": round(usd.get(key, 0.0) or usd.get(f"{sid}-{src.name}", 0.0), 6)})
    return rows


async def stage_arms(ph: Phase, which: Sequence[str], tids: Sequence[str]) -> list[dict[str, Any]]:
    jobs = []
    a = ph.a
    for tid in tids:
        if "student" in which:
            jobs.append(ph.job(a.student, tid, k=a.k, seed=SEEDS["student"] + a.seed_offset, purpose="selection", concurrency=min(a.k, a.student_concurrency),
                               tag="student"))
        if "frontier" in which:
            for m in a.frontier:
                jobs.append(ph.job(m, tid, k=a.frontier_k, seed=SEEDS.get(m, 300) + a.seed_offset, purpose="fairness", concurrency=a.frontier_k, tag=arm_name(m)))
    with LoadMonitor(ph.out / "load.jsonl", every=a.load_every):
        return list(await asyncio.gather(*jobs))


async def stage_diag(ph: Phase, tids: Sequence[str]) -> list[dict[str, Any]]:
    """A budget diagnostic (never a keep-rule input): the frontier solvers on ``tids`` at this process's ``--tool-limit``
    (e.g. 300 against the env's 150), arms ``diag<limit>-<model>``, purpose ``smoke``. It answers whether a task every
    solver failed at the env's budget is solvable with more budget (the limits owner's question)."""
    a = ph.a
    jobs = [ph.job(m, tid, k=a.frontier_k, seed=SEEDS.get(m, 300) + 1, purpose="smoke", concurrency=a.frontier_k, tag=f"diag{a.tool_limit}-{arm_name(m)}")
            for tid in tids for m in a.frontier]
    with LoadMonitor(ph.out / "load.jsonl", every=a.load_every):
        return list(await asyncio.gather(*jobs))


async def stage_pilot(ph: Phase) -> dict[str, Any]:
    a = ph.a
    tid = a.pilot_task
    t0 = time.time()
    spent0 = ph.meter.spent
    with LoadMonitor(ph.out / "load.jsonl", every=a.load_every):
        res = await asyncio.gather(ph.job(a.student, tid, k=2, seed=SEEDS["student"], purpose="smoke", concurrency=2, tag="pilot-student"),
                                   ph.job(a.frontier[0], tid, k=1, seed=SEEDS[a.frontier[0]], purpose="smoke", concurrency=1,
                                          tag=f"pilot-{arm_name(a.frontier[0])}"))
    eps = {r["arm"]: episodes(ph.out, r["arm"], tid) for r in res if "arm" in r}
    by = {}
    for arm, rows in eps.items():
        live = [r for r in rows if not r["masked"]]
        by[arm] = {"episodes": len(rows), "masked": len(rows) - len(live), "passed": sum(r["passed"] for r in live),
                   "usd_per_episode": round(statistics.fmean([r["usd"] for r in rows]), 4) if rows else None,
                   "turns": [r["turns"] for r in rows], "tool_calls": [r["tool_calls"] for r in rows], "stops": [r["stop"] for r in rows],
                   "phi": [r["phi"] for r in rows], "wall_s_job": next((x.get("wall_s") for x in res if x.get("arm") == arm), None)}
    rep = {"task": tid, "settings": ph.settings(), "arms": by, "episodes": eps, "jobs": res, "spent": round(ph.meter.spent - spent0, 4),
           "wall_s": round(time.time() - t0, 1), "load": load_summary(ph.out / "load.jsonl", since=t0)}
    st = by.get("pilot-student", {}).get("usd_per_episode") or 0.0
    fr = by.get(f"pilot-{arm_name(a.frontier[0])}", {}).get("usd_per_episode") or 0.0
    n = len(ph.tasks.rows)
    # DeepSeek-V4.1-Flash costs about a third of the 397B per token (09 §2): projected at fr / 2.75
    rep["projection"] = {"tasks": n, "student_episodes": n * a.k, "frontier_episodes": n * a.frontier_k * len(a.frontier),
                         "usd": round(n * a.k * st + n * a.frontier_k * (fr + fr / 2.75), 2), "cap": a.cap,
                         "note": "audit and judge calls are extra (GLM-5.3 / -Flash, about 0.2 USD per failing trace)"}
    jsave(ph.out / "pilot.json", rep)
    print(json.dumps({k: rep[k] for k in ("arms", "spent", "wall_s", "load", "projection")}, indent=1, default=str), flush=True)
    return rep


# ---- the policy World fix (node) ------------------------------------------------------------------------------------------------
async def stage_worldfix(ph: Phase, tids: Sequence[str]) -> dict[str, Any]:
    """S15 found the node Worlds broken for the policy (World bugs, owner World): the World's file tools fail (no
    ``python3`` on node:22) and ``pnpm`` tries to download itself offline (corepack's cache under the build's ``HOME``).
    Per task with a profile fix (``runtime.policy_world``): build the derived policy World, and accept it only on executed
    evidence: the reproduction fails in the pinned World and passes in the fixed one (``python3`` present, the repo's
    ``pnpm`` answers offline, ``repo.read_file`` works and the existing tests run through the World's own shell), and the
    profile's leak probes pass on the fixed image. The grader keeps the pinned image. Accepted fixes are task World
    versions (``world_versions.jsonl``) and every later episode of the task runs in the fixed World."""
    import subprocess as sp

    from crucible.core.schema.actions import ToolCall
    from crucible.factory.generators.debug import profile_of, scope
    from crucible.factory.lanes.windows import new_test_names
    from crucible.factory.lanes.windows_git import RepoGit
    from crucible.factory.langs.runtime import policy_world

    out: dict[str, Any] = {}
    for tid in tids:
        task = ph.tasks.task(tid)
        prof = profile_of(task.lang)
        jp = ph.out / "worldfix" / f"{tid}.json"
        if (j := jload(jp)) and j.get("accepted") is not None and not ph.a.redo:
            out[tid] = j
            continue
        pinned = task.window["world"]["image"]
        pw = await asyncio.to_thread(policy_world, pinned, prof)
        if pw["image"] == pinned:
            continue
        j = {"task": tid, "pinned": pinned, "fixed": pw["image"], "dockerfile": pw["dockerfile"], "build_ok": pw["ok"], "repro": {}}
        # the World defects as the policy meets them: python3 for the file tools, the repo's pnpm answering offline, and (300
        # calls, h3) the repo's own ``pnpm test`` finishing offline without reaching for the registry (SCRIPT_OK)
        repro = ("command -v python3 >/dev/null && echo PY_OK || echo PY_MISSING; cd /w && ln -sfn /opt/deps/node_modules node_modules; "
                 "(timeout 60 pnpm --version 2>&1 || echo PNPM_FAIL) | tail -1; "
                 "if [ -f pnpm-lock.yaml ] && grep -q '\"test\"' package.json; then timeout 180 pnpm test > /tmp/st 2>&1; rc=$?; "
                 "if [ $rc = 124 ]; then echo SCRIPT_HANG; elif grep -q registry.npmjs.org /tmp/st; then echo SCRIPT_NET; else echo SCRIPT_OK rc=$rc; fi; "
                 "else echo SCRIPT_NA; fi")
        prior = jload(Path(ph.a.prior_worldfix) / f"{tid}.json") if ph.a.prior_worldfix else None
        images = [("pinned", pinned), *([("prior", prior["fixed"])] if prior and prior.get("accepted") and prior.get("fixed") not in (pinned, pw["image"])
                                        else []), ("fixed", pw["image"])]
        for name, im in images:
            r = await asyncio.to_thread(sp.run, ["docker", "run", "--rm", "--network", "none", "--name", f"{PREFIX}worldfix-{name}-{os.getpid()}", "-e",
                                                 "HOME=/tmp", "-v", f"{task.base_dir}:/src:ro", im, "sh", "-c", f"cp -a /src/. /w/ && {repro}"],
                                        capture_output=True, text=True, timeout=900)
            j["repro"][name] = r.stdout.strip().splitlines()[-3:]

        def works(lines: Sequence[str]) -> bool:
            txt = " ".join(lines)
            return "PY_OK" in txt and ("SCRIPT_OK" in txt or "SCRIPT_NA" in txt) and not any("Error" in x or "FAIL" in x or "Network" in x for x in lines)

        ok_pinned = works(j["repro"]["pinned"])
        ok_fixed = works(j["repro"]["fixed"])
        if "prior" in j["repro"]:
            j["prior"] = prior["fixed"]
            j["prior_ok"] = works(j["repro"]["prior"])  # the previous accepted fix: it must fail the same reproduction
            ok_pinned = ok_pinned or j["prior_ok"]
        # the World's own tools on the fixed image: read a file, run one existing test file through the repo's pnpm
        task.world_image = pw["image"]
        test_file = next((p for p in sorted(task.window.get("tests_b") or {}) if (task.base_dir / p).is_file()), None)
        if test_file is None:  # the hidden test files are new in B: any existing test file of the start tree
            from crucible.factory.lanes.windows import suite_targets

            with scope(task.lang):
                test_file = next(iter(sorted(suite_targets(task.base_dir, {}))), None)
        world = await task.open(ph.out / "worldfix" / f"{tid}-world")
        try:
            async def call(fn: str, _w: Any = world, **args: Any) -> Any:
                res = await _w.step("agent", ToolCall(tool_call_id=f"wf-{fn}", function_name=fn, arguments=args))
                return res.observation.tool_results[0] if res.observation.tool_results else None

            rd = await call("repo.read_file", path="package.json")
            j["read_file_ok"] = bool(getattr(rd, "ok", False))
            sh = await call("repo.shell", command=f"pnpm --version && timeout 300 node_modules/.bin/vitest run --reporter=dot {test_file} 2>&1 | tail -6" if test_file else "true", timeout_s=600)
            j["suite_tail"] = str(getattr(sh, "content", ""))[-600:]
            j["suite_ok"] = bool(re.search(r"\b\d+ passed", j["suite_tail"]))
        finally:
            await world.close()
            task.world_image = None
        with scope(task.lang):
            names = new_test_names(task.window, RepoGit(Path(task.window["repo_dir"]) if task.window.get("repo_dir") else ph.tasks.gen(ph.tasks.rows[tid]["split"])
                                                        .clones[task.window["repo"]]), task.base_dir)
            leak = await prof.leak_probes(pw["image"], task.base_dir, new_test_names=names, name_prefix=f"{PREFIX}leak-")
        j["leak"] = {"ok": leak.get("ok"), "probes": leak.get("probes")}
        j["accepted"] = bool(pw["ok"] and not ok_pinned and ok_fixed and j["read_file_ok"] and j["suite_ok"] and leak.get("ok"))
        j["why"] = "" if j["accepted"] else f"pinned_ok={ok_pinned} fixed_ok={ok_fixed} read_file={j['read_file_ok']} suite={j['suite_ok']} leak={leak.get('ok')}"
        jsave(jp, j)
        if j["accepted"]:
            with (ph.out / "world_versions.jsonl").open("a") as f:
                f.write(json.dumps({"task": tid, "kind": "world_fix", "owner": "world", "from": pinned, "to": pw["image"], "grader": pinned,
                                    "evidence": {k: j[k] for k in ("repro", "read_file_ok", "suite_ok", "leak")}, "at": time.time()}, default=str) + "\n")
        out[tid] = j
        print(f"[worldfix] {tid}: accepted={j['accepted']} {j['why']} repro={j['repro']}", flush=True)
    return out


# ---- the disclosure check ------------------------------------------------------------------------------------------------------
async def stage_disclose(ph: Phase, tids: Sequence[str]) -> dict[str, Any]:
    """``generators.debug.disclosure_check`` per task (local grading, no model call): which milestones fail when B lacks the
    window's silent units (reverted fixes without a milestone). The evidence goes to ``<out>/disclosure/<id>.json`` for
    review. A milestone that needs a fix the report never describes is a spec defect, recorded in ``world_issues.json``
    with ``kind`` spec_defect after a reviewer reads the report."""
    from crucible.factory.generators.debug import disclosure_check

    res = {}
    for tid in tids:
        jp = ph.out / "disclosure" / f"{tid}.json"
        if (j := jload(jp)) and not ph.a.redo:
            res[tid] = j
            continue
        j = await disclosure_check(ph.tasks.task(tid))
        jsave(jp, j)
        res[tid] = j
        print(f"[disclose] {tid}: silent {len(j['silent'])} measured {j['measured']} needs {sorted(j['needs'])} ({j['why']})", flush=True)
    return res


# ---- re-grading after a path-check fix ----------------------------------------------------------------------------------------
REGRADE_CHECKS = ("path:network", "path:install")


async def stage_regrade(ph: Phase, tids: Sequence[str]) -> dict[str, Any]:
    """Every finished (arm, task) whose episodes failed a trail path check (``REGRADE_CHECKS``) is re-graded under the
    current path checks (``factory.tinker.regrade``: the stored patch and trail, no model call). The S15 run found two false
    positives (a heredoc body read as commands, ``go mod init`` of a scratch module read as an install); an episode whose
    violation is real fails again."""
    from crucible.factory.tinker import regrade
    from crucible.observe.traces import attempt_dirs

    a = ph.a
    models = {"student": a.student, **{arm_name(m): m for m in a.frontier}}
    out: dict[str, Any] = {}
    for d in sorted((ph.out / "arms").glob("*/*/done.json")):
        arm, tid = d.parent.parent.name, d.parent.name
        if arm not in models or tid not in tids:
            continue
        j = jload(d, {}) or {}
        sids = set()
        for adir in attempt_dirs(d.parent / "rollouts"):
            rep = jload(adir / "verifier" / "report.json", {}) or {}
            if any(c.get("id") in REGRADE_CHECKS and c.get("status") == "fail" for c in ((rep.get("meta") or {}).get("path") or {}).get("checks") or []):
                sids.add(adir.parent.name)
        if not sids:
            continue
        print(f"[regrade] {arm} {tid}: {len(sids)} episode(s)", flush=True)
        res = await regrade([ph.tasks.task(tid)], model=models[arm], k=int(j["k"]), run_dir=d.parent, codec=ph.codec(models[arm]),
                            select=lambda r, s=frozenset(sids): str(r.attempt or "").rsplit("-a", 1)[0] in s, max_tokens=a.max_tokens,
                            enable_thinking=a.thinking, concurrency=int(j["k"]) if arm == "student" else a.frontier_k, seed=int(j["seed"]),
                            purpose=str(j["purpose"]), **ph.env_kw())
        out[f"{arm}/{tid}"] = {"samples": sorted(sids), **res}
        jsave(d.parent / "regrade.json", out[f"{arm}/{tid}"])
        print(f"[regrade] {arm} {tid}: {res}", flush=True)
    return out


# ---- the trace audit ---------------------------------------------------------------------------------------------------------
REPRO = r"\b(go\s+run|go\s+build|cargo\s+(run|build)|node\s|npx\s+(tsx|ts-node)|python3?\s|mvn\b)"
NON_SKILL = ("world_bug", "spec_defect", "unstated_format", "verifier_defect", "env_flake")
CONTROLS = Path(__file__).resolve().parents[1] / "docs" / "research" / "task-factory" / "runs" / "controls.json"


def controls() -> list[Any]:
    """The seeded control bank (P3/Phase A): every gating control rides in every labelled batch under an opaque id."""
    from crucible.factory.audit import SeededCase, Trace

    bank = json.loads(CONTROLS.read_text())["bank"] if CONTROLS.exists() else []
    return [SeededCase(Trace(c["id"], c["text"], c.get("verifier_pass", False), policy="control", failing_checks=list(c.get("failing_checks") or [])),
                       c["cause"], c.get("solution_correct"), prompt=c.get("prompt")) for c in bank if c.get("gating", True)]


def run_id_of(adir: Path) -> str:
    return adir.parents[2].name


def bundles_of(out: Path, arm: str, tid: str) -> list[Any]:
    from crucible.factory.traceaudit import load_bundle
    from crucible.observe.traces import attempt_dirs

    return [load_bundle(d, run=run_id_of(d)) for d in attempt_dirs(out / "arms" / arm / tid / "rollouts")]


def binding_of(ph: Phase, arm: str, tid: str) -> dict[str, Any]:
    """Per arm and task: how many failing episodes ran into the tool limit, and of those how many were still gaining Phi
    in the last 30% of their calls (``traceaudit.budget_binding``: the cap bound them) versus stuck."""
    from crucible.factory.traceaudit import budget_binding

    out: Counter[str] = Counter()
    rows = []
    for b in bundles_of(ph.out, arm, tid):
        v = budget_binding(b, tool_limit=ph.a.tool_limit)
        if v is not None:
            out[v["verdict"]] += 1
            rows.append({"episode": b.id, **v})
    return {"counts": dict(out), "episodes": rows}


def honest_of(task: Any, b: Any) -> list[Any]:
    import re

    from crucible.factory.generators.debug import scope
    from crucible.factory.traceaudit import ToolCall, commands_of, honest_trace_facts

    rec = task.window
    with scope(task.lang) as prof:
        hidden = [p for p in (rec.get("tests_b") or {}) if not (task.base_dir / p).exists()]
        fs = honest_trace_facts(b, requirement_milestones={r["id"]: r.get("milestone") for r in task.requirements},
                                protected=lambda p: prof.is_protected(p) and not prof.is_test_file(p), hidden_paths=hidden)
    first_edit = next((st.i for st in b.steps if re.search(r"write|edit|create|replace|patch", st.function, re.IGNORECASE)), None)
    from crucible.factory.traceaudit import TEST_RUN

    ran = [st.i for st in b.steps for _, c in commands_of([ToolCall(st.function, st.arguments)]) if TEST_RUN.search(c) or re.search(REPRO, c)]
    b.meta["ran_before_edit"] = bool(ran and (first_edit is None or min(ran) < first_edit))
    b.meta["test_runs"] = len(ran)
    return fs


async def stage_audit(ph: Phase, tids: Sequence[str], *, arm: str = "student") -> dict[str, Any]:
    """Every failing, unmasked student episode: facts, first wrong step, the honest-trace checks, then the cause labels of the
    GLM-5.3 auditor and the GLM-5.3-Flash judge (``audit.audit``: closed taxonomy, verbatim quotes, the seeded controls in every
    batch; a batch with a mislabelled control is untrusted and its labels are dropped). The family rule: ``ModelBook``
    refuses a judge of the builder's or the student's family at construction, and ``audit.audit`` refuses a trace whose
    policy shares the judge's family."""
    from types import SimpleNamespace

    from crucible.factory.audit import AuditConfig, Auditor, audit, cross_family_problems
    from crucible.factory.models import ModelBook
    from crucible.factory.traceaudit import deterministic_cause, facts, first_wrong_step, trace_of

    book = ModelBook()  # the default tinker map: auditor GLM-5.3, judge GLM-5.3-Flash (SameFamilyError on a violation)
    judges = {role: Auditor(book.client(role), model=book.spec(role).model, config=AuditConfig(concurrency=ph.a.audit_concurrency,
                                                                                                           quote_retries=ph.a.quote_retries),
                            max_tokens=ph.a.audit_max_tokens) for role in ("auditor", "judge")}
    bank = controls()
    out_dir = ph.out / "audit"
    res: dict[str, Any] = {}
    for tid in tids:
        jp = out_dir / f"{tid}.json"
        if (j := jload(jp)) and j.get("done") and not ph.a.redo:
            res[tid] = j
            continue
        task = ph.tasks.task(tid)
        bs = bundles_of(ph.out, arm, tid)
        if not bs:
            continue
        probs = [p for r in judges.values() for p in cross_family_problems(r.model, [ph.a.student, *(b.policy for b in bs)])]
        if probs:
            raise SystemExit("same-family judging refused: " + "; ".join(probs))
        rows: dict[str, Any] = {}
        traces = []
        peers_ok = [b for b in bs if b.passed]
        oracle_files = sorted(task.window.get("oracle_files") or {})
        for b in bs:
            fs = facts(b)
            fws = first_wrong_step(b, [p for p in peers_ok if p.id != b.id], oracle_files=oracle_files)
            hon = honest_of(task, b)
            det = deterministic_cause(fs)
            rows[b.id] = {"episode": b.id, "passed": b.passed, "masked": b.masked, "stop": b.stop, "reward": b.reward, "phi": b.phi_final,
                          "milestones": b.milestones, "steps": len(b.steps), "facts": [vars(f) for f in fs], "honest": [vars(f) for f in hon],
                          "first_wrong_step": vars(fws), "deterministic_cause": det, "ran_before_edit": b.meta.get("ran_before_edit"),
                          "test_runs": b.meta.get("test_runs"), "status_claims": b.status_claims}
            if not b.passed and not b.masked and det is None:
                traces.append(trace_of(b, [*fs, *hon], fws, result_chars=ph.a.result_chars))
        reqs = [SimpleNamespace(id=r["id"], text=r["text"]) for r in task.requirements]
        reports = {}
        for role, aud in judges.items():
            if not traces or ph.meter.exhausted:
                continue
            rep = await audit(traces, aud, prompt=task.instruction, requirements=reqs, seeded=bank, salt=f"s15-{tid}-{role}")
            reports[role] = {"model": aud.model, "trusted": rep.auditor_trusted, "seeded": rep.seeded, "mix": {k: v.k for k, v in rep.mix.items() if v.k},
                             "fp": rep.fp_candidates, "fn": rep.fn_candidates, "abstained": rep.abstained}
            for t_id, lab in rep.labels.items():
                rows[t_id].setdefault("labels", {})[role] = {**vars(lab), "trusted": rep.auditor_trusted is True}
        for r in rows.values():
            r["final"] = final_label(r)
        j = {"task": tid, "arm": arm, "done": True, "rows": rows, "reports": reports, "chars": [len(t.text) for t in traces]}
        jsave(jp, j)
        res[tid] = j
        print(f"[audit] {tid}: {len(traces)} labelled; " + "; ".join(f"{k} trusted={v['trusted']} mix={v['mix']}" for k, v in reports.items())
              + f" (spent {ph.meter.spent:.3f})", flush=True)
    return res


def final_label(r: Mapping[str, Any]) -> dict[str, Any]:
    """One cause per failing episode: a deterministic fact first (budget, env_flake); else the judge's (GLM-5.3-Flash)
    trusted label, cross-checked by the auditor's (GLM-5.3): when the auditor names a non-skill cause the judge did not, the
    episode is ``disputed`` and never counts as an on-intent skill gap (conservative: a dispute never raises a count)."""
    if r.get("passed") or r.get("masked"):
        return {"cause": None, "source": "passed" if r.get("passed") else "masked"}
    if r.get("deterministic_cause"):
        return {"cause": r["deterministic_cause"], "source": "facts"}
    labs = r.get("labels") or {}
    j, a = labs.get("judge") or {}, labs.get("auditor") or {}
    jc = j.get("cause") if j.get("trusted") and j.get("status") == "labelled" else None
    ac = a.get("cause") if a.get("trusted") and a.get("status") == "labelled" else None
    cause, src = (jc, "judge") if jc else ((ac, "auditor") if ac else (None, "unlabelled"))
    disputed = bool(jc and ac and jc != ac and (jc in NON_SKILL or ac in NON_SKILL))
    if cause == "skill_gap" and any(f.get("kind") in ("world_fault", "harness_misuse") for f in r.get("facts") or []):
        disputed = True  # a deterministic World or harness fact beside a skill label: never counted as the intended skill
    return {"cause": cause, "source": src, "judge": jc, "auditor": ac, "agree": None if not (jc and ac) else jc == ac, "disputed": disputed,
            "evidence": (j if src == "judge" else a).get("evidence") if cause else None}


# ---- failure mining ------------------------------------------------------------------------------------------------------------
def stage_mine(ph: Phase, tids: Sequence[str], *, arm: str = "student") -> dict[str, Any]:
    from crucible.factory.audit import AuditLabel
    from crucible.factory.failures import builder_intents, mine, write_jsonl
    from crucible.factory.generators.debug import failure_skill, task_intent

    findings = []
    for tid in tids:
        aj = jload(ph.out / "audit" / f"{tid}.json")
        if not aj:
            continue
        task = ph.tasks.task(tid)
        intent = (task.window.get("debug") or {}).get("intent") or task_intent(task.window)
        bs = bundles_of(ph.out, arm, tid)
        labels = {}
        for b in bs:
            fin = (aj["rows"].get(b.id) or {}).get("final") or {}
            if fin.get("cause") and fin.get("source") in ("judge", "auditor"):
                labels[b.id] = AuditLabel(b.id, fin["cause"], None, evidence=str(fin.get("evidence") or ""), status="labelled")
        for f in mine(bs, labels=labels):
            row = aj["rows"].get(f.trace) or {}
            f.honest = [h["kind"] for h in row.get("honest") or []]
            f.facts += row.get("honest") or []
            if (row.get("final") or {}).get("disputed"):
                f.cause_source = "disputed"
            sk = failure_skill(f.to_json(), intent, ran_before_edit=row.get("ran_before_edit"), honest=f.honest)
            f.skill = sk["skill"]
            row["skill"] = sk
            findings.append(f)
        jsave(ph.out / "audit" / f"{tid}.json", aj)
    # V9: a held-out (benchmark) task never informs what is built for training: its findings are kept (sealed, never
    # committed) but only training-split findings become builder-intent requests
    train = [f for f in findings if ph.tasks.rows.get(f.task, {}).get("split") == "train"]
    intents, _ = builder_intents(train, min_traces=2, prefix="bi-debug")
    _, routed = builder_intents(findings, min_traces=2, prefix="bi-debug")
    write_jsonl(ph.out / "findings.jsonl", findings)
    write_jsonl(ph.out / "builder_intents.jsonl", [{**i.to_json(), "intent_card": i.intent_card(surface="debugging, any language"),
                                                     "planner_intent": i.planner_intent()} for i in intents])
    write_jsonl(ph.out / "routed.jsonl", routed)
    print(f"[mine] {len(findings)} findings, {len(intents)} builder intents, {len(routed)} routed", flush=True)
    return {"findings": findings, "intents": intents, "routed": routed}


# ---- the report: keep rule, targeting records, W&B ------------------------------------------------------------------------------
def task_record(ph: Phase, tid: str) -> dict[str, Any]:
    from crucible.factory.targeting import band, cause_mix, keep_verdict, wilson

    a = ph.a
    row = ph.tasks.rows[tid]
    st = episodes(ph.out, "student", tid)
    live = [e for e in st if not e["masked"]]
    passes = sum(e["passed"] for e in live)
    ms_ids = [m["id"] for m in ph.tasks.task(tid).milestones]
    per_ms = {m: round(sum(bool((e["milestones"] or {}).get(m)) for e in live) / len(live), 4) if live else None for m in ms_ids}
    phis = [float(e["phi"]) for e in live if e["phi"] is not None]
    fr = {}
    for m in a.frontier:
        es = [e for e in episodes(ph.out, arm_name(m), tid) if not e["masked"]]
        if es:
            fr[m.split(":", 1)[0]] = [e["passed"] for e in es]
    aj = jload(ph.out / "audit" / f"{tid}.json", {}) or {}
    rows = aj.get("rows") or {}
    finals = [r.get("final") or {} for r in rows.values() if not r.get("passed") and not r.get("masked")]
    labels = Counter(f["cause"] for f in finals if f.get("cause"))
    on_intent = sum(1 for r in rows.values() if (r.get("final") or {}).get("cause") == "skill_gap" and not (r.get("final") or {}).get("disputed")
                    and (r.get("skill") or {}).get("on_intent"))
    cm = cause_mix(labels)
    # an open defect the trace review or an executed check found: a World defect (owner World), or with ``kind`` spec_defect
    # a task-spec defect (owner the builder: a hidden milestone needs a fault the report never describes)
    issue = (jload(ph.out / "world_issues.json", {}) or {}).get(tid)
    text = (issue or {}).get("issue") if isinstance(issue, dict) else issue
    spec = isinstance(issue, dict) and issue.get("kind") == "spec_defect" and issue.get("status", "open") == "open"
    world = bool(text) and not spec and (not isinstance(issue, dict) or issue.get("status", "open") == "open")
    keep = keep_verdict(split=row["split"], passes=passes, n=len(live), on_intent=on_intent, labelled=cm["n"], frontier=fr,
                        world_issue=text if world else None, spec_issue=text if spec else None)
    budget = labels.get("budget", 0)
    if budget and budget >= 0.3 * max(1, cm["n"]):
        # owner: the env's limits. A sensitivity number only (never the verdict): the on-intent share without the budget stops
        keep["flags"].append(f"budget-bound: {budget} of {cm['n']} labelled failures stopped at the tool limit ({a.tool_limit}); owner limits: "
                             f"re-measure at a larger limit (on-intent share without them {on_intent}/{max(1, cm['n'] - budget)})")
    honest = Counter(h["kind"] for r in rows.values() for h in r.get("honest") or [])
    modes = Counter()
    for f in (json.loads(x) for x in (ph.out / "findings.jsonl").read_text().splitlines() if x.strip()) if (ph.out / "findings.jsonl").exists() else []:
        if f["task"] == tid:
            modes[f"{f['mode']}" + (f"/{f['skill']}" if f.get("skill") else "")] += 1
    lo, hi = wilson(passes, len(live))
    binding = {arm: binding_of(ph, arm, tid)["counts"] for arm in ["student", *(arm_name(m) for m in a.frontier)]}
    return {"task": tid, "budget_binding": binding, "model": a.student, "lang": row["lang"], "repo": row["repo"], "split": row["split"], "faults": row["faults"], "level": row["level"],
            "skills": row.get("skills"), "n": len(st), "masked": len(st) - len(live), "passes": passes, "pass_rate": keep.get("pass_rate"),
            "pass_ci": [round(lo, 4), round(hi, 4)], "mean_phi": round(statistics.fmean(phis), 4) if phis else None, "per_milestone": per_ms,
            "band": band([e["reward"] for e in live]), "cause": {**cm, "on_intent": on_intent, "source": "GLM-5.3-Flash judge, GLM-5.3 auditor cross-check"},
            "fairness": fr, "keep": keep, "verdict": {"ok": keep["keep"], "route": None if keep["keep"] else keep["verdict"], "findings": [keep["why"]],
                                                       "flags": keep["flags"]},
            "honest": dict(honest), "modes": dict(modes.most_common()), "usd": round(sum(e["usd"] for e in st), 4),
            "usd_per_episode": round(statistics.fmean([e["usd"] for e in st]), 4) if st else None,
            "turns_mean": round(statistics.fmean([e["turns"] for e in st if e["turns"] is not None]), 1) if st else None,
            "stops": dict(Counter(str(e["stop"]) for e in st)), "settings": ph.settings()}


def stage_report(ph: Phase, tids: Sequence[str]) -> dict[str, Any]:
    recs = {t: task_record(ph, t) for t in tids if (ph.run_dir("student", t) / "done.json").exists()}
    spend = Counter()
    for ln in (ph.out / "spend.jsonl").read_text().splitlines() if (ph.out / "spend.jsonl").exists() else []:
        if ln.strip():
            r = json.loads(ln)
            spend[str(r.get("model"))] += float(r.get("usd") or 0.0)
    rep = {"settings": ph.settings(), "tasks": recs, "spend_by_model": {k: round(v, 4) for k, v in spend.items()}, "spend_total": round(sum(spend.values()), 4),
           "cap": ph.a.cap, "load": load_summary(ph.out / "load.jsonl"),
           "verdicts": dict(Counter(r["keep"]["verdict"] for r in recs.values()))}
    jsave(ph.out / "report.json", rep)
    base = ph.a.student.split(":", 1)[0]
    jsave(ph.out / "targeting.json", {t: {"model": base, "band": r["band"], "keep": r["keep"], "split": r["split"]} for t, r in recs.items()
                                      if r["keep"]["verdict"] == "train"})
    jsave(ph.out / "targeting_all.json", {t: {"model": base, "band": r["band"], "keep": r["keep"], "split": r["split"]} for t, r in recs.items()})
    if not ph.a.no_observe:
        rep["observe"] = observe_report(ph, recs, rep)
        jsave(ph.out / "report.json", rep)
    return rep


def observe_report(ph: Phase, recs: Mapping[str, Mapping[str, Any]], rep: Mapping[str, Any]) -> dict[str, Any]:
    """The phase in W&B (``crucible-factory``, offline until a key exists): targeting, the audit, findings attached to every
    episode's Weave-shaped trace, the builder intents, spend; ``scripts/observe.py sync`` uploads them."""
    from crucible.factory.failures import BuilderIntent, FailureFinding
    from crucible.observe import ObserveConfig
    from crucible.observe import factory_views as fv
    from crucible.observe.naming import factory_name, project, tags
    from crucible.observe.run import ObsRun
    from crucible.observe.traces import TraceStore

    cfg = ObserveConfig.from_env()
    proj = project("factory")
    store = TraceStore(cfg.root, proj)
    n_tr = 0
    for arm in ["student", *(arm_name(m) for m in ph.a.frontier)]:
        for tid in recs:
            ro = ph.out / "arms" / arm / tid / "rollouts"
            if ro.exists() and not ph.a.obs_rev:
                n_tr += len(store.export_all(ro, arm=arm, run=f"s15-t{ph.a.tool_limit}-{arm}", model=ph.a.student if arm == "student" else arm,
                                             attrs={"phase": "S15", "lane": "debug", "task": tid, "tool_limit": ph.a.tool_limit}, max_chars=cfg.max_chars))
    finds = [FailureFinding(**{k: v for k, v in json.loads(x).items() if k != "routed"}) for x in (ph.out / "findings.jsonl").read_text().splitlines()
             if x.strip()] if (ph.out / "findings.jsonl").exists() else []
    intents = []
    for x in (ph.out / "builder_intents.jsonl").read_text().splitlines() if (ph.out / "builder_intents.jsonl").exists() else []:
        if x.strip():
            d = json.loads(x)
            intents.append(BuilderIntent(**{k: v for k, v in d.items() if k in BuilderIntent.__dataclass_fields__}))
    import hashlib

    ident = hashlib.sha256(json.dumps([sorted(recs), rep["settings"], ph.a.obs_rev], sort_keys=True, default=str).encode()).hexdigest()
    name = factory_name("debug", f"s15-t{ph.a.tool_limit}", ident[:8])
    tg = tags(model=ph.a.student, phase="S15", lane="debug", lang=sorted({r["lang"] for r in recs.values()}), tool_limit=ph.a.tool_limit)
    with ObsRun(proj, name, group="lane-debug", job_type="targeting", cfg=cfg, config={"settings": rep["settings"], "tasks": sorted(recs)}, tags=tg) as run:
        fv.targeting(run, recs, model=ph.a.student)
        run.table("s15/tasks", ["task", "lang", "split", "faults", "n", "masked", "passes", "pass_rate", "ci_lo", "ci_hi", "mean_phi", "on_intent", "labelled",
                                "fair", "verdict", "why", "usd", "usd_per_episode"],
                  [[t, r["lang"], r["split"], r["faults"], r["n"], r["masked"], r["passes"], r["pass_rate"], r["pass_ci"][0], r["pass_ci"][1], r["mean_phi"],
                    r["keep"]["on_intent"], r["keep"]["labelled"], r["keep"]["fair"], r["keep"]["verdict"], r["keep"]["why"], r["usd"], r["usd_per_episode"]]
                   for t, r in sorted(recs.items())])
        run.table("s15/milestones", ["task", "milestone", "pass_rate"], [[t, m, v] for t, r in sorted(recs.items()) for m, v in r["per_milestone"].items()])
        from crucible.factory.targeting import wilson

        arm_rows = []
        for arm in ["student", *(arm_name(m) for m in ph.a.frontier)]:
            for t in sorted(recs):
                es = [e for e in episodes(ph.out, arm, t) if not e["masked"]]
                if es:
                    k_, n_ = sum(e["passed"] for e in es), len(es)
                    lo, hi = wilson(k_, n_)
                    phis = [float(e["phi"]) for e in es if e["phi"] is not None]
                    arm_rows.append([arm, t, k_, n_, round(k_ / n_, 4), round(lo, 4), round(hi, 4), round(statistics.fmean(phis), 4) if phis else None,
                                     round(sum(e["usd"] for e in es), 4), sum(1 for e in es if str(e["stop"]) == "max_steps")])
        run.table("s15/arms", ["arm", "task", "passes", "n", "pass@1", "ci_lo", "ci_hi", "mean_phi", "usd", "tool_limit_stops"], arm_rows)
        run.table("s15/budget_binding", ["task", "arm", "binding", "stuck", "unmeasured"],
                  [[t, arm, c.get("binding", 0), c.get("stuck", 0), c.get("unmeasured", 0)] for t, r in sorted(recs.items())
                   for arm, c in (r.get("budget_binding") or {}).items()])
        run.table("s15/honest", ["task", "check", "episodes"], [[t, k, v] for t, r in sorted(recs.items()) for k, v in r["honest"].items()])
        audit_rep = audit_rollup(ph, recs)
        fv.audit_summary(run, audit_rep)
        # a revision run (``--obs-rev``) re-logs the tables only: the episodes' traces and their audits are already in Weave
        fv.findings(run, finds, store=store if not ph.a.obs_rev else None)
        if ph.a.obs_rev:
            idx = store.index()
            run.table("findings/trace_links", ["finding", "episode", "trace"], [[f.id, f.trace, (idx.get(f.trace) or {}).get("url")] for f in finds])
        fv.intent_cards(run, intents)
        fv.spend(run, rep["spend_by_model"])
        run.summary({"s15/verdicts": json.dumps(rep["verdicts"]), "s15/spend_total": rep["spend_total"], "s15/traces": n_tr, "s15/tool_limit": ph.a.tool_limit,
                     "s15/env_budget": json.dumps(rep["settings"].get("env_budget"))})
    return {"run": str(run.dir), "traces": n_tr, "project": proj, "name": name}


def audit_rollup(ph: Phase, recs: Mapping[str, Any]) -> dict[str, Any]:
    facts_eps: Counter[str] = Counter()
    fws: Counter[str] = Counter()
    det: Counter[str] = Counter()
    lab: Counter[str] = Counter()
    n = failing = passed = masked = 0
    agree = disputed = both = 0
    for tid in recs:
        aj = jload(ph.out / "audit" / f"{tid}.json", {}) or {}
        for r in (aj.get("rows") or {}).values():
            n += 1
            passed += bool(r.get("passed"))
            masked += bool(r.get("masked"))
            if r.get("passed") or r.get("masked"):
                continue
            failing += 1
            for k in {f["kind"] for f in [*(r.get("facts") or []), *(r.get("honest") or [])]}:
                facts_eps[k] += 1
            fws[(r.get("first_wrong_step") or {}).get("kind") or "none"] += 1
            fin = r.get("final") or {}
            if fin.get("source") == "facts":
                det[fin["cause"]] += 1
            elif fin.get("cause"):
                lab[fin["cause"]] += 1
            if fin.get("agree") is not None:
                both += 1
                agree += bool(fin["agree"])
            disputed += bool(fin.get("disputed"))
    return {"bundles": n, "failing": failing, "passed": passed, "masked": masked, "facts_episodes": dict(facts_eps), "first_wrong_step": {"kinds": dict(fws)},
            "causes": {"deterministic": dict(det), "labelled_trusted": dict(lab),
                       "skill_gap_share": round(lab.get("skill_gap", 0) / max(1, sum(lab.values()) + sum(det.values())), 3)},
            "auditor": {"model": "zai-org/GLM-5.3:peft:262144 (auditor) + zai-org/GLM-5.3-Flash:peft:262144 (judge)", "trusted_labels": sum(lab.values()),
                        "judge_auditor_agreement": f"{agree}/{both}", "disputed": disputed}}


def sealed(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The committed copy of a summary (V9: held-out evidence stays under ``data/targeting``): a held-out task keeps its
    numbers and verdict, but its keep rule's ``why`` and ``flags`` and its issue text (which can name hidden tests or the
    reverted fixes) are replaced by the verdict and the issue's kind."""
    out = json.loads(json.dumps(summary, default=str))
    held = {r["task"] for r in out.get("tasks") or [] if r.get("split") == "heldout"}
    for r in out.get("tasks") or []:
        if r["task"] in held:
            r["keep"] = {**r["keep"], "why": f"{r['keep'].get('verdict')} (held out: details sealed under data/targeting)", "flags": []}
    # an issue's text quotes the window (hidden tests, reverted fixes): the committed copy keeps what and whose, never the text
    out["issues"] = {t: {k: v for k, v in (i.items() if isinstance(i, Mapping) else []) if k in ("kind", "status", "owner", "date")}
                     for t, i in (out.get("issues") or {}).items()}
    return out


def stage_summary(ph: Phase, *, docs: Path | None = None) -> dict[str, Any]:
    """The committed summary (``docs/research/swarm-pipeline/runs/s15_debug_targeting.json``) and a markdown table: per
    task, settings, spend, $ and wall time per episode, honest-trace counts, routed failures, builder intents."""
    rep = jload(ph.out / "report.json", {}) or {}
    prep = (jload(Path(ph.a.compare) / "report.json", {}) or {}) if ph.a.compare else {}
    prev = prep.get("tasks") or {}
    lim = ph.a.tool_limit
    rows = []
    cmp_head = f" at {(prep.get('settings') or {}).get('tool_limit')} calls (student; frontier; verdict) |" if prev else ""
    md = [(f"| task | lang | split | faults | student pass@1 (k) [95% CI] | mean Φ | per milestone | frontier (397B; DS-Flash) | budget stops "
           f"(student; frontier) | on-intent skill gap | dominant failure modes | verdict |{cmp_head}"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if prev else "")]
    budget = {"max_steps", "timeout", "context_overflow", "budget", "max_tokens"}

    def stops(arm: str, t: str) -> int:
        return sum(1 for e in episodes(ph.out, arm, t) if str(e["stop"]) in budget)

    for t, r in sorted((rep.get("tasks") or {}).items()):
        k = r["keep"]
        fr = r.get("fairness") or {}
        frs = "; ".join(f"{sum(v)}/{len(v)}" for _m, v in sorted(fr.items(), key=lambda kv: "deepseek" in kv[0])) or "not run"
        modes = ", ".join(f"{m} {n}" for m, n in list((r.get("modes") or {}).items())[:3]) or "-"
        pm = " ".join(f"{m}:{v:.2f}" for m, v in (r.get("per_milestone") or {}).items() if v is not None)
        share = f"{k['on_intent']}/{k['labelled']}" + (f" ({k['intent_share']:.2f})" if k.get("intent_share") is not None else "")
        bst = {"student": stops("student", t), "frontier": sum(stops(arm_name(m), t) for m in ph.a.frontier)}
        nfr = sum(len(v) for v in fr.values())
        old = prev.get(t)
        oldc = ""
        if prev:
            ofr = "; ".join(f"{sum(v)}/{len(v)}" for _m, v in sorted((old or {}).get("fairness", {}).items(), key=lambda kv: "deepseek" in kv[0])) if old else ""
            oldc = (f" {old['passes']}/{old['n'] - old['masked']}; {ofr or 'not run'}; {old['keep']['verdict']} |" if old else " not measured |")
        md.append(f"| {t.removeprefix('db-')} | {r['lang']} | {r['split']} | {r['faults']} | {r['passes']}/{r['n'] - r['masked']} = {r['pass_rate']} "
                  f"[{r['pass_ci'][0]:.2f}, {r['pass_ci'][1]:.2f}] | {r['mean_phi']} | {pm} | {frs} | {bst['student']}/{r['n']}; {bst['frontier']}/{nfr} | "
                  f"{share} | {modes} | **{k['verdict']}** |{oldc}")
        rows.append({"task": t, "lang": r["lang"], "split": r["split"], "faults": r["faults"], "level": r["level"], "n": r["n"], "masked": r["masked"],
                     "passes": r["passes"], "pass_rate": r["pass_rate"], "pass_ci": r["pass_ci"], "mean_phi": r["mean_phi"], "per_milestone": r["per_milestone"],
                     "frontier": fr, "cause": r["cause"], "keep": {x: k.get(x) for x in ("verdict", "keep", "why", "flags", "on_intent", "labelled", "intent_share",
                                                                                         "fair")},
                     "honest": r.get("honest"), "modes": r.get("modes"), "stops": r.get("stops"), "usd": r["usd"], "usd_per_episode": r["usd_per_episode"],
                     "turns_mean": r.get("turns_mean"), "budget_binding": r.get("budget_binding"), "budget_stops": bst, "tool_limit": lim,
                     "previous": None if not old else {"tool_limit": (old.get("settings") or {}).get("tool_limit"), "passes": old["passes"],
                                                       "n": old["n"] - old["masked"], "mean_phi": old["mean_phi"], "frontier": old.get("fairness"),
                                                       "verdict": old["keep"]["verdict"]}})
    intents = [json.loads(x) for x in (ph.out / "builder_intents.jsonl").read_text().splitlines() if x.strip()] if (ph.out / "builder_intents.jsonl").exists() else []
    routed = [json.loads(x) for x in (ph.out / "routed.jsonl").read_text().splitlines() if x.strip()] if (ph.out / "routed.jsonl").exists() else []
    eps = [e for d in (ph.out / "arms").glob("*/*/done.json") if not d.parent.parent.name.startswith("pilot")
           for e in episodes(ph.out, d.parent.parent.name, d.parent.name)]
    per_arm: dict[str, Any] = {}
    for arm in sorted({e["arm"] for e in eps}):
        es = [e for e in eps if e["arm"] == arm]
        per_arm[arm] = {"episodes": len(es), "masked": sum(e["masked"] for e in es), "passes": sum(e["passed"] for e in es),
                        "usd": round(sum(e["usd"] for e in es), 3), "usd_per_episode": round(statistics.fmean([e["usd"] for e in es]), 4),
                        "wall_s_per_episode": round(statistics.fmean([e["wall_s"] for e in es if e["wall_s"]]), 1) if any(e["wall_s"] for e in es) else None,
                        "tool_calls_mean": round(statistics.fmean([e["tool_calls"] for e in es]), 1), "stops": dict(Counter(str(e["stop"]) for e in es))}
    out = {"phase": "S15 targeting, debug lane", "date": time.strftime("%Y-%m-%d"), "settings": rep.get("settings"), "tasks": rows, "per_arm": per_arm,
           "spend_by_model": rep.get("spend_by_model"), "spend_total": rep.get("spend_total"), "cap": rep.get("cap"), "verdicts": rep.get("verdicts"),
           "load": rep.get("load"), "builder_intents": [{k: i.get(k) for k in ("id", "student", "mode", "behaviour", "failure_mode", "n", "tasks", "skills",
                                                                                 "quotes", "status")} for i in intents],
           "carried_over": {t: {"verdict": o["keep"]["verdict"], "passes": o["passes"], "n": o["n"] - o["masked"], "tool_limit": (o.get("settings") or {})
                                .get("tool_limit")} for t, o in sorted(prev.items()) if t not in (rep.get("tasks") or {})},
           "issues": jload(ph.out / "world_issues.json", {}) or {},
           "routed": dict(Counter(f"{r['owner']}:{r['mode']}:{r['cause']}" for r in routed)), "regrade": {str(p.parent.parent.name) + "/" + p.parent.name: jload(p)
                                                                                                        for p in (ph.out / "arms").glob("*/*/regrade.json")}}
    jsave(ph.out / "summary.json", out)
    (ph.out / "summary.md").write_text("\n".join(md) + "\n")
    if docs is not None:
        # the 150-call run kept the plain name; a phase at another budget gets its own file
        jsave(docs / ("s15_debug_targeting.json" if ph.a.tool_limit == 150 else f"s15_debug_targeting_t{ph.a.tool_limit}.json"), sealed(out))
    print("\n".join(md), flush=True)
    return out


def cleanup() -> None:
    r = subprocess.run(["docker", "ps", "-a", "--filter", f"name={PREFIX}", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
    names = [x for x in r.stdout.split() if x.startswith(PREFIX)]
    if names:
        subprocess.run(["docker", "rm", "-f", *names], capture_output=True, check=False)
    print(f"removed {len(names)} {PREFIX}* containers")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from crucible.factory.generators.debug import ENV_BUDGET

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--debug", default="", help="the debug lane's data dir (default <data>/debug)")
    ap.add_argument("--cap", type=float, default=10.0, help="hard USD cap of the whole phase (one meter: spend.jsonl)")
    ap.add_argument("--tasks", default="", help="comma list (default: every candidate)")
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--frontier", default=",".join(FRONTIER))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0, help="added to every arm's run seed: a re-measurement draws fresh samples and its run ids "
                    "(bench-<purpose>-s<seed>-<task>, the Weave trace keys) never collide with an earlier phase's")
    ap.add_argument("--frontier-k", type=int, default=2)
    ap.add_argument("--student-concurrency", type=int, default=8)
    ap.add_argument("--slots", type=int, default=10, help="Worlds open at once on this host")
    ap.add_argument("--grade-parallel", type=int, default=2, help="grading containers at once per World image")
    ap.add_argument("--tool-limit", type=int, default=ENV_BUDGET.tool_limit, help="the lane's env budget (generators.debug.ENV_BUDGET)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="per reply (factory.student.STUDENT_MAX_TOKENS)")
    ap.add_argument("--thinking", action="store_true", help="enable_thinking (default off; recorded)")
    ap.add_argument("--agent-s", type=float, default=ENV_BUDGET.agent_s)
    ap.add_argument("--rollout-s", type=float, default=ENV_BUDGET.rollout_s)
    ap.add_argument("--ckpt-every", type=int, default=ENV_BUDGET.checkpoint_every)
    ap.add_argument("--ckpt-cap", type=int, default=ENV_BUDGET.checkpoint_cap)
    ap.add_argument("--load-every", type=float, default=60.0)
    ap.add_argument("--pilot-task", default=PILOT_TASK)
    ap.add_argument("--est-usd", default="", help="model=usd per episode for job admission (default: measured, else the pilot's)")
    ap.add_argument("--audit-reserve", type=float, default=25.0, help="USD kept for the audit when admitting episode jobs")
    ap.add_argument("--audit-concurrency", type=int, default=6)
    ap.add_argument("--audit-max-tokens", type=int, default=3000)
    ap.add_argument("--result-chars", type=int, default=1500, help="per tool result in the audited trace")
    ap.add_argument("--quote-retries", type=int, default=1, help="audit.AuditConfig.quote_retries: a non-verbatim quote is asked again")
    ap.add_argument("--prior-worldfix", default="", help="worldfix: a previous phase's worldfix dir; its accepted World must fail the reproduction")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-observe", action="store_true")
    ap.add_argument("--obs-rev", type=int, default=0, help="report: a revision W&B run (tables only; traces and audits are already logged)")
    ap.add_argument("--compare", default="", help="summary: a previous phase's out dir; its result per task sits beside this one's")
    ap.add_argument("--docs", default="", help="summary: also write runs/s15_debug_targeting.json here (the committed copy)")
    a = ap.parse_args(argv)
    d = data_dir()
    a.out = a.out or str(d / "targeting" / "debug")
    a.debug = a.debug or str(d / "debug")
    a.frontier = [m for m in a.frontier.split(",") if m]
    return a


async def amain(a: argparse.Namespace) -> None:
    if any(s in a.stages for s in ("pilot", "student", "frontier")) and not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . ../.env; set +a)")
    ph = Phase(a)
    tids = [t for t in a.tasks.split(",") if t] or sorted(ph.tasks.rows)
    try:
        for st in a.stages:
            if st == "pilot":
                await stage_pilot(ph)
            elif st in ("student", "frontier", "arms"):
                await stage_arms(ph, ("student", "frontier") if st == "arms" else (st,), tids)
            elif st == "worldfix":
                await stage_worldfix(ph, tids)
            elif st == "diag":
                await stage_diag(ph, tids)
            elif st == "disclose":
                await stage_disclose(ph, tids)
            elif st == "regrade":
                await stage_regrade(ph, tids)
            elif st == "audit":
                await stage_audit(ph, tids)
            elif st == "mine":
                stage_mine(ph, tids)
            elif st == "report":
                r = stage_report(ph, tids)
                print(json.dumps({"verdicts": r["verdicts"], "spend_total": r["spend_total"], "spend_by_model": r["spend_by_model"]}, indent=1), flush=True)
            elif st == "summary":
                stage_summary(ph, docs=Path(a.docs) if a.docs else None)
            elif st == "cleanup":
                cleanup()
            else:
                raise SystemExit(f"unknown stage {st}")
    finally:
        print(f"spent {ph.meter.spent:.4f} of {ph.meter.cap_usd}", flush=True)


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
