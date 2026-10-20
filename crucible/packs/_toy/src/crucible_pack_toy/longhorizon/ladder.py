"""The long-horizon step ladder (M16 design §4): the backlog at ~100 / 500 / 1,000 / 2,500 / 5,000 World calls, worked by
scripted RLM models that keep no hidden memory (``crucible.runtime.scripted_rlm``), through the M10 ``Runner`` with
``rollout_mode="train"`` (the records a trainer would read).

Each rung is one episode, and its report says what the harness cost and what the model lost:

- ``reward`` (Φ at the end, the share of tickets correct) and ``phi_segments``: Φ at each segment boundary and ΔΦ per
  segment (milestone Φ from the runner's ``progress.jsonl``), so the reward lost after a compaction is visible;
- ``constraints``: every refund and status change the agent made, judged by the rule in force when it was made (FIN-7:
  an approval above the threshold of the time, one refund per order, awaiting tickets untouched before their reply);
  ``lost_constraint_rate`` = rule-breaking actions after the first compaction / rule-bearing actions after it, with the
  rate before it as the control; ``missed_awaiting`` = awaiting tickets never resolved;
- ``compactions``, ``segments`` (calls, entered tokens, largest prompt, all with their ``source``: ``tito`` = the codec's
  real token ids, ``estimate`` = chars/4), ``b1``: the largest real prompt against the window and the trigger;
- ``rereads``: cells after a compaction that read history back (the model's own cell kinds; a real model is counted from
  the ledger when the harness has one);
- ``wall_s``, ms per World call, ``restore``: the time to rebuild the final kernel (journal replay today; a snapshot
  restore when the harness offers one), ``checkpoints``: bytes per save and in all.

The harness is a hook: ``LadderConfig.rlm`` holds ``RLMConfig`` fields by name, and fields the installed ``RLMConfig``
does not have yet are reported as ``ignored`` (so the new harness's ``window_tokens``, ``tail_tokens``, ``seq_len``,
``snapshot_every`` ... apply as soon as they exist). ``make_harness`` can be replaced wholesale.

    python scripts/lh_ladder.py --rungs 100,500,1000 --variants faithful,forgetful --budget 32000 --out report.json
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible_pack_toy.longhorizon.task import BacklogTask
from crucible_pack_toy.longhorizon.world import Backlog

# tickets per rung, calibrated so the faithful model makes about that many World calls (seed 0; ~6 calls per ticket,
# the policy read and the pending listing first). ``tests/unit/test_m16_e_ladder.py`` checks the small rungs.
RUNGS: dict[int, int] = {100: 15, 500: 80, 1000: 160, 2500: 410, 5000: 820}
FIRST_SEED = 0


@dataclass
class LadderConfig:
    rung: int = 100  # target World calls (a key of RUNGS) or, with ``tickets``, a label
    tickets: int | None = None
    seed: int = FIRST_SEED
    variant: str = "faithful"  # a scripted_rlm.BACKLOG_VARIANTS key
    harness: str = "rlm"  # rlm | native (the native harness replays the diligent plan: a reference, not a no-memory model)
    tito: str | None = "qwen"  # qwen (the real Qwen2.5 tokenizer) | byte | None (estimated usage, no token records)
    kernel: str = "docker"
    budget_tokens: int = 32_000
    trigger: float = 0.8
    keep_recent_turns: int = 3
    summary_max_tokens: int = 2_000
    enforce_window: bool = False  # the engine refuses a prompt longer than budget_tokens (ContextOverflow), as a served model would
    checkpoint_every: int = 500
    max_steps: int = 1_000_000
    timeout_s: float | None = None
    restore_probe: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)  # BacklogRLM knobs (per_cell, print_chars, replies, ...)
    # faults on attempt 1 (``scripted_rlm`` helpers): crash_at_call (a lost worker: masked, resumed from the last checkpoint),
    # overflow_at [calls], overflow_summaries (the first k compaction calls overflow), silent (silent overflows),
    # kill_at_world_call (the kernel is killed mid-cell at that World call)
    faults: dict[str, Any] = field(default_factory=dict)
    rlm: dict[str, Any] = field(default_factory=dict)  # extra RLMConfig fields (the new harness's, when they exist)

    @property
    def n(self) -> int:
        return self.tickets if self.tickets is not None else RUNGS[self.rung]

    def label(self) -> str:
        return f"r{self.rung}-n{self.n}-s{self.seed}-{self.harness}-{self.variant}-b{self.budget_tokens}"


# ---- the harness hook --------------------------------------------------------------------------------------------------

def rlm_config(cfg: LadderConfig) -> tuple[Any, dict[str, Any]]:
    """The ``RLMConfig`` for a rung and what happened to ``cfg.rlm``: {"applied": {...}, "ignored": {...}}."""
    from crucible.runtime.harness.rlm import RLMConfig

    names = {f.name for f in dataclasses.fields(RLMConfig)}
    base = {"max_tokens": 4_000, "temperature": 0.0, "max_turns": 1_000_000, "kernel": cfg.kernel, "allow_host": cfg.kernel == "local",
            "budget_tokens": cfg.budget_tokens, "trigger": cfg.trigger, "keep_recent_turns": cfg.keep_recent_turns,
            "summary_max_tokens": cfg.summary_max_tokens}
    applied = {k: v for k, v in cfg.rlm.items() if k in names}
    ignored = {k: v for k, v in cfg.rlm.items() if k not in names}
    return RLMConfig(**{k: v for k, v in {**base, **applied}.items() if k in names}), {"applied": applied, "ignored": ignored}


class _Capture:
    """Keeps the policies a harness built (the RLM policy is made inside the harness) and the faults installed on them."""

    def __init__(self):
        self.policies: list[Any] = []
        self.kills: list[dict[str, Any]] = []


def make_harness(cfg: LadderConfig, capture: _Capture) -> tuple[Any, dict[str, Any]]:
    from crucible.actors.policy import EpisodeLimits
    from crucible.runtime.harness.native import NativeHarness
    from crucible.runtime.harness.rlm import RLMHarness

    limits = EpisodeLimits(max_steps=cfg.max_steps, max_tool_errors=None, timeout_s=cfg.timeout_s)
    if cfg.harness == "native":
        return NativeHarness(limits=limits, checkpoint_every=cfg.checkpoint_every), {"applied": {}, "ignored": dict(cfg.rlm)}
    rc, how = rlm_config(cfg)

    class Harness(RLMHarness):
        def wrap(self, attempt, actor, x):
            p = super().wrap(attempt, actor, x)
            capture.policies.append(p)
            at = cfg.faults.get("kill_at_world_call")
            if at and attempt.n == 1 and hasattr(p, "_host"):
                from crucible.runtime.scripted_rlm import kill_mid_cell

                capture.kills.append(kill_mid_cell(p, at_world_call=int(at)))
            return p

    return Harness(rc, limits=limits, checkpoint_every=cfg.checkpoint_every), {**how, "config": dataclasses.asdict(rc), "id": None}


class WindowedLLM:
    """A served model's window: a prompt longer than ``window`` real tokens raises ``ContextOverflow`` (TITO only)."""

    def __init__(self, inner: Any, window: int):
        self.inner, self.window = inner, window
        self.spend = getattr(inner, "spend", None)
        self.refused = 0
        gen = inner.engine.gen

        async def guarded(prompt, max_tokens, temperature, **kw):
            from crucible.actors.llm import ContextOverflow

            if len(prompt) > self.window:
                self.refused += 1
                raise ContextOverflow(f"prompt of {len(prompt)} tokens exceeds the {self.window}-token context window")
            return await gen(prompt, max_tokens, temperature, **kw)

        inner.engine.gen = guarded

    async def complete(self, messages, **kw):
        return await self.inner.complete(messages, **kw)


def make_model(cfg: LadderConfig, codec: Any = None) -> tuple[Any, Callable[[], Any]]:
    """(the scripted model, a factory of the LLM client the harness gets: one per attempt, as a restarted worker has no
    conversation; the model itself is stateless)."""
    from crucible.runtime.scripted_rlm import backlog_model
    from crucible.train.chat import ByteChatCodec
    from crucible.train.tito_scripted import TitoLLM, qwen_codec

    model = backlog_model(cfg.variant, **cfg.model_kwargs)
    if cfg.tito is None:
        return model, lambda: model
    codec = codec or (qwen_codec() if cfg.tito == "qwen" else ByteChatCodec())

    def client() -> Any:
        llm: Any = TitoLLM(model, codec)
        return WindowedLLM(llm, cfg.budget_tokens) if cfg.enforce_window else llm

    return model, client


def with_faults(cfg: LadderConfig, llm: Any, attempt: Any, fired: list[Any]) -> Any:
    from crucible.runtime.scripted_rlm import FaultLLM

    f = cfg.faults
    if getattr(attempt, "n", 1) != 1 or not (f.get("crash_at_call") is not None or f.get("overflow_at") or f.get("overflow_summaries")):
        return llm
    x = FaultLLM(llm, crash_at=f.get("crash_at_call"), overflow_at=tuple(f.get("overflow_at") or ()),
                 overflow_when=(lambda v: v.role == "summary") if f.get("overflow_summaries") else None, overflow_times=f.get("overflow_summaries"),
                 silent=bool(f.get("silent")), window=cfg.budget_tokens)
    fired.append(x)
    return x


# ---- checkpoint accounting ---------------------------------------------------------------------------------------------

@contextlib.contextmanager
def checkpoint_meter():
    """Times and sizes every checkpoint save while active (the save itself is unchanged)."""
    from crucible.runtime import checkpoint as ck

    saves: list[dict[str, Any]] = []
    orig = ck.Checkpointer.save

    async def save(self, step):
        t0 = time.perf_counter()
        await orig(self, step)
        p = Path(self.dir) / ck.NAME
        saves.append({"step": step, "save_s": round(time.perf_counter() - t0, 4), "bytes": p.stat().st_size if p.exists() else None})

    ck.Checkpointer.save = save
    try:
        yield saves
    finally:
        ck.Checkpointer.save = orig


# ---- metrics from the World trace ----------------------------------------------------------------------------------------

def _norm(ts: str | None) -> str:
    return (ts or "").replace("Z", "").split("+")[0][:19]


def _agent_action(e: Any) -> bool:
    return e.actor == "agent" and (e.kind == "action" and e.type == "tool_call" or e.kind == "message" and e.type == "message")


def constraint_audit(events: list[Any], b: Backlog, boundaries: list[int]) -> dict[str, Any]:
    """Every refund and status change of the agent judged by the rule in force when it was made (FIN-7 rules 1, 3, 6, 8),
    assigned to the segment its World step falls in (``boundaries``: the World steps at which compactions happened)."""
    from crucible_pack_toy.forge.task import ground_refunds

    grounded = {g["seq"]: g for g in ground_refunds(events, "supervisor", skip_failed=True)}
    tickets = {t.id: t for t in b.tickets}
    refunded = {r["order_id"] for r in b.prior_refunds}
    fired: dict[str, str] = {}
    results = {(e.data or {}).get("id"): bool((e.data or {}).get("ok")) for e in events if e.type == "tool_result"}
    rows: list[dict[str, Any]] = []
    step = 0
    for e in events:
        d = e.data or {}
        if e.type == "event:nl" and str(d.get("event", "")).startswith("followup_"):
            fired.setdefault(d["event"], _norm(e.time))
        if not _agent_action(e):
            continue
        step += 1
        if e.type != "tool_call" or not results.get(d.get("id")):
            continue
        fn, args = d.get("function"), d.get("arguments") or {}
        if fn not in ("tickets.issue_refund", "tickets.set_status"):
            continue
        t = tickets.get(args.get("ticket_id"))
        broke: list[str] = []
        if t is not None and t.follow_up and t.follow_up["id"] not in fired:
            broke.append("acted_before_reply")
        if fn == "tickets.issue_refund":
            try:
                amt = float(args.get("amount_usd"))
            except (TypeError, ValueError):
                amt = float("nan")
            need = amt > b.threshold_at(_norm(e.time))
            g = grounded.get(e.seq)
            if need and not (args.get("approved_by") == b.supervisor and g is not None and g["grounded"]):
                broke.append("approval_missing")
            if args.get("order_id") in refunded:
                broke.append("second_refund")
            refunded.add(args.get("order_id"))
        rows.append({"step": step, "fn": fn.split(".")[-1], "ticket": args.get("ticket_id"), "broke": broke,
                     "segment": sum(1 for x in boundaries if step > x)})
    awaiting = [t for t in b.tickets if t.kind == "awaiting"]
    resolved = {r["ticket"] for r in rows if r["fn"] == "set_status"}
    first = boundaries[0] if boundaries else None

    def rate(sel: list[dict]) -> dict[str, Any]:
        bad = [r for r in sel if r["broke"]]
        by: dict[str, int] = {}
        for r in bad:
            for k in r["broke"]:
                by[k] = by.get(k, 0) + 1
        return {"actions": len(sel), "broken": len(bad), "rate": round(len(bad) / len(sel), 4) if sel else None, "by_rule": by}

    return {"before_first_compaction": rate([r for r in rows if first is None or r["step"] <= first]),
            "after_first_compaction": rate([r for r in rows if first is not None and r["step"] > first]),
            "by_segment": [{"segment": s, **rate([r for r in rows if r["segment"] == s])} for s in range(len(boundaries) + 1)],
            "lost_constraint_rate": rate([r for r in rows if first is not None and r["step"] > first])["rate"],
            "missed_awaiting": sum(1 for t in awaiting if t.id not in resolved), "awaiting": len(awaiting),
            "examples": [r for r in rows if r["broke"]][:10]}


def compaction_steps(calls: list[tuple[str | None, int | None]]) -> list[int]:
    """The World step of each compaction, from (kind, step) per model call in order; a compaction call that records no
    step (the native policy's) takes the step of the next call that does."""
    out: list[int] = []
    waiting = 0
    for kind, step in calls:
        if kind == "compaction":
            if step is not None:
                out.append(step)
            else:
                waiting += 1
        elif step is not None and waiting:
            out += [step] * waiting
            waiting = 0
    return sorted(out)


def phi_segments(rows: list[dict], boundaries: list[int]) -> list[dict[str, Any]]:
    """Φ at each segment boundary and ΔΦ per segment (milestone Φ)."""
    edges = [0, *boundaries, max((r["step"] for r in rows), default=0)]

    def phi_at(step: int) -> float:
        v = 0.0
        for r in rows:
            if r["step"] <= step:
                v = r["phi"]
            else:
                break
        return v

    out = []
    for i in range(len(edges) - 1):
        a, z = edges[i], edges[i + 1]
        out.append({"segment": i, "steps": [a, z], "phi_start": round(phi_at(a), 6), "phi_end": round(phi_at(z), 6),
                    "delta_phi": round(phi_at(z) - phi_at(a), 6)})
    return out


def token_segments(ro: Any, policy: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per segment: calls, entered tokens (deltas + first prompt + completions), largest full prompt; and the largest prompt
    by call kind. From the rollout's token records (``tito``) or the policy's model calls (``usage`` / ``estimate``)."""
    from crucible.core.schema.rollout import full_prompts

    segs: dict[int, dict[str, Any]] = {}
    peak: dict[str, int] = {}
    if ro is not None and ro.turns and all(t.prompt_token_ids is not None for t in ro.turns):
        source = "tito"
        fp = full_prompts(ro)
        for t in ro.turns:
            s = segs.setdefault(t.segment, {"segment": t.segment, "calls": 0, "entered": 0, "max_prompt": 0, "kinds": {}})
            s["calls"] += 1
            s["kinds"][t.kind] = s["kinds"].get(t.kind, 0) + 1
            if t.kind != "sub":
                s["entered"] += len(t.prompt_token_ids or []) + len(t.completion_token_ids or [])
            n = len(fp.get(t.call_id) or [])
            s["max_prompt"] = max(s["max_prompt"], n)
            peak[t.kind] = max(peak.get(t.kind, 0), n)
    else:
        source = "estimate"
        for c in getattr(policy, "model_calls", None) or []:
            u = c.get("usage") or {}
            s = segs.setdefault(c.get("segment", 0), {"segment": c.get("segment", 0), "calls": 0, "entered": 0, "max_prompt": 0, "kinds": {}})
            s["calls"] += 1
            s["kinds"][c.get("kind")] = s["kinds"].get(c.get("kind"), 0) + 1
            p = int(u.get("prompt_tokens") or 0)
            s["entered"] += int(u.get("completion_tokens") or 0)
            s["max_prompt"] = max(s["max_prompt"], p)
            peak[c.get("kind")] = max(peak.get(c.get("kind"), 0), p)
    return [dict(v, source=source) for _, v in sorted(segs.items())], {"source": source, **{f"max_{k}_prompt": v for k, v in peak.items()}}


def harness_history_reads(policy: Any) -> dict[str, Any] | None:
    """History reads the harness counted (a real model's re-reads after compaction): ``counters["history_reads"]`` (and
    ``counters["history_reads_after_compaction"]``) when the harness keeps them, else the ledger's own stats."""
    c = getattr(policy, "counters", None) or {}
    keys = {k: v for k, v in c.items() if str(k).startswith("history")}
    if keys:
        return keys
    ledger = getattr(policy, "ledger", None)
    stats = getattr(ledger, "stats", None)
    return {"ledger": stats()} if callable(stats) else None


async def restore_probe(policy: Any, kernel: str) -> dict[str, Any] | None:
    """The time to rebuild the episode's final kernel in a fresh sandbox. A harness with a snapshot restore exposes
    ``restore_probe()`` on its policy and is timed through it; the current harness replays the whole journal."""
    if policy is None:
        return None
    hook = getattr(policy, "restore_probe", None)
    if callable(hook):
        t0 = time.perf_counter()
        info = await hook()
        return {"method": "policy.restore_probe", "s": round(time.perf_counter() - t0, 3), **(info or {})}
    k = getattr(policy, "kernel", None)
    if k is None:
        return None
    from crucible.actors.rlm.kernel import DockerLauncher, Kernel, LocalLauncher

    st = k.state()
    launcher = DockerLauncher() if kernel == "docker" else LocalLauncher(allow_host=True)
    fresh = Kernel(launcher, seed=st["seed"], cell_timeout_s=st["cell_timeout_s"])
    t0 = time.perf_counter()
    err = None
    try:
        await fresh.replay(st["journal"])
    except Exception as e:  # noqa: BLE001 -- a replay that fails is a measurement, not a crash of the ladder
        err = f"{type(e).__name__}: {str(e)[:200]}"
    s = round(time.perf_counter() - t0, 3)
    with contextlib.suppress(Exception):
        await fresh.close()
    with contextlib.suppress(Exception):
        await launcher.aclose()
    return {"method": "journal_replay", "s": s, "cells": len(st["journal"]), "journal_bytes": len(json.dumps(st["journal"], default=str)), "error": err}


# ---- one rung ------------------------------------------------------------------------------------------------------------

async def run_rung(cfg: LadderConfig, *, root: Path | None = None, codec: Any = None, keep: bool = False,
                   harness_factory: Callable[[LadderConfig, _Capture], tuple[Any, dict[str, Any]]] | None = None) -> dict[str, Any]:
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.run import Runner, RunSpec
    from crucible.verify.progress import read_rows
    from crucible.world.world import World

    task = BacklogTask(cfg.seed, cfg.n)
    work = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="lh-ladder-"))
    cap = _Capture()
    harness, how = (harness_factory or make_harness)(cfg, cap)
    model = None
    fired: list[Any] = []
    if cfg.harness == "native":
        from crucible.actors.policy import ContextConfig, LLMToolPolicy
        from crucible.runtime.inject import ReplayLLM
        from crucible_pack_toy.longhorizon.world import diligent

        acts = diligent(task.b)

        def make(attempt):
            p = LLMToolPolicy("agent", ReplayLLM(acts), context=ContextConfig(budget_tokens=cfg.budget_tokens, trigger=cfg.trigger,
                                                                              keep_recent_turns=cfg.keep_recent_turns, summary_max_tokens=cfg.summary_max_tokens))
            cap.policies.append(p)
            return {"agent": p}

        # M10 lab audit R-4: a policy's configuration is its run identity (PolicyRef.config)
        ref = PolicyRef(id="replay:diligent", make=make, config={"script": "diligent replay", "budget_tokens": cfg.budget_tokens, "trigger": cfg.trigger,
                                                                 "keep_recent_turns": cfg.keep_recent_turns, "summary_max_tokens": cfg.summary_max_tokens})
    else:
        model, client = make_model(cfg, codec)
        ref = PolicyRef(id=f"scripted:{model.model}", make=lambda attempt: {"agent": with_faults(cfg, client(), attempt, fired)},
                        config={"model": model.model, "variant": cfg.variant, "model_kwargs": cfg.model_kwargs, "faults": cfg.faults, "tito": cfg.tito,
                                "enforce_window": cfg.enforce_window, "budget_tokens": cfg.budget_tokens})
    t0 = time.perf_counter()
    try:
        with checkpoint_meter() as saves:
            s = await Runner(work).run(RunSpec(run_id="lad", tasks=[task], policies=[ref], harnesses=[harness], k=1,
                                                    rollout_mode="train" if cfg.tito and cfg.harness == "rlm" else "eval"))
        wall = time.perf_counter() - t0
        adirs = sorted((work / "lad" / "samples").glob("*/a*"))
        last = adirs[-1]
        rec = json.loads((last / "record.json").read_text())
        ro = RolloutRecord.model_validate_json((last / "rollout.json").read_text()) if (last / "rollout.json").exists() else None
        events = World.read_trace(last / "trace.jsonl")
        policy = cap.policies[-1] if cap.policies else None
        comp_steps = compaction_steps([(t.kind, t.world_step) for t in ro.turns] if ro is not None else [])
        if not comp_steps and policy is not None:
            comp_steps = compaction_steps([(c.get("kind"), c.get("step")) for c in getattr(policy, "model_calls", [])])
        rows = read_rows(last / "progress.jsonl") if (last / "progress.jsonl").exists() else []
        segs, peaks = token_segments(ro, policy)
        steps = int(rec.get("steps") or 0)
        rereads = sum(1 for x in (model.log if model else []) if x["kind"] == "reread")
        prompts = [x.get("prompt_tokens") or 0 for x in (model.log if model else [])]
        threshold = int(cfg.trigger * cfg.budget_tokens)
        out: dict[str, Any] = {
            "label": cfg.label(), "config": {k: v for k, v in dataclasses.asdict(cfg).items()}, "harness_id": getattr(harness, "id", None),
            "rlm_fields": {k: v for k, v in how.items() if k != "config"}, "tickets": cfg.n, "world_calls": steps, "attempts": len(adirs),
            "reward": None if ro is None else ro.reward, "mask": None if ro is None else ro.mask_sample,
            "failure_kind": None if ro is None else ro.failure_kind, "stop": (rec.get("meta") or {}).get("stop"),
            "mean_reward": s.mean_reward, "valid_for_training": None if ro is None else list(ro.valid_for_training()),
            "compactions": len(comp_steps), "compaction_steps": comp_steps,
            "phi_segments": phi_segments(rows, comp_steps), "constraints": constraint_audit(events, task.b, comp_steps),
            "segments": segs, "b1": {"window": cfg.budget_tokens, "trigger_tokens": threshold, **peaks,
                                    "over_window_calls": sum(1 for p in prompts if p > cfg.budget_tokens),
                                    "max_prompt_over_window": round(max(prompts, default=0) / cfg.budget_tokens, 3),
                                    "source": (model.log[-1].get("usage_source") if model and model.log else peaks["source"])},
            "rereads_after_compaction": rereads, "history_reads_harness": harness_history_reads(policy),
            "model_calls": len(model.log) if model else None,
            "cells": {k: sum(1 for x in model.log if x["kind"] == k) for k in sorted({x["kind"] for x in model.log})} if model else None,
            "wall_s": round(wall, 2), "ms_per_world_call": round(1000 * wall / max(1, steps), 2),
            "checkpoints": {"saves": len(saves), "last_bytes": saves[-1]["bytes"] if saves else None,
                            "total_bytes": sum(x["bytes"] or 0 for x in saves), "max_save_s": max((x["save_s"] for x in saves), default=None)},
            "rlm_counters": dict(getattr(policy, "counters", {}) or {}) or None,
            "resumed_from": ((rec.get("meta") or {}).get("artifacts") or {}).get("resumed_from"),
            "attempt_failures": [((json.loads((d / "record.json").read_text()).get("failure") or {}).get("kind")) for d in adirs if (d / "record.json").exists()],
            "faults": {"llm": [x for f in fired for x in f.fired], "kills": cap.kills} if cfg.faults else None,
        }
        spec = getattr(policy, "context_spec", None)  # the new harness's ContextSpec (M16 §3.2), when the policy exposes it
        if spec is not None and callable(getattr(spec, "threshold", None)):
            out["b1"]["harness_threshold"] = spec.threshold()
            out["b1"]["harness_window"] = getattr(spec, "window_tokens", None)
        out["restore"] = await restore_probe(policy, cfg.kernel) if cfg.restore_probe and cfg.harness == "rlm" else None
        if not keep:
            out["dir"] = None
        else:
            out["dir"] = str(last)
        return out
    finally:
        task.cleanup()
        if root is None and not keep:
            shutil.rmtree(work, ignore_errors=True)


async def run_ladder(rungs: list[int], variants: list[str], *, base: LadderConfig | None = None, out: Path | None = None,
                     codec: Any = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Every (rung, variant) pair in order; the report is rewritten after each run, so a long ladder keeps what it finished."""
    base = base or LadderConfig()
    if base.tito == "qwen" and codec is None:
        from crucible.train.tito_scripted import qwen_codec

        codec = qwen_codec()
    report: dict[str, Any] = {"ladder": "toy/helpdesk_backlog step ladder (M16 §4)", "runs": []}
    for rung in rungs:
        for v in variants:
            cfg = dataclasses.replace(base, rung=rung, variant=v)
            log(f"[ladder] {cfg.label()} ...")
            r = await run_rung(cfg, codec=codec)
            report["runs"].append(r)
            log(f"[ladder] {cfg.label()}: reward {r['reward']} steps {r['world_calls']} compactions {r['compactions']} "
                f"lost-constraint {r['constraints']['lost_constraint_rate']} missed {r['constraints']['missed_awaiting']} wall {r['wall_s']} s")
            if out is not None:
                Path(out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report
