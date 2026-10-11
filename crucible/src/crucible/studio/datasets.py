"""Training datasets from labelled trajectories (M15 ``crucible label``): SFT, DPO and KTO rows in TRL's canonical
formats, carrying Crucible's dense credit. Study: ``docs/research/bench-studio/01`` §3.

Formats (TRL ``dataset_formats.md``; one JSONL loads in TRL, prime-rl, NeMo-RL and verl):
- **SFT** ``{"messages": [...], "tools": [...]}``: OpenAI-style messages with assistant ``tool_calls`` preserved, plus
  prime-rl's per-message ``message_loss_mask`` (train only on the policy's own turns; ``--mask-negative`` also drops turns
  whose credit was negative) and a per-message ``weights`` column (the turn's ΔΦ), so a trainer can weight by credit.
- **DPO** ``{"prompt": [...], "chosen": [...], "rejected": [...], "tools": [...]}`` with an explicit prompt.
- **KTO** ``{"prompt": [...], "completion": [...], "label": bool}`` (unpaired).
Every row carries provenance in the shape of Harbor's trace rows and SWE-smith's ``collect_trajs``: ``task``, ``run``,
``episode``, ``model``, ``reward``, ``resolved``, and ``traj_id`` (a content hash, so a row is traceable and duplicates
are visible).

Sources:
- **World runs** (the M10 journal): ``trajectory.<actor>.json`` (ATIF) becomes messages.
- **Workflow traces** (M3): the policy role's events become assistant turns; everyone else's become user turns; credit is
  ΔΦ per assistant turn from the pack's milestone series.

DPO pairs, which no framework builds offline from agent traces:
- **best vs worst of k** on the same task: chosen passes (reward ≥ threshold), the reward margin is ≥ ``margin``, and
  the prompt is the shared prefix (system + task). This is TRL online DPO's reward argmax, offline.
- **correction pivots** (``signals.pivots``): a policy action graded wrong and then corrected by the same role. The
  prompt is the history before the wrong action, rejected is the wrong action, chosen is the correction.

Curation (rLLM's ``dataset from-eval`` semantics): a filter over per-task statistics in a closed expression grammar
(names ``avg best worst n n_correct solved``, numbers, comparisons, ``and/or/not``, arithmetic). Nothing else is
evaluated. ``select``: ``correct`` | ``best`` | ``all``; ``max_per_task``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Labelled:
    """One trajectory as messages, with credit and provenance."""

    task: str
    messages: list[dict[str, Any]]
    reward: float | None
    weights: list[float | None] = field(default_factory=list)  # per message: ΔΦ for assistant turns, None otherwise
    tools: list[dict[str, Any]] = field(default_factory=list)
    run: str = ""
    episode: str = ""
    model: str = ""
    label: str | None = None  # good | bad | unknown (workflow traces)
    pivots: list[dict[str, Any]] = field(default_factory=list)  # [{"index": message index of the wrong turn, "chosen": message}]
    pivots_unpaired: int = 0  # graded-wrong policy turns whose fix was not the policy's own action (no chosen turn)

    @property
    def traj_id(self) -> str:
        return hashlib.sha256(json.dumps(self.messages, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def prompt_len(self) -> int:
        """Messages before the first assistant turn (system + task + any context)."""
        return next((i for i, m in enumerate(self.messages) if m["role"] == "assistant"), len(self.messages))


# ---- sources ------------------------------------------------------------------------------------------------------------
def messages_from_atif(traj: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An ATIF trajectory (Harbor's schema, as Crucible writes it) as OpenAI chat messages."""
    out: list[dict[str, Any]] = []
    for s in traj.get("steps") or []:
        src = s.get("source")
        msg = s.get("message") or ""
        if src in ("user", "system"):
            out.append({"role": src, "content": msg})
            continue
        calls = s.get("tool_calls") or []
        if not calls and not msg.strip():
            continue  # a model call with no visible action (``extra.no_action``)
        a: dict[str, Any] = {"role": "assistant", "content": msg}
        if calls:
            from crucible.core.schema.actions import openai_name

            a["tool_calls"] = [{"id": c.get("tool_call_id"), "type": "function",  # "app.tool" -> an OpenAI-legal name, as the policy saw it
                                "function": {"name": openai_name(c.get("function_name") or ""), "arguments": json.dumps(c.get("arguments") or {}, sort_keys=True)}}
                               for c in calls]
        out.append(a)
        for r in ((s.get("observation") or {}).get("results") or []):
            if r.get("source_call_id"):
                out.append({"role": "tool", "tool_call_id": r["source_call_id"], "content": r.get("content") or ""})
            elif r.get("content"):
                out.append({"role": "user", "content": r["content"]})
    return out


def from_run(run_dir: Path, *, actor: str = "agent") -> list[Labelled]:
    """Every terminal, scored episode of an M10 run (masked episodes have no reward and are skipped)."""
    from crucible.bench.metrics import load_run

    out = []
    for e in load_run(run_dir):
        if e.masked:
            continue
        sid, n = e.attempt.rsplit("-a", 1)
        path = Path(run_dir) / "samples" / sid / f"a{n}" / f"trajectory.{actor}.json"
        if not path.exists():
            continue
        traj = json.loads(path.read_text())
        msgs = messages_from_atif(traj)
        model = ((traj.get("agent") or {}).get("model_name")) or e.policy
        out.append(Labelled(task=e.task, messages=msgs, reward=e.reward, weights=[None] * len(msgs), run=Path(run_dir).name, episode=e.attempt,
                            model=str(model)))
    return out


def from_trace(trace: Any, *, policy_roles: Sequence[str] = ("agent",), phi_by_seq: Mapping[int, float] | None = None,
               reward: float | None = None) -> Labelled:
    """A workflow trace as a conversation from the policy role's point of view. Consecutive policy events form one
    assistant turn (messages become its content, actions its tool calls, each answered by a tool message); every other
    actor's event is a user turn. A turn's weight is Φ after its last event minus Φ before its first."""
    roles = {a.id: a.role for a in trace.actors}
    is_policy = lambda e: roles.get(e.actor or "") in policy_roles
    phi_by_seq = dict(phi_by_seq or {})
    msgs: list[dict[str, Any]] = []
    weights: list[float | None] = []
    turn_of_seq: dict[int, int] = {}
    phi = 0.0
    pending: dict[str, Any] | None = None
    pend_tools: list[dict[str, Any]] = []
    phi_before = 0.0

    def flush(phi_after: float) -> None:
        nonlocal pending, pend_tools
        if pending is not None:
            msgs.append(pending)
            weights.append(round(phi_after - phi_before, 6) if phi_by_seq else None)
            for t in pend_tools:
                msgs.append(t)
                weights.append(None)
        pending, pend_tools = None, []

    for e in sorted(trace.events, key=lambda e: e.seq):
        if is_policy(e):
            if pending is None:
                pending = {"role": "assistant", "content": ""}
                phi_before = phi
            if e.kind == "message":
                pending["content"] = (pending["content"] + "\n" + (e.text or "")).strip()
            else:
                cid = f"call_{e.seq}"
                args = {"object": e.target.object if e.target else None, "text": e.text,
                        **{k: v for k, v in (e.data or {}).items() if isinstance(v, (str, int, float, bool)) and k not in ("writes",)}}
                pending.setdefault("tool_calls", []).append({"id": cid, "type": "function",
                                                             "function": {"name": e.type, "arguments": json.dumps({k: v for k, v in args.items() if v is not None}, sort_keys=True)}})
                pend_tools.append({"role": "tool", "tool_call_id": cid, "content": f"{e.type}: recorded"})
            turn_of_seq[e.seq] = len(msgs)
        else:
            flush(phi)
            who = roles.get(e.actor or "", "system")
            msgs.append({"role": "user", "content": f"[{who} {e.actor or ''}] {e.text or e.type}".replace("  ", " ")})
            weights.append(None)
        if e.seq in phi_by_seq:
            phi = phi_by_seq[e.seq]
    flush(phi)
    pivots: list[dict[str, Any]] = []
    unpaired = 0
    from crucible.signals.pivots import pivots as find_pivots

    by_seq = {e.seq: e for e in trace.events}
    sig = {s.id: s for s in trace.signals}
    for graded, sids in sorted(find_pivots(trace).items()):
        if graded not in turn_of_seq:
            continue
        for sid in sids:
            fix = by_seq.get(sig[sid].source_event) if sig[sid].source_event is not None else None
            if fix is not None and is_policy(fix) and fix.seq in turn_of_seq and turn_of_seq[fix.seq] > turn_of_seq[graded]:
                pivots.append({"index": turn_of_seq[graded], "chosen": msgs[turn_of_seq[fix.seq]], "signal": sid})
            else:
                unpaired += 1
    return Labelled(task=trace.family or "", messages=msgs, reward=reward, weights=weights, run=trace.source.connector, episode=trace.trace_id,
                    label=trace.outcome.label, pivots=pivots, pivots_unpaired=unpaired)


# ---- curation -----------------------------------------------------------------------------------------------------------
_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq,
            ast.NotEq, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Name, ast.Load, ast.Constant)
NAMES = ("avg", "best", "worst", "n", "n_correct", "solved")


def compile_filter(expr: str | None) -> Callable[[Mapping[str, float]], bool]:
    if not expr:
        return lambda _: True
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"filter: {type(node).__name__} is not allowed (names: {', '.join(NAMES)}; comparisons, and/or/not, + - * /)")  # noqa: TRY004 - a user input error
        if isinstance(node, ast.Name) and node.id not in NAMES:
            raise ValueError(f"filter: unknown name {node.id!r} (one of {', '.join(NAMES)})")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("filter: only numbers are allowed as constants")  # noqa: TRY004 - a user input error
    code = compile(tree, "<filter>", "eval")
    return lambda stats: bool(eval(code, {"__builtins__": {}}, dict(stats)))


def task_stats(items: Sequence[Labelled], threshold: float) -> dict[str, float]:
    rs = [float(x.reward) for x in items if x.reward is not None]
    nc = sum(r >= threshold for r in rs)
    return {"avg": sum(rs) / len(rs) if rs else 0.0, "best": max(rs, default=0.0), "worst": min(rs, default=0.0), "n": len(rs), "n_correct": nc,
            "solved": float(nc > 0)}


def curate(items: Iterable[Labelled], *, filter: str | None = None, select: str = "correct", max_per_task: int | None = None,
           threshold: float = 1.0) -> list[Labelled]:
    keep = compile_filter(filter)
    by: dict[str, list[Labelled]] = {}
    for x in items:
        by.setdefault(x.task, []).append(x)
    out = []
    for task, xs in sorted(by.items()):
        if not keep(task_stats(xs, threshold)):
            continue
        xs = sorted(xs, key=lambda x: (-(x.reward if x.reward is not None else -1), x.traj_id))
        if select == "correct":
            xs = [x for x in xs if x.reward is not None and x.reward >= threshold]
        elif select == "best":
            xs = xs[:1]
        elif select != "all":
            raise ValueError("select: correct | best | all")
        out += xs[:max_per_task] if max_per_task else xs
    return out


# ---- rows ---------------------------------------------------------------------------------------------------------------
def _prov(x: Labelled, threshold: float) -> dict[str, Any]:
    return {"task": x.task, "run": x.run, "episode": x.episode, "model": x.model, "reward": x.reward,
            "resolved": x.reward is not None and x.reward >= threshold, "traj_id": x.traj_id, **({"label": x.label} if x.label else {})}


def sft_rows(items: Iterable[Labelled], *, threshold: float = 1.0, mask_negative: bool = False) -> list[dict[str, Any]]:
    out = []
    for x in items:
        mask = [m["role"] == "assistant" and not (mask_negative and (w is not None and w < 0)) for m, w in zip(x.messages, x.weights or [None] * len(x.messages))]
        if not any(mask):
            continue  # nothing to train on (prime-rl drops all-zero-mask samples)
        out.append({"messages": x.messages, "tools": x.tools, "message_loss_mask": mask, "weights": list(x.weights or [None] * len(x.messages)),
                    **_prov(x, threshold)})
    return out


def dpo_best_vs_worst(items: Iterable[Labelled], *, threshold: float = 1.0, margin: float = 0.25) -> list[dict[str, Any]]:
    by: dict[str, list[Labelled]] = {}
    for x in items:
        if x.reward is not None:
            by.setdefault(x.task, []).append(x)
    out = []
    for task, xs in sorted(by.items()):
        xs = sorted(xs, key=lambda x: (-float(x.reward or 0.0), x.traj_id))
        best, worst = xs[0], xs[-1]
        if len(xs) < 2 or float(best.reward or 0) < threshold or float(best.reward or 0) - float(worst.reward or 0) < margin:
            continue
        p = min(best.prompt_len(), worst.prompt_len())
        if best.messages[:p] != worst.messages[:p]:
            continue  # not the same prompt: not a preference pair
        out.append({"prompt": best.messages[:p], "chosen": best.messages[p:], "rejected": worst.messages[p:], "tools": best.tools, "kind": "best_vs_worst",
                    "task": task, "chosen_reward": best.reward, "rejected_reward": worst.reward, "chosen_traj": best.traj_id, "rejected_traj": worst.traj_id})
    return out


def dpo_pivots(items: Iterable[Labelled]) -> list[dict[str, Any]]:
    out = []
    for x in items:
        for pv in x.pivots:
            i = pv["index"]
            out.append({"prompt": x.messages[:i], "chosen": [pv["chosen"]], "rejected": [x.messages[i]], "tools": x.tools, "kind": "correction_pivot",
                        "task": x.task, "episode": x.episode, "signal": pv.get("signal"), "traj_id": x.traj_id})
    return out


def kto_rows(items: Iterable[Labelled], *, threshold: float = 1.0) -> list[dict[str, Any]]:
    out = []
    for x in items:
        good = (x.label == "good") if x.label in ("good", "bad") else (x.reward is not None and x.reward >= threshold)
        if x.label == "unknown" or (x.label is None and x.reward is None):
            continue
        p = x.prompt_len()
        out.append({"prompt": x.messages[:p], "completion": x.messages[p:], "label": bool(good), "tools": x.tools, **_prov(x, threshold)})
    return out


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows))
    return len(rows)
