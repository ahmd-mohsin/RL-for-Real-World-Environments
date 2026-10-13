"""Training datasets from labelled trajectories (M15 ``crucible label``): SFT, DPO and KTO rows in TRL's canonical
formats, carrying Crucible's dense credit. Study: ``docs/research/bench-studio/01`` §3; audit
``docs/audit/2026-10-13/M15-study-defects-H1.md`` (V1-V19) and ``M15-sweep-G1.md`` (D1-D15).

**One decision per row.** TRL trains every token of a DPO ``chosen`` / ``rejected`` and of a KTO ``completion``
(``trl/trainer/dpo_trainer.py:1054-1101``, ``kto_trainer.py:1067-1108``), and TRL / verl ignore a per-message loss
mask. So every row's trained part is exactly one policy turn and its prompt is everything the policy saw before it
(the multi-harness article's recipe; NeMo-RL's single-response preference rows). The multi-turn ``messages`` +
``message_loss_mask`` SFT row (prime-rl) stays available as ``sft_format="chain"``.

Sources:
- **World runs** (M10). The runner's *training view* is the source of truth: ``rollout.json`` turns (``trainable``,
  ``kind`` turn / compaction / sub, ``segment``, the prompt and completion **token ids** the engine sampled), the
  record's ``mask_sample`` / failure kind, the rollout ``flags``, the run's ``validity.json`` (M13) and the build's
  ``splits.json``. A row's text is the policy's tokens decoded with its own tokenizer, and the row carries the ids
  (``prompt_ids`` / ``completion_ids``; for SFT also ``input_ids`` + ``completion_mask``, which TRL's SFTTrainer takes
  as a pre-tokenized dataset, ``sft_trainer.py:1264, 1424-1443``), so the trained tokens are the policy's, with its
  system prompt, tool schema and parallel calls as it saw and emitted them. RLM episodes are their ``python`` cells;
  compaction summaries and trainable sub-calls are rows of their own. Only model-produced episodes are used: scripted,
  control and adversary policies are never data, except scripted demonstrations under ``allow_scripted`` (labelled
  ``source: scripted_demo`` in every row).
- **Workflow traces** (M3): the policy role's messages and *recorded* tool calls (``tool_call`` events, the tool's own
  name and arguments) become assistant turns, recorded tool results become ``tool`` messages, everyone else's events
  become (merged) user turns. Recorder metadata (row ids, step ids, call ids, writes) never reaches a message; policy
  actions with no recorded tool call are text in the turn; state records and system checks (grader verdicts) are not
  messages. Credit is ΔΦ per assistant turn from the pack's milestone series.

**One positive rule** (``verdict``): a positive has a reward ≥ the threshold and an outcome label that does not say
``bad`` (``unknown`` only under ``allow_unknown``); a negative has a reward below the threshold. Where reward and
outcome disagree the trajectory is neither, and counted. SFT ``correct`` / ``best``, DPO ``chosen`` and KTO ``True``
all use it. Invalid, held-out, masked, exposed and flagged episodes never get this far.

DPO pairs:
- **best vs worst** on a task: the best positive against the worst negative, a positive reward margin, at the first
  step where both saw the *identical* prompt and acted differently (different contexts are never paired).
- **correction pivots** (``signals.pivots``): the graded call (rejected) against the same role's fix (chosen), each
  alone; only when no feedback came between them (``state_shift``) unless ``allow_state_shift``.

Curation (rLLM's ``dataset from-eval`` semantics): a filter over per-task statistics in a closed expression grammar
(names ``avg best worst n n_correct solved``, numbers, comparisons, ``and/or/not``, arithmetic). Nothing else is
evaluated. ``select``: ``correct`` | ``best`` (the best *passing* episode) | ``all``; ``max_per_task`` ≥ 1.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _h(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Step:
    """One policy decision: the context it saw and the one turn it produced. ``prompt`` / ``completion`` are chat
    messages (workflow traces, scripted demos) or decoded text (World runs, with the ids beside them)."""

    prompt: list[dict[str, Any]] | str
    completion: dict[str, Any] | str
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    weight: float | None = None  # the turn's ΔΦ (None: no dense credit)
    kind: str = "turn"  # turn | compaction | sub
    call_id: str = ""
    index: int = 0  # the message index of the turn (message items) or the turn's position (token items)
    wrong: bool = False  # graded wrong (a pivot's graded turn): never a positive target
    ends_eos: bool = True  # the completion ends with the tokenizer's EOS (TRL appends one to text rows that do not)
    exact: bool = True  # the text re-tokenizes to exactly the ids
    redacted: bool = False
    blank: bool = False  # the completion has no text but special tokens (an EOS alone)

    def key(self) -> Any:
        return self.prompt_ids if self.prompt_ids is not None else self.prompt

    def target(self) -> Any:
        return self.completion_ids if self.completion_ids is not None else self.completion

    def empty(self) -> bool:
        if isinstance(self.completion, str):
            return self.blank or not (self.completion_ids if self.completion_ids is not None else self.completion.strip())
        return not ((self.completion.get("content") or "").strip() or self.completion.get("tool_calls"))


@dataclass
class Labelled:
    """One trajectory with credit and provenance: chat messages (workflow traces, scripted demos) or the runner's token
    turns (``token_steps``, World runs)."""

    task: str
    messages: list[dict[str, Any]]
    reward: float | None
    weights: list[float | None] = field(default_factory=list)  # per message: ΔΦ for assistant turns, None otherwise
    tools: list[dict[str, Any]] = field(default_factory=list)
    run: str = ""
    episode: str = ""
    model: str = ""
    label: str | None = None  # good | bad | unknown (workflow traces)
    pivots: list[dict[str, Any]] = field(default_factory=list)
    pivots_unpaired: int = 0  # graded-wrong policy turns with no usable chosen turn
    wrong: set[int] = field(default_factory=set)  # message indices of policy turns graded wrong (paired or not)
    policy: str = ""
    policy_kind: str = "model"  # model | scripted | control | adversary | trace
    harness: str = ""
    split: str = "train"
    source: str = "trace"  # trace | run | scripted_demo
    token_steps: list[Step] | None = None
    chains: list[dict[str, Any]] = field(default_factory=list)  # token items: {"input_ids", "mask", "weights"} per linear chain

    @property
    def traj_id(self) -> str:
        """Unique per trajectory: task, run, episode and content (M15 H1 V16: a hash of the messages alone gave 66 ids to
        285 rows across tasks)."""
        content = [[s.prompt_ids, s.completion_ids] for s in self.token_steps] if self.token_steps is not None else self.messages
        return _h(self.task, self.run, self.episode, content, self.tools)

    def prompt_len(self) -> int:
        """Messages before the first assistant turn (system + task + any context)."""
        return next((i for i, m in enumerate(self.messages) if m["role"] == "assistant"), len(self.messages))

    def steps(self) -> list[Step]:
        if self.token_steps is not None:
            return self.token_steps
        out = []
        ws = list(self.weights or [None] * len(self.messages))
        for i, m in enumerate(self.messages):
            if m["role"] != "assistant" or i == 0 or self.messages[i - 1]["role"] not in ("user", "tool"):
                continue  # TRL needs a prompt ending on user / tool (``apply_chat_template``: add_generation_prompt)
            out.append(Step(prompt=self.messages[:i], completion=m, weight=ws[i] if i < len(ws) else None, index=i, wrong=i in self.wrong))
        return out


# ---- policy kinds (V1) --------------------------------------------------------------------------------------------------
NON_MODEL_PREFIXES = (("scripted:", "scripted"), ("pack:", "scripted"), ("control:", "control"), ("adv:", "adversary"),
                      ("adversary:", "adversary"), ("declared:", "adversary"), ("new:", "adversary"), ("prev:", "adversary"))
_SCRIPTED_WORD = re.compile(r"(?:^|[:+/_.-])(oracle|gold|noop|no-op|no_op|replay)(?:$|[:+/_.-])", re.IGNORECASE)


def policy_kind(policy: str, *, model_turns: bool = True, adversaries: Iterable[str] = ()) -> str:
    """``model`` only for a policy whose episode holds trainable model calls and whose id names no control, scripted
    or adversary policy (``scripted:`` / ``pack:`` / ``control:`` / ``adv:`` / ``adversary:`` / ``declared:``; an
    oracle, gold, no-op or replay)."""
    p = str(policy or "")
    for pre, kind in NON_MODEL_PREFIXES:
        if p.startswith(pre):
            return kind
    if p in set(adversaries):
        return "adversary"
    if _SCRIPTED_WORD.search(p):
        return "scripted"
    return "model" if model_turns else "scripted"


# ---- the positive rule (V7) ---------------------------------------------------------------------------------------------
def verdict(x: Labelled, threshold: float, *, allow_unknown: bool = False) -> str:
    """``pos`` | ``neg`` | why the trajectory is neither (``no_reward``, ``outcome_disagrees``, ``unknown_outcome``)."""
    if x.reward is None:
        return "no_reward"
    passed = float(x.reward) >= threshold
    if (x.label == "bad" and passed) or (x.label == "good" and not passed):
        return "outcome_disagrees"
    if x.label == "unknown" and passed and not allow_unknown:
        return "unknown_outcome"
    return "pos" if passed else "neg"


# ---- sources ------------------------------------------------------------------------------------------------------------
_STUB = frozenset({"script finished", "environment_terminal", "game_over"})


def _args(a: Any) -> dict[str, Any] | None:
    """Tool-call arguments as a dict (TRL ``dataset_formats.md``: a JSON string double-encodes under Qwen2.5's
    ``tojson``). None when they are not a JSON object."""
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            v = json.loads(a)
        except ValueError:
            return None
        return v if isinstance(v, dict) else None
    return {} if a is None else None


def merge_roles(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Consecutive user turns become one (strict templates refuse two in a row: Gemma's "roles must alternate")."""
    out: list[dict[str, Any]] = []
    for m in msgs:
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            out[-1] = {**out[-1], "content": (out[-1].get("content") or "") + "\n\n" + (m.get("content") or "")}
        else:
            out.append(dict(m))
    return out


def messages_from_atif(traj: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An ATIF trajectory as OpenAI chat messages, for scripted demonstrations (model episodes use the token view):
    steps of one model call (``extra.call_id``) are one assistant turn with all its tool calls, arguments are dicts,
    scripted stubs ("script finished") are dropped, consecutive user turns are merged."""
    from crucible.core.schema.actions import openai_name

    out: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    cur_call: Any = object()

    for s in traj.get("steps") or []:
        src = s.get("source")
        msg = s.get("message") or ""
        extra = s.get("extra") or {}
        if src in ("user", "system"):
            out.extend(results)
            results.clear()
            out.append({"role": src, "content": msg})
            cur_call = object()
            continue
        if extra.get("done") and msg.strip() in _STUB:
            msg = ""
        calls = s.get("tool_calls") or []
        reasoning = s.get("reasoning_content") if isinstance(s.get("reasoning_content"), str) and s["reasoning_content"].strip() else ""
        if not calls and not msg.strip():
            continue
        cid = extra.get("call_id")
        if cid is not None and cid == cur_call and out and out[-1]["role"] == "assistant":
            a = out[-1]  # another step of the same model call: one turn (parallel calls)
            if msg.strip():
                a["content"] = ((a.get("content") or "") + "\n" + msg).strip()
        else:
            out.extend(results)
            results.clear()
            a = {"role": "assistant", "content": msg}
            out.append(a)
        if reasoning and reasoning not in (a.get("reasoning_content") or ""):  # kept (G1 D13: Qwen3 / R1 thinking turns)
            a["reasoning_content"] = ((a.get("reasoning_content") or "") + "\n" + reasoning).strip()
        cur_call = cid if cid is not None else object()
        for c in calls:
            a.setdefault("tool_calls", []).append({"id": c.get("tool_call_id"), "type": "function",
                                                   "function": {"name": openai_name(c.get("function_name") or ""), "arguments": _args(c.get("arguments")) or {}}})
        for r in ((s.get("observation") or {}).get("results") or []):
            if r.get("source_call_id"):
                results.append({"role": "tool", "tool_call_id": r["source_call_id"], "content": r.get("content") or ""})
            elif r.get("content"):
                results.append({"role": "user", "content": r["content"]})
    out.extend(results)
    return merge_roles(out)


def tool_schema(traj: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The World's tool cards the policy was offered, from the ATIF ``agent.tool_definitions`` (Harbor's field, OpenAI
    function-calling schema), as OpenAI ``{"type": "function", "function": {...}}`` entries with the names the policy
    called (``openai_name``). Empty when the trajectory does not record them (G1 D13)."""
    from crucible.core.schema.actions import openai_name

    out = []
    for d in ((traj or {}).get("agent") or {}).get("tool_definitions") or []:
        if not isinstance(d, Mapping):
            continue
        fn = dict(d["function"]) if isinstance(d.get("function"), Mapping) else {k: v for k, v in d.items() if k != "type"}
        if not fn.get("name"):
            continue
        fn["name"] = openai_name(str(fn["name"]))
        out.append({"type": "function", "function": fn})
    return out


def redactor(policy: Any = None) -> Any:
    """An M3 ``Redactor`` (``connectors.redact``) for the run path: the same rules as the pack path (V19)."""
    from crucible.connectors.redact import RedactionPolicy, Redactor

    if isinstance(policy, Redactor):
        return policy
    return Redactor(policy if policy is not None else RedactionPolicy())


def redact_messages(msgs: list[dict[str, Any]], r: Any) -> list[dict[str, Any]]:
    """Every string of the messages through the redactor (structure kept)."""
    return [{k: (v if k in ("role", "tool_call_id", "type", "id") else r.value(v)) for k, v in m.items()} for m in msgs]


_TOKENIZERS: dict[str, Any] = {}


def load_tokenizer(name: str) -> Any:
    """The policy's own tokenizer (local cache; ``HF_HUB_OFFLINE`` is honoured)."""
    if name not in _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZERS[name]


def _enc(tok: Any, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def token_steps(rollout: Mapping[str, Any], tok: Any, *, redact: Any = None) -> tuple[list[Step], list[dict[str, Any]], list[str]]:
    """The policy's own turns from ``rollout.json``: per trainable turn the full prompt it saw (delta storage rebuilt
    per actor and segment, as ``core.schema.rollout.full_prompts``) and the completion it sampled, decoded. Sub-calls
    are standalone; non-trainable turns (another model's sub-calls) are never targets. A string the redactor changes is
    re-encoded (``redacted``; the text then no longer is the sampled ids). Returns the steps, the linear chains (one
    multi-turn token row each) and the problems (a trainable turn without a token record refuses the episode)."""
    from crucible.core.schema.rollout import RolloutRecord

    ro = RolloutRecord.model_validate(rollout)
    problems = [f"{t.call_id}: {', '.join(m)}" for t in ro.turns
                if t.trainable and (m := [x for x in t.missing() if x in ("prompt_token_ids", "completion_token_ids", "prefix_break")])]
    if problems:
        return [], [], problems
    eos = getattr(tok, "eos_token", None)
    prev: dict[tuple[str, int], str] = {}
    steps: list[Step] = []
    chains: dict[str, dict[str, Any]] = {}

    def piece(ids: list[int]) -> tuple[list[int], str, bool]:
        text = tok.decode(ids, skip_special_tokens=False)
        if redact is not None:
            red = redact.text(text)
            if red != text:
                return _enc(tok, red), red, True
        return list(ids), text, False

    for k, t in enumerate(ro.turns):
        if not t.trainable or not t.completion_token_ids:
            continue
        key = (t.actor, t.segment)
        if t.kind != "sub" and t.prompt_is_delta and key in prev:
            ck = prev[key]
        else:
            ck = f"{t.actor}:{t.segment}:{t.call_id}"
            chains[ck] = {"pieces": [], "kind": t.kind}
        if t.kind != "sub":
            prev[key] = ck
        ch = chains[ck]
        p_ids, p_text, p_red = piece(list(t.prompt_token_ids or []))
        ch["pieces"].append((p_ids, p_text, None, p_red))
        prompt_ids = [i for ids, _, _, _ in ch["pieces"] for i in ids]
        prompt_text = "".join(x for _, x, _, _ in ch["pieces"])
        c_ids, c_text, c_red = piece(list(t.completion_token_ids))
        w = float(t.advantage) if t.advantage is not None else None
        ch["pieces"].append((c_ids, c_text, w, c_red))
        exact = _enc(tok, prompt_text) == prompt_ids and _enc(tok, prompt_text + c_text) == prompt_ids + c_ids
        steps.append(Step(prompt=prompt_text, completion=c_text, prompt_ids=prompt_ids, completion_ids=c_ids, weight=w, kind=t.kind, call_id=t.call_id,
                          index=k, ends_eos=bool(eos) and c_text.endswith(eos), exact=exact, redacted=any(p[3] for p in ch["pieces"]),
                          blank=not tok.decode(c_ids, skip_special_tokens=True).strip()))
    out_chains = []
    for ck, ch in chains.items():
        ids: list[int] = []
        mask: list[int] = []
        weights: list[float] = []
        for k, (p, _text, w, _r) in enumerate(ch["pieces"]):
            pol = k % 2 == 1  # prompt piece, completion piece, prompt piece, ...
            ids += p
            mask += [1 if pol else 0] * len(p)
            weights += [float(w or 0.0) if pol else 0.0] * len(p)
        out_chains.append({"chain": ck, "kind": ch["kind"], "input_ids": ids, "mask": mask, "weights": weights,
                           "credit": any(p[2] is not None for p in ch["pieces"][1::2])})
    return steps, out_chains, []


def _json(p: Path) -> Any:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _find_up(run_dir: Path, name: str, levels: int = 2) -> Path | None:
    for d in [run_dir, *list(run_dir.resolve().parents)[:levels]]:
        if (d / name).is_file():
            return d / name
    return None


def run_splits(run_dir: Path) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """The build's ``splits.json`` for this run (in the run dir or the build dir above ``runs/``), its seal verified
    with the build's ``bench.key`` when that key is present (a tampered file is refused). (None, info) without one."""
    p = _find_up(Path(run_dir), "splits.json")
    if p is None:
        return None, {"file": None}
    doc = _json(p) or {}
    info: dict[str, Any] = {"file": str(p), "seal": doc.get("seal"), "verified": None}
    key = p.parent / "bench.key"
    if key.is_file():
        from crucible.bench import dynamic, splits

        if not splits.verify_file(p, dynamic.load_key(file=key) + b"|splits"):
            raise ValueError(f"{p}: the split seal does not verify with {key}: refusing to label (the held-out split may have been edited)")
        info["verified"] = True
    return {str(k): str(v) for k, v in (doc.get("splits") or {}).items()}, info


HACK_KINDS = frozenset({"crucible:answer_exposed", "crucible:grader_tampered", "crucible:policy_misbehavior"})


def from_run(run_dir: Path, *, tokenizer: str | None = None, policies: Sequence[str] | None = None, allow_scripted: bool = False,
             include_bench: bool = False, redaction: Any = None, adversaries: Iterable[str] = (), validity: Path | None = None,
             skipped: Counter | None = None, info: dict[str, Any] | None = None) -> list[Labelled]:
    """Every usable episode of an M10 run, from the runner's training view. Skipped, and counted in ``skipped`` by
    reason: invalid tasks (``validity.json``), held-out tasks (``splits.json`` ``bench`` / test, unless
    ``include_bench``), exposed / tampered attempts, masked attempts (``mask_sample``, no reward), flagged attempts
    (rollout ``flags``, ``hack/*`` components), control and adversary policies, scripted policies (unless
    ``allow_scripted``), policies not in ``policies``, episodes whose trainable turns lack a token record."""
    from crucible.bench.metrics import load_run
    from crucible.core.schema.run import RunRecord

    run_dir = Path(run_dir)
    if not (run_dir / "results.jsonl").is_file() or not (run_dir / "samples").is_dir():
        raise ValueError(f"{run_dir} is not an M10 run directory (no results.jsonl and samples/)")
    skipped = Counter() if skipped is None else skipped
    info = {} if info is None else info
    vfile = Path(validity) if validity else run_dir / "validity.json"
    vdoc = _json(vfile) if vfile.is_file() else None
    if validity and vdoc is None:
        raise ValueError(f"validity manifest {vfile} is missing or unreadable")
    invalid = {t for t, v in (vdoc or {}).items() if isinstance(v, Mapping) and not v.get("valid", True)}
    split_of, sinfo = run_splits(run_dir)
    info.update({"validity": str(vfile) if vdoc is not None else None, "invalid_tasks": sorted(invalid), "splits": sinfo, "policies": {},
                 "warnings": [w for w, bad in (("no validity.json: task validity unchecked (run with controls, or pass --validity)", vdoc is None),
                                               ("no splits.json: every task is treated as train (label only runs on a train split)", split_of is None)) if bad]})
    red = redactor(redaction)
    out: list[Labelled] = []
    for e in load_run(run_dir, plan=False):
        sid, n = e.attempt.rsplit("-a", 1)
        adir = run_dir / "samples" / sid / f"a{n}"
        ro = _json(adir / "rollout.json") or {}
        rec_doc = _json(adir / "record.json")
        rec = RunRecord.model_validate(rec_doc) if rec_doc else None
        trainable = [t for t in ro.get("turns") or [] if t.get("trainable", True)]
        kind = policy_kind(e.policy, model_turns=bool(trainable), adversaries=adversaries)
        info["policies"][e.policy] = kind
        fkind = e.failure_kind or (rec.failure.kind if rec is not None and rec.failure else None) or ro.get("failure_kind")
        comps = ro.get("reward_components") or {}
        why = None
        if policies and e.policy not in policies:
            why = "other_policy"
        elif not (adir / "rollout.json").is_file():
            # no training view: its flags, mask and trainable turns are unknown, so it can never be a positive, not even
            # as a scripted demo (M15 campaign B19: it was classed scripted and its flagged passes became positives)
            why = "no_training_view"
        elif e.task in invalid:
            why = "invalid_task"
        elif split_of is not None and split_of.get(e.task) != "train" and not (include_bench and e.task in split_of):
            why = "held_out" if e.task in split_of else "no_split"
        elif e.exposed or fkind in HACK_KINDS:
            why = "exposed_or_tampered"
        elif e.reward is None or ro.get("mask_sample") or (rec is not None and rec.scored_reward() is None):
            why = "masked"
        elif ro.get("flags") or e.flags or any(k.startswith("hack/") and float(v or 0) > 0 for k, v in comps.items()):
            why = "flagged"
        elif kind in ("control", "adversary"):
            why = kind
        elif kind == "scripted" and not allow_scripted:
            why = "scripted"
        if why:
            skipped[why] += 1
            continue
        reward = rec.scored_reward() if rec is not None else e.reward
        base = {"task": e.task, "reward": reward, "run": run_dir.name, "episode": e.attempt, "policy": e.policy, "policy_kind": kind,
                "harness": e.harness, "split": (split_of or {}).get(e.task, "train")}
        traj = _json(adir / "trajectory.agent.json")
        base["tools"] = tool_schema(traj)
        if kind == "scripted":
            msgs = redact_messages(messages_from_atif(traj), red) if traj else []
            if not any(m["role"] == "assistant" for m in msgs):
                skipped["no_policy_turn"] += 1
                continue
            out.append(Labelled(messages=msgs, weights=[None] * len(msgs), model=e.policy, source="scripted_demo", **base))
            continue
        name = tokenizer or _model_name(adir)
        if not name:
            skipped["no_tokenizer"] += 1
            continue
        try:
            tok = load_tokenizer(name)
        except Exception as exc:  # noqa: BLE001 - a missing local tokenizer is the user's to fix: say which
            raise ValueError(f"cannot load tokenizer {name!r} for policy {e.policy} ({type(exc).__name__}); pass --tokenizer") from None
        steps, chains, problems = token_steps(ro, tok, redact=red)
        if problems:
            skipped["no_token_record"] += 1
            continue
        if not steps:
            skipped["no_policy_turn"] += 1
            continue
        out.append(Labelled(messages=[], token_steps=steps, chains=chains, model=name, source="run", **base))
    return out


def _model_name(adir: Path) -> str | None:
    calls = _json(adir / "calls.json") or {}
    for cs in calls.values() if isinstance(calls, dict) else []:
        for c in cs:
            if c.get("model") and c.get("kind") != "sub":
                return str(c["model"])
    traj = _json(adir / "trajectory.agent.json") or {}
    return ((traj.get("agent") or {}).get("model_name")) or None


def _event_piece(e: Any) -> dict[str, Any] | None:
    """What one policy event contributes to its assistant turn: a recorded tool call (its tool's name and its own
    arguments: M15 G1 D4), a message, or the text of an action the log recorded without a tool call. State records
    (``final_patch``, a status the log derived) are not policy output."""
    from crucible.core.schema.actions import openai_name

    data = e.data or {}
    if e.type == "tool_call" and (data.get("tool") or data.get("function")):
        args = _args(data.get("arguments"))
        if args is None:
            return {"content": e.text or ""}
        cid = str(data.get("call_id") or data.get("id") or f"call_{e.seq}")
        return {"tool_calls": [{"id": cid, "type": "function", "function": {"name": openai_name(str(data.get("tool") or data.get("function"))), "arguments": args}}]}
    if e.kind == "message":
        out: dict[str, Any] = {"content": e.text or ""}
        if isinstance(data.get("reasoning"), str) and data["reasoning"].strip() and data["reasoning"].strip() != (e.text or "").strip():
            out["reasoning_content"] = data["reasoning"]
        return out
    if e.kind == "action" and (e.text or "").strip():
        return {"content": e.text.strip()}
    return None


def from_trace(trace: Any, *, policy_roles: Sequence[str] = ("agent",), phi_by_seq: Mapping[int, float] | None = None,
               reward: float | None = None, tools: list[dict[str, Any]] | None = None, split: str = "train") -> Labelled:
    """A workflow trace as a conversation from the policy role's point of view. Consecutive policy events form one
    assistant turn; a recorded tool result is a ``tool`` message answering its call; a leading system message is the
    system prompt; every other actor's event is a user turn (consecutive ones merged); a system actor's check (a
    grader verdict) is no message. A turn's weight is Φ after its last event minus Φ before its first."""
    roles = {a.id: a.role for a in trace.actors}
    handle = next((a.handle for a in trace.actors if a.role in policy_roles and a.kind == "agent" and a.handle), "")
    is_policy = lambda e: roles.get(e.actor or "") in policy_roles
    phi_by_seq = dict(phi_by_seq or {})
    msgs: list[dict[str, Any]] = []
    weights: list[float | None] = []
    turn_of_seq: dict[int, int] = {}
    piece_of_seq: dict[int, dict[str, Any]] = {}
    call_of_seq: dict[int, str] = {}
    open_calls: dict[str, str] = {}  # recorded call id -> message call id
    phi = phi_before = 0.0
    pending: dict[str, Any] | None = None
    pending_seqs: list[int] = []

    def flush() -> None:
        nonlocal pending, pending_seqs
        if pending is not None and ((pending.get("content") or "").strip() or pending.get("tool_calls")):
            for s in pending_seqs:
                turn_of_seq[s] = len(msgs)
            msgs.append(pending)
            weights.append(round(phi - phi_before, 6) if phi_by_seq else None)
        pending, pending_seqs = None, []

    def context(text: str) -> None:
        flush()
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + "\n\n" + text}
        else:
            msgs.append({"role": "user", "content": text})
            weights.append(None)

    for e in sorted(trace.events, key=lambda e: e.seq):
        data = e.data or {}
        who = roles.get(e.actor or "", "system")
        if e.type == "tool_result":
            rid = data.get("call_id") or data.get("id")
            cid = open_calls.get(str(rid)) if rid else None
            if cid is None and e.reply_to is not None:
                cid = call_of_seq.get(e.reply_to)
            if cid is not None:
                flush()
                msgs.append({"role": "tool", "tool_call_id": cid, "content": e.text or ""})
                weights.append(None)
            else:
                context(f"[{who}] {e.text or ''}".strip())
        elif is_policy(e):
            piece = _event_piece(e)
            if piece is not None:
                if pending is None:
                    pending = {"role": "assistant", "content": ""}
                    phi_before = phi
                if piece.get("content"):
                    pending["content"] = ((pending.get("content") or "") + "\n" + piece["content"]).strip()
                if piece.get("reasoning_content"):
                    pending["reasoning_content"] = ((pending.get("reasoning_content") or "") + "\n" + piece["reasoning_content"]).strip()
                for c in piece.get("tool_calls") or []:
                    pending.setdefault("tool_calls", []).append(c)
                    call_of_seq[e.seq] = c["id"]
                    open_calls[str(data.get("call_id") or data.get("id") or c["id"])] = c["id"]
                piece_of_seq[e.seq] = {"role": "assistant", "content": piece.get("content", ""), **({"tool_calls": piece["tool_calls"]} if piece.get("tool_calls") else {})}
            if pending is not None:
                pending_seqs.append(e.seq)  # a state record of an open turn is graded with that turn
        elif e.kind == "check" and who == "system":
            pass  # a grader's / monitor's verdict is the label, never context (M15 G1 D3)
        elif who == "system" and e.kind == "message" and all(m["role"] == "system" for m in msgs) and pending is None:
            if msgs:
                msgs[-1]["content"] += "\n\n" + (e.text or "")
            else:
                msgs.append({"role": "system", "content": e.text or ""})
                weights.append(None)
        elif (e.text or "").strip() or e.kind != "state":
            context(f"[{who}] {e.text or e.type}")
        if e.seq in phi_by_seq:
            phi = phi_by_seq[e.seq]
    flush()
    pivots: list[dict[str, Any]] = []
    unpaired = 0
    wrong: set[int] = set()
    from crucible.signals.pivots import pivots as find_pivots

    by_seq = {e.seq: e for e in trace.events}
    sig = {s.id: s for s in trace.signals}
    seen_pairs: set[tuple[int, int]] = set()
    for graded, sids in sorted(find_pivots(trace).items()):
        if graded not in turn_of_seq:
            continue
        wrong.add(turn_of_seq[graded])  # graded wrong whether or not a fix is known (V8)
        for sid in sids:
            s = sig[sid]
            fix = by_seq.get(s.source_event) if s.source_event is not None else None
            # a revert says the step was wrong, not what was right (M4 audit A20); only a correction or an applied
            # suggestion carries the right step, and only the fix event's own call is the chosen action (V13)
            chosen = piece_of_seq.get(fix.seq) if fix is not None else None
            rejected = piece_of_seq.get(graded) or msgs[turn_of_seq[graded]]
            if (s.type in PAIRABLE and fix is not None and is_policy(fix) and fix.seq in turn_of_seq and chosen is not None
                    and turn_of_seq[fix.seq] > turn_of_seq[graded]):
                pair = (turn_of_seq[graded], turn_of_seq[fix.seq])
                if pair in seen_pairs:  # two graded calls in one turn are one decision (M4 audit A21)
                    continue
                seen_pairs.add(pair)
                between = [e for e in trace.events if graded < e.seq < fix.seq and not is_policy(e)]
                pivots.append({"index": pair[0], "chosen": chosen, "rejected": rejected, "signal": sid, "signal_type": s.type,
                               "reliability": s.reliability,
                               # the fix came after feedback the prompt does not contain: valid only if it needs none (A24)
                               "state_shift": bool(between)})
            else:
                unpaired += 1
    return Labelled(task=trace.family or "", messages=msgs, reward=reward, weights=weights, tools=list(tools or []), run=trace.source.connector,
                    episode=trace.trace_id, model=handle or "", label=trace.outcome.label, pivots=pivots, pivots_unpaired=unpaired, wrong=wrong,
                    policy=",".join(policy_roles), policy_kind="trace", split=split, source="trace")


# ---- curation -----------------------------------------------------------------------------------------------------------
_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq,
            ast.NotEq, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Name, ast.Load, ast.Constant)
NAMES = ("avg", "best", "worst", "n", "n_correct", "solved")


def compile_filter(expr: str | None) -> Callable[[Mapping[str, Any]], bool]:
    """Every user error is a ``ValueError`` (V15): a syntax error, a disallowed node, and an expression that fails on a
    task's statistics (a division by a zero count), which names the task."""
    if not expr:
        return lambda _: True
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"filter: not an expression ({e.msg})") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"filter: {type(node).__name__} is not allowed (names: {', '.join(NAMES)}; comparisons, and/or/not, + - * /)")  # noqa: TRY004 - a user input error
        if isinstance(node, ast.Name) and node.id not in NAMES:
            raise ValueError(f"filter: unknown name {node.id!r} (one of {', '.join(NAMES)})")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
            raise ValueError("filter: only numbers are allowed as constants")  # noqa: TRY004 - a user input error
    code = compile(tree, "<filter>", "eval")

    def keep(stats: Mapping[str, Any]) -> bool:
        try:
            return bool(eval(code, {"__builtins__": {}}, {k: v for k, v in stats.items() if k in NAMES}))
        except ArithmeticError as e:
            raise ValueError(f"filter {expr!r} failed on task {stats.get('task', '?')!r}: {type(e).__name__}: {e}") from None

    return keep


def task_stats(items: Sequence[Labelled], threshold: float, *, allow_unknown: bool = False) -> dict[str, float]:
    rs = [float(x.reward) for x in items if x.reward is not None]
    nc = sum(verdict(x, threshold, allow_unknown=allow_unknown) == "pos" for x in items)
    return {"avg": sum(rs) / len(rs) if rs else 0.0, "best": max(rs, default=0.0), "worst": min(rs, default=0.0), "n": len(rs), "n_correct": nc,
            "solved": float(nc > 0)}


def curate(items: Iterable[Labelled], *, filter: str | None = None, select: str = "correct", max_per_task: int | None = None,
           threshold: float = 1.0, allow_unknown: bool = False) -> list[Labelled]:
    """``correct``: every positive; ``best``: the best positive (none when nothing passes: rLLM ``_ranked_candidates``);
    ``all``: every trajectory, failures included. ``max_per_task`` ≥ 1 (None: no cap)."""
    if max_per_task is not None and max_per_task < 1:
        raise ValueError(f"max_per_task must be >= 1 (got {max_per_task}); omit it for no cap")
    if select not in ("correct", "best", "all"):
        raise ValueError("select: correct | best | all")
    keep = compile_filter(filter)
    by: dict[str, list[Labelled]] = {}
    for x in items:
        by.setdefault(x.task, []).append(x)
    out = []
    for task, xs in sorted(by.items()):
        if not keep({**task_stats(xs, threshold, allow_unknown=allow_unknown), "task": task}):
            continue
        xs = sorted(xs, key=lambda x: (-(x.reward if x.reward is not None else -1), x.traj_id))
        if select in ("correct", "best"):
            xs = [x for x in xs if verdict(x, threshold, allow_unknown=allow_unknown) == "pos"]
        if select == "best":
            xs = xs[:1]
        out += xs[:max_per_task] if max_per_task else xs
    return out


# ---- rows ---------------------------------------------------------------------------------------------------------------
def _prov(x: Labelled, threshold: float) -> dict[str, Any]:
    out = {"task": x.task, "run": x.run, "episode": x.episode, "model": x.model, "policy": x.policy, "policy_kind": x.policy_kind,
           "harness": x.harness, "split": x.split, "source": x.source, "reward": x.reward,
           "resolved": x.reward is not None and x.reward >= threshold, "traj_id": x.traj_id, "outcome": x.label}
    if x.source == "scripted_demo":
        out["demonstration"] = "scripted policy (not model-produced; may act on the task's answer key)"
    return out


def _turn_fields(s: Step, *, sft: bool) -> dict[str, Any]:
    if isinstance(s.completion, str):
        out: dict[str, Any] = {"prompt": s.prompt, "completion": s.completion, "prompt_ids": s.prompt_ids, "completion_ids": s.completion_ids,
                               "text_exact": s.exact, "redacted": s.redacted}
        if sft:  # TRL SFT's pre-tokenized path: labels are input_ids where every mask column is 1 (sft_trainer.py:1424-1443)
            n_p, n_c = len(s.prompt_ids or []), len(s.completion_ids or [])
            out |= {"input_ids": list(s.prompt_ids or []) + list(s.completion_ids or []), "completion_mask": [0] * n_p + [1] * n_c,
                    "assistant_masks": [0] * n_p + [1] * n_c}
        return out
    return {"prompt": s.prompt, "completion": [s.completion]}


def dedup(rows: list[dict[str, Any]], skipped: Counter, name: str) -> list[dict[str, Any]]:
    """Identical rows (same task, context and target) are written once (rLLM ``--dedup``)."""
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["row_id"] in seen:
            skipped[f"{name}_duplicate"] += 1
            continue
        seen.add(r["row_id"])
        out.append(r)
    return out


def sft_rows(items: Iterable[Labelled], *, threshold: float = 1.0, mask_negative: bool = False, fmt: str = "turns",
             skipped: Counter | None = None) -> list[dict[str, Any]]:
    """``turns`` (default): one prompt-completion row per kept policy turn (TRL, verl, NeMo-RL, OpenRLHF and prime-rl
    all train exactly the completion). ``chain``: one multi-turn row per trajectory: ``messages`` +
    ``message_loss_mask`` (prime-rl) for messages, ``input_ids`` + ``assistant_masks`` (TRL's pre-tokenized path) per
    linear token chain. Turns graded wrong (V8), empty turns and, under ``mask_negative``, turns with negative ΔΦ are
    never trained. Weights are numbers (0.0 without credit; ``credit`` says whether dense credit exists)."""
    skipped = Counter() if skipped is None else skipped
    if fmt not in ("turns", "chain"):
        raise ValueError("sft format: turns | chain")
    out = []
    for x in items:
        prov = _prov(x, threshold)
        if fmt == "chain" and x.token_steps is not None:
            for ch in x.chains:
                row_mask = [m if not (mask_negative and w < 0) else 0 for m, w in zip(ch["mask"], ch["weights"])]
                if not any(row_mask):
                    skipped["sft_nothing_to_train"] += 1
                    continue
                out.append({"input_ids": ch["input_ids"], "assistant_masks": row_mask, "weights": ch["weights"], "credit": ch["credit"],
                            "call_kind": ch["kind"], **prov, "row_id": _h(x.task, ch["input_ids"], row_mask)})
        elif fmt == "chain":
            ws = list(x.weights or [None] * len(x.messages))
            mask = [m["role"] == "assistant" and i not in x.wrong and not (mask_negative and w is not None and w < 0)
                    and bool((m.get("content") or "").strip() or m.get("tool_calls")) for i, (m, w) in enumerate(zip(x.messages, ws))]
            if not any(mask):
                skipped["sft_nothing_to_train"] += 1
                continue
            out.append({"messages": x.messages, "tools": x.tools, "message_loss_mask": mask, "weights": [float(w or 0.0) for w in ws],
                        "credit": any(w is not None for w in ws), **prov, "row_id": _h(x.task, x.messages, mask)})
        else:
            for s in x.steps():
                if s.wrong:
                    skipped["sft_graded_wrong"] += 1
                elif mask_negative and s.weight is not None and s.weight < 0:
                    skipped["sft_negative_credit"] += 1
                elif s.empty():
                    skipped["sft_empty_completion"] += 1
                else:
                    out.append({**_turn_fields(s, sft=True), "tools": x.tools, "weight": float(s.weight or 0.0), "credit": s.weight is not None,
                                "call_kind": s.kind, "call_id": s.call_id, "step": s.index, **prov, "row_id": _h(x.task, s.key(), s.target())})
    return dedup(out, skipped, "sft")


def _pref_ok(s: Step) -> bool:
    """A preference / KTO completion must be the policy's whole turn: not empty, and (text rows) ending with EOS, since
    TRL appends one to a text completion that lacks it (``dpo_trainer.py`` / ``kto_trainer.py`` ``add_eos``)."""
    return not s.empty() and (not isinstance(s.completion, str) or s.ends_eos)


def dpo_best_vs_worst(items: Iterable[Labelled], *, threshold: float = 1.0, margin: float = 0.25, allow_unknown: bool = False,
                      skipped: Counter | None = None) -> list[dict[str, Any]]:
    """Per task and harness: the best positive vs the worst negative, at the first step where both saw the identical
    prompt and acted differently (V11). Chosen passes, rejected fails, the reward gap is ≥ ``margin`` > 0."""
    skipped = Counter() if skipped is None else skipped
    if margin <= 0:
        raise ValueError(f"margin must be > 0 (got {margin}): a pair needs a reward gap")
    by: dict[tuple[str, str], list[Labelled]] = {}
    for x in items:
        by.setdefault((x.task, x.harness), []).append(x)
    out = []
    for (task, _harness), xs in sorted(by.items()):
        pos = sorted((x for x in xs if verdict(x, threshold, allow_unknown=allow_unknown) == "pos"), key=lambda x: (-float(x.reward or 0), x.traj_id))
        neg = sorted((x for x in xs if verdict(x, threshold, allow_unknown=allow_unknown) == "neg"), key=lambda x: (float(x.reward or 0), x.traj_id))
        if not pos or not neg:
            continue
        best, worst = pos[0], neg[0]
        if float(best.reward or 0) - float(worst.reward or 0) < margin:
            skipped["dpo_margin"] += 1
            continue
        pair = None
        for a, b in zip([s for s in best.steps() if s.kind == "turn"], [s for s in worst.steps() if s.kind == "turn"], strict=False):
            if a.key() != b.key():
                break  # the contexts diverged before the decisions did: not a preference at one state
            if a.target() != b.target():
                pair = (a, b)
                break
        if pair is None:
            skipped["dpo_no_shared_decision"] += 1
            continue
        a, b = pair
        if a.wrong or not _pref_ok(a) or not _pref_ok(b):
            skipped["dpo_unusable_turn"] += 1
            continue
        # the first differing turn need not be why one episode passed and the other failed (M15 campaign B20: every
        # finance_ap pair preferred a probe for the hidden tests over the correct first action). A pair is kept only with
        # evidence at that turn: the rejected turn was graded wrong, or the chosen turn's dense credit beats the rejected's.
        evidence = b.wrong or (a.weight is not None and b.weight is not None and float(a.weight) > float(b.weight)) \
            or (a.weight is not None and b.weight is None and float(a.weight) > 0)
        if not evidence:
            skipped["dpo_no_evidence_at_turn"] += 1
            continue
        row: dict[str, Any] = {"prompt": a.prompt, "chosen": a.completion if isinstance(a.completion, str) else [a.completion],
                               "rejected": b.completion if isinstance(b.completion, str) else [b.completion], "tools": best.tools}
        if a.prompt_ids is not None:
            row |= {"prompt_ids": a.prompt_ids, "chosen_ids": a.completion_ids, "rejected_ids": b.completion_ids, "text_exact": a.exact and b.exact}
        row |= {"kind": "best_vs_worst", "task": task, "harness": best.harness, "run": best.run, "step": a.index,
                "chosen_episode": best.episode, "rejected_episode": worst.episode, "chosen_policy": best.policy, "rejected_policy": worst.policy,
                "chosen_model": best.model, "rejected_model": worst.model, "chosen_reward": best.reward, "rejected_reward": worst.reward,
                "chosen_traj": best.traj_id, "rejected_traj": worst.traj_id, "source": best.source, "split": best.split,
                "row_id": _h(task, a.key(), a.target(), b.target())}
        out.append(row)
    return dedup(out, skipped, "dpo")


PAIRABLE = frozenset({"correction", "suggestion_accepted"})


def dpo_pivots(items: Iterable[Labelled], *, allow_state_shift: bool = False, skipped: Counter | None = None) -> list[dict[str, Any]]:
    """Correction pivots: the graded call (rejected) and the fix (chosen), each only its own event, at the state before
    the graded turn. A fix that came after someone's feedback (``state_shift``) needs context the prompt lacks: left out
    unless ``allow_state_shift`` (V13)."""
    skipped = Counter() if skipped is None else skipped
    out = []
    for x in items:
        for pv in x.pivots:
            i = pv["index"]
            prompt = x.messages[:i]
            chosen, rejected = pv["chosen"], pv.get("rejected") or x.messages[i]
            if pv.get("state_shift") and not allow_state_shift:
                skipped["pivot_state_shift"] += 1
            elif not prompt or prompt[-1]["role"] not in ("user", "tool"):
                skipped["pivot_no_prompt"] += 1
            elif chosen == rejected or Step(prompt, chosen).empty() or Step(prompt, rejected).empty():
                skipped["pivot_unusable"] += 1
            else:
                out.append({"prompt": prompt, "chosen": [chosen], "rejected": [rejected], "tools": x.tools, "kind": "correction_pivot",
                            "task": x.task, "episode": x.episode, "chosen_episode": x.episode, "rejected_episode": x.episode, "policy": x.policy,
                            "signal": pv.get("signal"), "traj_id": x.traj_id, "source": x.source, "split": x.split,
                            # M4's estimate of how often this kind of signal is right: 1 - reliability is the pair's label-noise
                            # rate (robust / conservative DPO); None until reliability was estimated
                            "reliability": pv.get("reliability"), "state_shift": bool(pv.get("state_shift", False)),
                            "row_id": _h(x.task, prompt, chosen, rejected)})
    return dedup(out, skipped, "dpo")


def kto_rows(items: Iterable[Labelled], *, threshold: float = 1.0, allow_unknown: bool = False, skipped: Counter | None = None) -> list[dict[str, Any]]:
    """One unpaired row per policy turn: ``label`` is a bool (TRL ``kto_trainer.py:206, 1139``), True for a turn of a
    positive trajectory, False for a turn of a negative one or a turn graded wrong; the outcome label is ``outcome``."""
    skipped = Counter() if skipped is None else skipped
    out = []
    for x in items:
        v = verdict(x, threshold, allow_unknown=allow_unknown)
        if v not in ("pos", "neg"):
            skipped[f"kto_{v}"] += 1
            continue
        for s in x.steps():
            if not _pref_ok(s):
                skipped["kto_unusable_turn"] += 1
                continue
            good = v == "pos" and not s.wrong
            row = {**_turn_fields(s, sft=False), "tools": x.tools, "weight": float(s.weight or 0.0), "credit": s.weight is not None,
                   "call_kind": s.kind, "step": s.index, **_prov(x, threshold), "row_id": _h(x.task, s.key(), s.target(), good)}
            row["label"] = bool(good)  # last: provenance never overwrites it (V6)
            out.append(row)
    # a (prompt, completion) seen with both labels (a shared prefix of a passing and a failing episode) is no evidence
    # either way: drop every copy (M15 campaign B21: 47 of 61 toy negatives were also positives)
    seen: dict[Any, set[bool]] = {}
    for r in out:
        seen.setdefault(_h(r.get("task"), json.dumps(r.get("prompt"), sort_keys=True, default=str), json.dumps(r.get("completion"), sort_keys=True, default=str)), set()).add(r["label"])
    keep = []
    for r in out:
        k = _h(r.get("task"), json.dumps(r.get("prompt"), sort_keys=True, default=str), json.dumps(r.get("completion"), sort_keys=True, default=str))
        if len(seen[k]) > 1:
            skipped["kto_conflicting_labels"] += 1
        else:
            keep.append(r)
    return dedup(keep, skipped, "kto")


# ---- templates ----------------------------------------------------------------------------------------------------------
def adapt_for_template(msgs: list[dict[str, Any]], *, system: bool = True, tool: bool = True) -> list[dict[str, Any]]:
    """For a chat template without a system or tool role (Gemma-2): the system prompt opens the first user turn, a
    tool result is a user turn, an assistant's tool calls are its text; consecutive user turns are merged."""
    out: list[dict[str, Any]] = []
    sys_text = ""
    for m in msgs:
        m = dict(m)
        if m["role"] == "system" and not system:
            sys_text += (m.get("content") or "") + "\n\n"
            continue
        if m["role"] == "tool" and not tool:
            m = {"role": "user", "content": f"[tool result] {m.get('content') or ''}"}
        if m["role"] == "assistant" and m.get("tool_calls") and not tool:
            calls = "\n".join(json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]}, sort_keys=True) for c in m.pop("tool_calls"))
            m["content"] = ((m.get("content") or "") + "\n" + calls).strip()
        if sys_text and m["role"] == "user":
            m["content"] = sys_text + (m.get("content") or "")
            sys_text = ""
        out.append(m)
    return merge_roles(out)


def template_roles(tok: Any) -> dict[str, bool]:
    """Which optional roles the tokenizer's chat template renders (a template that raises on a role lacks it)."""
    def renders(msgs: list[dict[str, Any]]) -> bool:
        try:
            tok.apply_chat_template(msgs, tokenize=False)
        except Exception:  # noqa: BLE001 - jinja raises TemplateError / ValueError for an unsupported role
            return False
        return True

    u, a = {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]}
    return {"system": renders([{"role": "system", "content": "s"}, u, a]),
            "tool": renders([u, call, {"role": "tool", "tool_call_id": "c1", "content": "r"}, a])}


def adapt_rows(rows: list[dict[str, Any]], roles: Mapping[str, bool]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        r = dict(r)
        for k in ("messages", "prompt", "completion", "chosen", "rejected"):
            if isinstance(r.get(k), list):
                r[k] = adapt_for_template(r[k], system=roles.get("system", True), tool=roles.get("tool", True))
        if not roles.get("tool", True):
            r["tools"] = []
        out.append(r)
    return out


def check_render(rows: list[dict[str, Any]], tok: Any, skipped: Counter, name: str) -> list[dict[str, Any]]:
    """Keep the rows the tokenizer's chat template renders (each conversation through ``apply_chat_template``)."""
    out = []
    for r in rows:
        convs = [r["messages"]] if isinstance(r.get("messages"), list) else []
        if isinstance(r.get("prompt"), list):
            convs += [r["prompt"] + r[k] for k in ("completion", "chosen", "rejected") if isinstance(r.get(k), list)]
        try:
            for c in convs:
                tok.apply_chat_template(c, tools=r.get("tools") or None, tokenize=False)
        except Exception:  # noqa: BLE001 - the template refuses this row
            skipped[f"{name}_template_refused"] += 1
            continue
        out.append(r)
    return out


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> int:
    """Atomic: a temp file renamed into place (a crash never leaves a half-written dataset)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows))
    tmp.replace(path)
    return len(rows)
