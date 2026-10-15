"""The rollout record contract ``crucible.rollout/v1`` (DESIGN §10.3 point 1; ARCHITECTURE M12 ``rollout.py``).

The trainer's only input from an episode. Per model call (``Turn``): the prompt and completion token ids and the
behaviour logprobs exactly as the inference engine sampled them (token-in-token-out; never detokenized and
re-tokenized), the policy version that generated the span, and the sampling parameters. Per rollout: harness@version,
truncation, compaction segments, timings, infra status, reward channels and per-turn credit, and the World trace id.

``valid_for_training`` is the gate every trainer bridge applies: a rollout missing any token record is masked, never
guessed (the corrections in ``mismatch.py`` and the staleness cap need both the logprobs and the version).
Shapes: verl ``TokenTrace`` / uni_agent ``Trajectory`` with a per-span version (research m9-m10/02 §2, verl row).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any, ClassVar, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel, SchemaKeyed

SCHEMA = "crucible.rollout/v1"


# How a call relates to the one before it in the episode's message graph (M16 §3.5; research long-horizon-rlm/01 R9:
# verifiers ``trace.py`` / nano-rlm ``semantic.py`` edges). Typed for readers; only ``compaction_attempt`` changes packing.
EDGES = ("continuation", "compaction_attempt", "compaction", "subagent_call", "subagent_return", "agent_message", "subagent_cancel", "sub")


class Turn(CrucibleModel):
    call_id: str
    actor: str
    kind: str = "turn"  # turn | compaction (the policy's own summary: trained with the episode's advantage) | sub
    segment: int = 0
    # M16 (very-long-horizon RLM, design §3.5): the agent of the RLM tree that made the call ("" = the root, a child = its
    # path "scan", "scan/sub2") and the typed edge to its predecessor. A linear chain is keyed (actor, agent, segment):
    # a child's turns, the root's turns and the sub-calls interleave in the record without breaking each other's chain
    # (M16 stress B7: one shared chain made the trainer forward 7x the entered tokens, 19.7x with 256 sub-calls).
    agent: str = ""
    edge: str | None = None
    world_step: int | None = None
    # Token storage is linear in the trajectory, not quadratic (research long-horizon/01: verifiers MessageNode): within a
    # segment a call stores only the prompt tokens it adds after the previous call's prompt + completion; the first call of
    # a segment stores its whole (compacted) prompt. ``prefix_break``: the engine's prompt did not extend the previous one.
    prompt_is_delta: bool = False
    prefix_break: bool = False
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None  # behaviour logprobs, one per completion token
    policy_version: int | str | None = None  # the version that produced the call (call-level switching: the only one)
    # Token-level switching (AReaL-style abort + resubmit with the partial output, DeepSeek pause at a token boundary):
    # run-length spans [start_completion_token, version], covering every completion token, in order. Absent = one version.
    version_spans: list[tuple[int, int]] | None = None
    sampling: dict[str, Any] = Field(default_factory=dict)  # temperature, top_p, max_tokens, seed
    finish: str | None = None
    routing: list[list[int]] | None = None  # MoE routing ids per token, when the engine exposes them
    candidates: list[list[int]] | None = None  # per completion token, the top-p nucleus it was sampled from (checklist #37)
    advantage: float | None = None  # per-turn credit from steps.jsonl (M6), attached to this call's tokens
    # on-policy distillation / KL reference (m13 C-5, ``train.teachers``): per completion token, the teacher's and the
    # reference model's log-prob of the sampled token given exactly this prompt
    teacher_logprobs: list[float] | None = None
    format_errors: int = 0  # malformed tool calls this call issued (m13 C-8): ``TrainerConfig.tool_format_penalty`` each, on its tokens
    step_reward: float | None = None  # a per-step reward for this call (process / speed shaping): reward-to-go credit (MiniMax M2)
    depends_on: list[str] | None = None  # call ids this call waited for (multi-agent DAG): critical-path latency (DeepSeek Agent Team)
    ref_logprobs: list[float] | None = None
    atif_step_ids: list[int] = Field(default_factory=list)
    # vision-language prompts (``train.chat.HFVisionChatCodec``): the images whose placeholder runs this call's prompt adds
    # (linear like the tokens: earlier images of the segment live on earlier calls), each ``{kind, mime, ref, offset,
    # length, grid}``: ``ref`` is the content address of the bytes (``sha256:<hex>``, the World store's key), ``offset``/
    # ``length`` locate the expanded placeholder run in the call's *full* prompt, ``grid`` is the processor's (t, h, w)
    # patch grid. Pixels never enter the record: the trainer rebuilds them from the refs (``train.vision``).
    media: list[dict[str, Any]] | None = None
    # the model that answered this call when it is not the trained policy (an RLM ``sub_llm``: M14 audit G1 D1); such a call
    # is ``trainable=False``: recorded, never trained, exempt from the token-record checks
    model: str | None = None
    trainable: bool = True

    def node(self) -> bool:
        """A stand-alone sequence (a tree node), never a link of an agent's linear chain: an RLM sub-LLM call, or a
        summary request (edge ``compaction_attempt``) whose prompt is the summarizer's bounded view of the context, not
        the context itself (several attempts share one prefix, so none extends another; M16 §3.5). The first call of a
        segment stores its whole prompt; a node always does."""
        return self.kind == "sub" or (self.kind == "compaction" and self.edge in ("compaction_attempt", "compaction"))

    def chain(self) -> tuple[Any, ...]:
        """The linear chain this call extends: (actor, agent, segment); a node is its own chain."""
        return (self.actor, self.agent, "node", self.call_id) if self.node() else (self.actor, self.agent, self.segment)

    def versions(self) -> list[int]:
        if self.version_spans:
            return [int(v) for _, v in self.version_spans]
        return [int(self.policy_version)] if isinstance(self.policy_version, int) else []

    def spans_valid(self) -> bool:
        """Version purity is provable: no spans (one version), or spans starting at token 0, strictly increasing starts,
        non-decreasing versions, all inside the completion."""
        if not self.version_spans:
            return True
        starts = [s for s, _ in self.version_spans]
        vs = [v for _, v in self.version_spans]
        n = len(self.completion_token_ids or [])
        return starts[0] == 0 and all(a < b for a, b in pairwise(starts)) and all(a <= b for a, b in pairwise(vs)) \
            and starts[-1] < max(n, 1)

    def candidates_valid(self) -> bool:
        """Truncated sampling (top_p < 1) needs the candidate set of every completion token, containing that token: the
        trainer renormalizes π_θ over it, else the behaviour logprob is compared with the wrong distribution."""
        if float(self.sampling.get("top_p", 1.0)) >= 1.0:
            return True
        toks = self.completion_token_ids or []
        return self.candidates is not None and len(self.candidates) == len(toks) and all(t in c for t, c in zip(toks, self.candidates, strict=True))

    def missing(self) -> list[str]:
        """What keeps this call out of training (empty: train-valid). Named, so a masked run says why."""
        if not self.trainable:
            return []
        n = len(self.completion_token_ids or [])
        out = [k for k, bad in (("prompt_token_ids", self.prompt_token_ids is None), ("completion_token_ids", n == 0),
                                ("logprobs", self.logprobs is None or len(self.logprobs) != n),
                                ("policy_version (set the provider's policy_version for a served checkpoint)", self.policy_version is None),
                                ("prefix_break", self.prefix_break), ("version_spans", not self.spans_valid()),
                                ("top_p candidates", not self.candidates_valid())) if bad]
        return out

    def token_complete(self) -> bool:
        n = len(self.completion_token_ids or [])
        return self.prompt_token_ids is not None and n > 0 and self.logprobs is not None and len(self.logprobs) == n and self.policy_version is not None


class Segment(CrucibleModel):
    """A context-compaction segment: the turns generated under one context (``compaction.py``). Per agent of an RLM tree
    (``agent``: "" = the root, a child's path): every agent numbers its own segments (M16 §3.5)."""

    index: int
    start_call: str | None = None
    end_call: str | None = None
    summary_call: str | None = None  # the compaction call that opened it
    context_before: int | None = None
    context_after: int | None = None
    agent: str = ""
    # why the context was rebuilt: compaction | compaction_failed | rerender | resume | restore | sub (H13). A resume opens
    # a new segment whose first call stores its whole prompt (M16 stress B3: it kept the segment and every resumed
    # rollout was masked capture_incomplete as a prefix break)
    boundary_reason: str | None = None
    attempts: int | None = None  # summary requests made at this boundary (accepted or not)
    failed: bool = False  # every attempt failed: the episode ended here (compaction_failed, a truncation)
    # the unit of context_before / context_after: "usage" (engine-measured prompt + completion tokens) or "estimate"
    # (chars/4); None = not stated (M16 stress B8: estimates read 102.4k against 146-148k real tokens)
    tokens_source: Literal["usage", "estimate"] | None = None


class RolloutRecord(SchemaKeyed):
    SCHEMA: ClassVar[str] = SCHEMA
    schema_: str = Field(SCHEMA, alias="schema")
    rollout_id: str  # the attempt id
    sample: str
    task: str
    harness: str  # harness@version: one harness per GRPO group
    policy: str
    group: str | None = None  # the GRPO group key (task + seed family)
    turns: list[Turn] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    reward: float | None = None  # None = masked: never trained on as 0
    reward_components: dict[str, float] = Field(default_factory=dict)
    gates: dict[str, bool] = Field(default_factory=dict)
    truncated: bool = False
    stop: str | None = None
    failure_kind: str | None = None
    blame: str | None = None
    mask_sample: bool = False
    timings: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None
    tier: str = "fast"
    trajectory_tokens: int | None = None  # every token that entered the policy's context over the episode, before compaction
    progress: dict[str, Any] = Field(default_factory=dict)  # dense progress summary (phi0, phi_final, sum_delta, drift)
    credit_fallback: bool = False  # the auxiliary credit channel missed its deadline: outcome-only advantage (checklist #40)
    # group grading stage (m13 C-2): a group-level operator (GAR, tournament, hack zeroing) may set the sequence advantage
    # of every scored member; the trainer then uses it instead of the group statistic. ``group_credit_fallback``: the
    # stage missed its deadline (or failed) and the raw advantages were used.
    advantage: float | None = None
    group_credit_fallback: bool = False
    # level-1 hack suspicions (m13 C-4, ``credit.hack_detector.rule_flags``): intercepted calls and rule hits, with evidence;
    # the level-2 judge in the group stage turns them into ``hack/confirmed`` (or not)
    flags: list[dict[str, Any]] = Field(default_factory=list)
    source: str | None = None  # the scheduler source (domain) the prompt came from: routes teachers, selects per-source KL
    teacher_id: str | None = None
    opd_fallback: bool = False  # the teacher missed its deadline or failed: this record trains on the RL signal only
    # per-seat records of a multi-agent game (``train.seats``): the seat whose turns (and only those) this record holds and
    # whose outcome ``reward`` is; None for a single-actor episode or a combined trace
    actor: str | None = None

    def oldest_version(self) -> int | str | None:
        vs = [v for t in self.turns for v in (t.versions() or ([t.policy_version] if t.policy_version is not None else []))]
        return min(vs, key=lambda v: (isinstance(v, str), v)) if vs else None

    def valid_for_training(self) -> tuple[bool, str]:
        if self.mask_sample or self.reward is None:
            return False, f"masked ({self.failure_kind or 'no reward'})"
        if not self.turns:
            return False, "no model calls recorded"
        bad = [(t.call_id, t.missing()) for t in self.turns if t.missing()]
        if bad:
            return False, f"token record incomplete for {len(bad)}/{len(self.turns)} calls (first: {bad[0][0]}: {', '.join(bad[0][1])})"
        return True, "ok"


def full_prompts(ro: RolloutRecord) -> dict[str, list[int]]:
    """Reconstruct each call's full prompt from linear (delta) storage: within a segment a delta prompt extends the previous
    call's prompt + completion; the first call of a segment stores its whole prompt. Per chain (actor, agent, segment):
    a child agent's calls never extend the root's (M16 §3.5)."""
    out: dict[str, list[int]] = {}
    prev: dict[tuple[Any, ...], list[int]] = {}
    for t in ro.turns:
        if t.node():  # RLM sub-LLM calls and summary attempts are stand-alone sequences (m14, M16): never a chain link
            out[t.call_id] = list(t.prompt_token_ids or [])
            continue
        key = t.chain()
        if t.prompt_is_delta and key in prev:
            full = prev[key] + list(t.prompt_token_ids or [])
        else:
            full = list(t.prompt_token_ids or [])
        out[t.call_id] = full
        prev[key] = full + list(t.completion_token_ids or [])
    return out

