"""M16 records (design §3.5; research long-horizon-rlm/03 B3, B7, B8; /01 R9, R10; /02 H13): how an RLM tree's model calls
(root turns, sub-calls, summary attempts, child agents, resumes) become ``crucible.rollout/v1`` records and trainer
sequences.

- B7: a sub-call or a child's turn never breaks the root's chain: forwarded tokens = entered tokens (+ the nodes' own).
- B3: a resume that opens a new segment (``boundary_reason="resume"``) is train-valid on the root and on a child; a
  re-rendered prompt inside a segment is still a masked prefix break.
- Summary attempts: rejected → recorded, never trained; accepted → trained with the episode's advantage.
- Children: their own chains, the spawning root turn's credit, their version spans; frozen when another model ran them.
- R10 / H13: every trainable token weighs the same whether the rollout has 1, 2 or 8 segments, under every loss_agg.
- Over-length rows are masked and counted, never tail-cut.
- B8: segments carry measured context sizes with their ``tokens_source``.
"""

from __future__ import annotations

import copy
import random

import pydantic
import pytest
import torch

from crucible.core.schema.reward import StepRow
from crucible.core.schema.rollout import RolloutRecord, Turn, full_prompts
from crucible.core.schema.run import RunRecord
from crucible.runtime.record import build
from crucible.train.bridges.common import flatten
from crucible.train.credit import turn_credit_map
from crucible.train.engine import TinyPolicy
from crucible.train.mismatch import Correction
from crucible.train.objectives import LOSS_AGG, resolve_agg, row_weights
from crucible.train.trainer import TorchTrainer, TrainerConfig

MARK = 3  # the chat template's "assistant:" token: every prompt ends with it
V = 64


class Ep:
    """A scripted RLM tree episode emitting ``model_calls`` as the M16 loop (F) does: one exact-TITO context per agent,
    linear capture (``prompt_delta`` + ``prompt_prefix_len``) inside a segment, the whole prompt at a segment's first call."""

    def __init__(self, seed: int = 0, *, linear: bool = True):
        self.rng = random.Random(seed)
        self.linear = linear
        self.calls: list[dict] = []
        self.segments: list[dict] = []
        self.ctx: dict[str, list[int]] = {}
        self.seg: dict[str, int] = {}
        self.mark: dict[str, int | None] = {}  # len(previous prompt + completion) in this segment; None = next call opens one

    def _toks(self, n: int) -> list[int]:
        return [self.rng.randrange(10, V) for _ in range(n)]

    def _cid(self) -> str:
        return f"agent:c{len(self.calls) + 1}"

    def turn(self, agent: str = "", *, comp: list[int] | None = None, parent: str | None = None, edge: str | None = None, version: int = 0,
             spans=None, trainable: bool | None = None, model: str | None = None) -> str:
        ctx = self.ctx.setdefault(agent, [1, 2 if not agent else 4, MARK])
        self.seg.setdefault(agent, 0)
        comp = comp if comp is not None else self._toks(self.rng.randrange(2, 6))
        tok = {"completion_token_ids": comp, "logprobs": [-2.0] * len(comp), "policy_version": version}
        if spans:
            tok["version_spans"] = spans
        m = self.mark.get(agent)
        if self.linear and m is not None:
            tok["prompt_token_ids"], tok["prompt_delta"], tok["prompt_prefix_len"] = None, ctx[m:], m
        else:
            tok["prompt_token_ids"] = list(ctx)
        cid = self._cid()
        call = {"call_id": cid, "kind": "turn", "segment": self.seg[agent], "agent": agent, "depth": agent.count("/") + (1 if agent else 0),
                "edge": edge or "continuation", "tokens": tok, "actions": [], "step": len(self.calls)}
        if parent:
            call["parent"] = parent
        if trainable is not None:
            call["trainable"] = trainable
        if model:
            call["model"] = model
        self.calls.append(call)
        self.mark[agent] = len(ctx) + len(comp)
        self.ctx[agent] = ctx + comp + [5, 6] + self._toks(self.rng.randrange(1, 4)) + [MARK]  # the observation, then the next turn
        return cid

    def sub(self, parent: str, agent: str = "", *, trainable: bool = True) -> str:
        comp = self._toks(3)
        cid = self._cid()
        self.calls.append({"call_id": cid, "kind": "sub", "edge": "sub", "parent": parent, "agent": agent, "segment": self.seg.get(agent, 0),
                           "model": None if trainable else "helper", "trainable": trainable, "actions": [],
                           "tokens": {"prompt_token_ids": [9, 9, *self._toks(4), MARK], "completion_token_ids": comp, "logprobs": [-2.0] * 3,
                                      "policy_version": 0}})
        return cid

    def compaction(self, agent: str = "", *, accepted: list[bool]) -> list[str]:
        """Summary attempts over the summarizer's bounded view (never an extension of the context); the accepted one opens
        the next segment, whose first call stores its whole prompt."""
        ctx, ids, summary = self.ctx[agent], [], None
        for ok in accepted:
            comp = self._toks(4)
            cid = self._cid()
            self.calls.append({"call_id": cid, "kind": "compaction", "edge": "compaction_attempt", "accepted": ok, "agent": agent,
                               "segment": self.seg[agent], "actions": [],
                               "tokens": {"prompt_token_ids": [8, *ctx[-6:], MARK], "completion_token_ids": comp, "logprobs": [-2.0] * 4,
                                          "policy_version": 0}})
            ids.append(cid)
            summary = comp if ok else summary
        if summary is not None:
            self._boundary(agent, "compaction", [1, 2, 7, *summary, MARK], attempts=len(accepted), summary_call=ids[accepted.index(True)])
        return ids

    def _boundary(self, agent: str, reason: str, new_ctx: list[int], **kw) -> None:
        before = len(self.ctx[agent])
        self.seg[agent] += 1
        self.ctx[agent], self.mark[agent] = new_ctx, None
        self.segments.append({"index": self.seg[agent], "agent": agent, "boundary_reason": reason, "context_before": before,
                              "context_after": len(new_ctx), "tokens_source": "usage", **kw})

    def resume(self, agent: str = "", *, new_segment: bool = True) -> None:
        """A crash + resume: the new engine re-renders the whole history (a re-tokenization: the first ids differ)."""
        rerendered = [1, 2, 11, *self.ctx[agent][3:]]
        if new_segment:
            self._boundary(agent, "resume", rerendered)
        else:  # the pre-M16 behaviour: the policy kept its segment and the next call stored the re-rendered prompt
            self.ctx[agent], self.mark[agent] = rerendered, None

    def record(self, *, mode: str = "train", reward: float = 1.0, rows=(), rid: str = "r") -> RolloutRecord:
        rec = RunRecord(run_id=rid, task="t", world="w", policy="p", started_at="2026-01-01T00:00:00Z", steps=len(self.calls), outcome="completed",
                        reward=reward, attempt=f"{rid}-a1")
        return build(rec, calls={"agent": self.calls}, segments={"agent": self.segments} if self.segments else None, mode=mode, rows=rows)


def _tree(seed: int = 0, *, linear: bool = True, subs_every: int = 2, root_turns: int = 24) -> Ep:
    """A root with interleaved sub-calls, a compaction with one rejected attempt, and a child agent whose turns interleave
    with the root's (spawned by root turn 4, resumed once)."""
    ep = Ep(seed, linear=linear)
    spawner = None
    for k in range(root_turns):
        cid = ep.turn(edge="compaction" if k == 12 else None)
        if k % subs_every == 0:
            ep.sub(cid)
        if k == 4:
            spawner = cid
            ep.turn("scan", parent=spawner, edge="subagent_call")
        elif k > 4 and k % 3 == 0:
            ep.turn("scan")
            if k == 9:
                ep.resume("scan")
        if k == 11:
            ep.compaction(accepted=[False, True])
    return ep


def _entered(ro: RolloutRecord) -> tuple[int, int]:
    """(tokens that entered each chain: the last call's full prompt + completion per (actor, agent, segment), the trainable
    nodes' own prompt + completion)."""
    fp = full_prompts(ro)
    last: dict = {}
    node = 0
    for t in ro.turns:
        n = len(fp[t.call_id]) + len(t.completion_token_ids or [])
        if t.node():
            node += n if t.trainable else 0
        else:
            last[t.chain()] = n
    return sum(last.values()), node


def _legacy_forward(ro: RolloutRecord) -> int:
    """Forward tokens of the pre-M16 packing (bridges/common.flatten before M16): ONE open sequence, broken by any turn of
    another chain."""
    fp = full_prompts(ro)
    seqs: list[list] = []
    cur = None
    for t in ro.turns:
        full, comp = fp[t.call_id], list(t.completion_token_ids or [])
        if cur is not None and cur[0] == (t.actor, t.segment) and t.kind != "sub" and not t.prefix_break and full[: len(cur[1])] == cur[1]:
            cur[1] = full + comp
        else:
            cur = [(t.actor, t.segment) if t.kind != "sub" else None, full + comp, False]
            seqs.append(cur)
        cur[2] = cur[2] or t.trainable
    return sum(len(s[1]) for s in seqs if s[2])


# ---- B7: sub-calls and children never break the root chain -----------------------------------------------------------
@pytest.mark.parametrize("linear", [True, False])
def test_B7_root_chain_survives_interleaved_subs_and_children(linear):
    ro = _tree(linear=linear).record()
    assert ro.valid_for_training() == (True, "ok"), ro.stop
    seqs = flatten(ro, seq_adv=1.0)
    root = [s for s in seqs if not s.agent and not s.node]
    assert [s.segment for s in root] == [0, 1]  # one sequence per root segment, whatever interleaves with it
    assert sorted({s.segment for s in seqs if s.agent == "scan"}) == [0, 1]  # the child: one per its own segment (resumed once)
    entered, nodes = _entered(ro)
    fwd = sum(len(s.prompt_ids) + len(s.response_ids) for s in seqs if any(s.response_mask))
    assert fwd == entered + nodes
    assert _legacy_forward(ro) > 3 * fwd  # the same record under the pre-M16 single-chain packing
    assert seqs[-1] is root[-1] and seqs[-1].last and not any(s.last for s in seqs[:-1])


def test_B7_trainer_forwards_the_entered_tokens():
    ro = _tree(subs_every=1, root_turns=40).record()
    tr = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(item_unit="segment", token_budget=1 << 20))
    two = [ro, ro.model_copy(update={"rollout_id": "r2", "reward": 0.0})]
    m = tr.step(two, ["g", "g"])
    entered, nodes = _entered(ro)
    assert m["train/forward_tokens"] == 2 * (entered + nodes)
    assert m["train/tokens"] == 2 * sum(len(t.completion_token_ids or []) for t in ro.turns if t.trainable)


def test_B7_episode_ending_on_a_sub_call_closes_on_the_root():
    ep = Ep(1)
    c = ep.turn()
    ep.turn()
    c = ep.turn()
    ep.sub(c)
    ep.sub(c)
    ro = ep.record()
    seqs = flatten(ro)
    assert not seqs[-1].node and seqs[-1].last and [s.last for s in seqs].count(True) == 1


def test_full_prompts_keep_each_agent_chain_exact():
    ep = _tree(linear=True)
    ro = ep.record()
    full = _tree(linear=False).record()  # the same episode, every prompt stored whole
    fp, ref = full_prompts(ro), full_prompts(full)
    assert fp == ref
    for c in ep.calls:  # every rebuilt prompt is exactly what the engine saw
        if c["tokens"].get("prompt_token_ids") is not None and c.get("kind") != "compaction":
            assert fp[c["call_id"]] == c["tokens"]["prompt_token_ids"]


# ---- B3: resume opens a segment, on the root and on a child ------------------------------------------------------------
def test_B3_resume_with_a_new_segment_is_train_valid():
    ep = Ep(2)
    s = None
    for k in range(6):
        cid = ep.turn()
        if k == 1:
            s = cid
            ep.turn("scan", parent=s, edge="subagent_call")
            ep.turn("scan")
    ep.resume("")
    ep.resume("scan")
    for _ in range(3):
        ep.turn()
        ep.turn("scan")
    ro = ep.record(mode="train")
    assert not ro.mask_sample and ro.valid_for_training() == (True, "ok"), ro.stop
    assert not any(t.prefix_break for t in ro.turns)
    first = [t for t in ro.turns if t.segment == 1]
    assert {t.agent for t in first} == {"", "scan"} and not first[0].prompt_is_delta
    resumes = [sg for sg in ro.segments if sg.boundary_reason == "resume"]
    assert {(sg.agent, sg.index) for sg in resumes} == {("", 1), ("scan", 1)}
    assert all(any(s.response_mask) for s in flatten(ro, 1.0))


@pytest.mark.parametrize("agent", ["", "scan"])
def test_B3_rerender_inside_a_segment_is_still_a_masked_prefix_break(agent):
    ep = Ep(3)
    root = ep.turn()
    if agent:
        ep.turn("scan", parent=root, edge="subagent_call")
    ep.turn(agent)
    ep.resume(agent, new_segment=False)
    ep.turn(agent)
    ro = ep.record(mode="train")
    assert ro.mask_sample and ro.failure_kind == "crucible:capture_incomplete" and "prefix_break" in (ro.stop or "")
    assert [t.agent for t in ro.turns if t.prefix_break] == [agent]


def test_child_turns_never_break_the_root_chain_in_the_record():
    ep = Ep(4)
    root = ep.turn()
    for _ in range(4):
        ep.turn("scan", parent=root)
        ep.turn("scan/deep", parent=root)
        ep.turn()
    ro = ep.record(mode="train")
    assert ro.valid_for_training() == (True, "ok")
    assert all(t.prompt_is_delta for t in ro.turns if t.agent == "" and t.call_id != root)


# ---- summary attempts: only an accepted one trains ---------------------------------------------------------------------
def test_rejected_summary_attempts_are_recorded_not_trained():
    ep = Ep(5)
    for _ in range(3):
        ep.turn()
    bad1, bad2, good = ep.compaction(accepted=[False, False, True])
    for _ in range(2):
        ep.turn()
    ro = ep.record(mode="train")
    assert ro.valid_for_training() == (True, "ok")
    by = {t.call_id: t for t in ro.turns}
    for cid in (bad1, bad2):
        assert by[cid].trainable is False and by[cid].completion_token_ids is None and by[cid].kind == "compaction"
    assert by[good].trainable and by[good].node() and by[good].completion_token_ids
    assert [t.call_id for t in ro.turns][-2:] == [bad1, bad2]  # frozen nodes after the chains
    seg = next(s for s in ro.segments if s.agent == "" and s.index == 1)
    assert seg.boundary_reason == "compaction" and seg.attempts == 3 and seg.summary_call == good and not seg.failed
    seqs = flatten(ro, seq_adv=0.7)
    summ = next(s for s in seqs if s.call_spans[0][0] == good)
    assert summ.node and all(m == 1 for m in summ.response_mask) and all(a == pytest.approx(0.7) for a in summ.advantages)
    assert not [s for s in seqs if s.call_spans and s.call_spans[0][0] in (bad1, bad2) and any(s.response_mask)]
    tr = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(token_budget=1 << 20))
    m = tr.step([ro, ro.model_copy(update={"rollout_id": "r2", "reward": 0.0})], ["g", "g"])
    trained = sum(len(t.completion_token_ids or []) for t in ro.turns if t.trainable)
    assert m["train/tokens"] == 2 * trained


def test_a_failed_compaction_segment_is_recorded():
    ep = Ep(6)
    ep.turn()
    ep.compaction(accepted=[False, False])
    ep.segments.append({"index": 1, "agent": "", "boundary_reason": "compaction_failed", "attempts": 2, "failed": True})
    ro = ep.record(mode="train")
    assert ro.valid_for_training()[0]
    sg = ro.segments[-1]
    assert sg.failed and sg.attempts == 2 and sg.boundary_reason == "compaction_failed"


# ---- children ----------------------------------------------------------------------------------------------------------
def test_children_share_the_spawning_root_turns_credit_and_keep_version_spans():
    ep = Ep(7)
    r1 = ep.turn()
    r2 = ep.turn()
    k1 = ep.turn("scan", parent=r2, edge="subagent_call", comp=[20, 21, 22], spans=[[0, 0], [2, 1]], version=0)
    k2 = ep.turn("scan")
    g1 = ep.turn("scan/deep", parent=k2, edge="subagent_call")  # a grandchild: spawned by the child
    s1 = ep.sub(k2, agent="scan")  # a sub-call from the child's cell
    r3 = ep.turn()
    rows = [StepRow(step=1, call_id=r1, phi=0.1, delta_phi=0.1), StepRow(step=2, call_id=r2, phi=0.3, delta_phi=0.2),
            StepRow(step=3, call_id=k2, phi=0.6, delta_phi=0.3),  # a child on the shared World made progress itself
            StepRow(step=4, call_id=r3, phi=1.0, delta_phi=0.4)]
    ro = ep.record(rows=rows)
    by = {t.call_id: t for t in ro.turns}
    assert by[r2].advantage == pytest.approx(0.5)  # its own ΔΦ + what the child it spawned achieved (Φ telescopes on the root)
    for cid in (k1, k2, g1, s1):
        assert by[cid].advantage == pytest.approx(0.5), cid
    assert by[k1].depends_on == [r2] and by[k2].depends_on == [r2] and by[g1].depends_on == [k2] and by[s1].depends_on == [k2]
    assert by[k1].version_spans == [(0, 0), (2, 1)] and by[k1].agent == "scan" and by[k1].edge == "subagent_call"
    assert sum(t.advantage or 0.0 for t in ro.turns if not t.agent and not t.node()) == pytest.approx(1.0)
    # M6 turn credit: only the root chain is Φ's chain; children and nodes take their parent's composed credit
    other = ep.record(rows=[StepRow(step=1, call_id=r1, phi=0.0, delta_phi=0.0)], rid="q", reward=0.0)
    tc = turn_credit_map([ro, other], ["g", "g"], eta=1.0)
    rid = ro.rollout_id
    for cid in (k1, k2):
        assert tc[(rid, cid)] == pytest.approx(tc[(rid, r2)])
    assert tc[(rid, g1)] == pytest.approx(tc[(rid, k2)]) and tc[(rid, s1)] == pytest.approx(tc[(rid, k2)])
    assert tc[(rid, r2)] != pytest.approx(tc[(rid, r1)])


def test_a_child_on_another_model_is_recorded_not_trained():
    ep = Ep(8)
    r = ep.turn()
    ep.turn("helper", parent=r, edge="subagent_call", trainable=False, model="other-model")
    ep.turn("helper", trainable=False, model="other-model")
    ep.turn()
    ro = ep.record(mode="train")
    assert ro.valid_for_training() == (True, "ok")
    kids = [t for t in ro.turns if t.agent == "helper"]
    assert all(not t.trainable and t.model == "other-model" for t in kids)
    seqs = flatten(ro, 1.0)
    assert all(not any(s.response_mask) for s in seqs if s.agent == "helper")
    tr = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(token_budget=1 << 20))
    m = tr.step([ro, ro.model_copy(update={"rollout_id": "r2", "reward": 0.0})], ["g", "g"])
    assert m["train/tokens"] == 2 * sum(len(t.completion_token_ids) for t in ro.turns if not t.agent)


def test_critic_gives_each_child_its_own_chain():
    from crucible.train.critic import CriticConfig, TinyValue

    ro = _tree(9).record()
    tr = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(item_unit="call"), critic=TinyValue(V, 8, seed=2), critic_cfg=CriticConfig())
    with_child = tr._gae_advantages([ro])
    alone = ro.model_copy(update={"turns": [t for t in ro.turns if t.agent != "scan"]})
    without = tr._gae_advantages([alone])
    for t in alone.turns:  # the root's Bellman chain is the same with or without the child interleaved
        if t.trainable and t.completion_token_ids:
            assert with_child[(ro.rollout_id, t.call_id)] == pytest.approx(without[(ro.rollout_id, t.call_id)])
    assert all((ro.rollout_id, t.call_id) in with_child for t in ro.turns if t.agent == "scan")


# ---- R10 / H13: per-logical-rollout weighting --------------------------------------------------------------------------
class Bigram(TinyPolicy):
    """log π(token | context) depends on the previous token only: every prompt ends with MARK, so a completion token's
    term is the same however the rollout is segmented, and the loss differs only through the weights."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(ids))


def _segmented(n_seg: int, seed: int, *, tree: bool) -> Ep:
    rng = random.Random(seed)
    comps = [[rng.randrange(10, V) for _ in range(rng.randrange(2, 7))] for _ in range(16)]
    ep = Ep(100 + n_seg)  # the observations (prompt tokens) differ by variant: only the completions are shared
    cut = {round(16 * i / n_seg) for i in range(1, n_seg)}
    for k, c in enumerate(comps):
        if k in cut:
            ep.resume("")
        cid = ep.turn(comp=c)
        if tree and k == 3:
            ep.turn("scan", parent=cid, edge="subagent_call", comp=[30, 31, 32])
            ep.turn("scan", comp=[33, 34])
        if tree and k % 5 == 0:
            ep.calls.append({"call_id": f"agent:c{len(ep.calls) + 1}", "kind": "sub", "parent": cid, "segment": ep.seg[""], "actions": [],
                             "tokens": {"prompt_token_ids": [9, MARK], "completion_token_ids": [40 + k, 41], "logprobs": [-2.0] * 2, "policy_version": 0}})
    return ep


def _token_weights(tr: TorchTrainer, recs, groups, mode: str) -> dict[str, list[float]]:
    items = tr._items(recs, groups)
    lengths = [sum(it.mask) for it in items]
    w = row_weights(lengths, [it.seq for it in items], [it.prompt_index for it in items], mode, max_len=tr.cfg.loss_max_len)
    out: dict[str, list[float]] = {}
    for it, wi in zip(items, w, strict=True):
        out.setdefault(it.seq, []).extend(wi for m in it.mask if m)
    return {k: sorted(v) for k, v in out.items()}


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("mode", LOSS_AGG)
def test_R10_every_trainable_token_weighs_the_same_whatever_the_segment_count(mode, tree):
    tr = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(token_budget=1 << 20))
    other = _segmented(1, seed=99, tree=False).record(rid="b", reward=0.0)
    ref = None
    for n_seg in (1, 2, 8):
        ro = _segmented(n_seg, seed=5, tree=tree).record(rid="a")
        assert ro.valid_for_training()[0] and len({t.segment for t in ro.turns if not t.agent}) == n_seg
        got = _token_weights(tr, [ro, other], ["g", "g"], mode)
        assert len(got["a-a1"]) == sum(len(t.completion_token_ids) for t in ro.turns if t.trainable)
        if ref is None:
            ref = got
        assert got == pytest.approx(ref), (mode, n_seg)


@pytest.mark.parametrize("correction,mode", [("ppo", "auto"), ("ppo", "token_mean_global"), ("ppo", "seq_mean_token_mean"), ("ppo", "prompt_mean"),
                                             ("gspo", "auto"), ("hispo", "auto")])
def test_R10_loss_and_gradient_are_segment_count_invariant(correction, mode):
    """End to end through ``TorchTrainer.step``: a model whose per-token terms do not depend on the segmentation gives the
    same loss and gradient for 1, 2 and 8 segments, so the trainer's weights are per logical rollout."""
    base = Bigram(V, 8, seed=3)
    ref = None
    for n_seg in (1, 2, 8):
        recs = [_segmented(n_seg, seed=5, tree=True).record(rid="a", reward=1.0), _segmented(n_seg, seed=6, tree=True).record(rid="b", reward=0.0)]
        m = copy.deepcopy(base)
        tr = TorchTrainer(m, TrainerConfig(lr=1e-2, token_budget=1 << 20, loss_agg=mode, correction=Correction(correction)))
        assert resolve_agg(mode, correction) in LOSS_AGG
        met = tr.step(recs, ["g", "g"])
        grads = {n: q.grad.detach().clone() for n, q in m.named_parameters() if q.grad is not None}
        if ref is None:
            ref = (met, grads)
            continue
        assert met["train/loss"] == pytest.approx(ref[0]["train/loss"], abs=1e-6) and met["train/tokens"] == ref[0]["train/tokens"]
        assert max(float((grads[k] - ref[1][k]).abs().max()) for k in grads) < 1e-6


# ---- over-length rows: masked and counted, never tail-cut ----------------------------------------------------------------
def test_overlength_rows_are_masked_and_counted_never_cut():
    ep = Ep(11)
    for _ in range(6):
        ep.turn()
    ep.resume("")
    ep.turn()
    ro = ep.record()
    seqs = flatten(ro, 1.0)
    lens = [len(s.prompt_ids) + len(s.response_ids) for s in seqs]
    cap = min(lens)  # the long first segment is over it, the short second one fits
    assert max(lens) > cap
    tr = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(token_budget=1 << 20, max_seq_len=cap))
    seen: list[int] = []
    orig = tr._plain_logprobs

    def spy(items, idx):
        seen.extend(len(items[i].prompt) + len(items[i].completion) for i in idx)
        return orig(items, idx)

    tr._plain_logprobs = spy
    two = [ro, ro.model_copy(update={"rollout_id": "r2", "reward": 0.0})]
    m = tr.step(two, ["g", "g"])
    long_trained = sum(sum(s.response_mask) for s in seqs if len(s.prompt_ids) + len(s.response_ids) > cap)
    assert m["train/rows_overlength"] == 2 * sum(1 for x in lens if x > cap) and m["train/tokens_overlength"] == 2 * long_trained
    assert seen and all(x <= cap for x in seen) and sorted(set(seen)) == sorted({x for x in lens if x <= cap})  # whole rows only
    # no cap: every row is forwarded whole (the trainer never cuts a row)
    tr2 = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(token_budget=8))
    m2 = tr2.step(two, ["g", "g"])
    assert m2["train/forward_tokens"] == 2 * sum(lens) and "train/rows_overlength" not in m2
    # every row over the cap: nothing trains, and the loss of signal is counted
    m3 = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(max_seq_len=1)).step(two, ["g", "g"])
    assert m3["train/items"] == 0.0 and m3["train/rows_overlength"] == 2 * len(lens)


# ---- B8: measured segment sizes ------------------------------------------------------------------------------------------
def test_B8_segments_carry_measured_tokens_and_their_source():
    ep = Ep(12)
    for _ in range(3):
        ep.turn()
    ep.compaction(accepted=[True])
    ep.turn()
    ro = ep.record()
    sg = ro.segments[0]
    assert sg.tokens_source == "usage"
    assert (sg.context_before, sg.context_after) == (ep.segments[0]["context_before"], ep.segments[0]["context_after"]) and sg.context_after > 0
    assert sg.agent == "" and sg.boundary_reason == "compaction" and sg.attempts == 1
    # legacy segment dicts (actors/policy.py, actors/rlm/policy.py) still build: strategy -> boundary_reason, source unstated
    legacy = build(RunRecord(run_id="l", task="t", world="w", policy="p", started_at="2026-01-01T00:00:00Z", steps=1, outcome="completed", reward=1.0),
                   calls={"agent": ep.calls}, segments={"agent": [{"index": 1, "strategy": "summary", "summary_call": "agent:c4", "context_before": 900,
                                                                   "context_after": 120, "kept_turns": 3},
                                                                  {"index": 2, "strategy": "rerender", "summary_call": None}]})
    assert [(s.boundary_reason, s.tokens_source) for s in legacy.segments] == [("compaction", None), ("rerender", None)]
    with pytest.raises(pydantic.ValidationError):
        build(RunRecord(run_id="x", task="t", world="w", policy="p", started_at="2026-01-01T00:00:00Z", steps=1, outcome="completed", reward=1.0),
              calls={"agent": ep.calls}, segments={"agent": [{"index": 1, "tokens_source": "chars"}]})


def test_old_records_without_m16_fields_build_as_before():
    """Absent fields = the old behaviour: kind="compaction" without an edge extends the chain (actors/policy.py), subs are
    nodes, and a record round-trips through JSON with the new defaults."""
    ep = Ep(13)
    ep.turn()
    c = ep.turn()
    ep.sub(c)
    for call in ep.calls:
        for k in ("agent", "depth", "edge"):
            call.pop(k, None)
    ctx = ep.ctx[""]
    m = ep.mark[""]
    ep.calls.append({"call_id": "agent:c9", "kind": "compaction", "segment": 0, "actions": [],
                     "tokens": {"prompt_token_ids": None, "prompt_delta": ctx[m:] + [7, MARK], "prompt_prefix_len": m, "completion_token_ids": [12, 13],
                                "logprobs": [-1.0, -1.0], "policy_version": 0}})
    ro = ep.record(mode="train")
    assert ro.valid_for_training() == (True, "ok")
    t = ro.turns[-1]
    assert t.kind == "compaction" and not t.node() and t.prompt_is_delta and t.agent == "" and t.edge is None
    assert RolloutRecord.model_validate_json(ro.model_dump_json(by_alias=True)) == ro
    assert Turn(call_id="x", actor="a").chain() == ("a", "", 0)


def test_critic_obeys_max_seq_len():
    from crucible.train.critic import CriticConfig, TinyValue

    ep = Ep(14)
    for _ in range(6):
        ep.turn()
    ep.resume("")
    ep.turn()
    ro = ep.record()
    fp = full_prompts(ro)
    lens = [len(fp[t.call_id]) + len(t.completion_token_ids) for t in ro.turns]
    cap = sorted(lens)[len(lens) // 2]
    critic = TinyValue(V, 8, seed=2)
    seen: list[int] = []
    orig = critic.token_values

    def spy(prompt, comp):
        seen.append(len(prompt) + len(comp))
        return orig(prompt, comp)

    critic.token_values = spy
    tr = TorchTrainer(TinyPolicy(V, 8), TrainerConfig(item_unit="call", token_budget=1 << 20, max_seq_len=cap), critic=critic, critic_cfg=CriticConfig())
    m = tr.step([ro, ro.model_copy(update={"rollout_id": "r2", "reward": 0.0})], ["g", "g"])
    over = [x for x in lens if x > cap]
    assert over and seen and max(seen) <= cap  # never forwarded, never cut
    assert m["critic/rows_overlength"] == 2 * len(over) and m["train/rows_overlength"] == 2 * len(over)
    assert m["critic/tokens_overlength"] == 2 * sum(len(t.completion_token_ids) for t, x in zip(ro.turns, lens, strict=True) if x > cap)


# ---- M15 label: child and sub turns never splice into the root's rows ----------------------------------------------------
class CharTok:
    """A lossless toy tokenizer: one character per id."""

    eos_token = None

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(0x4E00 + int(i)) for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) - 0x4E00 for c in text]}


def test_label_rows_keep_child_and_sub_turns_out_of_the_root_chain():
    from crucible.studio.datasets import token_steps

    ep = Ep(15)
    root = ep.turn()
    ep.sub(root)
    ep.turn("scan", parent=root, edge="subagent_call")
    for _ in range(3):
        ep.turn()
        ep.turn("scan")
        c = ep.turn()
        ep.sub(c)
    ep.turn("helper", parent=c, edge="subagent_call", trainable=False, model="other")
    ep.turn("helper", trainable=False, model="other")
    ro = ep.record()
    fp = full_prompts(ro)
    steps, chains, problems = token_steps(ro.model_dump(by_alias=True, mode="json"), CharTok())
    assert not problems
    by = {t.call_id: t for t in ro.turns}
    assert {s.call_id for s in steps} == {t.call_id for t in ro.turns if t.trainable and t.completion_token_ids}
    for s in steps:  # every step's prompt is exactly the prompt its call saw: no other chain's tokens spliced in
        assert s.prompt_ids == fp[s.call_id] and s.completion_ids == by[s.call_id].completion_token_ids and s.exact
    rows = {ch["chain"]: ch for ch in chains}
    root_turns = [t for t in ro.turns if not t.agent and not t.node()]
    root_row = next(ch for ch in chains if ch["chain"].endswith(root_turns[0].call_id))
    last = root_turns[-1]
    assert root_row["input_ids"] == fp[last.call_id] + last.completion_token_ids
    trained = [i for i, m in zip(root_row["input_ids"], root_row["mask"], strict=True) if m]
    assert trained == [x for t in root_turns for x in t.completion_token_ids]  # only root completions are targets
    kids = [ch for ch in chains if ":scan:" in ch["chain"]]
    kid_turns = [t for t in ro.turns if t.agent == "scan"]
    assert len(kids) == 1 and kids[0]["input_ids"] == fp[kid_turns[-1].call_id] + kid_turns[-1].completion_token_ids
    assert not any(":helper:" in k for k in rows)  # another model's child: no row
    assert sum(1 for ch in chains if ch["kind"] == "sub") == sum(1 for t in ro.turns if t.kind == "sub")
