"""The TRL bridge (research m11-m12/04 §C.4, §C.7: ``GRPOTrainer`` + ``rollout_func`` is the path TRL supports as-is), against
TRL 72cf37d (``T`` = ``trl/trainer/grpo_trainer.py``, ``VG`` = ``trl/generation/vllm_generation.py``).

``make_rollout_func(rollout, num_generations=G)`` returns ``rollout_func(prompts, trainer) -> dict`` (``T:2144-2152``):

- **One Crucible World episode per received prompt.** TRL's ``RepeatSampler`` already repeats each prompt G times
  (``T:1270-1277``, ``trl/trainer/utils.py:947-951``) and the global batch is split across processes contiguously
  (``VG:648-655``), so row i of process p is copy ``(p·len(prompts) + i) mod G`` of its prompt: that is its seed. The
  output holds exactly ``len(prompts)`` rows in the received order, because TRL pairs row i with ``inputs[i]``
  (``T:2670-2677``), builds ``rewards_per_func`` with ``len(prompts)`` rows (``T:1542``, ``:1592-1597``) and groups rewards
  by ``view(-1, G)`` (``T:2694``). (M12 audit V23 / G5 D4 b: the bridge ran G episodes per row, P·G² rows.)
- **A masked episode keeps its row** (``adapters.trl_rollout``): one pad token, ``env_mask=[0]`` (``T:2199-2201`` → the
  loss mask ``completion_mask * tool_mask``, ``T:2426``), logprob 0.0 (never read) and ``crucible_reward=None``: a reward
  function returning it gives NaN (``T:1586-1587``), so the row is unscorable, out of the nan-aware group baseline, and its
  advantage is 0 (``T:2685-2740``). Never a reward of 0, never shifted onto another row (V24 / G5 D4 c). The same holds
  for an episode whose record is not train-valid (``common.untrainable``) or that splits into several sequences
  (compaction / multi-actor: TRL rows are single sequences, use the native runtime). ``crucible_masked`` says why.
- **Generation** goes through TRL's own ``vllm_generation.generate(prompts, images, num_generations)`` (``VG:550-576``). It
  returns ``(prompt_ids, completion_ids, logprobs, logprob_token_ids)``, with ``logprobs`` of shape ``(batch, len,
  num_logprobs)`` sorted by rank and NaN as None (``VG:70-99``). The behaviour logprob is the *sampled* token's entry,
  found through ``logprob_token_ids``: with ``logprobs=N`` the sampled token need not be rank 1, and with ``logprobs=0``
  it is the only entry. (V22 / G5 D4 a: the bridge called ``float()`` on the per-token list.) A None (NaN) logprob fails
  the call, so the episode is masked and never trained on a guessed value.
- **No concurrent generate.** The batch's episodes run on one event loop. A call waits until every live episode is
  waiting on the engine. Then **one** ``generate`` serves the whole round, on the loop's thread. (G5 D4 d: per-call
  ``asyncio.to_thread`` ran concurrent offline ``LLM.generate`` calls from threads.) Where ``generate`` is a collective,
  the ranks run the rounds in **lockstep**. That is server mode with several processes (``gather_object`` and
  ``broadcast_object_list``, ``VG:594-650``) and colocate with TP > 1 (``all_gather_object`` over the TP group,
  ``VG:690-705``). Each round the ranks exchange (pending calls, live episodes, token cap). Every rank calls ``generate``
  with the same batch size and cap, until no rank has a live episode. Short batches are padded with a dummy prompt whose
  output is discarded, because server mode slices the gathered batch by ``process_index·len(prompts)`` (``VG:648-655``).
  So per-rank call counts never differ.
- **Sampling** (V28). ``generate`` reads ``temperature`` and ``max_completion_length`` from the generation object at call
  time (``VG:577-584``). A round uses the largest ``max_tokens`` among its calls, capped at TRL's
  ``max_completion_length``. Each completion is then cut to its own call's cap: a prefix of a sample is a sample under
  the shorter cap. A call at a temperature other than TRL's is refused, because TRL computes π_θ at its own temperature
  (``T:766``, ``:1047``). ``finish`` is ``"stop"`` when the completion ends on an EOS or pad id (TRL's own truncation
  test, ``T:2224-2226``), and ``"length"`` when it reached the cap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.train.bridges.adapters import trl_rollout
from crucible.train.bridges.common import TrainSeq, flatten, untrainable
from crucible.train.engine import GenResult


def _sampled_logprobs(comp: list[int], lps: Any, lp_ids: Any) -> list[float]:
    """The sampled token's logprob per position from TRL's nested ``(len, num_logprobs)`` lists (``VG:563-575``)."""
    if lps is None:
        raise ValueError("TRL vllm_generation returned no logprobs (GRPOConfig vllm logprobs off): a behaviour logprob is required")
    if len(lps) != len(comp):
        raise ValueError(f"{len(lps)} logprob rows for {len(comp)} completion tokens")
    out: list[float] = []
    for k, (tok, row) in enumerate(zip(comp, lps, strict=True)):
        row = list(row) if isinstance(row, list | tuple) else [row]
        if lp_ids is not None:
            ids = list(lp_ids[k])
            if tok not in ids:
                raise ValueError(f"sampled token {tok} missing from the logprob entries at position {k}")
            v = row[ids.index(tok)]
        elif len(row) == 1:
            v = row[0]
        else:
            raise ValueError("top-k logprobs without logprob_token_ids: the sampled token's entry is unknown")
        if v is None:
            raise ValueError(f"NaN behaviour logprob at completion position {k} (vLLM reported None)")
        out.append(float(v))
    return out


@dataclass
class _Req:
    ids: list[int]
    max_tokens: int
    fut: asyncio.Future


@dataclass
class Rounds:
    """Round-batched, single-threaded generation over a TRL generation object (see the module docstring)."""

    vg: Any
    version: Callable[[], int]
    temperature: float | None = None
    stop_ids: frozenset[int] = frozenset()
    gather: Callable[[Any], list[Any]] | None = None  # all-gather of one object per rank (None: generate is not a collective)
    pad_prompt: list[int] = field(default_factory=lambda: [0])
    live: int = 0
    pending: list[_Req] = field(default_factory=list)
    calls: int = 0  # generate calls made (equal across ranks in lockstep)
    _wake: asyncio.Event | None = None
    _dead: BaseException | None = None

    def wake(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def check_temperature(self, temperature: float) -> None:
        if self.temperature is not None and abs(float(temperature) - float(self.temperature)) > 1e-9:
            raise ValueError(f"temperature {temperature} != TRL's {self.temperature}: TRL scores π_θ at its own temperature")

    async def gen(self, ids: list[int], max_tokens: int, temperature: float) -> GenResult:
        self.check_temperature(temperature)
        if self._dead is not None:
            raise RuntimeError("the generation round driver stopped") from self._dead
        fut = asyncio.get_running_loop().create_future()
        self.pending.append(_Req(list(ids), max(1, int(max_tokens)), fut))
        self.wake()
        return await fut

    def episode(self, coro):
        """Count the episode live now (before the driver's first look), until it ends."""
        self.live += 1

        async def run():
            try:
                return await coro
            finally:
                self.live -= 1
                self.wake()
        return run()

    def generate(self, prompts: list[list[int]], cap: int) -> tuple[list[list[int]], Any, Any]:
        trl_cap = getattr(self.vg, "max_completion_length", None)
        try:
            if trl_cap is not None:
                self.vg.max_completion_length = max(1, min(int(cap), int(trl_cap)))
            out = self.vg.generate(prompts, None, 1)
        finally:
            if trl_cap is not None:
                self.vg.max_completion_length = trl_cap
        self.calls += 1
        _, comp, lps, lp_ids = (tuple(out) + (None,) * 4)[:4]
        return comp, lps, lp_ids

    def result(self, prompt: list[int], comp: list[int], lps: Any, lp_ids: Any, max_tokens: int) -> GenResult:
        ids = list(comp)[:max_tokens]
        lp = _sampled_logprobs(ids, None if lps is None else list(lps)[: len(ids)], None if lp_ids is None else list(lp_ids)[: len(ids)])
        finish = "stop" if ids and ids[-1] in self.stop_ids else ("length" if len(ids) >= max_tokens else "stop")
        return GenResult(ids, lp, int(self.version()), finish, None, len(prompt))

    async def drive(self) -> None:
        self._wake = asyncio.Event()
        try:
            await self._drive()
        except BaseException as e:  # a failed collective: fail every waiting call, never leave an episode hanging
            self._dead = e
            for r in self.pending:
                if not r.fut.done():
                    r.fut.set_exception(RuntimeError(f"generation round failed: {e!r}"))
            self.pending = []
            raise

    async def _drive(self) -> None:
        while True:
            while not (self.live == 0 or (self.pending and len(self.pending) >= self.live)):
                self._wake.clear()
                await self._wake.wait()
            batch, self.pending = self.pending, []
            cap = max((r.max_tokens for r in batch), default=1)
            n = len(batch)
            if self.gather is not None:
                infos = self.gather((n, self.live, cap))
                if not any(int(k) or int(live) for k, live, _ in infos):
                    return
                n = max(int(k) for k, _, _ in infos)
                cap = max(int(c) for _, _, c in infos)
            elif n == 0:
                return
            if n == 0:
                continue
            prompts = [r.ids for r in batch] + [list(self.pad_prompt)] * (n - len(batch))
            try:
                comp, lps, lp_ids = self.generate(prompts, cap)
            except Exception as e:  # noqa: BLE001 - this round's episodes fail (masked rows); the loop goes on
                for r in batch:
                    r.fut.set_exception(e)
                continue
            for k, r in enumerate(batch):
                try:
                    r.fut.set_result(self.result(r.ids, comp[k], None if lps is None else lps[k], None if lp_ids is None else lp_ids[k], r.max_tokens))
                except Exception as e:  # noqa: BLE001
                    r.fut.set_exception(e)


def _collective(trainer: Any) -> Callable[[Any], list[Any]] | None:
    """The all-gather a lockstep round needs where TRL's ``generate`` is a collective (``VG:594``, ``:690-705``)."""
    vg = trainer.vllm_generation
    acc = getattr(trainer, "accelerator", None) or getattr(vg, "accelerator", None)
    if getattr(vg, "mode", None) == "server" and int(getattr(acc, "num_processes", 1) or 1) > 1:
        from accelerate.utils import gather_object

        return lambda obj: list(gather_object([obj]))
    if getattr(vg, "mode", None) == "colocate" and int(getattr(vg, "tensor_parallel_size", 1) or 1) > 1:
        import torch.distributed as dist

        def tp_gather(obj: Any) -> list[Any]:
            out: list[Any] = [None] * int(vg.tensor_parallel_size)
            dist.all_gather_object(out, obj, group=vg.tp_group)
            return out
        return tp_gather
    return None


def _tokenizer(trainer: Any) -> Any:
    tok = getattr(trainer, "processing_class", None)
    return getattr(tok, "tokenizer", tok)


def _stop_ids(trainer: Any) -> frozenset[int]:
    """TRL's own ``eos_and_pad`` (``T:2224``): ``eos_token_ids`` (``T:398-405``) and the pad id."""
    tok = _tokenizer(trainer)
    ids = [*(getattr(trainer, "eos_token_ids", None) or []), getattr(tok, "eos_token_id", None), getattr(tok, "pad_token_id", None)]
    return frozenset(int(x) for x in ids if isinstance(x, int))


def _pad_id(trainer: Any) -> int:
    tok = _tokenizer(trainer)
    for t in (getattr(tok, "pad_token_id", None), *(getattr(trainer, "eos_token_ids", None) or []), getattr(tok, "eos_token_id", None)):
        if isinstance(t, int):
            return t
    return 0


def trl_rounds(trainer: Any, *, gather: Callable[[Any], list[Any]] | None = None) -> Rounds:
    """The round driver over ``trainer.vllm_generation`` (version = ``trainer.state.global_step``)."""
    vg = trainer.vllm_generation
    return Rounds(vg, lambda: int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0),
                  temperature=getattr(vg, "temperature", getattr(trainer, "temperature", None)), stop_ids=_stop_ids(trainer),
                  gather=gather if gather is not None else _collective(trainer), pad_prompt=[_pad_id(trainer)])


def trl_gen(trainer: Any) -> Callable:
    """A stand-alone ``gen`` over TRL's generation for a single caller: one ``generate`` per call, no batching. A
    collective mode needs ``make_rollout_func``'s lockstep rounds, so it is refused here."""
    rounds = trl_rounds(trainer, gather=None)
    if _collective(trainer) is not None:
        raise ValueError("TRL generate is a collective in this mode (server with several processes, or TP > 1): use make_rollout_func")

    async def gen(ids: list[int], max_tokens: int, temperature: float) -> GenResult:
        rounds.check_temperature(temperature)
        cap = max(1, int(max_tokens))
        comp, lps, lp_ids = rounds.generate([list(ids)], cap)
        return rounds.result(list(ids), comp[0], None if lps is None else lps[0], None if lp_ids is None else lp_ids[0], cap)
    return gen


def make_rollout_func(rollout: Callable, *, num_generations: int, rounds_factory: Callable[[Any], Rounds] = trl_rounds,
                      temperature: float | None = None, frozen_actors: tuple[str, ...] = ()):
    """``rollout``: a ``world_env.world_rollout(...)`` function (rid, prompt, seed, gen, temperature=…) -> RolloutRecord | None.
    ``temperature`` defaults to TRL's (``vllm_generation.temperature``). Another value is refused."""
    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        rounds = rounds_factory(trainer)
        if temperature is not None:
            rounds.check_temperature(temperature)
        temp = float(temperature if temperature is not None else (rounds.temperature if rounds.temperature is not None else 1.0))
        acc = getattr(trainer, "accelerator", None)
        offset = int(getattr(acc, "process_index", 0) or 0) * len(prompts)
        step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)

        async def all_():
            jobs = [rounds.episode(rollout(f"trl-s{step}-r{offset + i}", p if isinstance(p, str) else str(p), (offset + i) % num_generations,
                                           rounds.gen, temperature=temp)) for i, p in enumerate(prompts)]
            driver = asyncio.ensure_future(rounds.drive())
            try:
                return await asyncio.gather(*jobs, return_exceptions=True)
            finally:
                rounds.wake()
                await driver

        recs = asyncio.run(all_())
        rows: list[TrainSeq | None] = []
        rewards: list[float | None] = []
        why: list[str | None] = []
        for r in recs:
            if isinstance(r, list):  # a multi-seat rollout: one record per trained seat
                r = r[0] if len(r) == 1 else ValueError(f"{len(r)} seat records: TRL rows are single sequences")
            seqs: list[TrainSeq] = []
            if isinstance(r, BaseException):
                reason = f"error: {type(r).__name__}: {r}"
            elif r is None:
                reason = "masked episode"
            else:
                reason = untrainable(r, frozen_actors)
                if reason is None:
                    seqs = flatten(r, frozen_actors=frozen_actors)
                    if len(seqs) != 1:
                        reason = f"{len(seqs)} sequences (compaction / multi-actor): TRL rows are single sequences"
            rows.append(None if reason else seqs[0])
            rewards.append(None if reason else float(r.reward))
            why.append(reason)
        return trl_rollout(rows, rewards=rewards, masked=why, pad_id=rounds.pad_prompt[0])

    return rollout_func
