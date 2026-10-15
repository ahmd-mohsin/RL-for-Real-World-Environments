"""Teacher pool for on-policy distillation and the KL reference (m13 C-5; article route ④: DeepSeek full-vocabulary OPD,
Kimi MOPD, MiMo MOPD2, GLM cross-stage OPD; A2-9..A2-14).

A finished rollout is scored by its teacher *off the critical path*: the runtime calls ``TeacherPool.teach`` while other
rollouts keep generating, under ``RuntimeConfig.teacher_deadline_s``. Scoring sets ``Turn.teacher_logprobs`` (log π_T
of each sampled completion token, given exactly the prompt the policy saw) and ``RolloutRecord.teacher_id``; a
``reference`` scorer sets ``Turn.ref_logprobs`` for the KL-to-reference term. The trainer turns them into the OPD
signal (``TrainerConfig.opd_coef``) and the k3 KL penalty (``kl_coef`` / ``kl_coef_by_source``).

- **Routing (A2-11)**: ``route`` maps a rollout's source to a teacher id (MOPD: one expert per domain / effort level);
  unrouted sources use ``default``; ``None`` = no teacher for that source (RL only).
- **Tokenizer gate (A2-12)**: every teacher declares ``vocab_id`` (``vocab_signature(tokenizer)``); a pool refuses a
  student whose vocabulary differs, before any rollout: log-probs of token ids are meaningless across vocabularies.
- **Teachers**: ``TorchTeacher`` (a local model with ``token_logprobs`` / ``batch_token_logprobs``, run in a thread
  under ``no_grad``) and ``VLLMTeacher`` (a vLLM server: ``/v1/completions`` with ``prompt_logprobs``, the teacher's
  log-prob of every prompt token, of which the completion's are kept).

**Same distribution as the student (M12 audit H1 V7).** Behaviour log-probs are taken at the turn's sampling
temperature and, under top-p, renormalized over the nucleus the token was drawn from (``Turn.candidates``); the trainer
evaluates π_θ the same way. A teacher or reference scored at temperature 1 over the full vocabulary gave a non-zero OPD
signal and k3 KL even when it *was* the policy (top_p 0.8: β·log(nucleus mass) < 0 on every token). So ``score`` takes,
per pair, the turn's ``temperatures`` and its ``candidates``, and returns log π_T(token | prompt) at that temperature,
renormalized over those candidates when given: ``Turn.teacher_logprobs`` / ``Turn.ref_logprobs`` are directly comparable
with ``Turn.logprobs`` and with π_θ recomputed at the turn's sampling parameters.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from crucible.core.schema.rollout import RolloutRecord, SamplingParams, full_prompts


class Teacher(Protocol):
    id: str
    vocab_id: str

    async def score(self, pairs: Sequence[tuple[list[int], list[int]]], *, temperatures: Sequence[float | None] | None = None,
                    candidates: Sequence[list[list[int]] | None] | None = None) -> list[list[float]]: ...


def _takes_sampling(fn: Any) -> bool:
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "temperatures" in ps and "candidates" in ps


def vocab_signature(tokenizer: Any) -> str:
    """A stable id of a tokenizer's vocabulary (token string -> id map + special tokens)."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else dict(tokenizer)
    blob = json.dumps(sorted(vocab.items(), key=lambda kv: kv[1]), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TorchTeacher:
    def __init__(self, model: Any, *, id: str, vocab_id: str, temperature: float = 1.0):
        self.model, self.id, self.vocab_id, self.temperature = model, id, vocab_id, temperature

    def _score(self, pairs: Sequence[tuple[list[int], list[int]]], temperatures: Sequence[float | None] | None = None,
               candidates: Sequence[list[list[int]] | None] | None = None) -> list[list[float]]:
        """``temperatures`` / ``candidates``: per pair, the turn's sampling temperature (None: ``self.temperature``) and
        top-p nucleus per token (None: the full vocabulary)."""
        import torch

        temps = [float(t) if t is not None else self.temperature for t in (temperatures or [None] * len(pairs))]
        cands = list(candidates) if candidates is not None else [None] * len(pairs)
        with torch.no_grad():
            if hasattr(self.model, "batch_token_logprobs") and len(set(temps)) <= 1 and not any(cands):
                rows = self.model.batch_token_logprobs(list(pairs), temps[0] if temps else self.temperature)
            elif hasattr(self.model, "batch_token_logprobs") and len(set(temps)) <= 1:
                rows = self.model.batch_token_logprobs(list(pairs), temps[0], candidates=cands)
            else:
                rows = [self.model.token_logprobs(p, c, t, candidates=k) if k else self.model.token_logprobs(p, c, t)
                        for (p, c), t, k in zip(pairs, temps, cands, strict=True)]
        return [[float(x) for x in r.detach().cpu().tolist()] for r in rows]

    async def score(self, pairs: Sequence[tuple[list[int], list[int]]], *, temperatures: Sequence[float | None] | None = None,
                    candidates: Sequence[list[list[int]] | None] | None = None) -> list[list[float]]:
        return await asyncio.to_thread(self._score, pairs, temperatures, candidates)


class VLLMTeacher:
    """A teacher served by vLLM: one ``/v1/completions`` request per call with the prompt + completion as token ids,
    ``max_tokens=1`` and ``prompt_logprobs`` (the log-prob of each prompt token under the teacher).

    vLLM's prompt log-probs are the model's raw (temperature 1, full vocabulary) distribution. A turn sampled under top-p
    is scored exactly from the top-``candidates_k`` prompt log-probs: log π_T(x)/T renormalized over the turn's nucleus
    (the normalizer cancels), which needs every nucleus member among the returned entries (else the call fails, never a
    guess). A turn sampled at T ≠ 1 over the full vocabulary cannot be rebuilt from top-k entries: refused."""

    def __init__(self, base_url: str, model: str, *, id: str, vocab_id: str, client: Any = None, timeout_s: float = 600.0, candidates_k: int = 20):
        import httpx

        self.base, self.model, self.id, self.vocab_id = base_url.rstrip("/").removesuffix("/v1"), model, id, vocab_id
        self.candidates_k = candidates_k
        self.client = client or httpx.AsyncClient(timeout=timeout_s)

    @staticmethod
    def _lp(entry: Any, tok: int) -> float | None:
        got = entry.get(str(tok))
        if got is None:
            got = entry.get(tok)
        if got is None:
            return None
        return float(got["logprob"] if isinstance(got, dict) else got)

    async def _one(self, prompt: list[int], completion: list[int], temperature: float | None = None,
                   candidates: list[list[int]] | None = None) -> list[float]:
        T = 1.0 if temperature is None else float(temperature)
        if candidates is None and abs(T - 1.0) > 1e-9:
            raise NotImplementedError(f"teacher {self.id}: vLLM prompt log-probs are at temperature 1; a full-vocabulary turn sampled at T={T} "
                                      "cannot be scored exactly (use a TorchTeacher, or sample the student with top_p < 1)")
        k = self.candidates_k if candidates is not None else 0
        body = {"model": self.model, "prompt": list(prompt) + list(completion), "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": k}
        r = await self.client.post(self.base + "/v1/completions", json=body)
        r.raise_for_status()
        pl = r.json()["choices"][0].get("prompt_logprobs") or []
        out = []
        for i, (pos, tok) in enumerate(zip(range(len(prompt), len(prompt) + len(completion)), completion, strict=True)):
            entry = pl[pos] if pos < len(pl) else None
            if not entry:
                raise ValueError(f"teacher {self.id}: no prompt logprob at position {pos}")
            got = self._lp(entry, tok)
            if got is None:
                raise ValueError(f"teacher {self.id}: sampled token {tok} missing from prompt_logprobs at {pos}")
            if candidates is None:
                out.append(got)
                continue
            members = [self._lp(entry, c) for c in candidates[i]]
            if any(m is None for m in members):
                raise ValueError(f"teacher {self.id}: position {pos}: nucleus members outside the top-{k} prompt logprobs; raise candidates_k")
            zs = [m / T for m in members]  # type: ignore[operator]
            top = max(zs)
            out.append(got / T - (top + math.log(sum(math.exp(z - top) for z in zs))))
        return out

    async def score(self, pairs: Sequence[tuple[list[int], list[int]]], *, temperatures: Sequence[float | None] | None = None,
                    candidates: Sequence[list[list[int]] | None] | None = None) -> list[list[float]]:
        temps = list(temperatures) if temperatures is not None else [None] * len(pairs)
        cands = list(candidates) if candidates is not None else [None] * len(pairs)
        return list(await asyncio.gather(*(self._one(p, c, t, k) for (p, c), t, k in zip(pairs, temps, cands, strict=True))))


class TeacherPool:
    def __init__(self, teachers: Sequence[Teacher], *, route: Mapping[str, str | None] | Callable[[str], str | None] | None = None,
                 default: str | None = None, reference: Teacher | None = None):
        self.teachers = {t.id: t for t in teachers}
        self.route, self.default, self.reference = route or {}, default, reference
        missing = [x for x in ([default] + (list(route.values()) if isinstance(route, Mapping) else [])) if x is not None and x not in self.teachers]
        if missing:
            raise ValueError(f"routes name unknown teachers: {sorted(set(missing))}")
        self.seconds = 0.0

    def check_vocab(self, student_vocab_id: str) -> None:
        bad = {t.id: t.vocab_id for t in [*self.teachers.values(), *([self.reference] if self.reference else [])] if t.vocab_id != student_vocab_id}
        if bad:
            raise ValueError(f"teacher vocabularies differ from the student's ({student_vocab_id}): {bad}; token log-probs are not comparable")

    def teacher_for(self, source: str | None) -> str | None:
        if callable(self.route):
            return self.route(source or "")
        return self.route.get(source or "", self.default) if (source or "") in self.route else self.default

    async def teach(self, rec: RolloutRecord, source: str | None) -> RolloutRecord:
        """Score every completion of ``rec`` with its routed teacher (and the reference, if any)."""
        import time

        t0 = time.monotonic()
        prompts = full_prompts(rec)
        calls = [t for t in rec.turns if t.completion_token_ids]
        pairs = [(prompts[t.call_id], list(t.completion_token_ids)) for t in calls]
        temps, cands = sampling_of(calls)
        tid = self.teacher_for(source)
        jobs = []
        if tid is not None and pairs:
            jobs.append(("teacher", _score(self.teachers[tid], pairs, temps, cands)))
        if self.reference is not None and pairs:
            jobs.append(("ref", _score(self.reference, pairs, temps, cands)))
        results = await asyncio.gather(*(j for _, j in jobs))
        for (kind, _), rows in zip(jobs, results, strict=True):
            for t, row in zip(calls, rows, strict=True):
                if len(row) != len(t.completion_token_ids):
                    raise ValueError(f"{kind} scored {len(row)} of {len(t.completion_token_ids)} tokens for {t.call_id}")
                if kind == "teacher":
                    t.teacher_logprobs = row
                else:
                    t.ref_logprobs = row
        if tid is not None and pairs:
            rec.teacher_id = tid
        self.seconds += time.monotonic() - t0
        return rec


def sampling_of(turns: Sequence[Any]) -> tuple[list[float | None], list[list[list[int]] | None]]:
    """Per turn, the sampling temperature (``Turn.sampling.temperature``, None if unrecorded) and the candidate set of
    every completion token (``Turn.candidates`` when the turn was sampled truncated: top_p < 1, top_k or min_p)."""
    temps: list[float | None] = []
    cands: list[list[list[int]] | None] = []
    for t in turns:
        s = getattr(t, "sampling", None)
        s = s if isinstance(s, SamplingParams) else SamplingParams.model_validate(s or {})
        temps.append(float(s.temperature) if s.temperature is not None else None)
        nucleus = s.truncated or bool(t.candidates)
        cands.append([list(c) for c in t.candidates] if nucleus and t.candidates else None)
        if nucleus and not t.candidates:
            raise ValueError(f"{t.call_id}: sampled truncated (top_p < 1, top_k or min_p) but no candidate sets: the teacher cannot score the student's distribution")
    return temps, cands


async def _score(teacher: Any, pairs: list[tuple[list[int], list[int]]], temps: list[float | None], cands: list[Any]) -> list[list[float]]:
    if _takes_sampling(teacher.score):
        return await teacher.score(pairs, temperatures=temps, candidates=cands)
    if any(c is not None for c in cands) or any(t is not None and abs(t - 1.0) > 1e-9 for t in temps):
        raise ValueError(f"teacher {teacher.id} scores at temperature 1 over the full vocabulary only; these turns were sampled at "
                         f"{sorted({t for t in temps if t is not None})} / top-p: its log-probs would not compare with the student's (M12 audit H1 V7)")
    return await teacher.score(pairs)
