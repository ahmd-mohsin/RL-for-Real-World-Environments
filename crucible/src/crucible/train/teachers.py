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
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from crucible.train.rollout import RolloutRecord, full_prompts


class Teacher(Protocol):
    id: str
    vocab_id: str

    async def score(self, pairs: Sequence[tuple[list[int], list[int]]]) -> list[list[float]]: ...


def vocab_signature(tokenizer: Any) -> str:
    """A stable id of a tokenizer's vocabulary (token string -> id map + special tokens)."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else dict(tokenizer)
    blob = json.dumps(sorted(vocab.items(), key=lambda kv: kv[1]), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TorchTeacher:
    def __init__(self, model: Any, *, id: str, vocab_id: str, temperature: float = 1.0):
        self.model, self.id, self.vocab_id, self.temperature = model, id, vocab_id, temperature

    def _score(self, pairs: Sequence[tuple[list[int], list[int]]]) -> list[list[float]]:
        import torch

        with torch.no_grad():
            if hasattr(self.model, "batch_token_logprobs"):
                rows = self.model.batch_token_logprobs(list(pairs), self.temperature)
            else:
                rows = [self.model.token_logprobs(p, c, self.temperature) for p, c in pairs]
        return [[float(x) for x in r.detach().cpu().tolist()] for r in rows]

    async def score(self, pairs: Sequence[tuple[list[int], list[int]]]) -> list[list[float]]:
        return await asyncio.to_thread(self._score, pairs)


class VLLMTeacher:
    """A teacher served by vLLM: one ``/v1/completions`` request per call with the prompt + completion as token ids,
    ``max_tokens=1`` and ``prompt_logprobs=0`` (the log-prob of each prompt token under the teacher)."""

    def __init__(self, base_url: str, model: str, *, id: str, vocab_id: str, client: Any = None, timeout_s: float = 600.0):
        import httpx

        self.base, self.model, self.id, self.vocab_id = base_url.rstrip("/").removesuffix("/v1"), model, id, vocab_id
        self.client = client or httpx.AsyncClient(timeout=timeout_s)

    async def _one(self, prompt: list[int], completion: list[int]) -> list[float]:
        body = {"model": self.model, "prompt": list(prompt) + list(completion), "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
        r = await self.client.post(self.base + "/v1/completions", json=body)
        r.raise_for_status()
        pl = r.json()["choices"][0].get("prompt_logprobs") or []
        out = []
        for pos, tok in zip(range(len(prompt), len(prompt) + len(completion)), completion, strict=True):
            entry = pl[pos] if pos < len(pl) else None
            if not entry:
                raise ValueError(f"teacher {self.id}: no prompt logprob at position {pos}")
            got = entry.get(str(tok)) or entry.get(tok)
            if got is None:
                raise ValueError(f"teacher {self.id}: sampled token {tok} missing from prompt_logprobs at {pos}")
            out.append(float(got["logprob"] if isinstance(got, dict) else got))
        return out

    async def score(self, pairs: Sequence[tuple[list[int], list[int]]]) -> list[list[float]]:
        return list(await asyncio.gather(*(self._one(p, c) for p, c in pairs)))


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
        tid = self.teacher_for(source)
        jobs = []
        if tid is not None and pairs:
            jobs.append(("teacher", self.teachers[tid].score(pairs)))
        if self.reference is not None and pairs:
            jobs.append(("ref", self.reference.score(pairs)))
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
