"""Vision-language models on the trainer side (M12; M1 item 8 media): log-prob recomputation from a record's image refs.

A vision record (``chat.HFVisionChatCodec`` through ``EngineLLM``) keeps the token ids exactly as the engine saw them,
placeholder runs already expanded, plus ``Turn.media``: per call, the images its prompt added as content addresses
with the placeholder offset (in the call's full prompt), length and patch grid. No pixels.

``VisionHFPolicy`` is ``HFPolicy`` for a Qwen2-VL-style model (``Qwen2VLForConditionalGeneration`` and kin):

- ``pixel_inputs(ids, media)`` rebuilds the vision inputs: the bytes from ``loader(ref)`` (the codec's store, or a
  ``core.store.Store``), the codec's image processor with the codec's fixed kwargs, giving ``pixel_values`` and
  ``image_grid_thw``; ``mm_token_type_ids`` (needed for M-RoPE) from the ids. Every image is checked: the ids hold its
  full placeholder run at the recorded offset, and the recomputed grid equals the recorded one (prime-rl's materialize
  check), so a record whose images were resized differently fails loudly instead of training on shifted positions;
- ``token_logprobs(prompt, completion, temperature, media=...)`` runs the decoder with those inputs and the same
  FP32 LM head and chunked log-probs as ``HFPolicy``; without media it is ``HFPolicy`` unchanged;
- ``record_logprobs(ro)`` recomputes every call of a record (full prompts from delta storage, every image of the call's
  segment from ``prompt_media``).

Pattern: prime-rl ``trainer/multimodal.py`` (``materialize_mm_refs``) and ``multimodal/qwen_vl.py``; verl
``models/transformers/qwen2_vl.py`` (vision inputs beside ``input_ids`` in the actor forward). Written anew.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from crucible.core.schema.rollout import RolloutRecord, full_prompts
from crucible.train.chat import load_image
from crucible.train.hf import HFPolicy


def prompt_media(ro: RolloutRecord) -> dict[str, list[dict[str, Any]]]:
    """Every image in each call's full prompt: within an (actor, segment) chain a delta prompt keeps the images of the
    previous calls (their offsets are absolute in the growing prompt) and adds its own; a whole prompt starts afresh."""
    out: dict[str, list[dict[str, Any]]] = {}
    prev: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for t in ro.turns:
        own = [dict(m) for m in t.media or []]
        if t.kind == "sub":
            out[t.call_id] = own
            continue
        key = (t.actor, t.segment)
        full = [*prev.get(key, []), *own] if t.prompt_is_delta else own
        out[t.call_id] = full
        prev[key] = full
    return out


class VisionHFPolicy(HFPolicy):
    def __init__(self, model: torch.nn.Module, image_processor: Any, loader: Callable[[str], bytes], *, image_token_id: int,
                 processor_kwargs: dict | None = None, **kw: Any):
        super().__init__(model, **kw)
        self.image_processor, self.loader, self.image_token_id = image_processor, loader, int(image_token_id)
        self.pkw = dict(processor_kwargs or {})
        self.merge = int(getattr(image_processor, "merge_size", 1) or 1)

    @classmethod
    def from_codec(cls, model: torch.nn.Module, codec: Any, **kw: Any) -> VisionHFPolicy:
        """Same image processor, kwargs, placeholder token and store as the codec that rendered the rollouts."""
        return cls(model, codec.image_processor, codec.load, image_token_id=codec.image_token_id, processor_kwargs=codec.pkw, **kw)

    def pixel_inputs(self, ids: Sequence[int], media: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        media = sorted(media, key=lambda m: int(m["offset"]))
        for m in media:
            o, n = int(m["offset"]), int(m["length"])
            if o + n > len(ids) or any(int(t) != self.image_token_id for t in ids[o:o + n]) or (o + n < len(ids) and int(ids[o + n]) == self.image_token_id):
                raise ValueError(f"image {m['ref'][:19]}: no placeholder run of {n} tokens at offset {o}")
        if sum(int(t) == self.image_token_id for t in ids) != sum(int(m["length"]) for m in media):
            raise ValueError("the sequence holds image placeholders the record has no image for")
        enc = self.image_processor(images=[load_image(self.loader(m["ref"])) for m in media], return_tensors="pt", **self.pkw)
        grid = torch.as_tensor(enc["image_grid_thw"]).reshape(-1, 3)
        for m, g in zip(media, grid.tolist(), strict=True):
            if m.get("grid") is not None and [int(x) for x in m["grid"]] != [int(x) for x in g]:
                raise ValueError(f"image {m['ref'][:19]}: recomputed grid {g} != recorded {m['grid']} (processor kwargs changed?)")
            if int(g[0]) * int(g[1]) * int(g[2]) // (self.merge * self.merge) != int(m["length"]):
                raise ValueError(f"image {m['ref'][:19]}: grid {g} does not match {m['length']} placeholders")
        ids_t = torch.tensor([list(ids)], device=self.device)
        return {"pixel_values": torch.as_tensor(enc["pixel_values"]).to(self.device), "image_grid_thw": grid.to(self.device),
                "mm_token_type_ids": (ids_t == self.image_token_id).long()}

    def _hidden_mm(self, ids: torch.Tensor, mm: dict[str, torch.Tensor]) -> torch.Tensor:
        base = getattr(self.model, getattr(self.model, "base_model_prefix", "model"), None)

        def run() -> torch.Tensor:
            if base is not None and base is not self.model:
                return base(input_ids=ids, **mm).last_hidden_state
            return self.model(input_ids=ids, output_hidden_states=True, **mm).hidden_states[-1]

        if self.autocast_dtype is not None:
            with torch.autocast(device_type=ids.device.type, dtype=self.autocast_dtype):
                return run()
        return run()

    def token_logprobs(self, prompt: list[int], completion: list[int], temperature: float = 1.0, candidates: list[list[int]] | None = None,
                       media: Sequence[dict[str, Any]] | None = None) -> torch.Tensor:
        if not media:
            return super().token_logprobs(prompt, completion, temperature, candidates)
        if not completion:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        seq = list(prompt) + list(completion[:-1])
        h = self._hidden_mm(torch.tensor([seq], device=self.device), self.pixel_inputs(seq, media))[0, len(prompt) - 1: len(prompt) - 1 + len(completion)]
        if candidates is not None:
            from crucible.train.sampling import renormalized

            return renormalized(torch.log_softmax(self._head_rows(h, temperature), -1), completion, candidates)
        return self._head_logprobs(h, torch.tensor(completion, device=self.device), temperature)

    def batch_token_logprobs(self, pairs: Sequence[tuple[list[int], list[int]]], temperature: float = 1.0,
                             candidates: Sequence[list[list[int]] | None] | None = None,
                             media: Sequence[Sequence[dict[str, Any]] | None] | None = None) -> list[torch.Tensor]:
        """Text-only pairs batch as in ``HFPolicy``; a pair with images runs on its own (its pixel inputs are its own)."""
        if not media or not any(media):
            return super().batch_token_logprobs(pairs, temperature, candidates=candidates)
        return [self.token_logprobs(p, c, temperature, candidates[i] if candidates else None, media[i]) for i, (p, c) in enumerate(pairs)]

    def record_logprobs(self, ro: RolloutRecord, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        """Recompute the log-prob of every completion token of a record, each call with its full prompt and images."""
        prompts, media = full_prompts(ro), prompt_media(ro)
        return {t.call_id: self.token_logprobs(prompts[t.call_id], list(t.completion_token_ids or []), temperature, t.candidates, media[t.call_id])
                for t in ro.turns}
