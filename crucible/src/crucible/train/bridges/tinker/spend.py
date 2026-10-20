"""The spend meter and its hard cap (swarm-pipeline/09 §4).

Every Tinker call is booked as tokens by kind and priced from the pinned table (``models.PRICES_AS_OF``):

- **sample**: prefill tokens, split into cached and uncached by the service's own count (``SampleResponse.
  prompt_cache_hit_tokens``, billed at the cached rate); a response without the count is booked all uncached (the
  conservative side). Sampled tokens at the sample rate.
- **train**: every ``forward_backward`` books the datums' input tokens at the model's train rate (the variant's: a
  ``:peft:262144`` run pays its own rate).
- **checkpoint**: counted (``save_weights_for_sampler``, ``save_state``); storage is billed per GB-month, not per call,
  so it is listed in the receipt as not metered.

**The cap is hard.** Before a call the meter *reserves* its worst case (a sample: the whole prompt uncached plus
``max_tokens`` sampled; a train call: its tokens) and refuses with ``TinkerCapReached`` when spent + held + this call
would pass the cap; the reservation is settled with the measured cost when the call returns. Calls in flight count, so a
burst of concurrent episodes cannot overshoot (the M2 audit's ``LLMClient._reserve`` rule). ``TinkerCapReached`` is an
``actors.llm.BudgetExceeded``: inside an episode the policy ends it as ``crucible:credits_exhausted`` (masked, never a
0), and the trainer stops the run.

**Receipts.** Each booked call is one JSONL row (``spend.jsonl``: kind, model, tokens, usd, cumulative usd, the sampler
version, the episode key), written before the next call is allowed; ``summary()`` is the receipt's ``spend`` block. A
meter re-opened on an existing file starts from the booked total, so a resumed run keeps one cap across restarts.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.actors.llm import BudgetExceeded, book_scoped
from crucible.train.bridges.tinker.models import PRICES_AS_OF, PRICES_SOURCE, Prices, TinkerModel


class TinkerCapReached(BudgetExceeded):
    """The run's Tinker dollar cap: the call was refused before it was sent."""


@dataclass
class Totals:
    usd: float = 0.0
    calls: int = 0
    prefill: int = 0
    prefill_cached: int = 0
    sample: int = 0
    train: int = 0
    checkpoints: int = 0
    by_kind_usd: dict[str, float] = field(default_factory=dict)
    by_model_usd: dict[str, float] = field(default_factory=dict)


class SpendMeter:
    """One cap for a run (or a smoke, or a benchmark pass): every sampler and trainer of it books here."""

    def __init__(self, cap_usd: float, *, path: Path | None = None):
        if not cap_usd >= 0:
            raise ValueError("cap_usd must be a non-negative number")
        self.cap_usd = float(cap_usd)
        self.path = Path(path) if path is not None else None
        self.totals = Totals()
        self._held = 0.0
        self._lock = threading.Lock()
        self.refused = 0
        if self.path is not None and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._add(json.loads(line))

    # ---- state ----
    @property
    def spent(self) -> float:
        return self.totals.usd

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.totals.usd - self._held)

    @property
    def exhausted(self) -> bool:
        """The cap is reached, or a call was already refused for it (the run must stop)."""
        return self.refused > 0 or self.totals.usd >= self.cap_usd - 1e-12

    # ---- reserve / book ----
    @contextmanager
    def hold(self, est_usd: float, what: str) -> Iterator[None]:
        """Reserve ``est_usd`` for one call (``TinkerCapReached`` when the cap cannot pay it), release it on exit."""
        with self._lock:
            if self.totals.usd + self._held + est_usd > self.cap_usd + 1e-12:
                self.refused += 1
                raise TinkerCapReached(f"Tinker cap {self.cap_usd:.4f} USD: {what} (worst case {est_usd:.5f}) refused; {self.totals.usd:.5f} spent, "
                                       f"{self._held:.5f} held by calls in flight")
            self._held += est_usd
        try:
            yield
        finally:
            with self._lock:
                self._held -= est_usd

    def _add(self, row: dict[str, Any]) -> None:
        t = self.totals
        usd = float(row.get("usd") or 0.0)
        t.usd += usd
        t.calls += 1
        for k in ("prefill", "prefill_cached", "sample", "train"):
            setattr(t, k, getattr(t, k) + int(row.get(k) or 0))
        if row.get("kind") == "checkpoint":
            t.checkpoints += 1
        t.by_kind_usd[row["kind"]] = t.by_kind_usd.get(row["kind"], 0.0) + usd
        t.by_model_usd[row.get("model") or "?"] = t.by_model_usd.get(row.get("model") or "?", 0.0) + usd

    def book(self, kind: str, model: TinkerModel | None, *, prefill: int = 0, prefill_cached: int = 0, sample: int = 0, train: int = 0,
             **extra: Any) -> float:
        """Price and record one call; returns its USD. ``model`` None (a checkpoint) costs 0 here."""
        prices: Prices | None = model.prices if model is not None else None
        usd = prices.usd(prefill=prefill, prefill_cached=prefill_cached, sample=sample, train=train) if prices else 0.0
        row = {"t": round(time.time(), 3), "kind": kind, "model": model.name if model else None, "prefill": int(prefill), "prefill_cached": int(prefill_cached),
               "sample": int(sample), "train": int(train), "usd": round(usd, 9), **{k: v for k, v in extra.items() if v is not None}}
        with self._lock:
            self._add(row)
            row["cum_usd"] = round(self.totals.usd, 9)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        book_scoped(usd)  # a factory SpendScope around the caller (a Tinker model in a factory role) sees it too
        return usd

    def summary(self) -> dict[str, Any]:
        t = self.totals
        return {"usd": round(t.usd, 6), "cap_usd": self.cap_usd, "calls": t.calls, "tokens": {"prefill_uncached": t.prefill, "prefill_cached": t.prefill_cached,
                "sample": t.sample, "train": t.train}, "checkpoints": t.checkpoints, "by_kind_usd": {k: round(v, 6) for k, v in t.by_kind_usd.items()},
                "by_model_usd": {k: round(v, 6) for k, v in t.by_model_usd.items()}, "refused_calls": self.refused, "prices_as_of": PRICES_AS_OF,
                "prices_source": PRICES_SOURCE, "not_metered": ["checkpoint storage (USD 0.10 per GB-month, billed by size and time)"]}


def sample_estimate(model: TinkerModel, prompt_tokens: int, max_tokens: int) -> float:
    """The worst case of one sample call: the whole prompt uncached, every token of ``max_tokens`` sampled."""
    return model.prices.usd(prefill=prompt_tokens, sample=max_tokens)
