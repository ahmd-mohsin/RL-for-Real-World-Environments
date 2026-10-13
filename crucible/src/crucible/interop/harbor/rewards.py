"""Harbor's reward-file rules, one implementation for every Crucible reader (PORT of ``harbor/verifier/verifier.py``
``_parse_reward_json`` / ``_parse_reward_text``, Apache-2.0; M11 audit H1 V5, G1 D6: patch mode accepted NaN, inf and
strings, and crashed on a valid single-key ``reward.json``).

- ``reward.json`` first: a flat object of finite numbers (booleans count, as in Harbor); an empty file, a parse error, a
  non-numeric or non-finite value is a verifier error, never a reward.
- else ``reward.txt``: exactly one finite number (Harbor's ``float(text)``: ``"1\\n0"`` is an error).
- the scalar: ``reward`` when present, else the single key, else ``reward_key``; several keys and no rule is ambiguous.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _finite(v: object) -> bool:
    return isinstance(v, int | float) and (not isinstance(v, float) or math.isfinite(v))


def parse_reward_dir(d: Path) -> tuple[dict[str, float] | None, str | None]:
    """``(rewards, None)`` or ``(None, problem)``; ``problem`` starts with the failure kind."""
    d = Path(d)
    js, txt = d / "reward.json", d / "reward.txt"
    if js.exists():
        raw = js.read_text(errors="replace")
        if not raw.strip():
            return None, "RewardFileEmptyError: reward.json is empty"
        try:
            doc = json.loads(raw)
        except ValueError:
            return None, "VerifierOutputParseError: reward.json is not JSON"
        if not isinstance(doc, dict) or not doc:
            return None, "VerifierOutputParseError: reward.json is not a flat object of numbers"
        bad = {k: v for k, v in doc.items() if not _finite(v)}
        if bad:
            return None, f"VerifierOutputParseError: non-numeric or non-finite rewards {sorted(bad)[:5]}"
        return {str(k): float(v) for k, v in doc.items()}, None
    if txt.exists():
        raw = txt.read_text(errors="replace")
        if not raw.strip():
            return None, "RewardFileEmptyError: reward.txt is empty"
        try:
            v = float(raw)
        except ValueError:
            return None, "VerifierOutputParseError: reward.txt is not one number"
        if not math.isfinite(v):
            return None, f"VerifierOutputParseError: non-finite reward {v!r}"
        return {"reward": v}, None
    return None, "RewardFileNotFoundError: no reward.json or reward.txt"


def scalar_reward(rewards: dict[str, float] | None, reward_key: str | None = None) -> tuple[float | None, str | None]:
    """Harbor reward dict → (scalar, problem). ``reward`` key, else the single key, else ``reward_key``."""
    if not rewards:
        return None, "no_reward"
    if "reward" in rewards:
        return float(rewards["reward"]), None
    if len(rewards) == 1:
        return float(next(iter(rewards.values()))), None
    if reward_key and reward_key in rewards:
        return float(rewards[reward_key]), None
    return None, f"ambiguous_reward: keys {sorted(rewards)} and no reward / reward_key"


def row_scalar(row: object, reward_key: str | None = None) -> float | None:
    """One trial's reward row (``reward.json`` as Harbor records it) → its scalar, or None (masked or ambiguous)."""
    if not isinstance(row, dict) or not row or not all(_finite(v) for v in row.values()):
        return None
    r, _ = scalar_reward({str(k): float(v) for k, v in row.items()}, reward_key)
    return r
