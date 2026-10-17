"""Self-calibration for long-horizon coding tasks (study 08 §7.2; intent I5): the agent's own claims, scored against the
verifier with a proper scoring rule, and an honest way to stop.

- **Claims.** At the end the agent writes ``STATUS.json`` at the repo root: ``{requirement_id: probability met}``. It is
  read from the final tree (``status_from_tree``) or the final patch (``status_from_patch``: a file the patch creates).
  A missing, unreadable or non-object file is **no claim**: every requirement gets p = 0.5 (no information, Brier 0.25).
  A key that is not a requirement the verifier checks is ignored (listed in ``ignored``), so nothing unverifiable is ever
  scored, let alone rewarded; a value that is not a number in [0, 1] (bools count as 1 / 0) is treated as missing.
- **Scoring.** ``brier`` = mean over the verified requirements of (p − y)², a proper scoring rule: for a requirement met
  with probability q the expected score is minimised by reporting p = q (tested numerically). The reward term is
  ``-λ · brier`` (λ = 0.1 by default) with its components exposed (``HonestyScore.components``).
- **Giving up.** ``GIVE_UP: <reason>`` at the start of a line of the final assistant message (or the RLM's
  ``rlm.give_up``, which ends with ``Done(reason="gave up: ...")``) ends the episode honestly: the outcome is scored as it
  stands and nothing else changes. A correct "not done" claim (p = 0 on an unmet requirement) costs nothing.
- **Composition.** The term never enters the outcome reward: folded into R before group centring, an honest failure in
  an all-fail group would gain a *positive* advantage over a lying one. It is added to the outcome **advantage**
  (``compose_advantage``; ``honesty_group_fn`` for the M12 group stage): the term is ≤ 0, and the result keeps the
  outcome advantage's sign (a failure never becomes positive; a success is floored at 0), as M6's turn credit does.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATUS_FILE = "STATUS.json"
NO_CLAIM = 0.5  # p for a requirement without a valid claim: no information (Brier 0.25)
DEFAULT_LAMBDA = 0.1
MAX_STATUS_BYTES = 256_000
_GIVE_UP = re.compile(r"(?m)^\s*GIVE_UP:\s*(.*\S)?")
_RLM_GAVE_UP = "gave up: "  # actors.rlm.agent: rlm.give_up() ends with Done(reason="gave up: <reason>")


def detect_give_up(text: str | None) -> str | None:
    """The reason of a ``GIVE_UP: <reason>`` line (or of the RLM's ``gave up: <reason>`` stop), else None. An empty
    reason is still a give-up (reason "")."""
    if not text:
        return None
    m = _GIVE_UP.search(text)
    if m:
        return (m.group(1) or "").strip()[:2000]
    if text.startswith(_RLM_GAVE_UP):
        return text[len(_RLM_GAVE_UP):].strip()[:2000]
    return None


def give_up_of_action(action: Any) -> str | None:
    """A give-up carried by the episode's final action (``Done``: its ``answer`` when the reason is a cut of it)."""
    if getattr(action, "kind", None) != "done":
        return None
    return detect_give_up(getattr(action, "answer", None) or getattr(action, "reason", None))


def parse_status(text: str | bytes | None) -> dict[str, float] | None:
    """``STATUS.json`` text -> {key: p}; None when it is not a JSON object (no claim). Invalid values are dropped."""
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) > MAX_STATUS_BYTES:
        return None
    try:
        doc = json.loads(text)
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    out: dict[str, float] = {}
    for k, v in doc.items():
        if isinstance(v, bool):
            out[str(k)] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0:
            out[str(k)] = float(v)
    return out


def status_from_tree(root: Path | str) -> dict[str, float] | None:
    p = Path(root) / STATUS_FILE
    if not p.is_file() or p.is_symlink():
        return None
    return parse_status(p.read_bytes()[: MAX_STATUS_BYTES + 1])


def status_from_patch(patch: str) -> dict[str, float] | None:
    """The ``STATUS.json`` a git patch creates (a new file, or a full rewrite whose hunk starts at line 0); a partial edit
    of an existing file cannot be read without its base and counts as no claim."""
    blocks = re.split(r"(?m)^diff --git ", patch or "")
    for b in blocks[1:]:
        head = b.split("\n", 1)[0]
        if not head.endswith(f" b/{STATUS_FILE}"):
            continue
        if re.search(r"(?m)^deleted file mode", b):
            return None
        hunks = re.split(r"(?m)^@@ ", b)
        if len(hunks) != 2 or not re.match(r"-0,0 \+1", hunks[1]):
            return None
        lines = hunks[1].split("\n")[1:]
        body = [ln[1:] for ln in lines if ln.startswith("+")]
        return parse_status("\n".join(body))
    return None


@dataclass
class HonestyScore:
    brier: float | None  # None: no requirement was verified (nothing scored)
    term: float  # -λ · brier (0 when nothing was scored)
    lam: float
    scored: dict[str, tuple[float, int]] = field(default_factory=dict)  # requirement -> (claimed p, verified y)
    defaulted: list[str] = field(default_factory=list)  # verified requirements without a valid claim (p = 0.5)
    ignored: list[str] = field(default_factory=list)  # claimed keys the verifier does not check
    claimed: bool = False  # a STATUS.json object was found
    give_up: str | None = None

    @property
    def components(self) -> dict[str, float]:
        out = {"honesty/term": self.term, "honesty/n": float(len(self.scored)), "honesty/defaulted": float(len(self.defaulted)),
               "honesty/claimed": 1.0 if self.claimed else 0.0, "honesty/give_up": 1.0 if self.give_up is not None else 0.0}
        if self.brier is not None:
            out["honesty/brier"] = self.brier
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"brier": self.brier, "term": self.term, "lam": self.lam, "scored": {k: list(v) for k, v in self.scored.items()},
                "defaulted": self.defaulted, "ignored": self.ignored, "claimed": self.claimed, "give_up": self.give_up}


def score(claims: Mapping[str, float] | None, verified: Mapping[str, bool | None], *, requirement_ids: Sequence[str] | None = None,
          lam: float = DEFAULT_LAMBDA, give_up: str | None = None) -> HonestyScore:
    """Brier of ``claims`` against ``verified`` over the requirements the verifier checked (``verified[r]`` not None, and r
    in ``requirement_ids`` when given). ``claims`` None = no claim (p = 0.5 everywhere)."""
    if not (lam >= 0 and math.isfinite(lam)):
        raise ValueError("lam must be a finite, non-negative weight")
    reqs = list(requirement_ids) if requirement_ids is not None else list(verified)
    checked = [r for r in reqs if verified.get(r) is not None]
    c = dict(claims or {})
    scored: dict[str, tuple[float, int]] = {}
    defaulted: list[str] = []
    for r in checked:
        p = c.get(r)
        if p is None:
            p = NO_CLAIM
            defaulted.append(r)
        scored[r] = (float(p), 1 if verified[r] else 0)
    ignored = sorted(k for k in c if k not in set(checked))
    if not scored:
        return HonestyScore(None, 0.0, lam, {}, [], ignored, claims is not None, give_up)
    b = sum((p - y) ** 2 for p, y in scored.values()) / len(scored)
    return HonestyScore(b, -lam * b, lam, scored, defaulted, ignored, claims is not None, give_up)


def compose_advantage(outcome_adv: float, term: float) -> float:
    """The one place the honesty term meets the outcome: ``Â + term`` with ``term ≤ 0``. Â ≤ 0 (a failure, or a group
    without outcome spread) can only go down, so a failure never gains a positive advantage; Â > 0 is floored at 0, so a
    miscalibrated claim never turns a success into a loss. In a group without spread the term alone separates the honest
    (0) from the miscalibrated (< 0)."""
    if term > 0:
        raise ValueError("the honesty term is a penalty (≤ 0)")
    a = outcome_adv + term
    return max(a, 0.0) if outcome_adv > 0 else min(a, 0.0)


def _term_of(rec: Any) -> float:
    h = (getattr(rec, "progress", None) or {}).get("honesty") or {}
    if "term" in h:
        return float(h["term"])
    return float((getattr(rec, "reward_components", None) or {}).get("honesty/term", 0.0))


def honesty_group_fn(*, term_of: Callable[[Any], float] = _term_of) -> Callable:
    """``Runtime(group_fn=...)`` stage: each scored member's ``advantage`` = ``compose_advantage(Â, term)``, Â being an
    earlier stage's advantage when set (GAR, hack zeroing) or R − the group mean of the live outcomes (Dr.GRPO); the term
    comes from the checkpoint sidecar's ``progress["honesty"]`` (or ``reward_components["honesty/term"]``). Masked members
    are left alone. The outcome ``reward`` is never changed."""

    def fn(gid: str, members: list[Any]) -> list[Any]:
        live = [m for m in members if not getattr(m, "mask_sample", False) and m.reward is not None]
        if not live:
            return members
        mu = sum(float(m.reward) for m in live) / len(live)
        for m in live:
            a = m.advantage if getattr(m, "advantage", None) is not None else float(m.reward) - mu
            t = min(0.0, term_of(m))
            m.reward_components["honesty/term"] = t
            m.advantage = compose_advantage(float(a), t)
        return members

    return fn
