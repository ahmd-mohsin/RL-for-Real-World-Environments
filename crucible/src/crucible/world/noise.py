"""Environment noise at the World's choke points (M12-infra D; art. §3.8, LongCat-Flash-Thinking-2601: training under
instruction noise and tool noise of rising intensity makes agents robust to real environments).

``WorldNoise`` perturbs what the policy *sees*, never the World's state or the verifier's view:

- **instruction** (at the first observation): word typos (adjacent swap), word dropout, case flips, and an irrelevant
  distractor sentence. **Value tokens are never touched**: words the task protects (``protect``: the exact strings
  the task must reproduce), and by default anything with a digit or a non-letter character (identifiers, paths,
  amounts), quoted text, and capitalized words (names). Found live: unprotected typos changed the word a task asked
  the agent to write, and the agent faithfully wrote the typo (scored 0 for following the instruction it saw);
- **tool, before execution** (``transient_error``): the call is not executed and returns a retryable
  "temporarily unavailable" error with ``fault="tool"`` (a tool's own error the policy must handle: scored, not masked);
- **tool, after execution** (successful results only): ``truncate`` (a tail cut with a banner), ``typo`` (character
  noise inside words), ``empty`` (the output comes back empty), ``shuffle_lines`` (line order), ``drop_json_field``
  (one top-level field of a JSON object result removed).

Every probability scales with ``intensity`` ∈ [0, 1] (the curriculum's knob, ``train.noise.NoiseCurriculum``). Every
perturbation is a ``noise`` event in the trace (operation, tool), so analysis can separate noisy from clean episodes;
the RNG is seeded per episode, so a rollout's noise is reproducible.
"""

from __future__ import annotations

import fnmatch
import json
import random
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.actions import ToolResult

DISTRACTORS = (
    "Note: the office kitchen will be closed for cleaning on Friday.",
    "Reminder: quarterly badge renewals are due by the end of the month.",
    "FYI, the shared calendar was migrated last week; some old invites may show twice.",
    "Unrelated: please ignore any newsletter emails from the facilities team.",
    "Side note: the printer on the third floor is out of toner.",
)


@dataclass
class NoiseConfig:
    intensity: float = 0.0
    instruction: dict[str, float] = field(default_factory=lambda: {"typo": 0.05, "drop_word": 0.02, "case": 0.03, "distractor": 0.5})
    tool_before: dict[str, float] = field(default_factory=lambda: {"transient_error": 0.15})
    tool_after: dict[str, float] = field(default_factory=lambda: {"truncate": 0.1, "typo": 0.02, "empty": 0.05, "shuffle_lines": 0.1, "drop_json_field": 0.15})
    tools: str = "*"  # fnmatch over qualified tool names that may be perturbed
    protect: list[str] = field(default_factory=list)  # value tokens instruction noise must never alter (case-insensitive)
    max_transient_per_tool: int = 2  # a tool fails transiently at most this many times per episode (retries succeed after)

    def p(self, base: float) -> float:
        return max(0.0, min(1.0, base * self.intensity))


class WorldNoise:
    def __init__(self, cfg: NoiseConfig, seed: Any = 0):
        if not 0.0 <= cfg.intensity <= 1.0:
            raise ValueError(f"noise intensity must be in [0, 1], got {cfg.intensity}")
        self.cfg, self.rng = cfg, random.Random(f"noise|{seed}")
        self.counts: dict[str, int] = {}
        self._transient: dict[str, int] = {}

    def _hit(self, op: str, base: float) -> bool:
        if self.rng.random() < self.cfg.p(base):
            self.counts[op] = self.counts.get(op, 0) + 1
            return True
        return False

    def _typo_word(self, w: str) -> str:
        if len(w) < 4:
            return w
        i = self.rng.randrange(1, len(w) - 2)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]

    def instruction(self, text: str | None) -> tuple[str | None, list[str]]:
        if not text or self.cfg.intensity == 0:
            return text, []
        before = dict(self.counts)
        c = self.cfg.instruction
        guard = {x.lower() for x in self.cfg.protect}
        words = []
        quoted = False
        for w in text.split(" "):
            core = w.strip(".,;:!?()[]{}")
            starts_q, ends_q = core[:1] in "\"'`", core[-1:] in "\"'`"
            in_quote = quoted or starts_q
            quoted = (quoted or starts_q) and not (ends_q and len(core) > 1)
            if in_quote or core.lower() in guard or not core.isalpha() or core[:1].isupper():
                words.append(w)  # a value token: names, identifiers, numbers, quoted text, protected words
                continue
            if self._hit("instr_drop_word", c.get("drop_word", 0.0)) and len(w) > 3 and w.isalpha() and w.islower():
                continue  # only plain lowercase words: names, numbers and identifiers survive
            if self._hit("instr_typo", c.get("typo", 0.0)) and w.isalpha():
                w = self._typo_word(w)
            if self._hit("instr_case", c.get("case", 0.0)):
                w = w.upper() if w.islower() else w.lower()
            words.append(w)
        out = " ".join(words)
        if self._hit("instr_distractor", c.get("distractor", 0.0)):
            out = out + "\n\n" + self.rng.choice(DISTRACTORS)
        return out, [k for k in self.counts if self.counts[k] != before.get(k, 0)]

    def _applies(self, qualified: str) -> bool:
        return self.cfg.intensity > 0 and fnmatch.fnmatchcase(qualified, self.cfg.tools)

    def before(self, qualified: str, call_id: str) -> ToolResult | None:
        if not self._applies(qualified) or self._transient.get(qualified, 0) >= self.cfg.max_transient_per_tool:
            return None
        if self._hit("tool_transient_error", self.cfg.tool_before.get("transient_error", 0.0)):
            self._transient[qualified] = self._transient.get(qualified, 0) + 1
            return ToolResult(tool_call_id=call_id, function_name=qualified, ok=False, error="transient_unavailable", fault="tool",
                              content=f"503 Service Unavailable: {qualified} is temporarily unavailable. Retry the call.")
        return None

    def after(self, qualified: str, r: ToolResult) -> tuple[ToolResult, list[str]]:
        if not r.ok or not self._applies(qualified) or not r.content:
            return r, []
        a, ops, text = self.cfg.tool_after, [], r.content
        if self._hit("tool_empty", a.get("empty", 0.0)):
            return r.model_copy(update={"content": ""}), ["tool_empty"]
        if text.lstrip().startswith("{") and self._hit("tool_drop_json_field", a.get("drop_json_field", 0.0)):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and len(obj) > 1:
                    obj.pop(self.rng.choice(sorted(obj)))
                    text = json.dumps(obj)
                    ops.append("tool_drop_json_field")
            except json.JSONDecodeError:
                pass
        if "\n" in text and self._hit("tool_shuffle_lines", a.get("shuffle_lines", 0.0)):
            lines = text.split("\n")
            self.rng.shuffle(lines)
            text = "\n".join(lines)
            ops.append("tool_shuffle_lines")
        if a.get("typo", 0.0) and self.cfg.intensity:
            ws, hit = text.split(" "), False
            for i, w in enumerate(ws):
                if w.isalpha() and self._hit("tool_typo", a["typo"]):
                    ws[i], hit = self._typo_word(w), True
            if hit:
                text = " ".join(ws)
                ops.append("tool_typo")
        if len(text) > 80 and self._hit("tool_truncate", a.get("truncate", 0.0)):
            keep = self.rng.randrange(len(text) // 3, len(text) * 2 // 3)
            text = text[:keep] + "\n[... output truncated by the service ...]"
            ops.append("tool_truncate")
        return (r.model_copy(update={"content": text}) if ops else r), ops
