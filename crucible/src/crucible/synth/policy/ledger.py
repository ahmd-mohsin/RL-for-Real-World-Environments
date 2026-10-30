"""The policy ledger: every principle and rule as a hash-chained, append-only list of entries (DESIGN §7.7 item 24).

A policy version is a ledger position: ``Policy.version`` is the number of entries replayed and ``Policy.tag`` is
``policy-v<version>@<sha of the last entry>``. Certificates and run reports record the tag they ran under.

Entry kinds:
- ``principles`` (the genesis entry; later only inside a person's answer to a ``principle_change`` escalation);
- ``add_rule``, ``amend_rule`` (``set``: param -> value; optional ``ladder``, ``statement``), ``retire_rule`` (the rule
  is moved to ``retired``, never removed);
- ``answer``: a person's answer to an escalation, with the choice and the amendment it makes (any of the kinds above,
  or none). Every answer is an entry, so even "no" is a new, recorded version.

Guards (``check_entry``; the ledger refuses an entry that fails one):
- an entry that relaxes a validity parameter (``model.VALIDITY``) or changes a principle is accepted only inside an
  ``answer`` to a ``principle_change`` escalation;
- the Part 2 total (``model.PERSON_PARAMS``) changes only inside an answer to ``budget_beyond_total`` (or a principle
  change);
- an entry from an ``inferred`` source may add a rule or tighten one, and may change a stage cap only when it records the
  Part 2 remainder that covers the increase (``data.part2_remaining_usd``); it may not retire a rule or amend a lead's
  rule in any other way;
- the chain: every entry names its parent's sha, and its own sha is the hash of its content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.policy.model import (
    AUTO_PARAMS,
    PERSON_PARAMS,
    SCHEMA,
    Escalation,
    PolicyViolation,
    Principle,
    Rule,
    Source,
    relaxes,
)

LEDGER_PATH = Path(__file__).with_name("ledger.jsonl")

EntryKind = Literal["principles", "add_rule", "amend_rule", "retire_rule", "answer"]


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def entry_sha(entry: dict[str, Any]) -> str:
    return hashlib.sha256(_canon({k: v for k, v in entry.items() if k != "sha"}).encode()).hexdigest()


class Policy(CrucibleModel):
    """The policy a ledger replays to: ordered principles, rules by id, retired rules."""

    version: int = 0
    sha: str = ""
    principles: list[Principle] = Field(default_factory=list)
    rules: dict[str, Rule] = Field(default_factory=dict)
    retired: dict[str, Rule] = Field(default_factory=dict)

    @property
    def tag(self) -> str:
        return f"policy-v{self.version}@{self.sha[:12]}"

    def rule(self, rid: str) -> Rule:
        return self.rules[rid]

    def param(self, rid: str, name: str) -> Any:
        return self.rules[rid].params[name]

    def principle(self, pid: str) -> Principle:
        return next(p for p in self.principles if p.id == pid)

    def rules_for(self, kind: str) -> list[Rule]:
        """The rules that apply to a condition kind, in ledger order; the ``*`` rules only when no other rule does."""
        named = [r for r in self.rules.values() if kind in r.applies_to]
        return named or [r for r in self.rules.values() if "*" in r.applies_to]


# ---------------------------------------------------------------------------------------------------------- applying entries

def _amendment_of(entry: dict[str, Any]) -> dict[str, Any] | None:
    if entry["kind"] == "answer":
        return entry.get("amendment")
    return entry


def _apply(pol: Policy, entry: dict[str, Any]) -> Policy:
    am = _amendment_of(entry)
    if not am:
        return pol
    kind = am["kind"]
    if kind == "principles":
        pol.principles = sorted((Principle.model_validate(p) for p in am["principles"]), key=lambda p: p.rank)
    elif kind == "add_rule":
        r = Rule.model_validate(am["rule"])
        if r.id in pol.rules or r.id in pol.retired:
            raise PolicyViolation("P-ledger", f"rule {r.id} exists: amend it, never re-add it")
        pol.rules[r.id] = r
    elif kind == "amend_rule":
        old = pol.rules[am["rule_id"]]
        doc = old.model_dump(mode="json")
        doc["params"] = {**doc["params"], **am.get("set", {})}
        for k in ("ladder", "statement", "applies_to", "enforced_at", "title"):
            if k in am:
                doc[k] = am[k]
        doc["version"] = old.version + 1
        doc["source"] = (entry.get("source") or am.get("source") or doc["source"])
        pol.rules[old.id] = Rule.model_validate(doc)
    elif kind == "retire_rule":
        pol.retired[am["rule_id"]] = pol.rules.pop(am["rule_id"])
    else:
        raise PolicyViolation("P-ledger", f"unknown entry kind {kind!r}")
    return pol


def changes_of(pol: Policy, am: dict[str, Any]) -> list[tuple[str, str, Any, Any]]:
    """(rule, param, old, new) for every parameter an amendment changes."""
    if am.get("kind") != "amend_rule":
        return []
    old = pol.rules[am["rule_id"]].params
    return [(am["rule_id"], k, old.get(k), v) for k, v in am.get("set", {}).items() if old.get(k) != v]


def check_entry(pol: Policy, entry: dict[str, Any]) -> None:
    """The ledger's guards (module docstring). Raises ``PolicyViolation``."""
    kind = entry["kind"]
    src = Source.model_validate(entry.get("source") or {"kind": "inferred"})
    esc = entry.get("escalation") or {}
    esc_kind = esc.get("kind") if kind == "answer" else None
    am = _amendment_of(entry) or {}
    if kind == "answer" and src.kind not in ("person", "lead"):
        raise PolicyViolation("P-person", "an escalation is answered by a person")
    if am.get("kind") == "principles" and pol.version > 0 and esc_kind != "principle_change":
        raise PolicyViolation("P-person", "the principles change only by a person's answer to a principle_change escalation")
    for rid, param, old, new in changes_of(pol, am):
        if relaxes(rid, param, old, new) and esc_kind != "principle_change":
            raise PolicyViolation("P-valid", f"{rid}.{param} {old!r} -> {new!r} relaxes a validity rule: only a person, through a "
                                             f"principle_change escalation")
        if (rid, param) in PERSON_PARAMS and esc_kind not in ("budget_beyond_total", "principle_change"):
            raise PolicyViolation("P-person", f"{rid}.{param} changes only by a person's answer (budget beyond the Part 2 total)")
        if src.kind == "inferred" and (rid, param) not in AUTO_PARAMS and not _tightens(rid, param, old, new):
            raise PolicyViolation("P-ledger", f"an inferred entry may only tighten {rid}.{param} or reallocate a stage cap")
        if (rid, param) in AUTO_PARAMS and kind != "answer":
            inc = sum(max(0.0, float(v) - float((old or {}).get(k, 0.0))) for k, v in (new or {}).items())
            if inc > float((entry.get("data") or {}).get("part2_remaining_usd", 0.0)) + 1e-9:
                raise PolicyViolation("P-person", f"a stage cap rises by {inc:.2f} beyond the recorded Part 2 remainder")
    if am.get("kind") == "retire_rule" and src.kind == "inferred":
        raise PolicyViolation("P-ledger", "retiring a rule removes a decision: a person's answer")
    if am.get("kind") == "amend_rule" and src.kind == "inferred" and pol.rules[am["rule_id"]].source.kind != "inferred":
        for k in ("ladder", "statement", "applies_to"):
            if k in am:
                raise PolicyViolation("P-ledger", f"an inferred entry may not rewrite the {k} of a lead's rule")


def _tightens(rid: str, param: str, old: Any, new: Any) -> bool:
    from crucible.synth.policy.model import VALIDITY

    return (rid, param) in VALIDITY and not relaxes(rid, param, old, new) and VALIDITY[(rid, param)] != "fixed"


def replay(entries: Iterable[dict[str, Any]]) -> Policy:
    """Verify the chain and every guard, and build the policy."""
    pol = Policy()
    parent = ""
    for i, e in enumerate(entries):
        if e.get("seq") != i or e.get("parent") != parent or e.get("sha") != entry_sha(e):
            raise PolicyViolation("P-ledger", f"ledger entry {i}: the chain is broken")
        check_entry(pol, e)
        pol = _apply(pol, e)
        pol.version, pol.sha = i + 1, e["sha"]
        parent = e["sha"]
    return pol


# ---------------------------------------------------------------------------------------------------------- the ledger file

class PolicyLedger:
    """An append-only JSONL ledger (the package's ``ledger.jsonl`` by default; tests pass their own)."""

    def __init__(self, path: str | Path = LEDGER_PATH) -> None:
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def policy(self) -> Policy:
        return replay(self.entries())

    def append(self, draft: dict[str, Any]) -> dict[str, Any]:
        """Check ``draft`` against the current policy, chain it and append it. Returns the stored entry."""
        es = self.entries()
        pol = replay(es)
        entry = {"schema": SCHEMA, "seq": len(es), "parent": es[-1]["sha"] if es else "", "created": draft.get("created") or _now(),
                 **{k: v for k, v in draft.items() if k not in ("seq", "parent", "sha", "schema", "created")}}
        check_entry(pol, entry)
        _apply(pol.model_copy(deep=True), entry)  # the amendment must apply (an unknown rule id, a duplicate rule)
        entry["sha"] = entry_sha(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(_canon(entry) + "\n")
        current.cache_clear()
        return entry

    def write_genesis(self, entries: Sequence[dict[str, Any]], *, created: str = "2026-10-30T00:00:00Z") -> None:
        if self.entries():
            raise PolicyViolation("P-ledger", f"{self.path} already holds a ledger: append, never rewrite")
        for e in entries:
            self.append({**e, "created": created})

    def answer(self, esc: Escalation, option: str, *, by: str, date: str, rationale: str, queue: EscalationQueue | None = None) -> dict[str, Any]:
        """A person's answer to an escalation: one ledger entry holding the escalation, the chosen option and its amendment."""
        opt = next((o for o in esc.options if o["id"] == option), None)
        if opt is None:
            raise ValueError(f"escalation {esc.id} has no option {option!r} ({[o['id'] for o in esc.options]})")
        entry = self.append({"kind": "answer", "escalation": {"id": esc.id, "kind": esc.kind, "question": esc.question,
                                                             "condition": esc.condition.model_dump(mode="json"), "asked_under": esc.policy_version},
                             "choice": option, "amendment": opt.get("amendment"), "rationale": rationale,
                             "source": Source(kind="person", who=by, date=date, ref=f"escalation {esc.id}", escalation=esc.id).model_dump(mode="json")})
        if queue is not None:
            queue.mark_answered(esc.id, entry["sha"])
        return entry


class EscalationQueue:
    """Open questions for a person (JSONL; an answer appends a row, nothing is rewritten)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def add(self, esc: Escalation) -> Escalation:
        if any(r.get("id") == esc.id for r in self._rows() if "kind" in r):
            return esc  # the same question is asked once
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(esc.model_dump_json() + "\n")
        return esc

    def mark_answered(self, esc_id: str, sha: str) -> None:
        with self.path.open("a") as fh:
            fh.write(_canon({"answered": esc_id, "ledger_sha": sha}) + "\n")

    def open(self) -> list[Escalation]:
        rows = self._rows()
        done = {r["answered"] for r in rows if "answered" in r}
        return [Escalation.model_validate(r) for r in rows if "kind" in r and r["id"] not in done]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@lru_cache(maxsize=1)
def current() -> Policy:
    """The policy the package ledger replays to (cached; ``PolicyLedger.append`` clears the cache)."""
    return PolicyLedger(LEDGER_PATH).policy()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crucible.synth.policy.ledger")
    ap.add_argument("--path", default=str(LEDGER_PATH))
    ap.add_argument("--genesis", action="store_true", help="write the genesis entries into an empty ledger")
    a = ap.parse_args(argv)
    led = PolicyLedger(a.path)
    if a.genesis:
        from crucible.synth.policy.genesis import genesis_entries

        led.write_genesis(genesis_entries())
    pol = led.policy()
    print(pol.tag, f"{len(pol.principles)} principles, {len(pol.rules)} rules, {len(pol.retired)} retired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LEDGER_PATH", "EscalationQueue", "Policy", "PolicyLedger", "changes_of", "check_entry", "current", "entry_sha", "replay"]
