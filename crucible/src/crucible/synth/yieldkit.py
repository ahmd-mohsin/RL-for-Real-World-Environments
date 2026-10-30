"""Yield tools for the batch runner (``synth.batch``): dedup, cheap pre-filters, the instruction-clarity self-check and
the lemma batch of the synthesizer loop (part2-paper-core §9.5 S). None of them touches an admission rule: a pre-filter
only *drops* a proposal before money is spent on it, so it can lower the number admitted but never admit a task the
certificate would refuse.

**Dedup** (``Deduper``), before any paid stage. Scope: one World instance (domain, start-state seed). *Exact*: the same
normalised instruction. *Near*: either the same entity set (ids such as ``INV-54032``, ``T-1003``) and the same derive
values (the same decision on the same records, reworded), or word 3-shingle Jaccard ≥ ``near`` on the same entity set.
A template shared by different entities ("process invoice X") is not a duplicate: a different invoice is a different
decision path.

**Pre-filters** (``pre_checks``, free, before the oracle): ``unknown_entity`` (an id-like token of the instruction occurs
nowhere in the start state: the task names something the World does not have) and ``leak`` (a derive value occurs in the
instruction: TAINT's derive rule, which would confirm D7 after the oracle, the contract and the falsifiers were paid).

**Agreement** (``agreement``, free, after the oracle): with ``RunConfig.oracle_sees_derive=False`` the oracle programmer
decides without the proposer's derive values; the decision values the oracle wrote (short scalar values of its effects
and answer) must include every derive value, else two independent derivations disagree (``disagree``).

**Clarity self-check** (``clarity_screen``, one ``CLARITY_CHECKER`` call, after the oracle, before the contract writer):
a builder family other than the proposer's and the oracle programmer's solves the task on paper from what the agent will
see (instruction, tools, start state) and says whether it is clear. Dropped: ``unclear`` (it names an issue) and
``disagree`` (a decision value the oracle wrote is not among the values it derived).

**Lemma batch** (``repair_pairs``, ``lemma_clusters``, ``lemma_leaks``): (counterexample, repair) pairs from run records
(the round's counterexamples by class, operator and defect; what the accepted repair changed), clustered and shown to
the abstractor (``roles.ABSTRACTOR``: clusters only, never tasks). The lemmas it returns are refused by the deterministic
leakage critic (``meta.update.leakage_critic``: an entity key or literal of the evidence in the text) and then accepted
only by ``meta.lemmas.lemma_acceptance`` on an equal-budget arm comparison (``synth.batch``).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.synth.canon import cjson, norm_text
from crucible.synth.program import is_structural, text_spans
from crucible.synth.roles import CLARITY_CHECKER, ClarityOut, RoleClient, ask, compact, tool_catalogue

ENTITY = re.compile(r"\b[A-Z]{1,6}-\d{2,}[A-Za-z0-9-]*\b|\b[A-Z]{2,6}\d{2,}\b")  # INV-54032, T-1003, GRP-02, KB04
WORD = re.compile(r"[a-z0-9]+")
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}")  # clock values a tool writes are not decisions


# ---------------------------------------------------------------------------------------------------------- dedup

def entities(text: str) -> frozenset[str]:
    """Id-like tokens (letters then digits: INV-54032, T-1003, PO 5009), normalised (case, separators)."""
    return frozenset(re.sub(r"[-_ ]", "", m.group(0)).upper() for m in ENTITY.finditer(text or ""))


def shingles(text: str, k: int = 3) -> frozenset[tuple[str, ...]]:
    w = WORD.findall((text or "").lower())
    return frozenset(tuple(w[i:i + k]) for i in range(max(1, len(w) - k + 1))) if w else frozenset()


def exact_key(text: str) -> str:
    """Case, spacing and punctuation folded: two instructions with one key are the same proposal."""
    return " ".join(WORD.findall((text or "").lower()))


def jaccard(a: frozenset[Any], b: frozenset[Any]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _vals(xs: Iterable[Any]) -> frozenset[str]:
    return frozenset(norm_text(str(x)) for x in xs if not is_structural(x))


class Deduper:
    """Exact and near-duplicate proposals per scope (module docstring). ``check`` returns (kind, id of the earlier one)
    or (None, None); ``add`` records a kept proposal."""

    def __init__(self, near: float = 0.8):
        self.near = near
        self.items: dict[str, list[tuple[str, str, frozenset[str], frozenset[tuple[str, ...]], frozenset[str]]]] = defaultdict(list)

    def check(self, scope: str, instruction: str, derive: Sequence[Any] = ()) -> tuple[str | None, str | None]:
        ex, ents, sh, dv = exact_key(instruction), entities(instruction), shingles(instruction), _vals(derive)
        for cid, ex2, ents2, sh2, dv2 in self.items.get(scope, []):
            if ex == ex2:
                return "exact", cid
            if ents and ents == ents2 and (dv == dv2 or jaccard(sh, sh2) >= self.near):
                return "near", cid
        return None, None

    def add(self, cid: str, scope: str, instruction: str, derive: Sequence[Any] = ()) -> None:
        self.items[scope].append((cid, exact_key(instruction), entities(instruction), shingles(instruction), _vals(derive)))


# ---------------------------------------------------------------------------------------------------------- pre-filters

def pre_checks(instruction: str, derive: Sequence[Any], world_text: str) -> list[str]:
    """Free checks before the oracle (module docstring). ``world_text`` is the start state as text (every row)."""
    out: list[str] = []
    wt = re.sub(r"[-_ ]", "", world_text).upper()
    for e in sorted(entities(instruction)):
        if e not in wt:
            out.append(f"prefilter:unknown_entity:{e}")
    for d in derive:
        if not is_structural(d) and text_spans(d, instruction):
            out.append(f"prefilter:leak:{cjson(d)[:40]}")
    return out


def decisions(effects: Sequence[Mapping[str, Any]], answer: Any = None) -> dict[str, Any]:
    """The short scalar values a run wrote or reported (statuses, codes, ids, amounts; not free text): normalised -> raw."""
    out: dict[str, Any] = {}

    def add(v: Any) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, int | float):
            out.setdefault(norm_text(str(v)), v)
        elif isinstance(v, str) and 0 < len(v) <= 40 and len(v.split()) <= 3 and not STAMP.search(v):
            out.setdefault(norm_text(v), v)
        elif isinstance(v, list):
            for x in v:
                add(x)
        elif isinstance(v, dict):
            for x in v.values():
                add(x)

    for e in effects:
        add(e.get("new"))
    add(answer)
    return out


def decision_values(effects: Sequence[Mapping[str, Any]], answer: Any = None) -> frozenset[str]:
    return frozenset(decisions(effects, answer))


def reconcile(derive: Sequence[Any], effects: Sequence[Mapping[str, Any]], answer: Any, instruction: str) -> list[Any]:
    """The intent card's derive values after an independent oracle disagreed with the proposer: the proposer's values the
    oracle also reached, plus the oracle's decision values the proposer did not name and the instruction does not state.
    The executed reference is the authority (on stored finance_ap proposals the blind v2 oracle matched the pack's policy
    on 29 of 31, the proposer's derive on about half); the proposer's own values were leak-checked before the oracle ran
    (``pre_checks``), so TAINT's derive check loses nothing it had."""
    have = decisions(effects, answer)
    keep = [d for d in derive if not covered([d], frozenset(have))]
    named = {norm_text(str(d)) for d in keep}
    extra = [raw for k, raw in have.items() if k not in named and not text_spans(raw, instruction)]
    return keep + extra


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def covered(want: Iterable[Any], have: frozenset[str]) -> list[str]:
    """The values of ``want`` that are not in ``have`` (numbers numerically)."""
    miss = []
    for w in want:
        if is_structural(w):
            continue
        n = norm_text(str(w))
        if n not in have and not any(_num_eq(n, h) for h in have):
            miss.append(str(w))
    return miss


def agreement(derive: Sequence[Any], effects: Sequence[Mapping[str, Any]], answer: Any = None) -> str | None:
    """``disagree:<values>`` when a derive value of the proposer is not among the oracle's decision values (each derived
    independently, ``oracle_sees_derive=False``), else None. Derive values that are free text (more than three words)
    are not compared."""
    short = [d for d in derive if not (isinstance(d, str) and len(d.split()) > 3)]
    miss = covered(short, decision_values(effects, answer))
    return f"prefilter:disagree:proposer:{','.join(m[:30] for m in miss[:4])}" if miss else None


async def clarity_screen(client: RoleClient, *, instruction: str, tools: Sequence[Any], world: Any, effects: Sequence[Mapping[str, Any]],
                         answer: Any = None, tag: str = "clarity", world_chars: int = 24000) -> tuple[str | None, dict[str, Any]]:
    """The instruction-clarity self-check (module docstring): (drop reason or None, the record of the check)."""
    out: ClarityOut = await ask(client, CLARITY_CHECKER, tag=tag, retries=1, instruction=instruction, tools=tool_catalogue(tools) if tools and
                                not isinstance(tools[0], dict) else list(tools), world=compact(world, world_chars))
    have = decision_values(effects, answer)
    got = {norm_text(str(v)) for v in out.values if not is_structural(v)}
    oracle_only = [h for h in sorted(have) if h not in got and not any(_num_eq(h, g) for g in got)]
    rec = {"clear": out.clear, "issues": [i.model_dump(mode="json") for i in out.issues], "values": out.values, "oracle_values": sorted(have),
           "oracle_only": oracle_only, "analysis": out.analysis[:2000]}
    if not out.clear and out.issues:
        return f"prefilter:unclear:{out.issues[0].kind}", rec
    if oracle_only:
        return f"prefilter:disagree:clarity:{','.join(x[:30] for x in oracle_only[:4])}", rec
    return None, rec


# ---------------------------------------------------------------------------------------------------------- lemma batch

def _scrub(text: str) -> str:
    """Repair notes in words with ids, quoted values and numbers replaced (the abstractor sees no entity)."""
    t = ENTITY.sub("<id>", text or "")
    t = re.sub(r"'[^']{1,60}'|\"[^\"]{1,60}\"", "<value>", t)
    return re.sub(r"\b\d+(\.\d+)?\b", "<n>", t)[:400]


def repair_pairs(run_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """(counterexample, repair) pairs of runs: per accepted repair, the classes/operators/defects of the round's
    counterexamples and what the repair changed (instruction, oracle, contract, intent), with its scrubbed notes; plus
    the evidence the leakage critic searches (task ids, entity keys, literals)."""
    out: list[dict[str, Any]] = []
    for d in map(Path, run_dirs):
        arch: dict[str, dict[str, Any]] = {}
        dom = json.loads((d / "report.json").read_text()).get("domain") if (d / "report.json").exists() else None
        ap = d.parent / "archive.jsonl" if (d.parent / "archive.jsonl").exists() else None
        for p in [x for x in (ap, d / "archive" / str(dom) / "archive.jsonl") if x is not None and x.exists()]:
            for line in p.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line).get("rec") or {}
                    arch[rec.get("id", "")] = rec
        for cj in sorted((d / "tasks").glob("*/cegis.json")):
            rep = json.loads(cj.read_text())
            rows = [json.loads(x) for x in (cj.parent / "repairs.jsonl").read_text().splitlines() if x.strip()] if (cj.parent / "repairs.jsonl").exists() else []
            cands = [r for r in rows if r.get("result") == "candidate"]
            task = rep.get("task") or {}
            ents = sorted(entities(task.get("instruction", "")))
            lits = [v for v in (task.get("intent") or {}).get("derive") or [] if not is_structural(v)]
            for rnd in rep.get("rounds") or []:
                if rnd.get("repair") != "accepted":
                    continue
                cx = [arch[i] for i in rnd.get("counterexamples") or [] if i in arch]
                fix = next((r for r in cands if r.get("from_version") == rnd.get("task_version")), None)
                out.append({"task": rep.get("task_id"), "domain": rep.get("domain"), "cx": sorted({f"{c['cls']}/{c['op']}/{c.get('defect')}" for c in cx}),
                            "changed": sorted((fix or {}).get("changed") or []), "notes": _scrub((fix or {}).get("notes", "")),
                            "evidence": {"task_ids": [rep.get("task_id")], "entity_keys": ents, "literals": lits}})
    return out


def lemma_clusters(pairs: Sequence[Mapping[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    """Pairs grouped by (counterexample signature, what changed): the abstractor's ``clusters`` input."""
    g: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for p in pairs:
        g[("+".join(p["cx"]) or "?", "+".join(p["changed"]) or "?")].append(p)
    rows = sorted(g.items(), key=lambda kv: -len(kv[1]))[:limit]
    return [{"cluster": f"r{i}", "counterexamples": k[0], "repair_changed": k[1], "size": len(v), "tasks": len({x["task"] for x in v}),
             "repair_notes": [x["notes"] for x in v[:3] if x["notes"]]} for i, (k, v) in enumerate(rows)]


def lemma_leaks(texts: Sequence[str], pairs: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """The deterministic leakage critic (``meta.update.leakage_critic``) on each lemma text, against the pairs' evidence."""
    from crucible.synth.meta.update import Evidence, Update, leakage_critic

    ev = Evidence(task_ids=sorted({t for p in pairs for t in p["evidence"]["task_ids"] if t}),
                  entity_keys=sorted({e for p in pairs for e in p["evidence"]["entity_keys"]}),
                  literals=sorted({cjson(v) for p in pairs for v in p["evidence"]["literals"]}))
    ev.literals = [json.loads(x) for x in ev.literals]
    return {t: leakage_critic(Update(id="lemma", component="S", diff=t, evidence=ev)) for t in texts}


def cell_histogram(kinds: Iterable[str]) -> dict[str, int]:
    return dict(Counter(kinds).most_common())


__all__ = ["Deduper", "agreement", "cell_histogram", "clarity_screen", "covered", "decision_values", "entities", "jaccard", "lemma_clusters",
           "lemma_leaks", "pre_checks", "repair_pairs", "shingles"]
