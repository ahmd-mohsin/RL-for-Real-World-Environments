"""The lemma lift of the per-task loop (DESIGN §7.7 item 25; R-lemma-lift; runs/taskloop-2.md).

Step 5 of the probe loop (runs/rsi-stats.md §9.5'): "the repair becomes a candidate design lemma for the writers". Until
taskloop-2 the hook existed (``ProbeLoop(lift_lemmas=)``) but nothing filled it, so the 3 accepted repairs of
taskloop-1 never reached the lemma bank. ``LemmaLift`` is that hook:

1. **Record.** Every accepted repair is a ``RepairPair``: the classes of the counterexamples it closed (the defect of
   each, D1-D11; ``cls/op`` when no defect is named), what it changed (instruction, oracle, contract, intent, World) and
   the repairer's notes scrubbed of ids, quoted values and numbers (``yieldkit._scrub``), plus the evidence the leakage
   critic searches (task ids, entity keys, derived literals). Pairs are appended to ``lemma_pairs.jsonl``.
2. **Recur.** A counterexample class (``R-lemma-lift`` ``key``: the defect) that ``min_tasks`` distinct lineages
   repaired becomes a ``LemmaCandidate``: its pairs grouped as one cluster. A class seen on one lineage is no candidate
   (one task's fix is that task's, not a design rule).
3. **Draft.** The abstractor (``roles.ABSTRACTOR``, a non-builder family; it sees clusters and the defect taxonomy,
   never a task id) writes the rule; the deterministic leakage critic (``yieldkit.lemma_leaks``) refuses a rule that
   names a task, entity or value of the evidence. No abstractor wired: the candidate stays unwritten.
4. **Gate.** The candidate raises ``lemma_candidate`` with its measured facts (``units_needed``: the paired cells the
   exact McNemar test needs for ``target_gain`` at ``alpha_t`` and ``power`` (``stats.plants_needed``);
   ``units_available``: the cells the gate budget buys for both arms at ``usd_per_cell``) and takes the policy's remedy.
   ``rerun_control_equal_spend`` runs the gate the caller wired (``meta.lemmas.lemma_acceptance`` on an equal-budget
   paired build, J under R1-R10); only an accepted decision adds the lemma to the bank. Otherwise ``no_update``: the
   candidate is **deferred** (underpowered, leaking, unwritten or unfunded) and the bank does not change.

Nothing here edits a task or reads the student. The bank (``lemma_bank.json``) is the S component's ``LemmaBank``.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson, content_hash
from crucible.synth.falsify.base import Counterexample
from crucible.synth.meta.lemmas import Lemma, LemmaBank, LemmaDecision, lemma_id
from crucible.synth.task import Task

PARTS = ("instruction", "oracle", "contract", "intent", "world")


class RepairPair(CrucibleModel):
    id: str
    task_id: str
    lineage: str
    domain: str
    from_version: int
    to_version: int
    defects: list[str] = Field(default_factory=list)  # the counterexample classes the repair closed (the recurrence key)
    cx: list[str] = Field(default_factory=list)  # cls/op/defect signatures
    changed: list[str] = Field(default_factory=list)
    notes: str = ""  # scrubbed: no id, quoted value or number
    round: int = 0
    evidence: dict[str, list[Any]] = Field(default_factory=dict)  # task_ids, entity_keys, literals (the leakage critic's)


class LemmaCandidate(CrucibleModel):
    id: str
    key: str
    pairs: list[str] = Field(default_factory=list)
    lineages: list[str] = Field(default_factory=list)
    lemmas: list[dict[str, Any]] = Field(default_factory=list)  # the abstractor's rules {applies_to, rule, evidence}
    leaks: dict[str, list[str]] = Field(default_factory=dict)
    status: Literal["unwritten", "leaking", "deferred", "accepted", "refused"] = "unwritten"
    gate: dict[str, Any] = Field(default_factory=dict)
    remedy: dict[str, Any] = Field(default_factory=dict)
    policy_version: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)


Abstract = Callable[[list[dict[str, Any]]], Awaitable[Sequence[Any]]]  # clusters -> roles.Lemma-like {applies_to, rule, evidence}
Gate = Callable[[LemmaCandidate, int], Awaitable[LemmaDecision]]  # (candidate, paired cells) -> lemma_acceptance's decision


def changed_parts(old: Task, new: Task) -> list[str]:
    """What a repair changed, by part of the task."""
    out = []
    for part in PARTS:
        a, b = getattr(old, part), getattr(new, part)
        a = a.model_dump(mode="json") if hasattr(a, "model_dump") else a
        b = b.model_dump(mode="json") if hasattr(b, "model_dump") else b
        if cjson(a) != cjson(b):
            out.append(part)
    return out


def _key(c: Counterexample) -> str:
    return c.defect or f"{c.cls}/{c.op}"


def _role(applies_to: str) -> str:
    return {"contract_writer": "writer", "oracle_programmer": "proposer"}.get(applies_to, applies_to)


class LemmaLift:
    """The ``LemmaLift`` hook of ``ProbeLoop``/``TaskLoop`` (module docstring). ``abstract`` drafts rules from clusters;
    ``gate`` runs the equal-budget lemma test; ``gate_budget_usd()`` is what the gate may spend (0: deferred);
    ``notes_of(task_id, from_version)`` the repairer's notes; ``lineage_of(task_id)`` the lineage root."""

    def __init__(self, root: Path, *, abstract: Abstract | None = None, gate: Gate | None = None, gate_budget_usd: Callable[[], float] = lambda: 0.0,
                 notes_of: Callable[[str, int], str] | None = None, lineage_of: Callable[[str], str] | None = None, policy: Any = None):
        self.root = Path(root)
        self.abstract, self.gate, self.gate_budget_usd = abstract, gate, gate_budget_usd
        self.notes_of, self.lineage_of, self.policy = notes_of, lineage_of, policy
        self.pairs: dict[str, RepairPair] = {}
        self.candidates: dict[str, LemmaCandidate] = {}
        self.bank = LemmaBank()
        self._lock = asyncio.Lock()
        self._load()

    # -- state ----------------------------------------------------------------------------------------------------------
    def _load(self) -> None:
        p = self.root / "lemma_pairs.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    x = RepairPair.model_validate_json(line)
                    self.pairs[x.id] = x
        c = self.root / "lemma_candidates.json"
        if c.exists():
            self.candidates = {k: LemmaCandidate.model_validate(v) for k, v in json.loads(c.read_text()).items()}
        b = self.root / "lemma_bank.json"
        if b.exists():
            self.bank = LemmaBank.model_validate_json(b.read_text())

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name, doc in (("lemma_candidates.json", {k: v.model_dump(mode="json") for k, v in self.candidates.items()}),
                          ("lemma_bank.json", self.bank.model_dump(mode="json"))):
            tmp = self.root / (name + ".tmp")
            tmp.write_text(json.dumps(doc, indent=1, default=str))
            tmp.replace(self.root / name)

    def _pol(self) -> Any:
        if self.policy is not None:
            return self.policy
        from crucible.synth import policy

        return policy.current()

    def param(self, name: str) -> Any:
        return self._pol().param("R-lemma-lift", name)

    # -- 1. record ------------------------------------------------------------------------------------------------------
    def pair(self, task: Task, new: Task, cxs: Sequence[Counterexample], *, notes: str = "", round: int = 0, changed: Sequence[str] | None = None) -> RepairPair:
        from crucible.synth.program import is_structural
        from crucible.synth.yieldkit import _scrub, entities

        lin = self.lineage_of(task.id) if self.lineage_of is not None else task.id
        if not notes and self.notes_of is not None:
            notes = self.notes_of(task.id, task.version) or ""
        lits = [v for v in [*task.intent.derive, *new.intent.derive] if not is_structural(v)]
        return RepairPair(id="pair-" + content_hash({"task": task.id, "from": task.version, "to": new.version}, 16), task_id=task.id, lineage=lin,
                          domain=task.domain, from_version=task.version, to_version=new.version, defects=sorted({_key(c) for c in cxs}),
                          cx=sorted({f"{c.cls}/{c.op}/{c.defect}" for c in cxs}), changed=list(changed if changed is not None else changed_parts(task, new)),
                          notes=_scrub(notes), round=round,
                          evidence={"task_ids": [task.id], "entity_keys": sorted(entities(task.instruction) | entities(new.instruction)),
                                    "literals": [json.loads(x) for x in sorted({cjson(v) for v in lits})]})

    async def __call__(self, task: Task, new: Task, cxs: list[Counterexample]) -> list[LemmaCandidate]:
        """The ``LemmaLift`` hook: record the accepted repair, then reconsider every class it closed that now recurs."""
        return await self.add(self.pair(task, new, cxs))

    async def add(self, pair: RepairPair) -> list[LemmaCandidate]:
        async with self._lock:  # concurrent loop steps: one candidate update at a time
            if pair.id not in self.pairs:
                self.pairs[pair.id] = pair
                self.root.mkdir(parents=True, exist_ok=True)
                with (self.root / "lemma_pairs.jsonl").open("a") as fh:
                    fh.write(pair.model_dump_json() + "\n")
            out = []
            for key in pair.defects:
                c = await self.consider(key)
                if c is not None:
                    out.append(c)
            self._save()
            return out

    # -- 2-4. recur, draft, gate ----------------------------------------------------------------------------------------
    def recurring(self) -> dict[str, list[RepairPair]]:
        """Counterexample class -> its pairs, for the classes ``min_tasks`` distinct lineages repaired."""
        by: dict[str, list[RepairPair]] = {}
        for p in self.pairs.values():
            for k in p.defects:
                by.setdefault(k, []).append(p)
        need = int(self.param("min_tasks"))
        return {k: v for k, v in by.items() if len({p.lineage for p in v}) >= need}

    def units(self) -> tuple[int, int]:
        """(units the gate needs, units the gate budget buys), paired cells."""
        from crucible.synth.meta.stats import plants_needed

        need = plants_needed(float(self.param("target_gain")), alpha=float(self.param("alpha_t")), power=float(self.param("power")))
        per = 2 * float(self.param("usd_per_cell"))  # both arms build the cell: equal spend
        have = math.floor(max(0.0, float(self.gate_budget_usd())) / per) if per > 0 else 0
        return (need if need is not None else 10**9), have

    async def consider(self, key: str) -> LemmaCandidate | None:
        rec = self.recurring()
        if key not in rec:
            return None
        pairs = sorted(rec[key], key=lambda p: p.id)
        cid = "cand-" + content_hash(key, 10)
        cand = self.candidates.get(cid) or LemmaCandidate(id=cid, key=key)
        if cand.status in ("accepted", "refused"):  # decided by the gate: a candidate is tested once (alpha is spent once)
            return cand
        grown = [p.id for p in pairs] != cand.pairs
        cand.pairs, cand.lineages = [p.id for p in pairs], sorted({p.lineage for p in pairs})
        if self.abstract is not None and (grown or not cand.lemmas):
            cluster = {"cluster": "r0", "counterexamples": key, "signatures": sorted({s for p in pairs for s in p.cx}),
                       "repair_changed": sorted({c for p in pairs for c in p.changed}), "size": len(pairs), "tasks": len(cand.lineages),
                       "repair_notes": [p.notes for p in pairs if p.notes][:4]}
            try:
                got = await self.abstract([cluster])
                cand.lemmas = [x.model_dump(mode="json") if hasattr(x, "model_dump") else dict(x) for x in got]
            except Exception as e:  # noqa: BLE001 - an abstractor that gives no lawful answer leaves the candidate unwritten
                cand.history.append({"draft_error": f"{type(e).__name__}: {str(e)[:200]}"})
                cand.lemmas = []
        texts = [str(x.get("rule", "")) for x in cand.lemmas if x.get("rule")]
        if texts:
            from crucible.synth.yieldkit import lemma_leaks

            cand.leaks = {t: v for t, v in lemma_leaks(texts, [p.model_dump(mode="json") for p in pairs]).items() if v}
        clean = [t for t in texts if not cand.leaks.get(t)]
        need, have = self.units()
        facts = {"recurs_tasks": len(cand.lineages), "leak_free": bool(clean), "units_needed": need, "units_available": have,
                 "budget_ok": self.gate is not None and float(self.gate_budget_usd()) > 0}
        from crucible.synth.policy import Condition, resolve

        rem = resolve(Condition(kind="lemma_candidate", subject=cid, facts=facts, raised_by="meta.lift.LemmaLift"), self._pol())
        cand.policy_version = rem.policy_version
        cand.remedy = {"rule": rem.rule, "action": rem.action, "facts": facts, "considered": rem.considered[:6]}
        cand.gate = {"units_needed": need, "units_available": have, "alpha_t": self.param("alpha_t"), "target_gain": self.param("target_gain"),
                     "power": self.param("power")}
        if rem.action == "rerun_control_equal_spend" and self.gate is not None:
            dec = await self.gate(cand, need)
            cand.gate["decision"] = dec.model_dump(mode="json")
            if dec.accepted:
                lem = [Lemma(id=lemma_id(t), text=t, roles=[_role(str(x.get("applies_to", "proposer")))],  # type: ignore[list-item]
                             sources=cand.pairs, defects=[key] if key.startswith("D") else []) for x in cand.lemmas
                       if (t := str(x.get("rule", ""))) in clean]
                have_ids = {x.id for x in self.bank.lemmas}
                self.bank = self.bank.add(*[x for x in lem if x.id not in have_ids])
                cand.status = "accepted"
            else:
                cand.status = "refused"
        else:
            cand.status = "unwritten" if not texts else "leaking" if not clean else "deferred"
        cand.history.append({"pairs": len(cand.pairs), "status": cand.status, "action": rem.action, "units": [need, have]})
        self.candidates[cid] = cand
        return cand

    async def reconsider(self) -> list[LemmaCandidate]:
        async with self._lock:
            out = [c for k in sorted(self.recurring()) if (c := await self.consider(k)) is not None]
            self._save()
            return out

    async def seed(self, pairs: Iterable[RepairPair]) -> list[LemmaCandidate]:
        """Pairs of earlier runs (``pairs_from_run``): recorded, then every recurring class reconsidered."""
        async with self._lock:
            for p in pairs:
                if p.id not in self.pairs:
                    self.pairs[p.id] = p
                    self.root.mkdir(parents=True, exist_ok=True)
                    with (self.root / "lemma_pairs.jsonl").open("a") as fh:
                        fh.write(p.model_dump_json() + "\n")
        return await self.reconsider()

    def summary(self) -> dict[str, Any]:
        return {"pairs": len(self.pairs), "candidates": {k: {"key": c.key, "lineages": len(c.lineages), "status": c.status, "units": c.gate}
                                                          for k, c in self.candidates.items()},
                "bank": {"version": self.bank.version, "lemmas": len(self.bank.lemmas)}}


def swarm_abstract(client: Any, *, tag: str = "lift") -> Abstract:
    """``roles.ABSTRACTOR`` on one client (the abstractor's non-builder family): clusters and the defect taxonomy only."""
    from crucible.synth.roles import ABSTRACTOR, DEFECTS_DOC, ask

    async def run(clusters: list[dict[str, Any]]) -> Sequence[Any]:
        out = await ask(client, ABSTRACTOR, tag=f"{tag}:{content_hash(clusters, 10)}", retries=1, clusters=clusters, defects=DEFECTS_DOC)
        return list(out.lemmas)

    return run


def pairs_from_run(run: Path, lift: LemmaLift) -> list[RepairPair]:
    """The accepted repairs of an earlier TaskLoop run as pairs (offline lift; no model call, no student): the REPAIR
    decisions of ``loop/task_state.json``, the repairer's record of the same (task, from_version) in ``repairs.jsonl``
    (what changed, its notes) and the counterexamples of that version in the run's archive."""
    from crucible.synth.archive import Archive
    from crucible.synth.meta.rsi import TaskDecision

    st = json.loads((run / "loop" / "task_state.json").read_text())
    of = dict(st.get("of", {}))
    decs = [TaskDecision.model_validate(x) for x in st.get("decisions", [])]
    reps = [json.loads(x) for x in (run / "repairs.jsonl").read_text().splitlines() if x.strip()] if (run / "repairs.jsonl").exists() else []
    arch = Archive(run / "archive")
    out = []
    for d in decs:
        if d.move != "REPAIR" or not d.accepted or d.new_version is None:
            continue
        r = next((x for x in reps if x.get("task_id") == d.task_id and x.get("from_version") == d.version and x.get("result") == "candidate"), None)
        doms = sorted(p.name for p in (run / "archive").iterdir() if p.is_dir()) if (run / "archive").exists() else []
        cxs: list[Counterexample] = []
        for dm in doms:
            cxs += [c for c in arch.records(dm, d.task_id) if c.task_version == d.version]
        if not cxs:
            continue
        p = RepairPair(id="pair-" + content_hash({"task": d.task_id, "from": d.version, "to": d.new_version}, 16), task_id=d.task_id,
                       lineage=of.get(d.task_id, d.task_id), domain=cxs[0].domain, from_version=d.version, to_version=d.new_version,
                       defects=sorted({_key(c) for c in cxs}), cx=sorted({f"{c.cls}/{c.op}/{c.defect}" for c in cxs}),
                       changed=list((r or {}).get("changed") or []), round=d.round, evidence={"task_ids": [d.task_id], "entity_keys": [], "literals": []})
        if r and r.get("notes"):
            from crucible.synth.yieldkit import _scrub, entities

            p.notes = _scrub(str(r["notes"]))
            p.evidence["entity_keys"] = sorted(entities(str(r["notes"])))
        out.append(p)
    return out


def repairer_notes(path: Path) -> Callable[[str, int], str]:
    """``notes_of`` over a run's ``repairs.jsonl`` (the pipeline repairer's log)."""

    def get(task_id: str, version: int) -> str:
        if not path.exists():
            return ""
        rows: Iterable[dict[str, Any]] = (json.loads(x) for x in path.read_text().splitlines() if x.strip())
        hit = [r for r in rows if r.get("task_id") == task_id and r.get("from_version") == version and r.get("result") == "candidate"]
        return str(hit[-1].get("notes", "")) if hit else ""

    return get


__all__ = ["LemmaCandidate", "LemmaLift", "RepairPair", "changed_parts", "pairs_from_run", "repairer_notes", "swarm_abstract"]
