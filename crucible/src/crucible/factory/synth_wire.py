"""Binding of ``crucible.synth``'s roles to the swarm on Tinker (part2-paper-core §12, M4).

``SynthWire`` is the only place synth meets a model:

- **Assignment.** ``SYNTH_ROLES`` maps each ``synth.roles`` role to one or more models of the ``swarm`` preset
  (``factory.models``: Kimi-K2.6, GLM-5.3, DeepSeek-V4.1-Flash, Nemotron-3 Super/Ultra, gpt-oss-120b). The table is
  loaded into ``ModelBook(preset="swarm")`` (its own family rules run there), then checked against the synth roles' own
  constraints (``roles.family_problems``: differs-from, independence from the student's family, minimum families);
  any violation raises ``SameFamilyError`` before a client exists. The student (Qwen3.6-35B-A3B) is never called, and no
  Qwen model serves any role: every judging, attacking or solving-as-reference role is ``independent``.
- **One hard spend cap per run.** One ``SpendMeter(cap_usd, path=<run>/spend.jsonl)`` serves every client; each call
  holds its worst case first and is refused (``TinkerCapReached``) when spent + held + this call would pass the cap, so
  the cap is checked before any call. ``milestone_spent`` sums every ``spend.jsonl`` under the data root, so a CLI can
  give a run only what the milestone budget has left.
- **Attribution.** ``WireClient`` implements ``roles.RoleClient`` over a Tinker chat client: JSON asked for in the
  prompt, parsed and schema-checked with retries, and every call's measured cost booked under its ``tag``
  (``<stage>:<task>``) in ``usage.jsonl``; ``cost()`` sums it by stage, role, family and model.
- **Domains.** ``finance_ap_env`` (the existing pack through ``CrucibleWorldAdapter``, one World per start-state seed; the
  slice shows the ERP tables and the policy document) and ``generated_env`` (a ``worldgen`` App in the sandbox).

The Tinker key is read by the SDK from the process environment (the caller sources the repo's ``.env`` into the
subprocess); it is never passed, stored or printed here.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.models import (
    DEEPSEEK,
    GLM,
    GPT_OSS,
    INKLING,
    INKLING_SMALL,
    KIMI,
    NEMOTRON,
    NEMOTRON_ULTRA,
    ModelBook,
    SameFamilyError,
    backend,
    parse_spec,
)
from crucible.synth.batch import budget_cause
from crucible.synth.iface import CrucibleWorldAdapter
from crucible.synth.pipeline import DomainEnv, RunRoles
from crucible.synth.policy import stage_cap as policy_stage_cap
from crucible.synth.roles import family_problems
from crucible.synth.task import WorldRef
from crucible.synth.worldroles import ALL_ROLES

MILESTONE_CAP_USD = 60.0  # M4's Tinker budget (lead, 2026-10-29): enforced by the CLI through ``milestone_spent``

# synth role -> models (the swarm preset's families; see the module docstring for the rules they satisfy)
SYNTH_ROLES: dict[str, tuple[str, ...]] = {
    # M4: Kimi-K2.6 deliberated in plain text past 12k-16k tokens on the generated domain (thinking on or off) and never
    # reached its JSON; DeepSeek proposes concisely. The proposer must differ only from the contract writer (GLM).
    "proposer": (DEEPSEEK,),
    "oracle_programmer": (DEEPSEEK,),
    "contract_writer": (GLM,),
    "repairer": (DEEPSEEK,),
    # M4 run 1: gpt-oss-120b quits after one call (it reports success it did not make) and Nemotron-3-Super misapplies the
    # AP policy (ALT passed on 5 of 17 finance_ap tasks); Nemotron-3-Ultra solves best and leads the solvers.
    # Yield work: the table said Ultra + Super + gpt-oss, but both Nemotron solvers were named "alt_solver-nemotron" and ALT
    # ran Ultra twice (Super never ran as a solver in M4 or RSI-1). With unique names that is now explicit: on 12 finance_ap
    # tasks whose oracle is right, Ultra passed 4, Super 0 (2 with the v2 harness), gpt-oss 0, so the second attempt stays
    # Ultra's. gpt-oss stays because ALT needs two families.
    "alt_solver": (NEMOTRON_ULTRA, NEMOTRON_ULTRA, GPT_OSS),
    "attacker": (NEMOTRON, GPT_OSS),
    "panel": (NEMOTRON_ULTRA, GPT_OSS),
    "abstractor": (GPT_OSS,),
    "tool_designer": (KIMI,),
    "schema_deriver": (KIMI,),
    "app_coder": (KIMI,),
    "blind_tester": (DEEPSEEK,),
    "debugger": (KIMI,),
    "test_approver": (NEMOTRON_ULTRA,),
    "seeder": (KIMI,),
    # L2 leak evidence (DESIGN §7.7 item 20): the probe on the sixth family (Inkling-Small: nothing builds, solves or
    # judges on it), the leak panel on the calibrated panel's two families
    "leak_probe": (INKLING_SMALL,),
    "leak_panel": (NEMOTRON_ULTRA, GPT_OSS),
    # §10 composition (synth/compose.py): one instruction for a composed task, and knob rewrites; a builder family (the
    # proposer's, which writes instructions), never a solver's, attacker's or judge's
    "instruction_writer": (DEEPSEEK,),
    # yield work: the instruction-clarity self-check, an independent derivation in a builder family other than the
    # proposer's and the oracle programmer's (GLM already builds: it adds no family to the builder set ALT must avoid)
    "clarity_checker": (GLM,),
}

# The six-family assignment (§14.1 "family exhaustion", 2026-10-29): with Inkling, the panel shares no family with the
# ALT solvers or attackers whose traces it judges (``roles.checker_overlap`` is empty). Kimi builds Worlds only (F1), not
# tasks, so it may judge tasks. Not the default: the default panel is the calibrated one; switching needs a calibration.
SYNTH_ROLES_DISJOINT: dict[str, tuple[str, ...]] = {
    **SYNTH_ROLES,
    "alt_solver": (NEMOTRON_ULTRA, NEMOTRON, GPT_OSS),
    "attacker": (NEMOTRON, GPT_OSS),
    "panel": (INKLING, KIMI),
    "leak_panel": (INKLING, NEMOTRON_ULTRA),
    "leak_probe": (GPT_OSS,),
}

# the stage a role's calls are booked under (per-stage hard caps: ``SynthWire(stage_caps=...)``)
ROLE_STAGE: dict[str, str] = {"proposer": "propose", "oracle_programmer": "oracle", "contract_writer": "contract", "repairer": "repair",
                              "alt_solver": "ALT", "attacker": "ATK", "panel": "adjudicate", "abstractor": "abstract", "clarity_checker": "clarity",
                              "tool_designer": "worldgen", "schema_deriver": "worldgen", "app_coder": "worldgen", "blind_tester": "worldgen",
                              "debugger": "worldgen", "test_approver": "worldgen", "seeder": "worldgen"}


# World families (synth.families, synth.catalogue, synth.worldgates): builders over three families (Kimi codes as in F1,
# GLM writes specs and documents, DeepSeek lists domains, tests blind and writes colleagues); the realism raters are two
# other families (never a builder's). Merged over SYNTH_ROLES by the ``worlds`` CLI.
WORLD_ROLE_TABLE: dict[str, tuple[str, ...]] = {
    "domain_cataloguer": (DEEPSEEK,),
    "domain_spec_writer": (GLM,),
    "schema_deriver": (GLM,),  # pilot 1: Kimi spent the 8192-token schema budget reasoning (finish = length)
    "family_designer": (KIMI,),
    "data_spec_writer": (KIMI,),
    "doc_writer": (GLM,),
    "actor_writer": (DEEPSEEK,),
    "realism_rater": (NEMOTRON_ULTRA, GPT_OSS),
}
WORLDS_CAP_USD = policy_stage_cap("worlds")  # World families and instances (lead, 2026-10-29; R-budget in the policy ledger): every receipt under a path carrying "worlds"


def _book_roles(table: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {f"synth_{r}_{i}": m for r, ms in table.items() for i, m in enumerate(ms)}


def check_assignment(table: Mapping[str, Sequence[str]], *, students: Sequence[str] | None = None) -> ModelBook:
    """The ModelBook of the assignment, after both rule sets; ``SameFamilyError`` on any violation."""
    book = ModelBook(roles=_book_roles(table), preset="swarm")
    studs = list(students) if students is not None else [s.family for s in book.solvers]
    fams = {r: [parse_spec(m).family for m in ms] for r, ms in table.items()}
    probs = family_problems(fams, students=studs, roles=ALL_ROLES)
    qwen = sorted(r for r, fs in fams.items() if "qwen" in fs)
    if qwen:
        probs.append(f"no Qwen model may serve a synth role (the student is Qwen): {qwen}")
    if probs:
        raise SameFamilyError("; ".join(probs))
    return book


# ---------------------------------------------------------------------------------------------------------- the client

def extract_json(text: str) -> dict[str, Any]:
    """The reply's JSON object. Some models (Kimi-K2.6) reason in plain text before the object, and that text holds
    braces of its own, so the first-to-last-brace cut fails: take the last complete top-level object that decodes."""
    from crucible.train.bridges.tinker.sampler import parse_json_reply

    try:
        return parse_json_reply(text)
    except ValueError:
        pass
    t = text or ""
    dec = json.JSONDecoder()
    best: dict[str, Any] | None = None
    i = t.find("{")
    while i >= 0:
        try:
            obj, end = dec.raw_decode(t, i)
        except ValueError:
            i = t.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            best = obj
            i = t.find("{", end)
        else:
            i = t.find("{", i + 1)
    if best is None:
        raise ValueError(f"not a JSON object: {t[:200]}")
    return best


@dataclass
class Usage:
    path: Path | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    def by(self, key: str) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for r in self.rows:
            k = r.get(key) or "?"
            out[k] += float(r.get("usd") or 0.0)
        return {k: round(v, 5) for k, v in sorted(out.items())}


# The chat engine wraps every non-LLM error of a call in ``EngineCallError`` (``train.world_env``), the meter's
# ``TinkerCapReached`` included, so before yield-2 a reached cap reached the synth pipeline as a model failure: panel
# members abstained, ALT attempts failed, and the run went on refusing calls instead of stopping. ``WireClient`` re-raises
# the budget refusal behind such an error (``synth.batch.budget_cause``).


def tag_seed(tag: str, attempt: int = 0, salt: str = "") -> int:
    """The sampling seed of one call (DESIGN §7.7 item 22): a pure function of (salt, tag, JSON retry), so the same tag
    gives the same seed and two tags (attempts, rollouts, panel members) give independent samples. An empty salt is the
    item-19 formula (``crc32(tag)``), which the student store's ``seed=tag`` records were drawn with."""
    import zlib

    key = tag if not salt else f"{salt}|{tag}"
    return (zlib.crc32(key.encode()) + 1009 * attempt) % 2_147_483_647


class WireClient:
    """``roles.RoleClient`` over an ``LLMClient``-shaped chat client (``complete(messages, ...) -> LLMResponse``).

    **Seeds (DESIGN §7.7 item 22).** Every call is seeded from its tag by default (``tag_seed``). Before, the default
    seed was the JSON retry index (0): Tinker samples a prompt identically at one seed, so k student rollouts were one
    trajectory copied k times (P2 pilot), the two Ultra ALT attempts were one trajectory in 120 of 216 pairs (yield-1/2),
    and every M4, RSI-1 and yield-1/2 certificate's ALT, ATK and panel calls ran on seed 0. ``seed_by_tag=False`` keeps
    the old behaviour for reproducing those runs (``--same-seed``). ``salt`` separates the seed streams of two clients
    (two Ultra solvers on one tag) and of two runs (a re-certification must not redraw the stored run's samples)."""

    def __init__(self, inner: Any, *, role: str, model: str, family: str, usage: Usage, json_retries: int = 2, seed_by_tag: bool = True,
                 salt: str = ""):
        self.inner, self.role, self.model, self.family, self.usage, self.json_retries = inner, role, model, family, usage, json_retries
        self.calls = 0
        self.seed_by_tag = seed_by_tag
        self.salt = salt

    def _seed(self, tag: str, attempt: int) -> int:
        if not self.seed_by_tag:
            return attempt
        return tag_seed(tag, attempt, self.salt)

    async def complete_json(self, messages: list[dict[str, Any]], *, schema: Any = None, max_tokens: int = 2048, temperature: float = 0.0,
                            tag: str = "") -> dict[str, Any]:
        from crucible.actors.llm import LLMOutputError
        from crucible.train.bridges.tinker.sampler import check_schema

        msgs = list(messages)
        last = ""
        for attempt in range(self.json_retries + 1):
            t0 = time.monotonic()
            try:
                r = await self.inner.complete(msgs, temperature=temperature, max_tokens=max_tokens, seed=self._seed(tag, attempt))
            except Exception as e:
                cap = budget_cause(e)
                if cap is not None:  # the meter refused the call: a budget stop, not a model or provider failure
                    raise cap from e
                raise
            self.calls += 1
            stage = (tag or self.role).split(":", 1)[0]
            self.usage.add({"t": round(time.time(), 3), "tag": tag, "stage": stage, "role": self.role, "model": self.model, "family": self.family,
                            "usd": round(float(getattr(r, "cost_usd", 0.0) or 0.0), 7), "latency_s": round(time.monotonic() - t0, 2),
                            "finish": getattr(r, "raw_finish", None), "attempt": attempt})
            try:
                doc = extract_json(r.text)
                check_schema(doc, schema)
                return doc
            except ValueError as e:
                last = str(e)
                if getattr(r, "raw_finish", None) == "length":  # cut off at the token limit: ask for a shorter reply, not a repair
                    msgs = [*messages, {"role": "user", "content": f"Your previous reply was cut off at the {max_tokens}-token limit. Reply again "
                                                                    "with the whole JSON object, more concisely (shorter descriptions, no comments)."}]
                    continue
                msgs = [*messages, {"role": "assistant", "content": (r.text or "")[-6000:]},
                        {"role": "user", "content": f"That reply was not usable: {last[:400]}. Reply again with one JSON object only."}]
        raise LLMOutputError(f"{self.model}: no valid JSON after {self.json_retries + 1} attempts: {last[:300]}")


# ---------------------------------------------------------------------------------------------------------- the wire

def stage_meter(parent: Any, stage: str, cap_usd: float) -> Any:
    """A per-stage hard cap on top of the run's ``SpendMeter`` (yield work). It is a ``SpendMeter``: a call holds its worst
    case in the stage meter AND the run meter (refused with ``TinkerCapReached`` when either cannot pay it) and books its
    measured cost in both; the run's receipt rows carry ``stage``, so a resumed run restarts the stage meter from them."""
    from contextlib import contextmanager

    from crucible.train.bridges.tinker.spend import SpendMeter

    class StageMeter(SpendMeter):
        def __init__(self) -> None:
            super().__init__(cap_usd)
            self.parent, self.stage = parent, stage
            if parent.path is not None and parent.path.exists():
                for line in parent.path.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if row.get("stage") == stage:
                            self._add(row)

        @contextmanager
        def hold(self, est_usd: float, what: str):  # type: ignore[override]
            with self.parent.hold(est_usd, what), super().hold(est_usd, f"stage {stage}: {what}"):
                yield

        def book(self, kind: str, model: Any, **kw: Any) -> float:  # type: ignore[override]
            usd = self.parent.book(kind, model, stage=stage, **kw)
            with self._lock:
                self._add({"kind": kind, "model": getattr(model, "name", None), "usd": usd})
            return usd

    return StageMeter()


class SynthWire:
    """Clients for every synth role on one run's meter (module docstring). ``stage_caps`` (stage -> USD; stages are
    ``ROLE_STAGE``'s) adds a hard cap per stage on top of the run's cap.

    Seeds by tag are the default (``WireClient``; DESIGN §7.7 item 22). Client ``(role, i)`` salts its seeds with
    ``<seed_salt>|<role>#<i>``; ``seed_salt`` defaults to the run directory's name, so a resumed run redraws its own
    seeds and another run (a re-certification of stored tasks) draws new ones."""

    def __init__(self, run_dir: str | Path, *, cap_usd: float, table: Mapping[str, Sequence[str]] | None = None, make: Any = None,
                 stage_caps: Mapping[str, float] | None = None, seed_by_tag: bool = True, seed_salt: str | None = None):
        from crucible.train.bridges.tinker.spend import SpendMeter

        # distinct samples per call tag (the yield arms: yield-1/2 ran the two Ultra ALT attempts on one seed, and 120 of
        # 216 attempt pairs were the same trajectory; every SynthWire before item 22 defaulted to seed 0)
        self.seed_by_tag = seed_by_tag

        self.dir = Path(run_dir)
        self.seed_salt = self.dir.name if seed_salt is None else seed_salt
        self.table = dict(table or SYNTH_ROLES)
        self.book = check_assignment(self.table)
        self.meter = SpendMeter(cap_usd, path=self.dir / "spend.jsonl")
        self.stage_meters = {s: stage_meter(self.meter, s, float(c)) for s, c in (stage_caps or {}).items()}
        self.usage = Usage(self.dir / "usage.jsonl")
        self._make = make  # tests inject a fake maker(model, role) -> chat client
        self._clients: dict[tuple[str, int], WireClient] = {}

    def meter_for(self, role: str) -> Any:
        return self.stage_meters.get(ROLE_STAGE.get(role, ""), self.meter)

    def client(self, role: str, i: int = 0) -> WireClient:
        if role not in ALL_ROLES:
            raise KeyError(f"unknown synth role {role!r}")
        k = (role, i)
        if k not in self._clients:
            spec = self.book.spec(f"synth_{role}_{i}")
            if self._make is not None:
                inner = self._make(spec.model, role)
            else:
                inner = backend(spec.provider).make(spec.model, role=role, meter=self.meter_for(role))
            self._clients[k] = WireClient(inner, role=role, model=spec.model, family=spec.family, usage=self.usage, seed_by_tag=self.seed_by_tag,
                                          salt=f"{self.seed_salt}|{role}#{i}")
        return self._clients[k]

    def clients(self, role: str) -> list[WireClient]:
        return [self.client(role, i) for i in range(len(self.table[role]))]

    def run_roles(self) -> RunRoles:
        def named(role: str) -> list[tuple[str, WireClient]]:
            # unique names: two models of one family (Nemotron Ultra and Super) once shared "alt_solver-nemotron", and the
            # ALT falsifier, which finds its runner by name, ran the first of them twice (yield work, M4 records)
            cs = self.clients(role)
            fams = Counter(c.family for c in cs)
            return [(f"{role}-{c.family}" + (f"-{i}" if fams[c.family] > 1 else ""), c) for i, c in enumerate(cs)]

        return RunRoles(proposer=self.client("proposer"), oracle_programmer=self.client("oracle_programmer"), contract_writer=self.client("contract_writer"),
                        repairer=self.client("repairer"), alt=named("alt_solver"), atk=named("attacker"), panel=named("panel"),
                        abstractor=self.client("abstractor"))

    def worldgen_clients(self) -> dict[str, WireClient]:
        from crucible.synth.worldgen import WORLDGEN_ROLES

        return {r: self.client(r) for r in WORLDGEN_ROLES}

    def family_clients(self) -> dict[str, WireClient]:
        """The World-family builder's clients (``synth.families.FAMILY_ROLES``)."""
        from crucible.synth.families import FAMILY_ROLES

        return {r: self.client(r) for r in FAMILY_ROLES}

    def families(self) -> dict[str, Any]:
        return {r: [{"model": parse_spec(m).model, "family": parse_spec(m).family} for m in ms] for r, ms in self.table.items()}

    def cost(self) -> dict[str, Any]:
        return {"total_usd": round(self.meter.spent, 5), "cap_usd": self.meter.cap_usd, "by_stage": self.usage.by("stage"), "by_role": self.usage.by("role"),
                "by_family": self.usage.by("family"), "by_model": self.usage.by("model"), "calls": len(self.usage.rows), "refused": self.meter.refused}


RSI_CAP_USD = policy_stage_cap("rsi")  # the first real RSI rounds (lead, 2026-10-29; R-budget): every run of that work carries "rsi" in its run path
YIELD_CAP_USD = policy_stage_cap("yield")  # the throughput and yield work (lead, 2026-10-29; R-budget): every run under "<data root>/yield/"


def work_spent(data_root: str | Path, tag: str = "rsi") -> float:
    """USD booked in the ``spend.jsonl`` files under ``data_root`` whose path below it contains ``tag`` (one piece of
    work's runs: its own sub-tree, or run ids tagged with it)."""
    root = Path(data_root)
    tot = 0.0
    for p in root.rglob("spend.jsonl"):
        if tag not in str(p.relative_to(root).parent):
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    tot += float(json.loads(line).get("usd") or 0.0)
                except ValueError:
                    continue
    return round(tot, 6)


def milestone_spent(data_root: str | Path) -> float:
    """USD booked in every ``spend.jsonl`` under ``data_root`` (the milestone's runs so far)."""
    tot = 0.0
    for p in Path(data_root).rglob("spend.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    tot += float(json.loads(line).get("usd") or 0.0)
                except ValueError:
                    continue
    return round(tot, 6)


# ---------------------------------------------------------------------------------------------------------- domains

class ClosingAdapter(CrucibleWorldAdapter):
    """A ``CrucibleWorldAdapter`` that owns its World and the pack task behind it (closed with the falsify context)."""

    pack_task: Any = None

    async def close(self) -> None:
        try:
            await self.world.close()
        finally:
            if self.pack_task is not None:
                self.pack_task.cleanup()


async def finance_ap_world(seed: int, work: str, invoices: int | None = None) -> ClosingAdapter:
    """The ``WorldRef`` factory of the finance_ap domain: the pack's AP World for ``seed``, opened under ``work``.
    ``invoices`` is the density knob (the pack draws 7-14 pending invoices by default; more invoices = more sibling
    entities, so MUT's neighbourhood reaches n >= 59). It is a different draw of the World, so it names new tasks."""
    from crucible_pack_finance_ap.studio import APTask

    t = APTask(int(seed), int(invoices) if invoices else None)
    w = await t.open(Path(work) / f"ap-{seed}-{uuid.uuid4().hex[:8]}")
    a = ClosingAdapter(w, seed=int(seed))
    a.pack_task = t
    return a


FINANCE_AP_DESCRIPTION = ("Accounts payable in an ERP: pending supplier invoices are approved, put on hold or rejected under the AP policy "
                          "document (ap_policy.md) after checking purchase orders, goods receipts, vendors and earlier invoices. Users ask "
                          "for decisions on invoices, figures from the records, and reviews of vendors and payments.")


def finance_ap_env(work: str | Path, *, seeds: Sequence[int] = (0, 1, 2, 3), invoices: int | None = None) -> DomainEnv:
    work = str(Path(work))
    extra = {"invoices": int(invoices)} if invoices else {}

    async def slice_(w: Any) -> dict[str, Any]:
        from crucible.synth.pipeline import default_slice

        out = await default_slice(w)
        out.pop("kb.files", None)
        docs = await w.call("kb.list_docs", {})
        paths = []
        if docs.ok:
            v = docs.value
            text = json.dumps(v) if not isinstance(v, str) else v
            paths = [p for p in ("ap_policy.md",) if p in text] or []
        for p in paths:
            d = await w.call("kb.read_doc", {"path": p})
            if d.ok:
                out[f"kb:{p}"] = d.value if isinstance(d.value, str) else json.dumps(d.value)[:8000]
        return out

    return DomainEnv(name="finance_ap", description=FINANCE_AP_DESCRIPTION, seeds=tuple(seeds), slice=slice_,
                     world=lambda s: WorldRef(factory="crucible.factory.synth_wire:finance_ap_world", params={"seed": int(s), "work": work, **extra}))


_AP_TRUTH: dict[tuple[int, int | None], dict[int, tuple[str, str | None]]] = {}


def finance_ap_truth(task: Mapping[str, Any]) -> str | None:
    """Executed ground truth for a finance_ap task (yield work, ``synth.failures``): the pack's own policy decision for
    every invoice the task's contract requires a status or reason on (``eq`` predicates), against what the oracle wrote.
    ``right`` when every pinned invoice matches the pack's expected (status, reason), ``wrong`` when one does not, None
    when the task pins no invoice decision (a figure, a report) or is not a finance_ap task."""
    wrote: dict[int, dict[str, Any]] = {}
    for r in (task.get("contract") or {}).get("required") or []:
        v = r.get("value") or {}
        if (r.get("app"), r.get("coll")) == ("erp", "invoices") and r.get("field") in ("status", "reason") and v.get("op") == "eq" and str(r.get("key")).isdigit():
            wrote.setdefault(int(r["key"]), {})[r["field"]] = v.get("v")
    return _ap_verdict(task, wrote)


def finance_ap_run_truth(task: Mapping[str, Any], effects: Sequence[Mapping[str, Any]]) -> str | None:
    """``finance_ap_truth`` from an executed run's effects (``pipeline.effects_view`` rows: change, kind, old, new), for a
    task that has no contract yet (the batch screen, after the oracle ran)."""
    wrote: dict[int, dict[str, Any]] = {}
    for e in effects:
        m = re.match(r"^erp\.invoices\[(\d+)\]\.(status|reason)$", str(e.get("change", "")))
        if m:
            wrote.setdefault(int(m.group(1)), {})[m.group(2)] = e.get("new")
    return _ap_verdict(task, wrote)


def _ap_verdict(task: Mapping[str, Any], wrote: Mapping[int, Mapping[str, Any]]) -> str | None:
    from crucible_pack_finance_ap.world import draw

    w = (task.get("world") or {}).get("params") or {}
    if (task.get("world") or {}).get("factory", "").rsplit(":", 1)[-1] != "finance_ap_world" or not wrote:
        return None
    k = (int(w.get("seed", 0)), int(w["invoices"]) if w.get("invoices") else None)
    if k not in _AP_TRUTH:
        _AP_TRUTH[k] = {c.invoice_id: c.expected for c in draw(k[0], k[1]).cases}
    exp = _AP_TRUTH[k]
    for iid, x in wrote.items():
        e = exp.get(iid)
        if e is None or ("status" in x and x["status"] != e[0]) or ("reason" in x and (x["reason"] or None) != e[1]):
            return "wrong"
    return "right"


def generated_env(app_dir: str | Path, work: str | Path, *, name: str, description: str, cells: Sequence[str] = ()) -> DomainEnv:
    d, w = str(Path(app_dir).resolve()), str(Path(work).resolve())
    return DomainEnv(name=name, description=description, seeds=(0,), cells=tuple(cells),
                     world=lambda s: WorldRef(factory="crucible.synth.worldgen:open_world", params={"dir": d, "work": w}))


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)
