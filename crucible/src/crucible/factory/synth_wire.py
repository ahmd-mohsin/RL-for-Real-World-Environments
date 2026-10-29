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
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.models import DEEPSEEK, GLM, GPT_OSS, KIMI, NEMOTRON, NEMOTRON_ULTRA, ModelBook, SameFamilyError, backend, parse_spec
from crucible.synth.iface import CrucibleWorldAdapter
from crucible.synth.pipeline import DomainEnv, RunRoles
from crucible.synth.roles import ROLES, family_problems
from crucible.synth.task import WorldRef

MILESTONE_CAP_USD = 60.0  # M4's Tinker budget (lead, 2026-10-29): enforced by the CLI through ``milestone_spent``

# synth role -> models (the swarm preset's families; see the module docstring for the rules they satisfy)
SYNTH_ROLES: dict[str, tuple[str, ...]] = {
    "proposer": (KIMI,),
    "oracle_programmer": (DEEPSEEK,),
    "contract_writer": (GLM,),
    "repairer": (DEEPSEEK,),
    "alt_solver": (NEMOTRON, GPT_OSS),
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
}


def _book_roles(table: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {f"synth_{r}_{i}": m for r, ms in table.items() for i, m in enumerate(ms)}


def check_assignment(table: Mapping[str, Sequence[str]], *, students: Sequence[str] | None = None) -> ModelBook:
    """The ModelBook of the assignment, after both rule sets; ``SameFamilyError`` on any violation."""
    book = ModelBook(roles=_book_roles(table), preset="swarm")
    studs = list(students) if students is not None else [s.family for s in book.solvers]
    fams = {r: [parse_spec(m).family for m in ms] for r, ms in table.items()}
    probs = family_problems(fams, students=studs)
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


class WireClient:
    """``roles.RoleClient`` over an ``LLMClient``-shaped chat client (``complete(messages, ...) -> LLMResponse``)."""

    def __init__(self, inner: Any, *, role: str, model: str, family: str, usage: Usage, json_retries: int = 2):
        self.inner, self.role, self.model, self.family, self.usage, self.json_retries = inner, role, model, family, usage, json_retries
        self.calls = 0

    async def complete_json(self, messages: list[dict[str, Any]], *, schema: Any = None, max_tokens: int = 2048, temperature: float = 0.0,
                            tag: str = "") -> dict[str, Any]:
        from crucible.actors.llm import LLMOutputError
        from crucible.train.bridges.tinker.sampler import check_schema

        msgs = list(messages)
        last = ""
        for attempt in range(self.json_retries + 1):
            t0 = time.monotonic()
            r = await self.inner.complete(msgs, temperature=temperature, max_tokens=max_tokens, seed=attempt)
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

class SynthWire:
    """Clients for every synth role on one run's meter (module docstring)."""

    def __init__(self, run_dir: str | Path, *, cap_usd: float, table: Mapping[str, Sequence[str]] | None = None, make: Any = None):
        from crucible.train.bridges.tinker.spend import SpendMeter

        self.dir = Path(run_dir)
        self.table = dict(table or SYNTH_ROLES)
        self.book = check_assignment(self.table)
        self.meter = SpendMeter(cap_usd, path=self.dir / "spend.jsonl")
        self.usage = Usage(self.dir / "usage.jsonl")
        self._make = make  # tests inject a fake maker(model, role) -> chat client
        self._clients: dict[tuple[str, int], WireClient] = {}

    def client(self, role: str, i: int = 0) -> WireClient:
        if role not in ROLES:
            raise KeyError(f"unknown synth role {role!r}")
        k = (role, i)
        if k not in self._clients:
            spec = self.book.spec(f"synth_{role}_{i}")
            if self._make is not None:
                inner = self._make(spec.model, role)
            else:
                inner = backend(spec.provider).make(spec.model, role=role, meter=self.meter)
            self._clients[k] = WireClient(inner, role=role, model=spec.model, family=spec.family, usage=self.usage)
        return self._clients[k]

    def clients(self, role: str) -> list[WireClient]:
        return [self.client(role, i) for i in range(len(self.table[role]))]

    def run_roles(self) -> RunRoles:
        def named(role: str) -> list[tuple[str, WireClient]]:
            return [(f"{role}-{c.family}", c) for c in self.clients(role)]

        return RunRoles(proposer=self.client("proposer"), oracle_programmer=self.client("oracle_programmer"), contract_writer=self.client("contract_writer"),
                        repairer=self.client("repairer"), alt=named("alt_solver"), atk=named("attacker"), panel=named("panel"),
                        abstractor=self.client("abstractor"))

    def worldgen_clients(self) -> dict[str, WireClient]:
        from crucible.synth.worldgen import WORLDGEN_ROLES

        return {r: self.client(r) for r in WORLDGEN_ROLES}

    def families(self) -> dict[str, Any]:
        return {r: [{"model": parse_spec(m).model, "family": parse_spec(m).family} for m in ms] for r, ms in self.table.items()}

    def cost(self) -> dict[str, Any]:
        return {"total_usd": round(self.meter.spent, 5), "cap_usd": self.meter.cap_usd, "by_stage": self.usage.by("stage"), "by_role": self.usage.by("role"),
                "by_family": self.usage.by("family"), "by_model": self.usage.by("model"), "calls": len(self.usage.rows), "refused": self.meter.refused}


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


async def finance_ap_world(seed: int, work: str) -> ClosingAdapter:
    """The ``WorldRef`` factory of the finance_ap domain: the pack's AP World for ``seed``, opened under ``work``."""
    from crucible_pack_finance_ap.studio import APTask

    t = APTask(int(seed))
    w = await t.open(Path(work) / f"ap-{seed}-{uuid.uuid4().hex[:8]}")
    a = ClosingAdapter(w, seed=int(seed))
    a.pack_task = t
    return a


FINANCE_AP_DESCRIPTION = ("Accounts payable in an ERP: pending supplier invoices are approved, put on hold or rejected under the AP policy "
                          "document (ap_policy.md) after checking purchase orders, goods receipts, vendors and earlier invoices. Users ask "
                          "for decisions on invoices, figures from the records, and reviews of vendors and payments.")


def finance_ap_env(work: str | Path, *, seeds: Sequence[int] = (0, 1, 2, 3)) -> DomainEnv:
    work = str(Path(work))

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
                     world=lambda s: WorldRef(factory="crucible.factory.synth_wire:finance_ap_world", params={"seed": int(s), "work": work}))


def generated_env(app_dir: str | Path, work: str | Path, *, name: str, description: str) -> DomainEnv:
    d, w = str(Path(app_dir).resolve()), str(Path(work).resolve())
    return DomainEnv(name=name, description=description, seeds=(0,),
                     world=lambda s: WorldRef(factory="crucible.synth.worldgen:open_world", params={"dir": d, "work": w}))


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)
