"""Studio hooks of the finance_ap pack (declared in ``pack.yaml``): the AP invoice World as an M10 ``RunTask`` with Φ per
step (one subgoal per invoice), gold and designed negatives, a seeded generator (v3, ``world.py``), scripted policies
and the adversaries a sound grader must hold down.

Grade (state only, from the ERP database):
- **Gate `controls`.** No invoice that a policy rule should have stopped was approved. Paying a duplicate, a blocked
  vendor, an unmatched purchase order, unverified bank details, an over-tolerance price or unreceived goods is a
  control failure, and the episode scores 0 however many other invoices were right.
- **Gate `outcome`.** Every invoice is in its correct final state with the policy's reason code. The reward is 1 only
  when both gates pass (M9 A2/A3/A16: with partial credit a dropped decision kept 0.89 and no mutant died);
  ``outcome`` in the report keeps the share decided correctly.

Φ is the dense credit, per invoice: an invoice counts while it is decided correctly **and** no control has failed
anywhere, so Φ_T equals the reported outcome share (1 exactly when the reward is 1) and a breach drops Φ to 0 at the
step that pays it (M14 audit V5).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import weakref
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from crucible.core.schema.state import Query
from crucible.core.store import Store
from crucible.runtime.identity import seed_digest, tree_digest
from crucible.verify.progress import Subgoal
from crucible.verify.runtime.plan import GradeReport
from crucible.world.world import World
from crucible_pack_finance_ap import PACK_ROOT
from crucible_pack_finance_ap.world import CONTROLS_SENTENCE, DECIDE_SENTENCE, GENERATOR_VERSION, APWorld, cents, decide, decision_call, draw, instruction, neighbour_tolerance, oracle, write_pack

TASK_VERSION = GENERATOR_VERSION  # task ids end in @3: generator v3 worlds are not v2's (M14 audit D8, campaign B7)
# every live task this process made, by id: stages() finds anchors and generated items here (M14 audit V12: anchors were
# never registered); weak, so a task nobody holds is collected and its temp dir removed
_MADE: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()


def breaches(w: APWorld, st: dict[int, tuple[str, str | None]]) -> list[int]:
    """Invoices a rule should have stopped that are approved: the ``controls`` gate."""
    return [c.invoice_id for c in w.cases if c.expected[0] != "approved" and (st.get(c.invoice_id) or ("", None))[0] == "approved"]


class InvoiceSubgoals:
    """One subgoal per invoice: decided correctly, and no control failed anywhere (M14 audit V5: Φ scored each invoice on
    its own, so paying a duplicate after six correct decisions kept Φ at 0.857 while the grade was 0). Every subgoal
    holds every invoice key, so a breach on one invoice re-evaluates them all."""

    def __init__(self, w: APWorld):
        self.w = w
        self.cases = {f"invoice:{c.invoice_id}": c for c in w.cases}
        every = frozenset(self.cases)
        self.subgoals = [Subgoal(k, every | {f"vendor:{c.vendor_id}", f"po:{c.po_id}"}) for k, c in self.cases.items()]

    async def state(self, world: Any) -> dict[int, tuple[str, str | None]]:
        res = (await world.verifier_view().query("erp", Query(op="sql", args={"sql": "SELECT id, status, reason FROM invoices", "max_rows": 100000}))).value or {}
        return {int(r[0]): (r[1], r[2]) for r in res.get("rows") or []}

    async def evaluate(self, world: Any, ids: Iterable[str]) -> dict[str, bool]:
        st = await self.state(world)
        ok = not breaches(self.w, st)
        return {i: ok and st.get(self.cases[i].invoice_id) == self.cases[i].expected for i in ids if i in self.cases}


class APTask:
    def __init__(self, seed: int = 0, n: int | None = None):
        self.seed = seed
        self.w = draw(seed, n)
        self.root = Path(tempfile.mkdtemp(prefix="finance-ap-"))
        # M14 audit V12: callers that never call cleanup() (pack check's fixture level, bench) leaked one dir per task
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(self.root), True)
        write_pack(self.w, self.root)
        self.id = f"finance_ap/invoices/s{seed}-n{len(self.w.cases)}@{TASK_VERSION}"
        self.version = TASK_VERSION
        self.world_id = "finance_ap/invoices"
        self.instruction = instruction(self.w)
        self.subgoals = InvoiceSubgoals(self.w)
        self.oracle = oracle(self.w)
        _MADE[self.id] = self

    def identity_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: the pack's content (the generator, the grader and the policy text: ``tree_digest(PACK_ROOT)``, so an
        edited grader refuses the resume), the seed and the number of invoices the world is drawn from, the task version and the
        instruction. ``root`` is the temporary directory the world is written to (a path, operational)."""
        return {"pack": tree_digest(PACK_ROOT), "seed": seed_digest(self.seed), "n": len(self.w.cases), "version": self.version, "instruction": self.instruction}

    async def open(self, workdir: Path) -> World:
        from crucible.core.schema.world import WorldSpec

        w = World(WorldSpec.load(self.root / "worlds" / "ap.yaml"), instructions={"agent": self.instruction}, store=Store(Path(workdir) / "s"),
                  workdir=Path(workdir) / "w", pack_root=self.root)
        await w.start()
        await w.reset()
        return w

    async def grade(self, world: World, workdir: Path) -> GradeReport:
        st = await self.subgoals.state(world)
        correct = {c.invoice_id: st.get(c.invoice_id) == c.expected for c in self.w.cases}
        bad = breaches(self.w, st)
        frac = sum(correct.values()) / len(correct)
        gates = {"controls": "fail" if bad else "pass", "outcome": "pass" if frac == 1.0 else "fail"}
        return GradeReport(task=self.id, tier="full", checks=[], gates=gates, reward=1.0 if all(v == "pass" for v in gates.values()) else 0.0,
                           outcome=0.0 if bad else round(frac, 6), infra={"status": "ok"}, trusted=True, elapsed_s=0.0,
                           meta={"correct": sum(correct.values()), "invoices": len(correct), "control_breaches": bad,
                                 "wrong": sorted(k for k, v in correct.items() if not v), "tolerance": self.w.tolerance, "hold_order": self.w.hold_order,
                                 "generator": GENERATOR_VERSION})

    def oracle_actions(self) -> list[dict]:
        return list(self.oracle)

    def cleanup(self) -> None:
        self._finalizer()

    # ---- the M9 Acceptance Contract's capabilities (``pack check --level accept``; studio.accept_adapter delegates) ----

    horizon = "H2"
    attack_surface = "world"

    @property
    def properties(self) -> list:
        """The instruction's stated properties (A7)."""
        from crucible.forge.schema import StatedProperty

        return [StatedProperty(id="decide", text=DECIDE_SENTENCE, kind="outcome"), StatedProperty(id="controls", text=CONTROLS_SENTENCE, kind="policy")]

    @property
    def checks(self) -> list:
        """What ``grade`` checks, as M9 check specs: each grades a stated property and cites the policy section (A7, A8)."""
        from crucible.forge.schema import CheckSpec

        return [CheckSpec(id="controls/no_stopped_invoice_approved", gate="controls", gate_kind="gate", kind="invariant", app="erp", states=["controls"],
                          provenance="kb:ap_policy.md#rules-1-6"),
                CheckSpec(id="outcome/invoices_decided_as_ap3", gate="outcome", gate_kind="gate", kind="predicate", app="erp", states=["decide"],
                          provenance="kb:ap_policy.md#rules-1-7")]

    @property
    def negatives(self) -> list:
        """Designed negatives: naive floors (pay everything, do nothing), each control breached once on the oracle, and a valid
        alternative (the decisions alone, in reverse order: no read is graded)."""
        from crucible.forge.schema import NegativeSpec

        decisions = [a for a in self.oracle if a["function_name"] in ("erp.approve_invoice", "erp.hold_invoice", "erp.reject_invoice")]
        out = [NegativeSpec(id="approve_all", task_id=self.id, kind="cheat", expected_gate="controls", provenance="kb:ap_policy.md#rules-1-6",
                            actions=[decision_call(c.invoice_id, ("approved", None), f"n{c.invoice_id}") for c in self.w.cases],
                            description="pays every invoice that arrives"),
               NegativeSpec(id="naive_noop", task_id=self.id, kind="floor", expected_gate="outcome", actions=[], provenance="kb:ap_policy.md#rules-1-7",
                            description="decides nothing"),
               NegativeSpec(id="decisions_only_reversed", task_id=self.id, kind="valid_alternative", expected="pass", actions=decisions[::-1],
                            provenance="kb:ap_policy.md#rules-1-7", description="the same decisions without the reads, last invoice first")]
        seen: set[str] = set()
        for c in self.w.cases:
            if c.expected[0] == "approved" or c.kind in seen:
                continue
            seen.add(c.kind)
            acts = [decision_call(c.invoice_id, ("approved", None), a["tool_call_id"]) if a.get("arguments", {}).get("invoice_id") == c.invoice_id
                    and a in decisions else a for a in self.oracle]
            out.append(NegativeSpec(id=f"approves_{c.kind}", task_id=self.id, kind="cheat", expected_gate="controls", actions=acts,
                                    provenance="kb:ap_policy.md#rules-1-6", description=f"pays the invoice rule {c.expected[1]} stops"))
        return out

    def mutants(self) -> list:
        """The action mutants of the oracle that change what is graded. The grade reads the ERP state only, so a mutant that
        corrupts only a read call (``get_invoice(101)`` instead of ``100``) cannot change it: equivalent by construction, and
        left out (M9 counts only ``drop_read`` as possibly equivalent)."""
        import difflib

        from crucible.accept.mutation import actions

        def is_write(a: dict) -> bool:
            return (a.get("function_name") or "").split(".")[-1].split("_")[0] in ("approve", "hold", "reject")

        def touches_write(sol: list) -> bool:
            key = lambda a: json.dumps(a, sort_keys=True)
            sm = difflib.SequenceMatcher(a=[key(a) for a in self.oracle], b=[key(a) for a in sol], autojunk=False)
            return any(op != "equal" and (any(is_write(a) for a in self.oracle[i1:i2]) or any(is_write(a) for a in sol[j1:j2]))
                       for op, i1, i2, j1, j2 in sm.get_opcodes())

        return [m for m in actions.generate(self.oracle) if touches_write(m.solution)]

    def hidden_strings(self) -> list[str]:
        """The answer key the agent must never read (A5): each invoice's expected decision as the generator records it
        (``forge/world.json``), and each serialized decision of the oracle."""
        out = []
        for c in self.w.cases:
            out.append(json.dumps(c.__dict__, indent=1))  # world.json's own serialization (A5 also matches it whitespace-normalized)
            out.append(json.dumps({"invoice_id": c.invoice_id, "expected": list(c.expected)}))
        for a in self.oracle:
            if a["function_name"] in ("erp.hold_invoice", "erp.reject_invoice", "erp.approve_invoice"):
                out.append(json.dumps(a))
        return out

    def facts(self) -> dict:
        return {"images": False, "fresh_gates": False, "perf_bars": False, "reference_build": False, "sibling_overlap": False}

    def declared_env(self) -> dict:
        return {"kind": "world", "world": str(self.root / "worlds" / "ap.yaml"), "backend": "local", "network": "none (no network tools)"}

    async def _run(self, actions: list, *, query: tuple[str, str] | None = None):
        from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

        d = Path(tempfile.mkdtemp(prefix="finance-ap-run-"))
        try:
            w = await self.open(d / "work")
            try:
                acts = _acts([{"kind": "tool_call", **a} if "function_name" in a and "kind" not in a else a for a in actions])
                await run_episode(w, {"agent": ScriptedPolicy("agent", acts)}, limits=EpisodeLimits(max_steps=max(50, 2 * len(acts))), reset=False)
                if query is not None:
                    v = (await w.verifier_view().query(query[0], Query(op="sql", args={"sql": query[1]}))).value or {}
                    rows = v.get("rows") or []
                    return rows[0][0] if rows else None
                fps = await w.verifier_view().fingerprints()
                return await self.grade(w, d / "grade"), json.dumps(fps, sort_keys=True, default=str)
            finally:
                await w.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def isolation_audit(self) -> dict[str, str]:
        """The World containment audit with the answer key as forbidden material (A6)."""
        from crucible.world.audit import audit_world

        d = Path(tempfile.mkdtemp(prefix="finance-ap-audit-"))
        try:
            w = await self.open(d / "work")
            try:
                rep = await audit_world(w, forbidden=self.hidden_strings(), actor="agent")
                return {c.id: c.status for c in rep.checks if c.severity in ("critical", "major")}
            finally:
                await w.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def tamper_attempts(self) -> list:
        """Out-of-contract writes through the policy's only surface: an undeclared SQL tool, a system restore, the verifier's
        view. Each must be refused (reward 0) and leave the ERP untouched (A6)."""
        from crucible.forge.schema import Attack

        tries = {"undeclared_sql": {"function_name": "erp.execute_sql", "arguments": {"sql": "UPDATE invoices SET status = 'approved'"}},
                 "system_restore": {"function_name": "__world.restore", "arguments": {"ref": "gold"}},
                 "verifier_view": {"function_name": "verifier.query", "arguments": {"app": "erp", "sql": "UPDATE invoices SET status = 'approved'"}}}
        out = []
        for k, a in tries.items():
            sol = [{**a, "kind": "tool_call", "tool_call_id": f"tamper-{k}"}]

            async def changed(sol=sol) -> bool:
                return (await self._run(sol))[1] != (await self._run([]))[1]

            out.append(Attack(k, "out_of_contract_write", sol, None, changed))
        return out

    def attacks(self) -> list:
        """Policy cheats against this grading (A13): pay everything; and do the oracle's work but pay the duplicate an exact
        number match misses. Each attack's control is that its payment really happened in the ERP."""
        from crucible.forge.schema import Attack

        def paid(inv: int, sol: list):
            async def took() -> bool:
                return (await self._run(sol, query=("erp", f"SELECT COUNT(*) FROM invoices WHERE id = {int(inv)} AND status = 'approved'")) or 0) > 0
            return took

        out = []
        stop = next((c for c in self.w.cases if c.expected[0] != "approved"), None)
        if stop is not None:
            sol = [decision_call(c.invoice_id, ("approved", None), f"a{c.invoice_id}") for c in self.w.cases]
            out.append(Attack("pay_everything", "skip_the_controls", sol, "controls", paid(stop.invoice_id, sol)))
        dup = next((c for c in self.w.cases if c.expected == ("rejected", "duplicate") and "dup_variant" in c.traits), None)
        if dup is not None:
            sol = [decision_call(dup.invoice_id, ("approved", None), a["tool_call_id"]) if a.get("arguments", {}).get("invoice_id") == dup.invoice_id
                   and a["function_name"] in ("erp.approve_invoice", "erp.hold_invoice", "erp.reject_invoice") else a for a in self.oracle]
            out.append(Attack("pay_variant_duplicate", "exact_match_only", sol, "controls", paid(dup.invoice_id, sol)))
        return out

    def solver_trajectories(self) -> list[list[dict]]:
        return [list(self.oracle)]

    async def property_audit(self, actions: list[dict]) -> bool:
        """A16's independent audit: the stated properties hold (every invoice decided as AP-3 says, no control failed)."""
        rep, _ = await self._run(list(actions))
        return rep.reward == 1.0


def _acts(raw: list[dict]) -> list:
    from crucible.core.schema.actions import Message, ToolCall

    return [ToolCall(tool_call_id=a["tool_call_id"], function_name=a["function_name"], arguments=a.get("arguments") or {}) if a["kind"] == "tool_call"
            else Message(to=a["to"], text=a["text"]) for a in raw]


def tasks() -> list:
    return [(t, _acts(t.oracle)) for t in (APTask(1), APTask(2))]


def generate(seed: int, item_id: str | None = None) -> APTask:
    """A fresh benchmark item. ``crucible.bench.dynamic.build`` passes the published opaque ``item_id``; otherwise the id
    is a one-way hash of the full seed (M13 audit V3: ``s{seed % 100_000}`` published the seed and an enumerable pool)."""
    t = APTask(seed)
    _MADE.pop(t.id, None)
    t.id = item_id or f"finance_ap/invoices/g{hashlib.sha256(f'finance_ap/invoices|{seed}'.encode()).hexdigest()[:16]}-n{len(t.w.cases)}@{TASK_VERSION}"
    _MADE[t.id] = t
    return t


def stages(task_id: str):
    t = _MADE.get(task_id)
    if t is None:  # re-keyed after generate
        t = next((x for x in _MADE.values() if getattr(x, "id", None) == task_id), None)
    if t is None:
        return None
    ids = [g.id for g in t.subgoals.subgoals]
    return [(f"q{q}", ids[: max(1, len(ids) * q // 4)]) for q in (1, 2, 3, 4)]


# ------------------------------------------------------------------------------------------------ scripted policies

def _decisions(task: APTask, decide_fn: Callable[[APWorld, dict], tuple[str, str | None]], prefix: str) -> list[dict]:
    """A policy that reads the ERP perfectly and decides every invoice with ``decide_fn`` (no policy document)."""
    inv = {i["id"]: i for i in task.w.invoices}
    return [decision_call(c.invoice_id, decide_fn(task.w, inv[c.invoice_id]), f"{prefix}{c.invoice_id}") for c in task.w.cases]


# The adversaries (M14 sweep G2 pick 2, D8; campaign B7): policies that skip the work a real clerk must do, scripted from
# the ERP data. ``expect_max`` bounds the reward of EVERY task (anchors and generated items: ``pack check --level semantic``
# runs each one on each). Each scores 0 on every generator v3 world by construction:
# - approve_all, exact_match_duplicates and dup_needs_same_amount breach a control in every world (each holds a variant
#   duplicate billed at a restated amount);
# - hold_all is wrong on every invoice to approve and every one to reject;
# - wrong_tolerance (another company's tolerance) gets one of the two price probes wrong in every world;
# - price_round_units approves the one-cent-over probe, whose whole-unit comparison is not over the bar;
# - reverse_rule_order gets a compound case's reason wrong in every world (two hold rules apply, reasons differ);
# - alias_vendor_args pays the stopped payee (a blocked vendor's alias, or a blocked vendor) in every world.
# Without a per-task bound (2,000 seeds, tests/unit/test_m14_audit_packs.py test_B7_*): any fixed hold order solves ~1/6
# of worlds (the world draws its order uniformly; v2: textbook 57.8%), a memorized 5% ~1/7, ignoring vendor aliases ~15%.
ADVERSARIES: dict[str, tuple[Callable[[APWorld, dict], tuple[str, str | None]], float, str]] = {
    "approve_all": (lambda w, i: ("approved", None), 0.0, "pays every invoice that arrives"),
    "hold_all": (lambda w, i: ("on_hold", "price_variance"), 0.0, "holds everything, so nothing is ever paid wrongly"),
    "exact_match_duplicates": (lambda w, i: decide(w, i, normalize=False)[0], 0.0,
                               "AP-3 otherwise, but duplicates only by an exact invoice-number match (misses INV-012345 vs inv 12345)"),
    "wrong_tolerance": (lambda w, i: decide(w, i, tol_pm=neighbour_tolerance(w.tol_pm))[0], 0.0,
                        "AP-3 otherwise, with another company's price tolerance (one not read from ap_policy.md)"),
    "reverse_rule_order": (lambda w, i: decide(w, i, reverse=True)[0], 0.0, "every rule, applied last-to-first"),
    "price_round_units": (lambda w, i: decide(w, i, whole_units=True)[0], 0.0, "AP-3 otherwise, with prices compared in whole currency units"),
    "dup_needs_same_amount": (lambda w, i: decide(w, i, dup_same_amount=True)[0], 0.0,
                              "AP-3 otherwise, but a duplicate only when the earlier invoice billed the same amount"),
}


def _alias_vendor_args(task: APTask, prefix: str = "al") -> list[dict]:
    """The campaign's alias_vendor_args: AP-3 on every invoice except those the vendor rule stops, which it pays, passing
    an active vendor id as an extra argument (the call's extra argument changes nothing in the ERP)."""
    active = next((v["id"] for v in task.w.vendors if v["status"] == "active" and not v["bank_changed_unverified"]), "V-100")
    out = []
    for c in task.w.cases:
        if c.expected == ("rejected", "vendor_blocked"):
            out.append({"kind": "tool_call", "tool_call_id": f"{prefix}{c.invoice_id}", "function_name": "erp.approve_invoice",
                        "arguments": {"invoice_id": c.invoice_id, "vendor_id": active}})
            out.append(decision_call(c.invoice_id, ("approved", None), f"{prefix}2{c.invoice_id}"))
        else:
            out.append(decision_call(c.invoice_id, c.expected, f"{prefix}{c.invoice_id}"))
    return out


def adversaries() -> list[dict]:
    """``[{"name", "policy": task -> [Action], "expect_max": float, "why"}]``: the ``fixtures`` adversary entries (the
    framework runs each on every anchor and generated item and fails the pack when one scores above its bound)."""

    def make(name: str, fn):
        return lambda task: _acts(_decisions(task, fn, name[:3]))

    return [*({"name": f"adversary:{name}", "policy": make(name, fn), "expect_max": bound, "why": why} for name, (fn, bound, why) in ADVERSARIES.items()),
            {"name": "adversary:alias_vendor_args", "policy": lambda task: _acts(_alias_vendor_args(task)), "expect_max": 0.0,
             "why": "pays the payee the vendor rule stops (a blocked vendor's alias or the blocked vendor), naming an active vendor id"}]


def policies() -> list:
    from crucible.actors.policy import ScriptedPolicy
    from crucible.runtime.harness.native import PolicyRef

    approve_all = ADVERSARIES["approve_all"][0]
    # M10 lab audit R-4: a policy's configuration is its run identity (PolicyRef.config); these scripts derive from the task
    return [PolicyRef(id="scripted:oracle", make=lambda a: {"agent": ScriptedPolicy("agent", _acts(a.task.oracle))}, config={"script": "task oracle"}),
            PolicyRef(id="scripted:approve_all", make=lambda a: {"agent": ScriptedPolicy("agent", _acts(_decisions(a.task, approve_all, "a")))},
                      config={"script": "approve every invoice"}),
            PolicyRef(id="scripted:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])}, config={"script": "noop"})]


# ------------------------------------------------------------------------------------------------ fixtures

def _one_cent_over(w: APWorld, c) -> bool:
    po = next((p for p in w.pos if p["id"] == c.po_id), None)
    return c.kind == "price_variance" and po is not None and 0 < cents(c.amount) * 1000 - cents(po["amount"]) * (1000 + w.tol_pm) <= 1000


def _fixture_task() -> APTask:
    """The first seed (from 3) whose world holds every case the fixtures name: each control, a compound case whose rule
    order changes the reason, a price exactly at the bar and one cent over it, a variant duplicate at a restated amount,
    a vendor alias and an earlier same-number invoice that was rejected or held."""
    for seed in range(3, 2000):
        w = draw(seed, 14)
        inv = {i["id"]: i for i in w.invoices}
        kinds = {c.kind for c in w.cases}
        one_cent = any(_one_cent_over(w, c) for c in w.cases)
        if ({"duplicate", "vendor_blocked", "bank_change", "price_variance", "missing_receipt", "po_mismatch"} <= kinds and one_cent
                and any("at_tolerance" in c.traits and c.expected[0] == "approved" for c in w.cases)
                and any("dup_earlier_rejected" in c.traits and c.expected[0] == "approved" for c in w.cases)
                and any(decide(w, inv[c.invoice_id], reverse=True)[0] != c.expected for c in w.cases)
                and any("vendor_alias" in c.traits for c in w.cases)):
            return APTask(seed, 14)
    raise RuntimeError("no fixture world in seeds 3..1999")


def fixtures() -> list[dict]:
    """Gold, and designed negatives that each break one control, one decision or one reading of the policy."""
    t = _fixture_task()
    gold = list(t.oracle)
    inv = {i["id"]: i for i in t.w.invoices}

    def decide_as(c, status: str, reason: str | None = None) -> list[dict]:
        out = []
        for a in gold:
            if a.get("arguments", {}).get("invoice_id") == c.invoice_id and a["function_name"] in ("erp.approve_invoice", "erp.hold_invoice", "erp.reject_invoice"):
                out.append(decision_call(c.invoice_id, (status, reason), a["tool_call_id"]))
            else:
                out.append(a)
        return out

    def first(pred) -> Any:
        return next(c for c in t.w.cases if pred(c))

    cases = [{"name": "gold", "task": t, "actions": _acts(gold), "reward": 1.0}]
    for kind in ("duplicate", "vendor_blocked", "bank_change", "price_variance", "missing_receipt", "po_mismatch"):
        cases.append({"name": f"approves_{kind}", "task": t, "actions": _acts(decide_as(first(lambda c, kind=kind: c.kind == kind), "approved")),
                      "reward": 0.0, "fails_gate": "controls"})
    c = first(lambda c: _one_cent_over(t.w, c))
    cases.append({"name": "approves_one_cent_over_tolerance", "task": t, "actions": _acts(decide_as(c, "approved")), "reward": 0.0, "fails_gate": "controls"})
    c = first(lambda c: "at_tolerance" in c.traits and c.expected[0] == "approved")
    cases.append({"name": "holds_exactly_at_tolerance", "task": t, "actions": _acts(decide_as(c, "on_hold", "price_variance")), "reward": 0.0, "fails_gate": "outcome"})
    c = first(lambda c: "dup_earlier_rejected" in c.traits and c.expected[0] == "approved")
    cases.append({"name": "rejects_number_whose_earlier_invoice_was_rejected", "task": t, "actions": _acts(decide_as(c, "rejected", "duplicate")),
                  "reward": 0.0, "fails_gate": "outcome"})
    c = first(lambda c: "vendor_alias" in c.traits)
    cases.append({"name": "approves_vendor_alias", "task": t, "actions": _acts(decide_as(c, "approved")), "reward": 0.0, "fails_gate": "controls"})
    c = first(lambda c: "dup_restated" in c.traits)
    cases.append({"name": "approves_restated_duplicate", "task": t, "actions": _acts(decide_as(c, "approved")), "reward": 0.0, "fails_gate": "controls"})
    c = first(lambda c: c.kind == "vendor_blocked")
    cases.append({"name": "reject_with_wrong_reason", "task": t, "actions": _acts(decide_as(c, "rejected", "duplicate" if c.expected[1] != "duplicate" else "vendor_blocked")),
                  "reward": 0.0, "fails_gate": "outcome"})
    c = first(lambda c: c.kind == "duplicate")
    cases.append({"name": "holds_instead_of_rejecting", "task": t, "actions": _acts(decide_as(c, "on_hold", "price_variance")), "reward": 0.0, "fails_gate": "outcome"})
    c = first(lambda c: c.kind == "price_variance")
    cases.append({"name": "wrong_hold_reason", "task": t, "actions": _acts(decide_as(c, "on_hold", "missing_receipt")), "reward": 0.0, "fails_gate": "outcome"})
    # whole-episode adversaries, graded exactly here: each misses at least one invoice, so each scores 0
    for name in ("exact_match_duplicates", "reverse_rule_order", "wrong_tolerance"):
        fn = ADVERSARIES[name][0]
        dec = {c.invoice_id: fn(t.w, inv[c.invoice_id]) for c in t.w.cases}
        bad = any(c.expected[0] != "approved" and dec[c.invoice_id][0] == "approved" for c in t.w.cases)
        cases.append({"name": f"adversary_{name}", "task": t, "actions": _acts(_decisions(t, fn, name[:3])), "reward": 0.0,
                      "fails_gate": "controls" if bad else "outcome"})
    cases.append({"name": "noop", "task": t, "actions": [], "reward": 0.0})
    return cases + adversaries()  # bounded on every task and generated item at the semantic level
