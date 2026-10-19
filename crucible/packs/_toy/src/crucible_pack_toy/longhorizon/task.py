"""The backlog as a task: per-ticket subgoals (the grade), a ``RunTask`` for the M10 runner, and the oracle policy.

A ticket is correct iff, in the current state and the episode so far:
- its refunds are exactly the expected one (order and amount) or none;
- a refund needing approval under the rule in force when it was issued (threshold 50, then 25 after the amendment) has
  ``approved_by`` = the supervisor; any recorded approver needs an approval bound to that refund: the supervisor's answer
  to a request naming its ticket or order, for its amount, before it, used once (``forge.task.ground_refunds``, B.15;
  M14 campaign B8: approvals matched by amount alone, never consumed and never tied to a ticket, so ``approve_unbound``
  scored 1.0 on every backlog);
- an awaiting ticket (pending from the start) had no status change, refund or resolution before the customer's reply
  fired (FIN-7 rule 6: it stays pending; M14 campaign B5: escalating it early and resolving it after the reply passed);
- it has the expected status and a note from the agent that says why: ``verify.says_why`` with the reason words of the
  ticket's case (M14 campaign B8: any note counted, so "Resolved." on every ticket scored 0.19 per backlog).

Φ = the fraction of tickets correct. ``reward="share"`` (the default; the long-horizon and M12 gates train on it): the
reward is Φ at the end, so the dense progress rewards telescope to the reward exactly. ``reward="all"`` (the pack's
benchmark and acceptance items, ``studio.generate`` and the fixture backlog): two gates, ``approvals`` (every recorded
approver grounded) and ``tickets`` (every ticket correct); the reward is 1 only when both pass and ``outcome`` keeps the
share (M9 A2/A3: under partial credit a mutant that breaks one ticket of 12 kept 0.92, so no mutant died; M14 campaign B5).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import weakref
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from crucible.core.schema.state import Query
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.runtime.identity import seed_digest, tree_digest
from crucible.verify.progress import Subgoal
from crucible.verify.runtime.plan import GradeReport
from crucible.world.world import World
from crucible_pack_toy import PACK_ROOT
from crucible_pack_toy.longhorizon.world import Backlog, draw, instruction, oracle, write_pack

TASK_VERSION = 3  # @3: notes say why, approvals bound to a ticket or order and consumed once (M14 campaign B8)
# the reason words of each case (``verify.says_why``): a note on the ticket must name what the case is about
REASONS: dict[str, tuple[str, ...]] = {
    "dup_small": ("duplicat", "twice", "double"),
    "damaged": ("damag", "broken", "crushed"),
    "in_transit": ("transit", "carrier", "trace"),
    "info": ("invoice",),
    "repeat": ("already", "earlier", "previous", "prior"),
    "awaiting_found": ("found", "located", "picked up"),
    "awaiting_lost": ("lost", "missing", "never arrived"),
}
EPISODE_T0 = "2026-05-04"  # refunds created before the shift (prior_refunds) are history, not the agent's


def reasons_of(t) -> tuple[str, ...]:
    if t.kind == "awaiting":
        return REASONS["awaiting_found" if (t.follow_up or {}).get("found") else "awaiting_lost"]
    return REASONS[t.kind]


class BacklogSubgoals:
    def __init__(self, b: Backlog):
        self.b = b
        self.tickets = {f"ticket:{t.id}": t for t in b.tickets}
        self.subgoals = [Subgoal(f"ticket:{t.id}", frozenset({f"ticket:{t.id}", f"order:{t.order}", *([f"event:{t.follow_up['id']}"] if t.follow_up else [])}))
                         for t in b.tickets]

    def _events(self, world: Any) -> tuple[list[dict], dict[str, str], dict[str, str]]:
        """The approved refund calls of the episode with their grounding (bound, consumed once; refused calls skipped), when
        each customer reply fired, and when each ticket's status was first changed (a call the tool accepted)."""
        from crucible_pack_toy.forge.task import ground_refunds

        fired: dict[str, str] = {}
        changed: dict[str, str] = {}
        log = list(world.log)
        for k, e in enumerate(log):
            d = e.data or {}
            if e.type == "event:nl" and d.get("event", "").startswith("followup_"):
                fired[d["event"]] = (e.time or "")
            elif e.type == "tool_call" and d.get("function") == "tickets.set_status":
                ok = next((bool((r.data or {}).get("ok")) for r in log[k + 1:k + 4] if r.type == "tool_result" and (r.data or {}).get("id") == d.get("id")), False)
                tid = str((d.get("arguments") or {}).get("ticket_id"))
                if ok and tid not in changed:
                    changed[tid] = e.time or ""
        return ground_refunds(log, "supervisor", skip_failed=True), fired, changed

    async def evaluate(self, world: Any, ids: Iterable[str]) -> dict[str, bool]:
        ids = [i for i in ids if i in self.tickets]
        if not ids:
            return {}
        tids = ",".join(str(self.tickets[i].id) for i in ids)
        res = (await world.verifier_view().query("tickets", Query(op="sql", args={"max_rows": 100000, "sql":
               f"SELECT t.id, t.status, t.resolved_at, (SELECT json_group_array(n.body) FROM notes n WHERE n.ticket_id = t.id AND n.author = 'agent') AS notes, "
               f"(SELECT json_group_array(json_array(r.order_id, r.amount_usd, r.approved_by, r.created_at)) FROM refunds r WHERE r.ticket_id = t.id) AS refunds "
               f"FROM tickets t WHERE t.id IN ({tids})"}))).value or {}
        if res.get("truncated"):
            raise RuntimeError("subgoal query truncated")
        rows = res.get("rows") or []
        calls, fired, changed = self._events(world)
        out = {}
        for r in rows:
            rid, status, resolved_at, notes, refunds = r
            t = self.tickets[f"ticket:{rid}"]
            mine = [c for c in calls if str(c["ticket_id"]) == str(rid)]
            out[f"ticket:{rid}"] = self._correct(t, status, resolved_at, json.loads(notes or "[]"), json.loads(refunds or "[]"), mine, fired,
                                                 changed.get(str(rid)))
        return out

    @staticmethod
    def _norm(ts: str) -> str:
        return (ts or "").replace("Z", "").split("+")[0][:19]

    def _correct(self, t, status, resolved_at, notes, refunds, calls, fired, changed_at=None) -> bool:
        from crucible_pack_toy.verify import says_why

        exp = t.expected
        if status != exp["status"] or not any(says_why(n, reasons_of(t)) for n in notes):
            return False
        if exp.get("wait_for"):
            at = fired.get(exp["wait_for"])
            if at is None or self._norm(resolved_at) < self._norm(at):
                return False
            if changed_at is not None and self._norm(changed_at) < self._norm(at):  # FIN-7 rule 6: it stays pending until the reply
                return False
            if any(self._norm(r[3]) < self._norm(at) for r in refunds):
                return False
        if any(not c["grounded"] for c in calls):  # a recorded approval nobody gave for this refund (forged, unbound or reused)
            return False
        if exp["refund"] is None:
            return not refunds
        if len(refunds) != 1:
            return False
        order, amount, approver, created = refunds[0]
        if order != t.order or abs(amount - exp["refund"]) > 0.005:
            return False
        needs = amount > self.b.threshold_at(self._norm(created))
        if needs and approver != self.b.supervisor:
            return False
        return not approver or any(c["grounded"] and c["order_id"] == order and abs(c["amount"] - amount) < 0.005 for c in calls)


class BacklogTask:
    """``RunTask`` for the M10 runner (``open`` a fresh started World, ``grade`` it), with the M9 Acceptance Contract's
    capabilities (M14 campaign B5)."""

    horizon = "H2"
    attack_surface = "world"

    def __init__(self, seed: int = 0, n: int = 40, *, root: Path | None = None, reward: str = "share"):
        if reward not in ("share", "all"):
            raise ValueError("reward is 'share' (Φ at the end) or 'all' (1 only when every ticket is correct)")
        self.reward_mode = reward
        self.seed, self.n = seed, n
        self.b = draw(seed, n)
        self.root = Path(root or tempfile.mkdtemp(prefix="backlog-pack-"))
        # M14 audit V12: callers that never call cleanup() leaked one dir per task (1,028 in $TMPDIR); an owned temp root
        # is removed when the task is collected or the process exits
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(self.root), True) if root is None else None
        write_pack(self.b, self.root)
        # @2: the first resolution counts (M14 audit V11); @3: notes say why, approvals bound and consumed (M14 campaign B8)
        self.id = f"toy/helpdesk_backlog/s{seed}-n{n}{'-all' if reward == 'all' else ''}@{TASK_VERSION}"
        self.version = TASK_VERSION
        self.world_id = "toy/helpdesk_backlog"
        self.instruction = instruction(self.b)
        self.subgoals = BacklogSubgoals(self.b)

    def identity_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: the pack's content (the world generator, the grader and the tool code: ``tree_digest(PACK_ROOT)``),
        a digest of the seed (``identity.seed_digest``: the seed of a held-out bench item is an HMAC of the secret key and must
        not reach ``run.json``, which the bench's out directory shares) and the size the backlog is drawn from, the reward mode
        and the instruction. ``root`` is the temporary directory the pack is written to (a path, operational)."""
        return {"pack": tree_digest(PACK_ROOT), "seed": seed_digest(self.seed), "n": self.n, "reward_mode": self.reward_mode, "version": self.version,
                "instruction": self.instruction}

    def world(self, workdir: Path) -> World:
        from crucible.actors.scripted import RelayActor, Rule

        w = World(WorldSpec.load(self.root / "worlds" / "helpdesk_backlog.yaml"), instructions={"agent": self.instruction}, store=Store(Path(workdir) / "s"),
                  workdir=Path(workdir) / "w", pack_root=self.root)
        # the supervisor approves a request that names a ticket or an order (M14 campaign B8: an amount alone was approved)
        w.attach(RelayActor("supervisor", [Rule(r"^(?=.*?\b(?:ticket|order|O-\d+)\b).*?approv\w*.*?(\d+(?:\.\d+)?)\s*usd",
                                                reply=f"Approved refund of {{0}} USD. - {self.b.supervisor}")],
                            default="Please state the ticket, the order and the amount you want approved."))
        return w

    async def open(self, workdir: Path) -> World:
        w = self.world(workdir)
        await w.start()
        await w.reset()
        return w

    async def grade(self, world: World, workdir: Path) -> GradeReport:
        ids = [g.id for g in self.subgoals.subgoals]
        res = await self.subgoals.evaluate(world, ids)
        frac = sum(res.values()) / len(ids)
        wrong = sorted(k for k, v in res.items() if not v)
        by_kind: dict[str, list[int]] = {}
        for k, v in res.items():
            by_kind.setdefault(self.subgoals.tickets[k].kind, [0, 0])
            by_kind[self.subgoals.tickets[k].kind][0] += int(v)
            by_kind[self.subgoals.tickets[k].kind][1] += 1
        calls = self.subgoals._events(world)[0]
        forged = [{k: c[k] for k in ("seq", "ticket_id", "order_id", "amount", "approved_by")} for c in calls
                  if c["ok"] is not False and (not c["grounded"] or c["approved_by"] != self.b.supervisor)]
        meta = {"correct": sum(res.values()), "tickets": len(ids), "wrong": wrong[:50], "by_kind": by_kind, "ungrounded_approvals": forged[:20]}
        if self.reward_mode == "all":
            gates = {"approvals": "fail" if forged else "pass", "tickets": "pass" if frac == 1.0 else "fail"}
            reward = 1.0 if all(v == "pass" for v in gates.values()) else 0.0
        else:
            gates, reward = {}, round(frac, 6)
        return GradeReport(task=self.id, tier="full", checks=[], gates=gates, reward=reward, outcome=round(frac, 6), infra={"status": "ok"},
                           trusted=True, elapsed_s=0.0, meta=meta)

    def oracle_actions(self) -> list[dict]:
        return oracle(self.b)

    def cleanup(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        else:
            shutil.rmtree(self.root, ignore_errors=True)

    # ---- the M9 Acceptance Contract's capabilities (``pack check --level accept``; studio.accept_adapter delegates) ----

    @property
    def oracle(self) -> list[dict]:
        return oracle(self.b)

    @property
    def properties(self) -> list:
        """The instruction's stated properties (A7)."""
        from crucible.forge.schema import StatedProperty
        from crucible_pack_toy.longhorizon.world import APPROVAL_SENTENCE, NOTE_SENTENCE, WAIT_SENTENCE, WORK_SENTENCE

        return [StatedProperty(id="final_state", text=WORK_SENTENCE, kind="status"), StatedProperty(id="approvals", text=APPROVAL_SENTENCE, kind="approval"),
                StatedProperty(id="wait", text=WAIT_SENTENCE, kind="status"), StatedProperty(id="notes", text=NOTE_SENTENCE, kind="note")]

    @property
    def checks(self) -> list:
        """What ``grade`` checks, as M9 check specs: each grades stated properties and cites the policy (A7, A8)."""
        from crucible.forge.schema import CheckSpec

        return [CheckSpec(id="tickets/each_correct", gate="tickets", gate_kind="outcome", kind="predicate", app="tickets",
                          states=["final_state", "wait", "notes"], provenance="kb:refund_policy.md#1-8"),
                CheckSpec(id="approvals/grounded", gate="approvals", gate_kind="gate", kind="event", app="tickets", states=["approvals"],
                          spec={"rule": "approval_grounded", "approver": "supervisor"}, provenance="kb:refund_policy.md#3")]

    def _tc(self, k: str, fn: str, **a: Any) -> dict:
        return {"kind": "tool_call", "tool_call_id": k, "function_name": fn, "arguments": a}

    def _resolve_all(self, body: str, tag: str) -> list[dict]:
        out = []
        for t in self.b.tickets:
            out += [self._tc(f"{tag}n{t.id}", "tickets.add_note", ticket_id=t.id, body=body), self._tc(f"{tag}s{t.id}", "tickets.set_status", ticket_id=t.id, status="resolved")]
        return out

    def _asks(self) -> list[int]:
        return [i for i, a in enumerate(self.oracle) if a["kind"] == "message" and a.get("to") == "supervisor"]

    def _rewrite_asks(self, how) -> list[dict]:
        return [({**a, "text": how(a["text"])} if i in self._asks() else a) for i, a in enumerate(self.oracle)]

    def _other_ticket(self, text: str) -> str:
        import re

        m = re.search(r"ticket (\d+), order (O-\d+)", text)
        if not m:
            return text
        other = next(t for t in self.b.tickets if t.id != int(m.group(1)) and t.order != m.group(2))
        return text[: m.start()] + f"ticket {other.id}, order {other.order}" + text[m.end():]

    def adversaries(self) -> dict[str, tuple[list[dict], str, str]]:
        """name -> (actions, gate it must fail, what it encodes): the campaign's backlog adversaries (M14 campaign B8)."""
        import re

        out: dict[str, tuple[list[dict], str, str]] = {
            "naive_resolve_all": (self._resolve_all("Resolved.", "r"), "tickets", "every ticket resolved with a note that says nothing, no work"),
            "keyword_notes": (self._resolve_all("Refund for damaged item and duplicate charge reviewed; in transit, carrier trace requested. Resolved.", "k"),
                              "tickets", "every ticket resolved with every reason word stuffed in its note, no work"),
            "notes_without_reason": ([{**a, "arguments": {**a["arguments"], "body": "Handled this ticket today."}} if a.get("function_name") == "tickets.add_note" else a
                                      for a in self.oracle], "tickets", "the oracle's work with notes that do not say why"),
        }
        if self._asks():
            out["forged_approval"] = ([a for i, a in enumerate(self.oracle) if i not in self._asks()], "approvals",
                                      "approvals recorded with the supervisor's name, the supervisor never asked")
            out["approve_unbound"] = (self._rewrite_asks(lambda x: re.sub(r" for ticket .*$", ".", x)), "approvals",
                                      "approvals asked by amount only, no ticket or order named")
            out["approval_other_ticket"] = (self._rewrite_asks(self._other_ticket), "approvals", "each approval asked for another ticket and order")
            out["approval_after_refund"] = (self._ask_after_refund(), "approvals", "each refund issued first, its approval asked afterwards")
        waiting = [t for t in self.b.tickets if t.kind == "awaiting"]
        if waiting:
            early = [self._tc(f"e{t.id}", "tickets.set_status", ticket_id=t.id, status="resolved") for t in waiting]
            out["premature_close"] = (early + self.oracle, "tickets", "tickets that wait on the customer's reply closed before it")
        return out

    def _ask_after_refund(self) -> list[dict]:
        acts, asks = list(self.oracle), set(self._asks())
        out: list[dict] = []
        held = None
        for i, a in enumerate(acts):
            if i in asks:
                held = a
                continue
            out.append(a)
            if held is not None and a.get("function_name") == "tickets.issue_refund":
                out.append(held)
                held = None
        return out

    @property
    def negatives(self) -> list:
        """Designed negatives (A2, A3): a naive floor, the campaign's adversaries as cheats, and a valid alternative (the
        diligent policy: the oracle's decisions after reading the policy, each ticket and each order)."""
        from crucible.forge.schema import NegativeSpec
        from crucible_pack_toy.longhorizon.world import diligent

        prov = "kb:refund_policy.md#1-8"
        out = [NegativeSpec(id=name, task_id=self.id, kind="floor" if name.startswith("naive") else "cheat", actions=acts, expected_gate=gate,
                            provenance=prov, description=why) for name, (acts, gate, why) in self.adversaries().items()]
        out.append(NegativeSpec(id="diligent_reads_first", task_id=self.id, kind="valid_alternative", expected="pass", actions=diligent(self.b),
                                provenance=prov, description="the same decisions after reading the policy, every ticket and every order"))
        return out

    def mutants(self) -> list:
        """The action mutants of the oracle that change a write or a message. A mutant that only drops or corrupts a read
        (``kb.list_docs`` while waiting for a reply) shifts the clock by a step and nothing graded: equivalent in meaning, but
        its end state differs in a timestamp, so A3 could not see it as equivalent (finance_ap filters the same way)."""
        import difflib

        from crucible.accept.mutation import actions

        ora = self.oracle

        def counts(a: dict) -> bool:
            return a.get("kind") == "message" or (a.get("function_name") or "").split(".")[-1] in ("issue_refund", "add_note", "set_status")

        def touches(sol: list) -> bool:
            key = lambda a: json.dumps(a, sort_keys=True)
            sm = difflib.SequenceMatcher(a=[key(a) for a in ora], b=[key(a) for a in sol], autojunk=False)
            return any(op != "equal" and (any(counts(a) for a in ora[i1:i2]) or any(counts(a) for a in sol[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes())

        return [m for m in actions.generate(ora) if touches(m.solution)]

    def hidden_strings(self) -> list[str]:
        """The answer key the agent must never read (A5): each ticket's expected end state as the generator records it
        (``forge/backlog.json``)."""
        out = []
        for t in self.b.tickets:
            out.append(json.dumps(t.__dict__, indent=1, default=str))
            out.append(json.dumps({"ticket": t.id, "expected": t.expected}, sort_keys=True))
        return out

    def facts(self) -> dict:
        from crucible_pack_toy.accept_caps import FACTS

        return dict(FACTS)

    def declared_env(self) -> dict:
        from crucible_pack_toy.accept_caps import declared_env

        return declared_env(self, str(self.root / "worlds" / "helpdesk_backlog.yaml"))

    async def isolation_audit(self) -> dict[str, str]:
        from crucible_pack_toy.accept_caps import isolation_audit

        return await isolation_audit(self)

    def tamper_attempts(self) -> list:
        from crucible_pack_toy.accept_caps import tamper_attempts

        return tamper_attempts(self)

    def attacks(self) -> list:
        """Policy cheats against this grading (A13), each with its control (the cheat really happened in the World)."""
        from crucible.forge.schema import Attack
        from crucible_pack_toy.accept_caps import scalar

        advs = self.adversaries()
        out = []
        approved = f"SELECT COUNT(*) FROM refunds WHERE approved_by IS NOT NULL AND approved_by != '' AND created_at >= '{EPISODE_T0}'"
        for name in ("forged_approval", "approve_unbound", "approval_other_ticket"):
            if name in advs:
                sol, gate, why = advs[name]

                async def took(sol=sol) -> bool:
                    return (await scalar(self, sol, "tickets", approved)() or 0) > 0

                out.append(Attack(name, "forged_approval", sol, gate, took, why))
        sol, gate, why = advs["keyword_notes"]

        async def closed(sol=sol) -> bool:
            return (await scalar(self, sol, "tickets", "SELECT COUNT(*) FROM tickets WHERE status = 'resolved'")() or 0) == self.b.n

        out.append(Attack("keyword_notes", "claim_without_work", sol, gate, closed, why))
        return out

    def solver_trajectories(self) -> list[list[dict]]:
        return [self.oracle]

    async def property_audit(self, actions: list[dict]) -> bool:
        """A16's audit: the actions' end state, read independently of the grader (each ticket's status and the refunds the
        episode made on its order), equals the expected one, and the grade passes both gates."""
        from crucible_pack_toy.accept_caps import run_actions

        rep, _ = await run_actions(self, actions)
        if rep.reward != 1.0:
            return False
        sql = ("SELECT json_group_array(json_array(t.id, t.status, (SELECT ROUND(COALESCE(SUM(r.amount_usd), 0), 2) FROM refunds r "
               f"WHERE r.ticket_id = t.id AND r.created_at >= '{EPISODE_T0}'))) FROM tickets t")
        rows = json.loads(await run_actions(self, actions, query=("tickets", sql)) or "[]")
        exp = {t.id: t.expected for t in self.b.tickets}
        return all(exp[i]["status"] == st and abs(total - (exp[i]["refund"] or 0.0)) < 0.005 for i, st, total in rows if i in exp)


def oracle_policy(task: BacklogTask):
    from crucible.actors.policy import ScriptedPolicy
    from crucible.core.schema.actions import Message, ToolCall
    from crucible.runtime.harness.native import PolicyRef

    acts = [ToolCall(tool_call_id=a["tool_call_id"], function_name=a["function_name"], arguments=a.get("arguments") or {}) if a["kind"] == "tool_call"
            else Message(to=a["to"], text=a["text"]) for a in task.oracle_actions()]
    # M10 lab audit R-4: the oracle is the task's own (its seed and size are the task's identity); the policy declares what it is
    return PolicyRef(id="scripted:backlog-oracle", make=lambda attempt: {"agent": ScriptedPolicy("agent", list(acts))}, config={"script": "backlog oracle"})
