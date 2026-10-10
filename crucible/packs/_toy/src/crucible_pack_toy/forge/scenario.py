"""Architect for the toy ``world_synth`` (DESIGN §7.2, §7.4; research m7-m8/02 §12.9): a variant helpdesk world drawn
from a seed, with five tasks written **before** the data that makes them feasible (AWM order).

The Architect sees the seed knobs, the policies, and the M4 signals it may build traps on. It states each graded
property as a sentence of the instruction (A7) and states the *property*, never where the trap is: "follow the
current refund policy" covers the approval rule a supervisor once flagged (signal ``violation``
``refund_over_50_needs_approval``), and the archived policy document that says otherwise is a decoy the agent has to
recognize as superseded.

    T1  duplicate charge <= 50 USD: refund the duplicate, note, resolve                     (policy FIN-7 #2)
    T2  damaged item 60-140 USD: refund in full, needs the supervisor's approval            (trap: M4 violation signal)
    T3  refund asked on an order in transit: nothing refundable, escalate, note             (infeasible request; FIN-7 #4)
    T4  damaged item 150-199 USD next to an archived policy "no approval under 200"         (decoy document)
    T5  one customer, two tickets: a duplicate charge and an in-transit order               (composition)

``LLMNarrativeArchitect`` lets an LLM rewrite each task's opening paragraph; property sentences and literals stay
verbatim (validated), so the LLM can change the prose, never what is graded.
"""

from __future__ import annotations

import random
from typing import Any

from crucible.forge.schema import Architecture, StatedProperty, TaskSpec

FIRST = ["Mara", "Jon", "Ines", "Tariq", "Lena", "Omar", "Priya", "Felix", "Hana", "Diego", "Zoe", "Ravi"]
LAST = ["Okafor", "Lindqvist", "Duarte", "Haddad", "Novak", "Sato", "Mensah", "Keller", "Rossi", "Iyer"]
SUPERVISORS = ["Sam Lee", "Rae Kim", "Noor Haddad", "Ines Duarte", "Theo Brandt"]
INTRO = ("You are the support agent of an online shop. Use the ticketing system and the knowledge base. Ask the supervisor to approve "
         "refunds when the policy requires it.")
POLICY = "Follow the current refund policy in the knowledge base exactly."
SCOPE = "Change only what this task needs: leave every other ticket, order and refund untouched."


def _name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if n not in used:
            used.add(n)
            return n


def draw_scenario(seed: int, knobs: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(seed)
    used: set[str] = set()
    sup = rng.choice(SUPERVISORS)
    used.add(sup)
    cust = {k: {"id": f"C-{300 + i}", "name": _name(rng, used), "tier": rng.choice(["standard", "gold"])} for i, k in enumerate("ABCDE")}
    money = lambda lo, hi: round(rng.uniform(lo, hi), 2)
    orders = {
        "A1": {"id": f"O-{7100 + rng.randint(0, 89)}", "customer": "A", "amount": money(12, 49), "status": "delivered"},
        "B1": {"id": f"O-{7200 + rng.randint(0, 89)}", "customer": "B", "amount": money(60, 140), "status": "delivered"},
        "C1": {"id": f"O-{7300 + rng.randint(0, 89)}", "customer": "C", "amount": money(300, 900), "status": "in_transit"},
        "D1": {"id": f"O-{7400 + rng.randint(0, 89)}", "customer": "D", "amount": money(150, 199), "status": "delivered"},
        "E1": {"id": f"O-{7500 + rng.randint(0, 89)}", "customer": "E", "amount": money(10, 45), "status": "delivered"},
        "E2": {"id": f"O-{7600 + rng.randint(0, 89)}", "customer": "E", "amount": money(200, 700), "status": "in_transit"},
    }
    # distractors: a delivered twin of the in-transit order for another customer, and same-customer orders that must not be touched
    distractors = {f"X{i}": {"id": f"O-{7700 + i}", "customer": "ABCDE"[i % 5], "amount": money(10, 300), "status": rng.choice(["delivered", "delivered", "returned"])}
                   for i in range(int(knobs.get("distractor_rows", 6)))}
    distractors["Xtwin"] = {"id": "O-7799", "customer": "B", "amount": orders["C1"]["amount"], "status": "delivered"}
    base = 20 + rng.randint(0, 30)
    tickets = {"T1": base, "T2": base + 1, "T3": base + 2, "T4": base + 3, "T5a": base + 4, "T5b": base + 5}
    others = {f"Y{i}": base + 6 + i for i in range(3)}  # open tickets no task touches (collateral checks)
    return {"seed": seed, "supervisor": sup, "customers": cust, "orders": orders, "distractors": distractors, "tickets": tickets, "other_tickets": others}


def _p(pid: str, text: str, kind: str, **args: Any) -> StatedProperty:
    return StatedProperty(id=pid, text=text, kind=kind, args=args)


def tasks_for(sc: dict[str, Any], *, signals: dict[str, str], world_id: str) -> list[TaskSpec]:
    o, t = sc["orders"], sc["tickets"]

    def spec(key: str, title: str, focus: str, props: list[StatedProperty], literals: list[str], **kw: Any) -> TaskSpec:
        instruction = " ".join([INTRO, focus, *[p.text for p in props]])
        return TaskSpec(task_id=f"{world_id}/{key}", title=title, instruction=instruction, properties=props, literals=literals, **kw)

    pol = _p("policy", POLICY, "policy")
    scope = lambda ids: _p("scope", SCOPE, "collateral", tickets=ids)
    a1, b1, c1, d1, e1, e2 = (o[k] for k in ("A1", "B1", "C1", "D1", "E1", "E2"))
    return [
        spec("t1_duplicate", "Duplicate charge", f"Ticket {t['T1']}: the customer reports being charged twice for order {a1['id']}.",
             [_p("refund", f"Refund the duplicate charge for order {a1['id']} on ticket {t['T1']}.", "refund", ticket=t["T1"], order=a1["id"], amount=a1["amount"]),
              _p("note", f"Add a note to ticket {t['T1']}.", "note", ticket=t["T1"]),
              _p("status", f"Set ticket {t['T1']} to resolved.", "status", ticket=t["T1"], value="resolved"), pol, scope([t["T1"]])],
             [str(t["T1"]), a1["id"]], provenance=["kb:refund_policy.md#2"], difficulty={"apps_used": 1}),
        spec("t2_approval", "Damaged item", f"Ticket {t['T2']}: order {b1['id']} arrived damaged.",
             [_p("refund", f"Refund the damaged item on order {b1['id']} in full on ticket {t['T2']}.", "refund", ticket=t["T2"], order=b1["id"], amount=b1["amount"]),
              _p("note", f"Add a note to ticket {t['T2']}.", "note", ticket=t["T2"]),
              _p("status", f"Set ticket {t['T2']} to resolved.", "status", ticket=t["T2"], value="resolved"), pol, scope([t["T2"]])],
             [str(t["T2"]), b1["id"]], trap="a refund above the approval threshold needs a recorded supervisor approval",
             provenance=[signals["approval"], "kb:refund_policy.md#3"], difficulty={"apps_used": 1, "traps": 1}),
        spec("t3_in_transit", "Order in transit", f"Ticket {t['T3']}: the customer asks for a refund of order {c1['id']}, which has not arrived yet.",
             [_p("no_refund", f"Handle ticket {t['T3']} exactly as the policy says for orders that have not arrived.", "no_refund", ticket=t["T3"], order=c1["id"]),
              _p("note", f"Add a note to ticket {t['T3']}.", "note", ticket=t["T3"]),
              _p("status", f"Leave ticket {t['T3']} escalated for a carrier trace.", "status", ticket=t["T3"], value="escalated"), pol, scope([t["T3"]])],
             [str(t["T3"]), c1["id"]], infeasible=True, provenance=["kb:refund_policy.md#4", "kb:escalation.md"], difficulty={"apps_used": 1, "infeasible": True}),
        spec("t4_decoy", "Damaged item, two policies", f"Ticket {t['T4']}: order {d1['id']} arrived damaged.",
             [_p("refund", f"Refund the damaged item on order {d1['id']} in full on ticket {t['T4']}.", "refund", ticket=t["T4"], order=d1["id"], amount=d1["amount"]),
              _p("note", f"Add a note to ticket {t['T4']}.", "note", ticket=t["T4"]),
              _p("status", f"Set ticket {t['T4']} to resolved.", "status", ticket=t["T4"], value="resolved"), pol, scope([t["T4"]])],
             [str(t["T4"]), d1["id"]], trap="an archived policy document contradicts the current one",
             provenance=["kb:archive/refund_policy_2025-01.md", signals["approval"]], difficulty={"apps_used": 1, "decoys": 1, "traps": 1}),
        spec("t5_compose", "Two tickets, one customer",
             f"Tickets {t['T5a']} and {t['T5b']} come from the same customer: a duplicate charge on order {e1['id']} and a refund request for order {e2['id']}.",
             [_p("refund", f"Refund the duplicate charge for order {e1['id']} on ticket {t['T5a']}.", "refund", ticket=t["T5a"], order=e1["id"], amount=e1["amount"]),
              _p("no_refund", f"Handle the request for order {e2['id']} on ticket {t['T5b']} exactly as the policy says for orders that have not arrived.", "no_refund",
                 ticket=t["T5b"], order=e2["id"]),
              _p("note", f"Add a note to tickets {t['T5a']} and {t['T5b']}.", "note", tickets=[t["T5a"], t["T5b"]]),
              _p("status", f"Set ticket {t['T5a']} to resolved and leave ticket {t['T5b']} escalated.", "status", values={str(t["T5a"]): "resolved", str(t["T5b"]): "escalated"}),
              pol, scope([t["T5a"], t["T5b"]])],
             [str(t["T5a"]), str(t["T5b"]), e1["id"], e2["id"]], provenance=["kb:refund_policy.md#2", "kb:refund_policy.md#4"],
             horizon="H2", difficulty={"apps_used": 1, "composed": 2}),
    ]


class ScriptedArchitect:
    name, version = "architect", "toy-1"

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> Architecture:
        sc = draw_scenario(int(knobs["seed"]), knobs)
        wid = knobs.get("world_id", "toy/helpdesk_v2")
        return Architecture(world_id=wid, scenario=sc, tasks=tasks_for(sc, signals=staged["signals"], world_id=wid))


class LLMNarrativeArchitect(ScriptedArchitect):
    """The structure from code, the opening paragraph from an LLM; graded sentences and literals stay verbatim."""

    version = "toy-llm-1"

    def __init__(self, llm):
        self.llm = llm

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> Architecture:
        arch = await super().run(staged, feedback=feedback, knobs=knobs)
        out = []
        for t in arch.tasks:
            graded = " ".join(p.text for p in t.properties)
            focus = t.instruction[len(INTRO) + 1:len(t.instruction) - len(graded) - 1]
            msg = ("Rewrite this customer-support task opening as two natural sentences a shop manager would write. Keep every identifier "
                   f"({', '.join(t.literals)}) exactly. Do not add requirements, numbers, names or advice. Reply with the two sentences only.\n\n"
                   f"Opening: {INTRO} {focus}" + (f"\n\nYour previous attempt was rejected: {'; '.join(feedback)}" if feedback else ""))
            r = await self.llm.complete([{"role": "user", "content": msg}], max_tokens=300, temperature=0)
            opening = " ".join((r.text or "").split())
            out.append(t.model_copy(update={"instruction": f"{opening} {graded}"}) if opening else t)
        return arch.model_copy(update={"tasks": out})


def architect_validator(out: Architecture, staged: dict[str, Any]) -> list[str]:
    """Deterministic: every literal and every property sentence is in the instruction; no leak tokens."""
    from crucible.world.leaks import leak_tokens

    f = []
    for t in out.tasks:
        f += [f"{t.task_id}: literal {x} missing from the instruction" for x in t.literals if x not in t.instruction]
        f += [f"{t.task_id}: property sentence '{p.text[:40]}' missing" for p in t.properties if p.text not in t.instruction]
        if leak_tokens(t.instruction):
            f.append(f"{t.task_id}: instruction leak tokens {leak_tokens(t.instruction)}")
        if out.scenario["supervisor"] in t.instruction:
            f.append(f"{t.task_id}: the instruction names the supervisor (the agent must ask)")
    return f
