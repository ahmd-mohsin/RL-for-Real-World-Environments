"""M2 gate, toy half (helpdesk).

(a) Grounded customers: 200 questions (answerable and not) to the record-grounded customers; every reply
    is audited (deterministic citation check + a judge sample). Gate: fabrication <= 1%, all
    non-"no record" replies cited.
(b) Scenario simulation end to end: an LLM agent works the helpdesk while a scenario-driven simulated
    customer (LLM) answers it, each reply audited online by a consistency judge from a different model
    family; budget-capped. Reports inconsistency rate, masking, and the expert end-state checks.
Run: ``python -m crucible_pack_toy.m2_gate out.json [episodes]``.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

from crucible.actors.fidelity import consistency_summary, grounding_audit
from crucible.actors.grounded import GroundedResponder
from crucible.actors.grounding import NO_RECORD
from crucible.actors.llm import LLMClient
from crucible.actors.policy import LLMToolPolicy, run_episode
from crucible.actors.records import RecordStore
from crucible.actors.scripted import RelayActor, Rule
from crucible.actors.simulated import Scenario, SimulatedActor
from crucible.core.store import Store
from crucible.world.world import World
from crucible_pack_toy import INSTRUCTION, PACK_ROOT, helpdesk_spec

QUESTIONS_ANSWERABLE = [
    ("customer_dana", "What happened to the blender?"), ("customer_dana", "Do you prefer a refund or a replacement?"),
    ("customer_dana", "Can you send photos of the crack?"), ("customer_dana", "Do you still have the box and packing slip?"),
    ("customer_lee", "How many times were you charged for order O-5002?"), ("customer_lee", "How much was each charge?"),
    ("customer_lee", "When was the TV supposed to arrive?"), ("customer_lee", "When did the tracking last move?"),
]
QUESTIONS_UNANSWERABLE = [
    ("customer_dana", "What is your favourite colour?"), ("customer_lee", "Which bank issued your card?"),
    ("customer_dana", "Did you buy anything else last year?"), ("customer_lee", "What is your home address?"),
]


async def grounded_customers(n: int = 200, judge_budget: float = 0.01) -> dict:
    store = RecordStore.from_jsonl(PACK_ROOT / "data" / "customers.jsonl")
    actors = {a: GroundedResponder(a, store, mode="verbatim", roles={a}, min_score=1.0) for a in ("customer_dana", "customer_lee")}
    rnd = random.Random(3)
    qs = [rnd.choice(QUESTIONS_ANSWERABLE) for _ in range(int(n * 0.8))] + [rnd.choice(QUESTIONS_UNANSWERABLE) for _ in range(n - int(n * 0.8))]
    rnd.shuffle(qs)
    replies = []
    for who, q in qs:
        text, cites = await actors[who].answer(q, now="2026-03-02T09:00:00Z")
        replies.append({"who": who, "q": q, "a": text, "cites": cites})
    judge = LLMClient("judge-default", budget_usd=judge_budget)
    judged = [r for r in replies if r["a"] != NO_RECORD][:20]
    a1 = await grounding_audit([(r["a"], r["cites"], "2026-03-02T09:00:00Z") for r in judged], store, judge=judge)
    a2 = await grounding_audit([(r["a"], r["cites"], "2026-03-02T09:00:00Z") for r in replies if r not in judged], store)
    factual = a1.factual_sentences + a2.factual_sentences
    bad = a1.unsupported_sentences + a1.judge_rejected + a2.unsupported_sentences
    unans = {q for _, q in QUESTIONS_UNANSWERABLE}
    return {
        "replies": len(replies),
        "no_record": sum(r["a"] == NO_RECORD for r in replies),
        "unanswerable_correct_no_record": sum(1 for r in replies if r["q"] in unans and r["a"] == NO_RECORD),
        "unanswerable": sum(1 for r in replies if r["q"] in unans),
        "answerable_answered": sum(1 for r in replies if r["q"] not in unans and r["a"] != NO_RECORD),
        "uncited_non_no_record_replies": sum(1 for r in replies if r["a"] != NO_RECORD and not r["cites"]),
        "factual_sentences": factual,
        "unsupported_or_judge_rejected": bad,
        "fabrication_rate": round(bad / factual, 4) if factual else 0.0,
        "judged": a1.judged, "judge_spend_usd": round(judge.spend.usd, 5),
        "examples": (a1.examples + a2.examples)[:5],
    }


LEE = Scenario(
    persona="Lee Park, a regular customer of the shop, polite but busy.",
    reason_for_contact="You were charged twice (two charges of 39.50 USD) for order O-5002 and want one charge refunded.",
    known_info="Order O-5002. Both charges of 39.50 USD appeared on your card statement on the same day. You placed the order once.",
    unknown_info="Your card's bank, any internal ticket numbers, the shop's refund policy.",
    task_instructions="Answer the support agent's questions. Be satisfied once they confirm the 39.50 USD duplicate charge is refunded.",
)


async def scenario_episodes(episodes: int = 2, budget: float = 0.06) -> dict:
    agent_llm = LLMClient("policy-default", budget_usd=budget)
    sim_llm = LLMClient("sim-default", budget_usd=budget / 2)
    judge = LLMClient("judge-default", budget_usd=budget / 3)
    results = []
    for ep in range(episodes):
        with tempfile.TemporaryDirectory() as d:
            w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION + " Start with ticket 2."}, store=Store(Path(d) / "s"), workdir=Path(d) / "w", pack_root=PACK_ROOT)
            lee = SimulatedActor("customer_lee", LEE, sim_llm, judge=judge, reply_latency="10m")
            store = RecordStore.from_jsonl(PACK_ROOT / "data" / "customers.jsonl")
            w.attach(lee)
            w.attach(GroundedResponder("customer_dana", store, mode="verbatim", roles={"customer_dana"}, min_score=0.5))
            w.attach(RelayActor("supervisor", [Rule(r"approv\w*.*?(\d+(?:\.\d+)?)\s*usd", reply="Approved refund of {0} USD. - Sam Lee")],
                                default="Please state the ticket, order and amount."))
            await w.start()
            try:
                agent = LLMToolPolicy("agent", agent_llm)
                res = await run_episode(w, {"agent": agent}, max_steps=int(14 + ep))
                v = w.verifier_view()
                refund = (await v.query("tickets", {"op": "sql", "args": {"sql": "SELECT order_id, amount_usd FROM refunds WHERE ticket_id = 2"}})).value["rows"]
                status = (await v.query("tickets", {"op": "scalar", "args": {"sql": "SELECT status FROM tickets WHERE id = 2"}})).value
                talked = sum(1 for e in res.trace.events if e.actor == "customer_lee" and e.type == "message")
                results.append({"episode": ep, "steps": res.steps, "done": res.done, "masked": res.masked, "failures": res.failures,
                                "customer_messages": talked, "ticket2_refund": refund, "ticket2_status": status,
                                "consistency": consistency_summary(lee.audit)})
            finally:
                await w.close()
    return {"episodes": results, "spend_usd": {"agent": round(agent_llm.spend.usd, 5), "sim": round(sim_llm.spend.usd, 5), "judge": round(judge.spend.usd, 5)},
            "models": {"agent": agent_llm.model, "sim": sim_llm.model, "judge": judge.model}}


async def main_async(out: Path, episodes: int) -> dict:
    res = {"grounded_customers": await grounded_customers(), "scenario_episodes": await scenario_episodes(episodes)}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("m2_toy_gate.json")
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(json.dumps(asyncio.run(main_async(out, episodes)), indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
