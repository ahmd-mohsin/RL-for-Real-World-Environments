"""Crucible toy pack: a small helpdesk world (tickets in SQLite, a knowledge base, a mailbox, grounded
customers, a supervisor bot, events and permissions). Exists to keep the core domain-agnostic: every
module's gate runs on it, on CPU, in CI, with no paid calls."""

from __future__ import annotations

from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = (
    "You are the support agent of an online shop. Work through the open tickets in the ticketing system. "
    "Follow the refund policy in the knowledge base exactly. Ask a customer (customer_dana, customer_lee) when you need "
    "facts from them, and ask the supervisor to approve refunds above 50 USD before issuing them. Add a note to every "
    "ticket you work on and set its final status."
)


def register(registry) -> None:
    registry.register("pack", "toy", "crucible_pack_toy:PACK_ROOT", source="crucible-pack-toy")
    registry.register("signal_mapper", "toy/support_ticket", "crucible_pack_toy.signals:ToyTicketSignalMapper", source="crucible-pack-toy")


def helpdesk_spec():
    from crucible.core.schema.world import WorldSpec

    return WorldSpec.load(PACK_ROOT / "worlds" / "helpdesk.yaml")


def responders():
    """The helpdesk's non-policy actors: grounded customers (verbatim, no LLM) and a rule-based supervisor."""
    from crucible.actors.grounded import GroundedResponder
    from crucible.actors.records import RecordStore
    from crucible.actors.scripted import RelayActor, Rule

    store = RecordStore.from_jsonl(PACK_ROOT / "data" / "customers.jsonl")
    return [
        GroundedResponder("customer_dana", store, who="Dana Ortiz, a customer", mode="verbatim", roles={"customer_dana"}, reply_latency="10m", min_score=0.5),
        GroundedResponder("customer_lee", store, who="Lee Park, a customer", mode="verbatim", roles={"customer_lee"}, reply_latency="10m", min_score=0.5),
        RelayActor("supervisor", [
            Rule(r"approv\w*.*?(\d+(?:\.\d+)?)\s*usd.*?(?:transit|o-5003)", reply="Rejected: orders in transit are not refundable; escalate for a carrier trace. - Sam Lee"),
            Rule(r"approv\w*.*?(\d+(?:\.\d+)?)\s*usd", reply="Approved refund of {0} USD. - Sam Lee"),
        ], default="Please state the ticket, the order and the refund amount you want approved.", reply_latency="15m"),
    ]
