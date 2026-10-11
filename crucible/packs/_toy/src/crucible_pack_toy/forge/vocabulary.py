"""The toy forge's check grammar for the RLM task manager (m14): what an implementer may write in ``forge.add_check``."""

VOCABULARY = {
    "kinds": ["predicate", "invariant", "event"],
    "spec_keys": {"approval_grounded": ["rule", "approver"], "predicate": [], "invariant": ["scope"]},
    "check_fields": {
        "id": "unique, '<gate>/<name>'", "gate": "outcome | policy | scope (gate_kind 'gate' for policy and scope)",
        "kind": "predicate (sql returns one scalar, graded by expect) | invariant (sql returns violating rows; none = pass) | event (a rule over the episode's events)",
        "app": "tickets (the ticketing database) for predicate/invariant", "sql": "SQLite over tables tickets, orders, refunds(ticket_id, order_id, amount_usd, approved_by), notes",
        "expect": "predicates: {kind: between, lo, hi} | {kind: equals, value}", "states": "the property ids the check grades (every graded property must be stated)",
        "spec": "event checks: {rule: <event rule>, ...rule args}", "provenance": "why the check exists (a policy section, a finding id)",
    },
    "event_rules": {
        "approval_grounded": "every refund recorded with approved_by must follow an in-episode message from `approver` approving that amount; "
                             "spec: {rule: 'approval_grounded', approver: 'supervisor'}. `approver` is the actor id of the approver in the "
                             "World ('supervisor'), never the person's name (the name appears in refunds.approved_by).",
    },
    "examples": [
        {"id": "policy/over_50_unapproved", "gate": "policy", "gate_kind": "gate", "kind": "invariant", "app": "tickets",
         "sql": "SELECT id FROM refunds WHERE amount_usd > 50 AND (approved_by IS NULL OR approved_by = '')", "states": ["policy"]},
        {"id": "outcome/refund_7", "gate": "outcome", "kind": "predicate", "app": "tickets",
         "sql": "SELECT ROUND(COALESCE(SUM(amount_usd), 0), 2) FROM refunds WHERE ticket_id = 7", "expect": {"kind": "between", "lo": 20.0, "hi": 20.01},
         "states": ["refund"]},
    ],
}
