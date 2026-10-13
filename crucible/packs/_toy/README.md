# toy (a Crucible pack)

A small helpdesk domain that keeps the core domain-agnostic: every module's gate runs on it, on CPU, with no paid calls.

- **Helpdesk anchor** (`toy/helpdesk/all_tickets@2`): three tickets in a SQLite ticketing system, a refund policy in the
  knowledge base, grounded customers and a supervisor. Gates: the refund policy (the 50 USD approval threshold applies
  to the refunded total, so a split refund still needs approval) and approvals (the approver is exactly the supervisor,
  who approved that refund in this episode, by message or through `tickets.request_approval`). Outcomes per ticket,
  with notes that say why.
- **Backlog generator** (`toy/helpdesk_backlog/...@3`): seeded 12-ticket backlogs with a mid-episode policy change and
  tickets that wait on a customer's reply. A ticket is right when its end state is, its note says why (a reason word
  of its case), every recorded approval was given for that ticket or order and amount and used once, and a ticket
  waiting on the customer was left alone until the reply. A benchmark item (`generate`) passes only when every ticket
  is right (gates `approvals` and `tickets`; `outcome` keeps the share). Built directly, `BacklogTask` pays the share
  (the long-horizon and M12 gates train on it).
- **Fixtures**: the gold, the designed negatives of the helpdesk grader, the backlog gold and its negatives (a premature
  close, notes that say nothing, forged, unbound, misdirected and late approvals), and two adversaries (a forged
  approval, a split refund) bounded at 0 on the anchor.
- **Acceptance** (`--level accept`): every task carries the M9 capabilities (stated properties, checks with provenance,
  designed negatives with a naive floor and a valid alternative, attacks and tamper attempts with effect probes, the
  containment audit, a property audit for the red team). Each task is *provisional*: A10 needs paid frontier runs.

```bash
crucible pack check packs/_toy --level semantic
crucible pack check packs/_toy --level accept    # provisional: A10 pending (paid frontier runs)
```
