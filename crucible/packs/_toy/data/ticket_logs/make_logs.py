"""Writes tickets.jsonl: helpdesk event logs as a ticketing tool would export them (one row per event).

Columns: case, id, time, actor, role, kind, type, object (what the event is about), writes (objects it changed),
reply_to, text, and score / team / amount where relevant. Hand-written for the toy pack (it keeps the core
domain-agnostic; its M4 labels are on data the builder wrote, a weaker check than the coding pack's). Tickets
T1001-T1012 are the dev split, T1013-T1024 the test split, written in different words.

Row format: "DAY HH:MM | actor | role | kind | type | object | writes | reply_to | extra | text".
"""

import json
from pathlib import Path

TICKETS = {
"T1001": """
2 09:00 | ana | customer | message | ticket_opened | ticket | request | | | I was charged twice for order 5521.
2 09:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
2 09:20 | li | agent | message | agent_reply | resolution | resolution | | | I see two charges of 39.50. I have refunded the duplicate.
2 09:21 | li | agent | action | refund_issued | refund | refund | | amount=39.50 | refund 39.50
2 09:22 | li | agent | state | status_change | status | status | | status=resolved | resolved
3 10:00 | ana | customer | message | csat | ticket | | | score=5 | Quick and clear, thank you.
""",
"T1002": """
2 10:00 | bo | customer | message | ticket_opened | ticket | request | | | My invoice PDF is blank.
2 10:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
2 11:00 | li | agent | message | agent_reply | resolution | resolution | | | Please clear your browser cache and download it again.
2 11:01 | li | agent | state | status_change | status | status | | status=resolved | resolved
2 15:00 | bo | customer | message | customer_reply | ticket | | 3 | | Still blank, I cleared everything twice.
2 15:01 | workflow | system | state | reopened | status | status | | status=open | reopened by customer reply
2 15:30 | li | agent | action | reassigned | assignee | assignee | | team=billing_engineering | escalated to billing engineering
3 09:00 | mo | tier2 | message | agent_reply | resolution | resolution | | | The PDF renderer had a font bug for your locale; we regenerated the invoice.
3 09:05 | mo | tier2 | state | status_change | status | status | | status=resolved | resolved
4 09:00 | bo | customer | message | csat | ticket | | | score=3 | Took two tries.
""",
"T1003": """
2 12:00 | cy | customer | message | ticket_opened | ticket | request | | | Can I get a refund for the annual plan? I bought it yesterday by mistake.
2 12:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
2 12:30 | li | agent | message | agent_reply | resolution | resolution | | | Refunded the full 240.00 to your card.
2 12:31 | li | agent | action | refund_issued | refund | refund | | amount=240.00 | refund 240.00
2 13:10 | sam | supervisor | check | policy_flag | refund | | | rule=refund_over_50_needs_approval | Refund above 50 issued without supervisor approval.
2 13:20 | sam | supervisor | action | refund_reversed | refund | refund | | amount=240.00 | Reversed: annual plans refund pro rata after approval.
2 13:25 | sam | supervisor | message | supervisor_edit | resolution | resolution | 3 | | Correction: annual plans are refunded pro rata; you will receive 230.00 after approval today.
2 14:00 | sam | supervisor | action | refund_issued | refund | refund | | amount=230.00 | refund 230.00 approved
2 14:01 | sam | supervisor | state | status_change | status | status | | status=resolved | resolved
3 09:00 | cy | customer | message | csat | ticket | | | score=2 | Confusing, I got two different answers.
""",
"T1004": """
2 08:00 | dee | customer | message | ticket_opened | ticket | request | | | The app logs me out every few minutes.
2 08:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
2 12:05 | monitor | system | check | sla_breach | assignee | | | sla=first_response_4h | First response SLA (4h) breached.
2 12:30 | kim | agent | message | agent_reply | resolution | resolution | | | Sorry for the wait. Please update to version 7.2, it fixes session expiry.
2 12:31 | kim | agent | state | status_change | status | status | | status=pending | waiting for customer
2 18:00 | dee | customer | message | customer_reply | ticket | | 4 | | Updated and it works now, thanks!
2 18:05 | kim | agent | state | status_change | status | status | | status=resolved | resolved
""",
"T1005": """
3 09:00 | eli | customer | message | ticket_opened | ticket | request | | | Where is my order 7781? It says delivered but nothing arrived.
3 09:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
3 09:30 | kim | agent | message | agent_reply | resolution | resolution | | | The carrier shows it at your building's parcel room.
3 09:31 | kim | agent | state | status_change | status | status | | status=resolved | resolved
3 14:00 | eli | customer | message | customer_reply | ticket | | 3 | | There is no parcel room in my building. This answer didn't help at all.
3 14:01 | workflow | system | state | reopened | status | status | | status=open | reopened by customer reply
3 14:20 | kim | agent | message | internal_note | ticket | | | | Carrier claim needed.
3 14:30 | kim | agent | message | agent_reply | resolution | resolution | | | I filed a carrier claim and shipped a replacement, arriving Thursday.
3 14:31 | kim | agent | state | status_change | status | status | | status=resolved | resolved
4 12:00 | eli | customer | message | csat | ticket | | | score=4 | Fine in the end.
""",
"T1006": """
3 10:00 | fay | customer | message | ticket_opened | ticket | request | | | How do I add a second user to my account?
3 10:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
3 10:10 | li | agent | message | agent_reply | resolution | resolution | | | Go to Settings > Team > Invite and enter their email.
3 10:11 | li | agent | state | status_change | status | status | | status=resolved | resolved
3 10:40 | fay | customer | message | customer_reply | ticket | | 3 | | Does the invited user need their own payment method?
3 10:50 | li | agent | message | agent_reply | resolution | resolution | | | No, they share your plan.
3 10:51 | li | agent | state | status_change | status | status | | status=resolved | resolved
""",
"T1007": """
3 11:00 | gus | customer | message | ticket_opened | ticket | request | | | Your API returns 500 for every request since this morning.
3 11:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
3 11:15 | kim | agent | message | agent_reply | resolution | resolution | | | Please regenerate your API key and retry.
3 11:40 | gus | customer | message | customer_reply | ticket | | 3 | | Same problem with a new key. It's your side.
3 11:45 | kim | agent | action | reassigned | assignee | assignee | | team=platform_oncall | escalated to platform on-call
3 12:30 | ray | tier2 | message | agent_reply | resolution | resolution | | | Confirmed an outage in the EU region; fixed at 12:25. Sorry about that.
3 12:31 | ray | tier2 | state | status_change | status | status | | status=resolved | resolved
4 08:00 | gus | customer | message | csat | ticket | | | score=4 | Good once it reached the right team.
""",
"T1008": """
4 09:00 | hal | customer | message | ticket_opened | ticket | request | | | Please delete my account and all data.
4 09:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
4 09:20 | li | agent | action | account_deleted | account | account | | | account deleted
4 09:21 | li | agent | message | agent_reply | resolution | resolution | | | Done, your account and data are deleted.
4 09:40 | sam | supervisor | check | policy_flag | account | | | rule=identity_check_before_deletion | Deleted without the identity check the policy requires.
4 09:41 | li | agent | state | status_change | status | status | | status=resolved | resolved
""",
"T1009": """
4 10:00 | ivy | customer | message | ticket_opened | ticket | request | | | I need a copy of my January receipt.
4 10:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
4 10:05 | kim | agent | message | agent_reply | resolution | resolution | | | Sent to your email just now.
4 10:06 | kim | agent | state | status_change | status | status | | status=resolved | resolved
4 10:30 | ivy | customer | message | customer_reply | ticket | | 3 | | Got it, perfect.
""",
"T1010": """
4 13:00 | jon | customer | message | ticket_opened | ticket | request | | | I want to change my plan to monthly.
4 13:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
4 13:10 | li | agent | message | agent_reply | resolution | resolution | | | Switched you to monthly starting next cycle.
4 13:11 | li | agent | action | plan_changed | plan | plan | | plan=monthly | plan changed
4 13:30 | sam | supervisor | message | supervisor_approval | plan | | | | Checked, the change and proration are right.
4 13:31 | li | agent | state | status_change | status | status | | status=resolved | resolved
5 09:00 | jon | customer | message | csat | ticket | | | score=5 | Easy.
""",
"T1011": """
5 08:00 | kai | customer | message | ticket_opened | ticket | request | | | Charged in USD instead of EUR.
5 08:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
5 08:30 | kim | agent | message | agent_reply | resolution | resolution | | | Your billing country is US, so prices are in USD.
5 08:31 | kim | agent | state | status_change | status | status | | status=resolved | resolved
5 11:00 | kai | customer | message | customer_reply | ticket | | 3 | | That's wrong, my billing address is in Berlin.
5 11:01 | workflow | system | state | reopened | status | status | | status=open | reopened by customer reply
5 16:05 | monitor | system | check | sla_breach | assignee | | | sla=reopen_response_4h | Reopened ticket waited more than 4h.
""",
"T1012": """
5 09:00 | lu | customer | message | ticket_opened | ticket | request | | | Is there a student discount?
5 09:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
5 09:20 | li | agent | message | agent_reply | resolution | resolution | | | Yes, 30% with a valid student email.
5 09:21 | li | agent | state | status_change | status | status | | status=resolved | resolved
6 09:00 | lu | customer | message | csat | ticket | | | score=1 | The discount code never worked.
""",
"T1013": """
6 08:00 | mia | customer | message | ticket_opened | ticket | request | | | Two subscriptions show up on my statement for the same month.
6 08:02 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
6 08:40 | oz | agent | message | agent_reply | resolution | resolution | | | One was a pending authorization; it has dropped off now, nothing to refund.
6 08:41 | oz | agent | state | status_change | status | status | | status=resolved | resolved
6 17:00 | mia | customer | message | customer_reply | ticket | | 3 | | Nope, both amounts were actually taken from my account.
6 17:01 | workflow | system | state | reopened | status | status | | status=open | reopened by customer reply
6 17:30 | oz | agent | action | refund_issued | refund | refund | | amount=19.99 | refund 19.99
6 17:31 | oz | agent | message | agent_reply | resolution | resolution | | | You're right, refunded the duplicate 19.99.
6 17:32 | oz | agent | state | status_change | status | status | | status=resolved | resolved
7 09:00 | mia | customer | message | csat | ticket | | | score=3 | Had to push back first.
""",
"T1014": """
6 09:00 | ned | customer | message | ticket_opened | ticket | request | | | Export to CSV hangs at 99%.
6 09:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
6 09:30 | pat | agent | action | reassigned | assignee | assignee | | team=data_engineering | handing over to data engineering
6 13:00 | vic | tier2 | message | agent_reply | resolution | resolution | | | Exports over 1M rows timed out; we raised the limit, try again please.
6 13:01 | vic | tier2 | state | status_change | status | status | | status=resolved | resolved
6 14:00 | ned | customer | message | customer_reply | ticket | | 4 | | Works, many thanks.
""",
"T1015": """
6 10:00 | ola | customer | message | ticket_opened | ticket | request | | | Refund my last payment please, I cancelled before renewal.
6 10:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
6 10:20 | pat | agent | action | refund_issued | refund | refund | | amount=89.00 | refund 89.00
6 10:21 | pat | agent | message | agent_reply | resolution | resolution | | | Refund of 89.00 is on its way.
6 11:00 | ana2 | supervisor | check | policy_flag | refund | | | rule=refund_over_50_needs_approval | 89.00 refund lacks approval.
6 11:05 | ana2 | supervisor | message | supervisor_approval | refund | | | | Approved after the fact: cancellation was on time.
6 11:06 | pat | agent | state | status_change | status | status | | status=resolved | resolved
7 10:00 | ola | customer | message | csat | ticket | | | score=5 | Smooth.
""",
"T1016": """
6 11:00 | pia | customer | message | ticket_opened | ticket | request | | | Password reset email never arrives.
6 11:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
6 11:10 | oz | agent | message | agent_reply | resolution | resolution | | | Check your spam folder, it is often filtered there.
6 11:11 | oz | agent | state | status_change | status | status | | status=resolved | resolved
6 11:40 | pia | customer | message | customer_reply | ticket | | 3 | | Not in spam either, and I checked twice.
6 11:41 | workflow | system | state | reopened | status | status | | status=open | reopened by customer reply
6 11:50 | rae | supervisor | message | supervisor_edit | resolution | resolution | 3 | | Update: our mail provider is blocking your domain; we whitelisted it and resent the reset link.
6 11:51 | rae | supervisor | state | status_change | status | status | | status=resolved | resolved
6 12:30 | pia | customer | message | customer_reply | ticket | | 7 | | Received it, I'm back in. Cheers.
""",
"T1017": """
7 08:00 | quin | customer | message | ticket_opened | ticket | request | | | Can you tell me which data centers you use?
7 08:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
7 08:30 | pat | agent | message | agent_reply | resolution | resolution | | | Frankfurt and Dublin for EU customers.
7 08:31 | pat | agent | state | status_change | status | status | | status=resolved | resolved
7 09:00 | quin | customer | message | customer_reply | ticket | | 3 | | And for backups?
7 09:10 | pat | agent | message | agent_reply | resolution | resolution | | | Backups stay in the same region.
7 09:11 | pat | agent | state | status_change | status | status | | status=resolved | resolved
""",
"T1018": """
7 09:00 | rui | customer | message | ticket_opened | ticket | request | | | I was billed after closing my account.
7 09:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
7 13:20 | monitor | system | check | sla_breach | assignee | | | sla=first_response_4h | First response SLA (4h) breached.
7 13:40 | oz | agent | action | refund_issued | refund | refund | | amount=12.00 | refund 12.00
7 13:41 | oz | agent | message | agent_reply | resolution | resolution | | | Apologies, the closure didn't stop billing; refunded 12.00.
7 13:42 | oz | agent | state | status_change | status | status | | status=resolved | resolved
8 09:00 | rui | customer | message | csat | ticket | | | score=2 | Slow and it should never have happened.
""",
"T1019": """
7 10:00 | sol | customer | message | ticket_opened | ticket | request | | | Your mobile app crashes when I open the camera scanner.
7 10:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
7 10:20 | pat | agent | message | agent_reply | resolution | resolution | | | Reinstall the app, that clears the camera cache.
7 10:21 | pat | agent | state | status_change | status | status | | status=resolved | resolved
7 12:00 | sol | customer | message | customer_reply | ticket | | 3 | | Reinstalled, it still crashes on the scanner.
7 12:01 | workflow | system | state | reopened | status | status | | status=open | reopened by customer reply
7 12:10 | pat | agent | action | reassigned | assignee | assignee | | team=mobile_engineering | escalated to mobile engineering
""",
"T1020": """
8 08:00 | tia | customer | message | ticket_opened | ticket | request | | | Requesting VAT invoice for company records.
8 08:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
8 08:20 | oz | agent | message | agent_reply | resolution | resolution | | | Attached your VAT invoice with your company's VAT number.
8 08:21 | oz | agent | state | status_change | status | status | | status=resolved | resolved
9 09:00 | tia | customer | message | csat | ticket | | | score=4 | Thanks.
""",
"T1021": """
8 09:00 | uma | customer | message | ticket_opened | ticket | request | | | Please refund the add-on I never used.
8 09:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
8 09:30 | pat | agent | action | refund_issued | refund | refund | | amount=15.00 | refund 15.00
8 09:31 | pat | agent | message | agent_reply | resolution | resolution | | | Refunded the 15.00 add-on.
8 10:30 | rae | supervisor | action | refund_reversed | refund | refund | | amount=15.00 | Reversed: the add-on was used 40 times this month.
8 10:35 | rae | supervisor | message | supervisor_edit | resolution | resolution | 4 | | Correction: usage logs show the add-on was used, so it is not refundable.
8 10:36 | rae | supervisor | state | status_change | status | status | | status=resolved | resolved
9 10:00 | uma | customer | message | csat | ticket | | | score=1 | Refund taken back without asking me.
""",
"T1022": """
8 11:00 | val | customer | message | ticket_opened | ticket | request | | | My team member can't see shared folders.
8 11:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
8 11:15 | oz | agent | message | agent_reply | resolution | resolution | | | Give them the Editor role under Team settings.
8 11:16 | oz | agent | state | status_change | status | status | | status=pending | waiting for customer
8 11:40 | val | customer | message | customer_reply | ticket | | 3 | | That did the trick, they see everything now.
8 11:45 | oz | agent | state | status_change | status | status | | status=resolved | resolved
""",
"T1023": """
9 08:00 | wes | customer | message | ticket_opened | ticket | request | | | Close my account, I'm moving to another provider.
9 08:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
9 08:10 | pat | agent | action | account_deleted | account | account | | | account deleted
9 08:11 | pat | agent | message | agent_reply | resolution | resolution | | | Account closed as requested.
9 08:30 | ana2 | supervisor | check | policy_flag | account | | | rule=retention_offer_before_closure | Closure processed without the required retention offer.
9 08:31 | pat | agent | state | status_change | status | status | | status=resolved | resolved
""",
"T1024": """
9 09:00 | xin | customer | message | ticket_opened | ticket | request | | | Is two-factor authentication available?
9 09:01 | queue | system | action | assigned | assignee | assignee | | team=tier1 | assigned to tier1
9 09:05 | oz | agent | message | agent_reply | resolution | resolution | | | Yes, enable it under Security > Two-factor.
9 09:06 | oz | agent | state | status_change | status | status | | status=resolved | resolved
9 09:30 | xin | customer | message | customer_reply | ticket | | 3 | | Where do I find the Security page on mobile?
""",
}


def rows():
    out = []
    for case, block in TICKETS.items():
        for i, line in enumerate(x for x in block.strip().splitlines() if x.strip()):
            head, actor, role, kind, typ, obj, writes, reply, extra, text = [p.strip() for p in line.split("|", 9)]
            day, hm = head.split()
            row = {"case": case, "id": f"{case}-{i + 1}", "time": f"2026-03-{int(day):02d}T{hm}:00Z", "actor": actor, "role": role,
                   "actor_kind": "system" if role == "system" else "human", "kind": kind, "type": typ, "text": text,
                   "object": f"ticket:{case}" + ("" if obj == "ticket" else f".{obj}"),
                   "writes": [f"ticket:{case}" + ("" if w == "ticket" else f".{w}") for w in writes.split(",") if w],
                   "reply_to": f"{case}-{reply}" if reply else None}
            for kv in filter(None, extra.split(",")):
                k, v = kv.split("=", 1)
                row[k] = int(v) if v.isdigit() else v
            out.append(row)
    return out


if __name__ == "__main__":
    path = Path(__file__).with_name("tickets.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows()))
    print(f"{len(TICKETS)} tickets, {len(rows())} events -> {path}")
