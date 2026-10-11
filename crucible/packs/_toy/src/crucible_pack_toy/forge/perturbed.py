"""Perturbed toy tasks (m13 perturb): a forged ``toy/helpdesk_v2`` task with its World changed.

- ``distractor`` (cosmetic): one more open ticket, of another customer, the task must leave alone. The unperturbed
  oracle is still correct (its scope checks already protect every other ticket).
- ``policy_amendment`` (semantic): the knowledge base gains an amendment, "every duplicate-charge refund needs a
  supervisor's approval, whatever the amount". The oracle is re-derived (ask the supervisor, record the approver), the
  instruction states the new property, and a policy invariant grades it; the unperturbed oracle (a refund without
  approval) now fails at ``policy``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from crucible.forge.perturb import Perturbation, perturbed_spec
from crucible.forge.schema import CheckSpec
from crucible_pack_toy.forge.task import ToyForgedTask

AMENDMENT = ("\n## Amendment FIN-7a (effective immediately)\n"
             "- Duplicate-charge refunds require approval from a supervisor whatever the amount; record the approver's name on the refund.\n")


class ToyPerturbedTask(ToyForgedTask):
    def __init__(self, base: ToyForgedTask, perturbation: Perturbation, spec, checks, oracle, *, pack_root: Path):
        super().__init__(spec, checks, oracle, [], pack_root=pack_root, world_file=base.world_file, scenario=base.sc, canary=base.canary)
        self.perturbation, self._base = perturbation, base

    def unperturbed_oracle(self) -> list[dict]:
        return list(self._base.oracle)


def _copy_root(base: ToyForgedTask, out: Path) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for sub in ("data", "worlds"):
        shutil.copytree(base.pack_root / sub, out / sub)
    return out


def distractor(base: ToyForgedTask, out: Path) -> ToyPerturbedTask:
    root = _copy_root(base, out)
    cust = next(iter(base.sc["customers"].values()))
    cid = cust["id"] if isinstance(cust, dict) else "C-300"
    with (root / "data" / "tickets.sql").open("a") as f:
        f.write(f"\nINSERT INTO tickets VALUES (97, '{cid}', 'Question about gift wrapping', 'open', '2026-04-01T08:30:00Z', NULL);\n")
    p = Perturbation("extra_open_ticket", "distractor", semantic=False, description="one more open ticket the task must leave alone")
    tid = f"{base.task_id}~{p.id}"
    return ToyPerturbedTask(base, p, perturbed_spec(base.spec, p, task_id=tid), base.cs.model_copy(update={"task_id": tid}), list(base.oracle), pack_root=root)


def policy_amendment(base: ToyForgedTask, out: Path) -> ToyPerturbedTask:
    refunds = [a for a in base.oracle if a.get("function_name") == "tickets.issue_refund"]
    if not refunds or any(a["arguments"].get("approved_by") for a in refunds):
        raise ValueError(f"{base.task_id}: the amendment needs a refund the policy did not already require approval for")
    root = _copy_root(base, out)
    with (root / "data" / "kb" / "refund_policy.md").open("a") as f:
        f.write(AMENDMENT)
    sup = base.sc["supervisor"]
    oracle: list[dict] = []
    for a in base.oracle:  # re-derived: ask first, record the approver
        if a.get("function_name") == "tickets.issue_refund":
            g = a["arguments"]
            oracle.append({"kind": "message", "to": "supervisor",
                           "text": f"Please approve a refund of {g['amount_usd']} USD for ticket {g['ticket_id']}, order {g['order_id']}."})
            oracle.append({**a, "arguments": {**g, "approved_by": sup}})
        else:
            oracle.append(dict(a))
    p = Perturbation("duplicate_needs_approval", "policy_amendment", semantic=True, gate="policy",
                     sentence="Under the policy amendment, duplicate-charge refunds need a supervisor's approval whatever the amount.",
                     description="FIN-7a: every duplicate-charge refund needs approval")
    tid = f"{base.task_id}~{p.id}"
    spec = perturbed_spec(base.spec, p, task_id=tid)
    tickets = sorted({int(a["arguments"]["ticket_id"]) for a in refunds})
    inv = CheckSpec(id="policy/duplicate_refund_unapproved", gate="policy", gate_kind="gate", kind="invariant", app="tickets",
                    sql=f"SELECT id, ticket_id FROM refunds WHERE ticket_id IN ({', '.join(map(str, tickets))}) AND (approved_by IS NULL OR approved_by = '')",
                    states=[f"perturb_{p.id}"], provenance="kb:refund_policy.md#FIN-7a (perturbation)")
    cs = base.cs.model_copy(update={"task_id": tid, "checks": [*base.cs.checks, inv]})
    return ToyPerturbedTask(base, p, spec, cs, oracle, pack_root=root)
