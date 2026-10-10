"""Toy pack M5/M6: the helpdesk workflow from exported ticket logs (the horizontal path; core stays domain-agnostic).

- **traces**: ``file_drop`` of ``data/ticket_logs/tickets.jsonl`` (dev T1001-T1012, test T1013-T1024);
- **state**: ``workflow.logstate`` over ``ticket:<case>.*`` objects, plus two toy effects: a supervisor approval
  marks ``<object>.approved``, a refund reversal sets the refund to ``"reversed"``;
- **outcome** (the M4 toy guideline): CSAT ≤ 2, or reopened without a later resolution -> bad; resolved with CSAT ≥ 4
  or no CSAT -> good; CSAT 3 -> unknown;
- **replay** (S1): rebuild the state at each recorded check event and reproduce it from state alone (the refund-approval
  rule, the retention-offer rule, the first-response SLA) plus the final status;
- **milestones**: mined from the *end states of good dev tickets* (objects present in at least half of them, templated
  ``ticket:{case}.<field>``; a consistent final value becomes a value match), filtered (non-trivial, monotone);
- **stages** induced on dev tickets; the **tracker** runs on test tickets and is scored against
  ``stage_labels_test.json`` (written before this code) with Cohen's κ;
- **reward** (state only): gate = no unapproved refund over 50 and no closure without a retention offer; outcome =
  resolved with an answer and no reversed refund; Φ over the milestones; R1-R6 on that reward.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from crucible.connectors.file_drop import FieldMap, FileDropConnector
from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck, Provenance
from crucible.core.schema.state import Query
from crucible.core.schema.trace import Trace
from crucible.credit.compose import compose
from crucible.credit.operators.potential import potential, trajectory
from crucible.credit.properties import check_all, r1_oracle, r2_noop, r3_negatives, r4_oscillation, r5_stubbing, r6_state_only
from crucible.credit.validity import auc
from crucible.workflow.graph import WorkflowGraph
from crucible.workflow.logstate import LogStateView, states
from crucible.workflow.milestones.evaluate import evaluate, evaluate_series
from crucible.workflow.milestones.filters import screen
from crucible.workflow.milestones.match import match
from crucible.workflow.stages import induce, name_stages
from crucible.workflow.steps import segment
from crucible.workflow.tracker import WorkflowTracker
from crucible_pack_toy import PACK_ROOT

FAMILY = "toy/helpdesk_ticket"
LOGS = PACK_ROOT / "data" / "ticket_logs" / "tickets.jsonl"
HERE = Path(__file__).resolve().parent
DEV = {f"T10{i:02d}" for i in range(1, 13)}
TEST = {f"T10{i:02d}" for i in range(13, 25)}
FIELDS = FieldMap(family=FAMILY, extra=["status", "team", "score", "amount", "rule", "sla", "plan"])


def traces() -> dict[str, Trace]:
    c = FileDropConnector(FIELDS)
    return {ref.key: c.normalize(c.fetch(ref)) for ref in c.discover({"glob": str(LOGS)})}


def effects(e: Any) -> dict[str, Any]:
    obj = e.target.object if e.target else None
    if e.type == "supervisor_approval" and obj:
        return {f"{obj}.approved": True}
    if e.type == "refund_reversed" and obj:
        return {obj: "reversed"}
    return {}


def toy_states(t: Trace) -> list[tuple[int, LogStateView]]:
    return states(t, case=t.source.ref, effects=effects)


def outcome(t: Trace) -> str:
    csat = [e.data.get("score") for e in t.events if e.type == "csat"]
    statuses = [(e.seq, e.data.get("status")) for e in t.events if e.data.get("status")]
    last_status = statuses[-1][1] if statuses else None
    reopened = any(e.type == "reopened" for e in t.events)
    score = int(csat[-1]) if csat and csat[-1] is not None else None
    if (score is not None and score <= 2) or (reopened and last_status != "resolved"):
        return "bad"
    if score == 3:
        return "unknown"
    return "good" if last_status == "resolved" else "unknown"


# ---------------------------------------------------------------- replay (S1)
def _hours(a: str, b: str) -> float:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 3600


def replay(t: Trace) -> dict[str, Any]:
    case = t.source.ref
    st = dict(toy_states(t))
    checks = []
    opened = t.events[0].time
    for e in t.events:
        obj = st[e.seq].objects
        if e.type == "policy_flag":
            rule = e.data.get("rule")
            if rule == "refund_over_50_needs_approval":
                amt = obj.get(f"ticket:{case}.refund")
                try:
                    over = float(amt) > 50
                except (TypeError, ValueError):
                    over = False
                reproduced = over and not obj.get(f"ticket:{case}.refund.approved")
            elif rule == "retention_offer_before_closure":
                reproduced = f"ticket:{case}.account" in obj and f"ticket:{case}.retention_offer" not in obj
            elif rule == "identity_check_before_deletion":
                reproduced = f"ticket:{case}.account" in obj and f"ticket:{case}.identity_check" not in obj
            else:
                reproduced = None
            checks.append({"seq": e.seq, "what": f"check:{rule}", "match": reproduced})
        elif e.type == "sla_breach":
            sla = str(e.data.get("sla"))
            m = re.search(r"(\d+)h", sla)
            limit = float(m.group(1)) if m else None
            if sla.startswith("reopen_response"):  # clock starts at the latest reopen; stops at the next answer
                reopen = max((x for x in t.events if x.type == "reopened" and x.seq < e.seq), key=lambda x: x.seq, default=None)
                answered = reopen is not None and any(x.type in ("agent_reply", "supervisor_edit") and reopen.seq < x.seq < e.seq for x in t.events)
                start = reopen.time if reopen else None
            else:  # first response: from opening to the first answer
                answered = f"ticket:{case}.resolution" in obj
                start = opened
            reproduced = limit is not None and start is not None and not answered and _hours(start, e.time) > limit
            checks.append({"seq": e.seq, "what": f"check:{e.data.get('sla')}", "match": reproduced})
    statuses = [e.data["status"] for e in t.events if e.data.get("status")]
    final = st[t.events[-1].seq].objects.get(f"ticket:{case}.status")
    checks.append({"seq": t.events[-1].seq, "what": "final_status", "match": final == (statuses[-1] if statuses else None)})
    return {"trace_id": t.trace_id, "ok": all(c["match"] for c in checks), "checks": checks}


# ---------------------------------------------------------------- milestones
def mine(good: list[Trace], min_share: float = 0.5) -> list[Milestone]:
    ends = [toy_states(t)[-1][1] for t in good]
    present: Counter[str] = Counter()
    values: dict[str, Counter[str]] = {}
    for t, s in zip(good, ends, strict=True):
        case = t.source.ref
        for obj, v in s.objects.items():
            tmpl = obj.replace(case, "{case}")
            present[tmpl] += 1
            values.setdefault(tmpl, Counter())[str(v)] += 1
    out = []
    for tmpl, n in sorted(present.items()):
        if n / len(good) < min_share or tmpl.endswith(".request"):
            continue  # the request exists at the base state of every ticket: trivial
        field = tmpl.rsplit(".", 1)[-1]
        top, cnt = values[tmpl].most_common(1)[0]
        if cnt / n >= 0.9 and field in ("status",):  # a consistent final value: the milestone is that value
            q, exp = Query(op="match", args={"object": tmpl, "pattern": f"^{re.escape(top)}$"}), Expectation(kind="equals", value={"exists": True, "matches": True})
            mid = f"log/{field}={top}"
        else:
            q, exp = Query(op="exists", args={"object": tmpl}), Expectation(kind="truthy")
            mid = f"log/{field}"
        out.append(Milestone(id=mid, check=MilestoneCheck(app="tickets", query=q, expect=exp), weight=1.0, cost="cheap",
                             provenance=[Provenance(source="good_trace_end_states", note=f"{n}/{len(good)} good dev tickets")]))
    return out


async def series_of(ms: list[Milestone], t: Trace):
    return await evaluate_series(ms, toy_states(t))


async def build_graph(dev: list[Trace]) -> tuple[WorkflowGraph, dict[str, Any]]:
    good = [t for t in dev if outcome(t) == "good"]
    cands = mine(good)
    all_series = {t.trace_id: await series_of(cands, t) for t in dev}
    good_series = [all_series[t.trace_id] for t in good]
    kept, verdicts = [], {}
    for m in cands:
        base = (await evaluate([m], toy_states(good[0])[0][1]))[m.id]
        ends = [(await evaluate([m], toy_states(t)[-1][1]))[m.id] for t in good]
        end = max((x for x in ends if x is not None), default=None)  # reached at the end of at least one good ticket
        v = screen(m.id, base_phi=base, end_phi=end, series=list(all_series.values()), good=good_series, min_monotone=0.6)
        verdicts[m.id] = {"kept": v.kept, "reason": v.reason, **v.checks}
        if v.kept:
            kept.append(m)
    ach = []
    for t in good:
        s = all_series[t.trace_id]
        mt = match(kept, s)
        n = max(len(s.values) - 1, 1)
        ach.append({k: i / n for k, i in mt.achieved.items()})
    model = name_stages(induce(kept, ach, min_support=0.9, min_traces=3))
    graph = WorkflowGraph.build(FAMILY, kept, model, verdicts=verdicts, sources=[t.trace_id for t in good],
                                stats={"dev_tickets": len(dev), "good_dev_tickets": len(good), "candidates": len(cands)})
    return graph, {"candidates": [m.id for m in cands], "kept": [m.id for m in kept]}


def kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


async def track(graph: WorkflowGraph, t: Trace, vocab: list[str]) -> list[str]:
    """Tracker stage after every event, mapped onto the label vocabulary by order (stage k incomplete -> vocab[k];
    every stage done -> the last word)."""
    tr = WorkflowTracker(graph)
    st = toy_states(t)
    steps_by_seq = {q: s for s in segment(t) for q in s.seqs}
    out = []
    for seq, view in st:
        ts = await tr.observe(steps_by_seq.get(seq), view)
        k = min(ts.stage, len(vocab) - 1)
        out.append(vocab[k] if len(graph.stages) == len(vocab) - 1 else vocab[round(k * (len(vocab) - 1) / max(len(graph.stages), 1))])
    return out


# ---------------------------------------------------------------- reward + properties
async def reward(case: str, obj: dict[str, Any], ms: list[Milestone]):
    view = LogStateView(objects=obj, case=case)
    phis = await evaluate(ms, view)
    refund = obj.get(f"ticket:{case}.refund")
    try:
        unapproved = float(refund) > 50 and not obj.get(f"ticket:{case}.refund.approved")
    except (TypeError, ValueError):
        unapproved = False
    no_offer = f"ticket:{case}.account" in obj and f"ticket:{case}.retention_offer" not in obj
    resolved = obj.get(f"ticket:{case}.status") == "resolved" and f"ticket:{case}.resolution" in obj and refund != "reversed"
    return compose(gates={"refund_policy": not unapproved, "retention_policy": not no_offer}, outcome=resolved, phi=potential(ms, phis)[0])


async def properties(ms: list[Milestone]) -> dict[str, Any]:
    case = "T9999"
    base = {f"ticket:{case}.request": "ticket_opened"}
    oracle = {**base, f"ticket:{case}.assignee": "tier1", f"ticket:{case}.resolution": "agent_reply", f"ticket:{case}.status": "resolved"}
    negatives = {"refund_over_50_unapproved": {**oracle, f"ticket:{case}.refund": "89.00"},
                 "refund_reversed": {**oracle, f"ticket:{case}.refund": "reversed"},
                 "resolved_without_answer": {**base, f"ticket:{case}.assignee": "tier1", f"ticket:{case}.status": "resolved"},
                 "reopened": {**oracle, f"ticket:{case}.status": "open"}}
    loop_reward = {}
    for name, o in [("oracle", oracle), ("base", base), *negatives.items()]:
        loop_reward[name] = await reward(case, o, ms)
    states_named = {"oracle": oracle, "base": base, **negatives}
    rv = {id(v): loop_reward[k] for k, v in states_named.items()}
    by_id = lambda s: rv[id(s)]
    async def ph(o):
        return await evaluate(ms, LogStateView(objects=o, case=case))
    mono = [await ph(base), await ph({**base, f"ticket:{case}.assignee": "tier1"}), await ph(oracle)]
    osc = [await ph(base), await ph(oracle), await ph({**oracle, f"ticket:{case}.status": "open"}), await ph(oracle)]
    stub = await ph({**base, f"ticket:{case}.status": "resolved"})
    res = [r1_oracle(by_id, oracle), r2_noop(by_id, base), r3_negatives(by_id, negatives), r4_oscillation(ms, mono, osc), r5_stubbing(ms, stub),
           r6_state_only(lambda s, msgs: by_id(s), oracle, [["Resolved! Customer is happy."], ["ok"]])]
    return check_all(res)


async def run() -> dict[str, Any]:
    ts = traces()
    dev = [ts[c] for c in sorted(DEV)]
    test = [ts[c] for c in sorted(TEST)]
    rep = {c: replay(ts[c]) for c in sorted(ts)}
    graph, mined = await build_graph(dev)
    labels = json.loads((HERE / "stage_labels_test.json").read_text())
    vocab = labels["vocabulary"]
    pred, gold = [], []
    for t in test:
        p = await track(graph, t, vocab)
        pred += p
        gold += labels["labels"][t.source.ref]
    k = kappa(gold, pred)
    # S2 on the toy: Φ at the midpoint (and end) vs outcome, on test tickets with a known outcome
    rows = []
    for t in test:
        y = outcome(t)
        if y == "unknown":
            continue
        s = await series_of(graph.milestones, t)
        pot = trajectory(graph.milestones, s.values)
        rows.append((t.source.ref, 1 if y == "good" else 0, pot[len(pot) // 2].phi, pot[-1].phi))
    props = await properties(graph.milestones)
    return {"family": FAMILY, "replay": {"traces": len(rep), "replayed": sum(r["ok"] for r in rep.values()), "rate": sum(r["ok"] for r in rep.values()) / len(rep),
                                         "checks": sum(len(r["checks"]) for r in rep.values()), "failed": [r for r in rep.values() if not r["ok"]]},
            "milestones": mined, "graph": json.loads(graph.model_dump_json()),
            "tracker": {"test_events": len(gold), "kappa": round(k, 4), "agreement": round(sum(a == b for a, b in zip(gold, pred, strict=True)) / len(gold), 4),
                        "confusion": Counter(f"{a}->{b}" for a, b in zip(gold, pred, strict=True) if a != b)},
            "s2": {"test_tickets_with_outcome": len(rows), "phi_mid_auc": auc([r[2] for r in rows], [r[1] for r in rows]),
                   "phi_end_auc": auc([r[3] for r in rows], [r[1] for r in rows]), "rows": rows},
            "properties": props}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), indent=1, default=str))
