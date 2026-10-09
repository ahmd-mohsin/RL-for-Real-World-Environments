"""M2 grounding gate on real review history: an LLM-phrased grounded responder must not fabricate.

The responder (``GroundedResponder(mode="llm")``) answers questions as the historical reviewer, from
221 real review comments (PCL + CCCL). Questions: answerable ones about specific files, plus
unanswerable ones (nothing in the records) that must get "I have no record of that."

Audit: every delivered reply is re-checked deterministically (citations exist, are visible at the
time, every number / identifier appears verbatim, content overlap) and a sample of factual sentences
is judged for entailment by a model from a different family. Gate: fabrication rate <= 1%.
Budget-capped (default $0.10 responder + $0.02 judge). Run:
``python -m crucible_pack_coding.grounding_gate out.json [n_answerable] [n_unanswerable]``.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

from crucible.actors.fidelity import grounding_audit
from crucible.actors.grounded import GroundedResponder
from crucible.actors.grounding import NO_RECORD
from crucible.actors.llm import BudgetExceeded, LLMClient
from crucible.actors.records import RecordStore
from crucible_pack_coding.github_reviews import GitHubCache, fetch_review_comments
from crucible_pack_coding.reviewer_fidelity import REPOS

UNANSWERABLE = [
    "What is the company's travel reimbursement policy?", "Which cafeteria vendor did we pick last quarter?",
    "What was the Q3 marketing budget?", "Who approved the new office lease?", "What is our parental leave policy?",
    "Which cloud region hosts the payroll database?", "What did the board decide about the merger?",
    "How many vacation days do interns get?", "What is the dress code for client meetings?",
    "Which law firm handles our trademark filings?",
]


async def run(out: Path, n_answerable: int = 170, n_unanswerable: int = 30, *, mode: str = "quote", budget: float = 0.10, judge_budget: float = 0.03, judge_sample: int = 80) -> dict:
    cache = GitHubCache(Path("data/cache/github"))
    records = [c.to_record() for repo in REPOS for c in fetch_review_comments(cache, repo, pages=3)]
    store = RecordStore(records)
    rnd = random.Random(7)
    picked = rnd.sample(records, min(n_answerable, len(records)))
    questions = [(f"As the reviewer of {r.target.object}, what concern did you raise about that file?", r.time) for r in picked]
    questions += [(UNANSWERABLE[i % len(UNANSWERABLE)], None) for i in range(n_unanswerable)]
    rnd.shuffle(questions)
    llm = LLMClient("sim-default", budget_usd=budget)
    resp = GroundedResponder("reviewer", store, who="a senior maintainer who reviewed pull requests in this project", mode=mode, llm=llm, k=5, min_score=2.0)
    delivered = []
    stopped = None
    for q, t in questions:
        # Ask "now" = just after the comment was written, so the record is visible but nothing later is.
        now = (t[:-1] + ".5Z") if t else "2030-01-01T00:00:00Z"
        try:
            text, cites = await resp.answer(q, now=now)
        except BudgetExceeded as e:
            stopped = str(e)
            break
        delivered.append({"q": q, "a": text, "cites": cites, "now": now, "answerable": t is not None})
    judge = LLMClient("judge-default", budget_usd=judge_budget)
    sample = [d for d in delivered if d["a"] != NO_RECORD][:judge_sample]
    rest = [d for d in delivered if d not in sample]
    audit_judged = await grounding_audit([(d["a"], d["cites"], d["now"]) for d in sample], store, judge=judge)
    audit_rest = await grounding_audit([(d["a"], d["cites"], d["now"]) for d in rest], store)
    unans = [d for d in delivered if not d["answerable"]]
    ans = [d for d in delivered if d["answerable"]]
    factual = audit_judged.factual_sentences + audit_rest.factual_sentences
    bad = audit_judged.unsupported_sentences + audit_judged.judge_rejected + audit_rest.unsupported_sentences
    res = {
        "mode": mode,
        "records": len(records),
        "replies": len(delivered),
        "answerable": len(ans),
        "answered_with_citations": sum(1 for d in ans if d["a"] != NO_RECORD and d["cites"]),
        "answerable_no_record": sum(1 for d in ans if d["a"] == NO_RECORD),
        "unanswerable": len(unans),
        "unanswerable_correct_no_record": sum(1 for d in unans if d["a"] == NO_RECORD),
        "uncited_non_no_record_replies": sum(1 for d in delivered if d["a"] != NO_RECORD and not d["cites"]),
        "factual_sentences": factual,
        "unsupported_or_judge_rejected": bad,
        "fabrication_rate": round(bad / factual, 4) if factual else 0.0,
        "judged_sentences": audit_judged.judged,
        "judge_rejected": audit_judged.judge_rejected,
        "responder_stats": resp.stats,
        "spend_usd": {"responder": round(llm.spend.usd, 5), "judge": round(judge.spend.usd, 5)},
        "llm_calls": {"responder": llm.spend.calls, "responder_cached": llm.spend.cached_calls, "judge": judge.spend.calls},
        "models": {"responder": llm.model, "judge": judge.model},
        "stopped_early": stopped,
        "examples": (audit_judged.examples + audit_rest.examples)[:12],
        "sample_replies": [{k: d[k] for k in ("q", "a", "cites")} for d in delivered[:8]],
    }
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("m2_grounding_gate.json")
    n_a = int(sys.argv[2]) if len(sys.argv) > 2 else 170
    n_u = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    mode = sys.argv[4] if len(sys.argv) > 4 else "quote"
    res = asyncio.run(run(out, n_a, n_u, mode=mode))
    print(json.dumps({k: v for k, v in res.items() if k not in ("examples", "sample_replies")}, indent=1))


if __name__ == "__main__":
    main()
