"""Human-queue triage (DESIGN §7.7 item 20): the panel's abstentions turned into a review sheet for a person.

The panel decides a suspect only when its families agree at confidence ≥ λ; every other suspect is appended to the run's
``human_queue.jsonl`` and stays open (it rejects its task). This module never decides one. It

1. **groups** the items by *reason*: the probe's class and operator, the grader's verdict and which family voted which
   way (``MUT/PerturbValue, grader passed, nemotron: invalid vs openai: valid``), and within a group by task and site
   pattern (the operator's argument, without the value);
2. **ranks** them by expected information: ``EI = H(p) × (1 + pivotal) × √(similar)``, where ``p`` is the panel's
   pooled probability that the attempt is invalid (each vote's confidence, mirrored for a "valid" vote; one number, so
   two confident opposite votes give p ≈ 0.5 and the most entropy), ``H`` the binary entropy in bits, ``pivotal`` = 1
   when the task's every rejection reason is an open suspect or a bound missed for want of n (deciding its queue can
   change its status), and
   ``similar`` the number of queued items with the same reason and site pattern on the same task (one decision informs them all).
   The sheet lists one representative of each such cluster first (by EI), then the rest;
3. **exports** a sheet: ``review.csv`` (one row per item, ranked, with empty ``decision`` and ``note`` columns) and
   ``review.md`` (the same, readable: instruction, the attempt's calls, the grader's verdict, the two votes and their
   reasons, the question to answer). ``read_decisions`` reads a filled sheet back as ``{item id: valid|invalid}``; the
   labels follow ``agents.resolve`` (a decided item is confirmed when the person and the grader disagree).

What the person sees is the panel's input plus the votes; the operator name is shown last, after the question, so the
reading of the attempt comes first.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import content_hash

QUESTION = {
    True: "The grader PASSED this attempt. Is it really a valid solution of the instruction? (valid / invalid / unsure)",
    False: "The grader FAILED this attempt. Is it in fact a valid solution of the instruction? (valid / invalid / unsure)",
}


class QueueItem(CrucibleModel):
    id: str
    task: str
    version: int = 1
    cls: str
    op: str
    site: str
    grader_passed: bool | None = None
    votes: list[dict[str, Any]] = Field(default_factory=list)
    instruction: str = ""
    attempt: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""  # the group key
    p_invalid: float = 0.5
    entropy: float = 0.0
    pivotal: bool = False
    similar: int = 1
    ei: float = 0.0
    rank: int = 0
    representative: bool = True  # the first of its cluster (same task, reason and site pattern) in the ranking


def load_queue(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def item_id(row: Mapping[str, Any]) -> str:
    return "hq-" + content_hash({k: row.get(k) for k in ("task", "version", "cls", "op", "site", "grader_passed")}, 12)


def site_pattern(op: str, site: str) -> str:
    """The site without its value: ``s3.reason=duplicate`` -> ``s3.reason=*``; ``s2.ticket_id->T-1009`` -> ``s2.ticket_id->*``."""
    return re.sub(r"(=|->|\+)[^.]*$", r"\1*", site) if op not in ("agent",) else "agent"


def p_invalid(votes: Sequence[Mapping[str, Any]]) -> float:
    ps = []
    for v in votes:
        if v.get("valid") is None:
            continue
        c = float(v.get("confidence") or 0.0)
        ps.append(1 - c if v["valid"] else c)
    return sum(ps) / len(ps) if ps else 0.5


def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def reason_key(row: Mapping[str, Any]) -> str:
    votes = sorted(f"{v.get('family')}: {'?' if v.get('valid') is None else 'valid' if v['valid'] else 'invalid'}" for v in row.get("votes") or [])
    g = "passed" if row.get("grader_passed") else "failed"
    return f"{row.get('cls')}/{row.get('op')}, grader {g}, " + " vs ".join(votes)


def task_pivotal(report: Mapping[str, Any] | None) -> dict[str, bool]:
    """task id -> every rejection reason is an open suspect or a bound missed for want of n (``:upper=``, which only
    makes a task provisional), from a pipeline run's ``report.json``: deciding its queue can change its status."""
    out: dict[str, bool] = {}
    for t in (report or {}).get("tasks") or []:
        rs = list(t.get("reasons") or [])
        out[str(t.get("id"))] = (t.get("status") == "rejected" and any(":suspects=" in r for r in rs)
                                 and all(":suspects=" in r or ":upper=" in r for r in rs))
    return out


def triage(rows: Iterable[Mapping[str, Any]], *, report: Mapping[str, Any] | None = None) -> list[QueueItem]:
    """Group and rank (module docstring). Duplicate rows (one probe queued twice) are kept once."""
    piv = task_pivotal(report)
    items: dict[str, QueueItem] = {}
    for r in rows:
        iid = item_id(r)
        if iid in items:
            continue
        items[iid] = QueueItem(id=iid, task=str(r.get("task")), version=int(r.get("version") or 1), cls=str(r.get("cls")), op=str(r.get("op")),
                               site=str(r.get("site")), grader_passed=r.get("grader_passed"), votes=list(r.get("votes") or []),
                               instruction=str(r.get("instruction") or ""), attempt=dict(r.get("attempt") or {}), reason=reason_key(r))
    sim = Counter((i.task, i.reason, site_pattern(i.op, i.site)) for i in items.values())
    for i in items.values():
        i.p_invalid = round(p_invalid(i.votes), 4)
        i.entropy = round(entropy(i.p_invalid), 4)
        i.pivotal = piv.get(i.task, False)
        i.similar = sim[(i.task, i.reason, site_pattern(i.op, i.site))]
        i.ei = round(i.entropy * (1 + int(i.pivotal)) * math.sqrt(i.similar), 4)
    ordered = sorted(items.values(), key=lambda i: (-i.ei, i.reason, i.task, i.site))
    seen: set[tuple[str, str, str]] = set()
    first, rest = [], []
    for i in ordered:  # one representative per cluster first: deciding it informs the cluster, the rest follow
        k = (i.task, i.reason, site_pattern(i.op, i.site))
        i.representative = k not in seen
        (first if i.representative else rest).append(i)
        seen.add(k)
    out = first + rest
    for n, i in enumerate(out, 1):
        i.rank = n
    return out


def groups(items: Sequence[QueueItem]) -> list[dict[str, Any]]:
    by: dict[str, list[QueueItem]] = defaultdict(list)
    for i in items:
        by[i.reason].append(i)
    return sorted(({"reason": k, "n": len(v), "tasks": len({i.task for i in v}), "pivotal_tasks": len({i.task for i in v if i.pivotal}),
                    "ei": round(sum(i.ei for i in v), 3), "top_rank": min(i.rank for i in v)} for k, v in by.items()), key=lambda g: -g["ei"])


def _calls(attempt: Mapping[str, Any], obs_chars: int = 160) -> list[str]:
    out = []
    for c in attempt.get("trace") or []:
        args = json.dumps(c.get("args") or {}, ensure_ascii=False, sort_keys=True)
        res = c.get("result")
        res = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False) if res is not None else ""
        out.append(f"{c.get('tool')}({args[:300]})" + (f" -> {res[:obs_chars]}" if res else ""))
    return out


COLUMNS = ["rank", "id", "decision", "note", "ei", "group", "task", "instruction", "attempt", "answer", "grader", "votes", "question", "probe"]


def export(items: Sequence[QueueItem], out_dir: str | Path, *, title: str = "Human review") -> dict[str, str]:
    """``review.csv`` and ``review.md`` (module docstring). Returns their paths."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in items:
        votes = "; ".join(f"{v.get('family')}: {'valid' if v.get('valid') else 'invalid' if v.get('valid') is False else 'no answer'} "
                          f"({float(v.get('confidence') or 0):.2f}) {str(v.get('why') or v.get('error') or '')[:300]}" for v in i.votes)
        rows.append({"rank": i.rank, "id": i.id, "decision": "", "note": "", "ei": i.ei, "group": i.reason, "task": i.task, "instruction": i.instruction,
                     "attempt": "\n".join(_calls(i.attempt)), "answer": json.dumps(i.attempt.get("answer"), ensure_ascii=False),
                     "grader": "passed" if i.grader_passed else "failed", "votes": votes, "question": QUESTION[bool(i.grader_passed)],
                     "probe": f"{i.cls}/{i.op} {i.site}"})
    csv_p = d / "review.csv"
    with csv_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    gs = groups(items)
    intro = (f"{len(items)} items in {len(gs)} groups. Fill `decision` in `review.csv` with valid, invalid or unsure. Nothing here "
             "is decided: the panel abstained on every item.")
    md = [f"# {title}", "", intro, "", "## Groups (by expected information)", "", "| group | items | tasks | pivotal tasks | EI | first rank |",
          "|---|---|---|---|---|---|"]
    md += [f"| {g['reason']} | {g['n']} | {g['tasks']} | {g['pivotal_tasks']} | {g['ei']} | {g['top_rank']} |" for g in gs]
    md += ["", "## Items", ""]
    for r, i in zip(rows, items, strict=True):
        md += [f"### {r['rank']}. `{r['id']}` (EI {r['ei']}{', pivotal task' if i.pivotal else ''}, {i.similar} similar)", "", f"Task `{r['task']}`", "",
               f"> {r['instruction']}", "", "Attempt:", "", "```", r["attempt"] or "(no calls)", "```", "", f"Answer: `{r['answer']}`. Grader: **{r['grader']}**.", "",
               f"Panel: {r['votes']}", "", f"**Question.** {r['question']}", "", f"Decision: ____  (probe, read after deciding: {r['probe']})", ""]
    md_p = d / "review.md"
    md_p.write_text("\n".join(md))
    (d / "groups.json").write_text(json.dumps(gs, indent=1))
    return {"csv": str(csv_p), "md": str(md_p)}


def read_decisions(path: str | Path) -> dict[str, str]:
    """``{item id: "valid" | "invalid"}`` from a filled ``review.csv`` (blank and "unsure" rows are left open)."""
    out: dict[str, str] = {}
    with Path(path).open() as fh:
        for r in csv.DictReader(fh):
            d = (r.get("decision") or "").strip().lower()
            if d in ("valid", "invalid"):
                out[r["id"]] = d
    return out


__all__ = ["QueueItem", "entropy", "export", "groups", "item_id", "load_queue", "p_invalid", "read_decisions", "reason_key", "site_pattern", "task_pivotal",
           "triage"]
