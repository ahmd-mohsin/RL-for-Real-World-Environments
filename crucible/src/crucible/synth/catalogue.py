"""The domain catalogue: which enterprise domains World families are built for (part2-paper-core §11, §12).

The seed list of **function areas** (``FUNCTION_AREAS``) is framework data; the domains themselves are written by the
swarm (the ``domain_cataloguer`` role, a non-Qwen builder family), a few per area, shown every domain already in the
catalogue so it does not repeat one. Each entry names a keyword, its function, a description, the main records, the
kinds of work, and whether it is common practice (an obscure domain stops teaching transferable skills).

**Dedupe rule** (``duplicate_of``): two entries are the same domain when, after lower-casing, removing stop words and
light stemming, their keyword token sets have Jaccard ≥ 0.6, or they share a function and their main-record sets have
Jaccard ≥ 0.6 (common acronyms are expanded first: AP, AR, PO, GL, HR, CRM, KYC, ...). The earlier entry is kept; the
later one is recorded with the entry it duplicates. Entries that are not common practice are kept out with the reason.
An entry's ``function`` is the area it was asked for (the framework's, not the model's: in the first pilot the replies
put a description there for a third of the areas). ``pick`` chooses a diverse subset (round-robin over functions).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crucible.synth.roles import Role, RoleClient, RoleFailed, ask

FUNCTION_AREAS: tuple[str, ...] = (
    "finance: accounts payable, accounts receivable, general ledger, expenses, treasury",
    "procurement: requisitions, purchase orders, supplier onboarding, contracts",
    "HR: recruiting, onboarding, leave, payroll changes, performance cycles",
    "IT: service desk, access requests, asset management, change management",
    "sales and CRM: leads, opportunities, quotes, account management",
    "customer support: cases, returns and refunds, warranty claims, SLAs",
    "legal and compliance: contract review, policy attestations, audits, data-subject requests",
    "supply chain: inventory, warehouse receiving, shipments, demand planning",
    "insurance: claims intake, adjudication, underwriting referrals",
    "healthcare administration: scheduling, prior authorisation, billing and coding, referrals",
    "facilities: work orders, space booking, preventive maintenance, visitor management",
    "marketing operations: campaigns, budgets, asset approvals, event registrations",
    "project and portfolio management: resourcing, timesheets, change requests",
    "manufacturing operations: work orders, quality inspections, non-conformance reports",
    "banking operations: loan applications, KYC reviews, dispute handling",
    "real estate and property management: leases, tenant requests, rent collection",
)

STOP = frozenset(["a", "an", "and", "the", "of", "for", "to", "in", "on", "with", "by", "management", "processing", "system", "systems", "process", "operations", "ops", "desk", "request", "requests", "workflow", "tracking", "handling"])


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CatalogueEntry(_M):
    keyword: str
    function: str
    description: str = ""
    records: list[str] = Field(default_factory=list)
    work: list[str] = Field(default_factory=list)
    common_practice: bool = True
    why: str = ""


class CatalogueOut(_M):
    domains: list[CatalogueEntry]


class Catalogue(_M):
    entries: list[CatalogueEntry] = Field(default_factory=list)
    duplicates: list[dict[str, Any]] = Field(default_factory=list)
    refused: list[dict[str, Any]] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "es", "s", "ment", "ion"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


ACRONYMS = {"ap": "accounts payable", "ar": "accounts receivable", "po": "purchase order", "pos": "purchase orders", "gl": "general ledger",
            "hr": "human resources", "crm": "customer relationship", "kyc": "know your customer", "sla": "service level",
            "rfq": "request for quotation", "rma": "return authorisation", "pto": "paid time off", "ncr": "non conformance report"}


def tokens(text: str) -> set[str]:
    words = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        words += ACRONYMS[w].split() if w in ACRONYMS else [w]
    return {_stem(w) for w in words if w not in STOP and len(w) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def duplicate_of(e: CatalogueEntry, kept: Sequence[CatalogueEntry], *, kw: float = 0.6, rec: float = 0.6) -> str | None:
    """The keyword of the kept entry ``e`` duplicates (module docstring), else None."""
    te = tokens(e.keyword)
    re_ = {_stem(x) for r in e.records for x in tokens(r)}
    for k in kept:
        if jaccard(te, tokens(k.keyword)) >= kw:
            return k.keyword
        same_fn = tokens(e.function.split(":")[0]) & tokens(k.function.split(":")[0])
        if same_fn and re_ and jaccard(re_, {_stem(x) for r in k.records for x in tokens(r)}) >= rec:
            return k.keyword
    return None


def add_entries(cat: Catalogue, new: Sequence[CatalogueEntry]) -> list[CatalogueEntry]:
    added = []
    for e in new:
        if not e.common_practice:
            cat.refused.append({"keyword": e.keyword, "why": "not common practice"})
            continue
        d = duplicate_of(e, cat.entries)
        if d is not None:
            cat.duplicates.append({"keyword": e.keyword, "duplicate_of": d})
            continue
        cat.entries.append(e)
        added.append(e)
    return added


DOMAIN_CATALOGUER = Role(
    name="domain_cataloguer", stage="W0", output=CatalogueOut, temperature=0.7, max_tokens=4096,
    sees=("function_area", "existing", "count"),
    never=("tasks", "tests"),
    system=("You list enterprise software domains for one business function: the everyday work that staff in a mid-sized company do in "
            "an internal system (an ERP, a CRM, a service desk, a case system). Each domain is one coherent application area with its "
            "own records and decisions, common practice that a new hire would recognise (not niche, not fictional). Do not repeat or "
            "rename a domain in 'existing'.\n"
            'Reply {"domains": [{"keyword": "<3 to 6 words>", "function": "<the function>", "description": "<1 sentence>", '
            '"records": [<the 3 to 6 main record types>], "work": [<3 to 5 kinds of decisions or tasks>], "common_practice": true, '
            '"why": "<why it is common>"}]} with exactly the requested count.'))


async def build_catalogue(client: RoleClient, *, per_area: int = 3, areas: Sequence[str] = FUNCTION_AREAS, cat: Catalogue | None = None,
                          path: str | Path | None = None, target: int | None = None) -> Catalogue:
    """Ask the swarm for ``per_area`` domains per function area, dedupe, and save after every area (resumable: areas
    already asked are skipped). Stops once ``target`` entries exist."""
    cat = cat or Catalogue()
    done = {c["area"] for c in cat.calls if c.get("ok")}
    for area in areas:
        if area in done:
            continue
        if target is not None and len(cat.entries) >= target:
            break
        try:
            out: CatalogueOut = await ask(client, DOMAIN_CATALOGUER, tag="catalogue:area", retries=1, function_area=area,
                                          existing=[e.keyword for e in cat.entries] or None, count=per_area)
            fn = area.split(":")[0].strip()
            added = add_entries(cat, [e.model_copy(update={"function": fn}) for e in out.domains[:per_area + 1]])
            cat.calls.append({"area": area, "ok": True, "returned": len(out.domains), "added": [e.keyword for e in added]})
        except RoleFailed as e:
            cat.calls.append({"area": area, "ok": False, "error": str(e)[:300]})
        if path is not None:
            save(cat, path)
    normalize_functions(cat)
    if path is not None:
        save(cat, path)
    return cat


def normalize_functions(cat: Catalogue) -> None:
    """Set every entry's function to the area it was asked for (from the call log)."""
    area_of = {kw: c["area"].split(":")[0].strip() for c in cat.calls if c.get("ok") for kw in c.get("added") or []}
    for i, e in enumerate(cat.entries):
        if e.keyword in area_of and e.function != area_of[e.keyword]:
            cat.entries[i] = e.model_copy(update={"function": area_of[e.keyword]})


def pick(cat: Catalogue, n: int, *, exclude: Sequence[str] = ()) -> list[CatalogueEntry]:
    """``n`` diverse entries: one per function in turn (catalogue order within a function)."""
    by: dict[str, list[CatalogueEntry]] = {}
    for e in cat.entries:
        if e.keyword in exclude:
            continue
        by.setdefault(e.function.split(":")[0].strip().lower(), []).append(e)
    out: list[CatalogueEntry] = []
    while len(out) < n and any(by.values()):
        for fn in list(by):
            if by[fn] and len(out) < n:
                out.append(by[fn].pop(0))
    return out


def save(cat: Catalogue, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cat.model_dump(mode="json"), indent=1, ensure_ascii=False) + "\n")
    tmp.replace(p)


def load(path: str | Path) -> Catalogue:
    p = Path(path)
    return Catalogue.model_validate_json(p.read_text()) if p.exists() else Catalogue()


__all__ = ["DOMAIN_CATALOGUER", "FUNCTION_AREAS", "Catalogue", "CatalogueEntry", "add_entries", "build_catalogue", "duplicate_of", "jaccard", "load", "pick",
           "save", "tokens"]
