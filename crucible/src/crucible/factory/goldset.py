"""The family VerifierGoldSet (05 §8.2, R4, R5): the verifier is trusted only through its measured error on labelled end
states.

Items are real solver passes and fails, requirement mutants, valid alternatives and confirmed hacks; at least half are
real solver outputs. Truth comes from:
- **proof** for constructed items: a mutant or a hack should fail, a valid alternative should pass;
- **human labels** (``factory.labels``: append-only, bound to the item's sha256, superseded never edited) for real items.
  Two humans who disagree leave the item unlabelled until one supersedes; a label on an older hash is stale and ignored.
  A human label on a constructed item overrides the proof, and the conflict is reported.

Measured: false-pass rate = verifier passes / truly-failing items, false-fail rate = verifier fails / truly-passing
items, each with its Wilson upper bound (*v1* ≤ 5%), and Cohen's κ between the model reviewer and the human on the
overlap (*v1* ≥ 0.7 on ≥ 10% of the model-reviewed items; below it the human stays a gate for the family).

Note: zero errors in n items give a Wilson upper bound of ``3.84 / (n + 3.84)``, so a 5% bound needs ≥ 73 truly-failing
*and* ≥ 73 truly-passing measured items. The *v1* minimum of 40 items cannot meet the *v1* 5% bound; the two thresholds
are kept as separate config values and both are enforced.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from crucible.factory.floors import Frac, Graded, call, graded_from

SOURCES = ("real_pass", "real_fail", "mutant", "alternative", "hack")
REAL = frozenset({"real_pass", "real_fail"})
PROVEN = {"mutant": False, "hack": False, "alternative": True}
PASS_VALUES = frozenset({"pass", "correct", "true", "1", "yes"})
FAIL_VALUES = frozenset({"fail", "incorrect", "false", "0", "no"})


@dataclass
class GoldItem:
    id: str
    source: Literal["real_pass", "real_fail", "mutant", "alternative", "hack"]
    solution: Any
    sha256: str  # labels.item_sha256 of the item content
    proven_by: str = ""  # "mutant:<requirement>", "hack:<attack id>", ...


@dataclass
class GoldConfig:
    min_items: int = 40  # v1
    min_real_share: float = 0.5  # v1: at least half are real solver outputs
    required_sources: tuple[str, ...] = ("real_pass", "real_fail", "mutant", "alternative")
    fp_upper_max: float = 0.05  # v1
    fn_upper_max: float = 0.05  # v1
    kappa_min: float = 0.7  # v1
    kappa_share: float = 0.10  # v1: >= 10% of the model-reviewed items carry a human label too
    human_lane: bool = False
    pass_at: float = 1.0


def _value(v: str) -> bool | None:
    s = v.strip().lower()
    return True if s in PASS_VALUES else False if s in FAIL_VALUES else None


def effective(labels: Any, item: str, sha: str) -> list[Any]:
    """Current labels on ``item`` bound to ``sha``: from a ``LabelStore`` or a plain list of ``Label`` records."""
    if hasattr(labels, "effective"):
        return list(labels.effective(item, sha))
    superseded = {lab.supersedes for lab in labels if lab.supersedes}
    return [lab for lab in labels if lab.item == item and lab.id not in superseded and lab.item_sha256 == sha]


def _stale(labels: Any, item: str, sha: str) -> int:
    if hasattr(labels, "stale"):
        return len(labels.stale(item, sha))
    superseded = {lab.supersedes for lab in labels if lab.supersedes}
    return sum(1 for lab in labels if lab.item == item and lab.id not in superseded and lab.item_sha256 != sha)


def truths(items: Sequence[GoldItem], labels: Any = ()) -> tuple[dict[str, bool | None], dict]:
    """Truth per item and the bookkeeping (human disagreements, stale labels, proof/label conflicts)."""
    out: dict[str, bool | None] = {}
    info: dict[str, list] = {"human_disagreement": [], "stale_labels": [], "proof_conflict": [], "unlabelled_real": []}
    for it in items:
        hum = [lab for lab in effective(labels, it.id, it.sha256) if lab.labeller.startswith("human:")]
        if _stale(labels, it.id, it.sha256):
            info["stale_labels"].append(it.id)
        vals = {_value(lab.value) for lab in hum} - {None}
        human = None if len(vals) != 1 else next(iter(vals))
        if len(vals) > 1:
            info["human_disagreement"].append(it.id)
        proof = PROVEN.get(it.source)
        if human is not None:
            out[it.id] = human
            if proof is not None and proof != human:
                info["proof_conflict"].append(it.id)
        elif proof is not None and not vals:
            out[it.id] = proof
        else:
            out[it.id] = None
            if it.source in REAL:
                info["unlabelled_real"].append(it.id)
    return out, info


def kappa(items: Sequence[GoldItem], labels: Any) -> dict:
    """Model reviewer vs human on items both labelled (one current label of each kind; the first labeller of each kind)."""
    from crucible.factory.labels import cohen_kappa

    pairs = []
    reviewed = 0
    for it in items:
        eff = effective(labels, it.id, it.sha256)
        model = sorted((lab for lab in eff if lab.labeller.startswith("model:")), key=lambda lab: lab.labeller)
        human = sorted((lab for lab in eff if lab.labeller.startswith("human:")), key=lambda lab: lab.labeller)
        if model:
            reviewed += 1
            if human:
                pairs.append((model[0].value.strip().lower(), human[0].value.strip().lower()))
    k = cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
    return {"kappa": None if k is None else round(k, 4), "n": len(pairs), "model_reviewed": reviewed,
            "observed": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None}


@dataclass
class GoldReport:
    fp: Frac
    fn: Frac
    kappa: dict
    counts: dict[str, int]
    real_share: float
    unmeasured: list[str]
    info: dict
    errors: dict[str, list[str]]  # "false_pass" / "false_fail" -> item ids
    config: GoldConfig = field(default_factory=GoldConfig)

    @property
    def human_gate(self) -> bool:
        """The human stays a gate for the family (human lane, κ unmeasured, under-sampled, or below the bar)."""
        c, k = self.config, self.kappa
        need = math.ceil(round(c.kappa_share * k["model_reviewed"], 9))
        return c.human_lane and (k["kappa"] is None or k["kappa"] < c.kappa_min or k["n"] < max(1, need))

    def reasons(self) -> list[str]:
        c, out = self.config, []
        n = sum(self.counts.values())
        if n < c.min_items:
            out.append(f"{n} gold items, need {c.min_items}")
        if self.real_share < c.min_real_share:
            out.append(f"real solver outputs are {self.real_share:.0%} of the gold set, need {c.min_real_share:.0%}")
        missing = [s for s in c.required_sources if not self.counts.get(s)]
        if missing:
            out.append(f"no gold items of kind {missing}")
        for name, f, mx in (("false-pass", self.fp, c.fp_upper_max), ("false-fail", self.fn, c.fn_upper_max)):
            hi = f.interval[1]
            if f.n == 0 or hi > mx:
                out.append(f"{name} rate {f.k}/{f.n}, Wilson upper {hi:.3f} > {mx}")
        if self.info["unlabelled_real"]:
            out.append(f"real items without a human truth label: {len(self.info['unlabelled_real'])}")
        if self.info["proof_conflict"]:
            out.append(f"human labels contradict the construction (a 'mutant' that is right, or an 'alternative' that is wrong): {self.info['proof_conflict']}")
        return out

    @property
    def ok(self) -> bool:
        return not self.reasons()

    def card(self) -> dict:
        """``VerifierCard`` fields: fp, fn, kappa."""
        return {"fp": self.fp.pair(), "fn": self.fn.pair(), "kappa": self.kappa["kappa"]}

    def to_dict(self) -> dict:
        return {"fp": self.fp.to_dict(), "fn": self.fn.to_dict(), "kappa": self.kappa, "counts": self.counts, "real_share": round(self.real_share, 4),
                "unmeasured": self.unmeasured, "info": self.info, "errors": self.errors, "human_gate": self.human_gate, "ok": self.ok, "reasons": self.reasons()}


def score(items: Sequence[GoldItem], verdicts: Mapping[str, Any], labels: Any = (), *, config: GoldConfig | None = None) -> GoldReport:
    """``verdicts``: item id -> the verifier's verdict (``Graded`` or anything ``graded_from`` reads)."""
    cfg = config or GoldConfig()
    truth, info = truths(items, labels)
    fp_k = fp_n = fn_k = fn_n = 0
    unmeasured, errors = [], {"false_pass": [], "false_fail": []}
    for it in items:
        t = truth[it.id]
        g: Graded = graded_from(verdicts.get(it.id))
        if t is None:
            continue
        if g.masked or not g.finite:
            unmeasured.append(it.id)
            continue
        p = bool(g.passed(cfg.pass_at))
        if t:
            fn_n += 1
            if not p:
                fn_k += 1
                errors["false_fail"].append(it.id)
        else:
            fp_n += 1
            if p:
                fp_k += 1
                errors["false_pass"].append(it.id)
    counts = {s: sum(1 for it in items if it.source == s) for s in SOURCES}
    real = sum(counts[s] for s in REAL)
    return GoldReport(Frac(fp_k, fp_n), Frac(fn_k, fn_n), kappa(items, labels), counts, real / len(items) if items else 0.0, unmeasured, info, errors, cfg)


async def run_goldset(items: Sequence[GoldItem], grade: Callable[[Any], Awaitable[Any]], labels: Any = (), *, config: GoldConfig | None = None) -> GoldReport:
    """Grade every gold item with the verifier under test, then ``score``."""
    verdicts = {it.id: graded_from(await call(grade, it.solution)) for it in items}
    return score(items, verdicts, labels, config=config)
