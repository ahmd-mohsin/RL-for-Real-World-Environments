"""The regularised update rule (part2-paper-core §9.4): propose → evaluate on a fresh view → accept or refuse, over the
versioned system state Θ = (S, F, P, K, A), with an append-only ledger of every decision.

A proposed update ``u`` touches **one** component and is accepted only if **every** rule holds; the rules are
non-compensatory, so every failing rule is recorded, and a gain on one never buys a loss on another:

| rule | as built |
|---|---|
| R1 fresh view | each decision samples a new view (``ViewSampler``); a view hash seen before, a view drawn from the locked set, or a view whose planted batch omits a past planter version is refused. View hashes are kept in the ledger. |
| R2 noise floor | δ comes from ``calibrate``: ≥ 3 evaluations of the unchanged Θ on fresh views, δ = max(2·sd, range/2) of their J; it is set once and a recalibration can only raise it. Accept only if LCB₉₅(J′ − J*) > δ, J* being the incumbent (the best accepted Θ) on the same view, paired by unit. |
| R3 non-regression | the archive still rejects every archived counterexample; FA ≤ ``fa_max``; no class's FR_d below the incumbent's by more than its calibrated noise band; the P3 on-intent share and the P4 coverage likewise. |
| R4 trust region | KL(q′ ‖ q) over the cell histogram ≤ κ (additive smoothing) and H(q′) ≥ H_min. |
| R5 sparsity, cost | at most ``edit_budget(round)`` edits, annealed 3 → 1 on a cosine over ``rounds``; when the update costs more, its marginal mean recall per extra probe-dollar must be ≥ β; ``prune`` removes items with no marginal gain over W rounds. |
| R6 leakage, families | the critic (injected; default ``leakage_critic``: task ids, entity keys, literal values, planted-defect ids and anchor content found in the diff) must return nothing, and the author family must not be one that built the evidence's tasks. |
| R7 transfer | the paired gain on the units of domains *not* in the update's evidence must exceed δ (``transfer_margin`` overrides). |
| R8 round caps | an update whose evidence is judge-only, with no new executed counterexamples, is refused once its component has accepted ``judge_round_cap`` judge-only rounds in a row. |
| R9 equal-budget control | the incumbent re-evaluated on the same view with the candidate's spend (``budget=cost′``) must also be beaten: LCB(J′ − J_ctrl) > δ. |
| R10 rollback | an update whose data does not round-trip through JSON (not restorable by a plain write) is refused; every accepted one stores a version and checks that the previous version restores byte-identically; ``audit`` compares Θ with the previous good Θ on a fresh view and rolls back on a regression. |

Static rules (R5's budget, R6, R8, R10's round-trip) run first, so a refused update spends no evaluation; the
decision still enters the ledger with ``evaluated = False``. Everything model-driven (the proposer, an evaluator that
needs roles, a critic of another family) is injected; nothing here calls a model.

**Storage.** ``ThetaStore(root)`` keeps every version as a write-once JSON file (``<root>/<component>/v<N>.json``) and
the heads in ``<root>/heads.json`` (replaced atomically); ``root=None`` keeps them in memory (tests). ``Ledger(path)``
is a hash-chained JSONL (the archive's format): an edited, removed or reordered line breaks the chain.
"""

from __future__ import annotations

import inspect
import json
import math
import os
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from crucible.core import canonical
from crucible.core.ids import sha256_text
from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson, content_hash, is_number
from crucible.synth.program import is_structural, text_spans

COMPONENTS = ("S", "F", "P", "K", "A")
GENESIS = "0" * 64
Z_ONE_SIDED_95 = 1.644854


# ---------------------------------------------------------------------------------------------------------- data

class Evidence(CrucibleModel):
    """What an update was derived from: the leakage critic's token sources, R7's evidence domains, R8's kind."""

    domains: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    plant_ids: list[str] = Field(default_factory=list)
    entity_keys: list[str] = Field(default_factory=list)
    literals: list[Any] = Field(default_factory=list)
    builder_families: list[str] = Field(default_factory=list)
    kind: Literal["executed", "judge"] = "executed"
    new_executed: int = 0  # executed counterexamples that arrived since the component's last accepted update


class Update(CrucibleModel):
    id: str
    component: str
    data: Any = None  # the component's new data (must round-trip through JSON: R10)
    edits: list[str] = Field(default_factory=list)  # one entry per atomic edit (R5's budget counts these)
    diff: str = ""  # the text the leakage critic reads (default: the data's JSON)
    items: list[str] = Field(default_factory=list)  # named sub-items the update adds or changes (pruning, R5)
    author_family: str = ""
    evidence: Evidence = Field(default_factory=Evidence)


class View(CrucibleModel):
    """A freshly sampled validation view: new proposals plus a newly planted dev batch."""

    hash: str
    round: int = 0
    domains: list[str] = Field(default_factory=list)
    planters: list[str] = Field(default_factory=list)  # planter versions the planted batch was drawn from
    locked: bool = False
    seed: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


class Measurement(CrucibleModel):
    """One evaluation of a Θ on a view. ``j_units`` are per-unit contributions to J (perfect tasks per dollar), paired
    across Θs on one view by index; ``unit_domains`` names each unit's domain (R7)."""

    j_units: list[float]
    unit_domains: list[str] = Field(default_factory=list)
    fr: dict[str, float] = Field(default_factory=dict)  # defect class -> population recall
    fa: float = 0.0
    archive_ok: bool = True
    p3: float | None = None
    p4: float | None = None
    cells: dict[str, float] = Field(default_factory=dict)  # the cell histogram of the tasks this Θ produces
    cost: float = 0.0
    item_gain: dict[str, float] = Field(default_factory=dict)  # marginal recall of each named item (pruning)

    @property
    def j(self) -> float:
        return sum(self.j_units) / len(self.j_units) if self.j_units else 0.0


class Theta(CrucibleModel):
    versions: dict[str, int] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)

    def with_component(self, component: str, data: Any, version: int = -1) -> Theta:
        return Theta(versions={**self.versions, component: version}, data={**self.data, component: data})


class RuleFailure(CrucibleModel):
    rule: str
    detail: str


class Decision(CrucibleModel):
    kind: Literal["update", "prune", "rollback", "calibrate"] = "update"
    round: int
    update_id: str = ""
    component: str = ""
    from_version: int | None = None
    to_version: int | None = None
    accepted: bool = False
    evaluated: bool = False
    failed: list[RuleFailure] = Field(default_factory=list)
    view: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def rules(self) -> list[str]:
        return sorted({f.rule for f in self.failed}, key=lambda r: int(r[1:]) if r[1:].isdigit() else 99)


class NoiseFloor(CrucibleModel):
    delta: float
    j_values: list[float]
    bands: dict[str, float] = Field(default_factory=dict)  # "fr:D2", "p3", "p4" -> calibrated noise band
    views: list[str] = Field(default_factory=list)


class UpdateConfig(CrucibleModel):
    rounds: int = 10  # the horizon of R5's cosine edit budget
    edit_hi: int = 3
    edit_lo: int = 1
    z: float = Z_ONE_SIDED_95
    min_repeats: int = 3
    fa_max: float = 0.0
    min_band: float = 0.0  # floor of every R3 noise band
    kappa: float = 0.1  # R4: KL bound
    h_min: float = 0.0  # R4: entropy floor (nats)
    smoothing: float = 1e-3
    beta: float = 0.0  # R5: marginal recall per extra probe-dollar
    prune_window: int = 3  # R5: W
    transfer_margin: float | None = None  # R7: None = the noise floor δ
    judge_round_cap: int = 2  # R8


class IrreversibleUpdate(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------- protocols

class Evaluator(Protocol):
    async def __call__(self, theta: Theta, view: View, budget: float | None) -> Measurement: ...


class ViewSampler(Protocol):
    async def __call__(self, round: int, purpose: str, used: frozenset[str]) -> View: ...


LeakageCritic = Callable[[Update], Sequence[str] | Awaitable[Sequence[str]]]


# ---------------------------------------------------------------------------------------------------------- storage

def _roundtrip(data: Any) -> Any:
    try:
        return json.loads(canonical.dumps(data))
    except (TypeError, ValueError) as e:
        raise IrreversibleUpdate(f"the component data is not plain JSON ({type(e).__name__})") from None


class ThetaStore:
    """Versioned components (see the module docstring). Version numbers start at 1 per component."""

    def __init__(self, root: str | os.PathLike[str] | None = None):
        self.root = Path(root) if root is not None else None
        self._versions: dict[str, dict[int, Any]] = {}
        self._heads: dict[str, int] = {}
        if self.root is not None and (self.root / "heads.json").exists():
            self._heads = {k: int(v) for k, v in json.loads((self.root / "heads.json").read_text()).items()}
            for c in self._heads:
                for p in sorted((self.root / c).glob("v*.json")):
                    self._versions.setdefault(c, {})[int(p.stem[1:])] = json.loads(p.read_text())

    def put(self, component: str, data: Any) -> int:
        if component not in COMPONENTS:
            raise ValueError(f"unknown component {component!r}")
        d = _roundtrip(data)
        vs = self._versions.setdefault(component, {})
        v = max(vs, default=0) + 1
        vs[v] = d
        if self.root is not None:
            p = self.root / component / f"v{v}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "x") as fh:  # write-once
                fh.write(canonical.dumps(d))
        return v

    def get(self, component: str, version: int | None = None) -> Any:
        v = self._heads[component] if version is None else version
        return json.loads(canonical.dumps(self._versions[component][v]))

    def digest(self, component: str, version: int) -> str:
        return content_hash(self._versions[component][version], 32)

    def reread_digest(self, component: str, version: int) -> str:
        """The digest of a version as a restore would read it (from disk when the store is rooted)."""
        if self.root is None:
            return content_hash(self.get(component, version), 32)
        return content_hash(json.loads((self.root / component / f"v{version}.json").read_text()), 32)

    def head(self, component: str) -> int | None:
        return self._heads.get(component)

    def set_head(self, component: str, version: int) -> None:
        if version not in self._versions.get(component, {}):
            raise KeyError(f"{component} has no version {version}")
        self._heads[component] = version
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / "heads.json.tmp"
            tmp.write_text(json.dumps(self._heads, sort_keys=True))
            os.replace(tmp, self.root / "heads.json")

    def init(self, component: str, data: Any) -> int:
        v = self.put(component, data)
        self.set_head(component, v)
        return v

    def theta(self) -> Theta:
        return Theta(versions=dict(self._heads), data={c: self.get(c) for c in self._heads})

    def versions(self, component: str) -> list[int]:
        return sorted(self._versions.get(component, {}))


class Ledger:
    """Append-only, hash-chained JSONL of decisions (``path=None``: in memory)."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else None
        self._mem: list[dict[str, Any]] = []

    def _lines(self) -> list[dict[str, Any]]:
        if self.path is None:
            return list(self._mem)
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def records(self) -> list[Decision]:
        prev, out = GENESIS, []
        for n, line in enumerate(self._lines(), 1):
            if line["seq"] != n or line["prev"] != prev or line["h"] != sha256_text(f"{prev}\n{n}\n{canonical.dumps(line['rec'])}"):
                raise ValueError(f"ledger line {n}: the hash chain is broken")
            out.append(Decision.model_validate(line["rec"]))
            prev = line["h"]
        return out

    def append(self, d: Decision) -> None:
        lines = self._lines()
        prev = lines[-1]["h"] if lines else GENESIS
        seq = len(lines) + 1
        rec = json.loads(d.model_dump_json())
        line = {"seq": seq, "prev": prev, "h": sha256_text(f"{prev}\n{seq}\n{canonical.dumps(rec)}"), "rec": rec}
        if self.path is None:
            self._mem.append(line)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------------------------------------- statistics

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def lcb(a: Sequence[float], b: Sequence[float], z: float = Z_ONE_SIDED_95) -> float:
    """One-sided lower confidence bound on mean(a) − mean(b): paired by index when the lengths agree, else Welch."""
    if not a or not b:
        return -math.inf
    if len(a) == len(b):
        d = [x - y for x, y in zip(a, b, strict=True)]
        return _mean(d) - z * math.sqrt(_var(d) / len(d))
    return _mean(a) - _mean(b) - z * math.sqrt(_var(a) / len(a) + _var(b) / len(b))


def _dist(h: dict[str, float], keys: Iterable[str], eps: float) -> dict[str, float]:
    ks = sorted(set(keys))
    tot = sum(max(0.0, h.get(k, 0.0)) + eps for k in ks)
    return {k: (max(0.0, h.get(k, 0.0)) + eps) / tot for k in ks}


def kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-3) -> float:
    """KL(p ‖ q) of two histograms over the union of their cells, with additive smoothing ``eps``."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    pp, qq = _dist(p, keys, eps), _dist(q, keys, eps)
    return sum(pp[k] * math.log(pp[k] / qq[k]) for k in keys)


def entropy(p: dict[str, float]) -> float:
    tot = sum(max(0.0, v) for v in p.values())
    if tot <= 0:
        return 0.0
    return -sum((v / tot) * math.log(v / tot) for v in p.values() if v > 0)


def edit_budget(round: int, rounds: int, hi: int = 3, lo: int = 1) -> int:
    """R5: the cosine-annealed edit budget, ``hi`` at round 0 down to ``lo`` at round ``rounds - 1`` and after."""
    if rounds <= 1:
        return lo
    t = min(max(round, 0), rounds - 1) / (rounds - 1)
    return math.floor(lo + (hi - lo) * (1 + math.cos(math.pi * t)) / 2 + 0.5)


# ---------------------------------------------------------------------------------------------------------- R6

def leakage_critic(u: Update, anchor_tokens: Sequence[str] = ()) -> list[str]:
    """The default, deterministic leakage check: every task id, planted-defect id, entity key and literal value of the
    update's evidence, and every anchor token, searched in the diff (word-bounded, numbers numerically). Numbers below
    10, booleans and empty strings carry no identity and are not searched; strings need 3+ characters."""
    try:
        text = u.diff or cjson(u.data)
    except (TypeError, ValueError):  # not JSON (R10 refuses it anyway): read its text form
        text = repr(u.data)
    out: list[str] = []

    def hit(kind: str, v: Any) -> None:
        if is_structural(v) or (is_number(v) and abs(v) < 10) or (isinstance(v, str) and len(v.strip()) < 3):
            return
        if text_spans(v, text):
            out.append(f"{kind}:{v}")

    for t in u.evidence.task_ids:
        hit("task_id", t)
    for t in u.evidence.plant_ids:
        hit("plant_id", t)
    for t in u.evidence.entity_keys:
        hit("entity", t)
    for v in u.evidence.literals:
        hit("literal", v)
    for t in anchor_tokens:
        hit("anchor", t)
    return out


# ---------------------------------------------------------------------------------------------------------- the engine

class _Good(CrucibleModel):
    versions: dict[str, int]
    round: int


class UpdateEngine:
    """The regularised update rule over a ``ThetaStore`` (see the module docstring)."""

    def __init__(self, store: ThetaStore, evaluate: Evaluator, sample: ViewSampler, *, config: UpdateConfig | None = None,
                 critic: LeakageCritic | None = None, anchor_tokens: Sequence[str] = (), ledger: Ledger | None = None):
        self.store = store
        self.evaluate = evaluate
        self.sample = sample
        self.config = config or UpdateConfig()
        self.critic = critic
        self.anchor_tokens = list(anchor_tokens)
        self.ledger = ledger or Ledger()
        self.floor: NoiseFloor | None = None
        self.used: set[str] = set()
        self.judge_streak: dict[str, int] = {}
        self.last_gain: dict[str, int] = {}  # item -> last round with a positive marginal gain (or its birth)
        self.good: list[_Good] = [_Good(versions=dict(store.theta().versions), round=-1)]
        self.head_digest: dict[str, str] = {c: store.digest(c, v) for c, v in store.theta().versions.items()}  # R10

    # -- views -------------------------------------------------------------------------------------------------------
    def _planter_history(self) -> list[str]:
        return [f"P@{v}" for v in self.store.versions("P")]

    async def _view(self, round: int, purpose: str, fail: list[RuleFailure]) -> View | None:
        v = await self.sample(round, purpose, frozenset(self.used))
        if v.hash in self.used:
            fail.append(RuleFailure(rule="R1", detail=f"view {v.hash[:12]} was used before"))
            return None
        if v.locked:
            fail.append(RuleFailure(rule="R1", detail="the view draws on the locked set (report-only)"))
            return None
        missing = sorted(set(self._planter_history()) - set(v.planters))
        if missing:
            fail.append(RuleFailure(rule="R1", detail=f"the planted batch omits past planters {missing}"))
            return None
        self.used.add(v.hash)
        return v

    # -- R2 calibration ------------------------------------------------------------------------------------------------
    async def calibrate(self, round: int = 0, repeats: int | None = None) -> NoiseFloor:
        """δ from ≥ 3 evaluations of the unchanged Θ on fresh views (never tuned: a recalibration only raises it)."""
        n = repeats or self.config.min_repeats
        if n < self.config.min_repeats:
            raise ValueError(f"the noise floor needs at least {self.config.min_repeats} repeats")
        theta = self.store.theta()
        ms: list[Measurement] = []
        views: list[str] = []
        for _ in range(n):
            fail: list[RuleFailure] = []
            v = await self._view(round, "calibrate", fail)
            if v is None:
                raise RuntimeError(f"calibration needs fresh views: {fail[0].detail}")
            views.append(v.hash)
            ms.append(await self.evaluate(theta, v, None))
        js = [m.j for m in ms]
        delta = max(2 * math.sqrt(_var(js)), (max(js) - min(js)) / 2)
        bands: dict[str, float] = {}
        for d in sorted({d for m in ms for d in m.fr}):
            xs = [m.fr.get(d, 0.0) for m in ms]
            bands[f"fr:{d}"] = max(self.config.min_band, 2 * math.sqrt(_var(xs)), (max(xs) - min(xs)) / 2)
        for key in ("p3", "p4"):
            xs = [getattr(m, key) for m in ms if getattr(m, key) is not None]
            if xs:
                bands[key] = max(self.config.min_band, 2 * math.sqrt(_var(xs)), (max(xs) - min(xs)) / 2)
        if self.floor is not None:  # the floor never moves down
            delta = max(delta, self.floor.delta)
            bands = {k: max(v, self.floor.bands.get(k, 0.0)) for k, v in {**self.floor.bands, **bands}.items()}
        self.floor = NoiseFloor(delta=delta, j_values=js, bands=bands, views=views)
        self.ledger.append(Decision(kind="calibrate", round=round, accepted=True, evaluated=True, metrics={"delta": delta, "j": js, "bands": bands}))
        return self.floor

    def set_floor(self, delta: float) -> None:
        raise PermissionError("the noise floor is measured by calibrate(), never tuned (R2)")

    # -- the rule ------------------------------------------------------------------------------------------------------
    async def _critic(self, u: Update) -> list[str]:
        out = list(leakage_critic(u, self.anchor_tokens)) if self.critic is None else self.critic(u)
        if inspect.isawaitable(out):
            out = await out
        return list(out)  # type: ignore[arg-type]

    async def decide(self, u: Update, round: int) -> Decision:
        if self.floor is None:
            raise RuntimeError("calibrate() first: the noise floor is measured before the loop (R2)")
        cfg = self.config
        comp = u.component
        prev = self.store.head(comp)
        dec = Decision(round=round, update_id=u.id, component=comp, from_version=prev)
        fail: list[RuleFailure] = []
        # static rules: no evaluation is spent on an update that fails one
        b = edit_budget(round, cfg.rounds, cfg.edit_hi, cfg.edit_lo)
        if len(u.edits) > b:
            fail.append(RuleFailure(rule="R5", detail=f"{len(u.edits)} edits > budget {b} at round {round}"))
        leaks = await self._critic(u)
        if leaks:
            fail.append(RuleFailure(rule="R6", detail="leakage: " + ", ".join(leaks[:8])))
        if u.author_family and u.author_family in u.evidence.builder_families:
            fail.append(RuleFailure(rule="R6", detail=f"the author family {u.author_family} built the source tasks"))
        if u.evidence.kind == "judge" and u.evidence.new_executed == 0 and self.judge_streak.get(comp, 0) >= cfg.judge_round_cap:
            fail.append(RuleFailure(rule="R8", detail=f"{self.judge_streak[comp]} judge-only rounds and no new executed counterexamples"))
        try:
            data = _roundtrip(u.data)
        except IrreversibleUpdate as e:
            fail.append(RuleFailure(rule="R10", detail=str(e)))
            data = None
        if fail:
            dec.failed = fail
            self.ledger.append(dec)
            return dec
        view = await self._view(round, "decide", fail)
        if view is None:
            dec.failed = fail
            self.ledger.append(dec)
            return dec
        dec.view = view.hash
        dec.evaluated = True
        theta = self.store.theta()
        cand = theta.with_component(comp, data)
        m0 = await self.evaluate(theta, view, None)
        m1 = await self.evaluate(cand, view, None)
        delta = self.floor.delta
        # R2
        g = lcb(m1.j_units, m0.j_units, cfg.z)
        if not g > delta:
            fail.append(RuleFailure(rule="R2", detail=f"LCB(J'-J*) = {g:.4f} <= delta {delta:.4f}"))
        # R3
        if not m1.archive_ok:
            fail.append(RuleFailure(rule="R3", detail="an archived counterexample re-opened"))
        if m1.fa > cfg.fa_max:
            fail.append(RuleFailure(rule="R3", detail=f"FA {m1.fa:.3f} > {cfg.fa_max}"))
        for d, x in sorted(m0.fr.items()):
            band = self.floor.bands.get(f"fr:{d}", cfg.min_band)
            if m1.fr.get(d, 0.0) < x - band:
                fail.append(RuleFailure(rule="R3", detail=f"FR_{d} {m1.fr.get(d, 0.0):.3f} < {x:.3f} - band {band:.3f}"))
        for key in ("p3", "p4"):
            a, b0 = getattr(m1, key), getattr(m0, key)
            band = self.floor.bands.get(key, cfg.min_band)
            if b0 is not None and (a is None or a < b0 - band):
                fail.append(RuleFailure(rule="R3", detail=f"{key} {a} < {b0:.3f} - band {band:.3f}"))
        # R4
        if m0.cells or m1.cells:
            k = kl(m1.cells, m0.cells, cfg.smoothing)
            if k > cfg.kappa:
                fail.append(RuleFailure(rule="R4", detail=f"KL(q'||q) = {k:.4f} > kappa {cfg.kappa}"))
            h = entropy(m1.cells)
            if h < cfg.h_min:
                fail.append(RuleFailure(rule="R4", detail=f"H(q') = {h:.4f} < H_min {cfg.h_min}"))
        # R5 cost bar
        dc = m1.cost - m0.cost
        if dc > 0 and (m1.fr or m0.fr):
            gain = _mean(list(m1.fr.values())) - _mean(list(m0.fr.values()))
            if gain / dc < cfg.beta:
                fail.append(RuleFailure(rule="R5", detail=f"marginal recall per probe-dollar {gain / dc:.5f} < beta {cfg.beta}"))
        # R7
        held = [i for i, d in enumerate(m1.unit_domains) if d not in set(u.evidence.domains)]
        margin = delta if cfg.transfer_margin is None else cfg.transfer_margin
        if not held or len(m0.j_units) != len(m1.j_units):
            fail.append(RuleFailure(rule="R7", detail="the view has no domain outside the update's evidence"))
        else:
            tg = _mean([m1.j_units[i] - m0.j_units[i] for i in held])
            if not tg > margin:
                fail.append(RuleFailure(rule="R7", detail=f"held-out-domain gain {tg:.4f} <= {margin:.4f}"))
        # R9
        mc = await self.evaluate(theta, view, m1.cost)
        gc = lcb(m1.j_units, mc.j_units, cfg.z)
        if not gc > delta:
            fail.append(RuleFailure(rule="R9", detail=f"LCB(J'-J_control) = {gc:.4f} <= delta {delta:.4f} (control at spend {m1.cost:.3f})"))
        dec.metrics = {"j_star": m0.j, "j_new": m1.j, "j_control": mc.j, "lcb": g, "lcb_control": gc, "delta": delta, "fr_new": m1.fr, "fa_new": m1.fa,
                       "cost_new": m1.cost}
        for item, gain in m1.item_gain.items():
            if gain > 0:
                self.last_gain[item] = round
        if not fail:
            v = self.store.put(comp, data)
            if prev is not None and self.store.reread_digest(comp, prev) != self.head_digest.get(comp):  # R10: what we leave must restore exactly
                fail.append(RuleFailure(rule="R10", detail="the previous version does not restore exactly"))
            if not fail:
                self.store.set_head(comp, v)
                self.head_digest[comp] = self.store.digest(comp, v)
                dec.accepted, dec.to_version = True, v
                self.good.append(_Good(versions=dict(self.store.theta().versions), round=round))
                self.judge_streak[comp] = self.judge_streak.get(comp, 0) + 1 if u.evidence.kind == "judge" and u.evidence.new_executed == 0 else 0
                for item in u.items:
                    self.last_gain.setdefault(item, round)
        dec.failed = fail
        self.ledger.append(dec)
        return dec

    # -- R10 rollback ----------------------------------------------------------------------------------------------------
    def _restore(self, versions: dict[str, int]) -> None:
        for c, v in versions.items():
            self.store.set_head(c, v)

    async def audit(self, round: int) -> Decision:
        """On a fresh view, compare Θ with the previous good Θ (paired). A significant loss (LCB(J_prev − J) > δ), a
        re-opened archived counterexample or FA above the bar rolls back to the previous good Θ."""
        if self.floor is None:
            raise RuntimeError("calibrate() first")
        dec = Decision(kind="rollback", round=round)
        fail: list[RuleFailure] = []
        if len(self.good) < 2:
            dec.metrics = {"note": "nothing accepted yet"}
            self.ledger.append(dec)
            return dec
        view = await self._view(round, "audit", fail)
        if view is None:
            dec.failed = fail
            self.ledger.append(dec)
            return dec
        dec.view, dec.evaluated = view.hash, True
        cur = self.store.theta()
        prev = self.good[-2].versions
        prev_theta = Theta(versions=dict(prev), data={c: self.store.get(c, v) for c, v in prev.items()})
        m_cur = await self.evaluate(cur, view, None)
        m_prev = await self.evaluate(prev_theta, view, None)
        loss = lcb(m_prev.j_units, m_cur.j_units, self.config.z)
        why = []
        if loss > self.floor.delta:
            why.append(f"LCB(J_prev - J) = {loss:.4f} > delta {self.floor.delta:.4f}")
        if not m_cur.archive_ok:
            why.append("an archived counterexample re-opened")
        if m_cur.fa > self.config.fa_max:
            why.append(f"FA {m_cur.fa:.3f} > {self.config.fa_max}")
        dec.metrics = {"j": m_cur.j, "j_prev": m_prev.j, "loss_lcb": loss, "regressed": why}
        if why:
            self._restore(prev)
            self.head_digest = {c: self.store.digest(c, v) for c, v in prev.items()}
            self.good.pop()
            dec.accepted = True  # the rollback was applied
            dec.from_version, dec.to_version = None, None
            dec.metrics["restored"] = prev
        self.ledger.append(dec)
        return dec

    # -- R5 pruning ----------------------------------------------------------------------------------------------------
    def stale(self, round: int) -> list[str]:
        w = self.config.prune_window
        return sorted(i for i, r in self.last_gain.items() if round - r >= w)

    async def prune(self, component: str, round: int, remove: Callable[[Any, list[str]], Any]) -> Decision | None:
        """Remove the component's items with no marginal gain over W rounds. Accepted on a fresh view when R3 holds and
        J does not drop beyond the noise floor (LCB(J − J') ≤ δ): pruning buys cost and sparsity, not J."""
        if self.floor is None:
            raise RuntimeError("calibrate() first")
        items = self.stale(round)
        if not items:
            return None
        dec = Decision(kind="prune", round=round, component=component, from_version=self.store.head(component), update_id=f"prune:{','.join(items)}")
        fail: list[RuleFailure] = []
        theta = self.store.theta()
        data = _roundtrip(remove(theta.data[component], items))
        view = await self._view(round, "prune", fail)
        if view is not None:
            dec.view, dec.evaluated = view.hash, True
            m0 = await self.evaluate(theta, view, None)
            m1 = await self.evaluate(theta.with_component(component, data), view, None)
            loss = lcb(m0.j_units, m1.j_units, self.config.z)
            if loss > self.floor.delta:
                fail.append(RuleFailure(rule="R5", detail=f"pruning {items} loses J: LCB = {loss:.4f} > delta"))
            if not m1.archive_ok or m1.fa > self.config.fa_max:
                fail.append(RuleFailure(rule="R3", detail="pruning breaks non-regression"))
            for d, x in m0.fr.items():
                if m1.fr.get(d, 0.0) < x - self.floor.bands.get(f"fr:{d}", self.config.min_band):
                    fail.append(RuleFailure(rule="R3", detail=f"pruning drops FR_{d}"))
            dec.metrics = {"items": items, "j": m0.j, "j_pruned": m1.j}
        if not fail and view is not None:
            v = self.store.put(component, data)
            self.store.set_head(component, v)
            self.head_digest[component] = self.store.digest(component, v)
            dec.accepted, dec.to_version = True, v
            self.good.append(_Good(versions=dict(self.store.theta().versions), round=round))
            for i in items:
                self.last_gain.pop(i, None)
        dec.failed = fail
        self.ledger.append(dec)
        return dec
