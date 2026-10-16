"""The root curator of the task factory (M17; 05 §11, 03 §3; Prime Agent's root that "wrote no product code", PATTERN).

The root **writes no tasks**. It holds the coverage matrix (behaviour rows x surface columns), the backlog of open cells and
the leases; it runs per-task machines (``driver.task_run``) in parallel up to a concurrency cap, and merges only tasks
whose machine reached ``accepted`` *and* whose gate verdict passed (checked again here: a nomination is coordination,
never verification).

- **Leases** (``LeaseBook``): one live lease per cell, in a file under a lock (``fcntl``), so two roots sharing a
  directory never work the same cell at once; a lease expires after ``ttl_s`` (a crashed holder's cell reopens), and the
  same holder re-acquiring after a crash gets its run id back, so the machine resumes from its journal.
- **Stops** (05 §1.6: no numeric targets): the shared forge ``Ledger`` (budget: launch nothing more, in-flight machines
  stop themselves at their next reserve); marginal yield through ``forge/economy.YieldBook`` stop rules (a paused
  generator's cells are not launched); diversity through the YieldBook quota (an accepted task that would break it is
  deferred, not merged) and a surface-share cap (05 §8 v1: one surface <= 40% of the accepted tasks). The run ends when
  no cell can be launched and nothing is in flight; the report says why.
- **Escalations** are queued as decision requests (``decisions``) and keep their cell's lease (renewed; after ``ttl_s``
  another holder may take the cell over and resumes the same paused run); ``answer(run_id, decision)`` resumes it.
- **Persistent specialists** (``Specialists``): a hacker memory per family (``forge/specialists.FamilyMemory`` +
  ``SpecialistHacker``), a World expert per World, any other named specialist: created once per root and shared by
  every machine (the ``actors/rlm/tree.ChildRegistry`` pattern: get-or-spawn by name, kept for the run).
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.machine import MachineRun, RunResult
from crucible.forge.budget import Ledger
from crucible.forge.economy import YieldBook
from crucible.forge.specialists import FamilyMemory, SpecialistHacker

MAX_SURFACE_SHARE = 0.40  # 05 §8 v1
SURFACE_SHARE_FLOOR = 5  # below this many accepted tasks a share is noise
_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def slug(s: str) -> str:
    return _SLUG.sub("-", s).strip("-") or "x"


# ---- coverage matrix ---------------------------------------------------------------------------------------------------
@dataclass
class Cell:
    behaviour: str
    surface: str
    intent: dict[str, Any]  # an approved IntentCard as JSON
    surface_spec: dict[str, Any] = field(default_factory=dict)
    quota: int = 1  # accepted tasks wanted in this cell
    observations: list[dict[str, Any]] = field(default_factory=list)
    accepted: list[str] = field(default_factory=list)
    attempts: int = 0
    status: str = "open"  # open | filled | escalated | closed

    @property
    def key(self) -> str:
        return f"{slug(self.behaviour)}--{slug(self.surface)}"

    def as_param(self) -> dict[str, Any]:
        return {"key": self.key, "behaviour": self.behaviour, "surface": self.surface, "surface_spec": self.surface_spec, "quota": self.quota}


class CoverageMatrix:
    """Behaviours x surfaces. A cell is open until its quota of accepted tasks is merged (or it is closed)."""

    def __init__(self, cells: Sequence[Cell] = ()):
        self.cells: dict[str, Cell] = {}
        for c in cells:
            self.add(c)

    def add(self, cell: Cell) -> Cell:
        if cell.key in self.cells:
            raise ValueError(f"cell {cell.key} already exists")
        self.cells[cell.key] = cell
        return cell

    def backlog(self) -> list[Cell]:
        """Open cells, the least covered first (fewest accepted, then fewest attempts, then key)."""
        return sorted((c for c in self.cells.values() if c.status == "open"), key=lambda c: (len(c.accepted), c.attempts, c.key))

    def mark_accepted(self, key: str, task_id: str) -> None:
        c = self.cells[key]
        if task_id not in c.accepted:
            c.accepted.append(task_id)
        if len(c.accepted) >= c.quota:
            c.status = "filled"

    def surface_share(self, surface: str, *, adding: int = 1) -> float:
        total = sum(len(c.accepted) for c in self.cells.values()) + adding
        mine = sum(len(c.accepted) for c in self.cells.values() if c.surface == surface) + adding
        return mine / total if total else 0.0

    def view(self) -> dict[str, dict[str, dict[str, Any]]]:
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for c in self.cells.values():
            out.setdefault(c.behaviour, {})[c.surface] = {"accepted": len(c.accepted), "quota": c.quota, "attempts": c.attempts, "status": c.status}
        return out


# ---- leases ------------------------------------------------------------------------------------------------------------
@dataclass
class Lease:
    cell: str
    holder: str
    run_id: str
    expires: float


class LeaseBook:
    """File-backed leases (``leases.json`` under ``leases.lock``). ``acquire`` gives a cell to one holder at a time."""

    def __init__(self, root: Path, *, ttl_s: float = 3600.0, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path, self.lock = self.root / "leases.json", self.root / "leases.lock"
        self.ttl_s, self.clock = ttl_s, clock

    @contextmanager
    def _locked(self) -> Any:
        with self.lock.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                data = json.loads(self.path.read_text()) if self.path.exists() and self.path.read_text().strip() else {}
                for k in ("leases", "runs", "filled", "stopped"):
                    data.setdefault(k, {})
                yield data
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, sort_keys=True))
                tmp.replace(self.path)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def acquire(self, cell: str, holder: str, *, run_id: str | None = None) -> Lease | None:
        """A lease on ``cell`` for ``holder``, or None when another holder's lease is live. The same holder gets its
        unfinished run id back (a resumed machine); ``run_id`` re-leases a known run (an answered escalation); otherwise
        a new run id ``<cell>/<n>``."""
        now = self.clock()
        with self._locked() as data:
            cur = data["leases"].get(cell)
            if cur and cur["expires"] > now and cur["holder"] != holder:
                return None
            if data["filled"].get(cell) and run_id is None:  # another holder (or an earlier run) filled it: no duplicate work
                return None
            if run_id is not None:
                if run_id not in data["runs"].get(cell, []):
                    raise KeyError(f"run {run_id} was never leased on cell {cell}")
            elif cur:
                run_id = cur["run_id"]  # an unfinished run (crashed holder or expired lease): resume it from its journal
            elif data["stopped"].get(cell):
                # (M17 e2e) a run stopped by a cap (budget, wall clock) is resumed from its journal by the next lease, not
                # replaced by a new run that repays every state the stopped one already settled
                run_id = data["stopped"][cell]
            else:
                n = len(data["runs"].get(cell, [])) + 1
                run_id = f"{cell}/{n:03d}"
                data["runs"].setdefault(cell, []).append(run_id)
            data["leases"][cell] = {"holder": holder, "run_id": run_id, "expires": now + self.ttl_s}
            data["stopped"].pop(cell, None)
            return Lease(cell, holder, run_id, now + self.ttl_s)

    def renew(self, lease: Lease) -> bool:
        with self._locked() as data:
            cur = data["leases"].get(lease.cell)
            if not cur or cur["holder"] != lease.holder or cur["run_id"] != lease.run_id:
                return False
            cur["expires"] = lease.expires = self.clock() + self.ttl_s
            return True

    def release(self, lease: Lease, *, filled: bool = False, stopped: bool = False) -> None:
        """End a lease; ``filled`` records that the cell's quota is met, so no holder leases it again; ``stopped`` that
        the run stopped on a cap and is the one the next lease resumes (M17 e2e)."""
        with self._locked() as data:
            cur = data["leases"].get(lease.cell)
            if cur and cur["holder"] == lease.holder and cur["run_id"] == lease.run_id:
                del data["leases"][lease.cell]
            if filled:
                data["filled"][lease.cell] = lease.run_id
            if stopped:
                data["stopped"][lease.cell] = lease.run_id

    def filled(self, cell: str) -> bool:
        with self._locked() as data:
            return bool(data["filled"].get(cell))

    def live(self) -> dict[str, dict[str, Any]]:
        now = self.clock()
        with self._locked() as data:
            return {k: v for k, v in data["leases"].items() if v["expires"] > now}

    def runs(self, cell: str) -> list[str]:
        with self._locked() as data:
            return list(data["runs"].get(cell, []))


# ---- persistent specialists --------------------------------------------------------------------------------------------
class Specialists:
    """Named specialists shared by every machine of a root, created on first use and kept (get-or-spawn by name).

    - ``hacker(family)``: a ``SpecialistHacker`` over the family's persistent ``FamilyMemory`` (confirmed hack
      strategies), with ``hacker_fallback(family, client)`` as its expensive search (None: memory only). A fresh hacker
      object per call (its per-search state is not shared between concurrent machines); the memory is shared.
    - ``get(kind, key)``: any other specialist (a World expert per World, the intent librarian), built by
      ``factories[kind](key)`` once.
    """

    def __init__(self, root: Path, *, hacker_fallback: Callable[[str, Any], Any] | None = None,
                 factories: Mapping[str, Callable[[str], Any]] | None = None):
        self.root = Path(root)
        self.hacker_fallback = hacker_fallback
        self.factories = dict(factories or {})
        self._memories: dict[str, FamilyMemory] = {}
        self._named: dict[tuple[str, str], Any] = {}

    def memory(self, family: str) -> FamilyMemory:
        if family not in self._memories:
            self._memories[family] = FamilyMemory(self.root / slug(family) / "hacker.jsonl")
        return self._memories[family]

    def hacker(self, family: str, *, client: Any = None) -> SpecialistHacker:
        fb = self.hacker_fallback(family, client) if self.hacker_fallback else None
        return SpecialistHacker(family, self.memory(family), fallback=fb)

    def remember_hacks(self, family: str, task_id: str, strategies: Sequence[str]) -> None:
        known = set(self.memory(family).strategies(family))
        new = [s for s in dict.fromkeys(strategies) if s not in known]
        if new:
            self.memory(family).remember(family, task_id, new)

    def get(self, kind: str, key: str) -> Any:
        if (kind, key) not in self._named:
            if kind not in self.factories:
                raise KeyError(f"no specialist factory for {kind!r}")
            self._named[(kind, key)] = self.factories[kind](key)
        return self._named[(kind, key)]

    def list(self) -> list[dict[str, str]]:
        return [{"kind": "hacker", "key": f} for f in sorted(self._memories)] + [{"kind": k, "key": v} for k, v in sorted(self._named)]


# ---- the root ----------------------------------------------------------------------------------------------------------
MakeRun = Callable[[Cell, str, Path], MachineRun]


@dataclass
class RootReport:
    stop_reason: str
    merged: list[str]
    deferred: list[str]
    rejected: list[str]
    escalated: list[str]
    stopped: list[str]
    usd: float
    max_parallel: int
    matrix: dict[str, Any]
    launches: list[dict[str, Any]]


class RootCurator:
    """Runs machines over the coverage matrix. ``make_run(cell, run_id, run_dir)`` builds a cell's machine (normally
    ``driver.task_run``); ``merge(card)`` takes a gate-accepted TaskCard (the publish step)."""

    def __init__(self, root: Path, matrix: CoverageMatrix, *, make_run: MakeRun, ledger: Ledger, book: YieldBook | None = None,
                 concurrency: int = 4, holder: str = "root", merge: Callable[[dict[str, Any]], Any] | None = None,
                 max_attempts_per_cell: int = 3, generator: Callable[[Cell], str] | None = None, lease_ttl_s: float = 3600.0,
                 max_surface_share: float = MAX_SURFACE_SHARE, reload_spend: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.matrix, self.make_run, self.ledger, self.concurrency, self.holder = matrix, make_run, ledger, concurrency, holder
        self.book = book if book is not None else YieldBook(path=self.root / "yield.jsonl")
        self.merge_fn = merge
        self.max_attempts = max_attempts_per_cell
        self.generator = generator or (lambda c: f"m17:{c.intent.get('id', c.behaviour)}")
        self.leases = LeaseBook(self.root / "leases", ttl_s=lease_ttl_s)
        self.max_surface_share = max_surface_share
        self.journal = self.root / "root.jsonl"
        self.decisions: dict[str, dict[str, Any]] = {}  # run id -> decision request (escalated machines)
        self.runs: dict[str, MachineRun] = {}
        self.cards: dict[str, dict[str, Any]] = {}
        self.out: dict[str, list[str]] = {"merged": [], "deferred": [], "rejected": [], "escalated": [], "stopped": []}
        self.launches: list[dict[str, Any]] = []
        self._active = 0
        self._max_active = 0
        self._cell_of: dict[str, str] = {}
        self.past_usd = self.reload_spend() if reload_spend else 0.0

    def reload_spend(self) -> float:
        """(M17 e2e) Book into the ledger what earlier processes of this root already spent (the ``spend`` events of
        every machine journal under ``runs/``): a restarted root started at 0 and could spend its whole cap again. A
        machine resumed later shares this ledger and does not book its past spend twice."""
        total = 0.0
        for j in sorted((self.root / "runs").rglob("journal.jsonl")) if (self.root / "runs").exists() else []:
            usd = 0.0
            for line in j.read_text().splitlines():
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "spend":
                    usd += float(ev.get("usd") or 0.0)
            if usd:
                self.ledger.settle(f"past:{j.parent.relative_to(self.root / 'runs').as_posix()}", usd)
                total += usd
        if total:
            self._log({"event": "reload_spend", "usd": round(total, 6)})
        return total

    def _log(self, ev: dict[str, Any]) -> None:
        with self.journal.open("a") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), **ev}, default=str) + "\n")

    # ---- what may launch ------------------------------------------------------------------------------------------
    def _launchable(self, cell: Cell, busy: set[str]) -> str | None:
        """None when ``cell`` may launch now, else why not."""
        if cell.key in busy:
            return "in flight"
        if cell.attempts >= self.max_attempts:
            return "attempts spent"
        why = self.book.paused(self.generator(cell))
        if why:
            return f"marginal yield: generator {self.generator(cell)} paused: {why}"
        accepted = sum(len(c.accepted) for c in self.matrix.cells.values())
        if accepted + 1 >= SURFACE_SHARE_FLOOR and self.matrix.surface_share(cell.surface) > self.max_surface_share + 1e-12:
            return f"diversity: surface {cell.surface} would exceed {self.max_surface_share:.0%} of accepted tasks"
        return None

    def _stop_reason(self, busy: set[str]) -> str | None:
        if self.ledger.exhausted:
            return "budget: the USD cap is spent"
        backlog = self.matrix.backlog()
        if not backlog:
            return "coverage: every cell is filled, escalated or closed"
        whys = {c.key: self._launchable(c, busy) for c in backlog}
        if all(w is not None for w in whys.values()) and not busy:
            return "no launchable cell: " + "; ".join(sorted({w for w in whys.values() if w}))[:800]
        return None

    # ---- one machine ------------------------------------------------------------------------------------------------
    async def _one(self, cell: Cell, lease: Any) -> tuple[Cell, Any, RunResult | None, str | None]:
        self._active += 1
        self._max_active = max(self._max_active, self._active)
        try:
            run = self.runs.get(lease.run_id) or self.make_run(cell, lease.run_id, self.root / "runs" / lease.run_id)
            self.runs[lease.run_id] = run
            return cell, lease, await run.run(), None
        except Exception as e:  # noqa: BLE001 -- one machine's crash never ends the root
            return cell, lease, None, f"{type(e).__name__}: {e}"[:500]
        finally:
            self._active -= 1

    def _settle(self, cell: Cell, lease: Any, res: RunResult | None, err: str | None) -> None:
        run_id = lease.run_id
        status = res.status if res is not None else "failed"
        gen = self.generator(cell)
        fam = str(cell.intent.get("id", cell.behaviour))
        usd = res.usd if res is not None else 0.0
        self._log({"event": "finish", "run_id": run_id, "cell": cell.key, "status": status, "reason": (res.reason if res else err), "usd": usd})
        if status == "paused":
            self.decisions[run_id] = res.decision_request or {}
            self._cell_of[run_id] = cell.key
            cell.status = "escalated"
            self.out["escalated"].append(run_id)
            self.leases.renew(lease)  # the cell stays leased while a person decides: no other root starts a duplicate
            return
        if status == "stopped":
            self.out["stopped"].append(run_id)
            # (M17 e2e) the run keeps its journal and the next lease on this cell resumes it (it was a new attempt, which
            # paid again for every settled state); a stopped run is not a finished attempt
            self.leases.release(lease, stopped=True)
            cell.attempts = max(0, cell.attempts - 1)
            return
        if status == "done":
            card = ((res.outputs.get("select") or {}).get("card")) if res else None
            gate = ((res.outputs.get("gate") or {}).get("verdict") or {}) if res else {}
            if not card or gate.get("ok") is not True or card.get("status") != "accepted":
                # never merge what the gate did not accept (a machine edited to skip it must not ship)
                self._log({"event": "refused_merge", "run_id": run_id, "why": "no gate-accepted card"})
                self.book.record_construction(run_id, gen, seed=cell.key, family=fam, state="rejected", cost_usd=usd)
                self.out["rejected"].append(run_id)
            else:
                self.book.record_construction(run_id, gen, seed=cell.key, family=fam, state="constructed", cost_usd=usd)
                self.book.record_acceptance(run_id, True, hacks=int(((card.get("verifier") or {}).get("hacks_confirmed")) or 0))
                admitted, why = self.book.admit(run_id)
                self.cards[run_id] = card
                if admitted:
                    self._merge(cell, run_id, card)
                else:
                    self.out["deferred"].append(run_id)
                    self._log({"event": "deferred", "run_id": run_id, "why": why})
        else:  # rejected (dropped by select, or a person dropped it) or failed
            self.book.record_construction(run_id, gen, seed=cell.key, family=fam, state="rejected" if status == "rejected" else "abandoned", cost_usd=usd)
            self.out["rejected"].append(run_id)
        self.leases.release(lease, filled=cell.status == "filled")

    def _merge(self, cell: Cell, run_id: str, card: dict[str, Any]) -> None:
        if self.merge_fn is not None:
            self.merge_fn(card)
        self.matrix.mark_accepted(cell.key, run_id)
        self.out["merged"].append(run_id)
        self._log({"event": "merge", "run_id": run_id, "cell": cell.key, "task": card.get("id")})

    # ---- the loop ---------------------------------------------------------------------------------------------------
    async def run(self) -> RootReport:
        inflight: dict[asyncio.Task, str] = {}
        stop: str | None = None
        while True:
            if not self.ledger.exhausted:
                for cell in self.matrix.backlog():
                    if len(inflight) >= self.concurrency or self.ledger.exhausted:
                        break
                    if self._launchable(cell, set(inflight.values())) is not None:
                        continue
                    lease = self.leases.acquire(cell.key, self.holder)
                    if lease is None:  # another root holds it, or filled it
                        if self.leases.filled(cell.key):
                            cell.status = "closed"
                        self._log({"event": "lease_busy", "cell": cell.key})
                        continue
                    cell.attempts += 1
                    self._cell_of[lease.run_id] = cell.key
                    self.launches.append({"cell": cell.key, "run_id": lease.run_id})
                    self._log({"event": "launch", "cell": cell.key, "run_id": lease.run_id})
                    inflight[asyncio.ensure_future(self._one(cell, lease))] = cell.key
            if not inflight:
                stop = self._stop_reason(set()) or "the remaining open cells are leased by another holder"
                break
            done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                inflight.pop(t)
                self._settle(*t.result())
        for rid in self.book.release():  # deferred tasks the quota now admits
            if rid in self.cards and rid in self.out["deferred"]:
                self.out["deferred"].remove(rid)
                self._merge(self.matrix.cells[self._cell_of[rid]], rid, self.cards[rid])
        self._log({"event": "stop", "reason": stop})
        return self.report(stop or "done")

    async def answer(self, run_id: str, decision: Mapping[str, Any]) -> RunResult:
        """Resume an escalated machine with a person's (or a compiled gate's) decision; its outcome is settled like any
        other (merged only if gate-accepted)."""
        if run_id not in self.decisions:
            raise KeyError(f"no escalation pending for {run_id}")
        cell = self.matrix.cells[self._cell_of[run_id]]
        lease = self.leases.acquire(cell.key, self.holder, run_id=run_id)
        if lease is None:
            raise RuntimeError(f"cell {cell.key} is leased by another holder; answer {run_id} later")
        self._log({"event": "decision", "run_id": run_id, "decision": dict(decision)})
        del self.decisions[run_id]
        self.out["escalated"].remove(run_id)
        cell.status = "open" if len(cell.accepted) < cell.quota else "filled"
        res = await self.runs[run_id].resume(decision)
        self._settle(cell, lease, res, None)
        return res

    def report(self, stop_reason: str) -> RootReport:
        return RootReport(stop_reason, list(self.out["merged"]), list(self.out["deferred"]), list(self.out["rejected"]),
                          list(self.out["escalated"]), list(self.out["stopped"]), round(self.ledger.committed, 6), self._max_active,
                          self.matrix.view(), list(self.launches))

    def as_dict(self, rep: RootReport) -> dict[str, Any]:
        return {**asdict(rep), "yield": self.book.report()["total"], "decisions": self.decisions}
