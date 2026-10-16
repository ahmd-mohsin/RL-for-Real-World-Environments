"""M17 end to end on CPU: the executable lane through the whole task factory with real OpenRouter models.

    python scripts/factory_run.py --work <dir> --out docs/research/task-factory/runs --cap 20 <phase> [<phase> ...]

Phases (each resumable; every paid call is booked on one forge Ledger whose spend persists in ``<work>/spend.jsonl``
and in the machine journals, so a restarted phase never spends the cap twice):

- ``mine``      mine fix + test commits from the local clones under ``<work>/repos`` and validate them in Docker;
- ``controls``  build the seeded auditor controls from known cases (executed) and pre-test the auditor on them;
- ``discover``  author a request per mined task, run the solver rungs, audit every failure (05 §4 steps 1-2);
- ``intents``   cluster the skill-gap failures into intent candidates (>= 3 traces, >= 2 model families) and write
                IntentCards with status ``candidate``; the decision store records each as pending a person's approval;
- ``family``    run the per-task machine (``RootCurator`` over ``ExecLane.task_run``) on the top candidate's tasks with
                ``provisional_intent=True``;
- ``accept``    family acceptance (05 §8): gold set per class, orthogonality, transfer proxy on a held-out repository;
- ``requests``  everything only a person can answer, queued in the human-labelling intake (the agentenv explorer's label
                routes): intent approvals, gold-set truth for real end states, escalations, the auditor's 10% sample;
- ``report``    the funnel per stage, spend per stage and role, wall time.

Raw results go to ``--out``; heavy artifacts (start trees, journals, caches) stay under ``--work``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from crucible.actors.llm import LLMOutputError
from crucible.factory.audit import Auditor, Trace, audit
from crucible.factory.cards import IntentCard, ObservationRecord
from crucible.factory.lanes.executable import (
    REGRESSION_REQ,
    ControlSpec,
    DockerSandbox,
    ExecLane,
    LaneStore,
    build_control,
    mine,
    regression_pool,
    sha256_text,
)
from crucible.factory.models import ModelBook, booked, family_of
from crucible.forge.budget import Ledger

FAMILY_REPOS = ("python-semver", "parse", "boltons", "cachetools", "humanize", "more-itertools")
HELD_OUT_REPO = "python-tabulate"  # the transfer probes' surface: never used by discovery or the family
INTENT_DISCOVERY = {"id": "discovery", "behaviour": "make the change the user reports in a Python library", "failure_mode": "unknown",
                    "difficulty_axes": []}


# ---- the run: ledger, spend journal, lane, models ---------------------------------------------------------------------
class Run:
    def __init__(self, work: Path, out: Path, cap: float):
        self.work, self.out, self.cap = work, out, cap
        self.out.mkdir(parents=True, exist_ok=True)
        self.spend_path = work / "spend.jsonl"
        self.book = ModelBook.from_config(budget_usd_per_client=cap)
        self.ledger = Ledger(cap_usd=cap)
        past = self.past_spend()
        if past:
            self.ledger.settle("past:phases", past)
        self.store = LaneStore(work / "lane")
        self.sandbox = DockerSandbox(parallel=8)
        self.sandbox.ensure_image()
        controls = json.loads((out / "controls.json").read_text())["bank"] if (out / "controls.json").exists() else []
        self.lane = ExecLane(self.store, self.sandbox, repo_dirs={p.name: str(p) for p in (work / "repos").iterdir() if p.is_dir()},
                             reviewer=self.book.client("reviewer"), control_bank=controls, gold_path=work / "gold.jsonl")
        self.t0 = time.time()

    def past_spend(self) -> float:
        total = 0.0
        if self.spend_path.exists():
            for line in self.spend_path.read_text().splitlines():
                if line.strip():
                    total += float(json.loads(line).get("usd") or 0.0)
        return total + _machine_spend(self.work)

    def phase_spend(self) -> float:
        """Spend outside the machines (the machines' own spend is in their journals: ``RootCurator.reload_spend``)."""
        return self.past_spend() - _machine_spend(self.work)

    @asynccontextmanager
    async def spend(self, phase: str, role: str, who: str, estimate: float = 0.0) -> Any:
        scope = None
        try:
            async with booked(self.ledger, f"{phase}/{role}/{who}", estimate_usd=estimate) as scope:
                yield scope
        finally:
            usd = scope.usd if scope is not None else 0.0
            with self.spend_path.open("a") as fh:
                fh.write(json.dumps({"ts": round(time.time(), 2), "phase": phase, "role": role, "who": who, "usd": round(usd, 8)}) + "\n")

    def committed(self) -> float:
        return round(self.ledger.committed, 6)

    def write(self, name: str, data: Any) -> Path:
        p = self.out / name
        p.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
        return p


def phase_tasks(run: Run, repos: tuple[str, ...], per_repo: int) -> list[Any]:
    """``per_repo`` mined tasks of each repository, interleaved (round robin) so a partial run covers every surface."""
    lists = [[run.store.get(i) for i in sorted(i for i in run.store.ids() if run.store.get(i).repo == r)][:per_repo] for r in repos]
    return [ts[k] for k in range(max(map(len, lists), default=0)) for ts in lists if k < len(ts)]


# ---- mine -------------------------------------------------------------------------------------------------------------
async def phase_mine(run: Run, repos: list[str], per_repo: int) -> None:
    log = (run.work / "mine_log.jsonl").open("a")
    funnel = {}
    for r in repos:
        if any(run.store.get(i).repo == r for i in run.store.ids()):
            continue  # already mined (resumable)
        f = await mine({r: run.work / "repos" / r}, run.store, DockerSandbox(guard=False, parallel=8), per_repo=per_repo, reruns=3,
                       since="2016-01-01", log=lambda ev: log.write(json.dumps(ev) + "\n"))
        funnel.update(f)
    prev = run.work / "mine_funnel.jsonl"
    if prev.exists():
        for line in prev.read_text().splitlines():
            funnel.update({k: v for k, v in json.loads(line).items() if k not in funnel})
    run.write("mine_funnel.json", funnel)


# ---- controls ---------------------------------------------------------------------------------------------------------
PROMPT_LIST = ("humanize.natural_list() crashes when it is given an empty list. natural_list([]) must return an empty string. Lists with "
               "items must keep their current output, for example natural_list(['one', 'two', 'three']) returns 'one, two and three'.")
PROMPT_SS = ("boltons.strutils.singularize() mangles words that are already singular and end in a double s: singularize('glass') returns "
             "'glas' and singularize('boss') returns 'bos'. Words ending in 'ss' that are already singular (glass, boss, class, kiss, "
             "address, business) must be returned unchanged, keeping the caller's capitalization (singularize('Glass') returns 'Glass'). "
             "Real plurals must still be singularized: singularize('glasses') returns 'glass'.")
PROMPT_GUID = ("boltons.iterutils.GUIDerator(size=20) works, but the ValueError raised for an out-of-range size says 20 is not allowed. "
               "The error for a size below 20 or above 36 must state the real allowed range: 20 to 36, both inclusive. Valid sizes must "
               "keep working.")
_LIST_OLD = "        str: A string with commas and 'and' in the right places.\n    \"\"\"\n    if len(items) == 1:\n"


def control_specs(store: LaneStore) -> list[ControlSpec]:
    """Known cases, executed for real (Q's open item: an auditor with no seeded controls is untested and cannot gate)."""
    hl, bs, gd = store.get("humanize-0a06a3d4a1"), store.get("boltons-1e61524a79"), store.get("boltons-f5f5cd867e")
    guid_msg = {"edits": [{"path": "boltons/iterutils.py", "old": "            raise ValueError('expected 20 < size <= 36')\n",
                           "new": "            raise ValueError('size must be between 20 and 36, both inclusive')\n"}]}
    lazy = {"edits": [{"path": "src/humanize/lists.py", "old": _LIST_OLD,
                       "new": "        str: A string with commas and 'and' in the right places.\n\n    Note: empty lists are not supported yet.\n"
                              "    \"\"\"\n    if len(items) == 1:\n"}]}
    return [
        ControlSpec("ctl-world-bug", hl.id, "world_bug", PROMPT_LIST, hl.oracle, None, corrupt="break:src/humanize/time.py",
                    why="the real fix, graded in a World where a module the change never touches is truncated: the package cannot import"),
        ControlSpec("ctl-lazy", hl.id, "skill_gap", PROMPT_LIST, lazy, False, why="a lazy change: a docstring note, no behaviour change"),
        ControlSpec("ctl-verifier-defect", bs.id, "verifier_defect", PROMPT_SS, bs.oracle, True,
                    test_edits=[("tests/test_strutils.py", "assert singularize('glass') == 'glass'", "assert singularize('glass') == 'glas'")],
                    why="the real fix, graded by a test whose expectation contradicts the request"),
        ControlSpec("ctl-unstated-format", gd.id, "unstated_format", PROMPT_GUID, guid_msg, None,
                    why="the message now states the real inclusive range, but in other words than the exact text the test matches; "
                        "the request never states the wording", gating=False),
    ]


async def phase_controls(run: Run, reps: int) -> None:
    bank = [await build_control(s, run.store, run.sandbox) for s in control_specs(run.store)]
    auditor = Auditor(run.book.client("auditor"), model=run.book.spec("auditor").ref, max_tokens=12000)
    async def one(c: dict[str, Any], k: int) -> dict[str, Any]:
        tr = Trace(f"{c['id']}#{k}", c["text"] + ("" if k == 0 else f"\n(review {k})"), False, policy="control", failing_checks=["c1"])
        async with run.spend("controls", "auditor", f"{c['id']}#{k}"):
            lab = await auditor.label(tr, prompt=c["prompt"])
        return {"control": c["id"], "rep": k, "expected": c["cause"], "got": lab.cause, "solution_correct": lab.solution_correct,
                "status": lab.status, "why": lab.why, "rationale": lab.rationale, "evidence": lab.evidence,
                "ok": lab.cause == c["cause"] and (c["solution_correct"] is None or lab.solution_correct == c["solution_correct"])}

    rows = list(await asyncio.gather(*[one(c, k) for c in bank for k in range(reps)]))
    for c in bank:
        c.pop("results", None)
    run.write("controls.json", {"bank": bank, "pretest": rows, "recall": [sum(r["ok"] for r in rows), len(rows)],
                                "spend_usd": run.committed()})


# ---- discover ---------------------------------------------------------------------------------------------------------
def default_checkset(t: Any, spec: dict[str, Any]) -> dict[str, Any]:
    obs = t.observation["id"]
    reqs = [r["id"] for r in spec["requirements"] if r["id"] != REGRESSION_REQ]
    checks = [{"id": "c1", "requirement_ids": reqs, "observation_ids": [obs], "gating": True, "spec": {"tests": list(t.f2p)}}]
    pool = regression_pool(t)
    if pool and any(r["id"] == REGRESSION_REQ for r in spec["requirements"]):
        checks.append({"id": "c2", "requirement_ids": [REGRESSION_REQ], "observation_ids": [obs], "gating": True, "spec": {"tests": pool}})
    return {"checks": checks}


async def author_draft(run: Run, t: Any, phase: str, *, tries: int = 3) -> dict[str, Any]:
    """The lane's spec and author handlers outside a machine (discovery and the transfer probes need requests too)."""
    from crucible.factory.machine import StateContext

    h = run.lane.handlers()
    cell = {"key": t.id, "surface_spec": {"task": t.id, "repo": t.repo}}
    outs: dict[str, Any] = {}
    for state, role, inputs in (("spec", "planner", {"intent": INTENT_DISCOVERY, "cell": cell}),
                                ("author", "author", None)):
        last: Exception | None = None
        for k in range(1, tries + 1):
            ctx = StateContext(f"{phase}/{t.id}", state, 1, k, inputs or {"intent": INTENT_DISCOVERY, "cell": cell, "spec": outs["spec"]}, [],
                               {"cell": cell}, role, run.book.spec(role), run.book.client(role))
            try:
                async with run.spend(phase, role, t.id):
                    outs.update((await h[state](ctx)).outputs)
                last = None
                break
            except (LLMOutputError, KeyError, TypeError, ValueError) as e:
                last = e
        if last is not None:
            raise RuntimeError(f"{state} failed after {tries} tries: {last}")
    return {"task_id": f"{phase}/{t.id}", "world": outs["world"], "spec": outs["spec"], "spans": outs["spans"], "prompt": outs["prompt"],
            "checkset": default_checkset(t, outs["spec"]), "cell": cell}


async def run_solvers(run: Run, task: Any, phase: str, attempts: int, *, preamble: str = "", salt: str = "") -> list[dict[str, Any]]:
    async def rung(spec: Any, client: Any) -> list[dict[str, Any]]:
        async with run.spend(phase, f"solver:{spec.model}", task.mined.id):
            tr = await run.lane.solver_runner(spec.model, client, preamble=preamble)(task, attempts, salt=salt)
        return [{"id": f"{spec.model}#{i}", "policy": spec.model, **x} for i, x in enumerate(tr)]

    out = []
    for xs in await asyncio.gather(*[rung(s, c) for s, c in run.book.solver_clients()]):
        out += xs
    return out


async def audit_traces(run: Run, phase: str, draft: dict[str, Any], traces: list[dict[str, Any]], salt: str) -> Any:
    from crucible.factory.driver import ledger_of

    ts = [Trace(t["id"], t["text"], t["passed"], policy=t["policy"], failing_checks=list(t.get("failing_checks") or []), meta=dict(t.get("meta") or {}))
          for t in traces]
    async with run.spend(phase, "auditor", salt):
        return await audit(ts, Auditor(run.book.client("auditor"), model=run.book.spec("auditor").ref, max_tokens=12000), prompt=draft["prompt"],
                           requirements=list(ledger_of(draft).requirements), seeded=run.lane.seeded_controls(), salt=salt)


async def discover_one(run: Run, t: Any, attempts: int, sem: asyncio.Semaphore) -> dict[str, Any]:
    async with sem:
        t0 = time.time()
        try:
            draft = await author_draft(run, t, "discover")
        except Exception as e:  # noqa: BLE001 - a task the author cannot state is a recorded drop
            return {"task": t.id, "repo": t.repo, "dropped": f"author: {e}"[:400]}
        task = run.lane.materialize(draft)
        traces = await run_solvers(run, task, "discover", attempts)
        rep = await audit_traces(run, "discover", draft, traces, salt=t.id)
        labels = {tid: lab.__dict__ for tid, lab in rep.labels.items()}
        return {"task": t.id, "repo": t.repo, "prompt": draft["prompt"], "requirements": draft["spec"]["requirements"],
                "traces": [{k: v for k, v in x.items() if k != "solution"} | {"solution": x.get("solution"), "label": labels.get(x["id"])} for x in traces],
                "seeded": rep.seeded, "auditor_trusted": rep.auditor_trusted, "mix": {k: v.to_dict() for k, v in rep.mix.items()},
                "seconds": round(time.time() - t0, 1)}


async def phase_discover(run: Run, per_repo: int, attempts: int, parallel: int, stop_at: float) -> None:
    path = run.work / "discovery.jsonl"
    done = {r["task"] for r in (json.loads(x) for x in path.read_text().splitlines()) if "dropped" not in r} if path.exists() else set()
    tasks = [t for t in phase_tasks(run, FAMILY_REPOS, per_repo) if t.id not in done]
    sem = asyncio.Semaphore(parallel)

    async def go(t: Any) -> None:
        if run.committed() > stop_at:
            return
        row = await discover_one(run, t, attempts, sem)
        with path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(f"discover {t.id}: {'dropped' if 'dropped' in row else 'ok'} {row.get('seconds', '')}s committed {run.committed():.4f}", flush=True)

    await asyncio.gather(*[go(t) for t in tasks])
    rows = discovery_rows(path)
    run.write("discovery_summary.json", summarize_discovery(rows) | {"spend_usd": run.committed()})


def discovery_rows(path: Path) -> list[dict[str, Any]]:
    """The last row per task (a dropped task that was retried keeps only its retry)."""
    rows: dict[str, dict[str, Any]] = {}
    for x in path.read_text().splitlines():
        r = json.loads(x)
        rows[r["task"]] = r
    return list(rows.values())


def summarize_discovery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # pass, fail, masked
    causes: Counter = Counter()
    seeded = [0, 0]
    trusted = Counter()
    per_task = {}
    for r in rows:
        if "dropped" in r:
            continue
        trusted[str(r.get("auditor_trusted"))] += 1
        for s in r.get("seeded") or []:
            seeded[0] += bool(s["ok"])
            seeded[1] += 1
        pt = defaultdict(list)
        for x in r["traces"]:
            st = by_policy[x["policy"]]
            st[0 if x["passed"] is True else 1 if x["passed"] is False else 2] += 1
            pt[x["policy"]].append(x["passed"])
            if x["passed"] is not True:
                causes[(x.get("label") or {}).get("cause") or "unlabelled"] += 1
        per_task[r["task"]] = {p: [v.count(True), len(v)] for p, v in pt.items()}
    return {"tasks": len(rows), "dropped": [r for r in rows if "dropped" in r], "by_policy": {p: {"pass": v[0], "fail": v[1], "masked": v[2]}
                                                                                               for p, v in by_policy.items()},
            "failure_causes": dict(causes), "seeded_recall": seeded, "auditor_trusted_tasks": dict(trusted), "per_task": per_task}


# ---- intents ----------------------------------------------------------------------------------------------------------
DESCRIBE = ("Each item is a failed attempt by a model to make a requested change in a Python library, with the request and the "
            "test failure. For each item, name in at most 12 words the GENERAL behaviour the attempt got wrong, phrased so it "
            "applies across libraries (for example 'handles empty or boundary inputs', 'preserves existing behaviour when "
            "extending an API', 'validates arguments and raises the stated error'). Reply with JSON {\"items\": [{\"id\", "
            "\"behaviour\"}]}.")
CLUSTER = ("Group these failure behaviours into at most 8 clusters of the same underlying skill. Reply with JSON {\"clusters\": "
           "[{\"name\": \"<kebab-case id>\", \"behaviour\": \"<one sentence: the behaviour policies get wrong>\", \"failure_mode\": "
           "\"<one sentence>\", \"difficulty_axes\": [\"<what makes it genuinely harder>\"], \"members\": [<ids>]}]}. Every id belongs "
           "to exactly one cluster.")


async def phase_intents(run: Run) -> None:
    from crucible.factory.labels import LabelStore

    rows = discovery_rows(run.work / "discovery.jsonl")
    fails = []
    for r in rows:
        if "dropped" in r or r.get("auditor_trusted") is False:
            continue
        for x in r["traces"]:
            if (x.get("label") or {}).get("cause") == "skill_gap":
                fails.append({"id": f"{r['task']}|{x['id']}", "task": r["task"], "repo": r["repo"], "policy": x["policy"], "request": r["prompt"][:1200],
                              "failure": x["text"][-1800:]})
    described: dict[str, str] = {}
    client = run.book.client("reviewer")
    for i in range(0, len(fails), 12):
        chunk = fails[i:i + 12]
        async with run.spend("intents", "reviewer", f"describe-{i}"):
            out = await client.complete_json([{"role": "system", "content": DESCRIBE},
                                              {"role": "user", "content": json.dumps([{k: f[k] for k in ("id", "request", "failure")} for f in chunk])}],
                                             max_tokens=6000)
        for it in out.get("items") or []:
            if isinstance(it, dict) and it.get("id") in {f["id"] for f in chunk}:
                described[it["id"]] = str(it.get("behaviour") or "")[:160]
    async with run.spend("intents", "reviewer", "cluster"):
        cl = await client.complete_json([{"role": "system", "content": CLUSTER},
                                         {"role": "user", "content": json.dumps([{"id": k, "behaviour": v} for k, v in described.items()])}], max_tokens=12000)
    meta = {f["id"]: f for f in fails}
    store = LabelStore(run.work / "decisions.jsonl")
    cards, clusters = [], []
    for c in cl.get("clusters") or []:
        members = [m for m in c.get("members") or [] if m in meta]
        fams = sorted({family_of(meta[m]["policy"]) for m in members})
        repos = sorted({meta[m]["repo"] for m in members})
        tasks = sorted({meta[m]["task"] for m in members})
        name = re.sub(r"[^a-z0-9-]+", "-", str(c.get("name") or "cluster").lower()).strip("-")[:60] or "cluster"
        candidate = len(members) >= 3 and len(fams) >= 2
        row = {"name": name, "behaviour": c.get("behaviour"), "traces": len(members), "families": fams, "repos": repos, "tasks": tasks,
               "candidate": candidate, "members": members}
        clusters.append(row)
        if not candidate:
            continue
        obs = []
        for m in members:
            f = meta[m]
            rec = ObservationRecord(id=f"obs-fail-{sha256_text(m)[:16]}", type="observed_model_failure", pointer=f"trace:discover/{m}",
                                    sha256=sha256_text(f["failure"]), created=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                                    by=f"model:{f['policy']}")
            obs.append(rec.model_dump(mode="json"))
        card = IntentCard(id=f"intent-{name}", behaviour=str(c.get("behaviour") or name), failure_mode=str(c.get("failure_mode") or c.get("behaviour") or name),
                          evidence=[o["id"] for o in obs], difficulty_axes=[str(a) for a in c.get("difficulty_axes") or []][:6],
                          coverage={"behaviour": name, "surface": "python-library-change"}, status="candidate")
        prev = store.latest_decision(card.id)
        if prev is None:
            store.add_decision(subject=card.id, actor="model:intent-librarian", action="keep", criteria=["pending_human_approval"],
                               reason=f"candidate from {len(members)} skill-gap traces, {len(fams)} model families, {len(repos)} repositories; "
                                      "awaiting a person's approve or reject", recipe_version="m17-v1")
        cards.append({"card": card.model_dump(mode="json"), "observations": obs, "tasks": tasks, "repos": repos, "families": fams})
    cards.sort(key=lambda c: (-len(c["repos"]), -len(c["card"]["evidence"])))
    run.write("intents.json", {"skill_gap_traces": len(fails), "described": len(described), "clusters": clusters, "cards": cards,
                               "decisions": [d.model_dump(mode="json") for d in store.decisions()], "spend_usd": run.committed()})


# ---- family -----------------------------------------------------------------------------------------------------------
async def phase_family(run: Run, *, slots: int, max_attempts: int, concurrency: int, intent_index: int, target: str | None,
                       stop_at: float, root_name: str = "root", only: list[str] | None = None, max_author: int = 2, max_verifier: int = 2) -> None:
    from crucible.factory.driver import driver_machine
    from crucible.factory.root import Cell, CoverageMatrix, RootCurator, Specialists

    if only:  # a trial on named tasks under a placeholder intent (never merged into the family's report)
        card = IntentCard(id="intent-trial", behaviour="make the change the user reports", failure_mode="trial", coverage={"behaviour": "trial",
                          "surface": "python-library-change"}).model_dump(mode="json")
        chosen = {"card": card, "tasks": only}
    else:
        intents = json.loads((run.out / "intents.json").read_text())
        chosen = intents["cards"][intent_index]
        card = chosen["card"]
    disc = {r["task"]: r for r in discovery_rows(run.work / "discovery.jsonl")}
    tgt = target or "openai/gpt-oss-20b"

    def mixed(tid: str) -> int:  # prefer tasks the target neither always passed nor always failed in discovery
        tr = [x for x in disc.get(tid, {}).get("traces", []) if x["policy"] == tgt]
        k = sum(1 for x in tr if x["passed"] is True)
        return 0 if 0 < k < len(tr) else 1

    by_repo: dict[str, list[str]] = defaultdict(list)
    for tid in sorted(chosen["tasks"], key=lambda x: (mixed(x), x)):
        by_repo[run.store.get(tid).repo].append(tid)
    rootdir = run.work / root_name
    assign_path = rootdir / "assign.json"
    assign_path.parent.mkdir(parents=True, exist_ok=True)
    assign: dict[str, str] = json.loads(assign_path.read_text()) if assign_path.exists() else {}
    cells = []
    for repo, tids in sorted(by_repo.items()):
        for k in range(min(slots, len(tids))):
            cells.append(Cell(behaviour=f"{card['coverage']['behaviour']}-s{k + 1}", surface=repo, intent=card,
                              surface_spec={"repo": repo, "pool": tids}, quota=1))
    matrix = CoverageMatrix(cells)
    specialists = Specialists(rootdir / "specialists")
    env = run.lane.env(run.book, family=card["id"], target=tgt, specialists=specialists)
    cards_dir = rootdir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    def make_run(cell: Any, run_id: str, run_dir: Path) -> Any:
        if run_id not in assign:
            used = set(assign.values())
            free = [x for x in cell.surface_spec["pool"] if x not in used]
            if not free:
                raise RuntimeError(f"no unassigned task left in {cell.surface}")
            assign[run_id] = free[0]
            assign_path.write_text(json.dumps(assign, indent=1, sort_keys=True))
        tid = assign[run_id]
        param = {**cell.as_param(), "surface_spec": {"task": tid, "repo": cell.surface}}
        return run.lane.task_run(run_dir=run_dir, intent=card, cell=param, env=env, clients=run.book.clients(), models=run.book.roles,
                                 ledger=run.ledger, run_id=run_id,
                                 machine=driver_machine(max_spec=2, max_author=max_author, max_verifier=max_verifier, budget_s=5400))

    def merge(c: dict[str, Any]) -> None:
        (cards_dir / f"{c['id'].replace('/', '__')}.json").write_text(json.dumps(c, indent=1, default=str))

    # the machines and the root share one ledger capped at --stop-at: what the phases spent is settled here, what earlier
    # machine runs spent is booked by the root from their journals (RootCurator.reload_spend, the M17 e2e fix)
    run.ledger = Ledger(cap_usd=stop_at)
    own = sum(float(json.loads(x).get("usd") or 0.0) for j in rootdir.glob("runs/**/journal.jsonl") for x in j.read_text().splitlines()
              if x.strip() and json.loads(x).get("event") == "spend")
    run.ledger.settle("past:phases-and-other-roots", run.past_spend() - own)
    root = RootCurator(rootdir, matrix, make_run=make_run, ledger=run.ledger, concurrency=concurrency, holder="factory-run", merge=merge,
                       max_attempts_per_cell=max_attempts)
    rep = await root.run()
    out = root.as_dict(rep)
    out["assign"] = assign
    out["intent"] = card["id"]
    out["provisional_intent"] = True
    out["target"] = tgt
    out["spend_usd"] = run.ledger.committed
    out["config"] = {"slots": slots, "max_attempts_per_cell": max_attempts, "concurrency": concurrency, "max_author": max_author,
                     "max_verifier": max_verifier, "root": root_name}
    run.write("family_run.json" if root_name == "root" else f"{root_name}_run.json", out)


def _machine_spend(work: Path) -> float:
    """What every machine under every root of this work directory booked (``<work>/<root>/runs/**/journal.jsonl``)."""
    total = 0.0
    for j in work.glob("*/runs/**/journal.jsonl"):
        for line in j.read_text().splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "spend":
                total += float(ev.get("usd") or 0.0)
    return total



# ---- accept -----------------------------------------------------------------------------------------------------------
def journal_outputs(run_dir: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for line in (run_dir / "journal.jsonl").read_text().splitlines():
        ev = json.loads(line)
        if ev.get("event") == "settle":
            latest[ev["state"]] = ev["outputs"]
        if ev.get("event") == "start":
            latest["_params"] = ev.get("params") or {}
    return latest


def final_draft(run_dir: Path) -> dict[str, Any]:
    o = journal_outputs(run_dir)
    p = o["_params"]
    return {"task_id": run_dir.relative_to(run_dir.parents[1]).as_posix(), "world": o["author"]["world"], "spec": o["spec"]["spec"],
            "spans": o["author"]["spans"], "prompt": o["author"]["prompt"], "checkset": o["verifier"]["checkset"], "cell": p.get("cell"), "outputs": o}


async def phase_accept(run: Run, *, probes_n: int, attempts: int, k_exemplars: int, target: str | None) -> None:
    from crucible.factory.family import GoldRecord, accept_family, load_gold
    from crucible.factory.floors import graded_from
    from crucible.factory.labels import LabelStore
    from crucible.factory.transfer import ProbeTask, TransferConfig, disjoint, exemplar_block, transfer_proxy

    fr = json.loads((run.out / "family_run.json").read_text())
    tgt = target or fr.get("target") or "openai/gpt-oss-20b"
    cards = [json.loads((run.work / "root" / "cards" / f"{rid.replace('/', '__')}.json").read_text()) for rid in fr["merged"]]
    extra, finals, exemplars = {}, {}, []
    gold_extra: list[GoldRecord] = []
    for c in cards:
        rd = run.work / "root" / "runs" / c["id"]
        d = final_draft(rd)
        task = run.lane.materialize({**d, "task_id": c["id"]})
        finals[c["id"]] = task
        o = d["outputs"]
        traces = (o.get("solvers") or {}).get("runs", {}).get("traces") or []
        causes = Counter(x.get("cause") for x in ((o.get("audit") or {}).get("verdict") or {}).get("measured", {}).get("causes") or [])
        extra[c["id"]] = {"surface": {"repo": task.mined.repo, "domain": "python library"}, "start_fp": sha256_text(task.mined.parent),
                          "end_fp": task.end_fingerprint(task.oracle), "footprint": sorted(task.mined.src_files),
                          "checks": sorted({x for xs in task.check_tests.values() for x in xs}),
                          "passes": {f"{x['policy']}#{i % 4}": x.get("passed") for i, x in enumerate(traces)}, "causes": dict(causes)}
        for x in traces:
            if x.get("passed") is True and x["policy"] == tgt and len(exemplars) < 50:
                exemplars.append(f"Request:\n{task.instruction}\n\nAccepted change:\n{json.dumps(x.get('solution'))[:2500]}")
        for n in task.negatives:  # the deterministic constructed items of the accepted task, graded now
            if n.kind in ("defect_mutant", "valid_alternative"):
                g = graded_from(await task.grade(task.solution_of(n)))
                gold_extra.append(GoldRecord(task.task_id, "alternative" if n.expected == "pass" else "mutant", task.solution_of(n), g.reward, dict(g.checks),
                                             proven_by=f"negative:{n.id}"))
        for m in task.mutants():
            g = graded_from(await task.grade(m.solution))
            gold_extra.append(GoldRecord(task.task_id, "mutant", m.solution, g.reward, dict(g.checks), proven_by=f"mutant:{m.id}"))
    keep = {t.task_id for t in finals.values()}
    gold = [r for r in load_gold(run.work / "gold.jsonl") if r.task in keep] + gold_extra
    # transfer proxy on the held-out repository
    probe_tasks = [run.store.get(i) for i in run.store.ids() if run.store.get(i).repo == HELD_OUT_REPO][:probes_n]
    probes, probe_drafts = [], {}
    for t in probe_tasks:
        try:
            d = await author_draft(run, t, "transfer")
        except Exception as e:  # noqa: BLE001
            print("probe author failed", t.id, e, file=sys.stderr)
            continue
        probe_drafts[t.id] = d
        probes.append(ProbeTask(t.id, d["prompt"], {"repo": t.repo}, world=t.repo, template="lane-author", verifier=f"tests:{t.repo}"))
    fam_probe = [ProbeTask(c["id"], c["prompt"], {"repo": extra[c["id"]]["surface"]["repo"]}, world=extra[c["id"]]["surface"]["repo"],
                           template="lane-author", verifier=f"tests:{extra[c['id']]['surface']['repo']}") for c in cards]
    problems = disjoint(probes, fam_probe, held_out_axis="repo")  # probes and family share the lane's authoring template: reported
    client = next(c for s, c in run.book.solver_clients() if s.model == tgt)
    block = exemplar_block(exemplars, k_exemplars, header="Solved examples of related requests in other libraries (request, then the accepted change):")
    cache: dict[tuple[str, str], list[bool | None]] = {}

    async def arm(name: str, preamble: str) -> Any:
        async def runner(p: Any, a: int) -> bool | None:
            if (name, p.id) not in cache:
                task = run.lane.materialize(probe_drafts[p.id])
                async with run.spend("transfer", f"solver:{tgt}:{name}", p.id):
                    tr = await run.lane.solver_runner(tgt, client, preamble=preamble)(task, attempts, salt=f"transfer-{name}")
                cache[(name, p.id)] = [x["passed"] for x in tr]
            return cache[(name, p.id)][a]

        return runner

    tp = None
    if exemplars and probes:
        tp = await transfer_proxy(probes, await arm("base", ""), await arm("treated", block), family=fam_probe, config=TransferConfig(attempts=attempts))
        tp["exemplars"] = {"available": len(exemplars), "k": k_exemplars, "policy": tgt}
    elif not exemplars:
        tp = {"ok": False, "reasons": [f"no solved family trace by the target {tgt} to put in context"], "probes": len(probes)}
    labels = LabelStore(run.work / "factory" / "labels.jsonl")  # what people answered through the label routes (phase requests)
    res = accept_family(cards=cards, extra=extra, gold=gold, labels=labels.labels(), transfer=tp, probes=probes, probe_problems=problems,
                        provisional_intent=True)
    res["spend_usd"] = run.committed()
    run.write("family_acceptance.json", res)


# ---- requests: what only a person can answer ---------------------------------------------------------------------------
def _safe_id(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:~/-]+", "-", x).strip("-/")[:240]


def phase_requests(run: Run) -> None:
    """Every judgement this run could not make itself becomes an item in the human-labelling intake (the agentenv
    explorer's ``/api/v1/crucible/labels`` routes, ``interop/agentenv/plugin/labels.py``, with ``data_root = <work>``):

    - intent cards awaiting approval (``approve`` / ``reject``);
    - real solver end states of accepted tasks, the gold set's truth (``pass`` / ``fail``, blind: the verifier's verdict
      is hidden); without them the family's gold set cannot pass;
    - escalated machine runs (``revise`` / ``drop``), each with its decision request;
    - the 10% human sample of the auditor's failure causes (blind; the closed FailureCause set).

    Answers come back as labels in ``<work>/factory/labels.jsonl``; ``accept`` reads them (gold-set truth, kappa)."""
    from crucible.factory.audit import CAUSES, human_sample
    from crucible.factory.family import gold_items, load_gold

    items: list[dict[str, Any]] = []
    intents = json.loads((run.out / "intents.json").read_text()) if (run.out / "intents.json").exists() else {"cards": []}
    for c in intents["cards"]:
        items.append({"id": f"intent:{c['card']['id']}", "family": "intent-approval", "choices": ["approve", "reject"],
                      "payload": {"card": c["card"], "repos": c["repos"], "families": c["families"], "tasks": c["tasks"][:20]}})
    fr = json.loads((run.out / "family_run.json").read_text()) if (run.out / "family_run.json").exists() else {}
    for rid, req in (fr.get("decisions") or {}).items():
        items.append({"id": _safe_id(f"escalation:{rid}"), "family": "escalations", "choices": ["revise", "drop"], "payload": req})
    merged = set(fr.get("merged") or [])
    gold = [r for r in load_gold(run.work / "gold.jsonl") if r.source in ("real_pass", "real_fail") and r.task.split("~")[0] in merged]
    gi, _ = gold_items(gold)
    by_id = {r.item_id: r for r in gold}
    for it in gi:
        r = by_id[it.id]
        task_dir = run.work / "root" / "runs" / r.task.split("~")[0]
        prompt = final_draft(task_dir)["prompt"] if (task_dir / "journal.jsonl").exists() else ""
        items.append({"id": _safe_id(f"gold:{it.id}"), "family": fr.get("intent", "family"), "blind": True, "choices": ["pass", "fail"],
                      "payload": {"request": prompt, "change": r.solution, "question": "Does this change do what the request asks?"}})
    rows = discovery_rows(run.work / "discovery.jsonl") if (run.work / "discovery.jsonl").exists() else []
    failing = {f"{r['task']}#{x['id']}": (r, x) for r in rows if "dropped" not in r for x in r["traces"] if x["passed"] is not True}
    for key in human_sample(sorted(failing)):
        r, x = failing[key]
        items.append({"id": _safe_id(f"audit:{key.replace('|', '-')}"), "family": "audit-sample", "blind": True, "choices": list(CAUSES),
                      "payload": {"request": r["prompt"], "trace": x["text"], "question": "Why did this attempt fail?"}})
    out = run.work / "factory" / "label_items.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(i, default=str) + "\n" for i in items))
    run.write("human_requests.json", {"items": len(items), "by_family": dict(Counter(i["family"] for i in items)), "path": str(out),
                                      "route": "/api/v1/crucible/labels/queue", "labels": str(run.work / "factory" / "labels.jsonl")})


# ---- report -----------------------------------------------------------------------------------------------------------
STAGES = ("spec", "author", "realism", "oracle", "verifier", "ledger", "floors", "probes", "hackers", "second_verifier", "solvers", "differential",
          "audit", "flake", "horizon", "calibrate", "gate", "select")


def machine_funnel(work: Path) -> dict[str, Any]:
    """Per run: states entered, verdicts, loop-backs with their findings, the final status and reason."""
    from crucible.factory.driver import driver_machine

    roles = {s["id"]: s.get("role") or s.get("fn") or s["id"] for s in driver_machine()["states"]}
    runs, reached, passed, spend_stage, spend_role, loops = {}, Counter(), Counter(), Counter(), Counter(), []
    secs = Counter()
    for j in sorted((work / "root" / "runs").rglob("journal.jsonl")) if (work / "root" / "runs").exists() else []:
        rid = j.parent.relative_to(work / "root" / "runs").as_posix()
        status, reason, entered, verdicts = "running", None, [], {}
        for line in j.read_text().splitlines():
            ev = json.loads(line)
            k = ev.get("event")
            if k == "enter":
                entered.append(ev["state"])
            elif k == "settle":
                v = (ev.get("outputs") or {}).get("verdict")
                if isinstance(v, dict):
                    verdicts.setdefault(ev["state"], []).append({"ok": v.get("ok"), "owner": v.get("owner"), "findings": (v.get("findings") or [])[:4]})
            elif k == "spend":
                spend_stage[ev["state"]] += float(ev.get("usd") or 0.0)
                spend_role[roles.get(ev["state"], ev["state"])] += float(ev.get("usd") or 0.0)
                secs[ev["state"]] += float(ev.get("seconds") or 0.0)
            elif k == "transition":
                if STAGES.index(ev["to"]) < STAGES.index(ev["from"]) if ev["to"] in STAGES and ev["from"] in STAGES else False:
                    loops.append({"run": rid, "from": ev["from"], "to": ev["to"], "findings": (ev.get("findings") or [])[:3]})
            elif k == "status":
                status, reason = ev["status"], ev.get("reason")
        for s in set(entered):
            reached[s] += 1
        for s, vs in verdicts.items():
            if vs[-1]["ok"]:
                passed[s] += 1
        runs[rid] = {"status": status, "reason": reason, "entered": entered, "verdicts": verdicts}
    return {"runs": runs, "reached": dict(reached), "last_verdict_ok": dict(passed), "loops": loops, "spend_by_stage": dict(spend_stage),
            "spend_by_role": dict(spend_role), "seconds_by_stage": dict(secs)}


def phase_report(run: Run) -> None:
    spend = defaultdict(float)
    by_phase_role = defaultdict(float)
    if run.spend_path.exists():
        for line in run.spend_path.read_text().splitlines():
            e = json.loads(line)
            spend[e["phase"]] += e["usd"]
            by_phase_role[f"{e['phase']}:{e['role']}"] += e["usd"]
    mf = machine_funnel(run.work)
    total = sum(spend.values()) + sum(mf["spend_by_stage"].values())
    run.write("summary.json", {"spend_usd_total": round(total, 6), "cap_usd": run.cap, "spend_by_phase": {k: round(v, 6) for k, v in spend.items()},
                               "spend_by_phase_role": {k: round(v, 6) for k, v in sorted(by_phase_role.items())},
                               "machine": {k: (v if k in ("runs", "loops") else {s: round(x, 6) for s, x in v.items()}) for k, v in mf.items()}})
    print(json.dumps({"spend_usd_total": round(total, 4), "phases": {k: round(v, 4) for k, v in spend.items()},
                      "machine_stages": {k: round(v, 4) for k, v in mf["spend_by_stage"].items()}}, indent=1))


# ---- main -------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="+", choices=["mine", "controls", "discover", "intents", "family", "accept", "requests", "report"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/research/task-factory/runs"))
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--stop-at", type=float, default=18.0, help="stop launching paid work past this committed spend")
    ap.add_argument("--repos", default=",".join((*FAMILY_REPOS, HELD_OUT_REPO)))
    ap.add_argument("--per-repo", type=int, default=40)
    ap.add_argument("--discover-per-repo", type=int, default=20)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--intent", type=int, default=0)
    ap.add_argument("--target", default=None)
    ap.add_argument("--probes", type=int, default=12)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--control-reps", type=int, default=3)
    ap.add_argument("--root-name", default="root")
    ap.add_argument("--only", default=None, help="comma-separated mined task ids (a trial under a placeholder intent)")
    ap.add_argument("--max-author", type=int, default=2)
    ap.add_argument("--max-verifier", type=int, default=2)
    a = ap.parse_args()
    run = Run(a.work, a.out, a.cap)
    print(f"past spend {run.committed():.4f} USD of {a.cap}", flush=True)
    for ph in a.phases:
        t0 = time.time()
        if run.committed() > a.stop_at and ph not in ("report", "mine"):
            print(f"stop: committed {run.committed():.4f} > {a.stop_at}", flush=True)
            break
        if ph == "mine":
            asyncio.run(phase_mine(run, a.repos.split(","), a.per_repo))
        elif ph == "controls":
            asyncio.run(phase_controls(run, a.control_reps))
        elif ph == "discover":
            asyncio.run(phase_discover(run, a.discover_per_repo, a.attempts, a.parallel, a.stop_at))
        elif ph == "intents":
            asyncio.run(phase_intents(run))
        elif ph == "family":
            asyncio.run(phase_family(run, slots=a.slots, max_attempts=a.max_attempts, concurrency=a.concurrency, intent_index=a.intent,
                                     target=a.target, stop_at=a.stop_at, root_name=a.root_name, only=a.only.split(",") if a.only else None,
                                     max_author=a.max_author, max_verifier=a.max_verifier))
        elif ph == "accept":
            asyncio.run(phase_accept(run, probes_n=a.probes, attempts=4, k_exemplars=a.k, target=a.target))
        elif ph == "requests":
            phase_requests(run)
        elif ph == "report":
            phase_report(run)
        with (a.work / "phases.jsonl").open("a") as fh:
            fh.write(json.dumps({"phase": ph, "seconds": round(time.time() - t0, 1), "committed_usd": run.committed()}) + "\n")
        print(f"{ph}: {time.time() - t0:.1f}s, committed {run.committed():.4f} USD", flush=True)


if __name__ == "__main__":
    main()
