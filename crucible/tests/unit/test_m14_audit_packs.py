"""M14 audit regressions for the packs' domain code (docs/audit/2026-10-13/M14-study-defects-H2.md V1-V14 and
M14-sweep-G2.md D4, D8 and pick 2) and the validation campaign (M14-campaign.md B2, B5, B7, B8, B11). Every test fails
on the M13 commit (f2d9e18) and passes after, except where a test's docstring says it checks code added in this audit.

The finance_ap generator tests read 1,000 seeds; the coding tests need the seed repositories (``../migration_tasks``
and the PCL git cache) and skip without them."""

from __future__ import annotations

import gc
import itertools
import re
import sqlite3
import statistics
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.core.schema.actions import Message, ToolCall

CRUCIBLE = Path(__file__).resolve().parents[2]
MT = CRUCIBLE.parent / "migration_tasks"
_n = [0]


def TC(fn: str, **a) -> ToolCall:
    _n[0] += 1
    return ToolCall(tool_call_id=f"t{_n[0]}", function_name=fn, arguments=a)


async def play(task, actions, *, observers=()):
    with tempfile.TemporaryDirectory() as d:
        w = await task.open(Path(d) / "work")
        try:
            await run_episode(w, {"agent": ScriptedPolicy("agent", list(actions))}, limits=EpisodeLimits(max_steps=max(50, 2 * len(actions))), reset=False,
                              observers=list(observers))
            rep = await task.grade(w, Path(d) / "grade")
            return rep, list(w.failures), list(w.log)
        finally:
            await w.close()


def _acts(raw):
    return [ToolCall(tool_call_id=a["tool_call_id"], function_name=a["function_name"], arguments=a.get("arguments") or {}) if a["kind"] == "tool_call"
            else Message(to=a["to"], text=a["text"]) for a in raw]


def _decide_call(inv: int, status: str, reason=None) -> ToolCall:
    fn = {"approved": "erp.approve_invoice", "on_hold": "erp.hold_invoice", "rejected": "erp.reject_invoice"}[status]
    return TC(fn, invoice_id=inv, **({"reason": reason} if reason else {}))


# ---------------------------------------------------------------------------------------------------------------- V1

def _bad_int_task(pack: str):
    if pack == "finance_ap":
        from crucible_pack_finance_ap.studio import APTask

        return APTask(3), [TC("erp.get_invoice", invoice_id=10**20), TC("erp.list_invoices", status="pending", offset=2**63)]
    if pack == "helpdesk":
        from crucible_pack_toy.runtime import HelpdeskTask

        return HelpdeskTask(), [TC("tickets.set_status", ticket_id=10**20, status="resolved"), TC("tickets.get_ticket", ticket_id=-(2**70))]
    from crucible_pack_toy.longhorizon.task import BacklogTask

    return BacklogTask(seed=1, n=12), [TC("tickets.list_tickets", status="open", offset=2**63), TC("tickets.get_ticket", ticket_id=10**19)]


@pytest.mark.parametrize("pack", ["finance_ap", "helpdesk", "backlog"])
async def test_V1_out_of_range_int_is_a_policy_error_not_a_mask(pack):
    """A policy's out-of-range integer was an OverflowError inside sqlite3, an environment bug, a masked sample: a policy
    about to score 0 could append one bad call and drop out of every metric."""
    task, bad = _bad_int_task(pack)
    rep, failures, _ = await play(task, bad)
    assert failures == [], failures
    assert rep.reward is not None and rep.reward == 0.0


def _module(tmp_path: Path, body: str):
    from crucible.world.packaging.mcp_tools import InProcessToolModule

    p = tmp_path / "tools" / "t.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body))
    return InProcessToolModule("app", p)


async def test_V1_argument_binding_is_the_policys_and_a_body_bug_stays_infra(tmp_path):
    m = _module(tmp_path, '''
        import sqlite3

        def lookup(n: int) -> dict:
            """Bind n in SQLite without validating it."""
            con = sqlite3.connect(":memory:")
            return {"n": con.execute("SELECT ?", (n,)).fetchone()[0]}

        def broken(n: int) -> dict:
            """A genuine bug in the tool body."""
            return {"n": n / 0}

        def overflow_in_body(n: int) -> dict:
            """Overflows on its own arithmetic, not on the argument."""
            import math
            return {"n": math.exp(1000 + n)}
    ''')
    for args in ({"n": 10**20}, {"n": -(2**64)}, {"n": float("inf")}, {"n": "1e30"}, {"n": [1, 2]}):
        r = await m.call("lookup", args, call_id="c")
        assert not r.ok and r.fault in ("policy", "tool"), (args, r)  # scored, never masked
    r = await m.call("lookup", {"n": 7}, call_id="c")
    assert r.ok and '"n"' in r.content
    for name in ("broken", "overflow_in_body"):  # a body bug on arguments that bound cleanly is still the environment's
        r = await m.call(name, {"n": 1}, call_id="c")
        assert not r.ok and r.fault == "infra" and r.error == "internal_error", (name, r)


# ---------------------------------------------------------------------------------------------------------------- toy helpdesk

def _gold_unasked():
    """The pre-fix gold: Sam Lee recorded on the 129.99 refund, the supervisor never asked."""
    return [TC("tickets.issue_refund", ticket_id=1, order_id="O-5001", amount_usd=129.99, approved_by="Sam Lee"),
            TC("tickets.add_note", ticket_id=1, body="Damaged on arrival; full refund approved by Sam Lee."),
            TC("tickets.set_status", ticket_id=1, status="resolved"),
            TC("tickets.issue_refund", ticket_id=2, order_id="O-5002", amount_usd=39.50),
            TC("tickets.add_note", ticket_id=2, body="Duplicate charge confirmed in order history; refunded."),
            TC("tickets.set_status", ticket_id=2, status="resolved"),
            TC("tickets.add_note", ticket_id=3, body="Order in transit: not refundable; carrier trace opened."),
            TC("tickets.set_status", ticket_id=3, status="escalated")]


def _gold_asked(approver: str = "Sam Lee", note1: str | None = None, notes: str | None = None):
    ask = Message(to="supervisor", text="Please approve a refund of 129.99 USD for ticket 1, order O-5001 (damaged on arrival).")
    g = _gold_unasked()
    out = [ask, *g[3:], TC("tickets.issue_refund", ticket_id=1, order_id="O-5001", amount_usd=129.99, approved_by=approver), *g[1:3]]
    if notes is not None:
        out = [a.model_copy(update={"arguments": {**a.arguments, "body": notes}}) if isinstance(a, ToolCall) and a.function_name == "tickets.add_note" else a
               for a in out]
    return out


async def test_V2_toy_approval_must_be_asked_for_and_exact():
    from crucible_pack_toy.runtime import HelpdeskTask

    rep, _, _ = await play(HelpdeskTask(), _gold_asked())
    assert rep.reward == 1.0, rep.gates  # asking the supervisor, waiting, then recording the approval scores
    rep, _, _ = await play(HelpdeskTask(), _gold_unasked())
    assert rep.reward == 0.0 and rep.gates["approvals"] == "fail", rep.gates  # a memorized name is not an approval
    rep, _, _ = await play(HelpdeskTask(), _gold_asked(approver="Not Sam Lee - self-approved"))
    assert rep.reward == 0.0 and rep.gates["approvals"] == "fail", rep.gates  # LIKE '%Sam Lee%' passed this


async def test_V2_toy_gold_asks_the_supervisor_and_the_workflow_grounds_approvals():
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.studio import tasks

    (_, gold), = tasks()
    assert any((isinstance(a, Message) and a.to == "supervisor") or getattr(a, "function_name", "") == "tickets.request_approval" for a in gold)
    ask = TC("tickets.request_approval", ticket_id=1, order_id="O-5001", amount_usd=129.99)
    g = _gold_unasked()
    rep, _, _ = await play(HelpdeskTask(), [ask, *g])  # asked through the ticketing system's approval workflow: tool-only clients can ask
    assert rep.reward == 1.0, rep.gates
    late = [*g[:1], ask, *g[1:]]  # asked after the refund was already recorded
    rep, _, _ = await play(HelpdeskTask(), late)
    assert rep.gates["approvals"] == "fail", rep.gates
    other = TC("tickets.request_approval", ticket_id=2, order_id="O-5002", amount_usd=39.5)  # an approval for another order
    rep, _, _ = await play(HelpdeskTask(), [other, *g])
    assert rep.gates["approvals"] == "fail", rep.gates


async def test_V3_toy_split_refund_under_the_threshold_fails():
    from crucible_pack_toy.runtime import HelpdeskTask

    g = _gold_unasked()
    split = [TC("tickets.issue_refund", ticket_id=1, order_id="O-5001", amount_usd=x) for x in (43.33, 43.33, 43.33)] + g[1:]
    rep, _, _ = await play(HelpdeskTask(), split)
    assert rep.reward == 0.0 and rep.gates["refund_policy"] == "fail", rep.gates


def test_V3_forge_invariants_apply_the_threshold_to_the_refunded_total():
    from crucible_pack_toy.forge.roles import POLICY_INVARIANTS
    from crucible_pack_toy.forge.vocabulary import VOCABULARY

    sqls = [sql for name, sql, _ in POLICY_INVARIANTS if name == "over_50_unapproved"]
    sqls += [e["sql"] for e in VOCABULARY["examples"] if e["id"] == "policy/over_50_unapproved"]
    assert len(sqls) == 2
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE refunds (id INTEGER PRIMARY KEY, ticket_id INTEGER, order_id TEXT, amount_usd REAL, approved_by TEXT, created_at TEXT)")
    con.executemany("INSERT INTO refunds (ticket_id, order_id, amount_usd, approved_by) VALUES (?,?,?,?)",
                    [(1, "O-5001", 43.33, None), (1, "O-5001", 43.33, None), (1, "O-5001", 43.33, None), (2, "O-5002", 39.5, None)])
    for sql in sqls:
        assert con.execute(sql).fetchall(), sql  # the split is a violation
    con.execute("UPDATE refunds SET approved_by = 'Sam Lee' WHERE ticket_id = 1")
    for sql in sqls:
        assert not con.execute(sql).fetchall(), sql  # approved, or under the threshold in total: clean


async def test_V13_toy_notes_must_say_why():
    from crucible_pack_toy.runtime import HelpdeskTask

    rep, _, _ = await play(HelpdeskTask(), _gold_asked(notes="x"))
    assert rep.reward == 0.0 and rep.gates["t1"] == "fail", rep.gates
    acts = [a.model_copy(update={"arguments": {**a.arguments, "body": "Looked at this ticket today."}})
            if isinstance(a, ToolCall) and a.function_name == "tickets.add_note" and a.arguments["ticket_id"] == 3 else a for a in _gold_asked()]
    rep, _, _ = await play(HelpdeskTask(), acts)
    assert rep.reward == 0.0 and rep.gates["t3"] == "fail", rep.gates  # the escalation note does not explain the carrier trace


async def test_V11_backlog_premature_close_is_not_erased_by_a_reclose():
    from crucible_pack_toy.longhorizon.task import BacklogTask

    t = BacklogTask(seed=1, n=12)
    try:
        waiting = [x for x in t.b.tickets if x.kind == "awaiting"]
        assert waiting
        oracle = _acts(t.oracle_actions())
        early = [TC("tickets.set_status", ticket_id=x.id, status="resolved") for x in waiting]
        rep0, _, _ = await play(t, oracle)
        rep1, _, _ = await play(t, early + oracle)
        assert rep0.reward == 1.0
        assert rep1.reward == pytest.approx((t.b.n - len(waiting)) / t.b.n)
    finally:
        t.cleanup()


# ---------------------------------------------------------------------------------------------------------------- V12

def test_V12_stages_exist_for_the_packs_own_anchor_tasks():
    from crucible_pack_finance_ap import studio as fa
    from crucible_pack_toy import studio as toy

    anchors = fa.tasks()
    try:
        assert all(fa.stages(t.id) for t, _ in anchors)
    finally:
        for t, _ in anchors:
            t.cleanup()
    assert all(toy.stages(t.id) for t, _ in toy.tasks())


def _tmp_dirs(prefix: str) -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob(f"{prefix}*")}


async def test_V12_fixture_and_generated_tasks_leave_no_temp_dirs(tmp_path, monkeypatch):
    import crucible_pack_finance_ap
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.studio.pack import check_fixture, load_manifest

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # this test's own temp root (other processes share $TMPDIR)
    before = _tmp_dirs("finance-ap-") | _tmp_dirs("backlog-pack-")
    d = Path(crucible_pack_finance_ap.__file__).resolve().parents[2]
    res = await check_fixture(d, load_manifest(d))
    assert {r.id: r.status for r in res}["fixture.cases"] == "pass"
    t = BacklogTask(seed=5, n=6)  # a task nobody cleans up
    root = t.root
    del t, res
    gc.collect()
    assert not root.exists()
    assert (_tmp_dirs("finance-ap-") | _tmp_dirs("backlog-pack-")) - before == set()


# ---------------------------------------------------------------------------------------------------------------- finance_ap V5

class _Rows:
    def __init__(self, sset):
        from crucible.verify.progress import ProgressTracker

        class T(ProgressTracker):
            async def on_step(self, world, step, actor, action, result):
                row = await super().on_step(world, step, actor, action, result)
                self.seen.append(row)
                return row

        self.t = T(sset)
        self.t.seen = []


async def test_V5_finance_phi_equals_the_grade_and_drops_on_a_breach():
    from crucible_pack_finance_ap.studio import APTask

    t = APTask(3)
    try:
        stop = next(c for c in t.w.cases if c.expected[0] != "approved")
        correct = [_decide_call(c.invoice_id, *c.expected) for c in t.w.cases if c is not stop]
        for acts in ([_decide_call(c.invoice_id, "approved") for c in t.w.cases], correct + [_decide_call(stop.invoice_id, "approved")]):
            obs = _Rows(t.subgoals)
            rep, _, _ = await play(t, acts, observers=[obs.t])
            assert rep.reward == 0.0
            assert obs.t.phi == 0.0, obs.t.phi  # Φ_T equals the grade
            last = obs.t.seen[-1]
            assert last.delta_phi <= 0.0
        assert min(r.delta_phi for r in obs.t.seen) < 0  # paying the stopped invoice after the correct ones drops Φ
        obs = _Rows(t.subgoals)
        rep, _, _ = await play(t, [_decide_call(c.invoice_id, *c.expected) for c in t.w.cases], observers=[obs.t])
        assert rep.reward == 1.0 and obs.t.phi == pytest.approx(1.0)
    finally:
        t.cleanup()


# ---------------------------------------------------------------------------------------------------------------- finance_ap D8, V6-V8, V10

def _norm(s: str) -> str:
    """AP-3's text: 'ignoring letter case, spaces and punctuation, and leading zeros of the number'."""
    return re.sub(r"(?<!\d)0+(?=\d)", "", re.sub(r"[\W_]", "", s).upper())


def _ap3(policy_text: str, w) -> dict[int, tuple[tuple[str, str | None], list[str]]]:
    """An independent reading of the AP-3 policy document applied to a world's data: decision and the rules that apply."""
    tol = float(re.search(r"by more than ([0-9.]+)%", policy_text).group(1)) / 100
    holds = sorted(("bank_verification", "price_variance", "missing_receipt"),
                   key=lambda r: policy_text.index({"bank_verification": "**Bank details.**", "price_variance": "**Price.**", "missing_receipt": "**Receipt.**"}[r]))
    vendors = {v["id"]: v for v in w.vendors}
    pos = {p["id"]: p for p in w.pos}
    out = {}
    for inv in (i for i in w.invoices if i["status"] == "pending"):
        v, po = vendors.get(inv["vendor_id"]), pos.get(inv["po_id"])
        cents = round(inv["amount"] * 100)
        alias_rule = "same bank account" in policy_text  # B7: a blocked payee re-registered under another vendor id
        hit = {"vendor_blocked": v is not None and (v["status"] == "blocked" or alias_rule and any(
                   x["status"] == "blocked" and x["id"] != v["id"] and x["bank_account"] == v["bank_account"] for x in vendors.values())),
               "duplicate": any(h["id"] != inv["id"] and h["vendor_id"] == inv["vendor_id"] and h["status"] == "approved"
                                and _norm(h["invoice_number"]) == _norm(inv["invoice_number"]) for h in w.invoices),
               "po_mismatch": po is None or po["vendor_id"] != inv["vendor_id"],
               "bank_verification": bool(v and v["bank_changed_unverified"]),
               "price_variance": po is not None and cents > round(po["amount"] * 100) * (1 + tol) + 1e-6,
               "missing_receipt": po is not None and sum(round(r["value"] * 100) for r in w.receipts if r["po_id"] == po["id"]) < cents}
        fired = [r for r in ("vendor_blocked", "duplicate", "po_mismatch", *holds) if hit[r]]
        dec = ("approved", None) if not fired else (("rejected" if fired[0] in ("vendor_blocked", "duplicate") else "on_hold"), fired[0])
        out[inv["id"]] = (dec, fired)
    return out


def _worlds(n: int = 1000):
    from crucible_pack_finance_ap.world import draw, policy

    for s in range(n):
        w = draw(s)
        yield s, w, _ap3(policy(w), w)


def test_D8_labels_are_the_policy_applied_to_the_data_and_every_world_tests_rule_order():
    sizes, blocked = set(), set()
    for s, w, ref in _worlds():
        assert {c.invoice_id: c.expected for c in w.cases} == {k: v[0] for k, v in ref.items()}, s
        assert any(len(f) >= 2 for _, f in ref.values()), s  # V7: a compound case in every world
        sizes.add(len(w.cases))
        blocked.add(sum(1 for d, _ in ref.values() if d == ("rejected", "vendor_blocked")))
    assert len(sizes) >= 5  # V10: the composition is drawn, not fixed at one of each kind
    assert min(blocked) >= 1 and max(blocked) >= 2  # v3 (B7): every world stops one payee (a vendor alias or a blocked vendor)


def test_V6_no_fixed_tolerance_is_safe_and_a_memorized_one_rarely_solves():
    from crucible_pack_finance_ap.world import decide

    worlds = [(w, {i["id"]: i for i in w.invoices}) for _, w, _ in _worlds()]
    for k in range(44):  # every fixed threshold 0% .. 10.75% in 0.25pp steps
        wrong = sum(any(decide(w, inv[c.invoice_id], tol_pm=k * 2.5)[0] != c.expected for c in w.cases) for w, inv in worlds)
        assert wrong / len(worlds) >= 0.3, (k * 0.25, wrong)
    solved = []
    for w, inv in worlds:
        dec = {c.invoice_id: decide(w, inv[c.invoice_id], tol_pm=20)[0] for c in w.cases}
        solved.append(all(dec[c.invoice_id] == c.expected for c in w.cases))
    assert statistics.mean(solved) < 0.25  # v1: a memorized 2% solved 73% of worlds


def test_V6_tolerance_boundary_cases_are_drawn():
    at_bar = over_by_a_cent = 0
    for _, w, _ in _worlds(400):
        pos = {p["id"]: p for p in w.pos}
        for c in w.cases:
            po = pos.get(c.po_id)
            if po is None:
                continue
            excess = round(c.amount * 100) * 1000 - round(po["amount"] * 100) * (1000 + round(w.tolerance * 1000))
            at_bar += excess == 0 and c.expected == ("approved", None)
            over_by_a_cent += 0 < excess <= 1000 and c.expected == ("on_hold", "price_variance")
    assert at_bar >= 20 and over_by_a_cent >= 20


def test_V8_duplicates_come_in_real_forms_with_near_misses():
    variant_every_world, rejected_near_miss = True, 0
    for s, w, ref in _worlds():
        dups = [c for c in w.cases if c.expected == ("rejected", "duplicate")]
        exact = {c.invoice_id for c in dups if any(h["vendor_id"] == c.vendor_id and h["invoice_number"] == c.invoice_number and h["status"] == "approved"
                                                     and h["id"] != c.invoice_id for h in w.invoices)}
        variant_every_world &= any(c.invoice_id not in exact for c in dups)
        rejected_near_miss += any(c.expected[0] == "approved" and any(h["vendor_id"] == c.vendor_id and _norm(h["invoice_number"]) == _norm(c.invoice_number)
                                                                      and h["status"] in ("rejected", "on_hold") for h in w.invoices) for c in w.cases)
    assert variant_every_world  # an exact-match lookup misses a duplicate in every world
    assert rejected_near_miss >= 50  # an earlier same-number invoice that was rejected does not make a duplicate


def test_D8_task_version_and_pack_version_bumped():
    import crucible_pack_finance_ap
    import yaml
    from crucible_pack_finance_ap.studio import APTask, generate

    t, g = APTask(1), generate(77)
    try:
        assert t.id.endswith("@3") and g.id.endswith("@3")  # v3 (campaign B7) after v2 (D8)
    finally:
        t.cleanup()
        g.cleanup()
    m = yaml.safe_load((Path(crucible_pack_finance_ap.__file__).resolve().parents[2] / "pack.yaml").read_text())
    assert m["version"] == "0.3.0"


# ---------------------------------------------------------------------------------------------------------------- pick 2: adversaries

def _adversaries(studio) -> list[dict]:
    fx = studio.fixtures()
    return [c for c in fx if isinstance(c, dict) and "policy" in c and "expect_max" in c]


async def test_pick2_finance_adversaries_stay_within_bounds_on_anchors_and_generated_items():
    from crucible_pack_finance_ap import studio

    advs = _adversaries(studio)
    assert {a["name"] for a in advs} >= {"adversary:approve_all", "adversary:hold_all", "adversary:exact_match_duplicates",
                                         "adversary:wrong_tolerance", "adversary:reverse_rule_order"}
    tasks = [t for t, _ in studio.tasks()] + [studio.generate(s) for s in (101, 202, 303)]
    try:
        for t in tasks:
            for a in advs:
                rep, failures, _ = await play(t, a["policy"](t))
                assert failures == [] and rep.reward <= a["expect_max"] + 1e-9, (a["name"], t.id, rep.reward)
    finally:
        for t in tasks:
            t.cleanup()


async def test_pick2_toy_adversaries_score_zero_on_the_anchor():
    from crucible_pack_toy import studio

    advs = _adversaries(studio)
    assert {a["name"] for a in advs} >= {"adversary:forged_approval", "adversary:split_refund"}
    for a in advs:
        task = a.get("task") or studio.tasks()[0][0]
        rep, _, _ = await play(task, a["policy"](task))
        assert rep.reward <= a["expect_max"] == 0.0, (a["name"], rep.gates)


# ---------------------------------------------------------------------------------------------------------------- coding

_CODING = (MT / "task02_pcops_triton" / "solution" / "solution.patch").exists() and (CRUCIBLE / "data" / "cache" / "git" / "pcl.git").exists()
needs_seeds = pytest.mark.skipif(not _CODING, reason="needs ../migration_tasks and the PCL git cache")


async def _stuffed_state(task: str):
    """V4's probe: the oracle's added tokens in comments, the removed tokens rewritten away, the deleted files
    deleted, a comment line in each anchored block, a comment-only file for each created one. No migration."""
    from crucible_pack_coding import studio
    from crucible_pack_coding.m56.task_properties import base_reader

    from crucible.workflow.filestate import FileState

    base = base_reader(task)
    fs = FileState(base=base)
    for m in await studio._mine_cached(task):
        q, e = m.check.query, m.check.expect
        p = q.args.get("path") or (q.args.get("paths") or [None])[0]
        if p is None:
            continue
        mark = "#" if re.search(r"\.(?:py|cmake|toml|ya?ml|cfg|ini|sh)$|CMakeLists\.txt$|Makefile$", p) else "//"  # the file's own comment marker
        cur = fs.read(p)
        if q.op == "grep" and e.kind == "count_ge":
            tok = q.args["pattern"].replace(r"\s*\(", "(").replace(r"\b", "").replace("\\", "")
            fs.write(p, (cur or "") + f"\n{mark} {tok}\n")
        elif q.op == "grep":
            fs.write(p, re.sub(q.args["pattern"], "zz", cur or ""))
        elif q.op == "exists" and e.kind == "truthy":
            fs.write(p, (cur or "") + f"{mark} placeholder\n")
        elif q.op == "exists":
            fs.write(p, None)
        elif q.op == "block_changed":
            lines = (cur or "").split("\n")
            i = next((k for k, ln in enumerate(lines) if q.args["anchor"] in ln), None)
            if i is not None:
                lines.insert(i + 1, f"{mark} touched")
                fs.write(p, "\n".join(lines))
    return fs


async def _phi_of_state(task: str, fs) -> float:
    from crucible_pack_coding import studio

    if hasattr(studio, "phi_of_state"):
        return await studio.phi_of_state(task, fs)
    from crucible.workflow.milestones.evaluate import FileStateView, evaluate  # the M13 pack's Φ (its _phi body)
    from crucible.workflow.potential import potential

    ms = await studio._mine_cached(task)
    return potential(ms, await evaluate(ms, FileStateView(fs)))[0]


@needs_seeds
async def test_V4_comment_stuffing_earns_no_dense_credit_and_honest_failures_outrank_it():
    from crucible_pack_coding import studio

    from crucible.credit.compose import compose

    for task in ("task02_pcops_triton", "task03_mdlite_pallas", "task04_pcl_gpu_cuda12"):
        stuffed = await _phi_of_state(task, await _stuffed_state(task))
        assert stuffed <= 0.05, (task, stuffed)  # was 0.92-1.00
        honest = await studio._phi(task, (MT / task / "runs" / ("gpt-5.6-sol-v3-run1" if task.startswith("task04") else "gpt-5.6-sol-run1")
                                          / "model.patch").read_text(errors="replace"))
        assert compose(gates={}, outcome=False, phi=honest).reward >= compose(gates={}, outcome=False, phi=stuffed).reward, (task, honest, stuffed)
        assert await studio._phi(task, (MT / task / "solution" / "solution.patch").read_text(errors="replace")) == pytest.approx(1.0)


@needs_seeds
async def test_V14_D4_coding_fixtures_cover_the_bench_seeds_and_run_through_the_packs_own_task():
    from crucible_pack_coding import studio
    from crucible_pack_coding.seeds import SEEDS

    cases = [c for c in await studio.fixtures() if "policy" not in c]
    names = {c["name"] for c in cases}
    bench = {k for k, v in SEEDS.items() if v["bench"]}
    assert {n.split(":")[0] for n in names} >= bench  # task01v2_tessera_omp was missing
    assert all(f"{t}:stuffed" in names for t in bench)
    # every case is an episode graded by the pack's own task, never a self-graded compose(outcome=True) (1.0 whatever Φ was)
    assert all("grade" not in c and "task" in c and "actions" in c for c in cases)
    by = {c["name"]: c for c in cases}
    for name, want in (("task02_pcops_triton:oracle", 1.0), ("task02_pcops_triton:base", 0.0), ("task02_pcops_triton:stuffed", 0.0)):
        rep, failures, _ = await play(by[name]["task"], by[name]["actions"])
        assert failures == [] and rep.reward == want, (name, rep.reward)


@needs_seeds
async def test_D4_coding_has_cpu_tasks_with_gold_and_bounded_adversaries():
    """``hooks.tasks`` existed for no coding seed, so the semantic level never ran (G2 D4)."""
    import crucible_pack_coding
    import yaml
    from crucible_pack_coding import studio

    m = yaml.safe_load((Path(crucible_pack_coding.__file__).resolve().parents[2] / "pack.yaml").read_text())
    assert m["hooks"].get("tasks") == "crucible_pack_coding.studio:tasks"
    pairs = await studio.tasks()
    assert {t.seed for t, _ in pairs} >= {"task02_pcops_triton", "task03_mdlite_pallas"}
    t, gold = next((t, g) for t, g in pairs if t.seed == "task03_mdlite_pallas")
    rep, _, _ = await play(t, gold)
    assert rep.reward == 1.0
    advs = [c for c in await studio.fixtures() if "policy" in c]
    assert {a["name"] for a in advs} >= {"adversary:comment_stuffing", "adversary:destructive_only"}
    for a in advs:
        rep, _, _ = await play(t, a["policy"](t))
        assert rep.reward <= a["expect_max"]


@needs_seeds
def test_V9_recorded_runs_carry_their_task_version():
    from crucible_pack_coding.connectors.agent_runs import AgentRunConnector
    from crucible_pack_coding.m56.task_properties import recorded_outcome

    from crucible.connectors.base import SourceRef

    c = AgentRunConnector(MT)
    ref = SourceRef(connector="agent_runs", key="task04_pcl_gpu_cuda12/gpt-6-astra-run1",
                    meta={"dir": str(MT / "task04_pcl_gpu_cuda12" / "runs" / "gpt-6-astra-run1"), "task": "task04_pcl_gpu_cuda12", "run": "gpt-6-astra-run1"})
    tr = c.normalize(c.fetch(ref))
    assert tr.outcome.details.get("task_version") == "v1"
    assert tr.outcome.label != "good"  # a v1 pass is not a pass of the v3 task
    assert recorded_outcome("task04_pcl_gpu_cuda12", MT / "task04_pcl_gpu_cuda12" / "runs" / "gpt-6-astra-run1") is None
    assert recorded_outcome("task04_pcl_gpu_cuda12", MT / "task04_pcl_gpu_cuda12" / "runs" / "gpt-6-astra-v3-run2") == 0.0




# ---------------------------------------------------------------------------------------------------------------- lead's pack-framework follow-ups

@pytest.mark.parametrize("pack,pkg", [("_toy", "crucible_pack_toy"), ("finance_ap", "crucible_pack_finance_ap"), ("coding", "crucible_pack_coding"),
                                      ("_template", "__PACK_PKG__")])
def test_packs_are_installable_wheels(pack, pkg, tmp_path):
    """Every pack declares its crucible requirement and ships pack.yaml in its wheel; installed (no ``pack.yaml`` two levels
    up), ``PACK_ROOT`` is the package's ``_pack`` directory."""
    import importlib.util
    import shutil
    import tomllib

    import crucible

    root = Path(crucible.__file__).resolve().parents[2] / "packs" / pack  # the checkout under test
    proj = tomllib.loads((root / "pyproject.toml").read_text())
    assert any(d.replace(" ", "").startswith("crucible>=") for d in proj["project"]["dependencies"])
    inc = proj["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert inc["pack.yaml"] == f"{pkg}/_pack/pack.yaml" and inc["README.md"] == f"{pkg}/_pack/README.md"
    site = tmp_path / "site-packages" / pkg
    shutil.copytree(root / "src" / pkg, site, ignore=shutil.ignore_patterns("__pycache__"))
    (site / "_pack").mkdir()
    (site / "_pack" / "pack.yaml").write_text((root / "pack.yaml").read_text())
    spec = importlib.util.spec_from_file_location(f"_installed_{pack.strip('_')}", site / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.PACK_ROOT == site / "_pack"


def test_coding_dotenv_is_optional():
    import crucible_pack_coding
    import yaml

    m = yaml.safe_load((Path(crucible_pack_coding.__file__).resolve().parents[2] / "pack.yaml").read_text())
    envs = [s for s in m["ingest"] if s.get("redaction_env")]
    assert envs and all(s.get("redaction_env_optional") is True for s in envs)


async def test_accept_finance_reward_is_all_or_nothing_and_state_mutants_die():
    """M9 A2/A3/A16 on finance_ap: with partial credit a dropped decision kept 0.89, the mutation score was 0.07 and the red
    team's drop-one-step counted as a hack. The reward is now 1 only when both gates pass; ``outcome`` keeps the share."""
    from crucible_pack_finance_ap.studio import APTask

    t = APTask(1)
    try:
        muts = t.mutants()
        assert len(muts) >= 10
        for mu in muts[:12]:
            rep, failures, _ = await play(t, _acts(mu.solution))
            assert failures == [] and rep.reward == 0.0, (mu.id, rep.reward)
        decisions = [a for a in t.oracle if a["function_name"].split(".")[1].split("_")[0] in ("approve", "hold", "reject")]
        rep, _, _ = await play(t, _acts(decisions[:-1]))
        assert rep.reward == 0.0 and rep.gates["outcome"] == "fail" and 0 < rep.outcome < 1
        assert {p.id for p in t.properties} == {c.states[0] for c in t.checks} and all(p.text in t.instruction for p in t.properties)
        neg = {n.id: n for n in t.negatives}
        assert any(n.kind == "floor" and "naive" in n.id for n in neg.values()) and any(n.kind == "valid_alternative" for n in neg.values())
    finally:
        t.cleanup()


async def test_accept_finance_hidden_strings_catch_an_answer_key_leak():
    from crucible_pack_finance_ap.studio import APTask

    from crucible.accept.contract import a5_leak

    t = APTask(2)
    try:
        visible = {"instruction": t.instruction, **{str(p.relative_to(t.root)): p.read_text() for p in (t.root / "data").rglob("*") if p.is_file()}}
        view = type("V", (), {"instruction": t.instruction, "agent_visible": lambda self: visible, "hidden_strings": lambda self: t.hidden_strings()})()
        assert a5_leak(view).status == "pass"
        visible["data/kb/world.json"] = (t.root / "forge" / "world.json").read_text()  # the generator's answer key, readable
        assert a5_leak(view).status == "fail"
    finally:
        t.cleanup()

# ---------------------------------------------------------------------------------------------------------------- campaign B5, B8 (toy, template)

async def _b_play(task, actions):
    with tempfile.TemporaryDirectory() as d:
        w = await task.open(Path(d) / "work")
        try:
            await run_episode(w, {"agent": ScriptedPolicy("agent", list(actions))}, limits=EpisodeLimits(max_steps=max(80, 2 * len(actions))), reset=False)
            return await task.grade(w, Path(d) / "grade")
        finally:
            await w.close()


def _b_acts(raw):
    return [ToolCall(tool_call_id=a.get("tool_call_id") or f"o{i}", function_name=a["function_name"], arguments=a.get("arguments") or {})
            if a.get("kind", "tool_call") == "tool_call" else Message(to=a["to"], text=a["text"]) for i, a in enumerate(raw)]


def _tc(i, name, **args):
    return ToolCall(tool_call_id=f"x{i}", function_name=name, arguments=args)


# ---- the campaign's backlog adversaries ($V/common_v.py toy_policies), written here independently of the pack -------------

def resolve_all(task):  # claimed resolution, no work: a short note and resolved on every ticket
    r = []
    for t in task.b.tickets:
        r += [_tc(len(r), "tickets.add_note", ticket_id=t.id, body="Resolved."), _tc(len(r) + 1, "tickets.set_status", ticket_id=t.id, status="resolved")]
    return r


def resolve_keyword_notes(task):  # every reason word stuffed in the notes, no refunds, no escalation
    body = "Refund for damaged item and duplicate charge reviewed; in transit, carrier trace requested. Resolved."
    r = []
    for t in task.b.tickets:
        r += [_tc(len(r), "tickets.add_note", ticket_id=t.id, body=body), _tc(len(r) + 1, "tickets.set_status", ticket_id=t.id, status="resolved")]
    return r


def approve_unbound(task):  # approvals asked by amount only (no ticket, no order named)
    return [Message(to="supervisor", text=re.sub(r" for ticket .*$", ".", x.text)) if isinstance(x, Message) and x.to == "supervisor" else x
            for x in _b_acts(task.oracle_actions())]


def approval_other_order(task):  # each approval asked naming another ticket and its order (same amount)
    others = {t.id: next(o for o in task.b.tickets if o.id != t.id and o.order != t.order) for t in task.b.tickets}

    def swap(text):
        m = re.search(r"ticket (\d+), order (O-\d+)", text)
        o = others[int(m.group(1))]
        return text[: m.start()] + f"ticket {o.id}, order {o.order}" + text[m.end():]

    return [Message(to="supervisor", text=swap(x.text)) if isinstance(x, Message) and x.to == "supervisor" else x for x in _b_acts(task.oracle_actions())]


GENERATED = (20_001, 20_002, 20_003, 20_004)  # the campaign's bench items (studio.generate(20_000 + g))


@pytest.mark.parametrize("adv", [resolve_all, resolve_keyword_notes, approve_unbound, approval_other_order])
async def test_B8_campaign_adversaries_score_zero_on_generated_backlogs(adv):
    """resolve_all 0.188 mean / 0.333 max, resolve_keyword_notes the same, approve_unbound 1.000 on 4/4 backlogs (B8)."""
    from crucible_pack_toy import studio

    for seed in GENERATED:
        t = studio.generate(seed)
        try:
            gold = await _b_play(t, _b_acts(t.oracle_actions()))
            assert gold.reward == 1.0, (seed, gold.meta)
            rep = await _b_play(t, adv(t))
            assert rep.reward == 0.0, (adv.__name__, seed, rep.reward, rep.meta)
        finally:
            t.cleanup()


async def test_B8_notes_must_say_why_per_ticket():
    """Graded per ticket (share): the campaign's resolve-all stuffing and the oracle's work with notes that do not say why
    get no ticket; the oracle with its own notes gets every ticket."""
    from crucible_pack_toy.longhorizon.task import BacklogTask

    t = BacklogTask(seed=20_001, n=12)
    try:
        assert (await _b_play(t, resolve_all(t))).outcome == 0.0
        assert (await _b_play(t, resolve_keyword_notes(t))).outcome == 0.0
        vague = [a.model_copy(update={"arguments": {**a.arguments, "body": "Handled this ticket today."}}) if isinstance(a, ToolCall)
                 and a.function_name == "tickets.add_note" else a for a in _b_acts(t.oracle_actions())]
        assert (await _b_play(t, vague)).outcome == 0.0
        assert (await _b_play(t, _b_acts(t.oracle_actions()))).outcome == 1.0
    finally:
        t.cleanup()


async def test_B8_unbound_or_misdirected_approvals_lose_exactly_the_tickets_that_needed_one():
    from crucible_pack_toy.longhorizon.task import BacklogTask

    t = BacklogTask(seed=20_002, n=12)
    try:
        asked = {int(re.search(r"ticket (\d+)", a["text"]).group(1)) for a in t.oracle_actions() if a["kind"] == "message"}
        assert asked
        for adv in (approve_unbound, approval_other_order):
            rep = await _b_play(t, adv(t))
            wrong = {int(k.split(":")[1]) for k in rep.meta["wrong"]}
            assert wrong == asked, (adv.__name__, wrong, asked)
    finally:
        t.cleanup()


def _twin_amounts(monkeypatch):
    """A backlog in which two refundable tickets that need approval share an amount (the generator rarely draws one)."""
    import crucible_pack_toy.longhorizon.task as task_mod
    from crucible_pack_toy.longhorizon.world import draw

    def twin_draw(seed, n=120, **kw):
        b = draw(seed, n, **kw)
        a, c = [t for t in b.tickets if t.kind in ("dup_small", "damaged") and t.expected["refund"] is not None][:2]
        amt = max(a.amount, 60.0)
        for t in (a, c):
            t.amount = amt
            t.expected["refund"] = amt
            b.orders[t.order]["amount"] = amt
        return b

    monkeypatch.setattr(task_mod, "draw", twin_draw)
    t = task_mod.BacklogTask(seed=7, n=12)
    a, c = [x for x in t.b.tickets if x.kind in ("dup_small", "damaged")][:2]
    return t, a, c


@pytest.mark.parametrize("how", ["reused_for_another_ticket", "one_approval_two_refunds"])
async def test_B8_one_approval_covers_one_refund_of_the_ticket_it_names(monkeypatch, how):
    """``approval_reused_twice`` / ``one_approval_two_refunds`` on a backlog: two tickets with one amount, one approval. By
    amount alone (unconsumed) the second refund was grounded too."""
    t, a, c = _twin_amounts(monkeypatch)
    try:
        acts = _b_acts(t.oracle_actions())
        ask_c = [i for i, x in enumerate(acts) if isinstance(x, Message) and re.search(rf"ticket {c.id}\b", x.text)]
        assert ask_c and (await _b_play(t, acts)).reward == 1.0
        if how == "reused_for_another_ticket":  # c's own request dropped: a's approval (same amount) must not ground c's refund
            acts = [x for i, x in enumerate(acts) if i not in ask_c]
        else:  # one request naming both tickets answered once: it covers one of the two refunds
            ask_a = next(i for i, x in enumerate(acts) if isinstance(x, Message) and re.search(rf"ticket {a.id}\b", x.text))
            both = acts[ask_a].text.rstrip(".") + f" and ticket {c.id}, order {c.order}."
            acts = [Message(to="supervisor", text=both) if i == ask_a else x for i, x in enumerate(acts) if i not in ask_c]
        rep = await _b_play(t, acts)
        assert f"ticket:{c.id}" in rep.meta["wrong"] and f"ticket:{a.id}" not in rep.meta["wrong"], rep.meta
    finally:
        t.cleanup()


async def test_B8_supervisor_answers_only_a_request_that_names_the_ticket_or_order():
    from crucible_pack_toy.longhorizon.task import BacklogTask

    t = BacklogTask(seed=20_003, n=12)
    try:
        with tempfile.TemporaryDirectory() as d:
            w = await t.open(Path(d) / "w")
            try:
                await run_episode(w, {"agent": ScriptedPolicy("agent", [Message(to="supervisor", text="Please approve a refund of 30.00 USD."),
                                                                        Message(to="supervisor", text="Please approve a refund of 31.00 USD for ticket 101.")])},
                                  limits=EpisodeLimits(max_steps=5), reset=False)
                replies = [e.text for e in w.log if e.type == "message" and e.actor == "supervisor"]
            finally:
                await w.close()
        assert replies[0].startswith("Please state") and replies[1].startswith("Approved refund of 31.00")
    finally:
        t.cleanup()


async def test_B8_waiting_ticket_stays_pending_until_the_reply():
    """FIN-7 rule 6: escalating a waiting ticket early and resolving it after the reply passed (no early-action rule but the
    resolution time)."""
    from crucible_pack_toy.longhorizon.task import BacklogTask

    t = BacklogTask(seed=1, n=12)
    try:
        waiting = [x for x in t.b.tickets if x.kind == "awaiting"]
        assert waiting
        early = [_tc(900 + k, "tickets.set_status", ticket_id=x.id, status="escalated") for k, x in enumerate(waiting)]
        rep = await _b_play(t, early + _b_acts(t.oracle_actions()))
        assert {f"ticket:{x.id}" for x in waiting} == set(rep.meta["wrong"]), rep.meta
    finally:
        t.cleanup()


# ---- B5 ------------------------------------------------------------------------------------------------------------------

async def _accept_states(pack_dir: Path) -> dict[str, dict]:
    from crucible.accept.run import accept_task, verdict
    from crucible.studio.accept_adapter import pack_accept_tasks
    from crucible.studio.pack import load_manifest

    tasks = await pack_accept_tasks(pack_dir, load_manifest(pack_dir), n_fresh=1)
    out = {}
    try:
        with tempfile.TemporaryDirectory() as d:
            for t in tasks:
                v = verdict(await accept_task(t, cache_root=Path(d), resolve=lambda s: False, reuse=False))
                out[t.task_id] = v
    finally:
        for t in tasks:
            t.cleanup()
    return out


async def test_B5_toy_accept_is_provisional_with_only_A10_pending():
    import crucible_pack_toy

    states = await _accept_states(Path(crucible_pack_toy.__file__).resolve().parents[2])
    assert len(states) == 3  # the helpdesk anchor, the fixture backlog, a generated backlog
    for tid, v in states.items():
        assert v["state"] == "provisional" and v["blocking"] == {"A10": "pending"}, (tid, v)


async def test_B5_scaffolded_pack_accept_is_provisional_with_only_A10_pending(tmp_path):
    from crucible.studio.pack import plan_scaffold, write_scaffold

    dest = tmp_path / "demo"
    write_scaffold(plan_scaffold("demo", dest), tmp_path)
    states = await _accept_states(dest)
    assert len(states) == 3  # two anchors and a generated item
    for tid, v in states.items():
        assert v["state"] == "provisional" and v["blocking"] == {"A10": "pending"}, (tid, v)


async def test_B5_backlog_benchmark_items_pass_only_when_every_ticket_is_right():
    """A3 kills a mutant only at reward 0: under partial credit a mutant that broke one ticket of 12 kept 0.92 and no mutant
    died. Benchmark items (generate) gate on every ticket; outcome keeps the share; a directly built task pays the share."""
    from crucible_pack_toy import studio
    from crucible_pack_toy.longhorizon.task import BacklogTask

    g, s = studio.generate(20_001), BacklogTask(seed=20_001, n=12)
    try:
        acts = _b_acts(g.oracle_actions())
        drop = next(i for i, a in enumerate(acts) if isinstance(a, ToolCall) and a.function_name == "tickets.set_status")
        part = acts[:drop] + acts[drop + 1:]
        rg, rs = await _b_play(g, part), await _b_play(s, part)
        assert rg.reward == 0.0 and rg.gates == {"approvals": "pass", "tickets": "fail"} and rg.outcome == pytest.approx(11 / 12)
        assert rs.reward == pytest.approx(11 / 12) and rs.gates == {}
    finally:
        g.cleanup()
        s.cleanup()


def test_B5_toy_and_backlog_versions_bumped():
    import crucible_pack_toy
    import yaml
    from crucible_pack_toy import studio
    from crucible_pack_toy.longhorizon.task import BacklogTask

    m = yaml.safe_load((Path(crucible_pack_toy.__file__).resolve().parents[2] / "pack.yaml").read_text())
    assert m["version"] == "0.3.0"
    t, g = BacklogTask(seed=1, n=12), studio.generate(5)
    try:
        assert t.id.endswith("@3") and g.id.endswith("@3")
    finally:
        t.cleanup()
        g.cleanup()

# ---------------------------------------------------------------------------------------------------------------- campaign B7 (finance_ap)

HOLDS = ("bank_verification", "price_variance", "missing_receipt")
_MARK = {"bank_verification": "**Bank details.**", "price_variance": "**Price.**", "missing_receipt": "**Receipt.**"}
N_RATES = 2000


def _b7_decide(policy_text: str, w, inv: dict, *, order=None, whole_units=False, dup_same_amount=False, aliases=True, tol=None, exact=False,
               reverse=False) -> tuple[str, str | None]:
    """AP-3 as the policy document states it, with one shortcut deviation at a time."""
    tol = float(re.search(r"by more than ([0-9.]+)%", policy_text).group(1)) / 100 if tol is None else tol
    holds = list(order) if order else sorted(HOLDS, key=lambda r: policy_text.index(_MARK[r]))
    vendors = {v["id"]: v for v in w.vendors}
    v, po = vendors.get(inv["vendor_id"]), next((p for p in w.pos if p["id"] == inv["po_id"]), None)
    c = round(inv["amount"] * 100)
    key = str if exact else _norm
    alias = aliases and "same bank account" in policy_text and v is not None and any(
        x["status"] == "blocked" and x["id"] != v["id"] and x.get("bank_account") == v.get("bank_account") for x in vendors.values())
    if po is None:
        over = False
    elif whole_units:
        over = round(inv["amount"]) > round(po["amount"] * (1 + tol))
    else:
        over = c > round(po["amount"] * 100) * (1 + tol) + 1e-6
    hit = {"vendor_blocked": v is not None and (v["status"] == "blocked" or alias),
           "duplicate": any(h["id"] != inv["id"] and h["vendor_id"] == inv["vendor_id"] and h["status"] == "approved"
                            and key(h["invoice_number"]) == key(inv["invoice_number"]) and (not dup_same_amount or round(h["amount"] * 100) == c)
                            for h in w.invoices),
           "po_mismatch": po is None or po["vendor_id"] != inv["vendor_id"],
           "bank_verification": bool(v and v["bank_changed_unverified"]),
           "price_variance": over,
           "missing_receipt": po is not None and sum(round(r["value"] * 100) for r in w.receipts if r["po_id"] == po["id"]) < c}
    seq = ["vendor_blocked", "duplicate", "po_mismatch", *holds]
    fired = [r for r in (seq[::-1] if reverse else seq) if hit[r]]
    if not fired:
        return ("approved", None)
    return ("rejected" if fired[0] in ("vendor_blocked", "duplicate") else "on_hold", fired[0])


def _shortcuts() -> dict:
    out = {f"fixed_order_{'>'.join(o)}": {"order": o} for o in itertools.permutations(HOLDS)}  # textbook order is one of them
    out |= {"whole_units": {"whole_units": True}, "dup_needs_same_amount": {"dup_same_amount": True}, "ignores_vendor_alias": {"aliases": False},
            "exact_match_duplicates": {"exact": True}, "reverse_rule_order": {"reverse": True}}
    out |= {f"memorized_{t / 10:g}pct": {"tol": t / 1000} for t in (5, 10, 20, 30, 50, 75, 100)}
    out["memorized_2pct_textbook_order"] = {"tol": 0.02, "order": HOLDS}
    return out


@pytest.fixture(scope="module")
def worlds_2000():
    from crucible_pack_finance_ap.world import draw, policy

    out = []
    for s in range(N_RATES):
        w = draw(10_000_000 + s)  # the campaign's seed range
        out.append((w, policy(w), {i["id"]: i for i in w.invoices}))
    return out


def test_B7_no_shortcut_solves_more_than_a_fifth_of_2000_worlds(worlds_2000):
    """Over 2,000 v2 worlds: ignoring the per-world hold order solved 57.8%, whole-unit prices 68.0%, duplicates only at the
    same amount 54.5%, and no world held a vendor alias. Now every shortcut solves at most ~1/6 (any fixed hold order: the
    world draws its order uniformly; a memorized tolerance: 1/7 of worlds drew it)."""
    rates = {}
    for name, kw in _shortcuts().items():
        solved = sum(all(_b7_decide(pt, w, inv[c.invoice_id], **kw) == c.expected for c in w.cases) for w, pt, inv in worlds_2000)
        rates[name] = solved / len(worlds_2000)
    print({k: round(v, 4) for k, v in rates.items()})
    assert all(r <= 0.20 for r in rates.values()), {k: v for k, v in rates.items() if v > 0.20}
    for name in ("whole_units", "dup_needs_same_amount", "exact_match_duplicates", "reverse_rule_order"):
        assert rates[name] == 0.0, (name, rates[name])  # each guaranteed case defeats it in every world


def test_B7_every_world_holds_the_guaranteed_cases(worlds_2000):
    alias_worlds = 0
    for w, pt, inv in worlds_2000:
        order = sorted(HOLDS, key=lambda r: pt.index(_MARK[r]))
        world_dec = {c.invoice_id: _b7_decide(pt, w, inv[c.invoice_id]) for c in w.cases}
        assert world_dec == {c.invoice_id: c.expected for c in w.cases}, w.seed  # labels = AP-3 applied to the data
        # (a) every fixed hold order other than the world's own decides some invoice differently
        for o in itertools.permutations(HOLDS):
            if list(o) != order:
                assert any(_b7_decide(pt, w, inv[c.invoice_id], order=o) != world_dec[c.invoice_id] for c in w.cases), (w.seed, o)
        # (b) a duplicate an exact match misses, billed at a restated amount
        assert any(c.expected == ("rejected", "duplicate") and _b7_decide(pt, w, inv[c.invoice_id], dup_same_amount=True) != c.expected
                   and _b7_decide(pt, w, inv[c.invoice_id], exact=True) != c.expected for c in w.cases), w.seed
        # (c) a price one cent over the bar that a whole-unit comparison passes
        pos = {p["id"]: p for p in w.pos}
        assert any(c.expected == ("on_hold", "price_variance") and c.po_id in pos
                   and round(c.amount * 100) * 1000 - round(pos[c.po_id]["amount"] * 100) * (1000 + w.tol_pm) in range(1, 1001)
                   and _b7_decide(pt, w, inv[c.invoice_id], whole_units=True)[0] == "approved" for c in w.cases), w.seed
        # (d) a payee the vendor rule stops; an alias is a blocked vendor's bank account under another id and name
        assert any(c.expected == ("rejected", "vendor_blocked") for c in w.cases), w.seed
        vendors = {v["id"]: v for v in w.vendors}
        alias = [c for c in w.cases if c.expected == ("rejected", "vendor_blocked") and vendors[c.vendor_id]["status"] == "active"]
        for c in alias:
            twin = [x for x in w.vendors if x["status"] == "blocked" and x["bank_account"] == vendors[c.vendor_id]["bank_account"]]
            assert twin and all(x["name"] != vendors[c.vendor_id]["name"] for x in twin), w.seed  # a name lookup does not find it
        alias_worlds += bool(alias)
        assert len({v["bank_account"] for v in w.vendors}) == len(w.vendors) - len(alias), w.seed  # accounts are unique otherwise
    assert 0.8 <= alias_worlds / len(worlds_2000) <= 0.9, alias_worlds




async def _b7_play(task, actions):
    with tempfile.TemporaryDirectory() as d:
        w = await task.open(Path(d) / "work")
        try:
            await run_episode(w, {"agent": ScriptedPolicy("agent", list(actions))}, limits=EpisodeLimits(max_steps=max(50, 2 * len(actions))), reset=False)
            return await task.grade(w, Path(d) / "grade"), list(w.failures), list(w.log)
        finally:
            await w.close()


async def test_B7_campaign_adversaries_are_declared_and_score_zero_everywhere():
    from crucible_pack_finance_ap import studio

    advs = {a["name"]: a for a in studio.adversaries()}
    assert {"adversary:price_round_units", "adversary:dup_needs_same_amount", "adversary:alias_vendor_args"} <= set(advs)
    assert all(a["expect_max"] == 0.0 for a in advs.values())
    tasks = [t for t, _ in studio.tasks()] + [studio.generate(s) for s in (101, 202, 303, 404)]
    try:
        for t in tasks:
            for a in advs.values():
                rep, failures, _ = await _b7_play(t, a["policy"](t))
                assert failures == [] and rep.reward == 0.0, (a["name"], t.id, rep.reward)
    finally:
        for t in tasks:
            t.cleanup()


async def test_B7_the_alias_is_found_through_the_vendor_master():
    from crucible_pack_finance_ap.studio import APTask, _acts
    from crucible_pack_finance_ap.world import draw

    t = APTask(next(s for s in range(50) if any("vendor_alias" in c.traits for c in draw(s).cases)))
    try:
        c = next(c for c in t.w.cases if "vendor_alias" in c.traits)
        acct = next(v["bank_account"] for v in t.w.vendors if v["id"] == c.vendor_id)
        with tempfile.TemporaryDirectory() as d:  # the agent's own tool call, through the World
            w = await t.open(Path(d) / "work")
            try:
                obs = (await w.step("agent", TC("erp.list_vendors", bank_account=acct))).observation.model_dump_json()
                assert obs.count(acct) == 2 and '\\"blocked\\"' in obs and c.vendor_id in obs, obs[:2000]
                obs = (await w.step("agent", TC("erp.get_vendor", vendor_id=c.vendor_id))).observation.model_dump_json()
                assert acct in obs, obs[:2000]
            finally:
                await w.close()
        assert "same bank account" in (t.root / "data" / "kb" / "ap_policy.md").read_text()
        rep, _, _ = await _b7_play(t, _acts(t.oracle))
        assert rep.reward == 1.0
    finally:
        t.cleanup()


# ---------------------------------------------------------------------------------------------------------------- campaign B2, B11 (coding)

def _campaign_token(pat: str) -> str:
    if pat.endswith(r"\s*\("):
        return re.sub(r"\\(.)", r"\1", pat[: -len(r"\s*\(")]) + "("
    return re.sub(r"\\(.)", r"\1", pat.replace(r"\b", ""))


async def _campaign_attack(task: str, mode: str):
    """The validation campaign's probe ($V/coding_adv.py): nothing migrated, every added token in a string literal
    (``string``) or under ``#if 0`` / ``if False:`` / ``if(FALSE)`` (``dead``), the removed tokens rewritten away, the
    deleted files deleted, a no-op statement in each anchored block, a live placeholder in each created file."""
    from crucible_pack_coding import studio
    from crucible_pack_coding.m56.task_properties import base_reader

    from crucible.workflow.filestate import FileState

    fs = FileState(base=base_reader(task))
    n = 0
    for m in await studio._mine_cached(task):
        q, e = m.check.query, m.check.expect
        p = q.args.get("path") or (q.args.get("paths") or [None])[0]
        if p is None:
            continue
        cur, py, hashc = fs.read(p), p.endswith(".py"), bool(re.search(r"(?:^|/)CMakeLists\.txt$|\.(?:cmake|toml|ya?ml|cfg|ini|sh)$", p))
        n += 1
        if q.op == "grep" and e.kind == "count_ge":
            tok = _campaign_token(q.args["pattern"])
            if mode == "string":
                line = f'_crucible_s{n} = "{tok}"' if py else (f'set(_CRUCIBLE_S{n} "{tok}")' if hashc else f'static const char *crucible_s{n} = "{tok}";')
            else:
                line = f"if False:\n    {tok}" if py else (f"if(FALSE)\n  {tok}\nendif()" if hashc else f"#if 0\n{tok}\n#endif")
            fs.write(p, (cur or "") + "\n" + line + "\n")
        elif q.op == "grep":
            fs.write(p, re.sub(q.args["pattern"], "zz", cur or ""))
        elif q.op == "exists" and e.kind == "truthy":
            fs.write(p, (cur or "") + ('_crucible_placeholder = "x"\n' if py else ("set(_X 1)\n" if hashc else "static int crucible_placeholder;\n")))
        elif q.op == "exists":
            fs.write(p, None)
        elif q.op == "block_changed":
            lines = (cur or "").split("\n")
            i = next((k for k, ln in enumerate(lines) if q.args["anchor"] in ln), None)
            if i is not None:
                lines.insert(i + 1, "    pass" if py else ("" if hashc else "    (void)0;"))
                fs.write(p, "\n".join(lines))
    return fs


@needs_seeds
@pytest.mark.parametrize("task", ["task02_pcops_triton", "task03_mdlite_pallas", "task04_pcl_gpu_cuda12"])
async def test_B2_string_literal_and_dead_code_stuffing_earn_no_dense_credit(task):
    """Campaign B2 (P0): the oracle's tokens in string literals or dead code reached Φ 1.0 on every bench seed with nothing
    migrated, the same as gold, while honest failures scored 0.06-0.53."""
    from crucible_pack_coding import studio

    for mode in ("string", "dead"):
        phi = await _phi_of_state(task, await _campaign_attack(task, mode))
        assert phi == 0.0, (task, mode, phi)  # was 1.0
    assert await studio._phi(task, (MT / task / "solution" / "solution.patch").read_text(errors="replace")) == pytest.approx(1.0)
    honest = await studio._phi(task, (MT / task / "runs" / ("gpt-5.6-sol-v3-run1" if task.startswith("task04") else "gpt-5.6-sol-run1")
                                       / "model.patch").read_text(errors="replace"))
    assert 0.03 <= honest <= 0.6, (task, honest)  # honest partial work keeps its credit


def test_B2_code_view_drops_literals_and_dead_code_and_keeps_live_code():
    from crucible_pack_coding.codeonly import code_only, undefined_macros, unused_new

    c = textwrap.dedent(r'''
        static const char *s = "cudaLit1(";  // cudaComment(
        auto r = R"x(cudaLit2( )" )x";
        char q = '\'';
        int big = 1'000'000;
        #if 0
        cudaDead1();
        #elif 1
        liveA();
        #endif
        #ifdef CRUCIBLE_NEVER_DEFINED
        cudaDead2();
        #endif
        #ifdef __CUDACC__
        liveB();
        #endif
        #if CUDART_VERSION >= 12000
        liveC();
        #endif
        static void crucible_dead(void) { cudaDead3(); }
        static int f(int x) {
          if (0) { cudaDead4(); } else { liveD(); }
          if (1) liveE(); else cudaDead5();
          while (false) { cudaDead6(); }
          do { liveF(); } while (0);
          switch (x) { case 1: return 1; cudaDead7(); case 2: liveG(); }
          return x;
          cudaDead8();
        }
        int g(void) { return f(1); }
        #define CRUCIBLE_CHECK(x) if (!(x)) return;
        static int liveH = 1;
        #if (0)
        #endif
        int h(int x) { for (;;) if (x) break; liveI(); if (x) return 2; liveJ(); return liveH; }
    ''')
    files = {"a.cu": c}
    undef = undefined_macros(files, {"f", "g", "h", "x"})
    drop = unused_new(files, {"f", "g", "h", "x"}, undef)
    assert undef == {"CRUCIBLE_NEVER_DEFINED"} and drop == {"a.cu": {"crucible_dead"}}
    code = code_only("a.cu", c, undef, drop.get("a.cu", frozenset()))
    assert not re.search(r"cuda(?:Lit|Comment|Dead)", code), code
    assert all(f"live{k}()" in code for k in "ABCDEFGIJ") and "liveH = 1" in code and "1'000'000" in code, code
    py = textwrap.dedent('''
        """cudaDoc"""
        X = "torch.lit1("
        def used():
            if False:
                torch.dead1()
            y = torch.dead2() if False else torch.live1()
            z = False and torch.dead3()
            return f"a{torch.live2()}b"
            torch.dead4()
        def _crucible_dead():
            torch.dead5()
        @register
        def kept():
            torch.live3()
        used()
    ''')
    code = code_only("m.py", py, frozenset(), unused_new({"m.py": py}, {"used", "register"}).get("m.py", frozenset()))
    assert not re.search(r"torch\.(?:lit|dead)|cudaDoc", code) and all(f"torch.live{k}()" in code for k in "123"), code
    assert code_only("m.py", "def f(:\n    torch.live()\n") is None  # a file that does not parse has no code
    cm = 'set(X "target_compile_options(")\nif(FALSE)\n  dead_a()\nelse()\n  live_a()\nendif()\nfunction(_crucible_dead)\n  dead_b()\nendfunction()\n'
    code = code_only("x.cmake", cm, frozenset(), unused_new({"x.cmake": cm}, set()).get("x.cmake", frozenset()))
    assert "target_compile_options" not in code and "dead_" not in code and "live_a()" in code, code
    toml = '[project]\ndescription = "operators"  # reduction\ndependencies = ["triton"]\n'
    assert code_only("pyproject.toml", toml) == '[project]\ndescription = ""  \ndependencies = [""]\n'
    assert code_only("pyproject.toml", "x = [broken") is None


@needs_seeds
async def test_B2_coding_declares_literal_and_dead_code_adversaries_bounded_at_zero():
    from crucible_pack_coding import studio

    advs = {a["name"]: a for a in studio.adversaries()}
    assert {"adversary:string_literal_stuffing", "adversary:dead_code_stuffing"} <= set(advs)
    assert all(a["expect_max"] == 0.0 for a in advs.values())
    for seed in ("task01v2_tessera_omp", "task04_pcl_gpu_cuda12"):
        for mode in ("literal", "dead"):
            assert await studio.phi_of_state(seed, await studio.stuffed_literal_state(seed, mode)) == 0.0, (seed, mode)
    t = next(t for t, _ in await studio.tasks() if t.seed == "task02_pcops_triton")
    for name in ("adversary:string_literal_stuffing", "adversary:dead_code_stuffing"):
        rep, failures, _ = await play(t, advs[name]["policy"](t))
        assert failures == [] and rep.reward == 0.0, (name, rep.reward)


def test_B11_coding_semantic_level_needs_no_gpu():
    """Campaign B11: ``requires: [gpu, docker]`` skipped the coding pack's semantic and adversary level on every CPU host,
    although every hook (``tasks``, ``fixtures``, ``credit``) runs on CPU. Live GPU grading is outside the pack's hooks."""
    import crucible_pack_coding

    from crucible.studio.pack import load_manifest, missing_requirements

    d = Path(crucible_pack_coding.__file__).resolve().parents[2]
    m = load_manifest(d)
    assert "gpu" not in m.requires and missing_requirements(m) == []

if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
