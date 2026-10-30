"""The typed per-task loop (meta.rsi.TaskLoop) and the ORACLE class (falsify.oracle), with a scripted student and
classifier on the fixture AP task (no model is called): REPAIR, DESCEND, ASCEND, KEEP, RETIRE and HOLD, each regulariser
refusing what it must, sample splitting of diagnosis and acceptance rollouts, and the reference-decision precondition."""

from __future__ import annotations

import pytest

from crucible.synth.archive import Archive
from crucible.synth.falsify import FalsifyContext
from crucible.synth.falsify.base import falsify
from crucible.synth.falsify.oracle import ORACLE, decision_disagreement, pinned
from crucible.synth.meta.rsi import KnobMove, ProbeOutcome, TaskLoop, TaskRules, archive_recheck, replay_program
from crucible.synth.program import Program, Step, lit, ref
from tests.unit import synth_fixtures as fx

BASE = fx.in_domain(fx.t1_hold())


def _hold(reason: str) -> dict:
    p = Program(steps=[Step(id="s1", tool="ap.find_invoice", args={"number": lit("INV-1003")}),
                       Step(id="s3", tool="ap.hold_invoice", args={"invoice_id": ref("s1.invoices.0.id"), "reason": lit(reason)})])
    return p.model_dump(mode="json")


GOOD = BASE.oracle.model_dump(mode="json")
READ_ONLY = Program(steps=BASE.oracle.steps[:2]).model_dump(mode="json")  # gives up after reading: a skill-gap failure


class R:
    def __init__(self, i, passed, program):
        self.i, self.passed, self.program = i, passed, program


class Student:
    """Pass rate by the task id's tag: 'easy' all, 'hard' none, otherwise one in three. Failures are read-only give-ups
    (or, with ``wrong``, a hold with another reason). Records every rollout index it was asked for."""

    def __init__(self, ctx, wrong: str | None = None):
        self.ctx, self.wrong, self.asked = ctx, wrong, []
        self.student, self.checkpoint, self.simulator, self.time_mode = "student", "base", "json-agent/v1", "frozen"

    def _pass(self, task, i):
        tag = task.id.rsplit("|", 1)[-1]
        return {"easy": True, "hard": False}.get(tag, i % 3 == 0)

    async def rollouts(self, task, n, *, start=0):
        out = []
        for i in range(start, start + n):
            self.asked.append((task.id, i))
            prog = GOOD if self._pass(task, i) else (_hold(self.wrong) if self.wrong else READ_ONLY)
            out.append(R(i, await replay_program(task, prog, self.ctx), prog))
        return out


class Classifier:
    """intended_pass / intended_fail by the grader's verdict, unless ``override`` (rollout index → (kind, mechanism, evidence))."""

    def __init__(self, override=None):
        self.override = override or {}

    async def classify(self, task, rollouts):
        out = []
        for r in rollouts:
            k, m, e = self.override.get(r.i, ("intended_pass" if r.passed else "intended_fail", "", "executed"))
            out.append(ProbeOutcome(i=r.i, passed=r.passed, kind=k, mechanism=m, evidence=e, program=r.program))
        return out


class Knobs:
    def __init__(self, allow=("easier", "harder"), cell_change=False):
        self.allow, self.cell_change = allow, cell_change

    async def moves(self, task, direction):
        return [KnobMove(knob="toy", direction=direction)] if direction in self.allow else []

    async def apply(self, task, move):
        tag = "band"
        upd = {"id": f"{task.id.rsplit('|', 1)[0]}|{move.direction}-{tag}"}
        if self.cell_change:
            upd["intent"] = task.intent.model_copy(update={"skill": "a different skill"})
        return task.model_copy(update=upd, deep=True)


def _loop(tmp_path, *, tag="hard", knobs=None, classifier=None, oracle=None, alt=True, repair=None, wrong=None, rules=None):
    ctx = FalsifyContext()
    arch = Archive(tmp_path / "arch")
    task = BASE.model_copy(update={"id": f"{BASE.id}|{tag}"}, deep=True)

    async def rep(t, cxs):
        return None

    async def replay(t, prog):
        return await replay_program(t, prog, ctx)

    async def alt_check(t):
        return alt

    async def p3(t):
        return True, []

    student = Student(ctx, wrong=wrong)
    loop = TaskLoop(tmp_path / "loop", probe=student, classify=classifier or Classifier(), repair=repair or rep, replay=replay,
                    recheck=archive_recheck(arch, ctx), knobs=knobs or Knobs(), ctx=ctx, oracle=oracle, p3_check=p3, alt_check=alt_check,
                    cell_of=lambda t: t.intent.skill, archive=arch, rules=rules or TaskRules(rollouts=8, kappa=5.0, student_family="student"))
    loop.add_tasks([task])
    return loop, task, student


async def test_too_hard_descends_to_a_new_variant_on_fresh_rollouts_and_keeps_the_original(tmp_path):
    loop, task, _student = _loop(tmp_path, tag="hard")
    (d,) = await loop.run_typed_round(1)
    assert d.move == "DESCEND" and d.accepted and d.variant_id and d.band["decision"] == "too_hard" and d.acceptance_band["decision"] != "too_hard"
    assert task.id in loop.tasks and d.variant_id in loop.tasks  # the original is kept as the harder rung
    diag, acc = set(d.rollouts["diagnosis"]), set(d.rollouts["acceptance"])
    assert diag and acc and diag.isdisjoint(acc)  # G4: the rollouts that diagnosed never accept
    assert loop.lineages[task.id].direction == "down" and loop.knob_calibration[-1]["moved"]


async def test_too_easy_ascends_and_a_lineage_never_moves_both_ways(tmp_path):
    loop, task, _ = _loop(tmp_path, tag="easy")
    (d,) = await loop.run_typed_round(1)
    assert d.move == "ASCEND" and d.accepted and loop.lineages[task.id].direction == "up"
    hard = task.model_copy(update={"id": f"{BASE.id}|hard"}, deep=True)  # a lineage member that now reads too hard
    loop.tasks[hard.id] = hard
    loop.of[hard.id] = task.id
    d2 = await loop.step(hard, 2)
    assert d2.move == "RETIRE" and "oscillate" in " ".join(d2.why)


async def test_in_band_keeps_and_no_lawful_move_retires(tmp_path):
    loop, _, _ = _loop(tmp_path, tag="band", rules=TaskRules(rollouts=8, kappa=5.0, looks=(8, 16, 24)))
    (d,) = await loop.run_typed_round(1)
    assert d.move in ("KEEP", "HOLD")  # one in three: in band, or undecided at k_max (never an edit)
    loop2, _, _ = _loop(tmp_path / "b", tag="hard", knobs=Knobs(allow=("harder",)))
    (d2,) = await loop2.run_typed_round(1)
    assert d2.move == "RETIRE" and "no declared knob" in " ".join(d2.why)


async def test_variants_that_change_what_is_taught_or_lack_an_independent_solver_are_refused(tmp_path):
    loop, _, _ = _loop(tmp_path / "a", tag="hard", knobs=Knobs(cell_change=True))
    (d,) = await loop.run_typed_round(1)
    assert d.move == "DESCEND" and not d.accepted and any(r.startswith("G2") for r in d.refused_by)
    loop2, _, _ = _loop(tmp_path / "b", tag="hard", alt=False)
    (d2,) = await loop2.run_typed_round(1)
    assert not d2.accepted and any(r.startswith("G6") for r in d2.refused_by)
    loop3, _, _ = _loop(tmp_path / "c", tag="hard", knobs=Knobs(cell_change=True), rules=TaskRules(rollouts=8, kappa=0.0001))  # a new cell moves the pool
    (d3,) = await loop3.run_typed_round(1)
    assert not d3.accepted and any(r.startswith("G7") for r in d3.refused_by)


async def test_an_unintended_pass_is_repaired_and_accepted_only_on_fresh_rollouts(tmp_path):
    over = {0: ("unintended_pass", "leak", "executed")}  # the diagnosis's first rollout (a pass) is a leak

    async def fix(t, cxs):
        return t.model_copy(update={"version": t.version + 1, "instruction": t.instruction + " "})

    loop, _task, _ = _loop(tmp_path, tag="band", classifier=Classifier(over), repair=fix)
    (d,) = await loop.run_typed_round(1)
    assert d.move == "REPAIR" and d.refused_by and any(x.startswith("T4") for x in d.refused_by)  # the leak's program still passes: not closed
    keep = {}

    async def fixed(t, cxs):
        keep["n"] = len(cxs)
        return t.model_copy(update={"version": t.version + 1})

    class Fresh(Classifier):  # the diagnosis says leak; the fresh acceptance rollouts are clean
        async def classify(self, t, rollouts):
            outs = await super().classify(t, rollouts)
            return [o.model_copy(update={"kind": "intended_pass"}) if o.i >= 8 and o.kind.startswith("unintended") else o for o in outs]

    async def no_replay_check(t, prog):
        return None  # T2-T4 are replay checks; here only the fresh-rollout rule decides

    loop2, _task2, _ = _loop(tmp_path / "b", tag="band", classifier=Fresh(over), repair=fixed)
    loop2.replay = no_replay_check

    async def clean(t):
        return []

    loop2.recheck = clean  # the archive rule is tested elsewhere: here only the fresh-rollout rule decides
    (d2,) = await loop2.run_typed_round(1)
    assert d2.move == "REPAIR" and d2.accepted and keep["n"] == 1 and set(d2.rollouts["acceptance"]).isdisjoint(d2.rollouts["diagnosis"])


async def test_a_wrong_reference_decision_is_repaired_first_and_a_disputed_one_is_held(tmp_path):
    truth = ORACLE(truths=[lambda t: "wrong" if t.id.startswith(BASE.id) else None])
    loop, _task, _ = _loop(tmp_path / "a", tag="hard", oracle=truth)
    (d,) = await loop.run_typed_round(1)
    assert d.move == "REPAIR" and d.oracle["status"] == "wrong" and "ORACLE" in " ".join(d.why)  # never DESCENDed
    loop2, _, _ = _loop(tmp_path / "b", tag="hard", wrong="missing_receipt")  # every failing rollout holds with one other reason
    (d2,) = await loop2.run_typed_round(1)
    assert d2.move == "HOLD" and d2.oracle["status"] == "disputed" and loop2.lineages[d2.task_id].status == "held"


async def test_mostly_uncertain_evidence_holds():
    import tempfile
    from pathlib import Path

    unc = {i: ("unintended_fail", "routed", "none") for i in range(8)}
    with tempfile.TemporaryDirectory() as td:
        loop, _, _ = _loop(Path(td), tag="hard", classifier=Classifier(unc))
        (d,) = await loop.run_typed_round(1)
        assert d.move == "HOLD" and "uncertain" in " ".join(d.why)


async def test_the_oracle_class_and_solver_disagreement_on_the_fixture():
    ctx = FalsifyContext()
    assert {p[3] for p in pinned(BASE)} >= {"status", "reason"}
    r = await falsify(BASE, ORACLE(truths=[lambda t: "wrong"]), ctx)
    assert r.c == 1 and r.confirmed[0].defect == "D6" and r.confirmed[0].op == "policy"
    assert (await falsify(BASE, ORACLE(truths=[lambda t: "right"]), ctx)).c == 0
    na = await falsify(BASE, ORACLE(), ctx)
    assert not na.applicable and "no executable policy truth" in na.reason
    d = await decision_disagreement(BASE, [_hold("missing_receipt"), _hold("missing_receipt")], ctx)
    assert d.disputed and d.fields[0]["alternative"] == '"missing_receipt"' and d.fields[0]["oracle"] == "price_variance"
    assert not (await decision_disagreement(BASE, [_hold("missing_receipt"), READ_ONLY], ctx)).disputed  # one dissenting run is not a dispute
    assert not (await decision_disagreement(BASE, [READ_ONLY, READ_ONLY], ctx)).disputed  # runs that decide nothing


@pytest.mark.parametrize("k", [1])
def test_rank_order(k):
    from crucible.synth.meta.rsi import RANK

    assert RANK["too_easy"] < RANK["in_band"] < RANK["too_hard"]
