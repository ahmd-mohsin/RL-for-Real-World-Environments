"""A grade never blames the policy for an outage (M1 audit 2026-10-14 R-11): a check that dies on a lost container is
masked as crucible:env_container_lost, and SqlAssertion fails only when the policy broke the schema it reads (the rule
state_predicates already applied); any other failed query is infrastructure."""

from __future__ import annotations

from pathlib import Path

from crucible.core.schema.state import QueryResult
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, _run_check
from crucible.verify.runtime.trusted import Trusted
from crucible.verify.sql_checks import SqlAssertion
from crucible.world.exec import EnvContainerLost, ExecError


class View:
    """A verifier view whose queries answer with ``result`` (a QueryResult) or raise it (an exception)."""

    def __init__(self, result):
        self.result = result

    async def query(self, app, q):
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def wrapped() -> ExecError:
    try:
        raise EnvContainerLost("the task container c1 was removed")
    except EnvContainerLost as e:
        try:
            raise ExecError("python snippet failed") from e
        except ExecError as outer:
            return outer


async def test_a_check_dying_on_a_lost_container_is_infra():
    def raising(exc):
        def fn(ctx):
            raise exc
        return Check("c", "g", fn)

    r = await _run_check(raising(EnvContainerLost("the task container c1 was removed")), None)
    assert r.status == "infra" and r.detail == {"infra": "the task container c1 was removed", "failure_kind": "crucible:env_container_lost"}
    assert (await _run_check(raising(wrapped()), None)).status == "infra"  # found through the cause chain
    assert (await _run_check(raising(KeyError("x")), None)).status == "fail"  # any other crash stays the candidate's


async def test_sql_assertion_fails_only_on_the_policys_schema_change():
    a = SqlAssertion("refunds_made", "erp", "SELECT COUNT(*) FROM refunds", 1)
    ok = await a.evaluate(View(QueryResult(ok=True, value={"columns": ["n"], "rows": [[1]]})))
    dropped = await a.evaluate(View(QueryResult(ok=False, error="ExecError: python snippet failed: sqlite3.OperationalError: no such table: refunds")))
    transport = await a.evaluate(View(QueryResult(ok=False, error="ExecError: docker cp from /db failed: connection reset by peer")))
    assert ok.passed and ok.infra is None
    assert not dropped.passed and dropped.infra is None  # scored: the policy dropped the table
    assert not transport.passed and "could not be run" in (transport.infra or "")  # masked: before, a scored fail


async def test_a_grade_on_a_lost_container_is_masked(tmp_path: Path):
    check = SqlAssertion("refunds_made", "erp", "SELECT COUNT(*) FROM refunds", 1).check("state")
    rep = await Plan("t/lost", [Gate("state", kind="outcome")], [check]).arun(
        Context(Trusted(tmp_path / "g"), params={"view": View(EnvContainerLost("the task container c1 was removed"))}))
    assert rep.infra["status"] == "masked" and rep.infra["failure_kind"] == "crucible:env_container_lost"
    assert "c1 was removed" in rep.infra["reason"]


def test_the_shipped_verifier_runtime_imports_only_itself():
    """``crucible/verify/runtime`` is copied alone into verifier images (``/crucible_verify``): an import of any other
    crucible module crashes every grade there (R-11 first imported ``crucible.world.exec``; the red-team graders caught it)."""
    import ast

    import crucible.verify.runtime as rt

    root = Path(rt.__file__).parent
    bad = []
    for f in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            bad += [f"{f.relative_to(root)}: {n}" for n in names if n.split(".")[0] == "crucible" and not n.startswith("crucible.verify.runtime")]
    assert not bad, bad
