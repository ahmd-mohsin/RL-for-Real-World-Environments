"""M3 audit (2026-10-12), batch D: event logs, CI logs, git and GitHub history. Each test reproduced a defect before
its fix (``docs/audit/2026-10-12/M03-connectors.md`` §2.4)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from crucible.connectors.file_drop import FieldMap, FileDropConnector, rows_to_trace
from crucible.connectors.trace_store import TraceStore
from crucible.core.store import Store


def _ci(tmp_path: Path, text: str | bytes, name: str = "run.log"):
    from crucible_pack_coding.connectors.ci_logs import CILogConnector

    p = tmp_path / name
    p.write_bytes(text if isinstance(text, bytes) else text.encode())
    c = CILogConnector(tmp_path)
    [ref] = [r for r in c.discover({"glob": name})]
    return c.normalize(c.fetch(ref))


@pytest.mark.parametrize("log", [
    "tests/t.py::test_ok PASSED [ 50%]\ntests/t.py::test_p[a b] FAILED [100%]\nFAILED tests/t.py::test_p[a b] - assert 0\n=== 1 failed, 1 passed in 0.1s ===\n",
    "[==========] 2 tests from 1 test suite ran.\n[       OK ] Suite.Ok (0 ms)\n[  FAILED  ] Inst/Suite.Test/1, where GetParam() = 4 (0 ms)\n[==========] 2 tests ran. (1 ms total)\n",
    "ERROR tests/test_a.py\n!!!!!!! Interrupted: 1 error during collection !!!!!!!\n=== 1 error in 0.12s ===\n",
])
def test_D1_D3_a_failing_run_is_never_good(tmp_path, log):
    t = _ci(tmp_path, log)
    assert t.outcome.label == "bad", (t.outcome, [e.text for e in t.events])  # was good (missed failure) or unknown (collection error)


def test_D2_a_test_failing_on_a_refused_connection_is_a_test_failure(tmp_path):
    log = ("tests/t.py::test_ok PASSED\ntests/t.py::test_api FAILED\n"
           "FAILED tests/t.py::test_api - requests.exceptions.ConnectionError: [Errno 111] Connection refused\n=== 1 failed, 1 passed in 0.1s ===\n")
    assert _ci(tmp_path, log).outcome.label == "bad"  # masked as infra: a code failure dropped from the data


def test_D4_a_run_killed_midway_is_not_good(tmp_path):
    t = _ci(tmp_path, "tests/t.py::test_ok PASSED\nError: Process completed with exit code 137.\n")
    assert t.outcome.label == "unknown" and t.events[0].data["failure_kind"] == "infra"


def test_D22_downloaded_logs_keep_their_tests(tmp_path):
    log = ("2026-10-11T12:00:00.1234567Z tests/t.py::test_ok \x1b[32mPASSED\x1b[0m\n2026-10-11T12:00:01.0000000Z tests/t.py::test_bad \x1b[31mFAILED\x1b[0m\n"
           "2026-10-11T12:00:02.0000000Z === 1 failed, 1 passed in 0.1s ===\n")
    t = _ci(tmp_path, log)
    assert sorted(e.data["name"] for e in t.events[1:]) == ["tests/t.py::test_bad", "tests/t.py::test_ok"]  # zero checks: flaky() lost them
    junit = '<?xml version="1.0"?><testsuite><testcase classname="a" name="b"><failure message="x"/></testcase></testsuite>'
    assert _ci(tmp_path, "﻿" + junit, "j.xml").outcome.label == "bad"
    assert _ci(tmp_path, junit.encode("utf-16"), "k.xml").outcome.label == "bad"  # detected as unknown
    prose = "tests/t.py::test_x FAILED\nall tests passed, thanks\n=== 1 failed in 0.1s ===\n"
    assert _ci(tmp_path, prose, "p.log").events[0].data["format"] == "pytest"  # read as CTest


def test_D28_D29_ctest_exceptions_and_reruns(tmp_path):
    ctest = ("1/2 Test #1: ok ...............   Passed    0.01 sec\n2/2 Test #2: segv .............***Exception: SegFault  0.02 sec\n"
             "50% tests passed, 1 tests failed out of 2\n")
    t = _ci(tmp_path, ctest)
    assert any(e.data.get("name") == "segv" and e.data["status"] == "fail" for e in t.events)
    junit = ('<?xml version="1.0"?><testsuite><testcase classname="a" name="flappy"><flakyFailure message="x"/></testcase></testsuite>')
    t = _ci(tmp_path, junit, "f.xml")
    assert t.events[1].data.get("flaky") is True


def test_D17_ci_reruns_at_one_path_are_separate_runs(tmp_path):
    a = _ci(tmp_path, "tests/t.py::test_x PASSED\n=== 1 passed in 0.1s ===\n").trace_id
    b = _ci(tmp_path, "tests/t.py::test_x FAILED\n=== 1 failed in 0.1s ===\n").trace_id
    assert a != b  # a rerun overwrote the first run, so flaky() never saw both


def test_D12_D14_D15_D20_D21_file_drop_exports(tmp_path):
    (tmp_path / "latin.csv").write_bytes("case,time,actor,text\n1,2024-05-01T10:00:00Z,Jos\xe9,hola\n".encode("latin-1"))
    assert [r.key for r in FileDropConnector().discover({"glob": str(tmp_path / "latin.csv")})] == ["1"]  # UnicodeDecodeError
    (tmp_path / "upper.CSV").write_text("case,time,actor,text\n2,2024-05-01T10:00:00Z,a,x\n")
    assert [r.key for r in FileDropConnector().discover({"glob": str(tmp_path / "upper.CSV")})] == ["2"]  # parsed as JSONL
    (tmp_path / "ls.jsonl").write_text(json.dumps({"case": "3", "time": "2024-05-01T10:00:00Z", "text": "a b"}, ensure_ascii=False) + "\n")
    assert [r.key for r in FileDropConnector().discover({"glob": str(tmp_path / "ls.jsonl")})] == ["3"]  # splitlines() split U+2028
    (tmp_path / "day1.jsonl").write_text('{"case":"4","id":"1","time":"2024-05-01T10:00:00Z","type":"opened"}\n')
    (tmp_path / "day2.jsonl").write_text('{"case":"4","id":"1","time":"2024-05-01T10:00:00Z","type":"opened"}\n{"case":"4","id":"2","time":"2024-05-01T11:00:00Z","type":"closed"}\n')
    c = FileDropConnector()
    [ref] = list(c.discover({"glob": str(tmp_path / "day*.jsonl")}))
    assert [e.type for e in c.normalize(c.fetch(ref)).events] == ["opened", "closed"]  # overlapping exports: opened twice
    r1 = {"case": "5", "time": "2024-05-01T10:00:00Z", "actor": "a", "text": "x"}
    r2 = {"case": "5", "time": "2024-05-01T10:00:00Z", "actor": "b", "text": "y"}
    f = FieldMap()
    assert rows_to_trace([r1, r2], f, case="5", connector="file_drop").content_hash() == rows_to_trace([r2, r1], f, case="5", connector="file_drop").content_hash()
    t = rows_to_trace([{"case": "6", "time": "03/02/2026 09:00", "actor": "copilot", "actor_kind": "Bot"}], f, case="6", connector="file_drop")
    assert t.events[0].time == "2026-03-02T09:00:00Z" and t.actors[0].kind == "agent"  # kept as is (crashed M2 later); a bot became a human


def test_D10_row_ids_are_kept_and_signals_survive_a_grown_log(tmp_path, monkeypatch):
    from crucible.core.schema.signal import Signal

    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "s")
    f = FieldMap()
    v1 = rows_to_trace([{"case": "T1", "id": "1", "time": "2024-05-01T01:00:00Z", "type": "refund", "text": "refund approved"}], f, case="T1", connector="file_drop")
    assert v1.events[0].data["row_id"] == "1"  # the source row id was dropped
    ts = TraceStore(Store(tmp_path / "store"))
    ts.put(v1)
    ts.put(v1.model_copy(update={"signals": [Signal(id="s1", type="correction", polarity=-1, grades=[0], source_event=0)]}))
    v2 = rows_to_trace([{"case": "T1", "id": "0", "time": "2024-05-01T00:00:00Z", "type": "opened", "text": "customer opened"},
                        {"case": "T1", "id": "1", "time": "2024-05-01T01:00:00Z", "type": "refund", "text": "refund approved"}], f, case="T1", connector="file_drop")
    ts.put(v2)
    got = ts.get("file_drop:T1")
    assert len(got.signals) == 1 and got.signals[0].grades == [1]  # 0 signals: M4's labels were lost, or now pointed at the wrong event


def test_D17_sql_and_file_drop_ids_do_not_collide():
    f = FieldMap(family="toy/support_ticket")
    a = rows_to_trace([{"case": "T1", "time": "2024-05-01T00:00:00Z"}], f, case="T1", connector="file_drop").trace_id
    b = rows_to_trace([{"case": "T1", "time": "2024-05-01T00:00:00Z"}], f, case="T1", connector="sql_audit_log").trace_id
    assert a != b and a == "toy/support_ticket:T1"  # one overwrote the other; file_drop ids stay as they were


# ---------------------------------------------------------------- git history (local repos, no network)
def _git(cwd: Path, *args: str, date: str | None = None) -> str:
    env = {"GIT_AUTHOR_NAME": "A", "GIT_AUTHOR_EMAIL": "a@x", "GIT_COMMITTER_NAME": "A", "GIT_COMMITTER_EMAIL": "a@x", "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
           "HOME": str(cwd), **({"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date} if date else {})}
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout


def _repo(tmp_path: Path, *, branch: str = "main", after: list[tuple[str, dict[str, str], str]] = ()) -> tuple[Path, Path]:
    """A repo whose default branch has PR #12 merged, then ``after`` commits (message, files, date)."""
    w = tmp_path / "work"
    w.mkdir()
    _git(w, "init", "-q", "-b", branch)
    (w / "gpu").mkdir()
    (w / "gpu" / "k.cu").write_text("a\nb\nc\nd\ne\n")
    _git(w, "add", ".")
    _git(w, "commit", "-qm", "base", date="2026-01-01T00:00:00Z")
    _git(w, "checkout", "-qb", "feat")
    (w / "gpu" / "k.cu").write_text("a\nb\nC2\nD2\ne\n")
    (w / "gpu" / "my file.cu").write_text("x\n")
    (w / "gpu" / "café.cu").write_text("y\n")
    _git(w, "add", ".")
    _git(w, "commit", "-qm", "Tweak launch bounds", date="2026-01-02T00:00:00Z")
    _git(w, "checkout", "-q", branch)
    _git(w, "merge", "-q", "--no-ff", "feat", "-m", "Merge pull request #12 from alice/feat\n\nTweak launch bounds", date="2026-01-03T00:00:00Z")
    for msg, files, date in after:
        for name, content in files.items():
            if content is None:
                continue
            if name.startswith("mv:"):
                src, dst = name[3:].split(">")
                _git(w, "mv", src, dst)
            else:
                (w / name).parent.mkdir(parents=True, exist_ok=True)
                (w / name).write_text(content)
        _git(w, "add", "-A")
        _git(w, "commit", "-qm", msg, date=date)
    git_dir = tmp_path / "git"
    git_dir.mkdir()
    subprocess.run(["git", "clone", "-q", "--bare", str(w), str(git_dir / "repo.git")], check=True, capture_output=True)
    return w, git_dir


def _pr_trace(tmp_path: Path, git_dir: Path, **scope):
    from crucible_pack_coding.connectors.github_pr import GitHubPRConnector

    c = GitHubPRConnector(tmp_path / "cache", git_dir, offline=True, now="2026-03-01T00:00:00+00:00")
    [ref] = list(c.discover_git({"repo": "acme/repo", **scope}))
    return c, c.normalize(c.fetch(ref))


def test_D7_a_repo_on_main_is_read(tmp_path):
    _, gd = _repo(tmp_path)
    _, t = _pr_trace(tmp_path, gd)  # bad revision '<merge>..master'
    assert any(e.type == "merged" for e in t.events)


def test_D5_another_prs_revert_is_not_this_prs(tmp_path):
    _, gd = _repo(tmp_path, branch="master", after=[('Revert "Tweak launch bounds (#123)"', {"gpu/k.cu": "A\nb\nC2\nD2\ne\n"}, "2026-01-05T00:00:00Z")])
    _, t = _pr_trace(tmp_path, gd)
    assert not any(e.type == "post_merge_commit" and e.data["is_revert"] for e in t.events)  # "#12" matched "#123"


def test_D6_a_rename_is_not_a_fix_and_later_edits_follow_the_file(tmp_path):
    _, gd = _repo(tmp_path, branch="master", after=[("fix: move gpu/k.cu to gpu/kernels.cu", {"mv:gpu/k.cu>gpu/kernels.cu": ""}, "2026-01-05T00:00:00Z"),
                                   ("fix launch", {"gpu/kernels.cu": "a\nb\nC3\nD2\ne\n"}, "2026-01-06T00:00:00Z")])
    _, t = _pr_trace(tmp_path, gd)
    post = [e for e in t.events if e.type == "post_merge_commit"]
    assert [e.text.splitlines()[0] for e in post] == ["fix launch"]  # the rename touched "every PR line"; the edit after it was lost
    assert post[0].data["touched_lines"] == {"gpu/kernels.cu": [[3, 3]]} or post[0].data["touched_lines"] == {"gpu/kernels.cu": [(3, 3)]}


def test_D25_paths_with_spaces_and_accents(tmp_path):
    _, gd = _repo(tmp_path, branch="master")
    _, t = _pr_trace(tmp_path, gd)
    commit = next(e for e in t.events if e.type == "commit")
    assert set(commit.data["added"]) == {"gpu/k.cu", "gpu/my file.cu", "gpu/café.cu"}  # 'gpu/my file.cu\t'; café missing


def test_D8_reingest_is_idempotent_without_a_pinned_clock(tmp_path):
    from crucible_pack_coding.connectors.github_pr import GitHubPRConnector

    _, gd = _repo(tmp_path, branch="master")
    hashes = []
    for _ in range(2):
        c = GitHubPRConnector(tmp_path / "cache", gd, offline=True)
        [ref] = list(c.discover_git({"repo": "acme/repo"}))
        hashes.append(c.normalize(c.fetch(ref)).content_hash())
    assert hashes[0] == hashes[1]  # fetched_at = now() was in the hash


def test_D23_github_cache_pages_and_skips_incomplete_check_runs(tmp_path, monkeypatch):
    import httpx
    from crucible_pack_coding.github_reviews import GitHubCache

    pages = {1: [{"n": i} for i in range(100)], 2: [{"n": 100 + i} for i in range(50)]}

    def handler(req: httpx.Request) -> httpx.Response:
        if "check-runs" in str(req.url):
            return httpx.Response(200, json={"check_runs": [{"name": "ci", "completed_at": None}]})
        return httpx.Response(200, json=pages[int(req.url.params.get("page", 1))])

    gh = GitHubCache(tmp_path / "c")
    gh.http = httpx.Client(transport=httpx.MockTransport(handler))
    assert len(gh.get_all("https://api.github.com/repos/a/b/pulls/1/comments?per_page=100")) == 150  # page 1 only
    gh.get_json("https://api.github.com/repos/a/b/commits/x/check-runs?per_page=100")
    assert not gh._path("https://api.github.com/repos/a/b/commits/x/check-runs?per_page=100").exists()  # in progress: cached forever
    assert not list((tmp_path / "c").glob("*.tmp"))


def test_D11_the_audit_query_only_reads(tmp_path):
    import sqlite3

    from crucible.connectors.sql_audit_log import SqlAuditLogConnector

    db = tmp_path / "a.db"
    con = sqlite3.connect(db)
    con.execute("create table audit(case_id, time, actor)")
    con.execute("insert into audit values ('C1', '2024-01-01T00:00:00Z', 'al')")
    con.commit()
    con.close()
    c = SqlAuditLogConnector(FieldMap(case="case_id"))
    for q in ("UPDATE audit SET actor='x' RETURNING *", "SELECT * FROM audit; DELETE FROM audit"):
        with pytest.raises(ValueError, match="one SELECT"):
            list(c.discover({"dsn": str(db), "query": q}))
    assert [r.key for r in c.discover({"dsn": str(db), "query": "-- the log\nSELECT * FROM audit WHERE actor <> ';'"})] == ["C1"]
