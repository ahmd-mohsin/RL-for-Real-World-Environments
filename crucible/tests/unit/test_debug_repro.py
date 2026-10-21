"""Gate S7r (``generators.debug_repro``): the per-requirement ``repro`` field of a debug prompt, its leak lint, its block
in the prompt, and its execution at the start tree A' (must show the bug) and the oracle B (must pass), with fake Worlds
and, marked ``docker``, in a real World container."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m17_debug_lane import _gen, make_repo

from crucible.factory.generators.base import Candidate, git, out
from crucible.factory.generators.debug import author_debug
from crucible.factory.generators.debug_repro import Repro, hidden_lines, judge, lint_repro, parse_repros, parse_shell, render, s7r
from crucible.factory.lanes.windows_git import RepoGit


def test_parse_repros_reads_the_field_and_reports_format_problems() -> None:
    authored = {"repro": {"m01": {"command": "python -c 'print(1)'", "expect": {"exit": 0, "contains": "1"}, "timeout_s": 30},
                          "m02": "echo hi", "m09": {"command": "x"}, "m03": {"expect": {}}, "m04": {"command": "y", "expect": []}}}
    rs, problems = parse_repros(authored, ["m01", "m02", "m03", "m04"])
    assert rs["m01"] == Repro("m01", "python -c 'print(1)'", 0, ("1",), None, 30) and rs["m02"].command == "echo hi" and rs["m02"].exit == 0
    assert problems == ["repro unknown_requirement_id: m09", "repro m03: no command", "repro m04: expect must be an object {exit, contains, equals}"]
    assert parse_repros({}, ["m01"]) == ({}, [])  # no field: no repro, never a problem
    assert parse_repros({"repro": ["x"]}, ["m01"])[1] == ["repro: must map requirement ids to {command, expect}"]


def test_shell_header_and_judge() -> None:
    assert parse_shell("[exit 3]\nboom") == (3, False, "boom") and parse_shell("[TIMEOUT]\n") == (None, True, "") and parse_shell("odd")[0] is None
    r = Repro("m01", "c", 0, ("-3",), None)
    assert judge(r, 0, False, "value -3\n") == (True, "as expected")
    assert judge(r, 1, False, "-3")[1] == "exit 1, expected 0" and judge(r, 0, False, "3")[1].startswith("output lacks")
    assert not judge(r, 0, True, "-3")[0]
    eq = Repro("m01", "c", 0, (), "a  b\n")
    assert judge(eq, 0, False, "a b")[0] and not judge(eq, 0, False, "a b c")[0]


def test_lint_repro_keeps_the_leak_rules_but_allows_scratch_paths() -> None:
    hidden_src = {"calc_test.go": 'func TestParseNegative(t *testing.T) {\n\tif Parse("-3") != -3 { t.Fatal("negative numbers must parse") }\n}\n'}
    rec = {"lang": "go", "tests_b": hidden_src, "milestones": [{"id": "m01", "tests": ["calc_test.go::TestParseNegative"]}]}
    lines = hidden_lines(rec)
    ok = Repro("m01", "mkdir -p /tmp/r && cat > /tmp/r/main.go <<'EOF'\npackage main\nEOF\ngo run /tmp/r/main.go", 0, ("-3",))
    assert lint_repro(ok, faulty={"parseSigned"}, hidden={"TestParseNegative"}, paths={"calc_test.go"}, lines=lines) == []
    bad = Repro("m01", 'go test -run TestParseNegative ./... ; cat calc_test.go ; parseSigned ; if Parse("-3") != -3 { t.Fatal("negative numbers must parse") }', 0)
    found = " ".join(lint_repro(bad, faulty={"parseSigned"}, hidden={"TestParseNegative"}, paths={"calc_test.go"}, lines=lines))
    for rule in ("test_name", "hidden_test", "faulty_name", "file_path", "repro_hidden_test_file", "repro_copies_hidden_test"):
        assert rule in found, rule
    assert "lint version" in " ".join(lint_repro(Repro("m01", "echo 1.2.3", 0), faulty=set(), hidden=set()))


def _window(tmp_path: Path):
    repo, shas = make_repo(tmp_path)
    gen = _gen(tmp_path, repo)
    b = out(git(repo, "rev-parse", "HEAD"))
    win, _ = gen.synthesize(Candidate("db-go-calc-r", "debug", "calc", params={"lang": "go", "b": b, "b_date": "2026-05-13T00:00:00+00:00",
                                                                                "fixes": [shas["fix_neg"], shas["fix_fmt"]]}))
    rec = {**win.to_json(), "lang": "go", "p2p": ["calc_test.go::TestParse"], "collateral": [],
           "milestones": [{"id": "m01", "unit": win.units[0].sha, "tests": ["calc_test.go::TestParseNegative"]},
                          {"id": "m02", "unit": win.units[1].sha, "tests": ["calc_test.go::TestFmt"], "requires": ["m01"]}]}
    return rec, RepoGit(gen.clones["calc"])


GOOD = {"builder": "claude-code:opus-5.5", "opening": "Two things in the calculator package are off.",
        "requirements": {"m01": "Reading \"-3\" gives a large positive number instead of minus three.",
                         "m02": "Formatting a number always produces an empty string."}}
REPRO = {"m01": {"command": "mkdir -p /tmp/r1 && cat > /tmp/r1/main_test.go <<'EOF'\npackage calc\nEOF\necho parse", "expect": {"exit": 0, "contains": ["-3"]}}}


def test_author_debug_renders_and_lints_the_repro_and_leaves_old_prompts_unchanged(tmp_path: Path) -> None:
    rec, repo = _window(tmp_path)
    plain = author_debug(rec, GOOD, repo)
    with_r = author_debug(rec, {**GOOD, "repro": REPRO}, repo)
    assert plain["findings"] == [] and plain["repro"] == {} and plain["repro_status"] == "unverified"
    assert with_r["findings"] == [] and with_r["repro_status"] == "unexecuted" and with_r["repro"]["m01"]["command"].startswith("mkdir")
    assert "How to reproduce" in with_r["prompt"] and "- [R1] \nmkdir -p /tmp/r1" in with_r["prompt"] and "How to reproduce" not in plain["prompt"]
    assert with_r["prompt"].index("How to reproduce") > with_r["prompt"].index("What I am seeing:")
    assert set(with_r["spans"]) == set(plain["spans"])  # every requirement still located
    # an old prompt (no repro field) is byte-identical to what the lane wrote before S7r existed
    assert plain["provenance"]["sha256"] == author_debug(rec, dict(GOOD), repo)["provenance"]["sha256"]
    leak = author_debug(rec, {**GOOD, "repro": {"m02": {"command": "go test -run TestFmt ./...", "expect": {"exit": 0}}}}, repo)
    assert any("repro m02" in f and "test_name" in f for f in leak["findings"])


class FakeWorld:
    """A World whose shell runs a tiny "program": ``calc`` prints Parse("-3") from the workspace's calc.go (buggy until
    the oracle's fix is written)."""

    def __init__(self, files: dict[str, str]):
        self.files, self.closed, self.calls = dict(files), False, []

    async def step(self, actor, call):
        self.calls.append(call.function_name)
        a = call.arguments
        if call.function_name == "repo.write_file":
            self.files[a["path"]] = a["content"]
            res = SimpleNamespace(ok=True, content="written")
        elif a["command"].startswith("rm -f"):
            res = SimpleNamespace(ok=True, content="[exit 0]\n")
        elif a["command"] == "calc":
            val = "3" if "BUG" in self.files.get("calc.go", "") else "-3"
            res = SimpleNamespace(ok=True, content=f"[exit 0]\n{val}\n")
        elif a["command"] == "crash":
            res = SimpleNamespace(ok=False, content="tool error")
        else:
            res = SimpleNamespace(ok=True, content="[exit 127]\nnot found\n")
        return SimpleNamespace(observation=SimpleNamespace(tool_results=[res]))

    async def close(self):
        self.closed = True


class FakeTask:
    def __init__(self, *, fail_open: bool = False):
        self.task_id = "db-fake"
        self.window = {"milestones": [{"id": "m01"}, {"id": "m02"}, {"id": "m03"}], "oracle_files": {"calc.go": "package calc // fixed\n", "old.go": None}}
        self.worlds: list[FakeWorld] = []
        self.fail_open = fail_open

    async def open(self, workdir: Path):
        if self.fail_open:
            raise RuntimeError("no image")
        w = FakeWorld({"calc.go": "package calc // BUG\n", "old.go": "x"})
        self.worlds.append(w)
        return w


def test_s7r_verifies_shows_the_bug_at_the_start_and_passes_at_the_oracle() -> None:
    t = FakeTask()
    res = asyncio.run(s7r(t, {"m01": Repro("m01", "calc", 0, ("-3",))}))
    assert res["status"] == "partial" and res["ok"] and res["verified"] == ["m01"] and res["unverified"] == ["m02", "m03"]
    row = res["per_requirement"]["m01"]
    assert row["start"]["passes"] is False and row["oracle"]["passes"] is True and "lacks" in row["start"]["why"]
    assert len(t.worlds) == 2 and all(w.closed for w in t.worlds) and t.worlds[1].files["calc.go"].endswith("fixed\n")
    assert "repo.write_file" in t.worlds[1].calls and "repo.write_file" not in t.worlds[0].calls
    full = asyncio.run(s7r(t, {m: Repro(m, "calc", 0, ("-3",)) for m in ("m01", "m02", "m03")}))
    assert full["status"] == "verified"


def test_s7r_fails_a_present_but_wrong_repro_and_never_a_missing_one() -> None:
    t = FakeTask()
    no_bug = asyncio.run(s7r(t, {"m01": Repro("m01", "calc", 0, ("3",))}))  # "3" is in both "3" and "-3": the start tree passes
    assert no_bug["status"] == "failed" and not no_bug["ok"] and no_bug["per_requirement"]["m01"]["status"] == "no_bug_at_start"
    wrong = asyncio.run(s7r(t, {"m01": Repro("m01", "calc", 0, ("-4",))}))  # the oracle does not show it either
    assert wrong["status"] == "failed" and wrong["per_requirement"]["m01"]["status"] == "fails_at_oracle"
    opened = len(t.worlds)
    none = asyncio.run(s7r(t, {}))
    assert none["status"] == "unverified" and none["ok"] and none["unverified"] == ["m01", "m02", "m03"] and len(t.worlds) == opened  # no World opened


def test_s7r_infrastructure_is_unmeasured_not_a_failure() -> None:
    res = asyncio.run(s7r(FakeTask(fail_open=True), {"m01": Repro("m01", "calc", 0, ("-3",))}))
    assert res["status"] == "unmeasured" and res["ok"] and "did not open" in res["per_requirement"]["m01"]["why"]
    res2 = asyncio.run(s7r(FakeTask(), {"m01": Repro("m01", "crash", 0)}))
    assert res2["status"] == "unmeasured" and "shell tool failed" in res2["per_requirement"]["m01"]["why"]


def test_render_names_the_prompts_requirement_ids() -> None:
    txt = render({"m02": Repro("m02", "calc", 0, ("-3",)), "m01": Repro("m01", "a\nb", 1, (), "x")}, {"m01": "R1", "m02": "R2"})
    assert txt.index("[R1]") < txt.index("[R2]") and "exit code 1" in txt and "the output contains '-3'" in txt and "- [R1] \na\nb" in txt
    assert render({}, {}) == ""


# ---- a real World container -----------------------------------------------------------------------------------------------------
IMAGE = "crucible-lh-w:parse-6c176c2fa980"  # a Python World image the window lane built (scripts/tinker_rl.py DEFAULT_IMAGE)


def _have_image() -> bool:
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False)
    return r.returncode == 0


class DockerTask:
    """A tiny Python task in a real ``--network none`` World: ``neg.py`` mis-parses negatives until the oracle's fix."""

    def __init__(self, base: Path):
        self.task_id, self.base = "db-docker-repro", base
        self.window = {"milestones": [{"id": "m01"}, {"id": "m02"}],
                       "oracle_files": {"neg.py": "def parse(s):\n    return int(s)\n"}}

    async def open(self, workdir: Path):
        from crucible.factory.lanes.windows_world import open_world

        return await open_world(image=IMAGE, base_dir=self.base, workdir=workdir, instruction="s7r test", name_prefix="crucible-lh-s7r-test-")


@pytest.mark.docker
@pytest.mark.skipif(not _have_image(), reason=f"needs Docker and the {IMAGE} image")
def test_s7r_in_a_real_world(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "neg.py").write_text("def parse(s):\n    return abs(int(s))\n")
    t = DockerTask(base)
    cmd = "python -c 'import neg; print(neg.parse(\"-3\"))'"
    res = asyncio.run(s7r(t, {"m01": Repro("m01", cmd, 0, ("-3",)), "m02": Repro("m02", "python -c 'import neg; print(neg.parse(\"5\"))'", 0, ("6",))},
                          workdir=tmp_path / "w"))
    assert res["per_requirement"]["m01"]["status"] == "verified", res["per_requirement"]["m01"]
    assert res["per_requirement"]["m02"]["status"] == "fails_at_oracle" and res["status"] == "failed"
    left = subprocess.run(["docker", "ps", "-a", "--filter", "name=crucible-lh-s7r-test-", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
    assert left.stdout.strip() == ""  # closing each World removed its container
