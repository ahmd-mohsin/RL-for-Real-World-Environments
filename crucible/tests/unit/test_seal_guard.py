"""The seal guard (factory.sealguard) and the contamination audit (factory.benchaudit), on throwaway git repositories."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from crucible.factory import benchaudit as ba
from crucible.factory import sealguard as sg

HELD = {"zzlib": "D", "qqcalc": "D"}
URLS = {"zzlib": "acme/zzlib", "qqcalc": "acme-calc/qqcalc", "echo": "labstack/echo"}
FIX = "    if err := writer.Flush(); err != nil { return fmt.Errorf(\"flush stream: %w\", err) }"
TEST = "streamHandler + relay (with Failure and its own code)"


def git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "tester")
    (r / "README.md").write_text("hello\n")
    git(r, "add", ".")
    git(r, "commit", "-qm", "init")
    return r


def held_task() -> ba.HeldTask:
    return ba.HeldTask("db-node-zzlib-a6cd4654", "zzlib", "debug", "When an adapted handler pipes a readable stream into the response and the stream "
                       "errors after some data was sent, the error hook is never called.", [f"test/app.test.ts::app::node::{TEST}"], [FIX], [])


def seal_for(repo: Path) -> sg.Seal:
    sg.refresh(repo, [held_task()], key=bytes.fromhex("ab" * 32), held_out=HELD, urls=URLS)
    return sg.Seal.load(repo, held_out=HELD, urls=URLS)


def stage(repo: Path, path: str, text: str, *, force: bool = False) -> None:
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    git(repo, "add", *(["-f"] if force else []), path)


@pytest.mark.parametrize("path, text, rule", [
    ("crucible/data/sealed/heldout_draw.json", "{}", "sealed_path"),
    ("crucible/data/debug/gen/store/heldout/tasks/x/window.json", "{}", "sealed_path"),
    ("out/bench.key", "k", "sealed_path"),
    ("crucible/data/debug/prompts/db-node-zzlib-a6cd4654.json", "{}", "heldout_id"),
    ("notes.md", "the next swarm round mines acme/zzlib again\n", "heldout_repo"),
    ("notes.md", f"a line from a fix: {FIX.strip()}\n", "hidden_content"),
    ("notes.md", "  it('" + TEST + "', async () => {})\n", "hidden_content"),
    ("cfg.txt", "key = " + "ab" * 32 + "\n", "bench_key"),
])
def test_every_rule_refuses_its_content(repo: Path, path: str, text: str, rule: str) -> None:
    seal = seal_for(repo)
    stage(repo, path, text, force=True)  # git add -f: what .gitignore cannot stop
    found = sg.staged(repo, seal)
    assert rule in {f.rule for f in found}, found
    ok, msg = sg.decide(repo, found, reason=None)
    assert not ok and "refused" in msg


def test_a_common_idiom_inside_a_hidden_line_is_not_a_copy(repo: Path) -> None:
    seal = seal_for(repo)
    stage(repo, "src/train.go", '\t\treturn fmt.Errorf("flush stream: %w", err)\n')  # part of FIX, under 60% of it
    assert sg.staged(repo, seal) == []


def test_clean_training_content_passes(repo: Path) -> None:
    seal = seal_for(repo)
    stage(repo, "crucible/data/debug/prompts/db-go-echo-61eca05b-e2.json", json.dumps({"opening": "a training report about labstack/echo"}))
    assert sg.staged(repo, seal) == []


def test_the_registry_may_name_held_out_repositories_but_never_hold_their_content(repo: Path) -> None:
    seal = seal_for(repo)
    stage(repo, sg.REGISTRY[0], "HELD_OUT_REPOS.update({'zzlib': 'D'})\nREPO_URLS = {'zzlib': 'acme/zzlib'}\n")
    assert sg.staged(repo, seal) == []
    stage(repo, sg.REGISTRY[0], f"# {FIX.strip()}\n")
    assert {f.rule for f in sg.staged(repo, seal)} == {"hidden_content"}


def test_an_override_needs_a_reason_and_is_journaled_without_the_text(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seal = seal_for(repo)
    stage(repo, "notes.md", f"{FIX.strip()}\n")
    found = sg.staged(repo, seal)
    with pytest.raises(ValueError):
        sg.journal(repo, found, "  ")
    monkeypatch.setenv(sg.OVERRIDE_ENV, "the lead publishes the retired benchmark")
    ok, msg = sg.decide(repo, found)
    assert ok and "override" in msg
    rows = [json.loads(x) for x in (repo / sg.JOURNAL).read_text().splitlines()]
    assert rows[-1]["reason"] == "the lead publishes the retired benchmark" and rows[-1]["who"] == "tester"
    assert "flush stream" not in (repo / sg.JOURNAL).read_text()  # findings name the rule and line, never the text


def test_fingerprints_hold_hashes_never_text(repo: Path) -> None:
    seal_for(repo)
    raw = (repo / sg.FINGERPRINTS).read_text()
    assert "flush stream" not in raw and "piping" not in raw and "ab" * 32 not in raw
    assert sg.check_path(sg.FINGERPRINTS)  # and the file itself is a sealed path


def test_a_commit_range_and_the_tree_ratchet(repo: Path) -> None:
    seal = seal_for(repo)
    stage(repo, "results.md", "db-node-zzlib-a6cd4654 kept\n")
    git(repo, "commit", "-qm", "results")
    rng = sg.commit_range(repo, "HEAD~1..HEAD", seal)
    assert [f.rule for f in rng] == ["heldout_id"]
    whole = sg.tree(repo, "HEAD", seal)
    base = [{"rule": f.rule, "path": f.path} for f in whole]
    assert sg.ratchet(whole, base) == []  # yesterday's exposure is the audit's, not a new refusal
    stage(repo, "more.md", "acme/zzlib again\n")
    git(repo, "commit", "-qm", "more")
    assert [f.path for f in sg.ratchet(sg.tree(repo, "HEAD", seal), base)] == ["more.md"]


def test_a_range_reads_only_what_the_branch_adds(repo: Path) -> None:
    """A branch behind main does not carry main's later redaction of a held-out id as its own addition."""
    seal = seal_for(repo)
    stage(repo, "notes.md", "db-node-zzlib-a6cd4654 kept\n", force=True)
    git(repo, "commit", "-qm", "exposed", "--no-verify")
    git(repo, "branch", "side")
    stage(repo, "notes.md", "held-out task 1 kept\n")
    git(repo, "commit", "-qm", "redact")
    git(repo, "checkout", "-q", "side")
    stage(repo, "other.md", "training notes\n")
    git(repo, "commit", "-qm", "side work")
    assert sg.commit_range(repo, "main..side", seal) == []
    stage(repo, "leak.md", "db-node-zzlib-a6cd4654 again\n")
    git(repo, "commit", "-qm", "leak")
    assert [f.path for f in sg.commit_range(repo, "main..side", seal)] == ["leak.md"]


def test_the_hook_script_exists_and_calls_the_guard() -> None:
    hook = Path(__file__).resolve().parents[3] / ".githooks" / "pre-commit"
    assert hook.is_file() and "seal_guard.py\" check --staged" in hook.read_text()


def test_local_draw_holds_its_repositories_out_and_refuses_a_side_change(tmp_path: Path) -> None:
    from crucible.factory.lanes import windows

    p = tmp_path / "heldout_draw.json"
    p.write_text(json.dumps({"repos": {"newlib": "someone/newlib"}}))
    assert windows.local_draw(p) == {"newlib": "someone/newlib"}
    p.write_text(json.dumps({"repos": {"echo": "labstack/echo"}}))
    with pytest.raises(ValueError, match="training sources"):
        windows.local_draw(p)
    p.write_text(json.dumps({"repos": ["x"]}))
    with pytest.raises(ValueError):
        windows.local_draw(p)
    assert windows.local_draw(tmp_path / "absent.json") == {}


# ---- the audit ---------------------------------------------------------------------------------------------------------
def test_audit_finds_public_grading_content_and_keeps_a_clean_task_sealed(repo: Path) -> None:
    t = held_task()
    clean = ba.HeldTask("db-go-qqcalc-583585d2", "qqcalc", "debug", "Formula results are wrong for nested arrays inside a sum of products.",
                        ["TestCalcSumProductNestedArrays"], ["    return newNumberFormulaArg(sum * weight)  // nested arrays"], [])
    (repo / "data").mkdir()
    (repo / "data" / "authored.json").write_text(json.dumps({"id": t.id, "prompt": t.prompt, "tests": [TEST]}) + "\n" + FIX + "\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "publish")
    rep = ba.audit([t, clean], ba.git_grep(repo, "HEAD"), [], repo=repo)
    v = {r["id"]: r for r in rep["tasks"]}
    assert v[t.id]["verdict"] == "contaminated" and v[t.id]["exposure"]["hidden_test"]["matched"] == 1
    assert v[clean.id]["verdict"] == "sealed"
    assert next(iter(v[t.id]["first_public"].values())).split()[0] == git(repo, "rev-parse", "--short", "HEAD").strip()
    assert FIX.strip() not in json.dumps(rep)  # the report never re-publishes a probe


def test_audit_counts_training_overlap_and_skips_template_and_a_moved_repository() -> None:
    t = held_task()
    harness = "How this works: you are working in a sandboxed copy of the repository with no network access at all."
    t.prompt += "\n" + harness
    corpus = [(f"train{i}:prompt", harness) for i in range(3)] + [("db-go-x-1:diff", FIX + "\n" + FIX.replace("flush", "drain"))]
    rep = ba.audit([t], lambda pats: {}, corpus, in_draw=lambda _t: True)
    row = rep["tasks"][0]
    assert row["verdict"] == "contaminated" and any(h["kind"] == "code-substring" for h in row["train_overlap"])
    assert harness not in ba.probes(t, ba.template_sentences([t.prompt] + [x for _, x in corpus[:3]]))["prompt"]
    moved = ba.audit([t], lambda pats: {}, [], in_draw=lambda _t: False)
    assert moved["tasks"][0]["verdict"] == "not_heldout"


def test_weak_template_like_13_grams_do_not_contaminate() -> None:
    hits = [{"kind": "13-gram", "overlap": 9, "source": f"db-go-x-{i}:prompt"} for i in range(4)] + [{"kind": "13-gram", "overlap": 2, "source": "x"}]
    strong, weak = ba.split_hits(hits)
    assert strong == [] and len(weak) == 5
    assert ba.split_hits([{"kind": "13-gram", "overlap": 6, "source": "a"}])[0]


def test_this_repository_exposes_nothing_beyond_the_frozen_baseline() -> None:
    """The ratchet: every tracked path and held-out name in HEAD is in seal_baseline.json (exposures public before the
    guard, measured by the audit). A new one fails here until the lead publishes it on purpose and re-freezes."""
    root = Path(__file__).resolve().parents[3]
    base = json.loads((root / "crucible/docs/research/swarm-pipeline/runs/seal_baseline.json").read_text())["findings"]
    new = sg.ratchet(sg.tree(root, "HEAD", sg.Seal.load(root)), base)
    assert new == [], new[:10]
