"""M15 studio. The exit gate's toy part runs through the CLI, as a new user following the README would:
- ``crucible pack new demo`` creates a pack that passes ``crucible pack check`` at both levels (gold reaches the top
  reward, no-op does not).
- ``crucible label --pack toy`` turns the toy ticket logs into traces with dense rewards and TRL-format SFT / KTO data,
  one policy turn per row with per-turn Φ credit, the held-out test tickets left out.
- ``crucible build toy`` ingests, forges (no paid calls), accepts, exports canonical outputs and reports a benchmark.
- ``crucible label --run`` and ``crucible bench compare`` work on that benchmark run (label writes no rows from its
  scripted policies; ``test_m15_audit_label.py`` covers model runs).

Unit checks cover the conflict-safe scaffold, the closed filter grammar, the manifest contract and the dataset rows."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from crucible.cli.main import cli
from crucible.studio import datasets as ds


def run(*args):
    r = CliRunner().invoke(cli, list(args), catch_exceptions=False)
    return r


def test_scaffold_is_conflict_safe_and_manifest_contract(tmp_path):
    from crucible.core.schema.pack import PackManifest
    from crucible.studio.pack import ScaffoldConflict, new

    d = tmp_path / "demo"
    first = new("demo", d)
    assert first["created"] and not first["existing"]
    again = new("demo", d)  # identical files: a no-op
    assert not again["created"] and len(again["existing"]) == len(first["created"])
    (d / "README.md").write_text("edited by the user\n")
    with pytest.raises(ScaffoldConflict, match="nothing was written"):
        new("demo", d)
    assert (d / "README.md").read_text() == "edited by the user\n"
    with pytest.raises(ValueError, match="pack name"):
        new("Bad-Name", tmp_path / "x")
    base = {"name": "xy", "version": "1", "description": "d", "license": "MIT", "package": "p", "entry_point": "p:register"}  # names are 2+ chars (M14 V10)
    with pytest.raises(ValueError, match="ingest"):
        PackManifest.model_validate(base | {"hooks": {"credit": "p:credit"}})
    with pytest.raises(ValueError, match="module:attr"):
        PackManifest.model_validate(base | {"hooks": {"tasks": "nocolon"}})


def test_filter_grammar_and_rows():
    f = ds.compile_filter("0 < avg < 1 and n >= 2")
    assert f({"avg": 0.5, "n": 3, "best": 1, "worst": 0, "n_correct": 1, "solved": 1}) and not f({"avg": 1.0, "n": 3, "best": 1, "worst": 1, "n_correct": 3, "solved": 1})
    for bad in ("__import__('os')", "avg.real > 0", "x > 1", "'a' == 'a'", "[1][0] == 1"):
        with pytest.raises(ValueError):
            ds.compile_filter(bad)
    prompt = [{"role": "user", "content": "do it"}]
    good = ds.Labelled("t", [*prompt, {"role": "assistant", "content": "done"}], 1.0, weights=[None, 0.5], episode="g")
    bad = ds.Labelled("t", [*prompt, {"role": "assistant", "content": "nope"}], 0.0, weights=[None, -0.2], episode="b")
    sft = ds.sft_rows(ds.curate([good, bad], select="correct"))  # one prompt-completion row per policy turn (M15 G1 D11)
    assert len(sft) == 1 and sft[0]["prompt"] == prompt and sft[0]["completion"] == [good.messages[1]] and sft[0]["weight"] == 0.5
    assert sft[0]["resolved"] and sft[0]["traj_id"] == good.traj_id
    (chain,) = ds.sft_rows([good], fmt="chain")  # prime-rl's multi-turn row: every message's mask, numeric weights
    assert chain["message_loss_mask"] == [False, True] and chain["weights"] == [0.0, 0.5]
    assert ds.sft_rows([bad], mask_negative=True) == []  # its only assistant turn had negative credit
    (pair,) = ds.dpo_best_vs_worst([good, bad])
    assert pair["prompt"] == prompt and pair["chosen"] == [good.messages[1]] and pair["rejected"] == [bad.messages[1]]
    assert pair["chosen_episode"] == "g" and pair["rejected_episode"] == "b"
    other = ds.Labelled("t", [{"role": "user", "content": "something else"}, {"role": "assistant", "content": "x"}], 0.0)
    assert ds.dpo_best_vs_worst([good, other]) == []  # different prompts are not a preference pair
    labelled = ds.Labelled("t", [*prompt, {"role": "assistant", "content": "ok"}], 1.0, label="bad")  # outcome and reward disagree
    assert [r["label"] for r in ds.kto_rows([good, bad, labelled])] == [True, False]  # bools; the disagreeing trace is neither


def test_readme_path_through_the_cli(tmp_path):
    demo = tmp_path / "demo"
    r = run("pack", "new", "demo", "--dir", str(demo))
    assert r.exit_code == 0 and "next: crucible pack check" in r.output
    r = run("pack", "check", str(demo), "--level", "semantic", "--json")
    rep = json.loads(r.output)
    assert r.exit_code == 0 and rep["ok"] and {x["id"] for x in rep["results"] if x["status"] == "pass"} >= {"semantic.gold", "semantic.noop", "static.hooks"}
    (demo / "pack.yaml").write_text((demo / "pack.yaml").read_text().replace("license: Apache-2.0", "license: unknown"))
    r = run("pack", "check", str(demo))
    assert r.exit_code == 1 and "static.license" in r.output
    r = run("bench", "run", str(tmp_path / "nowhere"))
    assert r.exit_code == 4 and r.output.startswith("Error: ValueError: no pack")  # a user error is one line, no traceback (exit 4: M15 H2 V16)

    r = run("label", "--pack", "toy", "--out", str(tmp_path / "label"), "--json")
    s = json.loads(r.output)
    assert r.exit_code == 0 and s["traces"] == 24 and s["credited"] == 24 and s["outcomes"]["good"] > s["outcomes"]["bad"] > 0
    assert s["datasets"]["sft"] > 0 and s["datasets"]["kto"] > 0 and sum(s["signals"].values()) > 0
    sft = [json.loads(x) for x in (tmp_path / "label" / "sft.jsonl").read_text().splitlines()]
    for row in sft:  # every row: one prompt, one assistant turn as the target, numeric credit (not just line 0: M15 H1 V14)
        assert set(row) >= {"prompt", "completion", "tools", "weight", "credit", "task", "reward", "resolved", "traj_id", "split", "outcome"}
        assert row["prompt"][-1]["role"] in ("user", "tool") and [m["role"] for m in row["completion"]] == ["assistant"]
        assert row["split"] == "train" and row["outcome"] == "good" and isinstance(row["weight"], float)
    # the toy's Φ does not predict held-out outcomes, so neither of its uses is released (M5 R1, credit.channels): no dense
    # credit on the turns, and the summary says why
    ch = s["credit_channels"]["toy/support_ticket"]
    assert not ch["step"]["released"] and not ch["terminal"]["released"] and ch["step"]["reason"] and ch["terminal"]["reason"]
    assert all(r["credit"] is False and r["weight"] == 0.0 for r in sft)
    kto = [json.loads(x) for x in (tmp_path / "label" / "kto.jsonl").read_text().splitlines()]
    assert kto and all(isinstance(r["label"], bool) for r in kto) and s["held_out"]
    assert (tmp_path / "label" / "configs" / "resolved" / "label.json").exists() and (tmp_path / "label" / "rewards.jsonl").exists()

    r = run("build", "toy", "--out", str(tmp_path / "build"), "--fresh", "2", "--json")
    b = json.loads(r.output)
    a = b["accept"]
    # the full contract (M9 audit V2): without --docker/--paid the checks that need them are pending, so a sound task is
    # provisional, never "accepted"; nothing here may be rejected, and what is not rejected is exported (labelled)
    assert r.exit_code == 0 and b["forge"]["tasks"] == 5 and a["rejected"] == 0 and a["accepted"] + a["provisional"] == 5 and b["ok"]
    assert b["export"]["canonical"] == 5
    assert set(b["export"]["oracle_rewards"].values()) == {1.0} and b["bench"]["items"] == {"anchor": 5, "fresh": 2}
    # only the bench split is benchmarked (M15 audit H2 V8); the train split is the RL training set
    n_bench = b["bench"]["splits"]["counts"]["bench"]
    assert sum(b["bench"]["benched"].values()) == n_bench and n_bench + b["bench"]["splits"]["counts"]["train"] == 7 and b["bench"]["splits"]["verified"]
    assert b["bench"]["models"]["scripted:oracle"] == 1.0 and b["bench"]["models"]["scripted:noop"] == 0.0 and "contamination" in b
    run_dir = tmp_path / "build" / "runs" / "bench-toy-build"
    r = run("label", "--run", str(run_dir), "--out", str(tmp_path / "label_run"), "--json")
    lr = json.loads(r.output)
    # the build's benchmark is scripted oracle / half / no-op: no model-produced episode, so no training row (M15 H1 V1)
    assert r.exit_code == 0 and lr["datasets"]["sft"] == lr["datasets"]["dpo"] == lr["datasets"]["kto"] == 0
    assert lr["skipped"].get("held_out") and set(lr["policies"].values()) <= {"scripted", "control", "adversary"}, lr["skipped"]
    r = run("bench", "compare", str(run_dir), str(run_dir), "--base-policy", "scripted:oracle", "--cand-policy", "scripted:noop", "--json")
    c = json.loads(r.output)
    # every valid bench task flips (a contaminated or invalid one is left out of the comparison: M13 validity)
    f = c["flips"]
    assert 1 <= f["pass_to_fail"] <= n_bench and f["fail_to_pass"] == f["both_pass"] == 0 and (f["pass_to_fail"] < 5 or c["mcnemar_p"] < 0.05)
