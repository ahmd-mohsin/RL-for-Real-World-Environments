"""M15 studio. The exit gate's toy part runs through the CLI, as a new user following the README would:
- ``crucible pack new demo`` creates a pack that passes ``crucible pack check`` at both levels (gold reaches the top
  reward, no-op does not).
- ``crucible label --pack toy`` turns the toy ticket logs into traces with dense rewards and TRL-format SFT / KTO data
  with loss masks and per-turn Φ credit.
- ``crucible build toy`` ingests, forges (no paid calls), accepts, exports canonical outputs and reports a benchmark.
- ``crucible label --run`` and ``crucible bench compare`` work on that benchmark run.

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
    base = {"name": "x", "version": "1", "description": "d", "license": "MIT", "package": "p", "entry_point": "p:register"}
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
    good = ds.Labelled("t", [*prompt, {"role": "assistant", "content": "done"}], 1.0, weights=[None, 0.5])
    bad = ds.Labelled("t", [*prompt, {"role": "assistant", "content": "nope"}], 0.0, weights=[None, -0.2])
    sft = ds.sft_rows(ds.curate([good, bad], select="correct"))
    assert len(sft) == 1 and sft[0]["message_loss_mask"] == [False, True] and sft[0]["resolved"] and sft[0]["traj_id"] == good.traj_id
    assert ds.sft_rows([bad], mask_negative=True) == []  # its only assistant turn had negative credit
    (pair,) = ds.dpo_best_vs_worst([good, bad])
    assert pair["prompt"] == prompt and pair["chosen"][0]["content"] == "done" and pair["rejected"][0]["content"] == "nope"
    other = ds.Labelled("t", [{"role": "user", "content": "something else"}, {"role": "assistant", "content": "x"}], 0.0)
    assert ds.dpo_best_vs_worst([good, other]) == []  # different prompts are not a preference pair
    assert [r["label"] for r in ds.kto_rows([good, bad])] == [True, False]


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
    assert r.exit_code != 0 and r.output.startswith("Error: ValueError: no pack")  # a user error is one line, no traceback

    r = run("label", "--pack", "toy", "--out", str(tmp_path / "label"), "--json")
    s = json.loads(r.output)
    assert r.exit_code == 0 and s["traces"] == 24 and s["credited"] == 24 and s["outcomes"]["good"] > s["outcomes"]["bad"] > 0
    assert s["datasets"]["sft"] > 0 and s["datasets"]["kto"] > 0 and sum(s["signals"].values()) > 0
    row = json.loads((tmp_path / "label" / "sft.jsonl").read_text().splitlines()[0])
    assert set(row) >= {"messages", "tools", "message_loss_mask", "weights", "task", "reward", "resolved", "traj_id"}
    assert any(row["message_loss_mask"]) and not any(m for m, msg in zip(row["message_loss_mask"], row["messages"]) if msg["role"] != "assistant")
    assert any(w is not None for w in row["weights"])  # dense Φ credit on the assistant turns
    assert (tmp_path / "label" / "configs" / "resolved" / "label.json").exists() and (tmp_path / "label" / "rewards.jsonl").exists()

    r = run("build", "toy", "--out", str(tmp_path / "build"), "--fresh", "2", "--json")
    b = json.loads(r.output)
    a = b["accept"]
    # the full contract (M9 audit V2): without --docker/--paid the checks that need them are pending, so a sound task is
    # provisional, never "accepted"; nothing here may be rejected, and what is not rejected is exported (labelled)
    assert r.exit_code == 0 and b["forge"]["tasks"] == 5 and a["rejected"] == 0 and a["accepted"] + a["provisional"] == 5
    assert b["export"]["canonical"] == 5
    assert set(b["export"]["oracle_rewards"].values()) == {1.0} and b["bench"]["items"] == {"anchor": 5, "fresh": 2}
    assert b["bench"]["models"]["scripted:oracle"] == 1.0 and b["bench"]["models"]["scripted:noop"] == 0.0
    run_dir = tmp_path / "build" / "runs" / "bench-toy-build"
    r = run("label", "--run", str(run_dir), "--out", str(tmp_path / "label_run"), "--json")
    lr = json.loads(r.output)
    assert lr["datasets"]["dpo_kinds"]["best_vs_worst"] == 7
    pair = json.loads((tmp_path / "label_run" / "dpo.jsonl").read_text().splitlines()[0])
    names = {c["function"]["name"] for m in pair["chosen"] if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    assert names and all("." not in n for n in names)  # OpenAI-legal tool names
    r = run("bench", "compare", str(run_dir), str(run_dir), "--base-policy", "scripted:oracle", "--cand-policy", "scripted:noop", "--json")
    c = json.loads(r.output)
    assert c["flips"]["pass_to_fail"] == 7 and c["mcnemar_p"] < 0.05
