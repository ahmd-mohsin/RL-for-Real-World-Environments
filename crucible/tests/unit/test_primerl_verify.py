"""The real student's GPU verification on DeltaAI (``delta_rl.py render-verify``, ``train --debug``; swarm-pipeline/14 §8):
training on the debug lane's own tasks (training split only), the job's Worlds through the Apptainer shim, and the job's
phases. No GPU and no cluster: the rendered files are checked."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

from crucible.train.bridges.primerl import deploy

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3.6-35B-A3B"
TASK = "db-go-go-toml-a02114e3"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_job_reads_the_hf_cache_it_is_given() -> None:
    dr = _load("delta_rl")
    d = deploy.DeltaConfig()
    assert d.hf_cache == f"{d.workdir}/hf_cache" and f"HF_HOME={d.workdir}/hf_cache " in dr.uv_env(d)
    d2 = deploy.DeltaConfig(hf_home="/work/hdd/x/hf")
    assert d2.hf_cache == "/work/hdd/x/hf" and "HF_HOME=/work/hdd/x/hf " in dr.setup_lines(d2, version=deploy.PRIME_RL_VERSION)


def test_debug_tasks_refuse_held_out_and_unknown_ids(tmp_path: Path) -> None:
    dr = _load("delta_rl")
    rows = [{"id": "db-go-a-1", "split": "train", "repo": "a", "lang": "go", "gates_ok": True},
            {"id": "db-go-h-2", "split": "heldout", "repo": "h", "lang": "go", "gates_ok": True}]
    (tmp_path / "candidates.json").write_text(json.dumps(rows))
    (tmp_path / "authored").mkdir()
    for r in rows:
        (tmp_path / "authored" / f"{r['id']}.json").write_text("{}")
    with pytest.raises(SystemExit, match="held out"):
        dr.debug_tasks(tmp_path, ["db-go-h-2"])
    with pytest.raises(SystemExit, match="not a gated debug-lane task"):
        dr.debug_tasks(tmp_path, ["db-go-nope-3"])


def _targeting(split: str = "train", in_band: bool = True) -> dict:
    return {TASK: {"model": MODEL, "split": split, "band": {"in_band": in_band, "n": 8, "rewards": [0.0, 0.25]}},
            "db-go-other-9": {"model": MODEL, "split": "train", "band": {"in_band": True}}}


def test_verify_targeting_keeps_the_named_training_tasks_only(tmp_path: Path) -> None:
    dr = _load("delta_rl")
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_targeting()))
    assert list(dr.verify_targeting(p, [TASK], MODEL)) == [TASK]
    p.write_text(json.dumps(_targeting(split="heldout")))
    with pytest.raises(SystemExit, match="held out"):
        dr.verify_targeting(p, [TASK], MODEL)
    p.write_text(json.dumps(_targeting(in_band=False)))
    with pytest.raises(SystemExit, match="refused"):
        dr.verify_targeting(p, [TASK], MODEL)


def _render(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    dr = _load("delta_rl")
    t = tmp_path / "t.json"
    t.write_text(json.dumps(_targeting()))
    stage = tmp_path / "stage"
    argv = ["delta_rl.py", "render-verify", "--stage", str(stage), "--run-name", "v", "--tasks", TASK, "--targeting", str(t),
            "--world-image", "crucible-lh-d:go-go-toml-d4fe68839d05", "--group", "6", "--tool-limit", "110", "--ckpt-every", "20", "--ckpt-cap", "15",
            "--seq-len", "65536", "--cp", "1", "--ep", "2", "--workdir", "/n/ct", "--hf-home", "/h/hf", "--partition", "ghx4-interactive", "--time", "01:55:00"]
    monkeypatch.setattr("sys.argv", argv)
    dr.main()
    return stage, Path("/n/ct/runs/v")


def test_render_verify_writes_a_probe_and_the_phased_job(tmp_path: Path, monkeypatch) -> None:
    stage, out = _render(tmp_path, monkeypatch)
    probe = tomllib.loads((stage / "trainer_probe.toml").read_text())
    assert probe["data"]["fake"]["batch_size"] == 2 and probe["enable_router_replay"] is False and probe["model"]["seq_len"] == 65536
    sync = tomllib.loads((stage / "trainer_sync.toml").read_text())
    asy = tomllib.loads((stage / "trainer_async.toml").read_text())
    assert sync["max_steps"] == 3 and asy["max_steps"] == 2 and sync["ckpt"]["keep_last"] == 2 and sync["output_dir"] == str(out / "sync")
    assert asy["output_dir"] == str(out / "async") and sync["model"]["cp"] == 1 and sync["model"]["ep"] == 2
    inf = tomllib.loads((stage / "inference.toml").read_text())
    assert inf["vllm"]["max_loras"] == 2 and inf["vllm"]["max_model_len"] == 65536  # the async phase shares this server
    assert list(json.loads((stage / "targeting.json").read_text())) == [TASK]
    p = (stage / "probe.sbatch").read_text()
    assert "--account=bjhg-dtai-gh" in p and "--gpus-per-node=2" in p and "--mem=120G" in p and "trainer_probe.toml" in p and "HF_HOME=/h/hf " in p
    v = (stage / "verify.sbatch").read_text()
    assert "--account=bjhg-dtai-gh" in v and "--gpus-per-node=4" in v and "unset SBATCH_ACCOUNT SLURM_ACCOUNT" in v
    # the Worlds: the shim first on PATH, node-local data without ACLs, the lockdown probe before vLLM starts
    assert "/n/ct/crucible/scripts/deltaai:$PATH" in v and "rsync -a --no-acls --no-xattrs" in v and "CRUCIBLE_SHIM_ALLOW=$J" in v
    assert v.index("LOCKDOWN OK") < v.index("uv run inference")
    # the phases, in order, each loop's run dir node-local; async flags only in the async phase; the resume via launch_toml
    i_sync, i_res, i_async = (v.index(f"watch_run {ph} ") for ph in ("sync", "resume", "async"))
    assert i_sync < i_res < i_async
    lines = {ph: next(x for x in v.splitlines() if x.strip().startswith(f"watch_run {ph} ")) for ph in ("sync", "resume", "async")}
    assert "--run-dir $J/runs/sync" in lines["sync"] and "--steps 2" in lines["sync"] and "--async-groups" not in lines["sync"]
    assert "--run-dir $J/runs/sync" in lines["resume"] and "--steps 3" in lines["resume"] and "--async-groups" not in lines["resume"]
    assert "--run-dir $J/runs/async" in lines["async"] and "--async-groups 4 --max-staleness 1" in lines["async"] and "--steps 2" in lines["async"]
    assert all("--debug $J/data/debug" in x and f"--tasks {TASK}" in x and f"--targeting {out}/targeting.json" in x for x in lines.values())
    assert all("--world-image" not in x and "--stage" not in x and "--sync-steps" not in x for x in lines.values())
    res = v[i_sync:i_res]
    assert "stop_trainer" in v[i_sync:i_async] and v.count("launch_toml(") == 3 and "trainer_sync_run.toml" in res
    assert v.count("torchrun --role=trainer") == 3 and "-resume " in v and "-async " in v


def test_the_in_job_train_command_parses_to_the_phase_config(tmp_path: Path, monkeypatch) -> None:
    stage, _ = _render(tmp_path, monkeypatch)
    v = (stage / "verify.sbatch").read_text()
    line = next(x for x in v.splitlines() if x.strip().startswith("watch_run async "))
    cmd = line.split('"', 2)[1].replace("$J", str(tmp_path / "J"))
    argv = cmd.split()[cmd.split().index("train"):]
    dr = _load("delta_rl")
    seen = {}

    def show(a):
        tl = dr._tinker_cli()
        cfg, d = tl.train_config(a), dr.delta_config(a)
        seen.update(cfg=cfg, d=d, env=dr.env_kw(a, tl), args=a)

    monkeypatch.setattr(dr, "cmd_train", show)
    monkeypatch.setattr("sys.argv", ["delta_rl.py", *argv])
    dr.main()
    cfg, d, env, a = seen["cfg"], seen["d"], seen["env"], seen["args"]
    assert (cfg.steps, cfg.group_size, cfg.async_groups, cfg.max_staleness) == (2, 6, 4, 1)
    assert (d.seq_len, d.cp, d.ep, d.data_ranks, d.hf_cache, d.pg_correction) == (65536, 1, 2, 2, "/h/hf", "sc+mis")
    assert env["tool_limit"] == 110 and (env["phi_checkpoint"].every, env["phi_checkpoint"].cap) == (20, 15)
    assert a.tasks == [TASK] and a.debug.endswith("/J/data/debug")


def test_passthrough_drops_the_verify_only_flags() -> None:
    dr = _load("delta_rl")
    got = dr.passthrough(["--tasks", "a", "--steps", "9", "--stage=s", "--world-image", "w", "--group", "6"], drop=dr.VERIFY_ONLY)
    assert got == ["--tasks", "a", "--group", "6"]
