"""P6 exit criterion (swarm-pipeline/00 §15 P6): a scripted-solver run of the same tasks through the native path
(``Runner`` + ``WorkspaceHarness`` + checkpoint sidecar + ``InfraWatch``, ``factory.envs.make_env``) and through the slime
generate path (``train.bridges.slime.generate`` over the World's ``Sandbox``) gives identical rewards, identical per-turn
credit (raw ΔΦ, M6's group rule, the honesty-composed sequence advantage and the per-token advantages) and identical
masking.

The task is the window lane's toy window (``test_m17_w_task.make_task``: three milestones, m03 requires m02, graded by
pytest on the host interpreter, path checks included), opened as a host World (``HostWindow``) so no Docker is needed;
``scripts/lh_bench.py equiv`` runs the same comparison on a real window in its Docker World."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m17_w_task import make_task

from crucible.core.schema.rollout import RolloutRecord
from crucible.factory.checkpoint import CheckpointConfig
from crucible.factory.envs import SlimeProvider, Timeouts, drop_infra_groups, make_env, register, register_script, reset_provider, scripted_policy, slime_rows
from crucible.factory.lanes.windows import WindowTask
from crucible.train.bridges.slime import credit as sc
from crucible.train.bridges.slime.episode import SNAPSHOT_CMD, ScriptedModel
from crucible.train.bridges.slime.filters import group_filter
from crucible.train.bridges.slime.generate import generate
from crucible.train.bridges.slime.types import MirrorSample

ETA = 0.5


class HostWindow(WindowTask):
    """The toy window in a host World (git_repo app on the local backend); ``_dead`` simulates the sandbox dying."""

    _dead = False

    async def open(self, workdir: Path):
        from crucible.core.schema.world import WorldSpec
        from crucible.core.store import Store
        from crucible.world.world import World

        workdir = Path(workdir).resolve()
        ws = workdir / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        spec = WorldSpec.model_validate({
            "world": "factory/window", "network": "no-network",
            "apps": [{"id": "repo", "template": "common/git_repo", "state": {"adapter": "fs", "params": {"root": str(ws)}},
                      "params": {"shell": True, "init_git": True, "seed_dir": str(self.base_dir), "max_output": 30000, "allow_unsafe_local_shell": True}}],
            "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}]})
        w = World(spec, instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
        await w.start()
        await w.reset()
        w._crucible_ws = ws
        w._crucible_alive = lambda: not type(self)._dead
        return w

    async def snapshot_patch(self, world) -> str:
        import asyncio

        r = await asyncio.to_thread(subprocess.run, ["bash", "-c", SNAPSHOT_CMD.format(base="base")], cwd=world._crucible_ws, capture_output=True,
                                    text=True, check=True)
        return r.stdout


def kill() -> None:
    HostWindow._dead = True


def scripts(rec: dict) -> dict[str, list[tuple]]:
    o = rec["oracle_files"]
    commit = "git add -A && git -c user.email=a@b -c user.name=a commit -qm wip"
    core_a = o["pkg/core.py"]
    status_ok = json.dumps({"R1": 0.9, "R2": 1.0, "R3": 0.8})
    return {
        "pass": [("write", "pkg/core.py", core_a), ("write", "pkg/beta.py", o["pkg/beta.py"]), ("shell", commit), ("write", "pkg/gamma.py", o["pkg/gamma.py"]),
                 ("write", "STATUS.json", status_ok), ("final", "All done.")],
        "fail": [("shell", "ls"), ("final", "Nothing to do.")],
        "hack": [("write", "pkg/core.py", core_a), ("write", "pkg/beta.py", o["pkg/beta.py"]), ("write", "pkg/gamma.py", o["pkg/gamma.py"]),
                 ("write", "conftest.py", "import pytest\n"), ("final", "Done.")],
        "partial": [("write", "pkg/core.py", core_a), ("write", "pkg/beta.py", o["pkg/beta.py"]), ("shell", commit), ("write", "pkg/core.py", "def base():\n    return 0\n"),
                    ("shell", commit),
                    ("write", "STATUS.json", json.dumps({"R1": 1.0, "R2": 1.0, "R3": 0.0})), ("final", "GIVE_UP: gamma is beyond me")],
        "infra": [("write", "pkg/core.py", core_a), ("chaos", kill), ("write", "pkg/beta.py", o["pkg/beta.py"]), ("final", "Done.")],
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    import asyncio

    tmp = tmp_path_factory.mktemp("p6eq")
    task, rec, _ = make_task(tmp)
    task.__class__ = HostWindow
    cfg = CheckpointConfig(every=2, cap=20, concurrency=1)
    env = make_env(task, checkpoint=cfg, timeouts=Timeouts(agent_s=300, rollout_s=900))
    sc_ = scripts(rec)
    native: dict[str, tuple[dict, RolloutRecord | None]] = {}

    async def run_native() -> None:
        from crucible.runtime.run import Runner

        for name, steps in sc_.items():
            HostWindow._dead = False
            spec = env.run_spec(run_id=f"n-{name}", policies=[scripted_policy(steps, pid=f"scripted-{name}")], k=1)
            await Runner(tmp / "native").run(spec)
            adir = min((tmp / "native" / f"n-{name}" / "samples").glob("*/a*"))
            r = json.loads((adir / "record.json").read_text())
            ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text()) if (adir / "rollout.json").exists() else None
            native[name] = (r, ro)

    asyncio.run(run_native())
    reset_provider(SlimeProvider(checkpoint=cfg))
    register(task)
    (row,) = slime_rows([task], agent="scripted")
    args = SimpleNamespace(crucible_agent_timeout_s=300, crucible_rollout_timeout_s=900, crucible_turn_eta=ETA, reward_key=None)
    slime: dict[str, MirrorSample] = {}

    async def run_slime() -> None:
        for i, (name, steps) in enumerate(sc_.items()):
            HostWindow._dead = False
            register_script(name, lambda steps=steps: ScriptedModel(steps))
            s = MirrorSample(index=i, group_index=0, prompt=row["prompt"], metadata={**row["metadata"], "script": name})
            (out,) = await generate(args, s, {})
            slime[name] = out

    asyncio.run(run_slime())
    HostWindow._dead = False
    yield SimpleNamespace(task=task, rec=rec, native=native, slime=slime, args=args, env=env, tmp=tmp)
    reset_provider(None)
    shutil.rmtree(tmp, ignore_errors=True)


def test_rewards_and_masking_are_identical(runs) -> None:
    got = {}
    for name, (r, _) in runs.native.items():
        s = runs.slime[name]
        assert bool(r["mask_sample"]) == bool(s.remove_sample), name
        if not r["mask_sample"]:
            assert r["reward"] == pytest.approx(s.reward), name
        got[name] = (None if r["mask_sample"] else r["reward"], None if s.remove_sample else s.reward)
    # the scripted outcomes are what they say: pass 1, no-op 0, hack 0 (conftest plumbing), partial Φ 1/3 -> 1/6, infra masked
    assert got == {"pass": (1.0, 1.0), "fail": (0.0, 0.0), "hack": (0.0, 0.0), "partial": (pytest.approx(1 / 6, abs=1e-5), pytest.approx(1 / 6, abs=1e-5)), "infra": (None, None)}
    assert runs.slime["hack"].metadata["crucible"]["hack_suspect"] and "path:plumbing" in runs.slime["hack"].metadata["crucible"]["path_violations"]
    assert runs.slime["infra"].metadata["abort_reason"] == "sandbox_terminated" and runs.slime["infra"].loss_mask == [0]
    f = runs.native["infra"][0]["failure"]
    assert f["kind"] == "crucible:env_container_lost" and "sandbox died" in f["reason"]  # infra blame (core.schema.run), masked


def test_per_turn_delta_phi_is_identical(runs) -> None:
    for name, (r, ro) in runs.native.items():
        if r["mask_sample"]:
            continue
        nat = {t.call_id: round(float(t.advantage or 0.0), 9) for t in ro.turns}
        sl = {t["call_id"]: t["delta_phi"] for t in runs.slime[name].metadata["crucible"]["turn_credit"]}
        assert nat == sl, name
    part = {t["call_id"]: t["delta_phi"] for t in runs.slime["partial"].metadata["crucible"]["turn_credit"]}
    assert min(part.values()) < 0 < max(part.values())  # a milestone gained, then lost (alpha overwritten)
    assert runs.slime["pass"].metadata["crucible"]["progress"]["phi_final"] == 1.0


def test_group_credit_honesty_and_token_advantages_are_identical(runs) -> None:
    from crucible.factory.honesty import compose_advantage, honesty_group_fn
    from crucible.train.credit import turn_credit_map

    names = list(runs.native)
    recs = [runs.native[n][1] if runs.native[n][1] is not None else RolloutRecord.model_validate({"rollout_id": n, "mask_sample": True}) for n in names]
    nat = turn_credit_map(recs, ["g"] * len(recs), eta=ETA)
    samples = [runs.slime[n] for n in names]
    gc = sc.group_credit(samples, eta=ETA)
    for i, n in enumerate(names):
        ro = recs[i]
        if runs.native[n][0]["mask_sample"]:
            assert gc[("rollout", i)]["masked"]
            continue
        want = [nat[(ro.rollout_id, t.call_id)] for t in ro.turns if t.completion_token_ids]
        assert gc[("rollout", i)]["turns"] == pytest.approx(want), n
    # the honesty-composed sequence advantage: the native group stage vs the slime post-process
    live = [ro for ro, n in zip(recs, names, strict=True) if not runs.native[n][0]["mask_sample"]]
    for ro in live:
        ro.advantage = None
    honesty_group_fn()("g", live)
    _, adv = sc.reward_post_process(runs.args, samples)
    for ro, n in zip(recs, names, strict=True):
        if runs.native[n][0]["mask_sample"]:
            continue
        i = names.index(n)
        assert adv[i] == pytest.approx(ro.advantage), n
    assert sc.compose_advantage(0.3, -0.1) == compose_advantage(0.3, -0.1) and sc.compose_advantage(-0.2, -0.05) == compose_advantage(-0.2, -0.05)
    # per token: the slime convert's turn advantages equal native's turn credit on each call's own completion tokens
    data = sc.convert(runs.args, samples)
    for i, n in enumerate(names):
        if runs.native[n][0]["mask_sample"]:
            assert not any(data["turn_advantages"][i]) and not any(data["loss_masks"][i])
            continue
        ro = recs[i]
        spans = [t["span"] for t in runs.slime[n].metadata["crucible"]["turn_credit"]]
        for t, (a, b) in zip([t for t in ro.turns if t.completion_token_ids], spans, strict=True):
            assert len(t.completion_token_ids) == b - a
            assert data["turn_advantages"][i][a:b] == pytest.approx([nat[(ro.rollout_id, t.call_id)]] * (b - a))
        assert data["rewards"][i] == pytest.approx(ro.advantage)


def test_whole_group_drop_matches_on_both_sides(runs) -> None:
    names = list(runs.native)
    recs = [runs.native[n][1] or RolloutRecord.model_validate({"rollout_id": n, "mask_sample": True}) for n in names]
    for ro, n in zip(recs, names, strict=True):
        ro.mask_sample = bool(runs.native[n][0]["mask_sample"])
    kept, _, refill = drop_infra_groups(recs, ["g"] * len(recs))
    out = group_filter(runs.args, [runs.slime[n] for n in names])
    assert kept == [] and refill == {"g": len(names)} and not out.keep and out.reason == "crucible_infra_in_group"
    live = [runs.slime[n] for n in names if n != "infra"]
    assert group_filter(runs.args, live).keep
    kept2, _, refill2 = drop_infra_groups([r for r, n in zip(recs, names, strict=True) if n != "infra"], ["g"] * (len(names) - 1))
    assert len(kept2) == len(names) - 1 and refill2 == {}
