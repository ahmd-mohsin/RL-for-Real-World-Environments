"""P6 slime bridge units (swarm-pipeline/00 §11; 03 §4.3, §5): the ported sandbox protocol and its World provider, the
JSONL rows (held-out refused), the generate function's two timeouts and masking, the group filters (whole-group infra
drop, staleness cap, nonzero spread), the advantage hook, the field-list patch against the real slime checkout, and the
mirrors checked field by field against slime's own source."""

from __future__ import annotations

import ast
import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.runtime.harness.sandbox import EXIT_TIME_BUDGET_EXCEEDED, ExecSandbox, Sandbox, SandboxTerminatedError, exec_and_wait
from crucible.train.bridges.slime import credit as sc
from crucible.train.bridges.slime.episode import SNAPSHOT_CMD, ScriptedCLIHarness, ScriptedModel
from crucible.train.bridges.slime.filters import group_filter, staleness, staleness_sample_filter
from crucible.train.bridges.slime.generate import generate, outcome_reward
from crucible.train.bridges.slime.rows import SealedRowError, check_row, read_rows, slime_row, write_rows
from crucible.train.bridges.slime.types import MirrorDynamicFilterOutput, MirrorSample
from crucible.world.exec import LocalExec


def _frameworks() -> Path:
    """``CRUCIBLE_FRAMEWORKS_DIR``, else ``RL-Environments`` next to the checkout (``_upstream.py``), else next to the main
    checkout of a git worktree."""
    import os

    if os.environ.get("CRUCIBLE_FRAMEWORKS_DIR"):
        return Path(os.environ["CRUCIBLE_FRAMEWORKS_DIR"])
    here = Path(__file__).resolve().parents[3] / "RL-Environments" / "06-frameworks-and-env-code"
    if here.is_dir():
        return here
    common = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--git-common-dir"], capture_output=True, text=True, check=False).stdout.strip()
    return Path(common).resolve().parent / "RL-Environments" / "06-frameworks-and-env-code"


SLIME = _frameworks() / "thudm-slime"
needs_slime = pytest.mark.skipif(not (SLIME / "slime").is_dir(), reason="the slime study clone is not present")


# ---- the sandbox protocol ---------------------------------------------------------------------------------------------------
def test_exec_sandbox_is_slimes_sandbox_and_round_trips(tmp_path: Path) -> None:
    async def go() -> None:
        async with ExecSandbox(LocalExec(), root=tmp_path) as sb:
            assert isinstance(sb, Sandbox)
            assert await sb.exec("echo hi; echo err >&2; exit 3") == (3, "hi\n", "err\n")
            await sb.write_file("d/a.txt", "x" * 300_000)  # past the 128 KiB argument cap: content travels on stdin
            await sb.write_file("b.bin", b"\x00\x01")
            assert (tmp_path / "d" / "a.txt").read_text() == "x" * 300_000 and (tmp_path / "b.bin").read_bytes() == b"\x00\x01"
            assert await sb.read_file("d/a.txt") == "x" * 300_000 and await sb.read_file("missing.txt") == ""
            if shutil.which("setsid"):  # slime's detached spawn needs util-linux setsid (the docker test runs it everywhere)
                await _exec_and_wait_cases(sb)
        with pytest.raises(SandboxTerminatedError):
            await sb.exec("true")  # closed

    asyncio.run(go())


async def _exec_and_wait_cases(sb) -> None:
    code, out = await exec_and_wait(sb, cmd="sh -c 'echo done; exit 4'", time_budget_sec=10, tag=f"t{uuid.uuid4().hex[:6]}", want_output=True, poll_s=0.05)
    assert (code, out.strip()) == (4, "done")
    code, _ = await exec_and_wait(sb, cmd="sleep 5", time_budget_sec=0.3, tag=f"t{uuid.uuid4().hex[:6]}", poll_s=0.05)
    assert code == EXIT_TIME_BUDGET_EXCEEDED


def test_a_dead_sandbox_raises_terminated(tmp_path: Path) -> None:
    class Gone:
        name = "docker"

        async def run(self, cmd, **kw):
            return SimpleNamespace(rc=137, stdout="", stderr="Error: No such container: x")

    async def dead() -> bool:
        return False

    async def go() -> None:
        sb = ExecSandbox(Gone(), root="/w", alive=dead)
        with pytest.raises(SandboxTerminatedError):
            await sb.exec("ls")
        assert sb.terminated and "No such container" in sb.terminated
        with pytest.raises(SandboxTerminatedError):
            await sb.read_file("a")

    asyncio.run(go())


@pytest.mark.docker
def test_world_container_has_no_egress_and_its_death_is_infra() -> None:
    from crucible.world.exec import DockerExec

    img = next((x for x in subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True,
                                          check=False).stdout.split() if x.startswith("crucible-lh-w:")), None)
    if img is None:
        pytest.skip("no World image")
    name = f"crucible-p6-test-{uuid.uuid4().hex[:8]}"
    subprocess.run(["docker", "run", "-d", "--rm", "--network", "none", "--name", name, img, "sleep", "120"], check=True, capture_output=True)
    try:
        async def go() -> None:
            sb = ExecSandbox(DockerExec(name), root="/tmp")
            probe = await sb.egress_probe()
            assert probe["public_blocked"] and probe["ok"], probe
            await _exec_and_wait_cases(sb)
            await asyncio.to_thread(subprocess.run, ["docker", "kill", name], capture_output=True, check=False)
            with pytest.raises(SandboxTerminatedError):
                await sb.exec("ls")

        asyncio.run(go())
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


# ---- rows --------------------------------------------------------------------------------------------------------------------
def _ref(split: str = "train", tid: str = "w1") -> dict:
    return {"lane": "window", "task_id": tid, "source_id": tid, "store": "/s", "split": split, "prompt_sha256": "p", "checkset_sha256": "c",
            "verifier_version": "v"}


def test_rows_carry_the_metadata_and_never_a_held_out_task(tmp_path: Path) -> None:
    good = slime_row(instance_id="w1", prompt="Do it.", epilogue="How this works.", image="img@sha", workdir="/w", agent="claude_code", crucible_task_ref=_ref())
    assert good["prompt"].endswith("How this works.\n") and good["metadata"]["split"] == "train" and check_row(good) == []
    info = write_rows([good], tmp_path / "rows.jsonl")
    assert info["rows"] == 1 and read_rows(tmp_path / "rows.jsonl") == [good]
    held = slime_row(instance_id="h1", prompt="x", image="i", workdir="/w", agent="codex", crucible_task_ref=_ref("heldout", "h1"))
    assert "the task ref is split 'heldout'" in check_row(held)
    with pytest.raises(SealedRowError):
        write_rows([good, held], tmp_path / "bad.jsonl")
    assert not (tmp_path / "bad.jsonl").exists()
    with pytest.raises(SealedRowError):
        write_rows([good], tmp_path / "bad.jsonl", sealed={"w1"})  # an id a benchmark manifest seals
    with pytest.raises(ValueError):
        slime_row(instance_id="x", prompt="x", image="i", workdir="/w", agent="gpt", crucible_task_ref=_ref())


# ---- generate: masking and the two timeouts --------------------------------------------------------------------------------
class FakeEnv:
    """A host task: write ``ok.txt`` containing ``ok``; Φ = 1 when it does. No checkpoint observer."""

    task_id, objective_id, verifier_version, phi_weight = "fake", "obj", "v1", 0.5
    prompt = "Write ok.txt containing ok."

    def __init__(self, steps, *, delay=0.0, die=False):
        self.steps, self.delay, self.die = steps, delay, die

    async def open(self, workdir: Path):
        ws = Path(workdir) / "ws"
        ws.mkdir(parents=True)
        await asyncio.to_thread(subprocess.run, ["bash", "-c", "git init -q && git -c user.email=a@b -c user.name=a commit -q --allow-empty -m base && git tag base"],
                                cwd=ws, check=True)
        return SimpleNamespace(world=None, sandbox=ExecSandbox(LocalExec(), root=ws), ws=ws)

    def observer(self, handle, out_dir):
        return None

    async def snapshot(self, handle) -> str:
        return (await handle.sandbox.exec(SNAPSHOT_CMD.format(base="base")))[1]

    async def alive(self, handle) -> bool:
        return not self.die

    async def close(self, handle) -> None:
        pass

    async def grade(self, patch, out_dir, *, trail):
        from crucible.verify.runtime.plan import GradeReport

        ok = "+ok" in patch
        return GradeReport(task="fake", tier="full", checks=[], gates={"c_ok": "pass" if ok else "fail"}, reward=float(ok), outcome=float(ok),
                           infra={"status": "ok"}, trusted=True, elapsed_s=0.0, meta={"phi": 1.0 if ok else 0.0, "full_pass": ok})

    def harness(self, agent):
        return ScriptedCLIHarness()

    def model(self, md, sp):
        return ScriptedModel(self.steps, delay_s=self.delay)


def _gen(env, md=None, **args):
    a = SimpleNamespace(crucible_env_provider=SimpleNamespace(resolve=lambda _md: env), reward_key=None, **args)
    s = MirrorSample(index=0, group_index=0, metadata={"split": "train", **(md or {})})
    return asyncio.run(generate(a, s, {}))[0]


def test_generate_grades_and_sets_the_reward_rule() -> None:
    s = _gen(FakeEnv([("write", "ok.txt", "ok\n"), ("final", "done")]))
    assert not s.remove_sample and s.reward == 1.0 and s.status.value == "completed"
    c = s.metadata["crucible"]
    assert c["reward_rule"] == "0.5*phi + 0.5*[all gates]" and c["objective_id"] == "obj" and c["turns"] == 2
    assert sum(s.loss_mask) == sum(b - a for a, b in (t["span"] for t in c["turn_credit"])) and len(s.loss_mask) == s.response_length
    assert _gen(FakeEnv([("final", "nothing")])).reward == 0.0


def test_agent_timeout_is_a_graded_limit_stop_and_rollout_timeout_is_masked() -> None:
    steps = [("write", "ok.txt", "ok\n"), ("shell", "true"), ("shell", "true"), ("final", "done")]
    s = _gen(FakeEnv(steps, delay=0.4), crucible_agent_timeout_s=1.0, crucible_rollout_timeout_s=60)
    assert not s.remove_sample and s.status.value == "truncated" and s.metadata["crucible"]["truncated"] and s.reward == 1.0  # graded as it stands
    m = _gen(FakeEnv(steps, delay=0.4), crucible_agent_timeout_s=30, crucible_rollout_timeout_s=0.5)
    assert m.remove_sample and m.metadata["abort_reason"] == "rollout_timeout" and m.loss_mask == [0]


def test_masked_failures_are_removed_samples_never_zero_rewards() -> None:
    dead = _gen(FakeEnv([("write", "ok.txt", "ok\n"), ("final", "done")], die=True))
    assert dead.remove_sample and dead.metadata["abort_reason"] == "sandbox_terminated" and dead.metadata["crucible_masked"]
    held = _gen(FakeEnv([("final", "x")]), md={"split": "heldout"})
    assert held.remove_sample and held.metadata["abort_reason"].startswith("refused")

    class Unmeasured(FakeEnv):
        async def grade(self, patch, out_dir, *, trail):
            r = await super().grade(patch, out_dir, trail=trail)
            r.infra = {"status": "masked", "why": "verifier timeout"}
            return r

    um = _gen(Unmeasured([("final", "x")]))
    assert um.remove_sample and um.metadata["abort_reason"] == "grade_unmeasured"


def test_outcome_reward_rule() -> None:
    rep = SimpleNamespace(reward=0.75, meta={"phi": 0.5, "full_pass": True})
    assert outcome_reward(rep, 0.5) == (0.75, "0.5*phi + 0.5*[all gates]")
    assert outcome_reward(SimpleNamespace(reward=1.0, meta={"phi": 1.0, "full_pass": True, "hack_suspect": True}), 0.5)[0] == 0.0
    assert outcome_reward(SimpleNamespace(reward=0.3, meta={}), 0.5) == (0.3, "task reward")


# ---- filters -----------------------------------------------------------------------------------------------------------------
def _s(r: float, *, versions=("5",), masked=False, i=0) -> MirrorSample:
    return MirrorSample(index=i, group_index=0, reward=r, weight_versions=list(versions), remove_sample=masked)


def test_group_filter_drops_infra_groups_stale_groups_and_flat_groups() -> None:
    args = SimpleNamespace(reward_key=None, crucible_max_staleness=2)
    assert isinstance(group_filter(args, [_s(1), _s(0)]), MirrorDynamicFilterOutput)
    assert group_filter(args, [_s(1), _s(0)]).keep
    out = group_filter(args, [_s(1), _s(0), _s(0, masked=True)])
    assert not out.keep and out.reason == "crucible_infra_in_group"
    assert group_filter(args, [_s(1), _s(1)]).reason == "zero_std_1.0"
    assert group_filter(args, [[_s(0), _s(1)], [_s(0)]]).keep  # a trajectory of segments counts its terminal reward once
    stale = group_filter(args, [_s(1, versions=("1", "5")), _s(0)], current_weight_version=5)
    assert not stale.keep and stale.reason == "crucible_stale"
    assert group_filter(args, [_s(1, versions=("3",)), _s(0)], current_weight_version=5).keep  # age 2 = tau: kept
    assert staleness(_s(1, versions=("3", "4")), 6) == 3 and staleness(_s(1, versions=("x",)), 6) is None


def test_staleness_sample_filter_removes_stale_groups_in_place() -> None:
    args = SimpleNamespace(reward_key=None, crucible_max_staleness=1)
    groups = [[_s(1, versions=("9",)), _s(0, versions=("10",))], [_s(1, versions=("7",)), _s(0, versions=("10",))]]
    staleness_sample_filter(args, groups)
    assert len(groups) == 1 and groups[0][0].weight_versions == ["9"]  # newest version in the batch is 10; age 3 > 1 dropped


# ---- credit hooks ------------------------------------------------------------------------------------------------------------
def test_advantage_adds_turn_credit_and_refuses_a_missing_patch() -> None:
    torch = pytest.importorskip("torch")
    args = SimpleNamespace(context_parallel_size=1)
    data = {"kl": [torch.zeros(4), torch.zeros(2)], "rewards": [0.5, -0.25], "turn_advantages": [[0.0, 0.1, 0.1, 0.0], [-0.2, 0.0]]}
    sc.advantage(args, data)
    assert data["advantages"][0].tolist() == pytest.approx([0.5, 0.6, 0.6, 0.5]) and data["advantages"][1].tolist() == pytest.approx([-0.45, -0.25])
    assert data["returns"][1].tolist() == pytest.approx([-0.45, -0.25])
    bare = {"kl": [torch.zeros(2)], "rewards": [1.0]}
    with pytest.raises(KeyError, match="batch_builder_turn_advantages.patch"):
        sc.advantage(args, bare)
    sc.advantage(SimpleNamespace(context_parallel_size=1, crucible_allow_no_turn_credit=True), bare)
    assert bare["advantages"][0].tolist() == [1.0, 1.0]


def test_compose_advantage_mirror_matches_the_factory() -> None:
    from crucible.factory.honesty import compose_advantage

    for a in (-1.0, -0.2, 0.0, 0.1, 0.7):
        for t in (0.0, -0.05, -0.5):
            assert sc.compose_advantage(a, t) == compose_advantage(a, t)


# ---- against the real slime source -------------------------------------------------------------------------------------------
@needs_slime
def test_the_field_list_patch_applies_to_slime() -> None:
    patch = Path(sc.__file__).with_name(sc.PATCH_FILE)
    r = subprocess.run(["git", "-C", str(SLIME), "apply", "--check", str(patch)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    text = patch.read_text()
    assert '+            "turn_advantages",' in text and "@@ -485,6 +485,7 @@" in text  # inside batch_builder.py:467-487


def _dataclass_fields(path: Path, cls: str) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    return {s.target.id: ast.unparse(s.value) if s.value is not None else "" for s in node.body if isinstance(s, ast.AnnAssign)}


@needs_slime
def test_mirrors_match_slime_field_by_field() -> None:
    real = _dataclass_fields(SLIME / "slime" / "utils" / "types.py", "Sample")
    mine = _dataclass_fields(Path(__file__).resolve().parents[2] / "src" / "crucible" / "train" / "bridges" / "slime" / "types.py", "MirrorSample")
    for k, v in mine.items():
        assert k in real, k
        assert v.replace("Status.", "Status.") == real[k] or k == "status", (k, v, real[k])
    df_real = _dataclass_fields(SLIME / "slime" / "rollout" / "filter_hub" / "base_types.py", "DynamicFilterOutput")
    df_mine = _dataclass_fields(Path(__file__).resolve().parents[2] / "src" / "crucible" / "train" / "bridges" / "slime" / "types.py", "MirrorDynamicFilterOutput")
    assert df_mine == df_real


@needs_slime
def test_sandbox_protocol_matches_slime() -> None:
    import inspect

    from crucible.runtime.harness import sandbox as ours

    tree = ast.parse((SLIME / "slime" / "agent" / "sandbox.py").read_text())
    proto = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Sandbox")
    real = {f.name: [a.arg for a in [*f.args.args, *f.args.kwonlyargs]] for f in proto.body if isinstance(f, ast.AsyncFunctionDef)}
    mine = {n: list(inspect.signature(getattr(ours.Sandbox, n)).parameters) for n in real}
    assert mine == real
    assert json.dumps(sorted(real)) == json.dumps(["__aenter__", "__aexit__", "exec", "read_file", "write_file"])
