"""M11 lab audit R-3: ``crucible-vf`` loads against current verifiers, never grades a separate-verifier task in the
solver's box, quiesces the box before a shared grade, and runs from a recorded environment.

verifiers removed ``HarborEnv`` after ae819de (isolated scoring moved into its rollout lifecycle); the plugin imported and
exported it, so a rebuilt ``.envs/verifiers`` could not load the taskset. ``ignore_separate_verifier`` and an agent-level
``scoring.mode=shared`` graded a separate task in the agent's box; the shared path ran no host-side kill before ``test.sh``.

Without ``crucible/.envs/verifiers`` (``scripts/build_verifiers_env.sh``) the plugin tests skip; the box shell, the
distribution provenance and the packaging run everywhere."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.interop.verifiers import box
from crucible.runtime import identity

CRUCIBLE = Path(__file__).resolve().parents[2]
SCRIPT = CRUCIBLE / "scripts" / "build_verifiers_env.sh"
VF_PY = CRUCIBLE / ".envs" / "verifiers" / "bin" / "python"


def _all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no __all__ in {path}")


# ------------------------------------------------------------------------------------------- packaging (no verifiers)

def test_the_plugin_exports_only_the_taskset_and_never_imports_harborenv():
    for p in (CRUCIBLE / "src" / "crucible" / "interop" / "verifiers" / "vf_taskset.py", CRUCIBLE / "src" / "crucible_vf" / "__init__.py"):
        assert _all(p) == ["CrucibleTaskset"], p
        assert "tasksets.harbor.env" not in p.read_text(encoding="utf-8")  # gone upstream at dca5562


def test_the_env_build_pins_a_full_verifiers_commit_and_harbor_0_24():
    text = SCRIPT.read_text(encoding="utf-8")
    commit = next(x.split('"')[1] for x in text.splitlines() if x.startswith("VERIFIERS_COMMIT="))
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    assert 'HARBOR_VERSION="0.24.0"' in text and "--frozen" in text
    assert commit[:7] in (CRUCIBLE / "THIRD_PARTY.md").read_text(encoding="utf-8")  # the recorded pin is the built one


# ------------------------------------------------------------------------------------------- provenance (no verifiers)

class _Dist:
    def __init__(self, version: str, direct_url: dict | None):
        self.version, self._du = version, direct_url

    def read_text(self, name: str):
        return json.dumps(self._du) if name == "direct_url.json" and self._du is not None else None


def test_dist_source_reads_a_vcs_commit(monkeypatch):
    monkeypatch.setattr(identity.importlib.metadata, "distribution",
                        lambda n: _Dist("0.4.0", {"url": "https://github.com/x/verifiers", "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}))
    assert identity.dist_source("verifiers") == {"version": "0.4.0", "commit": "a" * 40, "editable": None, "dirty": None}


def test_dist_source_reads_an_editable_checkouts_head(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("no git")
    repo = tmp_path / "verifiers"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "f").write_text("x")
    subprocess.run([*git, "add", "f"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(identity.importlib.metadata, "distribution",
                        lambda n: _Dist("0.0.0", {"url": repo.as_uri(), "dir_info": {"editable": True}}))
    assert identity.dist_source("verifiers") == {"version": "0.0.0", "commit": head, "editable": True, "dirty": False}  # 0.0.0 says nothing
    (repo / "f").write_text("changed")
    assert identity.dist_source("verifiers")["dirty"] is True
    inner = repo / "sub"  # a directory inside another checkout is not a checkout: never the enclosing repository's HEAD
    inner.mkdir()
    monkeypatch.setattr(identity.importlib.metadata, "distribution", lambda n: _Dist("0.0.0", {"url": inner.as_uri(), "dir_info": {"editable": True}}))
    assert identity.dist_source("verifiers")["commit"] is None


def test_dist_source_of_a_missing_distribution_tells_nothing(monkeypatch):
    def missing(name):
        raise identity.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(identity.importlib.metadata, "distribution", missing)
    assert identity.dist_source("verifiers") == {"version": None, "commit": None, "editable": None, "dirty": None}


def test_code_versions_record_the_verifiers_commit():
    identity.code_versions.cache_clear()
    try:
        assert "verifiers_commit" in identity.code_versions()
    finally:
        identity.code_versions.cache_clear()


# ------------------------------------------------------------------------------------------- the box quiesce (no verifiers)

def _sh(script: str, env: dict) -> str:
    return subprocess.run(["sh", "-c", script], env={**os.environ, **env}, capture_output=True, text=True, check=True, timeout=60).stdout


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="needs /proc")
def test_the_quiesce_kills_what_started_after_the_baseline_and_spares_the_rest(tmp_path):
    """Run on this host, so the kill is a stand-in that kills only the test's own process (``$CRUCIBLE_KILL``): every pid
    the selection names is recorded, and only the planted one dies."""
    base = box.baseline(_sh(box.LIST_PIDS, {}))
    planted = subprocess.Popen(["sleep", "60"])  # a process the policy left behind
    elder = os.getpid()  # in the baseline: spared
    try:
        time.sleep(0.2)
        named = tmp_path / "named"
        fake = tmp_path / "kill"
        fake.write_text(f'#!/bin/sh\necho "$2" >> {named}\n[ "$2" = "{planted.pid}" ] && exec kill "$1" "$2"\nexit 0\n')
        fake.chmod(0o755)
        killed, left, ran = box.parse(_sh(box.QUIESCE, {"CRUCIBLE_BASE": " ".join(base), "CRUCIBLE_KILL": str(fake)}))
        assert ran and planted.pid in killed and planted.pid not in {x["pid"] for x in left}
        assert planted.wait(timeout=10) == -9
        chosen = {int(x) for x in named.read_text().split()}
        base_pids = {int(t.partition(":")[0]) for t in base}
        assert planted.pid in chosen and elder not in chosen and not (chosen & base_pids)  # a baseline process is never a target
    finally:
        if planted.poll() is None:
            planted.kill()


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="needs /proc")
def test_a_process_that_survives_is_listed_and_judged(tmp_path):
    """No permission to kill (``$CRUCIBLE_KILL`` fails): what the policy started is left. A child of a baseline process
    (here: of pytest) reads as a respawn (masked); a process the policy's own process started is a survivor (scored 0)."""
    base = box.baseline(_sh(box.LIST_PIDS, {}))
    planted = subprocess.Popen(["sh", "-c", "sleep 60 & wait"])  # sh: a child of pytest; its sleep: a child of the policy's sh
    try:
        time.sleep(0.3)
        killed, left, ran = box.parse(_sh(box.QUIESCE, {"CRUCIBLE_BASE": " ".join(base), "CRUCIBLE_KILL": "false"}))
        by_pid = {x["pid"]: x for x in left}
        assert ran and planted.pid in by_pid and not killed
        assert any(x["ppid"] == str(planted.pid) for x in left)  # the sleep
        assert box.verdict(left, base, ran) == "survivors"
        assert box.verdict([by_pid[planted.pid]], base, ran) == "respawned"
    finally:
        planted.kill()
        planted.wait()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_a_reused_pid_is_not_spared():
    """The baseline is keyed by start time: a pid that is in the baseline but now belongs to another process is a target."""
    base = box.baseline(_sh(box.LIST_PIDS, {}))
    planted = subprocess.Popen(["sleep", "60"])
    try:
        time.sleep(0.2)
        forged = [*base, f"{planted.pid}:1"]  # its pid, another start time (a process long gone held it)
        killed, left, ran = box.parse(_sh(box.QUIESCE, {"CRUCIBLE_BASE": " ".join(forged), "CRUCIBLE_KILL": "false"}))
        assert planted.pid in {x["pid"] for x in left}
    finally:
        planted.kill()
        planted.wait()


def test_parse_and_verdict_read_the_receipt():
    base = ["1:5", "40:9"]
    assert box.parse("killed 12\nleft 13 12 S\nquiesced\n") == ([12], [{"pid": 13, "ppid": "12", "state": "S"}], True)
    assert box.parse("killed 12\n") == ([12], [], False)  # cut short: not quiesced
    assert box.baseline("1:5\n7:x\n40:9\n1:5\n") == base
    assert box.verdict([], base, False) == "error"
    assert box.verdict([], base, True) == "clean"
    assert box.verdict([{"pid": 50, "ppid": "40", "state": "S"}], base, True) == "respawned"  # a baseline service's child
    assert box.verdict([{"pid": 50, "ppid": "1", "state": "S"}], base, True) == "survivors"  # reparented to PID 1: the policy's
    assert box.verdict([{"pid": 50, "ppid": "?", "state": "?"}], base, True) == "survivors"  # unreadable: never a pass
    assert box.verdict([{"pid": 50, "ppid": "60", "state": "D"}], base, True) == "uninterruptible"
    assert box.verdict([{"pid": 50, "ppid": "40", "state": "S"}, {"pid": 51, "ppid": "70", "state": "S"}], base, True) == "survivors"


# ------------------------------------------------------------------------------------------- in .envs/verifiers

def _vf(code: str) -> str:
    if not VF_PY.exists():
        pytest.skip("verifiers env not built (scripts/build_verifiers_env.sh)")
    r = subprocess.run([str(VF_PY), "-c", code], capture_output=True, text=True, check=False, timeout=600, cwd=CRUCIBLE)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-3000:])
    return r.stdout


def _task_dir(root: Path, name: str, *, separate: bool) -> Path:
    d = root / name
    (d / "tests").mkdir(parents=True)
    (d / "solution").mkdir()
    (d / "instruction.md").write_text("write 1 to /tmp/answer\n")
    verifier = '[verifier]\ntimeout_sec = 60\n' + ('environment_mode = "separate"\n' if separate else "")
    (d / "task.toml").write_text('version = "1.0"\n\n[agent]\ntimeout_sec = 60\n\n' + verifier +
                                 '\n[environment]\ndocker_image = "python:3.12-slim"\n')
    (d / "tests" / "test.sh").write_text("#!/bin/bash\nmkdir -p /logs/verifier\n"
                                        'if [ "$(cat /tmp/answer 2>/dev/null)" = 1 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n')
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\necho 1 > /tmp/answer\n")
    return d


def test_the_plugin_loads_by_id_with_verifiers_own_loader(tmp_path):
    _task_dir(tmp_path, "shared", separate=False)
    out = _vf(f'''
import verifiers.v1 as vf
from verifiers.v1.utils.loaders import import_taskset, _plugin_class
m = import_taskset("crucible-vf")
TS = _plugin_class(m, vf.Taskset, "taskset")
from crucible.interop.verifiers.vf_taskset import CrucibleConfig
tasks = list(TS(CrucibleConfig(path="{tmp_path}")).load())
print("LOADED", TS.__name__, m.__all__, len(tasks), type(tasks[0]).__name__, tasks[0].data.verifier is None)
''')
    assert "LOADED CrucibleTaskset ['CrucibleTaskset'] 1 CrucibleTask True" in out


def test_a_separate_task_is_never_graded_in_the_agents_box(tmp_path):
    _task_dir(tmp_path, "sep", separate=True)
    out = _vf(f'''
from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.errors import TaskError
from crucible.interop.verifiers.vf_taskset import CrucibleConfig, CrucibleTaskset
try:
    CrucibleConfig(path="{tmp_path}", ignore_separate_verifier=True); print("IGNORE accepted")
except ValueError as e:
    print("IGNORE refused", "M11 lab audit R-3" in str(e))
t = next(iter(CrucibleTaskset(CrucibleConfig(path="{tmp_path}")).load()))
print("SEPARATE", t.data.verifier is not None, t.scoring.mode)
try:
    t.grading_task(None, scoring=ScoringConfig(mode="shared")); print("SHARED accepted")
except TaskError as e:
    print("SHARED refused", "sep" in str(e))
''')
    assert "IGNORE refused True" in out and "SEPARATE True isolated" in out and "SHARED refused True" in out


def test_a_shared_grade_quiesces_first_and_planted_files_do_not_count(tmp_path):
    """A fake runtime over a directory: the quiesce runs before staging, a survivor refuses the grade, and a mask.json or
    steps.jsonl planted before ``test.sh`` never reaches the result."""
    _task_dir(tmp_path / "ds", "shared", separate=False)
    out = _vf(f'''
import asyncio, json, shlex
from pathlib import Path
from verifiers.v1.errors import TaskError
from crucible.interop.verifiers import box
from crucible.interop.verifiers.vf_taskset import CrucibleConfig, CrucibleTaskset
D = Path("{tmp_path}") / "box"; D.mkdir()
class R:
    def __init__(self, left=""): self.cmds, self.left = [], left
    async def run(self, cmd, env=None):
        self.cmds.append(cmd[2] if cmd[:2] == ["sh", "-c"] else " ".join(cmd))
        out = ""
        if cmd[:2] == ["sh", "-c"] and cmd[2] == box.LIST_PIDS: out = "1:10\\n7:20\\n"
        elif cmd[:2] == ["sh", "-c"] and cmd[2] == box.QUIESCE: out = "killed 99\\n" + self.left + "quiesced\\n"
        elif cmd[:2] == ["sh", "-c"] and cmd[2].startswith("rm -f /logs/verifier/mask.json"):
            for p in shlex.split(cmd[2])[2:]: (D / Path(p).name).unlink(missing_ok=True)
        elif cmd == ["bash", "/tests/test.sh"]: (D / "reward.json").write_text(json.dumps({{"reward": 0.0}}))
        return type("X", (), {{"exit_code": 0, "stdout": out, "stderr": ""}})()
    async def write(self, path, data): pass
    async def read(self, path, max_bytes=None):
        p = D / Path(path).name
        if not p.exists(): raise OSError(path)
        return p.read_bytes()
class Tr:
    def __init__(self): self.metrics, self.info = {{}}, {{}}
    def record_metrics(self, mm): self.metrics.update(mm)
t = next(iter(CrucibleTaskset(CrucibleConfig(path="{tmp_path / 'ds'}")).load()))
tr, r = Tr(), R()
asyncio.run(t.setup(r, None, tr))
print("BASE", tr.info["crucible_baseline"])
(D / "mask.json").write_text(json.dumps({{"mask_sample": True}})); (D / "steps.jsonl").write_text("{{}}\\n")
g = t.grading_task(r, scoring=t.scoring)
asyncio.run(g.stage_verifier(tr, r))
first_q = next(i for i, c in enumerate(r.cmds) if c == box.QUIESCE); first_stage = next(i for i, c in enumerate(r.cmds) if "tests.tgz" in c)
print("ORDER", first_q < first_stage, tr.info["crucible_quiesce"])
out = asyncio.run(g.run_verifier(r, tr))
print("GRADE", out, "crucible_steps" in tr.info, "verifiers" in tr.info["crucible_provenance"])
def fresh(left):
    t = next(iter(CrucibleTaskset(CrucibleConfig(path="{tmp_path / 'ds'}")).load()))
    tr, r = Tr(), R(left=left)
    asyncio.run(t.setup(r, None, tr))
    return t.grading_task(r, scoring=t.scoring), tr, r
g2, tr2, r2 = fresh("left 99 55 S\\n")  # a policy process (its parent is no baseline service) outlived the kill
asyncio.run(g2.stage_verifier(tr2, r2))
staged = any("tests.tgz" in c for c in r2.cmds)
out2 = asyncio.run(g2.solved(r2, tr2))
print("SURVIVOR", out2, staged, tr2.metrics, [f["rule"] for f in tr2.info["crucible_flags"]], any(c == "bash /tests/test.sh" for c in r2.cmds))
g3, tr3, r3 = fresh("left 99 7 S\\n")  # only a baseline service's child: a respawn
try:
    asyncio.run(g3.stage_verifier(tr3, r3)); print("RESPAWN accepted")
except TaskError as e:
    print("RESPAWN masked", "quiesce_respawned" in str(e))
try:
    asyncio.run(fresh("")[0].stage_verifier(Tr(), r)); print("NOBASE accepted")
except TaskError as e:
    print("NOBASE refused")
g4, tr4, r4 = fresh("")
r4.config = type("C", (), {{"type": "subprocess"}})()
try:
    asyncio.run(g4.stage_verifier(tr4, r4)); print("SUBPROCESS accepted")
except TaskError as e:
    print("SUBPROCESS refused")
''')
    assert "BASE ['1:10', '7:20']" in out and "ORDER True" in out and "GRADE {'reward': 0.0} False True" in out
    assert "SURVIVOR {'reward': 0.0} False {'crucible/policy_process_survived': 1.0} ['policy_process_survived'] False" in out
    assert "RESPAWN masked True" in out and "NOBASE refused" in out and "SUBPROCESS refused" in out


def test_the_env_check_passes_where_the_env_is_built():
    """What CI runs on a host with the env: the env's verifiers is the pinned commit and Harbor is 0.24.0."""
    if not VF_PY.exists():
        pytest.skip("verifiers env not built (scripts/build_verifiers_env.sh)")
    r = subprocess.run(["bash", str(SCRIPT), "--check"], capture_output=True, text=True, check=False, timeout=600)
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_vf_validate_reports_gold_and_noop_valid_for_a_shared_task(tmp_path):
    """Acceptance 2: gold (solve.sh then the grade scores 1) and noop (the untouched box does not pass). Needs a container
    runtime for verifiers' default backend."""
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=60).returncode != 0:
        pytest.skip("vf-validate needs a running Docker daemon (verifiers' docker runtime)")
    _task_dir(tmp_path, "shared", separate=False)
    vfv = VF_PY.parent / "vf-validate"
    if not vfv.exists():
        pytest.skip("verifiers env not built (scripts/build_verifiers_env.sh)")
    out = tmp_path / "out"
    r = subprocess.run([str(vfv), "crucible-vf", "--taskset.path", str(tmp_path / "shared"), "--runtime.type", "docker", "--output-dir", str(out)],
                       capture_output=True, text=True, check=False, timeout=1800, cwd=CRUCIBLE)  # the default runtime is Prime's (remote)
    summaries = list(out.rglob("summary.json"))  # vf-validate exits 0 whatever it found: the summary is the verdict
    assert r.returncode == 0 and summaries, (r.stdout[-2000:], r.stderr[-2000:])
    checks = json.loads(summaries[0].read_text())["checks"]
    assert checks["gold"]["valid"] == 1 and checks["noop"]["valid"] == 1, checks
