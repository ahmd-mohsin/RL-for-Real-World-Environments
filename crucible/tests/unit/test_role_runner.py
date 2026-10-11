"""m13 G-11: forge roles as sandboxed Harbor wrapper tasks. The wrapper holds only the role's visible files, refuses
hidden material, declares no-network for the environment and the verifier and the model API only for the agent; the
stdlib validator gates the deliverable; HarborRole maps failures onto the orchestrator's retry and infra paths. With
Docker: a real Harbor job (the oracle agent stands in for the LLM) sees exactly the visible files, reaches the model API
and nothing else, and a sandboxed hacker's proposal is graded by A16 on a forged task with B.15 re-introduced."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import BaseModel

from crucible.interop.harbor.role_runner import HarborRole, LeakRefused, assemble_role_task


class Out(BaseModel):
    files_seen: list[str]
    net: str = ""


def test_wrapper_holds_only_visible_files_and_declares_network(tmp_path):
    from harbor.models.task.config import NetworkMode
    from harbor.models.task.task import Task

    w = assemble_role_task("builder", {"spec.md": "build it", "data/seed.json": "{}"}, "You are the Builder.", Out, tmp_path / "w")
    files = sorted(str(p.relative_to(w / "environment")) for p in (w / "environment").rglob("*") if p.is_file())
    assert files == ["data/seed.json", "spec.md"]
    t = Task(w)  # Harbor's own model parses the wrapper
    c = t.config
    assert c.environment.network_mode == NetworkMode.NO_NETWORK and c.verifier.network_mode == NetworkMode.NO_NETWORK
    assert c.agent.network_mode == NetworkMode.ALLOWLIST and c.agent.allowed_hosts == ["api.anthropic.com"]
    ins = (w / "instruction.md").read_text()
    assert "/app/builder-out.json` is your only deliverable" in ins and "data/seed.json" in ins and '"files_seen"' in ins
    with pytest.raises(LeakRefused, match="hidden material"):
        assemble_role_task("tester", {"spec.md": "x CRUCIBLE-CANARY-oracle-1234 y"}, "p", Out, tmp_path / "w2", forbidden=["CRUCIBLE-CANARY-oracle-1234"])
    with pytest.raises(ValueError, match="inside the workdir"):
        assemble_role_task("tester", {"../escape": "x"}, "p", Out, tmp_path / "w3")


def test_validator_gates_the_deliverable(tmp_path):
    w = assemble_role_task("r", {}, "p", Out, tmp_path / "w")

    def check(content):
        f = tmp_path / "r-out.json"
        if content is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(content)
        return subprocess.run([sys.executable, str(w / "tests" / "validate.py"), str(f)], capture_output=True, text=True, check=False)

    assert check(json.dumps({"files_seen": ["a"], "net": "x"})).returncode == 0
    bad = check(json.dumps({"files_seen": "a"}))
    assert bad.returncode == 1 and "missing key: files_seen" not in bad.stdout and "files_seen: expected array" in bad.stdout
    assert check(json.dumps({"net": "x"})).returncode == 1 and "missing key: files_seen" in check(json.dumps({"net": "x"})).stdout
    assert check("not json").returncode == 1 and check(None).returncode == 1


async def test_harbor_role_maps_failures(monkeypatch, tmp_path):
    from crucible.actors.llm import LLMOutputError
    from crucible.forge.orchestrator import RoleInfraError
    from crucible.interop.harbor import role_runner

    results = iter([{"output": None, "feedback": "your deliverable was not valid: missing key: files_seen", "cost_usd": 0.01, "error": None},
                    {"output": None, "feedback": None, "cost_usd": 0.0, "error": "trial failed: docker"},
                    {"output": Out(files_seen=["a"]), "feedback": None, "cost_usd": 0.02, "error": None}])
    seen = []

    async def fake_job(wrappers, role, output_model, **kw):
        seen.append((wrappers[0] / "instruction.md").read_text())
        return [{"wrapper": wrappers[0], **next(results)}]

    monkeypatch.setattr(role_runner, "run_role_job", fake_job)
    r = HarborRole("builder", Out, "Build.", agent="claude-code", workdir=tmp_path)
    with pytest.raises(LLMOutputError, match="missing key"):
        await r.run({"spec": {"a": 1}}, feedback=[], knobs={})
    with pytest.raises(RoleInfraError, match="docker"):
        await r.run({"spec": {"a": 1}}, feedback=["missing key: files_seen"], knobs={})
    out = await r.run({"spec": {"a": 1}}, feedback=[], knobs={})
    assert out.files_seen == ["a"] and abs(r.spend_usd - 0.03) < 1e-9
    assert "inputs/spec.json" in seen[0] and "Feedback on your previous attempt" in seen[1]


PROBE = r'''#!/bin/bash
cd /app
F=$(find . -type f -not -name '*-out.json' | sort | python3 -c "import sys,json; print(json.dumps([l.strip()[2:] for l in sys.stdin]))")
NET=$(python3 -c "
import urllib.request, urllib.error
out = []
for h in ('https://example.com', 'https://api.anthropic.com/v1/models'):
    try:
        urllib.request.urlopen(h, timeout=8); out.append(h + '=reached')
    except urllib.error.HTTPError as e: out.append(h + '=reached')
    except Exception as e: out.append(h + '=blocked')
print(' '.join(out))
")
echo "{\"files_seen\": $F, \"net\": \"$NET\"}" > /app/builder-out.json
'''


@pytest.mark.docker
async def test_real_harbor_job_isolation_and_network(tmp_path):
    from crucible.interop.harbor.role_runner import run_role_job

    w = assemble_role_task("builder", {"spec.md": "build it", "src/app.py": "print(1)\n"}, "You are the Builder.", Out, tmp_path / "w", solution_sh=PROBE)
    r = (await run_role_job([w], "builder", Out, agent="oracle", jobs_dir=tmp_path / "jobs"))[0]
    assert r["error"] is None and r["output"] is not None, r
    assert r["output"].files_seen == ["spec.md", "src/app.py"]  # only the visible files
    # request level, not TCP connect (a filtering proxy accepts the connect and drops the request)
    assert r["output"].net == "https://example.com=blocked https://api.anthropic.com/v1/models=reached", r["output"].net


@pytest.mark.docker
async def test_sandboxed_hacker_feeds_a16(tmp_path):
    """A sandboxed hacker (the oracle agent replays a scripted proposal in place of an LLM) proposes the forged approval
    against the B.15-weakened task; A16 grades it and confirms the hack."""
    import tempfile
    from pathlib import Path

    from crucible_pack_toy.forge.task import ToyForgedTask
    from crucible_pack_toy.forge.world_synth import forge

    from crucible.accept.policy import Policy
    from crucible.accept.run import accept_task
    from crucible.forge.hacker import SandboxedHacker

    col: list = []
    await forge(collect=col)
    t = next(x for x, _ in col if x.task_id.endswith("t2_approval"))
    cs = t.cs.model_copy(update={"checks": [c for c in t.cs.checks if c.id != "policy/approval_given_in_episode"]})
    weak = ToyForgedTask(t.spec, cs, t.oracle, t.negatives, pack_root=t.pack_root, world_file=t.world_file, scenario=t.sc, canary=t.canary)
    forged = [a for a in t.oracle if a.get("to") != "supervisor"]
    proposal = {"attempts": [{"strategy": "authority_not_obtained", "actions": forged, "rationale": "skip asking; reuse the approver from the trajectory"}]}
    sol = "#!/bin/bash\ncat > /app/hacker-out.json <<'JSON'\n" + json.dumps(proposal) + "\nJSON\n"
    weak.hacker = lambda: SandboxedHacker(agent="oracle", forbidden=[t.canary] if t.canary else [], role_kwargs={"solution_sh": sol, "workdir": tmp_path})
    r = await accept_task(weak, policy=Policy(required=["A16"]), cache_root=Path(tempfile.mkdtemp()), only=["A16"], reuse=False)
    a16 = r.results["A16"]
    assert str(a16.status) == "fail" and a16.measured["confirmed_hacks"] == 1 and a16.measured["hacks"][0]["strategy"] == "authority_not_obtained"
