"""Forge roles as sandboxed Harbor wrapper tasks (m13 G-11; research m7-m8/02 §7.3). DEPEND on Harbor; PORT of the
structure of ``harbor/analyze/checker.py`` (``assemble_check_task`` / ``_run_check_job`` / ``_extract_check_result``,
Apache-2.0, credited in THIRD_PARTY.md).

A role (Builder, Test writer, a coding hacker) runs as an installed agent (Claude Code, Codex, mini-swe-agent, …) inside a
wrapper task whose sandbox holds **only the role's visible inputs**:

- ``environment/`` is exactly the visible files (a prebuilt image with no Dockerfile: Harbor uploads it to the workdir).
  ``forbidden`` strings (zone canaries, hidden strings, oracle lines) are scanned for before assembly: a wrapper that
  would carry one is refused, never run.
- ``instruction.md``: the role prompt, the visible file tree, and "``/app/<role>-out.json`` is your only deliverable",
  with the output's JSON schema.
- **Network:** the environment baseline and the verifier are ``no-network``; the agent phase is an allowlist of the
  model API hosts only (Harbor's egress-control sidecar enforces it).
- ``tests/validate.py`` (stdlib only) checks the deliverable against the schema's required keys and top-level types;
  reward 1 means a well-formed deliverable, never a correct one. The full pydantic parse happens here; a failure becomes
  the next attempt's feedback (the orchestrator's retry loop).

``HarborRole`` adapts this to the orchestrator's ``Role`` protocol (``staged`` inputs → visible files).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

MODEL_HOSTS = {"anthropic": ["api.anthropic.com"], "openai": ["api.openai.com"]}

VALIDATE = r'''"""Validate the role deliverable (stdlib only): JSON object, required keys, top-level types."""
import json
import sys
from pathlib import Path

TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool}


def main() -> int:
    schema = json.loads((Path(__file__).parent / "schema.json").read_text())
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"missing deliverable: {p}")
        return 1
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"invalid JSON: {e}")
        return 1
    if not isinstance(data, dict):
        print("the deliverable must be a JSON object")
        return 1
    errors = [f"missing key: {k}" for k in schema.get("required", []) if k not in data]
    for k, spec in (schema.get("properties") or {}).items():
        t = spec.get("type")
        if k in data and isinstance(t, str) and t in TYPES and not isinstance(data[k], TYPES[t]):
            errors.append(f"{k}: expected {t}")
    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
'''

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
rm -f /tmp/rllm/reward.json /logs/verifier/reward.txt /logs/verifier/reward.json
if python3 /tests/validate.py /app/{out}; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


class LeakRefused(RuntimeError):
    pass


def _toml_list(xs: list[str]) -> str:
    return "[" + ", ".join(json.dumps(x) for x in xs) + "]"


def assemble_role_task(role: str, visible: dict[str, str], prompt: str, output_model: type[BaseModel], dest: Path, *,
                       model_hosts: list[str] | None = None, forbidden: list[str] = (), image: str = "python:3.13-slim",
                       agent_timeout_s: float = 1800.0, solution_sh: str | None = None) -> Path:
    """The wrapper task directory. ``solution_sh`` (tests only) lets Harbor's ``oracle`` agent produce the deliverable."""
    leaks = sorted({f"{p}: {s[:24]}…" for p, v in [*visible.items(), ("instruction", prompt)] for s in forbidden if s and s in v})
    if leaks:
        raise LeakRefused(f"{role}: the wrapper would carry hidden material: {leaks[:5]}")
    for p in visible:
        if p.startswith("/") or ".." in Path(p).parts:
            raise ValueError(f"visible path must be relative and inside the workdir: {p}")
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "environment").mkdir(parents=True)
    for p, v in visible.items():
        f = dest / "environment" / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(v)
    out = f"{role}-out.json"
    schema = output_model.model_json_schema()
    (dest / "tests").mkdir()
    (dest / "tests" / "validate.py").write_text(VALIDATE)
    (dest / "tests" / "schema.json").write_text(json.dumps(schema, indent=1))
    (dest / "tests" / "test.sh").write_text(TEST_SH.format(out=out))
    (dest / "tests" / "test.sh").chmod(0o755)
    hosts = model_hosts if model_hosts is not None else MODEL_HOSTS["anthropic"]
    agent_net = f'network_mode = "allowlist"\nallowed_hosts = {_toml_list(hosts)}' if hosts else 'network_mode = "no-network"'
    (dest / "task.toml").write_text(f'''schema_version = "1.4"
artifacts = [{{ source = "/app/{out}", destination = "{out}" }}]

[agent]
timeout_sec = {agent_timeout_s}
{agent_net}

[verifier]
timeout_sec = 120.0
network_mode = "no-network"

[environment]
docker_image = "{image}"
workdir = "/app"
network_mode = "no-network"
''')
    tree = "\n".join(sorted(visible)) or "(no files)"
    (dest / "instruction.md").write_text(f"""{prompt.rstrip()}

## Your inputs (in /app, read-only material for this role)

{tree}

## Deliverable

`/app/{out}` is your only deliverable: one JSON object matching this JSON schema. Nothing else you write is kept.

```json
{json.dumps(schema, indent=1)}
```
""")
    if solution_sh is not None:
        (dest / "solution").mkdir()
        (dest / "solution" / "solve.sh").write_text(solution_sh)
        (dest / "solution" / "solve.sh").chmod(0o755)
    return dest


def extract_role_result(trial_result: Any, trial_dir: Path, role: str, output_model: type[BaseModel]) -> tuple[BaseModel | None, str | None, float]:
    """(parsed deliverable | None, feedback for the next attempt | None, cost USD)."""
    from harbor.models.trial.paths import TrialPaths

    try:
        cost = float(trial_result.compute_token_cost_totals()[3] or 0.0)
    except Exception:  # noqa: BLE001 (agents without token accounting)
        cost = 0.0
    if trial_result.exception_info is not None:
        raise RuntimeError(f"{role} trial failed: {trial_result.exception_info.exception_type}: {trial_result.exception_info.exception_message}")
    paths = TrialPaths(trial_dir)
    rewards = trial_result.verifier_result.rewards if trial_result.verifier_result else None
    f = paths.artifacts_dir / f"{role}-out.json"
    if (rewards or {}).get("reward") != 1 or not f.exists():
        why = paths.test_stdout_path.read_text().strip() if paths.test_stdout_path.exists() else "no verifier output"
        return None, f"your deliverable was not valid: {why[:800]}", cost
    try:
        return output_model.model_validate(json.loads(f.read_text())), None, cost
    except Exception as ex:  # noqa: BLE001 (pydantic or JSON errors are the role's, fed back)
        return None, f"your deliverable did not parse: {str(ex)[:800]}", cost


async def run_role_job(wrappers: list[Path], role: str, output_model: type[BaseModel], *, agent: str, model: str | None = None,
                       n_concurrent: int = 4, jobs_dir: Path | None = None, agent_env: dict[str, str] | None = None,
                       agent_kwargs: dict[str, Any] | None = None, environment: str = "docker") -> list[dict[str, Any]]:
    """All wrappers as one Harbor job; per wrapper ``{wrapper, output, feedback, cost_usd, error}`` in input order."""
    from harbor.job import Job
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.job.config import JobConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

    config = JobConfig()
    config.jobs_dir = Path(jobs_dir or tempfile.mkdtemp(prefix="crucible-roles-"))
    config.n_concurrent_trials = n_concurrent
    config.n_attempts = 1
    config.quiet = True
    config.agents = [AgentConfig(name=agent, model_name=model, kwargs=agent_kwargs or {}, env=agent_env or {})]
    config.environment = EnvironmentConfig(type=EnvironmentType(environment))
    config.tasks = [TaskConfig(path=w) for w in wrappers]
    config.datasets = []
    job = await Job.create(config)
    res = await job.run()
    by = {}
    for tr in res.trial_results:
        key = str(Path(tr.config.task.path).resolve())
        try:
            out, fb, cost = extract_role_result(tr, job.job_dir / tr.trial_name, role, output_model)
            by[key] = {"output": out, "feedback": fb, "cost_usd": cost, "error": None}
        except RuntimeError as ex:
            by[key] = {"output": None, "feedback": None, "cost_usd": 0.0, "error": str(ex)}
    return [{"wrapper": w, **by.get(str(Path(w).resolve()), {"output": None, "feedback": None, "cost_usd": 0.0, "error": "no trial result"})} for w in wrappers]


class HarborRole:
    """An orchestrator ``Role`` run by a Harbor installed agent in a wrapper task (staged inputs → visible files)."""

    def __init__(self, name: str, output_model: type[BaseModel], prompt: str, *, agent: str, model: str | None = None, version: str = "1",
                 forbidden: list[str] = (), model_hosts: list[str] | None = None, agent_env: dict[str, str] | None = None, workdir: Path | None = None,
                 solution_sh: str | None = None):
        self.name, self.version, self.output_model, self.prompt = name, version, output_model, prompt
        self.agent, self.model, self.forbidden, self.model_hosts, self.agent_env = agent, model, list(forbidden), model_hosts, agent_env
        self.workdir, self.solution_sh = Path(workdir or tempfile.mkdtemp(prefix=f"crucible-{name}-")), solution_sh
        self.spend_usd = 0.0
        self._n = 0

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> BaseModel:
        from crucible.actors.llm import LLMOutputError
        from crucible.forge.orchestrator import RoleInfraError

        self._n += 1
        visible = {f"inputs/{k}.json" if not isinstance(v, str) else f"inputs/{k}.txt": (v if isinstance(v, str) else json.dumps(v, indent=1))
                   for k, v in staged.items()}
        prompt = self.prompt + ("\n\n## Feedback on your previous attempt\n\n" + "\n".join(f"- {f}" for f in feedback) if feedback else "")
        w = assemble_role_task(self.name, visible, prompt, self.output_model, self.workdir / f"wrap-{self._n}", model_hosts=self.model_hosts,
                               forbidden=self.forbidden, solution_sh=self.solution_sh)
        r = (await run_role_job([w], self.name, self.output_model, agent=self.agent, model=self.model, n_concurrent=1,
                                jobs_dir=self.workdir / "jobs", agent_env=self.agent_env))[0]
        self.spend_usd += r["cost_usd"]
        if r["error"]:
            raise RoleInfraError(r["error"])
        if r["output"] is None:
            raise LLMOutputError(r["feedback"] or "no deliverable")
        return r["output"]
