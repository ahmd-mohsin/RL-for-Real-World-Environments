"""``crucible`` command line (grows with each module; M15 makes it the enterprise-facing tool).

    crucible registry                         plugins, templates, backends, actors
    crucible task show <dir>                  Harbor validation + Crucible overlay summary
    crucible seeds                            the coding pack's seed tasks and their calibration
    crucible world check-mimo <envs> [--transport mcp] [--backend agentenv] [--stride N] [--out f]
    crucible world check-awm [N] [--out f]
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click


@click.group()
def cli() -> None:
    """Crucible: workflow traces -> dense-reward RL environments, tasks and benchmarks."""


@cli.command()
def registry() -> None:
    """Show everything registered (built-ins and installed plugins)."""
    from crucible.core.registry import get_registry

    for kind, entries in get_registry().describe().items():
        click.echo(f"[{kind}]")
        for name, target in entries.items():
            click.echo(f"  {name:28s} {target}")


@cli.group()
def task() -> None:
    """Inspect tasks."""


@task.command("show")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def task_show(path: Path) -> None:
    """Validate a Harbor task dir (or a Crucible overlay) and summarize it."""
    if (path / "crucible" / "task.yaml").exists():
        from crucible.interop.harbor.task_io import load_crucible_task

        t = load_crucible_task(path)
        click.echo(f"crucible task {t.task.id}  horizon={t.task.horizon}  apps={[a.id for a in t.world.apps]}  backend={t.world.backend}")
        for f in t.card.value_proof.frontier:
            click.echo(f"  frontier {f.model:14s} {f.task_version or '':4s} score={f.score} gates={f.failing_gates}")
        s = t.harbor.summary
    else:
        from crucible.interop.harbor.task_io import load_harbor_task

        s = load_harbor_task(path).summary
    click.echo(json.dumps(s.model_dump(mode="json"), indent=1))


@cli.command()
def seeds() -> None:
    """List the coding pack's seed tasks (MVP showcase)."""
    from crucible.interop.harbor.task_io import load_crucible_task
    from crucible_pack_coding import seed_dirs

    for d in seed_dirs():
        t = load_crucible_task(d)
        last = t.card.value_proof.frontier[-1] if t.card.value_proof.frontier else None
        click.echo(f"{t.task.id:55s} {t.task.horizon}  harbor={t.harbor.summary.task_hash[:12]}  latest frontier: "
                   f"{last.model + ' ' + str(last.score) if last else '-'}")


@cli.group()
def world() -> None:
    """World runtime utilities."""


@world.command("check-mimo")
@click.argument("envs_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--transport", default="inprocess", type=click.Choice(["inprocess", "mcp"]))
@click.option("--backend", default="local", type=click.Choice(["local", "agentenv"]))
@click.option("--stride", default=1, type=int)
@click.option("--out", type=click.Path(path_type=Path))
def check_mimo(envs_root: Path, transport: str, backend: str, stride: int, out: Path | None) -> None:
    """Load MiMo envs as Worlds and run the generality checks."""
    import tempfile
    from dataclasses import asdict

    from crucible.core.store import Store
    from crucible.interop.mimo.check import sweep
    from crucible.interop.mimo.importer import is_complete, list_mimo_envs

    envs = [e for e in list_mimo_envs(envs_root) if is_complete(e)][::stride]
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(sweep(envs, Store(Path(d) / "store"), Path(d) / "worlds", transport=transport, backend=backend))
    click.echo(f"{sum(r.ok for r in res)}/{len(res)} pass")
    if out:
        out.write_text(json.dumps([asdict(r) for r in res], indent=1))


@world.command("check-awm")
@click.argument("n", default=20, type=int)
@click.option("--out", type=click.Path(path_type=Path))
def check_awm(n: int, out: Path | None) -> None:
    """Load N OpenEnv AgentWorldModel-1K scenarios as Worlds and run the generality checks."""
    from dataclasses import asdict

    from crucible.interop.openenv.awm_check import sweep

    res = asyncio.run(sweep(n))
    click.echo(f"{sum(r.ok for r in res)}/{len(res)} pass")
    if out:
        out.write_text(json.dumps([asdict(r) for r in res], indent=1))


if __name__ == "__main__":
    cli()
