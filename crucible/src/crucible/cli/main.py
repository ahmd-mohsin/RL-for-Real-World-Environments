"""``crucible`` command line: the enterprise-facing tool (ARCHITECTURE §4 M15).

    crucible label --pack toy                 workflows -> traces with dense rewards + SFT / DPO / KTO datasets
    crucible build toy                        workflows -> RL tasks (forge, accept, export) + a benchmark report
    crucible pack new <name> | pack check     scaffold / validate a domain pack
    crucible bench run|report|compare         M13 benchmarks
    crucible ingest | signals | model | forge | accept | export | train   the module steps one at a time
    crucible registry | task show | seeds | world ...       inspection and World audits

Subcommands are imported only when invoked (rLLM's ``_LazyGroup``, NeMo-Gym's command table), so ``crucible --help``
stays fast. A user error (bad input, missing pack, scaffold conflict) prints one line; ``--verbose`` shows the
traceback (agentenv's ``_UserErrorsAreNotCrashes``).

Exit codes (M15 audit H2 V16, G2 R7; OpenEnv ``validate``'s 0/1/2/3 plus a usage code):

    0  ok
    1  the command ran and something failed (a check, a rejected task, an export error, nothing measured)
    2  click usage error; for ``pack check``: a level was skipped for a missing requirement
    3  internal error (a bug in crucible; the traceback is printed)
    4  user error (bad input: no such pack, an existing different run, an invalid key)

With ``--json`` every failure also prints ``{"error": {"type", "message", "kind", "exit"}}`` on stdout. An exception from
crucible's own code (``KeyError``, ``IndexError``, ``TypeError``, ``AttributeError``, a JSON decode error, ...) is an
internal error, never shown as a one-line user error.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

import click

LAZY = {  # name -> (module:attr, short help)
    "label": ("crucible.cli.studio:label", "Workflows -> dense-reward traces + SFT / DPO / KTO datasets."),
    "build": ("crucible.cli.studio:build", "Workflows -> RL tasks and a benchmark (ingest, forge, accept, export, bench)."),
    "pack": ("crucible.cli.studio:pack", "Create and validate domain packs."),
    "bench": ("crucible.cli.studio:bench", "Run, report and compare benchmarks."),
    "ingest": ("crucible.cli.studio:ingest", "M3: ingest a pack's workflow sources."),
    "signals": ("crucible.cli.studio:signals", "M4: signals, outcomes and pivots of a pack's traces."),
    "forge": ("crucible.cli.studio:forge", "M8: construct tasks from a pack's workflows."),
    "accept": ("crucible.cli.studio:accept", "M9: the full acceptance contract on a pack's forged tasks."),
    "train": ("crucible.cli.studio:train", "M12: run the toy training launcher (TinyPolicy) from a job spec."),
    "model": ("crucible.cli.studio:model", "M5: the workflow model (milestones, stages) of a pack's logs."),
    "export": ("crucible.cli.studio:export", "M11: export tasks (canonical, NeMo-Gym, OpenEnv) or episodes (ATIF)."),
}
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INTERNAL, EXIT_USER = 0, 1, 2, 3, 4


class UserError(ValueError):
    """Bad input to a command (raise this at the input checks)."""


# ValueError stays a user error: the library's input checks raise it ("no pack ...", "use a new --out"). Its subclasses
# that come from crucible's own parsing are bugs, as are KeyError / IndexError (LookupError was a user error: V16).
USER_ERRORS = (UserError, ValueError, FileNotFoundError, FileExistsError, PermissionError, NotADirectoryError, IsADirectoryError)
INTERNAL = (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError, UnicodeError, AssertionError, NameError, ZeroDivisionError)


class _Failed(click.ClickException):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.exit_code = code


def _json_requested(ctx: click.Context) -> bool:
    return "--json" in [*(getattr(ctx, "_protected_args", None) or []), *ctx.args]


def _error_json(e: BaseException, kind: str, code: int) -> None:
    click.echo(json.dumps({"error": {"type": type(e).__name__, "message": str(e)[:2000], "kind": kind, "exit": code}}, indent=1, sort_keys=True))


class CrucibleGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(super().list_commands(ctx)) | set(LAZY))

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        if name in LAZY:
            mod, attr = LAZY[name][0].split(":")
            return getattr(importlib.import_module(mod), attr)
        return super().get_command(ctx, name)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [(n, LAZY[n][1]) for n in sorted(LAZY)]
        rows += [(n, (c.get_short_help_str(60) if c else "")) for n, c in ((n, super(CrucibleGroup, self).get_command(ctx, n)) for n in super().list_commands(ctx))]
        with formatter.section("Commands"):
            formatter.write_dl(sorted(rows))

    def invoke(self, ctx: click.Context):
        as_json = _json_requested(ctx)
        verbose = bool(ctx.obj and ctx.obj.get("verbose"))
        try:
            return super().invoke(ctx)
        except (click.exceptions.Exit, click.exceptions.Abort):
            raise
        except click.ClickException as e:
            if as_json:
                _error_json(e, "usage", e.exit_code)
            raise
        except SystemExit:  # a command that failed after printing its own result (JSON or human) sets its code
            raise
        except Exception as e:
            from crucible.studio.pack import ScaffoldConflict

            internal = isinstance(e, INTERNAL) and not isinstance(e, UserError)
            if not internal and isinstance(e, (*USER_ERRORS, ScaffoldConflict)):
                if as_json:
                    _error_json(e, "user", EXIT_USER)
                if verbose:
                    raise
                msg = str(e) if isinstance(e, ScaffoldConflict) else f"{type(e).__name__}: {e}"
                raise _Failed(msg, EXIT_USER) from None
            import traceback

            if as_json:
                _error_json(e, "internal", EXIT_INTERNAL)
            click.echo("internal error (a bug in crucible; please report it with this traceback):", err=True)
            click.echo("".join(traceback.format_exception(e)), err=True)
            raise SystemExit(EXIT_INTERNAL) from None


@click.group(cls=CrucibleGroup)
@click.option("--verbose", is_flag=True, help="show tracebacks for user errors")
@click.option("--set", "sets", multiple=True, metavar="KEY=VALUE", help="override a crucible.toml setting (dotted key, TOML value); the top layer")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, sets: tuple[str, ...]) -> None:
    """Crucible: workflow traces -> dense-reward RL environments, tasks, datasets and benchmarks."""
    ctx.obj = {"verbose": verbose}
    if sets:
        from crucible.core.config import set_overrides

        set_overrides(list(sets))


@cli.command("config")
def config_cmd() -> None:
    """Show the configuration in use: the layers read (lowest first) and the merged settings, secrets masked."""
    from crucible.core.config import get_config

    cfg = get_config()
    click.echo(json.dumps({"sources": cfg.sources, "config": {k: v for k, v in cfg.redacted().items() if k != "sources"}}, indent=1, default=str))


@cli.command()
@click.option("--check", is_flag=True, help="Resolve every pointer; exit 1 on a broken pointer or a failed plugin (CI).")
def registry(check: bool) -> None:
    """Show everything registered (built-ins and installed plugins)."""
    from crucible.core.registry import get_registry

    if check:
        rep = get_registry().check()
        for section, rows in rep.items():
            click.echo(f"[{section}] {len(rows)}")
            for name, why in rows.items():
                click.echo(f"  {name:36s} {why}")
        raise SystemExit(1 if rep["broken"] or rep["failed_plugins"] else 0)
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
    from crucible_pack_coding import seed_dirs

    from crucible.interop.harbor.task_io import load_crucible_task

    for d in seed_dirs():
        t = load_crucible_task(d)
        last = t.card.value_proof.frontier[-1] if t.card.value_proof.frontier else None
        click.echo(f"{t.task.id:55s} {t.task.horizon}  harbor={t.harbor.summary.task_hash[:12]}  latest frontier: "
                   f"{last.model + ' ' + str(last.score) if last else '-'}")


@cli.group()
def world() -> None:
    """World runtime utilities."""


@world.command("pin")
@click.argument("image")
@click.option("--docker-host", default=None, help="the daemon that holds the image (default: DOCKER_HOST, else the local one)")
def world_pin(image: str, docker_host: str | None) -> None:
    """Print IMAGE's identity (the digest of its filesystem layers) to pin as ``image_digest``: a sandbox's
    ``backend_params.sandbox.image_digest``, or Harbor's ``backend_params.image_digest``. A World whose image differs then
    refuses to start (crucible:stale_image) instead of running a re-pushed tag (M1 audit R-14)."""
    from crucible.world.sandbox import image_id

    digest = asyncio.run(image_id(image, docker_host))
    if digest is None:
        raise ValueError(f"image {image!r} is not on {docker_host or 'this daemon'}: pull or build it first")
    click.echo(digest)


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


@world.command("validate")
@click.argument("world_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--gold", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="JSON list of actions (the oracle trajectory)")
@click.option("--forbidden", "forbidden_file", type=click.File("r"), default=None,
              help="a file (or - for stdin) with one string per line that must never reach the policy; never given on argv")
@click.option("--instruction", default=None, help="the policy's instruction (checked for fix pointers)")
@click.option("--pack-root", type=click.Path(exists=True, file_okay=False, path_type=Path), help="where the spec's relative seeds live")
@click.option("--out", type=click.Path(path_type=Path))
def validate(world_yaml: Path, gold: Path | None, forbidden_file: Any, instruction: str | None, pack_root: Path | None, out: Path | None) -> None:
    """Audit a World (OpenEnv RFC 008 check ids): reset / state contract, tools, leakage, determinism, containment. The
    forbidden strings (hidden answers) come from a file or stdin, never argv (they showed in ps and shell history), and
    the report names them by index, never by value (M15 audit H2 V18)."""
    import asyncio

    from pydantic import TypeAdapter

    from crucible.core.schema.actions import Action
    from crucible.core.schema.world import WorldSpec
    from crucible.world.audit import audit_world
    from crucible.world.world import World

    forbidden = [x.rstrip("\n") for x in forbidden_file.read().splitlines() if x.strip()] if forbidden_file else []
    spec = WorldSpec.load(world_yaml)
    actions = TypeAdapter(list[Action]).validate_json(gold.read_text()) if gold else None

    if pack_root is None:
        # a pack's Worlds live under ``worlds/`` and name seeds from the pack root: the nearest directory with a pack.yaml
        # (smoke 2026-10-29: the yaml's own directory made ``world validate packs/_toy/worlds/helpdesk.yaml`` fail on its
        # bundle, data/helpdesk); a World outside any pack keeps its own directory
        here = world_yaml.resolve().parent
        pack_root = next((d for d in (here, *here.parents) if (d / "pack.yaml").is_file()), here)

    async def run():
        w = World(spec, pack_root=pack_root)
        if instruction:
            w.instructions = {spec.policy_actors[0].id: instruction}
        await w.start()
        try:
            return await audit_world(w, gold=actions, forbidden=list(forbidden))
        finally:
            await w.close()

    rep = asyncio.run(run())
    for c in rep.checks:
        click.echo(f"  {c.status.upper():4s} {c.severity:8s} {c.id}" + (f"  {json.dumps(_mask(c.detail, forbidden))[:160]}" if c.status != "pass" else ""))
    click.echo(rep.summary())
    if out:
        out.write_text(json.dumps(_mask(rep.to_dict(), forbidden), indent=1))
    raise SystemExit(0 if rep.ok else 1)


def _mask(obj: Any, forbidden: list[str]) -> Any:
    """Every forbidden string in a report replaced by its index (``<forbidden#2>``)."""
    if isinstance(obj, str):
        for i, f in sorted(enumerate(forbidden), key=lambda x: -len(x[1])):
            if f:
                obj = obj.replace(f, f"<forbidden#{i}>")
        return obj
    if isinstance(obj, dict):
        return {_mask(k, forbidden): _mask(v, forbidden) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_mask(v, forbidden) for v in obj]
    return obj


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
