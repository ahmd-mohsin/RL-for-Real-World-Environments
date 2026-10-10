"""CLI smoke tests (click runner)."""

from click.testing import CliRunner
from crucible_pack_coding import seed_dirs

from crucible.cli.main import cli


def test_cli_registry_seeds_and_task_show():
    import pytest

    from crucible.interop.harbor.task_io import load_crucible_task

    try:
        dirs = seed_dirs()
        for d in dirs:  # the overlays ship with the pack; their Harbor side (../migration_tasks) may not be staged
            load_crucible_task(d)
    except (FileNotFoundError, OSError):
        dirs = []
    if len(dirs) < 5:
        pytest.skip("needs the seed tasks (../migration_tasks), not shipped with the package")
    r = CliRunner().invoke(cli, ["registry"])
    assert r.exit_code == 0 and "world_backend" in r.output and "agentenv" in r.output
    r = CliRunner().invoke(cli, ["seeds"])
    assert r.exit_code == 0 and r.output.count("legacy_gpu_migration") == 5
    r = CliRunner().invoke(cli, ["task", "show", str(seed_dirs()[-1])])
    assert r.exit_code == 0 and "task04" in r.output and "gpt-6-astra" in r.output and '"ignored_keys"' in r.output
