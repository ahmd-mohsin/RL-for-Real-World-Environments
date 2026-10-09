"""Crucible domain pack: coding.

MVP family ``legacy_gpu_migration`` (DESIGN §12): tasks 01-04 are imported as seed overlays under
``families/legacy_gpu_migration/seeds/<task>/crucible/`` that point at the untouched Harbor task
directories in ``migration_tasks/``.
"""

from __future__ import annotations

from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[2]
FAMILY_ROOT = PACK_ROOT / "families" / "legacy_gpu_migration"
SEEDS_ROOT = FAMILY_ROOT / "seeds"


def register(registry) -> None:
    registry.register("pack", "coding", "crucible_pack_coding:PACK_ROOT", source="crucible-pack-coding")
    registry.register("app_template", "coding/git_repo", "crucible.world.apps.workspace:GitRepoApp", source="crucible-pack-coding")
    registry.register("app_template", "coding/gpu_toolchain", "crucible.world.apps.toolchain:ToolchainApp", source="crucible-pack-coding")
    registry.register("connector", "github_pr", "crucible_pack_coding.connectors.github_pr:GitHubPRConnector", source="crucible-pack-coding")
    registry.register("signal_mapper", "coding/pull_request", "crucible_pack_coding.signals:CodingSignalMapper", source="crucible-pack-coding")
    registry.register("outcome_labeler", "coding/pull_request", "crucible_pack_coding.outcomes:PullRequestOutcome", source="crucible-pack-coding")


def seed_dirs() -> list[Path]:
    return sorted(p for p in SEEDS_ROOT.iterdir() if (p / "crucible" / "task.yaml").is_file())
