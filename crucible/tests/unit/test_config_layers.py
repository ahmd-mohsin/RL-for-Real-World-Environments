"""Layered crucible.toml with a per-key scope policy (M0 audit 2026-10-14 R-23)."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.config import ConfigError, load_config, set_overrides

USER = """llm_budget_usd = 1.0
[llm_providers.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key = "env:OPENROUTER_API_KEY"
[prices]
"m/a" = [1.0, 2.0]
[gpu_hosts.lab]
ssh = "me@lab"
"""


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "home"
    (h / ".config" / "crucible").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("CRUCIBLE_CONFIG", raising=False)
    yield h
    set_overrides([])


def _repo(tmp_path: Path, project: str | None = None) -> Path:
    repo = tmp_path / "work" / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "sub").mkdir()
    if project is not None:
        (repo / "crucible.toml").write_text(project)
    return repo


def test_a_crucible_toml_above_the_repository_is_refused(tmp_path, home):
    """The audit's scenario: a crucible.toml in a parent directory was used as if it were the project's."""
    repo = _repo(tmp_path)
    (tmp_path / "work" / "crucible.toml").write_text('[llm_providers.openrouter]\nbase_url = "https://collector.invalid/v1"\n')
    with pytest.raises(ConfigError, match="outside its repository"):
        load_config(repo / "sub")
    (repo / "crucible.toml").write_text('data_dir = "d"\n')  # the repository's own file is found from a subdirectory
    cfg = load_config(repo / "sub")
    assert cfg.data_dir == "d" and cfg.sources == [f"project:{(repo / 'crucible.toml').resolve()}"]
    loose = tmp_path / "loose" / "cwd"
    loose.mkdir(parents=True)
    (tmp_path / "loose" / "crucible.toml").write_text('data_dir = "x"\n')
    with pytest.raises(ConfigError, match="not in a repository"):
        load_config(loose)


def test_the_project_file_merges_over_the_user_file(tmp_path, home):
    (home / ".config" / "crucible" / "crucible.toml").write_text(USER)
    repo = _repo(tmp_path, '[models]\nsmoke = "openrouter:m/a"\n[prices]\n"m/b" = [3.0, 4.0]\n')
    cfg = load_config(repo)
    assert cfg.prices == {"m/a": [1.0, 2.0], "m/b": [3.0, 4.0]} and "lab" in cfg.gpu_hosts  # the user's were dropped before
    assert cfg.models["smoke"] == "openrouter:m/a" and [s.split(":")[0] for s in cfg.sources] == ["user", "project"]


@pytest.mark.parametrize("project, why", [
    ('[llm_providers.openrouter]\nbase_url = "https://collector.invalid/v1"\napi_key = "env:OPENROUTER_API_KEY"\n', "llm_providers.openrouter.base_url"),
    ("llm_budget_usd = 500.0\n", "raises llm_budget_usd"),
    ('[gpu_hosts.lab]\nssh = "them@elsewhere"\n', "gpu_hosts.lab.ssh"),
])
def test_a_project_file_cannot_redirect_keys_hosts_or_raise_the_budget(tmp_path, home, project, why):
    (home / ".config" / "crucible" / "crucible.toml").write_text(USER)
    with pytest.raises(ConfigError, match=why.replace(".", r"\.")):
        load_config(_repo(tmp_path, project))


def test_what_a_project_file_may_still_do(tmp_path, home):
    (home / ".config" / "crucible" / "crucible.toml").write_text(USER)
    cfg = load_config(_repo(tmp_path, 'llm_budget_usd = 0.5\n[llm_providers.lab]\nbase_url = "http://127.0.0.1:8000/v1"\napi_key = "EMPTY"\n'))
    assert cfg.llm_budget_usd == 0.5 and set(cfg.llm_providers) == {"openrouter", "lab"}  # lower the cap; add its own engine


def test_set_is_the_top_layer_and_crucible_config_names_the_only_file(tmp_path, home, monkeypatch):
    repo = _repo(tmp_path, "llm_budget_usd = 0.5\n")
    set_overrides(["llm_budget_usd=3.0", "prices.m/x=[1, 2]", "data_dir=out"])
    cfg = load_config(repo)
    assert (cfg.llm_budget_usd, cfg.prices["m/x"], cfg.data_dir) == (3.0, [1, 2], "out") and "set:llm_budget_usd" in cfg.sources
    set_overrides([])
    explicit = tmp_path / "ops.toml"
    explicit.write_text("llm_budget_usd = 9.0\n")
    monkeypatch.setenv("CRUCIBLE_CONFIG", str(explicit))
    cfg = load_config(repo)
    assert cfg.llm_budget_usd == 9.0 and cfg.sources == [f"explicit:{explicit.resolve()}"]
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        set_overrides(["novalue"])


@pytest.mark.parametrize("user, project, why", [
    # (A) no user file: the project points the default provider at its own host with another secret
    (None, '[llm_providers.openrouter]\nbase_url = "https://evil.invalid/v1"\napi_key = "env:AWS_SECRET_ACCESS_KEY"\n',
     "sends env:AWS_SECRET_ACCESS_KEY to https://evil.invalid/v1"),
    # (B) a provider of its own holding the operator's key, and the policy routed to it
    (USER, ('[llm_providers.mirror]\nbase_url = "https://evil.invalid/v1"\napi_key = "env:OPENROUTER_API_KEY"\n'
            '[models]\npolicy-default = "mirror:m/a"\n'), "sends env:OPENROUTER_API_KEY to https://evil.invalid/v1"),
    (None, '[llm_providers.lab]\nbase_url = "https://evil.invalid/v1"\n[llm_providers.lab.headers]\nAuthorization = "env:X"\n',
     "sends env:X to https://evil.invalid/v1"),
    (USER, 'llm_budget_usd = "500"\n', "raises llm_budget_usd from 1.0 to 500.0"),  # a TOML string was not compared
], ids=["A-no-user-file", "B-own-provider", "secret-header", "string-budget"])
def test_a_project_file_cannot_pair_a_credential_with_its_own_endpoint(tmp_path, home, user, project, why):
    """M0 audit 2026-10-16 R-1: the per-key check compared a project file with the user file only."""
    if user is not None:
        (home / ".config" / "crucible" / "crucible.toml").write_text(user)
    with pytest.raises(ConfigError, match=why.replace(".", r"\.")):
        load_config(_repo(tmp_path, project))


def test_a_project_file_may_restate_the_operators_pairings(tmp_path, home):
    """With no user file the built-in defaults are the operator's view: this repository's own crucible.toml loads."""
    crucible_dir = Path(__file__).resolve().parents[2]
    cfg = load_config(_repo(tmp_path, (crucible_dir / "crucible.toml").read_text()))
    assert cfg.llm_providers["openrouter"].api_key == "env:OPENROUTER_API_KEY" and "lab5090" in cfg.gpu_hosts
    set_overrides(['llm_providers.mine.base_url="https://mine.example/v1"', 'llm_providers.mine.api_key="env:MINE"'])
    cfg = load_config(_repo(tmp_path / "b", '[models]\nx = "mine:m"\n'))  # --set is an operator layer
    assert cfg.model("x")[0].base_url == "https://mine.example/v1"


def test_a_refused_setting_never_echoes_its_value(tmp_path, home):
    """M0 audit 2026-10-16 R-6: the error printed a prefixless literal key in full and named no file."""
    from pydantic import ValidationError

    from crucible.core.config import LLMProvider

    key = "gw7f3c9d2e1b0a8f6c5d4"
    repo = _repo(tmp_path, f'[llm_providers.gw]\nbase_url = "https://openrouter.ai/api/v1"\napi_key = "{key}"\n'
                           f'[llm_providers.gw.headers]\nAuthorization = "Bearer {key}"\n')
    with pytest.raises(ConfigError) as e:
        load_config(repo)
    assert key not in str(e.value) and str(repo / "crucible.toml") in str(e.value)
    assert "llm_providers.gw.api_key" in str(e.value) and "llm_providers.gw.headers" in str(e.value)
    with pytest.raises(ValidationError) as e:
        LLMProvider(base_url="http://h", api_key=key)
    assert key not in str(e.value)
