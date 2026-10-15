"""Regression tests for the M0 audit (docs/audit/2026-10-12/M00-core.md, defects D1-D11).

Each test reproduces one reproduced defect; property tests cover every core model, not only the ones the audit
happened to find.
"""

from __future__ import annotations

import importlib
import inspect
import math
import pkgutil
import shutil
import threading
import typing

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import crucible.core.schema as schema_pkg
from crucible.core.config import Config, LLMProvider
from crucible.core.registry import Registry, RegistryError
from crucible.core.schema.base import CrucibleModel, VersionedDocument
from crucible.core.schema.milestone import MilestoneSet
from crucible.core.schema.reward import RewardJson, StepRow
from crucible.core.schema.run import RunRecord
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store, StoreError


def _all_models() -> list[type[CrucibleModel]]:
    out = []
    for info in pkgutil.iter_modules(schema_pkg.__path__):
        mod = importlib.import_module(f"crucible.core.schema.{info.name}")
        for obj in vars(mod).values():
            if inspect.isclass(obj) and issubclass(obj, CrucibleModel) and obj.__module__ == mod.__name__:
                out.append(obj)
    return out


BASE_WORLD = {
    "world": "t/w",
    "apps": [{"id": "db", "template": "common/sqlite_app", "state": {"adapter": "sql"}}],
    "actors": [{"id": "agent", "kind": "policy", "tools": ["db.*"]}],
}


# ------------------------------------------------------------------------------------------------ D1: lossless None
def test_no_model_has_a_nullable_field_with_a_non_null_default():
    """An explicit None must survive canonical JSON; exclude_none drops it, so such fields may not exist."""
    offenders = []
    for cls in _all_models():
        for name, f in cls.model_fields.items():
            nullable = type(None) in typing.get_args(f.annotation)
            if nullable and not f.is_required() and f.default_factory is None and f.default is not None:
                offenders.append(f"{cls.__name__}.{name}")
    assert offenders == []


def test_world_round_trip_keeps_value_and_hash():
    w = WorldSpec.model_validate({**BASE_WORLD, "isolation": {"pids_limit": 0}})
    back = WorldSpec.model_validate_json(w.canonical_json())
    assert back.isolation.pids_limit == w.isolation.pids_limit
    assert back.content_hash() == w.content_hash()


# ------------------------------------------------------------------------------------------------ D2: finite floats
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(bad):
    with pytest.raises(ValueError):
        RewardJson(reward=bad)


def test_step_phi_is_a_potential_in_unit_interval():
    with pytest.raises(ValueError):
        StepRow(step=0, phi=3.0, delta_phi=0.0)


@settings(max_examples=200, deadline=None)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
def test_canonical_json_is_strict_json_and_stable(x):
    r = RewardJson(reward=x)
    text = r.canonical_json()
    import json

    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert RewardJson.model_validate_json(text).content_hash() == r.content_hash()


# ------------------------------------------------------------------------------------------------ D4: store races
def test_put_file_never_stores_bytes_that_mismatch_their_key(tmp_path, monkeypatch):
    st_ = Store(tmp_path / "s")
    src = tmp_path / "live.bin"
    src.write_bytes(b"A" * 1_000_000)
    orig = shutil.copyfile

    def racing_copy(a, b, *args, **kw):  # the source changes while the store reads it
        src.write_bytes(b"B" * 1_000_000)
        return orig(a, b, *args, **kw)

    monkeypatch.setattr(shutil, "copyfile", racing_copy)
    key = st_.put_file(src)
    monkeypatch.setattr(shutil, "copyfile", orig)
    st_.get_bytes(key)  # raises StoreError if the blob does not match its key


def test_concurrent_put_file_of_one_new_file(tmp_path):
    st_ = Store(tmp_path / "s")
    src = tmp_path / "x.bin"
    src.write_bytes(bytes(range(256)) * 40_000)
    errors: list[BaseException] = []

    def put():
        try:
            st_.put_file(src)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=put) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []


def test_corrupt_blob_is_detected(tmp_path):
    st_ = Store(tmp_path / "s")
    key = st_.put_bytes(b"hello")
    p = st_.blob_path(key)
    p.chmod(0o644)
    p.write_bytes(b"hellO")
    with pytest.raises(StoreError):
        st_.get_bytes(key)


# ------------------------------------------------------------------------------------------------ D5: spec validation
@pytest.mark.parametrize(
    "extra",
    [
        {"network": "allowlist"},
        {"network": "no-network", "allowed_hosts": ["api.example.com"]},
        {"clock": {"step_dt_s": -60}},
        {"clock": {"t0": "yesterday"}},
        {"limits": {"tool_timeout_s": -1}},
        {"limits": {"max_tool_output_chars": -5}},
        {"limits": {"tool_quota": {"db.*": -3}}},
        {"clock": {}, "events": [{"id": "e", "when": {"after": "1h"}, "actor": "agent", "action": {"type": "tool", "tool": "ghost.run"}}]},
        {"clock": {}, "events": [{"id": "e", "when": {"after": "1h"}, "actor": "agent", "action": {"type": "permission", "grant": ["ghost.*"]}}]},
    ],
)
def test_world_spec_rejects_mistakes(extra):
    with pytest.raises(ValueError):
        WorldSpec.model_validate({**BASE_WORLD, **extra})


# ------------------------------------------------------------------------------------------------ D6: redaction
def test_redacted_config_never_contains_raw_secrets():
    c = Config(llm_providers={"x": LLMProvider(base_url="http://h", api_key="sk-REALSECRET123", headers={"Authorization": "Bearer tok-REAL"})})
    text = str(c.redacted())
    assert "sk-REALSECRET123" not in text and "tok-REAL" not in text
    ref = Config(llm_providers={"x": LLMProvider(base_url="http://h", api_key="env:MY_KEY")})
    assert "env:MY_KEY" in str(ref.redacted())


# ------------------------------------------------------------------------------------------------ D7: plugin errors
def test_registry_reports_why_a_plugin_is_missing(monkeypatch):
    class EP:
        name = "broken_pack"

        def load(self):
            raise ImportError("No module named 'crucible_pack_broken'")

    monkeypatch.setattr("crucible.core.registry.entry_points", lambda group: [EP()])
    r = Registry()
    with pytest.raises(RegistryError, match="broken_pack"):
        r.get("pack", "broken")


# ------------------------------------------------------------------------------------------------ D8: versions
@pytest.mark.parametrize("cls", [RewardJson, RunRecord])
def test_persisted_formats_are_versioned(cls):
    assert "schema_version" in cls.model_fields


def test_unversioned_reward_document_still_loads_as_v1():
    r = RewardJson.model_validate({"reward": 1.0})
    assert r.schema_version.endswith("/v1")


# ------------------------------------------------------------------------------------------------ D9: version bypass
def test_schema_version_cannot_be_reassigned_to_unknown_major():
    ms = MilestoneSet(family="f", milestones=[])
    with pytest.raises(ValueError):
        ms.schema_version = "crucible.milestones/v9"


# ------------------------------------------------------------------------------------------------ D10: names
def test_store_lists_a_document_under_every_name(tmp_path):
    st_ = Store(tmp_path / "s")
    doc = RewardJson(reward=1.0)
    st_.put_doc("reward", doc, name="a")
    st_.put_doc("reward", doc, name="b")
    assert len(st_.list_docs("reward", name="a")) == 1
    assert len(st_.list_docs("reward", name="b")) == 1


# ------------------------------------------------------------------------------------------------ D11: run states
def test_run_record_rejects_contradictions():
    with pytest.raises(ValueError):
        RunRecord(run_id="r", world="w", policy="p", started_at="2026-01-01T00:00:00Z", outcome="completed", mask_sample=True, reward=0.7)
    with pytest.raises(ValueError):
        RunRecord(run_id="r", world="w", policy="p", started_at="not-a-time")


# ------------------------------------------------------------------------------------------------ every model
@pytest.mark.parametrize("cls", [c for c in _all_models() if issubclass(c, VersionedDocument) and c is not VersionedDocument], ids=lambda c: c.__name__)
def test_versioned_documents_declare_their_schema(cls):
    assert cls.SCHEMA.startswith("crucible.") and "/v" in cls.SCHEMA


# ------------------------------------------------------------------------------------------------ D12: normalization
def test_negative_zero_and_positive_zero_hash_alike():
    assert RewardJson(reward=0.0).content_hash() == RewardJson(reward=-0.0).content_hash()


def test_unicode_normalization_forms_hash_alike():
    from crucible.core.schema.milestone import Milestone, MilestoneCheck
    from crucible.core.schema.state import Query

    def ms(name: str) -> str:
        return MilestoneSet(family=name, milestones=[Milestone(id="m", check=MilestoneCheck(app="a", query=Query(op="exists")))]).content_hash()

    assert ms("café") == ms("café")


# ------------------------------------------------------------------------------------------------ D13: hash drift
GOLDEN = {
    # pinned content hashes: a schema change that alters an existing id must update these on purpose (scheme bump)
    "reward_one": ("RewardJson", {"reward": 1.0}),
    "world_min": ("WorldSpec", BASE_WORLD),
}


def test_hash_scheme_is_declared():
    from crucible.core.ids import HASH_SCHEME

    assert isinstance(HASH_SCHEME, int) and HASH_SCHEME >= 1


def test_golden_content_hashes_do_not_drift():
    import json
    from pathlib import Path

    from crucible.core.schema import reward as reward_mod
    from crucible.core.schema import world as world_mod

    classes = {"RewardJson": reward_mod.RewardJson, "WorldSpec": world_mod.WorldSpec}
    pin = Path(__file__).with_name("golden_hashes.json")
    got = {k: classes[c].model_validate(v).content_hash() for k, (c, v) in GOLDEN.items()}
    want = json.loads(pin.read_text())
    assert got == want, "content hashes drifted: bump HASH_SCHEME and regenerate tests/unit/golden_hashes.json"


# ------------------------------------------------------------------------------------------------ D15: store hygiene
def test_store_uses_wal_and_sweeps_stale_temp_files(tmp_path):
    root = tmp_path / "s"
    st_ = Store(root)
    assert st_._db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    stale = root / "blobs" / "ab" / "tmpdeadwriter.tmp"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"partial")
    import os
    import time

    old = time.time() - 7200
    os.utime(stale, (old, old))
    st_.close()
    Store(root)
    assert not stale.exists()


def test_put_repairs_a_corrupt_blob(tmp_path):
    st_ = Store(tmp_path / "s")
    key = st_.put_bytes(b"payload")
    p = st_.blob_path(key)
    p.chmod(0o644)
    p.write_bytes(b"trunc")
    assert st_.put_bytes(b"payload") == key
    assert st_.get_bytes(key) == b"payload"


# ------------------------------------------------------------------------------------------------ D16: ids
@pytest.mark.parametrize("bad", ["../escape", "a/../b", ".", "..", "x\u200by", "", "a" * 300])
def test_ids_used_as_path_segments_are_validated(bad):
    from crucible.core.ids import validate_id

    with pytest.raises(ValueError):
        validate_id(bad)


def test_valid_ids_pass():
    from crucible.core.ids import validate_id

    for ok in ["toy/helpdesk", "task04@v3", "finance_ap-invoices.s1", "café"]:
        assert validate_id(ok)


# ------------------------------------------------------------------------------------------------ R-4: ids as paths
@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "x\0y", "", "café"])
def test_child_path_refuses_unsafe_names(tmp_path, bad):
    from crucible.core.ids import child_path

    with pytest.raises(ValueError):
        child_path(tmp_path, bad)


def test_child_path_refuses_a_symlink_out_of_root(tmp_path):
    from crucible.core.ids import child_path

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "link").symlink_to(tmp_path)
    assert child_path(tmp_path / "runs", "ok-run", "a1") == tmp_path / "runs" / "ok-run" / "a1"
    with pytest.raises(ValueError, match="outside"):
        child_path(tmp_path / "runs", "link")


def test_run_journal_refuses_a_run_id_that_escapes_the_root(tmp_path):
    from crucible.runtime.journal import RunJournal

    with pytest.raises(ValueError):
        RunJournal(tmp_path / "runs", "../escaped", "identity", {})
    assert not (tmp_path / "escaped").exists()
    assert RunJournal(tmp_path / "runs", "bench-toy-bench", "identity", {}).dir == tmp_path / "runs" / "bench-toy-bench"


# ------------------------------------------------------------------------------------------------ D17: pack kinds
def test_packs_can_register_namespaced_failure_kinds():
    from crucible.core.schema.run import Failure, register_failure_kind

    register_failure_kind("finance_ap:erp_down", blame="infra")
    f = Failure(kind="finance_ap:erp_down", stage="agent")
    r = RunRecord(run_id="r", world="w", policy="p", started_at="2026-01-01T00:00:00Z")
    r.mark_failure(f)
    assert r.mask_sample and r.scored_reward() is None
    with pytest.raises(ValueError):
        register_failure_kind("no_namespace", blame="infra")


def test_failure_reason_is_bounded():
    from crucible.core.schema.run import Failure

    assert len(Failure(kind="agent_timeout", stage="agent", reason="x" * 10_000).reason) <= 2000


# ------------------------------------------------------------------------------------------------ R-5: secrets by value
def test_a_resolved_secret_with_no_vendor_shape_is_scrubbed(tmp_path, monkeypatch):
    from crucible.core.config import resolve_secret
    from crucible.core.schema.run import Failure
    from crucible.core.secrets import scrub_json, scrub_secrets

    key = "gw7Kq2PzR8xLm4Tn9Qa-r5test"  # a gateway key no pattern knows
    assert scrub_secrets(f"invalid key {key}") == f"invalid key {key}"  # before it is resolved
    monkeypatch.setenv("R5_GATEWAY_KEY", key)
    assert resolve_secret("env:R5_GATEWAY_KEY", search_from=tmp_path) == key
    assert scrub_secrets(f"401 Unauthorized: invalid key {key}") == "401 Unauthorized: invalid key <secret>"
    rec = Failure(kind="agent_run_error", stage="agent", reason=f"provider said {key}").model_dump_json()
    assert key not in scrub_json(rec)


def test_short_values_are_not_registered_as_secrets():
    from crucible.core.secrets import register_secret, scrub_secrets

    register_secret("abc1234")  # 7 characters: would mask ordinary words
    assert scrub_secrets("abc1234 stays") == "abc1234 stays"


# ------------------------------------------------------------------------------------------------ R-6: private stores
def test_a_new_store_is_private_to_its_owner(tmp_path):
    import os
    import stat

    old = os.umask(0o022)
    try:
        st_ = Store(tmp_path / "store")
        key = st_.put_bytes(b"trace with customer data")
    finally:
        os.umask(old)
    mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)
    assert mode(tmp_path / "store") == 0o700
    assert mode(tmp_path / "store" / "index.sqlite") == 0o600
    assert (tmp_path / "store" / ".gitignore").read_text() == "*\n"
    assert st_.get_bytes(key) == b"trace with customer data"


def test_an_existing_store_root_keeps_its_mode(tmp_path):
    import os
    import stat

    (tmp_path / "shared").mkdir(mode=0o755)
    os.chmod(tmp_path / "shared", 0o755)
    Store(tmp_path / "shared")
    assert stat.S_IMODE(os.stat(tmp_path / "shared").st_mode) == 0o755
    assert not (tmp_path / "shared" / ".gitignore").exists()
