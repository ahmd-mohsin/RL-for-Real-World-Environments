"""Open risks from the async-RL audit (2026-10-28) on the DeltaAI bridge. Each test fails on the code before its fix.

- R5: MoE routing replay of a sequence whose calls span a weight swap took the prefix routes of the first version for the
  later version's tokens; such a sequence is now refused (the step recomputes the router), never replayed inexactly.
- Test isolation: ``sc_head.install`` scanned ``sys.modules`` with ``getattr``, so a lazy module (transformers', present in
  any process that imported it) imported a submodule with a missing optional dependency and raised; the score-centering
  unit test failed only in full-suite order."""

from __future__ import annotations

import base64
import sys
import types

import numpy as np
import pytest

pytest.importorskip("torch")

from crucible.train.bridges.primerl import wire
from crucible.train.bridges.primerl.routing import RoutingStore, decode
from crucible.train.bridges.tinker.backend import TokenDatum


def _field(rows, start):
    a = np.asarray(rows, dtype=np.uint8)
    return {"data": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": "uint8", "start": start}


def _datum(v1: int, v2: int) -> TokenDatum:
    """One sequence: prompt 0-3, call 1 samples 4-6, an observation at 7, call 2 samples 8-9 (``full`` = 0..9)."""
    full = list(range(10))
    sampled = {4, 5, 6, 8, 9}
    mask = [int(j + 1 in sampled) for j in range(9)]
    return TokenDatum(input_ids=full[:-1], target_ids=full[1:], logprobs=[-0.5 * m for m in mask], advantages=[1.0 * m for m in mask], mask=mask,
                      meta={"rollout": "r", "segment": 0, "prompt_len": 4, "call_spans": [["c1", 0, 3], ["c2", 4, 6]],
                            "min_version": min(v1, v2), "max_version": max(v1, v2)})


def _store() -> RoutingStore:
    store = RoutingStore()
    full = list(range(10))
    store.put(full[:7], decode(_field([[[1, 2]]] * 6, start=0)))  # call 1 (version 1): positions 0..5
    # call 2 after a hot-load: the prefix cache is keyed by adapter, so it re-forwards the whole prefix under version 2,
    # with other experts for positions 0..5 than call 1 chose
    store.put(full[:10], decode(_field([[[7, 8]]] * 9, start=0)))
    return store


def test_a_sequence_spanning_two_policy_versions_is_never_replayed_with_the_first_versions_prefix() -> None:
    row = wire.to_sample(_datum(1, 2), temperature=1.0, routing=_store())
    # before the fix: coverage 1 and call 2's tokens (8, 9) trained over positions 0..5 routed as version 1 routed them
    assert row["routed_experts"] is None and row["routing_coverage"] == 0.0 and row["routing_version_mixed"] is True
    assert wire.replay_all_or_none([row]) == [None]  # the step recomputes the router, as for any coverage gap
    assert wire.version_mixed_share([row]) == 1.0


def test_a_single_version_sequence_is_still_replayed_exactly() -> None:
    row = wire.to_sample(_datum(2, 2), temperature=1.0, routing=_store())
    assert row["routing_coverage"] == 1.0 and row["routing_version_mixed"] is False and row["routed_experts"]["shape"] == [10, 1, 2]
    assert wire.version_mixed_share([row]) == 0.0


def test_samples_warns_once_per_step_for_version_mixed_sequences(caplog) -> None:
    with caplog.at_level("WARNING", logger="crucible.train.bridges.primerl.wire"):
        rows = wire.samples([_datum(1, 2), _datum(2, 2)], temperature=1.0, routing=_store())
    assert [r["routing_version_mixed"] for r in rows] == [True, False]
    assert sum("span more than one policy version" in r.message for r in caplog.records) == 1


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


class _Lazy(types.ModuleType):
    """transformers' ``_LazyModule``: an attribute it does not hold imports a submodule, which can fail."""

    def __getattr__(self, name):
        raise ModuleNotFoundError("No module named 'torchvision'")


def test_sc_head_install_is_not_broken_by_a_lazy_module_in_sys_modules(monkeypatch) -> None:
    from crucible.train.bridges.primerl import sc_head

    class Fused:
        def forward(self, hidden_states, labels=None, temperature=None, sampling_mask=None):
            return {}

    class Vanilla(Fused):
        pass

    def compute_loss(**kw):
        return "loss", {}

    lmh = _module("prime_rl.trainer.models.layers.lm_head", FusedOutputLinear=Fused, VanillaOutputLinear=Vanilla, PrimeLmOutput=dict)
    prl_loss = _module("prime_rl.trainer.rl.loss", compute_loss=compute_loss)
    train = _module("prime_rl.trainer.rl.train", compute_loss=compute_loss, gather_for_cp=lambda t, g: t, setup_rl_loss_fn=lambda c: None)
    for name, mod in (("prime_rl", _module("prime_rl")), ("prime_rl.trainer", _module("prime_rl.trainer")),
                      ("prime_rl.trainer.models", _module("prime_rl.trainer.models")),
                      ("prime_rl.trainer.models.layers", _module("prime_rl.trainer.models.layers", lm_head=lmh)),
                      ("prime_rl.trainer.models.layers.lm_head", lmh), ("prime_rl.trainer.rl", _module("prime_rl.trainer.rl", loss=prl_loss)),
                      ("prime_rl.trainer.rl.loss", prl_loss), ("prime_rl.trainer.rl.train", train),
                      ("crucible_test_lazy_module", _Lazy("crucible_test_lazy_module"))):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sc_head.CTX, "installed", False)
    sc_head.install()  # before the fix: ModuleNotFoundError out of getattr(<lazy module>, "compute_loss", None)
    assert sc_head.CTX.installed and getattr(train.compute_loss, "__crucible_sc__", False)
