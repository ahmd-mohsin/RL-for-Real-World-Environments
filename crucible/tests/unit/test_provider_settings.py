"""Provider settings carry no credentials (M0 audit 2026-10-16 R-2): ``request_defaults`` is merged into every request
and written into run locks, so a header or key there bypassed the ``env:`` rule for ``headers`` and the scrubber."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crucible.core.config import LLMProvider


@pytest.mark.parametrize("settings, field", [
    ({"request_defaults": {"extra_headers": {"Authorization": "Bearer attacker"}}}, "extra_headers"),  # replaced the key
    ({"request_defaults": {"extra_headers": {"X-Gateway-Token": "gw4f9a2c7e1b9d3a"}}}, "extra_headers"),  # reached lock.json
    ({"request_defaults": {"extra_query": {"key": "abc"}}}, "extra_query"),
    ({"request_defaults": {"extra_body": {"api_key": "sk-1"}}}, "extra_body.api_key"),  # reached every Turn.sampling.extra
    ({"request_defaults": {"extra_body": {"auth_token": 12345678}}}, "extra_body.auth_token"),
    ({"request_defaults": {"stream": True}}, "stream"),
    ({"template_kwargs": {"hf_token": "hf_x"}}, "hf_token"),
    ({"base_url": "https://Zq81mXv40LpQ7tR2@gateway.example/v1"}, "base_url carries credentials"),
    ({"base_url": "https://user:pw@gateway.example/v1"}, "base_url carries credentials"),
])
def test_provider_settings_refuse_credentials(settings, field):
    with pytest.raises(ValidationError, match=field.replace(".", r"\.")) as e:
        LLMProvider.model_validate({"base_url": "http://127.0.0.1:8000/v1", **settings})
    assert "attacker" not in str(e.value).split("input_value")[0]  # the message names the setting, not its value


def test_engine_settings_stay_allowed():
    """Token capture, sampling and length settings are not credentials, whatever their names hold."""
    p = LLMProvider(base_url="http://127.0.0.1:8000/v1", request_defaults={
        "logprobs": True, "top_p": 0.9, "max_tokens": 512, "max_completion_tokens": 256,
        "extra_body": {"return_token_ids": True, "skip_special_tokens": False}}, template_kwargs={"enable_thinking": False})
    assert p.request_defaults["max_tokens"] == 512
