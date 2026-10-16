"""Media types are canonical and agree with the media's kind (M0 audit 2026-10-14 R-26)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crucible.actors.policy import media_parts
from crucible.core.schema.actions import ImageRef, MediaRef


@pytest.mark.parametrize("kind, given, canonical", [
    ("audio", "audio/wave", "audio/wav"), ("audio", "audio/mp3", "audio/mpeg"), ("audio", "Audio/X-M4A", "audio/mp4"),
    ("image", "image/jpg", "image/jpeg"), ("image", "image/png; charset=binary", "image/png"), ("image", "image/*", "image/*"),
])
def test_aliases_and_spellings_are_normalized(kind, given, canonical):
    assert MediaRef(kind=kind, mime=given, url="https://x/y").mime == canonical


@pytest.mark.parametrize("kind, mime", [("image", "audio/wav"), ("audio", "image/png"), ("image", "png"), ("audio", "audio/")])
def test_a_type_that_disagrees_with_the_kind_is_refused(kind, mime):
    with pytest.raises(ValidationError, match="is not an"):
        MediaRef(kind=kind, mime=mime, url="https://x/y")
    with pytest.raises(ValidationError):
        ImageRef(mime="audio/wav", ref="sha256:" + "0" * 64)


def test_audio_goes_out_in_the_format_providers_name():
    """audio/wave was sent as the input_audio format 'wave', which providers reject."""
    parts = media_parts("hi", [MediaRef(kind="audio", mime="audio/wave", ref="sha256:" + "0" * 64, data_b64="AAAA"),
                                  MediaRef(kind="audio", mime="audio/mp3", ref="sha256:" + "1" * 64, data_b64="AAAA")], vision=False, audio=True)
    assert [p["input_audio"]["format"] for p in parts if p["type"] == "input_audio"] == ["wav", "mp3"]
