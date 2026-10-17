"""How a rollout's tokens were produced: engine, log-prob mode, tokenizer identity (M0 audit 2026-10-14 R-17)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from crucible.core.schema.rollout import Capture, RolloutRecord, TokenizerId


class _Tok:
    """A Hugging Face-like tokenizer: vocabulary, special tokens, chat template, fast backend."""

    def __init__(self, vocab: dict[str, int], template: str, backend: str = '{"normalizer": null}'):
        self.vocab, self.chat_template, self.special_tokens_map, self.name_or_path = vocab, template, {"eos_token": "</s>"}, "toy"
        self.backend_tokenizer = NS(to_str=lambda: backend)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)


VOCAB = {"</s>": 0, "a": 1, "b": 2}


def test_two_tokenizers_with_one_vocabulary_differ_in_render_only():
    """The audit's probe p_tb02: Crucible's vocab signature called these equal and nothing recorded the template."""
    a, b = TokenizerId.of(_Tok(VOCAB, "{{ m }}")), TokenizerId.of(_Tok(VOCAB, "<|im_start|>{{ m }}"))
    assert a.token_map == b.token_map and a.render != b.render and a.vocab_size == 3
    assert TokenizerId.of(_Tok(VOCAB, "{{ m }}", backend='{"normalizer": "NFKC"}')).render != a.render  # a normalizer change
    assert TokenizerId.of(_Tok({**VOCAB, "c": 3}, "{{ m }}")).token_map != a.token_map
    assert TokenizerId.of(_Tok(dict(reversed(list(VOCAB.items()))), "{{ m }}")) == a  # dict order is not identity


def _rec(rid: str, capture: Capture | None, temperature: float = 1.0, top_p: float | None = None) -> RolloutRecord:
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h@1", policy="p", reward=1.0, capture=capture,
                         turns=[{"call_id": "c1", "actor": "agent", "prompt_token_ids": [1], "completion_token_ids": [2], "logprobs": [-0.1],
                                 "policy_version": 0, "sampling": {"temperature": temperature, **({"top_p": top_p} if top_p else {})}}])


def test_the_trainer_refuses_mixed_tokenizers_and_raw_logprobs_it_cannot_compare():
    pytest.importorskip("torch")
    from crucible.train.engine import TinyPolicy
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    a, b = TokenizerId(token_map="sha256:aa"), TokenizerId(token_map="sha256:bb")
    tr = TorchTrainer(TinyPolicy(8, 8, seed=0), TrainerConfig(temperature=1.0))
    tr._check_capture([_rec("r1", Capture(tokenizer=a)), _rec("r2", None), _rec("r3", Capture(tokenizer=a, logprobs="raw"))])  # raw at T=1: fine
    with pytest.raises(ValueError, match="different tokenizers"):
        tr._check_capture([_rec("r1", Capture(tokenizer=a)), _rec("r2", Capture(tokenizer=b))])
    tr.tokenizer_id = b
    with pytest.raises(ValueError, match="the trainer"):
        tr._check_capture([_rec("r1", Capture(tokenizer=a))])
    with pytest.raises(ValueError, match="top-p"):
        tr._check_capture([_rec("r1", Capture(logprobs="raw"), top_p=0.9)])
    hot = TorchTrainer(TinyPolicy(8, 8, seed=0), TrainerConfig(temperature=0.7))
    with pytest.raises(ValueError, match="raw log-probs"):
        hot._check_capture([_rec("r1", Capture(logprobs="raw"), temperature=0.7)])
    hot._check_capture([_rec("r1", Capture(logprobs="processed"), temperature=0.7)])


async def test_the_runtime_records_the_engine_and_its_logprob_mode(tmp_path):
    pytest.importorskip("torch")
    from tests.unit.test_train_state import _rt

    rt, _, tr = _rt("colocated_sync", 1)
    tr.tokenizer_id = TokenizerId(token_map="sha256:cc")
    rep = await rt.run()
    caps = {r.capture.model_dump_json() for r in rep.trained_records}
    assert caps == {Capture(engine="TorchEngine", logprobs="processed", tokenizer=tr.tokenizer_id).model_dump_json()}


def test_a_chat_codec_records_its_tokenizer_and_template_arguments():
    pytest.importorskip("torch")  # train.world_env imports the engine (M0 audit 2026-10-16: this failed without torch)
    from crucible.train.world_env import codec_capture

    cap = codec_capture(NS(tok=_Tok(VOCAB, "{{ m }}"), kw={"enable_thinking": False}))
    assert cap.tokenizer == TokenizerId.of(_Tok(VOCAB, "{{ m }}")) and cap.template_kwargs == {"enable_thinking": False}
    assert codec_capture(NS()) is None  # a scripted codec: nothing to say


def test_a_record_without_capture_reads_and_keeps_its_bytes():
    old = RolloutRecord.model_validate({"schema": "crucible.rollout/v1", "rollout_id": "r", "sample": "s", "task": "t", "harness": "h@1", "policy": "p"})
    assert old.capture is None and "capture" not in old.model_dump(by_alias=True)  # a record that says nothing keeps its bytes
