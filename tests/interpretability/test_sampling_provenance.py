"""Regression tests for requested-versus-effective generation metadata."""

from types import SimpleNamespace

import torch

from interpretability.evaluation import HybridLanguageModel, TransformerLensWrapper


class _TransformerTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages[0]["role"] == "user"
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "answer" if token_ids.numel() else ""


class _TransformerModel:
    def __init__(self, *, reject_frequency_penalty=False):
        self.cfg = SimpleNamespace(n_ctx=1024, d_model=4)
        self.tokenizer = _TransformerTokenizer()
        self.reject_frequency_penalty = reject_frequency_penalty
        self.generate_calls = []

    def to_tokens(self, prompt, *, truncate, prepend_bos):
        assert prompt
        assert truncate is True
        assert prepend_bos is False
        return torch.tensor([[1, 2]])

    def generate(self, tokens, **kwargs):
        self.generate_calls.append(dict(kwargs))
        if self.reject_frequency_penalty and "freq_penalty" in kwargs:
            raise TypeError("freq_penalty is unsupported")
        return torch.cat((tokens, torch.tensor([[3]])), dim=1)


def _transformer_wrapper(*, reject_frequency_penalty=False):
    wrapper = object.__new__(TransformerLensWrapper)
    wrapper.model = _TransformerModel(
        reject_frequency_penalty=reject_frequency_penalty
    )
    wrapper.default_max_tokens = 64
    wrapper.capture_mean_pooled = False
    wrapper.hook_names = []
    wrapper._current_activations = {}
    wrapper._call_count = 0
    wrapper._last_prompt = None
    wrapper._last_sampling_config = {}
    wrapper._steering_direction = None
    return wrapper


class _Batch:
    def __init__(self):
        self.input_ids = torch.tensor([[1, 2]])

    def to(self, device):
        assert device == "cpu"
        return self


class _HybridTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages[0]["role"] == "user"
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]

    def __call__(self, prompt, **kwargs):
        assert prompt
        assert kwargs == {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": 768,
        }
        return _Batch()

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "answer" if token_ids.numel() else ""


class _HybridModel:
    def __init__(self, *, fail_sampling=False, fail_greedy=False):
        self.config = SimpleNamespace(
            max_position_embeddings=1024,
            vocab_size=100,
        )
        self.fail_sampling = fail_sampling
        self.fail_greedy = fail_greedy
        self.generate_calls = []

    def generate(self, input_ids, *, attention_mask, **kwargs):
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        self.generate_calls.append(dict(kwargs))
        if self.fail_sampling and kwargs["do_sample"]:
            raise RuntimeError("probability tensor contains nan")
        if self.fail_greedy and not kwargs["do_sample"]:
            raise RuntimeError("greedy generation failed")
        return torch.cat((input_ids, torch.tensor([[3]])), dim=1)


def _hybrid_wrapper(*, fail_sampling=False, fail_greedy=False):
    wrapper = object.__new__(HybridLanguageModel)
    wrapper.hf_model = _HybridModel(
        fail_sampling=fail_sampling,
        fail_greedy=fail_greedy,
    )
    wrapper.tokenizer = _HybridTokenizer()
    wrapper.device = "cpu"
    wrapper.default_max_tokens = 64
    wrapper._current_activations = {}
    wrapper._current_sae_features = None
    wrapper._call_count = 0
    wrapper._last_prompt = None
    wrapper._last_sampling_config = {}
    return wrapper


def _assert_edge_request(config):
    assert config["temperature"] == 0.01
    assert config["top_p"] == 0.8
    assert config["top_k"] == 12
    assert config["seed"] == 7
    assert config["requested"] == {
        "max_tokens": 999,
        "temperature": 0.01,
        "top_p": 0.8,
        "top_k": 12,
        "seed": 7,
        "do_sample": True,
    }
    assert config["effective"]["max_tokens"] == 256
    assert config["effective"]["temperature"] == 0.1
    assert config["effective"]["top_p"] == 0.8
    assert config["effective"]["top_k"] == 12
    assert config["effective"]["seed"] == 7
    assert config["effective"]["do_sample"] is True
    assert config["max_tokens_cap"] == 256
    assert config["temperature_floor"] == 0.1
    assert config["fallback_used"] is False
    assert config["fallback_reason"] is None


def test_transformer_provenance_records_effective_caps_and_primary_path():
    wrapper = _transformer_wrapper()

    response = wrapper.sample_text(
        "prompt",
        max_tokens=999,
        temperature=0.01,
        top_p=0.8,
        top_k=12,
        seed=7,
        capture_activations=False,
    )

    assert response == "answer"
    config = wrapper.get_last_sampling_config()
    _assert_edge_request(config)
    assert config["frequency_penalty"] == 1.0
    assert config["repetition_penalty"] is None
    assert config["generation_path"] == "transformer_lens_sampling"
    assert wrapper.model.generate_calls == [
        {
            "freq_penalty": 1.0,
            "max_new_tokens": 256,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 12,
            "stop_at_eos": True,
        }
    ]


def test_transformer_provenance_records_penalty_compatibility_fallback():
    wrapper = _transformer_wrapper(reject_frequency_penalty=True)

    assert wrapper.sample_text("prompt", capture_activations=False) == "answer"

    config = wrapper.get_last_sampling_config()
    assert config["generation_path"] == (
        "transformer_lens_sampling_without_frequency_penalty"
    )
    assert config["fallback_used"] is True
    assert config["fallback_reason"] == "freq_penalty_unsupported"
    assert config["frequency_penalty"] is None
    assert len(wrapper.model.generate_calls) == 2
    assert wrapper.model.generate_calls[0]["freq_penalty"] == 1.0
    assert "freq_penalty" not in wrapper.model.generate_calls[1]


def test_transformer_provenance_records_requested_greedy_path():
    wrapper = _transformer_wrapper()

    assert wrapper.sample_text(
        "prompt", temperature=0, capture_activations=False
    ) == "answer"

    config = wrapper.get_last_sampling_config()
    assert config["requested"]["temperature"] == 0.0
    assert config["requested"]["do_sample"] is False
    assert config["effective"]["temperature"] == 0.1
    assert config["effective"]["top_p"] == 0.9
    assert config["effective"]["top_k"] == 50
    assert config["effective"]["do_sample"] is False
    assert config["generation_path"] == "transformer_lens_greedy"
    assert config["fallback_used"] is False
    assert wrapper.model.generate_calls[0]["do_sample"] is False


def test_hybrid_provenance_records_effective_caps_and_primary_path():
    wrapper = _hybrid_wrapper()

    response = wrapper.sample_text(
        "prompt",
        max_tokens=999,
        temperature=0.01,
        top_p=0.8,
        top_k=12,
        seed=7,
        capture_activations=False,
    )

    assert response == "answer"
    config = wrapper.get_last_sampling_config()
    _assert_edge_request(config)
    assert config["frequency_penalty"] is None
    assert config["repetition_penalty"] == 1.15
    assert config["generation_path"] == "huggingface_sampling"


def test_hybrid_provenance_records_sampling_to_greedy_fallback():
    wrapper = _hybrid_wrapper(fail_sampling=True)

    assert wrapper.sample_text(
        "prompt", temperature=0.7, capture_activations=False
    ) == "answer"

    config = wrapper.get_last_sampling_config()
    assert config["requested"]["temperature"] == 0.7
    assert config["requested"]["do_sample"] is True
    assert config["effective"]["temperature"] is None
    assert config["effective"]["top_p"] is None
    assert config["effective"]["top_k"] is None
    assert config["effective"]["do_sample"] is False
    assert config["do_sample"] is False
    assert config["repetition_penalty"] == 1.15
    assert config["generation_path"] == "huggingface_greedy_fallback"
    assert config["fallback_used"] is True
    assert config["fallback_reason"] == "sampling_RuntimeError"
    assert len(wrapper.hf_model.generate_calls) == 2
    assert wrapper.hf_model.generate_calls[0]["do_sample"] is True
    assert wrapper.hf_model.generate_calls[1]["do_sample"] is False
    assert "temperature" not in wrapper.hf_model.generate_calls[1]
    assert wrapper.hf_model.generate_calls[1]["repetition_penalty"] == 1.15


def test_hybrid_provenance_records_sampling_and_greedy_failure():
    wrapper = _hybrid_wrapper(fail_sampling=True, fail_greedy=True)

    assert wrapper.sample_text(
        "prompt", temperature=0.7, capture_activations=False
    ) == ""

    config = wrapper.get_last_sampling_config()
    assert config["requested"]["temperature"] == 0.7
    assert config["requested"]["do_sample"] is True
    assert config["effective"]["temperature"] is None
    assert config["effective"]["top_p"] is None
    assert config["effective"]["top_k"] is None
    assert config["effective"]["do_sample"] is False
    assert config["do_sample"] is False
    assert config["repetition_penalty"] == 1.15
    assert config["generation_path"] == "huggingface_generation_failed"
    assert config["fallback_used"] is True
    assert config["fallback_reason"] == "sampling_and_greedy_RuntimeError"
    assert len(wrapper.hf_model.generate_calls) == 2
    assert wrapper.hf_model.generate_calls[0]["do_sample"] is True
    assert wrapper.hf_model.generate_calls[1]["do_sample"] is False
