import inspect

import torch

from interpretability.evaluation import (
    FastModelWrapper,
    HybridLanguageModel,
    InterpretabilityRunner,
    TransformerLensWrapper,
    _tokens_through_stored_response,
)


class _Tokenizer:
    _pieces = {1: "hello", 2: " world", 3: "<STOP>", 4: "ignored", 99: ""}

    def decode(self, token_ids, skip_special_tokens=True):
        ids = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        return "".join(self._pieces.get(int(token_id), "") for token_id in ids)


def test_activation_position_excludes_eos_and_terminator_suffix():
    generated = torch.tensor([[42, 1, 2, 99, 3, 4]])

    retained = _tokens_through_stored_response(
        _Tokenizer(), generated, prompt_length=1, response="hello world"
    )

    assert retained.tolist() == [[42, 1, 2]]


def test_model_wrappers_accept_latest_concordia_sampling_parameters():
    for wrapper_type in (TransformerLensWrapper, HybridLanguageModel):
        parameters = inspect.signature(wrapper_type.sample_text).parameters
        assert "top_p" in parameters
        assert "top_k" in parameters


class _BaseModel:
    def __init__(self):
        self.kwargs = None
        self._calls = 0

    def sample_text(self, prompt, **kwargs):
        self.kwargs = kwargs
        self._calls += 1
        return "alpha"

    @property
    def call_count(self):
        return self._calls


def test_fast_model_disables_capture_and_steering():
    base = _BaseModel()
    fast = FastModelWrapper(base)

    assert fast.sample_text("prompt", top_p=0.8, top_k=12) == "alpha"
    assert base.kwargs["capture_activations"] is False
    assert base.kwargs["apply_steering"] is False
    assert base.kwargs["top_p"] == 0.8
    assert base.kwargs["top_k"] == 12


def test_actor_and_game_master_memory_duplicate_contracts():
    runner = object.__new__(InterpretabilityRunner)

    actor_memory = runner._create_memory_bank()
    gm_memory = runner._create_memory_bank(allow_duplicates=True)

    assert actor_memory._allow_duplicates is False
    assert gm_memory._allow_duplicates is True
