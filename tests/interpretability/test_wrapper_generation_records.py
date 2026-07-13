"""Adapter integration tests for immutable scoped generation records."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from interpretability.evaluation import (
    FastModelWrapper,
    HuggingFaceTextModel,
    HybridLanguageModel,
    InterpretabilityRunner,
    TransformerLensWrapper,
)
from interpretability.runtime.model_call import (
    CallPurpose,
    CaptureMode,
    GenerationCallSpec,
    GenerationRecord,
    GenerationRecorder,
    active_generation_recorder,
    generation_call,
)


class _Tokenizer:
    pad_token_id = 0
    _pieces = {
        3: 'hello',
        4: ' world',
        5: '',  # EOS/chat marker removed by skip_special_tokens.
        6: '<STOP>',
        7: 'ignored',
    }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]['content']

    def decode(self, token_ids, *, skip_special_tokens=True):
        assert skip_special_tokens is True
        ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else token_ids
        return ''.join(self._pieces.get(int(token_id), '') for token_id in ids)

    def __call__(self, prompt, **kwargs):
        assert prompt
        assert kwargs['return_tensors'] == 'pt'
        return _Batch()


class _Batch:
    def __init__(self):
        self.input_ids = torch.tensor([[1, 2]])

    def to(self, device):
        assert device == 'cpu'
        return self


class _ReplayModel:
    def __init__(self, *, nonfinite=False):
        self.cfg = SimpleNamespace(n_ctx=1024, d_model=4)
        self.tokenizer = _Tokenizer()
        self.nonfinite = nonfinite
        self.replayed_tokens = None
        self.hooks_added = []

    def to_tokens(self, prompt, *, truncate, prepend_bos):
        assert prompt and truncate is True and prepend_bos is False
        return torch.tensor([[1, 2]])

    def generate(self, tokens, **kwargs):
        del kwargs
        suffix = torch.tensor([[3, 4, 5, 6, 7]])
        return torch.cat((tokens, suffix), dim=1)

    def run_with_cache(self, tokens, *, names_filter):
        self.replayed_tokens = tokens.clone()
        hook_name = 'blocks.1.hook_resid_post'
        assert names_filter(hook_name)
        activation = torch.arange(
            tokens.shape[1] * 4, dtype=torch.float32
        ).reshape(1, tokens.shape[1], 4)
        if self.nonfinite:
            activation[0, -1, 0] = float('nan')
        return None, {hook_name: activation}

    def reset_hooks(self):
        self.hooks_added.clear()

    def add_hook(self, name, hook):
        self.hooks_added.append((name, hook))


class _HFModel:
    def __init__(self):
        self.config = SimpleNamespace(
            max_position_embeddings=1024,
            vocab_size=100,
        )

    def generate(self, input_ids, *, attention_mask, **kwargs):
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        del kwargs
        return torch.cat(
            (input_ids, torch.tensor([[3, 4, 5, 6, 7]])), dim=1
        )


def _transformer_wrapper(*, capture=True, nonfinite=False):
    wrapper = object.__new__(TransformerLensWrapper)
    wrapper.model = _ReplayModel(nonfinite=nonfinite)
    wrapper.default_max_tokens = 64
    wrapper.capture_mean_pooled = False
    wrapper.hook_names = ['blocks.1.hook_resid_post'] if capture else []
    wrapper._current_activations = {}
    wrapper._call_count = 0
    wrapper._last_prompt = None
    wrapper._last_sampling_config = {}
    wrapper._steering_direction = None
    return wrapper


def _hybrid_wrapper():
    wrapper = object.__new__(HybridLanguageModel)
    wrapper.hf_model = _HFModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.tl_model = _ReplayModel()
    wrapper.device = 'cpu'
    wrapper.default_max_tokens = 64
    wrapper.hook_names = ['blocks.1.hook_resid_post']
    wrapper.use_sae = False
    wrapper.sae = None
    wrapper._current_activations = {}
    wrapper._current_sae_features = None
    wrapper._call_count = 0
    wrapper._last_prompt = None
    wrapper._last_sampling_config = {}
    return wrapper


def _text_wrapper():
    wrapper = object.__new__(HuggingFaceTextModel)
    wrapper.hf_model = _HFModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.device = 'cpu'
    wrapper.default_max_tokens = 64
    wrapper.hook_names = []
    wrapper.use_sae = False
    wrapper.sae = None
    wrapper._current_activations = {}
    wrapper._current_sae_features = None
    wrapper._call_count = 0
    wrapper._last_prompt = None
    wrapper._last_sampling_config = {}
    return wrapper


def _spec(
    sequence: int,
    *,
    actor_id: str = 'seller',
    purpose: CallPurpose = CallPurpose.ACTOR_ACTION,
    capture_mode: CaptureMode = CaptureMode.TEACHER_FORCED_REPLAY,
) -> GenerationCallSpec:
    return GenerationCallSpec(
        run_id='run-records',
        trial_id='trial-9',
        attempt=0,
        sequence=sequence,
        actor_id=actor_id,
        purpose=purpose,
        model_revision='model@abc',
        tokenizer_revision='tokenizer@def',
        concordia_version='2.4.0',
        capture_mode=capture_mode,
    )


@pytest.mark.parametrize('wrapper_factory', [_transformer_wrapper, _hybrid_wrapper])
def test_wrappers_publish_exact_replay_record(wrapper_factory) -> None:
    wrapper = wrapper_factory()
    recorder = GenerationRecorder('run-records')

    with active_generation_recorder(recorder):
        with generation_call(_spec(1)):
            response = wrapper.sample_text(
                'assembled private and public context',
                max_tokens=999,
                temperature=0.01,
                top_p=0.8,
                top_k=12,
                seed=7,
                terminators=('<STOP>',),
            )

    assert response == 'hello world'
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.call_id == _spec(1).call_id
    assert record.assembled_prompt == 'assembled private and public context'
    assert record.input_token_ids == (1, 2)
    assert record.output_token_ids == (3, 4, 5, 6, 7)
    assert record.retained_token_ids == (3, 4)
    assert record.output_text == response
    assert record.terminator == '<STOP>'
    assert record.requested_sampling.max_tokens == 999
    assert record.effective_sampling.max_tokens == 256
    assert record.requested_sampling.temperature == pytest.approx(0.01)
    assert record.effective_sampling.temperature == pytest.approx(0.1)
    assert record.capture_mode is CaptureMode.TEACHER_FORCED_REPLAY
    assert record.activation_position == 'last_retained_response_token'
    assert record.retained_token_index == 1
    assert record.replay_call_id and record.replay_call_id != record.call_id
    assert record.replay_call_id == _spec(1).replay_call_id
    assert len(record.activation_artifacts) == 1
    artifact = record.activation_artifacts[0]
    assert artifact.artifact_hash.startswith('sha256:')
    assert artifact.token_index == 1
    assert artifact.shape == (4,)
    assert artifact.dtype == 'float32'
    assert torch.equal(
        (wrapper.model if hasattr(wrapper, 'model') else wrapper.tl_model).replayed_tokens,
        torch.tensor([[1, 2, 3, 4]]),
    )
    assert GenerationRecord.from_dict(
        json.loads(json.dumps(record.to_dict()))
    ) == record


def test_consecutive_calls_keep_identity_and_counterpart_cannot_overwrite() -> None:
    wrapper = _transformer_wrapper(capture=False)
    recorder = GenerationRecorder('run-records')
    checkpoint = recorder.checkpoint()

    calls = (
        _spec(1, capture_mode=CaptureMode.NONE),
        _spec(2, capture_mode=CaptureMode.NONE),
        _spec(
            3,
            actor_id='buyer',
            purpose=CallPurpose.COUNTERPART_ACTION,
            capture_mode=CaptureMode.NONE,
        ),
    )
    with active_generation_recorder(recorder):
        for spec in calls:
            with generation_call(spec):
                wrapper.sample_text('prompt', capture_activations=False)

    assert len({record.call_id for record in recorder.records}) == 3
    selected = InterpretabilityRunner._select_final_acting_call(
        recorder,
        trial_id='trial-9',
        actor_id='seller',
        start_index=checkpoint,
    )
    assert selected.sequence == 2
    assert selected.actor_id == 'seller'
    assert recorder.records[-1].actor_id == 'buyer'


def test_fast_wrapper_forces_none_capture_and_disables_active_steering() -> None:
    base = _transformer_wrapper()
    base._steering_direction = torch.ones(4)
    base._steering_layer = 1
    base._steering_magnitude = 1.0
    base._steering_hook_name = 'blocks.1.hook_resid_post'
    fast = FastModelWrapper(base)
    recorder = GenerationRecorder('run-records')

    with active_generation_recorder(recorder):
        with generation_call(_spec(1, capture_mode=CaptureMode.NONE)):
            response = fast.sample_text(
                'counterpart prompt',
                capture_activations=True,
                apply_steering=True,
            )

    assert response == 'hello world<STOP>ignored'
    record = recorder.records[0]
    assert record.capture_mode is CaptureMode.NONE
    assert record.activation_position is None
    assert record.activation_artifacts == ()
    assert base.get_activations() == {}
    assert base.model.hooks_added == []


def test_text_wrapper_publishes_generation_without_white_box_capture() -> None:
    wrapper = _text_wrapper()
    recorder = GenerationRecorder('run-records')

    with active_generation_recorder(recorder):
        with generation_call(_spec(1, capture_mode=CaptureMode.NONE)):
            response = wrapper.sample_text(
                'text-only prompt',
                capture_activations=True,
                apply_steering=True,
            )

    assert response == 'hello world<STOP>ignored'
    assert not hasattr(wrapper, 'tl_model')
    assert wrapper.get_activations() == {}
    assert wrapper.activation_dim == 0
    record = recorder.records[0]
    assert record.capture_mode is CaptureMode.NONE
    assert record.activation_artifacts == ()
    assert record.activation_position is None


def test_text_wrapper_initialization_never_imports_transformer_lens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    tokenizer = _Tokenizer()
    tokenizer.eos_token = '<eos>'
    tokenizer.pad_token = None
    hf_model = _HFModel()
    monkeypatch.setitem(sys.modules, 'transformer_lens', None)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        'from_pretrained',
        lambda *_args, **_kwargs: hf_model,
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        'from_pretrained',
        lambda *_args, **_kwargs: tokenizer,
    )

    wrapper = HuggingFaceTextModel(
        model_name='offline-model',
        device='cpu',
        torch_dtype=torch.float32,
    )

    assert wrapper.hf_model is hf_model
    assert wrapper.tokenizer is tokenizer
    assert wrapper.tokenizer.pad_token == '<eos>'
    assert not hasattr(wrapper, 'tl_model')
    assert wrapper.hook_names == []


def test_nonfinite_activation_fails_before_publication() -> None:
    wrapper = _transformer_wrapper(nonfinite=True)
    recorder = GenerationRecorder('run-records')

    with active_generation_recorder(recorder):
        with generation_call(_spec(1)):
            with pytest.raises(ValueError, match='finite'):
                wrapper.sample_text('prompt')

    assert recorder.records == ()


def test_active_recorder_without_call_identity_fails_closed() -> None:
    wrapper = _transformer_wrapper(capture=False)
    recorder = GenerationRecorder('run-records')

    with active_generation_recorder(recorder):
        with pytest.raises(RuntimeError, match='explicit generation_call'):
            wrapper.sample_text('prompt', capture_activations=False)

    assert recorder.records == ()
