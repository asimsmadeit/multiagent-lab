"""Regression tests for optional interpretability dependencies."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from interpretability.llm_evaluation import create_remote_ollama_model


def test_remote_ollama_factory_explains_how_to_install_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error remains deterministic even when Ollama is installed locally."""
    original_import = builtins.__import__

    def import_without_ollama(name, *args, **kwargs):
        if name == "ollama":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ollama)

    with pytest.raises(ImportError) as exc_info:
        create_remote_ollama_model(host_ip="192.0.2.1")

    assert "python -m pip install -e '.[apis]'" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ImportError)
    assert str(exc_info.value.__cause__) == "simulated missing optional dependency"


def test_remote_ollama_factory_uses_available_optional_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_calls = []

    class FakeClient:
        def __init__(self, *, host: str) -> None:
            client_calls.append(host)

        def generate(self, **kwargs):
            return {"response": "accepted STOP ignored"}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))

    model = create_remote_ollama_model(
        model_name="test-model",
        host_ip="192.0.2.1",
        port=1234,
    )

    assert client_calls == ["http://192.0.2.1:1234"]
    assert model.sample_text("prompt", terminators=("STOP",)) == "accepted "
