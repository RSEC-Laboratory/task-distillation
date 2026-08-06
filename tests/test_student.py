import requests
import pytest

from taskcompiler.config import (
    OLLAMA_BACKEND,
    OLLAMA_OPENROUTER_BACKEND,
    OPENROUTER_BACKEND,
    RunConfig,
)
from taskcompiler.student import StudentModel, _normalize_ollama_url


class _OllamaResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "message": {"role": "assistant", "content": "  hello  "},
            "prompt_eval_count": 7,
            "eval_count": 2,
        }


class _TagsResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "models": [
                {
                    "name": "qwen2.5:1.5b",
                    "digest": "sha256:test-digest",
                }
            ]
        }


class _OpenRouterResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "model": "meta-llama/llama-3.2-3b-instruct",
            "provider": "test-provider",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": " entailment ",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
            },
        }


def test_ollama_backend_uses_remote_chat_api(monkeypatch):
    request = {}

    def post(self, url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return _OllamaResponse()

    monkeypatch.setattr("taskcompiler.student.requests.Session.post", post)
    monkeypatch.setattr(
        "taskcompiler.student.requests.Session.get",
        lambda self, *args, **kwargs: _TagsResponse(),
    )
    student = StudentModel(
        "qwen2.5:1.5b",
        backend=OLLAMA_BACKEND,
        ollama_base_url="127.0.0.1",
        ollama_timeout_seconds=30,
        max_new_tokens=64,
    )

    schema = {"type": "object"}
    result = student.ask(
        "question",
        system="instructions",
        response_schema=schema,
    )

    assert request["url"] == "http://127.0.0.1:11434/api/chat"
    assert request["timeout"] == 30
    assert request["json"]["model"] == "qwen2.5:1.5b"
    assert request["json"]["messages"] == [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "question"},
    ]
    assert request["json"]["options"]["num_predict"] == 64
    assert request["json"]["options"]["num_ctx"] == 32768
    assert request["json"]["think"] is False
    assert request["json"]["format"] == schema
    assert result.text == "hello"
    assert result.input_tokens == 7
    assert result.output_tokens == 2
    assert result.finish_reason is None
    assert student.identity["digest"] == "sha256:test-digest"
    assert student._session.adapters["http://"].max_retries.read == 0


def test_openrouter_backend_uses_remote_chat_api(monkeypatch):
    request = {}

    def post(self, url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return _OpenRouterResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("taskcompiler.student.requests.Session.post", post)
    student = StudentModel(
        "meta-llama/llama-3.2-3b-instruct",
        backend=OPENROUTER_BACKEND,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_timeout_seconds=45,
        openrouter_provider="test-provider",
        max_new_tokens=80,
        seed=7,
    )

    result = student.ask("question", system="instructions")

    assert request["url"] == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert request["timeout"] == 45
    assert request["json"]["model"] == (
        "meta-llama/llama-3.2-3b-instruct"
    )
    assert request["json"]["temperature"] == 0
    assert request["json"]["seed"] == 7
    assert request["json"]["max_tokens"] == 80
    assert request["json"]["reasoning"] == {"enabled": False}
    assert request["json"]["provider"] == {
        "order": ["test-provider"],
        "allow_fallbacks": False,
    }
    assert result.text == "entailment"
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.finish_reason == "stop"
    assert student.identity["provider"] == "test-provider"


def test_openrouter_backend_requires_a_hosted_model_identifier(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="provider/model"):
        StudentModel("llama3.2:3b", backend=OPENROUTER_BACKEND)


def test_fallback_prefers_an_available_ollama_model(monkeypatch):
    monkeypatch.setattr(
        "taskcompiler.student.requests.Session.get",
        lambda self, *args, **kwargs: _TagsResponse(),
    )
    student = StudentModel(
        "qwen2.5:1.5b",
        backend=OLLAMA_OPENROUTER_BACKEND,
        openrouter_model_name="qwen/qwen-2.5-7b-instruct",
    )

    assert student.backend == OLLAMA_BACKEND
    assert student.identity["requested_backend"] == OLLAMA_OPENROUTER_BACKEND
    assert student.identity["fallback_used"] is False
    assert student.identity["digest"] == "sha256:test-digest"


def test_new_runs_default_to_openrouter():
    config = RunConfig(track="rules_track", dataset="synthetic")

    assert config.student_backend == OPENROUTER_BACKEND
    assert config.student_model == "meta-llama/llama-3.2-3b-instruct"
    assert config.student_openrouter_model is None


def test_fallback_uses_openrouter_when_ollama_is_unavailable(monkeypatch):
    request = {}

    def get(self, *args, **kwargs):
        raise requests.ConnectionError("LAN server unavailable")

    def post(self, url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return _OpenRouterResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("taskcompiler.student.requests.Session.get", get)
    monkeypatch.setattr("taskcompiler.student.requests.Session.post", post)
    student = StudentModel(
        "llama3.2:3b",
        backend=OLLAMA_OPENROUTER_BACKEND,
        openrouter_model_name="meta-llama/llama-3.2-3b-instruct",
        openrouter_provider="test-provider",
    )
    result = student.ask("question")

    assert student.backend == OPENROUTER_BACKEND
    assert student.identity["requested_backend"] == OLLAMA_OPENROUTER_BACKEND
    assert student.identity["fallback_used"] is True
    assert "LAN server unavailable" in student.identity["fallback_reason"]
    assert request["json"]["model"] == "meta-llama/llama-3.2-3b-instruct"
    assert result.text == "entailment"


def test_ollama_url_accepts_local_and_explicit_ports():
    assert _normalize_ollama_url("127.0.0.1") == "http://127.0.0.1:11434"
    assert (
        _normalize_ollama_url("http://127.0.0.1:1234/")
        == "http://127.0.0.1:1234"
    )


@pytest.mark.parametrize(
    "url",
    ["", "ftp://127.0.0.1", "http://user:secret@host", "http://host/api"],
)
def test_ollama_url_rejects_invalid_values(url):
    with pytest.raises(ValueError):
        _normalize_ollama_url(url)
