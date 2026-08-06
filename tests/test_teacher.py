import requests

from taskcompiler import teacher
from taskcompiler.config import OPENROUTER_TEACHER


def test_teacher_loads_api_key_from_project_env(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(teacher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        teacher.anthropic,
        "Anthropic",
        lambda *, api_key: captured.setdefault("client", api_key),
    )
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=test-key\n", encoding="utf-8")

    teacher.TeacherClient("test-model")

    assert captured["client"] == "test-key"


class _OpenRouterResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "model": "deepseek/deepseek-r1-0528",
            "provider": "test-provider",
            "choices": [{"message": {"content": "{\"obligations\": []}"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }


def test_openrouter_teacher_uses_chat_completions(monkeypatch):
    request = {}

    def post(self, url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return _OpenRouterResponse()

    monkeypatch.setattr(teacher.requests.Session, "post", post)
    client = teacher.TeacherClient(
        "deepseek/deepseek-r1-0528",
        api_key="test-openrouter-key",
        backend=OPENROUTER_TEACHER,
        provider="test-provider",
    )

    response = client.call(
        "system",
        "prompt",
        max_tokens=80,
        response_format={"type": "json_object"},
        seed=7,
    )

    assert request["url"] == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert request["json"]["model"] == "deepseek/deepseek-r1-0528"
    assert request["json"]["response_format"] == {"type": "json_object"}
    assert request["json"]["seed"] == 7
    assert request["json"]["reasoning"] == {"enabled": False}
    assert request["json"]["provider"] == {
        "order": ["test-provider"],
        "allow_fallbacks": False,
    }
    assert request["headers"]["Authorization"] == (
        "Bearer test-openrouter-key"
    )
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert response.backend == OPENROUTER_TEACHER


def test_openrouter_error_includes_bounded_provider_detail(monkeypatch):
    class _FailedResponse:
        text = '{"error":{"message":"schema is unsupported"}}'

        def raise_for_status(self):
            error = requests.HTTPError("400 Client Error")
            error.response = self
            raise error

    monkeypatch.setattr(
        teacher.requests.Session,
        "post",
        lambda self, url, **kwargs: _FailedResponse(),
    )
    client = teacher.TeacherClient(
        "test-model",
        api_key="test-openrouter-key",
        backend=OPENROUTER_TEACHER,
    )

    try:
        client.call("system", "prompt")
    except RuntimeError as exc:
        assert "schema is unsupported" in str(exc)
    else:
        raise AssertionError("provider error should be reported")
