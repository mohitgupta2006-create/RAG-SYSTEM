import pytest
from pydantic import ValidationError

from rag.config import Settings


def settings(**overrides):
    values = dict(api_key="test-token", gemini_api_key="test-provider", gemini_model="test-model")
    values.update(overrides)
    return Settings(**values)


def test_secrets_are_not_in_repr():
    value = settings()
    assert "test-token" not in repr(value)
    assert "test-provider" not in repr(value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": "your_api_key_here"},
        {"gemini_model": " "},
        {"max_concurrent_requests": 0},
        {"download_timeout": -1},
    ],
)
def test_invalid_configuration(overrides):
    with pytest.raises(ValidationError):
        settings(**overrides)


def test_environment_overrides_and_hostname_list(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MAX_PAGES=7\n", encoding="utf-8")
    values = {
        "API_KEY": "test-token",
        "GEMINI_API_KEY": "test-provider",
        "GEMINI_MODEL": "test-model",
        "MAX_PAGES": "11",
        "ALLOWED_DOCUMENT_HOSTS": "POLICY.EXAMPLE, files.example ,",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = Settings.from_env()
    assert config.max_pages == 11
    assert config.allowed_document_hosts == ("policy.example", "files.example")
