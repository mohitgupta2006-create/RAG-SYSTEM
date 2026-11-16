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


@pytest.mark.parametrize("overrides", [
    {"api_key": "your_api_key_here"}, {"gemini_model": " "},
    {"max_concurrent_requests": 0}, {"download_timeout": -1},
])
def test_invalid_configuration(overrides):
    with pytest.raises(ValidationError):
        settings(**overrides)
