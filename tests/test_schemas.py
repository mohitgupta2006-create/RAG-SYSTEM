import pytest
from pydantic import ValidationError

from rag.auth import valid_api_key
from rag.schemas import EndpointRequest


@pytest.mark.parametrize(
    "change",
    [
        {"document": "file:///etc/passwd"},
        {"document": "https://user:pass@example.com/a.pdf"},
        {"queries": []},
        {"queries": [" "]},
        {"queries": ["x" * 2001]},
        {"queries": ["question"] * 21},
        {"unexpected": "value"},
    ],
)
def test_rejects_invalid_requests(change):
    payload = dict(document="https://example.com/policy.pdf", queries=["Coverage?"])
    payload.update(change)
    with pytest.raises(ValidationError):
        EndpointRequest(**payload)


def test_trims_query_whitespace():
    request = EndpointRequest(document="https://example.com/a.pdf", queries=[" Coverage? "])
    assert request.queries == ["Coverage?"]


def test_api_key_comparison_handles_unicode():
    assert valid_api_key("token-✓", "token-✓")
    assert not valid_api_key("wrong", "token-✓")
