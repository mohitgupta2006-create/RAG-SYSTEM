import json
from unittest.mock import Mock

import pytest
from google.genai import errors

from rag.errors import GenerationError
from rag.generation import NO_ANSWER, generate_answer, render_answer


CHUNKS = [dict(id="known", content="The waiting period is 30 days.",
               metadata=dict(page_number=1, section_header="Waiting Periods"))]


def good_answer():
    return json.dumps(dict(answerable=True, answer="The waiting period is 30 days.", source_ids=["known"]))


def test_citations_are_rendered_from_real_metadata():
    result = render_answer(good_answer(), CHUNKS, "policy.pdf")
    assert "[Source: policy.pdf, Page 1, Section: Waiting Periods]" in result


@pytest.mark.parametrize("raw", ["not JSON", "", "null", json.dumps(dict(
    answerable=True, answer="Invented claim", source_ids=["unknown"])),
    json.dumps(dict(answerable=True, answer="Unsupported", source_ids=[])),
])
def test_malformed_or_unknown_citations_are_rejected(raw):
    with pytest.raises(GenerationError):
        render_answer(raw, CHUNKS, "policy.pdf")


def test_empty_context_never_calls_provider():
    client = Mock()
    assert generate_answer("question", [], client, "model", "a.pdf") == NO_ANSWER
    client.models.generate_content.assert_not_called()


def test_retries_transient_error_but_never_sleeps_after_last_attempt():
    client, sleeper = Mock(), Mock()
    client.models.generate_content.side_effect = [errors.APIError(429, {"message": "limited"})] * 3
    with pytest.raises(GenerationError):
        generate_answer("question", CHUNKS, client, "model", "a.pdf", sleep=sleeper)
    assert client.models.generate_content.call_count == 3
    assert sleeper.call_count == 2


def test_provider_error_does_not_become_answer_or_leak_details():
    client, sleeper = Mock(), Mock()
    client.models.generate_content.side_effect = errors.APIError(403, {"message": "secret upstream body"})
    with pytest.raises(GenerationError) as result:
        generate_answer("question", CHUNKS, client, "model", "a.pdf", sleep=sleeper)
    assert "secret" not in result.value.public_message
    sleeper.assert_not_called()


def test_structured_config_and_success():
    client = Mock()
    client.models.generate_content.return_value.text = good_answer()
    assert "30 days" in generate_answer("question", CHUNKS, client, "model", "a.pdf")
    config = client.models.generate_content.call_args.kwargs["config"]
    from google.genai.types import GenerateContentConfig
    assert GenerateContentConfig(**config).response_mime_type == "application/json"
