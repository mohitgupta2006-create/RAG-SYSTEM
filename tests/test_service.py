import json
from unittest.mock import Mock

import pymupdf as fitz
import pytest

from rag.config import Settings
from rag.errors import BusyError, GenerationError
from rag.service import RAGService


def build_service(tmp_path, embedding, monkeypatch):
    config = Settings(api_key="test-token", gemini_api_key="provider-token", gemini_model="test-model", cache_dir=tmp_path)
    client = Mock()

    def reply(**kwargs):
        passages = json.loads(kwargs["contents"])["passages"]
        return Mock(text=json.dumps(dict(answerable=True, answer="The waiting period is 30 days.",
                                         source_ids=[passages[0]["id"]])))

    client.models.generate_content.side_effect = reply
    reranker = Mock()
    reranker.rerank.side_effect = lambda query, chunks, top_k=5: chunks[:top_k]
    service = RAGService(config, embedding, client, reranker)
    downloads = []

    def download(url, path, **kwargs):
        downloads.append(path)
        with fitz.open() as doc:
            doc.new_page().insert_text((72, 150), "The waiting period is 30 days.")
            doc.save(path, no_new_id=True)

    monkeypatch.setattr("rag.service.download_pdf", download)
    return service, downloads


def test_pdf_to_answer_and_cache_reuse_with_request_filename(tmp_path, embedding, monkeypatch):
    service, downloads = build_service(tmp_path, embedding, monkeypatch)
    first = service.answer("https://example.com/policy.pdf?secret=token", ["Waiting period?"])
    assert "Source: policy.pdf, Page 1" in first[0]
    # Replay the identical PDF bytes so the next URL takes the content-cache path.
    original_download = __import__("rag.service", fromlist=["download_pdf"]).download_pdf
    saved = tmp_path / "fixture.pdf"
    original_download("ignored", saved)
    data = saved.read_bytes()
    monkeypatch.setattr("rag.service.download_pdf", lambda url, path, **kw: path.write_bytes(data))
    service.answer("https://example.com/original.pdf", ["Waiting?"])
    from rag.service import run_ingestion_pipeline
    build = Mock(wraps=run_ingestion_pipeline)
    monkeypatch.setattr("rag.service.run_ingestion_pipeline", build)
    second = service.answer("https://example.com/renamed.pdf", ["Waiting?"])
    build.assert_not_called()
    assert "Source: renamed.pdf, Page 1" in second[0]
    assert all(not path.exists() for path in downloads if path != saved)
    assert not list(tmp_path.glob("request-*"))


def test_temporary_pdf_cleanup_after_generation_failure(tmp_path, embedding, monkeypatch):
    service, downloads = build_service(tmp_path, embedding, monkeypatch)
    service.client.models.generate_content.side_effect = RuntimeError("provider failure")
    with pytest.raises(GenerationError):
        service.answer("https://example.com/policy.pdf", ["Waiting?"])
    assert not list(tmp_path.glob("request-*"))
    assert all(not path.exists() for path in downloads)


def test_admission_limit_rejects_extra_work(tmp_path, embedding, monkeypatch):
    service, _ = build_service(tmp_path, embedding, monkeypatch)
    for _ in range(service.settings.max_concurrent_requests):
        service._slots.acquire()
    try:
        with pytest.raises(BusyError):
            service.answer("https://example.com/policy.pdf", ["Waiting?"])
    finally:
        for _ in range(service.settings.max_concurrent_requests):
            service._slots.release()
