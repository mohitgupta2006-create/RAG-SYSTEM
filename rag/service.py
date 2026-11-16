"""Application orchestration with isolated temporary files and bounded work."""

import tempfile
import threading
from pathlib import Path

from ingestion_pipeline.ingestionPipeline import run_ingestion_pipeline
from inference_pipeline.inferencePipeline import run_inference_pipeline
from rag.cache import KnowledgeBase, KnowledgeBaseCache, cache_key, file_hash
from rag.download import download_pdf, source_filename
from rag.errors import BusyError
from rag.retrieval import Reranker


class RAGService:
    def __init__(self, settings, embedding_model=None, client=None, reranker=None):
        self.settings = settings
        self.cache = KnowledgeBaseCache(settings.cache_dir, settings.max_cached_documents)
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(settings.embedding_model)
        self.embedding_model = embedding_model
        if client is None:
            from google import genai
            client = genai.Client(
                api_key=settings.gemini_api_key.get_secret_value(),
                http_options={"timeout": int(settings.generation_timeout * 1000),
                              "retry_options": {"attempts": 1}},
            )
        self.client = client
        self.reranker = reranker if reranker is not None else Reranker()
        self._embedding_guard = threading.Lock()
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_requests)

    def answer(self, document: str, queries: list[str]) -> list[str]:
        if not self._slots.acquire(blocking=False):
            raise BusyError()
        try:
            with tempfile.TemporaryDirectory(prefix="request-", dir=self.cache.directory) as work:
                pdf_path = Path(work) / "document.pdf"
                download_pdf(document, pdf_path, max_bytes=self.settings.max_document_bytes,
                             timeout=self.settings.download_timeout,
                             allowed_hosts=self.settings.allowed_document_hosts)
                name = source_filename(document)
                key = cache_key(file_hash(pdf_path), self.settings.embedding_model, self.settings.max_pages)

                def build():
                    with self._embedding_guard:
                        index, chunks = run_ingestion_pipeline(
                            str(pdf_path), self.embedding_model, source_name=name,
                            max_pages=self.settings.max_pages,
                        )
                    return KnowledgeBase(index, chunks)

                kb = self.cache.get_or_create(key, build)
                with self._embedding_guard:
                    vectors = self.embedding_model.encode(queries, convert_to_numpy=True)
                # Use this request's filename, including when the same bytes were cached under another URL.
                return run_inference_pipeline(
                    queries, kb, vectors, self.client, self.settings.gemini_model,
                    name, self.reranker, timeout=self.settings.generation_timeout,
                )
        finally:
            self._slots.release()

    def close(self):
        self.client.close()
