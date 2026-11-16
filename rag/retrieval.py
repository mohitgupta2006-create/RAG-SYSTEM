"""Hybrid retrieval with cached BM25 and identity-based rank fusion."""

import logging
import re
import threading

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)
_bm25_guard = threading.Lock()


def tokenize(text):
    return re.findall(r"\b\w+\b", text.casefold())


def semantic_search(query_vector, kb, top_k=10):
    if not kb.chunks:
        return []
    vector = np.array(query_vector, dtype="float32", copy=True).reshape(1, -1)
    if not np.isfinite(vector).all():
        raise ValueError("Query embedding must contain finite values")
    faiss.normalize_L2(vector)
    scores, indices = kb.index.search(vector, min(top_k, len(kb.chunks)))
    return [dict(chunk=kb.chunks[int(i)], score=float(score))
            for i, score in zip(indices[0], scores[0])
            if 0 <= i < len(kb.chunks) and np.isfinite(score)]


def bm25_search(query, kb, top_k=10):
    if not kb.chunks:
        return []
    with _bm25_guard:
        if kb.bm25 is None:
            corpus = [tokenize(chunk["content"]) for chunk in kb.chunks]
            # Punctuation-only chunks must not produce a zero-length BM25 corpus.
            kb.bm25 = (BM25Okapi([tokens or ["__empty__"] for tokens in corpus]),
                       [set(tokens) for tokens in corpus])
    model, token_sets = kb.bm25
    tokens = tokenize(query)
    scores = model.get_scores(tokens)
    candidates = [i for i, words in enumerate(token_sets) if words.intersection(tokens)]
    candidates.sort(key=lambda i: (-scores[i], i))
    return [dict(chunk=kb.chunks[i], score=float(scores[i])) for i in candidates[:top_k]]


def reciprocal_rank_fusion(*rankings, k=60):
    scores, chunks = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            chunk = item["chunk"]
            identity = chunk["id"]
            if identity in seen:
                continue
            seen.add(identity)
            chunks[identity] = chunk
            scores[identity] = scores.get(identity, 0) + 1 / (k + rank)
    return [chunks[key] for key in sorted(scores, key=lambda key: -scores[key])]


class Reranker:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._failed = False

    def rerank(self, query, chunks, top_k=5):
        if not chunks:
            return []
        with self._lock:
            if self._failed:
                return chunks[:top_k]
            try:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                scores = self._model.predict([[query, chunk["content"]] for chunk in chunks])
                if len(scores) != len(chunks) or not np.isfinite(scores).all():
                    raise ValueError("Invalid reranker scores")
                return [chunk for chunk, _ in sorted(zip(chunks, scores), key=lambda pair: -pair[1])[:top_k]]
            except Exception as exc:
                self._failed = True
                logger.warning("Reranker unavailable; using fused ranking (%s)", type(exc).__name__)
                return chunks[:top_k]


def retrieve(query, query_vector, kb, reranker, top_k=5):
    fused = reciprocal_rank_fusion(semantic_search(query_vector, kb), bm25_search(query, kb))
    return reranker.rerank(query, fused, top_k=top_k)
