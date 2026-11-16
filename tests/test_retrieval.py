from unittest.mock import Mock

import faiss
import numpy as np

from rag.cache import KnowledgeBase
from rag.retrieval import Reranker, bm25_search, reciprocal_rank_fusion, semantic_search


def knowledge_base():
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [0, 1]], dtype="float32"))
    return KnowledgeBase(index, [dict(id="waiting", content="Waiting period is 30 days"),
                                 dict(id="maternity", content="Maternity has a separate limit")])


def test_semantic_search_normalizes_query_and_skips_negative_indices():
    kb = knowledge_base()
    results = semantic_search([10, 0], kb)
    assert results[0]["chunk"]["id"] == "waiting"
    assert results[0]["score"] == 1
    kb.index = Mock()
    kb.index.search.return_value = (np.array([[0.5, 0.2]]), np.array([[0, -1]]))
    assert len(semantic_search([1, 0], kb)) == 1


def test_bm25_cached_and_no_keyword_matches_are_excluded():
    kb = knowledge_base()
    assert bm25_search("waiting", kb)[0]["chunk"]["id"] == "waiting"
    model = kb.bm25
    assert bm25_search("unrelatedword", kb) == []
    assert kb.bm25 is model


def test_rank_fusion_uses_stable_ids_across_copies():
    one = dict(chunk=dict(id="a", content="A"))
    duplicate = dict(chunk=dict(id="a", content="A"))
    other = dict(chunk=dict(id="b", content="B"))
    result = reciprocal_rank_fusion([one, other], [duplicate])
    assert [chunk["id"] for chunk in result] == ["a", "b"]


def test_reranker_falls_back_and_does_not_repeat_failed_model_calls():
    reranker = Reranker()
    model = Mock()
    model.predict.side_effect = RuntimeError("unavailable")
    reranker._model = model
    chunks = knowledge_base().chunks
    assert reranker.rerank("question", chunks) == chunks
    assert reranker.rerank("question", chunks) == chunks
    model.predict.assert_called_once()
