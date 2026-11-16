from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import faiss
import numpy as np

from rag.cache import KnowledgeBase, KnowledgeBaseCache, cache_key


def make_kb():
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0]], dtype="float32"))
    return KnowledgeBase(index, [dict(id="chunk1", content="Coverage", metadata=dict(page_number=1))])


def test_disk_roundtrip_and_version_keys(tmp_path):
    key = cache_key("document", "model", 500)
    assert key != cache_key("document", "different-model", 500)
    assert key != cache_key("document", "model", 20)
    original = KnowledgeBaseCache(tmp_path).get_or_create(key, make_kb)
    builder = Mock(side_effect=AssertionError("Cache should be reused"))
    restored = KnowledgeBaseCache(tmp_path).get_or_create(key, builder)
    assert restored.chunks == original.chunks
    assert restored.index.ntotal == 1


def test_corrupt_pair_is_rebuilt_without_partial_memory(tmp_path):
    key = cache_key("document", "model", 500)
    cache = KnowledgeBaseCache(tmp_path)
    cache.get_or_create(key, make_kb)
    (tmp_path / f"{key}.json").write_text("[]")
    builder = Mock(side_effect=make_kb)
    rebuilt = KnowledgeBaseCache(tmp_path).get_or_create(key, builder)
    builder.assert_called_once()
    assert len(rebuilt.chunks) == 1


def test_concurrent_requests_build_once_and_lru_is_bounded(tmp_path):
    cache = KnowledgeBaseCache(tmp_path, max_entries=1)
    key = cache_key("same", "model", 500)
    builder = Mock(side_effect=make_kb)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: cache.get_or_create(key, builder), range(4)))
    assert builder.call_count == 1
    assert all(value is values[0] for value in values)
    cache.get_or_create(cache_key("another", "model", 500), make_kb)
    assert len(cache._memory) == 1
    assert key not in cache._memory
