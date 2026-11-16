"""Versioned disk cache with integrity checks and a bounded in-memory LRU."""

import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
from filelock import FileLock, Timeout

from rag.errors import CacheError

PIPELINE_VERSION = "blocks-token-overlap-cosine-v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(document_hash: str, embedding_model: str, max_pages: int) -> str:
    value = json.dumps([document_hash, embedding_model, max_pages, PIPELINE_VERSION])
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class KnowledgeBase:
    index: Any
    chunks: list[dict]
    bm25: Any = None


class KnowledgeBaseCache:
    def __init__(self, directory: Path, max_entries: int = 8):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._memory = OrderedDict()
        self._guard = threading.Lock()

    def _paths(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise CacheError()
        return [self.directory / f"{key}.{suffix}" for suffix in ("index", "json", "manifest.json")]

    def _load(self, key):
        index_path, chunks_path, manifest_path = self._paths(key)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest["version"] != PIPELINE_VERSION or manifest["key"] != key
                    or manifest["index_sha256"] != file_hash(index_path)
                    or manifest["chunks_sha256"] != file_hash(chunks_path)):
                return None
            index = faiss.read_index(str(index_path))
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            if not isinstance(chunks, list) or not chunks or index.ntotal != len(chunks):
                return None
            for chunk in chunks:
                if (not isinstance(chunk["content"], str) or not chunk["content"].strip()
                        or not chunk["id"] or chunk["metadata"]["page_number"] < 1):
                    return None
            if len({chunk["id"] for chunk in chunks}) != len(chunks):
                return None
            return KnowledgeBase(index, chunks)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return None

    def _save(self, key, kb):
        final_index, final_chunks, final_manifest = self._paths(key)
        temporary = []
        try:
            for _ in range(3):
                fd, name = tempfile.mkstemp(prefix=f"{key}.", suffix=".tmp", dir=self.directory)
                os.close(fd)
                temporary.append(Path(name))
            index_path, chunks_path, manifest_path = temporary
            faiss.write_index(kb.index, str(index_path))
            chunks_path.write_text(json.dumps(kb.chunks, ensure_ascii=False), encoding="utf-8")
            manifest_path.write_text(json.dumps(dict(
                key=key, version=PIPELINE_VERSION, index_sha256=file_hash(index_path),
                chunks_sha256=file_hash(chunks_path),
            )), encoding="utf-8")
            os.replace(index_path, final_index)
            os.replace(chunks_path, final_chunks)
            # Publish the integrity manifest last; interrupted pairs are rebuilt.
            os.replace(manifest_path, final_manifest)
        except OSError as exc:
            raise CacheError() from exc
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)

    def get_or_create(self, key, builder):
        self._paths(key)
        with self._guard:
            if key in self._memory:
                self._memory.move_to_end(key)
                return self._memory[key]
        try:
            with FileLock(str(self.directory / f"{key}.lock"), timeout=120):
                with self._guard:
                    if key in self._memory:
                        self._memory.move_to_end(key)
                        return self._memory[key]
                kb = self._load(key)
                if kb is None:
                    kb = builder()
                    self._save(key, kb)
                with self._guard:
                    self._memory[key] = kb
                    self._memory.move_to_end(key)
                    while len(self._memory) > self.max_entries:
                        self._memory.popitem(last=False)
                return kb
        except (Timeout, OSError) as exc:
            raise CacheError() from exc
