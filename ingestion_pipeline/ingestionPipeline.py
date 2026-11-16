"""Extract page-aware PDF text and build a document's vector knowledge base."""

import json
import os
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pymupdf as fitz

from rag.errors import DocumentError
from rag.chunking import create_chunks

HEADER_FOOTER_MARGIN = 0.08
REPEATING_CONTENT_THRESHOLD = 0.5
BATCH_SIZE = 32


def parse_pdf(pdf_path: str, max_pages: int = 500) -> list[dict]:
    """Preserve PyMuPDF blocks and remove only repeated margin blocks."""
    try:
        with fitz.open(pdf_path) as doc:
            if not doc.is_pdf or doc.needs_pass or not 0 < doc.page_count <= max_pages:
                raise DocumentError()
            elements = []
            occurrences = defaultdict(set)
            for page_index, page in enumerate(doc):
                height = page.rect.height
                for block in page.get_text("blocks", sort=True):
                    if block[6] != 0:
                        continue
                    content = block[4].strip()
                    if not content:
                        continue
                    margin = block[3] <= height * HEADER_FOOTER_MARGIN or block[1] >= height * (1 - HEADER_FOOTER_MARGIN)
                    key = " ".join(content.split()).casefold()
                    elements.append(dict(content=content, page_num=page_index + 1, is_margin=margin, repeat_key=key))
                    if margin:
                        occurrences[key].add(page_index)
            suppressed = {
                key for key, pages in occurrences.items()
                if len(pages) >= 2 and len(pages) / doc.page_count > REPEATING_CONTENT_THRESHOLD
            }
            elements = [e for e in elements if not (e["is_margin"] and e["repeat_key"] in suppressed)]
            if not elements:
                raise DocumentError()
            return elements
    except DocumentError:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        raise DocumentError() from exc


def run_ingestion_pipeline(pdf_path: str, doc_hash: str, embedding_model, cache_dir: str,
                           source_name: str | None = None, max_pages: int = 500):
    elements = parse_pdf(pdf_path, max_pages=max_pages)
    token_limit = min(220, embedding_model.max_seq_length - embedding_model.tokenizer.num_special_tokens_to_add(pair=False))
    chunks = create_chunks(elements, source_name or os.path.basename(pdf_path),
                           embedding_model.tokenizer, max_tokens=token_limit,
                           overlap=min(32, max(0, token_limit // 4)))
    if not chunks:
        raise DocumentError()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = np.vstack([
        embedding_model.encode(texts[i:i + BATCH_SIZE], convert_to_numpy=True)
        for i in range(0, len(texts), BATCH_SIZE)
    ]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(cache_dir, f"{doc_hash}.json"), "w", encoding="utf-8") as output:
        json.dump(chunks, output, ensure_ascii=False)
    faiss.write_index(index, os.path.join(cache_dir, f"{doc_hash}.index"))
    return index, chunks
