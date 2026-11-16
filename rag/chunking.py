"""Token-bounded chunks with stable identities and accurate page metadata."""

import hashlib
import re

HEADING = re.compile(r"^(?:(?:[IVXLCDM]+|\d+|[A-Z])[.)]\s+)?([A-Z][A-Z\s&:/()\-]{2,100})$")
LIST_ITEM = re.compile(r"^\s*(?:[a-z][.)]|[ivx]+\.|[•\-])\s+", re.IGNORECASE)


def create_chunks(elements: list[dict], source_name: str, tokenizer,
                  max_tokens: int = 220, overlap: int = 32) -> list[dict]:
    if not 0 <= overlap < max_tokens:
        raise ValueError("Chunk overlap must be smaller than the token limit")
    groups = []
    section = "Preamble"
    for element in elements:
        text = element["content"].strip()
        if not text:
            continue
        heading = HEADING.fullmatch(text)
        if heading:
            section = heading.group(1).strip().rstrip(":")
        if (not heading and LIST_ITEM.match(text) and groups
                and groups[-1]["page_num"] == element["page_num"]
                and groups[-1]["section"] == section):
            groups[-1]["content"] += "\n" + text
        else:
            groups.append(dict(content=text, page_num=element["page_num"], section=section))

    chunks = []
    for group in groups:
        text = group["content"]
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        for start in range(0, len(offsets), max_tokens - overlap):
            end = min(start + max_tokens, len(offsets))
            content = text[offsets[start][0]:offsets[end - 1][1]]
            identity = f"{group['page_num']}:{len(chunks)}:{content}"
            chunks.append({
                "id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                "content": content,
                "metadata": {
                    "source_document": source_name, "page_number": group["page_num"],
                    "section_header": group["section"], "chunk_type": "content",
                },
            })
            if end == len(offsets):
                break
    return chunks
