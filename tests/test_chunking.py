from conftest import FakeTokenizer

from rag.chunking import create_chunks


def test_long_chunks_are_bounded_and_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = create_chunks([dict(content=text, page_num=1)], "policy.pdf", FakeTokenizer(), 20, 5)
    assert all(len(c["content"].split()) <= 20 for c in chunks)
    assert chunks[0]["content"].split()[-5:] == chunks[1]["content"].split()[:5]
    assert chunks[-1]["content"].endswith("word99")


def test_lists_never_merge_across_pages():
    elements = [
        dict(content="1. BENEFITS", page_num=1),
        dict(content="a) Waiting period is 30 days.", page_num=1),
        dict(content="b) Maternity has separate limits.", page_num=2),
    ]
    chunks = create_chunks(elements, "policy.pdf", FakeTokenizer())
    assert [c["metadata"]["page_number"] for c in chunks] == [1, 2]
    assert all(c["metadata"]["section_header"] == "BENEFITS" for c in chunks)
    assert chunks[0]["metadata"]["source_document"] == "policy.pdf"
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert create_chunks(elements, "policy.pdf", FakeTokenizer()) == chunks
