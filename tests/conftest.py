import re

import numpy as np
import pytest


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class FakeEmbedding:
    tokenizer = FakeTokenizer()
    max_seq_length = 256

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        # Deterministic vectors for offline orchestration tests, not quality evaluation.
        return np.array([[1 + text.lower().count("waiting"),
                          1 + text.lower().count("maternity"), 1] for text in texts], dtype="float32")


@pytest.fixture
def embedding():
    return FakeEmbedding()
