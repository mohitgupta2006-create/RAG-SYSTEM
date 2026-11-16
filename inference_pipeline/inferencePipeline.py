"""Question answering over a validated, cached knowledge base."""

from rag.generation import generate_answer
from rag.retrieval import retrieve


def run_inference_pipeline(queries, knowledge_base, query_vectors, generative_model,
                           model_name, source_name, reranker, timeout=60):
    answers = []
    for query, vector in zip(queries, query_vectors, strict=True):
        chunks = retrieve(query, vector, knowledge_base, reranker)
        answers.append(generate_answer(query, chunks, generative_model, model_name,
                                       source_name, timeout=timeout))
    return answers
